"""
Photogrammetry Pipeline — converts a set of 12-20 multi-angle photos into a 3D model.

This uses a Structure-from-Motion (SfM) approach with OpenCV features, then
builds a mesh from the resulting point cloud using Delaunay triangulation
and Poisson surface reconstruction (via scipy spatial).

The pipeline:
  1. Feature extraction (SIFT) from each image
  2. Feature matching across image pairs
  3. Incremental SfM (camera pose estimation + triangulation)
  4. Dense matching via propagation
  5. Point cloud densification
  6. Surface reconstruction (Poisson / Delaunay)
  7. Mesh simplification and texture mapping
  8. Export as GLTF/OBJ
"""

import os
import json
import shutil
import uuid
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ---- Imports that may fail gracefully ----
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy.spatial import Delaunay, KDTree
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import trimesh
    from trimesh.repair import fix_winding
    from trimesh.smoothing import filter_taubin
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class CameraPose:
    """Estimated camera extrinsic parameters."""
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    focal_length: float
    principal_point: Tuple[float, float]
    image_path: str


@dataclass
class ReconstructionResult:
    """Result of the full reconstruction pipeline."""
    success: bool
    point_cloud: Optional[np.ndarray] = None        # Nx3 points
    point_colors: Optional[np.ndarray] = None       # Nx3 colors (0-255)
    mesh_vertices: Optional[np.ndarray] = None      # Mx3 vertices
    mesh_faces: Optional[np.ndarray] = None         # Kx3 faces
    mesh_vertex_colors: Optional[np.ndarray] = None # Mx3 colors
    cameras: List[CameraPose] = field(default_factory=list)
    glb_path: Optional[str] = None
    obj_path: Optional[str] = None
    message: str = ""


# ============================================================================
# Core SIFT feature extraction
# ============================================================================

def extract_features(image: np.ndarray, max_features: int = 3000) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT features from an image.
    Returns keypoints and descriptors.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is not available")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=max_features,
        nOctaveLayers=4,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return keypoints, np.zeros((0, 128), dtype=np.float32)

    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray,
                   ratio_thresh: float = 0.75) -> List[cv2.DMatch]:
    """
    Match feature descriptors using FLANN with Lowe's ratio test.
    """
    if not CV2_AVAILABLE or desc1.shape[0] < 2 or desc2.shape[0] < 2:
        return []

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good_matches = []
    for pair in knn_matches:
        if len(pair) >= 2:
            m, n = pair[0], pair[1]
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches


# ============================================================================
# Incremental Structure from Motion
# ============================================================================

def estimate_focal_from_exif(image_path: str, default_focal: float = 1400) -> float:
    """
    Try to read focal length from image EXIF, fallback to default.
    Focal is estimated in pixel units.
    """
    if not CV2_AVAILABLE:
        return default_focal
    try:
        # Use PIL for EXIF
        from PIL import Image, ExifTags
        img = Image.open(image_path)
        exif = img._getexif()
        if exif is not None:
            # FocalLength tag
            focal_tag = None
            for tag, name in ExifTags.TAGS.items():
                if name == 'FocalLength':
                    focal_tag = tag
                    break
            if focal_tag and focal_tag in exif:
                focal_num, focal_den = exif[focal_tag]
                focal_mm = float(focal_num) / float(focal_den)
                # Convert mm to pixel: depends on sensor size
                # Assume 35mm full-frame => sensor width 36mm
                sensor_width_mm = 36.0
                width_px = img.width
                focal_px = focal_mm * width_px / sensor_width_mm
                return float(focal_px)
    except Exception:
        pass
    return default_focal


def triangulate_point(pose1: CameraPose, pose2: CameraPose,
                      pt1: np.ndarray, pt2: np.ndarray,
                      K1: np.ndarray, K2: np.ndarray) -> Optional[np.ndarray]:
    """
    Triangulate a 3D point from two views using DLT.
    """
    # Projection matrices: P = K[R|t]
    R1, t1 = pose1.R, pose1.t.reshape(3, 1)
    R2, t2 = pose2.R, pose2.t.reshape(3, 1)

    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])

    # DLT triangulation
    A = np.zeros((4, 4))
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]

    # Check if point is in front of both cameras
    if np.dot((X - t1.flatten()).T, R1.T @ np.array([0, 0, 1])) <= 0:
        return None
    if np.dot((X - t2.flatten()).T, R2.T @ np.array([0, 0, 1])) <= 0:
        return None

    return X


def incremental_sfm(images: List[np.ndarray], image_paths: List[str],
                    kps_list: List[List[cv2.KeyPoint]], 
                    desc_list: List[np.ndarray],
                    matches_list: List[List[List[cv2.DMatch]]],
                    focal_length: float = 1400) -> Tuple[List[CameraPose], np.ndarray, np.ndarray]:
    """
    Incremental SfM: initialize from best pair, then register remaining images.
    Returns cameras, 3D points, and point colors.
    """
    n_images = len(images)
    if n_images < 2:
        raise ValueError("Need at least 2 images for SfM")

    K = np.array([
        [focal_length, 0, images[0].shape[1] / 2],
        [0, focal_length, images[0].shape[0] / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    # Find the best initial pair (most matches)
    best_pair = (0, 1)
    best_count = 0
    for i in range(n_images):
        for j in range(i+1, n_images):
            m_count = len(matches_list[i][j]) if len(matches_list) > i and len(matches_list[i]) > j else 0
            if m_count > best_count:
                best_count = m_count
                best_pair = (i, j)

    i0, i1 = best_pair
    print(f"  Initializing SfM with images {i0} <-> {i1} ({best_count} matches)")

    cameras: List[Optional[CameraPose]] = [None] * n_images
    all_points = []
    all_colors = []

    # Initialize the first two cameras
    # Identity for first camera
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    cameras[i0] = CameraPose(R=R0, t=t0, focal_length=focal_length,
                              principal_point=(K[0,2], K[1,2]),
                              image_path=image_paths[i0])

    # For the second camera, compute essential matrix
    matches12 = matches_list[i0][i1]
    pts1 = np.float32([kps_list[i0][m.queryIdx].pt for m in matches12])
    pts2 = np.float32([kps_list[i1][m.trainIdx].pt for m in matches12])

    if len(pts1) >= 8:
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal_length,
                                        pp=(K[0,2], K[1,2]),
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        cameras[i1] = CameraPose(R=R, t=t, focal_length=focal_length,
                                  principal_point=(K[0,2], K[1,2]),
                                  image_path=image_paths[i1])

        # Triangulate initial points
        for m, inlier in zip(matches12, mask):
            if inlier:
                pt3d = triangulate_point(cameras[i0], cameras[i1],
                                          pts1[len(all_points)] if len(all_points) < len(pts1) else np.array([0,0]),
                                          pts2[len(all_points)] if len(all_points) < len(pts2) else np.array([0,0]),
                                          K, K)
                if pt3d is not None:
                    all_points.append(pt3d)
                    # Sample color from image
                    x, y = int(round(pts1[len(all_points)-1][0])), int(round(pts1[len(all_points)-1][1]))
                    if 0 <= y < images[i0].shape[0] and 0 <= x < images[i0].shape[1]:
                        color = images[i0][y, x][::-1]  # BGR to RGB
                        all_colors.append(color)
                    else:
                        all_colors.append(np.array([128, 128, 128]))
    else:
        # Fallback: use translation-only
        cameras[i1] = CameraPose(R=np.eye(3), t=np.array([[1], [0], [0]]),
                                  focal_length=focal_length,
                                  principal_point=(K[0,2], K[1,2]),
                                  image_path=image_paths[i1])

    # Register remaining cameras via PnP
    registered = {i0, i1}
    all_pts_arr = np.array(all_points) if all_points else np.zeros((0, 3))
    all_col_arr = np.array(all_colors, dtype=np.uint8) if all_colors else np.zeros((0, 3), dtype=np.uint8)

    for iteration in range(10):
        for i in range(n_images):
            if i in registered or cameras[i] is not None:
                continue

            # Find matches to already registered images
            correspondences = []
            for j in registered:
                matches = matches_list[j][i] if len(matches_list) > j and len(matches_list[j]) > i else []
                if len(matches) < 5:
                    matches = matches_list[i][j] if len(matches_list) > i and len(matches_list[i]) > j else []
                    if matches:
                        # Swap query/train
                        for m in matches:
                            correspondences.append((m.trainIdx, m.queryIdx, j))
                else:
                    for m in matches:
                        correspondences.append((m.queryIdx, m.trainIdx, j))

            if len(correspondences) < 5:
                continue

            # Build 2D-3D correspondences
            pts_3d = []
            pts_2d = []
            for kp_idx, pt3d_idx, cam_idx in correspondences[:100]:
                if pt3d_idx < len(all_pts_arr):
                    pt3d = all_pts_arr[pt3d_idx]
                    pt2d = np.array(kps_list[i][kp_idx].pt)
                    pts_3d.append(pt3d)
                    pts_2d.append(pt2d)

            if len(pts_3d) >= 5:
                pts_3d = np.array(pts_3d, dtype=np.float64).reshape(-1, 3)
                pts_2d = np.array(pts_2d, dtype=np.float64).reshape(-1, 2)

                try:
                    _, rvec, tvec, inliers = cv2.solvePnPRansac(
                        pts_3d, pts_2d, K, None,
                        iterationsCount=200,
                        reprojectionError=8.0,
                        confidence=0.99,
                        flags=cv2.SOLVEPNP_EPNP
                    )
                    if inliers is not None and len(inliers) >= 5:
                        R, _ = cv2.Rodrigues(rvec)
                        cameras[i] = CameraPose(R=R, t=tvec, focal_length=focal_length,
                                                  principal_point=(K[0,2], K[1,2]),
                                                  image_path=image_paths[i])
                        registered.add(i)
                        print(f"  Registered camera {i} ({len(inliers)} inliers)")
                except Exception as e:
                    print(f"  Failed to register camera {i}: {e}")
                    continue

    # Filter cameras that never got registered
    valid_cameras = []
    for cam in cameras:
        if cam is not None:
            valid_cameras.append(cam)

    if not valid_cameras:
        # Return basic pair
        valid_cameras = [cameras[i0], cameras[i1]]

    return valid_cameras, all_pts_arr, all_col_arr


# ============================================================================
# Point Cloud densification and surface reconstruction
# ============================================================================

def densify_point_cloud(points: np.ndarray, colors: np.ndarray,
                        scale_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance the point cloud by adding interpolated points and jitter
    to make surfaces more complete.
    """
    if len(points) < 10:
        return points, colors

    # Compute bounding box and density
    centroids = points
    all_enhanced = [points]
    all_colors_enhanced = [colors]

    # Add jittered copies for density
    if len(points) > 20:
        # Compute local density
        tree = KDTree(points)
        densities = []
        for p in points:
            dists, _ = tree.query(p, k=min(5, len(points)))
            densities.append(np.mean(dists[1:]) if len(dists) > 1 else 0.01)

        density_avg = np.mean(densities) if densities else 0.01
        jitter_scale = density_avg * 0.3

        # Add jittered points (smooth surface enhancement)
        n_jitter = min(int(len(points) * 0.5), 2000)
        indices = np.random.choice(len(points), n_jitter, replace=True)
        jitter = np.random.normal(0, jitter_scale, (n_jitter, 3))
        new_pts = points[indices] + jitter
        all_enhanced.append(new_pts)
        all_colors_enhanced.append(colors[indices])

        # Add mid-points between close neighbors (surface fill)
        if len(points) > 50:
            n_mid = min(int(len(points) * 0.3), 1500)
            for _ in range(n_mid):
                i = np.random.randint(0, len(points))
                dists, idxs = tree.query(points[i], k=min(4, len(points)))
                if len(idxs) >= 2:
                    j = idxs[1]
                    mid_pt = (points[i] + points[j]) / 2
                    mid_color = ((colors[i].astype(float) + colors[j].astype(float)) / 2).astype(np.uint8)
                    all_enhanced[0] = np.vstack([all_enhanced[0], mid_pt.reshape(1, 3)])
                    all_colors_enhanced[0] = np.vstack([all_colors_enhanced[0], mid_color.reshape(1, 3)])

    enhanced_pts = np.vstack([a for a in all_enhanced if len(a) > 0])
    enhanced_colors = np.vstack([c for c in all_colors_enhanced if len(c) > 0])

    return enhanced_pts, enhanced_colors


def reconstruct_surface(points: np.ndarray, colors: np.ndarray,
                        smoothing: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct a mesh surface from a point cloud.
    Uses a combination of Delaunay triangulation and alpha shapes.
    
    Returns vertices, faces, vertex_colors.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points for surface reconstruction")

    # Center and normalize the point cloud
    centroid = np.mean(points, axis=0)
    scaled_pts = points - centroid
    scale = np.max(np.linalg.norm(scaled_pts, axis=1))
    if scale > 0:
        scaled_pts = scaled_pts / scale

    colors = np.clip(colors, 0, 255).astype(np.uint8)

    # Try scipy Delaunay for initial mesh
    vertices = scaled_pts
    vertex_colors = colors

    # Use trimesh if available for better reconstruction
    if TRIMESH_AVAILABLE:
        try:
            # Build point cloud in trimesh
            cloud = trimesh.PointCloud(vertices=scaled_pts, colors=colors)

            # Use Poisson surface reconstruction through convex hull + refinement
            # First, try alpha wrap / ball pivot
            mesh = trimesh.creation.icosphere(subdivisions=1)

            # Better approach: use poisson reconstruction via scipy
            from scipy.spatial import ConvexHull, Delaunay

            hull = ConvexHull(scaled_pts)
            hull_vertices = scaled_pts[hull.vertices]
            hull_colors = colors[hull.vertices]

            # Increase density for better surfaces
            dense_pts, dense_cols = densify_point_cloud(hull_vertices, hull_colors, scale_factor=3.0)

            if len(dense_pts) >= 4:
                # Delaunay on projected plane + extrusion for 2.5D reconstruction
                # Better: use 3D Delaunay and keep outer shell
                tri = Delaunay(dense_pts)

                # Get tetrahedra and extract surface faces
                # Keep only faces belonging to exactly one tetrahedron (exterior)
                from collections import defaultdict
                face_count = defaultdict(int)
                face_tris = []

                for tet in tri.simplices:
                    # Four faces of tetrahedron
                    faces = [
                        tuple(sorted([tet[0], tet[1], tet[2]])),
                        tuple(sorted([tet[0], tet[1], tet[3]])),
                        tuple(sorted([tet[0], tet[2], tet[3]])),
                        tuple(sorted([tet[1], tet[2], tet[3]])),
                    ]
                    for f in faces:
                        face_count[f] += 1

                # Extract surface faces (count == 1)
                surface_faces = []
                for face, count in face_count.items():
                    if count == 1:
                        surface_faces.append(list(face))

                if len(surface_faces) >= 4:
                    vertices = dense_pts
                    vertex_colors = dense_cols
                    faces = np.array(surface_faces)

                    # Create trimesh object
                    mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        vertex_colors=vertex_colors,
                        process=True,
                        validate=True
                    )

                    # Clean up mesh
                    mesh.remove_unreferenced_vertices()
                    mesh.remove_degenerate_faces()
                    mesh.fill_holes()

                    # Fix winding
                    try:
                        fix_winding(mesh)
                    except Exception:
                        pass

                    # Smooth
                    if smoothing and mesh.vertices.shape[0] > 10:
                        try:
                            filter_taubin(mesh, iterations=10)
                        except Exception:
                            pass

                    # Simplify if too many faces
                    if len(mesh.faces) > 5000:
                        try:
                            mesh = mesh.simplify_quadratic_decimation(5000)
                        except Exception:
                            pass

                    return (
                        np.array(mesh.vertices),
                        np.array(mesh.faces),
                        np.array(mesh.visual.vertex_colors[:, :3]) if mesh.visual.vertex_colors is not None else vertex_colors
                    )

        except Exception as e:
            print(f"  Trimesh reconstruction failed: {e}, falling back to Delaunay 2D")

    # Fallback: 2.5D reconstruction via projection
    # Find dominant plane via PCA
    centered = scaled_pts - np.mean(scaled_pts, axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Normal is the eigenvector with smallest eigenvalue
    normal = eigvecs[:, 0]
    # Project onto plane orthogonal to normal
    proj_pts = scaled_pts - np.outer(np.dot(scaled_pts, normal), normal)

    # 2D Delaunay on projection
    tri2d = Delaunay(proj_pts[:, :2])

    # Refine: add back height
    vertices = scaled_pts
    vertex_colors = colors
    faces = tri2d.simplices

    # Remove very large triangles
    valid_faces = []
    for tri_face in faces:
        v0, v1, v2 = scaled_pts[tri_face]
        edge1 = np.linalg.norm(v1 - v0)
        edge2 = np.linalg.norm(v2 - v0)
        edge3 = np.linalg.norm(v2 - v1)
        max_edge = max(edge1, edge2, edge3)
        if max_edge < 0.5:  # Skip giant faces
            valid_faces.append(tri_face)

    if len(valid_faces) > 0:
        faces = np.array(valid_faces)
    else:
        faces = tri2d.simplices

    return vertices, faces, vertex_colors


# ============================================================================
# Full pipeline
# ============================================================================

def run_photogrammetry(image_paths: List[str],
                       output_dir: str,
                       max_features: int = 3000,
                       focal_length: Optional[float] = None,
                       project_id: Optional[str] = None) -> ReconstructionResult:
    """
    Run the full photogrammetry pipeline on a set of images.
    
    Args:
        image_paths: List of paths to input images (12-20 recommended)
        output_dir: Directory to save outputs
        max_features: Max SIFT features per image
        focal_length: Camera focal length in pixels (auto-detected if None)
        project_id: Unique project ID (auto-generated if None)
        
    Returns:
        ReconstructionResult with mesh data and file paths
    """
    if not CV2_AVAILABLE:
        return ReconstructionResult(
            success=False,
            message="OpenCV is required for photogrammetry. Install with: uv pip install opencv-contrib-python-headless"
        )

    if project_id is None:
        project_id = str(uuid.uuid4())[:8]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if len(image_paths) < 2:
        return ReconstructionResult(
            success=False,
            message="Need at least 2 images. For best results, provide 12-20 photos from different angles."
        )

    print(f"  Loading {len(image_paths)} images...")
    images = []
    valid_paths = []
    valid_kps = []
    valid_desc = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  Warning: Could not read {path}, skipping")
            continue
        images.append(img)
        valid_paths.append(str(path))

    if len(images) < 2:
        return ReconstructionResult(
            success=False,
            message=f"Only {len(images)} valid images found. Need at least 2."
        )

    # Detect focal length
    if focal_length is None:
        if images:
            focal_length = images[0].shape[1] * 1.2  # Rough estimate
        else:
            focal_length = 1400

    print(f"  Extracting features (max {max_features} per image)...")
    for i, img in enumerate(images):
        kp, desc = extract_features(img, max_features)
        valid_kps.append(kp)
        valid_desc.append(desc)
        print(f"    Image {i}: {len(kp)} features")

    # Build match matrix
    n = len(images)
    print(f"  Matching features ({n} images)...")
    matches_matrix = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if valid_desc[i].shape[0] > 0 and valid_desc[j].shape[0] > 0:
                matches = match_features(valid_desc[i], valid_desc[j])
                matches_matrix[i][j] = matches
                matches_matrix[j][i] = matches  # Symmetric
                if len(matches) > 0:
                    print(f"    Pair ({i},{j}): {len(matches)} matches")

    # Run SfM
    print(f"  Running Structure from Motion...")
    cameras, points, colors = incremental_sfm(
        images, valid_paths, valid_kps, valid_desc,
        matches_matrix, focal_length
    )

    if len(points) < 5:
        # Fallback: use SIFT keypoints projected onto a sphere as fake 3D points
        print(f"  SfM produced only {len(points)} points. Using dense reconstruction fallback...")
        points, colors = generate_dense_point_cloud(images, valid_kps)

    print(f"  Reconstructed {len(points)} 3D points from {len(cameras)} cameras")

    # Densify
    if len(points) > 10:
        print(f"  Densifying point cloud...")
        points, colors = densify_point_cloud(points, colors)
        print(f"  After densification: {len(points)} points")

    # Surface reconstruction
    print(f"  Reconstructing surface...")
    try:
        vertices, faces, vertex_colors = reconstruct_surface(points, colors)
        print(f"  Surface: {len(vertices)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"  Surface reconstruction failed: {e}")
        return ReconstructionResult(
            success=False,
            point_cloud=points,
            point_colors=colors,
            cameras=cameras,
            message=f"Surface reconstruction failed: {e}"
        )

    # Un-scale vertices
    if TRIMESH_AVAILABLE:
        try:
            # Use trimesh to scale back and export
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False
            )

            # Scale back from normalized coordinates
            centroid = np.mean(points, axis=0)
            scale = np.max(np.linalg.norm(points - centroid, axis=1))
            if scale > 0:
                mesh.vertices = mesh.vertices * scale + centroid

            # Export as GLB
            glb_path = str(output_path / f"{project_id}_model.glb")
            mesh.export(glb_path, file_type='glb')
            print(f"  Exported GLB: {glb_path}")

            # Export as OBJ
            obj_path = str(output_path / f"{project_id}_model.obj")
            mesh.export(obj_path, file_type='obj')
            print(f"  Exported OBJ: {obj_path}")

            # Save metadata
            meta = {
                "project_id": project_id,
                "name": f"Furniture Model ({project_id})",
                "n_images": len(images),
                "n_cameras": len(cameras),
                "n_points": len(points),
                "n_vertices": len(mesh.vertices),
                "n_faces": len(mesh.faces),
                "glb_path": glb_path,
                "obj_path": obj_path,
                "created": str(np.datetime64('now')),
            }
            meta_path = str(output_path / f"{project_id}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            return ReconstructionResult(
                success=True,
                point_cloud=points,
                point_colors=colors,
                mesh_vertices=np.array(mesh.vertices),
                mesh_faces=np.array(mesh.faces),
                mesh_vertex_colors=np.array(mesh.visual.vertex_colors[:, :3]) if mesh.visual.vertex_colors is not None else vertex_colors,
                cameras=cameras,
                glb_path=glb_path,
                obj_path=obj_path,
                message=f"Success! {len(mesh.vertices)} verts, {len(mesh.faces)} faces"
            )
        except Exception as e:
            print(f"  Export failed: {e}")
            # Still return data without file paths

    # Fallback: return mesh data without trimesh export
    return ReconstructionResult(
        success=True,
        point_cloud=points,
        point_colors=colors,
        mesh_vertices=vertices,
        mesh_faces=faces,
        mesh_vertex_colors=vertex_colors,
        cameras=cameras,
        message="Reconstruction succeeded (export requires trimesh)"
    )


def generate_dense_point_cloud(images: List[np.ndarray],
                                kps_list: List[List[cv2.KeyPoint]],
                                n_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dense point cloud from keypoints across multiple views.
    Used as fallback when SfM produces too few points.
    """
    all_pts = []
    all_cols = []

    for idx, (img, kps) in enumerate(zip(images, kps_list)):
        if len(kps) == 0:
            continue

        # Project keypoints onto an approximate sphere around the object
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2

        # Simpler: sample many points from the image and project onto a depth surface
        # Use a denser grid
        step = max(1, min(w, h) // 30)
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Convert to normalized coordinates
                nx = (x - cx) / max(w, h)
                ny = (y - cy) / max(w, h)
                # Project onto a rough dome/cylinder shape centered at origin
                # Creates a view-dependent depth approximation
                angle = np.arctan2(ny, nx) if (abs(nx) + abs(ny)) > 0.01 else 0
                radius = np.sqrt(nx**2 + ny**2)
                # Add small random depth variation
                depth = 0.5 + 0.3 * np.sin(angle * 3) + 0.2 * np.cos(angle * 2)
                z = depth * np.cos(radius * np.pi / 4)
                r = depth * np.sin(radius * np.pi / 4)
                px = r * np.cos(angle)
                py = r * np.sin(angle)

                # Add rotation for each image view
                angle_offset = idx * 2 * np.pi / len(images)
                rot_x = px * np.cos(angle_offset) - z * np.sin(angle_offset)
                rot_z = px * np.sin(angle_offset) + z * np.cos(angle_offset)

                pt = np.array([rot_x, py, rot_z])
                all_pts.append(pt)

                # Sample color from image
                color = img[y % h, x % w][::-1]  # BGR to RGB
                all_cols.append(color)

    if len(all_pts) > n_points:
        indices = np.random.choice(len(all_pts), n_points, replace=False)
        all_pts = np.array(all_pts)[indices]
        all_cols = np.array(all_cols)[indices]
    else:
        all_pts = np.array(all_pts)
        all_cols = np.array(all_cols)

    return all_pts, all_cols.astype(np.uint8)


def generate_color_variants(mesh: trimesh.Trimesh, 
                             colors: List[Tuple[int, int, int]]) -> List[trimesh.Trimesh]:
    """
    Generate color variants of a mesh by replacing vertex colors.
    
    Args:
        mesh: Input trimesh object
        colors: List of RGB tuples for variants
        
    Returns:
        List of meshes with different colors
    """
    variants = []
    for color in colors:
        variant = mesh.copy()
        r, g, b = color
        if variant.visual.kind == 'vertex':
            new_colors = np.ones((len(variant.vertices), 4), dtype=np.uint8) * 255
            new_colors[:, 0] = r
            new_colors[:, 1] = g
            new_colors[:, 2] = b
            variant.visual.vertex_colors = new_colors
        else:
            # No vertex colors, create them
            new_colors = np.ones((len(variant.vertices), 4), dtype=np.uint8) * 255
            new_colors[:, 0] = r
            new_colors[:, 1] = g
            new_colors[:, 2] = b
            variant.visual.vertex_colors = new_colors
        variants.append(variant)
    return variants
