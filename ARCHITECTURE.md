# Architecture

## System Overview

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│   Browser   │────▶│  FastAPI Server       │────▶│  COLMAP     │
│ (Three.js)  │◀────│  (port 8777)          │◀────│  (SfM)      │
└─────────────┘     │                      │     └─────────────┘
                    │  ┌────────────────┐   │
                    │  │ Photogrammetry  │   │
                    │  │ Pipeline        │   │
                    │  │                │   │
                    │  │ 1. SfM (COLMAP)│   │
                    │  │ 2. MVS (SIFT)  │   │
                    │  │ 3. Mesh Recon  │   │
                    │  │ 4. Texture Xfer│   │
                    │  │ 5. GLB Export  │   │
                    │  └────────────────┘   │
                    └──────────────────────┘
```

## Pipeline Stages

### 1. COLMAP Structure-from-Motion
- Extracts SIFT features (up to 16K per image)
- Exhaustive matching across all image pairs
- Incremental SfM with relaxed constraints for furniture
- Output: Sparse point cloud with RGB colors

### 2. Multi-View Stereo (MVS) — Dense Points
- SIFT feature extraction on all images using OpenCV
- Feature matching with Lowe's ratio test (ratio=0.7)
- Direct Linear Transform triangulation from camera poses
- Outlier rejection via MAD threshold (3x median absolute deviation)
- Fallback: gradient-based dense hemisphere sampling
- Output: Dense 3D point cloud (up to 120K points)

### 3. Surface Reconstruction
Three strategies tried in order:
1. **3D Delaunay Tetrahedralization** → surface extraction (most common)
2. **Convex Hull** (for point clouds on object surface)
3. **2.5D Projection** → 2D Delaunay (for flat-ish point sets)

After initial mesh: **Loop subdivision** (up to 6 iterations, 4x per iteration)
to reach target face count (100K+), then **Taubin smoothing** (3 iterations)
for noise reduction.

### 4. Multi-View Texture Transfer
- Per-vertex normal computation from face adjacency
- Project each vertex into every camera view
- Weighted blending: facing² / depth² (prefers frontal, close views)
- Fallback: KD-tree propagation for uncovered vertices
- Output: >95% vertex coverage with textured colors

### 5. PBR Material Estimation
- Laplacian variance + texture standard deviation analysis
- Classifies as: Metal, Leather, Wood, Fabric, or Default
- Sets roughness and metalness parameters for GLB export

### 6. GLB/OBJ Export
- Uses trimesh for binary GLB with vertex normals
- Embeds PBR material parameters
- Metadata JSON with all processing metrics

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CPU-only pipeline | No GPU dependency for server compatibility |
| Per-vertex colors (not UV maps) | Simpler, avoids UV unwrapping artifacts |
| MVS fallback when COLMAP fails | Handles real-world photos without calibrated cameras |
| Uniform image resize (720x540) | Consistent camera model for COLMAP |
| Subdivision then simplification | Predictable face counts regardless of input quality |
| Background thread processing | Non-blocking API for long-running jobs |

## Data Flow

```
Upload (4-20 photos)
    │
    ▼
Image Loading + Resize (720x540)
    │
    ├─▶ COLMAP SfM ──▶ Sparse Cloud
    │
    ▼
Multi-View Stereo ──▶ Dense Cloud (120K pts)
    │
    ▼
Merge COLMAP + MVS points
    │
    ▼
Point Cloud Densification
    │
    ▼
Surface Reconstruction ──▶ Mesh (~100K faces)
    │
    ▼
Texture Transfer ──▶ Colored Mesh
    │
    ▼
GLB/OBJ Export ──▶ Static Files
    │
    ▼
API Response (Job result)
```
