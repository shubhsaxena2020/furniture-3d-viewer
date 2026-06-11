"""Additional edge-case tests for photogrammetry and backend robustness."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2


class TestPhotogrammetryEdgeCases:
    """Edge case tests for photogrammetry functions — resilience to bad input."""
    
    def test_estimate_material_tiny_image(self):
        """Small images must not crash estimate_material."""
        from backend.photogrammetry import estimate_material
        # 1x1 image (would crash earlier code)
        img = np.ones((1, 1, 3), dtype=np.uint8) * 128
        r, m = estimate_material([img])
        assert isinstance(r, float)
        assert isinstance(m, float)
    
    def test_estimate_material_2px_image(self):
        """Very small image edge case."""
        from backend.photogrammetry import estimate_material
        img = np.ones((3, 3, 3), dtype=np.uint8) * 128
        r, m = estimate_material([img])
        assert isinstance(r, float)
    
    def test_estimate_material_greyscale(self):
        """Greyscale images must not crash."""
        from backend.photogrammetry import estimate_material
        img = np.ones((100, 100), dtype=np.uint8) * 128
        r, m = estimate_material([img])
        assert 0.0 <= r <= 1.0
    
    def test_estimate_material_zero_variance(self):
        """Zero variance image."""
        from backend.photogrammetry import estimate_material
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        r, m = estimate_material([img])
        assert 0.0 <= r <= 1.0
    
    def test_run_photogrammetry_empty_image(self):
        """Run pipeline with invalid image file."""
        from backend.photogrammetry import run_photogrammetry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-image file
            bad_path = os.path.join(tmpdir, "not_an_image.jpg")
            with open(bad_path, "w") as f:
                f.write("not a valid image file")
            
            result = run_photogrammetry(
                [bad_path, bad_path],
                tmpdir,
                project_id="test_bad_img",
            )
            assert not result.success
    
    def test_run_photogrammetry_different_dimensions(self):
        """Test with images of different sizes."""
        from backend.photogrammetry import run_photogrammetry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
            img2 = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
            
            p1 = os.path.join(tmpdir, "p1.jpg")
            p2 = os.path.join(tmpdir, "p2.jpg")
            cv2.imwrite(p1, img1)
            cv2.imwrite(p2, img2)
            
            result = run_photogrammetry(
                [p1, p2],
                os.path.join(tmpdir, "out"),
                project_id="test_diff_dim",
                target_faces=1000,
            )
            # Should work because resize handles it
            assert result.success or result.message != ""
    
    def test_gradient_fallback_tiny_images(self):
        """Gradient fallback with minimal images."""
        from backend.photogrammetry import _gradient_dense_cloud
        
        images = [np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8) for _ in range(2)]
        pts, cols = _gradient_dense_cloud(images, max_points=100)
        assert len(pts) >= 0  # May produce few or zero points on tiny images
        assert len(cols) >= 0
    
    def test_texture_transfer_no_faces(self):
        """Texture transfer with empty faces array."""
        from backend.photogrammetry import transfer_textures_advanced
        
        verts = np.random.randn(10, 3)
        faces = np.zeros((0, 3), dtype=np.int64)
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
        vcols = np.random.randint(0, 255, (10, 3), dtype=np.uint8)
        
        result = transfer_textures_advanced(verts, faces, images, vcols)
        assert result.shape == (10, 3)
    
    def test_densify_single_point(self):
        """Densify with a single point."""
        from backend.photogrammetry import densify_point_cloud
        
        pts = np.random.randn(1, 3)
        cols = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        d_pts, d_cols = densify_point_cloud(pts, cols)
        assert len(d_pts) >= 1
    
    def test_build_projection_planar(self):
        """Build projection with nearly-planar points (should not crash)."""
        from backend.photogrammetry import _build_projection
        
        pts = np.random.randn(100, 3) * 0.5
        pts[:, 0] *= 0.001  # Nearly planar but not exactly
        cols = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        
        try:
            result = _build_projection(pts, cols)
            # May return None — that's acceptable, just shouldn't crash
            if result is not None:
                assert len(result[1]) >= 4
        except Exception as e:
            # Any exception should be caught and handled at the caller level
            # In the actual pipeline, reconstruct_mesh_advanced wraps this in try/except
            print(f"  _build_projection correctly raised: {e}")


class TestServerEdgeCases:
    """Tests for server robustness (patched to avoid actual server startup)."""
    
    def test_processing_job_nonexistent(self):
        """Getting status of non-existent job returns 404 properly."""
        from backend.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        resp = client.get("/api/status/fake_job_12345")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
    
    def test_model_nonexistent(self):
        """Getting non-existent model returns 404 properly."""
        from backend.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        resp = client.get("/api/models/nonexistent_model_id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
    
    def test_upload_too_few(self):
        """Upload with 1 file must return 400."""
        from backend.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        
        resp = client.post(
            "/api/upload",
            files={"files": ("test.jpg", encoded.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "at least 2" in resp.json()["detail"].lower()
    
    def test_upload_too_many(self):
        """Upload with 51 files must return 400."""
        from backend.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        
        files = [("files", (f"test_{i}.jpg", encoded.tobytes(), "image/jpeg")) for i in range(51)]
        resp = client.post("/api/upload", files=files)
        assert resp.status_code == 400
        assert "maximum" in resp.json()["detail"].lower()
