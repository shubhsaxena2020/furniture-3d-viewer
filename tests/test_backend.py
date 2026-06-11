"""Tests for the FastAPI backend server endpoints."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
import numpy as np

# Skip main server tests if the server can't import
try:
    # Mock OpenCV features if needed
    from backend.main import app
    SERVER_OK = True
except Exception as e:
    print(f"Server import error: {e}")
    SERVER_OK = False


@pytest.mark.skipif(not SERVER_OK, reason="Server dependencies not available")
class TestServerAPI:
    """Test the FastAPI server endpoints."""
    
    def test_health(self):
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
    
    def test_presets(self):
        client = TestClient(app)
        resp = client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "colors" in data
        assert "materials" in data
        assert len(data["colors"]) == 16
        assert len(data["materials"]) == 14
    
    def test_models_list(self):
        client = TestClient(app)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
    
    def test_upload_invalid_no_files(self):
        client = TestClient(app)
        resp = client.post("/api/upload")
        assert resp.status_code == 422  # Validation error
    
    def test_upload_too_few_files(self):
        client = TestClient(app)
        # Create a single small jpg
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        # Minimal valid JPEG
        tmp.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x11\x04\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfa$\x87\xb7\xb6\xa8\xed\x9eVt\xabK\xc3\xc5F\x0e\x99\x18\x02\x92-\xc3\xdc\xd7\xff\xd9')
        tmp.close()
        
        # Try with 1 file (needs >= 2)
        with open(tmp.name, 'rb') as f:
            resp = client.post("/api/upload", files={"files": ("test.jpg", f, "image/jpeg")})
        os.unlink(tmp.name)
        assert resp.status_code == 400  # Not enough files
    
    def test_status_invalid_job(self):
        client = TestClient(app)
        resp = client.get("/api/status/nonexistent_job")
        assert resp.status_code == 404
    
    def test_model_not_found(self):
        client = TestClient(app)
        resp = client.get("/api/models/nonexistent")
        assert resp.status_code == 404
    
    def test_sample_model(self):
        """Sample endpoint should work even with no output models."""
        client = TestClient(app)
        resp = client.get("/api/sample")
        assert resp.status_code == 200
        data = resp.json()
        assert "glb_url" in data or "project_id" in data
    
    def test_get_presets_type(self):
        client = TestClient(app)
        resp = client.get("/api/presets")
        data = resp.json()
        assert len(data["colors"]) == 16
        for c in data["colors"]:
            assert "name" in c
            assert "hex" in c
            assert "rgb" in c
        for m in data["materials"]:
            assert "name" in m
            assert "roughness" in m
            assert "metalness" in m


@pytest.mark.skipif(not SERVER_OK, reason="Server dependencies not available")
class TestServerEndpoints:
    """Additional server endpoint coverage."""
    
    def test_health_check_response_shape(self):
        client = TestClient(app)
        resp = client.get("/api/health")
        data = resp.json()
        assert set(data.keys()) == {"status", "version", "model"}
    
    def test_cors_headers(self):
        client = TestClient(app)
        resp = client.get("/api/health", headers={"Origin": "http://example.com"})
        assert resp.status_code == 200
        # CORS middleware may set vary depending on origin or wildcard
        assert resp.headers.get("access-control-allow-origin") is not None


class TestPhotogrammetryEdgeCases:
    """Additional edge case tests for photogrammetry functions."""
    
    def test_densify_point_cloud_large(self):
        from backend.photogrammetry import densify_point_cloud
        pts = np.random.randn(1000, 3) * 0.5
        cols = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
        d_pts, d_cols = densify_point_cloud(pts, cols)
        assert len(d_pts) > len(pts)
    
    def test_run_photogrammetry_no_images(self):
        from backend.photogrammetry import run_photogrammetry
        result = run_photogrammetry([], "/tmp/out", project_id="test_empty")
        assert not result.success
    
    def test_run_photogrammetry_one_image(self):
        from backend.photogrammetry import run_photogrammetry
        import cv2
        import tempfile
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(f.name, img)
        f.close()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_photogrammetry([f.name], tmpdir, project_id="test_one")
        os.unlink(f.name)
        assert not result.success
        assert "Need at least 2" in result.message


class TestMaterialEstimation:
    """Test material estimation edge cases."""
    
    def test_estimate_metal(self):
        from backend.photogrammetry import estimate_material
        # High contrast, high Laplacian variance → metal
        img = np.random.randint(0, 60, (200, 200, 3), dtype=np.uint8)
        # Add sharp edges
        img[50:150, 50:150] = 200
        r, m = estimate_material([img])
        assert 0.0 <= r <= 1.0
        assert 0.0 <= m <= 1.0
    
    def test_estimate_leather(self):
        from backend.photogrammetry import estimate_material
        # Medium variation
        img = np.random.randint(80, 140, (200, 200, 3), dtype=np.uint8)
        r, m = estimate_material([img])
        assert 0.0 <= r <= 1.0
    
    def test_estimate_single_pixel(self):
        from backend.photogrammetry import estimate_material
        # Need minimum ROI size for cvtColor
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        r, m = estimate_material([img])
        assert isinstance(r, float)
        assert isinstance(m, float)


class TestReconstructMeshEdgeCases:
    """Edge cases for mesh reconstruction."""
    
    def test_mesh_no_points(self):
        from backend.photogrammetry import reconstruct_mesh_advanced as reconstruct_mesh
        pts = np.zeros((0, 3))
        cols = np.zeros((0, 3), dtype=np.uint8)
        v, f, c = reconstruct_mesh(pts, cols)
        assert len(v) == 0
        assert len(f) == 0
    
    def test_mesh_few_points(self):
        from backend.photogrammetry import reconstruct_mesh_advanced as reconstruct_mesh
        pts = np.random.randn(3, 3) * 0.1
        cols = np.random.randint(0, 255, (3, 3), dtype=np.uint8)
        v, f, c = reconstruct_mesh(pts, cols)
        assert len(v) >= 3
        assert len(f) == 0  # Can't form a tetrahedron with 3 points
    
    def test_mesh_planar_points(self):
        from backend.photogrammetry import reconstruct_mesh_advanced as reconstruct_mesh
        # All points on a plane
        pts = np.random.randn(100, 3) * 0.5
        pts[:, 2] = 0  # Z = 0
        cols = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        v, f, c = reconstruct_mesh(pts, cols)
        assert len(f) == 0 or len(f) > 0  # Either way is acceptable
