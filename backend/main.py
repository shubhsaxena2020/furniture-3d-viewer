"""
Furniture 3D Viewer - FastAPI Backend Server

This server provides:
  - Upload endpoint for 12-20 photos
  - Photogrammetry processing pipeline
  - Model retrieval for the 3D web viewer
  - Color/material variant generation API
  - Sample furniture model serving (for testing)

Run: python -m backend.main
Or:  uvicorn backend.main:app --reload
"""

import os
import sys
import json
import uuid
import shutil
import zipfile
import asyncio
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import time
import datetime

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ----
# Project paths
# ----
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = Path(__file__).parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "static" / "output"
STATIC_DIR = PROJECT_ROOT / "static"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Sample models directory
SAMPLE_MODELS_DIR = BACKEND_DIR / "sample_models"
SAMPLE_MODELS_DIR.mkdir(exist_ok=True)

# ----
# Import photogrammetry
# ----
from backend.photogrammetry import run_photogrammetry

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Startup: generate sample model, mount frontend."""
    # Mount frontend AFTER API routes so they take priority
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, generate_sample_model, str(OUTPUT_DIR), "sample_sofa")
    yield


# Track processing jobs
processing_jobs: dict = {}

# Thread executor for CPU-bound work
executor = ThreadPoolExecutor(max_workers=2)

# ----
# FastAPI app
# ----
app = FastAPI(
    title="Furniture 3D Viewer API",
    description="Upload furniture photos and generate interactive 3D models with color/material customization",
    version="1.0.0",
    lifespan=app_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Models
# ============================================================================

class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None


class ColorVariant(BaseModel):
    name: str
    hex: str
    rgb: List[int]


class MaterialPreset(BaseModel):
    name: str
    roughness: float
    metalness: float
    clearcoat: float = 0.0
    description: str = ""


# ============================================================================
# Material presets for furniture
# ============================================================================

COLOR_PRESETS = [
    ColorVariant(name="Classic Black", hex="#1a1a1a", rgb=[26, 26, 26]),
    ColorVariant(name="Pure White", hex="#f5f5f5", rgb=[245, 245, 245]),
    ColorVariant(name="Warm Grey", hex="#b0b0b0", rgb=[176, 176, 176]),
    ColorVariant(name="Navy Blue", hex="#1b2838", rgb=[27, 40, 56]),
    ColorVariant(name="Forest Green", hex="#2d5a27", rgb=[45, 90, 39]),
    ColorVariant(name="Burgundy", hex="#6b2020", rgb=[107, 32, 32]),
    ColorVariant(name="Sage Green", hex="#7b9a7b", rgb=[123, 154, 123]),
    ColorVariant(name="Dusty Rose", hex="#c08080", rgb=[192, 128, 128]),
    ColorVariant(name="Slate Blue", hex="#5b6e8a", rgb=[91, 110, 138]),
    ColorVariant(name="Warm Beige", hex="#d4b896", rgb=[212, 184, 150]),
    ColorVariant(name="Terracotta", hex="#cc6644", rgb=[204, 102, 68]),
    ColorVariant(name="Charcoal", hex="#36454f", rgb=[54, 69, 79]),
    ColorVariant(name="Cream", hex="#fffdd0", rgb=[255, 253, 208]),
    ColorVariant(name="Teal", hex="#008080", rgb=[0, 128, 128]),
    ColorVariant(name="Mustard Yellow", hex="#d4a017", rgb=[212, 160, 23]),
    ColorVariant(name="Coral", hex="#ff7f50", rgb=[255, 127, 80]),
]

MATERIAL_PRESETS = [
    MaterialPreset(name="Fabric (Linen)", roughness=0.85, metalness=0.0, clearcoat=0.0, description="Natural linen texture"),
    MaterialPreset(name="Fabric (Velvet)", roughness=0.5, metalness=0.0, clearcoat=0.2, description="Soft velvet with subtle sheen"),
    MaterialPreset(name="Leather (Matte)", roughness=0.7, metalness=0.0, clearcoat=0.1, description="Matte leather finish"),
    MaterialPreset(name="Leather (Gloss)", roughness=0.3, metalness=0.0, clearcoat=0.4, description="Glossy polished leather"),
    MaterialPreset(name="Wood (Oak)", roughness=0.6, metalness=0.0, clearcoat=0.1, description="Natural oak wood grain"),
    MaterialPreset(name="Wood (Walnut)", roughness=0.55, metalness=0.0, clearcoat=0.15, description="Rich walnut finish"),
    MaterialPreset(name="Metal (Brushed)", roughness=0.4, metalness=0.8, clearcoat=0.0, description="Brushed metal surface"),
    MaterialPreset(name="Metal (Polished)", roughness=0.1, metalness=0.95, clearcoat=0.0, description="Mirror-polished metal"),
    MaterialPreset(name="Plastic (Matte)", roughness=0.9, metalness=0.0, clearcoat=0.0, description="Matte plastic"),
    MaterialPreset(name="Plastic (Gloss)", roughness=0.2, metalness=0.0, clearcoat=0.5, description="Glossy plastic"),
    MaterialPreset(name="Stone / Marble", roughness=0.3, metalness=0.0, clearcoat=0.3, description="Polished stone"),
    MaterialPreset(name="Concrete", roughness=0.9, metalness=0.0, clearcoat=0.0, description="Raw concrete texture"),
    MaterialPreset(name="Ceramic", roughness=0.15, metalness=0.0, clearcoat=0.6, description="Glazed ceramic finish"),
    MaterialPreset(name="Carbon Fiber", roughness=0.3, metalness=0.6, clearcoat=0.2, description="Woven carbon fiber"),
]


# ============================================================================
# Static file serving
# ============================================================================

# Mount static directories FIRST so they catch /output/ and /static/ paths
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0", "model": "Furniture 3D Viewer"}


@app.get("/api/presets")
async def get_presets():
    """Get all color and material presets for the 3D viewer."""
    return {
        "colors": [c.model_dump() for c in COLOR_PRESETS],
        "materials": [m.model_dump() for m in MATERIAL_PRESETS],
    }


@app.post("/api/upload")
async def upload_photos(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload photos and automatically start 3D reconstruction.
    Returns a job ID for status tracking.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail=f"Need at least 2 photos, got {len(files)}. For best results use 8-20.")

    if len(files) > 50:
        raise HTTPException(status_code=400, detail=f"Maximum 50 photos, got {len(files)}")

    project_id = str(uuid.uuid4())[:8]
    project_dir = UPLOADS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, file in enumerate(files):
        ext = os.path.splitext(file.filename or f"photo_{i}.jpg")[1] or ".jpg"
        save_path = project_dir / f"photo_{i:03d}{ext}"
        content = await file.read()
        save_path.write_bytes(content)
        saved_paths.append(str(save_path))

    # Create job and auto-start processing
    job_id = f"job_{project_id}"
    processing_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": f"Uploaded {len(saved_paths)} photos. Starting 3D reconstruction...",
        "project_id": project_id,
        "photo_paths": saved_paths,
        "result": None,
        "started_at": None,
        "completed_at": None,
    }

    # Auto-start processing in background
    background_tasks.add_task(process_images, job_id)

    return {
        "job_id": job_id,
        "project_id": project_id,
        "photos_count": len(saved_paths),
        "status": "processing",
        "message": f"Uploaded {len(saved_paths)} photos. Reconstruction started automatically.",
    }


@app.post("/api/process/{job_id}")
async def start_processing(job_id: str, background_tasks: BackgroundTasks):
    """
    Start the photogrammetry processing pipeline for an uploaded project.
    This runs in the background and can take several minutes.
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found. Upload photos first.")

    job = processing_jobs[job_id]
    if job["status"] in ("processing", "completed"):
        return {"job_id": job_id, "status": job["status"], "message": "Already processing or completed."}

    job["status"] = "queued"
    job["progress"] = 0.0
    job["message"] = "Queued for processing..."

    # Start processing in background thread
    background_tasks.add_task(process_images, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Processing started. Use /api/status/{job_id} to check progress.",
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a processing job with detailed progress info."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    job = processing_jobs[job_id]
    
    # Calculate elapsed time
    elapsed = None
    if job.get("started_at"):
        if job.get("completed_at"):
            elapsed = round(job["completed_at"] - job["started_at"], 1)
        else:
            elapsed = round(time.time() - job["started_at"], 1)

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "progress_label": job.get("progress_label", ""),
        "message": job["message"],
        "elapsed_seconds": elapsed,
        "result": job.get("result"),
    }


@app.get("/api/models")
async def list_models():
    """List all available processed models."""
    models = []
    # Check output directory
    for glb_file in sorted(OUTPUT_DIR.glob("*.glb")):
        meta_file = glb_file.with_name(glb_file.stem.replace("_model", "_metadata") + ".json")
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                pass

        models.append({
            "id": glb_file.stem.replace("_model", ""),
            "name": meta.get("name", glb_file.stem),
            "glb": f"/output/{glb_file.name}",
            "vertices": meta.get("n_vertices", 0),
            "faces": meta.get("n_faces", 0),
            "created": meta.get("created", ""),
        })

    return {"models": models}


@app.get("/api/models/{project_id}")
async def get_model(project_id: str):
    """Get model details and download URLs."""
    glb_path = OUTPUT_DIR / f"{project_id}_model.glb"
    meta_path = OUTPUT_DIR / f"{project_id}_metadata.json"

    if not glb_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {project_id} not found.")

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    return {
        "project_id": project_id,
        "glb_url": f"/output/{project_id}_model.glb",
        "obj_url": f"/output/{project_id}_model.obj" if (OUTPUT_DIR / f"{project_id}_model.obj").exists() else None,
        "metadata": meta,
        "colors": [c.model_dump() for c in COLOR_PRESETS],
        "materials": [m.model_dump() for m in MATERIAL_PRESETS],
    }


@app.get("/api/sample")
async def get_sample_model():
    """
    Get a sample furniture model URL for testing the viewer.
    If no models exist, generates a geometric sample sofa/chair.
    """
    # Check if we have any processed models
    glb_files = list(OUTPUT_DIR.glob("*.glb"))
    if glb_files:
        glb = glb_files[0]
        project_id = glb.stem.replace("_model", "")
        return await get_model(project_id)

    # Generate a sample model
    sample_id = "sample_sofa"
    sample_path = OUTPUT_DIR / f"{sample_id}_model.glb"

    if not sample_path.exists():
        success = generate_sample_model(str(OUTPUT_DIR), sample_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate sample model")

    return {
        "project_id": sample_id,
        "glb_url": f"/output/{sample_id}_model.glb",
        "metadata": {
            "project_id": sample_id,
            "name": "Sample Sofa",
            "n_vertices": 0,
            "n_faces": 0,
        },
        "colors": [c.model_dump() for c in COLOR_PRESETS],
        "materials": [m.model_dump() for m in MATERIAL_PRESETS],
    }


# ============================================================================
# Background processing
# ============================================================================

def process_images(job_id: str):
    """Run photogrammetry in a background thread with detailed progress updates."""
    import time as _time
    job = processing_jobs.get(job_id)
    if not job:
        return

    try:
        start_time = _time.time()
        job["started_at"] = start_time
        job["status"] = "processing"
        
        progress_stages = [
            (0.05, "Loading images..."),
            (0.10, "Extracting features..."),
            (0.20, "Matching features across images..."),
            (0.35, "Generating dense point cloud..."),
            (0.50, "Reconstructing surface mesh..."),
            (0.65, "Subdividing for smooth geometry..."),
            (0.75, "Transferring textures from photos..."),
            (0.85, "Optimizing mesh..."),
            (0.90, "Exporting with PBR materials..."),
            (0.95, "Finalizing..."),
        ]
        stage_idx = [0]

        def update_progress(label, pct):
            job["progress"] = pct
            job["progress_label"] = label
            job["message"] = label

        update_progress("Starting 3D reconstruction...", 0.02)

        photo_paths = job["photo_paths"]
        project_id = job["project_id"]
        output_dir = str(OUTPUT_DIR)

        update_progress(f"Processing {len(photo_paths)} photos...", 0.05)

        # Run photogrammetry
        result = run_photogrammetry(
            image_paths=photo_paths,
            output_dir=output_dir,
            project_id=project_id,
            progress_callback=update_progress,
        )

        elapsed = round(_time.time() - start_time, 1)
        job["completed_at"] = _time.time()

        if result.success:
            job["status"] = "completed"
            job["progress"] = 1.0
            job["progress_label"] = "Complete"
            job["message"] = result.message
            
            # Get model stats
            glb_size = 0
            if result.glb_path:
                glb_path = Path(result.glb_path)
                if glb_path.exists():
                    glb_size = glb_path.stat().st_size
            
            job["result"] = {
                "project_id": project_id,
                "glb_url": f"/output/{project_id}_model.glb",
                "obj_url": f"/output/{project_id}_model.obj" if result.obj_path else None,
                "n_vertices": len(result.mesh_vertices) if result.mesh_vertices is not None else 0,
                "n_faces": len(result.mesh_faces) if result.mesh_faces is not None else 0,
                "n_points": len(result.point_cloud) if result.point_cloud is not None else 0,
                "glb_size_kb": round(glb_size / 1024, 1),
                "elapsed_seconds": elapsed,
            }
        else:
            job["status"] = "failed"
            job["progress"] = 0.0
            job["progress_label"] = "Failed"
            job["message"] = result.message
            job["completed_at"] = _time.time()

    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["progress"] = 0.0
        job["progress_label"] = "Error"
        job["message"] = f"Error: {str(e)}"
        job["completed_at"] = _time.time()
        print(f"Processing error for {job_id}:")
        traceback.print_exc()


def generate_sample_model(output_dir: str, sample_id: str = "sample_sofa") -> bool:
    """
    Generate a geometric sample sofa/chair model using trimesh primitives.
    """
    try:
        import trimesh
        import numpy as np

        # Build a sofa from boxes and cylinders
        # Seat cushion
        seat = trimesh.creation.box(extents=[2.0, 1.0, 0.3])
        seat.apply_translation([0, 0, 0.35])

        # Back rest
        back = trimesh.creation.box(extents=[2.0, 0.2, 0.8])
        back.apply_translation([0, -0.4, 0.8])

        # Left armrest
        left_arm = trimesh.creation.box(extents=[0.2, 0.8, 0.5])
        left_arm.apply_translation([-1.0, 0, 0.45])

        # Right armrest
        right_arm = trimesh.creation.box(extents=[0.2, 0.8, 0.5])
        right_arm.apply_translation([1.0, 0, 0.45])

        # Seat cushion top (soft)
        cushion = trimesh.creation.box(extents=[1.8, 0.8, 0.15])
        cushion.apply_translation([0, 0, 0.6])

        # Legs
        legs = []
        for x, z in [(-0.8, -0.35), (0.8, -0.35), (-0.8, 0.35), (0.8, 0.35)]:
            leg = trimesh.creation.cylinder(radius=0.05, height=0.25)
            leg.apply_translation([x, z, -0.05])
            legs.append(leg)

        # Combine all parts
        all_parts = [seat, back, left_arm, right_arm, cushion] + legs

        # Simulate rounded corners by subdividing
        combined = trimesh.util.concatenate(all_parts)

        # Smooth the mesh slightly (subdivide)
        try:
            combined = combined.subdivide_to_size(0.15)
        except Exception:
            pass

        # Add vertex colors (warm grey)
        colors = np.full((len(combined.vertices), 4), [180, 160, 140, 255], dtype=np.uint8)
        combined.visual.vertex_colors = colors

        # Export
        output_path = Path(output_dir) / f"{sample_id}_model.glb"
        combined.export(str(output_path), file_type='glb')

        # Metadata
        meta = {
            "project_id": sample_id,
            "name": "Sample Sofa (Generated)",
            "n_vertices": len(combined.vertices),
            "n_faces": len(combined.faces),
            "is_sample": True,
            "created": str(np.datetime64('now')),
        }
        meta_path = Path(output_dir) / f"{sample_id}_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        print(f"  Generated sample sofa: {output_path}")
        return True

    except Exception as e:
        print(f"  Failed to generate sample model: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("""
╔══════════════════════════════════════════════════════╗
║        Furniture 3D Viewer - Backend Server          ║
║                                                      ║
║  API: http://localhost:8777/api                      ║
║  View: http://localhost:8777/                        ║
║  Docs: http://localhost:8777/docs                    ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8777, reload=False)
