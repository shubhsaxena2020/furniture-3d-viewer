# Furniture 3D Viewer

Turn furniture photos into interactive 3D models with color/material customization.

## Features

- **Multi-view Photogrammetry**: Upload 4-20 photos from different angles of a furniture piece
- **3D Model Generation**: Automatic reconstruction using COLMAP + custom Multi-View Stereo
- **100K+ Faces**: High-quality subdivision for detailed geometry
- **95%+ Texture Coverage**: Multi-view per-vertex color blending
- **PBR Materials**: Auto-detected material properties (fabric, leather, metal, wood)
- **16 Color Presets + 14 Material Presets**: Instant visual customization
- **GLB/OBJ Export**: Standard 3D formats for embedding anywhere
- **Interactive Viewer**: Three.js-based web viewer with auto-rotate, wireframe, screenshot

## Quick Start

```bash
# Clone the repo
git clone https://github.com/shubhsaxena2020/furniture-3d-viewer.git
cd furniture-3d-viewer

# Install dependencies (Python 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the server
python -m backend.main
```

Open http://localhost:8777 in your browser.

## Docker

```bash
# Build
docker build -t furniture-3d-viewer .

# Run
docker run -p 8777:8777 furniture-3d-viewer
```

Or pull from GitHub Container Registry:
```bash
docker pull ghcr.io/shubhsaxena2020/furniture-3d-viewer:latest
docker run -p 8777:8777 ghcr.io/shubhsaxena2020/furniture-3d-viewer:latest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/presets` | GET | Get color and material presets |
| `/api/upload` | POST | Upload photos (4-50) and start reconstruction |
| `/api/status/{job_id}` | GET | Check processing status |
| `/api/models` | GET | List all processed models |
| `/api/models/{id}` | GET | Get model details and download URLs |
| `/api/sample` | GET | Get a sample model for testing |

### Upload Example

```bash
curl -X POST http://localhost:8777/api/upload \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg" \
  -F "files=@photo4.jpg"
```

Returns a `job_id` for status tracking.

### Check Status

```bash
curl http://localhost:8777/api/status/job_abc12345
```

### View Model

Open http://localhost:8777 in your browser and click the model ID.

## Photography Tips

For best results:
1. Take 8-20 photos around the object (every 15-30 degrees)
2. Keep consistent lighting (no bright shadows)
3. Capture from slightly above (~30° elevation)
4. Ensure 50%+ overlap between adjacent photos
5. Avoid plain white/black objects (need texture features)
6. Resolution: 720p-1080p works best
7. Plain background helps reconstruction

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment guides.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## License

MIT
