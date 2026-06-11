# Furniture 3D Viewer

**Turn furniture photos into interactive 3D models** — a complete tool for furniture shops to create stunning 3D product views for their websites.

Take 12–20 photos of a piece of furniture from different angles, and this tool generates a fully interactive 3D model with built-in color and material customization. Embed the viewer on any website so customers can rotate, zoom, and see the furniture in different colors and textures.

---

## ✨ Features

| Feature | Description |
|---|---|
| **📸 Photo → 3D** | Upload 12–20 photos, get a textured 3D model |
| **🎨 16 Colors** | Click to change furniture color instantly |
| **🧵 14 Materials** | Switch between fabric, leather, wood, metal, and more |
| **🔄 Auto-Rotate** | 360° spinning product view |
| **🖼️ Screenshots** | Capture high-res product images |
| **◇ Wireframe** | Toggle wireframe overlay |
| **🔗 Embeddable** | Copy-paste `<iframe>` code for your website |
| **📱 Responsive** | Works on desktop and mobile |
| **🏪 Ready for shops** | White-label, self-hosted, full control |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/shubhsaxena2020/furniture-3d-viewer.git
cd furniture-3d-viewer

# 2. Set up the environment
chmod +x setup.sh
./setup.sh

# 3. Run the server
./run.sh

# 4. Open your browser
open http://localhost:8777
```

### Manual setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start
python3 -m backend.main
```

---

## 📖 How To Use

### Step 1: Photograph the furniture

Take **12–20 photos** of the furniture from different angles:

```
        Top-down view
           📷
            |
   Left ◀━━━⬤━━━▶ Right
    📷      |      📷
            |
        📷 Front
```

**Photography tips:**
- Use a plain, well-lit background (white/grey)
- Keep the camera at a consistent distance
- Ensure the entire piece is visible in every shot
- Move around the furniture in a smooth arc (not random angles)
- Overlap each photo by ~30% with the previous one
- Avoid motion blur — use a tripod if possible
- Consistent lighting (no harsh shadows)

### Step 2: Upload & generate

1. Open `http://localhost:8777`
2. Drag & drop your photos into the upload area
3. Click **"Generate 3D Model"**
4. Wait while the photogrammetry pipeline processes (30s–2min)

### Step 3: Customize colors & materials

- Click any color swatch to instantly recolor the furniture
- Choose from 16 furniture-appropriate colors
- Pick a material (linen, velvet, leather, wood, metal, etc.)

### Step 4: Embed on your website

Click **"Copy"** to get an embed URL or `<iframe>` code. Paste it into any website:

```html
<iframe 
  src="http://your-server.com/?model=/output/your_model.glb" 
  width="100%" 
  height="500" 
  frameborder="0" 
  allowfullscreen>
</iframe>
```

---

## 🖼️ 3D Viewer Controls

| Control | Action |
|---|---|
| **Drag** | Rotate the model |
| **Scroll** | Zoom in/out |
| **Right-click + drag** | Pan the view |
| **Reset View** | Back to default camera |
| **Auto-Rotate** | Toggle automatic 360° rotation |
| **Screenshot** | Download PNG of current view |
| **Wireframe** | Toggle wireframe overlay |

---

## 🏗️ Architecture

```
furniture-3d-viewer/
│
├── backend/                  # Python FastAPI server
│   ├── main.py              # API endpoints, file management
│   └── photogrammetry.py    # SfM pipeline (feature extraction,
│                            # matching, point cloud, mesh)
│
├── frontend/                 # Web UI (Three.js)
│   └── index.html           # Complete 3D viewer app
│
├── uploads/                  # Uploaded photos (per project)
├── static/
│   └── output/              # Generated GLB models + metadata
│
├── run.sh                   # Launch script
├── setup.sh                 # One-command setup
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

### Pipeline

```
Photos (12-20)
    │
    ▼
SIFT Feature Extraction (cv2.SIFT)
    │
    ▼
Feature Matching (FLANN + Lowe's ratio test)
    │
    ▼
Structure from Motion (incremental SfM)
    │
    ▼
Point Cloud Densification
    │
    ▼
Surface Reconstruction (Delaunay / Poisson)
    │
    ▼
Mesh Smoothing & Cleanup
    │
    ▼
GLB Export (glTF Binary)
    │
    ▼
3D Web Viewer (Three.js)
```

---

## 🧵 Material Presets

| Material | Roughness | Metalness | Best For |
|---|---|---|---|
| Fabric (Linen) | 0.85 | 0.0 | Sofas, chairs, upholstery |
| Fabric (Velvet) | 0.50 | 0.0 | Premium seating |
| Leather (Matte) | 0.70 | 0.0 | Couches, armchairs |
| Leather (Gloss) | 0.30 | 0.0 | Modern furniture |
| Wood (Oak) | 0.60 | 0.0 | Tables, cabinets |
| Wood (Walnut) | 0.55 | 0.0 | Premium wood pieces |
| Metal (Brushed) | 0.40 | 0.8 | Legs, frames, accents |
| Metal (Polished) | 0.10 | 0.95 | Chrome/details |
| Plastic (Matte) | 0.90 | 0.0 | Modern/outdoor |
| Plastic (Gloss) | 0.20 | 0.0 | Contemporary |
| Stone / Marble | 0.30 | 0.0 | Tabletops, decor |
| Concrete | 0.90 | 0.0 | Industrial style |
| Ceramic | 0.15 | 0.0 | Accents, decor |
| Carbon Fiber | 0.30 | 0.6 | Modern/tech |

---

## 🎨 Color Presets

Classic Black, Pure White, Warm Grey, Navy Blue, Forest Green, Burgundy, Sage Green, Dusty Rose, Slate Blue, Warm Beige, Terracotta, Charcoal, Cream, Teal, Mustard Yellow, Coral

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server health check |
| GET | `/api/presets` | Get color & material presets |
| GET | `/api/models` | List all available models |
| GET | `/api/models/{id}` | Get model details |
| POST | `/api/upload` | Upload photos (multipart) |
| POST | `/api/process/{job_id}` | Start 3D reconstruction |
| GET | `/api/status/{job_id}` | Check processing status |
| GET | `/api/sample` | Get sample sofa model |
| GET | `/output/{file}` | Download model files (GLB/OBJ) |

---

## 📦 Deployment

### Run as a service (systemd)

```bash
# /etc/systemd/system/furniture-viewer.service
[Unit]
Description=Furniture 3D Viewer
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/furniture-3d-viewer
ExecStart=/opt/furniture-3d-viewer/.venv/bin/python3 -m backend.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8777
CMD ["python3", "-m", "backend.main"]
```

### Nginx reverse proxy

```nginx
server {
    listen 80;
    server_name furniture.your-shop.com;

    location / {
        proxy_pass http://127.0.0.1:8777;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 86400s;
    }

    client_max_body_size 200M;
}
```

---

## 🧪 Testing

```bash
# Start server
./run.sh &
sleep 3

# Test API
curl http://localhost:8777/api/health
curl http://localhost:8777/api/presets
curl http://localhost:8777/api/models

# Test sample model serving
curl -o /dev/null -w "%{http_code}" http://localhost:8777/output/sample_sofa_model.glb
```

---

## 🛠️ Requirements

- **Python 3.10+** (tested on 3.14)
- **Node.js** (for Three.js CDN — no local install needed)
- **4GB+ RAM** (8GB+ recommended for photogrammetry)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies

- `fastapi` — Web API framework
- `uvicorn` — ASGI server
- `opencv-contrib-python-headless` — SIFT features, image processing
- `numpy` — Numerical computing
- `scipy` — Delaunay triangulation, spatial operations
- `trimesh` — 3D mesh processing and export
- `pillow` — Image loading and EXIF parsing
- `python-multipart` — File upload parsing

---

## 📄 License

MIT License — free for commercial and personal use.

---

## 🤝 Contributing

Pull requests welcome! For major changes:

1. Fork the repo
2. Create a feature branch
3. Test your changes
4. Open a PR with a clear description

---

## ⚡ Performance Notes

- **CPU-only**: The photogrammetry pipeline runs entirely on CPU (no GPU required).
- **Processing time**: ~30s–2min for 12-20 photos depending on resolution and CPU.
- **Model size**: Output GLB files are typically 200KB–5MB depending on complexity.
- **Memory**: Peak RAM ~2-4GB during photogrammetry processing.
- **Concurrency**: Supports multiple simultaneous processing jobs.

---

*Built with ❤️ for furniture shops everywhere.*
