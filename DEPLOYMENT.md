# Deployment Guide

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- 4GB+ RAM (8GB recommended for larger photo sets)
- Linux (recommended) or WSL on Windows

## Option 1: Docker (Recommended)

```bash
# Build the image
docker build -t furniture-3d-viewer .

# Run with default settings
docker run -d \
  --name furniture-3d-viewer \
  -p 8777:8777 \
  -v furniture-data:/app/uploads \
  -v furniture-output:/app/static/output \
  --restart unless-stopped \
  furniture-3d-viewer

# Check logs
docker logs -f furniture-3d-viewer
```

## Option 2: Direct Python

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8777

# Or with gunicorn (production)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app \
  --bind 0.0.0.0:8777 \
  --timeout 120 \
  --max-requests 1000 \
  --max-requests-jitter 50
```

## Option 3: Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8777:8777"
    volumes:
      - uploads:/app/uploads
      - output:/app/static/output
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8777/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

volumes:
  uploads:
  output:
```

Run:
```bash
docker-compose up -d
```

## Environment Variables

Create `.env` file (optional — defaults work out of the box):

```env
# Server config
HOST=0.0.0.0
PORT=8777
WORKERS=4

# Path config
UPLOAD_DIR=./uploads
OUTPUT_DIR=./static/output

# Processing config
TARGET_FACES=100000
MAX_PHOTOS=50
MIN_PHOTOS=2
```

## Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name furniture.example.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8777;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:

1. On push/PR to `main`:
   - Runs ruff linting
   - Runs pytest with coverage
   - Runs mypy type checking

2. On push to `main` or tag `v*`:
   - Builds Docker image
   - Pushes to `ghcr.io/shubhsaxena2020/furniture-3d-viewer`
   - On tag push: creates a GitHub Release

## Monitoring

### Health Check
```bash
curl http://localhost:8777/api/health
# {"status":"ok","version":"1.0.0","model":"Furniture 3D Viewer"}
```

### Metrics to Watch
- **Processing time**: Should be < 3 min for 4 photos
- **Face count**: Should be 100K+ for quality models
- **Texture coverage**: Should be 95%+
- **Disk usage**: Each model is ~5-8MB GLB
- **Memory**: ~1-2GB during COLMAP processing
