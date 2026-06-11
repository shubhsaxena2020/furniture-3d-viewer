#!/usr/bin/env bash
#
# Furniture 3D Viewer — Launch Script
# ====================================
# Starts the backend server and opens the web UI.
#
# Usage:
#   ./run.sh              # Start server
#   ./run.sh --dev        # Start with live reload
#   ./run.sh --port=9999  # Custom port
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1#--port=}"
if [[ "$1" == "--dev" ]]; then
    PORT="${2#--port=}"
    RELOAD="--reload"
fi
PORT="${PORT:-8777}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        Furniture 3D Viewer                          ║"
echo "║                                                     ║"
echo "║  Open in your browser:                              ║"
echo "║  →  http://localhost:${PORT}                        ║"
echo "║                                                     ║"
echo "║  Upload 12-20 photos of furniture from different    ║"
echo "║  angles to generate an interactive 3D model with    ║"
echo "║  customizable colors and materials.                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

python3 -m backend.main --port "$PORT" ${RELOAD:-}
