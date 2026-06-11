#!/usr/bin/env bash
#
# Furniture 3D Viewer — Setup Script
# ===================================
# One-command setup: creates venv, installs deps, generates sample model.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════╗"
echo "║     Furniture 3D Viewer — Setup                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# 1. Check Python
PYTHON=""
for cmd in python3.11 python3.12 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.10+ is required. Install it first."
    exit 1
fi

echo "✓ Using: $($PYTHON --version)"

# 2. Check UV (install if missing)
UV=""
if command -v uv &>/dev/null; then
    UV="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    echo "→ Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | bash
    UV="$HOME/.local/bin/uv"
fi
echo "✓ UV: $($UV --version)"

# 3. Create venv
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    $UV venv .venv --python "$PYTHON"
fi
source .venv/bin/activate

# 4. Install dependencies
echo "→ Installing Python dependencies..."
$UV pip install -r requirements.txt 2>&1 | tail -3

# 5. Create directories
mkdir -p uploads static/output
echo "✓ Directories created"

# 6. Generate sample model
echo "→ Generating sample model..."
python3 -c "
import sys
sys.path.insert(0, '.')
from backend.main import generate_sample_model
success = generate_sample_model('static/output', 'sample_sofa')
print('✓ Sample model generated' if success else '⚠️ Could not generate sample model')
"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                     ║"
echo "║                                                     ║"
echo "║  Run:  ./run.sh                                      ║"
echo "║  Or:   python3 -m backend.main                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
