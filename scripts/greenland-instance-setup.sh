#!/usr/bin/env bash
#
# greenland-instance-setup.sh — Run ON the Greenland box (via sync deploy or SSH)
# ------------------------------------------------------------------------------
# Builds a Python venv that INHERITS the base image's CUDA-enabled torch and
# scientific stack (--system-site-packages), then installs only the RL deps
# that are missing. Never reinstalls torch (would replace the CUDA build).
#
# Usage (on the box):  bash scripts/greenland-instance-setup.sh [quick|full]
#   quick (default) — create/reuse venv, install missing RL deps
#   full            — same as quick (kept for interface parity)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/SatTwin}"
VENV_DIR="$PROJECT_DIR/.venv-gl"
cd "$PROJECT_DIR"

echo "============================================"
echo " SatTwin Greenland instance setup"
echo "============================================"

echo "[1/4] Base interpreter + CUDA torch check..."
python3 --version
python3 -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.device_count(), 'GPUs')"

echo "[2/4] Creating venv (inherits system site-packages)..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
  echo "  ✓ created $VENV_DIR"
else
  echo "  ✓ reusing $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip

echo "[3/4] Installing missing RL deps (torch is inherited, NOT reinstalled)..."
python - <<'PY'
import importlib, subprocess, sys
need = []
for mod, pkg in [("torch_geometric", "torch-geometric"),
                 ("pettingzoo", "pettingzoo"),
                 ("gymnasium", "gymnasium"),
                 ("dash", "dash"),
                 ("plotly", "plotly")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    print("  installing:", need)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *need])
else:
    print("  nothing missing")
PY

echo "[4/4] Verifying the full stack under the venv..."
python - <<'PY'
import torch, torch_geometric, pettingzoo, gymnasium, networkx, numpy, scipy, pandas, matplotlib, yaml
print("  torch           ", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.device_count(), "GPUs")
print("  torch_geometric ", torch_geometric.__version__)
print("  pettingzoo      ", pettingzoo.__version__)
print("  gymnasium       ", gymnasium.__version__)
print("  networkx        ", networkx.__version__)
print("  numpy           ", numpy.__version__)
PY

echo "============================================"
echo " ✅ Setup complete."
echo "    Activate with: source $VENV_DIR/bin/activate"
echo "============================================"
