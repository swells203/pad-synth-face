#!/usr/bin/env bash
# Bootstrap the DGX Spark for the PAD scaling sweep.
# Idempotent: safe to re-run. Assumes ~/ml/projects/pad-spark/ is the
# project checkout (created by the caller via git clone / rsync).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/ml/projects/pad-spark}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: PROJECT_DIR=$PROJECT_DIR does not exist." >&2
  echo "  Push the repo to the Spark first (git clone / rsync)." >&2
  exit 2
fi

cd "$PROJECT_DIR"

# 1) uv (skip if already installed)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer adds ~/.local/bin to PATH for new shells; export for this one.
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Venv (skip if already created)
if [[ ! -d ".venv" ]]; then
  uv venv --python 3.12
fi

# 3) Install PyTorch nightly with CUDA 12.8 (Blackwell sm_121 support)
#    + the small runtime dep set the sweep needs.
uv pip install --upgrade \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  torch torchvision

uv pip install --upgrade numpy pillow pyyaml pytest

# 4) Freeze the resolved versions for reproducibility (one-shot snapshot;
#    re-runs overwrite to reflect the latest nightly).
uv pip freeze > requirements.spark.txt

# 5) Sanity-check: torch sees CUDA + the GPU is the expected GB10.
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available after install"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
PY

echo "Spark setup OK."
