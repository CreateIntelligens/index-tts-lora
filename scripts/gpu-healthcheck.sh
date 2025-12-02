#!/bin/bash
set -euo pipefail

# Fast check that the NVIDIA runtime and GPU are visible.
if ! timeout 3 nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU healthcheck: nvidia-smi not available or GPU missing" >&2
  exit 1
fi

# Ensure the deep learning stack sees CUDA (not just the driver).
python3 - <<'PY'
import sys
try:
    import torch  # noqa: F401
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)
PY
