#!/usr/bin/env bash
set -euo pipefail

echo "== Basic =="
date
hostname
pwd
whoami || true

echo
echo "== GPU =="
nvidia-smi || true

echo
echo "== Storage =="
df -h

echo
echo "== Memory =="
free -h || true

echo
echo "== Python =="
which python || true
python --version || true

echo
echo "== Conda =="
which conda || true
conda env list || true

echo
echo "== PyTorch =="
python - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"device {idx}:", torch.cuda.get_device_name(idx))
except Exception as exc:
    print("torch import failed:", repr(exc))
PY

echo
echo "== Optional Packages =="
python - <<'PY' || true
packages = ["yaml", "PIL", "cv2", "numpy", "skimage", "lpips"]
for name in packages:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: missing ({exc})")
PY
