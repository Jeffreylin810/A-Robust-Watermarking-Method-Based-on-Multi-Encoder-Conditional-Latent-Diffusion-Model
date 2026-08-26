#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${TWCC_CONDA_ENV_NAME:-o256-watermark}"
if [[ -n "${TWCC_CONDA_ROOT:-}" ]]; then
  CONDA_ROOT="$TWCC_CONDA_ROOT"
elif [[ -d "/work/${USER:-}" ]]; then
  CONDA_ROOT="/work/${USER}/miniforge3"
else
  CONDA_ROOT="$HOME/miniforge3"
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    echo "conda was not found in PATH. Installing Miniforge to: $CONDA_ROOT"
    installer="/tmp/Miniforge3-Linux-x86_64.sh"
    if command -v wget >/dev/null 2>&1; then
      wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$installer"
    elif command -v curl >/dev/null 2>&1; then
      curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$installer"
    else
      echo "Neither wget nor curl was found; cannot download Miniforge." >&2
      exit 1
    fi
    bash "$installer" -b -p "$CONDA_ROOT"
  fi
  export PATH="$CONDA_ROOT/bin:$PATH"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Reuse existing conda env: $ENV_NAME"
else
  echo "Create conda env: $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.10 pip
fi

conda activate "$ENV_NAME"

echo "Install V100-compatible PyTorch stack..."
# TWCC PyTorch containers may start from a newer CUDA/PyTorch image that drops
# V100 sm_70 support. Install the official cu121 wheels inside this env so the
# runtime is independent of the container's built-in torch package.
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "Install project dependencies..."
conda install -y -c conda-forge \
  numpy=1.26 scipy pandas pillow opencv scikit-image matplotlib pyyaml tqdm \
  tensorboard tensorboardx kornia requests tmux

python -m pip install --upgrade pip
python -m pip install \
  lpips==0.1.4 \
  pytorch-msssim==1.0.0 \
  bm3d==4.0.3 \
  bm4d==4.2.5 \
  compressai==1.2.8 \
  diffusers==0.37.0 \
  accelerate==1.13.0 \
  safetensors==0.7.0 \
  einops==0.8.2 \
  imageio==2.37.3 \
  psutil==7.2.2 \
  rich==14.3.3 \
  "transformers>=4.40,<5"

echo
echo "Verify PyTorch CUDA arch support..."
python - <<'PY'
import torch
print("python ok")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("arch list:", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.ones((1,), device="cuda")
    print("cuda tensor:", x.item())
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit("ERROR: this PyTorch build does not list sm_70; V100 may not be supported.")
PY

echo
echo "Done. Activate with:"
echo "conda activate $ENV_NAME"
