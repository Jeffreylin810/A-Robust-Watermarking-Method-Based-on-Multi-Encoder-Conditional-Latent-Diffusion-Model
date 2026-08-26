# Environment Notes

This project was developed in a local conda environment named `diffw`.
For portability, `environment.yml` is a cleaned environment specification instead of a raw local export.

## Tested Local Environment

- Python: 3.10.18
- PyTorch: 2.5.1
- CUDA used by PyTorch: 12.1
- torchvision in the original local env: 0.15.2a0, with a known `torchvision.io.image` warning
- Recommended portable torchvision in `environment.yml`: 0.20.1, matching PyTorch 2.5.1
- LPIPS: 0.1.4

## Create Environment

```bash
conda env create -f environment.yml
conda activate o256-watermark
```

If the target machine, such as TWCC, already provides a CUDA/PyTorch module, prefer the site-supported PyTorch installation and then install the remaining Python packages with pip or conda.

## Why Not Use a Raw `conda env export`

Raw exports include local paths, build strings, and many system libraries. Those files reproduce the exact machine more closely, but they are brittle across machines. The checked-in `environment.yml` is intended as the portable default.

## Expected Caveats

- CUDA driver compatibility still depends on the machine. The environment requests CUDA 12.1 runtime through `pytorch-cuda=12.1`.
- Dataset paths and checkpoint paths are not encoded in the environment. Use the experiment scripts and config files to set those paths.
- Very old checkpoints should be loaded with the same model code and config that produced them.
