# Final Code Version Note

This folder contains a cleaned lightweight copy of the latest O256-PAPER working code from:

```bash
/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
```

It includes the full O256 / StageE code path:

- base model training
- base model evaluation
- RGB refiner training
- RGB refiner application
- formal attack evaluation
- BA / PSNR / SSIM / LPIPS metrics
- EXP52 / EXP53 / EXP54 StageE runner scripts
- configs / models / datasets loader / noise layers / utils

It excludes large or generated folders:

```text
ckpts/
logs/
results/
Result/
temp/
webp_50/
__pycache__/
*.pth
*.pth.tar
*.npy
raw images
attacked outputs
```

The final protocol is not a single experiment folder. It is implemented by the StageE runner scripts inside `O256-PAPER/scripts/`.

