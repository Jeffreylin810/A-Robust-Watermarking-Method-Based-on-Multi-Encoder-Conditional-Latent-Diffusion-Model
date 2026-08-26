# O256 base continuation BA-push summary

## Main Table

| Variant | Method | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| formal_base30k | O256-Base | 24.7037 | 0.7023 | 0.0934 | 99.30 | 87.33 | 70.38 | Paper_Dropout_r03 | 01:41:49.74 |
| formal_rgb | O256-RGB | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 |
| plus5k | O256-Base | 24.7481 | 0.7045 | 0.0913 | 99.63 | 91.00 | 76.12 | Paper_Dropout_r03 | 1093.22s |
| plus5k | O256-RGB | 32.6743 | 0.9583 | 0.0495 | 97.87 | 92.07 | 78.77 | Paper_Dropout_r03 | 60.60s |
| plus10k | O256-Base | 24.7590 | 0.7064 | 0.0903 | 99.88 | 93.17 | 77.97 | Paper_Dropout_r03 | 1088.47s |
| plus10k | O256-RGB | 32.4787 | 0.9581 | 0.0514 | 98.97 | 94.64 | 82.27 | Paper_Dropout_r03 | 60.77s |

## Notes

- `plus5k` means continuing the formal base from 30k to 35k total steps.
- `plus10k` means continuing from 35k to 40k total steps.
- `O256-RGB` variants reuse the existing formal RGB refinement module instead of retraining a new refiner.
