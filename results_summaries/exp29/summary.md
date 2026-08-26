# O256 exp28 quality recovery refresh500 summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| plus20k_refresh500 | rgb | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |
| A_weakattack_plus5k_rgb | rgb | 32.9388 | 0.9715 | 0.0483 | 99.89 | 98.21 | 89.61 | Paper_Dropout_r03 | 1109.26s | 60.40s |
| A_dropout_focus_plus5k_rgb_reuse_refresh500 | rgb | 32.5223 | 0.9705 | 0.0523 | 99.91 | 98.64 | 91.43 | Paper_Dropout_r03 | 1091.72s | 60.94s |
| exp28_quality_recovery_refresh500 | rgb | 33.6062 | 0.9770 | 0.0412 | 99.88 | 98.17 | 88.63 | Paper_Dropout_r03 | 285.17s | 61.69s |

## Notes

- This experiment tests whether a short RGB refiner refresh can recover image quality after exp28's dropout-focus robustness scheduling.
- Success criterion: PSNR returns to 33+ while Min BA remains above 90%.
- If PSNR improves but Min BA falls below 90%, the next step should use weaker image-loss pressure or stronger attack-message preservation.
