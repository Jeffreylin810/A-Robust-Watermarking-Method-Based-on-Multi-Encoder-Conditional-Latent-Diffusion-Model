# O256 +20k weak-targeted refresh1000 summary

## Main Table

| Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| formal_rgb | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 | N/A |
| plus20k_refresh500 | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |
| plus20k_refresh1500 | 34.2602 | 0.9768 | 0.0355 | 99.33 | 95.72 | 82.76 | Paper_Dropout_r03 | 503.79s | 61.57s |
| plus20k_weaktargeted1000 | 34.1123 | 0.9757 | 0.0364 | 99.35 | 95.78 | 82.87 | Paper_Dropout_r03 | 282.02s | 61.72s |

## Notes

- `plus20k_weaktargeted1000` is the short weak-attack-targeted refresh experiment: it resumes from `plus20k_refresh500` and adds 500 more targeted refiner steps.
- `plus20k_refresh1500` remains the more aggressive quality-push reference point.
- The comparison is intended to reveal whether a shorter weak-targeted refresh can recover weakest attacks more efficiently than the longer quality-push branch.
