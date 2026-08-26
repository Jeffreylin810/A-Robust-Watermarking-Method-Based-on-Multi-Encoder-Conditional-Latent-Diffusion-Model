# O256 +20k refiner refresh1500 summary

## Main Table

| Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| formal_rgb | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 | N/A |
| plus20k_rgb_reuse | 32.2933 | 0.9579 | 0.0529 | 99.55 | 96.49 | 85.68 | Paper_Dropout_r03 | reuse_base_continuation | 65.43s |
| plus20k_refresh500 | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |
| plus20k_refresh1500 | 34.2602 | 0.9768 | 0.0355 | 99.33 | 95.72 | 82.76 | Paper_Dropout_r03 | 472.89s | 60.91s |

## Notes

- `plus20k_rgb_reuse` reuses the formal RGB refiner on the +20k base.
- `plus20k_refresh500` is the current balanced +20k refresh result.
- `plus20k_refresh1500` continues for another 1000 steps from `plus20k_refresh500` to test a larger quality-push step on the stronger +20k base.
