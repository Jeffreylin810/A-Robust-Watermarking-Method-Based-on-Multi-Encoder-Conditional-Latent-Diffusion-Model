# O256 +15k refiner refresh1500 summary

## Main Table

| Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| formal_rgb | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 | N/A |
| plus15k_rgb_reuse | 32.3000 | 0.9578 | 0.0535 | 99.51 | 96.25 | 85.18 | Paper_Dropout_r03 | reuse_base_continuation | 60.71s |
| plus15k_refresh500 | 33.5967 | 0.9718 | 0.0410 | 99.34 | 95.76 | 83.48 | Paper_Dropout_r03 | 281.46s | 60.63s |
| plus15k_refresh1000 | 33.8667 | 0.9739 | 0.0392 | 99.27 | 95.69 | 83.16 | Paper_Dropout_r03 | 286.95s | 62.71s |
| plus15k_refresh1500 | 33.9835 | 0.9749 | 0.0385 | 99.24 | 95.66 | 83.05 | Paper_Dropout_r03 | 286.87s | 61.41s |

## Notes

- `plus15k_rgb_reuse` reuses the formal RGB refiner on the +15k base.
- `plus15k_refresh500` and `plus15k_refresh1000` are the existing short targeted refresh results.
- `plus15k_refresh1500` continues for another 500 steps from `plus15k_refresh1000` to test whether refiner-side quality push still preserves strong BA.
