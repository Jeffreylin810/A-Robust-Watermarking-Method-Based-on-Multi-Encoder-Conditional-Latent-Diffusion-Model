# O256 +20k base then refresh500 summary

## Main Table

| Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| formal_rgb | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 | N/A |
| plus15k_rgb_reuse | 32.3000 | 0.9578 | 0.0535 | 99.51 | 96.25 | 85.18 | Paper_Dropout_r03 | reuse_base_continuation | 60.71s |
| plus15k_refresh500 | 33.5967 | 0.9718 | 0.0410 | 99.34 | 95.76 | 83.48 | Paper_Dropout_r03 | 281.46s | 60.63s |
| plus15k_refresh1000 | 33.8667 | 0.9739 | 0.0392 | 99.27 | 95.69 | 83.16 | Paper_Dropout_r03 | 286.95s | 62.71s |
| plus20k_rgb_reuse | 32.2933 | 0.9579 | 0.0529 | 99.55 | 96.49 | 85.68 | Paper_Dropout_r03 | reuse_base_continuation | 65.43s |
| plus20k_refresh500 | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |

## Notes

- `plus20k_rgb_reuse` reuses the formal RGB refiner on the +20k base.
- `plus20k_refresh500` performs 500 targeted refiner-refresh steps on the +20k base, starting from the formal refiner.
- Baseline rows include formal RGB and the existing strongest +15k family variants for direct comparison.
