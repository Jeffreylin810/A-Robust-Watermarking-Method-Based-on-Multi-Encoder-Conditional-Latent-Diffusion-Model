# O256 +15k base then refresh500 summary

## Main Table

| Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| formal_rgb | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 | N/A |
| plus10k_rgb_reuse | 32.4787 | 0.9581 | 0.0514 | 98.97 | 94.64 | 82.27 | Paper_Dropout_r03 | 60.77s | reuse_formal_refiner |
| refresh500_plus10k | 33.5891 | 0.9702 | 0.0411 | 98.59 | 94.10 | 80.79 | Paper_Dropout_r03 | 284.74s | 63.42s |
| plus15k_rgb_reuse | 32.3000 | 0.9578 | 0.0535 | 99.51 | 96.25 | 85.18 | Paper_Dropout_r03 | reuse_base_continuation | 60.71s |
| plus15k_refresh500 | 33.5967 | 0.9718 | 0.0410 | 99.34 | 95.76 | 83.48 | Paper_Dropout_r03 | 281.46s | 60.63s |

## Notes

- `plus15k_rgb_reuse` reuses the existing formal RGB refinement module on the +15k base.
- `plus15k_refresh500` performs 500 targeted refiner-refresh steps on the +15k base, starting from the formal refiner.
- Baseline rows include the current formal RGB, the strongest reuse variant on +10k base, and the best-balanced `refresh500_plus10k` variant for direct comparison.
