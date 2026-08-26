# O256 RGB refiner refresh on +10k base

## Main Table

| Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| formal_rgb | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 | N/A |
| plus10k_rgb_reuse | 32.4787 | 0.9581 | 0.0514 | 98.97 | 94.64 | 82.27 | Paper_Dropout_r03 | 60.77s | reuse_formal_refiner |
| refresh500 | 33.5891 | 0.9702 | 0.0411 | 98.59 | 94.10 | 80.79 | Paper_Dropout_r03 | 284.74s | 63.42s |
| refresh1000 | 33.7365 | 0.9716 | 0.0401 | 98.56 | 94.09 | 80.73 | Paper_Dropout_r03 | 279.71s | 60.79s |

## Notes

- `plus10k_rgb_reuse` reuses the existing formal RGB refinement module on the +10k base.
- `refresh500` performs 500 targeted refiner-refresh steps starting from the formal refiner.
- `refresh1000` continues from `refresh500` for another 500 steps (1000 total refresh steps).
- Both refresh candidates use the same fixed +10k O256-Base checkpoint and full finaltest1000 non-geometric evaluation.
