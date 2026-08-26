# O256 +20k weak-attack base continuation (+5k) summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| formal_rgb | rgb | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 02:33:57.31 | N/A |
| plus20k_rgb_reuse | rgb | 32.2933 | 0.9579 | 0.0529 | 99.55 | 96.49 | 85.68 | Paper_Dropout_r03 | reuse_base_continuation | 65.43s |
| plus20k_refresh500 | rgb | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |
| plus20k_weakattack_plus5k_base | base | 24.6837 | 0.7073 | 0.0957 | 99.99 | 97.41 | 86.23 | Paper_Dropout_r03 | 1109.26s | N/A |
| plus20k_weakattack_plus5k_rgb_reuse_refresh500 | rgb | 32.9388 | 0.9715 | 0.0483 | 99.89 | 98.21 | 89.61 | Paper_Dropout_r03 | 1109.26s | 60.40s |

## Notes

- `plus20k_weakattack_plus5k_base` continues the +20k base by one extra 5k stage using the new `identity_nongeom_targeted_weak` profile.
- `plus20k_weakattack_plus5k_rgb_reuse_refresh500` applies the existing `plus20k_refresh500` refiner to that weak-targeted base to isolate the base-side effect on the final RGB operating point.
- Baseline rows keep the current formal / +20k reference points in view for direct trade-off comparison.
