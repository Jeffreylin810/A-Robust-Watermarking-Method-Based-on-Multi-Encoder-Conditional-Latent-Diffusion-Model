# O256 base-side bit-weighted weak-attack continuation summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| plus20k_refresh500 | rgb | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |
| A_weakattack_plus5k_rgb | rgb | 32.9388 | 0.9715 | 0.0483 | 99.89 | 98.21 | 89.61 | Paper_Dropout_r03 | 1109.26s | 60.40s |
| bitweighted_refiner_repair | rgb | 34.1088 | 0.9756 | 0.0364 | 99.33 | 95.76 | 82.78 | Paper_Dropout_r03 | 287.88s | 61.81s |
| base_bitweighted_weakattack_plus5k | base | 24.6803 | 0.7072 | 0.0957 | 100.00 | 97.36 | 85.96 | Paper_Dropout_r03 | 1095.38s | N/A |
| base_bitweighted_weakattack_plus5k_rgb_reuse_refresh500 | rgb | 32.9349 | 0.9714 | 0.0480 | 99.83 | 98.12 | 89.63 | Paper_Dropout_r03 | 1095.38s | 60.36s |

## Notes

- This experiment moves weak-bit weighting from the refiner-only stage into base-side continuation.
- The main comparison is against `A_weakattack_plus5k_rgb`: if this row improves BA without a larger quality drop, bit-aware base training is a real breakthrough.
- If it matches A but does not improve it, the diagnosis is useful but the simple weighting scheme is not enough; the next step should be channel/architecture-aware routing.
