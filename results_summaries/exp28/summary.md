# O256 attack scheduling dropout-focus from A summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| plus20k_refresh500 | rgb | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |
| A_weakattack_plus5k_rgb | rgb | 32.9388 | 0.9715 | 0.0483 | 99.89 | 98.21 | 89.61 | Paper_Dropout_r03 | 1109.26s | 60.40s |
| bitweighted_refiner_repair | rgb | 34.1088 | 0.9756 | 0.0364 | 99.33 | 95.76 | 82.78 | Paper_Dropout_r03 | 287.88s | 61.81s |
| base_bitweighted_weakattack_plus5k_rgb | rgb | 32.9349 | 0.9714 | 0.0480 | 99.83 | 98.12 | 89.63 | Paper_Dropout_r03 | 1095.38s | 60.36s |
| A_dropout_focus_plus5k | base | 24.6456 | 0.7073 | 0.0986 | 100.00 | 98.02 | 87.95 | Paper_Dropout_r03 | 1091.72s | N/A |
| A_dropout_focus_plus5k_rgb_reuse_refresh500 | rgb | 32.5223 | 0.9705 | 0.0523 | 99.91 | 98.64 | 91.43 | Paper_Dropout_r03 | 1091.72s | 60.94s |

## Notes

- This experiment resumes from A_weakattack_plus5k and adds one dropout-focused scheduling epoch.
- The main comparison is against `A_weakattack_plus5k_rgb`: if Dropout rises without collapsing GN/SP or quality, attack scheduling is useful.
- If it matches or hurts A, simple post-A attack scheduling is not enough and the next step should move toward channel/architecture-aware routing.
