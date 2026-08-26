# O256 exp42A dropout-floor safety margin probe summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Extra base train s | Refiner train s | Eval/apply s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| exp40_seed61_full_rerun_reference_rgb | reference | 33.9372 | 0.9819 | 0.0409 | 99.96 | 98.77 | 91.01 | Paper_Dropout_r03 | N/A | 2527.77 | 59.65 |
| exp41_seed62_full_rerun_reference_rgb | reference | 34.0293 | 0.9820 | 0.0443 | 99.98 | 98.68 | 89.94 | Paper_Dropout_r03 | N/A | 2522.55 | 59.98 |
| exp42A_base70_dropout_margin_plus10k | base | 24.6605 | 0.7084 | 0.1011 | 100.00 | 98.27 | 86.92 | Paper_Dropout_r03 | 1071.43 | N/A | 100.03 |
| exp42A_rgb_base70_dropout_balanced | rgb | 34.0741 | 0.9824 | 0.0441 | 100.00 | 98.73 | 89.80 | Paper_Dropout_r03 | 1071.43 | 2533.70 | 59.75 |

## RGB Per-Attack BA

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 89.80 |
| Paper_GaussianBlur_sigma2 | 100.00 |
| Paper_GaussianBlur_sigma3 | 99.99 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 98.14 |
| Paper_GaussianNoise_sigma03_var009 | 99.64 |
| Paper_JPEG_Q30 | 99.87 |
| Paper_JPEG_Q40 | 99.95 |
| Paper_JPEG_Q50 | 99.98 |
| Paper_MedianFilter_w5 | 99.99 |
| Paper_SaltPepper_r02 | 98.69 |

## Notes

- exp42A is a continuation probe, not a formal main result.
- It starts from exp41 seed62 base65 and adds one extra `identity_nongeom_dropout_margin` epoch to form base70.
- The RGB refiner is trained again with the fixed exp40/41 A/B/B+/C/D recipe; no seed-specific refiner tweak is introduced.
- Promotion criterion for exp42B: PSNR >= 33.5, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Probe did not pass; do not promote this fixed-margin idea to a full from-scratch rerun yet.
