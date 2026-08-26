# O256 exp41 seed62 full from-scratch formal rerun summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Base train s | Refiner train s | Eval/apply s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| exp40_seed61_full_rerun_reference_rgb | reference | 33.9372 | 0.9819 | 0.0409 | 99.96 | 98.77 | 91.01 | Paper_Dropout_r03 | 13632.29 | 2527.77 | 59.65 |
| exp38_seed62_continuation_reference_rgb | reference | 33.1252 | 0.9745 | 0.0467 | 99.93 | 98.60 | 90.31 | Paper_Dropout_r03 | N/A | N/A | N/A |
| exp41_base65_dropout_margin | base | 24.6709 | 0.7082 | 0.1005 | 100.00 | 98.12 | 86.64 | Paper_Dropout_r03 | 13587.65 | N/A | 99.42 |
| exp41_rgb_full_from_scratch_dropout_balanced | rgb | 34.0293 | 0.9820 | 0.0443 | 99.98 | 98.68 | 89.94 | Paper_Dropout_r03 | 13587.65 | 2522.55 | 59.98 |

## RGB Per-Attack BA

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 89.94 |
| Paper_GaussianBlur_sigma2 | 99.99 |
| Paper_GaussianBlur_sigma3 | 99.98 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 97.82 |
| Paper_GaussianNoise_sigma03_var009 | 99.60 |
| Paper_JPEG_Q30 | 99.81 |
| Paper_JPEG_Q40 | 99.89 |
| Paper_JPEG_Q50 | 99.93 |
| Paper_MedianFilter_w5 | 99.97 |
| Paper_SaltPepper_r02 | 98.53 |

## Notes

- This is the seed62 stability rerun of exp40: identity20k, nongeom-light to base50, weakattack +5k, dropout-focus +5k, dropout-margin +5k, regenerated train10000 pairs, and a freshly trained RGB refiner.
- Refiner stages are A10k, B2k, B+500, targeted C500, and final dropout-balanced D500.
- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Did not pass all seed62 full from-scratch thresholds; compare against exp40 seed61 and exp38 seed62 continuation.
