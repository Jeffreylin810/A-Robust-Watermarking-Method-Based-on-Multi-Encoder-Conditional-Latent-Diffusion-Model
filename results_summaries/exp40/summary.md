# O256 exp40 full from-scratch formal rerun summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Base train s | Refiner train s | Eval/apply s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| exp38_seed62_continuation_reference_rgb | reference | 33.1252 | 0.9745 | 0.0467 | 99.93 | 98.60 | 90.31 | Paper_Dropout_r03 | N/A | N/A | N/A |
| exp40_base65_dropout_margin | base | 24.6354 | 0.7084 | 0.0992 | 100.00 | 98.45 | 89.18 | Paper_Dropout_r03 | 13632.29 | N/A | 105.04 |
| exp40_rgb_full_from_scratch_dropout_balanced | rgb | 33.9372 | 0.9819 | 0.0409 | 99.96 | 98.77 | 91.01 | Paper_Dropout_r03 | 13632.29 | 2527.77 | 59.65 |

## RGB Per-Attack BA

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 91.01 |
| Paper_GaussianBlur_sigma2 | 99.98 |
| Paper_GaussianBlur_sigma3 | 99.98 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 97.75 |
| Paper_GaussianNoise_sigma03_var009 | 99.58 |
| Paper_JPEG_Q30 | 99.87 |
| Paper_JPEG_Q40 | 99.88 |
| Paper_JPEG_Q50 | 99.93 |
| Paper_MedianFilter_w5 | 99.98 |
| Paper_SaltPepper_r02 | 98.48 |

## Notes

- This is the full rerun version: identity20k, nongeom-light to base50, weakattack +5k, dropout-focus +5k, dropout-margin +5k, regenerated train10000 pairs, and a freshly trained RGB refiner.
- Refiner stages are A10k, B2k, B+500, targeted C500, and final dropout-balanced D500.
- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Passed full from-scratch formal rerun thresholds.
