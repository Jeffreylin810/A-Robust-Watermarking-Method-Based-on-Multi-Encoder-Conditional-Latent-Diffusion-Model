# O256 exp54 seed63 full from-scratch StageE formal rerun summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Dropout BA | Min attack | Base train s | Refiner train s | Eval/apply s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| exp40_seed61_formal_stageD_reference | reference | 33.9372 | 0.9819 | 0.0409 | 99.96 | 98.77 | 91.01 | 91.01 | Paper_Dropout_r03 | 13632.29 | 2527.77 | 59.65 |
| exp53_seed61_formal_stageE_reference | reference | 33.9522 | 0.9825 | 0.0400 | 99.85 | 98.58 | 91.36 | 91.36 | Paper_Dropout_r03 | 13711.58 | 2829.02 | 60.92 |
| exp52_seed62_formal_stageE_reference | reference | 34.0379 | 0.9825 | 0.0451 | 99.99 | 98.76 | 90.33 | 90.33 | Paper_Dropout_r03 | 13635.71 | 2789.87 | 60.14 |
| exp54_base65_dropout_margin | base | 24.6000 | 0.7081 | 0.0989 | 100.00 | 98.21 | 88.86 | 88.86 | Paper_Dropout_r03 | 13527.84 | N/A | 100.95 |
| exp54_rgb_full_from_scratch_stageE | rgb_stageE | 33.4948 | 0.9815 | 0.0447 | 99.94 | 98.35 | 89.98 | 89.98 | Paper_Dropout_r03 | 13527.84 | 2781.41 | 59.40 |

## Exp53 Per-Attack BA


### exp54_base65_dropout_margin

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 88.86 |
| Paper_GaussianBlur_sigma2 | 99.99 |
| Paper_GaussianBlur_sigma3 | 99.98 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 95.30 |
| Paper_GaussianNoise_sigma03_var009 | 99.35 |
| Paper_JPEG_Q30 | 99.95 |
| Paper_JPEG_Q40 | 100.00 |
| Paper_JPEG_Q50 | 99.99 |
| Paper_MedianFilter_w5 | 100.00 |
| Paper_SaltPepper_r02 | 96.88 |

### exp54_rgb_full_from_scratch_stageE

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 89.98 |
| Paper_GaussianBlur_sigma2 | 99.95 |
| Paper_GaussianBlur_sigma3 | 99.95 |
| Paper_GaussianBlur_sigma4 | 99.95 |
| Paper_GaussianNoise_arg03 | 96.33 |
| Paper_GaussianNoise_sigma03_var009 | 99.18 |
| Paper_JPEG_Q30 | 99.55 |
| Paper_JPEG_Q40 | 99.74 |
| Paper_JPEG_Q50 | 99.82 |
| Paper_MedianFilter_w5 | 99.93 |
| Paper_SaltPepper_r02 | 97.45 |

## Notes

- This is the seed63 full from-scratch formal rerun with StageE included in the locked refiner protocol.
- Base protocol remains identity20k, nongeom-light to base50, weakattack +5k, dropout-focus +5k, dropout-margin +5k.
- Refiner protocol is StageA image10k, StageB message2k, StageB+ message500, StageC targeted500, StageD dropout-balanced500, then StageE decoder-aware250.
- Success criterion: PSNR >= 33.8, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Did not pass the locked full from-scratch StageE formal thresholds.
