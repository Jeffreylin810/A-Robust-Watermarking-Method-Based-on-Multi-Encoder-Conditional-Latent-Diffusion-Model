# O256 exp52 seed62 full from-scratch StageE formal rerun summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Dropout BA | Min attack | Base train s | Refiner train s | Eval/apply s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| exp40_seed61_formal_stageD_reference | reference | 33.9372 | 0.9819 | 0.0409 | 99.96 | 98.77 | 91.01 | 91.01 | Paper_Dropout_r03 | 13632.29 | 2527.77 | 59.65 |
| exp41_seed62_formal_stageD_reference | reference | 34.0293 | 0.9820 | 0.0443 | 99.98 | 98.68 | 89.94 | 89.94 | Paper_Dropout_r03 | 13632.29 | 2522.55 | 59.98 |
| exp51_seed62_stageE_continuation_reference | continuation_reference | 33.9524 | 0.9820 | 0.0446 | 99.98 | 98.72 | 90.16 | 90.16 | Paper_Dropout_r03 | N/A | 260.55 | 60.51 |
| exp52_base65_dropout_margin | base | 24.6601 | 0.7084 | 0.1008 | 100.00 | 98.23 | 87.10 | 87.10 | Paper_Dropout_r03 | 13635.71 | N/A | 104.82 |
| exp52_rgb_full_from_scratch_stageE | rgb_stageE | 34.0379 | 0.9825 | 0.0451 | 99.99 | 98.76 | 90.33 | 90.33 | Paper_Dropout_r03 | 13635.71 | 2789.87 | 60.14 |

## Exp52 Per-Attack BA


### exp52_base65_dropout_margin

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 87.10 |
| Paper_GaussianBlur_sigma2 | 100.00 |
| Paper_GaussianBlur_sigma3 | 100.00 |
| Paper_GaussianBlur_sigma4 | 100.00 |
| Paper_GaussianNoise_arg03 | 96.21 |
| Paper_GaussianNoise_sigma03_var009 | 99.62 |
| Paper_JPEG_Q30 | 99.95 |
| Paper_JPEG_Q40 | 99.99 |
| Paper_JPEG_Q50 | 100.00 |
| Paper_MedianFilter_w5 | 100.00 |
| Paper_SaltPepper_r02 | 97.63 |

### exp52_rgb_full_from_scratch_stageE

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 90.33 |
| Paper_GaussianBlur_sigma2 | 100.00 |
| Paper_GaussianBlur_sigma3 | 99.99 |
| Paper_GaussianBlur_sigma4 | 99.99 |
| Paper_GaussianNoise_arg03 | 98.07 |
| Paper_GaussianNoise_sigma03_var009 | 99.67 |
| Paper_JPEG_Q30 | 99.88 |
| Paper_JPEG_Q40 | 99.92 |
| Paper_JPEG_Q50 | 99.96 |
| Paper_MedianFilter_w5 | 99.99 |
| Paper_SaltPepper_r02 | 98.58 |

## Notes

- This is the seed62 full from-scratch formal rerun with StageE included in the locked refiner protocol.
- Base protocol remains identity20k, nongeom-light to base50, weakattack +5k, dropout-focus +5k, dropout-margin +5k.
- Refiner protocol is StageA image10k, StageB message2k, StageB+ message500, StageC targeted500, StageD dropout-balanced500, then StageE decoder-aware250.
- Success criterion: PSNR >= 33.8, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Passed the locked full from-scratch StageE formal thresholds.
