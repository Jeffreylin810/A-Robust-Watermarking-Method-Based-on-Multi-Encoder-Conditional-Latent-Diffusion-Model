# O256 exp47 guarded StageD refiner probe summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Dropout BA | Min attack | Train s | Eval/apply s | Guard |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| exp41_seed62_full_rerun_reference_rgb | reference | 34.0293 | 0.9820 | 0.0443 | 99.98 | 98.68 | 89.94 | 89.94 | Paper_Dropout_r03 | 2522.55 | 59.98 |  |
| exp44_hardcase_dropout_replay500 | negative_reference | 34.5561 | 0.9839 | 0.0391 | 99.98 | 98.47 | 88.13 | 88.13 | Paper_Dropout_r03 | 124.42 | 59.64 |  |
| exp47A_stageD_mild_residual_guard | mild_guard | 30.4427 | 0.9356 | 0.0834 | 99.99 | 98.51 | 88.55 | 88.55 | Paper_Dropout_r03 | 312.58 | 63.88 | lap=8.0, anti=40.0, margin=0.935 |
| exp47B_stageD_strong_residual_guard | strong_guard | 29.7203 | 0.9189 | 0.0890 | 99.99 | 98.34 | 87.44 | 87.44 | Paper_Dropout_r03 | 317.04 | 64.06 | lap=16.0, anti=80.0, margin=0.925 |

## Exp47 Per-Attack BA


### exp47A_stageD_mild_residual_guard

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 88.55 |
| Paper_GaussianBlur_sigma2 | 99.99 |
| Paper_GaussianBlur_sigma3 | 99.99 |
| Paper_GaussianBlur_sigma4 | 99.99 |
| Paper_GaussianNoise_arg03 | 97.50 |
| Paper_GaussianNoise_sigma03_var009 | 99.56 |
| Paper_JPEG_Q30 | 99.84 |
| Paper_JPEG_Q40 | 99.92 |
| Paper_JPEG_Q50 | 99.96 |
| Paper_MedianFilter_w5 | 99.99 |
| Paper_SaltPepper_r02 | 98.32 |

### exp47B_stageD_strong_residual_guard

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 87.44 |
| Paper_GaussianBlur_sigma2 | 99.99 |
| Paper_GaussianBlur_sigma3 | 99.99 |
| Paper_GaussianBlur_sigma4 | 99.99 |
| Paper_GaussianNoise_arg03 | 97.09 |
| Paper_GaussianNoise_sigma03_var009 | 99.52 |
| Paper_JPEG_Q30 | 99.78 |
| Paper_JPEG_Q40 | 99.92 |
| Paper_JPEG_Q50 | 99.96 |
| Paper_MedianFilter_w5 | 99.99 |
| Paper_SaltPepper_r02 | 98.06 |

## Interpretation

- `exp47A_stageD_mild_residual_guard` does not recover the Dropout floor: PSNR `30.4427`, Avg BA `98.51`, Dropout BA `88.55`.
- `exp47B_stageD_strong_residual_guard` does not recover the Dropout floor: PSNR `29.7203`, Avg BA `98.34`, Dropout BA `87.44`.

## Notes

- exp47 is a continuation architecture probe, not a formal result.
- Both A/B restart from exp41 StageC and retrain only the final StageD with fixed guard losses.
- If either branch passes, the next step should be a full from-scratch protocol rerun using the chosen guarded StageD.
