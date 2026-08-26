# O256 exp50 decoder-aware refiner probe summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Dropout BA | Min attack | Train s | Eval/apply s | Decoder-aware setting |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| exp41_seed62_full_from_scratch_reference_rgb | reference | 34.0293 | 0.9820 | 0.0443 | 99.98 | 98.68 | 89.94 | 89.94 | Paper_Dropout_r03 | 2522.55 | 59.98 | src=exp41_stageD_dropout_balanced500, msg=1.4, atk_msg=0.45, qgate=33.0/70.0 |
| exp44_hardcase_dropout_replay_negative_reference | negative_reference | 34.5561 | 0.9839 | 0.0391 | 99.98 | 98.47 | 88.13 | 88.13 | Paper_Dropout_r03 | 124.42 | 59.64 | src=exp41_stageD_dropout_balanced500, msg=N/A, atk_msg=N/A, qgate=N/A/N/A |
| exp50A_decoderaware_stageD_from_stageC | replace_stageD | 33.7450 | 0.9810 | 0.0470 | 99.97 | 98.78 | 90.77 | 90.77 | Paper_Dropout_r03 | 311.03 | 60.60 | src=refiner_final.pth, msg=1.6, atk_msg=0.75, qgate=33.6/90.0 |
| exp50B_decoderaware_refresh_from_stageD | micro_refresh | 33.9305 | 0.9820 | 0.0446 | 99.98 | 98.72 | 90.15 | 90.15 | Paper_Dropout_r03 | 208.77 | 59.64 | src=refiner_final.pth, msg=1.5, atk_msg=0.65, qgate=33.8/90.0 |

## Exp50 Per-Attack BA


### exp50A_decoderaware_stageD_from_stageC

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 90.77 |
| Paper_GaussianBlur_sigma2 | 99.98 |
| Paper_GaussianBlur_sigma3 | 99.98 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 98.03 |
| Paper_GaussianNoise_sigma03_var009 | 99.62 |
| Paper_JPEG_Q30 | 99.77 |
| Paper_JPEG_Q40 | 99.91 |
| Paper_JPEG_Q50 | 99.93 |
| Paper_MedianFilter_w5 | 99.96 |
| Paper_SaltPepper_r02 | 98.67 |

### exp50B_decoderaware_refresh_from_stageD

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 90.15 |
| Paper_GaussianBlur_sigma2 | 99.99 |
| Paper_GaussianBlur_sigma3 | 99.99 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 97.92 |
| Paper_GaussianNoise_sigma03_var009 | 99.64 |
| Paper_JPEG_Q30 | 99.79 |
| Paper_JPEG_Q40 | 99.91 |
| Paper_JPEG_Q50 | 99.93 |
| Paper_MedianFilter_w5 | 99.98 |
| Paper_SaltPepper_r02 | 98.61 |

## Interpretation

- `exp50A_decoderaware_stageD_from_stageC` recovers Dropout but needs quality/average BA inspection: PSNR `33.7450`, Clean BA `99.97`, Avg BA `98.78`, Dropout BA `90.77`.
- `exp50B_decoderaware_refresh_from_stageD` passes the probe threshold and is worth a full from-scratch rerun: PSNR `33.9305`, Clean BA `99.98`, Avg BA `98.72`, Dropout BA `90.15`.

## Notes

- exp50 is a continuation probe, not a formal result.
- It follows exp48/49: evaluation is valid, clean BA is not the main damaged area, and the remaining issue is the Dropout-after-refiner floor.
- Branch A replaces only the final StageD from exp41 StageC; Branch B lightly refreshes the existing exp41 StageD.
- If either branch passes, the next formal step should be a full from-scratch rerun with the chosen decoder-aware StageD setting.
