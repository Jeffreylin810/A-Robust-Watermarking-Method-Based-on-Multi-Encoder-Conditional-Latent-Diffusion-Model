# O256 exp44 train-side hard-case dropout replay summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train s | Eval/apply s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| exp40_seed61_full_rerun_reference_rgb | reference | 33.9372 | 0.9819 | 0.0409 | 99.96 | 98.77 | 91.01 | Paper_Dropout_r03 | N/A | 59.65 |
| exp41_seed62_full_rerun_reference_rgb | reference | 34.0293 | 0.9820 | 0.0443 | 99.98 | 98.68 | 89.94 | Paper_Dropout_r03 | N/A | 59.98 |
| exp42A_dropout_margin_probe_rgb | reference | 34.0741 | 0.9824 | 0.0441 | 100.00 | 98.73 | 89.80 | Paper_Dropout_r03 | 2533.70 | 59.75 |
| exp44_hardcase_dropout_replay500 | rgb | 34.5561 | 0.9839 | 0.0391 | 99.98 | 98.47 | 88.13 | Paper_Dropout_r03 | 124.42 | 59.64 |

## RGB Per-Attack BA

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 88.13 |
| Paper_GaussianBlur_sigma2 | 99.99 |
| Paper_GaussianBlur_sigma3 | 99.99 |
| Paper_GaussianBlur_sigma4 | 99.99 |
| Paper_GaussianNoise_arg03 | 97.57 |
| Paper_GaussianNoise_sigma03_var009 | 99.57 |
| Paper_JPEG_Q30 | 99.78 |
| Paper_JPEG_Q40 | 99.87 |
| Paper_JPEG_Q50 | 99.93 |
| Paper_MedianFilter_w5 | 99.98 |
| Paper_SaltPepper_r02 | 98.33 |

## Train-Side Hardcase Selection

- processed_images: `10000`
- below_threshold_images: `5147`
- selected_images: `1000`
- selected_avg_ba: `65.29`
- selected_ba_range: `40.62` to `71.88`

## Notes

- exp44 is a continuation probe, not a formal result.
- It selects hard cases from train10000 only, not from finaltest1000.
- It resumes the exp41 StageD refiner and runs a short Dropout r03 replay on the train-side hard subset.
- Success criterion: PSNR >= 33.5, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Hard-case replay did not pass the probe threshold; treat as diagnostic/negative evidence.
