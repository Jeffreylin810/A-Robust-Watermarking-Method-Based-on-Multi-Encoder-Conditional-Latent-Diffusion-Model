# O256 exp38 base-side dropout-margin seed62 validation summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply/Eval time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| exp34_full_candidate_validation_rgb_seed61 | rgb | 33.2333 | 0.9741 | 0.0458 | 99.90 | 98.35 | 89.72 | Paper_Dropout_r03 | 2325.12s | 60.40s |
| exp37_base_dropout_margin_plus1epoch_seed61 | rgb | 33.1895 | 0.9748 | 0.0457 | 99.91 | 98.58 | 90.11 | Paper_Dropout_r03 | 312.06s | 60.84s |
| exp38_base_dropout_margin_seed62 | base | 24.6189 | 0.7074 | 0.1007 | 100.00 | 98.36 | 88.72 | Paper_Dropout_r03 | 1079.07s | 100.94s |
| exp38_rgb_dropout_margin_seed62_balanced_refiner | rgb | 33.1252 | 0.9745 | 0.0467 | 99.93 | 98.60 | 90.31 | Paper_Dropout_r03 | 306.47s | 59.46s |

## Notes

- This experiment validates exp37 with seed62 while keeping the same base-side dropout-margin idea and fixed dropout-balanced RGB refiner settings.
- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Passed seed62 stability validation thresholds.
