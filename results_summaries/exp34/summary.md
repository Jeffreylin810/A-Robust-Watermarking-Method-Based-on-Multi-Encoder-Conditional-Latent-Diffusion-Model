# O256 exp34 full candidate validation summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply/Eval time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| exp28_original_candidate_base_rgb | rgb | 32.5223 | 0.9705 | 0.0523 | 99.91 | 98.64 | 91.43 | Paper_Dropout_r03 | 1091.72s | 60.94s |
| exp33_dropout_balanced_candidate | rgb | 33.2346 | 0.9744 | 0.0457 | 99.89 | 98.41 | 90.05 | Paper_Dropout_r03 | 309.46s | 60.04s |
| exp34_full_candidate_dropout_focus_base | base | 24.6518 | 0.7079 | 0.0981 | 100.00 | 97.93 | 87.28 | Paper_Dropout_r03 | 1088.54s | 99.20s |
| exp34_full_candidate_dropout_balanced_rgb | rgb | 33.2333 | 0.9741 | 0.0458 | 99.90 | 98.35 | 89.72 | Paper_Dropout_r03 | 2325.12s | 60.40s |

## Notes

- This validation reruns the candidate path from the A/exp23 checkpoint: dropout-focus base continuation, train-pair generation, dropout-balanced refiner training, and full finaltest1000 nongeom evaluation.
- It is stricter than exp33 because the exp28-like base and refiner train pairs are regenerated in a fresh experiment root.
- Candidate status: Did not pass all formal-candidate thresholds; keep as exploratory validation.
