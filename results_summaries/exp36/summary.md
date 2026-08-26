# O256 exp36 native dropout proxy refresh500 summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| exp33_dropout_balanced_candidate | rgb | 33.2346 | 0.9744 | 0.0457 | 99.89 | 98.41 | 90.05 | Paper_Dropout_r03 | 309.46s | 60.04s |
| exp34_full_candidate_validation_rgb | rgb | 33.2333 | 0.9741 | 0.0458 | 99.90 | 98.35 | 89.72 | Paper_Dropout_r03 | 2325.12s | 60.40s |
| exp35_dropout_compensation_refresh150 | rgb | 33.3079 | 0.9748 | 0.0455 | 99.89 | 98.35 | 89.74 | Paper_Dropout_r03 | 239.93s | 59.68s |
| exp36_native_dropout_proxy_refresh500 | rgb | 33.2246 | 0.9744 | 0.0457 | 99.90 | 98.36 | 89.75 | Paper_Dropout_r03 | 313.37s | 59.81s |

## Notes

- This experiment tests a theory-level correction: aligning the training dropout proxy with the native formal Dropout(0.3) direction.
- It starts from the exp34 regenerated base and uses exp33-style dropout-balanced refiner settings, but with `dropout_r03_native_only`.
- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Proxy alignment did not pass all thresholds; stop refiner-loss micro-tuning.
