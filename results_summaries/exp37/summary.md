# O256 exp37 base-side dropout-margin plus1epoch summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply/Eval time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| exp33_dropout_balanced_candidate | rgb | 33.2346 | 0.9744 | 0.0457 | 99.89 | 98.41 | 90.05 | Paper_Dropout_r03 | 309.46s | 60.04s |
| exp34_full_candidate_validation_rgb | rgb | 33.2333 | 0.9741 | 0.0458 | 99.90 | 98.35 | 89.72 | Paper_Dropout_r03 | 2325.12s | 60.40s |
| exp36_native_dropout_proxy_refresh500 | rgb | 33.2246 | 0.9744 | 0.0457 | 99.90 | 98.36 | 89.75 | Paper_Dropout_r03 | 313.37s | 59.81s |
| exp37_base_dropout_margin_plus1epoch | base | 24.6238 | 0.7077 | 0.1003 | 100.00 | 98.31 | 88.43 | Paper_Dropout_r03 | 1081.42s | 101.93s |
| exp37_rgb_dropout_margin_plus1epoch_balanced_refiner | rgb | 33.1895 | 0.9748 | 0.0457 | 99.91 | 98.58 | 90.11 | Paper_Dropout_r03 | 312.06s | 60.84s |

## Notes

- This experiment stops refiner-loss micro-tuning and tests a base-side hypothesis: the final Dropout r03 gap may require stronger base watermark margin.
- The base starts from exp34 and runs one extra epoch with `identity_nongeom_dropout_margin`.
- The RGB stage uses the fixed exp33 dropout-balanced refiner settings.
- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Passed formal-candidate thresholds after base-side dropout-margin continuation.
