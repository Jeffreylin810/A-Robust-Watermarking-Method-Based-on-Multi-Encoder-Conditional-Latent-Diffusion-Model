# O256 exp28 message-preserving attack-aware refresh500 summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| plus20k_refresh500 | rgb | 33.8299 | 0.9738 | 0.0383 | 99.39 | 95.91 | 83.36 | Paper_Dropout_r03 | 283.14s | 61.13s |
| A_weakattack_plus5k_rgb | rgb | 32.9388 | 0.9715 | 0.0483 | 99.89 | 98.21 | 89.61 | Paper_Dropout_r03 | 1109.26s | 60.40s |
| A_dropout_focus_plus5k_rgb_reuse_refresh500 | rgb | 32.5223 | 0.9705 | 0.0523 | 99.91 | 98.64 | 91.43 | Paper_Dropout_r03 | 1091.72s | 60.94s |
| exp28_quality_recovery_refresh500 | rgb | 33.6062 | 0.9731 | 0.0407 | 99.43 | 97.76 | 88.63 | Paper_Dropout_r03 | 394.14s | 61.41s |
| exp28_conservative_refresh250 | rgb | 32.5422 | 0.9699 | 0.0516 | 99.93 | 98.61 | 91.13 | Paper_Dropout_r03 | 197.15s | 61.17s |
| exp28_message_preserving_attackaware_refresh500 | rgb | 32.8792 | 0.9699 | 0.0476 | 99.92 | 98.39 | 89.70 | Paper_Dropout_r03 | 317.17s | 60.71s |

## Notes

- This architecture-oriented experiment resumes from exp28 and tests whether RGB quality recovery can preserve the message path.
- Compared with exp29, it increases message/attack-message preservation and adds latent/message-sensitive residual constraints.
- Success criterion: PSNR returns to 33+ while Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Formal-candidate rule: This result is still exploratory. Use it to decide the next architecture or loss direction, not as a final main result.
