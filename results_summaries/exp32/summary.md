# O256 exp28 dropout-specific refresh500 summary

## Main Table

| Variant | Kind | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | Train time | Apply time |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| exp28 | rgb | 32.5223 | 0.9705 | 0.0523 | 99.91 | 98.64 | 91.43 | Paper_Dropout_r03 | 1091.72s | 60.94s |
| exp29_quality_recovery_refresh500 | rgb | 33.6062 | 0.9731 | 0.0407 | 99.43 | 97.76 | 88.63 | Paper_Dropout_r03 | 394.14s | 61.41s |
| exp30_conservative_refresh250 | rgb | 32.5422 | 0.9699 | 0.0516 | 99.93 | 98.61 | 91.13 | Paper_Dropout_r03 | 197.15s | 61.17s |
| exp31_message_preserving_attackaware_refresh500 | rgb | 32.8792 | 0.9699 | 0.0476 | 99.92 | 98.39 | 89.70 | Paper_Dropout_r03 | 317.17s | 60.71s |
| exp28_dropout_specific_refresh500 | rgb | 32.0162 | 0.9656 | 0.0589 | 99.92 | 98.86 | 92.87 | Paper_Dropout_r03 | 311.23s | 60.51s |

## Notes

- This experiment isolates the current bottleneck: Paper_Dropout_r03.
- It keeps message/latent preservation but uses a dropout-r03-only training proxy instead of averaging over several weak attacks.
- Success criterion: PSNR >= 33.0, Clean BA >= 99%, Avg BA >= 98%, and Min BA >= 90%.
- Candidate status: Exploratory: use this to decide the next architecture/loss setting.
