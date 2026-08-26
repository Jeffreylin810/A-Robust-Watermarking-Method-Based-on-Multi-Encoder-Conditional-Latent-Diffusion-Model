# O256 exp46 residual/frequency diagnosis

## Main Question

Does exp44 improve image quality by smoothing or cancelling residual evidence that exp41 preserved under Dropout?

## Overall

| Count | Mean delta BA | Mean delta PSNR | Mean delta Lap ratio | Mean delta cancellation cosine | Mean exp44-exp41 delta L1 |
|---:|---:|---:|---:|---:|---:|
| 1000 | -1.8031 | 0.5267 | -0.002210 | 0.008867 | 0.002922 |

## Group Summary

| Group | Count | Mean delta BA | Mean delta PSNR | Mean delta Lap ratio | Mean delta cancellation cosine | Mean exp44-exp41 delta L1 |
|---|---:|---:|---:|---:|---:|---:|
| both_ge_threshold | 531 | -0.9416 | 0.4690 | -0.002404 | 0.007910 | 0.002733 |
| newly_below_threshold | 72 | -5.5990 | 0.5651 | -0.002160 | 0.009270 | 0.002992 |
| persistent_below_threshold | 397 | -2.2670 | 0.5970 | -0.001961 | 0.010075 | 0.003161 |
| all | 1000 | -1.8031 | 0.5267 | -0.002210 | 0.008867 | 0.002922 |
| improved_1bitplus | 10 | 3.1250 | 0.5046 | -0.002040 | 0.008396 | 0.002693 |
| degraded_1bitplus | 419 | -4.3780 | 0.5634 | -0.002060 | 0.009758 | 0.003052 |

## Top Correlations With Delta BA

| Feature | Pearson r |
|---|---:|
| exp44_psnr | 0.2417 |
| delta_psnr | -0.2401 |
| replay_delta_l2 | -0.2078 |
| replay_delta_l1 | -0.1950 |
| delta_res_l2_ratio_to_base | 0.1485 |
| delta_res_cos_base_res | -0.1190 |
| delta_lap_ratio_to_base | -0.0827 |
| delta_cancel_cos | -0.0782 |
| exp44_lap_ratio_to_base | 0.0613 |
| delta_fft_high_power_ratio_to_base | -0.0505 |

## Reading Guide

- `delta_ba` is exp44 Dropout BA minus exp41 Dropout BA; negative means exp44 is worse.
- `delta_lap_ratio_to_base` measures whether exp44 keeps more or less high-frequency residual than exp41, normalized by base residual.
- `delta_cancel_cos` measures whether exp44's refiner update is more aligned with cancelling base residual than exp41.
- `replay_delta_l1` is the direct exp44-vs-exp41 image change magnitude.
- If newly failed images show BA loss while PSNR rises and residual/frequency preservation drops, the refiner is likely smoothing away watermark evidence.

## Outputs

- `per_image_residual_frequency.csv`
- `group_summary.csv`
- `correlations.csv`
- `figures/delta_lap_ratio_vs_delta_ba.png`
- `figures/delta_cancel_cos_vs_delta_ba.png`
- `figures/replay_delta_l1_vs_delta_ba.png`
- `figures/delta_psnr_vs_delta_ba.png`
- `figures/group_delta_ba_psnr_bar.png`
