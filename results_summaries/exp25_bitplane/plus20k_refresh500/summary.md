# Per-bit BA analysis

- attack_dir: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630/attacked_Result/o256_rgb_plus20k_refresh500_targeted_nongeom`
- gt_message_dir: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630/eval_outputs/o256_base_plus20k/finaltest1000/CoCo`
- weak_attacks: `Paper_Dropout_r03, Paper_GaussianNoise_arg03, Paper_SaltPepper_r02`

## Attack summary

| Attack | Images | Avg BA | Min BA | Min bit |
|---|---:|---:|---:|---:|
| Paper_Dropout_r03 | 1000 | 83.36 | 79.70 | 26 |
| Paper_GaussianBlur_sigma2 | 1000 | 99.59 | 98.40 | 19 |
| Paper_GaussianBlur_sigma3 | 1000 | 99.53 | 98.40 | 19 |
| Paper_GaussianBlur_sigma4 | 1000 | 99.52 | 98.50 | 19 |
| Paper_GaussianNoise_arg03 | 1000 | 90.32 | 86.60 | 11 |
| Paper_GaussianNoise_sigma03_var009 | 1000 | 96.35 | 93.60 | 18 |
| Paper_JPEG_Q30 | 1000 | 97.87 | 96.00 | 11 |
| Paper_JPEG_Q40 | 1000 | 98.50 | 96.40 | 11 |
| Paper_JPEG_Q50 | 1000 | 98.77 | 97.30 | 11 |
| Paper_MedianFilter_w5 | 1000 | 99.43 | 97.30 | 11 |
| Paper_SaltPepper_r02 | 1000 | 91.77 | 88.60 | 11 |

## Weakest bits from selected weak attacks

| Rank | Bit | Mean weak-attack BA |
|---:|---:|---:|
| 1 | 11 | 85.63 |
| 2 | 3 | 86.80 |
| 3 | 19 | 87.10 |
| 4 | 18 | 87.43 |
| 5 | 23 | 87.47 |
| 6 | 31 | 87.47 |
| 7 | 26 | 87.57 |
| 8 | 9 | 87.67 |
| 9 | 14 | 87.83 |
| 10 | 2 | 87.87 |
| 11 | 12 | 87.87 |
| 12 | 7 | 87.90 |
| 13 | 27 | 88.10 |
| 14 | 10 | 88.10 |
| 15 | 8 | 88.20 |
| 16 | 21 | 88.20 |

## Outputs

- `per_attack_summary.csv`
- `per_attack_per_bit_ba.csv`
- `weakest_bits.csv`
- `bit_weights.json`
