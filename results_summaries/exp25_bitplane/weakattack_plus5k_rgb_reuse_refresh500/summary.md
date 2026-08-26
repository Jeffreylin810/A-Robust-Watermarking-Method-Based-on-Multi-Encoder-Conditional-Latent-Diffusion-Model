# Per-bit BA analysis

- attack_dir: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/23_supplementary_o256_base_plus20k_weakattack_plus5k_seed61_20260630/attacked_Result/o256_rgb_plus20k_weakattack_plus5k_reuse_refresh500_nongeom`
- gt_message_dir: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/23_supplementary_o256_base_plus20k_weakattack_plus5k_seed61_20260630/eval_outputs/o256_base_plus20k_weakattack_plus5k/finaltest1000/CoCo`
- weak_attacks: `Paper_Dropout_r03, Paper_GaussianNoise_arg03, Paper_SaltPepper_r02`

## Attack summary

| Attack | Images | Avg BA | Min BA | Min bit |
|---|---:|---:|---:|---:|
| Paper_Dropout_r03 | 1000 | 89.61 | 87.50 | 9 |
| Paper_GaussianBlur_sigma2 | 1000 | 99.95 | 99.80 | 3 |
| Paper_GaussianBlur_sigma3 | 1000 | 99.95 | 99.80 | 3 |
| Paper_GaussianBlur_sigma4 | 1000 | 99.94 | 99.70 | 11 |
| Paper_GaussianNoise_arg03 | 1000 | 95.94 | 94.00 | 0 |
| Paper_GaussianNoise_sigma03_var009 | 1000 | 99.02 | 97.80 | 18 |
| Paper_JPEG_Q30 | 1000 | 99.58 | 99.00 | 3 |
| Paper_JPEG_Q40 | 1000 | 99.75 | 99.40 | 3 |
| Paper_JPEG_Q50 | 1000 | 99.81 | 99.40 | 9 |
| Paper_MedianFilter_w5 | 1000 | 99.91 | 99.70 | 25 |
| Paper_SaltPepper_r02 | 1000 | 96.80 | 94.10 | 11 |

## Weakest bits from selected weak attacks

| Rank | Bit | Mean weak-attack BA |
|---:|---:|---:|
| 1 | 11 | 92.67 |
| 2 | 9 | 92.97 |
| 3 | 5 | 93.03 |
| 4 | 23 | 93.27 |
| 5 | 2 | 93.37 |
| 6 | 3 | 93.47 |
| 7 | 10 | 93.50 |
| 8 | 31 | 93.53 |
| 9 | 0 | 93.60 |
| 10 | 18 | 93.70 |
| 11 | 15 | 93.73 |
| 12 | 6 | 93.93 |
| 13 | 19 | 93.97 |
| 14 | 17 | 94.07 |
| 15 | 14 | 94.10 |
| 16 | 4 | 94.10 |

## Outputs

- `per_attack_summary.csv`
- `per_attack_per_bit_ba.csv`
- `weakest_bits.csv`
- `bit_weights.json`
