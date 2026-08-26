# O256 exp48 bitplane x refiner-damage tracking

## What This Diagnoses

- Whether weak bit positions are stable across seeds and stages.
- Whether RGB refiner damage happens on clean decoding or mainly after Dropout.
- Whether failed Dropout images are repaired, newly damaged, or persist after refiner changes.

## Variant Summary

| Sample | Seed | Stage | Condition | Images | Avg BA | Hard <90 | Min bit | Min bit BA | Weak bits top8 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| exp40_base_clean | seed61 | base | clean | 1000 | 100.00 | 0 | 0 | 100.00 | 0 1 2 3 4 5 6 7 |
| exp40_base_dropout | seed61 | base | dropout_r03 | 1000 | 89.18 | 395 | 22 | 85.60 | 22 14 25 17 27 8 11 0 |
| exp40_rgb_clean | seed61 | rgb | clean | 1000 | 99.96 | 0 | 1 | 99.80 | 1 3 10 2 11 4 24 20 |
| exp40_rgb_dropout | seed61 | rgb | dropout_r03 | 1000 | 91.01 | 330 | 22 | 87.90 | 22 27 8 18 16 14 10 31 |
| exp41_base_clean | seed62 | base | clean | 1000 | 100.00 | 0 | 0 | 100.00 | 0 1 2 3 4 5 6 7 |
| exp41_base_dropout | seed62 | base | dropout_r03 | 1000 | 86.64 | 501 | 10 | 84.40 | 10 18 23 9 24 1 13 29 |
| exp41_rgb_clean | seed62 | rgb | clean | 1000 | 99.98 | 0 | 8 | 99.80 | 8 26 14 5 28 4 3 0 |
| exp41_rgb_dropout | seed62 | rgb | dropout_r03 | 1000 | 89.94 | 397 | 10 | 87.30 | 10 9 18 19 26 23 13 11 |
| exp44_rgb_clean | seed62 | rgb_hardcase_replay | clean | 1000 | 99.98 | 0 | 8 | 99.80 | 8 5 14 28 30 26 3 0 |
| exp44_rgb_dropout | seed62 | rgb_hardcase_replay | dropout_r03 | 1000 | 88.13 | 469 | 9 | 85.90 | 9 10 23 18 11 26 2 13 |
| exp47A_rgb_clean | seed62 | rgb_guard_mild | clean | 1000 | 99.99 | 0 | 5 | 99.90 | 5 8 26 2 3 4 6 0 |
| exp47A_rgb_dropout | seed62 | rgb_guard_mild | dropout_r03 | 1000 | 88.55 | 436 | 10 | 85.90 | 10 9 23 18 19 11 13 6 |
| exp47B_rgb_clean | seed62 | rgb_guard_strong | clean | 1000 | 99.99 | 0 | 5 | 99.90 | 5 26 2 0 3 4 6 1 |
| exp47B_rgb_dropout | seed62 | rgb_guard_strong | dropout_r03 | 1000 | 87.44 | 477 | 9 | 85.40 | 9 23 18 10 19 15 8 11 |

## Transition Summary

| Transition | Δ Avg BA | Hard <90 L/R | Recovered | New below | Weak-bit top8 overlap | Bit BA corr | Error-event J | Damaged/Repaired events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp40_base_clean_to_rgb_clean | -0.04 | 0/0 | 0 | 0 | 4 | nan | 0.000 | 12/0 |
| exp40_base_dropout_to_rgb_dropout | 1.83 | 395/330 | 106 | 41 | 4 | 0.696 | 0.600 | 499/1084 |
| exp40_base_clean_to_base_dropout | -10.82 | 0/395 | 0 | 395 | 1 | nan | 0.000 | 3461/0 |
| exp40_rgb_clean_to_rgb_dropout | -8.95 | 0/330 | 0 | 330 | 1 | -0.006 | 0.003 | 2867/3 |
| exp41_base_clean_to_rgb_clean | -0.02 | 0/0 | 0 | 0 | 4 | nan | 0.000 | 7/0 |
| exp41_base_dropout_to_rgb_dropout | 3.29 | 501/397 | 129 | 25 | 5 | 0.849 | 0.615 | 367/1421 |
| exp41_base_clean_to_base_dropout | -13.36 | 0/501 | 0 | 501 | 1 | nan | 0.000 | 4274/0 |
| exp41_rgb_clean_to_rgb_dropout | -10.04 | 0/397 | 0 | 397 | 1 | 0.082 | 0.002 | 3214/1 |
| cross_seed_base_dropout_exp40_to_exp41 | -2.54 | 395/501 | 84 | 190 | 0 | -0.019 | 0.102 | 3560/2747 |
| cross_seed_rgb_dropout_exp40_to_exp41 | -1.08 | 330/397 | 90 | 157 | 2 | 0.216 | 0.088 | 2727/2383 |
| exp41_rgb_clean_to_exp44_rgb_clean | 0.00 | 0/0 | 0 | 0 | 7 | 0.888 | 0.750 | 1/1 |
| exp41_rgb_dropout_to_exp44_rgb_dropout | -1.80 | 397/469 | 0 | 72 | 7 | 0.935 | 0.839 | 596/19 |
| exp44_rgb_clean_to_rgb_dropout | -11.84 | 0/469 | 0 | 469 | 1 | -0.026 | 0.001 | 3792/2 |
| exp41_rgb_clean_to_exp47A_rgb_clean | 0.01 | 0/0 | 0 | 0 | 6 | 0.856 | 0.429 | 0/4 |
| exp41_rgb_dropout_to_exp47A_rgb_dropout | -1.39 | 397/436 | 23 | 62 | 7 | 0.944 | 0.789 | 628/184 |
| exp47A_rgb_clean_to_rgb_dropout | -11.44 | 0/436 | 0 | 436 | 1 | -0.141 | 0.001 | 3661/0 |
| exp41_rgb_clean_to_exp47B_rgb_clean | 0.02 | 0/0 | 0 | 0 | 5 | 0.608 | 0.286 | 0/5 |
| exp41_rgb_dropout_to_exp47B_rgb_dropout | -2.50 | 397/477 | 11 | 91 | 6 | 0.923 | 0.734 | 955/156 |
| exp47B_rgb_clean_to_rgb_dropout | -12.55 | 0/477 | 0 | 477 | 0 | -0.272 | 0.000 | 4017/0 |

## Reading Guide

- For `base_clean_to_rgb_clean`, high damaged events mean the refiner breaks clean decoding.
- For `base_dropout_to_rgb_dropout`, high damaged events mean the refiner breaks attack-robust evidence.
- If weak-bit overlap across seed61/seed62 is high, bit-specific training may be a stable target.
- If weak-bit overlap is low, bitplane should stay as a diagnosis tool rather than a fixed repair rule.
- If clean transitions are safe but Dropout transitions are damaged, the next model change should be attack-aware or decoder-aware refiner design.

## Outputs

- `variant_summary.csv`
- `per_bit_summary.csv`
- `per_image_summary.csv`
- `transition_summary.csv`
- `transition_per_bit_damage.csv`
- `transition_per_image_damage.csv`
- `diagnosis_summary.json`
