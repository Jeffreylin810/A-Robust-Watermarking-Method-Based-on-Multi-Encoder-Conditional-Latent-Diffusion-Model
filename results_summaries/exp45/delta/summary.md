# O256 exp45 Dropout Floor Delta Diagnosis

## Main Question

Does exp44 hardcase replay recover the exp41 seed62 Dropout floor, or does it trade robustness for image quality?

## Delta Summary

| Left | Right | Common | Avg BA L/R | Δ Avg BA | Min BA L/R | Hard<90 L/R | Recovered | Newly <90 | Persistent <90 | Improved >=1 bit | Degraded >=1 bit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp41_rgb_dropout | exp44_rgb_dropout | 1000 | 89.94/88.13 | -1.80 | 40.62/37.50 | 397/469 | 0 | 72 | 397 | 10 | 419 |

## Quality Delta

| Left PSNR | Right PSNR | Δ PSNR |
|---:|---:|---:|
| 34.0293 | 34.5561 | 0.5267 |

## Top Recovered Images

| Image | BA L | BA R | Δ BA | Err L | Err R | Transition |
|---|---:|---:|---:|---:|---:|---|
| N/A |  |  |  |  |  |  |

## Top Newly Failed Images

| Image | BA L | BA R | Δ BA | Err L | Err R | Transition |
|---|---:|---:|---:|---:|---:|---|
| 000000205020 | 96.88 | 81.25 | -15.62 | 1 | 6 | newly_below_threshold |
| 000000231687 | 90.62 | 78.12 | -12.50 | 3 | 7 | newly_below_threshold |
| 000000269955 | 93.75 | 81.25 | -12.50 | 2 | 6 | newly_below_threshold |
| 000000323761 | 93.75 | 81.25 | -12.50 | 2 | 6 | newly_below_threshold |
| 000000563732 | 96.88 | 84.38 | -12.50 | 1 | 5 | newly_below_threshold |
| 000000007188 | 96.88 | 87.50 | -9.38 | 1 | 4 | newly_below_threshold |
| 000000076516 | 90.62 | 81.25 | -9.38 | 3 | 6 | newly_below_threshold |
| 000000102790 | 93.75 | 84.38 | -9.38 | 2 | 5 | newly_below_threshold |
| 000000137978 | 93.75 | 84.38 | -9.38 | 2 | 5 | newly_below_threshold |
| 000000147692 | 90.62 | 81.25 | -9.38 | 3 | 6 | newly_below_threshold |

## Top Degraded Images

| Image | BA L | BA R | Δ BA | Err L | Err R | Transition |
|---|---:|---:|---:|---:|---:|---|
| 000000205020 | 96.88 | 81.25 | -15.62 | 1 | 6 | newly_below_threshold |
| 000000267716 | 75.00 | 59.38 | -15.62 | 8 | 13 | persistent_below_threshold |
| 000000075238 | 87.50 | 75.00 | -12.50 | 4 | 8 | persistent_below_threshold |
| 000000231687 | 90.62 | 78.12 | -12.50 | 3 | 7 | newly_below_threshold |
| 000000269955 | 93.75 | 81.25 | -12.50 | 2 | 6 | newly_below_threshold |
| 000000323761 | 93.75 | 81.25 | -12.50 | 2 | 6 | newly_below_threshold |
| 000000414328 | 84.38 | 71.88 | -12.50 | 5 | 9 | persistent_below_threshold |
| 000000474581 | 78.12 | 65.62 | -12.50 | 7 | 11 | persistent_below_threshold |
| 000000563732 | 96.88 | 84.38 | -12.50 | 1 | 5 | newly_below_threshold |
| 000000007188 | 96.88 | 87.50 | -9.38 | 1 | 4 | newly_below_threshold |

## Interpretation Guide

- If `newly_below_threshold` is larger than `recovered_to_ge_threshold`, hardcase replay is hurting the finaltest Dropout floor more than it helps.
- If PSNR improves while BA delta is negative, the replay is acting like quality recovery rather than robustness recovery.
- If many images are persistent below 90, the bottleneck is not a small set of easy-to-replay cases.

## Outputs

- `per_image_delta.csv`
- `transition_summary.csv`
- `top_recovered.csv`
- `top_newly_failed.csv`
- `persistent_hard.csv`
- `top_improved.csv`
- `top_degraded.csv`
- `figures/exp41_vs_exp44_dropout_ba_scatter.png`
- `figures/exp44_minus_exp41_delta_ba_hist.png`
- `figures/delta_psnr_vs_delta_ba.png`
