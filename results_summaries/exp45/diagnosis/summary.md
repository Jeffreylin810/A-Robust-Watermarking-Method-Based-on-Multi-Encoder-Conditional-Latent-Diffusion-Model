# O256 Dropout Floor Error Diagnosis

## Variant Summary

| Variant | Kind | Images | Avg BA | P05 image BA | Hard <90 | Min bit BA | Min bit | Weak bits top8 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| exp41_base_dropout | base | 1000 | 86.64 | 65.62 | 501 | 84.40 | 10 | 10 18 23 9 24 1 13 29 |
| exp41_rgb_dropout | rgb_reference | 1000 | 89.94 | 71.88 | 397 | 87.30 | 10 | 10 9 18 19 26 23 13 11 |
| exp44_rgb_dropout | rgb_hardcase_replay | 1000 | 88.13 | 68.75 | 469 | 85.90 | 9 | 9 10 23 18 11 26 2 13 |

## Comparisons

| Comparison | Δ Avg BA | Hard <90 L/R | Hard overlap J | Weak-bit overlap | Error-event J | GT msg equal | Image BA corr | Bit BA corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp41_base_dropout -> exp41_rgb_dropout | 3.29 | 501/397 | 0.707 | 5 | 0.615 | 100.0% | 0.897 | 0.849 |
| exp41_rgb_dropout -> exp44_rgb_dropout | -1.80 | 397/469 | 0.846 | 7 | 0.839 | 100.0% | 0.965 | 0.935 |
| exp41_base_dropout -> exp44_rgb_dropout | 1.49 | 501/469 | 0.773 | 5 | 0.674 | 100.0% | 0.912 | 0.801 |

## Reading Guide

- High hard-image overlap means the same images are repeatedly vulnerable under Dropout r03.
- High weak-bit overlap means the floor is concentrated in specific message bit positions.
- High error-event Jaccard means two variants are making the same bit errors; low Jaccard means the failure is more stochastic or redistributed.
- If exp41 RGB and exp42A RGB have high overlap but exp42A does not improve Min BA, continuing the same margin direction is unlikely to help.

## Outputs

- `variant_summary.csv`
- `per_bit_ba.csv`
- `per_image_ba.csv`
- `worst_images.csv`
- `comparisons.csv`
- `diagnosis_summary.json`
