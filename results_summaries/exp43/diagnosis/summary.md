# O256 Dropout Floor Error Diagnosis

## Variant Summary

| Variant | Kind | Images | Avg BA | P05 image BA | Hard <90 | Min bit BA | Min bit | Weak bits top8 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| exp40_rgb_dropout | rgb_reference | 1000 | 91.01 | 71.88 | 330 | 87.90 | 22 | 22 27 8 18 16 14 10 31 |
| exp41_base_dropout | base | 1000 | 86.64 | 65.62 | 501 | 84.40 | 10 | 10 18 23 9 24 1 13 29 |
| exp41_rgb_dropout | rgb | 1000 | 89.94 | 71.88 | 397 | 87.30 | 10 | 10 9 18 19 26 23 13 11 |
| exp42A_base_dropout | base | 1000 | 86.92 | 68.59 | 520 | 84.60 | 9 | 9 23 8 25 24 28 5 15 |
| exp42A_rgb_dropout | rgb | 1000 | 89.80 | 71.88 | 391 | 87.50 | 28 | 28 23 25 8 27 24 18 9 |

## Comparisons

| Comparison | Δ Avg BA | Hard <90 L/R | Hard overlap J | Weak-bit overlap | Error-event J | GT msg equal | Image BA corr | Bit BA corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp41_base_dropout -> exp41_rgb_dropout | 3.29 | 501/397 | 0.707 | 5 | 0.615 | 100.0% | 0.897 | 0.849 |
| exp42A_base_dropout -> exp42A_rgb_dropout | 2.87 | 520/391 | 0.706 | 6 | 0.645 | 100.0% | 0.909 | 0.736 |
| exp41_rgb_dropout -> exp42A_rgb_dropout | -0.14 | 397/391 | 0.602 | 3 | 0.218 | 0.0% | 0.736 | 0.445 |
| exp40_rgb_dropout -> exp41_rgb_dropout | -1.08 | 330/397 | 0.493 | 2 | 0.088 | 0.0% | 0.636 | 0.216 |

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
