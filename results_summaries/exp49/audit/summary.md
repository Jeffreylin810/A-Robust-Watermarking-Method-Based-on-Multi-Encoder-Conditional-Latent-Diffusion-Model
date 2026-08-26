# O256 exp49 evaluation integrity audit

## Purpose

- Verify PSNR is evaluated by matching output image stems to locked GT image stems.
- Compare old sorted-order PSNR pairing against explicit stem-matched PSNR.
- Run BA negative controls by intentionally pairing decoded messages with shifted wrong messages.
- Check whether high BA disappears under wrong-message controls.

## Audit Table

| Sample | Kind | Count | Stem OK | PSNR stem | PSNR sorted-stem diff | PSNR wrong GT | BA correct | BA wrong msg | BA gap |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| exp40_base_clean | base_clean | 1000 | True | 24.6354 | 0.000000 | 8.7091 | 100.00 | 49.69 | 50.31 |
| exp40_rgb_clean | rgb_clean | 1000 | True | 33.9372 | 0.000000 | 8.6370 | 99.96 | 49.69 | 50.27 |
| exp40_rgb_dropout | rgb_dropout | 1000 | True | 39.0235 | 0.000000 | 8.6052 | 91.01 | 49.81 | 41.21 |
| exp41_base_clean | base_clean | 1000 | True | 24.6709 | 0.000000 | 8.7127 | 100.00 | 49.56 | 50.44 |
| exp41_rgb_clean | rgb_clean | 1000 | True | 34.0293 | 0.000000 | 8.6361 | 99.98 | 49.57 | 50.41 |
| exp41_rgb_dropout | rgb_dropout | 1000 | True | 39.0991 | 0.000000 | 8.6050 | 89.94 | 49.66 | 40.27 |

## Reading Guide

- `PSNR sorted-stem diff` should be near 0. If not, sorted-order pairing is unsafe.
- `PSNR wrong GT` should be much lower than the correct PSNR.
- `BA wrong msg` should be near random-message accuracy, roughly around 50%.
- A large `BA gap` means the reported BA depends on the correct message pairing, not a trivial decoder bias.

## Outputs

- `evaluation_integrity_audit.csv`
- `evaluation_integrity_audit.json`
