# Task 01: Dataset Generalization Evaluation

## Goal

Evaluate the existing O256 StageE protocol/model on 3-4 common non-COCO datasets.

Do not start with full retraining.

## Suggested Datasets

| Priority | Dataset | Suggested size |
|---|---|---:|
| 1 | Kodak24 | 24 |
| 2 | BSD100 / BSDS subset | 100-200 |
| 3 | DIV2K validation | 100 |
| 4 | Flickr2K / Unsplash subset / FFHQ subset | 100-500 |

## Required Metrics

For each dataset, report:

- PSNR
- SSIM
- LPIPS
- Clean BA
- 11 formal non-geometric attacks per-attack BA
- Nongeom Avg BA
- Min BA
- Min attack
- Paper_Dropout_r03 BA
- End-to-end apply runtime if possible

## Important Rules

- Keep image size 256x256 unless the experiment explicitly states otherwise.
- Keep message length 32 bits.
- Use the same formal non-geometric attack set when comparing to thesis results.
- Do not mix Clean BA into Nongeom Avg BA.
- Do not describe non-COCO results as formal main results unless the protocol is explicitly locked.

## Output Files

Please write:

```text
summary.csv
per_attack_ba.csv
run_notes.md
```

