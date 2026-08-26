# GitHub Backup Guide

## Goal

The GitHub version should preserve the method, scripts, configs, and important result summaries. It should not store large generated data.

Use GitHub for:

- Code
- Config files
- Reproducible shell scripts
- Lightweight result summaries
- Experiment timeline and interpretation documents
- Environment and setup notes

Do not use GitHub for:

- Checkpoints
- Full generated image folders
- Attacked image folders
- Dataset copies
- LPIPS caches
- Temporary folders

## Recommended Upload Content

### Core Project Code

Upload these folders/files from `O256-PAPER`:

| Path | Reason |
|---|---|
| `models/` | diffusion/model implementation |
| `noise_layers/` | attack/noise implementation |
| `utils/` | helper code |
| `datasets/` only if it contains code, not data | dataset loader code |
| `*.py` in repo root | training/evaluation entrypoints |
| `scripts/` | experiment, summary, audit, and diagnosis scripts |
| `configs/` | reproducible experiment configs |
| `README.md` | project entry point |
| `environment.yml` | portable conda environment |
| `ENVIRONMENT.md` | environment notes |
| `.gitignore` | prevents large files from being pushed |

If `datasets/` contains real images, do not upload the image files. Keep only loader code.

### Important Experiment Scripts

Keep all scripts, but these are especially important:

| Experiment | Why Keep |
|---|---|
| actual10k / repeatability | deployment/inference cost evidence |
| DiffW-like epoch7 / 60k | key baseline comparison |
| exp23 | first evidence that base-side weak attack scheduling matters |
| exp28 | first clear Dropout-focused BA breakthrough |
| exp37 / exp38 | cross-seed continuation success of base dropout-margin |
| exp40 / exp41 | full from-scratch formal candidate runs |
| exp48 | bitplane/refiner damage diagnosis |
| exp49 | evaluation integrity audit |
| exp50 | decoder-aware refiner probe that crosses seed62 Dropout floor |
| exp51 | cross-seed StageE probe, if completed |

### Important Result Summaries

Copy lightweight summaries from `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/` into a GitHub folder such as `results_summaries/`.

Suggested minimum:

| Source experiment | Files to keep |
|---|---|
| exp40 | `summary.md`, `summary.csv`, `per_attack_ba.csv` |
| exp41 | `summary.md`, `summary.csv`, `per_attack_ba.csv` |
| exp48 | `diagnosis/summary.md`, `diagnosis/variant_summary.csv`, `diagnosis/transition_summary.csv` |
| exp49 | `audit/summary.md` |
| exp50 | `summary.md`, `summary.csv`, `per_attack_ba.csv` |
| exp51 | `summary.md`, `summary.csv`, `per_attack_ba.csv`, after it finishes |

Optional but useful:

- exp21 / exp22 trade-off summaries
- exp23 / exp27 base-side weak attack summaries
- exp28 / exp33 / exp34 candidate-development summaries
- exp44 / exp45 / exp46 / exp47 negative diagnosis summaries

### Important Narrative Documents

Copy these from `/home/jeffrey/projects/實驗結果/14_mainline_formal_protocol/` into `docs/` or `docs/experiment_notes/`:

| File | Reason |
|---|---|
| `O256_20260629之後實驗代表意義簡表_20260706.md` | quick experiment map |
| `O256_逐實驗實作方式與中文脈絡_20260707.md` | detailed implementation narrative |
| `O256_20260629之後逐實驗脈絡_20260706.md` | step-by-step experiment logic |
| `O256_20260629至20260705實驗總覽與研究脈絡.md` | broader research narrative |
| `正式與探索實驗分類規則.md` | explains which experiments are formal vs exploratory |
| `重要實驗結果保留清單_DO_NOT_DELETE_20260702.md` | preservation policy |
| `DiffW_like與O256_RGB最佳版本完整對照表_20260630.md` | baseline comparison |
| `Training_side口試答法與最終總表_20260629.md` | training-side comparison language |
| `O256_exp48_bitplane_refiner_damage_tracking規劃_20260708.md` | bitplane diagnosis plan |
| `O256_exp49_evaluation_integrity_audit規劃_20260708.md` | evaluation audit plan |
| `O256_exp50_decoderaware_refiner_probe規劃_20260708.md` | decoder-aware probe plan |
| `O256_exp51_decoderaware_stageE_crossseed_probe規劃_20260708.md` | StageE cross-seed plan |

## GitHub Folder Layout

Recommended lightweight layout:

```text
O256-PAPER/
  README.md
  environment.yml
  ENVIRONMENT.md
  .gitignore
  configs/
  models/
  noise_layers/
  utils/
  scripts/
  docs/
    GITHUB_BACKUP_GUIDE_20260708.md
    experiment_notes/
  results_summaries/
    exp40/
    exp41/
    exp48/
    exp49/
    exp50/
    exp51/
```

## TWCC / NCHC Notes

TWCC does not need a scheduler wrapper for your use case. The future TWCC package should focus on:

- `SETUP_TWCC.md`
- dataset placement instructions
- path variables or `.env` template
- smoke-test script
- one command for each formal experiment

Do not hard-code local `/home/jeffrey/projects/...` paths in the final TWCC version. Use variables such as:

```bash
PROJECT_ROOT=/work/your_project/O256-PAPER
DATA_ROOT=/work/your_dataset/CoCo_256_paper_final
RESULT_ROOT=/work/your_results
```
