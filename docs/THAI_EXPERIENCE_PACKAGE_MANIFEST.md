# Thai_experience Package Manifest

Created for handing off the O256 StageE watermarking project to new students.

This package is intentionally lightweight. It includes runnable code, configs,
scripts, summaries, thesis-ready tables, and research notes, but it does not
include large datasets, checkpoints, generated images, attack outputs, or train
pairs.

## Package Size Check

- Current package size: about 4.1M
- Number of files: about 529
- Large artifacts intentionally excluded:
  - `*.pth`, `*.pth.tar`, `*.pt`, `*.ckpt`
  - `*.png`, `*.jpg`, `*.jpeg`, `*.webp`
  - `*.npy`, `*.npz`
  - raw experiment workspaces, generated outputs, attacked images, train pairs

## Folder Map

| Folder | Purpose |
|---|---|
| `01_FINAL_O256_STAGEE_CODE/` | Main runnable code for the final O256 StageE version. This is the first place students should inspect for training and evaluation scripts. |
| `02_FINAL_RESULTS_AND_THESIS_TABLES/` | Final StageE results, exp52/exp53/exp54 summaries, per-attack BA, and thesis audit tables. |
| `03_LEGACY_EXPERIMENTS_AND_HISTORY/` | Lightweight historical record of key earlier experiments. This explains why the final protocol moved toward StageE. |
| `04_STUDENT_TASKS/` | Suggested next tasks for students, especially dataset generalization experiments. |
| `99_MISC_DOCS/` | Extra handoff notes, dependency map, external resources, and original handoff docs. |

## Recommended Entry Order

1. Read `START_HERE.md`.
2. Read `01_FINAL_O256_STAGEE_CODE/FINAL_VERSION_NOTE.md`.
3. Read `01_FINAL_O256_STAGEE_CODE/RUN_STAGEE_FROM_SCRATCH.md`.
4. Read `02_FINAL_RESULTS_AND_THESIS_TABLES/README.md`.
5. For the first student task, read `04_STUDENT_TASKS/task_01_dataset_generalization/README.md`.
6. If the student needs research context, read `03_LEGACY_EXPERIMENTS_AND_HISTORY/README.md`.

## Code Source

The final code was copied from:

`/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/`

This was chosen because it contains the latest StageE scripts, including exp52,
exp53, and exp54. The older GitHub backup was not used as the source of truth
because it did not contain the full latest StageE runner/summarizer set.

## Final StageE Scripts Included

Inside `01_FINAL_O256_STAGEE_CODE/O256-PAPER/scripts/`:

- `run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh`
- `run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh`
- `run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh`
- `summarize_o256_exp52_full_from_scratch_stageE_seed62_20260708.py`
- `summarize_o256_exp53_full_from_scratch_stageE_seed61_20260709.py`
- `summarize_o256_exp54_full_from_scratch_stageE_seed63_20260714.py`

## Final Result Status

| Experiment | Seed | Status |
|---|---:|---|
| exp52 StageE full from scratch | 62 | Final main result candidate, strict-pass |
| exp53 StageE full from scratch | 61 | Final main result candidate, strict-pass |
| exp54 StageE full from scratch | 63 | Completed validation, but should be described cautiously as near-miss unless the final thesis table explicitly supports strict pass |

## What Is Not Included

This package does not include:

- SD 2.1 model files
- COCO/finaltest1000 image data
- checkpoints
- generated watermarked images
- attack output images
- RGB refinement train pairs
- full experiment workspaces

See `DATA_NOT_INCLUDED.md` for details.
