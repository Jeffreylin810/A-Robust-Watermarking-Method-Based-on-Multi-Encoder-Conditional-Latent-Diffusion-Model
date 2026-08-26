# O256 StageE Watermarking

Thesis title:

`以擴散模型潛在空間為基礎之多編碼器條件擴散強健浮水印方法`

This repository contains the lightweight runnable code and experiment summaries
for the O256 StageE robust image watermarking protocol.

The project studies robust image watermarking in Stable Diffusion 2.1 VAE latent
space. The final thesis protocol is O256 StageE, which combines:

- O256 base diffusion watermark training
- base-side weak/non-geometric attack scheduling
- RGB refinement
- decoder-aware StageE refinement
- formal finaltest1000 evaluation with 32-bit messages at 256x256 resolution

## Architecture Figure

Recommended public figure path:

```text
assets/figures/o256_stagee_overall_architecture.png
```

Once the final figure is placed there, it can be shown near the top of the
GitHub homepage.

## Thesis / Paper

Recommended public thesis path:

```text
docs/thesis/thesis.pdf
```

Optional paper or preprint path:

```text
docs/thesis/paper.pdf
```

See:

- `docs/thesis/README.md`
- `docs/REPO_PRESENTATION_LAYOUT.md`

## Recommended Starting Points

Read these files first:

- `docs/PORTFOLIO_OVERVIEW.md`
- `docs/PORTFOLIO_OVERVIEW_zh.md`
- `docs/ACADEMIC_HANDOFF_OVERVIEW.md`
- `docs/ACADEMIC_HANDOFF_OVERVIEW_zh.md`
- `docs/HISTORICAL_MILESTONES.md`
- `docs/HISTORICAL_MILESTONES_zh.md`
- `docs/PACKAGE_TARGETS.md`
- `docs/PACKAGE_TARGETS_zh.md`
- `docs/PARAMETER_ORIGIN_NOTES.md`
- `docs/PARAMETER_ORIGIN_NOTES_zh.md`
- `docs/REPO_PRESENTATION_LAYOUT.md`
- `docs/FINAL_VERSION_NOTE.md`
- `docs/RUN_STAGEE_FROM_SCRATCH.md`
- `docs/STAGEE_DEPENDENCY_MAP.md`
- `docs/STAGEE_RUN_FLOW_MANUAL.md`
- `docs/EXTERNAL_RESOURCES.md`
- `docs/THIRD_PARTY_AND_LICENSE_NOTES.md`

For the experiment history:

- `docs/research_history/O256_20260629之後實驗代表意義簡表_20260706.md`
- `docs/research_history/O256_20260629之後逐實驗脈絡_20260706.md`
- `results_summaries/stageE_thesis_audit/`

For student handoff tasks:

- `docs/student_handoff/README.md`

For public release preparation:

- `docs/PUBLIC_RELEASE_CHECKLIST.md`

## Final StageE Formal Runs

The main StageE formal scripts are:

```bash
bash scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh
bash scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh
```

Check-only mode:

```bash
bash scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh --check-only
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh --check-only
bash scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh --check-only
```

Important result summaries:

- `results_summaries/exp52/`
- `results_summaries/exp53/`
- `results_summaries/exp54/`
- `results_summaries/stageE_thesis_audit/`
- `results_summaries/stageE_final_tables/`

## What Is Not Included

This lightweight repository does not include large external resources:

- COCO image dataset
- Stable Diffusion 2.1 / VAE model files
- checkpoints
- generated watermarked images
- attacked image outputs
- RGB refinement train pairs

See `docs/EXTERNAL_RESOURCES.md` and `docs/DATA_NOT_INCLUDED.md` before trying
to rerun the formal experiments.

## Relationship to Other Local Folders

- `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER`
  - Original local working directory used during development.
- `/home/jeffrey/projects/O256-PAPER-github-backup`
  - This lightweight repository-style backup.
- `/home/jeffrey/projects/613410280_data/O256_handoff`
  - Documentation-only handoff index, not runnable code.
- `/home/jeffrey/projects/Thai_experience`
  - Lightweight student handoff package derived from the latest StageE code and
    selected summaries.
