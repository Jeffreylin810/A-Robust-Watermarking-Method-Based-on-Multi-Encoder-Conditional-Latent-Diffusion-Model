# GitHub StageE Handoff Index

This document explains how this lightweight GitHub-style backup is organized and
how it differs from the Thai student package.

## Repository Role

This folder is intended to be the lightweight, shareable code repository for the
final O256 StageE project. It should contain:

- runnable code
- configs
- one-command experiment scripts
- evaluation utilities
- final result summaries
- selected exploration-history summaries
- documentation needed to reconstruct the experiments

It should not contain:

- datasets
- checkpoints
- generated image outputs
- attacked image folders
- train-pair image folders
- temporary experiment workspaces

## Source Of Truth

The latest code was synchronized from:

```bash
/home/jeffrey/projects/Thai_experience/01_FINAL_O256_STAGEE_CODE/O256-PAPER
```

That Thai package itself was created from the latest development source:

```bash
/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
```

## Final Formal Results

| Experiment | Seed | Role | Summary path |
|---|---:|---|---|
| exp52 | 62 | StageE full-from-scratch final main result candidate | `results_summaries/exp52/` |
| exp53 | 61 | StageE full-from-scratch final main result candidate | `results_summaries/exp53/` |
| exp54 | 63 | StageE seed63 validation; describe cautiously as near-miss unless final thesis table supports strict pass | `results_summaries/exp54/` |

The thesis audit package is stored at:

```bash
results_summaries/stageE_thesis_audit/
```

The single GPT-ready thesis revision package is stored at:

```bash
results_summaries/stageE_final_tables/
```

## Exploration And Earlier Formal Results

This GitHub backup keeps earlier exploration and pre-StageE formal summaries in
`results_summaries/` and `docs/research_history/`.

The goal is not to preserve every raw file. The goal is to preserve enough
evidence and provenance to understand why the final protocol moved from the
original O256-RGB baseline toward StageD and then StageE.

Important references:

- `docs/research_history/O256_20260629之後實驗代表意義簡表_20260706.md`
- `docs/research_history/O256_20260629之後逐實驗脈絡_20260706.md`
- `docs/research_history/O256後續研究方向與高成本實驗暫存_20260713.md`

## Thai Experience Package Difference

`Thai_experience` is a student-facing lightweight package. It contains:

- the final StageE code
- exp52/53/54 summaries
- a smaller curated legacy-history folder
- first student tasks
- report templates

This GitHub backup is broader. It is the better place to keep:

- the full lightweight code repository
- more complete `results_summaries`
- historical experiment summaries
- thesis audit tables
- long-term future-work notes

## First Suggested Student Task

Do dataset generalization evaluation first, not full retraining.

Suggested datasets:

- Kodak24
- BSD100 / BSDS subset
- DIV2K validation
- one additional natural-image dataset such as Flickr2K, Unsplash subset, or FFHQ subset

The student should report:

- PSNR
- SSIM
- LPIPS
- Clean BA
- 11 formal non-geometric per-attack BA
- Nongeom Avg BA
- Min BA
- Paper_Dropout_r03 BA
- runtime

See:

```bash
docs/student_handoff/README.md
```

## Minimal Reproducibility Requirements

To rerun the final experiments, the student needs external resources listed in:

```bash
docs/EXTERNAL_RESOURCES.md
docs/DATA_NOT_INCLUDED.md
```

Without the external dataset and VAE/model resources, this repository can still
be read and audited, but the final formal experiments cannot be rerun.
