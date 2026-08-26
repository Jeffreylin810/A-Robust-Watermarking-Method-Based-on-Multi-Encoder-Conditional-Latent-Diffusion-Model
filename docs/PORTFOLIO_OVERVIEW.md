# Portfolio Overview

Chinese version: `docs/PORTFOLIO_OVERVIEW_zh.md`

This document is the public-facing entry point for readers who want to
understand the project as a research and engineering portfolio piece.

## Project Summary

This repository presents the final lightweight code and curated experiment
record for an image watermarking research project built on Stable Diffusion 2.1
VAE latent space.

The final protocol is **O256 StageE**:

- 256x256 images
- 32-bit messages
- base-side weak/non-geometric attack scheduling
- RGB refinement
- decoder-aware StageE refinement
- formal finaltest1000 evaluation

The goal was not only to maximize one metric, but to find a repeatable balance
between:

- image fidelity
- clean decoding accuracy
- worst-case robustness under non-geometric attacks
- deployment practicality

## What This Portfolio Demonstrates

- End-to-end experiment design rather than isolated model training.
- Careful handling of negative results instead of hiding failed branches.
- Transition from early exploratory prototypes to a locked formal protocol.
- Reproducible code, configs, runner scripts, and summary artifacts.
- Honest reporting of both successful seeds and near-miss validation.

## Recommended Reading Order

Start here if you want the shortest path:

1. `README.md`
2. `docs/HISTORICAL_MILESTONES.md`
3. `results_summaries/stageE_thesis_audit/stageE_thesis_audit_report.md`
4. `results_summaries/exp52/summary.md`
5. `results_summaries/exp53/summary.md`
6. `results_summaries/exp54/summary.md`

## Main Result Narrative

The final public story should be read conservatively:

- `exp52` and `exp53` are the strongest final main results.
- `exp54` is valuable validation, but should be described as a seed63
  near-miss rather than a strict-pass main result.
- The project should be framed as a study of trade-offs and protocol design,
  not as a claim of universal superiority over all baselines.

## What Is Included

This repository intentionally keeps:

- runnable code
- experiment configs
- formal runner scripts
- lightweight summaries
- research-history documents
- thesis audit tables

It intentionally excludes large artifacts such as datasets, checkpoints, and
generated image folders.

## What Interviewers Should Notice

- The method evolved through multiple controlled comparisons, not one lucky run.
- Several branches were stopped because they improved one dimension while
  harming another.
- The final protocol was revalidated with full from-scratch reruns instead of
  being justified only by continuation checkpoints.
- The repository preserves both the final answer and the reasoning process that
  led there.
