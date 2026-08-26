# Academic Handoff Overview

Chinese version: `docs/ACADEMIC_HANDOFF_OVERVIEW_zh.md`

This document is the academic-facing entry point for lab handoff and future
follow-up work.

## Purpose

The handoff package has two goals:

1. Preserve the final StageE protocol and its formal evidence.
2. Preserve enough historical reasoning that future researchers can recover if
   the current mainline stops being productive.

This is not intended to preserve every exploratory branch. It is intended to
preserve the branches that changed the research direction.

## Core Handoff Boundary

The handoff keeps:

- final StageE code and configs
- final formal summaries for exp52/53/54
- thesis audit and thesis-ready tables
- selected earlier milestones that explain why the protocol looks the way it
  does
- path/setup notes for recreating the environment

The handoff does not keep:

- every raw experiment folder
- all temporary checkpoints
- every generated image artifact
- every exploratory branch from the pre-formal period

## Milestone Policy

The academic handoff uses three levels of importance.

### 1. Core milestones

These are the main line that should remain easy to read and revisit:

- `actual10k`, `repeatability`
- `diffw_epoch7`, `diffw_60k`
- `exp16` to `exp54`, with special focus on:
  - `exp23`
  - `exp28`
  - `exp33` / `exp34`
  - `exp37` / `exp38`
  - `exp40` / `exp41`
  - `exp43`
  - `exp46`
  - `exp50` / `exp51`
  - `exp52` / `exp53` / `exp54`

### 2. Front-history narrative only

These older branches are valuable for explaining why the later protocol uses
256x256 images, 32-bit messages, and specific training ideas, but they do not
need checkpoint preservation in the NAS package:

- `/home/jeffrey/projects/Pixel_Bridge_Diffw`
- `/home/jeffrey/projects/fidelity_tradeoff_experiment`
- `/home/jeffrey/projects/paper`

### 3. Front-history note candidates

These are useful as archive references, but are not part of the core handoff:

- `/home/jeffrey/projects/Pixel_Bridge_Diffw_text_archive_20260516`

These may be cited in notes if a specific diagnostic becomes important later.

## Folder-Level Decisions From Discussion

- `/home/jeffrey/projects/geometry_experiment`
  - Excluded from the main handoff narrative.
- `/home/jeffrey/projects/speed_up_experiment`
  - Not treated as a direct ancestor of the final StageE method.
  - It influenced earlier paper-side and efficiency framing, but not the final
    StageE protocol definition.

## NAS Packaging Policy

### Minimal NAS package

Keep:

- SD2.1 / VAE backup
- final formal summaries
- thesis audit tables
- selected historical summaries
- key setup documents

### Full NAS package

Keep everything in the minimal package, plus:

- selected final checkpoints if they are still available and worth retaining
- selected StageE-related support artifacts
- additional curated milestone summaries

Pre-formal checkpoints are intentionally excluded unless a future decision
changes this policy.

The VAE backup is treated as a required lab-handoff asset, not an optional
public-release artifact.

## Recommended First Recovery Path

If future researchers are blocked, do not immediately jump to full StageE
retraining. Recover context in this order:

1. Read `docs/HISTORICAL_MILESTONES.md`
2. Read `docs/research_history/`
3. Revisit `exp43` to `exp46` diagnosis summaries
4. Revisit `exp50` / `exp51` StageE-design evidence
5. Only then decide whether to modify base-side scheduling, refiner behavior,
   or evaluation scope
