# Historical Milestones

Chinese version: `docs/HISTORICAL_MILESTONES_zh.md`

This document defines which historical branches matter for the public portfolio
story and which ones matter mainly for academic handoff.

## Decision Summary

The project history is split into four layers:

1. Front history: pre-O256/StageE exploratory roots
2. Early paper-side standardization
3. Main O256 milestone line after 2026-06-29
4. Final StageE formal reruns and thesis audit

## Was The Final Version Influenced By `speed_up_experiment`?

Short answer: **indirectly yes, directly no**.

- `speed_up_experiment` influenced earlier paper-side thinking about training
  budget, step matching, and efficiency framing.
- It appears in older paper-side notes and result bookkeeping.
- It does **not** appear to define the final O256 StageE protocol itself.
- The final StageE design is better understood as descending from the later
  O256 trade-off / dropout-margin / StageE line, not from the speed-up branch.

Therefore:

- mention it only as a historical side note if needed
- do not treat it as a core milestone in the final handoff

## Front History That Should Be Mentioned

### `/home/jeffrey/projects/Pixel_Bridge_Diffw`

Why it matters:

- captures the transition from original DiffW assumptions toward latent-native
  thinking
- contains early diagnostic reasoning about weak latent watermark signals
- helps explain why later work moved away from naive latent adaptation

Use in handoff:

- academic handoff: yes
- public portfolio: brief mention only

### `/home/jeffrey/projects/fidelity_tradeoff_experiment`

Why it matters:

- records early curriculum-style attempts to recover image fidelity while
  keeping watermark decoding useful
- helps explain why later protocols cared so much about balanced trade-offs
- provides early evidence for the importance of staged scheduling

Use in handoff:

- academic handoff: yes
- public portfolio: short historical note

### `/home/jeffrey/projects/paper`

Why it matters:

- represents an intermediate paper-oriented cleanup stage
- helps explain how splits, evaluation framing, and summary conventions became
  more standardized
- is part of the bridge between exploratory work and more disciplined paper-side
  protocols

Use in handoff:

- academic handoff: yes
- public portfolio: short historical note

## Front History That Should Only Be Footnoted

### `/home/jeffrey/projects/Pixel_Bridge_Diffw_text_archive_20260516`

Why it matters:

- contains valuable early text archives and diagnostics
- includes early exploration around message length, injection, bitplane, and
  decoder behavior

Why it is not core:

- it is more archive-like than milestone-like
- too many branches are preserved at once
- it is better used as a note source than as a main handoff package target

### `/home/jeffrey/projects/geometry_experiment`

Decision:

- excluded from the main portfolio and academic handoff story

Reason:

- it was not selected as a necessary turning point for the final StageE line

## Mainline Milestones After 2026-06-29

These are the main milestones that should remain visible.

### Baseline and scope-setting

- `actual10k`
- `repeatability`
- `diffw_epoch7`
- `diffw_60k`

### O256 trade-off formation

- `exp16` to `exp23`
- especially `exp23` as the first strong base-side weak-attack signal

### Dropout-focused protocol formation

- `exp28`
- `exp33`
- `exp34`
- `exp36`
- `exp37`
- `exp38`

### Formalization and diagnosis

- `exp40`
- `exp41`
- `exp43`
- `exp44`
- `exp45`
- `exp46`

### StageE design and final validation

- `exp50`
- `exp51`
- `exp52`
- `exp53`
- `exp54`

## Why 256x256 And 32-Bit Messages?

The older exploratory folders should be referenced mainly to explain decisions
like:

- why the final line settled on 256x256 inputs
- why 32-bit messages became the practical target
- why later protocols favored selected non-geometric pressure over broader
  unfocused attack mixes
- why some earlier parameter choices were abandoned

These should be documented as historical motivation, not as final evidence.
