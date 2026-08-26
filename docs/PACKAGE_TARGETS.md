# Package Targets

Chinese version: `docs/PACKAGE_TARGETS_zh.md`

## Goal

The project is being prepared toward **two separate deliverables**, not one:

1. a local folder that can be pushed to public GitHub
2. a fuller lab-facing folder that can be uploaded to the lab server

This split is intentional and matches the current handoff strategy.

## Target 1: Public GitHub Folder

Suggested target name:

`O256-github-public`

Primary purpose:

- public portfolio
- clean reproducibility story
- lightweight code repository

It should include:

- code
- configs
- scripts
- lightweight summaries
- research-history notes
- `LICENSE`
- `docs/THIRD_PARTY_AND_LICENSE_NOTES.md`
- public release checklists

It should exclude:

- datasets
- SD2.1 / VAE large files
- checkpoints
- large generated outputs
- raw attack folders

## Target 2: Lab Server Folder

Suggested target name:

`O256-lab-handoff-full`

Primary purpose:

- academic handoff
- future lab reuse
- fuller recovery package

It should include:

- everything needed from the public GitHub folder
- SD2.1 / VAE backup
- final formal summaries
- thesis audit tables
- selected historical milestone notes
- selected final checkpoints if available and worth preserving

It does not need to foreground public-facing legal packaging concerns.
Completeness is prioritized over public polish for this lab-facing version.

For alignment:

- the lab-facing version is expected to include the VAE backup
- the public GitHub version is expected to exclude the VAE backup

## Current Policy For The Lab Folder

- pre-formal checkpoints: excluded
- geometry experiment: excluded
- `speed_up_experiment`: historical side note only
- `Pixel_Bridge_Diffw`: include as front-history narrative reference
- `fidelity_tradeoff_experiment`: include as front-history narrative reference
- `paper`: include as front-history narrative reference

## Relationship Between The Two Targets

The public GitHub folder is the cleaner subset.

The lab server folder is the broader academic package.

The public version should be readable by interviewers.
The lab version should be usable by future students.
