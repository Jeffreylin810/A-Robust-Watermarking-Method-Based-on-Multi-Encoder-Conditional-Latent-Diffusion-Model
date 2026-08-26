# Run StageE From Scratch

Use this only after external resources are restored.

```bash
cd 01_FINAL_O256_STAGEE_CODE/O256-PAPER
```

Check paths and disk requirements first:

```bash
bash scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh --check-only
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh --check-only
bash scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh --check-only
```

Full runs:

```bash
bash scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh
bash scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh
```

Expected interpretation:

- EXP52 seed62: strict-pass final result.
- EXP53 seed61: strict-pass final result.
- EXP54 seed63: near-miss validation in the original run.

Do not compare new runs to thesis numbers unless dataset split, attack set, message length, image resolution, and evaluation scripts are identical.

