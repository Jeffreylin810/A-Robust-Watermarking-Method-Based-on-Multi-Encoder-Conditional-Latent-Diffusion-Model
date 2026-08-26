# Commands: M07-clean same-COCO alternate-train10000 module reuse

## Goal

This is the **new** M07 design.

It supersedes the old `first5000 / second5000` split experiment and instead
tests a cleaner question:

> Can the formal seed61 RGB refinement module, originally produced from the
> `base-train10000 -> refine10000 pairs` route, be reused on a **different
> COCO train10000 base checkpoint** without retraining the module?

In short, this is a `10000 + 10000` same-COCO-family reuse validation, not a
`5000 / 5000` split validation.

## Classification

- Supporting experiment
- Not a chapter-4 main result
- Intended to support or reject a cautious module-amortization discussion

## Important note

Do **not** use the old script below for the new thesis discussion:

- `scripts/run_m07_coco_subset_module_reuse.sh`

That older script corresponds to the deprecated `first5000 / second5000`
design.

Use the new script instead:

- `scripts/run_m07_clean_same_coco_alt10000_module_reuse.sh`

## Check-only

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m07_clean_same_coco_alt10000_module_reuse.sh --check-only
```

Expected behavior:

- build an alternate COCO train10000 filename list from the larger raw COCO
  train2017 pool
- exclude the formal train10000 manifest
- create a same-image alias dataset for train/val
- generate alternate-train10000 configs
- stop before training

## Full run

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m07_clean_same_coco_alt10000_module_reuse.sh
```

## Main output

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/04_cross_seed/module_reuse_same_coco_alt10000_seed61/summary.md
```

## Notes

- This experiment **does train a new alternate-train10000 base checkpoint**.
- It **does not retrain the RGB refinement module**.
- The transferred module remains:
  - `results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth`
- Evaluation still uses:
  - locked `finaltest1000`
  - the same formal nongeom attack suite
  - the same final metrics as the chapter-4 main line
