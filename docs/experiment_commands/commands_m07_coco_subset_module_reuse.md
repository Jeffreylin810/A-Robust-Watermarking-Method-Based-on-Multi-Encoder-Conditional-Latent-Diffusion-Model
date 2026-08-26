# Commands: M07 COCO subset module reuse

## Goal

Train two new COCO-only base checkpoints on disjoint train-image subsets, then
test whether the existing formal seed61 RGB refinement module can be reused
without retraining the module.

## Classification

- Supporting experiment
- Not a chapter-4 main result
- Used to evaluate whether module cost may be discussed as a COCO-family
  amortized preparation cost

## Check-only

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m07_coco_subset_module_reuse.sh --check-only
```

Expected behavior:

- create two train5000 alias datasets
- generate subset-specific configs
- stop before training

## Full run

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m07_coco_subset_module_reuse.sh
```

## Main output

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/04_cross_seed/module_reuse_same_coco_seed61/summary.md
```

## Notes

- This experiment **does train new base checkpoints**.
- It **does not retrain the RGB refinement module**.
- It is intended to answer whether the same formal module can be reused across
  different COCO-trained checkpoints under the same recipe.
