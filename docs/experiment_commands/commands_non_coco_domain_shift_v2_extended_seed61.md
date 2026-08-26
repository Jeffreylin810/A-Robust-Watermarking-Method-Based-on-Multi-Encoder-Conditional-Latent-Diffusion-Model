# Non-COCO Domain Shift v2 Extended Commands

Purpose: run a stronger supplementary generalization check without retraining and without overwriting the existing v1 non-COCO results.

Output root:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended
```

Datasets:

| Dataset | Count | Role |
|---|---:|---|
| NaturalMix_DIV2K_BSDS_eval1000 | 1000 | larger non-COCO natural-image generalization check |
| Kodak_eval24 | 24 | classic image-quality sanity check |
| ABO_demo_renders_eval273 | 273 | rendered-object / CG-like stress check |

The natural mixed set is composed of 800 DIV2K train images, 100 DIV2K valid images, and 100 BSDS500 images, all resized to 256x256 with center crop.

## 1. Prepare Data

This downloads DIV2K train HR and BSDS500 if they are not already present, resizes them, and creates the NaturalMix manifest. It checks for at least 20 GB free space before downloading.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/prepare_non_coco_domain_shift_v2_extended.sh
```

## 2. Preflight Only

This verifies dataset manifests, creates alias datasets/configs, and does not run inference.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_non_coco_domain_shift_v2_extended_seed61.sh --check-only
```

## 3. Full Run With 11 Non-Geometric Attacks

This runs O256 base inference, O256-RGB refinement, PSNR/SSIM, LPIPS, Clean BA, 11 non-geometric attacks, attack BA, and final summary.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_non_coco_domain_shift_v2_extended_seed61.sh
```

## 4. Quick Run Without Attacks

Use this only if time is tight. It will not produce robustness results.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_non_coco_domain_shift_v2_extended_seed61.sh --no-attacks
```

## 5. Check Progress

```bash
ps -eo pid,etime,cmd | grep -E 'run_non_coco_domain_shift_v2|eval_diffusion|apply_rgb_refinement|attack_eval|calculate_ba|calculate_lpips|calculate_psnr' | grep -v grep
```

```bash
tail -n 40 /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended/logs/*.log
```

## 6. Expected Final Files

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended/run_config.md
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended/domain_shift_v2_summary.csv
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended/domain_shift_v2_per_attack_ba.csv
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended/domain_shift_v2_summary.md
```

## Notes

- This experiment does not retrain O256 base or RGB Refinement Module.
- This experiment uses new `non_coco_v2_*` output names and does not overwrite the old `non_coco_domain_*` v1 results.
- The result should be described as supplementary cross-domain/generalization evidence, not as a replacement for locked COCO finaltest1000 main results.
