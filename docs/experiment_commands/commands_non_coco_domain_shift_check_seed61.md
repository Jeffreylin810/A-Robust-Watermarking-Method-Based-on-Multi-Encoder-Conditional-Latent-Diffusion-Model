# Non-COCO Domain Shift Check Commands

Purpose:

- Evaluate the already trained O256-Base and O256-RGBREFINE pipeline on non-COCO / rendered input images.
- Do not retrain O256-Base.
- Do not retrain RGB Refinement Module.
- Do not overwrite formal Chapter 4 main results.

Prepared image pools:

| Dataset | Role | Images | Source root |
|---|---|---:|---|
| DIV2K_valid_eval100 | non-COCO natural images | 100 | `/home/jeffrey/projects/DataSet/DIV2K_valid` |
| Kodak_eval24 | classic image-quality sanity set | 24 | `/home/jeffrey/projects/DataSet/Kodak` |
| ABO_demo_renders_eval100 | CG/rendered stress check | 100 | `/home/jeffrey/projects/DataSet/ABO_demo_renders` |

Output root:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61
```

## Preflight Only

Run this first. It prepares symlink alias datasets and generated configs, but does not run inference.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_non_coco_domain_shift_check_seed61.sh --check-only
```

## Quick Evaluation

This runs base/refined inference, PSNR/SSIM, LPIPS, and clean BA only.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_non_coco_domain_shift_check_seed61.sh
```

## Full Evaluation With Non-Geometric Attacks

Run this only if the quick evaluation looks stable and you want formal attack robustness on the non-COCO / rendered subsets.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_non_coco_domain_shift_check_seed61.sh --with-attacks
```

## Recommended Use In Thesis

Treat this as a supporting / limitation-aware experiment, not as a replacement for the locked COCO finaltest1000 main results.

Safe wording:

> The trained O256-RGB pipeline was additionally evaluated on non-COCO natural images and rendered product images without additional training, to probe test-time domain-shift behavior.

Avoid:

- universal transfer
- cross-dataset training generalization
- cross-model transfer
- replacing the formal COCO finaltest1000 main table
