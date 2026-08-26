# Train10000 RGB Refinement Paper-Matched BA

Purpose:

```text
Run paper-matched attack BA for the train10000 RGB refinement result.
This is a diagnostic override: clean BA is 88.76%, slightly below the automatic 89% gate,
but the result is close enough that we need paper-matched BA to decide whether train10000
can replace train5000 as the final RGB refinement anchor.
```

Run from:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
```

## 1. Paper-Matched Attack Generation

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61/refined_images \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_paper_matched_attack.log
```

Expected attacked output:

```text
results/attacked_Result/paper_matched_O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61
```

## 2. Calculate BA

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_paper_matched_ba.log
```

Expected BA report:

```text
results/attacked_Result/paper_matched_O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61/ba_report.txt
```

## Decision Rule

Compare against the current train5000 Stage B anchor:

| Version | PSNR | SSIM | Clean BA | Paper-matched Avg BA | Non-geometry Focus Avg |
|---|---:|---:|---:|---:|---:|
| train5000 Stage B | 30.1107 | 0.9116 | 89.12% | 65.39% | 70.72% |
| train10000 Stage B | 31.4216 | 0.9221 | 88.76% | pending | pending |

Use train10000 as the final anchor only if:

```text
Paper-matched Avg BA >= 65.39%
and Non-geometry Focus Avg >= 70.72%
```

If paper-matched BA is worse than train5000, keep train5000 as the main result and report train10000 as COCO scale-up / generalization evidence.
