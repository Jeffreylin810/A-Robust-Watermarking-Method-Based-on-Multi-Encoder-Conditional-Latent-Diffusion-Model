# Formal RGB Refinement Train5000 Stage-B Scale-Up

Purpose:

```text
Scale the current best formal RGB-refinement setting from train2000 to train5000.
This keeps the successful Stage A -> Stage B recipe unchanged and evaluates on
the same unseen eval1000 split.

Primary question:
Can the RGB refiner still keep PSNR > 30 and clean BA around 88% when trained
on a larger formal train subset?
```

Run from:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
```

## 1. Create train5000 alias dataset

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/create_coco_split_alias.py \
  --source_root /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen \
  --source_split train \
  --output_root results/formal_refine_datasets/CoCo256_train5000_as_val_seed61 \
  --num_images 5000 \
  --overwrite_links
```

## 2. Generate O256 base watermarked outputs on train5000

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u eval_diffusion.py \
  --config CoCo_refiner_train5000_as_val_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --image_folder results/test_result/O256_FORMAL_TRAIN5000_BASE_seed61 \
  --run_name train5000 \
  --seed 61 \
  2>&1 | tee logs/O256_FORMAL_TRAIN5000_BASE_seed61_eval.log
```

Expected watermarked train output:

```text
results/test_result/O256_FORMAL_TRAIN5000_BASE_seed61/train5000/CoCo
```

## 3. Stage A: image-priority RGB refinement

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --watermarked_dir results/test_result/O256_FORMAL_TRAIN5000_BASE_seed61/train5000/CoCo \
  --gt_dir results/formal_refine_datasets/CoCo256_train5000_as_val_seed61/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_M0P2_5K_seed61 \
  --image_size 256 \
  --steps 5000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1e-4 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.2 \
  --regularize_weight 0.01 \
  --eval_every 500 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_M0P2_5K_seed61_train.log
```

## 4. Stage B: message recovery

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --resume_refiner results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_M0P2_5K_seed61/refiner_final.pth \
  --watermarked_dir results/test_result/O256_FORMAL_TRAIN5000_BASE_seed61/train5000/CoCo \
  --gt_dir results/formal_refine_datasets/CoCo256_train5000_as_val_seed61/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_THEN_MSG0P75_1K_seed61 \
  --image_size 256 \
  --steps 1000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 5e-5 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.75 \
  --regularize_weight 0.01 \
  --eval_every 250 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_THEN_MSG0P75_1K_seed61_train.log
```

## 5. Apply Stage B refiner to unseen eval1000

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python scripts/apply_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --refiner results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_THEN_MSG0P75_1K_seed61/refiner_final.pth \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61 \
  --image_size 256 \
  --batch_size 4 \
  --num_workers 2 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_apply.log
```

## 6. Evaluate unseen eval1000 PSNR / SSIM

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61/refined_images \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_psnr_ssim.log
```

## 7. Evaluate unseen eval1000 clean BA

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61/attacked_clean \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_clean_ba.log
```

## 8. Optional paper-matched attack evaluation

Run this only if:

```text
PSNR >= 30
Clean BA >= 88%
```

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61/refined_images \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_paper_matched_ba.log
```

## Decision rule

If Stage B train5000 gives:

```text
PSNR >= 30
Clean BA >= 88%
Paper-matched avg BA >= 65%
```

then this becomes the next formal RGB refinement anchor.

If clean BA drops below train2000 Stage B, do not immediately add attack-aware
training. First inspect whether larger train data changed the Stage A / Stage B
balance, then tune message_weight lightly.
