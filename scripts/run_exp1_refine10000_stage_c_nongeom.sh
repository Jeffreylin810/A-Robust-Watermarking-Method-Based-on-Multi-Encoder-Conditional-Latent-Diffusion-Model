#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
BASE_CKPT="Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar"
CONFIG="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"
TRAIN_WM="results/test_result/O256_FORMAL_TRAIN10000_BASE_seed61/train10000/CoCo"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61/data/CoCo/val/gt"
EVAL_WM="results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo"
STAGE_B="results/refinement/O256_RGBREFINE_FORMAL_TRAIN10000_IMGPRI_THEN_MSG0P75_2K_seed61/refiner_final.pth"
STAGE_C="results/refinement/refine10000_A10k_B2k_C500_nongeom_seed61"
APPLY="results/refinement/refine10000_A10k_B2k_C500_nongeom_apply_eval1000_seed61"
RUN="exp1_nongeom_refine10000_A10k_B2k_C500_seed61"

cd "$ROOT"
mkdir -p logs

for required in "$BASE_CKPT" "$STAGE_B" "$TRAIN_WM" "$TRAIN_GT" "$EVAL_WM"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done
for output in "$STAGE_C" "$APPLY" "results/attacked_Result/$RUN"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output"
    exit 1
  fi
done

/usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --resume_refiner "$STAGE_B" \
  --watermarked_dir "$TRAIN_WM" \
  --gt_dir "$TRAIN_GT" \
  --output_dir "$STAGE_C" \
  --image_size 256 \
  --steps 500 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1e-5 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.75 \
  --attack_message_weight 0.10 \
  --attack_profile nongeom_formal \
  --attack_noise_sigma 0.05 \
  --attack_blur_sigma 1.0 \
  --attack_jpeg_levels 32 \
  --attack_dropout_prob 0.05 \
  --attack_salt_pepper_prob 0.10 \
  --regularize_weight 0.01 \
  --eval_every 100 \
  --log_every 25 \
  --save_eval_checkpoints \
  --skip_final_dataset_images \
  --seed 61 \
  2>&1 | tee "logs/${RUN}_train.log"

"$PY" scripts/apply_rgb_refinement.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --refiner "$STAGE_C/refiner_final.pth" \
  --watermarked_dir "$EVAL_WM" \
  --gt_dir "$GT" \
  --output_dir "$APPLY" \
  --image_size 256 \
  --batch_size 4 \
  --num_workers 2 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --seed 61 \
  2>&1 | tee "logs/${RUN}_apply.log"

"$PY" calculate_psnr_ssim.py \
  --gt_path "$GT" \
  --results_path "$APPLY/refined_images" \
  2>&1 | tee "logs/${RUN}_psnr_ssim.log"

"$PY" calculate_ba.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "$APPLY/attacked_clean" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee "logs/${RUN}_clean_ba.log"

"$PY" attack_eval_diffw_paper_matched.py \
  --gt_path "$GT" \
  --results_path "$APPLY/refined_images" \
  --output_dir results/attacked_Result \
  --run_name "$RUN" \
  --attack_scope nongeom \
  --seed 61 \
  2>&1 | tee "logs/${RUN}_attack.log"

"$PY" calculate_ba.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "results/attacked_Result/$RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee "logs/${RUN}_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$RUN/ba_report.txt" \
  --output "results/attacked_Result/$RUN/ba_summary.txt"
