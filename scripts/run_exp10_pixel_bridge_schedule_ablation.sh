#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
STAGE1_CONFIG="CoCo_train10000_ours256_msg32_ablation_latentdirect20k.yml"
STAGE2_CONFIG="CoCo_train10000_ours256_msg32_ablation_latentdirect20k_then_bridge_nongeomlight5k.yml"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
SEED=61

STAGE1_OUTPUT="results/exploratory_training/exp10_latentdirect20k_seed61"
STAGE1_SOURCE="$STAGE1_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
STAGE1_CKPT="Result/CoCo_ddpm_exp10_latentdirect20k_seed61_final.pth.tar"

STAGE2_OUTPUT="results/exploratory_training/exp10_latentdirect20k_then_bridge_nongeomlight5k_seed61"
STAGE2_SOURCE="$STAGE2_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
STAGE2_CKPT="Result/CoCo_ddpm_exp10_latentdirect20k_then_bridge_nongeomlight5k_seed61_final.pth.tar"

cd "$ROOT"
mkdir -p logs Result

evaluate_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  local eval_output="results/test_result/$label"
  local clean_run="clean_${label}"
  local attack_run="${label}_nongeom"
  local wm="$eval_output/eval1000/CoCo"

  for output in "$eval_output" \
    "results/attacked_Result/$clean_run" \
    "results/attacked_Result/$attack_run"; do
    if [[ -e "$output" ]]; then
      echo "Refusing to overwrite existing evaluation output: $output"
      exit 1
    fi
  done

  "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$checkpoint" \
    --image_folder "$eval_output" \
    --run_name eval1000 \
    --seed "$SEED" \
    2>&1 | tee "logs/${label}_eval.log"

  if [[ ! -d "$wm" ]]; then
    echo "Expected eval output not found: $wm"
    exit 1
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$GT" \
    --results_path "$wm" \
    2>&1 | tee "logs/${label}_psnr_ssim.log"

  mkdir -p "results/attacked_Result/$clean_run"
  ln -s "$(realpath "$wm")" "results/attacked_Result/$clean_run/Clean"

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$checkpoint" \
    --attack_dir "results/attacked_Result/$clean_run" \
    --gt_message_dir "$wm" \
    2>&1 | tee "logs/${label}_clean_ba.log"

  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$GT" \
    --results_path "$wm" \
    --output_dir results/attacked_Result \
    --run_name "$attack_run" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "logs/${label}_attack.log"

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$checkpoint" \
    --attack_dir "results/attacked_Result/$attack_run" \
    --gt_message_dir "$wm" \
    2>&1 | tee "logs/${label}_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "results/attacked_Result/$attack_run/ba_report.txt" \
    --output "results/attacked_Result/$attack_run/ba_summary.txt"
}

echo "Experiment 10 classification: exploratory Pixel Bridge schedule ablation"
echo "Control already exists: bridge identity20k -> bridge nongeom-light5k"
echo "New branch: latent-direct20k -> bridge nongeom-light5k"

echo
echo "[$(date '+%F %T')] Step 1/4: train latent-direct message path for 20k"
if [[ -f "$STAGE1_CKPT" ]]; then
  echo "Reuse completed Stage 1 checkpoint: $STAGE1_CKPT"
elif [[ -e "$STAGE1_OUTPUT" ]]; then
  echo "Incomplete Stage 1 output exists: $STAGE1_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$STAGE1_CONFIG" \
    --resume ./ckpts/nonexistent_exp10_latentdirect20k_start.pth.tar \
    --image_folder results/validation/exp10_latentdirect20k_seed61 \
    --seed "$SEED" \
    2>&1 | tee logs/exp10_latentdirect20k_seed61_train.log

  if [[ ! -f "$STAGE1_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $STAGE1_SOURCE"
    exit 1
  fi
  mv "$STAGE1_SOURCE" "$STAGE1_CKPT"
fi

echo
echo "[$(date '+%F %T')] Step 2/4: evaluate latent-direct 20k after real RGB save/re-encode"
evaluate_checkpoint "exp10_latentdirect20k_seed61" "$STAGE1_CKPT"

echo
echo "[$(date '+%F %T')] Step 3/4: enable Pixel Bridge and continue with light attacks for 5k"
if [[ -f "$STAGE2_CKPT" ]]; then
  echo "Reuse completed Stage 2 checkpoint: $STAGE2_CKPT"
elif [[ -e "$STAGE2_OUTPUT" ]]; then
  echo "Incomplete Stage 2 output exists: $STAGE2_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$STAGE2_CONFIG" \
    --resume "$STAGE1_CKPT" \
    --image_folder results/validation/exp10_latentdirect20k_then_bridge_nongeomlight5k_seed61 \
    --seed "$SEED" \
    2>&1 | tee logs/exp10_latentdirect20k_then_bridge_nongeomlight5k_seed61_train.log

  if [[ ! -f "$STAGE2_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $STAGE2_SOURCE"
    exit 1
  fi
  mv "$STAGE2_SOURCE" "$STAGE2_CKPT"
fi

echo
echo "[$(date '+%F %T')] Step 4/4: evaluate final 25k branch"
evaluate_checkpoint \
  "exp10_latentdirect20k_then_bridge_nongeomlight5k_seed61" \
  "$STAGE2_CKPT"

echo
echo "[$(date '+%F %T')] Experiment 10 complete"
echo "Compare the new final branch against:"
echo "  PSNR: logs/exp2b_nongeom_train10000_base20k_then_light5k_seed61_psnr_ssim.log"
echo "  Clean BA: results/attacked_Result/clean_exp2b_train10000_base20k_then_light5k_seed61/ba_report.txt"
echo "  Nongeom BA: results/attacked_Result/exp2b_nongeom_train10000_base20k_then_light5k_seed61/ba_summary.txt"
