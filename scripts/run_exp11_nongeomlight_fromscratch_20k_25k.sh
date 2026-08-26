#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"
CONFIG20="CoCo_train10000_ours256_msg32_nongeomlight_fromscratch20k.yml"
CONFIG25="CoCo_train10000_ours256_msg32_nongeomlight_fromscratch20k_then25k.yml"
SEED=61

OUT20="results/exploratory_training/exp11_nongeomlight_fromscratch20k_seed61"
SRC20="$OUT20/ckpts/CoCo_ddpm_final.pth.tar"
CKPT20="Result/CoCo_ddpm_exp11_nongeomlight_fromscratch20k_seed61_final.pth.tar"

OUT25="results/exploratory_training/exp11_nongeomlight_fromscratch25k_seed61"
SRC25="$OUT25/ckpts/CoCo_ddpm_final.pth.tar"
CKPT25="Result/CoCo_ddpm_exp11_nongeomlight_fromscratch25k_seed61_final.pth.tar"

CONTROL_PREFIX="exp2b_nongeom_train10000_base20k_then_light5k_seed61"
SUMMARY_DIR="results/experiment_summaries/exp11_nongeomlight_curriculum_seed61"

cd "$ROOT"

required_inputs=(
  "$PY"
  "configs/$EVAL_CONFIG"
  "configs/$CONFIG20"
  "configs/$CONFIG25"
  "$GT"
  "logs/${CONTROL_PREFIX}_psnr_ssim.log"
  "results/attacked_Result/clean_${CONTROL_PREFIX/exp2b_nongeom_/exp2b_}/ba_report.txt"
  "results/attacked_Result/$CONTROL_PREFIX/ba_summary.json"
)

for required in "${required_inputs[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

evaluate_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  local eval_output="results/test_result/$label"
  local clean_run="clean_${label}"
  local attack_run="${label}_nongeom"
  local wm="$eval_output/eval1000/CoCo"

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

outputs=(
  "$OUT20"
  "$CKPT20"
  "results/test_result/exp11_nongeomlight_fromscratch20k_seed61"
  "results/attacked_Result/clean_exp11_nongeomlight_fromscratch20k_seed61"
  "results/attacked_Result/exp11_nongeomlight_fromscratch20k_seed61_nongeom"
  "$OUT25"
  "$CKPT25"
  "results/test_result/exp11_nongeomlight_fromscratch25k_seed61"
  "results/attacked_Result/clean_exp11_nongeomlight_fromscratch25k_seed61"
  "results/attacked_Result/exp11_nongeomlight_fromscratch25k_seed61_nongeom"
  "$SUMMARY_DIR"
)

for output in "${outputs[@]}"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output"
    exit 1
  fi
done

if [[ "${1:-}" == "--check-only" ]]; then
  echo "Experiment 11 preflight passed."
  echo "This is an exploratory dev1000 experiment; finaltest1000 is not used."
  echo "It will train light-from-scratch 20k, evaluate it, continue to 25k, and compare both with the existing staged control."
  exit 0
fi

mkdir -p logs Result

echo "Experiment 11 classification: E (exploratory curriculum selection)"
echo "[$(date '+%F %T')] Step 1/5: train Pixel Bridge + nongeom-light from scratch to 20k"
/usr/bin/time -p "$PY" -u train_diffusion.py \
  --config "$CONFIG20" \
  --resume ./ckpts/nonexistent_exp11_light20k_start.pth.tar \
  --image_folder results/validation/exp11_nongeomlight_fromscratch20k_seed61 \
  --seed "$SEED" \
  2>&1 | tee logs/exp11_nongeomlight_fromscratch20k_seed61_train.log

if [[ ! -f "$SRC20" ]]; then
  echo "Training finished without expected checkpoint: $SRC20"
  exit 1
fi
mv "$SRC20" "$CKPT20"

echo "[$(date '+%F %T')] Step 2/5: evaluate 20k candidate"
evaluate_checkpoint "exp11_nongeomlight_fromscratch20k_seed61" "$CKPT20"

echo "[$(date '+%F %T')] Step 3/5: continue the same light curriculum from 20k to 25k"
/usr/bin/time -p "$PY" -u train_diffusion.py \
  --config "$CONFIG25" \
  --resume "$CKPT20" \
  --image_folder results/validation/exp11_nongeomlight_fromscratch25k_seed61 \
  --seed "$SEED" \
  2>&1 | tee logs/exp11_nongeomlight_fromscratch25k_seed61_train.log

if [[ ! -f "$SRC25" ]]; then
  echo "Training finished without expected checkpoint: $SRC25"
  exit 1
fi
mv "$SRC25" "$CKPT25"

echo "[$(date '+%F %T')] Step 4/5: evaluate 25k candidate"
evaluate_checkpoint "exp11_nongeomlight_fromscratch25k_seed61" "$CKPT25"

echo "[$(date '+%F %T')] Step 5/5: summarize curriculum decision"
"$PY" scripts/summarize_exp11_curriculum.py

echo "[$(date '+%F %T')] Experiment 11 complete"
echo "Summary: $SUMMARY_DIR/summary.md"

