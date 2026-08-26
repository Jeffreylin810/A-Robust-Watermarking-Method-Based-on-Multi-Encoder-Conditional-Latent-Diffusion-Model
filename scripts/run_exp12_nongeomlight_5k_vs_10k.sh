#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
CONFIG="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k.yml"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
SEED=61

START_CKPT="Result/CoCo_ddpm_train10000_base20k_then_nongeomlight5k_seed61_final.pth.tar"
TRAIN_OUTPUT="results/exploratory_training/exp12_identity20k_then_nongeomlight10k_seed61"
SOURCE_CKPT="$TRAIN_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
FINAL_CKPT="Result/CoCo_ddpm_exp12_identity20k_then_nongeomlight10k_seed61_final.pth.tar"
LABEL="exp12_identity20k_then_nongeomlight10k_seed61"
EVAL_OUTPUT="results/test_result/$LABEL"
CLEAN_RUN="clean_$LABEL"
ATTACK_RUN="${LABEL}_nongeom"
SUMMARY_DIR="results/experiment_summaries/exp12_nongeomlight_5k_vs_10k_seed61"

cd "$ROOT"

required=(
  "$PY"
  "configs/$CONFIG"
  "configs/$EVAL_CONFIG"
  "$GT"
  "$START_CKPT"
  "logs/exp2b_nongeom_train10000_base20k_then_light5k_seed61_psnr_ssim.log"
  "results/attacked_Result/clean_exp2b_train10000_base20k_then_light5k_seed61/ba_report.txt"
  "results/attacked_Result/exp2b_nongeom_train10000_base20k_then_light5k_seed61/ba_summary.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

outputs=(
  "$TRAIN_OUTPUT"
  "$FINAL_CKPT"
  "$EVAL_OUTPUT"
  "results/attacked_Result/$CLEAN_RUN"
  "results/attacked_Result/$ATTACK_RUN"
  "$SUMMARY_DIR"
)
for path in "${outputs[@]}"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing output: $path"
    exit 1
  fi
done

if [[ "${1:-}" == "--check-only" ]]; then
  echo "Experiment 12 preflight passed."
  echo "It will continue the healthy identity20k -> light5k checkpoint for another light5k."
  echo "The final comparison is light5k (25k total) vs light10k (30k total) on dev1000."
  exit 0
fi

mkdir -p logs Result

echo "Experiment 12 classification: E (exploratory step selection)"
echo "[$(date '+%F %T')] Step 1/5: continue light curriculum from total step 25k to 30k"
/usr/bin/time -p "$PY" -u train_diffusion.py \
  --config "$CONFIG" \
  --resume "$START_CKPT" \
  --image_folder results/validation/$LABEL \
  --seed "$SEED" \
  2>&1 | tee "logs/${LABEL}_train.log"

if [[ ! -f "$SOURCE_CKPT" ]]; then
  echo "Training finished without expected checkpoint: $SOURCE_CKPT"
  exit 1
fi
mv "$SOURCE_CKPT" "$FINAL_CKPT"

echo "[$(date '+%F %T')] Step 2/5: evaluate 30k checkpoint"
"$PY" -u eval_diffusion.py \
  --config "$EVAL_CONFIG" \
  --resume "$FINAL_CKPT" \
  --image_folder "$EVAL_OUTPUT" \
  --run_name eval1000 \
  --seed "$SEED" \
  2>&1 | tee "logs/${LABEL}_eval.log"

WM="$EVAL_OUTPUT/eval1000/CoCo"
if [[ ! -d "$WM" ]]; then
  echo "Expected eval output not found: $WM"
  exit 1
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$GT" \
  --results_path "$WM" \
  2>&1 | tee "logs/${LABEL}_psnr_ssim.log"

echo "[$(date '+%F %T')] Step 3/5: calculate clean BA"
mkdir -p "results/attacked_Result/$CLEAN_RUN"
ln -s "$(realpath "$WM")" "results/attacked_Result/$CLEAN_RUN/Clean"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$FINAL_CKPT" \
  --attack_dir "results/attacked_Result/$CLEAN_RUN" \
  --gt_message_dir "$WM" \
  2>&1 | tee "logs/${LABEL}_clean_ba.log"

echo "[$(date '+%F %T')] Step 4/5: evaluate 11 non-geometric attacks"
"$PY" attack_eval_diffw_paper_matched.py \
  --gt_path "$GT" \
  --results_path "$WM" \
  --output_dir results/attacked_Result \
  --run_name "$ATTACK_RUN" \
  --attack_scope nongeom \
  --seed "$SEED" \
  2>&1 | tee "logs/${LABEL}_attack.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$FINAL_CKPT" \
  --attack_dir "results/attacked_Result/$ATTACK_RUN" \
  --gt_message_dir "$WM" \
  2>&1 | tee "logs/${LABEL}_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$ATTACK_RUN/ba_summary.txt"

echo "[$(date '+%F %T')] Step 5/5: select light5k or light10k"
"$PY" scripts/summarize_exp12_light_steps.py

echo "[$(date '+%F %T')] Experiment 12 complete"
echo "Summary: $SUMMARY_DIR/summary.md"

