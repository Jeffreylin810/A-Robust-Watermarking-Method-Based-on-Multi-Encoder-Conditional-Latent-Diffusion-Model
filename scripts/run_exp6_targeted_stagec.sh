#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

BASE_CKPT="Result/CoCo_ddpm_train10000_base20k_then_nongeomlight5k_seed61_final.pth.tar"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_WM="results/test_result/refine10000_nongeomlight_base_pairs_seed61/train10000/CoCo"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61/data/CoCo/val/gt"
EVAL_WM="results/test_result/train10000_base20k_then_nongeomlight5k_seed61/eval1000/CoCo"
EVAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
START_REFINER="results/refinement/refine10000_nongeomlightbase_BplusM1P00_500_seed61/refiner_final.pth"

STAGE_C="results/refinement/refine10000_targetedformal_C500_seed61"
SUMMARY_DIR="results/experiment_summaries/exp6_targetedformal_stagec_seed61"
SUMMARY="$SUMMARY_DIR/summary.md"

cd "$ROOT"
mkdir -p logs

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$BASE_CKPT" "configs/$EVAL_CONFIG" "$TRAIN_WM" "$TRAIN_GT" \
  "$EVAL_WM" "$EVAL_GT" "$START_REFINER"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

train_png_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
train_msg_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
eval_png_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
eval_msg_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
if [[ "$train_png_count" -ne 10000 || "$train_msg_count" -ne 10000 ]]; then
  echo "Expected refine10000 pairs, found images=$train_png_count messages=$train_msg_count"
  exit 1
fi
if [[ "$eval_png_count" -ne 1000 || "$eval_msg_count" -ne 1000 ]]; then
  echo "Expected eval1000 images/messages, found images=$eval_png_count messages=$eval_msg_count"
  exit 1
fi

if [[ -f "$SUMMARY" ]]; then
  echo "Experiment 6 is already complete:"
  echo "$SUMMARY"
  exit 0
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  stage_c_state="clear"
  if [[ -e "$STAGE_C" ]]; then
    stage_c_state="existing"
  fi
  echo "Experiment 6 preflight passed."
  echo "Starting checkpoint: $START_REFINER"
  echo "Train pairs: $train_png_count images / $train_msg_count messages"
  echo "Eval: $eval_png_count images / $eval_msg_count messages"
  echo "Stage C: 500 steps, lr=1e-5, attack_message_weight=0.15"
  echo "Formal-scale RGB proxies:"
  echo "  GN arg03 sigma in [0,1]: sqrt(0.3)/2 = 0.2738612788"
  echo "  GN sigma03 sigma in [0,1]: 0.3/2 = 0.15"
  echo "  Dropout r03: keep 30%, replace 70%"
  echo "  SaltPepper r02: probability 0.2"
  echo "Stage C state: $stage_c_state"
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

echo "[$(date '+%F %T')] Start Experiment 6"
echo "One targeted Stage C run; evaluate checkpoints at 250 and 500 steps."

echo
echo "[$(date '+%F %T')] Step 1/4: train targeted Stage C"
if [[ -e "$STAGE_C" ]]; then
  for required in "$STAGE_C/refiner_step000250.pth" "$STAGE_C/refiner_final.pth" \
    "$STAGE_C/report.txt"; do
    if [[ ! -f "$required" ]]; then
      echo "Incomplete existing Stage C output: missing $required"
      echo "Remove only this incomplete directory before retrying: $STAGE_C"
      exit 1
    fi
  done
  echo "Reuse completed targeted Stage C checkpoints."
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --resume_refiner "$START_REFINER" \
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
    --message_weight 1.0 \
    --attack_message_weight 0.15 \
    --attack_profile targeted_formal_weak \
    --attack_noise_sigma 0.2738612788 \
    --attack_noise_sigma_secondary 0.15 \
    --attack_dropout_prob 0.70 \
    --attack_salt_pepper_prob 0.20 \
    --regularize_weight 0.01 \
    --eval_every 250 \
    --log_every 50 \
    --save_eval_checkpoints \
    --skip_final_dataset_images \
    --seed 61 \
    2>&1 | tee logs/exp6_refine10000_targetedformal_C500_seed61_train.log
fi

evaluate_candidate() {
  local tag="$1"
  local checkpoint="$2"
  local prefix="exp6_refine10000_targetedformal_C${tag}_seed61"
  local apply_dir="results/refinement/refine10000_targetedformal_C${tag}_apply_eval1000_seed61"
  local attack_run="exp6_nongeom_refine10000_targetedformal_C${tag}_seed61"

  echo
  echo "[$(date '+%F %T')] Evaluate Stage C-$tag"

  if [[ -e "$apply_dir" ]]; then
    local apply_png_count
    local apply_msg_count
    apply_png_count="$(find "$apply_dir/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
    apply_msg_count="$(find "$apply_dir/refined_images" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
    if [[ ! -f "$apply_dir/report.txt" || "$apply_png_count" -ne 1000 || \
      "$apply_msg_count" -ne 1000 || ! -d "$apply_dir/attacked_clean/Clean" ]]; then
      echo "Incomplete existing apply output: $apply_dir"
      echo "Remove only this incomplete directory before retrying."
      exit 1
    fi
    echo "Reuse completed unseen eval1000 apply output."
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$EVAL_CONFIG" \
      --resume "$BASE_CKPT" \
      --refiner "$checkpoint" \
      --watermarked_dir "$EVAL_WM" \
      --gt_dir "$EVAL_GT" \
      --output_dir "$apply_dir" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed 61 \
      2>&1 | tee "logs/${prefix}_apply.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$EVAL_GT" \
    --results_path "$apply_dir/refined_images" \
    2>&1 | tee "logs/${prefix}_psnr_ssim.log"

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$apply_dir/attacked_clean" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${prefix}_clean_ba.log"

  if [[ -e "results/attacked_Result/$attack_run" ]]; then
    local attack_dir_count
    attack_dir_count="$(find "results/attacked_Result/$attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_dir_count" -ne 11 ]]; then
      echo "Incomplete existing attack output: directories=$attack_dir_count"
      echo "Remove only this incomplete directory before retrying."
      exit 1
    fi
    echo "Reuse completed 11 non-geometric attack outputs."
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$EVAL_GT" \
      --results_path "$apply_dir/refined_images" \
      --output_dir results/attacked_Result \
      --run_name "$attack_run" \
      --attack_scope nongeom \
      --seed 61 \
      2>&1 | tee "logs/${prefix}_attack.log"
  fi

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "results/attacked_Result/$attack_run" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${prefix}_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "results/attacked_Result/$attack_run/ba_report.txt" \
    --output "results/attacked_Result/$attack_run/ba_summary.txt"
}

echo
echo "[$(date '+%F %T')] Step 2/4: evaluate checkpoint 250"
evaluate_candidate "250" "$STAGE_C/refiner_step000250.pth"

echo
echo "[$(date '+%F %T')] Step 3/4: evaluate checkpoint 500"
evaluate_candidate "500" "$STAGE_C/refiner_final.pth"

echo
echo "[$(date '+%F %T')] Step 4/4: build Experiment 6 comparison"
mkdir -p "$SUMMARY_DIR"
"$PY" scripts/summarize_exp6_targeted_stagec.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] Experiment 6 complete"
echo "Summary: $SUMMARY"
