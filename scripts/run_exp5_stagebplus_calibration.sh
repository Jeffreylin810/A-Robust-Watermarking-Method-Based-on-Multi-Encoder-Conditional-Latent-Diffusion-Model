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
START_REFINER="results/refinement/refine10000_nongeomlightbase_A10k_B2k_seed61/refiner_final.pth"

SUMMARY_DIR="results/experiment_summaries/exp5_stagebplus_calibration_seed61"
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
  echo "Experiment 5 is already complete:"
  echo "$SUMMARY"
  exit 0
fi

candidate_state() {
  local stage_dir="$1"
  local apply_dir="$2"
  local attack_run="$3"
  local state="clear"

  if [[ -e "$stage_dir" || -e "$apply_dir" || -e "results/attacked_Result/$attack_run" ]]; then
    state="partial-or-complete"
  fi
  printf '%s' "$state"
}

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Experiment 5 preflight passed."
  echo "Starting checkpoint: $START_REFINER"
  echo "Train pairs: $train_png_count images / $train_msg_count messages"
  echo "Eval: $eval_png_count images / $eval_msg_count messages"
  echo "Candidate 5A: message_weight=0.90, 500 steps, lr=2e-5"
  echo "Candidate 5B: message_weight=1.00, 500 steps, lr=2e-5"
  echo "5A state: $(candidate_state \
    results/refinement/refine10000_nongeomlightbase_BplusM0P90_500_seed61 \
    results/refinement/refine10000_nongeomlightbase_BplusM0P90_500_apply_eval1000_seed61 \
    exp5A_nongeom_refine10000_nongeomlightbase_BplusM0P90_500_seed61)"
  echo "5B state: $(candidate_state \
    results/refinement/refine10000_nongeomlightbase_BplusM1P00_500_seed61 \
    results/refinement/refine10000_nongeomlightbase_BplusM1P00_500_apply_eval1000_seed61 \
    exp5B_nongeom_refine10000_nongeomlightbase_BplusM1P00_500_seed61)"
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

run_candidate() {
  local label="$1"
  local weight="$2"
  local weight_tag="$3"

  local run="exp5${label}_refine10000_nongeomlightbase_BplusM${weight_tag}_500_seed61"
  local stage_dir="results/refinement/refine10000_nongeomlightbase_BplusM${weight_tag}_500_seed61"
  local apply_dir="results/refinement/refine10000_nongeomlightbase_BplusM${weight_tag}_500_apply_eval1000_seed61"
  local attack_run="exp5${label}_nongeom_refine10000_nongeomlightbase_BplusM${weight_tag}_500_seed61"

  echo
  echo "[$(date '+%F %T')] Candidate 5${label}: Stage B+ message_weight=$weight"

  echo "Step 1/5: train 500-step Stage B+"
  if [[ -e "$stage_dir" ]]; then
    if [[ ! -f "$stage_dir/refiner_final.pth" || ! -f "$stage_dir/report.txt" ]]; then
      echo "Incomplete existing Stage B+ output: $stage_dir"
      echo "Remove only this incomplete directory before retrying."
      exit 1
    fi
    echo "Reuse completed Stage B+ checkpoint."
  else
    /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
      --config "$EVAL_CONFIG" \
      --resume "$BASE_CKPT" \
      --resume_refiner "$START_REFINER" \
      --watermarked_dir "$TRAIN_WM" \
      --gt_dir "$TRAIN_GT" \
      --output_dir "$stage_dir" \
      --image_size 256 \
      --steps 500 \
      --batch_size 4 \
      --num_workers 2 \
      --lr 2e-5 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --image_mse_weight 10.0 \
      --image_l1_weight 2.5 \
      --message_weight "$weight" \
      --regularize_weight 0.01 \
      --eval_every 250 \
      --log_every 50 \
      --skip_final_dataset_images \
      --seed 61 \
      2>&1 | tee "logs/${run}_train.log"
  fi

  echo "Step 2/5: apply to fixed unseen eval1000"
  if [[ -e "$apply_dir" ]]; then
    local apply_png_count
    local apply_msg_count
    apply_png_count="$(find "$apply_dir/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
    apply_msg_count="$(find "$apply_dir/refined_images" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
    if [[ ! -f "$apply_dir/report.txt" || "$apply_png_count" -ne 1000 || \
      "$apply_msg_count" -ne 1000 || ! -d "$apply_dir/attacked_clean/Clean" ]]; then
      echo "Incomplete existing eval1000 apply output: $apply_dir"
      echo "Remove only this incomplete directory before retrying."
      exit 1
    fi
    echo "Reuse completed unseen eval1000 apply output."
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$EVAL_CONFIG" \
      --resume "$BASE_CKPT" \
      --refiner "$stage_dir/refiner_final.pth" \
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
      2>&1 | tee "logs/${run}_apply.log"
  fi

  echo "Step 3/5: PSNR / SSIM and Clean BA"
  "$PY" calculate_psnr_ssim.py \
    --gt_path "$EVAL_GT" \
    --results_path "$apply_dir/refined_images" \
    2>&1 | tee "logs/${run}_psnr_ssim.log"

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$apply_dir/attacked_clean" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${run}_clean_ba.log"

  echo "Step 4/5: 11 non-geometric attacks"
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
      2>&1 | tee "logs/${run}_attack.log"
  fi

  echo "Step 5/5: attack BA and compact summary"
  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "results/attacked_Result/$attack_run" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${run}_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "results/attacked_Result/$attack_run/ba_report.txt" \
    --output "results/attacked_Result/$attack_run/ba_summary.txt"
}

echo "[$(date '+%F %T')] Start Experiment 5"
echo "Bounded Stage B+ calibration from one shared Experiment 4 checkpoint."

run_candidate "A" "0.90" "0P90"
run_candidate "B" "1.00" "1P00"

echo
echo "[$(date '+%F %T')] Build Experiment 5 comparison"
mkdir -p "$SUMMARY_DIR"
"$PY" scripts/summarize_exp5_stagebplus.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] Experiment 5 complete"
echo "Summary: $SUMMARY"
