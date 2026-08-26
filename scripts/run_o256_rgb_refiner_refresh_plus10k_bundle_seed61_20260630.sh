#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

BASE_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/16_supplementary_o256_base_continuation_ba_push_seed61_20260629"
BASE_CKPT="$BASE_EXP_ROOT/workspace/o256_base_plus10k/ckpts/CoCo_ddpm_final.pth.tar"
BASE_EVAL_WM="$BASE_EXP_ROOT/eval_outputs/o256_base_plus10k/finaltest1000/CoCo"
FORMAL_REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

TRAIN_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt"
EVAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/17_supplementary_o256_rgb_refiner_refresh_plus10k_seed61_20260630"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

PAIR_OUTPUT_ROOT="$EXP_ROOT/train_pairs/o256_base_plus10k_pairs"
TRAIN_WM="$PAIR_OUTPUT_ROOT/train10000/CoCo"

REFRESH500_DIR="$EXP_ROOT/refinement/o256_rgb_refresh500_targeted"
REFRESH1000_DIR="$EXP_ROOT/refinement/o256_rgb_refresh1000_targeted"

APPLY500_DIR="$EXP_ROOT/apply/o256_rgb_refresh500_targeted_eval1000"
APPLY1000_DIR="$EXP_ROOT/apply/o256_rgb_refresh1000_targeted_eval1000"

ATTACK_RUN_500="o256_rgb_refresh500_targeted_nongeom"
ATTACK_RUN_1000="o256_rgb_refresh1000_targeted_nongeom"

MIN_FREE_GB=10
EST_REQUIRED_GB=6

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips" "$EXP_ROOT/refinement" "$EXP_ROOT/apply" "$EXP_ROOT/train_pairs"

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

count_pngs() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l
}

count_msgs() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l
}

run_lpips() {
  local results_dir="$1"
  local output_dir="$2"
  local log_path="$3"
  if [[ -f "$output_dir/summary.json" ]]; then
    echo "Reuse LPIPS summary: $output_dir"
  else
    mkdir -p "$output_dir"
    "$PY" scripts/calculate_lpips_dir.py \
      --gt_dir "$EVAL_GT" \
      --results_dir "$results_dir" \
      --output_dir "$output_dir" \
      --batch_size 8 \
      --net alex \
      2>&1 | tee "$log_path"
  fi
}

require_path "$BASE_CKPT"
require_path "$BASE_EVAL_WM"
require_path "$FORMAL_REFINER"
require_path "configs/$TRAIN_CONFIG"
require_path "configs/$EVAL_CONFIG"
require_path "$TRAIN_GT"
require_path "$EVAL_GT"

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    exit 1
  fi
  echo "O256 +10k base targeted refiner refresh preflight passed."
  echo "Base ckpt: $BASE_CKPT"
  echo "Base eval watermarked dir: $BASE_EVAL_WM"
  echo "Will run sequentially:"
  echo "  (1) generate train10000 pairs from +10k base"
  echo "  (2) targeted refiner refresh 500 steps"
  echo "  (3) targeted refiner refresh 1000 total steps"
  echo "Both candidates evaluate full finaltest1000 quality + clean BA + 11 attacks."
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/5: generate +10k base train10000 refinement pairs"
if [[ -d "$TRAIN_WM" ]]; then
  train_png_count="$(count_pngs "$TRAIN_WM")"
  train_msg_count="$(count_msgs "$TRAIN_WM")"
  if [[ "$train_png_count" -ne 10000 || "$train_msg_count" -ne 10000 ]]; then
    echo "Incomplete existing train pairs: images=$train_png_count messages=$train_msg_count"
    echo "Remove only this incomplete directory before retrying: $PAIR_OUTPUT_ROOT"
    exit 1
  fi
  echo "Reuse completed train10000 pairs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$TRAIN_CONFIG" \
    --resume "$BASE_CKPT" \
    --image_folder "$PAIR_OUTPUT_ROOT" \
    --run_name train10000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/train_pairs_plus10k_eval.log"
fi

run_candidate() {
  local tag="$1"
  local resume_refiner="$2"
  local steps="$3"
  local output_dir="$4"
  local apply_dir="$5"
  local attack_run="$6"
  local train_log="$LOG_DIR/${tag}_train.log"
  local apply_log="$LOG_DIR/${tag}_apply.log"
  local psnr_log="$LOG_DIR/${tag}_psnr_ssim.log"
  local clean_log="$LOG_DIR/${tag}_clean_ba.log"
  local attack_log="$LOG_DIR/${tag}_attack.log"
  local attack_ba_log="$LOG_DIR/${tag}_attack_ba.log"
  local lpips_dir="$EXP_ROOT/lpips/${tag}"

  echo
  echo "[$(date '+%F %T')] ===== Candidate: $tag ====="

  if [[ -d "$output_dir" ]]; then
    if [[ ! -f "$output_dir/refiner_final.pth" || ! -f "$output_dir/report.txt" ]]; then
      echo "Incomplete existing refiner output: $output_dir"
      echo "Remove only this incomplete directory before retrying."
      exit 1
    fi
    echo "Reuse completed refiner checkpoint."
  else
    /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
      --config "$EVAL_CONFIG" \
      --resume "$BASE_CKPT" \
      --resume_refiner "$resume_refiner" \
      --watermarked_dir "$TRAIN_WM" \
      --gt_dir "$TRAIN_GT" \
      --output_dir "$output_dir" \
      --image_size 256 \
      --steps "$steps" \
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
      --skip_final_dataset_images \
      --seed "$SEED" \
      2>&1 | tee "$train_log"
  fi

  if [[ -d "$apply_dir/refined_images" && -f "$apply_dir/report.txt" ]]; then
    apply_png_count="$(count_pngs "$apply_dir/refined_images")"
    apply_msg_count="$(count_msgs "$apply_dir/refined_images")"
    if [[ "$apply_png_count" -ne 1000 || "$apply_msg_count" -ne 1000 || ! -d "$apply_dir/attacked_clean/Clean" ]]; then
      echo "Incomplete existing refined apply output: $apply_dir"
      echo "Remove only this incomplete directory before retrying."
      exit 1
    fi
    echo "Reuse completed refined eval output."
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$EVAL_CONFIG" \
      --resume "$BASE_CKPT" \
      --refiner "$output_dir/refiner_final.pth" \
      --watermarked_dir "$BASE_EVAL_WM" \
      --gt_dir "$EVAL_GT" \
      --output_dir "$apply_dir" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed "$SEED" \
      2>&1 | tee "$apply_log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$EVAL_GT" \
    --results_path "$apply_dir/refined_images" \
    2>&1 | tee "$psnr_log"

  run_lpips "$apply_dir/refined_images" "$lpips_dir" "$LOG_DIR/${tag}_lpips.log"

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$apply_dir/attacked_clean" \
    --gt_message_dir "$BASE_EVAL_WM" \
    2>&1 | tee "$clean_log"

  if [[ -d "$EXP_ROOT/attacked_Result/$attack_run" ]]; then
    attack_count="$(find "$EXP_ROOT/attacked_Result/$attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete existing nongeom attacks for $tag"
      echo "Remove only this incomplete directory before retrying: $EXP_ROOT/attacked_Result/$attack_run"
      exit 1
    fi
    echo "Reuse completed nongeom attacks."
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$EVAL_GT" \
      --results_path "$apply_dir/refined_images" \
      --output_dir "$EXP_ROOT/attacked_Result" \
      --run_name "$attack_run" \
      --attack_scope nongeom \
      --seed "$SEED" \
      2>&1 | tee "$attack_log"
  fi

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$EXP_ROOT/attacked_Result/$attack_run" \
    --gt_message_dir "$BASE_EVAL_WM" \
    2>&1 | tee "$attack_ba_log"

  "$PY" scripts/summarize_ba_report.py \
    "$EXP_ROOT/attacked_Result/$attack_run/ba_report.txt" \
    --output "$EXP_ROOT/attacked_Result/$attack_run/ba_summary.txt"
}

echo
echo "[$(date '+%F %T')] Step 2/5: targeted refresh 500 steps"
run_candidate \
  "refresh500" \
  "$FORMAL_REFINER" \
  "500" \
  "$REFRESH500_DIR" \
  "$APPLY500_DIR" \
  "$ATTACK_RUN_500"

echo
echo "[$(date '+%F %T')] Step 3/5: targeted refresh 1000 total steps"
run_candidate \
  "refresh1000" \
  "$REFRESH500_DIR/refiner_final.pth" \
  "500" \
  "$REFRESH1000_DIR" \
  "$APPLY1000_DIR" \
  "$ATTACK_RUN_1000"

echo
echo "[$(date '+%F %T')] Step 4/5: summarize"
"$PY" scripts/summarize_o256_rgb_refiner_refresh_plus10k_20260630.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Step 5/5: done"
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
