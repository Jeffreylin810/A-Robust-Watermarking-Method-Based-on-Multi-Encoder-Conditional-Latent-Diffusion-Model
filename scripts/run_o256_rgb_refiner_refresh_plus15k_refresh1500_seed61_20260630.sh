#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"

EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"

BASE_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/18_supplementary_o256_base_plus15k_then_refresh500_seed61_20260630"
PREV_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/19_supplementary_o256_rgb_refiner_refresh_plus15k_refresh1000_seed61_20260630"
PLUS15K_CKPT="$BASE_EXP_ROOT/workspace/o256_base_plus15k/ckpts/CoCo_ddpm_final.pth.tar"
PLUS15K_BASE_WM="$BASE_EXP_ROOT/eval_outputs/o256_base_plus15k/finaltest1000/CoCo"
TRAIN_WM="$BASE_EXP_ROOT/train_pairs/o256_base_plus15k_pairs/train10000/CoCo"
REFRESH1000_DIR="$PREV_EXP_ROOT/refinement/o256_rgb_plus15k_refresh1000_targeted"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/20_supplementary_o256_rgb_refiner_refresh_plus15k_refresh1500_seed61_20260630"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

REFRESH1500_DIR="$EXP_ROOT/refinement/o256_rgb_plus15k_refresh1500_targeted"
APPLY1500_DIR="$EXP_ROOT/apply/o256_rgb_plus15k_refresh1500_targeted_eval1000"
ATTACK_RUN_1500="o256_rgb_plus15k_refresh1500_targeted_nongeom"

MIN_FREE_GB=3
EST_REQUIRED_GB=2

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips" "$EXP_ROOT/refinement" "$EXP_ROOT/apply"

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
      --gt_dir "$FINAL_GT" \
      --results_dir "$results_dir" \
      --output_dir "$output_dir" \
      --batch_size 8 \
      --net alex \
      2>&1 | tee "$log_path"
  fi
}

require_path "$PLUS15K_CKPT"
require_path "$PLUS15K_BASE_WM"
require_path "$TRAIN_WM"
require_path "$REFRESH1000_DIR/refiner_final.pth"
require_path "configs/$EVAL_CONFIG"
require_path "configs/$EVAL1000_CONFIG"
require_path "$FINAL_GT"

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    exit 1
  fi
  echo "O256 +15k refresh1500 preflight passed."
  echo "Base ckpt: $PLUS15K_CKPT"
  echo "Reuse train pairs: $TRAIN_WM"
  echo "Resume refiner: $REFRESH1000_DIR/refiner_final.pth"
  echo "Will run sequentially:"
  echo "  (1) refresh1500 targeted refiner (extra 500 steps from refresh1000)"
  echo "  (2) apply refined images to finaltest1000"
  echo "  (3) quality metrics + clean BA + 11 attacks"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/4: train refresh1500 (extra 500 steps from refresh1000)"
if [[ -d "$REFRESH1500_DIR" ]]; then
  if [[ ! -f "$REFRESH1500_DIR/refiner_final.pth" || ! -f "$REFRESH1500_DIR/report.txt" ]]; then
    echo "Incomplete existing refresh1500 refiner output: $REFRESH1500_DIR"
    exit 1
  fi
  echo "Reuse completed refresh1500 refiner."
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$PLUS15K_CKPT" \
    --resume_refiner "$REFRESH1000_DIR/refiner_final.pth" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt" \
    --output_dir "$REFRESH1500_DIR" \
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
    --skip_final_dataset_images \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus15k_refresh1500_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 2/4: apply refresh1500 to finaltest1000"
if [[ -d "$APPLY1500_DIR/refined_images" && -f "$APPLY1500_DIR/report.txt" ]]; then
  apply_pngs="$(count_pngs "$APPLY1500_DIR/refined_images")"
  apply_msgs="$(count_msgs "$APPLY1500_DIR/refined_images")"
  if [[ "$apply_pngs" -ne 1000 || "$apply_msgs" -ne 1000 || ! -d "$APPLY1500_DIR/attacked_clean/Clean" ]]; then
    echo "Incomplete existing refresh1500 apply output."
    exit 1
  fi
  echo "Reuse completed refresh1500 apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$PLUS15K_CKPT" \
    --refiner "$REFRESH1500_DIR/refiner_final.pth" \
    --watermarked_dir "$PLUS15K_BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$APPLY1500_DIR" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus15k_refresh1500_apply.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/4: metrics and BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY1500_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/plus15k_refresh1500_psnr_ssim.log"

run_lpips "$APPLY1500_DIR/refined_images" "$EXP_ROOT/lpips/o256_rgb_plus15k_refresh1500" "$LOG_DIR/plus15k_refresh1500_lpips.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS15K_CKPT" \
  --attack_dir "$APPLY1500_DIR/attacked_clean" \
  --gt_message_dir "$PLUS15K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus15k_refresh1500_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$ATTACK_RUN_1500" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$ATTACK_RUN_1500" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete refresh1500 attack output."
    exit 1
  fi
  echo "Reuse completed refresh1500 attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$APPLY1500_DIR/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$ATTACK_RUN_1500" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus15k_refresh1500_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS15K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$ATTACK_RUN_1500" \
  --gt_message_dir "$PLUS15K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus15k_refresh1500_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$ATTACK_RUN_1500/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$ATTACK_RUN_1500/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 4/4: summarize"
"$PY" scripts/summarize_o256_rgb_refiner_refresh_plus15k_refresh1500_20260630.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
