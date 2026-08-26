#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"

EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"

BASE_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630"
PLUS20K_CKPT="$BASE_EXP_ROOT/workspace/o256_base_plus20k/ckpts/CoCo_ddpm_final.pth.tar"
PLUS20K_BASE_WM="$BASE_EXP_ROOT/eval_outputs/o256_base_plus20k/finaltest1000/CoCo"
TRAIN_WM="$BASE_EXP_ROOT/train_pairs/o256_base_plus20k_pairs/train10000/CoCo"
REFRESH500_DIR="$BASE_EXP_ROOT/refinement/o256_rgb_plus20k_refresh500_targeted"

DIAG_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/25_supplementary_o256_bitplane_weakbit_diagnosis_20260701"
BIT_WEIGHT_JSON="$DIAG_ROOT/plus20k_refresh500/bit_weights.json"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/26_supplementary_o256_bitweighted_refiner_repair_plus20k_seed61_20260701"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

BITWEIGHTED_DIR="$EXP_ROOT/refinement/o256_rgb_plus20k_bitweighted_repair"
APPLY_BITWEIGHTED_DIR="$EXP_ROOT/apply/o256_rgb_plus20k_bitweighted_repair_eval1000"
ATTACK_RUN_BITWEIGHTED="o256_rgb_plus20k_bitweighted_repair_nongeom"

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

require_path "$PLUS20K_CKPT"
require_path "$PLUS20K_BASE_WM"
require_path "$TRAIN_WM"
require_path "$REFRESH500_DIR/refiner_final.pth"
require_path "$BIT_WEIGHT_JSON"
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
  echo "O256 +20k bit-weighted refiner repair preflight passed."
  echo "Base ckpt: $PLUS20K_CKPT"
  echo "Reuse train pairs: $TRAIN_WM"
  echo "Resume refiner: $REFRESH500_DIR/refiner_final.pth"
  echo "Bit weights: $BIT_WEIGHT_JSON"
  echo "Will run sequentially:"
  echo "  (1) bit-weighted targeted refiner repair (extra 500 steps from plus20k_refresh500)"
  echo "  (2) apply refined images to finaltest1000"
  echo "  (3) quality metrics + clean BA + 11 attacks"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/4: train bit-weighted targeted refiner repair"
if [[ -d "$BITWEIGHTED_DIR" ]]; then
  if [[ ! -f "$BITWEIGHTED_DIR/refiner_final.pth" || ! -f "$BITWEIGHTED_DIR/report.txt" ]]; then
    echo "Incomplete existing bit-weighted refiner output: $BITWEIGHTED_DIR"
    exit 1
  fi
  echo "Reuse completed bit-weighted refiner."
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$PLUS20K_CKPT" \
    --resume_refiner "$REFRESH500_DIR/refiner_final.pth" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt" \
    --output_dir "$BITWEIGHTED_DIR" \
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
    --bit_weight_json "$BIT_WEIGHT_JSON" \
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
    2>&1 | tee "$LOG_DIR/plus20k_bitweighted_repair_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 2/4: apply bit-weighted repair to finaltest1000"
if [[ -d "$APPLY_BITWEIGHTED_DIR/refined_images" && -f "$APPLY_BITWEIGHTED_DIR/report.txt" ]]; then
  apply_pngs="$(count_pngs "$APPLY_BITWEIGHTED_DIR/refined_images")"
  apply_msgs="$(count_msgs "$APPLY_BITWEIGHTED_DIR/refined_images")"
  if [[ "$apply_pngs" -ne 1000 || "$apply_msgs" -ne 1000 || ! -d "$APPLY_BITWEIGHTED_DIR/attacked_clean/Clean" ]]; then
    echo "Incomplete existing bit-weighted repair apply output."
    exit 1
  fi
  echo "Reuse completed bit-weighted repair apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$PLUS20K_CKPT" \
    --refiner "$BITWEIGHTED_DIR/refiner_final.pth" \
    --watermarked_dir "$PLUS20K_BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$APPLY_BITWEIGHTED_DIR" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_bitweighted_repair_apply.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/4: metrics and BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY_BITWEIGHTED_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/plus20k_bitweighted_repair_psnr_ssim.log"

run_lpips "$APPLY_BITWEIGHTED_DIR/refined_images" "$EXP_ROOT/lpips/o256_rgb_plus20k_bitweighted_repair" "$LOG_DIR/plus20k_bitweighted_repair_lpips.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$APPLY_BITWEIGHTED_DIR/attacked_clean" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_bitweighted_repair_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$ATTACK_RUN_BITWEIGHTED" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$ATTACK_RUN_BITWEIGHTED" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete bit-weighted repair attack output."
    exit 1
  fi
  echo "Reuse completed bit-weighted repair attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$APPLY_BITWEIGHTED_DIR/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$ATTACK_RUN_BITWEIGHTED" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_bitweighted_repair_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$ATTACK_RUN_BITWEIGHTED" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_bitweighted_repair_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$ATTACK_RUN_BITWEIGHTED/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$ATTACK_RUN_BITWEIGHTED/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 4/4: summarize"
"$PY" scripts/summarize_o256_bitweighted_refiner_repair_plus20k_20260701.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
