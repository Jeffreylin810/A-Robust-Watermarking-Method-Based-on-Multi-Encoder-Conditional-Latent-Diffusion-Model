#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt"

TRAIN_PAIR_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"

PLUS20K_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630"
REFRESH500_INIT="$PLUS20K_ROOT/refinement/o256_rgb_plus20k_refresh500_targeted/refiner_final.pth"

EXP28_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/28_supplementary_o256_attack_schedule_dropout_focus_from_A_seed61_20260701"
EXP28_CKPT="$EXP28_ROOT/workspace/o256_base_A_dropout_focus_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
EXP28_BASE_WM="$EXP28_ROOT/eval_outputs/o256_base_A_dropout_focus_plus5k/finaltest1000/CoCo"
EXP29_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/29_supplementary_o256_exp28_quality_recovery_refresh500_seed61_20260702"
EXP29_TRAIN_WM="$EXP29_ROOT/train_pairs/o256_base_A_dropout_focus_plus5k_pairs/train10000/CoCo"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/30_supplementary_o256_exp28_conservative_refresh250_seed61_20260702"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

TRAIN_WM="$EXP29_TRAIN_WM"
REFINER_DIR="$EXP_ROOT/refinement/o256_rgb_exp28_conservative_refresh250"
APPLY_DIR="$EXP_ROOT/apply/o256_rgb_exp28_conservative_refresh250_eval1000"
ATTACK_RUN="o256_rgb_exp28_conservative_refresh250_nongeom"

MIN_FREE_GB=4
EST_REQUIRED_GB=2

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
      --gt_dir "$FINAL_GT" \
      --results_dir "$results_dir" \
      --output_dir "$output_dir" \
      --batch_size 8 \
      --net alex \
      2>&1 | tee "$log_path"
  fi
}

require_path "$EXP28_CKPT"
require_path "$EXP28_BASE_WM"
require_path "$REFRESH500_INIT"
require_path "configs/$EVAL_CONFIG"
require_path "configs/$EVAL1000_CONFIG"
require_path "$TRAIN_GT"
require_path "$FINAL_GT"
require_path "$TRAIN_WM"

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
    exit 1
  fi
  echo "O256 exp28 conservative refresh250 preflight passed."
  echo "Base ckpt: $EXP28_CKPT"
  echo "Base finaltest1000 WM: $EXP28_BASE_WM"
  echo "Initial refiner: $REFRESH500_INIT"
  echo "Reuse train pairs: $TRAIN_WM"
  echo "Will run sequentially:"
  echo "  (1) verify/reuse exp28 train10000 refinement pairs from exp29"
  echo "  (2) train conservative RGB refiner for 250 steps"
  echo "  (3) apply refiner to finaltest1000"
  echo "  (4) compute PSNR/SSIM/LPIPS/Clean BA/full nongeom BA"
  echo "  (5) summarize"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/5: verify exp28 train10000 refinement pairs from exp29"
pair_pngs="$(count_pngs "$TRAIN_WM")"
pair_msgs="$(count_msgs "$TRAIN_WM")"
if [[ "$pair_pngs" -ne 10000 || "$pair_msgs" -ne 10000 ]]; then
  echo "Incomplete exp28 train pairs: images=$pair_pngs messages=$pair_msgs"
  exit 1
fi
echo "Reuse completed exp28 train pairs: $TRAIN_WM"

echo
echo "[$(date '+%F %T')] Step 2/5: train exp28 conservative refresh250"
if [[ -d "$REFINER_DIR" ]]; then
  if [[ ! -f "$REFINER_DIR/refiner_final.pth" || ! -f "$REFINER_DIR/report.txt" ]]; then
    echo "Incomplete existing refiner output: $REFINER_DIR"
    exit 1
  fi
  echo "Reuse completed exp28 quality-recovery refiner."
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$EXP28_CKPT" \
    --resume_refiner "$REFRESH500_INIT" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
    --output_dir "$REFINER_DIR" \
    --image_size 256 \
    --steps 250 \
    --batch_size 4 \
    --num_workers 2 \
    --lr 1e-5 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --image_mse_weight 6.0 \
    --image_l1_weight 1.5 \
    --message_weight 1.5 \
    --attack_message_weight 0.30 \
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
    2>&1 | tee "$LOG_DIR/exp28_conservative_refresh250_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/5: apply exp28 conservative refresh250"
if [[ -d "$APPLY_DIR/refined_images" && -f "$APPLY_DIR/report.txt" ]]; then
  apply_pngs="$(count_pngs "$APPLY_DIR/refined_images")"
  apply_msgs="$(count_msgs "$APPLY_DIR/refined_images")"
  if [[ "$apply_pngs" -ne 1000 || "$apply_msgs" -ne 1000 || ! -d "$APPLY_DIR/attacked_clean/Clean" ]]; then
    echo "Incomplete existing apply output."
    exit 1
  fi
  echo "Reuse completed apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$EXP28_CKPT" \
    --refiner "$REFINER_DIR/refiner_final.pth" \
    --watermarked_dir "$EXP28_BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$APPLY_DIR" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp28_conservative_refresh250_apply.log"
fi

echo
echo "[$(date '+%F %T')] Step 4/5: metrics and BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/exp28_conservative_refresh250_psnr_ssim.log"

run_lpips "$APPLY_DIR/refined_images" "$EXP_ROOT/lpips/o256_rgb_exp28_conservative_refresh250" "$LOG_DIR/exp28_conservative_refresh250_lpips.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$EXP28_CKPT" \
  --attack_dir "$APPLY_DIR/attacked_clean" \
  --gt_message_dir "$EXP28_BASE_WM" \
  2>&1 | tee "$LOG_DIR/exp28_conservative_refresh250_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete attack output."
    exit 1
  fi
  echo "Reuse completed attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$APPLY_DIR/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp28_conservative_refresh250_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$EXP28_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$ATTACK_RUN" \
  --gt_message_dir "$EXP28_BASE_WM" \
  2>&1 | tee "$LOG_DIR/exp28_conservative_refresh250_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 5/5: summarize"
"$PY" scripts/summarize_o256_exp28_conservative_refresh250_20260702.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
