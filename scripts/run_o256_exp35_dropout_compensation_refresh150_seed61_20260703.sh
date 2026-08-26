#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt"

EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"

EXP34_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/34_supplementary_o256_exp33_full_candidate_validation_seed61_20260703"
EXP34_CKPT="$EXP34_ROOT/workspace/exp34_base_dropout_focus_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
EXP34_BASE_WM="$EXP34_ROOT/eval_outputs/exp34_full_candidate_dropout_focus_base/finaltest1000/CoCo"
EXP34_TRAIN_WM="$EXP34_ROOT/train_pairs/exp34_base_dropout_focus_plus5k_pairs/train10000/CoCo"
EXP34_REFINER="$EXP34_ROOT/refinement/exp34_full_candidate_dropout_balanced_rgb/refiner_final.pth"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/35_supplementary_o256_exp34_dropout_compensation_refresh150_seed61_20260703"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

VARIANT="exp35_dropout_compensation_refresh150"
REFINER_DIR="$EXP_ROOT/refinement/$VARIANT"
APPLY_DIR="$EXP_ROOT/apply/${VARIANT}_eval1000"
ATTACK_RUN="${VARIANT}_nongeom"

MIN_FREE_GB=5
EST_REQUIRED_GB=3

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

require_path "$EXP34_CKPT"
require_path "$EXP34_BASE_WM"
require_path "$EXP34_TRAIN_WM"
require_path "$EXP34_REFINER"
require_path "$FINAL_GT"
require_path "$TRAIN_GT"
require_path "configs/$EVAL_CONFIG"
require_path "configs/$EVAL1000_CONFIG"

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
    exit 1
  fi
  pair_pngs="$(count_pngs "$EXP34_TRAIN_WM")"
  pair_msgs="$(count_msgs "$EXP34_TRAIN_WM")"
  echo "O256 exp35 dropout compensation refresh150 preflight passed."
  echo "Base ckpt: $EXP34_CKPT"
  echo "Base finaltest1000 WM: $EXP34_BASE_WM"
  echo "Resume refiner: $EXP34_REFINER"
  echo "Reuse train pairs: $EXP34_TRAIN_WM"
  echo "Train pair images/messages: $pair_pngs / $pair_msgs"
  echo "Will run sequentially:"
  echo "  (1) train 150-step dropout compensation refiner from exp34 refiner"
  echo "  (2) apply to finaltest1000"
  echo "  (3) compute PSNR/SSIM/LPIPS/Clean BA/full nongeom BA"
  echo "  (4) summarize"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/4: train 150-step dropout compensation refiner"
if [[ -d "$REFINER_DIR" ]]; then
  if [[ ! -f "$REFINER_DIR/refiner_final.pth" || ! -f "$REFINER_DIR/report.txt" ]]; then
    echo "Incomplete existing refiner output: $REFINER_DIR"
    exit 1
  fi
  echo "Reuse completed refiner: $REFINER_DIR"
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$EXP34_CKPT" \
    --resume_refiner "$EXP34_REFINER" \
    --watermarked_dir "$EXP34_TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
    --output_dir "$REFINER_DIR" \
    --image_size 256 \
    --steps 150 \
    --batch_size 4 \
    --num_workers 2 \
    --lr 7e-6 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --image_mse_weight 8.5 \
    --image_l1_weight 2.2 \
    --message_weight 1.35 \
    --attack_message_weight 0.55 \
    --attack_profile dropout_r03_only \
    --attack_dropout_prob 0.70 \
    --latent_preserve_weight 0.40 \
    --sensitive_residual_weight 2.2 \
    --quality_gate_psnr 33.05 \
    --quality_gate_weight 90.0 \
    --regularize_weight 0.01 \
    --eval_every 75 \
    --log_every 25 \
    --skip_final_dataset_images \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${VARIANT}_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 2/4: apply dropout compensation refiner"
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
    --resume "$EXP34_CKPT" \
    --refiner "$REFINER_DIR/refiner_final.pth" \
    --watermarked_dir "$EXP34_BASE_WM" \
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
    2>&1 | tee "$LOG_DIR/${VARIANT}_apply.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/4: metrics and BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/${VARIANT}_psnr_ssim.log"

run_lpips "$APPLY_DIR/refined_images" "$EXP_ROOT/lpips/$VARIANT" "$LOG_DIR/${VARIANT}_lpips.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$EXP34_CKPT" \
  --attack_dir "$APPLY_DIR/attacked_clean" \
  --gt_message_dir "$EXP34_BASE_WM" \
  2>&1 | tee "$LOG_DIR/${VARIANT}_clean_ba.log"

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
    2>&1 | tee "$LOG_DIR/${VARIANT}_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$EXP34_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$ATTACK_RUN" \
  --gt_message_dir "$EXP34_BASE_WM" \
  2>&1 | tee "$LOG_DIR/${VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 4/4: summarize"
"$PY" scripts/summarize_o256_exp35_dropout_compensation_refresh150_20260703.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
