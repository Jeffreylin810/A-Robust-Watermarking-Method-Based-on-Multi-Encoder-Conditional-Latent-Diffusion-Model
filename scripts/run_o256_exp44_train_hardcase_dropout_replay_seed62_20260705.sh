#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=62

EXP41_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/44_supplementary_o256_train_hardcase_dropout_replay_seed62_20260705"

CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed62_exp41_20260705.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
BASE_CKPT="$EXP41_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
BASE_WM="$EXP41_ROOT/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo"
TRAIN_WM="$EXP41_ROOT/train_pairs/exp41_base65_pairs/train10000/CoCo"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed62_exp41_formal/data/CoCo/val/gt"
REFINER_INIT="$EXP41_ROOT/refinement/exp41_stageD_dropout_balanced500/refiner_final.pth"
FINAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"

LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"
PER_ATTACK_CSV="$EXP_ROOT/per_attack_ba.csv"

HARDCASE_DIR="$EXP_ROOT/hardcase_selection"
HARD_WM="$HARDCASE_DIR/hard_subset/watermarked"
HARD_GT="$HARDCASE_DIR/hard_subset/gt"

VARIANT="exp44_hardcase_dropout_replay500"
REFINER_DIR="$EXP_ROOT/refinement/$VARIANT"
APPLY_DIR="$EXP_ROOT/apply/${VARIANT}_finaltest1000"
ATTACK_RUN="${VARIANT}_nongeom"

MIN_FREE_GB=8
EST_REQUIRED_GB=6

CHECK_ONLY=false
QUICK=false
MAX_IMAGES=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ "${1:-}" == "--quick" ]]; then
  QUICK=true
  MAX_IMAGES="${2:-1000}"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only] [--quick N]"
  exit 2
fi

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/refinement" "$EXP_ROOT/apply" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips"

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

count_pngs() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f -iname '*.png' | wc -l
}

count_images() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
}

count_msgs() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l
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

require_path "$PY"
require_path "scripts/prepare_dropout_hardcase_subset.py"
require_path "scripts/summarize_o256_exp44_hardcase_dropout_replay_20260705.py"
require_path "configs/$CONFIG"
require_path "configs/$EVAL1000_CONFIG"
require_path "$BASE_CKPT"
require_path "$BASE_WM"
require_path "$TRAIN_WM"
require_path "$TRAIN_GT"
require_path "$REFINER_INIT"
require_path "$FINAL_GT"

train_pngs="$(count_pngs "$TRAIN_WM")"
train_msgs="$(count_msgs "$TRAIN_WM")"
train_gt_images="$(count_images "$TRAIN_GT")"
base_pngs="$(count_pngs "$BASE_WM")"
base_msgs="$(count_msgs "$BASE_WM")"
if [[ "$train_pngs" -lt 1 || "$train_msgs" -lt 1 || "$train_gt_images" -lt "$train_pngs" || "$base_pngs" -ne 1000 || "$base_msgs" -ne 1000 ]]; then
  echo "Invalid input counts: train_pngs=$train_pngs train_msgs=$train_msgs train_gt_images=$train_gt_images base_pngs=$base_pngs base_msgs=$base_msgs"
  exit 1
fi

free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
  echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  exit 1
fi

if [[ "$CHECK_ONLY" == true ]]; then
  echo "O256 exp44 train-side hardcase dropout replay preflight passed."
  echo "Train pairs: images=$train_pngs messages=$train_msgs gt_images=$train_gt_images"
  echo "Finaltest base outputs: images=$base_pngs messages=$base_msgs"
  echo "Output root: $EXP_ROOT"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Estimated runtime: about 1.5-3 hours"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/5: select train-side Dropout r03 hard cases"
hardcase_ready=false
if [[ -f "$HARDCASE_DIR/hardcase_summary.json" && -d "$HARD_WM" && -d "$HARD_GT" ]]; then
  hard_pngs="$(count_pngs "$HARD_WM")"
  hard_msgs="$(count_msgs "$HARD_WM")"
  hard_gt_images="$(count_images "$HARD_GT")"
  if [[ "$hard_pngs" -ge 1 && "$hard_msgs" -ge 1 && "$hard_gt_images" -ge 1 ]]; then
    echo "Reuse hardcase subset: images=$hard_pngs messages=$hard_msgs gt_images=$hard_gt_images"
    hardcase_ready=true
  else
    echo "Incomplete hardcase subset, rebuilding: images=$hard_pngs messages=$hard_msgs gt_images=$hard_gt_images"
    rm -rf "$HARDCASE_DIR/hard_subset" \
      "$HARDCASE_DIR/hardcase_summary.json" \
      "$HARDCASE_DIR/summary.md" \
      "$HARDCASE_DIR/train_dropout_image_ba.csv" \
      "$HARDCASE_DIR/hard_cases.txt"
  fi
fi
if [[ "$hardcase_ready" != true ]]; then
  "$PY" scripts/prepare_dropout_hardcase_subset.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
    --output_dir "$HARDCASE_DIR" \
    --dropout_prob 0.3 \
    --threshold_ba 90 \
    --min_hard 512 \
    --max_hard 1000 \
    --max_images "$MAX_IMAGES" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${VARIANT}_hardcase_select.log"
fi

echo
echo "[$(date '+%F %T')] Step 2/5: short hardcase Dropout replay from exp41 StageD refiner"
refiner_ready=false
if [[ -d "$REFINER_DIR" ]]; then
  if [[ ! -f "$REFINER_DIR/refiner_final.pth" || ! -f "$REFINER_DIR/report.txt" ]]; then
    echo "Incomplete existing refiner output, rebuilding: $REFINER_DIR"
    rm -rf "$REFINER_DIR"
  else
    echo "Reuse completed refiner: $REFINER_DIR"
    refiner_ready=true
  fi
fi
if [[ "$refiner_ready" != true ]]; then
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$BASE_CKPT" \
    --resume_refiner "$REFINER_INIT" \
    --watermarked_dir "$HARD_WM" \
    --gt_dir "$HARD_GT" \
    --output_dir "$REFINER_DIR" \
    --image_size 256 \
    --steps 500 \
    --batch_size 4 \
    --num_workers 2 \
    --lr 1e-5 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --image_mse_weight 8.0 \
    --image_l1_weight 2.0 \
    --message_weight 1.4 \
    --attack_message_weight 0.45 \
    --attack_profile dropout_r03_only \
    --attack_dropout_prob 0.70 \
    --latent_preserve_weight 0.45 \
    --sensitive_residual_weight 2.5 \
    --quality_gate_psnr 33.00 \
    --quality_gate_weight 70.0 \
    --regularize_weight 0.01 \
    --eval_every 250 \
    --log_every 50 \
    --skip_final_dataset_images \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${VARIANT}_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/5: apply replay refiner to locked finaltest1000"
apply_ready=false
if [[ -d "$APPLY_DIR/refined_images" && -f "$APPLY_DIR/report.txt" ]]; then
  apply_pngs="$(count_pngs "$APPLY_DIR/refined_images")"
  apply_msgs="$(count_msgs "$APPLY_DIR/refined_images")"
  if [[ "$apply_pngs" -ne 1000 || "$apply_msgs" -ne 1000 || ! -d "$APPLY_DIR/attacked_clean/Clean" ]]; then
    echo "Incomplete existing apply output, rebuilding: $APPLY_DIR"
    rm -rf "$APPLY_DIR"
  else
    echo "Reuse completed apply output."
    apply_ready=true
  fi
fi
if [[ "$apply_ready" != true ]]; then
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$BASE_CKPT" \
    --refiner "$REFINER_DIR/refiner_final.pth" \
    --watermarked_dir "$BASE_WM" \
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

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/${VARIANT}_psnr_ssim.log"

run_lpips "$APPLY_DIR/refined_images" "$EXP_ROOT/lpips/$VARIANT" "$LOG_DIR/${VARIANT}_lpips.log"

"$PY" calculate_ba.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "$APPLY_DIR/attacked_clean" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${VARIANT}_clean_ba.log"

echo
echo "[$(date '+%F %T')] Step 4/5: full nongeom attack evaluation"
attack_ready=false
if [[ -d "$EXP_ROOT/attacked_Result/$ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete attack output, rebuilding: $EXP_ROOT/attacked_Result/$ATTACK_RUN"
    rm -rf "$EXP_ROOT/attacked_Result/$ATTACK_RUN"
  else
    echo "Reuse completed attacks."
    attack_ready=true
  fi
fi
if [[ "$attack_ready" != true ]]; then
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
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 5/5: summarize"
"$PY" scripts/summarize_o256_exp44_hardcase_dropout_replay_20260705.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV" \
  --per_attack_csv "$PER_ATTACK_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
echo "Per-attack csv: $PER_ATTACK_CSV"
