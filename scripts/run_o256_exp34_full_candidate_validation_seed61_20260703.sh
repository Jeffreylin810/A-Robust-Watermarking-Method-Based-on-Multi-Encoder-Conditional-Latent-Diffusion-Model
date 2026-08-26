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
TRAIN_PAIR_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_formal.yml"
TRAIN_CONFIG_TEMPLATE="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_plus20k_seed61_supp_20260630.yml"
TRAIN_CONFIG="CoCo_train10000_ours256_msg32_exp34_full_candidate_dropout_focus_seed61_20260703.yml"

A_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/23_supplementary_o256_base_plus20k_weakattack_plus5k_seed61_20260630"
A_CKPT="$A_EXP_ROOT/workspace/o256_base_plus20k_weakattack_plus5k/ckpts/CoCo_ddpm_final.pth.tar"

PLUS20K_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630"
REFRESH500_INIT="$PLUS20K_ROOT/refinement/o256_rgb_plus20k_refresh500_targeted/refiner_final.pth"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/34_supplementary_o256_exp33_full_candidate_validation_seed61_20260703"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

WORKSPACE="$EXP_ROOT/workspace/exp34_base_dropout_focus_plus5k"
CKPT="$WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"

BASE_VARIANT="exp34_full_candidate_dropout_focus_base"
BASE_OUTPUT="$EXP_ROOT/eval_outputs/$BASE_VARIANT"
BASE_WM="$BASE_OUTPUT/finaltest1000/CoCo"
BASE_ATTACK_RUN="${BASE_VARIANT}_nongeom"

PAIR_OUTPUT_ROOT="$EXP_ROOT/train_pairs/exp34_base_dropout_focus_plus5k_pairs"
TRAIN_WM="$PAIR_OUTPUT_ROOT/train10000/CoCo"

RGB_VARIANT="exp34_full_candidate_dropout_balanced_rgb"
REFINER_DIR="$EXP_ROOT/refinement/$RGB_VARIANT"
APPLY_DIR="$EXP_ROOT/apply/${RGB_VARIANT}_eval1000"
RGB_ATTACK_RUN="${RGB_VARIANT}_nongeom"

MIN_FREE_GB=12
EST_REQUIRED_GB=8

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/workspace" "$EXP_ROOT/eval_outputs" "$EXP_ROOT/train_pairs" "$EXP_ROOT/refinement" "$EXP_ROOT/apply" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips"

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

generate_train_config() {
  "$PY" - <<PY
from pathlib import Path
import yaml

src = Path("$ROOT/configs/$TRAIN_CONFIG_TEMPLATE")
dst = Path("$ROOT/configs/$TRAIN_CONFIG")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["data"]["output_dir"] = "$WORKSPACE"
cfg["training"]["n_epochs"] = 12
cfg["watermarking"]["attack_profile"] = "identity_nongeom_dropout_focus"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst)
PY
}

require_path "$A_CKPT"
require_path "$REFRESH500_INIT"
require_path "$FINAL_GT"
require_path "$TRAIN_GT"
require_path "configs/$EVAL_CONFIG"
require_path "configs/$EVAL1000_CONFIG"
require_path "configs/$TRAIN_PAIR_CONFIG"
require_path "configs/$TRAIN_CONFIG_TEMPLATE"
generate_train_config >/dev/null

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
    exit 1
  fi
  echo "O256 exp34 full candidate validation preflight passed."
  echo "Resume A/exp23 ckpt: $A_CKPT"
  echo "Initial RGB refiner: $REFRESH500_INIT"
  echo "Generated train config: configs/$TRAIN_CONFIG"
  echo "Will run sequentially:"
  echo "  (1) retrain dropout-focus base continuation from A/exp23"
  echo "  (2) evaluate regenerated base on finaltest1000 + full nongeom BA"
  echo "  (3) regenerate train10000 pairs from regenerated base"
  echo "  (4) train exp33 dropout-balanced refiner settings"
  echo "  (5) evaluate RGB on finaltest1000 + full nongeom BA"
  echo "  (6) summarize candidate validation"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/6: retrain dropout-focus base continuation from A/exp23"
if [[ -f "$CKPT" ]]; then
  echo "Reuse trained checkpoint: $CKPT"
elif [[ -d "$WORKSPACE" ]]; then
  echo "Existing incomplete workspace detected: $WORKSPACE"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$TRAIN_CONFIG" \
    --resume "$A_CKPT" \
    --image_folder "results/validation/exp34_base_dropout_focus_plus5k" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp34_base_dropout_focus_train.log"
fi
[[ -f "$CKPT" ]] || { echo "Missing expected checkpoint: $CKPT"; exit 1; }

echo
echo "[$(date '+%F %T')] Step 2/6: evaluate regenerated base"
if [[ -d "$BASE_WM" ]]; then
  png_count="$(count_pngs "$BASE_WM")"
  msg_count="$(count_msgs "$BASE_WM")"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete base eval output: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse completed base eval outputs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$CKPT" \
    --image_folder "$BASE_OUTPUT" \
    --run_name finaltest1000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp34_base_dropout_focus_eval.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_psnr_ssim.log"

run_lpips "$BASE_WM" "$EXP_ROOT/lpips/$BASE_VARIANT" "$LOG_DIR/${BASE_VARIANT}_lpips.log"

mkdir -p "$EXP_ROOT/attacked_Result/clean_$BASE_VARIANT"
ln -sfn "$(realpath "$BASE_WM")" "$EXP_ROOT/attacked_Result/clean_$BASE_VARIANT/Clean"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/clean_$BASE_VARIANT" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete base attack output."
    exit 1
  fi
  echo "Reuse completed base attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$BASE_WM" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$BASE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 3/6: regenerate train10000 pairs from regenerated base"
if [[ -d "$TRAIN_WM" ]]; then
  pair_pngs="$(count_pngs "$TRAIN_WM")"
  pair_msgs="$(count_msgs "$TRAIN_WM")"
  if [[ "$pair_pngs" -ne 10000 || "$pair_msgs" -ne 10000 ]]; then
    echo "Incomplete train pairs: images=$pair_pngs messages=$pair_msgs"
    exit 1
  fi
  echo "Reuse completed train pairs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$TRAIN_PAIR_CONFIG" \
    --resume "$CKPT" \
    --image_folder "$PAIR_OUTPUT_ROOT" \
    --run_name train10000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp34_train_pairs_eval.log"
fi

echo
echo "[$(date '+%F %T')] Step 4/6: train dropout-balanced refiner settings"
if [[ -d "$REFINER_DIR" ]]; then
  if [[ ! -f "$REFINER_DIR/refiner_final.pth" || ! -f "$REFINER_DIR/report.txt" ]]; then
    echo "Incomplete existing refiner output: $REFINER_DIR"
    exit 1
  fi
  echo "Reuse completed refiner: $REFINER_DIR"
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$CKPT" \
    --resume_refiner "$REFRESH500_INIT" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
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
    2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 5/6: evaluate dropout-balanced RGB result"
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
    --resume "$CKPT" \
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
    2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_apply.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_psnr_ssim.log"

run_lpips "$APPLY_DIR/refined_images" "$EXP_ROOT/lpips/$RGB_VARIANT" "$LOG_DIR/${RGB_VARIANT}_lpips.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$APPLY_DIR/attacked_clean" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete RGB attack output."
    exit 1
  fi
  echo "Reuse completed RGB attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$APPLY_DIR/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$RGB_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 6/6: summarize"
"$PY" scripts/summarize_o256_exp34_full_candidate_validation_20260703.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
