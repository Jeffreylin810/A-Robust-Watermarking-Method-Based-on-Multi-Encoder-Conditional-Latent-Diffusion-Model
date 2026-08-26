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
PLUS20K_BASE_CKPT="$BASE_EXP_ROOT/workspace/o256_base_plus20k/ckpts/CoCo_ddpm_final.pth.tar"
PLUS20K_BASE_EVAL_WM="$BASE_EXP_ROOT/eval_outputs/o256_base_plus20k/finaltest1000/CoCo"
PLUS20K_REFRESH500_REFINER="$BASE_EXP_ROOT/refinement/o256_rgb_plus20k_refresh500_targeted/refiner_final.pth"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/23_supplementary_o256_base_plus20k_weakattack_plus5k_seed61_20260630"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"
CONFIG_NAME="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_plus20k_weakattack_plus5k_seed61_supp_20260630.yml"

WEAKPLUS5K_WORKSPACE="$EXP_ROOT/workspace/o256_base_plus20k_weakattack_plus5k"
WEAKPLUS5K_CKPT="$WEAKPLUS5K_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
WEAKPLUS5K_BASE_OUTPUT="$EXP_ROOT/eval_outputs/o256_base_plus20k_weakattack_plus5k"
WEAKPLUS5K_BASE_WM="$WEAKPLUS5K_BASE_OUTPUT/finaltest1000/CoCo"

WEAKPLUS5K_RGB_REUSE="$EXP_ROOT/refinement/o256_rgb_plus20k_weakattack_plus5k_reuse_refresh500"

BASE_ATTACK_RUN="o256_base_plus20k_weakattack_plus5k_nongeom"
RGB_REUSE_ATTACK_RUN="o256_rgb_plus20k_weakattack_plus5k_reuse_refresh500_nongeom"

MIN_FREE_GB=8
EST_REQUIRED_GB=6

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips" "$EXP_ROOT/workspace" "$EXP_ROOT/refinement"

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
src = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_plus20k_seed61_supp_20260630.yml")
dst = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/configs/$CONFIG_NAME")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["data"]["output_dir"] = "$WEAKPLUS5K_WORKSPACE"
cfg["training"]["n_epochs"] = 11
cfg["watermarking"]["attack_profile"] = "identity_nongeom_targeted_weak"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst)
PY
}

require_path "$PLUS20K_BASE_CKPT"
require_path "$PLUS20K_BASE_EVAL_WM"
require_path "$PLUS20K_REFRESH500_REFINER"
require_path "configs/$EVAL_CONFIG"
require_path "configs/$EVAL1000_CONFIG"
require_path "$FINAL_GT"

generate_train_config >/dev/null

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    exit 1
  fi
  echo "O256 +20k weak-attack base continuation preflight passed."
  echo "Base resume ckpt: $PLUS20K_BASE_CKPT"
  echo "Will run sequentially:"
  echo "  (1) continue +20k base by +5k with identity_nongeom_targeted_weak"
  echo "  (2) evaluate O256-Base weak-targeted +5k"
  echo "  (3) evaluate O256-RGB with reused plus20k_refresh500 refiner"
  echo "  (4) summarize weak-attack continuation results"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/4: continue +20k base by +5k with weak-attack profile"
if [[ -f "$WEAKPLUS5K_CKPT" ]]; then
  echo "Reuse trained weak-targeted +5k checkpoint."
elif [[ -d "$WEAKPLUS5K_WORKSPACE" ]]; then
  echo "Existing incomplete workspace detected: $WEAKPLUS5K_WORKSPACE"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$CONFIG_NAME" \
    --resume "$PLUS20K_BASE_CKPT" \
    --image_folder "results/validation/o256_base_plus20k_weakattack_plus5k" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_train.log"
fi
[[ -f "$WEAKPLUS5K_CKPT" ]] || { echo "Missing expected weak-targeted +5k checkpoint: $WEAKPLUS5K_CKPT"; exit 1; }

echo
echo "[$(date '+%F %T')] Step 2/4: evaluate O256-Base weak-targeted +5k"
if [[ -d "$WEAKPLUS5K_BASE_WM" ]]; then
  png_count="$(count_pngs "$WEAKPLUS5K_BASE_WM")"
  msg_count="$(count_msgs "$WEAKPLUS5K_BASE_WM")"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete weak-targeted +5k base eval output: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse completed weak-targeted +5k base eval outputs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$WEAKPLUS5K_CKPT" \
    --image_folder "$WEAKPLUS5K_BASE_OUTPUT" \
    --run_name finaltest1000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_base_eval.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$WEAKPLUS5K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_base_psnr_ssim.log"

run_lpips "$WEAKPLUS5K_BASE_WM" "$EXP_ROOT/lpips/o256_base_plus20k_weakattack_plus5k" "$LOG_DIR/plus20k_weakattack_plus5k_base_lpips.log"

mkdir -p "$EXP_ROOT/attacked_Result/clean_o256_base_plus20k_weakattack_plus5k"
ln -sfn "$(realpath "$WEAKPLUS5K_BASE_WM")" "$EXP_ROOT/attacked_Result/clean_o256_base_plus20k_weakattack_plus5k/Clean"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$WEAKPLUS5K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/clean_o256_base_plus20k_weakattack_plus5k" \
  --gt_message_dir "$WEAKPLUS5K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_base_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete weak-targeted +5k base attack output."
    exit 1
  fi
  echo "Reuse completed weak-targeted +5k base attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$WEAKPLUS5K_BASE_WM" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$BASE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_base_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$WEAKPLUS5K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$WEAKPLUS5K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_base_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 3/4: evaluate O256-RGB with reused plus20k_refresh500 refiner"
if [[ -d "$WEAKPLUS5K_RGB_REUSE/refined_images" && -f "$WEAKPLUS5K_RGB_REUSE/report.txt" ]]; then
  reuse_png="$(count_pngs "$WEAKPLUS5K_RGB_REUSE/refined_images")"
  if [[ "$reuse_png" -ne 1000 || ! -d "$WEAKPLUS5K_RGB_REUSE/attacked_clean/Clean" ]]; then
    echo "Incomplete weak-targeted +5k rgb reuse output."
    exit 1
  fi
  echo "Reuse completed weak-targeted +5k rgb reuse outputs."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$WEAKPLUS5K_CKPT" \
    --refiner "$PLUS20K_REFRESH500_REFINER" \
    --watermarked_dir "$WEAKPLUS5K_BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$WEAKPLUS5K_RGB_REUSE" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_rgb_reuse_apply.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$WEAKPLUS5K_RGB_REUSE/refined_images" \
  2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_rgb_reuse_psnr_ssim.log"
run_lpips "$WEAKPLUS5K_RGB_REUSE/refined_images" "$EXP_ROOT/lpips/o256_rgb_plus20k_weakattack_plus5k_reuse_refresh500" "$LOG_DIR/plus20k_weakattack_plus5k_rgb_reuse_lpips.log"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$WEAKPLUS5K_CKPT" \
  --attack_dir "$WEAKPLUS5K_RGB_REUSE/attacked_clean" \
  --gt_message_dir "$WEAKPLUS5K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_rgb_reuse_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete weak-targeted +5k rgb reuse attack output."
    exit 1
  fi
  echo "Reuse completed weak-targeted +5k rgb reuse attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$WEAKPLUS5K_RGB_REUSE/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$RGB_REUSE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_rgb_reuse_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$WEAKPLUS5K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN" \
  --gt_message_dir "$WEAKPLUS5K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_weakattack_plus5k_rgb_reuse_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 4/4: summarize"
"$PY" scripts/summarize_o256_base_plus20k_weakattack_plus5k_20260630.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
