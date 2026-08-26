#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"

EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_CONFIG_TEMPLATE="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_plus20k_seed61_supp_20260630.yml"
TRAIN_CONFIG="CoCo_train10000_ours256_msg32_A_dropout_focus_plus5k_seed61_20260701.yml"

BASE_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630"
A_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/23_supplementary_o256_base_plus20k_weakattack_plus5k_seed61_20260630"
A_CKPT="$A_EXP_ROOT/workspace/o256_base_plus20k_weakattack_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
PLUS20K_REFRESH500_REFINER="$BASE_EXP_ROOT/refinement/o256_rgb_plus20k_refresh500_targeted/refiner_final.pth"


EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/28_supplementary_o256_attack_schedule_dropout_focus_from_A_seed61_20260701"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

WORKSPACE="$EXP_ROOT/workspace/o256_base_A_dropout_focus_plus5k"
CKPT="$WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
BASE_OUTPUT="$EXP_ROOT/eval_outputs/o256_base_A_dropout_focus_plus5k"
BASE_WM="$BASE_OUTPUT/finaltest1000/CoCo"
RGB_REUSE="$EXP_ROOT/refinement/o256_rgb_A_dropout_focus_plus5k_reuse_refresh500"

BASE_ATTACK_RUN="o256_base_A_dropout_focus_plus5k_nongeom"
RGB_ATTACK_RUN="o256_rgb_A_dropout_focus_plus5k_reuse_refresh500_nongeom"

MIN_FREE_GB=7
EST_REQUIRED_GB=5

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
require_path "$PLUS20K_REFRESH500_REFINER"
require_path "configs/$EVAL_CONFIG"
require_path "configs/$EVAL1000_CONFIG"
require_path "configs/$TRAIN_CONFIG_TEMPLATE"
require_path "$FINAL_GT"

generate_train_config >/dev/null

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
    echo "Please clean space before running this experiment."
    exit 1
  fi
  echo "O256 base-side dropout-focus scheduling from A +5k preflight passed."
  echo "Base resume ckpt: $A_CKPT"
    echo "Will run sequentially:"
  echo "  (1) continue A checkpoint by +5k with identity_nongeom_dropout_focus"
  echo "  (2) evaluate base result"
  echo "  (3) apply reused plus20k_refresh500 RGB refiner"
  echo "  (4) compute PSNR/SSIM/LPIPS/Clean BA/full nongeom BA and summarize"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/4: base-side dropout-focus scheduling from A +5k continuation"
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
    --image_folder "results/validation/o256_base_A_dropout_focus_plus5k" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/A_dropout_focus_plus5k_train.log"
fi
[[ -f "$CKPT" ]] || { echo "Missing expected checkpoint: $CKPT"; exit 1; }

echo
echo "[$(date '+%F %T')] Step 2/4: evaluate dropout-focus scheduled result"
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
    2>&1 | tee "$LOG_DIR/A_dropout_focus_plus5k_eval.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/A_dropout_focus_plus5k_psnr_ssim.log"

run_lpips "$BASE_WM" "$EXP_ROOT/lpips/o256_base_A_dropout_focus_plus5k" "$LOG_DIR/A_dropout_focus_plus5k_lpips.log"

mkdir -p "$EXP_ROOT/attacked_Result/clean_o256_base_A_dropout_focus_plus5k"
ln -sfn "$(realpath "$BASE_WM")" "$EXP_ROOT/attacked_Result/clean_o256_base_A_dropout_focus_plus5k/Clean"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/clean_o256_base_A_dropout_focus_plus5k" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/A_dropout_focus_plus5k_clean_ba.log"

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
    2>&1 | tee "$LOG_DIR/A_dropout_focus_plus5k_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/A_dropout_focus_plus5k_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 3/4: apply reused plus20k_refresh500 RGB refiner"
if [[ -d "$RGB_REUSE/refined_images" && -f "$RGB_REUSE/report.txt" ]]; then
  reuse_png="$(count_pngs "$RGB_REUSE/refined_images")"
  if [[ "$reuse_png" -ne 1000 || ! -d "$RGB_REUSE/attacked_clean/Clean" ]]; then
    echo "Incomplete RGB reuse output."
    exit 1
  fi
  echo "Reuse completed RGB reuse outputs."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$CKPT" \
    --refiner "$PLUS20K_REFRESH500_REFINER" \
    --watermarked_dir "$BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$RGB_REUSE" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/rgb_A_dropout_focus_plus5k_reuse_apply.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$RGB_REUSE/refined_images" \
  2>&1 | tee "$LOG_DIR/rgb_A_dropout_focus_plus5k_reuse_psnr_ssim.log"

run_lpips "$RGB_REUSE/refined_images" "$EXP_ROOT/lpips/o256_rgb_A_dropout_focus_plus5k_reuse_refresh500" "$LOG_DIR/rgb_A_dropout_focus_plus5k_reuse_lpips.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$RGB_REUSE/attacked_clean" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/rgb_A_dropout_focus_plus5k_reuse_clean_ba.log"

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
    --results_path "$RGB_REUSE/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$RGB_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/rgb_A_dropout_focus_plus5k_reuse_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/rgb_A_dropout_focus_plus5k_reuse_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 4/4: summarize"
"$PY" scripts/summarize_o256_attack_schedule_dropout_focus_from_A_20260701.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
