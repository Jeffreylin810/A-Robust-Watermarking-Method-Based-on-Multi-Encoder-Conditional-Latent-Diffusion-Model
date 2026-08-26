#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"

TRAIN_PAIR_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
FORMAL_REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt"

BASE_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/18_supplementary_o256_base_plus15k_then_refresh500_seed61_20260630"
PLUS15K_BASE_CKPT="$BASE_EXP_ROOT/workspace/o256_base_plus15k/ckpts/CoCo_ddpm_final.pth.tar"
PLUS15K_BASE_EVAL_WM="$BASE_EXP_ROOT/eval_outputs/o256_base_plus15k/finaltest1000/CoCo"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"
CONFIG_NAME="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_plus20k_seed61_supp_20260630.yml"

PLUS20K_WORKSPACE="$EXP_ROOT/workspace/o256_base_plus20k"
PLUS20K_CKPT="$PLUS20K_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
PLUS20K_BASE_OUTPUT="$EXP_ROOT/eval_outputs/o256_base_plus20k"
PLUS20K_BASE_WM="$PLUS20K_BASE_OUTPUT/finaltest1000/CoCo"

PLUS20K_RGB_REUSE="$EXP_ROOT/refinement/o256_rgb_plus20k_reuse"
PAIR_OUTPUT_ROOT="$EXP_ROOT/train_pairs/o256_base_plus20k_pairs"
TRAIN_WM="$PAIR_OUTPUT_ROOT/train10000/CoCo"
REFRESH500_DIR="$EXP_ROOT/refinement/o256_rgb_plus20k_refresh500_targeted"
APPLY500_DIR="$EXP_ROOT/apply/o256_rgb_plus20k_refresh500_targeted_eval1000"

BASE_ATTACK_RUN="o256_base_plus20k_nongeom"
RGB_REUSE_ATTACK_RUN="o256_rgb_plus20k_reuse_nongeom"
REFRESH500_ATTACK_RUN="o256_rgb_plus20k_refresh500_targeted_nongeom"

MIN_FREE_GB=9
EST_REQUIRED_GB=7

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips" "$EXP_ROOT/workspace" "$EXP_ROOT/refinement" "$EXP_ROOT/apply" "$EXP_ROOT/train_pairs"

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
src = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml")
dst = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/configs/$CONFIG_NAME")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["data"]["output_dir"] = "$PLUS20K_WORKSPACE"
cfg["training"]["n_epochs"] = 10
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst)
PY
}

require_path "$PLUS15K_BASE_CKPT"
require_path "$PLUS15K_BASE_EVAL_WM"
require_path "$FORMAL_REFINER"
require_path "configs/$TRAIN_PAIR_CONFIG"
require_path "configs/$EVAL_CONFIG"
require_path "configs/$EVAL1000_CONFIG"
require_path "$TRAIN_GT"
require_path "$FINAL_GT"

generate_train_config >/dev/null

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    exit 1
  fi
  echo "O256 +20k base then refresh500 preflight passed."
  echo "Base resume ckpt: $PLUS15K_BASE_CKPT"
  echo "Will run sequentially:"
  echo "  (1) continue +15k base to +20k total (50k total steps)"
  echo "  (2) evaluate O256-Base +20k"
  echo "  (3) evaluate O256-RGB +20k with reused formal refiner"
  echo "  (4) generate +20k train10000 refinement pairs"
  echo "  (5) targeted refiner refresh 500 steps"
  echo "  (6) evaluate refreshed O256-RGB on finaltest1000 + full 11 attacks"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/6: continue +15k base to +20k total"
if [[ -f "$PLUS20K_CKPT" ]]; then
  echo "Reuse trained +20k checkpoint."
elif [[ -d "$PLUS20K_WORKSPACE" ]]; then
  echo "Existing incomplete workspace detected: $PLUS20K_WORKSPACE"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$CONFIG_NAME" \
    --resume "$PLUS15K_BASE_CKPT" \
    --image_folder "results/validation/o256_base_plus20k" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_train.log"
fi
[[ -f "$PLUS20K_CKPT" ]] || { echo "Missing expected +20k checkpoint: $PLUS20K_CKPT"; exit 1; }

echo
echo "[$(date '+%F %T')] Step 2/6: evaluate O256-Base +20k"
if [[ -d "$PLUS20K_BASE_WM" ]]; then
  png_count="$(count_pngs "$PLUS20K_BASE_WM")"
  msg_count="$(count_msgs "$PLUS20K_BASE_WM")"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete +20k base eval output: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse completed +20k base eval outputs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$PLUS20K_CKPT" \
    --image_folder "$PLUS20K_BASE_OUTPUT" \
    --run_name finaltest1000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_base_eval.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_base_psnr_ssim.log"

run_lpips "$PLUS20K_BASE_WM" "$EXP_ROOT/lpips/o256_base_plus20k" "$LOG_DIR/plus20k_base_lpips.log"

mkdir -p "$EXP_ROOT/attacked_Result/clean_o256_base_plus20k"
ln -sfn "$(realpath "$PLUS20K_BASE_WM")" "$EXP_ROOT/attacked_Result/clean_o256_base_plus20k/Clean"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/clean_o256_base_plus20k" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_base_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete +20k base attack output."
    exit 1
  fi
  echo "Reuse completed +20k base attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$PLUS20K_BASE_WM" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$BASE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_base_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_base_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 3/6: evaluate O256-RGB +20k with reused formal refiner"
if [[ -d "$PLUS20K_RGB_REUSE/refined_images" && -f "$PLUS20K_RGB_REUSE/report.txt" ]]; then
  reuse_png="$(count_pngs "$PLUS20K_RGB_REUSE/refined_images")"
  if [[ "$reuse_png" -ne 1000 || ! -d "$PLUS20K_RGB_REUSE/attacked_clean/Clean" ]]; then
    echo "Incomplete +20k rgb reuse output."
    exit 1
  fi
  echo "Reuse completed +20k rgb reuse outputs."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$PLUS20K_CKPT" \
    --refiner "$FORMAL_REFINER" \
    --watermarked_dir "$PLUS20K_BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$PLUS20K_RGB_REUSE" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_rgb_reuse_apply.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$PLUS20K_RGB_REUSE/refined_images" \
  2>&1 | tee "$LOG_DIR/plus20k_rgb_reuse_psnr_ssim.log"
run_lpips "$PLUS20K_RGB_REUSE/refined_images" "$EXP_ROOT/lpips/o256_rgb_plus20k_reuse" "$LOG_DIR/plus20k_rgb_reuse_lpips.log"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$PLUS20K_RGB_REUSE/attacked_clean" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_rgb_reuse_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete +20k rgb reuse attack output."
    exit 1
  fi
  echo "Reuse completed +20k rgb reuse attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$PLUS20K_RGB_REUSE/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$RGB_REUSE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_rgb_reuse_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_rgb_reuse_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$RGB_REUSE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 4/6: generate +20k train10000 refinement pairs"
if [[ -d "$TRAIN_WM" ]]; then
  pair_pngs="$(count_pngs "$TRAIN_WM")"
  pair_msgs="$(count_msgs "$TRAIN_WM")"
  if [[ "$pair_pngs" -ne 10000 || "$pair_msgs" -ne 10000 ]]; then
    echo "Incomplete +20k train pairs: images=$pair_pngs messages=$pair_msgs"
    exit 1
  fi
  echo "Reuse completed +20k train pairs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$TRAIN_PAIR_CONFIG" \
    --resume "$PLUS20K_CKPT" \
    --image_folder "$PAIR_OUTPUT_ROOT" \
    --run_name train10000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_train_pairs_eval.log"
fi

echo
echo "[$(date '+%F %T')] Step 5/6: targeted refresh500 on +20k base"
if [[ -d "$REFRESH500_DIR" ]]; then
  if [[ ! -f "$REFRESH500_DIR/refiner_final.pth" || ! -f "$REFRESH500_DIR/report.txt" ]]; then
    echo "Incomplete existing +20k refresh500 refiner output: $REFRESH500_DIR"
    exit 1
  fi
  echo "Reuse completed +20k refresh500 refiner."
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$PLUS20K_CKPT" \
    --resume_refiner "$FORMAL_REFINER" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
    --output_dir "$REFRESH500_DIR" \
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
    2>&1 | tee "$LOG_DIR/plus20k_refresh500_train.log"
fi

if [[ -d "$APPLY500_DIR/refined_images" && -f "$APPLY500_DIR/report.txt" ]]; then
  apply_pngs="$(count_pngs "$APPLY500_DIR/refined_images")"
  if [[ "$apply_pngs" -ne 1000 || ! -d "$APPLY500_DIR/attacked_clean/Clean" ]]; then
    echo "Incomplete +20k refresh500 apply output."
    exit 1
  fi
  echo "Reuse completed +20k refresh500 apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$PLUS20K_CKPT" \
    --refiner "$REFRESH500_DIR/refiner_final.pth" \
    --watermarked_dir "$PLUS20K_BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$APPLY500_DIR" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_refresh500_apply.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY500_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/plus20k_refresh500_psnr_ssim.log"
run_lpips "$APPLY500_DIR/refined_images" "$EXP_ROOT/lpips/o256_rgb_plus20k_refresh500" "$LOG_DIR/plus20k_refresh500_lpips.log"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$APPLY500_DIR/attacked_clean" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_refresh500_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$REFRESH500_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$REFRESH500_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete +20k refresh500 attack output."
    exit 1
  fi
  echo "Reuse completed +20k refresh500 attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$APPLY500_DIR/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$REFRESH500_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/plus20k_refresh500_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$REFRESH500_ATTACK_RUN" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  2>&1 | tee "$LOG_DIR/plus20k_refresh500_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$REFRESH500_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$REFRESH500_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 6/6: summarize"
"$PY" scripts/summarize_o256_base_plus20k_then_refresh500_20260630.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
