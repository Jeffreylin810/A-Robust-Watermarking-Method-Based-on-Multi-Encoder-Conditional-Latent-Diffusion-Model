#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"

FORMAL_TRAIN_CONFIG="configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml"
FORMAL_EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
FORMAL_BASE_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
FORMAL_REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/16_supplementary_o256_base_continuation_ba_push_seed61_20260629"
GEN_CONFIG_DIR="$ROOT/configs"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"

PLUS5K_TRAIN_CONFIG="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_plus5k_seed61_supp_20260629.yml"
PLUS10K_TRAIN_CONFIG="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_plus10k_seed61_supp_20260629.yml"

PLUS5K_WORKSPACE="$EXP_ROOT/workspace/o256_base_plus5k"
PLUS10K_WORKSPACE="$EXP_ROOT/workspace/o256_base_plus10k"
PLUS5K_CKPT="$PLUS5K_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
PLUS10K_CKPT="$PLUS10K_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"

PLUS5K_BASE_OUTPUT="$EXP_ROOT/eval_outputs/o256_base_plus5k"
PLUS10K_BASE_OUTPUT="$EXP_ROOT/eval_outputs/o256_base_plus10k"

PLUS5K_BASE_WM="$PLUS5K_BASE_OUTPUT/finaltest1000/CoCo"
PLUS10K_BASE_WM="$PLUS10K_BASE_OUTPUT/finaltest1000/CoCo"

PLUS5K_REFINE_OUTPUT="$EXP_ROOT/refinement/o256_rgb_plus5k"
PLUS10K_REFINE_OUTPUT="$EXP_ROOT/refinement/o256_rgb_plus10k"

MIN_FREE_GB=15
EST_REQUIRED_GB=11

cd "$ROOT"
mkdir -p "$LOG_DIR" "$EXP_ROOT" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips" "$EXP_ROOT/workspace"

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

require_file() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

count_images() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l
}

count_messages() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l
}

generate_train_config() {
  local out_name="$1"
  local output_dir="$2"
  local n_epochs="$3"
  "$PY" - <<PY
from pathlib import Path
import yaml

src = Path("$FORMAL_TRAIN_CONFIG")
dst = Path("$GEN_CONFIG_DIR") / "$out_name"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["data"]["output_dir"] = "$output_dir"
cfg["training"]["n_epochs"] = int($n_epochs)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst)
PY
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

run_base_variant() {
  local tag="$1"
  local train_config="$2"
  local resume_ckpt="$3"
  local train_output="$4"
  local final_ckpt="$5"
  local base_output="$6"
  local base_wm="$7"
  local refine_output="$8"

  local base_attack_run="o256_base_${tag}_nongeom"
  local rgb_attack_run="o256_rgb_${tag}_nongeom"
  local clean_base_dir="$EXP_ROOT/attacked_Result/clean_o256_base_${tag}"

  echo
  echo "[$(date '+%F %T')] ===== Variant: $tag ====="

  if [[ -f "$final_ckpt" ]]; then
    echo "Reuse trained checkpoint: $final_ckpt"
  elif [[ -d "$train_output" ]]; then
    echo "Existing incomplete training output detected: $train_output"
    echo "Inspect it before retrying; this script will not overwrite it."
    exit 1
  else
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$train_config" \
      --resume "$resume_ckpt" \
      --image_folder "results/validation/${tag}" \
      --seed "$SEED" \
      2>&1 | tee "$LOG_DIR/${tag}_train.log"

    [[ -f "$final_ckpt" ]] || { echo "Training finished without expected checkpoint: $final_ckpt"; exit 1; }
  fi

  if [[ -d "$base_wm" ]]; then
    local png_count msg_count
    png_count="$(count_images "$base_wm")"
    msg_count="$(count_messages "$base_wm")"
    if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
      echo "Incomplete base eval output for $tag: images=$png_count messages=$msg_count"
      exit 1
    fi
    echo "Reuse completed base eval outputs for $tag."
  else
    /usr/bin/time -p "$PY" -u eval_diffusion.py \
      --config "$FORMAL_EVAL_CONFIG" \
      --resume "$final_ckpt" \
      --image_folder "$base_output" \
      --run_name finaltest1000 \
      --seed "$SEED" \
      2>&1 | tee "$LOG_DIR/${tag}_base_eval.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$base_wm" \
    2>&1 | tee "$LOG_DIR/${tag}_base_psnr_ssim.log"

  run_lpips "$base_wm" "$EXP_ROOT/lpips/o256_base_${tag}" "$LOG_DIR/${tag}_base_lpips.log"

  mkdir -p "$clean_base_dir"
  ln -sfn "$(realpath "$base_wm")" "$clean_base_dir/Clean"
  "$PY" calculate_ba.py \
    --config "$FORMAL_EVAL_CONFIG" \
    --resume "$final_ckpt" \
    --attack_dir "$clean_base_dir" \
    --gt_message_dir "$base_wm" \
    2>&1 | tee "$LOG_DIR/${tag}_base_clean_ba.log"

  if [[ -d "$EXP_ROOT/attacked_Result/$base_attack_run" ]]; then
    local attack_count
    attack_count="$(find "$EXP_ROOT/attacked_Result/$base_attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete base attack output for $tag"
      exit 1
    fi
    echo "Reuse completed base nongeom attacks for $tag."
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$base_wm" \
      --output_dir "$EXP_ROOT/attacked_Result" \
      --run_name "$base_attack_run" \
      --attack_scope nongeom \
      --seed "$SEED" \
      2>&1 | tee "$LOG_DIR/${tag}_base_attack.log"
  fi

  "$PY" calculate_ba.py \
    --config "$FORMAL_EVAL_CONFIG" \
    --resume "$final_ckpt" \
    --attack_dir "$EXP_ROOT/attacked_Result/$base_attack_run" \
    --gt_message_dir "$base_wm" \
    2>&1 | tee "$LOG_DIR/${tag}_base_attack_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "$EXP_ROOT/attacked_Result/$base_attack_run/ba_report.txt" \
    --output "$EXP_ROOT/attacked_Result/$base_attack_run/ba_summary.txt"

  if [[ -d "$refine_output/refined_images" && -f "$refine_output/report.txt" ]]; then
    local refine_png_count
    refine_png_count="$(count_images "$refine_output/refined_images")"
    if [[ "$refine_png_count" -ne 1000 || ! -d "$refine_output/attacked_clean/Clean" ]]; then
      echo "Incomplete refined output for $tag"
      exit 1
    fi
    echo "Reuse completed refined outputs for $tag."
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$FORMAL_EVAL_CONFIG" \
      --resume "$final_ckpt" \
      --refiner "$FORMAL_REFINER" \
      --watermarked_dir "$base_wm" \
      --gt_dir "$FINAL_GT" \
      --output_dir "$refine_output" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed "$SEED" \
      2>&1 | tee "$LOG_DIR/${tag}_rgb_apply.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$refine_output/refined_images" \
    2>&1 | tee "$LOG_DIR/${tag}_rgb_psnr_ssim.log"

  run_lpips "$refine_output/refined_images" "$EXP_ROOT/lpips/o256_rgb_${tag}" "$LOG_DIR/${tag}_rgb_lpips.log"

  "$PY" calculate_ba.py \
    --config "$FORMAL_EVAL_CONFIG" \
    --resume "$final_ckpt" \
    --attack_dir "$refine_output/attacked_clean" \
    --gt_message_dir "$base_wm" \
    2>&1 | tee "$LOG_DIR/${tag}_rgb_clean_ba.log"

  if [[ -d "$EXP_ROOT/attacked_Result/$rgb_attack_run" ]]; then
    local refine_attack_count
    refine_attack_count="$(find "$EXP_ROOT/attacked_Result/$rgb_attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$refine_attack_count" -ne 11 ]]; then
      echo "Incomplete refined attack output for $tag"
      exit 1
    fi
    echo "Reuse completed refined nongeom attacks for $tag."
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$refine_output/refined_images" \
      --output_dir "$EXP_ROOT/attacked_Result" \
      --run_name "$rgb_attack_run" \
      --attack_scope nongeom \
      --seed "$SEED" \
      2>&1 | tee "$LOG_DIR/${tag}_rgb_attack.log"
  fi

  "$PY" calculate_ba.py \
    --config "$FORMAL_EVAL_CONFIG" \
    --resume "$final_ckpt" \
    --attack_dir "$EXP_ROOT/attacked_Result/$rgb_attack_run" \
    --gt_message_dir "$base_wm" \
    2>&1 | tee "$LOG_DIR/${tag}_rgb_attack_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "$EXP_ROOT/attacked_Result/$rgb_attack_run/ba_report.txt" \
    --output "$EXP_ROOT/attacked_Result/$rgb_attack_run/ba_summary.txt"
}

require_file "$PY"
require_file "$FORMAL_TRAIN_CONFIG"
require_file "$ROOT/configs/$FORMAL_EVAL_CONFIG"
require_file "$FORMAL_BASE_CKPT"
require_file "$FORMAL_REFINER"
require_file "$FINAL_GT"

finaltest_count="$(find -L "$FINAL_GT" -maxdepth 1 -type f | wc -l)"
if [[ "$finaltest_count" -ne 1000 ]]; then
  echo "Expected 1000 locked finaltest images, found: $finaltest_count"
  exit 1
fi

free_gb="$("$PY" - <<'PY'
import os
st = os.statvfs("/home/jeffrey/projects")
print(f"{st.f_bavail * st.f_frsize / (1024**3):.2f}")
PY
)"

if [[ "$CHECK_ONLY" == true ]]; then
  echo "O256 base continuation BA-push preflight passed."
  echo "Will run sequentially:"
  echo "  (1) +5k continuation (30k -> 35k)"
  echo "  (2) +10k continuation (35k -> 40k total)"
  echo "Each variant evaluates both:"
  echo "  - O256-Base"
  echo "  - O256-RGB (reuse existing formal refiner)"
  echo "Estimated extra disk: ~11G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  if "$PY" - <<PY
free_gb = float("$free_gb")
raise SystemExit(0 if free_gb >= $MIN_FREE_GB else 1)
PY
  then
    echo "Disk check: OK"
    exit 0
  else
    echo "Disk check: insufficient free space (${free_gb}G available, need at least ${MIN_FREE_GB}G)."
    exit 1
  fi
fi

"$PY" - <<PY
free_gb = float("$free_gb")
if free_gb < $MIN_FREE_GB:
    raise SystemExit(f"Insufficient free space: {free_gb:.2f}G available, require at least ${MIN_FREE_GB}G.")
PY

generate_train_config "$PLUS5K_TRAIN_CONFIG" "$PLUS5K_WORKSPACE" 7 >/dev/null
generate_train_config "$PLUS10K_TRAIN_CONFIG" "$PLUS10K_WORKSPACE" 8 >/dev/null

echo "[$(date '+%F %T')] Start O256 base continuation BA-push bundle"
echo "Experiment root: $EXP_ROOT"

run_base_variant \
  "plus5k" \
  "$PLUS5K_TRAIN_CONFIG" \
  "$FORMAL_BASE_CKPT" \
  "$PLUS5K_WORKSPACE" \
  "$PLUS5K_CKPT" \
  "$PLUS5K_BASE_OUTPUT" \
  "$PLUS5K_BASE_WM" \
  "$PLUS5K_REFINE_OUTPUT"

run_base_variant \
  "plus10k" \
  "$PLUS10K_TRAIN_CONFIG" \
  "$PLUS5K_CKPT" \
  "$PLUS10K_WORKSPACE" \
  "$PLUS10K_CKPT" \
  "$PLUS10K_BASE_OUTPUT" \
  "$PLUS10K_BASE_WM" \
  "$PLUS10K_REFINE_OUTPUT"

"$PY" scripts/summarize_o256_base_continuation_ba_push_20260629.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
