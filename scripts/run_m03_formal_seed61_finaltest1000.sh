#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
BASE_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

FINAL_GT="$FORMAL_ROOT/data/CoCo/val/gt"
BASE_OUTPUT="results/test_result/m03_formal_seed61_base_finaltest1000"
BASE_WM="$BASE_OUTPUT/finaltest1000/CoCo"
REFINE_OUTPUT="results/refinement/m03_formal_seed61_fullmodule_finaltest1000_apply"

BASE_ATTACK_RUN="m03_formal_seed61_base_finaltest1000_nongeom"
REFINE_ATTACK_RUN="m03_formal_seed61_fullmodule_finaltest1000_nongeom"
SUMMARY="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_finaltest1000/summary.md"

cd "$ROOT"
mkdir -p logs

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$PY" "configs/$CONFIG" "$BASE_CKPT" "$REFINER" "$FINAL_GT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

finaltest_count="$(find -L "$FINAL_GT" -maxdepth 1 -type f | wc -l)"
if [[ "$finaltest_count" -ne 1000 ]]; then
  echo "Expected 1000 locked finaltest images, found: $finaltest_count"
  exit 1
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "M03 preflight passed."
  echo "Classification: F (formal chapter-4 main results)"
  echo "Frozen base: $BASE_CKPT"
  echo "Frozen module: $REFINER"
  echo "Locked final test root: $FORMAL_ROOT"
  echo "This step evaluates finaltest1000 exactly once for:"
  echo "  (1) base only"
  echo "  (2) base + RGB refinement module"
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

echo "[$(date '+%F %T')] Start M03 formal seed61 finaltest1000"
echo "Classification: F"

echo
echo "[$(date '+%F %T')] Step 1/8: base-only finaltest1000 eval"
if [[ -e "$BASE_WM" ]]; then
  png_count="$(find "$BASE_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
  msg_count="$(find "$BASE_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete base finaltest output: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse completed base finaltest outputs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --image_folder "$BASE_OUTPUT" \
    --run_name finaltest1000 \
    --seed 61 \
    2>&1 | tee logs/m03_formal_seed61_base_finaltest1000_eval.log
fi

echo
echo "[$(date '+%F %T')] Step 2/8: base-only finaltest1000 PSNR / clean BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$BASE_WM" \
  2>&1 | tee logs/m03_formal_seed61_base_finaltest1000_psnr_ssim.log

mkdir -p results/attacked_Result/clean_m03_formal_seed61_base_finaltest1000/Clean
find "$BASE_WM" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.webp' \) -print0 |
  while IFS= read -r -d '' image; do
    target="results/attacked_Result/clean_m03_formal_seed61_base_finaltest1000/Clean/$(basename "$image")"
    [[ -e "$target" ]] || ln "$image" "$target"
  done

"$PY" calculate_ba.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir results/attacked_Result/clean_m03_formal_seed61_base_finaltest1000 \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee logs/m03_formal_seed61_base_finaltest1000_clean_ba.log

echo
echo "[$(date '+%F %T')] Step 3/8: base-only finaltest1000 nongeom attacks"
if [[ -e "results/attacked_Result/$BASE_ATTACK_RUN" ]]; then
  attack_count="$(find "results/attacked_Result/$BASE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete base attack output."
    exit 1
  fi
  echo "Reuse completed base nongeom attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$BASE_WM" \
    --output_dir results/attacked_Result \
    --run_name "$BASE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed 61 \
    2>&1 | tee logs/m03_formal_seed61_base_finaltest1000_attack.log
fi

"$PY" calculate_ba.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "results/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee logs/m03_formal_seed61_base_finaltest1000_ba.log

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 4/8: apply full module to finaltest1000"
if [[ -e "$REFINE_OUTPUT" ]]; then
  refine_png_count="$(find "$REFINE_OUTPUT/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  if [[ ! -f "$REFINE_OUTPUT/report.txt" || "$refine_png_count" -ne 1000 || ! -d "$REFINE_OUTPUT/attacked_clean/Clean" ]]; then
    echo "Incomplete refined finaltest output."
    exit 1
  fi
  echo "Reuse completed refined finaltest outputs."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --refiner "$REFINER" \
    --watermarked_dir "$BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$REFINE_OUTPUT" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed 61 \
    2>&1 | tee logs/m03_formal_seed61_fullmodule_finaltest1000_apply.log
fi

echo
echo "[$(date '+%F %T')] Step 5/8: refined finaltest1000 PSNR / clean BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$REFINE_OUTPUT/refined_images" \
  2>&1 | tee logs/m03_formal_seed61_fullmodule_finaltest1000_psnr_ssim.log

"$PY" calculate_ba.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "$REFINE_OUTPUT/attacked_clean" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee logs/m03_formal_seed61_fullmodule_finaltest1000_clean_ba.log

echo
echo "[$(date '+%F %T')] Step 6/8: refined finaltest1000 nongeom attacks"
if [[ -e "results/attacked_Result/$REFINE_ATTACK_RUN" ]]; then
  refine_attack_count="$(find "results/attacked_Result/$REFINE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$refine_attack_count" -ne 11 ]]; then
    echo "Incomplete refined attack output."
    exit 1
  fi
  echo "Reuse completed refined nongeom attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$REFINE_OUTPUT/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$REFINE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed 61 \
    2>&1 | tee logs/m03_formal_seed61_fullmodule_finaltest1000_attack.log
fi

"$PY" calculate_ba.py \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "results/attacked_Result/$REFINE_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee logs/m03_formal_seed61_fullmodule_finaltest1000_ba.log

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$REFINE_ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$REFINE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 7/8: write formal summary"
"$PY" scripts/summarize_m03_formal_finaltest.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] Step 8/8: done"
echo "Summary: $SUMMARY"
