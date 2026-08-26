#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

TRAIN_CONFIG="CoCo_train10000_finaltest1000_locked_linear_steps20k_seed61_formal.yml"
EVAL_CONFIG="CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
FINAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"

FINAL_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"
OUTPUT_ROOT="$DIFFW_ROOT/results/test_result/m04_formal_diffw_like_seed61"
OUTPUT_DIR="$OUTPUT_ROOT/finaltest1000/CoCo/CoCo"
ATTACK_RUN="m04_formal_diffw_like_finaltest1000_nongeom"
SUMMARY="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_diffw_like_finaltest1000/summary.md"

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$PY" "$DIFFW_ROOT" "$DIFFW_ROOT/configs/$TRAIN_CONFIG" "$DIFFW_ROOT/configs/$EVAL_CONFIG" "$FINAL_GT" "$ROOT/attack_eval_diffw_paper_matched.py" "$ROOT/scripts/summarize_ba_report.py"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

train_count="$(find -L /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/train/gt -maxdepth 1 -type f | wc -l)"
final_count="$(find -L "$FINAL_GT" -maxdepth 1 -type f | wc -l)"
if [[ "$train_count" -ne 10000 ]]; then
  echo "Expected 10000 locked train images, found: $train_count"
  exit 1
fi
if [[ "$final_count" -ne 1000 ]]; then
  echo "Expected 1000 locked finaltest images, found: $final_count"
  exit 1
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "M04 preflight passed."
  echo "Classification: F (formal chapter-4 baseline results)"
  echo "This step trains a formal-clean DiffW-like baseline and evaluates finaltest1000 exactly once."
  echo "Train config: $TRAIN_CONFIG"
  echo "Eval config: $EVAL_CONFIG"
  echo "DiffW root: $DIFFW_ROOT"
  echo "Free disk:"
  df -h "$DIFFW_ROOT" | tail -n 1
  exit 0
fi

echo "[$(date '+%F %T')] Start M04 formal DiffW-like finaltest1000"
echo "Classification: F"

echo
echo "[$(date '+%F %T')] Step 1/6: formal-clean DiffW-like 20k training"
if [[ -f "$FINAL_CKPT" ]]; then
  echo "Reuse existing formal DiffW-like final checkpoint."
else
  (
    cd "$DIFFW_ROOT"
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$TRAIN_CONFIG" \
      --resume ./ckpts/nonexistent_m04_formal_diffw_like_seed61_start.pth.tar \
      --image_folder results/validation/m04_formal_diffw_like_seed61 \
      --seed 61 \
      2>&1 | tee logs/m04_formal_diffw_like_train.log
    if [[ ! -f ckpts/CoCo_ddpm_final.pth.tar ]]; then
      echo "Missing training output ckpts/CoCo_ddpm_final.pth.tar"
      exit 1
    fi
    cp ckpts/CoCo_ddpm_final.pth.tar "$FINAL_CKPT"
  )
fi

echo
echo "[$(date '+%F %T')] Step 2/6: finaltest1000 evaluation"
if [[ -d "$OUTPUT_DIR" ]]; then
  png_count="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"
  msg_count="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete DiffW-like finaltest output: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse existing DiffW-like finaltest outputs."
else
  (
    cd "$DIFFW_ROOT"
    /usr/bin/time -p "$PY" -u eval_diffusion.py \
      --config "$EVAL_CONFIG" \
      --resume "$FINAL_CKPT" \
      --test_set CoCo \
      --image_folder "$OUTPUT_ROOT" \
      --run_name finaltest1000 \
      --seed 61 \
      2>&1 | tee logs/m04_formal_diffw_like_finaltest1000_eval.log
  )
fi

echo
echo "[$(date '+%F %T')] Step 3/6: PSNR / SSIM and Clean BA"
(
  cd "$DIFFW_ROOT"
  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$OUTPUT_DIR" \
    2>&1 | tee logs/m04_formal_diffw_like_finaltest1000_psnr_ssim.log
)

mkdir -p "$DIFFW_ROOT/results/attacked_Result/clean_m04_formal_diffw_like_finaltest1000/Clean"
find "$OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.webp' \) -print0 |
  while IFS= read -r -d '' image; do
    target="$DIFFW_ROOT/results/attacked_Result/clean_m04_formal_diffw_like_finaltest1000/Clean/$(basename "$image")"
    [[ -e "$target" ]] || ln "$image" "$target"
  done

(
  cd "$DIFFW_ROOT"
  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$FINAL_CKPT" \
    --attack_dir results/attacked_Result/clean_m04_formal_diffw_like_finaltest1000 \
    --gt_message_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/m04_formal_diffw_like_finaltest1000_clean_ba.log
)

echo
echo "[$(date '+%F %T')] Step 4/6: 11 non-geometric attacks"
if [[ -e "$DIFFW_ROOT/results/attacked_Result/$ATTACK_RUN" ]]; then
  attack_count="$(find "$DIFFW_ROOT/results/attacked_Result/$ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete DiffW-like nongeom attack output."
    exit 1
  fi
  echo "Reuse existing DiffW-like nongeom attacks."
else
  (
    cd "$ROOT"
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$OUTPUT_DIR" \
      --output_dir "$DIFFW_ROOT/results/attacked_Result" \
      --run_name "$ATTACK_RUN" \
      --attack_scope nongeom \
      --seed 61 \
      2>&1 | tee "$DIFFW_ROOT/logs/m04_formal_diffw_like_finaltest1000_attack.log"
  )
fi

(
  cd "$DIFFW_ROOT"
  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$FINAL_CKPT" \
    --attack_dir "results/attacked_Result/$ATTACK_RUN" \
    --gt_message_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/m04_formal_diffw_like_finaltest1000_ba.log
)

"$PY" "$ROOT/scripts/summarize_ba_report.py" \
  "$DIFFW_ROOT/results/attacked_Result/$ATTACK_RUN/ba_report.txt" \
  --output "$DIFFW_ROOT/results/attacked_Result/$ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 5/6: write formal summary"
"$PY" "$ROOT/scripts/summarize_m04_formal_diffw_like_finaltest.py" \
  --diffw_root "$DIFFW_ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] Step 6/6: done"
echo "Summary: $SUMMARY"
