#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

EXP41_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
EXP44_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/44_supplementary_o256_train_hardcase_dropout_replay_seed62_20260705"
EXP45_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/45_supplementary_o256_exp44_dropout_floor_delta_diagnosis_seed62_20260706"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/46_supplementary_o256_residual_frequency_diagnosis_seed62_20260706"

BASE_DIR="$EXP41_ROOT/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo"
EXP41_RGB_DIR="$EXP41_ROOT/apply/exp41_rgb_full_from_scratch_dropout_balanced_finaltest1000/refined_images"
EXP44_RGB_DIR="$EXP44_ROOT/apply/exp44_hardcase_dropout_replay500_finaltest1000/refined_images"
FINAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"
DELTA_CSV="$EXP45_ROOT/delta/per_image_delta.csv"

OUT_DIR="$EXP_ROOT/residual_frequency"
LOG_DIR="$EXP_ROOT/logs"

MIN_FREE_GB=2
EST_REQUIRED_GB=1

CHECK_ONLY=false
MAX_IMAGES=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ "${1:-}" == "--quick" ]]; then
  MAX_IMAGES="${2:-100}"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only] [--quick N]"
  exit 2
fi

cd "$ROOT"

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

count_images() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l
}

count_csv_rows() {
  local path="$1"
  awk 'NR > 1 {count += 1} END {print count + 0}' "$path"
}

require_image_dir() {
  local label="$1"
  local dir="$2"
  require_path "$dir"
  local count
  count="$(count_images "$dir")"
  if [[ "$count" -lt 1 ]]; then
    echo "No images for $label: $dir"
    exit 1
  fi
  if [[ "$MAX_IMAGES" -eq 0 && "$count" -ne 1000 ]]; then
    echo "Unexpected image count for $label: $count (expected 1000)"
    exit 1
  fi
  echo "$label: images=$count"
}

require_path "$PY"
require_path "scripts/analyze_o256_exp46_residual_frequency_diagnosis.py"
require_image_dir "exp41_base" "$BASE_DIR"
require_image_dir "exp41_rgb" "$EXP41_RGB_DIR"
require_image_dir "exp44_rgb" "$EXP44_RGB_DIR"
require_image_dir "final_gt" "$FINAL_GT"
require_path "$DELTA_CSV"

delta_rows="$(count_csv_rows "$DELTA_CSV")"
if [[ "$MAX_IMAGES" -eq 0 && "$delta_rows" -ne 1000 ]]; then
  echo "Unexpected exp45 delta row count: $delta_rows (expected 1000)"
  exit 1
fi
echo "exp45_delta_rows=$delta_rows"

free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
  echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  exit 1
fi

if [[ "$CHECK_ONLY" == true ]]; then
  echo "O256 exp46 residual/frequency diagnosis preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Estimated runtime: about 2-8 minutes"
  echo "Disk check: OK"
  exit 0
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[$(date '+%F %T')] Step 1/1: residual/frequency diagnosis"
"$PY" scripts/analyze_o256_exp46_residual_frequency_diagnosis.py \
  --base_dir "$BASE_DIR" \
  --exp41_rgb_dir "$EXP41_RGB_DIR" \
  --exp44_rgb_dir "$EXP44_RGB_DIR" \
  --gt_dir "$FINAL_GT" \
  --delta_csv "$DELTA_CSV" \
  --output_dir "$OUT_DIR" \
  --max_images "$MAX_IMAGES" \
  --fft_cutoff 0.35 \
  2>&1 | tee "$LOG_DIR/exp46_residual_frequency.log"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $OUT_DIR/summary.md"
echo "Per-image csv: $OUT_DIR/per_image_residual_frequency.csv"
echo "Group summary csv: $OUT_DIR/group_summary.csv"
echo "Correlations csv: $OUT_DIR/correlations.csv"
