#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
GT_DIR="$FORMAL_ROOT/data/CoCo/val/gt"
BASE_DIR="$ROOT/results/test_result/m03_formal_seed61_base_finaltest1000/finaltest1000/CoCo"
REFINED_DIR="$ROOT/results/refinement/m03_formal_seed61_fullmodule_finaltest1000_apply/refined_images"

OUT_DIR="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/05_qualitative/formal_seed61_main_qualitative"

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

cd "$ROOT"
required=(
  "$PY"
  "$GT_DIR"
  "$BASE_DIR"
  "$REFINED_DIR"
  "scripts/build_formal_qualitative_seed61.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

gt_count="$(find -L "$GT_DIR" -maxdepth 1 -type f | wc -l)"
base_count="$(find "$BASE_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"
refined_count="$(find "$REFINED_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"

if [[ "$gt_count" -ne 1000 ]]; then
  echo "Expected 1000 GT images, found: $gt_count"
  exit 1
fi
if [[ "$base_count" -ne 1000 ]]; then
  echo "Expected 1000 base output images, found: $base_count"
  exit 1
fi
if [[ "$refined_count" -ne 1000 ]]; then
  echo "Expected 1000 refined output images, found: $refined_count"
  exit 1
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "M05 preflight passed."
  echo "Classification: F (chapter-4 qualitative evidence)"
  echo "GT dir: $GT_DIR"
  echo "Base dir: $BASE_DIR"
  echo "Refined dir: $REFINED_DIR"
  echo "Output dir: $OUT_DIR"
  exit 0
fi

echo "[$(date '+%F %T')] Start M05 formal qualitative seed61"
echo "Classification: F"
echo
echo "[$(date '+%F %T')] Step 1/1: build fixed qualitative panels"
"$PY" scripts/build_formal_qualitative_seed61.py \
  --gt_dir "$GT_DIR" \
  --base_dir "$BASE_DIR" \
  --refined_dir "$REFINED_DIR" \
  --output_dir "$OUT_DIR" \
  --diff_gain 12 \
  --num_samples 9

echo "[$(date '+%F %T')] M05 complete"
echo "Output: $OUT_DIR"
