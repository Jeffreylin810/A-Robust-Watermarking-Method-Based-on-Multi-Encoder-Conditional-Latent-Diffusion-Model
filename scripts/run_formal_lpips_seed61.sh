#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
PIP="/home/jeffrey/miniconda3/envs/diffw/bin/pip"

GT_DIR="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"
DIFFW_DIR="$DIFFW_ROOT/results/test_result/m04_formal_diffw_like_seed61/finaltest1000/CoCo/CoCo"
BASE_DIR="$ROOT/results/test_result/m03_formal_seed61_base_finaltest1000/finaltest1000/CoCo"
REFINED_DIR="$ROOT/results/refinement/m03_formal_seed61_fullmodule_finaltest1000_apply/refined_images"
OUT_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_lpips_seed61"

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

required=(
  "$PY"
  "$PIP"
  "$GT_DIR"
  "$DIFFW_DIR"
  "$BASE_DIR"
  "$REFINED_DIR"
  "$ROOT/scripts/calculate_lpips_dir.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Formal LPIPS preflight passed."
  echo "Classification: F-support (formal metric supplement)"
  echo "NOTE: LPIPS package is currently not installed in diffw env."
  echo "Install command:"
  echo "  $PIP install lpips"
  echo "Output root: $OUT_ROOT"
  exit 0
fi

echo "Step 0/3: ensure lpips package"
"$PIP" install lpips

echo "Step 1/3: DiffW-like LPIPS"
"$PY" "$ROOT/scripts/calculate_lpips_dir.py" \
  --gt_dir "$GT_DIR" \
  --results_dir "$DIFFW_DIR" \
  --output_dir "$OUT_ROOT/diffw_like" \
  --batch_size 8 \
  --net alex \
  --device auto

echo "Step 2/3: O256 base LPIPS"
"$PY" "$ROOT/scripts/calculate_lpips_dir.py" \
  --gt_dir "$GT_DIR" \
  --results_dir "$BASE_DIR" \
  --output_dir "$OUT_ROOT/o256_base" \
  --batch_size 8 \
  --net alex \
  --device auto

echo "Step 3/3: O256-RGBREFINE LPIPS"
"$PY" "$ROOT/scripts/calculate_lpips_dir.py" \
  --gt_dir "$GT_DIR" \
  --results_dir "$REFINED_DIR" \
  --output_dir "$OUT_ROOT/o256_rgbrefine" \
  --batch_size 8 \
  --net alex \
  --device auto

echo "Done. Output root: $OUT_ROOT"
