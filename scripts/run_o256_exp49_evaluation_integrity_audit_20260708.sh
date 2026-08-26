#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
CHECK_ONLY=0
MAX_IMAGES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --quick)
      if [[ $# -lt 2 ]]; then
        echo "--quick requires a number" >&2
        exit 2
      fi
      MAX_IMAGES="$2"
      shift 2
      ;;
    --max-images)
      if [[ $# -lt 2 ]]; then
        echo "--max-images requires a number" >&2
        exit 2
      fi
      MAX_IMAGES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/49_supplementary_o256_evaluation_integrity_audit_20260708"
OUT_DIR="$EXP_ROOT/audit"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

FINAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"

EXP40="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/40_formal_o256_full_from_scratch_dropout_margin_seed61_20260704"
EXP41="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"

CFG40="CoCo_train10000_ours256_msg32_exp40_base65_dropout_margin_plus5k_seed61_20260704.yml"
CFG41="CoCo_train10000_ours256_msg32_exp41_base65_dropout_margin_plus5k_seed62_20260705.yml"
CKPT40="$EXP40/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
CKPT41="$EXP41/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"

EXP40_BASE_CLEAN="$EXP40/eval_outputs/exp40_base65_dropout_margin/finaltest1000/CoCo"
EXP40_RGB_CLEAN="$EXP40/apply/exp40_rgb_full_from_scratch_dropout_balanced_finaltest1000/refined_images"
EXP40_RGB_DROP="$EXP40/attacked_Result/exp40_rgb_full_from_scratch_dropout_balanced_nongeom/Paper_Dropout_r03"

EXP41_BASE_CLEAN="$EXP41/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo"
EXP41_RGB_CLEAN="$EXP41/apply/exp41_rgb_full_from_scratch_dropout_balanced_finaltest1000/refined_images"
EXP41_RGB_DROP="$EXP41/attacked_Result/exp41_rgb_full_from_scratch_dropout_balanced_nongeom/Paper_Dropout_r03"

ARGS=(
  --output_dir "$OUT_DIR"
  --max_images "$MAX_IMAGES"
  --sample "exp40_base_clean|base_clean|$CFG40|$CKPT40|$EXP40_BASE_CLEAN|$EXP40_BASE_CLEAN|$FINAL_GT"
  --sample "exp40_rgb_clean|rgb_clean|$CFG40|$CKPT40|$EXP40_RGB_CLEAN|$EXP40_RGB_CLEAN|$FINAL_GT"
  --sample "exp40_rgb_dropout|rgb_dropout|$CFG40|$CKPT40|$EXP40_RGB_DROP|$EXP40_RGB_CLEAN|$FINAL_GT"
  --sample "exp41_base_clean|base_clean|$CFG41|$CKPT41|$EXP41_BASE_CLEAN|$EXP41_BASE_CLEAN|$FINAL_GT"
  --sample "exp41_rgb_clean|rgb_clean|$CFG41|$CKPT41|$EXP41_RGB_CLEAN|$EXP41_RGB_CLEAN|$FINAL_GT"
  --sample "exp41_rgb_dropout|rgb_dropout|$CFG41|$CKPT41|$EXP41_RGB_DROP|$EXP41_RGB_CLEAN|$FINAL_GT"
)

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  ARGS+=(--check-only)
fi

echo "[$(date '+%F %T')] exp49 evaluation integrity audit"
echo "Experiment root: $EXP_ROOT"
echo "Max images: $MAX_IMAGES"
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  "$PY" scripts/analyze_o256_exp49_evaluation_integrity_audit.py "${ARGS[@]}"
else
  { time "$PY" scripts/analyze_o256_exp49_evaluation_integrity_audit.py "${ARGS[@]}"; } \
    2>&1 | tee "$LOG_DIR/exp49_evaluation_integrity_audit.log"
  echo
  echo "[$(date '+%F %T')] Done."
  echo "Summary md: $OUT_DIR/summary.md"
  echo "Audit csv: $OUT_DIR/evaluation_integrity_audit.csv"
fi
