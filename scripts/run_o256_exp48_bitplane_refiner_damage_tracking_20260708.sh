#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

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
        echo "--quick requires a number, e.g. --quick 100" >&2
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

PY="${PYTHON:-/home/jeffrey/miniconda3/envs/diffw/bin/python}"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/48_supplementary_o256_bitplane_refiner_damage_tracking_20260708"
LOG_DIR="$EXP_ROOT/logs"
OUT_DIR="$EXP_ROOT/diagnosis"
mkdir -p "$LOG_DIR" "$OUT_DIR"

EXP40="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/40_formal_o256_full_from_scratch_dropout_margin_seed61_20260704"
EXP41="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
EXP44="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/44_supplementary_o256_train_hardcase_dropout_replay_seed62_20260705"
EXP47="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/47_supplementary_o256_guarded_stageD_refiner_probe_seed62_20260706"

CFG40="CoCo_train10000_ours256_msg32_exp40_base65_dropout_margin_plus5k_seed61_20260704.yml"
CFG41="CoCo_train10000_ours256_msg32_exp41_base65_dropout_margin_plus5k_seed62_20260705.yml"
CKPT40="$EXP40/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
CKPT41="$EXP41/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"

EXP40_BASE_CLEAN="$EXP40/eval_outputs/exp40_base65_dropout_margin/finaltest1000/CoCo"
EXP40_BASE_DROP="$EXP40/attacked_Result/exp40_base65_dropout_margin_nongeom/Paper_Dropout_r03"
EXP40_RGB_CLEAN="$EXP40/apply/exp40_rgb_full_from_scratch_dropout_balanced_finaltest1000/refined_images"
EXP40_RGB_DROP="$EXP40/attacked_Result/exp40_rgb_full_from_scratch_dropout_balanced_nongeom/Paper_Dropout_r03"

EXP41_BASE_CLEAN="$EXP41/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo"
EXP41_BASE_DROP="$EXP41/attacked_Result/exp41_base65_dropout_margin_nongeom/Paper_Dropout_r03"
EXP41_RGB_CLEAN="$EXP41/apply/exp41_rgb_full_from_scratch_dropout_balanced_finaltest1000/refined_images"
EXP41_RGB_DROP="$EXP41/attacked_Result/exp41_rgb_full_from_scratch_dropout_balanced_nongeom/Paper_Dropout_r03"

EXP44_RGB_CLEAN="$EXP44/apply/exp44_hardcase_dropout_replay500_finaltest1000/refined_images"
EXP44_RGB_DROP="$EXP44/attacked_Result/exp44_hardcase_dropout_replay500_nongeom/Paper_Dropout_r03"

EXP47A_RGB_CLEAN="$EXP47/apply/exp47A_stageD_mild_residual_guard_finaltest1000/refined_images"
EXP47A_RGB_DROP="$EXP47/attacked_Result/exp47A_stageD_mild_residual_guard_nongeom/Paper_Dropout_r03"
EXP47B_RGB_CLEAN="$EXP47/apply/exp47B_stageD_strong_residual_guard_finaltest1000/refined_images"
EXP47B_RGB_DROP="$EXP47/attacked_Result/exp47B_stageD_strong_residual_guard_nongeom/Paper_Dropout_r03"

ARGS=(
  --output_dir "$OUT_DIR"
  --max_images "$MAX_IMAGES"
  --sample "exp40_base_clean|seed61|base|clean|$CFG40|$CKPT40|$EXP40_BASE_CLEAN|$EXP40_BASE_CLEAN"
  --sample "exp40_base_dropout|seed61|base|dropout_r03|$CFG40|$CKPT40|$EXP40_BASE_DROP|$EXP40_BASE_CLEAN"
  --sample "exp40_rgb_clean|seed61|rgb|clean|$CFG40|$CKPT40|$EXP40_RGB_CLEAN|$EXP40_RGB_CLEAN"
  --sample "exp40_rgb_dropout|seed61|rgb|dropout_r03|$CFG40|$CKPT40|$EXP40_RGB_DROP|$EXP40_RGB_CLEAN"
  --sample "exp41_base_clean|seed62|base|clean|$CFG41|$CKPT41|$EXP41_BASE_CLEAN|$EXP41_BASE_CLEAN"
  --sample "exp41_base_dropout|seed62|base|dropout_r03|$CFG41|$CKPT41|$EXP41_BASE_DROP|$EXP41_BASE_CLEAN"
  --sample "exp41_rgb_clean|seed62|rgb|clean|$CFG41|$CKPT41|$EXP41_RGB_CLEAN|$EXP41_RGB_CLEAN"
  --sample "exp41_rgb_dropout|seed62|rgb|dropout_r03|$CFG41|$CKPT41|$EXP41_RGB_DROP|$EXP41_RGB_CLEAN"
  --sample "exp44_rgb_clean|seed62|rgb_hardcase_replay|clean|$CFG41|$CKPT41|$EXP44_RGB_CLEAN|$EXP44_RGB_CLEAN"
  --sample "exp44_rgb_dropout|seed62|rgb_hardcase_replay|dropout_r03|$CFG41|$CKPT41|$EXP44_RGB_DROP|$EXP44_RGB_CLEAN"
  --sample "exp47A_rgb_clean|seed62|rgb_guard_mild|clean|$CFG41|$CKPT41|$EXP47A_RGB_CLEAN|$EXP47A_RGB_CLEAN"
  --sample "exp47A_rgb_dropout|seed62|rgb_guard_mild|dropout_r03|$CFG41|$CKPT41|$EXP47A_RGB_DROP|$EXP47A_RGB_CLEAN"
  --sample "exp47B_rgb_clean|seed62|rgb_guard_strong|clean|$CFG41|$CKPT41|$EXP47B_RGB_CLEAN|$EXP47B_RGB_CLEAN"
  --sample "exp47B_rgb_dropout|seed62|rgb_guard_strong|dropout_r03|$CFG41|$CKPT41|$EXP47B_RGB_DROP|$EXP47B_RGB_CLEAN"
  --transition "exp40_base_clean_to_rgb_clean|exp40_base_clean|exp40_rgb_clean"
  --transition "exp40_base_dropout_to_rgb_dropout|exp40_base_dropout|exp40_rgb_dropout"
  --transition "exp40_base_clean_to_base_dropout|exp40_base_clean|exp40_base_dropout"
  --transition "exp40_rgb_clean_to_rgb_dropout|exp40_rgb_clean|exp40_rgb_dropout"
  --transition "exp41_base_clean_to_rgb_clean|exp41_base_clean|exp41_rgb_clean"
  --transition "exp41_base_dropout_to_rgb_dropout|exp41_base_dropout|exp41_rgb_dropout"
  --transition "exp41_base_clean_to_base_dropout|exp41_base_clean|exp41_base_dropout"
  --transition "exp41_rgb_clean_to_rgb_dropout|exp41_rgb_clean|exp41_rgb_dropout"
  --transition "cross_seed_base_dropout_exp40_to_exp41|exp40_base_dropout|exp41_base_dropout"
  --transition "cross_seed_rgb_dropout_exp40_to_exp41|exp40_rgb_dropout|exp41_rgb_dropout"
  --transition "exp41_rgb_clean_to_exp44_rgb_clean|exp41_rgb_clean|exp44_rgb_clean"
  --transition "exp41_rgb_dropout_to_exp44_rgb_dropout|exp41_rgb_dropout|exp44_rgb_dropout"
  --transition "exp44_rgb_clean_to_rgb_dropout|exp44_rgb_clean|exp44_rgb_dropout"
  --transition "exp41_rgb_clean_to_exp47A_rgb_clean|exp41_rgb_clean|exp47A_rgb_clean"
  --transition "exp41_rgb_dropout_to_exp47A_rgb_dropout|exp41_rgb_dropout|exp47A_rgb_dropout"
  --transition "exp47A_rgb_clean_to_rgb_dropout|exp47A_rgb_clean|exp47A_rgb_dropout"
  --transition "exp41_rgb_clean_to_exp47B_rgb_clean|exp41_rgb_clean|exp47B_rgb_clean"
  --transition "exp41_rgb_dropout_to_exp47B_rgb_dropout|exp41_rgb_dropout|exp47B_rgb_dropout"
  --transition "exp47B_rgb_clean_to_rgb_dropout|exp47B_rgb_clean|exp47B_rgb_dropout"
)

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  ARGS+=(--check-only)
fi

echo "[$(date '+%F %T')] exp48 bitplane/refiner-damage tracking"
echo "Experiment root: $EXP_ROOT"
echo "Max images: $MAX_IMAGES"
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "Mode: check-only"
else
  echo "Mode: full diagnosis"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  "$PY" scripts/analyze_o256_exp48_bitplane_refiner_damage.py "${ARGS[@]}"
else
  { time "$PY" scripts/analyze_o256_exp48_bitplane_refiner_damage.py "${ARGS[@]}"; } \
    2>&1 | tee "$LOG_DIR/exp48_bitplane_refiner_damage.log"
  echo
  echo "[$(date '+%F %T')] Done."
  echo "Summary md: $OUT_DIR/summary.md"
  echo "Variant summary csv: $OUT_DIR/variant_summary.csv"
  echo "Transition summary csv: $OUT_DIR/transition_summary.csv"
fi
