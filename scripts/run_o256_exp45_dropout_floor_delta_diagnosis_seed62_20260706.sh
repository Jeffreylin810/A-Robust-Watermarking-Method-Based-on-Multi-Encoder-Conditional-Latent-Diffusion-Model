#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

EXP41_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
EXP44_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/44_supplementary_o256_train_hardcase_dropout_replay_seed62_20260705"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/45_supplementary_o256_exp44_dropout_floor_delta_diagnosis_seed62_20260706"

CONFIG41="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed62_exp41_20260705.yml"
CKPT41="$EXP41_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"

GT_MESSAGES="$EXP41_ROOT/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo"
FINAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"

EXP41_BASE_DROPOUT="$EXP41_ROOT/attacked_Result/exp41_base65_dropout_margin_nongeom/Paper_Dropout_r03"
EXP41_RGB_DROPOUT="$EXP41_ROOT/attacked_Result/exp41_rgb_full_from_scratch_dropout_balanced_nongeom/Paper_Dropout_r03"
EXP44_RGB_DROPOUT="$EXP44_ROOT/attacked_Result/exp44_hardcase_dropout_replay500_nongeom/Paper_Dropout_r03"

EXP41_RGB_IMAGES="$EXP41_ROOT/apply/exp41_rgb_full_from_scratch_dropout_balanced_finaltest1000/refined_images"
EXP44_RGB_IMAGES="$EXP44_ROOT/apply/exp44_hardcase_dropout_replay500_finaltest1000/refined_images"

DIAG_DIR="$EXP_ROOT/diagnosis"
DELTA_DIR="$EXP_ROOT/delta"
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

count_msgs() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l
}

require_variant_dir() {
  local label="$1"
  local attack_dir="$2"
  local count
  require_path "$attack_dir"
  count="$(count_images "$attack_dir")"
  if [[ "$count" -ne 1000 && "$MAX_IMAGES" -eq 0 ]]; then
    echo "Unexpected image count for $label: $count (expected 1000)"
    exit 1
  fi
  if [[ "$count" -lt 1 ]]; then
    echo "No images for $label: $attack_dir"
    exit 1
  fi
  echo "$label: dropout_images=$count"
}

require_path "$PY"
require_path "scripts/diagnose_dropout_floor_errors.py"
require_path "scripts/compare_dropout_floor_delta.py"
require_path "configs/$CONFIG41"
require_path "$CKPT41"
require_path "$GT_MESSAGES"
require_path "$FINAL_GT"
require_path "$EXP41_RGB_IMAGES"
require_path "$EXP44_RGB_IMAGES"

msg_count="$(count_msgs "$GT_MESSAGES")"
gt_count="$(count_images "$FINAL_GT")"
if [[ "$msg_count" -ne 1000 || "$gt_count" -ne 1000 ]]; then
  echo "Unexpected finaltest counts: messages=$msg_count gt_images=$gt_count"
  exit 1
fi

require_variant_dir "exp41_base_dropout" "$EXP41_BASE_DROPOUT"
require_variant_dir "exp41_rgb_dropout" "$EXP41_RGB_DROPOUT"
require_variant_dir "exp44_rgb_dropout" "$EXP44_RGB_DROPOUT"

exp41_rgb_count="$(count_images "$EXP41_RGB_IMAGES")"
exp44_rgb_count="$(count_images "$EXP44_RGB_IMAGES")"
if [[ "$exp41_rgb_count" -ne 1000 || "$exp44_rgb_count" -ne 1000 ]]; then
  echo "Unexpected refined image counts: exp41_rgb=$exp41_rgb_count exp44_rgb=$exp44_rgb_count"
  exit 1
fi

free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
  echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  exit 1
fi

if [[ "$CHECK_ONLY" == true ]]; then
  echo "O256 exp45 dropout-floor delta diagnosis preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Finaltest messages/GT: $msg_count/$gt_count"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Estimated runtime: about 8-20 minutes on GPU"
  echo "Disk check: OK"
  exit 0
fi

mkdir -p "$DIAG_DIR" "$DELTA_DIR" "$LOG_DIR"

echo "[$(date '+%F %T')] Step 1/2: per-image Dropout r03 diagnosis"
"$PY" scripts/diagnose_dropout_floor_errors.py \
  --output_dir "$DIAG_DIR" \
  --max_images "$MAX_IMAGES" \
  --variant "exp41_base_dropout|base|$CONFIG41|$CKPT41|$GT_MESSAGES|$EXP41_BASE_DROPOUT" \
  --variant "exp41_rgb_dropout|rgb_reference|$CONFIG41|$CKPT41|$GT_MESSAGES|$EXP41_RGB_DROPOUT" \
  --variant "exp44_rgb_dropout|rgb_hardcase_replay|$CONFIG41|$CKPT41|$GT_MESSAGES|$EXP44_RGB_DROPOUT" \
  --compare "exp41_base_dropout|exp41_rgb_dropout" \
  --compare "exp41_rgb_dropout|exp44_rgb_dropout" \
  --compare "exp41_base_dropout|exp44_rgb_dropout" \
  2>&1 | tee "$LOG_DIR/exp45_dropout_floor_diagnosis.log"

echo
echo "[$(date '+%F %T')] Step 2/2: exp41 -> exp44 per-image delta analysis"
"$PY" scripts/compare_dropout_floor_delta.py \
  --per_image_csv "$DIAG_DIR/per_image_ba.csv" \
  --left "exp41_rgb_dropout" \
  --right "exp44_rgb_dropout" \
  --baseline "exp41_base_dropout" \
  --output_dir "$DELTA_DIR" \
  --threshold_ba 90 \
  --left_image_dir "$EXP41_RGB_IMAGES" \
  --right_image_dir "$EXP44_RGB_IMAGES" \
  --gt_dir "$FINAL_GT" \
  2>&1 | tee "$LOG_DIR/exp45_delta_analysis.log"

echo
echo "[$(date '+%F %T')] Done."
echo "Diagnosis summary md: $DIAG_DIR/summary.md"
echo "Delta summary md: $DELTA_DIR/summary.md"
echo "Per-image delta csv: $DELTA_DIR/per_image_delta.csv"
