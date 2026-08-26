#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

EXP40_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/40_formal_o256_full_from_scratch_dropout_margin_seed61_20260704"
EXP41_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
EXP42_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/42_supplementary_o256_exp41_dropout_floor_safety_margin_probe_seed62_20260705"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/43_supplementary_o256_dropout_floor_error_diagnosis_seed62_20260705"
SUMMARY_DIR="$EXP_ROOT/diagnosis"

CONFIG40="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
CONFIG41="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed62_exp41_20260705.yml"
CONFIG42="CoCo_finaltest1000_ours256_msg32_exp42A_dropout_floor_probe_seed62_20260705.yml"

EXP40_CKPT="$EXP40_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
EXP41_CKPT="$EXP41_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
EXP42_CKPT="$EXP42_ROOT/workspace/base70_dropout_margin_plus10k/ckpts/CoCo_ddpm_final.pth.tar"

EXP40_GT="$EXP40_ROOT/eval_outputs/exp40_base65_dropout_margin/finaltest1000/CoCo"
EXP41_GT="$EXP41_ROOT/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo"
EXP42_GT="$EXP42_ROOT/eval_outputs/exp42A_base70_dropout_margin_plus10k/finaltest1000/CoCo"

EXP40_RGB_DROPOUT="$EXP40_ROOT/attacked_Result/exp40_rgb_full_from_scratch_dropout_balanced_nongeom/Paper_Dropout_r03"
EXP41_BASE_DROPOUT="$EXP41_ROOT/attacked_Result/exp41_base65_dropout_margin_nongeom/Paper_Dropout_r03"
EXP41_RGB_DROPOUT="$EXP41_ROOT/attacked_Result/exp41_rgb_full_from_scratch_dropout_balanced_nongeom/Paper_Dropout_r03"
EXP42_BASE_DROPOUT="$EXP42_ROOT/attacked_Result/exp42A_base70_dropout_margin_plus10k_nongeom/Paper_Dropout_r03"
EXP42_RGB_DROPOUT="$EXP42_ROOT/attacked_Result/exp42A_rgb_base70_dropout_balanced_nongeom/Paper_Dropout_r03"

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

count_pngs() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l
}

count_msgs() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l
}

require_variant_io() {
  local label="$1"
  local config="$2"
  local ckpt="$3"
  local gt_dir="$4"
  local attack_dir="$5"
  require_path "configs/$config"
  require_path "$ckpt"
  require_path "$gt_dir"
  require_path "$attack_dir"
  local png_count
  local msg_count
  png_count="$(count_pngs "$attack_dir")"
  msg_count="$(count_msgs "$gt_dir")"
  if [[ "$png_count" -lt 1 || "$msg_count" -lt 1 ]]; then
    echo "Invalid input counts for $label: dropout_pngs=$png_count messages=$msg_count"
    exit 1
  fi
  echo "$label: dropout_pngs=$png_count messages=$msg_count"
}

require_path "$PY"
require_path "scripts/diagnose_dropout_floor_errors.py"

require_variant_io "exp40_rgb_dropout" "$CONFIG40" "$EXP40_CKPT" "$EXP40_GT" "$EXP40_RGB_DROPOUT"
require_variant_io "exp41_base_dropout" "$CONFIG41" "$EXP41_CKPT" "$EXP41_GT" "$EXP41_BASE_DROPOUT"
require_variant_io "exp41_rgb_dropout" "$CONFIG41" "$EXP41_CKPT" "$EXP41_GT" "$EXP41_RGB_DROPOUT"
require_variant_io "exp42A_base_dropout" "$CONFIG42" "$EXP42_CKPT" "$EXP42_GT" "$EXP42_BASE_DROPOUT"
require_variant_io "exp42A_rgb_dropout" "$CONFIG42" "$EXP42_CKPT" "$EXP42_GT" "$EXP42_RGB_DROPOUT"

free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
  echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  exit 1
fi

if [[ "$CHECK_ONLY" == true ]]; then
  echo "O256 exp43 dropout-floor diagnosis preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Estimated runtime: about 10-25 minutes on GPU"
  echo "Disk check: OK"
  exit 0
fi

mkdir -p "$SUMMARY_DIR"

echo "[$(date '+%F %T')] Step 1/1: diagnose Dropout r03 error structure"
"$PY" scripts/diagnose_dropout_floor_errors.py \
  --output_dir "$SUMMARY_DIR" \
  --max_images "$MAX_IMAGES" \
  --variant "exp40_rgb_dropout|rgb_reference|$CONFIG40|$EXP40_CKPT|$EXP40_GT|$EXP40_RGB_DROPOUT" \
  --variant "exp41_base_dropout|base|$CONFIG41|$EXP41_CKPT|$EXP41_GT|$EXP41_BASE_DROPOUT" \
  --variant "exp41_rgb_dropout|rgb|$CONFIG41|$EXP41_CKPT|$EXP41_GT|$EXP41_RGB_DROPOUT" \
  --variant "exp42A_base_dropout|base|$CONFIG42|$EXP42_CKPT|$EXP42_GT|$EXP42_BASE_DROPOUT" \
  --variant "exp42A_rgb_dropout|rgb|$CONFIG42|$EXP42_CKPT|$EXP42_GT|$EXP42_RGB_DROPOUT" \
  --compare "exp41_base_dropout|exp41_rgb_dropout" \
  --compare "exp42A_base_dropout|exp42A_rgb_dropout" \
  --compare "exp41_rgb_dropout|exp42A_rgb_dropout" \
  --compare "exp40_rgb_dropout|exp41_rgb_dropout"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_DIR/summary.md"
echo "Variant summary csv: $SUMMARY_DIR/variant_summary.csv"
echo "Comparisons csv: $SUMMARY_DIR/comparisons.csv"
