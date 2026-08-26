#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
PLUS20K_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/21_supplementary_o256_base_plus20k_then_refresh500_seed61_20260630"
PLUS20K_CKPT="$PLUS20K_ROOT/workspace/o256_base_plus20k/ckpts/CoCo_ddpm_final.pth.tar"
PLUS20K_BASE_WM="$PLUS20K_ROOT/eval_outputs/o256_base_plus20k/finaltest1000/CoCo"
PLUS20K_REFRESH500_ATTACKS="$PLUS20K_ROOT/attacked_Result/o256_rgb_plus20k_refresh500_targeted_nongeom"

A_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/23_supplementary_o256_base_plus20k_weakattack_plus5k_seed61_20260630"
A_CKPT="$A_ROOT/workspace/o256_base_plus20k_weakattack_plus5k/ckpts/CoCo_ddpm_final.pth.tar"
A_BASE_WM="$A_ROOT/eval_outputs/o256_base_plus20k_weakattack_plus5k/finaltest1000/CoCo"
A_ATTACKS="$A_ROOT/attacked_Result/o256_rgb_plus20k_weakattack_plus5k_reuse_refresh500_nongeom"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/25_supplementary_o256_bitplane_weakbit_diagnosis_20260701"
OUT_PLUS20K="$EXP_ROOT/plus20k_refresh500"
OUT_A="$EXP_ROOT/weakattack_plus5k_rgb_reuse_refresh500"

cd "$ROOT"
mkdir -p "$EXP_ROOT"

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

require_path "configs/$EVAL_CONFIG"
require_path "$PLUS20K_CKPT"
require_path "$PLUS20K_BASE_WM"
require_path "$PLUS20K_REFRESH500_ATTACKS"

if [[ -e "$A_ROOT/summary.md" ]]; then
  require_path "$A_CKPT"
  require_path "$A_BASE_WM"
  require_path "$A_ATTACKS"
fi

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  echo "O256 bitplane weak-bit diagnosis preflight passed."
  echo "Will analyze plus20k_refresh500 and, if available, A weakattack +5k."
  echo "Estimated extra disk: <0.2G"
  echo "Current free disk: ${free_gb}G"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/2: per-bit BA for plus20k_refresh500"
"$PY" scripts/calculate_per_bit_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$PLUS20K_CKPT" \
  --attack_dir "$PLUS20K_REFRESH500_ATTACKS" \
  --gt_message_dir "$PLUS20K_BASE_WM" \
  --output_dir "$OUT_PLUS20K" \
  --top_k 8 \
  --base_weight 1.0 \
  --max_weight 3.0

if [[ -e "$A_ROOT/summary.md" ]]; then
  echo
  echo "[$(date '+%F %T')] Step 2/2: per-bit BA for A weakattack +5k"
  "$PY" scripts/calculate_per_bit_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$A_CKPT" \
    --attack_dir "$A_ATTACKS" \
    --gt_message_dir "$A_BASE_WM" \
    --output_dir "$OUT_A" \
    --top_k 8 \
    --base_weight 1.0 \
    --max_weight 3.0
else
  echo
  echo "[$(date '+%F %T')] Step 2/2: skip A diagnosis because A summary is not present yet."
fi

cat > "$EXP_ROOT/README.md" <<EOF
# O256 bitplane weak-bit diagnosis

This diagnostic runs per-bit BA on weak-attack outputs.

## Outputs

- plus20k_refresh500: $OUT_PLUS20K
- A weakattack +5k: $OUT_A

Use \`plus20k_refresh500/bit_weights.json\` as the default input for the bit-weighted refiner repair experiment.
EOF

echo
echo "[$(date '+%F %T')] Done."
echo "Diagnosis root: $EXP_ROOT"
echo "Default bit weights: $OUT_PLUS20K/bit_weights.json"
