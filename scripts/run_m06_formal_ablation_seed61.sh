#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"

CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
BASE_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
FINAL_GT="$FORMAL_ROOT/data/CoCo/val/gt"
BASE_WM="results/test_result/m03_formal_seed61_base_finaltest1000/finaltest1000/CoCo"
SUMMARY="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/02_ablation/formal_stagewise_seed61/summary.md"

declare -A REFINERS=(
  ["stageA"]="results/refinement/m02_formal_seed61_A10k/refiner_final.pth"
  ["stageB"]="results/refinement/m02_formal_seed61_A10k_B2k/refiner_final.pth"
  ["stageBP"]="results/refinement/m02_formal_seed61_A10k_B2k_BP500/refiner_final.pth"
)

declare -A PREFIXES=(
  ["stageA"]="m06_formal_seed61_stageA_finaltest1000"
  ["stageB"]="m06_formal_seed61_stageB_finaltest1000"
  ["stageBP"]="m06_formal_seed61_stageBP_finaltest1000"
)

declare -A LABELS=(
  ["stageA"]="Stage A"
  ["stageB"]="Stage B"
  ["stageBP"]="Stage B+"
)

cd "$ROOT"
mkdir -p logs

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$PY" "configs/$CONFIG" "$BASE_CKPT" "$FINAL_GT" "$BASE_WM"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done
for variant in "${!REFINERS[@]}"; do
  if [[ ! -f "${REFINERS[$variant]}" ]]; then
    echo "Missing refiner checkpoint: ${REFINERS[$variant]}"
    exit 1
  fi
done

png_count="$(find "$BASE_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
msg_count="$(find "$BASE_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
  echo "Expected base finaltest output with 1000 images/messages, got images=$png_count messages=$msg_count"
  exit 1
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "M06 preflight passed."
  echo "Classification: F (formal chapter-4 ablation)"
  echo "Frozen base checkpoint: $BASE_CKPT"
  echo "Frozen base output: $BASE_WM"
  for variant in stageA stageB stageBP; do
    echo "  ${LABELS[$variant]} -> ${REFINERS[$variant]}"
  done
  echo "Full module result will be reused from M03."
  exit 0
fi

run_variant() {
  local variant="$1"
  local prefix="${PREFIXES[$variant]}"
  local refiner="${REFINERS[$variant]}"
  local output_dir="results/refinement/${prefix}_apply"
  local attack_run="${prefix}_nongeom"

  echo
  echo "[$(date '+%F %T')] ${LABELS[$variant]}: apply to finaltest1000"
  if [[ -e "$output_dir" ]]; then
    local refine_png_count
    refine_png_count="$(find "$output_dir/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
    if [[ ! -f "$output_dir/report.txt" || "$refine_png_count" -ne 1000 || ! -d "$output_dir/attacked_clean/Clean" ]]; then
      echo "Incomplete refined output for ${LABELS[$variant]}"
      exit 1
    fi
    echo "Reuse completed refined outputs for ${LABELS[$variant]}"
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$CONFIG" \
      --resume "$BASE_CKPT" \
      --refiner "$refiner" \
      --watermarked_dir "$BASE_WM" \
      --gt_dir "$FINAL_GT" \
      --output_dir "$output_dir" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed 61 \
      2>&1 | tee "logs/${prefix}_apply.log"
  fi

  echo "[$(date '+%F %T')] ${LABELS[$variant]}: PSNR / clean BA"
  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$output_dir/refined_images" \
    2>&1 | tee "logs/${prefix}_psnr_ssim.log"

  "$PY" calculate_ba.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$output_dir/attacked_clean" \
    --gt_message_dir "$BASE_WM" \
    2>&1 | tee "logs/${prefix}_clean_ba.log"

  echo "[$(date '+%F %T')] ${LABELS[$variant]}: nongeom attacks"
  if [[ -e "results/attacked_Result/$attack_run" ]]; then
    local attack_count
    attack_count="$(find "results/attacked_Result/$attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete attack output for ${LABELS[$variant]}"
      exit 1
    fi
    echo "Reuse completed nongeom attacks for ${LABELS[$variant]}"
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$output_dir/refined_images" \
      --output_dir results/attacked_Result \
      --run_name "$attack_run" \
      --attack_scope nongeom \
      --seed 61 \
      2>&1 | tee "logs/${prefix}_attack.log"
  fi

  "$PY" calculate_ba.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "results/attacked_Result/$attack_run" \
    --gt_message_dir "$BASE_WM" \
    2>&1 | tee "logs/${prefix}_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "results/attacked_Result/$attack_run/ba_report.txt" \
    --output "results/attacked_Result/$attack_run/ba_summary.txt"
}

echo "[$(date '+%F %T')] Start M06 formal stage-wise ablation"
echo "Classification: F"

for variant in stageA stageB stageBP; do
  run_variant "$variant"
done

echo
echo "[$(date '+%F %T')] Write M06 summary"
"$PY" scripts/summarize_m06_formal_ablation.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] M06 complete"
echo "Summary: $SUMMARY"
