#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
CONFIG="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"
CONFIG_FILE="configs/$CONFIG"
BASE_CKPT="Result/CoCo_ddpm_train10000_base20k_then_nongeomlight5k_seed61_final.pth.tar"
EVAL_WM="results/test_result/train10000_base20k_then_nongeomlight5k_seed61/eval1000/CoCo"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
SUMMARY_DIR="results/experiment_summaries/exp3_refiner_transfer_to_nongeomlight_base_seed61"

STAGE_B_REFINER="results/refinement/O256_RGBREFINE_FORMAL_TRAIN10000_IMGPRI_THEN_MSG0P75_2K_seed61/refiner_final.pth"
STAGE_C_REFINER="results/refinement/refine10000_A10k_B2k_C500_nongeom_seed61/refiner_final.pth"
APPLY_A="results/refinement/exp3A_stageB_transfer_to_nongeomlight_base_seed61"
ATTACK_A="results/attacked_Result/exp3A_stageB_nongeom_transfer_to_nongeomlight_base_seed61"
APPLY_B="results/refinement/exp3B_stageC_transfer_to_nongeomlight_base_seed61"
ATTACK_B="results/attacked_Result/exp3B_stageC_nongeom_transfer_to_nongeomlight_base_seed61"

cd "$ROOT"
mkdir -p logs

for required in "$CONFIG_FILE" "$BASE_CKPT" "$EVAL_WM" "$GT" \
  "$STAGE_B_REFINER" "$STAGE_C_REFINER"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

for output in "$ATTACK_A" "$ATTACK_B" "$SUMMARY_DIR"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output"
    exit 1
  fi
done

run_transfer() {
  local label="$1"
  local refiner="$2"
  local apply_dir="$3"
  local attack_dir="$4"
  local attack_run
  attack_run="$(basename "$attack_dir")"
  local log_prefix="exp3${label}_transfer_to_nongeomlight_base_seed61"

  if [[ -e "$apply_dir" ]]; then
    if [[ ! -f "$apply_dir/report.txt" || ! -d "$apply_dir/refined_images" || \
      ! -d "$apply_dir/attacked_clean" ]]; then
      echo "Incomplete existing apply output: $apply_dir"
      exit 1
    fi
    echo
    echo "[$(date '+%F %T')] Experiment 3${label}: reuse completed apply output"
    echo "$apply_dir"
  else
    echo
    echo "[$(date '+%F %T')] Experiment 3${label}: apply frozen refiner"
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$CONFIG" \
      --resume "$BASE_CKPT" \
      --refiner "$refiner" \
      --watermarked_dir "$EVAL_WM" \
      --gt_dir "$GT" \
      --output_dir "$apply_dir" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed 61 \
      2>&1 | tee "logs/${log_prefix}_apply.log"
  fi

  echo
  echo "[$(date '+%F %T')] Experiment 3${label}: PSNR / SSIM"
  "$PY" calculate_psnr_ssim.py \
    --gt_path "$GT" \
    --results_path "$apply_dir/refined_images" \
    2>&1 | tee "logs/${log_prefix}_psnr_ssim.log"

  echo
  echo "[$(date '+%F %T')] Experiment 3${label}: clean BA"
  "$PY" calculate_ba.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$apply_dir/attacked_clean" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${log_prefix}_clean_ba.log"

  echo
  echo "[$(date '+%F %T')] Experiment 3${label}: 11 non-geometric attacks"
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$GT" \
    --results_path "$apply_dir/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$attack_run" \
    --attack_scope nongeom \
    --seed 61 \
    2>&1 | tee "logs/${log_prefix}_attack.log"

  "$PY" calculate_ba.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$attack_dir" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${log_prefix}_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "$attack_dir/ba_report.txt" \
    --output "$attack_dir/ba_summary.txt"
}

run_transfer "A_stageB" "$STAGE_B_REFINER" "$APPLY_A" "$ATTACK_A"
run_transfer "B_stageC" "$STAGE_C_REFINER" "$APPLY_B" "$ATTACK_B"

mkdir -p "$SUMMARY_DIR"
"$PY" scripts/summarize_exp3_refiner_transfer.py \
  --project_root "$ROOT" \
  --output "$SUMMARY_DIR/comparison.md"

echo
echo "[$(date '+%F %T')] Experiment 3 complete"
echo "Comparison: $SUMMARY_DIR/comparison.md"
