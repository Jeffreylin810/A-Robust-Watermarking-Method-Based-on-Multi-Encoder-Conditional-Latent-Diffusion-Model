#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=62

BASE20_CONFIG="CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_nongeom_seed62.yml"
BASE25_CONFIG="CoCo_train10000_ours256_msg32_base20k_then_nongeomlight5k_seed62.yml"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"

BASE20_OUTPUT="results/formal_training/train10000_base20k_nongeom_seed62"
BASE20_SOURCE="$BASE20_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
BASE20_CKPT="Result/CoCo_ddpm_train10000_base20k_nongeom_seed62_final.pth.tar"

BASE25_OUTPUT="results/formal_training/train10000_base20k_then_nongeomlight5k_seed62"
BASE25_SOURCE="$BASE25_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
BASE25_CKPT="Result/CoCo_ddpm_train10000_base20k_then_nongeomlight5k_seed62_final.pth.tar"

EVAL_OUTPUT="results/test_result/train10000_base20k_then_nongeomlight5k_seed62"
EVAL_WM="$EVAL_OUTPUT/eval1000/CoCo"

SEED61_PLUGIN="results/refinement/refine10000_targetedformal_C500_seed61/refiner_final.pth"
TRANSFER_APPLY="results/refinement/exp7_seed62_with_seed61_C500_plugin_apply_eval1000"

BASE_CLEAN_RUN="clean_exp7_seed62_base25k"
BASE_ATTACK_RUN="exp7_seed62_base25k_nongeom"
TRANSFER_ATTACK_RUN="exp7_seed62_with_seed61_C500_plugin_nongeom"
SUMMARY_DIR="results/experiment_summaries/exp7_seed62_cross_seed_transfer"
SUMMARY="$SUMMARY_DIR/summary.md"

cd "$ROOT"
mkdir -p logs Result

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "configs/$BASE20_CONFIG" "configs/$BASE25_CONFIG" \
  "configs/$EVAL_CONFIG" "$GT" "$SEED61_PLUGIN"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

if [[ "$CHECK_ONLY" == true ]]; then
  echo "Experiment 7 preflight passed."
  echo "Only intended base-training change: seed 61 -> 62"
  echo "Base recipe: from-scratch 20k + nongeom-light continuation 5k"
  echo "Transferred plugin: $SEED61_PLUGIN"
  echo "The plugin will not be retrained in Experiment 7."
  echo "Existing outputs are reused only when their expected files/counts are complete."
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

if [[ -f "$SUMMARY" ]]; then
  echo "Experiment 7 is already complete:"
  echo "$SUMMARY"
  exit 0
fi

echo "[$(date '+%F %T')] Start Experiment 7: seed62 cross-seed plugin validation"

echo
echo "[$(date '+%F %T')] Step 1/8: train seed62 base from scratch to 20k"
if [[ -f "$BASE20_CKPT" ]]; then
  echo "Reuse completed seed62 base20k checkpoint."
elif [[ -e "$BASE20_OUTPUT" ]]; then
  echo "Incomplete or unarchived base20k output exists: $BASE20_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$BASE20_CONFIG" \
    --resume ./ckpts/nonexistent_exp7_seed62_base20k_start.pth.tar \
    --image_folder results/validation/exp7_seed62_base20k \
    --seed "$SEED" \
    2>&1 | tee logs/exp7_seed62_base20k_train.log

  if [[ ! -f "$BASE20_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $BASE20_SOURCE"
    exit 1
  fi
  mv "$BASE20_SOURCE" "$BASE20_CKPT"
fi

echo
echo "[$(date '+%F %T')] Step 2/8: continue seed62 base with nongeom-light for 5k"
if [[ -f "$BASE25_CKPT" ]]; then
  echo "Reuse completed seed62 base25k checkpoint."
elif [[ -e "$BASE25_OUTPUT" ]]; then
  echo "Incomplete or unarchived base25k output exists: $BASE25_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$BASE25_CONFIG" \
    --resume "$BASE20_CKPT" \
    --image_folder results/validation/exp7_seed62_base25k \
    --seed "$SEED" \
    2>&1 | tee logs/exp7_seed62_base25k_train.log

  if [[ ! -f "$BASE25_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $BASE25_SOURCE"
    exit 1
  fi
  mv "$BASE25_SOURCE" "$BASE25_CKPT"
fi

echo
echo "[$(date '+%F %T')] Step 3/8: generate fixed unseen eval1000 with seed62 base"
if [[ -e "$EVAL_OUTPUT" ]]; then
  eval_png_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  eval_msg_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ "$eval_png_count" -ne 1000 || "$eval_msg_count" -ne 1000 ]]; then
    echo "Incomplete existing seed62 eval output: images=$eval_png_count messages=$eval_msg_count"
    exit 1
  fi
  echo "Reuse completed seed62 unseen eval1000."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE25_CKPT" \
    --image_folder "$EVAL_OUTPUT" \
    --run_name eval1000 \
    --seed "$SEED" \
    2>&1 | tee logs/exp7_seed62_base25k_eval.log
fi

echo
echo "[$(date '+%F %T')] Step 4/8: evaluate seed62 base before plugin"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$GT" \
  --results_path "$EVAL_WM" \
  2>&1 | tee logs/exp7_seed62_base25k_psnr_ssim.log

if [[ ! -e "results/attacked_Result/$BASE_CLEAN_RUN" ]]; then
  mkdir -p "results/attacked_Result/$BASE_CLEAN_RUN"
  ln -s "$(realpath "$EVAL_WM")" "results/attacked_Result/$BASE_CLEAN_RUN/Clean"
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE25_CKPT" \
  --attack_dir "results/attacked_Result/$BASE_CLEAN_RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp7_seed62_base25k_clean_ba.log

if [[ ! -e "results/attacked_Result/$BASE_ATTACK_RUN" ]]; then
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$GT" \
    --results_path "$EVAL_WM" \
    --output_dir results/attacked_Result \
    --run_name "$BASE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee logs/exp7_seed62_base25k_attack.log
else
  base_attack_count="$(find "results/attacked_Result/$BASE_ATTACK_RUN" \
    -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$base_attack_count" -ne 11 ]]; then
    echo "Incomplete existing seed62 base attack output: directories=$base_attack_count"
    exit 1
  fi
  echo "Reuse completed seed62 base nongeom attacks."
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE25_CKPT" \
  --attack_dir "results/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp7_seed62_base25k_ba.log
"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 5/8: apply frozen seed61 plugin to seed62 outputs"
if [[ -e "$TRANSFER_APPLY" ]]; then
  transfer_png_count="$(find "$TRANSFER_APPLY/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  transfer_msg_count="$(find "$TRANSFER_APPLY/refined_images" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ ! -f "$TRANSFER_APPLY/report.json" || "$transfer_png_count" -ne 1000 || \
    "$transfer_msg_count" -ne 1000 || ! -d "$TRANSFER_APPLY/attacked_clean/Clean" ]]; then
    echo "Incomplete existing transfer output: $TRANSFER_APPLY"
    exit 1
  fi
  echo "Reuse completed seed62 transfer apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE25_CKPT" \
    --refiner "$SEED61_PLUGIN" \
    --watermarked_dir "$EVAL_WM" \
    --gt_dir "$GT" \
    --output_dir "$TRANSFER_APPLY" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee logs/exp7_seed62_with_seed61_C500_plugin_apply.log
fi

echo
echo "[$(date '+%F %T')] Step 6/8: evaluate transferred plugin PSNR and Clean BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$GT" \
  --results_path "$TRANSFER_APPLY/refined_images" \
  2>&1 | tee logs/exp7_seed62_with_seed61_C500_plugin_psnr_ssim.log
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE25_CKPT" \
  --attack_dir "$TRANSFER_APPLY/attacked_clean" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp7_seed62_with_seed61_C500_plugin_clean_ba.log

echo
echo "[$(date '+%F %T')] Step 7/8: evaluate transferred plugin under 11 nongeom attacks"
if [[ ! -e "results/attacked_Result/$TRANSFER_ATTACK_RUN" ]]; then
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$GT" \
    --results_path "$TRANSFER_APPLY/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$TRANSFER_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee logs/exp7_seed62_with_seed61_C500_plugin_attack.log
else
  transfer_attack_count="$(find "results/attacked_Result/$TRANSFER_ATTACK_RUN" \
    -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$transfer_attack_count" -ne 11 ]]; then
    echo "Incomplete existing transferred-plugin attack output: directories=$transfer_attack_count"
    exit 1
  fi
  echo "Reuse completed transferred-plugin nongeom attacks."
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE25_CKPT" \
  --attack_dir "results/attacked_Result/$TRANSFER_ATTACK_RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp7_seed62_with_seed61_C500_plugin_ba.log
"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$TRANSFER_ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$TRANSFER_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 8/8: build fixed-threshold cross-seed summary"
mkdir -p "$SUMMARY_DIR"
"$PY" scripts/summarize_exp7_seed62_cross_seed.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] Experiment 7 complete"
echo "Summary: $SUMMARY"
