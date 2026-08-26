#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
CONFIG="CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_nongeom.yml"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
TRAIN_OUTPUT="results/formal_training/train10000_base20k_nongeom_seed61"
SOURCE_CKPT="$TRAIN_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
FINAL_CKPT="Result/CoCo_ddpm_train10000_base20k_nongeom_seed61_final.pth.tar"
EVAL_OUTPUT="results/test_result/train10000_base20k_nongeom_seed61"
RUN="exp2_nongeom_train10000_base20k_seed61"
CLEAN_RUN="clean_exp2_train10000_base20k_nongeom_seed61"

cd "$ROOT"
mkdir -p logs Result

for output in "$TRAIN_OUTPUT" "$FINAL_CKPT" "$EVAL_OUTPUT" \
  "results/attacked_Result/$RUN" "results/attacked_Result/$CLEAN_RUN"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output"
    exit 1
  fi
done

/usr/bin/time -p "$PY" -u train_diffusion.py \
  --config "$CONFIG" \
  --resume ./ckpts/nonexistent_train10000_base20k_nongeom_seed61_start.pth.tar \
  --image_folder results/validation/train10000_base20k_nongeom_seed61 \
  --seed 61 \
  2>&1 | tee "logs/${RUN}_train.log"

if [[ ! -f "$SOURCE_CKPT" ]]; then
  echo "Training finished without expected checkpoint: $SOURCE_CKPT"
  exit 1
fi
mv "$SOURCE_CKPT" "$FINAL_CKPT"

"$PY" eval_diffusion.py \
  --config "$EVAL_CONFIG" \
  --resume "$FINAL_CKPT" \
  --image_folder "$EVAL_OUTPUT" \
  --run_name eval1000 \
  --seed 61 \
  2>&1 | tee "logs/${RUN}_eval.log"

WM="$EVAL_OUTPUT/eval1000/CoCo"
if [[ ! -d "$WM" ]]; then
  echo "Expected eval output not found: $WM"
  exit 1
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$GT" \
  --results_path "$WM" \
  2>&1 | tee "logs/${RUN}_psnr_ssim.log"

mkdir -p "results/attacked_Result/$CLEAN_RUN"
ln -s "$(realpath "$WM")" "results/attacked_Result/$CLEAN_RUN/Clean"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$FINAL_CKPT" \
  --attack_dir "results/attacked_Result/$CLEAN_RUN" \
  --gt_message_dir "$WM" \
  2>&1 | tee "logs/${RUN}_clean_ba.log"

"$PY" attack_eval_diffw_paper_matched.py \
  --gt_path "$GT" \
  --results_path "$WM" \
  --output_dir results/attacked_Result \
  --run_name "$RUN" \
  --attack_scope nongeom \
  --seed 61 \
  2>&1 | tee "logs/${RUN}_attack.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$FINAL_CKPT" \
  --attack_dir "results/attacked_Result/$RUN" \
  --gt_message_dir "$WM" \
  2>&1 | tee "logs/${RUN}_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$RUN/ba_report.txt" \
  --output "results/attacked_Result/$RUN/ba_summary.txt"
