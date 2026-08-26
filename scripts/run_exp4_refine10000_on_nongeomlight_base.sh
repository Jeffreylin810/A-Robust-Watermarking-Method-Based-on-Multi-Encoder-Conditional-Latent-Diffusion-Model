#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

BASE_CKPT="Result/CoCo_ddpm_train10000_base20k_then_nongeomlight5k_seed61_final.pth.tar"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base.yml"
DATA_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen"
TRAIN_ALIAS="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61"
TRAIN_GT="$TRAIN_ALIAS/data/CoCo/val/gt"
EVAL_GT="$DATA_ROOT/data/CoCo/val/gt"
EVAL_WM="results/test_result/train10000_base20k_then_nongeomlight5k_seed61/eval1000/CoCo"

PAIR_OUTPUT="results/test_result/refine10000_nongeomlight_base_pairs_seed61"
TRAIN_WM="$PAIR_OUTPUT/train10000/CoCo"
STAGE_A="results/refinement/refine10000_nongeomlightbase_A10k_seed61"
STAGE_B="results/refinement/refine10000_nongeomlightbase_A10k_B2k_seed61"
APPLY="results/refinement/refine10000_nongeomlightbase_A10k_B2k_apply_eval1000_seed61"
ATTACK_RUN="exp4_nongeom_refine10000_nongeomlightbase_A10k_B2k_seed61"
SUMMARY_DIR="results/experiment_summaries/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61"
RUN="exp4_refine10000_nongeomlightbase_A10k_B2k_seed61"

cd "$ROOT"
mkdir -p logs

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$BASE_CKPT" "configs/$EVAL_CONFIG" "configs/$TRAIN_CONFIG" \
  "$TRAIN_ALIAS" "$TRAIN_GT" "$EVAL_GT" "$EVAL_WM"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

alias_count="$(find -L "$TRAIN_GT" -maxdepth 1 -type f | wc -l)"
eval_png_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
eval_msg_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
if [[ "$alias_count" -ne 10000 ]]; then
  echo "Expected 10000 train alias images, found: $alias_count"
  exit 1
fi
if [[ "$eval_png_count" -ne 1000 || "$eval_msg_count" -ne 1000 ]]; then
  echo "Expected eval1000 images/messages, found images=$eval_png_count messages=$eval_msg_count"
  exit 1
fi

if [[ -f "$SUMMARY_DIR/summary.md" ]]; then
  echo "Experiment 4 is already complete:"
  echo "$SUMMARY_DIR/summary.md"
  exit 0
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Experiment 4 preflight passed."
  echo "Base checkpoint: $BASE_CKPT"
  echo "Train alias images: $alias_count"
  echo "Eval images/messages: $eval_png_count/$eval_msg_count"
  echo "Stage A/B: 10000/2000 steps"
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

echo "[$(date '+%F %T')] Start Experiment 4"
echo "Base: $BASE_CKPT"
echo "Train pairs: refine10000 (same COCO train10000 alias as the previous formal refiner)"
echo "Stage A: 10000 steps; Stage B: 2000 steps"

echo
echo "[$(date '+%F %T')] Step 1/7: generate attack-aware base outputs for refine10000"
if [[ -e "$PAIR_OUTPUT" ]]; then
  pair_png_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  pair_msg_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ "$pair_png_count" -ne 10000 || "$pair_msg_count" -ne 10000 ]]; then
    echo "Incomplete existing refine10000 pairs: images=$pair_png_count messages=$pair_msg_count"
    echo "Remove only this incomplete directory before retrying: $PAIR_OUTPUT"
    exit 1
  fi
  echo "Reuse completed refine10000 pairs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$TRAIN_CONFIG" \
    --resume "$BASE_CKPT" \
    --image_folder "$PAIR_OUTPUT" \
    --run_name train10000 \
    --seed 61 \
    2>&1 | tee "logs/${RUN}_generate_pairs.log"

  pair_png_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
  pair_msg_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
  if [[ "$pair_png_count" -ne 10000 || "$pair_msg_count" -ne 10000 ]]; then
    echo "Incomplete refine10000 pairs: images=$pair_png_count messages=$pair_msg_count"
    exit 1
  fi
fi

echo
echo "[$(date '+%F %T')] Step 2/7: train refiner Stage A (image priority)"
if [[ -e "$STAGE_A" ]]; then
  if [[ ! -f "$STAGE_A/refiner_final.pth" || ! -f "$STAGE_A/report.txt" ]]; then
    echo "Incomplete existing Stage A output: $STAGE_A"
    exit 1
  fi
  echo "Reuse completed Stage A checkpoint."
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
    --output_dir "$STAGE_A" \
    --image_size 256 \
    --steps 10000 \
    --batch_size 4 \
    --num_workers 2 \
    --lr 1e-4 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --image_mse_weight 10.0 \
    --image_l1_weight 2.5 \
    --message_weight 0.2 \
    --regularize_weight 0.01 \
    --eval_every 1000 \
    --seed 61 \
    2>&1 | tee "logs/${RUN}_stageA_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/7: train refiner Stage B (message recovery)"
if [[ -e "$STAGE_B" ]]; then
  if [[ ! -f "$STAGE_B/refiner_final.pth" || ! -f "$STAGE_B/report.txt" ]]; then
    echo "Incomplete existing Stage B output: $STAGE_B"
    exit 1
  fi
  echo "Reuse completed Stage B checkpoint."
else
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --resume_refiner "$STAGE_A/refiner_final.pth" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
    --output_dir "$STAGE_B" \
    --image_size 256 \
    --steps 2000 \
    --batch_size 4 \
    --num_workers 2 \
    --lr 5e-5 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --image_mse_weight 10.0 \
    --image_l1_weight 2.5 \
    --message_weight 0.75 \
    --regularize_weight 0.01 \
    --eval_every 500 \
    --seed 61 \
    2>&1 | tee "logs/${RUN}_stageB_train.log"
fi

echo
echo "[$(date '+%F %T')] Step 4/7: apply Stage B to fixed unseen eval1000"
if [[ -e "$APPLY" ]]; then
  apply_png_count="$(find "$APPLY/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  if [[ ! -f "$APPLY/report.txt" || "$apply_png_count" -ne 1000 || \
    ! -d "$APPLY/attacked_clean" ]]; then
    echo "Incomplete existing eval1000 apply output: $APPLY"
    exit 1
  fi
  echo "Reuse completed unseen eval1000 apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --refiner "$STAGE_B/refiner_final.pth" \
    --watermarked_dir "$EVAL_WM" \
    --gt_dir "$EVAL_GT" \
    --output_dir "$APPLY" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed 61 \
    2>&1 | tee "logs/${RUN}_apply.log"
fi

echo
echo "[$(date '+%F %T')] Step 5/7: PSNR / SSIM and clean BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$EVAL_GT" \
  --results_path "$APPLY/refined_images" \
  2>&1 | tee "logs/${RUN}_psnr_ssim.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "$APPLY/attacked_clean" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee "logs/${RUN}_clean_ba.log"

echo
echo "[$(date '+%F %T')] Step 6/7: 11 non-geometric attacks"
if [[ -e "results/attacked_Result/$ATTACK_RUN" ]]; then
  attack_dir_count="$(find "results/attacked_Result/$ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_dir_count" -ne 11 ]]; then
    echo "Incomplete existing non-geometric attack output: directories=$attack_dir_count"
    exit 1
  fi
  echo "Reuse completed 11 non-geometric attack outputs."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$EVAL_GT" \
    --results_path "$APPLY/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$ATTACK_RUN" \
    --attack_scope nongeom \
    --seed 61 \
    2>&1 | tee "logs/${RUN}_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "results/attacked_Result/$ATTACK_RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee "logs/${RUN}_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 7/7: build formal summary"
mkdir -p "$SUMMARY_DIR"
"$PY" scripts/summarize_exp4_refiner.py \
  --project_root "$ROOT" \
  --output "$SUMMARY_DIR/summary.md"

echo
echo "[$(date '+%F %T')] Experiment 4 complete"
echo "Summary: $SUMMARY_DIR/summary.md"
