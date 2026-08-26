#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

BASE_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_formal.yml"
FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
TRAIN_ALIAS="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal"
TRAIN_GT="$TRAIN_ALIAS/data/CoCo/val/gt"
DEV_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
DEV_WM="results/test_result/m01_formal_seed61_base30k_dev1000/eval1000/CoCo"

PAIR_OUTPUT="results/test_result/m02_formal_seed61_refine10000_pairs"
TRAIN_WM="$PAIR_OUTPUT/train10000/CoCo"

STAGE_A="results/refinement/m02_formal_seed61_A10k"
STAGE_B="results/refinement/m02_formal_seed61_A10k_B2k"
STAGE_BP="results/refinement/m02_formal_seed61_A10k_B2k_BP500"
STAGE_C="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500"

APPLY_DIR="results/refinement/m02_formal_seed61_fullmodule_dev1000_apply"
ATTACK_RUN="m02_formal_seed61_fullmodule_dev1000_nongeom"
SUMMARY="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_full_module/summary.md"

cd "$ROOT"
mkdir -p logs

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$PY" "$BASE_CKPT" "configs/$EVAL_CONFIG" "configs/$TRAIN_CONFIG" "$FORMAL_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

if [[ ! -d "$TRAIN_ALIAS" ]]; then
  "$PY" scripts/create_coco_split_alias.py \
    --source_root "$FORMAL_ROOT" \
    --source_split train \
    --output_root "$TRAIN_ALIAS" \
    --num_images 10000 \
    --overwrite_links >/dev/null
fi

for required in "$TRAIN_GT" "$DEV_GT" "$DEV_WM"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required runtime input: $required"
    exit 1
  fi
done

train_count="$(find -L "$TRAIN_GT" -maxdepth 1 -type f | wc -l)"
dev_png_count="$(find "$DEV_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
dev_msg_count="$(find "$DEV_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"

if [[ "$train_count" -ne 10000 ]]; then
  echo "Expected 10000 formal train alias images, found: $train_count"
  exit 1
fi
if [[ "$dev_png_count" -ne 1000 || "$dev_msg_count" -ne 1000 ]]; then
  echo "Expected dev1000 images/messages, found images=$dev_png_count messages=$dev_msg_count"
  exit 1
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "M02 preflight passed."
  echo "Classification: F (formal artifact generation; dev1000 health-check only)"
  echo "Uses M01 formal base and retrains the full RGB refinement module from random initialization."
  echo "Locked module recipe: A10k -> B2k -> B+500 -> targeted C500"
  echo "Formal train root: $FORMAL_ROOT"
  echo "Does NOT evaluate finaltest1000."
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

train_stage() {
  local stage_type="$1"
  local output_dir="$2"
  local resume_refiner="$3"
  local steps="$4"
  local lr="$5"
  local message_weight="$6"
  local eval_every="$7"
  local log_name="$8"

  if [[ -e "$output_dir" ]]; then
    if [[ ! -f "$output_dir/refiner_final.pth" || ! -f "$output_dir/report.txt" ]]; then
      echo "Incomplete stage output: $output_dir"
      exit 1
    fi
    echo "Reuse completed stage: $output_dir"
    return
  fi

  local command=(
    "$PY" -u scripts/train_rgb_refinement.py
    --config "$EVAL_CONFIG"
    --resume "$BASE_CKPT"
    --watermarked_dir "$TRAIN_WM"
    --gt_dir "$TRAIN_GT"
    --output_dir "$output_dir"
    --image_size 256
    --steps "$steps"
    --batch_size 4
    --num_workers 2
    --lr "$lr"
    --hidden_channels 48
    --depth 5
    --max_delta 0.50
    --conditioning cover
    --image_mse_weight 10.0
    --image_l1_weight 2.5
    --message_weight "$message_weight"
    --regularize_weight 0.01
    --eval_every "$eval_every"
    --log_every 50
    --skip_final_dataset_images
    --seed 61
  )

  if [[ -n "$resume_refiner" ]]; then
    command+=(--resume_refiner "$resume_refiner")
  fi
  if [[ "$stage_type" == "targeted_c" ]]; then
    command+=(
      --attack_message_weight 0.15
      --attack_profile targeted_formal_weak
      --attack_noise_sigma 0.2738612788
      --attack_noise_sigma_secondary 0.15
      --attack_dropout_prob 0.70
      --attack_salt_pepper_prob 0.20
    )
  fi

  /usr/bin/time -p "${command[@]}" 2>&1 | tee "logs/${log_name}.log"
}

echo "[$(date '+%F %T')] Start M02 formal seed61 full module"
echo "Base: $BASE_CKPT"
echo "Classification: F"

echo
echo "[$(date '+%F %T')] Step 1/8: create / verify formal train10000 alias"
"$PY" scripts/create_coco_split_alias.py \
  --source_root "$FORMAL_ROOT" \
  --source_split train \
  --output_root "$TRAIN_ALIAS" \
  --num_images 10000 \
  --overwrite_links \
  2>&1 | tee logs/m02_formal_seed61_create_alias.log

echo
echo "[$(date '+%F %T')] Step 2/8: generate formal refine10000 pairs from M01 base"
if [[ -e "$PAIR_OUTPUT" ]]; then
  pair_png_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  pair_msg_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ "$pair_png_count" -ne 10000 || "$pair_msg_count" -ne 10000 ]]; then
    echo "Incomplete pair output: images=$pair_png_count messages=$pair_msg_count"
    exit 1
  fi
  echo "Reuse completed formal refine10000 pairs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$TRAIN_CONFIG" \
    --resume "$BASE_CKPT" \
    --image_folder "$PAIR_OUTPUT" \
    --run_name train10000 \
    --seed 61 \
    2>&1 | tee logs/m02_formal_seed61_generate_pairs.log
fi

pair_png_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
pair_msg_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
if [[ "$pair_png_count" -ne 10000 || "$pair_msg_count" -ne 10000 ]]; then
  echo "Expected 10000 formal refine pairs, found images=$pair_png_count messages=$pair_msg_count"
  exit 1
fi

echo
echo "[$(date '+%F %T')] Step 3/8: train Stage A"
train_stage image "$STAGE_A" "" 10000 1e-4 0.2 1000 "m02_formal_seed61_stageA"

echo
echo "[$(date '+%F %T')] Step 4/8: train Stage B"
train_stage message "$STAGE_B" "$STAGE_A/refiner_final.pth" 2000 5e-5 0.75 500 "m02_formal_seed61_stageB"

echo
echo "[$(date '+%F %T')] Step 5/8: train Stage B+"
train_stage message "$STAGE_BP" "$STAGE_B/refiner_final.pth" 500 2e-5 1.0 500 "m02_formal_seed61_stageBplus"

echo
echo "[$(date '+%F %T')] Step 6/8: train targeted Stage C"
train_stage targeted_c "$STAGE_C" "$STAGE_BP/refiner_final.pth" 500 1e-5 1.0 500 "m02_formal_seed61_stageC"

echo
echo "[$(date '+%F %T')] Step 7/8: apply full module to dev1000"
if [[ -e "$APPLY_DIR" ]]; then
  apply_png_count="$(find "$APPLY_DIR/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  if [[ ! -f "$APPLY_DIR/report.txt" || "$apply_png_count" -ne 1000 || ! -d "$APPLY_DIR/attacked_clean/Clean" ]]; then
    echo "Incomplete apply output: $APPLY_DIR"
    exit 1
  fi
  echo "Reuse completed dev1000 apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --refiner "$STAGE_C/refiner_final.pth" \
    --watermarked_dir "$DEV_WM" \
    --gt_dir "$DEV_GT" \
    --output_dir "$APPLY_DIR" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed 61 \
    2>&1 | tee logs/m02_formal_seed61_fullmodule_dev1000_apply.log
fi

echo
echo "[$(date '+%F %T')] Step 8/8: dev1000 PSNR / clean BA / nongeom BA / summary"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$DEV_GT" \
  --results_path "$APPLY_DIR/refined_images" \
  2>&1 | tee logs/m02_formal_seed61_fullmodule_dev1000_psnr_ssim.log

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "$APPLY_DIR/attacked_clean" \
  --gt_message_dir "$DEV_WM" \
  2>&1 | tee logs/m02_formal_seed61_fullmodule_dev1000_clean_ba.log

if [[ -e "results/attacked_Result/$ATTACK_RUN" ]]; then
  attack_count="$(find "results/attacked_Result/$ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete nongeom attack output: results/attacked_Result/$ATTACK_RUN"
    exit 1
  fi
  echo "Reuse completed nongeom attack output."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$DEV_GT" \
    --results_path "$APPLY_DIR/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$ATTACK_RUN" \
    --attack_scope nongeom \
    --seed 61 \
    2>&1 | tee logs/m02_formal_seed61_fullmodule_dev1000_attack.log
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "results/attacked_Result/$ATTACK_RUN" \
  --gt_message_dir "$DEV_WM" \
  2>&1 | tee logs/m02_formal_seed61_fullmodule_dev1000_ba.log

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$ATTACK_RUN/ba_summary.txt"

"$PY" scripts/summarize_m02_formal_full_module.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] M02 complete"
echo "Summary: $SUMMARY"
