#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
BASE_CKPT="Result/CoCo_ddpm_exp12_identity20k_then_nongeomlight10k_seed61_final.pth.tar"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base.yml"
TRAIN_ALIAS="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61"
TRAIN_GT="$TRAIN_ALIAS/data/CoCo/val/gt"
EVAL_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
EVAL_WM="results/test_result/exp12_identity20k_then_nongeomlight10k_seed61/eval1000/CoCo"
PAIR_OUTPUT="results/test_result/exp13_refine10000_base30k_pairs_seed61"
TRAIN_WM="$PAIR_OUTPUT/train10000/CoCo"
SUMMARY_DIR="results/experiment_summaries/exp13_refiner_stage_steps_seed61"
SUMMARY="$SUMMARY_DIR/summary.md"
SEED=61

cd "$ROOT"
mkdir -p logs

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$PY" "$BASE_CKPT" "configs/$EVAL_CONFIG" "configs/$TRAIN_CONFIG" \
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

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Experiment 13 preflight passed."
  echo "Classification: E (exploratory; dev1000 only)"
  echo "Base: identity20k -> nongeom-light10k (30k total)"
  echo "Shared pairs: refine10000 generated once from the locked Base"
  echo "Full:    A10k / B2k / B+500 / C500"
  echo "Reduced: A5k  / B1k / B+250 / C250"
  echo "Minimal: A3k  / B1k / B+250 / C250; runs only if Reduced passes"
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
      echo "Inspect or remove only this incomplete directory before retrying."
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
    --seed "$SEED"
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

evaluate_final() {
  local label="$1"
  local refiner="$2"
  local apply_dir="results/refinement/${label}_apply_eval1000"
  local attack_run="${label}_nongeom"

  if [[ -e "$apply_dir" ]]; then
    local png_count
    local msg_count
    png_count="$(find "$apply_dir/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
    msg_count="$(find "$apply_dir/refined_images" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
    if [[ ! -f "$apply_dir/report.txt" || "$png_count" -ne 1000 || \
      "$msg_count" -ne 1000 || ! -d "$apply_dir/attacked_clean/Clean" ]]; then
      echo "Incomplete apply output: $apply_dir"
      exit 1
    fi
    echo "Reuse completed apply output: $apply_dir"
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$EVAL_CONFIG" \
      --resume "$BASE_CKPT" \
      --refiner "$refiner" \
      --watermarked_dir "$EVAL_WM" \
      --gt_dir "$EVAL_GT" \
      --output_dir "$apply_dir" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed "$SEED" \
      2>&1 | tee "logs/${label}_apply.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$EVAL_GT" \
    --results_path "$apply_dir/refined_images" \
    2>&1 | tee "logs/${label}_psnr_ssim.log"

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$apply_dir/attacked_clean" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${label}_clean_ba.log"

  if [[ -e "results/attacked_Result/$attack_run" ]]; then
    local attack_count
    attack_count="$(find "results/attacked_Result/$attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete attack output: results/attacked_Result/$attack_run"
      exit 1
    fi
    echo "Reuse completed attack output: $attack_run"
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$EVAL_GT" \
      --results_path "$apply_dir/refined_images" \
      --output_dir results/attacked_Result \
      --run_name "$attack_run" \
      --attack_scope nongeom \
      --seed "$SEED" \
      2>&1 | tee "logs/${label}_attack.log"
  fi

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "results/attacked_Result/$attack_run" \
    --gt_message_dir "$EVAL_WM" \
    2>&1 | tee "logs/${label}_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "results/attacked_Result/$attack_run/ba_report.txt" \
    --output "results/attacked_Result/$attack_run/ba_summary.txt"
}

run_pipeline() {
  local label="$1"
  local a_steps="$2"
  local b_steps="$3"
  local bp_steps="$4"
  local c_steps="$5"

  local root="results/refinement/exp13_${label}"
  local stage_a="${root}_A${a_steps}"
  local stage_b="${root}_A${a_steps}_B${b_steps}"
  local stage_bp="${root}_A${a_steps}_B${b_steps}_BP${bp_steps}"
  local stage_c="${root}_A${a_steps}_B${b_steps}_BP${bp_steps}_C${c_steps}"
  local final_label="exp13_${label}_final_seed61"

  echo
  echo "[$(date '+%F %T')] Pipeline $label: A=$a_steps B=$b_steps B+=$bp_steps C=$c_steps"

  train_stage image "$stage_a" "" "$a_steps" 1e-4 0.2 \
    "$((a_steps < 1000 ? a_steps : 1000))" "exp13_${label}_stageA"
  train_stage message "$stage_b" "$stage_a/refiner_final.pth" "$b_steps" 5e-5 0.75 \
    "$((b_steps < 500 ? b_steps : 500))" "exp13_${label}_stageB"
  train_stage message "$stage_bp" "$stage_b/refiner_final.pth" "$bp_steps" 2e-5 1.0 \
    "$bp_steps" "exp13_${label}_stageBplus"
  train_stage targeted_c "$stage_c" "$stage_bp/refiner_final.pth" "$c_steps" 1e-5 1.0 \
    "$c_steps" "exp13_${label}_stageC"

  evaluate_final "$final_label" "$stage_c/refiner_final.pth"
}

echo "[$(date '+%F %T')] Start Experiment 13"
echo "Classification: E (exploratory Stage-step selection)"

echo
echo "Step 1: generate one shared refine10000 pair set from the selected 30k Base"
if [[ -e "$PAIR_OUTPUT" ]]; then
  pair_png_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  pair_msg_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ "$pair_png_count" -ne 10000 || "$pair_msg_count" -ne 10000 ]]; then
    echo "Incomplete pair output: images=$pair_png_count messages=$pair_msg_count"
    exit 1
  fi
  echo "Reuse completed shared pairs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$TRAIN_CONFIG" \
    --resume "$BASE_CKPT" \
    --image_folder "$PAIR_OUTPUT" \
    --run_name train10000 \
    --seed "$SEED" \
    2>&1 | tee logs/exp13_generate_refine10000_base30k_pairs_seed61.log
fi

pair_png_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
pair_msg_count="$(find "$TRAIN_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
if [[ "$pair_png_count" -ne 10000 || "$pair_msg_count" -ne 10000 ]]; then
  echo "Expected 10000 shared pairs, found images=$pair_png_count messages=$pair_msg_count"
  exit 1
fi

echo
echo "Step 2: Full reference"
run_pipeline full 10000 2000 500 500

echo
echo "Step 3: Reduced candidate"
run_pipeline reduced 5000 1000 250 250

mkdir -p "$SUMMARY_DIR"
"$PY" scripts/summarize_exp13_stage_steps.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

reduced_pass="$("$PY" scripts/summarize_exp13_stage_steps.py \
  --project_root "$ROOT" \
  --output "$SUMMARY" \
  --print_reduced_pass)"

if [[ "$reduced_pass" == "true" ]]; then
  echo
  echo "Step 4: Reduced passed; run Minimal candidate"
  run_pipeline minimal 3000 1000 250 250
  "$PY" scripts/summarize_exp13_stage_steps.py \
    --project_root "$ROOT" \
    --output "$SUMMARY"
else
  echo
  echo "Step 4: Reduced failed; skip Minimal candidate by pre-registered rule."
fi

echo
echo "[$(date '+%F %T')] Experiment 13 complete"
echo "Summary: $SUMMARY"

