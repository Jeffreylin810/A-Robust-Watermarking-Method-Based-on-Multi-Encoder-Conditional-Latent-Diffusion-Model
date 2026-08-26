#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
DATE_TAG="20260627"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_${DATE_TAG}"
GT_DIR="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train1000_eval1000_unseen/data/CoCo/val/gt"
WAIT_MINUTES="${WAIT_MINUTES:-1440}"

PIXEL_ID20K_CONFIG="CoCo_train1000_supp_pixel_bitplane_identity20k_seed61_${DATE_TAG}.yml"
PIXEL_BASE30K_CONFIG="CoCo_train1000_supp_pixel_bitplane_identity20k_then_nongeomlight10k_seed61_${DATE_TAG}.yml"
PIXEL_EVAL_CONFIG="CoCo_eval1000_supp_pixel_bitplane_train1000_seed61_${DATE_TAG}.yml"
LATENT_LINEAR_ID20K_CONFIG="CoCo_train1000_supp_latent_linear_identity20k_seed61_${DATE_TAG}.yml"
LATENT_LINEAR_BASE30K_CONFIG="CoCo_train1000_supp_latent_linear_identity20k_then_nongeomlight10k_seed61_${DATE_TAG}.yml"
LATENT_LINEAR_EVAL_CONFIG="CoCo_eval1000_supp_latent_linear_train1000_seed61_${DATE_TAG}.yml"
LATENT_BITPLANE_ID20K_CONFIG="CoCo_train1000_supp_latent_bitplane_identity20k_seed61_${DATE_TAG}.yml"
LATENT_BITPLANE_BASE30K_CONFIG="CoCo_train1000_supp_latent_bitplane_identity20k_then_nongeomlight10k_seed61_${DATE_TAG}.yml"
LATENT_BITPLANE_EVAL_CONFIG="CoCo_eval1000_supp_latent_bitplane_train1000_seed61_${DATE_TAG}.yml"

MODE="${1:-all}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_supplementary_latent_vs_pixel_pipeline_20260627.sh
  bash scripts/run_supplementary_latent_vs_pixel_pipeline_20260627.sh all
  WAIT_MINUTES=720 bash scripts/run_supplementary_latent_vs_pixel_pipeline_20260627.sh all

What it does:
  1. Ensures the supplementary scaffold/configs exist
  2. Waits for the currently running pixel identity20k checkpoint if needed
  3. Continues pixel bitplane -> base30k + eval
  4. Trains/evals latent linear
  5. Trains/evals latent bitplane
  6. Runs round-trip survival tests for latent bitplane and pixel bitplane

Guardrails:
  - Skips steps that already have the expected outputs
  - Refuses to continue from a missing prerequisite checkpoint
  - Does not overwrite formal mainline artifacts
USAGE
}

if [[ "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "$EXP_ROOT/logs/pipeline"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  log "RUN: $*"
  (
    cd "$ROOT"
    "$@"
  ) 2>&1 | tee "$log_path"
}

has_eval_outputs() {
  local results_dir="$1"
  if [[ ! -d "$results_dir" ]]; then
    return 1
  fi
  local first_message
  first_message="$(find "$results_dir" -maxdepth 1 -type f -name '*_message.npy' -print -quit 2>/dev/null || true)"
  [[ -n "$first_message" ]]
}

resolve_eval_results_dir() {
  local run_name="$1"
  local candidates=(
    "$EXP_ROOT/eval_outputs/$run_name/CoCo/CoCo"
    "$EXP_ROOT/eval_outputs/$run_name/CoCo"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if has_eval_outputs "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  echo ""
  return 1
}

wait_for_checkpoint() {
  local ckpt="$1"
  local process_pattern="$2"
  local waited=0
  local sleep_seconds=60
  if [[ -f "$ckpt" ]]; then
    log "Checkpoint already present: $ckpt"
    return 0
  fi
  if pgrep -af "$process_pattern" >/dev/null; then
    log "Prerequisite checkpoint not ready yet. Waiting for: $ckpt"
    while [[ ! -f "$ckpt" ]]; do
      if (( waited >= WAIT_MINUTES * 60 )); then
        log "Timed out waiting for checkpoint: $ckpt"
        return 1
      fi
      sleep "$sleep_seconds"
      waited=$((waited + sleep_seconds))
      log "Still waiting (${waited}s elapsed): $ckpt"
    done
    log "Prerequisite checkpoint appeared: $ckpt"
    return 0
  fi
  log "Missing checkpoint and no matching training process found:"
  log "  ckpt: $ckpt"
  log "  process_pattern: $process_pattern"
  log "Please finish or rerun the prerequisite step first."
  return 1
}

assert_exists() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "Required path missing: $path"
    exit 1
  fi
}

ensure_setup() {
  log "Ensuring supplementary scaffold/configs exist"
  run_logged \
    "$EXP_ROOT/logs/pipeline/setup.log" \
    "$PY" scripts/setup_supplementary_latent_vs_pixel_experiment.py --date_tag "$DATE_TAG"
}

train_step() {
  local label="$1"
  local config_name="$2"
  local resume_path="$3"
  local image_folder="$4"
  local expected_ckpt="$5"
  local log_name="$6"

  if [[ -f "$expected_ckpt" ]]; then
    log "Skip train step already complete: $label"
    return 0
  fi

  run_logged \
    "$EXP_ROOT/logs/pipeline/$log_name" \
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$config_name" \
      --resume "$resume_path" \
      --image_folder "$image_folder" \
      --seed 61

  assert_exists "$expected_ckpt"
  log "Completed train step: $label"
}

ensure_clean_symlink_root() {
  local clean_root="$1"
  local results_dir="$2"
  mkdir -p "$clean_root"
  rm -rf "$clean_root/Clean"
  ln -sfn "$results_dir" "$clean_root/Clean"
}

eval_suite() {
  local label="$1"
  local config_name="$2"
  local ckpt="$3"
  local run_name="$4"
  local prefix="$5"

  local results_dir
  local lpips_dir="$EXP_ROOT/lpips/$prefix"
  local clean_root="$EXP_ROOT/attacked_Result/${prefix}_clean"
  local attack_run="${prefix}_nongeom"
  local attack_dir="$EXP_ROOT/attacked_Result/$attack_run"

  assert_exists "$ckpt"

  results_dir="$(resolve_eval_results_dir "$run_name" || true)"
  if [[ -z "$results_dir" ]]; then
    run_logged \
      "$EXP_ROOT/logs/pipeline/${prefix}_eval.log" \
      "$PY" -u eval_diffusion.py \
        --config "$config_name" \
        --resume "$ckpt" \
        --image_folder "$EXP_ROOT/eval_outputs" \
        --run_name "$run_name" \
        --seed 61
    results_dir="$(resolve_eval_results_dir "$run_name" || true)"
    if [[ -z "$results_dir" ]]; then
      log "Eval finished, but no usable output directory was found for run_name=$run_name"
      log "Checked:"
      log "  $EXP_ROOT/eval_outputs/$run_name/CoCo/CoCo"
      log "  $EXP_ROOT/eval_outputs/$run_name/CoCo"
      exit 1
    fi
  else
    log "Skip eval outputs already exist: $label"
  fi
  log "Using eval results directory for $label: $results_dir"

  run_logged \
    "$EXP_ROOT/logs/pipeline/${prefix}_psnr_ssim.log" \
    "$PY" calculate_psnr_ssim.py \
      --gt_path "$GT_DIR" \
      --results_path "$results_dir"

  if [[ ! -f "$lpips_dir/summary.json" ]]; then
    run_logged \
      "$EXP_ROOT/logs/pipeline/${prefix}_lpips.log" \
      "$PY" scripts/calculate_lpips_dir.py \
        --gt_dir "$GT_DIR" \
        --results_dir "$results_dir" \
        --output_dir "$lpips_dir" \
        --batch_size 8 \
        --net alex
  else
    log "Skip LPIPS already complete: $label"
  fi

  ensure_clean_symlink_root "$clean_root" "$results_dir"
  if [[ ! -f "$clean_root/ba_report.txt" ]]; then
    run_logged \
      "$EXP_ROOT/logs/pipeline/${prefix}_clean_ba.log" \
      "$PY" calculate_ba.py \
        --config "$config_name" \
        --resume "$ckpt" \
        --attack_dir "$clean_root" \
        --gt_message_dir "$results_dir"
  else
    log "Skip clean BA already complete: $label"
  fi

  if [[ ! -f "$attack_dir/metrics_report.txt" ]]; then
    run_logged \
      "$EXP_ROOT/logs/pipeline/${prefix}_attack.log" \
      "$PY" attack_eval_diffw_paper_matched.py \
        --gt_path "$GT_DIR" \
        --results_path "$results_dir" \
        --output_dir "$EXP_ROOT/attacked_Result" \
        --run_name "$attack_run" \
        --attack_scope nongeom \
        --seed 61 \
        --overwrite
  else
    log "Skip attacked image generation already complete: $label"
  fi

  if [[ ! -f "$attack_dir/ba_report.txt" ]]; then
    run_logged \
      "$EXP_ROOT/logs/pipeline/${prefix}_attack_ba.log" \
      "$PY" calculate_ba.py \
        --config "$config_name" \
        --resume "$ckpt" \
        --attack_dir "$attack_dir" \
        --gt_message_dir "$results_dir"
  else
    log "Skip attacked BA already complete: $label"
  fi
}

roundtrip_step() {
  local label="$1"
  local config_name="$2"
  local ckpt="$3"
  local output_dir="$4"
  local log_name="$5"

  assert_exists "$ckpt"
  if [[ -f "$output_dir/report.json" ]]; then
    log "Skip round-trip already complete: $label"
    return 0
  fi

  run_logged \
    "$EXP_ROOT/logs/pipeline/$log_name" \
    "$PY" scripts/evaluate_roundtrip_survival.py \
      --config "$config_name" \
      --resume "$ckpt" \
      --test_set CoCo \
      --seed 61 \
      --num_images 128 \
      --output_dir "$output_dir"
}

ensure_setup

PIXEL_ID20K_CKPT="$EXP_ROOT/runs/pixel_bitplane_identity20k_train1000/ckpts/CoCo_ddpm_final.pth.tar"
PIXEL_BASE30K_CKPT="$EXP_ROOT/runs/pixel_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar"
LATENT_LINEAR_ID20K_CKPT="$EXP_ROOT/runs/latent_linear_identity20k_train1000/ckpts/CoCo_ddpm_final.pth.tar"
LATENT_LINEAR_BASE30K_CKPT="$EXP_ROOT/runs/latent_linear_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar"
LATENT_BITPLANE_ID20K_CKPT="$EXP_ROOT/runs/latent_bitplane_identity20k_train1000/ckpts/CoCo_ddpm_final.pth.tar"
LATENT_BITPLANE_BASE30K_CKPT="$EXP_ROOT/runs/latent_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar"

wait_for_checkpoint "$PIXEL_ID20K_CKPT" "$PIXEL_ID20K_CONFIG"

train_step \
  "pixel bitplane base30k" \
  "$PIXEL_BASE30K_CONFIG" \
  "$PIXEL_ID20K_CKPT" \
  "results/validation/supp_pixel_bitplane_base30k_seed61_${DATE_TAG}" \
  "$PIXEL_BASE30K_CKPT" \
  "pixel_bitplane_base30k_train.log"

eval_suite \
  "pixel bitplane base30k eval" \
  "$PIXEL_EVAL_CONFIG" \
  "$PIXEL_BASE30K_CKPT" \
  "pixel_bitplane_base30k_eval1000" \
  "pixel_bitplane_base30k_eval1000"

train_step \
  "latent linear identity20k" \
  "$LATENT_LINEAR_ID20K_CONFIG" \
  "./ckpts/nonexistent_supp_latent_linear_identity20k_seed61_start.pth.tar" \
  "results/validation/supp_latent_linear_identity20k_seed61_${DATE_TAG}" \
  "$LATENT_LINEAR_ID20K_CKPT" \
  "latent_linear_identity20k_train.log"

train_step \
  "latent linear base30k" \
  "$LATENT_LINEAR_BASE30K_CONFIG" \
  "$LATENT_LINEAR_ID20K_CKPT" \
  "results/validation/supp_latent_linear_base30k_seed61_${DATE_TAG}" \
  "$LATENT_LINEAR_BASE30K_CKPT" \
  "latent_linear_base30k_train.log"

eval_suite \
  "latent linear base30k eval" \
  "$LATENT_LINEAR_EVAL_CONFIG" \
  "$LATENT_LINEAR_BASE30K_CKPT" \
  "latent_linear_base30k_eval1000" \
  "latent_linear_base30k_eval1000"

train_step \
  "latent bitplane identity20k" \
  "$LATENT_BITPLANE_ID20K_CONFIG" \
  "./ckpts/nonexistent_supp_latent_bitplane_identity20k_seed61_start.pth.tar" \
  "results/validation/supp_latent_bitplane_identity20k_seed61_${DATE_TAG}" \
  "$LATENT_BITPLANE_ID20K_CKPT" \
  "latent_bitplane_identity20k_train.log"

train_step \
  "latent bitplane base30k" \
  "$LATENT_BITPLANE_BASE30K_CONFIG" \
  "$LATENT_BITPLANE_ID20K_CKPT" \
  "results/validation/supp_latent_bitplane_base30k_seed61_${DATE_TAG}" \
  "$LATENT_BITPLANE_BASE30K_CKPT" \
  "latent_bitplane_base30k_train.log"

eval_suite \
  "latent bitplane base30k eval" \
  "$LATENT_BITPLANE_EVAL_CONFIG" \
  "$LATENT_BITPLANE_BASE30K_CKPT" \
  "latent_bitplane_base30k_eval1000" \
  "latent_bitplane_base30k_eval1000"

roundtrip_step \
  "latent bitplane round-trip" \
  "$LATENT_BITPLANE_EVAL_CONFIG" \
  "$LATENT_BITPLANE_BASE30K_CKPT" \
  "$EXP_ROOT/roundtrip/latent_bitplane_base30k" \
  "latent_bitplane_roundtrip.log"

roundtrip_step \
  "pixel bitplane round-trip" \
  "$PIXEL_EVAL_CONFIG" \
  "$PIXEL_BASE30K_CKPT" \
  "$EXP_ROOT/roundtrip/pixel_bitplane_base30k" \
  "pixel_bitplane_roundtrip.log"

log "Supplementary latent-vs-pixel pipeline completed."
log "Experiment root: $EXP_ROOT"
