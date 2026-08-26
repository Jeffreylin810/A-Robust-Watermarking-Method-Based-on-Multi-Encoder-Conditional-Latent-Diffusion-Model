#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/14_supplementary_qualitymatched_inference_cost_20260628"
FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"

DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"
DIFFW_EVAL_CONFIG="CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
O256_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
O256_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"
GT_DIR="$FORMAL_ROOT/data/CoCo/val/gt"
NUM_IMAGES=1000
GAMMA="0.50"
GAMMA_TAG="gamma_0p50"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_quality_matched_inference_cost_diffw_gamma0p50_vs_o256rgb_seed61_20260628.sh --check-only
  bash scripts/run_quality_matched_inference_cost_diffw_gamma0p50_vs_o256rgb_seed61_20260628.sh
USAGE
}

run_timed() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  echo "[$(date '+%F %T')] RUN: $*" | tee "$log_path"
  /usr/bin/time -p "$@" 2>&1 | tee -a "$log_path"
}

check_gpu_idle() {
  local gpu_processes
  gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$gpu_processes" && "${ALLOW_GPU_BUSY:-0}" != "1" ]]; then
    echo "Refusing to start while another GPU compute process is active:"
    echo "$gpu_processes"
    echo "Set ALLOW_GPU_BUSY=1 only if you intentionally want to run anyway."
    exit 1
  fi
}

force_or_refuse() {
  local path="$1"
  if [[ -e "$path" ]]; then
    if [[ "${FORCE:-0}" == "1" ]]; then
      rm -rf "$path"
    else
      echo "Output already exists: $path"
      echo "Use FORCE=1 to remove and rerun."
      exit 1
    fi
  fi
}

assert_preflight() {
  local required=(
    "$PY"
    "$ROOT/scripts/generate_diffw_quality_sweep_outputs.py"
    "$ROOT/scripts/apply_rgb_refinement.py"
    "$ROOT/scripts/summarize_quality_matched_inference_cost.py"
    "$ROOT/configs/$O256_CONFIG"
    "$ROOT/$O256_CKPT"
    "$ROOT/$REFINER"
    "$DIFFW_ROOT/eval_diffusion.py"
    "$DIFFW_ROOT/configs/$DIFFW_EVAL_CONFIG"
    "$DIFFW_CKPT"
    "$GT_DIR"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required input: $path"
      exit 1
    fi
  done
}

mkdir -p "$EXP_ROOT"/{logs,eval_outputs,lpips,generated_outputs}

if [[ "${1:-}" == "--check-only" ]]; then
  assert_preflight
  echo "Quality-matched inference cost preflight passed."
  echo "Output root: $EXP_ROOT"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  usage
  exit 2
fi

check_gpu_idle
assert_preflight

DIFFW_RAW_OUT="$EXP_ROOT/eval_outputs/diffw_like_raw"
DIFFW_RAW_RESULTS="$DIFFW_RAW_OUT/finaltest1000/CoCo/CoCo"
DIFFW_MATCHED_ROOT="$EXP_ROOT/generated_outputs/diffw_like_gamma0p50"
DIFFW_MATCHED_RESULTS="$DIFFW_MATCHED_ROOT/scaled_outputs/$GAMMA_TAG/CoCo/CoCo"
O256_BASE_OUT="$EXP_ROOT/eval_outputs/o256_base"
O256_BASE_RESULTS="$O256_BASE_OUT/finaltest1000/CoCo/CoCo"
O256_RGB_OUT="$EXP_ROOT/eval_outputs/o256_rgb/apply"

force_or_refuse "$DIFFW_RAW_OUT"
force_or_refuse "$DIFFW_MATCHED_ROOT"
force_or_refuse "$O256_BASE_OUT"
force_or_refuse "$EXP_ROOT/eval_outputs/o256_rgb"

echo "[$(date '+%F %T')] Step 1/7: DiffW-like raw eval"
(
  cd "$DIFFW_ROOT"
  run_timed "$EXP_ROOT/logs/diffw_raw_eval.log" \
    "$PY" eval_diffusion.py \
      --config "$DIFFW_EVAL_CONFIG" \
      --resume "$DIFFW_CKPT" \
      --test_set CoCo \
      --image_folder "$DIFFW_RAW_OUT" \
      --run_name finaltest1000 \
      --seed 61
)

echo "[$(date '+%F %T')] Step 2/7: DiffW-like gamma=0.50 post-process"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/${GAMMA_TAG}_apply.log" \
    "$PY" scripts/generate_diffw_quality_sweep_outputs.py \
      --source_results_dir "$DIFFW_RAW_RESULTS" \
      --gt_dir "$GT_DIR" \
      --output_root "$DIFFW_MATCHED_ROOT" \
      --gammas "$GAMMA" \
      --overwrite
)

echo "[$(date '+%F %T')] Step 3/7: verify matched-quality DiffW-like metrics"
(
  cd "$DIFFW_ROOT"
  "$PY" calculate_psnr_ssim.py \
    --gt_path "$GT_DIR" \
    --results_path "$DIFFW_MATCHED_RESULTS" \
    2>&1 | tee "$EXP_ROOT/logs/${GAMMA_TAG}_psnr_ssim.log"
)
(
  cd "$ROOT"
  "$PY" scripts/calculate_lpips_dir.py \
    --gt_dir "$GT_DIR" \
    --results_dir "$DIFFW_MATCHED_RESULTS" \
    --output_dir "$EXP_ROOT/lpips/$GAMMA_TAG" \
    --batch_size 8 \
    --net alex \
    2>&1 | tee "$EXP_ROOT/logs/${GAMMA_TAG}_lpips.log"
)

echo "[$(date '+%F %T')] Step 4/7: O256-Base eval"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/o256_base_eval.log" \
    "$PY" eval_diffusion.py \
      --config "$O256_CONFIG" \
      --resume "$O256_CKPT" \
      --test_set CoCo \
      --image_folder "$O256_BASE_OUT" \
      --run_name finaltest1000 \
      --seed 61
)

echo "[$(date '+%F %T')] Step 5/7: O256-RGB apply"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/o256_rgb_refine.log" \
    "$PY" scripts/apply_rgb_refinement.py \
      --config "$O256_CONFIG" \
      --resume "$O256_CKPT" \
      --refiner "$REFINER" \
      --watermarked_dir "$O256_BASE_RESULTS" \
      --gt_dir "$GT_DIR" \
      --output_dir "$O256_RGB_OUT" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed 61
)

echo "[$(date '+%F %T')] Step 6/7: verify O256-RGB metrics"
(
  cd "$ROOT"
  "$PY" calculate_psnr_ssim.py \
    --gt_path "$GT_DIR" \
    --results_path "$O256_RGB_OUT" \
    2>&1 | tee "$EXP_ROOT/logs/o256_rgb_psnr_ssim.log"
)
(
  cd "$ROOT"
  "$PY" scripts/calculate_lpips_dir.py \
    --gt_dir "$GT_DIR" \
    --results_dir "$O256_RGB_OUT" \
    --output_dir "$EXP_ROOT/lpips/o256_rgb" \
    --batch_size 8 \
    --net alex \
    2>&1 | tee "$EXP_ROOT/logs/o256_rgb_lpips.log"
)

echo "[$(date '+%F %T')] Step 7/7: summarize"
(
  cd "$ROOT"
  "$PY" scripts/summarize_quality_matched_inference_cost.py \
    --exp_root "$EXP_ROOT" \
    --num_images "$NUM_IMAGES" \
    --gamma "$GAMMA"
)

echo "Done. Summary:"
echo "  $EXP_ROOT/quality_matched_inference_cost_summary.md"
