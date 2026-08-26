#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/14_supplementary_end_to_end_repeatability_sanity_20260628"
FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"

DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"
DIFFW_EVAL_CONFIG="CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
O256_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
O256_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"
GT_DIR="$FORMAL_ROOT/data/CoCo/val/gt"
NUM_IMAGES=1000
REPEATS="${REPEATS:-3}"
MIN_FREE_GB=2
KEEP_OUTPUTS="${KEEP_OUTPUTS:-0}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_end_to_end_repeatability_sanity_seed61_20260628.sh --check-only
  bash scripts/run_end_to_end_repeatability_sanity_seed61_20260628.sh

Environment:
  REPEATS=3           number of repeats (default 3)
  KEEP_OUTPUTS=1      keep temp outputs for each repeat
  ALLOW_GPU_BUSY=1    bypass active-GPU-process refusal
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

available_gb() {
  df -Pk "$ROOT" | awk 'NR==2 {printf "%.2f", $4/1024/1024}'
}

assert_free_space() {
  local free_gb
  free_gb="$(available_gb)"
  awk -v free="$free_gb" -v need="$MIN_FREE_GB" 'BEGIN {exit !(free+0 >= need+0)}' || {
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    exit 1
  }
}

assert_preflight() {
  local required=(
    "$PY"
    "$ROOT/scripts/apply_rgb_refinement.py"
    "$ROOT/scripts/summarize_end_to_end_repeatability_sanity.py"
    "$ROOT/configs/$O256_CONFIG"
    "$ROOT/$O256_CKPT"
    "$ROOT/$REFINER"
    "$DIFFW_ROOT/eval_diffusion.py"
    "$DIFFW_ROOT/configs/$DIFFW_EVAL_CONFIG"
    "$DIFFW_CKPT"
    "$FORMAL_ROOT/data/CoCo/val/input"
    "$GT_DIR"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required input: $path"
      exit 1
    fi
  done
}

mkdir -p "$EXP_ROOT"/{logs,temp}

if [[ "${1:-}" == "--check-only" ]]; then
  assert_preflight
  assert_free_space
  echo "End-to-end repeatability sanity preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Estimated disk use: <1G if KEEP_OUTPUTS=0"
  echo "Estimated wall-clock: ~35m to ~60m for REPEATS=$REPEATS"
  echo "Free disk now: $(available_gb)G"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  usage
  exit 2
fi

assert_preflight
assert_free_space
check_gpu_idle

for repeat in $(seq 1 "$REPEATS"); do
  echo "[$(date '+%F %T')] Repeat $repeat/$REPEATS"
  REPEAT_ROOT="$EXP_ROOT/temp/repeat$repeat"
  rm -rf "$REPEAT_ROOT"
  mkdir -p "$REPEAT_ROOT"

  DIFFW_OUT="$REPEAT_ROOT/diffw_like"
  O256_BASE_OUT="$REPEAT_ROOT/o256_base"
  O256_RGB_APPLY="$REPEAT_ROOT/o256_rgb/apply"

  (
    cd "$DIFFW_ROOT"
    run_timed "$EXP_ROOT/logs/repeat${repeat}_diffw_like_eval.log" \
      "$PY" eval_diffusion.py \
        --config "$DIFFW_EVAL_CONFIG" \
        --resume "$DIFFW_CKPT" \
        --test_set CoCo \
        --image_folder "$DIFFW_OUT" \
        --run_name finaltest1000 \
        --seed 61
  )

  (
    cd "$ROOT"
    run_timed "$EXP_ROOT/logs/repeat${repeat}_o256_base_eval.log" \
      "$PY" eval_diffusion.py \
        --config "$O256_CONFIG" \
        --resume "$O256_CKPT" \
        --test_set CoCo \
        --image_folder "$O256_BASE_OUT" \
        --run_name finaltest1000 \
        --seed 61
  )

  O256_BASE_RESULTS="$O256_BASE_OUT/finaltest1000/CoCo/CoCo"
  if [[ ! -d "$O256_BASE_RESULTS" ]]; then
    echo "Missing O256-Base output directory for repeat $repeat: $O256_BASE_RESULTS"
    exit 1
  fi

  (
    cd "$ROOT"
    run_timed "$EXP_ROOT/logs/repeat${repeat}_o256_rgb_refine.log" \
      "$PY" scripts/apply_rgb_refinement.py \
        --config "$O256_CONFIG" \
        --resume "$O256_CKPT" \
        --refiner "$REFINER" \
        --watermarked_dir "$O256_BASE_RESULTS" \
        --gt_dir "$GT_DIR" \
        --output_dir "$O256_RGB_APPLY" \
        --image_size 256 \
        --batch_size 4 \
        --num_workers 2 \
        --hidden_channels 48 \
        --depth 5 \
        --max_delta 0.50 \
        --conditioning cover \
        --seed 61
  )

  if [[ "$KEEP_OUTPUTS" != "1" ]]; then
    rm -rf "$REPEAT_ROOT"
  fi
done

(
  cd "$ROOT"
  "$PY" scripts/summarize_end_to_end_repeatability_sanity.py \
    --exp_root "$EXP_ROOT" \
    --repeats "$REPEATS" \
    --num_images "$NUM_IMAGES"
)

echo "Done. Summary:"
echo "  $EXP_ROOT/repeatability_summary.md"
