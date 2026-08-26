#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

OURS_CONFIG="configs/CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
OURS_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
OURS_REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

DIFFW_CONFIG="configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"

FORMAL_DIR="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_speed_cost_seed61"
RAW_DIR="$FORMAL_DIR/raw"
SUMMARY="$FORMAL_DIR/summary.md"
DIFFW_BENCH="$RAW_DIR/diffw_like_formal_benchmark.json"
OURS_BENCH="$RAW_DIR/o256_formal_benchmark.json"

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

mkdir -p "$RAW_DIR" "$ROOT/logs"
cd "$ROOT"

required=(
  "$PY"
  "$ROOT/$OURS_CONFIG"
  "$ROOT/$OURS_CKPT"
  "$ROOT/$OURS_REFINER"
  "$DIFFW_ROOT/$DIFFW_CONFIG"
  "$DIFFW_CKPT"
  "$ROOT/scripts/benchmark_exp8_inference.py"
  "$ROOT/scripts/summarize_formal_efficiency_seed61.py"
  "$ROOT/logs/m01_formal_seed61_identity20k_train.log"
  "$ROOT/logs/m01_formal_seed61_base30k_train.log"
  "$ROOT/logs/m02_formal_seed61_generate_pairs.log"
  "$ROOT/logs/m02_formal_seed61_stageA.log"
  "$ROOT/logs/m02_formal_seed61_stageB.log"
  "$ROOT/logs/m02_formal_seed61_stageBplus.log"
  "$ROOT/logs/m02_formal_seed61_stageC.log"
  "$DIFFW_ROOT/logs/m04_formal_diffw_like_train.log"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

gpu_info="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)"
gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Formal efficiency preflight passed."
  echo "Classification: F (chapter-4 formal efficiency experiment)"
  echo "Summary output: $SUMMARY"
  if [[ -n "$gpu_info" ]]; then
    echo "GPU:"
    echo "$gpu_info"
  else
    echo "WARNING: nvidia-smi did not return GPU info."
  fi
  if [[ -n "$gpu_processes" ]]; then
    echo "WARNING: active GPU compute processes detected:"
    echo "$gpu_processes"
    echo "Run the benchmark only when the GPU is idle for clean timing."
  else
    echo "GPU compute status: idle"
  fi
  echo "Artifacts:"
  echo "  DiffW-like checkpoint: $DIFFW_CKPT"
  echo "  O256 base checkpoint: $ROOT/$OURS_CKPT"
  echo "  O256 module: $ROOT/$OURS_REFINER"
  exit 0
fi

if [[ -n "$gpu_processes" ]]; then
  echo "Refusing to start the formal efficiency benchmark while another GPU compute process is active:"
  echo "$gpu_processes"
  echo "Wait until the GPU is idle, then run this script again."
  exit 1
fi

echo "[$(date '+%F %T')] Start formal efficiency benchmark (seed61)"
echo "Classification: F"

echo
echo "[$(date '+%F %T')] Step 1/3: benchmark formal DiffW-like inference"
if [[ -f "$DIFFW_BENCH" ]]; then
  echo "Reuse existing DiffW-like benchmark JSON."
else
  "$PY" scripts/benchmark_exp8_inference.py \
    --project_root "$DIFFW_ROOT" \
    --config "$DIFFW_CONFIG" \
    --resume "$DIFFW_CKPT" \
    --method_name "DiffW-like formal train10000/msg32/20k" \
    --output "$DIFFW_BENCH" \
    --batch_sizes 1 4 \
    --sampling_timesteps 3 \
    --warmup 20 \
    --iterations 100 \
    --refiner_iterations 500 \
    --repeats 5 \
    --seed 61 \
    2>&1 | tee "$ROOT/logs/formal_efficiency_diffw_like_benchmark.log"
fi

echo
echo "[$(date '+%F %T')] Step 2/3: benchmark formal O256 base and module inference"
if [[ -f "$OURS_BENCH" ]]; then
  echo "Reuse existing O256 benchmark JSON."
else
  "$PY" scripts/benchmark_exp8_inference.py \
    --project_root "$ROOT" \
    --config "$OURS_CONFIG" \
    --resume "$OURS_CKPT" \
    --refiner "$OURS_REFINER" \
    --method_name "O256-RGBREFINE formal seed61" \
    --output "$OURS_BENCH" \
    --batch_sizes 1 4 \
    --sampling_timesteps 3 \
    --warmup 20 \
    --iterations 100 \
    --refiner_iterations 500 \
    --repeats 5 \
    --seed 61 \
    2>&1 | tee "$ROOT/logs/formal_efficiency_o256_benchmark.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/3: summarize formal efficiency"
"$PY" scripts/summarize_formal_efficiency_seed61.py \
  --diffw_benchmark "$DIFFW_BENCH" \
  --ours_benchmark "$OURS_BENCH" \
  --diffw_train_log "$DIFFW_ROOT/logs/m04_formal_diffw_like_train.log" \
  --base20_log "$ROOT/logs/m01_formal_seed61_identity20k_train.log" \
  --base10_log "$ROOT/logs/m01_formal_seed61_base30k_train.log" \
  --pairs_log "$ROOT/logs/m02_formal_seed61_generate_pairs.log" \
  --stage_a_log "$ROOT/logs/m02_formal_seed61_stageA.log" \
  --stage_b_log "$ROOT/logs/m02_formal_seed61_stageB.log" \
  --stage_bplus_log "$ROOT/logs/m02_formal_seed61_stageBplus.log" \
  --stage_c_log "$ROOT/logs/m02_formal_seed61_stageC.log" \
  --output "$SUMMARY"

echo "[$(date '+%F %T')] Formal efficiency benchmark complete"
echo "Summary: $SUMMARY"
