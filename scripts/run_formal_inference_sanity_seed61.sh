#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

CONFIG="configs/CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
BASE_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

OUT_DIR="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_sanity_seed61"
RAW_JSON="$OUT_DIR/o256_formal_inference_sanity.json"
SUMMARY="$OUT_DIR/summary.md"

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

mkdir -p "$OUT_DIR" "$ROOT/logs"
cd "$ROOT"

required=(
  "$PY"
  "$CONFIG"
  "$BASE_CKPT"
  "$REFINER"
  "scripts/benchmark_exp8_inference.py"
  "scripts/summarize_formal_inference_sanity_seed61.py"
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
  echo "Formal inference sanity preflight passed."
  echo "Classification: F (formal benchmark sanity)"
  echo "Output summary: $SUMMARY"
  if [[ -n "$gpu_info" ]]; then
    echo "GPU:"
    echo "$gpu_info"
  fi
  if [[ -n "$gpu_processes" ]]; then
    echo "WARNING: active GPU compute processes detected:"
    echo "$gpu_processes"
    echo "Run this sanity benchmark only when GPU is idle."
  else
    echo "GPU compute status: idle"
  fi
  exit 0
fi

if [[ -n "$gpu_processes" ]]; then
  echo "Refusing to start the formal inference sanity benchmark while another GPU compute process is active:"
  echo "$gpu_processes"
  echo "Wait until the GPU is idle, then run this script again."
  exit 1
fi

echo "[$(date '+%F %T')] Start formal inference sanity (seed61)"
echo "Classification: F"

echo
echo "[$(date '+%F %T')] Step 1/2: repeated O256 base/base+module inference benchmark"
"$PY" scripts/benchmark_exp8_inference.py \
  --project_root "$ROOT" \
  --config "$CONFIG" \
  --resume "$BASE_CKPT" \
  --refiner "$REFINER" \
  --method_name "O256 formal inference sanity seed61" \
  --output "$RAW_JSON" \
  --batch_sizes 1 4 \
  --sampling_timesteps 3 \
  --warmup 30 \
  --iterations 200 \
  --refiner_iterations 1000 \
  --repeats 20 \
  --seed 61 \
  2>&1 | tee logs/formal_inference_sanity_seed61.log

echo
echo "[$(date '+%F %T')] Step 2/2: summarize repeat stability"
"$PY" scripts/summarize_formal_inference_sanity_seed61.py \
  --benchmark "$RAW_JSON" \
  --output "$SUMMARY"

echo "[$(date '+%F %T')] Formal inference sanity complete"
echo "Summary: $SUMMARY"
