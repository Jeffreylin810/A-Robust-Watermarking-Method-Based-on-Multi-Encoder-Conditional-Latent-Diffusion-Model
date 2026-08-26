#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

DIFFW_CONFIG="CoCo_train10000_val100_monitor_linear_steps20k.yml"
DIFFW_EVAL_CONFIG="configs/CoCo_train10000_eval1000_unseen_linear.yml"
DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_exp8_diffw_like_train10000_msg32_steps20k_seed61_final.pth.tar"
DIFFW_SOURCE="$DIFFW_ROOT/ckpts/CoCo_ddpm_final.pth.tar"
DIFFW_LOG="$DIFFW_ROOT/logs/exp8_diffw_like_train10000_msg32_steps20k_seed61_train.log"

OURS_CONFIG="configs/CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
OURS_CKPT="Result/CoCo_ddpm_train10000_base20k_then_nongeomlight5k_seed61_final.pth.tar"
PLUGIN="results/refinement/refine10000_targetedformal_C500_seed61/refiner_final.pth"

SUMMARY_DIR="$ROOT/results/experiment_summaries/exp8_formal_speed_cost_seed61"
RAW_DIR="$SUMMARY_DIR/raw"
DIFFW_BENCH="$RAW_DIR/diffw_like_benchmark.json"
OURS_BENCH="$RAW_DIR/o256_plugin_benchmark.json"
SUMMARY="$SUMMARY_DIR/summary.md"

CHECK_ONLY=false
SKIP_TRAIN=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ "${1:-}" == "--skip-train" ]]; then
  SKIP_TRAIN=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only|--skip-train]"
  exit 2
fi

cd "$ROOT"
mkdir -p "$RAW_DIR" "$DIFFW_ROOT/logs" "$DIFFW_ROOT/Result"

gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)"

required=(
  "$DIFFW_ROOT/train_diffusion.py"
  "$DIFFW_ROOT/configs/$DIFFW_CONFIG"
  "$DIFFW_ROOT/$DIFFW_EVAL_CONFIG"
  "$OURS_CONFIG"
  "$OURS_CKPT"
  "$PLUGIN"
  "scripts/benchmark_exp8_inference.py"
  "scripts/summarize_exp8_speed_cost.py"
  "logs/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_train.log"
  "logs/exp2b_nongeom_train10000_base20k_then_light5k_seed61_train.log"
  "logs/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61_generate_pairs.log"
  "logs/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61_stageA_train.log"
  "logs/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61_stageB_train.log"
  "logs/exp5B_refine10000_nongeomlightbase_BplusM1P00_500_seed61_train.log"
  "logs/exp6_refine10000_targetedformal_C500_seed61_train.log"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

if [[ "$CHECK_ONLY" == true ]]; then
  echo "Experiment 8 preflight passed."
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  if [[ -n "$gpu_processes" ]]; then
    echo "WARNING: GPU currently has active compute processes:"
    echo "$gpu_processes"
    echo "Run the formal timing experiment only when these processes have finished."
  else
    echo "GPU compute status: idle"
  fi
  if [[ -f "$DIFFW_CKPT" ]]; then
    echo "DiffW-like checkpoint: ready"
  else
    echo "DiffW-like checkpoint: missing; the full run will retrain it (~2.5 hours)"
  fi
  echo "O256 checkpoint: ready"
  echo "Plugin checkpoint: ready"
  echo "Output: $SUMMARY_DIR"
  exit 0
fi

if [[ -n "$gpu_processes" ]]; then
  echo "Refusing to start a formal timing run while another GPU compute process is active:"
  echo "$gpu_processes"
  echo "Wait until the GPU is idle, then run this script again."
  exit 1
fi

echo "[$(date '+%F %T')] Experiment 8: formal speed and cost evaluation"

if [[ ! -f "$DIFFW_CKPT" ]]; then
  if [[ "$SKIP_TRAIN" == true ]]; then
    echo "Missing DiffW-like checkpoint while --skip-train was requested:"
    echo "$DIFFW_CKPT"
    exit 1
  fi
  if [[ -f "$DIFFW_SOURCE" ]]; then
    echo "Refusing to overwrite an unarchived DiffW-like final checkpoint:"
    echo "$DIFFW_SOURCE"
    exit 1
  fi
  echo "[$(date '+%F %T')] Step 1/4: retrain fair DiffW-like train10000/msg32/20k baseline"
  (
    cd "$DIFFW_ROOT"
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$DIFFW_CONFIG" \
      --resume ./ckpts/nonexistent_exp8_diffw_like_start.pth.tar \
      --image_folder results/validation/exp8_diffw_like_train10000_msg32_steps20k_seed61 \
      --seed 61 \
      2>&1 | tee "$DIFFW_LOG"
  )
  if [[ ! -f "$DIFFW_SOURCE" ]]; then
    echo "Training completed without expected checkpoint: $DIFFW_SOURCE"
    exit 1
  fi
  mv "$DIFFW_SOURCE" "$DIFFW_CKPT"
  rm -f "$DIFFW_ROOT/ckpts/CoCo_ddpm.pth.tar"
else
  echo "[$(date '+%F %T')] Step 1/4: reuse completed fair DiffW-like checkpoint"
fi

echo "[$(date '+%F %T')] Step 2/4: benchmark DiffW-like GPU inference"
"$PY" scripts/benchmark_exp8_inference.py \
  --project_root "$DIFFW_ROOT" \
  --config "$DIFFW_EVAL_CONFIG" \
  --resume "$DIFFW_CKPT" \
  --method_name "DiffW-like train10000/msg32/20k" \
  --output "$DIFFW_BENCH" \
  --batch_sizes 1 4 \
  --sampling_timesteps 3 \
  --warmup 20 \
  --iterations 100 \
  --refiner_iterations 500 \
  --repeats 5 \
  --seed 61 \
  2>&1 | tee "$ROOT/logs/exp8_diffw_like_inference_benchmark.log"

echo "[$(date '+%F %T')] Step 3/4: benchmark O256 base and plugin GPU inference"
"$PY" scripts/benchmark_exp8_inference.py \
  --project_root "$ROOT" \
  --config "$OURS_CONFIG" \
  --resume "$OURS_CKPT" \
  --refiner "$PLUGIN" \
  --method_name "O256 train10000/msg32/20k+5k" \
  --output "$OURS_BENCH" \
  --batch_sizes 1 4 \
  --sampling_timesteps 3 \
  --warmup 20 \
  --iterations 100 \
  --refiner_iterations 500 \
  --repeats 5 \
  --seed 61 \
  2>&1 | tee logs/exp8_o256_plugin_inference_benchmark.log

echo "[$(date '+%F %T')] Step 4/4: summarize inference and one-time costs"
"$PY" scripts/summarize_exp8_speed_cost.py \
  --diffw_benchmark "$DIFFW_BENCH" \
  --ours_benchmark "$OURS_BENCH" \
  --diffw_train_log "$DIFFW_LOG" \
  --base20_log logs/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_train.log \
  --base5_log logs/exp2b_nongeom_train10000_base20k_then_light5k_seed61_train.log \
  --pairs_log logs/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61_generate_pairs.log \
  --stage_a_log logs/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61_stageA_train.log \
  --stage_b_log logs/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61_stageB_train.log \
  --stage_bplus_log logs/exp5B_refine10000_nongeomlightbase_BplusM1P00_500_seed61_train.log \
  --stage_c_log logs/exp6_refine10000_targetedformal_C500_seed61_train.log \
  --output "$SUMMARY"

echo "[$(date '+%F %T')] Experiment 8 complete"
echo "Summary: $SUMMARY"
