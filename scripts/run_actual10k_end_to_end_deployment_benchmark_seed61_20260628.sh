#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_actual10k_end_to_end_deployment_benchmark_20260628"
FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
DEPLOY_ROOT="$EXP_ROOT/deploy10000_dataset"

DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"
O256_CONFIG_BASE="configs/CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
O256_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"
TRAIN_INPUT_DIR="$FORMAL_ROOT/data/CoCo/train/input"
TRAIN_GT_DIR="$FORMAL_ROOT/data/CoCo/train/gt"
NUM_IMAGES=10000
DATE_TAG="20260628"
MIN_FREE_GB=12

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_actual10k_end_to_end_deployment_benchmark_seed61_20260628.sh --check-only
  bash scripts/run_actual10k_end_to_end_deployment_benchmark_seed61_20260628.sh

Environment:
  FORCE=1             remove existing outputs for the requested run and rerun
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
    "$ROOT/scripts/apply_rgb_refinement.py"
    "$ROOT/scripts/summarize_actual10k_end_to_end_deployment_benchmark.py"
    "$ROOT/$O256_CONFIG_BASE"
    "$ROOT/$O256_CKPT"
    "$ROOT/$REFINER"
    "$DIFFW_ROOT/eval_diffusion.py"
    "$DIFFW_ROOT/configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
    "$DIFFW_CKPT"
    "$TRAIN_INPUT_DIR"
    "$TRAIN_GT_DIR"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required input: $path"
      exit 1
    fi
  done
  local input_count gt_count
  input_count="$(find -L "$TRAIN_INPUT_DIR" -maxdepth 1 -type f | wc -l)"
  gt_count="$(find -L "$TRAIN_GT_DIR" -maxdepth 1 -type f | wc -l)"
  if [[ "$input_count" -ne "$NUM_IMAGES" || "$gt_count" -ne "$NUM_IMAGES" ]]; then
    echo "Expected train10000 input/gt counts = $NUM_IMAGES/$NUM_IMAGES, found: $input_count/$gt_count"
    exit 1
  fi
}

prepare_deploy_dataset() {
  echo "[$(date '+%F %T')] Preparing actual deploy10000 dataset root"
  rm -rf "$DEPLOY_ROOT/data/CoCo/val"
  mkdir -p "$DEPLOY_ROOT/data/CoCo/val" "$DEPLOY_ROOT/data/CoCo"
  ln -sfn "$FORMAL_ROOT/data/CoCo/train" "$DEPLOY_ROOT/data/CoCo/train"
  ln -sfn "$FORMAL_ROOT/data/CoCo/train/input" "$DEPLOY_ROOT/data/CoCo/val/input"
  ln -sfn "$FORMAL_ROOT/data/CoCo/train/gt" "$DEPLOY_ROOT/data/CoCo/val/gt"
  local input_count gt_count
  input_count="$(find -L "$DEPLOY_ROOT/data/CoCo/val/input" -maxdepth 1 -type f | wc -l)"
  gt_count="$(find -L "$DEPLOY_ROOT/data/CoCo/val/gt" -maxdepth 1 -type f | wc -l)"
  if [[ "$input_count" -ne "$NUM_IMAGES" || "$gt_count" -ne "$NUM_IMAGES" ]]; then
    echo "deploy10000 dataset root validation failed: $input_count inputs, $gt_count gt"
    exit 1
  fi
}

write_diffw_config() {
  local config_name="CoCo_actual10k_deploy_diffw_seed61_${DATE_TAG}.yml"
  cat > "$DIFFW_ROOT/configs/$config_name" <<YAML
data:
    dataset: "CoCo"
    image_size: 256
    channels: 3
    num_workers: 0
    data_dir: "./"
    dataset_root: "$DEPLOY_ROOT"
    conditional: True
model:
    in_channels: 3
    out_ch: 3
    ch: 128
    ch_mult: [1, 2, 3, 4]
    num_res_blocks: 2
    attn_resolutions: [16, ]
    dropout: 0.0
    ema_rate: 0.999
    ema: True
    resamp_with_conv: True
diffusion:
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 1000
training:
    patch_n: 1
    batch_size: 2
    n_epochs: 4
    n_iters: 2000000
    snapshot_freq: 100000000
    validation_freq: 100000000
sampling:
    batch_size: 1
    last_only: True
optim:
    weight_decay: 0.000
    optimizer: "Adam"
    lr: 0.00002
    amsgrad: False
    eps: 0.00000001
watermarking:
    message_length: 32
    strength_factor: 1
    combined_attack: True
YAML
  cp "$DIFFW_ROOT/configs/$config_name" "$EXP_ROOT/generated_configs/$config_name"
  echo "$config_name"
}

write_o256_config() {
  local config_name="CoCo_actual10k_deploy_o256_seed61_${DATE_TAG}.yml"
  cat > "$ROOT/configs/$config_name" <<YAML
data:
    dataset: "CoCo"
    image_size: 256
    channels: 3
    num_workers: 0
    data_dir: "./"
    dataset_root: "$DEPLOY_ROOT"
    output_dir: "./"
    conditional: True
model:
    in_channels: 3
    out_ch: 3
    resolution: 256
    ch: 128
    ch_mult: [1, 2, 3, 4]
    num_res_blocks: 2
    attn_resolutions: [16, ]
    dropout: 0.0
    ema_rate: 0.999
    ema: True
    resamp_with_conv: True
diffusion:
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 1000
training:
    patch_n: 1
    batch_size: 2
    n_epochs: 5
    n_iters: 2000000
    snapshot_freq: 100000000
    validation_freq: 100000000
    save_checkpoints: False
    save_final_checkpoint: True
    detailed_profile: False
    profile_interval: 10
sampling:
    batch_size: 1
    last_only: True
optim:
    weight_decay: 0.000
    optimizer: "Adam"
    lr: 0.00001
    amsgrad: False
    eps: 0.00000001
watermarking:
    message_length: 32
    strength_factor: 1
    x0_loss_weight: 1
    message_loss_weight: 30
    rgb_loss_weight: 20
    rgb_psnr_hinge_weight: 200
    rgb_psnr_target: 36.0
    rgb_psnr_max_value: 2.0
    message_map_mode: "bitplane"
    attack_profile: "identity_nongeom_curriculum_light"
    attack_uses_cover: False
    combined_attack: True
latent:
    enabled: True
    model_path: "/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base"
    scaling_factor:
    deterministic_encode: True
YAML
  cp "$ROOT/configs/$config_name" "$EXP_ROOT/generated_configs/$config_name"
  echo "$config_name"
}

write_run_config() {
  mkdir -p "$EXP_ROOT"/{logs,end_to_end_outputs,generated_configs}
  cat > "$EXP_ROOT/run_config.json" <<JSON
{
  "experiment": "Actual 10k End-to-End Deployment Benchmark",
  "classification": "supplementary deployment throughput benchmark",
  "seed": 61,
  "dataset_root": "$DEPLOY_ROOT",
  "source_split": "train10000 reused as deployment input set",
  "num_images": $NUM_IMAGES,
  "methods": [
    "DiffW-like",
    "O256-Base",
    "O256-RGB"
  ],
  "includes": [
    "dataset read",
    "model inference",
    "PNG writing",
    "message output",
    "RGB refiner apply for O256-RGB"
  ],
  "excludes": [
    "training cost",
    "PSNR/SSIM/LPIPS",
    "BA evaluation"
  ],
  "note": "Use for deployment-oriented throughput discussion, not full life-cycle cost superiority."
}
JSON
}

if [[ "${1:-}" == "--check-only" ]]; then
  assert_preflight
  assert_free_space
  write_run_config
  prepare_deploy_dataset
  echo "Actual 10k end-to-end deployment benchmark preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Estimated disk use: ~7G"
  echo "Estimated wall-clock: ~1h50m to ~2h20m"
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
write_run_config
prepare_deploy_dataset

DIFFW_CONFIG="$(write_diffw_config)"
O256_CONFIG="$(write_o256_config)"

DIFFW_OUT="$EXP_ROOT/end_to_end_outputs/diffw_like/deploy10k"
O256_BASE_OUT="$EXP_ROOT/end_to_end_outputs/o256_base/deploy10k"
O256_RGB_ROOT="$EXP_ROOT/end_to_end_outputs/o256_rgb/deploy10k"
O256_RGB_APPLY="$O256_RGB_ROOT/apply"

force_or_refuse "$DIFFW_OUT"
force_or_refuse "$O256_BASE_OUT"
force_or_refuse "$O256_RGB_ROOT"

echo "[$(date '+%F %T')] Step 1/4: DiffW-like actual 10k deployment run"
(
  cd "$DIFFW_ROOT"
  run_timed "$EXP_ROOT/logs/deploy10k_diffw_like_eval.log" \
    "$PY" eval_diffusion.py \
      --config "$DIFFW_CONFIG" \
      --resume "$DIFFW_CKPT" \
      --test_set CoCo \
      --image_folder "$DIFFW_OUT" \
      --run_name deploy10k \
      --seed 61
)

echo "[$(date '+%F %T')] Step 2/4: O256-Base actual 10k deployment run"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/deploy10k_o256_base_eval.log" \
    "$PY" eval_diffusion.py \
      --config "$O256_CONFIG" \
      --resume "$O256_CKPT" \
      --test_set CoCo \
      --image_folder "$O256_BASE_OUT" \
      --run_name deploy10k \
      --seed 61
)

O256_BASE_RESULTS="$O256_BASE_OUT/deploy10k/CoCo/CoCo"
if [[ ! -d "$O256_BASE_RESULTS" ]]; then
  echo "Missing O256-Base output directory: $O256_BASE_RESULTS"
  exit 1
fi

echo "[$(date '+%F %T')] Step 3/4: O256-RGB actual 10k refiner apply"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/deploy10k_o256_rgb_refine.log" \
    "$PY" scripts/apply_rgb_refinement.py \
      --config "$O256_CONFIG" \
      --resume "$O256_CKPT" \
      --refiner "$REFINER" \
      --watermarked_dir "$O256_BASE_RESULTS" \
      --gt_dir "$DEPLOY_ROOT/data/CoCo/val/gt" \
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

echo "[$(date '+%F %T')] Step 4/4: summarize"
(
  cd "$ROOT"
  "$PY" scripts/summarize_actual10k_end_to_end_deployment_benchmark.py \
    --exp_root "$EXP_ROOT" \
    --num_images "$NUM_IMAGES"
)

echo "Done. Summary:"
echo "  $EXP_ROOT/actual10k_end_to_end_deployment_summary.md"
