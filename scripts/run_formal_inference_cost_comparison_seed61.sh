#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626"
FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
DATE_TAG="20260626"

DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"
O256_CONFIG="configs/CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
O256_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_formal_inference_cost_comparison_seed61.sh --check-only
  bash scripts/run_formal_inference_cost_comparison_seed61.sh compute
  bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end 100
  bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end 300
  bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end 1000
  bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end all
  bash scripts/run_formal_inference_cost_comparison_seed61.sh summarize

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

assert_preflight() {
  local required=(
    "$PY"
    "$ROOT/scripts/benchmark_exp8_inference.py"
    "$ROOT/scripts/apply_rgb_refinement.py"
    "$ROOT/scripts/summarize_formal_inference_cost_comparison.py"
    "$ROOT/$O256_CONFIG"
    "$ROOT/$O256_CKPT"
    "$ROOT/$REFINER"
    "$DIFFW_ROOT/eval_diffusion.py"
    "$DIFFW_ROOT/configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
    "$DIFFW_CKPT"
    "$FORMAL_ROOT/data/CoCo/val/input"
    "$FORMAL_ROOT/data/CoCo/val/gt"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required input: $path"
      exit 1
    fi
  done
  local val_count
  val_count="$(find -L "$FORMAL_ROOT/data/CoCo/val/input" -maxdepth 1 -type f | wc -l)"
  if [[ "$val_count" -ne 1000 ]]; then
    echo "Expected locked finaltest1000 input count = 1000, found: $val_count"
    exit 1
  fi
}

write_run_config() {
  mkdir -p "$EXP_ROOT"/{raw,logs,figures,subsets,end_to_end_outputs,generated_configs}
  cat > "$EXP_ROOT/run_config.json" <<JSON
{
  "experiment": "Formal Inference Cost Comparison",
  "classification": "formal efficiency support plus supplementary deployment estimate",
  "seed": 61,
  "gpu_target": "NVIDIA RTX 4090",
  "formal_dataset_root": "$FORMAL_ROOT",
  "output_root": "$EXP_ROOT",
  "compute_only": {
    "batch_sizes": [1, 2, 4],
    "warmup": 20,
    "iterations": 100,
    "refiner_iterations": 500,
    "repeats": 5,
    "excludes": ["dataset I/O", "PNG writing", "PSNR", "SSIM", "BA", "attacks"]
  },
  "end_to_end": {
    "subset_sizes": [100, 300, 1000],
    "includes": ["dataset read", "model inference", "PNG writing", "message output"],
    "o256_rgb_definition": "O256-Base eval_diffusion plus RGB refiner apply"
  },
  "projection": {
    "num_images": 111000,
    "status": "projection from measured per-image timing; not direct 111k measurement"
  }
}
JSON
}

write_diffw_config() {
  local n="$1"
  local subset_root="$EXP_ROOT/subsets/eval$n"
  local config_name="CoCo_inferencecost_diffw_eval${n}_seed61_${DATE_TAG}.yml"
  cat > "$DIFFW_ROOT/configs/$config_name" <<YAML
data:
    dataset: "CoCo"
    image_size: 256
    channels: 3
    num_workers: 0
    data_dir: "./"
    dataset_root: "$subset_root"
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
  local n="$1"
  local subset_root="$EXP_ROOT/subsets/eval$n"
  local config_name="CoCo_inferencecost_o256_eval${n}_seed61_${DATE_TAG}.yml"
  cat > "$ROOT/configs/$config_name" <<YAML
data:
    dataset: "CoCo"
    image_size: 256
    channels: 3
    num_workers: 0
    data_dir: "./"
    dataset_root: "$subset_root"
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

prepare_subset() {
  local n="$1"
  local subset="$EXP_ROOT/subsets/eval$n"
  local val_input="$subset/data/CoCo/val/input"
  local val_gt="$subset/data/CoCo/val/gt"
  echo "[$(date '+%F %T')] Preparing locked finaltest subset eval$n"
  mkdir -p "$subset/data/CoCo/val/input" "$subset/data/CoCo/val/gt" "$subset/data/CoCo"
  ln -sfn "$FORMAL_ROOT/data/CoCo/train" "$subset/data/CoCo/train"
  if [[ "$(find -L "$val_input" -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq "$n" ]]; then
    echo "[$(date '+%F %T')] Reusing existing subset eval$n ($n images)"
    return
  fi
  rm -f "$val_input"/* "$val_gt"/*
  local all_inputs=()
  mapfile -t all_inputs < <(find -L "$FORMAL_ROOT/data/CoCo/val/input" -maxdepth 1 -type f | sort)
  if [[ "${#all_inputs[@]}" -lt "$n" ]]; then
    echo "Not enough locked finaltest images for eval$n: found ${#all_inputs[@]}"
    exit 1
  fi
  local idx input_path name
  for ((idx = 0; idx < n; idx++)); do
    input_path="${all_inputs[$idx]}"
    name="$(basename "$input_path")"
    ln -sfn "$(readlink -f "$input_path")" "$val_input/$name"
    ln -sfn "$(readlink -f "$FORMAL_ROOT/data/CoCo/val/gt/$name")" "$val_gt/$name"
  done
  local count
  count="$(find -L "$val_input" -maxdepth 1 -type f | wc -l)"
  if [[ "$count" -ne "$n" ]]; then
    echo "Failed to prepare subset eval$n: found $count images"
    exit 1
  fi
  echo "[$(date '+%F %T')] Prepared subset eval$n ($count images)"
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

run_compute() {
  check_gpu_idle
  force_or_refuse "$EXP_ROOT/raw/compute_diffw_like.json"
  force_or_refuse "$EXP_ROOT/raw/compute_o256_rgb.json"
  run_timed "$EXP_ROOT/logs/compute_diffw_like.log" \
    "$PY" "$ROOT/scripts/benchmark_exp8_inference.py" \
      --project_root "$DIFFW_ROOT" \
      --config "configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml" \
      --resume "$DIFFW_CKPT" \
      --method_name "DiffW-like formal seed61 compute-only" \
      --output "$EXP_ROOT/raw/compute_diffw_like.json" \
      --batch_sizes 1 2 4 \
      --sampling_timesteps 3 \
      --warmup 20 \
      --iterations 100 \
      --repeats 5 \
      --seed 61

  run_timed "$EXP_ROOT/logs/compute_o256_rgb.log" \
    "$PY" "$ROOT/scripts/benchmark_exp8_inference.py" \
      --project_root "$ROOT" \
      --config "$O256_CONFIG" \
      --resume "$O256_CKPT" \
      --refiner "$REFINER" \
      --method_name "O256-RGB formal seed61 compute-only" \
      --output "$EXP_ROOT/raw/compute_o256_rgb.json" \
      --batch_sizes 1 2 4 \
      --sampling_timesteps 3 \
      --warmup 20 \
      --iterations 100 \
      --refiner_iterations 500 \
      --repeats 5 \
      --seed 61

  "$PY" "$ROOT/scripts/summarize_formal_inference_cost_comparison.py" --exp_root "$EXP_ROOT"
}

run_end2end_one() {
  local n="$1"
  check_gpu_idle
  echo "[$(date '+%F %T')] Step end2end eval$n: start"
  prepare_subset "$n"
  local diffw_config o256_config
  diffw_config="$(write_diffw_config "$n")"
  o256_config="$(write_o256_config "$n")"
  echo "[$(date '+%F %T')] Step end2end eval$n: generated configs"

  local diffw_out="$EXP_ROOT/end_to_end_outputs/diffw_like/eval$n"
  local o256_base_out="$EXP_ROOT/end_to_end_outputs/o256_base/eval$n"
  local o256_rgb_out="$EXP_ROOT/end_to_end_outputs/o256_rgb/eval$n/apply"
  force_or_refuse "$diffw_out"
  force_or_refuse "$o256_base_out"
  force_or_refuse "$EXP_ROOT/end_to_end_outputs/o256_rgb/eval$n"

  echo "[$(date '+%F %T')] Step end2end eval$n: DiffW-like embedding"
  (
    cd "$DIFFW_ROOT"
    run_timed "$EXP_ROOT/logs/end2end_eval${n}_diffw_like_eval.log" \
      "$PY" eval_diffusion.py \
        --config "$diffw_config" \
        --resume "$DIFFW_CKPT" \
        --test_set CoCo \
        --image_folder "$diffw_out" \
        --run_name finaltest1000 \
        --seed 61
  )

  echo "[$(date '+%F %T')] Step end2end eval$n: O256-Base embedding"
  (
    cd "$ROOT"
    run_timed "$EXP_ROOT/logs/end2end_eval${n}_o256_base_eval.log" \
      "$PY" eval_diffusion.py \
        --config "$o256_config" \
        --resume "$O256_CKPT" \
        --test_set CoCo \
        --image_folder "$o256_base_out" \
        --run_name finaltest1000 \
        --seed 61
  )

  local o256_wm="$o256_base_out/finaltest1000/CoCo/CoCo"
  if [[ ! -d "$o256_wm" ]]; then
    echo "Missing O256-Base watermarked output directory: $o256_wm"
    echo "Check eval_diffusion output layout before applying RGB refinement."
    exit 1
  fi
  local gt_dir="$EXP_ROOT/subsets/eval$n/data/CoCo/val/gt"
  echo "[$(date '+%F %T')] Step end2end eval$n: RGB refinement"
  (
    cd "$ROOT"
    run_timed "$EXP_ROOT/logs/end2end_eval${n}_o256_rgb_refine.log" \
      "$PY" scripts/apply_rgb_refinement.py \
        --config "$o256_config" \
        --resume "$O256_CKPT" \
        --refiner "$REFINER" \
        --watermarked_dir "$o256_wm" \
        --gt_dir "$gt_dir" \
        --output_dir "$o256_rgb_out" \
        --image_size 256 \
        --batch_size 4 \
        --num_workers 2 \
        --hidden_channels 48 \
        --depth 5 \
        --max_delta 0.50 \
        --conditioning cover \
        --seed 61
  )

  echo "[$(date '+%F %T')] Step end2end eval$n: summarize"
  "$PY" "$ROOT/scripts/summarize_formal_inference_cost_comparison.py" --exp_root "$EXP_ROOT"
}

run_summarize() {
  "$PY" "$ROOT/scripts/summarize_formal_inference_cost_comparison.py" --exp_root "$EXP_ROOT"
}

main() {
  if [[ "${1:-}" == "--check-only" ]]; then
    assert_preflight
    write_run_config
    echo "Formal inference cost comparison preflight passed."
    echo "Output root: $EXP_ROOT"
    echo "Suggested first command:"
    echo "  cd $ROOT"
    echo "  bash scripts/run_formal_inference_cost_comparison_seed61.sh compute"
    echo "Free disk:"
    df -h "$EXP_ROOT" 2>/dev/null | tail -n 1 || df -h "$(dirname "$EXP_ROOT")" | tail -n 1
    local gpu_processes
    gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$gpu_processes" ]]; then
      echo "WARNING: active GPU compute processes:"
      echo "$gpu_processes"
    else
      echo "GPU compute status: idle"
    fi
    exit 0
  fi

  if [[ $# -lt 1 ]]; then
    usage
    exit 2
  fi

  assert_preflight
  write_run_config
  case "$1" in
    compute)
      run_compute
      ;;
    end2end)
      case "${2:-}" in
        100|300|1000)
          run_end2end_one "$2"
          ;;
        all)
          run_end2end_one 100
          run_end2end_one 300
          run_end2end_one 1000
          ;;
        *)
          usage
          exit 2
          ;;
      esac
      ;;
    summarize)
      run_summarize
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  echo "Done. Summary: $EXP_ROOT/inference_cost_summary.md"
}

main "$@"
