#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
PIP="/home/jeffrey/miniconda3/envs/diffw/bin/pip"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/diffw_like_cost_to_o256rgb_target_20260625"
FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"
TRAIN_GT="$FORMAL_DATASET_ROOT/data/CoCo/train/gt"
EVAL_CONFIG="CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
SEED=61
DATE_TAG="20260625"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_diffw_like_cost_to_target_seed61.sh --check-only
  bash scripts/run_diffw_like_cost_to_target_seed61.sh <steps> quick
  bash scripts/run_diffw_like_cost_to_target_seed61.sh <steps> full

Examples:
  bash scripts/run_diffw_like_cost_to_target_seed61.sh 40000 quick
  bash scripts/run_diffw_like_cost_to_target_seed61.sh 40000 full

Allowed steps:
  40000, 60000, 80000, 100000

Modes:
  quick  train/eval PSNR/SSIM/LPIPS/Clean BA only
  full   run 11 non-geometric attacks after quick gate exists
USAGE
}

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  echo "[$(date '+%F %T')] RUN: $*" | tee "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

hhmmss_estimate() {
  local steps="$1"
  "$PY" - "$steps" <<'PY'
import sys
steps = int(sys.argv[1])
seconds = steps * 0.4367
total = int(round(seconds))
print(f"{total//3600:02d}:{(total%3600)//60:02d}:{total%60:02d}")
PY
}

validate_steps() {
  case "$1" in
    40000|60000|80000|100000) ;;
    *)
      echo "Unsupported steps: $1"
      echo "Allowed steps: 40000, 60000, 80000, 100000"
      exit 2
      ;;
  esac
}

write_metadata() {
  mkdir -p "$EXP_ROOT"/{generated_configs,logs,runs,eval,attacked,lpips}
  cat > "$EXP_ROOT/target_thresholds.json" <<'JSON'
{
  "source": "O256-RGB formal locked finaltest1000 result",
  "psnr": 32.9034,
  "ssim": 0.9589,
  "lpips": 0.0461,
  "clean_ba": 95.81,
  "nongeom_avg_ba": 87.38,
  "minimum_ba": 72.93
}
JSON
  cat > "$EXP_ROOT/run_config.json" <<JSON
{
  "experiment": "DiffW-like Cost-to-Target Analysis",
  "classification": "supplementary cost-to-target analysis, not formal main-result replacement",
  "seed": $SEED,
  "dataset_root": "$FORMAL_DATASET_ROOT",
  "train_split": "COCO train10000",
  "test_split": "locked finaltest1000",
  "resolution": "256x256",
  "message_length": 32,
  "diffw_root": "$DIFFW_ROOT",
  "o256_paper_root": "$ROOT",
  "output_root": "$EXP_ROOT",
  "fixed_variables": [
    "architecture",
    "optimizer",
    "learning_rate",
    "batch_size",
    "dataset train10000",
    "locked finaltest1000",
    "message length 32",
    "image size 256",
    "seed 61"
  ],
  "variable": "training steps only",
  "candidate_steps": [40000, 60000, 80000, 100000],
  "quick_gate": ["PSNR", "SSIM", "LPIPS", "Clean BA"],
  "full_gate": ["11 non-geometric attacks", "Nongeom Avg BA", "Minimum BA"]
}
JSON
}

write_train_config() {
  local steps="$1"
  local epochs=$((steps / 5000))
  local run_root="$EXP_ROOT/runs/steps_$steps"
  local config_name="CoCo_costtarget_diffw_like_steps${steps}_seed61_${DATE_TAG}.yml"
  local config_path="$DIFFW_ROOT/configs/$config_name"
  mkdir -p "$run_root"/{ckpts,logs,validation} "$EXP_ROOT/generated_configs"
  cat > "$config_path" <<YAML
data:
  dataset: CoCo
  image_size: 256
  channels: 3
  num_workers: 0
  data_dir: $run_root
  dataset_root: $FORMAL_DATASET_ROOT
  conditional: true
model:
  in_channels: 3
  out_ch: 3
  ch: 128
  ch_mult:
  - 1
  - 2
  - 3
  - 4
  num_res_blocks: 2
  attn_resolutions:
  - 16
  dropout: 0.0
  ema_rate: 0.999
  ema: true
  resamp_with_conv: true
diffusion:
  beta_schedule: linear
  beta_start: 0.0001
  beta_end: 0.02
  num_diffusion_timesteps: 1000
training:
  patch_n: 1
  batch_size: 2
  n_epochs: $epochs
  n_iters: 2000000
  snapshot_freq: 1000000000
  validation_freq: 1000000000
  save_checkpoints: false
  save_final_checkpoint: true
sampling:
  batch_size: 1
  last_only: true
optim:
  weight_decay: 0.0
  optimizer: Adam
  lr: 0.00002
  amsgrad: false
  eps: 0.00000001
watermarking:
  message_length: 32
  strength_factor: 1
  combined_attack: true
YAML
  cp "$config_path" "$EXP_ROOT/generated_configs/$config_name"
  echo "$config_name"
}

assert_preflight() {
  for required in \
    "$PY" \
    "$PIP" \
    "$ROOT/attack_eval_diffw_paper_matched.py" \
    "$ROOT/scripts/calculate_lpips_dir.py" \
    "$ROOT/scripts/summarize_ba_report.py" \
    "$ROOT/scripts/summarize_diffw_like_cost_to_target.py" \
    "$DIFFW_ROOT/train_diffusion.py" \
    "$DIFFW_ROOT/eval_diffusion.py" \
    "$DIFFW_ROOT/calculate_psnr_ssim.py" \
    "$DIFFW_ROOT/calculate_ba.py" \
    "$DIFFW_ROOT/configs/$EVAL_CONFIG" \
    "$TRAIN_GT" \
    "$FINAL_GT"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing required input: $required"
      exit 1
    fi
  done
  local train_count final_count
  train_count="$(find -L "$TRAIN_GT" -maxdepth 1 -type f | wc -l)"
  final_count="$(find -L "$FINAL_GT" -maxdepth 1 -type f | wc -l)"
  if [[ "$train_count" -ne 10000 ]]; then
    echo "Expected train10000 images, found: $train_count"
    exit 1
  fi
  if [[ "$final_count" -ne 1000 ]]; then
    echo "Expected finaltest1000 images, found: $final_count"
    exit 1
  fi
}

run_training_if_needed() {
  local steps="$1"
  local config_name="$2"
  local run_root="$EXP_ROOT/runs/steps_$steps"
  local ckpt="$run_root/ckpts/CoCo_ddpm_final.pth.tar"
  if [[ -f "$ckpt" ]]; then
    echo "[$(date '+%F %T')] Reuse existing checkpoint: $ckpt"
    return
  fi
  if [[ -d "$run_root" ]] && [[ -n "$(find "$run_root" -mindepth 1 -maxdepth 2 -type f | head -n 1)" ]]; then
    echo "Incomplete output exists without final checkpoint:"
    echo "$run_root"
    echo "Move or delete it before rerunning this steps value."
    exit 1
  fi
  mkdir -p "$run_root"/{ckpts,logs,validation}
  (
    cd "$DIFFW_ROOT"
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$config_name" \
      --resume "./ckpts/nonexistent_costtarget_steps${steps}_seed61_start.pth.tar" \
      --image_folder "$run_root/validation" \
      --seed "$SEED" \
      2>&1 | tee "$EXP_ROOT/logs/steps_${steps}_train.log"
  )
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing expected checkpoint after training: $ckpt"
    exit 1
  fi
}

run_eval_if_needed() {
  local steps="$1"
  local run_root="$EXP_ROOT/runs/steps_$steps"
  local ckpt="$run_root/ckpts/CoCo_ddpm_final.pth.tar"
  local eval_root="$EXP_ROOT/eval/steps_$steps"
  local output_dir="$eval_root/finaltest1000/CoCo/CoCo"
  if [[ -d "$output_dir" ]]; then
    local png_count msg_count
    png_count="$(find "$output_dir" -maxdepth 1 -type f -name '*.png' | wc -l)"
    msg_count="$(find "$output_dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
    if [[ "$png_count" -eq 1000 && "$msg_count" -eq 1000 ]]; then
      echo "[$(date '+%F %T')] Reuse existing eval output: $output_dir"
      return
    fi
    echo "Incomplete eval output exists: images=$png_count messages=$msg_count"
    echo "$output_dir"
    echo "Move or delete it before rerunning."
    exit 1
  fi
  (
    cd "$DIFFW_ROOT"
    /usr/bin/time -p "$PY" -u eval_diffusion.py \
      --config "$EVAL_CONFIG" \
      --resume "$ckpt" \
      --test_set CoCo \
      --image_folder "$eval_root" \
      --run_name finaltest1000 \
      --seed "$SEED" \
      2>&1 | tee "$EXP_ROOT/logs/steps_${steps}_eval.log"
  )
}

run_quick_gate() {
  local steps="$1"
  local output_dir="$EXP_ROOT/eval/steps_$steps/finaltest1000/CoCo/CoCo"
  local ckpt="$EXP_ROOT/runs/steps_$steps/ckpts/CoCo_ddpm_final.pth.tar"
  local clean_dir="$EXP_ROOT/attacked/clean_steps_$steps/Clean"

  run_logged "$EXP_ROOT/logs/steps_${steps}_psnr_ssim.log" \
    "$PY" "$DIFFW_ROOT/calculate_psnr_ssim.py" \
      --gt_path "$FINAL_GT" \
      --results_path "$output_dir"

  if ! "$PY" - <<'PY' >/dev/null 2>&1
import lpips  # type: ignore
PY
  then
    "$PIP" install lpips >/dev/null
  fi
  run_logged "$EXP_ROOT/logs/steps_${steps}_lpips.log" \
    "$PY" "$ROOT/scripts/calculate_lpips_dir.py" \
      --gt_dir "$FINAL_GT" \
      --results_dir "$output_dir" \
      --output_dir "$EXP_ROOT/lpips/steps_$steps" \
      --batch_size 8 \
      --net alex

  mkdir -p "$clean_dir"
  find "$output_dir" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.webp' \) -print0 |
    while IFS= read -r -d '' image; do
      ln -sf "$image" "$clean_dir/$(basename "$image")"
    done

  run_logged "$EXP_ROOT/logs/steps_${steps}_clean_ba.log" \
    "$PY" "$DIFFW_ROOT/calculate_ba.py" \
      --config "$DIFFW_ROOT/configs/$EVAL_CONFIG" \
      --resume "$ckpt" \
      --attack_dir "$EXP_ROOT/attacked/clean_steps_$steps" \
      --gt_message_dir "$output_dir"

  "$PY" "$ROOT/scripts/summarize_diffw_like_cost_to_target.py" \
    --exp_root "$EXP_ROOT" \
    --steps "$steps" \
    --mode quick
}

run_full_gate() {
  local steps="$1"
  local output_dir="$EXP_ROOT/eval/steps_$steps/finaltest1000/CoCo/CoCo"
  local ckpt="$EXP_ROOT/runs/steps_$steps/ckpts/CoCo_ddpm_final.pth.tar"
  local attack_run="nongeom_steps_$steps"
  local attack_dir="$EXP_ROOT/attacked/$attack_run"
  if [[ ! -f "$EXP_ROOT/quick_gate_results.csv" ]]; then
    echo "Missing quick_gate_results.csv. Run quick mode first."
    exit 1
  fi
  if [[ ! -d "$attack_dir" ]]; then
    run_logged "$EXP_ROOT/logs/steps_${steps}_attack_nongeom.log" \
      "$PY" "$ROOT/attack_eval_diffw_paper_matched.py" \
        --gt_path "$FINAL_GT" \
        --results_path "$output_dir" \
        --output_dir "$EXP_ROOT/attacked" \
        --run_name "$attack_run" \
        --attack_scope nongeom \
        --seed "$SEED"
  else
    local attack_count
    attack_count="$(find "$attack_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete attack output exists: $attack_dir"
      echo "Expected 11 attack folders, found $attack_count"
      exit 1
    fi
    echo "[$(date '+%F %T')] Reuse existing attack output: $attack_dir"
  fi

  run_logged "$EXP_ROOT/logs/steps_${steps}_nongeom_ba.log" \
    "$PY" "$DIFFW_ROOT/calculate_ba.py" \
      --config "$DIFFW_ROOT/configs/$EVAL_CONFIG" \
      --resume "$ckpt" \
      --attack_dir "$attack_dir" \
      --gt_message_dir "$output_dir"

  "$PY" "$ROOT/scripts/summarize_ba_report.py" \
    "$attack_dir/ba_report.txt" \
    --output "$attack_dir/ba_summary.txt"

  "$PY" "$ROOT/scripts/summarize_diffw_like_cost_to_target.py" \
    --exp_root "$EXP_ROOT" \
    --steps "$steps" \
    --mode full
}

if [[ "${1:-}" == "--check-only" ]]; then
  assert_preflight
  write_metadata
  echo "DiffW-like cost-to-target preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Allowed first command:"
  echo "  cd $ROOT"
  echo "  bash scripts/run_diffw_like_cost_to_target_seed61.sh 40000 quick"
  echo "Estimated 40k train time: $(hhmmss_estimate 40000)"
  echo "Free disk:"
  df -h "$EXP_ROOT" 2>/dev/null | tail -n 1 || df -h "$(dirname "$EXP_ROOT")" | tail -n 1
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

STEPS="$1"
MODE="$2"
validate_steps "$STEPS"
case "$MODE" in
  quick|full) ;;
  *)
    echo "Unsupported mode: $MODE"
    usage
    exit 2
    ;;
esac

assert_preflight
write_metadata
CONFIG_NAME="$(write_train_config "$STEPS")"
echo "[$(date '+%F %T')] DiffW-like cost-to-target steps=$STEPS mode=$MODE"
echo "Output root: $EXP_ROOT"
echo "Generated train config: $DIFFW_ROOT/configs/$CONFIG_NAME"

if [[ "$MODE" == "quick" ]]; then
  run_training_if_needed "$STEPS" "$CONFIG_NAME"
  run_eval_if_needed "$STEPS"
  run_quick_gate "$STEPS"
else
  if [[ ! -f "$EXP_ROOT/runs/steps_$STEPS/ckpts/CoCo_ddpm_final.pth.tar" ]]; then
    echo "Missing checkpoint for steps=$STEPS. Run quick mode first."
    exit 1
  fi
  if [[ ! -d "$EXP_ROOT/eval/steps_$STEPS/finaltest1000/CoCo/CoCo" ]]; then
    echo "Missing eval output for steps=$STEPS. Run quick mode first."
    exit 1
  fi
  run_full_gate "$STEPS"
fi

echo "[$(date '+%F %T')] Done."
echo "Summary: $EXP_ROOT/cost_to_target_summary.md"
