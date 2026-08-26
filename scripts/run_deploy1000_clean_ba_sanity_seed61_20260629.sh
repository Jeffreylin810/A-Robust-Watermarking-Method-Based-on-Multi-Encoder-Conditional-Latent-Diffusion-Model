#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_actual10k_end_to_end_deployment_benchmark_20260628/clean_ba_sanity_deploy1000_20260629"
FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"

DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"
O256_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

TRAIN_INPUT_DIR="$FORMAL_ROOT/data/CoCo/train/input"
TRAIN_GT_DIR="$FORMAL_ROOT/data/CoCo/train/gt"
NUM_IMAGES="${NUM_IMAGES:-1000}"
DATE_TAG="20260629"
MIN_FREE_GB="${MIN_FREE_GB:-3}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_deploy1000_clean_ba_sanity_seed61_20260629.sh --check-only
  bash scripts/run_deploy1000_clean_ba_sanity_seed61_20260629.sh

Environment:
  NUM_IMAGES=1000       number of deployment images sampled from train10000
  FORCE=1               remove existing outputs for this sanity run
  ALLOW_GPU_BUSY=1      bypass active-GPU-process refusal
  MIN_FREE_GB=3         minimum free disk space required

Purpose:
  Low-cost clean-BA sanity check for the actual train10000 deployment pipeline.
  This is not a non-geometric robustness experiment and does not replace the
  locked finaltest1000 main protocol.
USAGE
}

run_timed() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  echo "[$(date '+%F %T')] RUN: $*" | tee "$log_path"
  /usr/bin/time -p "$@" 2>&1 | tee -a "$log_path"
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

check_gpu_idle() {
  local gpu_processes
  gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$gpu_processes" && "${ALLOW_GPU_BUSY:-0}" != "1" ]]; then
    echo "Refusing to start while another GPU compute process is active:"
    echo "$gpu_processes"
    echo "Run this sanity check after the current training finishes, or set ALLOW_GPU_BUSY=1 intentionally."
    exit 1
  fi
}

assert_preflight() {
  local required=(
    "$PY"
    "$ROOT/eval_diffusion.py"
    "$ROOT/calculate_ba.py"
    "$ROOT/scripts/apply_rgb_refinement.py"
    "$ROOT/$O256_CKPT"
    "$ROOT/$REFINER"
    "$DIFFW_ROOT/eval_diffusion.py"
    "$DIFFW_ROOT/calculate_ba.py"
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
  if [[ "$input_count" -lt "$NUM_IMAGES" || "$gt_count" -lt "$NUM_IMAGES" ]]; then
    echo "Not enough train images for deploy sanity: input=$input_count, gt=$gt_count, need=$NUM_IMAGES"
    exit 1
  fi
}

prepare_deploy_subset() {
  local subset_root="$EXP_ROOT/deploy${NUM_IMAGES}_dataset"
  local val_input="$subset_root/data/CoCo/val/input"
  local val_gt="$subset_root/data/CoCo/val/gt"
  mkdir -p "$val_input" "$val_gt" "$subset_root/data/CoCo"
  ln -sfn "$FORMAL_ROOT/data/CoCo/train" "$subset_root/data/CoCo/train"

  if [[ "$(find -L "$val_input" -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq "$NUM_IMAGES" ]]; then
    echo "$subset_root"
    return
  fi

  rm -f "$val_input"/* "$val_gt"/*
  mapfile -t inputs < <(find -L "$TRAIN_INPUT_DIR" -maxdepth 1 -type f | sort | head -n "$NUM_IMAGES")
  local src name gt
  for src in "${inputs[@]}"; do
    name="$(basename "$src")"
    gt="$TRAIN_GT_DIR/$name"
    if [[ ! -f "$gt" ]]; then
      echo "Missing matching GT for $name: $gt"
      exit 1
    fi
    ln -s "$src" "$val_input/$name"
    ln -s "$gt" "$val_gt/$name"
  done
  echo "$subset_root"
}

write_diffw_config() {
  local subset_root="$1"
  local config_name="CoCo_deploy${NUM_IMAGES}_cleanba_diffw_seed61_${DATE_TAG}.yml"
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
  local subset_root="$1"
  local config_name="CoCo_deploy${NUM_IMAGES}_cleanba_o256_seed61_${DATE_TAG}.yml"
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

link_clean_dir() {
  local source_dir="$1"
  local clean_root="$2"
  mkdir -p "$clean_root/Clean"
  find "$source_dir" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.webp' \) -print0 |
    while IFS= read -r -d '' image; do
      local target="$clean_root/Clean/$(basename "$image")"
      [[ -e "$target" ]] || ln "$image" "$target"
    done
}

write_run_config() {
  mkdir -p "$EXP_ROOT"/{logs,outputs,generated_configs,clean_ba}
  cat > "$EXP_ROOT/run_config.json" <<JSON
{
  "experiment": "deploy clean BA sanity",
  "date": "$DATE_TAG",
  "classification": "supplementary deployment clean-BA sanity check",
  "num_images": $NUM_IMAGES,
  "source_split": "first NUM_IMAGES images from train10000 deployment split",
  "purpose": "verify that the deployment embedding pipeline does not collapse clean message recovery",
  "attacks": "none; clean BA only",
  "notes": [
    "This is not a formal non-geometric robustness experiment.",
    "This does not replace locked finaltest1000 formal BA.",
    "Use it only to support the actual 10k deployment throughput benchmark."
  ]
}
JSON
}

parse_and_summarize() {
  "$PY" - "$EXP_ROOT" "$NUM_IMAGES" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
num_images = int(sys.argv[2])
specs = [
    ("DiffW-like", root / "clean_ba" / "diffw_like_clean" / "ba_report.txt"),
    ("O256-Base", root / "clean_ba" / "o256_base_clean" / "ba_report.txt"),
    ("O256-RGB", root / "outputs" / "o256_rgb" / "apply" / "attacked_clean" / "ba_report.txt"),
]
rows = []
for method, path in specs:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Clean\s*\|\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        raise SystemExit(f"Could not parse Clean BA from {path}")
    rows.append({"method": method, "num_images": num_images, "clean_ba": float(match.group(1)), "source": str(path)})

csv_path = root / "deploy_clean_ba_sanity_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "num_images", "clean_ba", "source"])
    writer.writeheader()
    writer.writerows(rows)

json_path = root / "deploy_clean_ba_sanity_summary.json"
json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

md_lines = [
    "# Deploy Clean BA Sanity Summary",
    "",
    f"- num_images: `{num_images}`",
    "- source split: first images from formal train10000 deployment split",
    "- attacks: none; clean BA only",
    "- classification: supplementary sanity check for deployment throughput experiment",
    "",
    "| Method | N | Clean BA | Source |",
    "|---|---:|---:|---|",
]
for row in rows:
    md_lines.append(f"| {row['method']} | {row['num_images']} | {row['clean_ba']:.2f}% | `{row['source']}` |")
md_lines += [
    "",
    "## Interpretation",
    "",
    "This check verifies clean message recovery on the deployment embedding pipeline.",
    "It should not be used as a replacement for locked finaltest1000 robustness results,",
    "and it does not evaluate non-geometric attacks.",
]
(root / "deploy_clean_ba_sanity_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(root / "deploy_clean_ba_sanity_summary.md")
PY
}

if [[ "${1:-}" == "--check-only" ]]; then
  assert_preflight
  assert_free_space
  write_run_config
  subset_root="$(prepare_deploy_subset)"
  echo "Deploy clean BA sanity preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Subset root: $subset_root"
  echo "NUM_IMAGES: $NUM_IMAGES"
  echo "Estimated disk use: <1G for NUM_IMAGES=1000"
  echo "Estimated wall-clock: ~15-30 minutes after GPU is free"
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
if [[ -e "$EXP_ROOT/outputs" && "${FORCE:-0}" != "1" ]]; then
  echo "Output already exists: $EXP_ROOT/outputs"
  echo "Use FORCE=1 to remove and rerun."
  exit 1
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -rf "$EXP_ROOT/outputs" "$EXP_ROOT/clean_ba" "$EXP_ROOT/logs"
fi

write_run_config
subset_root="$(prepare_deploy_subset)"
diffw_config="$(write_diffw_config "$subset_root")"
o256_config="$(write_o256_config "$subset_root")"

DIFFW_OUT="$EXP_ROOT/outputs/diffw_like/deploy${NUM_IMAGES}"
O256_BASE_OUT="$EXP_ROOT/outputs/o256_base/deploy${NUM_IMAGES}"
O256_RGB_APPLY="$EXP_ROOT/outputs/o256_rgb/apply"

echo "[$(date '+%F %T')] Step 1/6: DiffW-like deployment embedding"
(
  cd "$DIFFW_ROOT"
  run_timed "$EXP_ROOT/logs/diffw_like_eval.log" \
    "$PY" eval_diffusion.py \
      --config "$diffw_config" \
      --resume "$DIFFW_CKPT" \
      --test_set CoCo \
      --image_folder "$DIFFW_OUT" \
      --run_name deploy${NUM_IMAGES} \
      --seed 61
)
DIFFW_RESULTS="$DIFFW_OUT/deploy${NUM_IMAGES}/CoCo/CoCo"
if [[ ! -d "$DIFFW_RESULTS" ]]; then
  echo "Missing DiffW-like output directory: $DIFFW_RESULTS"
  exit 1
fi
link_clean_dir "$DIFFW_RESULTS" "$EXP_ROOT/clean_ba/diffw_like_clean"

echo "[$(date '+%F %T')] Step 2/6: DiffW-like clean BA"
(
  cd "$DIFFW_ROOT"
  run_timed "$EXP_ROOT/logs/diffw_like_clean_ba.log" \
    "$PY" calculate_ba.py \
      --config "$diffw_config" \
      --resume "$DIFFW_CKPT" \
      --attack_dir "$EXP_ROOT/clean_ba/diffw_like_clean" \
      --gt_message_dir "$DIFFW_RESULTS"
)

echo "[$(date '+%F %T')] Step 3/6: O256-Base deployment embedding"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/o256_base_eval.log" \
    "$PY" eval_diffusion.py \
      --config "$o256_config" \
      --resume "$O256_CKPT" \
      --test_set CoCo \
      --image_folder "$O256_BASE_OUT" \
      --run_name deploy${NUM_IMAGES} \
      --seed 61
)
O256_BASE_RESULTS="$O256_BASE_OUT/deploy${NUM_IMAGES}/CoCo/CoCo"
if [[ ! -d "$O256_BASE_RESULTS" ]]; then
  echo "Missing O256-Base output directory: $O256_BASE_RESULTS"
  exit 1
fi
link_clean_dir "$O256_BASE_RESULTS" "$EXP_ROOT/clean_ba/o256_base_clean"

echo "[$(date '+%F %T')] Step 4/6: O256-Base clean BA"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/o256_base_clean_ba.log" \
    "$PY" calculate_ba.py \
      --config "$o256_config" \
      --resume "$O256_CKPT" \
      --attack_dir "$EXP_ROOT/clean_ba/o256_base_clean" \
      --gt_message_dir "$O256_BASE_RESULTS"
)

echo "[$(date '+%F %T')] Step 5/6: O256-RGB refinement apply and clean BA"
(
  cd "$ROOT"
  run_timed "$EXP_ROOT/logs/o256_rgb_apply.log" \
    "$PY" scripts/apply_rgb_refinement.py \
      --config "$o256_config" \
      --resume "$O256_CKPT" \
      --refiner "$REFINER" \
      --watermarked_dir "$O256_BASE_RESULTS" \
      --gt_dir "$subset_root/data/CoCo/val/gt" \
      --output_dir "$O256_RGB_APPLY" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed 61

  run_timed "$EXP_ROOT/logs/o256_rgb_clean_ba.log" \
    "$PY" calculate_ba.py \
      --config "$o256_config" \
      --resume "$O256_CKPT" \
      --attack_dir "$O256_RGB_APPLY/attacked_clean" \
      --gt_message_dir "$O256_BASE_RESULTS"
)

echo "[$(date '+%F %T')] Step 6/6: summarize"
parse_and_summarize

echo "Done. Summary:"
echo "  $EXP_ROOT/deploy_clean_ba_sanity_summary.md"

