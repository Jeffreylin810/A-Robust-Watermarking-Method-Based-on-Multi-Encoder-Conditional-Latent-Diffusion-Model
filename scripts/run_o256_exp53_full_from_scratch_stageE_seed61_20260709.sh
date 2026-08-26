#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"
TRAIN_ALIAS="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_exp53_formal"
TRAIN_GT="$TRAIN_ALIAS/data/CoCo/val/gt"

EVAL_TEMPLATE="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_exp53_20260709.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_PAIR_TEMPLATE="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_formal.yml"
TRAIN_PAIR_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_exp53_20260709.yml"

ID_TEMPLATE="CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed61_formal.yml"
BASE_TEMPLATE="CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml"

ID_CONFIG="CoCo_train10000_ours256_msg32_exp53_identity20k_seed61_20260709.yml"
BASE30_CONFIG="CoCo_train10000_ours256_msg32_exp53_base30_nongeomlight10k_seed61_20260709.yml"
BASE50_CONFIG="CoCo_train10000_ours256_msg32_exp53_base50_nongeomlight_plus20k_seed61_20260709.yml"
BASE55_CONFIG="CoCo_train10000_ours256_msg32_exp53_base55_weakattack_plus5k_seed61_20260709.yml"
BASE60_CONFIG="CoCo_train10000_ours256_msg32_exp53_base60_dropout_focus_plus5k_seed61_20260709.yml"
BASE65_CONFIG="CoCo_train10000_ours256_msg32_exp53_base65_dropout_margin_plus5k_seed61_20260709.yml"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"
PER_ATTACK_CSV="$EXP_ROOT/per_attack_ba.csv"

ID_WORKSPACE="$EXP_ROOT/workspace/base_identity20k"
BASE30_WORKSPACE="$EXP_ROOT/workspace/base30_nongeomlight10k"
BASE50_WORKSPACE="$EXP_ROOT/workspace/base50_nongeomlight_plus20k"
BASE55_WORKSPACE="$EXP_ROOT/workspace/base55_weakattack_plus5k"
BASE60_WORKSPACE="$EXP_ROOT/workspace/base60_dropout_focus_plus5k"
BASE65_WORKSPACE="$EXP_ROOT/workspace/base65_dropout_margin_plus5k"

ID_CKPT="$ID_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
BASE30_CKPT="$BASE30_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
BASE50_CKPT="$BASE50_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
BASE55_CKPT="$BASE55_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
BASE60_CKPT="$BASE60_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
BASE65_CKPT="$BASE65_WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"

BASE_VARIANT="exp53_base65_dropout_margin"
BASE_OUTPUT="$EXP_ROOT/eval_outputs/$BASE_VARIANT"
BASE_WM="$BASE_OUTPUT/finaltest1000/CoCo"
BASE_ATTACK_RUN="${BASE_VARIANT}_nongeom"

PAIR_OUTPUT_ROOT="$EXP_ROOT/train_pairs/exp53_base65_pairs"
TRAIN_WM="$PAIR_OUTPUT_ROOT/train10000/CoCo"

REF_A="$EXP_ROOT/refinement/exp53_stageA_image10k"
REF_B="$EXP_ROOT/refinement/exp53_stageB_message2k"
REF_BP="$EXP_ROOT/refinement/exp53_stageBplus_message500"
REF_C="$EXP_ROOT/refinement/exp53_stageC_targeted500"
REF_D="$EXP_ROOT/refinement/exp53_stageD_dropout_balanced500"
REF_E="$EXP_ROOT/refinement/exp53_stageE_decoderaware250"

RGB_VARIANT="exp53_rgb_full_from_scratch_stageE"
APPLY_DIR="$EXP_ROOT/apply/${RGB_VARIANT}_finaltest1000"
RGB_ATTACK_RUN="${RGB_VARIANT}_nongeom"

MIN_FREE_GB=35
EST_REQUIRED_GB=30

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/workspace" "$EXP_ROOT/eval_outputs" "$EXP_ROOT/train_pairs" "$EXP_ROOT/refinement" "$EXP_ROOT/apply" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips"

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

count_pngs() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l
}

count_msgs() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l
}

run_lpips() {
  local results_dir="$1"
  local output_dir="$2"
  local log_path="$3"
  if [[ -f "$output_dir/summary.json" ]]; then
    echo "Reuse LPIPS summary: $output_dir"
  else
    mkdir -p "$output_dir"
    "$PY" scripts/calculate_lpips_dir.py \
      --gt_dir "$FINAL_GT" \
      --results_dir "$results_dir" \
      --output_dir "$output_dir" \
      --batch_size 8 \
      --net alex \
      2>&1 | tee "$log_path"
  fi
}

ensure_train_alias() {
  if [[ ! -d "$TRAIN_ALIAS" ]]; then
    "$PY" scripts/create_coco_split_alias.py \
      --source_root "$FORMAL_DATASET_ROOT" \
      --source_split train \
      --output_root "$TRAIN_ALIAS" \
      --num_images 10000 \
      --overwrite_links >/dev/null
  fi
}

generate_train_configs() {
  "$PY" - <<PY
from pathlib import Path
import yaml

root = Path("$ROOT/configs")

def write_config(template, name, output_dir, n_epochs, attack_profile=None):
    cfg = yaml.safe_load((root / template).read_text(encoding="utf-8"))
    cfg["data"]["dataset_root"] = "$FORMAL_DATASET_ROOT"
    cfg["data"]["output_dir"] = output_dir
    cfg["training"]["n_epochs"] = n_epochs
    if attack_profile is not None:
        cfg["watermarking"]["attack_profile"] = attack_profile
    (root / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

write_config("$ID_TEMPLATE", "$ID_CONFIG", "$ID_WORKSPACE", 4, "identity_only")
write_config("$BASE_TEMPLATE", "$BASE30_CONFIG", "$BASE30_WORKSPACE", 6, "identity_nongeom_curriculum_light")
write_config("$BASE_TEMPLATE", "$BASE50_CONFIG", "$BASE50_WORKSPACE", 10, "identity_nongeom_curriculum_light")
write_config("$BASE_TEMPLATE", "$BASE55_CONFIG", "$BASE55_WORKSPACE", 11, "identity_nongeom_targeted_weak")
write_config("$BASE_TEMPLATE", "$BASE60_CONFIG", "$BASE60_WORKSPACE", 12, "identity_nongeom_dropout_focus")
write_config("$BASE_TEMPLATE", "$BASE65_CONFIG", "$BASE65_WORKSPACE", 13, "identity_nongeom_dropout_margin")

eval_cfg = yaml.safe_load((root / "$EVAL_TEMPLATE").read_text(encoding="utf-8"))
eval_cfg["data"]["dataset_root"] = "$FORMAL_DATASET_ROOT"
eval_cfg["data"]["output_dir"] = "$EXP_ROOT/eval_outputs"
(root / "$EVAL_CONFIG").write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")

pair_cfg = yaml.safe_load((root / "$TRAIN_PAIR_TEMPLATE").read_text(encoding="utf-8"))
pair_cfg["data"]["dataset_root"] = "$ROOT/$TRAIN_ALIAS"
pair_cfg["data"]["output_dir"] = "$PAIR_OUTPUT_ROOT"
(root / "$TRAIN_PAIR_CONFIG").write_text(yaml.safe_dump(pair_cfg, sort_keys=False), encoding="utf-8")
PY
}

train_base_stage() {
  local label="$1"
  local config="$2"
  local resume="$3"
  local output_dir="$4"
  local ckpt="$5"
  local validation_name="$6"
  local log_name="$7"

  if [[ -f "$ckpt" ]]; then
    echo "Reuse completed base stage: $label"
    return
  fi
  if [[ -d "$output_dir" ]]; then
    local ts
    local archived_dir
    ts="$(date '+%Y%m%d_%H%M%S')"
    archived_dir="${output_dir}_incomplete_${ts}"
    echo "Existing incomplete base workspace detected: $output_dir"
    echo "Archiving incomplete workspace to: $archived_dir"
    mv "$output_dir" "$archived_dir"
    if [[ -f "$LOG_DIR/$log_name" ]]; then
      mv "$LOG_DIR/$log_name" "$LOG_DIR/${log_name%.log}_incomplete_${ts}.log"
      echo "Archived previous log to: $LOG_DIR/${log_name%.log}_incomplete_${ts}.log"
    fi
  fi

  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$config" \
    --resume "$resume" \
    --image_folder "results/validation/$validation_name" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/$log_name"

  [[ -f "$ckpt" ]] || { echo "Missing expected checkpoint for $label: $ckpt"; exit 1; }
}

train_refiner_stage() {
  local stage_type="$1"
  local output_dir="$2"
  local resume_refiner="$3"
  local steps="$4"
  local lr="$5"
  local message_weight="$6"
  local eval_every="$7"
  local log_name="$8"

  if [[ -d "$output_dir" ]]; then
    if [[ ! -f "$output_dir/refiner_final.pth" || ! -f "$output_dir/report.txt" ]]; then
      echo "Incomplete refiner stage output: $output_dir"
      exit 1
    fi
    echo "Reuse completed refiner stage: $output_dir"
    return
  fi

  local image_mse=10.0
  local image_l1=2.5
  local command=(
    "$PY" -u scripts/train_rgb_refinement.py
    --config "$EVAL1000_CONFIG"
    --resume "$BASE65_CKPT"
    --watermarked_dir "$TRAIN_WM"
    --gt_dir "$TRAIN_GT"
    --output_dir "$output_dir"
    --image_size 256
    --steps "$steps"
    --batch_size 4
    --num_workers 2
    --lr "$lr"
    --hidden_channels 48
    --depth 5
    --max_delta 0.50
    --conditioning cover
    --image_mse_weight "$image_mse"
    --image_l1_weight "$image_l1"
    --message_weight "$message_weight"
    --regularize_weight 0.01
    --eval_every "$eval_every"
    --log_every 50
    --skip_final_dataset_images
    --seed "$SEED"
  )

  if [[ -n "$resume_refiner" ]]; then
    command+=(--resume_refiner "$resume_refiner")
  fi
  if [[ "$stage_type" == "targeted_c" ]]; then
    command+=(
      --attack_message_weight 0.15
      --attack_profile targeted_formal_weak
      --attack_noise_sigma 0.2738612788
      --attack_noise_sigma_secondary 0.15
      --attack_dropout_prob 0.70
      --attack_salt_pepper_prob 0.20
    )
  elif [[ "$stage_type" == "dropout_balanced" ]]; then
    command=(
      "$PY" -u scripts/train_rgb_refinement.py
      --config "$EVAL1000_CONFIG"
      --resume "$BASE65_CKPT"
      --resume_refiner "$resume_refiner"
      --watermarked_dir "$TRAIN_WM"
      --gt_dir "$TRAIN_GT"
      --output_dir "$output_dir"
      --image_size 256
      --steps 500
      --batch_size 4
      --num_workers 2
      --lr 1e-5
      --hidden_channels 48
      --depth 5
      --max_delta 0.50
      --conditioning cover
      --image_mse_weight 8.0
      --image_l1_weight 2.0
      --message_weight 1.4
      --attack_message_weight 0.45
      --attack_profile dropout_r03_only
      --attack_dropout_prob 0.70
      --latent_preserve_weight 0.45
      --sensitive_residual_weight 2.5
      --quality_gate_psnr 33.00
      --quality_gate_weight 70.0
      --regularize_weight 0.01
      --eval_every 250
      --log_every 50
      --skip_final_dataset_images
      --seed "$SEED"
    )
  elif [[ "$stage_type" == "decoderaware_e" ]]; then
    command=(
      "$PY" -u scripts/train_rgb_refinement.py
      --config "$EVAL1000_CONFIG"
      --resume "$BASE65_CKPT"
      --resume_refiner "$resume_refiner"
      --watermarked_dir "$TRAIN_WM"
      --gt_dir "$TRAIN_GT"
      --output_dir "$output_dir"
      --image_size 256
      --steps 250
      --batch_size 4
      --num_workers 2
      --lr 5e-6
      --hidden_channels 48
      --depth 5
      --max_delta 0.50
      --conditioning cover
      --image_mse_weight 8.0
      --image_l1_weight 2.0
      --message_weight 1.50
      --attack_message_weight 0.65
      --attack_profile dropout_r03_only
      --attack_dropout_prob 0.70
      --latent_preserve_weight 0.45
      --sensitive_residual_weight 3.00
      --quality_gate_psnr 33.80
      --quality_gate_weight 90.0
      --regularize_weight 0.01
      --residual_lap_preserve_weight 0.0
      --anti_cancel_weight 0.0
      --anti_cancel_margin 0.0
      --eval_every 125
      --log_every 50
      --skip_final_dataset_images
      --seed "$SEED"
    )
  fi

  /usr/bin/time -p "${command[@]}" 2>&1 | tee "$LOG_DIR/$log_name"
}

require_path "$PY"
require_path "$FORMAL_DATASET_ROOT"
require_path "$FINAL_GT"
require_path "configs/$ID_TEMPLATE"
require_path "configs/$BASE_TEMPLATE"
require_path "configs/$EVAL_TEMPLATE"
require_path "configs/$EVAL1000_CONFIG"
require_path "configs/$TRAIN_PAIR_TEMPLATE"
ensure_train_alias
require_path "$TRAIN_GT"
generate_train_configs
require_path "configs/$EVAL_CONFIG"
require_path "configs/$TRAIN_PAIR_CONFIG"

if [[ "$CHECK_ONLY" == true ]]; then
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
  if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
    echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
    exit 1
  fi
  echo "O256 exp53 seed61 full from-scratch StageE formal preflight passed."
  echo "Generated configs:"
  echo "  configs/$ID_CONFIG"
  echo "  configs/$BASE30_CONFIG"
  echo "  configs/$BASE50_CONFIG"
  echo "  configs/$BASE55_CONFIG"
  echo "  configs/$BASE60_CONFIG"
  echo "  configs/$BASE65_CONFIG"
  echo "  configs/$EVAL_CONFIG"
  echo "  configs/$TRAIN_PAIR_CONFIG"
  echo "Will run sequentially:"
  echo "  (1) identity20k from scratch"
  echo "  (2) nongeom-light to base30"
  echo "  (3) nongeom-light +20k to base50"
  echo "  (4) weakattack +5k to base55"
  echo "  (5) dropout-focus +5k to base60"
  echo "  (6) dropout-margin +5k to base65"
  echo "  (7) base65 finaltest1000 metrics + full nongeom BA"
  echo "  (8) regenerate train10000 pairs"
  echo "  (9) refiner A10k/B2k/B+500/C500/D-dropout-balanced500/E-decoderaware250 from scratch"
  echo "  (10) RGB StageE finaltest1000 metrics + full nongeom BA + summary"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/10: train identity20k from scratch"
train_base_stage "identity20k" "$ID_CONFIG" "./ckpts/nonexistent_exp53_identity20k_start.pth.tar" "$ID_WORKSPACE" "$ID_CKPT" "exp53_identity20k" "exp53_identity20k_train.log"

echo
echo "[$(date '+%F %T')] Step 2/10: continue to base30 nongeom-light10k"
train_base_stage "base30" "$BASE30_CONFIG" "$ID_CKPT" "$BASE30_WORKSPACE" "$BASE30_CKPT" "exp53_base30_nongeomlight10k" "exp53_base30_nongeomlight10k_train.log"

echo
echo "[$(date '+%F %T')] Step 3/10: continue to base50 nongeom-light +20k"
train_base_stage "base50" "$BASE50_CONFIG" "$BASE30_CKPT" "$BASE50_WORKSPACE" "$BASE50_CKPT" "exp53_base50_nongeomlight_plus20k" "exp53_base50_nongeomlight_plus20k_train.log"

echo
echo "[$(date '+%F %T')] Step 4/10: continue to base55 weakattack +5k"
train_base_stage "base55" "$BASE55_CONFIG" "$BASE50_CKPT" "$BASE55_WORKSPACE" "$BASE55_CKPT" "exp53_base55_weakattack_plus5k" "exp53_base55_weakattack_plus5k_train.log"

echo
echo "[$(date '+%F %T')] Step 5/10: continue to base60 dropout-focus +5k"
train_base_stage "base60" "$BASE60_CONFIG" "$BASE55_CKPT" "$BASE60_WORKSPACE" "$BASE60_CKPT" "exp53_base60_dropout_focus_plus5k" "exp53_base60_dropout_focus_plus5k_train.log"

echo
echo "[$(date '+%F %T')] Step 6/10: continue to base65 dropout-margin +5k"
train_base_stage "base65" "$BASE65_CONFIG" "$BASE60_CKPT" "$BASE65_WORKSPACE" "$BASE65_CKPT" "exp53_base65_dropout_margin_plus5k" "exp53_base65_dropout_margin_plus5k_train.log"

echo
echo "[$(date '+%F %T')] Step 7/10: evaluate base65 on locked finaltest1000"
if [[ -d "$BASE_WM" ]]; then
  png_count="$(count_pngs "$BASE_WM")"
  msg_count="$(count_msgs "$BASE_WM")"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete base65 eval output: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse completed base65 eval outputs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE65_CKPT" \
    --image_folder "$BASE_OUTPUT" \
    --run_name finaltest1000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_apply.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_psnr_ssim.log"

run_lpips "$BASE_WM" "$EXP_ROOT/lpips/$BASE_VARIANT" "$LOG_DIR/${BASE_VARIANT}_lpips.log"

mkdir -p "$EXP_ROOT/attacked_Result/clean_$BASE_VARIANT"
ln -sfn "$(realpath "$BASE_WM")" "$EXP_ROOT/attacked_Result/clean_$BASE_VARIANT/Clean"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE65_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/clean_$BASE_VARIANT" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete base attack output."
    exit 1
  fi
  echo "Reuse completed base attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$BASE_WM" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$BASE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE65_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 8/10: regenerate train10000 pairs from base65"
if [[ -d "$TRAIN_WM" ]]; then
  pair_pngs="$(count_pngs "$TRAIN_WM")"
  pair_msgs="$(count_msgs "$TRAIN_WM")"
  if [[ "$pair_pngs" -ne 10000 || "$pair_msgs" -ne 10000 ]]; then
    echo "Incomplete train pairs: images=$pair_pngs messages=$pair_msgs"
    exit 1
  fi
  echo "Reuse completed train pairs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$TRAIN_PAIR_CONFIG" \
    --resume "$BASE65_CKPT" \
    --image_folder "$PAIR_OUTPUT_ROOT" \
    --run_name train10000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp53_train_pairs_eval.log"
fi

echo
echo "[$(date '+%F %T')] Step 9/10: train RGB refiner from scratch with StageE"
train_refiner_stage image "$REF_A" "" 10000 1e-4 0.2 1000 "exp53_refiner_stageA.log"
train_refiner_stage message "$REF_B" "$REF_A/refiner_final.pth" 2000 5e-5 0.75 500 "exp53_refiner_stageB.log"
train_refiner_stage message "$REF_BP" "$REF_B/refiner_final.pth" 500 2e-5 1.0 500 "exp53_refiner_stageBplus.log"
train_refiner_stage targeted_c "$REF_C" "$REF_BP/refiner_final.pth" 500 1e-5 1.0 500 "exp53_refiner_stageC.log"
train_refiner_stage dropout_balanced "$REF_D" "$REF_C/refiner_final.pth" 500 1e-5 1.4 250 "exp53_refiner_stageD_dropout_balanced.log"
train_refiner_stage decoderaware_e "$REF_E" "$REF_D/refiner_final.pth" 250 5e-6 1.5 125 "exp53_refiner_stageE_decoderaware.log"

echo
echo "[$(date '+%F %T')] Step 10/10: evaluate full-rerun RGB StageE on locked finaltest1000"
if [[ -d "$APPLY_DIR/refined_images" && -f "$APPLY_DIR/report.txt" ]]; then
  apply_pngs="$(count_pngs "$APPLY_DIR/refined_images")"
  apply_msgs="$(count_msgs "$APPLY_DIR/refined_images")"
  if [[ "$apply_pngs" -ne 1000 || "$apply_msgs" -ne 1000 || ! -d "$APPLY_DIR/attacked_clean/Clean" ]]; then
    echo "Incomplete existing apply output."
    exit 1
  fi
  echo "Reuse completed apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL1000_CONFIG" \
    --resume "$BASE65_CKPT" \
    --refiner "$REF_E/refiner_final.pth" \
    --watermarked_dir "$BASE_WM" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$APPLY_DIR" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_apply.log"
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FINAL_GT" \
  --results_path "$APPLY_DIR/refined_images" \
  2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_psnr_ssim.log"

run_lpips "$APPLY_DIR/refined_images" "$EXP_ROOT/lpips/$RGB_VARIANT" "$LOG_DIR/${RGB_VARIANT}_lpips.log"

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE65_CKPT" \
  --attack_dir "$APPLY_DIR/attacked_clean" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_clean_ba.log"

if [[ -d "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" ]]; then
  attack_count="$(find "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete RGB attack output."
    exit 1
  fi
  echo "Reuse completed RGB attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FINAL_GT" \
    --results_path "$APPLY_DIR/refined_images" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$RGB_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_attack.log"
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE65_CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_summary.txt"

"$PY" scripts/summarize_o256_exp53_full_from_scratch_stageE_seed61_20260709.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV" \
  --per_attack_csv "$PER_ATTACK_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
echo "Per-attack csv: $PER_ATTACK_CSV"
