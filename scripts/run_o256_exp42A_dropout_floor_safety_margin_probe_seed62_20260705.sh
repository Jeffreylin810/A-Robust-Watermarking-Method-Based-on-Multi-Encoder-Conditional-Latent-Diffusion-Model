#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=62

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"
TRAIN_ALIAS="results/formal_refine_datasets/CoCo256_train10000_as_val_seed62_exp42A_formal"
TRAIN_GT="$TRAIN_ALIAS/data/CoCo/val/gt"

EVAL_TEMPLATE="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed62_exp41_20260705.yml"
EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_exp42A_dropout_floor_probe_seed62_20260705.yml"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
TRAIN_PAIR_TEMPLATE="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed62_exp41_20260705.yml"
TRAIN_PAIR_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_exp42A_dropout_floor_probe_seed62_20260705.yml"
TRAIN_TEMPLATE="CoCo_train10000_ours256_msg32_exp41_base65_dropout_margin_plus5k_seed62_20260705.yml"
TRAIN_CONFIG="CoCo_train10000_ours256_msg32_exp42A_base70_dropout_margin_plus10k_seed62_20260705.yml"

EXP41_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
EXP41_BASE65_CKPT="$EXP41_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/42_supplementary_o256_exp41_dropout_floor_safety_margin_probe_seed62_20260705"
LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"
PER_ATTACK_CSV="$EXP_ROOT/per_attack_ba.csv"

WORKSPACE="$EXP_ROOT/workspace/base70_dropout_margin_plus10k"
CKPT="$WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"

BASE_VARIANT="exp42A_base70_dropout_margin_plus10k"
BASE_OUTPUT="$EXP_ROOT/eval_outputs/$BASE_VARIANT"
BASE_WM="$BASE_OUTPUT/finaltest1000/CoCo"
BASE_ATTACK_RUN="${BASE_VARIANT}_nongeom"

PAIR_OUTPUT_ROOT="$EXP_ROOT/train_pairs/exp42A_base70_pairs"
TRAIN_WM="$PAIR_OUTPUT_ROOT/train10000/CoCo"

REF_A="$EXP_ROOT/refinement/exp42A_stageA_image10k"
REF_B="$EXP_ROOT/refinement/exp42A_stageB_message2k"
REF_BP="$EXP_ROOT/refinement/exp42A_stageBplus_message500"
REF_C="$EXP_ROOT/refinement/exp42A_stageC_targeted500"
REF_D="$EXP_ROOT/refinement/exp42A_stageD_dropout_balanced500"

RGB_VARIANT="exp42A_rgb_base70_dropout_balanced"
APPLY_DIR="$EXP_ROOT/apply/${RGB_VARIANT}_finaltest1000"
RGB_ATTACK_RUN="${RGB_VARIANT}_nongeom"

MIN_FREE_GB=20
EST_REQUIRED_GB=14

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

generate_configs() {
  "$PY" - <<PY
from pathlib import Path
import yaml

root = Path("$ROOT/configs")

train_cfg = yaml.safe_load((root / "$TRAIN_TEMPLATE").read_text(encoding="utf-8"))
train_cfg["data"]["dataset_root"] = "$FORMAL_DATASET_ROOT"
train_cfg["data"]["output_dir"] = "$WORKSPACE"
train_cfg["training"]["n_epochs"] = 14
train_cfg["watermarking"]["attack_profile"] = "identity_nongeom_dropout_margin"
(root / "$TRAIN_CONFIG").write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")

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

  local command=(
    "$PY" -u scripts/train_rgb_refinement.py
    --config "$EVAL1000_CONFIG"
    --resume "$CKPT"
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
    --image_mse_weight 10.0
    --image_l1_weight 2.5
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
      --resume "$CKPT"
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
  fi

  /usr/bin/time -p "${command[@]}" 2>&1 | tee "$LOG_DIR/$log_name"
}

require_path "$PY"
require_path "$FORMAL_DATASET_ROOT"
require_path "$FINAL_GT"
require_path "$EXP41_BASE65_CKPT"
require_path "configs/$TRAIN_TEMPLATE"
require_path "configs/$EVAL_TEMPLATE"
require_path "configs/$EVAL1000_CONFIG"
require_path "configs/$TRAIN_PAIR_TEMPLATE"
ensure_train_alias
require_path "$TRAIN_GT"
generate_configs
require_path "configs/$TRAIN_CONFIG"
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
  echo "O256 exp42A dropout-floor safety margin probe preflight passed."
  echo "Resume base ckpt: $EXP41_BASE65_CKPT"
  echo "Generated configs:"
  echo "  configs/$TRAIN_CONFIG"
  echo "  configs/$EVAL_CONFIG"
  echo "  configs/$TRAIN_PAIR_CONFIG"
  echo "Will run sequentially:"
  echo "  (1) one extra base epoch from exp41 base65 to base70 using identity_nongeom_dropout_margin"
  echo "  (2) base70 finaltest1000 metrics + full nongeom BA"
  echo "  (3) regenerate train10000 pairs from base70"
  echo "  (4) train fixed A/B/B+/C/D RGB refiner from scratch"
  echo "  (5) RGB finaltest1000 metrics + full nongeom BA"
  echo "  (6) summarize"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Minimum recommended free disk: ${MIN_FREE_GB}G"
  echo "Disk check: OK"
  exit 0
fi

echo "[$(date '+%F %T')] Step 1/6: add one dropout-margin epoch from exp41 base65"
if [[ -f "$CKPT" ]]; then
  echo "Reuse trained checkpoint: $CKPT"
elif [[ -d "$WORKSPACE" ]]; then
  echo "Existing incomplete workspace detected: $WORKSPACE"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$TRAIN_CONFIG" \
    --resume "$EXP41_BASE65_CKPT" \
    --image_folder "results/validation/exp42A_base70_dropout_margin_plus10k" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp42A_base70_dropout_margin_plus10k_train.log"
fi
[[ -f "$CKPT" ]] || { echo "Missing expected checkpoint: $CKPT"; exit 1; }

echo
echo "[$(date '+%F %T')] Step 2/6: evaluate base70 on locked finaltest1000"
if [[ -d "$BASE_WM" ]]; then
  png_count="$(count_pngs "$BASE_WM")"
  msg_count="$(count_msgs "$BASE_WM")"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete base70 eval output: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse completed base70 eval outputs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$CKPT" \
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
  --resume "$CKPT" \
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
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${BASE_VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 3/6: regenerate train10000 pairs from base70"
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
    --resume "$CKPT" \
    --image_folder "$PAIR_OUTPUT_ROOT" \
    --run_name train10000 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/exp42A_train_pairs_eval.log"
fi

echo
echo "[$(date '+%F %T')] Step 4/6: train fixed A/B/B+/C/D RGB refiner from scratch"
train_refiner_stage image "$REF_A" "" 10000 1e-4 0.2 1000 "exp42A_refiner_stageA.log"
train_refiner_stage message "$REF_B" "$REF_A/refiner_final.pth" 2000 5e-5 0.75 500 "exp42A_refiner_stageB.log"
train_refiner_stage message "$REF_BP" "$REF_B/refiner_final.pth" 500 2e-5 1.0 500 "exp42A_refiner_stageBplus.log"
train_refiner_stage targeted_c "$REF_C" "$REF_BP/refiner_final.pth" 500 1e-5 1.0 500 "exp42A_refiner_stageC.log"
train_refiner_stage dropout_balanced "$REF_D" "$REF_C/refiner_final.pth" 500 1e-5 1.4 250 "exp42A_refiner_stageD_dropout_balanced.log"

echo
echo "[$(date '+%F %T')] Step 5/6: evaluate RGB result"
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
    --resume "$CKPT" \
    --refiner "$REF_D/refiner_final.pth" \
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
  --resume "$CKPT" \
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
  --resume "$CKPT" \
  --attack_dir "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN" \
  --gt_message_dir "$BASE_WM" \
  2>&1 | tee "$LOG_DIR/${RGB_VARIANT}_attack_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_report.txt" \
  --output "$EXP_ROOT/attacked_Result/$RGB_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 6/6: summarize"
"$PY" scripts/summarize_o256_exp42A_dropout_floor_safety_margin_probe_20260705.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV" \
  --per_attack_csv "$PER_ATTACK_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
echo "Per-attack csv: $PER_ATTACK_CSV"
