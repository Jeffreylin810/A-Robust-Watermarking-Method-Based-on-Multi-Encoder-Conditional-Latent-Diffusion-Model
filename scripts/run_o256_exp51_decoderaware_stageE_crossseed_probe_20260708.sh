#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_DATASET_ROOT/data/CoCo/val/gt"
EVAL1000_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"

EXP40_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/40_formal_o256_full_from_scratch_dropout_margin_seed61_20260704"
EXP41_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
BASE_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/51_supplementary_o256_decoderaware_stageE_crossseed_probe_20260708"

MIN_FREE_GB=12
EST_REQUIRED_GB=7

CHECK_ONLY=false
MAX_IMAGES=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ "${1:-}" == "--quick" ]]; then
  MAX_IMAGES="${2:-100}"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only] [--quick N]"
  exit 2
fi

EXP_ROOT="$BASE_EXP_ROOT"
if [[ "$MAX_IMAGES" -gt 0 ]]; then
  EXP_ROOT="${BASE_EXP_ROOT}_quick${MAX_IMAGES}"
fi

LOG_DIR="$EXP_ROOT/logs"
SUMMARY_MD="$EXP_ROOT/summary.md"
SUMMARY_CSV="$EXP_ROOT/summary.csv"
PER_ATTACK_CSV="$EXP_ROOT/per_attack_ba.csv"

cd "$ROOT"
mkdir -p "$EXP_ROOT" "$LOG_DIR" "$EXP_ROOT/refinement" "$EXP_ROOT/apply" "$EXP_ROOT/attacked_Result" "$EXP_ROOT/lpips"

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

count_pngs() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f -iname '*.png' | wc -l
}

count_images() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
}

count_msgs() {
  local dir="$1"
  find -L "$dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l
}

run_lpips() {
  local variant="$1"
  local results_dir="$2"
  local output_dir="$EXP_ROOT/lpips/$variant"
  local log_path="$LOG_DIR/${variant}_lpips.log"
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

require_path "$PY"
require_path "scripts/train_rgb_refinement.py"
require_path "scripts/apply_rgb_refinement.py"
require_path "scripts/calculate_lpips_dir.py"
require_path "scripts/summarize_ba_report.py"
require_path "scripts/summarize_o256_exp51_decoderaware_stageE_crossseed_probe_20260708.py"
require_path "attack_eval_diffw_paper_matched.py"
require_path "calculate_psnr_ssim.py"
require_path "calculate_ba.py"
require_path "configs/$EVAL1000_CONFIG"
require_path "$FINAL_GT"

free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
free_gb="$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
if awk "BEGIN {exit !($free_gb < $MIN_FREE_GB)}"; then
  echo "Insufficient free space: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  exit 1
fi

check_seed_inputs() {
  local seed_label="$1"
  local eval_config="$2"
  local base_ckpt="$3"
  local train_wm="$4"
  local train_gt="$5"
  local base_wm="$6"
  local stageD_refiner="$7"

  require_path "configs/$eval_config"
  require_path "$base_ckpt"
  require_path "$train_wm"
  require_path "$train_gt"
  require_path "$base_wm"
  require_path "$stageD_refiner"

  local train_pngs train_msgs train_gt_images base_pngs base_msgs final_gt_images
  train_pngs="$(count_pngs "$train_wm")"
  train_msgs="$(count_msgs "$train_wm")"
  train_gt_images="$(count_images "$train_gt")"
  base_pngs="$(count_pngs "$base_wm")"
  base_msgs="$(count_msgs "$base_wm")"
  final_gt_images="$(count_images "$FINAL_GT")"
  if [[ "$train_pngs" -lt 1 || "$train_msgs" -lt 1 || "$train_gt_images" -lt "$train_pngs" || "$base_pngs" -ne 1000 || "$base_msgs" -ne 1000 || "$final_gt_images" -ne 1000 ]]; then
    echo "Invalid $seed_label input counts: train_pngs=$train_pngs train_msgs=$train_msgs train_gt_images=$train_gt_images base_pngs=$base_pngs base_msgs=$base_msgs final_gt_images=$final_gt_images"
    exit 1
  fi
  echo "$seed_label inputs: train_images=$train_pngs train_messages=$train_msgs finaltest_images=$base_pngs finaltest_messages=$base_msgs"
}

check_seed_inputs \
  "seed61" \
  "CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml" \
  "$EXP40_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar" \
  "$EXP40_ROOT/train_pairs/exp40_base65_pairs/train10000/CoCo" \
  "results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt" \
  "$EXP40_ROOT/eval_outputs/exp40_base65_dropout_margin/finaltest1000/CoCo" \
  "$EXP40_ROOT/refinement/exp40_stageD_dropout_balanced500/refiner_final.pth"

check_seed_inputs \
  "seed62" \
  "CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed62_exp41_20260705.yml" \
  "$EXP41_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar" \
  "$EXP41_ROOT/train_pairs/exp41_base65_pairs/train10000/CoCo" \
  "results/formal_refine_datasets/CoCo256_train10000_as_val_seed62_exp41_formal/data/CoCo/val/gt" \
  "$EXP41_ROOT/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo" \
  "$EXP41_ROOT/refinement/exp41_stageD_dropout_balanced500/refiner_final.pth"

if [[ "$CHECK_ONLY" == true ]]; then
  echo "O256 exp51 decoder-aware StageE cross-seed probe preflight passed."
  echo "Output root: $EXP_ROOT"
  echo "Estimated extra disk: ~${EST_REQUIRED_GB}G"
  echo "Current free disk: ${free_gb}G"
  echo "Estimated runtime: about 2-4 hours for both seeds"
  echo "Disk check: OK"
  exit 0
fi

run_seed_variant() {
  local seed="$1"
  local variant="$2"
  local eval_config="$3"
  local base_ckpt="$4"
  local train_wm="$5"
  local train_gt="$6"
  local base_wm="$7"
  local stageD_refiner="$8"
  local refiner_dir="$EXP_ROOT/refinement/$variant"
  local apply_dir="$EXP_ROOT/apply/${variant}_finaltest1000"
  local attack_run="${variant}_nongeom"

  echo
  echo "[$(date '+%F %T')] $variant Step 1/5: train decoder-aware StageE"
  local refiner_ready=false
  if [[ -d "$refiner_dir" ]]; then
    if [[ -f "$refiner_dir/refiner_final.pth" && -f "$refiner_dir/report.txt" ]]; then
      echo "Reuse completed StageE refiner: $refiner_dir"
      refiner_ready=true
    else
      echo "Incomplete StageE output, rebuilding: $refiner_dir"
      rm -rf "$refiner_dir"
    fi
  fi
  if [[ "$refiner_ready" != true ]]; then
    /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
      --config "$EVAL1000_CONFIG" \
      --resume "$base_ckpt" \
      --resume_refiner "$stageD_refiner" \
      --watermarked_dir "$train_wm" \
      --gt_dir "$train_gt" \
      --output_dir "$refiner_dir" \
      --image_size 256 \
      --max_images "$MAX_IMAGES" \
      --steps 250 \
      --batch_size 4 \
      --num_workers 2 \
      --lr 5e-6 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --image_mse_weight 8.0 \
      --image_l1_weight 2.0 \
      --message_weight 1.50 \
      --attack_message_weight 0.65 \
      --attack_profile dropout_r03_only \
      --attack_dropout_prob 0.70 \
      --latent_preserve_weight 0.45 \
      --sensitive_residual_weight 3.00 \
      --quality_gate_psnr 33.80 \
      --quality_gate_weight 90.0 \
      --regularize_weight 0.01 \
      --residual_lap_preserve_weight 0.0 \
      --anti_cancel_weight 0.0 \
      --anti_cancel_margin 0.0 \
      --eval_every 125 \
      --log_every 50 \
      --skip_final_dataset_images \
      --seed "$seed" \
      2>&1 | tee "$LOG_DIR/${variant}_train.log"
  fi

  echo
  echo "[$(date '+%F %T')] $variant Step 2/5: apply to locked finaltest1000"
  local apply_ready=false
  if [[ -d "$apply_dir/refined_images" && -f "$apply_dir/report.txt" ]]; then
    local apply_pngs apply_msgs expected_count
    apply_pngs="$(count_pngs "$apply_dir/refined_images")"
    apply_msgs="$(count_msgs "$apply_dir/refined_images")"
    expected_count=1000
    if [[ "$MAX_IMAGES" -gt 0 ]]; then
      expected_count="$MAX_IMAGES"
    fi
    if [[ "$apply_pngs" -eq "$expected_count" && "$apply_msgs" -eq "$expected_count" && -d "$apply_dir/attacked_clean/Clean" ]]; then
      echo "Reuse completed apply output: images=$apply_pngs"
      apply_ready=true
    else
      echo "Incomplete apply output, rebuilding: $apply_dir"
      rm -rf "$apply_dir"
    fi
  fi
  if [[ "$apply_ready" != true ]]; then
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$EVAL1000_CONFIG" \
      --resume "$base_ckpt" \
      --refiner "$refiner_dir/refiner_final.pth" \
      --watermarked_dir "$base_wm" \
      --gt_dir "$FINAL_GT" \
      --output_dir "$apply_dir" \
      --image_size 256 \
      --max_images "$MAX_IMAGES" \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed "$seed" \
      2>&1 | tee "$LOG_DIR/${variant}_apply.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$apply_dir/refined_images" \
    2>&1 | tee "$LOG_DIR/${variant}_psnr_ssim.log"

  run_lpips "$variant" "$apply_dir/refined_images"

  "$PY" calculate_ba.py \
    --config "$eval_config" \
    --resume "$base_ckpt" \
    --attack_dir "$apply_dir/attacked_clean" \
    --gt_message_dir "$base_wm" \
    2>&1 | tee "$LOG_DIR/${variant}_clean_ba.log"

  echo
  echo "[$(date '+%F %T')] $variant Step 3/5: full nongeom attack evaluation"
  local attack_ready=false
  if [[ -d "$EXP_ROOT/attacked_Result/$attack_run" ]]; then
    local attack_count
    attack_count="$(find "$EXP_ROOT/attacked_Result/$attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -eq 11 ]]; then
      echo "Reuse completed attacks: $attack_run"
      attack_ready=true
    else
      echo "Incomplete attack output, rebuilding: $EXP_ROOT/attacked_Result/$attack_run"
      rm -rf "$EXP_ROOT/attacked_Result/$attack_run"
    fi
  fi
  if [[ "$attack_ready" != true ]]; then
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$apply_dir/refined_images" \
      --output_dir "$EXP_ROOT/attacked_Result" \
      --run_name "$attack_run" \
      --attack_scope nongeom \
      --seed "$seed" \
      2>&1 | tee "$LOG_DIR/${variant}_attack.log"
  fi

  echo
  echo "[$(date '+%F %T')] $variant Step 4/5: BA after attacks"
  "$PY" calculate_ba.py \
    --config "$eval_config" \
    --resume "$base_ckpt" \
    --attack_dir "$EXP_ROOT/attacked_Result/$attack_run" \
    --gt_message_dir "$base_wm" \
    2>&1 | tee "$LOG_DIR/${variant}_attack_ba.log"

  "$PY" scripts/summarize_ba_report.py \
    "$EXP_ROOT/attacked_Result/$attack_run/ba_report.txt" \
    --output "$EXP_ROOT/attacked_Result/$attack_run/ba_summary.txt"
}

run_seed_variant \
  "61" \
  "exp51_seed61_stageE_decoderaware_refresh" \
  "CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml" \
  "$EXP40_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar" \
  "$EXP40_ROOT/train_pairs/exp40_base65_pairs/train10000/CoCo" \
  "results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt" \
  "$EXP40_ROOT/eval_outputs/exp40_base65_dropout_margin/finaltest1000/CoCo" \
  "$EXP40_ROOT/refinement/exp40_stageD_dropout_balanced500/refiner_final.pth"

run_seed_variant \
  "62" \
  "exp51_seed62_stageE_decoderaware_refresh" \
  "CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed62_exp41_20260705.yml" \
  "$EXP41_ROOT/workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar" \
  "$EXP41_ROOT/train_pairs/exp41_base65_pairs/train10000/CoCo" \
  "results/formal_refine_datasets/CoCo256_train10000_as_val_seed62_exp41_formal/data/CoCo/val/gt" \
  "$EXP41_ROOT/eval_outputs/exp41_base65_dropout_margin/finaltest1000/CoCo" \
  "$EXP41_ROOT/refinement/exp41_stageD_dropout_balanced500/refiner_final.pth"

echo
echo "[$(date '+%F %T')] Step 5/5: summarize exp51"
"$PY" scripts/summarize_o256_exp51_decoderaware_stageE_crossseed_probe_20260708.py \
  --experiment_root "$EXP_ROOT" \
  --output_md "$SUMMARY_MD" \
  --output_csv "$SUMMARY_CSV" \
  --per_attack_csv "$PER_ATTACK_CSV"

echo
echo "[$(date '+%F %T')] Done."
echo "Summary md: $SUMMARY_MD"
echo "Summary csv: $SUMMARY_CSV"
echo "Per-attack csv: $PER_ATTACK_CSV"
