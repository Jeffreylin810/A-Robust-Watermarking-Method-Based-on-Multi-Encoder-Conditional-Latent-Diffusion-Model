#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/25_supplementary_message_preserving_refinement_seed61_20260701"
LOG_DIR="$EXP_ROOT/logs"

FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$FORMAL_ROOT/data/CoCo/val/gt"

CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
TRAIN_PAIR_CONFIG="CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed61_formal.yml"
BASE_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

BASE_WM="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/end_to_end_outputs/o256_base/eval1000/finaltest1000/CoCo/CoCo"
REFINED_FORMAL="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/end_to_end_outputs/o256_rgb/eval1000/apply/refined_images"
TRAIN_WM="results/test_result/m02_formal_seed61_refine10000_pairs/train10000/CoCo"
TRAIN_GT="results/formal_refine_datasets/CoCo256_train10000_as_val_seed61_formal/data/CoCo/val/gt"

DIAG_DIR="$EXP_ROOT/diagnostics/formal_m03_damage"
LATENT_VARIANT="latent_preserve500_qgate"
SENSITIVE_VARIANT="sensitive_residual500_qgate"
LATENT_DIR="$EXP_ROOT/refinement/$LATENT_VARIANT"
SENSITIVE_DIR="$EXP_ROOT/refinement/$SENSITIVE_VARIANT"
LATENT_APPLY="$EXP_ROOT/apply/$LATENT_VARIANT"
SENSITIVE_APPLY="$EXP_ROOT/apply/$SENSITIVE_VARIANT"

cd "$ROOT"
mkdir -p "$LOG_DIR" "$EXP_ROOT/refinement" "$EXP_ROOT/apply" "$EXP_ROOT/attacked_Result"

count_pngs() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l
}

count_msgs() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l
}

free_gb() {
  df -BG "$ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path"
    exit 1
  fi
}

report_path() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "[OK] $label: $path"
  else
    echo "[MISSING] $label: $path"
  fi
}

require_common_inputs() {
  require_path "$PY"
  require_path "configs/$CONFIG"
  require_path "configs/$TRAIN_PAIR_CONFIG"
  require_path "$BASE_CKPT"
  require_path "$REFINER"
  require_path "$FINAL_GT"
  require_path "$BASE_WM"
  require_path "$REFINED_FORMAL"
  require_path "$TRAIN_WM"
  require_path "$TRAIN_GT"
}

require_diagnose_inputs() {
  require_path "$PY"
  require_path "configs/$CONFIG"
  require_path "$BASE_CKPT"
  require_path "$FINAL_GT"
  require_path "$BASE_WM"
  require_path "$REFINED_FORMAL"
}

require_training_inputs() {
  require_common_inputs
}

check_counts() {
  local final_count
  final_count="$(find -L "$FINAL_GT" -maxdepth 1 -type f | wc -l)"
  local base_pngs base_msgs refined_pngs train_pngs train_msgs
  base_pngs="$(count_pngs "$BASE_WM")"
  base_msgs="$(count_msgs "$BASE_WM")"
  refined_pngs="$(count_pngs "$REFINED_FORMAL")"
  train_pngs="$(count_pngs "$TRAIN_WM")"
  train_msgs="$(count_msgs "$TRAIN_WM")"
  echo "Locked finaltest GT images: $final_count"
  echo "Base finaltest images/messages: $base_pngs / $base_msgs"
  echo "Formal refined finaltest images: $refined_pngs"
  echo "Refiner train pairs images/messages: $train_pngs / $train_msgs"
}

cmd_check() {
  require_path "$PY"
  report_path "eval config" "configs/$CONFIG"
  report_path "train-pair config" "configs/$TRAIN_PAIR_CONFIG"
  report_path "base checkpoint" "$BASE_CKPT"
  report_path "formal refiner checkpoint" "$REFINER"
  report_path "locked finaltest GT" "$FINAL_GT"
  report_path "base finaltest watermarked images" "$BASE_WM"
  report_path "formal refined finaltest images" "$REFINED_FORMAL"
  report_path "refiner train watermarked pairs" "$TRAIN_WM"
  report_path "refiner train GT pairs" "$TRAIN_GT"
  if [[ -e "$FINAL_GT" && -e "$BASE_WM" && -e "$REFINED_FORMAL" ]]; then
    check_counts
  else
    echo "Counts skipped because one or more diagnostic paths are missing."
  fi
  if [[ -e "$REFINER" ]]; then
    echo "Formal refiner checkpoint: found"
  else
    echo "Formal refiner checkpoint: MISSING -> $REFINER"
    echo "Training commands that resume from the formal module will fail until this checkpoint is restored or regenerated."
  fi
  echo
  echo "Experiment root: $EXP_ROOT"
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  echo
  echo "Space guidance:"
  echo "- diagnose: usually < 200 MB if save_top_maps is small."
  echo "- one 500-step train with --skip_final_dataset_images: usually < 200 MB."
  echo "- one finaltest1000 apply + clean BA: roughly 300-600 MB."
  echo "- one full nongeom attack set: can require several GB; do not run with only ~4 GB free."
  echo
  local free
  free="$(free_gb)"
  if [[ "$free" -lt 2 ]]; then
    echo "WARNING: free disk < 2 GB. Do not run even minimal apply before cleanup."
  elif [[ "$free" -lt 6 ]]; then
    echo "WARNING: free disk is limited. Run diagnose / one clean candidate only; avoid full attacks."
  else
    echo "Disk looks acceptable for diagnose + one clean candidate. Full attacks still need caution."
  fi
}

cmd_diagnose() {
  require_diagnose_inputs
  "$PY" -u scripts/analyze_refinement_message_damage_seed61.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --base_dir "$BASE_WM" \
    --refined_dir "$REFINED_FORMAL" \
    --gt_dir "$FINAL_GT" \
    --output_dir "$DIAG_DIR" \
    --image_size 256 \
    --with_sensitivity \
    --save_top_maps 10 \
    --diff_gain 12 \
    2>&1 | tee "$LOG_DIR/diagnose_formal_m03_damage.log"
}

train_candidate() {
  local variant="$1"
  local output_dir="$2"
  shift 2
  if [[ -f "$output_dir/refiner_final.pth" && -f "$output_dir/report.json" ]]; then
    echo "Reuse completed candidate: $output_dir"
    return
  fi
  if [[ -e "$output_dir" && ! -f "$output_dir/refiner_final.pth" ]]; then
    echo "Incomplete candidate output exists: $output_dir"
    echo "Move/delete it before rerun."
    exit 1
  fi
  /usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
    --config "$TRAIN_PAIR_CONFIG" \
    --resume "$BASE_CKPT" \
    --resume_refiner "$REFINER" \
    --watermarked_dir "$TRAIN_WM" \
    --gt_dir "$TRAIN_GT" \
    --output_dir "$output_dir" \
    --image_size 256 \
    --steps 500 \
    --batch_size 4 \
    --num_workers 2 \
    --lr 1e-5 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --image_mse_weight 10.0 \
    --image_l1_weight 2.5 \
    --message_weight 1.0 \
    --attack_message_weight 0.15 \
    --attack_profile targeted_formal_weak \
    --attack_noise_sigma 0.03 \
    --attack_noise_sigma_secondary 0.03 \
    --attack_dropout_prob 0.30 \
    --attack_salt_pepper_prob 0.20 \
    --regularize_weight 0.01 \
    --quality_gate_psnr 32.90 \
    --quality_gate_weight 50.0 \
    --eval_every 100 \
    --log_every 50 \
    --skip_final_dataset_images \
    --seed 61 \
    "$@" \
    2>&1 | tee "$LOG_DIR/${variant}_train.log"
}

cmd_train_latent500() {
  require_training_inputs
  train_candidate "$LATENT_VARIANT" "$LATENT_DIR" \
    --latent_preserve_weight 1.0
}

cmd_train_sensitive500() {
  require_training_inputs
  train_candidate "$SENSITIVE_VARIANT" "$SENSITIVE_DIR" \
    --latent_preserve_weight 0.5 \
    --sensitive_residual_weight 10.0
}

apply_candidate_clean() {
  local variant="$1"
  local refiner_dir="$2"
  local apply_dir="$3"
  require_path "$refiner_dir/refiner_final.pth"
  if [[ -f "$apply_dir/report.json" && -d "$apply_dir/refined_images" ]]; then
    local pngs msgs
    pngs="$(count_pngs "$apply_dir/refined_images")"
    msgs="$(count_msgs "$apply_dir/refined_images")"
    if [[ "$pngs" -eq 1000 && "$msgs" -eq 1000 ]]; then
      echo "Reuse completed apply output: $apply_dir"
    else
      echo "Incomplete apply output: $apply_dir images=$pngs messages=$msgs"
      exit 1
    fi
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$CONFIG" \
      --resume "$BASE_CKPT" \
      --refiner "$refiner_dir/refiner_final.pth" \
      --watermarked_dir "$BASE_WM" \
      --gt_dir "$FINAL_GT" \
      --output_dir "$apply_dir" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed 61 \
      2>&1 | tee "$LOG_DIR/${variant}_apply.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$apply_dir/refined_images" \
    2>&1 | tee "$LOG_DIR/${variant}_psnr_ssim.log"

  "$PY" calculate_ba.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$apply_dir/attacked_clean" \
    --gt_message_dir "$BASE_WM" \
    2>&1 | tee "$LOG_DIR/${variant}_clean_ba.log"
}

cmd_apply_latent500_clean() {
  require_common_inputs
  apply_candidate_clean "$LATENT_VARIANT" "$LATENT_DIR" "$LATENT_APPLY"
}

cmd_apply_sensitive500_clean() {
  require_common_inputs
  apply_candidate_clean "$SENSITIVE_VARIANT" "$SENSITIVE_DIR" "$SENSITIVE_APPLY"
}

full_attack_candidate() {
  local variant="$1"
  local apply_dir="$2"
  require_path "$apply_dir/refined_images"
  local attack_run="${variant}_nongeom"
  if [[ -d "$EXP_ROOT/attacked_Result/$attack_run" ]]; then
    local attack_count
    attack_count="$(find "$EXP_ROOT/attacked_Result/$attack_run" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete full attack output: $EXP_ROOT/attacked_Result/$attack_run"
      exit 1
    fi
    echo "Reuse completed full attacks: $attack_run"
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$apply_dir/refined_images" \
      --output_dir "$EXP_ROOT/attacked_Result" \
      --run_name "$attack_run" \
      --attack_scope nongeom \
      --seed 61 \
      2>&1 | tee "$LOG_DIR/${variant}_nongeom_attack.log"
  fi

  "$PY" calculate_ba.py \
    --config "$CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "$EXP_ROOT/attacked_Result/$attack_run" \
    --gt_message_dir "$BASE_WM" \
    2>&1 | tee "$LOG_DIR/${variant}_nongeom_ba.log"
}

cmd_full_attack_latent500() {
  require_common_inputs
  full_attack_candidate "$LATENT_VARIANT" "$LATENT_APPLY"
}

cmd_full_attack_sensitive500() {
  require_common_inputs
  full_attack_candidate "$SENSITIVE_VARIANT" "$SENSITIVE_APPLY"
}

cmd_summarize() {
  "$PY" scripts/summarize_message_preserving_refinement_seed61.py \
    --exp_root "$EXP_ROOT" \
    --output_csv "$EXP_ROOT/summary.csv" \
    --output_md "$EXP_ROOT/summary.md" \
    "$LATENT_VARIANT" "$SENSITIVE_VARIANT"
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  check                     Preflight paths, counts, and disk guidance.
  diagnose                  Diagnostic only: BA drop, residual, latent drift, sensitivity overlap.

  train_latent500           500-step refresh from formal refiner with latent preservation + quality gate.
  apply_latent500_clean     Apply latent500 to finaltest1000 and compute PSNR + clean BA.
  full_attack_latent500     Optional: run 11 non-geometric attacks for latent500.

  train_sensitive500        500-step refresh with latent + message-sensitive residual penalty.
  apply_sensitive500_clean  Apply sensitive500 to finaltest1000 and compute PSNR + clean BA.
  full_attack_sensitive500  Optional: run 11 non-geometric attacks for sensitive500.

  summarize                 Summarize any completed candidate outputs.

Recommended order:
  1) $0 check
  2) $0 diagnose
  3) $0 train_latent500
  4) $0 apply_latent500_clean
  5) $0 summarize
  6) only if PSNR/clean BA pass, run $0 full_attack_latent500

EOF
}

case "${1:-help}" in
  check) cmd_check ;;
  diagnose) cmd_diagnose ;;
  train_latent500) cmd_train_latent500 ;;
  apply_latent500_clean) cmd_apply_latent500_clean ;;
  full_attack_latent500) cmd_full_attack_latent500 ;;
  train_sensitive500) cmd_train_sensitive500 ;;
  apply_sensitive500_clean) cmd_apply_sensitive500_clean ;;
  full_attack_sensitive500) cmd_full_attack_sensitive500 ;;
  summarize) cmd_summarize ;;
  help|-h|--help) usage ;;
  *) echo "Unknown command: $1"; usage; exit 2 ;;
esac
