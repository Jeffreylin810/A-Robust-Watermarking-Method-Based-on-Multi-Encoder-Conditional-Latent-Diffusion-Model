#!/usr/bin/env bash
set -euo pipefail

PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
O256_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/08_supplementary_diffw_quality_tradeoff_20260627"
SOURCE_RESULTS="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/results/test_result/m04_formal_diffw_like_seed61/finaltest1000/CoCo/CoCo"
GT_DIR="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"
FORMAL_EVAL_CONFIG="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
FORMAL_CKPT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"

MODE="${1:-quick}"
shift || true

GAMMAS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.25 0.2 0.15 0.1)
TARGET_PSNR="${TARGET_PSNR:-32.9034}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-2}"

mkdir -p "$EXP_ROOT"/{logs,lpips,attacked_Result,scaled_outputs}

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

gamma_tag() {
  local gamma="$1"
  printf 'gamma_%0.2f' "$gamma" | tr '.' 'p'
}

generate_outputs() {
  log "Generating gamma-scaled DiffW-like outputs"
  (
    cd "$O256_ROOT"
    "$PY" scripts/generate_diffw_quality_sweep_outputs.py \
      --source_results_dir "$SOURCE_RESULTS" \
      --gt_dir "$GT_DIR" \
      --output_root "$EXP_ROOT" \
      --gammas "${GAMMAS[@]}"
  )
}

run_quick_metrics_for_tag() {
  local tag="$1"
  local results_dir="$EXP_ROOT/scaled_outputs/$tag/CoCo/CoCo"
  local psnr_log="$EXP_ROOT/logs/${tag}_psnr_ssim.log"
  local lpips_dir="$EXP_ROOT/lpips/$tag"

  if [[ ! -f "$psnr_log" ]]; then
    log "PSNR/SSIM for $tag"
    "$PY" "$DIFFW_ROOT/calculate_psnr_ssim.py" \
      --gt_path "$GT_DIR" \
      --results_path "$results_dir" \
      2>&1 | tee "$psnr_log"
  else
    log "Skip existing PSNR/SSIM: $tag"
  fi

  if [[ ! -f "$lpips_dir/summary.json" ]]; then
    log "LPIPS for $tag"
    "$PY" "$O256_ROOT/scripts/calculate_lpips_dir.py" \
      --gt_dir "$GT_DIR" \
      --results_dir "$results_dir" \
      --output_dir "$lpips_dir" \
      --batch_size 8 \
      --net alex \
      2>&1 | tee "$EXP_ROOT/logs/${tag}_lpips.log"
  else
    log "Skip existing LPIPS: $tag"
  fi
}

run_quick_all() {
  for gamma in "${GAMMAS[@]}"; do
    run_quick_metrics_for_tag "$(gamma_tag "$gamma")"
  done
}

summarize() {
  log "Summarizing current quality sweep"
  (
    cd "$O256_ROOT"
    "$PY" scripts/summarize_diffw_quality_tradeoff.py \
      --exp_root "$EXP_ROOT" \
      --target_psnr "$TARGET_PSNR"
  )
}

select_nearest_tags() {
  "$PY" - "$EXP_ROOT/diffw_quality_sweep_summary.csv" "$CANDIDATE_COUNT" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
count = int(sys.argv[2])
rows = []
with summary_csv.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        gap = row.get("psnr_gap_to_o256_rgb", "")
        if not gap:
            continue
        rows.append((float(gap), float(row["gamma"]), row))
rows.sort()
for _, gamma, _ in rows[:count]:
    print(f"gamma_{gamma:.2f}".replace(".", "p"))
PY
}

run_full_eval_for_tag() {
  local tag="$1"
  local results_dir="$EXP_ROOT/scaled_outputs/$tag/CoCo/CoCo"
  local clean_root="$EXP_ROOT/attacked_Result/${tag}_clean"
  local clean_dir="$clean_root/Clean"
  local attack_run="${tag}_nongeom"
  local attack_dir="$EXP_ROOT/attacked_Result/$attack_run"

  mkdir -p "$clean_root"
  rm -rf "$clean_dir"
  ln -sfn "$results_dir" "$clean_dir"

  if [[ ! -f "$clean_root/ba_report.txt" ]]; then
    log "Clean BA for $tag"
    "$PY" "$DIFFW_ROOT/calculate_ba.py" \
      --config "$FORMAL_EVAL_CONFIG" \
      --resume "$FORMAL_CKPT" \
      --attack_dir "$clean_root" \
      --gt_message_dir "$results_dir" \
      2>&1 | tee "$EXP_ROOT/logs/${tag}_clean_ba.log"
  else
    log "Skip existing clean BA: $tag"
  fi

  if [[ ! -f "$attack_dir/metrics_report.txt" ]]; then
    log "Generate 11 non-geometric attacks for $tag"
    "$PY" "$O256_ROOT/attack_eval_diffw_paper_matched.py" \
      --gt_path "$GT_DIR" \
      --results_path "$results_dir" \
      --output_dir "$EXP_ROOT/attacked_Result" \
      --run_name "$attack_run" \
      --attack_scope nongeom \
      --seed 61 \
      --overwrite \
      2>&1 | tee "$EXP_ROOT/logs/${tag}_attack.log"
  else
    log "Skip existing attacked images: $tag"
  fi

  if [[ ! -f "$attack_dir/ba_report.txt" ]]; then
    log "Attacked BA for $tag"
    "$PY" "$DIFFW_ROOT/calculate_ba.py" \
      --config "$FORMAL_EVAL_CONFIG" \
      --resume "$FORMAL_CKPT" \
      --attack_dir "$attack_dir" \
      --gt_message_dir "$results_dir" \
      2>&1 | tee "$EXP_ROOT/logs/${tag}_attack_ba.log"
  else
    log "Skip existing attacked BA: $tag"
  fi
}

usage() {
  cat <<EOF
Usage:
  bash scripts/run_diffw_quality_sweep_to_o256rgb_threshold_20260627.sh quick
  bash scripts/run_diffw_quality_sweep_to_o256rgb_threshold_20260627.sh full-auto
  bash scripts/run_diffw_quality_sweep_to_o256rgb_threshold_20260627.sh full gamma_0p30 gamma_0p25
  bash scripts/run_diffw_quality_sweep_to_o256rgb_threshold_20260627.sh all

Modes:
  quick     Generate gamma outputs + run PSNR/SSIM/LPIPS + write summary CSV
  full-auto After quick summary, pick the closest ${CANDIDATE_COUNT} gamma candidates to O256-RGB PSNR and run Clean BA + 11 attacks
  full      Run Clean BA + 11 attacks only for explicitly listed gamma tags
  all       quick + full-auto

Environment overrides:
  TARGET_PSNR=32.9034
  CANDIDATE_COUNT=2
EOF
}

case "$MODE" in
  quick)
    generate_outputs
    run_quick_all
    summarize
    ;;
  full-auto)
    if [[ ! -f "$EXP_ROOT/diffw_quality_sweep_summary.csv" ]]; then
      generate_outputs
      run_quick_all
      summarize
    fi
    mapfile -t tags < <(select_nearest_tags)
    if [[ "${#tags[@]}" -eq 0 ]]; then
      echo "No gamma candidates found in summary CSV."
      exit 1
    fi
    printf '%s\n' "${tags[@]}" | sed 's/^/[candidate] /'
    for tag in "${tags[@]}"; do
      run_full_eval_for_tag "$tag"
    done
    summarize
    ;;
  full)
    if [[ "$#" -eq 0 ]]; then
      echo "Please provide at least one gamma tag, e.g. gamma_0p30 gamma_0p25"
      exit 1
    fi
    for tag in "$@"; do
      run_full_eval_for_tag "$tag"
    done
    summarize
    ;;
  all)
    generate_outputs
    run_quick_all
    summarize
    mapfile -t tags < <(select_nearest_tags)
    if [[ "${#tags[@]}" -eq 0 ]]; then
      echo "No gamma candidates found in summary CSV."
      exit 1
    fi
    printf '%s\n' "${tags[@]}" | sed 's/^/[candidate] /'
    for tag in "${tags[@]}"; do
      run_full_eval_for_tag "$tag"
    done
    summarize
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: $MODE"
    usage
    exit 1
    ;;
esac

log "Done. Experiment root: $EXP_ROOT"
