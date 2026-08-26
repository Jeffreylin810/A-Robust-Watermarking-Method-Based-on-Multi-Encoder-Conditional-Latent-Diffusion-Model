#!/usr/bin/env bash
set -euo pipefail

OURS_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"

DIFFW_CONFIG="CoCo_train10000_eval1000_unseen_linear.yml"
DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_exp8_diffw_like_train10000_msg32_steps20k_seed61_final.pth.tar"
RUN_NAME="exp9_diffw_like_train10000_msg32_20k_seed61"
DIFFW_OUTPUT="$DIFFW_ROOT/results/test_result/$RUN_NAME/CoCo/CoCo"
DIFFW_PSNR_LOG="$DIFFW_ROOT/logs/${RUN_NAME}_psnr_ssim.log"
DIFFW_CLEAN_ROOT="$DIFFW_ROOT/results/attacked_Result/clean_${RUN_NAME}"
DIFFW_CLEAN_DIR="$DIFFW_CLEAN_ROOT/Clean"
DIFFW_CLEAN_BA="$DIFFW_CLEAN_ROOT/ba_report.txt"
DIFFW_ATTACK_ROOT="$DIFFW_ROOT/results/attacked_Result"
DIFFW_ATTACK_RUN="nongeom_${RUN_NAME}"
DIFFW_ATTACK_DIR="$DIFFW_ATTACK_ROOT/$DIFFW_ATTACK_RUN"
DIFFW_ATTACK_BA="$DIFFW_ATTACK_DIR/ba_report.txt"

OURS_REPORT="$OURS_ROOT/results/refinement/refine10000_targetedformal_C500_apply_eval1000_seed61/report.json"
OURS_CLEAN_BA="$OURS_ROOT/results/refinement/refine10000_targetedformal_C500_apply_eval1000_seed61/attacked_clean/ba_report.txt"
OURS_ATTACK_BA="$OURS_ROOT/results/attacked_Result/exp6_nongeom_refine10000_targetedformal_C500_seed61/ba_report.txt"
OURS_BASE_PSNR_LOG="$OURS_ROOT/logs/exp2b_nongeom_train10000_base20k_then_light5k_seed61_psnr_ssim.log"
OURS_BASE_CLEAN_BA="$OURS_ROOT/results/attacked_Result/clean_exp2b_train10000_base20k_then_light5k_seed61/ba_report.txt"
OURS_BASE_ATTACK_BA="$OURS_ROOT/results/attacked_Result/exp2b_nongeom_train10000_base20k_then_light5k_seed61/ba_report.txt"

MAINLINE_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments"
SUMMARY="$MAINLINE_ROOT/01_method_comparison/summary.md"

MODE="${1:-run}"
if [[ "$MODE" != "run" && "$MODE" != "--check-only" && "$MODE" != "--summarize-only" ]]; then
  echo "Usage: $0 [--check-only|--summarize-only]"
  exit 2
fi

required=(
  "$DIFFW_ROOT/eval_diffusion.py"
  "$DIFFW_ROOT/calculate_psnr_ssim.py"
  "$DIFFW_ROOT/calculate_ba.py"
  "$DIFFW_ROOT/configs/$DIFFW_CONFIG"
  "$DIFFW_CKPT"
  "$OURS_ROOT/attack_eval_diffw_paper_matched.py"
  "$OURS_ROOT/scripts/audit_mainline_protocol.py"
  "$OURS_ROOT/scripts/summarize_exp9_formal_method_comparison.py"
  "$OURS_REPORT"
  "$OURS_CLEAN_BA"
  "$OURS_ATTACK_BA"
  "$OURS_BASE_PSNR_LOG"
  "$OURS_BASE_CLEAN_BA"
  "$OURS_BASE_ATTACK_BA"
  "$GT"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

cd "$OURS_ROOT"
"$PY" scripts/audit_mainline_protocol.py

if [[ "$MODE" == "--check-only" ]]; then
  echo "Experiment 9 preflight passed."
  echo "DiffW checkpoint: $DIFFW_CKPT"
  echo "O256 report: $OURS_REPORT"
  echo "Output summary: $SUMMARY"
  exit 0
fi

if [[ "$MODE" != "--summarize-only" ]]; then
  if ! "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "Refusing to run Experiment 9 without an available CUDA GPU."
    echo "This prevents the 1000-image formal evaluation from silently falling back to CPU."
    exit 1
  fi

  gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$gpu_processes" ]]; then
    echo "Refusing to start while another GPU compute process is active:"
    echo "$gpu_processes"
    exit 1
  fi

  if [[ ! -d "$DIFFW_OUTPUT" ]]; then
    echo "[$(date '+%F %T')] Step 1/5: DiffW-like unseen eval1000"
    (
      cd "$DIFFW_ROOT"
      /usr/bin/time -p "$PY" -u eval_diffusion.py \
        --config "$DIFFW_CONFIG" \
        --resume "$DIFFW_CKPT" \
        --image_folder results/test_result \
        --run_name "$RUN_NAME" \
        --test_set CoCo \
        --seed 61 \
        2>&1 | tee "logs/${RUN_NAME}_eval.log"
    )
  else
    echo "[$(date '+%F %T')] Step 1/5: reuse existing DiffW-like eval output"
  fi

  echo "[$(date '+%F %T')] Step 2/5: DiffW-like PSNR / SSIM"
  (
    cd "$DIFFW_ROOT"
    "$PY" calculate_psnr_ssim.py \
      --gt_path "$GT" \
      --results_path "$DIFFW_OUTPUT" \
      2>&1 | tee "$DIFFW_PSNR_LOG"
  )

  if [[ ! -f "$DIFFW_CLEAN_BA" ]]; then
    echo "[$(date '+%F %T')] Step 3/5: DiffW-like Clean BA"
    mkdir -p "$DIFFW_CLEAN_DIR"
    find "$DIFFW_OUTPUT" -maxdepth 1 -type f \
      \( -name '*.png' -o -name '*.jpg' -o -name '*.webp' \) -print0 |
      while IFS= read -r -d '' image; do
        target="$DIFFW_CLEAN_DIR/$(basename "$image")"
        [[ -e "$target" ]] || ln "$image" "$target"
      done
    (
      cd "$DIFFW_ROOT"
      "$PY" calculate_ba.py \
        --config "$DIFFW_CONFIG" \
        --resume "$DIFFW_CKPT" \
        --attack_dir "$DIFFW_CLEAN_ROOT" \
        --gt_message_dir "$DIFFW_OUTPUT" \
        2>&1 | tee "logs/${RUN_NAME}_clean_ba.log"
    )
  else
    echo "[$(date '+%F %T')] Step 3/5: reuse existing DiffW-like Clean BA"
  fi

  if [[ ! -f "$DIFFW_ATTACK_BA" ]]; then
    echo "[$(date '+%F %T')] Step 4/5: unified 11 non-geometric attacks and BA"
    if [[ ! -d "$DIFFW_ATTACK_DIR" ]]; then
      (
        cd "$OURS_ROOT"
        "$PY" attack_eval_diffw_paper_matched.py \
          --gt_path "$GT" \
          --results_path "$DIFFW_OUTPUT" \
          --output_dir "$DIFFW_ATTACK_ROOT" \
          --run_name "$DIFFW_ATTACK_RUN" \
          --attack_scope nongeom \
          --seed 61 \
          2>&1 | tee "$DIFFW_ROOT/logs/${RUN_NAME}_nongeom_attack.log"
      )
    fi
    (
      cd "$DIFFW_ROOT"
      "$PY" calculate_ba.py \
        --config "$DIFFW_CONFIG" \
        --resume "$DIFFW_CKPT" \
        --attack_dir "$DIFFW_ATTACK_DIR" \
        --gt_message_dir "$DIFFW_OUTPUT" \
        2>&1 | tee "logs/${RUN_NAME}_nongeom_ba.log"
    )
  else
    echo "[$(date '+%F %T')] Step 4/5: reuse existing DiffW-like attack BA"
  fi
fi

for result in "$DIFFW_PSNR_LOG" "$DIFFW_CLEAN_BA" "$DIFFW_ATTACK_BA"; do
  if [[ ! -f "$result" ]]; then
    echo "Missing result required for summary: $result"
    exit 1
  fi
done

echo "[$(date '+%F %T')] Step 5/5: write unified formal table"
cd "$OURS_ROOT"
"$PY" scripts/summarize_exp9_formal_method_comparison.py \
  --diffw_psnr_log "$DIFFW_PSNR_LOG" \
  --diffw_clean_ba "$DIFFW_CLEAN_BA" \
  --diffw_attack_ba "$DIFFW_ATTACK_BA" \
  --ours_base_psnr_log "$OURS_BASE_PSNR_LOG" \
  --ours_base_clean_ba "$OURS_BASE_CLEAN_BA" \
  --ours_base_attack_ba "$OURS_BASE_ATTACK_BA" \
  --ours_report_json "$OURS_REPORT" \
  --ours_clean_ba "$OURS_CLEAN_BA" \
  --ours_attack_ba "$OURS_ATTACK_BA" \
  --output "$SUMMARY"

echo "[$(date '+%F %T')] Experiment 9 complete"
echo "Summary: $SUMMARY"
