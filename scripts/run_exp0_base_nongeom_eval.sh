#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
WM="results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo"
RUN="exp0_nongeom_train10000_base20k_seed61"

cd "$ROOT"
mkdir -p logs

if [[ -d "results/attacked_Result/$RUN" ]]; then
  echo "Refusing to overwrite results/attacked_Result/$RUN"
  echo "Remove that directory manually only if this run should be repeated."
  exit 1
fi

"$PY" attack_eval_diffw_paper_matched.py \
  --gt_path "$GT" \
  --results_path "$WM" \
  --output_dir results/attacked_Result \
  --run_name "$RUN" \
  --attack_scope nongeom \
  --seed 61 \
  2>&1 | tee "logs/${RUN}_attack.log"

"$PY" calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir "results/attacked_Result/$RUN" \
  --gt_message_dir "$WM" \
  2>&1 | tee "logs/${RUN}_ba.log"

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$RUN/ba_report.txt" \
  --output "results/attacked_Result/$RUN/ba_summary.txt"
