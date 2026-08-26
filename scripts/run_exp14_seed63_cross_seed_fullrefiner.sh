#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=63

BASE20_TEMPLATE="configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed62_corrected.yml"
BASE30_TEMPLATE="configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k.yml"
BASE20_CONFIG="configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed63_corrected.yml"
BASE30_CONFIG="configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed63.yml"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"

GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"
BASE20_OUTPUT="results/formal_training/exp14_seed63_identity20k"
BASE20_SOURCE="$BASE20_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
BASE20_CKPT="Result/CoCo_ddpm_exp14_seed63_identity20k_final.pth.tar"

BASE30_OUTPUT="results/formal_training/exp14_seed63_identity20k_then_nongeomlight10k"
BASE30_SOURCE="$BASE30_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
BASE30_CKPT="Result/CoCo_ddpm_exp14_seed63_identity20k_then_nongeomlight10k_final.pth.tar"

EVAL_OUTPUT="results/test_result/exp14_seed63_base30k"
EVAL_WM="$EVAL_OUTPUT/eval1000/CoCo"

SEED61_MODULE="results/refinement/exp13_full_A10000_B2000_BP500_C500/refiner_final.pth"
TRANSFER_APPLY="results/refinement/exp14_seed63_with_seed61_fullmodule_apply_eval1000"

BASE_CLEAN_RUN="clean_exp14_seed63_base30k"
BASE_ATTACK_RUN="exp14_seed63_base30k_nongeom"
TRANSFER_ATTACK_RUN="exp14_seed63_with_seed61_fullmodule_nongeom"
SUMMARY_DIR="results/experiment_summaries/exp14_seed63_cross_seed_fullmodule"
SUMMARY="$SUMMARY_DIR/summary.md"

cd "$ROOT"
mkdir -p logs Result

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$PY" "$BASE20_TEMPLATE" "$BASE30_TEMPLATE" "configs/$EVAL_CONFIG" "$GT" "$SEED61_MODULE"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

echo "[$(date '+%F %T')] Step 0/8: prepare seed63 configs from locked templates"
"$PY" - <<'PY'
from pathlib import Path
import yaml

root = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER")
jobs = [
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed62_corrected.yml",
        "dst": root / "configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed63_corrected.yml",
        "output_dir": "./results/formal_training/exp14_seed63_identity20k",
    },
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k.yml",
        "dst": root / "configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed63.yml",
        "output_dir": "./results/formal_training/exp14_seed63_identity20k_then_nongeomlight10k",
    },
]
for job in jobs:
    data = yaml.safe_load(job["src"].read_text(encoding="utf-8"))
    data["data"]["output_dir"] = job["output_dir"]
    job["dst"].write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

"$PY" scripts/verify_config_equivalence.py \
  --reference "$BASE20_TEMPLATE" \
  --candidate "$BASE20_CONFIG" \
  --ignore data.output_dir

"$PY" scripts/verify_config_equivalence.py \
  --reference "$BASE30_TEMPLATE" \
  --candidate "$BASE30_CONFIG" \
  --ignore data.output_dir

if [[ "$CHECK_ONLY" == true ]]; then
  echo "Experiment 14 preflight passed."
  echo "Classification: E (exploratory; cross-seed validation before formal chapter-4 runs)"
  echo "Base recipe: identity 20k -> nongeom-light 10k"
  echo "Transferred module: $SEED61_MODULE"
  echo "The module will not be retrained in Experiment 14."
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

if [[ -f "$SUMMARY" ]]; then
  echo "Experiment 14 is already complete:"
  echo "$SUMMARY"
  exit 0
fi

echo "[$(date '+%F %T')] Start Experiment 14: seed63 cross-seed Full-module validation"

echo
echo "[$(date '+%F %T')] Step 1/8: train seed63 base from scratch to identity 20k"
if [[ -f "$BASE20_CKPT" ]]; then
  echo "Reuse completed seed63 identity20k checkpoint."
elif [[ -e "$BASE20_OUTPUT" ]]; then
  echo "Incomplete or unarchived identity20k output exists: $BASE20_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$(basename "$BASE20_CONFIG")" \
    --resume ./ckpts/nonexistent_exp14_seed63_identity20k_start.pth.tar \
    --image_folder results/validation/exp14_seed63_identity20k \
    --seed "$SEED" \
    2>&1 | tee logs/exp14_seed63_identity20k_train.log

  if [[ ! -f "$BASE20_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $BASE20_SOURCE"
    exit 1
  fi
  mv "$BASE20_SOURCE" "$BASE20_CKPT"
fi

"$PY" scripts/check_training_message_convergence.py \
  --log logs/exp14_seed63_identity20k_train.log \
  --tail 20 \
  --max_mean 0.10 \
  --max_last 0.15

echo
echo "[$(date '+%F %T')] Step 2/8: continue seed63 base with nongeom-light to 30k"
if [[ -f "$BASE30_CKPT" ]]; then
  echo "Reuse completed seed63 base30k checkpoint."
elif [[ -e "$BASE30_OUTPUT" ]]; then
  echo "Incomplete or unarchived base30k output exists: $BASE30_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$(basename "$BASE30_CONFIG")" \
    --resume "$BASE20_CKPT" \
    --image_folder results/validation/exp14_seed63_base30k \
    --seed "$SEED" \
    2>&1 | tee logs/exp14_seed63_base30k_train.log

  if [[ ! -f "$BASE30_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $BASE30_SOURCE"
    exit 1
  fi
  mv "$BASE30_SOURCE" "$BASE30_CKPT"
fi

echo
echo "[$(date '+%F %T')] Step 3/8: generate fixed dev1000 outputs with seed63 base"
if [[ -e "$EVAL_OUTPUT" ]]; then
  eval_png_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  eval_msg_count="$(find "$EVAL_WM" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ "$eval_png_count" -ne 1000 || "$eval_msg_count" -ne 1000 ]]; then
    echo "Incomplete existing seed63 eval output: images=$eval_png_count messages=$eval_msg_count"
    exit 1
  fi
  echo "Reuse completed seed63 dev1000 eval output."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE30_CKPT" \
    --image_folder "$EVAL_OUTPUT" \
    --run_name eval1000 \
    --seed "$SEED" \
    2>&1 | tee logs/exp14_seed63_base30k_eval.log
fi

echo
echo "[$(date '+%F %T')] Step 4/8: evaluate seed63 base before module transfer"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$GT" \
  --results_path "$EVAL_WM" \
  2>&1 | tee logs/exp14_seed63_base30k_psnr_ssim.log

if [[ ! -e "results/attacked_Result/$BASE_CLEAN_RUN" ]]; then
  mkdir -p "results/attacked_Result/$BASE_CLEAN_RUN"
  ln -s "$(realpath "$EVAL_WM")" "results/attacked_Result/$BASE_CLEAN_RUN/Clean"
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$BASE_CLEAN_RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp14_seed63_base30k_clean_ba.log

"$PY" scripts/check_clean_ba_gate.py \
  --report "results/attacked_Result/$BASE_CLEAN_RUN/ba_report.txt" \
  --minimum 90.0

if [[ ! -e "results/attacked_Result/$BASE_ATTACK_RUN" ]]; then
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$GT" \
    --results_path "$EVAL_WM" \
    --output_dir results/attacked_Result \
    --run_name "$BASE_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee logs/exp14_seed63_base30k_attack.log
else
  base_attack_count="$(find "results/attacked_Result/$BASE_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$base_attack_count" -ne 11 ]]; then
    echo "Incomplete existing seed63 base attack output: directories=$base_attack_count"
    exit 1
  fi
  echo "Reuse completed seed63 base nongeom attacks."
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$BASE_ATTACK_RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp14_seed63_base30k_ba.log
"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$BASE_ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$BASE_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 5/8: apply frozen seed61 Full module to seed63 outputs"
if [[ -e "$TRANSFER_APPLY" ]]; then
  transfer_png_count="$(find "$TRANSFER_APPLY/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  transfer_msg_count="$(find "$TRANSFER_APPLY/refined_images" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ ! -f "$TRANSFER_APPLY/report.json" || "$transfer_png_count" -ne 1000 || "$transfer_msg_count" -ne 1000 || ! -d "$TRANSFER_APPLY/attacked_clean/Clean" ]]; then
    echo "Incomplete existing transfer output: $TRANSFER_APPLY"
    exit 1
  fi
  echo "Reuse completed seed63 transfer apply output."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE30_CKPT" \
    --refiner "$SEED61_MODULE" \
    --watermarked_dir "$EVAL_WM" \
    --gt_dir "$GT" \
    --output_dir "$TRANSFER_APPLY" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed "$SEED" \
    2>&1 | tee logs/exp14_seed63_with_seed61_fullmodule_apply.log
fi

echo
echo "[$(date '+%F %T')] Step 6/8: evaluate transferred module PSNR and Clean BA"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$GT" \
  --results_path "$TRANSFER_APPLY/refined_images" \
  2>&1 | tee logs/exp14_seed63_with_seed61_fullmodule_psnr_ssim.log
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE30_CKPT" \
  --attack_dir "$TRANSFER_APPLY/attacked_clean" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp14_seed63_with_seed61_fullmodule_clean_ba.log

echo
echo "[$(date '+%F %T')] Step 7/8: evaluate transferred module under 11 nongeom attacks"
if [[ ! -e "results/attacked_Result/$TRANSFER_ATTACK_RUN" ]]; then
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$GT" \
    --results_path "$TRANSFER_APPLY/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$TRANSFER_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee logs/exp14_seed63_with_seed61_fullmodule_attack.log
else
  transfer_attack_count="$(find "results/attacked_Result/$TRANSFER_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$transfer_attack_count" -ne 11 ]]; then
    echo "Incomplete existing transferred-module attack output: directories=$transfer_attack_count"
    exit 1
  fi
  echo "Reuse completed transferred-module nongeom attacks."
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$TRANSFER_ATTACK_RUN" \
  --gt_message_dir "$EVAL_WM" \
  2>&1 | tee logs/exp14_seed63_with_seed61_fullmodule_ba.log
"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$TRANSFER_ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$TRANSFER_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 8/8: build cross-seed summary"
mkdir -p "$SUMMARY_DIR"
"$PY" scripts/summarize_exp14_seed63_cross_seed.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo
echo "[$(date '+%F %T')] Experiment 14 complete"
echo "Summary: $SUMMARY"
