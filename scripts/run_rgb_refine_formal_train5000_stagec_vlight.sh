#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

cd "$PROJECT_ROOT"
mkdir -p logs

RUN_NAME="O256_RGBREFINE_FORMAL_TRAIN5000_STAGEC_VLIGHT"
BASE_CKPT="Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"
DATA_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen"
EVAL_GT="$DATA_ROOT/data/CoCo/val/gt"
EVAL_WM_DIR="results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo"
TRAIN_ALIAS="results/formal_refine_datasets/CoCo256_train5000_as_val_seed61"
TRAIN_WM_DIR="results/test_result/O256_FORMAL_TRAIN5000_BASE_seed61/train5000/CoCo"
STAGE_B_DIR="results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_THEN_MSG0P75_1K_seed61"
STAGE_C_DIR="results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_MSG0P75_AWARE0P05_250_seed61"
APPLY_DIR="results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61"
PAPER_RUN="paper_matched_O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61"

echo "[$(date '+%F %T')] Start $RUN_NAME"

echo
echo "[$(date '+%F %T')] Step 1/5: train very-light attack-aware Stage C from train5000 Stage B"
/usr/bin/time -p "$PY" -u scripts/train_rgb_refinement.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE_CKPT" \
  --resume_refiner "$STAGE_B_DIR/refiner_final.pth" \
  --watermarked_dir "$TRAIN_WM_DIR" \
  --gt_dir "$TRAIN_ALIAS/data/CoCo/val/gt" \
  --output_dir "$STAGE_C_DIR" \
  --image_size 256 \
  --steps 250 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 2e-5 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 15.0 \
  --image_l1_weight 3.75 \
  --message_weight 0.75 \
  --regularize_weight 0.02 \
  --attack_message_weight 0.05 \
  --attack_profile nongeom_light \
  --attack_noise_sigma 0.02 \
  --attack_blur_sigma 0.8 \
  --attack_jpeg_levels 48 \
  --attack_dropout_prob 0.03 \
  --eval_every 125 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_MSG0P75_AWARE0P05_250_seed61_train.log

echo
echo "[$(date '+%F %T')] Step 2/5: apply Stage C refiner to unseen eval1000"
/usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE_CKPT" \
  --refiner "$STAGE_C_DIR/refiner_final.pth" \
  --watermarked_dir "$EVAL_WM_DIR" \
  --gt_dir "$EVAL_GT" \
  --output_dir "$APPLY_DIR" \
  --image_size 256 \
  --batch_size 4 \
  --num_workers 2 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61_apply.log

echo
echo "[$(date '+%F %T')] Step 3/5: evaluate unseen eval1000 PSNR / SSIM"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$EVAL_GT" \
  --results_path "$APPLY_DIR/refined_images" \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61_psnr_ssim.log

echo
echo "[$(date '+%F %T')] Step 4/5: evaluate unseen eval1000 clean BA"
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE_CKPT" \
  --attack_dir "$APPLY_DIR/attacked_clean" \
  --gt_message_dir "$EVAL_WM_DIR" \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61_clean_ba.log

PSNR="$("$PY" - <<'PY'
import re
from pathlib import Path
text = Path("logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61_psnr_ssim.log").read_text(errors="ignore")
matches = re.findall(r"Testing set, PSNR is ([0-9.]+)", text)
print(matches[-1] if matches else "nan")
PY
)"
CLEAN_BA="$("$PY" - <<'PY'
import re
from pathlib import Path
text = Path("logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61_clean_ba.log").read_text(errors="ignore")
matches = re.findall(r"Clean\s+\|\s+([0-9.]+)", text)
if not matches:
    matches = re.findall(r"Average Bit Accuracy \(BA\):\s*([0-9.]+)%", text)
print(matches[-1] if matches else "nan")
PY
)"

echo
echo "[$(date '+%F %T')] Gate check"
echo "PSNR: $PSNR"
echo "Clean BA: $CLEAN_BA"

RUN_PAPER="$("$PY" - <<PY
psnr = float("$PSNR") if "$PSNR" != "nan" else float("nan")
ba = float("$CLEAN_BA") if "$CLEAN_BA" != "nan" else float("nan")
print("yes" if psnr >= 30.0 and ba >= 89.0 else "no")
PY
)"

if [[ "$RUN_PAPER" == "yes" ]]; then
  echo
  echo "[$(date '+%F %T')] Step 5/5: gate passed, run paper-matched attack evaluation"
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$EVAL_GT" \
    --results_path "$APPLY_DIR/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$PAPER_RUN" \
    2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61_paper_matched_attack.log

  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE_CKPT" \
    --attack_dir "results/attacked_Result/$PAPER_RUN" \
    --gt_message_dir "$EVAL_WM_DIR" \
    2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_MSG0P75_AWARE0P05_250_seed61_paper_matched_ba.log
else
  echo
  echo "[$(date '+%F %T')] Step 5/5: gate failed, skip paper-matched attacks"
fi

echo
echo "[$(date '+%F %T')] Finished $RUN_NAME"
echo "Apply output: $APPLY_DIR"
