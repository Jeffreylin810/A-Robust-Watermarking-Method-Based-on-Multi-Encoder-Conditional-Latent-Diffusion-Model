#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
SEED=61

BASE20_TEMPLATE="configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k.yml"
BASE30_TEMPLATE="configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k.yml"
BASE20_CONFIG="configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed61_formal.yml"
BASE30_CONFIG="configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml"
EVAL_CONFIG="CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"

FORMAL_DATASET_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FORMAL_TRAIN_GT="$FORMAL_DATASET_ROOT/data/CoCo/train/gt"
DEV_GT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt"

BASE20_OUTPUT="results/formal_training/m01_formal_seed61_identity20k"
BASE20_SOURCE="$BASE20_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
BASE20_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_final.pth.tar"

BASE30_OUTPUT="results/formal_training/m01_formal_seed61_identity20k_then_nongeomlight10k"
BASE30_SOURCE="$BASE30_OUTPUT/ckpts/CoCo_ddpm_final.pth.tar"
BASE30_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"

DEV_EVAL_OUTPUT="results/test_result/m01_formal_seed61_base30k_dev1000"
DEV_EVAL_WM="$DEV_EVAL_OUTPUT/eval1000/CoCo"
DEV_CLEAN_RUN="clean_m01_formal_seed61_base30k_dev1000"
DEV_ATTACK_RUN="m01_formal_seed61_base30k_dev1000_nongeom"

SUMMARY_DIR="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_base"
SUMMARY_MD="$SUMMARY_DIR/summary.md"

cd "$ROOT"
mkdir -p logs Result

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in \
  "$PY" \
  "$BASE20_TEMPLATE" \
  "$BASE30_TEMPLATE" \
  "$BASE20_CONFIG" \
  "$BASE30_CONFIG" \
  "configs/$EVAL_CONFIG" \
  "$FORMAL_DATASET_ROOT" \
  "$FORMAL_TRAIN_GT" \
  "$DEV_GT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

"$PY" scripts/verify_config_equivalence.py \
  --reference "$BASE20_TEMPLATE" \
  --candidate "$BASE20_CONFIG" \
  --ignore data.output_dir \
  --ignore data.dataset_root

"$PY" scripts/verify_config_equivalence.py \
  --reference "$BASE30_TEMPLATE" \
  --candidate "$BASE30_CONFIG" \
  --ignore data.output_dir \
  --ignore data.dataset_root

train_count="$(find -L "$FORMAL_TRAIN_GT" -maxdepth 1 -type f | wc -l)"
if [[ "$train_count" -ne 10000 ]]; then
  echo "Expected formal train10000, found: $train_count"
  exit 1
fi

finaltest_count="$(find -L "$FORMAL_DATASET_ROOT/data/CoCo/val/gt" -maxdepth 1 -type f | wc -l)"
if [[ "$finaltest_count" -ne 1000 ]]; then
  echo "Expected formal finaltest1000, found: $finaltest_count"
  exit 1
fi

if [[ "$CHECK_ONLY" == true ]]; then
  echo "M01 formal-clean seed61 base preflight passed."
  echo "Classification: F (formal chapter-4 artifact generation)"
  echo "This script only builds the formal seed61 O256 base and runs dev1000 health checks."
  echo "It does NOT evaluate finaltest1000 and does NOT retrain the RGB refinement module."
  echo "Locked base recipe: identity 20k -> nongeom-light 10k"
  echo "Formal train root: $FORMAL_DATASET_ROOT"
  echo "Free disk:"
  df -h "$ROOT" | tail -n 1
  exit 0
fi

echo "[$(date '+%F %T')] Start M01 formal-clean seed61 base"
echo "Classification: F (formal artifact generation; no finaltest1000 evaluation in this step)"

echo
echo "[$(date '+%F %T')] Step 1/6: train formal seed61 identity20k from scratch"
if [[ -f "$BASE20_CKPT" ]]; then
  echo "Reuse completed formal identity20k checkpoint."
elif [[ -e "$BASE20_OUTPUT" ]]; then
  echo "Incomplete or unarchived identity20k output exists: $BASE20_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$(basename "$BASE20_CONFIG")" \
    --resume ./ckpts/nonexistent_m01_formal_seed61_identity20k_start.pth.tar \
    --image_folder results/validation/m01_formal_seed61_identity20k \
    --seed "$SEED" \
    2>&1 | tee logs/m01_formal_seed61_identity20k_train.log

  if [[ ! -f "$BASE20_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $BASE20_SOURCE"
    exit 1
  fi
  mv "$BASE20_SOURCE" "$BASE20_CKPT"
fi

"$PY" scripts/check_training_message_convergence.py \
  --log logs/m01_formal_seed61_identity20k_train.log \
  --tail 20 \
  --max_mean 0.10 \
  --max_last 0.15

echo
echo "[$(date '+%F %T')] Step 2/6: continue formal seed61 base to identity20k -> nongeom-light10k"
if [[ -f "$BASE30_CKPT" ]]; then
  echo "Reuse completed formal base30k checkpoint."
elif [[ -e "$BASE30_OUTPUT" ]]; then
  echo "Incomplete or unarchived base30k output exists: $BASE30_OUTPUT"
  echo "Inspect it before retrying; this script will not overwrite it."
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$(basename "$BASE30_CONFIG")" \
    --resume "$BASE20_CKPT" \
    --image_folder results/validation/m01_formal_seed61_base30k \
    --seed "$SEED" \
    2>&1 | tee logs/m01_formal_seed61_base30k_train.log

  if [[ ! -f "$BASE30_SOURCE" ]]; then
    echo "Training finished without expected checkpoint: $BASE30_SOURCE"
    exit 1
  fi
  mv "$BASE30_SOURCE" "$BASE30_CKPT"
fi

echo
echo "[$(date '+%F %T')] Step 3/6: dev1000 health-check evaluation (allowed; no recipe changes)"
if [[ -e "$DEV_EVAL_OUTPUT" ]]; then
  eval_png_count="$(find "$DEV_EVAL_WM" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  eval_msg_count="$(find "$DEV_EVAL_WM" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ "$eval_png_count" -ne 1000 || "$eval_msg_count" -ne 1000 ]]; then
    echo "Incomplete existing dev1000 eval output: images=$eval_png_count messages=$eval_msg_count"
    exit 1
  fi
  echo "Reuse completed formal base dev1000 eval output."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$BASE30_CKPT" \
    --image_folder "$DEV_EVAL_OUTPUT" \
    --run_name eval1000 \
    --seed "$SEED" \
    2>&1 | tee logs/m01_formal_seed61_base30k_dev1000_eval.log
fi

echo
echo "[$(date '+%F %T')] Step 4/6: dev1000 health-check metrics"
"$PY" calculate_psnr_ssim.py \
  --gt_path "$DEV_GT" \
  --results_path "$DEV_EVAL_WM" \
  2>&1 | tee logs/m01_formal_seed61_base30k_dev1000_psnr_ssim.log

if [[ ! -e "results/attacked_Result/$DEV_CLEAN_RUN" ]]; then
  mkdir -p "results/attacked_Result/$DEV_CLEAN_RUN"
  ln -s "$(realpath "$DEV_EVAL_WM")" "results/attacked_Result/$DEV_CLEAN_RUN/Clean"
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$DEV_CLEAN_RUN" \
  --gt_message_dir "$DEV_EVAL_WM" \
  2>&1 | tee logs/m01_formal_seed61_base30k_dev1000_clean_ba.log

echo
echo "[$(date '+%F %T')] Step 5/6: dev1000 nongeom health-check"
if [[ -e "results/attacked_Result/$DEV_ATTACK_RUN" ]]; then
  attack_count="$(find "results/attacked_Result/$DEV_ATTACK_RUN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete existing dev1000 attack output: directories=$attack_count"
    exit 1
  fi
  echo "Reuse completed formal base dev1000 nongeom attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$DEV_GT" \
    --results_path "$DEV_EVAL_WM" \
    --output_dir results/attacked_Result \
    --run_name "$DEV_ATTACK_RUN" \
    --attack_scope nongeom \
    --seed "$SEED" \
    2>&1 | tee logs/m01_formal_seed61_base30k_dev1000_attack.log
fi

"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$DEV_ATTACK_RUN" \
  --gt_message_dir "$DEV_EVAL_WM" \
  2>&1 | tee logs/m01_formal_seed61_base30k_dev1000_ba.log

"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$DEV_ATTACK_RUN/ba_report.txt" \
  --output "results/attacked_Result/$DEV_ATTACK_RUN/ba_summary.txt"

echo
echo "[$(date '+%F %T')] Step 6/6: write formal base summary"
mkdir -p "$SUMMARY_DIR"
"$PY" - <<'PY'
from pathlib import Path
import json
import re

root = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER")
summary_md = Path("/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_base/summary.md")

psnr_text = (root / "logs/m01_formal_seed61_base30k_dev1000_psnr_ssim.log").read_text(encoding="utf-8", errors="ignore")
clean_text = (root / "results/attacked_Result/clean_m01_formal_seed61_base30k_dev1000/ba_report.txt").read_text(encoding="utf-8", errors="ignore")
attack = json.loads((root / "results/attacked_Result/m01_formal_seed61_base30k_dev1000_nongeom/ba_summary.json").read_text(encoding="utf-8"))

psnr = float(re.findall(r"Testing set,\s*PSNR is\s*([0-9.]+)", psnr_text)[-1])
ssim = float(re.findall(r"SSIM is\s*([0-9.]+)", psnr_text)[-1])
clean_patterns = (
    r"\[Clean\]\s+Average Bit Accuracy \(BA\):\s*([0-9.]+)%",
    r"^\s*Clean\s*\|\s*([0-9.]+)\s*$",
)
clean = None
for pattern in clean_patterns:
    matches = re.findall(pattern, clean_text, re.MULTILINE)
    if matches:
        clean = float(matches[-1])
        break
if clean is None:
    raise RuntimeError("Could not parse Clean BA")

text = f"""# M01 Formal-Clean Seed61 Base Summary

Classification: F (formal artifact generation; dev1000 health-check only)

This step builds the formal seed61 O256 base from scratch using the locked recipe:

```text
identity 20k -> nongeom-light 10k
```

No finaltest1000 evaluation is performed in this step.

## Outputs

- Formal identity20k checkpoint:
  `Result/CoCo_ddpm_m01_formal_seed61_identity20k_final.pth.tar`
- Formal base30k checkpoint:
  `Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar`

## dev1000 health-check

| PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---:|---:|---:|---:|---:|---|
| {psnr:.4f} | {ssim:.4f} | {clean:.2f}% | {float(attack['average_ba']):.2f}% | {float(attack['minimum_ba']):.2f}% | {attack['minimum_attack']} |

## Interpretation

If these dev1000 health-check values are normal, the next formal step is:

```text
use this formal base to generate formal refine10000 pairs
and retrain the formal RGB refinement module from random initialization
```
"""
summary_md.write_text(text, encoding="utf-8")
print(text)
PY

echo
echo "[$(date '+%F %T')] M01 formal-clean seed61 base complete"
echo "Summary: $SUMMARY_MD"
