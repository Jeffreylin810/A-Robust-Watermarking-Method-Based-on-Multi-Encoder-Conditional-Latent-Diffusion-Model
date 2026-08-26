#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

SOURCE_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FINAL_GT="$SOURCE_ROOT/data/CoCo/val/gt"
BASE20_TEMPLATE="configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed61_formal.yml"
BASE30_TEMPLATE="configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml"
EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

SUBSET_SIZE=5000
FIRST_ALIAS="results/formal_reuse_datasets/CoCo256_train5000_first_formal"
SECOND_ALIAS="results/formal_reuse_datasets/CoCo256_train5000_second_formal"

FIRST_BASE20_CONFIG="configs/CoCo_train5000_first_identity20k_seed61_formal.yml"
FIRST_BASE30_CONFIG="configs/CoCo_train5000_first_identity20k_then_nongeomlight10k_seed61_formal.yml"
SECOND_BASE20_CONFIG="configs/CoCo_train5000_second_identity20k_seed61_formal.yml"
SECOND_BASE30_CONFIG="configs/CoCo_train5000_second_identity20k_then_nongeomlight10k_seed61_formal.yml"

FIRST_BASE20_OUT="results/formal_training/m07_first5000_identity20k"
FIRST_BASE30_OUT="results/formal_training/m07_first5000_base30k"
SECOND_BASE20_OUT="results/formal_training/m07_second5000_identity20k"
SECOND_BASE30_OUT="results/formal_training/m07_second5000_base30k"

FIRST_BASE20_CKPT="Result/CoCo_ddpm_m07_first5000_identity20k_final.pth.tar"
FIRST_BASE30_CKPT="Result/CoCo_ddpm_m07_first5000_base30k_final.pth.tar"
SECOND_BASE20_CKPT="Result/CoCo_ddpm_m07_second5000_identity20k_final.pth.tar"
SECOND_BASE30_CKPT="Result/CoCo_ddpm_m07_second5000_base30k_final.pth.tar"

FIRST_EVAL_OUT="results/test_result/m07_first5000_base30k_finaltest1000"
SECOND_EVAL_OUT="results/test_result/m07_second5000_base30k_finaltest1000"
FIRST_EVAL_WM="$FIRST_EVAL_OUT/finaltest1000/CoCo"
SECOND_EVAL_WM="$SECOND_EVAL_OUT/finaltest1000/CoCo"

FIRST_TRANSFER_OUT="results/refinement/m07_first5000_transfer_apply_finaltest1000"
SECOND_TRANSFER_OUT="results/refinement/m07_second5000_transfer_apply_finaltest1000"

FIRST_BASE_CLEAN="clean_m07_first5000_base30k"
SECOND_BASE_CLEAN="clean_m07_second5000_base30k"
FIRST_BASE_ATTACK="m07_first5000_base30k_nongeom"
SECOND_BASE_ATTACK="m07_second5000_base30k_nongeom"
FIRST_TRANSFER_ATTACK="m07_first5000_transfer_nongeom"
SECOND_TRANSFER_ATTACK="m07_second5000_transfer_nongeom"

SUMMARY_DIR="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/04_cross_seed/module_reuse_same_coco_seed61"
SUMMARY="$SUMMARY_DIR/summary.md"

cd "$ROOT"
mkdir -p logs Result "$SUMMARY_DIR"

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

for required in "$PY" "$SOURCE_ROOT" "$FINAL_GT" "$BASE20_TEMPLATE" "$BASE30_TEMPLATE" "configs/$EVAL_CONFIG" "$REFINER"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

echo "[$(date '+%F %T')] Step 0/9: create two disjoint COCO subset aliases"
"$PY" scripts/create_coco_range_alias.py \
  --source_root "$SOURCE_ROOT" \
  --source_split train \
  --output_root "$FIRST_ALIAS" \
  --start_index 0 \
  --num_images "$SUBSET_SIZE"

"$PY" scripts/create_coco_range_alias.py \
  --source_root "$SOURCE_ROOT" \
  --source_split train \
  --output_root "$SECOND_ALIAS" \
  --start_index "$SUBSET_SIZE" \
  --num_images "$SUBSET_SIZE"

echo "[$(date '+%F %T')] Step 1/9: generate subset-specific configs"
"$PY" - <<'PY'
from pathlib import Path
import yaml

root = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER")
jobs = [
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed61_formal.yml",
        "dst": root / "configs/CoCo_train5000_first_identity20k_seed61_formal.yml",
        "dataset_root": str(root / "results/formal_reuse_datasets/CoCo256_train5000_first_formal"),
        "output_dir": "./results/formal_training/m07_first5000_identity20k",
    },
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml",
        "dst": root / "configs/CoCo_train5000_first_identity20k_then_nongeomlight10k_seed61_formal.yml",
        "dataset_root": str(root / "results/formal_reuse_datasets/CoCo256_train5000_first_formal"),
        "output_dir": "./results/formal_training/m07_first5000_base30k",
    },
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed61_formal.yml",
        "dst": root / "configs/CoCo_train5000_second_identity20k_seed61_formal.yml",
        "dataset_root": str(root / "results/formal_reuse_datasets/CoCo256_train5000_second_formal"),
        "output_dir": "./results/formal_training/m07_second5000_identity20k",
    },
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml",
        "dst": root / "configs/CoCo_train5000_second_identity20k_then_nongeomlight10k_seed61_formal.yml",
        "dataset_root": str(root / "results/formal_reuse_datasets/CoCo256_train5000_second_formal"),
        "output_dir": "./results/formal_training/m07_second5000_base30k",
    },
]
for job in jobs:
    data = yaml.safe_load(job["src"].read_text(encoding="utf-8"))
    data["data"]["dataset_root"] = job["dataset_root"]
    data["data"]["output_dir"] = job["output_dir"]
    job["dst"].write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

if [[ "$CHECK_ONLY" == true ]]; then
  echo "M07 preflight passed."
  echo "Classification: supporting experiment (COCO same-dataset module reuse validation)"
  echo "Subset aliases:"
  echo "  - $FIRST_ALIAS"
  echo "  - $SECOND_ALIAS"
  echo "Transferred module:"
  echo "  - $REFINER"
  echo "Locked final test:"
  echo "  - $FINAL_GT"
  echo "This experiment trains two new base checkpoints on disjoint COCO subsets and tests module reuse without retraining the module."
  exit 0
fi

if [[ -f "$SUMMARY" ]]; then
  echo "M07 summary already exists:"
  echo "$SUMMARY"
  exit 0
fi

run_subset() {
  local tag="$1"
  local base20_cfg="$2"
  local base30_cfg="$3"
  local base20_out="$4"
  local base30_out="$5"
  local base20_ckpt="$6"
  local base30_ckpt="$7"
  local eval_out="$8"
  local eval_wm="$9"
  local transfer_out="${10}"
  local clean_run="${11}"
  local base_attack="${12}"
  local transfer_attack="${13}"

  echo
  echo "[$(date '+%F %T')] ${tag}: train identity20k"
  if [[ -f "$base20_ckpt" ]]; then
    echo "Reuse completed $tag identity20k checkpoint."
  elif [[ -e "$base20_out" ]]; then
    echo "Incomplete or unarchived output exists: $base20_out"
    exit 1
  else
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$(basename "$base20_cfg")" \
      --resume ./ckpts/nonexistent_"$tag"_identity20k_start.pth.tar \
      --image_folder "results/validation/${tag}_identity20k" \
      --seed 61 \
      2>&1 | tee "logs/${tag}_identity20k_train.log"
    if [[ ! -f "$base20_out/ckpts/CoCo_ddpm_final.pth.tar" ]]; then
      echo "Missing $tag identity20k checkpoint."
      exit 1
    fi
    mv "$base20_out/ckpts/CoCo_ddpm_final.pth.tar" "$base20_ckpt"
  fi
  "$PY" scripts/check_training_message_convergence.py \
    --log "logs/${tag}_identity20k_train.log" \
    --tail 20 \
    --max_mean 0.10 \
    --max_last 0.15

  echo
  echo "[$(date '+%F %T')] ${tag}: continue to base30k"
  if [[ -f "$base30_ckpt" ]]; then
    echo "Reuse completed $tag base30k checkpoint."
  elif [[ -e "$base30_out" ]]; then
    echo "Incomplete or unarchived output exists: $base30_out"
    exit 1
  else
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$(basename "$base30_cfg")" \
      --resume "$base20_ckpt" \
      --image_folder "results/validation/${tag}_base30k" \
      --seed 61 \
      2>&1 | tee "logs/${tag}_base30k_train.log"
    if [[ ! -f "$base30_out/ckpts/CoCo_ddpm_final.pth.tar" ]]; then
      echo "Missing $tag base30k checkpoint."
      exit 1
    fi
    mv "$base30_out/ckpts/CoCo_ddpm_final.pth.tar" "$base30_ckpt"
  fi

  echo
  echo "[$(date '+%F %T')] ${tag}: evaluate base on locked finaltest1000"
  if [[ -e "$eval_wm" ]]; then
    png_count="$(find "$eval_wm" -maxdepth 1 -type f -name '*.png' | wc -l)"
    msg_count="$(find "$eval_wm" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
    if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
      echo "Incomplete eval output for $tag: images=$png_count messages=$msg_count"
      exit 1
    fi
    echo "Reuse completed $tag finaltest outputs."
  else
    /usr/bin/time -p "$PY" -u eval_diffusion.py \
      --config "$EVAL_CONFIG" \
      --resume "$base30_ckpt" \
      --image_folder "$eval_out" \
      --run_name finaltest1000 \
      --seed 61 \
      2>&1 | tee "logs/${tag}_base30k_eval.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$eval_wm" \
    2>&1 | tee "logs/${tag}_base30k_psnr_ssim.log"

  mkdir -p "results/attacked_Result/$clean_run/Clean"
  find "$eval_wm" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.webp' \) -print0 |
    while IFS= read -r -d '' image; do
      target="results/attacked_Result/$clean_run/Clean/$(basename "$image")"
      [[ -e "$target" ]] || ln "$image" "$target"
    done
  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$base30_ckpt" \
    --attack_dir "results/attacked_Result/$clean_run" \
    --gt_message_dir "$eval_wm" \
    2>&1 | tee "logs/${tag}_base30k_clean_ba.log"
  "$PY" scripts/check_clean_ba_gate.py \
    --report "results/attacked_Result/$clean_run/ba_report.txt" \
    --minimum 95.0

  if [[ -e "results/attacked_Result/$base_attack" ]]; then
    attack_count="$(find "results/attacked_Result/$base_attack" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete base attack output for $tag."
      exit 1
    fi
    echo "Reuse completed $tag base nongeom attacks."
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$eval_wm" \
      --output_dir results/attacked_Result \
      --run_name "$base_attack" \
      --attack_scope nongeom \
      --seed 61 \
      2>&1 | tee "logs/${tag}_base30k_attack.log"
  fi
  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$base30_ckpt" \
    --attack_dir "results/attacked_Result/$base_attack" \
    --gt_message_dir "$eval_wm" \
    2>&1 | tee "logs/${tag}_base30k_ba.log"
  "$PY" scripts/summarize_ba_report.py \
    "results/attacked_Result/$base_attack/ba_report.txt" \
    --output "results/attacked_Result/$base_attack/ba_summary.json"

  echo
  echo "[$(date '+%F %T')] ${tag}: transfer frozen formal module"
  if [[ -e "$transfer_out" ]]; then
    png_count="$(find "$transfer_out/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
    msg_count="$(find "$transfer_out/refined_images" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
    if [[ ! -f "$transfer_out/report.json" || "$png_count" -ne 1000 || "$msg_count" -ne 1000 || ! -d "$transfer_out/attacked_clean/Clean" ]]; then
      echo "Incomplete transfer output for $tag."
      exit 1
    fi
    echo "Reuse completed transfer output for $tag."
  else
    /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
      --config "$EVAL_CONFIG" \
      --resume "$base30_ckpt" \
      --refiner "$REFINER" \
      --watermarked_dir "$eval_wm" \
      --gt_dir "$FINAL_GT" \
      --output_dir "$transfer_out" \
      --image_size 256 \
      --batch_size 4 \
      --num_workers 2 \
      --hidden_channels 48 \
      --depth 5 \
      --max_delta 0.50 \
      --conditioning cover \
      --seed 61 \
      2>&1 | tee "logs/${tag}_transfer_apply.log"
  fi

  "$PY" calculate_psnr_ssim.py \
    --gt_path "$FINAL_GT" \
    --results_path "$transfer_out/refined_images" \
    2>&1 | tee "logs/${tag}_transfer_psnr_ssim.log"
  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$base30_ckpt" \
    --attack_dir "$transfer_out/attacked_clean" \
    --gt_message_dir "$eval_wm" \
    2>&1 | tee "logs/${tag}_transfer_clean_ba.log"

  if [[ -e "results/attacked_Result/$transfer_attack" ]]; then
    attack_count="$(find "results/attacked_Result/$transfer_attack" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "$attack_count" -ne 11 ]]; then
      echo "Incomplete transfer attack output for $tag."
      exit 1
    fi
    echo "Reuse completed transfer nongeom attacks for $tag."
  else
    "$PY" attack_eval_diffw_paper_matched.py \
      --gt_path "$FINAL_GT" \
      --results_path "$transfer_out/refined_images" \
      --output_dir results/attacked_Result \
      --run_name "$transfer_attack" \
      --attack_scope nongeom \
      --seed 61 \
      2>&1 | tee "logs/${tag}_transfer_attack.log"
  fi
  "$PY" calculate_ba.py \
    --config "$EVAL_CONFIG" \
    --resume "$base30_ckpt" \
    --attack_dir "results/attacked_Result/$transfer_attack" \
    --gt_message_dir "$eval_wm" \
    2>&1 | tee "logs/${tag}_transfer_ba.log"
  "$PY" scripts/summarize_ba_report.py \
    "results/attacked_Result/$transfer_attack/ba_report.txt" \
    --output "results/attacked_Result/$transfer_attack/ba_summary.json"
}

run_subset "m07_first5000" "$FIRST_BASE20_CONFIG" "$FIRST_BASE30_CONFIG" \
  "$FIRST_BASE20_OUT" "$FIRST_BASE30_OUT" "$FIRST_BASE20_CKPT" "$FIRST_BASE30_CKPT" \
  "$FIRST_EVAL_OUT" "$FIRST_EVAL_WM" "$FIRST_TRANSFER_OUT" \
  "$FIRST_BASE_CLEAN" "$FIRST_BASE_ATTACK" "$FIRST_TRANSFER_ATTACK"

run_subset "m07_second5000" "$SECOND_BASE20_CONFIG" "$SECOND_BASE30_CONFIG" \
  "$SECOND_BASE20_OUT" "$SECOND_BASE30_OUT" "$SECOND_BASE20_CKPT" "$SECOND_BASE30_CKPT" \
  "$SECOND_EVAL_OUT" "$SECOND_EVAL_WM" "$SECOND_TRANSFER_OUT" \
  "$SECOND_BASE_CLEAN" "$SECOND_BASE_ATTACK" "$SECOND_TRANSFER_ATTACK"

echo
echo "[$(date '+%F %T')] Step 9/9: summarize module reuse support experiment"
"$PY" scripts/summarize_m07_coco_subset_module_reuse.py \
  --project_root "$ROOT" \
  --output "$SUMMARY"

echo "Summary: $SUMMARY"
