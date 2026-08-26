#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

FORMAL_ROOT="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
FORMAL_TRAIN_MANIFEST="$FORMAL_ROOT/manifests/train.txt"
FORMAL_FINAL_GT="$FORMAL_ROOT/data/CoCo/val/gt"

RAW_COCO_TRAIN_DIR="/home/jeffrey/projects/DiffW/datasets/CoCo_Dataset_2017/train2017"

BASE20_TEMPLATE="configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed61_formal.yml"
BASE30_TEMPLATE="configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml"
EVAL_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

ALT_FILELIST="results/formal_reuse_datasets/CoCo256_alt_train10000_seed61/names.txt"
ALT_ALIAS="results/formal_reuse_datasets/CoCo256_alt_train10000_seed61"
ALT_RESIZED_CACHE="$ALT_ALIAS/source_images_256"

ALT_BASE20_CONFIG="configs/CoCo_alt_train10000_identity20k_seed61_formal.yml"
ALT_BASE30_CONFIG="configs/CoCo_alt_train10000_identity20k_then_nongeomlight10k_seed61_formal.yml"

ALT_BASE20_OUT="results/formal_training/m07clean_alt10000_identity20k"
ALT_BASE30_OUT="results/formal_training/m07clean_alt10000_base30k"
ALT_BASE20_CKPT="Result/CoCo_ddpm_m07clean_alt10000_identity20k_final.pth.tar"
ALT_BASE30_CKPT="Result/CoCo_ddpm_m07clean_alt10000_base30k_final.pth.tar"

ALT_EVAL_OUT="results/test_result/m07clean_alt10000_base30k_finaltest1000"
ALT_EVAL_WM="$ALT_EVAL_OUT/finaltest1000/CoCo"
ALT_TRANSFER_OUT="results/refinement/m07clean_alt10000_transfer_apply_finaltest1000"

ALT_BASE_CLEAN="clean_m07clean_alt10000_base30k"
ALT_BASE_ATTACK="m07clean_alt10000_base30k_nongeom"
ALT_TRANSFER_ATTACK="m07clean_alt10000_transfer_nongeom"

SUMMARY_DIR="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/04_cross_seed/module_reuse_same_coco_alt10000_seed61"
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

for required in "$PY" "$FORMAL_ROOT" "$FORMAL_TRAIN_MANIFEST" "$FORMAL_FINAL_GT" "$RAW_COCO_TRAIN_DIR" "$BASE20_TEMPLATE" "$BASE30_TEMPLATE" "configs/$EVAL_CONFIG" "$REFINER"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

echo "[$(date '+%F %T')] Step 0/7: prepare alternate COCO train10000 file list"
mkdir -p "$(dirname "$ALT_FILELIST")"
"$PY" - <<'PY'
from pathlib import Path
import json

formal_train = Path("/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/manifests/train.txt")
raw_dir = Path("/home/jeffrey/projects/DiffW/datasets/CoCo_Dataset_2017/train2017")
out_list = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/formal_reuse_datasets/CoCo256_alt_train10000_seed61/names.txt")
meta = out_list.with_name("selection_metadata.json")

formal_names = {line.strip() for line in formal_train.read_text(encoding="utf-8").splitlines() if line.strip()}
raw_names = sorted(p.name for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
eligible = [name for name in raw_names if name not in formal_names]
selected = eligible[:10000]
if len(selected) != 10000:
    raise RuntimeError(f"Need 10000 alternate images, found {len(selected)}")

out_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
meta.write_text(json.dumps({
    "formal_train_manifest": str(formal_train),
    "raw_image_dir": str(raw_dir),
    "formal_train_count": len(formal_names),
    "raw_count": len(raw_names),
    "eligible_after_exclusion": len(eligible),
    "selected_count": len(selected),
    "overlap_with_formal_train": 0,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"selected alternate train10000 names: {len(selected)}")
print(f"filelist: {out_list}")
PY

echo "[$(date '+%F %T')] Step 1/7: create alternate COCO train10000 alias dataset"
"$PY" scripts/create_coco_filelist_alias.py \
  --image_dir "$RAW_COCO_TRAIN_DIR" \
  --filelist "$ALT_FILELIST" \
  --output_root "$ALT_ALIAS" \
  --resized_cache_dir "$ALT_RESIZED_CACHE" \
  --image_size 256 \
  --overwrite_links

echo "[$(date '+%F %T')] Step 2/7: generate alternate-train10000 configs"
"$PY" - <<'PY'
from pathlib import Path
import yaml

root = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER")
jobs = [
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k_identity_seed61_formal.yml",
        "dst": root / "configs/CoCo_alt_train10000_identity20k_seed61_formal.yml",
        "dataset_root": str(root / "results/formal_reuse_datasets/CoCo256_alt_train10000_seed61"),
        "output_dir": "./results/formal_training/m07clean_alt10000_identity20k",
    },
    {
        "src": root / "configs/CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml",
        "dst": root / "configs/CoCo_alt_train10000_identity20k_then_nongeomlight10k_seed61_formal.yml",
        "dataset_root": str(root / "results/formal_reuse_datasets/CoCo256_alt_train10000_seed61"),
        "output_dir": "./results/formal_training/m07clean_alt10000_base30k",
    },
]
for job in jobs:
    data = yaml.safe_load(job["src"].read_text(encoding="utf-8"))
    data["data"]["dataset_root"] = job["dataset_root"]
    data["data"]["output_dir"] = job["output_dir"]
    job["dst"].write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

if [[ "$CHECK_ONLY" == true ]]; then
  echo "M07-clean preflight passed."
  echo "Classification: supporting experiment (same-COCO-family alternate-train10000 module reuse)"
  echo "Status: supersedes old M07 first5000/second5000 design."
  echo "Formal module source: base-train10000 + refine10000 pairs"
  echo "New base source: alternate COCO train10000 subset disjoint from formal train10000"
  echo "Alternate alias:"
  echo "  - $ALT_ALIAS"
  echo "Alternate resized cache:"
  echo "  - $ALT_RESIZED_CACHE"
  echo "Transferred module:"
  echo "  - $REFINER"
  echo "Locked final test:"
  echo "  - $FORMAL_FINAL_GT"
  exit 0
fi

if [[ -f "$SUMMARY" ]]; then
  echo "M07-clean summary already exists:"
  echo "$SUMMARY"
  exit 0
fi

echo
echo "[$(date '+%F %T')] Step 3/7: train alternate train10000 identity20k"
if [[ -f "$ALT_BASE20_CKPT" ]]; then
  echo "Reuse completed alternate identity20k checkpoint."
elif [[ -e "$ALT_BASE20_OUT" ]]; then
  echo "Incomplete or unarchived output exists: $ALT_BASE20_OUT"
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$(basename "$ALT_BASE20_CONFIG")" \
    --resume ./ckpts/nonexistent_m07clean_alt10000_identity20k_start.pth.tar \
    --image_folder results/validation/m07clean_alt10000_identity20k \
    --seed 61 \
    2>&1 | tee logs/m07clean_alt10000_identity20k_train.log
  if [[ ! -f "$ALT_BASE20_OUT/ckpts/CoCo_ddpm_final.pth.tar" ]]; then
    echo "Missing alternate identity20k checkpoint."
    exit 1
  fi
  mv "$ALT_BASE20_OUT/ckpts/CoCo_ddpm_final.pth.tar" "$ALT_BASE20_CKPT"
fi
"$PY" scripts/check_training_message_convergence.py \
  --log logs/m07clean_alt10000_identity20k_train.log \
  --tail 20 \
  --max_mean 0.10 \
  --max_last 0.15

echo
echo "[$(date '+%F %T')] Step 4/7: continue alternate train10000 to base30k"
if [[ -f "$ALT_BASE30_CKPT" ]]; then
  echo "Reuse completed alternate base30k checkpoint."
elif [[ -e "$ALT_BASE30_OUT" ]]; then
  echo "Incomplete or unarchived output exists: $ALT_BASE30_OUT"
  exit 1
else
  /usr/bin/time -p "$PY" -u train_diffusion.py \
    --config "$(basename "$ALT_BASE30_CONFIG")" \
    --resume "$ALT_BASE20_CKPT" \
    --image_folder results/validation/m07clean_alt10000_base30k \
    --seed 61 \
    2>&1 | tee logs/m07clean_alt10000_base30k_train.log
  if [[ ! -f "$ALT_BASE30_OUT/ckpts/CoCo_ddpm_final.pth.tar" ]]; then
    echo "Missing alternate base30k checkpoint."
    exit 1
  fi
  mv "$ALT_BASE30_OUT/ckpts/CoCo_ddpm_final.pth.tar" "$ALT_BASE30_CKPT"
fi

echo
echo "[$(date '+%F %T')] Step 5/7: evaluate alternate base on locked finaltest1000"
if [[ -e "$ALT_EVAL_WM" ]]; then
  png_count="$(find "$ALT_EVAL_WM" -maxdepth 1 -type f -name '*.png' | wc -l)"
  msg_count="$(find "$ALT_EVAL_WM" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
  if [[ "$png_count" -ne 1000 || "$msg_count" -ne 1000 ]]; then
    echo "Incomplete eval output for alternate base: images=$png_count messages=$msg_count"
    exit 1
  fi
  echo "Reuse completed alternate finaltest outputs."
else
  /usr/bin/time -p "$PY" -u eval_diffusion.py \
    --config "$EVAL_CONFIG" \
    --resume "$ALT_BASE30_CKPT" \
    --image_folder "$ALT_EVAL_OUT" \
    --run_name finaltest1000 \
    --seed 61 \
    2>&1 | tee logs/m07clean_alt10000_base30k_eval.log
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FORMAL_FINAL_GT" \
  --results_path "$ALT_EVAL_WM" \
  2>&1 | tee logs/m07clean_alt10000_base30k_psnr_ssim.log

mkdir -p "results/attacked_Result/$ALT_BASE_CLEAN/Clean"
find "$ALT_EVAL_WM" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.webp' \) -print0 |
  while IFS= read -r -d '' image; do
    target="results/attacked_Result/$ALT_BASE_CLEAN/Clean/$(basename "$image")"
    [[ -e "$target" ]] || ln "$image" "$target"
  done
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$ALT_BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$ALT_BASE_CLEAN" \
  --gt_message_dir "$ALT_EVAL_WM" \
  2>&1 | tee logs/m07clean_alt10000_base30k_clean_ba.log
"$PY" scripts/check_clean_ba_gate.py \
  --report "results/attacked_Result/$ALT_BASE_CLEAN/ba_report.txt" \
  --minimum 95.0

if [[ -e "results/attacked_Result/$ALT_BASE_ATTACK" ]]; then
  attack_count="$(find "results/attacked_Result/$ALT_BASE_ATTACK" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete base attack output for alternate base."
    exit 1
  fi
  echo "Reuse completed alternate base nongeom attacks."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FORMAL_FINAL_GT" \
    --results_path "$ALT_EVAL_WM" \
    --output_dir results/attacked_Result \
    --run_name "$ALT_BASE_ATTACK" \
    --attack_scope nongeom \
    --seed 61 \
    2>&1 | tee logs/m07clean_alt10000_base30k_attack.log
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$ALT_BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$ALT_BASE_ATTACK" \
  --gt_message_dir "$ALT_EVAL_WM" \
  2>&1 | tee logs/m07clean_alt10000_base30k_ba.log
"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$ALT_BASE_ATTACK/ba_report.txt" \
  --output "results/attacked_Result/$ALT_BASE_ATTACK/ba_summary.json"

echo
echo "[$(date '+%F %T')] Step 6/7: transfer frozen formal module onto alternate base"
if [[ -e "$ALT_TRANSFER_OUT" ]]; then
  png_count="$(find "$ALT_TRANSFER_OUT/refined_images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  msg_count="$(find "$ALT_TRANSFER_OUT/refined_images" -maxdepth 1 -type f -name '*_message.npy' 2>/dev/null | wc -l)"
  if [[ ! -f "$ALT_TRANSFER_OUT/report.json" || "$png_count" -ne 1000 || "$msg_count" -ne 1000 || ! -d "$ALT_TRANSFER_OUT/attacked_clean/Clean" ]]; then
    echo "Incomplete transfer output for alternate base."
    exit 1
  fi
  echo "Reuse completed transfer output for alternate base."
else
  /usr/bin/time -p "$PY" scripts/apply_rgb_refinement.py \
    --config "$EVAL_CONFIG" \
    --resume "$ALT_BASE30_CKPT" \
    --refiner "$REFINER" \
    --watermarked_dir "$ALT_EVAL_WM" \
    --gt_dir "$FORMAL_FINAL_GT" \
    --output_dir "$ALT_TRANSFER_OUT" \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --hidden_channels 48 \
    --depth 5 \
    --max_delta 0.50 \
    --conditioning cover \
    --seed 61 \
    2>&1 | tee logs/m07clean_alt10000_transfer_apply.log
fi

"$PY" calculate_psnr_ssim.py \
  --gt_path "$FORMAL_FINAL_GT" \
  --results_path "$ALT_TRANSFER_OUT/refined_images" \
  2>&1 | tee logs/m07clean_alt10000_transfer_psnr_ssim.log
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$ALT_BASE30_CKPT" \
  --attack_dir "$ALT_TRANSFER_OUT/attacked_clean" \
  --gt_message_dir "$ALT_EVAL_WM" \
  2>&1 | tee logs/m07clean_alt10000_transfer_clean_ba.log

if [[ -e "results/attacked_Result/$ALT_TRANSFER_ATTACK" ]]; then
  attack_count="$(find "results/attacked_Result/$ALT_TRANSFER_ATTACK" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "$attack_count" -ne 11 ]]; then
    echo "Incomplete transfer attack output for alternate base."
    exit 1
  fi
  echo "Reuse completed transfer nongeom attacks for alternate base."
else
  "$PY" attack_eval_diffw_paper_matched.py \
    --gt_path "$FORMAL_FINAL_GT" \
    --results_path "$ALT_TRANSFER_OUT/refined_images" \
    --output_dir results/attacked_Result \
    --run_name "$ALT_TRANSFER_ATTACK" \
    --attack_scope nongeom \
    --seed 61 \
    2>&1 | tee logs/m07clean_alt10000_transfer_attack.log
fi
"$PY" calculate_ba.py \
  --config "$EVAL_CONFIG" \
  --resume "$ALT_BASE30_CKPT" \
  --attack_dir "results/attacked_Result/$ALT_TRANSFER_ATTACK" \
  --gt_message_dir "$ALT_EVAL_WM" \
  2>&1 | tee logs/m07clean_alt10000_transfer_ba.log
"$PY" scripts/summarize_ba_report.py \
  "results/attacked_Result/$ALT_TRANSFER_ATTACK/ba_report.txt" \
  --output "results/attacked_Result/$ALT_TRANSFER_ATTACK/ba_summary.json"

echo
echo "[$(date '+%F %T')] Step 7/7: summarize M07-clean"
"$PY" scripts/summarize_m07_clean_alt10000_module_reuse.py \
  --project_root "$ROOT" \
  --alias_manifest "$ALT_ALIAS/manifest.json" \
  --output "$SUMMARY"

echo "Summary: $SUMMARY"
