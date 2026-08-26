#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
DATA_ROOT="/home/jeffrey/projects/DataSet"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended"

BASE_CONFIG="CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
BASE_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"

CHECK_ONLY="false"
RUN_ATTACKS="true"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ "${1:-}" == "--no-attacks" ]]; then
  RUN_ATTACKS="false"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only|--no-attacks]"
  exit 2
fi

cd "$ROOT"
mkdir -p "$EXP_ROOT"/{dataset_roots,generated_configs,logs,summaries,lpips}

DATASETS=(
  "NaturalMix_DIV2K_BSDS_eval1000|$DATA_ROOT/Natural_Mix_DIV2K_BSDS_eval1000/images_256|$DATA_ROOT/Natural_Mix_DIV2K_BSDS_eval1000/manifests/eval1000.txt|1000|non_coco_natural_large"
  "Kodak_eval24|$DATA_ROOT/Kodak/images_256|$DATA_ROOT/Kodak/manifests/eval24.txt|24|classic_quality_sanity"
  "ABO_demo_renders_eval273|$DATA_ROOT/ABO_demo_renders/images_256|$DATA_ROOT/ABO_demo_renders/manifests/all.txt|273|cg_rendered_object_stress"
)

ATTACK_NAMES=(
  "Paper_Dropout_r03"
  "Paper_GaussianBlur_sigma2"
  "Paper_GaussianBlur_sigma3"
  "Paper_GaussianBlur_sigma4"
  "Paper_GaussianNoise_arg03"
  "Paper_GaussianNoise_sigma03_var009"
  "Paper_JPEG_Q30"
  "Paper_JPEG_Q40"
  "Paper_JPEG_Q50"
  "Paper_MedianFilter_w5"
  "Paper_SaltPepper_r02"
)

count_manifest() {
  local manifest="$1"
  grep -v '^[[:space:]]*$' "$manifest" | wc -l
}

count_files() {
  local dir="$1"
  local pattern="$2"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name "$pattern" | wc -l
}

attack_complete() {
  local attack_root="$1"
  local expected="$2"
  for attack in "${ATTACK_NAMES[@]}"; do
    if [[ "$(count_files "$attack_root/$attack" '*.png')" -ne "$expected" ]]; then
      return 1
    fi
  done
  return 0
}

prepare_alias_dataset() {
  local name="$1"
  local image_dir="$2"
  local manifest="$3"
  local expected="$4"
  local root="$EXP_ROOT/dataset_roots/$name"
  local input_dir="$root/data/CoCo/val/input"
  local gt_dir="$root/data/CoCo/val/gt"
  local train_input_dir="$root/data/CoCo/train/input"
  local train_gt_dir="$root/data/CoCo/train/gt"
  rm -rf "$root"
  mkdir -p "$input_dir" "$gt_dir" "$train_input_dir" "$train_gt_dir"

  local count
  count="$(count_manifest "$manifest")"
  if [[ "$count" -ne "$expected" ]]; then
    echo "Dataset $name manifest count mismatch: expected=$expected actual=$count"
    exit 1
  fi

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    local src="$image_dir/$file"
    if [[ ! -f "$src" ]]; then
      echo "Missing image for $name: $src"
      exit 1
    fi
    ln -sfn "$src" "$input_dir/$file"
    ln -sfn "$src" "$gt_dir/$file"
    # eval_diffusion.py constructs both train and val loaders in eval mode.
    # This evaluation-only alias mirrors val into train to keep loader setup non-empty.
    ln -sfn "$src" "$train_input_dir/$file"
    ln -sfn "$src" "$train_gt_dir/$file"
  done < "$manifest"
  echo "$root"
}

write_config() {
  local name="$1"
  local dataset_root="$2"
  local config_path="$ROOT/configs/CoCo_non_coco_v2_${name}_seed61.yml"
  "$PY" - "$ROOT/configs/$BASE_CONFIG" "$config_path" "$dataset_root" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst, dataset_root = map(Path, sys.argv[1:4])
cfg = yaml.safe_load(src.read_text())
cfg["data"]["dataset_root"] = str(dataset_root)
cfg["training"]["save_checkpoints"] = False
cfg["training"]["save_final_checkpoint"] = False
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  cp "$config_path" "$EXP_ROOT/generated_configs/$(basename "$config_path")"
  echo "$(basename "$config_path")"
}

run_logged() {
  local log="$1"
  shift
  mkdir -p "$(dirname "$log")"
  {
    echo "COMMAND:"
    printf '%q ' "$@"
    echo
    echo
    "$@"
  } 2>&1 | tee "$log"
}

for required in "$PY" "configs/$BASE_CONFIG" "$BASE_CKPT" "$REFINER" "$DATA_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required"
    exit 1
  fi
done

echo "# Non-COCO Domain Shift v2 Extended Run Config" > "$EXP_ROOT/run_config.md"
echo >> "$EXP_ROOT/run_config.md"
echo "- date: 2026-06-25" >> "$EXP_ROOT/run_config.md"
echo "- base config: \`$BASE_CONFIG\`" >> "$EXP_ROOT/run_config.md"
echo "- base checkpoint: \`$BASE_CKPT\`" >> "$EXP_ROOT/run_config.md"
echo "- refiner: \`$REFINER\`" >> "$EXP_ROOT/run_config.md"
echo "- run attacks: \`$RUN_ATTACKS\`" >> "$EXP_ROOT/run_config.md"
echo "- output isolation: v2 uses \`non_coco_v2_*\` result names and does not overwrite v1 outputs." >> "$EXP_ROOT/run_config.md"
echo >> "$EXP_ROOT/run_config.md"
echo "| dataset | expected images | role | dataset_root | config |" >> "$EXP_ROOT/run_config.md"
echo "|---|---:|---|---|---|" >> "$EXP_ROOT/run_config.md"

declare -a PREPARED=()
for row in "${DATASETS[@]}"; do
  IFS='|' read -r name image_dir manifest expected role <<< "$row"
  dataset_root="$(prepare_alias_dataset "$name" "$image_dir" "$manifest" "$expected")"
  config="$(write_config "$name" "$dataset_root")"
  PREPARED+=("$name|$dataset_root|$config|$expected|$role")
  echo "| $name | $expected | $role | \`$dataset_root\` | \`$config\` |" >> "$EXP_ROOT/run_config.md"
done

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Non-COCO domain shift v2 extended preflight passed."
  echo "Experiment root: $EXP_ROOT"
  echo "Prepared datasets:"
  printf '  - %s\n' "${PREPARED[@]}"
  echo "No inference was executed."
  exit 0
fi

echo "[$(date '+%F %T')] Start non-COCO domain shift v2 extended"
for row in "${PREPARED[@]}"; do
  IFS='|' read -r name dataset_root config expected role <<< "$row"
  gt_dir="$dataset_root/data/CoCo/val/gt"
  base_output="results/test_result/non_coco_v2_${name}_base"
  base_wm="$base_output/eval/CoCo"
  refine_output="results/refinement/non_coco_v2_${name}_rgbrefine"

  echo "[$(date '+%F %T')] Dataset: $name"

  base_png_count="$(count_files "$base_wm" '*.png')"
  base_msg_count="$(count_files "$base_wm" '*_message.npy')"
  if [[ -d "$base_wm" && ( "$base_png_count" -ne "$expected" || "$base_msg_count" -ne "$expected" ) ]]; then
    echo "Removing incomplete base output for $name: images=$base_png_count messages=$base_msg_count expected=$expected"
    rm -rf "$base_output"
  fi

  if [[ ! -d "$base_wm" ]]; then
    run_logged "$EXP_ROOT/logs/${name}_base_eval.log" \
      "$PY" -u eval_diffusion.py \
        --config "$config" \
        --resume "$BASE_CKPT" \
        --image_folder "$base_output" \
        --run_name eval \
        --seed 61
  fi

  run_logged "$EXP_ROOT/logs/${name}_base_psnr_ssim.log" \
    "$PY" calculate_psnr_ssim.py --gt_path "$gt_dir" --results_path "$base_wm"

  run_logged "$EXP_ROOT/logs/${name}_base_lpips.log" \
    "$PY" scripts/calculate_lpips_dir.py \
      --gt_dir "$gt_dir" \
      --results_dir "$base_wm" \
      --output_dir "$EXP_ROOT/lpips/$name/base" \
      --batch_size 8 \
      --net alex \
      --device auto

  clean_base="results/attacked_Result/clean_non_coco_v2_${name}_base"
  rm -rf "$clean_base"
  mkdir -p "$clean_base/Clean"
  find "$base_wm" -maxdepth 1 -type f -name '*.png' -print0 |
    while IFS= read -r -d '' image; do
      ln -f "$image" "$clean_base/Clean/$(basename "$image")"
    done
  run_logged "$EXP_ROOT/logs/${name}_base_clean_ba.log" \
    "$PY" calculate_ba.py --config "$config" --resume "$BASE_CKPT" --attack_dir "$clean_base" --gt_message_dir "$base_wm"

  refine_png_count="$(count_files "$refine_output/refined_images" '*.png')"
  if [[ -d "$refine_output/refined_images" && "$refine_png_count" -ne "$expected" ]]; then
    echo "Removing incomplete RGB refinement output for $name: images=$refine_png_count expected=$expected"
    rm -rf "$refine_output"
  fi

  if [[ ! -d "$refine_output/refined_images" ]]; then
    run_logged "$EXP_ROOT/logs/${name}_rgbrefine_apply.log" \
      "$PY" scripts/apply_rgb_refinement.py \
        --config "$config" \
        --resume "$BASE_CKPT" \
        --refiner "$REFINER" \
        --watermarked_dir "$base_wm" \
        --gt_dir "$gt_dir" \
        --output_dir "$refine_output" \
        --image_size 256 \
        --batch_size 4 \
        --num_workers 2 \
        --hidden_channels 48 \
        --depth 5 \
        --max_delta 0.50 \
        --conditioning cover \
        --seed 61
  fi

  run_logged "$EXP_ROOT/logs/${name}_rgb_psnr_ssim.log" \
    "$PY" calculate_psnr_ssim.py --gt_path "$gt_dir" --results_path "$refine_output/refined_images"

  run_logged "$EXP_ROOT/logs/${name}_rgb_lpips.log" \
    "$PY" scripts/calculate_lpips_dir.py \
      --gt_dir "$gt_dir" \
      --results_dir "$refine_output/refined_images" \
      --output_dir "$EXP_ROOT/lpips/$name/rgbrefine" \
      --batch_size 8 \
      --net alex \
      --device auto

  run_logged "$EXP_ROOT/logs/${name}_rgb_clean_ba.log" \
    "$PY" calculate_ba.py --config "$config" --resume "$BASE_CKPT" --attack_dir "$refine_output/attacked_clean" --gt_message_dir "$base_wm"

  if [[ "$RUN_ATTACKS" == "true" ]]; then
    base_attack="non_coco_v2_${name}_base_nongeom"
    rgb_attack="non_coco_v2_${name}_rgbrefine_nongeom"
    if [[ -d "results/attacked_Result/$base_attack" ]] && ! attack_complete "results/attacked_Result/$base_attack" "$expected"; then
      echo "Removing incomplete base attack output for $name"
      rm -rf "results/attacked_Result/$base_attack"
    fi
    if [[ ! -f "results/attacked_Result/$base_attack/ba_report.txt" ]]; then
      run_logged "$EXP_ROOT/logs/${name}_base_attack.log" \
        "$PY" attack_eval_diffw_paper_matched.py \
          --gt_path "$gt_dir" \
          --results_path "$base_wm" \
          --output_dir results/attacked_Result \
          --run_name "$base_attack" \
          --attack_scope nongeom \
          --seed 61
      run_logged "$EXP_ROOT/logs/${name}_base_attack_ba.log" \
        "$PY" calculate_ba.py --config "$config" --resume "$BASE_CKPT" --attack_dir "results/attacked_Result/$base_attack" --gt_message_dir "$base_wm"
    fi

    if [[ -d "results/attacked_Result/$rgb_attack" ]] && ! attack_complete "results/attacked_Result/$rgb_attack" "$expected"; then
      echo "Removing incomplete RGB attack output for $name"
      rm -rf "results/attacked_Result/$rgb_attack"
    fi
    if [[ ! -f "results/attacked_Result/$rgb_attack/ba_report.txt" ]]; then
      run_logged "$EXP_ROOT/logs/${name}_rgb_attack.log" \
        "$PY" attack_eval_diffw_paper_matched.py \
          --gt_path "$gt_dir" \
          --results_path "$refine_output/refined_images" \
          --output_dir results/attacked_Result \
          --run_name "$rgb_attack" \
          --attack_scope nongeom \
          --seed 61
      run_logged "$EXP_ROOT/logs/${name}_rgb_attack_ba.log" \
        "$PY" calculate_ba.py --config "$config" --resume "$BASE_CKPT" --attack_dir "results/attacked_Result/$rgb_attack" --gt_message_dir "$base_wm"
    fi
  fi
done

run_logged "$EXP_ROOT/logs/summarize_v2_extended.log" \
  "$PY" scripts/summarize_non_coco_domain_shift_v2_extended.py --exp_root "$EXP_ROOT"

echo "[$(date '+%F %T')] Done."
echo "Experiment root: $EXP_ROOT"
