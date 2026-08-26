#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PROTOCOL_ROOT="/home/jeffrey/projects/實驗結果/14_mainline_formal_protocol"
FORMAL_EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments"
DEST_ROOT="/home/jeffrey/projects/O256-PAPER-github-backup"

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required path: $path"; exit 1; }
}

copy_file() {
  local src="$1"
  local dst="$2"
  if [[ "$CHECK_ONLY" == true ]]; then
    echo "file: $src -> $dst"
    return
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

copy_dir() {
  local src="$1"
  local dst="$2"
  if [[ "$CHECK_ONLY" == true ]]; then
    echo "dir:  $src -> $dst"
    return
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  find "$dst" -type d -name '__pycache__' -prune -exec rm -rf {} +
  find "$dst" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
}

copy_tree_filtered() {
  local src="$1"
  local dst="$2"
  local pattern="$3"
  if [[ "$CHECK_ONLY" == true ]]; then
    echo "filtered dir: $src [$pattern] -> $dst"
    return
  fi
  mkdir -p "$dst"
  while IFS= read -r -d '' file; do
    local rel="${file#$src/}"
    mkdir -p "$dst/$(dirname "$rel")"
    cp -a "$file" "$dst/$rel"
  done < <(find "$src" -type f -name "$pattern" -print0)
}

copy_summary_dir() {
  local exp_id="$1"
  local src="$2"
  local dst="$DEST_ROOT/results_summaries/$exp_id"
  if [[ ! -d "$src" ]]; then
    echo "warn: missing summary source: $src"
    return
  fi
  if [[ "$CHECK_ONLY" == true ]]; then
    echo "summary: $src -> $dst"
    return
  fi
  mkdir -p "$dst"
  find "$src" -maxdepth 2 -type f \( -name 'summary.md' -o -name 'summary.csv' -o -name 'per_attack_ba.csv' -o -name 'variant_summary.csv' -o -name 'transition_summary.csv' -o -name 'comparisons.csv' \) -print0 |
    while IFS= read -r -d '' file; do
      local rel="${file#$src/}"
      mkdir -p "$dst/$(dirname "$rel")"
      cp -a "$file" "$dst/$rel"
    done
}

require_path "$SRC_ROOT"
require_path "$PROTOCOL_ROOT"
require_path "$FORMAL_EXP_ROOT"
require_path "$SRC_ROOT/environment.yml"
require_path "$SRC_ROOT/ENVIRONMENT.md"
require_path "$SRC_ROOT/.gitignore"

if [[ "$CHECK_ONLY" != true ]]; then
  rm -rf "$DEST_ROOT"
  mkdir -p "$DEST_ROOT"
fi

cd "$SRC_ROOT"

for file in README.md environment.yml ENVIRONMENT.md .gitignore; do
  [[ -f "$file" ]] && copy_file "$SRC_ROOT/$file" "$DEST_ROOT/$file"
done

find "$SRC_ROOT" -maxdepth 1 -type f -name '*.py' -print0 |
  while IFS= read -r -d '' file; do
    copy_file "$file" "$DEST_ROOT/$(basename "$file")"
  done

copy_dir "$SRC_ROOT/configs" "$DEST_ROOT/configs"
copy_dir "$SRC_ROOT/scripts" "$DEST_ROOT/scripts"
copy_dir "$SRC_ROOT/models" "$DEST_ROOT/models"
copy_dir "$SRC_ROOT/noise_layers" "$DEST_ROOT/noise_layers"
copy_dir "$SRC_ROOT/utils" "$DEST_ROOT/utils"
copy_tree_filtered "$SRC_ROOT/datasets" "$DEST_ROOT/datasets" "*.py"
copy_dir "$SRC_ROOT/docs" "$DEST_ROOT/docs"

mkdir -p "$DEST_ROOT/docs/experiment_notes"
for doc in \
  "O256_20260629之後實驗代表意義簡表_20260706.md" \
  "O256_逐實驗實作方式與中文脈絡_20260707.md" \
  "O256_20260629之後逐實驗脈絡_20260706.md" \
  "O256_20260629至20260705實驗總覽與研究脈絡.md" \
  "正式與探索實驗分類規則.md" \
  "DiffW_like與O256_RGB最佳版本完整對照表_20260630.md" \
  "Training_side口試答法與最終總表_20260629.md" \
  "O256_exp48_bitplane_refiner_damage_tracking規劃_20260708.md" \
  "O256_exp49_evaluation_integrity_audit規劃_20260708.md" \
  "O256_exp50_decoderaware_refiner_probe規劃_20260708.md" \
  "O256_exp51_decoderaware_stageE_crossseed_probe規劃_20260708.md"; do
  if [[ -f "$PROTOCOL_ROOT/$doc" ]]; then
    copy_file "$PROTOCOL_ROOT/$doc" "$DEST_ROOT/docs/experiment_notes/$doc"
  else
    echo "warn: missing doc: $doc"
  fi
done

if [[ -f "/home/jeffrey/projects/重要實驗結果保留清單_DO_NOT_DELETE_20260702.md" ]]; then
  copy_file "/home/jeffrey/projects/重要實驗結果保留清單_DO_NOT_DELETE_20260702.md" \
    "$DEST_ROOT/docs/experiment_notes/重要實驗結果保留清單_DO_NOT_DELETE_20260702.md"
fi

copy_summary_dir "exp40" "$FORMAL_EXP_ROOT/40_formal_o256_full_from_scratch_dropout_margin_seed61_20260704"
copy_summary_dir "exp41" "$FORMAL_EXP_ROOT/41_formal_o256_full_from_scratch_dropout_margin_seed62_20260705"
copy_summary_dir "exp48" "$FORMAL_EXP_ROOT/48_supplementary_o256_bitplane_refiner_damage_tracking_20260708"
copy_summary_dir "exp49" "$FORMAL_EXP_ROOT/49_supplementary_o256_evaluation_integrity_audit_20260708"
copy_summary_dir "exp50" "$FORMAL_EXP_ROOT/50_supplementary_o256_decoderaware_refiner_probe_seed62_20260708"
copy_summary_dir "exp51" "$FORMAL_EXP_ROOT/51_supplementary_o256_decoderaware_stageE_crossseed_probe_20260708"

if [[ "$CHECK_ONLY" == true ]]; then
  echo "Check-only complete. No files copied."
else
  du -sh "$DEST_ROOT"
  echo "GitHub backup prepared: $DEST_ROOT"
fi
