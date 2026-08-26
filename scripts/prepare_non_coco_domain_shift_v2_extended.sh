#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/jeffrey/projects/DataSet"
RESIZE_PY="/home/jeffrey/projects/Tools/resize_square.py"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

DIV2K_TRAIN_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
BSDS500_URL="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

MIN_FREE_GB=20

download_file() {
  local url="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]]; then
    echo "Already exists: $dst"
    return 0
  fi
  echo "Downloading: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - --output "$dst" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$dst" "$url"
  else
    echo "Neither curl nor wget is available."
    exit 1
  fi
}

count_manifest() {
  local manifest="$1"
  grep -v '^[[:space:]]*$' "$manifest" | wc -l
}

require_free_space() {
  local path="$1"
  local min_gb="$2"
  local avail_gb
  avail_gb="$(df -BG "$path" | awk 'NR==2 {gsub("G", "", $4); print $4}')"
  if [[ "$avail_gb" -lt "$min_gb" ]]; then
    echo "Insufficient disk space at $path: available=${avail_gb}GB required>=${min_gb}GB"
    exit 1
  fi
  echo "Disk space OK at $path: available=${avail_gb}GB"
}

prepare_div2k_train() {
  local root="$DATA_ROOT/DIV2K_train"
  local raw="$root/raw"
  local zip="$raw/DIV2K_train_HR.zip"
  local extracted="$raw/DIV2K_train_HR"
  mkdir -p "$raw"
  download_file "$DIV2K_TRAIN_URL" "$zip"
  if [[ ! -d "$extracted" ]]; then
    echo "Extracting DIV2K train..."
    unzip -q "$zip" -d "$raw"
  fi
  if [[ ! -f "$root/manifests/eval800.txt" ]]; then
    "$PY" "$RESIZE_PY" \
      --input_dir "$extracted" \
      --output_dir "$root/images_256" \
      --manifest_dir "$root/manifests" \
      --dataset_name DIV2K_train \
      --size 256 \
      --mode center_crop \
      --limit 800 \
      --eval_sizes 100 300 500 800
    cp "$root/manifests/eval800.txt" "$root/manifests/all.txt"
  fi
}

prepare_bsds500() {
  local root="$DATA_ROOT/BSDS500"
  local raw="$root/raw"
  local tgz="$raw/BSR_bsds500.tgz"
  local extracted="$raw/BSR/BSDS500/data/images"
  mkdir -p "$raw"
  download_file "$BSDS500_URL" "$tgz"
  if [[ ! -d "$extracted" ]]; then
    echo "Extracting BSDS500..."
    tar -xzf "$tgz" -C "$raw"
  fi
  if [[ ! -f "$root/manifests/eval100.txt" ]]; then
    "$PY" "$RESIZE_PY" \
      --input_dir "$extracted" \
      --output_dir "$root/images_256" \
      --manifest_dir "$root/manifests" \
      --dataset_name BSDS500 \
      --size 256 \
      --mode center_crop \
      --limit 500 \
      --prefix bsds_ \
      --eval_sizes 100 300 500
  fi
}

prepare_natural_mix_eval1000() {
  local root="$DATA_ROOT/Natural_Mix_DIV2K_BSDS_eval1000"
  local out="$root/images_256"
  local manifests="$root/manifests"
  mkdir -p "$out" "$manifests"

  local manifest="$manifests/eval1000.txt"
  : > "$manifest"

  local idx=0
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    idx=$((idx + 1))
    local out_name
    out_name="$(printf 'natural_mix_%04d_div2k_train_%s' "$idx" "$name")"
    ln -sfn "$DATA_ROOT/DIV2K_train/images_256/$name" "$out/$out_name"
    echo "$out_name" >> "$manifest"
  done < "$DATA_ROOT/DIV2K_train/manifests/eval800.txt"

  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    idx=$((idx + 1))
    local out_name
    out_name="$(printf 'natural_mix_%04d_div2k_valid_%s' "$idx" "$name")"
    ln -sfn "$DATA_ROOT/DIV2K_valid/images_256/$name" "$out/$out_name"
    echo "$out_name" >> "$manifest"
  done < "$DATA_ROOT/DIV2K_valid/manifests/eval100.txt"

  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    idx=$((idx + 1))
    local out_name
    out_name="$(printf 'natural_mix_%04d_bsds500_%s' "$idx" "$name")"
    ln -sfn "$DATA_ROOT/BSDS500/images_256/$name" "$out/$out_name"
    echo "$out_name" >> "$manifest"
  done < "$DATA_ROOT/BSDS500/manifests/eval100.txt"

  cp "$manifest" "$manifests/all.txt"
  local count
  count="$(count_manifest "$manifest")"
  if [[ "$count" -ne 1000 ]]; then
    echo "Natural mix manifest count mismatch: expected=1000 actual=$count"
    exit 1
  fi

  cat > "$manifests/README.md" <<EOF
# Natural_Mix_DIV2K_BSDS_eval1000

- purpose: non-COCO natural-image generalization check
- count: 1000
- composition:
  - DIV2K train HR resized: 800
  - DIV2K valid HR resized: 100
  - BSDS500 resized: 100
- image size: 256x256
- resize mode: center crop
- manifest: \`eval1000.txt\`
EOF
}

prepare_abo_eval273() {
  local manifest="$DATA_ROOT/ABO_demo_renders/manifests/all.txt"
  if [[ ! -f "$manifest" ]]; then
    echo "Missing ABO all manifest: $manifest"
    exit 1
  fi
  local count
  count="$(count_manifest "$manifest")"
  if [[ "$count" -ne 273 ]]; then
    echo "ABO manifest count mismatch: expected=273 actual=$count"
    exit 1
  fi
}

main() {
  for required in "$PY" "$RESIZE_PY" "$DATA_ROOT/DIV2K_valid/manifests/eval100.txt" "$DATA_ROOT/ABO_demo_renders/manifests/all.txt" "$DATA_ROOT/Kodak/manifests/eval24.txt"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing required input: $required"
      exit 1
    fi
  done

  require_free_space "$DATA_ROOT" "$MIN_FREE_GB"
  prepare_div2k_train
  prepare_bsds500
  prepare_natural_mix_eval1000
  prepare_abo_eval273

  echo "Prepared v2 extended non-COCO datasets:"
  echo "  - $DATA_ROOT/Natural_Mix_DIV2K_BSDS_eval1000/manifests/eval1000.txt ($(count_manifest "$DATA_ROOT/Natural_Mix_DIV2K_BSDS_eval1000/manifests/eval1000.txt"))"
  echo "  - $DATA_ROOT/Kodak/manifests/eval24.txt ($(count_manifest "$DATA_ROOT/Kodak/manifests/eval24.txt"))"
  echo "  - $DATA_ROOT/ABO_demo_renders/manifests/all.txt ($(count_manifest "$DATA_ROOT/ABO_demo_renders/manifests/all.txt"))"
}

main "$@"
