#!/usr/bin/env bash

set -euo pipefail

SOURCE_REPO="${SOURCE_REPO:-/home/jeffrey/projects/O256-PAPER-github-backup}"
PUBLIC_TARGET="${PUBLIC_TARGET:-/home/jeffrey/projects/O256-github-public}"
LAB_TARGET="${LAB_TARGET:-/home/jeffrey/projects/O256-lab-handoff-full}"
VAE_SOURCE="${VAE_SOURCE:-/home/jeffrey/projects/VAE_backups/sd21_original_vae_20260604}"
OVERWRITE=0
SAFE_CWD="${SAFE_CWD:-/home/jeffrey/projects}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/assemble_release_targets.sh [--overwrite]

Default outputs:
  Public GitHub target:
    /home/jeffrey/projects/O256-github-public
  Lab server full target:
    /home/jeffrey/projects/O256-lab-handoff-full

Environment overrides:
  SOURCE_REPO
  PUBLIC_TARGET
  LAB_TARGET
  VAE_SOURCE
  SAFE_CWD
EOF
}

for arg in "$@"; do
  case "$arg" in
    --overwrite)
      OVERWRITE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 1
  fi
}

switch_to_safe_cwd() {
  if [[ ! -d "$SAFE_CWD" ]]; then
    echo "Safe working directory not found: $SAFE_CWD" >&2
    exit 1
  fi
  cd "$SAFE_CWD"
}

prepare_target_dir() {
  local path="$1"
  if [[ -e "$path" ]]; then
    if [[ "$OVERWRITE" -ne 1 ]]; then
      echo "Target already exists: $path" >&2
      echo "Re-run with --overwrite if you want to recreate it." >&2
      exit 1
    fi
    rm -rf "$path"
  fi
  mkdir -p "$path"
}

sync_repo_tree() {
  local target="$1"
  rsync -a \
    --exclude '.git/' \
    --exclude 'release_targets/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "$SOURCE_REPO"/ "$target"/
}

write_public_note() {
  local target="$1"
  cat > "$target/TARGET_PACKAGE.md" <<'EOF'
# Target Package

This folder is the public GitHub export target.

It is intended for:

- portfolio presentation
- public code sharing
- lightweight reproducibility

It intentionally excludes:

- datasets
- checkpoints
- SD2.1 / VAE large files
- large generated outputs
EOF
}

write_lab_note() {
  local target="$1"
  cat > "$target/TARGET_PACKAGE.md" <<'EOF'
# Target Package

This folder is the lab server full handoff target.

It is intended for:

- academic handoff
- future student reuse
- fuller internal preservation

This version is expected to include the SD2.1 / VAE backup.
EOF
}

copy_vae() {
  local target="$1"
  local vae_target="$target/external_assets/vae/sd21_original_vae_20260604"
  mkdir -p "$(dirname "$vae_target")"
  rsync -a "$VAE_SOURCE"/ "$vae_target"/
}

print_summary() {
  local public_target="$1"
  local lab_target="$2"
  echo
  echo "Created release targets:"
  echo "  Public: $public_target"
  echo "  Lab:    $lab_target"
  echo
  du -sh "$public_target" "$lab_target"
}

require_dir "$SOURCE_REPO"
require_dir "$VAE_SOURCE"
switch_to_safe_cwd

prepare_target_dir "$PUBLIC_TARGET"
prepare_target_dir "$LAB_TARGET"

sync_repo_tree "$PUBLIC_TARGET"
sync_repo_tree "$LAB_TARGET"

write_public_note "$PUBLIC_TARGET"
write_lab_note "$LAB_TARGET"
copy_vae "$LAB_TARGET"

print_summary "$PUBLIC_TARGET" "$LAB_TARGET"
