#!/usr/bin/env bash
set -euo pipefail

PROJECTS_ROOT="${PROJECTS_ROOT:-/home/jeffrey/projects}"
SOURCE_REPO="${SOURCE_REPO:-$PROJECTS_ROOT/O256-PAPER-github-backup}"
SOURCE_DATASET="${SOURCE_DATASET:-$PROJECTS_ROOT/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked}"
SOURCE_VAE="${SOURCE_VAE:-$PROJECTS_ROOT/VAE_backups/sd21_original_vae_20260604}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PROJECTS_ROOT/twcc_stagee_exp53_upload}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$PROJECTS_ROOT/twcc_stagee_exp53_upload.tar.gz}"

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
}

require_path "$SOURCE_REPO"
require_path "$SOURCE_DATASET"
require_path "$SOURCE_VAE"

rm -rf "$BUNDLE_ROOT"
mkdir -p "$BUNDLE_ROOT"

echo "Copy repo..."
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'twcc_results' \
  "$SOURCE_REPO/" "$BUNDLE_ROOT/O256-PAPER-github-backup/"

echo "Copy locked COCO dataset..."
mkdir -p "$BUNDLE_ROOT/experiment_input_folder/CoCo_256_paper_final"
# The locked dataset uses relative symlinks to source_images_256 on the
# original machine. Dereference them so the TWCC upload is self-contained.
rsync -aL "$SOURCE_DATASET/" "$BUNDLE_ROOT/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/"

echo "Copy SD2.1 VAE into Diffusers model layout..."
mkdir -p "$BUNDLE_ROOT/stable-diffusion-2-1-base/vae"
rsync -a "$SOURCE_VAE/" "$BUNDLE_ROOT/stable-diffusion-2-1-base/vae/"

cat > "$BUNDLE_ROOT/SET_PATHS_AND_PREFLIGHT.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Run from the uploaded bundle root on TWCC.
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TWCC_O256_ROOT="$BUNDLE_ROOT/O256-PAPER-github-backup"
export TWCC_FORMAL_DATASET_ROOT="$BUNDLE_ROOT/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
export TWCC_MODEL_PATH="$BUNDLE_ROOT/stable-diffusion-2-1-base"
export TWCC_EXP_ROOT="${TWCC_EXP_ROOT:-$BUNDLE_ROOT/results/53_formal_o256_full_from_scratch_stageE_seed61_twcc}"
export TWCC_PYTHON="${TWCC_PYTHON:-python}"

mkdir -p "$TWCC_EXP_ROOT"

cd "$TWCC_O256_ROOT"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx o256-watermark; then
    conda activate o256-watermark
    export TWCC_PYTHON="$(which python)"
  fi
fi
bash scripts/twcc/collect_twcc_env.sh | tee "$TWCC_EXP_ROOT/twcc_env_$(date +%Y%m%d_%H%M%S).log"
bash scripts/twcc/run_twcc_exp53_full_from_scratch_stageE_seed61.sh --check-only

echo
echo "Preflight passed. To start the full run:"
echo "cd \"$TWCC_O256_ROOT\""
echo "bash scripts/twcc/run_twcc_exp53_full_from_scratch_stageE_seed61.sh 2>&1 | tee \"$TWCC_EXP_ROOT/twcc_exp53_full_run.log\""
EOF
chmod +x "$BUNDLE_ROOT/SET_PATHS_AND_PREFLIGHT.sh"

cat > "$BUNDLE_ROOT/README_UPLOAD_ME.md" <<'EOF'
# TWCC Upload Bundle

Upload this whole folder to TWCC, preferably under `/work/<user>/`.

After upload, install the tested Python environment and run preflight:

```bash
cd /work/<user>/twcc_stagee_exp53_upload
bash O256-PAPER-github-backup/scripts/twcc/install_twcc_o256_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate o256-watermark
bash SET_PATHS_AND_PREFLIGHT.sh
```

If the preflight passes, start the full run in `tmux`:

```bash
tmux new -s stagee_exp53
cd /work/<user>/twcc_stagee_exp53_upload
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate o256-watermark
bash SET_PATHS_AND_PREFLIGHT.sh
cd "$TWCC_O256_ROOT"
bash scripts/twcc/run_twcc_exp53_full_from_scratch_stageE_seed61.sh 2>&1 | tee "$TWCC_EXP_ROOT/twcc_exp53_full_run.log"
```

Detach from tmux with `Ctrl-b`, then `d`.
EOF

echo "Create archive..."
tar -C "$(dirname "$BUNDLE_ROOT")" -czf "$ARCHIVE_PATH" "$(basename "$BUNDLE_ROOT")"

echo
echo "Bundle folder: $BUNDLE_ROOT"
du -sh "$BUNDLE_ROOT"
echo "Archive: $ARCHIVE_PATH"
du -sh "$ARCHIVE_PATH"
