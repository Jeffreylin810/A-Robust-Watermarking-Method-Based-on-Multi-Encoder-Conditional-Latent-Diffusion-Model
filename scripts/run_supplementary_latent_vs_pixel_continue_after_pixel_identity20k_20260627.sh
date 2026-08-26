#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PIPELINE_SCRIPT="$ROOT/scripts/run_supplementary_latent_vs_pixel_pipeline_20260627.sh"

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "Missing pipeline script: $PIPELINE_SCRIPT"
  exit 1
fi

echo "[continue-after-pixel-identity20k] This wrapper assumes pixel-bitplane identity20k is already running or already completed."
echo "[continue-after-pixel-identity20k] It will wait for that checkpoint, then continue the remaining latent-vs-pixel steps."

cd "$ROOT"
bash "$PIPELINE_SCRIPT"
