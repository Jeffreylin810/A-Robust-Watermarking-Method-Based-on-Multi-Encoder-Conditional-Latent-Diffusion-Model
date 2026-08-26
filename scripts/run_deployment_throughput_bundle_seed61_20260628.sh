#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
MIN_FREE_GB=12

available_gb() {
  df -Pk "$ROOT" | awk 'NR==2 {printf "%.2f", $4/1024/1024}'
}

assert_free_space() {
  local free_gb
  free_gb="$(available_gb)"
  awk -v free="$free_gb" -v need="$MIN_FREE_GB" 'BEGIN {exit !(free+0 >= need+0)}' || {
    echo "Insufficient free space for overnight bundle: ${free_gb}G available, require at least ${MIN_FREE_GB}G."
    exit 1
  }
}

cd "$ROOT"
assert_free_space

echo "[$(date '+%F %T')] Overnight bundle start"
echo "Free disk: $(available_gb)G"

bash scripts/run_actual10k_end_to_end_deployment_benchmark_seed61_20260628.sh
bash scripts/run_end_to_end_repeatability_sanity_seed61_20260628.sh

echo "[$(date '+%F %T')] Overnight bundle complete"
echo "Actual 10k summary:"
echo "  /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_actual10k_end_to_end_deployment_benchmark_20260628/actual10k_end_to_end_deployment_summary.md"
echo "Repeatability summary:"
echo "  /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/14_supplementary_end_to_end_repeatability_sanity_20260628/repeatability_summary.md"
