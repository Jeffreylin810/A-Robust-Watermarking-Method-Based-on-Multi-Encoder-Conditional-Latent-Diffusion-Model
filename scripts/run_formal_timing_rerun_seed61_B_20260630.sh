#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"

OUT="${OUT:-/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_timing_rerun_seed61_B_20260630}"
RAW_DIR="$OUT/raw"
LOG_DIR="$OUT/logs"
SUMMARY_MD="$OUT/summary.md"
SUMMARY_CSV="$OUT/summary.csv"

DIFFW_CONFIG="configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"
DIFFW_CKPT="$DIFFW_ROOT/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar"
DIFFW_JSON="$RAW_DIR/diffw_like.json"

O256_CONFIG="configs/CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml"
O256_CKPT="Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar"
O256_REFINER="results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth"
O256_JSON="$RAW_DIR/o256_rgb.json"

CHECK_ONLY=false
OVERWRITE=false

for arg in "$@"; do
  case "$arg" in
    --check-only)
      CHECK_ONLY=true
      ;;
    --overwrite)
      OVERWRITE=true
      ;;
    *)
      echo "Usage: $0 [--check-only] [--overwrite]"
      exit 2
      ;;
  esac
done

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required input: $path"; exit 1; }
}

gpu_processes() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
}

summarize() {
  "$PY" - "$DIFFW_JSON" "$O256_JSON" "$SUMMARY_MD" "$SUMMARY_CSV" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

diffw_path = Path(sys.argv[1])
o256_path = Path(sys.argv[2])
md_path = Path(sys.argv[3])
csv_path = Path(sys.argv[4])

def ci95(std, repeats):
    return 1.96 * std / math.sqrt(repeats) if repeats else 0.0

def load_rows(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    repeats = int(payload["repeats"])
    for row in payload["results"]:
        runs = row["run_ms_per_image"]
        rows.append({
            "method_name": payload["method_name"],
            "batch_size": row["batch_size"],
            "mode": row["mode"],
            "mean_ms_per_image": row["mean_ms_per_image"],
            "std_ms_per_image": row["std_ms_per_image"],
            "ci95_half_width": ci95(row["std_ms_per_image"], repeats),
            "min_run": min(runs),
            "max_run": max(runs),
            "cv_percent": row["cv_percent"],
            "images_per_second": row["images_per_second"],
            "gpu": payload.get("gpu", ""),
            "warmup": payload["warmup"],
            "iterations": payload["iterations"],
            "refiner_iterations": payload["refiner_iterations"],
            "repeats": payload["repeats"],
            "sampling_timesteps": payload["sampling_timesteps"],
            "timing_scope": payload["timing_scope"],
        })
    return rows

rows = []
rows.extend(load_rows(diffw_path))
rows.extend(load_rows(o256_path))

table_rows = []
for r in rows:
    if "DiffW-like" in r["method_name"]:
        method = "DiffW-like"
    elif r["mode"] == "base":
        method = "O256-Base"
    elif r["mode"] == "base_plus_refiner":
        method = "O256-RGB"
    elif r["mode"] == "refiner_only":
        method = "RGB refinement only"
    else:
        method = r["method_name"]
    table_rows.append({**r, "method": method})

csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "method_name",
            "batch_size",
            "mode",
            "mean_ms_per_image",
            "std_ms_per_image",
            "ci95_half_width",
            "min_run",
            "max_run",
            "cv_percent",
            "images_per_second",
            "gpu",
            "warmup",
            "iterations",
            "refiner_iterations",
            "repeats",
            "sampling_timesteps",
            "timing_scope",
        ],
    )
    writer.writeheader()
    writer.writerows(table_rows)

def get(method, batch):
    for r in table_rows:
        if r["method"] == method and r["batch_size"] == batch:
            return r
    return None

lines = [
    "# Formal Timing Rerun Seed61 B",
    "",
    "## Scope",
    "",
    "- Purpose: repeated compute-only inference timing for oral-defense table cleanup.",
    "- Dataset I/O, image writing, PSNR/SSIM/LPIPS, BA, attacks, and training are excluded.",
    "- Protocol: warm-up 30, iterations 200, repeats 20, sampling_timesteps 3.",
    "- All values are measured on the same machine/GPU under the same script protocol.",
    "",
    "## Main Table",
    "",
    "| Method | Batch size 1 mean | Batch size 1 CV | Batch size 4 mean | Batch size 4 CV |",
    "|---|---:|---:|---:|---:|",
]
for method in ["DiffW-like", "O256-Base", "O256-RGB"]:
    b1 = get(method, 1)
    b4 = get(method, 4)
    lines.append(
        f"| {method} | {b1['mean_ms_per_image']:.4f} ms/image | {b1['cv_percent']:.2f}% | "
        f"{b4['mean_ms_per_image']:.4f} ms/image | {b4['cv_percent']:.2f}% |"
    )

lines.extend([
    "",
    "## RGB Refinement Overhead",
    "",
    "| Batch size | O256-Base | O256-RGB | Delta | Relative overhead |",
    "|---:|---:|---:|---:|---:|",
])
for batch in [1, 4]:
    base = get("O256-Base", batch)
    full = get("O256-RGB", batch)
    delta = full["mean_ms_per_image"] - base["mean_ms_per_image"]
    rel = 100.0 * delta / base["mean_ms_per_image"]
    lines.append(
        f"| {batch} | {base['mean_ms_per_image']:.4f} | {full['mean_ms_per_image']:.4f} | "
        f"{delta:+.4f} | {rel:+.2f}% |"
    )

lines.extend([
    "",
    "## Detailed Rows",
    "",
    "| Method | Batch | Mode | Mean | Std | 95% CI half-width | Min | Max | CV |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|",
])
for r in table_rows:
    lines.append(
        f"| {r['method']} | {r['batch_size']} | {r['mode']} | "
        f"{r['mean_ms_per_image']:.4f} | {r['std_ms_per_image']:.4f} | "
        f"{r['ci95_half_width']:.4f} | {r['min_run']:.4f} | {r['max_run']:.4f} | "
        f"{r['cv_percent']:.2f}% |"
    )

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"summary md: {md_path}")
print(f"summary csv: {csv_path}")
PY
}

mkdir -p "$RAW_DIR" "$LOG_DIR"
cd "$ROOT"

require_path "$PY"
require_path "$ROOT/scripts/benchmark_exp8_inference.py"
require_path "$DIFFW_ROOT/$DIFFW_CONFIG"
require_path "$DIFFW_CKPT"
require_path "$ROOT/$O256_CONFIG"
require_path "$ROOT/$O256_CKPT"
require_path "$ROOT/$O256_REFINER"

gpu_info="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)"
active_gpu_processes="$(gpu_processes)"

if [[ "$CHECK_ONLY" == true ]]; then
  echo "Formal timing rerun B preflight passed."
  echo "Output root: $OUT"
  echo "DiffW-like JSON: $DIFFW_JSON"
  echo "O256 JSON: $O256_JSON"
  echo "Summary: $SUMMARY_MD"
  if [[ -n "$gpu_info" ]]; then
    echo "GPU: $gpu_info"
  fi
  if [[ -n "$active_gpu_processes" ]]; then
    echo "WARNING: active GPU compute processes detected:"
    echo "$active_gpu_processes"
  else
    echo "GPU compute status: idle"
  fi
  df -h "$ROOT" || true
  exit 0
fi

if [[ -n "$active_gpu_processes" ]]; then
  echo "WARNING: active GPU compute processes detected:"
  echo "$active_gpu_processes"
  echo "Timing will be cleaner if you stop other GPU jobs first."
  echo
fi

echo "[$(date '+%F %T')] Start formal timing rerun seed61 B"
echo "Output root: $OUT"

if [[ "$OVERWRITE" == true ]]; then
  rm -f "$DIFFW_JSON" "$O256_JSON" "$SUMMARY_MD" "$SUMMARY_CSV"
fi

echo
echo "[$(date '+%F %T')] Step 1/3: DiffW-like repeated timing"
if [[ -f "$DIFFW_JSON" ]]; then
  echo "Reuse existing JSON: $DIFFW_JSON"
else
  /usr/bin/time -p "$PY" scripts/benchmark_exp8_inference.py \
    --project_root "$DIFFW_ROOT" \
    --config "$DIFFW_CONFIG" \
    --resume "$DIFFW_CKPT" \
    --method_name "DiffW-like timing rerun seed61 B" \
    --output "$DIFFW_JSON" \
    --batch_sizes 1 4 \
    --sampling_timesteps 3 \
    --warmup 30 \
    --iterations 200 \
    --refiner_iterations 1000 \
    --repeats 20 \
    --seed 61 \
    2>&1 | tee "$LOG_DIR/diffw_like_timing.log"
fi

echo
echo "[$(date '+%F %T')] Step 2/3: O256-Base / O256-RGB repeated timing"
if [[ -f "$O256_JSON" ]]; then
  echo "Reuse existing JSON: $O256_JSON"
else
  /usr/bin/time -p "$PY" scripts/benchmark_exp8_inference.py \
    --project_root "$ROOT" \
    --config "$O256_CONFIG" \
    --resume "$O256_CKPT" \
    --refiner "$O256_REFINER" \
    --method_name "O256-RGB timing rerun seed61 B" \
    --output "$O256_JSON" \
    --batch_sizes 1 4 \
    --sampling_timesteps 3 \
    --warmup 30 \
    --iterations 200 \
    --refiner_iterations 1000 \
    --repeats 20 \
    --seed 61 \
    2>&1 | tee "$LOG_DIR/o256_rgb_timing.log"
fi

echo
echo "[$(date '+%F %T')] Step 3/3: summarize"
summarize

echo
echo "[$(date '+%F %T')] Done."
echo "Summary: $SUMMARY_MD"
echo "CSV: $SUMMARY_CSV"
