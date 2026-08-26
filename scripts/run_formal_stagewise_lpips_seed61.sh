#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
PIP="/home/jeffrey/miniconda3/envs/diffw/bin/pip"

GT_DIR="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"
OUT_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/02_ablation/formal_stagewise_lpips_seed61"
SUMMARY="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/02_ablation/formal_stagewise_lpips_seed61/summary.md"

declare -A RESULT_DIRS=(
  ["stageA"]="$ROOT/results/refinement/m06_formal_seed61_stageA_finaltest1000_apply/refined_images"
  ["stageB"]="$ROOT/results/refinement/m06_formal_seed61_stageB_finaltest1000_apply/refined_images"
  ["stageBP"]="$ROOT/results/refinement/m06_formal_seed61_stageBP_finaltest1000_apply/refined_images"
)

CHECK_ONLY="false"
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY="true"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check-only]"
  exit 2
fi

required=(
  "$PY"
  "$PIP"
  "$GT_DIR"
  "$ROOT/scripts/calculate_lpips_dir.py"
  "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/02_ablation/formal_stagewise_seed61/summary.md"
  "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_lpips_seed61/o256_base/summary.json"
  "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_lpips_seed61/o256_rgbrefine/summary.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done
for key in "${!RESULT_DIRS[@]}"; do
  if [[ ! -d "${RESULT_DIRS[$key]}" ]]; then
    echo "Missing stage output directory: ${RESULT_DIRS[$key]}"
    exit 1
  fi
done

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Formal stage-wise LPIPS preflight passed."
  echo "Output root: $OUT_ROOT"
  exit 0
fi

mkdir -p "$OUT_ROOT"

echo "Step 0/4: ensure lpips package"
"$PIP" install lpips >/dev/null

run_stage() {
  local key="$1"
  local output_dir="$OUT_ROOT/$key"
  echo "Step: LPIPS for $key"
  "$PY" "$ROOT/scripts/calculate_lpips_dir.py" \
    --gt_dir "$GT_DIR" \
    --results_dir "${RESULT_DIRS[$key]}" \
    --output_dir "$output_dir" \
    --batch_size 8 \
    --net alex \
    --device auto
}

run_stage stageA
run_stage stageB
run_stage stageBP

echo "Step 4/4: summarize stage-wise LPIPS"
"$PY" - <<'PY'
import json
from pathlib import Path

ablation_summary = Path("/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/02_ablation/formal_stagewise_seed61/summary.md").read_text(encoding="utf-8")
main_base = json.loads(Path("/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_lpips_seed61/o256_base/summary.json").read_text(encoding="utf-8"))
main_full = json.loads(Path("/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_lpips_seed61/o256_rgbrefine/summary.json").read_text(encoding="utf-8"))

def parse_row(name: str):
    for line in ablation_summary.splitlines():
        if line.startswith(f"| {name} |"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            return {
                "variant": parts[0],
                "psnr": parts[1],
                "ssim": parts[2],
                "clean_ba": parts[3],
                "avg_ba": parts[4],
                "min_ba": parts[5],
            }
    raise RuntimeError(f"Row not found: {name}")

lpips_root = Path("/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/02_ablation/formal_stagewise_lpips_seed61")
stage_lpips = {
    "Stage A": json.loads((lpips_root / "stageA" / "summary.json").read_text(encoding="utf-8")),
    "Stage B": json.loads((lpips_root / "stageB" / "summary.json").read_text(encoding="utf-8")),
    "Stage B+": json.loads((lpips_root / "stageBP" / "summary.json").read_text(encoding="utf-8")),
}

rows = []
base = parse_row("O256 base")
rows.append(("O256 base", base["psnr"], base["ssim"], f'{main_base["mean_lpips"]:.4f}', base["clean_ba"], base["avg_ba"], base["min_ba"], "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_lpips_seed61/o256_base/summary.json"))
for name in ("Stage A", "Stage B", "Stage B+"):
    row = parse_row(name)
    rows.append((name, row["psnr"], row["ssim"], f'{stage_lpips[name]["mean_lpips"]:.4f}', row["clean_ba"], row["avg_ba"], row["min_ba"], str(lpips_root / ({"Stage A":"stageA","Stage B":"stageB","Stage B+":"stageBP"}[name]) / "summary.json")))
full = parse_row("Full module")
rows.append(("Full module", full["psnr"], full["ssim"], f'{main_full["mean_lpips"]:.4f}', full["clean_ba"], full["avg_ba"], full["min_ba"], "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_lpips_seed61/o256_rgbrefine/summary.json"))

lines = [
    "# Formal Stage-wise LPIPS Summary (seed61)",
    "",
    "- All variants use the same locked `finaltest1000` split.",
    "- Reference images: `/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt`",
    "- LPIPS setting: AlexNet backbone (`net=alex`), same script and batch size 8.",
    "",
    "| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Clean BA ↑ | Nongeom Avg BA ↑ | Minimum BA ↑ | Source |",
    "| ------- | -----: | -----: | ------: | ---------: | ---------------: | -----------: | ------ |",
]
for r in rows:
    lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | `{r[7]}` |")

out = lpips_root / "summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

echo "Done. Summary: $SUMMARY"
