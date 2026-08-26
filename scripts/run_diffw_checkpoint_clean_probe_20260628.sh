#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"          # sf0p5 | sf0p75 | both
CKPT_KIND="${2:-prefinal}" # prefinal | final

PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
O256_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/11_supplementary_diffw_checkpoint_clean_probe_20260628"
GT_DIR="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"
EVAL_BASE_CONFIG="$DIFFW_ROOT/configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"

mkdir -p "$EXP_ROOT"/{logs,eval_outputs,lpips,attacked_Result}

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

resolve_ckpt_file() {
  case "$CKPT_KIND" in
    prefinal) echo "CoCo_ddpm.pth.tar" ;;
    final) echo "CoCo_ddpm_final.pth.tar" ;;
    *)
      echo "Unsupported CKPT_KIND: $CKPT_KIND" >&2
      exit 1
      ;;
  esac
}

run_probe() {
  local tag="$1"
  local sf="$2"
  local source_root="$3"
  local ckpt_file
  ckpt_file="$(resolve_ckpt_file)"
  local ckpt_path="$source_root/workspace/ckpts/$ckpt_file"

  if [[ ! -f "$ckpt_path" ]]; then
    echo "Missing checkpoint: $ckpt_path" >&2
    exit 1
  fi

  local config_name="CoCo_finaltest1000_locked_linear_eval_seed61_${tag}_${CKPT_KIND}_cleanprobe_20260628.yml"
  local config_path="$DIFFW_ROOT/configs/$config_name"
  local run_name="${tag}_${CKPT_KIND}_cleanprobe"
  local results_dir="$EXP_ROOT/eval_outputs/$run_name/CoCo/CoCo"
  local lpips_dir="$EXP_ROOT/lpips/$run_name"
  local clean_root="$EXP_ROOT/attacked_Result/${run_name}_clean"

  log "Writing eval config for $tag ($CKPT_KIND)"
  "$PY" - <<PY
from pathlib import Path
import yaml
payload = yaml.safe_load(Path("$EVAL_BASE_CONFIG").read_text())
payload["watermarking"]["strength_factor"] = float("$sf")
payload["data"]["data_dir"] = "$EXP_ROOT"
Path("$config_path").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print("wrote", "$config_path")
PY

  local msg_count=0
  if [[ -d "$results_dir" ]]; then
    msg_count="$(find "$results_dir" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
  fi
  if [[ "$msg_count" -ne 1000 ]]; then
    log "Eval $tag ($CKPT_KIND)"
    (
      cd "$DIFFW_ROOT"
      "$PY" -u eval_diffusion.py \
        --config "$config_name" \
        --resume "$ckpt_path" \
        --test_set CoCo \
        --image_folder "$EXP_ROOT/eval_outputs" \
        --run_name "$run_name" \
        --seed 61 \
        2>&1 | tee "$EXP_ROOT/logs/${run_name}_eval.log"
    )
  else
    log "Skip eval for $run_name"
  fi

  if [[ ! -f "$EXP_ROOT/logs/${run_name}_psnr_ssim.log" ]]; then
    log "PSNR/SSIM for $run_name"
    (
      cd "$DIFFW_ROOT"
      "$PY" calculate_psnr_ssim.py \
        --gt_path "$GT_DIR" \
        --results_path "$results_dir" \
        2>&1 | tee "$EXP_ROOT/logs/${run_name}_psnr_ssim.log"
    )
  fi

  if [[ ! -f "$lpips_dir/summary.json" ]]; then
    log "LPIPS for $run_name"
    (
      cd "$O256_ROOT"
      "$PY" scripts/calculate_lpips_dir.py \
        --gt_dir "$GT_DIR" \
        --results_dir "$results_dir" \
        --output_dir "$lpips_dir" \
        --batch_size 8 \
        --net alex \
        2>&1 | tee "$EXP_ROOT/logs/${run_name}_lpips.log"
    )
  fi

  mkdir -p "$clean_root"
  rm -rf "$clean_root/Clean"
  ln -sfn "$results_dir" "$clean_root/Clean"

  if [[ ! -f "$clean_root/ba_report.txt" ]]; then
    log "Clean BA for $run_name"
    (
      cd "$DIFFW_ROOT"
      "$PY" calculate_ba.py \
        --config "$config_name" \
        --resume "$ckpt_path" \
        --attack_dir "$clean_root" \
        --gt_message_dir "$results_dir" \
        2>&1 | tee "$EXP_ROOT/logs/${run_name}_clean_ba.log"
    )
  fi
}

build_summary() {
  log "Building checkpoint clean probe summary"
  "$PY" - <<PY
from pathlib import Path
import csv, json, re

root = Path("$EXP_ROOT")
rows = []
for tag, sf in [("sf0p5", "0.5"), ("sf0p75", "0.75")]:
    run_name = f"{tag}_${CKPT_KIND}_cleanprobe"
    psnr_log = root / "logs" / f"{run_name}_psnr_ssim.log"
    lpips_json = root / "lpips" / run_name / "summary.json"
    clean_report = root / "attacked_Result" / f"{run_name}_clean" / "ba_report.txt"
    if not (psnr_log.exists() and lpips_json.exists() and clean_report.exists()):
        continue

    psnr_text = psnr_log.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Testing set, PSNR is ([0-9.]+) and SSIM is ([0-9.]+)", psnr_text)
    if not m:
        raise RuntimeError(f"Could not parse PSNR/SSIM from {psnr_log}")
    psnr = float(m.group(1))
    ssim = float(m.group(2))

    lpips = json.loads(lpips_json.read_text())["mean_lpips"]
    clean_text = clean_report.read_text(encoding="utf-8", errors="ignore")
    m2 = re.search(r"Clean\\s*\\|\\s*([0-9.]+)", clean_text)
    if not m2:
        raise RuntimeError(f"Could not parse Clean BA from {clean_report}")
    clean_ba = float(m2.group(1))

    rows.append({
        "variant": tag,
        "checkpoint_kind": "$CKPT_KIND",
        "strength_factor": sf,
        "psnr": f"{psnr:.4f}",
        "ssim": f"{ssim:.4f}",
        "lpips": f"{lpips:.6f}",
        "clean_ba": f"{clean_ba:.2f}",
    })

    out = root / f"{run_name}_summary.txt"
    out.write_text(
        "\\n".join([
            f"variant: {tag}",
            f"checkpoint_kind: $CKPT_KIND",
            f"strength_factor: {sf}",
            f"psnr: {psnr:.4f}",
            f"ssim: {ssim:.4f}",
            f"lpips: {lpips:.6f}",
            f"clean_ba: {clean_ba:.2f}",
        ]) + "\\n",
        encoding="utf-8",
    )

summary_csv = root / f"checkpoint_clean_probe_{CKPT_KIND}_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "checkpoint_kind", "strength_factor", "psnr", "ssim", "lpips", "clean_ba"])
    writer.writeheader()
    writer.writerows(rows)
print("saved", summary_csv)
PY
}

case "$MODE" in
  sf0p5)
    run_probe "sf0p5" "0.5" "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/09_supplementary_diffw_actual_train_strength0p5_20260628"
    ;;
  sf0p75)
    run_probe "sf0p75" "0.75" "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/10_supplementary_diffw_actual_train_strength0p75_20260628"
    ;;
  both)
    run_probe "sf0p5" "0.5" "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/09_supplementary_diffw_actual_train_strength0p5_20260628"
    run_probe "sf0p75" "0.75" "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/10_supplementary_diffw_actual_train_strength0p75_20260628"
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    echo "Usage: bash scripts/run_diffw_checkpoint_clean_probe_20260628.sh [sf0p5|sf0p75|both] [prefinal|final]" >&2
    exit 1
    ;;
esac

build_summary
log "Done."
