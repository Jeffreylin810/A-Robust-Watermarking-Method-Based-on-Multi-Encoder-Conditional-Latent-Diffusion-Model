#!/usr/bin/env bash
set -euo pipefail

PY="/home/jeffrey/miniconda3/envs/diffw/bin/python"
O256_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_ROOT="/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
EXP_ROOT="/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_diffw_actual_train_attackpressure_epoch7_truebranch_20260628"
WORKSPACE="$EXP_ROOT/workspace"
GT_DIR="/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt"
TRAIN_BASE_CONFIG="$DIFFW_ROOT/configs/CoCo_train10000_finaltest1000_locked_linear_steps20k_seed61_formal.yml"
EVAL_BASE_CONFIG="$DIFFW_ROOT/configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml"

TRAIN_CONFIG_NAME="CoCo_train10000_finaltest1000_locked_linear_epoch7_truebranch_seed61_20260628.yml"
EVAL_CONFIG_NAME="CoCo_finaltest1000_locked_linear_epoch7_truebranch_seed61_20260628.yml"
TRAIN_CONFIG_PATH="$DIFFW_ROOT/configs/$TRAIN_CONFIG_NAME"
EVAL_CONFIG_PATH="$DIFFW_ROOT/configs/$EVAL_CONFIG_NAME"

FINAL_CKPT="$WORKSPACE/ckpts/CoCo_ddpm_final.pth.tar"
OUTPUT_ROOT="$EXP_ROOT/eval_outputs"
RUN_NAME="finaltest1000"
RESULTS_DIR="$OUTPUT_ROOT/$RUN_NAME/CoCo/CoCo"
LPIPS_DIR="$EXP_ROOT/lpips"
CLEAN_ROOT="$EXP_ROOT/attacked_Result/clean_diffw_attackpressure_epoch7_truebranch_finaltest1000"
ATTACK_RUN="diffw_attackpressure_epoch7_truebranch_finaltest1000_nongeom"
ATTACK_DIR="$EXP_ROOT/attacked_Result/$ATTACK_RUN"

mkdir -p "$EXP_ROOT"/{logs,attacked_Result,eval_outputs,lpips}
mkdir -p "$WORKSPACE"/{ckpts,results}

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

write_configs() {
  log "Writing epoch7 true-branch configs (combined_attack=False, n_epochs=7, sf=1.0)"
  "$PY" - <<PY
from pathlib import Path
import yaml

train_base = yaml.safe_load(Path("$TRAIN_BASE_CONFIG").read_text())
eval_base = yaml.safe_load(Path("$EVAL_BASE_CONFIG").read_text())

train_base["watermarking"]["strength_factor"] = 1.0
train_base["watermarking"]["combined_attack"] = False
train_base["training"]["n_epochs"] = 7
train_base["data"]["data_dir"] = "$WORKSPACE"

eval_base["watermarking"]["strength_factor"] = 1.0
eval_base["watermarking"]["combined_attack"] = False
eval_base["data"]["data_dir"] = "$WORKSPACE"

Path("$TRAIN_CONFIG_PATH").write_text(yaml.safe_dump(train_base, sort_keys=False), encoding="utf-8")
Path("$EVAL_CONFIG_PATH").write_text(yaml.safe_dump(eval_base, sort_keys=False), encoding="utf-8")
print("wrote", "$TRAIN_CONFIG_PATH")
print("wrote", "$EVAL_CONFIG_PATH")
PY
}

run_train() {
  if [[ -f "$FINAL_CKPT" ]]; then
    log "Skip training, checkpoint already exists: $FINAL_CKPT"
    return 0
  fi

  log "Start actual training: DiffW-like epoch7 true-branch test"
  (
    cd "$DIFFW_ROOT"
    /usr/bin/time -p "$PY" -u train_diffusion.py \
      --config "$TRAIN_CONFIG_NAME" \
      --resume ./ckpts/nonexistent_diffw_attackpressure_epoch7_truebranch_seed61_start.pth.tar \
      --image_folder "$EXP_ROOT/validation/attackpressure_epoch7_truebranch_train" \
      --seed 61 \
      2>&1 | tee "$EXP_ROOT/logs/train_attackpressure_epoch7_truebranch.log"
  )

  if [[ ! -f "$FINAL_CKPT" ]]; then
    echo "Missing final checkpoint: $FINAL_CKPT" >&2
    exit 1
  fi
}

run_eval() {
  local msg_count=0
  if [[ -d "$RESULTS_DIR" ]]; then
    msg_count="$(find "$RESULTS_DIR" -maxdepth 1 -type f -name '*_message.npy' | wc -l)"
  fi
  if [[ "$msg_count" -eq 1000 ]]; then
    log "Skip eval, outputs already complete"
    return 0
  fi

  log "Eval epoch7 true-branch checkpoint on locked finaltest1000"
  (
    cd "$DIFFW_ROOT"
    "$PY" -u eval_diffusion.py \
      --config "$EVAL_CONFIG_NAME" \
      --resume "$FINAL_CKPT" \
      --test_set CoCo \
      --image_folder "$OUTPUT_ROOT" \
      --run_name "$RUN_NAME" \
      --seed 61 \
      2>&1 | tee "$EXP_ROOT/logs/eval_attackpressure_epoch7_truebranch.log"
  )
}

run_psnr_lpips() {
  if [[ ! -f "$EXP_ROOT/logs/psnr_ssim_attackpressure_epoch7_truebranch.log" ]]; then
    log "Compute PSNR/SSIM"
    (
      cd "$DIFFW_ROOT"
      "$PY" calculate_psnr_ssim.py \
        --gt_path "$GT_DIR" \
        --results_path "$RESULTS_DIR" \
        2>&1 | tee "$EXP_ROOT/logs/psnr_ssim_attackpressure_epoch7_truebranch.log"
    )
  fi

  if [[ ! -f "$LPIPS_DIR/summary.json" ]]; then
    log "Compute LPIPS"
    (
      cd "$O256_ROOT"
      "$PY" scripts/calculate_lpips_dir.py \
        --gt_dir "$GT_DIR" \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$LPIPS_DIR" \
        --batch_size 8 \
        --net alex \
        2>&1 | tee "$EXP_ROOT/logs/lpips_attackpressure_epoch7_truebranch.log"
    )
  fi
}

run_clean_ba() {
  mkdir -p "$CLEAN_ROOT"
  rm -rf "$CLEAN_ROOT/Clean"
  ln -sfn "$RESULTS_DIR" "$CLEAN_ROOT/Clean"

  if [[ -f "$CLEAN_ROOT/ba_report.txt" ]]; then
    log "Skip clean BA"
    return 0
  fi

  log "Compute Clean BA"
  (
    cd "$DIFFW_ROOT"
    "$PY" calculate_ba.py \
      --config "$EVAL_CONFIG_NAME" \
      --resume "$FINAL_CKPT" \
      --attack_dir "$CLEAN_ROOT" \
      --gt_message_dir "$RESULTS_DIR" \
      2>&1 | tee "$EXP_ROOT/logs/clean_ba_attackpressure_epoch7_truebranch.log"
  )
}

run_attacks_and_ba() {
  if [[ ! -f "$ATTACK_DIR/metrics_report.txt" ]]; then
    log "Generate 11 non-geometric attacks"
    (
      cd "$O256_ROOT"
      "$PY" attack_eval_diffw_paper_matched.py \
        --gt_path "$GT_DIR" \
        --results_path "$RESULTS_DIR" \
        --output_dir "$EXP_ROOT/attacked_Result" \
        --run_name "$ATTACK_RUN" \
        --attack_scope nongeom \
        --seed 61 \
        --overwrite \
        2>&1 | tee "$EXP_ROOT/logs/attack_attackpressure_epoch7_truebranch.log"
    )
  fi

  if [[ -f "$ATTACK_DIR/ba_report.txt" ]]; then
    log "Skip attacked BA"
    return 0
  fi

  log "Compute attacked BA"
  (
    cd "$DIFFW_ROOT"
    "$PY" calculate_ba.py \
      --config "$EVAL_CONFIG_NAME" \
      --resume "$FINAL_CKPT" \
      --attack_dir "$ATTACK_DIR" \
      --gt_message_dir "$RESULTS_DIR" \
      2>&1 | tee "$EXP_ROOT/logs/attack_ba_attackpressure_epoch7_truebranch.log"
  )
}

write_configs
run_train
run_eval
run_psnr_lpips
run_clean_ba
run_attacks_and_ba

log "Done."
log "Checkpoint: $FINAL_CKPT"
log "Results dir: $RESULTS_DIR"
log "Clean BA: $CLEAN_ROOT/ba_report.txt"
log "Attack BA: $ATTACK_DIR/ba_report.txt"
