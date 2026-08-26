# Non-COCO Domain Shift v3 With DiffW-like Commands

Purpose: upgrade the current `non_coco_domain_shift_check_seed61_v2_extended` experiment into a more complete **cross-method supplementary comparison** by adding **DiffW-like** results on the same three non-COCO subsets, without overwriting the existing v2 O256-only results.

Output root:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v3_with_diffw_like
```

Datasets:

| Dataset | Count | Role |
|---|---:|---|
| NaturalMix_DIV2K_BSDS_eval1000 | 1000 | larger non-COCO natural-image generalization check |
| Kodak_eval24 | 24 | classic image-quality sanity check |
| ABO_demo_renders_eval273 | 273 | rendered-object / CG-like stress check |

Notes:

- This v3 command file only **adds DiffW-like** rows.
- It reuses the already prepared v2 dataset roots.
- It does **not** retrain any checkpoint.
- It does **not** overwrite the existing v2 O256 base / O256-RGB results.
- It uses the **formal 11 non-geometric `Paper_*` attacks** via `attack_eval_diffw_paper_matched.py`.

## 1. Export Variables And Create Output Root

```bash
export PY=/home/jeffrey/miniconda3/envs/diffw/bin/python
export DIFFW_ROOT=/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline
export O256_ROOT=/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
export EXP_ROOT=/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v3_with_diffw_like
export SOURCE_ROOT=/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/06_domain_shift/non_coco_domain_shift_check_seed61_v2_extended/dataset_roots
export BASE_CONFIG=/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml
export DIFFW_CKPT=/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar

mkdir -p "$EXP_ROOT"/{logs,lpips,attacked_Result,summaries}
```

## 2. Optional Preflight Check

```bash
test -f "$DIFFW_CKPT" && echo "DiffW-like checkpoint OK" || echo "DiffW-like checkpoint MISSING"
test -d "$SOURCE_ROOT/NaturalMix_DIV2K_BSDS_eval1000" && echo "NaturalMix root OK" || echo "NaturalMix root MISSING"
test -d "$SOURCE_ROOT/Kodak_eval24" && echo "Kodak root OK" || echo "Kodak root MISSING"
test -d "$SOURCE_ROOT/ABO_demo_renders_eval273" && echo "ABO root OK" || echo "ABO root MISSING"
```

## 3. Full DiffW-like Supplement Run

This block will:

1. generate dataset-specific DiffW-like eval configs;
2. run DiffW-like eval on each domain-shift subset;
3. compute PSNR/SSIM;
4. compute LPIPS;
5. compute Clean BA;
6. run the formal 11 non-geometric `Paper_*` attacks;
7. compute attacked BA.

```bash
for NAME in NaturalMix_DIV2K_BSDS_eval1000 Kodak_eval24 ABO_demo_renders_eval273; do
  DATASET_ROOT="$SOURCE_ROOT/$NAME"
  GT_DIR="$DATASET_ROOT/data/CoCo/val/gt"
  CONFIG_NAME="CoCo_non_coco_v3_${NAME}_diffw_seed61.yml"
  CONFIG_PATH="$DIFFW_ROOT/configs/$CONFIG_NAME"
  RUN_NAME="non_coco_v3_${NAME}_diffw_like"
  RESULTS_DIR="$DIFFW_ROOT/results/test_result/$RUN_NAME/CoCo/CoCo"
  CLEAN_ROOT="$EXP_ROOT/attacked_Result/non_coco_v3_${NAME}_diffw_like_clean"
  CLEAN_DIR="$CLEAN_ROOT/Clean"
  ATTACK_RUN="non_coco_v3_${NAME}_diffw_like_nongeom"
  ATTACK_DIR="$EXP_ROOT/attacked_Result/$ATTACK_RUN"
  LPIPS_DIR="$EXP_ROOT/lpips/$NAME/diffw_like"

  python - <<PYCFG
from pathlib import Path
import yaml
base = yaml.safe_load(Path("$BASE_CONFIG").read_text())
base["data"]["dataset_root"] = "$DATASET_ROOT"
Path("$CONFIG_PATH").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
print("wrote config:", "$CONFIG_PATH")
PYCFG

  (
    cd "$DIFFW_ROOT"
    "$PY" eval_diffusion.py \
      --config "$CONFIG_NAME" \
      --resume "$DIFFW_CKPT" \
      --test_set CoCo \
      --image_folder results/test_result \
      --run_name "$RUN_NAME" \
      --seed 61 \
      2>&1 | tee "$EXP_ROOT/logs/${NAME}_diffw_eval.log"
  )

  (
    cd "$DIFFW_ROOT"
    "$PY" calculate_psnr_ssim.py \
      --gt_path "$GT_DIR" \
      --results_path "$RESULTS_DIR" \
      2>&1 | tee "$EXP_ROOT/logs/${NAME}_diffw_psnr_ssim.log"
  )

  "$PY" "$O256_ROOT/scripts/calculate_lpips_dir.py" \
    --gt_dir "$GT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$LPIPS_DIR" \
    --batch_size 8 \
    --net alex \
    2>&1 | tee "$EXP_ROOT/logs/${NAME}_diffw_lpips.log"

  mkdir -p "$CLEAN_DIR"
  find "$CLEAN_DIR" -maxdepth 1 -type l -delete
  while IFS= read -r f; do
    ln -sfn "$f" "$CLEAN_DIR/$(basename "$f")"
  done < <(find "$RESULTS_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) | sort)

  (
    cd "$DIFFW_ROOT"
    "$PY" calculate_ba.py \
      --config "$CONFIG_NAME" \
      --resume "$DIFFW_CKPT" \
      --attack_dir "$CLEAN_ROOT" \
      --gt_message_dir "$RESULTS_DIR" \
      2>&1 | tee "$EXP_ROOT/logs/${NAME}_diffw_clean_ba.log"
  )

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
      2>&1 | tee "$EXP_ROOT/logs/${NAME}_diffw_attack.log"
  )

  (
    cd "$DIFFW_ROOT"
    "$PY" calculate_ba.py \
      --config "$CONFIG_NAME" \
      --resume "$DIFFW_CKPT" \
      --attack_dir "$ATTACK_DIR" \
      --gt_message_dir "$RESULTS_DIR" \
      2>&1 | tee "$EXP_ROOT/logs/${NAME}_diffw_attack_ba.log"
  )
done
```

## 4. Check Progress

```bash
tail -n 40 "$EXP_ROOT"/logs/*.log
```

```bash
ps -eo pid,etime,cmd | grep -E 'eval_diffusion|calculate_psnr_ssim|calculate_lpips_dir|calculate_ba|attack_eval_diffw_paper_matched' | grep -v grep
```

## 5. Expected DiffW-like Outputs

```text
$EXP_ROOT/logs/NaturalMix_DIV2K_BSDS_eval1000_diffw_eval.log
$EXP_ROOT/logs/Kodak_eval24_diffw_eval.log
$EXP_ROOT/logs/ABO_demo_renders_eval273_diffw_eval.log

$EXP_ROOT/lpips/NaturalMix_DIV2K_BSDS_eval1000/diffw_like/summary.md
$EXP_ROOT/lpips/Kodak_eval24/diffw_like/summary.md
$EXP_ROOT/lpips/ABO_demo_renders_eval273/diffw_like/summary.md

$EXP_ROOT/attacked_Result/non_coco_v3_NaturalMix_DIV2K_BSDS_eval1000_diffw_like_nongeom/ba_report.txt
$EXP_ROOT/attacked_Result/non_coco_v3_Kodak_eval24_diffw_like_nongeom/ba_report.txt
$EXP_ROOT/attacked_Result/non_coco_v3_ABO_demo_renders_eval273_diffw_like_nongeom/ba_report.txt
```

## 6. Interpretation Guardrail

- Even after adding DiffW-like, this experiment should still be described as **supplementary domain-shift evidence**, not a replacement for the locked COCO finaltest1000 main protocol.
- The v3 addition mainly addresses **cross-method completeness** for the non-COCO check.
- If later you want a single merged CSV/MD table with `DiffW-like + O256 base + O256-RGB`, create it from:
  - the existing v2 O256 summaries; and
  - the new DiffW-like logs/LPIPS/BA outputs generated here.
