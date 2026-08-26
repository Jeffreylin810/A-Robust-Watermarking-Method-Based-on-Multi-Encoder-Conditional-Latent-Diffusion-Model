# DiffW-like 往 O256-RGB 門檻靠的 quality sweep 指令

目的：

- 不重訓 current formal `DiffW-like`
- 直接拿 formal checkpoint 既有輸出做 `gamma` 強度掃描
- 先快速找出哪個 `gamma` 最接近 `O256-RGB` 的 `PSNR = 32.9034`
- 再只對最接近的 1~2 個候選做 `Clean BA + 11 attacks`

這條是口試前最值得優先做的 supplementary experiment。

注意：

- 這不是重新訓練 DiffW-like
- 這是在 formal DiffW-like output 上做 post-hoc residual strength scaling
- 所以結論只能寫成 supplementary observation，不是 official retraining result

---

## 0. 固定路徑

```bash
export PY=/home/jeffrey/miniconda3/envs/diffw/bin/python
export O256_ROOT=/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
export DIFFW_ROOT=/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline
export EXP_ROOT=/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/08_supplementary_diffw_quality_tradeoff_20260627
export SOURCE_RESULTS=/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/results/test_result/m04_formal_diffw_like_seed61/finaltest1000/CoCo/CoCo
export GT_DIR=/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked/data/CoCo/val/gt
export FORMAL_EVAL_CONFIG=/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/configs/CoCo_finaltest1000_locked_linear_eval_seed61_formal.yml
export FORMAL_CKPT=/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/Result/CoCo_ddpm_m04_formal_diffw_like_train10000_steps20k_seed61_final.pth.tar

mkdir -p "$EXP_ROOT"/{logs,lpips,attacked_Result,scaled_outputs}
```

---

## 1. 先產生 gamma 掃描版本

建議先掃：

- `1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.25 0.2 0.15 0.1`

```bash
cd "$O256_ROOT"

"$PY" scripts/generate_diffw_quality_sweep_outputs.py \
  --source_results_dir "$SOURCE_RESULTS" \
  --gt_dir "$GT_DIR" \
  --output_root "$EXP_ROOT" \
  --gammas 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.25 0.2 0.15 0.1
```

---

## 2. 先做快速品質掃描（口試前最先看這一步）

這一步只跑：

- PSNR
- SSIM
- LPIPS

不先跑 BA，目的是快速找出最接近 `PSNR 32.9034` 的候選 gamma。

```bash
for TAG in gamma_1p00 gamma_0p90 gamma_0p80 gamma_0p70 gamma_0p60 gamma_0p50 gamma_0p40 gamma_0p30 gamma_0p25 gamma_0p20 gamma_0p15 gamma_0p10; do
  RESULTS_DIR="$EXP_ROOT/scaled_outputs/$TAG/CoCo/CoCo"

  "$PY" "$DIFFW_ROOT/calculate_psnr_ssim.py" \
    --gt_path "$GT_DIR" \
    --results_path "$RESULTS_DIR" \
    2>&1 | tee "$EXP_ROOT/logs/${TAG}_psnr_ssim.log"

  "$PY" "$O256_ROOT/scripts/calculate_lpips_dir.py" \
    --gt_dir "$GT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$EXP_ROOT/lpips/$TAG" \
    --batch_size 8 \
    --net alex \
    2>&1 | tee "$EXP_ROOT/logs/${TAG}_lpips.log"
done
```

跑完後先彙總：

```bash
"$PY" "$O256_ROOT/scripts/summarize_diffw_quality_tradeoff.py" \
  --exp_root "$EXP_ROOT"
```

看這個：

```bash
column -s, -t < "$EXP_ROOT/diffw_quality_sweep_summary.csv" | less -S
```

---

## 3. 只對最接近 O256-RGB 門檻的 1~2 個 gamma 跑 BA

假設快速掃描後，你選：

- `gamma_0p30`
- `gamma_0p25`

請把下面的 `TARGET_TAGS` 換成你實際選的候選。

```bash
TARGET_TAGS="gamma_0p30 gamma_0p25"

for TAG in $TARGET_TAGS; do
  RESULTS_DIR="$EXP_ROOT/scaled_outputs/$TAG/CoCo/CoCo"
  CLEAN_ROOT="$EXP_ROOT/attacked_Result/${TAG}_clean"
  CLEAN_DIR="$CLEAN_ROOT/Clean"
  ATTACK_RUN="${TAG}_nongeom"
  ATTACK_DIR="$EXP_ROOT/attacked_Result/$ATTACK_RUN"

  mkdir -p "$CLEAN_ROOT"
  rm -rf "$CLEAN_DIR"
  ln -sfn "$RESULTS_DIR" "$CLEAN_DIR"

  "$PY" "$DIFFW_ROOT/calculate_ba.py" \
    --config "$FORMAL_EVAL_CONFIG" \
    --resume "$FORMAL_CKPT" \
    --attack_dir "$CLEAN_ROOT" \
    --gt_message_dir "$RESULTS_DIR" \
    2>&1 | tee "$EXP_ROOT/logs/${TAG}_clean_ba.log"

  "$PY" "$O256_ROOT/attack_eval_diffw_paper_matched.py" \
    --gt_path "$GT_DIR" \
    --results_path "$RESULTS_DIR" \
    --output_dir "$EXP_ROOT/attacked_Result" \
    --run_name "$ATTACK_RUN" \
    --attack_scope nongeom \
    --seed 61 \
    --overwrite \
    2>&1 | tee "$EXP_ROOT/logs/${TAG}_attack.log"

  "$PY" "$DIFFW_ROOT/calculate_ba.py" \
    --config "$FORMAL_EVAL_CONFIG" \
    --resume "$FORMAL_CKPT" \
    --attack_dir "$ATTACK_DIR" \
    --gt_message_dir "$RESULTS_DIR" \
    2>&1 | tee "$EXP_ROOT/logs/${TAG}_attack_ba.log"
done
```

再彙總一次：

```bash
"$PY" "$O256_ROOT/scripts/summarize_diffw_quality_tradeoff.py" \
  --exp_root "$EXP_ROOT"
```

---

## 4. 這條實驗要怎麼解讀

你要回答的不是：

- `DiffW-like` 能不能被做成 BA 死光的高畫質圖

而是：

- 如果以 `O256-RGB` 現在的畫質區間當門檻，
- `DiffW-like` 往那個區域靠時，
- 它的 `Clean BA / Avg BA / Min BA` 會掉多少。

這條如果結果清楚，就能在口試時比較穩地說：

> 在本文統一設定下，若任務偏重輸出影像品質，且可接受一定程度的 robustness trade-off，則 O256-RGB 可視為合理的折衷點。

---

## 5. 後續如果口試後還要補

若這條 post-hoc sweep 結果已經很有訊號，口試前可以先停在這裡。  
口試後若要再補，再考慮：

- train-time `strength_factor` sweep
- attack pressure / attack schedule variants
- checkpoint selection variants

