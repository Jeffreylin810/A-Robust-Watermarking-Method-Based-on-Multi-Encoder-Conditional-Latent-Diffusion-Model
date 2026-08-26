# M04 Formal DiffW-like Finaltest1000

這是 `DiffW-like` 正式 baseline 的第四章主結果收集步驟。

目的：

1. 從 random initialization 重新訓練 formal-clean DiffW-like 20k。
2. 在 locked `finaltest1000` 上正式評估：
   - PSNR / SSIM
   - Clean BA
   - 11 種 non-geometric attack BA
3. 產生可與 `M03` 對齊的正式 baseline summary。

## Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m04_formal_diffw_like_finaltest1000.sh --check-only
```

## Full run

請在 `tmux` 內執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m04_formal_diffw_like_finaltest1000.sh
```

## 摘要輸出

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_diffw_like_finaltest1000/summary.md
```

## 注意

- 這一步是 `F` 類正式 baseline results。
- 這一步會正式使用 locked `finaltest1000`。
- 跑完後即可補齊：
  - DiffW-like
  - O256 base
  - O256-RGBREFINE
  的三角色正式主表。
