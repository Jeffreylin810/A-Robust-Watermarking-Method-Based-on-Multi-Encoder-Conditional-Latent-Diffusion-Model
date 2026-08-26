# M03 Formal Seed61 Finaltest1000

這是第一個真正的第四章正式主結果收集步驟。

目的：

1. 使用已凍結的 formal base（`M01`）。
2. 使用已凍結的 formal full RGB refinement 模組（`M02`）。
3. 在 locked `finaltest1000` 上各跑一次：
   - base only
   - base + module
4. 蒐集：
   - PSNR / SSIM
   - Clean BA
   - 11 種 non-geometric attack BA

## Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m03_formal_seed61_finaltest1000.sh --check-only
```

## Full run

請在 `tmux` 內執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m03_formal_seed61_finaltest1000.sh
```

## 摘要輸出

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_finaltest1000/summary.md
```

## 注意

- 這一步是 `F` 類正式 chapter-4 main results。
- 這一步會第一次正式使用 locked `finaltest1000`。
- 跑完後就不應再拿 `finaltest1000` 去調 recipe。
