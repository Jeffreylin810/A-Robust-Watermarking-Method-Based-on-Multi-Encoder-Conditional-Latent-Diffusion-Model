# Experiment 13: RGB Refinement Stage-Step Reduction

分類：`E`（探索實驗，只使用 dev1000）

Exp12 已鎖定 Base：

```text
Pixel Bridge + identity20k -> nongeom-light10k
30k total Base updates
```

Exp13 會先使用這個 Base 產生一套新的固定 `refine10000` pairs，然後比較：

| 版本 | Stage A | Stage B | Stage B+ | targeted Stage C | 總 refine-steps |
|---|---:|---:|---:|---:|---:|
| Full reference | 10k | 2k | 500 | 500 | 13k |
| Reduced | 5k | 1k | 250 | 250 | 6.5k |
| Minimal | 3k | 1k | 250 | 250 | 4.5k |

三個版本的 Base、pairs、loss、學習率、模組架構與 dev1000 完全相同，只改 steps。
Minimal 只有在 Reduced 通過 gate 後才會執行。

## 執行前檢查

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp13_refiner_stage_steps.sh --check-only
```

## 使用現有 train tmux

```bash
tmux attach -t train
```

進入後執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp13_refiner_stage_steps.sh
```

使用 `Ctrl-b d` 離開但不中止。

完成後提供：

```text
results/experiment_summaries/exp13_refiner_stage_steps_seed61/summary.md
```

Gate 是相對 Full reference：

```text
PSNR drop <= 0.5 dB
Clean BA drop <= 1.0 pp
Nongeom Avg BA drop <= 1.5 pp
Minimum BA drop <= 2.0 pp
```

最後選擇符合 gate 的最低成本版本，作為正式 Stage A/B/B+/C recipe。

