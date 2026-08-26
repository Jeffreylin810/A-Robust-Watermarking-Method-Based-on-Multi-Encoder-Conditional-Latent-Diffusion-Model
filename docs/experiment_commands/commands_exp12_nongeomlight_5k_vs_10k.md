# Experiment 12: Nongeom-Light 5k vs 10k

分類：`E`（探索實驗，只使用 dev1000）

Exp11 已證明 light attack 從 step 1 加入會讓 message learning 失敗，因此 Exp12
固定使用勝出的分段 curriculum：

```text
identity 20k -> nongeom-light
```

本實驗從既有健康的 25k checkpoint 再續訓 5k，得到 30k 候選：

```text
Control：identity20k -> light5k，總計 25k
Candidate：identity20k -> light10k，總計 30k
```

## 執行前檢查

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp12_nongeomlight_5k_vs_10k.sh --check-only
```

## 使用現有 train tmux

```bash
tmux attach -t train
```

進入後執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp12_nongeomlight_5k_vs_10k.sh
```

使用 `Ctrl-b d` 離開但不中止。

完成後提供：

```text
results/experiment_summaries/exp12_nongeomlight_5k_vs_10k_seed61/summary.md
```

只有當 light10k 的 Nongeom Avg BA 或 Minimum BA 至少改善 1 percentage point，
同時 Clean BA 下降不超過 1 point、PSNR 下降不超過 0.3 dB，才會選擇較長的
30k 版本。否則保留成本較低的 25k。

