# Experiment 11: Nongeom-Light Curriculum From Scratch

分類：`E`（探索實驗，只使用 dev1000）

目的：比較以下三種條件，確認非幾何攻擊應何時加入，而不是直接假定後續 5k 是最佳：

```text
Control：Pixel Bridge + identity 20k -> nongeom-light 5k
Candidate A：Pixel Bridge + nongeom-light from scratch 20k
Candidate B：Pixel Bridge + nongeom-light from scratch 25k
```

本實驗不使用 finaltest1000。結果會寫入：

```text
results/experiment_summaries/exp11_nongeomlight_curriculum_seed61/summary.md
```

## 1. 執行前檢查

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp11_nongeomlight_fromscratch_20k_25k.sh --check-only
```

看到 `Experiment 11 preflight passed.` 才進行正式執行。

## 2. 在 tmux 執行

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

tmux new-session -s exp11_light_curriculum \
  'bash scripts/run_exp11_nongeomlight_fromscratch_20k_25k.sh; exec bash'
```

離開但不中止：

```text
Ctrl-b d
```

重新進入：

```bash
tmux attach -t exp11_light_curriculum
```

## 3. 完成後提供

```text
results/experiment_summaries/exp11_nongeomlight_curriculum_seed61/summary.md
```

Exp12 的設定取決於 Exp11 選出的 curriculum，因此不要在 Exp11 結果出來前自行啟動
5k-vs-10k 實驗。Exp13 的 Stage A/B/C 步數縮減又必須等待 Exp12 鎖定 base
checkpoint，否則會用不同 base 產生 pairs，失去公平性。

