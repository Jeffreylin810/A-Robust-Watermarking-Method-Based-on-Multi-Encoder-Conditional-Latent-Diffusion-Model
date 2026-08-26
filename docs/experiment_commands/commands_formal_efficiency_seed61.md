# Formal Efficiency Experiment (seed61)

這一步是 **F 類正式實驗**，用途是補齊第四章的：

1. `base training cost`
2. `module preparation cost`
3. `complete preparation cost`
4. `inference cost`
5. `peak GPU memory`

它只使用已凍結的正式 artifact：

- `DiffW-like formal train10000/msg32/20k`
- `O256 formal base`
- `O256 formal RGB refinement module`

不會再改 recipe，也不會再碰 finaltest1000 的方法選擇。

## 0. Preflight check

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_formal_efficiency_seed61.sh --check-only
```

## 1. 正式執行

請在 **GPU idle** 的時候跑，避免 timing 被其他工作污染：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_formal_efficiency_seed61.sh
```

## 2. 產出位置

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_speed_cost_seed61/summary.md
```

raw benchmark JSON 會在：

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_speed_cost_seed61/raw
```

## 3. 說明

這個 summary 會把三種不同層級的成本拆開：

- `DiffW-like complete preparation`
- `O256 base training`
- `O256-RGBREFINE complete preparation`

以及三種推論角色：

- `DiffW-like`
- `O256 base`
- `O256 + RGB refinement module`

寫第四章時不要把它們混成一個模糊的「速度比較」。
