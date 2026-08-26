# Experiment 14: seed63 cross-seed Full-module reproducibility

這是 `E` 類探索實驗，不是第四章最終主表。目的不是再找更好的 recipe，而是驗證：

```text
已鎖定的 base30k recipe
+ 已鎖定的 Full RGB refinement 模組
```

是否能跨到新的 `seed63` base，而不需要重訓新的模組。

## 目的

固定以下條件：

- train10000 / dev1000 split
- 解析度 256x256
- message length 32
- SD2.1 latent / VAE
- Base recipe：`identity 20k -> nongeom-light 10k`
- 模組 recipe：`Stage A 10k -> Stage B 2k -> Stage B+ 500 -> targeted Stage C 500`

唯一 intended 變因：

```text
base training seed: 61 -> 63
```

## Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp14_seed63_cross_seed_fullrefiner.sh --check-only
```

## Full run

請在 `tmux` 內執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp14_seed63_cross_seed_fullrefiner.sh
```

## 產出

```text
results/experiment_summaries/exp14_seed63_cross_seed_fullmodule/summary.md
```

## 判斷邏輯

先檢查 seed63 base 是否健康：

- Clean BA 必須 `>= 90%`

只有 base 健康，才允許解讀後續模組轉移結果。

cross-seed 轉移比較基準：

- 參考組：`seed61 base + seed61 Full module`
- 測試組：`seed63 base + transferred seed61 Full module`

容許負向退化：

| 指標 | 容許範圍 |
|---|---:|
| PSNR | 不可下降超過 1.0 dB |
| Clean BA | 不可下降超過 3.0 pp |
| Nongeom Avg BA | 不可下降超過 5.0 pp |
| Minimum BA | 不可下降超過 5.0 pp |

若通過，表示：

```text
seed61 的 Full 模組可以直接轉移到 seed63 的同 recipe base
```

若失敗，下一步才需要考慮：

```text
用 seed63 重新產生 refine10000 pairs
+ 訓練 seed63 專屬模組
```
