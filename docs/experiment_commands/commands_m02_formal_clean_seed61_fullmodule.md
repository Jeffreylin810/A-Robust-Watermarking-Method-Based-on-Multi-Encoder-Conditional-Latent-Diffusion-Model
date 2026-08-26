# M02 Formal-Clean Seed61 Full Module

這是第二個 `F` 類正式實驗入口。

目的：

1. 使用 `M01` 產生的正式 seed61 base30k checkpoint。
2. 重新生成 formal `refine10000 pairs`。
3. 從 random initialization 重訓完整 RGB refinement 模組：
   `Stage A 10k -> Stage B 2k -> Stage B+ 500 -> targeted Stage C 500`
4. 只做 **dev1000 health-check**。
5. **不在這一步碰 finaltest1000**。

## Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m02_formal_clean_seed61_fullmodule.sh --check-only
```

## Full run

請在 `tmux` 內執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m02_formal_clean_seed61_fullmodule.sh
```

## 主要輸出

```text
results/test_result/m02_formal_seed61_refine10000_pairs
results/refinement/m02_formal_seed61_fullmodule_A10k_B2k_BP500_C500/refiner_final.pth
```

摘要：

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_full_module/summary.md
```

## 注意

- 這一步是 `F` 類正式 artifact generation。
- 這一步會重新產生 formal refine10000 pairs，並重訓完整模組。
- 真正的 `finaltest1000` 正式主結果，要等 formal base 與 formal full module 都凍結後，再一次性評估。
