# M01 Formal-Clean Seed61 Base

這是第一個 `F` 類正式實驗入口。

目的：

1. 依已鎖定正式 recipe，從隨機初始化重新訓練正式 seed61 O256 base。
2. 產生正式的 20k 與 30k checkpoint。
3. 只做 **dev1000 health-check**，確認 formal base 正常。
4. **不在這一步碰 finaltest1000**。
5. **不在這一步重訓 RGB refinement 模組**。

正式 recipe：

```text
identity 20k -> nongeom-light 10k
```

資料：

```text
train root:
/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked

dev health-check GT:
/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt
```

## Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m01_formal_clean_seed61_base.sh --check-only
```

## Full run

請在 `tmux` 內執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m01_formal_clean_seed61_base.sh
```

## 主要輸出

```text
Result/CoCo_ddpm_m01_formal_seed61_identity20k_final.pth.tar
Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar
```

摘要：

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_seed61_base/summary.md
```

## 注意

- 這一步是 `F` 類正式 artifact generation。
- 但它還不是第四章最終主表。
- 真正的 `finaltest1000` 正式數字，應在 formal base、formal refine10000 pairs、
  formal RGB refinement 模組全部完成後，再一次性評估。
