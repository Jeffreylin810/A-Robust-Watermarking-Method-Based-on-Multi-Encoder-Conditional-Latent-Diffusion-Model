# RGB Refinement Reuse Amortization Commands

用途：把既有的 pair-generation + refinement-training 固定成本，轉成 `K 次重用` 的 accounting table。  
注意：這是 **potential amortization via repeated reuse**，不是 formal total-cost superiority proof。

## 執行指令

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python \
  scripts/analyze_rgb_refinement_reuse_amortization.py \
  --output_root /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/rgb_refinement_reuse_amortization_20260627 \
  --reuse_counts 1 2 3 5 10
```

## 輸出

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/rgb_refinement_reuse_amortization_20260627/
  amortized_module_reuse_analysis.csv
  amortized_module_reuse_analysis.json
  amortized_module_reuse_summary.md
```

## 推薦搭配的證據

- Same-COCO Module Reuse (`M07-clean`)
- unseen COCO subset reuse (`eval100 / eval300 / eval1000`)

## 不要寫成

- 「已正式證明 O256-RGB 總成本低於 DiffW-like」
- 「40k crossover 已正式成立」

比較安全的說法是：

> RGB refinement module 的一次性成本，在存在可重複使用情境時，具有被多次 deployment / reuse 分攤的可能性。
