# M05 Formal Qualitative (seed61)

這一步是 **F 類正式實驗**，用途是補齊第四章的定性圖片證據。

它使用已完成的正式結果：

- `M03` 的 O256 base finaltest1000 輸出
- `M03` 的 O256-RGBREFINE finaltest1000 輸出
- locked `finaltest1000` GT

並且用**固定規則**選圖，而不是手挑。

## 0. Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m05_formal_qualitative_seed61.sh --check-only
```

## 1. 正式執行

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m05_formal_qualitative_seed61.sh
```

## 2. 輸出位置

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/05_qualitative/formal_seed61_main_qualitative
```

重點檔案：

- `selected_manifest.csv`
- `contact_sheet.png`
- `panels/*.png`
- `diff_maps/*.png`

## 3. 這步的選圖規則

這次不是人工挑圖，而是：

1. 對 1000 張 formal finaltest 圖片計算 `O256 base PSNR`
2. 依 base PSNR 由低到高排序
3. 用固定 quantile 位置抽出 9 張

所以它的角色是：

```text
固定、可重現、可追溯的 qualitative evidence
```

而不是人工 cherry-pick。  
