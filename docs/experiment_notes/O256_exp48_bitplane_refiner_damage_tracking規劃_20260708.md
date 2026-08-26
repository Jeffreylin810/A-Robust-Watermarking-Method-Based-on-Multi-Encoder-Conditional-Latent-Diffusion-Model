# O256 exp48 bitplane x refiner-damage tracking 規劃

Date: 2026-07-08

## 1. 為什麼做

前面 exp40/41 已經證明 O256-RGB formal protocol 不是單次運氣；但 exp41 的 Dropout floor 還差 0.06。  
exp44/45/46/47 又顯示，單純 hardcase replay 或 residual guard 不是好解法。

因此 exp48 不再先訓練新模型，而是做診斷：

1. 哪些 bit 在 base 階段就弱？
2. 哪些 bit 是 RGB 影像修復模組之後才變弱？
3. 弱 bit 在 seed61 / seed62 是否穩定？
4. refiner 是破壞 clean message，還是主要破壞 Dropout attack 後的 message evidence？
5. exp44/47 這些失敗 refiner 和 exp41 正常 refiner 的差異在哪裡？

## 2. 實驗定位

exp48 是 diagnosis-only experiment：

- 不訓練 base model
- 不訓練 RGB refiner
- 不產生新的 attack 圖片
- 只讀既有 exp40 / exp41 / exp44 / exp47 的 clean 與 Dropout 輸出
- 重新用 decoder 做 per-image / per-bit 統計

## 3. 會分析哪些樣本

| 樣本 | seed | stage | condition |
|---|---|---|---|
| exp40_base_clean | seed61 | base | clean |
| exp40_base_dropout | seed61 | base | Paper_Dropout_r03 |
| exp40_rgb_clean | seed61 | RGB refiner | clean |
| exp40_rgb_dropout | seed61 | RGB refiner | Paper_Dropout_r03 |
| exp41_base_clean | seed62 | base | clean |
| exp41_base_dropout | seed62 | base | Paper_Dropout_r03 |
| exp41_rgb_clean | seed62 | RGB refiner | clean |
| exp41_rgb_dropout | seed62 | RGB refiner | Paper_Dropout_r03 |
| exp44_rgb_clean | seed62 | hardcase replay refiner | clean |
| exp44_rgb_dropout | seed62 | hardcase replay refiner | Paper_Dropout_r03 |
| exp47A_rgb_clean | seed62 | mild guarded refiner | clean |
| exp47A_rgb_dropout | seed62 | mild guarded refiner | Paper_Dropout_r03 |
| exp47B_rgb_clean | seed62 | strong guarded refiner | clean |
| exp47B_rgb_dropout | seed62 | strong guarded refiner | Paper_Dropout_r03 |

## 4. 會比較哪些關係

### Base 到 RGB

這是回答 refiner 是否破壞訊息：

- exp40 base clean -> exp40 RGB clean
- exp40 base Dropout -> exp40 RGB Dropout
- exp41 base clean -> exp41 RGB clean
- exp41 base Dropout -> exp41 RGB Dropout

如果 clean 沒壞，但 Dropout 壞，代表 refiner 問題不是 clean BA，而是 attack 後的 watermark evidence。

### Clean 到 Dropout

這是回答攻擊本身造成哪些 bit / image 掉 BA：

- exp40 base clean -> exp40 base Dropout
- exp40 RGB clean -> exp40 RGB Dropout
- exp41 base clean -> exp41 base Dropout
- exp41 RGB clean -> exp41 RGB Dropout
- exp44 RGB clean -> exp44 RGB Dropout
- exp47A/B RGB clean -> exp47A/B RGB Dropout

### Cross-seed

這是回答 weak bit 是否穩定：

- exp40 base Dropout -> exp41 base Dropout
- exp40 RGB Dropout -> exp41 RGB Dropout

如果 weak-bit overlap 高，bit-specific training 比較有機會；如果 overlap 低，bitplane 比較適合當診斷工具。

### 失敗 refiner 對照

這是回答 exp44/47 和 exp41 的差異：

- exp41 RGB clean -> exp44 RGB clean
- exp41 RGB Dropout -> exp44 RGB Dropout
- exp41 RGB clean -> exp47A/B RGB clean
- exp41 RGB Dropout -> exp47A/B RGB Dropout

## 5. 輸出

Experiment root:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/48_supplementary_o256_bitplane_refiner_damage_tracking_20260708
```

主要輸出：

- `diagnosis/summary.md`
- `diagnosis/variant_summary.csv`
- `diagnosis/per_bit_summary.csv`
- `diagnosis/per_image_summary.csv`
- `diagnosis/transition_summary.csv`
- `diagnosis/transition_per_bit_damage.csv`
- `diagnosis/transition_per_image_damage.csv`
- `diagnosis/diagnosis_summary.json`

## 6. 成本與空間

目前 check-only 已通過：

- 14 個樣本
- 每個樣本 1000 張圖片
- 每個樣本 1000 份 message

空間：

- 目前 `/home/jeffrey/projects` 可用約 52G
- exp48 只輸出 CSV/MD/JSON，預估新增遠低於 1G
- 空間足夠

時間：

- 不訓練、不重新攻擊
- 主要時間是 decoder 解碼 14,000 張圖片
- 預估約 20-60 分鐘，視 GPU 狀態而定

## 7. 執行指令

先檢查：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp48_bitplane_refiner_damage_tracking_20260708.sh --check-only
```

快速測試：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp48_bitplane_refiner_damage_tracking_20260708.sh --quick 100
```

正式執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp48_bitplane_refiner_damage_tracking_20260708.sh
```

