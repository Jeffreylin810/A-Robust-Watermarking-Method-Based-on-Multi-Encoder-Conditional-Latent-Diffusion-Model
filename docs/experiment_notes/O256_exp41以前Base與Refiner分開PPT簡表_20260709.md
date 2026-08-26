# O256 exp41 以前 Base / Refiner 分開 PPT 簡表

這份是投影片用的精簡版。完整流水帳請看：

- `O256_exp41以前PSNR_BA變化整理_20260709.md`
- `O256_exp41以前PSNR_BA變化整理_20260709.csv`

建議 PPT 不要把所有實驗塞同一張表，而是拆成：

1. Base-only 診斷：證明 BA 主要是從 base-side attack schedule 被推上去。
2. Base-side 策略套到 RGB 後：證明改 base 後，最終輸出也能吃到 BA 改善。
3. Refiner trade-off：說明 refiner 主要負責畫質，但太偏畫質會傷 Min BA。
4. 收斂結論：為什麼最後走到 exp40。

---

## Slide 1：Base-only 診斷結果

> 這張只看 base model 本身，不看 RGB 修復模組。重點不是 PSNR，而是 BA floor 怎麼被 attack schedule 推上去。

| 代號 | Base 訓練策略 | PSNR | Clean BA | Avg BA | Min BA | 重點 |
|---|---|---:|---:|---:|---:|---|
| formal base | 原正式 base | 24.7037 | 99.30 | 87.33 | 70.38 | BA floor 很低，主要弱點是 Dropout。 |
| exp16 base +10k | 單純多訓練 base | 24.7590 | 99.88 | 93.17 | 77.97 | 多訓練有效，但還不夠。 |
| exp23 weak base | weakattack +5k | 24.6837 | 99.99 | 97.41 | 86.23 | weak attack 放進 base-side 後，Min BA 大幅上升。 |
| exp28 dropout base | dropout-focus +5k | 24.6456 | 100.00 | 98.02 | 87.95 | 專門壓 Dropout 後，base floor 再上升。 |
| exp37 margin base s61 | dropout-margin seed61 | 24.6238 | 100.00 | 98.31 | 88.43 | dropout-margin 比單純 focus 更穩。 |
| exp38 margin base s62 | dropout-margin seed62 | 24.6189 | 100.00 | 98.36 | 88.72 | 跨 seed 仍成立。 |
| exp40 base scratch | 完整重頭 base | 24.6354 | 100.00 | 98.45 | 89.18 | 從頭重跑也能重現 base-side 改善。 |

**這張可以講：**  
Base-only 的 PSNR 低是預期的，因為 RGB 畫質修復還沒做；但 BA 趨勢很清楚，真正把 robustness 推起來的是 base-side attack schedule，尤其是 weakattack / Dropout / dropout-margin。

---

## Slide 2：Base-side 策略套到 RGB 後的主結果

> 這張看最終輸出，也就是 base 改完後再套 RGB refiner。它回答：「base-side 的 BA 改善最後有沒有反映到可用圖片上？」

| 代號 | 主要變因 | PSNR | Clean BA | Avg BA | Min BA | 重點 |
|---|---|---:|---:|---:|---:|---|
| formal RGB | 原正式 O256-RGB | 32.9034 | 95.81 | 87.38 | 72.93 | 起點：畫質可以，但 BA floor 弱。 |
| exp16 +10k RGB | base +10k | 32.4787 | 98.97 | 94.64 | 82.27 | 單純拉長 base，BA 明顯上升。 |
| exp21 +20k reuse | base +20k | 32.2933 | 99.55 | 96.49 | 85.68 | base 再強，BA 繼續上升。 |
| exp23 weak +5k | weakattack base +5k | 32.9388 | 99.89 | 98.21 | 89.61 | 第一次接近 Min BA 90。 |
| exp28 dropout focus | dropout-focus base +5k | 32.5223 | 99.91 | 98.64 | 91.43 | 第一次明確過 Min BA 90。 |
| exp37 dropout margin s61 | dropout-margin seed61 | 33.1895 | 99.91 | 98.58 | 90.11 | 用 margin 讓結果比較平衡。 |
| exp38 dropout margin s62 | dropout-margin seed62 | 33.1252 | 99.93 | 98.60 | 90.31 | seed62 也過門檻。 |
| exp40 full scratch s61 | 完整重頭訓練 | 33.9372 | 99.96 | 98.77 | 91.01 | 正式候選：從頭訓練也成立。 |

**這張可以講：**  
從 formal 到 exp28，Min BA 從 `72.93` 推到 `91.43`，證明 base-side attack schedule 是最主要的 robustness 來源；exp37/38 再把這個方向改成更穩的 dropout-margin，最後 exp40 用完整重頭訓練確認不是 checkpoint continuation 的偶然。

---

## Slide 3：Refiner 早期 trade-off：畫質上升，但 BA floor 下降

> 這張只看 refiner 變化。重點是 refiner 能救 PSNR，但如果只追畫質，Min BA 會掉。

| 代號 | Refiner 設定 | PSNR | Clean BA | Avg BA | Min BA | 重點 |
|---|---|---:|---:|---:|---:|---|
| exp16 +10k RGB | 舊 refiner / base +10k | 32.4787 | 98.97 | 94.64 | 82.27 | +10k base 的 RGB 起點。 |
| exp17 r500 | refresh500 | 33.5891 | 98.59 | 94.10 | 80.79 | PSNR +1.11，但 Min BA -1.48。 |
| exp17 r1000 | refresh1000 | 33.7365 | 98.56 | 94.09 | 80.73 | 再加步數主要只加畫質。 |
| exp21 +20k reuse | 舊 refiner / base +20k | 32.2933 | 99.55 | 96.49 | 85.68 | +20k base 的 RGB 起點。 |
| exp21 +20k r500 | refresh500 | 33.8299 | 99.39 | 95.91 | 83.36 | PSNR 明顯上升，但 Min BA 掉。 |
| exp22 +20k r1500 | refresh1500 | 34.2602 | 99.33 | 95.72 | 82.76 | PSNR 最高，但 BA floor 更弱。 |

**這張可以講：**  
Refiner 本身不是壞事，它確實把 PSNR 拉上來；但如果沒有訊息保護或 attack-aware 設計，refiner 會傾向修圖，導致 watermark robustness 尤其是 Dropout floor 下降。

---

## Slide 4：Refiner 補救實驗：單靠 refiner 不夠穩

> 這張承接 Slide 3：我們不是沒試 refiner，而是試過不同 refiner 補法後，發現它不能單獨解決問題。

| 代號 | Refiner 補法 | PSNR | Clean BA | Avg BA | Min BA | 結論 |
|---|---|---:|---:|---:|---:|---|
| exp24 weak refiner | weak-targeted refiner | 34.1123 | 99.35 | 95.78 | 82.87 | 只改 refiner 救不了 weak attack floor。 |
| exp26 bitweighted refiner | bit-weighted refiner | 34.1088 | 99.33 | 95.76 | 82.78 | bit weighting 放 refiner 端效果有限。 |
| exp29 quality recovery | quality recovery r500 | 33.6062 | 99.88 | 98.17 | 88.63 | 畫質救回，但 Min BA 掉破 90。 |
| exp30 conservative r250 | conservative refresh250 | 32.5422 | 99.93 | 98.61 | 91.13 | 能守 BA，但 PSNR 幾乎沒救回。 |
| exp31 msg-preserving | message-preserving | 32.8792 | 99.92 | 98.39 | 89.70 | 有保訊息概念，但還不夠穩。 |
| exp32 dropout-specific | dropout-specific | 32.0162 | 99.92 | 98.86 | 92.87 | BA 很強，但 PSNR 代價太大。 |
| exp33 dropout-balanced | balanced refiner | 33.2346 | 99.89 | 98.41 | 90.05 | 第一個平衡候選。 |
| exp34 full validation | exp33 重驗 | 33.2333 | 99.90 | 98.35 | 89.72 | 重驗後未過 90，不能定案。 |

**這張可以講：**  
Refiner 的角色比較像「把 base 的結果修成高畫質圖片」，但它不是 robustness 的主要來源；如果 base 沒把 watermark margin 打好，refiner 很難自己補回來。這也是為什麼後面轉向 base-side dropout-margin。

---

## Slide 5：最後收斂點

| 階段 | 最代表實驗 | PSNR | Clean BA | Avg BA | Min BA | 結論 |
|---|---|---:|---:|---:|---:|---|
| 原始正式結果 | formal RGB | 32.9034 | 95.81 | 87.38 | 72.93 | 畫質可用，但 robustness 不夠。 |
| base-side 找到方向 | exp23 weak +5k | 32.9388 | 99.89 | 98.21 | 89.61 | weakattack schedule 是關鍵。 |
| 第一次過 90 | exp28 dropout focus | 32.5223 | 99.91 | 98.64 | 91.43 | BA 過門檻，但畫質略低。 |
| 平衡候選 | exp33 dropout-balanced | 33.2346 | 99.89 | 98.41 | 90.05 | 初步平衡，但需重驗。 |
| 穩定方向 | exp37 / exp38 margin | 33.1895 / 33.1252 | 99.91 / 99.93 | 98.58 / 98.60 | 90.11 / 90.31 | 跨 seed continuation 都通過。 |
| 正式重頭驗證 | exp40 full scratch | 33.9372 | 99.96 | 98.77 | 91.01 | 從頭訓練也成立，可作正式候選。 |

**一句話總結：**  
Base-side 負責把 watermark robustness margin 建起來，refiner 負責把影像品質拉回來；前面的實驗證明「只靠 refiner」不穩，所以最後 protocol 是先用 base-side dropout-margin 建 robustness，再用 balanced refiner 做畫質修復。
