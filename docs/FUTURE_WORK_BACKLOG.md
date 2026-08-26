# O256 / StageE 後續研究 Backlog

更新日期：2026-07-23

這份文件整理目前適合交給後續學生延伸的研究方向。來源主要是先前整理的：

```bash
/home/jeffrey/projects/實驗結果/14_mainline_formal_protocol/O256後續研究方向與高成本實驗暫存_20260713.md
```

這裡已更新到 EXP54 完成後的狀態。

## 目前主線狀態

| 項目 | 狀態 |
|---|---|
| EXP52 StageE seed62 full from-scratch | 已完成，strict pass |
| EXP53 StageE seed61 full from-scratch | 已完成，strict pass |
| EXP54 StageE seed63 full from-scratch | 已完成，near-miss validation |
| 目前論文 final main results | 建議以 EXP52 / EXP53 為主 |
| EXP54 定位 | 補充驗證；不寫成 strict-pass final result |

## 方向一：非 COCO / domain shift 資料集驗證

這是最適合給新接手學生的第一條線。

| 項目 | 內容 |
|---|---|
| 目的 | 確認 StageE 是否只在 COCO finaltest1000 上有效，或能推廣到其他資料集 |
| 建議資料 | Kodak24、BSD100 / BSDS subset、DIV2K validation、Flickr2K / Unsplash / FFHQ subset |
| 成本 | 低到中，若只做 evaluation 不重訓 |
| 第一階段建議 | 使用既有 EXP52/EXP53 StageE checkpoint + refiner，先跑 inference/evaluation |
| 輸出 | PSNR / SSIM / LPIPS / Clean BA / 11 attacks BA / Avg BA / Min BA / Dropout BA |

不要一開始就對新 dataset full-from-scratch 重訓。先做 domain-shift evaluation，再依結果決定是否需要 adaptation。

## 方向二：更多 seed 的 StageE 穩定性

| 項目 | 內容 |
|---|---|
| 已做 | seed62 EXP52、seed61 EXP53、seed63 EXP54 |
| 目前觀察 | EXP52/53 通過，EXP54 near-miss |
| 後續問題 | StageE 是否存在 seed-dependent boundary？ |
| 成本 | 高，每條都接近完整 StageE rerun |
| 建議 | 若要做，先選一個新 seed 做完整 rerun，不要一次展開很多 seed |

## 方向三：S / strength trade-off

| 項目 | 內容 |
|---|---|
| 目的 | 建立浮水印強度、PSNR、BA 的可調式 trade-off |
| 成本 | 中到高 |
| 風險 | 容易變成大規模 sweep |
| 建議 | 先用少量 pilot 找範圍，再決定是否完整 sweep |

## 方向四：bitplane 對應與弱 bit 定位

| 項目 | 內容 |
|---|---|
| 已有基礎 | EXP25、EXP48 |
| 目前觀察 | bitplane 可作診斷，但弱 bit 不一定固定 |
| 後續問題 | 是否能追蹤單一 bit / channel 在 base、refiner、attack 後仍可定位？ |
| 成本 | 中到高 |
| 可能產出 | bit-specific loss、bit-specific sampling schedule、bitplane-aware watermarking |

## 方向五：refiner evidence preservation

| 項目 | 內容 |
|---|---|
| 已有基礎 | EXP31、EXP46、EXP47、EXP48、EXP50、EXP51 |
| 目前觀察 | refiner 能補畫質，但設計太粗時可能抹掉 watermark evidence |
| 後續問題 | refiner 常在哪些區域、頻率、bit 或 image group 破壞訊號？ |
| 成本 | 中到高 |
| 可能產出 | local guard、frequency-aware guard、decoder-gradient-aware refiner loss |

## 方向六：PSNR 36-37 dB 高畫質路線

| 項目 | 內容 |
|---|---|
| 目標 | 在 Clean BA >= 99、Avg BA >= 98、Min BA >= 90 下把 PSNR 往 36-37 dB 推 |
| 成本 | 高 |
| 風險 | 可能再次犧牲 Dropout floor |
| 可能路線 | VAE / decoder fine-tune、更強 refiner、LPIPS-aware loss、frequency-preserving refiner |

這條比較像下一代架構型研究，不建議作為新學生第一個任務。

## 暫存規則

只要新想法符合以下任一條件，先記錄在 backlog，不要直接變成短期必做：

1. 需要完整 from-scratch base + refiner 重訓。
2. 需要三個以上 seed。
3. 需要大規模 sweep。
4. 需要改模型架構。
5. 需要大量 per-image / per-bit / per-attack 診斷。
6. 主要是後續研究，不是修補目前主結論。

