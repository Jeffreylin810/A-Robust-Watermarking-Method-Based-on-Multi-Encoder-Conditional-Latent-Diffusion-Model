# O256 exp50 decoder-aware refiner probe 規劃

## 目的

exp48 顯示目前正式 refiner 幾乎不會破壞 clean BA，反而會幫 Dropout；真正問題是 seed62 的 Dropout floor 還差一點，且 exp44/47 那種硬案例重播或 residual guard 會把 Dropout 與畫質一起弄壞。exp49 又確認 PSNR/BA 評估沒有配對錯誤或 message leakage，所以 exp50 回到模型側，直接測「更強的 decoder-aware Dropout 訊息保護」能不能修補這個 floor。

## 實驗設計

| 分支 | 做法 | 想回答的問題 |
|---|---|---|
| exp50A | 從 exp41 StageC 重新訓練最後 StageD，將 clean/message/Dropout attack-message 權重拉高，並提高 quality gate。 | 如果重做最後一段 refiner，而不是接著錯誤方向硬補，能不能讓 Dropout BA 過 90 並守住 PSNR？ |
| exp50B | 從 exp41 正式 StageD 做 250 steps 小幅 refresh，低 learning rate、提高 Dropout attack-message 權重。 | exp41 seed62 的 89.94 是否只是差一個很小的安全邊界？ |

## 判讀標準

- 若 PSNR >= 33.8、Clean BA >= 99%、Avg BA >= 98%、Min BA >= 90，代表這個 refiner 設定值得完整重頭訓練驗證。
- 若 Dropout 有上升但未過 90，代表方向可能有效，但還不能當正式結果。
- 若 Dropout 沒改善或 PSNR 明顯掉，表示 refiner 小修補路線應該停損，下一步要往 base-side 或架構型方法走。

## 執行指令

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp50_decoderaware_refiner_probe_seed62_20260708.sh
```

## 預估

- 額外空間：約 4-6G。
- 預估時間：約 1.5-3 小時。
- 這是 continuation probe，不是正式實驗；若通過，才需要重新做完整 from-scratch protocol。
