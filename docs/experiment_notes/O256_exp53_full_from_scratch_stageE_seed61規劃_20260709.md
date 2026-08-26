# O256 exp53 full from-scratch StageE seed61 規劃

## 目的

exp52 已經把 StageE 寫進 seed62 的完整 from-scratch protocol 並通過。exp53 要做 seed61 對應版本，確認同一套 StageE protocol 在 seed61 也能從頭訓練成立。若 exp53 也通過，就會形成 seed61/seed62 兩顆完整正式 StageE protocol 結果。

## 實驗流程

完整重跑 seed61：

1. base identity20k from scratch
2. nongeom-light 到 base30
3. nongeom-light +20k 到 base50
4. weakattack +5k 到 base55
5. dropout-focus +5k 到 base60
6. dropout-margin +5k 到 base65
7. base65 finaltest1000 評估
8. 重新產生 train10000 pairs
9. RGB refiner StageA/B/B+/C/D
10. StageE decoder-aware250
11. 用 StageE 輸出跑 finaltest1000、Clean BA、完整 non-geom attack BA、PSNR/SSIM/LPIPS

## StageE 固定設定

- `steps=250`
- `lr=5e-6`
- `message_weight=1.50`
- `attack_message_weight=0.65`
- `attack_profile=dropout_r03_only`
- `attack_dropout_prob=0.70`
- `latent_preserve_weight=0.45`
- `sensitive_residual_weight=3.00`
- `quality_gate_psnr=33.80`
- `quality_gate_weight=90.0`

## 判讀標準

- PSNR >= 33.8
- Clean BA >= 99%
- Avg BA >= 98%
- Min BA >= 90%

如果通過，exp53 可與 exp52 一起作為正式 StageE protocol 的雙 seed 結果。

## 執行指令

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh
```

## 預估

- 額外空間：約 25-30G。
- 建議剩餘空間：至少 35G。
- 預估時間：約 5-7 小時。

目前若可用空間低於 35G，建議先清理再執行。
