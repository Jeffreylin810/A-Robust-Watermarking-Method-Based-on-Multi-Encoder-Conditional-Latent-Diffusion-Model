# O256 exp51 decoder-aware StageE cross-seed probe 規劃

## 目的

exp50B 在 seed62 上用 250 steps 的 decoder-aware micro-refresh，把 Dropout BA 從 89.94 推到 90.15，且 PSNR 只小幅下降。exp51 要確認這不是 seed62 單點救火，而是可以放進正式 protocol 的最後 StageE 候選。

## 實驗設計

固定 exp50B 的設定，不再新增參數：

- 從正式 StageD refiner 接續 250 steps。
- `message_weight=1.50`
- `attack_message_weight=0.65`
- `attack_profile=dropout_r03_only`
- `attack_dropout_prob=0.70`
- `sensitive_residual_weight=3.00`
- `quality_gate_psnr=33.80`
- `quality_gate_weight=90.0`
- learning rate `5e-6`

分別套到：

| 分支 | 起點 | 想回答的問題 |
|---|---|---|
| exp51 seed61 | exp40 formal seed61 StageD | StageE 會不會破壞原本已通過的 seed61？ |
| exp51 seed62 | exp41 formal seed62 StageD | StageE 能不能重現 exp50B，把 seed62 Dropout floor 推過 90？ |

## 判讀標準

- 兩個 seed 都要 PSNR >= 33.8、Clean BA >= 99%、Avg BA >= 98%、Min BA >= 90。
- seed61 不能為了救 seed62 而明顯犧牲原本已通過的結果。
- 若兩個 seed 都通過，下一步才值得做完整 from-scratch formal rerun，把 StageE 寫進鎖定 protocol。

## 執行指令

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp51_decoderaware_stageE_crossseed_probe_20260708.sh
```

## 預估

- 額外空間：約 5-7G。
- 預估時間：約 2-4 小時。
- exp51 仍是 continuation probe，不是正式 from-scratch 結果。
