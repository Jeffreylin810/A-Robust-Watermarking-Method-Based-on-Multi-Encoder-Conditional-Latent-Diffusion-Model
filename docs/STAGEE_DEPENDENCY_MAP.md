# StageE 正式實驗 Dependency Map

更新日期：2026-07-17

這份文件列出 EXP52 / EXP53 / EXP54 正式 StageE 實驗用到哪些程式、config 與資料。

## 主要 runner scripts

| 實驗 | 一鍵腳本 | 角色 |
|---|---|---|
| EXP52 seed62 | `scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh` | StageE formal seed62 full-from-scratch。 |
| EXP53 seed61 | `scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh` | StageE formal seed61 full-from-scratch。 |
| EXP54 seed63 | `scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh` | StageE seed63 robustness/near-miss validation。 |

## Core Python 程式

| 程式 | 角色 |
|---|---|
| `train_diffusion.py` | 訓練 O256 base diffusion watermark model。 |
| `eval_diffusion.py` | 對 finaltest1000 或 train10000 alias 產生 watermarked output。 |
| `attack_eval_diffusion.py` | 執行正式 non-geometric attack suite 並輸出 attacked images / BA report。 |
| `calculate_ba.py` | 計算 clean BA 或 attacked BA。 |
| `calculate_psnr_ssim.py` | 計算 PSNR / SSIM。 |
| `scripts/calculate_lpips_dir.py` | 計算 LPIPS。 |
| `scripts/train_rgb_refinement.py` | 訓練 RGB refiner StageA-E。 |
| `scripts/apply_rgb_refinement.py` | 把 refiner 套到 base output 上，產生 O256-RGB / StageE output。 |

RGB refinement 的詳細實作請看：

```bash
docs/RGB_REFINEMENT_IMPLEMENTATION_NOTE.md
```

## Summary scripts

| 實驗 | Summary script |
|---|---|
| EXP52 | `scripts/summarize_o256_exp52_full_from_scratch_stageE_seed62_20260708.py` |
| EXP53 | `scripts/summarize_o256_exp53_full_from_scratch_stageE_seed61_20260709.py` |
| EXP54 | `scripts/summarize_o256_exp54_full_from_scratch_stageE_seed63_20260714.py` |

## 重要 config 類型

StageE 正式實驗會動態產生或使用以下類型 config：

- identity20k train config
- base30 nongeom-light config
- base50 nongeom-light plus20k config
- base55 weakattack plus5k config
- base60 dropout-focus plus5k config
- base65 dropout-margin plus5k config
- finaltest1000 eval config
- refiner train10000-as-val pair generation config

EXP52 代表性 config：

```bash
configs/CoCo_train10000_ours256_msg32_exp52_identity20k_seed62_20260708.yml
configs/CoCo_train10000_ours256_msg32_exp52_base30_nongeomlight10k_seed62_20260708.yml
configs/CoCo_train10000_ours256_msg32_exp52_base50_nongeomlight_plus20k_seed62_20260708.yml
configs/CoCo_train10000_ours256_msg32_exp52_base55_weakattack_plus5k_seed62_20260708.yml
configs/CoCo_train10000_ours256_msg32_exp52_base60_dropout_focus_plus5k_seed62_20260708.yml
configs/CoCo_train10000_ours256_msg32_exp52_base65_dropout_margin_plus5k_seed62_20260708.yml
configs/CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed62_exp52_20260708.yml
configs/CoCo_refiner_train10000_as_val_ours256_msg32_nongeomlight_base_seed62_exp52_20260708.yml
```

EXP53 / EXP54 同理，只是 seed 與 exp 編號不同。

## Model / dataset modules

| 路徑 | 角色 |
|---|---|
| `models/` | diffusion / restoration / U-Net / auxiliary model code。 |
| `noise_layers/` | Gaussian noise、salt pepper、JPEG、filter、dropout 等攻擊與訓練 noise layer。 |
| `datasets/` | CoCo dataset loader。 |
| `configs/` | 所有訓練與評估設定。 |

## 外部資料依賴

| 資料 | 預期路徑 |
|---|---|
| locked CoCo dataset | `/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked` |
| SD2.1 VAE backup | `/home/jeffrey/projects/VAE_backups/sd21_original_vae_20260604` |
| SD2.1 placeholder path | `/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base` |

## 結果輸出

EXP52/53/54 會輸出到：

```bash
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714
```

最重要的是：

```bash
summary.md
summary.csv
per_attack_ba.csv
logs/
```

若空間不足，raw images、attacked outputs、train pairs 可重建；但 summary、logs、script/config provenance 應保留。
