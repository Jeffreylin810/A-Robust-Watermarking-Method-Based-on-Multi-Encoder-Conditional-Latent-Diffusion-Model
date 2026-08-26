# O256-PAPER GitHub Backup Manifest（2026-07-09）

## 目的

這個 repo 是「輕量研究備份版」，目標是在本機硬碟故障或工作目錄遺失時，可以最快恢復 O256-PAPER 的程式、設定、重大實驗摘要與後續研究脈絡。

它不是完整 raw archive，也不是最終公開發表版。

## 已備份內容

### 程式本體

- `models/`
- `noise_layers/`
- `utils/`
- `datasets/`（只保留 loader code，不放資料集）
- repo 根目錄 `*.py`
- `scripts/`
- `configs/`
- `environment.yml`
- `README.md`

這代表已整理進來的重大節點，對應的 runner / summarizer / analyzer script 原則上都在 `scripts/`，設定檔在 `configs/`。

### 重大結果摘要

只放小型文字/CSV 摘要，不放 raw images、attacked images、workspace、checkpoint。

已備份到 `results_summaries/` 的節點：

| 類別 | 節點 |
|---|---|
| 主線正式摘要 | `01_method_comparison`, `03_efficiency`, `04_cross_seed` |
| 補充基準 | `07_latent_vs_pixel`, `actual10k_deployment`, `repeatability_sanity` |
| 6/29 後 O256 探索 | `exp16` 到 `exp38` |
| 正式 StageD | `exp40`, `exp41` |
| Dropout floor / hardcase 診斷 | `exp42`, `exp43`, `exp44`, `exp45`, `exp46`, `exp47` |
| Bitplane / audit / StageE | `exp48`, `exp49`, `exp50`, `exp51`, `exp52`, `exp53` |

其中 `exp39` 是被完整 from-scratch 正式實驗取代的 locked continuation validation；目前未找到獨立 summary 檔，因此沒有放進 `results_summaries/`。

### 研究脈絡文檔

已備份到 `docs/experiment_notes/` 的重點文件：

- `O256_20260629之後實驗代表意義簡表_20260706.md`
- `O256_20260629之後逐實驗脈絡_20260706.md`
- `O256_20260629至20260705實驗總覽與研究脈絡.md`
- `O256_逐實驗實作方式與中文脈絡_20260707.md`
- `O256_exp41以前Base與Refiner分開PPT簡表_20260709.md`
- `O256_exp41以前PSNR_BA變化整理_20260709.md`
- exp48 / exp49 / exp50 / exp51 / exp53 規劃文件
- `正式與探索實驗分類規則.md`
- `DiffW_like與O256_RGB最佳版本完整對照表_20260630.md`
- `Training_side口試答法與最終總表_20260629.md`

## 沒有放進 GitHub 的內容

以下內容刻意不放 GitHub，避免 repo 變成數十 GB 或包含本機機器狀態：

- checkpoint：`*.pth`, `*.pth.tar`, `*.ckpt`, `*.pt`
- 生成圖片、attacked images、eval outputs
- `workspace/`, `train_pairs/`, `attacked_Result/`, `eval_outputs/`, `apply/`
- dataset copies
- Stable Diffusion / VAE weights
- LPIPS cache 或大量中間檔
- 本機磁碟清理規劃
- 本機 DO_NOT_DELETE / 清理保留清單

## 從 GitHub 恢復研究時還需要另外準備

1. 建立 conda 環境：

   ```bash
   conda env create -f environment.yml
   conda activate diffw
   ```

2. 放回資料集：

   O256 configs 目前仍使用本機路徑格式，例如：

   ```text
   /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final
   ```

   若換機器，需要把資料集放到對應位置，或批次修改 configs 的 dataset path。

3. 放回 SD2.1 VAE：

   O256 latent training/eval 需要：

   ```text
   /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base/vae
   ```

   這裡只需要 VAE，不需要完整 Stable Diffusion pipeline。若要跑 ZoDiac/ReSD 類完整 pipeline 攻擊，才需要另外準備完整 SD2.1 pipeline。

4. 若要從既有 checkpoint continuation：

   GitHub 不保存 checkpoint。需要從本機備份、雲端硬碟、或重新訓練產生對應 checkpoint。

5. 若要完全重跑正式節點：

   使用 `scripts/` 裡的 formal runner，例如：

   - `run_o256_exp40_full_from_scratch_formal_seed61_20260704.sh`
   - `run_o256_exp41_full_from_scratch_formal_seed62_20260705.sh`
   - `run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh`
   - `run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh`

## 推送前安全檢查

每次 push 前先跑：

```bash
find . -type f -size +50M -print
find . -type f \( -name '*.pth' -o -name '*.pth.tar' -o -name '*.ckpt' -o -name '*.pt' -o -name '*.png' -o -name '*.jpg' -o -name '*.npy' -o -name '*.npz' \) -print
git status --short
```

正常情況下，前兩個 `find` 不應輸出任何檔案。

