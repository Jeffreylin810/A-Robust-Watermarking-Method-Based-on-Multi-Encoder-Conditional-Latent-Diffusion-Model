# 外部資源與大檔來源說明

更新日期：2026-07-17

這份文件用來交代：哪些東西不適合放 GitHub、哪些東西之後 server 帳號刪掉後可能不會保留，以及後續研究者要如何重新取得。

## 我可以從本機鑑定什麼

我可以從目前程式碼、config、文檔與路徑推斷：

- 實驗使用哪些本機路徑。
- 哪些資源是重跑必需，哪些只是為了省時間。
- 哪些 config 指向 SD2.1 VAE / CoCo locked dataset。
- 哪些 checkpoint 是 continuation 才需要。

但我無法 100% 從本機自動鑑定：

- 你當初是否從官方網站、Hugging Face、同學給的檔案、或其他備份取得某個資源。
- 某個外部帳號權限是否仍有效。
- 如果原始下載來源需要登入或授權，後續研究者是否也能取得。

所以這份文件可以先由我整理「目前可推斷的版本」，但若要交給學弟妹，最好再由你補充一欄「我當初實際怎麼下載/拿到」。

## 必要外部資源

| 資源 | 目前本機路徑 | 重跑必要性 | 說明 |
|---|---|---:|---|
| COCO 2017 locked train/test dataset | `/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked` | 必要 | EXP52/53/54 formal protocol 使用的 locked train10000 / finaltest1000。原始資料集應標註為 COCO 2017，下載入口：`https://cocodataset.org/#download`。 |
| SD2.1 VAE 備份 | `/home/jeffrey/projects/VAE_backups/sd21_original_vae_20260604` | 必要 | O256 latent / VAE 相關流程需要。 |
| SD2.1 model path placeholder | `/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base` | 必要 | 多數 config 指向此路徑；目前重點是其 `vae` 是否能正確連到上方 VAE。 |
| final checkpoints | 各實驗資料夾的 `workspace/.../ckpts/CoCo_ddpm_final.pth.tar` | 非必要但強烈建議 | 沒有 checkpoint 仍可從頭重跑，但會花訓練時間。 |
| refiner checkpoints | 各 EXP52/53/54 的 `refinement/.../refiner_final.pth` | 非必要但強烈建議 | 沒有 refiner checkpoint 仍可重訓 StageA-E。 |
| final summaries | `summary.md` / `summary.csv` / `per_attack_ba.csv` | 必要 | 即使不保存大檔，也應保留最終結果與 attack breakdown。 |

## 如果 server 帳號會被刪掉

如果無法保存大檔，最低限度要保存：

1. GitHub repo：程式碼、config、script、summary。
2. 本文件：資源來源與預期路徑。
3. 代表性 metric 文件：EXP52/53/54、StageE audit、早期 formal O256-base/O256-RGB。

缺少 checkpoint 的後果：

- 無法直接 continuation。
- 無法直接做「接著某顆 checkpoint 微調」的實驗。
- 但如果 dataset / VAE / code 都能重新取得，可以完整從頭重跑 formal protocol。

缺少 dataset / VAE 的後果：

- 不能重跑主實驗。
- 只能閱讀結果與程式碼，不能復現。

## 目前 SD2.1 / VAE 路徑確認

目前本機確認到的狀態是：

```bash
/home/jeffrey/projects/Test/ZoDiac/stable-diffusion-2-1-base
  -> /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base

/home/jeffrey/projects/ZoDiac/stable-diffusion-2-1-base
  -> /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base

/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base/vae
  -> /home/jeffrey/projects/VAE_backups/sd21_original_vae_20260604
```

也就是說，現在 O256 正式實驗實際使用的是 `Test/DiffW-main/stable-diffusion-2-1-base` 這個 model path，其中只有 `vae` 子路徑存在，並且它連到 `VAE_backups/sd21_original_vae_20260604`。ZoDiac 的 SD2.1 路徑目前也是 symlink 到同一個 DiffW-main placeholder。

因此交接時建議寫成：

> 本研究沿用 ZoDiac/DiffW 系列實驗中使用的 Stable Diffusion 2.1 VAE 路徑配置；目前正式 O256 config 指向 `/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base`，其 `vae` 子目錄連到 SD2.1 original VAE 備份。

不要寫成「完整 SD2.1 pipeline 已保留」，因為目前這個路徑只確認有 `vae`，不是完整 pipeline。

## 建議由 Jeffrey 補充的欄位

- CoCo locked dataset 原始來源：
  - 下載網址：`https://cocodataset.org/#download`
  - 使用版本：COCO 2017
  - 是否需要帳號：
  - 是否需要自己 resize/crop：
  - locked split 是如何建立：
- SD2.1 / VAE 原始來源：
  - 下載網址或模型名稱：
  - 使用的是 full pipeline 還是只用 VAE：目前程式確認為使用 `AutoencoderKL` VAE，非完整 SD2.1 pipeline。
  - 是否有手動 symlink：
- 是否有要另外交接的 checkpoint 壓縮包：
  - 保存位置：
  - 檔案大小：
  - 對應實驗：
