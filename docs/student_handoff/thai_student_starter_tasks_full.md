# Thai Student Starter Tasks for O256 / StageE

更新日期：2026-07-23

這份文件是給短期接手 O256 / StageE 實驗的學生使用。目標不是一開始就重訓 StageE，而是先熟悉程式碼、確認環境可跑、再做小規模 domain-shift / dataset generalization 測試。

## 0. 應該交給學生哪一份版本？

建議交付以下三部分：

1. 程式碼版本：

```bash
/home/jeffrey/projects/O256-PAPER-github-backup
```

這是比較適合分享或上傳 GitHub 的輕量程式版本。不要讓學生直接改你的主要工作目錄：

```bash
/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
```

2. 研究交接索引：

```bash
/home/jeffrey/projects/613410280_data/O256_handoff
```

這裡放的是說明文件、依賴地圖、正式跑法、重大版本保留規劃、後續研究 backlog。

3. 外部大檔資源：

請依照：

```bash
docs/EXTERNAL_RESOURCES.md
```

準備 dataset、SD2.1 VAE、checkpoint、refiner checkpoint 等資源。

目前：

```bash
/home/jeffrey/projects/Thai_experience
```

是空資料夾，比較適合之後讓學生放自己的新實驗輸出，不適合作為正式交接版本本體。

## 1. 第一週建議任務

### Task 1: 讀文件與跑環境檢查

先讀：

```bash
README.md
docs/STAGEE_DEPENDENCY_MAP.md
docs/STAGEE_RUN_FLOW_MANUAL.md
docs/RGB_REFINEMENT_IMPLEMENTATION_NOTE.md
docs/EXTERNAL_RESOURCES.md
```

確認他們能回答：

- base model 是什麼？
- RGB refiner 是什麼？
- StageA/B/B+/C/D/E 分別在做什麼？
- EXP52 / EXP53 / EXP54 的差異是什麼？
- finaltest1000、train10000、32-bit message、256x256 是什麼？

然後只跑 check，不跑正式訓練：

```bash
cd /home/jeffrey/projects/O256-PAPER-github-backup

bash scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh --check-only
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh --check-only
bash scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh --check-only
```

如果路徑不對，先修 local path / symlink，不要直接改核心訓練程式。

### Task 2: 先複現 summary，不重訓

請學生先確認既有結果能被讀懂：

```bash
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/summary.csv
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/summary.csv
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/summary.csv
```

並能說明：

- EXP52 / EXP53 是 strict-pass final main results。
- EXP54 是 seed63 near-miss validation，不是 strict pass。
- Min BA attack 主要是 `Paper_Dropout_r03`。

### Task 3: 跑小規模 dataset generalization pilot

第一個真正實驗建議不是 full retraining，而是用已訓練好的 StageE model/refiner 去測其他資料集的 domain shift。

建議資料集從 3 到 4 個開始：

| 優先順序 | Dataset | 建議原因 | 建議規模 |
|---|---|---|---:|
| 1 | Kodak24 | 經典小型影像品質 benchmark，成本低 | 24 |
| 2 | BSD100 / BSDS500 subset | 常見自然影像 benchmark | 100 或 200 |
| 3 | DIV2K validation | 高品質自然影像，常用於 restoration / SR | 100 |
| 4 | Flickr2K / Unsplash subset / FFHQ subset | 測更不同 domain，可視資料取得情況選一個 | 100-500 |

建議流程：

1. 把資料集 resize / center crop 到 256x256。
2. 用同一顆 StageE final model 對資料集產生 watermarked image。
3. 套用同一個 StageE refiner。
4. 計算 PSNR / SSIM / LPIPS。
5. 計算 clean BA。
6. 跑同一組 11 種 formal non-geometric attacks。
7. 輸出 per-attack BA、Nongeom Avg BA、Min BA。

一開始請只做 inference/evaluation，不要重訓：

- 不要改 base training protocol。
- 不要改 StageD / StageE loss。
- 不要先做新 dataset full from-scratch。

## 2. Dataset generalization 的成功與失敗怎麼判讀？

如果新資料集上 PSNR / BA 接近 COCO finaltest1000：

- 可以說 StageE 有初步 domain generalization。
- 下一步可以擴大資料集或增加 seed。

如果 PSNR 還可以，但 BA 掉：

- 可能代表 watermark evidence 在該 domain 較不穩。
- 下一步可接 bitplane / evidence preservation 診斷。

如果 BA 還可以，但 PSNR 掉：

- 可能是 refiner 對該資料集畫質分布不適應。
- 下一步可研究 refiner domain adaptation。

如果 Dropout 特別弱：

- 這和目前主線瓶頸一致。
- 可以接續 StageD / StageE 的 Dropout-aware refinement 方向。

## 3. 不建議學生一開始做的事情

不要一開始就做：

- 多 seed full-from-scratch StageE 重訓。
- 大規模 strength / S sweep。
- 修改 refiner architecture。
- 改 attack set 後直接和論文正式數字比較。
- 把 non-finaltest1000 或不同 attack set 的結果寫成 formal main result。
- 使用舊 O256-RGB compute-only timing 代替 StageE-specific timing。

## 4. 建議學生回報格式

每個新資料集至少回報：

| 欄位 | 說明 |
|---|---|
| dataset | 資料集名稱 |
| num_images | 測試影像數 |
| preprocessing | resize / crop / normalization |
| model | 使用 EXP52 / EXP53 / EXP54 哪一個 checkpoint/refiner |
| PSNR / SSIM / LPIPS | 影像品質 |
| Clean BA | 未攻擊 BA |
| Nongeom Avg BA | 11 attacks 平均，不含 Clean BA |
| Min BA | 最弱攻擊 BA |
| Min attack | 最弱攻擊名稱 |
| Paper_Dropout_r03 BA | Dropout 指標 |
| source scripts | 使用哪些 scripts / configs |
| notes | 任何失敗或路徑問題 |

建議輸出：

```bash
summary.csv
per_attack_ba.csv
run_notes.md
```

## 5. 可以給學生的第一句任務描述

> Please first read the O256 / StageE handoff documents and verify that the code, dataset path, VAE path, and StageE scripts are understandable. Do not retrain the full model in the first task. Your first experiment is to evaluate the existing StageE model on 3-4 common image datasets, such as Kodak24, BSD100/BSDS subset, DIV2K validation, and one additional natural-image or face/product dataset. For each dataset, report PSNR, SSIM, LPIPS, Clean BA, 11-attack per-attack BA, Nongeom Avg BA, Min BA, and Paper_Dropout_r03 BA under the same formal attack set.

