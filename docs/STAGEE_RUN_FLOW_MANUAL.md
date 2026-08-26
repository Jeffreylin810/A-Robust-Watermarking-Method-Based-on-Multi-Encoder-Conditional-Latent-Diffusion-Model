# StageE 正式實驗腳本流程與手動重跑說明

更新日期：2026-07-17

這份文件說明 EXP52 / EXP53 / EXP54 的 StageE 正式實驗腳本實際上做了什麼。重點是讓後續研究者即使不想用一鍵腳本，也能理解每一步要呼叫哪些程式。

## 程式碼本體

目前 O256-PAPER 程式碼本體在：

```bash
/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
```

GitHub 備份工作樹在：

```bash
/home/jeffrey/projects/O256-PAPER-github-backup
```

後續交接建議以 GitHub 版本為準；本機工作目錄是你目前實驗用的原始開發位置。

## 一鍵腳本

正式 StageE 三個 full-from-scratch 實驗分別是：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh
bash scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh
```

先做空間與路徑檢查可用：

```bash
bash scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh --check-only
bash scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh --check-only
bash scripts/run_o256_exp54_full_from_scratch_stageE_seed63_20260714.sh --check-only
```

## 腳本實際流程

以下以 EXP52 為例，EXP53/54 只有 seed、輸出資料夾、config 名稱不同。

### Step 0: 前置檢查

腳本會先確認：

- Python 環境：`/home/jeffrey/miniconda3/envs/diffw/bin/python`
- O256-PAPER 根目錄存在。
- locked CoCo dataset 存在。
- base/eval/refiner template config 存在。
- 硬碟空間至少達到腳本設定門檻。

主要寫在：

```bash
scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh
```

### Step 1: 建立 train alias

腳本會把 formal train split 建成 refiner 訓練時可讀的 alias：

```bash
python scripts/create_coco_split_alias.py
```

目的：讓 refiner pair generation 可以把 train10000 當成 `val` 形式讀取。

### Step 2: 產生 EXP 專用 config

腳本會根據 template config 動態寫出 EXP52 專用 config，例如：

- identity20k
- base30 nongeom-light
- base50 nongeom-light plus20k
- base55 weakattack plus5k
- base60 dropout-focus plus5k
- base65 dropout-margin plus5k
- finaltest1000 eval config
- refiner train-pair config

這些 config 實際放在：

```bash
configs/
```

### Step 3: 訓練 base model 到 65k

腳本會分階段呼叫：

```bash
python train_diffusion.py --config <config.yml> --resume <checkpoint or none> --seed <seed>
```

大致順序是：

1. identity-only 20k
2. nongeom-light 到 30k
3. nongeom-light plus20k 到 50k
4. weakattack plus5k 到 55k
5. dropout-focus plus5k 到 60k
6. dropout-margin plus5k 到 65k

最後得到：

```bash
workspace/base65_dropout_margin_plus5k/ckpts/CoCo_ddpm_final.pth.tar
```

這是 O256 base 的正式 StageE 起點。

### Step 4: 評估 base65

腳本會呼叫：

```bash
python eval_diffusion.py --config <finaltest1000_config> --resume <base65_ckpt> --seed <seed>
```

產生 finaltest1000 的 base output。

### Step 5: 產生 refiner 訓練 pair

腳本會再用 base65 對 train10000 產生 watermarked image / GT pair，供 RGB refiner 訓練。

核心仍是：

```bash
python eval_diffusion.py
```

輸出通常在：

```bash
train_pairs/<exp>_base65_pairs/
```

### Step 6: 訓練 RGB refiner StageA-E

腳本會分階段呼叫：

```bash
python scripts/train_rgb_refinement.py
```

StageE formal protocol 的 refiner 訓練順序是：

| Stage | 目的 | 大致步數 |
|---|---|---:|
| StageA | 先學基本影像修復 | 10000 |
| StageB | 加入 message preservation | 2000 |
| StageB+ | 短補 message preservation | 500 |
| StageC | targeted weak attack 補強 | 500 |
| StageD | dropout-balanced 補強 | 500 |
| StageE | decoder-aware dropout 補強 | 250 |

最後得到：

```bash
refinement/<exp>_stageE_decoderaware250/refiner_final.pth
```

### Step 7: 套用 refiner

腳本會呼叫：

```bash
python scripts/apply_rgb_refinement.py
```

輸入 base finaltest1000 output，輸出 RGB refined images。

### Step 8: 計算 PSNR / SSIM / LPIPS

PSNR / SSIM：

```bash
python calculate_psnr_ssim.py
```

LPIPS：

```bash
python scripts/calculate_lpips_dir.py
```

### Step 9: 做 clean BA 與攻擊 BA

Clean BA：

```bash
python calculate_ba.py
```

Non-geometric attacks：

```bash
python attack_eval_diffusion.py
```

攻擊表包含 Gaussian noise、Salt & Pepper、JPEG、Gaussian filter、Dropout 等正式 non-geometric attack。

### Step 10: 產生 summary

腳本最後呼叫對應 summarizer：

```bash
python scripts/summarize_o256_exp52_full_from_scratch_stageE_seed62_20260708.py
python scripts/summarize_o256_exp53_full_from_scratch_stageE_seed61_20260709.py
python scripts/summarize_o256_exp54_full_from_scratch_stageE_seed63_20260714.py
```

輸出：

```bash
summary.md
summary.csv
per_attack_ba.csv
```

## 不用一鍵腳本時的重點

如果學弟妹不想用一鍵腳本，他們可以照上面順序自己跑，但要注意：

1. 每一階段的 checkpoint 路徑要手動接對。
2. Config 裡的 dataset root / output dir 要改成自己的機器路徑。
3. Refiner 訓練 pair 必須由同一個 base65 產生。
4. BA 評估使用的 message 檔必須和 output image 正確配對。
5. Attack table 要固定，否則不能和論文數字直接比較。

所以一鍵腳本不是黑盒，它只是把這些容易接錯的步驟固定下來。

