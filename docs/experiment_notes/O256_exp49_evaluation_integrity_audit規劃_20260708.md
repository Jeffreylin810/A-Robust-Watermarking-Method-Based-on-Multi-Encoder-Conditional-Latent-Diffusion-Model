# O256 exp49 evaluation integrity audit 規劃

Date: 2026-07-08

## 1. 為什麼做

使用者提出一個很重要的疑問：目前 PSNR 與 BA 的評估是否可能因為程式實作而被高估，甚至出現「為了結果好看而操作評估程式」的風險。

因此 exp49 不訓練模型，也不改模型，而是專門檢查評估流程是否可靠。

## 2. 目前評估程式做法

### BA

`calculate_ba.py` 的做法是：

1. 讀取 `attack_dir` 底下每個 attack 子資料夾。
2. 對每張 attacked / clean 圖片，用 checkpoint 裡的 decoder 解碼出 message。
3. 用圖片檔名 stem 去找同名的 `*_message.npy`。
4. 將 decoder output round / clip 成 0/1。
5. 和原始 message 做 bit-wise 比對。
6. 對所有有效圖片取平均。

這個流程不是只挑好圖，也不是直接讀答案；但仍需要 audit 確認 message stem 對應沒有問題。

### PSNR / SSIM

`calculate_psnr_ssim.py` 的做法是：

1. 讀取 results folder 和 GT folder。
2. 對兩邊檔名排序後一一配對。
3. 用 RGB/BGR 全通道計算 PSNR / SSIM。

這裡有一個需要 audit 的點：  
如果兩邊檔名順序完全一致，排序配對是安全的；如果檔名有缺漏或順序錯位，就可能配錯圖。因此 exp49 會另外做 stem-matched PSNR 檢查。

## 3. exp49 檢查項目

### PSNR stem matching audit

對每個輸出資料夾：

- 檢查 pred 圖片數量
- 檢查 GT 圖片數量
- 檢查 pred stem 是否和 GT stem 完全一致
- 用 stem-to-stem 方式重新算 PSNR
- 同時計算原本 sorted-order pairing 的 PSNR
- 比較兩者差異是否接近 0

如果差異明顯，代表舊 PSNR 評估有配對風險。

### PSNR wrong-GT negative control

故意把 GT 圖片 shift 一格後再算 PSNR。  
如果正確 PSNR 高，但 wrong-GT PSNR 明顯降低，代表 PSNR 不是靠錯誤配對或資料洩漏撐出來。

### BA wrong-message negative control

對同一批圖片：

- 正常：用同名 `*_message.npy` 計算 BA
- 負控制：故意把 message shift 一格再計算 BA

如果正常 BA 高，但 wrong-message BA 接近隨機約 50%，代表 BA 不是因為 decoder 偏向某個固定輸出或 message 配對漏洞造成。

## 4. 會 audit 的樣本

| sample | 用途 |
|---|---|
| exp40_base_clean | seed61 base clean BA / PSNR audit |
| exp40_rgb_clean | seed61 RGB clean BA / PSNR audit |
| exp40_rgb_dropout | seed61 RGB Dropout BA negative-control audit |
| exp41_base_clean | seed62 base clean BA / PSNR audit |
| exp41_rgb_clean | seed62 RGB clean BA / PSNR audit |
| exp41_rgb_dropout | seed62 RGB Dropout BA negative-control audit |

## 5. 輸出

Experiment root:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/49_supplementary_o256_evaluation_integrity_audit_20260708
```

輸出：

- `audit/summary.md`
- `audit/evaluation_integrity_audit.csv`
- `audit/evaluation_integrity_audit.json`

## 6. 執行方式

已通過 check-only：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp49_evaluation_integrity_audit_20260708.sh --check-only
```

正式執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_o256_exp49_evaluation_integrity_audit_20260708.sh
```

快速測試：

```bash
bash scripts/run_o256_exp49_evaluation_integrity_audit_20260708.sh --quick 100
```

## 7. 預期判讀

如果評估可靠，應該看到：

- pred / GT / message 都是 1000。
- stem set 完全一致。
- stem-matched PSNR 和 sorted-order PSNR 幾乎相同。
- wrong-GT PSNR 明顯低於正確 PSNR。
- wrong-message BA 約接近 50%。
- correct BA 和 wrong-message BA 有很大 gap。

如果以上任一項不成立，就必須回頭修正評估流程，不能直接使用目前結果。

