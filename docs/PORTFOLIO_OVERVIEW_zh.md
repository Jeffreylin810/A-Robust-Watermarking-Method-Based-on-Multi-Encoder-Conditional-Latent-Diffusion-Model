# Portfolio Overview（中文版）

English version: `docs/PORTFOLIO_OVERVIEW.md`

## 專案摘要

這份 repository 是公開作品集導向的輕量版研究程式碼與實驗紀錄，主題是
在 Stable Diffusion 2.1 VAE latent space 上做強健式影像浮水印。

最終方法是 **O256 StageE**，核心設定包括：

- 256x256 影像
- 32-bit 訊息
- base-side weak/non-geometric attack scheduling
- RGB refinement
- decoder-aware StageE refinement
- formal finaltest1000 evaluation

這個專案的重點不是只追一個漂亮數字，而是追求幾個面向之間可重複的平衡：

- 畫質
- clean decoding accuracy
- 非幾何攻擊下的 worst-case robustness
- 實際部署可行性

## 這份作品集想展示什麼

- 不是只會跑模型，而是會做完整實驗設計。
- 不是只保留成功結果，也保留重要的 negative results。
- 不是靠單一 continuation checkpoint 撐結論，而是有 full from-scratch formal rerun。
- 不是只有 code，還有研究歷程、診斷、表格與可追溯的 summary artifacts。

## 建議閱讀順序

如果你想用最短時間看懂這個專案，建議依序閱讀：

1. `README.md`
2. `docs/HISTORICAL_MILESTONES.md`
3. `results_summaries/stageE_thesis_audit/stageE_thesis_audit_report.md`
4. `results_summaries/exp52/summary.md`
5. `results_summaries/exp53/summary.md`
6. `results_summaries/exp54/summary.md`

## 主結果敘事

公開版本應該用保守且誠實的方式描述：

- `exp52` 與 `exp53` 是最強的 final main results。
- `exp54` 有價值，但應該描述成 seed63 near-miss validation，而不是
  strict-pass main result。
- 這個專案應該被描述成「trade-off 與 protocol design 研究」，而不是
  宣稱全面打敗所有 baseline。

## Repository 會保留什麼

這個公開版 repo 會保留：

- 可執行程式碼
- configs
- formal runner scripts
- 輕量 summary
- research-history 文件
- thesis audit tables

它不會保留：

- dataset
- VAE/model 大檔
- checkpoints
- 大量生成結果資料夾

## 面試官應該看見的重點

- 方法是經過多輪控制變因比較才收斂出來的。
- 有些分支雖然讓某個指標上升，但會破壞別的面向，因此被明確停損。
- 最後的 StageE protocol 是經過完整正式重跑驗證，而不是只靠補丁式續訓。
- 這個 repo 同時保留了最終答案與得到答案的推理過程。
