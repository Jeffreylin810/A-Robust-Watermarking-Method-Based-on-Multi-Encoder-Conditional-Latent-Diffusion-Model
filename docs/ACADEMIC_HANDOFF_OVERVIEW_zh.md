# Academic Handoff Overview（中文版）

English version: `docs/ACADEMIC_HANDOFF_OVERVIEW.md`

## 目的

這份文件是學術交接導向的入口，用來給實驗室後續研究者接手。

交接包有兩個主要目標：

1. 保存最終 StageE protocol 與它的正式證據。
2. 保存足夠的研究歷程，讓後續研究者在主線卡住時可以回頭重建思路。

這份交接不是要保存每一個探索分支，而是保存真正改變研究方向的分支。

## 核心交接邊界

學術交接包保留：

- final StageE code 與 configs
- exp52 / exp53 / exp54 的正式 summary
- thesis audit 與 thesis-ready tables
- 能解釋 protocol 為何長成現在這樣的歷史里程碑
- path/setup 說明

學術交接包不保留：

- 每個 raw experiment folder
- 所有暫時 checkpoint
- 全部生成圖檔
- 正式化前的所有探索分支

## 里程碑分級

### 1. 核心 milestones

這些是主線，後續研究者應該最容易回頭閱讀：

- `actual10k`, `repeatability`
- `diffw_epoch7`, `diffw_60k`
- `exp16` 到 `exp54`

其中尤其重要的有：

- `exp23`
- `exp28`
- `exp33` / `exp34`
- `exp37` / `exp38`
- `exp40` / `exp41`
- `exp43`
- `exp46`
- `exp50` / `exp51`
- `exp52` / `exp53` / `exp54`

### 2. 前史敘事保留，但不保留 checkpoint

這些較早期資料夾對於說明「為什麼最後變成 256x256、32-bit、這種訓練邏輯」
很重要，但不需要把 checkpoint 帶進 NAS：

- `/home/jeffrey/projects/Pixel_Bridge_Diffw`
- `/home/jeffrey/projects/fidelity_tradeoff_experiment`
- `/home/jeffrey/projects/paper`

### 3. 前史備註候選

這些更像 archive 參考來源，而不是核心交接內容：

- `/home/jeffrey/projects/Pixel_Bridge_Diffw_text_archive_20260516`

## 已對齊的資料夾決策

- `/home/jeffrey/projects/geometry_experiment`
  - 不納入主交接敘事。
- `/home/jeffrey/projects/speed_up_experiment`
  - 視為前史側註，不列為最終 StageE 的直接祖先 milestone。

## NAS 打包政策

### Minimal NAS package

保留：

- SD2.1 / VAE backup
- final formal summaries
- thesis audit tables
- selected historical summaries
- key setup documents

### Full NAS package

在 minimal 之上增加：

- selected final checkpoints
- curated StageE support artifacts
- additional curated milestone summaries

依目前政策，正式化前 checkpoint 不納入 NAS。

VAE backup 在學術交接版中視為必要資產，不是可有可無的公開版附加物。

## 建議的恢復路徑

如果後續研究者卡住，不要第一步就回去 full StageE retraining。建議順序是：

1. 先讀 `docs/HISTORICAL_MILESTONES.md`
2. 再讀 `docs/research_history/`
3. 回頭看 `exp43` 到 `exp46` 的 diagnosis
4. 回頭看 `exp50` / `exp51` 的 StageE 設計證據
5. 最後才決定要改 base-side scheduling、refiner behavior 還是 evaluation scope
