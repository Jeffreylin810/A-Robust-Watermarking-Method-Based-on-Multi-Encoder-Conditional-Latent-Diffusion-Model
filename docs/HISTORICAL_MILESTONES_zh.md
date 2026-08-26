# Historical Milestones（中文版）

English version: `docs/HISTORICAL_MILESTONES.md`

## 決策摘要

這個專案的歷史我目前切成四層：

1. 前史探索
2. 早期 paper-side 標準化
3. `2026-06-29` 之後的 O256 主線
4. 最終 StageE formal rerun 與 thesis audit

## 最終版本有沒有被 `speed_up_experiment` 影響？

短答案：**有間接影響，但不是直接祖先。**

- 它影響了早期 paper-side 對 training budget、step matching、效率 framing 的理解。
- 它也出現在舊的 paper-side 筆記與結果整理中。
- 但它不是定義最終 O256 StageE protocol 的主線來源。
- 真正導向最終 StageE 的，是後面的 O256 trade-off / dropout-margin / StageE 這條線。

因此：

- 可以當歷史側註
- 不列為核心交接 milestone

## 應該被提到的前史資料夾

### `/home/jeffrey/projects/Pixel_Bridge_Diffw`

重要原因：

- 記錄了從原始 DiffW 假設走向 latent-native thinking 的轉折
- 有早期弱 watermark signal 的診斷脈絡
- 有助於說明為什麼後來不再沿用 naive latent adaptation

用途：

- 學術交接：要提
- 面試展示：簡短提即可

### `/home/jeffrey/projects/fidelity_tradeoff_experiment`

重要原因：

- 記錄了早期「先救 fidelity，再把 robustness 加回來」的 curriculum 思路
- 幫助說明為什麼後來 protocol 很重視平衡而不是單一指標
- 提供 staged scheduling 重要性的早期證據

用途：

- 學術交接：要提
- 面試展示：可作簡短歷史註解

### `/home/jeffrey/projects/paper`

重要原因：

- 它像是一個 paper-oriented cleanup stage
- 幫助說明 splits、evaluation framing、summary 慣例如何標準化
- 是探索期與較正式 paper-side protocol 之間的橋樑

用途：

- 學術交接：要提
- 面試展示：短註解即可

## 只適合做備註的前史資料夾

### `/home/jeffrey/projects/Pixel_Bridge_Diffw_text_archive_20260516`

重要原因：

- 裡面保留了 message length、injection、bitplane、decoder behavior 等早期文字檔

不列核心的原因：

- 更像 archive，不像整理好的 milestone
- 分支太多，不適合直接當主交接包內容

### `/home/jeffrey/projects/geometry_experiment`

決策：

- 不納入主交接與面試版主線

原因：

- 目前沒有被選為 final StageE 主線不可缺少的轉折點

## `2026-06-29` 之後的核心 milestones

### 基線與問題定義

- `actual10k`
- `repeatability`
- `diffw_epoch7`
- `diffw_60k`

### O256 trade-off 成形

- `exp16` 到 `exp23`
- 其中 `exp23` 很重要，因為它第一次明確顯示 base-side weak attack scheduling 的價值

### Dropout 主線成形

- `exp28`
- `exp33`
- `exp34`
- `exp36`
- `exp37`
- `exp38`

### 正式化與診斷

- `exp40`
- `exp41`
- `exp43`
- `exp44`
- `exp45`
- `exp46`

### StageE 設計與最終驗證

- `exp50`
- `exp51`
- `exp52`
- `exp53`
- `exp54`

## 為什麼最後是 256x256 和 32-bit？

這些較早期資料夾主要要被用來說明：

- 為什麼最後固定在 256x256
- 為什麼 32-bit 成為實際可用的目標 payload
- 為什麼後來偏向 selected non-geometric pressure，而不是把所有攻擊混在一起
- 為什麼有些早期參數設定最後被放棄

這些內容應該被當成歷史動機，而不是最終正式證據。
