# Package Targets（中文版）

English version: `docs/PACKAGE_TARGETS.md`

## 目標

這個專案最後要整理成 **兩個不同用途的資料夾**，不是只做一個版本：

1. 一個可直接在本機整理後推上 public GitHub 的資料夾
2. 一個可直接上傳到實驗室伺服器的較完整交接資料夾

這個切法就是目前已對齊的正式策略。

## 目標 1：Public GitHub 資料夾

建議名稱：

`O256-github-public`

主要用途：

- 公開作品集
- 乾淨的可重現敘事
- 輕量程式碼 repository

應包含：

- code
- configs
- scripts
- lightweight summaries
- research-history notes
- `LICENSE`
- `docs/THIRD_PARTY_AND_LICENSE_NOTES.md`
- public release checklists

不應包含：

- dataset
- SD2.1 / VAE 大檔
- checkpoints
- 大量生成輸出
- raw attack folders

## 目標 2：實驗室伺服器交接資料夾

建議名稱：

`O256-lab-handoff-full`

主要用途：

- 學術交接
- 實驗室後續重用
- 較完整的恢復包

應包含：

- public GitHub 版本中所有必要內容
- SD2.1 / VAE backup
- final formal summaries
- thesis audit tables
- selected historical milestone notes
- 若值得保留則放 selected final checkpoints

這個版本不用以公開版的授權包裝為主要考量，重點是完整度與交接可用性。

對齊你的要求：

- 實驗室版本預期必須包含 VAE backup
- public GitHub 版本預期不包含 VAE backup

## 目前已對齊的實驗室版政策

- 正式前 checkpoints：不放
- `geometry_experiment`：不放
- `speed_up_experiment`：只作歷史側註
- `Pixel_Bridge_Diffw`：作為前史重要節點敘事保留
- `fidelity_tradeoff_experiment`：作為前史重要節點敘事保留
- `paper`：作為前史重要節點敘事保留

## 兩個目標之間的關係

public GitHub 版本是比較乾淨的子集合。

實驗室伺服器版本是比較完整的學術交接包。

公開版應該讓面試官容易閱讀。
實驗室版應該讓學弟妹真的接得下去。
