# Training-side 口試答法與最終總表（2026-06-29）

## 一、最終總表

> 說明：  
> - `formal DiffW-like` 與 `O256-RGB formal` 為正式主線結果。  
> - `epoch7` 與 `epoch12 / 60k` 為 supplementary actual-training variants。  
> - `O256-RGB formal` 的時間使用 `complete preparation total`，包含 base training 與 RGB refinement module preparation。  
> - `epoch7` 與 `epoch12 / 60k` 的時間是單條 variant 實際訓練時間。

| Method / Variant | Setting | PSNR | SSIM | LPIPS | Clean BA | Nongeom Avg BA | Min BA | Time | 備註 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| formal DiffW-like | 20k / formal baseline | 26.4236 | 0.8362 | 0.2060 | 100.00 | 99.35 | 95.47 | 02:25:31.24 | robustness-oriented baseline |
| DiffW-like epoch7 true-branch | `combined_attack=False`, `n_epochs=7` | 28.7506 | 0.9169 | 0.1043 | 100.00 | 94.76 | 70.00 | 04:15:42 | 第一次真正進入 `epoch > 5` branch |
| DiffW-like epoch12 / 60k | `combined_attack=False`, `n_epochs=12` | 37.4562 | 0.9732 | 0.0088 | 100.00 | 91.91 | 68.18 | 07:18:03 | 更高畫質，但 worst-case robustness 再下降 |
| O256-RGB formal | fixed recipe / formal full pipeline | 32.9034 | 0.9589 | 0.0461 | 95.81 | 87.38 | 72.93 | 02:33:57.31 | complete preparation total |

## 二、這張表最重要的讀法

### 1. formal DiffW-like 與 O256-RGB formal 的差別

- formal DiffW-like：
  - robustness 很強
  - 畫質較低
- O256-RGB formal：
  - 畫質明顯更高
  - average BA 與 minimum BA 較低
  - 但仍維持可接受的 robustness

### 2. `epoch7` 的意義

- 證明 local DiffW-like 的 trade-off 不是完全卡死
- 真正進入 `epoch > 5` branch 之後，畫質會上升
- 但還沒有接近 O256-RGB formal 的品質門檻

### 3. `epoch12 / 60k` 的意義

- 證明 local DiffW-like **可以**被推到更高畫質 operating point
- 而且不只是接近 O256-RGB，甚至在 PSNR / SSIM / LPIPS 上更高
- 但代價是：
  - 需要更長訓練（`07:18:03`）
  - 需要 branch-level tuning
  - worst-case robustness（Min BA）下降到 `68.18`

## 三、更新後的 training-side 核心說法

這條 60k 新結果出來後，training-side 最穩的說法不再是：

> O256-RGB 用更少訓練時間跑出更好的結果。

這句現在太強，不建議再講。

更穩的說法應該改成：

> 在本文統一設定下，O256-RGB 的 training-side 優勢不在於全面成本壓過 DiffW-like，而在於它能以固定 recipe 更直接地到達論文主結果所需要的 high-quality operating point；相對地，DiffW-like 雖然可以透過更長訓練與額外 branch tuning 被推到更高畫質區，但這需要更高的額外探索成本，且 worst-case robustness 會先變脆。

## 四、口試現場比較口語的 4 句版

1. 如果只看 training-side，我現在不會把 O256-RGB 講成全面比較省，因為 DiffW-like 確實可以靠更長訓練和額外 branch tuning 被推到很高畫質。  
2. 但我這邊比較想強調的是，O256-RGB 的優勢是它比較直接到位，用固定 recipe 就能穩定到我論文正式主結果那個品質點。  
3. 相對地，DiffW-like 若要往同樣方向推，目前看起來需要更長時間和更多訓練探索，而且它最脆弱 attack 下的 robustness 會先掉。  
4. 所以 training-side 我最有信心的主張不是「全面成本優勢」，而是 O256-RGB 比較像一條可直接交付、可重現、較少額外摸索的 high-quality recipe。  

## 五、更像現場回答的版本

> 我覺得這邊最值得講的，不是我在所有訓練成本上都贏。  
> 比較準確的是，我這個方法比較容易直接到我要的正式結果點。  
> DiffW-like 後來我們確實證明可以再往高畫質推，但它是靠更長訓練和額外 branch tuning 才走到那裡，不是原本固定 recipe 自然就到。  
> 所以如果從研究和實作角度看，我的方法在 training-side 的優勢，比較像是 directness、reproducibility，還有較低的額外探索負擔。  

## 六、答辯時建議避免的過強說法

不要再直接講：

- `O256-RGB 用更少訓練時間跑出更好結果`
- `O256-RGB training cost 已全面優於 DiffW-like`
- `DiffW-like 就算多訓練也追不上`

可以改講：

- `O256-RGB 在固定 recipe 下更直接到達論文主結果需要的品質點`
- `DiffW-like 若要往更高畫質方向推，目前需要更長訓練與額外 branch-level tuning`
- `training-side 的優勢比較像 fixed-recipe efficiency，而不是 universal cost superiority`
