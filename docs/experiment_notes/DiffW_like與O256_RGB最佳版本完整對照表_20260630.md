# DiffW-like 與 O256-RGB 最佳版本完整對照表（2026-06-30）

用途：整理目前兩條方法線上，**最值得主打** 的版本，以及其訓練時間、整體 BA 與完整 per-attack BA。

## 版本選擇原則

### DiffW-like
目前最值得對照的是：

- **`epoch12 / 60k`**

理由：
- 它是 actual-training、不是 post-hoc gamma；
- 畫質已推到很高（PSNR `37.4562`）；
- Clean BA 與 Avg BA 仍很強；
- 能代表「DiffW-like 經較長訓練與 branch tuning 後」的高品質 operating point。

### O256-RGB
目前最值得主打的是：

- **`plus15k_refresh500`**

理由：
- 它不是純 robustness 版，而是目前最均衡的高畫質 + 高 BA 版本；
- 相較 formal RGB，畫質與 BA 都同時提升；
- 相較 `plus15k_rgb_reuse`，畫質明顯更好，而 BA 仍維持很強。

> 補充：若只看純 robustness ceiling，O256 目前最強的是 `plus15k_rgb_reuse`，但本表主打的是「最適合答辯主線」的版本。

## 核心總表

| Method | Variant | PSNR | SSIM | LPIPS | Clean BA | Avg BA | Min BA | Min attack | 訓練時間 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| DiffW-like | `epoch12 / 60k` | 37.4562 | 0.9732 | 0.0088 | 100.00 | 91.91 | 68.18 | Paper_GaussianNoise_arg03 | 07:18:03 |
| O256-RGB | `plus15k_refresh500` | 33.5967 | 0.9718 | 0.0410 | 99.34 | 95.76 | 83.48 | Paper_Dropout_r03 | 04:03:49 *(從頭總成本)* |

## 完整 BA 對照表

| Attack | DiffW-like `epoch12 / 60k` | O256-RGB `plus15k_refresh500` |
|---|---:|---:|
| Paper_Dropout_r03 | 99.74 | 83.48 |
| Paper_GaussianBlur_sigma2 | 100.00 | 99.52 |
| Paper_GaussianBlur_sigma3 | 100.00 | 99.43 |
| Paper_GaussianBlur_sigma4 | 100.00 | 99.39 |
| Paper_GaussianNoise_arg03 | 68.18 | 89.88 |
| Paper_GaussianNoise_sigma03_var009 | 89.78 | 96.23 |
| Paper_JPEG_Q30 | 89.47 | 97.58 |
| Paper_JPEG_Q40 | 92.33 | 98.31 |
| Paper_JPEG_Q50 | 94.36 | 98.68 |
| Paper_MedianFilter_w5 | 100.00 | 99.38 |
| Paper_SaltPepper_r02 | 77.13 | 91.53 |

## 一句話解讀

- **DiffW-like `epoch12 / 60k`**：畫質非常高、clean BA 幾乎滿分，但最脆弱 attack 下的 minimum robustness 較弱，且訓練時間更長。
- **O256-RGB `plus15k_refresh500`**：畫質已很高，同時在 noise / JPEG / salt-pepper 這些較脆弱攻擊下顯得更均衡，且整體訓練成本低於高品質 DiffW-like 60k 版本。

## 口試時最穩的說法

> 若把比較對象放在「各自已推到高品質、且 BA 仍可接受的版本」，那目前 DiffW-like 最值得對照的是 `epoch12 / 60k`，而 O256-RGB 最值得主打的是 `plus15k_refresh500`。前者畫質更高，但 worst-case robustness 較弱且訓練時間更長；後者則提供更均衡的 BA 分布，並以較短的總訓練成本達到高畫質結果。
