# Thesis-ready Tables for StageE

## Table A: Final main results

| 方法/實驗 | 類型 | Seed | PSNR | SSIM | LPIPS | Clean BA | Nongeom Avg BA | Min BA | Min Attack | Dropout BA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Original O256-RGB baseline | formal baseline | 61 | 32.9034 | 0.9589 | 0.046085 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 72.93 |
| StageD exp40 | formal full scratch | 61 | 33.9372 | 0.9819 | 0.040887 | 99.96 | 98.77 | 91.01 | Paper_Dropout_r03 | 91.01 |
| StageD exp41 | formal full scratch | 62 | 34.0293 | 0.9820 | 0.044326 | 99.98 | 98.68 | 89.94 | Paper_Dropout_r03 | 89.94 |
| StageE exp52 | formal full scratch strict pass | 62 | 34.0379 | 0.9825 | 0.045133 | 99.99 | 98.76 | 90.33 | Paper_Dropout_r03 | 90.33 |
| StageE exp53 | formal full scratch strict pass | 61 | 33.9522 | 0.9825 | 0.040024 | 99.85 | 98.58 | 91.36 | Paper_Dropout_r03 | 91.36 |
| StageE exp54 | formal full scratch near-miss | 63 | 33.4948 | 0.9815 | 0.044652 | 99.94 | 98.35 | 89.98 | Paper_Dropout_r03 | 89.98 |

Table A source notes:
- Original O256-RGB baseline: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/17_supplementary_o256_rgb_refiner_refresh_plus10k_seed61_20260630/summary.csv` row `formal_rgb`; the same values are reused as the historical formal O256-RGB reference in later summaries.
- StageD exp40 / exp41: their own `summary.csv` files under experiments 40 and 41.
- StageE exp52 / exp53 / exp54: their own `summary.csv` and `per_attack_ba.csv` files under experiments 52, 53, and 54.

## Table B: StageE per-attack BA

| Attack | exp52 BA | exp53 BA | exp54 BA |
| --- | ---: | ---: | ---: |
| Paper_Dropout_r03 | 90.33 | 91.36 | 89.98 |
| Paper_GaussianBlur_sigma2 | 100.00 | 99.91 | 99.95 |
| Paper_GaussianBlur_sigma3 | 99.99 | 99.91 | 99.95 |
| Paper_GaussianBlur_sigma4 | 99.99 | 99.91 | 99.95 |
| Paper_GaussianNoise_arg03 | 98.07 | 97.12 | 96.33 |
| Paper_GaussianNoise_sigma03_var009 | 99.67 | 99.38 | 99.18 |
| Paper_JPEG_Q30 | 99.88 | 99.62 | 99.55 |
| Paper_JPEG_Q40 | 99.92 | 99.71 | 99.74 |
| Paper_JPEG_Q50 | 99.96 | 99.77 | 99.82 |
| Paper_MedianFilter_w5 | 99.99 | 99.93 | 99.93 |
| Paper_SaltPepper_r02 | 98.58 | 97.74 | 97.45 |

## Table C: StageE design evolution summary

| 實驗區塊 | 代表實驗 | 論文定位 | 主要觀察 | 是否可作正式主結果 |
| --- | --- | --- | --- | --- |
| Original O256-RGB baseline | M03 original O256-RGB | 舊正式 baseline | 畫質改善，但 worst-case BA 72.93 不足 | 可作 historical baseline |
| Base-side robustness exploration | exp16-exp23 | 探索/設計依據 | weak attack pressure 放入 base-side 後 BA floor 明顯改善 | 不作 final main result |
| StageD | exp40 / exp41 | 正式候選 | seed61 通過，seed62 Dropout 89.94 差 0.06pp | exp40 可作候選；exp41 說明 boundary |
| StageE diagnosis | exp48-exp51 | 診斷/設計依據 | decoder-aware refiner 可補 seed62 Dropout floor | 不作 final main result |
| StageE final formal results | exp52 / exp53 | 最終主結果 | 兩個 full from-scratch seeds 通過 strict gates | 是 |
| Seed63 validation | exp54 | 補充驗證 | Clean/Avg BA 強，但 PSNR 33.4948 與 Dropout/Min BA 89.98 near-miss | 不作 strict-pass main result |

## Table D: Cost / inference summary

### Version 1: StageE training cost complete version

| Experiment | Seed | Base training | Pair generation | Refiner training | Total prep | Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| exp52 | 62 | 03:47:16 | 00:15:39 | 00:46:30 | 04:49:25 | formal_log |
| exp53 | 61 | 03:48:32 | 00:16:02 | 00:47:09 | 04:51:42 | formal_log |
| exp54 | 63 | 03:45:28 | 00:15:25 | 00:46:21 | 04:47:14 | formal_log |

### Version 2: Conservative cost version

StageE training logs are complete for exp52/53/54, so Version 1 can be used. If the thesis avoids training-cost claims, use the following inference-only table.

| Method | N | Total time (s) | ms/image | img/s | Speedup vs DiffW-like | Overhead vs O256-Base | Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DiffW-like | 1000 | 399.85 | 399.850000 | 2.500938 | 1.000000 | 4.085939 | formal_timing_baseline |
| O256-Base | 1000 | 97.86 | 97.860000 | 10.218680 | 4.085939 | 1.000000 | formal_timing_baseline |
| O256-RGB | 1000 | 159.09 | 159.090000 | 6.285750 | 2.513357 | 1.625690 | formal_timing_baseline |
| O256-StageE-exp52 | 1000 | 164.96 | 164.96 | 6.062076 | 2.423921 | 1.685673 | partial_formal_log |
| O256-StageE-exp53 | 1000 | 165.77 | 165.77 | 6.032455 | 2.412077 | 1.693951 | partial_formal_log |
| O256-StageE-exp54 | 1000 | 160.35 | 160.35 | 6.236358 | 2.493608 | 1.638565 | partial_formal_log |
