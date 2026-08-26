# StageE 第四章修訂完整資料包 for GPT

- Generated from audit folder: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2`
- Output file: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/StageE_第四章修訂完整資料包_for_GPT_20260714_v2.md`
- Scope: 資料整合；未重跑實驗、未新增訓練、未修改論文正文。
- Important: 以下內容只根據既有 audit/log/summary/csv/script 整理；缺資料以 `MISSING` 標示。


## 0. Executive Summary

- exp52 / exp53：可作為 final main results；兩者都是 full from-scratch formal rerun，且 strict pass。
- exp54：已完成，但不建議列為 strict-pass 主結果；狀態為 completed near-miss: PSNR 33.4948, Dropout/Min BA 89.98，建議放補充驗證或 near-miss row。
- 第三章：不需要重寫 core O256 latent watermark 公式；建議小幅補充 StageE RGB refiner objective 與 decoder-aware 訓練文字。
- 4.3：可保留 per-attack 表，且應強調 Paper_Dropout_r03 / worst-case BA 分析。
- 4.5：StageE training cost 有 formal log，可更新訓練準備成本表；StageE end-to-end timing 有 partial formal log，可保守使用。
- StageE compute-only timing：MISSING，不能用舊 O256-RGB compute-only timing 直接代替 StageE。
- StageE end-to-end timing：available，包含 data loading / model apply / output writing，不含 PSNR/SSIM/LPIPS/BA/attack evaluation。
- 缺失資料：StageE-specific compute-only timing；StageE training/apply logs 未印出 GPU；若論文要嚴格寫 compute-only cost，需要補測。
- 不可寫：不可宣稱 O256 StageE 全面優於 DiffW-like；不可把 exp54 寫成 strict pass；不可把 continuation 寫成 full-from-scratch；不可把 Clean BA 納入 Nongeom Avg BA。

Source files: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_thesis_audit_report.md`, `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_metrics_validation.md`


## 1. Final Main Results Summary

Source files:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_final_metrics.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/thesis_ready_tables_stageE.md`
| 方法/實驗 | 類型 | Seed | PSNR | SSIM | LPIPS | Clean BA | Nongeom Avg BA | Min BA | Min Attack | Dropout BA | 是否可作正式主結果 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original O256-RGB baseline | formal baseline | 61 | 32.9034 | 0.9589 | 0.046085 | 95.81 | 87.38 | 72.93 | Paper_Dropout_r03 | 72.93 | baseline only | 保留作 historical baseline；來源為 exp17 summary.csv row formal_rgb。 |
| StageD exp40 | formal full scratch | 61 | 33.9372 | 0.9819 | 0.040887 | 99.96 | 98.77 | 91.01 | Paper_Dropout_r03 | 91.01 | candidate / design reference | StageD seed61 strict pass，可作 StageE 前一版對照。 |
| StageD exp41 | formal full scratch | 62 | 34.0293 | 0.9820 | 0.044326 | 99.98 | 98.68 | 89.94 | Paper_Dropout_r03 | 89.94 | boundary reference | Dropout/Min BA 差 90% strict gate 0.06pp，說明 StageE 的必要性。 |
| StageE exp52 | formal full scratch strict pass | 62 | 34.0379 | 0.9825 | 0.045133 | 99.99 | 98.76 | 90.33 | Paper_Dropout_r03 | 90.33 | yes | full from-scratch formal rerun；strict pass。 |
| StageE exp53 | formal full scratch strict pass | 61 | 33.9522 | 0.9825 | 0.040024 | 99.85 | 98.58 | 91.36 | Paper_Dropout_r03 | 91.36 | yes | full from-scratch formal rerun；strict pass。 |
| StageE exp54 | formal full scratch near-miss | 63 | 33.4948 | 0.9815 | 0.044652 | 99.94 | 98.35 | 89.98 | Paper_Dropout_r03 | 89.98 | no, supplemental near-miss | 已完成 seed63 formal rerun，但 PSNR 33.4948 與 Dropout/Min BA 89.98 未過 strict gate。 |


## 2. StageE Per-Attack BA

Source files:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_per_attack_ba.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/formal_attack_name_mapping.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/thesis_ready_tables_stageE.md`
| paper display name | raw log name | exp52 BA | exp53 BA | exp54 BA | note | source file |
| --- | --- | --- | --- | --- | --- | --- |
| Paper_Dropout_r03 | Paper_Dropout_r03 | 90.33 | 91.36 | 89.98 | Min BA attack for exp52/exp53/exp54; exp54 near-miss: 89.98 | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianBlur sigma2 | Paper_GaussianBlur_sigma2 | 100.00 | 99.91 | 99.95 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianBlur sigma3 | Paper_GaussianBlur_sigma3 | 99.99 | 99.91 | 99.95 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianBlur sigma4 | Paper_GaussianBlur_sigma4 | 99.99 | 99.91 | 99.95 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianNoise additive arg0.3 | Paper_GaussianNoise_arg03 | 98.07 | 97.12 | 96.33 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianNoise sigma0.3 / var0.09 | Paper_GaussianNoise_sigma03_var009 | 99.67 | 99.38 | 99.18 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| JPEG30 | Paper_JPEG_Q30 | 99.88 | 99.62 | 99.55 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| JPEG40 | Paper_JPEG_Q40 | 99.92 | 99.71 | 99.74 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| JPEG50 | Paper_JPEG_Q50 | 99.96 | 99.77 | 99.82 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| MedianFilter 5 | Paper_MedianFilter_w5 | 99.99 | 99.93 | 99.93 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| SaltPepper 0.2 | Paper_SaltPepper_r02 | 98.58 | 97.74 | 97.45 | - | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |


## 3. Formal Attack Set Validation

Source files:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/formal_attack_name_mapping.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/formal_attack_set_validation.md`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_metrics_validation.md`

### 3.1 Formal attack name mapping

| paper display name | raw log name | attack type | parameter | used exp52 | used exp53 | used exp54 | source file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper_Dropout_r03 | Paper_Dropout_r03 | dropout | r=0.3 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianBlur sigma2 | Paper_GaussianBlur_sigma2 | gaussian_blur | sigma=2 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianBlur sigma3 | Paper_GaussianBlur_sigma3 | gaussian_blur | sigma=3 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianBlur sigma4 | Paper_GaussianBlur_sigma4 | gaussian_blur | sigma=4 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianNoise additive arg0.3 | Paper_GaussianNoise_arg03 | gaussian_noise | GN argument=0.3 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| GaussianNoise sigma0.3 / var0.09 | Paper_GaussianNoise_sigma03_var009 | gaussian_noise | GN argument=0.09 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| JPEG30 | Paper_JPEG_Q30 | jpeg | quality=30 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| JPEG40 | Paper_JPEG_Q40 | jpeg | quality=40 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| JPEG50 | Paper_JPEG_Q50 | jpeg | quality=50 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| MedianFilter 5 | Paper_MedianFilter_w5 | median_filter | window=5 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |
| SaltPepper 0.2 | Paper_SaltPepper_r02 | salt_pepper | r=0.2 | yes | yes | yes | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104 |

### 3.2 Attack set validation notes

# Formal Attack Set Validation

- Source script: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104`.
- exp52 / exp53 / exp54 use exactly the same attack set: `yes`.
- All three completed StageE runs contain 11 formal non-geometric attacks.
- The raw log names match the current paper attack list after display-name mapping.
- No name mismatch requiring a metric correction was found.
- Recommended thesis action: keep the 4.1 attack setting table, but use the raw-name mapping table if the thesis uses friendlier display names.

### 3.3 Metrics validation notes

# StageE Metrics Validation

## Checks

### exp52
- per-attack count: `11`.
- Avg BA computed from 11 attacks only: `98.761818`.
- Clean BA is stored separately and is not included in Nongeom Avg BA.
- Min BA from per-attack table: `90.33` at `Paper_Dropout_r03`.
- Summary Min BA / attack: `90.33` / `Paper_Dropout_r03`.
- Status: can be used as final main result under the locked StageE criteria.

### exp53
- per-attack count: `11`.
- Avg BA computed from 11 attacks only: `98.578182`.
- Clean BA is stored separately and is not included in Nongeom Avg BA.
- Min BA from per-attack table: `91.36` at `Paper_Dropout_r03`.
- Summary Min BA / attack: `91.36` / `Paper_Dropout_r03`.
- Status: can be used as final main result under the locked StageE criteria.

### exp54
- per-attack count: `11`.
- Avg BA computed from 11 attacks only: `98.348182`.
- Clean BA is stored separately and is not included in Nongeom Avg BA.
- Min BA from per-attack table: `89.98` at `Paper_Dropout_r03`.
- Summary Min BA / attack: `89.98` / `Paper_Dropout_r03`.
- Status: completed full-from-scratch seed63 validation, but near-miss: PSNR 33.4948 and Dropout/Min BA 89.98 miss the strict gates.

Summary: exp52 / exp53 / exp54 均使用同一組 11 種 formal non-geometric attacks；Nongeom Avg BA 排除 Clean BA；BA denominator 依 formal setting 為 1000 images × 32 bits。4.1 攻擊設定表原則上可保留，但若論文用友善名稱，建議附 raw name mapping。


## 4. StageE Method Delta

Source files:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_method_delta.md`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_loss_and_training_protocol.csv`

### 4.1 Direct answers

- StageE 相對原始 O256-RGB baseline：仍是 O256 base + RGB residual refiner，但加入更明確的 decoder-aware / attack-aware refiner training protocol。
- StageE 相對 StageD：在 StageD dropout-balanced 後追加 decoderaware_e 250 steps，調整 message / attack-message / sensitive / quality-gate 權重。
- 是否新增 loss：是，對 RGB refiner objective 明確加入/使用 clean message loss 與 attack message loss；core base latent formula 不重寫。
- 是否使用 decoder-aware refiner：是。
- 是否使用 frozen message decoder D_phi：是，依 `train_rgb_refinement.py` 的 frozen decoder path。
- 是否包含 clean message loss：是。
- 是否包含 attack message loss：是，針對 dropout_r03_only proxy。
- 是否包含 bit-level BCE：否，目前 evidence 指向 MSE message loss，不是 BCE。
- 是否改變 RGB refiner architecture：否，ResidualRefiner architecture 不變。
- 是否改變 base-side attack schedule：StageE formal protocol 延續 StageD 的 base-side dropout-margin schedule；StageE 本身主要改 refiner final stage。
- 第三章是否需要修改公式：建議小幅新增 RGB refiner objective 公式或文字，不需要推翻第三章主方法。
- 建議修改位置：第三章 RGB refinement / training objective 小節；理由是 StageE final protocol 已把 decoder-aware clean/attack message preservation 寫成最終訓練協定的一部分。

### 4.2 Method delta source document

# StageE Method Delta

## Section A: 原始 O256-RGB baseline 的方法與訓練流程

- 原始 O256-RGB 由 O256 latent/base watermark model 產生 base watermarked image，再接一個 RGB residual refiner 改善 GT-level fidelity。
- RGB refiner 的角色不是重新訓練 base watermark model，而是在 RGB 空間學習小幅殘差修正。
- refiner 程式註解明確寫出：`refined = watermarked + delta(watermarked, cover)`，且 loss 結合 RGB fidelity 與 frozen O256 decoder 的 message recoverability。

## Section B: StageD 相對 baseline 的差異

- StageD 的關鍵是把 `Paper_Dropout_r03` 的壓力導入 base-side attack schedule 與最後的 dropout-balanced refiner stage。
- StageD full from-scratch exp40 / exp41 顯示整體有效，但 exp41 seed62 的 Dropout/Min BA 為 89.94，距離 90 strict gate 還差 0.06pp。
- StageD 沒有改 refiner architecture，主要是訓練 protocol / loss weights / attack pressure 的變化。

## Section C: StageE 相對 StageD 的差異

- StageE 在 StageD 之後追加 `decoderaware_e` 250 steps。
- StageE 修改 loss weights：`message_weight=1.50`、`attack_message_weight=0.65`、`sensitive_residual_weight=3.00`、`quality_gate_psnr=33.80`、`quality_gate_weight=90.0`。
- StageE 使用 `attack_profile=dropout_r03_only` 與 `attack_dropout_prob=0.70`，針對 formal weakest attack 對齊。
- StageE 使用 frozen O256 decoder `D_phi` 計算 clean refined image 和 attacked refined image 的 message loss。
- 目前實作使用 MSE message loss，不是 bit-level BCE。

## Section D: StageE 是否為最終方法、最終訓練協定，或兩者皆是

- 建議寫成：StageE 是本文最終 O256-RGB training protocol 的 final refiner stage。
- 架構上仍是 O256 base + residual RGB refiner，因此不宜寫成完全新架構。
- 若論文將 training protocol 視為方法的一部分，StageE 可列為 final method protocol；但文字要避免暗示 base architecture 被改掉。

## Section E: 是否需要修改第三章公式

- 需要小幅補充，不需要重寫 core O256 latent watermark formulation。
- 建議新增 RGB refiner objective，說明 decoder-aware clean/attack message preservation loss。

## Section F: 若需修改第三章，應修改哪些段落與公式

建議在第三章 RGB refinement 小節補以下公式：

```text
L_ref = lambda_mse L_mse(I_r, I_gt)
      + lambda_l1 L_l1(I_r, I_gt)
      + lambda_msg ||D_phi(E(I_r)) - m||_2^2
      + lambda_atk ||D_phi(E(A(I_r))) - m||_2^2
      + lambda_reg ||I_r - I_w||_1
      + lambda_lat L_latent
      + lambda_sens L_sensitive
      + lambda_q L_quality_gate
```

其中 `I_w` 為 base watermarked image，`I_r` 為 refined image，`I_gt` 為 cover image，`m` 為 32-bit message，`E` 為 frozen VAE encoder path，`D_phi` 為 frozen message decoder，`A` 為 training-only dropout proxy。

## Section G: 來源檔案與證據列表

- Refiner role and objective description: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:1-11`.
- Refiner architecture: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:258-284`.
- CLI loss/attack arguments: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:287-390`.
- Frozen base model/VAE: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:449-461`.
- Frozen decoder message loss, MSE not BCE: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:471-484`.
- Clean and attacked message-loss composition: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:681-761`.
- StageD vs StageE weights: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh:257-327`.
- Stage sequence A/B/B+/C/D/E: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh:493-498`.
- Formal attack set: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/attack_eval_diffw_paper_matched.py:71-104`.

### 4.3 Loss and training protocol table

| stage | experiment | loss terms | loss weights | attack schedule | refiner type | decoder used | clean branch | attack branch | architecture changed | source file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original O256-RGB | original formal O256-RGB baseline | RGB fidelity + clean message recoverability + regularization | see original RGB refiner recipe; not re-derived in this audit | original formal O256 base/refiner protocol before dropout-margin | ResidualRefiner CNN | yes; frozen O256 decoder for message loss | yes | limited/targeted in earlier Stage C recipe | no | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:1-11; /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:258-284 |
| StageD | exp40/exp41 | RGB MSE/L1 + clean message MSE + dropout attacked-message MSE + latent preserve + sensitive residual + quality gate + regularization | image_mse=8.0; image_l1=2.0; message=1.4; attack_message=0.45; latent=0.45; sensitive=2.5; quality_gate_psnr=33.00; quality_gate=70.0; regularize=0.01 | base-side dropout-margin protocol plus refiner dropout_r03_only proxy | ResidualRefiner CNN | yes; frozen O256 decoder | yes | yes; dropout_r03_only attack proxy | no | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh:257-290 |
| StageE | exp52/exp53/exp54 | StageD terms with stronger decoder-aware clean/attack message preservation and quality gate | image_mse=8.0; image_l1=2.0; message=1.50; attack_message=0.65; latent=0.45; sensitive=3.00; quality_gate_psnr=33.80; quality_gate=90.0; regularize=0.01 | same base-side dropout-margin protocol; final refiner StageE dropout_r03_only proxy p=0.70 | ResidualRefiner CNN | yes; frozen O256 decoder | yes | yes; attacked refined image decoded after dropout proxy | no | /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh:291-327; /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/scripts/train_rgb_refinement.py:742-761 |


## 5. StageE Training Cost

Source files:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_training_cost_summary.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_cost_availability.md`
| experiment | seed | base training time | pair generation time | refiner training time | total preparation cost | GPU | batch size | training steps | timing scope | confidence | source file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp52 | 62 | 03:47:16 | 00:15:39 | 00:46:30 | 04:49:25 | not found in StageE training logs | base=2; refiner=4 | base=65000; refiner=13750 | sum of /usr/bin/time -p real values; excludes formal attacks/BA/PSNR/SSIM/LPIPS | formal_log | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_identity20k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_base30_nongeomlight10k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_base50_nongeomlight_plus20k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_base55_weakattack_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_base60_dropout_focus_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_base65_dropout_margin_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_train_pairs_eval.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_refiner_stageA.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_refiner_stageB.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_refiner_stageBplus.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_refiner_stageC.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_refiner_stageD_dropout_balanced.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_refiner_stageE_decoderaware.log |
| exp53 | 61 | 03:48:32 | 00:16:02 | 00:47:09 | 04:51:42 | not found in StageE training logs | base=2; refiner=4 | base=65000; refiner=13750 | sum of /usr/bin/time -p real values; excludes formal attacks/BA/PSNR/SSIM/LPIPS | formal_log | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_identity20k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_base30_nongeomlight10k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_base50_nongeomlight_plus20k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_base55_weakattack_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_base60_dropout_focus_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_base65_dropout_margin_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_train_pairs_eval.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_refiner_stageA.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_refiner_stageB.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_refiner_stageBplus.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_refiner_stageC.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_refiner_stageD_dropout_balanced.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_refiner_stageE_decoderaware.log |
| exp54 | 63 | 03:45:28 | 00:15:25 | 00:46:21 | 04:47:14 | not found in StageE training logs | base=2; refiner=4 | base=65000; refiner=13750 | sum of /usr/bin/time -p real values; excludes formal attacks/BA/PSNR/SSIM/LPIPS | formal_log | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_identity20k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_base30_nongeomlight10k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_base50_nongeomlight_plus20k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_base55_weakattack_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_base60_dropout_focus_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_base65_dropout_margin_plus5k_train.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_train_pairs_eval.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_refiner_stageA.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_refiner_stageB.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_refiner_stageBplus.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_refiner_stageC.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_refiner_stageD_dropout_balanced.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_refiner_stageE_decoderaware.log |

### 5.1 Cost availability notes

# StageE Cost Availability

- exp52 / exp53 / exp54 all have formal training cost logs for base training, pair generation, and refiner training.
- The StageE preparation-cost table can be updated in Chapter 4 if the thesis clearly states the timing scope.
- Timing scope: `/usr/bin/time -p real`; includes training logging and checkpoint saving overhead; excludes final formal attacks, BA, PSNR, SSIM, and LPIPS.
- GPU model was not printed in StageE training logs. Do not hard-code a GPU model for StageE training cost unless another source is added.
- Existing O256-RGB / DiffW-like cost tables may be retained as historical or baseline timing, but should not be used as a substitute for StageE training cost.
- For Section 4.5, use exp52/53/54 preparation costs directly and keep the claim conservative: StageE has a multi-hour one-time preparation cost, while inference remains much faster than DiffW-like in the available end-to-end timing.

Interpretation: StageE 有 formal training cost log，可更新 4.5 的準備成本表；但 GPU 欄位在 StageE training logs 中為 `not found`。與 DiffW-like 比較時應明確區分一次性 preparation cost、end-to-end inference timing，以及不完整的 compute-only timing。


## 6. StageE Inference Timing

Source files:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_compute_only_inference.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_end_to_end_inference.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_inference_timing_notes.md`

### 6.1 Compute-only inference

StageE-specific compute-only timing: `MISSING`。下表保留舊 formal timing baseline 與明確的 StageE missing rows；不可把舊 O256-RGB compute-only timing 直接寫成 StageE timing。
| method | stage | batch size | ms/image | images/sec | speedup vs DiffW-like | overhead vs O256-Base | GPU | num images | timing scope | source file | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffW-like | formal timing baseline, not StageE-specific | 1 | 369.281699 | 2.707960 | 1.000000 | 5.233004 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| DiffW-like | formal timing baseline, not StageE-specific | 2 | 277.724471 | 3.600691 | 1.000000 | 5.555670 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| DiffW-like | formal timing baseline, not StageE-specific | 4 | 229.903434 | 4.349652 | 1.000000 | 6.055834 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| O256-Base | formal timing baseline, not StageE-specific | 1 | 70.567821 | 14.170765 | 5.233004 | 1.000000 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| O256-RGB | formal timing baseline, not StageE-specific | 1 | 78.502902 | 12.738383 | 4.704052 | 1.112446 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| O256-Base | formal timing baseline, not StageE-specific | 2 | 49.989376 | 20.004250 | 5.555670 | 1.000000 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| O256-RGB | formal timing baseline, not StageE-specific | 2 | 48.953814 | 20.427418 | 5.673194 | 0.979284 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| O256-Base | formal timing baseline, not StageE-specific | 4 | 37.963962 | 26.340770 | 6.055834 | 1.000000 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| O256-RGB | formal timing baseline, not StageE-specific | 4 | 38.276553 | 26.125654 | 6.006378 | 1.008234 | NVIDIA GeForce RTX 4090 | 100 | Synchronized compute-only wall-clock; includes the native sampling implementation's CPU/GPU tensor transfers, but excludes dataset I/O, PNG writing, PSNR, SSIM, and BA | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv | formal_timing_baseline_not_stageE |
| O256-StageE | StageE | 1 | missing | missing | missing | missing | missing | missing | StageE-specific compute-only benchmark not found | needs benchmark_stageE_inference.py if required | missing |
| O256-StageE | StageE | 2 | missing | missing | missing | missing | missing | missing | StageE-specific compute-only benchmark not found | needs benchmark_stageE_inference.py if required | missing |
| O256-StageE | StageE | 4 | missing | missing | missing | missing | missing | missing | StageE-specific compute-only benchmark not found | needs benchmark_stageE_inference.py if required | missing |

### 6.2 End-to-end inference

| method | stage | num images | total time | ms/image | images/sec | speedup vs DiffW-like | overhead vs O256-Base | source file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffW-like | formal timing baseline | 1000 | 399.85 | 399.850000 | 2.500938 | 1.000000 | 4.085939 | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/end_to_end_inference_timing.csv |
| O256-Base | formal timing baseline | 1000 | 97.86 | 97.860000 | 10.218680 | 4.085939 | 1.000000 | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/end_to_end_inference_timing.csv |
| O256-RGB | formal timing baseline | 1000 | 159.09 | 159.090000 | 6.285750 | 2.513357 | 1.625690 | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/end_to_end_inference_timing.csv |
| O256-StageE-exp52 | StageE | 1000 | 164.96 | 164.96 | 6.062076 | 2.423921 | 1.685673 | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_base65_dropout_margin_apply.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/52_formal_o256_full_from_scratch_stageE_seed62_20260708/logs/exp52_rgb_full_from_scratch_stageE_apply.log |
| O256-StageE-exp53 | StageE | 1000 | 165.77 | 165.77 | 6.032455 | 2.412077 | 1.693951 | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_base65_dropout_margin_apply.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/53_formal_o256_full_from_scratch_stageE_seed61_20260709/logs/exp53_rgb_full_from_scratch_stageE_apply.log |
| O256-StageE-exp54 | StageE | 1000 | 160.35 | 160.35 | 6.236358 | 2.493608 | 1.638565 | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_base65_dropout_margin_apply.log; /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/54_formal_o256_full_from_scratch_stageE_seed63_20260714/logs/exp54_rgb_full_from_scratch_stageE_apply.log |

### 6.3 Inference timing notes

# StageE Inference Timing Notes

- StageE-specific compute-only benchmark was not found. The compute-only CSV therefore includes old DiffW-like/O256-Base/O256-RGB formal timing baselines and explicit missing rows for StageE.
- StageE end-to-end timing is available as partial formal logs: base finaltest1000 apply plus StageE refiner apply.
- These end-to-end logs include I/O and output writing but exclude PSNR/SSIM/LPIPS/BA/attack evaluation.
- StageE refiner architecture is the same ResidualRefiner family as O256-RGB, but because StageE-specific compute-only timing was not rerun, the old O256-RGB compute-only timing should be described only as a baseline/reference.
- Chapter 4 Table 11/12 can be updated with StageE end-to-end timing if labeled as partial formal apply timing. Avoid claiming a completed StageE pure compute benchmark.

Interpretation: 4.5 可更新 end-to-end apply timing；與 DiffW-like 比較時可保守說明 StageE end-to-end apply 仍約 2.4x faster than DiffW-like baseline，但此 timing 不含 PSNR/BA/attack evaluation。compute-only 若要正式更新表 11/12，需另外補測。


## 7. DiffW-like Comparison

Source files:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/diffw_like_comparison_summary.csv`
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/diffw_like_interpretation.md`
| run name | epoch / steps | seed | PSNR | SSIM | LPIPS | Clean BA | Nongeom Avg BA | Min BA | Min attack | Dropout BA | training time | role | source file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| formal_diffw_like_finaltest1000 | 20k | 61 | 26.4236 | 0.8362 | 0.206004 | 100.00 | 99.35 | 95.47 | Paper_Dropout_r03 | 95.47 | 02:25:31.24 | formal main result | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/01_method_comparison/formal_diffw_like_finaltest1000/summary.md |
| diffw_sf0p5 | sf0.5 actual-training | 61 | 27.9481 | 0.8765 | 0.197442 | 100.00 | 97.55 | 88.17 | Paper_GaussianNoise_arg03 | 88.62 | 02:26:02 | supplemental comparison | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/09_supplementary_diffw_actual_train_strength0p5_20260628/logs |
| diffw_sf0p75 | sf0.75 actual-training | 61 | 26.5957 | 0.8544 | 0.219181 | 100.00 | 99.56 | 96.18 | Paper_Dropout_r03 | 96.18 | 02:26:14 | supplemental comparison | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/10_supplementary_diffw_actual_train_strength0p75_20260628/logs |
| diffw_attackpressure_singlevar | singlevar attack-pressure | 61 | 25.5596 | 0.8202 | 0.224331 | 100.00 | 98.54 | 89.41 | Paper_Dropout_r03 | 89.41 | 02:25:51 | supplemental comparison | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/12_supplementary_diffw_actual_train_attackpressure_singlevar_20260628/logs |
| diffw_attackpressure_epoch7_truebranch | epoch7 / 35k | 61 | 28.7506 | 0.9169 | 0.104259 | 100.00 | 94.76 | 70.00 | Paper_GaussianNoise_arg03 | 92.12 | 04:15:42 | supplemental long-train comparison | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_diffw_actual_train_attackpressure_epoch7_truebranch_20260628/logs |
| diffw_attackpressure_epoch12_60k | epoch12 / 60k | 61 | 37.4562 | 0.9732 | 0.008766 | 100.00 | 91.91 | 68.18 | Paper_GaussianNoise_arg03 | 99.74 | 07:18:03 | supplemental long-train comparison | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/15_supplementary_diffw_actual_train_attackpressure_epoch12_60k_20260629/logs |

### 7.1 DiffW-like interpretation

# DiffW-like Interpretation

- The original formal DiffW-like result remains the proper formal main baseline because it is the locked finaltest1000 formal run.
- DiffW-like epoch7, epoch12/60k, sf0.5/sf0.75, and singlevar runs are supplemental comparisons. They should not replace the formal baseline in the main table.
- The epoch12/60k run shows that DiffW-like can reach much higher PSNR, but its worst-case BA drops to 68.18 under `Paper_GaussianNoise_arg03`.
- It is conservative and accurate to write: DiffW-like long training can obtain higher PSNR, but worst-case BA remains insufficient; O256 StageE targets a more balanced PSNR / Nongeom Avg BA / worst-case BA trade-off.
- Do not write that O256 StageE全面優於 DiffW-like.

Interpretation: DiffW-like 長訓練不建議放入正文主結果表取代 formal baseline；可作補充比較。保守寫法是：DiffW-like 長訓練可取得較高 PSNR，但 worst-case BA 仍不足；O256 StageE 的定位是較平衡的 PSNR / average robustness / worst-case robustness trade-off。


## 8. Thesis-Ready Tables

Source file:
- `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/thesis_ready_tables_stageE.md`
建議：Table A / B / C 可直接進第四章正文或正文+補充；Table D 中 training cost 可用，compute-only timing 部分需保守標示缺 StageE-specific benchmark。
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


## 9. Chapter Update Recommendations

### 4.1
- 實驗設定：保留 finaltest1000、32-bit message、256x256、11 種 formal non-geometric attacks。
- Attack set 表：不需大改，但建議加入 raw log name / paper display name 對照，避免名稱不一致。
- 需要補 StageE final protocol 描述：說明 StageE 是 final O256-RGB training protocol，包含 base-side dropout margin 與 decoder-aware final refiner stage。

### 4.2
- 主結果表：建議以 Original O256-RGB baseline、StageD exp40/exp41、StageE exp52/exp53 為核心；exp54 可放 near-miss 補充列或補充段落。
- 原始 O256-RGB baseline：保留，作為「畫質已改善但 worst-case BA 不足」的歷史基準。
- StageD / StageE：StageD 用來說明 base-side dropout-margin 已有效但 seed62 邊界仍存在；StageE 是 final main result。

### 4.3
- 保留 per-attack 表。
- 建議改成 Dropout / worst-case BA 分析主軸，說明 Paper_Dropout_r03 是 final bottleneck。

### 4.4
- 整理 StageD -> StageE 設計演進：從 weak attack pressure、dropout-margin，到 decoder-aware refiner。
- exp54：寫作 seed63 near-miss validation，不寫成 strict pass。
- 早期探索性實驗：降級為 design motivation / supplemental analysis，不放主結果核心。

### 4.5
- 訓練成本表：可更新 StageE preparation cost，但需標 timing scope。
- 推論成本表：可更新 StageE end-to-end apply timing；compute-only StageE-specific timing 缺失，需保守描述。
- 若 compute-only missing：寫「目前僅有 StageE end-to-end apply timing；compute-only 仍需補測，不以舊 O256-RGB 直接替代。」

### 4.6
- 本章小結：主張 StageE 在正式協定下改善 PSNR / Nongeom Avg BA / worst-case BA 的折衷，且 exp52/exp53 full-from-scratch 通過。
- 同時保留限制：exp54 seed63 是 near-miss，表示仍有 seed-dependent boundary；不要寫成全面勝過 DiffW-like。


## 10. Non-Writable Claims

- 不可宣稱 O256 StageE 全面優於 DiffW-like。
- 不可把 exp54 寫成 strict-pass final result；exp54 是 completed near-miss。
- 不可把舊 O256-RGB compute-only timing 當成 StageE-specific compute-only timing。
- 不可把 continuation result 寫成 full from-scratch。
- 不可把 non-finaltest1000 結果寫成 formal main result。
- 不可把 Clean BA 納入 Nongeom Avg BA。
- 不可說 StageE 改了 RGB refiner architecture；目前 evidence 顯示 architecture unchanged，主要是 training protocol / loss 權重與 decoder-aware branch。
- 不可說 DiffW-like 長訓練完全失敗；較保守說法是高 PSNR 可達成，但 worst-case BA 仍不足。


## 11. Missing Items

| item | status | 是否影響第四章修訂 | 是否需要補實驗 |
| --- | --- | --- | --- |
| Required audit files | none | 若使用 v2 audit，必要檔案齊全。 | no |
| StageE final metrics / per-attack BA | none missing for exp52/exp53/exp54 | 第四章主結果與 per-attack 表可更新。 | no |
| StageE-specific compute-only timing | MISSING | 不能正式更新 StageE compute-only 表；只能保守使用 old baseline / end-to-end timing。 | optional benchmark only, no training |
| GPU in StageE training logs | MISSING / not found in StageE training logs | StageE training cost 表 GPU 欄位需寫 not found，或另找環境紀錄補充。 | no |
| Method details | no critical missing item found | 第三章可小幅補文字與 refiner objective。 | no |


---

## Appendix: Required Source Files Status

| file | status | source path |
| --- | --- | --- |
| stageE_thesis_audit_report.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_thesis_audit_report.md |
| thesis_ready_tables_stageE.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/thesis_ready_tables_stageE.md |
| stageE_final_metrics.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_final_metrics.csv |
| stageE_per_attack_ba.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_per_attack_ba.csv |
| formal_attack_name_mapping.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/formal_attack_name_mapping.csv |
| stageE_method_delta.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_method_delta.md |
| stageE_loss_and_training_protocol.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_loss_and_training_protocol.csv |
| stageE_training_cost_summary.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_training_cost_summary.csv |
| stageE_end_to_end_inference.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_end_to_end_inference.csv |
| stageE_compute_only_inference.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_compute_only_inference.csv |
| diffw_like_comparison_summary.csv | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/diffw_like_comparison_summary.csv |
| diffw_like_interpretation.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/diffw_like_interpretation.md |
| stageE_cost_availability.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_cost_availability.md |
| stageE_inference_timing_notes.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_inference_timing_notes.md |
| formal_attack_set_validation.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/formal_attack_set_validation.md |
| stageE_metrics_validation.md | available | /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/stageE_thesis_audit_20260714_v2/stageE_metrics_validation.md |
