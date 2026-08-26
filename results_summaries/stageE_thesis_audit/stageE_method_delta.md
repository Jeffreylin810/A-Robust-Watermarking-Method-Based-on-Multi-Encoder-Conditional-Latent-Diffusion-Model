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
