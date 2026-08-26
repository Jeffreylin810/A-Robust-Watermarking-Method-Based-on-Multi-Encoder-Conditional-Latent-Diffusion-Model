# O256-PAPER Diagnostic Commands

These are diagnosis-only runs. They are not final paper main-table results unless we later freeze this protocol.

## 0. Existing O256 clean BA

Purpose: check whether the already trained O256-PAPER checkpoint can decode the message before any attack.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

mkdir -p results/attacked_Result/clean_O256_PAPER_seed61
ln -s /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_PAPER_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_PAPER_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg64_paper_protocol.yml \
  --resume Result/CoCo_ddpm_O256_PAPER_seed61_500k_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_PAPER_seed61 \
  --gt_message_dir results/test_result/O256_PAPER_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_PAPER_seed61_clean_ba.log
```

## 1. O256-MSG32-SANITY20K

Purpose: keep image size / dataset / latent bridge / attack profile the same as O256-PAPER, but change payload from msg64 to msg32.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg32_sanity20k.yml \
  --resume ./ckpts/nonexistent_O256_MSG32_SANITY20K_start.pth.tar \
  --image_folder results/validation/O256_MSG32_SANITY20K_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg32_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG32_SANITY20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_SANITY20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_psnr_ssim.log

mkdir -p results/attacked_Result/clean_O256_MSG32_SANITY20K_seed61
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_SANITY20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG32_SANITY20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_SANITY20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_SANITY20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_clean_ba.log

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_SANITY20K_seed61/CoCo/CoCo \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_MSG32_SANITY20K_seed61 \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_MSG32_SANITY20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_SANITY20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_paper_matched_ba.log
```

Note for the already-run 20260602 eval:

The first O256-MSG32 eval was run before `--run_name` was added, so its actual output is:

```text
results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo
```

For that already-run output, use the fixed commands below instead of the stable-path commands above.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

mkdir -p results/attacked_Result/clean_O256_MSG32_SANITY20K_seed61_fixed
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  results/attacked_Result/clean_O256_MSG32_SANITY20K_seed61_fixed/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_SANITY20K_seed61_fixed \
  --gt_message_dir results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_clean_ba_fixed.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_psnr_ssim_fixed.log

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_MSG32_SANITY20K_seed61_fixed \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_paper_matched_attack_fixed.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_MSG32_SANITY20K_seed61_fixed \
  --gt_message_dir results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_paper_matched_ba_fixed.log
```

## 2. O256-MSG64-ID20K

Purpose: keep msg64, but train with identity-only attack profile. This checks whether paper-scale msg64 can learn a clean recoverable signal before robustness pressure.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg64_identity_sanity20k.yml \
  --resume ./ckpts/nonexistent_O256_MSG64_ID20K_start.pth.tar \
  --image_folder results/validation/O256_MSG64_ID20K_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG64_ID20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG64_ID20K_seed61_final.pth.tar

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg64_identity_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG64_ID20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG64_ID20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG64_ID20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG64_ID20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG64_ID20K_seed61_psnr_ssim.log

mkdir -p results/attacked_Result/clean_O256_MSG64_ID20K_seed61
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG64_ID20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG64_ID20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg64_identity_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG64_ID20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG64_ID20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG64_ID20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG64_ID20K_seed61_clean_ba.log
```

## 3. O256-PAPER visual check

Purpose: export cover / watermarked / absolute difference panels for the failed O256-PAPER run.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/make_watermark_comparison_grid.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_PAPER_seed61/CoCo/CoCo \
  --output_dir results/figures/O256_PAPER_seed61_comparison \
  --num_images 12 \
  --diff_gain 10
```

## 4. O256-PAPER eval decode gap diagnostic

Purpose: compare in-memory sampling decode, final latent decode, RGB tensor round-trip decode, and PNG saved-image decode.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/diagnose_eval_decode_gap.py \
  --config CoCo_eval1000_ours256_msg64_paper_protocol.yml \
  --resume Result/CoCo_ddpm_O256_PAPER_seed61_500k_final.pth.tar \
  --test_set CoCo \
  --num_images 16 \
  --output_dir results/diagnostics/O256_PAPER_seed61_eval_decode_gap \
  2>&1 | tee logs/O256_PAPER_seed61_eval_decode_gap.log
```

## 5. O256-MSG32 residual attenuation sweep

Purpose: O256-MSG32-SANITY20K has high clean BA but PSNR is far below the 30+ requirement. This post-hoc sweep scales the saved watermark residual by alpha:

```text
scaled_image = cover + alpha * (watermarked_image - cover)
```

This is not a final method result. It is a cheap gate before retraining: if PSNR can be moved above 30 while clean BA remains useful, the next formal training should use weaker residual / stronger fidelity control. If BA collapses near PSNR 30, the current representation needs architecture or embedding redesign instead of more long loss-weight searches.

Run the sweep first:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/scale_watermark_residual.py \
  --cover_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  --output_root results/test_result/O256_MSG32_SANITY20K_seed61_residual_sweep \
  --alphas 0.25,0.30
```

Evaluate alpha 0.25:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_SANITY20K_seed61_residual_sweep/alpha_0p25/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_alpha0p25_psnr_ssim.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar \
  --attack_dir results/test_result/O256_MSG32_SANITY20K_seed61_residual_sweep/alpha_0p25 \
  --gt_message_dir results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_alpha0p25_clean_ba.log
```

Evaluate alpha 0.30:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_SANITY20K_seed61_residual_sweep/alpha_0p30/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_alpha0p30_psnr_ssim.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_sanity20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_SANITY20K_seed61_final.pth.tar \
  --attack_dir results/test_result/O256_MSG32_SANITY20K_seed61_residual_sweep/alpha_0p30 \
  --gt_message_dir results/test_result/O256_MSG32_SANITY20K_seed61/20260602_004053/CoCo \
  2>&1 | tee logs/O256_MSG32_SANITY20K_seed61_alpha0p30_clean_ba.log
```

## 6. O256-MSG32-PSNR-GATE20K

Purpose: train a short identity-only, fidelity-heavy msg32 branch to test whether the current O256 representation can learn clean BA while keeping PSNR above 30. This is the next training gate after the residual sweep.

Config changes from O256-MSG32-SANITY20K:

| item | O256-MSG32-SANITY20K | O256-MSG32-PSNR-GATE20K |
|---|---:|---:|
| message length | 32 | 32 |
| steps | 20k | 20k |
| attack profile | `identity_gn_sp_webp_bc` | `identity_only` |
| message loss weight | 30 | 10 |
| RGB loss weight | 20 | 100 |

Success gate:

| metric | minimum target |
|---|---:|
| PSNR | > 30 |
| clean BA | preferably > 90 |

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg32_psnr_gate20k.yml \
  --resume ./ckpts/nonexistent_O256_MSG32_PSNR_GATE20K_start.pth.tar \
  --image_folder results/validation/O256_MSG32_PSNR_GATE20K_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG32_PSNR_GATE20K_seed61_final.pth.tar
```

Evaluate:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg32_psnr_gate20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_PSNR_GATE20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG32_PSNR_GATE20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_PSNR_GATE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_seed61_psnr_ssim.log

mkdir -p results/attacked_Result/clean_O256_MSG32_PSNR_GATE20K_seed61
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_PSNR_GATE20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG32_PSNR_GATE20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_psnr_gate20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_PSNR_GATE20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_PSNR_GATE20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_PSNR_GATE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_seed61_clean_ba.log
```

Only if the success gate passes, then run paper-matched attacks:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_PSNR_GATE20K_seed61/CoCo/CoCo \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_MSG32_PSNR_GATE20K_seed61 \
  2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_psnr_gate20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_PSNR_GATE20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_MSG32_PSNR_GATE20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_PSNR_GATE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_seed61_paper_matched_ba.log
```

## 7. O256-MSG32-PSNR-GATE20K-RGB300

Purpose: O256-MSG32-PSNR-GATE20K improved PSNR from 19.0830 to 24.5007 while keeping clean BA at 89.79%. This run keeps the same identity-only msg32 gate, but increases RGB loss weight from 100 to 300.

Config changes:

| item | O256-MSG32-PSNR-GATE20K | O256-MSG32-PSNR-GATE20K-RGB300 |
|---|---:|---:|
| message length | 32 | 32 |
| steps | 20k | 20k |
| attack profile | `identity_only` | `identity_only` |
| message loss weight | 10 | 10 |
| RGB loss weight | 100 | 300 |

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg32_psnr_gate20k_rgb300.yml \
  --resume ./ckpts/nonexistent_O256_MSG32_PSNR_GATE20K_RGB300_start.pth.tar \
  --image_folder results/validation/O256_MSG32_PSNR_GATE20K_RGB300_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_RGB300_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG32_PSNR_GATE20K_RGB300_seed61_final.pth.tar
```

Evaluate:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg32_psnr_gate20k_rgb300.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_PSNR_GATE20K_RGB300_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG32_PSNR_GATE20K_RGB300_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_RGB300_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_PSNR_GATE20K_RGB300_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_RGB300_seed61_psnr_ssim.log

mkdir -p results/attacked_Result/clean_O256_MSG32_PSNR_GATE20K_RGB300_seed61
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_PSNR_GATE20K_RGB300_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG32_PSNR_GATE20K_RGB300_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_psnr_gate20k_rgb300.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_PSNR_GATE20K_RGB300_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_PSNR_GATE20K_RGB300_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_PSNR_GATE20K_RGB300_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_PSNR_GATE20K_RGB300_seed61_clean_ba.log
```

## 8. O256-MSG32-STR0P5-RGB100-GATE20K

Purpose: RGB300 showed that simply increasing RGB loss breaks message learning. This run returns to RGB100 and reduces `strength_factor` from 1.0 to 0.5, so the model is trained from the start under a weaker watermark residual.

Config changes:

| item | O256-MSG32-PSNR-GATE20K RGB100 | O256-MSG32-STR0P5-RGB100-GATE20K |
|---|---:|---:|
| message length | 32 | 32 |
| steps | 20k | 20k |
| attack profile | `identity_only` | `identity_only` |
| strength factor | 1.0 | 0.5 |
| message loss weight | 10 | 10 |
| RGB loss weight | 100 | 100 |

Success gate:

| metric | target |
|---|---:|
| PSNR | > 30 |
| clean BA | preferably > 85-90 |

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg32_str0p5_rgb100_gate20k.yml \
  --resume ./ckpts/nonexistent_O256_MSG32_STR0P5_RGB100_GATE20K_start.pth.tar \
  --image_folder results/validation/O256_MSG32_STR0P5_RGB100_GATE20K_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG32_STR0P5_RGB100_GATE20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG32_STR0P5_RGB100_GATE20K_seed61_final.pth.tar
```

Evaluate:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg32_str0p5_rgb100_gate20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_STR0P5_RGB100_GATE20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG32_STR0P5_RGB100_GATE20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG32_STR0P5_RGB100_GATE20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_STR0P5_RGB100_GATE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_STR0P5_RGB100_GATE20K_seed61_psnr_ssim.log

mkdir -p results/attacked_Result/clean_O256_MSG32_STR0P5_RGB100_GATE20K_seed61
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_STR0P5_RGB100_GATE20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG32_STR0P5_RGB100_GATE20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_str0p5_rgb100_gate20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_STR0P5_RGB100_GATE20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_STR0P5_RGB100_GATE20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_STR0P5_RGB100_GATE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_STR0P5_RGB100_GATE20K_seed61_clean_ba.log
```

## 9. CL3: O256-MSG32-PSNR30-HINGE20K

Purpose: previous scalar searches showed two bad extremes:

- `rgb_loss_weight=300` improves fidelity only slightly and collapses clean BA.
- `strength_factor=0.5` improves PSNR only slightly and also collapses clean BA.

CL3 keeps `strength_factor=1` and adds a PSNR-constrained RGB hinge loss. The hinge only activates when decoded RGB MSE is worse than the target PSNR threshold. This tests whether we can push toward PSNR 30 without constantly crushing the message branch.

Config changes from `O256-MSG32-SANITY20K`:

| item | O256-MSG32-SANITY20K | CL3 |
|---|---:|---:|
| message length | 32 | 32 |
| steps | 20k | 20k |
| attack profile | `identity_gn_sp_webp_bc` | `identity_only` |
| strength factor | 1.0 | 1.0 |
| message loss weight | 30 | 30 |
| RGB loss weight | 20 | 20 |
| PSNR hinge target | none | 30 dB |
| PSNR hinge weight | none | 200 |

Success gate:

| metric | stop / continue rule |
|---|---|
| PSNR < 28 | fidelity control is still too weak |
| PSNR >= 30 and clean BA < 75 | message signal is not efficient enough under PSNR constraint |
| PSNR >= 30 and clean BA 75-80 | borderline; tune hinge/message weights once |
| PSNR >= 30 and clean BA >= 80 | run paper-matched attacks |

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg32_cl3_psnr30_hinge20k.yml \
  --resume ./ckpts/nonexistent_O256_MSG32_CL3_PSNR30_HINGE20K_start.pth.tar \
  --image_folder results/validation/O256_MSG32_CL3_PSNR30_HINGE20K_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final.pth.tar
```

Evaluate PSNR and clean BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr30_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG32_CL3_PSNR30_HINGE20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_CL3_PSNR30_HINGE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_psnr_ssim.log

mkdir -p results/attacked_Result/clean_O256_MSG32_CL3_PSNR30_HINGE20K_seed61
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_CL3_PSNR30_HINGE20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG32_CL3_PSNR30_HINGE20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr30_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_CL3_PSNR30_HINGE20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR30_HINGE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_clean_ba.log
```

Run sampling-gap diagnosis only after PSNR/clean BA are known:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/diagnose_eval_decode_gap.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr30_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final.pth.tar \
  --num_images 64 \
  --sampling_timesteps 3 \
  --output_dir results/diagnostics/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_eval_decode_gap64 \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_eval_decode_gap64.log
```

Only if PSNR >= 30 and clean BA >= 80, run paper-matched attacks:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_CL3_PSNR30_HINGE20K_seed61/CoCo/CoCo \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_MSG32_CL3_PSNR30_HINGE20K_seed61 \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr30_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_MSG32_CL3_PSNR30_HINGE20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR30_HINGE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_paper_matched_ba.log
```

## 10. CL3 calibration: O256-MSG32-PSNR36-HINGE20K

Purpose: `CL3-PSNR30` reached clean BA 97.03%, but final eval PSNR was only 24.2751. Its training log showed direct-RGB PSNR near 30, so this run raises the direct-RGB target to 36 dB while keeping all other CL3 settings fixed.

Config changes from `CL3-PSNR30`:

| item | CL3-PSNR30 | CL3-PSNR36 |
|---|---:|---:|
| message length | 32 | 32 |
| steps | 20k | 20k |
| attack profile | `identity_only` | `identity_only` |
| strength factor | 1.0 | 1.0 |
| message loss weight | 30 | 30 |
| RGB loss weight | 20 | 20 |
| PSNR hinge target | 30 dB | 36 dB |
| PSNR hinge weight | 200 | 200 |

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume ./ckpts/nonexistent_O256_MSG32_CL3_PSNR36_HINGE20K_start.pth.tar \
  --image_folder results/validation/O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final.pth.tar
```

Evaluate PSNR and clean BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_psnr_ssim.log

mkdir -p results/attacked_Result/clean_O256_MSG32_CL3_PSNR36_HINGE20K_seed61
ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_clean_ba.log
```

Run sampling-gap diagnosis only after PSNR/clean BA are known:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/diagnose_eval_decode_gap.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final.pth.tar \
  --num_images 64 \
  --sampling_timesteps 3 \
  --output_dir results/diagnostics/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_eval_decode_gap64 \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_eval_decode_gap64.log
```

Only if PSNR >= 30 and clean BA >= 80, run paper-matched attacks:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_paper_matched_ba.log
```

## 11. Final sampling fidelity decomposition

Purpose: CL3-PSNR30/36 reached high clean BA, but final eval PSNR stayed near 24 dB. This diagnostic decomposes image fidelity across direct watermark outputs and final diffusion sampling outputs.

Measured paths:

| path | meaning |
|---|---|
| `input_vs_gt` | dataset sanity check; input and gt should be identical for this setup |
| `vae_roundtrip_gt` | VAE encode/decode upper bound |
| `direct_target_t0` | direct watermark encoder on clean target latent at t=0 |
| `direct_target_t999` | direct watermark encoder on clean target latent at t=999 |
| `xs_final` | final DDIM latent state |
| `x0_pred_final_formal_eval` | formal eval output path used by `eval_diffusion.py` |
| `x_noise_final` | noised-image branch returned by sampling |
| `x0_pred_png_saved` | formal eval output after PNG save/reload |

Run on CL3-PSNR36 first:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/diagnose_final_sampling_fidelity.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final.pth.tar \
  --num_images 64 \
  --sampling_timesteps 3 \
  --seed 61 \
  --output_dir results/diagnostics/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final_sampling_fidelity64 \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final_sampling_fidelity64.log
```

Optional comparison on CL3-PSNR30:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/diagnose_final_sampling_fidelity.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr30_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final.pth.tar \
  --num_images 64 \
  --sampling_timesteps 3 \
  --seed 61 \
  --output_dir results/diagnostics/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final_sampling_fidelity64 \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_final_sampling_fidelity64.log
```

Output files:

```text
results/diagnostics/<run_name>_final_sampling_fidelity64/report.txt
results/diagnostics/<run_name>_final_sampling_fidelity64/report.json
results/diagnostics/<run_name>_final_sampling_fidelity64/summary.csv
results/diagnostics/<run_name>_final_sampling_fidelity64/x0_pred_png_roundtrip/
```

## 12. VAE-reference fidelity evaluation

Purpose: final sampling decomposition showed the frozen VAE round-trip itself is around 24-25 dB. This evaluator separates the VAE ceiling from watermark-specific distortion.

Measured comparisons:

| comparison | meaning |
|---|---|
| `GT vs VAE cover` | frozen VAE reconstruction ceiling |
| `GT vs watermarked` | conventional PSNR/SSIM against original GT |
| `VAE cover vs watermarked` | watermark-specific distortion after VAE reconstruction |

Run on CL3-PSNR36:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/evaluate_vae_reference_fidelity.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  --output_dir results/diagnostics/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_vae_reference_fidelity \
  --save_vae_cover \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_vae_reference_fidelity.log
```

Optional comparison on CL3-PSNR30:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/evaluate_vae_reference_fidelity.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr30_hinge20k.yml \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR30_HINGE20K_seed61/CoCo/CoCo \
  --output_dir results/diagnostics/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_vae_reference_fidelity \
  --save_vae_cover \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR30_HINGE20K_seed61_vae_reference_fidelity.log
```

Output files:

```text
results/diagnostics/<run_name>_vae_reference_fidelity/report.txt
results/diagnostics/<run_name>_vae_reference_fidelity/report.json
results/diagnostics/<run_name>_vae_reference_fidelity/summary.csv
results/diagnostics/<run_name>_vae_reference_fidelity/vae_reconstructed_cover/
```

## 13. GT-RGB compensation sanity: O256-MSG32-GT-RGBCOMP20K

Purpose: previous CL3 runs optimized RGB fidelity against the VAE reconstructed cover, so final GT PSNR was bounded by the frozen VAE round-trip ceiling. This run changes the RGB target to the original GT and weakens the latent x0 anchor.

Key config changes:

| item | value |
|---|---|
| train config | `CoCo_train10000_ours256_msg32_gt_rgbcomp20k.yml` |
| eval config | `CoCo_eval1000_ours256_msg32_gt_rgbcomp20k.yml` |
| message length | 32 |
| steps | 20k |
| attack profile | `identity_only` |
| x0 loss weight | 0.1 |
| message loss weight | 30 |
| RGB loss weight | 50 |
| RGB loss target | `original_gt` |
| PSNR hinge target | 30 dB |
| PSNR hinge weight | 200 |

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

mkdir -p logs Result

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg32_gt_rgbcomp20k.yml \
  --resume ./ckpts/nonexistent_O256_MSG32_GT_RGBCOMP20K_start.pth.tar \
  --image_folder results/validation/O256_MSG32_GT_RGBCOMP20K_seed61 \
  --seed 61 2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_MSG32_GT_RGBCOMP20K_seed61_final.pth.tar
```

Evaluate PSNR / SSIM:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg32_gt_rgbcomp20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_GT_RGBCOMP20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_MSG32_GT_RGBCOMP20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_psnr_ssim.log
```

Evaluate clean BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

mkdir -p results/attacked_Result/clean_O256_MSG32_GT_RGBCOMP20K_seed61

ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_MSG32_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_MSG32_GT_RGBCOMP20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_gt_rgbcomp20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_GT_RGBCOMP20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_GT_RGBCOMP20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_clean_ba.log
```

Evaluate VAE-reference fidelity:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/evaluate_vae_reference_fidelity.py \
  --config CoCo_eval1000_ours256_msg32_gt_rgbcomp20k.yml \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_MSG32_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  --output_dir results/diagnostics/O256_MSG32_GT_RGBCOMP20K_seed61_vae_reference_fidelity \
  --save_vae_cover \
  2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_vae_reference_fidelity.log
```

Run final sampling decomposition if PSNR is still near 24-25:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/diagnose_final_sampling_fidelity.py \
  --config CoCo_eval1000_ours256_msg32_gt_rgbcomp20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_GT_RGBCOMP20K_seed61_final.pth.tar \
  --num_images 64 \
  --sampling_timesteps 3 \
  --seed 61 \
  --output_dir results/diagnostics/O256_MSG32_GT_RGBCOMP20K_seed61_final_sampling_fidelity64 \
  2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_final_sampling_fidelity64.log
```

Only if PSNR >= 30 and clean BA >= 80, run paper-matched attacks:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_MSG32_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_MSG32_GT_RGBCOMP20K_seed61 \
  2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_gt_rgbcomp20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_GT_RGBCOMP20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_MSG32_GT_RGBCOMP20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_MSG32_GT_RGBCOMP20K_seed61_paper_matched_ba.log
```

## 14. Image-only upper bound: O256-IMGONLY-GT-RGBCOMP20K

Purpose: this is not a watermark result. It removes message recovery pressure and tests whether the O256 latent/VAE path can exceed the frozen VAE round-trip PSNR ceiling when only image fidelity is optimized.

Key config changes:

| item | value |
|---|---|
| train config | `CoCo_train10000_ours256_imgonly_gt_rgbcomp20k.yml` |
| eval config | `CoCo_eval1000_ours256_imgonly_gt_rgbcomp20k.yml` |
| message length | 32, interface only |
| steps | 20k |
| attack profile | `identity_only` |
| x0 loss weight | 0 |
| message loss weight | 0 |
| RGB loss weight | 100 |
| RGB loss target | `original_gt` |
| PSNR hinge target | 30 dB |
| PSNR hinge weight | 200 |

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

mkdir -p logs Result

echo "[run_env] TORCH_CUDNN_V8_API_DISABLED=1, CUDA_MODULE_LOADING=LAZY" \
  2>&1 | tee logs/O256_IMGONLY_GT_RGBCOMP20K_seed61_train.log

TORCH_CUDNN_V8_API_DISABLED=1 CUDA_MODULE_LOADING=LAZY \
/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_imgonly_gt_rgbcomp20k.yml \
  --resume ./ckpts/nonexistent_O256_IMGONLY_GT_RGBCOMP20K_start.pth.tar \
  --image_folder results/validation/O256_IMGONLY_GT_RGBCOMP20K_seed61 \
  --seed 61 2>&1 | tee -a logs/O256_IMGONLY_GT_RGBCOMP20K_seed61_train.log

cp ckpts/CoCo_ddpm_final.pth.tar Result/CoCo_ddpm_O256_IMGONLY_GT_RGBCOMP20K_seed61_final.pth.tar
```

Evaluate PSNR / SSIM:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python eval_diffusion.py \
  --config CoCo_eval1000_ours256_imgonly_gt_rgbcomp20k.yml \
  --resume Result/CoCo_ddpm_O256_IMGONLY_GT_RGBCOMP20K_seed61_final.pth.tar \
  --image_folder results/test_result \
  --run_name O256_IMGONLY_GT_RGBCOMP20K_seed61 \
  --test_set CoCo \
  --seed 61 2>&1 | tee logs/O256_IMGONLY_GT_RGBCOMP20K_seed61_eval.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt/ \
  --results_path results/test_result/O256_IMGONLY_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_IMGONLY_GT_RGBCOMP20K_seed61_psnr_ssim.log
```

Evaluate VAE-reference fidelity:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/evaluate_vae_reference_fidelity.py \
  --config CoCo_eval1000_ours256_imgonly_gt_rgbcomp20k.yml \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_IMGONLY_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  --output_dir results/diagnostics/O256_IMGONLY_GT_RGBCOMP20K_seed61_vae_reference_fidelity \
  --save_vae_cover \
  2>&1 | tee logs/O256_IMGONLY_GT_RGBCOMP20K_seed61_vae_reference_fidelity.log
```

Optional clean BA check, for diagnosis only:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

mkdir -p results/attacked_Result/clean_O256_IMGONLY_GT_RGBCOMP20K_seed61

ln -sfn /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/test_result/O256_IMGONLY_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  results/attacked_Result/clean_O256_IMGONLY_GT_RGBCOMP20K_seed61/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_imgonly_gt_rgbcomp20k.yml \
  --resume Result/CoCo_ddpm_O256_IMGONLY_GT_RGBCOMP20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_IMGONLY_GT_RGBCOMP20K_seed61 \
  --gt_message_dir results/test_result/O256_IMGONLY_GT_RGBCOMP20K_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_IMGONLY_GT_RGBCOMP20K_seed61_clean_ba.log
```

Run final sampling decomposition if GT PSNR still does not exceed the VAE ceiling:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/diagnose_final_sampling_fidelity.py \
  --config CoCo_eval1000_ours256_imgonly_gt_rgbcomp20k.yml \
  --resume Result/CoCo_ddpm_O256_IMGONLY_GT_RGBCOMP20K_seed61_final.pth.tar \
  --num_images 64 \
  --sampling_timesteps 3 \
  --seed 61 \
  --output_dir results/diagnostics/O256_IMGONLY_GT_RGBCOMP20K_seed61_final_sampling_fidelity64 \
  2>&1 | tee logs/O256_IMGONLY_GT_RGBCOMP20K_seed61_final_sampling_fidelity64.log
```

## 15. VAE screening: reconstruction ceiling before more watermark training

Purpose: evaluate candidate VAEs without watermark training. This checks whether a VAE can reconstruct the eval1000 GT images near the required PSNR target.

Download first-round VAE candidates:

```bash
mkdir -p /home/jeffrey/projects/VAE_candidates
cd /home/jeffrey/projects/VAE_candidates

git lfs install

git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
git clone https://huggingface.co/stabilityai/sd-vae-ft-ema
```

Current local SD2.1 VAE baseline:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --output_dir results/diagnostics/vae_screening_eval1000_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_eval1000_seed61.log
```

Quick smoke test after downloading candidates:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd_vae_ft_mse=/home/jeffrey/projects/VAE_candidates/sd-vae-ft-mse \
  --vae sd_vae_ft_ema=/home/jeffrey/projects/VAE_candidates/sd-vae-ft-ema \
  --output_dir results/diagnostics/_smoke_vae_screening_sdvae_ft \
  --image_size 256 \
  --max_images 10 \
  2>&1 | tee logs/_smoke_vae_screening_sdvae_ft.log
```

Full eval1000 screening after the smoke test passes:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd_vae_ft_mse=/home/jeffrey/projects/VAE_candidates/sd-vae-ft-mse \
  --vae sd_vae_ft_ema=/home/jeffrey/projects/VAE_candidates/sd-vae-ft-ema \
  --output_dir results/diagnostics/vae_screening_eval1000_sdvae_ft \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_eval1000_sdvae_ft.log
```

Quick smoke test on 2 images:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --output_dir results/diagnostics/_smoke_vae_screening \
  --image_size 256 \
  --max_images 2
```

## 16. Decoder-only VAE fine-tune: test whether the VAE ceiling can exceed 30 dB

Purpose: first-round VAE screening showed the current SD2.1 VAE and `sd-vae-ft-mse` are effectively the same and both stay at 24.8510 dB. This experiment freezes the VAE encoder and fine-tunes only the decoder side:

```text
GT image x -> frozen encoder E(x) -> trainable decoder D'(z) -> reconstructed x_hat
```

This is not a watermark result. It only tests whether the latent/VAE output path can reach PSNR > 30 before spending time on O256 watermark training.

### 16.1 Smoke test command

Use this first to make sure the script and path are correct:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_decoder_ft_coco256_smoke100_seed61 \
  --image_size 256 \
  --steps 100 \
  --batch_size 2 \
  --lr 1e-5 \
  --eval_images 20 \
  --eval_every 50 \
  --log_every 10 \
  --seed 61 \
  2>&1 | tee logs/sd21_decoder_ft_coco256_smoke100_seed61_train.log
```

Smoke-test screening:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_smoke100=results/vae_finetune/sd21_decoder_ft_coco256_smoke100_seed61/final \
  --output_dir results/diagnostics/vae_screening_decoder_ft_smoke100_seed61 \
  --image_size 256 \
  --max_images 100 \
  2>&1 | tee logs/vae_screening_decoder_ft_smoke100_seed61.log
```

### 16.2 Main 10k-step decoder-only run

Run this only after the smoke test works:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61 \
  --image_size 256 \
  --steps 10000 \
  --batch_size 2 \
  --lr 1e-5 \
  --eval_images 100 \
  --eval_every 1000 \
  --log_every 50 \
  --seed 61 \
  2>&1 | tee logs/sd21_decoder_ft_coco256_10ksteps_seed61_train.log
```

Full eval1000 screening after training:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --output_dir results/diagnostics/vae_screening_decoder_ft_10ksteps_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_decoder_ft_10ksteps_seed61.log
```

Interpretation:

```text
If sd21_decoder_ft_10k >= 30 dB:
  Use this VAE path in a new O256 msg32 watermark run.

If sd21_decoder_ft_10k is still around 24-26 dB:
  Do not run O256 watermark yet. Decoder-only fine-tune is insufficient.
```

## 17. Decoder + post_quant_conv VAE fine-tune

Purpose: Section 16 improved the VAE ceiling from `24.8510` to `26.0018`, but did not reach PSNR 30+. This run keeps the encoder frozen, but trains both:

```text
post_quant_conv + decoder
```

Everything else remains the same as Section 16, so the only intended difference is whether post-quant adaptation helps reconstruction.

### 17.1 Smoke test command

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_decoder_postquant_ft_coco256_smoke100_seed61 \
  --image_size 256 \
  --steps 100 \
  --batch_size 2 \
  --lr 1e-5 \
  --eval_images 20 \
  --eval_every 50 \
  --log_every 10 \
  --train_post_quant_conv \
  --seed 61 \
  2>&1 | tee logs/sd21_decoder_postquant_ft_coco256_smoke100_seed61_train.log
```

Smoke-test screening:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_decoder_postquant_ft_smoke100=results/vae_finetune/sd21_decoder_postquant_ft_coco256_smoke100_seed61/final \
  --output_dir results/diagnostics/vae_screening_decoder_postquant_ft_smoke100_seed61 \
  --image_size 256 \
  --max_images 100 \
  2>&1 | tee logs/vae_screening_decoder_postquant_ft_smoke100_seed61.log
```

### 17.2 Main 10k-step decoder + post_quant run

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_decoder_postquant_ft_coco256_10ksteps_seed61 \
  --image_size 256 \
  --steps 10000 \
  --batch_size 2 \
  --lr 1e-5 \
  --eval_images 100 \
  --eval_every 1000 \
  --log_every 50 \
  --train_post_quant_conv \
  --seed 61 \
  2>&1 | tee logs/sd21_decoder_postquant_ft_coco256_10ksteps_seed61_train.log
```

Full eval1000 screening after training:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_decoder_postquant_ft_10k=results/vae_finetune/sd21_decoder_postquant_ft_coco256_10ksteps_seed61/final \
  --output_dir results/diagnostics/vae_screening_decoder_postquant_ft_10ksteps_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_decoder_postquant_ft_10ksteps_seed61.log
```

If CUDA reports `RuntimeError: FIND was unable to find an engine to execute this computation`, rerun the same training command with:

```bash
  --disable_cudnn
```

Interpretation:

```text
If decoder + post_quant_conv is still under 27 dB:
  This low-risk VAE adaptation path is probably not enough.

If it reaches 27-30 dB:
  Consider longer steps or LR search before watermark retraining.

If it reaches >= 30 dB:
  Use this VAE path in a new O256 msg32 watermark experiment.
```

## 18. Full VAE fine-tune upper-bound diagnosis

Purpose: Section 16/17 show that low-risk decoder-only adaptation improves the VAE ceiling only to about 26 dB. This run opens the full VAE:

```text
encoder + quant_conv + post_quant_conv + decoder
```

This is an upper-bound diagnosis. It changes the latent distribution, so the resulting VAE cannot be mixed with old O256 checkpoints. If it reaches PSNR 30+, a new O256 watermark model must be trained from scratch with this VAE.

### 18.1 Smoke test command

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_fullvae_ft_coco256_smoke100_seed61 \
  --image_size 256 \
  --steps 100 \
  --batch_size 2 \
  --lr 1e-6 \
  --mse_weight 1.0 \
  --l1_weight 0.25 \
  --kl_weight 1e-6 \
  --eval_images 20 \
  --eval_every 50 \
  --log_every 10 \
  --train_full_vae \
  --seed 61 \
  2>&1 | tee logs/sd21_fullvae_ft_coco256_smoke100_seed61_train.log
```

Smoke-test screening:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_fullvae_ft_smoke100=results/vae_finetune/sd21_fullvae_ft_coco256_smoke100_seed61/final \
  --output_dir results/diagnostics/vae_screening_fullvae_ft_smoke100_seed61 \
  --image_size 256 \
  --max_images 100 \
  2>&1 | tee logs/vae_screening_fullvae_ft_smoke100_seed61.log
```

### 18.2 Main 10k-step full VAE run

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae /home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_fullvae_ft_coco256_10ksteps_seed61 \
  --image_size 256 \
  --steps 10000 \
  --batch_size 2 \
  --lr 1e-6 \
  --mse_weight 1.0 \
  --l1_weight 0.25 \
  --kl_weight 1e-6 \
  --eval_images 100 \
  --eval_every 1000 \
  --log_every 50 \
  --train_full_vae \
  --seed 61 \
  2>&1 | tee logs/sd21_fullvae_ft_coco256_10ksteps_seed61_train.log
```

Full eval1000 screening after training:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_decoder_postquant_ft_10k=results/vae_finetune/sd21_decoder_postquant_ft_coco256_10ksteps_seed61/final \
  --vae sd21_fullvae_ft_10k=results/vae_finetune/sd21_fullvae_ft_coco256_10ksteps_seed61/final \
  --output_dir results/diagnostics/vae_screening_fullvae_ft_10ksteps_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_fullvae_ft_10ksteps_seed61.log
```

If CUDA reports `RuntimeError: FIND was unable to find an engine to execute this computation`, rerun the same training command with:

```bash
  --disable_cudnn
```

Interpretation:

```text
If full VAE is still below 27 dB:
  This SD2.1 VAE architecture/path is unlikely to satisfy GT PSNR 30 in this setup.

If full VAE reaches 27-30 dB:
  Consider longer steps or LR search.

If full VAE reaches >= 30 dB:
  Use this VAE path in a new O256 msg32 run, trained from scratch.
```

## 19. Alternative VAE / decoder screening

Goal:

```text
Before another long O256 training run, test whether already-downloaded VAE
candidates can improve the pure VAE round-trip ceiling on the same eval1000 set.

This does not train the watermark model. It only answers:
GT image -> VAE encode/decode -> reconstruction

If this still cannot approach 30 dB, the candidate is not worth using for a
full O256 retraining run.
```

Run:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_decoder_postquant_ft_10k=results/vae_finetune/sd21_decoder_postquant_ft_coco256_10ksteps_seed61/final \
  --vae sd21_fullvae_ft_10k=results/vae_finetune/sd21_fullvae_ft_coco256_10ksteps_seed61/final \
  --vae sdxl_vae_fp16_fix=/home/jeffrey/projects/VAE_candidates/madebyollin_sdxl-vae-fp16-fix \
  --vae consistency_decoder=/home/jeffrey/projects/VAE_candidates/openai_consistency-decoder \
  --output_dir results/diagnostics/vae_screening_alternative_candidates_eval1000_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_alternative_candidates_eval1000_seed61.log
```

Expected output:

```text
results/diagnostics/vae_screening_alternative_candidates_eval1000_seed61/report.txt
results/diagnostics/vae_screening_alternative_candidates_eval1000_seed61/report.json
results/diagnostics/vae_screening_alternative_candidates_eval1000_seed61/summary.csv
```

Decision rule:

```text
If no candidate reaches at least 30 dB:
  Do not start a full O256 retraining with a replacement VAE yet.

If one candidate reaches 30 dB or above:
  Prepare a new O256 config using that VAE path and retrain from scratch.

If a candidate improves only slightly, for example 26 -> 27 dB:
  Record it as evidence that VAE replacement alone is insufficient.
```

## 20. Oracle VAE residual correction sweep

Goal:

```text
This is an upper-bound diagnostic, not a deployable method.

It tests whether GT-PSNR can be recovered if we add back the cover detail lost
by the VAE round-trip:

vae_cover = D(E(cover))
residual = cover - vae_cover
corrected = watermarked + beta * residual

If this improves PSNR toward 30 dB, the bottleneck is strongly tied to the VAE
round-trip / latent bottleneck rather than only the watermark loss.
```

Run PSNR/SSIM sweep on the high-clean-BA O256 msg32 diagnostic output:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/oracle_vae_residual_sweep.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  --output_root results/diagnostics/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  --betas 0,0.25,0.5,0.75,1.0 \
  --save_debug \
  2>&1 | tee logs/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_seed61.log
```

Expected output:

```text
results/diagnostics/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/report.txt
results/diagnostics/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/summary.csv
results/diagnostics/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/debug/
```

Important:

```text
This experiment currently evaluates PSNR/SSIM only.
BA for corrected images requires the exact corresponding decoder checkpoint.
Some 20k diagnostic checkpoints were intentionally cleaned to recover disk space,
so do not run calculate_ba on these corrected images unless the matching checkpoint
is restored or the run is repeated.
```

Decision rule:

```text
If beta=0.5-1.0 raises PSNR above 30:
  RGB residual/refinement is a promising direction.

If PSNR remains below 30:
  The visible degradation is not only from VAE cover detail loss.

If debug difference maps show obvious localized changes:
  Those artifacts can coexist with 20-26 dB PSNR; PSNR is an average pixel metric.
```

## 21. VAE ceiling stratified analysis

Goal:

```text
Group the eval1000 images by GT-vs-VAE-cover PSNR and show whether
GT-vs-watermarked PSNR follows the same strata.

This turns the VAE-ceiling observation into reportable evidence:
VAE-friendly images can reach 30+ dB, while VAE-unfriendly images are already
low before watermark-specific distortion.
```

Run:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/analyze_vae_ceiling_strata.py \
  --sweep_dir results/diagnostics/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  --beta 1.00 \
  --output_dir results/diagnostics/vae_ceiling_strata_O256_MSG32_CL3_PSNR36_HINGE20K_seed61 \
  --thresholds 20,25,30 \
  --samples_per_group 4 \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_seed61/CoCo/CoCo \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --save_visual_samples \
  2>&1 | tee logs/vae_ceiling_strata_O256_MSG32_CL3_PSNR36_HINGE20K_seed61.log
```

Expected output:

```text
results/diagnostics/vae_ceiling_strata_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/report.txt
results/diagnostics/vae_ceiling_strata_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/group_summary.csv
results/diagnostics/vae_ceiling_strata_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/selected_samples.csv
results/diagnostics/vae_ceiling_strata_O256_MSG32_CL3_PSNR36_HINGE20K_seed61/visual_samples/
```

Use in report:

```text
The correlation between GT-vs-VAE-cover PSNR and GT-vs-watermarked PSNR
quantifies how strongly the frozen VAE ceiling predicts final fidelity.
```

## 22. REPRO20K corrected evaluation paths

Important:

```text
eval_diffusion.py produced this run under a timestamped folder:

results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo

Do not use:

results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/CoCo/CoCo

That path does not exist for this run.
```

PSNR / SSIM:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_psnr_ssim.log
```

Clean BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

rm -rf results/attacked_Result/clean_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61
mkdir -p results/attacked_Result/clean_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/Clean

cp results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo/*.png \
  results/attacked_Result/clean_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/Clean/

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_clean_ba.log
```

Oracle VAE residual sweep:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/oracle_vae_residual_sweep.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  --output_root results/diagnostics/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61 \
  --betas 0,0.25,0.5,0.75,1.0 \
  --save_debug \
  2>&1 | tee logs/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61.log
```

Corrected beta=1 clean BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

rm -rf results/attacked_Result/clean_corrected_beta1_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61
mkdir -p results/attacked_Result/clean_corrected_beta1_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/Clean

cp results/diagnostics/oracle_vae_residual_sweep_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/beta_1p00/*.png \
  results/attacked_Result/clean_corrected_beta1_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/Clean/

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/clean_corrected_beta1_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_corrected_beta1_clean_ba.log
```

## 23. Extended VAE fine-tune after backup

Purpose:

```text
Test whether SD2.1 VAE reconstruction ceiling can move clearly beyond 26 dB
with higher-cost fine-tuning.

This is still a VAE diagnostic, not a watermark result.
Do not overwrite the original SD2.1 VAE.
All outputs are saved under results/vae_finetune/.
```

Backup status:

```text
Original VAE:
/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base/vae

Backup:
/home/jeffrey/projects/VAE_backups/sd21_original_vae_20260604

Backup checked on 2026-06-04:
original size: 958M
backup size:   958M
file count:    5 / 5
```

Decision rule:

```text
If decoder total30k is still < 27 dB:
  Stop long VAE fine-tuning. Prefer RGB refinement / residual correction.

If decoder total30k reaches 27-29 dB:
  Run decoder total50k.

If decoder total30k or total50k reaches >= 30 dB:
  Prepare a new O256 watermark training run using that VAE consistently.

Full-VAE total30k is upper-bound only:
  If it works, old O256 watermark checkpoints cannot be reused.
```

### 23.1 Decoder-only extension to total30k

This continues from the existing decoder-ft 10k VAE by running 20k more steps.
Because the script does not restore Adam optimizer state, this is a weight-level
continuation, not a perfect optimizer resume.

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_decoder_ft_coco256_total30k_from10k_seed61 \
  --image_size 256 \
  --steps 20000 \
  --batch_size 2 \
  --lr 1e-5 \
  --mse_weight 1.0 \
  --l1_weight 0.25 \
  --eval_images 100 \
  --eval_every 1000 \
  --save_every 0 \
  --seed 61 \
  2>&1 | tee logs/sd21_decoder_ft_coco256_total30k_from10k_seed61_train.log
```

Screen eval1000:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_decoder_ft_total30k=results/vae_finetune/sd21_decoder_ft_coco256_total30k_from10k_seed61/final \
  --output_dir results/diagnostics/vae_screening_decoder_ft_total30k_from10k_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_decoder_ft_total30k_from10k_seed61.log
```

Expected outputs:

```text
results/vae_finetune/sd21_decoder_ft_coco256_total30k_from10k_seed61/report.txt
results/vae_finetune/sd21_decoder_ft_coco256_total30k_from10k_seed61/final
results/diagnostics/vae_screening_decoder_ft_total30k_from10k_seed61/report.txt
```

### 23.2 Decoder-only extension to total50k

Run this only if total30k reaches at least about 27 dB or shows a clear upward
trend. This continues from total30k by running 20k more steps with a smaller LR.

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae results/vae_finetune/sd21_decoder_ft_coco256_total30k_from10k_seed61/final \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_decoder_ft_coco256_total50k_from30k_seed61 \
  --image_size 256 \
  --steps 20000 \
  --batch_size 2 \
  --lr 5e-6 \
  --mse_weight 1.0 \
  --l1_weight 0.25 \
  --eval_images 100 \
  --eval_every 1000 \
  --save_every 0 \
  --seed 61 \
  2>&1 | tee logs/sd21_decoder_ft_coco256_total50k_from30k_seed61_train.log
```

Screen eval1000:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_decoder_ft_total30k=results/vae_finetune/sd21_decoder_ft_coco256_total30k_from10k_seed61/final \
  --vae sd21_decoder_ft_total50k=results/vae_finetune/sd21_decoder_ft_coco256_total50k_from30k_seed61/final \
  --output_dir results/diagnostics/vae_screening_decoder_ft_total50k_from30k_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_decoder_ft_total50k_from30k_seed61.log
```

### 23.3 Full-VAE extension to total30k upper bound

Run this only as an upper-bound diagnostic. It changes the encoder and latent
distribution, so old O256 checkpoints cannot be reused with this VAE.

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_vae_decoder_finetune.py \
  --base_vae results/vae_finetune/sd21_fullvae_ft_coco256_10ksteps_seed61/final \
  --train_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/train/gt \
  --val_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/vae_finetune/sd21_fullvae_ft_coco256_total30k_from10k_seed61 \
  --image_size 256 \
  --steps 20000 \
  --batch_size 2 \
  --lr 5e-7 \
  --mse_weight 1.0 \
  --l1_weight 0.25 \
  --kl_weight 1e-6 \
  --train_full_vae \
  --eval_images 100 \
  --eval_every 1000 \
  --save_every 0 \
  --seed 61 \
  2>&1 | tee logs/sd21_fullvae_ft_coco256_total30k_from10k_seed61_train.log
```

Screen eval1000:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/screen_vae_reconstruction.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --vae sd21_current=/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base \
  --vae sd21_decoder_ft_10k=results/vae_finetune/sd21_decoder_ft_coco256_10ksteps_seed61/final \
  --vae sd21_fullvae_ft_10k=results/vae_finetune/sd21_fullvae_ft_coco256_10ksteps_seed61/final \
  --vae sd21_fullvae_ft_total30k=results/vae_finetune/sd21_fullvae_ft_coco256_total30k_from10k_seed61/final \
  --output_dir results/diagnostics/vae_screening_fullvae_ft_total30k_from10k_seed61 \
  --image_size 256 \
  2>&1 | tee logs/vae_screening_fullvae_ft_total30k_from10k_seed61.log
```

## 24. RGB refinement / residual correction diagnostic v1

Purpose:

```text
Train a lightweight post-VAE RGB residual refiner while freezing the O256
watermark checkpoint.

This tests whether a small refinement head can raise GT-vs-output PSNR while
preserving clean BA. It is not yet a formal method or final result.
```

Input run:

```text
O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61

Original output:
PSNR / SSIM = 24.2085 / 0.6940
clean BA = 95.93%

Oracle beta=1 correction:
PSNR / SSIM = 28.7882 / 0.9194
clean BA = 82.42%
```

Interpretation of this diagnostic:

```text
If learned RGB refinement raises PSNR above 26-28 while keeping BA close to 95%:
  This is a promising method direction.

If PSNR improves but BA drops strongly:
  The model repeats the oracle residual problem; need stronger message loss or
  watermark-aware design.

If PSNR barely improves:
  A small CNN refiner is insufficient; need a stronger refinement architecture
  or a different output path.
```

### 24.1 Train RGB refiner v1

This uses `watermarked image + cover image` as input. The cover image is
available during embedding, so this is not the same as test-time oracle
correction. However, because this first version trains on the existing eval1000
outputs, it is a diagnostic only.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61 \
  --image_size 256 \
  --steps 5000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1e-4 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.25 \
  --conditioning cover \
  --image_mse_weight 1.0 \
  --image_l1_weight 0.25 \
  --message_weight 10.0 \
  --regularize_weight 0.05 \
  --eval_every 500 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG10_5K_seed61_train.log
```

Expected output:

```text
results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61/report.txt
results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61/refiner_final.pth
results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61/refined_images/
results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61/attacked_clean/Clean/
```

### 24.2 Evaluate refined PSNR / SSIM

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61/refined_images \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG10_5K_seed61_psnr_ssim.log
```

### 24.3 Evaluate refined clean BA

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61/attacked_clean \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG10_5K_seed61_clean_ba.log
```

### 24.4 Optional: paper-matched attacks after clean success

Only run this if clean BA remains high and PSNR improves meaningfully.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_COVER_MSG10_5K_seed61/refined_images \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_RGBREFINE_COVER_MSG10_5K_seed61 \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG10_5K_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_RGBREFINE_COVER_MSG10_5K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG10_5K_seed61_paper_matched_ba.log
```

## 25. RGB refinement v2: image-priority loss balance

Purpose:

```text
Section 24 v1 preserved clean BA but reduced PSNR:

O256 original: PSNR 24.2085, clean BA 95.93%
RGBREFINE v1: PSNR 22.9810, clean BA 96.84%

This means message_weight=10 dominated the training objective.
v2 makes image fidelity dominate and keeps only a small message constraint.
```

Key change:

| Setting | v1 | v2 |
|---|---:|---:|
| image_mse_weight | 1.0 | 10.0 |
| image_l1_weight | 0.25 | 2.5 |
| message_weight | 10.0 | 0.2 |
| regularize_weight | 0.05 | 0.01 |
| max_delta | 0.25 | 0.50 |

Decision rule:

```text
If PSNR rises and clean BA remains >= 90%:
  v2 is promising. Run paper-matched attack BA.

If PSNR rises but clean BA drops below 90%:
  run Section 26 two-stage message-aware fine-tune from the v2 checkpoint.

If PSNR does not rise above original 24.2085:
  small CNN refinement is insufficient in this form.
```

### 25.1 Train image-priority RGB refiner v2

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61 \
  --image_size 256 \
  --steps 5000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1e-4 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.2 \
  --regularize_weight 0.01 \
  --eval_every 500 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61_train.log
```

### 25.2 Evaluate v2 PSNR / SSIM

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61/refined_images \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61_psnr_ssim.log
```

### 25.3 Evaluate v2 clean BA

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/refinement/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61/attacked_clean \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61_clean_ba.log
```

### 25.4 Optional: paper-matched attacks for v2

Only run this if v2 clean BA remains acceptable.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61/refined_images \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61 \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61_paper_matched_ba.log
```

## 26. Optional two-stage BA recovery from image-priority refiner

Run this only if Section 25 improves PSNR but clean BA drops too much.

Purpose:

```text
Start from the image-priority v2 checkpoint and do a short message-aware
fine-tune with smaller learning rate. This tests whether BA can be recovered
without losing all PSNR improvement.
```

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --resume_refiner results/refinement/O256_RGBREFINE_COVER_IMGPRI_M0P2_5K_seed61/refiner_final.pth \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_COVER_IMGPRI_THEN_MSG2_2K_seed61 \
  --image_size 256 \
  --steps 2000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 5e-5 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 2.0 \
  --regularize_weight 0.01 \
  --eval_every 250 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_THEN_MSG2_2K_seed61_train.log
```

Evaluate PSNR:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_COVER_IMGPRI_THEN_MSG2_2K_seed61/refined_images \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_THEN_MSG2_2K_seed61_psnr_ssim.log
```

Evaluate clean BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/refinement/O256_RGBREFINE_COVER_IMGPRI_THEN_MSG2_2K_seed61/attacked_clean \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_COVER_IMGPRI_THEN_MSG2_2K_seed61_clean_ba.log
```

## RGB refinement v7: v5 -> stronger attack-aware proxy

Purpose:

```text
Start from the best PSNR-clean trade-off checkpoint v5 and increase
attack-aware message loss from 0.25 to 0.50.

This is a short headroom test:
- if PSNR stays above 30 and paper-matched BA improves clearly, continue this route;
- if PSNR drops or attack BA barely improves, stop scalar searching on refiner loss weights.
```

Train:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --resume_refiner results/refinement/O256_RGBREFINE_COVER_IMGPRI_THEN_MSG0P75_1K_seed61/refiner_final.pth \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61 \
  --image_size 256 \
  --steps 500 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 2e-5 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.75 \
  --attack_message_weight 0.50 \
  --attack_profile nongeom_light \
  --attack_noise_sigma 0.03 \
  --attack_blur_sigma 1.0 \
  --attack_jpeg_levels 32 \
  --attack_dropout_prob 0.05 \
  --regularize_weight 0.01 \
  --eval_every 100 \
  --log_every 25 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61_train.log
```

Evaluate PSNR / SSIM:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61/refined_images \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61_psnr_ssim.log
```

Evaluate clean BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/refinement/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61/attacked_clean \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61_clean_ba.log
```

Run paper-matched attacks:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61/refined_images \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61 \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61_paper_matched_attack.log
```

Evaluate paper-matched BA:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_COVER_MSG0P75_AWARE0P50_500_seed61_paper_matched_ba.log
```

## Formal RGB refinement sanity: train2000 -> unseen eval1000

Purpose:

```text
The previous RGB refinement v5/v6/v7 diagnostics trained and evaluated on the
same eval1000 outputs. This section checks whether the RGB refiner can learn
from training images and generalize to unseen eval1000 images.

This is a low-cost formal sanity:
- train split subset: 2000 images
- eval split: fixed unseen eval1000 images
- base O256 checkpoint: O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61
- refinement route: image-priority -> message recovery -> attack-aware recovery
```

### 1. Create train2000 dataset alias

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/create_coco_split_alias.py \
  --source_root /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen \
  --source_split train \
  --output_root results/formal_refine_datasets/CoCo256_train2000_as_val_seed61 \
  --num_images 2000
```

### 2. Generate O256 watermarked outputs on train2000

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u eval_diffusion.py \
  --config CoCo_refiner_train2000_as_val_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --image_folder results/test_result/O256_FORMAL_TRAIN2000_BASE_seed61 \
  --run_name train2000 \
  --seed 61 \
  2>&1 | tee logs/O256_FORMAL_TRAIN2000_BASE_seed61_eval.log
```

Expected watermarked train output:

```text
results/test_result/O256_FORMAL_TRAIN2000_BASE_seed61/train2000/CoCo
```

### 3. Formal refiner stage A: image-priority

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --watermarked_dir results/test_result/O256_FORMAL_TRAIN2000_BASE_seed61/train2000/CoCo \
  --gt_dir results/formal_refine_datasets/CoCo256_train2000_as_val_seed61/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_IMGPRI_M0P2_5K_seed61 \
  --image_size 256 \
  --steps 5000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1e-4 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.2 \
  --regularize_weight 0.01 \
  --eval_every 500 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_IMGPRI_M0P2_5K_seed61_train.log
```

### 4. Formal refiner stage B: message recovery

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --resume_refiner results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_IMGPRI_M0P2_5K_seed61/refiner_final.pth \
  --watermarked_dir results/test_result/O256_FORMAL_TRAIN2000_BASE_seed61/train2000/CoCo \
  --gt_dir results/formal_refine_datasets/CoCo256_train2000_as_val_seed61/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_IMGPRI_THEN_MSG0P75_1K_seed61 \
  --image_size 256 \
  --steps 1000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 5e-5 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.75 \
  --regularize_weight 0.01 \
  --eval_every 250 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_IMGPRI_THEN_MSG0P75_1K_seed61_train.log
```

### 5. Formal refiner stage C: attack-aware recovery

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u scripts/train_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --resume_refiner results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_IMGPRI_THEN_MSG0P75_1K_seed61/refiner_final.pth \
  --watermarked_dir results/test_result/O256_FORMAL_TRAIN2000_BASE_seed61/train2000/CoCo \
  --gt_dir results/formal_refine_datasets/CoCo256_train2000_as_val_seed61/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_MSG0P75_AWARE0P25_500_seed61 \
  --image_size 256 \
  --steps 500 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 2e-5 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --image_mse_weight 10.0 \
  --image_l1_weight 2.5 \
  --message_weight 0.75 \
  --attack_message_weight 0.25 \
  --attack_profile nongeom_light \
  --attack_noise_sigma 0.03 \
  --attack_blur_sigma 1.0 \
  --attack_jpeg_levels 32 \
  --attack_dropout_prob 0.05 \
  --regularize_weight 0.01 \
  --eval_every 100 \
  --log_every 25 \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_MSG0P75_AWARE0P25_500_seed61_train.log
```

### 6. Apply formal refiner to unseen eval1000

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/apply_rgb_refinement.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --refiner results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_MSG0P75_AWARE0P25_500_seed61/refiner_final.pth \
  --watermarked_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --output_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61 \
  --image_size 256 \
  --batch_size 4 \
  --num_workers 2 \
  --hidden_channels 48 \
  --depth 5 \
  --max_delta 0.50 \
  --conditioning cover \
  --seed 61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61_apply.log
```

### 7. Evaluate unseen eval1000 PSNR / SSIM

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61/refined_images \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61_psnr_ssim.log
```

### 8. Evaluate unseen eval1000 clean BA

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61/attacked_clean \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61_clean_ba.log
```

### 9. Optional paper-matched attack evaluation

Run only if unseen eval1000 PSNR > 30 and clean BA is acceptable.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/refinement/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61/refined_images \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61 \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61_paper_matched_attack.log

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml \
  --resume Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61 \
  --gt_message_dir results/test_result/O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61/20260603_230732/CoCo \
  2>&1 | tee logs/O256_RGBREFINE_FORMAL_TRAIN2000_APPLY_EVAL1000_MSG0P75_AWARE0P25_seed61_paper_matched_ba.log
```

## Formal RGB refinement scale-up: train5000 -> unseen eval1000

Purpose:

```text
Scale the current best formal RGB-refinement setting from train2000 to train5000.
This is the next formal experiment after the first RGB-refinement breakthrough:

- keep the same base O256 checkpoint
- keep the same Stage A -> Stage B recipe
- do not add Stage C attack-aware refinement yet
- evaluate on the same unseen eval1000 split

The full copy-paste commands are stored in:

commands_rgb_refine_formal_train5000.md
```

Primary command file:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
sed -n '1,260p' commands_rgb_refine_formal_train5000.md
```

New config:

```text
configs/CoCo_refiner_train5000_as_val_ours256_msg32_cl3_psnr36_hinge20k.yml
```

Main expected outputs:

```text
results/test_result/O256_FORMAL_TRAIN5000_BASE_seed61/train5000/CoCo
results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_M0P2_5K_seed61
results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_IMGPRI_THEN_MSG0P75_1K_seed61
results/refinement/O256_RGBREFINE_FORMAL_TRAIN5000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61
```

Decision rule:

```text
PSNR >= 30
Clean BA >= 88%
Paper-matched avg BA >= 65%
```
