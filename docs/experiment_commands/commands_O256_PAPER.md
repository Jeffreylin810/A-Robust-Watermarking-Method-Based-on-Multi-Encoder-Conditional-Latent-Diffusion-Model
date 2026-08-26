# O256-PAPER Commands

Run from:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
```

## Train

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train10000_ours256_msg64_paper_protocol_100epoch.yml \
  --resume ./ckpts/nonexistent_O256_PAPER_start.pth.tar \
  --image_folder results/validation/O256_PAPER_seed61 \
  --seed 61 2>&1 | tee logs/O256_PAPER_seed61_train.log
```

## Save Final Checkpoint

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

cp ckpts/CoCo_ddpm_final.pth.tar \
  Result/CoCo_ddpm_O256_PAPER_seed61_final.pth.tar
```

## Eval

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python -u eval_diffusion.py \
  --config CoCo_eval1000_ours256_msg64_paper_protocol.yml \
  --resume Result/CoCo_ddpm_O256_PAPER_seed61_final.pth.tar \
  --test_set CoCo \
  --image_folder results/test_result \
  --run_name O256_PAPER_seed61 \
  --seed 61 2>&1 | tee logs/O256_PAPER_seed61_eval.log
```

## PSNR / SSIM

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/test_result/O256_PAPER_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_PAPER_seed61_psnr_ssim.log
```

## Paper-Matched Attacks

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python -u attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_eval1000_unseen/data/CoCo/val/gt \
  --results_path results/test_result/O256_PAPER_seed61/CoCo/CoCo \
  --output_dir results/attacked_Result \
  --run_name paper_matched_O256_PAPER_seed61 \
  --seed 61 2>&1 | tee logs/O256_PAPER_seed61_paper_matched_attack.log
```

## Paper-Matched BA

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_ours256_msg64_paper_protocol.yml \
  --resume Result/CoCo_ddpm_O256_PAPER_seed61_final.pth.tar \
  --attack_dir results/attacked_Result/paper_matched_O256_PAPER_seed61 \
  --gt_message_dir results/test_result/O256_PAPER_seed61/CoCo/CoCo \
  2>&1 | tee logs/O256_PAPER_seed61_paper_matched_ba.log
```
