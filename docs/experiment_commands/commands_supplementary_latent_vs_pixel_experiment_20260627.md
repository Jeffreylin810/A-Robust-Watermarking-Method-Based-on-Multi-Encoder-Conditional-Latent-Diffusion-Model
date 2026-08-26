# Supplementary Latent vs Pixel Experiment Commands

用途：在 **目前 O256-PAPER 專案** 內建立一組乾淨的 supplementary comparison，用 `train1000 / eval1000` 做 proof-of-concept，比較：

1. `latent bitplane`（matched O256 miniature reference）
2. `latent linear`（message-representation ablation）
3. `pixel bitplane`（matched pixel-space baseline）

注意：

- 這不是 formal `locked finaltest1000` 主結果。
- 不覆蓋任何正式 checkpoint。
- 所有新輸出都放到獨立 supplementary root。

---

## Shortcut：把剩下步驟交給自動 pipeline

如果你已經手動啟動過 `pixel-bitplane identity20k`，不想再一條條貼，現在可以直接用：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_supplementary_latent_vs_pixel_pipeline_20260627.sh
```

如果你想用一個語意更明確的入口，也可以直接跑：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_supplementary_latent_vs_pixel_continue_after_pixel_identity20k_20260627.sh
```

這支腳本會：

1. 確認 supplementary scaffold/configs 存在
2. 等待 `pixel-bitplane identity20k` 的 final checkpoint（如果它還在跑）
3. 自動接著做：
   - pixel bitplane base30k + eval
   - latent linear identity20k/base30k + eval
   - latent bitplane identity20k/base30k + eval
   - VAE round-trip survival tests
4. 已完成的步驟會自動跳過

注意：

- 如果 `pixel-bitplane identity20k` 沒有真的產生 final checkpoint，腳本不會硬接下去，而是會停下來提醒你。
- 預設最長等待前置 checkpoint `1440` 分鐘；若你想縮短等待時間，可以：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

WAIT_MINUTES=720 bash scripts/run_supplementary_latent_vs_pixel_pipeline_20260627.sh
```

---

## 0. 先建立實驗骨架與 generated configs

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python \
  scripts/setup_supplementary_latent_vs_pixel_experiment.py \
  --date_tag 20260627
```

建立後，主要根目錄會是：

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627
```

---

## 1. Experiment A：O256-PixelBase matched baseline

### 1A. train1000 pixel-bitplane identity20k

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train1000_supp_pixel_bitplane_identity20k_seed61_20260627.yml \
  --resume ./ckpts/nonexistent_supp_pixel_bitplane_identity20k_seed61_start.pth.tar \
  --image_folder results/validation/supp_pixel_bitplane_identity20k_seed61_20260627 \
  --seed 61
```

### 1B. continue to pixel-bitplane base30k

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train1000_supp_pixel_bitplane_identity20k_then_nongeomlight10k_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/pixel_bitplane_identity20k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --image_folder results/validation/supp_pixel_bitplane_base30k_seed61_20260627 \
  --seed 61
```

### 1C. eval1000 unseen + PSNR/SSIM + LPIPS + clean BA + 11 attacks

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python -u eval_diffusion.py \
  --config CoCo_eval1000_supp_pixel_bitplane_train1000_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/pixel_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --image_folder /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/eval_outputs \
  --run_name pixel_bitplane_base30k_eval1000 \
  --seed 61

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_psnr_ssim.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train1000_eval1000_unseen/data/CoCo/val/gt \
  --results_path /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/eval_outputs/pixel_bitplane_base30k_eval1000/CoCo/CoCo

/home/jeffrey/miniconda3/envs/diffw/bin/python scripts/calculate_lpips_dir.py \
  --gt_dir /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train1000_eval1000_unseen/data/CoCo/val/gt \
  --results_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/eval_outputs/pixel_bitplane_base30k_eval1000/CoCo/CoCo \
  --output_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/lpips/pixel_bitplane_base30k_eval1000 \
  --batch_size 8 \
  --net alex

mkdir -p /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/attacked_Result/pixel_bitplane_base30k_clean
ln -sfn /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/eval_outputs/pixel_bitplane_base30k_eval1000/CoCo/CoCo \
  /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/attacked_Result/pixel_bitplane_base30k_clean/Clean

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_supp_pixel_bitplane_train1000_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/pixel_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --attack_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/attacked_Result/pixel_bitplane_base30k_clean \
  --gt_message_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/eval_outputs/pixel_bitplane_base30k_eval1000/CoCo/CoCo

/home/jeffrey/miniconda3/envs/diffw/bin/python attack_eval_diffw_paper_matched.py \
  --gt_path /home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train1000_eval1000_unseen/data/CoCo/val/gt \
  --results_path /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/eval_outputs/pixel_bitplane_base30k_eval1000/CoCo/CoCo \
  --output_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/attacked_Result \
  --run_name pixel_bitplane_base30k_nongeom \
  --attack_scope nongeom \
  --seed 61 \
  --overwrite

/home/jeffrey/miniconda3/envs/diffw/bin/python calculate_ba.py \
  --config CoCo_eval1000_supp_pixel_bitplane_train1000_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/pixel_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --attack_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/attacked_Result/pixel_bitplane_base30k_nongeom \
  --gt_message_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/eval_outputs/pixel_bitplane_base30k_eval1000/CoCo/CoCo
```

---

## 2. Experiment C：Latent linear vs bitplane matched ablation

### 2A. latent bitplane identity20k -> base30k

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train1000_supp_latent_bitplane_identity20k_seed61_20260627.yml \
  --resume ./ckpts/nonexistent_supp_latent_bitplane_identity20k_seed61_start.pth.tar \
  --image_folder results/validation/supp_latent_bitplane_identity20k_seed61_20260627 \
  --seed 61

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train1000_supp_latent_bitplane_identity20k_then_nongeomlight10k_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/latent_bitplane_identity20k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --image_folder results/validation/supp_latent_bitplane_base30k_seed61_20260627 \
  --seed 61
```

### 2B. latent linear identity20k -> base30k

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train1000_supp_latent_linear_identity20k_seed61_20260627.yml \
  --resume ./ckpts/nonexistent_supp_latent_linear_identity20k_seed61_start.pth.tar \
  --image_folder results/validation/supp_latent_linear_identity20k_seed61_20260627 \
  --seed 61

/usr/bin/time -p /home/jeffrey/miniconda3/envs/diffw/bin/python -u train_diffusion.py \
  --config CoCo_train1000_supp_latent_linear_identity20k_then_nongeomlight10k_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/latent_linear_identity20k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --image_folder results/validation/supp_latent_linear_base30k_seed61_20260627 \
  --seed 61
```

### 2C. eval latent bitplane / latent linear

把上面 `Experiment A / 1C` 那一段指令各跑兩次，只把：

- config 換成：
  - `CoCo_eval1000_supp_latent_bitplane_train1000_seed61_20260627.yml`
  - `CoCo_eval1000_supp_latent_linear_train1000_seed61_20260627.yml`
- checkpoint 換成：
  - `runs/latent_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar`
  - `runs/latent_linear_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar`
- run_name 換成：
  - `latent_bitplane_base30k_eval1000`
  - `latent_linear_base30k_eval1000`

---

## 3. Experiment B：VAE round-trip survival test

### 3A. latent bitplane round-trip

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python \
  scripts/evaluate_roundtrip_survival.py \
  --config CoCo_eval1000_supp_latent_bitplane_train1000_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/latent_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --test_set CoCo \
  --seed 61 \
  --num_images 128 \
  --output_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/roundtrip/latent_bitplane_base30k
```

### 3B. pixel bitplane round-trip

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python \
  scripts/evaluate_roundtrip_survival.py \
  --config CoCo_eval1000_supp_pixel_bitplane_train1000_seed61_20260627.yml \
  --resume /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/runs/pixel_bitplane_base30k_train1000/ckpts/CoCo_ddpm_final.pth.tar \
  --test_set CoCo \
  --seed 61 \
  --num_images 128 \
  --output_dir /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_20260627/roundtrip/pixel_bitplane_base30k
```

---

## 4. Existing-domain-shift / cost-to-target / reuse lines

### 4A. Three-method non-COCO domain shift (already scaffolded earlier)

Commands file:

```text
/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/commands_non_coco_domain_shift_v3_with_diffw_like_seed61.md
```

### 4B. DiffW-like cost-to-target

Quick 40k:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh 40000 quick
```

Full 40k:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh 40000 full
```

Note:

- Current evidence already suggests `40k quick` is probably enough as a negative supplementary result.
- I do **not** recommend jumping to `60k / 80k / 100k` before reviewing 40k full.

### 4C. RGB refinement reuse amortization accounting

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

/home/jeffrey/miniconda3/envs/diffw/bin/python \
  scripts/analyze_rgb_refinement_reuse_amortization.py \
  --output_root /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/rgb_refinement_reuse_amortization_20260627
```

---

## 5. What to run first

If we want the highest signal-to-cost order, I recommend:

1. `setup_supplementary_latent_vs_pixel_experiment.py`
2. `pixel bitplane base30k`
3. `latent linear base30k`
4. `latent bitplane base30k`
5. `roundtrip survival`
6. only then decide whether we still need more `cost-to-target`
