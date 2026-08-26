# Experiment 9: Controlled Development-Set Method Comparison

## Classification Correction

Classification: `E` (development / exploratory evidence), not the final
Chapter 4 test.

The COCO split previously called `unseen eval1000` is disjoint from training
images, but it was reused for Stage, loss, checkpoint, and module candidate
selection. It is therefore called `dev1000` from 2026-06-11 onward.

The results below remain valid as a controlled development comparison. They
must not be presented as unbiased final-test numbers. The frozen methods must
be evaluated once on `finaltest1000_locked` before this table becomes M01.

## Claim Boundary

The `DiffW-like` row is our locally trained 256x256 / 32-bit controlled
baseline. It is not the untouched 128x128 DiffW paper result.

The DiffW paper reports that encoded-image PSNR was deliberately controlled
to `33.5 +/- 0.5 dB` for its comparison tables. The paper does not fully
disclose the strength-control procedure required to reproduce that operating
point. Therefore:

- We may claim that O256-RGBREFINE improves fidelity over our controlled
  DiffW-like baseline and our O256 base under the protocol below.
- We may not claim that O256-RGBREFINE has higher fidelity than the official
  DiffW paper result.
- The official paper number must be shown as a reported reference, not mixed
  into the same directly comparable row.

## Locked Protocol

- COCO train10000: byte-identical manifest
- COCO dev1000 (formerly called unseen eval1000)
- Image size: 256x256
- Message length: 32 bits
- Main seed: 61
- Evaluation sampling steps: 3
- Robustness: 11 non-geometric attacks
- Geometry attacks: excluded

## Same and Different Settings

| Setting | DiffW-like | O256 base / O256-RGBREFINE | Alignment |
|---|---|---|---|
| Train images | same COCO train10000 manifest | same COCO train10000 manifest | same |
| Evaluation images | same unseen COCO eval1000 | same unseen COCO eval1000 | same |
| Image size | 256x256 | 256x256 | same |
| Message length | 32 bits | 32 bits | same |
| Main seed | 61 | 61 | same |
| Training batch size | 2 | 2 | same |
| Optimizer | Adam | Adam | same |
| Initial learning rate | 2e-5 | 2e-5 | same |
| Diffusion schedule | linear, 1000 steps | linear, 1000 steps | same |
| Evaluation sampling steps | 3 | 3 | same |
| Base training steps | 20k | 20k + 5k continuation | different |
| Continuation learning rate | none | 1e-5 | different |
| Message representation | linear | bitplane multi-scale | method difference |
| Training attack schedule | combined attack | identity -> nongeom-light | method difference |
| Latent VAE | no | frozen SD2.1 VAE | method difference |
| RGB refinement | no | refine10000 Stage A/B/B+/targeted C | method difference |

This is a development-set final-method candidate comparison. It is neither an
unbiased final-test result nor an identical training-budget claim.

## Official Paper Reference

| Source | Image size | Payload | PSNR condition | Directly comparable here? |
|---|---:|---:|---:|---|
| DiffW paper reported setting | 128x128 | 30/64/etc. | 33.5 +/- 0.5 dB, explicitly strength-controlled | No |
| Our DiffW-like controlled baseline | 256x256 | 32 | naturally obtained 26.6109 dB | Yes |

## Main Results

| Method | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
| DiffW-like | 26.6109 | 0.8466 | 100.00% | 98.89% | 92.65% | Paper_Dropout_r03 |
| O256 base, no module | 24.4281 | 0.7001 | 98.57% | 80.99% | 63.35% | Paper_GaussianNoise_arg03 |
| O256-RGBREFINE | 31.5358 | 0.9306 | 94.41% | 80.54% | 66.96% | Paper_GaussianNoise_arg03 |

## Per-Attack BA

| Attack | DiffW-like | O256 base | O256-RGBREFINE | Final - DiffW-like |
|---|---:|---:|---:|---:|
| Paper_Dropout_r03 | 92.65 | 65.97 | 67.22 | -25.43 |
| Paper_GaussianBlur_sigma2 | 100.00 | 92.08 | 90.21 | -9.79 |
| Paper_GaussianBlur_sigma3 | 100.00 | 89.91 | 89.19 | -10.81 |
| Paper_GaussianBlur_sigma4 | 100.00 | 88.99 | 88.56 | -11.44 |
| Paper_GaussianNoise_arg03 | 96.33 | 63.35 | 66.96 | -29.37 |
| Paper_GaussianNoise_sigma03_var009 | 100.00 | 72.03 | 74.35 | -25.65 |
| Paper_JPEG_Q30 | 100.00 | 83.24 | 80.88 | -19.12 |
| Paper_JPEG_Q40 | 100.00 | 87.42 | 84.35 | -15.65 |
| Paper_JPEG_Q50 | 100.00 | 89.71 | 86.63 | -13.37 |
| Paper_MedianFilter_w5 | 100.00 | 93.35 | 90.33 | -9.67 |
| Paper_SaltPepper_r02 | 98.84 | 64.82 | 67.30 | -31.54 |

## Source Paths

- DiffW PSNR log: `/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/logs/exp9_diffw_like_train10000_msg32_20k_seed61_psnr_ssim.log`
- DiffW clean BA: `/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/results/attacked_Result/clean_exp9_diffw_like_train10000_msg32_20k_seed61/ba_report.txt`
- DiffW attack BA: `/home/jeffrey/projects/diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline/results/attacked_Result/nongeom_exp9_diffw_like_train10000_msg32_20k_seed61/ba_report.txt`
- O256 base PSNR log: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/logs/exp2b_nongeom_train10000_base20k_then_light5k_seed61_psnr_ssim.log`
- O256 base clean BA: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/attacked_Result/clean_exp2b_train10000_base20k_then_light5k_seed61/ba_report.txt`
- O256 base attack BA: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/attacked_Result/exp2b_nongeom_train10000_base20k_then_light5k_seed61/ba_report.txt`
- O256 report: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/refinement/refine10000_targetedformal_C500_apply_eval1000_seed61/report.json`
- O256 clean BA: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/refinement/refine10000_targetedformal_C500_apply_eval1000_seed61/attacked_clean/ba_report.txt`
- O256 attack BA: `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/results/attacked_Result/exp6_nongeom_refine10000_targetedformal_C500_seed61/ba_report.txt`
