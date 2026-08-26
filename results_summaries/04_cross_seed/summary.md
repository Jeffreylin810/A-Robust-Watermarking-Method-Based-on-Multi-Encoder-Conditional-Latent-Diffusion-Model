# Experiment 7b corrected: seed62 cross-seed module reproducibility

## Controlled design

- Dataset/split, architecture, message length, training steps, loss weights,
  VAE, attack profile, and evaluation images are unchanged.
- The only intended base-training change is `seed: 61 -> 62`.
- Base training is verified as `identity-only 20k -> nongeom-light 5k`.
- The RGB refinement module is **not retrained** in this experiment.
- Transferred module:
  `results/refinement/refine10000_targetedformal_C500_seed61/refiner_final.pth`

## Results

| Version | PSNR | SSIM | Clean BA | Nongeom Avg BA | Minimum BA | Minimum attack |
|---|---:|---:|---:|---:|---:|---|
| seed62 base, no module | 24.4879 | 0.7035 | 98.61% | 84.78% | 67.93% | Paper_Dropout_r03 |
| seed61 + seed61 module | 31.5358 | 0.9306 | 94.41% | 80.54% | 66.96% | Paper_GaussianNoise_arg03 |
| seed62 + transferred seed61 module | 31.1335 | 0.9313 | 96.04% | 86.75% | 72.28% | Paper_Dropout_r03 |

## Pre-registered reproducibility checks

| Check | Seed62 - Seed61 | Tolerance | Result |
|---|---:|---:|---|
| Seed62 base Clean BA precondition | 98.61% | >= 90.0% | PASS |
| PSNR | -0.4022 dB | absolute delta <= 1.0 dB | PASS |
| Clean BA | +1.63 pp | absolute delta <= 3.0 pp | PASS |
| Nongeom Avg BA | +6.21 pp | absolute delta <= 5.0 pp | FAIL |
| Minimum BA | +5.32 pp | may not drop more than 5.0 pp | PASS |

These thresholds test **cross-seed reproducibility**, not the final thesis quality gate.

## Transfer usability checks

For transfer usability, positive improvements are not treated as failures.

| Check | Seed62 - Seed61 | Allowed negative degradation | Result |
|---|---:|---:|---|
| PSNR | -0.4022 dB | may not drop more than 1.0 dB | PASS |
| Clean BA | +1.63 pp | may not drop more than 3.0 pp | PASS |
| Nongeom Avg BA | +6.21 pp | may not drop more than 5.0 pp | PASS |
| Minimum BA | +5.32 pp | may not drop more than 5.0 pp | PASS |

## Decision

PASS TRANSFER / STRICT EQUIVALENCE FLAG：seed61 模組可直接轉移到 seed62，所有品質指標都沒有超出允許的負向退化；實際上 Clean BA、平均 BA 與最低 BA 都更高。嚴格重現性檢查只有平均 BA 因正向提升 +6.21 pp 超過 absolute 5 pp 而標記 FAIL，代表不同 seed 帶來可觀變異，不代表模組轉移失敗。不需要重訓 seed62 專屬模組。
