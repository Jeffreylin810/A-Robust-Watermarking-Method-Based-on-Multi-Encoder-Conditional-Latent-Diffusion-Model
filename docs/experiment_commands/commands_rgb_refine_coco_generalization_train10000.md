# RGB Refinement COCO Generalization: Train10000 Stage-B

Purpose:

```text
Check whether the RGB refinement module can be trained on the full COCO train10000
split and still repair arbitrary unseen COCO eval1000 images to PSNR >= 30 while
keeping clean BA around the current formal anchor.
```

This experiment does **not** retrain the O256 latent watermark base. It freezes:

```text
Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_REPRO20K_seed61_final.pth.tar
```

Then it trains only a small RGB residual refiner.

## One-Shot Command

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

tmux new -s rgb_refine_train10000

./scripts/run_rgb_refine_formal_train10000.sh
```

Detach from tmux:

```text
Ctrl-b, then d
```

Reattach:

```bash
tmux attach -t rgb_refine_train10000
```

## What This Script Does

1. Creates a COCO train10000 alias dataset:

```text
results/formal_refine_datasets/CoCo256_train10000_as_val_seed61
```

2. Generates O256 base watermarked outputs for these 10000 training images:

```text
results/test_result/O256_FORMAL_TRAIN10000_BASE_seed61/train10000/CoCo
```

3. Trains Stage A image-priority refiner:

```text
results/refinement/O256_RGBREFINE_FORMAL_TRAIN10000_IMGPRI_M0P2_10K_seed61
```

4. Trains Stage B message-aware refiner:

```text
results/refinement/O256_RGBREFINE_FORMAL_TRAIN10000_IMGPRI_THEN_MSG0P75_2K_seed61
```

5. Applies Stage B to the fixed unseen eval1000 split:

```text
results/refinement/O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61/refined_images
```

6. Computes PSNR / SSIM and clean BA.

7. If the gate passes, runs paper-matched attack BA automatically.

## Gate

Paper-matched attacks are only run if:

```text
PSNR >= 30
Clean BA >= 89
```

Reason:

```text
If the train10000 refiner cannot preserve clean BA, it should not be promoted to
paper-matched robustness evaluation yet. First tune the Stage B message weight.
```

## Expected Output Logs

PSNR / SSIM:

```text
logs/O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_psnr_ssim.log
```

Clean BA:

```text
logs/O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61_clean_ba.log
```

Paper-matched BA, if gate passes:

```text
results/attacked_Result/paper_matched_O256_RGBREFINE_FORMAL_TRAIN10000_APPLY_EVAL1000_IMGPRI_THEN_MSG0P75_seed61/ba_report.txt
```

## How To Interpret

If train10000 Stage B is better than train5000 Stage B:

```text
This supports the claim that RGB refinement generalizes better with more COCO
training coverage and can be used as the formal O256 refinement anchor.
```

If train10000 Stage B is worse:

```text
Do not immediately discard RGB refinement. The larger train split may require a
different Stage B message_weight or more Stage B steps. Compare Stage A and Stage B
logs first.
```

Current train5000 anchor for comparison:

| Version | PSNR | SSIM | Clean BA | Paper-matched Avg BA |
|---|---:|---:|---:|---:|
| Formal train5000 Stage B | 30.1107 | 0.9116 | 89.12% | 65.39% |
