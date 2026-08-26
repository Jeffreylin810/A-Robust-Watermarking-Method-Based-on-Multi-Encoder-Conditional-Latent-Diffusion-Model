# Non-geometric Experiments 0-2

All commands exclude Crop, Cropout, Rotate, and every other geometric attack.

## Experiment 0: frozen base non-geometric diagnostic

No training. It measures the current `train10000-base20k` checkpoint before RGB refinement.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_exp0_base_nongeom_eval.sh
```

## Experiment 1: refine10000 Stage C

It loads the completed `refine10000-A10k-B2k` checkpoint and trains the same
RGB refiner for a conservative 500-step non-geometric Stage C. The O256 base,
VAE, and message decoder remain frozen.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_exp1_refine10000_stage_c_nongeom.sh
```

## Experiment 2: train10000 base20k non-geometric from scratch

This is a separate full O256 base-model training run. It does not initialize
from the identity-only checkpoint. Its final checkpoint is written to a unique
path under `Result/`.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
tmux new -s train10000_base20k_nongeom
bash scripts/run_exp2_train10000_base20k_nongeom.sh
```

Detach from tmux with `Ctrl-b`, then `d`.

### Experiment 2 result

The from-scratch full-suite run completed on 2026-06-06:

```text
PSNR 24.6486 / SSIM 0.7134
Clean BA 50.29%
Non-geometric average BA 50.29%
```

It failed to learn the message. The local `Dropout(0.1)` keeps only 10% of the
watermarked pixels and replaces the remaining 90% with cover pixels, so the
full suite was too strong from step 1.

## Experiment 2b: train10000 base20k then light non-geometric 5k

This resumes the successful identity-trained 20k base and adds one formal,
light 5k curriculum stage. Geometry and Dropout are excluded. The profile uses
3/7 identity probability (about 42.9%) plus mild GN, SP, JPEG, and blur.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
tmux new -s train10000_nongeomlight5k
bash scripts/run_exp2b_train10000_base20k_then_nongeomlight5k.sh
```

## Experiment 3: transfer existing refiners to the attack-aware base

This experiment performs no training. It applies the existing refine10000
Stage B and Stage C checkpoints to the same attack-aware base eval1000 outputs.
Both candidates use the same base checkpoint, images, message files, decoder,
and 11 non-geometric attacks.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp3_transfer_refiners_to_nongeomlight_base.sh
```

Final comparison:

```text
results/experiment_summaries/exp3_refiner_transfer_to_nongeomlight_base_seed61/comparison.md
```

Decision gate:

```text
PSNR >= 30 dB
Clean BA >= 95%
Non-geometric average BA >= 70%
```

## Experiment 4: retrain refine10000 Stage A/B on the attack-aware base

Experiment 3 showed that the old Stage C refiner nearly passed the formal gate.
Experiment 4 keeps the same COCO train10000 alias and the same refiner
architecture/loss schedule, but regenerates the training pairs with
`train10000-base20k-then-nongeomlight5k`.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp4_refine10000_on_nongeomlight_base.sh
```

Pipeline:

```text
generate refine10000 pairs with attack-aware base
-> Stage A 10k image-priority
-> Stage B 2k message recovery
-> fixed unseen eval1000
-> PSNR / Clean BA / 11 non-geometric attacks
```

Final summary:

```text
results/experiment_summaries/exp4_refine10000_nongeomlightbase_A10k_B2k_seed61/summary.md
```

### Experiment 4 result

```text
PSNR 32.1909 / SSIM 0.9401
Clean BA 93.12%
Non-geometric average BA 79.10%
Minimum BA 65.51% (Paper_GaussianNoise_arg03)
```

## Experiment 5: bounded Stage B+ message calibration

This experiment reuses the completed Experiment 4 Stage B checkpoint. It does
not retrain Stage A and does not regenerate refine10000 pairs.

The two parallel candidates differ only in `message_weight`:

```text
5A: message_weight=0.90, 500 steps, lr=2e-5
5B: message_weight=1.00, 500 steps, lr=2e-5
```

Preflight only:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_exp5_stagebplus_calibration.sh --check-only
```

Run:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_exp5_stagebplus_calibration.sh
```

The script trains and evaluates both candidates, including PSNR, Clean BA, and
the same 11 non-geometric attacks. It is resumable and refuses incomplete
existing output directories.

Final comparison:

```text
results/experiment_summaries/exp5_stagebplus_calibration_seed61/summary.md
```

## Experiment 6: formal-scale targeted Stage C

Experiment 6 starts from Experiment 5B and targets only the four weakest formal
non-geometric attacks. One 500-step run saves checkpoints at steps 250 and 500;
both are evaluated on the same unseen eval1000 split and full 11-attack suite.

Preflight:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_exp6_targeted_stagec.sh --check-only
```

Run:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_exp6_targeted_stagec.sh
```

Final comparison:

```text
results/experiment_summaries/exp6_targetedformal_stagec_seed61/summary.md
```
