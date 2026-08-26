# Experiment 7b corrected: seed62 cross-seed module reproducibility

The first Experiment 7 run is retained only as a diagnostic because its first
20k steps accidentally used `identity_nongeom_formal` instead of the seed61
reference recipe `identity_only`. It must not be reported as cross-seed
reproducibility evidence.

## Purpose

This experiment tests whether the RGB refinement module trained with the
`seed61` base is tied to that exact base checkpoint.

Only the base-model random seed changes:

```text
seed61 -> seed62
```

The dataset/split, model architecture, msg32, base training recipe
(`identity-only 20k + nongeom-light 5k`), VAE, losses, module checkpoint, and
unseen eval1000
are unchanged.

The first experiment does **not** retrain the module. It transfers:

```text
results/refinement/refine10000_targetedformal_C500_seed61/refiner_final.pth
```

to the newly trained seed62 base.

## Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp7b_corrected_seed62_cross_seed_transfer.sh --check-only
```

## Full run

Run this inside `tmux`:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp7b_corrected_seed62_cross_seed_transfer.sh
```

The script is restart-aware. Completed checkpoints and complete eval1000
outputs are reused; incomplete outputs are not overwritten automatically.

## Estimated cost

- Seed62 base training: about 1 hour 25 minutes based on seed61.
- Eval, module application, and attacks: additional minutes.
- Module retraining: not included and not performed unless transfer fails.

## Pre-registered transfer criteria

Compared with `seed61 base + seed61 C500 module`:

| Metric | Allowed cross-seed difference |
|---|---:|
| PSNR | absolute difference <= 1.0 dB |
| Clean BA | absolute difference <= 3.0 percentage points |
| Nongeom average BA | absolute difference <= 5.0 percentage points |
| Minimum BA | may not decrease by more than 5.0 percentage points |

These criteria measure reproducibility, not the final thesis quality gate.

## Output

```text
results/experiment_summaries/exp7b_corrected_seed62_cross_seed_transfer/summary.md
```

If the transfer passes, the same module can remain the formal candidate.

The script first checks late-training bit error after identity-only 20k. It
also requires the final seed62 base Clean BA to reach at least 90% before any
module transfer evaluation. If either gate fails, the script stops instead of
spending time on attacks or module retraining.

Only after a healthy seed62 base passes these gates may a transfer failure be
interpreted as possible module/base binding. A seed62-specific module must not
be trained before that condition is met.
