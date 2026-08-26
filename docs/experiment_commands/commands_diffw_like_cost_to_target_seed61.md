# DiffW-like Cost-to-Target Analysis Commands

Purpose: supplementary check for how much DiffW-like training cost is needed to approach the current O256-RGB formal target on locked COCO finaltest1000.

This is **not** a replacement for the formal main experiment.

## Output Root

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/diffw_like_cost_to_o256rgb_target_20260625
```

## Target Thresholds

Target is the current O256-RGB formal locked finaltest1000 result:

| Metric | Target |
|---|---:|
| PSNR | >= 32.9034 |
| SSIM | >= 0.9589 |
| LPIPS | <= 0.0461 |
| Clean BA | >= 95.81% |
| Nongeom Avg BA | >= 87.38% |
| Minimum BA | >= 72.93% |

## Step 0: Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh --check-only
```

## Step 1: Run 40k Quick Gate

This trains a new DiffW-like 40k checkpoint in the independent output root, then evaluates only:

- PSNR
- SSIM
- LPIPS
- Clean BA

It does **not** run the 11 non-geometric attacks yet.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh 40000 quick
```

Estimated training time: about `04:51:00`, plus finaltest1000 inference / LPIPS / Clean BA.

## Step 2: Read Quick Gate Summary

```bash
less /home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/diffw_like_cost_to_o256rgb_target_20260625/cost_to_target_summary.md
```

CSV:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/diffw_like_cost_to_o256rgb_target_20260625/quick_gate_results.csv
```

## Decision After 40k

Recommended rule:

| 40k quick result | Decision |
|---|---|
| PSNR < 30 and LPIPS > 0.08 | Stop. Do not continue unless advisor specifically requests. |
| PSNR 30-32 or LPIPS 0.05-0.08 | Consider 60k if time allows. |
| PSNR >= 32 and Clean BA >= 95% | Run full 11-attack gate for 40k. |
| PSNR >= 32.9034, SSIM >= 0.9589, LPIPS <= 0.0461 | Run full 11-attack gate immediately. |

## Optional: Run 40k Full Gate

Only run this if the 40k quick gate looks close enough.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh 40000 full
```

This runs:

- 11 formal non-geometric attacks
- Nongeom Avg BA
- Minimum BA

CSV:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/diffw_like_cost_to_o256rgb_target_20260625/full_gate_results.csv
```

## Optional: Continue to 60k / 80k / 100k

Only continue if the previous checkpoint is close enough.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh 60000 quick
```

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh 80000 quick
```

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_diffw_like_cost_to_target_seed61.sh 100000 quick
```

Full gate examples:

```bash
bash scripts/run_diffw_like_cost_to_target_seed61.sh 60000 full
bash scripts/run_diffw_like_cost_to_target_seed61.sh 80000 full
bash scripts/run_diffw_like_cost_to_target_seed61.sh 100000 full
```

## Safety Notes

- The script writes new checkpoints under the independent output root.
- It does not overwrite the formal DiffW-like 20k checkpoint.
- It does not overwrite the formal main-result summaries.
- It refuses to continue if an incomplete output folder already exists for the same steps value.
- `quick` should always be run before `full`.

## Thesis Wording Guardrail

Use:

```text
supplementary cost-to-target analysis
```

Do not use:

```text
formal main result
```

Do not claim that this proves anything about the official DiffW checkpoint. This experiment only evaluates the local DiffW-like baseline under the thesis unified protocol.

