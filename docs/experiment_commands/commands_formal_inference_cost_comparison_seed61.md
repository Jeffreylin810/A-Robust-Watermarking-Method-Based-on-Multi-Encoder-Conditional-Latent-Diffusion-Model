# Formal Inference Cost Comparison Commands

Purpose: compare DiffW-like and O256-RGB inference cost under two definitions:

1. compute-only model inference
2. end-to-end image processing

This experiment is designed to output both line charts and full CSV tables.

## Output Root

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626
```

## Step 0: Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_formal_inference_cost_comparison_seed61.sh --check-only
```

Run benchmarks only when GPU is idle.

## Step 1: Compute-only Timing

This measures model-side inference cost and excludes dataset I/O, PNG writing, PSNR, BA, and attacks.

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_formal_inference_cost_comparison_seed61.sh compute
```

Outputs:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/compute_only_inference_timing.csv
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/figures/fig_compute_only_ms_per_image_vs_batch.png
```

## Step 2: End-to-end Timing

This measures:

```text
read images -> infer watermark -> write watermarked images/message files
```

For O256-RGB, total time is:

```text
O256-Base eval_diffusion + RGB refiner apply
```

Recommended staged execution:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end 100
```

If eval100 looks fine:

```bash
bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end 300
```

Then:

```bash
bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end 1000
```

Or run all three in one command:

```bash
bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end all
```

Outputs:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/end_to_end_inference_timing.csv
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/figures/fig_end_to_end_total_time_vs_num_images.png
```

## Step 3: 111k Projection

The projection is produced automatically whenever `summarize` runs and enough timing data exists.

```bash
bash scripts/run_formal_inference_cost_comparison_seed61.sh summarize
```

Outputs:

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/inference_cost_projection_111k.csv
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/figures/fig_projection_111k_total_time.png
```

Important: 111k is projected from measured per-image timing. It is not a directly measured 111k run.

## Final Summary

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_cost_comparison_seed61_20260626/inference_cost_summary.md
```

## Overwrite / Rerun

By default, the script refuses to overwrite an existing output for the requested run.

To rerun a stage:

```bash
FORCE=1 bash scripts/run_formal_inference_cost_comparison_seed61.sh compute
```

or:

```bash
FORCE=1 bash scripts/run_formal_inference_cost_comparison_seed61.sh end2end 100
```

## If GPU Is Busy

The script refuses to run if another GPU compute process is active. Only bypass this if you intentionally want to share the GPU:

```bash
ALLOW_GPU_BUSY=1 bash scripts/run_formal_inference_cost_comparison_seed61.sh compute
```

## Thesis Wording

Use:

```text
O256-RGB is faster than DiffW-like under the measured inference protocol.
```

Use:

```text
RGB refinement adds only a small overhead relative to O256-Base.
```

Do not write:

```text
RGB refinement accelerates inference.
```

Do not write:

```text
111k was fully measured.
```

Unless you actually run all 111k images, write:

```text
111k deployment time is projected from measured per-image timing.
```

