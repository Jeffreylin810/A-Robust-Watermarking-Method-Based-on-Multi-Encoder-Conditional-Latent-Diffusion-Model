# Actual 10k End-to-End Deployment Benchmark

- `num_images`: `10000`
- Classification: supplementary deployment throughput benchmark
- Scope: model-prepared deployment only; does not include training cost

| Method | Total time | ms/image | images/sec | vs DiffW-like |
|---|---:|---:|---:|---:|
| DiffW-like | 01:04:04 | 384.363000 | 2.601707 | 1.000000x |
| O256-Base | 00:15:39 | 93.858000 | 10.654393 | 4.095154x |
| O256-RGB | 00:25:07 | 150.660000 | 6.637462 | 2.551195x |

## Notes
- `DiffW-like`: actual 10k end-to-end eval_diffusion deployment run
  - output images: `10000`
  - results dir: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_actual10k_end_to_end_deployment_benchmark_20260628/end_to_end_outputs/diffw_like/deploy10k/deploy10k/CoCo/CoCo`
- `O256-Base`: actual 10k end-to-end O256 base eval_diffusion deployment run
  - output images: `10000`
  - results dir: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_actual10k_end_to_end_deployment_benchmark_20260628/end_to_end_outputs/o256_base/deploy10k/deploy10k/CoCo/CoCo`
- `O256-RGB`: actual 10k end-to-end O256 base eval_diffusion plus RGB refiner apply
  - output images: `10000`
  - results dir: `/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/13_supplementary_actual10k_end_to_end_deployment_benchmark_20260628/end_to_end_outputs/o256_rgb/deploy10k/apply/refined_images`
