# Experiment 8: formal speed and cost evaluation

## Protocol

- GPU: NVIDIA GeForce RTX 4090
- Sampling steps: 3
- Warm-up: 20 iterations
- Repeats: 5
- Synchronized compute-only timing excludes dataset I/O, PNG writing, PSNR/SSIM, and BA.
- Native sampling CPU/GPU intermediate-tensor transfers remain included.
- Training/preparation cost uses wall-clock `real` time from the original logs.

## Compute-only inference

| Batch | Method | ms/image | std | image/s | CV | Peak incremental MiB |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DiffW-like | 365.7766 | 0.1365 | 2.73 | 0.04% | 649.8 |
| 1 | O256 base | 70.7176 | 0.5182 | 14.14 | 0.73% | 192.1 |
| 1 | Module only | 0.2942 | 0.0047 | 3399.33 | 1.58% | 26.3 |
| 1 | O256 base + module | 70.6972 | 0.2469 | 14.14 | 0.35% | 192.1 |
| 4 | DiffW-like | 230.3526 | 0.0294 | 4.34 | 0.01% | 2595.6 |
| 4 | O256 base | 38.0417 | 0.0200 | 26.29 | 0.05% | 965.8 |
| 4 | Module only | 0.2905 | 0.0025 | 3442.93 | 0.86% | 102.3 |
| 4 | O256 base + module | 38.3488 | 0.0092 | 26.08 | 0.02% | 965.8 |

## One-time training and preparation cost

| Stage | Seconds | Wall-clock |
|---|---:|---:|
| DiffW-like base training | 8822.30 | 02:27:02.30 |
| O256 identity-only 20k | 4046.86 | 01:07:26.86 |
| O256 nongeom-light continuation 5k | 1082.81 | 00:18:02.81 |
| Generate refine10000 pairs | 962.94 | 00:16:02.94 |
| Module Stage A | 1834.86 | 00:30:34.86 |
| Module Stage B | 944.67 | 00:15:44.67 |
| Module Stage B+ | 245.24 | 00:04:05.24 |
| Module targeted Stage C | 277.68 | 00:04:37.68 |
| **O256 base total** | **5129.67** | **01:25:29.67** |
| **Module preparation total** | **4265.39** | **01:11:05.39** |
| **O256 complete preparation** | **9395.06** | **02:36:35.06** |

## Interpretation

- O256 base training speed ratio vs DiffW-like: **1.720x faster**.
- Complete preparation ratio DiffW-like/O256+module: **0.939x**.
- Module adds **4265.39 s (01:11:05.39)** of one-time preparation.
- Batch-1 module-only overhead: **0.2942 ms/image**.
- Batch-1 measured end-to-end increase: **-0.0204 ms/image (-0.03%)**.

Do not claim a speed advantage unless the comparison scope is stated explicitly:
`base training`, `complete preparation`, or `inference` answer different questions.
