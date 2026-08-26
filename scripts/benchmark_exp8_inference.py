#!/usr/bin/env python3
"""Compute-only inference benchmark for Experiment 8.

Run this script once for a DiffW-like project and once for O256. It imports the
model implementation from ``--project_root`` so both methods use their native
sampling path. Dataset I/O, PNG writing, PSNR, and BA evaluation are excluded
from the timed region.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--sampling_timesteps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--refiner_iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--refiner", default="")
    parser.add_argument("--refiner_hidden_channels", type=int, default=48)
    parser.add_argument("--refiner_depth", type=int, default=5)
    parser.add_argument("--refiner_max_delta", type=float, default=0.50)
    return parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def timed_repeats(fn, batch_size: int, warmup: int, iterations: int, repeats: int):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        run_ms_per_image = []
        peak_allocated_mib = []
        peak_incremental_mib = []
        for _ in range(repeats):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            for _iteration in range(iterations):
                fn()
            torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            peak_memory = torch.cuda.max_memory_allocated()
            run_ms_per_image.append(elapsed_ms / (iterations * batch_size))
            peak_allocated_mib.append(peak_memory / (1024**2))
            peak_incremental_mib.append(max(0, peak_memory - baseline_memory) / (1024**2))

    mean_ms = statistics.fmean(run_ms_per_image)
    std_ms = statistics.stdev(run_ms_per_image) if len(run_ms_per_image) > 1 else 0.0
    return {
        "run_ms_per_image": run_ms_per_image,
        "mean_ms_per_image": mean_ms,
        "std_ms_per_image": std_ms,
        "cv_percent": 100.0 * std_ms / mean_ms if mean_ms else 0.0,
        "images_per_second": 1000.0 / mean_ms if mean_ms else 0.0,
        "peak_allocated_mib": max(peak_allocated_mib),
        "peak_incremental_mib": max(peak_incremental_mib),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 8 requires a CUDA GPU.")

    project_root = Path(args.project_root).resolve()
    config_path = resolve_path(project_root, args.config)
    resume_path = resolve_path(project_root, args.resume)
    output_path = Path(args.output).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    import datasets  # noqa: E402
    from models import DenoisingDiffusion  # noqa: E402
    from models.ddm import data_transform, inverse_data_transform  # noqa: E402

    with config_path.open("r", encoding="utf-8") as handle:
        config = dict2namespace(yaml.safe_load(handle))
    device = torch.device("cuda")
    config.device = device
    config.data.num_workers = 0
    config.sampling.batch_size = max(args.batch_sizes)

    model_args = argparse.Namespace(
        resume=str(resume_path),
        sampling_timesteps=args.sampling_timesteps,
        image_folder="",
        grid_r=8,
        test_set="CoCo",
        seed=args.seed,
    )

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    diffusion = DenoisingDiffusion(model_args, config)
    diffusion.load_ddm_ckpt(str(resume_path), ema=True)
    diffusion.model.to(device)
    diffusion.model.eval()
    if getattr(diffusion, "vae", None) is not None:
        diffusion.vae.to(device)
        diffusion.vae.eval()

    dataset = datasets.__dict__[config.data.dataset](config)
    _train_loader, val_loader = dataset.get_loaders(parse_patches=True, validation="CoCo")
    input_batch, _names = next(iter(val_loader))
    if input_batch.ndim == 5:
        input_batch = input_batch.flatten(start_dim=0, end_dim=1)
    if input_batch.shape[0] < max(args.batch_sizes):
        repeats_needed = int(np.ceil(max(args.batch_sizes) / input_batch.shape[0]))
        input_batch = input_batch.repeat(repeats_needed, 1, 1, 1)
    input_batch = input_batch[: max(args.batch_sizes)].to(device, non_blocking=True)

    latent_enabled = bool(getattr(diffusion, "latent_enabled", False))
    refiner = None
    if args.refiner:
        if not latent_enabled:
            raise ValueError("--refiner is only valid for the O256 latent project.")
        refiner_path = resolve_path(project_root, args.refiner)
        if not refiner_path.is_file():
            raise FileNotFoundError(refiner_path)
        scripts_dir = project_root / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from train_rgb_refinement import ResidualRefiner  # noqa: E402

        refiner = ResidualRefiner(
            in_channels=6,
            hidden_channels=args.refiner_hidden_channels,
            depth=args.refiner_depth,
            max_delta=args.refiner_max_delta,
        ).to(device)
        checkpoint = torch.load(refiner_path, map_location=device)
        refiner.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        refiner.eval()
        # Importing the project-local refiner helper also imports the O256
        # model module. Reassert device placement before timing so no
        # environment-specific module import can leave VAE weights on CPU.
        diffusion.model.to(device)
        if getattr(diffusion, "vae", None) is not None:
            diffusion.vae.to(device)
            diffusion.vae.eval()

    results = []
    for batch_size in args.batch_sizes:
        raw = input_batch[:batch_size]
        cover_0_1 = raw[:, : config.data.channels]

        if latent_enabled:
            def run_base():
                _pair, condition, _target = diffusion.prepare_batch(raw)
                noise = diffusion.random_sample(batch_size)
                message = diffusion.random_message(batch_size)
                sampled = diffusion.sample_image(condition, noise, message)
                watermarked = diffusion.decode_latents(sampled[1][-1].to(device))
                return inverse_data_transform(watermarked)
        else:
            condition = data_transform(cover_0_1)

            def run_base():
                noise = torch.randn_like(condition)
                message = torch.randint(
                    0,
                    2,
                    (batch_size, config.watermarking.message_length),
                    device=device,
                    dtype=torch.float32,
                )
                sampled = diffusion.sample_image(condition, noise, message)
                return inverse_data_transform(sampled[1][-1].to(device))

        base_metrics = timed_repeats(
            run_base, batch_size, args.warmup, args.iterations, args.repeats
        )
        results.append({"batch_size": batch_size, "mode": "base", **base_metrics})

        if refiner is not None:
            with torch.inference_mode():
                fixed_watermarked = run_base()

            def run_refiner():
                refined, _delta = refiner(fixed_watermarked, cover_0_1)
                return refined

            refiner_metrics = timed_repeats(
                run_refiner,
                batch_size,
                args.warmup,
                args.refiner_iterations,
                args.repeats,
            )
            results.append(
                {"batch_size": batch_size, "mode": "refiner_only", **refiner_metrics}
            )

            def run_full():
                watermarked = run_base()
                refined, _delta = refiner(watermarked, cover_0_1)
                return refined

            full_metrics = timed_repeats(
                run_full, batch_size, args.warmup, args.iterations, args.repeats
            )
            results.append({"batch_size": batch_size, "mode": "base_plus_refiner", **full_metrics})

    device_index = torch.cuda.current_device()
    payload = {
        "method_name": args.method_name,
        "project_root": str(project_root),
        "config": str(config_path),
        "resume": str(resume_path),
        "refiner": str(resolve_path(project_root, args.refiner)) if args.refiner else "",
        "sampling_timesteps": args.sampling_timesteps,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "refiner_iterations": args.refiner_iterations,
        "repeats": args.repeats,
        "seed": args.seed,
        "timing_scope": (
            "Synchronized compute-only wall-clock; includes the native sampling "
            "implementation's CPU/GPU tensor transfers, but excludes dataset I/O, "
            "PNG writing, PSNR, SSIM, and BA"
        ),
        "gpu": torch.cuda.get_device_name(device_index),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved benchmark: {output_path}")
    for row in results:
        print(
            f"{args.method_name} batch={row['batch_size']} mode={row['mode']}: "
            f"{row['mean_ms_per_image']:.4f} +/- {row['std_ms_per_image']:.4f} ms/image, "
            f"{row['images_per_second']:.2f} image/s, CV={row['cv_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
