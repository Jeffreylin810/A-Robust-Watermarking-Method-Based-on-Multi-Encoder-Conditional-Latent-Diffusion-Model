#!/usr/bin/env python3
"""Apply a trained RGB refinement head to another split.

This is used for formal train/eval separation: train the refiner on train
watermarked outputs, then apply the frozen refiner to unseen eval watermarked
outputs and reuse the standard PSNR/SSIM + BA evaluators.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train_rgb_refinement import (
    PROJECT_ROOT,
    RefinementPairDataset,
    ResidualRefiner,
    build_diffusion,
    load_config,
    save_refined_images,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True, help="Frozen O256 watermark checkpoint.")
    parser.add_argument("--refiner", required=True, help="Trained RGB refiner checkpoint.")
    parser.add_argument("--watermarked_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=48)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max_delta", type=float, default=0.5)
    parser.add_argument("--conditioning", choices=("cover", "none"), default="cover")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    config = load_config(args.config)
    diffusion = build_diffusion(config, args.resume, device)
    dataset = RefinementPairDataset(args.watermarked_dir, args.gt_dir, args.image_size, args.max_images)

    in_channels = 6 if args.conditioning == "cover" else 3
    refiner = ResidualRefiner(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        max_delta=args.max_delta,
    ).to(device)
    checkpoint = torch.load(args.refiner, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    refiner.load_state_dict(state_dict, strict=True)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metrics = save_refined_images(
        refiner,
        dataset,
        diffusion,
        output_dir,
        device,
        args.conditioning,
        args.batch_size,
        args.num_workers,
    )
    metrics["elapsed_seconds"] = time.time() - start
    payload = {"args": vars(args), "count": len(dataset), "metrics": metrics}
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("RGB refinement apply report\n")
        f.write("===========================\n")
        f.write(f"refiner: {args.refiner}\n")
        f.write(f"watermarked_dir: {args.watermarked_dir}\n")
        f.write(f"gt_dir: {args.gt_dir}\n")
        f.write(f"count: {len(dataset)}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"applied refiner: {args.refiner}")
    print(f"count: {len(dataset)}")
    print(f"report: {output_dir / 'report.txt'}")
    print(f"refined images: {metrics['refined_dir']}")
    print(f"clean attack dir: {metrics['clean_attack_dir']}")


if __name__ == "__main__":
    main()
