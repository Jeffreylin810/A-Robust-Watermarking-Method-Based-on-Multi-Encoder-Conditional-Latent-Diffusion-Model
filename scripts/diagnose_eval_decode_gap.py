#!/usr/bin/env python3
"""Diagnose whether O256-PAPER loses watermark signal during image save/reload.

The training log reports bitwise errors from an in-memory decoder path. The
formal BA script decodes from saved PNGs. This script measures both paths on
the same sampled images so we can locate the gap.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
import torchvision.utils as tvu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import datasets  # noqa: E402
from models import DenoisingDiffusion  # noqa: E402
from models.ddm import data_transform, inverse_data_transform  # noqa: E402


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args():
    parser = argparse.ArgumentParser(description="O256-PAPER eval decode gap diagnostic")
    parser.add_argument(
        "--config",
        default="CoCo_eval1000_ours256_msg64_paper_protocol.yml",
        help="Config basename under configs/, or an explicit yaml path.",
    )
    parser.add_argument(
        "--resume",
        default="Result/CoCo_ddpm_O256_PAPER_seed61_500k_final.pth.tar",
        help="Checkpoint used for evaluation.",
    )
    parser.add_argument("--test_set", default="CoCo", help="Validation split name used by eval_diffusion.py.")
    parser.add_argument("--sampling_timesteps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument(
        "--output_dir",
        default="results/diagnostics/O256_PAPER_seed61_eval_decode_gap",
        help="Directory for report.json, report.txt, and temporary PNG round-trip images.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Use cuda when available by default.",
    )
    return parser.parse_args()


def resolve_config_path(config_arg):
    candidate = Path(config_arg)
    if candidate.is_file():
        return candidate
    candidate = PROJECT_ROOT / "configs" / config_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def load_config(config_arg):
    with open(resolve_config_path(config_arg), "r", encoding="utf-8") as f:
        return dict2namespace(yaml.safe_load(f))


def bit_accuracy(decoded, message):
    decoded_np = decoded.detach().cpu().numpy().round().clip(0, 1)
    message_np = message.detach().cpu().numpy()
    return float((1.0 - np.mean(np.abs(decoded_np - message_np))) * 100.0)


def load_png_like_calculate_ba(path, device, image_size):
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(path).convert("RGB")
    return data_transform(tf(img).unsqueeze(0).to(device))


def average(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    output_dir = PROJECT_ROOT / args.output_dir
    png_dir = output_dir / "png_roundtrip"
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config: {resolve_config_path(args.config)}")
    print(f"Resume: {args.resume}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    diffusion.model.eval()
    decoder = diffusion.model.module.decoder

    metrics = {
        "sampling_decoder_ba": [],
        "latent_direct_decoder_ba": [],
        "rgb_tensor_roundtrip_decoder_ba": [],
        "png_saved_roundtrip_decoder_ba": [],
    }
    samples = []
    processed = 0

    with torch.no_grad():
        for x, image_ids in val_loader:
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.size(0)
            x_pair, x_cond, _ = diffusion.prepare_batch(x)
            del x_pair
            x_sample = diffusion.random_sample(n)
            message = diffusion.random_message(x_cond.shape[0])

            sampled = diffusion.sample_image(x_cond, x_sample, message)
            pred_latent_or_img = sampled[0][-1].to(device)
            sampling_message_pred = sampled[2][-1]

            sampling_ba = bit_accuracy(sampling_message_pred, message)
            latent_direct_ba = bit_accuracy(decoder(pred_latent_or_img), message)

            if diffusion.latent_enabled:
                rgb_minus1_1 = diffusion.decode_latents(pred_latent_or_img)
                rgb_0_1 = inverse_data_transform(rgb_minus1_1)
                encoded_from_rgb_tensor = diffusion.encode_images(data_transform(rgb_0_1))
                rgb_tensor_ba = bit_accuracy(decoder(encoded_from_rgb_tensor), message)
            else:
                rgb_0_1 = inverse_data_transform(pred_latent_or_img)
                rgb_tensor_ba = bit_accuracy(decoder(data_transform(rgb_0_1)), message)

            png_bas = []
            batch_ids = list(image_ids) if isinstance(image_ids, (list, tuple)) else [str(image_ids)]
            for i in range(n):
                image_id = str(batch_ids[i]) if i < len(batch_ids) else f"sample_{processed + i:04d}"
                png_path = png_dir / f"{processed + i:04d}_{image_id}.png"
                tvu.save_image(rgb_0_1[i], png_path)
                reloaded = load_png_like_calculate_ba(png_path, device, config.data.image_size)
                if diffusion.latent_enabled:
                    reloaded = diffusion.encode_images(reloaded)
                png_bas.append(bit_accuracy(decoder(reloaded), message[i : i + 1]))

            png_ba = average(png_bas)

            row = {
                "index": processed,
                "image_ids": batch_ids,
                "sampling_decoder_ba": sampling_ba,
                "latent_direct_decoder_ba": latent_direct_ba,
                "rgb_tensor_roundtrip_decoder_ba": rgb_tensor_ba,
                "png_saved_roundtrip_decoder_ba": png_ba,
            }
            samples.append(row)
            for key in metrics:
                metrics[key].append(row[key])

            processed += n
            print(
                "sample {:04d}: sampling={:.2f}, latent_direct={:.2f}, "
                "rgb_tensor={:.2f}, png_saved={:.2f}".format(
                    processed,
                    sampling_ba,
                    latent_direct_ba,
                    rgb_tensor_ba,
                    png_ba,
                )
            )
            if processed >= args.num_images:
                break

    summary = {
        "config": str(resolve_config_path(args.config)),
        "resume": args.resume,
        "num_images_requested": args.num_images,
        "num_images_processed": processed,
        "device": str(device),
        "averages": {key: average(value) for key, value in metrics.items()},
        "samples": samples,
    }

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("O256-PAPER eval decode gap diagnostic\n")
        f.write("=" * 45 + "\n")
        f.write(f"config: {summary['config']}\n")
        f.write(f"resume: {summary['resume']}\n")
        f.write(f"processed: {processed}\n\n")
        f.write("Average BA (%)\n")
        for key, value in summary["averages"].items():
            f.write(f"- {key}: {value:.4f}\n")
        f.write("\nInterpretation guide:\n")
        f.write("- sampling high, png low: signal is lost in decode/save/reload/eval path.\n")
        f.write("- sampling low too: the sampled final output itself does not carry the message.\n")
        f.write("- latent_direct high, rgb_tensor low: VAE decode/encode is the main gap.\n")
        f.write("- rgb_tensor high, png low: PNG quantization/saved-image path is the main gap.\n")

    print("\nAverage BA (%)")
    for key, value in summary["averages"].items():
        print(f"{key}: {value:.4f}")
    print(f"Report saved to: {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
