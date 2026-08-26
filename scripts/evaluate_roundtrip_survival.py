#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import datasets  # noqa: E402
from models import DenoisingDiffusion  # noqa: E402
from models.ddm import data_transform, inverse_data_transform  # noqa: E402

try:
    from diffusers import AutoencoderKL  # type: ignore
except Exception:  # pragma: no cover
    AutoencoderKL = None


def dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate watermark survival before/after VAE round-trip.")
    parser.add_argument("--config", required=True, help="Config basename under configs/ or explicit YAML path.")
    parser.add_argument("--resume", required=True, help="Checkpoint path.")
    parser.add_argument("--test_set", default="CoCo")
    parser.add_argument("--sampling_timesteps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--roundtrip_vae_path",
        default="/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base",
        help="External VAE used only when latent.enabled=false.",
    )
    parser.add_argument("--roundtrip_deterministic_encode", action="store_true", default=True)
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    candidate = Path(config_arg)
    if candidate.is_file():
        return candidate
    candidate = PROJECT_ROOT / "configs" / config_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def load_config(config_arg: str) -> argparse.Namespace:
    with open(resolve_config_path(config_arg), "r", encoding="utf-8") as handle:
        return dict2namespace(yaml.safe_load(handle))


def bit_accuracy(decoded: torch.Tensor, message: torch.Tensor) -> float:
    decoded_np = decoded.detach().cpu().numpy().round().clip(0, 1)
    message_np = message.detach().cpu().numpy()
    return float((1.0 - np.mean(np.abs(decoded_np - message_np))) * 100.0)


def average(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


class ExternalVAERoundTrip:
    def __init__(self, model_path: str, device: torch.device):
        if AutoencoderKL is None:
            raise ImportError("diffusers AutoencoderKL is required for pixel-space round-trip evaluation.")
        if os.path.isdir(os.path.join(model_path, "vae")):
            self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        else:
            self.vae = AutoencoderKL.from_pretrained(model_path)
        self.vae.to(device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        self.device = device

    def roundtrip(self, rgb_0_1: torch.Tensor) -> torch.Tensor:
        rgb_minus1_1 = data_transform(rgb_0_1)
        with torch.no_grad():
            posterior = self.vae.encode(rgb_minus1_1).latent_dist
            latents = posterior.mode() * self.scaling_factor
            decoded_minus1_1 = self.vae.decode(latents / self.scaling_factor).sample
        return decoded_minus1_1


def main() -> None:
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
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.__dict__[config.data.dataset](config)
    _, val_loader = dataset.get_loaders(parse_patches=False, validation=args.test_set)

    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    diffusion.model.eval()
    decoder = diffusion.model.module.decoder

    external_vae = None
    if not diffusion.latent_enabled:
        external_vae = ExternalVAERoundTrip(args.roundtrip_vae_path, device)

    metrics = {
        "direct_decoder_ba": [],
        "roundtrip_decoder_ba": [],
        "roundtrip_drop_pp": [],
    }
    samples: list[dict] = []
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
            pred_space = sampled[0][-1].to(device)

            direct_ba = bit_accuracy(decoder(pred_space), message)

            if diffusion.latent_enabled:
                rgb_minus1_1 = diffusion.decode_latents(pred_space)
                rgb_0_1 = inverse_data_transform(rgb_minus1_1)
                roundtrip_space = diffusion.encode_images(data_transform(rgb_0_1))
            else:
                rgb_0_1 = inverse_data_transform(pred_space)
                assert external_vae is not None
                roundtrip_space = external_vae.roundtrip(rgb_0_1)

            roundtrip_ba = bit_accuracy(decoder(roundtrip_space), message)
            drop_pp = direct_ba - roundtrip_ba

            batch_ids = list(image_ids) if isinstance(image_ids, (list, tuple)) else [str(image_ids)]
            sample_row = {
                "index": processed,
                "image_ids": batch_ids,
                "direct_decoder_ba": direct_ba,
                "roundtrip_decoder_ba": roundtrip_ba,
                "roundtrip_drop_pp": drop_pp,
                "model_space": "latent" if diffusion.latent_enabled else "pixel",
            }
            samples.append(sample_row)
            metrics["direct_decoder_ba"].append(direct_ba)
            metrics["roundtrip_decoder_ba"].append(roundtrip_ba)
            metrics["roundtrip_drop_pp"].append(drop_pp)
            processed += n

            print(
                "sample {:04d}: direct={:.2f}, roundtrip={:.2f}, drop={:.2f}".format(
                    processed, direct_ba, roundtrip_ba, drop_pp
                )
            )
            if processed >= args.num_images:
                break

    summary = {
        "config": str(resolve_config_path(args.config)),
        "resume": args.resume,
        "device": str(device),
        "num_images_processed": processed,
        "model_space": "latent" if diffusion.latent_enabled else "pixel",
        "averages": {key: average(values) for key, values in metrics.items()},
        "samples": samples,
    }

    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "Watermark round-trip survival report",
        "====================================",
        f"config: {summary['config']}",
        f"resume: {summary['resume']}",
        f"model_space: {summary['model_space']}",
        f"processed: {processed}",
        "",
        "Average BA (%)",
        f"- direct_decoder_ba: {summary['averages']['direct_decoder_ba']:.4f}",
        f"- roundtrip_decoder_ba: {summary['averages']['roundtrip_decoder_ba']:.4f}",
        f"- roundtrip_drop_pp: {summary['averages']['roundtrip_drop_pp']:.4f}",
        "",
        "Interpretation guide:",
        "- latent model: direct BA is the pre-VAE-roundtrip latent decoder accuracy.",
        "- pixel model: direct BA is the pixel decoder accuracy before external VAE encode/decode.",
        "- roundtrip_drop_pp > 0 means the VAE round-trip erodes message recoverability.",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved report: {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
