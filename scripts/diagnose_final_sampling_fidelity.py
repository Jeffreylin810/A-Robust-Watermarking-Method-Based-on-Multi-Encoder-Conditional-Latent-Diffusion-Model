#!/usr/bin/env python3
"""Decompose O256 final-sampling image fidelity.

This script compares training/direct watermark outputs with the final diffusion
sampling outputs used by eval_diffusion.py. It is intended to locate why
training-time RGB PSNR can be near the configured target while saved eval images
remain much lower.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
import torchvision.utils as tvu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import datasets  # noqa: E402
from models import DenoisingDiffusion  # noqa: E402
from models.ddm import inverse_data_transform  # noqa: E402
from utils.metrics import calculate_psnr, calculate_ssim  # noqa: E402


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args():
    parser = argparse.ArgumentParser(description="Final sampling fidelity decomposition")
    parser.add_argument(
        "--config",
        default="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml",
        help="Config basename under configs/, or an explicit yaml path.",
    )
    parser.add_argument(
        "--resume",
        default="Result/CoCo_ddpm_O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final.pth.tar",
        help="Checkpoint to diagnose.",
    )
    parser.add_argument("--test_set", default="CoCo")
    parser.add_argument("--sampling_timesteps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument(
        "--output_dir",
        default="results/diagnostics/O256_MSG32_CL3_PSNR36_HINGE20K_seed61_final_sampling_fidelity64",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
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


def get_core_model(model):
    return model.module if hasattr(model, "module") else model


def tensor_to_uint8_rgb(image):
    """Convert a CHW RGB tensor in [0, 1] to HWC uint8."""
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def read_png_uint8(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True)


def metric_pair(pred_rgb01, gt_rgb01):
    pred = tensor_to_uint8_rgb(pred_rgb01)
    gt = tensor_to_uint8_rgb(gt_rgb01)
    return {
        "psnr": float(calculate_psnr(pred, gt, test_y_channel=False)),
        "ssim": float(calculate_ssim(pred, gt, test_y_channel=False)),
    }


def bit_accuracy(decoded, message):
    decoded_np = decoded.detach().cpu().numpy().round().clip(0, 1)
    message_np = message.detach().cpu().numpy()
    return float((1.0 - np.mean(np.abs(decoded_np - message_np))) * 100.0)


def latent_or_image_to_rgb01(diffusion, tensor):
    if diffusion.latent_enabled:
        tensor = diffusion.decode_latents(tensor)
    return inverse_data_transform(tensor)


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
    png_dir = output_dir / "x0_pred_png_roundtrip"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config: {resolve_config_path(args.config)}")
    print(f"Resume: {args.resume}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")

    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    diffusion.model.eval()
    core_model = get_core_model(diffusion.model)
    decoder = core_model.decoder

    metric_names = [
        "input_vs_gt",
        "vae_roundtrip_gt",
        "direct_target_t0",
        "direct_target_t999",
        "xs_final",
        "x0_pred_final_formal_eval",
        "x_noise_final",
        "x0_pred_png_saved",
    ]
    metrics = {
        name: {"psnr": [], "ssim": [], "ba": []}
        for name in metric_names
    }
    rows = []
    processed = 0

    with torch.no_grad():
        for x, image_ids in val_loader:
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            x = x.to(device)
            n = x.size(0)
            input_rgb = x[:, : config.data.channels, :, :]
            gt_rgb = x[:, config.data.channels : config.data.channels * 2, :, :]

            _, x_cond, x_target = diffusion.prepare_batch(x)
            x_sample = diffusion.random_sample(n)
            message = diffusion.random_message(x_cond.shape[0])

            t0 = torch.zeros(n, device=device)
            t999 = torch.full((n,), diffusion.config.diffusion.num_diffusion_timesteps - 1, device=device)
            direct_t0 = core_model.encode_watermark(torch.cat([x_cond, x_target], dim=1), t0.float(), message)
            direct_t999 = core_model.encode_watermark(torch.cat([x_cond, x_target], dim=1), t999.float(), message)

            sampled = diffusion.sample_image(x_cond, x_sample, message)
            xs_final = sampled[0][-1].to(device)
            x0_pred_final = sampled[1][-1].to(device)
            x_noise_final = sampled[3][-1].to(device)

            candidate_rgb = {
                "input_vs_gt": input_rgb,
                "vae_roundtrip_gt": latent_or_image_to_rgb01(diffusion, x_target),
                "direct_target_t0": latent_or_image_to_rgb01(diffusion, direct_t0),
                "direct_target_t999": latent_or_image_to_rgb01(diffusion, direct_t999),
                "xs_final": latent_or_image_to_rgb01(diffusion, xs_final),
                "x0_pred_final_formal_eval": latent_or_image_to_rgb01(diffusion, x0_pred_final),
                "x_noise_final": latent_or_image_to_rgb01(diffusion, x_noise_final),
            }
            candidate_latents = {
                "direct_target_t0": direct_t0,
                "direct_target_t999": direct_t999,
                "xs_final": xs_final,
                "x0_pred_final_formal_eval": x0_pred_final,
                "x_noise_final": x_noise_final,
            }

            batch_ids = list(image_ids) if isinstance(image_ids, (list, tuple)) else [str(image_ids)]
            for i in range(n):
                image_id = str(batch_ids[i]) if i < len(batch_ids) else f"sample_{processed + i:04d}"
                row = {"index": processed + i, "image_id": image_id}

                for name, rgb_batch in candidate_rgb.items():
                    values = metric_pair(rgb_batch[i], gt_rgb[i])
                    row[f"{name}_psnr"] = values["psnr"]
                    row[f"{name}_ssim"] = values["ssim"]
                    metrics[name]["psnr"].append(values["psnr"])
                    metrics[name]["ssim"].append(values["ssim"])

                png_path = png_dir / f"{image_id}.png"
                tvu.save_image(candidate_rgb["x0_pred_final_formal_eval"][i], png_path)
                png_values = {
                    "psnr": float(
                        calculate_psnr(read_png_uint8(png_path), tensor_to_uint8_rgb(gt_rgb[i]), test_y_channel=False)
                    ),
                    "ssim": float(
                        calculate_ssim(read_png_uint8(png_path), tensor_to_uint8_rgb(gt_rgb[i]), test_y_channel=False)
                    ),
                }
                row["x0_pred_png_saved_psnr"] = png_values["psnr"]
                row["x0_pred_png_saved_ssim"] = png_values["ssim"]
                metrics["x0_pred_png_saved"]["psnr"].append(png_values["psnr"])
                metrics["x0_pred_png_saved"]["ssim"].append(png_values["ssim"])

                for name, latent_batch in candidate_latents.items():
                    ba = bit_accuracy(decoder(latent_batch[i : i + 1]), message[i : i + 1])
                    row[f"{name}_ba"] = ba
                    metrics[name]["ba"].append(ba)

                if diffusion.latent_enabled:
                    reloaded = torch.from_numpy(read_png_uint8(png_path)).permute(2, 0, 1).float() / 255.0
                    reloaded = reloaded.unsqueeze(0).to(device)
                    reloaded_latent = diffusion.encode_images(reloaded * 2.0 - 1.0)
                    png_ba = bit_accuracy(decoder(reloaded_latent), message[i : i + 1])
                    row["x0_pred_png_saved_ba"] = png_ba
                    metrics["x0_pred_png_saved"]["ba"].append(png_ba)

                rows.append(row)

            processed += n
            latest = rows[-1]
            print(
                "processed {:04d}: direct_t0_psnr={:.2f}, x0_pred_psnr={:.2f}, "
                "png_psnr={:.2f}, x0_pred_ba={:.2f}".format(
                    processed,
                    latest["direct_target_t0_psnr"],
                    latest["x0_pred_final_formal_eval_psnr"],
                    latest["x0_pred_png_saved_psnr"],
                    latest.get("x0_pred_final_formal_eval_ba", float("nan")),
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
        "averages": {
            name: {
                "psnr": average(values["psnr"]),
                "ssim": average(values["ssim"]),
                "ba": average(values["ba"]),
            }
            for name, values in metrics.items()
        },
    }

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": rows}, f, indent=2)

    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("Final sampling fidelity decomposition\n")
        f.write("=" * 45 + "\n")
        f.write(f"config: {summary['config']}\n")
        f.write(f"resume: {summary['resume']}\n")
        f.write(f"processed: {processed}\n\n")
        f.write("Average metrics\n")
        f.write(f"{'path':<32} | {'PSNR':>9} | {'SSIM':>9} | {'BA':>9}\n")
        f.write("-" * 62 + "\n")
        for name in metric_names:
            values = summary["averages"][name]
            f.write(
                f"{name:<32} | {values['psnr']:>9.4f} | {values['ssim']:>9.4f} | {values['ba']:>9.4f}\n"
            )
        f.write("\nInterpretation guide:\n")
        f.write("- direct_target_t0/t999 high but x0_pred low: final sampling loop is the fidelity gap.\n")
        f.write("- x0_pred and png_saved similar: PNG saving is not the PSNR bottleneck.\n")
        f.write("- vae_roundtrip_gt low: VAE reconstruction itself limits PSNR.\n")
        f.write("- x0_pred BA high with low PSNR: message survives but image fidelity is not controlled.\n")

    print("\nAverage metrics")
    for name in metric_names:
        values = summary["averages"][name]
        print(f"{name}: PSNR={values['psnr']:.4f}, SSIM={values['ssim']:.4f}, BA={values['ba']:.4f}")
    print(f"Report saved to: {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
