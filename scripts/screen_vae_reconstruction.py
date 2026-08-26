#!/usr/bin/env python3
"""Screen VAE reconstruction ceilings on a fixed GT image set.

This script evaluates only VAE round-trip fidelity:

    x -> VAE.encode(x) -> VAE.decode(z) -> x_hat

It is intended as a cheap pre-check before running watermark training. If a VAE
cannot reconstruct the chosen evaluation set near the desired PSNR target, the
downstream latent watermark model is unlikely to reach that target against the
original GT images.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.ddm import data_transform, inverse_data_transform  # noqa: E402
from utils.metrics import calculate_psnr, calculate_ssim  # noqa: E402

try:
    from diffusers import AutoencoderKL, ConsistencyDecoderVAE
except ImportError as exc:  # pragma: no cover - runtime dependency check
    AutoencoderKL = None
    ConsistencyDecoderVAE = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_GT_DIR = (
    "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
    "train10000_eval1000_unseen/data/CoCo/val/gt"
)
DEFAULT_SD21_VAE = "/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base"


def parse_vae_spec(spec):
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = Path(path).name
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Invalid VAE spec: {spec}")
    return name, path


def parse_args():
    parser = argparse.ArgumentParser(description="Screen VAE reconstruction PSNR/SSIM")
    parser.add_argument("--gt_dir", default=DEFAULT_GT_DIR, help="Directory with GT images.")
    parser.add_argument(
        "--vae",
        action="append",
        type=parse_vae_spec,
        help=(
            "VAE candidate as name=/local/path. Can be repeated. If omitted, "
            "the current local SD2.1 VAE is evaluated."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="results/diagnostics/vae_screening_eval1000",
        help="Directory for report.txt, report.json, and summary.csv.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--save_recon",
        action="store_true",
        help="Save reconstructed images under output_dir/reconstructions/<vae_name>/.",
    )
    return parser.parse_args()


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "vae"


def list_images(directory):
    directory = Path(directory)
    return [path for path in sorted(directory.iterdir()) if path.suffix.lower() in IMAGE_EXTS]


def load_rgb_tensor(path, image_size, device):
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8_rgb(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def metric_pair(pred_uint8, ref_uint8):
    return {
        "psnr": float(calculate_psnr(pred_uint8, ref_uint8, test_y_channel=False)),
        "ssim": float(calculate_ssim(pred_uint8, ref_uint8, test_y_channel=False)),
    }


def average(values):
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("inf")
    return float(np.mean(finite))


def load_vae(path, device):
    if AutoencoderKL is None:
        raise ImportError("diffusers is required for VAE screening") from IMPORT_ERROR
    if not Path(path).exists():
        raise FileNotFoundError(f"VAE path does not exist: {path}")
    load_errors = []
    if Path(path, "vae").is_dir():
        candidates = [(AutoencoderKL, {"subfolder": "vae"})]
    else:
        candidates = [(AutoencoderKL, {})]
        if ConsistencyDecoderVAE is not None:
            candidates.append((ConsistencyDecoderVAE, {}))

    vae = None
    vae_class_name = None
    for cls, kwargs in candidates:
        try:
            vae = cls.from_pretrained(path, **kwargs)
            vae_class_name = cls.__name__
            break
        except Exception as exc:  # noqa: BLE001 - keep fallback diagnostics in report
            load_errors.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    if vae is None:
        raise RuntimeError(f"Unable to load VAE candidate: {path}\n" + "\n".join(load_errors))

    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    return vae, scaling, vae_class_name


def reconstruct(vae, scaling_factor, rgb_0_1):
    rgb_minus1_1 = data_transform(rgb_0_1)
    posterior = vae.encode(rgb_minus1_1).latent_dist
    latent = posterior.mode() * scaling_factor
    decoded = vae.decode(latent / scaling_factor).sample
    return inverse_data_transform(decoded)


def write_reports(output_dir, args, candidate_results):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = {
        "gt_dir": str(Path(args.gt_dir)),
        "image_size": args.image_size,
        "max_images": args.max_images,
        "device": args.device,
        "candidates": candidate_results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vae_name", "loaded_as", "vae_path", "num_images", "avg_psnr", "avg_ssim"])
        for result in candidate_results:
            writer.writerow(
                [
                    result["name"],
                    result.get("loaded_as", ""),
                    result["path"],
                    result["num_images"],
                    f"{result['avg_psnr']:.6f}",
                    f"{result['avg_ssim']:.6f}",
                ]
            )

    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("VAE reconstruction screening report\n")
        f.write("==================================\n")
        f.write(f"gt_dir: {args.gt_dir}\n")
        f.write(f"image_size: {args.image_size}\n")
        f.write(f"max_images: {args.max_images}\n")
        f.write("\nAverage metrics\n")
        f.write(f"{'VAE':<28} | {'Loaded as':<21} | {'Images':>6} | {'PSNR':>9} | {'SSIM':>9}\n")
        f.write("-" * 86 + "\n")
        for result in candidate_results:
            f.write(
                f"{result['name']:<28} | {result.get('loaded_as', ''):<21} | "
                f"{result['num_images']:>6} | "
                f"{result['avg_psnr']:>9.4f} | {result['avg_ssim']:>9.4f}\n"
            )
        f.write("\nInterpretation:\n")
        f.write("- This is the VAE round-trip ceiling: GT -> VAE encode/decode -> reconstruction.\n")
        f.write("- If this PSNR is far below 30 dB, watermark training is unlikely to reach GT PSNR >30 without changing the VAE/output path.\n")


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    vae_specs = args.vae or [("sd21_current", DEFAULT_SD21_VAE)]
    output_dir = PROJECT_ROOT / args.output_dir
    images = list_images(args.gt_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found in {args.gt_dir}")

    all_results = []
    for name, path in vae_specs:
        safe_name = sanitize_name(name)
        print(f"==> Evaluating VAE: {name} ({path})")
        vae, scaling_factor, vae_class_name = load_vae(path, device)
        print(f"  loaded_as={vae_class_name}, scaling_factor={scaling_factor}")
        recon_dir = output_dir / "reconstructions" / safe_name
        if args.save_recon:
            recon_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        with torch.no_grad():
            for index, image_path in enumerate(images):
                gt_tensor = load_rgb_tensor(image_path, args.image_size, device)
                recon_tensor = reconstruct(vae, scaling_factor, gt_tensor)
                gt_uint8 = tensor_to_uint8_rgb(gt_tensor)
                recon_uint8 = tensor_to_uint8_rgb(recon_tensor)
                metrics = metric_pair(recon_uint8, gt_uint8)
                rows.append(
                    {
                        "index": index,
                        "stem": image_path.stem,
                        "psnr": metrics["psnr"],
                        "ssim": metrics["ssim"],
                    }
                )
                if args.save_recon:
                    Image.fromarray(recon_uint8, mode="RGB").save(recon_dir / f"{image_path.stem}.png")
                if (index + 1) % 50 == 0 or index == 0 or index + 1 == len(images):
                    print(
                        f"  {index + 1:04d}/{len(images)} {image_path.name}: "
                        f"PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}"
                    )

        per_image_csv = output_dir / f"{safe_name}_per_image.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "stem", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)

        result = {
            "name": name,
            "path": path,
            "loaded_as": vae_class_name,
            "scaling_factor": scaling_factor,
            "num_images": len(rows),
            "avg_psnr": average([row["psnr"] for row in rows]),
            "avg_ssim": average([row["ssim"] for row in rows]),
            "per_image_csv": str(per_image_csv),
        }
        print(f"==> {name}: PSNR={result['avg_psnr']:.4f}, SSIM={result['avg_ssim']:.4f}")
        all_results.append(result)
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_reports(output_dir, args, all_results)
    print(f"Report: {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
