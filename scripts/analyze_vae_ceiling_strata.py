#!/usr/bin/env python3
"""Stratify O256 fidelity by the VAE reconstruction ceiling.

This script consumes the outputs produced by oracle_vae_residual_sweep.py:

    baseline_per_image.csv
    beta_1p00_per_image.csv (or another beta file)

It groups images by GT-vs-VAE-cover PSNR and reports how watermarked PSNR and
oracle-corrected PSNR behave in each stratum.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.ddm import data_transform, inverse_data_transform  # noqa: E402

try:
    from diffusers import AutoencoderKL
except ImportError as exc:  # pragma: no cover
    AutoencoderKL = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_GT_DIR = (
    "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/"
    "train10000_eval1000_unseen/data/CoCo/val/gt"
)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PSNR strata by VAE ceiling")
    parser.add_argument(
        "--sweep_dir",
        required=True,
        help="Directory produced by oracle_vae_residual_sweep.py.",
    )
    parser.add_argument(
        "--beta",
        default="1.00",
        help="Beta file to summarize, e.g. 1.00 uses beta_1p00_per_image.csv.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for report.txt, summary CSVs, and optional visual samples.",
    )
    parser.add_argument(
        "--thresholds",
        default="20,25,30",
        help="Comma-separated GT-vs-VAE PSNR thresholds. Default makes <20, 20-25, 25-30, >=30.",
    )
    parser.add_argument("--samples_per_group", type=int, default=4)
    parser.add_argument("--gt_dir", default=DEFAULT_GT_DIR)
    parser.add_argument(
        "--watermarked_dir",
        default="",
        help="Optional watermarked dir for visual sample sheets.",
    )
    parser.add_argument(
        "--config",
        default="CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml",
        help="Config for optional VAE-cover visual reconstruction.",
    )
    parser.add_argument("--save_visual_samples", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def beta_name(beta):
    return "beta_" + re.sub(r"[^0-9A-Za-z-]+", "p", f"{float(beta):.2f}")


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


def read_csv_by_stem(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = row["stem"]
            parsed = {"stem": stem}
            for key, value in row.items():
                if key == "stem":
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows[stem] = parsed
    return rows


def parse_thresholds(raw):
    thresholds = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return sorted(thresholds)


def group_name(value, thresholds):
    lower = -math.inf
    for threshold in thresholds:
        if value < threshold:
            if math.isinf(lower):
                return f"lt_{threshold:g}"
            return f"{lower:g}_to_{threshold:g}"
        lower = threshold
    return f"ge_{lower:g}"


def pretty_group_name(name):
    if name.startswith("lt_"):
        return "< " + name[3:] + " dB"
    if name.startswith("ge_"):
        return ">= " + name[3:] + " dB"
    return name.replace("_to_", "-") + " dB"


def average(rows, key):
    values = [row[key] for row in rows]
    finite = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def percentile(rows, key, q):
    values = sorted(row[key] for row in rows if np.isfinite(row[key]))
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return float(values[index])


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def list_images_by_stem(directory):
    images = {}
    directory = Path(directory)
    if not directory.is_dir():
        return images
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.stem] = path
    return images


def load_rgb_tensor(path, image_size, device):
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8_rgb(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def load_uint8(path, image_size):
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def save_uint8(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def diff_map(a, b):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    if diff.max() > 0:
        diff = diff / diff.max() * 255.0
    return np.clip(np.rint(diff), 0, 255).astype(np.uint8)


def build_vae(config, device):
    if AutoencoderKL is None:
        raise ImportError("diffusers is required for visual sample VAE covers") from IMPORT_ERROR
    vae_path = config.latent.model_path
    if os.path.isdir(os.path.join(vae_path, "vae")):
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    scaling = getattr(config.latent, "scaling_factor", None)
    if scaling is None:
        scaling = getattr(vae.config, "scaling_factor", 1.0)
    return vae, scaling


def vae_reconstruct(vae, scaling_factor, rgb_0_1):
    rgb_minus1_1 = data_transform(rgb_0_1)
    posterior = vae.encode(rgb_minus1_1).latent_dist
    latent = posterior.mode() * scaling_factor
    decoded = vae.decode(latent / scaling_factor).sample
    return inverse_data_transform(decoded)


def make_contact_sheet(images, labels):
    cell_w, cell_h = images[0].shape[1], images[0].shape[0]
    label_h = 24
    sheet = Image.new("RGB", (cell_w * len(images), cell_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (image, label) in enumerate(zip(images, labels)):
        sheet.paste(Image.fromarray(image, mode="RGB"), (idx * cell_w, label_h))
        draw.text((idx * cell_w + 4, 4), label, fill=(0, 0, 0))
    return sheet


def write_visual_samples(args, selected_rows, output_dir):
    config = load_config(args.config)
    image_size = int(config.data.image_size)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    vae, scaling = build_vae(config, device)
    gt_images = list_images_by_stem(args.gt_dir)
    wm_images = list_images_by_stem(args.watermarked_dir)
    corrected_images = list_images_by_stem(Path(args.sweep_dir) / beta_name(args.beta))

    visual_dir = output_dir / "visual_samples"
    visual_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for row in selected_rows:
            stem = row["stem"]
            if stem not in gt_images or stem not in wm_images or stem not in corrected_images:
                continue
            gt_tensor = load_rgb_tensor(gt_images[stem], image_size, device)
            cover = tensor_to_uint8_rgb(gt_tensor)
            vae_cover = tensor_to_uint8_rgb(vae_reconstruct(vae, scaling, gt_tensor))
            watermarked = load_uint8(wm_images[stem], image_size)
            corrected = load_uint8(corrected_images[stem], image_size)
            diff_vae = diff_map(cover, vae_cover)
            diff_wm = diff_map(cover, watermarked)

            stem_dir = visual_dir / row["group"] / stem
            save_uint8(stem_dir / "cover.png", cover)
            save_uint8(stem_dir / "vae_cover.png", vae_cover)
            save_uint8(stem_dir / "watermarked.png", watermarked)
            save_uint8(stem_dir / f"corrected_{beta_name(args.beta)}.png", corrected)
            save_uint8(stem_dir / "diff_gt_vae.png", diff_vae)
            save_uint8(stem_dir / "diff_gt_watermarked.png", diff_wm)
            sheet = make_contact_sheet(
                [cover, vae_cover, watermarked, corrected, diff_vae, diff_wm],
                ["cover", "vae", "wm", f"corr {args.beta}", "diff vae", "diff wm"],
            )
            sheet.save(stem_dir / "contact_sheet.png")


def main():
    args = parse_args()
    sweep_dir = PROJECT_ROOT / args.sweep_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)

    baseline_path = sweep_dir / "baseline_per_image.csv"
    beta_path = sweep_dir / f"{beta_name(args.beta)}_per_image.csv"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing baseline CSV: {baseline_path}")
    if not beta_path.is_file():
        raise FileNotFoundError(f"Missing beta CSV: {beta_path}")

    baseline = read_csv_by_stem(baseline_path)
    beta_rows = read_csv_by_stem(beta_path)
    rows = []
    for stem in sorted(set(baseline) & set(beta_rows)):
        row = dict(baseline[stem])
        row["beta"] = beta_rows[stem]["beta"]
        row["gt_vs_corrected_psnr"] = beta_rows[stem]["gt_vs_corrected_psnr"]
        row["gt_vs_corrected_ssim"] = beta_rows[stem]["gt_vs_corrected_ssim"]
        row["group"] = group_name(row["gt_vs_vae_psnr"], thresholds)
        rows.append(row)

    if not rows:
        raise RuntimeError("No matched rows between baseline and beta CSV.")

    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)

    group_rows = []
    for name in sorted(groups, key=lambda g: (groups[g][0]["gt_vs_vae_psnr"])):
        gr = groups[name]
        group_rows.append(
            {
                "group": name,
                "label": pretty_group_name(name),
                "count": len(gr),
                "gt_vs_vae_psnr_mean": average(gr, "gt_vs_vae_psnr"),
                "gt_vs_vae_psnr_min": percentile(gr, "gt_vs_vae_psnr", 0.0),
                "gt_vs_vae_psnr_max": percentile(gr, "gt_vs_vae_psnr", 1.0),
                "gt_vs_wm_psnr_mean": average(gr, "gt_vs_wm_psnr"),
                "vae_vs_wm_psnr_mean": average(gr, "vae_vs_wm_psnr"),
                "gt_vs_corrected_psnr_mean": average(gr, "gt_vs_corrected_psnr"),
                "gt_vs_vae_ssim_mean": average(gr, "gt_vs_vae_ssim"),
                "gt_vs_wm_ssim_mean": average(gr, "gt_vs_wm_ssim"),
                "gt_vs_corrected_ssim_mean": average(gr, "gt_vs_corrected_ssim"),
            }
        )

    selected = []
    for name, gr in groups.items():
        by_low = sorted(gr, key=lambda r: r["gt_vs_vae_psnr"])
        by_high = sorted(gr, key=lambda r: r["gt_vs_vae_psnr"], reverse=True)
        for rank, row in enumerate(by_low[: args.samples_per_group]):
            item = dict(row)
            item["selection"] = f"lowest_{rank + 1}"
            selected.append(item)
        for rank, row in enumerate(by_high[: args.samples_per_group]):
            item = dict(row)
            item["selection"] = f"highest_{rank + 1}"
            selected.append(item)

    corr_vae_wm = pearson([r["gt_vs_vae_psnr"] for r in rows], [r["gt_vs_wm_psnr"] for r in rows])
    corr_vae_corrected = pearson([r["gt_vs_vae_psnr"] for r in rows], [r["gt_vs_corrected_psnr"] for r in rows])
    corr_wm_corrected = pearson([r["gt_vs_wm_psnr"] for r in rows], [r["gt_vs_corrected_psnr"] for r in rows])

    with open(output_dir / "group_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    with open(output_dir / "selected_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()) + ["selection"])
        writer.writeheader()
        writer.writerows(selected)

    report = {
        "sweep_dir": str(sweep_dir),
        "beta": float(args.beta),
        "thresholds": thresholds,
        "num_images": len(rows),
        "correlations": {
            "gt_vae_psnr_vs_gt_watermarked_psnr": corr_vae_wm,
            "gt_vae_psnr_vs_gt_corrected_psnr": corr_vae_corrected,
            "gt_watermarked_psnr_vs_gt_corrected_psnr": corr_wm_corrected,
        },
        "groups": group_rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("VAE ceiling stratified analysis\n")
        f.write("=" * 40 + "\n")
        f.write(f"sweep_dir: {sweep_dir}\n")
        f.write(f"beta summarized: {args.beta}\n")
        f.write(f"images: {len(rows)}\n\n")
        f.write("Correlations\n")
        f.write(f"GT-VAE PSNR vs GT-WM PSNR:        {corr_vae_wm:.4f}\n")
        f.write(f"GT-VAE PSNR vs GT-corrected PSNR: {corr_vae_corrected:.4f}\n")
        f.write(f"GT-WM PSNR vs GT-corrected PSNR:  {corr_wm_corrected:.4f}\n\n")
        f.write("Group summary\n")
        f.write(
            f"{'group':<12} | {'count':>5} | {'GT-VAE':>8} | {'GT-WM':>8} | "
            f"{'VAE-WM':>8} | {'corrected':>9} | {'GT-WM SSIM':>10} | {'corr SSIM':>9}\n"
        )
        f.write("-" * 98 + "\n")
        for row in group_rows:
            f.write(
                f"{row['label']:<12} | {row['count']:>5} | "
                f"{row['gt_vs_vae_psnr_mean']:>8.4f} | "
                f"{row['gt_vs_wm_psnr_mean']:>8.4f} | "
                f"{row['vae_vs_wm_psnr_mean']:>8.4f} | "
                f"{row['gt_vs_corrected_psnr_mean']:>9.4f} | "
                f"{row['gt_vs_wm_ssim_mean']:>10.4f} | "
                f"{row['gt_vs_corrected_ssim_mean']:>9.4f}\n"
            )
        f.write("\nInterpretation:\n")
        f.write("- Groups are defined by GT vs VAE-cover PSNR.\n")
        f.write("- If GT-WM PSNR follows GT-VAE PSNR, VAE ceiling is a strong predictor.\n")
        f.write("- Corrected PSNR uses an oracle cover residual and is not a deployable metric.\n")

    if args.save_visual_samples:
        if not args.watermarked_dir:
            raise ValueError("--watermarked_dir is required for visual samples.")
        write_visual_samples(args, selected, output_dir)

    print(f"Report saved to: {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
