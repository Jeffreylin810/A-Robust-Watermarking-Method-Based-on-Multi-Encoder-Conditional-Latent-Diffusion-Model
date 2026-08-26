#!/usr/bin/env python3
"""Calculate LPIPS between a GT directory and a result directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def list_images(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            mapping[path.stem] = path
    return mapping


def to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def batchify(paths_gt, paths_pred):
    gt_tensors = []
    pred_tensors = []
    stems = []
    for stem, gt_path, pred_path in zip(
        [item[0] for item in paths_gt], [item[1] for item in paths_gt], paths_pred
    ):
        gt_img = Image.open(gt_path).convert("RGB")
        pred_img = Image.open(pred_path).convert("RGB")
        if gt_img.size != pred_img.size:
            gt_img = gt_img.resize(pred_img.size, Image.Resampling.LANCZOS)
        gt_tensors.append(to_tensor(gt_img))
        pred_tensors.append(to_tensor(pred_img))
        stems.append(stem)
    return stems, torch.stack(gt_tensors, dim=0), torch.stack(pred_tensors, dim=0)


def main() -> None:
    args = parse_args()
    try:
        import lpips  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LPIPS package is not installed in the current environment. "
            "Install it with: /home/jeffrey/miniconda3/envs/diffw/bin/pip install lpips"
        ) from exc

    gt_dir = Path(args.gt_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_map = list_images(gt_dir)
    pred_map = list_images(results_dir)
    stems = sorted(set(gt_map) & set(pred_map))
    if not stems:
        raise RuntimeError("No matched images found between gt_dir and results_dir.")

    device = choose_device(args.device)
    loss_fn = lpips.LPIPS(net=args.net).to(device)
    loss_fn.eval()

    rows = []
    values = []
    with torch.inference_mode():
        for start in range(0, len(stems), args.batch_size):
            subset = stems[start : start + args.batch_size]
            gt_entries = [(stem, gt_map[stem]) for stem in subset]
            pred_entries = [pred_map[stem] for stem in subset]
            batch_stems, gt_batch, pred_batch = batchify(gt_entries, pred_entries)
            gt_batch = gt_batch.to(device)
            pred_batch = pred_batch.to(device)
            scores = loss_fn(gt_batch, pred_batch).reshape(-1).detach().cpu().numpy()
            for stem, score in zip(batch_stems, scores):
                score_f = float(score)
                rows.append({"stem": stem, "lpips": f"{score_f:.6f}"})
                values.append(score_f)

    mean = float(np.mean(values))
    std = float(np.std(values))
    min_v = float(np.min(values))
    max_v = float(np.max(values))

    with (output_dir / "per_image_lpips.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "count": len(values),
        "mean_lpips": mean,
        "std_lpips": std,
        "min_lpips": min_v,
        "max_lpips": max_v,
        "net": args.net,
        "gt_dir": str(gt_dir),
        "results_dir": str(results_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# LPIPS Summary",
                "",
                f"- count: {len(values)}",
                f"- mean LPIPS: {mean:.6f}",
                f"- std LPIPS: {std:.6f}",
                f"- min LPIPS: {min_v:.6f}",
                f"- max LPIPS: {max_v:.6f}",
                f"- net: {args.net}",
                f"- gt_dir: {gt_dir}",
                f"- results_dir: {results_dir}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"saved LPIPS summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
