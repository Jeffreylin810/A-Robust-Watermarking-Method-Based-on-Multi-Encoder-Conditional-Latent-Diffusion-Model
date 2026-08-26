import argparse
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from noise_layers.jpeg import JpegTest
from noise_layers.gaussian_filter import GF
from noise_layers.crop import Crop, Cropout, Dropout
from noise_layers.gaussian_noise import GN
from noise_layers.middle_filter import MF
from noise_layers.salt_pepper_noise import SP
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DiffW paper-matched native attacks for BA comparison."
    )
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/attacked_Result")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--attack_scope",
        choices=("all", "nongeom"),
        default="all",
        help="Use the full paper-matched suite or exclude Crop/Cropout geometry attacks.",
    )
    return parser.parse_args()


def get_img_tensor(img_path, device):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def save_tensor_img(tensor, path):
    img = tensor.squeeze(0).cpu().detach().clone()
    img = (img / 2 + 0.5).clamp(0, 1)
    img = transforms.ToPILImage()(img)
    img.save(path)


class NativeDiffWAttacker:
    def __init__(self, noise_layer, device):
        self.noise_layer = noise_layer.to(device)
        self.noise_layer.eval()
        self.device = device

    def attack(self, watermarked_path, cover_path, out_path):
        watermarked = get_img_tensor(watermarked_path, self.device)
        cover = get_img_tensor(cover_path, self.device)
        with torch.no_grad():
            attacked = self.noise_layer([watermarked, cover])
        save_tensor_img(attacked, out_path)


def init_paper_matched_attackers(device):
    # Table VII: trained attack comparison.
    # Table VIII: 64-bit capacity setting with combined attacks.
    # Some paper symbols use sigma/r/p. We keep names explicit so BA reports are easy to map.
    return {
        "Paper_GaussianBlur_sigma2": NativeDiffWAttacker(GF(2), device),
        "Paper_GaussianBlur_sigma3": NativeDiffWAttacker(GF(3), device),
        "Paper_GaussianBlur_sigma4": NativeDiffWAttacker(GF(4), device),
        "Paper_JPEG_Q30": NativeDiffWAttacker(JpegTest(30), device),
        "Paper_JPEG_Q40": NativeDiffWAttacker(JpegTest(40), device),
        "Paper_JPEG_Q50": NativeDiffWAttacker(JpegTest(50), device),
        "Paper_Crop_p0035": NativeDiffWAttacker(Crop(0.035, 0.035), device),
        "Paper_Crop_p0045": NativeDiffWAttacker(Crop(0.045, 0.045), device),
        "Paper_Crop_p0055": NativeDiffWAttacker(Crop(0.055, 0.055), device),
        "Paper_Cropout_r03": NativeDiffWAttacker(Cropout(0.3, 0.3), device),
        "Paper_Dropout_r03": NativeDiffWAttacker(Dropout(0.3), device),
        # The local GN class names its argument `var` and internally uses sqrt(var)
        # as the standard deviation. Keep both interpretations to make the
        # paper-matching check explicit instead of hiding the ambiguity.
        "Paper_GaussianNoise_arg03": NativeDiffWAttacker(GN(0.3), device),
        "Paper_GaussianNoise_sigma03_var009": NativeDiffWAttacker(GN(0.09), device),
        "Paper_MedianFilter_w5": NativeDiffWAttacker(MF(5), device),
        "Paper_SaltPepper_r02": NativeDiffWAttacker(SP(0.2), device),
    }


def select_attackers(attackers, scope):
    if scope == "all":
        return attackers
    geometry_prefixes = ("Paper_Crop_", "Paper_Cropout_")
    return {
        name: attacker
        for name, attacker in attackers.items()
        if not name.startswith(geometry_prefixes)
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_root = os.path.join(args.output_dir, args.run_name)
    if os.path.exists(output_root) and not args.overwrite:
        raise FileExistsError(
            f"{output_root} already exists. Use --overwrite only if you intentionally want to replace it."
        )
    os.makedirs(output_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Writing paper-matched attacks to: {output_root}")
    print(f"Attack scope: {args.attack_scope}")

    image_names = sorted(
        [f for f in os.listdir(args.results_path) if f.endswith((".png", ".jpg"))]
    )
    gt_names = sorted([f for f in os.listdir(args.gt_path) if f.endswith((".png", ".jpg"))])
    assert len(image_names) == len(gt_names), (
        f"Mismatch: {len(image_names)} results vs {len(gt_names)} GTs"
    )
    if not image_names:
        raise RuntimeError(f"No images found in {args.results_path}")

    attackers = select_attackers(init_paper_matched_attackers(device), args.attack_scope)
    for attack_name in attackers:
        os.makedirs(os.path.join(output_root, attack_name), exist_ok=True)

    results_data = []
    for attack_name, attacker in attackers.items():
        print(f"\n--- Paper-matched attack: {attack_name} ---")
        attack_out_dir = os.path.join(output_root, attack_name)
        cumulative_psnr = 0.0
        cumulative_ssim = 0.0
        valid_count = 0

        for img_name, gt_name in zip(image_names, gt_names):
            res_img_path = os.path.join(args.results_path, img_name)
            gt_img_path = os.path.join(args.gt_path, gt_name)
            attacked_img_path = os.path.join(attack_out_dir, img_name)

            attacker.attack(res_img_path, gt_img_path, attacked_img_path)

            attacked = cv2.imread(attacked_img_path, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
            if attacked is None or gt_img is None:
                continue
            if attacked.shape != gt_img.shape:
                attacked = cv2.resize(
                    attacked,
                    (gt_img.shape[1], gt_img.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            cumulative_psnr += calculate_psnr(attacked, gt_img, test_y_channel=False)
            cumulative_ssim += calculate_ssim(attacked, gt_img, test_y_channel=False)
            valid_count += 1

        avg_psnr = cumulative_psnr / valid_count
        avg_ssim = cumulative_ssim / valid_count
        print(f"[{attack_name}] Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")
        results_data.append([attack_name, avg_psnr, avg_ssim])

    report_path = os.path.join(output_root, "metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Attack scope: {args.attack_scope}\n")
        f.write(f"{'Attack Name':<32} | {'Average PSNR':<15} | {'Average SSIM':<15}\n")
        f.write("-" * 72 + "\n")
        for attack_name, avg_psnr, avg_ssim in results_data:
            f.write(f"{attack_name:<32} | {avg_psnr:<15.4f} | {avg_ssim:<15.4f}\n")

    print("\nPaper-matched attack evaluation complete.")
    print(f"Attacked outputs: {output_root}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
