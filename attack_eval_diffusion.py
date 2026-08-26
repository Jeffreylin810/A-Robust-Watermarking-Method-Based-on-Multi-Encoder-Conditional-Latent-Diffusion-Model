import argparse
import os
import sys
from datetime import datetime

import cv2
import torch
from PIL import Image
from torchvision import transforms

# Import ZoDiac modules when available.
zodiac_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ZoDiac"))
if zodiac_path not in sys.path:
    sys.path.append(zodiac_path)

try:
    from main.wmattacker import BM3DAttacker, DiffWMAttacker, RotateAttacker, VAEWMAttacker
except ImportError:
    BM3DAttacker = None
    DiffWMAttacker = None
    RotateAttacker = None
    VAEWMAttacker = None

from noise_layers.Adjust_Brightness import Adjust_Brightness
from noise_layers.Adjust_Contrast import Adjust_Contrast
from noise_layers.crop import Crop
from noise_layers.gaussian_filter import GF
from noise_layers.gaussian_noise import GN
from noise_layers.identity import Identity
from noise_layers.jpeg import JpegTest
from noise_layers.middle_filter import MF
from noise_layers.salt_pepper_noise import SP
from noise_layers.webp import WebP
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate watermarked images against various attacks.")
    parser.add_argument(
        "--gt_path",
        type=str,
        default="./data/CoCo/val/gt/",
        help="Path to ground truth images directory",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to generated (watermarked) images directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/attacked_Result",
        help="Base directory to save attacked images",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional output folder name. If omitted, a timestamped folder is created automatically.",
    )
    parser.add_argument(
        "--enable_ai_attacks",
        action="store_true",
        help="Enable expensive latent diffusion and VAE based attacks from ZoDiac",
    )
    parser.add_argument(
        "--zodiac_model_path",
        type=str,
        default="/home/jeffrey/projects/ZoDiac/stable-diffusion-2-1-base",
        help="Path to the Stable Diffusion model for DiffWMAttacker",
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
    def __init__(self, diffw_noise_layer, device):
        self.noise_layer = diffw_noise_layer.to(device)
        self.noise_layer.eval()
        self.device = device

    def attack(self, image_paths, out_paths):
        for img_path, out_path in zip(image_paths, out_paths):
            img_tensor = get_img_tensor(img_path, self.device)
            with torch.no_grad():
                attacked_tensor = self.noise_layer([img_tensor, img_tensor])
            save_tensor_img(attacked_tensor, out_path)


def init_attackers(args, device):
    attackers = {
        "DiffW_JPEG_50": NativeDiffWAttacker(JpegTest(50), device),
        "DiffW_Crop_0.15": NativeDiffWAttacker(Crop(0.15, 0.15), device),
        "DiffW_GaussianFilter_2": NativeDiffWAttacker(GF(2), device),
        "DiffW_GaussianNoise_0.1": NativeDiffWAttacker(GN(0.1), device),
        "DiffW_MiddleFilter_3": NativeDiffWAttacker(MF(3), device),
        "DiffW_SaltPepper_0.1": NativeDiffWAttacker(SP(0.1), device),
        "DiffW_WebP_50": NativeDiffWAttacker(WebP(50), device),
        "DiffW_Brightness_1.1": NativeDiffWAttacker(Adjust_Brightness(1.1), device),
        "DiffW_Contrast_1.1": NativeDiffWAttacker(Adjust_Contrast(1.1), device),
        "DiffW_Identity": NativeDiffWAttacker(Identity(), device),
    }

    if args.enable_ai_attacks:
        if any(x is None for x in (DiffWMAttacker, VAEWMAttacker, RotateAttacker, BM3DAttacker)):
            raise ImportError("ZoDiac attackers are unavailable. Check the ZoDiac environment/import path.")
        from main.attdiffusion import ReSDPipeline

        att_pipe = ReSDPipeline.from_pretrained(
            args.zodiac_model_path, torch_dtype=torch.float16, revision="fp16"
        )
        att_pipe.set_progress_bar_config(disable=True)
        att_pipe.to(device)

        attackers.update(
            {
                "ZoDiac_Rotate_90": RotateAttacker(degree=90),
                "ZoDiac_BM3D": BM3DAttacker(),
                "ZoDiac_DiffAttacker_60": DiffWMAttacker(att_pipe, batch_size=1, noise_step=60, captions={}),
                "ZoDiac_Cheng2020_3": VAEWMAttacker("cheng2020-anchor", quality=3, metric="mse", device=device),
                "ZoDiac_BmsHj2018_3": VAEWMAttacker(
                    "bmshj2018-factorized", quality=3, metric="mse", device=device
                ),
            }
        )

    return attackers


def main():
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs_name = sorted([f for f in os.listdir(args.results_path) if f.endswith((".png", ".jpg"))])
    gts_name = sorted([f for f in os.listdir(args.gt_path) if f.endswith((".png", ".jpg"))])

    assert len(imgs_name) == len(gts_name), f"Mismatch: {len(imgs_name)} results vs {len(gts_name)} GTs"
    if len(imgs_name) == 0:
        print(f"No images found in {args.results_path}")
        return

    attackers = init_attackers(args, device)
    for attack_name in attackers.keys():
        os.makedirs(os.path.join(args.output_dir, attack_name), exist_ok=True)

    results_data = []
    for attack_name, attacker in attackers.items():
        attack_out_dir = os.path.join(args.output_dir, attack_name)
        cumulative_psnr, cumulative_ssim = 0, 0
        for img_name, gt_name in zip(imgs_name, gts_name):
            res_img_path = os.path.join(args.results_path, img_name)
            gt_img_path = os.path.join(args.gt_path, gt_name)
            attacked_img_path = os.path.join(attack_out_dir, img_name)

            with torch.no_grad():
                attacker.attack([res_img_path], [attacked_img_path])

            res_attacked = cv2.imread(attacked_img_path, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
            if res_attacked is None or gt_img is None:
                continue
            if res_attacked.shape != gt_img.shape:
                res_attacked = cv2.resize(
                    res_attacked, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_CUBIC
                )
            cumulative_psnr += calculate_psnr(res_attacked, gt_img, test_y_channel=False)
            cumulative_ssim += calculate_ssim(res_attacked, gt_img, test_y_channel=False)

        avg_psnr = cumulative_psnr / len(imgs_name)
        avg_ssim = cumulative_ssim / len(imgs_name)
        print(f"[{attack_name}] Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")
        results_data.append([attack_name, avg_psnr, avg_ssim])

    report_path = os.path.join(args.output_dir, "metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'Attack Name':<25} | {'Average PSNR':<15} | {'Average SSIM':<15}\n")
        f.write("-" * 65 + "\n")
        for row in results_data:
            f.write(f"{row[0]:<25} | {row[1]:<15.4f} | {row[2]:<15.4f}\n")

    print(f"Evaluation complete. Attacked outputs: {args.output_dir}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
