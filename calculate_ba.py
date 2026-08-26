import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.ddm import DenoisingDiffusion, data_transform


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="L-DiffW BA calculation")
    parser.add_argument("--config", type=str, default="CoCo.yml", help="Path to the config file")
    parser.add_argument(
        "--resume",
        default="/home/jeffrey/projects/Test/L-DiffW/ckpts/CoCo_ddpm.pth.tar",
        type=str,
        help="Path for the diffusion model checkpoint to load",
    )
    parser.add_argument(
        "--attack_dir",
        type=str,
        required=True,
        help="Path to the timestamped attack directory",
    )
    parser.add_argument(
        "--gt_message_dir",
        type=str,
        required=True,
        help="Path where the original *_message.npy files are stored",
    )
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    return args, dict2namespace(config)


def get_img_tensor(img_path, device, image_size):
    tf = transforms.Compose(
        [
            # TEMPLATE: follow the branch config instead of assuming 128x128,
            # so copied projects can evaluate larger-capacity runs correctly.
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    img_tensor = tf(img).unsqueeze(0).to(device)
    return data_transform(img_tensor)


def main():
    print("====== Bit Accuracy (BA) Evaluator ======")
    args, config = parse_args_and_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    diffusion = DenoisingDiffusion(args, config)
    if not os.path.isfile(args.resume):
        print(f"Error: Model checkpoint {args.resume} not found!")
        return
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    diffusion.model.eval()
    decoder = diffusion.model.module.decoder

    attack_folders = [f.path for f in os.scandir(args.attack_dir) if f.is_dir()]
    if not attack_folders:
        print(f"No attack subdirectories found in {args.attack_dir}")
        return

    results_data = []
    for folder in sorted(attack_folders):
        attack_name = os.path.basename(folder)
        img_files = sorted([f for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".webp"))])
        if len(img_files) == 0:
            continue

        print(f"\n--- Extracting Watermarks for: {attack_name} ---")
        total_bit_error = 0.0
        valid_img_count = 0

        for img_name in img_files:
            img_path = os.path.join(folder, img_name)
            base_name = os.path.splitext(img_name)[0]
            gt_message_path = os.path.join(args.gt_message_dir, f"{base_name}_message.npy")
            if not os.path.exists(gt_message_path):
                print(f"Warning: GT message file not found for {img_name}")
                continue

            original_message = np.load(gt_message_path)
            img_tensor = get_img_tensor(img_path, device, config.data.image_size)
            if diffusion.latent_enabled:
                img_tensor = diffusion.encode_images(img_tensor)

            with torch.no_grad():
                decoded_msg_tensor = decoder(img_tensor)

            decoded_rounded = decoded_msg_tensor.detach().cpu().numpy().round().clip(0, 1)
            bitwise_err = np.mean(np.abs(decoded_rounded - original_message))
            total_bit_error += bitwise_err
            valid_img_count += 1

        if valid_img_count > 0:
            avg_bit_error = total_bit_error / valid_img_count
            avg_bit_accuracy = (1.0 - avg_bit_error) * 100.0
            print(f"[{attack_name}] Average Bit Accuracy (BA): {avg_bit_accuracy:.2f}%")
            results_data.append([attack_name, avg_bit_accuracy])
        else:
            print(f"[{attack_name}] No valid images processed.")

    report_path = os.path.join(args.attack_dir, "ba_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'Attack Name':<25} | {'Average BA (%)':<15}\n")
        f.write("-" * 45 + "\n")
        for row in results_data:
            f.write(f"{row[0]:<25} | {row[1]:<15.2f}\n")

    print(f"Bit Accuracy report has been saved to: {report_path}")


if __name__ == "__main__":
    main()
