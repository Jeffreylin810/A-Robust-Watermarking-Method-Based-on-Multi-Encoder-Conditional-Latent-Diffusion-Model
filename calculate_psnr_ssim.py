import argparse
import os

import cv2

from utils.metrics import calculate_psnr, calculate_ssim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate PSNR and SSIM between generated images and ground truth."
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/home/jeffrey/projects/Test/L-DiffW/data/CoCo/val/gt/",
        help="Path to ground truth images directory",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/home/jeffrey/projects/Test/L-DiffW/results/images_CoCo/CoCo/data/CoCo/val/input/",
        help="Path to generated images directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gt_path = args.gt_path
    results_path = args.results_path

    imgsName = sorted([f for f in os.listdir(results_path) if f.endswith((".png", ".jpg"))])
    gtsName = sorted([f for f in os.listdir(gt_path) if f.endswith((".png", ".jpg"))])
    assert len(imgsName) == len(gtsName)

    cumulative_psnr, cumulative_ssim = 0, 0
    for i in range(len(imgsName)):
        print("Processing image: %s" % (imgsName[i]))
        res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
        cur_psnr = calculate_psnr(res, gt, test_y_channel=False)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=False)
        print("PSNR is %.4f and SSIM is %.4f" % (cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim

    print(
        "Testing set, PSNR is %.4f and SSIM is %.4f"
        % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName))
    )
    print(results_path)


if __name__ == "__main__":
    main()
