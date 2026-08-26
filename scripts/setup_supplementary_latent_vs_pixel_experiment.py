#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_DATE_TAG = "20260627"

DATA_ROOT = Path("/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final")
TRAIN1000_MONITOR = DATA_ROOT / "train1000_val100_monitor"
EVAL1000_UNSEEN = DATA_ROOT / "train1000_eval1000_unseen"
LATENT_VAE_PATH = "/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base"

FORMAL_TRAIN_CONFIG = CONFIG_ROOT / "CoCo_train10000_ours256_msg32_identity20k_then_nongeomlight10k_seed61_formal.yml"
FORMAL_EVAL_CONFIG = CONFIG_ROOT / "CoCo_eval1000_ours256_msg32_cl3_psnr36_hinge20k.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up supplementary latent-vs-pixel experiment assets.")
    parser.add_argument("--date_tag", default=DEFAULT_DATE_TAG)
    parser.add_argument(
        "--output_root",
        default="",
        help="Optional override for the supplementary experiment root.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_csv_headers(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def build_train_config(
    base: dict,
    *,
    dataset_root: Path,
    output_dir: Path,
    n_epochs: int,
    attack_profile: str,
    latent_enabled: bool,
    message_map_mode: str,
) -> dict:
    cfg = deepcopy(base)
    cfg["data"]["dataset_root"] = str(dataset_root)
    cfg["data"]["output_dir"] = str(output_dir)
    cfg["training"]["n_epochs"] = n_epochs
    cfg["watermarking"]["attack_profile"] = attack_profile
    cfg["watermarking"]["message_map_mode"] = message_map_mode
    cfg["watermarking"]["message_recovery_path"] = "pixel_bridge"
    cfg["watermarking"]["attack_uses_cover"] = False
    if latent_enabled:
        cfg["latent"] = {
            "enabled": True,
            "model_path": LATENT_VAE_PATH,
            "scaling_factor": None,
            "deterministic_encode": True,
        }
    else:
        cfg["latent"] = {
            "enabled": False,
        }
    return cfg


def build_eval_config(
    base: dict,
    *,
    dataset_root: Path,
    output_dir: Path,
    latent_enabled: bool,
    message_map_mode: str,
) -> dict:
    cfg = deepcopy(base)
    cfg["data"]["dataset_root"] = str(dataset_root)
    cfg["data"]["output_dir"] = str(output_dir)
    cfg["watermarking"]["message_map_mode"] = message_map_mode
    cfg["watermarking"]["message_recovery_path"] = "pixel_bridge"
    cfg["watermarking"]["attack_uses_cover"] = False
    if latent_enabled:
        cfg["latent"] = {
            "enabled": True,
            "model_path": LATENT_VAE_PATH,
            "scaling_factor": None,
            "deterministic_encode": True,
        }
    else:
        cfg["latent"] = {
            "enabled": False,
        }
    return cfg


def main() -> None:
    args = parse_args()
    date_tag = args.date_tag
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(
            f"/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/07_supplementary_latent_vs_pixel_experiment_{date_tag}"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "generated_configs").mkdir(parents=True, exist_ok=True)

    train_base = load_yaml(FORMAL_TRAIN_CONFIG)
    eval_base = load_yaml(FORMAL_EVAL_CONFIG)

    train_specs = [
        {
            "name": f"CoCo_train1000_supp_latent_bitplane_identity20k_seed61_{date_tag}.yml",
            "dataset_root": TRAIN1000_MONITOR,
            "output_dir": output_root / "runs" / "latent_bitplane_identity20k_train1000",
            "n_epochs": 40,
            "attack_profile": "identity_only",
            "latent_enabled": True,
            "message_map_mode": "bitplane",
        },
        {
            "name": f"CoCo_train1000_supp_latent_bitplane_identity20k_then_nongeomlight10k_seed61_{date_tag}.yml",
            "dataset_root": TRAIN1000_MONITOR,
            "output_dir": output_root / "runs" / "latent_bitplane_base30k_train1000",
            "n_epochs": 60,
            "attack_profile": "identity_nongeom_curriculum_light",
            "latent_enabled": True,
            "message_map_mode": "bitplane",
        },
        {
            "name": f"CoCo_train1000_supp_latent_linear_identity20k_seed61_{date_tag}.yml",
            "dataset_root": TRAIN1000_MONITOR,
            "output_dir": output_root / "runs" / "latent_linear_identity20k_train1000",
            "n_epochs": 40,
            "attack_profile": "identity_only",
            "latent_enabled": True,
            "message_map_mode": "linear",
        },
        {
            "name": f"CoCo_train1000_supp_latent_linear_identity20k_then_nongeomlight10k_seed61_{date_tag}.yml",
            "dataset_root": TRAIN1000_MONITOR,
            "output_dir": output_root / "runs" / "latent_linear_base30k_train1000",
            "n_epochs": 60,
            "attack_profile": "identity_nongeom_curriculum_light",
            "latent_enabled": True,
            "message_map_mode": "linear",
        },
        {
            "name": f"CoCo_train1000_supp_pixel_bitplane_identity20k_seed61_{date_tag}.yml",
            "dataset_root": TRAIN1000_MONITOR,
            "output_dir": output_root / "runs" / "pixel_bitplane_identity20k_train1000",
            "n_epochs": 40,
            "attack_profile": "identity_only",
            "latent_enabled": False,
            "message_map_mode": "bitplane",
        },
        {
            "name": f"CoCo_train1000_supp_pixel_bitplane_identity20k_then_nongeomlight10k_seed61_{date_tag}.yml",
            "dataset_root": TRAIN1000_MONITOR,
            "output_dir": output_root / "runs" / "pixel_bitplane_base30k_train1000",
            "n_epochs": 60,
            "attack_profile": "identity_nongeom_curriculum_light",
            "latent_enabled": False,
            "message_map_mode": "bitplane",
        },
    ]

    eval_specs = [
        {
            "name": f"CoCo_eval1000_supp_latent_bitplane_train1000_seed61_{date_tag}.yml",
            "dataset_root": EVAL1000_UNSEEN,
            "output_dir": output_root / "eval" / "latent_bitplane_train1000",
            "latent_enabled": True,
            "message_map_mode": "bitplane",
        },
        {
            "name": f"CoCo_eval1000_supp_latent_linear_train1000_seed61_{date_tag}.yml",
            "dataset_root": EVAL1000_UNSEEN,
            "output_dir": output_root / "eval" / "latent_linear_train1000",
            "latent_enabled": True,
            "message_map_mode": "linear",
        },
        {
            "name": f"CoCo_eval1000_supp_pixel_bitplane_train1000_seed61_{date_tag}.yml",
            "dataset_root": EVAL1000_UNSEEN,
            "output_dir": output_root / "eval" / "pixel_bitplane_train1000",
            "latent_enabled": False,
            "message_map_mode": "bitplane",
        },
    ]

    manifest = {
        "experiment_root": str(output_root),
        "train_specs": [],
        "eval_specs": [],
        "notes": [
            "All configs are proof-of-concept train1000/eval1000 supplementary configs.",
            "They must not overwrite formal train10000/finaltest1000 artifacts.",
            "Latent bitplane is the matched current-project miniature reference.",
            "Latent linear is the message-representation ablation branch.",
            "Pixel bitplane is the matched pixel-space baseline branch.",
        ],
    }

    for spec in train_specs:
        cfg = build_train_config(
            train_base,
            dataset_root=spec["dataset_root"],
            output_dir=spec["output_dir"],
            n_epochs=spec["n_epochs"],
            attack_profile=spec["attack_profile"],
            latent_enabled=spec["latent_enabled"],
            message_map_mode=spec["message_map_mode"],
        )
        config_path = CONFIG_ROOT / spec["name"]
        dump_yaml(config_path, cfg)
        dump_yaml(output_root / "generated_configs" / spec["name"], cfg)
        manifest["train_specs"].append(
            {
                "config_name": spec["name"],
                "config_path": str(config_path),
                "output_dir": str(spec["output_dir"]),
                "n_epochs": spec["n_epochs"],
                "latent_enabled": spec["latent_enabled"],
                "message_map_mode": spec["message_map_mode"],
                "attack_profile": spec["attack_profile"],
            }
        )

    for spec in eval_specs:
        cfg = build_eval_config(
            eval_base,
            dataset_root=spec["dataset_root"],
            output_dir=spec["output_dir"],
            latent_enabled=spec["latent_enabled"],
            message_map_mode=spec["message_map_mode"],
        )
        config_path = CONFIG_ROOT / spec["name"]
        dump_yaml(config_path, cfg)
        dump_yaml(output_root / "generated_configs" / spec["name"], cfg)
        manifest["eval_specs"].append(
            {
                "config_name": spec["name"],
                "config_path": str(config_path),
                "output_dir": str(spec["output_dir"]),
                "latent_enabled": spec["latent_enabled"],
                "message_map_mode": spec["message_map_mode"],
            }
        )

    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme_lines = [
        "# Supplementary Latent vs Pixel Experiment",
        "",
        f"- experiment_root: `{output_root}`",
        "- purpose: build a current-project, train1000/eval1000 proof-of-concept comparison among",
        "  1. latent bitplane (matched O256 miniature reference)",
        "  2. latent linear (message-representation ablation)",
        "  3. pixel bitplane (matched pixel-space baseline)",
        "",
        "## Guardrails",
        "",
        "- This is a supplementary experiment only.",
        "- Do not mix these results into locked finaltest1000 formal main tables.",
        "- Pixel baseline keeps the same O256-PAPER training/evaluation skeleton, but disables latent mode (`latent.enabled: false`).",
        "- If pixel-space behavior differs strongly, describe it as an exploratory matched baseline rather than a universal pixel-space conclusion.",
        "",
        "## Dataset / cost choice",
        "",
        "- train split: `train1000_val100_monitor`",
        "- eval split: `train1000_eval1000_unseen`",
        "- message length: `32`",
        "- resolution: `256x256`",
        "- batch size: `2`",
        "- identity20k on train1000 = `40 epochs`",
        "- identity20k + nongeom-light10k on train1000 = `60 epochs total`",
        "",
        "## Outputs scaffolded here",
        "",
        "- `generated_configs/`",
        "- `summary.csv`",
        "- `per_attack_ba.csv`",
        "- `timing.csv`",
        "- `thesis_ready_summary.md`",
        "- `figures/`",
        "",
        "Run setup commands are documented in the top-level command file:",
        f"- `{PROJECT_ROOT / 'commands_supplementary_latent_vs_pixel_experiment_20260627.md'}`",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (output_root / "commands.md").write_text(
        "See `/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER/commands_supplementary_latent_vs_pixel_experiment_20260627.md`.\n",
        encoding="utf-8",
    )
    (output_root / "thesis_ready_summary.md").write_text(
        "# Thesis-ready Summary\n\nPending experiment runs.\n",
        encoding="utf-8",
    )

    write_csv_headers(
        output_root / "summary.csv",
        [
            "experiment",
            "subset",
            "method",
            "training_steps",
            "psnr",
            "ssim",
            "lpips",
            "clean_ba",
            "nongeom_avg_ba",
            "min_ba",
            "notes",
        ],
    )
    write_csv_headers(
        output_root / "per_attack_ba.csv",
        ["experiment", "subset", "method", "attack", "ba", "notes"],
    )
    write_csv_headers(
        output_root / "timing.csv",
        ["experiment", "subset", "method", "total_seconds", "seconds_per_image", "batch_size", "notes"],
    )

    print(f"experiment scaffold created at: {output_root}")
    print(f"manifest: {output_root / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
