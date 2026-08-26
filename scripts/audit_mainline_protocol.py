#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import torch
import yaml


OURS_ROOT = Path("/home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER")
DIFFW_ROOT = Path(
    "/home/jeffrey/projects/"
    "diffw_comparison_experiment_5_27_archived_noncurrent_20260528/"
    "DiffW_256_msg32_linear_baseline"
)
DATA_ROOT = Path(
    "/home/jeffrey/projects/experiment_input_folder/"
    "CoCo_256_paper_final"
)
OUTPUT_ROOT = Path(
    "/home/jeffrey/projects/實驗結果/"
    "16_mainline_formal_experiments/00_protocol_lock"
)

DIFFW_TRAIN_CONFIG = (
    DIFFW_ROOT / "configs/CoCo_train10000_val100_monitor_linear_steps20k.yml"
)
DIFFW_EVAL_CONFIG = (
    DIFFW_ROOT / "configs/CoCo_train10000_eval1000_unseen_linear.yml"
)
OURS_TRAIN_CONFIG = (
    OURS_ROOT / "configs/CoCo_train10000_ours256_msg32_cl3_psnr36_hinge20k.yml"
)
OURS_CONT_CONFIG = (
    OURS_ROOT
    / "configs/CoCo_train10000_ours256_msg32_base20k_then_nongeomlight5k.yml"
)
OURS_EVAL_CONFIG = (
    OURS_ROOT / "configs/CoCo_eval1000_ours256_msg32_nongeomlight_base.yml"
)

TRAIN_MANIFEST_A = DATA_ROOT / "train10000_val100_monitor/manifests/train.txt"
TRAIN_MANIFEST_B = DATA_ROOT / "train10000_eval1000_unseen/manifests/train.txt"
DEV_MANIFEST = DATA_ROOT / "train10000_eval1000_unseen/manifests/val.txt"
FINAL_TEST_MANIFEST = DATA_ROOT / "manifests/finaltest1000_locked.txt"

CHECKPOINTS = {
    "DiffW-like 20k": (
        DIFFW_ROOT
        / "Result/CoCo_ddpm_exp8_diffw_like_train10000_msg32_steps20k_seed61_final.pth.tar"
    ),
    "O256 base 20k+5k": (
        OURS_ROOT
        / "Result/CoCo_ddpm_train10000_base20k_then_nongeomlight5k_seed61_final.pth.tar"
    ),
    "O256 final plugin": (
        OURS_ROOT
        / "results/refinement/refine10000_targetedformal_C500_seed61/refiner_final.pth"
    ),
}


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def line_set(path):
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def checkpoint_metadata(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    metadata = {
        "type": "dict",
        "keys": sorted(payload.keys()),
    }
    for key in ("epoch", "step", "global_step", "elapsed_sec", "elapsed_seconds"):
        if key in payload:
            metadata[key] = payload[key]
    if "args" in payload:
        args = payload["args"]
        metadata["args"] = vars(args) if hasattr(args, "__dict__") else str(args)
    return metadata


def nested(config, *keys):
    value = config
    for key in keys:
        value = value[key]
    return value


def main():
    required = [
        DIFFW_TRAIN_CONFIG,
        DIFFW_EVAL_CONFIG,
        OURS_TRAIN_CONFIG,
        OURS_CONT_CONFIG,
        OURS_EVAL_CONFIG,
        TRAIN_MANIFEST_A,
        TRAIN_MANIFEST_B,
        DEV_MANIFEST,
        FINAL_TEST_MANIFEST,
        *CHECKPOINTS.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing protocol inputs:\n" + "\n".join(missing))

    diffw_train = load_yaml(DIFFW_TRAIN_CONFIG)
    diffw_eval = load_yaml(DIFFW_EVAL_CONFIG)
    ours_train = load_yaml(OURS_TRAIN_CONFIG)
    ours_cont = load_yaml(OURS_CONT_CONFIG)
    ours_eval = load_yaml(OURS_EVAL_CONFIG)

    train_sha_a = sha256(TRAIN_MANIFEST_A)
    train_sha_b = sha256(TRAIN_MANIFEST_B)
    dev_sha = sha256(DEV_MANIFEST)
    final_test_sha = sha256(FINAL_TEST_MANIFEST)
    train_names = line_set(TRAIN_MANIFEST_A)
    dev_names = line_set(DEV_MANIFEST)
    final_test_names = line_set(FINAL_TEST_MANIFEST)

    hard_checks = {
        "same_train_manifest": train_sha_a == train_sha_b,
        "train_count_10000": line_count(TRAIN_MANIFEST_A) == 10000,
        "dev_count_1000": line_count(DEV_MANIFEST) == 1000,
        "final_test_count_1000": line_count(FINAL_TEST_MANIFEST) == 1000,
        "train_dev_disjoint": not train_names.intersection(dev_names),
        "train_final_test_disjoint": not train_names.intersection(final_test_names),
        "dev_final_test_disjoint": not dev_names.intersection(final_test_names),
        "image_size_256": (
            nested(diffw_train, "data", "image_size")
            == nested(ours_train, "data", "image_size")
            == 256
        ),
        "message_length_32": (
            nested(diffw_train, "watermarking", "message_length")
            == nested(ours_train, "watermarking", "message_length")
            == 32
        ),
        "batch_size_2": (
            nested(diffw_train, "training", "batch_size")
            == nested(ours_train, "training", "batch_size")
            == 2
        ),
        "dev_dataset_root_same": (
            nested(diffw_eval, "data", "dataset_root")
            == nested(ours_eval, "data", "dataset_root")
        ),
    }

    checkpoint_rows = []
    for name, path in CHECKPOINTS.items():
        checkpoint_rows.append(
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "metadata": checkpoint_metadata(path),
            }
        )

    report = {
        "hard_checks": hard_checks,
        "manifests": {
            "train_a": {
                "path": str(TRAIN_MANIFEST_A),
                "count": line_count(TRAIN_MANIFEST_A),
                "sha256": train_sha_a,
            },
            "train_b": {
                "path": str(TRAIN_MANIFEST_B),
                "count": line_count(TRAIN_MANIFEST_B),
                "sha256": train_sha_b,
            },
            "dev": {
                "path": str(DEV_MANIFEST),
                "count": line_count(DEV_MANIFEST),
                "sha256": dev_sha,
                "role": "development/model selection",
            },
            "final_test": {
                "path": str(FINAL_TEST_MANIFEST),
                "count": line_count(FINAL_TEST_MANIFEST),
                "sha256": final_test_sha,
                "role": "locked final test; no tuning",
            },
        },
        "methods": {
            "DiffW-like": {
                "base_steps": 20000,
                "epochs": nested(diffw_train, "training", "n_epochs"),
                "learning_rate": nested(diffw_train, "optim", "lr"),
                "attack_profile": "combined_attack",
                "message_map": "linear",
                "latent_vae": False,
                "rgb_refinement": False,
            },
            "O256-RGBREFINE": {
                "base_steps": 25000,
                "epochs": nested(ours_cont, "training", "n_epochs"),
                "learning_rate_initial": nested(ours_train, "optim", "lr"),
                "learning_rate_continuation": nested(ours_cont, "optim", "lr"),
                "attack_profile_initial": nested(
                    ours_train, "watermarking", "attack_profile"
                ),
                "attack_profile_continuation": nested(
                    ours_cont, "watermarking", "attack_profile"
                ),
                "message_map": nested(
                    ours_train, "watermarking", "message_map_mode"
                ),
                "latent_vae": True,
                "rgb_refinement": True,
            },
        },
        "checkpoints": checkpoint_rows,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "protocol_audit.json"
    md_path = OUTPUT_ROOT / "protocol_audit.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    check_rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in hard_checks.items()
    )
    checkpoint_md = "\n".join(
        f"| {row['name']} | {row['metadata'].get('step', 'n/a')} | "
        f"{row['metadata'].get('epoch', 'n/a')} | `{row['sha256']}` | `{row['path']}` |"
        for row in checkpoint_rows
    )
    md = f"""# Development Artifact and Dataset Pre-run Audit

Generated by `scripts/audit_mainline_protocol.py`.

This audit does not run any model on the locked final test. It only verifies
dataset separation, frozen artifacts, and checkpoint metadata.

The checkpoint rows below are development artifacts that define the frozen
recipes. They are not the new formal-clean checkpoints. Formal-clean hashes
must replace them after from-scratch retraining.

## Hard Fairness Checks

| Check | Result |
|---|---|
{check_rows}

## Dataset Manifests

| Split | Count | SHA-256 |
|---|---:|---|
| train10000 | {line_count(TRAIN_MANIFEST_A)} | `{train_sha_a}` |
| dev1000 (old unseen eval1000) | {line_count(DEV_MANIFEST)} | `{dev_sha}` |
| finaltest1000_locked | {line_count(FINAL_TEST_MANIFEST)} | `{final_test_sha}` |

The DiffW-like and O256 projects use byte-identical train manifests.
The three manifests are pairwise disjoint. `dev1000` was used for method
selection; `finaltest1000_locked` must not be used for tuning.

## Checkpoint Lock

| Role | Step | Epoch | SHA-256 | Path |
|---|---:|---:|---|---|
{checkpoint_md}

## Method-Specific Training Differences

| Item | DiffW-like | O256-RGBREFINE |
|---|---|---|
| Base steps | 20k | 20k + 5k |
| LR | 2e-5 | 2e-5 -> 1e-5 |
| Attack schedule | combined attack | identity -> nongeom-light |
| Message map | linear | bitplane multi-scale |
| Latent VAE | no | frozen SD2.1 VAE |
| RGB refinement | no | refine10000 final plugin |

Conclusion: the development artifacts are traceable and a fresh final-test
split is locked. The next step is from-scratch formal-clean retraining with the
frozen recipes. Existing dev1000 results remain development evidence; the
Chapter 4 quality table begins only after new checkpoint hashes are recorded
and finaltest1000 is evaluated once.
"""
    md_path.write_text(md, encoding="utf-8")

    failed = [name for name, passed in hard_checks.items() if not passed]
    print(f"Protocol audit: {md_path}")
    print(f"Protocol JSON: {json_path}")
    if failed:
        raise RuntimeError("Protocol hard checks failed: " + ", ".join(failed))
    print("All hard fairness checks passed.")


if __name__ == "__main__":
    main()
