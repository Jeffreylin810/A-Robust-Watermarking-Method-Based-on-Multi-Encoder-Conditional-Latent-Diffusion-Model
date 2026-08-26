#!/home/jeffrey/miniconda3/envs/diffw/bin/python
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


DATE_TAG = "20260624"
ROOT = Path("/home/jeffrey/projects")
O256_REPO = ROOT / "diffw_comparison_experiment_5_27/O256-PAPER"
DIFFW_REPO = ROOT / "diffw_comparison_experiment_5_27_archived_noncurrent_20260528/DiffW_256_msg32_linear_baseline"
OUT_ROOT = ROOT / f"實驗結果/16_mainline_formal_experiments/03_efficiency/efficiency_amortized_cost_experiment_{DATE_TAG}"
GEN_CONFIG_DIR = OUT_ROOT / "generated_configs"
LOG_DIR = OUT_ROOT / "logs"
RUN_CONFIG_PATH = OUT_ROOT / "run_config_experiment_a.json"
CSV_PATH = OUT_ROOT / "training_timing_results.csv"
SUMMARY_PATH = OUT_ROOT / "training_timing_summary.md"

PYTHON = "/home/jeffrey/miniconda3/envs/diffw/bin/python"
GPU_NAME = "NVIDIA GeForce RTX 4090"
DATASET_ROOT = "/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked"
DATASET_NAME = "COCO"
NUM_TRAIN_IMAGES = 10000
IMAGE_SIZE = 256
BATCH_SIZE = 2
SEED = 61
WARMUP_STEPS = 100
STEP_ORDER = [40000, 20000, 10000, 5000]


@dataclass(frozen=True)
class Task:
    method: str
    num_steps: int
    repo: Path
    config_name: str
    config_path: Path
    run_dir: Path
    log_path: Path
    notes: str


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def epochs_for_steps(measure_steps: int) -> int:
    steps_per_epoch = NUM_TRAIN_IMAGES // BATCH_SIZE
    needed = WARMUP_STEPS + measure_steps
    return math.ceil(needed / steps_per_epoch)


def prepare_dirs():
    GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_o256_config(task_steps: int) -> Path:
    src = O256_REPO / "configs/CoCo_train10000_ours256_msg32_nongeomlight_fromscratch20k.yml"
    cfg = load_yaml(src)
    run_dir = OUT_ROOT / "training_scaling" / "o256_base" / f"steps_{task_steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["data"]["dataset_root"] = DATASET_ROOT
    cfg["data"]["output_dir"] = str(run_dir)
    cfg["training"]["n_epochs"] = epochs_for_steps(task_steps)
    cfg["training"]["snapshot_freq"] = 10**9
    cfg["training"]["validation_freq"] = 10**9
    cfg["training"]["save_checkpoints"] = False
    cfg["training"]["save_final_checkpoint"] = False
    cfg["training"]["detailed_profile"] = False
    cfg["training"]["profile_interval"] = 10
    cfg["training"]["benchmark_mode"] = True
    cfg["training"]["benchmark_warmup_steps"] = WARMUP_STEPS
    cfg["training"]["benchmark_measure_steps"] = task_steps
    cfg["training"]["benchmark_result_path"] = str(run_dir / "logs" / "benchmark_result.json")
    out = O256_REPO / "configs" / f"CoCo_benchmarkA_o256_nongeomlight_steps{task_steps}_seed{SEED}_{DATE_TAG}.yml"
    dump_yaml(out, cfg)
    return out


def make_diffw_config(task_steps: int) -> Path:
    src = DIFFW_REPO / "configs/CoCo_train10000_finaltest1000_locked_linear_steps20k_seed61_formal.yml"
    cfg = load_yaml(src)
    run_dir = OUT_ROOT / "training_scaling" / "diffw_like" / f"steps_{task_steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["data"]["dataset_root"] = DATASET_ROOT
    cfg["data"]["data_dir"] = str(run_dir)
    cfg["training"]["n_epochs"] = epochs_for_steps(task_steps)
    cfg["training"]["snapshot_freq"] = 10**9
    cfg["training"]["validation_freq"] = 10**9
    cfg["training"]["save_checkpoints"] = False
    cfg["training"]["save_final_checkpoint"] = False
    cfg["training"]["benchmark_mode"] = True
    cfg["training"]["benchmark_warmup_steps"] = WARMUP_STEPS
    cfg["training"]["benchmark_measure_steps"] = task_steps
    cfg["training"]["benchmark_result_path"] = str(run_dir / "logs" / "benchmark_result.json")
    out = DIFFW_REPO / "configs" / f"CoCo_benchmarkA_diffw_linear_steps{task_steps}_seed{SEED}_{DATE_TAG}.yml"
    dump_yaml(out, cfg)
    return out


def build_tasks():
    tasks = []
    for steps in STEP_ORDER:
        diffw_cfg = make_diffw_config(steps)
        tasks.append(
            Task(
                method="DiffW-like",
                num_steps=steps,
                repo=DIFFW_REPO,
                config_name=diffw_cfg.name,
                config_path=diffw_cfg,
                run_dir=OUT_ROOT / "training_scaling" / "diffw_like" / f"steps_{steps}",
                log_path=LOG_DIR / f"experimentA_diffw_like_steps{steps}.log",
                notes="Local DiffW-like linear baseline under unified train10000/256/msg32 benchmark protocol.",
            )
        )
        o256_cfg = make_o256_config(steps)
        tasks.append(
            Task(
                method="O256-Base",
                num_steps=steps,
                repo=O256_REPO,
                config_name=o256_cfg.name,
                config_path=o256_cfg,
                run_dir=OUT_ROOT / "training_scaling" / "o256_base" / f"steps_{steps}",
                log_path=LOG_DIR / f"experimentA_o256_base_steps{steps}.log",
                notes="Single-run O256 base timing benchmark using attack_profile=identity_nongeom_curriculum_light from scratch; timing-only benchmark, not a replacement for the formal staged recipe.",
            )
        )
    return tasks


def benchmark_json_for(task: Task) -> Path:
    return task.run_dir / "logs" / "benchmark_result.json"


def run_task(task: Task):
    if benchmark_json_for(task).is_file():
        return
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = task.run_dir / "ckpts" / "benchmark_do_not_resume.pth.tar"
    cmd = [PYTHON, "train_diffusion.py", "--config", task.config_name, "--resume", str(resume_path), "--seed", str(SEED)]
    start = datetime.now().isoformat(timespec="seconds")
    with open(task.log_path, "a", encoding="utf-8") as log:
        log.write(f"[{start}] COMMAND: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=task.repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        end = datetime.now().isoformat(timespec="seconds")
        log.write(f"[{end}] EXIT_CODE: {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark task failed: {task.method} {task.num_steps} steps")
    if not benchmark_json_for(task).is_file():
        raise FileNotFoundError(benchmark_json_for(task))


def collect_rows(tasks):
    rows = []
    for task in tasks:
        jp = benchmark_json_for(task)
        if not jp.is_file():
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": task.method,
                "num_steps": task.num_steps,
                "warmup_steps": data["warmup_steps"],
                "total_seconds": f"{data['measured_seconds']:.6f}",
                "seconds_per_step": f"{data['seconds_per_step']:.6f}",
                "gpu_name": data["gpu_name"],
                "batch_size": data["batch_size"],
                "image_size": data["image_size"],
                "dataset_name": DATASET_NAME,
                "num_train_images": NUM_TRAIN_IMAGES,
                "seed": data["seed"],
                "notes": task.notes,
            }
        )
    rows.sort(key=lambda r: (int(r["num_steps"]), r["method"]))
    return rows


def write_csv(rows):
    fieldnames = [
        "method",
        "num_steps",
        "warmup_steps",
        "total_seconds",
        "seconds_per_step",
        "gpu_name",
        "batch_size",
        "image_size",
        "dataset_name",
        "num_train_images",
        "seed",
        "notes",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def write_summary(rows, tasks):
    by_step = {}
    for row in rows:
        by_step.setdefault(int(row["num_steps"]), {})[row["method"]] = row

    lines = []
    lines.append("# Training Timing Summary")
    lines.append("")
    lines.append(f"- experiment root: `{OUT_ROOT}`")
    lines.append(f"- GPU: `{GPU_NAME}`")
    lines.append(f"- dataset root: `{DATASET_ROOT}`")
    lines.append(f"- warmup steps excluded from measured region: `{WARMUP_STEPS}`")
    lines.append("- execution order: DiffW-like 40k -> O256-Base 40k -> DiffW-like 20k -> O256-Base 20k -> DiffW-like 10k -> O256-Base 10k -> DiffW-like 5k -> O256-Base 5k")
    lines.append("")
    lines.append("| steps | method | measured time | seconds/step | speedup vs DiffW-like | notes |")
    lines.append("|---|---|---:|---:|---:|---|")
    for steps in sorted(set(STEP_ORDER)):
        diffw = by_step.get(steps, {}).get("DiffW-like")
        o256 = by_step.get(steps, {}).get("O256-Base")
        speedup_text = "pending"
        if diffw and o256:
            d = float(diffw["total_seconds"])
            o = float(o256["total_seconds"])
            speedup = (1.0 - (o / d)) * 100.0
            speedup_text = f"{speedup:+.2f}%"
        for method in ("DiffW-like", "O256-Base"):
            row = by_step.get(steps, {}).get(method)
            if row is None:
                lines.append(f"| {steps} | {method} | pending | pending | {speedup_text if method == 'O256-Base' else ''} | pending |")
            else:
                lines.append(
                    f"| {steps} | {method} | {format_hms(float(row['total_seconds']))} | {float(row['seconds_per_step']):.6f} | "
                    f"{speedup_text if method == 'O256-Base' else ''} | {row['notes']} |"
                )
    pending = [t for t in tasks if not benchmark_json_for(t).is_file()]
    lines.append("")
    if pending:
        lines.append("## Pending tasks")
        lines.append("")
        for task in pending:
            lines.append(f"- {task.method} {task.num_steps} steps")
    else:
        lines.append("## Completed")
        lines.append("")
        lines.append("- All requested Experiment A timing runs completed.")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_run_config(tasks):
    payload = {
        "experiment_name": "efficiency_amortized_cost_experiment_20260624",
        "date": DATE_TAG,
        "executed_now": ["Experiment A"],
        "gpu_name": GPU_NAME,
        "dataset_root": DATASET_ROOT,
        "dataset_name": DATASET_NAME,
        "num_train_images": NUM_TRAIN_IMAGES,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "warmup_steps": WARMUP_STEPS,
        "step_order": STEP_ORDER,
        "task_order": [{"method": t.method, "num_steps": t.num_steps} for t in tasks],
        "notes": {
            "DiffW-like": "local linear baseline benchmark under unified train10000/256/msg32 setup",
            "O256-Base": "single-run identity_nongeom_curriculum_light timing benchmark; timing-only, not a replacement for the formal staged recipe",
        },
        "paths": {
            "csv": str(CSV_PATH),
            "summary": str(SUMMARY_PATH),
            "log_dir": str(LOG_DIR),
        },
    }
    RUN_CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    prepare_dirs()
    tasks = build_tasks()
    update_run_config(tasks)
    rows = collect_rows(tasks)
    write_csv(rows)
    write_summary(rows, tasks)
    for task in tasks:
        run_task(task)
        rows = collect_rows(tasks)
        write_csv(rows)
        write_summary(rows, tasks)
    print(f"saved csv: {CSV_PATH}")
    print(f"saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
