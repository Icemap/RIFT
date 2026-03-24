"""
Ablation experiments for JASSS revision.

This script runs a small set of additional gpt-4o-mini experiments to probe
homophily strength, longer horizons, and alternative topologies / sizes.
Artifacts are written under `artifacts_ablation/` and existing runs are
skipped to avoid duplicate costs.

Usage (from repo root):

    UV_CACHE_DIR=.uv-cache uv run python run_ablation_experiments.py
    UV_CACHE_DIR=.uv-cache uv run python run_ablation_experiments.py --max-parallel 2
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

from main import build_run_name


OUTPUT_ROOT = Path("artifacts_ablation")


# Each task: (label, cmd kwargs)
TASKS = [
    # Homophily sweep at 0.7 and 0.9 (baseline: hard persona, homophily recommender)
    ("H-H70", dict(steps=12, aps=3, topo="small_world", hom=0.7, rec="homophily", persona="hard", seed=50)),
    ("H-H90", dict(steps=12, aps=3, topo="small_world", hom=0.9, rec="homophily", persona="hard", seed=50)),
    # Longer horizons under fixed structure / persona, comparing recommender
    ("H-H85-long", dict(steps=24, aps=3, topo="small_world", hom=0.85, rec="homophily", persona="hard", seed=51)),
    ("H-RAND-long", dict(steps=24, aps=3, topo="small_world", hom=0.85, rec="random", persona="hard", seed=51)),
    # Alternative topology (scale-free) at small N
    ("H-scale_free", dict(steps=12, aps=3, topo="scale_free", hom=0.85, rec="homophily", persona="hard", seed=52)),
    # Modest upscaling demos
    ("H-H85-aps5", dict(steps=10, aps=5, topo="small_world", hom=0.85, rec="homophily", persona="hard", seed=53)),
    ("H-H85-aps10", dict(steps=8, aps=10, topo="small_world", hom=0.85, rec="homophily", persona="hard", seed=54)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RIFT ablation experiments (gpt-4o-mini).")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum concurrent simulations (default: 1).",
    )
    return parser.parse_args()


def build_tasks(base_cmd: list[str]) -> list[tuple[list[str], str]]:
    tasks: list[tuple[list[str], str]] = []
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for label, cfg in TASKS:
        cmd = base_cmd + [
            "--steps",
            str(cfg["steps"]),
            "--agents-per-side",
            str(cfg["aps"]),
            "--topology",
            cfg["topo"],
            "--homophily",
            str(cfg["hom"]),
            "--recommender",
            cfg["rec"],
            "--persona-mode",
            cfg["persona"],
            "--seed",
            str(cfg["seed"]),
        ]

        fake_args = SimpleNamespace(
            persona_mode=cfg["persona"],
            model="openai",
            model_name="gpt-4o-mini",
            steps=cfg["steps"],
            agents_per_side=cfg["aps"],
            topology=cfg["topo"],
            homophily=cfg["hom"],
            recommender=cfg["rec"],
            seed=cfg["seed"],
            output_dir=OUTPUT_ROOT,
        )
        run_dir = OUTPUT_ROOT / build_run_name(fake_args)
        if run_dir.exists():
            print(f"Skip {label}: {run_dir} already exists.")
            continue
        tasks.append((cmd, label))
    return tasks


def run_serial(tasks: list[tuple[list[str], str]]) -> None:
    for cmd, label in tasks:
        print(f"\n==== Running ({label}) ====\n", " ".join(cmd))
        subprocess.run(cmd, check=True)


def run_parallel(tasks: list[tuple[list[str], str]], max_parallel: int) -> None:
    """Run multiple subprocesses with a simple concurrency limit."""
    procs: list[tuple[subprocess.Popen, list[str], str]] = []
    idx = 0

    while idx < len(tasks) or procs:
        while idx < len(tasks) and len(procs) < max_parallel:
            cmd, label = tasks[idx]
            idx += 1
            print(f"\n==== Running ({label}) ====\n", " ".join(cmd))
            p = subprocess.Popen(cmd)
            procs.append((p, cmd, label))

        p, cmd, label = procs.pop(0)
        ret = p.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def main() -> None:
    args = parse_args()

    base_cmd = [
        "python",
        "main.py",
        "--model",
        "openai",
        "--model-name",
        "gpt-4o-mini",
        "--embedder",
        "hash",
        "--output-dir",
        str(OUTPUT_ROOT),
    ]

    tasks = build_tasks(base_cmd)
    if not tasks:
        print("No new ablation runs to launch.")
        return

    if args.max_parallel <= 1:
        run_serial(tasks)
    else:
        run_parallel(tasks, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
