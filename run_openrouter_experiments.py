"""
Run a cross-model robustness suite using OpenRouter-backed LLMs.

We evaluate three OpenRouter models on the five core RIFT conditions:
  - H-H85: hard persona, h=0.85, homophily recommender
  - H-H50: hard persona, h=0.5, homophily recommender
  - H-RAND: hard persona, h=0.85, random recommender
  - S-H85: soft persona, h=0.85, homophily recommender
  - S-H50: soft persona, h=0.5, homophily recommender

Each model–condition combination is run for a single seed (42) and written under
`artifacts_multimodel/`.  Before launching a simulation, this script checks
whether the corresponding artifact directory already exists and skips rerunning
it to avoid wasting tokens.

Usage (from project root):

    UV_CACHE_DIR=.uv-cache uv run python run_openrouter_experiments.py
    UV_CACHE_DIR=.uv-cache uv run python run_openrouter_experiments.py --max-parallel 2
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

from main import build_run_name


OUTPUT_ROOT = Path("artifacts_multimodel")


MODELS = {
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "devstral-2512": "mistralai/devstral-2512",
}


CONDITIONS = [
    ("H-H85", dict(persona_mode="hard", homophily=0.85, recommender="homophily")),
    ("H-H50", dict(persona_mode="hard", homophily=0.5, recommender="homophily")),
    ("H-RAND", dict(persona_mode="hard", homophily=0.85, recommender="random")),
    ("S-H85", dict(persona_mode="soft", homophily=0.85, recommender="homophily")),
    ("S-H50", dict(persona_mode="soft", homophily=0.5, recommender="homophily")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RIFT experiments with multiple OpenRouter models."
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of simulations to run in parallel (default: 1).",
    )
    return parser.parse_args()


def build_tasks(base_cmd: list[str]) -> list[tuple[list[str], str]]:
    tasks: list[tuple[list[str], str]] = []
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for model_tag, model_name in MODELS.items():
        for cond_label, cfg in CONDITIONS:
            cmd = base_cmd + [
                "--model-name",
                model_name,
                "--persona-mode",
                cfg["persona_mode"],
                "--homophily",
                str(cfg["homophily"]),
                "--recommender",
                cfg["recommender"],
            ]

            fake_args = SimpleNamespace(
                persona_mode=cfg["persona_mode"],
                model="openrouter",
                model_name=model_name,
                steps=12,
                agents_per_side=3,
                topology="small_world",
                homophily=cfg["homophily"],
                recommender=cfg["recommender"],
                seed=42,
                output_dir=OUTPUT_ROOT,
            )
            run_dir = OUTPUT_ROOT / build_run_name(fake_args)
            if run_dir.exists():
                print(f"Skip {model_tag}-{cond_label}: {run_dir} already exists.")
                continue
            label = f"{model_tag}-{cond_label}"
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
        # Launch up to max_parallel processes.
        while idx < len(tasks) and len(procs) < max_parallel:
            cmd, label = tasks[idx]
            idx += 1
            print(f"\n==== Running ({label}) ====\n", " ".join(cmd))
            p = subprocess.Popen(cmd)
            procs.append((p, cmd, label))

        # Wait for the first process to finish.
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
        "openrouter",
        "--agents-per-side",
        "3",
        "--steps",
        "12",
        "--topology",
        "small_world",
        "--seed",
        "42",
        "--output-dir",
        str(OUTPUT_ROOT),
    ]

    tasks = build_tasks(base_cmd)
    if not tasks:
        print("No new experiments to run.")
        return

    if args.max_parallel <= 1:
        run_serial(tasks)
    else:
        run_parallel(tasks, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()

