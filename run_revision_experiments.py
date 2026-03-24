"""
Revision experiments for addressing reviewer comments.

This script runs multiple seeds for the five core RIFT conditions and stores
artifacts under `artifacts_multiseed/`.  It is designed to be invoked via:

    UV_CACHE_DIR=.uv-cache uv run python run_revision_experiments.py

Before launching each simulation, the script checks whether the corresponding
artifact directory already exists; if so, the run is skipped in order to
avoid wasting tokens on duplicate configurations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from main import build_run_name


SEEDS = [42, 43, 44, 45, 46]
OUTPUT_ROOT = Path("artifacts_multiseed")


def run(cmd: list[str]) -> None:
    print("\n==== Running ====\n", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    base = [
        "python",
        "main.py",
        "--model",
        "openai",
        "--model-name",
        "gpt-4o-mini",
        "--agents-per-side",
        "3",
        "--steps",
        "12",
        "--topology",
        "small_world",
        "--embedder",
        "hash",
        "--output-dir",
        str(OUTPUT_ROOT),
    ]

    conditions = [
        ("H-H85", dict(persona_mode="hard", homophily=0.85, recommender="homophily")),
        ("H-H50", dict(persona_mode="hard", homophily=0.5, recommender="homophily")),
        ("H-RAND", dict(persona_mode="hard", homophily=0.85, recommender="random")),
        ("S-H85", dict(persona_mode="soft", homophily=0.85, recommender="homophily")),
        ("S-H50", dict(persona_mode="soft", homophily=0.5, recommender="homophily")),
    ]

    for label, cfg in conditions:
        for seed in SEEDS:
            cmd = base + [
                "--persona-mode",
                cfg["persona_mode"],
                "--homophily",
                str(cfg["homophily"]),
                "--recommender",
                cfg["recommender"],
                "--seed",
                str(seed),
            ]

            # Mirror main.py's run-name logic so we can skip existing runs.
            fake_args = SimpleNamespace(
                persona_mode=cfg["persona_mode"],
                model="openai",
                model_name="gpt-4o-mini",
                steps=12,
                agents_per_side=3,
                topology="small_world",
                homophily=cfg["homophily"],
                recommender=cfg["recommender"],
                seed=seed,
                output_dir=OUTPUT_ROOT,
            )
            run_dir = OUTPUT_ROOT / build_run_name(fake_args)
            if run_dir.exists():
                print(f"Skip {label} seed={seed}: {run_dir} already exists.")
                continue
            run(cmd)


if __name__ == "__main__":
    main()

