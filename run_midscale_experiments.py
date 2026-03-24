"""
Mid-scale experiment (reviewer request a): N=50, longer horizon.

Runs two conditions for a clean, controlled contrast:
  - hard persona + homophily recommender (homophily=0.85)
  - hard persona + random recommender (homophily=0.85)

Artifacts are written under `artifacts_midscale/`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from main import build_run_name


OUTPUT_ROOT = Path("artifacts_midscale")


def run(cmd: list[str]) -> None:
    print("\n==== Running ====\n", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    base = [
        "python",
        "main.py",
        "--model",
        "openai",
        "--model-name",
        "gpt-5-nano",
        "--agents-per-side",
        "25",
        "--steps",
        "50",
        "--topology",
        "small_world",
        # "--service-tier",
        # "flex",
        "--embedder",
        "sbert",
        "--output-dir",
        str(OUTPUT_ROOT),
        "--seed",
        "42",
    ]

    conditions = [
        ("H-H85", dict(persona_mode="hard", homophily=0.85, recommender="homophily")),
        ("H-RAND", dict(persona_mode="hard", homophily=0.85, recommender="random")),
    ]

    for label, cfg in conditions:
        cmd = base + [
            "--persona-mode",
            cfg["persona_mode"],
            "--homophily",
            str(cfg["homophily"]),
            "--recommender",
            cfg["recommender"],
        ]

        fake_args = SimpleNamespace(
            persona_mode=cfg["persona_mode"],
            model="openai",
            model_name="gpt-5-nano",
            service_tier="flex",
            steps=50,
            agents_per_side=25,
            topology="small_world",
            homophily=cfg["homophily"],
            recommender=cfg["recommender"],
            seed=42,
            output_dir=OUTPUT_ROOT,
        )
        run_dir = OUTPUT_ROOT / build_run_name(fake_args)
        if run_dir.exists():
            print(f"Skip {label}: {run_dir} already exists.")
            continue
        run(cmd)


if __name__ == "__main__":
    main()
