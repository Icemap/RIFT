"""
Run a tiny 10-agent demo (5 per side) for two topologies as an illustrative benchmark.

This uses a single seed and short horizon. Default is a deterministic stub model
to avoid action-spec failures and extra cost; swap the model flags below if you
want a real LLM (e.g., gpt-4o-mini or z-ai/glm-4-32b on OpenRouter).

Usage (from repo root):
    UV_CACHE_DIR=.uv-cache uv run python scripts/run_demo_10_agents.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n==== Running ====\n", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    output_root = Path("artifacts_demo")
    output_root.mkdir(parents=True, exist_ok=True)

    base = [
        "python",
        "main.py",
        "--model",
        "stub",          # deterministic, no cost; change to openrouter/openai to use a real LLM
        "--model-name",
        "stub",
        "--agents-per-side",
        "5",             # 10 agents total
        "--steps",
        "6",             # short horizon
        "--homophily",
        "0.85",
        "--recommender",
        "homophily",
        "--seed",
        "42",
        "--embedder",
        "sbert",
        "--output-dir",
        str(output_root),
    ]

    tasks = [
        ("small_world", "hard"),
        ("scale_free", "hard"),
    ]

    for topology, persona in tasks:
        cmd = base + [
            "--topology",
            topology,
            "--persona-mode",
            persona,
        ]
        run(cmd)


if __name__ == "__main__":
    main()
