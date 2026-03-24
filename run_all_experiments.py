"""
Run a batch of polarization simulations for paper-ready comparisons.

This script launches multiple configurations (hard/soft personas, varying
homophily and recommender) and stores outputs in distinct subdirectories under
`artifacts/`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


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
    ]

    experiments = [
        (
            "hard-085-homo",
            base
            + [
                "--persona-mode",
                "hard",
                "--homophily",
                "0.85",
                "--recommender",
                "homophily",
            ],
        ),
        (
            "soft-085-homo",
            base
            + [
                "--persona-mode",
                "soft",
                "--homophily",
                "0.85",
                "--recommender",
                "homophily",
            ],
        ),
        (
            "hard-050-homo",
            base
            + [
                "--persona-mode",
                "hard",
                "--homophily",
                "0.5",
                "--recommender",
                "homophily",
            ],
        ),
        (
            "soft-050-homo",
            base
            + [
                "--persona-mode",
                "soft",
                "--homophily",
                "0.5",
                "--recommender",
                "homophily",
            ],
        ),
        (
            "hard-085-randomrec",
            base
            + [
                "--persona-mode",
                "hard",
                "--homophily",
                "0.85",
                "--recommender",
                "random",
            ],
        ),
    ]

    for name, cmd in experiments:
        # Derive output dir name the same way main.py does.
        if "--steps" in cmd:
            steps = cmd[cmd.index("--steps") + 1]
        else:
            steps = "12"
        output_dir = Path("artifacts") / f"persona-{cmd[cmd.index('--persona-mode')+1]}" \
            f"_model-{cmd[cmd.index('--model')+1]}" \
            f"_lm-{cmd[cmd.index('--model-name')+1]}" \
            f"_steps-{steps}" \
            f"_aps-{cmd[cmd.index('--agents-per-side')+1]}" \
            f"_topo-{cmd[cmd.index('--topology')+1]}" \
            f"_hom-{cmd[cmd.index('--homophily')+1]}" \
            f"_rec-{cmd[cmd.index('--recommender')+1]}" \
            f"_seed-42"

        if output_dir.exists():
            print(f"Skip {name}: {output_dir} already exists.")
            continue
        run(cmd)


if __name__ == "__main__":
    main()
