from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rift.embeddings import HashingEmbedder, get_embedder
from rift.llm import ModelConfig, MissingAPIKeyError, build_model
from rift.simulation_runner import ExperimentParams, run_experiment


def _load_env() -> None:
    load_dotenv()
    # Concordia expects GOOGLE_API_KEY for Gemini; mirror from GEMINI_API_KEY.
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = gemini_key


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-driven polarization simulation using Concordia."
    )
    parser.add_argument("--steps", type=int, default=6, help="Max simulation steps.")
    parser.add_argument(
        "--topology",
        choices=["small_world", "scale_free"],
        default="small_world",
        help="Network topology used for structural metrics.",
    )
    parser.add_argument(
        "--homophily",
        type=float,
        default=0.6,
        help="Probability of rewiring cross-group edges toward in-group.",
    )
    parser.add_argument(
        "--recommender",
        choices=["homophily", "random"],
        default="homophily",
        help="Narrative guidance for the game master.",
    )
    parser.add_argument(
        "--agents-per-side",
        type=int,
        default=3,
        help="Number of liberal and conservative personas each.",
    )
    parser.add_argument(
        "--model",
        choices=["openai", "gemini", "openrouter", "stub", "stub-random", "offline-null"],
        default="openai",
        help="Language model backend. Use stub for offline smoke tests.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-mini",
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "--service-tier",
        choices=["auto", "default", "flex", "scale", "priority"],
        default=None,
        help="OpenAI service tier (e.g., flex). Only applies when --model=openai.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for persona and network sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base directory for logs/metrics (subfolder auto-generated per run).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=96,
        help="Dimensionality of the lightweight hashing embedder.",
    )
    parser.add_argument(
        "--embedder",
        choices=["hash", "sbert"],
        default="hash",
        help="Text embedding backend for metrics and memory.",
    )
    parser.add_argument(
        "--persona-mode",
        choices=["hard", "soft"],
        default="hard",
        help="Use 'hard' for identity-protective prompts, 'soft' for a conciliatory baseline.",
    )
    return parser.parse_args()


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def build_run_name(args: argparse.Namespace) -> str:
    """Deterministic run-name builder shared across scripts.

    Keeping this logic in one place avoids inconsistencies between the CLI and
    batch runners when deciding where to store artifacts.
    """
    def _clean(text: str) -> str:
        return text.replace("/", "-").replace(":", "-").replace(" ", "")

    parts = [
        f"persona-{_clean(args.persona_mode)}",
        f"model-{_clean(args.model)}",
        f"lm-{_clean(args.model_name)}",
        *( [f"tier-{_clean(args.service_tier)}"] if getattr(args, "service_tier", None) else [] ),
        f"steps-{args.steps}",
        f"aps-{args.agents_per_side}",
        f"topo-{args.topology}",
        f"hom-{args.homophily}",
        f"rec-{args.recommender}",
        f"seed-{args.seed}",
    ]
    return "_".join(parts)


def _make_run_dir(args: argparse.Namespace) -> Path:
    run_name = build_run_name(args)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    _load_env()
    args = _parse_args()

    embedder = get_embedder(args.embedder, dim=args.embed_dim)
    exp_params = ExperimentParams(
        steps=args.steps,
        topology=args.topology,
        homophily=args.homophily,
        recommender=args.recommender,
        seed=args.seed,
        agents_per_side=args.agents_per_side,
        persona_mode=args.persona_mode,
    )
    model_cfg = ModelConfig(
        provider=args.model,
        model_name=args.model_name,
        service_tier=args.service_tier,
    )

    try:
        model = build_model(model_cfg)
    except MissingAPIKeyError as exc:
        raise SystemExit(
            "API key missing. Set OPENAI_API_KEY (or GEMINI_API_KEY) in .env, "
            "or use --model stub for offline smoke tests."
        ) from exc

    run_dir = _make_run_dir(args)
    outputs = run_experiment(params=exp_params, model=model, embedder=embedder)

    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "raw_log.json"
    actions_path = run_dir / "actions.json"
    graph_path = run_dir / "graph_edges.json"

    _save_json(metrics_path, outputs.metrics.to_dict())
    _save_json(log_path, outputs.raw_log)
    _save_json(
        actions_path,
        [action.__dict__ for action in outputs.actions],
    )
    _save_json(
        graph_path,
        [
            {"source": u, "target": v}
            for u, v in outputs.graph.edges()
        ],
    )

    print("=== Polarization metrics (best-effort) ===")
    for k, v in outputs.metrics.to_dict().items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nArtifacts written to", run_dir.resolve())
    print("Agents:", ", ".join(p.name for p in outputs.personas))


if __name__ == "__main__":
    main()
