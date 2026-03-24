"""Recompute polarization metrics with SBERT for existing artifacts.

Usage (from repo root):
    UV_CACHE_DIR=.uv-cache uv run python scripts/recompute_metrics_sbert.py
    UV_CACHE_DIR=.uv-cache uv run python scripts/recompute_metrics_sbert.py --root artifacts_multiseed
    UV_CACHE_DIR=.uv-cache uv run python scripts/recompute_metrics_sbert.py --root artifacts_multimodel

Prereq: download SBERT once via `UV_CACHE_DIR=.uv-cache uv run python scripts/download_sbert.py`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

# When invoked as a script via `uv run python scripts/...`,
# the repo root may not be on sys.path. Add it explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rift.embeddings import get_embedder  # noqa: E402
from rift.metrics import PolarizationMetrics, extract_actions  # noqa: E402


def recompute_run(run_dir: Path, embedder_kind: str = "sbert") -> None:
    raw_path = run_dir / "raw_log.json"
    edges_path = run_dir / "graph_edges.json"
    metrics_path = run_dir / "metrics.json"
    if not raw_path.exists() or not edges_path.exists():
        print(f"[skip] Missing raw_log or graph_edges in {run_dir}")
        return

    raw = json.loads(raw_path.read_text())
    edges = json.loads(edges_path.read_text())

    g = nx.Graph()
    for e in edges:
        g.add_edge(e["source"], e["target"])
    group_map = {
        n: ("liberal" if n.startswith("L") else "conservative" if n.startswith("C") else "unknown")
        for n in g.nodes
    }

    embedder = get_embedder(embedder_kind, dim=96)
    events = extract_actions(raw)
    if not events:
        print(f"[warn] No actions found in {run_dir}, writing empty metrics.")
    eng = PolarizationMetrics(graph=g, group_map=group_map, embedder=embedder)
    eng.ingest(events)
    metrics = eng.compute().to_dict()
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[ok] Recomputed metrics with {embedder_kind}: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics with SBERT for artifacts.")
    parser.add_argument(
        "--root",
        default="artifacts",
        help="Root directory containing run subfolders (artifacts, artifacts_multiseed, artifacts_multimodel).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    for run_dir in sorted(root.glob("*")):
        if run_dir.is_dir():
            recompute_run(run_dir)


if __name__ == "__main__":
    main()
