"""Compute additional polarization metrics (ECS, centroid gap) from existing artifacts.

No LLM calls are made; this only reads raw_log.json and graph_edges.json and writes
extra_metrics.json into each run directory.

Usage examples (from repo root):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HUGGINGFACE_HUB_CACHE=~/.cache/huggingface/hub \\
        UV_CACHE_DIR=.uv-cache PYTHONPATH=. uv run python scripts/compute_extra_metrics.py --root artifacts
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HUGGINGFACE_HUB_CACHE=~/.cache/huggingface/hub \\
        UV_CACHE_DIR=.uv-cache PYTHONPATH=. uv run python scripts/compute_extra_metrics.py --root artifacts_multiseed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.embeddings import get_embedder, cosine_similarity
from rift.metrics import extract_actions


def _extract_goals(raw_log: list[dict]) -> dict[str, str]:
    goals: dict[str, str] = {}
    for entry in raw_log:
        for key, value in entry.items():
            if not key.startswith("Entity [") or not isinstance(value, dict):
                continue
            actor = key[len("Entity [") :].rstrip("]")
            goal_text = value.get("Goal", {}).get("Value")
            if isinstance(goal_text, str) and goal_text.strip() and actor not in goals:
                goals[actor] = goal_text.strip()
        if goals and len(goals) >= 50:
            break
    return goals


def _structural_metrics(g: nx.Graph, group_map: dict[str, str]) -> dict[str, float | None]:
    # Standard network polarization metrics often reported alongside modularity:
    # - assortativity by group label (Newman 2003/2006)
    # - E-I index (Krackhardt & Stern 1988)
    if g.number_of_edges() == 0:
        return {"assortativity": None, "ei_index": None}
    nx.set_node_attributes(g, group_map, "group")
    try:
        assort = float(nx.attribute_assortativity_coefficient(g, "group"))
    except Exception:
        assort = None
    external = 0
    internal = 0
    for u, v in g.edges():
        if group_map.get(u) == group_map.get(v):
            internal += 1
        else:
            external += 1
    denom = external + internal
    ei = (external - internal) / denom if denom else None
    return {"assortativity": assort, "ei_index": ei}


def compute_extra(run_dir: Path, embedder_kind: str = "sbert") -> None:
    raw_path = run_dir / "raw_log.json"
    edges_path = run_dir / "graph_edges.json"
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

    events = extract_actions(raw)
    embedder = get_embedder(embedder_kind, dim=96)
    goals = _extract_goals(raw)

    embs_by_group: dict[str, list[np.ndarray]] = {"liberal": [], "conservative": []}
    for ev in events:
        emb = embedder(ev.text)
        embs_by_group.get(group_map.get(ev.actor, "unknown"), []).append(emb)

    centroids = {
        g: np.mean(np.stack(v), axis=0) for g, v in embs_by_group.items() if v
    }

    ecs_vals = []
    for (u, v) in g.edges():
        for ev in events:
            if ev.actor == u:
                emb = embedder(ev.text)
                tgt = centroids.get(group_map.get(v))
                if tgt is not None:
                    ecs_vals.append(cosine_similarity(emb, tgt))
            if ev.actor == v:
                emb = embedder(ev.text)
                tgt = centroids.get(group_map.get(u))
                if tgt is not None:
                    ecs_vals.append(cosine_similarity(emb, tgt))

    ecs = float(np.mean(ecs_vals)) if ecs_vals else None
    centroid_gap = None
    if "liberal" in centroids and "conservative" in centroids:
        centroid_gap = float(1 - cosine_similarity(centroids["liberal"], centroids["conservative"]))

    # Persona adherence diagnostic: similarity between each utterance and the
    # actor's persona goal prompt (goal text is stored in raw logs).
    goal_sims_by_actor: dict[str, list[float]] = {}
    goal_embs = {actor: embedder(text) for actor, text in goals.items()}
    for ev in events:
        gemb = goal_embs.get(ev.actor)
        if gemb is None:
            continue
        sim = float(cosine_similarity(embedder(ev.text), gemb))
        goal_sims_by_actor.setdefault(ev.actor, []).append(sim)
    goal_sims = [s for vals in goal_sims_by_actor.values() for s in vals]
    persona_adherence_mean = float(np.mean(goal_sims)) if goal_sims else None
    persona_adherence_by_actor = {
        actor: float(np.mean(vals)) for actor, vals in goal_sims_by_actor.items() if vals
    }

    structural = _structural_metrics(g, group_map)

    out = {
        "ecs_mean_similarity": ecs,
        "centroid_gap": centroid_gap,
        "assortativity": structural["assortativity"],
        "ei_index": structural["ei_index"],
        "persona_adherence_mean": persona_adherence_mean,
        "persona_adherence_by_actor": persona_adherence_by_actor,
        "embedder": embedder_kind,
    }
    out_path = run_dir / "extra_metrics.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[ok] Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ECS-style extras from artifacts.")
    parser.add_argument("--root", default="artifacts", help="Root directory containing run subfolders.")
    parser.add_argument("--embedder", default="sbert", help="Embedder kind (sbert/hash).")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    for run_dir in sorted(root.glob("*")):
        if run_dir.is_dir():
            compute_extra(run_dir, embedder_kind=args.embedder)


if __name__ == "__main__":
    main()
