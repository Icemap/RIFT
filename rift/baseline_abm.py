"""Classical opinion-dynamics baselines for comparison with RIFT.

We implement a small Deffuant--Weisbuch model on the same fixed networks used
for LLM-agent simulations.  This allows us to contextualise what RIFT adds
relative to scalar opinion dynamics on identical topologies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import networkx as nx
import numpy as np


@dataclass
class DeffuantConfig:
    steps: int = 12
    mu: float = 0.3
    epsilon: float = 0.5
    seed: int = 0


@dataclass
class DeffuantRun:
    """Container for per-step scalar opinions and simple metrics."""

    opinions_by_step: list[Dict[str, float]]
    in_group_interaction_ratio: list[float]


def initialise_opinions(
    nodes: Iterable[str],
    group_map: Mapping[str, str],
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Initialise scalar opinions in [-1, 1] with group-specific means.

    Liberal agents start near -0.5, conservative agents near +0.5, with
    small Gaussian noise.  This mirrors the initial bias baked into our
    personas while keeping values in a standardised range.
    """
    rng = np.random.default_rng(seed)
    opinions: Dict[str, float] = {}
    for node in nodes:
        group = group_map.get(node, "neutral")
        if group == "liberal":
            base = -0.5
        elif group == "conservative":
            base = 0.5
        else:
            base = 0.0
        val = base + rng.normal(scale=0.1)
        opinions[node] = float(np.clip(val, -1.0, 1.0))
    return opinions


def run_deffuant(
    graph: nx.Graph,
    group_map: Mapping[str, str],
    cfg: DeffuantConfig,
) -> DeffuantRun:
    """Simulate Deffuant dynamics on a fixed graph."""
    rng = np.random.default_rng(cfg.seed)
    opinions = initialise_opinions(graph.nodes(), group_map, seed=cfg.seed)
    edges = list(graph.edges())
    if not edges:
        raise ValueError("Graph has no edges; Deffuant dynamics are undefined.")

    opinions_by_step: list[Dict[str, float]] = [dict(opinions)]
    in_group_interaction_ratio: list[float] = []

    for _step in range(cfg.steps):
        same, total = 0, 0
        # One sweep over |E| randomly chosen interactions.
        for _ in range(len(edges)):
            u, v = edges[rng.integers(len(edges))]
            x, y = opinions[u], opinions[v]
            if abs(x - y) <= cfg.epsilon:
                delta = cfg.mu * (y - x)
                opinions[u] = float(np.clip(x + delta, -1.0, 1.0))
                opinions[v] = float(np.clip(y - delta, -1.0, 1.0))
                total += 1
                if group_map.get(u) == group_map.get(v):
                    same += 1
        opinions_by_step.append(dict(opinions))
        in_group_interaction_ratio.append(same / total if total > 0 else 0.0)

    return DeffuantRun(
        opinions_by_step=opinions_by_step,
        in_group_interaction_ratio=in_group_interaction_ratio,
    )


def scalar_centroid_distance(
    opinions: Mapping[str, float],
    group_map: Mapping[str, str],
) -> float | None:
    """Difference between liberal and conservative mean opinions."""
    by_group: dict[str, list[float]] = {"liberal": [], "conservative": []}
    for agent, val in opinions.items():
        g = group_map.get(agent)
        if g in by_group:
            by_group[g].append(val)
    if not by_group["liberal"] or not by_group["conservative"]:
        return None
    mu_l = float(np.mean(by_group["liberal"]))
    mu_c = float(np.mean(by_group["conservative"]))
    return abs(mu_l - mu_c)


def bimodality_coefficient_scalar(values: Iterable[float]) -> float | None:
    """Bimodality coefficient for a univariate distribution.

    Uses the Pfister et al.\ (2013) definition: BC = (gamma^2 + 1) / kappa,
    where gamma is skewness and kappa is kurtosis.  Returns None when the
    variance is zero or the sample is too small.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 3:
        return None
    mean = float(np.mean(arr))
    centered = arr - mean
    m2 = float(np.mean(centered**2))
    if m2 == 0:
        return None
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skewness = m3 / (m2 ** 1.5)
    kurtosis = m4 / (m2**2)
    return (skewness**2 + 1.0) / (kurtosis + 1e-9)

