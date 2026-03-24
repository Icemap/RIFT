"""Metric extraction for polarization experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import networkx as nx
import numpy as np

from rift.embeddings import HashingEmbedder, cosine_similarity


@dataclass
class ActionEvent:
    step: int
    actor: str
    text: str


@dataclass
class MetricResults:
    semantic_centroid_distance: float | None
    echo_chamber_score: float | None
    modularity: float | None
    in_group_exposure_ratio: float | None
    bimodality_coefficient: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "semantic_centroid_distance": self.semantic_centroid_distance,
            "echo_chamber_score": self.echo_chamber_score,
            "modularity": self.modularity,
            "in_group_exposure_ratio": self.in_group_exposure_ratio,
            "bimodality_coefficient": self.bimodality_coefficient,
        }


class PolarizationMetrics:
    """Aggregates agent actions into polarization metrics."""

    def __init__(
        self,
        graph: nx.Graph,
        group_map: Mapping[str, str],
        embedder: HashingEmbedder,
    ):
        self.graph = graph
        self.group_map = group_map
        self.embedder = embedder
        self.embeddings_by_agent: dict[str, list[np.ndarray]] = defaultdict(list)
        self.edge_messages: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    def ingest(self, actions: Iterable[ActionEvent]) -> None:
        for action in actions:
            emb = self.embedder(action.text)
            self.embeddings_by_agent[action.actor].append(emb)
            for neighbor in self.graph.neighbors(action.actor):
                self.edge_messages[(action.actor, neighbor)].append(emb)

    def _group_centroids(self) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for agent, embs in self.embeddings_by_agent.items():
            if agent not in self.group_map or not embs:
                continue
            grouped[self.group_map[agent]].extend(embs)

        centroids: dict[str, np.ndarray] = {}
        for group, embs in grouped.items():
            centroid = np.mean(np.stack(embs), axis=0)
            centroids[group] = centroid
        return centroids

    def _semantic_centroid_distance(self) -> float | None:
        centroids = self._group_centroids()
        if len(centroids) < 2:
            return None
        groups = list(centroids.keys())
        if np.linalg.norm(centroids[groups[0]]) == 0 or np.linalg.norm(
            centroids[groups[1]]
        ) == 0:
            return None
        return 1.0 - cosine_similarity(centroids[groups[0]], centroids[groups[1]])

    def _echo_chamber_score(self) -> float | None:
        centroids = self._group_centroids()
        if not centroids:
            return None

        within, cross = [], []
        for (src, dst), msgs in self.edge_messages.items():
            src_group = self.group_map.get(src)
            dst_group = self.group_map.get(dst)
            dst_proto = centroids.get(dst_group)
            if dst_proto is None:
                continue
            for msg in msgs:
                sim = cosine_similarity(msg, dst_proto)
                (within if src_group == dst_group else cross).append(sim)

        if not within or not cross:
            return None
        return float(np.mean(within) - np.mean(cross))

    def _modularity(self) -> float | None:
        if self.graph.number_of_edges() == 0:
            return None
        degree_sum = sum(d for _, d in self.graph.degree())
        if degree_sum == 0:
            return None

        communities: dict[str, set[str]] = defaultdict(set)
        for agent, group in self.group_map.items():
            if agent in self.graph:
                communities[group].add(agent)
        if len(communities) < 2:
            return None
        try:
            return float(
                nx.algorithms.community.quality.modularity(
                    self.graph, list(communities.values())
                )
            )
        except ZeroDivisionError:
            return None

    def _in_group_exposure_ratio(self) -> float | None:
        same, total = 0, 0
        for (src, dst), msgs in self.edge_messages.items():
            total += len(msgs)
            if self.group_map.get(src) == self.group_map.get(dst):
                same += len(msgs)
        if total == 0:
            return None
        return same / total

    def _bimodality(self) -> float | None:
        centroids = self._group_centroids()
        if len(centroids) != 2:
            return None

        groups = list(centroids.keys())
        diff = centroids[groups[0]] - centroids[groups[1]]
        if np.linalg.norm(diff) == 0:
            return None
        scores = []
        for embs in self.embeddings_by_agent.values():
            if not embs:
                continue
            mean_emb = np.mean(np.stack(embs), axis=0)
            scores.append(float(np.dot(mean_emb, diff)))
        if len(scores) < 3:
            return None

        arr = np.array(scores)
        mean = float(np.mean(arr))
        centered = arr - mean
        m2 = float(np.mean(centered**2))
        m3 = float(np.mean(centered**3))
        m4 = float(np.mean(centered**4))
        if m2 == 0:
            return None
        skewness = m3 / (m2 ** 1.5)
        kurtosis = m4 / (m2**2)
        return (skewness**2 + 1) / (kurtosis + 1e-9)

    def compute(self) -> MetricResults:
        return MetricResults(
            semantic_centroid_distance=self._semantic_centroid_distance(),
            echo_chamber_score=self._echo_chamber_score(),
            modularity=self._modularity(),
            in_group_exposure_ratio=self._in_group_exposure_ratio(),
            bimodality_coefficient=self._bimodality(),
        )


def extract_actions(raw_log: list[dict]) -> list[ActionEvent]:
    """Extract per-step agent actions from Concordia's raw log."""
    actions: list[ActionEvent] = []
    for entry in raw_log:
        step = int(entry.get("Step", 0))
        for key, value in entry.items():
            if not key.startswith("Entity ["):
                continue
            actor = key[len("Entity [") :].rstrip("]")
            action_text = (
                value.get("__act__", {}).get("Value")
                or value.get("__resolution__", {}).get("Value")
                or ""
            )
            cleaned = action_text.strip()
            if cleaned:
                actions.append(ActionEvent(step=step, actor=actor, text=cleaned))
    return actions
