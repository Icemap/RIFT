"""Lightweight, dependency-free text embedding utilities for Concordia runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity with zero-division guards."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class HashingEmbedder:
    """Simple deterministic embedder used for memory + metrics.

    It avoids heavyweight model downloads (important for sandboxed runs) while
    still providing a stable geometric space for measuring drift and distance.
    """

    dim: int = 64
    salt: str = "rift_polarization"

    def __call__(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=float)

        vec = np.zeros(self.dim, dtype=float)
        for tok in tokens:
            h = hashlib.sha256(f"{self.salt}:{tok}".encode("utf-8")).hexdigest()
            # Spread each token across two positions for a touch of redundancy.
            first = int(h[:8], 16) % self.dim
            second = int(h[8:16], 16) % self.dim
            vec[first] += 1.0
            vec[second] += 0.5

        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec

    def batch(self, texts: Iterable[str]) -> list[np.ndarray]:
        return [self(t) for t in texts]


class SbertEmbedder:
    """Sentence-transformers based embedder for higher-fidelity semantics.

    This relies on the `sentence-transformers` package. If model download or
    loading fails in a constrained environment, callers should fall back to
    `HashingEmbedder`.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = dim

    def __call__(self, text: str) -> np.ndarray:
        clean = text.replace("\n", " ")
        emb = self._model.encode([clean], normalize_embeddings=True)[0]
        vec = np.asarray(emb, dtype=float)
        if self._dim is not None and self._dim > 0 and self._dim < vec.shape[0]:
            vec = vec[: self._dim]
        return vec


def get_embedder(kind: str = "hash", dim: int = 64):
    """Factory for text embedders used in RIFT.

    kind:
      - 'hash': lightweight hashing-based encoder (default).
      - 'sbert': Sentence-Transformers encoder, falls back to hash on failure.
    """
    kind = kind.lower()
    if kind == "sbert":
        try:
            return SbertEmbedder(dim=None)
        except Exception:
            # Fall back gracefully in offline / restricted environments.
            return HashingEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)
