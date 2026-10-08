"""Stratified dev/test split that does not depend on generation order."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence


def _rank_key(seed: int, doc_id: str) -> str:
    return hashlib.sha256(f"{seed}:split:{doc_id}".encode()).hexdigest()


def assign_splits(strata: Iterable[Sequence[str]], *, seed: int, dev_fraction: float) -> dict[str, str]:
    """Send ``round(n * dev_fraction)`` documents of every stratum to dev and the rest to test.

    Documents are ranked by a hash of the seed and their id, so the split of a document
    changes only when its stratum's size crosses a rounding step.
    """
    splits: dict[str, str] = {}
    for doc_ids in strata:
        ranked = sorted(doc_ids, key=lambda doc_id: _rank_key(seed, doc_id))
        n_dev = round(len(ranked) * dev_fraction)
        for position, doc_id in enumerate(ranked):
            splits[doc_id] = "dev" if position < n_dev else "test"
    return splits
