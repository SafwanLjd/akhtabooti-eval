"""Hugging Face token-classification pipelines, such as CAMeLBERT Mix NER.

The module also holds the provenance helpers that the Nym and DialogPII
detectors share: the repository and commit recorded for downloaded files and
the thread count used for timing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..dataset import sha256_file
from ..fetch import FETCHED
from ..taxonomy import CAMELBERT_LABELS, LabelMap
from .base import Detection, Detector
from .chunking import Entity, chunk_text, merge_entities, to_detection

LABEL_TABLES: dict[str, LabelMap] = {"camelbert": CAMELBERT_LABELS}
HUB_METADATA = Path(".cache") / "huggingface" / "download"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
OVERLAP_TOKENS = 32
AGGREGATION = "simple"


def hub_revision(model_dir: Path, filename: str) -> str | None:
    """Commit of ``filename`` as recorded by ``huggingface_hub`` when it downloaded into ``model_dir``."""
    metadata = model_dir / HUB_METADATA / f"{filename}.metadata"
    if not metadata.is_file():
        return None
    return metadata.read_text(encoding="utf-8").partition("\n")[0].strip() or None


def _fetched(model_dir: Path) -> dict[str, Any]:
    marker = model_dir / FETCHED
    return json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}


def fetched_repo(model_dir: Path) -> str | None:
    """Repository named in the ``FETCHED.json`` that ``akheval fetch`` writes into ``model_dir``."""
    repo = _fetched(model_dir).get("repo")
    return str(repo) if repo else None


def fetched_files(model_dir: Path) -> dict[str, str] | None:
    """sha256 per file that ``akheval fetch`` downloaded into ``model_dir``, from its ``FETCHED.json``."""
    return _fetched(model_dir).get("files")


def file_record(path: Path) -> dict[str, Any]:
    """Path, size and sha256 of a model file; size and digest are ``None`` when the file is missing."""
    if not path.is_file():
        return {"path": str(path), "bytes": None, "sha256": None}
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def inference_threads() -> int:
    """Half the physical cores this process may use, at least one.

    The Nym, transformers and DialogPII detectors share this count, so their
    timings compare and the machine keeps cores free for the harness. In a
    container the CPU affinity mask lists the usable logical CPUs; the
    host's ratio of physical to logical cores converts that to cores.
    """
    import psutil

    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else logical
    return max(1, usable * physical // logical // 2)


def require_dir(path: Path) -> None:
    """Raise if a model directory has not been fetched."""
    if not path.is_dir():
        raise FileNotFoundError(f"model directory {path} is missing; fetch the models listed in configs/models first")


class HFTokenDetector(Detector):
    """A transformers token-classification pipeline on the CPU.

    ``model`` is a local snapshot directory. ``labels`` names the label table,
    for example ``camelbert``. The detector cuts text into chunks of at most
    ``max_length`` tokens, special tokens included.
    """

    kind = "hf-token"

    def __init__(self, model: str, labels: str, threshold: float = 0.5, max_length: int = 512) -> None:
        if labels not in LABEL_TABLES:
            raise ValueError(f"unknown label table {labels!r}; expected one of {sorted(LABEL_TABLES)}")
        self.model_dir = Path(model)
        self.labels = labels
        self.table = LABEL_TABLES[labels]
        self.threshold = threshold
        self.max_length = max_length
        self._pipe: Any = None

    def prepare(self) -> None:
        import torch
        from transformers import pipeline

        require_dir(self.model_dir)
        torch.set_num_threads(inference_threads())
        self._pipe = pipeline(
            "token-classification",
            model=str(self.model_dir),
            aggregation_strategy=AGGREGATION,
            device="cpu",
        )

    def info(self) -> dict[str, Any]:
        weights = next((self.model_dir / n for n in WEIGHT_FILES if (self.model_dir / n).is_file()), None)
        return {
            "kind": self.kind,
            "repo": fetched_repo(self.model_dir),
            "revision": hub_revision(self.model_dir, weights.name) if weights else None,
            "weights": file_record(weights) if weights else None,
            "fetched_files": fetched_files(self.model_dir),
            "labels": self.labels,
            "threshold": self.threshold,
            "max_length": self.max_length,
            "overlap_tokens": OVERLAP_TOKENS,
            "aggregation": AGGREGATION,
            "threads": inference_threads(),
        }

    def detect(self, text: str) -> Detection:
        if self._pipe is None:
            self.prepare()
        budget = self.max_length - self._pipe.tokenizer.num_special_tokens_to_add()
        entities: list[Entity] = []
        for chunk in chunk_text(text, budget, min(OVERLAP_TOKENS, budget // 2), self._count_tokens):
            if not chunk.text.strip():
                continue
            entities += [_pipeline_entity(found, chunk.start) for found in self._pipe(chunk.text)]
        return to_detection(text, merge_entities(entities), self.table).with_threshold(self.threshold)

    def _count_tokens(self, text: str) -> int:
        return len(self._pipe.tokenizer(text, add_special_tokens=False)["input_ids"])


def _pipeline_entity(found: Mapping[str, Any], offset: int) -> Entity:
    """Entity from one aggregated pipeline result, shifted by ``offset``."""
    start, end = int(found["start"]) + offset, int(found["end"]) + offset
    return Entity(start, end, str(found["entity_group"]), float(found["score"]))
