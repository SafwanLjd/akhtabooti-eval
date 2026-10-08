"""Dataset build configuration, read from a TOML file such as configs/dataset-v1.toml."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..dataset import SEVERITIES
from .doctypes import DOC_TYPES


@dataclass(frozen=True, slots=True)
class ImageShares:
    """Shares of documents that get each kind of image rendering, with their formats and severity mix."""

    card_scan: float
    card_scan_formats: tuple[str, ...]
    page_image: float
    page_photo: float
    page_scan_formats: tuple[str, ...]
    chat_screenshot: float
    severity: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SynthConfig:
    name: str
    version: str
    seed: int
    dev_fraction: float
    min_test_entities: int
    notes: str
    documents: Mapping[str, int]
    """Documents per language for each document type."""
    images: ImageShares
    verify_native: bool
    soffice: str
    soffice_timeout: int
    raw: Mapping[str, Any]


def load_config(path: Path) -> SynthConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: Mapping[str, Any]) -> SynthConfig:
    dataset, images, native = raw["dataset"], raw["images"], raw.get("native", {})
    documents = {str(k): int(v) for k, v in raw["documents"].items()}
    unknown = sorted(set(documents) - set(DOC_TYPES))
    if unknown:
        raise ValueError(f"unknown document types in config: {unknown}; known: {sorted(DOC_TYPES)}")
    severity = {str(k): float(v) for k, v in images["severity"].items()}
    if sorted(severity) != sorted(SEVERITIES):
        raise ValueError(f"images.severity needs weights for exactly {SEVERITIES}")
    return SynthConfig(
        name=str(dataset["name"]),
        version=str(dataset["version"]),
        seed=int(dataset["seed"]),
        dev_fraction=float(dataset["dev_fraction"]),
        min_test_entities=int(dataset.get("min_test_entities", 0)),
        notes=str(dataset.get("notes", "")),
        documents=documents,
        images=ImageShares(
            card_scan=float(images["card_scan_share"]),
            card_scan_formats=tuple(images["card_scan_formats"]),
            page_image=float(images["page_image_share"]),
            page_photo=float(images["page_photo_share"]),
            page_scan_formats=tuple(images["page_scan_formats"]),
            chat_screenshot=float(images["chat_screenshot_share"]),
            severity=severity,
        ),
        verify_native=bool(native.get("verify", True)),
        soffice=str(native.get("soffice", "soffice")),
        soffice_timeout=int(native.get("timeout_seconds", 900)),
        raw=raw,
    )
