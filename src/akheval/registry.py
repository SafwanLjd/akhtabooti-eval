"""Component kinds and the factory that builds named detectors and extractors from config tables.

Each kind maps to a "module:Class" target that is imported on first use, so
torch, onnxruntime and extractous load only when a run needs them.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .detectors.base import Detector
from .extractors.base import Extractor

DETECTOR_KINDS: dict[str, str] = {
    "core-rules": "akheval.detectors.core_rules:CoreRulesDetector",
    "regex-baseline": "akheval.detectors.regex_baseline:RegexBaselineDetector",
    "hybrid": "akheval.detectors.hybrid:HybridDetector",
    "nym": "akheval.detectors.nym:NymDetector",
    "hf-token": "akheval.detectors.hf_token:HFTokenDetector",
    "dialogpii": "akheval.detectors.dialogpii:DialogPIIDetector",
}

EXTRACTOR_KINDS: dict[str, str] = {
    "extractous": "akheval.extractors.extractous:ExtractousExtractor",
    "tesseract": "akheval.extractors.tesseract:TesseractExtractor",
    "ppocr": "akheval.extractors.ppocr:PaddleOcrExtractor",
    "pdf-ocr": "akheval.extractors.pdf_ocr:PdfOcrFallbackExtractor",
}


class ConfigError(ValueError):
    """An evaluation config that cannot be used as written."""


@dataclass(frozen=True)
class Component:
    """One named ``[detectors.<name>]`` or ``[extractors.<name>]`` table."""

    name: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)


def load_class(target: str) -> type[Any]:
    """Import ``module:Class`` and return the class."""
    module_name, _, class_name = target.partition(":")
    if not class_name:
        raise ConfigError(f"component target {target!r} must look like 'package.module:Class'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"cannot import {module_name} for {target!r}: {exc}") from exc
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        raise ConfigError(f"{module_name} has no class {class_name}")
    return cls


T = TypeVar("T", Detector, Extractor)


class Factory(Generic[T]):
    """Builds components by name once each, resolving references between them and rejecting cycles."""

    def __init__(self, specs: Mapping[str, Component], kinds: Mapping[str, str], base: type[Any], what: str) -> None:
        self._specs = specs
        self._kinds = kinds
        self._base = base
        self._what = what
        self._built: dict[str, T] = {}
        self._building: list[str] = []

    def build(self, name: str) -> T:
        if name in self._built:
            return self._built[name]
        if name in self._building:
            chain = " -> ".join([*self._building, name])
            raise ConfigError(f"{self._what} references form a cycle: {chain}")
        spec = self._specs.get(name)
        if spec is None:
            raise ConfigError(f"unknown {self._what} {name!r}; defined: {', '.join(sorted(self._specs)) or 'none'}")
        target = self._kinds.get(spec.kind)
        if target is None:
            raise ConfigError(f"{self._what} {name!r} has unknown kind {spec.kind!r}")
        cls = load_class(target)
        if not issubclass(cls, self._base):
            raise ConfigError(f"{target} is not a {self._base.__name__}")
        self._building.append(name)
        try:
            instance: T = cls.from_config(dict(spec.params), self.build)
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{self._what} {name!r}: bad parameters for {target}: {exc}") from exc
        finally:
            self._building.pop()
        self._built[name] = instance
        return instance


def detector_factory(specs: Mapping[str, Component]) -> Factory[Detector]:
    return Factory(specs, DETECTOR_KINDS, Detector, "detector")


def extractor_factory(specs: Mapping[str, Component]) -> Factory[Extractor]:
    return Factory(specs, EXTRACTOR_KINDS, Extractor, "extractor")
