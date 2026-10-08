"""Extractor contract. An extractor turns a rendered file into plain text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self


class ExtractionError(RuntimeError):
    """Raised when a file cannot be read; the runner records it per file."""


@dataclass(frozen=True, slots=True)
class Extraction:
    text: str
    seconds: float
    error: str | None = None


ExtractorResolver = Callable[[str], "Extractor"]


class Extractor(ABC):
    """Base class for text extractors. ``prepare`` loads models; ``__init__`` must stay cheap."""

    kind: ClassVar[str]

    @classmethod
    def from_config(cls, params: Mapping[str, Any], resolve: ExtractorResolver) -> Self:
        """Build from a config table. ``resolve`` returns other named extractors."""
        del resolve
        return cls(**params)

    def prepare(self) -> None:  # noqa: B027 - optional hook
        """Load models or start helper processes."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Stop helper processes and wait for them. Calling it twice is harmless; ``prepare`` starts them again."""

    @abstractmethod
    def info(self) -> dict[str, Any]: ...

    @abstractmethod
    def extract(self, path: Path) -> str:
        """Return the text of ``path`` or raise ExtractionError."""
