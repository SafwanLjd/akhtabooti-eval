"""Gold-entity coverage per language and label, and the gaps against a minimum count."""

from __future__ import annotations

from collections.abc import Iterable

from ..dataset import LANGUAGES, Document
from ..taxonomy import LABELS

Coverage = dict[str, dict[str, int]]


def entity_coverage(documents: Iterable[Document], split: str | None = "test") -> Coverage:
    """Count gold spans by language and label, for one split or for all documents when ``split`` is None."""
    table: Coverage = {lang: dict.fromkeys(LABELS, 0) for lang in LANGUAGES}
    for doc in documents:
        if split is None or doc.split == split:
            for span in doc.spans:
                table[doc.language][span.label] += 1
    return table


def coverage_gaps(table: Coverage, minimum: int) -> list[str]:
    """Every (language, label) cell below ``minimum``, as "lang/LABEL: count < minimum"."""
    return [
        f"{lang}/{label}: {count} < {minimum}"
        for lang, row in table.items()
        for label, count in row.items()
        if count < minimum
    ]


def format_coverage(table: Coverage) -> str:
    """A Markdown table with one row per label and one column per language."""
    langs = list(table)
    lines = ["| label | " + " | ".join(langs) + " |", "|---|" + "---:|" * len(langs)]
    lines += [f"| {label} | " + " | ".join(str(table[lang][label]) for lang in langs) + " |" for label in LABELS]
    return "\n".join(lines)
