import random
from collections import Counter

from akheval.dataset import Document, Span
from akheval.synth.coverage import coverage_gaps, entity_coverage, format_coverage
from akheval.synth.split import assign_splits
from akheval.taxonomy import LABELS


def test_each_stratum_sends_thirty_percent_to_dev() -> None:
    strata = [[f"a-{i}" for i in range(10)], [f"b-{i}" for i in range(14)], [f"c-{i}" for i in range(1)]]
    splits = assign_splits(strata, seed=1, dev_fraction=0.3)
    assert Counter(splits[i] for i in strata[0]) == {"dev": 3, "test": 7}
    assert Counter(splits[i] for i in strata[1]) == {"dev": 4, "test": 10}
    assert splits["c-0"] == "test"


def test_split_ignores_input_order() -> None:
    ids = [f"doc-{i}" for i in range(20)]
    shuffled = ids[:]
    random.Random(3).shuffle(shuffled)
    assert assign_splits([ids], seed=5, dev_fraction=0.3) == assign_splits([shuffled], seed=5, dev_fraction=0.3)


def test_split_depends_on_seed() -> None:
    ids = [f"doc-{i}" for i in range(20)]
    assert assign_splits([ids], seed=1, dev_fraction=0.3) != assign_splits([ids], seed=2, dev_fraction=0.3)


def _doc(doc_id: str, split: str, language: str, labels: list[str]) -> Document:
    spans = tuple(Span(i, i + 1, label, "x") for i, label in enumerate(labels))
    return Document(id=doc_id, split=split, language=language, doc_type="letter", text="x" * len(labels), spans=spans)


def test_coverage_counts_one_split_and_lists_gaps() -> None:
    docs = [
        _doc("a", "test", "ar", ["EMAIL", "EMAIL", "PHONE"]),
        _doc("b", "dev", "ar", ["EMAIL"]),
        _doc("c", "test", "mixed", ["PERSON"]),
    ]
    table = entity_coverage(docs, "test")
    assert table["ar"]["EMAIL"] == 2
    assert table["ar"]["PHONE"] == 1
    assert table["mixed"]["PERSON"] == 1
    assert entity_coverage(docs, None)["ar"]["EMAIL"] == 3
    gaps = coverage_gaps(table, 2)
    assert "ar/EMAIL: 2 < 2" not in gaps
    assert "ar/PHONE: 1 < 2" in gaps
    assert len(gaps) == 3 * len(LABELS) - 1


def test_format_coverage_has_a_row_per_label() -> None:
    lines = format_coverage(entity_coverage([], "test")).splitlines()
    assert lines[0] == "| label | ar | en | mixed |"
    assert len(lines) == 2 + len(LABELS)
