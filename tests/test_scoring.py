from __future__ import annotations

import pytest

from akheval.config import EXTRACT, PIPELINE, TEXT
from akheval.dataset import Document, Span
from akheval.detectors.base import Detection
from akheval.matching import ANY, LENIENT, STRICT, VALUE_LENIENT
from akheval.metrics import FN, FP, TP, Counts
from akheval.scoring import (
    BOUNDARY,
    DETECTION_FAILED,
    DOC_FALSE_NEGATIVE,
    DOC_FALSE_POSITIVE,
    DOCUMENT,
    EXTRACTION_FAILED,
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    NO_LOOK_ALIKE,
    VALUE,
    VALUE_MISSED,
    VALUE_NEAR,
    VALUE_PARTIAL,
    VALUE_SPURIOUS,
    VALUE_STRICT,
    ExtractionQuality,
    ItemScore,
    Prediction,
    context_window,
    document_count,
    extraction_quality,
    grouped,
    quality_grouped,
    score_predictions,
    score_rendering,
    score_text,
    series,
    summarize,
    words,
)
from harness_fixtures import CARD_BACK, CARD_FRONT, documents, span, two_face_card


def _doc(doc_id: str) -> Document:
    return next(d for d in documents() if d.id == doc_id)


def _text_prediction(doc: Document) -> Prediction:
    text = doc.text
    detection = Detection(
        spans=[
            span(text, "john.smith@example.org", "EMAIL"),
            span(text, "791234567", "PHONE"),
            span(text, "INV-2024-001", "NATIONAL_ID"),
        ]
    )
    return Prediction(item_id=doc.id, doc_id=doc.id, detection=detection, detect_seconds=0.01)


def test_text_item_counts_every_level() -> None:
    doc = _doc("doc-en-1")
    item = score_text(doc, _text_prediction(doc))
    assert item.levels[LENIENT] == {"EMAIL": TP, "PHONE": TP, "PERSON": FN, "NATIONAL_ID": FP}
    assert item.levels[STRICT] == {"EMAIL": TP, "PHONE": Counts(0, 1, 1), "PERSON": FN, "NATIONAL_ID": FP}
    assert item.levels[VALUE] == {"EMAIL": TP, "PHONE": TP, "PERSON": FN, "NATIONAL_ID": FP}
    assert item.levels[VALUE_LENIENT] == item.levels[VALUE]
    assert item.levels[VALUE_STRICT]["PHONE"] == Counts(0, 1, 1)
    assert item.levels[DOCUMENT] == {"EMAIL": TP, "PHONE": TP, "PERSON": FN, "NATIONAL_ID": FP, ANY: TP}
    assert item.facets == {"language": "en", "doc_type": "letter"}
    assert item.chars == len(doc.text)
    assert item.seconds == pytest.approx(0.01)


def test_text_item_errors_are_span_and_document_errors() -> None:
    doc = _doc("doc-en-1")
    errors = {(e.kind, e.label): e for e in score_text(doc, _text_prediction(doc)).errors}
    assert set(errors) == {
        (BOUNDARY, "PHONE"),
        (FALSE_NEGATIVE, "PERSON"),
        (FALSE_POSITIVE, "NATIONAL_ID"),
        (DOC_FALSE_POSITIVE, "NATIONAL_ID"),
        (DOC_FALSE_NEGATIVE, "PERSON"),
    }
    false_positive = errors[(FALSE_POSITIVE, "NATIONAL_ID")]
    assert false_positive.negative_kind == "invoice_number"
    assert "[[INV-2024-001]]" in false_positive.context
    assert errors[(BOUNDARY, "PHONE")].gold_text == "0791234567"


def test_failed_detection_scores_as_nothing_found() -> None:
    doc = _doc("doc-en-1")
    item = score_text(doc, Prediction(item_id=doc.id, doc_id=doc.id, detection_error="RuntimeError: boom"))
    assert item.levels[LENIENT] == {"PERSON": FN, "EMAIL": FN, "PHONE": FN}
    assert DETECTION_FAILED in {e.kind for e in item.errors}


def test_rendering_item_scores_values_documents_and_quality() -> None:
    doc = _doc("doc-en-1")
    rendering = doc.renderings[1]
    extracted = doc.text.replace("0791234567", "079I234567")
    detection = Detection(
        values=[("EMAIL", "john.smith@example.org"), ("PHONE", "079I234567"), ("IBAN", "INV-2024-001")]
    )
    prediction = Prediction(
        item_id=rendering.id,
        doc_id=doc.id,
        rendering_id=rendering.id,
        detection=detection,
        extracted=extracted,
        extract_seconds=0.5,
        detect_seconds=0.1,
    )
    item = score_rendering(doc, rendering, prediction, detect=True)
    assert item.levels[VALUE] == {"EMAIL": TP, "PHONE": TP, "PERSON": FN, "IBAN": FP}
    assert item.levels[VALUE_STRICT]["PHONE"] == Counts(0, 1, 1)
    assert next(e for e in item.errors if e.kind == VALUE_NEAR).gold_text == "0791234567"
    assert item.levels[DOCUMENT][ANY] == TP
    assert STRICT not in item.levels
    assert item.facets["capture"] == "scan"
    assert item.facets["severity"] == "mild"
    reference = words(doc.text)
    ref_chars = len(" ".join(reference))
    assert item.quality == ExtractionQuality(1, ref_chars, 1, len(reference), 1, len(reference) - 1, 4, 3)
    assert item.seconds == pytest.approx(0.6)
    spurious = next(e for e in item.errors if e.kind == VALUE_SPURIOUS)
    assert spurious.negative_kind == "invoice_number"
    assert "[[INV-2024-001]]" in spurious.context


def test_extract_only_rendering_has_quality_and_no_levels() -> None:
    doc = _doc("doc-ar-1")
    rendering = doc.renderings[1]
    prediction = Prediction(
        item_id=rendering.id, doc_id=doc.id, rendering_id=rendering.id, extracted="", extraction_error="unreadable"
    )
    item = score_rendering(doc, rendering, prediction, detect=False)
    assert item.levels == {}
    assert item.quality is not None
    assert item.quality.cer == 1.0
    assert item.quality.token_recall == 0.0
    assert [e.kind for e in item.errors] == [EXTRACTION_FAILED]


def test_extraction_quality_by_hand() -> None:
    q = extraction_quality("a b c d", "a  c\nb x")
    assert q.ref_chars == 7
    assert q.cer == pytest.approx(3 / 7)
    assert q.wer == pytest.approx(3 / 4)
    assert q.token_recall == pytest.approx(3 / 4)
    assert extraction_quality("", "x").cer is None


def test_pii_token_recall_counts_the_words_of_gold_values() -> None:
    values = ["Amal Haddad", "0791234567"]
    q = extraction_quality("Name: Amal Haddad, phone 0791234567.", "Name: Amal Hadad, phone 0791234567.", values)
    assert (q.pii_words, q.pii_found) == (3, 2)
    assert q.pii_token_recall == pytest.approx(2 / 3)
    assert extraction_quality("no values here", "no values here").pii_token_recall is None
    csv = extraction_quality("Amal Haddad | 0791234567", "Amal Haddad,0791234567", values)
    assert csv.pii_token_recall == 1.0


@pytest.mark.parametrize("cell_separator", ["\t", ","])
def test_table_separators_of_the_canonical_text_are_no_words(cell_separator: str) -> None:
    reference = "Ahmad Ali | 0791234567 | Amman\nSara Odeh | 0781234567 | Irbid"
    extracted = reference.replace(" | ", cell_separator)
    q = extraction_quality(reference, extracted)
    assert (q.token_recall, q.wer, q.cer) == (1.0, 0.0, 0.0)


def test_bidi_marks_around_latin_runs_are_dropped() -> None:
    q = extraction_quality("Email: amal@example.com", "Email: \u200eamal@example.com\u200f", ["amal@example.com"])
    assert (q.char_edits, q.token_recall, q.pii_token_recall) == (0, 1.0, 1.0)


def test_log_fields_and_bracketed_emails_keep_their_values() -> None:
    log = "kernel: SRC=192.0.2.10 DST=10.0.0.5\nReported by: Malek Haddad <malek98@shamdata.io>"
    values = ["192.0.2.10", "10.0.0.5", "malek98@shamdata.io", "Malek Haddad"]
    assert extraction_quality(log, log, values).pii_token_recall == 1.0


def test_mrz_filler_at_the_end_of_a_word_stays() -> None:
    mrz = "AL<HASAN<<FARAH<<<<"
    assert extraction_quality(mrz, "AL<HASAN<<FARAH<<", [mrz]).pii_token_recall == 0.0


def test_extraction_quality_adds_up() -> None:
    total = extraction_quality("a b", "a b") + extraction_quality("c d", "x y")
    assert total.items == 2
    assert total.wer == pytest.approx(2 / 4)
    assert total.to_dict()["token_recall"] == 0.5


def test_context_window_marks_the_hit() -> None:
    assert context_window("one two\nthree four", 4, 7, width=4) == "one [[two]] thr"


def test_prediction_round_trip() -> None:
    prediction = Prediction(
        item_id="r1",
        doc_id="d1",
        rendering_id="r1",
        extractor="text",
        detection=Detection(spans=[Span(0, 3, "EMAIL", "a@b", 0.5)], labels_present={"PASSPORT"}),
        extracted="a@b",
        extract_seconds=0.2,
    )
    assert Prediction.from_dict(prediction.to_dict()) == prediction


def test_score_predictions_dispatches_by_mode() -> None:
    docs = documents()
    doc = docs[1]
    text_items = score_predictions(docs, TEXT, [Prediction(item_id=doc.id, doc_id=doc.id)])
    assert LENIENT in text_items[0].levels
    rendering = doc.renderings[0]
    rendering_prediction = Prediction(
        item_id=rendering.id, doc_id=doc.id, rendering_id=rendering.id, extracted=doc.text
    )
    pipeline_items = score_predictions(docs, PIPELINE, [rendering_prediction])
    assert set(pipeline_items[0].levels) == {VALUE_LENIENT, VALUE, VALUE_STRICT, DOCUMENT}
    assert score_predictions(docs, EXTRACT, [rendering_prediction])[0].levels == {}
    with pytest.raises(KeyError):
        score_predictions(docs, TEXT, [Prediction(item_id="x", doc_id="missing")])


def test_series_and_groups_align_items() -> None:
    docs = documents()
    items = [score_text(d, Prediction(item_id=d.id, doc_id=d.id)) for d in docs]
    by_label = series(items, LENIENT, "label")
    assert all(len(values) == len(items) for values in by_label.values())
    assert grouped(items, LENIENT, "label")["EMAIL"] == Counts(fn=3)
    assert grouped(items, LENIENT, "family") == {"structured": Counts(fn=5), "contextual": Counts(fn=2)}
    assert grouped(items, LENIENT, "language")["ar"] == Counts(fn=3)
    assert quality_grouped(items) == {}


def test_series_sums_the_renderings_of_each_document() -> None:
    first = ItemScore("d1.a", "d1", "d1.a", {}, {LENIENT: {"EMAIL": Counts(tp=1)}})
    second = ItemScore("d1.b", "d1", "d1.b", {}, {LENIENT: {"EMAIL": Counts(fn=1)}})
    other = ItemScore("d2.a", "d2", "d2.a", {}, {LENIENT: {"PHONE": Counts(fp=1)}})
    assert series([other, first, second], LENIENT, "label") == {
        "EMAIL": [Counts(tp=1, fn=1), Counts()],
        "PHONE": [Counts(), Counts(fp=1)],
    }
    assert document_count([first, second, other]) == 2


def test_summarize_text_and_pipeline() -> None:
    docs = documents()
    doc = docs[1]
    text_summary = summarize(TEXT, [score_text(doc, _text_prediction(doc))])
    assert text_summary["main_level"] == LENIENT
    assert text_summary["documents"] == 1
    assert text_summary["levels"][LENIENT]["overall"]["tp"] == 2
    assert text_summary["levels"][LENIENT]["overall"]["ci95"] == dict.fromkeys(("precision", "recall", "f1", "f2"))
    assert text_summary["document_any"]["tp"] == 1
    assert text_summary["errors"][BOUNDARY] == 1
    assert "extraction" not in text_summary
    assert set(text_summary["levels"][LENIENT]) >= {"by_label", "by_family", "by_language", "by_doc_type"}
    rendering = doc.renderings[0]
    prediction = Prediction(item_id=rendering.id, doc_id=doc.id, rendering_id=rendering.id, extracted=doc.text)
    pipeline_summary = summarize(PIPELINE, [score_rendering(doc, rendering, prediction, detect=True)])
    assert pipeline_summary["main_level"] == VALUE_LENIENT
    assert pipeline_summary["extraction"]["overall"]["cer"] == 0.0
    assert "by_capture" in pipeline_summary["levels"][VALUE_LENIENT]
    assert "by_format" in pipeline_summary["extraction"]
    assert summarize(EXTRACT, [])["levels"] == {}


def test_summary_counts_false_positives_by_look_alike() -> None:
    doc = _doc("doc-en-1")
    rendering = doc.renderings[1]
    found = [("NATIONAL_ID", "INV-2024-001"), ("PHONE", "5550001111"), ("IBAN", "JO00 0000")]
    prediction = Prediction(
        item_id=rendering.id,
        doc_id=doc.id,
        rendering_id=rendering.id,
        detection=Detection(values=found),
        extracted=doc.text,
    )
    item = score_rendering(doc, rendering, prediction, detect=True)
    assert summarize(PIPELINE, [item])["false_positives_by_look_alike"] == {"invoice_number": 1, NO_LOOK_ALIKE: 2}
    text_item = score_text(doc, _text_prediction(doc))
    assert summarize(TEXT, [text_item])["false_positives_by_look_alike"] == {"invoice_number": 1}


def test_summary_intervals_cover_every_rate() -> None:
    docs = [d for d in documents() if d.split == "test"]
    items = [
        score_text(d, Prediction(item_id=d.id, doc_id=d.id, detection=Detection(spans=list(d.spans[:1])))) for d in docs
    ]
    intervals = summarize(TEXT, items)["levels"][LENIENT]["overall"]["ci95"]
    assert set(intervals) == {"precision", "recall", "f1", "f2"}
    assert intervals["recall"]["low"] <= intervals["recall"]["high"]


def test_partial_value_counts_only_at_the_lenient_level() -> None:
    doc = _doc("doc-ar-1")
    rendering = doc.renderings[0]
    detection = Detection(values=[("PERSON", "أحمد"), ("PHONE", "0781234567")])
    prediction = Prediction(item_id=rendering.id, doc_id=doc.id, rendering_id=rendering.id, detection=detection)
    item = score_rendering(doc, rendering, prediction, detect=True)
    assert item.levels[VALUE_LENIENT]["PERSON"] == TP
    assert item.levels[VALUE]["PERSON"] == Counts(0, 1, 1)
    partial = next(e for e in item.errors if e.kind == VALUE_PARTIAL)
    assert (partial.text, partial.gold_text) == ("أحمد", "أحمد علي")
    assert [e.label for e in item.errors if e.kind == VALUE_MISSED] == ["EMAIL"]


def _face_prediction(rendering_id: str, values: list[tuple[str, str]], extracted: str) -> Prediction:
    return Prediction(
        item_id=rendering_id,
        doc_id="card-1",
        rendering_id=rendering_id,
        detection=Detection(values=values),
        extracted=extracted,
    )


def test_each_face_is_scored_against_the_gold_it_shows() -> None:
    card = two_face_card()
    _, front, back = card.renderings
    found = [("PERSON", "Amal Haddad"), ("NATIONAL_ID", "9861234567")]
    item = score_rendering(card, front, _face_prediction(front.id, found, CARD_FRONT), detect=True)
    assert item.levels[VALUE_LENIENT] == {"PERSON": TP, "NATIONAL_ID": TP}
    assert item.levels[DOCUMENT] == {"PERSON": TP, "NATIONAL_ID": TP, ANY: TP}
    assert item.quality is not None
    assert item.quality.cer == 0.0
    assert item.quality.ref_chars == len(" ".join(words(CARD_FRONT)))
    assert not [e for e in item.errors if e.kind == VALUE_MISSED]
    back_prediction = _face_prediction(back.id, [("PHONE", "0791234567")], CARD_BACK)
    back_item = score_rendering(card, back, back_prediction, detect=True)
    assert back_item.levels[VALUE_LENIENT] == {"PHONE": TP, "ADDRESS": FN}
    assert [e.text for e in back_item.errors if e.kind == VALUE_MISSED] == ["12 Rainbow Street, Amman"]


def test_two_faces_add_up_to_the_gold_of_one_card() -> None:
    card = two_face_card()
    _, front, back = card.renderings
    predictions = [_face_prediction(front.id, [], CARD_FRONT), _face_prediction(back.id, [], CARD_BACK)]
    summary = summarize(PIPELINE, score_predictions([card], PIPELINE, predictions))
    assert summary["levels"][VALUE_LENIENT]["overall"]["support"] == len(card.spans)
    assert summary["levels"][DOCUMENT]["overall"]["support"] == len(card.spans)
    assert summary["extraction"]["overall"]["cer"] == 0.0
    assert (summary["documents"], summary["items"]) == (1, 2)
