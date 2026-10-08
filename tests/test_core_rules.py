from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest

from akheval.dataset import Span, sha256_file
from akheval.detectors.core_rules import (
    APP_LOCATED_CATEGORIES,
    BIN_DIR,
    CoreRulesDetector,
    drop_nested,
    locate,
    to_detection,
)
from akheval.synth.values import Values, to_indic
from akheval.taxonomy import CORE_LABELS, map_label

REPO = Path(__file__).resolve().parents[1]
MAIN, RELEASE = "68d1749", "97c7afb"
PROPOSED = REPO / "rules" / "definitions-proposed.json"
PROPOSED_LOCATED = ["Email", "Phone Number", "IBAN", "IP Address", "UNHCR Number"]
SEED = 20261011
PROC_FDS = Path("/proc/self/fd")
FAKE_SHIM = """
import json, sys
if sys.argv[1:] == ["--version"]:
    print(json.dumps({"core_api": "v2", "accepts_definitions": True}))
    sys.exit(0)
for line in sys.stdin:
    text = json.loads(line)["text"]
    if text == "CRASH":
        sys.exit(3)
    print(json.dumps({"detections": {"Email": [text]}, "keyword_categories": []}), flush=True)
"""


def test_locate_finds_every_occurrence() -> None:
    assert locate("a@b.co x a@b.co", "a@b.co") == [(0, 6), (9, 15)]
    assert locate("abc", "zz") == []
    assert locate("abc", "") == []


def test_locate_tolerates_other_whitespace() -> None:
    text = "Ministry of\n  Interior"
    assert locate(text, "Ministry of Interior") == [(0, len(text))]


def test_drop_nested_keeps_the_longest_span_per_label() -> None:
    spans = [Span(0, 3, "PASSPORT"), Span(0, 6, "PASSPORT"), Span(0, 3, "NATIONAL_ID"), Span(0, 6, "PASSPORT")]
    assert drop_nested(spans) == [Span(0, 6, "PASSPORT"), Span(0, 3, "NATIONAL_ID")]


V2_TEXT = "JORDAN passport. JORDAN again. Mail a@b.co, IBAN JO94CBJO0010000000000131000302"
V2_RECORD = {
    "detections": {
        "Email": ["a@b.co"],
        "Jordanian Passport": ["JORDAN", "passport"],
        "IBAN": ["JO94CBJO0010000000000131000302"],
    },
    "keyword_categories": [],
}


def test_located_categories_become_values_and_spans_the_rest_flags_the_document() -> None:
    detection = to_detection(V2_TEXT, V2_RECORD)
    assert APP_LOCATED_CATEGORIES == ("Email", "Phone Number")
    assert [(s.start, s.label, s.text) for s in detection.spans] == [(36, "EMAIL", "a@b.co")]
    assert detection.values == [("EMAIL", "a@b.co")]
    assert detection.labels_present == {"EMAIL", "PASSPORT", "IBAN"}
    assert detection.raw_labels == {"Email": 1, "Jordanian Passport": 2, "IBAN": 1}


def test_located_categories_widen_what_the_detector_reports() -> None:
    detection = to_detection(V2_TEXT, V2_RECORD, {"Email", "Jordanian Passport", "IBAN"})
    assert [(s.start, s.label, s.text) for s in detection.spans] == [
        (0, "PASSPORT", "JORDAN"),
        (7, "PASSPORT", "passport"),
        (17, "PASSPORT", "JORDAN"),
        (36, "EMAIL", "a@b.co"),
        (49, "IBAN", "JO94CBJO0010000000000131000302"),
    ]
    assert ("IBAN", "JO94CBJO0010000000000131000302") in (detection.values or [])


def test_located_categories_must_map_to_a_taxonomy_label() -> None:
    with pytest.raises(ValueError, match="map to no taxonomy label"):
        CoreRulesDetector(MAIN, located_categories=["Email", "Banking"])


def test_categories_sharing_a_label_are_located_by_name() -> None:
    text = "UNHCR asylum seeker, case 945-24C12345"
    record = {"detections": {"UNHCR Number": ["945-24C12345"], "Syrian Refugee Document": ["UNHCR", "asylum"]}}
    detection = to_detection(text, record, {"UNHCR Number"})
    assert [(s.label, s.text) for s in detection.spans] == [("REFUGEE_ID", "945-24C12345")]
    assert detection.values == [("REFUGEE_ID", "945-24C12345")]
    assert detection.labels_present == {"REFUGEE_ID"}


def test_v1_keyword_categories_flag_the_document_only() -> None:
    record = {"detections": {"Phone Number": ["0791234567"]}, "keyword_categories": ["Banking", "Jordanian Passport"]}
    detection = to_detection("tel 0791234567", record)
    assert [s.label for s in detection.spans] == ["PHONE"]
    assert detection.labels_present == {"PHONE", "PASSPORT"}
    assert detection.unmapped == {"Banking": 1}


def test_proposed_categories_map_onto_the_taxonomy() -> None:
    rules = json.loads(PROPOSED.read_text(encoding="utf-8"))
    assert {category: map_label(CORE_LABELS, category) for category in rules} == {
        "Email": "EMAIL",
        "Phone Number": "PHONE",
        "IBAN": "IBAN",
        "IP Address": "IP_ADDRESS",
        "Jordanian National ID": "NATIONAL_ID",
        "Jordanian Passport": "PASSPORT",
        "UNHCR Number": "REFUGEE_ID",
        "Syrian Refugee Document": "REFUGEE_ID",
    }


def test_unhcr_rule_evidence_is_the_number_itself() -> None:
    branches = json.loads(PROPOSED.read_text(encoding="utf-8"))["UNHCR Number"]["rules"]
    assert [[list(condition) for condition in branch] for branch in branches] == [[["regex"]], [["regex"]]]


def test_missing_binary_names_the_build_command(tmp_path: Path) -> None:
    detector = CoreRulesDetector("abc1234", binary=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match=r"build\.sh abc1234"):
        detector.prepare()


@pytest.mark.skipif(not PROC_FDS.is_dir(), reason="counts open files through /proc")
def test_restarts_after_a_crash_release_every_pipe(tmp_path: Path) -> None:
    shim = tmp_path / "shim"
    shim.write_text(f"#!{sys.executable}{FAKE_SHIM}", encoding="utf-8")
    shim.chmod(0o755)
    detector = CoreRulesDetector("abc1234", binary=str(shim))
    before = len(list(PROC_FDS.iterdir()))
    for _ in range(5):
        with pytest.raises(RuntimeError, match="exit code 3"):
            detector.detect("CRASH")
        assert detector.detect("a@b.co").values == [("EMAIL", "a@b.co")]
    detector.close()
    assert len(list(PROC_FDS.iterdir())) == before


def _binary(sha: str) -> Path:
    path = REPO / BIN_DIR / f"akh-core-rules-{sha}"
    if not os.access(path, os.X_OK):
        pytest.skip(f"shim for {sha} not built; run rust/core-rules/build.sh {sha}")
    return path


def _proposed() -> CoreRulesDetector:
    """The core-proposed detector as configs/eval-v1.toml sets it up."""
    return CoreRulesDetector(
        MAIN, definitions=str(PROPOSED), binary=str(_binary(MAIN)), located_categories=PROPOSED_LOCATED
    )


@pytest.mark.integration
def test_main_engine_flags_keyword_categories_for_the_document() -> None:
    detector = CoreRulesDetector(MAIN, binary=str(_binary(MAIN)))
    text = "Passport of the Hashemite Kingdom of JORDAN. Mail a.b@x.com, call 0791234567."
    detection = detector.detect(text)
    detector.close()
    assert detection.values == [("EMAIL", "a.b@x.com")]
    assert [s.label for s in detection.spans] == ["EMAIL"]
    assert "PASSPORT" in detection.labels_present
    assert "PHONE" not in detection.predicted_labels()


@pytest.mark.integration
def test_main_engine_reports_keyword_evidence_as_spans_when_located() -> None:
    detector = CoreRulesDetector(MAIN, binary=str(_binary(MAIN)), located_categories=["Email", "Jordanian Passport"])
    detection = detector.detect("Passport of the Hashemite Kingdom of JORDAN. Mail a.b@x.com.")
    info = detector.info()
    detector.close()
    # Keyword evidence is the original whitespace-separated token, punctuation included.
    assert {s.text for s in detection.spans if s.label == "PASSPORT"} == {"Passport", "Hashemite Kingdom", "JORDAN."}
    assert info["located_categories"] == ["Email", "Jordanian Passport"]


@pytest.mark.integration
def test_release_engine_flags_keyword_categories() -> None:
    detector = CoreRulesDetector(RELEASE, binary=str(_binary(RELEASE)))
    detection = detector.detect("Call 0791234567 or mail a.b@x.com. Passport JOR bank")
    detector.close()
    assert sorted(detection.values or []) == [("EMAIL", "a.b@x.com"), ("PHONE", "0791234567")]
    assert detection.labels_present == {"EMAIL", "PHONE", "PASSPORT"}
    assert "Banking" in detection.unmapped
    assert all(s.label != "PASSPORT" for s in detection.spans)


@pytest.mark.integration
def test_release_engine_refuses_another_rule_file() -> None:
    detector = CoreRulesDetector(RELEASE, definitions=str(PROPOSED), binary=str(_binary(RELEASE)))
    with pytest.raises(ValueError, match="built-in rules"):
        detector.prepare()


@pytest.mark.integration
def test_proposed_rules_load_in_the_main_engine() -> None:
    detector = _proposed()
    texts = [
        "Call 0791234567 or +962 79 123 4567, office 06-5123456",
        "هاتف ٠٧٩١٢٣٤٥٦٧ وموبايل سوري 0944 123 456",
        "IBAN JO94 CBJO 0010 0000 0000 0131 0003 02 from 192.168.1.20",
        "ref 07912345678 and 300.1.1.1",
    ]
    detector.prepare()
    results = [detector.detect(t) for t in texts]
    info = detector.info()
    detector.close()
    assert sorted(v for label, v in results[0].values or [] if label == "PHONE") == [
        "+962 79 123 4567",
        "06-5123456",
        "0791234567",
    ]
    assert {v for _, v in results[1].values or []} == {"٠٧٩١٢٣٤٥٦٧", "0944 123 456"}
    assert set(results[2].values or []) == {
        ("IBAN", "JO94 CBJO 0010 0000 0000 0131 0003 02"),
        ("IP_ADDRESS", "192.168.1.20"),
    }
    assert results[3].predicted_labels() == set()
    assert info["core_api"] == "v2"
    assert info["definitions"] == str(PROPOSED)
    assert info["definitions_sha256"] == sha256_file(PROPOSED)


@pytest.mark.integration
def test_proposed_unhcr_rule_finds_generated_numbers_and_no_look_alikes() -> None:
    values = Values(random.Random(SEED))
    cases = [values.refugee_case_number() for _ in range(20)]
    individuals = [values.refugee_individual_number(case) for case in cases]
    numbers = [*cases, *individuals, to_indic(values.refugee_individual_number())]
    draws = (values.invoice_number, values.reference_number, values.order_number, values.version_number)
    look_alikes = [" ".join(draw() for _ in range(50)) for draw in draws]
    detector = _proposed()
    found, *rest = [detector.detect(t) for t in ["; ".join(numbers), *look_alikes]]
    detector.close()
    assert {v for label, v in found.values or [] if label == "REFUGEE_ID"} == set(numbers)
    assert ["REFUGEE_ID" in detection.labels_present for detection in rest] == [False] * len(draws)


@pytest.mark.integration
def test_info_and_restart_after_close() -> None:
    detector = CoreRulesDetector(MAIN, binary=str(_binary(MAIN)))
    info = detector.info()
    assert info["core_sha"] == MAIN
    assert info["core_commit"].startswith("68d1749f6652de908e76957144f7f1e76948cceb")
    assert info["definitions"].endswith("vendor/68d1749/definitions.json")
    detector.prepare()
    detector.close()
    assert detector.detect("a@b.co").values == [("EMAIL", "a@b.co")]
    detector.close()
