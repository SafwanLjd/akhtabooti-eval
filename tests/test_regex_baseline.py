from __future__ import annotations

from akheval.detectors.regex_baseline import RegexBaselineDetector


def test_finds_emails_and_jordanian_mobiles() -> None:
    text = "Mail a.b@x.com, call +962 79 123 4567 or 0781234567; not 07812345678 or 0612345678."
    detection = RegexBaselineDetector().detect(text)
    assert [(s.label, s.text) for s in detection.spans] == [
        ("EMAIL", "a.b@x.com"),
        ("PHONE", "+962 79 123 4567"),
        ("PHONE", "0781234567"),
    ]
    assert all(text[s.start : s.end] == s.text for s in detection.spans)
    assert detection.raw_labels == {"EMAIL": 1, "PHONE": 2}


def test_info_lists_patterns() -> None:
    assert set(RegexBaselineDetector().info()["patterns"]) == {"EMAIL", "PHONE"}
