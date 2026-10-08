import pytest

from akheval.synth.builder import Ent, LayoutBuilder, Neg, block_side, has_back, side_ranges


def test_offsets_point_at_values_across_blocks() -> None:
    b = LayoutBuilder()
    b.title("Employee record")
    b.blank()
    b.field("Name", Ent("Lina Haddad", "PERSON"), key="name")
    b.row(Ent("Omar Odeh", "PERSON"), "Finance", (Ent("079 123 4567", "PHONE"), " (mobile)"))
    b.para("Ref. ", Neg("REF/2025/0001", "reference_number"), " dated ", Neg("01/02/2025", "date"))
    b.message("10:02", Ent("Sara", "PERSON"), "hi")
    b.mrz("P<JORHADDAD<<LINA<<<<<<<<<<<<<<<<<<<<<<<<<<<", "PASSPORT", side="back")
    layout = b.build()
    assert layout.text == "\n".join(block["text"] for block in layout.blocks)
    for span in (*layout.spans, *layout.negatives):
        assert layout.text[span.start : span.end] == span.text
    assert [s.label for s in layout.spans] == ["PERSON", "PERSON", "PHONE", "PERSON", "PASSPORT"]
    assert layout.blocks[2] == {"kind": "field", "text": "Name: Lina Haddad", "field": "name"}
    assert layout.blocks[3]["text"] == "Omar Odeh | Finance | 079 123 4567 (mobile)"
    assert layout.blocks[5]["text"] == "[10:02] Sara: hi"
    assert layout.blocks[6]["side"] == "back"
    assert layout.blocks[1] == {"kind": "blank", "text": ""}


def test_empty_builder_gives_empty_text() -> None:
    assert LayoutBuilder().build().text == ""


def test_builder_rejects_multiline_text() -> None:
    with pytest.raises(ValueError, match="one line"):
        LayoutBuilder().para("two\nlines")


def test_builder_rejects_unknown_kinds_labels_and_sides() -> None:
    with pytest.raises(ValueError, match="block kind"):
        LayoutBuilder().add("footer", "x")
    with pytest.raises(ValueError, match="side"):
        LayoutBuilder().title("x", side="top")
    with pytest.raises(ValueError, match="label"):
        Ent("x", "PHONE_NUMBER")
    with pytest.raises(ValueError, match="negative kind"):
        Neg("x", "phone")
    with pytest.raises(ValueError, match="empty"):
        Ent(" ", "PERSON")


def _card() -> LayoutBuilder:
    b = LayoutBuilder()
    b.title("Kingdom", side="front")
    b.field("Name", Ent("Lina Haddad", "PERSON"), side="front")
    b.blank(side="front")
    b.field("Residence", Ent("Irbid", "LOCATION"), side="back")
    b.blank(side="back")
    b.mrz("IDJORXYZ12345<<<<<<<<<<<<<<<<", "NATIONAL_ID", side="back")
    return b


def test_side_ranges_cover_each_side_without_edge_newlines() -> None:
    layout = _card().build()
    front, back = side_ranges(layout.blocks, "front"), side_ranges(layout.blocks, "back")
    assert [layout.text[a:b] for a, b in front] == ["Kingdom\nName: Lina Haddad"]
    assert [layout.text[a:b] for a, b in back] == ["Residence: Irbid\n\nIDJORXYZ12345<<<<<<<<<<<<<<<<"]
    for span in layout.spans:
        assert sum(a <= span.start and span.end <= b for a, b in (*front, *back)) == 1


def test_side_ranges_give_one_range_per_run() -> None:
    b = _card()
    b.para("Signature", side="front")
    blocks = b.build().blocks
    assert len(side_ranges(blocks, "front")) == 2
    assert len(side_ranges(blocks, "back")) == 1


def test_blocks_without_a_side_belong_to_the_front() -> None:
    b = LayoutBuilder()
    b.title("Passport")
    b.field("No.", Ent("N1234567", "PASSPORT"))
    blocks = b.build().blocks
    assert block_side(blocks[0]) == "front"
    assert not has_back(blocks)
    assert has_back(_card().build().blocks)
    assert side_ranges(blocks, "back") == ()
