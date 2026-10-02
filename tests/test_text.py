from ragdoc.text import sentence_spans, stem, tokenize


def test_tokenize_lowercases_drops_stopwords_and_stems():
    assert tokenize("The Deploys are Running on Fridays") == ["deploy", "runn", "friday"]


def test_stem_handles_common_suffixes():
    assert stem("policies") == "policy"
    assert stem("approvals") == "approval"
    assert stem("class") == "class"  # double-s is not a plural
    assert stem("bus") == "bus"


def test_sentence_spans_index_into_original_text():
    text = (
        "# Title\n\nFirst sentence here. Second one wraps\nonto the next line!\nThird.\n\n"
        "- A bullet item\n"
    )
    spans = sentence_spans(text)
    pieces = [text[s.start : s.end] for s in spans]
    assert pieces == [
        "# Title",
        "First sentence here.",
        "Second one wraps\nonto the next line!",
        "Third.",
        "- A bullet item",
    ]


def test_sentence_spans_do_not_split_on_decimals():
    text = "Version 2.5 ships today. It is fast."
    pieces = [text[s.start : s.end] for s in sentence_spans(text)]
    assert pieces == ["Version 2.5 ships today.", "It is fast."]
