from types import SimpleNamespace

from ragdoc.generation import NOT_FOUND, ClaudeGenerator, ExtractiveGenerator


def test_extractive_answer_quotes_the_policy_and_cites_it(kb):
    hits = kb.search("How many days of paid annual leave do employees get?", k=4)
    answer = ExtractiveGenerator().generate("How many days of paid annual leave do employees get?", hits)
    assert not answer.abstained
    assert "24 days of paid annual leave" in answer.text
    assert "[1]" in answer.text
    assert answer.citations[0].chunk_id.startswith("02-time-off")


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def text_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def test_claude_request_uses_document_blocks_with_citations(kb):
    hits = kb.search("parental leave", k=3)
    request = ClaudeGenerator(client=object()).build_request("How long is parental leave?", hits)
    content = request["messages"][0]["content"]
    assert request["model"] == "claude-opus-5-5"
    assert [b["type"] for b in content] == ["document"] * 3 + ["text"]
    assert all(b["citations"] == {"enabled": True} for b in content[:3])
    assert content[0]["source"]["data"] == hits[0].chunk.text
    assert request["fallbacks"] == "default"


def test_claude_response_citations_become_numbered_markers(kb):
    hits = kb.search("parental leave", k=3)
    cite = SimpleNamespace(type="char_location", document_index=0, cited_text=hits[0].chunk.text[:40])
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig"),
            text_block("Primary caregivers get "),
            text_block("16 weeks of fully paid parental leave", [cite]),
            text_block("."),
        ],
    )
    client, messages = fake_client(response)
    answer = ClaudeGenerator(client=client).generate("How long is parental leave?", hits)
    assert answer.text == "Primary caregivers get 16 weeks of fully paid parental leave [1]."
    assert len(messages.calls) == 1
    assert [c.source for c in answer.citations] == [1]
    assert not answer.abstained


def test_claude_answer_without_citations_is_marked_abstained(kb):
    hits = kb.search("stock price", k=2)
    response = SimpleNamespace(
        stop_reason="end_turn", model="m", content=[text_block("I couldn't find that in the knowledge base.")]
    )
    client, _ = fake_client(response)
    answer = ClaudeGenerator(client=client).generate("What is the stock price?", hits)
    assert answer.abstained


def test_claude_refusal_is_handled(kb):
    hits = kb.search("leave", k=2)
    response = SimpleNamespace(stop_reason="refusal", model="m", content=[])
    client, _ = fake_client(response)
    answer = ClaudeGenerator(client=client).generate("question", hits)
    assert answer.abstained and "declined" in answer.text


def test_no_hits_means_not_found_without_calling_the_model():
    client, messages = fake_client(None)
    answer = ClaudeGenerator(client=client).generate("anything", [])
    assert answer.text == NOT_FOUND and not messages.calls
