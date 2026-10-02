import pytest
from conftest import SAMPLE_DOCS
from fastapi.testclient import TestClient

from ragdoc.api import create_app
from ragdoc.generation import ExtractiveGenerator
from ragdoc.knowledge_base import KnowledgeBase


@pytest.fixture()
def client():
    kb = KnowledgeBase.from_directory(SAMPLE_DOCS)
    return TestClient(create_app(kb=kb, generator=ExtractiveGenerator()))


def test_health_and_documents(client):
    health = client.get("/api/health").json()
    assert health == {"status": "ok", "generator": "extractive", "documents": 9, "chunks": health["chunks"]}
    docs = client.get("/api/documents").json()
    assert {d["id"] for d in docs} >= {"02-time-off", "07-deployment"}


def test_ask_returns_answer_with_sources(client):
    res = client.post("/api/ask", json={"question": "How much is the home office stipend?"})
    assert res.status_code == 200
    body = res.json()
    assert "500 USD" in body["answer"]
    assert body["sources"][0]["n"] == 1
    assert body["citations"]


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "mode": "magic"}).status_code == 422


def test_search_endpoint(client):
    hits = client.post("/api/search", json={"query": "canary error rate", "k": 2}).json()
    assert len(hits) == 2 and hits[0]["label"].startswith("Deployment Process")


def test_upload_adds_a_searchable_document(client):
    content = b"# Pet Policy\n\nDogs are welcome in the office on Wednesdays.\n"
    res = client.post("/api/documents", files={"file": ("pets.md", content, "text/markdown")})
    assert res.status_code == 201 and res.json()["documents"] == 10
    answer = client.post("/api/ask", json={"question": "Which day can I bring my dog to the office?"}).json()
    assert "Wednesdays" in answer["answer"]


def test_upload_rejects_unsupported_types(client):
    res = client.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 415


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "RAG Doc Assistant" in res.text
