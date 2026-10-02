"""FastAPI service and web UI.

Run with ``ragdoc serve`` or ``uvicorn ragdoc.api:create_app --factory``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ragdoc.documents import SUPPORTED_SUFFIXES, load_document
from ragdoc.generation import Generator
from ragdoc.knowledge_base import KnowledgeBase
from ragdoc.pipeline import RAGPipeline, default_generator

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"


def create_app(
    kb: KnowledgeBase | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    if kb is None:
        kb = KnowledgeBase.from_directory(os.environ.get("RAGDOC_DOCS_DIR", "data/sample_docs"))
    if generator is None:
        generator = default_generator(os.environ.get("RAGDOC_GENERATOR", "auto"))
    pipeline = RAGPipeline(kb, generator)

    app = FastAPI(
        title="RAG Doc Assistant",
        version="0.1.0",
        description="Ask questions about your documents and get answers with citations.",
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "generator": generator.name, **kb.stats()}

    @app.get("/api/documents")
    def documents() -> list[dict]:
        counts: dict[str, int] = {}
        for chunk in kb.chunks:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        return [{"id": d.id, "title": d.title, "chunks": counts.get(d.id, 0)} for d in kb.documents]

    @app.post("/api/search")
    def search(req: SearchRequest) -> list[dict]:
        hits = kb.search(req.query, k=req.k, mode=req.mode)
        return [
            {
                "rank": h.rank, "chunk_id": h.chunk.id, "label": h.chunk.label, "text": h.chunk.text,
                "score": round(h.score, 4), "bm25": round(h.bm25, 3), "dense": round(h.dense, 3),
            }
            for h in hits
        ]

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        try:
            answer = pipeline.ask(req.question, k=req.k, mode=req.mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # upstream LLM failure: report it, don't crash the server
            raise HTTPException(status_code=502, detail=_describe_llm_error(exc)) from exc
        return answer.to_dict()

    @app.post("/api/documents", status_code=201)
    async def upload(file: Annotated[UploadFile, File()]) -> dict:
        name = Path(file.filename or "upload.txt").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(415, f"Unsupported file type {suffix!r}")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 5 MB)")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            try:
                doc = load_document(path)
            except (UnicodeDecodeError, RuntimeError, ValueError) as exc:
                raise HTTPException(400, f"Could not read {name}: {exc}") from exc
        kb.add_documents([doc])
        return {"id": doc.id, "title": doc.title, **kb.stats()}

    return app


def _describe_llm_error(exc: Exception) -> str:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return f"Generation failed: {exc}"
    if isinstance(exc, anthropic.AuthenticationError):
        return "Anthropic API authentication failed - check ANTHROPIC_API_KEY."
    if isinstance(exc, anthropic.RateLimitError):
        return "Anthropic API rate limit reached - try again shortly."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error {exc.status_code}."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API."
    return f"Generation failed: {type(exc).__name__}"
