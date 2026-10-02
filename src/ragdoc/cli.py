"""Command-line interface: ``ragdoc ask | search | eval | serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from ragdoc.evaluation import evaluate, format_table, load_questions
from ragdoc.knowledge_base import KnowledgeBase
from ragdoc.pipeline import RAGPipeline, default_generator
from ragdoc.retrieval import MODES

DEFAULT_DOCS = os.environ.get("RAGDOC_DOCS_DIR", "data/sample_docs")


def _kb(args: argparse.Namespace) -> KnowledgeBase:
    return KnowledgeBase.from_directory(args.docs, chunk_chars=args.chunk_chars)


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(_kb(args), default_generator(args.generator), k=args.k, mode=args.mode)
    answer = pipeline.ask(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0
    print(textwrap.fill(answer.text, width=100))
    print()
    cited = {c.source for c in answer.citations}
    for n, hit in enumerate(answer.sources, start=1):
        marker = "*" if n in cited else " "
        print(f" {marker}[{n}] {hit.chunk.label}  (score {hit.score:.3f})")
    print(f"\n{answer.generator} · {answer.latency_ms:.0f} ms")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    for hit in _kb(args).search(args.query, k=args.k, mode=args.mode):
        scores = f"score={hit.score:.4f} bm25={hit.bm25:.2f} dense={hit.dense:.3f}"
        print(f"[{hit.rank}] {hit.chunk.label}  {scores}")
        print(textwrap.indent(textwrap.fill(hit.chunk.text, width=96), "    "))
        print()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    kb = _kb(args)
    questions = load_questions(args.questions)
    results = evaluate(kb, questions)
    stats = kb.stats()
    print(f"{len(questions)} questions · {stats['documents']} documents · {stats['chunks']} chunks\n")
    print(format_table(results))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ["RAGDOC_DOCS_DIR"] = args.docs
    os.environ["RAGDOC_GENERATOR"] = args.generator
    uvicorn.run("ragdoc.api:create_app", factory=True, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdoc", description="Question answering over your documents."
    )
    parser.add_argument("--docs", default=DEFAULT_DOCS, help="directory of .md/.txt/.pdf files")
    parser.add_argument("--chunk-chars", type=int, default=700)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=4)
    ask.add_argument("--mode", choices=MODES, default="hybrid")
    ask.add_argument("--generator", choices=["auto", "claude", "extractive"], default="auto")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    search = sub.add_parser("search", help="show the ranked chunks for a query")
    search.add_argument("query")
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--mode", choices=MODES, default="hybrid")
    search.set_defaults(func=cmd_search)

    ev = sub.add_parser("eval", help="retrieval metrics on a labelled question set")
    ev.add_argument("--questions", default="eval/questions.jsonl")
    ev.set_defaults(func=cmd_eval)

    serve = sub.add_parser("serve", help="run the API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--generator", choices=["auto", "claude", "extractive"], default="auto")
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
