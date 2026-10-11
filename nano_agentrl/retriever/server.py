"""FastAPI retrieval server: POST /search {queries: [...], topk: 3} -> top-k docs per query.

Two backends behind the same API:
    pyserini  Lucene BM25 over the prebuilt wiki-18 index (PeterJinGo/wiki-18-bm25-index).
              Needs Java 21 and `pip install pyserini`. This is what Search-R1 uses.
    bm25      Pure-Python rank_bm25 over an in-memory corpus. Fine for tests and corpora
              up to ~100k passages; far too slow and memory-hungry for 21M wiki passages.

Corpus format (wiki-18, PeterJinGo/wiki-18-corpus): jsonl, one passage per line,
    {"id": "0", "contents": "\"Title\"\nPassage text ..."}
The first line of `contents` is the (quoted) title, as in Search-R1's _passages2string.

Run:
    python -m nano_agentrl.retriever.server --backend pyserini \
        --index indexes/wiki-18-bm25 --corpus data/wiki-18.jsonl --port 8000
API:
    GET  /health -> {"status": "ok", "backend": ..., "num_docs": ...}
    POST /search {"queries": ["q1", "q2"], "topk": 3}
         -> {"results": [[{"id", "title", "text", "score"}, ...], ...], "took_ms": 12.3}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

Doc = dict[str, Any]  # {"id": str, "title": str, "text": str, "score": float}


def split_contents(contents: str) -> tuple[str, str]:
    """'"Paris"\\nParis is the capital...' -> ('Paris', 'Paris is the capital...')."""
    title, _, text = contents.partition("\n")
    return title.strip().strip('"'), text.strip()


def load_corpus_jsonl(path: str | Path, limit: int | None = None) -> list[Doc]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            title, text = split_contents(row["contents"])
            docs.append({"id": str(row.get("id", i)), "title": title, "text": text})
    return docs


class SearchBackend(Protocol):
    def search(self, queries: list[str], topk: int) -> list[list[Doc]]: ...

    def __len__(self) -> int: ...


# ---------------------------------------------------------------- rank_bm25 backend

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Backend:
    """In-memory BM25 (Okapi, k1=0.9, b=0.4 like Pyserini's defaults) over title + text."""

    def __init__(self, docs: list[Doc]) -> None:
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(f"{d['title']} {d['text']}") for d in docs], k1=0.9, b=0.4)

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, queries: list[str], topk: int) -> list[list[Doc]]:
        results = []
        for q in queries:
            scores = self.bm25.get_scores(tokenize(q))
            k = min(topk, len(self.docs))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            results.append([{**self.docs[i], "score": float(scores[i])} for i in top])
        return results


# ---------------------------------------------------------------- pyserini backend


class PyseriniBackend:
    """Lucene BM25 via Pyserini. Docs are read from the index if it stores them, else from
    the corpus jsonl (loaded with `datasets`, memory-mapped, looked up by integer id)."""

    def __init__(self, index_dir: str, corpus_path: str | None, threads: int = 16) -> None:
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(index_dir)
        self.threads = threads
        self.corpus = None
        if corpus_path:
            from datasets import load_dataset

            self.corpus = load_dataset("json", data_files=corpus_path, split="train")

    def __len__(self) -> int:
        return self.searcher.num_docs

    def _doc(self, docid: str) -> tuple[str, str]:
        if self.corpus is not None:
            return split_contents(self.corpus[int(docid)]["contents"])
        raw = json.loads(self.searcher.doc(docid).raw())
        return split_contents(raw["contents"])

    def search(self, queries: list[str], topk: int) -> list[list[Doc]]:
        qids = [str(i) for i in range(len(queries))]
        hits = self.searcher.batch_search(queries, qids, k=topk, threads=self.threads)
        results = []
        for qid in qids:
            docs = []
            for h in hits.get(qid, []):
                title, text = self._doc(h.docid)
                docs.append({"id": h.docid, "title": title, "text": text, "score": float(h.score)})
            results.append(docs)
        return results


# ---------------------------------------------------------------- app


class SearchRequest(BaseModel):
    queries: list[str]
    topk: int = Field(default=3, ge=1, le=100)


def create_app(backend: SearchBackend, backend_name: str = "bm25") -> FastAPI:
    app = FastAPI(title="nano-agentrl retriever")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": backend_name, "num_docs": len(backend)}

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        start = time.perf_counter()
        results = backend.search(req.queries, req.topk) if req.queries else []
        return {"results": results, "took_ms": (time.perf_counter() - start) * 1000}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 retrieval server")
    parser.add_argument("--backend", choices=["pyserini", "bm25"], default="pyserini")
    parser.add_argument("--index", help="Lucene index dir (pyserini backend)")
    parser.add_argument("--corpus", help="corpus jsonl ({id, contents} per line)")
    parser.add_argument("--corpus_limit", type=int, default=None, help="bm25: first N docs only")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if args.backend == "pyserini":
        if not args.index:
            parser.error("--index is required for the pyserini backend")
        backend: SearchBackend = PyseriniBackend(args.index, args.corpus, args.threads)
    else:
        if not args.corpus:
            parser.error("--corpus is required for the bm25 backend")
        backend = BM25Backend(load_corpus_jsonl(args.corpus, args.corpus_limit))
    print(f"[retriever] {args.backend}: {len(backend)} docs")

    import uvicorn

    uvicorn.run(create_app(backend, args.backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
