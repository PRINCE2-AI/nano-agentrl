"""Retriever server + client on a tiny in-memory corpus (no downloads, no network)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nano_agentrl.retriever.client import (
    TINY_CORPUS,
    InMemoryRetriever,
    RetrieverClient,
    benchmark,
)
from nano_agentrl.retriever.server import (
    BM25Backend,
    create_app,
    load_corpus_jsonl,
    split_contents,
)


@pytest.fixture(scope="module")
def http():
    return TestClient(create_app(BM25Backend(TINY_CORPUS), "bm25"))


def test_health(http):
    body = http.get("/health").json()
    assert body == {"status": "ok", "backend": "bm25", "num_docs": len(TINY_CORPUS)}


def test_search_ranks_relevant_doc_first(http):
    body = http.post("/search", json={"queries": ["who wrote hamlet"], "topk": 2}).json()
    assert body["took_ms"] >= 0
    (docs,) = body["results"]
    assert len(docs) == 2
    assert docs[0]["title"] == "Hamlet"
    assert set(docs[0]) == {"id", "title", "text", "score"}
    assert docs[0]["score"] >= docs[1]["score"]


def test_batched_queries_keep_order(http):
    queries = ["capital of germany", "highest mountain", "jane austen novel"]
    results = http.post("/search", json={"queries": queries, "topk": 1}).json()["results"]
    assert [r[0]["title"] for r in results] == ["Berlin", "Mount Everest", "Pride and Prejudice"]


def test_empty_and_invalid_requests(http):
    assert http.post("/search", json={"queries": [], "topk": 3}).json()["results"] == []
    assert http.post("/search", json={"queries": ["x"], "topk": 0}).status_code == 422


def test_topk_larger_than_corpus(http):
    results = http.post("/search", json={"queries": ["paris"], "topk": 50}).json()["results"]
    assert len(results[0]) == len(TINY_CORPUS)


def test_client_over_http_matches_in_memory(http):
    client = RetrieverClient("http://testserver", session=http, batch_size=2)
    queries = ["capital of france", "python language", "who wrote hamlet"]
    assert client.health()["num_docs"] == len(TINY_CORPUS)
    over_http = client.search(queries, topk=2)  # split into 2 HTTP batches
    in_memory = InMemoryRetriever(TINY_CORPUS).search(queries, topk=2)
    assert over_http == in_memory


def test_benchmark_reports_qps(http, capsys):
    qps = benchmark(RetrieverClient("http://testserver", session=http), 20, 8, 3)
    assert qps > 0 and "q/s" in capsys.readouterr().out


def test_wiki18_corpus_format(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"id": "0", "contents": '"Paris"\nParis is in France.'},
        {"id": "1", "contents": "X\nY"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    docs = load_corpus_jsonl(path)
    assert docs[0] == {"id": "0", "title": "Paris", "text": "Paris is in France."}
    assert split_contents("Title only") == ("Title only", "")
    assert len(load_corpus_jsonl(path, limit=1)) == 1


def test_jsonl_corpus_random_access(tmp_path):
    from nano_agentrl.retriever.server import JsonlCorpus, build_line_offsets

    path = tmp_path / "c.jsonl"
    rows = [{"id": str(i), "contents": f'"T{i}"\nText ünïcode {i}'} for i in range(1000)]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    assert len(build_line_offsets(path, chunk_bytes=97)) == 1000  # chunk edges must not matter
    corpus = JsonlCorpus(path)
    assert len(corpus) == 1000
    assert corpus[0] == rows[0] and corpus[999] == rows[999] and corpus[123] == rows[123]
    assert (tmp_path / "c.jsonl.offsets.npy").exists()
    assert JsonlCorpus(path)[500] == rows[500]  # reuses the saved offsets


def test_lookup_by_id_tolerates_shifted_lines(tmp_path):
    from nano_agentrl.retriever.server import JsonlCorpus, lookup_by_id

    path = tmp_path / "c.jsonl"
    rows = [{"id": "x", "contents": "junk\nfirst line"}] + [
        {"id": str(i), "contents": f"T{i}\nbody {i}"} for i in range(50)
    ]  # one extra line at the top: every id sits one line below its number
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    corpus = JsonlCorpus(path)
    assert lookup_by_id(corpus, "7")["contents"] == "T7\nbody 7"
    assert lookup_by_id(corpus, "49")["contents"] == "T49\nbody 49"
    with pytest.raises(KeyError):
        lookup_by_id(corpus, "999")
