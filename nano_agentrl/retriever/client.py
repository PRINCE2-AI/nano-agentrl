"""HTTP client for the retrieval server, batched over queries, plus a queries/sec benchmark.

    client = RetrieverClient("http://127.0.0.1:8000")
    client.search(["who wrote hamlet", "capital of france"], topk=3)
    # -> [[{"id", "title", "text", "score"}, ...], [...]]

InMemoryRetriever has the same `search` method without HTTP, for tests and dry runs.

Benchmark:
    python -m nano_agentrl.retriever.client --url http://127.0.0.1:8000 --num_queries 512
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import requests

Doc = dict[str, Any]


class RetrieverClient:
    def __init__(
        self,
        url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
        retries: int = 3,
        batch_size: int = 256,
        session: Any = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.batch_size = batch_size
        self.session = session or requests.Session()

    def health(self) -> dict:
        resp = self.session.get(f"{self.url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, queries: list[str], topk: int) -> list[list[Doc]]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.post(
                    f"{self.url}/search",
                    json={"queries": queries, "topk": topk},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["results"]
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(
            f"retriever at {self.url} unreachable after {self.retries} tries. "
            "Is `python -m nano_agentrl.retriever.server` running?"
        ) from last_error

    def search(self, queries: list[str], topk: int = 3) -> list[list[Doc]]:
        """Top-k docs for each query, in query order. Splits large batches."""
        results: list[list[Doc]] = []
        for s in range(0, len(queries), self.batch_size):
            results += self._post(queries[s : s + self.batch_size], topk)
        return results


class InMemoryRetriever:
    """Same interface as RetrieverClient, backed by an in-process BM25 over `docs`."""

    def __init__(self, docs: list[Doc]) -> None:
        from nano_agentrl.retriever.server import BM25Backend

        self.backend = BM25Backend(docs)
        self.calls: list[list[str]] = []  # every batch of queries, for tests

    def search(self, queries: list[str], topk: int = 3) -> list[list[Doc]]:
        self.calls.append(list(queries))
        return self.backend.search(queries, topk)


TINY_CORPUS: list[Doc] = [
    {"id": "0", "title": "Paris", "text": "Paris is the capital and largest city of France."},
    {"id": "1", "title": "Hamlet", "text": "Hamlet is a tragedy written by William Shakespeare."},
    {"id": "2", "title": "Mount Everest", "text": "Mount Everest is Earth's highest mountain."},
    {"id": "3", "title": "Python", "text": "Python is a programming language created by Guido."},
    {"id": "4", "title": "Berlin", "text": "Berlin is the capital of Germany."},
    {"id": "5", "title": "Pride and Prejudice", "text": "A novel by Jane Austen, published 1813."},
]


def benchmark(client: RetrieverClient, num_queries: int, batch_size: int, topk: int) -> float:
    """Send `num_queries` queries in batches; print and return queries/sec."""
    words = ["capital", "city", "river", "novel", "author", "war", "king", "film", "band", "year"]
    queries = [f"{words[i % 10]} {words[(i * 7) % 10]} {i}" for i in range(num_queries)]
    client.batch_size = batch_size
    start = time.perf_counter()
    client.search(queries, topk)
    elapsed = time.perf_counter() - start
    qps = num_queries / elapsed
    print(f"{num_queries} queries, batch {batch_size}, topk {topk}: {elapsed:.2f}s, {qps:.1f} q/s")
    return qps


def main() -> None:
    parser = argparse.ArgumentParser(description="Retriever benchmark (queries/sec)")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    client = RetrieverClient(args.url)
    print(client.health())
    benchmark(client, args.num_queries, args.batch_size, args.topk)


if __name__ == "__main__":
    main()
