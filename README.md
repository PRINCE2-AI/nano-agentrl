# nano-AgentRL

A minimal, readable, from-scratch PyTorch re-implementation of
[Search-R1](https://arxiv.org/abs/2503.09516): multi-turn GRPO that teaches an LLM to
interleave reasoning with search-engine calls, plus an analysis of how agent
*reliability* changes during RL training.

Work in progress. See `CLAUDE.md` for the spec.

```bash
pip install -e ".[dev]"
pytest -q
```

## Retrieval corpus (wiki-18) and BM25 index

Needed for the search task only, on the GPU machine. Not on a laptop: the corpus is
~14 GB unzipped and the index ~2.3 GB.

```bash
pip install huggingface_hub pyserini      # pyserini needs Java 21 (e.g. apt install openjdk-21-jdk)

# corpus: 21M Wikipedia passages, {"id", "contents": "\"Title\"\ntext"} per line
hf download PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz --repo-type dataset --local-dir data/
gzip -d data/wiki-18.jsonl.gz

# prebuilt Lucene BM25 index
hf download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir indexes/wiki-18-bm25
# --index must point at the folder that contains the Lucene files (segments_*, *.cfs)
```

Start the server and check it:

```bash
python -m nano_agentrl.retriever.server --backend pyserini \
    --index indexes/wiki-18-bm25 --corpus data/wiki-18.jsonl --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/search -H "Content-Type: application/json" \
     -d '{"queries": ["who wrote hamlet"], "topk": 3}'
python -m nano_agentrl.retriever.client --url http://127.0.0.1:8000 --num_queries 512   # q/s
```

Without Java you can test on a small slice with the pure-Python backend:
`python -m nano_agentrl.retriever.server --backend bm25 --corpus data/wiki-18.jsonl --corpus_limit 50000`.
