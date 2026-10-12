#!/usr/bin/env bash
# Set up the wiki-18 BM25 retriever inside a Kaggle (Linux) notebook. Run once per session:
#
#   !bash scripts/kaggle/setup_retriever.sh
#
# Installs Java 21 + Pyserini (without touching Kaggle's torch), downloads the prebuilt BM25
# index (2.3 GB) and the corpus (5.1 GB gz -> ~14 GB jsonl) to $DATA, builds the line-offset
# file for random access, and runs one test query. Data lives outside /kaggle/working, so it
# is not saved as notebook output (which is capped at ~20 GB). Re-running skips finished steps.
set -euo pipefail

DATA="${DATA:-/tmp/wiki18}"
mkdir -p "$DATA"

free_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc '0-9')
echo "[setup] data dir $DATA, free disk ${free_gb} GB"
if [[ ! -f "$DATA/wiki-18.jsonl" && "$free_gb" -lt 25 ]]; then
  echo "[setup] need ~25 GB free for the corpus; set DATA=<bigger disk> and retry" >&2
  exit 1
fi

# 1. Java 21 (Anserini 2.x, inside Pyserini 2.4, needs it)
if ! java -version 2>&1 | grep -q 'version "21'; then
  echo "[setup] installing Java 21 ..."
  apt-get update -qq >/dev/null
  apt-get install -y -qq openjdk-21-jdk-headless >/dev/null
fi
java -version 2>&1 | head -1

# 2. Pyserini without its dependency list: it pins torch>=2.9 and would replace the CUDA
#    torch Kaggle ships. Only the small packages Kaggle lacks are added.
if ! python -c "import pyserini" 2>/dev/null; then
  echo "[setup] installing pyserini ..."
  pip install -q --no-deps "pyserini==2.4.0"
  pip install -q "pyjnius>=1.7" tiktoken openai onnxruntime sentencepiece
fi

# 3. Index and corpus from the Search-R1 HF collection
python - "$DATA" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

data = Path(sys.argv[1])
if not (data / "bm25" / "segments_1").exists():
    print("[setup] downloading BM25 index (2.3 GB) ...", flush=True)
    snapshot_download("PeterJinGo/wiki-18-bm25-index", repo_type="dataset", local_dir=data)
if not (data / "wiki-18.jsonl").exists() and not (data / "wiki-18.jsonl.gz").exists():
    print("[setup] downloading corpus (5.1 GB) ...", flush=True)
    hf_hub_download("PeterJinGo/wiki-18-corpus", "wiki-18.jsonl.gz", repo_type="dataset",
                    local_dir=data)
EOF
if [[ ! -f "$DATA/wiki-18.jsonl" ]]; then
  echo "[setup] unzipping corpus (~5-10 min) ..."
  gzip -d "$DATA/wiki-18.jsonl.gz"   # removes the .gz, so peak disk is ~19 GB
fi

# 4. Line offsets (one scan, ~170 MB) + a test query straight through Pyserini
JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
export JAVA_HOME
python - "$DATA" <<'EOF'
import sys, time
from nano_agentrl.retriever.server import JsonlCorpus, PyseriniBackend

data = sys.argv[1]
t = time.time()
corpus = JsonlCorpus(f"{data}/wiki-18.jsonl")
print(f"[setup] corpus: {len(corpus):,} passages (offsets ready in {time.time() - t:.0f}s)")
backend = PyseriniBackend(f"{data}/bm25", f"{data}/wiki-18.jsonl", threads=4)
print(f"[setup] index: {len(backend):,} docs")
for doc in backend.search(["who wrote hamlet"], topk=3)[0]:
    print(f"  {doc['score']:.2f}  {doc['title']}: {doc['text'][:80]}")
EOF
df -h "$DATA" | tail -1
echo "[setup] done. Start the server with: python scripts/kaggle/start_retriever.py"
