"""Start the BM25 retrieval server in the background (Kaggle notebook) and wait until it answers.

    !python scripts/kaggle/start_retriever.py            # after setup_retriever.sh

The server keeps running after this script exits (it is a detached child process), so the
next notebook cell can train against http://127.0.0.1:8000. Logs go to /tmp/retriever.log.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests


def java_home() -> str:
    java = shutil.which("java")
    if java is None:
        raise SystemExit("java not found: run scripts/kaggle/setup_retriever.sh first")
    return str(Path(os.path.realpath(java)).parent.parent)


def wait_healthy(url: str, timeout_s: float, log: Path, proc: subprocess.Popen) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            tail = log.read_text(errors="replace")[-3000:]
            raise SystemExit(f"retriever exited with code {proc.returncode}:\n{tail}")
        try:
            return requests.get(f"{url}/health", timeout=5).json()
        except requests.RequestException:
            time.sleep(5)
    raise SystemExit(f"retriever not healthy after {timeout_s:.0f}s; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the retriever server in the background")
    parser.add_argument("--data", default=os.environ.get("DATA", "/tmp/wiki18"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    try:
        print("[retriever] already running:", requests.get(f"{url}/health", timeout=3).json())
        return
    except requests.RequestException:
        pass

    log = Path("/tmp/retriever.log")
    env = {**os.environ, "JAVA_HOME": java_home()}
    cmd = [
        sys.executable,
        "-m",
        "nano_agentrl.retriever.server",
        "--backend",
        "pyserini",
        "--index",
        f"{args.data}/bm25",
        "--corpus",
        f"{args.data}/wiki-18.jsonl",
        "--port",
        str(args.port),
        "--threads",
        str(args.threads),
    ]
    proc = subprocess.Popen(  # noqa: S603 (fixed command)
        cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    print(f"[retriever] started pid {proc.pid}, waiting for {url}/health ...")
    print("[retriever]", wait_healthy(url, args.timeout, log, proc))

    resp = requests.post(f"{url}/search", json={"queries": ["capital of france"], "topk": 2})
    for doc in resp.json()["results"][0]:
        print(f"  {doc['title']}: {doc['text'][:80]}")
    print(f"[retriever] ready ({resp.json()['took_ms']:.0f} ms/query). Logs: {log}")


if __name__ == "__main__":
    main()
