from __future__ import annotations

from pathlib import Path
from .semantic_index import build_index, load_index, search


def build_and_query(root: str, query: str, tmp_index: str | None = None, top_k: int = 5):
    idx_path = tmp_index or (Path(root) / "knowledge_db" / "index.json")
    build_index(root, idx_path)
    idx = load_index(idx_path)
    return search(idx, query, top_k=top_k)


if __name__ == "__main__":  # pragma: no cover
    import sys, json
    root = sys.argv[1]
    q = sys.argv[2]
    res = build_and_query(root, q)
    print(json.dumps(res, indent=2))
