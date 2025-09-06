from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List, Dict, Any
import math
import re

TOKEN_RE = re.compile(r"[A-Za-z_]{3,}")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _score_tokens(docs: List[List[str]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    N = len(docs)
    idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}
    return idf


def _vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    return {w: (tf[w] / len(tokens)) * idf.get(w, 0.0) for w in tf}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[w] * b[w] for w in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


def build_index(root: str | Path, out_path: str | Path) -> None:
    root = Path(root)
    functions: List[Dict[str, Any]] = []
    for py in root.rglob("*.py"):
        if py.name.startswith("_"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            continue
        src_lines = py.read_text(encoding="utf-8").splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                start = node.lineno - 1
                end = node.end_lineno or start + 1
                snippet = "\n".join(src_lines[start:end])
                blob = f"{node.name}\n{doc}\n{snippet}"
                tokens = _tokenize(blob)
                functions.append({
                    "name": node.name,
                    "path": str(py.relative_to(root)),
                    "doc": doc,
                    "tokens": tokens,
                })
    idf = _score_tokens([f["tokens"] for f in functions] or [[]])
    for f in functions:
        f["vector"] = _vector(f["tokens"], idf)
        del f["tokens"]
    index = {"functions": functions, "meta": {"count": len(functions)}}
    Path(out_path).write_text(json.dumps(index, indent=2), encoding="utf-8")


def load_index(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def search(index: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    # rebuild idf from vectors approximate: use keys of first vector union
    # Simpler: treat query vector using average idf weight from functions
    all_weights: Dict[str, float] = {}
    for f in index.get("functions", []):
        for w, wv in f.get("vector", {}).items():
            all_weights[w] = max(all_weights.get(w, 0.0), wv)
    q_vec = {w: all_weights.get(w, 1.0) for w in q_tokens}
    scored = []
    for f in index.get("functions", []):
        sim = _cosine(q_vec, f["vector"])
        scored.append((sim, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(score=s, **meta) for s, meta in scored[:top_k]]
