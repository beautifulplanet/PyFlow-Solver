from pathlib import Path
from pyfoamclone.knowledge.semantic_index import build_index, load_index, search


def test_semantic_query_pressure_matrix(tmp_path: Path):
    # build index on current project root
    from pyfoamclone import __file__ as root_init
    root = Path(root_init).parent
    out = tmp_path / "index.json"
    build_index(root, out)
    idx = load_index(out)
    results = search(idx, "pressure matrix", top_k=10)
    # Expect build_pressure_matrix to appear in top names
    names = [r['name'] for r in results]
    assert any(n == 'build_pressure_matrix' for n in names)
