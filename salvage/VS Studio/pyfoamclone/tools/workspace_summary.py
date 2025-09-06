from __future__ import annotations
import json
import os
from collections import Counter
from pathlib import Path

# Generates a slim workspace_summary.json from a (potentially large) workspace_full_inventory.json
# If the full inventory file is missing, performs a lightweight on-the-fly scan.

def generate_summary(root: str | Path = None, full_inventory_path: str | Path = None, out_path: str | Path = None):
    root = Path(root or Path(__file__).resolve().parent.parent)
    full_inventory_path = Path(full_inventory_path or root / 'workspace_full_inventory.json')
    out_path = Path(out_path or root / 'workspace_summary.json')

    files = []
    if full_inventory_path.exists():
        try:
            data = json.loads(full_inventory_path.read_text(encoding='utf-8'))
            files = data.get('files', [])
        except Exception:
            files = []
    if not files:
        # fallback scan
        for dirpath, _dirs, filenames in os.walk(root):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                files.append(rel)
    total_files = len(files)
    exts = [Path(f).suffix or '' for f in files]
    c = Counter(exts)
    top10 = c.most_common(10)
    summary = {
        'total_files': total_files,
        'unique_extensions': len(c),
        'top_extensions': [{'ext': ext, 'count': count} for ext, count in top10],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary

if __name__ == '__main__':  # pragma: no cover
    print(json.dumps(generate_summary(), indent=2))
