import os
import json
from pathlib import Path

# This script will recursively scan the workspace, read all files (with size limits), and summarize their contents.
# It will output:
# - A file inventory (all files, types, sizes)
# - Summaries of code, notebooks, and text files
# - Key findings about AI code generation for CFD, successes, and failures

ROOT = Path(__file__).parent
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB per file limit for reading
SUMMARY = {
    'total_files': 0,
    'file_types': {},
    'files': [],
    'code_summaries': [],
    'notebook_summaries': [],
    'text_summaries': [],
    'large_files': [],
}

def summarize_file(path):
    ext = path.suffix.lower()
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            SUMMARY['large_files'].append(str(path))
            return None
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if ext in ['.py', '.cpp', '.c', '.h', '.js', '.ts']:
            # Summarize code: count functions, classes, main blocks
            func_count = content.count('def ') + content.count('function ')
            class_count = content.count('class ')
            main_count = content.count('__main__')
            SUMMARY['code_summaries'].append({
                'file': str(path),
                'functions': func_count,
                'classes': class_count,
                'main_blocks': main_count,
                'size': path.stat().st_size,
            })
        elif ext == '.ipynb':
            # Summarize notebook: count cells, look for keywords
            try:
                try:
                    import nbformat
                except ImportError:
                    SUMMARY['notebook_summaries'].append({
                        'file': str(path),
                        'cells': 'nbformat not installed',
                        'size': path.stat().st_size,
                    })
                    return
                nb = nbformat.reads(content, as_version=4)
                cell_count = len(nb.cells)
                SUMMARY['notebook_summaries'].append({
                    'file': str(path),
                    'cells': cell_count,
                    'size': path.stat().st_size,
                })
            except Exception:
                SUMMARY['notebook_summaries'].append({
                    'file': str(path),
                    'cells': 'error',
                    'size': path.stat().st_size,
                })
        elif ext in ['.txt', '.md', '.rst']:
            # Summarize text: count lines, look for keywords
            lines = content.splitlines()
            keywords = ['fail', 'success', 'solver', 'CFD', 'AI', 'Copilot', 'experiment', 'result']
            found = [k for k in keywords if any(k in line for line in lines)]
            SUMMARY['text_summaries'].append({
                'file': str(path),
                'lines': len(lines),
                'keywords_found': found,
                'size': path.stat().st_size,
            })
    except Exception as e:
        SUMMARY['text_summaries'].append({
            'file': str(path),
            'error': str(e),
        })

def scan_workspace(root):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            SUMMARY['total_files'] += 1
            ext = fpath.suffix.lower()
            SUMMARY['file_types'][ext] = SUMMARY['file_types'].get(ext, 0) + 1
            SUMMARY['files'].append(str(fpath))
            summarize_file(fpath)

def main():
    scan_workspace(ROOT)
    with open('workspace_full_inventory.json', 'w', encoding='utf-8') as f:
        json.dump(SUMMARY, f, indent=2)
    print(f"Scanned {SUMMARY['total_files']} files. See workspace_full_inventory.json for details.")

if __name__ == '__main__':
    main()
