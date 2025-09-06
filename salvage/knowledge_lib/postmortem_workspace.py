import os
import json
from collections import Counter

# Root directory for the workspace
ROOT = os.path.dirname(os.path.abspath(__file__))

summary = {
    'folders': [],
    'files': [],
    'file_types': Counter(),
    'key_files': [],
    'notebooks': [],
    'python_scripts': [],
    'text_files': [],
    'other_files': [],
}

def walk_workspace(root):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == '.':
            rel_dir = ''
        for d in dirnames:
            summary['folders'].append(os.path.join(rel_dir, d))
        for f in filenames:
            file_path = os.path.join(rel_dir, f)
            summary['files'].append(file_path)
            ext = os.path.splitext(f)[1].lower()
            summary['file_types'][ext] += 1
            if ext in ['.ipynb']:
                summary['notebooks'].append(file_path)
            elif ext in ['.py']:
                summary['python_scripts'].append(file_path)
            elif ext in ['.txt', '.md']:
                summary['text_files'].append(file_path)
            else:
                summary['other_files'].append(file_path)
            if f.lower().startswith(('readme', 'main', 'setup', 'requirements', 'license')):
                summary['key_files'].append(file_path)

def main():
    walk_workspace(ROOT)
    # Save summary as JSON
    with open('workspace_postmortem_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    # Print summary to console
    print('Folders:', len(summary['folders']))
    print('Files:', len(summary['files']))
    print('File types:', dict(summary['file_types']))
    print('Key files:', summary['key_files'])
    print('Notebooks:', summary['notebooks'])
    print('Python scripts:', summary['python_scripts'])
    print('Text files:', summary['text_files'])
    print('Other files:', summary['other_files'])

if __name__ == '__main__':
    main()
