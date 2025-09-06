"""Export knowledge DB & learning dataset into SQLite for advanced querying.
Tables:
  functions(id PRIMARY KEY, name, file, quality, complexity, length, snippet)
  files(path PRIMARY KEY, labels, evolution_tag, keywords, fail_reasons)
  vocab(term PRIMARY KEY, tag_freq INT, name_token_freq INT)
"""
from __future__ import annotations
import json, sqlite3
from pathlib import Path

ROOT = Path(__file__).parent
DB = ROOT/'cfd_knowledge.db'
KB = ROOT/'knowledge_db'

def load_jsonl(p: Path):
    out=[]
    if not p.exists(): return out
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        try: out.append(json.loads(line))
        except: pass
    return out

def main():
    funcs = load_jsonl(KB/'functions.jsonl')
    files = load_jsonl(KB/'files.jsonl')
    vocab_data = json.loads((KB/'vocab.json').read_text(encoding='utf-8')) if (KB/'vocab.json').exists() else {}
    if DB.exists():
        DB.unlink()
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('CREATE TABLE functions (id TEXT PRIMARY KEY, name TEXT, file TEXT, quality REAL, complexity INT, length INT, snippet TEXT)')
    c.execute('CREATE TABLE files (path TEXT PRIMARY KEY, labels TEXT, evolution_tag TEXT, keywords TEXT, fail_reasons TEXT)')
    c.execute('CREATE TABLE vocab (term TEXT PRIMARY KEY, tag_freq INT, name_token_freq INT)')
    for f in funcs:
        c.execute('INSERT INTO functions VALUES (?,?,?,?,?,?,?)', (
            f.get('id'), f.get('name'), f.get('file'), f.get('quality'), f.get('complexity'), f.get('length'), f.get('snippet')[:4000]))
    for fi in files:
        c.execute('INSERT INTO files VALUES (?,?,?,?,?)', (
            fi.get('path'), ','.join(fi.get('labels',[])), fi.get('evolution_tag'), ','.join(fi.get('keyword_hits',[])), '|'.join(fi.get('fail_reasons',[]))))
    tag_freq = {t:f for t,f in vocab_data.get('tag_frequency',[])}
    name_freq = {t:f for t,f in vocab_data.get('name_token_frequency',[])}
    terms = set(tag_freq)|set(name_freq)
    for term in terms:
        c.execute('INSERT INTO vocab VALUES (?,?,?)', (term, tag_freq.get(term,0), name_freq.get(term,0)))
    conn.commit()
    conn.close()
    print('SQLite DB created at', DB)

if __name__ == '__main__':
    main()
