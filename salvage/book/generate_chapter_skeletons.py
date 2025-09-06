"""Book Authoring Helper: Generate Chapter Skeletons (v0.1)
Reads TEXTBOOK_OUTLINE.md and creates chapter skeleton Markdown files under book/chapters with numbered filenames and headers.
"""
from __future__ import annotations
import os, re, io

OUTDIR='book/chapters'

CHAPTER_HEADER_TEMPLATE = """# {number}. {title}

Status: DRAFT

Learning Objectives:
- 
- 

Key Terms:
- 

Narrative:


Concepts:


Algorithms & Pseudocode:


Metrics & Validation:


Instrumentation & Observability:


Failure Points & Mitigations:


Case Study:


Exercises:


References:

"""

def parse_outline(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        text=f.read()
    # simple heuristic: lines starting with "### Part" and enumerated chapters below
    chapters=[]
    in_part=False
    for line in text.splitlines():
        m=re.match(r"^\s*\d+\.?\s+(.*)$", line)
        if line.strip().startswith('### Part'):
            in_part=True
            continue
        if in_part and m:
            title=m.group(1).strip()
            # Ignore bullets, keep clean title
            title=re.sub(r"\s*\(.*\)$","",title)
            chapters.append(title)
    return chapters

def safe_filename(i,title):
    base=re.sub(r"[^A-Za-z0-9_-]+","_", title)[:60].strip('_')
    return f"{i:02d}_{base}.md"

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    outline='TEXTBOOK_OUTLINE.md'
    if not os.path.isfile(outline):
        print('Outline not found; run from repo root')
        return
    chapters=parse_outline(outline)
    for i,title in enumerate(chapters, start=1):
        fname=os.path.join(OUTDIR, safe_filename(i,title))
        if not os.path.exists(fname):
            with open(fname,'w',encoding='utf-8') as f:
                f.write(CHAPTER_HEADER_TEMPLATE.format(number=i,title=title))
    print(f"Generated {len(chapters)} chapter skeletons in {OUTDIR}")

if __name__=='__main__':
    main()
