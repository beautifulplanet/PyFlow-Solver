#!/usr/bin/env python
"""Append a structured prompt/response entry to the canonical conversation log.

Usage (PowerShell example):
  python scripts/log_conversation.py --prompt "User text" --response "AI text"

Both --prompt and --response accept either raw text or an @file reference:
  python scripts/log_conversation.py --prompt @prompt.txt --response @reply.txt

Output file (default): conversation_logs/conversation_log.md
Override directory with env var CONVERSATION_LOG_DIR.
"""
from __future__ import annotations
import argparse, datetime as _dt, pathlib, sys, os, textwrap, json, errno, time

DEFAULT_DIR = pathlib.Path(os.environ.get("CONVERSATION_LOG_DIR", "conversation_logs"))
LOG_FILE = DEFAULT_DIR / "conversation_log.md"
LOCK_FILE = DEFAULT_DIR / ".conversation_log.lock"

def _read_arg(value: str) -> str:
    if value.startswith('@') and len(value) > 1:
        path = pathlib.Path(value[1:])
        return path.read_text(encoding='utf-8')
    return value

def _normalize(text: str) -> str:
    # Normalize newlines and trim trailing spaces (preserve intentional blank lines)
    lines = [ln.rstrip() for ln in text.replace('\r\n', '\n').replace('\r', '\n').split('\n')]
    return '\n'.join(lines).strip()  # strip leading/trailing overall

def _acquire_lock(timeout: float = 10.0, poll: float = 0.1):
    start = time.time()
    while True:
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode('utf-8'))
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Could not acquire log lock within {timeout}s: {LOCK_FILE}")
            time.sleep(poll)

def _release_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

def append_entry(prompt: str, response: str, role_prompt: str, role_response: str):
    timestamp = _dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    block = [
        f"### [{timestamp}]",
        f"**{role_prompt}:**",
        prompt,
        "",
        f"**{role_response}:**",
        response,
        "\n---\n"
    ]
    entry = '\n'.join(block)
    _acquire_lock()
    try:
        with LOG_FILE.open('a', encoding='utf-8') as f:
            # First time header
            if LOG_FILE.stat().st_size == 0:
                f.write("# Conversation Log\n\n")
            f.write(entry)
            if not entry.endswith('\n'):
                f.write('\n')
    finally:
        _release_lock()
    return entry

def main(argv=None):
    p = argparse.ArgumentParser(description="Append a prompt/response pair to the conversation log")
    p.add_argument('--prompt', required=True, help='User prompt text or @file')
    p.add_argument('--response', required=True, help='AI response text or @file')
    p.add_argument('--role-prompt', default='User Prompt', help='Header label for prompt section')
    p.add_argument('--role-response', default='AI Response', help='Header label for response section')
    p.add_argument('--dry-run', action='store_true', help='Print entry without appending')
    args = p.parse_args(argv)

    prompt_raw = _read_arg(args.prompt)
    response_raw = _read_arg(args.response)
    prompt = _normalize(prompt_raw)
    response = _normalize(response_raw)
    if args.dry_run:
        print("(dry-run) would append:\n")
        print(append_entry(prompt, response, args.role_prompt, args.role_response))
    else:
        entry = append_entry(prompt, response, args.role_prompt, args.role_response)
        print(f"Appended entry to {LOG_FILE} ({len(entry.splitlines())} lines)")

if __name__ == '__main__':  # pragma: no cover
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
