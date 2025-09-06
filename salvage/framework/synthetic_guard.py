"""Synthetic path guard.

Goal: prevent accidental reliance on synthetic/injected profiles in production mode.

Usage:
    from synthetic_guard import guard_synthetic
    guard_synthetic(enabled, context="lid_cavity_profile")

If SYNTHETIC_KILL=1 environment variable set, any synthetic usage raises.
"""
from __future__ import annotations
import os

class SyntheticUsageError(RuntimeError):
    pass

def guard_synthetic(is_synthetic: bool, context: str = ""):
    if not is_synthetic:
        return
    kill = os.environ.get("SYNTHETIC_KILL", "0") == "1"
    if kill:
        raise SyntheticUsageError(f"Synthetic code path blocked (context={context})")
    # Else: emit minimal diagnostic (could be upgraded to logging framework later)
    print(f"[SYNTHETIC WARNING] context={context} (set SYNTHETIC_KILL=1 to forbid)")
