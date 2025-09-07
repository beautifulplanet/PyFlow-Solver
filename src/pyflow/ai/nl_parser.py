from __future__ import annotations

"""Natural language parsing for simulation requests.

A lightweight rule/heuristic driven parser (no external LLM call) that
extracts structured parameters from a free-form prompt.

Future extension: replace internals with actual LLM invocation while keeping
stable function contract.
"""
import re

from .schema import SimulationRequest

# Regex helpers
NUM = r"(?P<val>\d+(?:\.\d+)?)"
GRID_RE = re.compile(r"(?P<nx>\d+)\s*[xX]\s*(?P<ny>\d+)")
RE_RE = re.compile(r"Re\s*[= ]\s*(?P<re>\d+(?:\.\d+)?)", re.IGNORECASE)
STEPS_RE = re.compile(r"(steps?|iterations?)\s*[= ]\s*(?P<steps>\d+)", re.IGNORECASE)
# Patterns like 'for 300 steps'
FOR_STEPS_RE = re.compile(r"for\s+(?P<steps>\d+)\s+(?:steps?|iterations?)", re.IGNORECASE)
CFL_RE = re.compile(r"CFL\s*[= ]\s*(?P<cfl>\d+(?:\.\d+)?)", re.IGNORECASE)
VEL_RE = re.compile(r"lid(?:[-_ ]velocity)?\s*[= ]\s*(?P<vel>\d+(?:\.\d+)?)", re.IGNORECASE)
SCHEME_RE = re.compile(r"\b(upwind|quick)\b", re.IGNORECASE)
THRESH_RE = re.compile(r"continuity\s*<\s*(?P<th>\d+(?:\.\d+)?e?-?\d*)", re.IGNORECASE)

PROBLEM_KEYWORDS = {
    'lid': 'lid_cavity',
    'cavity': 'lid_cavity',
}

def parse_natural_language(text: str, *, default: SimulationRequest | None = None) -> SimulationRequest:
    req = SimulationRequest() if default is None else default
    t = text.strip()
    low = t.lower()

    # Problem detection
    for kw, prob in PROBLEM_KEYWORDS.items():
        if kw in low:
            req.problem = prob
            break

    # Grid size
    m = GRID_RE.search(t)
    if m:
        req.nx = int(m.group('nx'))
        req.ny = int(m.group('ny'))

    # Reynolds number
    m = RE_RE.search(t)
    if m:
        req.Re = float(m.group('re'))

    # Steps
    m = STEPS_RE.search(t)
    if m:
        req.steps = int(m.group('steps'))
    else:
        m2 = FOR_STEPS_RE.search(t)
        if m2:
            req.steps = int(m2.group('steps'))

    # CFL
    m = CFL_RE.search(t)
    if m:
        req.cfl = float(m.group('cfl'))

    # Lid velocity
    m = VEL_RE.search(t)
    if m:
        req.lid_velocity = float(m.group('vel'))

    # Scheme
    m = SCHEME_RE.search(t)
    if m:
        req.scheme = m.group(1).lower()

    # Continuity threshold (continuity < value)
    m = THRESH_RE.search(t)
    if m:
        val = m.group('th')
        try:
            req.continuity_threshold = float(val)
        except ValueError:
            pass

    # Diagnostics toggle if explicitly requested
    if 'diagnostics' in low or 'verbose' in low:
        req.diagnostics = True

    # JSON streaming opt-out
    if 'no json' in low or 'human output' in low:
        req.json_stream = False

    return req

__all__ = ["parse_natural_language"]
