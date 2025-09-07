"""Configuration model & deterministic hashing utilities.

This module centralizes the logic for producing a *stable* configuration hash
used in checkpoints and restart compatibility enforcement. Only *semantic*
simulation parameters must influence the hash. Ephemeral/runtime fields that
can legitimately vary between runs (logging destinations, quiet flags, seeds,
progress toggles, injected callbacks, etc.) are **excluded** so they do not
invalidate reproducibility or restart matching.

Contract (v1):
  * Included: numeric / string scalar parameters that change numerical
	behavior or discretization (nx, ny, Re, lid_velocity, CFL targets, solver
	tolerances, scheme identifiers, geometry extents, boolean feature toggles
	that alter math such as disable_advection).
  * Excluded: force_quiet, log_path, enable_jacobi_pc, assert_invariants,
	seed, progress, any attribute whose name starts with an underscore, and
	any attribute whose value is callable or a module/type object.
  * Ordering: keys sorted lexicographically before hashing.
  * Serialization: JSON with separators (',',':') for canonical form.

Adding a new config attribute that affects solver mathematics MUST update the
regression test in tests/test_config_hash_regression.py if it should be
included (i.e. remove it from EXCLUDED_RUNTIME_FIELDS if present). Runtime
only additions should be added to EXCLUDED_RUNTIME_FIELDS.
"""

from __future__ import annotations

from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Iterable
import json, hashlib, inspect

# Central list of runtime / non-semantic fields intentionally excluded.
EXCLUDED_RUNTIME_FIELDS: set[str] = {
	"force_quiet",
	"log_path",
	"enable_jacobi_pc",
	"assert_invariants",
	"seed",  # seeding reproducibility handled separately; not part of math config
	"progress",  # CLI presentation concern
}

def _is_hashable_value(v: Any) -> bool:
	"""Return True if value is acceptable for inclusion (not a function/module/type)."""
	if callable(v):
		return False
	if inspect.ismodule(v) or inspect.isclass(v):  # pragma: no cover (defensive)
		return False
	return True

def config_core_dict(cfg: Any) -> Dict[str, Any]:
	"""Extract a dict of semantic configuration fields from an arbitrary object.

	Supports dataclasses or simple attribute containers. Filters excluded names
	and non-hashable runtime objects.
	"""
	if is_dataclass(cfg) and not isinstance(cfg, type):
		# asdict only on instance, not class objects
		raw = asdict(cfg)
	elif hasattr(cfg, "__dict__"):
		raw = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
	else:  # fallback to dir() inspection
		raw = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
	core: Dict[str, Any] = {}
	for k, v in raw.items():
		if k in EXCLUDED_RUNTIME_FIELDS:
			continue
		if not _is_hashable_value(v):
			continue
		core[k] = v
	return core

def config_hash(cfg: Any) -> str:
	"""Compute deterministic short hash of semantic config fields.

	Returns first 10 hex chars of SHA1 of canonical JSON encoding.
	"""
	core = config_core_dict(cfg)
	blob = json.dumps(core, sort_keys=True, separators=(",", ":"), default=str).encode()
	return hashlib.sha1(blob).hexdigest()[:10]

__all__ = [
	"config_hash",
	"config_core_dict",
	"EXCLUDED_RUNTIME_FIELDS",
]

