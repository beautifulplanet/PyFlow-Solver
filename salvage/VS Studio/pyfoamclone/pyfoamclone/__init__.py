"""pyfoamclone CFD toolkit (prototype).

Version metadata and lightweight run manifest helper for reproducibility.
"""

__all__ = ["__version__", "build_run_manifest"]

__version__ = "0.0.1-dev"

def build_run_manifest(config: dict, extra: dict | None = None) -> dict:
	import os, hashlib, json, datetime, platform, subprocess
	cfg_json = json.dumps(config, sort_keys=True).encode()
	cfg_hash = hashlib.sha256(cfg_json).hexdigest()[:12]
	try:
		git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
	except Exception:  # pragma: no cover
		git_sha = "unknown"
	manifest = {
		"version": __version__,
		"git_sha": git_sha,
		"config_hash": cfg_hash,
		"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		"python_version": platform.python_version(),
		"platform": platform.platform(),
		"pid": os.getpid(),
	}
	if extra:
		manifest.update(extra)
	return manifest
