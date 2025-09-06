
"""Lightweight provenance + structured run logging.

Creates a unique run directory containing a JSON metadata file with:
  project_name, run_id, timestamp_utc, cwd, python_version, platform,
  git_sha (if available), config_hash (optional), env_project_name, argv.

Usage:
	from framework.logging import init_run
	run = init_run(project_name="cfdmini", log_dir="logs")
	run.log_event(level="INFO", msg="starting", extra={"grids": [17,33,65]})

JSON Lines log file: events.jsonl (one dict per line).
Meta file: run_meta.json
"""
from __future__ import annotations
import os, sys, uuid, json, datetime, hashlib, platform, subprocess, threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

_lock = threading.Lock()

def _git_sha() -> Optional[str]:
	try:
		return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
	except Exception:
		return None

def _sanitize(name: str) -> str:
	return ''.join(c if c.isalnum() or c in ('-','_') else '_' for c in name) or 'project'

def _now_iso() -> str:
	return datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'

def hash_config(config: Any) -> Optional[str]:
	if config is None:
		return None
	try:
		blob = json.dumps(config, sort_keys=True).encode('utf-8')
		return hashlib.sha256(blob).hexdigest()[:16]
	except Exception:
		return None

@dataclass
class RunMeta:
	project_name: str
	run_id: str
	timestamp_utc: str
	cwd: str
	python_version: str
	platform: str
	git_sha: Optional[str]
	env_project_name: Optional[str]
	argv: list
	config_hash: Optional[str] = None

class RunLogger:
	def __init__(self, meta: RunMeta, run_dir: str):
		self.meta = meta
		self.run_dir = run_dir
		self._events_path = os.path.join(run_dir, 'events.jsonl')
		self._opened = True
		# Write meta
		with open(os.path.join(run_dir, 'run_meta.json'), 'w', encoding='utf-8') as f:
			json.dump(asdict(meta), f, indent=2)

	def log_event(self, level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
		if not self._opened:
			return
		rec = {
			"ts": _now_iso(),
			"level": level.upper(),
			"project": self.meta.project_name,
			"run_id": self.meta.run_id,
			"msg": msg,
		}
		if extra:
			rec.update(extra)
		line = json.dumps(rec, separators=(',', ':'))
		with _lock:
			with open(self._events_path, 'a', encoding='utf-8') as f:
				f.write(line + '\n')

	def close(self):
		self._opened = False

def init_run(project_name: Optional[str] = None, log_dir: str = 'logs', config: Any = None) -> RunLogger:
	env_name = os.environ.get('PROJECT_NAME')
	project = project_name or env_name or os.path.basename(os.getcwd()) or 'project'
	project = _sanitize(project)
	run_id = str(uuid.uuid4())[:8]
	date_part = datetime.date.today().isoformat()
	run_dir = os.path.join(log_dir, project, date_part, f'run_{run_id}')
	os.makedirs(run_dir, exist_ok=True)
	meta = RunMeta(
		project_name=project,
		run_id=run_id,
		timestamp_utc=_now_iso(),
		cwd=os.path.abspath(os.getcwd()),
		python_version=sys.version.split()[0],
		platform=platform.platform(),
		git_sha=_git_sha(),
		env_project_name=env_name,
		argv=sys.argv,
		config_hash=hash_config(config),
	)
	return RunLogger(meta, run_dir)
