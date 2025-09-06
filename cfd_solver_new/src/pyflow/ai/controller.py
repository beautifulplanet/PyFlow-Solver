from __future__ import annotations
"""High-level AI controller utilities.

Bridges natural language parsing to CLI execution (subprocess) using the
stable JSON stream for telemetry. This is intentionally lightweight; an
actual LLM invocation can be swapped into `interpret()` later.
"""
import subprocess, sys, json, threading, queue
from typing import Iterable, Dict, Any, Iterator
from .schema import SimulationRequest
from .nl_parser import parse_natural_language

class SimulationHandle:
    def __init__(self, process: subprocess.Popen, line_queue: "queue.Queue[str]"):
        self.process = process
        self._queue = line_queue
    def iter_json(self) -> Iterator[Dict[str, Any]]:
        while self.process.poll() is None or not self._queue.empty():
            try:
                line = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
    def wait(self) -> int:
        return self.process.wait()


def interpret(natural_text: str) -> SimulationRequest:
    """Convert free-form text into a structured request."""
    return parse_natural_language(natural_text)


def launch(req: SimulationRequest) -> SimulationHandle:
    """Launch the simulation as a CLI subprocess and return a handle."""
    args = [sys.executable, '-m', 'pyflow.cli', *req.to_cli_args()]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    q: "queue.Queue[str]" = queue.Queue()

    def reader(stream):
        for line in stream:
            if line.strip():
                q.put(line.strip())
        stream.close()
    threading.Thread(target=reader, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=reader, args=(proc.stderr,), daemon=True).start()
    return SimulationHandle(proc, q)

__all__ = ["interpret", "launch", "SimulationHandle"]
