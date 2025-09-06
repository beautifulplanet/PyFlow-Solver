from __future__ import annotations

import json
import traceback
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict

from ..logging_utils import get_logger


def log_failure(stage: str, err: BaseException, context: Dict[str, Any] | None = None, path: str | None = None) -> None:
    rec = {
        "ts": datetime.now(UTC).isoformat(),
        "stage": stage,
        "error_type": type(err).__name__,
        "error": str(err),
        "trace": traceback.format_exc().splitlines(),
    }
    if context:
        rec["context"] = context
    line = json.dumps(rec)
    get_logger().error(line)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
