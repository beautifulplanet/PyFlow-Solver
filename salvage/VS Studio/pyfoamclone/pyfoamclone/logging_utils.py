import logging
import os
from typing import Optional

_LOGGER: Optional[logging.Logger] = None

def get_logger() -> logging.Logger:
    """Return module-level logger singleton respecting env log level.

    Fixes previous bug where local 'logger' variable was referenced when
    _LOGGER already initialized (UnboundLocalError)."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("pyfoamclone")
        if not _LOGGER.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            _LOGGER.addHandler(handler)
    level_name = os.getenv('PYFOAMCLONE_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    _LOGGER.setLevel(level)
    return _LOGGER
