class PyFoamCloneError(Exception):
    """Base exception for pyfoamclone."""

class InvalidGridError(PyFoamCloneError):
    """Raised when grid dimensions or extents are invalid."""

class InvalidParameterError(PyFoamCloneError):
    """Raised on invalid user parameters (e.g., negative Reynolds)."""
