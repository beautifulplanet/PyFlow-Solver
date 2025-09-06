from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict
import math
import numpy as np


@dataclass(slots=True)
class ResidualSeries:
    name: str
    values: List[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        self.values.append(float(v))

    def last(self) -> float:
        return self.values[-1] if self.values else math.inf

    def drop_orders(self) -> float:
        if not self.values:
            return 0.0
        first = self.values[0]
        last = self.values[-1]
        if first <= 0 or last <= 0:
            return 0.0
        return math.log10(first/last)


class ResidualManager:
    """Unified residual tracking (supersedes ResidualTracker).

    Provides both the original manager API (track/plateau) and the simpler
    tracker API (add/last/slope/plateau_detect) so existing code & tests keep
    working while we converge on one implementation.
    """
    def __init__(self):
        self.series: Dict[str, ResidualSeries] = {}

    # --- Primary add/track methods ---
    def track(self, name: str, value: float) -> None:
        self.series.setdefault(name, ResidualSeries(name)).add(value)

    def add(self, name: str, value: float) -> None:  # compatibility with ResidualTracker
        self.track(name, value)

    # --- Query helpers ---
    def last(self, name: str) -> float:
        s = self.series.get(name)
        return s.last() if s else math.inf

    def _geom_mean(self, values: List[float]) -> float:
        prod = 1.0
        count = 0
        for x in values:
            if x > 0:
                prod *= x
                count += 1
        if count == 0:
            return math.inf
        return prod ** (1.0 / count)

    def plateau(self, name: str, window: int = 100, slope_threshold: float = -0.01) -> bool:
        """Detect plateau based on recent log-slope over the latter half of the window.

        A plateau is declared if the log10-slope of the most recent half-window
        is greater than a small negative threshold (i.e., near-flat or increasing).
        """
        s = self.series.get(name)
        if not s or len(s.values) < window:
            return False
        recent = s.values[-window:]
        half = window // 2
        second_half = [v for v in recent[half:] if v > 0]
        n = len(second_half)
        if n < 3:
            return False
        x = np.arange(n, dtype=float)
        y = np.log10(np.array(second_half, dtype=float))
        # Linear regression slope of y(x)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2))
        if denom <= 0:
            return False
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        return bool(slope > slope_threshold)

    # --- Slope-based API (mirrors old ResidualTracker) ---
    def slope(self, name: str, window: int = 20) -> float:
        s = self.series.get(name)
        if not s or len(s.values) < window:
            return 0.0
        seg = np.array(s.values[-window:], dtype=float)
        if np.any(seg <= 0):
            return 0.0
        x = np.arange(seg.size, dtype=float)
        y = np.log10(seg)
        # simple linear regression using numpy polyfit
        a, _b = np.polyfit(x, y, 1)
        return float(a)

    def plateau_detect(self, name: str, window: int = 100, threshold: float = -0.01) -> bool:
        return bool(self.slope(name, window) > threshold)

    def summary(self) -> Dict[str, float]:
        return {k: v.last() for k, v in self.series.items()}
