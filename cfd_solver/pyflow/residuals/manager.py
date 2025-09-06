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
        first = self.values[0]; last = self.values[-1]
        if first <= 0 or last <= 0:
            return 0.0
        return math.log10(first/last)

class ResidualManager:
    def __init__(self):
        self.series: Dict[str, ResidualSeries] = {}
    def track(self, name: str, value: float) -> None:
        self.series.setdefault(name, ResidualSeries(name)).add(value)
    def add(self, name: str, value: float) -> None:
        self.track(name, value)
    def last(self, name: str) -> float:
        s = self.series.get(name); return s.last() if s else math.inf
    def slope(self, name: str, window: int = 20) -> float:
        s = self.series.get(name)
        if not s or len(s.values) < window:
            return 0.0
        seg = np.array(s.values[-window:], dtype=float)
        if np.any(seg <= 0):
            return 0.0
        x = np.arange(seg.size, dtype=float); y = np.log10(seg)
        a, _b = np.polyfit(x, y, 1)
        return float(a)
    def plateau_detect(self, name: str, window: int = 100, threshold: float = -0.01) -> bool:
        return bool(self.slope(name, window) > threshold)
    def summary(self) -> Dict[str, float]:
        return {k: v.last() for k, v in self.series.items()}

__all__ = ['ResidualManager', 'ResidualSeries']
