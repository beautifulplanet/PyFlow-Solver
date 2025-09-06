from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass(slots=True)
class ResidualTracker:
    history: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.history.setdefault(name, []).append(float(value))

    def last(self, name: str) -> float:
        vals = self.history.get(name)
        return vals[-1] if vals else math.inf

    def slope(self, name: str, window: int = 20) -> float:
        vals = self.history.get(name, [])
        if len(vals) < window:
            return 0.0
        seg = np.array(vals[-window:])
        if np.any(seg <= 0):
            return 0.0
        x = np.arange(window)
        y = np.log10(seg)
        a, _b = np.polyfit(x, y, 1)
        return float(a)

    def plateau_detect(self, name: str, window: int = 100, threshold: float = -0.01) -> bool:
        s = self.slope(name, window)
        return bool(s > threshold)
