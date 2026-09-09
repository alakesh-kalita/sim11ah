"""Physical obstacles that attenuate radio propagation.

Purely opt-in: PhyLayer._obstruction_loss_db() only consults a Simulator's
``obstacles`` attribute if present and non-empty (``getattr(sim,
"obstacles", None)``). No existing evaluation script sets this, so this
module can never change a previously-published result -- it only takes
effect for callers (currently: the interactive GUI) that explicitly
populate ``sim.obstacles`` with a list of Obstacle instances describing
buildings, terrain, etc. in the same (x, y) metre coordinate system as
``Node.pos``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Obstacle:
    """A physical object whose footprint attenuates any TX-RX link whose
    straight-line path crosses it.

    kind: "rect" (axis-aligned, e.g. a building footprint) or "circle"
    (e.g. a hill/mountain treated as a circular attenuating region).
    loss_db: extra one-way path loss (dB) added when a link crosses this
    obstacle. Multiple crossed obstacles stack additively.
    """
    kind: str
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0
    loss_db: float = 15.0
    label: str = ""

    def intersects_segment(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        if self.kind == "circle":
            return _segment_circle_intersect(p0, p1, (self.cx, self.cy), self.r)
        return _segment_rect_intersect(p0, p1, self.x0, self.y0, self.x1, self.y1)


def _segment_rect_intersect(
    p0: Tuple[float, float], p1: Tuple[float, float],
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    """Liang-Barsky segment-vs-axis-aligned-box intersection test."""
    rx0, rx1 = min(rx0, rx1), max(rx0, rx1)
    ry0, ry1 = min(ry0, ry1), max(ry0, ry1)
    x1, y1 = p0
    x2, y2 = p1
    dx, dy = x2 - x1, y2 - y1
    tmin, tmax = 0.0, 1.0
    for p, q in ((-dx, x1 - rx0), (dx, rx1 - x1), (-dy, y1 - ry0), (dy, ry1 - y1)):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        t = q / p
        if p < 0.0:
            if t > tmax:
                return False
            if t > tmin:
                tmin = t
        else:
            if t < tmin:
                return False
            if t < tmax:
                tmax = t
    return tmin <= tmax


def _segment_circle_intersect(
    p0: Tuple[float, float], p1: Tuple[float, float],
    c: Tuple[float, float], r: float,
) -> bool:
    x1, y1 = p0
    x2, y2 = p1
    cx, cy = c
    dx, dy = x2 - x1, y2 - y1
    len2 = dx * dx + dy * dy
    if len2 < 1e-9:
        return math.hypot(x1 - cx, y1 - cy) <= r
    t = ((cx - x1) * dx + (cy - y1) * dy) / len2
    t = max(0.0, min(1.0, t))
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy) <= r
