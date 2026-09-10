"""
Headless mobility primitives.

Kept in sim11ah/, not ui/, on purpose: mobility today is GUI-only
(ui.topology_canvas.advance_uav_positions is driven from exactly one place,
the Tk dashboard's tick loop), but the multi-AP roaming experiment
(scripts/roam_experiment.py) needs controllable STA motion with no GUI at
all. A future GUI tick loop can drive this same primitive instead of
duplicating the stepping logic.
"""
from __future__ import annotations

import math
from typing import Tuple


def corridor_step(
    sim,
    sta_id: int,
    dt: float,
    speed_mps: float,
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
) -> bool:
    """
    Advance sta_id along the straight line from start_pos to end_pos at a
    constant speed_mps, moving it speed_mps * dt further this call.

    Progress is derived from the node's CURRENT position (distance already
    covered from start_pos), not separate tracked state -- so this can be
    called repeatedly, interleaved with sim.run_for(dt), without the caller
    needing to pass elapsed time/distance back in each time. Clamps at
    end_pos once reached; further calls after that are a no-op.

    Returns True while still moving, False once it has arrived, so a
    driving loop knows when the crossing is complete.
    """
    x0, y0 = float(start_pos[0]), float(start_pos[1])
    x1, y1 = float(end_pos[0]), float(end_pos[1])
    total_dx, total_dy = x1 - x0, y1 - y0
    total_dist = math.hypot(total_dx, total_dy)

    if total_dist <= 1e-9:
        sim.nodes[sta_id].pos = (x1, y1)
        return False

    cur_x, cur_y = sim.nodes[sta_id].pos
    traveled = math.hypot(cur_x - x0, cur_y - y0)
    new_traveled = min(total_dist, traveled + max(0.0, float(speed_mps)) * float(dt))

    frac = new_traveled / total_dist
    sim.nodes[sta_id].pos = (x0 + total_dx * frac, y0 + total_dy * frac)

    return new_traveled < total_dist
