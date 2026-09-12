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


def grid_road_step(
    sim,
    sta_id: int,
    dt: float,
    speed_mps: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    init_axis: str = "x",
    init_dir: float = 1.0,
) -> None:
    """
    Drive a road vehicle continuously through a rectangular road grid --
    replaces the old highway_loop_step/cross_street_loop_step, which
    reset (teleported) a vehicle back to the start the instant it
    reached the far end of its lane. That reset was itself a deliberate
    replacement for an even older direction-reversing bounce, but got
    reported as still visibly discontinuous ("position gets reset") --
    this is the actual fix: a vehicle never resets and never reverses in
    place, it turns 90 degrees onto the perpendicular road exactly where
    its current one ends and keeps driving, forever.

    State (which axis it's currently travelling along, and which
    direction) lives in sim._grid_road_state[sta_id], seeded from
    init_axis/init_dir only the first time a given sta_id is seen -- every
    call after that ignores init_axis/init_dir entirely and just reads
    back whatever this function itself last decided, so a single call
    site can drive EVERY vehicle (avenue-started or cross-street-started
    alike) through the same turning logic without needing to track "is
    this still an avenue vehicle" itself; topology.py's CarsUavsBuilder
    only needs init_axis/init_dir to match how each vehicle was actually
    placed (avenue: init_axis="x", init_dir=sign(lane_y); cross street:
    init_axis="y", init_dir=sign(lane_offset) -- the same sign
    convention the old functions used).

    x_bounds/y_bounds are shared by every vehicle (the corridor's own
    x=[0, span] and y=[-outer_avenue, +outer_avenue] -- see
    dashboard_tk.py's _advance_drones) and must each coincide with a
    real road for the turn to happen with zero position jump: x_bounds
    needs a cross street at both ends (CarsUavsBuilder.build adds
    boundary cross streets at x=0/x=span for exactly this), y_bounds
    needs a real avenue at both ends (the outermost one, by
    construction). A vehicle that starts on an INNER avenue (not at
    +/-y_bounds) simply drives straight through the interior on its
    first cross-street leg without turning there -- turns only ever
    happen where the vehicle's OWN current road ends, matching how a
    real driver doesn't turn at every intersection it merely passes.

    The four turns always rotate the same way (clockwise, viewed with
    +x east/+y north: heading east -> turn south, heading south -> turn
    west, heading west -> turn north, heading north -> turn east) --
    "turn right" every time, a fixed, simple rule rather than needing to
    pick a direction at each corner. overflow distance (this tick's step
    minus however much room was left on the current road) carries onto
    the new road in the same call, so a vehicle never visibly pauses at
    a corner even at low tick rates -- capped at a few turns per call as
    a sanity bound, never expected to matter at any real speed/dt.
    """
    if not hasattr(sim, "_grid_road_state"):
        sim._grid_road_state = {}
    state = sim._grid_road_state.get(sta_id)
    if state is None:
        state = {"axis": init_axis, "dir": 1.0 if float(init_dir) >= 0.0 else -1.0}
        sim._grid_road_state[sta_id] = state

    node = sim.nodes[sta_id]
    x, y = node.pos
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    remaining = max(0.0, float(speed_mps)) * float(dt)

    for _ in range(4):
        if remaining <= 0.0:
            break
        if state["axis"] == "x":
            if state["dir"] > 0.0:
                room = x_max - x
                if remaining < room:
                    x += remaining
                    remaining = 0.0
                else:
                    remaining -= room
                    x = x_max
                    state["axis"], state["dir"] = "y", -1.0
            else:
                room = x - x_min
                if remaining < room:
                    x -= remaining
                    remaining = 0.0
                else:
                    remaining -= room
                    x = x_min
                    state["axis"], state["dir"] = "y", 1.0
        else:
            if state["dir"] > 0.0:
                room = y_max - y
                if remaining < room:
                    y += remaining
                    remaining = 0.0
                else:
                    remaining -= room
                    y = y_max
                    state["axis"], state["dir"] = "x", 1.0
            else:
                room = y - y_min
                if remaining < room:
                    y -= remaining
                    remaining = 0.0
                else:
                    remaining -= room
                    y = y_min
                    state["axis"], state["dir"] = "x", -1.0

    node.pos = (x, y)


def uav_bounce_step(
    sim,
    sta_id: int,
    dt: float,
    speed_mps: float,
    region: Tuple[float, float, float, float],
) -> None:
    """
    Fly a UAV in a continuous straight line, reflecting its heading off
    whichever edge of `region` (x_min, y_min, x_max, y_max) it reaches --
    replaces the old uav_waypoint_step, which flew toward a random point
    and picked a fresh, independent random target the instant it arrived
    (reported as looking like the same kind of discontinuity as the
    cars' old position-reset, since the heading could flip to any new
    direction with no relation to the one just flown). A reflection is
    always continuous in position and, for anything but a dead-on
    perpendicular hit, reads as exactly the "turn" a vehicle reaching the
    end of its road takes -- just applied to free flight instead of a
    fixed road grid, since a UAV isn't confined to one.

    Heading is a persistent unit vector in sim._uav_bounce_heading
    (seeded once per sta_id from sim.engine.rng -- the simulator's own
    seeded RNG, so flight paths stay reproducible for a given seed like
    everything else here); every call after that just keeps flying that
    heading, reflecting one or both components on hitting a wall. Not
    anchored to any single AP: pass a region spanning the whole multi-AP
    corridor (see topology.py's CarsUavsBuilder.uav_region) so a UAV
    naturally wanders across every AP's coverage over time.
    """
    if not hasattr(sim, "_uav_bounce_heading"):
        sim._uav_bounce_heading = {}

    x_min, y_min, x_max, y_max = region
    node = sim.nodes[sta_id]
    x, y = node.pos

    heading = sim._uav_bounce_heading.get(sta_id)
    if heading is None:
        angle = sim.engine.rng.random() * 2.0 * math.pi
        heading = (math.cos(angle), math.sin(angle))

    hx, hy = heading
    step = max(0.0, float(speed_mps)) * float(dt)
    new_x = x + hx * step
    new_y = y + hy * step

    if new_x > x_max:
        new_x = x_max - (new_x - x_max)
        hx = -hx
    elif new_x < x_min:
        new_x = x_min + (x_min - new_x)
        hx = -hx
    if new_y > y_max:
        new_y = y_max - (new_y - y_max)
        hy = -hy
    elif new_y < y_min:
        new_y = y_min + (y_min - new_y)
        hy = -hy

    sim._uav_bounce_heading[sta_id] = (hx, hy)
    node.pos = (new_x, new_y)
