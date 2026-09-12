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
    x_stops: Tuple[float, ...],
    y_stops: Tuple[float, ...],
    init_axis: str = "x",
    init_dir: float = 1.0,
    turn_prob: float = 0.35,
) -> None:
    """
    Drive a road vehicle continuously through a real road GRID -- replaces
    the old highway_loop_step/cross_street_loop_step, which reset
    (teleported) a vehicle back to the start the instant it reached the
    far end of its lane. That reset was itself a deliberate replacement
    for an even older direction-reversing bounce, but got reported as
    still visibly discontinuous ("position gets reset") -- so a vehicle
    here never resets and never reverses in place, it turns 90 degrees
    onto a perpendicular road instead.

    First version of this function always turned the same way (a fixed
    "turn right" rule) and only ever turned at the outermost road on
    each axis -- every vehicle's path converged onto the exact same
    shared rectangle after its first lap, reported back as "all the
    vehicles are following the same pattern... should be random". Fixed
    here: x_stops/y_stops are every real cross-street x / avenue y a
    vehicle on this axis will actually pass over (not just the two
    outermost), and at each one reached mid-road there's a turn_prob
    chance of turning -- onto a RANDOMLY chosen direction (left or
    right, 50/50) rather than always the same way -- instead of always
    continuing straight. Reaching the outermost stop in the current
    direction of travel (nothing further to drive to) still forces a
    turn, same as before, just now picked randomly too. Uses
    sim.engine.rng (the simulator's own seeded RNG), not Python's global
    random module, so traffic patterns stay reproducible for a given
    seed like everything else in this codebase.

    State (current axis + direction) lives in sim._grid_road_state[sta_id],
    seeded from init_axis/init_dir only the first time a given sta_id is
    seen -- every call after that ignores init_axis/init_dir entirely and
    just reads back whatever this function itself last decided, so one
    call site can drive EVERY vehicle (avenue-started or cross-street-
    started alike) through the same logic without tracking "is this
    still an avenue vehicle" itself; topology.py's CarsUavsBuilder only
    needs init_axis/init_dir to match how each vehicle was actually
    placed (avenue: init_axis="x", init_dir=sign(lane_y); cross street:
    init_axis="y", init_dir=sign(lane_offset)).

    Every value in x_stops must coincide with a real cross street, every
    value in y_stops with a real avenue (CarsUavsBuilder.build adds
    boundary cross streets at x=0/x=span for exactly this, so the
    corridor's own edges are real turn points too) -- otherwise a turn
    would jump the vehicle onto a road that doesn't exist there. A
    vehicle currently on an axis whose OWN position isn't itself one of
    these stops (e.g. an avenue vehicle riding an inner avenue y that
    isn't in y_stops) just isn't offered a turn on that leg at all --
    only entering a leg exactly AT a stop (which every turn, by
    construction, does) makes turning possible there.
    """
    if not hasattr(sim, "_grid_road_state"):
        sim._grid_road_state = {}
    state = sim._grid_road_state.get(sta_id)
    if state is None:
        state = {"axis": init_axis, "dir": 1.0 if float(init_dir) >= 0.0 else -1.0}
        sim._grid_road_state[sta_id] = state

    node = sim.nodes[sta_id]
    x, y = node.pos
    xs = sorted(float(v) for v in x_stops)
    ys = sorted(float(v) for v in y_stops)
    rng = sim.engine.rng
    remaining = max(0.0, float(speed_mps)) * float(dt)

    for _ in range(4):
        if remaining <= 0.0:
            break
        if state["axis"] == "x":
            forward = state["dir"] > 0.0
            nxt = min((s for s in xs if s > x + 1e-9), default=None) if forward \
                else max((s for s in xs if s < x - 1e-9), default=None)
            if nxt is None:
                # Already at (or past, from float error) the outermost
                # stop with nowhere further to go -- force a turn in
                # place rather than drive off the edge of the grid.
                state["axis"] = "y"
                state["dir"] = 1.0 if rng.random() < 0.5 else -1.0
                continue
            room = abs(nxt - x)
            if remaining < room:
                x += state["dir"] * remaining
                remaining = 0.0
            else:
                remaining -= room
                x = nxt
                is_extreme = (nxt >= xs[-1] - 1e-9) if forward else (nxt <= xs[0] + 1e-9)
                if is_extreme or rng.random() < turn_prob:
                    state["axis"] = "y"
                    state["dir"] = 1.0 if rng.random() < 0.5 else -1.0
        else:
            forward = state["dir"] > 0.0
            nxt = min((s for s in ys if s > y + 1e-9), default=None) if forward \
                else max((s for s in ys if s < y - 1e-9), default=None)
            if nxt is None:
                state["axis"] = "x"
                state["dir"] = 1.0 if rng.random() < 0.5 else -1.0
                continue
            room = abs(nxt - y)
            if remaining < room:
                y += state["dir"] * remaining
                remaining = 0.0
            else:
                remaining -= room
                y = nxt
                is_extreme = (nxt >= ys[-1] - 1e-9) if forward else (nxt <= ys[0] + 1e-9)
                if is_extreme or rng.random() < turn_prob:
                    state["axis"] = "x"
                    state["dir"] = 1.0 if rng.random() < 0.5 else -1.0

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
