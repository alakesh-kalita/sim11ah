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


def highway_bounce_step(
    sim,
    sta_id: int,
    dt: float,
    speed_mps: float,
    lane_y: float,
    x_min: float,
    x_max: float,
) -> None:
    """
    Drive a road vehicle back and forth along a straight highway at
    constant speed: x oscillates between x_min and x_max on a fixed lane
    (y=lane_y), reversing direction at each end instead of stopping there.
    Meant to be called every tick indefinitely (unlike corridor_step,
    which is a one-shot crossing) so the vehicle keeps crossing every AP's
    overlap region for as long as the caller keeps driving it -- see
    topology.py's CarsUavsBuilder, which lays cars AND scooters out on
    lanes along the same axis the APs sit on (scooters closer to the
    centreline, cars further out).

    Vehicle-agnostic -- cars and scooters both call this exact same
    function, just with their own speed and lane. Direction is tracked in
    a small per-simulator dict (sim._highway_dirs), keyed by sta_id -- the
    same "side state lives on the Simulator object" pattern ui/
    topology_canvas.py's advance_uav_positions already uses for its own
    per-node mobility state (sim._uav_targets); there's no other home for
    this in the codebase, so this follows the existing precedent rather
    than inventing a new one (e.g. stashing it on the Node itself).
    """
    if not hasattr(sim, "_highway_dirs"):
        sim._highway_dirs = {}
    direction = sim._highway_dirs.get(sta_id, 1)

    node = sim.nodes[sta_id]
    x, _y = node.pos
    new_x = x + direction * max(0.0, float(speed_mps)) * float(dt)

    if new_x >= x_max:
        new_x = x_max
        direction = -1
    elif new_x <= x_min:
        new_x = x_min
        direction = 1

    node.pos = (new_x, float(lane_y))
    sim._highway_dirs[sta_id] = direction


def uav_waypoint_step(
    sim,
    sta_id: int,
    dt: float,
    speed_mps: float,
    region: Tuple[float, float, float, float],
    arrive_eps_m: float = 2.0,
) -> None:
    """
    Fly a UAV toward a random waypoint inside `region` (x_min, y_min,
    x_max, y_max), picking a fresh random waypoint on arrival -- classic
    random-waypoint mobility. Not anchored to any single AP: pass a region
    spanning the whole multi-AP corridor (see topology.py's
    CarsUavsBuilder.uav_region) so a UAV naturally wanders across every
    AP's coverage over time instead of circling just one.

    This is the multi-AP, headless counterpart to
    ui/topology_canvas.py's advance_uav_positions (the existing single-AP
    GUI mobility, which anchors on node 0 and is driven by the Tk tick
    loop) -- kept separate rather than generalizing that function in
    place, since the existing "uav" topology's random-waypoint behavior
    around a single AP is itself an established, regression-tested
    result. Target state lives in sim._multi_ap_uav_targets (a distinct
    attribute name from that function's own sim._uav_targets, even though
    both hold the same {sta_id: (x, y)} shape, so the two mobility drivers
    can never collide if a caller somehow mixed them).

    Uses sim.engine.rng (the simulator's own seeded RNG), not Python's
    global random module, so flight paths stay reproducible for a given
    seed like everything else in this codebase.
    """
    if not hasattr(sim, "_multi_ap_uav_targets"):
        sim._multi_ap_uav_targets = {}

    x_min, y_min, x_max, y_max = region
    node = sim.nodes[sta_id]
    x, y = node.pos

    target = sim._multi_ap_uav_targets.get(sta_id)
    if target is None or math.hypot(target[0] - x, target[1] - y) < arrive_eps_m:
        target = (
            x_min + sim.engine.rng.random() * (x_max - x_min),
            y_min + sim.engine.rng.random() * (y_max - y_min),
        )
        sim._multi_ap_uav_targets[sta_id] = target

    tx, ty = target
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    if dist <= 1e-9:
        return
    step = min(dist, max(0.0, float(speed_mps)) * float(dt))
    node.pos = (x + dx / dist * step, y + dy / dist * step)
