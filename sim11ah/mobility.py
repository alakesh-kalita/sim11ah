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


def highway_loop_step(
    sim,
    sta_id: int,
    dt: float,
    speed_mps: float,
    lane_y: float,
    x_min: float,
    x_max: float,
) -> None:
    """
    Drive a road vehicle one-way along a straight highway at constant
    speed, continuously looping back to the start the instant it reaches
    the far end -- real highway traffic flowing past and re-entering,
    not a back-and-forth bounce (a vehicle never reverses, so it always
    ends up moving through every AP's overlap region in the same
    direction on every lap). Meant to be called every tick indefinitely
    (unlike corridor_step, which is a one-shot crossing) -- see
    topology.py's CarsUavsBuilder, which lays cars AND scooters out on
    lanes along the same axis the APs sit on (scooters closer to the
    centreline, cars further out).

    Which way is "forward" is derived from lane_y's own sign, not any
    stored per-vehicle state: a vehicle on the positive-offset lane
    drives toward x_max and resets to x_min on arrival, one on the
    negative-offset lane drives the opposite way and resets to x_max --
    the same "opposite lanes carry opposite-direction traffic" divided-
    highway convention CarsUavsBuilder's own lane layout already implies
    (car_lane_offset_m/scooter_lane_offset_m alternate sign per
    vehicle), so this needs no extra state at all, unlike the direction-
    reversing bounce this replaced (which had to remember which way each
    vehicle was currently headed in sim._highway_dirs). Vehicle-agnostic
    -- cars and scooters both call this exact same function, just with
    their own speed and lane.
    """
    node = sim.nodes[sta_id]
    x, _y = node.pos
    if float(lane_y) >= 0.0:
        new_x = x + max(0.0, float(speed_mps)) * float(dt)
        if new_x >= x_max:
            new_x = x_min
    else:
        new_x = x - max(0.0, float(speed_mps)) * float(dt)
        if new_x <= x_min:
            new_x = x_max
    node.pos = (new_x, float(lane_y))


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
