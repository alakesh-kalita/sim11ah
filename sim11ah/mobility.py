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


def _grid_turn_arc(x: float, y: float, d1: float, d2: float, old_axis: str, r: float) -> dict:
    """Build the state["arc"] dict for a turn beginning at the vehicle's
    CURRENT (x, y) -- the arc's entry point, `r` metres before the stop --
    where d1 is the OLD axis's direction of travel and d2 is the NEW
    direction on the perpendicular axis once the turn completes.

    Derivation (verified numerically for all 8 (old_axis, d1, d2)
    combinations before this was written): for old_axis="x", centre =
    (x, y + d2*r) -- offset r along the NEW axis from the entry point --
    and exit = (centre.x + d1*r, centre.y), which always lands at exactly
    the stop's own old-axis coordinate, r further along the new axis. The
    rotation sense (CW/CCW) is whichever one is actually consistent with
    entry and exit both lying on the circle with the correct tangent
    directions, computed generically via the sign of the cross product of
    (entry-centre) and (exit-centre) -- no hardcoded sign table, and it
    generalises to the old_axis="y" case by symmetry (x/y swapped)."""
    if old_axis == "x":
        cx, cy = x, y + d2 * r
        exit_x, exit_y = cx + d1 * r, cy
        next_axis = "y"
    else:
        cx, cy = x + d2 * r, y
        exit_x, exit_y = cx, cy + d1 * r
        next_axis = "x"
    ev = (x - cx, y - cy)
    xv = (exit_x - cx, exit_y - cy)
    cross = ev[0] * xv[1] - ev[1] * xv[0]
    sweep = (math.pi / 2.0) if cross > 0.0 else (-math.pi / 2.0)
    theta0 = math.atan2(ev[1], ev[0])
    return {
        "cx": cx, "cy": cy, "r": r,
        "theta0": theta0, "sweep": sweep, "t": 0.0,
        "exit_x": exit_x, "exit_y": exit_y,
        "next_axis": next_axis, "next_dir": d2,
    }


def _grid_turn_heading(axis: str, direction: float) -> float:
    """0.0/pi for a straight x-axis leg, +/-pi/2 for a straight y-axis leg
    -- the same rule ui/topology_canvas.py's vehicle_heading used to
    derive on its own before grid_road_step started reporting a live
    heading directly (see this function's own docstring)."""
    if axis == "x":
        return 0.0 if direction > 0.0 else math.pi
    return math.pi / 2.0 if direction > 0.0 else -math.pi / 2.0


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
    turn_radius_m: float = 8.0,
) -> None:
    """
    Drive a road vehicle continuously through a real road GRID -- replaces
    the old highway_loop_step/cross_street_loop_step, which reset
    (teleported) a vehicle back to the start the instant it reached the
    far end of its lane. That reset was itself a deliberate replacement
    for an even older direction-reversing bounce, but got reported as
    still visibly discontinuous ("position gets reset") -- so a vehicle
    here never resets and never reverses in place, it turns onto a
    perpendicular road instead.

    First version of this function always turned the same way (a fixed
    "turn right" rule) and only ever turned at the outermost road on
    each axis -- every vehicle's path converged onto the exact same
    shared rectangle after its first lap, reported back as "all the
    vehicles are following the same pattern... should be random". Fixed
    by making x_stops/y_stops every real cross-street x / avenue y a
    vehicle on this axis will actually pass over (not just the two
    outermost), with a turn_prob chance of turning at each one reached
    mid-road -- onto a RANDOMLY chosen direction (left or right, 50/50)
    -- instead of always continuing straight; the outermost stop in the
    current direction of travel still forces a turn (nothing further to
    drive to), just randomly directed too.

    THIS version replaces that turn's instant 90-degree pivot with an
    actual quarter-circle arc of radius turn_radius_m (default 8.0,
    matching ui/web3d/static/js/world.js's CURB_FILLET_R -- the already-
    shipped DECORATIVE curb rounding at each intersection corner, so the
    vehicle's real path and the drawn curb agree), reported back as
    "roads should be curved". The turn/no-turn DECISION (same turn_prob
    coin flip, same random left/right choice, same forced-turn-at-the-
    edge rule) is unchanged -- only how the transition is expressed
    geometrically changes, from an instant flip to a smooth arc built by
    _grid_turn_arc.

    The decision now fires turn_radius_m metres BEFORE the stop instead
    of exactly at it (there has to be room left on the straight leg for
    the arc to bend into), which needed a `state["commit"]` field to
    avoid two bugs a design-review pass caught and stress-tested for
    before this was written: (1) naively snapping all the way to the
    stop once the decision fires, rather than only as far as `remaining`
    actually allows, silently violates the per-tick distance budget
    (reproduced: 11m moved in a 3m-budget tick); (2) re-rolling the
    decision on every subsequent tick while a vehicle crawls the last
    few metres to a stop it already committed to driving straight
    through inflates the true turn rate and can require an entry point
    behind where the vehicle already is. state["commit"], once set to a
    given stop's own coordinate, makes the decision for that
    stop-approach exactly once; every following tick approaching the
    same stop just moves normally (identical to the pre-this-change
    behaviour) until the stop is reached and commit is cleared.

    turn_radius_m=0.0 degenerates to EXACTLY the previous (pre-arc)
    behaviour -- decision_room becomes room itself (no early decision
    point), and any arc that could still be entered has zero length and
    is immediately treated as already at its exit point. This is the
    primary regression guard verified before this shipped: running the
    old and new code with turn_radius_m=0.0 against the same seed
    produces IDENTICAL position trajectories tick-for-tick.

    State lives in sim._grid_road_state[sta_id]: {"axis", "dir",
    "arc": None | {...}, "heading": current live heading in radians
    (None only before this function's first call for this sta_id --
    ui/topology_canvas.py's vehicle_heading() reads this directly so a
    mid-turn vehicle's rendered heading sweeps smoothly instead of
    snapping at the end of the arc, and doesn't need to duplicate any
    arc-angle math of its own), "commit": None | the stop coordinate
    already committed to (see above)}. Seeded from init_axis/init_dir
    only the first time a given sta_id is seen -- every call after that
    ignores init_axis/init_dir entirely and just reads back whatever
    this function itself last decided, so one call site can drive EVERY
    vehicle (avenue-started or cross-street-started alike) through the
    same logic without tracking "is this still an avenue vehicle"
    itself; topology.py's CarsUavsBuilder only needs init_axis/init_dir
    to match how each vehicle was actually placed (avenue: init_axis="x",
    init_dir=sign(lane_y); cross street: init_axis="y",
    init_dir=sign(lane_offset)).

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

    Uses sim.engine.rng (the simulator's own seeded RNG), not Python's
    global random module, so traffic patterns stay reproducible for a
    given seed like everything else in this codebase.
    """
    if not hasattr(sim, "_grid_road_state"):
        sim._grid_road_state = {}
    state = sim._grid_road_state.get(sta_id)
    if state is None:
        state = {
            "axis": init_axis,
            "dir": 1.0 if float(init_dir) >= 0.0 else -1.0,
            "arc": None,
            "heading": None,
            "commit": None,
        }
        sim._grid_road_state[sta_id] = state

    node = sim.nodes[sta_id]
    x, y = node.pos
    xs = sorted(float(v) for v in x_stops)
    ys = sorted(float(v) for v in y_stops)
    rng = sim.engine.rng
    remaining = max(0.0, float(speed_mps)) * float(dt)
    turn_r = max(0.0, float(turn_radius_m))

    # Left at 4 (the original bound), not bumped, even though a design
    # review pass initially suggested more headroom for a future faster
    # "Car Speed" control: a real turn never needs more than 2 iterations
    # at any realistic speed (arcs take multiple TICKS to traverse, not
    # multiple iterations within one), and bumping this turned out to
    # change how many times a vehicle sitting exactly at one of the
    # grid's 4 absolute corners (x AND y both at their own boundary
    # simultaneously -- both axes report "nxt is None" and force a turn
    # in place, which can oscillate x<->y without moving at all) bounces
    # before the tick ends -- caught by the turn_radius_m=0.0 exact-
    # equivalence regression test, which failed with this bumped to 6.
    # Left at 4 for byte-for-byte compatibility with the pre-arc code at
    # turn_radius_m=0.0 -- confirmed this specific scenario is the ONLY
    # thing sensitive to the bound; every real (non-corner-coincident)
    # turn resolves in 1-2 iterations regardless.
    for _ in range(4):
        # Not "if remaining <= 0: break" unconditionally -- a turn just
        # committed to this same tick (state["arc"] freshly set, possibly
        # with the tick's ENTIRE remaining budget already spent reaching
        # the stop) still needs its arc resolved even at remaining==0: a
        # zero-length arc (turn_radius_m=0, or rp collapsed to ~0)
        # collapses immediately with no distance required, and skipping
        # that collapse here would leave axis/dir un-flipped until
        # whenever this function next happens to be called with
        # remaining>0 -- a real, found-by-testing one-tick-delayed-turn
        # bug (round stop spacing relative to speed*dt hits this often,
        # not just as a float-precision fluke). A pending NON-degenerate
        # arc is unaffected either way -- stepping it with remaining=0
        # is a harmless no-op (advances 0 progress), bounded by this
        # same fixed iteration count regardless.
        if remaining <= 0.0 and state["arc"] is None:
            break

        arc = state["arc"]
        if arc is not None:
            arc_len = arc["r"] * (math.pi / 2.0)
            if arc_len <= 1e-9:
                # Degenerate (turn_radius_m/room collapsed to ~0) -- treat
                # as already having arrived at the exit point.
                x, y = arc["exit_x"], arc["exit_y"]
                state["axis"], state["dir"] = arc["next_axis"], arc["next_dir"]
                state["heading"] = _grid_turn_heading(state["axis"], state["dir"])
                state["arc"] = None
                continue
            remaining_in_arc = (1.0 - arc["t"]) * arc_len
            if remaining < remaining_in_arc:
                arc["t"] += remaining / arc_len
                remaining = 0.0
                theta = arc["theta0"] + arc["sweep"] * arc["t"]
                x = arc["cx"] + arc["r"] * math.cos(theta)
                y = arc["cy"] + arc["r"] * math.sin(theta)
                state["heading"] = theta + (math.pi / 2.0 if arc["sweep"] > 0.0 else -math.pi / 2.0)
            else:
                remaining -= remaining_in_arc
                x, y = arc["exit_x"], arc["exit_y"]
                state["axis"], state["dir"] = arc["next_axis"], arc["next_dir"]
                state["heading"] = _grid_turn_heading(state["axis"], state["dir"])
                state["arc"] = None
            continue

        if state["axis"] == "x":
            stops, moving, lane = xs, x, y
        else:
            stops, moving, lane = ys, y, x
        forward = state["dir"] > 0.0
        nxt = min((s for s in stops if s > moving + 1e-9), default=None) if forward \
            else max((s for s in stops if s < moving - 1e-9), default=None)
        if nxt is None:
            # Already at (or past, from float error) the outermost stop
            # with nowhere further to go -- force a turn in place (an
            # instant pivot, not an arc -- there's no natural "R before"
            # entry point for this rare defensive path) rather than
            # drive off the edge of the grid.
            state["axis"] = "y" if state["axis"] == "x" else "x"
            state["dir"] = 1.0 if rng.random() < 0.5 else -1.0
            state["commit"] = None
            state["heading"] = _grid_turn_heading(state["axis"], state["dir"])
            continue

        room = abs(nxt - moving)
        if state["commit"] == nxt:
            # Already decided (this stop-approach) not to turn here --
            # just keep moving, never re-rolling the decision.
            step = min(remaining, room)
            moving += state["dir"] * step
            remaining -= step
            if step >= room - 1e-9:
                state["commit"] = None
        else:
            decision_room = max(0.0, room - turn_r)
            if remaining < decision_room:
                moving += state["dir"] * remaining
                remaining = 0.0
            else:
                is_extreme = (nxt >= stops[-1] - 1e-9) if forward else (nxt <= stops[0] + 1e-9)
                if is_extreme or rng.random() < turn_prob:
                    remaining -= decision_room
                    moving += state["dir"] * decision_room
                    rp = min(turn_r, room)
                    d2 = 1.0 if rng.random() < 0.5 else -1.0
                    if state["axis"] == "x":
                        x, y = moving, lane
                    else:
                        y, x = moving, lane
                    # A degenerate (zero-length) arc must collapse to the
                    # instant pivot right here, in this same iteration --
                    # not via state["arc"] + a follow-up iteration that
                    # collapses it next time around. That extra iteration
                    # is a real cost against this tick's fixed loop-count
                    # budget: with turn_radius_m=0 (the equivalence-test
                    # regime, and also whenever room<=turn_radius_m already
                    # zeroed rp out) EVERY turn would burn one iteration
                    # just to create the arc and a second to collapse it,
                    # versus the old code's single-iteration instant pivot
                    # -- so any tick needing two turns (e.g. a grid corner)
                    # would run out of budget one step earlier than the old
                    # code and strand leftover `remaining` unconsumed. Found
                    # by the turn_radius_m=0 equivalence test diverging on a
                    # double-turn tick despite the single-turn case already
                    # matching exactly.
                    if rp <= 1e-9:
                        state["axis"] = "y" if state["axis"] == "x" else "x"
                        state["dir"] = d2
                        state["heading"] = _grid_turn_heading(state["axis"], state["dir"])
                        state["arc"] = None
                    else:
                        state["arc"] = _grid_turn_arc(x, y, state["dir"], d2, state["axis"], rp)
                    state["commit"] = None
                    continue
                else:
                    state["commit"] = nxt
                    step = min(remaining, room)
                    moving += state["dir"] * step
                    remaining -= step
                    if step >= room - 1e-9:
                        state["commit"] = None

        if state["axis"] == "x":
            x, y = moving, lane
        else:
            y, x = moving, lane
        state["heading"] = _grid_turn_heading(state["axis"], state["dir"])

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
