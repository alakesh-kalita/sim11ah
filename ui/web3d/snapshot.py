"""Builds the JSON-serialisable state snapshot the browser-side three.js
scene polls from ``/api/state``. Pure read-only extraction from the live
``Dashboard``/``Simulator`` objects -- never mutates anything, so it is
safe to call from the web server's own thread while the Tkinter main
thread keeps stepping the simulation concurrently (Python's GIL makes the
individual attribute reads/list iterations here atomic enough for a
visualisation feature; a torn read at worst drops one frame of motion).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

# Same canonical PHY-range formula the 2D topology canvas uses for its own
# selected-node range circle (topology_canvas.py's _draw_selection_overlay)
# and the Node Settings panel's "Distance to AP" readout (dashboard_tk.py) --
# imported rather than re-derived so every view agrees on one number instead
# of three formulas that could quietly drift apart.
from ui.topology_canvas import (
    range_m_for_node,
    altitude_m_for_node,
    _diagnose_unjoined,
    dist_to_ap_m,
    rssi_dbm_for_node,
    _ap_range_m,
    _UAV_ROAM_MAX_FRAC,
)


def _stable_bounds(canvas):
    """Same padded bounding box as NetworkCanvas._bounds(), but excludes
    flying relay/UAV nodes. Those orbit/wander continuously (aerial_relay,
    uav topology modes), so folding them into the box that sizes the
    procedural Smart City's road loops meant the box drifted every poll --
    eventually crossing world.js's 20m rebuild-trigger bucket and silently
    reshuffling the entire building/prop layout with no user action (a
    building that happened to land on a different grid cell after even a
    tiny R/loop change looks like it teleported).

    Falls back to a box centred on the AP with a radius derived from the
    UAV roam annulus (_UAV_ROAM_MAX_FRAC * the AP's PHY range -- the exact
    ceiling advance_uav_positions itself picks random-waypoint targets
    within, see topology_canvas.py) whenever there's at most one static
    reference point left to measure (e.g. "uav" mode, where every non-AP
    node flies -- only the AP itself remains, which alone can't define a
    box). This must be a *config-derived* constant, not every node's
    *current* position: an earlier version of this fallback used
    canvas._bounds() (every live node, movers included), which is exactly
    what NetworkCanvas._bounds() is for on the 2D canvas -- auto-fitting
    the view to wherever the nodes currently are. Reused here it meant the
    3D city's roads/buildings visibly breathed -- shrinking whenever the
    UAVs' independent random walks happened to cluster them together, and
    swallowing the near-centre "skyscraper" tier under the plaza-exclusion
    zone whenever they happened to spread out -- since road-loop size and
    the plaza clearing are both a fraction of this box. The AP's PHY range
    is cached and never changes mid-run, so this ceiling is a true
    constant: the city stops breathing, and the near-centre tall-building
    band stays put regardless of where the UAVs currently are.
    """
    moving = canvas.drone_ids | canvas.uav_ids | canvas.car_ids | canvas.scooter_ids
    xs, ys = [], []
    for nid, n in canvas.sim.nodes.items():
        if nid in moving:
            continue
        x, y = n.pos
        xs.append(x)
        ys.append(y)
    if len(xs) <= 1:
        ap = canvas.sim.nodes.get(0)
        acx, acy = ap.pos if ap is not None else (0.0, 0.0)
        r = _UAV_ROAM_MAX_FRAC * _ap_range_m(canvas.sim)
        return (acx - r, acx + r, acy - r, acy + r)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax - xmin < 1.0:
        xmin, xmax = xmin - 10.0, xmax + 10.0
    if ymax - ymin < 1.0:
        ymin, ymax = ymin - 10.0, ymax + 10.0
    padx = (xmax - xmin) * 0.18 + 8.0
    pady = (ymax - ymin) * 0.18 + 8.0
    return (xmin - padx, xmax + padx, ymin - pady, ymax + pady)


def _road_loops(canvas) -> List[Dict[str, float]]:
    """The exact rectangular loops vehicles drive -- single source of
    truth for both the vehicles' motion (_vehicles, below) and the 3D
    road geometry the browser draws (app.js builds an actual road mesh
    from these, not an independent decorative texture), so a vehicle can
    never visually stray off the pavement.

    Shape now varies by layout_variant, not just the buildings sitting on
    it -- world.js's addCityFillerBuildings/rebuildProps already treat
    `loops` generically (reduce() to find the smallest/largest, iterate
    `for (const l of loops)`), so a different loop COUNT/shape here needs
    no client-side changes to be picked up correctly:
      1 (Downtown Grid): two nested loops, unchanged from the original
         single layout every variant used to share.
      2 (Business District): three nested loops on a tighter grid -- a
         denser arterial network, matching "every block fronts a road".
      3 (Suburban Corridor): one single, wide, flattened loop -- reads as
         a single boulevard, not a downtown grid at all.
    """
    if canvas.environment != "Smart City" or canvas.sim is None:
        return []
    xmin, xmax, ymin, ymax = _stable_bounds(canvas)
    cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    wxs, wys = xmax - xmin, ymax - ymin
    variant = int(getattr(canvas, "layout_variant", 1) or 1)
    if variant == 3:
        return [
            {"cx": cx_w, "cy": cy_w, "hw": wxs * 0.40, "hh": wys * 0.12, "period_s": 20.0},
        ]
    if variant == 2:
        return [
            {"cx": cx_w, "cy": cy_w, "hw": wxs * 0.13, "hh": wys * 0.13, "period_s": 13.0},
            {"cx": cx_w, "cy": cy_w, "hw": wxs * 0.25, "hh": wys * 0.25, "period_s": 20.0},
            {"cx": cx_w, "cy": cy_w, "hw": wxs * 0.37, "hh": wys * 0.37, "period_s": 28.0},
        ]
    return [
        {"cx": cx_w, "cy": cy_w, "hw": wxs * 0.16, "hh": wys * 0.16, "period_s": 16.0},
        {"cx": cx_w, "cy": cy_w, "hw": wxs * 0.34, "hh": wys * 0.34, "period_s": 26.0},
    ]


def _vehicles(canvas) -> List[Dict[str, Any]]:
    """Same rectangular-racetrack math as NetworkCanvas._draw_vehicles_overlay
    (pure function of sim time), re-run here to compute world positions
    instead of drawing pixels -- built from _road_loops() so the cars are
    mathematically guaranteed to sit on the road mesh the browser draws
    from that same list, and kept in sync with the 2D view via the
    canvas's own _bounds()/_rect_loop_pos() rather than re-deriving the
    geometry, so the two views never drift apart."""
    loops_raw = _road_loops(canvas)
    if not loops_raw:
        return []
    from ui.topology_canvas import _CITY_VEHICLE_COLORS

    loops = [(l["cx"], l["cy"], l["hw"], l["hh"], l["period_s"]) for l in loops_raw]
    t_now = float(getattr(canvas.sim.engine, "now", 0.0))
    n = len(_CITY_VEHICLE_COLORS)
    out = []
    for i in range(n):
        lcx, lcy, lhw, lhh, period_s = loops[i % len(loops)]
        phase = (i // len(loops)) / max(1, -(-n // len(loops))) + (0.5 if i % 2 else 0.0)
        t = (t_now / period_s + phase) % 1.0
        wx, wy = canvas._rect_loop_pos(lcx, lcy, lhw, lhh, t)
        wx2, wy2 = canvas._rect_loop_pos(lcx, lcy, lhw, lhh, t + 0.004)
        heading = math.atan2(wy2 - wy, wx2 - wx)
        out.append({"x": float(wx), "y": float(wy), "heading": float(heading),
                     "color": _CITY_VEHICLE_COLORS[i]})
    return out


def _node_dict(canvas, nid: int, n, sim) -> Dict[str, Any]:
    ctx = getattr(getattr(n, "mac", None), "ctx", None)
    assoc_state = int(getattr(ctx, "_assoc_state", 0)) if ctx is not None else 0
    assoc_peer = getattr(ctx, "_assoc_peer_id", None) if ctx is not None else None
    aid = getattr(ctx, "_aid", None) if ctx is not None else None
    # canvas._ap_ids is a multi-AP-aware set (sim.config["topology"]["ap_ids"],
    # falling back to {0}) -- was hardcoded to nid==0, which under multi-AP
    # silently drew every AP past the first as a plain STA box in the 3D
    # scene too (same bug the 2D canvas had -- see topology_canvas.py's
    # sync_from_sim, which is where _ap_ids actually gets computed).
    role = "AP" if nid in canvas._ap_ids else ("RELAY" if nid in canvas._relay_ids else "STA")
    try:
        range_m = range_m_for_node(n) or None
    except Exception:
        range_m = None
    is_drone = nid in canvas.drone_ids
    is_uav = nid in canvas.uav_ids
    # Real network node on a highway_bounce_step crossing (see
    # sim11ah/topology.py's CarsUavsBuilder / sim11ah/mobility.py) --
    # distinct from is_drone/is_uav, entities.js dispatches it to its own
    # car/scooter mesh the same way. Heading comes straight from
    # sim._highway_dirs (the ±1 flag highway_bounce_step maintains, shared
    # by both cars and scooters), not from diffing consecutive positions
    # like the decorative Smart City vehicles below do -- cheaper and
    # exact rather than a one-poll-lagged estimate.
    is_car = nid in canvas.car_ids
    is_scooter = nid in canvas.scooter_ids
    heading = None
    if is_car or is_scooter:
        highway_dirs = getattr(sim, "_highway_dirs", {})
        heading = 0.0 if highway_dirs.get(nid, 1) >= 0 else math.pi
    altitude_m = None
    if is_drone or is_uav:
        try:
            altitude_m = altitude_m_for_node(n)
        except Exception:
            altitude_m = None
    # Same diagnostic the 2D topology canvas shows (E417/E118/E240/E102) --
    # computed once here so the 3D scene and real-map twin read the exact
    # same code for the exact same node, rather than each view guessing
    # independently (or not showing one at all).
    error_code = None
    if nid not in canvas._ap_ids:
        try:
            diag = _diagnose_unjoined(n, sim)
            if diag is not None:
                error_code = diag[0]
        except Exception:
            error_code = None
    try:
        dist_ap_m = dist_to_ap_m(n, sim)
    except Exception:
        dist_ap_m = None
    try:
        rssi_dbm = rssi_dbm_for_node(n)
    except Exception:
        rssi_dbm = None
    return {
        "id": nid,
        "role": role,
        "pos": [float(n.pos[0]), float(n.pos[1])],
        "assoc_state": assoc_state,
        "assoc_peer": None if assoc_peer is None else int(assoc_peer),
        "aid": None if aid is None else int(aid),
        "is_drone": is_drone,
        "is_uav": is_uav,
        "is_car": is_car,
        "is_scooter": is_scooter,
        "heading": heading,
        "range_m": range_m,
        "altitude_m": altitude_m,
        "error_code": error_code,
        "dist_ap_m": dist_ap_m,
        "rssi_dbm": rssi_dbm,
    }


def _obstacle_dict(obs) -> Dict[str, Any]:
    if obs.kind == "circle":
        return {"kind": "circle", "cx": float(obs.cx), "cy": float(obs.cy),
                "r": float(obs.r), "loss_db": float(obs.loss_db), "label": obs.label}
    return {"kind": "rect", "x0": float(obs.x0), "y0": float(obs.y0),
            "x1": float(obs.x1), "y1": float(obs.y1),
            "loss_db": float(obs.loss_db), "label": obs.label}


def build_snapshot(dashboard) -> Dict[str, Any]:
    sim = getattr(dashboard, "sim", None)
    canvas = getattr(dashboard, "_net_canvas", None)
    if sim is None or canvas is None or canvas.sim is None:
        return {"ready": False}

    nodes = [_node_dict(canvas, nid, n, sim) for nid, n in sim.nodes.items()]
    # visual_obstacles (when the canvas provides it) is the *unfiltered*
    # structure list -- includes anything skipped from sim.obstacles for
    # PHY purposes only (e.g. a structure enclosing the AP, see
    # NetworkCanvas._encloses_ap). The 2D canvas always draws those
    # structures too (drawing and PHY-registration are separate calls),
    # so the 3D view should render the same full scene rather than being
    # missing whichever structure happened to land on the AP.
    raw_obstacles = getattr(sim, "visual_obstacles", None)
    if raw_obstacles is None:
        raw_obstacles = getattr(sim, "obstacles", []) or []
    obstacles = [_obstacle_dict(o) for o in raw_obstacles]
    packets = [{"tx": pk["tx"], "rx": pk["rx"], "kind": pk["kind"], "p": float(pk["p"])}
               for pk in getattr(canvas, "_packets", [])]
    pulses = [{"tx": pl["tx"], "kind": pl["kind"], "p": float(pl["p"])}
              for pl in getattr(canvas, "_pulses", [])]

    metrics = {}
    try:
        m = dashboard._compute_metrics()
        metrics = {
            "pdr": float(m.get("pdr", 0.0)),
            "thr_kbps": float(m.get("thr_bps", 0.0)) / 1000.0,
            "lat_avg_ms": float(m.get("lat_avg", 0.0)) * 1000.0,
            "drop_rate": float(m.get("drop_rate", 0.0)),
        }
    except Exception:
        pass

    return {
        "ready": True,
        "time": float(getattr(sim.engine, "now", 0.0)),
        "environment": canvas.environment,
        "variant": int(canvas.layout_variant or 1),
        "nodes": nodes,
        "obstacles": obstacles,
        "packets": packets,
        "pulses": pulses,
        "vehicles": _vehicles(canvas),
        "road_loops": _road_loops(canvas),
        "metrics": metrics,
    }
