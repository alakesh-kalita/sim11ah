"""sim11ah — interactive drag-and-drop network topology canvas.

Node positions drawn here are the *real* ``node.pos`` coordinates the PHY
layer's log-distance path-loss model reads (see ``sim11ah/phy.py:
_distance_m``). Dragging a node in this view therefore changes real RSSI /
PER between that node and its peers, not just a picture. "Range" in the
properties panel is the distance at which the node's EIRP puts RSSI exactly
at the receiver sensitivity threshold; editing it re-solves for EIRP so the
displayed circle stays meaningful.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tkinter as tk
from typing import Callable, Dict, Optional, Set, Tuple

try:
    from sim11ah.obstacles import Obstacle
except ImportError:
    # Support `python ui/dashboard_tk.py` run directly, before that script's
    # own __main__ block has put the project root on sys.path.
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from sim11ah.obstacles import Obstacle

from sim11ah.mac.common import AssocState

# Small local palette (kept independent of ui.dashboard_tk to avoid a
# circular import between the two modules).
_BG      = "#ffffff"
_BORDER  = "#e2e8f0"
_MUTED   = "#64748b"
_BLUE    = "#3b82f6"
_BLUE_DK = "#1d4ed8"
_PURPLE  = "#a855f7"
_PURPLE_DK = "#7c3aed"
_CYAN    = "#06b6d4"
_TEAL    = "#14b8a6"
_AMBER   = "#f59e0b"
_GREEN   = "#22c55e"
_RED     = "#ef4444"
_STA_BODY = "#64748b"   # fixed plain-STA body colour -- status shown via a small LED instead
_GRAY    = "#94a3b8"

# Aerial-scene palettes -------------------------------------------------------
# Colour grading: everything below is deliberately desaturated toward the
# hazy, muted tones of real ortho-photo / drone-survey imagery (atmospheric
# scattering washes out chroma from altitude). Keep new colours in that key.
#
# Lighting convention (applies to EVERY scene element): one sun, low in the
# north-west -> in pixel space all cast shadows fall DOWN-AND-RIGHT, lit
# faces/lobes are on the upper-left, shaded faces on the lower-right.

# Paddy field ------------------------------------------------------------------
# Real rice-paddy ortho imagery reads as vivid, saturated green/emerald --
# not muddy olive -- so this palette favours clean, confident colour over
# desaturation; texture comes from the organic plot mosaic itself, not
# added noise (see _bg_paddy_v1/v2/v3: no whole-scene stipple wash).
_PADDY_BASE   = "#cdd9a0"   # dry-earth field base showing between plots
_PADDY_BUND   = "#9c8a5e"   # raised dirt bund walls between plots
_PADDY_PATH   = "#c7ae7f"   # farm track / dirt path
_PADDY_WATER  = "#7fb6cf"   # flooded (newly transplanted) plot
_PADDY_GLINT  = "#eef8fb"   # sun glint on standing water
_PADDY_CANAL  = "#5f9dbf"   # irrigation channel
_PADDY_CANAL_LT = "#a9d8e8" # channel sheen
_PADDY_CROPS  = ("#6fb04e", "#83c15f", "#9ed36f", "#57964a",
                 "#b6de84", "#c8c266", "#dcd28e")  # growth stages -> ripe/stubble
_PADDY_ROWS   = "#4c7a3e"   # crop-row striping
_TREE_CANOPY  = "#3f7a3f"
_TREE_EDGE    = "#2e5c30"   # shaded SE canopy lobe
_TREE_LOBE    = "#4f9450"   # sunlit NW canopy lobe
_HUT_WALL     = "#c9a36a"
_HUT_ROOF     = "#8a5a3b"
_HUT_ROOF_LT  = "#a5713f"   # sunlit NW roof slope
_SHADOW       = "#4b5563"   # generic cast-shadow tone (used with stipple)

# River valley / highland terrain ----------------------------------------------
_MTN_BASE     = "#7d8175"   # mountain lower slopes
_MTN_MID      = "#93968b"   # mid-slope rock/scrub
_MTN_RIDGE    = "#abaea1"   # sunlit ridge highlight
_MTN_SNOW     = "#e7ebeb"   # snow caps
_MTN_SHADE    = "#5d6156"   # shaded faces / crevices
_MTN_HAZE     = "#c3c6bb"   # aerial-perspective haze on the far range
_RIVER        = "#63849a"   # main river channel
_RIVER_LT     = "#a4bbc7"   # river sheen / riffles
_RIVER_BANK   = "#b2b995"   # floodplain grass along the banks
_BRIDGE_DECK  = "#8b7761"   # wooden bridge deck
_BRIDGE_RAIL  = "#645440"
_TERRACE_SHADES = ("#4f8a48", "#5f9852", "#71a75e", "#85b76e",
                   "#9bc880", "#b3d494", "#c9dda8")  # valley floor -> hilltop
_TERRACE_BUND = "#847a60"   # terrace retaining-wall lip

# Business park / process plant --------------------------------------------------
_LAWN         = "#a2b88f"   # landscaped lawn patch
_LAWN_LT      = "#b3c59e"   # mowing-stripe highlight
_LAWN_GRAIN   = "#8ba278"   # turf-texture stipple overlay
_PIPE         = "#7d8792"   # pipe-rack main runs
_PIPE_LT      = "#aeb7c0"   # pipe highlight
_PIPE_POST    = "#57616c"   # pipe-rack support posts

# Industrial site ----------------------------------------------------------------
_IND_BASE     = "#d2d5d8"   # concrete/gravel yard
_IND_GRAIN_DK = "#a9aeb3"   # gravel-grain stipple overlay (dark speckle)
_IND_GRAIN_LT = "#e4e6e8"   # gravel-grain stipple overlay (light speckle)
_IND_APRON    = "#c2c7cb"   # paved aprons around buildings
_IND_ROAD     = "#a5abb1"   # asphalt road
_IND_ROAD_EDGE = "#8d949b"
_IND_ROAD_MARK = "#e9ecee"
_IND_ROAD_GRAIN = "#959ba3" # asphalt-wear stipple overlay
_IND_WAREHOUSE = "#b3bac2"  # big-shed metal roof
_IND_PLANT    = "#a0a6ad"   # process-plant roof
_IND_OFFICE   = "#bdb3a2"   # office block roof
_IND_ROOF_EDGE = "#78828c"  # softened parapet edge (was near-black)
_IND_ROOF_LT  = "#d6dbdf"   # roof ridge/panel-seam highlight
_IND_ROOF_GRAIN = "#8d959d" # weathered-panel stipple overlay
_IND_HVAC     = "#dcdfe2"
_IND_TANK     = "#d3d7db"
_IND_TANK_RIM = "#8a919b"
_IND_DOCK     = "#c19b58"   # loading-dock hazard stripe (faded paint)
_IND_SMOKE    = "#9ca3af"
_IND_LOT      = "#9ba1a8"   # parking-lot asphalt
_IND_LAMP     = "#c7b389"   # floodlight sodium glow (muted)
_CAR_COLORS   = ("#8b95a0", "#96615c", "#5b6a7f", "#d6dadd", "#575f68")

# Smart City palette -------------------------------------------------------------
_CITY_BASE      = "#c7cbcf"   # concrete/urban ground
_CITY_BASE_ALT  = "#bec2c7"
_CITY_ROAD      = "#4a4f57"   # dark asphalt
_CITY_ROAD_EDGE = "#33373d"
_CITY_ROAD_MARK = "#e8c547"   # yellow lane markings
_CITY_SIDEWALK  = "#a8adb3"
_CITY_CROSSWALK = "#e5e8eb"
_CITY_PARK      = "#7fa35e"
_CITY_PARK_LT   = "#94b874"
_TOWER_GLASS_1  = "#4f6c8a"   # tallest tier -- deepest, most saturated glass
_TOWER_GLASS_2  = "#7089a1"
_TOWER_GLASS_3  = "#93a5b6"   # shortest tier -- palest
_TOWER_ROOF_LT  = "#c3d2de"
_TOWER_EDGE     = "#2c3946"
_TOWER_WINDOW   = "#dbe7f0"
_CITY_VEHICLE_COLORS = ("#c0392b", "#2a6f97", "#e0a13a", "#5c6672", "#3a7d5c", "#e6d24a")

# Real (non-decorative) cars_uavs-mode vehicle palettes -- deliberately
# vivid and green-free, since _assoc_color's green is reserved for "this
# node is ASSOCIATED" status everywhere else on the canvas; giving real
# cars/scooters their own hue keeps a healthy, fully-associated fleet from
# reading as a monochrome green scene. Distinct from _CITY_VEHICLE_COLORS
# (decorative Smart City traffic) so a glance can tell "real network node"
# from "background scenery" even before checking for a status LED.
_REAL_CAR_COLORS = ("#ef4444", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899")
_REAL_SCOOTER_COLORS = ("#fb923c", "#38bdf8", "#f472b6", "#fbbf24", "#c084fc", "#f87171")

# Military Zone palette -----------------------------------------------------------
_MIL_BASE       = "#8f8a5e"   # dry scrubland/dirt base
_MIL_BASE_ALT   = "#83805a"
_MIL_PATCH_1    = "#6f7a4a"   # camo blotch tones (irregular hashed polygons)
_MIL_PATCH_2    = "#8a7f4f"
_MIL_PATCH_3    = "#5c6b45"
_MIL_TRACK      = "#7a715a"   # dirt access track (no paved roads on a firebase)
_MIL_TRACK_EDGE = "#645d49"
_MIL_TRACK_RUT  = "#6b6350"
_MIL_WIRE       = "#767b6e"   # barbed-wire perimeter
_MIL_BUNKER     = "#8a8570"   # sandbag/concrete bunker wall
_MIL_BUNKER_DK  = "#6f6b5a"
_MIL_BUNKER_EDGE = "#524e40"
_MIL_SANDBAG_LT = "#a49a72"   # individual sandbag highlight
_MIL_TENT       = "#5c6b4a"   # canvas supply tent
_MIL_TENT_DK    = "#495638"
_MIL_CRATE      = "#6b5a3c"   # supply crate
_MIL_TOWER      = "#4c4a3e"   # watchtower timber/steel mast
_MIL_TANK_HULL  = "#586347"   # olive-drab tank hull
_MIL_TANK_DK    = "#454e37"   # shaded hull / tracks
_MIL_TANK_TURRET = "#657154"
_MIL_SOLDIER    = "#4a4f3c"   # camo fatigues
_MIL_SOLDIER_SKIN = "#c69a72"
_MIL_CRATER     = "#6e6650"   # scorched/blast crater
_MIL_FLAG       = "#a4392f"

# Drone / UAV palette -----------------------------------------------------------
# Matte carbon-fibre body (real consumer/commercial drones are almost never
# bright plastic orange) with aviation-standard nav lights: green
# front-left, red front-right, white strobe rear -- the same convention
# used on real aircraft, and a strong "this is a real vehicle" visual cue.
_DRONE_BODY    = "#2b2f36"
_DRONE_BODY_LT = "#42474f"   # lighter top-panel facet, subtle 3D shading
_DRONE_ARM     = "#1a1d21"
_DRONE_MOTOR   = "#565c64"   # motor-housing bump at each rotor base
_DRONE_ROTOR   = "#cdd3da"   # translucent spinning-blade blur
_DRONE_LED_L   = "#22c55e"   # front-left nav light (green, aviation convention)
_DRONE_LED_R   = "#ef4444"   # front-right nav light (red)
_DRONE_LED_AFT = "#f8fafc"   # rear strobe (white)
_DRONE_LENS    = "#7dd3fc"   # camera-gimbal lens glint
_DRONE_PATH    = "#f97316"   # AP<->drone link line / roam-boundary (kept distinct
                              # from the vehicle itself so it still reads as "radio link")
_DRONE_TRAIL   = "#8592a6"   # muted slate-grey contrail (not the link-line orange)

# Every scatter/roam/racetrack distance below is an explicit *fraction of
# the AP's own real, PHY-computed coverage radius* (range_m_for_node(ap)),
# resolved at seed/mobility time via _ap_range_m() -- not an independently
# hand-picked metre value. That keeps the whole layout self-consistent and
# physically possible (nothing placed or flown further than the active PHY
# config actually reaches) no matter what EIRP/sensitivity/path-loss the
# active config uses, instead of numbers tuned to one specific config that
# would silently go stale (or become physically nonsensical) if it changes.
_UAV_SPEED_MIN_MPS, _UAV_SPEED_MAX_MPS = 4.0, 14.0   # plausible small-UAV cruise speed
_UAV_ARRIVE_EPS_M = 2.0

# Default random-scatter fractions used the first time a topology is seeded
# -- IEEE 802.11ah's whole point is sub-GHz long range, so nodes should
# spread across that range, not bunch up near the AP the way a ~2.4/5 GHz
# Wi-Fi deployment would.
_STA_SCATTER_MIN_FRAC, _STA_SCATTER_MAX_FRAC = 0.04, 0.82
# Relays sit out at a meaningful distance (still comfortably within the
# AP's own range, so their own backhaul association works) so they
# actually extend coverage instead of being redundant with a STA that
# could just talk to the AP directly.
_RELAY_SCATTER_MIN_FRAC, _RELAY_SCATTER_MAX_FRAC = 0.33, 0.74
# Each relay's STAs spread out around *it*, not the AP, far enough that
# relay_dist + sta_offset routinely exceeds the AP's own range -- so a
# meaningful share of them genuinely need the relay, while others land in
# dual coverage (handover-eligible; see AssocManager._maybe_roam/_roam_to)
# instead of every STA being able to reach the AP directly regardless of
# whether a relay exists.
_RELAY_STA_MIN_FRAC, _RELAY_STA_MAX_FRAC = 0.12, 0.49
# UAV (end-node) random-waypoint annulus, and the aerial-relay racetrack.
# The racetrack's reach matches relay-scatter's own max fraction (not a
# smaller, independently-picked ellipse) so a relay's flight path actually
# passes back through the region it -- and its assigned STAs -- were
# scattered into, instead of being pulled into a tighter loop that can
# leave far-scattered relays (and their STAs) permanently out of reach.
_UAV_ROAM_MIN_FRAC, _UAV_ROAM_MAX_FRAC = 0.04, 0.74
# Same idea as _RELAY_STA_MIN/MAX_FRAC above, but for UAV *end-node*
# roaming instead of a one-time ground scatter: 0.74 (_UAV_ROAM_MAX_FRAC)
# was tuned for plain "uav" mode, where there's no relay to fall back on,
# so a UAV needs to stay within the AP's own direct range at all times.
# Reused unchanged for "relay_uav"/"aerial_relay_uav" mode, every UAV STA
# stayed inside AP range for its entire flight -- under the default "auto"
# (AP-first) association preference it never needed the relay at all, so
# STAs never associated with it regardless of how many relays existed or
# where they were placed. This wider ceiling (relay's own max distance
# from the AP, plus a relay-assigned ground STA's own further reach from
# it -- exactly matching the farthest a "relay"/"aerial_relay" mode ground
# STA can already legitimately land) pushes the roam annulus out past the
# AP's own range so a meaningful share of flight time is actually
# relay-dependent, matching that same "genuinely need the relay" design
# note above -- see _uav_roam_bounds, which picks this over the plain
# bounds whenever the topology actually has relays to roam near.
_UAV_RELAY_ROAM_MAX_FRAC = _RELAY_SCATTER_MAX_FRAC + _RELAY_STA_MAX_FRAC
_DRONE_RX_FRAC, _DRONE_RY_FRAC = _RELAY_SCATTER_MAX_FRAC, _RELAY_SCATTER_MAX_FRAC * 0.62
# Racetrack loop *period* is derived too, not fixed regardless of size: a
# 45s lap around a small ellipse and the same 45s around a much larger one
# imply wildly different, easily-unrealistic ground speeds. Target the
# same plausible small-UAV cruise band used above (mid-point of
# _UAV_SPEED_MIN/MAX_MPS) so widening the racetrack for a longer-range PHY
# config doesn't quietly turn the relay into something flying faster than
# a real small UAV can.
_DRONE_TARGET_SPEED_MPS = 0.5 * (_UAV_SPEED_MIN_MPS + _UAV_SPEED_MAX_MPS)

_PKT_COLORS = {
    "DATA":  _BLUE,
    "ACK":   _GREEN,
    "BEACON": _AMBER,
    "RTS":   _PURPLE,
    "CTS":   _PURPLE,
    "TWT":   _CYAN,
    "MGMT":  _GRAY,
}

_PKT_STEP = 0.22           # progress advanced per animation tick (~4-5 ticks/flight)
_MAX_ACTIVE_PACKETS = 60   # hard cap so a burst of TX events can't flood the canvas
_MAX_ACTIVE_PULSES = 8


def classify_frame_kind(ftype_str: Optional[str]) -> str:
    """Map a MacFrame.ftype's str() (e.g. 'FrameType.DATA') to an animation
    colour bucket."""
    s = (ftype_str or "").upper()
    if "BEACON" in s:
        return "BEACON"
    if "ACK" in s:
        return "ACK"
    if "RTS" in s:
        return "RTS"
    if "CTS" in s:
        return "CTS"
    if "TWT" in s:
        return "TWT"
    if "DATA" in s:
        return "DATA"
    return "MGMT"


def _fspl_db_1m(freq_mhz: float) -> float:
    return 32.44 + 20.0 * math.log10(max(1e-9, freq_mhz)) - 60.0


def range_m_for_node(node) -> float:
    """Nominal coverage radius (m): distance at which RSSI == rx sensitivity.
    Delegates to PhyLayer.nominal_range_m() (sim11ah/phy.py) -- the same
    formula, kept here only as a thin node->phy convenience wrapper for this
    module's existing call sites, so the math lives in exactly one place."""
    phy = getattr(node, "phy", None)
    if phy is None:
        return 0.0
    return float(phy.nominal_range_m())


_FALLBACK_RANGE_M = 1000.0  # only used if the AP/its PHY isn't built yet -- matches the ~1km default below

# sim11ah/config.py's global default path_loss_exp (2.78, "indoor/mixed
# environment") is the single shared default range for the AP, every STA,
# and every relay (~1km at this module's default EIRP/sensitivity/915MHz --
# see range_m_for_node() above) -- deliberately including every headless/
# paper script that calls default_config() without overriding path_loss_exp
# (compare_raw_policies.py etc.), not just the interactive GUI.
#
# This table layers *environment-specific* variation in the GUI on top of
# that same ~1km baseline -- applied per-node onto already-built PHY
# instances (see _apply_environment_path_loss below), never onto
# default_config()'s own dict, so a headless script is still exactly the
# plain ~1km default regardless of what environment the GUI happens to
# have selected. Values are chosen so the AP's own nominal (obstacle-free,
# LOS) coverage radius lands at a distance that's actually plausible for
# that terrain instead of every environment sharing one flat figure
# regardless of how open or built-up it visually is -- ordered most-open
# (lowest exponent, longest range) to most-obstructed (highest exponent,
# shortest nominal range); Smart City/Industrial Site's *actual* effective
# link range is further cut down by their own registered building loss_db
# obstacles on top of this baseline, same as before -- only the open-air
# baseline moves here.
_ENV_PATH_LOSS_EXP = {
    "Paddy Field":     2.72,  # ~1.16km -- flattest, most open terrain here
    "Military Zone":   2.75,  # ~1.07km -- open field/battlefield, few tall obstructions
    "Open Area":       2.78,  # ~0.99km -- generic open ground, light clutter (matches the plain default)
    "Smart City":      2.82,  # ~0.90km -- rooftop-mounted AP, partial LOS; buildings add loss_db on top
    "Industrial Site": 2.86,  # ~0.82km -- plant complex, moderate structural clutter
}


def _apply_environment_path_loss(sim, environment: str) -> None:
    """Overrides every node's already-built PhyLayer.path_loss_exp (an
    instance attribute copied from sim.config['phy'] at construction time,
    see PhyLayer.__init__ -- mutating the config dict alone does nothing to
    already-built nodes) to match the given environment, then invalidates
    _ap_range_m's cache so the next call recomputes against it. Must run
    BEFORE _seed_default_layout so the node scatter/roam/racetrack fractions
    (all derived from _ap_range_m -- see the _*_FRAC constants above) are
    sized to the environment's real range, not the config default's."""
    if sim is None:
        return
    target = _ENV_PATH_LOSS_EXP.get(environment)
    if target is None:
        return
    phy_cfg = sim.config.setdefault("phy", {})
    if phy_cfg.get("path_loss_exp") == target:
        return
    phy_cfg["path_loss_exp"] = target
    for node in sim.nodes.values():
        phy = getattr(node, "phy", None)
        if phy is not None:
            phy.path_loss_exp = target
    sim._cached_ap_range_m = None


def _ap_range_m(sim) -> float:
    """The AP's real, PHY-computed coverage radius -- the single source of
    truth every scatter/roam/racetrack distance in this module is derived
    from (see the _*_FRAC constants above). Cached on the Simulator object
    since the AP's PHY parameters don't change over a run, and a fresh
    Simulator (from Apply & Rebuild) naturally invalidates the cache."""
    cached = getattr(sim, "_cached_ap_range_m", None)
    if cached is not None:
        return cached
    ap = getattr(sim, "nodes", {}).get(0) if sim is not None else None
    r = range_m_for_node(ap) if ap is not None else 0.0
    r = r if r > 0.0 else _FALLBACK_RANGE_M
    try:
        sim._cached_ap_range_m = r
    except Exception:
        pass
    return r


def _uav_roam_bounds(sim):
    """(min_m, max_m) a UAV end-node's random-waypoint target is drawn
    within -- single source of truth for the seeding code, advance_uav_
    positions, and the 2D canvas's own dashed roam-boundary circle, so all
    three always agree on the same annulus. Widens the ceiling to
    _UAV_RELAY_ROAM_MAX_FRAC whenever the topology actually has relays to
    roam near (relay_uav / aerial_relay_uav) -- see that constant's own
    comment for why reusing plain "uav" mode's AP-range-capped ceiling
    here made every UAV STA permanently AP-reachable and the relay
    pointless."""
    r_ap = _ap_range_m(sim)
    has_relays = bool(sim.config.get("topology", {}).get("relay_ids")) if sim is not None else False
    max_frac = _UAV_RELAY_ROAM_MAX_FRAC if has_relays else _UAV_ROAM_MAX_FRAC
    return _UAV_ROAM_MIN_FRAC * r_ap, max_frac * r_ap


def _ellipse_circumference_m(rx: float, ry: float) -> float:
    """Ramanujan's second approximation -- accurate to a fraction of a
    percent for any rx/ry ratio, unlike the naive mean-radius estimate."""
    h = ((rx - ry) ** 2) / max(1e-9, (rx + ry) ** 2)
    return math.pi * (rx + ry) * (1.0 + 3.0 * h / (10.0 + math.sqrt(max(0.0, 4.0 - 3.0 * h))))


def _drone_racetrack_geometry(sim):
    """(cx, cy, rx, ry, period_s) for the aerial-relay racetrack.

    rx/ry default to the AP-range-derived values above (see the
    _DRONE_*_FRAC comments), but can be overridden live via
    sim._relay_path_rx_m / _relay_path_ry_m -- unset or <= 0 means "use
    the default", since a real racetrack can never have a zero radius, so
    that's an unambiguous sentinel rather than a separate on/off flag.
    Center defaults to (0, 0) -- the AP's own fixed position -- via
    sim._relay_path_cx_m / _relay_path_cy_m, which (unlike the radii) are
    real, meaningful values even at exactly 0 (centred directly on the AP,
    not "unset"), so no sentinel is needed there. All four follow the same
    "plain attribute on the live sim object, read fresh every call, GUI
    sets it directly, no rebuild needed" pattern sim._uav_speed_mps
    already uses. Lap speed reuses that same sim._uav_speed_mps rather
    than the fixed _DRONE_TARGET_SPEED_MPS whenever it's set -- one Flight
    Speed control now governs every kind of airborne motion (random-
    waypoint UAVs, "Random Path" relays, and this oval) instead of the
    oval quietly having its own separate, slider-immune constant."""
    r_ap = _ap_range_m(sim)
    rx_override = float(getattr(sim, "_relay_path_rx_m", 0.0) or 0.0)
    ry_override = float(getattr(sim, "_relay_path_ry_m", 0.0) or 0.0)
    rx = rx_override if rx_override > 0.0 else _DRONE_RX_FRAC * r_ap
    ry = ry_override if ry_override > 0.0 else _DRONE_RY_FRAC * r_ap
    cx = float(getattr(sim, "_relay_path_cx_m", 0.0) or 0.0)
    cy = float(getattr(sim, "_relay_path_cy_m", 0.0) or 0.0)
    speed = getattr(sim, "_uav_speed_mps", None)
    speed = float(speed) if speed and speed > 0 else _DRONE_TARGET_SPEED_MPS
    period_s = _ellipse_circumference_m(rx, ry) / speed
    return cx, cy, rx, ry, period_s


def _hash01(*keys: int) -> float:
    """Deterministic pseudo-random in [0, 1) from integer keys (FNV-1a
    style) -- stable across redraws, drags, and (for node-id-keyed callers
    like altitude_m_for_node) across the 3D scene and the real-map twin
    polling the same value from the same node, instead of each view
    inventing its own per-node variation independently."""
    h = 2166136261
    for k in keys:
        h = ((h ^ (int(k) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return (h % 100003) / 100003.0


# Small-UAV cruise altitude bands (metres AGL). Aerial relays fly higher
# than end-node UAVs -- real relay/backhaul drones deliberately gain
# altitude for a cleaner radio line-of-sight over their coverage area
# (less ground clutter/multipath), the same reason cell relay balloons and
# backhaul masts are placed as high as practical -- while end-node UAVs
# (sensors, mobile monitors) have no such incentive to fly higher than
# they need to. Both bands sit comfortably under common small-UAS
# ceilings (e.g. FAA Part 107's 400 ft / ~122 m AGL near-structure limit),
# so the numbers are a real operational constraint, not an arbitrary look.
_UAV_ALT_MIN_M, _UAV_ALT_MAX_M = 35.0, 70.0
_RELAY_ALT_MIN_M, _RELAY_ALT_MAX_M = 70.0, 115.0


def altitude_m_for_node(node) -> float:
    """Deterministic per-node cruise altitude (metres AGL) -- varies node
    to node (not one flat height for every drone/UAV) but is stable across
    polls/redraws, and is the single value both the 3D scene and the
    real-map twin read, so a node is never at two different heights in two
    views. Relay-role nodes get the higher band (see comment above)."""
    is_relay = str(getattr(node, "role", "")).upper() == "RELAY"
    lo, hi = (_RELAY_ALT_MIN_M, _RELAY_ALT_MAX_M) if is_relay else (_UAV_ALT_MIN_M, _UAV_ALT_MAX_M)
    frac = _hash01(int(getattr(node, "node_id", 0)), 0x414C54)  # 'ALT' salt
    return lo + frac * (hi - lo)


def _assoc_state(node) -> int:
    ctx = getattr(getattr(node, "mac", None), "ctx", None)
    return int(getattr(ctx, "_assoc_state", AssocState.UNASSOCIATED)) if ctx is not None else AssocState.UNASSOCIATED


def _assoc_peer(node) -> Optional[int]:
    ctx = getattr(getattr(node, "mac", None), "ctx", None)
    peer = getattr(ctx, "_assoc_peer_id", None) if ctx is not None else None
    return int(peer) if peer is not None else None


def _assoc_color(node) -> str:
    """Same three-colour MAC-association read as the other two views
    (entities.js's statusHex in the 3D scene, assocStateToLabel in the
    real-map twin) -- green once actually ASSOCIATED, amber while
    authenticating/associating, red if it hasn't even started. Real state
    from sim.nodes, not a fixed per-role colour."""
    state = _assoc_state(node)
    if state >= AssocState.ASSOCIATED:
        return _GREEN
    if state >= AssocState.AUTHENTICATING:
        return _AMBER
    return _RED


# Diagnostic codes shown on a node that hasn't joined -- each one is
# computed fresh every redraw from the node's *actual current* distance to
# every potential peer (range_m_for_node, the same real PHY-range formula
# used everywhere else in this module) and its live assoc_state, not a
# canned/guessed message. E417 vs E240 mirrors the same distinction this
# whole codebase has been built around all session: geometrically
# unreachable (no fix possible without moving the node or its peer) is a
# fundamentally different situation from "reachable, but hasn't managed to
# actually complete the handshake yet."
#
# E118 exists because E240 used to lie about that distinction: it only
# checked range_m_for_node's obstruction-free nominal circle, so a node
# sitting behind a Military Zone checkpoint bunker (or any obstacle with
# real loss_db) showed as "in range, just hasn't handshaken yet" even
# though its actual obstruction-inflated RSSI was permanently below
# sensitivity -- with shadowing off by default, that's not a slow link,
# it's a dead one, every single beacon interval, forever. E118 calls that
# out as its own distinct, correctly-labelled state.
_ERR_OUT_OF_RANGE = ("E417", "Out of Range")
_ERR_BLOCKED      = ("E118", "Blocked")
_ERR_WEAK_SIGNAL  = ("E240", "Weak Signal")
_ERR_CONNECTING   = ("E102", "Connecting")

ERROR_CODE_LEGEND = (
    (_ERR_OUT_OF_RANGE, "No AP/relay is within real PHY range of this node's current position."),
    (_ERR_BLOCKED,      "In nominal range, but an obstacle in the path pushes real RSSI below sensitivity."),
    (_ERR_WEAK_SIGNAL,  "In range of a peer, but hasn't managed to complete the handshake yet."),
    (_ERR_CONNECTING,   "Actively authenticating/associating right now -- should resolve shortly."),
)


def _obstruction_clear_margin(tx_node, rx_node, tx_phy) -> bool:
    """Whether tx_node's transmission would actually clear rx sensitivity at
    rx_node's position, accounting for real obstacles crossing the path --
    unlike range_m_for_node, which is a single obstruction-free nominal
    circle. Deliberately recomputes free-space + obstruction loss here
    instead of calling PhyLayer._path_loss_db/_rssi_dbm directly: those
    cache a shadowing draw from the simulation's own RNG on first use, and
    calling them from a UI redraw path would make that draw's timing (and
    so simulation behaviour, if shadowing is ever turned on) depend on how
    often the canvas happens to repaint. This intentionally omits the
    shadow term for that reason -- it's an obstruction-aware floor, not a
    full RSSI replica."""
    fspl1 = _fspl_db_1m(tx_phy.freq_mhz)
    d = max(1e-6, math.hypot(tx_node.pos[0] - rx_node.pos[0], tx_node.pos[1] - rx_node.pos[1]))
    pl = fspl1 + 10.0 * tx_phy.path_loss_exp * math.log10(d / max(1e-6, tx_phy.d0_m))
    obstruction_loss = getattr(tx_phy, "_obstruction_loss_db", None)
    if obstruction_loss is not None:
        try:
            pl += float(obstruction_loss(getattr(tx_node, "node_id", None), getattr(rx_node, "node_id", None)))
        except Exception:
            pass
    rssi = tx_phy.eirp_dbm - pl
    return rssi >= tx_phy.rx_sensitivity_dbm


def _diagnose_unjoined(node, sim) -> Optional[Tuple[str, str]]:
    """(code, short_label) for why `node` hasn't joined, or None if it's
    already ASSOCIATED. See ERROR_CODE_LEGEND for what each code means."""
    if _assoc_state(node) >= AssocState.ASSOCIATED:
        return None

    ap = sim.nodes.get(0) if sim is not None else None
    if ap is None:
        return None

    is_relay = str(getattr(node, "role", "")).upper() == "RELAY"
    peers = [ap]
    if not is_relay:
        # STAs/UAVs can also join via any relay -- a relay itself only
        # ever associates with the real AP (never with another relay).
        relay_ids = sim.config.get("topology", {}).get("relay_ids", [])
        for rid in relay_ids:
            r = sim.nodes.get(rid)
            if r is not None:
                peers.append(r)

    in_range = False
    clear = False  # at least one in-range peer's obstruction-aware RSSI clears sensitivity
    for p in peers:
        rng = range_m_for_node(p)
        if rng <= 0:
            continue
        d = math.hypot(node.pos[0] - p.pos[0], node.pos[1] - p.pos[1])
        if d <= rng:
            in_range = True
            p_phy = getattr(p, "phy", None)
            if p_phy is None or _obstruction_clear_margin(p, node, p_phy):
                clear = True
                break

    if not in_range:
        return _ERR_OUT_OF_RANGE
    if not clear:
        return _ERR_BLOCKED
    if _assoc_state(node) == AssocState.UNASSOCIATED:
        return _ERR_WEAK_SIGNAL
    return _ERR_CONNECTING


def dist_to_ap_m(node, sim) -> Optional[float]:
    """Straight-line distance (metres) from `node` to the real AP's
    current position -- the same real coordinates every range/diagnostic
    check in this module already uses, just exposed directly as a number
    instead of only feeding a range comparison."""
    if sim is None or node is None:
        return None
    ap = sim.nodes.get(0)
    if ap is None or ap is node:
        return None
    return math.hypot(node.pos[0] - ap.pos[0], node.pos[1] - ap.pos[1])


def _fmt_dist_m(d: Optional[float]) -> str:
    if d is None:
        return ""
    return f"{d/1000.0:.2f}km" if d >= 1000.0 else f"{d:.0f}m"


def rssi_dbm_for_node(node) -> Optional[float]:
    """Last *actually measured* RSSI (dBm) this node's own receiver saw
    (see phy.py's _rx_end) -- a real measurement, not re-derived from
    distance. Event-driven: only updates when the node actually receives
    something (at least every beacon interval for anything associated),
    so it's stale between receptions the same way a real signal-strength
    meter would be. None means no reception has happened yet."""
    phy = getattr(node, "phy", None)
    return getattr(phy, "_last_rssi_dbm_value", None) if phy is not None else None


def _fmt_rssi(node) -> str:
    dbm = rssi_dbm_for_node(node)
    return f"{dbm:.0f}dBm" if dbm is not None else ""


def set_range_m_for_node(node, range_m: float) -> None:
    """Set the node's EIRP so its nominal coverage radius equals range_m."""
    phy = getattr(node, "phy", None)
    if phy is None or range_m <= 0:
        return
    fspl1 = _fspl_db_1m(phy.freq_mhz)
    phy.eirp_dbm = float(
        phy.rx_sensitivity_dbm + fspl1
        + 10.0 * phy.path_loss_exp * math.log10(max(1e-6, range_m / max(1e-6, phy.d0_m)))
    )


# Real IIT (ISM) Dhanbad campus building footprints (lat/lon), anchored the
# same way ui/web3d/static/smart-city-simulation.html's own metersToLatLon/
# latLonToMeters anchor at accessPoint -- so an obstacle registered here
# lines up with the actual building the real-map twin shows, instead of a
# disconnected fictional layout. (lat, lon) here matches that file's own
# `landmarks` array; only genuine building-type entries are included
# (Main Gate/Sports Grounds/Bekar Bandh Edge/the campus-label point are
# excluded -- none of those are real blocking structures). Footprint size
# and loss_db are plausible per-building-type estimates, not surveyed
# dimensions -- swap in real footprint polygons here if they become
# available.
_ISM_AP_LAT, _ISM_AP_LON = 23.8136, 86.4423
_ISM_CAMPUS_BUILDINGS = (
    # label, lat, lon, half_width_m, half_depth_m, loss_db
    ("Heritage / Admin",        23.8121, 86.4411, 25.0, 20.0, 18.0),
    ("Penman Auditorium",       23.8118, 86.4402, 30.0, 22.0, 20.0),
    ("Library / Academic Core", 23.8137, 86.4425, 45.0, 35.0, 22.0),
    ("Student Activity Centre", 23.8110, 86.4437, 22.0, 17.0, 15.0),
    ("Hostel Zone",             23.8160, 86.4399, 50.0, 40.0, 18.0),
)


def _latlon_to_local_m(lat: float, lon: float) -> Tuple[float, float]:
    """(x, y) metres relative to the AP's fixed real-world anchor -- the
    same equirectangular projection, anchored at the same point, as the
    real-map twin's own metersToLatLon/latLonToMeters, so a building
    positioned here lines up with where it actually sits on that map."""
    m_per_deg_lon = 111320.0 * math.cos(math.radians(_ISM_AP_LAT))
    x = (lon - _ISM_AP_LON) * m_per_deg_lon
    y = (lat - _ISM_AP_LAT) * 111320.0
    return x, y


def ism_campus_obstacles() -> list:
    """Real IIT (ISM) Dhanbad building footprints as PHY Obstacles, in the
    same local (x, y) metre space every node position already uses (AP at
    the origin). Used instead of the Smart City environment's fictional
    procedural skyline when the scenario being visualised is the real
    campus map -- see _draw_background's Smart City branch: that skyline
    is purely a decorative backdrop for this canvas with no relationship
    to the actual campus, so using it as real PHY obstacles blocked nodes
    for reasons invisible on the real map."""
    # Unfiltered -- deliberately includes any building whose real-world
    # position happens to fall right on top of the AP's own anchor point
    # (e.g. Library / Academic Core is only ~20-30m from the AP anchor in
    # real coordinates). PHY-side AP-enclosure filtering happens once,
    # centrally, in _draw_background via _encloses_ap -- this function's
    # job is just "what buildings exist", visual and PHY callers each
    # decide what to do with that for their own purposes (see
    # _draw_background's Smart City branch).
    obstacles = []
    for label, lat, lon, hw, hd, loss_db in _ISM_CAMPUS_BUILDINGS:
        cx, cy = _latlon_to_local_m(lat, lon)
        x0, y0, x1, y1 = cx - hw, cy - hd, cx + hw, cy + hd
        obstacles.append(Obstacle(
            kind="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            loss_db=loss_db, label=label,
        ))
    return obstacles


def _scatter(rng: random.Random, cx: float, cy: float, r_min: float, r_max: float):
    """Uniform-in-area random point in the annulus [r_min, r_max] around
    (cx, cy) -- avoids the visual artifact of points bunching near the
    centre that a naive uniform-radius sample would produce."""
    r = math.sqrt(rng.uniform(0.0, 1.0) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
    ang = rng.uniform(0.0, 2.0 * math.pi)
    return (cx + r * math.cos(ang), cy + r * math.sin(ang))


def _scatter_clear(rng, cx, cy, r_min, r_max, peer_xy, obstacles, max_tries: int = 80):
    """Same as _scatter, but rejection-samples against `obstacles`: tries up
    to max_tries points, and keeps the one whose straight-line path to
    peer_xy crosses the FEWEST of them (0 if any candidate manages that),
    rather than accepting whatever the last attempt happened to land on --
    with a real obstacle field, "keep re-rolling and hope the final roll is
    clear" wastes every earlier attempt that was actually better. Without
    this, Military Zone's own scenery (checkpoint bunkers, the supply
    depot) routinely sat directly between a relay and the AP, or a STA and
    its assigned relay -- with this PHY config's thin ~6.5 dB link margin
    at a few hundred metres and shadowing off by default, crossing even
    one obstacle didn't just weaken that link, it killed it permanently
    (see E118)."""
    if not obstacles:
        return _scatter(rng, cx, cy, r_min, r_max)
    best_pos, best_crossings = None, None
    for _ in range(max_tries):
        pos = _scatter(rng, cx, cy, r_min, r_max)
        crossings = sum(1 for o in obstacles if o.intersects_segment(pos, peer_xy))
        if crossings == 0:
            return pos
        if best_crossings is None or crossings < best_crossings:
            best_pos, best_crossings = pos, crossings
    return best_pos


def _scatter_clear_sector(rng, cx, cy, r_min, r_max, angle_center, angle_half_width,
                           peer_xy, obstacles, max_tries: int = 40):
    """Same idea as _scatter_clear, but the candidate angle is drawn from
    [angle_center-angle_half_width, angle_center+angle_half_width] instead
    of the full circle -- "optimal" relay placement gives each relay its
    own angular sector so N relays land 360/N apart instead of each being
    independently randomised (and possibly clustered together, leaving
    the rest of the region with no nearby relay at all)."""
    def sample():
        r = math.sqrt(rng.uniform(0.0, 1.0) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
        a = angle_center + rng.uniform(-angle_half_width, angle_half_width)
        return (cx + r * math.cos(a), cy + r * math.sin(a))

    if not obstacles:
        return sample()
    best_pos, best_crossings = None, None
    for _ in range(max_tries):
        pos = sample()
        crossings = sum(1 for o in obstacles if o.intersects_segment(pos, peer_xy))
        if crossings == 0:
            return pos
        if best_crossings is None or crossings < best_crossings:
            best_pos, best_crossings = pos, crossings
    return best_pos


def _scatter_clear_spaced(rng, cx, cy, r_min, r_max, peer_xy, obstacles,
                           placed, min_gap, max_tries: int = 80):
    """Same as _scatter_clear, but also rejection-samples against every
    position already in `placed`: a candidate within min_gap of one of
    them scores as bad as an obstacle crossing, so independently-placed
    STAs stop landing almost on top of each other by pure chance. That
    coincidence is common, not rare -- order statistics for even a modest
    handful of uniformly-random points on a circle routinely produces one
    close pair (confirmed directly: a real Military Zone run had two STAs
    assigned to the same relay land 1 degree apart), and at map scale a
    real-world gap of a few dozen metres between two markers plus their
    radio links to the same relay is indistinguishable on screen from
    them sitting on top of each other. Obstacle-free still wins outright
    over merely-crowded (a blocked link is a real functional problem;
    visual crowding is not), so this only breaks ties among otherwise
    equally-clear candidates rather than fighting obstacle avoidance for
    priority."""
    best_pos, best_score = None, None
    for _ in range(max_tries):
        pos = _scatter(rng, cx, cy, r_min, r_max)
        crossings = sum(1 for o in obstacles if o.intersects_segment(pos, peer_xy)) if obstacles else 0
        crowded = any(math.hypot(pos[0] - p[0], pos[1] - p[1]) < min_gap for p in placed)
        if crossings == 0 and not crowded:
            return pos
        score = crossings * 1000 + (1 if crowded else 0)
        if best_score is None or score < best_score:
            best_pos, best_score = pos, score
    return best_pos


def _seed_default_layout(sim, mode: str, relay_ids: Set[int], obstacles=None,
                          relay_placement: str = "optimal") -> None:
    """Randomly scatter nodes around the AP (in metres) the first time a
    topology is built, so distance-based PHY behaviour starts realistic
    instead of every node sitting on top of the AP at (0, 0). Freshly
    randomised on every call (build or "Reset Layout"), not tied to the
    simulation's own RNG stream -- so it never perturbs simulated traffic
    / backoff / channel randomness.

    `obstacles`, if given, biases each relay/STA away from spots where its
    own real link (relay->AP backhaul, STA->its assigned relay) would
    cross one -- pass None (not sim.obstacles) on any call made before
    this canvas's view transform has settled against real node positions;
    see sync_from_sim's two-pass call for why that distinction matters.

    `relay_placement` only affects where the relays themselves land (STAs
    always scatter around their assigned relay the same way either mode):
    "optimal" spaces them 360/N degrees apart so their combined coverage
    reaches every direction around the AP instead of leaving gaps; "random"
    keeps each relay's own angle independently randomised, which can
    cluster relays together by chance -- useful for deliberately testing
    an uneven/clustered deployment."""
    nodes = sim.nodes
    ap = nodes.get(0)
    if ap is not None:
        ap.pos = (0.0, 0.0)

    rng = random.Random()
    r_ap = _ap_range_m(sim)
    obstacles = obstacles or []

    if mode in ("relay", "aerial_relay", "aerial_relay_uav", "relay_uav") and relay_ids:
        relay_list = sorted(relay_ids)
        rel_assign = sim.config.get("topology", {}).get("relay_assignment", {})
        if relay_placement == "optimal":
            slot = 2.0 * math.pi / len(relay_list)
            # Random starting orientation -- "optimal" only promises even
            # RELATIVE spacing (360/N apart), not a fixed absolute
            # direction, so repeated rebuilds/resets don't always plant
            # relay #1 due north.
            start_angle = rng.uniform(0.0, 2.0 * math.pi)
            # +/-15% of each slot -- enough wiggle room to dodge an
            # obstacle without undoing the point of "optimal": at 0.35
            # (tried first) two adjacent relays' independent jitter could
            # still swing their gap anywhere from 27 to 153 degrees on a
            # 4-relay/90-degree-slot layout, barely better than fully
            # random. 0.15 bounds that same case to roughly 63-117.
            for i, rid in enumerate(relay_list):
                nodes[rid].pos = _scatter_clear_sector(
                    rng, 0.0, 0.0,
                    _RELAY_SCATTER_MIN_FRAC * r_ap, _RELAY_SCATTER_MAX_FRAC * r_ap,
                    start_angle + i * slot, slot * 0.15,
                    (0.0, 0.0), obstacles)
        else:
            for rid in relay_list:
                nodes[rid].pos = _scatter_clear(rng, 0.0, 0.0,
                                                 _RELAY_SCATTER_MIN_FRAC * r_ap,
                                                 _RELAY_SCATTER_MAX_FRAC * r_ap,
                                                 (0.0, 0.0), obstacles)

        if mode in ("aerial_relay_uav", "relay_uav"):
            # STAs are UAV end-nodes here too -- seed them in the exact
            # same AP-centered roam annulus advance_uav_positions moves
            # them within on every subsequent tick (matching plain "uav"
            # mode's own seeding below), not the relay-centered scatter
            # "relay"/"aerial_relay" use for stationary ground STAs --
            # otherwise a STA's very first flight leg would jump from a
            # relay-centered seed position straight to a random
            # AP-relative target, a visible snap on the very first tick.
            # Applies just as much when the relay itself is grounded
            # ("relay_uav") -- it's the STA's own mobility that matters
            # here, not the relay's. Uses the wider _UAV_RELAY_ROAM_MAX_FRAC
            # ceiling (relays definitely exist in this branch), matching
            # what advance_uav_positions will fly these STAs within on
            # every subsequent tick -- see _uav_roam_bounds.
            min_m, max_m = _UAV_ROAM_MIN_FRAC * r_ap, _UAV_RELAY_ROAM_MAX_FRAC * r_ap
            for nid in nodes:
                if nid == 0 or nid in relay_ids:
                    continue
                nodes[nid].pos = _scatter(rng, 0.0, 0.0, min_m, max_m)
        else:
            # min_gap floor of 30m keeps tiny-range configs (a small r_ap
            # from a short-range PHY config) from getting an unreasonably
            # large minimum spacing forced on them; scales up for the
            # normal case so it stays a small fraction of the actual
            # scatter annulus rather than a fixed value that's trivial at
            # 2600m AP range and dominant at 300m.
            min_gap = max(30.0, 0.03 * r_ap)
            placed_by_relay: Dict[int, list] = {rid: [] for rid in relay_list}
            for nid, n in nodes.items():
                if nid == 0 or nid in relay_ids:
                    continue
                rid = rel_assign.get(nid, relay_list[0])
                rcx, rcy = nodes[rid].pos
                pos = _scatter_clear_spaced(rng, rcx, rcy,
                                             _RELAY_STA_MIN_FRAC * r_ap,
                                             _RELAY_STA_MAX_FRAC * r_ap,
                                             (rcx, rcy), obstacles,
                                             placed_by_relay[rid], min_gap)
                nodes[nid].pos = pos
                placed_by_relay[rid].append(pos)

        if mode in ("aerial_relay", "aerial_relay_uav"):
            # Ground STAs keep their random scatter (they don't fly); the
            # relay-role nodes are aerial relays, so put them on their
            # racetrack immediately instead of waiting for the first tick.
            advance_drone_positions(sim, relay_ids, 0.0)
    elif mode == "uav":
        # Seed each UAV end-node inside the exact same roam annulus
        # advance_uav_positions later confines it to (not the wider, generic
        # STA-scatter span below) -- otherwise a UAV could start outside its
        # own drawn roam boundary and only work its way back inside on its
        # first flight leg, which is honest about staying under the AP's
        # real range but not about staying within the *advertised* roam
        # circle every UAV is documented to move within from the start.
        min_m, max_m = _UAV_ROAM_MIN_FRAC * r_ap, _UAV_ROAM_MAX_FRAC * r_ap
        for nid in nodes:
            if nid == 0:
                continue
            nodes[nid].pos = _scatter(rng, 0.0, 0.0, min_m, max_m)
    else:
        min_gap = max(30.0, 0.03 * r_ap)
        placed: list = []
        for nid in nodes:
            if nid == 0:
                continue
            pos = _scatter_clear_spaced(rng, 0.0, 0.0,
                                         _STA_SCATTER_MIN_FRAC * r_ap,
                                         _STA_SCATTER_MAX_FRAC * r_ap,
                                         (0.0, 0.0), obstacles, placed, min_gap)
            nodes[nid].pos = pos
            placed.append(pos)


def _declutter_obstructed_nodes(sim, mode: str, relay_ids: Set[int]) -> int:
    """Repositions any stationary node whose straight-line path to its own
    peer newly crosses an obstacle, using the same obstacle-aware
    _scatter_clear _seed_default_layout uses at initial seed time.

    Exists for exactly one situation: the interactive GUI's Environment
    dropdown can be changed while a sim is already RUNNING (see dashboard_
    tk.py's _on_env_change), which swaps in a new environment's real
    buildings without moving any node -- a node whose position was chosen
    with the OLD environment's (possibly empty) obstacle field in mind can
    end up with a brand new building sitting directly between it and its
    peer. That's a permanent, physically genuine block: association.py's
    beacon-liveness/re-scan retry logic keeps working exactly as designed
    (it isn't the bug), but no amount of retrying decodes a beacon whose
    RSSI is pinned below sensitivity by a wall that never moves, and
    nothing else in this module ever repositions an already-placed node.
    Without this, a node in that spot reads as permanently, inexplicably
    "not trying to rejoin" with no recovery path at all.

    Only handles nodes _seed_default_layout itself treats as obstacle-
    aware: relays (unless airborne -- advance_drone_positions overwrites
    an aerial relay's position every tick regardless of what's set here)
    and non-UAV ground STAs. UAV end-nodes re-roll their own position
    every tick via advance_uav_positions and were never obstacle-aware
    even at initial seed time, so there's nothing stale to correct for
    them specifically. Returns the number of nodes moved, for logging."""
    obstacles = getattr(sim, "obstacles", None)
    if not obstacles:
        return 0
    aerial_relay = mode in ("aerial_relay", "aerial_relay_uav")
    uav_end_node = mode in ("uav", "aerial_relay_uav", "relay_uav")
    rel_assign = sim.config.get("topology", {}).get("relay_assignment", {})
    r_ap = _ap_range_m(sim)
    rng = random.Random()
    moved = 0
    for nid, n in sim.nodes.items():
        if nid == 0:
            continue
        if nid in relay_ids:
            if aerial_relay:
                continue
            peer_xy = (0.0, 0.0)
            wide_min, wide_max = _RELAY_SCATTER_MIN_FRAC * r_ap, _RELAY_SCATTER_MAX_FRAC * r_ap
        elif uav_end_node:
            continue
        elif mode in ("relay", "aerial_relay") and relay_ids:
            rn = sim.nodes.get(rel_assign.get(nid, sorted(relay_ids)[0]))
            peer_xy = tuple(rn.pos) if rn is not None else (0.0, 0.0)
            wide_min, wide_max = _RELAY_STA_MIN_FRAC * r_ap, _RELAY_STA_MAX_FRAC * r_ap
        else:
            peer_xy = (0.0, 0.0)
            wide_min, wide_max = _STA_SCATTER_MIN_FRAC * r_ap, _STA_SCATTER_MAX_FRAC * r_ap

        if not any(o.intersects_segment(n.pos, peer_xy) for o in obstacles):
            continue

        cur_r = math.hypot(n.pos[0] - peer_xy[0], n.pos[1] - peer_xy[1])
        r_min = max(1.0, cur_r * 0.85)
        r_max = max(r_min + 1.0, cur_r * 1.15)
        new_pos = _scatter_clear(rng, peer_xy[0], peer_xy[1], r_min, r_max, peer_xy, obstacles)
        if any(o.intersects_segment(new_pos, peer_xy) for o in obstacles):
            # The narrow band right around this node's old spot had no
            # genuinely clear angle at all (a big building, or a bad-luck
            # radius) -- fall back to the same full-width scatter range
            # _seed_default_layout itself would use for this node type,
            # trading "stays near its old position" for "actually finds a
            # clear line of sight" once the local neighbourhood can't
            # offer one.
            new_pos = _scatter_clear(rng, peer_xy[0], peer_xy[1], wide_min, wide_max,
                                      peer_xy, obstacles, max_tries=160)
        n.pos = new_pos
        moved += 1
    return moved


_TOPOLOGY_FILE_VERSION = 1


def save_topology(sim, path: str) -> None:
    """Serializes the current node layout (positions, roles, relay
    assignment) to a JSON file -- deliberately independent of Environment/
    Layout, which live entirely in the GUI/topology_canvas.py and are
    never written here. The point is re-loading the exact same physical
    topology (same node positions, same which-STA-goes-with-which-relay)
    under a *different* environment later (see load_topology/
    apply_topology below) to compare environments apples-to-apples,
    instead of every environment switch implicitly re-randomising the
    scatter too and confounding "the environment changed the result" with
    "the random layout changed the result"."""
    topo_cfg = sim.config.get("topology", {})
    relay_ids = set(int(i) for i in topo_cfg.get("relay_ids", []))
    data = {
        "sim11ah_topology_version": _TOPOLOGY_FILE_VERSION,
        "mode": topo_cfg.get("mode", "star"),
        "num_stas": sum(1 for nid in sim.nodes if nid != 0 and nid not in relay_ids),
        "num_relays": len(relay_ids),
        "relay_placement": topo_cfg.get("relay_placement", "optimal"),
        "relay_ids": sorted(relay_ids),
        "relay_assignment": {
            str(k): int(v) for k, v in topo_cfg.get("relay_assignment", {}).items()
        },
        "positions": {
            str(nid): [float(n.pos[0]), float(n.pos[1])] for nid, n in sim.nodes.items()
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_topology(path: str) -> dict:
    """Parses a topology JSON file written by save_topology. Returns the
    raw dict -- applying it onto a live Simulator is the separate
    apply_topology step below, since the sim generally needs to be
    (re)built with this data's num_stas/num_relays/mode FIRST so the node
    IDs it's about to receive positions for actually exist."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "positions" not in data or "mode" not in data:
        raise ValueError(f"{path!r} doesn't look like a sim11ah topology file")
    return data


def apply_topology(sim, data: dict) -> None:
    """Applies a loaded topology's positions (and relay assignment, purely
    informational post-seed -- see _seed_default_layout's own use of it)
    onto an already-built sim. Silently skips any node ID present in the
    file but not in this sim (or vice versa) instead of raising, so a
    topology saved at a different STA/relay count still applies whatever
    of it still fits rather than hard-failing -- callers that need an
    exact match (the GUI's Load Topology button) rebuild with the file's
    own num_stas/num_relays/mode first specifically to avoid relying on
    this leniency."""
    for nid_str, pos in data.get("positions", {}).items():
        try:
            nid = int(nid_str)
        except (TypeError, ValueError):
            continue
        node = sim.nodes.get(nid)
        if node is not None and isinstance(pos, (list, tuple)) and len(pos) == 2:
            node.pos = (float(pos[0]), float(pos[1]))

    relay_assignment = data.get("relay_assignment")
    if relay_assignment:
        sim.config.setdefault("topology", {})["relay_assignment"] = {
            int(k): int(v) for k, v in relay_assignment.items()
            if int(k) in sim.nodes and int(v) in sim.nodes
        }


def advance_drone_positions(sim, drone_ids: Set[int], t: float) -> None:
    """Fly every drone-role node along a shared elliptical racetrack
    (centred on the AP by default, or sim._relay_path_cx_m/_cy_m if set --
    see _drone_racetrack_geometry), phase-offset so multiple drones spread
    out along the loop instead of stacking. Position is a pure function of
    simulated time (not integrated from the previous position), so it's
    deterministic and can never drift or accumulate error."""
    cx, cy, rx, ry, period_s = _drone_racetrack_geometry(sim)
    ids = sorted(drone_ids)
    n = max(len(ids), 1)
    for i, did in enumerate(ids):
        node = sim.nodes.get(did)
        if node is None:
            continue
        phase = 2.0 * math.pi * i / n
        theta = 2.0 * math.pi * (float(t) / period_s) + phase
        node.pos = (cx + rx * math.cos(theta), cy + ry * math.sin(theta))


def advance_uav_positions(sim, uav_ids: Set[int], dt: float) -> None:
    """Random-waypoint mobility for UAV *end nodes*: each UAV independently
    picks a random point within an annulus around the AP, flies straight
    toward it at its own cruise speed, then (on arrival) picks a new random
    target and speed -- unlike advance_drone_positions's shared racetrack,
    every UAV wanders its own unrelated path.

    Cruise speed is user-controllable and takes effect immediately, even
    mid-flight: if the caller sets ``sim._uav_speed_mps`` (e.g. from a GUI
    slider), that value is read fresh every call and combined with a fixed
    per-UAV +/-15% multiplier (chosen once per UAV, not re-picked every
    tick, so different UAVs keep visibly different speeds rather than all
    moving in lockstep). Falls back to the original 4-14 m/s random spread
    if unset. Actual per-tick speed is therefore never cached -- only the
    multiplier is -- so moving the slider changes every UAV's speed on the
    very next tick instead of waiting for it to reach its current target.

    Needs persistent per-UAV state (current target / speed multiplier)
    across calls, since position is integrated from dt rather than a pure
    function of absolute time; that state is stashed directly on the
    Simulator object (``_uav_targets`` / ``_uav_speed_factors``, both plain
    dicts) so nothing here needs its own long-lived object -- callers just
    pass the current dt each tick."""
    if dt <= 0.0 or not uav_ids:
        return

    targets = getattr(sim, "_uav_targets", None)
    if targets is None:
        targets = {}
        sim._uav_targets = targets
    factors = getattr(sim, "_uav_speed_factors", None)
    if factors is None:
        factors = {}
        sim._uav_speed_factors = factors

    ap = sim.nodes.get(0)
    acx, acy = ap.pos if ap is not None else (0.0, 0.0)
    rng = random.Random()
    roam_min, roam_max = _uav_roam_bounds(sim)

    base_speed = getattr(sim, "_uav_speed_mps", None)
    have_base = bool(base_speed and base_speed > 0)

    def _current_speed(uid):
        if have_base:
            return float(base_speed) * factors[uid]
        return _UAV_SPEED_MIN_MPS + factors[uid] * (_UAV_SPEED_MAX_MPS - _UAV_SPEED_MIN_MPS)

    for uid in uav_ids:
        node = sim.nodes.get(uid)
        if node is None:
            continue

        if uid not in targets:
            targets[uid] = _scatter(rng, acx, acy, roam_min, roam_max)
        if uid not in factors:
            factors[uid] = rng.uniform(0.85, 1.15) if have_base else rng.random()

        tx, ty = targets[uid]
        x, y = node.pos
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)

        if dist <= _UAV_ARRIVE_EPS_M:
            targets[uid] = _scatter(rng, acx, acy, roam_min, roam_max)
            factors[uid] = rng.uniform(0.85, 1.15) if have_base else rng.random()
            continue

        step = _current_speed(uid) * float(dt)
        if step >= dist:
            node.pos = (tx, ty)
        else:
            node.pos = (x + dx / dist * step, y + dy / dist * step)


class NetworkCanvas(tk.Canvas):
    """Draggable node-map. Left-click selects a node (fires ``on_select``),
    left-drag repositions it live. No right-click menus are used anywhere
    in this view -- per-node settings are edited via the caller's side
    panel, driven by the ``on_select`` callback."""

    ENVIRONMENTS = ("Open Area", "Paddy Field", "Industrial Site", "Smart City", "Military Zone")
    _MIN_SCALE = 0.15   # px per metre
    _MAX_SCALE = 60.0

    def __init__(self, parent, on_select: Optional[Callable[[Optional[int]], None]] = None,
                 environment: str = "Open Area", **kw):
        kw.setdefault("bg", _BG)
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", _BORDER)
        super().__init__(parent, **kw)

        self.sim = None
        self.selected_id: Optional[int] = None
        # Multi-node drag: nodes currently in the drag GROUP (drag-starting
        # on any one of them moves all of them together, preserving their
        # relative offsets). Separate from selected_id, which just tracks
        # which one node the info panel is currently showing -- a group can
        # be selected while the panel still shows details for whichever
        # member was clicked/rubber-band-selected last. Ctrl/Shift-click
        # toggles individual nodes in/out; drag-starting on empty space
        # rubber-band-selects everything inside the dragged rectangle
        # (replacing the previous selection). The AP (id 0) can never be a
        # member -- same "never draggable" rule single-node drag already
        # enforced.
        self._selected_ids: Set[int] = set()
        self._relay_ids: Set[int] = set()
        self.drone_ids: Set[int] = set()
        self.uav_ids: Set[int] = set()
        self.car_ids: Set[int] = set()
        self.scooter_ids: Set[int] = set()
        self._ap_ids: Set[int] = {0}
        self._drone_trails: Dict[int, list] = {}
        self._drag_id: Optional[int] = None
        # Group-drag state: which nodes are actively being dragged right
        # now (a snapshot of _selected_ids taken at drag-start, since the
        # user could in principle Ctrl-click mid-drag -- though the mouse
        # button being held down for B1-Motion makes that unlikely) plus
        # each one's world position when the drag started, and the mouse's
        # own starting world position -- every motion event just re-applies
        # the same (start_pos + total_mouse_delta) to each member, rather
        # than accumulating per-tick deltas that could drift from float
        # rounding over a long drag.
        self._drag_ids: Optional[Set[int]] = None
        self._drag_start_positions: Dict[int, tuple] = {}
        self._drag_anchor_world: Optional[tuple] = None
        # Rubber-band select state -- set while a drag that started on
        # empty space (no node under the cursor) is in progress.
        self._rubber_start_px: Optional[tuple] = None
        self._rubber_rect_id: Optional[int] = None
        self._on_select = on_select
        self.environment = environment if environment in self.ENVIRONMENTS else "Open Area"
        self.layout_variant: int = 1
        self._scene_obstacles: list = []
        self._visual_obstacles: list = []

        # Manual zoom/pan. None means "auto-fit to the current node spread"
        # (the original behaviour); once the user zooms (wheel or +/-
        # buttons) these lock in an explicit scale/centre, like a normal
        # map view, until "Fit to View" clears them again.
        self._manual_scale: Optional[float] = None
        self._view_center: Optional[tuple] = None

        # Live packet-transmission animation state. Unicast frames fly from
        # tx to rx as a short comet-trail dot; broadcasts (beacons etc, no
        # single rx) pulse as an expanding ring from the sender instead.
        self._packets: list = []
        self._pulses: list = []

        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)     # Windows / macOS
        self.bind("<Button-4>", self._on_wheel)       # Linux scroll up
        self.bind("<Button-5>", self._on_wheel)       # Linux scroll down

    # ── Public API ────────────────────────────────────────────────────────
    def set_environment(self, environment: str) -> None:
        if environment in self.ENVIRONMENTS:
            self.environment = environment
            self._redraw()

    def set_layout_variant(self, variant: int) -> None:
        if variant in (1, 2, 3):
            self.layout_variant = variant
            self._redraw()

    # ── Manual zoom ───────────────────────────────────────────────────────
    def _ensure_manual_view(self) -> None:
        if self._manual_scale is None:
            self._manual_scale = self._transform()[0]
        if self._view_center is None:
            xmin, xmax, ymin, ymax = self._bounds()
            self._view_center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    def zoom_in(self) -> None:
        self._ensure_manual_view()
        self._manual_scale = min(self._manual_scale * 1.3, self._MAX_SCALE)
        self._redraw()

    def zoom_out(self) -> None:
        self._ensure_manual_view()
        self._manual_scale = max(self._manual_scale / 1.3, self._MIN_SCALE)
        self._redraw()

    def zoom_reset(self) -> None:
        """Return to auto-fit: always shows every node, re-centring itself
        as nodes move (drones flying, drags, rebuilds)."""
        self._manual_scale = None
        self._view_center = None
        self._redraw()

    def _on_wheel(self, event) -> None:
        if self.sim is None:
            return
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", None)
        if num == 5 or delta < 0:
            factor = 1.0 / 1.15
        elif num == 4 or delta > 0:
            factor = 1.15
        else:
            return

        self._ensure_manual_view()
        world_before = self._px_to_world(event.x, event.y)
        new_scale = min(max(self._manual_scale * factor, self._MIN_SCALE), self._MAX_SCALE)
        self._manual_scale = new_scale

        W = self.winfo_width() or int(self.cget("width") or 400)
        H = self.winfo_height() or int(self.cget("height") or 400)
        cx, cy = W / 2.0, H / 2.0
        new_wcx = world_before[0] - (event.x - cx) / new_scale
        new_wcy = world_before[1] - (cy - event.y) / new_scale
        self._view_center = (new_wcx, new_wcy)
        self._redraw()

    def sync_from_sim(self, sim) -> None:
        self.sim = sim
        _apply_environment_path_loss(sim, self.environment)
        topo_cfg = sim.config.get("topology", {}) if sim is not None else {}
        self._relay_ids = set(topo_cfg.get("relay_ids", []))
        mode = topo_cfg.get("mode", "star")
        relay_placement = topo_cfg.get("relay_placement", "optimal")
        # ap_ids is only ever set by MultiApBuilder/CarsUavsBuilder (mirrors
        # relay_ids' own convention) -- every other topology has exactly
        # one AP, at node 0, same as always.
        self._ap_ids = set(topo_cfg.get("ap_ids", [0]))
        self.drone_ids = set(self._relay_ids) if mode in ("aerial_relay", "aerial_relay_uav") else set()
        if mode == "cars_uavs":
            # Real ids from the builder, not "everything that isn't node
            # 0" -- that guess would misclassify the OTHER APs (node ids
            # 1..K-1) as end nodes under multi-AP.
            self.car_ids = set(topo_cfg.get("car_ids", []))
            self.scooter_ids = set(topo_cfg.get("scooter_ids", []))
            self.uav_ids = set(topo_cfg.get("uav_ids", []))
        else:
            self.car_ids = set()
            self.scooter_ids = set()
            self.uav_ids = (
                set(nid for nid in sim.nodes if nid != 0 and nid not in self._relay_ids)
                if mode in ("uav", "aerial_relay_uav", "relay_uav") else set()
            )
        self._drone_trails.clear()
        self._manual_scale = None    # a rebuilt/reset topology re-fits the view
        self._view_center = None

        non_ap = [n for nid, n in sim.nodes.items() if nid != 0] if sim else []
        all_origin = bool(non_ap) and all(tuple(n.pos) == (0.0, 0.0) for n in non_ap)
        if all_origin:
            # _seed_default_layout tries to avoid placing a relay/STA where
            # its own real link to its peer crosses an obstacle -- which
            # needs sim.obstacles already populated. That only happens as
            # a side effect of _draw_background (inside _redraw) -- BUT
            # _bg_military_v1 (and friends) place obstacles in world metres
            # via frect/wrect, which read this canvas's *current view
            # transform* (self._transform(), auto-fit to the node cloud's
            # bounding box). With every node still literally on the AP,
            # that box is degenerate, so drawing here first would compute
            # obstacle positions against a bogus transform -- tried that,
            # and it made Military Zone's own command-bunker (which
            # _encloses_ap is specifically supposed to exclude for
            # covering the AP) land inconsistently on or off the AP
            # between runs, worse than not compensating for obstacles at
            # all. So: seed once WITHOUT obstacle-awareness first (gets
            # nodes to a realistic spatial extent so the view settles
            # properly), draw once for real (settles the transform AND
            # populates sim.obstacles against it), then re-seed for real
            # with obstacle avoidance now that both are trustworthy.
            _seed_default_layout(sim, mode, self._relay_ids, obstacles=None,
                                  relay_placement=relay_placement)
            self._redraw()
            _seed_default_layout(sim, mode, self._relay_ids,
                                  obstacles=getattr(sim, "obstacles", None),
                                  relay_placement=relay_placement)

        if sim is None or self.selected_id not in sim.nodes:
            self.selected_id = None
        self._selected_ids = {i for i in self._selected_ids if sim is not None and i in sim.nodes}
        self._drag_id = None
        self._drag_ids = None
        self._rubber_start_px = None
        self._redraw()
        if self._on_select:
            self._on_select(self.selected_id)

    def refresh(self) -> None:
        self._redraw()

    def reset_layout(self) -> None:
        if self.sim is None:
            return
        for n in self.sim.nodes.values():
            n.pos = (0.0, 0.0)
        self.sync_from_sim(self.sim)

    # ── Packet-transmission animation ────────────────────────────────────
    def add_packet(self, tx_id: int, rx_id: int, kind: str) -> None:
        if self.sim is None or tx_id not in self.sim.nodes or rx_id not in self.sim.nodes:
            return
        if len(self._packets) >= _MAX_ACTIVE_PACKETS:
            self._packets.pop(0)
        self._packets.append({"tx": tx_id, "rx": rx_id, "kind": kind, "p": 0.0})

    def add_broadcast_pulse(self, tx_id: int, kind: str) -> None:
        if self.sim is None or tx_id not in self.sim.nodes:
            return
        if len(self._pulses) >= _MAX_ACTIVE_PULSES:
            self._pulses.pop(0)
        self._pulses.append({"tx": tx_id, "kind": kind, "p": 0.0})

    def clear_packets(self) -> None:
        self._packets.clear()
        self._pulses.clear()

    def tick_animations(self) -> None:
        """Advance in-flight packet/pulse animations by one frame and repaint
        just those overlay items (cheap -- does not touch the background,
        edges or the static node markers, so it's safe to call every GUI
        tick). Also repaints anything else that can change between full
        redraws: drone positions (pure function of sim time) and the
        selected node's highlight ring / range circle."""
        for pk in self._packets:
            pk["p"] += _PKT_STEP
        self._packets = [pk for pk in self._packets if pk["p"] < 1.0]
        for pl in self._pulses:
            pl["p"] += _PKT_STEP * 0.6
        self._pulses = [pl for pl in self._pulses if pl["p"] < 1.0]
        self._redraw_overlay()

    def _redraw_overlay(self) -> None:
        self.delete("ovl")
        if self.sim is None:
            return
        nodes = self.sim.nodes

        self._draw_selection_overlay(nodes)
        self._draw_drones_overlay(nodes)
        self._draw_uavs_overlay(nodes)
        self._draw_cars_overlay(nodes)
        self._draw_scooters_overlay(nodes)
        self._draw_vehicles_overlay(nodes)
        self._draw_military_overlay(nodes)

        for pl in self._pulses:
            n = nodes.get(pl["tx"])
            if n is None:
                continue
            px, py = self._world_to_px(*n.pos)
            r = 6.0 + pl["p"] * 46.0
            color = _PKT_COLORS.get(pl["kind"], _AMBER)
            self.create_oval(px - r, py - r, px + r, py + r,
                              outline=color, width=max(1, int(3 * (1.0 - pl["p"]))),
                              tags=("ovl",))

        for pk in self._packets:
            tx = nodes.get(pk["tx"])
            rx = nodes.get(pk["rx"])
            if tx is None or rx is None:
                continue
            p = pk["p"]
            tail = max(0.0, p - 0.22)
            x0 = tx.pos[0] + (rx.pos[0] - tx.pos[0]) * tail
            y0 = tx.pos[1] + (rx.pos[1] - tx.pos[1]) * tail
            x1 = tx.pos[0] + (rx.pos[0] - tx.pos[0]) * p
            y1 = tx.pos[1] + (rx.pos[1] - tx.pos[1]) * p
            px0, py0 = self._world_to_px(x0, y0)
            px1, py1 = self._world_to_px(x1, y1)
            color = _PKT_COLORS.get(pk["kind"], _BLUE)
            self.create_line(px0, py0, px1, py1, fill=color, width=3, tags=("ovl",))
            self.create_oval(px1 - 4, py1 - 4, px1 + 4, py1 + 4,
                              fill=color, outline="white", width=1, tags=("ovl",))

    def _draw_selection_overlay(self, nodes) -> None:
        if self.selected_id is None or self.selected_id not in nodes:
            return
        n = nodes[self.selected_id]
        px, py = self._world_to_px(*n.pos)
        if self.selected_id == 0:
            r = 14
        elif self.selected_id in self.drone_ids or self.selected_id in self.uav_ids:
            r = 15
        elif self.selected_id in self._relay_ids:
            r = 11
        else:
            r = 8
        self.create_oval(px - r - 4, py - r - 4, px + r + 4, py + r + 4,
                          outline=_AMBER, width=2, tags=("ovl",))
        rng = range_m_for_node(n)
        if rng > 0:
            scale = self._transform()[0]
            r_px = min(rng * scale, 4000.0)
            self.create_oval(px - r_px, py - r_px, px + r_px, py + r_px,
                              outline=_AMBER, dash=(4, 3), width=1.5, tags=("ovl",))

    def _draw_drones_overlay(self, nodes) -> None:
        if not self.drone_ids:
            return
        ap = nodes.get(0)
        rel_assign = self.sim.config.get("topology", {}).get("relay_assignment", {})
        by_drone: Dict[int, list] = {}
        for nid, n in nodes.items():
            # uav_ids STAs (aerial_relay_uav mode) get their own line drawn
            # by _draw_uavs_overlay instead, from each UAV's own real live
            # peer -- skip them here so the same relay<->STA edge doesn't
            # get drawn twice.
            if nid == 0 or nid in self.drone_ids or nid in self.uav_ids:
                continue
            did = rel_assign.get(nid)
            if did in self.drone_ids:
                by_drone.setdefault(did, []).append(nid)

        if ap is not None:
            scale = self._transform()[0]
            loop_cx, loop_cy, rx, ry, _ = _drone_racetrack_geometry(self.sim)
            ecx, ecy = self._world_to_px(loop_cx, loop_cy)
            self.create_oval(ecx - rx * scale, ecy - ry * scale,
                              ecx + rx * scale, ecy + ry * scale,
                              outline=_DRONE_PATH, dash=(3, 4), width=1, tags=("ovl",))

        for did in sorted(self.drone_ids):
            dn = nodes.get(did)
            if dn is None:
                continue
            dpx, dpy = self._world_to_px(*dn.pos)

            # by_drone (from the static topology-build-time relay_assignment)
            # is only the *candidate* set each drone was scattered near --
            # the line itself is drawn from each node's real, live
            # _assoc_peer_id, same rule the ground-node edges above use, so
            # a drone still authenticating with the AP (or a STA that ended
            # up elsewhere) doesn't get drawn as already connected.
            if ap is not None and _assoc_peer(dn) == 0:
                apx, apy = self._world_to_px(*ap.pos)
                self.create_line(apx, apy, dpx, dpy, fill=_DRONE_PATH,
                                  width=2, dash=(4, 2), tags=("ovl",))
            for sid in by_drone.get(did, []):
                sn = nodes.get(sid)
                if sn is None or _assoc_peer(sn) != did:
                    continue
                spx, spy = self._world_to_px(*sn.pos)
                self.create_line(dpx, dpy, spx, spy, fill=_BORDER, width=1, tags=("ovl",))

            trail = self._drone_trails.setdefault(did, [])
            trail.append(dn.pos)
            if len(trail) > 16:
                del trail[0]
            self._draw_fading_trail(trail)

            self._draw_drone_icon(dpx, dpy, did, prefix="D", node=dn)

    def _draw_uavs_overlay(self, nodes) -> None:
        """UAV *end nodes* on a random-waypoint walk: unlike relay drones,
        each one has no children hanging off it and follows its own
        independent path, so there's no shared racetrack to draw -- just a
        faint roam boundary for context, each UAV's own fading trail, a
        link to whatever it's actually associated with, and the same
        quadcopter glyph (labelled "U" instead of "D" to read as an end
        node, not a relay).

        In plain "uav" mode that live peer is always the AP -- there's
        nothing else to associate with. In "aerial_relay_uav" mode (UAV
        end-nodes flying alongside aerial relays) it's just as often a
        relay, so the link is drawn to whichever node id the UAV's real
        _assoc_peer_id resolves to, not hardcoded to the AP.

        The roam-boundary circle below is skipped for "cars_uavs" mode:
        it's anchored on a single AP (node 0), but that mode's UAVs fly
        random-waypoint across the WHOLE multi-AP corridor
        (uav_waypoint_step), not a fixed annulus around one AP -- drawing
        it there would show a boundary the UAVs routinely fly outside of,
        which is misleading rather than just incomplete."""
        if not self.uav_ids:
            return
        mode = self.sim.config.get("topology", {}).get("mode") if self.sim else None
        ap = nodes.get(0)
        if ap is not None and mode != "cars_uavs":
            apx, apy = self._world_to_px(*ap.pos)
            scale = self._transform()[0]
            _, roam_max = _uav_roam_bounds(self.sim)
            self.create_oval(apx - roam_max * scale, apy - roam_max * scale,
                              apx + roam_max * scale, apy + roam_max * scale,
                              outline=_DRONE_PATH, dash=(2, 5), width=1, tags=("ovl",))

        for uid in sorted(self.uav_ids):
            un = nodes.get(uid)
            if un is None:
                continue
            upx, upy = self._world_to_px(*un.pos)

            # Same live-peer rule as everywhere else -- a wandering UAV end
            # node that hasn't actually associated with anything yet
            # shouldn't be drawn already linked. The peer can be the AP or
            # (in aerial_relay_uav mode) a relay, so look it up rather than
            # assuming it's always node 0.
            peer_id = _assoc_peer(un)
            peer = nodes.get(peer_id) if peer_id is not None else None
            if peer is not None:
                ppx, ppy = self._world_to_px(*peer.pos)
                self.create_line(ppx, ppy, upx, upy, fill=_BORDER, width=1, tags=("ovl",))

            trail = self._drone_trails.setdefault(uid, [])
            trail.append(un.pos)
            if len(trail) > 16:
                del trail[0]
            self._draw_fading_trail(trail)

            self._draw_drone_icon(upx, upy, uid, prefix="U", node=un)

    def _draw_cars_overlay(self, nodes) -> None:
        """Cars in "cars_uavs" mode (see sim11ah/topology.py's
        CarsUavsBuilder): real simulator nodes on a live highway_bounce_step
        crossing, unlike _draw_vehicles_overlay's Smart-City traffic (pure
        decoration, no association/physics) -- reuses that method's own
        _draw_car_icon glyph, since the icon itself doesn't care whether
        its position came from a decorative time formula or a real node's
        actual (x, y), just heading and pixel position.

        Heading comes from sim._highway_dirs (the direction flag
        highway_bounce_step maintains, +1/-1 along the corridor's x-axis,
        shared with scooters -- see _draw_scooters_overlay) rather than
        from consecutive positions like the Smart City loop does --
        cheaper, and exact rather than a one-tick-lagged estimate."""
        if not self.car_ids or self.sim is None:
            return
        highway_dirs = getattr(self.sim, "_highway_dirs", {})
        n = len(_REAL_CAR_COLORS)
        for i, cid in enumerate(sorted(self.car_ids)):
            cn = nodes.get(cid)
            if cn is None:
                continue
            cpx, cpy = self._world_to_px(*cn.pos)

            peer_id = _assoc_peer(cn)
            peer = nodes.get(peer_id) if peer_id is not None else None
            if peer is not None:
                ppx, ppy = self._world_to_px(*peer.pos)
                self.create_line(ppx, ppy, cpx, cpy, fill=_BORDER, width=1, tags=("ovl",))

            trail = self._drone_trails.setdefault(cid, [])
            trail.append(cn.pos)
            if len(trail) > 16:
                del trail[0]
            self._draw_fading_trail(trail)

            heading = 0.0 if highway_dirs.get(cid, 1) >= 0 else math.pi
            self._draw_car_icon(cpx, cpy, heading, _REAL_CAR_COLORS[i % n])

    def _draw_scooters_overlay(self, nodes) -> None:
        """Scooters in "cars_uavs" mode: same real-node treatment as
        _draw_cars_overlay (live highway_bounce_step crossing, not
        decoration), just riding the inner lane (closer to the corridor
        centreline -- see CarsUavsBuilder's scooter_lane_offset_m) with
        their own smaller glyph (_draw_scooter_icon) and colour palette
        so the two vehicle kinds stay visually distinct at a glance."""
        if not self.scooter_ids or self.sim is None:
            return
        highway_dirs = getattr(self.sim, "_highway_dirs", {})
        n = len(_REAL_SCOOTER_COLORS)
        for i, sid in enumerate(sorted(self.scooter_ids)):
            sn = nodes.get(sid)
            if sn is None:
                continue
            spx, spy = self._world_to_px(*sn.pos)

            peer_id = _assoc_peer(sn)
            peer = nodes.get(peer_id) if peer_id is not None else None
            if peer is not None:
                ppx, ppy = self._world_to_px(*peer.pos)
                self.create_line(ppx, ppy, spx, spy, fill=_BORDER, width=1, tags=("ovl",))

            trail = self._drone_trails.setdefault(sid, [])
            trail.append(sn.pos)
            if len(trail) > 16:
                del trail[0]
            self._draw_fading_trail(trail)

            heading = 0.0 if highway_dirs.get(sid, 1) >= 0 else math.pi
            self._draw_scooter_icon(spx, spy, heading, _REAL_SCOOTER_COLORS[i % n])

    # ── Smart City traffic (pure scenery -- not simulator nodes) ─────────
    def _rect_loop_pos(self, cx: float, cy: float, hw: float, hh: float, t: float):
        """World-space position at fraction t in [0, 1) going clockwise
        around the perimeter of a rectangle centred at (cx, cy) with
        half-width hw, half-height hh -- a road-following "racetrack" for
        Smart City traffic, the same pure-function-of-time trick used for
        the aerial-relay racetrack, just rectangular instead of elliptical
        so vehicles visibly follow the street grid instead of cutting
        through blocks."""
        perim = 2.0 * (2.0 * hw + 2.0 * hh)
        d = (t % 1.0) * perim
        if d < 2.0 * hw:
            return (cx - hw + d, cy + hh)
        d -= 2.0 * hw
        if d < 2.0 * hh:
            return (cx + hw, cy + hh - d)
        d -= 2.0 * hh
        if d < 2.0 * hw:
            return (cx + hw - d, cy - hh)
        d -= 2.0 * hw
        return (cx - hw, cy - hh + d)

    def _draw_vehicles_overlay(self, nodes) -> None:
        """Cars driving laps around the Smart City street grid. Pure
        decoration -- not simulator nodes, no association/traffic/physics
        -- position is a pure function of sim time (like the drone
        racetrack), redrawn every tick as part of the cheap overlay pass.
        Only active for the Smart City environment; a no-op everywhere
        else, including while a real topology's own drones/UAVs are also
        animating on the same canvas."""
        if self.environment != "Smart City" or self.sim is None:
            return
        xmin, xmax, ymin, ymax = self._bounds()
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        wxs, wys = xmax - xmin, ymax - ymin
        t_now = float(getattr(self.sim.engine, "now", 0.0))

        loops = (
            (cx_w, cy_w, wxs * 0.16, wys * 0.16, 16.0),   # inner avenue, faster lap
            (cx_w, cy_w, wxs * 0.34, wys * 0.34, 26.0),   # outer ring road, slower lap
        )
        n = len(_CITY_VEHICLE_COLORS)
        for i in range(n):
            lcx, lcy, lhw, lhh, period_s = loops[i % len(loops)]
            phase = (i // len(loops)) / max(1, -(-n // len(loops))) + (0.5 if i % 2 else 0.0)
            t = (t_now / period_s + phase) % 1.0
            wx, wy = self._rect_loop_pos(lcx, lcy, lhw, lhh, t)
            wx2, wy2 = self._rect_loop_pos(lcx, lcy, lhw, lhh, t + 0.004)
            px, py = self._world_to_px(wx, wy)
            px2, py2 = self._world_to_px(wx2, wy2)
            heading = math.atan2(py2 - py, px2 - px)
            self._draw_car_icon(px, py, heading, _CITY_VEHICLE_COLORS[i])

    def _draw_car_icon(self, px: float, py: float, heading: float, color: str) -> None:
        """Small top-down car glyph, oriented along its heading: a cast
        shadow, a rotated rectangular body, a cabin/windshield inset, and
        head/tail light accents front and back -- echoes the same
        body/cabin/lights split the 3D car mesh uses (entities.js's
        buildCarBody), just flattened to a minimap-style icon rather than
        the tower/drone glyphs' fuller shadow+shading treatment (a car
        population stays small, so there's room for it, but this is still
        meant to read at a glance, not as a model kit)."""
        length, width = 9.0, 4.6
        ch, sh = math.cos(heading), math.sin(heading)
        perp = heading + math.pi / 2.0
        cp, sp = math.cos(perp), math.sin(perp)

        def _pt(dl, dw):
            return px + dl * ch + dw * cp, py + dl * sh + dw * sp

        self.create_oval(px - length * 0.6 + 2, py - width * 0.6 + 2,
                          px + length * 0.6 + 2, py + width * 0.6 + 2,
                          fill=_SHADOW, outline="", stipple="gray50", tags=("ovl",))

        corners = []
        for dl, dw in ((length * 0.5, -width * 0.5), (length * 0.5, width * 0.5),
                       (-length * 0.5, width * 0.5), (-length * 0.5, -width * 0.5)):
            corners.extend(_pt(dl, dw))
        self.create_polygon(*corners, fill=color, outline="#1f2937", width=1, tags=("ovl",))

        # Cabin/windshield inset -- a smaller, lighter rectangle set back
        # from the nose, not just a single dot, so the glyph reads as "a
        # car" rather than "a rounded rectangle with a headlight".
        cabin = []
        for dl, dw in ((length * 0.18, -width * 0.32), (length * 0.18, width * 0.32),
                       (-length * 0.32, width * 0.32), (-length * 0.32, -width * 0.32)):
            cabin.extend(_pt(dl, dw))
        self.create_polygon(*cabin, fill="#cfe8ff", outline="", stipple="gray25", tags=("ovl",))

        # Head/tail light dots, front and back -- same aviation-style "front
        # is a distinct colour from rear" convention _draw_drone_icon's nav
        # lights already establish for the other moving glyph on this canvas.
        for dl, lcolor in ((length * 0.52, "#fff4d6"), (-length * 0.52, "#ff5c5c")):
            lx, ly = _pt(dl, 0.0)
            self.create_oval(lx - 1.1, ly - 1.1, lx + 1.1, ly + 1.1,
                              fill=lcolor, outline="", tags=("ovl",))

    def _draw_scooter_icon(self, px: float, py: float, heading: float, color: str) -> None:
        """Small top-down scooter glyph -- a slimmer, shorter footprint
        than _draw_car_icon (narrow deck instead of a boxy body, two small
        wheel dots fore/aft instead of a cabin inset) so it reads as a
        distinct, lighter vehicle at a glance rather than just a smaller
        car."""
        length, width = 5.2, 2.2
        ch, sh = math.cos(heading), math.sin(heading)
        perp = heading + math.pi / 2.0
        cp, sp = math.cos(perp), math.sin(perp)

        def _pt(dl, dw):
            return px + dl * ch + dw * cp, py + dl * sh + dw * sp

        self.create_oval(px - length * 0.6 + 1.5, py - width * 0.6 + 1.5,
                          px + length * 0.6 + 1.5, py + width * 0.6 + 1.5,
                          fill=_SHADOW, outline="", stipple="gray50", tags=("ovl",))

        # Narrow deck body -- an elongated rounded rectangle rather than
        # the car's boxy corners, so the silhouette itself already reads
        # as "two-wheeler" before the wheel dots are added.
        body = []
        for dl, dw in ((length * 0.5, -width * 0.35), (length * 0.5, width * 0.35),
                       (-length * 0.5, width * 0.35), (-length * 0.5, -width * 0.35)):
            body.extend(_pt(dl, dw))
        self.create_polygon(*body, fill=color, outline="#1f2937", width=1, tags=("ovl",))

        # Fore/aft wheel dots on the centreline, plus a single headlight --
        # scooters don't get a cabin inset (nothing to put one on).
        for dl, wcolor in ((length * 0.48, "#1f2937"), (-length * 0.48, "#1f2937")):
            wx, wy = _pt(dl, 0.0)
            self.create_oval(wx - 0.9, wy - 0.9, wx + 0.9, wy + 0.9,
                              fill=wcolor, outline="", tags=("ovl",))
        hx, hy = _pt(length * 0.52, 0.0)
        self.create_oval(hx - 0.9, hy - 0.9, hx + 0.9, hy + 0.9,
                          fill="#fff4d6", outline="", tags=("ovl",))

    def _draw_tower_icon(self, px: float, py: float, R: float, arm_count: int,
                          dk_color: str, lt_color: str, hub_fill: str) -> None:
        """Shared top-down radio-mast glyph for AP and relay markers: a
        cast shadow, a hex base plate, `arm_count` radiating sector-antenna
        arms with a panel at each tip, and a small status hub -- replaces
        the old plain labelled circle for both role types, which read as
        flat next to _draw_drone_icon's level of shading/detail once
        cars/UAVs started sharing this canvas. arm_count mirrors
        entities.js's buildApOrRelay(isAp) in the 3D view exactly: 3 arms
        for an AP, 2 for a relay -- same role, same silhouette logic, just
        seen from above here instead of from the side.

        Callers (_draw_ap_icon / _draw_relay_icon) draw the id/role label
        below the glyph themselves, at a y-offset that accounts for R --
        not done here, so this stays purely the physical mast/antenna
        shape with no role-specific text baked in."""
        sh = 3.0
        self.create_oval(px - R * 0.9 + sh, py - R * 0.65 + sh,
                          px + R * 0.9 + sh, py + R * 0.65 + sh,
                          fill=_SHADOW, outline="", stipple="gray50")

        hex_pts = []
        for k in range(6):
            ang = math.pi / 6 + k * math.pi / 3
            hex_pts.extend([px + R * 0.6 * math.cos(ang), py + R * 0.6 * math.sin(ang)])
        self.create_polygon(*hex_pts, fill=dk_color, outline=_SHADOW, width=1)

        for k in range(arm_count):
            ang = k * (2.0 * math.pi / arm_count) - math.pi / 2.0
            ax, ay = px + R * 0.95 * math.cos(ang), py + R * 0.95 * math.sin(ang)
            self.create_line(px, py, ax, ay, fill=dk_color, width=3, capstyle="round")
            perp = ang + math.pi / 2.0
            wing, tip = R * 0.19, R * 0.5
            panel = []
            for dl, dw in ((tip, -wing), (tip, wing), (-tip * 0.25, wing), (-tip * 0.25, -wing)):
                panel.extend([ax + dl * math.cos(ang) + dw * math.cos(perp),
                              ay + dl * math.sin(ang) + dw * math.sin(perp)])
            self.create_polygon(*panel, fill=lt_color, outline=dk_color, width=1)

        hub_r = R * 0.42
        self.create_oval(px - hub_r, py - hub_r, px + hub_r, py + hub_r,
                          fill=hub_fill, outline=dk_color, width=2)

    def _draw_ap_icon(self, px: float, py: float, label: str) -> None:
        """AP marker: the _draw_tower_icon glyph (3 arms) plus a faint
        dashed coverage-ring accent (a hint of the AP's role, not its real
        PHY range circle -- that's the selection-time range circle drawn
        elsewhere) and the role label below. Fixed light-blue hub, not
        _assoc_color(n) -- the AP has no assoc_state of its own to reflect
        (statusHex() in entities.js makes the same exception for the 3D
        view's AP mesh)."""
        R = 16.0
        self.create_oval(px - R * 1.6, py - R * 1.6, px + R * 1.6, py + R * 1.6,
                          outline=_BLUE, width=1, dash=(1, 4))
        self._draw_tower_icon(px, py, R, arm_count=3,
                               dk_color=_BLUE_DK, lt_color=_BLUE, hub_fill="#60a5fa")
        self.create_text(px, py + R * 0.95 + 11, text=label,
                          font=("Arial", 8, "bold"), fill=_BLUE_DK)

    def _draw_relay_icon(self, px: float, py: float, nid: int, node) -> None:
        """Relay marker: the _draw_tower_icon glyph (2 arms, matching
        entities.js's buildApOrRelay(isAp=False)) with the hub filled by
        the relay's own live uplink-association colour -- a relay that
        hasn't associated with the AP yet can't actually relay anything,
        the same "outline = role, fill = MAC state" split the old plain
        circle already used, just on a more detailed glyph now."""
        R = 12.5
        self._draw_tower_icon(px, py, R, arm_count=2,
                               dk_color=_PURPLE_DK, lt_color=_PURPLE, hub_fill=_assoc_color(node))
        self.create_text(px, py + R * 0.95 + 11, text=f"R{nid}",
                          font=("Arial", 8, "bold"), fill=_PURPLE_DK)

    def _draw_fading_trail(self, trail: list) -> None:
        """Contrail-style flight trail: drawn segment-by-segment so it
        tapers from a solid line near the vehicle to a faint, stippled
        wisp at the oldest end -- a muted slate-grey, not the vehicle's own
        accent colour, so it reads as "recent path" rather than part of
        the drone itself."""
        n = len(trail)
        if n < 2:
            return
        for i in range(n - 1):
            age = (n - 2 - i) / max(1, n - 2)   # 0.0 = newest segment, 1.0 = oldest
            p0 = self._world_to_px(*trail[i])
            p1 = self._world_to_px(*trail[i + 1])
            width = max(1, round(3.0 * (1.0 - age)))
            stipple = "" if age < 0.35 else ("gray50" if age < 0.7 else "gray25")
            kw = {"stipple": stipple} if stipple else {}
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_DRONE_TRAIL,
                              width=width, capstyle="round", tags=("ovl",), **kw)

    def _drone_heading_px(self, did: int) -> float:
        """Heading angle (radians, canvas-pixel space) from the drone's
        last two trail points, so the icon visibly points the way it's
        actually flying instead of sitting at a fixed angle. Falls back to
        a diagonal default when there isn't trail history yet."""
        trail = self._drone_trails.get(did)
        if trail and len(trail) >= 2:
            px0, py0 = self._world_to_px(*trail[-2])
            px1, py1 = self._world_to_px(*trail[-1])
            if abs(px1 - px0) > 0.5 or abs(py1 - py0) > 0.5:
                return math.atan2(py1 - py0, px1 - px0)
        return -math.pi / 4.0

    def _draw_drone_icon(self, px: float, py: float, did: int, prefix: str = "D",
                          node=None) -> None:
        """Quadcopter glyph seen from above, oriented along its actual
        flight heading. Matte carbon-fibre body (not toy-bright plastic), a
        cast shadow so it reads as airborne, four arms with distinct motor
        housings and motion-blurred rotor discs (with blade-sweep streaks,
        not a flat tinted circle), aviation-convention nav lights (green
        front-left / red front-right / white strobe aft), and a two-tone
        tapered fuselage with a nose camera gimbal + lens glint.

        Every part of that glyph is a fixed decorative colour, so on its
        own a drone/UAV never visually showed whether it was actually
        associated -- unlike the STA/relay circle markers, which are
        filled with _assoc_color(). The id label below the glyph is
        colour-matched to that same green/amber/red read instead."""
        heading = self._drone_heading_px(did)
        r = 10.0
        arm_ang = heading + math.pi / 4.0

        status_color = _assoc_color(node) if node is not None else _MUTED

        sh = 4.0
        self.create_oval(px - r * 0.95 + sh, py - r * 0.68 + sh,
                          px + r * 0.95 + sh, py + r * 0.68 + sh,
                          fill=_SHADOW, outline="", stipple="gray50", tags=("ovl",))

        arm_tips = []
        for k in range(4):
            ang = arm_ang + k * math.pi / 2.0
            ax, ay = px + r * math.cos(ang), py + r * math.sin(ang)
            arm_tips.append((ax, ay, ang))
            self.create_line(px, py, ax, ay, fill=_DRONE_ARM, width=3,
                              capstyle="round", tags=("ovl",))

        for k, (ax, ay, ang) in enumerate(arm_tips):
            mr = 2.6
            self.create_oval(ax - mr, ay - mr, ax + mr, ay + mr,
                              fill=_DRONE_MOTOR, outline=_DRONE_ARM, width=1, tags=("ovl",))
            pr = 5.2
            self.create_oval(ax - pr, ay - pr, ax + pr, ay + pr,
                              fill=_DRONE_ROTOR, outline="", stipple="gray25", tags=("ovl",))
            self.create_oval(ax - pr, ay - pr, ax + pr, ay + pr,
                              outline=_DRONE_ARM, width=1, tags=("ovl",))
            for streak in (ang + 0.55, ang - 0.55):
                sx = ax + pr * 0.85 * math.cos(streak)
                sy = ay + pr * 0.85 * math.sin(streak)
                ox = ax - pr * 0.85 * math.cos(streak)
                oy = ay - pr * 0.85 * math.sin(streak)
                self.create_line(sx, sy, ox, oy, fill=_DRONE_ARM, width=1,
                                  stipple="gray50", tags=("ovl",))
            if k == 0:
                self.create_oval(ax - 1.5, ay - 1.5, ax + 1.5, ay + 1.5,
                                  fill=_DRONE_LED_L, outline="", tags=("ovl",))
            elif k == 3:
                self.create_oval(ax - 1.5, ay - 1.5, ax + 1.5, ay + 1.5,
                                  fill=_DRONE_LED_R, outline="", tags=("ovl",))

        nx, ny = px + r * 0.55 * math.cos(heading), py + r * 0.55 * math.sin(heading)
        tx, ty = px - r * 0.55 * math.cos(heading), py - r * 0.55 * math.sin(heading)
        perp = heading + math.pi / 2.0
        wx, wy = 3.2 * math.cos(perp), 3.2 * math.sin(perp)
        self.create_polygon(nx, ny, px + wx, py + wy, tx, ty, px - wx, py - wy,
                             fill=_DRONE_BODY, outline=_DRONE_ARM, width=1, tags=("ovl",))
        self.create_polygon(px, py, px + wx * 0.6, py + wy * 0.6,
                             nx, ny, px - wx * 0.6, py - wy * 0.6,
                             fill=_DRONE_BODY_LT, outline="", tags=("ovl",))

        self.create_oval(nx - 2.3, ny - 2.3, nx + 2.3, ny + 2.3,
                          fill=_DRONE_ARM, outline=_DRONE_ARM, width=1, tags=("ovl",))
        glint_ang = heading - 0.6
        gx = nx + 0.9 * math.cos(glint_ang)
        gy = ny + 0.9 * math.sin(glint_ang)
        self.create_oval(gx - 0.8, gy - 0.8, gx + 0.8, gy + 0.8,
                          fill=_DRONE_LENS, outline="", tags=("ovl",))

        self.create_oval(tx - 1.3, ty - 1.3, tx + 1.3, ty + 1.3,
                          fill=_DRONE_LED_AFT, outline=_DRONE_ARM, width=1, tags=("ovl",))

        if did == self.selected_id or did in self._selected_ids:
            self.create_oval(px - r - 4, py - r - 4, px + r + 4, py + r + 4,
                              outline=_AMBER, width=2, tags=("ovl",))
        self.create_text(px, py + r + 11, text=f"{prefix}{did}", font=("Arial", 9, "bold"),
                          fill=status_color, tags=("ovl",))
        if node is not None:
            self.create_text(px, py + r + 21, text=f"{altitude_m_for_node(node):.0f}m AGL",
                              font=("Arial", 7), fill=_CYAN, tags=("ovl",))
        self.create_text(px, py + r + 31, text=_fmt_dist_m(dist_to_ap_m(node, self.sim)),
                          font=("Arial", 7), fill=_MUTED, tags=("ovl",))
        if node is not None:
            self.create_text(px, py + r + 41, text=_fmt_rssi(node),
                              font=("Arial", 7), fill=_MUTED, tags=("ovl",))
        diag = _diagnose_unjoined(node, self.sim) if node is not None else None
        if diag is not None:
            code, _label = diag
            self.create_text(px, py + r + 53, text=code, font=("Arial", 8, "bold"),
                              fill=status_color, tags=("ovl",))

    # ── Coordinate transform ─────────────────────────────────────────────
    def _bounds(self):
        xs, ys = [], []
        for n in self.sim.nodes.values():
            x, y = n.pos
            xs.append(x)
            ys.append(y)
        if not xs:
            return (-10.0, 10.0, -10.0, 10.0)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax - xmin < 1.0:
            xmin, xmax = xmin - 10.0, xmax + 10.0
        if ymax - ymin < 1.0:
            ymin, ymax = ymin - 10.0, ymax + 10.0
        padx = (xmax - xmin) * 0.18 + 8.0
        pady = (ymax - ymin) * 0.18 + 8.0
        return (xmin - padx, xmax + padx, ymin - pady, ymax + pady)

    def _transform(self):
        W = self.winfo_width() or int(self.cget("width") or 400)
        H = self.winfo_height() or int(self.cget("height") or 400)
        if self._manual_scale is not None:
            scale = self._manual_scale
            if self._view_center is not None:
                wcx, wcy = self._view_center
            else:
                xmin, xmax, ymin, ymax = self._bounds()
                wcx, wcy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            return scale, W / 2.0, H / 2.0, wcx, wcy

        xmin, xmax, ymin, ymax = self._bounds()
        wx = max(xmax - xmin, 1e-6)
        wy = max(ymax - ymin, 1e-6)
        scale = max(min((W - 40) / wx, (H - 40) / wy), 1e-6)
        return scale, W / 2.0, H / 2.0, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    def _world_to_px(self, x: float, y: float):
        scale, cx, cy, wcx, wcy = self._transform()
        return (cx + (x - wcx) * scale, cy - (y - wcy) * scale)

    def _px_to_world(self, px: float, py: float):
        scale, cx, cy, wcx, wcy = self._transform()
        return ((px - cx) / scale + wcx, -(py - cy) / scale + wcy)

    def _node_at_px(self, px: float, py: float) -> Optional[int]:
        if self.sim is None:
            return None
        best, best_d = None, 1e18
        for nid, n in self.sim.nodes.items():
            nx, ny = self._world_to_px(*n.pos)
            d = math.hypot(nx - px, ny - py)
            r = 16 if nid == 0 else (13 if nid in self._relay_ids else 11)
            if d <= r and d < best_d:
                best, best_d = nid, d
        return best

    # ── Mouse handlers ───────────────────────────────────────────────────
    @staticmethod
    def _is_additive_click(e) -> bool:
        """Shift OR Control/Cmd held -- either toggles a node in/out of the
        multi-select group instead of replacing it. Checking both (not
        just one) is deliberately forgiving of platform differences in
        which modifier reads cleanly through Tk's event.state bitmask."""
        return bool(e.state & 0x0001) or bool(e.state & 0x0004)

    def _on_press(self, e):
        if self.sim is None:
            return
        nid = self._node_at_px(e.x, e.y)
        additive = self._is_additive_click(e)

        if nid is None:
            # Empty space: start a rubber-band select. Don't touch the
            # existing selection yet -- a plain click that never turns
            # into a real drag (see _on_release) should just deselect,
            # but we can't know that until release.
            self._rubber_start_px = (e.x, e.y)
            self._drag_id = None
            self._drag_ids = None
            return

        self.selected_id = nid
        if nid == 0:
            # The AP is a fixed reference point (world origin) -- it stays
            # selectable/clickable so its info still shows, but is never
            # draggable or part of the multi-select group, unlike every
            # other node.
            self._drag_id = None
            self._drag_ids = None
            self._redraw()
            if self._on_select:
                self._on_select(nid)
            return

        if additive:
            # Toggle membership, don't start a drag -- consistent with
            # most desktop apps' "ctrl/shift-click to build a selection"
            # convention (you then drag a SEPARATE press to move the
            # group, same as file managers/design tools).
            if nid in self._selected_ids:
                self._selected_ids.discard(nid)
            else:
                self._selected_ids.add(nid)
            self._drag_id = None
            self._drag_ids = None
            self._redraw()
            if self._on_select:
                self._on_select(nid)
            return

        # Plain click: if this node is already part of a multi-selection,
        # keep the group intact and drag all of it; otherwise collapse the
        # selection down to just this one node (a plain click/drag on an
        # unselected node has always meant "just this one", same as file
        # managers).
        if nid not in self._selected_ids:
            self._selected_ids = {nid}
        self._drag_id = nid
        self._drag_ids = set(self._selected_ids)
        self._drag_anchor_world = self._px_to_world(e.x, e.y)
        self._drag_start_positions = {
            i: self.sim.nodes[i].pos for i in self._drag_ids if i in self.sim.nodes
        }
        self._redraw()
        if self._on_select:
            self._on_select(nid)

    def _on_motion(self, e):
        if self.sim is None:
            return
        if self._drag_ids:
            wx, wy = self._px_to_world(e.x, e.y)
            ax, ay = self._drag_anchor_world
            dx, dy = wx - ax, wy - ay
            for nid in self._drag_ids:
                if nid not in self.sim.nodes:
                    continue
                sx, sy = self._drag_start_positions.get(nid, self.sim.nodes[nid].pos)
                self.sim.nodes[nid].pos = (sx + dx, sy + dy)
            self._redraw()
            if self._on_select and self._drag_id is not None:
                self._on_select(self._drag_id)
            return

        if self._rubber_start_px is not None:
            self._redraw()
            x0, y0 = self._rubber_start_px
            self.create_rectangle(x0, y0, e.x, e.y, outline=_AMBER, width=1,
                                   dash=(4, 3), tags=("ovl",))

    def _on_release(self, e):
        if self._rubber_start_px is not None:
            x0, y0 = self._rubber_start_px
            self._rubber_start_px = None
            # A rubber-band that never really moved is just a click on
            # empty space -- deselect everything, same as clicking empty
            # space in any selection-based UI.
            if abs(e.x - x0) < 4 and abs(e.y - y0) < 4:
                self._selected_ids = set()
                self.selected_id = None
            else:
                lo_x, hi_x = min(x0, e.x), max(x0, e.x)
                lo_y, hi_y = min(y0, e.y), max(y0, e.y)
                found = set()
                for nid, n in self.sim.nodes.items() if self.sim else ():
                    if nid == 0:
                        continue
                    px, py = self._world_to_px(*n.pos)
                    if lo_x <= px <= hi_x and lo_y <= py <= hi_y:
                        found.add(nid)
                self._selected_ids = found
                self.selected_id = min(found) if found else None
            self._redraw()
            if self._on_select:
                self._on_select(self.selected_id)

        self._drag_id = None
        self._drag_ids = None
        self._drag_start_positions = {}
        self._drag_anchor_world = None

    # ── Drawing ───────────────────────────────────────────────────────────
    def _redraw(self):
        self.delete("all")
        W = self.winfo_width() or int(self.cget("width") or 400)
        H = self.winfo_height() or int(self.cget("height") or 400)

        if self.sim is None or not self.sim.nodes:
            self.create_rectangle(0, 0, W, H, fill=_BG, outline="")
            self.create_text(W // 2, H // 2, text="No topology yet",
                              font=("Arial", 10), fill=_MUTED)
            return

        self._draw_background(W, H)

        nodes = self.sim.nodes

        # Edges (drones/UAVs excluded here -- their edges move every tick,
        # so they're drawn as part of the cheap per-tick overlay instead;
        # see _draw_drones_overlay / _draw_uavs_overlay). Drawn from each
        # node's *actual* live association peer (_assoc_peer_id, only ever
        # set once ASSOC_RESP actually succeeds -- see
        # sim11ah/mac/association.py's on_assoc_resp_received), not the
        # topology-build-time relay assignment: that assignment only ever
        # picked a STA's *initial* scatter position, and a STA can end up
        # associating with the AP directly, handing over between peers, or
        # simply not being associated with anyone yet. Drawing an edge
        # regardless of real MAC state made every node look connected even
        # when it wasn't.
        for nid, n in nodes.items():
            if (nid in self._ap_ids or nid in self.drone_ids or nid in self.uav_ids
                    or nid in self.car_ids or nid in self.scooter_ids):
                continue
            peer = _assoc_peer(n)
            if peer is None or peer not in nodes:
                continue  # not associated with anyone yet -- no edge to draw
            px1, py1 = self._world_to_px(*n.pos)
            pxp, pyp = self._world_to_px(*nodes[peer].pos)
            if nid in self._relay_ids:
                self.create_line(pxp, pyp, px1, py1, fill=_PURPLE, width=2, dash=(5, 3))
            elif peer in self._ap_ids:
                # Blue, not green -- green is reserved for the STA-body
                # status LED (ASSOCIATED), so an AP-link edge doubling it
                # up made every fully-connected scene read as solid green.
                self.create_line(pxp, pyp, px1, py1, fill=_BLUE, dash=(2, 3))
            else:
                self.create_line(pxp, pyp, px1, py1, fill=_BORDER, width=1)

        # Nodes (drones/UAVs/cars/scooters excluded -- drawn every tick by
        # their own overlay instead, since their position changes every
        # tick, not fixed).
        for nid, n in nodes.items():
            if (nid in self.drone_ids or nid in self.uav_ids
                    or nid in self.car_ids or nid in self.scooter_ids):
                continue
            px, py = self._world_to_px(*n.pos)
            if nid in self._ap_ids:
                label = "AP" if len(self._ap_ids) == 1 else f"AP{nid}"
                self._draw_ap_icon(px, py, label)
                sel_r, stat_y = 22, None  # AP has no dist/RSSI/diagnostic block below it
            elif nid in self._relay_ids:
                self._draw_relay_icon(px, py, nid, n)
                sel_r, stat_y = 17, 24  # clear of the "R{nid}" label _draw_relay_icon draws
            else:
                # Fixed slate body, not _assoc_color(n) as the whole fill --
                # that made every fully-associated scene (the common case)
                # read as a sea of solid green, the same "AP has no
                # assoc_state of its own" exception _draw_ap_icon's hub
                # already makes for the same reason. Live status now shows
                # as a small LED dot instead (see below), mirroring
                # entities.js's buildStation() convention: fixed body
                # colour + a separate small status light, not a body that
                # itself changes colour.
                r, fill, outline = 8, _STA_BODY, _TEAL
                sh = 2.5
                self.create_oval(px - r + sh, py - r + sh, px + r + sh, py + r + sh,
                                  fill=_SHADOW, outline="", stipple="gray50")
                # Small off-centre highlight -- a plain flat fill reads
                # noticeably flatter next to the AP/relay towers' and
                # drones' shaded glyphs once those got more detail, but a
                # STA population can run into the hundreds (see
                # _get_assoc_ready_t's own docstring on that), so this stays
                # a single extra shape rather than the towers' full
                # shadow+base+arms treatment.
                self.create_oval(px - r, py - r, px + r, py + r,
                                  fill=fill, outline=outline, width=2)
                self.create_oval(px - r * 0.4, py - r * 0.6, px + r * 0.15, py - r * 0.05,
                                  fill="", outline="#ffffff", width=0, stipple="gray25")
                self.create_text(px, py, text=str(nid), font=("Arial", 7, "bold"), fill="white")
                led_r = r * 0.34
                lx, ly = px + r * 0.62, py + r * 0.62
                self.create_oval(lx - led_r, ly - led_r, lx + led_r, ly + led_r,
                                  fill=_assoc_color(n), outline="#0f172a", width=1)
                sel_r, stat_y = r + 4, r + 9

            if nid in self._selected_ids:
                self.create_oval(px - sel_r, py - sel_r, px + sel_r, py + sel_r,
                                  outline=_AMBER, width=2)

            if stat_y is not None:
                self.create_text(px, py + stat_y, text=_fmt_dist_m(dist_to_ap_m(n, self.sim)),
                                  font=("Arial", 7), fill=_MUTED)
                self.create_text(px, py + stat_y + 10, text=_fmt_rssi(n),
                                  font=("Arial", 7), fill=_MUTED)
                diag = _diagnose_unjoined(n, self.sim)
                if diag is not None:
                    code, _label = diag
                    self.create_text(px, py + stat_y + 22, text=code, font=("Arial", 8, "bold"),
                                      fill=_assoc_color(n))

        self._draw_error_code_legend(W, H)
        self.create_rectangle(0, 0, W, 22, fill=_BG, outline="", stipple="gray50")
        self.create_text(8, 11, anchor="w",
                          text="Drag a node to move it (ctrl/shift-click or drag empty "
                               "space to select several) · click to edit its settings",
                          font=("Arial", 8, "italic"), fill=_MUTED)
        if self.environment != "Open Area":
            self.create_text(W - 8, 11, anchor="e", text=self.environment,
                              font=("Arial", 8, "bold"), fill=_MUTED)

        self._draw_packet_legend(W, H)
        self._redraw_overlay()

    def _draw_error_code_legend(self, W: int, H: int) -> None:
        """Explains E417/E118/E240/E102 (see _diagnose_unjoined) -- one panel,
        drawn only when at least one currently-visible node is actually
        showing a code, so a fully-associated topology doesn't carry dead
        chrome on screen."""
        if self.sim is None:
            return
        if not any(_diagnose_unjoined(n, self.sim) is not None
                   for nid, n in self.sim.nodes.items() if nid != 0):
            return

        x0, y0 = 8, 26
        line_h = 14
        box_w, box_h = 170, line_h * len(ERROR_CODE_LEGEND) + 22
        self.create_rectangle(x0, y0, x0 + box_w, y0 + box_h,
                               fill=_BG, outline=_BORDER, width=1, tags=("ovl",))
        self.create_text(x0 + 7, y0 + 7, anchor="nw", text="Not Joined — Why?",
                          font=("Arial", 7, "bold"), fill=_MUTED, tags=("ovl",))
        y = y0 + 20
        for (code, short), _desc in ERROR_CODE_LEGEND:
            color = _AMBER if code == "E102" else _RED
            self.create_text(x0 + 7, y, anchor="nw", text=code,
                              font=("Arial", 8, "bold"), fill=color, tags=("ovl",))
            self.create_text(x0 + 45, y + 1, anchor="nw", text=short,
                              font=("Arial", 7), fill=_MUTED, tags=("ovl",))
            y += line_h

    def _draw_packet_legend(self, W: int, H: int) -> None:
        legend = (("DATA", "DATA"), ("ACK", "ACK"), ("BEACON", "Beacon"),
                  ("RTS", "RTS/CTS"), ("TWT", "TWT"))
        self.create_rectangle(0, H - 18, W, H, fill=_BG, outline="", stipple="gray50")
        x = 8
        for kind, label in legend:
            color = _PKT_COLORS[kind]
            self.create_line(x, H - 9, x + 12, H - 9, fill=color, width=3)
            self.create_text(x + 16, H - 9, anchor="w", text=label,
                              font=("Arial", 7), fill=_MUTED)
            x += 16 + 8 * len(label) + 14

    def _draw_background(self, W: int, H: int) -> None:
        self._scene_obstacles = []
        self._visual_obstacles = []
        variant = self.layout_variant if self.layout_variant in (1, 2, 3) else 1

        if self.environment == "Paddy Field":
            {1: self._bg_paddy_v1, 2: self._bg_paddy_v2, 3: self._bg_paddy_v3}[variant](W, H)
        elif self.environment == "Industrial Site":
            {1: self._bg_industrial_v1, 2: self._bg_industrial_v2,
             3: self._bg_industrial_v3}[variant](W, H)
        elif self.environment == "Smart City":
            # cars_uavs mode gets its own dedicated highway background
            # instead of the generic downtown-grid v1/v2/v3 -- those draw
            # streets at arbitrary box-fraction offsets with no
            # relationship to where real cars/scooters actually drive
            # (see CarsUavsBuilder's car_lane_offset_m), so the real
            # vehicles were rendering beside the drawn road, not on it.
            # Ignores layout_variant entirely, same as the 3D fix
            # (ui/web3d/snapshot.py's _road_loops) this mirrors.
            mode = self.sim.config.get("topology", {}).get("mode") if self.sim else None
            if mode == "cars_uavs":
                self._bg_city_highway(W, H)
            else:
                {1: self._bg_city_v1, 2: self._bg_city_v2, 3: self._bg_city_v3}[variant](W, H)
        elif self.environment == "Military Zone":
            # Single consolidated layout -- no variant switch (see
            # _bg_military_v1's docstring for what it includes).
            self._bg_military_v1(W, H)
        else:
            self.create_rectangle(0, 0, W, H, fill=_BG, outline="")
            self._bg_grid(W, H, "#eef1ec", step_m=20.0)

        self._draw_lab_watermark(W, H)

        if self.sim is not None:
            if self.environment == "Smart City":
                # Real campus buildings (ism_campus_obstacles), NOT the
                # fictional procedural skyline just drawn above -- that
                # skyline is decorative only, for this canvas, and has no
                # relationship to the real IIT (ISM) Dhanbad campus the
                # real-map twin shows. ism_campus_obstacles() is unfiltered
                # (all real buildings) -- both the PHY obstacle list AND the
                # visual list drop any that would enclose the AP (e.g.
                # Library / Academic Core, only ~20-30m from the AP anchor)
                # via the same _encloses_ap rule every other environment
                # follows. Used to only filter self.sim.obstacles, leaving
                # self.sim.visual_obstacles unfiltered -- the PHY model
                # correctly ignored that building, but the 3D view still
                # rendered it sitting right on top of the AP mast, which
                # reads as a visual bug regardless of the (invisible) PHY
                # exclusion behind it.
                filtered_campus = [o for o in ism_campus_obstacles() if not self._encloses_ap(o)]
                self.sim.visual_obstacles = filtered_campus
                self.sim.obstacles = filtered_campus
            else:
                self.sim.obstacles = list(self._scene_obstacles)
                self.sim.visual_obstacles = list(self._visual_obstacles)

    def _draw_lab_watermark(self, W: int, H: int) -> None:
        """Lab branding, drawn identically under every environment/layout
        variant (called once from _draw_background rather than duplicated
        into each _bg_*_v{1,2,3} scene builder)."""
        self.create_text(W - 8, H - 24, anchor="se", text="WiNDS Lab",
                          font=("Arial", 10, "bold"), fill="#8a978a")

    def _encloses_ap(self, obstacle: Obstacle) -> bool:
        """True if this footprint would enclose the AP's own current
        position -- OR sits close enough to it to have essentially the
        same effect: scatter layouts are generally centred on the AP, so a
        scene's "central" structure (e.g. a command bunker placed at the
        middle of the map) can end up literally surrounding the AP
        itself, applying its loss_db to *every* link regardless of
        direction -- one nearby structure becomes a blanket network-wide
        penalty instead of a realistic, direction-dependent obstruction.
        (Found twice: the real ISM campus's Library building and
        Military Zone's command bunker both independently landed on top
        of the AP.)

        The margin below exists because literal containment turned out to
        be too narrow a test: every relay/STA link radiates FROM the AP,
        so an obstacle merely near it (not necessarily containing its
        exact point) still crosses nearly all of them regardless of
        direction -- Military Zone's command bunker is deliberately placed
        "at the command post" and, depending on where the AP's node-cloud-
        relative background layout happens to centre this specific redraw,
        sometimes missed literal containment by a small margin while still
        blocking almost every outgoing link."""
        if self.sim is None:
            return False
        ap = self.sim.nodes.get(0)
        if ap is None:
            return False
        ax, ay = ap.pos
        margin = 0.05 * _ap_range_m(self.sim)
        if obstacle.kind == "rect":
            return (obstacle.x0 - margin) <= ax <= (obstacle.x1 + margin) \
                and (obstacle.y0 - margin) <= ay <= (obstacle.y1 + margin)
        if obstacle.kind == "circle":
            return math.hypot(ax - obstacle.cx, ay - obstacle.cy) <= obstacle.r + margin
        return False

    def _register_obstacle(self, obstacle: Obstacle) -> None:
        """Record a physical obstacle (in world metres) for the current
        scene. Called by the _bg_*_v{1,2,3} scene builders for every solid
        structure they draw (buildings, terrain). Pushed to
        ``self.sim.obstacles`` once the whole background finishes drawing,
        so PhyLayer's path-loss model sees exactly what's on screen.

        Every registered structure also goes into ``self._visual_obstacles``
        unconditionally -- an AP-enclosing structure is still drawn right
        here on this 2D canvas (drawing and PHY-registration are separate
        calls in every _bg_*_v{1,2,3} builder), so the 3D browser view
        should see it too, via self.sim.visual_obstacles (see
        _draw_background). Only the PHY-facing self.sim.obstacles list
        drops it, via _encloses_ap."""
        self._visual_obstacles.append(obstacle)
        if not self._encloses_ap(obstacle):
            self._scene_obstacles.append(obstacle)

    def _wrect(self, cx_w: float, cy_w: float, wxs: float, wys: float,
               fx0: float, fy0: float, fx1: float, fy1: float):
        """Fractional (of half-span, centred) world rect -> sorted
        (x0, y0, x1, y1) in world metres -- the obstacle-registration
        counterpart to a scene's pixel-space frect() helper; pass the same
        fractional coordinates to both so the drawn footprint and the
        physical obstacle line up exactly."""
        wx0, wy0 = cx_w + fx0 * wxs, cy_w + fy0 * wys
        wx1, wy1 = cx_w + fx1 * wxs, cy_w + fy1 * wys
        return (min(wx0, wx1), min(wy0, wy1), max(wx0, wx1), max(wy0, wy1))

    def _bg_grid(self, W: int, H: int, color: str, step_m: float) -> None:
        xmin, xmax, ymin, ymax = self._bounds()
        x = math.floor(xmin / step_m) * step_m
        while x <= xmax:
            px, _ = self._world_to_px(x, 0.0)
            self.create_line(px, 0, px, H, fill=color)
            x += step_m
        y = math.floor(ymin / step_m) * step_m
        while y <= ymax:
            _, py = self._world_to_px(0.0, y)
            self.create_line(0, py, W, py, fill=color)
            y += step_m

    # ── Environment backgrounds ───────────────────────────────────────────
    @staticmethod
    def _hash01(*keys: int) -> float:
        return _hash01(*keys)

    @staticmethod
    def _shade_hex(hexcolor: str, factor: float) -> str:
        """Darken (factor < 1) or lighten (factor > 1) a #rrggbb colour --
        used to derive a building's shaded side-wall tone from its main
        face colour instead of hand-picking a second constant per tier."""
        hexcolor = hexcolor.lstrip("#")
        r, g, b = (int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))
        r, g, b = (max(0, min(255, int(c * factor))) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _quad_grid_lines(self, bl, br, tl, tr, ncols: int, nrows: int, color: str) -> None:
        """Perspective-correct grid of lines inside an arbitrary (possibly
        skewed) quadrilateral face, by interpolating along its edges --
        used for window rows/columns on extruded pseudo-3D building
        walls so the grid follows the wall's own oblique skew instead of
        coming out axis-aligned and looking pasted on."""
        for k in range(1, ncols):
            f = k / ncols
            x0, y0 = bl[0] + f * (br[0] - bl[0]), bl[1] + f * (br[1] - bl[1])
            x1, y1 = tl[0] + f * (tr[0] - tl[0]), tl[1] + f * (tr[1] - tl[1])
            self.create_line(x0, y0, x1, y1, fill=color, width=1)
        for k in range(1, nrows):
            g = k / nrows
            x0, y0 = bl[0] + g * (tl[0] - bl[0]), bl[1] + g * (tl[1] - bl[1])
            x1, y1 = br[0] + g * (tr[0] - br[0]), br[1] + g * (tr[1] - br[1])
            self.create_line(x0, y0, x1, y1, fill=color, width=1)

    def _tree(self, px: float, py: float, r: float) -> None:
        """Small tree seen from above under the NW sun: stippled cast shadow
        offset down-right, soft-edged canopy (no hard outline -- the dark
        foliage tone against the lighter ground is the boundary), sunlit
        NW lobe and shaded SE lobe for volume."""
        self.create_oval(px - r + r * 0.45, py - r * 0.8 + r * 0.45,
                          px + r + r * 0.45, py + r * 0.8 + r * 0.45,
                          fill=_SHADOW, outline="", stipple="gray50")
        self.create_oval(px - r, py - r, px + r, py + r,
                          fill=_TREE_CANOPY, outline="")
        self.create_oval(px + r * 0.05, py + r * 0.05, px + r * 0.85, py + r * 0.85,
                          fill=_TREE_EDGE, outline="")
        self.create_oval(px - r * 0.55, py - r * 0.6, px + r * 0.2, py + r * 0.1,
                          fill=_TREE_LOBE, outline="")

    def _map_chrome(self, W: int, H: int) -> None:
        """Survey-basemap chrome: small semi-transparent north arrow + scale
        bar in the bottom-left corner (just above the packet legend). The
        bar length is derived from the live px-per-metre transform so it
        stays honest across zoom levels; world +y renders screen-up, so
        north is straight up."""
        scale = self._transform()[0]
        metres = 10.0
        for m in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0):
            if m * scale <= 90.0:
                metres = m
        bar_px = max(24.0, metres * scale)
        x0, yb = 10.0, H - 30.0
        # Stippled backing card so the chrome reads over any terrain.
        self.create_rectangle(x0 - 5, yb - 26, x0 + bar_px + 38, yb + 6,
                               fill=_BG, outline="", stipple="gray50")
        # North arrow: filled half + open half (classic survey style).
        nx, ny = x0 + 5, yb - 20
        self.create_polygon(nx, ny, nx - 4, ny + 11, nx, ny + 8,
                             fill=_MUTED, outline="")
        self.create_polygon(nx, ny, nx + 4, ny + 11, nx, ny + 8,
                             fill="", outline=_MUTED)
        self.create_text(nx + 9, ny + 5, anchor="w", text="N",
                          font=("Arial", 7, "bold"), fill=_MUTED)
        # Scale bar: alternating filled/open halves + distance label.
        self.create_rectangle(x0, yb - 3, x0 + bar_px / 2, yb, fill=_MUTED, outline=_MUTED)
        self.create_rectangle(x0 + bar_px / 2, yb - 3, x0 + bar_px, yb,
                               fill="", outline=_MUTED)
        self.create_text(x0 + bar_px + 5, yb - 2, anchor="w",
                          text=f"{metres:g} m", font=("Arial", 7), fill=_GRAY)

    def _bg_paddy_v1(self, W: int, H: int) -> None:
        """Ortho-photo view of a rice-paddy landscape: an irregular mosaic
        of bunded plots at different growth stages (flooded, growing,
        ripening, stubble) with stippled photographic grain, wavy
        irrigation channels with soft wet-mud edges and tree-lined banks,
        a dirt farm track and a small farmhouse with a top-down hip roof."""
        self.create_rectangle(0, 0, W, H, fill=_PADDY_BASE, outline="")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        span = max(wxs, wys)
        # A real bunded paddy plot is ~15-40m -- was span/9.0 (bounding the
        # PLOT COUNT to a fixed 9 regardless of how large the AP-range-scaled
        # bounds are), which put plots at ~220m+ once the recalibrated
        # per-environment path-loss exponents (see topology_canvas.py's
        # _ENV_PATH_LOSS_EXP) made `span` realistically span ~2km. Fixed at a
        # realistic size instead -- the surrounding grid loop already scales
        # PLOT COUNT with `span` on its own (xi..xf/yi..yf below), which is
        # the right thing to scale, not plot size. The span/140 term is only
        # a safety valve against a pathologically large scatter driving the
        # cell count into the tens of thousands (slow Tkinter redraw), not
        # the normal-case driver -- it stays below the 28.0 floor for any
        # span under ~3.9km, comfortably past the realistic scatter range.
        step = max(28.0, span / 140.0)
        scale = self._transform()[0]
        cell_px = step * scale

        # Jittered lattice corners (shared by neighbouring plots -> no gaps,
        # organic non-rectangular field boundaries).
        def corner(gx: int, gy: int):
            jx = (self._hash01(gx, gy, 1) - 0.5) * 0.34 * step
            jy = (self._hash01(gx, gy, 2) - 0.5) * 0.34 * step
            return self._world_to_px(gx * step + jx, gy * step + jy)

        xi, xf = math.floor(xmin / step), math.ceil(xmax / step)
        yi, yf = math.floor(ymin / step), math.ceil(ymax / step)
        water_cells = []
        for gx in range(xi, xf + 1):
            for gy in range(yi, yf + 1):
                c00, c10 = corner(gx, gy), corner(gx + 1, gy)
                c11, c01 = corner(gx + 1, gy + 1), corner(gx, gy + 1)
                cell = (c00[0], c00[1], c10[0], c10[1],
                        c11[0], c11[1], c01[0], c01[1])
                h = self._hash01(gx, gy, 3)
                if h < 0.20:                                   # flooded plot
                    fill = _PADDY_WATER
                    water_cells.append((gx, gy, cell, c00, c11))
                else:                                          # crop stages
                    idx = int(self._hash01(gx, gy, 4) * len(_PADDY_CROPS))
                    fill = _PADDY_CROPS[idx % len(_PADDY_CROPS)]
                # Thin muted bund border: reads as the dirt wall between
                # plots, not a drawn vector outline.
                self.create_polygon(*cell, fill=fill,
                                     outline=_PADDY_BUND, width=1)
                # Crop-row striping on some growing plots (transplant
                # lines), hash-jittered so rows never repeat cell to cell.
                if 0.30 <= h < 0.52 and cell_px > 24:
                    for ki in range(2):
                        k = 0.28 + 0.16 * ki + 0.14 * self._hash01(gx, gy, 8 + ki)
                        rx0 = c00[0] + (c01[0] - c00[0]) * k
                        ry0 = c00[1] + (c01[1] - c00[1]) * k
                        rx1 = c10[0] + (c11[0] - c10[0]) * k
                        ry1 = c10[1] + (c11[1] - c10[1]) * k
                        self.create_line(rx0, ry0, rx1, ry1,
                                          fill=_PADDY_ROWS, width=1)

        # Flooded plots: a touch of sun glint (a clean, restrained accent,
        # not a stippled wash -- Tkinter's stipple is a crude fixed dither,
        # so it's used sparingly here rather than layered over everything).
        for gx, gy, cell, c00, c11 in water_cells:
            mx, my = (c00[0] + c11[0]) / 2.0, (c00[1] + c11[1]) / 2.0
            ox = (self._hash01(gx, gy, 5) - 0.5) * cell_px * 0.5
            oy = (self._hash01(gx, gy, 7) - 0.5) * cell_px * 0.5
            ln = 2.0 + 4.0 * self._hash01(gx, gy, 9)
            self.create_line(mx + ox - ln, my + oy, mx + ox + ln, my + oy,
                              fill=_PADDY_GLINT, width=1)

        # Two wavy irrigation channels: clean two-tone line (bank tone +
        # a thin sheen highlight) rather than a stippled fringe.
        canal_w = max(3, min(10, int(cell_px * 0.14)))
        for ci, band in enumerate((0.30, 0.72)):
            pts = []
            n = 12
            for i in range(n + 1):
                wx = xmin + wxs * i / n
                wy = ymin + wys * band + math.sin(i * 1.7 + ci * 2.3) * step * 0.16
                pts.extend(self._world_to_px(wx, wy))
            self.create_line(*pts, fill=_PADDY_CANAL, width=canal_w, smooth=True)
            self.create_line(*pts, fill=_PADDY_CANAL_LT, width=1, smooth=True)

        # Dirt farm track: packed-earth line with a thin wheel-rut accent.
        pts = []
        for i in range(11):
            wy = ymin + wys * i / 10.0
            wx = xmin + wxs * 0.56 + math.sin(i * 1.3) * step * 0.22
            pts.extend(self._world_to_px(wx, wy))
        track_w = max(2, int(canal_w * 0.7))
        self.create_line(*pts, fill=_PADDY_PATH, width=track_w, smooth=True)
        self.create_line(*pts, fill="#a3967c", width=1, smooth=True)

        # Trees: clusters along both channel banks plus a few field-edge ones.
        tr = max(3.0, min(9.0, cell_px * 0.14))
        for i in range(14):
            if i < 9:      # bank trees hug a channel
                band = 0.30 if i % 2 == 0 else 0.72
                fx = self._hash01(21, i)
                fy = band + (self._hash01(23, i) - 0.5) * 0.07
            else:          # scattered field-corner trees
                fx, fy = self._hash01(31, i), self._hash01(37, i)
            px, py = self._world_to_px(xmin + wxs * fx, ymin + wys * fy)
            self._tree(px, py, tr * (0.75 + 0.5 * self._hash01(41, i)))

        # Small farmhouse near the track, drawn properly top-down: NW-sun
        # cast shadow, hip roof seen from above (sunlit NW slope, shaded
        # main slab, ridge line) instead of a front-elevation gable.
        hx, hy = self._world_to_px(xmin + wxs * 0.60, ymin + wys * 0.50)
        hw = max(5.0, min(15.0, cell_px * 0.24))
        hh = hw * 0.7
        self.create_rectangle(hx - hw + hw * 0.3, hy - hh + hw * 0.3,
                               hx + hw + hw * 0.3, hy + hh + hw * 0.3,
                               fill=_SHADOW, outline="", stipple="gray50")
        self.create_rectangle(hx - hw, hy - hh, hx + hw, hy + hh,
                               fill=_HUT_ROOF, outline="")
        self.create_polygon(hx - hw, hy - hh, hx + hw, hy - hh,
                             hx + hw * 0.55, hy, hx - hw * 0.55, hy,
                             fill=_HUT_ROOF_LT, outline="")
        self.create_line(hx - hw * 0.55, hy, hx + hw * 0.55, hy,
                          fill=_HUT_WALL, width=1)

        self._map_chrome(W, H)

    def _mountain_range(self, W: int, base_frac: float, amp_frac: float,
                        seed: int, n_peaks: int = 10) -> None:
        """Jagged mountain range walling off the top (world-north) edge:
        hazier back range filling to the canvas top, darker front ridge
        with snow-capped tallest peaks, shaded east faces, a sunlit
        ridge-highlight polyline, and a soft cast shadow on the fields at
        the mountain base. base_frac/amp_frac are fractions of the world
        y-span; peaks stay deterministic via _hash01(seed, i)."""
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        scale = self._transform()[0]

        def ridge_pts(bfrac: float, afrac: float, skey: int):
            _, base_py = self._world_to_px(xmin, ymin + wys * bfrac)
            pts, tips = [], []
            for i in range(n_peaks + 1):
                px, _ = self._world_to_px(xmin + wxs * i / n_peaks, 0.0)
                pk = afrac * wys * scale * (0.30 + 0.70 * self._hash01(seed, skey, i))
                pts.extend((px, base_py - pk))
                tips.append((px, base_py - pk, pk))
            return base_py, pts, tips

        # Back range: fills everything above its base so no field shows
        # through between the peak silhouettes; washed with a stippled haze
        # layer (aerial perspective -- far terrain always reads paler).
        bb_py, bpts, _ = ridge_pts(base_frac + 0.055, amp_frac * 0.7, 1)
        back_poly = (-12, -12, -12, bb_py, *bpts, W + 12, bb_py, W + 12, -12)
        self.create_polygon(*back_poly, fill=_MTN_MID, outline="")
        self.create_polygon(*back_poly, fill=_MTN_HAZE, outline="",
                             stipple="gray50")
        # Front ridge silhouette + stippled rock grain so the mass isn't
        # one flat tone.
        fb_py, fpts, tips = ridge_pts(base_frac, amp_frac, 2)
        front_poly = (-12, fb_py, *fpts, W + 12, fb_py)
        self.create_polygon(*front_poly, fill=_MTN_BASE, outline="")
        self.create_polygon(*front_poly, fill=_MTN_SHADE, outline="",
                             stipple="gray12")
        # Two-step cast shadow falling south onto the fields below the base
        # (consistent with the NW sun: terrain shades the ground down-sun).
        sh = max(4.0, min(26.0, amp_frac * wys * scale * 0.22))
        self.create_rectangle(-12, fb_py, W + 12, fb_py + sh,
                               fill=_SHADOW, outline="", stipple="gray25")
        self.create_rectangle(-12, fb_py + sh, W + 12, fb_py + sh * 1.8,
                               fill=_SHADOW, outline="", stipple="gray12")
        # Sunlit ridge line along the front peaks (thin -- a lit edge, not
        # a drawn outline).
        self.create_line(*fpts, fill=_MTN_RIDGE, width=1)
        # Shaded SE faces (down-sun of each peak) + snow caps on the
        # tallest peaks.
        order = sorted(range(1, n_peaks), key=lambda i: -tips[i][2])
        for i in order[:4]:
            px, py, pk = tips[i]
            fx = (tips[i + 1][0] - px) * 0.5
            self.create_polygon(px, py, px + fx, fb_py, px + fx * 0.2, fb_py,
                                 fill=_MTN_SHADE, outline="", stipple="gray50")
        for i in order[:3]:
            px, py, pk = tips[i]
            cw, cd = pk * 0.30, pk * 0.34
            self.create_polygon(px, py, px - cw, py + cd, px + cw * 0.8, py + cd * 0.9,
                                 fill=_MTN_SNOW, outline="")

    def _river(self, f_band: float, amp: float, width_px: int, seed: int):
        """Meandering river across the scene at world-y fraction f_band:
        grassy floodplain band, blue channel, sheen line. Returns the
        world-space centreline samples [(wx, wy), ...] so callers can site
        bridges where tracks cross. NOT registered as an obstacle -- water
        adds negligible sub-GHz loss at ground level."""
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        n = 16
        wpts, pts = [], []
        for i in range(n + 1):
            wx = xmin + wxs * i / n
            f = f_band + amp * math.sin(i * 0.85 + seed * 1.7) \
                + amp * 0.4 * math.sin(i * 2.1 + seed)
            wy = ymin + wys * f
            wpts.append((wx, wy))
            pts.extend(self._world_to_px(wx, wy))
        # Floodplain grass, clean channel, thin sheen highlight.
        self.create_line(*pts, fill=_RIVER_BANK, width=int(width_px * 2.4),
                          smooth=True)
        self.create_line(*pts, fill=_RIVER, width=width_px, smooth=True)
        self.create_line(*pts, fill=_RIVER_LT, width=max(1, width_px // 5),
                          smooth=True)
        return wpts

    def _bg_paddy_v2(self, W: int, H: int) -> None:
        """Layout variant 2 -- "River Valley": bunded paddy mosaic on a
        valley floor, a wide meandering river with floodplain and a small
        wooden bridge where the farm track crosses it, and a jagged
        mountain range walling off the north edge. Each mountain mass is
        registered as a circular Obstacle at 30 dB (terrain fully blocks
        sub-GHz line of sight); the river/paddies are not (open water and
        crops add negligible loss)."""
        self.create_rectangle(0, 0, W, H, fill=_PADDY_BASE, outline="")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        span = max(wxs, wys)
        step = max(28.0, span / 140.0)  # see _bg_paddy_v1's comment for why
        scale = self._transform()[0]
        cell_px = step * scale

        # Paddy mosaic (same jittered lattice as v1, reseeded -> new farm).
        def corner(gx: int, gy: int):
            jx = (self._hash01(gx, gy, 101) - 0.5) * 0.34 * step
            jy = (self._hash01(gx, gy, 102) - 0.5) * 0.34 * step
            return self._world_to_px(gx * step + jx, gy * step + jy)

        xi, xf = math.floor(xmin / step), math.ceil(xmax / step)
        yi, yf = math.floor(ymin / step), math.ceil(ymax / step)
        for gx in range(xi, xf + 1):
            for gy in range(yi, yf + 1):
                c00, c10 = corner(gx, gy), corner(gx + 1, gy)
                c11, c01 = corner(gx + 1, gy + 1), corner(gx, gy + 1)
                cell = (c00[0], c00[1], c10[0], c10[1],
                        c11[0], c11[1], c01[0], c01[1])
                h = self._hash01(gx, gy, 103)
                if h < 0.14:
                    fill = _PADDY_WATER
                else:
                    idx = int(self._hash01(gx, gy, 104) * len(_PADDY_CROPS))
                    fill = _PADDY_CROPS[idx % len(_PADDY_CROPS)]
                self.create_polygon(*cell, fill=fill,
                                     outline=_PADDY_BUND, width=1)

        # River across the lower valley, then the farm track crossing it.
        river_w = max(6, min(24, int(cell_px * 0.30)))
        wpts = self._river(0.32, 0.05, river_w, seed=3)

        track_fx, track_amp = 0.60, step * 0.20
        pts = []
        for i in range(11):
            wy = ymin + wys * i / 10.0
            wx = xmin + wxs * track_fx + math.sin(i * 1.3) * track_amp
            pts.extend(self._world_to_px(wx, wy))
        track_w = max(3, int(river_w * 0.55))
        self.create_line(*pts, fill=_PADDY_PATH, width=track_w, smooth=True)

        # Wooden bridge where the track meets the river: NW-sun cast shadow
        # on the water, deck, plank seams, side rails.
        wx_b = xmin + wxs * track_fx + math.sin(0.32 * 10 * 1.3) * track_amp
        wy_b = min(wpts, key=lambda p: abs(p[0] - wx_b))[1]
        bx, by = self._world_to_px(wx_b, wy_b)
        dw = max(3.0, river_w * 0.42)
        dh = river_w * 1.15
        so = max(2.0, dw * 0.5)
        self.create_rectangle(bx - dw + so, by - dh + so, bx + dw + so, by + dh + so,
                               fill=_SHADOW, outline="", stipple="gray50")
        self.create_rectangle(bx - dw, by - dh, bx + dw, by + dh,
                               fill=_BRIDGE_DECK, outline="")
        for k in (-0.5, 0.0, 0.5):
            self.create_line(bx - dw, by + dh * k, bx + dw, by + dh * k,
                              fill=_BRIDGE_RAIL, width=1)
        self.create_line(bx - dw, by - dh, bx - dw, by + dh,
                          fill=_BRIDGE_RAIL, width=2)
        self.create_line(bx + dw, by - dh, bx + dw, by + dh,
                          fill=_BRIDGE_RAIL, width=2)

        # Mountain range along the north (top) edge + terrain obstacles.
        self._mountain_range(W, base_frac=0.84, amp_frac=0.14, seed=71)
        for k, fx in enumerate((0.16, 0.50, 0.84)):
            self._register_obstacle(Obstacle(
                kind="circle", cx=xmin + wxs * fx, cy=ymin + wys * 1.02,
                r=0.20 * wys, loss_db=30.0, label=f"valley-ridge-{k}"))

        # Trees along the riverbanks plus a few field-corner ones.
        tr = max(3.0, min(9.0, cell_px * 0.14))
        for i in range(12):
            if i < 8:
                wx, wy = wpts[(2 * i + 1) % len(wpts)]
                wy += (1 if i % 2 == 0 else -1) * (river_w * 1.9 / scale)
            else:
                wx = xmin + wxs * self._hash01(111, i)
                wy = ymin + wys * (0.05 + 0.70 * self._hash01(113, i))
            px, py = self._world_to_px(wx, wy)
            self._tree(px, py, tr * (0.75 + 0.5 * self._hash01(117, i)))

        self._map_chrome(W, H)

    def _bg_paddy_v3(self, W: int, H: int) -> None:
        """Layout variant 3 -- "Highland Terraces": paddies climbing a
        hillside as stepped curved bands (elevation shading dark valley
        floor -> pale hilltop, hashed plot-divider walls), a dominant
        three-peak mountain crown, and a river along the valley floor at
        the terrace base. Terraces stay obstacle-free (still open fields);
        only the mountain terrain is registered, as three large circular
        Obstacles at 32 dB."""
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        scale = self._transform()[0]
        self.create_rectangle(0, 0, W, H, fill=_RIVER_BANK, outline="")

        # Terrace bands, drawn valley-up: each band's polygon extends from
        # its lower lip to the canvas top, so the next (higher) band
        # overpaints it -- one polygon + one bund line per terrace.
        nb = len(_TERRACE_SHADES) + 3
        f0, f1 = 0.16, 0.82
        bh = (f1 - f0) / nb
        curves = []
        for b in range(nb + 1):
            fb = f0 + b * bh
            amp = 0.012 + 0.020 * b / nb
            pts = []
            for j in range(13):
                wx = xmin + wxs * j / 12.0
                f = fb + amp * math.sin(j * 0.8 + b * 0.55) \
                    + 0.006 * (self._hash01(131, b, j) - 0.5)
                pts.extend(self._world_to_px(wx, ymin + wys * f))
            curves.append(pts)
        band_px = bh * wys * scale
        for b in range(nb):
            shade = _TERRACE_SHADES[min(b * len(_TERRACE_SHADES) // nb,
                                        len(_TERRACE_SHADES) - 1)]
            band_poly = (-12, curves[b][1], *curves[b],
                         W + 12, curves[b][-1], W + 12, -12, -12, -12)
            self.create_polygon(*band_poly, fill=shade, outline="")
            # Terrace wall: the NW sun throws the retaining wall's shadow
            # onto the band below (screen-down), so each lip is a soft
            # stippled shadow curve plus a thin earth-tone wall line --
            # the step reads as relief, not as a drawn contour line.
            shadow_pts = [v + (2 if i % 2 else 0)
                          for i, v in enumerate(curves[b])]
            self.create_line(*shadow_pts, fill=_SHADOW, width=3,
                              smooth=True, stipple="gray25")
            self.create_line(*curves[b], fill=_TERRACE_BUND, width=1,
                              smooth=True)
        # Plot-divider walls: short cross-slope ticks at hashed positions,
        # denser low (small valley plots) than high (big upland ones).
        for b in range(nb):
            for k in range(max(2, 5 - 3 * b // nb)):
                j2 = 2 + int(self._hash01(137, b, k) * 9) * 2
                x0, y0 = curves[b][j2], curves[b][j2 + 1]
                self.create_line(x0, y0, x0 + band_px * 0.15, y0 - band_px,
                                  fill=_TERRACE_BUND, width=1, stipple="gray50")

        # River along the valley floor, below the lowest terrace.
        river_w = max(6, min(22, int(0.045 * wys * scale)))
        wpts = self._river(0.08, 0.03, river_w, seed=5)

        # Dominant mountain crown above the terraces + terrain obstacles.
        self._mountain_range(W, base_frac=0.83, amp_frac=0.19, seed=73,
                             n_peaks=7)
        for k, fx in enumerate((0.22, 0.52, 0.80)):
            self._register_obstacle(Obstacle(
                kind="circle", cx=xmin + wxs * fx, cy=ymin + wys * 1.03,
                r=0.23 * wys, loss_db=32.0, label=f"highland-peak-{k}"))

        # Trees: riverside cluster plus a few dotting the terrace lips.
        tr = max(3.0, min(8.0, 0.016 * wys * scale))
        for i in range(10):
            if i < 5:
                wx, wy = wpts[(3 * i + 2) % len(wpts)]
                wy += river_w * 2.0 / scale
                px, py = self._world_to_px(wx, wy)
            else:
                b = 1 + int(self._hash01(139, i) * (nb - 2))
                j2 = 2 + int(self._hash01(141, i) * 10) * 2
                px, py = curves[b][j2], curves[b][j2 + 1] - band_px * 0.4
            self._tree(px, py, tr * (0.75 + 0.5 * self._hash01(143, i)))

        self._map_chrome(W, H)

    def _ind_building(self, bx0: float, by0: float, bx1: float, by1: float,
                       fill: str, seed: int) -> None:
        """Extruded pseudo-3D warehouse/plant box, seen from the NW under a
        fixed sun: real front + side walls between the ground footprint
        and a roof slab shifted up-and-right, the same oblique-projection
        trick as _city_tower but a lower height multiplier -- real
        single/double-storey industrial sheds are wide and squat, not
        skyscrapers, so they should visibly stay much shorter than the
        Smart City towers. The roof keeps its weathered-panel stipple
        grain, sunlit seam and hashed HVAC units / vents, just translated
        onto the raised slab instead of the flat footprint."""
        w, h = bx1 - bx0, by1 - by0
        if w < 8 or h < 8:
            self.create_rectangle(bx0, by0, bx1, by1, fill=fill,
                                   outline=_IND_ROOF_EDGE, width=1)
            return
        eh = min(w, h) * 0.42 * (0.85 + 0.3 * self._hash01(seed, 9))
        skew = eh * 0.32
        wall = self._shade_hex(fill, 0.72)

        fbl, fbr = (bx0, by1), (bx1, by1)
        bbr = (bx1, by0)
        rx0, ry0, rx1, ry1 = bx0 + skew, by0 - eh, bx1 + skew, by1 - eh
        ftl, ftr = (rx0, ry1), (rx1, ry1)
        btr = (rx1, ry0)

        off = max(3.0, min(10.0, 0.09 * max(w, h)))
        self.create_rectangle(bx0 + off, by0 + off, bx1 + off, by1 + off,
                               fill=_SHADOW, outline="", stipple="gray50")

        # Side (east) and front (south) walls give the slab real thickness.
        self.create_polygon(fbr[0], fbr[1], bbr[0], bbr[1], btr[0], btr[1], ftr[0], ftr[1],
                             fill=wall, outline=_IND_ROOF_EDGE, width=1)
        self.create_polygon(fbl[0], fbl[1], fbr[0], fbr[1], ftr[0], ftr[1], ftl[0], ftl[1],
                             fill=wall, outline=_IND_ROOF_EDGE, width=1)

        self.create_rectangle(rx0, ry0, rx1, ry1, fill=fill, outline=_IND_ROOF_EDGE, width=1)
        self.create_rectangle(rx0, ry0, rx1, ry1, fill=_IND_ROOF_GRAIN,
                               outline="", stipple="gray12")
        # Shaded SE parapet (down-sun edges) -- gives the roof-slab a rim.
        self.create_line(rx0 + 1, ry1 - 1, rx1 - 1, ry1 - 1, rx1 - 1, ry0 + 1,
                          fill=_IND_ROOF_GRAIN, width=2)
        # Sunlit panel seams along the long axis (thin, like real roof ribs).
        if w >= h:
            for k in (0.38, 0.62):
                self.create_line(rx0 + 4, ry0 + h * k, rx1 - 4, ry0 + h * k,
                                  fill=_IND_ROOF_LT, width=1)
        else:
            for k in (0.38, 0.62):
                self.create_line(rx0 + w * k, ry0 + 4, rx0 + w * k, ry1 - 4,
                                  fill=_IND_ROOF_LT, width=1)
        # HVAC units (small light squares) + round vents at hashed spots.
        u = max(3.0, min(8.0, 0.09 * min(w, h)))
        for k in range(3):
            ux = rx0 + w * (0.15 + 0.7 * self._hash01(seed, k, 1))
            uy = ry0 + h * (0.15 + 0.7 * self._hash01(seed, k, 2))
            self.create_rectangle(ux - u, uy - u, ux + u, uy + u,
                                   fill=_IND_HVAC, outline=_IND_ROOF_GRAIN, width=1)
        for k in range(2):
            vx = rx0 + w * (0.12 + 0.76 * self._hash01(seed, k, 3))
            vy = ry0 + h * (0.12 + 0.76 * self._hash01(seed, k, 4))
            self.create_oval(vx - u * 0.5, vy - u * 0.5, vx + u * 0.5, vy + u * 0.5,
                              fill="#8f99a4", outline="", width=1)

    def _ind_tank(self, cx: float, cy: float, r: float) -> None:
        """Cylindrical storage tank from above, NW sun: cast shadow
        down-right, shell with weathering grain, sunlit NW arc + shaded SE
        arc for curvature, centre hatch."""
        self.create_oval(cx - r + r * 0.45, cy - r + r * 0.45,
                          cx + r + r * 0.45, cy + r + r * 0.45,
                          fill=_SHADOW, outline="", stipple="gray50")
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                          fill=_IND_TANK, outline=_IND_TANK_RIM, width=1)
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                          fill=_IND_ROOF_GRAIN, outline="", stipple="gray12")
        self.create_arc(cx - r * 0.62, cy - r * 0.62, cx + r * 0.62, cy + r * 0.62,
                         start=70, extent=130, style="arc",
                         outline="#eceff1", width=2)
        self.create_arc(cx - r * 0.62, cy - r * 0.62, cx + r * 0.62, cy + r * 0.62,
                         start=250, extent=120, style="arc",
                         outline="#a4abb4", width=2)
        self.create_oval(cx - r * 0.16, cy - r * 0.16, cx + r * 0.16, cy + r * 0.16,
                          fill=_IND_TANK_RIM, outline="")

    def _bg_industrial_v1(self, W: int, H: int) -> None:
        """Stylised satellite view of a fenced industrial estate: access-road
        cross with lane markings, warehouses and a process plant with cast
        shadows and rooftop plant, storage tanks, a striped loading dock, a
        parking lot with cars, perimeter floodlights and a smoking chimney.
        The four buildings are also registered as physical obstacles (see
        _register_obstacle) -- a link whose direct path crosses one takes
        real extra path loss in PhyLayer, not just a cosmetic overlap."""
        self.create_rectangle(0, 0, W, H, fill=_IND_BASE, outline="")
        # Two-layer gravel grain: dark speckle + sparse light speckle, so
        # the yard reads as worn concrete/gravel rather than a flat panel.
        self.create_rectangle(0, 0, W, H, fill=_IND_GRAIN_DK, outline="", stipple="gray12")
        self.create_rectangle(0, 0, W, H, fill=_IND_GRAIN_LT, outline="", stipple="gray12")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        scale = self._transform()[0]

        def frect(fx0, fy0, fx1, fy1):
            """Fractional (of half-span, centred) world rect -> sorted px box."""
            p0 = self._world_to_px(cx_w + fx0 * wxs, cy_w + fy0 * wys)
            p1 = self._world_to_px(cx_w + fx1 * wxs, cy_w + fy1 * wys)
            return (min(p0[0], p1[0]), min(p0[1], p1[1]),
                    max(p0[0], p1[0]), max(p0[1], p1[1]))

        def wrect(fx0, fy0, fx1, fy1):
            return self._wrect(cx_w, cy_w, wxs, wys, fx0, fy0, fx1, fy1)

        # Real-world-sized footprint (m) centred at a fractional position --
        # see Smart City's _bg_city_v1/frect_sized for why: the buildings
        # below used to span their whole hand-authored fractional box
        # directly, which put a "warehouse" at 600m+ once the per-
        # environment path-loss recalibration made `wxs`/`wys` realistically
        # span well over a kilometre. The (cx0,cy0)-(cx1,cy1) box still
        # controls WHERE the building sits (unchanged), just not how big it
        # is -- and every offset computed off the returned rect (loading
        # dock, chimney) automatically follows the smaller building since
        # they read the rect's own returned coordinates, not independent
        # fractions.
        def frect_sized(fcx, fcy, w_m, h_m):
            return frect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        def wrect_sized(fcx, fcy, w_m, h_m):
            return wrect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        # Access roads: soft stippled edge, asphalt slab, wear-band grain
        # down the middle, dashed centre line.
        road_px = max(8, min(22, int(6.0 * scale)))
        rh0, rh1 = self._world_to_px(xmin, cy_w), self._world_to_px(xmax, cy_w)
        rv0, rv1 = self._world_to_px(cx_w, ymin), self._world_to_px(cx_w, ymax)
        for p0, p1 in ((rh0, rh1), (rv0, rv1)):
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD_EDGE,
                              width=road_px + 3, stipple="gray25")
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD, width=road_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD_GRAIN,
                              width=road_px - 3, stipple="gray25")
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD_MARK,
                              width=1, dash=(7, 6))

        # Perimeter fence + floodlight poles (dot mast + muted sodium glow).
        f0 = self._world_to_px(xmin + wxs * 0.03, ymin + wys * 0.03)
        f1 = self._world_to_px(xmax - wxs * 0.03, ymax - wys * 0.03)
        fx0, fy0 = min(f0[0], f1[0]), min(f0[1], f1[1])
        fx1, fy1 = max(f0[0], f1[0]), max(f0[1], f1[1])
        self.create_rectangle(fx0, fy0, fx1, fy1,
                               outline=_GRAY, dash=(6, 3), width=1)
        poles = ((fx0, fy0), ((fx0 + fx1) / 2, fy0), (fx1, fy0),
                 (fx0, (fy0 + fy1) / 2), (fx1, (fy0 + fy1) / 2),
                 (fx0, fy1), ((fx0 + fx1) / 2, fy1), (fx1, fy1))
        for px, py in poles:
            self.create_oval(px - 5, py - 5, px + 5, py + 5,
                              fill=_IND_LAMP, outline="", stipple="gray25")
            self.create_oval(px - 2, py - 2, px + 2, py + 2,
                              fill="#556170", outline="")

        # Buildings (shadow + roof detail); plant building hosts the chimney.
        # Fractional footprints are shared between the pixel (frect) and
        # world (wrect) helpers so the drawn shape and the physical
        # obstacle line up exactly.
        # (fractional centre) -- same positions the old hand-authored boxes
        # were centred on; real-world footprint sizes (m) chosen per
        # building type instead of "however big that fraction of the map
        # happens to be" (see frect_sized's own comment).
        wh1_c = (-0.27, -0.28)   # warehouse (dock)
        plant_c = (0.24, -0.31)  # process plant (chimney)
        wh2_c = (-0.28, 0.26)    # second warehouse
        office_c = (0.33, 0.27)  # office block
        wh1 = frect_sized(*wh1_c, 100.0, 55.0)
        plant = frect_sized(*plant_c, 70.0, 46.0)
        wh2 = frect_sized(*wh2_c, 90.0, 50.0)
        office = frect_sized(*office_c, 40.0, 30.0)
        self._ind_building(*wh1, fill=_IND_WAREHOUSE, seed=11)
        self._ind_building(*plant, fill=_IND_PLANT, seed=13)
        self._ind_building(*wh2, fill=_IND_WAREHOUSE, seed=17)
        self._ind_building(*office, fill=_IND_OFFICE, seed=19)

        # Corrugated-metal warehouse walls: ~15 dB; reinforced-concrete
        # process plant: ~22 dB; lighter office-block construction: ~10 dB
        # (rough real-world sub-GHz building-penetration figures).
        for c, wh, loss, label in ((wh1_c, (100.0, 55.0), 15.0, "warehouse-1"),
                                    (plant_c, (70.0, 46.0), 22.0, "process-plant"),
                                    (wh2_c, (90.0, 50.0), 15.0, "warehouse-2"),
                                    (office_c, (40.0, 30.0), 10.0, "office")):
            wx0, wy0, wx1, wy1 = wrect_sized(*c, *wh)
            self._register_obstacle(Obstacle(kind="rect", x0=wx0, y0=wy0, x1=wx1, y1=wy1,
                                              loss_db=loss, label=label))

        # Loading dock on warehouse 1: concrete apron (with tyre-scrub
        # grain) + faded hazard stripes.
        dx0, dy0, dx1, dy1 = wh1
        apron_h = max(4.0, min(14.0, (dy1 - dy0) * 0.35))
        self.create_rectangle(dx0, dy1, dx1, dy1 + apron_h,
                               fill=_IND_APRON, outline="")
        self.create_rectangle(dx0, dy1, dx1, dy1 + apron_h,
                               fill=_IND_ROAD_GRAIN, outline="", stipple="gray25")
        nstripes = max(3, min(9, int((dx1 - dx0) / 14)))
        for i in range(nstripes):
            sx = dx0 + (dx1 - dx0) * (i + 0.5) / nstripes
            self.create_line(sx, dy1, sx, dy1 + apron_h, fill=_IND_DOCK, width=3)

        # Storage-tank farm between the plant and the office.
        tr = max(4.0, min(13.0, 3.2 * scale))
        tx, ty = self._world_to_px(cx_w + 0.10 * wxs, cy_w + 0.28 * wys)
        for i in range(3):
            self._ind_tank(tx + i * tr * 2.5, ty, tr * (1.0 - 0.12 * (i % 2)))

        # Parking lot east of the vertical road, north of the horizontal
        # one: darker asphalt with wear grain; parked cars jittered inside
        # their stalls (nobody parks dead-centre) with hashed colours.
        px0, py0, px1, py1 = frect(0.06, -0.12, 0.42, -0.03)
        self.create_rectangle(px0, py0, px1, py1,
                               fill=_IND_LOT, outline="")
        self.create_rectangle(px0, py0, px1, py1,
                               fill=_IND_ROAD_GRAIN, outline="", stipple="gray25")
        nstall = max(4, min(9, int((px1 - px0) / 16)))
        for i in range(1, nstall):
            sx = px0 + (px1 - px0) * i / nstall
            self.create_line(sx, py0 + 2, sx, py1 - 2, fill=_IND_ROAD_MARK, width=1)
        for i in range(nstall):
            if self._hash01(53, i) < 0.35:           # some stalls stay empty
                continue
            sx = px0 + (px1 - px0) * (i + 0.5) / nstall
            cw = (px1 - px0) / nstall * 0.32
            ch = (py1 - py0) * 0.36
            jx = (self._hash01(61, i) - 0.5) * cw * 0.6
            jy = (self._hash01(63, i) - 0.5) * ch * 0.35
            col = _CAR_COLORS[int(self._hash01(59, i) * len(_CAR_COLORS)) % len(_CAR_COLORS)]
            self.create_rectangle(sx + jx - cw, py0 + (py1 - py0) * 0.5 + jy - ch,
                                   sx + jx + cw, py0 + (py1 - py0) * 0.5 + jy + ch,
                                   fill=col, outline="#4a545e", width=1)

        # Chimney on the plant's corner, with a drifting translucent plume.
        chx, chy = plant[2] - (plant[2] - plant[0]) * 0.12, plant[1] + (plant[3] - plant[1]) * 0.22
        cr = max(3.0, min(7.0, 1.8 * scale))
        self.create_oval(chx - cr, chy - cr, chx + cr, chy + cr,
                          fill="#6b7280", outline="#3f4954", width=1)
        self.create_oval(chx - cr * 0.4, chy - cr * 0.4, chx + cr * 0.4, chy + cr * 0.4,
                          fill="#2f3944", outline="")
        for i, (drift, grow, stip) in enumerate(
                ((1.6, 1.5, "gray50"), (3.4, 2.3, "gray25"), (5.6, 3.2, "gray12"))):
            sx, sy = chx + cr * drift, chy - cr * drift * 0.8
            sr = cr * grow
            self.create_oval(sx - sr, sy - sr * 0.8, sx + sr, sy + sr * 0.8,
                              fill=_IND_SMOKE, outline="", stipple=stip)

        self._map_chrome(W, H)

    def _bg_industrial_v2(self, W: int, H: int) -> None:
        """Layout variant 2 -- "Process Plant": heavy industry. Two large
        process units and a small control room, an eight-tank storage farm,
        pipe-racks (parallel runs with support-post ticks) linking the farm
        to both units, three smoking chimneys, and only a small parking
        strip. Registered obstacles: process units at 25/22 dB (reinforced
        concrete + dense steel), the whole tank-farm footprint bundled as
        one 20 dB rect, control room 10 dB."""
        self.create_rectangle(0, 0, W, H, fill=_IND_BASE, outline="")
        # Two-layer gravel grain (see v1) for a worn-yard surface.
        self.create_rectangle(0, 0, W, H, fill=_IND_GRAIN_DK, outline="", stipple="gray12")
        self.create_rectangle(0, 0, W, H, fill=_IND_GRAIN_LT, outline="", stipple="gray12")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        scale = self._transform()[0]

        def frect(fx0, fy0, fx1, fy1):
            p0 = self._world_to_px(cx_w + fx0 * wxs, cy_w + fy0 * wys)
            p1 = self._world_to_px(cx_w + fx1 * wxs, cy_w + fy1 * wys)
            return (min(p0[0], p1[0]), min(p0[1], p1[1]),
                    max(p0[0], p1[0]), max(p0[1], p1[1]))

        def wrect(fx0, fy0, fx1, fy1):
            return self._wrect(cx_w, cy_w, wxs, wys, fx0, fy0, fx1, fy1)

        # See _bg_industrial_v1's frect_sized for why: fixed real-world
        # footprint (m) centred at a fractional position, instead of the
        # building spanning its whole hand-authored fractional box.
        def frect_sized(fcx, fcy, w_m, h_m):
            return frect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        def wrect_sized(fcx, fcy, w_m, h_m):
            return wrect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        # Single east-west haul road plus a short plant spur, with soft
        # stippled edges and a wear-band down the middle.
        road_px = max(8, min(22, int(6.0 * scale)))
        rh0, rh1 = self._world_to_px(xmin, cy_w), self._world_to_px(xmax, cy_w)
        self.create_line(rh0[0], rh0[1], rh1[0], rh1[1], fill=_IND_ROAD_EDGE,
                          width=road_px + 3, stipple="gray25")
        self.create_line(rh0[0], rh0[1], rh1[0], rh1[1], fill=_IND_ROAD, width=road_px)
        self.create_line(rh0[0], rh0[1], rh1[0], rh1[1], fill=_IND_ROAD_GRAIN,
                          width=road_px - 3, stipple="gray25")
        self.create_line(rh0[0], rh0[1], rh1[0], rh1[1], fill=_IND_ROAD_MARK,
                          width=1, dash=(7, 6))
        s0 = self._world_to_px(cx_w - 0.02 * wxs, cy_w)
        s1 = self._world_to_px(cx_w - 0.02 * wxs, ymax)
        self.create_line(s0[0], s0[1], s1[0], s1[1], fill=_IND_ROAD, width=road_px)
        self.create_line(s0[0], s0[1], s1[0], s1[1], fill=_IND_ROAD_GRAIN,
                          width=road_px - 3, stipple="gray25")

        # Perimeter fence (hazard sites are always fenced).
        f0 = self._world_to_px(xmin + wxs * 0.03, ymin + wys * 0.03)
        f1 = self._world_to_px(xmax - wxs * 0.03, ymax - wys * 0.03)
        self.create_rectangle(min(f0[0], f1[0]), min(f0[1], f1[1]),
                               max(f0[0], f1[0]), max(f0[1], f1[1]),
                               outline=_GRAY, dash=(6, 3), width=1)

        # Buildings -- fractional CENTRE (unchanged from before) + a
        # realistic real-world footprint size (m). The tank farm bundles 8
        # individual tanks, so it legitimately stays a larger multi-vessel
        # footprint than a single building, just nowhere near its old
        # 800x680m.
        unitA_c = (-0.26, 0.26)    # main reactor/process hall
        unitB_c = (-0.30, -0.28)   # secondary process unit
        ctrl_c = (0.37, 0.36)      # control room
        tankzone_c = (0.24, -0.25) # tank-farm footprint
        unitA = frect_sized(*unitA_c, 75.0, 48.0)
        unitB = frect_sized(*unitB_c, 65.0, 40.0)
        ctrl = frect_sized(*ctrl_c, 30.0, 22.0)
        self._ind_building(*unitA, fill=_IND_PLANT, seed=61)
        self._ind_building(*unitB, fill=_IND_PLANT, seed=67)
        self._ind_building(*ctrl, fill=_IND_OFFICE, seed=71)

        # Reinforced-concrete/steel process structures ~22-25 dB; the tank
        # farm is bundled as one rect (dense steel cylinders + pipework)
        # ~20 dB; light control room ~10 dB.
        for c, wh, loss, label in ((unitA_c, (75.0, 48.0), 25.0, "process-unit-A"),
                                    (unitB_c, (65.0, 40.0), 22.0, "process-unit-B"),
                                    (tankzone_c, (100.0, 55.0), 20.0, "tank-farm"),
                                    (ctrl_c, (30.0, 22.0), 10.0, "control-room")):
            wx0, wy0, wx1, wy1 = wrect_sized(*c, *wh)
            self._register_obstacle(Obstacle(kind="rect", x0=wx0, y0=wy0,
                                              x1=wx1, y1=wy1,
                                              loss_db=loss, label=label))

        # Tank farm: gravel pad (soft edge, stippled grain) + two rows of
        # four tanks.
        tz = frect_sized(*tankzone_c, 100.0, 55.0)
        self.create_rectangle(*tz, fill=_IND_APRON, outline="")
        self.create_rectangle(*tz, fill=_IND_ROAD_GRAIN, outline="", stipple="gray25")
        tr = min((tz[2] - tz[0]) / 9.5, (tz[3] - tz[1]) / 5.0)
        for row in range(2):
            for col in range(4):
                tx = tz[0] + (tz[2] - tz[0]) * (col + 0.5) / 4.0
                ty = tz[1] + (tz[3] - tz[1]) * (row + 0.5) / 2.0
                self._ind_tank(tx, ty, tr * (1.0 - 0.10 * ((row + col) % 2)))

        # Pipe-racks: twin runs + highlight, periodic support-post ticks.
        def pipe_rack(p0, p1):
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            ln = math.hypot(dx, dy) or 1.0
            ux, uy = dx / ln, dy / ln
            nx, ny = -uy, ux
            for off, col, wd in ((-3, _PIPE, 2), (3, _PIPE, 2), (0, _PIPE_LT, 1)):
                self.create_line(p0[0] + nx * off, p0[1] + ny * off,
                                  p1[0] + nx * off, p1[1] + ny * off,
                                  fill=col, width=wd)
            nposts = max(2, min(10, int(ln / 30)))
            for k in range(1, nposts + 1):
                t = k / (nposts + 1.0)
                px, py = p0[0] + dx * t, p0[1] + dy * t
                self.create_line(px - nx * 6, py - ny * 6, px + nx * 6, py + ny * 6,
                                  fill=_PIPE_POST, width=2)

        rackA0 = self._world_to_px(cx_w + 0.04 * wxs, cy_w - 0.26 * wys)
        rackA1 = self._world_to_px(cx_w - 0.16 * wxs, cy_w - 0.26 * wys)
        pipe_rack(rackA0, rackA1)                                  # farm -> unit B
        rackB0 = self._world_to_px(cx_w + 0.14 * wxs, cy_w - 0.08 * wys)
        rackB1 = self._world_to_px(cx_w - 0.08 * wxs, cy_w + 0.24 * wys)
        pipe_rack(rackB0, rackB1)                                  # farm -> unit A

        # Three chimneys with drifting plumes along the process units.
        cr = max(3.0, min(7.0, 1.8 * scale))
        stacks = ((unitA[2] - (unitA[2] - unitA[0]) * 0.14,
                   unitA[1] + (unitA[3] - unitA[1]) * 0.20),
                  (unitA[0] + (unitA[2] - unitA[0]) * 0.16,
                   unitA[1] + (unitA[3] - unitA[1]) * 0.30),
                  (unitB[2] - (unitB[2] - unitB[0]) * 0.18,
                   unitB[3] - (unitB[3] - unitB[1]) * 0.25))
        for chx, chy in stacks:
            self.create_oval(chx - cr, chy - cr, chx + cr, chy + cr,
                              fill="#6b7280", outline="#3f4954", width=1)
            self.create_oval(chx - cr * 0.4, chy - cr * 0.4,
                              chx + cr * 0.4, chy + cr * 0.4,
                              fill="#2f3944", outline="")
            for drift, grow, stip in ((1.6, 1.5, "gray50"), (3.4, 2.3, "gray25"),
                                       (5.6, 3.2, "gray12")):
                sx, sy = chx + cr * drift, chy - cr * drift * 0.8
                sr = cr * grow
                self.create_oval(sx - sr, sy - sr * 0.8, sx + sr, sy + sr * 0.8,
                                  fill=_IND_SMOKE, outline="", stipple=stip)

        # Minimal parking: one short strip by the control room, cars
        # jittered off stall-centre.
        px0, py0, px1, py1 = frect(0.06, 0.34, 0.26, 0.42)
        self.create_rectangle(px0, py0, px1, py1, fill=_IND_LOT, outline="")
        self.create_rectangle(px0, py0, px1, py1,
                               fill=_IND_ROAD_GRAIN, outline="", stipple="gray25")
        nstall = max(3, min(6, int((px1 - px0) / 16)))
        for i in range(1, nstall):
            sx = px0 + (px1 - px0) * i / nstall
            self.create_line(sx, py0 + 2, sx, py1 - 2, fill=_IND_ROAD_MARK, width=1)
        for i in range(nstall):
            if self._hash01(163, i) < 0.4:
                continue
            sx = px0 + (px1 - px0) * (i + 0.5) / nstall
            cw, ch = (px1 - px0) / nstall * 0.32, (py1 - py0) * 0.36
            jx = (self._hash01(168, i) - 0.5) * cw * 0.6
            jy = (self._hash01(169, i) - 0.5) * ch * 0.35
            col = _CAR_COLORS[int(self._hash01(167, i) * len(_CAR_COLORS)) % len(_CAR_COLORS)]
            self.create_rectangle(sx + jx - cw, (py0 + py1) / 2 + jy - ch,
                                   sx + jx + cw, (py0 + py1) / 2 + jy + ch,
                                   fill=col, outline="#4a545e", width=1)

        self._map_chrome(W, H)

    def _bg_industrial_v3(self, W: int, H: int) -> None:
        """Layout variant 3 -- "Business Park": light industry. Five
        office-style blocks around a road cross, landscaped lawns with
        mowing stripes and trees, two large parking lots, footpaths -- no
        tanks, no chimneys, no fence. Offices are registered as obstacles
        at 8-12 dB (light steel-frame/glass construction passes far more
        sub-GHz signal than plant concrete)."""
        self.create_rectangle(0, 0, W, H, fill=_IND_BASE, outline="")
        # Two-layer paving grain (see v1).
        self.create_rectangle(0, 0, W, H, fill=_IND_GRAIN_DK, outline="", stipple="gray12")
        self.create_rectangle(0, 0, W, H, fill=_IND_GRAIN_LT, outline="", stipple="gray12")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        scale = self._transform()[0]

        def frect(fx0, fy0, fx1, fy1):
            p0 = self._world_to_px(cx_w + fx0 * wxs, cy_w + fy0 * wys)
            p1 = self._world_to_px(cx_w + fx1 * wxs, cy_w + fy1 * wys)
            return (min(p0[0], p1[0]), min(p0[1], p1[1]),
                    max(p0[0], p1[0]), max(p0[1], p1[1]))

        def wrect(fx0, fy0, fx1, fy1):
            return self._wrect(cx_w, cy_w, wxs, wys, fx0, fy0, fx1, fy1)

        # See _bg_industrial_v1's frect_sized for why: fixed real-world
        # footprint (m) centred at a fractional position, instead of the
        # office block spanning its whole hand-authored fractional box.
        def frect_sized(fcx, fcy, w_m, h_m):
            return frect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        def wrect_sized(fcx, fcy, w_m, h_m):
            return wrect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        # Lawns first so buildings/roads sit on top of the landscaping:
        # turf-grain stipple + subtle mowing-stripe arcs.
        lawns = ((-0.46, -0.10, -0.16, 0.10), (0.14, -0.12, 0.46, 0.10),
                 (-0.12, 0.30, 0.10, 0.46), (-0.46, -0.46, -0.30, -0.30))
        for li, (lx0, ly0, lx1, ly1) in enumerate(lawns):
            r = frect(lx0, ly0, lx1, ly1)
            self.create_oval(*r, fill=_LAWN, outline="")
            self.create_oval(*r, fill=_LAWN_GRAIN, outline="", stipple="gray12")
            for k in (0.35, 0.62):
                self.create_arc(r[0] + 3, r[1] + 3, r[2] - 3, r[3] - 3,
                                 start=200 - k * 120, extent=52, style="arc",
                                 outline=_LAWN_LT, width=1)

        # Road cross: soft stippled edges, asphalt, wear band, lane dashes.
        road_px = max(8, min(22, int(6.0 * scale)))
        rh0, rh1 = self._world_to_px(xmin, cy_w), self._world_to_px(xmax, cy_w)
        rv0, rv1 = self._world_to_px(cx_w, ymin), self._world_to_px(cx_w, ymax)
        for p0, p1 in ((rh0, rh1), (rv0, rv1)):
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD_EDGE,
                              width=road_px + 3, stipple="gray25")
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD, width=road_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD_GRAIN,
                              width=road_px - 3, stipple="gray25")
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_IND_ROAD_MARK,
                              width=1, dash=(7, 6))

        # Five office blocks (light construction -> 8-12 dB obstacles).
        # Fractional CENTRE (unchanged from the old hand-authored boxes) +
        # a realistic light-office footprint size (m).
        offices = ((-0.32, 0.29, 9.0, "office-nw"),
                   (0.04, 0.29, 10.0, "office-n"),
                   (0.34, 0.30, 12.0, "office-ne"),
                   (-0.33, -0.30, 8.0, "office-sw"),
                   (0.34, -0.31, 11.0, "office-se"))
        for k, (fcx, fcy, loss, label) in enumerate(offices):
            self._ind_building(*frect_sized(fcx, fcy, 42.0, 30.0), fill=_IND_OFFICE, seed=81 + 2 * k)
            wx0, wy0, wx1, wy1 = wrect_sized(fcx, fcy, 42.0, 30.0)
            self._register_obstacle(Obstacle(kind="rect", x0=wx0, y0=wy0,
                                              x1=wx1, y1=wy1,
                                              loss_db=loss, label=label))

        # Two big parking lots south of the horizontal road: asphalt with
        # wear grain, stall lines, cars jittered off stall-centre.
        for pi, pf in enumerate(((-0.16, -0.34, 0.18, -0.20),
                                  (-0.16, -0.14, 0.18, -0.04))):
            px0, py0, px1, py1 = frect(*pf)
            self.create_rectangle(px0, py0, px1, py1, fill=_IND_LOT, outline="")
            self.create_rectangle(px0, py0, px1, py1,
                                   fill=_IND_ROAD_GRAIN, outline="", stipple="gray25")
            nstall = max(5, min(11, int((px1 - px0) / 15)))
            for i in range(1, nstall):
                sx = px0 + (px1 - px0) * i / nstall
                self.create_line(sx, py0 + 2, sx, py1 - 2,
                                  fill=_IND_ROAD_MARK, width=1)
            for i in range(nstall):
                if self._hash01(171 + pi, i) < 0.30:
                    continue
                sx = px0 + (px1 - px0) * (i + 0.5) / nstall
                cw, ch = (px1 - px0) / nstall * 0.32, (py1 - py0) * 0.36
                jx = (self._hash01(191 + pi, i) - 0.5) * cw * 0.6
                jy = (self._hash01(193 + pi, i) - 0.5) * ch * 0.35
                col = _CAR_COLORS[int(self._hash01(177 + pi, i) * len(_CAR_COLORS))
                                  % len(_CAR_COLORS)]
                self.create_rectangle(sx + jx - cw, (py0 + py1) / 2 + jy - ch,
                                       sx + jx + cw, (py0 + py1) / 2 + jy + ch,
                                       fill=col, outline="#4a545e", width=1)

        # Footpaths from the road cross to the north offices.
        for fx, fy in ((-0.32, 0.16), (0.04, 0.18), (0.34, 0.18)):
            p0 = self._world_to_px(cx_w + fx * wxs, cy_w)
            p1 = self._world_to_px(cx_w + fx * wxs, cy_w + fy * wys)
            self.create_line(p0[0], p0[1], p1[0], p1[1],
                              fill=_IND_APRON, width=max(2, road_px // 3))

        # Trees dotted over the lawns.
        trr = max(3.0, min(8.0, 1.6 * scale))
        for i in range(9):
            lx0, ly0, lx1, ly1 = lawns[i % len(lawns)]
            fx = lx0 + (lx1 - lx0) * (0.2 + 0.6 * self._hash01(181, i))
            fy = ly0 + (ly1 - ly0) * (0.2 + 0.6 * self._hash01(183, i))
            px, py = self._world_to_px(cx_w + fx * wxs, cy_w + fy * wys)
            self._tree(px, py, trr * (0.7 + 0.5 * self._hash01(187, i)))

        self._map_chrome(W, H)

    # ── Smart City ────────────────────────────────────────────────────────
    def _city_tower(self, bx0: float, by0: float, bx1: float, by1: float,
                     tier: int, seed: int) -> None:
        """Extruded pseudo-3D tower: a real box with a lit front (south)
        glass wall, a shaded side (east) wall and a roof cap, built by
        shifting the ground footprint up-and-right by an oblique "height"
        vector and connecting the two rectangles with wall polygons --
        the classic 2.5D city-builder trick for faking real height on an
        otherwise flat top-down map without re-projecting the whole
        scene. tier 1 = tallest/deepest glass/densest windows/rooftop
        helipad, tier 3 = shortest. Callers must draw towers in
        back-to-front (north-to-south / ascending screen-y) order so a
        nearer tower's extrusion correctly overlaps a farther one behind
        it, the same painter's-algorithm rule any skyline renderer needs."""
        w, fh = bx1 - bx0, by1 - by0
        base = min(w, fh)
        mult = {1: 2.3, 2: 1.5, 3: 0.85}.get(tier, 1.5)
        h = base * mult * (0.85 + 0.3 * self._hash01(seed, 7))
        skew = h * 0.36

        glass = {1: _TOWER_GLASS_1, 2: _TOWER_GLASS_2, 3: _TOWER_GLASS_3}.get(tier, _TOWER_GLASS_2)
        glass_side = self._shade_hex(glass, 0.74)
        roof = _TOWER_ROOF_LT

        fbl, fbr = (bx0, by1), (bx1, by1)
        bbr = (bx1, by0)
        ftl, ftr = (bx0 + skew, by1 - h), (bx1 + skew, by1 - h)
        btl, btr = (bx0 + skew, by0 - h), (bx1 + skew, by0 - h)

        sd = min(26.0, h * 0.30)
        self.create_rectangle(bx0 + sd, by0 + sd * 0.5, bx1 + sd, by1 + sd * 0.5,
                               fill=_SHADOW, outline="", stipple="gray50")

        self.create_polygon(fbr[0], fbr[1], bbr[0], bbr[1], btr[0], btr[1], ftr[0], ftr[1],
                             fill=glass_side, outline=_TOWER_EDGE, width=1)
        self.create_polygon(fbl[0], fbl[1], fbr[0], fbr[1], ftr[0], ftr[1], ftl[0], ftl[1],
                             fill=glass, outline=_TOWER_EDGE, width=2)
        self.create_polygon(ftl[0], ftl[1], ftr[0], ftr[1], btr[0], btr[1], btl[0], btl[1],
                             fill=roof, outline=_TOWER_EDGE, width=1)

        if h > 10 and w > 8:
            ncols = {1: 6, 2: 5, 3: 3}.get(tier, 4)
            nrows = max(2, int(h / 9))
            self._quad_grid_lines(fbl, fbr, ftl, ftr, ncols, nrows, _TOWER_WINDOW)
        if h > 14 and fh > 8:
            self._quad_grid_lines(fbr, bbr, ftr, btr, 2, max(2, int(h / 11)), _TOWER_WINDOW)

        if tier == 1:
            rx, ry = (bx0 + bx1) / 2.0 + skew, (by0 + by1) / 2.0 - h
            self.create_oval(rx - 3, ry - 3, rx + 3, ry + 3,
                              fill=roof, outline=_TOWER_EDGE, width=1)

    def _bg_city_highway(self, W: int, H: int) -> None:
        """cars_uavs mode's dedicated Smart City background: a real
        highway, not the generic 3x3 downtown grid v1/v2/v3 draw. Its two
        paved lanes sit at world y = +/-car_lane_offset_m -- the exact
        same y real cars/scooters drive at (see CarsUavsBuilder's
        car_lane_offset_m / sim11ah/mobility.py's highway_bounce_step) --
        rather than at some arbitrary fraction of the live view's
        bounding box, so a car always renders sitting on this road, not
        floating beside it. Cars/scooters only ever move along x at that
        fixed lane y (highway_bounce_step never touches y), so this is a
        standing guarantee, not something that can drift out of sync.

        Three depth rows of low-rise-to-mid-rise buildings line both
        shoulders the length of the corridor -- denser/shorter near the
        road, sparser/taller further back, the usual "skyline recedes
        from the highway" read -- instead of the single roadside row this
        used to draw, so the city reads as an actual district flanking
        the highway rather than a thin ribbon glued to its shoulder.
        Mirrors the 3D view's own dedicated cars_uavs road loops
        (ui/web3d/snapshot.py's _road_loops)."""
        self.create_rectangle(0, 0, W, H, fill=_CITY_PARK_LT, outline="")
        scale = self._transform()[0]

        topo_cfg = self.sim.config.get("topology", {})
        car_off = float(topo_cfg.get("car_lane_offset_m", 25.0))
        span = float(topo_cfg.get("corridor_span_m", 0.0))

        # The built-up zone extends well past each end AP and runs deep
        # back from the road on each side -- fixed world coordinates, not
        # the live auto-fit bounds (which shrink/grow as UAVs wander), so
        # the city always reads as a real district continuing past the
        # visible cluster rather than one that resizes under it.
        margin = 150.0
        x0, x1 = -margin, span + margin
        depth = 260.0

        # Paved urban ground (concrete checkerboard) under the whole
        # built-up zone, so buildings sit on pavement instead of floating
        # on bare grass -- same base-tone treatment v1-v3 give their own
        # block grid.
        n_bands = max(1, int((x1 - x0) / 350.0))
        band_w = (x1 - x0) / n_bands
        for bi in range(n_bands):
            gx0, gx1 = x0 + bi * band_w, x0 + (bi + 1) * band_w
            tone = _CITY_BASE if bi % 2 == 0 else _CITY_BASE_ALT
            px0, py0 = self._world_to_px(gx0, car_off + depth)
            px1, py1 = self._world_to_px(gx1, -(car_off + depth))
            self.create_rectangle(min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1),
                                   fill=tone, outline="")

        road_px = max(10, min(26, int(7.0 * scale)))
        sidewalk_px = max(3, int(road_px * 0.28))
        for lane_y in (car_off, -car_off):
            p0 = self._world_to_px(x0, lane_y)
            p1 = self._world_to_px(x1, lane_y)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_SIDEWALK,
                              width=road_px + sidewalk_px * 2)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD, width=road_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD_MARK,
                              width=1, dash=(8, 7))

        # Three depth rows per side: (offset from the road's outer edge,
        # along-corridor spacing, (P(tier==1), P(tier<=2))) -- row 0 is
        # dense/low-rise right off the shoulder, row 2 is sparse/tall
        # farthest back.
        losses = {1: 30.0, 2: 22.0, 3: 15.0}
        tower_wh = {1: (34.0, 30.0), 2: (27.0, 23.0), 3: (20.0, 18.0)}
        rows = (
            (car_off + 26.0, 70.0, (0.05, 0.45)),
            (car_off + 26.0 + 90.0, 85.0, (0.20, 0.65)),
            (car_off + 26.0 + 180.0, 100.0, (0.40, 0.85)),
        )
        for row_sign in (1.0, -1.0):
            # _city_tower must be drawn back-to-front (ascending screen-y,
            # i.e. north/farthest first) for its extrusion to occlude
            # correctly -- north is +y on the +1 side (farthest row =
            # largest depth = smallest screen-y = draw first) but -y on
            # the -1 side (farthest row = most-negative y = LARGEST
            # screen-y = draw last), so the two sides iterate in opposite
            # row order.
            row_order = range(len(rows) - 1, -1, -1) if row_sign > 0 else range(len(rows))
            for row_idx in row_order:
                row_depth, spacing, (p_t1, p_t12) = rows[row_idx]
                n_slots = max(1, int((x1 - x0) / spacing))
                for i in range(n_slots + 1):
                    wx = x0 + i * spacing
                    if self._hash01(int(wx), row_idx, int(row_sign), 41) < 0.28:
                        continue  # the occasional gap, not solid wall-to-wall
                    r01 = self._hash01(int(wx), row_idx, int(row_sign), 42)
                    tier = 1 if r01 < p_t1 else (2 if r01 < p_t12 else 3)
                    w_m, h_m = tower_wh[tier]
                    wy = row_sign * (row_depth + h_m / 2.0)
                    bx0, by0 = self._world_to_px(wx - w_m / 2.0, wy - h_m / 2.0)
                    bx1, by1 = self._world_to_px(wx + w_m / 2.0, wy + h_m / 2.0)
                    rect = (min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))
                    self._city_tower(*rect, tier=tier,
                                      seed=int(wx) * 7 + row_idx * 101 + (1 if row_sign > 0 else 0))
                    self._register_obstacle(Obstacle(
                        kind="rect",
                        x0=wx - w_m / 2.0, y0=wy - h_m / 2.0,
                        x1=wx + w_m / 2.0, y1=wy + h_m / 2.0,
                        loss_db=losses[tier], label=f"hwy-{row_idx}-{int(wx)}-{int(row_sign)}",
                    ))

        self._map_chrome(W, H)

    def _bg_city_v1(self, W: int, H: int) -> None:
        """Layout variant 1 -- "Downtown Grid": a 3x3 block street grid,
        sidewalks and crosswalks at every intersection, five towers of
        mixed height around a small central plaza and two park blocks.
        Each tower is registered as an obstacle scaled by height tier
        (tallest ~30 dB, mid ~22 dB, low ~15 dB)."""
        self.create_rectangle(0, 0, W, H, fill=_CITY_BASE, outline="")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        scale = self._transform()[0]

        def frect(fx0, fy0, fx1, fy1):
            p0 = self._world_to_px(cx_w + fx0 * wxs, cy_w + fy0 * wys)
            p1 = self._world_to_px(cx_w + fx1 * wxs, cy_w + fy1 * wys)
            return (min(p0[0], p1[0]), min(p0[1], p1[1]),
                    max(p0[0], p1[0]), max(p0[1], p1[1]))

        def wrect(fx0, fy0, fx1, fy1):
            return self._wrect(cx_w, cy_w, wxs, wys, fx0, fy0, fx1, fy1)

        # Building footprint of a realistic absolute size (metres), centred
        # at a FRACTIONAL position -- unlike frect/wrect above, whose whole
        # (fx0,fy0)-(fx1,fy1) span was previously used directly as a
        # building's footprint too. That made a "tower" as wide as its
        # entire grid block (hundreds of metres once the per-environment
        # path-loss recalibration made `wxs`/`wys` realistically span
        # ~1.5-2km): the BLOCK'S own position/extent legitimately scales
        # with the map (that's what keeps blocks apart at a sensible
        # street-grid spacing), but the actual building sitting on one
        # block should stay a realistic size regardless of how large the
        # map is. Also used to size the matching PHY obstacle (via
        # wrect_sized) so an oversized footprint can't apply its loss_db
        # over a much larger real-world area than the visible building.
        def frect_sized(fcx, fcy, w_m, h_m):
            return frect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        def wrect_sized(fcx, fcy, w_m, h_m):
            return wrect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        cols = ((-0.46, -0.19), (-0.11, 0.11), (0.19, 0.46))
        rows = ((-0.46, -0.19), (-0.11, 0.11), (0.19, 0.46))

        for ci, (cx0, cx1) in enumerate(cols):
            for ri, (ry0, ry1) in enumerate(rows):
                tone = _CITY_BASE_ALT if (ci + ri) % 2 == 0 else _CITY_BASE
                self.create_rectangle(*frect(cx0, ry0, cx1, ry1), fill=tone, outline="")

        road_px = max(10, min(26, int(7.0 * scale)))
        sidewalk_px = max(3, int(road_px * 0.28))
        for fx in (-0.15, 0.15):
            p0 = self._world_to_px(cx_w + fx * wxs, ymin)
            p1 = self._world_to_px(cx_w + fx * wxs, ymax)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_SIDEWALK,
                              width=road_px + sidewalk_px * 2)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD, width=road_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD_MARK,
                              width=1, dash=(8, 7))
        for fy in (-0.15, 0.15):
            p0 = self._world_to_px(xmin, cy_w + fy * wys)
            p1 = self._world_to_px(xmax, cy_w + fy * wys)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_SIDEWALK,
                              width=road_px + sidewalk_px * 2)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD, width=road_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD_MARK,
                              width=1, dash=(8, 7))

        for fx in (-0.15, 0.15):
            for fy in (-0.15, 0.15):
                ix, iy = self._world_to_px(cx_w + fx * wxs, cy_w + fy * wys)
                half = road_px * 0.55
                for k in range(-2, 3):
                    off = k * (half / 2.2)
                    self.create_line(ix - half, iy + off, ix + half, iy + off,
                                      fill=_CITY_CROSSWALK, width=2)

        layout = (
            (0, 0, 2, "twr-a"), (1, 0, 1, "twr-b"), (2, 0, 3, "twr-c"),
            (0, 1, None, "park-a"), (1, 1, None, "plaza"), (2, 1, 3, "twr-d"),
            (0, 2, 3, "twr-e"), (1, 2, None, "park-b"), (2, 2, 2, "twr-f"),
        )
        losses = {1: 30.0, 2: 22.0, 3: 15.0}
        # Real building footprints (m) per height tier -- was the whole
        # block span (see frect_sized's own comment).
        tower_wh = {1: (42.0, 34.0), 2: (34.0, 28.0), 3: (26.0, 22.0)}
        for ci, ri, tier, label in layout:
            cx0, cx1 = cols[ci]
            ry0, ry1 = rows[ri]
            if tier is None:
                r = frect(cx0 + 0.02, ry0 + 0.02, cx1 - 0.02, ry1 - 0.02)
                self.create_rectangle(*r, fill=_CITY_PARK, outline="")
                self.create_oval(r[0] + (r[2] - r[0]) * 0.2, r[1] + (r[3] - r[1]) * 0.2,
                                  r[0] + (r[2] - r[0]) * 0.8, r[1] + (r[3] - r[1]) * 0.8,
                                  fill=_CITY_PARK_LT, outline="")
                continue
            fcx, fcy = (cx0 + cx1) / 2.0, (ry0 + ry1) / 2.0
            w_m, h_m = tower_wh[tier]
            self._city_tower(*frect_sized(fcx, fcy, w_m, h_m), tier=tier, seed=hash(label) & 0xFFFF)
            wx0, wy0, wx1, wy1 = wrect_sized(fcx, fcy, w_m, h_m)
            self._register_obstacle(Obstacle(kind="rect", x0=wx0, y0=wy0, x1=wx1, y1=wy1,
                                              loss_db=losses[tier], label=label))

        self._map_chrome(W, H)

    def _bg_city_v2(self, W: int, H: int) -> None:
        """Layout variant 2 -- "Business District": a denser downtown
        core -- every block on the grid holds a tower (no parks), skewed
        toward the taller tiers for a proper skyline, on a wider arterial
        grid than v1."""
        self.create_rectangle(0, 0, W, H, fill=_CITY_BASE, outline="")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        scale = self._transform()[0]

        def frect(fx0, fy0, fx1, fy1):
            p0 = self._world_to_px(cx_w + fx0 * wxs, cy_w + fy0 * wys)
            p1 = self._world_to_px(cx_w + fx1 * wxs, cy_w + fy1 * wys)
            return (min(p0[0], p1[0]), min(p0[1], p1[1]),
                    max(p0[0], p1[0]), max(p0[1], p1[1]))

        def wrect(fx0, fy0, fx1, fy1):
            return self._wrect(cx_w, cy_w, wxs, wys, fx0, fy0, fx1, fy1)

        # See v1's frect_sized/wrect_sized for why buildings need a fixed
        # real-world size instead of the whole (oversized) block span.
        def frect_sized(fcx, fcy, w_m, h_m):
            return frect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        def wrect_sized(fcx, fcy, w_m, h_m):
            return wrect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        cols = ((-0.46, -0.19), (-0.11, 0.11), (0.19, 0.46))
        rows = ((-0.46, -0.19), (-0.11, 0.11), (0.19, 0.46))

        for ci, (cx0, cx1) in enumerate(cols):
            for ri, (ry0, ry1) in enumerate(rows):
                tone = _CITY_BASE_ALT if (ci + ri) % 2 == 0 else _CITY_BASE
                self.create_rectangle(*frect(cx0, ry0, cx1, ry1), fill=tone, outline="")

        road_px = max(12, min(30, int(8.5 * scale)))
        sidewalk_px = max(3, int(road_px * 0.24))
        for fx in (-0.15, 0.15):
            p0 = self._world_to_px(cx_w + fx * wxs, ymin)
            p1 = self._world_to_px(cx_w + fx * wxs, ymax)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_SIDEWALK,
                              width=road_px + sidewalk_px * 2)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD, width=road_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD_MARK,
                              width=1, dash=(8, 7))
        for fy in (-0.15, 0.15):
            p0 = self._world_to_px(xmin, cy_w + fy * wys)
            p1 = self._world_to_px(xmax, cy_w + fy * wys)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_SIDEWALK,
                              width=road_px + sidewalk_px * 2)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD, width=road_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD_MARK,
                              width=1, dash=(8, 7))

        for fx in (-0.15, 0.15):
            for fy in (-0.15, 0.15):
                ix, iy = self._world_to_px(cx_w + fx * wxs, cy_w + fy * wys)
                half = road_px * 0.55
                for k in range(-2, 3):
                    off = k * (half / 2.2)
                    self.create_line(ix - half, iy + off, ix + half, iy + off,
                                      fill=_CITY_CROSSWALK, width=2)

        losses = {1: 30.0, 2: 22.0, 3: 15.0}
        # Slightly bigger footprints than v1's downtown mix -- a business
        # district's towers really do run larger, but still a realistic
        # single-building size, not a whole block.
        tower_wh = {1: (48.0, 38.0), 2: (38.0, 30.0), 3: (28.0, 24.0)}
        # Row-outer / column-inner so every block is drawn back-to-front
        # (north to south, ascending screen-y) -- required for the new
        # extruded _city_tower boxes to occlude correctly.
        for ri in range(3):
            for ci in range(3):
                cx0, cx1 = cols[ci]
                ry0, ry1 = rows[ri]
                tval = 1 if self._hash01(200 + ci, 200 + ri) < 0.4 else (
                    2 if self._hash01(210 + ci, 210 + ri) < 0.75 else 3)
                fcx, fcy = (cx0 + cx1) / 2.0, (ry0 + ry1) / 2.0
                w_m, h_m = tower_wh[tval]
                self._city_tower(*frect_sized(fcx, fcy, w_m, h_m), tier=tval, seed=1000 + ci * 3 + ri)
                wx0, wy0, wx1, wy1 = wrect_sized(fcx, fcy, w_m, h_m)
                self._register_obstacle(Obstacle(kind="rect", x0=wx0, y0=wy0, x1=wx1, y1=wy1,
                                                  loss_db=losses[tval], label=f"tower-{ci}-{ri}"))

        self._map_chrome(W, H)

    def _bg_city_v3(self, W: int, H: int) -> None:
        """Layout variant 3 -- "Suburban Corridor": a single wide
        boulevard lined with low-rise (tier-3 only) commercial buildings,
        parking lots and green verges -- no towers, so overall
        obstruction is much lighter than the downtown variants, matching
        a lower-density deployment."""
        self.create_rectangle(0, 0, W, H, fill=_CITY_PARK_LT, outline="")
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        scale = self._transform()[0]

        def frect(fx0, fy0, fx1, fy1):
            p0 = self._world_to_px(cx_w + fx0 * wxs, cy_w + fy0 * wys)
            p1 = self._world_to_px(cx_w + fx1 * wxs, cy_w + fy1 * wys)
            return (min(p0[0], p1[0]), min(p0[1], p1[1]),
                    max(p0[0], p1[0]), max(p0[1], p1[1]))

        def wrect(fx0, fy0, fx1, fy1):
            return self._wrect(cx_w, cy_w, wxs, wys, fx0, fy0, fx1, fy1)

        # See v1's frect_sized/wrect_sized -- the LOT itself (drawn as the
        # `else` fallback below when a slot has no building) can stay
        # slot-sized, a real surface lot legitimately spans that much
        # ground; only the BUILDING footprint needs clamping to a
        # realistic low-rise commercial size.
        def frect_sized(fcx, fcy, w_m, h_m):
            return frect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        def wrect_sized(fcx, fcy, w_m, h_m):
            return wrect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        road_px = max(16, min(40, int(11.0 * scale)))
        p0 = self._world_to_px(xmin, cy_w)
        p1 = self._world_to_px(xmax, cy_w)
        self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_SIDEWALK, width=road_px + 14)
        self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD, width=road_px)
        self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_CITY_ROAD_MARK,
                          width=1, dash=(9, 8))

        n = 5
        for side in (-1, 1):
            for i in range(n):
                fx0 = -0.42 + i * (0.84 / n)
                fx1 = fx0 + 0.84 / n - 0.03
                fyA = 0.10 * side
                fyB = fyA + 0.28 * side
                bf = (fx0, min(fyA, fyB), fx1, max(fyA, fyB))
                if self._hash01(300 + i, side + 5) < 0.75:
                    fcx, fcy = (bf[0] + bf[2]) / 2.0, (bf[1] + bf[3]) / 2.0
                    self._city_tower(*frect_sized(fcx, fcy, 30.0, 22.0), tier=3, seed=300 + i * 2 + side)
                    wx0, wy0, wx1, wy1 = wrect_sized(fcx, fcy, 30.0, 22.0)
                    self._register_obstacle(Obstacle(
                        kind="rect", x0=wx0, y0=wy0, x1=wx1, y1=wy1,
                        loss_db=15.0, label=f"lowrise-{side}-{i}"))
                else:
                    r = frect(*bf)
                    self.create_rectangle(*r, fill=_IND_LOT, outline=_CITY_ROAD_EDGE, width=1)
                    nstall = 4
                    for s in range(1, nstall):
                        sx = r[0] + (r[2] - r[0]) * s / nstall
                        self.create_line(sx, r[1] + 2, sx, r[3] - 2,
                                          fill=_CITY_CROSSWALK, width=1)

                gr = frect(fx0, 0.0, fx1, fyA * 0.35)
                self.create_rectangle(*gr, fill=_CITY_PARK, outline="")

        self._map_chrome(W, H)

    # ── Military Zone props ──────────────────────────────────────────────
    def _draw_tank_icon(self, px: float, py: float, heading: float, color: str,
                         tags: tuple = ()) -> None:
        """Top-down main-battle-tank glyph, oriented along its heading: cast
        shadow, dark track rails flanking the hull, an angular hull body,
        an offset turret and a gun barrel projecting past the front glacis
        -- same "reads clearly at map scale" brief as _draw_car_icon, just
        armoured instead of civilian."""
        length, width = 13.0, 7.5
        ch, sh = math.cos(heading), math.sin(heading)
        perp = heading + math.pi / 2.0
        cp, sp = math.cos(perp), math.sin(perp)

        self.create_oval(px - length * 0.55 + 2, py - width * 0.55 + 2,
                          px + length * 0.55 + 2, py + width * 0.55 + 2,
                          fill=_SHADOW, outline="", stipple="gray50", tags=tags)

        for side in (-1, 1):
            tx0 = px + side * (width * 0.52) * cp - length * 0.52 * ch
            ty0 = py + side * (width * 0.52) * sp - length * 0.52 * sh
            tx1 = px + side * (width * 0.52) * cp + length * 0.52 * ch
            ty1 = py + side * (width * 0.52) * sp + length * 0.52 * sh
            self.create_line(tx0, ty0, tx1, ty1, fill=_MIL_TANK_DK, width=3,
                              capstyle="projecting", tags=tags)

        corners = []
        for dl, dw in ((length * 0.5, -width * 0.36), (length * 0.5, width * 0.36),
                       (-length * 0.5, width * 0.36), (-length * 0.5, -width * 0.36)):
            corners.extend([px + dl * ch + dw * cp, py + dl * sh + dw * sp])
        self.create_polygon(*corners, fill=color, outline=_MIL_TANK_DK, width=1, tags=tags)

        turx, tury = px - length * 0.05 * ch, py - length * 0.05 * sh
        tr = width * 0.32
        self.create_oval(turx - tr, tury - tr, turx + tr, tury + tr,
                          fill=_MIL_TANK_TURRET, outline=_MIL_TANK_DK, width=1, tags=tags)

        bx, by = turx + length * 0.62 * ch, tury + length * 0.62 * sh
        self.create_line(turx, tury, bx, by, fill=_MIL_TANK_DK, width=2,
                          capstyle="round", tags=tags)

    def _draw_soldier_icon(self, px: float, py: float, heading: float,
                            tags: tuple = ()) -> None:
        """Small top-down infantry glyph: a soft cast shadow, a camo-toned
        body dot, and a skin-toned "head" nub offset toward the direction
        of travel so a patrolling soldier visibly faces where it's
        walking, the same heading convention _draw_car_icon/_draw_tank_icon
        use."""
        self.create_oval(px - 2.6 + 1, py - 2.6 + 1, px + 2.6 + 1, py + 2.6 + 1,
                          fill=_SHADOW, outline="", stipple="gray50", tags=tags)
        self.create_oval(px - 2.6, py - 2.6, px + 2.6, py + 2.6,
                          fill=_MIL_SOLDIER, outline="#2f331f", width=1, tags=tags)
        hx, hy = px + 1.3 * math.cos(heading), py + 1.3 * math.sin(heading)
        self.create_oval(hx - 1.1, hy - 1.1, hx + 1.1, hy + 1.1,
                          fill=_MIL_SOLDIER_SKIN, outline="", tags=tags)

    def _mil_bunker(self, bx0: float, by0: float, bx1: float, by1: float, seed: int) -> None:
        """Squat sandbagged bunker seen from above: cast shadow, a low
        concrete/earth slab (not extruded tall like a warehouse -- real
        firing bunkers keep a low profile), a sandbag-row texture around
        the whole perimeter, and a dark firing slit facing out."""
        w, h = bx1 - bx0, by1 - by0
        off = max(2.0, min(6.0, 0.06 * max(w, h)))
        self.create_rectangle(bx0 + off, by0 + off, bx1 + off, by1 + off,
                               fill=_SHADOW, outline="", stipple="gray50")
        self.create_rectangle(bx0, by0, bx1, by1, fill=_MIL_BUNKER,
                               outline=_MIL_BUNKER_EDGE, width=1)
        self.create_rectangle(bx0, by0, bx1, by1, fill=_MIL_BUNKER_DK,
                               outline="", stipple="gray25")
        bag = max(2.5, min(5.0, 0.05 * min(w, h)))
        n_top = max(3, int(w / max(1.0, bag * 1.6)))
        for i in range(n_top):
            sx = bx0 + w * (i + 0.5) / n_top
            jy = (self._hash01(seed, i, 1) - 0.5) * bag * 0.4
            self.create_oval(sx - bag, by0 - bag * 0.5 + jy, sx + bag, by0 + bag * 0.5 + jy,
                              fill=_MIL_SANDBAG_LT, outline=_MIL_BUNKER_EDGE, width=1)
            self.create_oval(sx - bag, by1 - bag * 0.5 - jy, sx + bag, by1 + bag * 0.5 - jy,
                              fill=_MIL_SANDBAG_LT, outline=_MIL_BUNKER_EDGE, width=1)
        n_side = max(2, int(h / max(1.0, bag * 1.6)))
        for i in range(n_side):
            sy = by0 + h * (i + 0.5) / n_side
            self.create_oval(bx0 - bag * 0.5, sy - bag, bx0 + bag * 0.5, sy + bag,
                              fill=_MIL_SANDBAG_LT, outline=_MIL_BUNKER_EDGE, width=1)
            self.create_oval(bx1 - bag * 0.5, sy - bag, bx1 + bag * 0.5, sy + bag,
                              fill=_MIL_SANDBAG_LT, outline=_MIL_BUNKER_EDGE, width=1)
        fy = (by0 + by1) / 2.0
        self.create_line(bx0 + w * 0.2, fy, bx0 + w * 0.8, fy, fill="#2a281f", width=2)

    def _mil_watchtower(self, px: float, py: float, h: float = 26.0) -> None:
        """Timber/steel watchtower: cast shadow, angled support legs to a
        raised platform, and a small flag -- the tallest silhouette on the
        base, same visual role a floodlight pole plays on the Industrial
        Site perimeter."""
        self.create_oval(px - 5 + 2, py - 3 + 2, px + 5 + 2, py + 3 + 2,
                          fill=_SHADOW, outline="", stipple="gray50")
        py2 = py - h
        self.create_line(px, py, px, py2, fill=_MIL_TOWER, width=2)
        self.create_line(px - 4, py, px, py - h * 0.68, fill=_MIL_TOWER, width=1)
        self.create_line(px + 4, py, px, py - h * 0.68, fill=_MIL_TOWER, width=1)
        self.create_rectangle(px - 4.5, py2 - 3, px + 4.5, py2 + 3,
                               fill=_MIL_BUNKER_DK, outline=_MIL_BUNKER_EDGE, width=1)
        self.create_line(px, py2 - 3, px, py2 - 8, fill=_MIL_TOWER, width=1)
        self.create_polygon(px, py2 - 8, px + 4, py2 - 6.5, px, py2 - 5,
                             fill=_MIL_FLAG, outline="")

    def _mil_crate_stack(self, px: float, py: float, seed: int) -> None:
        """Small stacked supply crates -- a cheap, cheerful prop for
        depot/motor-pool dressing, same role _tree() plays elsewhere."""
        self.create_oval(px - 5 + 1.5, py - 4 + 1.5, px + 5 + 1.5, py + 4 + 1.5,
                          fill=_SHADOW, outline="", stipple="gray50")
        for i in range(3):
            jx = (self._hash01(seed, i, 5) - 0.5) * 5.0
            jy = (self._hash01(seed, i, 7) - 0.5) * 3.0
            cw, ch = 4.6, 3.6
            self.create_rectangle(px + jx - cw, py + jy - ch, px + jx + cw, py + jy + ch,
                                   fill=_MIL_CRATE, outline="#3d3221", width=1)
            self.create_line(px + jx - cw, py + jy, px + jx + cw, py + jy,
                              fill="#3d3221", width=1)

    def _mil_crater(self, px: float, py: float, r: float) -> None:
        """Scorched blast crater -- ground scarring for flavour, no PHY
        effect (unlike bunkers/tents, a crater doesn't block a radio
        link)."""
        self.create_oval(px - r, py - r, px + r, py + r,
                          fill=_MIL_CRATER, outline="", stipple="gray25")
        self.create_oval(px - r * 0.6, py - r * 0.6, px + r * 0.6, py + r * 0.6,
                          fill="#3a3527", outline="", stipple="gray50")

    def _mil_camo_ground(self, W: int, H: int, seed: int) -> None:
        """Base scrubland fill plus irregular hashed camo-blotch polygons
        (three tones) scattered across the canvas -- reads as a dirt/scrub
        field with mottled cover instead of a flat colour panel."""
        self.create_rectangle(0, 0, W, H, fill=_MIL_BASE, outline="")
        self.create_rectangle(0, 0, W, H, fill=_MIL_BASE_ALT, outline="", stipple="gray12")
        tones = (_MIL_PATCH_1, _MIL_PATCH_2, _MIL_PATCH_3)
        n = max(10, int((W * H) / 9000))
        for i in range(n):
            cx = self._hash01(seed, i, 1) * W
            cy = self._hash01(seed, i, 2) * H
            rw = 14 + self._hash01(seed, i, 3) * 34
            rh = 10 + self._hash01(seed, i, 4) * 24
            ang = self._hash01(seed, i, 5) * 2.0
            pts = []
            for k in range(6):
                a = ang + k * math.pi / 3.0 + (self._hash01(seed, i, k) - 0.5) * 0.6
                pts.extend([cx + rw * math.cos(a), cy + rh * math.sin(a)])
            self.create_polygon(*pts, fill=tones[i % 3], outline="", stipple="gray25")

    def _draw_military_overlay(self, nodes) -> None:
        """Patrolling tanks + soldiers, pure decoration (no association/
        traffic/physics -- position is a deterministic function of sim
        time, same trick as _draw_vehicles_overlay's cars and the drone
        racetrack). Only active for the Military Zone environment."""
        if self.environment != "Military Zone" or self.sim is None:
            return
        xmin, xmax, ymin, ymax = self._bounds()
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        wxs, wys = xmax - xmin, ymax - ymin
        t_now = float(getattr(self.sim.engine, "now", 0.0))

        tank_loop = (cx_w, cy_w, wxs * 0.30, wys * 0.30, 34.0)
        tank_colors = (_MIL_TANK_HULL, "#4f5a3f")
        for i in range(2):
            phase = i / 2.0 + 0.5 * i
            t = (t_now / tank_loop[4] + phase) % 1.0
            wx, wy = self._rect_loop_pos(*tank_loop[:4], t)
            wx2, wy2 = self._rect_loop_pos(*tank_loop[:4], t + 0.004)
            px, py = self._world_to_px(wx, wy)
            px2, py2 = self._world_to_px(wx2, wy2)
            heading = math.atan2(py2 - py, px2 - px)
            self._draw_tank_icon(px, py, heading, tank_colors[i % 2], tags=("ovl",))

        patrol_loop = (cx_w, cy_w, wxs * 0.18, wys * 0.18, 20.0)
        n_soldiers = 5
        for i in range(n_soldiers):
            phase = i / n_soldiers
            t = (t_now / patrol_loop[4] + phase) % 1.0
            wx, wy = self._rect_loop_pos(*patrol_loop[:4], t)
            wx2, wy2 = self._rect_loop_pos(*patrol_loop[:4], t + 0.006)
            px, py = self._world_to_px(wx, wy)
            px2, py2 = self._world_to_px(wx2, wy2)
            heading = math.atan2(py2 - py, px2 - px)
            self._draw_soldier_icon(px, py, heading, tags=("ovl",))

    def _mil_flagpole(self, px: float, py: float, h: float = 22.0) -> None:
        """Mast + a small triangular flag -- a bit of vertical colour near
        the command bunker, second-tallest silhouette on the base after
        the watchtowers."""
        self.create_oval(px - 2 + 1, py + 1, px + 2 + 1, py + 2 + 1,
                          fill=_SHADOW, outline="", stipple="gray50")
        self.create_line(px, py, px, py - h, fill=_MIL_TOWER, width=2)
        self.create_polygon(px, py - h, px + 9, py - h + 3, px, py - h + 6,
                             fill=_MIL_FLAG, outline="")

    def _bg_military_v1(self, W: int, H: int) -> None:
        """Forward Operating Base: camo scrub ground, a dirt-track access
        cross, a full barbed-wire perimeter with four corner watchtowers,
        a sandbagged command bunker at the centre with its own tower and
        flagpole, four checkpoint bunkers guarding the perimeter (each
        with its own watchtower), a tented supply depot, a motor pool
        with parked tanks behind a sandbag revetment, a forward mortar
        pit, a helipad, and blast craters scattered throughout. The
        command bunker, checkpoints, and supply depot are also
        registered as physical obstacles (see _register_obstacle) --
        same "cosmetic overlap has real PHY consequences" rule every
        other environment here follows."""
        self._mil_camo_ground(W, H, seed=401)
        xmin, xmax, ymin, ymax = self._bounds()
        wxs, wys = xmax - xmin, ymax - ymin
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        scale = self._transform()[0]

        def frect(fx0, fy0, fx1, fy1):
            p0 = self._world_to_px(cx_w + fx0 * wxs, cy_w + fy0 * wys)
            p1 = self._world_to_px(cx_w + fx1 * wxs, cy_w + fy1 * wys)
            return (min(p0[0], p1[0]), min(p0[1], p1[1]),
                    max(p0[0], p1[0]), max(p0[1], p1[1]))

        def wrect(fx0, fy0, fx1, fy1):
            return self._wrect(cx_w, cy_w, wxs, wys, fx0, fy0, fx1, fy1)

        # See Smart City's _bg_city_v1/frect_sized for why: fixed real-world
        # footprint (m) centred at a fractional position, instead of the
        # bunker spanning its whole hand-authored fractional box (which put
        # a "checkpoint bunker" at 200m+ once the per-environment path-loss
        # recalibration made `wxs`/`wys` realistically span well over a
        # kilometre).
        def frect_sized(fcx, fcy, w_m, h_m):
            return frect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        def wrect_sized(fcx, fcy, w_m, h_m):
            return wrect(fcx - (w_m / 2.0) / wxs, fcy - (h_m / 2.0) / wys,
                         fcx + (w_m / 2.0) / wxs, fcy + (h_m / 2.0) / wys)

        # Dirt access track cross: no lane paint (this isn't a paved road),
        # just a wear-band and tyre ruts.
        track_px = max(7, min(18, int(5.0 * scale)))
        rh0, rh1 = self._world_to_px(xmin, cy_w), self._world_to_px(xmax, cy_w)
        rv0, rv1 = self._world_to_px(cx_w, ymin), self._world_to_px(cx_w, ymax)
        for p0, p1 in ((rh0, rh1), (rv0, rv1)):
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_MIL_TRACK_EDGE,
                              width=track_px + 3, stipple="gray25")
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_MIL_TRACK, width=track_px)
            self.create_line(p0[0], p0[1], p1[0], p1[1], fill=_MIL_TRACK_RUT,
                              width=2, dash=(4, 5))

        # Barbed-wire perimeter + corner watchtowers.
        f0 = self._world_to_px(xmin + wxs * 0.04, ymin + wys * 0.04)
        f1 = self._world_to_px(xmax - wxs * 0.04, ymax - wys * 0.04)
        fx0, fy0 = min(f0[0], f1[0]), min(f0[1], f1[1])
        fx1, fy1 = max(f0[0], f1[0]), max(f0[1], f1[1])
        self.create_rectangle(fx0, fy0, fx1, fy1, outline=_MIL_WIRE, dash=(3, 2), width=1)
        for cx, cy in ((fx0, fy0), (fx1, fy0), (fx0, fy1), (fx1, fy1)):
            self._mil_watchtower(cx, cy, h=28.0)

        # Command bunker (centre) -- the biggest, best-protected structure,
        # with its own watchtower and flagpole. Realistic footprint (m)
        # centred at the same fractional position as before.
        cmd_c = (0.0, 0.0)
        cmd = frect_sized(*cmd_c, 36.0, 28.0)
        self._mil_bunker(*cmd, seed=23)
        self._mil_watchtower((cmd[0] + cmd[2]) / 2.0, cmd[1] - 10, h=14.0)
        self._mil_flagpole(cmd[2] + 10, cmd[3] - 6, h=22.0)
        wx0, wy0, wx1, wy1 = wrect_sized(*cmd_c, 36.0, 28.0)
        # Visual only (self._visual_obstacles directly, NOT
        # _register_obstacle) -- every other obstacle here sits off to one
        # side of the base, so it only blocks links crossing that specific
        # side. This one is centred on the base, which is also where every
        # relay/STA link converges (they all radiate from the AP): giving
        # it real PHY effect doesn't model "a bunker in the way of some
        # traffic," it models "nothing can talk to the AP from any
        # direction," which _encloses_ap's AP-proximity margin already
        # tries to catch but can't fully -- this obstacle's own size scales
        # with the current node scatter's span, so how close its edge
        # lands to the AP is exactly as volatile as that scatter is, not a
        # fixed distance a margin can reliably bound.
        self._visual_obstacles.append(Obstacle(kind="rect", x0=wx0, y0=wy0, x1=wx1, y1=wy1,
                                                loss_db=25.0, label="command-bunker"))

        # Eight checkpoint bunkers guarding the perimeter (N/S/E/W plus the
        # four diagonals), each with its own watchtower -- doubled from the
        # original N/S/E/W-only ring for a genuinely dense defensive line,
        # not just four widely-spaced posts.
        # Fractional CENTRE (unchanged from the old hand-authored boxes) --
        # real checkpoint/guard-post bunkers are small, ~10-15m.
        checkpoints = (
            (0.0, -0.36, "checkpoint-north"),
            (0.0, 0.36, "checkpoint-south"),
            (0.36, 0.0, "checkpoint-east"),
            (-0.36, 0.0, "checkpoint-west"),
            (0.29, -0.29, "checkpoint-northeast"),
            (-0.29, -0.29, "checkpoint-northwest"),
            (0.29, 0.29, "checkpoint-southeast"),
            (-0.29, 0.29, "checkpoint-southwest"),
        )
        for fcx, fcy, label in checkpoints:
            r = frect_sized(fcx, fcy, 13.0, 11.0)
            self._mil_bunker(*r, seed=hash(label) & 0xFFFF)
            self._mil_watchtower((r[0] + r[2]) / 2.0, r[1] - 8, h=16.0)
            wx0, wy0, wx1, wy1 = wrect_sized(fcx, fcy, 13.0, 11.0)
            self._register_obstacle(Obstacle(kind="rect", x0=wx0, y0=wy0, x1=wx1, y1=wy1,
                                              loss_db=12.0, label=label))

        # Supply depot (tents) east of the vertical track: three low ridge
        # tents in a row, each capped to about as wide as it is tall (a
        # canvas ridge tent from directly above is a stretched, low
        # silhouette -- letting the triangle inherit the depot footprint's
        # full height made them look like tall spikes instead).
        depot_c = (0.24, -0.06)
        dx0, dy0, dx1, dy1 = frect_sized(*depot_c, 30.0, 18.0)
        dw = (dx1 - dx0) / 3.0
        # A floor in pixels, not just "whichever of these two is smaller" --
        # an oddly-shaped node scatter can give this fractional footprint a
        # very thin absolute height, and min() alone would happily collapse
        # the tent down with it instead of staying legible.
        tent_h = max(8.0, min(dy1 - dy0, dw * 0.9))
        for i in range(3):
            tx0, tx1 = dx0 + i * dw + 2, dx0 + (i + 1) * dw - 2
            ty0, ty1 = dy1 - tent_h, dy1
            self.create_polygon(tx0, ty1, tx1, ty1, (tx0 + tx1) / 2, ty0,
                                 fill=_MIL_TENT, outline=_MIL_TENT_DK, width=1)
            self.create_line((tx0 + tx1) / 2, ty0, (tx0 + tx1) / 2, ty1,
                              fill=_MIL_TENT_DK, width=1)
        wx0, wy0, wx1, wy1 = wrect_sized(*depot_c, 30.0, 18.0)
        self._register_obstacle(Obstacle(kind="rect", x0=wx0, y0=wy0, x1=wx1, y1=wy1,
                                          loss_db=15.0, label="supply-depot"))
        for i in range(3):
            self._mil_crate_stack(dx0 + (i + 0.5) * dw, dy1 + 10, seed=71 + i)

        # Motor pool west of the vertical track: sandbag revetment ring
        # around 2 parked tanks.
        mp_f = (-0.32, -0.16, -0.14, 0.04)
        mx0, my0, mx1, my1 = frect(*mp_f)
        self.create_rectangle(mx0, my0, mx1, my1, fill=_MIL_TRACK, outline=_MIL_TRACK_EDGE, width=1)
        bag = max(3.0, min(6.0, 0.05 * min(mx1 - mx0, my1 - my0)))
        n_top = max(3, int((mx1 - mx0) / max(1.0, bag * 1.6)))
        for i in range(n_top):
            sx = mx0 + (mx1 - mx0) * (i + 0.5) / n_top
            self.create_oval(sx - bag, my0 - bag * 0.5, sx + bag, my0 + bag * 0.5,
                              fill=_MIL_SANDBAG_LT, outline=_MIL_BUNKER_EDGE, width=1)
            self.create_oval(sx - bag, my1 - bag * 0.5, sx + bag, my1 + bag * 0.5,
                              fill=_MIL_SANDBAG_LT, outline=_MIL_BUNKER_EDGE, width=1)
        for i in range(2):
            tx = mx0 + (mx1 - mx0) * (i + 0.5) / 2
            ty = (my0 + my1) / 2.0
            self._draw_tank_icon(tx, ty, -math.pi / 2.0, _MIL_TANK_HULL if i == 0 else "#4f5a3f")

        # Forward mortar pit south-east of the base: a circular sandbag
        # ring, no roof, decorative only (a shallow open pit isn't a
        # meaningful RF blocker).
        px, py = self._world_to_px(cx_w + wxs * 0.30, cy_w + wys * 0.30)
        r = max(8.0, min(20.0, 0.05 * min(wxs, wys) * scale))
        self.create_oval(px - r, py - r, px + r, py + r, fill=_MIL_TRACK,
                          outline=_MIL_BUNKER_EDGE, width=1)
        n_bag = 10
        for i in range(n_bag):
            a = 2.0 * math.pi * i / n_bag
            bx, by = px + r * math.cos(a), py + r * math.sin(a)
            self.create_oval(bx - 3, by - 3, bx + 3, by + 3,
                              fill=_MIL_SANDBAG_LT, outline=_MIL_BUNKER_EDGE, width=1)
        self.create_line(px, py, px + r * 0.5, py - r * 0.7, fill=_MIL_TANK_DK, width=2)

        # Helipad north-east of the base: painted circle + H marking, no
        # PHY effect.
        hx, hy = self._world_to_px(cx_w + wxs * 0.30, cy_w - wys * 0.32)
        hr = max(10.0, min(26.0, 0.06 * min(wxs, wys) * scale))
        self.create_oval(hx - hr, hy - hr, hx + hr, hy + hr,
                          fill=_MIL_TRACK, outline=_MIL_FLAG, width=2)
        self.create_line(hx - hr * 0.35, hy - hr * 0.5, hx - hr * 0.35, hy + hr * 0.5,
                          fill="#e7e2cf", width=2)
        self.create_line(hx + hr * 0.35, hy - hr * 0.5, hx + hr * 0.35, hy + hr * 0.5,
                          fill="#e7e2cf", width=2)
        self.create_line(hx - hr * 0.35, hy, hx + hr * 0.35, hy, fill="#e7e2cf", width=2)

        for i in range(6):
            cx = xmin + wxs * self._hash01(83, i, 1)
            cy = ymin + wys * self._hash01(83, i, 2)
            px, py = self._world_to_px(cx, cy)
            self._mil_crater(px, py, 6.0 + 4.0 * self._hash01(83, i, 3))

        self._map_chrome(W, H)
