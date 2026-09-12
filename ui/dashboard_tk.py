"""sim11ah — Professional Dashboard for IEEE 802.11ah Wi-Fi HaLow Simulator"""
from __future__ import annotations

# Installed before any import below that could plausibly fail (numpy not
# installed for whatever interpreter actually launched this, a missing
# stdlib Tk on some builds, etc.) -- whatever launched this process
# (VS Code's Run button, a double-clicked .command file, Terminal) may
# tear down its own window/panel the instant the process exits, taking
# any printed traceback with it before there's a chance to read it. This
# writes it somewhere that outlives that, in addition to printing
# normally. sys/traceback/datetime are stdlib and effectively can't fail
# to import themselves, so this is safe to do before anything riskier.
import sys as _sys
import traceback as _traceback


def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    _sys.__excepthook__(exc_type, exc_value, exc_tb)
    try:
        import datetime
        _tb_text = "".join(_traceback.format_exception(exc_type, exc_value, exc_tb))
        with open("/tmp/sim11ah_dashboard_crash.log", "a") as _f:
            _f.write(f"\n=== {datetime.datetime.now().isoformat()} ===\n{_tb_text}")
    except Exception:
        pass


_sys.excepthook = _log_uncaught_exception

import numpy
import csv
import math
import os
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    from ui.topology_canvas import (
        NetworkCanvas, range_m_for_node, set_range_m_for_node, classify_frame_kind,
        advance_drone_positions, advance_uav_positions, _apply_environment_path_loss,
        _declutter_obstructed_nodes, save_topology, load_topology, apply_topology,
        _ap_range_m, resolve_ap_peer,
    )
except ImportError:
    from topology_canvas import (
        NetworkCanvas, range_m_for_node, set_range_m_for_node, classify_frame_kind,
        advance_drone_positions, advance_uav_positions, _apply_environment_path_loss,
        _declutter_obstructed_nodes, save_topology, load_topology, apply_topology,
        _ap_range_m, resolve_ap_peer,
    )

from sim11ah.mobility import grid_road_step, uav_bounce_step
from sim11ah.topology import CarsUavsBuilder

from sim11ah.sensor_profiles import list_profiles as _sensor_list_profiles, get_profile as _sensor_get_profile

# ─── Colour Palette ───────────────────────────────────────────────────────────
P: Dict[str, str] = {
    "bg":       "#f1f5f9",
    "hdr":      "#0f172a",
    "side":     "#1e293b",
    "card":     "#ffffff",
    "log_bg":   "#0d1117",
    "border":   "#e2e8f0",
    "hdr_fg":   "#f8fafc",
    "side_fg":  "#cbd5e1",
    "fg":       "#1e293b",
    "muted":    "#64748b",
    "blue":     "#3b82f6",
    "blue_dk":  "#1d4ed8",
    "green":    "#22c55e",
    "green_dk": "#15803d",
    "amber":    "#f59e0b",
    "red":      "#ef4444",
    "red_dk":   "#b91c1c",
    "purple":   "#a855f7",
    "cyan":     "#06b6d4",
    "teal":     "#14b8a6",
    "orange":   "#f97316",
    "grid":     "#e2e8f0",
}

CHART_PALETTE = [
    "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7",
    "#06b6d4", "#f97316", "#ec4899", "#84cc16", "#14b8a6",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _card_shadow(canvas: tk.Canvas, x0, y0, x1, y1, r=8, **kw):
    """Draw a rounded rectangle on canvas."""
    pts = [
        x0+r, y0, x1-r, y0,
        x1, y0, x1, y0+r,
        x1, y1-r, x1, y1,
        x1-r, y1, x0+r, y1,
        x0, y1, x0, y1-r,
        x0, y0+r, x0, y0, x0+r, y0,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


def _darken(color: str, factor: float = 0.75) -> str:
    """Return a darker shade of a hex colour."""
    try:
        r = max(0, int(int(color[1:3], 16) * factor))
        g = max(0, int(int(color[3:5], 16) * factor))
        b = max(0, int(int(color[5:7], 16) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return color


def _tk_btn(parent, text, cmd, bg, fg="white", pad_x=12, pad_y=9):
    """
    Platform-reliable coloured button using tk.Label.
    tk.Button background is overridden by macOS Aqua; Label is not.
    Stores ._cmd and ._bg for dynamic updates (e.g. Start → Pause).
    """
    lbl = tk.Label(
        parent, text=text, bg=bg, fg=fg,
        font=("Arial", 10, "bold"),
        padx=pad_x, pady=pad_y, cursor="hand2",
        relief="flat", anchor="center",
    )
    lbl._cmd = cmd
    lbl._bg  = bg
    lbl.bind("<Button-1>", lambda _e: lbl._cmd())
    lbl.bind("<Enter>",    lambda _e: lbl.config(bg=_darken(lbl._bg)))
    lbl.bind("<Leave>",    lambda _e: lbl.config(bg=lbl._bg))
    return lbl


# ─── StatCard ─────────────────────────────────────────────────────────────────
class StatCard(tk.Frame):
    """Metric tile with a left accent bar, large value text, and a sub-label."""

    def __init__(self, parent, title: str, accent: str = P["blue"]):
        super().__init__(
            parent, bg=P["card"],
            highlightthickness=1, highlightbackground=P["border"],
        )
        accent_bar = tk.Frame(self, bg=accent, width=5)
        accent_bar.pack(side="left", fill="y")

        body = tk.Frame(self, bg=P["card"], padx=12, pady=10)
        body.pack(side="left", fill="both", expand=True)

        tk.Label(body, text=title.upper(), font=("Arial", 8, "bold"),
                 bg=P["card"], fg=P["muted"]).pack(anchor="w")
        self._big = tk.Label(body, text="—", font=("Arial", 21, "bold"),
                              bg=P["card"], fg=P["fg"])
        self._big.pack(anchor="w", pady=(3, 1))
        self._sub = tk.Label(body, text="", font=("Arial", 9),
                              bg=P["card"], fg=P["muted"])
        self._sub.pack(anchor="w")

    def set(self, big: str, sub: str = ""):
        self._big.config(text=big)
        self._sub.config(text=sub)


# ─── LineChart ────────────────────────────────────────────────────────────────
class LineChart(tk.Canvas):
    """Real-time multi-series line chart with automatic Y-axis scaling."""

    MAX_PTS = 400

    def __init__(
        self,
        parent,
        title: str,
        series: List[Tuple[str, str]],  # [(label, colour), ...]
        y_pct: bool = False,
        **kw,
    ):
        kw.setdefault("bg", P["card"])
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", P["border"])
        super().__init__(parent, **kw)
        self.title = title
        self.series = series
        self.y_pct = y_pct  # if True, Y ticks shown as percentages
        self._times: deque = deque(maxlen=self.MAX_PTS)
        self._data: List[deque] = [deque(maxlen=self.MAX_PTS) for _ in series]
        self._legend_font = tkfont.Font(font=("Arial", 8))
        self.bind("<Configure>", lambda e: self._draw())

    def push(self, t: float, *values: float):
        self._times.append(t)
        for buf, v in zip(self._data, values):
            buf.append(v)
        self._draw()

    def clear(self):
        self._times.clear()
        for d in self._data:
            d.clear()
        self._draw()

    def _draw(self):
        self.delete("all")
        W = self.winfo_width() or int(self.cget("width") or 400)
        H = self.winfo_height() or int(self.cget("height") or 200)
        PL, PR, PT, PB = 56, 16, 30, 32

        # Title
        self.create_text(PL + 4, 12, anchor="w", text=self.title,
                          font=("Arial", 10, "bold"), fill=P["fg"])

        # Legend (top right) -- text right-anchored to end exactly at the
        # canvas's own right margin, with the swatch immediately to its
        # left, so a longer label (e.g. "1.0 target") pushes the whole
        # entry further left instead of overflowing past the canvas edge
        # the way a fixed left-anchor a few px from that edge did (it
        # rendered fine for "PDR" but silently ran off-canvas, clipped by
        # the canvas boundary itself, for anything much wider).
        lx = W - PR
        for i, (name, color) in enumerate(self.series):
            ly = PT - 12 + i * 14
            self.create_text(lx, ly + 4, anchor="e",
                              text=name, font=("Arial", 8), fill=P["muted"])
            text_w = self._legend_font.measure(name)
            swatch_x1 = lx - text_w - 4
            self.create_rectangle(swatch_x1 - 12, ly, swatch_x1, ly + 8,
                                   fill=color, outline=color)

        x0, x1 = PL, W - PR
        y0, y1 = PT, H - PB
        cw = max(1, x1 - x0)
        ch = max(1, y1 - y0)

        # Compute dynamic y_max
        all_vals = [v for buf in self._data for v in buf]
        y_max = max(all_vals) if all_vals else (1.0 if self.y_pct else 100.0)
        if self.y_pct:
            y_max = 1.0

        # Grid + Y labels
        Y_TICKS = 4
        for i in range(Y_TICKS + 1):
            frac = i / Y_TICKS
            yp = y1 - frac * ch
            val = frac * y_max
            lbl = f"{val:.0%}" if self.y_pct else f"{val:.0f}"
            self.create_line(x0 - 4, yp, x0, yp, fill=P["muted"])
            self.create_text(x0 - 6, yp, anchor="e", text=lbl,
                              font=("Arial", 8), fill=P["muted"])
            if i > 0:
                self.create_line(x0, yp, x1, yp, fill=P["grid"], dash=(3, 5))

        # Axes
        self.create_line(x0, y0, x0, y1, fill=P["muted"], width=1)
        self.create_line(x0, y1, x1, y1, fill=P["muted"], width=1)

        if not self._times:
            self.create_text(W // 2, H // 2, text="No data yet",
                              font=("Arial", 10), fill=P["muted"])
            return

        n = len(self._times)
        t_min, t_max = self._times[0], max(self._times[-1], self._times[0] + 1e-9)

        # X labels
        for i in range(5):
            frac = i / 4
            tv = t_min + frac * (t_max - t_min)
            xp = x0 + frac * cw
            self.create_line(xp, y1, xp, y1 + 3, fill=P["muted"])
            self.create_text(xp, y1 + 8, anchor="n",
                              text=f"{tv:.0f}s", font=("Arial", 8), fill=P["muted"])

        # Series lines
        for (_, color), buf in zip(self.series, self._data):
            if len(buf) < 2:
                continue
            pts: List[float] = []
            for tv, v in zip(self._times, buf):
                xp = x0 + ((tv - t_min) / (t_max - t_min)) * cw
                yp = y1 - (min(v, y_max) / max(y_max, 1e-12)) * ch
                pts.extend([xp, yp])
            if len(pts) >= 4:
                self.create_line(*pts, fill=color, width=2, smooth=True)


# ─── BarChart ─────────────────────────────────────────────────────────────────
class BarChart(tk.Canvas):
    """Horizontally scrollable node-wise bar chart."""

    def __init__(
        self,
        parent,
        title: str,
        bar_color: Optional[str] = None,
        **kw,
    ):
        kw.setdefault("bg", P["card"])
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", P["border"])
        super().__init__(parent, **kw)
        self.title = title
        self.bar_color = bar_color
        self._last: Dict = {}
        self.configure(xscrollincrement=1)
        # Unconditional, matching LineChart's own Configure handler --
        # `self._last and self.draw(...)` looked like a harmless "only
        # redraw if there's something to redraw" guard, but self._last
        # starts as {} (falsy), so before the very first real .draw() call
        # from application code (apply_settings()/_refresh(), both only
        # reachable after Start/Apply & Rebuild), every Configure event
        # was silently skipped -- this canvas showed nothing at all, not
        # even its own title or the "No data yet" placeholder draw()
        # already knows how to render for empty data, until real per-node
        # metrics existed. A user opening Node Analytics before ever
        # starting the sim saw blank white boxes with zero indication of
        # what they even were.
        self.bind("<Configure>", lambda e: self.draw(self._last))

    def draw(self, data: Dict):
        self._last = data
        self.delete("all")
        W = self.winfo_width() or int(self.cget("width") or 500)
        H = self.winfo_height() or int(self.cget("height") or 200)
        PL, PR, PT, PB = 52, 14, 30, 30

        self.create_text(PL + 4, 12, anchor="w", text=self.title,
                          font=("Arial", 10, "bold"), fill=P["fg"])

        if not data:
            self.config(scrollregion=(0, 0, W, H))
            self.create_text(W // 2, H // 2, text="No data yet",
                              font=("Arial", 10), fill=P["muted"])
            return

        items = sorted(data.items())
        n = len(items)
        bar_w, gap = 18, 6
        total_w = max(W, PL + PR + n * (bar_w + gap) + gap)
        self.config(scrollregion=(0, 0, total_w, H))

        xa, xb = PL, total_w - PR
        ya, yb = PT, H - PB
        ch = max(1, yb - ya)
        max_v = max((v for _, v in items), default=1) or 1

        # Y grid
        for i in range(1, 5):
            frac = i / 4
            yp = yb - frac * ch
            val = frac * max_v
            lbl = str(int(val)) if max_v > 1 else f"{val:.2f}"
            self.create_line(xa - 4, yp, xa, yp, fill=P["muted"])
            self.create_text(xa - 6, yp, anchor="e", text=lbl,
                              font=("Arial", 8), fill=P["muted"])
            self.create_line(xa, yp, xb, yp, fill=P["grid"], dash=(3, 5))

        # Axes
        self.create_line(xa, ya, xa, yb, fill=P["muted"])
        self.create_line(xa, yb, xb, yb, fill=P["muted"])

        sv = max(1, math.ceil(n / 20))
        sx = max(1, math.ceil(n / 30))

        for idx, (nid, val) in enumerate(items):
            bx0 = xa + gap + idx * (bar_w + gap)
            bx1 = bx0 + bar_w
            bh = (val / max_v) * ch
            by0 = yb - bh
            color = self.bar_color or CHART_PALETTE[idx % len(CHART_PALETTE)]
            self.create_rectangle(bx0, by0, bx1, yb, fill=color, outline=color)
            if n <= 20 or idx % sv == 0:
                self.create_text(
                    (bx0 + bx1) / 2, max(ya - 4, by0 - 10),
                    text=str(val), font=("Arial", 7, "bold"), fill=P["fg"],
                )
            if n <= 25 or idx % sx == 0:
                self.create_text(
                    (bx0 + bx1) / 2, yb + 10,
                    text=str(nid), font=("Arial", 7), fill=P["muted"],
                )


# The single flat topology.mode string ("star" / "relay" / "aerial_relay" /
# "uav" / "relay_uav" / "aerial_relay_uav") every other module (net.py's
# routing, topology_canvas.py's seeding/mobility/rendering, the web3d
# snapshot) already keys off is still the one source of truth passed
# around internally -- these two functions are only the GUI-facing
# translation to/from the three separate, orthogonal controls a user
# actually thinks in: Network Topology (star/relay), STA Type (ground/UAV,
# the end nodes), and Relay Type (grounded/aerial, only meaningful when
# Network Topology is "relay"). Keeping that flat string as the only thing
# every non-GUI module reads means this three-control UI is purely
# additive -- nothing downstream needed to change to support it.
def _decompose_topology_mode(mode: str):
    """Inverse of _compose_topology_mode: flat mode string -> (network
    topology, sta_is_uav, relay_is_aerial)."""
    net_topo = "star" if mode in ("star", "uav") else "relay"
    sta_uav = mode in ("uav", "aerial_relay_uav", "relay_uav")
    relay_aerial = mode in ("aerial_relay", "aerial_relay_uav")
    return net_topo, sta_uav, relay_aerial


def _compose_topology_mode(net_topo: str, sta_uav: bool, relay_aerial: bool) -> str:
    """Inverse of _decompose_topology_mode. When net_topo is "star" there
    are no relays to be grounded/aerial at all, so relay_aerial is simply
    ignored in that branch -- same as the old single-dropdown "star"
    always did."""
    if net_topo == "star":
        return "uav" if sta_uav else "star"
    if relay_aerial and sta_uav:
        return "aerial_relay_uav"
    if relay_aerial:
        return "aerial_relay"
    if sta_uav:
        return "relay_uav"
    return "relay"


# ─── Dashboard ────────────────────────────────────────────────────────────────
class Dashboard(tk.Tk):

    # Simulation state constants
    IDLE    = "idle"
    RUNNING = "running"
    PAUSED  = "paused"

    _SPEEDS = {
        "Slow  (4 fps)":   250,
        "Normal (7 fps)":  145,
        "Fast  (15 fps)":   67,
        "Turbo (30 fps)":   33,
    }

    _LAYOUT_NAMES = {
        "Open Area":       ["1: Default"],
        "Paddy Field":     ["1: Delta Plains", "2: River Valley", "3: Highland Terraces"],
        "Industrial Site": ["1: Logistics Park", "2: Process Plant", "3: Business Park"],
        "Smart City":      ["1: Downtown Grid", "2: Business District", "3: Suburban Corridor",
                            "4: Real Map (IIT ISM Campus)"],
        "Military Zone":   ["1: Forward Operating Base"],
    }

    # NetworkCanvas.set_layout_variant() only accepts 1-3 (silently ignores
    # anything else, leaving whichever procedural variant was last active
    # underneath) -- variant 4 isn't a procedural layout at all, it's the
    # signal to open the real-map (MapLibre + live AP/STA/UAV) digital twin
    # in the browser instead, so it's special-cased in _on_layout_change.
    _REAL_MAP_LAYOUT = "4: Real Map (IIT ISM Campus)"

    def __init__(self, sim, sim_builder=None, initial_settings=None):
        super().__init__()
        self.sim = sim
        self.sim_builder = sim_builder
        self._state = self.IDLE
        self.running = False       # back-compat alias (True only when RUNNING)
        self.update_ms = 145
        self.step_dt = 0.2
        self._tput_max = 1.0
        self._pkt_log_ptr = 0
        self._trace_log_ptr = 0
        self._init_metrics_state()
        self._web3d_server = None

        self.title("sim11ah — IEEE 802.11ah Wi-Fi HaLow Simulator")
        self.geometry("1480x920")
        self.minsize(1100, 720)
        self.configure(bg=P["bg"])
        self._setup_styles()

        s = initial_settings or {}
        self._vars: Dict[str, tk.Variable] = {
            "num_stas":        tk.IntVar(value=int(s.get("num_stas", 50))),
            "seed":            tk.IntVar(value=int(s.get("seed", 0))),
            "traffic":         tk.StringVar(value=str(s.get("traffic", "periodic"))),
            "raw_enable":      tk.StringVar(
                                value="Enabled" if s.get("raw_enable", True) else "Disabled"),
            "raw_policy":      tk.StringVar(value=str(s.get("raw_policy", "static"))),
            "raw_groups":      tk.IntVar(value=int(s.get("raw_groups", 4))),
            # 8, not 4: matches scripts/compare_raw_policies.py's validated
            # benchmark config. The old default of 4 silently doubled
            # per-slot contention density versus that benchmark (adaptive
            # only ever tunes slot *duration*, never slot *count* -- see
            # sim11ah/mac/raw_policy_adaptive.py), which alone accounted for
            # most of a ~0.55 PDR gap between GUI and script runs at N=200.
            "raw_slots":       tk.IntVar(value=int(s.get("raw_slots", 8))),
            "raw_slot_duration_ms": tk.DoubleVar(value=float(s.get("raw_slot_duration_ms", 14.0))),
            "packet_size":     tk.IntVar(value=int(s.get("packet_size", 128))),
            "packet_interval": tk.DoubleVar(value=float(s.get("packet_interval", 5.0))),
            "topology":        tk.StringVar(value=str(s.get("topology", "star"))),
            "num_relays":      tk.IntVar(value=int(s.get("num_relays", 2))),
            # "optimal": relays spread evenly around the AP (360/N apart,
            # one per angular sector) so their combined coverage spans the
            # whole region instead of leaving gaps -- "random" keeps the
            # old independent-per-relay scatter, still useful for stress-
            # testing uneven/clustered layouts on purpose.
            "relay_placement": tk.StringVar(value=str(s.get("relay_placement", "optimal"))),
            "freq_mhz":        tk.DoubleVar(value=float(s.get("freq_mhz", 915.0))),
            "sim_speed":       tk.StringVar(value="Normal (7 fps)"),
            "log_filter":      tk.StringVar(value="ALL"),
            "log_autoscroll":  tk.BooleanVar(value=True),
            "trace_filter":    tk.StringVar(value="ALL"),
            "trace_autoscroll": tk.BooleanVar(value=True),
            "trace_node":      tk.StringVar(value=""),
            "trace_search":    tk.StringVar(value=""),
            "sensor_profile":  tk.StringVar(value="(none)"),
            "video_fps":       tk.DoubleVar(value=float(s.get("video_fps", 5.0))),
        }
        self._applied = self._sig()
        _skip_trace = {"sim_speed", "log_filter", "log_autoscroll",
                       "trace_filter", "trace_autoscroll", "trace_node", "trace_search"}
        for _k, _v in self._vars.items():
            if _k not in _skip_trace:
                _v.trace_add("write", self._on_settings_change)

        # GUI-facing decomposition of _vars["topology"] into the three
        # controls a user actually thinks in -- see _decompose_topology_mode
        # / _compose_topology_mode above. Not in self._vars: they only ever
        # feed the real "topology" var (which already has its own trace
        # above), never read independently by anything downstream.
        _net_topo0, _sta_uav0, _relay_aerial0 = _decompose_topology_mode(
            self._vars["topology"].get())
        self._net_topo_var = tk.StringVar(value=_net_topo0)
        self._sta_type_var = tk.StringVar(value="UAV" if _sta_uav0 else "Ground STA")
        self._relay_type_var = tk.StringVar(value="Aerial (UAV)" if _relay_aerial0 else "Grounded")

        self._build_header()
        self._build_main_area()
        self._build_statusbar()

        self._refresh_topology()
        self.after(150, self._refresh_topology)   # re-draw after canvas renders
        self.after(self.update_ms, self._tick)

    # ── Styles ────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TButton", font=("Arial", 10), padding=8)
        s.configure("TCombobox", padding=5, font=("Arial", 10))
        s.configure("TCheckbutton", background=P["bg"], font=("Arial", 10))
        s.configure("TNotebook", background=P["bg"], borderwidth=0,
                    tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab", font=("Arial", 10), padding=(14, 7),
                    background=P["border"], foreground=P["muted"])
        s.map("TNotebook.Tab",
              background=[("selected", P["bg"])],
              foreground=[("selected", P["blue"])])
        s.configure("Horizontal.TProgressbar",
                    troughcolor=P["border"], background=P["blue"])
        # Dark console theme to match the Log Viewer tab's tk.Text styling
        # (P["log_bg"]) -- ttk.Treeview ignores plain bg=/fg=, needs Style.
        s.configure("Trace.Treeview", background=P["log_bg"], fieldbackground=P["log_bg"],
                    foreground="#e2e8f0", font=("Courier New", 9), rowheight=20, borderwidth=0)
        s.configure("Trace.Treeview.Heading", background=P["side"], foreground="#cbd5e1",
                    font=("Arial", 9, "bold"), relief="flat")
        s.map("Trace.Treeview", background=[("selected", P["blue_dk"])])

    # ── Header bar ────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=P["hdr"], height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Left: logo + subtitle
        tk.Label(hdr, text=" sim11ah", font=("Arial", 15, "bold"),
                 bg=P["hdr"], fg="#f8fafc").pack(side="left", padx=(14, 0))
        tk.Label(hdr, text="IEEE 802.11ah Wi-Fi HaLow Simulator",
                 font=("Arial", 9), bg=P["hdr"], fg="#64748b").pack(side="left", padx=6)

        # Right: sim clock + status badge
        right = tk.Frame(hdr, bg=P["hdr"])
        right.pack(side="right", padx=16)

        self._hdr_badge = tk.Label(right, text="● STOPPED",
                                    font=("Arial", 10, "bold"),
                                    bg=P["hdr"], fg=P["red"])
        self._hdr_badge.pack(side="right", padx=(16, 0))

        self._hdr_time = tk.Label(right, text="T = 0.000 s",
                                   font=("Courier New", 12, "bold"),
                                   bg=P["hdr"], fg="#e2e8f0")
        self._hdr_time.pack(side="right")

    # ── Main layout: sidebar + notebook ───────────────────────────────────────
    def _build_main_area(self):
        pane = tk.Frame(self, bg=P["bg"])
        pane.pack(fill="both", expand=True)

        # Sidebar (right)
        sidebar = tk.Frame(pane, bg=P["side"], width=248)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Notebook (left)
        nb = ttk.Notebook(pane)
        nb.pack(side="left", fill="both", expand=True)
        self._notebook = nb
        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._tab_overview = tk.Frame(nb, bg=P["bg"])
        self._tab_nodes    = tk.Frame(nb, bg=P["bg"])
        self._tab_log      = tk.Frame(nb, bg=P["bg"])
        self._tab_trace    = tk.Frame(nb, bg=P["bg"])
        self._tab_settings = tk.Frame(nb, bg=P["bg"])
        self._tab_topology = tk.Frame(nb, bg=P["bg"])

        nb.add(self._tab_overview, text="  Overview  ")
        nb.add(self._tab_nodes,    text="  Node Analytics  ")
        nb.add(self._tab_log,      text="  Log Viewer  ")
        nb.add(self._tab_trace,    text="  Packet Trace  ")
        nb.add(self._tab_settings, text="  Settings  ")
        nb.add(self._tab_topology, text="  Topology  ")

        self._build_overview_tab(self._tab_overview)
        self._build_nodes_tab(self._tab_nodes)
        self._build_log_tab(self._tab_log)
        self._build_trace_tab(self._tab_trace)
        self._build_settings_tab(self._tab_settings)
        self._build_topology_tab(self._tab_topology)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        def section(text):
            tk.Label(parent, text=text.upper(), font=("Arial", 8, "bold"),
                     bg=P["side"], fg="#475569").pack(anchor="w", padx=14,
                                                       pady=(14, 4))

        def divider():
            tk.Frame(parent, bg="#334155", height=1).pack(fill="x",
                                                           padx=12, pady=6)

        # ── Mini logo ─────────────────────────────────────────────────────────
        logo = tk.Frame(parent, bg="#0f172a", pady=10)
        logo.pack(fill="x")
        tk.Label(logo, text="sim11ah", font=("Arial", 13, "bold"),
                 bg="#0f172a", fg="#f8fafc").pack()
        tk.Label(logo, text="802.11ah · Wi-Fi HaLow",
                 font=("Arial", 8), bg="#0f172a", fg="#64748b").pack()

        # ── Run Controls ──────────────────────────────────────────────────────
        run_panel = tk.Frame(parent, bg="#0f172a", pady=8, padx=8)
        run_panel.pack(fill="x", pady=(0, 4))
        tk.Label(run_panel, text="RUN CONTROLS", font=("Arial", 8, "bold"),
                 bg="#0f172a", fg="#475569").pack(anchor="w", padx=6, pady=(0, 6))
        self._btn_run = _tk_btn(run_panel, "▶  Start", self.start, bg=P["green"])
        self._btn_run.pack(fill="x", padx=6, pady=(0, 4))

        ctrl_row = tk.Frame(run_panel, bg="#0f172a")
        ctrl_row.pack(fill="x", padx=6, pady=(0, 2))
        _tk_btn(ctrl_row, "■  Stop & Reset", self.stop_reset,
                bg=P["red"]).pack(side="left", expand=True, fill="x", padx=(0, 3))
        _tk_btn(ctrl_row, "⏭ Step", self.step_once,
                bg=P["blue"], fg="white").pack(side="left", expand=True, fill="x",
                                               padx=(3, 0))

        divider()
        section("Simulation Speed")
        speed_cb = ttk.Combobox(parent, textvariable=self._vars["sim_speed"],
                                 values=list(self._SPEEDS), state="readonly")
        speed_cb.pack(fill="x", padx=10, pady=2)
        speed_cb.bind("<<ComboboxSelected>>", self._on_speed_change)

        divider()
        section("Settings")
        self._pending_lbl = tk.Label(
            parent, text="— No pending changes —",
            font=("Arial", 9, "italic"), bg=P["side"], fg="#475569",
            wraplength=210, anchor="w",
        )
        self._pending_lbl.pack(anchor="w", padx=14, pady=(0, 4))
        self._btn_apply_sidebar = _tk_btn(
            parent, "  Apply & Rebuild  ", self.apply_settings,
            bg="#334155", fg="#94a3b8",
        )
        self._btn_apply_sidebar.pack(fill="x", padx=10, pady=(0, 4))

        divider()
        section("Export & Results")
        _tk_btn(parent, "⬇  Export Logs CSV", self.export_csv,
                bg=P["teal"], fg="white").pack(fill="x", padx=10, pady=2)
        _tk_btn(parent, "📊  View Results", self.run_results,
                bg=P["purple"], fg="white").pack(fill="x", padx=10, pady=2)

        divider()
        section("Live Metrics")

        self._sl_time  = self._side_stat(parent, "Sim Time",   "—")
        self._sl_pdr   = self._side_stat(parent, "PDR",        "—")
        self._sl_tput  = self._side_stat(parent, "Throughput", "—")
        self._sl_delay = self._side_stat(parent, "Avg Delay",  "—")
        self._sl_drop  = self._side_stat(parent, "Drop Rate",  "—")
        self._sl_logs  = self._side_stat(parent, "Log Entries","0")

        divider()
        self._badge = tk.Label(parent, text="● Ready",
                                font=("Arial", 10, "bold"), bg=P["side"],
                                fg=P["amber"], anchor="w")
        self._badge.pack(anchor="w", padx=14, pady=(6, 2))
        self._badge_detail = tk.Label(
            parent, text="Press Start to begin.",
            font=("Arial", 9), bg=P["side"], fg="#64748b",
            wraplength=210, justify="left", anchor="w",
        )
        self._badge_detail.pack(anchor="w", padx=14, pady=(0, 8))

    def _side_stat(self, parent, name, init):
        row = tk.Frame(parent, bg=P["side"])
        row.pack(fill="x", padx=14, pady=1)
        tk.Label(row, text=f"{name}:", font=("Arial", 9),
                 bg=P["side"], fg="#475569", width=13, anchor="w").pack(side="left")
        lbl = tk.Label(row, text=init, font=("Arial", 9, "bold"),
                       bg=P["side"], fg=P["side_fg"], anchor="w")
        lbl.pack(side="left")
        return lbl

    # ── Tab: Overview ─────────────────────────────────────────────────────────
    def _build_overview_tab(self, parent):
        # Stat cards row
        card_row = tk.Frame(parent, bg=P["bg"])
        card_row.pack(fill="x", padx=12, pady=12)

        self._c_pdr   = StatCard(card_row, "Packet Delivery Ratio", P["blue"])
        self._c_tput  = StatCard(card_row, "Throughput",            P["green"])
        self._c_delay = StatCard(card_row, "Avg E2E Delay",         P["amber"])
        self._c_gen   = StatCard(card_row, "Generated",             P["purple"])
        self._c_del   = StatCard(card_row, "Delivered",             P["cyan"])
        self._c_drop  = StatCard(card_row, "Drop Rate",             P["red"])

        for c in (self._c_pdr, self._c_tput, self._c_delay,
                  self._c_gen, self._c_del, self._c_drop):
            c.pack(side="left", expand=True, fill="x", padx=4)

        # Charts + topology mini-view
        body = tk.Frame(parent, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left_charts = tk.Frame(body, bg=P["bg"])
        left_charts.pack(side="left", fill="both", expand=True)

        self._lc_pdr = LineChart(
            left_charts, "PDR Over Time",
            [("PDR", P["blue"]), ("1.0 target", "#d1fae5")],
            y_pct=True, height=190,
        )
        self._lc_pdr.pack(fill="x", pady=(0, 8))

        self._lc_tput = LineChart(
            left_charts, "Throughput Over Time (kb/s)",
            [("Tput", P["amber"])],
            y_pct=False, height=190,
        )
        self._lc_tput.pack(fill="x", pady=(0, 8))

        self._lc_delay = LineChart(
            left_charts, "Avg E2E Delay Over Time (ms)",
            [("Delay", P["purple"])],
            y_pct=False, height=190,
        )
        self._lc_delay.pack(fill="x", pady=(0, 8))

        self._lc_drop = LineChart(
            left_charts, "Drop Rate Over Time",
            [("Drop Rate", P["red"])],
            y_pct=True, height=190,
        )
        self._lc_drop.pack(fill="x")

        # Mini topology view
        right_topo = tk.Frame(body, bg=P["bg"])
        right_topo.pack(side="right", fill="y", padx=(10, 0))

        self._topo_mini = NetworkCanvas(right_topo, width=270, height=408)
        self._topo_mini.pack(fill="both", expand=True)

    # ── Tab: Node Analytics ───────────────────────────────────────────────────
    def _build_nodes_tab(self, parent):
        def chart_section(title, color, attr):
            wrap = tk.Frame(parent, bg=P["bg"])
            wrap.pack(fill="both", expand=True, padx=12, pady=(8, 4))
            bc = BarChart(wrap, title, bar_color=color, height=185)
            bc.pack(fill="x", side="top")
            sb = tk.Scrollbar(wrap, orient="horizontal", command=bc.xview)
            sb.pack(fill="x", side="bottom")
            bc.configure(xscrollcommand=sb.set)
            setattr(self, attr, bc)

        chart_section("Packets Generated per Node", P["blue"],  "_bc_gen")
        chart_section("Packets Delivered per Node", P["green"], "_bc_del")
        chart_section("Per-Node PDR  (×100 = %)",  P["purple"],"_bc_pdr")

    # ── Tab: Log Viewer ───────────────────────────────────────────────────────
    def _build_log_tab(self, parent):
        toolbar = tk.Frame(parent, bg=P["bg"], pady=6)
        toolbar.pack(fill="x", padx=12)

        tk.Label(toolbar, text="Layer filter:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10, "bold")).pack(side="left")

        ttk.Combobox(
            toolbar, textvariable=self._vars["log_filter"],
            values=["ALL", "APP", "MAC", "PHY", "NET", "TP"],
            width=8, state="readonly",
        ).pack(side="left", padx=(4, 12))

        ttk.Checkbutton(
            toolbar, text="Auto-scroll",
            variable=self._vars["log_autoscroll"],
        ).pack(side="left")

        ttk.Button(toolbar, text="Clear Log", command=self._clear_log
                   ).pack(side="right")

        # Log text widget
        wrap = tk.Frame(parent, bg=P["bg"])
        wrap.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        self._log = tk.Text(
            wrap, bg=P["log_bg"], fg="#e2e8f0", insertbackground="white",
            relief="flat", font=("Courier New", 10), wrap="none",
        )
        self._log.pack(side="left", fill="both", expand=True)

        sy = tk.Scrollbar(wrap, orient="vertical", command=self._log.yview)
        sy.pack(side="right", fill="y")
        self._log.configure(yscrollcommand=sy.set)

        sx = tk.Scrollbar(parent, orient="horizontal", command=self._log.xview)
        sx.pack(fill="x", padx=12, pady=(0, 8))
        self._log.configure(xscrollcommand=sx.set)

        # Colour tags per layer
        tag_colours = {
            "APP":  "#86efac",  # green
            "MAC":  "#67e8f9",  # cyan
            "PHY":  "#fde68a",  # yellow
            "NET":  "#c4b5fd",  # purple
            "TP":   "#fda4af",  # pink/rose
            "WARN": "#fef08a",
            "ERR":  "#fca5a5",
        }
        for tag, fg in tag_colours.items():
            self._log.tag_configure(tag, foreground=fg)

        self._log.insert("end", "Ready. Press Start to begin simulation.\n")

    # ── Tab: Settings ─────────────────────────────────────────────────────────
    def _build_settings_tab(self, parent):
        canvas = tk.Canvas(parent, bg=P["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        inner = tk.Frame(canvas, bg=P["bg"])
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))

        # Mouse-wheel scrolling
        def _wheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _wheel)

        def section(title):
            tk.Frame(inner, bg=P["border"], height=1).pack(
                fill="x", padx=20, pady=(18, 10))
            f = tk.Frame(inner, bg=P["bg"])
            f.pack(fill="x", padx=20, pady=(0, 6))
            tk.Label(f, text=title, font=("Arial", 11, "bold"),
                     bg=P["bg"], fg=P["fg"]).pack(side="left")

        def row(label, values, var, extras=None):
            f = tk.Frame(inner, bg=P["bg"])
            f.pack(fill="x", padx=20, pady=3)
            tk.Label(f, text=label, font=("Arial", 10), bg=P["bg"],
                     fg=P["fg"], width=26, anchor="w").pack(side="left")
            cb = ttk.Combobox(f, textvariable=var, values=values,
                               state="normal", width=22)
            cb.pack(side="left", padx=(8, 0))
            if extras:
                extras(f)

        # ── General ───────────────────────────────────────────────────────────
        section("General")
        row("Number of STAs",
            [5, 10, 20, 50, 100, 150, 200, 300, 500],
            self._vars["num_stas"])
        row("Random Seed",
            [0, 1, 2, 3, 4, 5, 10, 20, 42, 99, 100],
            self._vars["seed"])
        row("Traffic Model",
            ["periodic", "poisson", "cbr", "bursty", "onoff", "video"],
            self._vars["traffic"])
        row("Video FPS  (Traffic Model = video)",
            [1, 2, 5, 10, 15, 30],
            self._vars["video_fps"])

        # Sensor Profile: a preset that overrides traffic/packet-size/etc.
        # with realistic values for a named sensor type (see
        # sim11ah/sensor_profiles.py) -- offered options depend on the
        # environment picked in the Topology tab, so the list is
        # refreshed from postcommand (fires right before the dropdown
        # opens) rather than fixed at build time like the rows above.
        sp_row = tk.Frame(inner, bg=P["bg"])
        sp_row.pack(fill="x", padx=20, pady=3)
        tk.Label(sp_row, text="Sensor Profile", font=("Arial", 10), bg=P["bg"],
                 fg=P["fg"], width=26, anchor="w").pack(side="left")
        sp_cb = ttk.Combobox(sp_row, textvariable=self._vars["sensor_profile"],
                              state="readonly", width=22)

        def _refresh_sensor_profile_choices():
            env = self._env_var.get() if hasattr(self, "_env_var") else "Open Area"
            sp_cb["values"] = ["(none)"] + _sensor_list_profiles(env)

        sp_cb.configure(postcommand=_refresh_sensor_profile_choices)
        _refresh_sensor_profile_choices()
        sp_cb.pack(side="left", padx=(8, 0))
        tk.Label(sp_row, text="  overrides traffic/packet size with a realistic preset",
                 font=("Arial", 9), bg=P["bg"], fg=P["muted"]).pack(side="left", padx=(8, 0))

        row("Packet Size (bytes)",
            [32, 64, 128, 256, 512, 1024],
            self._vars["packet_size"])
        row("Packet Interval (s)",
            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            self._vars["packet_interval"])

        # ── RAW ───────────────────────────────────────────────────────────────
        section("Restricted Access Window (RAW)")
        row("RAW Enable",
            ["Enabled", "Disabled"],
            self._vars["raw_enable"])
        row("RAW Policy",
            ["static", "adaptive", "cluster_adaptive", "cluster_csv"],
            self._vars["raw_policy"])
        row("RAW Groups (G)",
            [1, 2, 4, 8, 16],
            self._vars["raw_groups"])
        row("RAW Sub-slots per Group (S)",
            [1, 2, 4, 8],
            self._vars["raw_slots"])
        row("RAW Slot Duration (ms)",
            [7, 10, 14, 20, 28],
            self._vars["raw_slot_duration_ms"])

        # Topology mode/count live in the Topology tab now (next to the map
        # itself), not here -- see _build_topology_tab.

        # ── PHY ───────────────────────────────────────────────────────────────
        section("PHY / Link")
        row("Carrier Frequency (MHz)",
            [433.0, 470.0, 780.0, 863.0, 868.0, 902.0, 915.0, 920.0, 928.0],
            self._vars["freq_mhz"])
        for lbl, val in [
            ("Channel Bandwidth", "1 MHz (S1G)"),
            ("Default MCS", "MCS0 — 150 kb/s, BPSK 1/2"),
        ]:
            f = tk.Frame(inner, bg=P["bg"])
            f.pack(fill="x", padx=20, pady=2)
            tk.Label(f, text=lbl, font=("Arial", 10), bg=P["bg"],
                     fg=P["fg"], width=26, anchor="w").pack(side="left")
            tk.Label(f, text=val, font=("Arial", 10), bg=P["bg"],
                     fg=P["muted"]).pack(side="left", padx=8)

        # Was a static "η=2.7" -- stale/misleading once the Topology tab's
        # Environment picker started giving each environment its own real
        # path-loss exponent (see topology_canvas.py's _ENV_PATH_LOSS_EXP);
        # every GUI-built sim overrides the plain 2.7 default the instant
        # it syncs with a canvas, so that fixed string never matched what
        # was actually running. Kept live via _update_propagation_label,
        # called from _refresh_topology (covers Apply & Rebuild/Stop &
        # Reset/Load Topology/initial build) and _on_env_change's
        # already-running branch (the one path that changes the exponent
        # without a rebuild).
        f = tk.Frame(inner, bg=P["bg"])
        f.pack(fill="x", padx=20, pady=2)
        tk.Label(f, text="Propagation Model", font=("Arial", 10), bg=P["bg"],
                 fg=P["fg"], width=26, anchor="w").pack(side="left")
        self._propagation_lbl = tk.Label(f, text="Log-distance, η=2.70", font=("Arial", 10),
                                          bg=P["bg"], fg=P["muted"])
        self._propagation_lbl.pack(side="left", padx=8)

        # ── Buttons ───────────────────────────────────────────────────────────
        tk.Frame(inner, bg=P["bg"], height=16).pack()
        btn_row = tk.Frame(inner, bg=P["bg"])
        btn_row.pack(padx=20, anchor="w", pady=(0, 24))

        _tk_btn(btn_row, "  Apply & Rebuild Simulation  ",
                self.apply_settings, bg=P["blue"]).pack(side="left", padx=(0, 8))
        _tk_btn(btn_row, "Reset Defaults",
                self._reset_defaults, bg="#334155", fg=P["side_fg"]
                ).pack(side="left")

        tk.Label(inner,
                 text="  Changes take effect after 'Apply & Rebuild'.",
                 font=("Arial", 9, "italic"), bg=P["bg"], fg=P["muted"]
                 ).pack(anchor="w", padx=20, pady=(0, 20))

    # ── Tab: Topology ─────────────────────────────────────────────────────────
    def _build_topology_tab(self, parent):
        # Every functional row below follows the same two-line pattern:
        # controls on their own row (nothing else competing for width), any
        # explanatory text on a second, plain anchor="w" row underneath --
        # packing long italic help text into the same row as the controls
        # it describes routinely overflowed a normal window width and
        # visually collided with (or clipped) the controls themselves.
        # Targets settings_panel (defined below), not parent -- every row
        # using this lives inside the collapsible section.
        def _help_row(text):
            tk.Label(
                settings_panel, text=text, font=("Arial", 8, "italic"),
                bg=P["bg"], fg=P["muted"], anchor="w", justify="left",
            ).pack(fill="x", padx=2, pady=(0, 6))

        toolbar = tk.Frame(parent, bg=P["bg"])
        toolbar.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(toolbar, text="Network Topology — drag nodes to reposition them",
                 font=("Arial", 11, "bold"), bg=P["bg"], fg=P["fg"]
                 ).pack(side="left")
        ttk.Button(toolbar, text="Reset Layout", command=self._reset_topo_layout
                   ).pack(side="right")
        _tk_btn(toolbar, "  🌐 Open 3D View  ", self._open_3d_view,
                bg=P["green"]).pack(side="right", padx=(0, 10))

        # View controls (map style / layout variant / zoom) get their own
        # row, in natural left-to-right reading order -- crammed onto the
        # title toolbar alongside "Open 3D View"/"Reset Layout" they
        # routinely overflowed a normal window width.
        view_row = tk.Frame(parent, bg=P["bg"])
        view_row.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(view_row, text="Map style:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left", padx=(0, 4))
        self._env_var = tk.StringVar(value="Open Area")
        env_cb = ttk.Combobox(
            view_row, textvariable=self._env_var, values=list(NetworkCanvas.ENVIRONMENTS),
            state="readonly", width=15,
        )
        env_cb.pack(side="left", padx=(0, 16))
        env_cb.bind("<<ComboboxSelected>>", self._on_env_change)

        tk.Label(view_row, text="Layout:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left", padx=(0, 4))
        self._layout_var = tk.StringVar(value=self._LAYOUT_NAMES["Open Area"][0])
        self._layout_cb = ttk.Combobox(
            view_row, textvariable=self._layout_var,
            values=self._LAYOUT_NAMES["Open Area"], state="readonly", width=20,
        )
        self._layout_cb.pack(side="left", padx=(0, 16))
        self._layout_cb.bind("<<ComboboxSelected>>", self._on_layout_change)

        tk.Label(view_row, text="Zoom:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left", padx=(0, 4))
        ttk.Button(view_row, text="−", width=3,
                   command=lambda: self._net_canvas.zoom_out()).pack(side="left")
        ttk.Button(view_row, text="+", width=3,
                   command=lambda: self._net_canvas.zoom_in()).pack(side="left")
        ttk.Button(view_row, text="Fit", width=4,
                   command=lambda: self._net_canvas.zoom_reset()).pack(side="left", padx=(4, 0))

        # Everything below (topology structure, relay path, save/load) is
        # once-per-setup configuration, not something worth staring at
        # while a sim runs -- collapsed by default so the actual map (the
        # thing you ARE looking at while it runs) gets the bulk of the
        # tab's vertical space instead of always competing with nine rows
        # of settings above it. Map style/Layout/Zoom above stay always
        # visible since those are worth changing mid-run. body (the map +
        # node panel) is packed once, immediately, right after this toggle
        # -- settings_panel is only ever packed/unpacked relative to it via
        # before=self._topo_body, so toggling can never reorder them.
        toggle_row = tk.Frame(parent, bg=P["bg"])
        toggle_row.pack(fill="x", padx=12, pady=(0, 4))
        self._topo_settings_open = False
        self._topo_settings_toggle = tk.Label(
            toggle_row, text="▶  Topology Settings  (structure, relay path, save/load)",
            font=("Arial", 9, "bold"), bg=P["bg"], fg=P["blue_dk"], cursor="hand2",
        )
        self._topo_settings_toggle.pack(side="left")
        self._topo_settings_toggle.bind("<Button-1>", lambda _e: self._toggle_topology_settings())

        settings_panel = tk.Frame(parent, bg=P["bg"])
        self._topo_settings_panel = settings_panel

        # Topology structure controls live here, next to the map they
        # affect, rather than buried in the Settings tab. Three independent
        # controls compose into the real self._vars["topology"] string (see
        # _compose_topology_mode / _decompose_topology_mode above the
        # class) -- this is purely a presentation-layer split; net.py/
        # topology_canvas.py/etc. still only ever see the one flat mode
        # string they already understood.
        topo_row = tk.Frame(settings_panel, bg=P["bg"])
        topo_row.pack(fill="x", pady=(0, 8))
        tk.Label(topo_row, text="Network Topology:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left")
        net_topo_cb = ttk.Combobox(
            topo_row, textvariable=self._net_topo_var,
            values=["star", "relay"], state="readonly", width=7,
        )
        net_topo_cb.pack(side="left", padx=(6, 12))

        tk.Label(topo_row, text="STA Type:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left")
        ttk.Combobox(
            topo_row, textvariable=self._sta_type_var,
            values=["Ground STA", "UAV"], state="readonly", width=11,
        ).pack(side="left", padx=(6, 12))

        tk.Label(topo_row, text="Relay Type:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left")
        self._relay_type_cb = ttk.Combobox(
            topo_row, textvariable=self._relay_type_var,
            values=["Grounded", "Aerial (UAV)"], state="readonly", width=12,
        )
        self._relay_type_cb.pack(side="left", padx=(6, 16))
        # Meaningless without relays -- greyed out whenever Network Topology
        # is "star", so it can't be picked in a combination that would just
        # be silently ignored.
        self._update_relay_type_enabled()

        for _v in (self._net_topo_var, self._sta_type_var, self._relay_type_var):
            _v.trace_add("write", self._on_topology_controls_change)

        # Relay count/placement + the Apply & Rebuild shortcut get their own
        # row -- alongside Network Topology/STA Type/Relay Type above, that
        # was six labelled controls competing for one line and routinely
        # ran the button off the right edge of the window (this tab's own
        # "Apply & Rebuild" is a convenience duplicate of the one on the
        # Settings tab, so it's worth keeping fully visible, not clipped).
        topo_row2 = tk.Frame(settings_panel, bg=P["bg"])
        topo_row2.pack(fill="x", pady=(0, 8))
        tk.Label(topo_row2, text="Relay/UAV Count:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left")
        ttk.Combobox(
            topo_row2, textvariable=self._vars["num_relays"],
            values=[1, 2, 3, 4, 5, 8, 10], state="normal", width=6,
        ).pack(side="left", padx=(6, 16))
        tk.Label(topo_row2, text="Relay Placement:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left")
        ttk.Combobox(
            topo_row2, textvariable=self._vars["relay_placement"],
            values=["optimal", "random"], state="readonly", width=8,
        ).pack(side="left", padx=(6, 16))
        _tk_btn(topo_row2, "Apply & Rebuild", self.apply_settings,
                bg=P["blue"]).pack(side="left")

        # Cruise speed for every kind of airborne motion -- random-waypoint
        # UAV end-nodes, "Random Path" relays, AND (see _drone_racetrack_
        # geometry) the "Oval Path" relay racetrack's lap speed, which used
        # to be a fixed constant immune to this slider. One shared control
        # instead of a second, easy-to-miss speed field just for the oval.
        # Takes effect live (no rebuild needed), like Zoom/Map style, since
        # it only affects the mobility model's target speed going forward,
        # not the topology structure. Own row: with Network Topology/STA
        # Type/Relay Type/Relay Count/Relay Placement/Apply & Rebuild above
        # already filling one row, this and Relay Motion routinely ran off
        # the right edge of a normal window.
        motion_row = tk.Frame(settings_panel, bg=P["bg"])
        motion_row.pack(fill="x", pady=(0, 2))
        tk.Label(motion_row, text="Flight Speed (m/s):", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left")
        self._uav_speed_var = tk.DoubleVar(value=8.0)
        ttk.Combobox(
            motion_row, textvariable=self._uav_speed_var,
            values=[2, 4, 6, 8, 10, 12, 15, 20, 30], state="normal", width=6,
        ).pack(side="left", padx=(6, 16))

        # aerial_relay-only: relays can either fly the fixed shared
        # racetrack (advance_drone_positions -- shape set on the row below)
        # or wander independently, same random-waypoint mobility "uav"
        # end-nodes already use (advance_uav_positions, which stays within
        # a fixed roam annulus around the AP -- see _advance_drones()) --
        # takes effect live too, same as Flight Speed above.
        tk.Label(motion_row, text="Relay Motion:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10)).pack(side="left")
        self._relay_motion_var = tk.StringVar(value="Oval Path")
        ttk.Combobox(
            motion_row, textvariable=self._relay_motion_var,
            values=["Oval Path", "Random Path"], state="readonly", width=11,
        ).pack(side="left")

        _help_row(
            "STA Type UAV: every STA wanders randomly (ignores count).  "
            "Relay Type Aerial: relay-role nodes fly per Relay Motion above "
            "(count ≤ Number of STAs, in Settings).  Mix both freely."
        )

        # The aerial relay's Oval Path shape itself -- radius and centre,
        # both overridable live (same no-rebuild-needed rule as Flight
        # Speed/Relay Motion above). Only visible effect is on Relay Type
        # "Aerial (UAV)" with Relay Motion "Oval Path", but (like every
        # other control here) stays editable regardless of whether that
        # combination is currently active. Drag-to-set sliders instead of
        # a type-or-pick-a-preset combobox -- radius/centre are exactly the
        # kind of "nudge it and watch the map" value a slider suits, and
        # four of them plus a Reset button was already a tight fit as
        # comboboxes; split across two rows (radius, then centre) rather
        # than risk the same overflow the rest of this tab just got fixed
        # for.
        def _oval_slider(row, label, var, lo, hi):
            tk.Label(row, text=label, bg=P["bg"], fg=P["fg"],
                     font=("Arial", 10)).pack(side="left")
            tk.Scale(
                row, from_=lo, to=hi, orient="horizontal", variable=var,
                resolution=25, length=170, showvalue=True,
                bg=P["bg"], fg=P["fg"], troughcolor="#cbd5e1",
                highlightthickness=0, font=("Arial", 8), sliderlength=16,
            ).pack(side="left", padx=(6, 18))

        path_row = tk.Frame(settings_panel, bg=P["bg"])
        path_row.pack(fill="x", pady=(0, 2))
        tk.Label(path_row, text="Relay Path (Oval) — Radius:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10, "bold")).pack(side="left", padx=(0, 10))
        self._relay_path_rx_var = tk.DoubleVar(value=0.0)
        self._relay_path_ry_var = tk.DoubleVar(value=0.0)
        _oval_slider(path_row, "X (m):", self._relay_path_rx_var, 0, 2000)
        _oval_slider(path_row, "Y (m):", self._relay_path_ry_var, 0, 2000)

        path_row2 = tk.Frame(settings_panel, bg=P["bg"])
        path_row2.pack(fill="x", pady=(0, 2))
        tk.Label(path_row2, text="Relay Path (Oval) — Center:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10, "bold")).pack(side="left", padx=(0, 10))
        self._relay_path_cx_var = tk.DoubleVar(value=0.0)
        self._relay_path_cy_var = tk.DoubleVar(value=0.0)
        _oval_slider(path_row2, "X (m):", self._relay_path_cx_var, -800, 800)
        _oval_slider(path_row2, "Y (m):", self._relay_path_cy_var, -800, 800)
        _tk_btn(path_row2, "Reset to Auto", self._reset_relay_path,
                bg=P["muted"]).pack(side="left")

        _help_row(
            "Radius 0 = auto (scaled from AP range). Center 0, 0 = centred on the AP. Takes effect live."
        )

        # Save/load the physical layout itself (node positions + relay
        # assignment), independent of Environment/Layout above -- lets the
        # same topology be re-run under a different environment instead of
        # every environment switch implicitly re-randomising the scatter
        # too (see topology_canvas.py's save_topology/load_topology/
        # apply_topology). scripts/main_gui.py's own --topology-file flag
        # is the command-line equivalent of Load.
        topo_io_row = tk.Frame(settings_panel, bg=P["bg"])
        topo_io_row.pack(fill="x", pady=(0, 2))
        tk.Label(topo_io_row, text="Saved Topology:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10, "bold")).pack(side="left", padx=(0, 10))
        _tk_btn(topo_io_row, "Save Topology…", self._save_topology,
                bg="#334155", fg=P["side_fg"]).pack(side="left", padx=(0, 8))
        _tk_btn(topo_io_row, "Load Topology…", self._load_topology,
                bg="#334155", fg=P["side_fg"]).pack(side="left")

        _help_row(
            "Save the current node layout to a file, then Load it again (here or with "
            "--topology-file on the command line) to re-run the exact same positions under "
            "a different Environment/Layout."
        )

        # Packed once, immediately, so it's always the LAST child of parent
        # at this point -- settings_panel toggles with before=body (see
        # _toggle_topology_settings), which is what keeps it correctly
        # positioned above the map on every show/hide instead of jumping
        # to the bottom of the packing order after the first toggle.
        body = tk.Frame(parent, bg=P["bg"])
        self._topo_body = body
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._net_canvas = NetworkCanvas(body, on_select=self._on_node_selected)
        self._net_canvas.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(body, bg=P["card"], width=230,
                          highlightthickness=1, highlightbackground=P["border"])
        panel.pack(side="right", fill="y", padx=(10, 0))
        panel.pack_propagate(False)
        self._build_node_props_panel(panel)

    def _build_node_props_panel(self, panel):
        tk.Label(panel, text="NODE SETTINGS", font=("Arial", 8, "bold"),
                 bg=P["card"], fg=P["muted"]).pack(anchor="w", padx=12, pady=(12, 6))

        self._np_empty = tk.Label(
            panel, text="Click a node in the map to\nedit its position and range.",
            font=("Arial", 9, "italic"), bg=P["card"], fg=P["muted"],
            justify="left", wraplength=200)
        self._np_empty.pack(anchor="w", padx=12, pady=6)

        self._np_form = tk.Frame(panel, bg=P["card"])

        self._np_title = tk.Label(self._np_form, text="", font=("Arial", 13, "bold"),
                                   bg=P["card"], fg=P["fg"])
        self._np_title.pack(anchor="w", padx=12, pady=(0, 0))
        self._np_role = tk.Label(self._np_form, text="", font=("Arial", 9),
                                  bg=P["card"], fg=P["muted"])
        self._np_role.pack(anchor="w", padx=12, pady=(0, 10))

        self._np_vars: Dict[str, tk.StringVar] = {
            "x": tk.StringVar(), "y": tk.StringVar(), "range": tk.StringVar(),
        }

        def field(label, key):
            f = tk.Frame(self._np_form, bg=P["card"])
            f.pack(fill="x", padx=12, pady=3)
            tk.Label(f, text=label, font=("Arial", 9), bg=P["card"],
                     fg=P["fg"], width=10, anchor="w").pack(side="left")
            e = tk.Entry(f, textvariable=self._np_vars[key], font=("Arial", 9), width=10)
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self._apply_node_props())
            return e

        self._np_x_entry = field("X (m)", "x")
        self._np_y_entry = field("Y (m)", "y")
        field("Range (m)", "range")

        self._np_ap_note = tk.Label(
            self._np_form, text="The AP is a fixed reference point --\nposition cannot be changed.",
            font=("Arial", 8, "italic"), bg=P["card"], fg=P["muted"], justify="left")

        # Association preference -- only shown for STA nodes in a topology
        # that actually has relays, since that's the only time "join AP or
        # relay" is a real choice.
        self._np_assoc_frame = tk.Frame(self._np_form, bg=P["card"])
        tk.Label(self._np_assoc_frame, text="Associate via", font=("Arial", 9),
                 bg=P["card"], fg=P["fg"], width=10, anchor="w").pack(side="left")
        self._np_assoc_var = tk.StringVar(value="Auto")
        self._np_assoc_cb = ttk.Combobox(
            self._np_assoc_frame, textvariable=self._np_assoc_var,
            values=["Auto", "AP only", "Relay only"], state="readonly", width=10,
        )
        self._np_assoc_cb.pack(side="left")
        self._np_assoc_cb.bind("<<ComboboxSelected>>", lambda _e: self._apply_node_props())

        tk.Label(self._np_form,
                 text="Range sets this node's transmit\npower so its link reaches that\ndistance before RSSI drops\nbelow receiver sensitivity.",
                 font=("Arial", 8, "italic"), bg=P["card"], fg=P["muted"],
                 justify="left", wraplength=200).pack(anchor="w", padx=12, pady=(4, 8))

        _tk_btn(self._np_form, "Apply", self._apply_node_props,
                bg=P["blue"]).pack(fill="x", padx=12, pady=(0, 6))

        self._np_dist_lbl = tk.Label(self._np_form, text="", font=("Arial", 8, "bold"),
                                      bg=P["card"], fg=P["muted"], justify="left",
                                      wraplength=200)
        self._np_dist_lbl.pack(anchor="w", padx=12, pady=(6, 12))

        self._np_selected: Optional[int] = None

    def _on_node_selected(self, node_id):
        for c in self._net_canvases():
            if c is not self._net_canvas:
                c.refresh()
        if node_id is None or self.sim is None or node_id not in self.sim.nodes:
            self._np_form.pack_forget()
            self._np_empty.pack(anchor="w", padx=12, pady=6)
            self._np_selected = None
            return
        self._np_empty.pack_forget()
        self._np_form.pack(fill="x")
        self._np_selected = node_id

        node = self.sim.nodes[node_id]
        role_lbl = {"AP": "Access Point", "RELAY": "Relay", "STA": "Station"}.get(
            node.role, node.role)
        self._np_title.config(text=f"Node {node_id}")
        self._np_role.config(text=role_lbl)
        x, y = node.pos
        self._np_vars["x"].set(f"{x:.1f}")
        self._np_vars["y"].set(f"{y:.1f}")
        self._np_vars["range"].set(f"{range_m_for_node(node):.0f}")
        self._update_node_dist_label(node)

        is_ap = (node_id == 0)
        self._np_x_entry.config(state="disabled" if is_ap else "normal")
        self._np_y_entry.config(state="disabled" if is_ap else "normal")
        if is_ap:
            self._np_ap_note.pack(anchor="w", padx=12, pady=(2, 8))
        else:
            self._np_ap_note.pack_forget()

        relay_ids = self.sim.config.get("topology", {}).get("relay_ids", [])
        show_assoc = (node.role == "STA" and bool(relay_ids))
        if show_assoc:
            pref = getattr(node.mac.ctx, "_assoc_preference", "auto")
            self._np_assoc_var.set({"ap": "AP only", "relay": "Relay only"}.get(pref, "Auto"))
            self._np_assoc_frame.pack(fill="x", padx=12, pady=(0, 6))
        else:
            self._np_assoc_frame.pack_forget()

    def _update_node_dist_label(self, node):
        # The AP `node` is actually associated with (falling back to the
        # nearest one if unassociated), not always node 0 -- see
        # resolve_ap_peer's own docstring. Under multi-AP a STA right next
        # to, and correctly associated with, AP5 needs its distance/RSSI
        # measured against AP5, not AP0.
        ap = resolve_ap_peer(node, self.sim)
        if ap is None or ap.phy is None or node.phy is None:
            self._np_dist_lbl.config(text="")
            return
        dist = math.hypot(node.pos[0] - ap.pos[0], node.pos[1] - ap.pos[1])
        # Real RSSI via the same PHY path-loss model the simulator uses for
        # actual transmissions -- includes obstruction loss from any
        # building/mountain between the two, not just nominal free-space
        # range, so a node sitting behind an obstacle correctly shows as
        # out of range even if it's well within the AP's raw distance.
        rssi = ap.phy._rssi_dbm(ap.node_id, node.node_id, ap.phy.eirp_dbm)
        sensitivity = node.phy.rx_sensitivity_dbm
        margin = rssi - sensitivity
        in_range = margin >= 0.0
        self._np_dist_lbl.config(
            text=(f"Distance to AP: {dist:.1f} m\n"
                  f"RSSI: {rssi:.1f} dBm (need ≥ {sensitivity:.0f} dBm)\n"
                  f"{'✓ In range' if in_range else '✗ Out of range'} "
                  f"({margin:+.0f} dB margin)"),
            fg=P["green_dk"] if in_range else P["red_dk"],
        )

    def _apply_node_props(self):
        if self._np_selected is None or self.sim is None:
            return
        node = self.sim.nodes.get(self._np_selected)
        if node is None:
            return
        try:
            x = float(self._np_vars["x"].get())
            y = float(self._np_vars["y"].get())
            rng = float(self._np_vars["range"].get())
        except ValueError:
            messagebox.showerror("Invalid value", "X, Y and Range must be numbers.")
            return
        if node.node_id != 0:
            node.pos = (x, y)
        if rng > 0:
            set_range_m_for_node(node, rng)
        if node.role == "STA":
            pref = {"AP only": "ap", "Relay only": "relay"}.get(self._np_assoc_var.get(), "auto")
            node.mac.ctx._assoc_preference = pref
        for c in self._net_canvases():
            c.refresh()
        self._update_node_dist_label(node)

    def _reset_topo_layout(self):
        if self.sim is None:
            return
        self._net_canvas.reset_layout()
        for c in self._net_canvases():
            if c is not self._net_canvas:
                c.sync_from_sim(self.sim)
        self._on_node_selected(self._net_canvas.selected_id)

    def _open_3d_view(self):
        """Launch (once) a local zero-dependency HTTP server that hosts a
        real three.js 3D scene fed live from this Dashboard's sim state,
        and open it in the default browser -- Tkinter's Canvas can't do
        real 3D, so this runs as a companion view rather than replacing
        the 2D topology panel. See ui/web3d/."""
        try:
            if self._web3d_server is None:
                from ui.web3d.server import Web3DServer
                self._web3d_server = Web3DServer(self)
                self._web3d_server.start()
            webbrowser.open(self._web3d_server.url)
        except Exception as e:
            messagebox.showerror("3D View", f"Could not start the 3D view: {e}")

    def _open_real_map_view(self):
        """Open the real-map (MapLibre satellite/street tiles + a live
        AP/STA/UAV 802.11ah overlay) digital twin. Its node layer polls
        the same /api/state endpoint the three.js Open 3D View uses, so it
        needs to be served by the same Web3DServer (not opened via a bare
        file:// URL) -- see ui/web3d/static/smart-city-simulation.html."""
        try:
            if self._web3d_server is None:
                from ui.web3d.server import Web3DServer
                self._web3d_server = Web3DServer(self)
                self._web3d_server.start()
            webbrowser.open(f"{self._web3d_server.url}smart-city-simulation.html")
        except Exception as e:
            messagebox.showerror("Real Map View", f"Could not open the real-map view: {e}")

    def _open_cars_uavs_map_view(self):
        """Open the cars_uavs-specific real-map twin (MapLibre real
        street/satellite tiles under live AP/car/scooter/UAV markers) --
        a leaner, purpose-built sibling of smart-city-simulation.html for
        this topology specifically: that page assumes exactly one AP at
        a hand-authored real campus location plus a load of Dhanbad-
        campus-specific decoration (landmarks, hand-drawn routes); this
        scenario has 6 APs and no real location of its own, and doesn't
        need any of that -- see ui/web3d/static/cars-uavs-map.html, which
        anchors the whole corridor on that same already-verified campus
        origin purely as a real-world coordinate to draw genuine map
        tiles under, not because the corridor represents an actual road
        there. Same /api/state polling, same Web3DServer, as every other
        web view."""
        try:
            if self._web3d_server is None:
                from ui.web3d.server import Web3DServer
                self._web3d_server = Web3DServer(self)
                self._web3d_server.start()
            webbrowser.open(f"{self._web3d_server.url}cars-uavs-map.html")
        except Exception as e:
            messagebox.showerror("Real Map View", f"Could not open the real-map view: {e}")

    def _on_topology_controls_change(self, *_):
        mode = _compose_topology_mode(
            self._net_topo_var.get(),
            self._sta_type_var.get() == "UAV",
            self._relay_type_var.get() == "Aerial (UAV)",
        )
        self._vars["topology"].set(mode)
        self._update_relay_type_enabled()

    def _update_relay_type_enabled(self):
        # Meaningless without relays -- greyed out rather than left pickable
        # in a combination ("star" + an aerial relay type) that would just
        # be silently ignored.
        self._relay_type_cb.config(
            state="readonly" if self._net_topo_var.get() == "relay" else "disabled")

    def _sync_topology_controls_from_mode(self):
        """Refresh the three GUI-facing controls from the current
        self._vars["topology"] value -- needed anywhere that var gets set
        directly rather than via the controls themselves (Military Zone
        auto-topology below, _reset_defaults), so the dropdowns never show
        a combination that doesn't actually match the applied topology."""
        net_topo, sta_uav, relay_aerial = _decompose_topology_mode(self._vars["topology"].get())
        self._net_topo_var.set(net_topo)
        self._sta_type_var.set("UAV" if sta_uav else "Ground STA")
        self._relay_type_var.set("Aerial (UAV)" if relay_aerial else "Grounded")
        self._update_relay_type_enabled()

    def _toggle_topology_settings(self):
        self._topo_settings_open = not self._topo_settings_open
        if self._topo_settings_open:
            self._topo_settings_toggle.config(text="▼  Topology Settings  (structure, relay path, save/load)")
            # before=self._topo_body is what keeps this correctly positioned
            # above the map every time it's re-shown, instead of pack()
            # appending it after whatever's already packed (which would put
            # it below the map on the second toggle onward).
            self._topo_settings_panel.pack(fill="x", padx=12, pady=(0, 4), before=self._topo_body)
        else:
            self._topo_settings_toggle.config(text="▶  Topology Settings  (structure, relay path, save/load)")
            self._topo_settings_panel.pack_forget()

    def _reset_relay_path(self):
        # Radius 0 is _drone_racetrack_geometry's own sentinel for "use the
        # AP-range-derived default" -- see that function's docstring.
        self._relay_path_rx_var.set(0.0)
        self._relay_path_ry_var.set(0.0)
        self._relay_path_cx_var.set(0.0)
        self._relay_path_cy_var.set(0.0)

    def _save_topology(self):
        if self.sim is None:
            messagebox.showerror("Save Topology", "No simulation is built yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Topology",
            defaultextension=".json",
            filetypes=[("Topology JSON", "*.json"), ("All files", "*.*")],
            initialfile="topology.json",
        )
        if not path:
            return
        try:
            save_topology(self.sim, path)
        except Exception as e:
            messagebox.showerror("Save Topology", f"Couldn't save: {e}")
            return
        self._badge_detail.config(text=f"Topology saved to {os.path.basename(path)}.")

    def _load_topology(self):
        path = filedialog.askopenfilename(
            title="Load Topology",
            filetypes=[("Topology JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = load_topology(path)
        except Exception as e:
            messagebox.showerror("Load Topology", f"Couldn't load {os.path.basename(path)}: {e}")
            return

        # Match this file's shape (STA/relay count, topology mode) BEFORE
        # rebuilding, so the rebuilt sim actually has the node IDs
        # apply_topology is about to hand positions to -- Environment/
        # Layout are deliberately left untouched, since the whole point is
        # re-running the same saved layout under whatever environment is
        # currently selected.
        self._vars["num_stas"].set(int(data.get("num_stas", self._vars["num_stas"].get())))
        self._vars["num_relays"].set(int(data.get("num_relays", self._vars["num_relays"].get())))
        self._vars["relay_placement"].set(str(data.get("relay_placement", self._vars["relay_placement"].get())))
        self._vars["topology"].set(str(data.get("mode", self._vars["topology"].get())))
        self._sync_topology_controls_from_mode()

        try:
            was_running = self._rebuild()
            apply_topology(self.sim, data)
            for c in self._net_canvases():
                c.refresh()
            if was_running:
                self._set_state(self.RUNNING)
        except Exception as e:
            messagebox.showerror("Load Topology", f"Couldn't apply {os.path.basename(path)}: {e}")
            return
        self._badge_detail.config(
            text=f"Topology loaded from {os.path.basename(path)} "
                 f"({data.get('num_stas', '?')} STAs, {data.get('num_relays', 0)} relays)."
        )

    def _on_env_change(self, _e=None):
        env = self._env_var.get()
        for c in self._net_canvases():
            c.set_environment(env)

        # Environment changes the AP's real PHY link budget too (see
        # topology_canvas.py's _ENV_PATH_LOSS_EXP / sync_from_sim) --
        # without this, picking an environment before ever pressing Start
        # left every node's PhyLayer on whatever exponent the last real
        # rebuild applied (the plain config default, if there hasn't been
        # one yet), so the layout stayed sized for the *previous*
        # environment's range instead of the one just picked. While
        # idle/paused, reseed the same way the "Reset Layout" button does
        # so node scatter actually matches the new range. A currently-
        # RUNNING sim keeps its accumulated state and node positions --
        # calling the full sync_from_sim there would still be *correct*
        # (its own all_origin check is false, so it wouldn't reposition
        # anything) but it also clears drone trails and resets the
        # canvas's manual pan/zoom back to auto-fit as a side effect,
        # which would be a surprising papercut for someone just previewing
        # scenery mid-run. _apply_environment_path_loss alone gives every
        # node's PhyLayer the new exponent for its live RF math without
        # touching any of that.
        sim = getattr(self, "sim", None)
        if sim is not None:
            if self._state != self.RUNNING:
                self._net_canvas.reset_layout()
                for c in self._net_canvases():
                    if c is not self._net_canvas:
                        c.sync_from_sim(sim)
            else:
                _apply_environment_path_loss(sim, env)
                # refresh() (not the heavier sync_from_sim) is enough to make
                # sim.obstacles reflect the new environment's real buildings
                # right now -- it otherwise only updates on whatever canvas
                # happens to redraw next, which the obstruction check right
                # after this needs to already be current. A node whose
                # straight-line path to its peer crosses a building that
                # only just appeared here (position unchanged, environment
                # swapped out from under it) would otherwise be stuck
                # UNASSOCIATED forever with no way back -- see
                # _declutter_obstructed_nodes's own docstring for why
                # that's a real dead end, not just a slow retry.
                for c in self._net_canvases():
                    c.refresh()
                n_moved = _declutter_obstructed_nodes(
                    sim, sim.config.get("topology", {}).get("mode", "star"),
                    set(sim.config.get("topology", {}).get("relay_ids", [])),
                )
                if n_moved:
                    sim.log(layer="TOPOLOGY", event="ENV_CHANGE_DECLUTTER",
                            node_id=-1, details={"environment": env, "nodes_moved": n_moved})
                    for c in self._net_canvases():
                        c.refresh()
            self._update_propagation_label()

        # Military Zone's whole premise is a forward base relaying back to
        # a rear AP -- but "Topology Mode" defaults to "star" everywhere
        # (see _reset_defaults/the module's own __main__ block), which
        # never builds relay nodes at all (main_gui.build_sim only calls
        # RelayBuilder.build() for topology in ("relay", "aerial_relay");
        # "star" silently ignores num_relays). Without this, "STAs aren't
        # joining relays" was actually "there are no relays" -- switching
        # environment alone never changed topology, so every entry point
        # that didn't explicitly request topology="relay" hit this. Only
        # steps in if the user hasn't already picked a relay-style topology
        # themselves (relay/aerial_relay/relay_uav/aerial_relay_uav), so it
        # won't clobber an explicit relay choice.
        if env == "Military Zone" and self._vars["topology"].get() not in (
            "relay", "aerial_relay", "relay_uav", "aerial_relay_uav",
        ):
            self._vars["topology"].set("relay")
            self._sync_topology_controls_from_mode()

        names = self._LAYOUT_NAMES.get(env, self._LAYOUT_NAMES["Open Area"])
        self._layout_cb.config(values=names)
        self._layout_var.set(names[0])
        self._on_layout_change()

    def _on_layout_change(self, _e=None):
        label = self._layout_var.get()
        if self._env_var.get() == "Smart City" and label == self._REAL_MAP_LAYOUT:
            self._open_real_map_view()
            return
        try:
            variant = int(label.split(":", 1)[0])
        except (ValueError, IndexError):
            variant = 1
        for c in self._net_canvases():
            c.set_layout_variant(variant)

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg="#0f172a", height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._sb_status = tk.Label(bar, text="Ready",
                                    font=("Arial", 9), bg="#0f172a", fg="#94a3b8")
        self._sb_status.pack(side="left", padx=10)
        tk.Label(bar, text="|", bg="#0f172a", fg="#334155").pack(side="left")

        self._sb_time = tk.Label(bar, text="T = 0.000 s",
                                  font=("Courier New", 9), bg="#0f172a", fg="#64748b")
        self._sb_time.pack(side="left", padx=10)
        tk.Label(bar, text="|", bg="#0f172a", fg="#334155").pack(side="left")

        self._sb_logs = tk.Label(bar, text="Logs: 0",
                                  font=("Arial", 9), bg="#0f172a", fg="#64748b")
        self._sb_logs.pack(side="left", padx=10)

        tk.Label(bar, text="sim11ah  v2.0  |  IEEE 802.11ah",
                 font=("Arial", 9), bg="#0f172a", fg="#334155").pack(side="right",
                                                                       padx=10)

    # ── Settings helpers ──────────────────────────────────────────────────────
    def _sig(self) -> dict:
        skip = {"sim_speed", "log_filter", "log_autoscroll"}
        out = {}
        for k, v in self._vars.items():
            if k in skip:
                continue
            try:
                out[k] = v.get()
            except Exception:
                out[k] = str(v)   # keep raw string if var parse fails
        return out

    def _on_speed_change(self, _=None):
        self.update_ms = self._SPEEDS.get(self._vars["sim_speed"].get(), 145)

    def _set_state(self, state: str):
        self._state = state
        self.running = (state == self.RUNNING)
        if state == self.RUNNING:
            self._hdr_badge.config(text="● RUNNING", fg=P["green"])
            self._badge.config(text="● Running", fg=P["green"])
            self._btn_run.config(text="⏸  Pause", bg="#f59e0b")
            self._btn_run._bg  = "#f59e0b"
            self._btn_run._cmd = self.pause
        elif state == self.PAUSED:
            self._hdr_badge.config(text="● PAUSED", fg=P["amber"])
            self._badge.config(text="● Paused", fg=P["amber"])
            self._btn_run.config(text="▶  Resume", bg="#22c55e")
            self._btn_run._bg  = "#22c55e"
            self._btn_run._cmd = self.resume
        else:  # IDLE
            self._hdr_badge.config(text="● STOPPED", fg=P["red"])
            self._badge.config(text="● Ready", fg=P["amber"])
            self._btn_run.config(text="▶  Start", bg="#22c55e")
            self._btn_run._bg  = "#22c55e"
            self._btn_run._cmd = self.start

    def _set_running(self, val: bool):
        self._set_state(self.RUNNING if val else self.IDLE)

    def _reset_defaults(self):
        defs = {
            "num_stas": 50, "seed": 0, "traffic": "periodic",
            "raw_enable": "Enabled", "raw_policy": "static",
            "raw_groups": 4, "raw_slots": 8, "raw_slot_duration_ms": 14.0,
            "packet_size": 128, "packet_interval": 5.0,
            "topology": "star", "num_relays": 2, "freq_mhz": 915.0,
            "sensor_profile": "(none)", "video_fps": 5.0,
        }
        for k, v in defs.items():
            if k in self._vars:
                self._vars[k].set(v)
        self._sync_topology_controls_from_mode()

    def _on_settings_change(self, *_):
        if not hasattr(self, "_pending_lbl"):
            return
        has_changes = self._sig() != self._applied
        if has_changes:
            self._pending_lbl.config(text="⚠ Pending changes — click Apply",
                                      fg=P["amber"])
            self._btn_apply_sidebar.config(bg=P["blue"], fg="white")
            self._btn_apply_sidebar._bg = P["blue"]
        else:
            self._pending_lbl.config(text="— No pending changes —", fg="#475569")
            self._btn_apply_sidebar.config(bg="#334155", fg="#94a3b8")
            self._btn_apply_sidebar._bg = "#334155"

    def _net_canvases(self):
        """All live NetworkCanvas views (Topology tab, Overview mini-map,
        Settings preview) -- kept as a single list so every sync/animation
        call site touches all three and they never drift apart."""
        return [c for c in (
            getattr(self, "_net_canvas", None),
            getattr(self, "_topo_mini", None),
            getattr(self, "_settings_topo_preview", None),
        ) if c is not None]

    def _shutdown_sim(self):
        try:
            for n in self.sim.nodes.values():
                try: n.stop()
                except Exception: pass
            for n in self.sim.nodes.values():
                try: n.finalize()
                except Exception: pass
        except Exception:
            pass

    # Settings that actually change *which nodes exist or where they get
    # scattered* -- topology mode/relay count/relay placement style change
    # the node set or its spatial pattern outright, and seed is the user
    # explicitly asking for a new random draw. Every other knob (RAW
    # policy, packet size/interval, traffic mode, freq, RAW group/slot
    # sizing) is a protocol/PHY parameter you sweep to compare against a
    # FIXED deployment -- re-randomizing node positions out from under a
    # policy comparison (e.g. static vs adaptive RAW) would confound the
    # comparison with an unrelated layout change. See _rebuild().
    _LAYOUT_AFFECTING_KEYS = {"topology", "num_stas", "num_relays", "relay_placement", "seed"}

    def _rebuild(self) -> bool:
        if self.sim_builder is None:
            raise RuntimeError("No sim_builder provided.")
        was_running = (self._state == self.RUNNING)
        self._set_state(self.IDLE)

        sig = self._sig()
        layout_changed = (
            self.sim is None
            or any(str(sig.get(k)) != str(self._applied.get(k)) for k in self._LAYOUT_AFFECTING_KEYS)
        )
        old_positions = None
        if not layout_changed and self.sim is not None:
            old_positions = {nid: n.pos for nid, n in self.sim.nodes.items()}

        self._shutdown_sim()

        # A chosen Sensor Profile (Settings tab) overrides traffic/packet-
        # size/etc. with a realistic preset for whatever environment is
        # active (Topology tab) -- looked up fresh here rather than cached
        # from the dropdown's own change event, so it can never go stale
        # relative to whichever environment is actually selected right now.
        profile_name = str(sig.get("sensor_profile", "(none)"))
        app_overrides = None
        if profile_name and profile_name != "(none)":
            try:
                app_overrides = _sensor_get_profile(self._env_var.get(), profile_name)
            except KeyError:
                app_overrides = None

        self.sim = self.sim_builder(
            num_stas=int(sig["num_stas"]),
            seed=int(sig["seed"]),
            traffic=str(sig["traffic"]),
            raw_enable=str(sig["raw_enable"]).lower().startswith("en"),
            raw_policy=str(sig["raw_policy"]),
            packet_size=int(sig["packet_size"]),
            packet_interval=float(sig["packet_interval"]),
            topology=str(sig["topology"]),
            num_relays=int(sig["num_relays"]),
            freq_mhz=float(sig["freq_mhz"]),
            raw_num_groups=int(sig["raw_groups"]),
            raw_num_slots=int(sig["raw_slots"]),
            raw_slot_duration=float(sig["raw_slot_duration_ms"]) / 1000.0,
            video_fps=float(sig["video_fps"]),
            app_overrides=app_overrides,
        )
        # sim_builder/RelayBuilder don't know about this -- it's purely a
        # layout-seeding concern (see topology_canvas.py's
        # _seed_default_layout), same "read from sim.config['topology']"
        # spot mode/relay_ids already live in.
        self.sim.config.setdefault("topology", {})["relay_placement"] = str(sig["relay_placement"])

        if old_positions is not None:
            # None of _LAYOUT_AFFECTING_KEYS changed, so the new sim has
            # the exact same node-id set as the old one -- carry the old
            # positions straight over. NetworkCanvas.sync_from_sim only
            # re-randomizes a layout when every non-AP node is still sitting
            # at the origin (its fresh-build default), so this is enough to
            # make it skip re-seeding and keep the topology stable.
            for nid, n in self.sim.nodes.items():
                if nid in old_positions:
                    n.pos = old_positions[nid]

        self._applied = sig
        self._tput_max = 1.0
        self._pkt_log_ptr = 0
        self._trace_log_ptr = 0
        self._init_metrics_state()
        self._lc_pdr.clear()
        self._lc_tput.clear()
        self._lc_delay.clear()
        self._lc_drop.clear()
        for c in self._net_canvases():
            c.clear_packets()
        self._clear_trace_table()
        self._refresh_topology()
        self.after(150, self._refresh_topology)   # re-draw after canvas re-renders
        return was_running

    def apply_settings(self):
        try:
            was_running = self._rebuild()
            self._clear_log()
            self._log.insert("end", "Settings applied. Press Start to begin.\n")
            for c in (self._c_pdr, self._c_tput, self._c_delay,
                      self._c_gen, self._c_del, self._c_drop):
                c.set("—")
            self._bc_gen.draw({})
            self._bc_del.draw({})
            self._bc_pdr.draw({})
            self._badge_detail.config(text="Settings applied.")
            if was_running:
                self._set_state(self.RUNNING)
                self._refresh()
            self._on_settings_change()
        except Exception as e:
            self._badge_detail.config(text=f"Error: {e}")
            messagebox.showerror("Settings Error", str(e))

    def _refresh_topology(self):
        try:
            for c in self._net_canvases():
                c.sync_from_sim(self.sim)
        except Exception:
            pass
        self._update_propagation_label()

    def _update_propagation_label(self):
        """Keeps the Settings tab's "Propagation Model" line honest --
        see where self._propagation_lbl is built for why a fixed string
        went stale. Called from _refresh_topology (Apply & Rebuild/Stop &
        Reset/Load Topology/initial build all funnel through it) and
        directly from _on_env_change's already-running branch, the one
        path that changes the exponent without a rebuild."""
        lbl = getattr(self, "_propagation_lbl", None)
        sim = getattr(self, "sim", None)
        if lbl is None or sim is None:
            return
        ap = sim.nodes.get(0)
        if ap is None or ap.phy is None:
            return
        text = f"Log-distance, η={ap.phy.path_loss_exp:.2f}"
        canvas = getattr(self, "_net_canvas", None)
        if canvas is not None:
            text += f" ({canvas.environment})"
        try:
            text += f" — AP range ≈ {_ap_range_m(sim):.0f} m"
        except Exception:
            pass
        lbl.config(text=text)

    def _on_tab_changed(self, _e=None):
        # ttk.Notebook only fires <Configure> on a widget's first realization,
        # not on every tab switch -- so a canvas edited while its tab was
        # hidden can go stale until something forces a redraw. Re-pull the
        # latest node positions from the live sim every time the visible
        # tab changes, so all three map views never drift apart.
        try:
            for c in self._net_canvases():
                c.refresh()
        except Exception:
            pass

    # ── Simulation controls ───────────────────────────────────────────────────
    def start(self):
        try:
            if self._sig() != self._applied:
                self._rebuild()
            if self._state != self.RUNNING:
                self._set_state(self.RUNNING)
                self._badge_detail.config(text="Simulation running…")
                self._refresh()
        except Exception as e:
            self._set_state(self.IDLE)
            self._badge_detail.config(text=f"Start error: {e}")

    def stop_reset(self):
        # Previously this only flipped the state to IDLE and left self.sim
        # (association states, packet queues, RNG-drawn positions, RAW
        # schedule, everything) exactly as the run left it -- "Stop &
        # Reset" looked like a full reset but wasn't one. _rebuild() is the
        # same fresh-simulator path Apply & Rebuild uses, so this now
        # actually discards the old sim and builds a brand new one from
        # the currently configured settings, same as a cold start.
        try:
            self._rebuild()
            self._badge_detail.config(text="Stopped & reset.")
        except Exception as e:
            self._badge_detail.config(text=f"Reset error: {e}")

    def stop(self):
        self.stop_reset()

    def pause(self):
        if self._state == self.RUNNING:
            self._set_state(self.PAUSED)
            self._badge_detail.config(text="Paused. Press Resume to continue.")

    def resume(self):
        if self._state == self.PAUSED:
            self._set_state(self.RUNNING)
            self._badge_detail.config(text="Simulation running…")
            self._refresh()

    def step_once(self):
        try:
            if self._sig() != self._applied:
                self._rebuild()
            self.sim.step(self.step_dt)
            self._refresh()
        except Exception as e:
            self._badge_detail.config(text=f"Step error: {e}")

    def _tick(self):
        try:
            if self._state == self.RUNNING:
                self.sim.step(self.step_dt)
                self._refresh()
        except Exception as e:
            self._set_state(self.IDLE)
            self._badge_detail.config(text=f"Tick error: {e}")
        self.after(self.update_ms, self._tick)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _init_metrics_state(self) -> None:
        """Running totals _compute_metrics folds new log entries into,
        instead of re-scanning the entire log from t=0 on every single
        call. See _compute_metrics's own comment for why this matters."""
        self._metrics_log_ptr = 0
        self._m_tx = 0
        self._m_rx = 0
        self._m_drop = 0
        self._m_delay_sum = 0.0
        self._m_delay_count = 0
        self._m_delay_max = 0.0
        self._m_delay_min = None
        self._m_node_gen: dict = {}
        self._m_node_del: dict = {}

    def _compute_metrics(self) -> dict:
        # _compute_metrics is called on every GUI tick AND every /api/state
        # poll -- it used to re-walk self.sim.logger.logs (every event ever
        # recorded, from t=0) from scratch each time. That's an ever-
        # growing full rescan on every single call: cheap early on, but it
        # gets slower call over call as the log grows (more sim time, more
        # nodes generating/receiving traffic), which is exactly the "gets
        # sluggish over time / with more nodes" pattern this was profiled
        # for. Only the NEW entries since the last call are processed now;
        # counts/sums are folded into running state (_init_metrics_state)
        # instead of rebuilt from a full delays list each time.
        logger = self.sim.logger
        n = logger.log_count()
        if n < self._metrics_log_ptr:
            self._init_metrics_state()  # log was reset/replaced out from under us
        # entries_from, not the `.logs` property -- `.logs` copies the
        # ENTIRE log on every access regardless of how much of it we
        # actually need; entries_from only copies the new tail.
        new = logger.entries_from(self._metrics_log_ptr)
        self._metrics_log_ptr = n

        for r in new:
            layer = str(r.get("layer", ""))
            event = str(r.get("event", ""))
            det   = r.get("details", {}) or {}
            nid   = r.get("node_id")

            if layer == "APP" and event == "GENERATE":
                self._m_tx += 1
                if nid is not None:
                    self._m_node_gen[nid] = self._m_node_gen.get(nid, 0) + 1

            elif layer == "APP" and event == "DELIVER":
                self._m_rx += 1
                # Per-node "delivered" must be credited to the packet's
                # original SOURCE (whichever STA generated it), not to nid
                # -- nid is whichever node this DELIVER event fires at,
                # which for uplink traffic is always the AP (app.py's
                # on_deliver logs from the *receiving* node's own APP
                # layer). r["src"] is populated by Simulator.log() straight
                # off the Packet object for every call that passes
                # packet=... (see simulator.py's log()), and stays the
                # true original generator across every hop, so it's valid
                # for relay/multi-hop topologies too.
                #
                # The previous version read r.get("packet") to find a
                # "dst" field -- but Simulator.log() flattens packet
                # fields (src/dst/packet_seq/...) onto the log record
                # directly, it never nests a "packet" object/dict on it,
                # so that lookup was always None. It fell through to
                # det.get("dst") (DELIVER's own details dict only ever
                # has delay_s/jitter_s/goodput_bps, also never "dst") and
                # then to nid every single time -- which is why "Packets
                # Delivered per Node" only ever showed activity on node 0
                # (the AP, since every uplink DELIVER event fires there)
                # and every STA's own per-node PDR read 0%, while the AP's
                # entry showed a nonsensical delivered/generated ratio
                # (the AP itself never generates any APP-layer traffic).
                src = r.get("src")
                if src is None:
                    src = nid
                if src is not None:
                    self._m_node_del[src] = self._m_node_del.get(src, 0) + 1
                d = det.get("delay_s")
                if d is not None:
                    try:
                        d = float(d)
                        self._m_delay_sum += d
                        self._m_delay_count += 1
                        if self._m_delay_count == 1 or d > self._m_delay_max:
                            self._m_delay_max = d
                        if self._m_delay_min is None or d < self._m_delay_min:
                            self._m_delay_min = d
                    except Exception:
                        pass

            elif (
                (layer == "APP" and event == "TX_FAILURE")
                or (layer == "MAC" and event == "DROP")
            ):
                self._m_drop += 1

        tx, rx, drop = self._m_tx, self._m_rx, self._m_drop
        pdr      = rx / tx if tx else 0.0
        lat_avg  = self._m_delay_sum / self._m_delay_count if self._m_delay_count else 0.0
        lat_max  = self._m_delay_max if self._m_delay_count else 0.0
        lat_min  = self._m_delay_min if self._m_delay_min is not None else 0.0
        pkt_b    = int(self.sim.config.get("app", {}).get("packet_size_bytes", 0))
        sim_t    = float(getattr(self.sim.engine, "now", 0.0))
        thr_bps  = rx * pkt_b * 8 / max(sim_t, 1e-12)
        drop_r   = drop / max(tx, 1)

        node_pdr: dict = {}
        for nid in set(self._m_node_gen) | set(self._m_node_del):
            g = self._m_node_gen.get(nid, 1) or 1
            node_pdr[nid] = int(self._m_node_del.get(nid, 0) / g * 100)

        return {
            "time": sim_t, "tx": tx, "rx": rx, "drop": drop,
            "pdr": pdr, "lat_avg": lat_avg, "lat_max": lat_max,
            "lat_min": lat_min, "thr_bps": thr_bps, "drop_rate": drop_r,
            "node_gen": dict(self._m_node_gen), "node_del": dict(self._m_node_del),
            "node_pdr": node_pdr,
        }

    def _refresh(self):
        try:
            m = self._compute_metrics()
            t = float(m["time"])
            thr_kbps = m["thr_bps"] / 1000

            # ── Stat cards ────────────────────────────────────────────────────
            self._c_pdr.set(
                f"{m['pdr'] * 100:.2f}%",
                f"TX {m['tx']}   RX {m['rx']}   Drop {m['drop']}",
            )
            self._c_tput.set(
                f"{thr_kbps:.2f} kb/s",
                f"sim time {t:.2f} s",
            )
            self._c_delay.set(
                f"{m['lat_avg'] * 1000:.1f} ms",
                f"max {m['lat_max'] * 1000:.1f} ms   min {m['lat_min'] * 1000:.1f} ms",
            )
            self._c_gen.set(str(m["tx"]), f"across {len(m['node_gen'])} nodes")
            self._c_del.set(str(m["rx"]), f"PDR {m['pdr'] * 100:.1f}%")
            self._c_drop.set(
                f"{m['drop_rate'] * 100:.1f}%",
                f"{m['drop']} packet{'s' if m['drop'] != 1 else ''} dropped",
            )

            # ── Time-series charts ────────────────────────────────────────────
            self._lc_pdr.push(t, m["pdr"], 1.0)
            self._tput_max = max(self._tput_max, m["thr_bps"])
            self._lc_tput.y_max = self._tput_max * 1.25 / 1000  # kb/s
            self._lc_tput.push(t, thr_kbps)
            self._lc_delay.push(t, m["lat_avg"] * 1000.0)
            self._lc_drop.push(t, m["drop_rate"])

            # ── Node charts ───────────────────────────────────────────────────
            self._bc_gen.draw(m["node_gen"])
            self._bc_del.draw(m["node_del"])
            self._bc_pdr.draw(m["node_pdr"])

            # ── Log ───────────────────────────────────────────────────────────
            self._refresh_log()
            self._refresh_trace_table()

            # ── Drone mobility + live packet-transmission animation ────────────
            self._advance_drones()
            self._animate_new_packets()

            # ── Sidebar live stats ────────────────────────────────────────────
            self._hdr_time.config(text=f"T = {t:.3f} s")
            self._sl_time.config(text=f"{t:.3f} s")
            self._sl_pdr.config(text=f"{m['pdr'] * 100:.2f}%")
            self._sl_tput.config(text=f"{thr_kbps:.2f} kb/s")
            self._sl_delay.config(text=f"{m['lat_avg'] * 1000:.1f} ms")
            self._sl_drop.config(text=f"{m['drop_rate'] * 100:.1f}%")
            # log_count(), not len(self.sim.logger.logs) -- the latter
            # copies every record in the log just to measure it.
            log_count = self.sim.logger.log_count()
            self._sl_logs.config(text=str(log_count))

            # ── Status bar ────────────────────────────────────────────────────
            self._sb_time.config(text=f"T = {t:.3f} s")
            self._sb_logs.config(text=f"Logs: {log_count}")
            _state_lbl = {self.RUNNING: "Running", self.PAUSED: "Paused", self.IDLE: "Idle"}
            self._sb_status.config(text=_state_lbl.get(self._state, "Idle"))

        except Exception as e:
            self._badge_detail.config(text=f"Refresh error: {e}")
            raise

    def _advance_drones(self):
        try:
            topo_cfg = self.sim.config.get("topology", {})
            mode = topo_cfg.get("mode")
            relay_ids = set(topo_cfg.get("relay_ids", []))

            # Oval Path shape overrides -- pushed fresh every tick, same
            # "live, no rebuild needed" rule as Flight Speed/Relay Motion
            # below (_drone_racetrack_geometry reads these straight off the
            # sim object). Harmless to set even when Relay Motion is
            # "Random Path" or there are no aerial relays this tick -- nothing
            # reads them in that case.
            self.sim._relay_path_rx_m = float(self._relay_path_rx_var.get() or 0.0)
            self.sim._relay_path_ry_m = float(self._relay_path_ry_var.get() or 0.0)
            self.sim._relay_path_cx_m = float(self._relay_path_cx_var.get() or 0.0)
            self.sim._relay_path_cy_m = float(self._relay_path_cy_var.get() or 0.0)

            # Relay-role motion: only the two modes with *aerial* relays
            # fly them at all -- "relay"/"relay_uav" keep relays grounded
            # at their seeded position for the whole run.
            if mode in ("aerial_relay", "aerial_relay_uav") and relay_ids:
                if self._relay_motion_var.get() == "Random Path":
                    # Same random-waypoint mobility "uav" end-nodes use
                    # below -- roams within a fixed annulus around the AP
                    # (comfortably inside its default PHY range, see
                    # advance_uav_positions/_UAV_ROAM_MAX_M), it just
                    # happens to be a relay-role node wandering instead of
                    # an end-node.
                    self.sim._uav_speed_mps = float(self._uav_speed_var.get())
                    advance_uav_positions(self.sim, relay_ids, float(self.step_dt))
                else:
                    advance_drone_positions(self.sim, relay_ids, float(self.sim.engine.now))

            # STA/end-node motion: any mode whose STAs are UAVs, whether or
            # not relays exist or fly too -- relay_ids is empty in plain
            # "uav" mode, so excluding it there is a no-op. Shares the one
            # UAV Speed slider with the relays' own "Random Path" option
            # above (both read sim._uav_speed_mps fresh each tick), so
            # there's one consistent wandering speed rather than two
            # independent ones to keep in sync.
            if mode in ("uav", "aerial_relay_uav", "relay_uav"):
                uav_ids = set(nid for nid in self.sim.nodes if nid != 0 and nid not in relay_ids)
                if uav_ids:
                    self.sim._uav_speed_mps = float(self._uav_speed_var.get())
                    advance_uav_positions(self.sim, uav_ids, float(self.step_dt))

            # Cars + UAVs multi-AP layout (see sim11ah/topology.py's
            # CarsUavsBuilder): a distinct mode from plain "uav" above --
            # multiple real APs (topo_cfg["ap_ids"]), cars driving a road
            # grid (sim11ah/mobility.py's grid_road_step -- turns onto
            # the next road, never resets, once it reaches the end of
            # its current one) rather than wandering, and UAVs flying a
            # continuous reflecting path across the WHOLE corridor
            # (uav_bounce_step) rather than circling a single AP, so this
            # can't reuse advance_uav_positions above (anchored on node 0
            # specifically). Reuses the existing UAV Speed slider for
            # the UAV leg -- no dedicated Car Speed control exists yet,
            # so highway speed is a fixed, realistic default for now.
            if mode == "cars_uavs":
                car_ids = topo_cfg.get("car_ids", [])
                uav_ids2 = topo_cfg.get("uav_ids", [])
                dt = float(self.step_dt)
                span = float(topo_cfg.get("corridor_span_m", 0.0))
                # A quarter of each fleet STARTS on a cross street (see
                # CarsUavsBuilder.build) instead of an avenue -- north-
                # south instead of east-west -- so vehicles actually move
                # in every direction the road network offers. cross_ids
                # is a SUFFIX of car_ids/scooter_ids (build() appends
                # them last), so set-membership below is exactly "is this
                # one of the last n_cross ids", nothing fuzzier. Only
                # matters for how grid_road_step's very first call for a
                # given vehicle is seeded (init_axis/init_dir) -- every
                # call after that reads back its own persisted turn state
                # instead (sim._grid_road_state), so a vehicle that has
                # since turned onto a different kind of road keeps being
                # driven correctly regardless of which set it started in.
                car_cross_ids = set(topo_cfg.get("car_cross_ids", []))
                cross_off = float(topo_cfg.get("cross_lane_offset_m", 10.0))
                scooter_cross_off = float(topo_cfg.get("scooter_cross_lane_offset_m", 7.0))
                interior_xs = topo_cfg.get("cross_street_xs_interior") or []
                full_xs = topo_cfg.get("cross_street_xs") or [0.0, span]
                boundary_xs = (min(full_xs), max(full_xs))
                # x_stops must include the SAME +/-lane_offset_m values
                # CarsUavsBuilder.build's own _cross_lane_positions placed
                # a fresh cross-street vehicle at (one per interior cross
                # street, both signs -- a real 2-lane road, matching how
                # every avenue already has a +/- pair of its own), plus
                # the two corridor-edge boundary streets bare (a lane
                # offset there would push a vehicle outside [0, span]) --
                # otherwise a vehicle's initial position wouldn't be a
                # valid stop grid_road_step can recognise, breaking the
                # zero-position-jump guarantee from its very first tick.
                # Different per fleet since cars/scooters use different
                # offsets (scooters ride closer to each street's centre).
                def _x_stops(lane_off):
                    pts = set(boundary_xs)
                    for cx in interior_xs:
                        pts.add(cx + lane_off)
                        pts.add(cx - lane_off)
                    return tuple(sorted(pts))
                car_x_stops = _x_stops(cross_off)
                scooter_x_stops = _x_stops(scooter_cross_off)
                # y_stops = every real avenue this fleet has, both signs
                # (car_avenue_offsets_m only stores the magnitude) -- not
                # just the outermost pair -- so a vehicle can turn onto
                # ANY avenue it actually reaches, not only the outer ring.
                # The single fixed "always turn right" rule from the
                # first version of this feature made every vehicle's path
                # converge onto that one shared outer loop after its
                # first lap ("all the vehicles are following the same
                # pattern"); grid_road_step's own random per-intersection
                # turn (see its docstring) is what actually varies routes
                # between vehicles, but it only has variety to pick FROM
                # if there's more than one real stop per axis to offer.
                car_avenue_offsets = topo_cfg.get("car_avenue_offsets_m") or [25.0]
                scooter_avenue_offsets = topo_cfg.get("scooter_avenue_offsets_m") or [15.0]
                car_y_stops = tuple(sorted({o for a in car_avenue_offsets for o in (a, -a)}))
                scooter_y_stops = tuple(sorted({o for a in scooter_avenue_offsets for o in (a, -a)}))
                if car_ids:
                    car_speed_mps = 15.0  # ~54 km/h
                    cross_j = 0
                    for cid in car_ids:
                        if cid not in self.sim.nodes:
                            continue
                        if cid in car_cross_ids:
                            # Same "even index = positive lane, odd =
                            # negative" pairing CarsUavsBuilder.build's
                            # own _cross_lane_positions used to place
                            # these to begin with -- has to match exactly,
                            # or this would seed a vehicle heading the
                            # wrong way relative to which lane it actually
                            # started on.
                            lane_offset = cross_off if cross_j % 2 == 0 else -cross_off
                            cross_j += 1
                            grid_road_step(
                                self.sim, cid, dt, car_speed_mps,
                                x_stops=car_x_stops, y_stops=car_y_stops,
                                init_axis="y", init_dir=lane_offset,
                            )
                        else:
                            lane_y = self.sim.nodes[cid].pos[1]
                            grid_road_step(
                                self.sim, cid, dt, car_speed_mps,
                                x_stops=car_x_stops, y_stops=car_y_stops,
                                init_axis="x", init_dir=lane_y,
                            )
                # Scooters share the exact same grid_road_step primitive
                # as cars (see sim11ah/mobility.py's docstring), just
                # slower and on their own inner lane/offset/y_stops --
                # no separate mobility function needed.
                scooter_ids = topo_cfg.get("scooter_ids", [])
                scooter_cross_ids = set(topo_cfg.get("scooter_cross_ids", []))
                if scooter_ids:
                    scooter_speed_mps = 8.0  # ~29 km/h
                    cross_j = 0
                    for sid in scooter_ids:
                        if sid not in self.sim.nodes:
                            continue
                        if sid in scooter_cross_ids:
                            lane_offset = scooter_cross_off if cross_j % 2 == 0 else -scooter_cross_off
                            cross_j += 1
                            grid_road_step(
                                self.sim, sid, dt, scooter_speed_mps,
                                x_stops=scooter_x_stops, y_stops=scooter_y_stops,
                                init_axis="y", init_dir=lane_offset,
                            )
                        else:
                            lane_y = self.sim.nodes[sid].pos[1]
                            grid_road_step(
                                self.sim, sid, dt, scooter_speed_mps,
                                x_stops=scooter_x_stops, y_stops=scooter_y_stops,
                                init_axis="x", init_dir=lane_y,
                            )
                if uav_ids2:
                    # ap_ids now holds every AP in the 2D grid (columns x
                    # num_ap_rows -- see CarsUavsBuilder.build), not just
                    # the columns -- len(ap_ids) alone overcounts by
                    # num_ap_rows. Read the real per-column ap_spacing_m
                    # topo_cfg already stores (MultiApBuilder.build sets
                    # it directly) and divide the AP count back down to
                    # columns, rather than re-deriving ap_spacing_m from
                    # corridor_span_m/(len(ap_ids)-1) -- that used to
                    # numerically cancel back out to the right answer only
                    # by algebraic coincidence (dividing by the wrong
                    # count then multiplying by wrong_count-1), which is
                    # exactly the kind of fragile derivation not worth
                    # relying on now that it's just as easy to read the
                    # real numbers directly.
                    num_ap_rows = max(1, int(topo_cfg.get("num_ap_rows", 1)))
                    num_aps = max(1, len(topo_cfg.get("ap_ids", [0])) // num_ap_rows)
                    ap_spacing_m = float(topo_cfg.get("ap_spacing_m", 0.0))
                    region = CarsUavsBuilder.uav_region(self.sim, ap_spacing_m=ap_spacing_m, num_aps=num_aps)
                    uav_speed_mps = float(self._uav_speed_var.get())
                    for uid in uav_ids2:
                        if uid in self.sim.nodes:
                            uav_bounce_step(self.sim, uid, dt, uav_speed_mps, region=region)
        except Exception:
            pass

    def _animate_new_packets(self):
        try:
            logger = self.sim.logger
            n = logger.log_count()
            if n < self._pkt_log_ptr:
                self._pkt_log_ptr = 0   # log was reset out from under us
            # entries_from, not `.logs` -- see _compute_metrics's comment;
            # same full-log-copy-on-every-call cost applied here too.
            new = logger.entries_from(self._pkt_log_ptr)
            self._pkt_log_ptr = n

            canvases = self._net_canvases()
            spawned = 0
            for rec in new:
                if rec.get("layer") != "PHY" or rec.get("event") != "TX_START":
                    continue
                tx_id = rec.get("node_id")
                det = rec.get("details") or {}
                rx_id = det.get("rx_id", rec.get("dst"))
                kind = classify_frame_kind(rec.get("ftype"))
                for c in canvases:
                    if rx_id is None or rx_id == -1:
                        c.add_broadcast_pulse(tx_id, kind)
                    else:
                        c.add_packet(tx_id, rx_id, kind)
                spawned += 1
                if spawned >= 40:   # cap spawns/tick so a TX burst stays legible
                    break

            for c in canvases:
                c.tick_animations()
        except Exception:
            pass

    def _refresh_log(self):
        try:
            filt = self._vars["log_filter"].get()
            tail = self.sim.logger.dump_tail(400, human_readable=True)
            lines = tail.splitlines(keepends=True)
            if filt != "ALL":
                lines = [ln for ln in lines if filt.upper() in ln.upper()]

            self._log.delete("1.0", "end")
            for line in lines:
                u = line.upper()
                tag = None
                for t in ("APP", "MAC", "PHY", "NET", "TP"):
                    if t in u:
                        tag = t
                        break
                if "WARN" in u:
                    tag = "WARN"
                if "ERROR" in u or " ERR " in u:
                    tag = "ERR"
                self._log.insert("end", line, (tag,) if tag else ())

            if self._vars["log_autoscroll"].get():
                self._log.see("end")
        except Exception:
            pass

    def _clear_log(self):
        self._log.delete("1.0", "end")

    # ── Tab: Packet Trace ───────────────────────────────────────────────────────
    _TRACE_MAX_ROWS = 3000
    # Outcome-bucketed events (from sim11ah/mac/dcf.py, sim11ah/phy.py's own
    # "event=" / self._log("...") call sites) -- gives the trace table's
    # colour coding real, load-bearing meaning instead of guessing from the
    # event name string at render time.
    _TRACE_OK_EVENTS = {"TX_SUCCESS", "ACK_RX", "ACK_RX_FRAG", "CTS_RX"}
    _TRACE_BAD_EVENTS = {"DROP", "ACK_TIMEOUT", "CTS_TIMEOUT", "RX_BELOW_SENSITIVITY",
                         "RX_LINK_MISSING", "RX_UNSUPPORTED_MODE"}
    _TRACE_RETRY_EVENTS = {"RTS_RETRY"}

    def _build_trace_tab(self, parent):
        toolbar = tk.Frame(parent, bg=P["bg"], pady=6)
        toolbar.pack(fill="x", padx=12)

        tk.Label(toolbar, text="Layer:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10, "bold")).pack(side="left")
        ttk.Combobox(
            toolbar, textvariable=self._vars["trace_filter"],
            values=["ALL", "APP", "MAC", "PHY", "NET", "TP"],
            width=6, state="readonly",
        ).pack(side="left", padx=(4, 10))

        tk.Label(toolbar, text="Node:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10, "bold")).pack(side="left")
        tk.Entry(toolbar, textvariable=self._vars["trace_node"], width=6,
                 bg="white", relief="solid", bd=1).pack(side="left", padx=(4, 10))

        tk.Label(toolbar, text="Search:", bg=P["bg"], fg=P["fg"],
                 font=("Arial", 10, "bold")).pack(side="left")
        tk.Entry(toolbar, textvariable=self._vars["trace_search"], width=14,
                 bg="white", relief="solid", bd=1).pack(side="left", padx=(4, 10))

        ttk.Checkbutton(
            toolbar, text="Auto-scroll",
            variable=self._vars["trace_autoscroll"],
        ).pack(side="left")

        ttk.Button(toolbar, text="Export Trace…", command=self._export_trace_csv
                   ).pack(side="right", padx=(6, 0))
        ttk.Button(toolbar, text="Clear", command=self._clear_trace_table
                   ).pack(side="right")

        # ── Live stats bar: commercial network-simulator trace views (e.g.
        # NetSim's Packet Trace) always pair the table with running
        # counts, not just raw rows -- these track every row actually
        # inserted (i.e. already passed the active filters), reset on Clear.
        stats = tk.Frame(parent, bg=P["side"], pady=4)
        stats.pack(fill="x", padx=12, pady=(0, 6))

        def _stat(label, color):
            f = tk.Frame(stats, bg=P["side"])
            f.pack(side="left", padx=14)
            tk.Label(f, text=label, bg=P["side"], fg=P["side_fg"],
                     font=("Arial", 8, "bold")).pack(anchor="w")
            v = tk.Label(f, text="0", bg=P["side"], fg=color, font=("Courier New", 13, "bold"))
            v.pack(anchor="w")
            return v

        self._trace_stat_total = _stat("TOTAL", "#e2e8f0")
        self._trace_stat_ok    = _stat("SUCCESS", "#86efac")
        self._trace_stat_bad   = _stat("DROP / TIMEOUT", "#fca5a5")
        self._trace_stat_retry = _stat("RETRY", "#fde68a")
        self._trace_stats = {"total": 0, "ok": 0, "bad": 0, "retry": 0}

        tk.Label(toolbar,
                 text="  click a heading to sort  ·  click a row to inspect below",
                 bg=P["bg"], fg=P["muted"], font=("Arial", 9)).pack(side="left")

        # ── Playback bar: step/scrub through the captured trace like a NetSim-
        # style packet animator, replayed after the fact rather than only
        # live -- every control here ends in a real selection_set() on the
        # table, which _on_trace_row_select already turns into both the
        # detail panel AND a real animated flash on the 2D/3D view, so
        # scrubbing genuinely "plays back" the run, not just a table cursor.
        playback = tk.Frame(parent, bg=P["side"], pady=6)
        playback.pack(fill="x", padx=12, pady=(0, 6))

        self._trace_play_btn = ttk.Button(playback, text="▶ Play", width=8,
                                           command=self._toggle_trace_playback)
        self._trace_play_btn.pack(side="left", padx=(4, 4))
        ttk.Button(playback, text="⏮", width=3,
                   command=lambda: self._trace_playback_step(-1)).pack(side="left")
        ttk.Button(playback, text="⏭", width=3,
                   command=lambda: self._trace_playback_step(1)).pack(side="left", padx=(2, 10))

        self._trace_scrub = ttk.Scale(playback, from_=0, to=0, orient="horizontal",
                                       command=self._on_trace_scrub)
        self._trace_scrub.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self._trace_scrub_lbl = tk.Label(playback, text="row 0 / 0", bg=P["side"],
                                          fg=P["side_fg"], font=("Courier New", 9), width=14)
        self._trace_scrub_lbl.pack(side="left")

        ttk.Button(playback, text="⏵ Live", command=self._trace_playback_jump_live
                   ).pack(side="left", padx=(10, 4))

        self._trace_playing = False
        self._trace_scrub_updating = False

        cols = ("time", "node", "dst", "layer", "event", "ftype", "seq")
        headings = {"time": "Time (s)", "node": "Node", "dst": "Dst", "layer": "Layer",
                    "event": "Event", "ftype": "Frame", "seq": "Seq"}
        widths = {"time": 80, "node": 50, "dst": 50, "layer": 55, "event": 140,
                  "ftype": 70, "seq": 60}

        wrap = tk.Frame(parent, bg=P["bg"])
        wrap.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        tv = ttk.Treeview(wrap, columns=cols, show="headings", style="Trace.Treeview",
                           selectmode="browse")
        for c in cols:
            tv.heading(c, text=headings[c],
                       command=lambda _c=c: self._sort_trace_column(_c, False))
            tv.column(c, width=widths[c], anchor="w", stretch=(c == "event"))
        tv.pack(side="left", fill="both", expand=True)

        sy = tk.Scrollbar(wrap, orient="vertical", command=tv.yview)
        sy.pack(side="right", fill="y")
        tv.configure(yscrollcommand=sy.set)

        # Outcome colour (foreground) x zebra stripe (background) -- both
        # combined into one tag per row rather than relying on multi-tag
        # option precedence, which Tk resolves per-option by tag order and
        # is easy to get subtly wrong (e.g. a stripe tag silently winning
        # over the outcome colour on some options but not others).
        base_tags = {
            "":            "#e2e8f0",
            "trace_ok":    "#86efac",
            "trace_bad":   "#fca5a5",
            "trace_retry": "#fde68a",
            "APP": "#86efac", "MAC": "#67e8f9", "PHY": "#fde68a",
            "NET": "#c4b5fd", "TP": "#fda4af",
        }
        stripe_bg = {"even": P["log_bg"], "odd": "#161b22"}
        for base, fg in base_tags.items():
            for parity, bg in stripe_bg.items():
                name = f"{base or 'none'}_{parity}"
                tv.tag_configure(name, foreground=fg, background=bg)

        detail = tk.Text(parent, height=4, bg=P["log_bg"], fg="#e2e8f0",
                          font=("Courier New", 9), relief="flat", wrap="word")
        detail.pack(fill="x", padx=12, pady=(4, 10))
        detail.insert("end", "Select a row to see its full detail fields.")
        detail.configure(state="disabled")

        tv.bind("<<TreeviewSelect>>", self._on_trace_row_select)

        self._trace_tv = tv
        self._trace_detail = detail
        self._trace_records = {}
        self._trace_row_seq = 0

    def _trace_row_outcome_tag(self, event: str, layer: str) -> str:
        eu = event.upper()
        lu = layer.upper()
        if eu in self._TRACE_BAD_EVENTS:
            return "trace_bad"
        if eu in self._TRACE_OK_EVENTS:
            return "trace_ok"
        if eu in self._TRACE_RETRY_EVENTS:
            return "trace_retry"
        if lu in ("APP", "MAC", "PHY", "NET", "TP"):
            return lu
        return ""

    def _refresh_trace_table(self):
        tv = getattr(self, "_trace_tv", None)
        if tv is None:
            return
        try:
            filt = self._vars["trace_filter"].get()
            node_filt = self._vars["trace_node"].get().strip()
            search = self._vars["trace_search"].get().strip().lower()
            logger = self.sim.logger
            n = logger.log_count()
            if n < self._trace_log_ptr:
                self._trace_log_ptr = 0   # log was reset out from under us
            new = logger.entries_from(self._trace_log_ptr)
            self._trace_log_ptr = n

            for rec in new:
                layer = str(rec.get("layer", ""))
                if filt != "ALL" and layer.upper() != filt:
                    continue
                node_id = rec.get("node_id", "")
                if node_filt and str(node_id) != node_filt:
                    continue
                event = str(rec.get("event", ""))
                details = rec.get("details") or {}
                dst = rec.get("dst")
                if dst is None:
                    dst = details.get("rx_id", rec.get("next_hop"))
                seq = rec.get("packet_seq")
                if seq is None:
                    seq = rec.get("frame_seq")
                if seq is None:
                    seq = rec.get("tx_seq")
                if seq is None:
                    seq = rec.get("net_seq")
                ftype = classify_frame_kind(rec.get("ftype")) if rec.get("ftype") else ""

                if search:
                    haystack = " ".join(str(x) for x in (
                        event, ftype, layer, node_id, dst, seq,
                        " ".join(f"{k}={v}" for k, v in details.items()),
                    )).lower()
                    if search not in haystack:
                        continue

                base_tag = self._trace_row_outcome_tag(event, layer)
                parity = "even" if (self._trace_row_seq % 2 == 0) else "odd"
                tag = f"{base_tag or 'none'}_{parity}"

                iid = str(self._trace_row_seq)
                self._trace_row_seq += 1
                tv.insert("", "end", iid=iid, values=(
                    f"{float(rec.get('time', 0.0)):.4f}",
                    node_id,
                    dst if dst is not None else "",
                    layer,
                    event,
                    ftype,
                    seq if seq is not None else "",
                ), tags=(tag,))
                self._trace_records[iid] = rec

                self._trace_stats["total"] += 1
                if base_tag == "trace_ok":
                    self._trace_stats["ok"] += 1
                elif base_tag == "trace_bad":
                    self._trace_stats["bad"] += 1
                elif base_tag == "trace_retry":
                    self._trace_stats["retry"] += 1

            # Cap total rows so a long run doesn't grow the widget (and this
            # dict) unbounded -- same rationale as SimLogger's own
            # max_records cap, just applied to the UI's copy. Stats keep
            # counting every row ever inserted, not just what's still
            # displayed -- they're a running total, the table is a window.
            children = tv.get_children()
            overflow = len(children) - self._TRACE_MAX_ROWS
            if overflow > 0:
                for old_iid in children[:overflow]:
                    self._trace_records.pop(old_iid, None)
                    tv.delete(old_iid)

            if self._vars["trace_autoscroll"].get():
                remaining = tv.get_children()
                if remaining:
                    tv.see(remaining[-1])

            st = self._trace_stats
            self._trace_stat_total.config(text=str(st["total"]))
            self._trace_stat_ok.config(text=str(st["ok"]))
            self._trace_stat_bad.config(text=str(st["bad"]))
            self._trace_stat_retry.config(text=str(st["retry"]))

            # Keep the scrub bar's range current as rows stream in, without
            # yanking the user's own scrub position -- only follow the tail
            # here when they haven't stepped away from it (autoscroll on).
            final_children = tv.get_children()
            if final_children:
                self._trace_scrub_updating = True
                self._trace_scrub.configure(to=max(0, len(final_children) - 1))
                if self._vars["trace_autoscroll"].get():
                    self._trace_scrub.set(len(final_children) - 1)
                    self._trace_scrub_lbl.config(
                        text=f"row {len(final_children)} / {len(final_children)}")
                self._trace_scrub_updating = False
        except Exception:
            pass

    def _clear_trace_table(self):
        tv = getattr(self, "_trace_tv", None)
        if tv is None:
            return
        for iid in tv.get_children():
            tv.delete(iid)
        self._trace_records.clear()
        self._trace_stats = {"total": 0, "ok": 0, "bad": 0, "retry": 0}
        self._trace_stat_total.config(text="0")
        self._trace_stat_ok.config(text="0")
        self._trace_stat_bad.config(text="0")
        self._trace_stat_retry.config(text="0")
        self._trace_playing = False
        self._trace_play_btn.config(text="▶ Play")
        self._trace_scrub_updating = True
        self._trace_scrub.configure(to=0)
        self._trace_scrub.set(0)
        self._trace_scrub_updating = False
        self._trace_scrub_lbl.config(text="row 0 / 0")

    def _export_trace_csv(self):
        tv = getattr(self, "_trace_tv", None)
        if tv is None or not tv.get_children():
            messagebox.showinfo("Export Trace", "No packet trace rows to export yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Packet Trace",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            cols = ("time", "node", "dst", "layer", "event", "ftype", "seq", "details")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for iid in tv.get_children():
                    vals = list(tv.item(iid, "values"))
                    rec = self._trace_records.get(iid) or {}
                    det = rec.get("details") or {}
                    vals.append("; ".join(f"{k}={v}" for k, v in det.items()))
                    w.writerow(vals)
            self._badge_detail.config(text=f"Exported {len(tv.get_children())} trace rows to {path}")
        except Exception as e:
            messagebox.showerror("Export Trace", f"Could not export: {e}")

    def _on_trace_row_select(self, _event=None):
        detail = self._trace_detail
        sel = self._trace_tv.selection()
        detail.configure(state="normal")
        detail.delete("1.0", "end")
        rec = self._trace_records.get(sel[0]) if sel else None
        if rec is not None:
            det = rec.get("details") or {}
            header = (f"node_id={rec.get('node_id')}  src={rec.get('src')}  "
                      f"dst={rec.get('dst')}  ftype={rec.get('ftype')}")
            body = "  ".join(f"{k}={v}" for k, v in det.items()) or "(no extra detail fields)"
            detail.insert("end", header + "\n" + body)
            self._flash_trace_row_on_canvas(rec)
            self._sync_trace_scrub_from_selection()
        else:
            detail.insert("end", "Select a row to see its full detail fields.")
        detail.configure(state="disabled")

    def _flash_trace_row_on_canvas(self, rec) -> None:
        """Replay the selected trace row as a real packet/pulse animation on
        the 2D canvas (and, transitively, the 3D view -- ui/web3d/snapshot.py
        polls the exact same canvas._packets/_pulses this appends to). Same
        add_packet()/add_broadcast_pulse() call _animate_new_packets() makes
        for a live TX_START, so the colour and motion are the real thing,
        not a lookalike -- this is what actually ties "a row in a table" to
        "a specific link lighting up" instead of just describing it in text."""
        if str(rec.get("layer", "")).upper() != "PHY" or str(rec.get("event", "")) != "TX_START":
            return  # only TX_START rows carry a real tx/rx pair worth animating
        tx_id = rec.get("node_id")
        det = rec.get("details") or {}
        rx_id = det.get("rx_id", rec.get("dst"))
        kind = classify_frame_kind(rec.get("ftype"))
        for c in self._net_canvases():
            if rx_id is None or rx_id == -1:
                c.add_broadcast_pulse(tx_id, kind)
            else:
                c.add_packet(tx_id, rx_id, kind)

    # ── Packet Trace: playback / scrub ──────────────────────────────────────
    def _toggle_trace_playback(self):
        self._trace_playing = not self._trace_playing
        self._trace_play_btn.config(text="⏸ Pause" if self._trace_playing else "▶ Play")
        if self._trace_playing:
            self._vars["trace_autoscroll"].set(False)
            self._trace_playback_tick()

    def _trace_playback_tick(self):
        if not self._trace_playing:
            return
        if not self._trace_playback_step(1):
            self._trace_playing = False
            self._trace_play_btn.config(text="▶ Play")
            return
        # ~5.5 rows/sec -- fast enough to feel like a replay, slow enough to
        # actually watch the flash land on the topology view each step.
        self.after(180, self._trace_playback_tick)

    def _trace_playback_step(self, delta: int) -> bool:
        tv = self._trace_tv
        children = tv.get_children()
        if not children:
            return False
        sel = tv.selection()
        if sel and sel[0] in children:
            idx = children.index(sel[0])
        else:
            idx = -1 if delta > 0 else len(children)
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(children):
            return False
        self._vars["trace_autoscroll"].set(False)
        iid = children[new_idx]
        tv.selection_set(iid)
        tv.see(iid)
        return True

    def _on_trace_scrub(self, value):
        if self._trace_scrub_updating:
            return
        tv = self._trace_tv
        children = tv.get_children()
        if not children:
            return
        idx = max(0, min(len(children) - 1, int(round(float(value)))))
        self._vars["trace_autoscroll"].set(False)
        iid = children[idx]
        tv.selection_set(iid)
        tv.see(iid)

    def _sync_trace_scrub_from_selection(self):
        tv = self._trace_tv
        children = tv.get_children()
        sel = tv.selection()
        if not children:
            return
        self._trace_scrub_updating = True
        self._trace_scrub.configure(to=max(0, len(children) - 1))
        if sel and sel[0] in children:
            idx = children.index(sel[0])
            self._trace_scrub.set(idx)
            self._trace_scrub_lbl.config(text=f"row {idx + 1} / {len(children)}")
        self._trace_scrub_updating = False

    def _trace_playback_jump_live(self):
        self._trace_playing = False
        self._trace_play_btn.config(text="▶ Play")
        tv = self._trace_tv
        children = tv.get_children()
        if children:
            tv.selection_set(children[-1])
            tv.see(children[-1])
        self._vars["trace_autoscroll"].set(True)

    def _sort_trace_column(self, col, reverse):
        tv = self._trace_tv
        items = [(tv.set(iid, col), iid) for iid in tv.get_children("")]

        def key(pair):
            v = pair[0]
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, v)

        items.sort(key=key, reverse=reverse)
        for index, (_, iid) in enumerate(items):
            tv.move(iid, "", index)
        tv.heading(col, command=lambda: self._sort_trace_column(col, not reverse))

    # ── Export / Results ──────────────────────────────────────────────────────
    def export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Save Simulator Logs",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            count = self.sim.logger.export_logs_csv(path)
            messagebox.showinfo("Export OK", f"Saved {count} entries to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def run_results(self):
        path = filedialog.asksaveasfilename(
            title="Save Logs for Analysis",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            count = self.sim.logger.export_logs_csv(path)
            if count <= 0:
                messagebox.showwarning("No Logs", "No log entries available.")
                return
            from analysis.plot_simulator_results import ResultsProcessor
            self._badge_detail.config(text="Processing results…")
            self.update_idletasks()
            ResultsProcessor(path).run()
            self._badge_detail.config(text="Results generated.")
            messagebox.showinfo("Results", "Plots generated successfully.")
        except Exception as e:
            self._badge_detail.config(text=f"Results error: {e}")
            messagebox.showerror("Results Error", str(e))

    # ── Backwards-compat alias ────────────────────────────────────────────────
    def _current_settings_signature(self):
        return self._sig()


# ─── Standalone entry-point ───────────────────────────────────────────────────
def _run_main() -> None:
    import sys
    import os

    # Make sure the project root is on sys.path regardless of where this file
    # is run from (e.g. `python ui/dashboard_tk.py` from the project root, or
    # via an absolute path from the IDE).
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from sim11ah.config import default_config
    from sim11ah.simulator import Simulator
    from sim11ah.topology import StarBuilder, RelayBuilder, MultiApBuilder
    from sim11ah.app import (
        PeriodicTraffic, PoissonTraffic, CBRTraffic,
        BurstyTraffic, OnOffTraffic,
    )

    def _make_traffic(cfg, traffic):
        ac = cfg.get("app", {})
        t = str(traffic).lower()
        if t == "periodic":
            return PeriodicTraffic(float(ac.get("periodic_interval", 5.0)))
        if t == "poisson":
            return PoissonTraffic(float(ac.get("poisson_lambda", 0.5)))
        if t == "cbr":
            return CBRTraffic(rate_bps=float(ac.get("cbr_rate_bps", 2000.0)),
                              packet_size_bytes=int(ac.get("packet_size_bytes", 128)))
        if t == "bursty":
            return BurstyTraffic(burst_size=int(ac.get("burst_size", 3)),
                                  intra_gap=float(ac.get("burst_intra_gap_s", 0.01)),
                                  off_time=float(ac.get("burst_off_time_s", 2.0)))
        if t == "onoff":
            return OnOffTraffic(lambda_on=float(ac.get("onoff_lambda_on", 2.0)),
                                on_time=float(ac.get("onoff_on_time_s", 1.0)),
                                off_time=float(ac.get("onoff_off_time_s", 3.0)))
        if t == "video":
            # Matches ApplicationLayer._build_traffic_model's "video" branch
            # (sim11ah/app.py) -- only the interval-generator half; size_mode/
            # size_table/traffic_type were already set correctly on node.app
            # during its own construction and set_traffic_model() (the
            # caller) only replaces the traffic-model object.
            fps = max(0.1, float(ac.get("video_fps", 5.0)))
            return PeriodicTraffic(1.0 / fps, float(ac.get("video_jitter_s", 0.0)))
        return PeriodicTraffic(float(ac.get("periodic_interval", 5.0)))

    def _build_sim(
        num_stas=50, seed=0, traffic="periodic", raw_enable=True,
        raw_policy="static", packet_size=128, packet_interval=5.0,
        topology="star", num_relays=2, freq_mhz=915.0,
        raw_num_groups=4, raw_num_slots=8, raw_slot_duration=0.014,
        video_fps=5.0, app_overrides=None,
        num_aps=2, ap_spacing_m=400.0,
    ):
        # See scripts/main_gui.py's build_sim() for why the effective
        # traffic mode must come from app_overrides (a sensor profile)
        # when one is given, not the plain traffic= argument.
        effective_traffic = str((app_overrides or {}).get("traffic", traffic))
        cfg = default_config(raw_enable=raw_enable, traffic_mode=effective_traffic)
        cfg["mac"]["raw_policy"] = raw_policy
        cfg["mac"]["raw_num_groups"] = int(raw_num_groups)
        cfg["mac"]["raw_num_slots"] = int(raw_num_slots)
        cfg["mac"]["raw_slot_duration"] = float(raw_slot_duration)
        # See scripts/main_gui.py's build_sim() for why this must scale
        # with num_stas / raw_num_groups instead of trusting the static
        # default -- otherwise a low group count with many STAs silently
        # strands STAs outside the covered AID range.
        cfg["mac"]["raw_nodes_per_group"] = max(
            125, -(-int(num_stas) // max(1, int(raw_num_groups)))
        )
        cfg["app"]["packet_size_bytes"] = int(packet_size)
        cfg["app"]["periodic_interval"] = float(packet_interval)
        cfg["app"]["video_fps"] = float(video_fps)
        cfg["phy"]["freq_mhz"] = float(freq_mhz)
        if app_overrides:
            cfg["app"].update(app_overrides)

        sim = Simulator(config=cfg, seed=seed)
        access = {"rate_bps": 300_000, "prop_delay": 3e-4, "per": 0.0}

        if topology in ("relay", "aerial_relay", "aerial_relay_uav", "relay_uav"):
            nr = max(1, int(num_relays))
            ns = max(nr, int(num_stas))
            RelayBuilder.build(
                sim, num_relays=nr, num_stas=ns,
                backhaul_cfg={"rate_bps": 600_000, "prop_delay": 1e-4, "per": 0.0},
                access_cfg=access,
            )
            if topology != "relay":
                sim.config.setdefault("topology", {})["mode"] = topology
        elif topology == "multi_ap":
            MultiApBuilder.build(
                sim, num_aps=max(1, int(num_aps)), ap_spacing_m=float(ap_spacing_m),
                num_stas=int(num_stas), link_cfg=access,
            )
        else:
            StarBuilder.build(sim, num_stas=int(num_stas), link_cfg=access)
            if topology == "uav":
                sim.config.setdefault("topology", {})["mode"] = "uav"

        for nid, node in sim.nodes.items():
            if node.is_ap or node.role == "RELAY":
                node.app.set_traffic_model(None)
            else:
                node.app.set_traffic_model(_make_traffic(cfg, effective_traffic))

        # Every AP starts beaconing with its own phase offset, not just node
        # 0 -- see scripts/main_gui.py's build_sim() for the full rationale
        # (ap_ids only set by MultiApBuilder; ap_start_beacons() is
        # idempotent so this explicit call, run before node.start(), wins
        # over MacLayer.start()'s automatic offset-0.0 follow-up call).
        ap_ids = sim.config.get("topology", {}).get("ap_ids", [0] if 0 in sim.nodes else [])
        beacon_interval = float(cfg["mac"]["beacon_interval"])
        for idx, ap_id in enumerate(ap_ids):
            sim.nodes[ap_id].mac.ap_start_beacons(
                phase_offset_s=idx * beacon_interval / max(1, len(ap_ids))
            )

        for node in sim.nodes.values():
            node.start()
        return sim

    _defaults = {
        "num_stas": 50, "seed": 0, "traffic": "periodic",
        "raw_enable": True, "raw_policy": "static",
        "packet_size": 128, "packet_interval": 5.0,
        "topology": "star", "num_relays": 2, "freq_mhz": 915.0,
    }

    _sim = _build_sim(
        num_stas=_defaults["num_stas"],
        seed=_defaults["seed"],
        traffic=_defaults["traffic"],
        raw_enable=_defaults["raw_enable"],
    )

    gui = Dashboard(sim=_sim, sim_builder=_build_sim, initial_settings=_defaults)

    def _on_close():
        try:
            for n in gui.sim.nodes.values():
                try: n.stop()
                except Exception: pass
            for n in gui.sim.nodes.values():
                try: n.finalize()
                except Exception: pass
        finally:
            gui.destroy()

    gui.protocol("WM_DELETE_WINDOW", _on_close)
    gui.mainloop()


if __name__ == "__main__":
    # Any failure here (or anywhere above, including the imports at the
    # very top of this file) is caught by the sys.excepthook installed
    # near the top -- no separate try/except needed here.
    _run_main()
