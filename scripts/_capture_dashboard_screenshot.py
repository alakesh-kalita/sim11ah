"""
One-off utility: launch the real Tkinter dashboard with a live simulation,
position it at a known screen location, and hand control to the mainloop
so screencapture (run externally) can grab real window pixels -- not a
mockup. Not part of the app itself; run manually to refresh
docs/dashboard_screenshot.png or similar.

Usage: python3 scripts/_capture_dashboard_screenshot.py [tab_name]
  tab_name: overview (default) | trace
Writes a marker file /tmp/_dashboard_ready when the window is on-screen
and populated, so the capturing process knows when to shoot.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from main_gui import build_sim
from ui.dashboard_tk import Dashboard

TAB = sys.argv[1] if len(sys.argv) > 1 else "overview"
READY_MARKER = "/tmp/_dashboard_ready"
GEOM_FILE = "/tmp/_dashboard_geom.json"

initial = {
    "num_stas": 60, "seed": 3, "traffic": "periodic", "raw_enable": True,
    "raw_policy": "cluster_adaptive", "packet_size": 128, "packet_interval": 5.0,
    "topology": "star", "num_relays": 2, "freq_mhz": 915.0,
}
sim = build_sim(num_stas=60, seed=3, traffic="periodic", raw_enable=True,
                 topology="star", num_relays=2)

app = Dashboard(sim=sim, sim_builder=build_sim, initial_settings=initial)
app.geometry("1480x920+60+60")
# Force this window above everything else via Tk's own window-level API --
# no macOS Accessibility/Automation permission needed (unlike AppleScript
# window control), since a window raising itself is a normal per-app call.
app.attributes("-topmost", True)
app.lift()
app.focus_force()

try:
    os.remove(READY_MARKER)
except OSError:
    pass
try:
    os.remove(GEOM_FILE)
except OSError:
    pass


def _boot():
    app.start()

    def _after_running():
        if TAB == "trace":
            app._notebook.select(app._tab_trace)
            # Select a real TX_START row so the detail panel + a flashed
            # packet on the canvas are both visible in the shot, not an
            # empty inspector.
            app.update()
            rows = app._trace_tv.get_children()
            for iid in rows:
                rec = app._trace_records.get(iid, {})
                if rec.get("layer") == "PHY" and rec.get("event") == "TX_START":
                    app._trace_tv.selection_set(iid)
                    app._trace_tv.see(iid)
                    app._on_trace_row_select(None)
                    break
        else:
            app._notebook.select(app._tab_overview)
        app.lift()
        app.focus_force()
        app.update()
        # Query the window's REAL on-screen position/size from Tk itself
        # rather than trusting the requested geometry() string blindly --
        # the window manager can adjust it (title bar offset, clamping to
        # stay on-screen, etc).
        geom = {
            "x": app.winfo_rootx(), "y": app.winfo_rooty(),
            "w": app.winfo_width(), "h": app.winfo_height(),
        }
        with open(GEOM_FILE, "w") as f:
            json.dump(geom, f)
        with open(READY_MARKER, "w") as f:
            f.write("ready")

    app.after(3500, _after_running)


app.after(200, _boot)
app.mainloop()
