"""Launches the Dashboard pre-set to the Cars + UAVs multi-AP layout.

Run this directly from your own terminal (not through an agent's sandboxed
shell) so the process isn't tied to that session's lifetime:

    python3 scripts/launch_cars_uavs.py

3 APs in a line, 24 cars + 16 scooters driving back and forth across a
4-avenue road network (8 distinct car lanes, 8 distinct scooter lanes --
see CarsUavsBuilder's car_avenue_offsets_m/scooter_avenue_offsets_m) that
spans a city footprint 5x the size of this layout's original single-
highway version, 8 UAVs flying random-waypoint across the whole thing --
watch the node colors/edges change as vehicles hand over between APs.
Runs in the Smart City environment for a fuller scene (roads, buildings,
its own decorative background traffic) -- the real simulator vehicles are
visually distinct from that decoration: their own vivid non-green colour
palette, a live peer-link line, a motion trail, and a small status LED,
none of which the background traffic has. Opens the 2D dashboard, the
procedural 3D view, AND a real-map view (MapLibre, real street/satellite
tiles under live AP/vehicle markers -- ui/web3d/static/cars-uavs-map.html)
-- three ways to watch the same live sim state. Ctrl+C or close the
Dashboard window to stop it.

No dropdown exposes this topology yet -- the interactive Network Topology
control only composes star/relay x ground/UAV STA x grounded/aerial relay,
and neither that nor multi_ap (built earlier, also never wired into it) has
a GUI control surface. This launcher is the same "specific preset scene"
pattern launch_military_zone_3d.py already uses for exactly that reason.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.main_gui import build_sim
from ui.dashboard_tk import Dashboard

initial = {
    "num_stas": 1,  # unused by build_sim for topology="cars_uavs" -- num_cars/num_uavs below drive it instead
    "seed": 0, "traffic": "periodic",
    "raw_enable": False,  # forced off for this topology regardless -- see CarsUavsBuilder's docstring
    "raw_policy": "static",
    "packet_size": 128, "packet_interval": 2.0, "freq_mhz": 915.0,
    "topology": "cars_uavs", "num_aps": 3, "ap_spacing_m": 900.0,
    "num_cars": 24, "num_uavs": 8, "num_scooters": 16,
}
sim = build_sim(
    num_stas=initial["num_stas"], seed=initial["seed"], traffic=initial["traffic"],
    raw_enable=initial["raw_enable"], packet_size=initial["packet_size"],
    packet_interval=initial["packet_interval"], freq_mhz=initial["freq_mhz"],
    topology=initial["topology"], num_aps=initial["num_aps"],
    ap_spacing_m=initial["ap_spacing_m"], num_cars=initial["num_cars"],
    num_uavs=initial["num_uavs"], num_scooters=initial["num_scooters"],
)
gui = Dashboard(sim=sim, sim_builder=build_sim, initial_settings=initial)

# Smart City for a much richer scene (roads, buildings, its own background
# traffic) than the flat "Open Area" default -- see this file's module
# docstring for how the real cars/scooters/UAVs stay visually distinct
# from that environment's purely decorative loop traffic.
gui._env_var.set("Smart City")
gui._on_env_change()

# Turbo (30fps/33ms ticks), not the "Normal" (7fps/145ms) default -- with
# 24 cars + 16 scooters + 8 UAVs all moving at once, 7 position updates a
# second reads as visibly choppy motion regardless of how cheap any single
# redraw is (confirmed the bottleneck is tick RATE, not redraw cost -- the
# expensive full background/building redraw only ever fires on real
# topology-changing events, never on a plain animation tick). Every other
# topology in this project defaults to "Normal" because a handful of nodes
# genuinely doesn't need faster updates to look smooth; a highway full of
# vehicles does.
gui._vars["sim_speed"].set("Turbo (30 fps)")
gui._on_speed_change()
gui.update_idletasks()

gui._open_3d_view()  # starts Web3DServer + opens the procedural 3D view in your browser
gui._open_cars_uavs_map_view()  # + a second tab: the real-map twin (ui/web3d/static/cars-uavs-map.html)


def _on_close():
    try:
        for n in gui.sim.nodes.values():
            try:
                n.stop()
            except Exception:
                pass
        for n in gui.sim.nodes.values():
            try:
                n.finalize()
            except Exception:
                pass
    finally:
        gui.destroy()


try:
    gui.protocol("WM_DELETE_WINDOW", _on_close)
except Exception:
    pass

gui.start()
gui.mainloop()
