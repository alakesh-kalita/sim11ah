"""Launches the Dashboard pre-set to the Cars + UAVs multi-AP layout.

Run this directly from your own terminal (not through an agent's sandboxed
shell) so the process isn't tied to that session's lifetime:

    python3 scripts/launch_cars_uavs.py

3 APs in a line, 4 cars driving a highway back and forth through all of
them, 3 UAVs flying random-waypoint across the whole corridor -- watch the
node colors/edges change as cars and UAVs hand over between APs. Opens
both the 2D dashboard and the live 3D view (a browser tab) -- same layout,
same live sim state, two ways to watch it. Ctrl+C or close the Dashboard
window to stop it.

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
    "num_cars": 4, "num_uavs": 3,
}
sim = build_sim(
    num_stas=initial["num_stas"], seed=initial["seed"], traffic=initial["traffic"],
    raw_enable=initial["raw_enable"], packet_size=initial["packet_size"],
    packet_interval=initial["packet_interval"], freq_mhz=initial["freq_mhz"],
    topology=initial["topology"], num_aps=initial["num_aps"],
    ap_spacing_m=initial["ap_spacing_m"], num_cars=initial["num_cars"],
    num_uavs=initial["num_uavs"],
)
gui = Dashboard(sim=sim, sim_builder=build_sim, initial_settings=initial)
gui.update_idletasks()

gui._open_3d_view()  # starts Web3DServer + opens the 3D view in your browser


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
