"""Launches the Dashboard pre-set to Military Zone and opens the 3D view.

Run this directly from your own terminal (not through an agent's sandboxed
shell) so the process isn't tied to that session's lifetime:

    python3 scripts/launch_military_zone_3d.py

Ctrl+C or close the Dashboard window to stop it.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.main_gui import build_sim
from ui.dashboard_tk import Dashboard

initial = {
    "num_stas": 50, "seed": 0, "traffic": "periodic",
    "raw_enable": True, "raw_policy": "static",
    "packet_size": 128, "packet_interval": 5.0,
    "topology": "relay", "num_relays": 2, "freq_mhz": 915.0,
}
# topology="star" (the old default here) never creates relay nodes at all --
# main_gui.build_sim only calls RelayBuilder.build (the thing that actually
# builds AP<->relay<->STA links and routes STA uplinks through a relay) when
# topology is "relay" or "aerial_relay"; num_relays is silently ignored
# otherwise. That's why STAs never associated via a relay before this.
sim = build_sim(num_stas=initial["num_stas"], seed=initial["seed"],
                 traffic=initial["traffic"], raw_enable=initial["raw_enable"],
                 topology=initial["topology"], num_relays=initial["num_relays"])
gui = Dashboard(sim=sim, sim_builder=build_sim, initial_settings=initial)

gui._env_var.set("Military Zone")
gui._on_env_change()
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
