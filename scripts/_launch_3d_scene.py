"""
One-off utility: launch the dashboard headless (Tk window withdrawn) with
the 3D web server running, set to a specific environment/layout variant,
so an external script can open it in an isolated browser profile and
screenshot the real render. Not part of the app; used to regenerate the
README's Deployment Scenarios gallery after a graphics change.

Usage: python3 scripts/_launch_3d_scene.py <environment> <variant>
Writes the server URL to /tmp/_web3d_url.txt once ready.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from main_gui import build_sim
from ui.dashboard_tk import Dashboard
from ui.web3d.server import Web3DServer

ENVIRONMENT = sys.argv[1]
VARIANT = int(sys.argv[2])
URL_FILE = "/tmp/_web3d_url.txt"

initial = {"num_stas": 60, "seed": 3, "traffic": "periodic", "raw_enable": True,
           "raw_policy": "cluster_adaptive", "packet_size": 128, "packet_interval": 5.0,
           "topology": "star", "num_relays": 2, "freq_mhz": 915.0}
sim = build_sim(num_stas=60, seed=3, traffic="periodic", raw_enable=True,
                 topology="star", num_relays=2)
app = Dashboard(sim=sim, sim_builder=build_sim, initial_settings=initial)
app.withdraw()
app._net_canvas.environment = ENVIRONMENT
app._net_canvas.set_layout_variant(VARIANT)

try:
    os.remove(URL_FILE)
except OSError:
    pass

server = Web3DServer(app)
server.start()
with open(URL_FILE, "w") as f:
    f.write(server.url)


def _boot():
    app.start()


app.after(200, _boot)
app.mainloop()
