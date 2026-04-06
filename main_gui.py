from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import (
    PeriodicTraffic,
    PoissonTraffic,
    CBRTraffic,
    BurstyTraffic,
    OnOffTraffic,
)
from ui.dashboard_tk import Dashboard


def _make_traffic_model(cfg, traffic: str):
    app_cfg = cfg.get("app", {})
    traffic = str(traffic).lower()

    if traffic == "periodic":
        return PeriodicTraffic(float(app_cfg.get("periodic_interval", 0.2)))

    if traffic == "poisson":
        return PoissonTraffic(float(app_cfg.get("poisson_lambda", 5.0)))

    if traffic == "cbr":
        rate_bps = float(app_cfg.get("cbr_rate_bps", 20_000.0))
        pkt_bytes = int(app_cfg.get("packet_size_bytes", 128))
        return CBRTraffic(rate_bps=rate_bps, packet_size_bytes=pkt_bytes)

    if traffic == "bursty":
        burst_size = int(app_cfg.get("burst_size", 10))
        intra_gap = float(app_cfg.get("burst_intra_gap_s", 0.001))
        off_time = float(app_cfg.get("burst_off_time_s", 1.0))
        return BurstyTraffic(
            burst_size=burst_size,
            intra_gap=intra_gap,
            off_time=off_time
        )

    if traffic == "onoff":
        lam_on = float(app_cfg.get("onoff_lambda_on", 20.0))
        on_time = float(app_cfg.get("onoff_on_time_s", 2.0))
        off_time = float(app_cfg.get("onoff_off_time_s", 2.0))
        return OnOffTraffic(
            lambda_on=lam_on,
            on_time=on_time,
            off_time=off_time
        )

    return PeriodicTraffic(float(app_cfg.get("periodic_interval", 0.2)))


def build_sim(num_stas, seed=0, traffic="periodic", raw_enable=True):
    cfg = default_config(raw_enable=raw_enable, traffic_mode=traffic)
    sim = Simulator(config=cfg, seed=seed)

    link_cfg = {
        "rate_bps": 300_000,
        "prop_delay": 0.0003,
        "per": 0.0,
    }
    StarBuilder.build(sim, num_stas=int(num_stas), link_cfg=link_cfg)

    for nid, node in sim.nodes.items():
        if nid == 0:
            node.app.set_traffic_model(None)
        else:
            tm = _make_traffic_model(cfg, traffic)
            node.app.set_traffic_model(tm)

    if 0 in sim.nodes:
        sim.nodes[0].mac.ap_start_beacons()

    for node in sim.nodes.values():
        node.start()

    return sim


if __name__ == "__main__":
    initial_settings = {
        "num_stas": 50,
        "seed": 0,
        "traffic": "periodic",
        "raw_enable": True,
    }

    sim = build_sim(
        num_stas=initial_settings["num_stas"],
        seed=initial_settings["seed"],
        traffic=initial_settings["traffic"],
        raw_enable=initial_settings["raw_enable"],
    )

    gui = Dashboard(
        sim=sim,
        sim_builder=build_sim,
        initial_settings=initial_settings,
    )

    def _on_close():
        try:
            current_sim = gui.sim
            for n in current_sim.nodes.values():
                try:
                    n.stop()
                except Exception:
                    pass
            for n in current_sim.nodes.values():
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

    gui.mainloop()