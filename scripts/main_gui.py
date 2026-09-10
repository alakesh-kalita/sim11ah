from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder, RelayBuilder, MultiApBuilder
from sim11ah.app import (
    PeriodicTraffic,
    PoissonTraffic,
    CBRTraffic,
    BurstyTraffic,
    OnOffTraffic,
)
from ui.dashboard_tk import Dashboard


def _make_traffic(cfg: dict, traffic: str):
    ac = cfg.get("app", {})
    traffic = str(traffic).lower()
    if traffic == "periodic":
        return PeriodicTraffic(float(ac.get("periodic_interval", 5.0)))
    if traffic == "poisson":
        return PoissonTraffic(float(ac.get("poisson_lambda", 0.5)))
    if traffic == "cbr":
        return CBRTraffic(
            rate_bps=float(ac.get("cbr_rate_bps", 2000.0)),
            packet_size_bytes=int(ac.get("packet_size_bytes", 128)),
        )
    if traffic == "bursty":
        return BurstyTraffic(
            burst_size=int(ac.get("burst_size", 3)),
            intra_gap=float(ac.get("burst_intra_gap_s", 0.01)),
            off_time=float(ac.get("burst_off_time_s", 2.0)),
        )
    if traffic == "onoff":
        return OnOffTraffic(
            lambda_on=float(ac.get("onoff_lambda_on", 2.0)),
            on_time=float(ac.get("onoff_on_time_s", 1.0)),
            off_time=float(ac.get("onoff_off_time_s", 3.0)),
        )
    if traffic == "video":
        # Matches ApplicationLayer._build_traffic_model's "video" branch
        # (sim11ah/app.py) -- only the interval-generator half; size_mode/
        # size_table/traffic_type were already set correctly on node.app
        # during its own construction and set_traffic_model() (the caller
        # of this function) only replaces the traffic-model object, not
        # those other attributes.
        fps = max(0.1, float(ac.get("video_fps", 5.0)))
        return PeriodicTraffic(1.0 / fps, float(ac.get("video_jitter_s", 0.0)))
    return PeriodicTraffic(float(ac.get("periodic_interval", 5.0)))


def build_sim(
    num_stas: int,
    seed: int = 0,
    traffic: str = "periodic",
    raw_enable: bool = True,
    raw_policy: str = "static",
    packet_size: int = 128,
    packet_interval: float = 5.0,
    topology: str = "star",
    num_relays: int = 2,
    freq_mhz: float = 915.0,
    raw_num_groups: int = 4,
    # 8, not 4: matches scripts/compare_raw_policies.py's validated
    # benchmark config. The adaptive/laca RAW policies only ever tune slot
    # *duration*, never slot *count* (see
    # sim11ah/mac/raw_policy_adaptive.py) -- half the slots means roughly
    # double the contending STAs packed into each one, which a 4-slot
    # default was silently doing versus every published benchmark result.
    raw_num_slots: int = 8,
    raw_slot_duration: float = 0.014,
    video_fps: float = 5.0,
    app_overrides: dict | None = None,
    num_aps: int = 2,
    ap_spacing_m: float = 400.0,
):
    # A sensor profile (sim11ah/sensor_profiles.py) carries its own
    # traffic/packet_size_bytes/etc., which should win over the plain
    # traffic/packet_size/packet_interval args below when both are given
    # (the GUI doesn't necessarily keep those in sync with a chosen
    # profile) -- so the *effective* traffic mode is whichever the
    # override dict specifies, falling back to the argument otherwise.
    effective_traffic = str((app_overrides or {}).get("traffic", traffic))
    cfg = default_config(raw_enable=raw_enable, traffic_mode=effective_traffic)
    cfg["mac"]["raw_policy"] = raw_policy
    cfg["mac"]["raw_num_groups"] = int(raw_num_groups)
    # raw_nodes_per_group caps how many AIDs a single fixed group can cover
    # (default_config's own default, 125, is only safe when raw_num_groups
    # stays at its own default of 4 -- 4*125 = 500, matching the GUI's STA
    # dropdown ceiling). Once raw_num_groups is user-adjustable, a low
    # group count with many STAs would otherwise silently strand every STA
    # whose AID falls outside the covered range (permanently
    # RAW_NOT_SCHEDULED, never able to transmit) -- so size it to the
    # actual STA count instead of trusting the static default.
    cfg["mac"]["raw_nodes_per_group"] = max(
        125, -(-int(num_stas) // max(1, int(raw_num_groups)))
    )
    cfg["mac"]["raw_num_slots"] = int(raw_num_slots)
    cfg["mac"]["raw_slot_duration"] = float(raw_slot_duration)
    cfg["app"]["packet_size_bytes"] = int(packet_size)
    cfg["app"]["periodic_interval"] = float(packet_interval)
    cfg["app"]["video_fps"] = float(video_fps)
    cfg["phy"]["freq_mhz"] = float(freq_mhz)
    if app_overrides:
        # A sensor profile's own video_fps (if any) wins over the plain
        # GUI field, same precedence as packet_size/periodic_interval above.
        cfg["app"].update(app_overrides)

    sim = Simulator(config=cfg, seed=seed)

    access_cfg = {
        "rate_bps":   300_000,
        "prop_delay": 3e-4,
        "per":        0.0,
    }

    if topology in ("relay", "aerial_relay", "aerial_relay_uav", "relay_uav"):
        num_relays = max(1, int(num_relays))
        num_stas   = max(num_relays, int(num_stas))
        backhaul_cfg = {
            "rate_bps":   600_000,
            "prop_delay": 1e-4,
            "per":        0.0,
        }
        RelayBuilder.build(
            sim,
            num_relays=num_relays,
            num_stas=num_stas,
            backhaul_cfg=backhaul_cfg,
            access_cfg=access_cfg,
        )
        if topology != "relay":
            # Same AP<->relay<->STA link topology as "relay" in every case
            # below -- only the mobility differs (ui.topology_canvas.
            # advance_drone_positions / advance_uav_positions), all driven
            # from this one mode string by ui.dashboard_tk.Dashboard.
            # _advance_drones:
            #   aerial_relay     -- relays fly a racetrack, STAs grounded
            #   relay_uav        -- relays grounded, STAs fly independently
            #   aerial_relay_uav -- both fly at once
            sim.config.setdefault("topology", {})["mode"] = topology
    elif topology == "multi_ap":
        MultiApBuilder.build(
            sim, num_aps=max(1, int(num_aps)), ap_spacing_m=float(ap_spacing_m),
            num_stas=int(num_stas), link_cfg=access_cfg,
        )
    else:
        StarBuilder.build(sim, num_stas=int(num_stas), link_cfg=access_cfg)
        if topology == "uav":
            # UAVs here are end nodes (plain STAs), NOT relays -- each one
            # associates with and generates traffic directly to/from the AP,
            # same as "star", but the GUI flies every STA on its own
            # independent random-waypoint path around the AP instead of
            # holding it fixed (see ui.topology_canvas.advance_uav_positions).
            sim.config.setdefault("topology", {})["mode"] = "uav"

    for nid, node in sim.nodes.items():
        if node.is_ap or node.role == "RELAY":
            node.app.set_traffic_model(None)
        else:
            node.app.set_traffic_model(_make_traffic(cfg, effective_traffic))

    # Every AP starts beaconing with its own phase offset, not just node 0 --
    # ap_ids is only set by MultiApBuilder (mirrors relay_ids' existing
    # convention); StarBuilder/RelayBuilder topologies fall back to [0],
    # their only AP, unchanged. MUST run BEFORE node.start(): MacLayer.
    # start() itself unconditionally calls ap_start_beacons() (offset 0.0)
    # for every AP, but ap_start_beacons() is now idempotent (see
    # mac/facade.py) so whichever call happens first wins -- this explicit,
    # phase-staggered call must be first so start()'s automatic zero-offset
    # call becomes the suppressed no-op instead of the other way around.
    ap_ids = sim.config.get("topology", {}).get("ap_ids", [0] if 0 in sim.nodes else [])
    beacon_interval = float(cfg["mac"]["beacon_interval"])
    for idx, ap_id in enumerate(ap_ids):
        sim.nodes[ap_id].mac.ap_start_beacons(
            phase_offset_s=idx * beacon_interval / max(1, len(ap_ids))
        )

    for node in sim.nodes.values():
        node.start()

    return sim


if __name__ == "__main__":
    import argparse
    from ui.topology_canvas import load_topology, apply_topology

    parser = argparse.ArgumentParser(description="sim11ah interactive dashboard")
    parser.add_argument(
        "--topology-file", metavar="PATH", default=None,
        help="Launch with a previously saved topology (see the dashboard's "
             "Save/Load Topology buttons) -- node positions and relay "
             "assignment come from the file; num_stas/num_relays/topology "
             "mode are taken from it too. Environment/Layout are unaffected, "
             "so the same saved topology can be run under any of them.",
    )
    args = parser.parse_args()

    initial = {
        "num_stas":        50,
        "seed":            0,
        "traffic":         "periodic",
        "raw_enable":      True,
        "raw_policy":      "static",
        "packet_size":     128,
        "packet_interval": 5.0,
        "topology":        "star",
        "num_relays":      2,
        "freq_mhz":        915.0,
    }

    topology_data = None
    if args.topology_file:
        topology_data = load_topology(args.topology_file)
        initial["num_stas"] = int(topology_data.get("num_stas", initial["num_stas"]))
        initial["num_relays"] = int(topology_data.get("num_relays", initial["num_relays"]))
        initial["topology"] = str(topology_data.get("mode", initial["topology"]))
        initial["relay_placement"] = str(topology_data.get("relay_placement", "optimal"))

    sim = build_sim(
        num_stas=initial["num_stas"],
        seed=initial["seed"],
        traffic=initial["traffic"],
        raw_enable=initial["raw_enable"],
        topology=initial["topology"],
        num_relays=initial["num_relays"],
    )
    if topology_data is not None:
        apply_topology(sim, topology_data)

    gui = Dashboard(sim=sim, sim_builder=build_sim, initial_settings=initial)

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

    try:
        gui.protocol("WM_DELETE_WINDOW", _on_close)
    except Exception:
        pass

    gui.mainloop()
