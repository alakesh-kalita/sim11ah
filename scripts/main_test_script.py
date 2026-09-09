import argparse
import csv
from typing import Dict, List

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


def _make_traffic_model(cfg, traffic: str):
    app_cfg = cfg.get("app", {})
    traffic = str(traffic).lower()

    if traffic == "periodic":
        return PeriodicTraffic(float(app_cfg.get("periodic_interval", 0.5)))

    if traffic == "poisson":
        return PoissonTraffic(float(app_cfg.get("poisson_lambda", 5.0)))

    if traffic == "cbr":
        return CBRTraffic(
            rate_bps=float(app_cfg.get("cbr_rate_bps", 20000)),
            packet_size_bytes=int(app_cfg.get("packet_size_bytes", 128)),
        )

    if traffic in ("burst", "bursty"):
        return BurstyTraffic(
            burst_size=int(app_cfg.get("burst_size", 10)),
            intra_gap=float(app_cfg.get("burst_intra_gap_s", 0.001)),
            off_time=float(app_cfg.get("burst_off_time_s", 1.0)),
        )

    if traffic == "onoff":
        return OnOffTraffic(
            lambda_on=float(app_cfg.get("onoff_lambda_on", 20.0)),
            on_time=float(app_cfg.get("onoff_on_time_s", 2.0)),
            off_time=float(app_cfg.get("onoff_off_time_s", 2.0)),
        )

    return PeriodicTraffic(float(app_cfg.get("periodic_interval", 0.2)))


def build_sim(
    num_stas: int,
    seed: int,
    traffic: str,
    raw_enable: bool,
    raw_policy: str = "none",
) -> Simulator:
    cfg = default_config(raw_enable=raw_enable, traffic_mode=traffic)

    # Override config here
    if raw_enable:
        cfg["mac"]["raw_policy"] = raw_policy

    sim = Simulator(config=cfg, seed=seed)

    StarBuilder.build(
        sim,
        num_stas=int(num_stas),
        link_cfg={"rate_bps": 300000, "prop_delay": 0.0003, "per": 0.0},
    )

    for nid, node in sim.nodes.items():
        if nid == 0:
            node.app.set_traffic_model(None)
        else:
            node.app.set_traffic_model(_make_traffic_model(cfg, traffic))

    return sim


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def get_summary(sim: Simulator, sim_time: float) -> Dict:
    s = sim.stats
    return {
        "generated": int(getattr(s, "packets_generated", 0)),
        "delivered": int(getattr(s, "packets_delivered", 0)),
        "dropped": int(getattr(s, "packets_dropped", 0)),
        "pdr": float(getattr(s, "packets_delivered", 0)) / max(1, int(getattr(s, "packets_generated", 0))),
        "throughput": float(getattr(s, "packets_delivered", 0)) / max(sim_time, 1e-12),
        "avg_delay": _safe_mean(list(getattr(s, "delays", []))),
        "retries": int(getattr(s, "mac_retries", 0)),
        "ack_timeouts": int(getattr(s, "mac_ack_timeouts", 0)),
        "tx_attempts": int(getattr(s, "mac_tx_attempts", 0)),
        "collisions": int(getattr(s, "phy_collisions", 0)),
        "per_drops": int(getattr(s, "phy_per_drops", 0)),
        "half_duplex_collisions": int(getattr(s, "phy_half_duplex_collisions", 0)),
        "below_sensitivity": int(getattr(s, "phy_below_sensitivity", 0)),
        "unsupported_mode": int(getattr(s, "phy_unsupported_mode", 0)),
        "net_duplicates": int(getattr(s, "net_duplicates", 0)),
        "transport_duplicates": int(getattr(s, "transport_duplicates", 0)),
        "net_forwarded": int(getattr(s, "net_forwarded", 0)),
        "app_in_flight_timeouts": int(getattr(s, "app_in_flight_timeouts", 0)),
    }


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--start-nodes", type=int, default=100)
    p.add_argument("--end-nodes", type=int, default=1200)
    p.add_argument("--step-nodes", type=int, default=100)
    p.add_argument("--sim-time", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--traffic",
        default="periodic",
        choices=["periodic", "poisson", "cbr", "bursty", "onoff"],
    )
    p.add_argument("--out", default="summary.csv")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    node_values = list(range(args.start_nodes, args.end_nodes + 1, args.step_nodes))
    rows: List[Dict] = []

    runs = [
        ("raw_on_adaptive", True, "cluster_csv"),
        ("raw_on_static", True, "static"),
        #("raw_off", False, "none"),
    ]

    metrics = [
        "generated",
        "delivered",
        "dropped",
        "pdr",
        "throughput",
        "avg_delay",
        "retries",
        "ack_timeouts",
        "tx_attempts",
        "collisions",
        "per_drops",
        "half_duplex_collisions",
        "below_sensitivity",
        "unsupported_mode",
        "net_duplicates",
        "transport_duplicates",
        "net_forwarded",
        "app_in_flight_timeouts",
    ]

    for num_stas in node_values:
        row = {
            "num_stas": num_stas,
            "seed": args.seed,
            "traffic": args.traffic,
        }

        for run_name, raw_enable, raw_policy in runs:
            print(f"Running: num_stas={num_stas}, case={run_name}")

            sim = build_sim(
                num_stas=num_stas,
                seed=args.seed,
                traffic=args.traffic,
                raw_enable=raw_enable,
                raw_policy=raw_policy,
            )

            sim.run_and_finalize(args.sim_time)
            summary = get_summary(sim, args.sim_time)

            for m in metrics:
                row[f"{run_name}_{m}"] = summary[m]

            print(
                f"Done: num_stas={num_stas}, case={run_name}, "
                f"PDR={summary['pdr']:.6f}, "
                f"Throughput={summary['throughput']:.6f}, "
                f"AvgDelay={summary['avg_delay']:.6f}"
            )

        rows.append(row)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved only: {args.out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])