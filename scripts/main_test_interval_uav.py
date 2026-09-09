import argparse
import csv
from pathlib import Path
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

from sim11ah.utils.generate_cluster_csv import generate_synthetic_cluster_csv


def _make_traffic_model(cfg, traffic: str, packet_interval: float):
    app_cfg = cfg.get("app", {})
    traffic = str(traffic).lower()

    if traffic == "periodic":
        return PeriodicTraffic(float(packet_interval))

    if traffic == "poisson":
        lam = 1.0 / max(packet_interval, 1e-12)
        return PoissonTraffic(float(lam))

    if traffic == "cbr":
        pkt_bytes = int(app_cfg.get("packet_size_bytes", 128))
        rate_bps = (pkt_bytes * 8.0) / max(packet_interval, 1e-12)
        return CBRTraffic(
            rate_bps=float(rate_bps),
            packet_size_bytes=pkt_bytes,
        )

    if traffic in ("burst", "bursty"):
        return BurstyTraffic(
            burst_size=int(app_cfg.get("burst_size", 3)),
            intra_gap=float(app_cfg.get("burst_intra_gap_s", 0.01)),
            off_time=float(app_cfg.get("burst_off_time_s", 2.0)),
        )

    if traffic == "onoff":
        return OnOffTraffic(
            lambda_on=float(app_cfg.get("onoff_lambda_on", 2.0)),
            on_time=float(app_cfg.get("onoff_on_time_s", 1.0)),
            off_time=float(app_cfg.get("onoff_off_time_s", 3.0)),
        )

    return PeriodicTraffic(float(packet_interval))


def build_sim(
    num_stas: int,
    seed: int,
    traffic: str,
    raw_enable: bool,
    raw_policy: str,
    packet_interval: float,
    csv_path: str,
) -> Simulator:

    cfg = default_config(raw_enable=raw_enable, traffic_mode=traffic)

    if raw_enable:
        cfg["mac"]["raw_policy"] = raw_policy

    cfg["mac"]["cluster_csv_path"] = csv_path

    cfg["app"]["periodic_interval"] = float(packet_interval)
    cfg["app"]["poisson_lambda"] = 1.0 / max(packet_interval, 1e-12)

    if traffic == "cbr":
        pkt_bytes = int(cfg["app"].get("packet_size_bytes", 128))
        cfg["app"]["cbr_rate_bps"] = float(
            (pkt_bytes * 8.0) / max(packet_interval, 1e-12)
        )

    sim = Simulator(config=cfg, seed=seed)

    StarBuilder.build(
        sim,
        num_stas=int(num_stas),
        link_cfg={
            "rate_bps": 300000,
            "prop_delay": 0.0003,
            "per": 0.0,
        },
    )

    for nid, node in sim.nodes.items():
        if nid == 0:
            node.app.set_traffic_model(None)
        else:
            node.app.set_traffic_model(
                _make_traffic_model(cfg, traffic, packet_interval)
            )



    return sim


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def get_summary(sim: Simulator, sim_time: float) -> Dict:
    s = sim.stats

    generated = int(getattr(s, "packets_generated", 0))
    delivered = int(getattr(s, "packets_delivered", 0))

    return {
        "generated": generated,
        "delivered": delivered,
        "dropped": int(getattr(s, "packets_dropped", 0)),
        "pdr": float(delivered) / max(1, generated),
        "throughput": float(delivered) / max(sim_time, 1e-12),
        "delay": _safe_mean(list(getattr(s, "delays", []))),
        "collisions": int(getattr(s, "phy_collisions", 0)),
        "retries": int(getattr(s, "mac_retries", 0)),
        "ack_timeouts": int(getattr(s, "mac_ack_timeouts", 0)),
        "tx_attempts": int(getattr(s, "mac_tx_attempts", 0)),
    }


def parse_args(argv):
    p = argparse.ArgumentParser()

    p.add_argument("--num-stas", type=int, default=500)
    p.add_argument("--sim-time", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--traffic",
        default="periodic",
        choices=["periodic", "poisson", "cbr", "bursty", "onoff"],
    )

    p.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Packet intervals in seconds",
    )

    p.add_argument("--csv-path", default="uav_cluster_data.csv")
    p.add_argument("--out", default="interval_summary.csv")

    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    csv_path = args.csv_path

    if not Path(csv_path).exists():
        print("Generating large CSV...")
        generate_synthetic_cluster_csv(
            path=csv_path,
            n_uavs=max(1200, args.num_stas),
            n_clusters=12,
            seed=args.seed,
        )

    runs = [
        ("cluster_csv", True, "cluster_csv"),
        ("cluster_adaptive", True, "cluster_adaptive"),
        ("adaptive", True, "adaptive"),
        ("static", True, "static"),
    ]

    metrics = [
        "generated",
        "delivered",
        "dropped",
        "pdr",
        "throughput",
        "delay",
        "collisions",
        "retries",
        "ack_timeouts",
        "tx_attempts",
    ]

    rows: List[Dict] = []

    for interval in args.intervals:
        row = {
            "packet_interval": float(interval),
            "num_stas": int(args.num_stas),
            "seed": int(args.seed),
            "traffic": args.traffic,
        }

        for run_name, raw_enable, raw_policy in runs:
            print(
                f"Running: interval={interval}, "
                f"num_stas={args.num_stas}, policy={run_name}"
            )

            sim = build_sim(
                num_stas=args.num_stas,
                seed=args.seed,
                traffic=args.traffic,
                raw_enable=raw_enable,
                raw_policy=raw_policy,
                packet_interval=float(interval),
                csv_path=csv_path,
            )

            sim.run_and_finalize(args.sim_time)
            summary = get_summary(sim, args.sim_time)

            for m in metrics:
                row[f"{run_name}_{m}"] = summary[m]

            print(
                f"Done: interval={interval}, policy={run_name}, "
                f"PDR={summary['pdr']:.4f}, "
                f"Thr={summary['throughput']:.4f}, "
                f"Delay={summary['delay']:.6f}"
            )

        rows.append(row)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])