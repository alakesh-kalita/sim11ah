import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt

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
from sim11ah.io_utils import safe_export_csv


# ==============================
# Traffic Model
# ==============================
def _make_traffic_model(cfg, traffic: str):
    app_cfg = cfg.get("app", {})
    traffic = str(traffic).lower()

    if traffic == "periodic":
        return PeriodicTraffic(float(app_cfg.get("periodic_interval", 0.2)))

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


# ==============================
# Build Simulator
# ==============================
def build_sim(num_stas: int, seed: int, traffic: str, raw_enable: bool) -> Simulator:
    cfg = default_config(raw_enable=raw_enable, traffic_mode=traffic)
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

    if 0 in sim.nodes:
        sim.nodes[0].mac.ap_start_beacons()

    for node in sim.nodes.values():
        node.start()

    return sim


# ==============================
# Utilities
# ==============================
def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


# ==============================
# Summary Extraction
# ==============================
def get_summary(sim: Simulator, sim_time: float, label: str) -> Dict:
    s = sim.stats
    return {
        "label": label,
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


# ==============================
# MAC ctx._stats extraction
# ==============================
def aggregate_mac_ctx_stats(sim: Simulator) -> Dict[str, int]:
    agg: Dict[str, int] = {}

    for node in sim.nodes.values():
        ctx = getattr(getattr(node, "mac", None), "ctx", None)
        stats = getattr(ctx, "_stats", None)
        if not isinstance(stats, dict):
            continue

        for k, v in stats.items():
            try:
                agg[k] = agg.get(k, 0) + int(v)
            except Exception:
                pass

    return agg


def per_node_mac_ctx_stats(sim: Simulator) -> List[Dict]:
    rows: List[Dict] = []

    keys_of_interest = [
        "backoff_starts",
        "tx_attempts",
        "ack_rx",
        "ack_timeouts",
        "retry_limit_drops",
        "pending_ttl_expired",
        "tx_retries",
        "tx_drops",
        "txq_tail_drops",
        "msdu_ttl_expired",
        "raw_deferred",
        "backoff_freezes",
        "tx_broadcast",
        "tx_unicast",
        "total_latency_s",
        "latency_samples",
    ]

    for nid, node in sorted(sim.nodes.items()):
        ctx = getattr(getattr(node, "mac", None), "ctx", None)
        stats = getattr(ctx, "_stats", None)
        if not isinstance(stats, dict):
            rows.append({"node_id": nid})
            continue

        row = {"node_id": nid}
        for k in keys_of_interest:
            row[k] = stats.get(k, 0)
        rows.append(row)

    return rows


# ==============================
# Save CSV
# ==============================
def save_summary_csv(summary: Dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def save_per_node(sim: Simulator, path: str) -> None:
    s = sim.stats
    node_ids = sorted(set(s.generated_by_src) | set(s.delivered_by_dst) | set(s.delivered_by_src))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "generated", "delivered_to_node", "delivered_from_src"])
        for nid in node_ids:
            writer.writerow([
                nid,
                int(s.generated_by_src.get(nid, 0)),
                int(s.delivered_by_dst.get(nid, 0)),
                int(s.delivered_by_src.get(nid, 0)),
            ])


def save_mac_ctx_agg_csv(mac_ctx_agg: Dict[str, int], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in sorted(mac_ctx_agg.keys()):
            writer.writerow([k, mac_ctx_agg[k]])


def save_mac_ctx_per_node_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return

    all_fields = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_fields.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


# ==============================
# Plots
# ==============================
def plot_packets(sim: Simulator, path: str) -> None:
    s = sim.stats
    nodes = sorted(set(s.generated_by_src) | set(s.delivered_by_dst))
    if not nodes:
        return

    gen = [s.generated_by_src.get(n, 0) for n in nodes]
    delivered = [s.delivered_by_dst.get(n, 0) for n in nodes]

    x = list(range(len(nodes)))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], gen, width=width, label="Generated")
    plt.bar([i + width / 2 for i in x], delivered, width=width, label="Delivered")
    plt.xticks(x, nodes, rotation=90)
    plt.xlabel("Node")
    plt.ylabel("Packets")
    plt.title("Per-node Packets Generated vs Delivered")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_delay(sim: Simulator, path: str) -> None:
    delays = list(getattr(sim.stats, "delays", []))
    if not delays:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(delays, bins=30)
    plt.xlabel("Delay (s)")
    plt.ylabel("Count")
    plt.title("Packet Delay Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_mac_ctx_agg(mac_ctx_agg: Dict[str, int], path: str) -> None:
    keys = [
        "tx_attempts",
        "ack_timeouts",
        "tx_retries",
        "retry_limit_drops",
        "msdu_ttl_expired",
        "txq_tail_drops",
        "raw_deferred",
        "backoff_freezes",
    ]
    names = [k for k in keys if k in mac_ctx_agg]
    values = [mac_ctx_agg[k] for k in names]

    if not names:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xticks(rotation=30)
    plt.ylabel("Count")
    plt.title("Aggregated MAC ctx Stats")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ==============================
# CLI
# ==============================
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--num-stas", type=int, default=1000)
    p.add_argument("--sim-time", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--traffic",
        default="bursty",
        choices=["periodic", "poisson", "cbr", "bursty", "onoff"],
    )
    p.add_argument("--raw-enable", default="1", choices=["0", "1"])
    p.add_argument("--out", default="results")
    return p.parse_args(argv)


# ==============================
# MAIN
# ==============================
def main(argv):
    args = parse_args(argv)

    root = os.path.join(args.out, f"run_{_timestamp()}")
    _mkdir(root)

    runs = [(False, "RAW_OFF")] if args.raw_enable == "0" else [
        (True, "RAW_ON"),
        #(False, "RAW_OFF"),
    ]

    comparison_rows = []

    for raw, label in runs:
        print(f"\n===== {label} =====")

        sim = build_sim(args.num_stas, args.seed, args.traffic, raw)
        run_dir = os.path.join(root, label.lower())
        _mkdir(run_dir)

        try:
            sim.run(args.sim_time)

            summary = get_summary(sim, args.sim_time, label)
            mac_ctx_agg = aggregate_mac_ctx_stats(sim)
            mac_ctx_rows = per_node_mac_ctx_stats(sim)

            comparison_rows.append(summary)

            # CSV files
            safe_export_csv(sim.logger, os.path.join(run_dir, "logs.csv"))
            save_summary_csv(summary, os.path.join(run_dir, "summary.csv"))
            save_per_node(sim, os.path.join(run_dir, "per_node.csv"))
            save_mac_ctx_agg_csv(mac_ctx_agg, os.path.join(run_dir, "mac_ctx_agg.csv"))
            save_mac_ctx_per_node_csv(mac_ctx_rows, os.path.join(run_dir, "mac_ctx_per_node.csv"))

            # Plots
            plot_packets(sim, os.path.join(run_dir, "packets.png"))
            plot_delay(sim, os.path.join(run_dir, "delay.png"))
            plot_mac_ctx_agg(mac_ctx_agg, os.path.join(run_dir, "mac_ctx_agg.png"))

            # Console summary
            print("Summary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")

            print("\nAggregated MAC ctx stats:")
            important = [
                "raw_groups_built",
                "raw_configs_built",
                "raw_scheduled",
                "raw_not_scheduled",
                "raw_enter_count",
                "raw_exit_count",
                "raw_invalid_slot_window",
                "raw_sleep_transitions",
                "tx_attempts",
                "ack_timeouts",
                "tx_retries",
                "retry_limit_drops",
                "msdu_ttl_expired",
                "txq_tail_drops",
                "raw_deferred",
                "backoff_freezes",
            ]
            for k in important:
                print(f"  {k}: {mac_ctx_agg.get(k, 0)}")

            print(f"\nSaved in: {run_dir}")

        finally:
            for node in sim.nodes.values():
                try:
                    node.stop()
                except Exception:
                    pass

            for node in sim.nodes.values():
                try:
                    node.finalize()
                except Exception:
                    pass

    # comparison summary across runs
    if comparison_rows:
        comp_path = os.path.join(root, "comparison_summary.csv")
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

    print(f"\nAll results stored in: {root}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])