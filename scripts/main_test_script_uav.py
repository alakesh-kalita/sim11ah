import csv
from pathlib import Path
from typing import Dict

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
from sim11ah.utils.priority_metrics import attach_priority_metrics


# =====================================================
# TRAFFIC MODEL
# =====================================================
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

    if traffic == "bursty":
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

    return PeriodicTraffic(0.5)


# =====================================================
# BUILD SIM
# =====================================================
def build_sim(
    num_stas: int,
    seed: int,
    traffic: str,
    raw_policy: str,
    csv_path: str,
    priority_metrics_enable: bool = True,
) -> Simulator:

    cfg = default_config(raw_enable=True, traffic_mode=traffic)

    # Common experiment config
    cfg["mac"]["raw_policy"] = raw_policy
    cfg["mac"]["cluster_csv_path"] = csv_path
    cfg["mac"]["priority_metrics_enable"] = bool(priority_metrics_enable)

    sim = Simulator(config=cfg, seed=seed)

    attach_priority_metrics(
        sim,
        csv_path=csv_path,
        enabled=bool(cfg["mac"].get("priority_metrics_enable", False)),
    )

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
            node.app.set_traffic_model(_make_traffic_model(cfg, traffic))



    return sim


# =====================================================
# METRICS
# =====================================================
def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def get_summary(sim: Simulator, sim_time: float) -> Dict:
    s = sim.stats

    generated = int(getattr(s, "packets_generated", 0))
    delivered = int(getattr(s, "packets_delivered", 0))

    out = {
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

    if hasattr(sim, "priority_metrics") and sim.priority_metrics.enabled:
        out.update(sim.priority_metrics.summary(sim_time))

    return out


# =====================================================
# MAIN
# =====================================================
def main():
    start_nodes = 100
    end_nodes = 1000
    step = 100

    sim_time = 200.0
    traffic = "periodic"
    seed = 42

    priority_metrics_enable = True

    csv_path = "uav_cluster_data.csv"

    # --------------------------------------------------
    # Generate large CSV only once
    # --------------------------------------------------
    if not Path(csv_path).exists():
        print("Generating large CSV (1200 UAVs)...")
        generate_synthetic_cluster_csv(
            path=csv_path,
            n_uavs=1200,
            n_clusters=12,
            seed=seed,
        )

    results = []
    node_values = list(range(start_nodes, end_nodes + 1, step))

    runs = [
        "cluster_csv",
        "cluster_adaptive",
        "adaptive",
        "static",
    ]

    for num_stas in node_values:
        print("\n==============================")
        print(f"Running for {num_stas} UAVs")
        print("==============================")

        row = {
            "num_stas": int(num_stas),
            "seed": int(seed),
            "traffic": traffic,
            "priority_metrics_enable": int(priority_metrics_enable),
        }

        for policy in runs:
            print(f"\n→ Policy: {policy}")

            sim = build_sim(
                num_stas=num_stas,
                seed=seed,
                traffic=traffic,
                raw_policy=policy,
                csv_path=csv_path,
                priority_metrics_enable=priority_metrics_enable,
            )

            sim.run_and_finalize(sim_time)
            summary = get_summary(sim, sim_time)

            for k, v in summary.items():
                row[f"{policy}_{k}"] = v

            print(
                f"PDR={summary['pdr']:.4f}, "
                f"Thr={summary['throughput']:.4f}, "
                f"Delay={summary['delay']:.6f}"
            )

        results.append(row)

    with open("results_scaling.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Scaling results saved → results_scaling.csv")


if __name__ == "__main__":
    main()