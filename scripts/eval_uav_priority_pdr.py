"""
UAV-paper priority-traffic PDR (critical/high/normal), matching
results/priority_pdr_results.csv / fig_uav_priority_pdr_vs_n.pdf, but with
the Cluster-Adaptive beacon-budget-clamp fix applied and 3-seed CIs
(the original was single-seed, pre-fix).

Same 7-policy set, N sweep, interval=5s fixed, 120s sim as
eval_uav_assoc_energy.py -- reuses compare_raw_policies.py's build_and_run
config, with sim11ah.utils.priority_metrics attached before the run.

Resumable: existing (policy, num_stas, seed) rows are skipped on re-run.

Outputs:
  results/uav_priority_pdr_vs_n_seeds.csv  -- raw per-seed rows
  results/uav_priority_pdr_vs_n.csv        -- mean/ci_lo/ci_hi per point
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import compare_raw_policies as crp  # noqa: E402
from sim11ah.utils.priority_metrics import attach_priority_metrics  # noqa: E402
from sim11ah.config import default_config  # noqa: E402
from sim11ah.simulator import Simulator  # noqa: E402
from sim11ah.topology import StarBuilder  # noqa: E402
from sim11ah.app import PeriodicTraffic  # noqa: E402

SEEDS = [42, 7, 99]
INTERVAL = 5.0
N_VALUES = [100, 200, 400, 600, 800, 1000]

POLICIES = [
    ("no_raw",           False, "none"),
    ("static",           True,  "static"),
    ("adaptive",         True,  "adaptive"),
    ("cluster_csv",      True,  "cluster_csv"),
    ("cluster_adaptive", True,  "cluster_adaptive"),
    ("laca",             True,  "laca"),
    ("chang2019",        True,  "chang2019"),
    ("etaroa",           True,  "etaroa"),
]

METRIC_FIELDS = [
    "critical_pdr", "high_pdr", "normal_pdr", "overall_pdr",
    "critical_throughput_share", "high_throughput_share", "normal_throughput_share",
]

SEED_FIELDS = ["policy", "num_stas", "seed"] + METRIC_FIELDS + ["elapsed_s"]

AGG_FIELDS = ["policy", "num_stas", "n_seeds"]
for _m in METRIC_FIELDS:
    AGG_FIELDS += [f"{_m}_mean", f"{_m}_ci_lo", f"{_m}_ci_hi"]


def run_one(label: str, raw_enable: bool, policy: str, num_stas: int, seed: int) -> dict:
    crp.TRAFFIC_INT = INTERVAL

    cfg = default_config(raw_enable=raw_enable, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = INTERVAL
    cfg["app"]["packet_size_bytes"] = crp.PKT_SIZE
    cfg["phy"]["default_mode"] = "MCS0"
    cfg["phy"]["control_mode"] = "MCS0"

    if raw_enable:
        cfg["mac"]["raw_policy"] = policy
        cfg["mac"]["raw_num_slots"] = crp.RAW_NUM_SLOTS
        cfg["mac"]["raw_slot_duration"] = crp.RAW_SLOT_DUR_S
        cfg["mac"]["cluster_csv_path"] = crp.CLUSTER_CSV_PATH

    t0 = time.time()
    sim = Simulator(config=cfg, seed=seed)
    attach_priority_metrics(sim, csv_path=crp.CLUSTER_CSV_PATH, enabled=True)

    StarBuilder.build(
        sim, num_stas=num_stas,
        link_cfg={"rate_bps": crp._MCS_RATES["MCS0"], "prop_delay": crp.PROP_DELAY, "per": 0.0},
    )
    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(PeriodicTraffic(INTERVAL) if nid > 0 else None)

    sim.run_and_finalize(crp.SIM_TIME_S)
    elapsed = time.time() - t0

    summary = sim.priority_metrics.summary(crp.SIM_TIME_S)
    row = {m: summary[m] for m in METRIC_FIELDS if m in summary}

    gen = sum(summary[f"{p}_generated"] for p in ("critical", "high", "normal"))
    delivered = sum(summary[f"{p}_delivered"] for p in ("critical", "high", "normal"))
    row["overall_pdr"] = delivered / max(1, gen)

    row["policy"] = label
    row["num_stas"] = num_stas
    row["seed"] = seed
    row["elapsed_s"] = round(elapsed, 2)
    return row


def load_existing_seed_rows(path: str) -> dict:
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["policy"], int(row["num_stas"]), int(row["seed"]))
            done[key] = row
    return done


def write_seed_rows(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SEED_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_agg_rows(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    seeds_path = os.path.join(out_dir, "uav_priority_pdr_vs_n_seeds.csv")
    agg_path = os.path.join(out_dir, "uav_priority_pdr_vs_n.csv")

    existing = load_existing_seed_rows(seeds_path)
    seed_rows = list(existing.values())
    agg_rows: list = []

    for num_stas in N_VALUES:
        for label, raw_enable, policy in POLICIES:
            per_seed = []
            for seed in SEEDS:
                key = (label, num_stas, seed)
                if key in existing:
                    row = existing[key]
                    row = {**row, "num_stas": int(row["num_stas"]), "seed": int(row["seed"]),
                           **{m: float(row[m]) for m in METRIC_FIELDS}}
                    print(f"N={num_stas} policy={label} seed={seed} -- cached, skipping", flush=True)
                else:
                    print(f"N={num_stas} policy={label} seed={seed} ...", flush=True)
                    row = run_one(label, raw_enable, policy, num_stas, seed)
                    print(
                        f"  -> overall_pdr={row['overall_pdr']:.4f} critical={row['critical_pdr']:.4f} "
                        f"high={row['high_pdr']:.4f} normal={row['normal_pdr']:.4f} "
                        f"elapsed={row['elapsed_s']:.1f}s",
                        flush=True,
                    )
                    seed_rows.append(row)
                    write_seed_rows(seeds_path, seed_rows)
                per_seed.append(row)

            metrics_only = [{m: row[m] for m in METRIC_FIELDS} for row in per_seed]
            agg = crp._aggregate(metrics_only)
            agg_row = {"policy": label, "num_stas": num_stas, "n_seeds": len(per_seed)}
            for m in METRIC_FIELDS:
                agg_row[f"{m}_mean"] = agg[f"{m}_mean"]
                agg_row[f"{m}_ci_lo"] = agg[f"{m}_ci_low"]
                agg_row[f"{m}_ci_hi"] = agg[f"{m}_ci_hi"]
            agg_rows.append(agg_row)
            write_agg_rows(agg_path, agg_rows)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
