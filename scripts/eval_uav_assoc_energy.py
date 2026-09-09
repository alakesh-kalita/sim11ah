"""
UAV-paper RAW policy comparison: association time + energy consumption
========================================================================
The UAV/MA-PRAW paper's policy set (no_raw, static, adaptive, cluster_csv
"Cluster-based", cluster_adaptive "Cluster-Adaptive", laca, chang2019
"TGah" -- see results/vary_interval_results.csv and
results/priority_pdr_results.csv, which back fig_uav_core4_vs_interval.pdf
and fig_uav_core4_vs_n.pdf) was only ever measured for PDR/throughput/
delay. This script reuses the already-audited association + energy
extraction from compare_raw_policies.py (built for the TRAW/TASM-RAW
paper) and runs it for the same UAV policy set and sweeps.

Two sweeps, matching the existing UAV figures:
  A. vary packet interval (1..5 s), N=500 fixed  -> vs fig_uav_core4_vs_interval.pdf
  B. vary N (100..1000), interval=5 s fixed      -> vs fig_uav_core4_vs_n.pdf

3 seeds (42, 7, 99), matching the TRAW/TASM-RAW paper convention
(compare_raw_policies.py). Mean + 95% CI (Student-t) reported per metric.

Outputs (written incrementally, resumable -- existing (policy, num_stas,
packet_interval, seed) triples are skipped on re-run):
  results/uav_assoc_energy_vs_interval_seeds.csv  -- raw per-seed rows
  results/uav_assoc_energy_vs_interval.csv        -- mean/ci_lo/ci_hi per point
  results/uav_assoc_energy_vs_n_seeds.csv
  results/uav_assoc_energy_vs_n.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import compare_raw_policies as crp  # noqa: E402

SEEDS = [42, 7, 99]
N_FIXED = 500
INTERVAL_FIXED = 5.0
INTERVALS = [1.0, 2.0, 3.0, 4.0, 5.0]
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

RAW_METRIC_FIELDS = [
    "pdr", "tput_kbps", "avg_delay_ms", "p95_delay_ms",
    "drop_rate", "fairness", "col_rate", "retry_rate", "raw_fit_pass",
    "assoc_mean_ms", "assoc_max_ms", "assoc_scan_ms", "assoc_auth_ms",
    "assoc_req_ms", "assoc_frac",
    "e_total_mj", "e_tx_mj", "e_rx_mj", "e_idle_mj", "e_sleep_mj",
    "e_retx_mj", "retx_energy_pct", "avg_tx_retries",
    "ee_kbit_j", "e_per_pkt_uj",
]

SEED_FIELDS = ["policy", "num_stas", "packet_interval", "seed"] + RAW_METRIC_FIELDS + ["elapsed_s"]

AGG_FIELDS = ["policy", "num_stas", "packet_interval", "n_seeds"]
for _m in RAW_METRIC_FIELDS:
    AGG_FIELDS += [f"{_m}_mean", f"{_m}_ci_lo", f"{_m}_ci_hi"]


def run_one(label: str, raw_enable: bool, policy: str, num_stas: int, interval: float, seed: int) -> dict:
    crp.TRAFFIC_INT = interval
    t0 = time.time()
    sim = crp.build_and_run(num_stas, seed, raw_enable, policy)
    elapsed = time.time() - t0
    row = crp.extract(sim)
    row["policy"] = label
    row["num_stas"] = num_stas
    row["packet_interval"] = interval
    row["seed"] = seed
    row["elapsed_s"] = round(elapsed, 2)
    return row


def load_existing_seed_rows(path: str) -> dict:
    """Return {(policy, num_stas, packet_interval, seed): row} already on disk."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["policy"], int(row["num_stas"]), float(row["packet_interval"]), int(row["seed"]))
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


def sweep(pairs: list, seeds_path: str, agg_path: str) -> None:
    existing = load_existing_seed_rows(seeds_path)
    seed_rows = list(existing.values())
    agg_rows: list = []

    for num_stas, interval in pairs:
        for label, raw_enable, policy in POLICIES:
            per_seed = []
            for seed in SEEDS:
                key = (label, num_stas, interval, seed)
                if key in existing:
                    row = existing[key]
                    row = {**row, "num_stas": int(row["num_stas"]),
                           "packet_interval": float(row["packet_interval"]),
                           "seed": int(row["seed"]),
                           **{m: float(row[m]) for m in RAW_METRIC_FIELDS}}
                    print(f"[{os.path.basename(seeds_path)}] N={num_stas} interval={interval} "
                          f"policy={label} seed={seed} -- cached, skipping", flush=True)
                else:
                    print(f"[{os.path.basename(seeds_path)}] N={num_stas} interval={interval} "
                          f"policy={label} seed={seed} ...", flush=True)
                    row = run_one(label, raw_enable, policy, num_stas, interval, seed)
                    print(
                        f"  -> pdr={row['pdr']:.4f} assoc_mean={row['assoc_mean_ms']:.1f}ms "
                        f"assoc_frac={row['assoc_frac']:.3f} e_total={row['e_total_mj']:.3f}mJ "
                        f"ee={row['ee_kbit_j']:.2f}kbit/J elapsed={row['elapsed_s']:.1f}s",
                        flush=True,
                    )
                    seed_rows.append(row)
                    write_seed_rows(seeds_path, seed_rows)
                per_seed.append(row)

            metrics_only = [{m: row[m] for m in RAW_METRIC_FIELDS} for row in per_seed]
            agg = crp._aggregate(metrics_only)
            agg_row = {"policy": label, "num_stas": num_stas, "packet_interval": interval,
                       "n_seeds": len(per_seed)}
            for m in RAW_METRIC_FIELDS:
                agg_row[f"{m}_mean"] = agg[f"{m}_mean"]
                agg_row[f"{m}_ci_lo"] = agg[f"{m}_ci_low"]
                agg_row[f"{m}_ci_hi"] = agg[f"{m}_ci_hi"]
            agg_rows.append(agg_row)
            write_agg_rows(agg_path, agg_rows)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    sweep_a_pairs = [(N_FIXED, iv) for iv in INTERVALS]
    sweep(
        sweep_a_pairs,
        os.path.join(out_dir, "uav_assoc_energy_vs_interval_seeds.csv"),
        os.path.join(out_dir, "uav_assoc_energy_vs_interval.csv"),
    )

    sweep_b_pairs = [(n, INTERVAL_FIXED) for n in N_VALUES]
    sweep(
        sweep_b_pairs,
        os.path.join(out_dir, "uav_assoc_energy_vs_n_seeds.csv"),
        os.path.join(out_dir, "uav_assoc_energy_vs_n.csv"),
    )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
