"""
E-TAROA comparison (Tian, Santi, Latre, Famaey, SenSys 2017) against the
policies already implemented in this simulator.

Two sweeps:
  A. vs N (100..1000), interval=5s -- this project's standard light-traffic
     convention, for direct comparability with every other policy already
     benchmarked here.
  B. vs packet interval (5s down to 0.3s, i.e. light to near-saturated),
     N=500 fixed -- matches the load regime the paper's own evaluation
     uses, where E-TAROA's More-Data-driven refinement actually engages
     (confirmed: 0/200 stations trigger it at interval=5s vs 200/200 at
     interval=0.3s under N=200).

Comparison set: no_raw, static, adaptive, laca, chang2019, etaroa --
excludes the UAV-specific cluster_csv/cluster_adaptive policies (those
require an external CSV cluster assignment, not a relevant baseline for a
general AID-range-partitioned algorithm like TAROA/E-TAROA).

3 seeds (42, 7, 99), matching this project's established convention.
Resumable: existing (policy, num_stas, packet_interval, seed) rows are
skipped on re-run.

Outputs:
  results/etaroa_vs_n_seeds.csv        results/etaroa_vs_n.csv
  results/etaroa_vs_interval_seeds.csv results/etaroa_vs_interval.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import compare_raw_policies as crp  # noqa: E402

SEEDS = [42]
N_FIXED = 300
N_VALUES = [100, 200, 400, 600, 800, 1000]
INTERVAL_FIXED = 5.0
INTERVALS = [5.0, 3.0, 2.0, 1.0, 0.5, 0.3]

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
    "pdr", "tput_kbps", "avg_delay_ms", "p95_delay_ms", "drop_rate", "fairness",
    "col_rate", "retry_rate", "raw_fit_pass", "assoc_mean_ms", "assoc_max_ms",
    "assoc_scan_ms", "assoc_auth_ms", "assoc_req_ms", "assoc_frac",
    "e_total_mj", "e_tx_mj", "e_rx_mj", "e_idle_mj", "e_sleep_mj", "e_retx_mj",
    "retx_energy_pct", "avg_tx_retries", "ee_kbit_j", "e_per_pkt_uj",
]

SEED_FIELDS = ["policy", "num_stas", "packet_interval", "seed"] + METRIC_FIELDS + ["elapsed_s"]

AGG_FIELDS = ["policy", "num_stas", "packet_interval", "n_seeds"]
for _m in METRIC_FIELDS:
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


def load_existing(path: str) -> dict:
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["policy"], int(row["num_stas"]), float(row["packet_interval"]), int(row["seed"]))
            done[key] = row
    return done


def write_rows(path: str, fields: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sweep(pairs: list, seeds_path: str, agg_path: str) -> None:
    existing = load_existing(seeds_path)
    seed_rows = list(existing.values())
    agg_rows: list = []

    for num_stas, interval in pairs:
        for label, raw_enable, policy in POLICIES:
            per_seed = []
            for seed in SEEDS:
                key = (label, num_stas, interval, seed)
                if key in existing:
                    r = existing[key]
                    row = {**r, "num_stas": int(r["num_stas"]), "packet_interval": float(r["packet_interval"]),
                           "seed": int(r["seed"]), **{m: float(r[m]) for m in METRIC_FIELDS}}
                    print(f"[{os.path.basename(seeds_path)}] N={num_stas} interval={interval} "
                          f"policy={label} seed={seed} -- cached, skipping", flush=True)
                else:
                    print(f"[{os.path.basename(seeds_path)}] N={num_stas} interval={interval} "
                          f"policy={label} seed={seed} ...", flush=True)
                    row = run_one(label, raw_enable, policy, num_stas, interval, seed)
                    print(f"  -> pdr={row['pdr']:.4f} tput={row['tput_kbps']:.2f}kbps "
                          f"delay={row['avg_delay_ms']:.0f}ms assoc_frac={row['assoc_frac']:.3f} "
                          f"ee={row['ee_kbit_j']:.2f}kbit/J elapsed={row['elapsed_s']:.1f}s", flush=True)
                    seed_rows.append(row)
                    write_rows(seeds_path, SEED_FIELDS, seed_rows)
                per_seed.append(row)

            metrics_only = [{m: row[m] for m in METRIC_FIELDS} for row in per_seed]
            agg = crp._aggregate(metrics_only)
            agg_row = {"policy": label, "num_stas": num_stas, "packet_interval": interval, "n_seeds": len(per_seed)}
            for m in METRIC_FIELDS:
                agg_row[f"{m}_mean"] = agg[f"{m}_mean"]
                agg_row[f"{m}_ci_lo"] = agg[f"{m}_ci_low"]
                agg_row[f"{m}_ci_hi"] = agg[f"{m}_ci_hi"]
            agg_rows.append(agg_row)
            write_rows(agg_path, AGG_FIELDS, agg_rows)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    sweep([(n, INTERVAL_FIXED) for n in N_VALUES],
          os.path.join(out_dir, "etaroa_vs_n_seeds.csv"),
          os.path.join(out_dir, "etaroa_vs_n.csv"))

    sweep([(N_FIXED, iv) for iv in INTERVALS],
          os.path.join(out_dir, "etaroa_vs_interval_seeds.csv"),
          os.path.join(out_dir, "etaroa_vs_interval.csv"))

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
