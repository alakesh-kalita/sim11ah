"""
UAV/MC-PRAW Poisson-traffic sweep: mirrors eval_uav_assoc_energy.py's
"vs N" sweep (interval=5s fixed, N=100..1000, 3 seeds, same 7-policy set)
but with Poisson-distributed packet arrivals instead of periodic/
deterministic ones -- PoissonTraffic(1/TRAFFIC_INT) gives the same MEAN
5s interval as the deterministic sweep, so the two are directly comparable
at the same nominal traffic intensity (matches the "Poisson Traffic (mean
5s interval)" panel convention from the earlier TRAW-paper figure set).

Requested explicitly to make a 3-panel "combined" figure (deterministic vs
N, Poisson vs N, vs interval) in that earlier figure's style, for the
current MC-PRAW policy set. This is a genuinely new experiment campaign,
not a replotting of existing data -- compare_raw_policies.build_and_run()
gained an `arrival` parameter for this.

3 seeds (42, 7, 99). Resumable -- existing (policy, num_stas, seed) triples
are skipped on re-run.

Outputs:
  results/uav_assoc_energy_vs_n_poisson_seeds.csv  -- raw per-seed rows
  results/uav_assoc_energy_vs_n_poisson.csv        -- mean/ci_lo/ci_hi per point
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import compare_raw_policies as crp  # noqa: E402

SEEDS = [42, 7, 99]
INTERVAL_FIXED = 5.0
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

SEED_FIELDS = ["policy", "num_stas", "seed"] + RAW_METRIC_FIELDS + ["elapsed_s"]

AGG_FIELDS = ["policy", "num_stas", "n_seeds"]
for _m in RAW_METRIC_FIELDS:
    AGG_FIELDS += [f"{_m}_mean", f"{_m}_ci_lo", f"{_m}_ci_hi"]


def run_one(label: str, raw_enable: bool, policy: str, num_stas: int, seed: int) -> dict:
    crp.TRAFFIC_INT = INTERVAL_FIXED
    t0 = time.time()
    sim = crp.build_and_run(num_stas, seed, raw_enable, policy, arrival="poisson")
    elapsed = time.time() - t0
    row = crp.extract(sim)
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
    seeds_path = os.path.join(out_dir, "uav_assoc_energy_vs_n_poisson_seeds.csv")
    agg_path = os.path.join(out_dir, "uav_assoc_energy_vs_n_poisson.csv")

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
                           **{m: float(row[m]) for m in RAW_METRIC_FIELDS}}
                    print(f"N={num_stas} policy={label} seed={seed} -- cached, skipping", flush=True)
                else:
                    print(f"N={num_stas} policy={label} seed={seed} ...", flush=True)
                    row = run_one(label, raw_enable, policy, num_stas, seed)
                    print(
                        f"  -> pdr={row['pdr']:.4f} assoc_mean={row['assoc_mean_ms']:.1f}ms "
                        f"assoc_frac={row['assoc_frac']:.3f} e_total={row['e_total_mj']:.3f}mJ "
                        f"elapsed={row['elapsed_s']:.1f}s",
                        flush=True,
                    )
                    seed_rows.append(row)
                    write_seed_rows(seeds_path, seed_rows)
                per_seed.append(row)

            metrics_only = [{m: row[m] for m in RAW_METRIC_FIELDS} for row in per_seed]
            agg = crp._aggregate(metrics_only)
            agg_row = {"policy": label, "num_stas": num_stas, "n_seeds": len(per_seed)}
            for m in RAW_METRIC_FIELDS:
                agg_row[f"{m}_mean"] = agg[f"{m}_mean"]
                agg_row[f"{m}_ci_lo"] = agg[f"{m}_ci_low"]
                agg_row[f"{m}_ci_hi"] = agg[f"{m}_ci_hi"]
            agg_rows.append(agg_row)
            write_agg_rows(agg_path, agg_rows)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
