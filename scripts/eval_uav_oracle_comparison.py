"""
Real oracle-baseline comparison for Reviewer 2 Comment 2: quantify the
MAC-layer cost (PDR/throughput/delay) attributable to LSTM mobility-
prediction error, by running the identical RAW-scheduling pipeline against
two REAL cluster CSVs built from the actual trained LSTM + K-Means
pipeline (see uav/build_cluster_csvs.py):

  predicted -- uav_cluster_data_predicted.csv (K-Means on LSTM-PREDICTED
               positions -- this is what the paper's "MA-PRAW" actually runs)
  oracle    -- uav_cluster_data_oracle.csv (K-Means on TRUE future
               positions -- the maximum-gain-from-perfect-knowledge ceiling)

Both CSVs cover the SAME 500 real-pipeline UAVs, same K=6, same snapshot
timestep, same priority/tx_rate/queue_len assignment procedure -- the
ONLY difference between the two conditions is whether clustering saw
predicted or true positions. The gap between them is exactly the
prediction-error cost the reviewer asked to quantify.

N capped at 1000 -- the synthetic pipeline was extended from 500 to 1000
UAVs (uav/data/uav_synthesis_pipeline.py, generate_all(1000); the original
500 are byte-identical to before, confirmed via the fixed-seed sequential
generation loop's determinism) specifically so N could match this
project's usual range without reusing AIDs or breaking the real-position
grounding this experiment exists to provide.

3 seeds (42, 7, 99), interval=5s fixed, matching this project's convention.
Resumable -- existing (condition, policy, num_stas, seed) combos are
skipped on re-run.

Outputs:
  results/uav_oracle_comparison_seeds.csv  -- raw per-seed rows
  results/uav_oracle_comparison.csv        -- mean/ci_lo/ci_hi per point
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import compare_raw_policies as crp  # noqa: E402

SEEDS = [42, 7, 99]
TRAFFIC_INT = 5.0
N_VALUES = [100, 200, 300, 400, 500, 600, 800, 1000]

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONDITIONS = [
    ("predicted", os.path.join(ROOT, "uav_cluster_data_predicted.csv")),
    ("oracle", os.path.join(ROOT, "uav_cluster_data_oracle.csv")),
]

POLICIES = [
    ("cluster_csv", "cluster_csv"),
    ("cluster_adaptive", "cluster_adaptive"),
]

METRIC_FIELDS = [
    "pdr", "tput_kbps", "avg_delay_ms", "p95_delay_ms",
    "drop_rate", "fairness", "col_rate", "retry_rate",
    "assoc_mean_ms", "assoc_frac",
    "e_total_mj", "ee_kbit_j",
]

SEED_FIELDS = ["condition", "policy", "num_stas", "seed"] + METRIC_FIELDS + ["elapsed_s"]
AGG_FIELDS = ["condition", "policy", "num_stas", "n_seeds"]
for _m in METRIC_FIELDS:
    AGG_FIELDS += [f"{_m}_mean", f"{_m}_ci_lo", f"{_m}_ci_hi"]


def run_one(condition: str, csv_path: str, policy: str, num_stas: int, seed: int) -> dict:
    crp.TRAFFIC_INT = TRAFFIC_INT
    crp.CLUSTER_CSV_PATH = csv_path
    t0 = time.time()
    sim = crp.build_and_run(num_stas, seed, True, policy)
    elapsed = time.time() - t0
    full = crp.extract(sim)
    row = {m: full[m] for m in METRIC_FIELDS}
    row["condition"] = condition
    row["policy"] = policy
    row["num_stas"] = num_stas
    row["seed"] = seed
    row["elapsed_s"] = round(elapsed, 2)
    return row


def load_existing(path: str) -> dict:
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["condition"], row["policy"], int(row["num_stas"]), int(row["seed"]))
            done[key] = row
    return done


def write_rows(path: str, fields: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    seeds_path = os.path.join(out_dir, "uav_oracle_comparison_seeds.csv")
    agg_path = os.path.join(out_dir, "uav_oracle_comparison.csv")

    existing = load_existing(seeds_path)
    seed_rows = list(existing.values())
    agg_rows: list = []

    for condition, csv_path in CONDITIONS:
        for policy_label, policy in POLICIES:
            for num_stas in N_VALUES:
                per_seed = []
                for seed in SEEDS:
                    key = (condition, policy_label, num_stas, seed)
                    if key in existing:
                        r = existing[key]
                        row = {**r, "num_stas": int(r["num_stas"]), "seed": int(r["seed"]),
                               **{m: float(r[m]) for m in METRIC_FIELDS}}
                        print(f"condition={condition} policy={policy_label} N={num_stas} "
                              f"seed={seed} -- cached, skipping", flush=True)
                    else:
                        print(f"condition={condition} policy={policy_label} N={num_stas} "
                              f"seed={seed} ...", flush=True)
                        row = run_one(condition, csv_path, policy, num_stas, seed)
                        print(f"  -> pdr={row['pdr']:.4f} delay={row['avg_delay_ms']:.0f}ms "
                              f"assoc_frac={row['assoc_frac']:.3f} elapsed={row['elapsed_s']:.1f}s",
                              flush=True)
                        seed_rows.append(row)
                        write_rows(seeds_path, SEED_FIELDS, seed_rows)
                    per_seed.append(row)

                metrics_only = [{m: row[m] for m in METRIC_FIELDS} for row in per_seed]
                agg = crp._aggregate(metrics_only)
                agg_row = {"condition": condition, "policy": policy_label,
                           "num_stas": num_stas, "n_seeds": len(per_seed)}
                for m in METRIC_FIELDS:
                    agg_row[f"{m}_mean"] = agg[f"{m}_mean"]
                    agg_row[f"{m}_ci_lo"] = agg[f"{m}_ci_low"]
                    agg_row[f"{m}_ci_hi"] = agg[f"{m}_ci_hi"]
                agg_rows.append(agg_row)
                write_rows(agg_path, AGG_FIELDS, agg_rows)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
