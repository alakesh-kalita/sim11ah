"""
Multi-AP Roaming Analysis
==========================
3-seed mean +/- 95% CI aggregation over scripts/roam_experiment.py's raw
per-run results, one row per swept speed. Mirrors
scripts/compare_raw_policies.py's _ci_stats/_aggregate (reused by import,
not reimplemented -- same Student-t CI math, same *_mean/*_ci_low/*_ci_hi
column convention, same paired aggregated-CSV + raw-seeds-CSV output
split).

This is the "does the residence-time effect hold up across seeds" checkpoint
for the multi-AP roaming work: scripts/roam_experiment.py already showed the
qualitative signature (PDR degrading as residence time drops) for individual
seeds; this aggregates that across seeds with a real confidence interval
instead of eyeballing single runs.

Outputs
-------
  <out>            — mean + 95% CI for every metric, one row per speed
  <out>_seeds.csv  — raw per-seed data for reproducibility
  Console          — summary table (mean ± half-width) per speed

Usage
-----
  python scripts/roam_analysis.py \
      --sweep-speed 0.5,1,2,5,10,20,40,80,160 \
      --ap-spacing 1600 --seeds 42,7,99 \
      --out results/roam_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.roam_experiment import run_one
from scripts.compare_raw_policies import _ci_stats, _aggregate  # noqa: F401  (_ci_stats used indirectly via _aggregate)

# Metrics that genuinely vary per seed and should be averaged with a CI.
# The rest of RunResult's fields (ap_spacing_m, ap_range_m, overlap_width_m,
# speed_mps, residence_time_s, beacon_interval_s) are constant across seeds
# for a given sweep point -- reported once per row instead of "averaged"
# over identical copies of themselves.
_AGG_METRICS = [
    "pdr_overall", "pdr_crossing", "generated_overall", "generated_crossing",
    "connectivity_gap_s", "link_loss_gap_s", "handover_count",
]


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-AP roaming: 3-seed CI analysis")
    parser.add_argument("--sweep-speed", type=str, default="0.5,1,2,5,10,20,40,80,160",
                         help="Comma-separated STA speeds (m/s) to sweep")
    parser.add_argument("--seeds", type=str, default="42,7,99")
    parser.add_argument("--ap-spacing", type=float, default=1600.0,
                         help="See scripts/roam_experiment.py's docstring for "
                              "why this default is sized against the PHY's "
                              "nominal range.")
    parser.add_argument("--num-aps", type=int, default=2)
    parser.add_argument("--packet-interval", type=float, default=0.5)
    parser.add_argument("--packet-size", type=int, default=128)
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--out", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "results", "roam_comparison.csv"))
    args = parser.parse_args()

    speeds = _parse_float_list(args.sweep_speed)
    seeds = _parse_int_list(args.seeds)

    agg_rows: List[Dict] = []
    seed_rows: List[Dict] = []

    for speed in speeds:
        per_seed_for_agg: List[Dict] = []
        last_run: Dict = {}
        for seed in seeds:
            r = run_one(
                seed=seed, speed_mps=speed, ap_spacing_m=args.ap_spacing,
                num_aps=args.num_aps, packet_interval=args.packet_interval,
                packet_size=args.packet_size, raw_enable=not args.no_raw,
            )
            d = asdict(r)
            seed_rows.append(d)
            per_seed_for_agg.append({k: d[k] for k in _AGG_METRICS})
            last_run = d

        agg = _aggregate(per_seed_for_agg)
        row: Dict = {
            "speed_mps": speed,
            "ap_spacing_m": args.ap_spacing,
            "ap_range_m": last_run["ap_range_m"],
            "overlap_width_m": last_run["overlap_width_m"],
            "residence_time_s": last_run["residence_time_s"],
            "beacon_interval_s": last_run["beacon_interval_s"],
            "n_seeds": len(seeds),
        }
        row.update(agg)
        agg_rows.append(row)

        print(
            f"speed={speed:7.2f} m/s  residence={row['residence_time_s']:7.3f}s  "
            f"pdr_overall={agg['pdr_overall_mean']:.3f} "
            f"[{agg['pdr_overall_ci_low']:.3f},{agg['pdr_overall_ci_hi']:.3f}]  "
            f"pdr_crossing={agg['pdr_crossing_mean']:.3f}  "
            f"handovers={agg['handover_count_mean']:.2f}  "
            f"conn_gap={agg['connectivity_gap_s_mean']:.3f}s"
        )

    out_path = args.out
    out_seed_path = os.path.splitext(out_path)[0] + "_seeds.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    with open(out_seed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)

    print(f"\nWrote {len(agg_rows)} aggregated rows to {out_path}")
    print(f"Wrote {len(seed_rows)} raw per-seed rows to {out_seed_path}")


if __name__ == "__main__":
    main()
