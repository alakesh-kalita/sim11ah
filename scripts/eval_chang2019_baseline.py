"""
Evaluate the chang2019 (regression-based traffic-aware sensor grouping,
Chang et al. 2019) RAW policy as an additional baseline, using the exact
same sweep (N, seeds, traffic, sim time) as compare_rl_trained.py, so its
results can be merged directly with results/rl_multiepisode_comparison.csv.

Outputs
-------
  results/chang2019_comparison.csv       — mean + 95% CI per N
  results/chang2019_comparison_seeds.csv — raw per-seed rows
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import PeriodicTraffic

from compare_rl_trained import (
    STA_COUNTS, SEEDS, TRAFFIC_INT, PKT_SIZE, LINK_RATE, PROP_DELAY,
    RAW_NUM_SLOTS, RAW_SLOT_DUR_S, SIM_TIME_S,
    extract, _aggregate, _ZERO_METRICS,
)

OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_CSV      = os.path.join(OUT_DIR, "chang2019_comparison.csv")
OUT_CSV_SEED = os.path.join(OUT_DIR, "chang2019_comparison_seeds.csv")


def build_and_run(num_stas: int, seed: int) -> "Simulator":
    cfg = default_config(raw_enable=True, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"] = PKT_SIZE
    cfg["mac"]["raw_policy"]        = "chang2019"
    cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
    cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S

    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(
        sim, num_stas=num_stas,
        link_cfg={"rate_bps": LINK_RATE, "prop_delay": PROP_DELAY, "per": 0.0},
    )
    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(PeriodicTraffic(TRAFFIC_INT) if nid > 0 else None)
    sim.run_and_finalize(SIM_TIME_S)
    return sim


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows_agg:  List[Dict] = []
    rows_seed: List[Dict] = []

    total = len(STA_COUNTS) * len(SEEDS)
    done  = 0
    t_start = time.time()

    print("=" * 70)
    print("  chang2019 baseline evaluation")
    print(f"  N={STA_COUNTS}  seeds={SEEDS}  sim={SIM_TIME_S:.0f}s")
    print("=" * 70)

    for n in STA_COUNTS:
        seed_results: List[Dict] = []
        for seed in SEEDS:
            done += 1
            elapsed = time.time() - t_start
            eta     = (elapsed / done) * (total - done) if done > 1 else 0
            print(f"  [{done:>2}/{total}]  N={n:>4}  seed={seed}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
            try:
                sim = build_and_run(n, seed)
                m   = extract(sim)
            except Exception as exc:
                print(f"  ERROR N={n} seed={seed}: {exc}")
                m = dict(_ZERO_METRICS)

            seed_results.append(m)
            rows_seed.append({
                "policy": "chang2019", "n": n, "seed": seed,
                **{k: round(v, 6) for k, v in m.items()},
            })

        agg = _aggregate(seed_results)
        rows_agg.append({"policy": "chang2019", "n": n,
                         **{k: round(v, 6) for k, v in agg.items()}})
        print(f"  N={n:>4}  PDR={agg['pdr_mean']:.4f}  "
              f"tput={agg['tput_kbps_mean']:.1f}kb/s  "
              f"delay={agg['avg_delay_ms_mean']:.0f}ms  "
              f"ee={agg['ee_kbit_j_mean']:.2f}kbit/J")

    wall = time.time() - t_start
    print(f"\n  Total wall-clock: {wall:.1f}s")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
        writer.writeheader()
        writer.writerows(rows_agg)

    with open(OUT_CSV_SEED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_seed[0].keys()))
        writer.writeheader()
        writer.writerows(rows_seed)

    print(f"\n  Results : {OUT_CSV}")
    print(f"  Per-seed: {OUT_CSV_SEED}")


if __name__ == "__main__":
    main()
