"""
Full TII-paper evaluation: all baselines + TRAW, with the AoI / deadline-
violation-rate / association metrics that traw_tii_paper.tex's tables need.

Rebuilds the sweep that originally produced the TRAW / No-RAW / Static /
Adaptive / LACA / TASM-RAW / TRAW numbers in the paper (that run's script
was never saved to this repo — see eval_chang2019_tii.py, which recovered
just the chang2019 column the same way). Uses the exact same sim setup as
eval_chang2019_tii.py / compare_rl_trained.py (same N sweep, seeds,
traffic, sim time) so results are directly comparable across all policies.

Policies (display label, raw_policy key)
-----------------------------------------
  no_raw     none            pure DCF, no RAW
  static     static          fixed equal-duration RAW slots
  adaptive   adaptive        CUSUM-EWMA + Bianchi slot-duration adaptation
  laca       laca            Load-Aware Channel Allocation (Taramit 2023)
  chang2019  chang2019       regression-based traffic-aware grouping (Chang 2019)
  tasm_raw   traffic_split   EWMA/CUSUM split/merge groups (TASM-RAW)
  traw       traffic_aware   proposed: demand-proportional + multi-pass + PRAW

Sweep
-----
  N in {100, 200, 400, 600, 800, 1000} STAs
  3 seeds (42, 7, 99), periodic 5 s traffic, 128 B packets, 120 s sim
  95% CI via Student-t (df = n_seeds - 1)

Outputs
-------
  results/traw_tii_full_comparison.csv       — mean + 95% CI, one row per (policy, N)
  results/traw_tii_full_comparison_seeds.csv — raw per-seed rows
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
    _aggregate,
)
from eval_chang2019_tii import extract_extra, _ZERO_METRICS_EXTRA

OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_CSV      = os.path.join(OUT_DIR, "traw_tii_full_comparison.csv")
OUT_CSV_SEED = os.path.join(OUT_DIR, "traw_tii_full_comparison_seeds.csv")

# ---------------------------------------------------------------------------
# Policy registry: (display_label, raw_enable, policy_key)
# ---------------------------------------------------------------------------
POLICIES: List[tuple] = [
    ("no_raw",    False, "none"),
    ("static",    True,  "static"),
    ("adaptive",  True,  "adaptive"),
    ("laca",      True,  "laca"),
    ("chang2019", True,  "chang2019"),
    ("tasm_raw",  True,  "traffic_split"),
    ("traw",      True,  "traffic_aware"),
]


def build_and_run(num_stas: int, seed: int, raw_enable: bool, policy: str) -> "Simulator":
    cfg = default_config(raw_enable=raw_enable, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"] = PKT_SIZE

    if raw_enable:
        cfg["mac"]["raw_policy"]        = policy
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

    total = len(STA_COUNTS) * len(POLICIES) * len(SEEDS)
    done  = 0
    t_start = time.time()

    print("=" * 78)
    print("  TRAW TII full evaluation: no_raw / static / adaptive / laca / "
          "chang2019 / tasm_raw / traw")
    print(f"  N={STA_COUNTS}  seeds={SEEDS}  sim={SIM_TIME_S:.0f}s  "
          f"({total} runs total)")
    print("=" * 78)

    for n in STA_COUNTS:
        for label, raw_en, policy in POLICIES:
            seed_results: List[Dict] = []
            for seed in SEEDS:
                done += 1
                elapsed = time.time() - t_start
                eta     = (elapsed / done) * (total - done) if done > 1 else 0
                print(f"  [{done:>3}/{total}]  N={n:>4}  policy={label:<10}  "
                      f"seed={seed}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
                try:
                    sim = build_and_run(n, seed, raw_en, policy)
                    m   = extract_extra(sim, n)
                except Exception as exc:
                    print(f"  ERROR  N={n}  {label}  seed={seed}: {exc}")
                    m = dict(_ZERO_METRICS_EXTRA)

                seed_results.append(m)
                rows_seed.append({
                    "policy": label, "n": n, "seed": seed,
                    **{k: round(v, 6) for k, v in m.items()},
                })

            agg = _aggregate(seed_results)
            rows_agg.append({"policy": label, "n": n,
                             **{k: round(v, 6) for k, v in agg.items()}})
            print(f"  N={n:>4}  {label:<10}  PDR={agg['pdr_mean']:.4f}  "
                  f"AoI={agg['aoi_s_mean']:.2f}s  DVR={agg['dvr_pct_mean']:.1f}%  "
                  f"delay={agg['avg_delay_ms_mean']:.0f}ms  "
                  f"ee={agg['ee_kbit_j_mean']:.2f}kbit/J  "
                  f"assoc={agg['assoc_mean_s_mean']:.2f}s/"
                  f"{agg['assoc_max_s_mean']:.2f}s/{agg['assoc_frac_mean']:.3f}")

        # Flush progressively so partial results survive an interruption.
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
            writer.writeheader()
            writer.writerows(rows_agg)
        with open(OUT_CSV_SEED, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_seed[0].keys()))
            writer.writeheader()
            writer.writerows(rows_seed)

    wall = time.time() - t_start
    print(f"\n  Total wall-clock: {wall:.1f}s")
    print(f"  Results : {OUT_CSV}")
    print(f"  Per-seed: {OUT_CSV_SEED}")


if __name__ == "__main__":
    main()
