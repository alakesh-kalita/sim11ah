"""
Paper experiment script.
Sweeps station count for three scenarios:
  1. No-RAW DCF
  2. Static RAW (4 groups x 4 slots x 7 ms)
Metrics: PDR, throughput_bps, avg_delay_ms, p95_delay_ms,
         fairness, collision_rate, retry_rate, raw_fit_pass_rate
"""

import csv
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import PeriodicTraffic


# -----------------------------------------------------------------------
# Experiment parameters
# -----------------------------------------------------------------------
STA_COUNTS   = [10, 25, 50, 75, 100, 150, 200]
SIM_TIME_S   = 120.0
SEEDS        = [42, 7, 99]          # three seeds → averaged
TRAFFIC_INT  = 5.0                  # periodic: one pkt every 5 s per STA
PKT_SIZE     = 128                  # bytes
LINK_RATE    = 150_000              # MCS0
PROP_DELAY   = 300e-6               # 300 µs ≈ 90 m
OUT_CSV      = os.path.join(os.path.dirname(__file__), "paper_results.csv")

RAW_GROUPS       = 4
RAW_SLOTS        = 4
RAW_SLOT_DUR_S   = 0.007   # 7 ms


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def jain(values):
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    n = len(vals)
    return (sum(vals) ** 2) / (n * sum(v * v for v in vals))


def build_and_run(num_stas, seed, raw_enable, raw_policy="static"):
    cfg = default_config(raw_enable=raw_enable, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"]  = PKT_SIZE

    if raw_enable:
        cfg["mac"]["raw_policy"]        = raw_policy
        cfg["mac"]["raw_num_groups"]    = RAW_GROUPS
        cfg["mac"]["raw_num_slots"]     = RAW_SLOTS
        cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S

    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(sim, num_stas=num_stas,
                      link_cfg={"rate_bps": LINK_RATE,
                                "prop_delay": PROP_DELAY,
                                "per": 0.0})

    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(
            PeriodicTraffic(TRAFFIC_INT) if nid > 0 else None
        )

    sim.run_and_finalize(SIM_TIME_S)
    return sim


def extract(sim):
    s   = sim.stats
    gen = max(1, int(getattr(s, "packets_generated", 0)))
    dly = list(getattr(s, "delays", []))

    # per-STA delivery for fairness
    per_src = dict(getattr(s, "delivered_by_src", {}))
    fair = jain(list(per_src.values()))

    tx   = max(1, int(getattr(s, "mac_tx_attempts", 0)))
    col  = int(getattr(s, "phy_collisions", 0))
    rtr  = int(getattr(s, "mac_retries", 0))

    # RAW fit-pass rate (only meaningful when RAW enabled)
    fp   = int(getattr(s, "raw_fit_pass", 0))
    ff   = int(getattr(s, "raw_fit_fail", 0))
    raw_fit = fp / max(1, fp + ff)

    return {
        "pdr":           getattr(s, "packets_delivered", 0) / gen,
        "tput_bps":      getattr(s, "delivered_bytes",   0) * 8.0 / SIM_TIME_S,
        "avg_delay_ms":  1000.0 * (sum(dly) / len(dly) if dly else 0.0),
        "p95_delay_ms":  1000.0 * percentile(dly, 0.95),
        "fairness":      fair,
        "col_rate":      col / tx,
        "retry_rate":    rtr / tx,
        "raw_fit_pass":  raw_fit,
    }


def average_dicts(dicts):
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------
def main():
    rows = []

    cases = [
        ("no_raw",      False, "none"),
        ("static_raw",  True,  "static"),
    ]

    total = len(STA_COUNTS) * len(cases) * len(SEEDS)
    done  = 0

    for n in STA_COUNTS:
        row = {"num_stas": n}

        for label, raw_en, policy in cases:
            seed_results = []
            for seed in SEEDS:
                done += 1
                print(f"[{done}/{total}] N={n:>3}  case={label:<12}  seed={seed}")
                sim = build_and_run(n, seed, raw_en, policy)
                seed_results.append(extract(sim))

            avg = average_dicts(seed_results)
            for k, v in avg.items():
                row[f"{label}_{k}"] = round(v, 6)

        rows.append(row)
        # print progress line
        nr = row
        print(f"  → N={n}: "
              f"no_raw PDR={nr['no_raw_pdr']:.3f}  "
              f"static_raw PDR={nr['static_raw_pdr']:.3f}  "
              f"no_raw delay={nr['no_raw_avg_delay_ms']:.1f}ms  "
              f"static_raw delay={nr['static_raw_avg_delay_ms']:.1f}ms")

    # save
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUT_CSV}")
    return rows


if __name__ == "__main__":
    main()
