"""
Evaluate the chang2019 (regression-based traffic-aware sensor grouping,
Chang et al. 2019) RAW policy under the exact simulation setup used in
paper/traw_tii_paper.tex (same N sweep, seeds, traffic, sim time as
compare_rl_trained.py / eval_chang2019_baseline.py), additionally computing
the metrics reported in that paper but not in
results/chang2019_comparison.csv:

  - Mean Age of Information (sawtooth area-integration over per-source
    delivery timelines, Table tab:aoi)
  - Deadline Violation Rate (Pr[delay > T_rep], Table tab:dvr)
  - Association (commissioning) time: mean, max, success fraction
    (Table tab:assoc)

Outputs
-------
  results/chang2019_tii_comparison.csv       — mean + 95% CI per N
  results/chang2019_tii_comparison_seeds.csv — raw per-seed rows
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
    extract, _aggregate,
)

OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_CSV      = os.path.join(OUT_DIR, "chang2019_tii_comparison.csv")
OUT_CSV_SEED = os.path.join(OUT_DIR, "chang2019_tii_comparison_seeds.csv")

T_REP = TRAFFIC_INT  # 5.0 s reporting deadline


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


def _aoi_mean(sim: "Simulator") -> float:
    """Mean AoI (s) averaged over per-source sawtooth area-integration."""
    s = sim.stats
    T = SIM_TIME_S
    n_stas = max(1, len(sim.nodes) - 1)

    per_src_means: List[float] = []
    for nid in sim.nodes:
        if nid == 0:
            continue
        records = s.aoi_records.get(nid, [])
        if not records:
            per_src_means.append(T / 2.0)
            continue
        records = sorted(records, key=lambda r: r[1])
        # first segment [0, r_1): AoI(t) = t, assuming AoI(0) = 0
        area = records[0][1] ** 2 / 2.0
        # remaining segments: [r_i, r_{i+1}) (or T for the last), AoI(t) = t - g_i
        for i in range(len(records)):
            gen_time, recv_time = records[i]
            d_i = recv_time - gen_time
            next_recv = records[i + 1][1] if i + 1 < len(records) else T
            area += ((next_recv - gen_time) ** 2 - d_i ** 2) / 2.0
        per_src_means.append(area / T)

    return sum(per_src_means) / len(per_src_means) if per_src_means else 0.0


def _dvr_pct(sim: "Simulator") -> float:
    """Deadline violation rate (%): Pr[delay > T_rep], dropped pkts count as violations."""
    s   = sim.stats
    gen = int(getattr(s, "packets_generated", 0))
    if gen == 0:
        return 0.0
    dly = list(getattr(s, "delays", []))
    n_violated_delivered = sum(1 for d in dly if d > T_REP)
    n_dropped = gen - int(getattr(s, "packets_delivered", 0))
    return 100.0 * (n_dropped + n_violated_delivered) / gen


def _assoc_stats(sim: "Simulator", num_stas: int) -> Dict[str, float]:
    s = sim.stats
    vals = [v for nid, v in s.assoc_total_time.items() if nid != 0]
    if not vals:
        return {"assoc_mean_s": 0.0, "assoc_max_s": 0.0, "assoc_frac": 0.0}
    return {
        "assoc_mean_s": sum(vals) / len(vals),
        "assoc_max_s":  max(vals),
        "assoc_frac":   len(vals) / num_stas,
    }


def extract_extra(sim: "Simulator", num_stas: int) -> Dict[str, float]:
    out = extract(sim)
    out["aoi_s"]   = _aoi_mean(sim)
    out["dvr_pct"] = _dvr_pct(sim)
    out.update(_assoc_stats(sim, num_stas))
    return out


_ZERO_METRICS_EXTRA = {
    "pdr": 0.0, "tput_kbps": 0.0, "avg_delay_ms": 0.0, "p95_delay_ms": 0.0,
    "drop_rate": 1.0, "fairness": 0.0, "col_rate": 0.0, "retry_rate": 0.0,
    "raw_fit_pass": 0.0, "e_total_mj": 0.0, "e_sleep_mj": 0.0,
    "e_idle_mj": 0.0, "ee_kbit_j": 0.0, "e_per_pkt_uj": 0.0,
    "aoi_s": 0.0, "dvr_pct": 0.0,
    "assoc_mean_s": 0.0, "assoc_max_s": 0.0, "assoc_frac": 0.0,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows_agg:  List[Dict] = []
    rows_seed: List[Dict] = []

    total = len(STA_COUNTS) * len(SEEDS)
    done  = 0
    t_start = time.time()

    print("=" * 70)
    print("  chang2019 TII baseline evaluation (PDR/AoI/DVR/delay/EE/assoc)")
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
                m   = extract_extra(sim, n)
            except Exception as exc:
                print(f"  ERROR N={n} seed={seed}: {exc}")
                m = dict(_ZERO_METRICS_EXTRA)

            seed_results.append(m)
            rows_seed.append({
                "policy": "chang2019", "n": n, "seed": seed,
                **{k: round(v, 6) for k, v in m.items()},
            })

        agg = _aggregate(seed_results)
        rows_agg.append({"policy": "chang2019", "n": n,
                         **{k: round(v, 6) for k, v in agg.items()}})
        print(f"  N={n:>4}  PDR={agg['pdr_mean']:.4f}  "
              f"AoI={agg['aoi_s_mean']:.2f}s  "
              f"DVR={agg['dvr_pct_mean']:.1f}%  "
              f"delay={agg['avg_delay_ms_mean']:.0f}ms  "
              f"ee={agg['ee_kbit_j_mean']:.2f}kbit/J  "
              f"assoc={agg['assoc_mean_s_mean']:.2f}s/"
              f"{agg['assoc_max_s_mean']:.2f}s/"
              f"{agg['assoc_frac_mean']:.3f}")

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
