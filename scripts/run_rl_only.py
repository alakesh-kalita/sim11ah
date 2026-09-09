"""
RL Algorithm Comparison — RAW Policy
======================================
Compares all four RL modes supported by RLRawPolicy across N values:
  tabular       — standard Q-table
  tabular_ddqn  — Double Q-learning (two tables)
  dqn           — Deep Q-Network (pure-numpy 2-layer ReLU)
  ddqn          — Double DQN (online + target net)

Sweep
-----
  N ∈ {100, 200, 400, 600, 800, 1000} STAs
  3 seeds per point (42, 7, 99)
  Periodic traffic, 5 s interval, 128-byte packets, 120 s simulation

Outputs
-------
  results/rl_algo_comparison.csv        — mean + 95 % CI per algo × N
  results/rl_algo_comparison_seeds.csv  — raw per-seed data
  Console                               — side-by-side summary tables
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import PeriodicTraffic

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
STA_COUNTS  = [100, 200, 400, 600, 800, 1000]
SIM_TIME_S  = 120.0
SEEDS       = [42, 7, 99]
TRAFFIC_INT = 5.0
PKT_SIZE    = 128
LINK_RATE   = 150_000
PROP_DELAY  = 300e-6

RAW_NUM_SLOTS  = 8
RAW_SLOT_DUR_S = 0.014

RL_MODES = ["tabular", "tabular_ddqn", "dqn", "ddqn", "ppo"]

OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_CSV      = os.path.join(OUT_DIR, "rl_algo_comparison.csv")
OUT_CSV_SEED = os.path.join(OUT_DIR, "rl_algo_comparison_seeds.csv")

_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------
def build_and_run(num_stas: int, seed: int, mode: str) -> "Simulator":
    cfg = default_config(raw_enable=True, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"] = PKT_SIZE
    cfg["mac"]["raw_policy"]        = "rl"
    cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
    cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S
    cfg["mac"]["rl_raw_mode"]       = mode

    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(
        sim,
        num_stas=num_stas,
        link_cfg={"rate_bps": LINK_RATE, "prop_delay": PROP_DELAY, "per": 0.0},
    )
    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(
            PeriodicTraffic(TRAFFIC_INT) if nid > 0 else None
        )
    sim.run_and_finalize(SIM_TIME_S)
    return sim


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s   = sorted(data)
    idx = (len(s) - 1) * p
    lo  = int(idx)
    hi  = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def _jain(values) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    n = len(vals)
    return (sum(vals) ** 2) / (n * sum(v * v for v in vals))


def extract(sim: "Simulator") -> Dict[str, float]:
    s    = sim.stats
    gen  = max(1, int(getattr(s, "packets_generated", 0)))
    del_ = int(getattr(s, "packets_delivered", 0))
    dly  = list(getattr(s, "delays", []))

    try:
        per_src = dict(getattr(sim.nodes[0].app, "_delivered_from_src", {}))
    except Exception:
        per_src = {}
    fair = _jain(list(per_src.values()))

    tx     = max(1, int(getattr(s, "mac_tx_attempts", 0)))
    rtr    = int(getattr(s, "mac_retries",      0))
    ack_to = int(getattr(s, "mac_ack_timeouts", 0))

    pdr       = del_ / gen
    tput_kbps = del_ * PKT_SIZE * 8.0 / SIM_TIME_S / 1000.0

    e_tx_j = e_rx_j = e_idle_j = e_sleep_j = e_total_j = 0.0
    for nid in sim.nodes:
        if nid == 0:
            continue
        e_tx_j    += s.energy_tx_j.get(nid, 0.0)
        e_rx_j    += s.energy_rx_j.get(nid, 0.0)
        e_idle_j  += s.energy_idle_j.get(nid, 0.0)
        e_sleep_j += s.energy_sleep_j.get(nid, 0.0)
        e_total_j += s.energy_total_j.get(nid, 0.0)

    n_stas       = max(1, len(sim.nodes) - 1)
    avg_total_mj = e_total_j * 1000.0 / n_stas
    avg_sleep_mj = e_sleep_j * 1000.0 / n_stas
    avg_idle_mj  = e_idle_j  * 1000.0 / n_stas

    delivered_bits = del_ * PKT_SIZE * 8.0
    ee_kbit_j      = (delivered_bits / 1000.0) / max(1e-12, e_total_j)
    pkt_per_sta    = max(1, del_) / n_stas
    e_per_pkt_uj   = (e_total_j * 1e6 / n_stas) / pkt_per_sta

    return {
        "pdr":          pdr,
        "tput_kbps":    tput_kbps,
        "avg_delay_ms": 1000.0 * (sum(dly) / len(dly) if dly else 0.0),
        "p95_delay_ms": 1000.0 * _percentile(dly, 0.95),
        "drop_rate":    1.0 - pdr,
        "fairness":     fair,
        "col_rate":     ack_to / tx,
        "retry_rate":   rtr / tx,
        "e_total_mj":   avg_total_mj,
        "e_sleep_mj":   avg_sleep_mj,
        "e_idle_mj":    avg_idle_mj,
        "ee_kbit_j":    ee_kbit_j,
        "e_per_pkt_uj": e_per_pkt_uj,
    }

_ZERO_METRICS: Dict[str, float] = {
    "pdr": 0.0, "tput_kbps": 0.0, "avg_delay_ms": 0.0,
    "p95_delay_ms": 0.0, "drop_rate": 1.0, "fairness": 0.0,
    "col_rate": 0.0, "retry_rate": 0.0, "e_total_mj": 0.0,
    "e_sleep_mj": 0.0, "e_idle_mj": 0.0, "ee_kbit_j": 0.0,
    "e_per_pkt_uj": 0.0,
}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _ci_stats(values: List[float]):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    var  = sum((v - mean) ** 2 for v in values) / (n - 1)
    std  = math.sqrt(var)
    t    = _T_CRIT.get(n - 1, 2.0)
    half = t * std / math.sqrt(n)
    return mean, mean - half, mean + half


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
_COL_W   = 14   # width per algo column
_MODE_HDR = {
    "tabular":      "Tabular-Q",
    "tabular_ddqn": "Tab-DDQN",
    "dqn":          "DQN",
    "ddqn":         "DDQN",
    "ppo":          "PPO",
}

def _line(char: str = "=", width: int = 80) -> None:
    print(char * width)

def _print_comparison_table(
    title: str,
    key: str,
    fmt: str,
    # agg_results[mode][n_idx] = dict with f"{key}_mean" etc.
    agg_results: Dict[str, List[Dict]],
    better: str = "↑",
) -> None:
    modes  = RL_MODES
    n_cols = len(modes)
    width  = 8 + n_cols * (_COL_W + 2)

    _line("=", width)
    print(f"  {title}  ({better} better)")
    _line("-", width)

    # Header row
    hdr = f"  {'N':>5}  "
    for m in modes:
        hdr += f"{_MODE_HDR[m]:>{_COL_W}}  "
    print(hdr.rstrip())
    _line("-", width)

    n_list = STA_COUNTS
    for i, n in enumerate(n_list):
        row = f"  {n:>5}  "
        best_val = None
        vals = {}
        for m in modes:
            mean = agg_results[m][i].get(f"{key}_mean", 0.0)
            vals[m] = mean
            if best_val is None:
                best_val = mean
            elif better == "↑" and mean > best_val:
                best_val = mean
            elif better == "↓" and mean < best_val:
                best_val = mean

        for m in modes:
            v   = vals[m]
            tag = "*" if abs(v - best_val) < 1e-9 else " "
            cell = f"{v:{fmt}}{tag}"
            row += f"{cell:>{_COL_W}}  "
        print(row.rstrip())

    _line("=", width)
    print(f"  (* marks best per row)")
    print()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    total = len(RL_MODES) * len(STA_COUNTS) * len(SEEDS)
    done  = 0

    # agg_results[mode][n_idx] = aggregated dict
    # seed_results[mode][n_idx][seed_idx] = metric dict
    agg_results:  Dict[str, List[Dict]]       = {m: [] for m in RL_MODES}
    seed_results: Dict[str, List[List[Dict]]] = {m: [] for m in RL_MODES}

    rows_agg:  List[Dict] = []
    rows_seed: List[Dict] = []

    _line()
    print("  IEEE 802.11ah — RL Algorithm Comparison")
    print(f"  Modes : {', '.join(RL_MODES)}")
    print(f"  N     : {STA_COUNTS}")
    print(f"  Seeds : {SEEDS}  |  {total} total runs")
    print(f"  Sim={SIM_TIME_S:.0f}s  Traffic={TRAFFIC_INT}s  Pkt={PKT_SIZE}B  MCS0")
    _line()

    t_start = time.time()

    for mode in RL_MODES:
        print(f"\n  === Mode: {mode} ===")
        for n in STA_COUNTS:
            n_seed_results: List[Dict] = []
            for seed in SEEDS:
                done += 1
                elapsed = time.time() - t_start
                eta     = (elapsed / done) * (total - done) if done > 1 else 0
                print(
                    f"  [{done:>3}/{total}]  mode={mode:<14}  N={n:>4}  seed={seed}"
                    f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  ",
                    end="\r", flush=True,
                )
                try:
                    sim = build_and_run(n, seed, mode)
                    m   = extract(sim)
                except Exception as exc:
                    print(f"\n  ERROR mode={mode} N={n} seed={seed}: {exc}")
                    m = dict(_ZERO_METRICS)

                n_seed_results.append(m)
                rows_seed.append({
                    "mode": mode, "n": n, "seed": seed,
                    **{k: round(v, 6) for k, v in m.items()},
                })

            seed_results[mode].append(n_seed_results)

            agg: Dict = {"mode": mode, "n": n}
            for k in n_seed_results[0]:
                vals         = [d[k] for d in n_seed_results]
                mean, lo, hi = _ci_stats(vals)
                agg[f"{k}_mean"]  = round(mean, 6)
                agg[f"{k}_ci_lo"] = round(lo,   6)
                agg[f"{k}_ci_hi"] = round(hi,   6)
            agg_results[mode].append(agg)
            rows_agg.append(agg)

            print(
                f"\n  mode={mode:<14}  N={n:>4}  "
                f"PDR={agg['pdr_mean']:.4f}  "
                f"tput={agg['tput_kbps_mean']:.1f}kb/s  "
                f"delay={agg['avg_delay_ms_mean']:.0f}ms  "
                f"ee={agg['ee_kbit_j_mean']:.2f}kbit/J"
            )

    wall = time.time() - t_start
    print(f"\n  Total wall-clock: {wall:.1f}s\n")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
        writer.writeheader()
        writer.writerows(rows_agg)

    with open(OUT_CSV_SEED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_seed[0].keys()))
        writer.writeheader()
        writer.writerows(rows_seed)

    # ------------------------------------------------------------------
    # Side-by-side comparison tables
    # ------------------------------------------------------------------
    metrics = [
        ("Packet Delivery Ratio  (PDR)",               "pdr",          ".4f", "↑"),
        ("Throughput  (kb/s)",                         "tput_kbps",    ".2f", "↑"),
        ("Average End-to-End Delay  (ms)",             "avg_delay_ms", ".1f", "↓"),
        ("95th-Percentile Delay  (ms)",                "p95_delay_ms", ".1f", "↓"),
        ("Collision / ACK-timeout Rate",               "col_rate",     ".4f", "↓"),
        ("Retry Rate  (retries / TX attempt)",         "retry_rate",   ".4f", "↓"),
        ("Jain Fairness Index  (0–1)",                 "fairness",     ".4f", "↑"),
        ("Energy Efficiency  (kbit/J)",                "ee_kbit_j",    ".2f", "↑"),
        ("Total Energy / STA  (mJ)",                   "e_total_mj",   ".2f", "↓"),
        ("Energy / Delivered Packet  (µJ)",            "e_per_pkt_uj", ".1f", "↓"),
    ]

    print("\n" + "=" * 80)
    print("  SIDE-BY-SIDE ALGORITHM COMPARISON")
    print("=" * 80 + "\n")

    for title, key, fmt, better in metrics:
        _print_comparison_table(title, key, fmt, agg_results, better)

    # ------------------------------------------------------------------
    # Win-count summary
    # ------------------------------------------------------------------
    win_counts = {m: 0 for m in RL_MODES}
    win_metrics = [
        ("pdr", "↑"), ("tput_kbps", "↑"), ("avg_delay_ms", "↓"),
        ("col_rate", "↓"), ("retry_rate", "↓"), ("fairness", "↑"),
        ("ee_kbit_j", "↑"),
    ]
    for key, better in win_metrics:
        for i in range(len(STA_COUNTS)):
            vals = {m: agg_results[m][i].get(f"{key}_mean", 0.0) for m in RL_MODES}
            best = max(vals.values()) if better == "↑" else min(vals.values())
            for m in RL_MODES:
                if abs(vals[m] - best) < 1e-9:
                    win_counts[m] += 1

    _line()
    print("  WIN COUNT  (best value across N×metric combinations)")
    print(f"  Metrics tracked: {[k for k, _ in win_metrics]}")
    _line("-")
    for m in RL_MODES:
        bar = "#" * win_counts[m]
        print(f"  {_MODE_HDR[m]:<14}  {win_counts[m]:>3}  {bar}")
    _line()

    print(f"\n  Results : {OUT_CSV}")
    print(f"  Per-seed: {OUT_CSV_SEED}\n")


if __name__ == "__main__":
    main()
