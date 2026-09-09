"""
RAW Policy Comparison
=====================
Benchmarks RAW scheduling policies against no-RAW DCF:

  Policies tested
  ---------------
  1. no_raw          — pure DCF, no RAW scheduling
  2. static          — fixed equal-duration slots, round-robin AID assignment
  3. adaptive        — CUSUM-EWMA demand prediction, Bianchi slot-duration adaptation
  4. cluster_csv     — cluster assignments from uav_cluster_data.csv
  5. traffic_split   — EWMA/CUSUM load prediction, dynamic group split / merge (TASM-RAW)
  6. laca            — Load-Aware Channel Allocation (Taramit et al. 2023)
  7. chang2019       — Regression-based traffic-aware sensor grouping (Chang et al. 2019)

Sweep
-----
  N ∈ {100, 200, 400, 600, 800, 1000} STAs
  3 seeds per point (seeds = 42, 7, 99)
  Periodic traffic, 5 s interval, 128-byte packets, 120 s simulation
  95 % confidence intervals via Student-t (df = n_seeds − 1)

IEEE 802.11ah alignment
-----------------------
  Carrier        : 915 MHz (sub-1 GHz)
  MCS            : MCS0 = 150 kb/s  (BPSK 1/2, 1 MHz channel)
  Slot time σ    : 52 µs
  SIFS           : 160 µs
  DIFS           : 264 µs  (= SIFS + 2σ)
  CW_min / CW_max: 15 / 1023
  RAW slot unit  : 120 µs  (Section 9.4.2.200 / MORSE encoding)
  RAW slot max   : 2047 × 120 + 500 = 246 140 µs  (11-bit cslot field)
  Beacon interval: 500 ms
  RAW slots/group: 8
  Base slot dur  : 14 ms per slot

Outputs
-------
  results/policy_comparison.csv       — mean + 95 % CI for every metric (one row per N)
  results/policy_comparison_seeds.csv — raw per-seed data for reproducibility
  Console                             — summary tables (mean ± CI)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import PeriodicTraffic, PoissonTraffic

# ---------------------------------------------------------------------------
# Experiment parameters (defaults; overridden by --mcs CLI flag)
# ---------------------------------------------------------------------------
STA_COUNTS  = [100, 200, 400, 600, 800, 1000]
SIM_TIME_S  = 120.0
SEEDS       = [42, 7, 99]
TRAFFIC_INT = 5.0
PKT_SIZE    = 128
LINK_RATE   = 150_000   # MCS0 default; replaced by --mcs
PROP_DELAY  = 300e-6

# IEEE 802.11ah 1 MHz S1G PHY rate table (kbps)
_MCS_RATES = {
    "MCS0": 150_000,
    "MCS1": 300_000,
    "MCS2": 450_000,
    "MCS3": 600_000,
}

# t-critical for 95 % two-sided CI, df = len(SEEDS) − 1
# df=2 → t_{0.025,2} = 4.3027
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}

RAW_NUM_SLOTS  = 8
RAW_SLOT_DUR_S = 0.014

CLUSTER_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "uav_cluster_data.csv"
)

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "rl_qtables")

OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
# Output paths are set at runtime after parsing --mcs; defaults below.
# The RL policy is included so results go to a separate file to preserve
# the original policy_comparison.csv as the clean baseline reference.
OUT_CSV      = os.path.join(OUT_DIR, "policy_comparison_with_rl.csv")
OUT_CSV_SEED = os.path.join(OUT_DIR, "policy_comparison_with_rl_seeds.csv")

# ---------------------------------------------------------------------------
# Policy registry: (display_label, raw_enable, policy_key)
# ---------------------------------------------------------------------------
POLICIES: List[tuple] = [
    ("no_raw",        False, "none"),
    ("static",        True,  "static"),
    ("adaptive",      True,  "adaptive"),
    ("cluster_csv",   True,  "cluster_csv"),
    ("traffic_split", True,  "traffic_split"),
    ("laca",          True,  "laca"),
    ("chang2019",     True,  "chang2019"),
    ("rl",            True,  "rl"),      # phase-based Q-learning (pre-trained)
]

_POLICY_LABELS = [p[0] for p in POLICIES]

# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------
def build_and_run(num_stas: int, seed: int, raw_enable: bool, policy: str,
                  mcs: str = "MCS0", arrival: str = "periodic") -> "Simulator":
    cfg = default_config(raw_enable=raw_enable, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"]  = PKT_SIZE

    # Override PHY mode so all policies see the same MCS.
    cfg["phy"]["default_mode"] = mcs
    cfg["phy"]["control_mode"] = mcs

    if raw_enable:
        cfg["mac"]["raw_policy"]        = policy
        cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
        cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S
        cfg["mac"]["cluster_csv_path"]  = CLUSTER_CSV_PATH

        # RL-specific: load pre-trained Q-table and run in exploitation mode
        if policy == "rl":
            qp = os.path.join(QTABLE_DIR, f"qtable_N{num_stas}.pkl")
            if os.path.exists(qp):
                cfg["mac"]["rl_raw_qtable_path"]   = qp
            cfg["mac"]["rl_raw_epsilon"]       = 0.02   # near-zero exploration
            cfg["mac"]["rl_raw_epsilon_decay"] = 1.0    # hold epsilon constant

    link_rate = _MCS_RATES.get(mcs, LINK_RATE)
    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(
        sim,
        num_stas=num_stas,
        link_cfg={"rate_bps": link_rate, "prop_delay": PROP_DELAY, "per": 0.0},
    )

    for nid, node in sim.nodes.items():
        if nid <= 0:
            node.app.set_traffic_model(None)
        elif arrival == "poisson":
            # PoissonTraffic takes a RATE (packets/s), not an interval --
            # 1/TRAFFIC_INT gives the same MEAN interval as the periodic
            # case, so "poisson" and "periodic" runs are directly
            # comparable at the same nominal traffic intensity.
            node.app.set_traffic_model(PoissonTraffic(1.0 / TRAFFIC_INT))
        else:
            node.app.set_traffic_model(PeriodicTraffic(TRAFFIC_INT))

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

    tx     = max(1, int(getattr(s, "mac_tx_attempts",  0)))
    rtr    = int(getattr(s, "mac_retries",              0))
    ack_to = int(getattr(s, "mac_ack_timeouts",         0))

    fp = ff = 0
    for nid, node in sim.nodes.items():
        if nid == 0:
            continue
        ctx_stats = getattr(getattr(node, "mac", None), "ctx", None)
        ctx_stats = getattr(ctx_stats, "_stats", {}) if ctx_stats else {}
        fp += int(ctx_stats.get("raw_fit_pass", 0))
        ff += int(ctx_stats.get("raw_fit_fail", 0))

    pdr       = del_ / gen
    tput_kbps = del_ * PKT_SIZE * 8.0 / SIM_TIME_S / 1000.0

    e_tx_j = e_rx_j = e_idle_j = e_sleep_j = e_total_j = e_retx_j = 0.0
    total_tx_retries = 0
    for nid in sim.nodes:
        if nid == 0:
            continue
        e_tx_j    += s.energy_tx_j.get(nid, 0.0)
        e_rx_j    += s.energy_rx_j.get(nid, 0.0)
        e_idle_j  += s.energy_idle_j.get(nid, 0.0)
        e_sleep_j += s.energy_sleep_j.get(nid, 0.0)
        e_total_j += s.energy_total_j.get(nid, 0.0)
        e_retx_j  += s.energy_retx_j.get(nid, 0.0)
        total_tx_retries += s.tx_retries_per_node.get(nid, 0)

    n_stas = max(1, len(sim.nodes) - 1)
    avg_total_mj = e_total_j   * 1000.0 / n_stas
    avg_tx_mj    = e_tx_j      * 1000.0 / n_stas
    avg_rx_mj    = e_rx_j      * 1000.0 / n_stas
    avg_idle_mj  = e_idle_j    * 1000.0 / n_stas
    avg_sleep_mj = e_sleep_j   * 1000.0 / n_stas
    avg_retx_mj  = e_retx_j    * 1000.0 / n_stas
    retx_energy_pct = 100.0 * e_retx_j / max(1e-12, e_total_j)
    avg_tx_retries_per_sta = total_tx_retries / n_stas

    delivered_bits    = del_ * PKT_SIZE * 8.0
    ee_kbit_per_j     = (delivered_bits / 1000.0) / max(1e-12, e_total_j)
    pkt_per_sta       = max(1, del_) / n_stas
    energy_per_pkt_uj = (e_total_j * 1e6 / n_stas) / pkt_per_sta

    assoc_times = [s.assoc_total_time.get(nid, 0.0) for nid in sim.nodes if nid != 0]
    scan_times  = [s.assoc_scan_time.get(nid, 0.0)  for nid in sim.nodes if nid != 0]
    auth_times  = [s.assoc_auth_time.get(nid, 0.0)  for nid in sim.nodes if nid != 0]
    req_times   = [s.assoc_req_time.get(nid, 0.0)   for nid in sim.nodes if nid != 0]

    n_assoc  = sum(1 for t in assoc_times if t > 0)
    assoc_ok = [t for t in assoc_times if t > 0]
    scan_ok  = [scan_times[i] for i, t in enumerate(assoc_times) if t > 0]
    auth_ok  = [auth_times[i] for i, t in enumerate(assoc_times) if t > 0]
    req_ok   = [req_times[i]  for i, t in enumerate(assoc_times) if t > 0]

    mean_assoc_ms = 1000.0 * (sum(assoc_ok) / n_assoc if n_assoc else 0.0)
    max_assoc_ms  = 1000.0 * (max(assoc_ok)            if assoc_ok else 0.0)
    mean_scan_ms  = 1000.0 * (sum(scan_ok)  / n_assoc  if n_assoc else 0.0)
    mean_auth_ms  = 1000.0 * (sum(auth_ok)  / n_assoc  if n_assoc else 0.0)
    mean_req_ms   = 1000.0 * (sum(req_ok)   / n_assoc  if n_assoc else 0.0)
    assoc_frac    = n_assoc / max(1, len(assoc_times))

    return {
        "pdr":           pdr,
        "tput_kbps":     tput_kbps,
        "avg_delay_ms":  1000.0 * (sum(dly) / len(dly) if dly else 0.0),
        "p95_delay_ms":  1000.0 * _percentile(dly, 0.95),
        "drop_rate":     1.0 - pdr,
        "fairness":      fair,
        "col_rate":      ack_to / tx,
        "retry_rate":    rtr / tx,
        "raw_fit_pass":  fp / max(1, fp + ff),
        "e_total_mj":    avg_total_mj,
        "e_tx_mj":       avg_tx_mj,
        "e_rx_mj":       avg_rx_mj,
        "e_idle_mj":     avg_idle_mj,
        "e_sleep_mj":    avg_sleep_mj,
        "e_retx_mj":     avg_retx_mj,
        "retx_energy_pct": retx_energy_pct,
        "avg_tx_retries": avg_tx_retries_per_sta,
        "ee_kbit_j":     ee_kbit_per_j,
        "e_per_pkt_uj":  energy_per_pkt_uj,
        "assoc_mean_ms": mean_assoc_ms,
        "assoc_max_ms":  max_assoc_ms,
        "assoc_scan_ms": mean_scan_ms,
        "assoc_auth_ms": mean_auth_ms,
        "assoc_req_ms":  mean_req_ms,
        "assoc_frac":    assoc_frac,
    }


# ---------------------------------------------------------------------------
# Statistics: mean ± 95 % CI (Student-t, df = n−1)
# ---------------------------------------------------------------------------
def _ci_stats(values: List[float]) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) for a list of values."""
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


def _aggregate(seed_results: List[Dict]) -> Dict[str, float]:
    """Return flat dict with _mean, _ci_low, _ci_high for every metric."""
    keys = seed_results[0].keys()
    out  = {}
    for k in keys:
        vals = [d[k] for d in seed_results]
        mean, lo, hi = _ci_stats(vals)
        out[f"{k}_mean"]   = round(mean, 6)
        out[f"{k}_ci_low"] = round(lo,   6)
        out[f"{k}_ci_hi"]  = round(hi,   6)
    return out


# ---------------------------------------------------------------------------
# Console table helpers
# ---------------------------------------------------------------------------
_COL_W  = 18   # wider to accommodate "mean ± half" format
_N_W    = 5
_LINE_W = 2 + _N_W + 2 + len(_POLICY_LABELS) * _COL_W


def _sep() -> str:
    return "=" * _LINE_W


def _table_header(title: str) -> None:
    print(f"\n{_sep()}")
    print(f"  {title}")
    print(_sep())
    hdr = f"  {'N':>{_N_W}}  "
    for lbl in _POLICY_LABELS:
        hdr += f"{lbl:>{_COL_W}}"
    print(hdr)
    print("-" * _LINE_W)


def _table_row(row: Dict, metric: str, fmt: str = ".3f") -> None:
    line = f"  {row['num_stas']:>{_N_W}}  "
    for lbl in _POLICY_LABELS:
        mean = row.get(f"{lbl}_{metric}_mean", 0.0)
        lo   = row.get(f"{lbl}_{metric}_ci_low", mean)
        hi   = row.get(f"{lbl}_{metric}_ci_hi",  mean)
        half = (hi - lo) / 2.0
        cell = f"{mean:{fmt}}±{half:{fmt}}"
        line += f"{cell:>{_COL_W}}"
    print(line)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    global OUT_CSV, OUT_CSV_SEED

    parser = argparse.ArgumentParser(description="IEEE 802.11ah RAW policy comparison")
    parser.add_argument("--mcs", default="MCS0",
                        choices=list(_MCS_RATES.keys()),
                        help="PHY MCS for all policies (default: MCS0)")
    args = parser.parse_args()
    mcs = args.mcs

    os.makedirs(OUT_DIR, exist_ok=True)
    if mcs != "MCS0":
        suffix = f"_{mcs.lower()}"
        OUT_CSV      = os.path.join(OUT_DIR, f"policy_comparison{suffix}.csv")
        OUT_CSV_SEED = os.path.join(OUT_DIR, f"policy_comparison{suffix}_seeds.csv")

    n_seeds  = len(SEEDS)
    total    = len(STA_COUNTS) * len(POLICIES) * n_seeds
    done     = 0
    rows_agg: List[Dict] = []   # aggregated (mean + CI) — one row per N
    rows_raw: List[Dict] = []   # per-seed — one row per N × policy × seed

    rate_kbps = _MCS_RATES[mcs] // 1000
    print(f"\n{_sep()}")
    print(f"  IEEE 802.11ah RAW Policy Comparison  (with 95 % CI, t-dist df={n_seeds-1})")
    print(f"  {total} runs  |  {n_seeds} seeds × {len(POLICIES)} policies × {len(STA_COUNTS)} N values")
    print(f"  Sim time: {SIM_TIME_S:.0f} s  |  Traffic: periodic {TRAFFIC_INT} s  "
          f"|  Packet: {PKT_SIZE} B  |  {mcs} {rate_kbps} kb/s")
    print(f"  RAW: {RAW_NUM_SLOTS} slots/group  |  base slot dur: {RAW_SLOT_DUR_S*1000:.0f} ms")
    print(_sep())

    for n in STA_COUNTS:
        row_agg: Dict = {"num_stas": n}
        print()

        for label, raw_en, policy in POLICIES:
            seed_results = []
            for seed in SEEDS:
                done += 1
                print(f"  [{done:>3}/{total}]  N={n:>4}  "
                      f"policy={label:<14}  seed={seed} ...",
                      end="\r", flush=True)
                try:
                    sim = build_and_run(n, seed, raw_en, policy, mcs=mcs)
                    m   = extract(sim)
                except Exception as exc:
                    print(f"\n  ERROR  N={n}  {label}  seed={seed}: {exc}")
                    m = {
                        "pdr": 0.0, "tput_kbps": 0.0,
                        "avg_delay_ms": 0.0, "p95_delay_ms": 0.0,
                        "drop_rate": 1.0, "fairness": 0.0,
                        "col_rate": 0.0, "retry_rate": 0.0, "raw_fit_pass": 0.0,
                        "e_total_mj": 0.0, "e_tx_mj": 0.0, "e_rx_mj": 0.0,
                        "e_idle_mj": 0.0, "e_sleep_mj": 0.0,
                        "ee_kbit_j": 0.0, "e_per_pkt_uj": 0.0,
                        "assoc_mean_ms": 0.0, "assoc_max_ms": 0.0,
                        "assoc_scan_ms": 0.0, "assoc_auth_ms": 0.0,
                        "assoc_req_ms": 0.0, "assoc_frac": 0.0,
                    }

                seed_results.append(m)
                rows_raw.append({
                    "num_stas": n, "policy": label, "seed": seed,
                    **{k: round(v, 6) for k, v in m.items()},
                })

            agg = _aggregate(seed_results)
            for k, v in agg.items():
                row_agg[f"{label}_{k}"] = v

        rows_agg.append(row_agg)

        # Per-N progress line
        def _fmt(lbl, metric):
            mean = row_agg.get(f"{lbl}_{metric}_mean", 0.0)
            lo   = row_agg.get(f"{lbl}_{metric}_ci_low", mean)
            hi   = row_agg.get(f"{lbl}_{metric}_ci_hi",  mean)
            return f"{mean:.3f}[{lo:.3f},{hi:.3f}]"

        pdr_parts = "  ".join(f"{lbl}={_fmt(lbl,'pdr')}" for lbl, _, _ in POLICIES)
        print(f"  N={n:>4}  PDR:   {pdr_parts}")

    # -----------------------------------------------------------------------
    # Save aggregated CSV  (mean + CI per N)
    # -----------------------------------------------------------------------
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
        writer.writeheader()
        writer.writerows(rows_agg)

    # -----------------------------------------------------------------------
    # Save per-seed CSV
    # -----------------------------------------------------------------------
    with open(OUT_CSV_SEED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_raw[0].keys()))
        writer.writeheader()
        writer.writerows(rows_raw)

    # -----------------------------------------------------------------------
    # Summary tables  (format: mean±half_CI)
    # -----------------------------------------------------------------------
    _table_header("Packet Delivery Ratio  (PDR, higher is better)")
    for row in rows_agg:
        _table_row(row, "pdr")

    _table_header("Throughput  (kb/s, higher is better)")
    for row in rows_agg:
        _table_row(row, "tput_kbps", ".2f")

    _table_header("Average End-to-End Delay  (ms, lower is better)")
    for row in rows_agg:
        _table_row(row, "avg_delay_ms", ".1f")

    _table_header("95th-Percentile Delay  (ms, lower is better)")
    for row in rows_agg:
        _table_row(row, "p95_delay_ms", ".1f")

    _table_header("Drop Rate  (fraction, lower is better)")
    for row in rows_agg:
        _table_row(row, "drop_rate")

    _table_header("Jain Fairness Index  (0–1, higher is better)")
    for row in rows_agg:
        _table_row(row, "fairness")

    _table_header("ACK-Timeout Rate  (collision proxy, lower is better)")
    for row in rows_agg:
        _table_row(row, "col_rate")

    _table_header("Retry Rate  (retries / TX attempt, lower is better)")
    for row in rows_agg:
        _table_row(row, "retry_rate")

    _table_header("Total Energy / STA  (mJ, lower is better)")
    for row in rows_agg:
        _table_row(row, "e_total_mj", ".1f")

    _table_header("Idle Energy / STA  (mJ)")
    for row in rows_agg:
        _table_row(row, "e_idle_mj", ".1f")

    _table_header("Sleep Energy / STA  (mJ — RAW deep sleep; 0 for no_raw)")
    for row in rows_agg:
        _table_row(row, "e_sleep_mj", ".1f")

    _table_header("Retransmission Energy / STA  (mJ — TX energy of retry frames only)")
    for row in rows_agg:
        _table_row(row, "e_retx_mj", ".2f")

    _table_header("Retransmission Energy Fraction  (% of total energy, lower is better)")
    for row in rows_agg:
        _table_row(row, "retx_energy_pct", ".2f")

    _table_header("Avg TX Retries / STA  (count, lower is better)")
    for row in rows_agg:
        _table_row(row, "avg_tx_retries", ".1f")

    _table_header("Energy Efficiency  (kbit/J delivered, higher is better)")
    for row in rows_agg:
        _table_row(row, "ee_kbit_j", ".2f")

    _table_header("Mean Association Time / STA  (ms, lower is better)")
    for row in rows_agg:
        _table_row(row, "assoc_mean_ms", ".1f")

    _table_header("Association Success Fraction  (1.0 = all STAs associated)")
    for row in rows_agg:
        _table_row(row, "assoc_frac")

    print(f"\n{_sep()}")
    print(f"  Aggregated results (mean + 95% CI): {OUT_CSV}")
    print(f"  Per-seed raw data:                  {OUT_CSV_SEED}")
    print(_sep())
    print()

    return rows_agg


if __name__ == "__main__":
    main()
