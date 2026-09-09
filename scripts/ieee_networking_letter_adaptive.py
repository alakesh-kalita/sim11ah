"""
IEEE Networking Letters — Adaptive RAW Slot Sizing Evaluation
===============================================================
Benchmarks the proposed CUSUM-EWMA + Bianchi adaptive RAW slot-sizing
scheme ("adaptive") against three baselines used in the letter:

  1. static      — fixed equal-duration RAW slots (IEEE 802.11ah default)
  2. laca        — Load-Aware Channel Allocation (Taramit et al., 2023)
  3. chang2019   — Traffic-aware sensor grouping (Chang et al., 2019)
  4. adaptive    — proposed scheme

Sweeps (mirrors the letter's Figs. 2-4)
----------------------------------------
  Fig. 2 — vary_n_deterministic : N in STA_COUNTS, periodic 5 s traffic
  Fig. 3 — vary_n_poisson       : N in STA_COUNTS, Poisson traffic (mean 5 s)
  Fig. 4 — vary_interval        : packet interval in PKT_INTERVALS, N=500,
                                   deterministic (periodic) traffic

  3 seeds per point (SEEDS = 42, 7, 99); 95% CI via Student-t (df = n-1).

Metrics
-------
  PDR, throughput, average/95th-pct delay, drop rate, fairness,
  collision/retry rate, RAW-fit-pass rate, energy consumption
  (total / tx / rx / idle / sleep / retransmission, per STA and
  energy-efficiency in kbit/J), and association time (mean/max,
  scan/auth/request phase breakdown, association success fraction).

Table I alignment
------------------
  Packet size          : 128 bytes
  Beacon interval, Tb   : 0.5 s
  RAW groups, Ng        : 8
  RAW slots/group, Ns   : 8
  RAW slot duration     : 7 ms  (static baseline; adaptive/laca/chang2019
                           re-derive their own slot durations at runtime)
  CWmin / CWmax         : 15 / 1023
  ACK timeout           : 1 ms

Outputs (results/)
-------------------
  ieee_letter_vary_n_deterministic.csv (+ _seeds.csv)
  ieee_letter_vary_n_poisson.csv       (+ _seeds.csv)
  ieee_letter_vary_interval.csv        (+ _seeds.csv)
  Console summary tables (mean ± 95% CI) for every sweep.
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
# Experiment parameters
# ---------------------------------------------------------------------------
STA_COUNTS    = [100, 200, 400, 600, 800, 1000]
PKT_INTERVALS = [1.0, 2.0, 3.0, 4.0, 5.0]     # seconds; Fig. 4 sweep
FIXED_N_FOR_INTERVAL_SWEEP = 500

SIM_TIME_S  = 120.0
SEEDS       = [42, 7, 99]
TRAFFIC_INT = 5.0          # default periodic / mean-Poisson interval (Figs. 2-3)
PKT_SIZE    = 128
LINK_RATE   = 150_000      # MCS0
PROP_DELAY  = 300e-6

# t-critical for 95% two-sided CI, df = len(SEEDS) - 1
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}

# Table I alignment
RAW_NUM_GROUPS = 8
RAW_NUM_SLOTS  = 8
RAW_SLOT_DUR_S = 0.007      # 7 ms static baseline

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# Policy registry: (display_label, policy_key)
# ---------------------------------------------------------------------------
POLICIES: List[tuple] = [
    ("static",    "static"),
    ("laca",      "laca"),
    ("chang2019", "chang2019"),
    ("adaptive",  "adaptive"),
]
_POLICY_LABELS = [p[0] for p in POLICIES]


# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------
def build_and_run(num_stas: int, seed: int, policy: str,
                   pkt_interval: float, poisson: bool) -> "Simulator":
    cfg = default_config(raw_enable=True, traffic_mode="periodic")
    cfg["app"]["packet_size_bytes"] = PKT_SIZE

    cfg["mac"]["raw_policy"]        = policy
    cfg["mac"]["raw_num_groups"]    = RAW_NUM_GROUPS
    cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
    cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S

    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(
        sim,
        num_stas=num_stas,
        link_cfg={"rate_bps": LINK_RATE, "prop_delay": PROP_DELAY, "per": 0.0},
    )

    for nid, node in sim.nodes.items():
        if nid == 0:
            continue
        if poisson:
            traffic = PoissonTraffic(1.0 / pkt_interval)
        else:
            traffic = PeriodicTraffic(pkt_interval)
        node.app.set_traffic_model(traffic)

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
    rtr    = int(getattr(s, "mac_retries", 0))
    ack_to = int(getattr(s, "mac_ack_timeouts", 0))

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
    avg_total_mj = e_total_j * 1000.0 / n_stas
    avg_tx_mj    = e_tx_j    * 1000.0 / n_stas
    avg_rx_mj    = e_rx_j    * 1000.0 / n_stas
    avg_idle_mj  = e_idle_j  * 1000.0 / n_stas
    avg_sleep_mj = e_sleep_j * 1000.0 / n_stas
    avg_retx_mj  = e_retx_j  * 1000.0 / n_stas
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
        "pdr":             pdr,
        "tput_kbps":       tput_kbps,
        "avg_delay_ms":    1000.0 * (sum(dly) / len(dly) if dly else 0.0),
        "p95_delay_ms":    1000.0 * _percentile(dly, 0.95),
        "drop_rate":       1.0 - pdr,
        "fairness":        fair,
        "col_rate":        ack_to / tx,
        "retry_rate":      rtr / tx,
        "raw_fit_pass":    fp / max(1, fp + ff),
        "e_total_mj":      avg_total_mj,
        "e_tx_mj":         avg_tx_mj,
        "e_rx_mj":         avg_rx_mj,
        "e_idle_mj":       avg_idle_mj,
        "e_sleep_mj":      avg_sleep_mj,
        "e_retx_mj":       avg_retx_mj,
        "retx_energy_pct": retx_energy_pct,
        "avg_tx_retries":  avg_tx_retries_per_sta,
        "ee_kbit_j":       ee_kbit_per_j,
        "e_per_pkt_uj":    energy_per_pkt_uj,
        "assoc_mean_ms":   mean_assoc_ms,
        "assoc_max_ms":    max_assoc_ms,
        "assoc_scan_ms":   mean_scan_ms,
        "assoc_auth_ms":   mean_auth_ms,
        "assoc_req_ms":    mean_req_ms,
        "assoc_frac":      assoc_frac,
    }


_ERROR_METRICS = {
    "pdr": 0.0, "tput_kbps": 0.0, "avg_delay_ms": 0.0, "p95_delay_ms": 0.0,
    "drop_rate": 1.0, "fairness": 0.0, "col_rate": 0.0, "retry_rate": 0.0,
    "raw_fit_pass": 0.0, "e_total_mj": 0.0, "e_tx_mj": 0.0, "e_rx_mj": 0.0,
    "e_idle_mj": 0.0, "e_sleep_mj": 0.0, "e_retx_mj": 0.0,
    "retx_energy_pct": 0.0, "avg_tx_retries": 0.0, "ee_kbit_j": 0.0,
    "e_per_pkt_uj": 0.0, "assoc_mean_ms": 0.0, "assoc_max_ms": 0.0,
    "assoc_scan_ms": 0.0, "assoc_auth_ms": 0.0, "assoc_req_ms": 0.0,
    "assoc_frac": 0.0,
}


# ---------------------------------------------------------------------------
# Statistics: mean +/- 95% CI (Student-t, df = n-1)
# ---------------------------------------------------------------------------
def _ci_stats(values: List[float]) -> tuple:
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
    keys = seed_results[0].keys()
    out: Dict[str, float] = {}
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
_COL_W = 18


def _sep(width: int) -> str:
    return "=" * width


def _table_header(title: str, x_label: str, x_w: int) -> int:
    width = 2 + x_w + 2 + len(_POLICY_LABELS) * _COL_W
    print(f"\n{_sep(width)}")
    print(f"  {title}")
    print(_sep(width))
    hdr = f"  {x_label:>{x_w}}  "
    for lbl in _POLICY_LABELS:
        hdr += f"{lbl:>{_COL_W}}"
    print(hdr)
    print("-" * width)
    return width


def _table_row(row: Dict, x_key: str, metric: str, x_w: int, fmt: str = ".3f") -> None:
    x_val = row[x_key]
    x_str = f"{x_val:.1f}" if isinstance(x_val, float) else str(x_val)
    line = f"  {x_str:>{x_w}}  "
    for lbl in _POLICY_LABELS:
        mean = row.get(f"{lbl}_{metric}_mean", 0.0)
        lo   = row.get(f"{lbl}_{metric}_ci_low", mean)
        hi   = row.get(f"{lbl}_{metric}_ci_hi",  mean)
        half = (hi - lo) / 2.0
        cell = f"{mean:{fmt}}±{half:{fmt}}"
        line += f"{cell:>{_COL_W}}"
    print(line)


_METRIC_TABLES = [
    ("pdr",             "Packet Delivery Ratio  (PDR, higher is better)",          ".3f"),
    ("tput_kbps",        "Throughput  (kb/s, higher is better)",                    ".2f"),
    ("avg_delay_ms",     "Average End-to-End Delay  (ms, lower is better)",         ".1f"),
    ("p95_delay_ms",     "95th-Percentile Delay  (ms, lower is better)",            ".1f"),
    ("drop_rate",        "Drop Rate  (fraction, lower is better)",                  ".3f"),
    ("fairness",         "Jain Fairness Index  (0-1, higher is better)",            ".3f"),
    ("col_rate",         "ACK-Timeout Rate  (collision proxy, lower is better)",    ".3f"),
    ("retry_rate",       "Retry Rate  (retries / TX attempt, lower is better)",     ".3f"),
    ("e_total_mj",       "Total Energy / STA  (mJ, lower is better)",               ".1f"),
    ("e_tx_mj",          "TX Energy / STA  (mJ, lower is better)",                  ".2f"),
    ("e_rx_mj",          "RX Energy / STA  (mJ, lower is better)",                  ".2f"),
    ("e_idle_mj",        "Idle Energy / STA  (mJ)",                                 ".1f"),
    ("e_sleep_mj",       "Sleep Energy / STA  (mJ; RAW deep sleep)",                ".1f"),
    ("e_retx_mj",        "Retransmission Energy / STA  (mJ)",                       ".2f"),
    ("retx_energy_pct",  "Retransmission Energy Fraction  (%, lower is better)",    ".2f"),
    ("ee_kbit_j",        "Energy Efficiency  (kbit/J delivered, higher is better)", ".2f"),
    ("e_per_pkt_uj",     "Energy per Delivered Packet  (uJ, lower is better)",      ".2f"),
    ("assoc_mean_ms",    "Mean Association Time / STA  (ms, lower is better)",      ".1f"),
    ("assoc_max_ms",     "Max Association Time / STA  (ms, lower is better)",       ".1f"),
    ("assoc_scan_ms",    "Mean Scan-Phase Time / STA  (ms)",                        ".1f"),
    ("assoc_auth_ms",    "Mean Auth-Phase Time / STA  (ms)",                        ".1f"),
    ("assoc_req_ms",     "Mean Assoc-Request-Phase Time / STA  (ms)",               ".1f"),
    ("assoc_frac",       "Association Success Fraction  (1.0 = all STAs)",          ".3f"),
]


def _print_all_tables(rows_agg: List[Dict], x_key: str, x_label: str, x_w: int) -> None:
    for metric, title, fmt in _METRIC_TABLES:
        _table_header(title, x_label, x_w)
        for row in rows_agg:
            _table_row(row, x_key, metric, x_w, fmt)


# ---------------------------------------------------------------------------
# Generic sweep runner
# ---------------------------------------------------------------------------
def run_sweep(name: str, x_key: str, x_values: List, x_label: str, x_w: int,
              num_stas_fn, pkt_interval_fn, poisson: bool, out_dir: str) -> List[Dict]:
    n_seeds = len(SEEDS)
    total   = len(x_values) * len(POLICIES) * n_seeds
    done    = 0
    rows_agg: List[Dict] = []
    rows_raw: List[Dict] = []

    width = 2 + x_w + 2 + len(_POLICY_LABELS) * _COL_W
    print(f"\n{_sep(width)}")
    print(f"  Sweep: {name}")
    print(f"  {total} runs  |  {n_seeds} seeds x {len(POLICIES)} policies x {len(x_values)} points")
    print(_sep(width))

    for xv in x_values:
        num_stas = num_stas_fn(xv)
        pkt_interval = pkt_interval_fn(xv)
        row_agg: Dict = {x_key: xv}
        print()

        for label, policy in POLICIES:
            seed_results = []
            for seed in SEEDS:
                done += 1
                print(f"  [{done:>4}/{total}]  {x_label}={xv}  "
                      f"policy={label:<10}  seed={seed} ...", end="\r", flush=True)
                try:
                    sim = build_and_run(num_stas, seed, policy, pkt_interval, poisson)
                    m   = extract(sim)
                except Exception as exc:
                    print(f"\n  ERROR  {x_label}={xv}  {label}  seed={seed}: {exc}")
                    m = dict(_ERROR_METRICS)

                seed_results.append(m)
                rows_raw.append({
                    x_key: xv, "num_stas": num_stas, "pkt_interval_s": pkt_interval,
                    "policy": label, "seed": seed,
                    **{k: round(v, 6) for k, v in m.items()},
                })

            agg = _aggregate(seed_results)
            for k, v in agg.items():
                row_agg[f"{label}_{k}"] = v

        rows_agg.append(row_agg)

        def _fmt(lbl, metric):
            mean = row_agg.get(f"{lbl}_{metric}_mean", 0.0)
            lo   = row_agg.get(f"{lbl}_{metric}_ci_low", mean)
            hi   = row_agg.get(f"{lbl}_{metric}_ci_hi",  mean)
            return f"{mean:.3f}[{lo:.3f},{hi:.3f}]"

        pdr_parts = "  ".join(f"{lbl}={_fmt(lbl,'pdr')}" for lbl, _ in POLICIES)
        print(f"  {x_label}={xv}  PDR:  {pdr_parts}")

    os.makedirs(out_dir, exist_ok=True)
    out_csv      = os.path.join(out_dir, f"ieee_letter_{name}.csv")
    out_csv_seed = os.path.join(out_dir, f"ieee_letter_{name}_seeds.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
        writer.writeheader()
        writer.writerows(rows_agg)

    with open(out_csv_seed, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_raw[0].keys()))
        writer.writeheader()
        writer.writerows(rows_raw)

    _print_all_tables(rows_agg, x_key, x_label, x_w)

    print(f"\n  Aggregated results: {out_csv}")
    print(f"  Per-seed raw data:  {out_csv_seed}")

    return rows_agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="IEEE Networking Letters: adaptive RAW slot sizing vs. "
                     "static / LACA / Chang2019"
    )
    parser.add_argument("--sweep", default="all",
                         choices=["all", "vary_n_deterministic", "vary_n_poisson", "vary_interval"],
                         help="Which sweep(s) to run (default: all)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  IEEE Networking Letters -- Adaptive RAW Slot Sizing Evaluation")
    print(f"  Policies: {', '.join(_POLICY_LABELS)}")
    print(f"  Sim time: {SIM_TIME_S:.0f} s  |  Packet: {PKT_SIZE} B  |  MCS0 150 kb/s")
    print(f"  RAW: Ng={RAW_NUM_GROUPS} groups  Ns={RAW_NUM_SLOTS} slots/group  "
          f"static slot dur={RAW_SLOT_DUR_S*1000:.0f} ms")
    print(f"{'='*70}")

    results = {}

    if args.sweep in ("all", "vary_n_deterministic"):
        results["vary_n_deterministic"] = run_sweep(
            name="vary_n_deterministic",
            x_key="num_stas", x_values=STA_COUNTS, x_label="N", x_w=5,
            num_stas_fn=lambda xv: xv,
            pkt_interval_fn=lambda xv: TRAFFIC_INT,
            poisson=False,
            out_dir=OUT_DIR,
        )

    if args.sweep in ("all", "vary_n_poisson"):
        results["vary_n_poisson"] = run_sweep(
            name="vary_n_poisson",
            x_key="num_stas", x_values=STA_COUNTS, x_label="N", x_w=5,
            num_stas_fn=lambda xv: xv,
            pkt_interval_fn=lambda xv: TRAFFIC_INT,
            poisson=True,
            out_dir=OUT_DIR,
        )

    if args.sweep in ("all", "vary_interval"):
        results["vary_interval"] = run_sweep(
            name="vary_interval",
            x_key="pkt_interval_s", x_values=PKT_INTERVALS, x_label="Interval(s)", x_w=12,
            num_stas_fn=lambda xv: FIXED_N_FOR_INTERVAL_SWEEP,
            pkt_interval_fn=lambda xv: xv,
            poisson=False,
            out_dir=OUT_DIR,
        )

    print(f"\n{'='*70}")
    print("  All sweeps complete.")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    main()
