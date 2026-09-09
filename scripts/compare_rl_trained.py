"""
Evaluation of Multi-Episode-Trained RL RAW Policies vs Baselines
==================================================================
Loads the Q-tables / network weights produced by train_rl_policy.py
(results/rl_qtables/qtable_<mode>_N<n>.pkl) and evaluates each trained RL
controller with near-zero exploration (epsilon at floor).
Compares against the three baselines (no_raw, static, laca) taken from
results/rl_policy_comparison.csv.

RL modes  : tabular, tabular_ddqn, dqn, ddqn, ppo
Baselines : no_raw, static, laca  (loaded from results/rl_policy_comparison.csv)

Sweep
-----
  N ∈ {100, 200, 400, 600, 800, 1000} STAs
  3 evaluation seeds (42, 7, 99) — distinct from the training seeds
  Periodic traffic, 5 s interval, 128-byte packets, 120 s simulation
  95 % CI via Student-t (df = n_seeds − 1)

Outputs
-------
  results/rl_multiepisode_comparison.csv        — long format (policy x N), mean + 95% CI
  results/rl_multiepisode_comparison_seeds.csv  — raw per-seed rows
  Console                                       — same section layout as compare_rl_policy.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import PeriodicTraffic

from train_rl_policy import RL_MODES, qtable_path, DEFAULT_N_VALUES

# ---------------------------------------------------------------------------
# Experiment parameters (mirrors compare_rl_policy.py)
# ---------------------------------------------------------------------------
STA_COUNTS  = DEFAULT_N_VALUES
SIM_TIME_S  = 120.0
SEEDS       = [42, 7, 99]
TRAFFIC_INT = 5.0
PKT_SIZE    = 128
LINK_RATE   = 150_000
PROP_DELAY  = 300e-6
RAW_NUM_SLOTS  = 8
RAW_SLOT_DUR_S = 0.014

_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

OUT_DIR        = os.path.join(os.path.dirname(__file__), "..", "results")
BASELINE_CSV   = os.path.join(OUT_DIR, "rl_policy_comparison.csv")
OUT_CSV        = os.path.join(OUT_DIR, "rl_multiepisode_comparison.csv")
OUT_CSV_SEED   = os.path.join(OUT_DIR, "rl_multiepisode_comparison_seeds.csv")

_BASE_LABELS = ["no_raw", "static", "laca"]
_RL_LABELS   = ["tabular", "tab_ddqn", "dqn", "ddqn", "ppo"]
_MODE_OF     = {"tabular": "tabular", "tab_ddqn": "tabular_ddqn",
                "dqn": "dqn", "ddqn": "ddqn", "ppo": "ppo"}

_SHORT = {
    "no_raw": "NoRAW", "static": "Static", "laca": "LACA",
    "tabular": "Tabular", "tab_ddqn": "TabDDQN",
    "dqn": "DQN", "ddqn": "DDQN", "ppo": "PPO",
}


# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------
def build_and_run(num_stas: int, seed: int, mode: str, n: int) -> "Simulator":
    cfg = default_config(raw_enable=True, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"] = PKT_SIZE
    cfg["mac"]["raw_policy"]        = "rl"
    cfg["mac"]["rl_raw_mode"]       = mode
    cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
    cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S

    qp = qtable_path(mode, n)
    if os.path.exists(qp):
        cfg["mac"]["rl_raw_qtable_path"]   = qp
        cfg["mac"]["rl_raw_epsilon"]       = 0.02
        cfg["mac"]["rl_raw_epsilon_decay"] = 1.0

    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(
        sim, num_stas=num_stas,
        link_cfg={"rate_bps": LINK_RATE, "prop_delay": PROP_DELAY, "per": 0.0},
    )
    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(PeriodicTraffic(TRAFFIC_INT) if nid > 0 else None)
    sim.run_and_finalize(SIM_TIME_S)
    return sim


# ---------------------------------------------------------------------------
# Metric extraction (identical to compare_rl_policy.py)
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

    fp = ff = 0
    for nid, node in sim.nodes.items():
        if nid == 0:
            continue
        ctx = getattr(getattr(node, "mac", None), "ctx", None)
        cs  = getattr(ctx, "_stats", {}) if ctx else {}
        fp += int(cs.get("raw_fit_pass", 0))
        ff += int(cs.get("raw_fit_fail", 0))

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
        "raw_fit_pass": fp / max(1, fp + ff),
        "e_total_mj":   avg_total_mj,
        "e_sleep_mj":   avg_sleep_mj,
        "e_idle_mj":    avg_idle_mj,
        "ee_kbit_j":    ee_kbit_j,
        "e_per_pkt_uj": e_per_pkt_uj,
    }


_ZERO_METRICS = {
    "pdr": 0.0, "tput_kbps": 0.0, "avg_delay_ms": 0.0, "p95_delay_ms": 0.0,
    "drop_rate": 1.0, "fairness": 0.0, "col_rate": 0.0, "retry_rate": 0.0,
    "raw_fit_pass": 0.0, "e_total_mj": 0.0, "e_sleep_mj": 0.0,
    "e_idle_mj": 0.0, "ee_kbit_j": 0.0, "e_per_pkt_uj": 0.0,
}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _ci_stats(values: List[float]) -> Tuple[float, float, float]:
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
    out = {}
    for k in seed_results[0]:
        vals         = [d[k] for d in seed_results]
        mean, lo, hi = _ci_stats(vals)
        out[f"{k}_mean"]  = round(mean, 6)
        out[f"{k}_ci_lo"] = round(lo,   6)
        out[f"{k}_ci_hi"] = round(hi,   6)
    return out


# ---------------------------------------------------------------------------
# Load baseline rows from the existing 3-baseline comparison CSV
# ---------------------------------------------------------------------------
def load_baselines() -> Dict[Tuple[str, int], Dict]:
    results: Dict[Tuple[str, int], Dict] = {}
    if not os.path.exists(BASELINE_CSV):
        return results
    with open(BASELINE_CSV) as f:
        for row in csv.DictReader(f):
            if row["policy"] not in _BASE_LABELS:
                continue
            n   = int(row["n"])
            agg = {k: float(v) for k, v in row.items() if k not in ("policy", "n")}
            results[(row["policy"], n)] = agg
    return results


# ---------------------------------------------------------------------------
# Console table helpers (mirrors compare_rl_policy.py)
# ---------------------------------------------------------------------------
_CW = 11

def _line(w: int = 80, char: str = "=") -> None:
    print(char * w)

def _table(title: str, labels: List[str], key: str,
           results: Dict[Tuple[str, int], Dict],
           fmt: str = ".4f", better: str = "↑") -> None:
    n_cols = len(labels)
    w = 7 + n_cols * (_CW + 1)
    _line(w)
    print(f"  {title}  ({better} better)")
    _line(w, "-")
    hdr = f"  {'N':>4}  " + "  ".join(f"{_SHORT.get(l, l):>{_CW}}" for l in labels)
    print(hdr)
    _line(w, "-")
    for n in STA_COUNTS:
        vals = {l: results[(l, n)].get(f"{key}_mean", 0.0) for l in labels}
        best = max(vals.values()) if better == "↑" else min(vals.values())
        row  = f"  {n:>4}  "
        for l in labels:
            v   = vals[l]
            tag = "*" if abs(v - best) < 1e-9 else " "
            row += f"{v:{fmt}}{tag}".rjust(_CW + 1) + " "
        print(row.rstrip())
    _line(w)
    print("  (* = best in row)")
    print()


def _delta_table(title: str, rl_labels: List[str], base_labels: List[str],
                 key: str, results: Dict[Tuple[str, int], Dict],
                 fmt: str = ".4f", better: str = "↑") -> None:
    w = 7 + len(rl_labels) * (_CW + 1)
    _line(w)
    print(f"  Δ {title}  (RL − best baseline, {better} means RL wins)")
    _line(w, "-")
    hdr = f"  {'N':>4}  " + "  ".join(f"{_SHORT.get(l,l):>{_CW}}" for l in rl_labels)
    print(hdr)
    _line(w, "-")
    for n in STA_COUNTS:
        best_base = (max if better == "↑" else min)(
            results[(b, n)].get(f"{key}_mean", 0.0) for b in base_labels
        )
        row = f"  {n:>4}  "
        for l in rl_labels:
            delta = results[(l, n)].get(f"{key}_mean", 0.0) - best_base
            sign  = "+" if delta >= 0 else ""
            cell  = f"{sign}{delta:{fmt}}"
            row  += f"{cell:>{_CW}}  "
        print(row.rstrip())
    _line(w)
    print()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results: Dict[Tuple[str, int], Dict] = load_baselines()
    if not results:
        print(f"  WARNING: no baseline data found at {BASELINE_CSV}")

    total = len(_RL_LABELS) * len(STA_COUNTS) * len(SEEDS)
    done  = 0

    rows_agg:  List[Dict] = []
    rows_seed: List[Dict] = []

    _line()
    print("  IEEE 802.11ah — Multi-Episode-Trained RL vs Baselines")
    print(f"  Baselines (from {BASELINE_CSV}): {_BASE_LABELS}")
    print(f"  RL modes (trained, evaluated with eps=0.02): {_RL_LABELS}")
    print(f"  N={STA_COUNTS}  seeds={SEEDS}")
    print(f"  {total} total eval runs  |  Sim={SIM_TIME_S:.0f}s")
    _line()

    t_start = time.time()

    for label in _RL_LABELS:
        mode = _MODE_OF[label]
        print(f"\n  === RL (multi-episode trained): {label} ({mode}) ===")
        for n in STA_COUNTS:
            seed_results: List[Dict] = []
            qp = qtable_path(mode, n)
            has_qtable = os.path.exists(qp)
            for seed in SEEDS:
                done += 1
                elapsed = time.time() - t_start
                eta     = (elapsed / done) * (total - done) if done > 1 else 0
                print(
                    f"  [{done:>3}/{total}]  {label:<14}  N={n:>4}  seed={seed}"
                    f"  qtable={'Y' if has_qtable else 'N'}"
                    f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  ",
                    end="\r", flush=True,
                )
                try:
                    sim = build_and_run(n, seed, mode, n)
                    m   = extract(sim)
                except Exception as exc:
                    print(f"\n  ERROR {label} N={n} seed={seed}: {exc}")
                    m = dict(_ZERO_METRICS)

                seed_results.append(m)
                rows_seed.append({
                    "policy": label, "n": n, "seed": seed,
                    **{k: round(v, 6) for k, v in m.items()},
                })

            agg = _aggregate(seed_results)
            results[(label, n)] = agg
            rows_agg.append({"policy": label, "n": n,
                             **{k: round(v, 6) for k, v in agg.items()}})

            print(
                f"\n  {label:<14}  N={n:>4}  "
                f"PDR={agg['pdr_mean']:.4f}  "
                f"tput={agg['tput_kbps_mean']:.1f}kb/s  "
                f"delay={agg['avg_delay_ms_mean']:.0f}ms  "
                f"ee={agg['ee_kbit_j_mean']:.2f}kbit/J"
            )

    wall = time.time() - t_start
    print(f"\n  Total wall-clock: {wall:.1f}s\n")

    # ------------------------------------------------------------------
    # Save CSVs (baselines + newly evaluated RL rows)
    # ------------------------------------------------------------------
    for label in _BASE_LABELS:
        for n in STA_COUNTS:
            if (label, n) in results:
                rows_agg.append({"policy": label, "n": n, **results[(label, n)]})
    rows_agg.sort(key=lambda r: (r["n"], (_BASE_LABELS + _RL_LABELS).index(r["policy"])))

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
        writer.writeheader()
        writer.writerows(rows_agg)

    if rows_seed:
        with open(OUT_CSV_SEED, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_seed[0].keys()))
            writer.writeheader()
            writer.writerows(rows_seed)

    # ------------------------------------------------------------------
    # Section 1: Baseline results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SECTION 1 — BASELINE POLICIES")
    print("=" * 80 + "\n")

    for title, key, fmt, better in [
        ("PDR",                        "pdr",          ".4f", "↑"),
        ("Throughput (kb/s)",          "tput_kbps",    ".2f", "↑"),
        ("Avg Delay (ms)",             "avg_delay_ms", ".1f", "↓"),
        ("Energy Efficiency (kbit/J)", "ee_kbit_j",    ".2f", "↑"),
        ("Jain Fairness",              "fairness",     ".4f", "↑"),
    ]:
        _table(title, _BASE_LABELS, key, results, fmt, better)

    # ------------------------------------------------------------------
    # Section 2: RL algorithm results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SECTION 2 — RL ALGORITHMS (multi-episode trained)")
    print("=" * 80 + "\n")

    for title, key, fmt, better in [
        ("PDR",                        "pdr",          ".4f", "↑"),
        ("Throughput (kb/s)",          "tput_kbps",    ".2f", "↑"),
        ("Avg Delay (ms)",             "avg_delay_ms", ".1f", "↓"),
        ("Collision Rate",             "col_rate",     ".4f", "↓"),
        ("Retry Rate",                 "retry_rate",   ".4f", "↓"),
        ("Energy Efficiency (kbit/J)", "ee_kbit_j",    ".2f", "↑"),
        ("Jain Fairness",              "fairness",     ".4f", "↑"),
    ]:
        _table(title, _RL_LABELS, key, results, fmt, better)

    # CI Width PDR
    w = 7 + len(_RL_LABELS) * (_CW + 1)
    _line(w)
    print(f"  CI Width PDR  (↓ = more stable across seeds)")
    _line(w, "-")
    hdr = f"  {'N':>4}  " + "  ".join(f"{_SHORT.get(l,l):>{_CW}}" for l in _RL_LABELS)
    print(hdr)
    _line(w, "-")
    for n in STA_COUNTS:
        row = f"  {n:>4}  "
        vals = {}
        for l in _RL_LABELS:
            r   = results[(l, n)]
            vals[l] = r.get("pdr_ci_hi", 0.0) - r.get("pdr_ci_lo", 0.0)
        best = min(vals.values())
        for l in _RL_LABELS:
            tag = "*" if abs(vals[l] - best) < 1e-9 else " "
            row += f"{vals[l]:.4f}{tag}".rjust(_CW + 1) + " "
        print(row.rstrip())
    _line(w)
    print("  (* = most stable)\n")

    # ------------------------------------------------------------------
    # Section 3: Head-to-head — RL vs baselines (Δ tables)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SECTION 3 — Δ (RL − BEST BASELINE)  positive = RL wins")
    print("=" * 80 + "\n")

    for title, key, fmt, better in [
        ("PDR",                        "pdr",       ".4f", "↑"),
        ("Energy Efficiency (kbit/J)", "ee_kbit_j", ".2f", "↑"),
        ("Avg Delay (ms)",             "avg_delay_ms", ".1f", "↓"),
    ]:
        _delta_table(title, _RL_LABELS, _BASE_LABELS, key, results, fmt, better)

    # ------------------------------------------------------------------
    # Section 4: Win-count summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("  SECTION 4 — WIN COUNT (best value per N × metric)")
    print("=" * 80)

    win_metrics = [
        ("pdr",       "↑"), ("tput_kbps",    "↑"), ("avg_delay_ms", "↓"),
        ("col_rate",  "↓"), ("retry_rate",   "↓"), ("fairness",     "↑"),
        ("ee_kbit_j", "↑"),
    ]
    all_labels    = _BASE_LABELS + _RL_LABELS
    win_counts    = {l: 0 for l in all_labels}
    total_chances = len(STA_COUNTS) * len(win_metrics)

    for key, better in win_metrics:
        for n in STA_COUNTS:
            vals = {l: results[(l, n)].get(f"{key}_mean", 0.0) for l in all_labels}
            best = max(vals.values()) if better == "↑" else min(vals.values())
            for l in all_labels:
                if abs(vals[l] - best) < 1e-9:
                    win_counts[l] += 1

    print(f"\n  Tracked: {[k for k,_ in win_metrics]}  ({total_chances} chances)\n")
    print(f"  {'Policy':<15}  {'Wins':>5}  {'%':>6}  Bar")
    print("  " + "-" * 60)
    for l in all_labels:
        wn  = win_counts[l]
        pct = 100 * wn / total_chances
        bar = "#" * wn
        tag = "  ← RL" if l in _RL_LABELS else ""
        print(f"  {_SHORT.get(l,l):<15}  {wn:>5}  {pct:>5.1f}%  {bar}{tag}")

    print("\n" + "=" * 80)
    print(f"  Results : {OUT_CSV}")
    print(f"  Per-seed: {OUT_CSV_SEED}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
