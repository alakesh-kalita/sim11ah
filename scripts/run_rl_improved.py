"""
RL Improved Variants — single-episode comparison
Each mode is run with its recommended improvement:
  tabular / tabular_ddqn : UCB1 (C=1.0) + 3-step returns
  dqn                    : 3-step returns
  ddqn                   : soft target (tau=0.01) + 3-step returns
  ppo                    : unchanged

Outputs: results/rl_improved_comparison.csv, results/rl_improved_seeds.csv
"""
from __future__ import annotations
import csv, math, os, sys, time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import PeriodicTraffic

STA_COUNTS  = [100, 200, 400, 600, 800, 1000]
SIM_TIME_S  = 120.0
SEEDS       = [42, 7, 99]
TRAFFIC_INT = 5.0
PKT_SIZE    = 128
LINK_RATE   = 150_000
PROP_DELAY  = 300e-6
RAW_NUM_SLOTS  = 8
RAW_SLOT_DUR_S = 0.014

OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_CSV      = os.path.join(OUT_DIR, "rl_improved_comparison.csv")
OUT_CSV_SEED = os.path.join(OUT_DIR, "rl_improved_seeds.csv")

_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

IMPROVED: List[Tuple[str, str, Dict]] = [
    ("tabular+",  "tabular",      {"rl_raw_use_ucb": True, "rl_raw_ucb_c": 1.0, "rl_raw_n_step": 3}),
    ("tab_ddqn+", "tabular_ddqn", {"rl_raw_use_ucb": True, "rl_raw_ucb_c": 1.0, "rl_raw_n_step": 3}),
    ("dqn+",      "dqn",          {"rl_raw_n_step": 3}),
    ("ddqn+",     "ddqn",         {"rl_raw_target_update_tau": 0.01, "rl_raw_n_step": 3}),
    ("ppo",       "ppo",          {}),
]

def build_and_run(num_stas: int, seed: int, base_mode: str, extra: Dict):
    cfg = default_config(raw_enable=True, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"] = PKT_SIZE
    cfg["mac"]["raw_policy"]        = "rl"
    cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
    cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S
    cfg["mac"]["rl_raw_mode"]       = base_mode
    for k, v in extra.items():
        cfg["mac"][k] = v
    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(sim, num_stas=num_stas,
        link_cfg={"rate_bps": LINK_RATE, "prop_delay": PROP_DELAY, "per": 0.0})
    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(PeriodicTraffic(TRAFFIC_INT) if nid > 0 else None)
    sim.run_and_finalize(SIM_TIME_S)
    return sim

def _percentile(data: list, p: float) -> float:
    if not data: return 0.0
    s = sorted(data); idx = (len(s)-1)*p; lo = int(idx); hi = min(lo+1,len(s)-1)
    return s[lo] + (idx-lo)*(s[hi]-s[lo])

def _jain(values) -> float:
    vals = [v for v in values if v > 0]
    if not vals: return 0.0
    return (sum(vals)**2) / (len(vals) * sum(v*v for v in vals))

def extract(sim: "Simulator") -> Dict[str, float]:
    s    = sim.stats
    gen  = max(1, int(getattr(s, "packets_generated", 0)))
    del_ = int(getattr(s, "packets_delivered", 0))
    dly  = list(getattr(s, "delays", []))
    try:
        per_src = dict(getattr(sim.nodes[0].app, "_delivered_from_src", {}))
    except Exception:
        per_src = {}
    tx  = max(1, int(getattr(s, "mac_tx_attempts", 0)))
    rtr = int(getattr(s, "mac_retries",      0))
    ato = int(getattr(s, "mac_ack_timeouts", 0))
    pdr = del_ / gen
    tput = del_ * PKT_SIZE * 8.0 / SIM_TIME_S / 1000.0
    e_tx=e_rx=e_idle=e_sleep=e_total = 0.0
    for nid in sim.nodes:
        if nid == 0: continue
        e_tx    += s.energy_tx_j.get(nid,    0.0)
        e_rx    += s.energy_rx_j.get(nid,    0.0)
        e_idle  += s.energy_idle_j.get(nid,  0.0)
        e_sleep += s.energy_sleep_j.get(nid, 0.0)
        e_total += s.energy_total_j.get(nid, 0.0)
    n_stas = max(1, len(sim.nodes)-1)
    ee = (del_*PKT_SIZE*8.0/1000.0) / max(1e-12, e_total)
    return {
        "pdr":          pdr,
        "tput_kbps":    tput,
        "avg_delay_ms": 1000.0*(sum(dly)/len(dly) if dly else 0.0),
        "p95_delay_ms": 1000.0*_percentile(dly, 0.95),
        "drop_rate":    1.0-pdr,
        "fairness":     _jain(list(per_src.values())),
        "col_rate":     ato/tx,
        "retry_rate":   rtr/tx,
        "e_total_mj":   e_total*1000.0/n_stas,
        "e_sleep_mj":   e_sleep*1000.0/n_stas,
        "e_idle_mj":    e_idle *1000.0/n_stas,
        "ee_kbit_j":    ee,
        "e_per_pkt_uj": (e_total*1e6/n_stas)/max(1, del_/n_stas),
    }

_ZERO = {k: 0.0 for k in ["pdr","tput_kbps","avg_delay_ms","p95_delay_ms",
                            "drop_rate","fairness","col_rate","retry_rate",
                            "e_total_mj","e_sleep_mj","e_idle_mj","ee_kbit_j","e_per_pkt_uj"]}

def _ci(values: List[float]):
    n = len(values)
    if n == 0: return 0.0, 0.0, 0.0
    m = sum(values)/n
    if n == 1: return m, m, m
    std = math.sqrt(sum((v-m)**2 for v in values)/(n-1))
    h   = _T_CRIT.get(n-1, 2.0) * std / math.sqrt(n)
    return m, m-h, m+h

def _agg(rows: List[Dict]) -> Dict:
    out = {}
    for k in rows[0]:
        m, lo, hi = _ci([r[k] for r in rows])
        out[f"{k}_mean"] = round(m, 6)
        out[f"{k}_ci_lo"] = round(lo, 6)
        out[f"{k}_ci_hi"] = round(hi, 6)
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = len(IMPROVED)*len(STA_COUNTS)*len(SEEDS)
    done, rows_agg, rows_seed = 0, [], []
    t0 = time.time()
    print("="*72)
    print("  RL Improved Variants  (UCB1 + N-step + Soft target)")
    for lbl, bm, ex in IMPROVED:
        print(f"    {lbl:<12}: mode={bm}  extras={ex}")
    print(f"  {total} runs  |  N={STA_COUNTS}  seeds={SEEDS}")
    print("="*72)
    for label, base_mode, extra in IMPROVED:
        print(f"\n  === {label} ({base_mode}) ===")
        for n in STA_COUNTS:
            seed_rows = []
            for seed in SEEDS:
                done += 1
                el  = time.time()-t0
                eta = (el/done)*(total-done) if done > 1 else 0
                print(f"  [{done:>3}/{total}] {label:<12} N={n:>4} seed={seed} "
                      f"elapsed={el:.0f}s ETA={eta:.0f}s  ", end="\r", flush=True)
                try:
                    m = extract(build_and_run(n, seed, base_mode, extra))
                except Exception as exc:
                    print(f"\n  ERROR {label} N={n} seed={seed}: {exc}")
                    import traceback; traceback.print_exc()
                    m = dict(_ZERO)
                seed_rows.append(m)
                rows_seed.append({"label": label, "n": n, "seed": seed,
                                   **{k: round(v,6) for k,v in m.items()}})
            agg = _agg(seed_rows)
            rows_agg.append({"label": label, "n": n, **agg})
            print(f"\n  {label:<12} N={n:>4}  PDR={agg['pdr_mean']:.4f}  "
                  f"EE={agg['ee_kbit_j_mean']:.2f}kbit/J  "
                  f"delay={agg['avg_delay_ms_mean']:.0f}ms")
    print(f"\n  Total wall-clock: {time.time()-t0:.1f}s")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_agg[0].keys()))
        w.writeheader(); w.writerows(rows_agg)
    with open(OUT_CSV_SEED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_seed[0].keys()))
        w.writeheader(); w.writerows(rows_seed)
    print(f"  Saved: {OUT_CSV}")
    print(f"  Saved: {OUT_CSV_SEED}")

if __name__ == "__main__":
    main()
