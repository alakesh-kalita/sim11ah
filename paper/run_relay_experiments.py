"""
Relay topology evaluation — reproduces Table VI of sim11ah_ieee_paper.tex.

Sweeps N in {50, 100, 150, 200} for three topologies:
  1. Star    (R=0, no relay)
  2. Relay   R=2
  3. Relay   R=4

All cases use no-RAW DCF so PDR differences are topology-only.
Each point averaged over SEEDS = [42, 7, 99].

Outputs:
  paper/relay_results.csv   — full per-column data
  Console table             — matches Table VI in the paper
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder, RelayBuilder
from sim11ah.app import PeriodicTraffic

# -----------------------------------------------------------------------
# Experiment parameters  (must match Table II / Table V in the paper)
# -----------------------------------------------------------------------
STA_COUNTS   = [50, 100, 150, 200]
SIM_TIME_S   = 120.0
SEEDS        = [42, 7, 99]
TRAFFIC_INT  = 5.0          # periodic: one packet every 5 s per STA
PKT_SIZE     = 128          # bytes

ACCESS_CFG   = {"rate_bps": 300_000, "prop_delay": 300e-6, "per": 0.0}
BACKHAUL_CFG = {"rate_bps": 600_000, "prop_delay": 100e-6, "per": 0.0}

OUT_CSV = os.path.join(os.path.dirname(__file__), "relay_results.csv")


# -----------------------------------------------------------------------
# Build + run helpers
# -----------------------------------------------------------------------
def _build_and_run_star(num_stas: int, seed: int):
    cfg = default_config(raw_enable=False, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"]  = PKT_SIZE

    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(sim, num_stas=num_stas, link_cfg=ACCESS_CFG)

    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(PeriodicTraffic(TRAFFIC_INT) if nid > 0 else None)

    if 0 in sim.nodes:
        sim.nodes[0].mac.ap_start_beacons()
    for node in sim.nodes.values():
        node.start()

    sim.run_and_finalize(SIM_TIME_S)
    return sim


def _build_and_run_relay(num_stas: int, num_relays: int, seed: int):
    cfg = default_config(raw_enable=False, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"]  = PKT_SIZE

    # Ensure num_stas >= num_relays (paper requirement)
    num_stas = max(num_stas, num_relays)

    sim = Simulator(config=cfg, seed=seed)
    RelayBuilder.build(
        sim,
        num_relays=num_relays,
        num_stas=num_stas,
        backhaul_cfg=BACKHAUL_CFG,
        access_cfg=ACCESS_CFG,
    )

    for nid, node in sim.nodes.items():
        is_ap_or_relay = (nid == 0) or getattr(node, "is_relay", False)
        node.app.set_traffic_model(None if is_ap_or_relay else PeriodicTraffic(TRAFFIC_INT))

    if 0 in sim.nodes:
        sim.nodes[0].mac.ap_start_beacons()
    for node in sim.nodes.values():
        node.start()

    sim.run_and_finalize(SIM_TIME_S)
    return sim


# -----------------------------------------------------------------------
# Metric extraction  (mirrors run_paper_experiments.py)
# -----------------------------------------------------------------------
def _percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p
    lo  = int(idx)
    hi  = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def _extract(sim) -> dict:
    s   = sim.stats
    gen = max(1, int(getattr(s, "packets_generated", 0)))
    del_ = int(getattr(s, "packets_delivered",  0))
    dly  = list(getattr(s, "delays", []))
    dbytes = int(getattr(s, "delivered_bytes",   0))

    pdr       = del_ / gen
    tput_kbps = dbytes * 8.0 / SIM_TIME_S / 1000.0
    avg_delay = 1000.0 * (sum(dly) / len(dly) if dly else 0.0)
    p95_delay = 1000.0 * _percentile(dly, 0.95)

    return {
        "pdr":           round(pdr,       4),
        "tput_kbps":     round(tput_kbps, 2),
        "avg_delay_ms":  round(avg_delay, 1),
        "p95_delay_ms":  round(p95_delay, 1),
        "gen":           gen,
        "delivered":     del_,
    }


def _avg(dicts: list) -> dict:
    keys = dicts[0].keys()
    return {k: round(sum(d[k] for d in dicts) / len(dicts), 4) for k in keys}


# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------
def main():
    # Cases: (label, topology, num_relays)
    cases = [
        ("star",     "star",  0),
        ("relay_r2", "relay", 2),
        ("relay_r4", "relay", 4),
    ]

    total = len(STA_COUNTS) * len(cases) * len(SEEDS)
    done  = 0
    rows  = []

    print(f"\nRelay Topology Evaluation  —  {total} runs  ({len(SEEDS)} seeds each)\n")
    print(f"{'N':>5}  {'Case':<12}  {'PDR':>6}  {'Tput(kb/s)':>10}  "
          f"{'AvgDly(ms)':>10}  {'P95Dly(ms)':>10}")
    print("-" * 62)

    for n in STA_COUNTS:
        row = {"num_stas": n}

        for label, topo, nr in cases:
            seed_results = []
            for seed in SEEDS:
                done += 1
                print(f"  [{done:>2}/{total}] N={n:>3}  {label:<12}  seed={seed} ...",
                      end="\r", flush=True)
                try:
                    if topo == "star":
                        sim = _build_and_run_star(n, seed)
                    else:
                        sim = _build_and_run_relay(n, nr, seed)
                    seed_results.append(_extract(sim))
                except Exception as e:
                    print(f"\n  ERROR N={n} {label} seed={seed}: {e}")
                    seed_results.append({"pdr": 0.0, "tput_kbps": 0.0,
                                         "avg_delay_ms": 0.0, "p95_delay_ms": 0.0,
                                         "gen": 0, "delivered": 0})

            avg = _avg(seed_results)
            for k, v in avg.items():
                row[f"{label}_{k}"] = v

            print(f"  N={n:>3}  {label:<12}  "
                  f"{avg['pdr']:>6.3f}  "
                  f"{avg['tput_kbps']:>10.1f}  "
                  f"{avg['avg_delay_ms']:>10.1f}  "
                  f"{avg['p95_delay_ms']:>10.1f}")

        rows.append(row)
        print()

    # ── Save CSV ────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Paper table  (Table VI) ─────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  TABLE VI — Relay Topology Results (No-RAW DCF, 3-Seed Average)")
    print("=" * 76)
    hdr = (f"{'N':>5} | "
           f"{'Star PDR':>8} {'Tput':>6} {'Dly':>7} | "
           f"{'R=2 PDR':>7} {'Tput':>6} {'Dly':>7} | "
           f"{'R=4 PDR':>7} {'Tput':>6} {'Dly':>5}")
    print(hdr)
    print("-" * 76)
    for row in rows:
        n = row["num_stas"]
        print(
            f"{n:>5} | "
            f"{row['star_pdr']:>8.3f} "
            f"{row['star_tput_kbps']:>6.1f} "
            f"{row['star_avg_delay_ms']:>7.0f} | "
            f"{row['relay_r2_pdr']:>7.3f} "
            f"{row['relay_r2_tput_kbps']:>6.1f} "
            f"{row['relay_r2_avg_delay_ms']:>7.0f} | "
            f"{row['relay_r4_pdr']:>7.3f} "
            f"{row['relay_r4_tput_kbps']:>6.1f} "
            f"{row['relay_r4_avg_delay_ms']:>5.0f}"
        )
    print("=" * 76)
    print(f"\nFull data saved to: {OUT_CSV}\n")

    return rows


if __name__ == "__main__":
    main()
