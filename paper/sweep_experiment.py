"""
Parametric sweep: No-RAW DCF over (N, T_pkt) grid.
Produces paper/sweep_results.csv  and  paper/sweep_analytical.csv

N      : station counts
T_pkt  : packet generation interval per STA (seconds)
Metrics: PDR, avg_delay_ms, p95_delay_ms, retry_rate, throughput_kbps
"""

import csv, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config    import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology  import StarBuilder
from sim11ah.app       import PeriodicTraffic

# -----------------------------------------------------------------------
# Sweep grid
# -----------------------------------------------------------------------
STA_COUNTS   = [10, 25, 50, 100, 200]
PKT_INTERVALS = [1.0, 2.0, 5.0, 10.0, 20.0]   # seconds/pkt per STA

SIM_TIME_S   = 120.0
SEEDS        = [42, 7, 99]
PKT_SIZE     = 128          # bytes
LINK_RATE    = 150_000      # MCS0 bps
PROP_DELAY   = 300e-6

OUT_SIM  = os.path.join(os.path.dirname(__file__), "sweep_results.csv")
OUT_ANA  = os.path.join(os.path.dirname(__file__), "sweep_analytical.csv")

# -----------------------------------------------------------------------
# 802.11ah parameters (must match config defaults)
# -----------------------------------------------------------------------
SIGMA   = 52e-6
SIFS    = 160e-6
DIFS    = 264e-6
T_PRE   = 320e-6
T_HDR   = 80e-6
CW_MIN  = 15
CW_MAX  = 1023
RETRY   = 7               # short retry limit (= 7 retries = 8 attempts total)
M_STAGE = 6               # CW doublings: 15→31→63→127→255→511→1023

# Frame sizes (payload + protocol overhead observed in simulation)
# MAC overhead = 36 B (FC+Dur+3×Addr+SeqCtrl+CCMP+FCS)
# NET header   = 16 B (net_header_bytes in config)
# Transport adds no on-wire bytes
FRAME_B = PKT_SIZE + 36 + 16   # 180 B total (36 MAC + 16 NET + 128 payload)
ACK_B   = 14

T_DATA = T_PRE + T_HDR + FRAME_B * 8 / LINK_RATE
T_ACK  = T_PRE + T_HDR + ACK_B  * 8 / LINK_RATE
T_S    = T_DATA + PROP_DELAY + SIFS + T_ACK + PROP_DELAY + DIFS + SIGMA
T_C    = T_DATA + PROP_DELAY + DIFS + SIGMA   # collision slot


# -----------------------------------------------------------------------
# Analytical models
# -----------------------------------------------------------------------
def bianchi_fixed_point(n, W0=CW_MIN, m=M_STAGE):
    """Return (p_col, tau, S_bps) from Bianchi saturation model."""
    def tau_from_p(p):
        two_p = 2.0 * p
        denom = (1.0 - two_p) * (W0 + 1) + p * W0 * (1.0 - two_p ** m)
        return 2.0 * (1.0 - two_p) / denom if abs(denom) > 1e-15 else 2.0 / (W0 + 1)

    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        tau = tau_from_p(mid)
        p_check = 1.0 - (1.0 - tau) ** (n - 1)
        (lo if p_check > mid else hi).__class__  # dummy
        if p_check > mid:
            lo = mid
        else:
            hi = mid
    p   = 0.5 * (lo + hi)
    tau = tau_from_p(p)

    P_tr = 1.0 - (1.0 - tau) ** n
    if P_tr < 1e-15:
        return p, tau, 0.0
    P_s = n * tau * (1.0 - tau) ** (n - 1) / P_tr
    num = P_s * P_tr * PKT_SIZE * 8
    den = (1.0 - P_tr) * SIGMA + P_tr * P_s * T_S + P_tr * (1.0 - P_s) * T_C
    return p, tau, num / den          # throughput in bps


def mean_backoff_at_stage(k):
    """Mean backoff duration at retry stage k."""
    cw = min(CW_MIN * (2 ** k), CW_MAX)
    return (cw + 1) / 2.0 * SIGMA   # seconds


def analytical_service_delay(p):
    """
    E[service time per packet] including backoff and retransmissions.
    Uses exact geometric-series formula over retry stages 0..RETRY.
    """
    E_d = 0.0
    cum_backoff = 0.0
    for k in range(RETRY + 1):          # attempts 0..RETRY (0=first, RETRY=last retry)
        prob_succeed = (1.0 - p) * (p ** k) if k < RETRY else p ** k  # last = drop
        if k < RETRY:
            # time to reach attempt k: k collisions + backoffs + T_C each, then one success T_S
            cum_backoff += mean_backoff_at_stage(k)
            t_reach_k = k * T_C + cum_backoff
            E_d += prob_succeed * (t_reach_k + T_S)
        # dropped frames contribute 0 measured delay (they never arrive)

    return E_d   # seconds


def analytical_metrics(n, t_pkt):
    """Return dict of analytical predictions for n stations, t_pkt interval."""
    offered_bps = n * PKT_SIZE * 8 / t_pkt
    p_col, tau, S_sat = bianchi_fixed_point(n)

    # Traffic intensity
    rho = offered_bps / S_sat if S_sat > 0 else float('inf')

    # PDR: for rho<1 → light load (use collision-based drop prob)
    #      for rho≥1 → throughput-saturated
    p_drop = p_col ** (RETRY + 1)       # all retry+1 = 8 attempts fail
    pdr_coll = 1.0 - p_drop             # collision-model PDR
    pdr_sat  = S_sat / offered_bps      # saturation-capped PDR
    pdr_a    = min(pdr_coll, pdr_sat)
    pdr_a    = max(0.0, pdr_a)

    # Throughput
    tput_a = pdr_a * offered_bps

    # Mean delay (service only; valid for rho < ~0.8; blows up near saturation)
    E_srv = analytical_service_delay(p_col)

    # M/G/1 queuing correction (Pollaczek–Khinchine, C²_s ≈ 1 for geometric retry)
    if rho < 0.95:
        E_wait = rho * E_srv / (1.0 - rho)  # M/G/1 mean waiting (C²=1)
    else:
        E_wait = float('inf')
    E_delay_a = E_srv + E_wait
    delay_ms_a = E_delay_a * 1000.0

    return {
        "offered_kbps": round(offered_bps / 1000, 3),
        "S_sat_kbps":   round(S_sat / 1000, 2),
        "rho":          round(rho, 4),
        "p_col":        round(p_col, 4),
        "pdr_a":        round(pdr_a, 4),
        "tput_a_kbps":  round(tput_a / 1000, 3),
        "delay_ms_a":   round(delay_ms_a, 1) if delay_ms_a < 1e6 else 99999.0,
    }


# -----------------------------------------------------------------------
# Simulation helpers
# -----------------------------------------------------------------------
def percentile(data, p):
    if not data: return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p
    lo  = int(idx)
    hi  = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def run_one(n, t_pkt, seed):
    cfg = default_config(raw_enable=False, traffic_mode="periodic")
    cfg["app"]["periodic_interval"]  = t_pkt
    cfg["app"]["packet_size_bytes"]  = PKT_SIZE
    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(sim, num_stas=n,
                      link_cfg={"rate_bps": LINK_RATE,
                                "prop_delay": PROP_DELAY,
                                "per": 0.0})
    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(
            PeriodicTraffic(t_pkt) if nid > 0 else None
        )
    sim.run_and_finalize(SIM_TIME_S)
    return sim


def extract(sim, t_pkt):
    s   = sim.stats
    gen = max(1, int(getattr(s, "packets_generated", 0)))
    dly = list(getattr(s, "delays", []))
    pdr = getattr(s, "packets_delivered", 0) / gen
    tx  = max(1, int(getattr(s, "mac_tx_attempts", 0)))
    rtr = int(getattr(s, "mac_retries", 0))
    return {
        "pdr":         round(pdr, 4),
        "avg_delay_ms": round(1000.0 * (sum(dly) / len(dly) if dly else 0.0), 2),
        "p95_delay_ms": round(1000.0 * percentile(dly, 0.95), 2),
        "retry_rate":   round(rtr / tx, 4),
        "tput_sim_kbps": round(pdr * sim.stats.packets_generated /
                               SIM_TIME_S * PKT_SIZE * 8 / 1000, 3)
                          if False else  # use offered×PDR formula
                         round(pdr * (SIM_TIME_S / t_pkt) * PKT_SIZE * 8 / SIM_TIME_S / 1000, 3),
    }


def avg_dicts(dicts):
    keys = dicts[0].keys()
    return {k: round(sum(d[k] for d in dicts) / len(dicts), 4) for k in keys}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    total = len(STA_COUNTS) * len(PKT_INTERVALS) * len(SEEDS)
    done  = 0
    sim_rows = []
    ana_rows = []

    for n in STA_COUNTS:
        for t_pkt in PKT_INTERVALS:
            seed_results = []
            for seed in SEEDS:
                done += 1
                print(f"[{done:>3}/{total}] N={n:>3}  T_pkt={t_pkt:>4}s  seed={seed}")
                sim  = run_one(n, t_pkt, seed)
                seed_results.append(extract(sim, t_pkt))
            avg = avg_dicts(seed_results)
            sim_rows.append({"N": n, "t_pkt_s": t_pkt, **avg})

            ana = analytical_metrics(n, t_pkt)
            ana_rows.append({"N": n, "t_pkt_s": t_pkt, **ana})

            print(f"  sim  PDR={avg['pdr']:.3f}  delay={avg['avg_delay_ms']:.1f}ms  "
                  f"retry={avg['retry_rate']:.3f}")
            print(f"  ana  PDR={ana['pdr_a']:.3f}  delay={ana['delay_ms_a']:.1f}ms  "
                  f"rho={ana['rho']:.3f}")

    # Save CSVs
    with open(OUT_SIM, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sim_rows[0].keys()))
        writer.writeheader(); writer.writerows(sim_rows)
    print(f"\nSim results → {OUT_SIM}")

    with open(OUT_ANA, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ana_rows[0].keys()))
        writer.writeheader(); writer.writerows(ana_rows)
    print(f"Ana results → {OUT_ANA}")

    return sim_rows, ana_rows


if __name__ == "__main__":
    main()
