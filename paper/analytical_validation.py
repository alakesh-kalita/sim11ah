"""
Analytical validation: Bianchi DCF saturation model vs simulation results.

For each N, computes:
  1. Offered load (light load: N × 128B × 8 / 5s)
  2. Bianchi saturation throughput (heavy load analytical bound)
  3. RAW capacity upper bound (slot budget analysis)

Uses the 802.11ah parameters from the simulation.
"""

import math
import csv
import os

# -----------------------------------------------------------------------
# 802.11ah physical / MAC parameters (must match sim config)
# -----------------------------------------------------------------------
SIGMA       = 52e-6        # slot time (s)
SIFS        = 160e-6       # SIFS (s)
DIFS        = 264e-6       # DIFS = SIFS + 2*sigma (s)
CW_MIN      = 15
CW_MAX      = 1023
RATE_BPS    = 150_000      # MCS0
T_PRE       = 320e-6       # preamble
T_HDR       = 80e-6        # header

# Frame composition (must match sim models)
PAYLOAD_B   = 128
# MAC + NET + Transport overhead observed in sim frames: ~34B total
FRAME_B     = PAYLOAD_B + 34   # 162 B
ACK_B       = 14               # short ACK frame

T_DATA = T_PRE + T_HDR + FRAME_B * 8 / RATE_BPS       # ~9.06 ms
T_ACK  = T_PRE + T_HDR + ACK_B  * 8 / RATE_BPS       # ~1.15 ms
PROP   = 300e-6                                         # propagation delay

# Successful exchange duration (basic access)
T_S = T_DATA + PROP + SIFS + T_ACK + PROP + DIFS + SIGMA
# Collision duration (transmitter waits T_DATA + DIFS + sigma before retry)
T_C = T_DATA + PROP + DIFS + SIGMA

# Traffic model
TRAFFIC_INTERVAL = 5.0  # seconds/packet per STA
SIM_TIME = 120.0

# Experiment N values
STA_COUNTS = [10, 25, 50, 75, 100, 150, 200]

# RAW parameters
RAW_GROUPS     = 4
RAW_SLOTS      = 4
RAW_SLOT_DUR   = 7e-3       # 7 ms
BEACON_INT     = 500e-3     # 500 ms


# -----------------------------------------------------------------------
# Bianchi fixed-point solver (no external deps)
# -----------------------------------------------------------------------
def _bianchi_tau(p, W0, m):
    """Equation (1): tau from p."""
    two_p = 2.0 * p
    if abs(1.0 - two_p) < 1e-15:
        return 2.0 / (W0 + 1)
    num = 2.0 * (1.0 - two_p)
    den = ((1.0 - two_p) * (W0 + 1)
           + p * W0 * (1.0 - two_p ** m))
    return num / den


def bianchi_throughput(n, W0=CW_MIN, m=6):
    """
    Return (throughput_bps, collision_prob, tx_prob) under Bianchi
    saturation model for n stations.

    m = number of backoff stages before CW stops doubling.
    For CW_min=15, CW_max=1023: m=6 (15→31→…→1023).
    """
    # Fixed-point: find p ∈ (0,1) such that p = 1 - (1-tau(p))^(n-1)
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        tau = _bianchi_tau(mid, W0, m)
        p_check = 1.0 - (1.0 - tau) ** (n - 1)
        if p_check > mid:
            lo = mid
        else:
            hi = mid
    p   = 0.5 * (lo + hi)
    tau = _bianchi_tau(p, W0, m)

    P_tr = 1.0 - (1.0 - tau) ** n
    if P_tr < 1e-15:
        return 0.0, p, tau
    P_s = n * tau * (1.0 - tau) ** (n - 1) / P_tr

    # Normalized throughput (fraction of time carrying payload bits)
    num = P_s * P_tr * PAYLOAD_B * 8
    den = (1.0 - P_tr) * SIGMA + P_tr * P_s * T_S + P_tr * (1.0 - P_s) * T_C

    norm_S = num / den                      # bits per second (normalized)
    return norm_S, p, tau


# -----------------------------------------------------------------------
# Offered load (light-load, unsaturated)
# -----------------------------------------------------------------------
def offered_load_bps(n):
    """Total offered load if every STA always has a packet to send at T=5s."""
    return n * PAYLOAD_B * 8 / TRAFFIC_INTERVAL


# -----------------------------------------------------------------------
# RAW capacity upper bound
# -----------------------------------------------------------------------
def raw_capacity_bound():
    """
    Upper bound on RAW throughput given slot budget.

    With T_data ≈ 9 ms > D = 7 ms, each exchange requires cross-slot mode,
    consuming two consecutive sub-slots (current + next).
    Effective transmission opportunities per group per beacon interval:
        floor(S / 2) = 2  (for S=4 sub-slots)
    Total per beacon interval: G × floor(S/2) = 4×2 = 8 frames
    """
    slots_per_group = RAW_SLOTS
    effective_per_group = slots_per_group // 2  # cross-slot pairs
    frames_per_beacon  = RAW_GROUPS * effective_per_group
    pkt_per_sec = frames_per_beacon / BEACON_INT
    return pkt_per_sec * PAYLOAD_B * 8   # bps


# -----------------------------------------------------------------------
# Simulation results (from paper_results.csv)
# -----------------------------------------------------------------------
def load_sim_results():
    csv_path = os.path.join(os.path.dirname(__file__), "paper_results.csv")
    rows = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["num_stas"])
            rows[n] = {k: float(v) for k, v in row.items() if k != "num_stas"}
    return rows


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    sim = load_sim_results()
    cap = raw_capacity_bound()

    print(f"\n{'='*72}")
    print(f"{'N':>5}  {'Offered':>9}  {'Bianchi-Sat':>11}  {'SimDCF-Tput':>12}  "
          f"{'RAW-Cap':>9}  {'SimRAW-Tput':>12}  {'p_col':>7}  {'p_drop':>7}")
    print(f"{'':>5}  {'kb/s':>9}  {'kb/s':>11}  {'kb/s':>12}  "
          f"{'kb/s':>9}  {'kb/s':>12}  {'':>7}  {'':>7}")
    print(f"{'-'*72}")

    rows_out = []
    for n in STA_COUNTS:
        offered   = offered_load_bps(n) / 1000  # kb/s
        b_tput, p_col, tau = bianchi_throughput(n)
        b_tput_kbps = b_tput / 1000

        s = sim[n]
        # Throughput = PDR × offered_load = PDR × N × L × 8 / T_pkt
        sim_dcf_tput = s["no_raw_pdr"]    * n * PAYLOAD_B * 8 / TRAFFIC_INTERVAL / 1000
        sim_raw_tput = s["static_raw_pdr"] * n * PAYLOAD_B * 8 / TRAFFIC_INTERVAL / 1000
        cap_kbps = cap / 1000

        # Bianchi predicts: in saturation, throughput ≈ b_tput_kbps
        # Frame drop rate (retry exhaustion) under saturation:
        # p_drop_per_frame = p^(m+1)  (all m+1 = 7 retries collide)
        p_drop = p_col ** (7)  # 7 retries * collision probability

        print(f"{n:>5}  {offered:>9.1f}  {b_tput_kbps:>11.2f}  {sim_dcf_tput:>12.2f}  "
              f"{cap_kbps:>9.2f}  {sim_raw_tput:>12.2f}  {p_col:>7.4f}  {p_drop:>7.4f}")

        rows_out.append({
            "N":            n,
            "offered_kbps": round(offered, 2),
            "bianchi_sat_kbps": round(b_tput_kbps, 2),
            "sim_dcf_tput_kbps": round(sim_dcf_tput, 2),
            "raw_cap_kbps": round(cap_kbps, 2),
            "sim_raw_tput_kbps": round(sim_raw_tput, 2),
            "p_collision": round(p_col, 4),
            "p_frame_drop": round(p_drop, 6),
        })

    print(f"{'='*72}")
    print(f"\nNotes:")
    print(f"  T_data  = {T_DATA*1000:.2f} ms  (128B payload + 34B overhead @ MCS0 150 kb/s)")
    print(f"  T_ACK   = {T_ACK*1000:.2f} ms")
    print(f"  T_S     = {T_S*1000:.2f} ms  (successful exchange)")
    print(f"  T_C     = {T_C*1000:.2f} ms  (collision)")
    print(f"  RAW cap = {cap/1000:.2f} kb/s  (cross-slot: floor({RAW_SLOTS}/2)={RAW_SLOTS//2} "
          f"pairs x {RAW_GROUPS} groups / {BEACON_INT*1000:.0f} ms)")
    print(f"  Bianchi: W0={CW_MIN}, m=6 (CW doubles 15→31→…→1023), basic access")
    print()

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "analytical_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved: {out_path}")
    return rows_out


if __name__ == "__main__":
    main()
