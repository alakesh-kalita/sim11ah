"""
Multi-AP Roaming Experiment
============================
Walks a single STA across a 2-AP corridor at a chosen constant speed and
measures how connectivity degrades as its residence time inside the
AP-overlap region (overlap_width_m / speed_mps) drops toward, and below,
the time a real handover takes to complete.

This is the actual research deliverable for the multi-AP roaming work: it
answers "how can a STA maintain uninterrupted communication when its
residence time in an overlap region is shorter than conventional handover
preparation time?" empirically, using the simulator's real MAC/PHY/
association stack (see sim11ah/mac/association.py's RSSI+hysteresis
_maybe_roam) rather than an abstract model.

Per-run metrics (see RunResult):
  pdr_overall     — delivered / generated, whole run. The primary signal:
                    this catches the FULL extent of roaming-induced loss,
                    including packets that time out shortly AFTER the
                    crossing because they were stuck behind a queue backlog
                    built up DURING the crossing (confirmed by tracing
                    individual packets: a STA can show strong RSSI and a
                    completed handover, yet still drop freshly-generated
                    packets for a few more seconds while that backlog
                    drains against max_msdu_lifetime_s).
  pdr_crossing    — delivered / generated, restricted to packets GENERATED
                    while the STA's x-position was inside the overlap band.
                    A narrower, conservative measurement -- it undercounts
                    the delayed backlog-drain losses described above, since
                    those are attributed to whatever position the STA was
                    at when each individual packet was generated, not when
                    it was actually dropped. Use pdr_overall as the
                    headline number; pdr_crossing is a secondary check that
                    loss is concentrated near the crossing at all.
  connectivity_gap_s — sum of (next ASSOC_ASSOCIATED - ASSOC_HANDOVER) for
                    every roam that happened during the crossing; this is
                    the actual "how long was the STA disconnected during a
                    proactive handover" measurement this feature exists to
                    produce. link_loss_gap_s separately reports any full
                    ASSOC_LINK_LOST -> ASSOCIATED gap (a STA that dropped
                    off BOTH APs entirely, a worse failure mode than a
                    clean handover).
  handover_count  — number of ASSOC_HANDOVER events during the run

Choosing --ap-spacing: the default (1600m) is sized against the default
PHY's ~994m nominal range to leave a genuine partial-overlap band ("AP0
only" / "overlap" / "AP1 only" as three distinct regions along the
corridor) -- a spacing much smaller than 2x the nominal range means both
APs blanket the ENTIRE corridor, leaving no non-overlap region to contrast
against. If you change PHY params (freq/eirp/sensitivity), recheck via
sim.nodes[0].phy.nominal_range_m() before picking a spacing.

Usage
-----
  Single run:
    python scripts/roam_experiment.py --speed 5.0 --seed 42

  Sweep (writes a CSV):
    python scripts/roam_experiment.py --sweep-speed 0.5,1,2,5,10,20,40,80,160 \
        --seeds 42,7,99 --out results/roam_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import MultiApBuilder
from sim11ah.app import PeriodicTraffic
from sim11ah.mobility import corridor_step


# ---------------------------------------------------------------------------
# Sim construction
# ---------------------------------------------------------------------------
def build_roam_sim(
    seed: int,
    num_aps: int,
    ap_spacing_m: float,
    packet_interval: float,
    packet_size: int,
    raw_enable: bool,
) -> Tuple[Simulator, int]:
    """Build a 2+-AP corridor with a single mobile STA (the only STA --
    keeps the residence-time effect clean and unconfounded by background
    contention, unlike the static multi_ap GUI/CLI topologies which scatter
    many STAs; that's a separate, future knob if background load turns out
    to matter for this specific research question)."""
    cfg = default_config(raw_enable=raw_enable, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = float(packet_interval)
    cfg["app"]["packet_size_bytes"] = int(packet_size)
    sim = Simulator(config=cfg, seed=seed)

    link_cfg = {"rate_bps": 300_000, "prop_delay": 3e-4, "per": 0.0}
    MultiApBuilder.build(
        sim, num_aps=int(num_aps), ap_spacing_m=float(ap_spacing_m),
        num_stas=1, link_cfg=link_cfg,
    )
    sta_id = int(num_aps)

    for nid, node in sim.nodes.items():
        if node.is_ap:
            node.app.set_traffic_model(None)
        else:
            node.app.set_traffic_model(PeriodicTraffic(float(packet_interval)))

    # Phase-stagger each AP's first beacon (see mac/facade.py's
    # ap_start_beacons) before node.start() -- ap_start_beacons() is
    # idempotent so this explicit call wins over start()'s automatic
    # offset-0.0 follow-up call. Same pattern as scripts/main_cli.py.
    ap_ids = sim.config.get("topology", {}).get("ap_ids", [])
    beacon_interval = float(cfg["mac"]["beacon_interval"])
    for idx, ap_id in enumerate(ap_ids):
        sim.nodes[ap_id].mac.ap_start_beacons(
            phase_offset_s=idx * beacon_interval / max(1, len(ap_ids))
        )
    for node in sim.nodes.values():
        node.start()

    return sim, sta_id


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    seed: int
    ap_spacing_m: float
    ap_range_m: float
    overlap_width_m: float
    speed_mps: float
    residence_time_s: float
    beacon_interval_s: float
    pdr_overall: float
    pdr_crossing: float
    generated_overall: int
    generated_crossing: int
    connectivity_gap_s: float
    link_loss_gap_s: float
    handover_count: int


def run_one(
    seed: int,
    speed_mps: float,
    ap_spacing_m: float = 1600.0,
    num_aps: int = 2,
    packet_interval: float = 0.5,
    packet_size: int = 128,
    raw_enable: bool = True,
    settle_s: float = 5.0,
    dt: float = 0.05,
) -> RunResult:
    sim, sta_id = build_roam_sim(
        seed, num_aps, ap_spacing_m, packet_interval, packet_size, raw_enable,
    )

    ap_range_m = float(sim.nodes[0].phy.nominal_range_m())
    overlap_width_m = float(MultiApBuilder.overlap_width_m(sim, ap_spacing_m))
    # The overlap band along the corridor: where both AP 0 (at x=0) and the
    # last AP (at x=(num_aps-1)*ap_spacing_m) are simultaneously in nominal
    # range. Only meaningful for the classic 2-AP case this script targets;
    # num_aps>2 would need a per-segment band instead of one global one.
    last_ap_x = (num_aps - 1) * ap_spacing_m
    overlap_lo = max(0.0, last_ap_x - ap_range_m)
    overlap_hi = min(last_ap_x, ap_range_m)

    beacon_interval = float(sim.config["mac"]["beacon_interval"])

    # Let the STA associate with its starting AP before moving.
    start_pos = (0.0, 0.0)
    end_pos = (last_ap_x, 0.0)
    sim.nodes[sta_id].pos = start_pos
    sim.run_for(settle_s)

    moving = True
    while moving:
        moving = corridor_step(sim, sta_id, dt, speed_mps, start_pos, end_pos)
        sim.run_for(dt)

    # Drain long enough for every packet generated right up to the moment
    # motion stopped to reach its natural verdict (delivered, or dropped on
    # max_msdu_lifetime_s) before finalize() -- a fixed settle_s here would
    # otherwise chop off in-flight packets mid-retry and miscount them as
    # roaming-caused losses. Confirmed empirically: with settle_s==
    # max_msdu_lifetime_s exactly, the last few packets generated in the
    # final beacon_interval before motion stopped hadn't reached their
    # timeout yet, showing up as spurious "undelivered" packets at the
    # destination AP where the link was actually fine.
    msdu_lifetime_s = float(sim.config["mac"].get("max_msdu_lifetime_s", 5.0))
    drain_s = max(float(settle_s), msdu_lifetime_s + beacon_interval)
    sim.run_for(drain_s)
    sim.finalize()

    logs = sim.logger.logs

    # Was the STA physically inside the overlap band at a given time? Built
    # from the same per-step positions corridor_step just drove, converted
    # to a time->x lookup via the GENERATE events' own timestamps below
    # (linear interpolation along the known constant-speed corridor instead
    # of re-simulating position, since x(t) is exactly known in closed form
    # for a straight constant-speed walk starting at settle_s).
    def x_at(t: float) -> float:
        elapsed = t - settle_s
        if elapsed <= 0.0:
            return start_pos[0]
        x = start_pos[0] + speed_mps * elapsed
        return min(end_pos[0], x)

    gen_events = [
        e for e in logs
        if e["event"] == "GENERATE" and e["node_id"] == sta_id
    ]
    deliver_seqs = {
        e["packet_seq"] for e in logs
        if e["event"] == "DELIVER" and e.get("src") == sta_id
    }

    generated_overall = len(gen_events)
    delivered_overall = sum(1 for e in gen_events if e["packet_seq"] in deliver_seqs)

    crossing_gen = [
        e for e in gen_events
        if overlap_lo <= x_at(float(e["time"])) <= overlap_hi
    ]
    generated_crossing = len(crossing_gen)
    delivered_crossing = sum(1 for e in crossing_gen if e["packet_seq"] in deliver_seqs)

    pdr_overall = delivered_overall / generated_overall if generated_overall else float("nan")
    pdr_crossing = (
        delivered_crossing / generated_crossing if generated_crossing else float("nan")
    )

    # Connectivity gap: from a proactive roam (ASSOC_HANDOVER, the state
    # goes ASSOCIATED -> UNASSOCIATED right there -- see association.py's
    # _roam_to) to the next ASSOC_ASSOCIATED for this STA. This is the
    # actual "how long was the STA disconnected during a handover" figure
    # the whole feature exists to measure.
    sta_assoc_events = [
        e for e in logs
        if e["node_id"] == sta_id and e["layer"] == "MAC"
        and e["event"] in ("ASSOC_HANDOVER", "ASSOC_ASSOCIATED", "ASSOC_LINK_LOST")
    ]
    sta_assoc_events.sort(key=lambda e: e["time"])

    connectivity_gap_s = 0.0
    link_loss_gap_s = 0.0
    handover_count = 0
    pending_gap_start = None
    pending_gap_kind = None
    for e in sta_assoc_events:
        if e["event"] == "ASSOC_HANDOVER":
            handover_count += 1
            if pending_gap_start is None:
                pending_gap_start = float(e["time"])
                pending_gap_kind = "handover"
        elif e["event"] == "ASSOC_LINK_LOST":
            if pending_gap_start is None:
                pending_gap_start = float(e["time"])
                pending_gap_kind = "link_loss"
        elif e["event"] == "ASSOC_ASSOCIATED":
            if pending_gap_start is not None:
                gap = float(e["time"]) - pending_gap_start
                if pending_gap_kind == "handover":
                    connectivity_gap_s += gap
                else:
                    link_loss_gap_s += gap
                pending_gap_start = None
                pending_gap_kind = None

    # A handover/link-loss that never resolved by the end of the run (the
    # STA never made it back to ASSOCIATED, e.g. under extreme speed/
    # congestion) would otherwise vanish from the total instead of counting
    # as the worst-case outcome it actually is -- close it out against the
    # simulation's final time.
    if pending_gap_start is not None:
        gap = float(sim.stats.sim_time) - pending_gap_start
        if pending_gap_kind == "handover":
            connectivity_gap_s += gap
        else:
            link_loss_gap_s += gap

    return RunResult(
        seed=seed,
        ap_spacing_m=float(ap_spacing_m),
        ap_range_m=ap_range_m,
        overlap_width_m=overlap_width_m,
        speed_mps=float(speed_mps),
        residence_time_s=(overlap_width_m / speed_mps) if speed_mps > 0 else float("inf"),
        beacon_interval_s=beacon_interval,
        pdr_overall=pdr_overall,
        pdr_crossing=pdr_crossing,
        generated_overall=generated_overall,
        generated_crossing=generated_crossing,
        connectivity_gap_s=connectivity_gap_s,
        link_loss_gap_s=link_loss_gap_s,
        handover_count=handover_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-AP roaming residence-time experiment")
    parser.add_argument("--speed", type=float, default=5.0,
                         help="Single-run STA speed in m/s (ignored if --sweep-speed given)")
    parser.add_argument("--sweep-speed", type=str, default=None,
                         help="Comma-separated list of speeds (m/s) to sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                         help="Comma-separated list of seeds (overrides --seed)")
    parser.add_argument("--ap-spacing", type=float, default=1600.0,
                         help="AP separation in m. Default gives a partial "
                              "overlap band (~390m) against the default PHY's "
                              "~994m nominal range -- a much smaller spacing "
                              "(e.g. 400m) means both APs blanket the entire "
                              "corridor, leaving no non-overlap region to "
                              "contrast against. If you change PHY params "
                              "(freq/eirp/sensitivity), recheck via "
                              "sim.nodes[0].phy.nominal_range_m().")
    parser.add_argument("--num-aps", type=int, default=2)
    parser.add_argument("--packet-interval", type=float, default=0.5,
                         help="Default keeps baseline (non-crossing) PDR "
                              "near 100%% under RAW's periodic per-slot "
                              "access cadence for a single STA -- a more "
                              "aggressive rate (e.g. 0.2s) saturates the "
                              "queue on its own, confounding any PDR dip "
                              "with ordinary overload rather than roaming.")
    parser.add_argument("--packet-size", type=int, default=128)
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path (required when sweeping)")
    args = parser.parse_args()

    speeds = _parse_float_list(args.sweep_speed) if args.sweep_speed else [args.speed]
    seeds = _parse_int_list(args.seeds) if args.seeds else [args.seed]

    results: List[RunResult] = []
    for speed in speeds:
        for seed in seeds:
            r = run_one(
                seed=seed,
                speed_mps=speed,
                ap_spacing_m=args.ap_spacing,
                num_aps=args.num_aps,
                packet_interval=args.packet_interval,
                packet_size=args.packet_size,
                raw_enable=not args.no_raw,
            )
            results.append(r)
            print(
                f"speed={speed:7.2f} m/s  seed={seed:4d}  "
                f"residence={r.residence_time_s:7.3f}s  "
                f"beacon_iv={r.beacon_interval_s:.3f}s  "
                f"pdr_overall={r.pdr_overall:.3f}  pdr_crossing={r.pdr_crossing:.3f}  "
                f"handovers={r.handover_count}  conn_gap={r.connectivity_gap_s:.3f}s  "
                f"link_loss_gap={r.link_loss_gap_s:.3f}s"
            )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
