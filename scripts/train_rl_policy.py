"""
Multi-Episode RL Training for IEEE 802.11ah RAW Policy
=======================================================
Trains each RL controller (tabular, tabular_ddqn, dqn, ddqn, ppo) across
multiple episodes (simulation runs) for each STA count, stopping early once
PDR has converged.

Convergence rule
-----------------
After at least MIN_EPISODES, training for a given (mode, N) stops once the
PDR change between the last two consecutive episodes is below
CONV_THRESHOLD for two checks in a row (i.e. PDR has stabilised), or once
MAX_EPISODES is reached.

Each episode picks up where the previous one left off (Q-table / network
weights persisted to results/rl_qtables/qtable_<mode>_N<n>.pkl).
After training, compare_rl_trained.py loads these files for evaluation.

Usage
-----
  # Train one mode across all N values (stops early on convergence)
  python scripts/train_rl_policy.py --mode tabular

  # Train all five RL modes across all N values
  python scripts/train_rl_policy.py --mode all

  # Custom N values / episode cap
  python scripts/train_rl_policy.py --mode dqn --n-values 100 200 --max-episodes 10
"""

from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_N_VALUES = [100, 200, 400, 600, 800, 1000]
RL_MODES         = ["tabular", "tabular_ddqn", "dqn", "ddqn", "ppo"]

MIN_EPISODES   = 3
MAX_EPISODES   = 20
CONV_THRESHOLD = 0.01   # |ΔPDR| below this for 2 consecutive episodes => converged

SIM_TIME_S     = 120.0
TRAFFIC_INT    = 5.0
PKT_SIZE       = 128
LINK_RATE      = 150_000
PROP_DELAY     = 300e-6
RAW_NUM_SLOTS  = 8
RAW_SLOT_DUR_S = 0.014

# Training seeds: deliberately distinct from evaluation seeds [42, 7, 99]
# so the Q-table / network learns GENERAL patterns, not seed-specific behaviors.
TRAIN_SEEDS = [0, 100, 200, 300, 400, 13, 77, 31, 55, 88,
               21, 34, 89, 144, 233, 377, 610, 987, 1597, 2584]

OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "results")
QTABLE_DIR = os.path.join(OUT_DIR, "rl_qtables")


def qtable_path(mode: str, n: int) -> str:
    return os.path.join(QTABLE_DIR, f"qtable_{mode}_N{n}.pkl")


# ---------------------------------------------------------------------------
# Single training episode
# ---------------------------------------------------------------------------
def run_episode(mode: str, n: int, seed: int, qpath: str) -> Dict[str, float]:
    cfg = default_config(raw_enable=True, traffic_mode="periodic")
    cfg["app"]["periodic_interval"] = TRAFFIC_INT
    cfg["app"]["packet_size_bytes"]  = PKT_SIZE
    cfg["mac"]["raw_policy"]        = "rl"
    cfg["mac"]["rl_raw_mode"]       = mode
    cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
    cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S
    cfg["mac"]["rl_raw_qtable_path"] = qpath   # load previous + save new

    sim = Simulator(config=cfg, seed=seed)
    StarBuilder.build(
        sim, num_stas=n,
        link_cfg={"rate_bps": LINK_RATE, "prop_delay": PROP_DELAY, "per": 0.0},
    )
    for nid, node in sim.nodes.items():
        node.app.set_traffic_model(PeriodicTraffic(TRAFFIC_INT) if nid > 0 else None)

    sim.run_and_finalize(SIM_TIME_S)

    policy = sim.nodes[0].mac.raw.policy
    policy.save_qtable()

    s   = sim.stats
    gen = max(1, int(getattr(s, "packets_generated", 0)))
    pdr = int(getattr(s, "packets_delivered",   0)) / gen

    return {
        "pdr":            pdr,
        "qtable_states":  len(policy._qtable),
        "epsilon":        policy.epsilon,
        "episode_count":  policy._episode_count + 1,
        "update_count":   policy._update_count,
    }


# ---------------------------------------------------------------------------
# Training loop with convergence-based early stopping
# ---------------------------------------------------------------------------
def train(mode: str, n_values: List[int]) -> List[Dict]:
    os.makedirs(QTABLE_DIR, exist_ok=True)
    rows: List[Dict] = []

    for n in n_values:
        qp = qtable_path(mode, n)
        if os.path.exists(qp):
            os.remove(qp)

        print(f"\n  [{mode}] N={n:<5} training:", end="", flush=True)
        pdr_history: List[float] = []

        for ep in range(MAX_EPISODES):
            seed = TRAIN_SEEDS[ep % len(TRAIN_SEEDS)]
            t0   = time.time()
            res  = run_episode(mode, n, seed, qp)
            dt   = time.time() - t0

            pdr_history.append(res["pdr"])
            print(
                f"  ep{ep+1}→PDR={res['pdr']:.3f}"
                f"(ε={res['epsilon']:.3f},{res['qtable_states']}Q,{dt:.0f}s)",
                end="", flush=True,
            )
            rows.append({"mode": mode, "n": n, "episode": ep + 1, "seed": seed, **res})

            if ep + 1 >= MIN_EPISODES:
                d1 = abs(pdr_history[-1] - pdr_history[-2])
                d2 = abs(pdr_history[-2] - pdr_history[-3])
                if d1 < CONV_THRESHOLD and d2 < CONV_THRESHOLD:
                    print(f"  [converged after {ep+1} episodes]", end="", flush=True)
                    break

        print()  # newline after N's episodes

    return rows


def append_training_log(rows: List[Dict]) -> None:
    log_path = os.path.join(OUT_DIR, "rl_training_log.csv")
    file_exists = os.path.exists(log_path)
    fieldnames = ["mode", "n", "episode", "seed", "pdr",
                   "qtable_states", "epsilon", "episode_count", "update_count"]
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  Training log: {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global MAX_EPISODES
    parser = argparse.ArgumentParser(
        description="Multi-episode (convergence-based) training for the "
                     "IEEE 802.11ah RL RAW policy"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=RL_MODES + ["all"],
        help="RL controller to train (default: all five)"
    )
    parser.add_argument(
        "--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES,
        help=f"STA counts to train (default: {DEFAULT_N_VALUES})"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=MAX_EPISODES,
        help=f"Maximum training episodes per N (default: {MAX_EPISODES})"
    )
    args = parser.parse_args()

    MAX_EPISODES = args.max_episodes

    modes = RL_MODES if args.mode == "all" else [args.mode]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  IEEE 802.11ah RL Multi-Episode Training (convergence-based)")
    print(f"  Modes: {modes}")
    print(f"  N values: {args.n_values}")
    print(f"  Max episodes: {MAX_EPISODES}  |  Min episodes: {MIN_EPISODES}  "
          f"|  Convergence threshold: {CONV_THRESHOLD}")
    print(f"  Sim time per episode: {SIM_TIME_S:.0f}s")
    print(f"  Q-tables → {QTABLE_DIR}")
    print(sep)

    t_total_start = time.time()
    all_rows: List[Dict] = []
    for mode in modes:
        rows = train(mode, args.n_values)
        all_rows.extend(rows)
        append_training_log(rows)

    wall = time.time() - t_total_start
    print(f"\n  Total training time: {wall:.0f}s")
    print(f"  Q-tables: {QTABLE_DIR}/qtable_<mode>_N<n>.pkl")
    print(f"\n  Run evaluation with:")
    print(f"    python scripts/compare_rl_trained.py\n")


if __name__ == "__main__":
    main()
