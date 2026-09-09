"""
Prediction-error analysis for the CUSUM-EWMA per-STA demand estimator
(AdaptiveRawPolicy._predict_sta_demands).

For each beacon at time t, the estimator emits a predicted per-STA demand
d_i(t). We compare this against the ground-truth number of packets STA i
actually generates during [t, t_next), where t_next is the following
beacon's prediction timestamp. This is a genuine one-step-ahead forecast
evaluation: d_i(t) is computed from observations up to t and is the value
actually used to size the RAW slot serving [t, t_next).

Ground truth is captured by patching ApplicationLayer._generate_one and
counting only ticks that actually enqueue a packet (post/pre packets_generated
delta), not queue snapshots -- this is independent of the estimator's own
(proxy) observation signal.

Metrics (pooled over STAs and beacons, first WARMUP_S excluded to let the
EWMA/CUSUM state settle):
  MAE  = mean(|d_i(t) - actual|)
  RMSE = sqrt(mean((d_i(t) - actual)^2))
  MAPE = mean(|d_i(t) - actual| / actual) * 100, computed only over
         samples with actual > 0 (standard convention -- MAPE is undefined
         at actual == 0, which dominates at this sparse per-beacon
         granularity since packets arrive roughly every 10 beacons).

Usage:
    python scripts/prediction_error_analysis.py
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder
from sim11ah.app import ApplicationLayer, PeriodicTraffic, PoissonTraffic
from sim11ah.mac.raw_policy_adaptive import AdaptiveRawPolicy

N_STAS       = 200
SIM_TIME_S   = 120.0
SEEDS        = [42]
WARMUP_S     = 5.0        # exclude first 10 beacons (EWMA/CUSUM settling)
PKT_SIZE     = 128
LINK_RATE    = 150_000
PROP_DELAY   = 300e-6
TRAFFIC_INT  = 5.0        # deterministic Tp; Poisson lambda_p = 1/Tp

RAW_NUM_GROUPS = 8
RAW_NUM_SLOTS  = 8
RAW_SLOT_DUR_S = 0.007


def _instrument():
    """Monkeypatch generation + predictor to log (time, node) events."""
    gen_events: Dict[int, List[float]] = defaultdict(list)
    pred_events: List[Tuple[float, Dict[int, float]]] = []

    orig_generate_one = ApplicationLayer._generate_one

    def patched_generate_one(self):
        before = self.sim.stats.packets_generated
        orig_generate_one(self)
        after = self.sim.stats.packets_generated
        if after > before:
            gen_events[self.node.node_id].append(float(self.sim.engine.now))

    orig_predict = AdaptiveRawPolicy._predict_sta_demands

    def patched_predict(self, aids):
        result = orig_predict(self, aids)
        now = float(self.ctx.sim.engine.now)
        pred_events.append((now, dict(result)))
        return result

    ApplicationLayer._generate_one = patched_generate_one
    AdaptiveRawPolicy._predict_sta_demands = patched_predict

    def restore():
        ApplicationLayer._generate_one = orig_generate_one
        AdaptiveRawPolicy._predict_sta_demands = orig_predict

    return gen_events, pred_events, restore


def run_once(seed: int, poisson: bool) -> Tuple[List[float], List[float]]:
    gen_events, pred_events, restore = _instrument()
    try:
        cfg = default_config(raw_enable=True, traffic_mode="periodic")
        cfg["app"]["packet_size_bytes"] = PKT_SIZE
        cfg["mac"]["raw_policy"]        = "adaptive"
        cfg["mac"]["raw_num_groups"]    = RAW_NUM_GROUPS
        cfg["mac"]["raw_num_slots"]     = RAW_NUM_SLOTS
        cfg["mac"]["raw_slot_duration"] = RAW_SLOT_DUR_S

        sim = Simulator(config=cfg, seed=seed)
        StarBuilder.build(
            sim, num_stas=N_STAS,
            link_cfg={"rate_bps": LINK_RATE, "prop_delay": PROP_DELAY, "per": 0.0},
        )
        for nid, node in sim.nodes.items():
            if nid == 0:
                continue
            traffic = PoissonTraffic(1.0 / TRAFFIC_INT) if poisson else PeriodicTraffic(TRAFFIC_INT)
            node.app.set_traffic_model(traffic)

        sim.run_and_finalize(SIM_TIME_S)
    finally:
        restore()

    # Pair each prediction d_i(t_k) with ground-truth arrivals in [t_k, t_{k+1})
    pred_events.sort(key=lambda e: e[0])
    preds: List[float] = []
    actuals: List[float] = []

    for k in range(len(pred_events) - 1):
        t0, d_map = pred_events[k]
        t1, _ = pred_events[k + 1]
        if t0 < WARMUP_S:
            continue
        for aid, d_i in d_map.items():
            times = gen_events.get(aid, [])
            actual = sum(1 for t in times if t0 <= t < t1)
            preds.append(float(d_i))
            actuals.append(float(actual))

    return preds, actuals


def compute_metrics(preds: List[float], actuals: List[float]) -> Dict[str, float]:
    n = len(preds)
    errs = [p - a for p, a in zip(preds, actuals)]
    mae = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)

    nz = [(p, a) for p, a in zip(preds, actuals) if a > 0]
    if nz:
        mape = 100.0 * sum(abs(p - a) / a for p, a in nz) / len(nz)
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape, "n_samples": n, "n_nonzero": len(nz)}


def main():
    for label, poisson in [("Deterministic, Tp=5s", False), ("Poisson, lambda_p=0.2 pkt/s/STA", True)]:
        seed_metrics = []
        for seed in SEEDS:
            preds, actuals = run_once(seed, poisson)
            m = compute_metrics(preds, actuals)
            seed_metrics.append(m)
            print(f"  [{label}] seed={seed}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                  f"MAPE={m['mape']:.2f}%  (n={m['n_samples']}, nonzero={m['n_nonzero']})")

        mae  = sum(m["mae"]  for m in seed_metrics) / len(seed_metrics)
        rmse = sum(m["rmse"] for m in seed_metrics) / len(seed_metrics)
        mapes = [m["mape"] for m in seed_metrics if not math.isnan(m["mape"])]
        mape = sum(mapes) / len(mapes) if mapes else float("nan")

        print(f"\n=== {label} (3-seed mean) ===")
        print(f"  MAE  = {mae:.4f} packets/beacon")
        print(f"  RMSE = {rmse:.4f} packets/beacon")
        print(f"  MAPE = {mape:.2f}%\n")


if __name__ == "__main__":
    main()
