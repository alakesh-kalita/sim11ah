from __future__ import annotations

import csv
import math
from typing import Any, Dict, List, Tuple

from sim11ah.mac.common import (
    MORSE_RAW_MIN_SLOT_DURATION_US,
    RawBeaconSpreading,
    RawConfig,
    RawPeriodic,
    RawSlotDefinition,
)
from sim11ah.mac.etaroa_traffic_estimator import EtaroaTrafficEstimator


class ClusterAdaptiveRawPolicy:
    """
    Cluster-Adaptive Dual RAW Allocation.

    Uses:
    - CSV-based cluster grouping
    - runtime queue + CSV fallback load estimation
    - EWMA-CUSUM demand prediction
    - priority-weighted cluster demand
    - Bianchi contention estimate
    - adaptive number of RAW slots per cluster
    - adaptive slot duration per cluster
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        self.mac_cfg = self.ctx.cfg["mac"]
        self.phy_cfg = self.ctx.cfg["phy"]
        self.net_cfg = self.ctx.cfg["net"]
        self.app_cfg = self.ctx.cfg["app"]

        self.cluster_csv_path = self.mac_cfg.get(
            "cluster_csv_path",
            "uav_cluster_data.csv",
        )

        self.fixed_num_slots = max(
            1,
            int(self.mac_cfg.get("raw_num_slots", getattr(self.ctx, "raw_num_slots", 1))),
        )

        self.min_slots_per_cluster = int(
            self.mac_cfg.get("cluster_adaptive_min_slots_per_cluster", 1)
        )

        self.max_slots_per_cluster = int(
            self.mac_cfg.get(
                "cluster_adaptive_max_slots_per_cluster",
                max(self.fixed_num_slots, self.fixed_num_slots * 2),
            )
        )

        self.slot_demand_exponent = float(
            self.mac_cfg.get("cluster_adaptive_slot_demand_exponent", 0.75)
        )

        base_slot_us = int(float(getattr(self.ctx, "raw_slot_duration", 0.007)) * 1e6)

        self.min_slot_us = int(
            self.mac_cfg.get(
                "adaptive_raw_min_slot_us",
                max(MORSE_RAW_MIN_SLOT_DURATION_US, max(7000, base_slot_us // 2)),
            )
        )

        self.max_slot_us = int(
            self.mac_cfg.get(
                "adaptive_raw_max_slot_us",
                max(self.min_slot_us, max(base_slot_us * 4, 20000)),
            )
        )

        self.initial_slot_us = int(
            self.mac_cfg.get("adaptive_raw_initial_slot_us", base_slot_us)
        )
        self.initial_slot_us = max(self.min_slot_us, min(self.max_slot_us, self.initial_slot_us))

        self.step_us = max(100, int(self.mac_cfg.get("adaptive_raw_step_us", 500)))
        self.smoothing_beta = float(self.mac_cfg.get("adaptive_raw_smoothing_beta", 0.6))
        self.hysteresis = float(self.mac_cfg.get("adaptive_raw_hysteresis", 1000.0))

        self.ewma_alpha = float(self.mac_cfg.get("adaptive_raw_ewma_alpha", 0.7))
        self.cusum_k = float(self.mac_cfg.get("adaptive_raw_cusum_k", 0.10))
        self.cusum_h = float(self.mac_cfg.get("adaptive_raw_cusum_h", 1.0))

        self.bianchi_eps = float(self.mac_cfg.get("adaptive_raw_bianchi_eps", 1e-4))
        self.bianchi_imax = int(self.mac_cfg.get("adaptive_raw_bianchi_imax", 50))

        self.tmax_s = float(self.mac_cfg.get("adaptive_raw_tmax_s", self.max_slot_us / 1e6))
        self.lth_slots = int(
            self.mac_cfg.get(
                "adaptive_raw_lth_slots",
                max(1, int(round((0.75 * self.tmax_s) / self._sigma_s()))),
            )
        )

        self.priority_weights = {
            "critical": float(self.mac_cfg.get("priority_weight_critical", 5.0)),
            "high": float(self.mac_cfg.get("priority_weight_high", 3.0)),
            "normal": float(self.mac_cfg.get("priority_weight_normal", 1.0)),
        }

        self.demand_gain = float(self.mac_cfg.get("cluster_adaptive_demand_gain", 2.5))
        self.size_gain = float(self.mac_cfg.get("cluster_adaptive_size_gain", 0.012))
        self.burst_gain = float(self.mac_cfg.get("cluster_adaptive_burst_gain", 1.25))

        # config.py documents raw_num_groups as "SAME for adaptive / cluster /
        # static" -- the other RAW policies always split the connected
        # population into this many parallel groups regardless of any
        # external clustering. cluster_adaptive previously ignored it
        # entirely, so whenever few CSV cluster_ids were populated at a
        # given N (e.g. all AIDs landing in a single cluster at low N) it
        # collapsed to far fewer contention-opportunities than every other
        # policy. Use it as a floor on total RAW groups, splitting the
        # largest populated cluster(s) as needed to reach it.
        self.target_min_groups = max(
            1, int(self.mac_cfg.get("raw_num_groups", getattr(self.ctx, "raw_num_groups", 4)))
        )
        # Diversity-flooring by COUNT alone still leaves groups unbalanced
        # (e.g. [70, 69, 43, 18] at N=200) since it stops as soon as
        # target_min_groups is reached, however lopsided the split -- while
        # free-partitioning policies (Adaptive/E-TAROA) always get
        # ~equal-sized groups via index-proportional splitting. Continue
        # splitting the largest group past the count floor whenever it
        # exceeds group_balance_factor x the population's fair share.
        # See raw_policy_cluster_csv.py's copy of this method for the
        # fixed-vs-growing-denominator convergence note.
        self.group_balance_factor = float(self.mac_cfg.get("cluster_group_balance_factor", 1.25))
        # The CSV's cluster boundaries are fixed AID ranges over the full
        # N=1200 population (12 clusters, each contiguous and sequentially
        # AID-ordered -- confirmed directly), so the number of NATURALLY
        # populated clusters keeps growing as N grows (1 at N=100, up to 10
        # at N=1000) -- unlike Adaptive/E-TAROA, which always use exactly
        # target_min_groups equal-sized groups regardless of N. More groups
        # sharing the SAME fixed per-beacon RAW budget means a smaller
        # window per group, which was measured to cost real delay at high N
        # (N=1000: +23% avg / +36% p95 vs. E-TAROA) despite similar PDR.
        # Cap the total group count via adjacent-cluster merging (see
        # merge step below) so it can't run away at high N; AID-adjacency
        # is required to keep every merged group's AID range contiguous.
        self.target_max_groups = max(
            self.target_min_groups,
            int(self.mac_cfg.get("cluster_group_max_count", 8)),
        )

        # Guaranteeing target_min_groups groups each at least
        # min_slots_per_cluster slots (above) can, at high N with many
        # populated clusters, sum to more RAW time than fits in one beacon
        # interval -- the schedule would silently overrun into the next
        # beacon. "adaptive" avoids this by capping its (fixed) group/slot
        # count against the beacon budget up front; cluster_adaptive's
        # group count is data-dependent, so it must check the budget after
        # sizing groups and scale every group's slot count down
        # proportionally (preserving relative demand) if it doesn't fit.
        # Uses its OWN key (not the shared adaptive_raw_budget_fraction
        # that Adaptive/Static/RL also read) so raising this can't silently
        # change any other policy's baseline -- verified: at N=1000 with
        # ~8 populated RAW groups, the group-count floor alone (4 slots x
        # 8 groups) already consumes essentially the entire 0.90 budget,
        # leaving zero room for demand-based (water-filling) allocation --
        # every group gets floor-locked to an identical slot count
        # regardless of real demand. Swept 0.90-0.97: 0.93 is the highest
        # value that stays at assoc_frac=1.0 across the full N range
        # (100-1000); 0.95+ reintroduces the beacon-budget overrun that an
        # earlier fix in this policy eliminated (association drops to
        # 0.93-0.94 at N=800-1000).
        self.budget_fraction = float(
            self.mac_cfg.get("cluster_adaptive_raw_budget_fraction", 0.93)
        )
        self.beacon_interval_us = float(self.mac_cfg.get("beacon_interval", 0.5)) * 1e6

        self._current_slot_us: int = self.initial_slot_us
        self._update_count: int = 0

        self._estimator = EtaroaTrafficEstimator(self.ctx, default_t_int_beacons=1.0)
        self._last_group_slots: Dict[int, int] = {}

    def init_configs(self) -> List["RawConfig"]:
        aids = self._effective_connected_aids([])
        max_aid = max(aids) if aids else max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))

        return [
            RawConfig(
                id=1,
                raw_type=self.ctx.raw_type,
                start_aid=1,
                end_aid=max_aid,
                start_time_us=self.ctx.raw_start_time_us,
                slot_definition=RawSlotDefinition(
                    num_slots=self.fixed_num_slots,
                    slot_duration_us=self._quantize_up(self._current_slot_us),
                    cross_slot_boundary=self.ctx.raw_cross_slot,
                ),
                beacon_spreading=RawBeaconSpreading(),
                periodic=RawPeriodic(),
                enabled=True,
            )
        ]

    def build_dynamic_configs(self, connected_aids: List[int]) -> List["RawConfig"]:
        aids = self._effective_connected_aids(connected_aids)
        if not aids:
            self._log("RAW_CLUSTER_ADAPT_NO_AIDS", {})
            return []

        rows = self._load_cluster_csv(aids)
        if not rows:
            self._log("RAW_CLUSTER_ADAPT_EMPTY_CSV", {"path": self.cluster_csv_path})
            return self.init_configs()

        clusters = self._group_by_cluster(rows)
        clusters = self._ensure_min_group_diversity(clusters, self.target_min_groups)

        sigma = self._sigma_s()
        lmax_slots = max(1, int(math.floor(self.tmax_s / sigma)))

        raw_lengths_slots: Dict[int, int] = {}
        group_debug: Dict[int, Dict[str, Any]] = {}

        for cluster_id, cluster_rows in clusters.items():
            Dgk, has_burst = self._compute_cluster_demand(cluster_rows)
            nact = sum(1 for r in cluster_rows if float(r["pred_load"]) > 0.0)

            if Dgk <= 0.0 or nact <= 0:
                raw_lengths_slots[cluster_id] = 0
                group_debug[cluster_id] = self._empty_debug(Dgk, nact)
                continue

            tau, p = self._solve_bianchi_fixed_point(nact)

            p_idle = (1.0 - tau) ** nact
            p_succ = nact * tau * ((1.0 - tau) ** (nact - 1))
            p_col = max(0.0, 1.0 - p_idle - p_succ)

            ts, tc = self._get_success_collision_times_s()
            l_s = ts / sigma
            l_c = tc / sigma
            l_bar = p_idle * 1.0 + p_succ * l_s + p_col * l_c

            raw_L = int(math.ceil(Dgk * l_bar / max(1e-9, p_succ)))
            raw_L = max(1, min(raw_L, lmax_slots))

            raw_lengths_slots[cluster_id] = raw_L

            group_debug[cluster_id] = {
                "Dgk": float(Dgk),
                "nact": int(nact),
                "has_burst": int(has_burst),
                "tau": float(tau),
                "p": float(p),
                "p_idle": float(p_idle),
                "p_succ": float(p_succ),
                "p_col": float(p_col),
                "l_bar": float(l_bar),
            }

        non_zero_lengths = [v for v in raw_lengths_slots.values() if v > 0]
        avg_Lg_slots = int(round(sum(non_zero_lengths) / len(non_zero_lengths))) if non_zero_lengths else 0
        avg_Lg_slots = min(avg_Lg_slots, lmax_slots)

        positive_demands = [
            float(group_debug[cid].get("Dgk", 0.0))
            for cid in group_debug
            if float(group_debug[cid].get("Dgk", 0.0)) > 0.0
        ]

        avg_demand = (
            sum(positive_demands) / len(positive_demands)
            if positive_demands else 1.0
        )
        avg_demand = max(avg_demand, 1e-9)

        eta_g = 1 if avg_Lg_slots > self.lth_slots else 0

        # --- Pass 1: size every group's (num_slots_k, slot_us) independently ---
        sized_groups: List[Dict[str, Any]] = []

        for cluster_id, cluster_rows in sorted(clusters.items()):
            member_aids = sorted(int(r["aid"]) for r in cluster_rows)
            if not member_aids:
                continue

            raw_L = int(raw_lengths_slots.get(cluster_id, 0))
            dbg = group_debug.get(cluster_id, {})
            Dgk = float(dbg.get("Dgk", 0.0))

            if raw_L <= 0 or Dgk <= 0.0:
                num_slots_k = self.min_slots_per_cluster
                slot_us = self._apply_safety_lower_bound(self.min_slot_us)
                final_Lgk_slots = 0
            else:
                final_Lgk_slots = raw_L

                demand_ratio = Dgk / avg_demand
                num_slots_k = int(
                    math.ceil(
                        self.fixed_num_slots * (demand_ratio ** self.slot_demand_exponent)
                    )
                )

                num_slots_k = max(self.min_slots_per_cluster, num_slots_k)
                num_slots_k = min(self.max_slots_per_cluster, num_slots_k)

                total_airtime_us = int(math.ceil(final_Lgk_slots * sigma * 1e6))
                total_airtime_us = max(total_airtime_us, self.min_slot_us)

                slot_us = int(math.ceil(total_airtime_us / max(1, num_slots_k)))

                slot_us = self._apply_safety_lower_bound(slot_us)
                slot_us = min(slot_us, self.max_slot_us)
                slot_us = self._quantize_up(slot_us)

            prev_slot = self._last_group_slots.get(cluster_id, self._current_slot_us)

            if abs(slot_us - prev_slot) > self.hysteresis:
                slot_us = int(
                    round(self.smoothing_beta * slot_us + (1.0 - self.smoothing_beta) * prev_slot)
                )
                slot_us = self._quantize_up(slot_us)
            else:
                slot_us = prev_slot

            slot_us = self._apply_safety_lower_bound(slot_us)
            slot_us = min(slot_us, self.max_slot_us)
            slot_us = self._quantize_up(slot_us)

            sized_groups.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_rows": cluster_rows,
                    "member_aids": member_aids,
                    "num_slots_k": num_slots_k,
                    "slot_us": slot_us,
                    "raw_L": raw_L,
                    "final_Lgk_slots": final_Lgk_slots,
                    "Dgk": Dgk,
                    "dbg": dbg,
                }
            )

        # --- Pass 1.5: beacon-interval budget clamp -----------------------
        # Pass 1 sizes every group independently against its own demand,
        # with no cross-group awareness of the shared beacon budget. At
        # high cluster counts (e.g. 10 populated clusters at N=1000) the
        # summed schedule can exceed the nominal beacon interval by ~1.27x.
        #
        # A naive uniform proportional cut across ALL groups' slot counts
        # was tried before and reverted -- it lowered average PDR by taking
        # real airtime away from clusters that were already fully served,
        # in exchange for merely shrinking (not eliminating) the shortfall
        # for tail clusters. But an overrunning schedule has a worse cost
        # than lower PDR alone: the AP's beacon fires on its own fixed
        # period regardless of whether the previous schedule finished
        # (facade.py's _next_beacon_target), so an overrun means the
        # schedule gets thrown away and rebuilt from scratch -- in
        # ascending cluster-id/AID order -- before it ever reaches the
        # tail groups. Those tail-AID clusters get zero RAW window, every
        # single beacon, for the entire run, and the boundary where the
        # old schedule's dangling tail-group timers overlap the new
        # schedule's front groups spikes channel contention right when the
        # AP most needs a clear channel to answer AUTH/ASSOC requests from
        # still-unassociated STAs -- this is what was suppressing
        # association completeness at N>=800 (see association audit).
        #
        # An earlier version of this clamp trimmed one slot at a time from
        # whichever group currently held the most (down to its floor).
        # That guarantees every group a window, but iterated to
        # convergence it degenerates into full equalization: once the
        # required cut is large enough that groups start tying for
        # "biggest", the loop keeps shaving the (arbitrarily tie-broken)
        # front of that tied set, walking every group down to the SAME
        # count regardless of how different their original demand was --
        # confirmed directly: pre-clamp counts spanning 4-16 (correctly
        # demand-proportional) all converged to a uniform 4 post-clamp at
        # N=1000. That silently defeats the entire point of demand-based
        # allocation exactly when it matters most (tight budgets, high N).
        #
        # This version instead scales every above-floor group's slot
        # count down by the SAME ratio, so a group with 2x another's
        # demand keeps roughly 2x its post-clamp slots too -- proportional
        # water-filling, not flattening. Groups already at the floor are
        # excluded from a scaling pass and their airtime is subtracted
        # from the budget first; repeated because floored groups can free
        # up slack that changes the ratio for the rest.
        total_us = sum(g["num_slots_k"] * g["slot_us"] for g in sized_groups)
        budget_us = self.budget_fraction * self.beacon_interval_us

        if total_us > budget_us and sized_groups:
            # The configured floor (cluster_adaptive_min_slots_per_cluster,
            # default 4) is itself just a floor -- at high enough cluster
            # counts (e.g. 9 populated clusters at N=1000), giving EVERY
            # group that many slots already exceeds the budget on its own
            # (9 * 4 slots * ~14ms slot_us > 450ms budget). When that
            # happens the water-filling below has zero room to
            # differentiate: every group gets floored in the first pass
            # regardless of demand, which is functionally identical to the
            # old trim-to-uniform bug even though the algorithm itself is
            # now correct. Shrink the EFFECTIVE floor (never below 1) only
            # when the configured one genuinely doesn't fit, so demand
            # differentiation survives as long as any slack exists at all;
            # low-N/few-cluster cases where 4 comfortably fits are
            # unaffected.
            total_slot_us = sum(g["slot_us"] for g in sized_groups)
            effective_floor = max(
                1, min(self.min_slots_per_cluster, int(budget_us // max(1, total_slot_us)))
            )

            for g in sized_groups:
                g["_orig_k"] = g["num_slots_k"]
                g["_floored"] = False

            for _ in range(len(sized_groups) + 1):
                active = [g for g in sized_groups if not g["_floored"]]
                if not active:
                    break
                floored_us = sum(
                    effective_floor * g["slot_us"] for g in sized_groups if g["_floored"]
                )
                active_orig_us = sum(g["_orig_k"] * g["slot_us"] for g in active)
                active_floor_us = sum(effective_floor * g["slot_us"] for g in active)
                slack_for_active = budget_us - floored_us

                denom = active_orig_us - active_floor_us
                scale = 0.0 if denom <= 0 else max(0.0, (slack_for_active - active_floor_us) / denom)

                newly_floored = False
                for g in active:
                    target = effective_floor + scale * (g["_orig_k"] - effective_floor)
                    new_k = max(effective_floor, int(round(target)))
                    if new_k <= effective_floor:
                        g["num_slots_k"] = effective_floor
                        g["_floored"] = True
                        newly_floored = True
                    else:
                        g["num_slots_k"] = new_k

                if not newly_floored:
                    break

            for g in sized_groups:
                del g["_orig_k"]
                del g["_floored"]

        # --- Pass 2 (was 3): emit RawConfig objects and logs ---
        configs: List[RawConfig] = []
        start_offset_us = int(self.ctx.raw_start_time_us)
        slot_values_us: List[int] = []
        num_slots_values: List[int] = []

        for cfg_id, g in enumerate(sized_groups, start=1):
            cluster_id = g["cluster_id"]
            cluster_rows = g["cluster_rows"]
            member_aids = g["member_aids"]
            num_slots_k = g["num_slots_k"]
            slot_us = g["slot_us"]
            raw_L = g["raw_L"]
            final_Lgk_slots = g["final_Lgk_slots"]
            Dgk = g["Dgk"]
            dbg = g["dbg"]

            self._last_group_slots[cluster_id] = slot_us
            slot_values_us.append(slot_us)
            num_slots_values.append(num_slots_k)

            cfg = RawConfig(
                id=cfg_id,
                raw_type=self.ctx.raw_type,
                start_aid=min(member_aids),
                end_aid=max(member_aids),
                start_time_us=start_offset_us,
                slot_definition=RawSlotDefinition(
                    num_slots=num_slots_k,
                    slot_duration_us=slot_us,
                    cross_slot_boundary=self.ctx.raw_cross_slot,
                ),
                beacon_spreading=RawBeaconSpreading(),
                periodic=RawPeriodic(),
                enabled=True,
            )
            configs.append(cfg)

            start_offset_us += num_slots_k * slot_us

            self._log(
                "RAW_CLUSTER_ADAPT_GROUP",
                {
                    "cluster_id": int(cluster_id),
                    "source_cluster_id": int(cluster_rows[0]["cluster_id"]) if cluster_rows else int(cluster_id),
                    "sub_group": int(cluster_id) % 1000,
                    "start_aid": min(member_aids),
                    "end_aid": max(member_aids),
                    "n_members": len(member_aids),
                    "Dgk": float(Dgk),
                    "avg_demand": float(avg_demand),
                    "demand_ratio": float(Dgk / avg_demand if avg_demand > 0 else 0.0),
                    "nact": int(dbg.get("nact", 0)),
                    "raw_Lgk_slots": int(raw_L),
                    "final_Lgk_slots": int(final_Lgk_slots),
                    "avg_Lg_slots": int(avg_Lg_slots),
                    "lmax_slots": int(lmax_slots),
                    "eta_g": int(eta_g),
                    "slot_duration_us": int(slot_us),
                    "num_slots": int(num_slots_k),
                    "base_num_slots": int(self.fixed_num_slots),
                    "tau": float(dbg.get("tau", 0.0)),
                    "p": float(dbg.get("p", 0.0)),
                    "p_idle": float(dbg.get("p_idle", 0.0)),
                    "p_succ": float(dbg.get("p_succ", 0.0)),
                    "p_col": float(dbg.get("p_col", 0.0)),
                    "l_bar": float(dbg.get("l_bar", 0.0)),
                    "update_count": int(self._update_count),
                },
            )

        if slot_values_us:
            self._current_slot_us = self._quantize_up(
                int(round(sum(slot_values_us) / len(slot_values_us)))
            )

        self._update_count += 1

        self._log(
            "RAW_CLUSTER_ADAPT_BUILD",
            {
                "groups_built": len(configs),
                "stas_total": len(aids),
                "clusters": len(clusters),
                "avg_slot_duration_us": int(self._current_slot_us),
                "group_slot_us_values": [int(x) for x in slot_values_us],
                "group_num_slots_values": [int(x) for x in num_slots_values],
                "avg_Lg_slots": int(avg_Lg_slots),
                "Lmax_slots": int(lmax_slots),
                "eta_g": int(eta_g),
                "total_duration_us": start_offset_us - int(self.ctx.raw_start_time_us),
                "csv_path": self.cluster_csv_path,
            },
        )

        return configs if configs else self.init_configs()

    def get_policy_snapshot(self) -> Dict[str, Any]:
        return {
            "policy": "cluster_adaptive_dual_raw_allocation",
            "cluster_csv_path": self.cluster_csv_path,
            "fixed_num_slots": self.fixed_num_slots,
            "min_slots_per_cluster": self.min_slots_per_cluster,
            "max_slots_per_cluster": self.max_slots_per_cluster,
            "slot_demand_exponent": self.slot_demand_exponent,
            "slot_duration_us": self._current_slot_us,
            "step_us": self.step_us,
            "update_count": self._update_count,
            "ewma_alpha": self.ewma_alpha,
            "cusum_k": self.cusum_k,
            "cusum_h": self.cusum_h,
            "tmax_s": self.tmax_s,
            "lth_slots": self.lth_slots,
            "demand_gain": self.demand_gain,
            "size_gain": self.size_gain,
            "burst_gain": self.burst_gain,
        }

    def _load_cluster_csv(self, connected_aids: List[int]) -> List[Dict[str, Any]]:
        connected = set(int(a) for a in connected_aids)
        rows: List[Dict[str, Any]] = []

        rates = self._estimator.update(sorted(connected))

        with open(self.cluster_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                aid = int(row["aid"])
                if aid not in connected:
                    continue

                priority = str(row.get("priority", "normal")).strip().lower()
                if priority not in self.priority_weights:
                    priority = "normal"

                observed = self._estimate_sta_observed_load(aid, row)
                pred_load = float(rates.get(aid, 0.0))

                rows.append(
                    {
                        "aid": aid,
                        "cluster_id": int(row["cluster_id"]),
                        "priority": priority,
                        "tx_rate_bps": float(row.get("tx_rate_bps", 150000.0)),
                        "queue_len": float(observed),
                        "pred_load": float(pred_load),
                        "packet_size_bytes": int(
                            row.get("packet_size_bytes", self.app_cfg.get("packet_size_bytes", 128))
                        ),
                    }
                )

        return rows

    def _estimate_sta_observed_load(self, aid: int, row: Dict[str, Any] | None = None) -> float:
        try:
            sim_nodes = getattr(self.ctx.sim, "nodes", {})
            node = sim_nodes.get(aid)

            if node is not None:
                mac = getattr(node, "mac", None)
                if mac is not None:
                    for attr in ("tx_queue", "_tx_queue", "txq", "_txq"):
                        q = getattr(mac, attr, None)
                        if q is not None:
                            try:
                                return float(len(q))
                            except Exception:
                                pass

                app = getattr(node, "app", None)
                if app is not None:
                    for attr in ("pending", "_pending", "queue", "_queue"):
                        q = getattr(app, attr, None)
                        if q is not None:
                            try:
                                return float(len(q))
                            except Exception:
                                pass
        except Exception:
            pass

        if row is not None:
            return float(row.get("queue_len", 0.0))

        return 0.25

    def _group_by_cluster(self, rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            clusters.setdefault(int(r["cluster_id"]), []).append(r)
        return clusters

    def _ensure_min_group_diversity(
        self,
        clusters: Dict[int, List[Dict[str, Any]]],
        target_min_groups: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Guarantee at least target_min_groups total RAW groups AND that no
        group is left far larger than the population's fair share, by
        repeatedly splitting the largest populated cluster into
        contiguous-AID halves. Each original cluster_id is remapped to
        cluster_id * 1000 [+ sub-index] so downstream keys stay plain ints.

        Splitting continues past the group-COUNT floor whenever the
        largest remaining group exceeds group_balance_factor x the fair
        share (total members / target_min_groups, FIXED at the original
        target -- not recomputed against the growing group count, which
        would create a self-chasing spiral that never converges and
        fragments into dozens of tiny groups at high N; see
        raw_policy_cluster_csv.py's identical method for the empirical
        confirmation of this failure mode).

        Before any splitting, merge down to at most target_max_groups if
        the CSV's naturally-populated cluster count already exceeds it
        (true from N~=800 upward) -- always merging the two AID-ADJACENT
        groups with the smallest combined size, which is what keeps every
        merged group's AID range contiguous (required by RawConfig's
        start_aid/end_aid; the 12 real clusters are already AID-ordered
        with no gaps, confirmed directly, so adjacency in AID order is
        well-defined and safe to merge across).
        """
        groups: Dict[int, List[Dict[str, Any]]] = {
            cid * 1000: sorted(rows, key=lambda r: int(r["aid"]))
            for cid, rows in clusters.items()
        }
        next_suffix: Dict[int, int] = {base: 1 for base in groups}

        while len(groups) > self.target_max_groups:
            ordered = sorted(groups.items(), key=lambda kv: kv[1][0]["aid"])
            costs = (
                (len(ordered[i][1]) + len(ordered[i + 1][1]), i)
                for i in range(len(ordered) - 1)
            )
            _, i = min(costs)
            (ka, ra), (kb, rb) = ordered[i], ordered[i + 1]
            del groups[ka]
            del groups[kb]
            groups[ka] = sorted(ra + rb, key=lambda r: int(r["aid"]))
            next_suffix.setdefault(ka, 1)

        total_members = sum(len(rows) for rows in groups.values())
        fair_share = total_members / max(1, target_min_groups)

        def _needs_split() -> bool:
            if len(groups) < target_min_groups:
                return True
            if len(groups) >= self.target_max_groups:
                return False
            largest = max(len(rows) for rows in groups.values())
            return largest > self.group_balance_factor * fair_share

        while _needs_split():
            big_key = max(groups, key=lambda k: len(groups[k]))
            rows = groups[big_key]
            if len(rows) < 2:
                break

            mid = len(rows) // 2
            base = (big_key // 1000) * 1000
            suffix = next_suffix.get(base, 1)

            groups[big_key] = rows[:mid]
            groups[base + suffix] = rows[mid:]
            next_suffix[base] = suffix + 1

        return groups

    def _compute_cluster_demand(self, rows: List[Dict[str, Any]]) -> Tuple[float, bool]:
        # Dgk is a weighted PACKET COUNT (not airtime- or slot-normalized):
        # raw_L below multiplies Dgk by l_bar/p_succ, which already carries
        # the "sigma-slots needed per successful packet" contention-overhead
        # dimension. Folding an extra airtime_s/sigma factor into Dgk here
        # (as earlier versions did) double-counted that slot normalization,
        # inflating Dgk into the millions and saturating raw_L at lmax_slots
        # regardless of actual demand.
        demand = 0.0
        total_pred = 0.0
        has_burst = False

        for r in rows:
            priority = str(r["priority"])
            weight = self.priority_weights.get(priority, 1.0)

            pred = max(0.0, float(r["pred_load"]))

            demand += weight * pred
            total_pred += pred

            if pred > self.cusum_h:
                has_burst = True

        demand *= self.demand_gain
        demand *= 1.0 + self.size_gain * max(0, len(rows) - 1)

        if has_burst:
            demand *= self.burst_gain

        demand *= 1.0 + 0.03 * total_pred

        return max(0.0, demand), has_burst

    def _solve_bianchi_fixed_point(self, n_active: int) -> Tuple[float, float]:
        if n_active <= 1:
            W0 = float(self.mac_cfg.get("cw_min", 15)) + 1.0
            tau = min(0.5, 2.0 / (W0 + 1.0))
            return tau, 0.0

        cw_min = float(self.mac_cfg.get("cw_min", 15))
        cw_max = float(self.mac_cfg.get("cw_max", 1023))
        W0 = cw_min + 1.0

        try:
            m = max(0, int(round(math.log2((cw_max + 1.0) / W0))))
        except Exception:
            m = 6

        tau = min(0.5, 2.0 / (W0 + 1.0))

        for _ in range(max(1, self.bianchi_imax)):
            p = 1.0 - (1.0 - tau) ** (n_active - 1)

            denom = (1.0 - 2.0 * p) * (W0 + 1.0) + p * W0 * (1.0 - (2.0 * p) ** m)
            if abs(denom) < 1e-12:
                break

            tau_new = (2.0 * (1.0 - 2.0 * p)) / denom
            tau_new = min(1.0, max(1e-9, tau_new))

            if abs(tau_new - tau) <= self.bianchi_eps:
                tau = tau_new
                break

            tau = tau_new

        p = 1.0 - (1.0 - tau) ** (n_active - 1)
        return tau, p

    def _sigma_s(self) -> float:
        return float(self.mac_cfg.get("slot_time", 52e-6))

    def _get_success_collision_times_s(self) -> Tuple[float, float]:
        payload_bytes = int(self.app_cfg.get("packet_size_bytes", 128))
        data_mac_oh = int(self.mac_cfg.get("data_mac_overhead_bytes", 36))
        net_oh = int(self.net_cfg.get("net_header_bytes", 16))
        ack_size = int(self.mac_cfg.get("ack_size_bytes", 14))

        preamble = float(self.phy_cfg.get("preamble_time", 320e-6))
        header = float(self.phy_cfg.get("header_time", 80e-6))
        sifs = float(self.mac_cfg.get("sifs", 160e-6))
        difs = float(self.mac_cfg.get("difs", 264e-6))
        ack_timeout = float(self.mac_cfg.get("ack_timeout", 1e-3))

        mode_table = self.phy_cfg.get("mode_table", {}) or {}
        data_mode = str(self.phy_cfg.get("default_mode", "MCS0"))
        ctrl_mode = str(self.phy_cfg.get("control_mode", "MCS0"))

        data_rate_bps = float(mode_table.get(data_mode, 150000.0))
        ctrl_rate_bps = float(mode_table.get(ctrl_mode, 150000.0))

        data_bits = 8.0 * (payload_bytes + data_mac_oh + net_oh)
        ack_bits = 8.0 * ack_size

        t_data = preamble + header + (data_bits / max(1.0, data_rate_bps))
        t_ack = preamble + header + (ack_bits / max(1.0, ctrl_rate_bps))

        ts = t_data + sifs + t_ack + difs
        tc = t_data + ack_timeout + difs
        return ts, tc

    def _apply_safety_lower_bound(self, slot_us: int) -> int:
        ts, _ = self._get_success_collision_times_s()
        raw_guard = float(self.mac_cfg.get("raw_guard", 100e-6))
        ack_guard = float(self.mac_cfg.get("ack_guard", 100e-6))

        cw_min = float(self.mac_cfg.get("cw_min", 15))
        sigma = self._sigma_s()
        avg_backoff_s = (cw_min / 2.0) * sigma

        safe_min_s = ts + raw_guard + ack_guard + avg_backoff_s
        safe_min_us = int(math.ceil(safe_min_s * 1e6))

        safe_min_us = max(
            safe_min_us,
            7000,
            self.min_slot_us,
            MORSE_RAW_MIN_SLOT_DURATION_US,
        )

        return max(slot_us, safe_min_us)

    def _quantize_up(self, slot_us: int) -> int:
        step = max(1, self.step_us)
        q = int(math.ceil(slot_us / step)) * step
        # Snap up onto the RAW slot-duration cslot grid (500 + 120*cslot us,
        # per IEEE 802.11ah §9.4.2.200) so that raw.py's us_to_cslot()
        # round-trip (which floors onto this grid) cannot return a value
        # below q.
        q = -(-(q - MORSE_RAW_MIN_SLOT_DURATION_US) // 120) * 120 + MORSE_RAW_MIN_SLOT_DURATION_US
        q = max(self.min_slot_us, min(self.max_slot_us, q))
        q = max(MORSE_RAW_MIN_SLOT_DURATION_US, q)
        return q

    def _effective_connected_aids(self, connected_aids: List[int]) -> List[int]:
        aids = sorted(int(a) for a in connected_aids if int(a) > 0)
        if aids:
            return aids

        try:
            assoc = getattr(self.ctx, "_associated_stas", None)
            if isinstance(assoc, dict) and assoc:
                vals = sorted(int(v) for v in assoc.values() if int(v) > 0)
                if vals:
                    return vals
        except Exception:
            pass

        try:
            sim_nodes = getattr(self.ctx.sim, "nodes", {})
            vals = sorted(int(nid) for nid in sim_nodes.keys() if int(nid) > 0)
            if vals:
                return vals
        except Exception:
            pass

        return []

    def _empty_debug(self, Dgk: float, nact: int) -> Dict[str, Any]:
        return {
            "Dgk": float(Dgk),
            "nact": int(nact),
            "has_burst": 0,
            "tau": 0.0,
            "p": 0.0,
            "p_idle": 1.0,
            "p_succ": 0.0,
            "p_col": 0.0,
            "l_bar": 0.0,
        }