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


class ClusterCsvAdaptiveRawPolicy:
    """
    Cluster-aware adaptive RAW policy.

    Uses:
    - CSV-defined cluster_id and priority
    - runtime queue load from simulator
    - per-cluster Bianchi contention estimate
    - priority-weighted airtime demand
    - adaptive slot_duration_us per cluster

    CSV columns:
        aid, cluster_id, priority, tx_rate_bps, queue_len, packet_size_bytes
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        self.mac_cfg = self.ctx.cfg["mac"]
        self.phy_cfg = self.ctx.cfg["phy"]
        self.net_cfg = self.ctx.cfg["net"]
        self.app_cfg = self.ctx.cfg["app"]

        self.cluster_csv_path = self.mac_cfg.get("cluster_csv_path", "uav_cluster_data.csv")

        self.fixed_num_slots = max(
            1,
            int(self.mac_cfg.get("raw_num_slots", getattr(self.ctx, "raw_num_slots", 1))),
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

        self.step_us = max(100, int(self.mac_cfg.get("adaptive_raw_step_us", 1000)))
        self.smoothing_beta = float(self.mac_cfg.get("adaptive_raw_smoothing_beta", 0.65))
        self.hysteresis = float(self.mac_cfg.get("adaptive_raw_hysteresis", 0.0))

        self.ewma_alpha = float(self.mac_cfg.get("adaptive_raw_ewma_alpha", 0.75))
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

        self.demand_gain = float(self.mac_cfg.get("cluster_csv_demand_gain", 3.0))
        self.cluster_size_gain = float(self.mac_cfg.get("cluster_csv_size_gain", 0.015))
        self.burst_gain = float(self.mac_cfg.get("cluster_csv_burst_gain", 1.35))
        self.use_runtime_queue = bool(self.mac_cfg.get("cluster_csv_use_runtime_queue", True))

        # Every populated CSV cluster becomes its own RAW group with the
        # SAME fixed_num_slots (unlike cluster_adaptive, whose slot count
        # already scales down per group). At high N, many populated
        # clusters (e.g. 10 at N=1000) x a constant slot count each summed
        # zero cross-group budget awareness, so the schedule can overrun
        # the beacon interval well past what cluster_adaptive did before
        # its fix -- causing the same tail-cluster starvation and
        # association-completeness collapse (see raw_policy_cluster_adaptive.py's
        # Pass 1.5 for the full mechanism). Same clamp, applied here too.
        self.min_slots_per_cluster = int(
            self.mac_cfg.get("cluster_csv_min_slots_per_cluster", 1)
        )
        # Uses its OWN key (not the shared adaptive_raw_budget_fraction
        # that Adaptive/Static/RL also read) so raising this can't silently
        # change any other policy's baseline. raw_policy_cluster_adaptive.py
        # raised this to 0.93 (floor-locked groups at high N otherwise
        # leave zero room for demand-based allocation) and verified 1.0
        # association across all 3 seeds at every N. Tried the same here,
        # but 0.91/0.92/0.93 all produced byte-identical output (a step
        # function, not a smooth trade-off) that drops N=1000/seed=99 to
        # assoc_frac=0.99 -- 10 stations short of full association. Given
        # this project's zero-tolerance precedent on association
        # completeness (the original reason cluster policies needed
        # fixing at all), kept at the fully-safe 0.90 default rather than
        # trade 1% of one seed's association for a delay improvement.
        self.budget_fraction = float(
            self.mac_cfg.get("cluster_csv_raw_budget_fraction", 0.90)
        )
        self.beacon_interval_us = float(self.mac_cfg.get("beacon_interval", 0.5)) * 1e6

        # config.py documents raw_num_groups as "SAME for adaptive / cluster
        # / static" -- every other RAW policy always splits the connected
        # population into at least this many parallel groups regardless of
        # any external clustering. This policy never enforced that floor:
        # the CSV's cluster boundaries are fixed for its full N=1200
        # generation population, so a low-N run that only connects a
        # contiguous AID prefix (e.g. 1..100) can land entirely inside a
        # single populated cluster -- confirmed directly (N=100 -> exactly
        # 1 populated cluster, N=200 -> 3), collapsing parallelism to 1
        # RAW group with only fixed_num_slots contenders sharing it,
        # instead of the 4-way parallelism every other policy gets at the
        # same N. This was the actual cause of Cluster-based's PDR
        # collapsing to ~0.45-0.48 at N=100-200 while every adaptive-family
        # policy sits at 0.88-0.99 -- not a fundamental limitation of the
        # CSV-driven approach, just a missing floor that
        # raw_policy_cluster_adaptive.py already has (_ensure_min_group_diversity).
        # Same fix, applied here too: split the largest populated cluster(s)
        # by contiguous AID range until at least raw_num_groups groups exist.
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
        self.group_balance_factor = float(self.mac_cfg.get("cluster_group_balance_factor", 1.25))
        # The CSV's cluster boundaries are fixed AID ranges over the full
        # N=1200 population (12 clusters, each contiguous and sequentially
        # AID-ordered), so the naturally-populated cluster count keeps
        # growing with N (up to 10 at N=1000) -- unlike Adaptive/E-TAROA,
        # which always use exactly target_min_groups equal-sized groups.
        # More groups sharing the same fixed per-beacon RAW budget means a
        # smaller window per group, measured to cost real delay at high N.
        # Cap the total group count via adjacent-cluster merging (see
        # _ensure_min_group_diversity) so it can't run away.
        self.target_max_groups = max(
            self.target_min_groups,
            int(self.mac_cfg.get("cluster_group_max_count", 8)),
        )

        self._current_slot_us: int = self.initial_slot_us
        self._update_count: int = 0
        self._estimator = EtaroaTrafficEstimator(self.ctx, default_t_int_beacons=1.0)
        self._last_group_slots: Dict[int, int] = {}

        # Parse the CSV once at init; filter by connected_aids at call time.
        self._csv_rows_all: List[Dict[str, Any]] = self._parse_cluster_csv()

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
            self._log("RAW_CLUSTER_NO_AIDS", {})
            return []

        sta_rows = self._load_cluster_csv(aids)
        if not sta_rows:
            self._log("RAW_CLUSTER_EMPTY_CSV", {"path": self.cluster_csv_path})
            return self.init_configs()

        clusters = self._group_by_cluster(sta_rows)
        clusters = self._ensure_min_group_diversity(clusters, self.target_min_groups)

        sigma = self._sigma_s()
        lmax_slots = max(1, int(math.floor(self.tmax_s / sigma)))

        raw_lengths_slots: Dict[int, int] = {}
        group_debug: Dict[int, Dict[str, Any]] = {}

        for cluster_id, rows in clusters.items():
            Dgk = self._compute_cluster_demand(rows)
            nact = sum(1 for r in rows if float(r["pred_load"]) > 0.0)

            if Dgk <= 0.0 or nact <= 0:
                raw_lengths_slots[cluster_id] = 0
                group_debug[cluster_id] = self._empty_debug(Dgk, nact)
                continue

            tau, p = self._solve_bianchi_fixed_point(nact)

            p_idle = (1.0 - tau) ** nact
            p_succ = nact * tau * ((1.0 - tau) ** (nact - 1))
            p_col = max(0.0, 1.0 - p_idle - p_succ)

            ts, tc = self._get_success_collision_times_s(rows)
            l_s = ts / sigma
            l_c = tc / sigma
            l_bar = p_idle * 1.0 + p_succ * l_s + p_col * l_c

            p_succ_eff = max(1e-9, p_succ)

            Lgk = int(math.ceil(Dgk * l_bar / p_succ_eff))
            Lgk = max(1, min(Lgk, lmax_slots))

            raw_lengths_slots[cluster_id] = Lgk
            group_debug[cluster_id] = {
                "Dgk": float(Dgk),
                "nact": int(nact),
                "tau": float(tau),
                "p": float(p),
                "p_idle": float(p_idle),
                "p_succ": float(p_succ),
                "p_col": float(p_col),
                "l_s": float(l_s),
                "l_c": float(l_c),
                "l_bar": float(l_bar),
            }

        non_zero_lengths = [v for v in raw_lengths_slots.values() if v > 0]
        avg_Lg_slots = int(round(sum(non_zero_lengths) / len(non_zero_lengths))) if non_zero_lengths else 0
        avg_Lg_slots = min(avg_Lg_slots, lmax_slots)
        eta_g = 1 if avg_Lg_slots > self.lth_slots else 0

        # --- Pass 1: size every group's slot_us, num_slots_k starts fixed --
        sized_groups: List[Dict[str, Any]] = []

        for cluster_id, rows in sorted(clusters.items()):
            member_aids = sorted(int(r["aid"]) for r in rows)
            if not member_aids:
                continue

            if raw_lengths_slots.get(cluster_id, 0) <= 0:
                final_Lgk_slots = 0
                slot_us = self._apply_safety_lower_bound(self.min_slot_us, rows)
            else:
                final_Lgk_slots = raw_lengths_slots[cluster_id]
                slot_us = int(math.ceil(final_Lgk_slots * sigma * 1e6))
                slot_us = self._apply_safety_lower_bound(slot_us, rows)
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

            self._last_group_slots[cluster_id] = slot_us

            sized_groups.append(
                {
                    "cluster_id": cluster_id,
                    "member_aids": member_aids,
                    "num_slots_k": self.fixed_num_slots,
                    "slot_us": slot_us,
                    "final_Lgk_slots": final_Lgk_slots,
                }
            )

        # --- Pass 1.5: beacon-interval budget clamp (see
        # raw_policy_cluster_adaptive.py's Pass 1.5 for the full mechanism
        # -- unbounded per-cluster slot counts summed across every
        # populated CSV cluster can exceed the fixed beacon interval,
        # which starves the tail (highest-AID) clusters every single
        # beacon and, via the resulting channel-contention spikes at each
        # beacon boundary, suppresses association completeness for
        # not-yet-associated STAs at high N. Trim one slot at a time from
        # whichever group currently holds the most, down to
        # min_slots_per_cluster, so every group reliably gets a window
        # every beacon.
        total_us = sum(g["num_slots_k"] * g["slot_us"] for g in sized_groups)
        budget_us = self.budget_fraction * self.beacon_interval_us

        if total_us > budget_us and sized_groups:
            trimmable = [g for g in sized_groups if g["num_slots_k"] > self.min_slots_per_cluster]
            while total_us > budget_us and trimmable:
                trimmable.sort(key=lambda g: g["num_slots_k"], reverse=True)
                g = trimmable[0]
                total_us -= g["slot_us"]
                g["num_slots_k"] -= 1
                if g["num_slots_k"] <= self.min_slots_per_cluster:
                    trimmable.pop(0)

        # --- Pass 2: emit RawConfig objects and logs ---
        configs: List[RawConfig] = []
        start_offset_us = int(self.ctx.raw_start_time_us)
        slot_values_us: List[int] = []

        for cfg_id, g in enumerate(sized_groups, start=1):
            cluster_id = g["cluster_id"]
            member_aids = g["member_aids"]
            num_slots_k = g["num_slots_k"]
            slot_us = g["slot_us"]
            final_Lgk_slots = g["final_Lgk_slots"]

            slot_values_us.append(slot_us)

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

            dbg = group_debug.get(cluster_id, {})
            self._log(
                "RAW_CLUSTER_GROUP",
                {
                    "cluster_id": int(cluster_id),
                    "start_aid": min(member_aids),
                    "end_aid": max(member_aids),
                    "n_members": len(member_aids),
                    "Dgk": float(dbg.get("Dgk", 0.0)),
                    "nact": int(dbg.get("nact", 0)),
                    "Lgk_slots_raw": int(raw_lengths_slots.get(cluster_id, 0)),
                    "Lg_bar_slots": int(avg_Lg_slots),
                    "Lgk_slots_final": int(final_Lgk_slots),
                    "slot_duration_us": int(slot_us),
                    "num_slots": int(num_slots_k),
                    "base_num_slots": int(self.fixed_num_slots),
                    "eta_g": int(eta_g),
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
            "RAW_CLUSTER_BUILD",
            {
                "groups_built": len(configs),
                "stas_total": len(aids),
                "clusters": len(clusters),
                "fixed_num_slots": self.fixed_num_slots,
                "avg_slot_duration_us": int(self._current_slot_us),
                "group_slot_us_values": [int(x) for x in slot_values_us],
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
            "policy": "cluster_csv_runtime_queue_priority_rate_bianchi",
            "cluster_csv_path": self.cluster_csv_path,
            "fixed_num_slots": self.fixed_num_slots,
            "slot_duration_us": self._current_slot_us,
            "step_us": self.step_us,
            "update_count": self._update_count,
            "ewma_alpha": self.ewma_alpha,
            "cusum_k": self.cusum_k,
            "cusum_h": self.cusum_h,
            "tmax_s": self.tmax_s,
            "lth_slots": self.lth_slots,
            "demand_gain": self.demand_gain,
            "cluster_size_gain": self.cluster_size_gain,
            "burst_gain": self.burst_gain,
            "use_runtime_queue": self.use_runtime_queue,
        }

    def _parse_cluster_csv(self) -> List[Dict[str, Any]]:
        """Read and parse the CSV file once. Returns raw rows (no runtime state)."""
        rows: List[Dict[str, Any]] = []
        try:
            with open(self.cluster_csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        priority = str(row.get("priority", "normal")).strip().lower()
                        if priority not in self.priority_weights:
                            priority = "normal"
                        rows.append(
                            {
                                "aid": int(row["aid"]),
                                "cluster_id": int(row["cluster_id"]),
                                "priority": priority,
                                "tx_rate_bps": float(row.get("tx_rate_bps", 150000.0)),
                                "csv_queue_len": float(row.get("queue_len", 0.0)),
                                "packet_size_bytes": int(
                                    row.get(
                                        "packet_size_bytes",
                                        self.app_cfg.get("packet_size_bytes", 128),
                                    )
                                ),
                            }
                        )
                    except (KeyError, ValueError):
                        pass
        except OSError:
            pass
        return rows

    def _load_cluster_csv(self, connected_aids: List[int]) -> List[Dict[str, Any]]:
        """Filter the cached CSV rows by connected_aids and inject runtime queue state."""
        connected = set(int(a) for a in connected_aids)

        # Per-station demand: E-TAROA's Algorithm 1 (per-station tx-interval
        # tracking from real success/failure + More-Data history) instead
        # of CUSUM-EWMA on a coarse queue-length snapshot. This is a
        # strictly more information-rich signal for the same periodic
        # sensor traffic this policy is benchmarked against -- swapped in
        # after E-TAROA was measured to consistently edge out this policy
        # on PDR/throughput despite otherwise-similar Bianchi/slot-sizing
        # machinery; the estimator is the actual source of that gap, not
        # the CSV-cluster grouping itself, so this closes it while keeping
        # the cluster-aware structure that already wins on energy.
        rates = self._estimator.update(sorted(connected))

        rows: List[Dict[str, Any]] = []

        for base in self._csv_rows_all:
            aid = int(base["aid"])
            if aid not in connected:
                continue

            queue_len = self._get_runtime_queue_len(aid, {"queue_len": base["csv_queue_len"]})
            pred_load = float(rates.get(aid, 0.0))

            rows.append(
                {
                    "aid": aid,
                    "cluster_id": base["cluster_id"],
                    "priority": base["priority"],
                    "tx_rate_bps": base["tx_rate_bps"],
                    "queue_len": float(queue_len),
                    "pred_load": float(pred_load),
                    "packet_size_bytes": base["packet_size_bytes"],
                }
            )

        return rows

    def _get_runtime_queue_len(self, aid: int, row: Dict[str, Any]) -> float:
        if not self.use_runtime_queue:
            return float(row.get("queue_len", 0.0))

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

        return float(row.get("queue_len", 0.0))

    def _group_by_cluster(self, rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            clusters.setdefault(int(row["cluster_id"]), []).append(row)
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
        would create a self-chasing spiral: each split shrinks the
        denominator almost as fast as it shrinks the largest group,
        never converging and fragmenting into dozens of tiny groups at
        high N. Confirmed empirically: a growing-denominator version
        produced 113 groups at N=1000, most sized 6-11, versus the ~10
        groups this fixed-anchor version produces) -- this closes the
        group-size-imbalance gap against free-partitioning policies
        (Adaptive/E-TAROA), which always split contiguous AID ranges
        into near-equal halves via _partition_fixed_by_aid. Without
        this, an external CSV clustering can satisfy the plain count
        floor while leaving e.g. [70, 69, 43, 18] instead of ~[50, 50,
        50, 50] -- still an unfairly congested largest group.

        Before any splitting, merge down to at most target_max_groups if
        the CSV's naturally-populated cluster count already exceeds it
        (true from N~=800 upward) -- always merging the two AID-ADJACENT
        groups with the smallest combined size, which is what keeps every
        merged group's AID range contiguous (required by RawConfig's
        start_aid/end_aid; the 12 real clusters are already AID-ordered
        with no gaps, so adjacency in AID order is well-defined).
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

    def _compute_cluster_demand(self, rows: List[Dict[str, Any]]) -> float:
        demand = 0.0
        has_burst = False

        for r in rows:
            priority = str(r["priority"])
            weight = self.priority_weights.get(priority, 1.0)

            pred_load = max(0.0, float(r["pred_load"]))
            tx_rate = max(1.0, float(r["tx_rate_bps"]))
            pkt_bytes = max(1, int(r["packet_size_bytes"]))

            airtime_s = (8.0 * pkt_bytes) / tx_rate
            demand += weight * pred_load * airtime_s / self._sigma_s()

            if pred_load > self.cusum_h:
                has_burst = True

        demand *= self.demand_gain
        demand *= 1.0 + self.cluster_size_gain * max(0, len(rows) - 1)

        if has_burst:
            demand *= self.burst_gain

        return max(0.0, demand)

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

    def _get_success_collision_times_s(self, rows: List[Dict[str, Any]]) -> Tuple[float, float]:
        if not rows:
            payload_bytes = int(self.app_cfg.get("packet_size_bytes", 128))
            data_rate_bps = 150000.0
        else:
            payload_bytes = max(int(r["packet_size_bytes"]) for r in rows)
            data_rate_bps = min(max(1.0, float(r["tx_rate_bps"])) for r in rows)

        data_mac_oh = int(self.mac_cfg.get("data_mac_overhead_bytes", 36))
        net_oh = int(self.net_cfg.get("net_header_bytes", 16))
        ack_size = int(self.mac_cfg.get("ack_size_bytes", 14))

        preamble = float(self.phy_cfg.get("preamble_time", 320e-6))
        header = float(self.phy_cfg.get("header_time", 80e-6))
        sifs = float(self.mac_cfg.get("sifs", 160e-6))
        difs = float(self.mac_cfg.get("difs", 264e-6))
        ack_timeout = float(self.mac_cfg.get("ack_timeout", 1e-3))

        mode_table = self.phy_cfg.get("mode_table", {}) or {}
        ctrl_mode = str(self.phy_cfg.get("control_mode", "MCS0"))
        ctrl_rate_bps = float(mode_table.get(ctrl_mode, 150000.0))

        data_bits = 8.0 * (payload_bytes + data_mac_oh + net_oh)
        ack_bits = 8.0 * ack_size

        t_data = preamble + header + (data_bits / max(1.0, data_rate_bps))
        t_ack = preamble + header + (ack_bits / max(1.0, ctrl_rate_bps))

        ts = t_data + sifs + t_ack + difs
        tc = t_data + ack_timeout + difs
        return ts, tc

    def _apply_safety_lower_bound(self, slot_us: int, rows: List[Dict[str, Any]]) -> int:
        ts, _ = self._get_success_collision_times_s(rows)
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
            "tau": 0.0,
            "p": 0.0,
            "p_idle": 1.0,
            "p_succ": 0.0,
            "p_col": 0.0,
            "l_s": 0.0,
            "l_c": 0.0,
            "l_bar": 0.0,
        }