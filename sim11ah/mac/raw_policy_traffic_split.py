from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from sim11ah.mac.common import (
    MORSE_RAW_MIN_SLOT_DURATION_US,
    RawBeaconSpreading,
    RawConfig,
    RawPeriodic,
    RawSlotDefinition,
    cslot_to_us,
    us_to_cslot,
)

# ---------------------------------------------------------------------------
# IEEE 802.11ah RAW slot encoding (Section 9.4.2.200 + MORSE convention)
# MORSE encoding:  slot_us = MORSE_RAW_MIN_SLOT_DURATION_US + cslot × 120
#   cslot field   : 11 bits  →  range 0 … 2047
#   Granularity   : 120 µs  (RAW Slot Duration Unit, 802.11ah Table 9-613)
#   Standard min  : MORSE_RAW_MIN_SLOT_DURATION_US = 500 µs  (cslot = 0)
#   Standard max  : 500 + 2047 × 120 = 246 140 µs  ≈ 246 ms  (cslot = 2047)
# ---------------------------------------------------------------------------
_RAW_SLOT_UNIT_US   = 120                                        # granularity
_RAW_SLOT_CSLOT_MAX = 2047                                       # 11-bit field
_RAW_SLOT_STD_MAX_US = cslot_to_us(_RAW_SLOT_CSLOT_MAX)         # 246 140 µs


class TrafficSplitRawPolicy:
    """
    RAW policy that dynamically adjusts the number of groups AND per-group
    slot durations based on per-STA traffic load predicted by CUSUM-EWMA.

    Improvements over a naive traffic-split approach
    ------------------------------------------------
    1. Good initial partition: starts with ceil(N / ideal_group_size) groups
       so the policy is useful from the very first beacon.
    2. Multiple splits per beacon (up to max_splits_per_beacon): converges
       to the right partition in O(log N) beacons instead of O(N).
    3. Per-group Bianchi-based slot sizing: each group gets a slot duration
       sized to its own contention level using the Bianchi fixed-point model,
       the same technique that makes the adaptive policy effective.
    4. CUSUM burst boost on top of EWMA: reacts quickly to load spikes.

    Each beacon interval:
      - Update CUSUM-EWMA demand d_i for each STA.
      - Compute group load D_gk = sum(d_i) for each group.
      - Apply up to max_splits_per_beacon splits on overloaded groups.
      - Apply one merge on the lowest-load qualifying adjacent pair.
      - Size each group's slot duration via Bianchi analysis.
    """

    # Target STAs per group used to seed the initial partition
    _DEFAULT_IDEAL_GROUP_SIZE = 25

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        mac_cfg = self.ctx.cfg["mac"]
        phy_cfg = self.ctx.cfg["phy"]
        app_cfg = self.ctx.cfg["app"]
        net_cfg = self.ctx.cfg["net"]

        # ------------------------------------------------------------------
        # EWMA / CUSUM parameters
        # ------------------------------------------------------------------
        self.alpha      = float(mac_cfg.get("traffic_split_ewma_alpha", 0.4))
        self.cusum_k    = float(mac_cfg.get("traffic_split_cusum_k", 0.15))
        self.cusum_h    = float(mac_cfg.get("traffic_split_cusum_h", 0.8))

        # ------------------------------------------------------------------
        # Split / merge parameters
        # ------------------------------------------------------------------
        # Load per AID above which a group gets split (queue items per beacon)
        self.split_threshold     = float(mac_cfg.get("traffic_split_split_threshold", 0.4))
        # Load per AID below which both adjacent groups may merge
        self.merge_threshold     = float(mac_cfg.get("traffic_split_merge_threshold", 0.05))
        # Max splits applied per beacon — kept modest to avoid over-fragmentation
        self.max_splits_per_beacon = int(mac_cfg.get("traffic_split_max_splits", 2))
        # Consecutive overloaded beacons required before a split is committed
        # (prevents reacting to transient queue spikes)
        self.split_confirm_beacons = int(mac_cfg.get("traffic_split_confirm_beacons", 2))
        # Beacons to lock out all further splits after any split event
        # (prevents oscillation between split and merge)
        self.split_cooldown_beacons = int(mac_cfg.get("traffic_split_cooldown", 4))
        # Minimum AIDs per sub-group after a split
        # (prevents near-empty groups that waste slot budget)
        self.min_aids_per_group = int(mac_cfg.get("traffic_split_min_aids", 5))

        # ------------------------------------------------------------------
        # Group count bounds
        # ------------------------------------------------------------------
        # max_groups is NOT stored as a fixed field — it is computed dynamically
        # via _effective_max_groups() from the beacon budget and current N.
        self.min_groups        = max(1, int(mac_cfg.get("traffic_split_min_groups", 1)))
        self.ideal_group_size  = max(1, int(mac_cfg.get("traffic_split_ideal_group_size",
                                                         self._DEFAULT_IDEAL_GROUP_SIZE)))

        # ------------------------------------------------------------------
        # Sub-slots per group and budget
        # ------------------------------------------------------------------
        self.num_slots        = max(1, int(mac_cfg.get("raw_num_slots",
                                                        getattr(self.ctx, "raw_num_slots", 4))))
        # Minimum sub-slots per group when S is reduced to fit more groups.
        # A group with fewer sub-slots has fewer delivery opportunities per beacon
        # but allows more groups within the budget, reducing per-group contention.
        self.min_slots_per_group = max(1, int(mac_cfg.get("traffic_split_min_slots_per_group", 1)))
        self.budget_fraction  = float(mac_cfg.get("traffic_split_budget_fraction", 0.90))

        # Slot duration bounds are computed below after MAC timings are ready.

        # ------------------------------------------------------------------
        # Bianchi parameters (for per-group slot sizing)
        # ------------------------------------------------------------------
        self._cw_min        = float(mac_cfg.get("cw_min", 15))
        self._cw_max        = float(mac_cfg.get("cw_max", 1023))
        self._sigma_s       = float(mac_cfg.get("slot_time", 52e-6))
        self._sifs          = float(mac_cfg.get("sifs", 160e-6))
        self._difs          = float(mac_cfg.get("difs", 264e-6))
        self._ack_timeout   = float(mac_cfg.get("ack_timeout", 1e-3))
        self._preamble      = float(phy_cfg.get("preamble_time", 320e-6))
        self._header        = float(phy_cfg.get("header_time", 80e-6))
        self._payload_bytes = int(app_cfg.get("packet_size_bytes", 128))
        self._data_oh       = int(mac_cfg.get("data_mac_overhead_bytes", 36))
        self._net_oh        = int(net_cfg.get("net_header_bytes", 16))
        self._ack_bytes     = int(mac_cfg.get("ack_size_bytes", 14))
        mode_table          = phy_cfg.get("mode_table", {})
        mode                = str(phy_cfg.get("default_mode", "MCS0"))
        self._data_rate_bps = float(mode_table.get(mode, 150_000.0))
        # Precompute success / collision times (seconds)
        data_bits     = 8.0 * (self._payload_bytes + self._data_oh + self._net_oh)
        ack_bits      = 8.0 * self._ack_bytes
        t_data        = self._preamble + self._header + (data_bits / max(1.0, self._data_rate_bps))
        t_ack         = self._preamble + self._header + (ack_bits / max(1.0, self._data_rate_bps))
        self._t_succ  = t_data + self._sifs + t_ack + self._difs   # seconds
        self._t_coll  = t_data + self._ack_timeout + self._difs    # seconds
        self._l_s     = self._t_succ / self._sigma_s               # in slot-time units
        self._l_c     = self._t_coll / self._sigma_s

        # ------------------------------------------------------------------
        # Slot duration bounds — derived from IEEE 802.11ah standard values
        # ------------------------------------------------------------------
        # Practical minimum: worst-case average backoff  +  one DATA+ACK exchange.
        #   avg_backoff = (CW_min / 2) × σ  (IEEE 802.11ah CW_min = 15, σ = 52 µs)
        avg_backoff_s   = (self._cw_min / 2.0) * self._sigma_s
        min_exchange_us = int(math.ceil((self._t_succ + avg_backoff_s) * 1e6))
        # Align up to the 120 µs RAW slot unit (Section 9.4.2.200)
        min_exchange_us = (
            math.ceil((min_exchange_us - MORSE_RAW_MIN_SLOT_DURATION_US) / _RAW_SLOT_UNIT_US)
            * _RAW_SLOT_UNIT_US
            + MORSE_RAW_MIN_SLOT_DURATION_US
        )
        # Apply MORSE hardware floor (500 µs), then allow config override
        self.min_slot_us = max(
            MORSE_RAW_MIN_SLOT_DURATION_US,
            min_exchange_us,
            int(mac_cfg.get("traffic_split_min_slot_us", min_exchange_us)),
        )
        # Standard maximum: cslot field is 11 bits → 2047 × 120 + 500 = 246 140 µs
        # Allow config override but never exceed the standard ceiling.
        self.max_slot_us = min(
            _RAW_SLOT_STD_MAX_US,
            int(mac_cfg.get("traffic_split_max_slot_us", _RAW_SLOT_STD_MAX_US)),
        )
        self.max_slot_us = max(self.min_slot_us, self.max_slot_us)

        # (max_groups is computed dynamically per beacon via _effective_max_groups())

        # Consecutive underloaded beacons required before a merge is committed
        # (symmetric hysteresis to split_confirm_beacons — prevents oscillation)
        self.merge_confirm_beacons = int(mac_cfg.get("traffic_split_merge_confirm", 3))
        # Beacons to lock out all further merges after any merge event
        self.merge_cooldown_beacons = int(mac_cfg.get("traffic_split_merge_cooldown", 4))

        # ------------------------------------------------------------------
        # Load-balanced rebalance
        # ------------------------------------------------------------------
        # Only rebalance when the per-AID load ratio between the heaviest and
        # lightest group exceeds this factor.  For uniform traffic the ratio stays
        # near 1.0 so the rebalance never fires (preventing harm).  For genuinely
        # heterogeneous traffic (some STAs have higher arrival rates) the ratio
        # exceeds the threshold and repartitioning helps.
        # Congestion under uniform traffic can push the ratio above 3×; the
        # threshold must be high enough to distinguish genuine heterogeneity
        # (e.g. some STAs at 10× higher arrival rate → ratio ~10×) from
        # temporary congestion imbalance in uniform traffic.
        self.rebalance_imbalance_threshold = float(
            mac_cfg.get("traffic_split_rebalance_threshold", 10.0)
        )
        # Don't rebalance before this beacon (EWMA warmup).
        # With start_phase_jitter_s=30 and beacon_interval=0.5s all STAs start
        # by beacon ~60; EWMA needs ~10 more beacons to converge → use 100.
        self.rebalance_min_beacon = int(mac_cfg.get("traffic_split_rebalance_min_beacon", 100))
        # Minimum beacons between consecutive rebalances.
        self.rebalance_interval   = int(mac_cfg.get("traffic_split_rebalance_interval", 20))
        self._next_rebalance: int = self.rebalance_min_beacon

        # ------------------------------------------------------------------
        # Per-STA EWMA / CUSUM state
        # ------------------------------------------------------------------
        self._ewma:    Dict[int, float] = {}   # pure q̂_i (EWMA baseline only)
        self._demand:  Dict[int, float] = {}   # d_i = q̂_i + burst correction
        self._cusum:   Dict[int, float] = {}

        # Current group partition: list of (start_aid, end_aid) inclusive
        self._groups: Optional[List[Tuple[int, int]]] = None
        # Per-group sub-slot count — may be reduced when G exceeds the fixed-S budget
        self._group_slots: Dict[Tuple[int, int], int] = {}

        # Consecutive beacons each group has been above split_threshold
        self._overload_count: Dict[Tuple[int, int], int] = {}
        # Consecutive beacons each group has been below merge_threshold
        self._underload_count: Dict[Tuple[int, int], int] = {}
        # Remaining beacons in the post-split cooldown window
        self._split_cooldown: int = 0
        # Remaining beacons in the post-merge cooldown window
        self._merge_cooldown: int = 0

        self._beacon_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init_configs(self) -> List[RawConfig]:
        aids       = self._effective_aids([])
        n_stas     = len(aids)
        max_aid    = max(aids) if aids else max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))
        budget_us  = int(self._beacon_interval_us() * self.budget_fraction)
        g_max      = min(
            max(1, budget_us // (self.num_slots * self.min_slot_us)),
            max(1, n_stas // self.min_aids_per_group) if n_stas > 0 else 1,
        )
        n_groups   = max(self.min_groups,
                         min(g_max, max(1, math.ceil(n_stas / self.ideal_group_size))))

        if n_groups > 1 and aids:
            self._groups = self._partition_aids(aids, n_groups)
        else:
            self._groups = [(1, max_aid)]

        init_s_g = self._slots_for_group_count(len(self._groups))
        self._group_slots = {g: init_s_g for g in self._groups}

        slot_us    = self._compute_uniform_slot_us(len(self._groups))
        configs    = []
        offset_us  = int(self.ctx.raw_start_time_us)
        for cfg_id, (s, e) in enumerate(self._groups, start=1):
            configs.append(self._make_config(cfg_id, s, e, offset_us, slot_us, init_s_g))
            offset_us += init_s_g * slot_us
        return configs

    def build_dynamic_configs(self, connected_aids: List[int]) -> List[RawConfig]:
        aids = self._effective_aids(connected_aids)
        if not aids:
            self._log("RAW_TRAFFIC_SPLIT_NO_AIDS", {})
            return []

        if self._groups is None:
            n_stas   = len(aids)
            g_max    = self._effective_max_groups([])
            n_groups = max(self.min_groups,
                           min(g_max, max(1, math.ceil(n_stas / self.ideal_group_size))))
            self._groups = self._partition_aids(aids, n_groups)
            init_s_g = self._slots_for_group_count(len(self._groups))
            self._group_slots = {g: init_s_g for g in self._groups}

        # Step 1: update CUSUM-EWMA demand for every active STA
        self._update_demands(aids)

        # Step 1b: imbalance-triggered load-balanced repartition.
        # EWMA measures queue depth (congestion), not offered load (arrival rate).
        # For uniform traffic all STAs have identical arrival rates so EWMA
        # differences reflect temporary congestion, not genuine load differences —
        # rebalancing in that case would concentrate congested STAs and worsen
        # performance.  Only rebalance when the per-AID load ratio between the
        # heaviest and lightest group exceeds rebalance_imbalance_threshold,
        # which indicates genuinely heterogeneous traffic.
        if self._beacon_count >= self._next_rebalance and sum(self._ewma.values()) > 0.0:
            g_lpas = [
                self._group_load(s, e) / max(1, self._group_size(s, e))
                for s, e in self._groups
            ]
            ratio = max(g_lpas) / max(1e-9, min(g_lpas))
            if ratio >= self.rebalance_imbalance_threshold:
                n_g       = len(self._groups)
                new_parts = self._load_balanced_partition(aids, n_g)
                old_loads = [round(self._group_load(s, e), 3) for s, e in self._groups]
                new_loads = [round(self._group_load(s, e), 3) for s, e in new_parts]
                if new_parts != self._groups:
                    self._groups = new_parts
                    self._overload_count.clear()
                    self._underload_count.clear()
                self._log("RAW_TRAFFIC_SPLIT_REBALANCE", {
                    "beacon":       self._beacon_count,
                    "n_groups":     n_g,
                    "imbalance_ratio": round(ratio, 2),
                    "old_loads":    old_loads,
                    "new_loads":    new_loads,
                })
            self._next_rebalance = self._beacon_count + self.rebalance_interval

        # Step 2: update per-group overload and underload counters
        groups = list(self._groups)
        for g in groups:
            s, e = g
            lpa = self._group_load(s, e) / max(1, self._group_size(s, e))
            if lpa > self.split_threshold:
                self._overload_count[g]  = self._overload_count.get(g, 0) + 1
                self._underload_count[g] = 0
            elif lpa < self.merge_threshold:
                self._underload_count[g] = self._underload_count.get(g, 0) + 1
                self._overload_count[g]  = 0
            else:
                self._overload_count[g]  = 0
                self._underload_count[g] = 0

        # Step 3: splits — only when split cooldown has expired
        if self._split_cooldown > 0:
            self._split_cooldown -= 1
        else:
            for _ in range(self.max_splits_per_beacon):
                if len(groups) >= self._effective_max_groups(groups):
                    break
                new_groups = self._try_split(groups)
                if new_groups is groups:
                    break
                # Split happened — start cooldown, discard stale counters
                self._split_cooldown = self.split_cooldown_beacons
                live = set(new_groups)
                self._overload_count  = {k: v for k, v in self._overload_count.items()  if k in live}
                self._underload_count = {k: v for k, v in self._underload_count.items() if k in live}
                groups = new_groups

        # Step 4: one merge per beacon — only when merge cooldown has expired
        if self._merge_cooldown > 0:
            self._merge_cooldown -= 1
        else:
            new_groups = self._try_merge(groups)
            if new_groups is not groups:
                self._merge_cooldown = self.merge_cooldown_beacons
            groups = new_groups

        self._groups = groups

        # Step 5: compute per-group renewal-process slot durations
        slot_us_list = self._compute_bianchi_slot_durations(self._groups)

        # Step 6: emit one RawConfig per group
        configs:   List[RawConfig] = []
        offset_us  = int(self.ctx.raw_start_time_us)
        cfg_id     = 1
        for (start_aid, end_aid), slot_us in zip(self._groups, slot_us_list):
            s_g  = self._group_slots.get((start_aid, end_aid), self.num_slots)
            load = self._group_load(start_aid, end_aid)
            self._log("RAW_TRAFFIC_SPLIT_GROUP", {
                "beacon":    self._beacon_count,
                "cfg_id":    cfg_id,
                "start_aid": start_aid,
                "end_aid":   end_aid,
                "load":      round(load, 3),
                "slot_us":   slot_us,
                "num_slots": s_g,
            })
            configs.append(self._make_config(cfg_id, start_aid, end_aid, offset_us, slot_us, s_g))
            offset_us += s_g * slot_us
            cfg_id    += 1

        self._log("RAW_TRAFFIC_SPLIT_BUILD", {
            "beacon":     self._beacon_count,
            "num_groups": len(self._groups),
            "num_stas":   len(aids),
            "slot_us_list": slot_us_list,
        })
        self._beacon_count += 1
        return configs if configs else self.init_configs()

    def get_policy_snapshot(self) -> dict:
        return {
            "policy":          "traffic_split",
            "num_groups":      len(self._groups) if self._groups else 0,
            "num_slots":       self.num_slots,
            "alpha":           self.alpha,
            "split_threshold": self.split_threshold,
            "merge_threshold": self.merge_threshold,
            "beacon_count":    self._beacon_count,
        }

    # ------------------------------------------------------------------
    # CUSUM-EWMA demand estimation
    # ------------------------------------------------------------------
    def _update_demands(self, aids: List[int]) -> None:
        for aid in aids:
            obs       = self._observe_sta_load(aid)
            prev_ewma = self._ewma.get(aid, obs)
            ewma      = self.alpha * obs + (1.0 - self.alpha) * prev_ewma
            err       = obs - prev_ewma
            cusum     = max(0.0, self._cusum.get(aid, 0.0) + err - self.cusum_k)
            burst     = max(0.0, cusum - self.cusum_h)
            self._ewma[aid]   = ewma                       # keep pure q̂_i as next baseline
            self._cusum[aid]  = cusum
            self._demand[aid] = max(0.0, ewma + burst)    # d_i used for group load

    def _observe_sta_load(self, aid: int) -> float:
        """MAC TX queue depth — best available per-STA load signal."""
        try:
            node = getattr(self.ctx.sim, "nodes", {}).get(aid)
            if node is None:
                return 0.0
            mac = getattr(node, "mac", None)
            ctx = getattr(mac, "ctx", None)
            if ctx is not None:
                txq = getattr(ctx, "_txq", None)
                if txq is not None:
                    return float(len(txq))
            if mac is not None:
                for attr in ("tx_queue", "_tx_queue", "txq", "_txq"):
                    q = getattr(mac, attr, None)
                    if q is not None:
                        return float(len(q))
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Group load
    # ------------------------------------------------------------------
    def _group_load(self, start_aid: int, end_aid: int) -> float:
        return sum(self._demand.get(aid, 0.0) for aid in range(start_aid, end_aid + 1))

    def _group_size(self, start_aid: int, end_aid: int) -> int:
        return end_aid - start_aid + 1

    def _effective_max_groups(self, groups: List[Tuple[int, int]]) -> int:
        """Hard upper bound = min(budget/min_slot with adaptive S, N/n_min)."""
        budget_us  = int(self._beacon_interval_us() * self.budget_fraction)
        g_max_budget = max(1, budget_us // (self.min_slots_per_group * self.min_slot_us))
        total_aids   = sum(e - s + 1 for s, e in groups) if groups else 1
        g_max_stas   = max(1, total_aids // self.min_aids_per_group)
        return min(g_max_budget, g_max_stas)

    def _slots_for_group_count(self, G: int) -> int:
        """Compute sub-slots per group for G groups.
        Uses the configured num_slots as long as G groups fit in the beacon budget.
        When G exceeds the fixed-S budget, reduces S proportionally so all groups
        still fit, down to min_slots_per_group."""
        budget_us = int(self._beacon_interval_us() * self.budget_fraction)
        if G * self.num_slots * self.min_slot_us <= budget_us:
            return self.num_slots
        return max(self.min_slots_per_group,
                   budget_us // (G * self.min_slot_us))

    # ------------------------------------------------------------------
    # Split: threshold per AID (normalised), up to max_splits_per_beacon
    # ------------------------------------------------------------------
    def _try_split(self, groups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(groups) >= self._effective_max_groups(groups):
            return groups

        best_idx      = -1
        best_load_pa  = self.split_threshold   # load per AID must exceed this

        for i, (s, e) in enumerate(groups):
            if e <= s:
                continue

            # Guard 1: sustained overload — must have been above threshold for
            # split_confirm_beacons consecutive beacons before we commit.
            if self._overload_count.get((s, e), 0) < self.split_confirm_beacons:
                continue

            n   = self._group_size(s, e)
            lpa = self._group_load(s, e) / max(1, n)
            if lpa <= best_load_pa:
                continue

            # Guard 2: minimum sub-group size — both halves must have at least
            # min_aids_per_group AIDs so we don't create near-empty groups.
            # Use the load-median split point (same as the actual split) so the
            # guard reflects the real boundary, not the AID midpoint.
            candidate_mid = self._load_median_split(s, e)
            if (candidate_mid - s + 1) < self.min_aids_per_group or (e - candidate_mid) < self.min_aids_per_group:
                continue

            best_load_pa = lpa
            best_idx     = i

        if best_idx == -1:
            return groups

        s, e  = groups[best_idx]
        mid   = self._load_median_split(s, e)
        new_g = groups[:best_idx] + [(s, mid), (mid + 1, e)] + groups[best_idx + 1:]

        # Compute adaptive sub-slots for the new group count.
        # First try keeping S fixed (best for contention reduction).
        # If that overflows the budget, reduce S proportionally.
        new_G   = len(new_g)
        new_s_g = self._slots_for_group_count(new_G)
        new_slots_map = {g: new_s_g for g in new_g}

        budget_us    = int(self._beacon_interval_us() * self.budget_fraction)
        trial_slots  = self._compute_bianchi_slot_durations(new_g, new_slots_map)
        total_raw_us = sum(new_s_g * d for d in trial_slots)
        if total_raw_us > budget_us:
            # Even with reduced S this configuration overflows — reject split.
            self._log("RAW_TRAFFIC_SPLIT_BUDGET_BLOCK", {
                "beacon": self._beacon_count, "proposed_groups": new_G,
                "slots_per_group": new_s_g, "total_raw_us": total_raw_us,
                "budget_us": budget_us,
            })
            return groups

        # Commit: update group_slots for all groups in the new partition.
        self._group_slots = new_slots_map

        self._log("RAW_TRAFFIC_SPLIT_SPLIT", {
            "beacon": self._beacon_count, "group_idx": best_idx,
            "start_aid": s, "end_aid": e, "mid": mid,
            "load_per_aid": round(best_load_pa, 3),
            "slots_per_group": new_s_g,
            "confirm_beacons": self._overload_count.get((s, e), 0),
            "load_left":  round(self._group_load(s,     mid), 3),
            "load_right": round(self._group_load(mid+1, e),   3),
        })
        return new_g

    # ------------------------------------------------------------------
    # Merge: threshold per AID (normalised), one per beacon
    # ------------------------------------------------------------------
    def _try_merge(self, groups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(groups) <= self.min_groups:
            return groups

        best_idx     = -1
        best_combined = float("inf")

        for i in range(len(groups) - 1):
            s1, e1 = groups[i]
            s2, e2 = groups[i + 1]
            if s2 != e1 + 1:
                continue
            n1   = self._group_size(s1, e1)
            n2   = self._group_size(s2, e2)
            lpa1 = self._group_load(s1, e1) / max(1, n1)
            lpa2 = self._group_load(s2, e2) / max(1, n2)
            # Both groups must be below merge_threshold for merge_confirm_beacons
            # consecutive beacons — symmetric hysteresis prevents oscillation.
            if (lpa1 < self.merge_threshold and lpa2 < self.merge_threshold
                    and self._underload_count.get((s1, e1), 0) >= self.merge_confirm_beacons
                    and self._underload_count.get((s2, e2), 0) >= self.merge_confirm_beacons):
                combined = lpa1 + lpa2
                if combined < best_combined:
                    best_combined = combined
                    best_idx      = i

        if best_idx == -1:
            return groups

        s1, e1  = groups[best_idx]
        _s2, e2 = groups[best_idx + 1]
        new_g   = groups[:best_idx] + [(s1, e2)] + groups[best_idx + 2:]
        # Reset both counters for the merged group.
        self._overload_count.pop((s1, e1),  None)
        self._overload_count.pop((_s2, e2), None)
        self._underload_count.pop((s1, e1),  None)
        self._underload_count.pop((_s2, e2), None)
        self._overload_count[(s1, e2)]  = 0
        self._underload_count[(s1, e2)] = 0
        # Recalculate adaptive S for the reduced group count.
        new_s_g = self._slots_for_group_count(len(new_g))
        self._group_slots = {g: new_s_g for g in new_g}
        self._log("RAW_TRAFFIC_SPLIT_MERGE", {
            "beacon": self._beacon_count, "group_idx": best_idx,
            "merged_range": (s1, e2), "combined_load_pa": round(best_combined, 3),
            "slots_per_group": new_s_g,
        })
        return new_g

    # ------------------------------------------------------------------
    # Load-balanced split point
    # ------------------------------------------------------------------
    def _load_median_split(self, s: int, e: int) -> int:
        """Return the AID boundary that most evenly divides the group's predicted load.
        Falls back to AID midpoint when no EWMA data exists."""
        total = self._group_load(s, e)
        if total <= 0.0:
            return (s + e) // 2
        half = total / 2.0
        cum = 0.0
        for aid in range(s, e):  # never let mid == e (would create empty right half)
            cum += self._demand.get(aid, 0.0)
            if cum >= half:
                return aid
        return (s + e) // 2

    # ------------------------------------------------------------------
    # Per-group Bianchi-based slot sizing
    # ------------------------------------------------------------------
    def _compute_bianchi_slot_durations(
        self,
        groups: List[Tuple[int, int]],
        group_slots: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> List[int]:
        """
        Compute per-group slot duration using the LACA two-level renewal process
        (Taramit et al., IEEE Access 2023, eq. 1 / 25 / 26).

        group_slots: optional mapping of group → num_slots override.  When not
        supplied, falls back to self._group_slots then self.num_slots.
        """
        beacon_us = self._beacon_interval_us()
        budget_us = int(beacon_us * self.budget_fraction)
        beta_s    = self._l_s * self._sigma_s

        slots_map  = group_slots if group_slots is not None else self._group_slots
        ideal_us:   List[int] = []
        slots_used: List[int] = []

        for s, e in groups:
            s_g    = slots_map.get((s, e), self.num_slots)
            D_gk   = self._group_load(s, e)
            # Cap at group size: can't have more concurrent transmitters than AIDs in group.
            # Backlogged queues inflate D_gk far beyond group_size; the cap prevents
            # astronomical T* values that collapse to min_slot after budget scaling.
            n_raw      = max(1, math.ceil(D_gk / max(1, s_g)))
            n_per_slot = min(n_raw, self._group_size(s, e))
            t_star_s   = self._laca_slot_s(n_per_slot, self._sigma_s, beta_s)
            dur_us     = self._quantise(int(math.ceil(t_star_s * 1e6)))
            ideal_us.append(dur_us)
            slots_used.append(s_g)

        # Scale down proportionally if total budget is exceeded
        total_ideal = sum(ideal_us[i] * slots_used[i] for i in range(len(groups)))
        if total_ideal > budget_us:
            scale = budget_us / total_ideal
            ideal_us = [self._quantise(int(v * scale)) for v in ideal_us]

        return ideal_us

    def _laca_slot_s(self, n_s: int, sigma: float, beta: float) -> float:
        """
        T*_S for n_s stations each holding one packet (eq. 1 / 25 / 26,
        Taramit et al. 2023).  No capture (ideal channel).

        T*_S = sum_{k=1}^{n_s} Z_k
        Z_k  = (1 / P_{k,s}) * (sigma * P_{k,i}/(1-P_{k,i}) + beta)

        n_k  = n_s - k + 1  (contenders in k-th delivery cycle)
        """
        t_star = 0.0
        for k in range(1, n_s + 1):
            n_k = n_s - k + 1
            tau, _ = self._bianchi_fixed_point(n_k)
            p_idle = (1.0 - tau) ** n_k
            if 1.0 - p_idle < 1e-12:
                t_star += beta
                continue
            p_succ_unc = n_k * tau * (1.0 - tau) ** (n_k - 1)
            p_k_s = max(1e-12, p_succ_unc / (1.0 - p_idle))
            t_star += (1.0 / p_k_s) * (sigma * p_idle / (1.0 - p_idle) + beta)
        return t_star

    def _bianchi_fixed_point(self, n: int) -> Tuple[float, float]:
        if n <= 1:
            W0  = self._cw_min + 1.0
            tau = min(0.5, 2.0 / (W0 + 1.0))
            return tau, 0.0
        W0  = self._cw_min + 1.0
        try:
            m = max(0, int(round(math.log2((self._cw_max + 1.0) / W0))))
        except Exception:
            m = 6
        tau = min(0.5, 2.0 / (W0 + 1.0))
        for _ in range(50):
            p       = 1.0 - (1.0 - tau) ** (n - 1)
            denom   = (1.0 - 2.0 * p) * (W0 + 1.0) + p * W0 * (1.0 - (2.0 * p) ** m)
            if abs(denom) < 1e-12:
                break
            tau_new = min(1.0, max(1e-9, (2.0 * (1.0 - 2.0 * p)) / denom))
            if abs(tau_new - tau) < 1e-4:
                tau = tau_new
                break
            tau = tau_new
        p = 1.0 - (1.0 - tau) ** (n - 1)
        return tau, p

    # ------------------------------------------------------------------
    # Uniform slot duration (fallback / init)
    # ------------------------------------------------------------------
    def _quantise(self, slot_us: int) -> int:
        """Round down to the nearest 120 µs RAW slot unit, then clamp."""
        q = (int(slot_us) // _RAW_SLOT_UNIT_US) * _RAW_SLOT_UNIT_US
        return max(self.min_slot_us, min(self.max_slot_us, q))

    def _compute_uniform_slot_us(self, num_groups: int) -> int:
        beacon_us  = self._beacon_interval_us()
        budget_us  = int(beacon_us * self.budget_fraction)
        s_g        = self._slots_for_group_count(max(1, num_groups))
        total_slots = max(1, num_groups) * s_g
        dur = budget_us // total_slots
        return self._quantise(dur)

    def _beacon_interval_us(self) -> int:
        try:
            bi_s = float(self.ctx.cfg["mac"].get("beacon_interval", 0.1024))
            return max(1, int(bi_s * 1_000_000))
        except Exception:
            return 102_400

    # ------------------------------------------------------------------
    # AID partitioning
    # ------------------------------------------------------------------
    def _partition_aids(self, aids: List[int], n_groups: int) -> List[Tuple[int, int]]:
        """Equal-width AID-range partition — used at init when EWMA is not yet warm."""
        aids     = sorted(aids)
        n        = len(aids)
        n_groups = max(1, min(n_groups, n))
        groups   = []
        for g in range(n_groups):
            lo = aids[g * n // n_groups]
            hi = aids[min((g + 1) * n // n_groups, n) - 1]
            groups.append((lo, hi))
        return groups

    def _load_balanced_partition(self, aids: List[int], n_groups: int) -> List[Tuple[int, int]]:
        """Partition aids into n_groups contiguous AID ranges minimising load imbalance.

        Uses prefix-sum greedy: each split point is placed where the cumulative
        load is closest to the proportional target (k/n_groups × total_load).
        Falls back to equal-width when no EWMA data exists.
        """
        aids     = sorted(aids)
        n        = len(aids)
        n_groups = max(1, min(n_groups, n))

        loads  = [self._ewma.get(a, 0.0) for a in aids]
        total  = sum(loads)
        if total <= 0.0:
            return self._partition_aids(aids, n_groups)

        # Prefix sums for O(1) range-load queries
        prefix = [0.0] * (n + 1)
        for i, v in enumerate(loads):
            prefix[i + 1] = prefix[i] + v

        groups: List[Tuple[int, int]] = []
        lo = 0
        for g in range(n_groups - 1):
            target_cum = total * (g + 1) / n_groups
            # Advance hi while moving one step closer to target_cum; always
            # leave enough AIDs for all remaining groups (min_aids_per_group each).
            remaining_groups = n_groups - g - 1
            max_hi = n - remaining_groups * max(1, self.min_aids_per_group) - 1
            # Also enforce min_aids_per_group for the current group itself.
            min_hi = lo + self.min_aids_per_group - 1
            hi = max(lo, min(min_hi, max_hi))
            while hi < max_hi:
                err_now  = (prefix[hi + 1] - target_cum) ** 2
                err_next = (prefix[hi + 2] - target_cum) ** 2
                if err_next >= err_now:
                    break
                hi += 1
            groups.append((aids[lo], aids[hi]))
            lo = hi + 1

        groups.append((aids[lo], aids[-1]))
        return groups

    # ------------------------------------------------------------------
    # RawConfig factory
    # ------------------------------------------------------------------
    def _make_config(
        self,
        cfg_id: int,
        start_aid: int,
        end_aid: int,
        start_time_us: int,
        slot_duration_us: int,
        num_slots: Optional[int] = None,
    ) -> RawConfig:
        return RawConfig(
            id=cfg_id,
            raw_type=self.ctx.raw_type,
            start_aid=start_aid,
            end_aid=end_aid,
            start_time_us=start_time_us,
            slot_definition=RawSlotDefinition(
                num_slots=num_slots if num_slots is not None else self.num_slots,
                slot_duration_us=slot_duration_us,
                cross_slot_boundary=self.ctx.raw_cross_slot,
            ),
            beacon_spreading=RawBeaconSpreading(),
            periodic=RawPeriodic(),
            enabled=True,
        )

    # ------------------------------------------------------------------
    # AID discovery
    # ------------------------------------------------------------------
    def _effective_aids(self, connected_aids: List[int]) -> List[int]:
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
            nodes = getattr(self.ctx.sim, "nodes", {})
            vals  = sorted(int(nid) for nid in nodes if int(nid) > 0)
            if vals:
                return vals
        except Exception:
            pass
        return []
