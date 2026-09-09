from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

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
# IEEE 802.11ah RAW slot encoding (Section 9.4.2.200)
#   slot_us = MORSE_RAW_MIN_SLOT_DURATION_US + cslot × 120
#   cslot   : 11 bits → 0 … 2047
# ---------------------------------------------------------------------------
_RAW_SLOT_UNIT_US    = 120
_RAW_SLOT_CSLOT_MAX  = 2047
_RAW_SLOT_STD_MAX_US = cslot_to_us(_RAW_SLOT_CSLOT_MAX)   # 246 140 µs


class TrafficAwareRawPolicy:
    """
    IEEE 802.11ah RAW policy: traffic-aware resource allocation.

    Three new mechanisms on top of AID-based grouping and EWMA+CUSUM
    demand prediction:

    A  Demand-proportional slot allocation
       ─────────────────────────────────────
       Instead of giving every group the same number of sub-slots (S),
       each group i receives

           S_i ∝ D_i / Σ_j D_j  × budget

       where D_i = Σ_k d_k  (sum of per-STA EWMA+CUSUM demand across the
       AID range of group i).  High-demand groups absorb more of the
       beacon budget; idle groups release resources to congested peers.

    B  Multi-pass scheduling for high-demand groups
       ─────────────────────────────────────────────
       Groups whose per-AID demand d_i/n_i exceeds `repeat_threshold`
       receive a second RAW window within the same beacon interval,
       doubling their channel-access opportunities.  The second window
       is sized from the budget remaining after primary allocation, up to
       `repeat_max_frac` of the full beacon budget.

    C  Periodic RAW (PRAW, IEEE 802.11ah §9.4.2.200) skip for idle groups
       ─────────────────────────────────────────────────────────────────────
       Groups whose per-AID demand stays below `skip_threshold` for
       `skip_confirm` consecutive beacons enter PRAW mode: they become
       active only once every `skip_period` beacon intervals.  When
       demand rises above `wakeup_threshold`, the group is immediately
       promoted back to active-every-beacon scheduling.

       The PRAW phase is tracked inside this policy (not delegated to the
       engine) because `build_dynamic_configs` is called every beacon and
       replaces the config list, so engine-side PRAW counters would be
       reset on each call.  In sleeping beacons the group's RawConfig is
       simply omitted from the returned list.

    Traffic prediction (AP-side, per STA)
    ──────────────────────────────────────
        d_i  = EWMA(q_i) + CUSUM_burst(q_i)
        q_i  = TX-queue depth observed at the AP for STA i

    Group structure
    ───────────────
    AID-range based (§9.22.3).  Initial partition seeded from
    `ideal_group_size`; maintained by split / merge with hysteresis.
    """

    _DEFAULT_IDEAL_GROUP_SIZE = 25

    # -----------------------------------------------------------------------
    def __init__(self, ctx, log_fn) -> None:
        self.ctx   = ctx
        self._log  = log_fn
        mac_cfg    = ctx.cfg["mac"]
        phy_cfg    = ctx.cfg["phy"]
        app_cfg    = ctx.cfg["app"]
        net_cfg    = ctx.cfg["net"]

        # ── EWMA / CUSUM ─────────────────────────────────────────────────────
        self.alpha   = float(mac_cfg.get("taw_ewma_alpha", 0.3))
        self.cusum_k = float(mac_cfg.get("taw_cusum_k",   0.15))
        self.cusum_h = float(mac_cfg.get("taw_cusum_h",   0.8))

        # ── Group structure ───────────────────────────────────────────────────
        self.ideal_group_size   = max(1, int(mac_cfg.get("taw_ideal_group_size",
                                                          self._DEFAULT_IDEAL_GROUP_SIZE)))
        self.min_aids_per_group = max(2, int(mac_cfg.get("taw_min_aids_per_group", 5)))
        self.min_groups         = max(1, int(mac_cfg.get("taw_min_groups", 1)))

        # ── Split / merge hysteresis ──────────────────────────────────────────
        self.split_threshold        = float(mac_cfg.get("taw_split_threshold",  0.4))
        self.merge_threshold        = float(mac_cfg.get("taw_merge_threshold",  0.05))
        self.split_confirm          = int(mac_cfg.get("taw_split_confirm",      2))
        self.merge_confirm          = int(mac_cfg.get("taw_merge_confirm",      3))
        self.split_cooldown_len     = int(mac_cfg.get("taw_split_cooldown",     4))
        self.merge_cooldown_len     = int(mac_cfg.get("taw_merge_cooldown",     4))
        self.max_splits_per_beacon  = int(mac_cfg.get("taw_max_splits",         2))

        # ── Budget ────────────────────────────────────────────────────────────
        self.budget_fraction = float(mac_cfg.get("taw_budget_fraction", 0.90))
        self.min_slots       = max(1, int(mac_cfg.get("taw_min_slots",  1)))
        self.max_slots       = max(1, int(mac_cfg.get("taw_max_slots",  16)))
        # Target slots per group used to compute the maximum number of groups
        # that still allow each group to have taw_target_slots slots.  Matching
        # the logic in traffic_split (_slots_for_group_count) ensures the initial
        # group count matches for a fair comparison.
        self.target_slots    = max(self.min_slots,
                                   int(mac_cfg.get("raw_num_slots",
                                                   mac_cfg.get("taw_target_slots", 4))))

        # ── B: multi-pass for high-demand groups ──────────────────────────────
        # demand-per-AID threshold above which a group earns a 2nd RAW window
        self.repeat_threshold = float(mac_cfg.get("taw_repeat_threshold",   1.5))
        # cap: 2nd-window budget ≤ this fraction of the total beacon budget
        self.repeat_max_frac  = float(mac_cfg.get("taw_repeat_max_fraction", 0.25))

        # ── C: PRAW skip for idle groups ──────────────────────────────────────
        # demand-per-AID below which idle skip is considered.
        # Deliberately conservative: PRAW should only activate for groups
        # with near-zero traffic (e.g., AID ranges with no associated STAs,
        # or STAs that have ceased transmitting entirely).  Lightly-loaded
        # groups in uniform periodic traffic should NOT be skipped.
        self.skip_threshold   = float(mac_cfg.get("taw_skip_threshold",  0.001))
        # Consecutive under-threshold beacons before PRAW is enabled.
        # Requires sustained idle evidence to prevent false positives from
        # transient queue-depth fluctuations.
        self.skip_confirm     = int(mac_cfg.get("taw_skip_confirm",       30))
        # Do not evaluate PRAW entry until EWMA has had time to warm up.
        self.praw_warmup_beacons = int(mac_cfg.get("taw_praw_warmup",    100))
        # PRAW period: group active 1 out of every skip_period BIs
        self.skip_period      = max(2, int(mac_cfg.get("taw_skip_period", 4)))
        # demand-per-AID above which a PRAW group wakes up immediately
        self.wakeup_threshold = float(mac_cfg.get("taw_wakeup_threshold", 0.005))

        # ── Bianchi / LACA timing ─────────────────────────────────────────────
        self._cw_min        = float(mac_cfg.get("cw_min",        15))
        self._cw_max        = float(mac_cfg.get("cw_max",        1023))
        self._sigma_s       = float(mac_cfg.get("slot_time",     52e-6))
        self._sifs          = float(mac_cfg.get("sifs",          160e-6))
        self._difs          = float(mac_cfg.get("difs",          264e-6))
        self._ack_timeout   = float(mac_cfg.get("ack_timeout",   1e-3))
        self._preamble      = float(phy_cfg.get("preamble_time", 320e-6))
        self._header        = float(phy_cfg.get("header_time",   80e-6))
        self._payload_bytes = int(app_cfg.get("packet_size_bytes", 128))
        self._data_oh       = int(mac_cfg.get("data_mac_overhead_bytes", 36))
        self._net_oh        = int(net_cfg.get("net_header_bytes", 16))
        self._ack_bytes     = int(mac_cfg.get("ack_size_bytes",  14))
        mode_table          = phy_cfg.get("mode_table", {})
        mode                = str(phy_cfg.get("default_mode", "MCS0"))
        self._data_rate_bps = float(mode_table.get(mode, 150_000.0))
        data_bits    = 8.0 * (self._payload_bytes + self._data_oh + self._net_oh)
        ack_bits     = 8.0 * self._ack_bytes
        t_data       = self._preamble + self._header + data_bits  / max(1.0, self._data_rate_bps)
        t_ack        = self._preamble + self._header + ack_bits   / max(1.0, self._data_rate_bps)
        self._t_succ = t_data + self._sifs + t_ack + self._difs
        self._t_coll = t_data + self._ack_timeout  + self._difs
        self._l_s    = self._t_succ / self._sigma_s
        self._l_c    = self._t_coll / self._sigma_s

        # ── Slot duration bounds (IEEE 802.11ah §9.4.2.200) ───────────────────
        avg_backoff_s   = (self._cw_min / 2.0) * self._sigma_s
        min_exchange_us = int(math.ceil((self._t_succ + avg_backoff_s) * 1e6))
        min_exchange_us = (
            math.ceil((min_exchange_us - MORSE_RAW_MIN_SLOT_DURATION_US) / _RAW_SLOT_UNIT_US)
            * _RAW_SLOT_UNIT_US + MORSE_RAW_MIN_SLOT_DURATION_US
        )
        self.min_slot_us = max(
            MORSE_RAW_MIN_SLOT_DURATION_US,
            min_exchange_us,
            int(mac_cfg.get("taw_min_slot_us", min_exchange_us)),
        )
        self.max_slot_us = min(
            _RAW_SLOT_STD_MAX_US,
            int(mac_cfg.get("taw_max_slot_us", _RAW_SLOT_STD_MAX_US)),
        )
        self.max_slot_us = max(self.min_slot_us, self.max_slot_us)

        # ── Per-STA demand state ──────────────────────────────────────────────
        self._ewma:   Dict[int, float] = {}   # pure EWMA baseline q̂_i
        self._cusum:  Dict[int, float] = {}   # CUSUM accumulator S_i
        self._demand: Dict[int, float] = {}   # d_i = q̂_i + CUSUM burst

        # ── Group partition ───────────────────────────────────────────────────
        self._groups: Optional[List[Tuple[int, int]]] = None

        # ── Split / merge counters ────────────────────────────────────────────
        self._overload_count:  Dict[Tuple[int, int], int] = {}
        self._underload_count: Dict[Tuple[int, int], int] = {}
        self._split_cooldown:  int = 0
        self._merge_cooldown:  int = 0

        # ── PRAW state (C) ────────────────────────────────────────────────────
        # _praw_mode[g]  : True  = group is currently in PRAW-skip mode
        # _praw_phase[g] : 0     = active this BI (emit config)
        #                  1..skip_period-1 = sleeping (omit config)
        # _under_count[g]: consecutive BIs with dpa < skip_threshold (active mode)
        # _over_count[g] : consecutive BIs with dpa > wakeup_threshold (PRAW, active BIs only)
        self._praw_mode:  Dict[Tuple[int, int], bool] = {}
        self._praw_phase: Dict[Tuple[int, int], int]  = {}
        self._under_count: Dict[Tuple[int, int], int] = {}
        self._over_count:  Dict[Tuple[int, int], int] = {}

        self._beacon_count: int = 0

    # =========================================================================
    # Public API (called by RawEngine)
    # =========================================================================

    def init_configs(self) -> List[RawConfig]:
        """Called once at startup before any beacon fires."""
        aids      = self._effective_aids([])
        n_stas    = len(aids)
        max_aid   = max(aids) if aids else max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))
        budget_us = int(self._beacon_interval_us() * self.budget_fraction)
        # Cap group count so each group gets at least target_slots slots.
        # This mirrors traffic_split's _slots_for_group_count cap and prevents
        # over-fragmenting the network at start, which reduces per-group contention.
        g_max_target = max(1, budget_us // (self.target_slots * self.min_slot_us))
        g_max_stas   = max(1, n_stas // self.min_aids_per_group) if n_stas > 0 else 1
        g_max        = min(g_max_target, g_max_stas)
        n_groups  = max(self.min_groups,
                        min(g_max, max(1, math.ceil(n_stas / self.ideal_group_size))))

        self._groups = (
            self._partition_aids(aids, n_groups) if aids
            else [(1, max_aid)]
        )
        # Initialise PRAW state for all groups (all active at start)
        for g in self._groups:
            self._praw_mode[g]   = False
            self._praw_phase[g]  = 0
            self._under_count[g] = 0
            self._over_count[g]  = 0

        # Equal slot allocation at init (no demand data yet)
        equal_slots = max(self.min_slots, min(
            self.max_slots,
            budget_us // (len(self._groups) * max(1, self.min_slot_us)),
        ))
        slot_us   = self._quantise(budget_us // max(1, len(self._groups) * equal_slots))
        configs   = []
        offset_us = int(self.ctx.raw_start_time_us)
        for cfg_id, (s, e) in enumerate(self._groups, start=1):
            configs.append(self._make_config(cfg_id, s, e, offset_us, slot_us, equal_slots))
            offset_us += equal_slots * slot_us
        return configs

    def build_dynamic_configs(self, connected_aids: List[int]) -> List[RawConfig]:
        """Called every beacon interval by RawEngine.refresh_dynamic_configs."""
        aids = self._effective_aids(connected_aids)
        if not aids:
            self._log("TAW_NO_AIDS", {})
            return []

        if self._groups is None:
            n_stas    = len(aids)
            budget_us = int(self._beacon_interval_us() * self.budget_fraction)
            g_max_target = max(1, budget_us // (self.target_slots * self.min_slot_us))
            g_max_stas   = max(1, n_stas // self.min_aids_per_group) if n_stas > 0 else 1
            g_max        = min(g_max_target, g_max_stas)
            n_groups = max(self.min_groups,
                           min(g_max, max(1, math.ceil(n_stas / self.ideal_group_size))))
            self._groups = self._partition_aids(aids, n_groups)
            for g in self._groups:
                self._praw_mode[g]   = False
                self._praw_phase[g]  = 0
                self._under_count[g] = 0
                self._over_count[g]  = 0

        self._beacon_count += 1

        # ── Step 1: update EWMA+CUSUM demand for every associated STA ─────────
        self._update_demands(aids)

        # ── Step 2: split / merge group boundaries ────────────────────────────
        self._update_split_merge_counters()
        if self._split_cooldown > 0:
            self._split_cooldown -= 1
        else:
            for _ in range(self.max_splits_per_beacon):
                new_g = self._try_split(self._groups)
                if new_g is self._groups:
                    break
                self._split_cooldown = self.split_cooldown_len
                live = set(new_g)
                self._overload_count  = {k: v for k, v in self._overload_count.items()  if k in live}
                self._underload_count = {k: v for k, v in self._underload_count.items() if k in live}
                self._groups = new_g

        if self._merge_cooldown > 0:
            self._merge_cooldown -= 1
        else:
            new_g = self._try_merge(self._groups)
            if new_g is not self._groups:
                self._merge_cooldown = self.merge_cooldown_len
            self._groups = new_g

        # ── Step 3: PRAW phase advance + transition checks (C) ────────────────
        sleeping = self._advance_praw_phases()
        self._update_praw_transitions(sleeping)

        # Edge case: all groups asleep → force-wake all (prevents complete blackout)
        active_groups = [g for g in self._groups if g not in sleeping]
        if not active_groups:
            for g in self._groups:
                self._praw_mode[g]  = False
                self._praw_phase[g] = 0
            sleeping      = set()
            active_groups = list(self._groups)

        # ── Step 4: pre-identify repeat candidates; reserve budget for them ──────
        # Check BEFORE primary allocation so high-demand groups are guaranteed
        # a second window — not left with zero budget after primary is full.
        budget_us = int(self._beacon_interval_us() * self.budget_fraction)
        repeat_candidates = sorted(
            [g for g in active_groups if self._group_dpa(*g) >= self.repeat_threshold],
            key=lambda g: self._group_dpa(*g),
            reverse=True,
        )
        if repeat_candidates:
            # Reserve repeat_max_frac of total budget for second-pass windows.
            # Primary allocation uses the remainder.
            repeat_reserve = int(budget_us * self.repeat_max_frac)
            primary_budget = budget_us - repeat_reserve
        else:
            primary_budget = budget_us
            repeat_reserve = 0

        # ── Step 5: demand-proportional budget allocation (A) ─────────────────
        slot_us_map, slots_map = self._allocate_budget(active_groups, primary_budget)

        # Infeasibility fallback: if even G_act groups at 1 slot x D_min
        # exceed the *primary* budget, the repeat reserve cannot be honoured
        # this beacon — give it back to the primary allocation entirely.
        primary_min_us = len(active_groups) * self.min_slots * self.min_slot_us
        if repeat_reserve > 0 and primary_min_us > primary_budget:
            primary_budget = budget_us
            repeat_reserve = 0
            repeat_candidates = []
            slot_us_map, slots_map = self._allocate_budget(active_groups, primary_budget)

        # ── Step 6: build primary RawConfig per active group ──────────────────
        configs:   List[RawConfig] = []
        offset_us  = int(self.ctx.raw_start_time_us)
        cfg_id     = 1
        for g in self._groups:
            if g in sleeping:
                continue
            s, e    = g
            slot_us = slot_us_map[g]
            n_slots = slots_map[g]
            self._log("TAW_GROUP", {
                "beacon":    self._beacon_count,
                "cfg_id":    cfg_id,
                "start_aid": s,
                "end_aid":   e,
                "num_slots": n_slots,
                "slot_us":   slot_us,
                "demand":    round(self._group_load(s, e), 3),
                "dpa":       round(self._group_dpa(s, e), 4),
                "praw":      False,
            })
            configs.append(self._make_config(cfg_id, s, e, offset_us, slot_us, n_slots))
            offset_us += n_slots * slot_us
            cfg_id    += 1

        # ── Step 7: multi-pass windows for high-demand groups (B) ─────────────
        repeat_budget = repeat_reserve

        if repeat_budget >= self.min_slot_us and repeat_candidates:
            for g in repeat_candidates:
                if repeat_budget < self.min_slot_us:
                    break
                s, e    = g
                slot_us = slot_us_map[g]
                # Give the repeat window proportionally the same num_slots as the primary,
                # capped so the window fits within remaining repeat budget.
                rep_slots = max(1, min(
                    slots_map[g],
                    self.max_slots // 2,
                    repeat_budget // max(1, slot_us),
                ))
                cost = rep_slots * slot_us
                if cost > repeat_budget:
                    rep_slots = max(1, repeat_budget // max(1, slot_us))
                    cost = rep_slots * slot_us
                if rep_slots < 1 or cost > repeat_budget:
                    continue
                self._log("TAW_REPEAT_WINDOW", {
                    "beacon":    self._beacon_count,
                    "cfg_id":    cfg_id,
                    "start_aid": s,
                    "end_aid":   e,
                    "rep_slots": rep_slots,
                    "slot_us":   slot_us,
                    "dpa":       round(self._group_dpa(s, e), 4),
                })
                configs.append(self._make_config(cfg_id, s, e, offset_us, slot_us, rep_slots))
                offset_us    += cost
                repeat_budget -= cost
                cfg_id        += 1

        primary_used = sum(slots_map[g] * slot_us_map[g] for g in active_groups)
        self._log("TAW_BUILD", {
            "beacon":          self._beacon_count,
            "total_groups":    len(self._groups),
            "active_groups":   len(active_groups),
            "sleeping_groups": len(sleeping),
            "num_configs":     len(configs),
            "primary_used_us": primary_used,
            "budget_us":       budget_us,
        })
        return configs if configs else self.init_configs()

    def get_policy_snapshot(self) -> dict:
        return {
            "policy":          "traffic_aware",
            "num_groups":      len(self._groups) if self._groups else 0,
            "beacon_count":    self._beacon_count,
            "praw_groups":     sum(1 for v in self._praw_mode.values() if v),
            "alpha":           self.alpha,
            "skip_threshold":  self.skip_threshold,
            "repeat_threshold": self.repeat_threshold,
        }

    # =========================================================================
    # EWMA + CUSUM demand estimation
    # =========================================================================

    def _update_demands(self, aids: List[int]) -> None:
        for aid in aids:
            obs       = self._observe_sta_load(aid)
            prev_ewma = self._ewma.get(aid, obs)
            ewma      = self.alpha * obs + (1.0 - self.alpha) * prev_ewma
            err       = obs - prev_ewma
            cusum     = max(0.0, self._cusum.get(aid, 0.0) + err - self.cusum_k)
            burst     = max(0.0, cusum - self.cusum_h)
            self._ewma[aid]   = ewma
            self._cusum[aid]  = cusum
            self._demand[aid] = max(0.0, ewma + burst)

    def _observe_sta_load(self, aid: int) -> float:
        """TX-queue depth at the AP — best available per-STA load signal."""
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

    # =========================================================================
    # Group load helpers
    # =========================================================================

    def _group_load(self, s: int, e: int) -> float:
        return sum(self._demand.get(aid, 0.0) for aid in range(s, e + 1))

    def _group_size(self, s: int, e: int) -> int:
        return e - s + 1

    def _group_dpa(self, s: int, e: int) -> float:
        """Demand per AID — normalised load intensity."""
        return self._group_load(s, e) / max(1, self._group_size(s, e))

    # =========================================================================
    # A: demand-proportional budget allocation
    # =========================================================================

    def _allocate_budget(
        self,
        active_groups: List[Tuple[int, int]],
        budget_us: int,
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        """
        Return (slot_us_map, num_slots_map) for each active group.

        Algorithm — three stable steps:
        ──────────────────────────────
        1. Compute a uniform baseline slot count S that fits within the
           budget (same as traffic_split's _slots_for_group_count), keeping
           all groups on equal footing as a starting point.

        2. Demand-proportional redistribution: treat S×G as a total slot
           budget and redistribute among groups proportional to their EWMA
           demand share.  Groups with above-average demand get more slots;
           groups with below-average demand get fewer.  This is the core
           new feature — more chances per BI for busy groups.

        3. Per-group Bianchi slot duration: now that num_slots_i is known,
           compute T*_i using n_per_slot = ceil(D_gk / num_slots_i) for each
           group.  Higher slot counts → smaller n_per_slot → shorter T*,
           naturally compressing the slot duration for busier groups.

        Budget constraint: Σ(num_slots[g] × slot_us[g]) ≤ budget_us.
        """
        if not active_groups:
            return {}, {}

        G = len(active_groups)

        # ── Step 1: uniform baseline slot count ───────────────────────────────
        # Same formula as traffic_split: pick the largest S such that
        # G × S × min_slot_us ≤ budget_us.
        if G * self.min_slots * self.min_slot_us <= budget_us:
            base_slots = self.min_slots
            # Try to increase S while still fitting
            for s_try in range(self.min_slots, self.max_slots + 1):
                if G * s_try * self.min_slot_us <= budget_us:
                    base_slots = s_try
                else:
                    break
        else:
            base_slots = max(self.min_slots,
                             budget_us // (G * max(1, self.min_slot_us)))

        total_budget_slots = base_slots * G   # pool to redistribute

        # ── Step 2: demand-proportional redistribution ────────────────────────
        # Each group is guaranteed a floor of `floor_slots` per BI to prevent
        # starvation.  Only the excess budget (total − n×floor) is redistributed
        # demand-proportionally.  This preserves fairness for uniform traffic
        # while rewarding genuinely high-demand groups with bonus slots.
        #
        # floor_frac=0.5 means every group gets at least half the base; with
        # fully-uniform demand the entire bonus is split equally so all groups
        # still end up at base_slots.
        # floor_frac=1.0 means equal slots for all active groups (no starvation).
        # Set taw_slot_floor_frac < 1.0 only for intentional demand-proportional
        # redistribution in heterogeneous deployments.
        floor_frac  = float(self.ctx.cfg["mac"].get("taw_slot_floor_frac", 1.0))
        floor_slots = max(self.min_slots, int(math.floor(base_slots * floor_frac)))

        total_demand = sum(self._group_load(*g) for g in active_groups)
        slots_map: Dict[Tuple[int, int], int] = {}

        # Guaranteed floor for every group
        for g in active_groups:
            slots_map[g] = floor_slots

        # Bonus slots distributed demand-proportionally from the excess pool
        excess_pool = total_budget_slots - floor_slots * G
        if excess_pool > 0 and total_demand > 0.0:
            remaining = excess_pool
            sorted_groups = sorted(active_groups,
                                   key=lambda g: self._group_load(*g),
                                   reverse=True)
            for i, g in enumerate(sorted_groups):
                if i == len(sorted_groups) - 1:
                    bonus = max(0, remaining)
                else:
                    share = self._group_load(*g) / total_demand
                    bonus = int(round(share * excess_pool))
                    bonus = max(0, min(bonus, remaining))
                    remaining -= bonus
                    remaining  = max(0, remaining)
                n = max(self.min_slots, min(self.max_slots,
                                            floor_slots + bonus))
                slots_map[g] = n
        elif excess_pool > 0:
            # No demand data — spread excess equally
            per_group_bonus = excess_pool // G
            for g in active_groups:
                slots_map[g] = max(self.min_slots,
                                   min(self.max_slots,
                                       floor_slots + per_group_bonus))

        # ── Step 3: per-group Bianchi slot duration based on num_slots ────────
        slot_us_map: Dict[Tuple[int, int], int] = {}
        beta_s = self._l_s * self._sigma_s
        for g in active_groups:
            s, e   = g
            D_gk   = self._group_load(s, e)
            s_g    = slots_map[g]
            n_raw  = max(1, math.ceil(D_gk / max(1, s_g)))
            n_eff  = min(n_raw, self._group_size(s, e))
            t_star = self._laca_slot_s(n_eff, self._sigma_s, beta_s)
            slot_us_map[g] = self._quantise(int(math.ceil(t_star * 1e6)))

        # ── Budget enforcement ─────────────────────────────────────────────────
        # First try: reduce num_slots proportionally
        used = sum(slots_map[g] * slot_us_map[g] for g in active_groups)
        if used > budget_us:
            ratio = budget_us / used
            for g in active_groups:
                slots_map[g] = max(self.min_slots, int(slots_map[g] * ratio))

        # Second try: if individual slot durations are too large even at
        # min_slots=1, scale slot durations down proportionally
        used = sum(slots_map[g] * slot_us_map[g] for g in active_groups)
        if used > budget_us:
            scale = budget_us / used
            for g in active_groups:
                slot_us_map[g] = self._quantise(int(slot_us_map[g] * scale))

        return slot_us_map, slots_map

    # =========================================================================
    # C: PRAW state machine
    # =========================================================================

    def _advance_praw_phases(self) -> Set[Tuple[int, int]]:
        """
        Advance the PRAW phase counter for each PRAW-mode group.

        Returns the set of groups that are sleeping (omitted) this BI.
        Phase semantics:
            0              → active this BI; advance to (skip_period − 1) after emit
            1 … skip_period-1 → sleeping; decrement each BI
        """
        sleeping: Set[Tuple[int, int]] = set()
        for g in (self._groups or []):
            if not self._praw_mode.get(g, False):
                continue
            phase = self._praw_phase.get(g, 0)
            if phase == 0:
                # Active this BI → set next phase to start sleep countdown
                self._praw_phase[g] = self.skip_period - 1
            else:
                # Sleeping this BI
                sleeping.add(g)
                self._praw_phase[g] = phase - 1
        return sleeping

    def _update_praw_transitions(self, sleeping: Set[Tuple[int, int]]) -> None:
        """
        Enter PRAW (active → skip) and exit PRAW (skip → active) based on
        per-group demand-per-AID thresholds.

        Entry:  group is active-mode and dpa < skip_threshold for skip_confirm BIs.
                Disabled before beacon praw_warmup_beacons (EWMA not yet reliable).
        Exit:   group is PRAW-mode, on its active BI, and dpa ≥ wakeup_threshold.
        """
        praw_entry_allowed = self._beacon_count >= self.praw_warmup_beacons
        for g in (self._groups or []):
            dpa = self._group_dpa(*g)
            if not self._praw_mode.get(g, False):
                # Active mode → check for PRAW entry (only after warmup)
                if praw_entry_allowed and dpa < self.skip_threshold:
                    self._under_count[g] = self._under_count.get(g, 0) + 1
                else:
                    self._under_count[g] = 0
                if praw_entry_allowed and self._under_count.get(g, 0) >= self.skip_confirm:
                    self._praw_mode[g]   = True
                    self._praw_phase[g]  = 0   # active on next BI, then starts sleeping
                    self._under_count[g] = 0
                    self._over_count[g]  = 0
                    self._log("TAW_PRAW_ENTER", {
                        "beacon": self._beacon_count, "group": list(g),
                        "dpa": round(dpa, 5), "skip_period": self.skip_period,
                    })
            else:
                # PRAW mode → check for wakeup only on the group's active BI
                if g not in sleeping:
                    if dpa >= self.wakeup_threshold:
                        self._over_count[g] = self._over_count.get(g, 0) + 1
                    else:
                        self._over_count[g] = 0
                    if self._over_count.get(g, 0) >= 1:
                        # Immediate wakeup: restore to active-every-BI scheduling
                        self._praw_mode[g]  = False
                        self._praw_phase[g] = 0
                        self._over_count[g] = 0
                        self._under_count[g] = 0
                        self._log("TAW_PRAW_EXIT", {
                            "beacon": self._beacon_count, "group": list(g),
                            "dpa": round(dpa, 5),
                        })

    # =========================================================================
    # Split / merge (boundary adaptation)
    # =========================================================================

    def _update_split_merge_counters(self) -> None:
        for g in (self._groups or []):
            s, e = g
            dpa  = self._group_dpa(s, e)
            if dpa > self.split_threshold:
                self._overload_count[g]  = self._overload_count.get(g, 0) + 1
                self._underload_count[g] = 0
            elif dpa < self.merge_threshold:
                self._underload_count[g] = self._underload_count.get(g, 0) + 1
                self._overload_count[g]  = 0
            else:
                self._overload_count[g]  = 0
                self._underload_count[g] = 0

    def _try_split(self, groups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Split the single most-overloaded group that meets all guards."""
        max_g = self._effective_max_groups(groups)
        if len(groups) >= max_g:
            return groups

        best_idx = -1
        best_dpa = self.split_threshold

        for i, (s, e) in enumerate(groups):
            if e <= s:
                continue
            if self._overload_count.get((s, e), 0) < self.split_confirm:
                continue
            dpa = self._group_dpa(s, e)
            if dpa <= best_dpa:
                continue
            mid = self._load_median_split(s, e)
            left_n  = mid - s + 1
            right_n = e - mid
            if left_n < self.min_aids_per_group or right_n < self.min_aids_per_group:
                continue
            best_dpa = dpa
            best_idx = i

        if best_idx == -1:
            return groups

        s, e  = groups[best_idx]
        mid   = self._load_median_split(s, e)
        new_g = groups[:best_idx] + [(s, mid), (mid + 1, e)] + groups[best_idx + 1:]

        # Budget check: verify the new partition still fits within budget.
        # Use min_slot_us (T* for n=1) rather than the inflated backlog-driven
        # T* — the actual slot durations are always budget-enforced in
        # _allocate_budget, so this check only gates physical feasibility.
        budget_us = int(self._beacon_interval_us() * self.budget_fraction)
        if len(new_g) * self.min_slots * self.min_slot_us > budget_us:
            self._log("TAW_SPLIT_BUDGET_BLOCK", {
                "beacon": self._beacon_count, "proposed_groups": len(new_g),
            })
            return groups

        # Inherit PRAW state: new sub-groups start as active
        for g_new in [(s, mid), (mid + 1, e)]:
            self._praw_mode[g_new]   = False
            self._praw_phase[g_new]  = 0
            self._under_count[g_new] = 0
            self._over_count[g_new]  = 0

        self._log("TAW_SPLIT", {
            "beacon": self._beacon_count, "group_idx": best_idx,
            "start_aid": s, "end_aid": e, "mid": mid,
            "dpa": round(best_dpa, 3),
        })
        return new_g

    def _try_merge(self, groups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge the lightest qualifying adjacent pair."""
        if len(groups) <= self.min_groups:
            return groups

        best_idx  = -1
        best_comb = float("inf")

        for i in range(len(groups) - 1):
            s1, e1 = groups[i]
            s2, e2 = groups[i + 1]
            if s2 != e1 + 1:
                continue
            if (self._underload_count.get((s1, e1), 0) < self.merge_confirm
                    or self._underload_count.get((s2, e2), 0) < self.merge_confirm):
                continue
            combined = self._group_dpa(s1, e1) + self._group_dpa(s2, e2)
            if combined < best_comb:
                best_comb = combined
                best_idx  = i

        if best_idx == -1:
            return groups

        s1, e1  = groups[best_idx]
        _s2, e2 = groups[best_idx + 1]
        new_g   = groups[:best_idx] + [(s1, e2)] + groups[best_idx + 2:]

        # Clean up old group state; merged group starts fresh
        for old in [(s1, e1), (_s2, e2)]:
            for d in (self._overload_count, self._underload_count,
                      self._praw_mode, self._praw_phase,
                      self._under_count, self._over_count):
                d.pop(old, None)
        merged = (s1, e2)
        self._overload_count[merged]  = 0
        self._underload_count[merged] = 0
        self._praw_mode[merged]       = False
        self._praw_phase[merged]      = 0
        self._under_count[merged]     = 0
        self._over_count[merged]      = 0

        self._log("TAW_MERGE", {
            "beacon": self._beacon_count, "group_idx": best_idx,
            "merged_range": [s1, e2], "combined_dpa": round(best_comb, 3),
        })
        return new_g

    def _load_median_split(self, s: int, e: int) -> int:
        """Load-median split point; falls back to AID midpoint when no demand data."""
        total = self._group_load(s, e)
        if total <= 0.0:
            return (s + e) // 2
        half = total / 2.0
        cum  = 0.0
        for aid in range(s, e):
            cum += self._demand.get(aid, 0.0)
            if cum >= half:
                return aid
        return (s + e) // 2

    # =========================================================================
    # Bianchi / LACA slot sizing
    # =========================================================================

    def _laca_slot_s(self, n_s: int, sigma: float, beta: float) -> float:
        """
        T*_S for n_s stations — renewal-process delivery time.

          T*_S = Σ_{k=1}^{n_s} Z_k
          Z_k  = (1 / P_{k,s}) × (σ × P_{k,i}/(1−P_{k,i}) + β)
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
            p_ks = max(1e-12, p_succ_unc / (1.0 - p_idle))
            t_star += (1.0 / p_ks) * (sigma * p_idle / (1.0 - p_idle) + beta)
        return t_star

    def _bianchi_fixed_point(self, n: int) -> Tuple[float, float]:
        if n <= 1:
            W0  = self._cw_min + 1.0
            tau = min(0.5, 2.0 / (W0 + 1.0))
            return tau, 0.0
        W0 = self._cw_min + 1.0
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

    # =========================================================================
    # Slot duration utilities
    # =========================================================================

    def _quantise(self, slot_us: int) -> int:
        """Round UP to the nearest valid RAW slot-duration cslot grid point
        (500 + 120*cslot µs, per IEEE 802.11ah §9.4.2.200), clamp to [min, max].

        Rounding up onto this grid guarantees the provisioned slot duration
        is never shorter than the analytically required T* after raw.py's
        us_to_cslot()/cslot_to_us() round-trip (which floors onto the same
        grid). Rounding to plain multiples of 120 (the previous behaviour)
        is off-grid by 20 us and could under-provision airtime by up to
        120 us per slot once re-encoded.
        """
        q = -(-(int(slot_us) - MORSE_RAW_MIN_SLOT_DURATION_US) // _RAW_SLOT_UNIT_US) \
            * _RAW_SLOT_UNIT_US + MORSE_RAW_MIN_SLOT_DURATION_US
        return max(self.min_slot_us, min(self.max_slot_us, q))

    def _beacon_interval_us(self) -> int:
        try:
            bi_s = float(self.ctx.cfg["mac"].get("beacon_interval", 0.1024))
            return max(1, int(bi_s * 1_000_000))
        except Exception:
            return 102_400

    # =========================================================================
    # Budget / partition helpers
    # =========================================================================

    def _effective_max_groups(self, groups: List[Tuple[int, int]]) -> int:
        budget_us    = int(self._beacon_interval_us() * self.budget_fraction)
        g_max_budget = max(1, budget_us // (self.min_slots * self.min_slot_us))
        total_aids   = sum(e - s + 1 for s, e in groups) if groups else 1
        g_max_stas   = max(1, total_aids // self.min_aids_per_group)
        return min(g_max_budget, g_max_stas)

    def _partition_aids(self, aids: List[int], n_groups: int) -> List[Tuple[int, int]]:
        """Equal-width AID-range partition — used when EWMA is not yet warm."""
        aids     = sorted(aids)
        n        = len(aids)
        n_groups = max(1, min(n_groups, n))
        groups   = []
        for g in range(n_groups):
            lo = aids[g * n // n_groups]
            hi = aids[min((g + 1) * n // n_groups, n) - 1]
            groups.append((lo, hi))
        return groups

    # =========================================================================
    # RawConfig factory
    # =========================================================================

    def _make_config(
        self,
        cfg_id: int,
        start_aid: int,
        end_aid: int,
        start_time_us: int,
        slot_duration_us: int,
        num_slots: int,
    ) -> RawConfig:
        return RawConfig(
            id=cfg_id,
            raw_type=self.ctx.raw_type,
            start_aid=start_aid,
            end_aid=end_aid,
            start_time_us=start_time_us,
            slot_definition=RawSlotDefinition(
                num_slots=num_slots,
                slot_duration_us=slot_duration_us,
                cross_slot_boundary=self.ctx.raw_cross_slot,
            ),
            beacon_spreading=RawBeaconSpreading(),
            periodic=RawPeriodic(),
            enabled=True,
        )

    # =========================================================================
    # AID discovery
    # =========================================================================

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
