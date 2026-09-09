"""
Traffic-aware sensor grouping for IEEE 802.11ah RAW, adapted from:

  S.-Y. Chang, C.-Y. Lin, B.-S. Lin and Y.-S. Chen, "Traffic-Aware Sensor
  Grouping for IEEE 802.11ah Networks: Regression Based Analysis and
  Design," IEEE Trans. Mobile Computing, vol. 18, no. 3, pp. 674-687,
  Mar. 2019. DOI: 10.1109/TMC.2018.2840149

Mechanisms adapted from the paper
----------------------------------
1. Per-STA traffic demand (Eq. 1):
       D_i = L_i * N_i / r_i
   L_i = bits per packet, N_i = expected packets/beacon (EWMA of observed
   AP-side TX-queue depth), r_i = PHY data rate.  D_i is a channel-time
   demand (seconds of airtime needed per beacon interval).

2. Regression-based contention-success-probability model (Eq. 3-6):
       r_delta(N) = b1 * ln(N) + b2
   fit between two bounds for a group given `delta` RAW sub-slots:
     - P_delta(1)   : single-station bound (Eq. 3 analogue)
     - P_delta(N*)  : saturated bound at N* = 50 contenders (Eq. 4 analogue)
   Both bounds are evaluated with the Bianchi (1) fixed-point model
   (the same model used by raw_policy_traffic_aware / raw_policy_laca),
   since the closed forms in Liu et al. / Zhang et al. cited by the paper
   require superframe-timing parameters not modelled in this simulator.
   Heterogeneous group demand is folded in via the Eq. 6 weighted average:
       P^succ(delta, group) = sum_i w_i * r_delta(N_i),  w_i = D_i / sum D
   where N_i is STA i's individual contender-count estimate.

3. Greedy traffic-aware grouping (Algorithm 1):
   IEEE 802.11ah RAW groups must be contiguous AID ranges (§9.22.3), so
   Algorithm 1's "assign sensor to worst group" step is adapted to a
   boundary-shift hill-climb: each beacon, the group with the lowest
   channel utilisation P^succ steals one boundary AID from its higher-
   utilisation neighbour, provided the move improves the global minimum
   utilisation (max-min fairness across groups).

4. Demand-proportional slot allocation (Eq. 1/6 weighting):
   Each group's RAW budget share (number of sub-slots) is proportional to
   its aggregate demand sum_i D_i; per-slot duration is then sized with the
   same Bianchi/LACA renewal-process T* used elsewhere in this codebase,
   which implicitly reserves protocol-overhead/backoff airtime.
"""

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
)

_RAW_SLOT_UNIT_US    = 120
_RAW_SLOT_CSLOT_MAX  = 2047
_RAW_SLOT_STD_MAX_US = cslot_to_us(_RAW_SLOT_CSLOT_MAX)

# Saturation reference point for the regression bounds (Eq. 4 analogue).
_N_STAR = 50


class ChangGroupingRawPolicy:
    """
    IEEE 802.11ah RAW policy: regression-based traffic-aware sensor grouping
    (Chang et al., IEEE TMC 2019).
    """

    _DEFAULT_IDEAL_GROUP_SIZE = 25

    # -----------------------------------------------------------------------
    def __init__(self, ctx, log_fn) -> None:
        self.ctx  = ctx
        self._log = log_fn
        mac_cfg = ctx.cfg["mac"]
        phy_cfg = ctx.cfg["phy"]
        app_cfg = ctx.cfg["app"]
        net_cfg = ctx.cfg["net"]

        # ── EWMA demand ──────────────────────────────────────────────────────
        self.alpha = float(mac_cfg.get("chang_ewma_alpha", 0.3))

        # ── Group structure ─────────────────────────────────────────────────
        self.ideal_group_size   = max(1, int(mac_cfg.get("chang_ideal_group_size",
                                                          self._DEFAULT_IDEAL_GROUP_SIZE)))
        self.min_aids_per_group = max(2, int(mac_cfg.get("chang_min_aids_per_group", 5)))
        self.min_groups         = max(1, int(mac_cfg.get("chang_min_groups", 1)))

        # ── Greedy boundary-shift rebalancing (Algorithm 1 analogue) ────────
        self.max_shifts_per_beacon = int(mac_cfg.get("chang_max_shifts_per_beacon", 4))

        # ── Budget ───────────────────────────────────────────────────────────
        self.budget_fraction = float(mac_cfg.get("chang_budget_fraction", 0.90))
        self.min_slots       = max(1, int(mac_cfg.get("chang_min_slots", 1)))
        self.max_slots       = max(1, int(mac_cfg.get("chang_max_slots", 16)))
        self.target_slots    = max(self.min_slots,
                                    int(mac_cfg.get("raw_num_slots",
                                                     mac_cfg.get("chang_target_slots", 4))))

        # ── Bianchi / timing parameters (shared with traffic_aware / laca) ──
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
        self._t_succ = (self._preamble + self._header + data_bits / max(1.0, self._data_rate_bps)
                        + self._sifs
                        + self._preamble + self._header + ack_bits / max(1.0, self._data_rate_bps)
                        + self._difs)
        self._l_s    = self._t_succ / self._sigma_s

        # Per-packet channel-time numerator for Eq. 1: D_i = L_i * N_i / r_i.
        # L_i is taken as the full over-the-air frame (payload + overhead +
        # preamble/header), expressed in seconds, matching _t_succ above.
        self._li_over_ri_s = self._t_succ

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
            int(mac_cfg.get("chang_min_slot_us", min_exchange_us)),
        )
        self.max_slot_us = min(
            _RAW_SLOT_STD_MAX_US,
            int(mac_cfg.get("chang_max_slot_us", _RAW_SLOT_STD_MAX_US)),
        )
        self.max_slot_us = max(self.min_slot_us, self.max_slot_us)

        # ── Per-STA demand state ────────────────────────────────────────────
        self._ewma:   Dict[int, float] = {}   # N_i: EWMA packets/BI
        self._demand: Dict[int, float] = {}   # D_i = L_i * N_i / r_i (seconds)

        # ── Group partition (contiguous AID ranges) ────────────────────────
        self._groups: Optional[List[Tuple[int, int]]] = None

        # ── Regression coefficients (Eq. 5), recomputed each beacon ────────
        self._b1: float = 0.0
        self._b2: float = 1.0

        self._beacon_count: int = 0

    # =========================================================================
    # Public API (called by RawEngine)
    # =========================================================================

    def init_configs(self) -> List[RawConfig]:
        aids    = self._effective_aids([])
        n_stas  = len(aids)
        max_aid = max(aids) if aids else max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))
        budget_us = int(self._beacon_interval_us() * self.budget_fraction)
        g_max_target = max(1, budget_us // (self.target_slots * self.min_slot_us))
        g_max_stas   = max(1, n_stas // self.min_aids_per_group) if n_stas > 0 else 1
        g_max        = min(g_max_target, g_max_stas)
        n_groups = max(self.min_groups,
                       min(g_max, max(1, math.ceil(n_stas / self.ideal_group_size))))

        self._groups = (
            self._partition_aids(aids, n_groups) if aids
            else [(1, max_aid)]
        )

        # Equal slot allocation at init (no demand data yet)
        equal_slots = max(self.min_slots, min(
            self.max_slots,
            budget_us // (len(self._groups) * max(1, self.min_slot_us)),
        ))
        slot_us = self._quantise(budget_us // max(1, len(self._groups) * equal_slots))
        configs   = []
        offset_us = int(self.ctx.raw_start_time_us)
        for cfg_id, (s, e) in enumerate(self._groups, start=1):
            configs.append(self._make_config(cfg_id, s, e, offset_us, slot_us, equal_slots))
            offset_us += equal_slots * slot_us
        return configs

    def build_dynamic_configs(self, connected_aids: List[int]) -> List[RawConfig]:
        aids = self._effective_aids(connected_aids)
        if not aids:
            self._log("CHANG_NO_AIDS", {})
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

        self._beacon_count += 1

        # ── Step 1: update per-STA traffic demand D_i (Eq. 1) ─────────────────
        self._update_demands(aids)

        # ── Step 2: refit the regression bounds for the current sub-slot
        #            allocation (Eq. 3-5) ─────────────────────────────────────
        budget_us = int(self._beacon_interval_us() * self.budget_fraction)
        avg_delta = max(1, budget_us // (len(self._groups) * max(1, self.min_slot_us)))
        avg_delta = min(self.max_slots, avg_delta)
        self._b1, self._b2 = self._fit_regression(avg_delta)

        # ── Step 3: greedy boundary-shift rebalancing (Algorithm 1 analogue) ──
        for _ in range(self.max_shifts_per_beacon):
            new_groups = self._rebalance_step(self._groups, avg_delta)
            if new_groups is self._groups:
                break
            self._groups = new_groups

        # ── Step 4: demand-proportional sub-slot allocation (Eq. 1/6) ─────────
        slot_us_map, slots_map = self._allocate_budget(self._groups, budget_us)

        # ── Step 5: build RAW configs ──────────────────────────────────────────
        configs:  List[RawConfig] = []
        offset_us = int(self.ctx.raw_start_time_us)
        for cfg_id, g in enumerate(self._groups, start=1):
            s, e    = g
            slot_us = slot_us_map[g]
            n_slots = slots_map[g]
            self._log("CHANG_GROUP", {
                "beacon":    self._beacon_count,
                "cfg_id":    cfg_id,
                "start_aid": s,
                "end_aid":   e,
                "num_slots": n_slots,
                "slot_us":   slot_us,
                "demand_s":  round(self._group_demand(s, e), 6),
                "p_succ":    round(self._group_p_succ(s, e, n_slots), 4),
            })
            configs.append(self._make_config(cfg_id, s, e, offset_us, slot_us, n_slots))
            offset_us += n_slots * slot_us

        self._log("CHANG_BUILD", {
            "beacon":      self._beacon_count,
            "num_groups":  len(self._groups),
            "b1":          round(self._b1, 5),
            "b2":          round(self._b2, 5),
            "avg_delta":   avg_delta,
        })
        return configs if configs else self.init_configs()

    def get_policy_snapshot(self) -> dict:
        return {
            "policy":       "chang2019_regression_grouping",
            "num_groups":   len(self._groups) if self._groups else 0,
            "beacon_count": self._beacon_count,
            "b1":           self._b1,
            "b2":           self._b2,
        }

    # =========================================================================
    # Eq. 1: per-STA traffic demand D_i = L_i * N_i / r_i
    # =========================================================================

    def _update_demands(self, aids: List[int]) -> None:
        for aid in aids:
            obs  = self._observe_sta_load(aid)
            prev = self._ewma.get(aid, obs)
            ewma = self.alpha * obs + (1.0 - self.alpha) * prev
            self._ewma[aid]   = ewma
            # D_i = L_i/r_i (seconds of airtime per packet) * N_i (packets/BI)
            self._demand[aid] = self._li_over_ri_s * ewma

    def _observe_sta_load(self, aid: int) -> float:
        """TX-queue depth at the AP — proxy for N_i (packets/BI)."""
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
    # Group demand / utilisation helpers
    # =========================================================================

    def _group_demand(self, s: int, e: int) -> float:
        return sum(self._demand.get(aid, 0.0) for aid in range(s, e + 1))

    def _group_size(self, s: int, e: int) -> int:
        return e - s + 1

    def _group_n_active(self, s: int, e: int) -> float:
        """Estimated total contenders N (sum of EWMA packets/BI), clipped to
        the group's STA count."""
        n = sum(self._ewma.get(aid, 0.0) for aid in range(s, e + 1))
        return max(0.0, min(n, float(self._group_size(s, e))))

    def _group_p_succ(self, s: int, e: int, delta: int) -> float:
        """
        Eq. 6 analogue: demand-weighted average of the per-STA regression
        success probability r_delta(N_i), where N_i is each STA's individual
        contention level (clipped to >=1 contender if it has any demand).
        """
        members = [aid for aid in range(s, e + 1)]
        weights = [self._demand.get(aid, 0.0) for aid in members]
        total_w = sum(weights)
        if total_w <= 0.0:
            n_group = self._group_n_active(s, e)
            return self._regression(self._b1, self._b2, max(1.0, n_group))

        acc = 0.0
        for aid, w in zip(members, weights):
            n_i = max(1.0, self._ewma.get(aid, 0.0))
            acc += (w / total_w) * self._regression(self._b1, self._b2, n_i)
        return acc

    @staticmethod
    def _regression(b1: float, b2: float, n: float) -> float:
        n = max(1.0, n)
        return max(0.0, min(1.0, b1 * math.log(n) + b2))

    # =========================================================================
    # Eq. 3-5: regression bounds P_delta(1), P_delta(N*) -> (b1, b2)
    # =========================================================================

    def _fit_regression(self, delta: int) -> Tuple[float, float]:
        delta = max(1, int(delta))
        p1     = self._bound_p_succ(1, delta)
        p_star = self._bound_p_succ(_N_STAR, delta)
        b2 = p1
        b1 = (p_star - p1) / math.log(_N_STAR)
        return b1, b2

    def _bound_p_succ(self, n: int, delta: int) -> float:
        """
        Per-sub-slot success probability for `n` contenders sharing `delta`
        RAW sub-slots (round-robin assignment -> n_per_slot = ceil(n/delta)),
        via the Bianchi (1) fixed point.  This is the shared building block
        for both the P_delta(1) and P_delta(N*) bounds in Eq. 3/4.
        """
        n_per_slot = max(1, math.ceil(n / max(1, delta)))
        tau, _ = self._bianchi_fixed_point(n_per_slot)
        if n_per_slot == 1:
            return 1.0
        p_idle = (1.0 - tau) ** n_per_slot
        if 1.0 - p_idle < 1e-12:
            return 0.0
        p_succ_unc = n_per_slot * tau * (1.0 - tau) ** (n_per_slot - 1)
        return max(0.0, min(1.0, p_succ_unc / (1.0 - p_idle)))

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
            p     = 1.0 - (1.0 - tau) ** (n - 1)
            denom = (1.0 - 2.0 * p) * (W0 + 1.0) + p * W0 * (1.0 - (2.0 * p) ** m)
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
    # Algorithm 1 analogue: greedy max-min-fairness boundary rebalancing
    # =========================================================================

    def _rebalance_step(
        self,
        groups: List[Tuple[int, int]],
        delta: int,
    ) -> List[Tuple[int, int]]:
        """
        One greedy step: find the group with the lowest channel utilisation
        P^succ (the "worst" group per Algorithm 1) and, if it has a neighbour
        with strictly higher utilisation, move one boundary AID from that
        neighbour to the worst group. The move is kept only if it raises the
        global minimum utilisation (max-min fairness improvement); otherwise
        the original partition is returned unchanged.
        """
        if len(groups) < 2:
            return groups

        utils = [self._group_p_succ(s, e, delta) for (s, e) in groups]
        worst_idx = min(range(len(groups)), key=lambda i: utils[i])

        candidates = []
        if worst_idx > 0:
            candidates.append(worst_idx - 1)
        if worst_idx < len(groups) - 1:
            candidates.append(worst_idx + 1)
        if not candidates:
            return groups

        # Prefer the neighbour with the higher utilisation (most "spare" capacity)
        neigh_idx = max(candidates, key=lambda i: utils[i])
        if utils[neigh_idx] <= utils[worst_idx]:
            return groups

        ws, we = groups[worst_idx]
        ns, ne = groups[neigh_idx]

        if neigh_idx == worst_idx - 1:
            # neighbour is to the left: give worst group neighbour's last AID
            if self._group_size(ns, ne) <= self.min_aids_per_group:
                return groups
            new_worst  = (ne, we)
            new_neigh  = (ns, ne - 1)
        else:
            # neighbour is to the right: give worst group neighbour's first AID
            if self._group_size(ns, ne) <= self.min_aids_per_group:
                return groups
            new_worst = (ws, ns)
            new_neigh = (ns + 1, ne)

        trial = list(groups)
        trial[worst_idx] = new_worst
        trial[neigh_idx] = new_neigh

        new_utils = [self._group_p_succ(s, e, delta) for (s, e) in trial]
        if min(new_utils) > min(utils) + 1e-9:
            self._log("CHANG_REBALANCE", {
                "beacon": self._beacon_count,
                "from_group": list(groups[neigh_idx]),
                "to_group":   list(groups[worst_idx]),
                "min_util_before": round(min(utils), 4),
                "min_util_after":  round(min(new_utils), 4),
            })
            return trial
        return groups

    # =========================================================================
    # Eq. 1/6-weighted demand-proportional sub-slot allocation
    # =========================================================================

    def _allocate_budget(
        self,
        groups: List[Tuple[int, int]],
        budget_us: int,
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        if not groups:
            return {}, {}

        G = len(groups)

        # Uniform baseline slot count S that fits within the budget
        if G * self.min_slots * self.min_slot_us <= budget_us:
            base_slots = self.min_slots
            for s_try in range(self.min_slots, self.max_slots + 1):
                if G * s_try * self.min_slot_us <= budget_us:
                    base_slots = s_try
                else:
                    break
        else:
            base_slots = max(self.min_slots, budget_us // (G * max(1, self.min_slot_us)))

        total_budget_slots = base_slots * G

        total_demand = sum(self._group_demand(s, e) for (s, e) in groups)

        # Every group is guaranteed a floor of min_slots; the slack between
        # the uniform baseline and min_slots is redistributed demand-
        # proportionally (Eq. 1/6 weighting) so higher-demand groups earn
        # more sub-slots.
        floor_slots = self.min_slots
        slots_map: Dict[Tuple[int, int], int] = {g: floor_slots for g in groups}

        excess_pool = total_budget_slots - floor_slots * G
        if excess_pool > 0 and total_demand > 0.0:
            remaining = excess_pool
            sorted_groups = sorted(groups, key=lambda g: self._group_demand(*g), reverse=True)
            for i, g in enumerate(sorted_groups):
                if i == len(sorted_groups) - 1:
                    bonus = max(0, remaining)
                else:
                    share = self._group_demand(*g) / total_demand
                    bonus = int(round(share * excess_pool))
                    bonus = max(0, min(bonus, remaining))
                    remaining -= bonus
                slots_map[g] = max(self.min_slots, min(self.max_slots, floor_slots + bonus))
        elif excess_pool > 0:
            # No demand data yet — spread the slack equally
            per_group_bonus = excess_pool // G
            for g in groups:
                slots_map[g] = max(self.min_slots,
                                    min(self.max_slots, floor_slots + per_group_bonus))

        # Per-group Bianchi/LACA renewal slot duration
        slot_us_map: Dict[Tuple[int, int], int] = {}
        beta_s = self._l_s * self._sigma_s
        for g in groups:
            s, e  = g
            n_act = self._group_n_active(s, e)
            s_g   = slots_map[g]
            n_raw = max(1, math.ceil(n_act / max(1, s_g)))
            n_eff = min(n_raw, self._group_size(s, e))
            t_star = self._laca_slot_s(n_eff, self._sigma_s, beta_s)
            slot_us_map[g] = self._quantise(int(math.ceil(t_star * 1e6)))

        # Budget enforcement (mirrors raw_policy_traffic_aware)
        used = sum(slots_map[g] * slot_us_map[g] for g in groups)
        if used > budget_us:
            ratio = budget_us / used
            for g in groups:
                slots_map[g] = max(self.min_slots, int(slots_map[g] * ratio))

        used = sum(slots_map[g] * slot_us_map[g] for g in groups)
        if used > budget_us:
            scale = budget_us / used
            for g in groups:
                slot_us_map[g] = self._quantise(int(slot_us_map[g] * scale))

        return slot_us_map, slots_map

    # =========================================================================
    # Bianchi / LACA slot sizing (renewal process T*_S)
    # =========================================================================

    def _laca_slot_s(self, n_s: int, sigma: float, beta: float) -> float:
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

    # =========================================================================
    # Slot duration / budget / partition utilities
    # =========================================================================

    def _quantise(self, slot_us: int) -> int:
        """Round UP to the nearest valid RAW slot-duration cslot grid point
        (500 + 120*cslot us, per IEEE 802.11ah §9.4.2.200), clamp to [min, max]."""
        q = -(-(int(slot_us) - MORSE_RAW_MIN_SLOT_DURATION_US) // _RAW_SLOT_UNIT_US) \
            * _RAW_SLOT_UNIT_US + MORSE_RAW_MIN_SLOT_DURATION_US
        return max(self.min_slot_us, min(self.max_slot_us, q))

    def _beacon_interval_us(self) -> int:
        try:
            bi_s = float(self.ctx.cfg["mac"].get("beacon_interval", 0.1024))
            return max(1, int(bi_s * 1_000_000))
        except Exception:
            return 102_400

    def _partition_aids(self, aids: List[int], n_groups: int) -> List[Tuple[int, int]]:
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
