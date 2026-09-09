from __future__ import annotations

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


class ETaroaRawPolicy:
    """
    Enhanced TAROA (E-TAROA) -- Tian, Santi, Latre, Famaey, "Accurate Sensor
    Traffic Estimation for Station Grouping in Highly Dense IEEE 802.11ah
    Networks," ACM SenSys 2017.

    The paper's own contribution is entirely in *traffic estimation*
    (Section 3, Algorithm 1): each station's packet transmission interval
    t_int^s is estimated per DTIM beacon from (a) success/failure of the
    previous beacon's reception, (b) the "More Data" header field (does the
    station still have packets queued after this transmission?), and (c)
    whether a transmission spanned two beacon intervals (cross slot
    boundary, CSB). The RAW-parameter-optimization step that consumes this
    estimate (number of groups/slots, group duration, station assignment)
    is described as "nearly identical to" the original TAROA algorithm,
    which is a *different* paper (ref [14], Tian et al., Sensors 2017) not
    available here -- so that step is implemented using this simulator's
    own already-validated adaptive-RAW machinery (identical structure to
    raw_policy_adaptive.py: fixed group/slot count, Bianchi contention
    model, per-group required-slot-count from aggregate demand), fed by
    the E-TAROA-estimated per-station rate instead of CUSUM-EWMA queue
    length. This is a defensible, standard realization of "RAW parameters
    from estimated demand via contention modeling," consistent with how
    every other adaptive-family policy in this simulator does it, and it
    inherits that pattern's beacon-budget safety (fixed group x fixed slot
    count means only slot *duration* is bounded, not slot *count* -- so
    this cannot suffer the demand-differentiation-flattening bug found and
    fixed in raw_policy_cluster_adaptive.py's clamp, since there is no
    variable slot count to clamp).

    Faithfully implemented from Algorithm 1:
      - All four main cases (previous failed / previous success after a
        failure / exactly one packet this beacon / more than one packet
        this beacon), including the "More Data"-driven refinement in case
        3.1 (lines 11-14) -- this is the paper's headline mechanism and
        the reason dcf.py now actually sets the more_data ctrl field
        (previously always False in this simulator; see dcf.py's drive()).

    Deliberately NOT modeled (documented simplification, not an oversight):
      - c_succ (whether the last successful reception spanned two beacon
        intervals) is always treated as False. Detecting this precisely
        would require tracking each PDU's original transmission-attempt
        timestamp through retries/CSB-extended exchanges, which this
        simulator does not currently instrument. This only disables the
        paper's case-2 timing correction (line 6-7, "t_succ[0] -= 1"), a
        minor secondary correction -- the primary, heavily-evaluated
        contribution (the More-Data-driven interval refinement) is fully
        implemented and does not depend on c_succ.
      - Delta_m^s (number of transmissions since a station's queue was last
        observed non-empty) is not given an update rule in the paper's
        Algorithm 1 (it's listed as an input, not derived there); this
        implementation increments it once per successful more_data=False
        transmission and resets it whenever more_data flips True or a
        case-3.1 refinement consumes it -- the natural reading of the
        paper's prose description around Eq. (3)-(4).
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        self.mac_cfg = self.ctx.cfg["mac"]
        self.phy_cfg = self.ctx.cfg["phy"]
        self.net_cfg = self.ctx.cfg["net"]
        self.app_cfg = self.ctx.cfg["app"]

        # ------------------------------------------------------------------
        # Fixed RAW structure (same convention as raw_policy_adaptive.py)
        # ------------------------------------------------------------------
        self.fixed_num_groups = max(
            1, int(self.mac_cfg.get("raw_num_groups", getattr(self.ctx, "raw_num_groups", 1)))
        )
        self.fixed_num_slots = max(
            1, int(self.mac_cfg.get("raw_num_slots", getattr(self.ctx, "raw_num_slots", 1)))
        )

        # ------------------------------------------------------------------
        # Slot-duration bounds, quantization, beacon-interval time budget
        # ------------------------------------------------------------------
        base_slot_us = int(float(getattr(self.ctx, "raw_slot_duration", 0.007)) * 1e6)

        self.min_slot_us = int(
            self.mac_cfg.get(
                "etaroa_min_slot_us",
                max(MORSE_RAW_MIN_SLOT_DURATION_US, max(7000, base_slot_us // 2)),
            )
        )
        self.max_slot_us = int(
            self.mac_cfg.get("etaroa_max_slot_us", max(self.min_slot_us, max(base_slot_us * 2, 12000)))
        )
        self.min_slot_us = max(MORSE_RAW_MIN_SLOT_DURATION_US, self.min_slot_us)

        self.budget_fraction = float(self.mac_cfg.get("etaroa_budget_fraction", 0.90))
        beacon_interval_us = float(self.mac_cfg.get("beacon_interval", 0.5)) * 1e6
        self.beacon_interval_s = beacon_interval_us / 1e6
        total_slots = max(1, self.fixed_num_groups * self.fixed_num_slots)
        budget_max_slot_us = int(math.floor((beacon_interval_us * self.budget_fraction) / total_slots))
        self.max_slot_us = min(self.max_slot_us, max(MORSE_RAW_MIN_SLOT_DURATION_US, budget_max_slot_us))
        self.min_slot_us = min(self.min_slot_us, self.max_slot_us)

        self.initial_slot_us = int(self.mac_cfg.get("etaroa_initial_slot_us", base_slot_us))
        self.initial_slot_us = max(self.min_slot_us, min(self.max_slot_us, self.initial_slot_us))

        self.step_us = max(100, int(self.mac_cfg.get("etaroa_step_us", 1000)))
        self.smoothing_beta = float(self.mac_cfg.get("etaroa_smoothing_beta", 0.5))
        self.hysteresis = float(self.mac_cfg.get("etaroa_hysteresis", 0.0))

        self.bianchi_eps = float(self.mac_cfg.get("etaroa_bianchi_eps", 1e-4))
        self.bianchi_imax = int(self.mac_cfg.get("etaroa_bianchi_imax", 50))

        self.tmax_s = float(self.mac_cfg.get("etaroa_tmax_s", self.max_slot_us / 1e6))
        self.lth_slots = int(
            self.mac_cfg.get(
                "etaroa_lth_slots",
                max(1, int(round((0.75 * self.tmax_s) / self._sigma_s()))),
            )
        )

        # Seed estimate for a station never observed before: assume it
        # transmits once every this-many beacon intervals until evidence
        # (success/failure history) refines the estimate. 1.0 == "assume
        # saturated until proven otherwise", matching TAROA/E-TAROA's own
        # convention of starting from the densest plausible assumption.
        self.default_t_int = float(self.mac_cfg.get("etaroa_default_t_int_beacons", 1.0))

        # ------------------------------------------------------------------
        # Persistent state
        # ------------------------------------------------------------------
        self._current_slot_us: int = self.initial_slot_us
        self._update_count: int = 0
        self._estimator = EtaroaTrafficEstimator(self.ctx, default_t_int_beacons=self.default_t_int)
        self._last_group_slots: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init_configs(self) -> List["RawConfig"]:
        aids = self._effective_connected_aids([])
        max_aid = max(aids) if aids else max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))

        slot_us = max(MORSE_RAW_MIN_SLOT_DURATION_US, self._current_slot_us)
        return [
            RawConfig(
                id=1,
                raw_type=self.ctx.raw_type,
                start_aid=1,
                end_aid=max_aid,
                start_time_us=self.ctx.raw_start_time_us,
                slot_definition=RawSlotDefinition(
                    num_slots=self.fixed_num_slots,
                    slot_duration_us=slot_us,
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
            self._log("RAW_ETAROA_NO_AIDS", {})
            return []

        groups = self._partition_fixed_by_aid(aids, self.fixed_num_groups)
        if not groups:
            return self.init_configs()

        # --- E-TAROA's own contribution: per-station traffic estimation ---
        sta_rate = self._estimator.update(aids)  # aid -> packets-per-beacon-interval

        group_info: List[Dict[str, Any]] = []
        for gid, member_aids in enumerate(groups, start=1):
            Dgk = sum(sta_rate.get(aid, 0.0) for aid in member_aids)
            nact = sum(1 for aid in member_aids if sta_rate.get(aid, 0.0) > 0.0)
            group_info.append({"gid": gid, "member_aids": member_aids, "Dgk": float(Dgk), "nact": int(nact)})

        sigma = self._sigma_s()
        lmax_slots = max(1, int(math.floor(self.tmax_s / sigma)))
        raw_lengths_slots: Dict[int, int] = {}
        group_debug: Dict[int, Dict[str, Any]] = {}

        for info in group_info:
            gid, Dgk, nact = int(info["gid"]), float(info["Dgk"]), int(info["nact"])
            if Dgk <= 0.0 or nact <= 0:
                raw_lengths_slots[gid] = 0
                group_debug[gid] = {"tau": 0.0, "p_succ": 0.0, "l_bar": 0.0}
                continue

            tau, _ = self._solve_bianchi_fixed_point(nact)
            p_idle = (1.0 - tau) ** nact
            p_succ = nact * tau * ((1.0 - tau) ** (nact - 1))
            p_col = max(0.0, 1.0 - p_idle - p_succ)

            ts, tc = self._get_success_collision_times_s()
            l_s, l_c = ts / sigma, tc / sigma
            l_bar = p_idle * 1.0 + p_succ * l_s + p_col * l_c
            p_succ_eff = max(1e-9, p_succ)

            Lgk = int(math.ceil(Dgk * l_bar / p_succ_eff))
            Lgk = max(1, min(Lgk, lmax_slots))
            raw_lengths_slots[gid] = Lgk
            group_debug[gid] = {"tau": float(tau), "p_succ": float(p_succ), "l_bar": float(l_bar)}

        non_zero = [v for v in raw_lengths_slots.values() if v > 0]
        avg_Lg_slots = int(round(sum(non_zero) / len(non_zero))) if non_zero else 0
        avg_Lg_slots = min(avg_Lg_slots, lmax_slots)
        eta_g = 1 if avg_Lg_slots > self.lth_slots else 0

        configs: List[RawConfig] = []
        start_offset_us = int(self.ctx.raw_start_time_us)
        slot_values_us: List[int] = []
        cfg_id = 1

        for info in group_info:
            gid = int(info["gid"])
            member_aids = info["member_aids"]
            if not member_aids:
                continue

            if raw_lengths_slots[gid] <= 0:
                final_Lgk = 0
                slot_us = self._apply_safety_lower_bound(self.min_slot_us)
            else:
                final_Lgk = min(raw_lengths_slots[gid], avg_Lg_slots if avg_Lg_slots > 0 else lmax_slots)
                slot_us = int(math.ceil(final_Lgk * sigma * 1e6))
                slot_us = self._apply_safety_lower_bound(slot_us)

            slot_us = min(slot_us, self.max_slot_us)
            slot_us = self._quantize_up(slot_us)

            prev_slot = self._last_group_slots.get(gid, self._current_slot_us)
            if abs(slot_us - prev_slot) > self.hysteresis:
                slot_us = int(round(self.smoothing_beta * slot_us + (1.0 - self.smoothing_beta) * prev_slot))
                slot_us = self._quantize_up(slot_us)
            else:
                slot_us = prev_slot

            self._last_group_slots[gid] = slot_us
            slot_values_us.append(slot_us)

            configs.append(
                RawConfig(
                    id=cfg_id,
                    raw_type=self.ctx.raw_type,
                    start_aid=min(member_aids),
                    end_aid=max(member_aids),
                    start_time_us=start_offset_us,
                    slot_definition=RawSlotDefinition(
                        num_slots=self.fixed_num_slots,
                        slot_duration_us=slot_us,
                        cross_slot_boundary=self.ctx.raw_cross_slot,
                    ),
                    beacon_spreading=RawBeaconSpreading(),
                    periodic=RawPeriodic(),
                    enabled=True,
                )
            )
            cfg_id += 1
            start_offset_us += self.fixed_num_slots * slot_us

            dbg = group_debug.get(gid, {})
            self._log(
                "RAW_ETAROA_GROUP",
                {
                    "gid": gid, "start_aid": min(member_aids), "end_aid": max(member_aids),
                    "n_members": len(member_aids), "Dgk": float(info["Dgk"]), "nact": int(info["nact"]),
                    "Lgk_slots_final": int(final_Lgk), "slot_duration_us": int(slot_us),
                    "num_slots": int(self.fixed_num_slots), "eta_g": int(eta_g),
                    "tau": float(dbg.get("tau", 0.0)), "p_succ": float(dbg.get("p_succ", 0.0)),
                    "update_count": int(self._update_count),
                },
            )

        if slot_values_us:
            self._current_slot_us = self._quantize_up(int(round(sum(slot_values_us) / len(slot_values_us))))
        self._update_count += 1

        self._log(
            "RAW_ETAROA_BUILD",
            {
                "groups_built": len(configs), "stas_total": len(aids),
                "fixed_num_groups": self.fixed_num_groups, "fixed_num_slots": self.fixed_num_slots,
                "avg_slot_duration_us": int(self._current_slot_us), "eta_g": int(eta_g),
            },
        )
        return configs if configs else self.init_configs()

    def get_policy_snapshot(self) -> Dict[str, Any]:
        return {
            "policy": "etaroa_more_data_csb_traffic_estimation",
            "fixed_num_groups": self.fixed_num_groups,
            "fixed_num_slots": self.fixed_num_slots,
            "slot_duration_us": self._current_slot_us,
            "update_count": self._update_count,
        }

    # ------------------------------------------------------------------
    # Bianchi fixed-point (identical to raw_policy_adaptive.py)
    # ------------------------------------------------------------------
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
        p = 0.0
        for _ in range(max(1, self.bianchi_imax)):
            p = 1.0 - (1.0 - tau) ** (n_active - 1)
            denom = (1.0 - 2.0 * p) * (W0 + 1.0) + p * W0 * (1.0 - (2.0 * p) ** m)
            if abs(denom) < 1e-12:
                break
            tau_new = min(1.0, max(1e-9, (2.0 * (1.0 - 2.0 * p)) / denom))
            if abs(tau_new - tau) <= self.bianchi_eps:
                tau = tau_new
                break
            tau = tau_new

        p = 1.0 - (1.0 - tau) ** (n_active - 1)
        return tau, p

    # ------------------------------------------------------------------
    # MAC timing (identical to raw_policy_adaptive.py)
    # ------------------------------------------------------------------
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
        data_rate_bps = float(mode_table.get(str(self.phy_cfg.get("default_mode", "MCS0")), 150_000.0))
        ctrl_rate_bps = float(mode_table.get(str(self.phy_cfg.get("control_mode", "MCS0")), 150_000.0))

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
        worst_case_backoff_s = cw_min * sigma

        safe_min_us = int(math.ceil((ts + raw_guard + ack_guard + worst_case_backoff_s) * 1e6))
        safe_min_us = max(safe_min_us, 7000, self.min_slot_us, MORSE_RAW_MIN_SLOT_DURATION_US)
        return max(slot_us, safe_min_us)

    def _quantize_up(self, slot_us: int) -> int:
        step = max(1, self.step_us)
        q = int(math.ceil(slot_us / step)) * step
        q = -(-(q - MORSE_RAW_MIN_SLOT_DURATION_US) // 120) * 120 + MORSE_RAW_MIN_SLOT_DURATION_US
        q = max(self.min_slot_us, min(self.max_slot_us, q))
        return max(MORSE_RAW_MIN_SLOT_DURATION_US, q)

    # ------------------------------------------------------------------
    # AID helpers (identical to raw_policy_adaptive.py)
    # ------------------------------------------------------------------
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

    def _partition_fixed_by_aid(self, aids: List[int], num_groups: int) -> List[List[int]]:
        aids = sorted(int(a) for a in aids if int(a) > 0)
        if not aids:
            return []
        groups: List[List[int]] = [[] for _ in range(max(1, num_groups))]
        for idx, aid in enumerate(aids):
            g = min(len(groups) - 1, int(idx * len(groups) / max(1, len(aids))))
            groups[g].append(aid)
        return groups
