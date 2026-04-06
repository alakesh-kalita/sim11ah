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


class AdaptiveRawPolicy:
    """
    Adaptive RAW slot-duration policy based on:

      1) per-STA CUSUM-EWMA demand prediction
      2) Bianchi fixed-point contention estimate
      3) RAW slot sizing from expected service rate

    Fixed:
    - number of RAW groups
    - AID partitioning across groups
    - number of slots per group

    Adapted:
    - slot_duration_us only
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        self.mac_cfg = self.ctx.cfg["mac"]
        self.phy_cfg = self.ctx.cfg["phy"]
        self.net_cfg = self.ctx.cfg["net"]
        self.app_cfg = self.ctx.cfg["app"]

        # ------------------------------------------------------------------
        # Fixed RAW structure
        # ------------------------------------------------------------------
        self.fixed_num_groups = max(
            1, int(self.mac_cfg.get("raw_num_groups", getattr(self.ctx, "raw_num_groups", 1)))
        )
        self.fixed_num_slots = max(
            1, int(self.mac_cfg.get("raw_num_slots", getattr(self.ctx, "raw_num_slots", 1)))
        )

        # ------------------------------------------------------------------
        # Slot-duration bounds and quantization
        # ------------------------------------------------------------------
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
                max(self.min_slot_us, max(base_slot_us * 2, 12000)),
            )
        )
        self.min_slot_us = max(MORSE_RAW_MIN_SLOT_DURATION_US, self.min_slot_us)

        self.initial_slot_us = int(self.mac_cfg.get("adaptive_raw_initial_slot_us", base_slot_us))
        self.initial_slot_us = max(self.min_slot_us, min(self.max_slot_us, self.initial_slot_us))

        self.step_us = int(self.mac_cfg.get("adaptive_raw_step_us", 1000))
        self.step_us = max(100, self.step_us)

        self.smoothing_beta = float(self.mac_cfg.get("adaptive_raw_smoothing_beta", 0.5))
        self.hysteresis = float(self.mac_cfg.get("adaptive_raw_hysteresis", 0.0))

        # ------------------------------------------------------------------
        # Predictor / Bianchi parameters
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Persistent predictor state
        # ------------------------------------------------------------------
        self._current_slot_us: int = self.initial_slot_us
        self._update_count: int = 0

        self._sta_ewma: Dict[int, float] = {}
        self._sta_cusum: Dict[int, float] = {}

        self._last_group_slots: Dict[int, int] = {}
        self._last_extra_raw_flag: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init_configs(self) -> List["RawConfig"]:
        aids = self._effective_connected_aids([])
        if not aids:
            max_aid = max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))
        else:
            max_aid = max(aids)

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
            self._log("RAW_ADAPT_NO_AIDS", {})
            return []

        groups = self._partition_fixed_by_aid(aids, self.fixed_num_groups)
        if not groups:
            return self.init_configs()

        # Step 1: Per-STA predicted demand d_i using CUSUM-EWMA
        sta_pred = self._predict_sta_demands(aids)

        # Per-group D_gk and n_act_gk
        group_info: List[Dict[str, Any]] = []
        for gid, member_aids in enumerate(groups, start=1):
            Dgk = sum(sta_pred.get(aid, 0.0) for aid in member_aids)
            nact = sum(1 for aid in member_aids if sta_pred.get(aid, 0.0) > 0.0)
            group_info.append(
                {
                    "gid": gid,
                    "member_aids": member_aids,
                    "Dgk": float(Dgk),
                    "nact": int(nact),
                }
            )

        raw_lengths_slots: Dict[int, int] = {}
        group_debug: Dict[int, Dict[str, Any]] = {}

        sigma = self._sigma_s()
        lmax_slots = max(1, int(math.floor(self.tmax_s / sigma)))

        for info in group_info:
            gid = int(info["gid"])
            Dgk = float(info["Dgk"])
            nact = int(info["nact"])

            if Dgk <= 0.0 or nact <= 0:
                raw_lengths_slots[gid] = 0
                group_debug[gid] = {
                    "tau": 0.0,
                    "p": 0.0,
                    "p_idle": 1.0,
                    "p_succ": 0.0,
                    "p_col": 0.0,
                    "l_s": 0.0,
                    "l_c": 0.0,
                    "l_bar": 0.0,
                }
                continue

            tau, p = self._solve_bianchi_fixed_point(nact)

            p_idle = (1.0 - tau) ** nact
            p_succ = nact * tau * ((1.0 - tau) ** (nact - 1))
            p_col = max(0.0, 1.0 - p_idle - p_succ)

            ts, tc = self._get_success_collision_times_s()
            l_s = ts / sigma
            l_c = tc / sigma
            l_bar = p_idle * 1.0 + p_succ * l_s + p_col * l_c

            p_succ_eff = max(1e-9, p_succ)

            # L_gk = ceil( D_gk * l_bar / P_succ )
            Lgk = int(math.ceil(Dgk * l_bar / p_succ_eff))
            Lgk = max(1, min(Lgk, lmax_slots))

            raw_lengths_slots[gid] = Lgk
            group_debug[gid] = {
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
        if non_zero_lengths:
            avg_Lg_slots = int(round(sum(non_zero_lengths) / len(non_zero_lengths)))
        else:
            avg_Lg_slots = 0

        avg_Lg_slots = min(avg_Lg_slots, lmax_slots)
        eta_g = 1 if avg_Lg_slots > self.lth_slots else 0

        configs: List[RawConfig] = []
        start_offset_us = int(self.ctx.raw_start_time_us)
        cfg_id = 1
        slot_values_us: List[int] = []

        for info in group_info:
            gid = int(info["gid"])
            member_aids = info["member_aids"]
            if not member_aids:
                continue

            if raw_lengths_slots[gid] <= 0:
                final_Lgk_slots = 0
                slot_us = self._apply_safety_lower_bound(self.min_slot_us)
                slot_us = self._quantize_up(slot_us)
            else:
                final_Lgk_slots = min(
                    raw_lengths_slots[gid],
                    avg_Lg_slots if avg_Lg_slots > 0 else lmax_slots,
                )
                slot_us = int(math.ceil(final_Lgk_slots * sigma * 1e6))
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
            self._last_extra_raw_flag[gid] = eta_g
            slot_values_us.append(slot_us)

            cfg = RawConfig(
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
            configs.append(cfg)
            cfg_id += 1

            start_offset_us += self.fixed_num_slots * slot_us

            dbg = group_debug.get(gid, {})
            self._log(
                "RAW_ADAPT_GROUP",
                {
                    "gid": gid,
                    "start_aid": min(member_aids),
                    "end_aid": max(member_aids),
                    "n_members": len(member_aids),
                    "Dgk": float(info["Dgk"]),
                    "nact": int(info["nact"]),
                    "Lgk_slots_raw": int(raw_lengths_slots.get(gid, 0)),
                    "Lg_bar_slots": int(avg_Lg_slots),
                    "Lgk_slots_final": int(final_Lgk_slots),
                    "slot_duration_us": int(slot_us),
                    "num_slots": int(self.fixed_num_slots),
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
            avg_slot_us = int(round(sum(slot_values_us) / len(slot_values_us)))
            self._current_slot_us = self._quantize_up(avg_slot_us)

        self._update_count += 1

        self._log(
            "RAW_ADAPT_BUILD",
            {
                "groups_built": len(configs),
                "stas_total": len(aids),
                "fixed_num_groups": self.fixed_num_groups,
                "fixed_num_slots": self.fixed_num_slots,
                "avg_slot_duration_us": int(self._current_slot_us),
                "group_slot_us_values": [int(x) for x in slot_values_us],
                "avg_Lg_slots": int(avg_Lg_slots),
                "Lmax_slots": int(lmax_slots),
                "eta_g": int(eta_g),
                "total_duration_us": start_offset_us - int(self.ctx.raw_start_time_us),
            },
        )

        return configs if configs else self.init_configs()

    def get_policy_snapshot(self) -> Dict[str, Any]:
        return {
            "policy": "adaptive_slot_duration_cusum_ewma_bianchi",
            "fixed_num_groups": self.fixed_num_groups,
            "fixed_num_slots": self.fixed_num_slots,
            "slot_duration_us": self._current_slot_us,
            "step_us": self.step_us,
            "update_count": self._update_count,
            "ewma_alpha": self.ewma_alpha,
            "cusum_k": self.cusum_k,
            "cusum_h": self.cusum_h,
            "tmax_s": self.tmax_s,
            "lth_slots": self.lth_slots,
        }

    # ------------------------------------------------------------------
    # Step 1: CUSUM-EWMA prediction
    # ------------------------------------------------------------------
    def _predict_sta_demands(self, aids: List[int]) -> Dict[int, float]:
        pred: Dict[int, float] = {}

        for aid in aids:
            observed = self._estimate_sta_observed_load(aid)

            prev_ewma = self._sta_ewma.get(aid, observed)
            ewma = self.ewma_alpha * observed + (1.0 - self.ewma_alpha) * prev_ewma

            err = observed - prev_ewma
            prev_cusum = self._sta_cusum.get(aid, 0.0)
            cusum = max(0.0, prev_cusum + err - self.cusum_k)

            burst_boost = max(0.0, cusum - self.cusum_h)
            di = max(0.0, ewma + burst_boost)

            self._sta_ewma[aid] = ewma
            self._sta_cusum[aid] = cusum
            pred[aid] = di

        return pred

    def _estimate_sta_observed_load(self, aid: int) -> float:
        try:
            sim_nodes = getattr(self.ctx.sim, "nodes", {})
            node = sim_nodes.get(aid)
            if node is not None:
                mac = getattr(node, "mac", None)
                if mac is not None:
                    for attr in ("tx_queue", "_tx_queue", "txq", "_txq"):
                        q = getattr(mac, attr, None)
                        if q is None:
                            continue
                        try:
                            return float(len(q))
                        except Exception:
                            pass

                app = getattr(node, "app", None)
                if app is not None:
                    for attr in ("pending", "_pending", "queue", "_queue"):
                        q = getattr(app, attr, None)
                        if q is None:
                            continue
                        try:
                            return float(len(q))
                        except Exception:
                            pass
        except Exception:
            pass

        total_pending = self._estimate_total_pending_packets()
        return 0.25 if total_pending > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Step 2: Bianchi fixed-point
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

            tau_new = (2.0 * (1.0 - 2.0 * p)) / denom
            tau_new = min(1.0, max(1e-9, tau_new))

            if abs(tau_new - tau) <= self.bianchi_eps:
                tau = tau_new
                break

            tau = tau_new

        p = 1.0 - (1.0 - tau) ** (n_active - 1)
        return tau, p

    # ------------------------------------------------------------------
    # MAC timing
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
        data_mode = str(self.phy_cfg.get("default_mode", "MCS0"))
        ctrl_mode = str(self.phy_cfg.get("control_mode", "MCS0"))

        data_rate_bps = float(mode_table.get(data_mode, 300_000.0))
        ctrl_rate_bps = float(mode_table.get(ctrl_mode, 300_000.0))

        data_bits = 8.0 * (payload_bytes + data_mac_oh + net_oh)
        ack_bits = 8.0 * ack_size

        t_data = preamble + header + (data_bits / max(1.0, data_rate_bps))
        t_ack = preamble + header + (ack_bits / max(1.0, ctrl_rate_bps))

        ts = t_data + sifs + t_ack + difs
        tc = t_data + ack_timeout + difs
        return ts, tc

    def _apply_safety_lower_bound(self, slot_us: int) -> int:
        """
        Ensure at least one complete contention + DATA + ACK exchange fits.
        """
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

    # ------------------------------------------------------------------
    # Load estimation
    # ------------------------------------------------------------------
    def _estimate_total_pending_packets(self) -> float:
        total = 0.0

        try:
            sim_nodes = getattr(self.ctx.sim, "nodes", {})
            for nid, node in sim_nodes.items():
                if int(nid) <= 0:
                    continue

                mac = getattr(node, "mac", None)
                if mac is not None:
                    for attr in ("tx_queue", "_tx_queue", "txq", "_txq"):
                        q = getattr(mac, attr, None)
                        if q is None:
                            continue
                        try:
                            total += float(len(q))
                            break
                        except Exception:
                            pass

                app = getattr(node, "app", None)
                if app is not None:
                    for attr in ("pending", "_pending", "queue", "_queue"):
                        q = getattr(app, attr, None)
                        if q is None:
                            continue
                        try:
                            total += float(len(q))
                            break
                        except Exception:
                            pass
        except Exception:
            pass

        if total <= 0.0:
            stats = getattr(self.ctx, "_stats", {}) or {}
            tx_attempts = float(stats.get("tx_attempts", 0))
            retries = float(stats.get("tx_retries", 0))
            raw_deferred = float(stats.get("raw_deferred", 0))
            total = max(0.0, tx_attempts - retries + 0.25 * raw_deferred)

        return max(0.0, total)

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------
    def _quantize_up(self, slot_us: int) -> int:
        step = max(1, self.step_us)
        q = int(math.ceil(slot_us / step)) * step
        q = max(self.min_slot_us, min(self.max_slot_us, q))
        q = max(MORSE_RAW_MIN_SLOT_DURATION_US, q)
        return q

    # ------------------------------------------------------------------
    # AID helpers
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

    # ------------------------------------------------------------------
    # Fixed grouping
    # ------------------------------------------------------------------
    def _partition_fixed_by_aid(self, aids: List[int], num_groups: int) -> List[List[int]]:
        aids = sorted(int(a) for a in aids if int(a) > 0)
        if not aids:
            return []

        groups: List[List[int]] = [[] for _ in range(max(1, num_groups))]
        for idx, aid in enumerate(aids):
            g = min(len(groups) - 1, int(idx * len(groups) / max(1, len(aids))))
            groups[g].append(aid)

        return [g for g in groups if g]