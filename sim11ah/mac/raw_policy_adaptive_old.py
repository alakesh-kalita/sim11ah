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
    Adaptive RAW policy that ONLY adapts slot duration.

    Fixed from static/AP configuration:
    - number of RAW groups
    - AID partitioning across groups
    - number of slots per group

    Adapted:
    - slot_duration_us per group, based on predicted demand and a Bianchi-style
      contention approximation.
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        mac_cfg = self.ctx.cfg["mac"]

        # Fixed RAW structure from baseline/static configuration
        self.fixed_num_groups = max(1, int(mac_cfg.get("raw_num_groups", self.ctx.raw_num_groups)))
        self.fixed_num_slots = max(1, int(mac_cfg.get("raw_num_slots", self.ctx.raw_num_slots)))

        # Adaptive slot-duration bounds only
        self.min_slot_us = int(
            mac_cfg.get(
                "adaptive_raw_min_slot_us",
                max(MORSE_RAW_MIN_SLOT_DURATION_US, int(self.ctx.raw_slot_duration * 1e6)),
            )
        )
        self.max_slot_us = int(
            mac_cfg.get(
                "adaptive_raw_max_slot_us",
                max(self.min_slot_us, int(self.ctx.raw_slot_duration * 2e6)),
            )
        )

        self.smoothing = float(mac_cfg.get("adaptive_raw_ewma_alpha", 0.7))
        self.bianchi_eps = float(mac_cfg.get("adaptive_raw_bianchi_eps", 1e-4))
        self.bianchi_imax = int(mac_cfg.get("adaptive_raw_bianchi_imax", 50))
        self.tmax_s = float(mac_cfg.get("adaptive_raw_tmax_s", 0.1))

        # Small fallback demand so policy does not collapse when queue visibility is imperfect
        self.baseline_demand = float(mac_cfg.get("adaptive_raw_baseline_demand", 0.15))

        self._load_ewma: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init_configs(self) -> List[RawConfig]:
        aids = self._effective_connected_aids([])
        if not aids:
            max_aid = max(1, int(self.ctx.raw_nodes_per_group))
        else:
            max_aid = max(aids)

        slot_us = max(MORSE_RAW_MIN_SLOT_DURATION_US, int(self.ctx.raw_slot_duration * 1e6))

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

    def build_dynamic_configs(self, connected_aids: List[int]) -> List[RawConfig]:
        aids = self._effective_connected_aids(connected_aids)
        if not aids:
            self._log("RAW_ADAPT_NO_AIDS", {})
            return []

        groups = self._partition_fixed_by_aid(aids, self.fixed_num_groups)

        sigma = float(self.ctx.slot_time)
        Lmax = max(1, int(self.tmax_s / max(1e-12, sigma)))

        configs: List[RawConfig] = []
        start_offset_us = int(self.ctx.raw_start_time_us)
        cfg_id = 1

        demand_sample = []

        for gid, member_aids in enumerate(groups, start=1):
            di = []
            for aid in member_aids:
                d = self._predict_sta_demand(aid)
                if d <= 0.0:
                    d = self.baseline_demand
                else:
                    d = max(d, self.baseline_demand * 0.5)
                di.append(d)

            Dgk = float(sum(di))
            nact = max(1, sum(1 for x in di if x > self.baseline_demand))


            if Dgk <= 0.0 or nact <= 0:
                continue

            tau, p = self._solve_bianchi_fixed_point(nact)

            Pidle = (1.0 - tau) ** nact
            Psucc = nact * tau * (1.0 - tau) ** (nact - 1)
            Pcol = max(0.0, 1.0 - Pidle - Psucc)

            Ts, Tc = self._estimate_event_times(member_aids)
            ls = Ts / max(1e-12, sigma)
            lc = Tc / max(1e-12, sigma)
            lbar = Pidle * 1.0 + Psucc * ls + Pcol * lc

            if Psucc <= 1e-12:
                Lgk = Lmax
            else:
                Lgk = int(math.ceil((Dgk * lbar) / max(Psucc, 1e-12)))

            Lgk = max(1, min(Lgk, Lmax))

            # ONLY adapt slot duration; slots/group stay fixed
            slot_us = int(Lgk * sigma * 1e6)
            slot_us = max(self.min_slot_us, min(self.max_slot_us, slot_us))
            slot_us = max(MORSE_RAW_MIN_SLOT_DURATION_US, slot_us)

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

            self._log(
                "RAW_ADAPT_GROUP",
                {
                    "gid": gid,
                    "start_aid": min(member_aids),
                    "end_aid": max(member_aids),
                    "n_members": len(member_aids),
                    "Dgk": round(Dgk, 4),
                    "nact": nact,
                    "tau": round(tau, 6),
                    "p": round(p, 6),
                    "Pidle": round(Pidle, 6),
                    "Psucc": round(Psucc, 6),
                    "Pcol": round(Pcol, 6),
                    "lbar": round(lbar, 6),
                    "Lgk": int(Lgk),
                    "slot_duration_us": int(slot_us),
                    "num_slots": int(self.fixed_num_slots),
                },
            )

        self._log("RAW_DEMAND_SAMPLE", {"sample": demand_sample})
        self._log(
            "RAW_ADAPT_BUILD",
            {
                "groups_built": len(configs),
                "stas_total": len(aids),
                "fixed_num_groups": self.fixed_num_groups,
                "fixed_num_slots": self.fixed_num_slots,
                "total_duration_us": start_offset_us - int(self.ctx.raw_start_time_us),
            },
        )

        if not configs:
            return self.init_configs()

        return configs

    def get_policy_snapshot(self) -> Dict[str, Any]:
        top = sorted(self._load_ewma.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "policy": "adaptive_slot_duration_only",
            "fixed_num_groups": self.fixed_num_groups,
            "fixed_num_slots": self.fixed_num_slots,
            "top_loads": top,
        }

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
    # Demand prediction
    # ------------------------------------------------------------------
    def _predict_sta_demand(self, aid: int) -> float:
        inst = self._estimate_queue_pressure(aid)
        prev = self._load_ewma.get(aid, inst)
        score = self.smoothing * inst + (1.0 - self.smoothing) * prev

        if score < 0.05:
            score = 0.0

        self._load_ewma[aid] = score
        return max(0.0, score)

    def _estimate_queue_pressure(self, aid: int) -> float:
        """
        Try several simulator-visible queue locations.
        """
        try:
            q = self.ctx._sta_ps_buffer.get(aid)
            if q is not None and len(q) > 0:
                return float(len(q))
        except Exception:
            pass

        try:
            sta = self.ctx.sim.nodes.get(aid)
            if sta is not None:
                mac_ctx = getattr(getattr(sta, "mac", None), "ctx", None)
                if mac_ctx is not None:
                    for attr in ["_txq", "txq", "_mac_queue", "mac_queue", "queue"]:
                        q = getattr(mac_ctx, attr, None)
                        if q is not None:
                            return float(len(q))

                mac_obj = getattr(sta, "mac", None)
                if mac_obj is not None:
                    for attr in ["_txq", "txq", "_mac_queue", "mac_queue", "queue"]:
                        q = getattr(mac_obj, attr, None)
                        if q is not None:
                            return float(len(q))
        except Exception:
            pass

        try:
            sta = self.ctx.sim.nodes.get(aid)
            if sta is not None:
                app_obj = getattr(sta, "app", None)
                if app_obj is not None:
                    for attr in ["pending_packets", "_pending_packets", "app_queue", "_app_queue"]:
                        q = getattr(app_obj, attr, None)
                        if q is not None:
                            return float(len(q))
        except Exception:
            pass

        return 0.0

    # ------------------------------------------------------------------
    # Fixed grouping
    # ------------------------------------------------------------------
    def _partition_fixed_by_aid(self, aids: List[int], num_groups: int) -> List[List[int]]:
        """
        Fixed AID partitioning, consistent with the static configuration idea.
        """
        aids = sorted(int(a) for a in aids if int(a) > 0)
        if not aids:
            return []

        groups: List[List[int]] = [[] for _ in range(max(1, num_groups))]
        for idx, aid in enumerate(aids):
            g = min(len(groups) - 1, int(idx * len(groups) / max(1, len(aids))))
            groups[g].append(aid)

        return [g for g in groups if g]

    # ------------------------------------------------------------------
    # Bianchi fixed point
    # ------------------------------------------------------------------
    def _solve_bianchi_fixed_point(self, nact: int) -> Tuple[float, float]:
        cw_min = max(1, int(self.ctx.cw_min))
        cw_max = max(cw_min, int(self.ctx.cw_max))

        try:
            m = int(round(math.log2((cw_max + 1) / (cw_min + 1))))
        except Exception:
            m = 4
        m = max(0, m)

        W0 = float(cw_min)

        if nact <= 1:
            tau = min(0.5, 2.0 / (W0 + 1.0))
            return tau, 0.0

        tau = min(0.5, 2.0 / (W0 + 1.0))
        p = 0.0

        for _ in range(self.bianchi_imax):
            p = 1.0 - (1.0 - tau) ** (nact - 1)

            denom = (1.0 - 2.0 * p) * (W0 + 1.0) + p * W0 * (1.0 - (2.0 * p) ** m)
            if abs(denom) < 1e-12:
                break

            tau_new = 2.0 * (1.0 - 2.0 * p) / denom
            tau_new = max(1e-6, min(0.5, tau_new))

            if abs(tau_new - tau) <= self.bianchi_eps:
                tau = tau_new
                break
            tau = tau_new

        return tau, p

    # ------------------------------------------------------------------
    # Event time estimation
    # ------------------------------------------------------------------
    def _estimate_event_times(self, member_aids: List[int]) -> Tuple[float, float]:
        sigma = float(self.ctx.slot_time)
        default_payload_bytes = int(self.ctx.cfg.get("app", {}).get("packet_size_bytes", 128))
        net_hdr = int(self.ctx.cfg.get("net", {}).get("net_header_bytes", 16))
        mac_overhead = int(self.ctx.data_mac_overhead_bytes)

        data_bytes = mac_overhead + net_hdr + default_payload_bytes
        mcs = self.ctx.node.phy.default_mode

        data_rate = int(self.ctx.node.phy.mode_table.get(mcs, 300_000))
        data_bits = max(0, data_bytes * 8)
        t_data = float(
            self.ctx.node.phy.preamble_time
            + self.ctx.node.phy.header_time
            + (data_bits / max(1, data_rate))
        )

        ack_rate = int(self.ctx.node.phy.mode_table.get(self.ctx.node.phy.control_mode, data_rate))
        ack_bits = max(0, int(self.ctx.ack_size_bytes) * 8)
        t_ack = float(
            self.ctx.node.phy.preamble_time
            + self.ctx.node.phy.header_time
            + (ack_bits / max(1, ack_rate))
        )

        Ts = t_data + self.ctx.sifs + t_ack + self.ctx.ack_guard
        Tc = t_data + max(float(self.ctx.ack_timeout_cfg), sigma)

        return Ts, Tc