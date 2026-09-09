"""
LACA: Load-Aware Channel Allocation for IEEE 802.11ah RAW slots.

Implements the two-level renewal-process slot-duration algorithm from:

  H. Taramit, L. Orozco-Barbosa, A. Haqiq, J. J. Camacho Escoto,
  and J. Gomez, "Load-Aware Channel Allocation for IEEE 802.11ah-Based
  Networks," IEEE Access, vol. 11, pp. 24484-24496, 2023.
  DOI: 10.1109/ACCESS.2023.3251896

Algorithm summary (Algorithm 1, p. 24491)
-----------------------------------------
Given N_S stations each holding one packet:

  T*_S = sum_{k=1}^{N_S} Z_k                              (eq. 1 / 26)

where the k-th renewal cycle has n_k = N_S - k + 1 contending stations
(each successful delivery removes one station):

  P_{k,i} = (1 - tau_k)^{n_k}                             (eq. 17)
  P_{k,s} = n_k * tau_k * (1-tau_k)^{n_k-1} / (1-P_{k,i})  (eq. 18)
  Z_k     = (1 / P_{k,s}) * (sigma * P_{k,i}/(1-P_{k,i}) + beta)  (eq. 25)

tau_k and collision probability p_k are solved via standard Bianchi
fixed-point equations (eq. 4 adapted for our simulator's CW parameters).

Adaptation notes
----------------
* Capture effect is disabled: our simulator uses an ideal channel with
  no Rayleigh fading, so P_{k,cap} = 0 throughout.
* With S sub-slots per group and round-robin STA-to-slot assignment, each
  sub-slot contains ceil(N_active / S) stations; LACA is evaluated at that
  reduced contender count.
* Group structure (number and AID ranges) is fixed; only slot duration adapts.
* Active-station count is estimated via per-STA EWMA of observed queue depth.
"""

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


class LacaRawPolicy:
    """
    LACA benchmark policy: renewal-process slot sizing with fixed groups.

    The slot duration for each sub-slot is set to T*_S computed for the
    per-sub-slot active station count (ceil(N_active / S)).
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        self.mac_cfg = ctx.cfg["mac"]
        self.phy_cfg = ctx.cfg["phy"]
        self.net_cfg = ctx.cfg["net"]
        self.app_cfg = ctx.cfg["app"]

        # ----------------------------------------------------------------
        # Fixed RAW structure (single group by default, same as adaptive)
        # ----------------------------------------------------------------
        self.fixed_num_groups = max(
            1,
            int(self.mac_cfg.get("raw_num_groups", getattr(ctx, "raw_num_groups", 1))),
        )
        self.fixed_num_slots = max(
            1,
            int(self.mac_cfg.get("raw_num_slots", getattr(ctx, "raw_num_slots", 1))),
        )

        # ----------------------------------------------------------------
        # Slot duration bounds (µs)
        # ----------------------------------------------------------------
        base_slot_us = int(float(getattr(ctx, "raw_slot_duration", 0.014)) * 1e6)
        # Hard minimum: enough for one DATA+ACK+guard (≈12.5 ms at MCS0 128 B)
        # Computed properly in _build_safe_min_us() after timing is known.
        self.min_slot_us = max(
            MORSE_RAW_MIN_SLOT_DURATION_US,
            int(self.mac_cfg.get("laca_min_slot_us", 12_020)),
        )
        self.max_slot_us = int(
            self.mac_cfg.get("laca_max_slot_us", max(base_slot_us * 4, 246_140))
        )

        # ----------------------------------------------------------------
        # Demand EWMA (tracks expected packets-per-STA per beacon)
        # ----------------------------------------------------------------
        self.ewma_alpha = float(self.mac_cfg.get("laca_ewma_alpha", 0.4))
        self._sta_ewma: Dict[int, float] = {}  # per-STA EWMA demand

        # ----------------------------------------------------------------
        # Bianchi solver tolerances
        # ----------------------------------------------------------------
        self._bianchi_eps = float(self.mac_cfg.get("laca_bianchi_eps", 1e-5))
        self._bianchi_imax = int(self.mac_cfg.get("laca_bianchi_imax", 100))

        self._current_slot_us: int = base_slot_us
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Public API (same interface as all other raw_policy_*.py)
    # ------------------------------------------------------------------
    def init_configs(self) -> List[RawConfig]:
        aids = self._effective_connected_aids([])
        max_aid = (
            max(aids)
            if aids
            else max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))
        )
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

    def build_dynamic_configs(self, connected_aids: List[int]) -> List[RawConfig]:
        aids = self._effective_connected_aids(connected_aids)
        if not aids:
            return []

        groups = self._partition_aids(aids, self.fixed_num_groups)
        if not groups:
            return self.init_configs()

        sigma_s, beta_s = self._timing_params()
        S = self.fixed_num_slots

        configs: List[RawConfig] = []
        start_offset_us = int(self.ctx.raw_start_time_us)
        cfg_id = 1

        for gid, member_aids in enumerate(groups, start=1):
            if not member_aids:
                continue

            n_active = self._estimate_n_active(member_aids)

            # LACA is formulated for one slot with N_S single-packet STAs.
            # Round-robin sub-slot assignment gives ceil(N/S) per sub-slot.
            # Always evaluate at n_per_slot >= 1 so the slot is at least
            # wide enough for one DATA+ACK exchange (LACA(1) ≈ 12 ms @ MCS0).
            n_per_slot = max(1, math.ceil(n_active / max(1, S)))
            t_star_s = self._laca_time_s(n_per_slot, sigma_s, beta_s)
            slot_us = self._to_slot_us(t_star_s)

            cfg = RawConfig(
                id=cfg_id,
                raw_type=self.ctx.raw_type,
                start_aid=min(member_aids),
                end_aid=max(member_aids),
                start_time_us=start_offset_us,
                slot_definition=RawSlotDefinition(
                    num_slots=S,
                    slot_duration_us=slot_us,
                    cross_slot_boundary=self.ctx.raw_cross_slot,
                ),
                beacon_spreading=RawBeaconSpreading(),
                periodic=RawPeriodic(),
                enabled=True,
            )
            configs.append(cfg)
            cfg_id += 1
            start_offset_us += S * slot_us

            self._log(
                "LACA_GROUP",
                {
                    "gid": gid,
                    "start_aid": min(member_aids),
                    "end_aid": max(member_aids),
                    "n_members": len(member_aids),
                    "n_active_est": n_active,
                    "n_per_slot": n_per_slot,
                    "t_star_ms": round(t_star_s * 1e3, 3),
                    "slot_us": slot_us,
                    "num_slots": S,
                    "update_count": self._update_count,
                },
            )

        self._update_count += 1
        if configs:
            self._current_slot_us = configs[0].slot_definition.slot_duration_us
        return configs if configs else self.init_configs()

    def get_policy_snapshot(self) -> Dict[str, Any]:
        return {
            "policy": "laca_renewal_process",
            "fixed_num_groups": self.fixed_num_groups,
            "fixed_num_slots": self.fixed_num_slots,
            "slot_duration_us": self._current_slot_us,
            "ewma_alpha": self.ewma_alpha,
            "update_count": self._update_count,
        }

    # ------------------------------------------------------------------
    # LACA core: two-level renewal process (eq. 1 / 25 / 26)
    # ------------------------------------------------------------------
    def _laca_time_s(self, n_s: int, sigma: float, beta: float) -> float:
        """
        Compute T*_S for n_s stations, each with one packet.

        The k-th level-2 renewal cycle has n_s - k + 1 stations contending;
        one station leaves after each successful delivery.

        T*_S = sum_{k=1}^{n_s} Z_k
        Z_k  = (1 / P_{k,s}) * (sigma * P_{k,i} / (1 - P_{k,i}) + beta)
        """
        t_star = 0.0
        for k in range(1, n_s + 1):
            n_k = n_s - k + 1  # contenders in k-th cycle

            tau, _ = self._bianchi(n_k)

            p_idle = (1.0 - tau) ** n_k

            if 1.0 - p_idle < 1e-12:
                # Degenerate: station never transmits — use one beta duration
                z_k = beta
            else:
                # Conditional success probability: given non-idle slot, success
                p_succ_unc = n_k * tau * (1.0 - tau) ** (n_k - 1)
                p_k_s = max(1e-12, p_succ_unc / (1.0 - p_idle))

                # Mean idles before a busy slot (geometric with param 1-P_idle)
                mean_idles = p_idle / (1.0 - p_idle)

                z_k = (1.0 / p_k_s) * (sigma * mean_idles + beta)

            t_star += z_k

        return t_star

    # ------------------------------------------------------------------
    # Bianchi fixed-point solver (standard infinite-retry form)
    # ------------------------------------------------------------------
    def _bianchi(self, n_active: int) -> Tuple[float, float]:
        """Return (tau, p) for n_active contenders."""
        if n_active <= 0:
            return 0.0, 0.0
        if n_active == 1:
            W0 = float(self.mac_cfg.get("cw_min", 15)) + 1.0
            return min(1.0, 2.0 / (W0 + 1.0)), 0.0

        cw_min = float(self.mac_cfg.get("cw_min", 15))
        cw_max = float(self.mac_cfg.get("cw_max", 1023))
        W0 = cw_min + 1.0
        try:
            m = max(1, int(round(math.log2((cw_max + 1.0) / W0))))
        except Exception:
            m = 6

        tau = min(0.5, 2.0 / (W0 + 1.0))
        for _ in range(self._bianchi_imax):
            p = 1.0 - (1.0 - tau) ** (n_active - 1)
            denom = (1.0 - 2.0 * p) * (W0 + 1.0) + p * W0 * (1.0 - (2.0 * p) ** m)
            if abs(denom) < 1e-14:
                break
            tau_new = max(1e-9, min(1.0, 2.0 * (1.0 - 2.0 * p) / denom))
            if abs(tau_new - tau) < self._bianchi_eps:
                tau = tau_new
                break
            tau = tau_new

        p = 1.0 - (1.0 - tau) ** (n_active - 1)
        return tau, p

    # ------------------------------------------------------------------
    # Active-STA estimation
    # ------------------------------------------------------------------
    def _estimate_n_active(self, member_aids: List[int]) -> int:
        """
        Estimate N_S = expected packets to deliver this beacon.

        Uses EWMA of observed queue depth per STA.  N_S = round(sum of EWMA
        demands), capped at the group size — the same demand-proportional
        contender count used in TASM-RAW.  This avoids the over-count that
        occurs when counting any EWMA-positive STA (EWMA decays slowly; a STA
        that delivered its last packet 10 beacons ago still has EWMA > 0).
        """
        d_total = 0.0
        for aid in member_aids:
            depth = self._observe_queue(aid)
            prev = self._sta_ewma.get(aid, depth)
            ewma = self.ewma_alpha * depth + (1.0 - self.ewma_alpha) * prev
            self._sta_ewma[aid] = ewma
            d_total += ewma
        # N_S = demand sum clipped to [0, group_size]
        return max(0, min(len(member_aids), round(d_total)))

    def _observe_queue(self, aid: int) -> float:
        try:
            node = getattr(self.ctx.sim, "nodes", {}).get(aid)
            if node is None:
                return 0.0
            mac = getattr(node, "mac", None)
            if mac is not None:
                for attr in ("tx_queue", "_tx_queue", "txq", "_txq"):
                    q = getattr(mac, attr, None)
                    if q is not None:
                        try:
                            return float(len(q))
                        except Exception:
                            pass
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # MAC / PHY timing parameters
    # ------------------------------------------------------------------
    def _timing_params(self) -> Tuple[float, float]:
        """Return (sigma, beta) in seconds.

        sigma = idle slot time
        beta  = busy slot time = T_DATA + SIFS + T_ACK + DIFS  (eq. 2 / paper)
        """
        payload_bytes = int(self.app_cfg.get("packet_size_bytes", 128))
        mac_oh = int(self.mac_cfg.get("data_mac_overhead_bytes", 36))
        net_oh = int(self.net_cfg.get("net_header_bytes", 16))
        ack_bytes = int(self.mac_cfg.get("ack_size_bytes", 14))

        preamble = float(self.phy_cfg.get("preamble_time", 320e-6))
        header_t = float(self.phy_cfg.get("header_time", 80e-6))
        sifs = float(self.mac_cfg.get("sifs", 160e-6))
        difs = float(self.mac_cfg.get("difs", 264e-6))
        sigma = float(self.mac_cfg.get("slot_time", 52e-6))

        mode_table = self.phy_cfg.get("mode_table", {}) or {}
        data_mode = str(self.phy_cfg.get("default_mode", "MCS0"))
        data_rate = float(mode_table.get(data_mode, 150_000.0))

        t_data = preamble + header_t + 8.0 * (payload_bytes + mac_oh + net_oh) / max(1.0, data_rate)
        t_ack = preamble + header_t + 8.0 * ack_bytes / max(1.0, data_rate)
        beta = t_data + sifs + t_ack + difs

        return sigma, beta

    # ------------------------------------------------------------------
    # Slot-duration quantization (120 µs RAW unit, MORSE encoding)
    # ------------------------------------------------------------------
    def _to_slot_us(self, t_s: float) -> int:
        slot_us = int(math.ceil(t_s * 1e6))
        slot_us = max(self.min_slot_us, min(self.max_slot_us, slot_us))
        # Quantize to 120 µs grid: slot_us = 500 + n*120
        offset = slot_us - MORSE_RAW_MIN_SLOT_DURATION_US
        n = int(math.ceil(max(0, offset) / 120))
        return MORSE_RAW_MIN_SLOT_DURATION_US + n * 120

    # ------------------------------------------------------------------
    # AID helpers (identical to AdaptiveRawPolicy)
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
            nodes = getattr(self.ctx.sim, "nodes", {})
            vals = sorted(int(n) for n in nodes.keys() if int(n) > 0)
            if vals:
                return vals
        except Exception:
            pass
        return []

    def _partition_aids(self, aids: List[int], num_groups: int) -> List[List[int]]:
        aids = sorted(int(a) for a in aids if int(a) > 0)
        if not aids:
            return []
        groups: List[List[int]] = [[] for _ in range(max(1, num_groups))]
        for idx, aid in enumerate(aids):
            g = min(len(groups) - 1, int(idx * len(groups) / max(1, len(aids))))
            groups[g].append(aid)
        return [g for g in groups if g]
