# sim11ah/phy.py
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
from collections import defaultdict

from sim11ah.models import MacFrame
from sim11ah.constants import FrameType


@dataclass
class TxRecord:
    tx_id: int
    rx_id: int
    start_time: float
    end_time: float
    eirp_dbm: float
    mode: str
    frame_seq: int


@dataclass
class RxRecord:
    frame: MacFrame
    tx_id: int
    rx_id: int
    start_time: float
    end_time: float
    rssi_dbm: float
    snr_db: float = 0.0
    sinr_db: float = 0.0
    noise_dbm: float = 0.0

    half_duplex_collision: bool = False
    collided: bool = False
    per_drop: bool = False
    below_sensitivity: bool = False
    unsupported_mode: bool = False
    rx_ok: bool = False

    drop_reason: str = ""
    _rx_resolved: bool = False


class PhyLayer:
    def __init__(self, node: "Node", cfg: Dict[str, Any]):
        self.node = node
        self.sim = node.sim
        self.cfg = cfg

        phy_cfg = cfg.get("phy", {})

        # Timing / rates
        self.preamble_time = float(phy_cfg.get("preamble_time", 0.00032))
        self.header_time = float(phy_cfg.get("header_time", 0.00008))
        self.slot_time = float(phy_cfg.get("slot_time", 0.000052))
        self.mode_table = {str(k): int(v) for k, v in dict(phy_cfg.get("mode_table", {})).items()}
        self.default_mode = str(phy_cfg.get("default_mode", "MCS0"))
        self.control_mode = str(phy_cfg.get("control_mode", self.default_mode))

        # IEEE 802.11ah 1 MHz S1G: MCS0 = 150 kb/s (BPSK 1/2).
        if self.default_mode not in self.mode_table:
            self.mode_table[self.default_mode] = int(phy_cfg.get("default_rate_bps", 150_000))
        if self.control_mode not in self.mode_table:
            self.mode_table[self.control_mode] = int(self.mode_table.get(self.default_mode, 150_000))

        self.available_modes: List[str] = sorted(
            self.mode_table.keys(),
            key=lambda m: int(self.mode_table[m]),
        ) or [self.default_mode]

        # Optional receiver capability filter
        self.supported_modes = {str(m) for m in phy_cfg.get("supported_modes", self.available_modes)}

        # CCA / energy detection / RX sensitivity
        self.cca_threshold_dbm = float(phy_cfg.get("cca_threshold_dbm", -96.0))
        self.rx_sensitivity_dbm = float(phy_cfg.get("rx_sensitivity_dbm", -98.0))

        # TX power / EIRP
        self.tx_power_dbm = float(phy_cfg.get("tx_power_dbm", 20.0))
        node_tx_key = f"tx_power_node_{self.node.node_id}_dbm"
        if node_tx_key in phy_cfg:
            self.tx_power_dbm = float(phy_cfg[node_tx_key])
        self.antenna_gain_db = float(phy_cfg.get("antenna_gain_db", 0.0))
        self.eirp_dbm = self.tx_power_dbm + self.antenna_gain_db

        # Propagation model
        self.freq_mhz = float(phy_cfg.get("freq_mhz", 915.0))
        self.path_loss_exp = float(phy_cfg.get("path_loss_exp", 2.7))
        self.d0_m = float(phy_cfg.get("pl_ref_distance_m", 1.0))
        self.shadow_sigma_db = float(phy_cfg.get("shadow_sigma_db", 4.0))
        self.shadow_enable = bool(phy_cfg.get("shadow_enable", True))
        self.shadow_symmetric = bool(phy_cfg.get("shadow_symmetric", True))

        # Noise floor
        self.channel_bw_hz = float(phy_cfg.get("channel_bw_hz", 1_000_000.0))
        self.noise_figure_db = float(phy_cfg.get("noise_figure_db", 5.0))

        # PER model
        self.per_floor = float(phy_cfg.get("per_floor", 1e-9))
        # per_alpha controls steepness of exp(-alpha*(sinr-threshold)) PER curve.
        # 2.0 gives ~14% PER at 1 dB above threshold; matches config.py default.
        self.per_alpha = float(phy_cfg.get("per_alpha", 2.0))
        self.mcs_min_snr_db: Dict[str, float] = {
            str(k): float(v) for k, v in dict(phy_cfg.get("mcs_min_snr_db", {})).items()
        }
        if not self.mcs_min_snr_db:
            self.mcs_min_snr_db = {
                "MCS0": 3.0,
                "MCS1": 6.0,
                "MCS2": 8.5,
                "MCS3": 11.5,
                "MCS4": 15.0,
                "MCS5": 18.0,
                "MCS6": 20.0,
                "MCS7": 22.0,
                "MCS8": 25.0,
                "MCS9": 28.0,
                "MCS10": 31.0,
            }

        self.mcs_sinr_penalty_db: Dict[str, float] = {
            str(k): float(v) for k, v in dict(phy_cfg.get("mcs_sinr_penalty_db", {})).items()
        }

        # Collision model
        cm = str(phy_cfg.get("collision_model", "capture")).strip().lower()
        if cm == "overlap":
            cm = "pessimistic"
        if cm not in {"capture", "pessimistic"}:
            cm = "capture"
        self.collision_model = cm
        self.capture_threshold_db = float(phy_cfg.get("capture_threshold_db", 10.0))

        # Local RX state per receiver node
        self._active_rx: Dict[int, List[RxRecord]] = defaultdict(list)

        # Shadowing cache
        self._shadow_db: Dict[Tuple[int, int], float] = {}
        self._last_rssi_dbm: Dict[Tuple[int, int], float] = {}

        # Shared/global medium occupancy for MAC compatibility
        if not hasattr(self.sim, "_medium_tx"):
            self.sim._medium_tx = []

        # Shared/global active-air TX registry for proper CCA/interference.
        # Keyed by tx_id (one active TX per node at a time, enforced by 802.11 half-duplex).
        if not hasattr(self.sim, "_active_air_tx"):
            self.sim._active_air_tx: Dict[int, TxRecord] = {}

    # --------------------------
    # Helpers: dB conversions
    # --------------------------
    @staticmethod
    def _dbm_to_mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    @staticmethod
    def _mw_to_dbm(mw: float) -> float:
        return 10.0 * math.log10(max(1e-30, mw))

    def _thermal_noise_dbm(self) -> float:
        return -174.0 + 10.0 * math.log10(max(1.0, self.channel_bw_hz)) + self.noise_figure_db

    # --------------------------
    # Stats helpers
    # --------------------------
    def _inc_stat(self, name: str, amount: int = 1) -> None:
        # Keep compatibility with any sim.stats object
        stats = getattr(self.sim, "stats", None)
        if stats is not None and hasattr(stats, name):
            try:
                setattr(stats, name, getattr(stats, name) + amount)
            except Exception:
                pass

        # Also update MAC-context stats when available so counters match the rest
        try:
            mac = getattr(self.node, "mac", None)
            ctx = getattr(mac, "ctx", None)
            d = getattr(ctx, "_stats", None)
            if isinstance(d, dict):
                d[name] = int(d.get(name, 0)) + int(amount)
        except Exception:
            pass

    # --------------------------
    # Propagation / RSSI
    # --------------------------
    def reset_shadowing_cache(self) -> None:
        self._shadow_db.clear()
        self._last_rssi_dbm.clear()

    def _distance_m(self, tx_id: int, rx_id: int) -> float:
        topo = getattr(self.sim, "topology", None)
        if topo is not None and hasattr(topo, "get_distance_m"):
            try:
                d = float(topo.get_distance_m(tx_id, rx_id))
                return max(1e-3, d)
            except Exception:
                pass

        tx_node = self.sim.nodes.get(tx_id)
        rx_node = self.sim.nodes.get(rx_id)
        if tx_node is not None and rx_node is not None:
            if hasattr(tx_node, "pos") and hasattr(rx_node, "pos"):
                try:
                    dx = float(tx_node.pos[0]) - float(rx_node.pos[0])
                    dy = float(tx_node.pos[1]) - float(rx_node.pos[1])
                    return max(1e-3, math.hypot(dx, dy))
                except Exception:
                    pass

            for keyx, keyy in (("x", "y"), ("pos_x", "pos_y")):
                if (
                    hasattr(tx_node, keyx) and hasattr(tx_node, keyy)
                    and hasattr(rx_node, keyx) and hasattr(rx_node, keyy)
                ):
                    dx = float(getattr(tx_node, keyx)) - float(getattr(rx_node, keyx))
                    dy = float(getattr(tx_node, keyy)) - float(getattr(rx_node, keyy))
                    return max(1e-3, math.hypot(dx, dy))

        return 1.0

    def _fspl_db_1m(self) -> float:
        return 32.44 + 20.0 * math.log10(max(1e-9, self.freq_mhz)) - 60.0

    def _shadow_key(self, tx_id: int, rx_id: int) -> Tuple[int, int]:
        if self.shadow_symmetric:
            return tuple(sorted((int(tx_id), int(rx_id))))
        return (int(tx_id), int(rx_id))

    def _path_loss_db(self, tx_id: int, rx_id: int) -> float:
        d = self._distance_m(tx_id, rx_id)
        pl1 = self._fspl_db_1m()
        pl = pl1 + 10.0 * self.path_loss_exp * math.log10(max(1e-6, d / max(1e-6, self.d0_m)))

        if self.shadow_enable:
            key = self._shadow_key(tx_id, rx_id)
            if key not in self._shadow_db:
                x = self.sim.engine.rng.gauss(0.0, self.shadow_sigma_db)
                self._shadow_db[key] = float(x)
            pl += self._shadow_db[key]

        pl += self._obstruction_loss_db(tx_id, rx_id)

        return float(pl)

    def _obstruction_loss_db(self, tx_id: int, rx_id: int) -> float:
        """Extra path loss (dB) from physical obstacles (buildings, terrain)
        whose footprint crosses the tx-rx line. Opt-in only: a Simulator
        has no ``obstacles`` attribute unless something explicitly sets one
        (currently: the interactive GUI's environment scenery), so this is
        a no-op for every scripted/headless simulation run."""
        obstacles = getattr(self.sim, "obstacles", None)
        if not obstacles:
            return 0.0
        tx_node = self.sim.nodes.get(tx_id)
        rx_node = self.sim.nodes.get(rx_id)
        if tx_node is None or rx_node is None:
            return 0.0
        total = 0.0
        for obs in obstacles:
            try:
                if obs.intersects_segment(tx_node.pos, rx_node.pos):
                    total += float(obs.loss_db)
            except Exception:
                continue
        return total

    def _rssi_dbm(self, tx_id: int, rx_id: int, eirp_dbm: float) -> float:
        rssi = float(eirp_dbm) - self._path_loss_db(tx_id, rx_id)
        self._last_rssi_dbm[(tx_id, rx_id)] = rssi
        return rssi

    def get_prop_delay(self, tx_id: int, rx_id: int) -> float:
        try:
            lk = self.sim.topology.get_link(tx_id, rx_id)
            return float(getattr(lk, "prop_delay", 0.0))
        except Exception:
            return 0.0

    def _get_link_params(self, tx_id: int, rx_id: int) -> Tuple[float, float, bool]:
        """
        Returns:
            (prop_delay, flat_per, link_found)
        """
        try:
            lk = self.sim.topology.get_link(tx_id, rx_id)
            prop = float(getattr(lk, "prop_delay", 0.0))
            flat_per = float(getattr(lk, "per", 0.0) or 0.0)
            return prop, flat_per, True
        except Exception:
            return 0.0, 0.0, False

    # --------------------------
    # Duration / rate
    # --------------------------
    def compute_tx_duration(self, frame: MacFrame, rate_bps: int) -> float:
        payload_bits = max(0, int(frame.size_bytes) * 8)
        return self.preamble_time + self.header_time + (payload_bits / max(1, int(rate_bps)))

    def _control_frame_types(self) -> set:
        types = {
            getattr(FrameType, "ACK", None),
            getattr(FrameType, "CTS", None),
            getattr(FrameType, "RTS", None),
            getattr(FrameType, "BLOCK_ACK", None),
            getattr(FrameType, "BLOCK_ACK_REQ", None),
            getattr(FrameType, "CF_END", None),
            getattr(FrameType, "BEACON", None),
        }
        types.discard(None)
        return types

    def _select_tx_mode(self, frame: MacFrame, phy_mode: Optional[str]) -> str:
        if phy_mode is not None:
            mode = str(phy_mode)
        elif frame.ftype in self._control_frame_types():
            mode = self.control_mode
        else:
            mode = str(
                (frame.ctrl.get("mcs") if getattr(frame, "ctrl", None) else None)
                or self.default_mode
            )

        if mode not in self.mode_table:
            mode = self.default_mode
        return mode

    def _receiver_phy(self, rx_id: int) -> "PhyLayer":
        rx_node = self.sim.nodes.get(rx_id)
        if rx_node is not None and getattr(rx_node, "phy", None) is not None:
            return rx_node.phy
        return self

    # --------------------------
    # Shared-air helpers
    # --------------------------
    def _get_global_active_air_tx(self) -> Dict[int, TxRecord]:
        return getattr(self.sim, "_active_air_tx")

    def _is_tx_active_now(self, txr: TxRecord, now: float) -> bool:
        return txr.start_time <= now < txr.end_time

    def medium_next_idle_time(self, now: Optional[float] = None) -> float:
        if now is None:
            now = self.sim.engine.now
        t = float(now)
        for txr in self._get_global_active_air_tx().values():
            if txr.start_time <= now < txr.end_time:
                t = max(t, txr.end_time)
        return t

    # --------------------------
    # Busy / CCA
    # --------------------------
    def is_channel_busy(self, node_id: int) -> bool:
        now = self.sim.engine.now
        rx_phy = self._receiver_phy(node_id)

        for txr in self._get_global_active_air_tx().values():
            if not self._is_tx_active_now(txr, now):
                continue
            if txr.tx_id == node_id:
                return True
            rssi = self._rssi_dbm(txr.tx_id, node_id, txr.eirp_dbm)
            if rssi >= rx_phy.cca_threshold_dbm:
                return True
        return False

    # --------------------------
    # PER model
    # --------------------------
    def _min_snr_for_mcs(self, mcs: str) -> float:
        if mcs in self.mcs_min_snr_db:
            return float(self.mcs_min_snr_db[mcs])
        return float(self.mcs_min_snr_db.get(self.default_mode, 99.0))

    def _effective_sinr_for_mcs(self, sinr_db: float, mcs: str) -> float:
        pen = float(self.mcs_sinr_penalty_db.get(mcs, 0.0))
        return float(sinr_db - pen)

    def _per_from_sinr(self, sinr_db: float, mcs: str) -> float:
        sinr_eff = self._effective_sinr_for_mcs(sinr_db, mcs)
        thr = self._min_snr_for_mcs(mcs)
        if sinr_eff < thr:
            return 1.0
        x = max(0.0, sinr_eff - thr)
        per = math.exp(-self.per_alpha * x)
        return float(min(1.0, max(self.per_floor, per)))

    def _effective_per(self, sinr_db: float, mcs: str, flat_per: float) -> float:
        if flat_per > 0.0:
            return float(min(1.0, max(0.0, flat_per)))
        return self._per_from_sinr(sinr_db, mcs)

    # --------------------------
    # Send / schedule RX
    # --------------------------
    def send(self, frame: MacFrame, tx_id: int, rx_id: int, phy_mode: Optional[str] = None) -> None:
        mode = self._select_tx_mode(frame, phy_mode)
        rate = int(self.mode_table.get(mode, self.mode_table.get(self.default_mode, 150_000)))
        dur = self.compute_tx_duration(frame, rate)
        now = self.sim.engine.now
        t1 = now + dur

        txr = TxRecord(
            tx_id=tx_id,
            rx_id=rx_id,
            start_time=now,
            end_time=t1,
            eirp_dbm=float(self.eirp_dbm),
            mode=mode,
            frame_seq=int(frame.frame_seq),
        )

        self._get_global_active_air_tx()[tx_id] = txr

        medium_entry = (now, t1, tx_id, rx_id, int(frame.frame_seq))
        self.sim._medium_tx.append(medium_entry)
        self.sim.engine.schedule(t1, self._tx_end_medium, medium_entry, name="PHY_TX_END_MEDIUM")
        self.sim.engine.schedule(t1, self._tx_end_air, txr, name="PHY_TX_END_AIR")

        # Energy: charge TX power for the entire air-time of this frame.
        _energy = getattr(self.sim, "energy", None)
        if _energy is not None:
            try:
                _energy.on_tx_start(
                    tx_id, dur,
                    is_retry=bool(getattr(frame, "retry", 0)),
                )
            except Exception:
                pass

        self.sim.log(
            node_id=tx_id,
            layer="PHY",
            event="TX_START",
            frame=frame,
            details={
                "rx_id": rx_id,
                "mode": mode,
                "rate_bps": rate,
                "tx_duration": dur,
                "eirp_dbm": float(self.eirp_dbm),
                "retry": getattr(frame, "retry", 0),
                "broadcast": bool(rx_id == -1),
            },
        )

        if rx_id != -1:
            self._schedule_one_rx(frame, txr, rx_id, now, dur, mode)
            return

        for nid in list(self.sim.nodes.keys()):
            if nid == tx_id:
                continue
            self._schedule_one_rx(frame, txr, nid, now, dur, mode)

    def _tx_end_medium(self, medium_entry: tuple) -> None:
        try:
            self.sim._medium_tx.remove(medium_entry)
        except ValueError:
            pass

    def _tx_end_air(self, txr: TxRecord) -> None:
        active = self._get_global_active_air_tx()
        if active.get(txr.tx_id) is txr:
            del active[txr.tx_id]

    def _schedule_one_rx(
        self,
        frame: MacFrame,
        txr: TxRecord,
        rx_id: int,
        t0: float,
        dur: float,
        mode: str,
    ) -> None:
        prop, flat_per, link_found = self._get_link_params(txr.tx_id, rx_id)

        topo = getattr(self.sim, "topology", None)
        if topo is not None and not link_found:
            self.sim.log(
                node_id=rx_id,
                layer="PHY",
                event="RX_LINK_MISSING",
                frame=frame,
                details={
                    "tx_id": txr.tx_id,
                    "mode": mode,
                },
            )
            return

        t_rx_start = t0 + prop
        t_rx_end = t_rx_start + dur

        # Always shallow-copy per receiver to avoid shared mutations
        fr_copy = copy.copy(frame)
        if getattr(frame, "ctrl", None) is not None:
            fr_copy.ctrl = dict(frame.ctrl)

        self.sim.engine.schedule(
            t_rx_start,
            self._rx_start,
            fr_copy,
            txr,
            rx_id,
            t_rx_start,
            t_rx_end,
            mode,
            flat_per,
            name="PHY_RX_START",
        )

    # --------------------------
    # RX start/end
    # --------------------------
    def _rx_start(
        self,
        frame: MacFrame,
        txr: TxRecord,
        rx_id: int,
        t_rx_start: float,
        t_rx_end: float,
        mode: str,
        flat_per: float,
    ) -> None:
        rx_phy = self._receiver_phy(rx_id)
        rssi = self._rssi_dbm(txr.tx_id, rx_id, txr.eirp_dbm)

        if mode not in rx_phy.supported_modes:
            self._inc_stat("phy_unsupported_mode")
            self.sim.log(
                node_id=rx_id,
                layer="PHY",
                event="RX_UNSUPPORTED_MODE",
                frame=frame,
                details={
                    "tx_id": txr.tx_id,
                    "mode": mode,
                    "supported_modes": list(sorted(rx_phy.supported_modes)),
                },
            )
            return

        if rssi < rx_phy.rx_sensitivity_dbm:
            self._inc_stat("phy_below_sensitivity")
            self.sim.log(
                node_id=rx_id,
                layer="PHY",
                event="RX_BELOW_SENSITIVITY",
                frame=frame,
                details={
                    "tx_id": txr.tx_id,
                    "rssi_dbm": rssi,
                    "rx_sens_dbm": rx_phy.rx_sensitivity_dbm,
                    "mode": mode,
                },
            )
            return

        rr = RxRecord(
            frame=frame,
            tx_id=txr.tx_id,
            rx_id=rx_id,
            start_time=t_rx_start,
            end_time=t_rx_end,
            rssi_dbm=rssi,
        )

        # Half-duplex collision: receiver is also transmitting during RX window.
        # O(1) lookup since dict is keyed by tx_id.
        oth_tx = self._get_global_active_air_tx().get(rx_id)
        if oth_tx is not None:
            overlap = not (t_rx_end <= oth_tx.start_time or t_rx_start >= oth_tx.end_time)
            if overlap:
                rr.half_duplex_collision = True
                rr.collided = True

        self._active_rx[rx_id].append(rr)

        if rx_phy.collision_model == "pessimistic":
            rx_phy._apply_pessimistic_overlap(rx_id)
        else:
            rx_phy._apply_capture_all_pairs(rx_id)

        # Energy: charge RX power for the full reception window.
        # We hook here (not _rx_end) because the radio is busy regardless of
        # decode success; duration is known at RX start.
        _energy = getattr(self.sim, "energy", None)
        if _energy is not None:
            try:
                _energy.on_rx_start(rx_id, t_rx_end - t_rx_start, received_ok=True)
            except Exception:
                pass

        self.sim.log(
            node_id=rx_id,
            layer="PHY",
            event="RX_START",
            frame=frame,
            details={
                "tx_id": txr.tx_id,
                "rssi_dbm": rr.rssi_dbm,
                "mode": mode,
                "collision_model": rx_phy.collision_model,
            },
        )

        self.sim.engine.schedule(
            t_rx_end,
            rx_phy._rx_end,
            rr,
            mode,
            float(flat_per),
            name="PHY_RX_END",
        )

    def _apply_pessimistic_overlap(self, rx_id: int) -> None:
        lst = self._active_rx.get(rx_id, [])
        for i, r1 in enumerate(lst):
            for r2 in lst[i + 1:]:
                overlap = not (r1.end_time <= r2.start_time or r1.start_time >= r2.end_time)
                if overlap:
                    r1.collided = True
                    r2.collided = True

    def _apply_capture_all_pairs(self, rx_id: int) -> None:
        lst = self._active_rx.get(rx_id, [])
        n = len(lst)
        if n <= 1:
            return

        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            r1 = lst[i]
            for j in range(i + 1, n):
                r2 = lst[j]
                overlap = not (r1.end_time <= r2.start_time or r1.start_time >= r2.end_time)
                if overlap:
                    adj[i].append(j)
                    adj[j].append(i)

        seen = [False] * n
        for i in range(n):
            if seen[i]:
                continue
            stack = [i]
            comp: List[int] = []
            seen[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)

            if len(comp) <= 1:
                continue

            comp_sorted = sorted(comp, key=lambda idx: lst[idx].rssi_dbm, reverse=True)
            strong = lst[comp_sorted[0]]

            for idx in comp:
                lst[idx].collided = True

            if strong.half_duplex_collision:
                strong.collided = True
                continue

            # SINR-based capture: signal / (thermal_noise + all_interferers).
            noise_mw = self._dbm_to_mw(self._thermal_noise_dbm())
            sig_mw = self._dbm_to_mw(strong.rssi_dbm)
            interf_mw = sum(self._dbm_to_mw(lst[idx].rssi_dbm) for idx in comp_sorted[1:])
            sinr_db = self._mw_to_dbm(sig_mw / max(1e-30, noise_mw + interf_mw))

            if sinr_db >= self.capture_threshold_db:
                strong.collided = False

    def _calc_interference_mw(self, rr: RxRecord) -> float:
        """
        Interference from all overlapping on-air transmissions at the receiver,
        excluding the desired transmission itself.
        """
        interf_mw = 0.0
        for txr in self._get_global_active_air_tx().values():
            if txr.tx_id == rr.tx_id and txr.frame_seq == int(rr.frame.frame_seq):
                continue
            # Half-duplex: the receiving node's own transmission is not external interference.
            if txr.tx_id == rr.rx_id:
                continue

            # Account for propagation delay: compare receiver-side arrival windows.
            prop, _, link_found = self._get_link_params(txr.tx_id, rr.rx_id)
            if not link_found and getattr(self.sim, "topology", None) is not None:
                continue  # no defined link → signal doesn't reach receiver
            txr_rx_start = txr.start_time + prop
            txr_rx_end   = txr.end_time   + prop
            overlap = not (rr.end_time <= txr_rx_start or rr.start_time >= txr_rx_end)
            if not overlap:
                continue

            rssi_i_dbm = self._rssi_dbm(txr.tx_id, rr.rx_id, txr.eirp_dbm)
            interf_mw += self._dbm_to_mw(rssi_i_dbm)

        return interf_mw

    def _rx_end(self, rr: RxRecord, mode: str, flat_per: float) -> None:
        noise_dbm = self._thermal_noise_dbm()
        noise_mw = self._dbm_to_mw(noise_dbm)

        interf_mw = self._calc_interference_mw(rr)
        sig_mw = self._dbm_to_mw(rr.rssi_dbm)
        sinr_mw = sig_mw / max(1e-30, noise_mw + interf_mw)
        sinr_db = 10.0 * math.log10(max(1e-30, sinr_mw))

        rr.noise_dbm = float(noise_dbm)
        rr.snr_db = float(rr.rssi_dbm - noise_dbm)
        rr.sinr_db = float(sinr_db)

        rx_list = self._active_rx.get(rr.rx_id, [])
        try:
            rx_list.remove(rr)
        except ValueError:
            pass
        if not rx_list and rr.rx_id in self._active_rx:
            del self._active_rx[rr.rx_id]

        collided = bool(rr.collided)
        per_drop = False
        rx_ok = True
        drop_reason = "ok"

        if rr.half_duplex_collision:
            drop_reason = "half_duplex_collision"

        if collided:
            rx_ok = False
            if rr.half_duplex_collision:
                self._inc_stat("phy_half_duplex_collisions")
            else:
                self._inc_stat("phy_collisions")
            if drop_reason == "ok":
                drop_reason = "collision"
        else:
            per = self._effective_per(rr.sinr_db, mode, float(flat_per))
            u = self.sim.engine.rng.random()
            if u < per:
                per_drop = True
                rx_ok = False
                self._inc_stat("phy_per_drops")
                drop_reason = "per_drop"

        rr.per_drop = bool(per_drop)
        rr.rx_ok = bool(rx_ok)
        rr.drop_reason = str(drop_reason)
        rr._rx_resolved = True

        interference_dbm = self._mw_to_dbm(interf_mw) if interf_mw > 0.0 else None

        self.sim.log(
            node_id=rr.rx_id,
            layer="PHY",
            event="RX_END",
            frame=rr.frame,
            details={
                "tx_id": rr.tx_id,
                "mode": mode,
                "rssi_dbm": rr.rssi_dbm,
                "noise_dbm": rr.noise_dbm,
                "snr_db": rr.snr_db,
                "sinr_db": rr.sinr_db,
                "interference_dbm": interference_dbm,
                "collided": collided,
                "half_duplex_collision": bool(rr.half_duplex_collision),
                "per_drop": per_drop,
                "rx_ok": rx_ok,
                "drop_reason": drop_reason,
                "flat_per_used": float(flat_per) > 0.0,
            },
        )

        rx_node = self.sim.nodes.get(rr.rx_id)
        # Last real, measured RSSI (dBm) this node's own receiver actually
        # saw, persisted so a visualisation can show "current" signal
        # strength for a node without needing to catch it mid-reception.
        # Event-driven like any real signal-strength meter -- it only
        # updates when this node actually receives something, which
        # happens at least every beacon interval for anything associated.
        if rx_node is not None:
            rx_phy = getattr(rx_node, "phy", None)
            if rx_phy is not None:
                rx_phy._last_rssi_dbm_value = float(rr.rssi_dbm)
                rx_phy._last_rssi_t = float(self.sim.engine.now)

        if rx_node is not None and getattr(rx_node, "mac", None) is not None:
            fn = getattr(rx_node.mac, "recv_up_from_phy", None)
            if callable(fn):
                fn(
                    rr.frame,
                    collided=bool(collided),
                    per_drop=bool(per_drop),
                    rx_ok=bool(rx_ok),
                )

    # --------------------------
    # Debug
    # --------------------------
    def debug_state(self) -> Dict[str, Any]:
        now = self.sim.engine.now
        active_air = self._get_global_active_air_tx()
        return {
            "node_id": int(self.node.node_id),
            "now": float(now),
            "active_air_tx_count": int(sum(1 for t in active_air.values() if t.start_time <= now < t.end_time)),
            "active_rx_count": int(len(self._active_rx.get(self.node.node_id, []))),
            "collision_model": self.collision_model,
            "cca_threshold_dbm": float(self.cca_threshold_dbm),
            "rx_sensitivity_dbm": float(self.rx_sensitivity_dbm),
            "eirp_dbm": float(self.eirp_dbm),
            "noise_floor_dbm": float(self._thermal_noise_dbm()),
            "shadow_cache_size": int(len(self._shadow_db)),
            "last_rssi_cache": {f"{k[0]}->{k[1]}": round(v, 1) for k, v in self._last_rssi_dbm.items()},
            "medium_tx_count": int(len(getattr(self.sim, "_medium_tx", []))),
            "channel_busy_self": bool(self.is_channel_busy(self.node.node_id)),
            "available_modes": list(self.available_modes),
            "supported_modes": list(sorted(self.supported_modes)),
            "default_mode": self.default_mode,
            "control_mode": self.control_mode,
            "shadow_symmetric": bool(self.shadow_symmetric),
        }