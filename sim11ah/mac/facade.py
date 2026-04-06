from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from sim11ah.constants import FrameType
from sim11ah.models import MacFrame, NetPDU, Packet

from sim11ah.mac.context import MacContext, build_mac_context
from sim11ah.mac.raw import RawEngine
from sim11ah.mac.dcf import DcfEngine
from sim11ah.mac.raw_metrics import MacMetrics


class MacLayer:
    """
    Public MAC facade.

    Keeps the same public API expected by the rest of the simulator while
    delegating internal responsibilities to submodules.
    """

    def __init__(self, node: "Node", cfg: Dict[str, Any]) -> None:
        self.ctx: MacContext = build_mac_context(node=node, cfg=cfg)

        self.node = self.ctx.node
        self.sim = self.ctx.sim
        self.cfg = self.ctx.cfg

        self.ctx._metrics = MacMetrics(self.ctx, self._log)

        self.raw = RawEngine(
            self.ctx,
            self._log,
            on_raw_enter_cb=self._on_raw_enter,
        )

        self.dcf = DcfEngine(self.ctx, self._log, self.raw)
        self.raw.init_from_cfg()

        # Stable duplicate cache for DATA frames across retransmissions
        if not hasattr(self.ctx, "_rx_data_key_cache"):
            self.ctx._rx_data_key_cache = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_ap(self) -> bool:
        return int(self.node.node_id) == 0

    def _extract_duplicate_key(self, frame: MacFrame) -> Tuple[int, str, int]:
        """
        Build a stable duplicate key for DATA reception.

        Preference order:
        1. (src, "net_seq", net_pdu.net_seq)
        2. (src, "pkt_seq", packet.packet_seq)
        3. (src, "frame_seq", frame.frame_seq) as fallback

        This avoids treating MAC retransmissions with a refreshed frame_seq
        as brand-new packets.
        """
        src = int(frame.src)

        try:
            if frame.net_pdu is not None:
                net_seq = getattr(frame.net_pdu, "net_seq", None)
                if net_seq is not None:
                    return (src, "net_seq", int(net_seq))
        except Exception:
            pass

        try:
            if frame.net_pdu is not None and frame.net_pdu.packet is not None:
                pkt_seq = getattr(frame.net_pdu.packet, "packet_seq", None)
                if pkt_seq is not None:
                    return (src, "pkt_seq", int(pkt_seq))
        except Exception:
            pass

        return (src, "frame_seq", int(frame.frame_seq))

    def _is_duplicate_data(self, frame: MacFrame) -> bool:
        """
        Duplicate suppression for received DATA.

        IMPORTANT:
        Retransmissions may carry a new MAC frame_seq, so duplicate detection
        must prefer a stable payload identity such as net_seq or packet_seq.
        """
        key = self._extract_duplicate_key(frame)

        cache_map = self.ctx._rx_data_key_cache
        src = key[0]

        seen = cache_map.setdefault(src, set())
        if key in seen:
            return True

        seen.add(key)

        maxsize = int(getattr(self.ctx, "_rx_cache_maxsize", 256))
        if len(seen) > maxsize:
            try:
                seen.pop()
            except Exception:
                pass

        return False

    def _on_raw_enter(self) -> None:
        self.dcf.drive(make_data_frame_cb=self._make_data_frame)

    def _schedule_ap_send_beacon(self) -> None:
        self._ap_send_beacon()

    def _schedule_send_ack(
        self,
        dst: int,
        ack_for_frame_seq: int,
        ack_for_frag_num: int = 0,
        deadline: Optional[float] = None,
    ) -> None:
        self._send_ack(dst, ack_for_frame_seq, ack_for_frag_num, deadline)

    def _phy_drop_reason(
        self,
        collided: bool,
        per_drop: bool,
        frame: Optional[MacFrame],
    ) -> str:
        if collided:
            return "collision"
        if per_drop:
            return "per_drop"
        if frame is not None and frame.dst not in (self.node.node_id, -1):
            return "not_for_me"
        return "phy_drop"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def post_build(self) -> None:
        pass

    def start(self) -> None:
        if self._is_ap():
            self.ap_start_beacons()

    def stop(self) -> None:
        pass

    def finalize(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(
        self,
        event: str,
        details: Dict[str, Any],
        frame: Optional[MacFrame] = None,
        net_pdu: Optional[NetPDU] = None,
        packet: Optional[Packet] = None,
    ) -> None:
        self.sim.log(
            node_id=self.node.node_id,
            layer="MAC",
            event=event,
            details=details,
            frame=frame,
            net_pdu=net_pdu,
            packet=packet,
        )

    # ------------------------------------------------------------------
    # Public MAC API
    # ------------------------------------------------------------------
    def send_down(self, net_pdu: NetPDU) -> None:
        self.dcf.send_down(
            net_pdu,
            on_queue_full_cb=getattr(self.node.net, "on_mac_queue_full", None),
            on_queue_depth_cb=getattr(self.node.net, "on_mac_queue_depth", None),
            make_data_frame_cb=self._make_data_frame,
            record_latency_cb=self._record_latency,
        )

    def recv_up_from_phy(
        self,
        frame: MacFrame,
        collided: bool,
        per_drop: bool,
        rx_ok: bool,
    ) -> None:
        if not rx_ok:
            self.ctx._stats["rx_crc_error"] += 1
            if "rx_phy_drops" not in self.ctx._stats:
                self.ctx._stats["rx_phy_drops"] = 0
            self.ctx._stats["rx_phy_drops"] += 1

            self._log(
                "RX_DROP",
                {
                    "reason": self._phy_drop_reason(collided, per_drop, frame),
                    "collided": bool(collided),
                    "per_drop": bool(per_drop),
                    "src": getattr(frame, "src", None),
                    "dst": getattr(frame, "dst", None),
                    "ftype": str(getattr(frame, "ftype", None)),
                },
                frame=frame,
            )
            return

        self._log(
            "RX_OK",
            {"ftype": str(frame.ftype), "src": frame.src, "dst": frame.dst},
            frame=frame,
        )

        if frame.ftype == FrameType.BEACON:
            self._handle_beacon(frame)
            return

        if frame.dst not in (self.node.node_id, -1):
            self._log(
                "RX_IGNORED_NOT_FOR_ME",
                {"src": frame.src, "dst": frame.dst, "ftype": str(frame.ftype)},
                frame=frame,
            )
            return

        if frame.ftype == FrameType.ACK:
            self.dcf.handle_ack_rx(frame, make_data_frame_cb=self._make_data_frame)
            return

        if frame.ftype == FrameType.DATA:
            is_dup = self._is_duplicate_data(frame)

            if is_dup:
                self.ctx._stats["rx_duplicate"] += 1
                self._log(
                    "RX_DUPLICATE",
                    {
                        "src": frame.src,
                        "frame_seq": frame.frame_seq,
                        "dup_key": self._extract_duplicate_key(frame),
                    },
                    frame=frame,
                )
            else:
                if frame.dst == -1:
                    self.ctx._stats["rx_broadcast"] += 1
                else:
                    self.ctx._stats["rx_unicast"] += 1

                if frame.net_pdu is not None:
                    self.node.net.recv_up_from_mac(frame.net_pdu)

            # ACK every successfully received unicast DATA frame, including duplicates.
            if frame.dst != -1:
                self.sim.engine.schedule_in(
                    self.ctx.sifs,
                    self._schedule_send_ack,
                    frame.src,
                    frame.frame_seq,
                    0,
                    None,
                    name="MAC_SEND_ACK",
                )
            return

        self._log(
            "RX_UNHANDLED_FRAME",
            {"ftype": str(frame.ftype), "src": frame.src, "dst": frame.dst},
            frame=frame,
        )

    def get_stats(self) -> Dict[str, Any]:
        s = dict(self.ctx._stats)
        n = s.get("latency_samples", 0)
        s["mean_latency_s"] = (s.get("total_latency_s", 0.0) / n) if n > 0 else 0.0
        return s

    def debug_queue_state(self) -> Dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "state": getattr(self.ctx.state, "name", str(self.ctx.state)),
            "assoc_state": self.ctx._assoc_state,
            "aid": self.ctx._aid,
            "q_len": len(self.ctx._txq),
            "pending": self.ctx._pending_frame is not None,
            "cw": self.ctx._cw,
            "bo_slots_left": self.ctx._backoff_slots_left,
            "saved_bo": self.ctx._saved_backoff,
            "short_retry": self.ctx._short_retry_count,
            "long_retry": self.ctx._long_retry_count,
            "nav_end": self.ctx._nav_end,
            "raw_enable": int(self.ctx.raw_enable),
            "raw_allowed": int(self.ctx.raw_allowed),
            "raw_slot": self.ctx.raw_assigned_slot,
            "raw_enter": self.ctx._raw_slot_enter_t,
            "raw_exit": self.ctx._raw_slot_exit_t,
            "raw_config_ids": [cfg.id for cfg in self.ctx._raw_configs if cfg.enabled],
            "twt_enable": int(self.ctx.twt_enable),
            "dozing": int(self.ctx._dozing),
            "ampdu_enable": int(self.ctx.ampdu_enable),
            "amsdu_enable": int(self.ctx.amsdu_enable),
            "q_occupancy": self.queue_occupancy(),
            "ch_util": self.channel_utilization(),
            "stats": self.get_stats(),
        }

    def debug_state(self) -> Dict[str, Any]:
        return self.debug_queue_state()

    def ap_start_beacons(self) -> None:
        if not self._is_ap():
            return

        self.ctx._ap_beacon_count = 0
        self.ctx._next_beacon_target = self.sim.engine.now
        self._log(
            "AP_BEACON_START",
            {
                "beacon_interval": self.ctx.beacon_interval,
                "dtim_period": self.ctx.dtim_period,
                "raw_enable": int(self.ctx.raw_enable),
            },
        )
        self.sim.engine.schedule(
            self.sim.engine.now,
            self._schedule_ap_send_beacon,
            name="MAC_AP_SEND_BEACON",
        )

    # ------------------------------------------------------------------
    # Beacon handling
    # ------------------------------------------------------------------
    def _ap_send_beacon(self) -> None:
        if self.node.phy.is_channel_busy(self.node.node_id):
            now = self.sim.engine.now
            t_idle = self.sim.medium_next_idle_time(now)
            delay = max(self.ctx.slot_time, t_idle - now)
            self._log("AP_BEACON_DEFER_BUSY", {"delay": delay})
            self.sim.engine.schedule_in(
                delay,
                self._schedule_ap_send_beacon,
                name="MAC_AP_SEND_BEACON",
            )
            return

        self.ctx._ap_beacon_count += 1
        is_dtim = (self.ctx._ap_beacon_count % self.ctx.dtim_period == 0)

        ctrl: Dict[str, Any] = {
            "beacon_count": self.ctx._ap_beacon_count,
            "is_dtim": bool(is_dtim),
            "dtim_period": int(self.ctx.dtim_period),
            "tim": dict(self.ctx._tim_bitmap),
        }

        if self.ctx.raw_enable and is_dtim:
            ctrl["rps"] = self.raw.build_rps()
            ctrl["raw_guard"] = float(self.ctx.raw_guard)

        self.ctx._tx_seq_ctr += 1
        beacon = MacFrame(
            ftype=FrameType.BEACON,
            src=0,
            dst=-1,
            size_bytes=self.ctx.beacon_size_bytes,
            frame_seq=self.sim.next_frame_seq(),
            tx_seq=self.ctx._tx_seq_ctr,
            retry=0,
            net_pdu=None,
            ctrl=ctrl,
        )

        self._log(
            "TX_BEACON",
            {
                "beacon_count": self.ctx._ap_beacon_count,
                "is_dtim": int(is_dtim),
                "rps_count": len(ctrl.get("rps", [])),
            },
            frame=beacon,
        )
        self.node.phy.send(beacon, tx_id=0, rx_id=-1)

        self.raw.update_periodic_after_beacon()

        self.ctx._next_beacon_target += self.ctx.beacon_interval
        delay = max(0.0, self.ctx._next_beacon_target - self.sim.engine.now)
        self.sim.engine.schedule_in(
            delay,
            self._schedule_ap_send_beacon,
            name="MAC_AP_SEND_BEACON",
        )

    def _handle_beacon(self, frame: MacFrame) -> None:
        if self._is_ap():
            return

        ctrl = frame.ctrl or {}
        is_dtim = bool(ctrl.get("is_dtim", False))

        self._log(
            "RX_BEACON",
            {
                "src": frame.src,
                "beacon_count": ctrl.get("beacon_count"),
                "is_dtim": int(is_dtim),
                "has_rps": int("rps" in ctrl),
            },
            frame=frame,
        )

        if self.ctx.raw_enable and is_dtim and "rps" in ctrl:
            self.raw.apply_rps(
                ctrl["rps"],
                float(ctrl.get("raw_guard", self.ctx.raw_guard)),
            )

        self.dcf.drive(make_data_frame_cb=self._make_data_frame)

    # ------------------------------------------------------------------
    # ACK TX
    # ------------------------------------------------------------------
    def _send_ack(
        self,
        dst: int,
        ack_for_frame_seq: int,
        ack_for_frag_num: int = 0,
        deadline: Optional[float] = None,
    ) -> None:
        now = self.sim.engine.now

        if deadline is None:
            deadline = now + 0.002  # 2 ms sanity bound

        if now > deadline:
            self._log(
                "ACK_DEADLINE_MISSED",
                {"dst": dst, "ack_for": ack_for_frame_seq, "frag": ack_for_frag_num},
            )
            return

        # ACK bypasses RAW admission, but avoid local PHY conflict.
        try:
            local_busy = self.node.phy.is_channel_busy(self.node.node_id)
        except Exception:
            local_busy = False

        if local_busy:
            delay = max(self.ctx.slot_time, 1e-6)
            self._log(
                "ACK_DEFER_LOCAL_BUSY",
                {
                    "dst": dst,
                    "ack_for": ack_for_frame_seq,
                    "frag": ack_for_frag_num,
                    "delay": delay,
                },
            )
            self.sim.engine.schedule_in(
                delay,
                self._schedule_send_ack,
                dst,
                ack_for_frame_seq,
                ack_for_frag_num,
                deadline,
                name="MAC_SEND_ACK",
            )
            return

        self.ctx._tx_seq_ctr += 1
        ack = MacFrame(
            ftype=FrameType.ACK,
            src=self.node.node_id,
            dst=dst,
            size_bytes=self.ctx.ack_size_bytes,
            frame_seq=self.sim.next_frame_seq(),
            tx_seq=self.ctx._tx_seq_ctr,
            retry=0,
            net_pdu=None,
            ctrl={
                "ack_for_frame_seq": ack_for_frame_seq,
                "ack_for_frag_num": int(ack_for_frag_num),
                "duration_s": 0.0,
            },
        )
        self._log(
            "TX_ACK",
            {"ack_for": ack_for_frame_seq, "frag": ack_for_frag_num},
            frame=ack,
        )
        self.node.phy.send(ack, tx_id=self.node.node_id, rx_id=dst)

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------
    def _compute_ack_tx_time(self, dst: int, mcs_str: str) -> float:
        if self.sim.topology is None:
            return (self.ctx.ack_size_bytes * 8) / 65_000.0

        try:
            lk = self.sim.topology.get_link(self.node.node_id, dst)
            rate = int(self.node.phy.mode_table.get(mcs_str, lk.rate_bps))
        except Exception:
            return (self.ctx.ack_size_bytes * 8) / 65_000.0

        stub = MacFrame(
            ftype=FrameType.ACK,
            src=0,
            dst=0,
            size_bytes=self.ctx.ack_size_bytes,
            frame_seq=0,
            tx_seq=0,
        )
        return float(self.node.phy.compute_tx_duration(stub, rate))

    def _make_data_frame(self, net_pdu: NetPDU, more_data: bool = False) -> MacFrame:
        self.ctx._tx_seq_ctr += 1
        net_hdr = int(self.cfg["net"]["net_header_bytes"])
        size_bytes = self.ctx.data_mac_overhead_bytes + net_hdr + net_pdu.packet.size_bytes
        mcs_str = self.node.phy.default_mode
        duration_s = self.ctx.sifs + self._compute_ack_tx_time(net_pdu.next_hop, mcs_str)

        return MacFrame(
            ftype=FrameType.DATA,
            src=self.node.node_id,
            dst=net_pdu.next_hop,
            size_bytes=size_bytes,
            frame_seq=self.sim.next_frame_seq(),
            tx_seq=self.ctx._tx_seq_ctr,
            retry=0,
            net_pdu=net_pdu,
            ctrl={
                "mcs": mcs_str,
                "duration_s": duration_s,
                "more_data": more_data,
                "power_mgmt": self.ctx._dozing,
                "color": 0,
            },
        )

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _record_latency(self, net_pdu: NetPDU, dropped: bool) -> None:
        enq = self.ctx._enqueue_time.pop(id(net_pdu), None)
        if enq is None:
            return

        latency = self.sim.engine.now - enq
        self.ctx._stats["total_latency_s"] += latency
        self.ctx._stats["latency_samples"] += 1

        self._log(
            "LATENCY",
            {"dropped": int(dropped), "latency_s": latency},
            net_pdu=net_pdu,
        )

    def queue_occupancy(self) -> float:
        return len(self.ctx._txq) / max(1, self.ctx.txq_max_depth)

    def channel_utilization(self) -> float:
        if not self.ctx._util_samples:
            return 0.0
        return sum(1 for _, b in self.ctx._util_samples if b) / len(self.ctx._util_samples)

    def get_publication_metrics(self, sim_duration_s: Optional[float] = None) -> Dict[str, Any]:
        if self.ctx._metrics is None:
            return {}
        return self.ctx._metrics.summary(sim_duration_s=sim_duration_s)