from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

from sim11ah.models import Packet


class TransportLayer:
    """
    Transport modes:
      - "udp"      : pass-through + metrics
      - "reorder"  : reorder buffer + timeout flush
      - "reliable" : sender-side RTO/flow-control bookkeeping, but requires
                     an external success/failure callback to fully close loop
    """

    _DSCP_MAP = {
        "sensor": 0,
        "control": 48,
        "alarm": 48,
        "telemetry": 8,
        "video": 32,
    }

    def __init__(self, node: "Node", cfg: Dict[str, Any]):
        self.node = node
        self.sim = node.sim
        self.cfg = cfg

        tp_cfg = cfg.get("transport", {})

        self.mode: str = str(tp_cfg.get("mode", "udp")).lower()

        self.assign_tp_seq: bool = bool(tp_cfg.get("assign_tp_seq", True))
        self.assign_created_at: bool = bool(tp_cfg.get("assign_created_at", True))
        self.seq_mod: int = int(tp_cfg.get("seq_mod", 65535))
        self._tx_seq_ctr: int = 0

        self.per_dst_seq: bool = bool(tp_cfg.get("per_dst_seq", False))
        self._tx_seq_per_dst: Dict[int, int] = {}

        self.enable_dscp_marking: bool = bool(tp_cfg.get("enable_dscp_marking", True))

        self._seen_cache_max: int = int(tp_cfg.get("seen_cache_max", 256))
        self._seen_tp_seqs: Dict[int, set] = {}

        self.reorder_window: int = int(tp_cfg.get("reorder_window", 16))
        self.reorder_timeout: float = float(tp_cfg.get("reorder_timeout", 2.0))
        self._rx_expected_seq: Dict[int, int] = {}
        self._rx_reorder_buf: Dict[int, Dict[int, Packet]] = {}

        self.expected_interval_s: float = float(tp_cfg.get("expected_interval_s", 0.0))
        self._last_delivery_time: Dict[int, float] = {}
        self._jitter_per_src: Dict[int, List[float]] = {}

        self.goodput_window_s: float = float(tp_cfg.get("goodput_window_s", 1.0))
        self._bytes_delivered: int = 0
        self._goodput_window_start: Optional[float] = None

        self.tx_window_size: int = int(tp_cfg.get("tx_window_size", 0))
        self.max_pending: int = int(tp_cfg.get("max_pending", 64))
        self.tx_in_flight: int = 0
        self._tx_pending: List[Packet] = []

        self.rto_base_s: float = float(tp_cfg.get("rto_base_s", 1.0))
        self.rto_max_s: float = float(tp_cfg.get("rto_max_s", 10.0))
        self.rto_backoff: float = float(tp_cfg.get("rto_backoff", 2.0))
        self.max_retransmissions: int = int(tp_cfg.get("max_retransmissions", 3))

        # tp_seq -> (send_time, packet, rto_s, retx_count)
        self._unacked: Dict[int, Tuple[float, Packet, float, int]] = {}

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _log(self, event: str, details: Dict[str, Any], packet: Optional[Packet] = None) -> None:
        self.sim.log(
            node_id=self.node.node_id,
            layer="TP",
            event=event,
            details=details,
            packet=packet,
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _ensure_meta(self, pkt: Packet) -> Dict[str, Any]:
        try:
            meta = getattr(pkt, "meta", None)
            if meta is None or not isinstance(meta, dict):
                pkt.meta = {}
            return pkt.meta
        except Exception:
            return {}

    def _mark_failed_once(self, packet: Packet) -> bool:
        meta = self._ensure_meta(packet)
        if meta.get("_tp_succeeded", False):
            return False
        if meta.get("_tp_failed", False):
            return False
        meta["_tp_failed"] = True
        return True

    def _mark_succeeded(self, packet: Packet) -> None:
        meta = self._ensure_meta(packet)
        meta["_tp_succeeded"] = True

    def _next_tp_seq(self, dst: Optional[int] = None) -> int:
        if self.per_dst_seq and dst is not None:
            cur = int(self._tx_seq_per_dst.get(dst, 0))
            cur = (cur % self.seq_mod) + 1
            self._tx_seq_per_dst[dst] = cur
            return cur

        self._tx_seq_ctr = (self._tx_seq_ctr % self.seq_mod) + 1
        return self._tx_seq_ctr

    def _is_duplicate_rx(self, src: int, tp_seq: int) -> bool:
        seen = self._seen_tp_seqs.setdefault(src, set())
        if tp_seq in seen:
            return True
        seen.add(tp_seq)
        if len(seen) > self._seen_cache_max:
            try:
                evict = sorted(seen)[: max(1, self._seen_cache_max // 2)]
                for s in evict:
                    seen.discard(s)
            except Exception:
                seen.clear()
                seen.add(tp_seq)
        return False

    def _update_stats_if_present(self, name: str, updater) -> None:
        try:
            st = getattr(self.sim, "stats", None)
            if st is None or not hasattr(st, name):
                return
            updater(st)
        except Exception:
            pass

    def _schedule_reorder_timeout_event(self, src: int, marker_seq: int) -> None:
        self._reorder_timeout(src, marker_seq)

    def _schedule_rto_event(self, tp_seq: int) -> None:
        self._rto_fired(tp_seq)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def _compute_e2e_latency(self, pkt: Packet) -> Optional[float]:
        created_at = getattr(pkt, "created_at", None)
        if created_at is None:
            return None
        try:
            return float(self.sim.engine.now) - float(created_at)
        except Exception:
            return None

    def _update_jitter(self, pkt: Packet) -> Optional[float]:
        try:
            now = float(self.sim.engine.now)
            src = int(getattr(pkt, "src", -1))
            last = self._last_delivery_time.get(src)
            self._last_delivery_time[src] = now
            if last is None:
                return None
            inter = now - float(last)
            jitter = abs(inter - float(self.expected_interval_s))
            self._jitter_per_src.setdefault(src, []).append(jitter)
            self._update_stats_if_present(
                "jitter_per_node",
                lambda st: st.jitter_per_node.setdefault(self.node.node_id, []).append(jitter),
            )
            return jitter
        except Exception:
            return None

    def _update_goodput(self, pkt: Packet) -> Optional[float]:
        try:
            now = float(self.sim.engine.now)
            if self._goodput_window_start is None:
                self._goodput_window_start = now

            sz = int(getattr(pkt, "size_bytes", 0))
            self._bytes_delivered += max(0, sz)

            elapsed = now - float(self._goodput_window_start)
            if elapsed < float(self.goodput_window_s):
                return None

            goodput_bps = (self._bytes_delivered * 8.0) / max(elapsed, 1e-12)

            self._update_stats_if_present(
                "goodput_bps_per_node",
                lambda st: st.goodput_bps_per_node.__setitem__(self.node.node_id, goodput_bps),
            )

            self._bytes_delivered = 0
            self._goodput_window_start = now
            return goodput_bps
        except Exception:
            return None

    # --------------------------------------------------
    # Up delivery
    # --------------------------------------------------

    def _deliver_up(self, packet: Packet) -> None:
        e2e = self._compute_e2e_latency(packet)
        jitter = self._update_jitter(packet)
        goodput = self._update_goodput(packet)

        details = {
            "src": getattr(packet, "src", None),
            "dst": getattr(packet, "dst", None),
            "tp_seq": getattr(packet, "tp_seq", None),
            "size_bytes": getattr(packet, "size_bytes", None),
            "dscp": getattr(packet, "dscp", None),
        }
        if e2e is not None:
            details["e2e_s"] = round(e2e, 6)
        if jitter is not None:
            details["jitter_s"] = round(jitter, 6)
        if goodput is not None:
            details["goodput_bps"] = round(goodput, 1)

        self._log("DELIVER", details, packet=packet)
        self.node.app.recv_up_from_transport(packet)

    # --------------------------------------------------
    # Reorder buffer
    # --------------------------------------------------

    def _schedule_reorder_timeout(self, src: int, marker_seq: int) -> None:
        try:
            self.sim.engine.schedule_in(
                self.reorder_timeout,
                self._schedule_reorder_timeout_event,
                src,
                marker_seq,
                name="TP_REORDER_TIMEOUT",
            )
        except Exception:
            pass

    def _reorder_timeout(self, src: int, marker_seq: int) -> None:
        expected = self._rx_expected_seq.get(src, 1)
        buf = self._rx_reorder_buf.get(src, {})
        if not buf:
            return
        if expected <= marker_seq:
            self._log("REORDER_TIMEOUT", {"src": src, "expected": expected, "buffered": len(buf)})
            self._flush_reorder_buffer(src)

    def _drain_reorder_buffer(self, src: int) -> None:
        buf = self._rx_reorder_buf.get(src, {})
        expected = self._rx_expected_seq.get(src, 1)
        while expected in buf:
            pkt = buf.pop(expected)
            self._deliver_up(pkt)
            expected += 1
        self._rx_expected_seq[src] = expected
        if not buf:
            self._rx_reorder_buf.pop(src, None)

    def _flush_reorder_buffer(self, src: int) -> None:
        buf = self._rx_reorder_buf.pop(src, {})
        if not buf:
            return
        for seq in sorted(buf.keys()):
            self._deliver_up(buf[seq])
        try:
            self._rx_expected_seq[src] = max(buf.keys()) + 1
        except Exception:
            pass

    # --------------------------------------------------
    # Sender-side RTO / flow control
    # --------------------------------------------------

    def _drain_pending(self) -> None:
        while self._tx_pending and (self.tx_window_size <= 0 or self.tx_in_flight < self.tx_window_size):
            pkt = self._tx_pending.pop(0)
            self._do_send(pkt)

    def _start_rto(self, packet: Packet) -> None:
        if self.mode != "reliable":
            return
        tp_seq = getattr(packet, "tp_seq", None)
        if tp_seq is None:
            return
        tp_seq = int(tp_seq)
        now = float(self.sim.engine.now)
        self._unacked[tp_seq] = (now, packet, float(self.rto_base_s), 0)
        try:
            self.sim.engine.schedule_in(
                float(self.rto_base_s),
                self._schedule_rto_event,
                tp_seq,
                name="TP_RTO",
            )
        except Exception:
            pass

    def _cancel_rto(self, tp_seq: int) -> None:
        self._unacked.pop(tp_seq, None)

    def _rto_fired(self, tp_seq: int) -> None:
        entry = self._unacked.get(tp_seq)
        if entry is None:
            return

        _send_time, packet, rto_s, retx = entry
        retx += 1

        if retx > self.max_retransmissions:
            self._unacked.pop(tp_seq, None)
            self._log("DROP", {"reason": "rto_max_retx", "tp_seq": tp_seq, "retx": retx}, packet=packet)

            if self.tx_window_size > 0:
                self.tx_in_flight = max(0, self.tx_in_flight - 1)
                self._drain_pending()

            if self._mark_failed_once(packet):
                notify = getattr(self.node.app, "on_tx_failure", None)
                if callable(notify):
                    notify(packet)
            return

        new_rto = min(float(self.rto_max_s), float(rto_s) * float(self.rto_backoff))
        self._unacked[tp_seq] = (float(self.sim.engine.now), packet, new_rto, retx)

        self._log("RTO_RETX", {"tp_seq": tp_seq, "retx": retx, "rto_s": round(new_rto, 3)}, packet=packet)
        self.node.net.send_down(packet)

        try:
            self.sim.engine.schedule_in(
                new_rto,
                self._schedule_rto_event,
                tp_seq,
                name="TP_RTO",
            )
        except Exception:
            pass

    def _do_send(self, packet: Packet) -> None:
        if self.tx_window_size > 0:
            self.tx_in_flight += 1

        self._log(
            "TX_DOWN",
            {
                "src": getattr(packet, "src", None),
                "dst": getattr(packet, "dst", None),
                "tp_seq": getattr(packet, "tp_seq", None),
                "size_bytes": getattr(packet, "size_bytes", None),
                "dscp": getattr(packet, "dscp", None),
                "in_flight": self.tx_in_flight if self.tx_window_size > 0 else None,
                "pending": len(self._tx_pending),
            },
            packet=packet,
        )

        self.node.net.send_down(packet)
        self._start_rto(packet)

    # --------------------------------------------------
    # Public TX API
    # --------------------------------------------------

    def send_down(self, packet: Packet) -> None:
        self._ensure_meta(packet)

        if self.enable_dscp_marking and getattr(packet, "dscp", None) is None:
            traffic_type = getattr(packet, "traffic_type", "sensor")
            packet.dscp = self._DSCP_MAP.get(str(traffic_type), 0)

        if self.assign_tp_seq and getattr(packet, "tp_seq", None) is None:
            dst = getattr(packet, "dst", None)
            packet.tp_seq = self._next_tp_seq(dst=int(dst) if dst is not None else None)

        if self.assign_created_at and getattr(packet, "created_at", None) is None:
            packet.created_at = self.sim.engine.now

        if self.tx_window_size > 0 and self.tx_in_flight >= self.tx_window_size:
            if len(self._tx_pending) >= self.max_pending:
                self._log(
                    "DROP",
                    {"reason": "tx_window_full", "in_flight": self.tx_in_flight, "pending": len(self._tx_pending)},
                    packet=packet,
                )
                if self._mark_failed_once(packet):
                    notify = getattr(self.node.app, "on_tx_failure", None)
                    if callable(notify):
                        notify(packet)
                return

            self._tx_pending.append(packet)
            self._log(
                "TX_QUEUED",
                {
                    "in_flight": self.tx_in_flight,
                    "pending": len(self._tx_pending),
                    "tp_seq": getattr(packet, "tp_seq", None),
                },
                packet=packet,
            )
            return

        self._do_send(packet)

    # --------------------------------------------------
    # Public RX API
    # --------------------------------------------------

    def recv_up_from_net(self, packet: Packet) -> None:
        src = int(getattr(packet, "src", -1))
        tp_seq = getattr(packet, "tp_seq", None)

        self._log(
            "RX_UP",
            {
                "src": getattr(packet, "src", None),
                "dst": getattr(packet, "dst", None),
                "tp_seq": tp_seq,
                "size_bytes": getattr(packet, "size_bytes", None),
            },
            packet=packet,
        )

        if tp_seq is None or self.mode not in ("reorder", "reliable"):
            self._deliver_up(packet)
            return

        tp_seq = int(tp_seq)

        if self._is_duplicate_rx(src, tp_seq):
            self._log("DROP", {"reason": "duplicate_tp_seq", "src": src, "tp_seq": tp_seq}, packet=packet)
            self._update_stats_if_present(
                "transport_duplicates",
                lambda st: setattr(st, "transport_duplicates", int(getattr(st, "transport_duplicates", 0)) + 1),
            )
            return

        expected = self._rx_expected_seq.get(src, 1)

        if tp_seq == expected:
            self._deliver_up(packet)
            self._rx_expected_seq[src] = expected + 1
            self._drain_reorder_buffer(src)
            return

        if tp_seq > expected:
            buf = self._rx_reorder_buf.setdefault(src, {})
            if len(buf) >= self.reorder_window:
                self._log(
                    "REORDER_FLUSH",
                    {"reason": "window_full", "src": src, "expected": expected, "buffered": len(buf)},
                )
                self._flush_reorder_buffer(src)
                self._deliver_up(packet)
                self._rx_expected_seq[src] = tp_seq + 1
                return

            buf[tp_seq] = packet
            self._log(
                "REORDER_BUFFER",
                {"src": src, "tp_seq": tp_seq, "expected": expected, "buffered": len(buf)},
                packet=packet,
            )
            self._schedule_reorder_timeout(src, tp_seq)
            return

        self._log("DROP", {"reason": "late_tp_seq", "src": src, "tp_seq": tp_seq, "expected": expected}, packet=packet)

    # --------------------------------------------------
    # External success/failure hooks
    # --------------------------------------------------

    def notify_tx_success(self, packet: Packet) -> None:
        tp_seq = getattr(packet, "tp_seq", None)

        self._log(
            "TX_SUCCESS",
            {
                "tp_seq": tp_seq,
                "dst": getattr(packet, "dst", None),
                "in_flight": self.tx_in_flight if self.tx_window_size > 0 else None,
            },
            packet=packet,
        )

        self._mark_succeeded(packet)

        if tp_seq is not None:
            try:
                self._cancel_rto(int(tp_seq))
            except Exception:
                pass

        if self.tx_window_size > 0:
            self.tx_in_flight = max(0, self.tx_in_flight - 1)
            self._drain_pending()

    def notify_tx_failure(self, packet: Packet) -> None:
        tp_seq = getattr(packet, "tp_seq", None)

        self._log(
            "TX_FAILURE",
            {
                "tp_seq": tp_seq,
                "dst": getattr(packet, "dst", None),
                "reason": "mac_retry_exhausted",
                "in_flight": self.tx_in_flight if self.tx_window_size > 0 else None,
            },
            packet=packet,
        )

        if tp_seq is not None:
            try:
                self._cancel_rto(int(tp_seq))
            except Exception:
                pass

        if self.tx_window_size > 0:
            self.tx_in_flight = max(0, self.tx_in_flight - 1)
            self._drain_pending()

        if self._mark_failed_once(packet):
            notify = getattr(self.node.app, "on_tx_failure", None)
            if callable(notify):
                notify(packet)

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def debug_state(self) -> Dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "mode": self.mode,
            "tx_seq_ctr": self._tx_seq_ctr,
            "per_dst_seq": self.per_dst_seq,
            "tx_in_flight": self.tx_in_flight,
            "pending_tx": len(self._tx_pending),
            "unacked": len(self._unacked),
            "reorder_srcs": {
                src: {"expected": self._rx_expected_seq.get(src, 1), "buffered": len(buf)}
                for src, buf in self._rx_reorder_buf.items()
            },
            "seen_cache": {src: len(s) for src, s in self._seen_tp_seqs.items()},
        }