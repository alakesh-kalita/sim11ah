from __future__ import annotations

from sim11ah.constants import FrameType, MacState
from sim11ah.models import MacFrame, NetPDU

from sim11ah.mac.context import MacContext
from sim11ah.mac.raw import RawEngine
from sim11ah.mac.context import record_queue_hint


class DcfEngine:
    """
    Core DCF/CSMA-CA engine migrated out of the monolithic MAC.

    Updated for:
      - clearer event tracing
      - better stats/logging for performance evaluation
      - more consistent cleanup paths
      - synchronized updates to sim.stats for summary reporting
      - safer RAW-boundary / ACK-timeout handling
    """

    def __init__(self, ctx: MacContext, log_fn, raw_engine: RawEngine) -> None:
        self.ctx = ctx
        self._log = log_fn
        self.raw = raw_engine
        self._ensure_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_stats(self) -> None:
        defaults = {
            "backoff_starts": 0,
            "tx_attempts": 0,
            "ack_rx": 0,
            "ack_timeouts": 0,
            "retry_limit_drops": 0,
            "pending_ttl_expired": 0,
        }
        for k, v in defaults.items():
            if k not in self.ctx._stats:
                self.ctx._stats[k] = v

    def _inc_stat(self, key: str, amount: int = 1) -> None:
        amount = int(amount)

        if key not in self.ctx._stats:
            self.ctx._stats[key] = 0
        self.ctx._stats[key] += amount

        # Keep global sim.stats aligned with MAC-local counters
        try:
            st = self.ctx.sim.stats

            if key == "tx_attempts":
                st.mac_tx_attempts += amount
            elif key == "ack_timeouts":
                st.mac_ack_timeouts += amount
            elif key == "retry_limit_drops":
                # final MAC failure => real packet drop
                st.packets_dropped += amount
        except Exception:
            pass

    def _inc_mac_retry(self, amount: int = 1) -> None:
        try:
            self.ctx.sim.stats.mac_retries += int(amount)
        except Exception:
            pass

    def _inc_packet_drop(self, amount: int = 1) -> None:
        try:
            self.ctx.sim.stats.packets_dropped += int(amount)
        except Exception:
            pass

    def _metric_sta_id(self, frame: MacFrame) -> int:
        try:
            aid = getattr(self.ctx, "_aid", None)
            if isinstance(aid, int) and aid > 0:
                return aid
        except Exception:
            pass
        return int(frame.src)

    def _schedule_backoff_tick_event(self, *, make_data_frame_cb=None) -> None:
        self._backoff_tick(make_data_frame_cb=make_data_frame_cb)

    def _schedule_ack_timeout_event(self, frame_seq: int, token: int, *, make_data_frame_cb=None) -> None:
        self._ack_timeout(frame_seq, token, make_data_frame_cb=make_data_frame_cb)

    def _schedule_next_fragment_event(self, *, make_data_frame_cb=None) -> None:
        self._tx_next_fragment(make_data_frame_cb=make_data_frame_cb)

    def _clear_wait_ack_state(self) -> None:
        self.ctx._waiting_ack_for_frame_seq = None
        self.ctx._waiting_ack_for_frag_num = None

    def _clear_fragment_state(self) -> None:
        self.ctx._pending_fragments = []
        self.ctx._pending_frag_index = 0

    def _clear_backoff_state(self) -> None:
        self.ctx._backoff_slots_left = None
        self.ctx._saved_backoff = None
        self.ctx._idle_since = None

    def _reset_after_completion(self) -> None:
        self.ctx._pending_frame = None
        self._clear_wait_ack_state()
        self._clear_fragment_state()
        self._clear_backoff_state()
        self._reset_retry_counters()
        self.ctx._cw = self.ctx.cw_min
        self.ctx.state = MacState.IDLE

    # ------------------------------------------------------------------
    # Helpers: upper-layer notifications
    # ------------------------------------------------------------------
    def _notify_tx_success(self, pkt) -> None:
        if pkt is None:
            return

        try:
            self.ctx.node.transport.notify_tx_success(pkt)
        except Exception:
            pass

        try:
            self.ctx.node.app.on_tx_success(pkt)
        except Exception:
            pass

    def _notify_tx_failure(self, pkt) -> None:
        if pkt is None:
            return

        try:
            self.ctx.node.transport.notify_tx_failure(pkt)
        except Exception:
            pass

        try:
            self.ctx.node.app.on_tx_failure(pkt)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def send_down(
        self,
        net_pdu: NetPDU,
        *,
        on_queue_full_cb=None,
        on_queue_depth_cb=None,
        make_data_frame_cb=None,
        record_latency_cb=None,
    ) -> None:
        self._ensure_stats()

        if len(self.ctx._txq) >= self.ctx.txq_max_depth:
            self.ctx._stats["txq_tail_drops"] += 1
            self._inc_packet_drop(1)

            self._log(
                "TXQUEUE_FULL_DROP",
                {
                    "q_len": len(self.ctx._txq),
                    "max": self.ctx.txq_max_depth,
                    "dst": net_pdu.next_hop,
                },
                net_pdu=net_pdu,
            )
            if on_queue_full_cb is not None:
                on_queue_full_cb(net_pdu)
            return

        self.ctx._txq.append(net_pdu)
        self.ctx._enqueue_time[id(net_pdu)] = self.ctx.sim.engine.now
        record_queue_hint(self.ctx, net_pdu.next_hop)

        if on_queue_depth_cb is not None:
            on_queue_depth_cb(len(self.ctx._txq), self.ctx.txq_max_depth)

        self._log(
            "ENQUEUE",
            {"q_len": len(self.ctx._txq), "dst": net_pdu.next_hop},
            net_pdu=net_pdu,
        )

        self._expire_queue(record_latency_cb=record_latency_cb)
        self.drive(make_data_frame_cb=make_data_frame_cb, record_latency_cb=record_latency_cb)

    # ------------------------------------------------------------------
    # Medium / guards
    # ------------------------------------------------------------------
    def is_busy(self) -> bool:
        now = self.ctx.sim.engine.now
        if self.ctx._nav_end > 0.0 and now >= self.ctx._nav_end:
            self._log("NAV_EXPIRED", {"was": self.ctx._nav_end})
            self.ctx._nav_end = 0.0
        if now < self.ctx._nav_end:
            return True
        return self.ctx.node.phy.is_channel_busy(self.ctx.node.node_id)

    def can_tx_now(self) -> bool:
        if self.ctx.node.node_id == 0:
            return True
        if self.ctx._dozing:
            return False
        if not self.ctx.raw_enable:
            return True
        return self.ctx.raw_allowed

    # ------------------------------------------------------------------
    # Core drive
    # ------------------------------------------------------------------
    def drive(self, *, make_data_frame_cb=None, record_latency_cb=None) -> None:
        self._expire_queue(record_latency_cb=record_latency_cb)

        if not self.can_tx_now():
            if self.ctx.state != MacState.WAIT_ACK:
                self.ctx.state = MacState.RAW_SLEEP
            return

        if self.ctx.state not in (MacState.IDLE, MacState.BACKOFF):
            return

        if self.ctx._pending_frame is None and self.ctx._txq:
            pdu = self.ctx._txq[0]
            if self.ctx._last_pending_pdu is not pdu:
                self._reset_retry_counters()
                self.ctx._last_pending_pdu = pdu
            if make_data_frame_cb is None:
                raise RuntimeError("make_data_frame_cb is required")
            self.ctx._pending_frame = make_data_frame_cb(pdu)

        if self.ctx._pending_frame is None:
            self.ctx.state = MacState.IDLE
            return

        self._start_or_continue_backoff(
            make_data_frame_cb=make_data_frame_cb,
            record_latency_cb=record_latency_cb,
        )

    def _start_or_continue_backoff(self, *, make_data_frame_cb=None, record_latency_cb=None) -> None:
        if not self.can_tx_now():
            self.drive(make_data_frame_cb=make_data_frame_cb, record_latency_cb=record_latency_cb)
            return

        if self.ctx.state not in (MacState.IDLE, MacState.BACKOFF):
            return

        if self.ctx._backoff_slots_left is None:
            if self.ctx._saved_backoff is not None:
                self.ctx._backoff_slots_left = self.ctx._saved_backoff
                self.ctx._saved_backoff = None
            else:
                self.ctx._backoff_slots_left = int(self.ctx.sim.engine.rng.randint(0, self.ctx._cw))

            self.ctx.state = MacState.BACKOFF
            self.ctx._idle_since = None
            self._inc_stat("backoff_starts")
            self._log(
                "BACKOFF_START",
                {"cw": self.ctx._cw, "slots": self.ctx._backoff_slots_left},
                frame=self.ctx._pending_frame,
            )

        self._schedule_backoff_tick(
            delay=self.ctx.slot_time,
            make_data_frame_cb=make_data_frame_cb,
        )

    def _schedule_backoff_tick(self, delay: float, *, make_data_frame_cb=None) -> None:
        if not self.ctx._backoff_tick_scheduled:
            self.ctx._backoff_tick_scheduled = True
            self.ctx.sim.engine.schedule_in(
                delay,
                self._schedule_backoff_tick_event,
                make_data_frame_cb=make_data_frame_cb,
                name="DCF_BACKOFF_TICK",
            )

    def _backoff_tick(self, *, make_data_frame_cb=None) -> None:
        self.ctx._backoff_tick_scheduled = False
        self._sample_utilization()

        if not self.can_tx_now():
            self.ctx._saved_backoff = self.ctx._backoff_slots_left
            self.ctx._backoff_slots_left = None
            self.ctx.state = MacState.RAW_SLEEP
            self.ctx._idle_since = None
            self._log(
                "BACKOFF_ABORT_BOUNDARY",
                {"slots_left": self.ctx._saved_backoff},
                frame=self.ctx._pending_frame,
            )
            return

        if self.ctx._pending_frame is None:
            self.ctx._backoff_slots_left = None
            self.ctx.state = MacState.IDLE
            self.ctx._idle_since = None
            return

        if self.is_busy():
            self.ctx._stats["backoff_freezes"] += 1
            self.ctx._idle_since = None
            now = self.ctx.sim.engine.now
            t_idle = self.ctx.sim.medium_next_idle_time(now)
            delay = max(self.ctx.slot_time, t_idle - now)
            self._log(
                "BACKOFF_FREEZE",
                {"slots_left": self.ctx._backoff_slots_left},
                frame=self.ctx._pending_frame,
            )
            self._schedule_backoff_tick(delay=delay, make_data_frame_cb=make_data_frame_cb)
            return

        now = self.ctx.sim.engine.now
        if self.ctx._idle_since is None:
            self.ctx._idle_since = now
            self._schedule_backoff_tick(delay=self.ctx.slot_time, make_data_frame_cb=make_data_frame_cb)
            return

        if (now - self.ctx._idle_since) < self.ctx.difs:
            self._schedule_backoff_tick(delay=self.ctx.slot_time, make_data_frame_cb=make_data_frame_cb)
            return

        slots = int(self.ctx._backoff_slots_left or 0)
        if slots > 0:
            self.ctx._backoff_slots_left = slots - 1
            self._schedule_backoff_tick(delay=self.ctx.slot_time, make_data_frame_cb=make_data_frame_cb)
            return

        self.ctx._backoff_slots_left = None
        self.ctx._idle_since = None
        self._tx_attempt(make_data_frame_cb=make_data_frame_cb)

    # ------------------------------------------------------------------
    # TX attempt
    # ------------------------------------------------------------------
    def _tx_attempt(self, *, make_data_frame_cb=None) -> None:
        if self.ctx._pending_frame is None:
            self.ctx.state = MacState.IDLE
            return

        if not self.can_tx_now():
            self.ctx.state = MacState.RAW_SLEEP
            return

        if self.is_busy():
            self.ctx.state = MacState.BACKOFF
            self._start_or_continue_backoff(make_data_frame_cb=make_data_frame_cb)
            return

        fr = self.ctx._pending_frame

        if fr.ftype == FrameType.DATA and fr.dst != -1:
            ok = self.raw.exchange_would_fit_in_raw(
                fr,
                compute_data_tx_time=self._compute_data_tx_time,
                compute_ack_tx_time=self._compute_ack_tx_time,
                get_prop=self._get_prop,
            )
            if not ok:
                self.ctx._stats["raw_deferred"] += 1

                # Preserve remaining backoff if any; otherwise keep a zero-slot resume point.
                if self.ctx._backoff_slots_left is not None:
                    self.ctx._saved_backoff = self.ctx._backoff_slots_left
                    self.ctx._backoff_slots_left = None
                elif self.ctx._saved_backoff is None:
                    self.ctx._saved_backoff = 0

                self.ctx._idle_since = None

                # Stay logically pending, but wait for next RAW opportunity.
                if self.ctx.state != MacState.WAIT_ACK:
                    self.ctx.state = MacState.RAW_SLEEP

                self._log(
                    "RAW_DEFER_NO_FIT",
                    {
                        "slot_exit": self.ctx._raw_slot_exit_t,
                        "saved_backoff": self.ctx._saved_backoff,
                        "frame_seq": fr.frame_seq,
                        "retry": fr.retry,
                    },
                    frame=fr,
                )
                return

        # Final guard against transmitting too close to RAW-slot end.
        if self.ctx.raw_enable and self.ctx.node.node_id != 0 and fr.ftype == FrameType.DATA and fr.dst != -1:
            now = self.ctx.sim.engine.now
            remaining = self.ctx._raw_slot_exit_t - now if self.ctx._raw_slot_exit_t > 0.0 else float("inf")
            final_margin = max(self.ctx.slot_time, self.ctx.ack_guard)

            if remaining <= final_margin:
                self.ctx._stats["raw_deferred"] += 1
                if self.ctx._saved_backoff is None:
                    self.ctx._saved_backoff = 0
                self.ctx.state = MacState.RAW_SLEEP
                self._log(
                    "RAW_DEFER_FINAL_MARGIN",
                    {
                        "remaining": remaining,
                        "final_margin": final_margin,
                        "slot_exit": self.ctx._raw_slot_exit_t,
                    },
                    frame=fr,
                )
                return

        self.ctx.state = MacState.TX
        self._inc_stat("tx_attempts")
        self._log("TX_ATTEMPT", {"cw": self.ctx._cw, "retry": fr.retry}, frame=fr)

        if fr.dst == -1:
            self.ctx._stats["tx_broadcast"] += 1
        else:
            self.ctx._stats["tx_unicast"] += 1

        if self.ctx._metrics is not None and fr.dst != -1:
            sta_id = self._metric_sta_id(fr)
            self.ctx._metrics.record_tx_attempt(sta_id, fr.size_bytes, src=sta_id, dst=fr.dst)
            self.ctx._metrics.record_raw_tx(
                now=self.ctx.sim.engine.now,
                src=sta_id,
                size_bytes=fr.size_bytes,
                success=False,
                retry=False,
                timeout=False,
            )

        self.ctx.node.phy.send(fr, tx_id=self.ctx.node.node_id, rx_id=fr.dst)

        if fr.ftype == FrameType.DATA and fr.dst != -1:
            self.ctx.state = MacState.WAIT_ACK
            self.ctx._waiting_ack_for_frame_seq = fr.frame_seq
            self.ctx._waiting_ack_for_frag_num = 0
            timeout_s = self._compute_ack_timeout(fr)
            self._arm_ack_timeout(fr.frame_seq, timeout_s, make_data_frame_cb=make_data_frame_cb)
        else:
            self._on_success(make_data_frame_cb=make_data_frame_cb)

    # ------------------------------------------------------------------
    # ACK RX / timeout
    # ------------------------------------------------------------------
    def handle_ack_rx(self, frame: MacFrame, *, make_data_frame_cb=None) -> None:
        if self.ctx.state != MacState.WAIT_ACK:
            return

        ctrl = frame.ctrl or {}
        ack_seq = ctrl.get("ack_for_frame_seq")
        ack_frag = int(ctrl.get("ack_for_frag_num", 0))

        if ack_seq is None:
            return
        if self.ctx._waiting_ack_for_frame_seq != ack_seq:
            return

        if self.ctx._pending_fragments:
            expected_frag = int(self.ctx._waiting_ack_for_frag_num or 0)
            if ack_frag != expected_frag:
                return
            self._inc_stat("ack_rx")
            self._log("ACK_RX_FRAG", {"seq": ack_seq, "frag": ack_frag}, frame=frame)
            self.ctx._ack_timer_token += 1
            self._on_fragment_ack(make_data_frame_cb=make_data_frame_cb)
            return

        self._inc_stat("ack_rx")
        self._log("ACK_RX", {"ack_for": ack_seq}, frame=frame)
        self.ctx._ack_timer_token += 1
        self._on_success(make_data_frame_cb=make_data_frame_cb)

    def _arm_ack_timeout(self, frame_seq: int, timeout_s: float, *, make_data_frame_cb=None) -> None:
        self.ctx._ack_timer_token += 1
        token = self.ctx._ack_timer_token
        self._log(
            "ACK_TIMEOUT_SET",
            {"timeout_s": timeout_s, "token": token},
            frame=self.ctx._pending_frame,
        )
        self.ctx.sim.engine.schedule_in(
            timeout_s,
            self._schedule_ack_timeout_event,
            frame_seq,
            token,
            make_data_frame_cb=make_data_frame_cb,
            name="DCF_ACK_TIMEOUT",
        )

    def _ack_timeout(self, frame_seq: int, token: int, *, make_data_frame_cb=None) -> None:
        if token != self.ctx._ack_timer_token:
            return
        if self.ctx.state != MacState.WAIT_ACK:
            return
        if self.ctx._waiting_ack_for_frame_seq != frame_seq:
            return

        self._inc_stat("ack_timeouts")

        if self.ctx._metrics is not None and self.ctx._pending_frame is not None:
            sta_id = self._metric_sta_id(self.ctx._pending_frame)
            self.ctx._metrics.record_timeout(sta_id, src=sta_id, dst=self.ctx._pending_frame.dst)
            self.ctx._metrics.record_raw_tx(
                now=self.ctx.sim.engine.now,
                src=sta_id,
                size_bytes=self.ctx._pending_frame.size_bytes,
                timeout=True,
            )

        self._log("ACK_TIMEOUT", {"frame_seq": frame_seq}, frame=self.ctx._pending_frame)
        self._handle_retry_or_drop(make_data_frame_cb=make_data_frame_cb)

    # ------------------------------------------------------------------
    # Retry / success
    # ------------------------------------------------------------------
    def _handle_retry_or_drop(self, *, make_data_frame_cb=None) -> None:
        fr = self.ctx._pending_frame
        if fr is None:
            self.ctx.state = MacState.IDLE
            self.drive(make_data_frame_cb=make_data_frame_cb)
            return

        self._increment_retry_counter(fr)
        fr.retry += 1

        if self._retry_limit_reached(fr):
            net_pdu = fr.net_pdu
            pkt = net_pdu.packet if net_pdu is not None else None

            if net_pdu is not None and self.ctx._txq and self.ctx._txq[0] is net_pdu:
                self.ctx._txq.popleft()
                self._record_latency(net_pdu, dropped=True)

            self.ctx._stats["tx_drops"] += 1
            self._inc_stat("retry_limit_drops")
            self._log("DROP", {"reason": "retry_limit", "retry": fr.retry}, frame=fr)

            if self.ctx._metrics is not None and fr.dst != -1:
                sta_id = self._metric_sta_id(fr)
                self.ctx._metrics.record_retry(sta_id, src=sta_id, dst=fr.dst)

            self._notify_tx_failure(pkt)

            self.ctx._pending_frame = None
            self._clear_wait_ack_state()
            self._clear_fragment_state()
            self._clear_backoff_state()
            self.ctx._cw = self.ctx.cw_min
            self._reset_retry_counters()
            self.ctx.state = MacState.IDLE
            self.drive(make_data_frame_cb=make_data_frame_cb)
            return

        self.ctx._stats["tx_retries"] += 1
        self._inc_mac_retry(1)
        self.ctx._cw = min(self.ctx.cw_max, self.ctx._cw * 2 + 1)

        if self.ctx._metrics is not None and fr.dst != -1:
            sta_id = self._metric_sta_id(fr)
            self.ctx._metrics.record_retry(sta_id, src=sta_id, dst=fr.dst)
            self.ctx._metrics.record_raw_tx(
                now=self.ctx.sim.engine.now,
                src=sta_id,
                size_bytes=fr.size_bytes,
                retry=True,
            )

        # IMPORTANT:
        # Preserve the same MAC frame sequence across retransmissions.
        # Only tx_seq and retry counter should change.
        fr.tx_seq += 1

        if fr.ctrl:
            mcs = fr.ctrl.get("mcs", self.ctx.node.phy.default_mode)
            fr.ctrl["duration_s"] = self.ctx.sifs + self._compute_ack_tx_time(fr.dst, mcs)

        self._clear_wait_ack_state()
        self.ctx.state = MacState.IDLE
        self._clear_backoff_state()
        self._log(
            "RETRY",
            {
                "retry": fr.retry,
                "cw": self.ctx._cw,
                "frame_seq_preserved": fr.frame_seq,
            },
            frame=fr,
        )
        self.drive(make_data_frame_cb=make_data_frame_cb)
        return

    def _on_success(self, *, make_data_frame_cb=None) -> None:
        fr = self.ctx._pending_frame
        if fr is None:
            self.ctx.state = MacState.IDLE
            self.drive(make_data_frame_cb=make_data_frame_cb)
            return

        net_pdu = fr.net_pdu
        pkt = net_pdu.packet if net_pdu is not None else None

        delay_s = None
        if net_pdu is not None:
            enq = self.ctx._enqueue_time.get(id(net_pdu), None)
            if enq is not None:
                delay_s = self.ctx.sim.engine.now - enq

        if net_pdu is not None and self.ctx._txq and self.ctx._txq[0] is net_pdu:
            self.ctx._txq.popleft()
            self._record_latency(net_pdu, dropped=False)
            self._log(
                "TX_SUCCESS",
                {"retry": fr.retry, "next_hop": net_pdu.next_hop},
                frame=fr,
                net_pdu=net_pdu,
            )
        else:
            self._log("TX_SUCCESS", {"retry": fr.retry}, frame=fr)

        self._notify_tx_success(pkt)

        if fr.dst != -1 and self.ctx._metrics is not None:
            sta_id = self._metric_sta_id(fr)
            self.ctx._metrics.record_tx_success(sta_id, fr.size_bytes, delay_s, src=sta_id, dst=fr.dst)
            self.ctx._metrics.record_raw_tx(
                now=self.ctx.sim.engine.now,
                src=sta_id,
                size_bytes=fr.size_bytes,
                success=True,
            )

        self._reset_after_completion()
        self.drive(make_data_frame_cb=make_data_frame_cb)

    # ------------------------------------------------------------------
    # Fragment sequence minimal support
    # ------------------------------------------------------------------
    def _on_fragment_ack(self, *, make_data_frame_cb=None) -> None:
        self.ctx._pending_frag_index += 1
        if self.ctx._pending_frag_index >= len(self.ctx._pending_fragments):
            self.ctx._pending_fragments = []
            self.ctx._pending_frag_index = 0
            self._on_success(make_data_frame_cb=make_data_frame_cb)
        else:
            self.ctx.sim.engine.schedule_in(
                self.ctx.sifs,
                self._schedule_next_fragment_event,
                make_data_frame_cb=make_data_frame_cb,
                name="DCF_TX_NEXT_FRAGMENT",
            )

    def _tx_next_fragment(self, *, make_data_frame_cb=None) -> None:
        if not self.ctx._pending_fragments:
            self._on_success(make_data_frame_cb=make_data_frame_cb)
            return

        idx = self.ctx._pending_frag_index
        if idx >= len(self.ctx._pending_fragments):
            self.ctx._pending_fragments = []
            self.ctx._pending_frag_index = 0
            self._on_success(make_data_frame_cb=make_data_frame_cb)
            return

        frag = self.ctx._pending_fragments[idx]
        self.ctx._pending_frame = frag
        self.ctx.state = MacState.TX
        self._log("TX_FRAG", {"frag_num": idx, "total": len(self.ctx._pending_fragments)}, frame=frag)
        self.ctx.node.phy.send(frag, tx_id=self.ctx.node.node_id, rx_id=frag.dst)

        self.ctx.state = MacState.WAIT_ACK
        self.ctx._waiting_ack_for_frame_seq = frag.frame_seq
        self.ctx._waiting_ack_for_frag_num = int(frag.ctrl.get("frag_num", 0)) if frag.ctrl else 0
        timeout_s = self._compute_ack_timeout(frag)
        self._arm_ack_timeout(frag.frame_seq, timeout_s, make_data_frame_cb=make_data_frame_cb)

    # ------------------------------------------------------------------
    # Queue / latency / utilization
    # ------------------------------------------------------------------
    def _expire_queue(self, *, record_latency_cb=None) -> None:
        now = self.ctx.sim.engine.now
        expired = []

        pending_pdu = None
        if self.ctx._pending_frame is not None and self.ctx._pending_frame.net_pdu is not None:
            pending_pdu = self.ctx._pending_frame.net_pdu

        for pdu in list(self.ctx._txq):
            if pdu is pending_pdu:
                continue
            enq_t = self.ctx._enqueue_time.get(id(pdu), now)
            if (now - enq_t) > self.ctx.max_msdu_lifetime_s:
                expired.append(pdu)

        for pdu in expired:
            enq = self.ctx._enqueue_time.get(id(pdu), now)
            age_s = now - enq
            try:
                self.ctx._txq.remove(pdu)
            except ValueError:
                pass

            pkt = pdu.packet if pdu is not None else None
            self._notify_tx_failure(pkt)
            self._inc_packet_drop(1)

            if record_latency_cb is not None:
                record_latency_cb(pdu, True)
            else:
                self._record_latency(pdu, dropped=True)

            self.ctx._stats["msdu_ttl_expired"] += 1
            self._log(
                "MSDU_LIFETIME_EXPIRED",
                {"next_hop": pdu.next_hop, "age_s": age_s},
                net_pdu=pdu,
            )

        if pending_pdu is not None:
            enq = self.ctx._enqueue_time.get(id(pending_pdu), now)
            if (now - enq) > self.ctx.max_msdu_lifetime_s:
                pkt = pending_pdu.packet if pending_pdu is not None else None
                self._notify_tx_failure(pkt)
                self._inc_packet_drop(1)

                self.ctx._stats["msdu_ttl_expired"] += 1
                self._inc_stat("pending_ttl_expired")
                self._log(
                    "MSDU_LIFETIME_EXPIRED_PENDING",
                    {"age_s": now - enq},
                    frame=self.ctx._pending_frame,
                )

                if record_latency_cb is not None:
                    record_latency_cb(pending_pdu, True)
                else:
                    self._record_latency(pending_pdu, dropped=True)

                self._reset_after_completion()

    def _record_latency(self, net_pdu: NetPDU, dropped: bool) -> None:
        enq = self.ctx._enqueue_time.pop(id(net_pdu), None)
        if enq is None:
            return
        latency = self.ctx.sim.engine.now - enq
        self.ctx._stats["total_latency_s"] += latency
        self.ctx._stats["latency_samples"] += 1
        self._log("LATENCY", {"dropped": int(dropped), "latency_s": latency}, net_pdu=net_pdu)

    def _sample_utilization(self) -> None:
        now = self.ctx.sim.engine.now
        busy = self.ctx.node.phy.is_channel_busy(self.ctx.node.node_id)
        self.ctx._util_samples.append((now, busy))
        cutoff = now - self.ctx._util_window_s
        while self.ctx._util_samples and self.ctx._util_samples[0][0] < cutoff:
            self.ctx._util_samples.pop(0)

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------
    def _increment_retry_counter(self, frame: MacFrame) -> None:
        if frame.size_bytes <= self.ctx.rts_threshold_bytes:
            self.ctx._short_retry_count += 1
        else:
            self.ctx._long_retry_count += 1

    def _retry_limit_reached(self, frame: MacFrame) -> bool:
        if frame.size_bytes <= self.ctx.rts_threshold_bytes:
            return self.ctx._short_retry_count >= self.ctx.short_retry_limit
        return self.ctx._long_retry_count >= self.ctx.long_retry_limit

    def _reset_retry_counters(self) -> None:
        self.ctx._short_retry_count = 0
        self.ctx._long_retry_count = 0

    # ------------------------------------------------------------------
    # PHY timing helpers
    # ------------------------------------------------------------------
    def _compute_ack_tx_time(self, dst: int, mcs_str: str) -> float:
        if self.ctx.sim.topology is None:
            return (self.ctx.ack_size_bytes * 8) / 65_000.0
        try:
            lk = self.ctx.sim.topology.get_link(self.ctx.node.node_id, dst)
            rate = int(self.ctx.node.phy.mode_table.get(mcs_str, lk.rate_bps))
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
        return float(self.ctx.node.phy.compute_tx_duration(stub, rate))

    def _compute_data_tx_time(self, data_frame: MacFrame) -> float:
        if self.ctx.sim.topology is None:
            return (data_frame.size_bytes * 8) / 65_000.0
        try:
            lk = self.ctx.sim.topology.get_link(self.ctx.node.node_id, data_frame.dst)
            mcs = (
                data_frame.ctrl.get("mcs", self.ctx.node.phy.default_mode)
                if data_frame.ctrl else self.ctx.node.phy.default_mode
            )
            rate = int(self.ctx.node.phy.mode_table.get(mcs, lk.rate_bps))
        except Exception:
            return (data_frame.size_bytes * 8) / 65_000.0
        return float(self.ctx.node.phy.compute_tx_duration(data_frame, rate))

    def _get_prop(self, dst: int) -> float:
        if self.ctx.sim.topology is None:
            return 0.0
        try:
            return float(self.ctx.sim.topology.get_link(self.ctx.node.node_id, dst).prop_delay)
        except Exception:
            return 0.0

    def _compute_ack_timeout(self, data_frame: MacFrame) -> float:
        prop = self._get_prop(data_frame.dst)
        mcs = (
            data_frame.ctrl.get("mcs", self.ctx.node.phy.default_mode)
            if data_frame.ctrl else self.ctx.node.phy.default_mode
        )
        t_data = self._compute_data_tx_time(data_frame)
        t_ack = self._compute_ack_tx_time(data_frame.dst, mcs)

        # Timer is armed immediately after phy.send(), so it must still cover
        # DATA airtime + ACK turnaround.
        needed = t_data + 2.0 * prop + self.ctx.sifs + t_ack + self.ctx.ack_guard

        return max(self.ctx.ack_timeout_cfg, needed)