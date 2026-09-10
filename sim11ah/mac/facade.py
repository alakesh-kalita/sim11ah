from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from sim11ah.constants import FrameType
from sim11ah.models import MacFrame, NetPDU, Packet

from sim11ah.mac.context import MacContext, build_mac_context
from sim11ah.mac.raw import RawEngine
from sim11ah.mac.dcf import DcfEngine
from sim11ah.mac.raw_metrics import MacMetrics
from sim11ah.mac.association import AssocManager, _MGMT_FTYPES
from sim11ah.mac.common import AssocState
from sim11ah.mac.twt import TwtManager


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

        # Association manager (handles IEEE 802.11ah open-system auth + assoc)
        self.assoc = AssocManager(self)
        self.ctx._on_mgmt_ack_cb  = self.assoc._on_mgmt_frame_acked
        self.ctx._on_mgmt_drop_cb = self.assoc._on_mgmt_frame_dropped

        # TWT manager (Target Wake Time negotiation); kicks off automatically
        # once association completes, if cfg["mac"]["twt_enable"] is True.
        self.twt = TwtManager(self)
        self.ctx._on_associated_cb = self.twt.on_associated

        # Stable duplicate cache for DATA frames across retransmissions
        if not hasattr(self.ctx, "_rx_data_key_cache"):
            self.ctx._rx_data_key_cache = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_ap(self) -> bool:
        return self.node.is_ap

    def _extract_duplicate_key(self, frame: MacFrame) -> Tuple[int, str, int]:
        """
        Build a stable duplicate key for DATA reception.

        Use the ORIGINATOR's identity (net_pdu.src) rather than the immediate
        sender (frame.src) so that relay-forwarded frames are deduplicated per
        original source, not per relay.  Falls back to frame.src when no
        net_pdu is present (e.g. direct star topology).

        Preference order:
        1. (net_pdu.src, "net_seq", net_pdu.net_seq)   — relay-safe
        2. (net_pdu.src, "pkt_seq", packet.packet_seq)
        3. (frame.src,   "frame_seq", frame.frame_seq)  — last-resort fallback
        """
        try:
            if frame.net_pdu is not None:
                orig_src = getattr(frame.net_pdu, "src", None)
                net_seq  = getattr(frame.net_pdu, "net_seq", None)
                if orig_src is not None and net_seq is not None:
                    return (int(orig_src), "net_seq", int(net_seq))
        except Exception:
            pass

        try:
            if frame.net_pdu is not None and frame.net_pdu.packet is not None:
                orig_src = getattr(frame.net_pdu, "src", None)
                pkt_seq  = getattr(frame.net_pdu.packet, "packet_seq", None)
                if orig_src is not None and pkt_seq is not None:
                    return (int(orig_src), "pkt_seq", int(pkt_seq))
        except Exception:
            pass

        return (int(frame.src), "frame_seq", int(frame.frame_seq))

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

        seen: OrderedDict = cache_map.setdefault(src, OrderedDict())
        if key in seen:
            return True

        seen[key] = None

        maxsize = int(getattr(self.ctx, "_rx_cache_maxsize", 256))
        while len(seen) > maxsize:
            seen.popitem(last=False)  # evict oldest entry (FIFO)

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
        # A relay's OWN beacon loop (advertising itself for STAs to join)
        # is deliberately NOT started here. relay_start_beacons() used to
        # fire unconditionally at t=0 for every relay, meaning a relay
        # advertised itself as joinable well before it had actually
        # associated with the real AP -- an STA could associate with a
        # relay that had no working backhaul yet (or, if that relay's own
        # join later failed, ever). association.py's on_assoc_resp_received
        # calls relay_start_beacons() itself once THIS node's own
        # association to the AP genuinely reaches ASSOCIATED -- a relay's
        # normal STA-side scan/association attempt still starts
        # automatically (AssocManager.__init__ arms it for every non-AP
        # node), it just doesn't broadcast its own beacon until that
        # succeeds.

    def stop(self) -> None:
        pass

    def finalize(self) -> None:
        self.assoc.finalize()

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

        if frame.ftype == FrameType.RTS:
            self._handle_rts_rx(frame)
            return

        if frame.ftype == FrameType.CTS:
            self.dcf.handle_cts_rx(frame, make_data_frame_cb=self._make_data_frame)
            return

        # IEEE 802.11ah management: authentication and association frames
        if frame.ftype == FrameType.AUTH:
            ctrl = frame.ctrl or {}
            if ctrl.get("for") == "req":
                self.assoc.on_auth_req_received(frame)
            else:
                self.assoc.on_auth_resp_received(frame)
            # ACK the management frame after SIFS (same as DATA ACK)
            self.sim.engine.schedule_in(
                self.ctx.sifs,
                self._schedule_send_ack,
                frame.src, frame.frame_seq, 0, None,
                name="MAC_SEND_ACK_MGMT",
            )
            return

        if frame.ftype == FrameType.ASSOC_REQ:
            self.assoc.on_assoc_req_received(frame)
            self.sim.engine.schedule_in(
                self.ctx.sifs,
                self._schedule_send_ack,
                frame.src, frame.frame_seq, 0, None,
                name="MAC_SEND_ACK_MGMT",
            )
            return

        if frame.ftype == FrameType.ASSOC_RESP:
            self.assoc.on_assoc_resp_received(frame)
            self.sim.engine.schedule_in(
                self.ctx.sifs,
                self._schedule_send_ack,
                frame.src, frame.frame_seq, 0, None,
                name="MAC_SEND_ACK_MGMT",
            )
            return

        # TWT (Target Wake Time) negotiation
        if frame.ftype == FrameType.TWT_SETUP_REQ:
            self.twt.on_setup_req_received(frame)
            self.sim.engine.schedule_in(
                self.ctx.sifs,
                self._schedule_send_ack,
                frame.src, frame.frame_seq, 0, None,
                name="MAC_SEND_ACK_MGMT",
            )
            return

        if frame.ftype == FrameType.TWT_SETUP_RESP:
            self.twt.on_setup_resp_received(frame)
            self.sim.engine.schedule_in(
                self.ctx.sifs,
                self._schedule_send_ack,
                frame.src, frame.frame_seq, 0, None,
                name="MAC_SEND_ACK_MGMT",
            )
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

                # Per-source reception count: the only signal RAW policies
                # (e.g. AdaptiveRawPolicy) may use to estimate per-STA uplink
                # demand, since it comes purely from locally observed
                # receptions with no control-packet overhead.
                rx_by_src = getattr(self.ctx, "_rx_count_by_src", None)
                if rx_by_src is None:
                    rx_by_src = {}
                    self.ctx._rx_count_by_src = rx_by_src
                rx_by_src[frame.src] = rx_by_src.get(frame.src, 0) + 1

                # More-Data flag + reception timestamp per source (E-TAROA's
                # traffic estimation, raw_policy_etaroa.py, needs both: the
                # station's live queue-backlog signal now that dcf.py sets
                # more_data for real, and the exact time each DATA frame
                # landed so the policy can tell whether a reception spilled
                # past its nominal beacon interval).
                more_data_by_src = getattr(self.ctx, "_more_data_by_src", None)
                if more_data_by_src is None:
                    more_data_by_src = {}
                    self.ctx._more_data_by_src = more_data_by_src
                more_data_by_src[frame.src] = bool((frame.ctrl or {}).get("more_data", False))

                last_rx_time_by_src = getattr(self.ctx, "_last_rx_time_by_src", None)
                if last_rx_time_by_src is None:
                    last_rx_time_by_src = {}
                    self.ctx._last_rx_time_by_src = last_rx_time_by_src
                last_rx_time_by_src[frame.src] = float(self.sim.engine.now)

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

    def ap_start_beacons(self, phase_offset_s: float = 0.0) -> None:
        if not self._is_ap():
            return
        # Idempotency guard: each call schedules its own independent,
        # self-perpetuating engine event chain (_ap_send_beacon reschedules
        # itself via schedule_in on every firing) -- it does NOT cancel a
        # chain from a prior call. MacLayer.start() unconditionally calls
        # this (offset 0.0) for every AP node; a caller that wants a
        # phase-staggered first beacon (multi-AP) must call this explicitly
        # BEFORE node.start(), and this guard makes that explicit call win,
        # suppressing start()'s automatic follow-up call as a no-op. Without
        # this guard, two chains end up interleaved and racing on the same
        # shared ctx._ap_beacon_count/_next_beacon_target state -- caught
        # empirically as a genuine duplicate beacon transmission ~7ms after
        # simulation start, present even in the plain single-AP case.
        if getattr(self.ctx, "_beacon_chain_started", False):
            return
        self.ctx._beacon_chain_started = True

        self.ctx._ap_beacon_count = 0
        self.ctx._next_beacon_target = self.sim.engine.now + max(0.0, float(phase_offset_s))
        self._log(
            "AP_BEACON_START",
            {
                "beacon_interval": self.ctx.beacon_interval,
                "dtim_period": self.ctx.dtim_period,
                "raw_enable": int(self.ctx.raw_enable),
                "phase_offset_s": float(phase_offset_s),
            },
        )
        # phase_offset_s staggers the FIRST beacon of each AP under multi-AP
        # (see MultiApBuilder) so beacon intervals don't all fire perfectly
        # in phase -- otherwise every overlap-region STA would see correlated
        # collisions on every interval instead of the independent pattern a
        # real deployment would have, which could be mistaken for the actual
        # residence-time effect this feature exists to measure.
        self.sim.engine.schedule_in(
            max(0.0, float(phase_offset_s)),
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
            # This node's own id, not a hardcoded 0 -- under multi-AP a
            # beacon from AP #2 (node_id != 0) previously claimed src=0
            # regardless, making every STA that heard it think it came from
            # AP #1 instead (assoc_peer_id, roam-trigger AP identification,
            # etc. all key off this field). _ap_send_beacon is only ever
            # scheduled from ap_start_beacons's already-_is_ap()-gated path,
            # so self.node here is always the actual AP sending it.
            src=self.node.node_id,
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

    # ------------------------------------------------------------------
    # Relay-hosted association: an unassociated STA whose direct link to
    # the real AP is blocked/out of range should still be able to
    # discover and join the BSS via any relay node it *can* hear, so a
    # STA "always tries to join either AP or relay" rather than being
    # permanently stuck unassociated. A relay-role node therefore also
    # runs its own lightweight beacon loop (deliberately NOT the full
    # ap_start_beacons()/_ap_send_beacon() pipeline -- a relay has no
    # independent RAW schedule of its own to announce; it re-embeds
    # whatever RAW Parameter Set it most recently heard from the real
    # AP, see the caching in _handle_beacon() below, so RAW-window
    # scheduling still reaches STAs that only ever hear the relay).
    # ------------------------------------------------------------------
    def relay_start_beacons(self) -> None:
        if self.node.role != "RELAY":
            return
        # Guard against a second overlapping beacon loop: association.py
        # calls this every time this node's OWN association to the AP
        # reaches ASSOCIATED, which can happen more than once if a relay
        # later loses and regains its backhaul link (_check_beacon_liveness
        # link-loss -> re-scan -> re-associate). _relay_send_beacon()
        # reschedules itself forever with no stop condition, so without
        # this flag a re-association would start a second loop running
        # alongside the first and the relay would transmit duplicate
        # beacons every interval from then on.
        if getattr(self.ctx, "_relay_beaconing_started", False):
            return
        self.ctx._relay_beaconing_started = True
        self.ctx._ap_beacon_count = 0

        # Stagger each relay's beacon phase across the beacon interval so
        # the AP and every relay don't all transmit at the exact same
        # simulated instant every cycle -- without this, every beacon
        # interval becomes a synchronized collision between the AP and all
        # relays (the same thundering-herd failure mode fixed for TWT wake
        # phases elsewhere; unstaggered here measurably collapsed relay
        # PDR in testing).
        relay_ids = sorted(self.ctx.cfg.get("topology", {}).get("relay_ids", [self.node.node_id]))
        try:
            idx = relay_ids.index(self.node.node_id)
        except ValueError:
            idx = 0
        n = max(len(relay_ids), 1)
        phase_offset_s = self.ctx.beacon_interval * (idx + 1) / (n + 1)

        self.ctx._next_beacon_target = self.sim.engine.now + phase_offset_s
        self._log("RELAY_BEACON_START", {
            "beacon_interval": self.ctx.beacon_interval,
            "phase_offset_s": phase_offset_s,
        })
        self.sim.engine.schedule_in(
            phase_offset_s,
            self._relay_send_beacon,
            name="MAC_RELAY_SEND_BEACON",
        )

    def _relay_send_beacon(self) -> None:
        if self.ctx._assoc_state != AssocState.ASSOCIATED:
            # This relay's own backhaul link to the AP is down -- keep the
            # beacon loop alive (so it resumes on the very next cycle once
            # backhaul comes back, phase intact) but skip the actual
            # transmission. Without this, a relay that lost its own uplink
            # (see association.py's _check_beacon_liveness) kept beaconing
            # "I'm here" regardless, forever: its STAs' own beacon-liveness
            # timers kept getting reset by those beacons, so they never
            # noticed anything was wrong and kept transmitting uplink data
            # into a relay that had nowhere left to forward it. Silence is
            # the signal here, not a new message type -- STAs already have
            # a working "peer went quiet -> deassociate -> rescan for a new
            # peer" path (on_beacon_received/_check_beacon_liveness); this
            # just lets them actually reach it instead of being kept
            # falsely convinced the relay is still a usable peer.
            self.ctx._next_beacon_target += self.ctx.beacon_interval
            delay = max(0.0, self.ctx._next_beacon_target - self.sim.engine.now)
            self.sim.engine.schedule_in(delay, self._relay_send_beacon, name="MAC_RELAY_SEND_BEACON")
            return

        if self.node.phy.is_channel_busy(self.node.node_id):
            now = self.sim.engine.now
            t_idle = self.sim.medium_next_idle_time(now)
            delay = max(self.ctx.slot_time, t_idle - now)
            self.sim.engine.schedule_in(delay, self._relay_send_beacon, name="MAC_RELAY_SEND_BEACON")
            return

        self.ctx._ap_beacon_count += 1
        cached_rps = getattr(self.ctx, "_relay_cached_rps", None)
        cached_guard = getattr(self.ctx, "_relay_cached_raw_guard", None)

        ctrl: Dict[str, Any] = {
            "beacon_count": self.ctx._ap_beacon_count,
            "is_dtim": bool(cached_rps),
        }
        if cached_rps:
            ctrl["rps"] = cached_rps
            ctrl["raw_guard"] = float(cached_guard if cached_guard is not None else self.ctx.raw_guard)

        self.ctx._tx_seq_ctr += 1
        beacon = MacFrame(
            ftype=FrameType.BEACON,
            src=self.node.node_id,
            dst=-1,
            size_bytes=self.ctx.beacon_size_bytes,
            frame_seq=self.sim.next_frame_seq(),
            tx_seq=self.ctx._tx_seq_ctr,
            retry=0,
            net_pdu=None,
            ctrl=ctrl,
        )
        self._log("TX_RELAY_BEACON", {"beacon_count": self.ctx._ap_beacon_count,
                                       "rps_relayed": int(bool(cached_rps))}, frame=beacon)
        self.node.phy.send(beacon, tx_id=self.node.node_id, rx_id=-1)

        self.ctx._next_beacon_target += self.ctx.beacon_interval
        delay = max(0.0, self.ctx._next_beacon_target - self.sim.engine.now)
        self.sim.engine.schedule_in(delay, self._relay_send_beacon, name="MAC_RELAY_SEND_BEACON")

    def _handle_beacon(self, frame: MacFrame) -> None:
        if self._is_ap():
            return

        ctrl = frame.ctrl or {}

        if self.node.role == "RELAY" and frame.src == 0 and "rps" in ctrl:
            self.ctx._relay_cached_rps = ctrl["rps"]
            self.ctx._relay_cached_raw_guard = float(ctrl.get("raw_guard", self.ctx.raw_guard))

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

        # Trigger association on first beacon (if not yet associated)
        self.assoc.on_beacon_received(frame)

        if self.ctx.raw_enable and is_dtim and "rps" in ctrl:
            self.raw.apply_rps(
                ctrl["rps"],
                float(ctrl.get("raw_guard", self.ctx.raw_guard)),
            )

        self.dcf.drive(make_data_frame_cb=self._make_data_frame)

    # ------------------------------------------------------------------
    # RTS / CTS RX
    # ------------------------------------------------------------------
    def _handle_rts_rx(self, frame: MacFrame) -> None:
        """
        Respond to an RTS addressed to us with CTS after SIFS, and set our
        own NAV from the RTS Duration field. Under the current unicast-only
        PHY delivery model, only the addressed destination ever reaches this
        handler (a non-addressed "hidden" node never receives a copy) — full
        third-party NAV-based hidden-node protection would additionally
        require broadcast-style RTS/CTS delivery, which is not modelled.
        """
        ctrl = frame.ctrl or {}
        duration_s = float(ctrl.get("duration_s", 0.0))
        self.ctx._nav_end = max(self.ctx._nav_end, self.sim.engine.now + duration_s)
        self.ctx._stats["nav_updates"] += 1

        self.sim.engine.schedule_in(
            self.ctx.sifs,
            self._schedule_send_cts,
            frame.src, frame.frame_seq, ctrl.get("protects_seq"), duration_s,
            name="MAC_SEND_CTS",
        )

    def _schedule_send_cts(self, dst: int, rts_frame_seq: int, protects_seq, rts_duration_s: float) -> None:
        self._send_cts(dst, rts_frame_seq, protects_seq, rts_duration_s)

    def _send_cts(self, dst: int, rts_frame_seq: int, protects_seq, rts_duration_s: float) -> None:
        try:
            local_busy = self.node.phy.is_channel_busy(self.node.node_id)
        except Exception:
            local_busy = False

        if local_busy:
            delay = max(self.ctx.slot_time, 1e-6)
            self._log("CTS_DEFER_LOCAL_BUSY", {"dst": dst, "rts_seq": rts_frame_seq})
            self.sim.engine.schedule_in(
                delay,
                self._schedule_send_cts,
                dst, rts_frame_seq, protects_seq, rts_duration_s,
                name="MAC_SEND_CTS",
            )
            return

        t_cts = self.dcf._compute_cts_tx_time(dst)
        # CTS's own duration = remaining protection after CTS ends
        # (SIFS+DATA+SIFS+ACK), derived from the RTS's duration field per
        # IEEE 802.11-2020 Section 9.3.2.5, not re-derived independently.
        cts_duration_s = max(0.0, rts_duration_s - self.ctx.sifs - t_cts)

        self.ctx._tx_seq_ctr += 1
        cts = MacFrame(
            ftype=FrameType.CTS,
            src=self.node.node_id,
            dst=dst,
            size_bytes=self.ctx.cts_size_bytes,
            frame_seq=self.sim.next_frame_seq(),
            tx_seq=self.ctx._tx_seq_ctr,
            retry=0,
            net_pdu=None,
            ctrl={"duration_s": cts_duration_s, "cts_for_seq": protects_seq},
        )
        self._log("TX_CTS", {"cts_for": protects_seq, "dst": dst}, frame=cts)
        self.node.phy.send(cts, tx_id=self.node.node_id, rx_id=dst)
        self.ctx._nav_end = max(self.ctx._nav_end, self.sim.engine.now + t_cts + cts_duration_s)

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