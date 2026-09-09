"""
IEEE 802.11ah Target Wake Time (TWT) -- Individual TWT negotiation.
=====================================================================
Implements a minimal but real Individual TWT agreement: once a STA
completes association, it sends a TWT Setup Request proposing a wake
interval/duration; the AP accepts as-is (the simplest standards-valid
negotiation outcome, "TWT Setup Command: Accept") and responds with a
TWT Setup Response carrying the agreed parameters and the absolute
time of the first Service Period (SP).

Interaction with RAW
---------------------
TWT governs *when a STA promises to be awake*; RAW governs *contention
grouping and channel access* within an active beacon interval. They are
complementary, not the same axis. To guarantee TWT can never remove a
transmission opportunity RAW would otherwise grant, this module only
acts (sets ctx._dozing / calls energy.on_sleep_start) when the STA is
NOT currently inside its own assigned RAW window
(``ctx.raw_enable and ctx.raw_allowed``). RawEngine's own raw_enter()
already unconditionally forces ctx._dozing = False and closes any open
energy sleep interval whenever a STA's RAW window begins, regardless of
TWT state, so a RAW opportunity is never suppressed by a TWT doze phase.

For RAW-enabled deployments, RawEngine already provides gapless
sleep/wake energy accounting for every non-RAW-window instant, so TWT's
own sleep calls there are (correctly) idempotent no-ops most of the
time. For RAW-disabled (No-RAW) deployments, RawEngine's sleep hooks
never run at all -- TWT is then the *only* power-save signal, providing
genuine new behaviour where none existed before.

Reference: IEEE 802.11ah-2016 / 802.11-2020 Section 9.4.2.198-199 (TWT
element, carried here in TWT_SETUP_REQ/RESP mgmt frames), Section 27.8
(TWT operation, informative for S1G).
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sim11ah.constants import FrameType, MacState
from sim11ah.models import MacFrame

if TYPE_CHECKING:
    from sim11ah.mac.facade import MacLayer

_DEFAULT_SETUP_SIZE_BYTES = 30   # 28-byte mgmt hdr + TWT element (approx) + FCS
_DEFAULT_WAKE_DURATION_S = 0.05  # 50 ms Service Period


class TwtManager:
    """Per-node TWT negotiation + wake/doze scheduling. One instance per node."""

    def __init__(self, mac: "MacLayer") -> None:
        self._mac = mac
        self._ctx = mac.ctx
        self._sim = mac.sim
        self._node = mac.node

        mac_cfg = self._ctx.cfg.get("mac", {})
        self._setup_size = int(mac_cfg.get("twt_setup_size_bytes", _DEFAULT_SETUP_SIZE_BYTES))

        # Wake interval defaults to the STA's own periodic reporting interval
        # when known -- this is what makes the negotiated schedule meaningful
        # (a STA asks to wake roughly when it expects to have data) rather
        # than an arbitrary fixed period.
        app_cfg = self._ctx.cfg.get("app", {})
        default_interval = float(app_cfg.get("periodic_interval", 5.0)) or 5.0
        self._wake_interval_s = float(mac_cfg.get("twt_wake_interval_s", default_interval))
        self._wake_duration_s = float(mac_cfg.get("twt_wake_duration_s", _DEFAULT_WAKE_DURATION_S))

    def _is_ap(self) -> bool:
        return int(self._node.node_id) == 0

    def _log(self, event: str, details: dict) -> None:
        self._sim.log(node_id=self._node.node_id, layer="MAC", event=f"TWT_{event}", details=details)

    # ------------------------------------------------------------------
    # STA side: setup request, triggered once association completes
    # ------------------------------------------------------------------
    def on_associated(self) -> None:
        if self._is_ap() or not self._ctx.twt_enable:
            return
        if self._ctx._own_twt is not None:
            return  # already negotiated (re-association edge case)
        self._send_setup_req()

    def _send_setup_req(self) -> None:
        self._ctx._tx_seq_ctr += 1
        frame = MacFrame(
            ftype=FrameType.TWT_SETUP_REQ,
            src=self._node.node_id,
            dst=0,
            size_bytes=self._setup_size,
            frame_seq=self._sim.next_frame_seq(),
            tx_seq=0,
            retry=0,
            net_pdu=None,
            ctrl={
                "wake_interval_s": self._wake_interval_s,
                "wake_duration_s": self._wake_duration_s,
            },
        )
        self._log("SETUP_REQ_TX", {
            "wake_interval_s": self._wake_interval_s,
            "wake_duration_s": self._wake_duration_s,
        })
        self._ctx._mgmt_txq.append(frame)
        self._mac.dcf.drive(make_data_frame_cb=self._mac._make_data_frame)

    def on_setup_resp_received(self, frame: MacFrame) -> None:
        if self._is_ap():
            return
        ctrl = frame.ctrl or {}
        if int(ctrl.get("status", 1)) != 0:
            self._log("SETUP_FAILED", {"status": ctrl.get("status")})
            return

        agreement = {
            "wake_interval_s": float(ctrl.get("wake_interval_s", self._wake_interval_s)),
            "wake_duration_s": float(ctrl.get("wake_duration_s", self._wake_duration_s)),
            "next_twt_s": float(ctrl.get("next_twt_s", self._sim.engine.now)),
        }
        self._ctx._own_twt = agreement
        self._log("NEGOTIATED", dict(agreement))

        self._ctx._twt_wake_gen += 1
        gen = self._ctx._twt_wake_gen
        now = float(self._sim.engine.now)
        delay = max(0.0, agreement["next_twt_s"] - now)
        self._sim.engine.schedule_in(delay, self._twt_wake, gen, name="TWT_SP_WAKE")

    # ------------------------------------------------------------------
    # STA side: self-rescheduling wake/doze ticks
    # ------------------------------------------------------------------
    def _twt_wake(self, gen: int) -> None:
        if gen != self._ctx._twt_wake_gen or self._ctx._own_twt is None:
            return
        agreement = self._ctx._own_twt
        self._ctx._twt_awake = True
        self._ctx._stats["twt_wake_entries"] += 1

        # RAW takes priority: if the STA is currently inside its own RAW
        # window, RawEngine already owns wake/sleep energy accounting there.
        in_raw_window = bool(self._ctx.raw_enable and self._ctx.raw_allowed)
        was_dozing = bool(self._ctx._dozing)
        if not in_raw_window:
            self._ctx._dozing = False
            energy = getattr(self._sim, "energy", None)
            if energy is not None:
                try:
                    energy.on_sleep_end(self._node.node_id, self._sim.engine.now)
                except Exception:
                    pass
        self._log("SP_WAKE", {"wake_duration_s": agreement["wake_duration_s"], "in_raw_window": in_raw_window})

        # Mirror RawEngine.raw_enter: waking from doze must (a) clear the
        # RAW_SLEEP state left behind by the doze transition -- drive()
        # only acts on IDLE/BACKOFF, so leaving state=RAW_SLEEP here makes
        # drive() a silent no-op -- and (b) actively re-kick the DCF engine,
        # since nothing else resumes a frame that was paused mid-backoff
        # when doze began until the next enqueue.
        if was_dozing and not in_raw_window:
            if self._ctx.state != MacState.WAIT_ACK:
                self._ctx.state = MacState.IDLE
            self._mac.dcf.drive(make_data_frame_cb=self._mac._make_data_frame)

        self._sim.engine.schedule_in(
            float(agreement["wake_duration_s"]), self._twt_doze, gen, name="TWT_SP_DOZE",
        )

    def _twt_doze(self, gen: int) -> None:
        if gen != self._ctx._twt_wake_gen or self._ctx._own_twt is None:
            return
        agreement = self._ctx._own_twt
        self._ctx._twt_awake = False

        # Never actually doze while inside our own RAW window -- RAW always
        # wins (see module docstring).
        in_raw_window = bool(self._ctx.raw_enable and self._ctx.raw_allowed)
        if not in_raw_window:
            self._ctx._dozing = True
            energy = getattr(self._sim, "energy", None)
            if energy is not None:
                try:
                    energy.on_sleep_start(self._node.node_id, self._sim.engine.now)
                except Exception:
                    pass
        self._log("SP_DOZE", {"in_raw_window": in_raw_window})

        agreement["next_twt_s"] = float(self._sim.engine.now) + float(agreement["wake_interval_s"])
        self._sim.engine.schedule_in(
            float(agreement["wake_interval_s"]), self._twt_wake, gen, name="TWT_SP_WAKE",
        )

    # ------------------------------------------------------------------
    # AP side: respond to Setup Request (accept as proposed)
    # ------------------------------------------------------------------
    def on_setup_req_received(self, frame: MacFrame) -> None:
        if not self._is_ap():
            return
        sta_id = frame.src
        ctrl = frame.ctrl or {}
        wake_interval_s = float(ctrl.get("wake_interval_s", self._wake_interval_s))
        wake_duration_s = float(ctrl.get("wake_duration_s", self._wake_duration_s))

        # Stagger each STA's first SP across the wake interval instead of
        # naively accepting "now + interval" for everyone. Without this,
        # STAs that associate around the same time (the common case) end up
        # with near-identical SP phases, so every wake cycle becomes a
        # thundering-herd contention burst squeezed into wake_duration_s --
        # defeating much of the point of negotiating TWT in the first place.
        # Round-robins STAs across floor(wake_interval_s / wake_duration_s)
        # phase slots (deterministic on sta_id, so re-associations land the
        # same place); this is the same kind of SP-offset assignment a real
        # AP would use, just via a simple index instead of a load-aware one.
        num_phase_slots = max(1, int(wake_interval_s // max(wake_duration_s, 1e-6)))
        phase_offset_s = (sta_id % num_phase_slots) * wake_duration_s
        next_twt_s = float(self._sim.engine.now) + phase_offset_s

        self._ctx._twt_agreements[sta_id] = {
            "wake_interval_s": wake_interval_s,
            "wake_duration_s": wake_duration_s,
            "next_twt_s": next_twt_s,
        }
        self._log("AP_SETUP_REQ", {
            "sta": sta_id, "wake_interval_s": wake_interval_s, "wake_duration_s": wake_duration_s,
        })

        self._ctx._tx_seq_ctr += 1
        resp = MacFrame(
            ftype=FrameType.TWT_SETUP_RESP,
            src=self._node.node_id,
            dst=sta_id,
            size_bytes=self._setup_size,
            frame_seq=self._sim.next_frame_seq(),
            tx_seq=0,
            retry=0,
            net_pdu=None,
            ctrl={
                "status": 0,
                "wake_interval_s": wake_interval_s,
                "wake_duration_s": wake_duration_s,
                "next_twt_s": next_twt_s,
            },
        )
        self._ctx._mgmt_txq.append(resp)
        self._mac.dcf.drive(make_data_frame_cb=self._mac._make_data_frame)
