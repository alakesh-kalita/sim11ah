"""
IEEE 802.11ah Association Manager
===================================
Implements the standard open-system authentication + association procedure
described in IEEE 802.11ah-2016 Section 9.3.3 / 9.6.

Procedure (per STA):
  1. Passive scan  — STA waits for beacon from AP
  2. Authentication:
       STA → AP  AUTH Request  (Algorithm=0 Open System, SeqNo=1)
       AP  → STA AUTH Response (SeqNo=2, Status=0 Success)
  3. Association:
       STA → AP  ASSOC Request  (capabilities, rates, S1G IE, listen interval)
       AP  → STA ASSOC Response (Status=0 Success, AID assigned)

All management frames use the normal DCF CSMA/CA channel-access mechanism.
They are placed in a priority queue (ctx._mgmt_txq) so they are transmitted
before any DATA frames.

AID assignment: AP assigns AID = sta_node_id for backward-compatibility with
the RAW scheduler (which already groups STAs by node_id ranges).

Timing statistics (written to sim.stats at finalization):
  assoc_scan_time [node_id] — seconds from STA start to first beacon
  assoc_auth_time [node_id] — seconds for auth request/response exchange
  assoc_req_time  [node_id] — seconds for assoc request/response exchange
  assoc_total_time[node_id] — total from start to ASSOCIATED

Reference:
  IEEE 802.11ah-2016, Clause 9.3.3, 9.6.3, 9.6.4
  Khorov et al., "A Tutorial on IEEE 802.11ah", IEEE Comms. Surveys 2019
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from sim11ah.constants import FrameType
from sim11ah.mac.common import AssocState
from sim11ah.models import MacFrame

if TYPE_CHECKING:
    from sim11ah.mac.facade import MacLayer


# ---------------------------------------------------------------------------
# Management frame sizes (IEEE 802.11ah, Section 9.3 / Table 9-33)
# ---------------------------------------------------------------------------
_DEFAULT_AUTH_SIZE_BYTES      = 40   # 28-byte Mgmt hdr + 6-byte body + 4 FCS + 2 pad
_DEFAULT_ASSOC_REQ_SIZE_BYTES = 72   # 28-byte Mgmt hdr + 40 bytes (cap + IE fields) + 4 FCS
_DEFAULT_ASSOC_RESP_SIZE_BYTES= 48   # 28-byte Mgmt hdr + 16 bytes (cap + status + AID + rates)

_MGMT_FTYPES = {FrameType.AUTH, FrameType.ASSOC_REQ, FrameType.ASSOC_RESP}


class AssocManager:
    """
    Per-node association state machine.

    Created and owned by MacLayer (one instance per node).  The manager
    handles both the STA side and the AP side in one class, branching on
    ``_is_ap``.
    """

    def __init__(self, mac: "MacLayer") -> None:
        self._mac  = mac
        self._ctx  = mac.ctx
        self._sim  = mac.sim
        self._node = mac.node

        mac_cfg = self._ctx.cfg.get("mac", {})
        self._auth_size      = int(mac_cfg.get("auth_frame_size_bytes",       _DEFAULT_AUTH_SIZE_BYTES))
        self._assoc_req_size = int(mac_cfg.get("assoc_req_size_bytes",         _DEFAULT_ASSOC_REQ_SIZE_BYTES))
        self._assoc_resp_size= int(mac_cfg.get("assoc_resp_size_bytes",        _DEFAULT_ASSOC_RESP_SIZE_BYTES))
        self._assoc_enable   = bool(mac_cfg.get("association_enable", True))
        # If a STA's in-progress handshake (AUTH/ASSOC_REQ retries) hasn't
        # completed within this many seconds, and a beacon from a DIFFERENT
        # source arrives, abandon the stuck attempt and retarget onto that
        # source instead. Without this, a STA that locked onto a peer which
        # later becomes unreachable (e.g. a relay flown out of range
        # mid-handshake) retries that same peer forever -- see
        # _on_mgmt_frame_dropped, which always re-sends to frame.dst -- even
        # while a closer, in-range peer (often the AP itself) keeps
        # beaconing the whole time. ~4 beacon intervals by default, so a
        # handshake gets several genuine chances before being abandoned.
        self._assoc_stuck_timeout = float(mac_cfg.get("assoc_stuck_timeout_s", 2.0))
        # "auto" preference means AP first, relay only as a fallback -- not
        # "whoever's beacon happens to arrive first". The AP always
        # transmits before any relay each interval (see
        # MacLayer.relay_start_beacons's stagger), so a STA that CAN hear
        # the AP normally hears it first anyway -- but a STA that starts
        # listening partway through a cycle (after that interval's AP
        # beacon already fired, before the next one) could otherwise hear
        # a relay first purely by timing luck, even while well within AP
        # range. This grace window makes the preference an explicit rule
        # instead of an incidental side effect of transmission order: a
        # STA's first-ever relay beacon doesn't get joined immediately,
        # it's held as a fallback until the window elapses without an AP
        # beacon arriving. ~2 beacon intervals, so the AP gets a real
        # second chance even if the STA's scan started right after its
        # first one.
        self._beacon_interval = float(mac_cfg.get("beacon_interval", 0.5))
        self._ap_first_grace_s = 2.0 * self._beacon_interval
        self._first_relay_heard_t: Optional[float] = None

        # Per-STA timing (AP side): node_id → float
        self._ap_auth_req_t: Dict[int, float]  = {}
        self._ap_assoc_req_t: Dict[int, float] = {}

        # STA side timing
        self._sta_start_t:     float = 0.0   # sim start time
        self._sta_beacon_t:    float = 0.0   # first beacon received
        self._sta_auth_req_t:  float = 0.0   # AUTH_REQ transmitted
        self._sta_auth_resp_t: float = 0.0   # AUTH_RESP received
        self._sta_assoc_req_t: float = 0.0   # ASSOC_REQ transmitted
        self._sta_assoc_t:     float = 0.0   # ASSOC_RESP received → ASSOCIATED

        self._first_beacon_seen = False
        self._target_peer: Optional[int] = None   # peer of the in-progress handshake
        self._attempt_start_t: float = 0.0         # when that handshake began
        self._assoc_ready_t: Optional[float] = None   # lazily computed, see _get_assoc_ready_t

        # Missed-beacon link-loss detection (STA side). Without this, once
        # ASSOCIATED a STA never re-checks reachability -- e.g. dragging a
        # node out of its peer's range in the GUI, or a relay/UAV wandering
        # away, left it reading "connected" forever with no path back to
        # UNASSOCIATED, since nothing else in this class transitions a STA
        # out of ASSOCIATED except an explicit handover to the AP.
        self._last_peer_beacon_t: float = 0.0
        self._beacon_liveness_token: int = 0

        # RSSI-based roam trigger (multi-AP). node_id -> (rssi_dbm, heard_t).
        # Updated on every beacon actually received from an AP-role source
        # (see on_beacon_received), not just the current peer, so a
        # comparison is available the moment a second AP becomes audible.
        # PhyLayer sets phy._last_rssi_dbm_value synchronously right before
        # invoking this callback (single-threaded engine, same call frame),
        # so it's always the RSSI of the beacon currently being processed.
        # Now a short rolling history per AP (capped at roam_rssi_trend_window
        # samples), not just the latest one -- _predict_crossover_s needs
        # several points to fit a trend against. Most call sites only ever
        # want the latest sample (history[-1]); see _maybe_roam.
        self._heard_ap_rssi: Dict[int, List[Tuple[float, float]]] = {}
        # -inf, not 0.0: a STA's very first handover (relay -> AP, or the
        # first AP-vs-AP roam) must not be blocked by roam_min_dwell_s just
        # because it hasn't happened yet.
        self._last_handover_t: float = float("-inf")

        # Predictive pre-association (optional, off by default -- see
        # roam_predictive_enable). A SEPARATE, minimal handshake state,
        # deliberately not sharing self._ctx._assoc_state/_target_peer:
        # those gate data-plane TX across the whole MAC stack (DcfEngine.
        # can_tx_now, app.py's sink checks, etc.), so reusing them for a
        # background pre-auth attempt would block the STA's PRIMARY traffic
        # on its current, still-valid peer for no reason. _shadow_state
        # mirrors AssocState's values but is private to this handshake.
        self._shadow_peer: Optional[int] = None
        self._shadow_state: int = AssocState.UNASSOCIATED
        self._shadow_start_t: float = 0.0
        self._shadow_auth_req_t: float = 0.0
        self._shadow_assoc_req_t: float = 0.0
        # Candidates already attempted (successfully or not) since the last
        # real handover -- avoids retrying the same candidate every beacon
        # once a shadow attempt has started, failed, or completed.
        self._preauth_attempted: set = set()

        if not self._is_ap() and self._assoc_enable:
            self._sta_start_t = float(self._sim.engine.now)
            # Immediately associated if not enabled (backward-compat mode)
        elif not self._assoc_enable:
            self._ctx._assoc_state = AssocState.ASSOCIATED
            self._ctx._aid = self._node.node_id or None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_ap(self) -> bool:
        return self._node.is_ap

    def _serves_associations(self) -> bool:
        """True for anything a STA can associate WITH: the real AP, or a
        relay-role node hosting its own local association point (see
        MacLayer.relay_start_beacons). A relay still separately runs the
        STA-side branch of this same class for its own uplink association
        to the real AP -- _is_ap() must stay node-id-0-only for that; this
        is deliberately a distinct, broader check used only by the
        AP-response-side handlers below."""
        return self._is_ap() or self._node.role == "RELAY"

    def _log(self, event: str, details: Dict[str, Any]) -> None:
        self._sim.log(
            node_id=self._node.node_id,
            layer="MAC",
            event=f"ASSOC_{event}",
            details=details,
        )

    def _make_mgmt_frame(self, ftype: str, dst: int, size: int, ctrl: Dict) -> MacFrame:
        return MacFrame(
            ftype=ftype,
            src=self._node.node_id,
            dst=dst,
            size_bytes=size,
            frame_seq=self._sim.next_frame_seq(),
            tx_seq=0,
            retry=0,
            net_pdu=None,
            ctrl=ctrl,
        )

    def _queue_for_tx(self, frame: MacFrame) -> None:
        """Insert management frame into the priority TX queue and kick DCF."""
        self._ctx._mgmt_txq.append(frame)
        self._mac.dcf.drive(make_data_frame_cb=self._mac._make_data_frame)

    # ------------------------------------------------------------------
    # STA side: beacon reception triggers association
    # ------------------------------------------------------------------

    def _get_assoc_ready_t(self) -> float:
        """Lazily-computed, per-STA stagger before this STA is allowed to
        act on a beacon and start authenticating. Computed on first use
        (not in __init__) because sim.nodes isn't fully populated until
        every node has finished construction (see topology.py's
        StarBuilder/RelayBuilder.build -- sim.nodes is assigned in one
        shot at the end from a locally-built list).

        Un-associated STAs have no AID yet, so they can't use RAW-slot
        scheduling -- they all contend on raw CSMA for AUTH/ASSOC_REQ.
        With only a couple dozen STAs that's fine, but at hundreds of
        STAs, if they all start scanning within the same beacon interval
        the AP can't win channel access often enough to answer any of
        them, and the whole association procedure deadlocks (confirmed:
        at 500 STAs, 0 ever associated even after 60s). Spreading the
        per-STA start over a window that scales with STA count keeps the
        number of simultaneous first-time contenders bounded regardless
        of N, mirroring how a real large-scale deployment wouldn't have
        every device power on in the same millisecond either."""
        if self._assoc_ready_t is not None:
            return self._assoc_ready_t

        mac_cfg = self._ctx.cfg.get("mac", {})
        # Exclude every AP (topo_cfg["ap_ids"], falling back to {0} for
        # every single-AP topology, same convention as everywhere else
        # multi-AP-aware code in this codebase), not just node 0 --
        # under multi-AP (multi_ap, cars_uavs) the other APs (node ids
        # 1..K-1) never call this themselves, but leaving them in
        # sta_ids inflated n_stas (and therefore the stagger window) and
        # shifted every real STA's idx/slot by however many other APs
        # sorted below it, both by however many APs exist -- harmless at
        # 1 extra id, not at 5.
        ap_ids = set(self._sim.config.get("topology", {}).get("ap_ids", [0]))
        sta_ids = sorted(i for i in self._sim.nodes.keys() if i not in ap_ids)
        n_stas = max(1, len(sta_ids))

        per_sta_gap_s = float(mac_cfg.get("assoc_start_gap_s", 0.05))
        min_spread_s = float(mac_cfg.get("assoc_start_min_spread_s", 1.0))
        spread = max(min_spread_s, n_stas * per_sta_gap_s)

        try:
            idx = sta_ids.index(self._node.node_id)
        except ValueError:
            idx = max(0, self._node.node_id - 1)
        slot = spread / n_stas
        base = idx * slot
        jitter = self._sim.engine.rng.random() * slot

        self._assoc_ready_t = min(spread, base + jitter)
        return self._assoc_ready_t

    def _arm_beacon_liveness_timer(self) -> None:
        """(Re)start the missed-beacon countdown for the current peer. Called
        on every ASSOCIATED and again each time a beacon from that peer is
        actually heard, so the deadline keeps sliding forward while the
        link stays alive. A stale token means a newer beacon (or a fresh
        association) already superseded this particular timer, so it's a
        no-op when it eventually fires."""
        mac_cfg = self._ctx.cfg.get("mac", {})
        beacon_interval = float(mac_cfg.get("beacon_interval", 0.5))
        max_missed = max(1, int(mac_cfg.get("assoc_max_missed_beacons", 3)))
        timeout_s = beacon_interval * max_missed

        self._last_peer_beacon_t = float(self._sim.engine.now)
        self._beacon_liveness_token += 1
        token = self._beacon_liveness_token
        self._sim.engine.schedule_in(
            timeout_s, self._check_beacon_liveness, token,
            name="ASSOC_BEACON_LIVENESS",
        )

    def _check_beacon_liveness(self, token: int) -> None:
        if token != self._beacon_liveness_token:
            return  # superseded by a newer beacon or a fresh association
        if self._ctx._assoc_state != AssocState.ASSOCIATED:
            return

        peer = getattr(self._ctx, "_assoc_peer_id", None)

        # If a predictive shadow handshake already completed with SOME
        # candidate, promote it now instead of falling all the way back to
        # a from-scratch scan+auth+assoc -- caught empirically: a fast-
        # moving STA's own shadow AUTH_REQ/ASSOC_REQ traffic can itself eat
        # enough channel time, right as it nears the crossover, that it
        # stops hearing ITS CURRENT peer's beacon before _maybe_roam ever
        # gets a beacon reception to react to. Without this, a STA that had
        # already finished pre-authenticating with the correct next AP
        # still paid the full multi-second reactive handshake cost anyway,
        # defeating the entire point of pre-associating in the first place.
        if self._shadow_state == AssocState.ASSOCIATED and self._shadow_peer is not None:
            self._log("LINK_LOST", {
                "peer": peer,
                "silent_s": round(float(self._sim.engine.now) - self._last_peer_beacon_t, 3),
                "recovered_via": "preassoc",
            })
            self._promote_shadow(self._shadow_peer, from_peer=peer if peer is not None else -1)
            return

        self._log("LINK_LOST", {
            "peer": peer,
            "silent_s": round(float(self._sim.engine.now) - self._last_peer_beacon_t, 3),
        })
        self._ctx._assoc_state = AssocState.UNASSOCIATED
        self._ctx._assoc_peer_id = None
        self._first_beacon_seen = False
        self._target_peer = None

        # Also abandon a still-in-flight (not yet ready) shadow attempt --
        # the normal scan/join flow that runs next can target ANY beacon
        # source, not necessarily the shadow's own candidate, and a fresh
        # primary handshake racing an abandoned-but-still-alive shadow
        # AUTH_REQ/ASSOC_REQ to that SAME peer is the same cross-talk risk
        # _roam_to already guards against (on_auth_resp_received/
        # on_assoc_resp_received route purely by frame.src + state, so two
        # simultaneous in-flight attempts to one peer can't be told apart).
        # A shadow that hadn't finished by the time the link was
        # already declared lost has already been too slow to save this
        # handover -- letting the fresh scan/join proceed uncontested is
        # both safer and likely faster than leaving it competing for the
        # same channel time and AP responses.
        if self._shadow_peer is not None:
            self._log("PREASSOC_ABANDON_STALE", {
                "candidate": self._shadow_peer, "reason": "link_lost",
            })
            self._shadow_peer = None
            self._shadow_state = AssocState.UNASSOCIATED

    def on_beacon_received(self, beacon_frame: MacFrame) -> None:
        """Called by MacLayer._handle_beacon() when a beacon is received."""
        if self._is_ap() or not self._assoc_enable:
            return

        # Whether the SOURCE of this beacon is an AP-role node, not
        # specifically node 0 -- with an old hardcoded ==0 check, a beacon
        # from any AP other than node 0 was misclassified as "not from an
        # AP" (i.e. treated like a relay), so a STA would always hold off
        # for node 0's beacon regardless of which AP it actually heard
        # first or how the APs' beacon timing was staggered.
        src_node = self._sim.nodes.get(int(beacon_frame.src))
        is_from_ap = bool(getattr(src_node, "is_ap", int(beacon_frame.src) == 0))

        if is_from_ap:
            # Record in every state, not just once ASSOCIATED, so a
            # comparison is already available the moment a STA becomes
            # associated instead of needing to wait for a second beacon.
            # Rolling window (see __init__), so a trend can be fit later.
            src_id = int(beacon_frame.src)
            hist = self._heard_ap_rssi.setdefault(src_id, [])
            hist.append((
                float(getattr(self._node.phy, "_last_rssi_dbm_value", float("-inf"))),
                float(self._sim.engine.now),
            ))
            max_hist = int(self._ctx.cfg.get("mac", {}).get("roam_rssi_trend_window", 5))
            if len(hist) > max_hist:
                del hist[: len(hist) - max_hist]

        if (self._ctx._assoc_state == AssocState.UNASSOCIATED
                and not self._first_beacon_seen
                and float(self._sim.engine.now) < self._get_assoc_ready_t()):
            return  # not this STA's turn to start scanning/associating yet

        if self._ctx._assoc_state == AssocState.ASSOCIATED:
            # Already on the BSS -- a beacon from the current peer proves
            # the link is still alive, so push the missed-beacon deadline
            # back out. Also keep listening for other APs' beacons, so a
            # STA that joined a relay (or a weaker/farther AP) can roam
            # once something better becomes reachable.
            if int(beacon_frame.src) == getattr(self._ctx, "_assoc_peer_id", None):
                self._arm_beacon_liveness_timer()
            if is_from_ap:
                self._maybe_roam(int(beacon_frame.src))
                self._maybe_start_preassoc(int(beacon_frame.src))
            return
        # Per-node association preference (GUI-settable, defaults to
        # "auto" = AP first, relay as a fallback -- see _ap_first_grace_s
        # above for why this needs an explicit grace window rather than
        # just "whoever's beacon arrives first"). "ap"/"relay" make the
        # STA hold off and keep waiting until a beacon from that specific
        # kind of source arrives, instead of ever falling back to the
        # other one.
        pref = getattr(self._ctx, "_assoc_preference", "auto")
        if pref == "ap" and not is_from_ap:
            return
        if pref == "relay" and is_from_ap:
            return

        if not self._first_beacon_seen and pref == "auto" and not is_from_ap:
            if self._first_relay_heard_t is None:
                self._first_relay_heard_t = float(self._sim.engine.now)
            if float(self._sim.engine.now) - self._first_relay_heard_t < self._ap_first_grace_s:
                return  # still holding out for the AP's own beacon

        if self._first_beacon_seen:
            # Already mid-handshake with self._target_peer. Normally just
            # wait it out -- but if that peer has gone unreachable (e.g. an
            # aerial relay flown out of range mid-handshake), abandon the
            # stuck attempt and retarget onto this beacon's source instead
            # of retrying the same dead peer forever.
            stuck = (
                self._ctx._assoc_state in (AssocState.AUTHENTICATING, AssocState.ASSOCIATING)
                and int(beacon_frame.src) != self._target_peer
                and (float(self._sim.engine.now) - self._attempt_start_t) >= self._assoc_stuck_timeout
            )
            if not stuck:
                return
            self._log("STUCK_RETARGET", {
                "from_peer": self._target_peer,
                "to_peer": int(beacon_frame.src),
                "stuck_s": round(float(self._sim.engine.now) - self._attempt_start_t, 3),
            })
        else:
            self._first_beacon_seen = True
            self._sta_beacon_t = float(self._sim.engine.now)
            self._log("SCAN_DONE", {
                "scan_time_ms": round((self._sta_beacon_t - self._sta_start_t) * 1000, 3),
            })

        self._begin_auth(beacon_frame.src)

    def _begin_auth(self, dst: int) -> None:
        self._ctx._assoc_state = AssocState.AUTHENTICATING
        self._target_peer = int(dst)
        self._attempt_start_t = float(self._sim.engine.now)
        self._send_auth_req(dst=dst)

    def _maybe_roam(self, beacon_src: int) -> None:
        """Hand over to a different AP-role peer than the one currently
        associated, triggered by hearing THAT peer's beacon directly
        (never a relay's -- on_beacon_received only calls this when
        is_from_ap is True). Two cases:

        1. Currently on a relay: direct AP reachability always wins over
           an indirect relay hop, no RSSI comparison needed -- mirrors the
           original relay->AP behavior, just no longer restricted to node
           0 specifically as "the" AP.
        2. Currently on a real AP, beacon heard from a DIFFERENT real AP:
           only roam if the candidate's measured RSSI is meaningfully
           stronger (roam_hysteresis_db) than the current peer's, and
           roam_min_dwell_s has elapsed since the last handover. Without
           hysteresis, two APs with near-identical RSSI in the overlap
           region would bounce the STA back and forth on every beacon;
           without min-dwell, even a hysteresis-respecting decision could
           ping-pong if both APs' RSSI estimates wobble near the margin.

        Re-runs the exact same auth/assoc procedure a fresh join uses, so
        this reuses all the existing state-machine/retry logic rather than
        adding a special-cased handover path."""
        peer = getattr(self._ctx, "_assoc_peer_id", None)
        if peer is None or int(peer) == beacon_src:
            return  # unknown peer, or already associated with this exact source
        peer_node = self._sim.nodes.get(int(peer))
        peer_is_ap = bool(getattr(peer_node, "is_ap", False))

        if not peer_is_ap:
            if getattr(self._ctx, "_assoc_preference", "auto") == "relay":
                return  # user explicitly pinned this STA to its relay
            self._roam_to(beacon_src, from_peer=int(peer))
            return

        mac_cfg = self._ctx.cfg.get("mac", {})
        hysteresis_db = float(mac_cfg.get("roam_hysteresis_db", 6.0))
        min_dwell_s = float(mac_cfg.get("roam_min_dwell_s", 2.0 * self._beacon_interval))

        now = float(self._sim.engine.now)
        if now - self._last_handover_t < min_dwell_s:
            return

        stale_after_s = 3.0 * self._beacon_interval
        candidate_hist = self._heard_ap_rssi.get(beacon_src)
        candidate_rssi = candidate_hist[-1][0] if candidate_hist else float("-inf")
        peer_hist = self._heard_ap_rssi.get(int(peer))
        peer_entry = peer_hist[-1] if peer_hist else None
        if peer_entry is None or (now - peer_entry[1]) > stale_after_s:
            # Current peer's own beacon hasn't been heard recently --
            # treat its signal as effectively gone rather than trusting a
            # possibly-ancient reading, so a STA can roam off a dying link
            # even before the (slower) missed-beacon liveness timeout
            # would declare it lost outright.
            peer_rssi = float("-inf")
        else:
            peer_rssi = peer_entry[0]

        if candidate_rssi <= peer_rssi + hysteresis_db:
            return  # not meaningfully stronger -- stay put

        self._roam_to(beacon_src, from_peer=int(peer))

    def _roam_to(self, new_peer: int, from_peer: int) -> None:
        if self._shadow_peer == new_peer:
            if self._shadow_state == AssocState.ASSOCIATED:
                # A predictive pre-association (see _maybe_start_preassoc)
                # already ran the full auth/assoc handshake with new_peer
                # in the background while still on from_peer -- promote it
                # instantly instead of paying that latency again now, at
                # exactly the moment it matters most (a short residence
                # window).
                self._promote_shadow(new_peer, from_peer)
            # else: a shadow handshake with this EXACT peer is already
            # mid-flight (AUTHENTICATING/ASSOCIATING) -- do NOT abandon it
            # and start a second, independent handshake to the same
            # target. Caught empirically: doing so left the abandoned
            # shadow's own request still in flight at the AP, and its
            # eventual response arrived indistinguishable from the new
            # handshake's own response (on_assoc_resp_received only keys
            # off frame.src == self._target_peer, and both attempts share
            # the same src), corrupting the fresh handshake with a stale
            # reply. Just wait -- _maybe_roam re-checks on every
            # subsequent beacon, so this fires again once the shadow
            # resolves one way or the other. Deliberately does not touch
            # _last_handover_t here: no handover has actually happened
            # yet, so the dwell timer must not reset.
            return

        # Abandon any other shadow attempt in flight -- it was for a
        # different candidate than the one actually being roamed to.
        self._shadow_peer = None
        self._shadow_state = AssocState.UNASSOCIATED
        self._preauth_attempted.clear()

        self._last_handover_t = float(self._sim.engine.now)
        self._log("HANDOVER", {"from_peer": from_peer, "to_peer": new_peer})
        self._ctx._assoc_state = AssocState.UNASSOCIATED
        self._ctx._assoc_peer_id = None
        self._first_beacon_seen = True  # skip scan-time bookkeeping; go straight to auth
        self._sta_auth_req_t = float(self._sim.engine.now)
        self._begin_auth(new_peer)

    def _promote_shadow(self, new_peer: int, from_peer: int) -> None:
        """Finish a handover whose auth/assoc already completed in the
        background (see _on_shadow_assoc_resp) -- no new frames, just the
        same state transition on_assoc_resp_received makes on a normal
        join, applied immediately.

        Logs the same "HANDOVER" and "ASSOCIATED" event names the reactive
        path uses (tagged with via="preassoc" in the details, not a
        different event name) rather than a bespoke event -- anything
        downstream that already parses ASSOC_HANDOVER/ASSOC_ASSOCIATED to
        measure connectivity_gap_s (see scripts/roam_experiment.py) picks
        this up automatically, and since both fire in the same tick here,
        the measured gap correctly comes out at ~0s instead of silently
        vanishing from the metric.

        Called from both _roam_to (reactive trigger, shadow already ready)
        and _check_beacon_liveness (missed-beacon recovery) -- updates
        _last_handover_t itself so BOTH paths correctly arm roam_min_dwell_s
        against a follow-up ping-pong, instead of only the reactive one."""
        now = float(self._sim.engine.now)
        self._last_handover_t = now
        self._log("HANDOVER", {
            "from_peer": from_peer, "to_peer": new_peer, "via": "preassoc",
            "preassoc_lead_s": round(now - self._shadow_start_t, 3),
        })
        self._ctx._assoc_state = AssocState.ASSOCIATED
        self._ctx._assoc_peer_id = new_peer
        self._ctx._aid = self._node.node_id
        self._arm_beacon_liveness_timer()
        self._first_beacon_seen = True

        s = self._sim.stats
        s.assoc_total_time[self._node.node_id] = now - self._shadow_start_t
        self._log("ASSOCIATED", {
            "aid": self._ctx._aid, "via": "preassoc",
            "total_ms": round((now - self._shadow_start_t) * 1000, 3),
        })

        cb = getattr(self._ctx, "_on_associated_cb", None)
        if callable(cb):
            try:
                cb()
            except Exception:
                pass
        if self._node.role == "RELAY":
            self._mac.relay_start_beacons()
        self._mac.dcf.drive(make_data_frame_cb=self._mac._make_data_frame)

        self._shadow_peer = None
        self._shadow_state = AssocState.UNASSOCIATED
        self._preauth_attempted.clear()

    # ------------------------------------------------------------------
    # Predictive pre-association: an optional second, minimal handshake
    # started ahead of the reactive roam trigger, based on an RSSI trend
    # extrapolation rather than a threshold already crossed. Off by
    # default (roam_predictive_enable) -- every method below is a no-op
    # when disabled, and nothing here touches self._ctx._assoc_state or
    # self._target_peer (the PRIMARY handshake's fields), so a disabled or
    # never-triggered predictive path changes no existing behavior.
    # ------------------------------------------------------------------

    def _fit_slope(self, history: List[Tuple[float, float]]) -> Tuple[Optional[float], float, float]:
        """Least-squares linear fit of an RSSI history (list of (rssi, t)
        samples, same tuple order _heard_ap_rssi already uses). Returns
        (slope_db_per_s, t_mean, rssi_mean) such that
        rssi(t) ~= rssi_mean + slope*(t - t_mean); slope is None if there
        are fewer than 2 samples or they're all at the same timestamp."""
        n = len(history)
        if n < 2:
            return None, 0.0, 0.0
        ts = [h[1] for h in history]
        vs = [h[0] for h in history]
        t_mean = sum(ts) / n
        v_mean = sum(vs) / n
        num = sum((t - t_mean) * (v - v_mean) for t, v in zip(ts, vs))
        den = sum((t - t_mean) ** 2 for t in ts)
        if den <= 1e-9:
            return None, t_mean, v_mean
        return num / den, t_mean, v_mean

    def _predict_crossover_s(self, candidate_id: int) -> Optional[float]:
        """Seconds from now until candidate_id's RSSI trend is projected to
        overtake the current peer's by roam_hysteresis_db -- the same
        trigger condition _maybe_roam reacts to, anticipated ahead of time
        from each side's recent trend instead of waited out. Returns None
        if there isn't enough history yet, the peer is unknown, or the
        trend doesn't converge (candidate not gaining on the peer, or
        already would have crossed -- the reactive trigger handles that)."""
        peer = getattr(self._ctx, "_assoc_peer_id", None)
        if peer is None:
            return None
        mac_cfg = self._ctx.cfg.get("mac", {})
        min_samples = int(mac_cfg.get("roam_predictive_min_samples", 3))

        cand_hist = self._heard_ap_rssi.get(candidate_id, [])
        peer_hist = self._heard_ap_rssi.get(int(peer), [])
        if len(cand_hist) < min_samples or len(peer_hist) < min_samples:
            return None

        cand_slope, cand_t0, cand_v0 = self._fit_slope(cand_hist)
        peer_slope, peer_t0, peer_v0 = self._fit_slope(peer_hist)
        if cand_slope is None or peer_slope is None:
            return None

        hysteresis_db = float(mac_cfg.get("roam_hysteresis_db", 6.0))
        rel_slope = cand_slope - peer_slope
        if rel_slope <= 1e-9:
            return None  # candidate isn't gaining on the peer

        # Solve cand_v0 + cand_slope*(t - cand_t0) == peer_v0 + peer_slope*(t - peer_t0) + hysteresis_db
        rhs = (peer_v0 - peer_slope * peer_t0 + hysteresis_db) - (cand_v0 - cand_slope * cand_t0)
        t_cross = rhs / rel_slope
        delta = t_cross - float(self._sim.engine.now)
        if delta <= 0.0:
            return None  # trend says it already crossed -- reactive path handles it
        return delta

    def _maybe_start_preassoc(self, candidate_id: int) -> None:
        if not bool(self._ctx.cfg.get("mac", {}).get("roam_predictive_enable", False)):
            return
        mac_cfg = self._ctx.cfg.get("mac", {})
        lead_time_s = float(mac_cfg.get("roam_predictive_lead_time_s", 2.0 * self._beacon_interval))

        if self._shadow_peer is not None:
            if self._shadow_peer == candidate_id:
                return  # already pre-associating with this exact candidate
            # A shadow for a DIFFERENT candidate is active or already ready.
            # Only ever one shadow attempt at a time -- a STA has a real,
            # bounded channel-time budget for background handshakes, and
            # silently overwriting an in-progress or completed attempt the
            # moment a second candidate's beacon happens to arrive would
            # both waste the work already done and (for a ready shadow)
            # throw away a completed pre-association for no reason. Only
            # replace it once it's gone stale -- sat unpromoted far longer
            # than the lead time that started it, e.g. a prediction that
            # never panned out because the STA changed course -- rather
            # than on every subsequent beacon from some other AP.
            stale_after_s = 3.0 * lead_time_s
            if float(self._sim.engine.now) - self._shadow_start_t <= stale_after_s:
                return
            self._log("PREASSOC_ABANDON_STALE", {
                "candidate": self._shadow_peer,
                "age_s": round(float(self._sim.engine.now) - self._shadow_start_t, 3),
            })
            self._shadow_peer = None
            self._shadow_state = AssocState.UNASSOCIATED

        if candidate_id in self._preauth_attempted:
            return  # already tried this candidate since the last real handover
        peer = getattr(self._ctx, "_assoc_peer_id", None)
        if peer is not None and int(peer) == candidate_id:
            return  # candidate is already the current peer
        if candidate_id == self._target_peer:
            return  # a primary handshake with this exact peer is already in flight

        predicted = self._predict_crossover_s(candidate_id)
        if predicted is None:
            return
        if predicted > lead_time_s:
            return  # too far out to be worth starting yet

        self._preauth_attempted.add(candidate_id)
        self._shadow_peer = candidate_id
        self._shadow_state = AssocState.AUTHENTICATING
        self._shadow_start_t = float(self._sim.engine.now)
        self._log("PREASSOC_START", {
            "candidate": candidate_id,
            "predicted_crossover_s": round(predicted, 3),
        })
        self._send_shadow_auth_req(candidate_id)

    def _send_shadow_auth_req(self, dst: int) -> None:
        self._shadow_auth_req_t = float(self._sim.engine.now)
        frame = self._make_mgmt_frame(
            ftype=FrameType.AUTH, dst=dst, size=self._auth_size,
            ctrl={"algorithm": 0, "seq_no": 1, "status": 0, "for": "req"},
        )
        self._log("PREASSOC_AUTH_REQ_TX", {"dst": dst})
        self._queue_for_tx(frame)

    def _send_shadow_assoc_req(self, dst: int) -> None:
        self._shadow_assoc_req_t = float(self._sim.engine.now)
        frame = self._make_mgmt_frame(
            ftype=FrameType.ASSOC_REQ, dst=dst, size=self._assoc_req_size,
            ctrl={"capability": 0x0001, "listen_interval": 10, "s1g_cap": True},
        )
        self._log("PREASSOC_ASSOC_REQ_TX", {"dst": dst})
        self._queue_for_tx(frame)

    def _on_shadow_auth_resp(self, frame: MacFrame) -> None:
        ctrl = frame.ctrl or {}
        if int(ctrl.get("seq_no", 0)) != 2:
            return
        if int(ctrl.get("status", 1)) != 0:
            self._log("PREASSOC_AUTH_FAILED", {"candidate": self._shadow_peer, "status": ctrl.get("status")})
            self._shadow_peer = None
            self._shadow_state = AssocState.UNASSOCIATED
            return
        self._shadow_state = AssocState.ASSOCIATING
        self._log("PREASSOC_AUTH_DONE", {"candidate": self._shadow_peer})
        self._send_shadow_assoc_req(self._shadow_peer)

    def _on_shadow_assoc_resp(self, frame: MacFrame) -> None:
        ctrl = frame.ctrl or {}
        if int(ctrl.get("status", 1)) != 0:
            self._log("PREASSOC_ASSOC_FAILED", {"candidate": self._shadow_peer, "status": ctrl.get("status")})
            self._shadow_peer = None
            self._shadow_state = AssocState.UNASSOCIATED
            return
        self._shadow_state = AssocState.ASSOCIATED
        self._log("PREASSOC_READY", {
            "candidate": self._shadow_peer,
            "preassoc_time_ms": round((float(self._sim.engine.now) - self._shadow_start_t) * 1000, 3),
        })
        # Deliberately does NOT touch self._ctx._assoc_state/_assoc_peer_id
        # -- the STA is still genuinely on its current peer for data-plane
        # purposes. _roam_to promotes this to the real association (see
        # _promote_shadow) once the reactive trigger actually fires.

    def _send_auth_req(self, dst: int) -> None:
        self._sta_auth_req_t = float(self._sim.engine.now)
        frame = self._make_mgmt_frame(
            ftype=FrameType.AUTH,
            dst=dst,
            size=self._auth_size,
            ctrl={"algorithm": 0, "seq_no": 1, "status": 0, "for": "req"},
        )
        self._log("AUTH_REQ_TX", {"dst": dst})
        self._queue_for_tx(frame)

    def on_auth_req_acked(self) -> None:
        """Called by DCF _on_success when AUTH_REQ ACK is received."""
        # AUTH_REQ was acknowledged by AP — STA is now waiting for AUTH_RESP.
        # No state change needed here; AUTH_RESP reception advances state.
        self._log("AUTH_REQ_ACKED", {})

    def on_auth_resp_received(self, frame: MacFrame) -> None:
        """Called by MacLayer when AUTH frame with seq_no=2 arrives at STA."""
        # A response for the predictive shadow handshake (see
        # _maybe_start_preassoc) arrives on this same callback -- dispatch
        # it before the primary-only checks below, which would otherwise
        # silently drop it (frame.src won't match self._target_peer, since
        # this isn't the primary handshake at all).
        if int(frame.src) == self._shadow_peer and self._shadow_state == AssocState.AUTHENTICATING:
            self._on_shadow_auth_resp(frame)
            return
        if self._ctx._assoc_state != AssocState.AUTHENTICATING:
            return
        if int(frame.src) != self._target_peer:
            return  # stale response from a peer we've since retargeted away from
        ctrl = frame.ctrl or {}
        if int(ctrl.get("seq_no", 0)) != 2:
            return
        if int(ctrl.get("status", 1)) != 0:
            self._log("AUTH_FAILED", {"status": ctrl.get("status")})
            self._ctx._assoc_state = AssocState.UNASSOCIATED
            self._first_beacon_seen = False
            return

        self._sta_auth_resp_t = float(self._sim.engine.now)
        self._ctx._assoc_state = AssocState.AUTHENTICATED
        self._log("AUTH_DONE", {
            "auth_time_ms": round((self._sta_auth_resp_t - self._sta_auth_req_t) * 1000, 3),
        })

        # Immediately proceed to association
        self._ctx._assoc_state = AssocState.ASSOCIATING
        self._send_assoc_req(dst=frame.src)

    def _send_assoc_req(self, dst: int) -> None:
        self._sta_assoc_req_t = float(self._sim.engine.now)
        frame = self._make_mgmt_frame(
            ftype=FrameType.ASSOC_REQ,
            dst=dst,
            size=self._assoc_req_size,
            ctrl={
                "capability": 0x0001,
                "listen_interval": 10,
                "s1g_cap": True,
            },
        )
        self._log("ASSOC_REQ_TX", {"dst": dst})
        self._queue_for_tx(frame)

    def on_assoc_req_acked(self) -> None:
        """Called by DCF _on_success when ASSOC_REQ ACK is received."""
        self._log("ASSOC_REQ_ACKED", {})

    def on_assoc_resp_received(self, frame: MacFrame) -> None:
        """Called by MacLayer when ASSOC_RESP frame arrives at STA."""
        # See on_auth_resp_received -- same shadow-handshake dispatch.
        if int(frame.src) == self._shadow_peer and self._shadow_state == AssocState.ASSOCIATING:
            self._on_shadow_assoc_resp(frame)
            return
        if self._ctx._assoc_state != AssocState.ASSOCIATING:
            return
        if int(frame.src) != self._target_peer:
            return  # stale response from a peer we've since retargeted away from
        ctrl = frame.ctrl or {}
        if int(ctrl.get("status", 1)) != 0:
            self._log("ASSOC_FAILED", {"status": ctrl.get("status")})
            self._ctx._assoc_state = AssocState.UNASSOCIATED
            self._first_beacon_seen = False
            return

        aid = int(ctrl.get("aid", self._node.node_id))
        self._sta_assoc_t = float(self._sim.engine.now)

        self._ctx._assoc_state = AssocState.ASSOCIATED
        self._ctx._aid = aid
        # Who this STA actually associated with (the real AP, id 0, or a
        # relay's own id) -- read by NetworkLayer.resolve_next_hop() so
        # uplink routing follows the STA's live association rather than
        # its fixed topology-build-time relay assignment (see net.py).
        self._ctx._assoc_peer_id = int(frame.src)
        self._arm_beacon_liveness_timer()

        scan_s = self._sta_beacon_t    - self._sta_start_t
        auth_s = self._sta_auth_resp_t - self._sta_auth_req_t
        req_s  = self._sta_assoc_t     - self._sta_assoc_req_t
        tot_s  = self._sta_assoc_t     - self._sta_start_t

        self._log("ASSOCIATED", {
            "aid": aid,
            "total_ms":    round(tot_s  * 1000, 3),
            "scan_ms":     round(scan_s * 1000, 3),
            "auth_ms":     round(auth_s * 1000, 3),
            "assoc_req_ms":round(req_s  * 1000, 3),
        })

        # Write timing into sim.stats
        s = self._sim.stats
        s.assoc_scan_time[self._node.node_id]  = scan_s
        s.assoc_auth_time[self._node.node_id]  = auth_s
        s.assoc_req_time[self._node.node_id]   = req_s
        s.assoc_total_time[self._node.node_id] = tot_s

        # Notify TwtManager (or any other post-association hook)
        cb = getattr(self._ctx, "_on_associated_cb", None)
        if callable(cb):
            try:
                cb()
            except Exception:
                pass

        # A relay only starts advertising itself (broadcasting its own
        # beacon for STAs to join) once it has genuinely joined the real
        # AP itself -- see facade.py's start()/relay_start_beacons() for
        # why. relay_start_beacons() is idempotent (won't double-start the
        # loop if this node somehow re-associates after a later
        # disconnect), so no extra guard is needed here.
        if self._node.role == "RELAY":
            self._mac.relay_start_beacons()

        # Kick DCF — data frames may now be pending in _txq
        self._mac.dcf.drive(make_data_frame_cb=self._mac._make_data_frame)

    # ------------------------------------------------------------------
    # DCF callback: dispatches ACK for a management frame
    # ------------------------------------------------------------------

    def _on_mgmt_frame_acked(self, frame: MacFrame) -> None:
        """
        Called by DCF._on_success when a management frame TX+ACK completes.
        Advances the local state machine for the relevant frame type.
        """
        ftype = getattr(frame, "ftype", None)
        ctrl  = frame.ctrl or {}

        if ftype == FrameType.AUTH:
            if ctrl.get("for") == "req":
                self.on_auth_req_acked()
            # AP side AUTH_RESP acked — nothing extra needed
        elif ftype == FrameType.ASSOC_REQ:
            self.on_assoc_req_acked()
        # ASSOC_RESP acked on AP side — nothing extra needed

    def _on_mgmt_frame_dropped(self, frame: MacFrame) -> None:
        """
        Called by DCF._handle_retry_or_drop when a management frame exhausts
        all DCF retries.  Re-queues the frame so association can still complete.
        A short random back-off delay prevents an immediate retry storm.
        """
        ftype = getattr(frame, "ftype", None)
        ctrl  = frame.ctrl or {}
        self._log("MGMT_DROP_RETRY", {"ftype": str(ftype), "retry": frame.retry})

        if self._serves_associations():
            # AP/relay re-queues the response
            if ftype == FrameType.AUTH and ctrl.get("for") == "resp":
                dst = frame.dst
                self._sim.engine.schedule_in(
                    self._ctx.difs,
                    lambda: self._queue_for_tx(self._make_mgmt_frame(
                        FrameType.AUTH, dst, self._auth_size,
                        {"algorithm": 0, "seq_no": 2, "status": 0, "for": "resp"},
                    )),
                )
            elif ftype == FrameType.ASSOC_RESP:
                dst = frame.dst
                ctrl_copy = dict(ctrl)
                self._sim.engine.schedule_in(
                    self._ctx.difs,
                    lambda: self._queue_for_tx(self._make_mgmt_frame(
                        FrameType.ASSOC_RESP, dst, self._assoc_resp_size, ctrl_copy,
                    )),
                )
            return

        # STA side — re-queue the request after a DIFS + random slot delay
        rng   = self._sim.engine.rng
        delay = self._ctx.difs + rng.randint(0, self._ctx.cw_min) * self._ctx.slot_time

        # Shadow (predictive pre-association) handshake retries are checked
        # FIRST, matched tightly against self._shadow_peer specifically --
        # the primary branches below don't check frame.dst at all (a
        # pre-existing property, left alone: retargeting via STUCK_RETARGET
        # means a stale primary frame's dst can legitimately differ from
        # self._target_peer, and the original code already tolerates that).
        # Checking shadow first with an explicit dst match means a shadow
        # frame can never be misrouted into the looser primary branch, even
        # in the narrow case where a stale shadow frame surfaces just after
        # _roam_to has already reset shadow state and moved the primary
        # machine into AUTHENTICATING for an unrelated target.
        if ftype == FrameType.AUTH and self._shadow_state == AssocState.AUTHENTICATING and int(frame.dst) == self._shadow_peer:
            dst = frame.dst
            self._sim.engine.schedule_in(delay, lambda: self._send_shadow_auth_req(dst))
        elif ftype == FrameType.ASSOC_REQ and self._shadow_state == AssocState.ASSOCIATING and int(frame.dst) == self._shadow_peer:
            dst = frame.dst
            self._sim.engine.schedule_in(delay, lambda: self._send_shadow_assoc_req(dst))
        elif ftype == FrameType.AUTH and self._ctx._assoc_state == AssocState.AUTHENTICATING:
            dst = frame.dst
            self._sim.engine.schedule_in(delay, lambda: self._send_auth_req(dst))
        elif ftype == FrameType.ASSOC_REQ and self._ctx._assoc_state == AssocState.ASSOCIATING:
            dst = frame.dst
            self._sim.engine.schedule_in(delay, lambda: self._send_assoc_req(dst))

    # ------------------------------------------------------------------
    # AP / relay side: respond to incoming management frames
    # ------------------------------------------------------------------

    def on_auth_req_received(self, frame: MacFrame) -> None:
        """AP or relay: respond to AUTH Request with AUTH Response. A relay
        hosting its own local association point (see
        MacLayer.relay_start_beacons) lets a STA that cannot hear the real
        AP directly still join the BSS via the relay."""
        if not self._serves_associations():
            return
        sta_id = frame.src
        self._ap_auth_req_t[sta_id] = float(self._sim.engine.now)
        self._log("AP_AUTH_REQ", {"sta": sta_id})

        resp = self._make_mgmt_frame(
            ftype=FrameType.AUTH,
            dst=sta_id,
            size=self._auth_size,
            ctrl={"algorithm": 0, "seq_no": 2, "status": 0, "for": "resp"},
        )
        self._queue_for_tx(resp)

    def on_assoc_req_received(self, frame: MacFrame) -> None:
        """AP or relay: assign AID and respond with ASSOC Response. RAW
        slot scheduling is decided BSS-wide by the real AP, so a STA
        associating via a relay is also mirrored into the real AP's own
        ``_associated_stas`` (see below) -- otherwise the AP's RAW policy
        would never learn the STA exists and could never grant it a slot."""
        if not self._serves_associations():
            return
        sta_id = frame.src
        self._ap_assoc_req_t[sta_id] = float(self._sim.engine.now)

        # AID = node_id preserves backward-compat with RAW slot grouping
        aid = sta_id
        self._ctx._associated_stas[sta_id] = aid
        if not self._is_ap():
            # Mirror into whichever AP THIS relay is itself uplinked to
            # (self._ctx._assoc_peer_id, set when the relay's own STA-side
            # association completed -- the same MacContext instance serves
            # both this relay's "AP-response-side" role for STAs below it
            # and its own "STA-side" uplink role, so the field is already
            # populated by the time it's serving associations at all under
            # normal operation). Previously hardcoded to node 0, which
            # silently mirrored into the wrong AP's table as soon as a
            # relay could be uplinked to anything other than the one AP
            # that happened to be node 0.
            target_ap_id = getattr(self._ctx, "_assoc_peer_id", None)
            if target_ap_id is None:
                target_ap_id = 0
            ap_node = self._sim.nodes.get(target_ap_id)
            ap_mac = getattr(ap_node, "mac", None)
            if ap_mac is not None:
                ap_mac.ctx._associated_stas[sta_id] = aid
        self._log("AP_ASSOC_REQ", {"sta": sta_id, "aid_assigned": aid,
                                    "via_relay": self._node.node_id if not self._is_ap() else None})

        resp = self._make_mgmt_frame(
            ftype=FrameType.ASSOC_RESP,
            dst=sta_id,
            size=self._assoc_resp_size,
            ctrl={"status": 0, "aid": aid, "capability": 0x0001},
        )
        self._queue_for_tx(resp)

    # ------------------------------------------------------------------
    # Finalization (called from MacLayer.finalize)
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """
        If association was enabled but the STA never completed association
        (e.g., simulation ended mid-way), leave assoc_total_time at 0.0 so
        the compare script's assoc_frac correctly counts this as a failure.
        """
        if self._is_ap() or not self._assoc_enable:
            return
        if self._ctx._assoc_state == AssocState.ASSOCIATED:
            return  # already recorded in on_assoc_resp_received

        # assoc_total_time is intentionally NOT set for incomplete associations;
        # defaultdict(float) returns 0.0, which is the sentinel for "failed".
        self._log("ASSOC_INCOMPLETE", {
            "state": self._ctx._assoc_state,
            "node":  self._node.node_id,
        })
