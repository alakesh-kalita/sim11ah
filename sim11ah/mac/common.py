from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sim11ah.models import NetPDU


# ---------------------------------------------------------------------------
# Access-Category constants
# ---------------------------------------------------------------------------
AC_BK = 0
AC_BE = 1
AC_VI = 2
AC_VO = 3

_EDCA_DEFAULTS: Dict[int, Dict[str, Any]] = {
    AC_BK: {"cw_min": 15, "cw_max": 1023, "aifsn": 7, "txop_limit": 0.0},
    AC_BE: {"cw_min": 15, "cw_max": 1023, "aifsn": 3, "txop_limit": 0.0},
    AC_VI: {"cw_min": 7,  "cw_max": 15,   "aifsn": 2, "txop_limit": 3.008e-3},
    AC_VO: {"cw_min": 3,  "cw_max": 7,    "aifsn": 2, "txop_limit": 1.504e-3},
}


# ---------------------------------------------------------------------------
# RAW type constants
# ---------------------------------------------------------------------------
class RawType:
    GENERIC = 0
    SOUNDING = 1
    SIMPLEX = 2
    TRIGGERING = 3


# ---------------------------------------------------------------------------
# RAW timing helpers
# ---------------------------------------------------------------------------
# RAW Slot Duration Count encoding, IEEE 802.11ah-2016 Section 9.4.2.200:
#   slot_us = 500 + cslot_index x 120 us (11-bit Slot Duration Count field,
#   cslot_index in [0, 2047], giving a max slot duration of 246,140 us).
# This matches the same formula used by the imec-idlab ns-3 802.11ah module
# and the Morse Micro RAW driver; it is the standard encoding, not a
# vendor-specific convention.
MORSE_RAW_MIN_SLOT_DURATION_US = 500


def cslot_to_us(x: int) -> int:
    return MORSE_RAW_MIN_SLOT_DURATION_US + (int(x) * 120)


def us_to_cslot(x_us: float) -> int:
    return int((float(x_us) - MORSE_RAW_MIN_SLOT_DURATION_US) / 120.0)


def us_to_two_tu(x_us: float) -> int:
    return int(float(x_us) / (1024.0 * 2.0))


def two_tu_to_us(x: int) -> int:
    return int(x) * 2048


# ---------------------------------------------------------------------------
# TIM page constants
# ---------------------------------------------------------------------------
TIM_PAGE_COUNT = 4
TIM_AIDS_PER_PAGE = 512


# ---------------------------------------------------------------------------
# Association state machine
# ---------------------------------------------------------------------------
class AssocState:
    UNASSOCIATED = 0
    AUTHENTICATING = 1
    AUTHENTICATED = 2
    ASSOCIATING = 3
    ASSOCIATED = 4


# ---------------------------------------------------------------------------
# RAW config/runtime models
# ---------------------------------------------------------------------------
@dataclass
class RawBeaconSpreading:
    nominal_sta_per_beacon: int = 0
    max_spread: int = 0
    last_aid: int = 0


@dataclass
class RawPeriodic:
    periodicity: int = 0
    validity: int = 0
    start_offset: int = 0
    cur_validity: int = 0
    cur_start_offset: int = 0
    refresh_praw: bool = False


@dataclass
class RawSlotDefinition:
    num_slots: int = 1
    slot_duration_us: int = MORSE_RAW_MIN_SLOT_DURATION_US
    cross_slot_boundary: bool = False


@dataclass
class RawConfig:
    id: int
    raw_type: int = RawType.GENERIC
    start_aid: int = 1
    end_aid: int = 1
    start_time_us: int = 0
    slot_definition: RawSlotDefinition = field(default_factory=RawSlotDefinition)
    beacon_spreading: RawBeaconSpreading = field(default_factory=RawBeaconSpreading)
    periodic: RawPeriodic = field(default_factory=RawPeriodic)
    enabled: bool = True
    dynamic_beacon_idx: Optional[int] = None

    # runtime/cache
    start_aid_idx: int = -1
    end_aid_idx: int = -1

    def is_periodic(self) -> bool:
        return self.periodic.periodicity > 0 and self.periodic.validity > 0

    def has_dynamic_bcn_idx(self) -> bool:
        return self.dynamic_beacon_idx is not None

    def is_valid(self) -> bool:
        return (
            self.slot_definition.slot_duration_us > 0
            and self.start_aid > 0
            and self.end_aid >= self.start_aid
            and self.slot_definition.num_slots > 0
        )


# ---------------------------------------------------------------------------
# Block-ACK reorder / scoreboard buffer
# ---------------------------------------------------------------------------
class BAReorderBuffer:
    def __init__(self, win_size: int = 64) -> None:
        self.win_size = int(win_size)
        self.win_start = 0
        self._buf: Dict[int, NetPDU] = {}

    def insert(self, seq: int, pdu: NetPDU) -> List[NetPDU]:
        slot = seq % self.win_size
        self._buf[slot] = pdu
        deliverable: List[NetPDU] = []
        while True:
            cur = self.win_start % self.win_size
            if cur not in self._buf:
                break
            deliverable.append(self._buf.pop(cur))
            self.win_start += 1
        return deliverable

    def bitmap(self) -> int:
        bits = 0
        for i in range(self.win_size):
            if (self.win_start + i) % self.win_size in self._buf:
                bits |= (1 << i)
        return bits

    def reset(self) -> None:
        self._buf.clear()
        self.win_start = 0


# ---------------------------------------------------------------------------
# Multi-AP ownership check
# ---------------------------------------------------------------------------
def sta_belongs_to_ap(ctx: Any, sta_node: Any) -> bool:
    """Does sta_node actually belong to ctx's own AP right now -- either
    directly associated with it, or associated with a relay that is itself
    uplinked to it? Under multi-AP, "belongs to me" is not simply
    "peer == my id": a relay-served STA's own _assoc_peer_id is the
    RELAY's node_id, not the AP's, even though it's legitimately part of
    that AP's BSS (see AssocManager.on_assoc_req_received's mirror-into-
    uplink-AP logic). Shared by RawEngine.connected_aids() and any RAW
    policy that reads ctx._associated_stas directly as its own fallback
    (e.g. raw_policy_adaptive.py) -- both need the identical ownership
    rule, and this is the one place it should be edited.

    Returns True (lenient/include) when peer info is unavailable, the same
    default this had before any ownership filtering existed (single-AP)."""
    my_id = ctx.node.node_id
    mac_ctx = getattr(getattr(sta_node, "mac", None), "ctx", None)
    peer = getattr(mac_ctx, "_assoc_peer_id", None) if mac_ctx is not None else None
    if peer is None:
        return True
    if int(peer) == int(my_id):
        return True
    sim_nodes = getattr(ctx.sim, "nodes", {})
    peer_node = sim_nodes.get(int(peer))
    if getattr(peer_node, "role", None) == "RELAY":
        relay_ctx = getattr(getattr(peer_node, "mac", None), "ctx", None)
        relay_upstream = getattr(relay_ctx, "_assoc_peer_id", None) if relay_ctx is not None else None
        return relay_upstream is not None and int(relay_upstream) == int(my_id)
    return False  # associated with a different AP directly -- not mine