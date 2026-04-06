from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from sim11ah.constants import MacState
from sim11ah.models import MacFrame, NetPDU

from sim11ah.mac.common import (
    _EDCA_DEFAULTS,
    AssocState,
    BAReorderBuffer,
    RawConfig,
    RawType,
    TIM_PAGE_COUNT,
)


@dataclass
class MacContext:
    # ---------------------------------------------------------------------
    # Immutable/shared references
    # ---------------------------------------------------------------------
    node: Any
    sim: Any
    cfg: Dict[str, Any]

    # ---------------------------------------------------------------------
    # Basic timing
    # ---------------------------------------------------------------------
    sifs: float = 0.0
    difs: float = 0.0
    slot_time: float = 52e-6
    ack_timeout_cfg: float = 0.0

    # ---------------------------------------------------------------------
    # Contention / retry
    # ---------------------------------------------------------------------
    cw_min: int = 15
    cw_max: int = 1023
    retry_limit: int = 7
    rts_threshold_bytes: int = 2347
    short_retry_limit: int = 7
    long_retry_limit: int = 4

    # ---------------------------------------------------------------------
    # Frame sizes
    # ---------------------------------------------------------------------
    ack_size_bytes: int = 14
    beacon_size_bytes: int = 120
    short_beacon_size_bytes: int = 40
    data_mac_overhead_bytes: int = 36
    ps_poll_size_bytes: int = 20
    ba_size_bytes: int = 32
    bar_size_bytes: int = 24
    rts_size_bytes: int = 20
    cts_size_bytes: int = 14
    cf_end_size_bytes: int = 20
    ndpa_size_bytes: int = 20
    probe_req_size_bytes: int = 40

    # ---------------------------------------------------------------------
    # RAW
    # ---------------------------------------------------------------------
    raw_enable: bool = False
    beacon_interval: float = 0.5
    dtim_period: int = 2
    raw_slot_duration: float = 0.004
    raw_num_slots: int = 6
    raw_num_groups: int = 1
    raw_nodes_per_group: int = 64
    raw_guard: float = 0.0005
    ack_guard: float = 0.0002
    raw_cross_slot: bool = False
    raw_type: int = RawType.GENERIC
    praw_enable: bool = False
    praw_period: int = 1
    praw_validity: int = 1
    praw_start_offset: int = 0
    praw_refresh: bool = False
    raw_start_time_us: int = 0
    raw_configs_enabled: bool = False

    _raw_configs: List[RawConfig] = field(default_factory=list)
    _raw_active_generation: int = 0
    _raw_event_gen: int = 0

    raw_assigned_slot: Optional[int] = None
    raw_allowed: bool = True
    _raw_slot_enter_t: float = 0.0
    _raw_slot_exit_t: float = 0.0

    # ---------------------------------------------------------------------
    # A-MPDU / Block ACK
    # ---------------------------------------------------------------------
    ampdu_enable: bool = False
    ampdu_max_subframes: int = 4
    ampdu_max_bytes: int = 8192
    _ba_rx_bufs: Dict[int, BAReorderBuffer] = field(default_factory=dict)
    _ba_tx_windows: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # A-MSDU
    # ---------------------------------------------------------------------
    amsdu_enable: bool = False
    amsdu_max_bytes: int = 3839
    amsdu_max_subframes: int = 8

    # ---------------------------------------------------------------------
    # Fragmentation
    # ---------------------------------------------------------------------
    frag_threshold_bytes: int = 2346
    reasm_timeout_s: float = 0.1
    _reasm_buf: Dict[Tuple[int, int], List[Optional[MacFrame]]] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Frame lifetime / queueing
    # ---------------------------------------------------------------------
    max_msdu_lifetime_s: float = 0.512
    txq_max_depth: int = 64

    # ---------------------------------------------------------------------
    # TWT
    # ---------------------------------------------------------------------
    twt_enable: bool = False
    _twt_agreements: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    _own_twt: Optional[Dict[str, Any]] = None
    _dozing: bool = False
    _twt_wake_gen: int = 0

    # ---------------------------------------------------------------------
    # EDCA
    # ---------------------------------------------------------------------
    _edca_queues: Dict[int, Deque[NetPDU]] = field(
        default_factory=lambda: {ac: deque() for ac in range(4)}
    )
    _edca_cw: Dict[int, int] = field(
        default_factory=lambda: {ac: _EDCA_DEFAULTS[ac]["cw_min"] for ac in range(4)}
    )
    _edca_bo: Dict[int, Optional[int]] = field(
        default_factory=lambda: {ac: None for ac in range(4)}
    )
    _edca_params: Dict[int, Dict[str, Any]] = field(
        default_factory=lambda: {ac: dict(v) for ac, v in _EDCA_DEFAULTS.items()}
    )
    _active_ac: Optional[int] = None

    # ---------------------------------------------------------------------
    # State / queues
    # ---------------------------------------------------------------------
    state: MacState = MacState.IDLE
    _txq: Deque[NetPDU] = field(default_factory=deque)
    _pending_frame: Optional[MacFrame] = None

    _cw: int = 15
    _backoff_slots_left: Optional[int] = None
    _backoff_tick_scheduled: bool = False
    _saved_backoff: Optional[int] = None

    _short_retry_count: int = 0
    _long_retry_count: int = 0
    _last_pending_pdu: Optional[NetPDU] = None

    _waiting_ack_for_frame_seq: Optional[int] = None
    _waiting_ack_for_frag_num: Optional[int] = None
    _ack_timer_token: int = 0

    _waiting_cts_for_frame_seq: Optional[int] = None
    _cts_timer_token: int = 0

    _pending_fragments: List[MacFrame] = field(default_factory=list)
    _pending_frag_index: int = 0

    _nav_end: float = 0.0

    _rx_seq_cache: Dict[int, Set[int]] = field(default_factory=dict)
    _rx_cache_maxsize: int = 256

    # ---------------------------------------------------------------------
    # PS buffering (AP side) + TIM
    # ---------------------------------------------------------------------
    _sta_ps_buffer: Dict[int, Deque[NetPDU]] = field(default_factory=dict)
    _tim_bitmap: Dict[int, bool] = field(default_factory=dict)
    _tim_pages: Dict[int, Dict[int, bool]] = field(
        default_factory=lambda: {p: {} for p in range(TIM_PAGE_COUNT)}
    )
    _tim_page_index: int = 0
    _sta_ps_mode: Dict[int, bool] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Association state
    # ---------------------------------------------------------------------
    _assoc_state: int = AssocState.UNASSOCIATED
    _aid: Optional[int] = None
    _next_aid: int = 1
    _associated_stas: Dict[int, int] = field(default_factory=dict)
    _sta_capabilities: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Beacon tracking
    # ---------------------------------------------------------------------
    _ap_beacon_count: int = 0
    _next_beacon_target: float = 0.0

    # ---------------------------------------------------------------------
    # Latency tracking
    # ---------------------------------------------------------------------
    _enqueue_time: Dict[int, float] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Local sequence counters
    # ---------------------------------------------------------------------
    _tx_seq_ctr: int = 0

    # ---------------------------------------------------------------------
    # DCF idle tracking
    # ---------------------------------------------------------------------
    _idle_since: Optional[float] = None

    # ---------------------------------------------------------------------
    # Channel utilization sampling
    # ---------------------------------------------------------------------
    _util_samples: List[Tuple[float, bool]] = field(default_factory=list)
    _util_window_s: float = 1.0

    # ---------------------------------------------------------------------
    # Adaptive RAW / grouping metrics
    # ---------------------------------------------------------------------
    _sta_tx_success: Dict[int, int] = field(default_factory=dict)
    _sta_tx_retries: Dict[int, int] = field(default_factory=dict)
    _sta_rx_success: Dict[int, int] = field(default_factory=dict)
    _sta_last_seen: Dict[int, float] = field(default_factory=dict)
    _sta_queue_hint: Dict[int, int] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------
    _stats: Dict[str, Any] = field(default_factory=dict)


def record_tx_success(ctx: MacContext, dst: int) -> None:
    ctx._sta_tx_success[dst] = ctx._sta_tx_success.get(dst, 0) + 1


def record_tx_retry(ctx: MacContext, dst: int) -> None:
    ctx._sta_tx_retries[dst] = ctx._sta_tx_retries.get(dst, 0) + 1


def record_rx_success(ctx: MacContext, src: int) -> None:
    ctx._sta_rx_success[src] = ctx._sta_rx_success.get(src, 0) + 1
    ctx._sta_last_seen[src] = ctx.sim.engine.now


def record_queue_hint(ctx: MacContext, dst: int) -> None:
    ctx._sta_queue_hint[dst] = ctx._sta_queue_hint.get(dst, 0) + 1


def clear_queue_hint(ctx: MacContext, dst: int) -> None:
    if dst in ctx._sta_queue_hint:
        ctx._sta_queue_hint[dst] = max(0, ctx._sta_queue_hint[dst] - 1)


def build_mac_context(node: Any, cfg: Dict[str, Any]) -> MacContext:
    mac_cfg = cfg["mac"]

    ctx = MacContext(
        node=node,
        sim=node.sim,
        cfg=cfg,

        sifs=float(mac_cfg["sifs"]),
        difs=float(mac_cfg["difs"]),
        slot_time=float(mac_cfg.get("slot_time", 52e-6)),
        ack_timeout_cfg=float(mac_cfg["ack_timeout"]),

        cw_min=int(mac_cfg.get("cw_min", 15)),
        cw_max=int(mac_cfg.get("cw_max", 1023)),
        retry_limit=int(mac_cfg.get("retry_limit", 7)),
        rts_threshold_bytes=int(mac_cfg.get("rts_threshold_bytes", 2347)),
        short_retry_limit=int(mac_cfg.get("short_retry_limit", 7)),
        long_retry_limit=int(mac_cfg.get("long_retry_limit", 4)),

        ack_size_bytes=int(mac_cfg["ack_size_bytes"]),
        beacon_size_bytes=int(mac_cfg["beacon_size_bytes"]),
        short_beacon_size_bytes=int(mac_cfg.get("short_beacon_size_bytes", 40)),
        data_mac_overhead_bytes=int(mac_cfg["data_mac_overhead_bytes"]),
        ps_poll_size_bytes=int(mac_cfg.get("ps_poll_size_bytes", 20)),
        ba_size_bytes=int(mac_cfg.get("ba_size_bytes", 32)),
        bar_size_bytes=int(mac_cfg.get("bar_size_bytes", 24)),
        rts_size_bytes=int(mac_cfg.get("rts_size_bytes", 20)),
        cts_size_bytes=int(mac_cfg.get("cts_size_bytes", 14)),
        cf_end_size_bytes=int(mac_cfg.get("cf_end_size_bytes", 20)),
        ndpa_size_bytes=int(mac_cfg.get("ndpa_size_bytes", 20)),
        probe_req_size_bytes=int(mac_cfg.get("probe_req_size_bytes", 40)),

        raw_enable=bool(mac_cfg["raw_enable"]),
        beacon_interval=float(mac_cfg["beacon_interval"]),
        dtim_period=int(mac_cfg["dtim_period"]),
        raw_slot_duration=float(mac_cfg["raw_slot_duration"]),
        raw_num_slots=int(mac_cfg["raw_num_slots"]),
        raw_num_groups=int(mac_cfg.get("raw_num_groups", 1)),
        raw_nodes_per_group=int(mac_cfg.get("raw_nodes_per_group", 64)),
        raw_guard=float(mac_cfg.get("raw_guard", 0.0005)),
        ack_guard=float(mac_cfg.get("ack_guard", 0.0002)),
        raw_cross_slot=bool(mac_cfg.get("raw_cross_slot", False)),
        raw_type=int(mac_cfg.get("raw_type", RawType.GENERIC)),
        praw_enable=bool(mac_cfg.get("praw_enable", False)),
        praw_period=int(mac_cfg.get("praw_period", 1)),
        praw_validity=int(mac_cfg.get("praw_validity", 1)),
        praw_start_offset=int(mac_cfg.get("praw_start_offset", 0)),
        praw_refresh=bool(mac_cfg.get("praw_refresh", False)),
        raw_start_time_us=int(mac_cfg.get("raw_start_time_us", 0)),
        raw_configs_enabled=bool(mac_cfg.get("raw_configs_enable", False)),

        ampdu_enable=bool(mac_cfg.get("ampdu_enable", False)),
        ampdu_max_subframes=int(mac_cfg.get("ampdu_max_subframes", 4)),
        ampdu_max_bytes=int(mac_cfg.get("ampdu_max_bytes", 8192)),

        amsdu_enable=bool(mac_cfg.get("amsdu_enable", False)),
        amsdu_max_bytes=int(mac_cfg.get("amsdu_max_bytes", 3839)),
        amsdu_max_subframes=int(mac_cfg.get("amsdu_max_subframes", 8)),

        frag_threshold_bytes=int(mac_cfg.get("frag_threshold_bytes", 2346)),
        reasm_timeout_s=float(mac_cfg.get("reasm_timeout_s", 0.1)),

        max_msdu_lifetime_s=float(mac_cfg.get("max_msdu_lifetime_s", 0.512)),
        txq_max_depth=int(mac_cfg.get("txq_max_depth", 64)),

        twt_enable=bool(mac_cfg.get("twt_enable", False)),

        raw_allowed=(not bool(mac_cfg["raw_enable"])) or (node.node_id == 0),
        _cw=int(mac_cfg.get("cw_min", 15)),
        _util_window_s=float(mac_cfg.get("util_window_s", 1.0)),
    )

    for ac in range(4):
        key = f"edca_ac{ac}"
        if key in mac_cfg:
            ctx._edca_params[ac].update(mac_cfg[key])

    ctx._stats = {
        "tx_unicast": 0,
        "tx_broadcast": 0,
        "tx_retries": 0,
        "tx_drops": 0,
        "tx_rts": 0,
        "tx_cts": 0,
        "tx_cf_end": 0,
        "rx_unicast": 0,
        "rx_broadcast": 0,
        "rx_duplicate": 0,
        "rx_crc_error": 0,
        "backoff_freezes": 0,
        "raw_deferred": 0,
        "raw_not_scheduled": 0,
        "raw_groups_built": 0,
        "raw_configs_built": 0,
        "ps_polls_sent": 0,
        "ps_polls_rcvd": 0,
        "twt_wake_entries": 0,
        "nav_updates": 0,
        "ampdu_tx": 0,
        "ampdu_rx": 0,
        "amsdu_tx": 0,
        "amsdu_rx": 0,
        "frag_tx": 0,
        "frag_rx": 0,
        "msdu_ttl_expired": 0,
        "txq_tail_drops": 0,
        "deauths_sent": 0,
        "deauths_rcvd": 0,
        "total_latency_s": 0.0,
        "latency_samples": 0,
        "raw_scheduled": 0,
        "raw_enter_count": 0,
        "raw_exit_count": 0,
        "raw_fit_blocked": 0,
        "raw_fit_pass": 0,
        "raw_fit_fail": 0,
        "raw_backoff_saved": 0,
        "raw_backoff_restored": 0,
        "raw_sleep_transitions": 0,

        "phy_collisions": 0,
        "phy_half_duplex_collisions": 0,
        "phy_per_drops": 0,
        "phy_below_sensitivity": 0,
        "phy_unsupported_mode": 0,
        
    }

    return ctx