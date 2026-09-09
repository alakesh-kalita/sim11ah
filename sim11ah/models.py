from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# --------------------------------------------------
# Application / Transport Packet
# --------------------------------------------------

@dataclass
class Packet:
    packet_seq: int
    src: int
    dst: int
    size_bytes: int

    # Backward-compatible timestamps
    gen_time: Optional[float] = None
    created_at: Optional[float] = None

    # Transport-related fields
    tp_seq: Optional[int] = None
    dscp: Optional[int] = None

    # Type / traffic class
    ptype: str = "DATA"
    traffic_type: str = "sensor"

    # Payload
    payload: bytes = field(default_factory=bytes)

    # Cross-layer metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.packet_seq = int(self.packet_seq)
        self.src = int(self.src)
        self.dst = int(self.dst)
        self.size_bytes = int(self.size_bytes)

        # Keep both fields available for old and new code
        if self.created_at is None and self.gen_time is None:
            self.created_at = 0.0
            self.gen_time = 0.0
        elif self.created_at is None and self.gen_time is not None:
            self.gen_time = float(self.gen_time)
            self.created_at = float(self.gen_time)
        elif self.created_at is not None and self.gen_time is None:
            self.created_at = float(self.created_at)
            self.gen_time = float(self.created_at)
        else:
            self.created_at = float(self.created_at)
            self.gen_time = float(self.gen_time)

        if self.tp_seq is not None:
            self.tp_seq = int(self.tp_seq)
        if self.dscp is not None:
            self.dscp = int(self.dscp)

        self.ptype = str(self.ptype)
        self.traffic_type = str(self.traffic_type)

        if self.payload is None:
            self.payload = b""

        if not isinstance(self.meta, dict):
            self.meta = dict(self.meta) if self.meta is not None else {}


# --------------------------------------------------
# Network Layer PDU
# --------------------------------------------------

@dataclass
class NetPDU:
    net_seq: int
    src: int
    dst: int
    next_hop: int
    packet: Packet

    ttl: int = 16
    hop_count: int = 0

    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.net_seq = int(self.net_seq)
        self.src = int(self.src)
        self.dst = int(self.dst)
        self.next_hop = int(self.next_hop)
        self.ttl = int(self.ttl)
        self.hop_count = int(self.hop_count)

        if not isinstance(self.meta, dict):
            self.meta = dict(self.meta) if self.meta is not None else {}


# --------------------------------------------------
# MAC Frame
# --------------------------------------------------

@dataclass
class MacFrame:
    ftype: Any
    src: int
    dst: int
    size_bytes: int

    frame_seq: int
    tx_seq: int

    retry: int = 0

    net_pdu: Optional[NetPDU] = None
    ctrl: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.src = int(self.src)
        self.dst = int(self.dst)
        self.size_bytes = int(self.size_bytes)
        self.frame_seq = int(self.frame_seq)
        self.tx_seq = int(self.tx_seq)
        self.retry = int(self.retry)

        if not isinstance(self.ctrl, dict):
            self.ctrl = dict(self.ctrl) if self.ctrl is not None else {}