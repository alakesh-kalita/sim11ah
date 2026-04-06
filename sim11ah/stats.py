from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
import numpy as np


@dataclass
class SimStats:
    # -----------------------------
    # Core counters
    # -----------------------------
    packets_generated: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    delivered_bytes: int = 0  # total successfully delivered payload bytes

    # -----------------------------
    # MAC / PHY counters
    # -----------------------------
    mac_retries: int = 0
    mac_ack_timeouts: int = 0
    mac_tx_attempts: int = 0

    phy_collisions: int = 0
    phy_half_duplex_collisions: int = 0
    phy_per_drops: int = 0
    phy_below_sensitivity: int = 0
    phy_unsupported_mode: int = 0

    # -----------------------------
    # Per-node counters
    # -----------------------------
    generated_by_src: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    delivered_by_dst: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    delivered_by_src: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # -----------------------------
    # Delay stats
    # -----------------------------
    delays: List[float] = field(default_factory=list)
    delays_per_node: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))

    # -----------------------------
    # RAW stats
    # -----------------------------
    raw_slot_tx_attempts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    raw_slot_data_success: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    raw_fit_blocked: int = 0
    raw_fit_fail: int = 0
    raw_fit_pass: int = 0

    raw_enter_count: int = 0
    raw_exit_count: int = 0

    raw_sleep_time: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    raw_active_time: Dict[int, float] = field(default_factory=lambda: defaultdict(float))

    # -----------------------------
    # Transport / network
    # -----------------------------
    transport_duplicates: int = 0
    net_duplicates: int = 0
    net_forwarded: int = 0

    # -----------------------------
    # Application
    # -----------------------------
    jitter_per_node: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    goodput_bps_per_node: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    tx_failures_by_src: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    app_in_flight_timeouts: int = 0

    # -----------------------------
    # Simulation metadata
    # -----------------------------
    sim_time: float = 0.0

    # ==========================================================
    # Derived Metrics
    # ==========================================================

    def throughput_bps(self) -> float:
        if self.sim_time <= 0:
            return 0.0
        return (self.delivered_bytes * 8.0) / self.sim_time

    def pdr(self) -> float:
        if self.packets_generated == 0:
            return 0.0
        return self.packets_delivered / self.packets_generated

    def avg_delay(self) -> float:
        return float(np.mean(self.delays)) if self.delays else 0.0

    def p95_delay(self) -> float:
        return float(np.percentile(self.delays, 95)) if self.delays else 0.0

    def max_delay(self) -> float:
        return max(self.delays) if self.delays else 0.0

    def fairness_index(self) -> float:
        values = list(self.delivered_by_src.values())
        if not values:
            return 0.0
        values = np.array(values, dtype=float)
        denom = len(values) * np.sum(values ** 2)
        if denom <= 0:
            return 0.0
        return float((values.sum() ** 2) / denom)

    def raw_block_rate(self) -> float:
        total = self.raw_fit_blocked + self.raw_fit_fail + self.raw_fit_pass
        if total == 0:
            return 0.0
        return self.raw_fit_blocked / total

    def raw_success_rate(self) -> float:
        total = self.raw_fit_pass + self.raw_fit_fail
        if total == 0:
            return 0.0
        return self.raw_fit_pass / total

    def summary(self) -> Dict:
        return {
            "throughput_bps": self.throughput_bps(),
            "pdr": self.pdr(),
            "avg_delay": self.avg_delay(),
            "p95_delay": self.p95_delay(),
            "max_delay": self.max_delay(),
            "fairness_index": self.fairness_index(),
            "mac_retries": self.mac_retries,
            "mac_ack_timeouts": self.mac_ack_timeouts,
            "mac_tx_attempts": self.mac_tx_attempts,
            "phy_collisions": self.phy_collisions,
            "phy_half_duplex_collisions": self.phy_half_duplex_collisions,
            "phy_per_drops": self.phy_per_drops,
            "phy_below_sensitivity": self.phy_below_sensitivity,
            "phy_unsupported_mode": self.phy_unsupported_mode,
            "net_duplicates": self.net_duplicates,
            "transport_duplicates": self.transport_duplicates,
            "net_forwarded": self.net_forwarded,
            "raw_block_rate": self.raw_block_rate(),
            "raw_success_rate": self.raw_success_rate(),
        }