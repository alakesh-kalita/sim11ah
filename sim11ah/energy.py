"""
IEEE 802.11ah Node Energy Model
================================
Per-node radio energy consumption tracking across four power states:

  TX    — transmitting a frame (DATA, ACK, beacon, RTS/CTS)
  RX    — receiving/decoding a frame (including failed receptions)
  IDLE  — radio ON, no TX/RX (backoff countdown, carrier sense, waiting)
  SLEEP — radio in deep sleep (RAW slot of another group, TWT doze)

Power reference values (IEEE 802.11ah sub-GHz IoT radio, MORSE MG100):
  P_tx    = 180 mW   (0 dBm output power; scales with tx_power_dbm setting)
  P_rx    =  62 mW   (active reception including preamble detect + decode)
  P_idle  =  20 mW   (carrier sense / CCA / CSMA backoff listening)
  P_sleep =   0.5 mW (deep sleep, reference oscillator only)

These defaults can be overridden via cfg["energy"] keys:
  tx_power_w, rx_power_w, idle_power_w, sleep_power_w

Energy breakdown per node (Joules):
  E_tx    = P_tx    × T_tx
  E_rx    = P_rx    × T_rx
  E_idle  = P_idle  × T_idle   (T_idle = T_sim − T_tx − T_rx − T_sleep)
  E_sleep = P_sleep × T_sleep
  E_total = E_tx + E_rx + E_idle + E_sleep

Derived per-node metrics:
  energy_per_bit_nj  = E_total × 1e9 / delivered_bits  (nJ/bit)
  energy_per_pkt_uj  = E_total × 1e6 / delivered_pkts  (µJ/packet)

References
----------
[1] MORSE Micro MG100 data sheet, sub-1 GHz 802.11ah module.
[2] Tian et al., "IEEE 802.11ah: A Long Range 802.11 WLAN at Sub 1 GHz,"
    IEEE Communications Magazine, 2016.
[3] Oteri et al., "Advanced power save mechanisms for outdoor 802.11ah
    sensor networks," IEEE WoWMoM 2015.
[4] Khorov et al., "A Tutorial on IEEE 802.11ah: A Wireless LAN for
    Internet of Things," IEEE Communications Surveys & Tutorials, 2019.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Power-state constants (MORSE MG100 reference, editable via config)
# ---------------------------------------------------------------------------
_DEFAULT_TX_W    = 0.180   # 180 mW — active transmit
_DEFAULT_RX_W    = 0.062   # 62 mW  — active receive
_DEFAULT_IDLE_W  = 0.020   # 20 mW  — CCA / backoff (radio on, no frame)
_DEFAULT_SLEEP_W = 0.0005  # 0.5 mW — deep sleep (RAW or TWT)


@dataclass
class EnergyConfig:
    tx_power_w:    float = _DEFAULT_TX_W
    rx_power_w:    float = _DEFAULT_RX_W
    idle_power_w:  float = _DEFAULT_IDLE_W
    sleep_power_w: float = _DEFAULT_SLEEP_W

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "EnergyConfig":
        e = cfg.get("energy", {})
        return cls(
            tx_power_w    = float(e.get("tx_power_w",    _DEFAULT_TX_W)),
            rx_power_w    = float(e.get("rx_power_w",    _DEFAULT_RX_W)),
            idle_power_w  = float(e.get("idle_power_w",  _DEFAULT_IDLE_W)),
            sleep_power_w = float(e.get("sleep_power_w", _DEFAULT_SLEEP_W)),
        )


@dataclass
class NodeEnergyBreakdown:
    node_id:        int
    tx_time_s:      float = 0.0
    rx_time_s:      float = 0.0
    sleep_time_s:   float = 0.0
    idle_time_s:    float = 0.0
    tx_j:           float = 0.0
    rx_j:           float = 0.0
    idle_j:         float = 0.0
    sleep_j:        float = 0.0
    total_j:        float = 0.0
    tx_frames:      int   = 0   # total TX events (DATA + ACK + control)
    tx_retries:     int   = 0   # TX events that were retransmissions
    rx_frames:      int   = 0   # total RX events (all receptions attempted)
    rx_ok:          int   = 0   # successful receptions
    # --- Retransmission energy breakdown (new) ---
    retx_tx_time_s: float = 0.0   # air-time of retransmitted frames only
    retx_tx_j:      float = 0.0   # TX energy attributable to retransmissions
    retx_pct:       float = 0.0   # retx_tx_j / total_j × 100 (%)


class _NodeTracker:
    """Accumulates raw time observations for one node."""

    def __init__(self, node_id: int) -> None:
        self.node_id          = node_id
        self.tx_total_s       = 0.0
        self.retx_tx_total_s  = 0.0   # air-time of retry frames only
        self.rx_total_s       = 0.0
        self.sleep_total_s    = 0.0
        self.tx_frames        = 0
        self.tx_retries       = 0
        self.rx_frames        = 0
        self.rx_ok_count      = 0
        self._sleep_start: Optional[float] = None

    def on_tx(self, duration_s: float, is_retry: bool) -> None:
        self.tx_total_s += duration_s
        self.tx_frames  += 1
        if is_retry:
            self.tx_retries      += 1
            self.retx_tx_total_s += duration_s

    def on_rx(self, duration_s: float, received_ok: bool) -> None:
        self.rx_total_s += duration_s
        self.rx_frames  += 1
        if received_ok:
            self.rx_ok_count += 1

    def on_sleep_start(self, now: float) -> None:
        if self._sleep_start is None:
            self._sleep_start = now

    def on_sleep_end(self, now: float) -> None:
        if self._sleep_start is not None:
            elapsed = max(0.0, now - self._sleep_start)
            self.sleep_total_s += elapsed
            self._sleep_start = None

    def finalize(self, sim_time: float, cfg: EnergyConfig) -> NodeEnergyBreakdown:
        # Close any open sleep interval (simulation ended while sleeping).
        if self._sleep_start is not None:
            self.on_sleep_end(sim_time)

        # Clamp accumulated radio-busy time to sim_time to avoid negative idle.
        tx_s    = min(self.tx_total_s,    sim_time)
        rx_s    = min(self.rx_total_s,    sim_time)
        sleep_s = min(self.sleep_total_s, sim_time)

        # Idle = everything that is not TX, RX, or sleep.
        # The three components can overlap slightly at boundaries (e.g., half-
        # duplex events) — we cap at zero rather than propagate negative time.
        idle_s = max(0.0, sim_time - tx_s - rx_s - sleep_s)

        retx_s  = min(self.retx_tx_total_s, tx_s)   # can't exceed total TX time
        tx_j    = cfg.tx_power_w    * tx_s
        rx_j    = cfg.rx_power_w    * rx_s
        idle_j  = cfg.idle_power_w  * idle_s
        sleep_j = cfg.sleep_power_w * sleep_s
        retx_j  = cfg.tx_power_w    * retx_s
        total_j = tx_j + rx_j + idle_j + sleep_j

        return NodeEnergyBreakdown(
            node_id        = self.node_id,
            tx_time_s      = tx_s,
            rx_time_s      = rx_s,
            sleep_time_s   = sleep_s,
            idle_time_s    = idle_s,
            tx_j           = tx_j,
            rx_j           = rx_j,
            idle_j         = idle_j,
            sleep_j        = sleep_j,
            total_j        = total_j,
            tx_frames      = self.tx_frames,
            tx_retries     = self.tx_retries,
            rx_frames      = self.rx_frames,
            rx_ok          = self.rx_ok_count,
            retx_tx_time_s = retx_s,
            retx_tx_j      = retx_j,
            retx_pct       = 100.0 * retx_j / total_j if total_j > 0 else 0.0,
        )


class EnergyModel:
    """
    Simulator-level energy model.

    Attached to ``sim.energy`` by Simulator.__init__.  PHY and RAW layers
    call the ``on_*`` methods; Simulator.finalize() calls ``finalize()``
    which writes results into ``sim.stats``.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg      = EnergyConfig.from_cfg(cfg)
        self._trackers: Dict[int, _NodeTracker] = {}

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _tracker(self, node_id: int) -> _NodeTracker:
        t = self._trackers.get(node_id)
        if t is None:
            t = _NodeTracker(node_id)
            self._trackers[node_id] = t
        return t

    # ------------------------------------------------------------------
    # PHY TX hooks
    # ------------------------------------------------------------------

    def on_tx_start(self, node_id: int, duration_s: float, is_retry: bool = False) -> None:
        """Called by PhyLayer.send() for every frame transmission."""
        self._tracker(node_id).on_tx(duration_s, is_retry)

    # ------------------------------------------------------------------
    # PHY RX hooks
    # ------------------------------------------------------------------

    def on_rx_start(self, node_id: int, duration_s: float, received_ok: bool = True) -> None:
        """Called by PhyLayer._rx_start() for every reception attempt."""
        self._tracker(node_id).on_rx(duration_s, received_ok)

    # ------------------------------------------------------------------
    # Sleep / wake hooks  (called by RawEngine)
    # ------------------------------------------------------------------

    def on_sleep_start(self, node_id: int, now: float) -> None:
        """Called when the MAC sets state → RAW_SLEEP."""
        self._tracker(node_id).on_sleep_start(now)

    def on_sleep_end(self, node_id: int, now: float) -> None:
        """Called when the MAC transitions out of RAW_SLEEP (raw_enter)."""
        self._tracker(node_id).on_sleep_end(now)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self, sim_time: float, stats: Any) -> Dict[int, NodeEnergyBreakdown]:
        """
        Compute final energy breakdown for every tracked node and write
        summary dicts into sim.stats energy fields.
        """
        results: Dict[int, NodeEnergyBreakdown] = {}

        for node_id, tracker in self._trackers.items():
            bd = tracker.finalize(sim_time, self._cfg)
            results[node_id] = bd
            stats.energy_tx_j[node_id]        = bd.tx_j
            stats.energy_rx_j[node_id]        = bd.rx_j
            stats.energy_idle_j[node_id]      = bd.idle_j
            stats.energy_sleep_j[node_id]     = bd.sleep_j
            stats.energy_total_j[node_id]     = bd.total_j
            stats.energy_retx_j[node_id]      = bd.retx_tx_j
            stats.tx_retries_per_node[node_id] = bd.tx_retries

        return results

    # ------------------------------------------------------------------
    # Read-only summary
    # ------------------------------------------------------------------

    @property
    def config(self) -> EnergyConfig:
        return self._cfg

    def tracker(self, node_id: int) -> Optional[_NodeTracker]:
        return self._trackers.get(node_id)
