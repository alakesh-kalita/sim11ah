from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sim11ah.engine import EventEngine
from sim11ah.logger import SimLogger
from sim11ah.stats import SimStats
from sim11ah.models import MacFrame, NetPDU, Packet
from sim11ah.topology import Topology


class Simulator:
    def __init__(self, config: Dict[str, Any], seed: int):
        self.config = config
        self.engine = EventEngine(seed=seed)

        max_logs = int(config.get("sim", {}).get("max_log_records", 0))
        self.logger = SimLogger(max_records=max_logs) if max_logs else SimLogger()

        self.stats = SimStats()
        self.topology: Optional[Topology] = None
        self.nodes: Dict[int, "Node"] = {}

        self.frame_seq_ctr: int = 0
        self.packet_seq_ctr: int = 0

        sim_cfg = config.get("sim", {})
        self.frame_seq_max: int = int(sim_cfg.get("frame_seq_max", 0))
        self.packet_seq_max: int = int(sim_cfg.get("packet_seq_max", 0))

        # Global medium occupancy: (start, end, tx_id, dst_id, frame_seq)
        self._medium_tx: List[Tuple[float, float, int, int, int]] = []

        self._started: bool = False
        self._finalized: bool = False

    # ------------------------------------------------------------------
    # Medium helpers
    # ------------------------------------------------------------------

    def _medium_gc(self, now: float) -> None:
        if not self._medium_tx:
            return
        self._medium_tx = [x for x in self._medium_tx if x[1] > now]

    def register_medium_tx(
        self,
        start: float,
        end: float,
        tx_id: int,
        dst_id: int,
        frame_seq: int,
    ) -> None:
        self._medium_gc(start)
        self._medium_tx.append(
            (float(start), float(end), int(tx_id), int(dst_id), int(frame_seq))
        )

    def medium_is_busy(self, t: float) -> bool:
        self._medium_gc(t)
        for t0, t1, _tx, _dst, _fseq in self._medium_tx:
            if t0 <= t < t1:
                return True
        return False

    def medium_next_idle_time(self, t: float) -> float:
        self._medium_gc(t)
        next_end: Optional[float] = None
        for t0, t1, _tx, _dst, _fseq in self._medium_tx:
            if t0 <= t < t1:
                if next_end is None or t1 < next_end:
                    next_end = t1
        return t if next_end is None else next_end

    # ------------------------------------------------------------------
    # Sequence numbers
    # ------------------------------------------------------------------

    def next_frame_seq(self) -> int:
        if self.frame_seq_max > 0:
            self.frame_seq_ctr = (self.frame_seq_ctr % self.frame_seq_max) + 1
        else:
            self.frame_seq_ctr += 1
        return self.frame_seq_ctr

    def next_packet_seq(self) -> int:
        if self.packet_seq_max > 0:
            self.packet_seq_ctr = (self.packet_seq_ctr % self.packet_seq_max) + 1
        else:
            self.packet_seq_ctr += 1
        return self.packet_seq_ctr

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        *,
        node_id: int,
        layer: str,
        event: str,
        details: Optional[Dict[str, Any]] = None,
        frame: Optional[MacFrame] = None,
        net_pdu: Optional[NetPDU] = None,
        packet: Optional[Packet] = None,
    ) -> None:
        d = dict(details) if details else {}

        top: Dict[str, Any] = {
            "time": float(self.engine.now),
            "node_id": int(node_id),
            "layer": str(layer),
            "event": str(event),
            "packet_seq": None,
            "net_seq": None,
            "tp_seq": None,
            "frame_seq": None,
            "tx_seq": None,
            "ftype": None,
            "src": None,
            "dst": None,
            "next_hop": None,
            "details": d,
        }

        if packet is None and net_pdu is not None:
            packet = net_pdu.packet
        if net_pdu is None and frame is not None:
            net_pdu = frame.net_pdu

        if packet is not None:
            top["packet_seq"] = getattr(packet, "packet_seq", None)
            top["tp_seq"] = getattr(packet, "tp_seq", None)
            top["src"] = getattr(packet, "src", top["src"])
            top["dst"] = getattr(packet, "dst", top["dst"])

        if net_pdu is not None:
            top["net_seq"] = getattr(net_pdu, "net_seq", None)
            top["next_hop"] = getattr(net_pdu, "next_hop", None)

            pkt2 = getattr(net_pdu, "packet", None)
            if pkt2 is not None and top["packet_seq"] is None:
                top["packet_seq"] = getattr(pkt2, "packet_seq", None)
                top["tp_seq"] = getattr(pkt2, "tp_seq", None)
                top["src"] = getattr(pkt2, "src", top["src"])
                top["dst"] = getattr(pkt2, "dst", top["dst"])

        if frame is not None:
            top["frame_seq"] = getattr(frame, "frame_seq", None)
            top["tx_seq"] = getattr(frame, "tx_seq", None)
            ftype = getattr(frame, "ftype", None)
            top["ftype"] = str(ftype) if ftype is not None else None
            top["src"] = getattr(frame, "src", top["src"])
            top["dst"] = getattr(frame, "dst", top["dst"])

            pdu2 = getattr(frame, "net_pdu", None)
            if pdu2 is not None:
                if top["net_seq"] is None:
                    top["net_seq"] = getattr(pdu2, "net_seq", None)
                if top["next_hop"] is None:
                    top["next_hop"] = getattr(pdu2, "next_hop", None)

        self.logger.log(top)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        # Let each layer manage its own startup behavior.
        # This avoids double-starting AP beacon processes.
        for n in self.nodes.values():
            n.start()

    def stop(self) -> None:
        self.engine.stop()

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        for n in self.nodes.values():
            n.stop()

        for n in self.nodes.values():
            n.finalize()

        # Save final simulation time for throughput / summary calculations
        self.stats.sim_time = float(self.engine.now)

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def run(self, sim_time: float) -> None:
        self.start()
        self.engine.run(until=float(sim_time))

    def run_for(self, dt: float) -> None:
        self.start()
        self.engine.run(until=self.engine.now + max(0.0, float(dt)))

    def step(self, dt: float) -> None:
        self.start()
        end = self.engine.now + max(0.0, float(dt))
        self.engine.run(until=end)

    def step_events(self, n: int = 1, dt_limit: Optional[float] = None) -> int:
        self.start()
        until = None
        if dt_limit is not None:
            until = self.engine.now + max(0.0, float(dt_limit))
        return self.engine.step(n=int(n), until=until)

    def run_and_finalize(self, sim_time: float) -> None:
        self.run(sim_time)
        self.finalize()

    # ------------------------------------------------------------------
    # Convenience helpers for experiments
    # ------------------------------------------------------------------

    def export_logs_csv(self, path: str) -> int:
        return self.logger.export_logs_csv(path)

    def get_log_counts(self) -> Dict[str, Any]:
        return {
            "by_layer": self.logger.count_by_layer(),
            "by_event": self.logger.count_by_event(),
            "by_node": self.logger.count_by_node(),
        }

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sim_time": float(self.engine.now),
            "engine_stats": self.engine.get_stats() if hasattr(self.engine, "get_stats") else {},
        }

        if hasattr(self.stats, "summary"):
            try:
                out["sim_stats"] = self.stats.summary()
            except Exception:
                out["sim_stats"] = {}
        else:
            out["sim_stats"] = {}

        return out