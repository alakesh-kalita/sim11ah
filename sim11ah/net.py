from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from sim11ah.models import NetPDU, Packet


class NetworkLayer:
    """
    Star-topology network layer (infrastructure BSS):
      - STA uplink: always send to AP (node_id 0)
      - AP downlink: forwards to destination STA (or broadcast)

    Features:
      - duplicate detection by net_seq
      - AP forwarding creates a NEW NetPDU object
      - optional queue-depth drop
      - optional size-only fragmentation / reassembly
      - delivery counters / logging
    """

    def __init__(self, node: "Node", cfg: Dict[str, Any]):
        self.node = node
        self.sim = node.sim
        self.cfg = cfg

        net_cfg = cfg.get("net", {})
        self.max_hops = int(net_cfg.get("max_hops", 2))

        self._seen_cache_max = int(net_cfg.get("seen_cache_max", 256))
        self._seen_net_seqs: Dict[int, set] = {}

        self._seq_mod = int(net_cfg.get("seq_mod", 65535))
        self.net_seq_ctr = 0

        self.max_queue_depth = int(net_cfg.get("max_queue_depth", 0))

        self.enable_fragmentation = bool(net_cfg.get("enable_fragmentation", False))
        self.max_msdu_bytes = int(net_cfg.get("max_msdu_bytes", 2304))

        self.reassembly_timeout_s = float(net_cfg.get("reassembly_timeout_s", 5.0))

        # key = (src, frag_id) ->
        #   {"total": int, "got": set, "acc_bytes": int, "dst": int,
        #    "orig_size": int, "t0": float}
        self._reassembly: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_ap(self) -> bool:
        return int(self.node.node_id) == 0

    def _log(
        self,
        event: str,
        details: Dict[str, Any],
        net_pdu: Optional[NetPDU] = None,
        packet: Optional[Packet] = None,
    ) -> None:
        self.sim.log(
            node_id=self.node.node_id,
            layer="NET",
            event=event,
            details=details,
            net_pdu=net_pdu,
            packet=packet,
        )

    def _ensure_meta(self, obj: Any) -> Dict[str, Any]:
        try:
            meta = getattr(obj, "meta", None)
            if meta is None or not isinstance(meta, dict):
                setattr(obj, "meta", {})
            return getattr(obj, "meta")
        except Exception:
            return {}

    def _next_net_seq(self) -> int:
        self.net_seq_ctr = (self.net_seq_ctr % self._seq_mod) + 1
        return self.net_seq_ctr

    def _is_duplicate_pdu(self, pdu: NetPDU) -> bool:
        src = int(pdu.src)
        seq = int(pdu.net_seq)
        seen = self._seen_net_seqs.setdefault(src, set())
        if seq in seen:
            return True

        seen.add(seq)

        if len(seen) > self._seen_cache_max:
            try:
                evict = sorted(seen)[: max(1, self._seen_cache_max // 2)]
                for s in evict:
                    seen.discard(s)
            except Exception:
                seen.clear()
                seen.add(seq)

        return False

    def _notify_local_tx_failure(self, packet: Optional[Packet]) -> None:
        if packet is None:
            return
        try:
            self.node.app.on_tx_failure(packet)
        except Exception:
            pass

    def _cleanup_reassembly(self) -> None:
        if self.reassembly_timeout_s <= 0:
            return

        now = float(self.sim.engine.now)
        stale = []
        for key, buf in list(self._reassembly.items()):
            t0 = float(buf.get("t0", now))
            if (now - t0) > self.reassembly_timeout_s:
                stale.append(key)

        for key in stale:
            self._reassembly.pop(key, None)

        if stale:
            self._log("REASM_CLEAN", {"removed": len(stale), "remaining": len(self._reassembly)})

    # ------------------------------------------------------------------
    # Topology rules
    # ------------------------------------------------------------------

    def resolve_next_hop(self, dst: int) -> int:
        """
        Broadcast:
          STA -> AP
          AP  -> broadcast
        Unicast:
          STA -> AP
          AP  -> destination STA
        """
        if dst == -1:
            return -1 if self._is_ap() else 0
        return dst if self._is_ap() else 0

    # ------------------------------------------------------------------
    # Optional queue-depth drop
    # ------------------------------------------------------------------

    def _maybe_drop_for_queue(self, packet: Packet) -> bool:
        if self.max_queue_depth <= 0:
            return False

        dscp = getattr(packet, "dscp", 0)

        try:
            if hasattr(self.node.mac, "get_queue_depth"):
                qd = int(self.node.mac.get_queue_depth(dscp=dscp))
                if qd >= self.max_queue_depth:
                    self.sim.stats.packets_dropped += 1
                    self._log(
                        "DROP",
                        {"reason": "net_queue_full", "depth": qd, "dscp": dscp},
                        packet=packet,
                    )
                    self._notify_local_tx_failure(packet)
                    return True
                return False

            # fallback to likely internal queue locations
            depth = 0
            if hasattr(self.node.mac, "ctx"):
                ctx = self.node.mac.ctx
                if hasattr(ctx, "_txq"):
                    depth += len(ctx._txq)
                if hasattr(ctx, "_edca_queues"):
                    try:
                        depth += sum(len(q) for q in ctx._edca_queues.values())
                    except Exception:
                        pass

            if depth >= self.max_queue_depth:
                self.sim.stats.packets_dropped += 1
                self._log(
                    "DROP",
                    {"reason": "net_queue_full", "depth": depth, "dscp": dscp},
                    packet=packet,
                )
                self._notify_local_tx_failure(packet)
                return True
        except Exception:
            pass

        return False

    # ------------------------------------------------------------------
    # Hop counter
    # ------------------------------------------------------------------

    def _bump_hops(self, pdu: NetPDU) -> int:
        try:
            meta = self._ensure_meta(pdu.packet)
            hops = int(meta.get("hops", 0)) + 1
            meta["hops"] = hops
            return hops
        except Exception as e:
            self._log("WARN", {"reason": "bump_hops_failed", "error": str(e)}, net_pdu=pdu)
            return self.max_hops + 1

    # ------------------------------------------------------------------
    # Fragmentation (size-only)
    # ------------------------------------------------------------------

    def _make_packet_view(self, original: Packet, frag_size: int) -> Packet:
        try:
            data = dict(vars(original))
            data["size_bytes"] = int(frag_size)
            pkt = original.__class__(**data)
            return pkt
        except Exception:
            return original

    def _fragment_and_send(self, packet: Packet) -> None:
        total_bytes = int(getattr(packet, "size_bytes", 0))
        if total_bytes <= self.max_msdu_bytes:
            self._send_single(packet)
            return

        frag_total = int(math.ceil(total_bytes / float(self.max_msdu_bytes)))
        frag_id = self._next_net_seq()

        meta0 = self._ensure_meta(packet)
        meta0.setdefault("originated_by", self.node.node_id)
        meta0.setdefault("hops", 0)

        sent = 0
        for frag_num in range(frag_total):
            frag_size = min(self.max_msdu_bytes, total_bytes - sent)
            sent += frag_size

            frag_pkt = self._make_packet_view(packet, frag_size)
            frag_meta = self._ensure_meta(frag_pkt)
            frag_meta.update(
                {
                    "frag": True,
                    "frag_id": frag_id,
                    "frag_num": frag_num,
                    "frag_total": frag_total,
                    "frag_size": frag_size,
                    "frag_more": frag_num < (frag_total - 1),
                    "originated_by": meta0.get("originated_by", self.node.node_id),
                    "hops": meta0.get("hops", 0),
                    "orig_size_bytes": total_bytes,
                }
            )

            net_seq = self._next_net_seq()
            next_hop = self.resolve_next_hop(int(getattr(packet, "dst", -1)))

            pdu = NetPDU(
                net_seq=net_seq,
                src=int(getattr(packet, "src", self.node.node_id)),
                dst=int(getattr(packet, "dst", -1)),
                next_hop=next_hop,
                packet=frag_pkt,
            )

            self._log(
                "TX_FRAG",
                {
                    "frag_id": frag_id,
                    "frag_num": frag_num,
                    "frag_total": frag_total,
                    "frag_size": frag_size,
                    "next_hop": next_hop,
                },
                net_pdu=pdu,
                packet=frag_pkt,
            )
            self.node.mac.send_down(pdu)

    def _try_reassemble_and_deliver(self, pdu: NetPDU) -> bool:
        self._cleanup_reassembly()

        meta = self._ensure_meta(pdu.packet)
        if not meta.get("frag", False):
            return False

        try:
            src = int(pdu.src)
            frag_id = int(meta["frag_id"])
            frag_num = int(meta["frag_num"])
            frag_total = int(meta["frag_total"])
            frag_size = int(meta.get("frag_size", getattr(pdu.packet, "size_bytes", 0)))
            orig_size = int(meta.get("orig_size_bytes", frag_total * self.max_msdu_bytes))
        except Exception as e:
            self.sim.stats.packets_dropped += 1
            self._log("DROP", {"reason": "frag_meta_invalid", "error": str(e)}, net_pdu=pdu)
            return True

        key = (src, frag_id)
        buf = self._reassembly.setdefault(
            key,
            {
                "total": frag_total,
                "got": set(),
                "acc_bytes": 0,
                "dst": int(pdu.dst),
                "orig_size": orig_size,
                "t0": float(self.sim.engine.now),
            },
        )

        if frag_num in buf["got"]:
            self._log(
                "DROP",
                {"reason": "duplicate_fragment", "frag_id": frag_id, "frag_num": frag_num},
                net_pdu=pdu,
            )
            return True

        buf["got"].add(frag_num)
        buf["acc_bytes"] += frag_size

        self._log(
            "RX_FRAG",
            {
                "frag_id": frag_id,
                "frag_num": frag_num,
                "frag_total": buf["total"],
                "acc_bytes": buf["acc_bytes"],
            },
            net_pdu=pdu,
        )

        if len(buf["got"]) < int(buf["total"]):
            return True

        try:
            orig_pkt = self._make_packet_view(pdu.packet, int(buf.get("orig_size", orig_size)))
            orig_meta = self._ensure_meta(orig_pkt)
            orig_meta.update(meta)
            orig_meta["frag_reassembled"] = True
            orig_meta["frag_id"] = frag_id
            orig_meta["frag_total"] = int(buf["total"])
        except Exception:
            orig_pkt = pdu.packet

        try:
            del self._reassembly[key]
        except Exception:
            pass

        self._log(
            "DELIVER_REASSEMBLED",
            {"frag_id": frag_id, "src": src, "dst": int(pdu.dst)},
            packet=orig_pkt,
        )
        self.node.transport.recv_up_from_net(orig_pkt)
        return True

    # ------------------------------------------------------------------
    # Optional callbacks for MAC
    # ------------------------------------------------------------------

    def on_mac_queue_full(self, net_pdu: NetPDU) -> None:
        self._log("MAC_QUEUE_FULL", {"dst": int(net_pdu.dst)}, net_pdu=net_pdu)
        try:
            if net_pdu is not None and net_pdu.packet is not None:
                self.node.app.on_tx_failure(net_pdu.packet)
        except Exception:
            pass

    def on_mac_queue_depth(self, depth: int, max_depth: int) -> None:
        self._log("MAC_QUEUE_DEPTH", {"depth": int(depth), "max": int(max_depth)})

    # ------------------------------------------------------------------
    # TX path
    # ------------------------------------------------------------------

    def _send_single(self, packet: Packet) -> None:
        if self._maybe_drop_for_queue(packet):
            return

        if getattr(packet, "dscp", None) is None:
            self._log("WARN", {"reason": "dscp_not_set", "defaulting_to": 0}, packet=packet)

        meta = self._ensure_meta(packet)
        meta.setdefault("hops", 0)
        meta.setdefault("originated_by", self.node.node_id)

        net_seq = self._next_net_seq()
        next_hop = self.resolve_next_hop(int(getattr(packet, "dst", -1)))

        pdu = NetPDU(
            net_seq=net_seq,
            src=int(getattr(packet, "src", self.node.node_id)),
            dst=int(getattr(packet, "dst", -1)),
            next_hop=next_hop,
            packet=packet,
        )

        self._log("TX_DOWN", {"next_hop": next_hop}, net_pdu=pdu, packet=packet)
        self.node.mac.send_down(pdu)

    def send_down(self, packet: Packet) -> None:
        if self.enable_fragmentation:
            self._fragment_and_send(packet)
        else:
            try:
                sz = int(getattr(packet, "size_bytes", 0))
                if sz > self.max_msdu_bytes:
                    self._log(
                        "WARN",
                        {
                            "reason": "oversized_msdu_no_fragmentation",
                            "size_bytes": sz,
                            "max_msdu_bytes": self.max_msdu_bytes,
                        },
                        packet=packet,
                    )
            except Exception:
                pass
            self._send_single(packet)

    # ------------------------------------------------------------------
    # AP forwarding
    # ------------------------------------------------------------------

    def _forward_down_from_ap(self, pdu_in: NetPDU) -> None:
        meta = self._ensure_meta(pdu_in.packet)
        originated_by = int(meta.get("originated_by", -999))

        if originated_by != 0:
            hops = self._bump_hops(pdu_in)
            if hops > self.max_hops:
                self.sim.stats.packets_dropped += 1
                self._log("DROP", {"reason": "max_hops_exceeded", "hops": hops}, net_pdu=pdu_in)
                return
        else:
            hops = int(meta.get("hops", 0))

        if int(pdu_in.dst) != -1 and int(pdu_in.dst) not in getattr(self.sim, "nodes", {}):
            self.sim.stats.packets_dropped += 1
            self._log("DROP", {"reason": "dst_node_not_found", "dst": int(pdu_in.dst)}, net_pdu=pdu_in)
            return

        next_hop = self.resolve_next_hop(int(pdu_in.dst))

        pdu_out = NetPDU(
            net_seq=int(pdu_in.net_seq),
            src=int(pdu_in.src),
            dst=int(pdu_in.dst),
            next_hop=next_hop,
            packet=pdu_in.packet,
        )

        self._log("FWD", {"next_hop": next_hop, "hops": hops}, net_pdu=pdu_out)
        self.sim.stats.net_forwarded += 1
        self.node.mac.send_down(pdu_out)

    # ------------------------------------------------------------------
    # RX path
    # ------------------------------------------------------------------

    def recv_up_from_mac(self, net_pdu: NetPDU) -> None:
        self._cleanup_reassembly()
        self._log("RX_UP", {"next_hop": net_pdu.next_hop}, net_pdu=net_pdu)

        if self._is_ap():
            # AP local delivery (packets addressed to AP or broadcast)
            if int(net_pdu.dst) == 0 or int(net_pdu.dst) == -1:
                if self._is_duplicate_pdu(net_pdu):
                    self.sim.stats.net_duplicates += 1
                    self._log(
                        "DROP",
                        {"reason": "duplicate_net_seq", "net_seq": int(net_pdu.net_seq), "src": int(net_pdu.src)},
                        net_pdu=net_pdu,
                    )
                    return

                if self.enable_fragmentation and self._try_reassemble_and_deliver(net_pdu):
                    return

                self._log(
                    "DELIVER",
                    {"src": int(net_pdu.src), "dst": int(net_pdu.dst), "net_seq": int(net_pdu.net_seq)},
                    net_pdu=net_pdu,
                )
                self.node.transport.recv_up_from_net(net_pdu.packet)

                # Forward only if this is a broadcast that did NOT originate from AP
                meta = self._ensure_meta(net_pdu.packet)
                if int(net_pdu.dst) == -1 and int(meta.get("originated_by", -1)) != 0:
                    self._forward_down_from_ap(net_pdu)
                return

            # downlink forwarding path
            self._forward_down_from_ap(net_pdu)
            return

        # STA side
        if int(net_pdu.dst) == -1:
            if self._is_duplicate_pdu(net_pdu):
                self.sim.stats.net_duplicates += 1
                self._log(
                    "DROP",
                    {"reason": "duplicate_net_seq", "net_seq": int(net_pdu.net_seq), "src": int(net_pdu.src)},
                    net_pdu=net_pdu,
                )
                return

            if self.enable_fragmentation and self._try_reassemble_and_deliver(net_pdu):
                return

            self._log(
                "DELIVER",
                {"src": int(net_pdu.src), "dst": -1, "net_seq": int(net_pdu.net_seq)},
                net_pdu=net_pdu,
            )
            self.node.transport.recv_up_from_net(net_pdu.packet)
            return

        if int(net_pdu.dst) == self.node.node_id:
            if self._is_duplicate_pdu(net_pdu):
                self.sim.stats.net_duplicates += 1
                self._log(
                    "DROP",
                    {"reason": "duplicate_net_seq", "net_seq": int(net_pdu.net_seq), "src": int(net_pdu.src)},
                    net_pdu=net_pdu,
                )
                return

            if self.enable_fragmentation and self._try_reassemble_and_deliver(net_pdu):
                return

            self._log(
                "DELIVER",
                {"src": int(net_pdu.src), "dst": int(net_pdu.dst), "net_seq": int(net_pdu.net_seq)},
                net_pdu=net_pdu,
            )
            self.node.transport.recv_up_from_net(net_pdu.packet)
            return

        # Not for this STA: ignore, do not count as packet drop
        self._log("IGNORE", {"reason": "not_for_me", "dst": int(net_pdu.dst)}, net_pdu=net_pdu)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def debug_state(self) -> Dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "net_seq_ctr": self.net_seq_ctr,
            "max_hops": self.max_hops,
            "seen_sources": len(self._seen_net_seqs),
            "reassembly_buffers": len(self._reassembly),
            "fragmentation": int(self.enable_fragmentation),
            "max_msdu_bytes": self.max_msdu_bytes,
            "max_queue_depth": self.max_queue_depth,
            "reassembly_timeout_s": self.reassembly_timeout_s,
        }