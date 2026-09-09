from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Link:
    a: int
    b: int
    rate_bps: int
    prop_delay: float
    per: float


class Topology:
    def __init__(self) -> None:
        self.links: Dict[Tuple[int, int], Link] = {}

    def add_link(self, a: int, b: int, rate_bps: int, prop_delay: float, per: float) -> None:
        a = int(a)
        b = int(b)
        rate_bps = int(rate_bps)
        prop_delay = float(prop_delay)
        per = float(per)

        if a == b:
            raise ValueError(f"Self-link is not allowed: node {a}")
        if rate_bps <= 0:
            raise ValueError(f"rate_bps must be > 0, got {rate_bps}")
        if prop_delay < 0.0:
            raise ValueError(f"prop_delay must be >= 0, got {prop_delay}")
        if not (0.0 <= per <= 1.0):
            raise ValueError(f"per must be in [0, 1], got {per}")

        lk_ab = Link(a=a, b=b, rate_bps=rate_bps, prop_delay=prop_delay, per=per)
        lk_ba = Link(a=b, b=a, rate_bps=rate_bps, prop_delay=prop_delay, per=per)

        self.links[(a, b)] = lk_ab
        self.links[(b, a)] = lk_ba

    def has_link(self, src: int, dst: int) -> bool:
        return (int(src), int(dst)) in self.links

    def get_link(self, src: int, dst: int) -> Link:
        src = int(src)
        dst = int(dst)
        lk = self.links.get((src, dst))
        if lk is None:
            raise ValueError(f"No link exists from node {src} to node {dst}")
        return lk

    def neighbors(self, node_id: int) -> List[int]:
        node_id = int(node_id)
        return sorted(dst for (src, dst) in self.links.keys() if src == node_id)

    def clear(self) -> None:
        self.links.clear()


class RelayBuilder:
    """
    Build a 1-hop multi-relay BSS (IEEE 802.11ah Section 10.37).

    Node layout:
      0          : AP
      1 .. R     : Relay nodes  (R = num_relays)
      R+1 .. R+N : STAs         (N = num_stas, distributed evenly across relays)

    Links created per relay r_i and its assigned STAs:
      AP   <-> r_i    : backhaul_cfg  (higher MCS / short range)
      r_i  <-> STA_j  : access_cfg   (MCS0 / longer range)
      AP   <-> STA_j  : access_cfg   (beacon broadcast delivery on shared medium)

    The network layer enforces routing: STA_j → r_i → AP (uplink), as long
    as the STA is actually associated with its assigned relay. A STA that
    associates directly with the AP instead (in range and preferred, or
    handed over back to it -- see mac/association.py's "auto" preference
    and _maybe_handover_to_ap) routes data straight over the AP<->STA
    link like a star-topology STA would; net.py's resolve_next_hop reads
    each STA's *live* association peer, not this static assignment, so
    the AP<->STA link is not beacon-only in every case.
    """

    @staticmethod
    def build(
        sim: "Simulator",
        num_relays: int,
        num_stas: int,
        backhaul_cfg: Dict[str, Any],
        access_cfg: Dict[str, Any],
    ) -> List["Node"]:
        from sim11ah.node import Node  # local import to avoid circular

        num_relays = int(num_relays)
        num_stas   = int(num_stas)
        if num_relays < 1:
            raise ValueError(f"num_relays must be >= 1, got {num_relays}")
        if num_stas < num_relays:
            raise ValueError(
                f"num_stas ({num_stas}) must be >= num_relays ({num_relays})"
            )

        for name, cfg in (("backhaul_cfg", backhaul_cfg), ("access_cfg", access_cfg)):
            for k in ("rate_bps", "prop_delay", "per"):
                if k not in cfg:
                    raise ValueError(f"Missing {name}[{k!r}]")

        topo = Topology()
        sim.topology = topo
        sim.nodes = {}

        ap_node     = Node(node_id=0, sim=sim)
        relay_nodes = [Node(node_id=i + 1, sim=sim, role="RELAY")
                       for i in range(num_relays)]
        sta_nodes   = [Node(node_id=num_relays + 1 + i, sim=sim)
                       for i in range(num_stas)]

        all_nodes: List["Node"] = [ap_node] + relay_nodes + sta_nodes

        # Distribute STAs evenly: first (num_stas % num_relays) relays get one
        # extra STA so every relay has at least one.
        relay_assignment: Dict[int, int] = {}   # sta_node_id → relay_node_id
        base  = num_stas // num_relays
        extra = num_stas % num_relays
        sta_idx = 0
        for r_idx, relay in enumerate(relay_nodes):
            count = base + (1 if r_idx < extra else 0)
            for _ in range(count):
                relay_assignment[sta_nodes[sta_idx].node_id] = relay.node_id
                sta_idx += 1

        # Backhaul links: AP <-> each relay
        for relay in relay_nodes:
            topo.add_link(
                0, relay.node_id,
                rate_bps=int(backhaul_cfg["rate_bps"]),
                prop_delay=float(backhaul_cfg["prop_delay"]),
                per=float(backhaul_cfg["per"]),
            )

        # Access links + AP beacon-hearing links per STA
        for sta in sta_nodes:
            relay_id = relay_assignment[sta.node_id]
            topo.add_link(
                relay_id, sta.node_id,
                rate_bps=int(access_cfg["rate_bps"]),
                prop_delay=float(access_cfg["prop_delay"]),
                per=float(access_cfg["per"]),
            )
            topo.add_link(
                0, sta.node_id,
                rate_bps=int(access_cfg["rate_bps"]),
                prop_delay=float(access_cfg["prop_delay"]),
                per=float(access_cfg["per"]),
            )

        sim.nodes = {n.node_id: n for n in all_nodes}

        # Inject relay topology info into config so NetworkLayer picks it up
        topo_cfg = sim.config.setdefault("topology", {})
        topo_cfg["mode"]             = "relay"
        topo_cfg["relay_ids"]        = [r.node_id for r in relay_nodes]
        topo_cfg["relay_assignment"] = relay_assignment

        for n in all_nodes:
            n.build_layers(sim.config)

        return all_nodes


class StarBuilder:
    @staticmethod
    def build(sim: "Simulator", num_stas: int, link_cfg: Dict[str, Any]) -> List["Node"]:
        from sim11ah.node import Node  # local import to avoid circular

        num_stas = int(num_stas)
        if num_stas < 0:
            raise ValueError(f"num_stas must be >= 0, got {num_stas}")

        required = ("rate_bps", "prop_delay", "per")
        for k in required:
            if k not in link_cfg:
                raise ValueError(f"Missing link_cfg[{k!r}]")

        rate_bps = int(link_cfg["rate_bps"])
        prop_delay = float(link_cfg["prop_delay"])
        per = float(link_cfg["per"])

        topo = Topology()
        sim.topology = topo
        sim.nodes = {}

        ap_id = 0
        nodes: List[Node] = [Node(node_id=ap_id, sim=sim)]

        for i in range(1, num_stas + 1):
            nodes.append(Node(node_id=i, sim=sim))

        for i in range(1, num_stas + 1):
            topo.add_link(
                ap_id,
                i,
                rate_bps=rate_bps,
                prop_delay=prop_delay,
                per=per,
            )

        sim.nodes = {n.node_id: n for n in nodes}

        for n in nodes:
            n.build_layers(sim.config)

        return nodes