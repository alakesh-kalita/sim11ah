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