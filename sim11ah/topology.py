from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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
    and _maybe_roam/_roam_to) routes data straight over the AP<->STA
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


class MultiApBuilder:
    """
    Build a multi-AP topology: K independent AP nodes in a 1D chain, N STA
    nodes scattered along that chain. Kept separate from StarBuilder (not
    a generalization of it) -- every existing benchmark result in this
    project depends on StarBuilder's exact behavior, and num_aps==1 there
    already covers the single-AP case; this class only ever runs when a
    caller explicitly asks for num_aps > 1.

    Node layout:
      0 .. K-1       : AP nodes, ap[i].pos = (i * ap_spacing_m, 0)
      K .. K+N-1     : STA nodes (default: scattered evenly along the AP
                       chain's span, or pass sta_positions to place them
                       yourself -- e.g. sim11ah/mobility.py's corridor
                       stepper drives them explicitly after this build)

    Every (AP, STA) pair gets a link, not just each STA's nearest AP.
    PhyLayer._schedule_one_rx refuses to even schedule a reception without
    an explicit Topology.Link for that (tx, rx) pair, regardless of real
    RSSI/distance -- a STA only linked to its "home" AP would silently
    never receive a second, in-range AP's beacon at all (no crash, just
    permanently invisible), which is easy to mistake for a mistuned RSSI
    threshold. add_link only gates *reachability* here (per=0.0); the real
    distance/SNR-based PER model in phy.py still governs actual link
    quality once a frame is scheduled.
    """

    @staticmethod
    def build(
        sim: "Simulator",
        num_aps: int,
        ap_spacing_m: float,
        num_stas: int,
        link_cfg: Dict[str, Any],
        sta_positions: List[Tuple[float, float]] = None,
    ) -> List["Node"]:
        from sim11ah.node import Node  # local import to avoid circular

        num_aps = int(num_aps)
        num_stas = int(num_stas)
        if num_aps < 1:
            raise ValueError(f"num_aps must be >= 1, got {num_aps}")
        if num_stas < 0:
            raise ValueError(f"num_stas must be >= 0, got {num_stas}")

        required = ("rate_bps", "prop_delay", "per")
        for k in required:
            if k not in link_cfg:
                raise ValueError(f"Missing link_cfg[{k!r}]")

        rate_bps = int(link_cfg["rate_bps"])
        prop_delay = float(link_cfg["prop_delay"])
        per = float(link_cfg["per"])
        ap_spacing_m = float(ap_spacing_m)

        topo = Topology()
        sim.topology = topo
        sim.nodes = {}

        ap_nodes = [
            Node(node_id=i, sim=sim, role="AP", pos=(i * ap_spacing_m, 0.0))
            for i in range(num_aps)
        ]

        sta_nodes: List["Node"] = []
        span = max(1.0, (num_aps - 1) * ap_spacing_m)
        for j in range(num_stas):
            sid = num_aps + j
            if sta_positions is not None and j < len(sta_positions):
                pos = sta_positions[j]
            else:
                # Default static scatter (evenly along the AP chain's span)
                # for callers that don't need to drive positions themselves
                # -- sim11ah/mobility.py's corridor stepper overrides this
                # per-STA after the build for the actual roaming experiment.
                frac = (j + 0.5) / max(1, num_stas)
                pos = (frac * span, 0.0)
            sta_nodes.append(Node(node_id=sid, sim=sim, pos=pos))

        all_nodes: List["Node"] = ap_nodes + sta_nodes

        for ap in ap_nodes:
            for sta in sta_nodes:
                topo.add_link(ap.node_id, sta.node_id, rate_bps=rate_bps,
                              prop_delay=prop_delay, per=per)

        # AP<->AP links too -- each AP's own beacon is a broadcast frame
        # (dst=-1) the PHY layer tries to schedule reception of at every
        # other node for CCA/collision-sensing purposes, same as it does
        # for STAs. Without a link here, one AP silently can never receive
        # another's beacon at all (RX_LINK_MISSING for the AP-to-AP pair
        # specifically -- caught empirically the first time this builder
        # was actually run and exercised end to end).
        for i, ap_a in enumerate(ap_nodes):
            for ap_b in ap_nodes[i + 1:]:
                topo.add_link(ap_a.node_id, ap_b.node_id, rate_bps=rate_bps,
                              prop_delay=prop_delay, per=per)

        sim.nodes = {n.node_id: n for n in all_nodes}

        # ap_ids mirrors relay_ids' existing convention (a plain list in
        # sim.config["topology"]) -- this is what GUI/analysis code should
        # key off to find "every AP" rather than assuming node 0.
        topo_cfg = sim.config.setdefault("topology", {})
        topo_cfg["mode"] = "multi_ap"
        topo_cfg["ap_ids"] = [ap.node_id for ap in ap_nodes]
        topo_cfg["ap_spacing_m"] = ap_spacing_m

        for n in all_nodes:
            n.build_layers(sim.config)

        return all_nodes

    @staticmethod
    def overlap_width_m(sim: "Simulator", ap_spacing_m: float) -> float:
        """Derived reporting value: how wide the region is where two
        adjacent APs' nominal coverage circles both reach, for the given
        spacing. Requires build() to have already run (needs a built PHY
        to read EIRP/sensitivity/path-loss from) -- pass the same
        ap_spacing_m used there. Negative means the APs don't overlap at
        all at this spacing."""
        ap0 = sim.nodes.get(0)
        if ap0 is None or ap0.phy is None:
            return 0.0
        return 2.0 * float(ap0.phy.nominal_range_m()) - float(ap_spacing_m)


class CarsUavsBuilder:
    """
    A multi-AP corridor populated with three kinds of mobile STA: cars and
    scooters driving a straight highway through the AP chain (same
    mobility function, different lane/speed), and UAVs flying
    random-waypoint across the same span -- a layout for visualizing
    inter-AP handover with vehicle-mounted and airborne devices instead of
    static ground STAs. Built ON TOP of MultiApBuilder (same AP layout +
    all-pairs linking), not a fork of it -- this class only adds role
    tagging and initial positions for the three mobility kinds; actually
    driving them each tick is sim11ah/mobility.py's job
    (highway_bounce_step for cars/scooters, uav_waypoint_step for UAVs),
    called from wherever runs the simulation (a GUI tick loop, or a
    headless script).

    Node layout:
      0 .. K-1                    : AP nodes (MultiApBuilder's layout)
      K .. K+num_cars-1           : car STAs
      K+num_cars .. +num_scooters : scooter STAs
      ... .. +num_uavs            : UAV STAs

    sim.config["topology"]["car_ids"] / "scooter_ids" / "uav_ids" record
    which STA ids are which, mirroring the existing ap_ids/relay_ids
    convention -- callers (GUI icon/mobility drivers, headless scripts)
    key off these rather than guessing from node_id ranges.

    KNOWN LIMITATION -- build with raw_enable=False for this layout. With
    RAW enabled, a STA whose reactive roam trigger fires repeatedly while
    flying/driving through a dense multi-AP overlap band (several APs'
    coverage circles all reaching it at once) can get permanently stuck:
    AUTH_REQ goes out, never gets ACKed, STUCK_RETARGET cycles to a
    different AP, repeat forever -- confirmed via direct event trace (the
    STA's own AUTH_REQ transmissions simply stop completing once this
    starts, for the rest of the run) with 3 APs at 900m spacing. Root
    cause not diagnosed (something in the RAW slot/schedule interaction
    for an unassociated STA mid-handshake near multiple APs' independent,
    staggered RAW schedules, not a mobility or topology bug -- confirmed
    by re-running the identical scenario with raw_enable=False, which
    associates and hands over cleanly). Fixing that is a separate,
    deeper investigation; this layout exists to visualize handover
    behavior, not stress-test RAW scheduling under it, so DCF-only is the
    right default until that's understood.
    """

    @staticmethod
    def build(
        sim: "Simulator",
        num_aps: int,
        ap_spacing_m: float,
        num_cars: int,
        num_uavs: int,
        link_cfg: Dict[str, Any],
        num_scooters: int = 0,
        car_lane_offset_m: float = 25.0,
        scooter_lane_offset_m: float = 12.0,
        uav_margin_m: float = 150.0,
    ) -> List["Node"]:
        num_aps = max(1, int(num_aps))
        num_cars = int(num_cars)
        num_uavs = int(num_uavs)
        num_scooters = int(num_scooters)
        if num_cars < 0 or num_uavs < 0 or num_scooters < 0:
            raise ValueError(
                f"num_cars ({num_cars}), num_scooters ({num_scooters}) and "
                f"num_uavs ({num_uavs}) must all be >= 0"
            )

        span = max(1.0, (num_aps - 1) * float(ap_spacing_m))

        # Cars and scooters both start on the highway itself (a fixed-
        # offset lane parallel to the AP axis, alternating sides so
        # opposite-direction traffic doesn't visually overlap -- purely
        # cosmetic, doesn't affect RSSI/PHY, which only cares about the
        # resulting (x, y)). Scooters get a SMALLER offset than cars --
        # riding closer to the centreline -- so the two vehicle kinds
        # occupy visually distinct lanes rather than overlapping.
        # highway_bounce_step then drives both back and forth for real.
        def _lane_positions(count: int, offset_m: float) -> List[Tuple[float, float]]:
            out = []
            for j in range(count):
                frac = (j + 0.5) / max(1, count)
                lane = offset_m if j % 2 == 0 else -offset_m
                out.append((frac * span, lane))
            return out

        car_positions = _lane_positions(num_cars, car_lane_offset_m)
        scooter_positions = _lane_positions(num_scooters, scooter_lane_offset_m)

        # UAVs start scattered near the corridor; uav_waypoint_step then
        # flies them on a random-waypoint pattern across the whole
        # AP-spanning region (see uav_region below), not anchored to a
        # single AP.
        uav_positions = [
            ((j + 0.5) / max(1, num_uavs) * span, 0.0)
            for j in range(num_uavs)
        ]

        sta_positions = car_positions + scooter_positions + uav_positions
        num_stas = num_cars + num_scooters + num_uavs

        all_nodes = MultiApBuilder.build(
            sim, num_aps=num_aps, ap_spacing_m=ap_spacing_m,
            num_stas=num_stas, link_cfg=link_cfg,
            sta_positions=sta_positions if sta_positions else None,
        )

        car_ids = [num_aps + j for j in range(num_cars)]
        scooter_ids = [num_aps + num_cars + j for j in range(num_scooters)]
        uav_ids = [num_aps + num_cars + num_scooters + j for j in range(num_uavs)]

        topo_cfg = sim.config.setdefault("topology", {})
        topo_cfg["mode"] = "cars_uavs"
        topo_cfg["car_ids"] = car_ids
        topo_cfg["scooter_ids"] = scooter_ids
        topo_cfg["uav_ids"] = uav_ids
        topo_cfg["corridor_span_m"] = span
        # Recorded so every consumer that needs to know how far vehicles
        # actually roam (uav_region's default margin, and the 3D Smart
        # City's procedural road/building sizing in
        # ui/web3d/snapshot.py's _stable_bounds) reads the SAME numbers
        # this build actually used, instead of each guessing its own
        # default and silently drifting apart -- that drift is exactly
        # what made the city render far narrower than the corridor
        # cars/UAVs actually drive/fly across.
        topo_cfg["car_lane_offset_m"] = float(car_lane_offset_m)
        topo_cfg["scooter_lane_offset_m"] = float(scooter_lane_offset_m)
        topo_cfg["uav_margin_m"] = float(uav_margin_m)

        return all_nodes

    @staticmethod
    def uav_region(
        sim: "Simulator", ap_spacing_m: float, num_aps: int, margin_m: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) spanning the whole AP corridor plus
        a margin -- the region to pass to mobility.uav_waypoint_step so
        UAVs wander across every AP's coverage instead of drifting off
        past the last one or clumping in the middle.

        margin_m defaults to whatever build() actually used for this sim
        (sim.config["topology"]["uav_margin_m"], falling back to 150.0 if
        the sim wasn't built via build() at all) rather than its own
        separately-hardcoded 150.0 -- the two used to be independent
        defaults that happened to agree only by coincidence, and
        ui/web3d/snapshot.py's _stable_bounds needs this exact same
        number to size the 3D Smart City's roads/buildings to actually
        contain the region UAVs fly in."""
        if margin_m is None:
            topo_cfg = getattr(sim, "config", {}).get("topology", {}) if sim is not None else {}
            margin_m = topo_cfg.get("uav_margin_m", 150.0)
        span = max(1.0, (max(1, int(num_aps)) - 1) * float(ap_spacing_m))
        return (-float(margin_m), -float(margin_m), span + float(margin_m), float(margin_m))


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