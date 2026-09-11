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
        ap_positions: List[Tuple[float, float]] = None,
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

        # ap_positions, when given (CarsUavsBuilder's 2D AP grid uses this
        # to place multiple AP rows), overrides the default single-line
        # (i * ap_spacing_m, 0) placement entirely -- every other existing
        # caller (roam_experiment.py, main_gui.py's "multi_ap" topology)
        # never passes it, so their AP chain is completely unaffected.
        ap_nodes = [
            Node(node_id=i, sim=sim, role="AP",
                 pos=(ap_positions[i] if ap_positions is not None and i < len(ap_positions)
                      else (i * ap_spacing_m, 0.0)))
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
    (highway_loop_step for cars/scooters, uav_waypoint_step for UAVs),
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
        # 15.0, not a value closer to 0 -- ui/web3d/snapshot.py's
        # _road_loops draws the 3D highway's paved band straddling
        # +/-car_lane_offset_m with an ~11m-wide unpaved median either
        # side of the centreline (ROAD_HALF_W+SIDEWALK_W=14 short of
        # car_lane_offset_m=25); 15.0 clears that median by a comfortable
        # 4m rather than sitting right on its edge, so scooters render on
        # pavement, not straddling the kerb.
        scooter_lane_offset_m: float = 15.0,
        # 4 avenues (not just the one original highway) -- "why do cars
        # only ever use two roads" was a fair complaint: everything used
        # to ride the same +/-car_lane_offset_m pair no matter how many
        # cars there were. Extra avenues at increasing distance from the
        # centreline give real road variety across a much bigger city
        # footprint instead of packing every vehicle onto one strip.
        num_car_avenues: int = 4,
        avenue_spacing_m: float = 220.0,
        # 1100.0, not 220.0 -- 5x the previous city footprint (per
        # explicit request), and this single number still drives
        # everything that needs to know how far the built-up
        # area/UAV envelope reaches: the 3D city-limits ring
        # (ui/web3d/snapshot.py's _road_loops outer loop), the 2D
        # highway background's margin/depth scale, and uav_region's
        # default UAV roam envelope.
        uav_margin_m: float = 1100.0,
        # 2 rows, not 1 -- a single AP chain sitting exactly on the
        # centreline can't reach vehicles out on the wider avenues (up to
        # +/-car_lane_offset_m + (num_car_avenues-1)*avenue_spacing_m off
        # that line) without relying on an unrealistically long PHY
        # range. Two rows straddling the centreline at +/-ap_row_offset_m
        # put every AP's own nominal range circle across most of the
        # avenue band instead of just the innermost one. num_aps/
        # ap_spacing_m keep meaning "columns"/"column spacing" exactly as
        # before -- this only adds rows, so corridor_span_m and every
        # x-axis formula (car/UAV x-range, avenue length, ...) are
        # unaffected by it.
        num_ap_rows: int = 2,
        ap_row_offset_m: float = 350.0,
        # Matches ui/web3d/snapshot.py's own _cross_streets spacing
        # (they used to each hardcode 220.0 independently -- see the
        # cross_street_xs comment below for why that stopped once real
        # vehicles needed to agree with drawn geometry, not just two
        # views agreeing with each other).
        cross_street_spacing_m: float = 220.0,
        cross_lane_offset_m: float = 10.0,
        scooter_cross_lane_offset_m: float = 7.0,
    ) -> List["Node"]:
        num_aps = max(1, int(num_aps))
        num_cars = int(num_cars)
        num_uavs = int(num_uavs)
        num_scooters = int(num_scooters)
        num_car_avenues = max(1, int(num_car_avenues))
        num_ap_rows = max(1, int(num_ap_rows))
        if num_cars < 0 or num_uavs < 0 or num_scooters < 0:
            raise ValueError(
                f"num_cars ({num_cars}), num_scooters ({num_scooters}) and "
                f"num_uavs ({num_uavs}) must all be >= 0"
            )

        span = max(1.0, (num_aps - 1) * float(ap_spacing_m))

        # AP grid: num_aps columns (unchanged meaning/spacing) x
        # num_ap_rows rows, rows evenly spaced and centred on the
        # highway's own centreline (row 0 and row -1 land at exactly
        # +/-ap_row_offset_m for the common num_ap_rows=2 case). Node ids
        # run row-major (row 0's columns first, then row 1's, ...) purely
        # as an ordering convention -- nothing downstream depends on
        # which AP id is which grid cell, only on topo_cfg["ap_ids"]
        # listing all of them.
        total_aps = num_aps * num_ap_rows
        if num_ap_rows > 1:
            row_ys = [
                (r - (num_ap_rows - 1) / 2.0) * (2.0 * ap_row_offset_m / (num_ap_rows - 1))
                for r in range(num_ap_rows)
            ]
        else:
            row_ys = [0.0]
        ap_positions = [
            (col * float(ap_spacing_m), row_ys[row])
            for row in range(num_ap_rows)
            for col in range(num_aps)
        ]

        # car_avenue_offsets_m: the distance of each avenue from the
        # centreline (+/- each value gives the actual lane y once mirrored
        # below) -- avenue 0 is the original highway at car_lane_offset_m,
        # each further one avenue_spacing_m farther out. Scooters ride the
        # same avenues, just inset toward the centreline by the same
        # car_lane_offset_m - scooter_lane_offset_m gap the original
        # single-avenue design used, so they stay inside each avenue's own
        # paved band (see ui/web3d/snapshot.py's _road_loops) rather than
        # needing separate roads of their own.
        car_avenue_offsets = [
            car_lane_offset_m + i * avenue_spacing_m for i in range(num_car_avenues)
        ]
        scooter_inset = max(0.0, car_lane_offset_m - scooter_lane_offset_m)
        scooter_avenue_offsets = [off - scooter_inset for off in car_avenue_offsets]

        # Cross streets: perpendicular to the avenues, at regular x
        # intervals, each spanning every avenue's y-extent -- the single
        # source of truth for their layout (ui/web3d/snapshot.py's
        # _cross_streets and ui/topology_canvas.py's _bg_city_highway
        # both read cross_street_xs/cross_street_y_reach back out of
        # topo_cfg below instead of recomputing this same formula
        # independently a second and third time, which is exactly the
        # kind of drift that let two separate copies of the "distance to
        # AP" computation quietly disagree earlier -- not repeating that
        # here now that REAL vehicles, not just drawn geometry, depend
        # on this matching exactly). 21.0+10.0 mirrors world.js's
        # ROAD_HALF_W+SIDEWALK_W (a cross street's own paved half-width)
        # plus a clearance margin -- a cross street reaches a little past
        # the outermost avenue's real kerb, not out to the city limit;
        # it exists to connect the avenues to each other, not to wander
        # into the sparse outskirts alongside them.
        outer_avenue = car_avenue_offsets[-1] if car_avenue_offsets else car_lane_offset_m
        cross_street_y_reach = outer_avenue + 21.0 + 10.0
        n_cross_streets = max(1, int(span / cross_street_spacing_m))
        cross_street_xs = [
            (i + 0.5) * (span / n_cross_streets) for i in range(n_cross_streets)
        ]

        # Cars/scooters mostly start on an avenue (round-robin across
        # every avenue x both sides, so vehicles actually spread across
        # the whole road network instead of clustering on avenue 0),
        # spaced along x the same way as before -- but a quarter of each
        # start on a cross street instead, driving north-south rather
        # than east-west, so the fleet actually moves in every
        # direction the road network offers instead of only ever along
        # the highway. Purely cosmetic either way -- doesn't affect
        # RSSI/PHY, which only cares about the resulting (x, y).
        # highway_loop_step/cross_street_loop_step then drive each one
        # continuously forward for real, forever staying on whichever
        # lane it started on (that lane's own sign is what both
        # functions read to decide which way "forward" is).
        def _lane_positions(count: int, offsets: List[float]) -> List[Tuple[float, float]]:
            lanes = []
            for off in offsets:
                lanes.append(off)
                lanes.append(-off)
            out = []
            for j in range(count):
                frac = (j + 0.5) / max(1, count)
                lane = lanes[j % len(lanes)]
                out.append((frac * span, lane))
            return out

        def _cross_lane_positions(count: int, lane_offset_m: float) -> List[Tuple[float, float]]:
            lanes = (lane_offset_m, -lane_offset_m)
            out = []
            for j in range(count):
                cx = cross_street_xs[j % len(cross_street_xs)]
                frac = (j + 0.5) / max(1, count)
                lane = lanes[j % len(lanes)]
                y = frac * (2.0 * cross_street_y_reach) - cross_street_y_reach
                out.append((cx + lane, y))
            return out

        n_car_cross = num_cars // 4
        n_car_ave = num_cars - n_car_cross
        car_positions = (_lane_positions(n_car_ave, car_avenue_offsets)
                         + _cross_lane_positions(n_car_cross, cross_lane_offset_m))

        n_scooter_cross = num_scooters // 4
        n_scooter_ave = num_scooters - n_scooter_cross
        scooter_positions = (_lane_positions(n_scooter_ave, scooter_avenue_offsets)
                             + _cross_lane_positions(n_scooter_cross, scooter_cross_lane_offset_m))

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
            sim, num_aps=total_aps, ap_spacing_m=ap_spacing_m,
            num_stas=num_stas, link_cfg=link_cfg,
            sta_positions=sta_positions if sta_positions else None,
            ap_positions=ap_positions,
        )

        car_ids = [total_aps + j for j in range(num_cars)]
        scooter_ids = [total_aps + num_cars + j for j in range(num_scooters)]
        uav_ids = [total_aps + num_cars + num_scooters + j for j in range(num_uavs)]
        # car_positions/scooter_positions each put their avenue vehicles
        # first, cross-street ones last (see the concatenation above) --
        # car_ids/scooter_ids are assigned in that exact same order, so
        # the LAST n_car_cross/n_scooter_cross ids are the cross-street
        # ones. Recorded explicitly (not re-derived from position) so
        # the mobility driver (ui/dashboard_tk.py) and reseed
        # (ui/topology_canvas.py's _seed_default_layout) know which
        # function/heading rule applies to which vehicle without having
        # to guess from where it currently happens to be.
        car_cross_ids = car_ids[n_car_ave:]
        scooter_cross_ids = scooter_ids[n_scooter_ave:]

        topo_cfg = sim.config.setdefault("topology", {})
        topo_cfg["mode"] = "cars_uavs"
        topo_cfg["car_ids"] = car_ids
        topo_cfg["scooter_ids"] = scooter_ids
        topo_cfg["uav_ids"] = uav_ids
        topo_cfg["car_cross_ids"] = car_cross_ids
        topo_cfg["scooter_cross_ids"] = scooter_cross_ids
        topo_cfg["cross_street_xs"] = [float(x) for x in cross_street_xs]
        topo_cfg["cross_street_y_reach"] = float(cross_street_y_reach)
        topo_cfg["cross_lane_offset_m"] = float(cross_lane_offset_m)
        topo_cfg["scooter_cross_lane_offset_m"] = float(scooter_cross_lane_offset_m)
        topo_cfg["corridor_span_m"] = span
        # num_ap_rows/ap_row_offset_m recorded alongside ap_ids/
        # ap_spacing_m so a reseed (ui/topology_canvas.py's
        # _seed_default_layout) can reconstruct this exact 2D AP grid --
        # ap_spacing_m alone only describes one row's column spacing, not
        # the row layout on top of it.
        topo_cfg["num_ap_rows"] = num_ap_rows
        topo_cfg["ap_row_offset_m"] = float(ap_row_offset_m)
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
        topo_cfg["car_avenue_offsets_m"] = [float(o) for o in car_avenue_offsets]
        topo_cfg["scooter_avenue_offsets_m"] = [float(o) for o in scooter_avenue_offsets]
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