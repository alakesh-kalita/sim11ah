from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, List, Tuple

from sim11ah.models import Packet

# P-frame-heavy / I-frame-light size mix for the "video" traffic model
# (see ApplicationLayer._build_traffic_model) -- (size_bytes, weight)
# pairs consumed by _pick_size's size_mode="table" weighted choice.
# ~10% I-frames approximates a GOP length of ~10 at the model's default
# 5 fps (a new I-frame roughly every 2s), without needing a strict
# repeating sequence the probabilistic table mechanism doesn't support.
DEFAULT_VIDEO_SIZE_TABLE = [(1200, 0.90), (6000, 0.10)]


class TrafficModel:
    def next_interval(self, rng: random.Random) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        return

    def set_now(self, now: float) -> None:
        return


class PeriodicTraffic(TrafficModel):
    def __init__(self, interval: float, jitter_s: float = 0.0):
        self.interval = float(interval)
        self.jitter_s = max(0.0, float(jitter_s))

    def next_interval(self, rng: random.Random) -> float:
        if self.jitter_s <= 0.0:
            return self.interval
        j = rng.uniform(-self.jitter_s, self.jitter_s)
        return max(0.0, self.interval + j)


class CBRTraffic(TrafficModel):
    def __init__(self, rate_bps: float, packet_size_bytes: int):
        self.rate_bps = float(rate_bps)
        self.packet_size_bytes = int(packet_size_bytes)
        self._recompute()

    def _recompute(self) -> None:
        self.interval = (self.packet_size_bytes * 8.0) / max(1.0, self.rate_bps)

    def set_packet_size(self, packet_size_bytes: int) -> None:
        self.packet_size_bytes = int(packet_size_bytes)
        self._recompute()

    def next_interval(self, rng: random.Random) -> float:
        return float(self.interval)


class PoissonTraffic(TrafficModel):
    def __init__(self, rate_lambda: float):
        self.rate_lambda = float(rate_lambda)

    def next_interval(self, rng: random.Random) -> float:
        u = rng.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return -math.log(1.0 - u) / max(1e-12, self.rate_lambda)


class BurstyTraffic(TrafficModel):
    def __init__(self, burst_size: int, intra_gap: float, off_time: float):
        self.burst_size = max(1, int(burst_size))
        self.intra_gap = float(intra_gap)
        self.off_time = float(off_time)
        self._remaining = self.burst_size

    def reset(self) -> None:
        self._remaining = self.burst_size

    def next_interval(self, rng: random.Random) -> float:
        if self._remaining > 1:
            self._remaining -= 1
            return max(0.0, self.intra_gap)
        self._remaining = self.burst_size
        return max(0.0, self.off_time)


class OnOffTraffic(TrafficModel):
    def __init__(self, lambda_on: float, on_time: float, off_time: float):
        self.lambda_on = float(lambda_on)
        self.on_time = float(on_time)
        self.off_time = float(off_time)
        self._state = "off"
        self._next_switch = 0.0
        self._now = 0.0

    def reset(self) -> None:
        self._state = "off"
        self._next_switch = 0.0
        self._now = 0.0

    def set_now(self, now: float) -> None:
        self._now = float(now)
        if self._next_switch <= 0.0:
            self._next_switch = self._now + self.off_time

        while self._now >= self._next_switch:
            if self._state == "on":
                self._state = "off"
                self._next_switch += self.off_time
            else:
                self._state = "on"
                self._next_switch += self.on_time

    def next_interval(self, rng: random.Random) -> float:
        if self._state != "on":
            return max(0.0, self._next_switch - self._now)

        u = rng.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return -math.log(1.0 - u) / max(1e-12, self.lambda_on)


class ApplicationLayer:
    _TRAFFIC_TYPE_DSCP: Dict[str, int] = {
        "sensor": 0,
        "control": 48,
        "alarm": 48,
        "telemetry": 8,
        "video": 32,
    }

    def __init__(self, node: "Node", cfg: Dict[str, Any]):
        self.node = node
        self.sim = node.sim
        self.cfg = cfg

        app_cfg = cfg.get("app", {})
        mac_cfg = cfg.get("mac", {})

        self.dst_mode = str(app_cfg.get("dst_mode", "random_sta")).lower()
        self.enable_sink = bool(app_cfg.get("enable_sink", True))

        self._running = False

        self.start_spread_s = float(
            app_cfg.get("start_spread_s", float(mac_cfg.get("beacon_interval", 0.5)))
        )
        self.start_phase_mode = str(app_cfg.get("start_phase_mode", "deterministic")).lower()
        self.start_phase_jitter_s = float(app_cfg.get("start_phase_jitter_s", 0.0))

        self.size_mode = str(app_cfg.get("size_mode", "fixed"))
        self.packet_size = int(app_cfg.get("packet_size_bytes", 128))
        self.size_min = int(app_cfg.get("size_min_bytes", 64))
        self.size_max = int(app_cfg.get("size_max_bytes", 256))
        self.pareto_alpha = float(app_cfg.get("pareto_alpha", 1.5))
        self.pareto_scale = int(app_cfg.get("pareto_scale_bytes", self.packet_size))
        self.size_table = app_cfg.get("size_table", None)

        self.traffic_type = str(app_cfg.get("traffic_type", "sensor"))
        self.dscp = app_cfg.get("dscp", None)

        # After size_mode/size_table/traffic_type so the "video" branch's
        # side-effect overrides (see _build_traffic_model) are the final
        # word for that traffic type, not silently clobbered by the plain
        # app_cfg.get() defaults above -- every other traffic type is
        # unaffected since only "video" mutates these.
        self._traffic: Optional[TrafficModel] = self._build_traffic_model(app_cfg)

        self._in_flight: Dict[Tuple[int, int], float] = {}
        self.max_in_flight = int(app_cfg.get("max_in_flight", 0))
        self.congestion_backoff_s = float(app_cfg.get("congestion_backoff_s", 0.05))
        self.in_flight_timeout_s = float(app_cfg.get("in_flight_timeout_s", 10.0))

        self._extra_generate_pending: bool = False

        self._dst_cache: Optional[List[int]] = None
        self._dst_cache_time: float = -1.0
        self._dst_cache_ttl: float = float(app_cfg.get("dst_cache_ttl_s", 5.0))

        self._delays_local: List[float] = []
        self._delivered_from_src: Dict[int, int] = {}
        self._rx_packets: int = 0

        self.expected_interval_s = float(app_cfg.get("expected_interval_s", 0.0))
        self._last_delivery_time: Dict[int, float] = {}
        self._jitter_samples: List[float] = []

        self.goodput_window_s = float(app_cfg.get("goodput_window_s", 1.0))
        self._bytes_received: int = 0
        self._goodput_window_start: Optional[float] = None

    def _record_priority_generated(self, aid: int) -> None:
        try:
            if hasattr(self.sim, "priority_metrics"):
                self.sim.priority_metrics.record_generated(aid=int(aid))
        except Exception:
            pass

    def _record_priority_delivered(self, packet: Packet, delay: float) -> None:
        try:
            if hasattr(self.sim, "priority_metrics"):
                src_aid = getattr(packet, "src", None)
                if src_aid is None:
                    src_aid = getattr(packet, "src_id", None)
                if src_aid is None:
                    src_aid = getattr(packet, "src_aid", None)
                if src_aid is None:
                    src_aid = getattr(packet, "source", None)

                if src_aid is not None:
                    self.sim.priority_metrics.record_delivered(
                        aid=int(src_aid),
                        delay=float(delay),
                    )
        except Exception:
            pass

    def _build_traffic_model(self, app_cfg: Dict[str, Any]) -> Optional[TrafficModel]:
        traffic = str(app_cfg.get("traffic", "periodic")).lower()

        if traffic == "periodic":
            return PeriodicTraffic(
                float(app_cfg.get("periodic_interval", 0.2)),
                float(app_cfg.get("periodic_jitter_s", 0.0)),
            )
        if traffic == "poisson":
            return PoissonTraffic(float(app_cfg.get("poisson_lambda", 5.0)))
        if traffic == "cbr":
            return CBRTraffic(
                float(app_cfg.get("cbr_rate_bps", 20000.0)),
                int(app_cfg.get("packet_size_bytes", 128)),
            )
        if traffic in ("burst", "bursty"):
            return BurstyTraffic(
                int(app_cfg.get("burst_size", 10)),
                float(app_cfg.get("burst_intra_gap_s", 0.001)),
                float(app_cfg.get("burst_off_time_s", 1.0)),
            )
        if traffic == "onoff":
            return OnOffTraffic(
                float(app_cfg.get("onoff_lambda_on", 20.0)),
                float(app_cfg.get("onoff_on_time_s", 2.0)),
                float(app_cfg.get("onoff_off_time_s", 2.0)),
            )
        if traffic == "video":
            # Low-power HaLow camera/video sensor: frames arrive roughly
            # periodically at a target frame rate, with a GOP-like size mix
            # -- most frames are small P-frames, a minority are much larger
            # I-frames. This is a probabilistic mix (size_mode="table"
            # weighted choice, see _pick_size), not a strict repeating I/P/
            # P/P sequence -- close enough to model the RAW-scheduling-
            # relevant property real video traffic has that periodic/CBR
            # traffic doesn't: highly variable per-packet size at a
            # otherwise-regular arrival cadence, which stresses whichever
            # RAW policy is under test differently than uniform small
            # packets do. Defaults sized for what HaLow's PHY can actually
            # sustain (MCS0-3 = 150-600 kb/s, see phy_mode_table in
            # config.py) -- 5 fps with a 1.2 KB/6 KB P/I mix averages
            # ~67 kb/s, a realistic low-frame-rate monitoring camera, not
            # an unrealistic full-motion-video target no HaLow link could
            # sustain. I-frames (6 KB) exceed rts_threshold (2346 B by
            # default), so they exercise the existing RTS/CTS path in
            # dcf.py. NOTE: there is no MAC-layer fragmentation to exercise
            # here -- dcf.py's TX_FRAG/_pending_fragments machinery exists
            # but nothing in this codebase ever populates
            # ctx._pending_fragments with real fragments (checked: it is
            # only ever assigned []), so it never actually fires. An
            # oversized frame is instead transmitted whole as a single,
            # proportionally longer-duration frame
            # (phy.py:compute_tx_duration scales linearly with
            # frame.size_bytes, no size cap) -- a reasonable simplification
            # for MAC-scheduling research, just not literal 802.11
            # fragmentation, so don't describe it as such elsewhere.
            fps = max(0.1, float(app_cfg.get("video_fps", 5.0)))
            self.size_mode = "table"
            # `or`, not .get(..., default) -- default_config() sets
            # video_size_table=None explicitly (present, not missing), so
            # .get()'s fallback would never trigger; `or` catches None (and
            # an empty list/table) the same way absence would.
            self.size_table = app_cfg.get("video_size_table") or DEFAULT_VIDEO_SIZE_TABLE
            self.traffic_type = "video"
            return PeriodicTraffic(
                1.0 / fps,
                float(app_cfg.get("video_jitter_s", 0.0)),
            )

        return PeriodicTraffic(
            float(app_cfg.get("periodic_interval", 0.2)),
            float(app_cfg.get("periodic_jitter_s", 0.0)),
        )

    def _log(self, event: str, details: Dict[str, Any], packet: Optional[Packet] = None) -> None:
        self.sim.log(
            node_id=self.node.node_id,
            layer="APP",
            event=event,
            details=details,
            packet=packet,
        )

    def set_traffic_model(self, tm: Optional[TrafficModel]) -> None:
        self._traffic = tm

    def _schedule_generate(self, delay_s: float) -> None:
        delay_s = max(0.0, float(delay_s))
        self._log("GEN_NEXT", {"dt": round(delay_s, 6)})
        self.sim.engine.schedule_in(delay_s, self._generate_one, name="APP_GENERATE")

    def _compute_start_offset(self) -> float:
        spread = max(0.0, float(self.start_spread_s))
        if spread <= 0.0:
            return 0.0

        sta_ids = sorted(i for i in self.sim.nodes.keys() if i != 0)
        n_stas = max(1, len(sta_ids))

        if self.start_phase_mode == "random":
            base = self.sim.engine.rng.random() * spread
        else:
            try:
                idx = sta_ids.index(self.node.node_id)
            except ValueError:
                idx = max(0, self.node.node_id - 1)
            base = idx * (spread / n_stas)

        jitter_cap = min(
            max(0.0, self.start_phase_jitter_s),
            spread / max(1, n_stas),
        )
        jitter = self.sim.engine.rng.random() * jitter_cap if jitter_cap > 0.0 else 0.0

        return min(spread, base + jitter)

    def start(self) -> None:
        if self._traffic is not None and hasattr(self._traffic, "reset"):
            try:
                self._traffic.reset()
            except Exception:
                pass

        if self.node.is_ap:
            self._running = True
            self._log("START", {"role": "ap_sink_only"})
            return

        if self._traffic is None:
            self._log("START_SKIP", {"reason": "no_traffic_model"})
            return

        self._running = True
        self._extra_generate_pending = False

        start_offset = self._compute_start_offset()
        self._log(
            "START",
            {
                "start_offset_s": round(start_offset, 6),
                "spread_s": round(float(self.start_spread_s), 6),
                "phase_mode": self.start_phase_mode,
            },
        )
        self._schedule_generate(start_offset)

    def stop(self) -> None:
        self._running = False
        self._log("STOP", {"in_flight": len(self._in_flight)})

    def _get_dst_list(self) -> List[int]:
        now = float(self.sim.engine.now)
        if (
            self._dst_cache is None
            or self._dst_cache_time < 0
            or (now - self._dst_cache_time) > self._dst_cache_ttl
        ):
            self._dst_cache = [i for i in self.sim.nodes.keys() if i != 0 and i != self.node.node_id]
            self._dst_cache_time = now
        return self._dst_cache

    def _pick_dst(self) -> int:
        if self.dst_mode == "ap":
            # STA's own live association peer (whichever AP it's actually
            # associated with right now), not a hardcoded 0 -- with a single
            # AP this was always node 0 anyway so the bug was invisible, but
            # under multi-AP it silently made every packet undeliverable to
            # any AP whose node_id isn't literally 0 (see the meta["ap_sink"]
            # tag in _generate_one, which is what actually makes delivery
            # work regardless of which AP ends up receiving it -- this
            # return value is for logging/delivered_by_dst bookkeeping only).
            try:
                peer = self.node.mac.ctx._assoc_peer_id
                if peer is not None:
                    return int(peer)
            except Exception:
                pass
            return 0
        if self.dst_mode == "broadcast":
            return -1

        stas = self._get_dst_list()
        if not stas:
            return 0
        return stas[self.sim.engine.rng.randrange(0, len(stas))]

    def _pick_size(self) -> int:
        if self.size_mode == "fixed":
            return int(self.packet_size)

        if self.size_mode == "uniform":
            lo = max(1, int(self.size_min))
            hi = max(lo, int(self.size_max))
            return int(self.sim.engine.rng.randrange(lo, hi + 1))

        if self.size_mode == "pareto":
            alpha = max(1e-6, float(self.pareto_alpha))
            scale = max(1, int(self.pareto_scale))
            u = self.sim.engine.rng.random()
            u = min(max(u, 1e-12), 1.0 - 1e-12)
            x = scale * ((1.0 - u) ** (-1.0 / alpha))
            hi = max(scale, int(self.size_max))
            return int(min(max(1.0, x), float(hi)))

        if self.size_mode == "table" and self.size_table is not None:
            try:
                if isinstance(self.size_table, dict):
                    sizes = list(self.size_table.get("sizes", []))
                    probs = list(self.size_table.get("probs", []))
                    items = list(zip(sizes, probs))
                else:
                    items = list(self.size_table)

                items = [(int(s), float(p)) for (s, p) in items if float(p) > 0]
                total = sum(p for _, p in items)
                if total <= 0 or not items:
                    return int(self.packet_size)

                r = self.sim.engine.rng.random() * total
                acc = 0.0
                for s, p in items:
                    acc += p
                    if r <= acc:
                        return max(1, int(s))
            except Exception:
                return int(self.packet_size)

        return int(self.packet_size)

    def _cleanup_in_flight(self) -> None:
        if self.in_flight_timeout_s <= 0:
            return

        now = float(self.sim.engine.now)
        stale: List[Tuple[int, int]] = []
        for key, t0 in list(self._in_flight.items()):
            if (now - float(t0)) > float(self.in_flight_timeout_s):
                stale.append(key)

        if not stale:
            return

        for key in stale:
            self._in_flight.pop(key, None)

        self._log(
            "IN_FLIGHT_CLEAN",
            {"removed": len(stale), "remaining": len(self._in_flight)},
        )

        try:
            if hasattr(self.sim, "stats") and hasattr(self.sim.stats, "app_in_flight_timeouts"):
                self.sim.stats.app_in_flight_timeouts += len(stale)
        except Exception:
            pass

    def _generate_one(self) -> None:
        self._extra_generate_pending = False

        self._log(
            "GEN_TICK",
            {
                "running": self._running,
                "traffic": type(self._traffic).__name__ if self._traffic is not None else None,
                "in_flight": len(self._in_flight),
                "max_in_flight": self.max_in_flight,
            },
        )

        if not self._running or self._traffic is None:
            self._log(
                "GEN_STOP",
                {
                    "running": self._running,
                    "traffic_none": self._traffic is None,
                },
            )
            return

        try:
            if hasattr(self._traffic, "set_now"):
                self._traffic.set_now(float(self.sim.engine.now))
        except Exception:
            pass

        self._cleanup_in_flight()

        if self.max_in_flight > 0 and len(self._in_flight) >= self.max_in_flight:
            self._log(
                "RATE_LIMIT",
                {
                    "in_flight": len(self._in_flight),
                    "max_in_flight": self.max_in_flight,
                },
            )
            self._schedule_generate(self.congestion_backoff_s)
            return

        dst = self._pick_dst()
        size_bytes = self._pick_size()

        pkt = Packet(
            packet_seq=self.sim.next_packet_seq(),
            src=self.node.node_id,
            dst=dst,
            size_bytes=size_bytes,
            gen_time=self.sim.engine.now,
            payload=b"x" * min(size_bytes, 32),
        )

        try:
            pkt.traffic_type = self.traffic_type
        except Exception:
            pass

        if self.dst_mode == "ap":
            # Delivery accepts this packet at whichever AP actually
            # receives it, not only the specific AP id captured in `dst`
            # at generation time (see recv_up_from_transport) -- routing
            # already follows the STA's live association peer, so a
            # packet queued just before a handover can legitimately arrive
            # at a different AP than the one `dst` names. Without this,
            # such packets are silently dropped as "not_for_me", biasing
            # exactly the handover-transient PDR measurements multi-AP
            # roaming exists to produce.
            pkt.meta["ap_sink"] = True

        try:
            if self.dscp is None:
                pkt.dscp = int(self._TRAFFIC_TYPE_DSCP.get(self.traffic_type, 0))
            else:
                pkt.dscp = int(self.dscp)
        except Exception:
            pass

        key = (int(pkt.src), int(pkt.packet_seq))
        self._in_flight[key] = float(pkt.gen_time)

        try:
            self.sim.stats.packets_generated += 1
            self._record_priority_generated(aid=self.node.node_id)
            self.sim.stats.generated_by_src[self.node.node_id] += 1
        except Exception:
            pass

        self._log(
            "GENERATE",
            {
                "size_bytes": pkt.size_bytes,
                "dst": dst,
                "in_flight": len(self._in_flight),
            },
            packet=pkt,
        )

        try:
            self.node.transport.send_down(pkt)
        except Exception as e:
            self._in_flight.pop(key, None)
            self._log("GENERATE_FAIL", {"error": str(e)}, packet=pkt)
            self._schedule_generate(self.congestion_backoff_s)
            return

        dt = max(0.0, float(self._traffic.next_interval(self.sim.engine.rng)))
        self._schedule_generate(dt)

    def recv_up_from_transport(self, packet: Packet) -> None:
        if not self.enable_sink:
            return

        # An "ap_sink" packet (dst_mode=="ap", see _generate_one) is
        # deliverable at whichever AP actually receives it, not only the
        # specific AP id its `dst` field named at generation time -- see
        # the comment there for why (live-peer routing vs. a static dst
        # snapshotted before a possible mid-flight handover).
        is_ap_sink = bool(self.node.is_ap and packet.meta.get("ap_sink"))
        if packet.dst != self.node.node_id and packet.dst != -1 and not is_ap_sink:
            self._log(
                "WARN",
                {"reason": "not_for_me", "dst": packet.dst, "my_id": self.node.node_id},
                packet=packet,
            )
            return

        self._rx_packets += 1

        delay = float(self.sim.engine.now) - float(packet.gen_time)

        try:
            self.sim.stats.packets_delivered += 1
            self.sim.stats.delivered_by_dst[self.node.node_id] += 1
        except Exception:
            pass

        self._record_priority_delivered(packet=packet, delay=delay)

        try:
            src = int(packet.src)
            self._delivered_from_src[src] = self._delivered_from_src.get(src, 0) + 1
        except Exception:
            pass

        try:
            self.sim.stats.delays.append(delay)
        except Exception:
            pass

        self._delays_local.append(delay)

        try:
            if hasattr(self.sim.stats, "delays_per_node"):
                self.sim.stats.delays_per_node.setdefault(self.node.node_id, []).append(delay)
        except Exception:
            pass

        try:
            if hasattr(self.sim.stats, "aoi_records"):
                self.sim.stats.aoi_records[int(packet.src)].append(
                    (float(packet.gen_time), float(self.sim.engine.now))
                )
        except Exception:
            pass

        jitter = None
        try:
            now = float(self.sim.engine.now)
            src = int(packet.src)
            last = self._last_delivery_time.get(src)
            if last is not None:
                jitter = abs((now - float(last)) - float(self.expected_interval_s))
                self._jitter_samples.append(jitter)
            self._last_delivery_time[src] = now
        except Exception:
            pass

        try:
            if jitter is not None and hasattr(self.sim.stats, "jitter_per_node"):
                self.sim.stats.jitter_per_node[self.node.node_id].append(float(jitter))
        except Exception:
            pass

        goodput_bps = None
        try:
            now = float(self.sim.engine.now)
            if self._goodput_window_start is None:
                self._goodput_window_start = now
            self._bytes_received += int(getattr(packet, "size_bytes", 0))
            elapsed = now - float(self._goodput_window_start)
            if elapsed >= float(self.goodput_window_s):
                goodput_bps = (self._bytes_received * 8.0) / max(elapsed, 1e-12)
                self._bytes_received = 0
                self._goodput_window_start = now
        except Exception:
            pass

        try:
            if goodput_bps is not None and hasattr(self.sim.stats, "goodput_bps_per_node"):
                self.sim.stats.goodput_bps_per_node[self.node.node_id] = float(goodput_bps)
        except Exception:
            pass

        details: Dict[str, Any] = {"delay_s": delay}
        if jitter is not None:
            details["jitter_s"] = jitter
        if goodput_bps is not None:
            details["goodput_bps"] = round(goodput_bps, 1)

        self._log("DELIVER", details, packet=packet)

    def on_tx_success(self, packet: Packet) -> None:
        try:
            key = (
                int(getattr(packet, "src", self.node.node_id)),
                int(getattr(packet, "packet_seq", -1)),
            )
            self._in_flight.pop(key, None)
        except Exception:
            pass

        self._log(
            "TX_SUCCESS",
            {
                "dst": getattr(packet, "dst", None),
                "remaining_in_flight": len(self._in_flight),
            },
            packet=packet,
        )

    def on_tx_failure(self, packet: Packet) -> None:
        try:
            key = (
                int(getattr(packet, "src", self.node.node_id)),
                int(getattr(packet, "packet_seq", -1)),
            )
            self._in_flight.pop(key, None)
        except Exception:
            pass

        try:
            self.sim.stats.tx_failures_by_src[self.node.node_id] += 1
        except Exception:
            pass

        self._log(
            "TX_FAILURE",
            {
                "dst": getattr(packet, "dst", None),
                "tp_seq": getattr(packet, "tp_seq", None),
                "remaining_in_flight": len(self._in_flight),
            },
            packet=packet,
        )

    def finalize(self) -> None:
        nid = self.node.node_id
        try:
            gen = int(self.sim.stats.generated_by_src.get(nid, 0))
        except Exception:
            gen = 0

        try:
            rx = int(self.sim.stats.delivered_by_dst.get(nid, 0))
        except Exception:
            rx = self._rx_packets

        delays = None
        try:
            if hasattr(self.sim.stats, "delays_per_node"):
                delays = self.sim.stats.delays_per_node.get(nid, [])
        except Exception:
            delays = None
        if delays is None:
            delays = self._delays_local

        mean_delay = (sum(delays) / len(delays)) if delays else 0.0
        max_delay = max(delays) if delays else 0.0

        details: Dict[str, Any] = {
            "generated": gen,
            "received": rx,
            "mean_delay_s": round(float(mean_delay), 6),
            "max_delay_s": round(float(max_delay), 6),
            "in_flight_remaining": len(self._in_flight),
            "received_from": dict(self._delivered_from_src),
        }

        if self._jitter_samples:
            details["mean_jitter_s"] = round(sum(self._jitter_samples) / len(self._jitter_samples), 6)
            details["max_jitter_s"] = round(max(self._jitter_samples), 6)

        try:
            if hasattr(self.sim.stats, "goodput_bps_per_node"):
                gp = self.sim.stats.goodput_bps_per_node.get(nid, None)
                if gp is not None:
                    details["goodput_bps"] = round(float(gp), 1)
        except Exception:
            pass

        self._log("FINAL_STATS", details)

    def debug_state(self) -> Dict[str, Any]:
        nid = self.node.node_id
        gen = 0
        rx = 0
        try:
            gen = int(self.sim.stats.generated_by_src.get(nid, 0))
            rx = int(self.sim.stats.delivered_by_dst.get(nid, 0))
        except Exception:
            pass

        return {
            "node_id": nid,
            "running": self._running,
            "traffic": type(self._traffic).__name__ if self._traffic is not None else None,
            "in_flight": len(self._in_flight),
            "generated": gen,
            "received": rx,
            "dst_mode": self.dst_mode,
            "dst_cache_len": (len(self._dst_cache) if self._dst_cache is not None else None),
            "dst_cache_age_s": (
                float(self.sim.engine.now) - self._dst_cache_time
            ) if self._dst_cache_time >= 0 else None,
            "extra_generate_pending": self._extra_generate_pending,
        }