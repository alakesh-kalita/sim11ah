from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RawSlotRecord:
    group_id: int
    slot_idx: int
    start_t: float
    end_t: float
    start_aid: int
    end_aid: int

    bytes_tx: int = 0
    pkts_tx: int = 0
    success_tx: int = 0
    retry_tx: int = 0
    timeout_tx: int = 0

    active_stas: int = 0
    backlog_sum: int = 0

    airtime_tx_s: float = 0.0

    def duration(self) -> float:
        return max(0.0, float(self.end_t) - float(self.start_t))

    def utilized(self) -> bool:
        return self.pkts_tx > 0 or self.bytes_tx > 0 or self.airtime_tx_s > 0.0

    def activity_ratio(self) -> float:
        return 1.0 if self.utilized() else 0.0

    def utilization_ratio(self) -> float:
        dur = self.duration()
        if dur <= 0.0:
            return 0.0
        return min(1.0, max(0.0, float(self.airtime_tx_s) / dur))

    def overloaded(self) -> bool:
        return self.backlog_sum > max(1, self.active_stas)


@dataclass
class TwtWakeRecord:
    sta_id: int
    wake_start: float
    wake_end: float = 0.0
    tx_pkts: int = 0
    rx_pkts: int = 0

    def awake_time(self, now: Optional[float] = None) -> float:
        end_t = self.wake_end
        if end_t <= self.wake_start and now is not None:
            end_t = float(now)
        return max(0.0, float(end_t) - float(self.wake_start))


class MacMetrics:
    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn

        self.delay_samples_s: List[float] = []

        self.tx_attempts_total: int = 0
        self.tx_success_total: int = 0
        self.tx_drop_total: int = 0
        self.ack_timeout_total: int = 0

        self.per_sta_attempts: Dict[int, int] = {}
        self.per_sta_success_pkts: Dict[int, int] = {}
        self.per_sta_success_bytes: Dict[int, int] = {}
        self.per_sta_retries: Dict[int, int] = {}
        self.per_sta_timeouts: Dict[int, int] = {}
        self.per_sta_drops: Dict[int, int] = {}
        self.per_sta_delays: Dict[int, List[float]] = {}

        self.raw_slots: List[RawSlotRecord] = []
        self.per_group_backlog: Dict[int, List[int]] = {}

        self._twt_open: Dict[int, TwtWakeRecord] = {}
        self.twt_completed: List[TwtWakeRecord] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_sta_id(
        self,
        sta_id: Optional[int] = None,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> int:
        if sta_id is not None:
            return int(sta_id)
        if src is not None:
            return int(src)
        if dst is not None:
            return int(dst)
        return -1

    def _ensure_sta_maps(self, sta_id: int) -> None:
        self.per_sta_attempts.setdefault(sta_id, 0)
        self.per_sta_success_pkts.setdefault(sta_id, 0)
        self.per_sta_success_bytes.setdefault(sta_id, 0)
        self.per_sta_retries.setdefault(sta_id, 0)
        self.per_sta_timeouts.setdefault(sta_id, 0)
        self.per_sta_drops.setdefault(sta_id, 0)
        self.per_sta_delays.setdefault(sta_id, [])

    def _resolve_raw_aid(
        self,
        aid: Optional[int] = None,
        sta_id: Optional[int] = None,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> int:
        """
        Resolve the STA/AID that should own RAW-slot usage.

        Preference:
        1. explicit aid
        2. explicit sta_id
        3. explicit src  (usually correct for uplink)
        4. explicit dst  (legacy fallback)
        """
        if aid is not None:
            return int(aid)
        if sta_id is not None:
            return int(sta_id)
        if src is not None:
            return int(src)
        if dst is not None:
            return int(dst)
        return -1

    # ------------------------------------------------------------------
    # Delay / success / retry / timeout / drop
    # ------------------------------------------------------------------
    def record_tx_attempt(
        self,
        sta_id: Optional[int] = None,
        size_bytes: int = 0,
        *,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> None:
        _ = int(size_bytes)
        key = self._resolve_sta_id(sta_id=sta_id, dst=dst, src=src)

        self.tx_attempts_total += 1
        self._ensure_sta_maps(key)
        self.per_sta_attempts[key] += 1

    def record_tx_success(
        self,
        sta_id: Optional[int] = None,
        size_bytes: int = 0,
        delay_s: Optional[float] = None,
        *,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> None:
        key = self._resolve_sta_id(sta_id=sta_id, dst=dst, src=src)

        self.tx_success_total += 1
        self._ensure_sta_maps(key)

        self.per_sta_success_pkts[key] += 1
        self.per_sta_success_bytes[key] += int(size_bytes)

        if delay_s is not None:
            d = float(delay_s)
            self.delay_samples_s.append(d)
            self.per_sta_delays[key].append(d)

        if key in self._twt_open:
            self._twt_open[key].tx_pkts += 1

    def record_retry(
        self,
        sta_id: Optional[int] = None,
        *,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> None:
        key = self._resolve_sta_id(sta_id=sta_id, dst=dst, src=src)
        self._ensure_sta_maps(key)
        self.per_sta_retries[key] += 1

    def record_timeout(
        self,
        sta_id: Optional[int] = None,
        *,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> None:
        key = self._resolve_sta_id(sta_id=sta_id, dst=dst, src=src)
        self._ensure_sta_maps(key)
        self.ack_timeout_total += 1
        self.per_sta_timeouts[key] += 1

    def record_drop(
        self,
        sta_id: Optional[int] = None,
        *,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> None:
        key = self._resolve_sta_id(sta_id=sta_id, dst=dst, src=src)
        self._ensure_sta_maps(key)
        self.tx_drop_total += 1
        self.per_sta_drops[key] += 1

    # ------------------------------------------------------------------
    # RAW slot metrics
    # ------------------------------------------------------------------
    def register_raw_slots(self, rps: List[Dict[str, Any]], beacon_rx_t: float, raw_guard: float) -> None:
        for group in rps:
            group_id = int(group["group_id"])
            start_aid = int(group["start_aid"])
            end_aid = int(group["end_aid"])
            num_slots = max(1, int(group["num_slots"]))
            slot_dur = float(group["slot_duration"])
            start_off = float(group["start_offset"])

            group_size = max(0, end_aid - start_aid + 1)
            backlog = self._estimate_group_backlog(start_aid, end_aid)
            self.per_group_backlog.setdefault(group_id, []).append(backlog)

            for slot_idx in range(num_slots):
                start_t = float(beacon_rx_t) + start_off + slot_idx * slot_dur + float(raw_guard)
                end_t = float(beacon_rx_t) + start_off + (slot_idx + 1) * slot_dur - float(raw_guard)

                if end_t <= start_t:
                    self._log(
                        "RAW_METRICS_INVALID_SLOT",
                        {
                            "group_id": group_id,
                            "slot_idx": slot_idx,
                            "start_t": start_t,
                            "end_t": end_t,
                            "slot_dur": slot_dur,
                            "raw_guard": raw_guard,
                        },
                    )
                    end_t = start_t

                contenders_this_slot = 0
                if group_size > 0 and slot_idx < num_slots:
                    contenders_this_slot = len(
                        [aid for aid in range(start_aid, end_aid + 1)
                         if ((aid - start_aid) % num_slots) == slot_idx]
                    )

                rec = RawSlotRecord(
                    group_id=group_id,
                    slot_idx=slot_idx,
                    start_t=start_t,
                    end_t=end_t,
                    start_aid=start_aid,
                    end_aid=end_aid,
                    active_stas=contenders_this_slot,
                    backlog_sum=backlog,
                )
                self.raw_slots.append(rec)

    def _estimate_group_backlog(self, start_aid: int, end_aid: int) -> int:
        total = 0
        sta_queue_hint = getattr(self.ctx, "_sta_queue_hint", {})
        ps_buf = getattr(self.ctx, "_sta_ps_buffer", {})

        for aid in range(start_aid, end_aid + 1):
            total += int(sta_queue_hint.get(aid, 0))
            if aid in ps_buf:
                total += len(ps_buf[aid])
        return total

    def record_raw_tx(
        self,
        now: float,
        size_bytes: int,
        success: bool = False,
        retry: bool = False,
        timeout: bool = False,
        airtime_s: float = 0.0,
        *,
        aid: Optional[int] = None,
        sta_id: Optional[int] = None,
        dst: Optional[int] = None,
        src: Optional[int] = None,
    ) -> None:
        """
        Backward-compatible RAW TX accounting.

        Preferred call:
            record_raw_tx(now=..., size_bytes=..., src=sta_aid, ...)

        Also accepts old calls using dst=...
        """
        resolved_aid = self._resolve_raw_aid(
            aid=aid,
            sta_id=sta_id,
            dst=dst,
            src=src,
        )
        if resolved_aid < 0:
            return

        rec = self._find_slot_for_sta(now=float(now), aid=int(resolved_aid))
        if rec is None:
            return

        rec.bytes_tx += int(size_bytes)
        rec.pkts_tx += 1
        rec.airtime_tx_s += max(0.0, float(airtime_s))

        if success:
            rec.success_tx += 1
        if retry:
            rec.retry_tx += 1
        if timeout:
            rec.timeout_tx += 1

    def _find_slot_for_sta(self, now: float, aid: int) -> Optional[RawSlotRecord]:
        best: Optional[RawSlotRecord] = None

        for rec in self.raw_slots:
            if rec.start_aid <= aid <= rec.end_aid and rec.start_t <= now <= rec.end_t:
                if best is None or rec.start_t > best.start_t:
                    best = rec

        return best

    # ------------------------------------------------------------------
    # TWT metrics
    # ------------------------------------------------------------------
    def twt_wake_enter(self, sta_id: int, now: float) -> None:
        self._twt_open[int(sta_id)] = TwtWakeRecord(sta_id=int(sta_id), wake_start=float(now))

    def twt_wake_exit(self, sta_id: int, now: float) -> None:
        rec = self._twt_open.pop(int(sta_id), None)
        if rec is None:
            return
        rec.wake_end = float(now)
        self.twt_completed.append(rec)

    def twt_record_tx(self, sta_id: int) -> None:
        sta_id = int(sta_id)
        if sta_id in self._twt_open:
            self._twt_open[sta_id].tx_pkts += 1

    def twt_record_rx(self, sta_id: int) -> None:
        sta_id = int(sta_id)
        if sta_id in self._twt_open:
            self._twt_open[sta_id].rx_pkts += 1

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    def mean_delay_s(self) -> float:
        if not self.delay_samples_s:
            return 0.0
        return sum(self.delay_samples_s) / len(self.delay_samples_s)

    def p95_delay_s(self) -> float:
        if not self.delay_samples_s:
            return 0.0
        vals = sorted(self.delay_samples_s)
        idx = max(0, min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1))
        return vals[idx]

    def pdr(self) -> float:
        if self.tx_attempts_total <= 0:
            return 0.0
        return self.tx_success_total / self.tx_attempts_total

    def fairness_index(self) -> float:
        xs = [float(v) for v in self.per_sta_success_bytes.values()]
        if not xs:
            return 0.0

        s1 = sum(xs)
        s2 = sum(x * x for x in xs)
        n = len(xs)

        if s2 <= 0.0 or n <= 0:
            return 0.0
        return (s1 * s1) / (n * s2)

    def raw_slot_activity_ratio(self) -> float:
        if not self.raw_slots:
            return 0.0
        active = sum(1 for r in self.raw_slots if r.utilized())
        return active / len(self.raw_slots)

    def raw_slot_utilization(self) -> float:
        if not self.raw_slots:
            return 0.0

        total_airtime = sum(max(0.0, r.airtime_tx_s) for r in self.raw_slots)
        total_slot_time = sum(r.duration() for r in self.raw_slots)

        if total_slot_time > 0.0 and total_airtime > 0.0:
            return min(1.0, max(0.0, total_airtime / total_slot_time))

        return self.raw_slot_activity_ratio()

    def idle_slots(self) -> int:
        return sum(1 for r in self.raw_slots if not r.utilized())

    def overloaded_slots(self) -> int:
        return sum(1 for r in self.raw_slots if r.overloaded())

    def mean_group_backlog(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for gid, vals in self.per_group_backlog.items():
            out[gid] = (sum(vals) / len(vals)) if vals else 0.0
        return out

    def twt_awake_time_proxy(self, now: Optional[float] = None) -> float:
        total = sum(r.awake_time() for r in self.twt_completed)
        if now is not None:
            total += sum(r.awake_time(now=float(now)) for r in self._twt_open.values())
        return total

    def summary(self, sim_duration_s: Optional[float] = None) -> Dict[str, Any]:
        throughput_bps = 0.0
        if sim_duration_s is not None and sim_duration_s > 0.0:
            throughput_bps = 8.0 * sum(self.per_sta_success_bytes.values()) / sim_duration_s

        now = None
        try:
            now = float(getattr(self.ctx.sim.engine, "now"))
        except Exception:
            try:
                now = float(getattr(self.ctx, "now"))
            except Exception:
                now = None

        return {
            "throughput_bps": throughput_bps,
            "pdr": self.pdr(),
            "mean_delay_s": self.mean_delay_s(),
            "p95_delay_s": self.p95_delay_s(),
            "retry_count": sum(self.per_sta_retries.values()),
            "timeout_count": self.ack_timeout_total,
            "drop_count": self.tx_drop_total,
            "fairness_index": self.fairness_index(),
            "raw_slot_activity_ratio": self.raw_slot_activity_ratio(),
            "raw_slot_utilization": self.raw_slot_utilization(),
            "idle_slots": self.idle_slots(),
            "overloaded_slots": self.overloaded_slots(),
            "per_group_backlog_mean": self.mean_group_backlog(),
            "twt_awake_time_proxy_s": self.twt_awake_time_proxy(now=now),
        }