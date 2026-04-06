import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass(order=True)
class ScheduledEvent:
    """
    Heap-ordered scheduled event.

    Ordering fields:
      - time
      - seq

    Non-ordering fields are ignored by heap comparisons.
    """
    time: float
    seq: int
    eid: int = field(compare=False)
    cb: Callable[..., None] = field(compare=False)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    name: str = field(compare=False, default="")


class EventEngine:
    """
    Deterministic discrete-event engine.

    Features:
      - schedule(time_abs, cb, *args, name=None, **kwargs) -> event_id
      - schedule_in(delay, cb, *args, name=None, **kwargs) -> event_id
      - cancel(event_id) best-effort (lazy skip)
      - run(until=None): executes all events up to 'until' (hard cutoff)
      - step(n=1, until=None): executes up to n events (and respects until)
      - step_current_time(until=None): executes all events at the next event time
      - run_for(dt): run until now+dt
      - stop(): stops current run/step loop
      - reset(seed=None): clear queue and reset state

    Debug / safety additions:
      - event names
      - kwargs support
      - helpful exception context
      - finite-time validation
      - basic execution statistics
      - live next-event introspection
      - runaway same-time execution protection
    """

    def __init__(
        self,
        seed: int = 0,
        strict_past_scheduling: bool = False,
        max_same_time_events: int = 1_000_000,
    ):
        self.now: float = 0.0

        self._q: List[ScheduledEvent] = []
        self._seq: int = 0
        self._eid: int = 0

        self.rng = random.Random(seed)

        self._stopped: bool = False
        self._cancelled: Set[int] = set()

        self.strict_past_scheduling: bool = bool(strict_past_scheduling)
        self.max_same_time_events: int = int(max_same_time_events)

        self._same_time_counter: int = 0
        self._last_exec_time: Optional[float] = None

        # Optional hook: hook(event, phase) where phase in {"before", "after", "cancelled_skip"}
        self.trace_hook: Optional[Callable[[ScheduledEvent, str], None]] = None

        self.stats: Dict[str, int] = {
            "scheduled": 0,
            "executed": 0,
            "cancel_requested": 0,
            "cancelled_skipped": 0,
            "max_queue_len": 0,
        }

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _validate_time(self, t: float) -> float:
        t = float(t)
        if not math.isfinite(t):
            raise ValueError(f"Event time must be finite, got {t!r}")
        return t

    def _validate_delay(self, dt: float) -> float:
        dt = float(dt)
        if not math.isfinite(dt):
            raise ValueError(f"Delay must be finite, got {dt!r}")
        return dt

    def _validate_callback(self, cb: Callable[..., None]) -> None:
        if not callable(cb):
            raise TypeError(f"Scheduled callback must be callable, got {type(cb).__name__}")

    def _make_event_name(self, cb: Callable[..., None], name: Optional[str]) -> str:
        if name is not None and str(name).strip():
            return str(name)
        return getattr(cb, "__name__", cb.__class__.__name__)

    def _update_max_queue_len(self) -> None:
        qlen = len(self._q)
        if qlen > self.stats["max_queue_len"]:
            self.stats["max_queue_len"] = qlen

    def _count_same_time_execution(self, t: float) -> None:
        if self._last_exec_time is None or t != self._last_exec_time:
            self._last_exec_time = t
            self._same_time_counter = 1
            return

        self._same_time_counter += 1
        if self._same_time_counter > self.max_same_time_events:
            raise RuntimeError(
                "Runaway event execution detected: too many events executed "
                f"at the same simulation time t={t}. "
                f"Threshold={self.max_same_time_events}."
            )

    # ------------------------------------------------------------
    # Scheduling / cancellation
    # ------------------------------------------------------------

    def schedule(
        self,
        time_abs: float,
        cb: Callable[..., None],
        *args: Any,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """
        Schedule callback at absolute simulation time.
        Returns integer event id.
        """
        self._validate_callback(cb)
        time_abs = self._validate_time(time_abs)

        if time_abs < self.now:
            if self.strict_past_scheduling:
                raise ValueError(
                    f"Cannot schedule event in the past: time_abs={time_abs}, now={self.now}"
                )
            time_abs = self.now

        self._seq += 1
        self._eid += 1
        eid = self._eid

        ev = ScheduledEvent(
            time=time_abs,
            seq=self._seq,
            eid=eid,
            cb=cb,
            args=args,
            kwargs=dict(kwargs),
            name=self._make_event_name(cb, name),
        )
        heapq.heappush(self._q, ev)

        self.stats["scheduled"] += 1
        self._update_max_queue_len()
        return eid

    def schedule_in(
        self,
        delay: float,
        cb: Callable[..., None],
        *args: Any,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """
        Schedule callback after delay relative to current simulation time.
        Negative delay is clamped to zero unless invalid (NaN/inf).
        """
        delay = self._validate_delay(delay)
        return self.schedule(self.now + max(0.0, delay), cb, *args, name=name, **kwargs)

    def cancel(self, event_id: int) -> bool:
        """
        Best-effort cancel. Returns True if the id was newly cancelled.
        Cancellation is lazy: heap entries remain until popped.
        """
        if event_id <= 0:
            return False
        if event_id in self._cancelled:
            return False

        self._cancelled.add(event_id)
        self.stats["cancel_requested"] += 1
        return True

    # ------------------------------------------------------------
    # Queue access
    # ------------------------------------------------------------

    def _pop_next(self) -> Optional[ScheduledEvent]:
        """
        Pops next non-cancelled event.
        Returns ScheduledEvent or None if queue is empty.
        """
        while self._q:
            ev = heapq.heappop(self._q)
            if ev.eid in self._cancelled:
                self._cancelled.discard(ev.eid)
                self.stats["cancelled_skipped"] += 1
                if self.trace_hook is not None:
                    self.trace_hook(ev, "cancelled_skip")
                continue
            return ev
        return None

    def _push_back(self, ev: ScheduledEvent) -> None:
        heapq.heappush(self._q, ev)
        self._update_max_queue_len()

    def _peek_next_live_event(self) -> Optional[ScheduledEvent]:
        """
        Returns the next non-cancelled event without permanently removing it.
        This may lazily discard cancelled events at the head of the heap.
        """
        while self._q:
            ev = self._q[0]
            if ev.eid in self._cancelled:
                heapq.heappop(self._q)
                self._cancelled.discard(ev.eid)
                self.stats["cancelled_skipped"] += 1
                if self.trace_hook is not None:
                    self.trace_hook(ev, "cancelled_skip")
                continue
            return ev
        return None

    def _execute_event(self, ev: ScheduledEvent) -> None:
        self.now = float(ev.time)
        self._count_same_time_execution(self.now)

        if self.trace_hook is not None:
            self.trace_hook(ev, "before")

        try:
            ev.cb(*ev.args, **ev.kwargs)
        except Exception as e:
            raise RuntimeError(
                "Error while executing scheduled event: "
                f"eid={ev.eid}, time={ev.time}, name={ev.name}, "
                f"callback={getattr(ev.cb, '__name__', repr(ev.cb))}, "
                f"args={ev.args}, kwargs={ev.kwargs}"
            ) from e

        self.stats["executed"] += 1

        if self.trace_hook is not None:
            self.trace_hook(ev, "after")

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def run(self, until: Optional[float] = None) -> None:
        """
        Run events in time order. If until is given, execute only events with t <= until.
        Hard cutoff: does not execute events beyond until.
        """
        self._stopped = False
        limit = None if until is None else self._validate_time(until)

        while not self._stopped:
            ev = self._pop_next()
            if ev is None:
                break

            if limit is not None and ev.time > limit:
                self._push_back(ev)
                self.now = limit
                break

            self._execute_event(ev)

    def step(self, n: int = 1, until: Optional[float] = None) -> int:
        """
        Execute up to n events (useful for GUI stepping).
        Respects until cutoff if given.
        Returns the number of executed callbacks.
        """
        self._stopped = False
        limit = None if until is None else self._validate_time(until)
        executed = 0
        target = max(0, int(n))

        while executed < target and not self._stopped:
            ev = self._pop_next()
            if ev is None:
                break

            if limit is not None and ev.time > limit:
                self._push_back(ev)
                self.now = limit
                break

            self._execute_event(ev)
            executed += 1

        return executed

    def step_current_time(self, until: Optional[float] = None) -> int:
        """
        Execute all events scheduled at the next live event time.
        Very useful for GUI stepping through one simulation timestamp at a time.
        Returns number of executed callbacks.
        """
        self._stopped = False
        limit = None if until is None else self._validate_time(until)

        first = self._peek_next_live_event()
        if first is None:
            return 0

        t0 = first.time
        if limit is not None and t0 > limit:
            self.now = limit
            return 0

        executed = 0
        while not self._stopped:
            ev = self._pop_next()
            if ev is None:
                break

            if ev.time != t0:
                self._push_back(ev)
                break

            if limit is not None and ev.time > limit:
                self._push_back(ev)
                self.now = limit
                break

            self._execute_event(ev)
            executed += 1

        return executed

    def run_for(self, dt: float) -> None:
        """Convenience: run until now + dt."""
        dt = self._validate_delay(dt)
        self.run(until=self.now + max(0.0, dt))

    def stop(self) -> None:
        """Stop the currently running run()/step() loop."""
        self._stopped = True

    # ------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------

    def pending_events(self) -> int:
        """
        Raw heap item count.
        Includes cancelled events that are still lazily present in the heap.
        """
        return len(self._q)

    def cancelled_pending(self) -> int:
        """Number of cancelled event ids still awaiting lazy removal."""
        return len(self._cancelled)

    def pending_live_events_estimate(self) -> int:
        """
        Approximate live event count.
        This subtracts currently tracked cancelled ids from heap size.
        It is still an estimate, but more useful than raw heap length.
        """
        return max(0, len(self._q) - len(self._cancelled))

    def has_pending(self) -> bool:
        """True if heap contains any items, even if some are cancelled."""
        return len(self._q) > 0

    def has_pending_live(self) -> bool:
        """True if at least one non-cancelled event remains."""
        return self._peek_next_live_event() is not None

    def peek_next_time(self) -> Optional[float]:
        """
        Returns the next non-cancelled event time if any.
        """
        ev = self._peek_next_live_event()
        if ev is None:
            return None
        return float(ev.time)

    def peek_next_event(self) -> Optional[Dict[str, Any]]:
        """
        Returns a lightweight description of the next non-cancelled event.
        """
        ev = self._peek_next_live_event()
        if ev is None:
            return None

        return {
            "time": float(ev.time),
            "seq": ev.seq,
            "eid": ev.eid,
            "name": ev.name,
            "callback": getattr(ev.cb, "__name__", repr(ev.cb)),
            "args": ev.args,
            "kwargs": ev.kwargs,
        }

    def is_stopped(self) -> bool:
        return self._stopped

    def current_time(self) -> float:
        return float(self.now)

    def get_stats(self) -> Dict[str, int]:
        """Returns a shallow copy of engine statistics."""
        return dict(self.stats)

    def reset(self, seed: Optional[int] = None) -> None:
        """Clear queue and reset time. Optionally reseed RNG."""
        self.now = 0.0
        self._q.clear()
        self._seq = 0
        self._eid = 0
        self._cancelled.clear()
        self._stopped = False
        self._same_time_counter = 0
        self._last_exec_time = None

        self.stats = {
            "scheduled": 0,
            "executed": 0,
            "cancel_requested": 0,
            "cancelled_skipped": 0,
            "max_queue_len": 0,
        }

        if seed is not None:
            self.rng = random.Random(int(seed))