"""
Shared per-station traffic-interval estimator: E-TAROA's Algorithm 1
(Tian, Santi, Latre, Famaey, "Accurate Sensor Traffic Estimation for
Station Grouping in Highly Dense IEEE 802.11ah Networks," SenSys 2017).

Extracted out of raw_policy_etaroa.py so any RAW policy can swap this in
as its demand-estimation step in place of a cruder CUSUM-EWMA-on-queue
estimator -- this is a strictly more information-rich signal (it exploits
per-station success/failure history and the real "More Data" ctrl field,
see dcf.py) for the same periodic/regular-interval sensor traffic every
policy in this simulator is benchmarked against, and it's the natural
piece to reuse for a policy that wants better demand accuracy while
keeping its own group-formation strategy (CSV clusters, adaptive
clusters, whatever) unchanged.

See raw_policy_etaroa.py's class docstring for the full paper citation
and the two documented simplifications (c_succ, Delta_m update rule) --
both apply here identically since this file *is* that same estimator.
"""
from __future__ import annotations

from typing import Dict, List, Optional

_SUCCESS = "success"
_FAILED = "failed"


class _StaState:
    """Per-station persistent state for Algorithm 1 (Table 1 in the paper).
    t_succ0/t_succ1 are Optional: None means "never had a successful
    reception yet" -- kept distinct from 0.0, which is a legitimate beacon
    index (beacon #0), so Case 1/2/3.2's formulas never silently treat a
    station that hasn't transmitted yet as having succeeded at tc=0."""

    __slots__ = (
        "t_int", "t_next", "t_succ0", "t_succ1", "trans0", "trans1",
        "m_succ0", "m_succ1", "failed", "delta_m", "last_rx_count", "seen",
    )

    def __init__(self, initial_t_int: float) -> None:
        self.t_int = float(initial_t_int)
        self.t_next: float = 0.0
        self.t_succ0: Optional[float] = None
        self.t_succ1: Optional[float] = None
        self.trans0: Optional[str] = None
        self.trans1: Optional[str] = None
        self.m_succ0: bool = False
        self.m_succ1: bool = False
        self.failed: int = 0
        self.delta_m: int = 0
        self.last_rx_count: int = 0
        self.seen: bool = False


class EtaroaTrafficEstimator:
    """Runs Algorithm 1 for a whole connected-AID set, once per DTIM
    beacon. `update(aids)` returns aid -> estimated packets per beacon
    interval (== 1/t_int), the natural "demand" value for RAW sizing."""

    def __init__(self, ctx, default_t_int_beacons: float = 1.0) -> None:
        self.ctx = ctx
        self.default_t_int = float(default_t_int_beacons)
        self._sta: Dict[int, _StaState] = {}
        self._update_count = 0

    def update(self, aids: List[int]) -> Dict[int, float]:
        tc = float(getattr(self.ctx, "_ap_beacon_count", self._update_count))
        rx_by_src = getattr(self.ctx, "_rx_count_by_src", None) or {}
        more_data_by_src = getattr(self.ctx, "_more_data_by_src", None) or {}

        rates: Dict[int, float] = {}
        for aid in aids:
            st = self._sta.get(aid)
            if st is None:
                st = _StaState(self.default_t_int)
                self._sta[aid] = st

            current_rx = int(rx_by_src.get(aid, 0))
            pi_b = max(0, current_rx - st.last_rx_count)
            st.last_rx_count = current_rx
            more_data_now = bool(more_data_by_src.get(aid, False))

            if not st.seen:
                st.seen = True
                st.trans0 = _SUCCESS if pi_b > 0 else _FAILED
                st.trans1 = None
                if pi_b > 0:
                    st.t_succ0 = tc
                    st.m_succ0 = more_data_now
                st.t_next = tc + st.t_int
                rates[aid] = 1.0 / max(1e-6, st.t_int)
                continue

            self._step(st, tc=tc, pi_b=pi_b, more_data_now=more_data_now)
            rates[aid] = 1.0 / max(1e-6, st.t_int)

        self._update_count += 1
        return rates

    def _step(self, st: _StaState, *, tc: float, pi_b: int, more_data_now: bool) -> None:
        """One DTIM-beacon update of Algorithm 1 for a single station.
        Bookkeeping-then-evaluate ordering -- see raw_policy_etaroa.py's
        module docstring for why (the paper's own case-2 text requires
        trans[0] to already reflect THIS beacon's outcome at dispatch
        time, not the stale pre-shift history)."""
        c_succ = False  # not modeled -- see raw_policy_etaroa.py docstring

        this_trans = _SUCCESS if pi_b > 0 else _FAILED
        prev_trans1 = st.trans0
        prev_m1 = st.m_succ0
        prev_failed = st.failed

        st.trans1 = prev_trans1
        st.trans0 = this_trans
        if this_trans == _SUCCESS:
            st.m_succ1 = prev_m1
            st.m_succ0 = more_data_now
            st.t_succ1 = st.t_succ0
            st.t_succ0 = tc

        if st.trans0 == _FAILED and st.m_succ0 is False:
            st.failed = prev_failed + 1
            if st.t_succ0 is not None:
                st.t_int = tc - st.t_succ0 + 2 * st.failed - 1

        elif st.trans0 == _SUCCESS and st.trans1 == _FAILED:
            st.failed = 0
            if st.t_succ1 is not None:
                if c_succ:
                    st.t_succ0 -= 1
                st.t_int = st.t_succ0 - st.t_succ1

        elif pi_b == 1:
            st.failed = 0
            if st.t_int > 1 and prev_m1 is True and st.m_succ0 is False:
                t_int_min = max(
                    st.t_int - 2 * (prev_failed - 1),
                    (st.delta_m / (st.delta_m + 2.0)) * st.t_int + 1.0,
                )
                t_int_max = st.t_int - 1
                st.t_int = (t_int_min + t_int_max) / 2.0
                st.delta_m = 0
            elif st.t_int > 1:
                if st.t_succ1 is not None:
                    st.t_int = st.t_succ0 - st.t_succ1
            elif not c_succ and st.m_succ0 is False:
                st.t_int = 1.0

        elif pi_b > 1:
            st.failed = 0
            if st.t_int > 1:
                st.t_int = st.t_int - 1
            else:
                inv = 1.0 / max(1e-6, st.t_int)
                if pi_b > inv:
                    inv += 1.0
                elif pi_b < inv and st.m_succ0 is False:
                    inv -= 1.0
                st.t_int = 1.0 / max(1e-6, inv)

        st.t_int = max(1e-3, st.t_int)

        if this_trans == _SUCCESS:
            if more_data_now:
                st.delta_m = 0
            else:
                st.delta_m += 1

        st.t_next = st.t_int + (st.t_succ0 if st.t_succ0 is not None else tc)
