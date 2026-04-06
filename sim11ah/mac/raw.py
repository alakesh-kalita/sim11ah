from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from sim11ah.constants import MacState
from sim11ah.models import MacFrame

from sim11ah.mac.common import (
    MORSE_RAW_MIN_SLOT_DURATION_US,
    RawConfig,
    RawType,
    cslot_to_us,
    us_to_cslot,
)
from sim11ah.mac.context import MacContext
from sim11ah.mac.raw_policy_static import StaticRawPolicy
from sim11ah.mac.raw_policy_adaptive import AdaptiveRawPolicy


class RawEngine:
    def __init__(
        self,
        ctx: MacContext,
        log_fn,
        on_raw_enter_cb: Optional[Callable[[], None]] = None,
    ) -> None:
        self.ctx = ctx
        self._log = log_fn
        self._on_raw_enter_cb = on_raw_enter_cb
        self.policy = self._build_policy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _inc_stat(self, key: str, amount: int = 1) -> None:
        try:
            if key not in self.ctx._stats:
                self.ctx._stats[key] = 0
            self.ctx._stats[key] += int(amount)
        except Exception:
            pass

    def _set_stat_if_missing(self, key: str, value: int = 0) -> None:
        try:
            if key not in self.ctx._stats:
                self.ctx._stats[key] = int(value)
        except Exception:
            pass

    def _ensure_raw_stats(self) -> None:
        for key in (
            "raw_groups_built",
            "raw_configs_built",
            "raw_not_scheduled",
            "raw_scheduled",
            "raw_enter_count",
            "raw_exit_count",
            "raw_fit_blocked",
            "raw_fit_pass",
            "raw_fit_fail",
            "raw_backoff_saved",
            "raw_backoff_restored",
            "raw_sleep_transitions",
            "raw_invalid_slot_window",
        ):
            self._set_stat_if_missing(key, 0)

    def _schedule_raw_enter(self, gen: int) -> None:
        self.raw_enter(gen)

    def _schedule_raw_exit(self, gen: int) -> None:
        self.raw_exit(gen)

    def _set_current_group(self, group: Optional[Dict[str, Any]]) -> None:
        try:
            self.ctx._raw_current_group = group
        except Exception:
            pass

    def _get_current_group(self) -> Optional[Dict[str, Any]]:
        try:
            group = getattr(self.ctx, "_raw_current_group", None)
            if isinstance(group, dict):
                return group
        except Exception:
            pass
        return None

    def _get_self_aid(self) -> int:
        """
        Return the station's associated AID if available.
        Fallback to node_id only when no explicit AID field exists.
        """
        if self.ctx.node.node_id == 0:
            return 0

        candidate_names = (
            "aid",
            "_aid",
            "associated_aid",
            "_associated_aid",
            "assoc_aid",
            "_assoc_aid",
        )

        for name in candidate_names:
            try:
                v = getattr(self.ctx, name, None)
                if isinstance(v, int) and v > 0:
                    return v
            except Exception:
                pass

        try:
            v = getattr(self.ctx.node, "aid", None)
            if isinstance(v, int) and v > 0:
                return v
        except Exception:
            pass

        try:
            assoc = getattr(self.ctx, "_associated_stas", None)
            if isinstance(assoc, dict):
                v = assoc.get(self.ctx.node.node_id)
                if isinstance(v, int) and v > 0:
                    return v
        except Exception:
            pass

        return int(self.ctx.node.node_id)

    def _get_aid_position_in_group(self, aid: int, start_aid: int, end_aid: int) -> int:
        """
        Map a STA to a stable relative position within the matched AID range.

        This is better than using (aid - start_aid) directly because real/simulated
        AIDs may be sparse or irregular.
        """
        try:
            all_aids = self.connected_aids()
            group_aids = [x for x in all_aids if start_aid <= x <= end_aid]
            if aid in group_aids:
                return group_aids.index(aid)
        except Exception:
            pass

        return max(0, int(aid) - int(start_aid))

    # ------------------------------------------------------------------
    # Policy selection
    # ------------------------------------------------------------------
    def _build_policy(self):
        mac_cfg = self.ctx.cfg["mac"]
        mode = str(mac_cfg.get("raw_policy", "static")).strip().lower()

        if mode == "adaptive":
            self._log("RAW_POLICY_SELECT", {"policy": "adaptive"})
            return AdaptiveRawPolicy(self.ctx, self._log)

        self._log("RAW_POLICY_SELECT", {"policy": "static"})
        return StaticRawPolicy(self.ctx, self._log)

    # ------------------------------------------------------------------
    # Public init hook
    # ------------------------------------------------------------------
    def init_from_cfg(self) -> None:
        self._ensure_raw_stats()

        if not self.ctx.raw_enable:
            self._log("RAW_DISABLED", {})
            return

        if hasattr(self.policy, "init_configs"):
            self.ctx._raw_configs = self.policy.init_configs()
        else:
            self.ctx._raw_configs = []

        self._set_current_group(None)

        self._log(
            "RAW_INIT",
            {
                "raw_enable": bool(self.ctx.raw_enable),
                "num_configs": len(self.ctx._raw_configs),
                "policy": str(self.ctx.cfg["mac"].get("raw_policy", "static")),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def connected_aids(self) -> List[int]:
        if self.ctx.node.node_id != 0:
            return []

        # 1) Preferred: AP association table
        try:
            assoc = getattr(self.ctx, "_associated_stas", None)
            if isinstance(assoc, dict) and assoc:
                aids = sorted(
                    int(aid) for aid in assoc.values()
                    if isinstance(aid, int) and aid > 0
                )
                if aids:
                    return aids
        except Exception:
            pass

        # 2) Fallback: node/mac context AID fields
        try:
            aids: List[int] = []
            for nid, node in getattr(self.ctx.sim, "nodes", {}).items():
                if int(nid) == 0:
                    continue

                found = None

                try:
                    mac_ctx = getattr(getattr(node, "mac", None), "ctx", None)
                    if mac_ctx is not None:
                        for name in (
                            "aid",
                            "_aid",
                            "associated_aid",
                            "_associated_aid",
                            "assoc_aid",
                            "_assoc_aid",
                        ):
                            v = getattr(mac_ctx, name, None)
                            if isinstance(v, int) and v > 0:
                                found = int(v)
                                break
                except Exception:
                    pass

                if found is None:
                    try:
                        for name in (
                            "aid",
                            "_aid",
                            "associated_aid",
                            "_associated_aid",
                            "assoc_aid",
                            "_assoc_aid",
                        ):
                            v = getattr(node, name, None)
                            if isinstance(v, int) and v > 0:
                                found = int(v)
                                break
                    except Exception:
                        pass

                if found is not None:
                    aids.append(found)

            aids = sorted(set(aids))
            if aids:
                return aids
        except Exception:
            pass

        # 3) Final fallback: simulator STA node IDs
        try:
            return sorted(int(nid) for nid in self.ctx.sim.nodes.keys() if int(nid) > 0)
        except Exception:
            return []

    def refresh_dynamic_configs(self) -> None:
        if not self.ctx.raw_enable or self.ctx.node.node_id != 0:
            return

        aids = self.connected_aids()

        self._log(
            "RAW_CONNECTED_AIDS",
            {
                "count": len(aids),
                "aids_sample": aids[:16],
            },
        )

        # Preferred interface: policy returns RawConfig objects
        if hasattr(self.policy, "build_dynamic_configs"):
            dynamic = self.policy.build_dynamic_configs(aids)

            # Important: allow empty list to replace stale configs.
            if dynamic is not None:
                self.ctx._raw_configs = dynamic
                self._log(
                    "RAW_DYNAMIC_CONFIGS_REFRESH",
                    {
                        "aids": len(aids),
                        "num_configs": len(dynamic),
                    },
                )
                self._log(
                    "RAW_CONFIG_SPAN",
                    {
                        "configs": [
                            {
                                "id": cfg.id,
                                "start_aid": cfg.start_aid,
                                "end_aid": cfg.end_aid,
                                "num_slots": cfg.slot_definition.num_slots,
                                "slot_us": cfg.slot_definition.slot_duration_us,
                                "enabled": cfg.enabled,
                            }
                            for cfg in self.ctx._raw_configs[:16]
                        ]
                    },
                )
            else:
                self._log(
                    "RAW_DYNAMIC_CONFIGS_REFRESH",
                    {
                        "aids": len(aids),
                        "num_configs": None,
                    },
                )
            return

        # Legacy/alternative interface: policy returns ready-made RPS entries.
        return

    def update_aid_indices(self) -> None:
        if self.ctx.node.node_id != 0:
            return

        aid_list = self.connected_aids()
        for cfg in self.ctx._raw_configs:
            cfg.start_aid_idx = -1
            cfg.end_aid_idx = -1

            if not aid_list:
                continue

            for i, aid in enumerate(aid_list):
                if aid >= cfg.start_aid:
                    cfg.start_aid_idx = i
                    break

            for i in range(len(aid_list) - 1, -1, -1):
                if aid_list[i] <= cfg.end_aid:
                    cfg.end_aid_idx = i
                    break

            if cfg.start_aid_idx < 0 or cfg.end_aid_idx < cfg.start_aid_idx:
                cfg.start_aid_idx = -1
                cfg.end_aid_idx = -1

    def generate_slot_definition_info(self, cfg: RawConfig) -> Dict[str, Any]:
        """
        Compatible with both static and adaptive policies.

        By default, preserve the configured num_slots because this is often
        preferable in simulation. If you want the old clamped behavior, set:

            mac["raw_preserve_num_slots"] = False
        """
        cslot = max(0, us_to_cslot(cfg.slot_definition.slot_duration_us))
        cross = bool(cfg.slot_definition.cross_slot_boundary)

        if cfg.slot_definition.slot_duration_us < MORSE_RAW_MIN_SLOT_DURATION_US:
            cslot = us_to_cslot(MORSE_RAW_MIN_SLOT_DURATION_US)

        if cfg.raw_type == RawType.SOUNDING:
            cslot_max = (1 << 8) - 1
            max_slots = 6
            fmt = 0
        else:
            if cslot > 255:
                fmt = 1
                cslot_max = (1 << 11) - 1
                max_slots = 3
            else:
                fmt = 0
                cslot_max = (1 << 8) - 1
                max_slots = 6

        preserve_slots = bool(self.ctx.cfg["mac"].get("raw_preserve_num_slots", True))

        if preserve_slots:
            num_slots = max(1, int(cfg.slot_definition.num_slots))
        else:
            num_slots = min(int(cfg.slot_definition.num_slots), max_slots)

        cslot = min(cslot, cslot_max)

        return {
            "format": fmt,
            "cslot": int(cslot),
            "slot_duration_us": int(cslot_to_us(cslot)),
            "num_slots": int(num_slots),
            "cross_slot_boundary": cross,
        }

    def cfg_active_this_beacon(self, cfg: RawConfig) -> bool:
        if not cfg.enabled or not cfg.is_valid():
            return False

        if cfg.has_dynamic_bcn_idx():
            return int(cfg.dynamic_beacon_idx or 0) == int(self.ctx._ap_beacon_count)

        if cfg.is_periodic():
            if cfg.periodic.cur_validity <= 0:
                if cfg.periodic.refresh_praw:
                    cfg.periodic.cur_validity = cfg.periodic.validity
                    cfg.periodic.cur_start_offset = cfg.periodic.start_offset
                else:
                    return False

            if cfg.periodic.cur_start_offset > 0:
                return False

        return True

    def update_periodic_after_beacon(self) -> None:
        for cfg in self.ctx._raw_configs:
            if not cfg.is_periodic():
                continue

            if cfg.periodic.cur_start_offset == 0:
                cfg.periodic.cur_start_offset = max(0, cfg.periodic.periodicity - 1)
            else:
                cfg.periodic.cur_start_offset -= 1

            if cfg.periodic.cur_start_offset == cfg.periodic.start_offset:
                cfg.periodic.cur_validity -= 1
                if cfg.periodic.cur_validity <= 0 and cfg.periodic.refresh_praw:
                    cfg.periodic.cur_validity = cfg.periodic.validity
                    cfg.periodic.cur_start_offset = cfg.periodic.start_offset

    def select_aid_range_for_cfg(self, cfg: RawConfig) -> Tuple[int, int]:
        return cfg.start_aid, cfg.end_aid

    # ------------------------------------------------------------------
    # RPS build/apply
    # ------------------------------------------------------------------
    def build_rps(self) -> List[Dict[str, Any]]:
        self._ensure_raw_stats()

        rps: List[Dict[str, Any]] = []
        self.ctx._stats["raw_groups_built"] = 0
        self.ctx._stats["raw_configs_built"] = 0

        if not self.ctx.raw_enable:
            return rps

        if self.ctx.node.node_id == 0:
            # If the policy outputs RawConfig objects, refresh them first.
            self.refresh_dynamic_configs()
            self.update_aid_indices()

            # If the policy directly builds RPS entries, use that path.
            if hasattr(self.policy, "build_rps") and not hasattr(self.policy, "build_dynamic_configs"):
                aids = self.connected_aids()
                rps = self.policy.build_rps(aids)
                self.ctx._stats["raw_groups_built"] = len(rps)
                self.ctx._stats["raw_configs_built"] = len(rps)

                policy_name = getattr(self.policy, "group_mode", "adaptive")
                self._log(
                    "RAW_RPS_BUILD",
                    {
                        "groups": len(rps),
                        "policy": policy_name,
                        "aids": len(aids),
                    },
                )
                return rps

        for cfg in self.ctx._raw_configs:
            if not self.cfg_active_this_beacon(cfg):
                continue

            slot_info = self.generate_slot_definition_info(cfg)
            start_aid, end_aid = self.select_aid_range_for_cfg(cfg)

            entry: Dict[str, Any] = {
                "config_id": cfg.id,
                "group_id": len(rps),
                "raw_type": cfg.raw_type,
                "start_aid": start_aid,
                "end_aid": end_aid,
                "num_slots": slot_info["num_slots"],
                "slot_duration": slot_info["slot_duration_us"] / 1e6,
                "slot_duration_us": slot_info["slot_duration_us"],
                "start_offset": cfg.start_time_us / 1e6,
                "start_time_us": cfg.start_time_us,
                "cross_slot": slot_info["cross_slot_boundary"],
                "slot_format": slot_info["format"],
                "cslot": slot_info["cslot"],
                "praw": cfg.is_periodic(),
                "praw_periodicity": cfg.periodic.periodicity if cfg.is_periodic() else 0,
                "praw_validity": cfg.periodic.cur_validity if cfg.is_periodic() else 0,
                "praw_start_offset": cfg.periodic.cur_start_offset if cfg.is_periodic() else 0,
            }

            if cfg.raw_type == RawType.TRIGGERING:
                entry["triggered_aids"] = list(self.ctx.cfg["mac"].get("triggered_aids", []))

            rps.append(entry)
            self.ctx._stats["raw_groups_built"] += 1
            self.ctx._stats["raw_configs_built"] += 1

        snap = {}
        if hasattr(self.policy, "get_policy_snapshot"):
            snap = self.policy.get_policy_snapshot() or {}

        self._log(
            "RAW_RPS_BUILD",
            {
                "groups": len(rps),
                "policy": snap.get("policy", self.ctx.cfg["mac"].get("raw_policy", "static")),
                "configs_considered": len(self.ctx._raw_configs),
            },
        )
        return rps

    def apply_rps(self, rps: List[Dict[str, Any]], raw_guard: float) -> None:
        self._ensure_raw_stats()

        if self.ctx._metrics is not None and self.ctx.node.node_id == 0:
            self.ctx._metrics.register_raw_slots(
                rps=rps,
                beacon_rx_t=self.ctx.sim.engine.now,
                raw_guard=raw_guard,
            )

        aid = self._get_self_aid()
        chosen: Optional[Tuple[int, float, float, int, Dict[str, Any]]] = None

        self.ctx._raw_event_gen += 1
        gen = self.ctx._raw_event_gen

        for group in rps:
            start_aid = int(group["start_aid"])
            end_aid = int(group["end_aid"])
            if aid < start_aid or aid > end_aid:
                continue

            num_slots = max(1, int(group["num_slots"]))
            slot_dur = float(group["slot_duration"])
            start_off = float(group["start_offset"])

            rel_idx = self._get_aid_position_in_group(aid, start_aid, end_aid)
            slot_idx = rel_idx % num_slots

            enter_t = self.ctx.sim.engine.now + start_off + slot_idx * slot_dur + raw_guard
            exit_t = self.ctx.sim.engine.now + start_off + (slot_idx + 1) * slot_dur - raw_guard

            if exit_t <= enter_t:
                self._inc_stat("raw_invalid_slot_window")
                self._log(
                    "RAW_INVALID_SLOT_WINDOW",
                    {
                        "aid": aid,
                        "enter": enter_t,
                        "exit": exit_t,
                        "slot_idx": slot_idx,
                        "slot_duration": slot_dur,
                        "raw_guard": raw_guard,
                        "group_id": int(group.get("group_id", -1)),
                        "config_id": group.get("config_id"),
                    },
                )
                continue

            candidate = (
                slot_idx,
                enter_t,
                exit_t,
                gen,
                group,
            )

            if chosen is None or enter_t < chosen[1]:
                chosen = candidate

        if chosen is None:
            self.ctx.raw_assigned_slot = None
            self.ctx.raw_allowed = False
            self.ctx._raw_slot_enter_t = 0.0
            self.ctx._raw_slot_exit_t = 0.0
            self._set_current_group(None)

            if self.ctx.state != MacState.WAIT_ACK:
                self.ctx.state = MacState.RAW_SLEEP
                self._inc_stat("raw_sleep_transitions")

            self.ctx._stats["raw_not_scheduled"] += 1
            self._log(
                "RAW_NOT_SCHEDULED",
                {
                    "aid": aid,
                    "node_id": self.ctx.node.node_id,
                    "rps_groups": len(rps),
                    "state": str(self.ctx.state),
                },
            )
            return

        slot_idx, enter_t, exit_t, ev_gen, group = chosen
        self.ctx.raw_assigned_slot = int(slot_idx)
        self.ctx._raw_slot_enter_t = float(enter_t)
        self.ctx._raw_slot_exit_t = float(exit_t)
        self._set_current_group(group)

        self.ctx.sim.engine.schedule(
            enter_t,
            self._schedule_raw_enter,
            ev_gen,
            name="RAW_ENTER_EVENT",
        )
        self.ctx.sim.engine.schedule(
            exit_t,
            self._schedule_raw_exit,
            ev_gen,
            name="RAW_EXIT_EVENT",
        )

        self.ctx.raw_allowed = False
        if self.ctx.state != MacState.WAIT_ACK:
            self.ctx.state = MacState.RAW_SLEEP
            self._inc_stat("raw_sleep_transitions")

        self._inc_stat("raw_scheduled")
        self._log(
            "RAW_SCHEDULED",
            {
                "aid": aid,
                "node_id": self.ctx.node.node_id,
                "slot": slot_idx,
                "enter": enter_t,
                "exit": exit_t,
                "duration": max(0.0, exit_t - enter_t),
                "group_id": int(group.get("group_id", -1)),
                "config_id": group.get("config_id"),
                "num_slots": int(group.get("num_slots", 1)),
                "slot_duration": float(group.get("slot_duration", 0.0)),
                "cross_slot": bool(group.get("cross_slot", False)),
            },
        )

    def raw_enter(self, gen: int) -> None:
        if gen != self.ctx._raw_event_gen:
            self._log(
                "RAW_ENTER_STALE",
                {
                    "gen": gen,
                    "current_gen": self.ctx._raw_event_gen,
                },
            )
            return

        if self.ctx.node.node_id == 0:
            self._log(
                "RAW_ENTER_SKIPPED",
                {
                    "reason": "ap",
                    "dozing": bool(getattr(self.ctx, "_dozing", False)),
                },
            )
            return

        if getattr(self.ctx, "_dozing", False):
            try:
                self.ctx._dozing = False
            except Exception:
                pass
            self._log(
                "RAW_WAKE_FOR_SLOT",
                {
                    "slot": self.ctx.raw_assigned_slot,
                    "enter_t": self.ctx._raw_slot_enter_t,
                    "exit_t": self.ctx._raw_slot_exit_t,
                },
            )

        self.ctx.raw_allowed = True
        self._inc_stat("raw_enter_count")

        saved_backoff_before = self.ctx._saved_backoff
        restored = False

        if self.ctx.state != MacState.WAIT_ACK:
            self.ctx.state = MacState.IDLE

        if self.ctx._saved_backoff is not None and self.ctx._backoff_slots_left is None:
            self.ctx._backoff_slots_left = self.ctx._saved_backoff
            self.ctx._saved_backoff = None
            restored = True
            self._inc_stat("raw_backoff_restored")

        self._log(
            "RAW_ENTER",
            {
                "slot": self.ctx.raw_assigned_slot,
                "enter_t": self.ctx._raw_slot_enter_t,
                "exit_t": self.ctx._raw_slot_exit_t,
                "slot_remaining": max(0.0, self.ctx._raw_slot_exit_t - self.ctx.sim.engine.now),
                "saved_backoff_before": saved_backoff_before,
                "backoff_restored": restored,
                "backoff_slots_left": self.ctx._backoff_slots_left,
                "state": str(self.ctx.state),
                "cross_slot": bool((self._get_current_group() or {}).get("cross_slot", False)),
            },
        )

        if self._on_raw_enter_cb is not None:
            self._on_raw_enter_cb()

    def raw_exit(self, gen: int) -> None:
        if gen != self.ctx._raw_event_gen:
            self._log(
                "RAW_EXIT_STALE",
                {
                    "gen": gen,
                    "current_gen": self.ctx._raw_event_gen,
                },
            )
            return

        if self.ctx.node.node_id == 0:
            return

        saved = False
        if self.ctx.state == MacState.BACKOFF and self.ctx._backoff_slots_left is not None:
            self.ctx._saved_backoff = self.ctx._backoff_slots_left
            self.ctx._backoff_slots_left = None
            saved = True
            self._inc_stat("raw_backoff_saved")

        self.ctx.raw_allowed = False
        self._inc_stat("raw_exit_count")

        self._log(
            "RAW_EXIT",
            {
                "slot": self.ctx.raw_assigned_slot,
                "saved_backoff": self.ctx._saved_backoff,
                "backoff_saved_now": saved,
                "state_before_sleep": str(self.ctx.state),
                "time_in_slot": max(0.0, self.ctx.sim.engine.now - self.ctx._raw_slot_enter_t),
                "cross_slot": bool((self._get_current_group() or {}).get("cross_slot", False)),
            },
        )

        if self.ctx.state != MacState.WAIT_ACK:
            self.ctx.state = MacState.RAW_SLEEP
            self._inc_stat("raw_sleep_transitions")

        self._set_current_group(None)

    # ------------------------------------------------------------------
    # Fit check
    # ------------------------------------------------------------------
    def exchange_would_fit_in_raw(
        self,
        frame: MacFrame,
        *,
        compute_data_tx_time,
        compute_ack_tx_time,
        get_prop,
    ) -> bool:
        self._ensure_raw_stats()

        if not self.ctx.raw_enable or self.ctx.node.node_id == 0:
            self._inc_stat("raw_fit_pass")
            return True

        if not self.ctx.raw_allowed:
            self._inc_stat("raw_fit_blocked")
            self._log(
                "RAW_FIT_CHECK",
                {
                    "result": False,
                    "reason": "raw_not_allowed",
                    "slot": self.ctx.raw_assigned_slot,
                    "now": self.ctx.sim.engine.now,
                },
            )
            return False

        if self.ctx._raw_slot_exit_t <= 0.0:
            self._inc_stat("raw_fit_pass")
            return True

        current_group = self._get_current_group() or {}
        cross_slot = bool(current_group.get("cross_slot", False))

        now = float(self.ctx.sim.engine.now)
        slot_exit = float(self.ctx._raw_slot_exit_t)
        remaining = slot_exit - now

        t_data = float(compute_data_tx_time(frame))
        mcs = (
            frame.ctrl.get("mcs", self.ctx.node.phy.default_mode)
            if frame.ctrl else self.ctx.node.phy.default_mode
        )
        t_ack = float(compute_ack_tx_time(frame.dst, mcs))
        prop = float(get_prop(frame.dst))

        full_budget = t_data + prop + self.ctx.sifs + t_ack + prop + self.ctx.ack_guard
        eps = max(1e-6, float(getattr(self.ctx, "ack_guard", 0.0)) * 0.25)

        if cross_slot:
            needed = t_data
            ok = needed <= (remaining + eps)
            reason = "cross_slot_data_only"
        else:
            needed = full_budget
            ok = needed <= (remaining + eps)
            reason = "full_exchange_with_tolerance"

        if ok:
            self._inc_stat("raw_fit_pass")
        else:
            self._inc_stat("raw_fit_fail")

        self._log(
            "RAW_FIT_CHECK",
            {
                "result": bool(ok),
                "reason": reason,
                "slot": self.ctx.raw_assigned_slot,
                "now": now,
                "slot_exit_t": slot_exit,
                "remaining": remaining,
                "needed": needed,
                "budget": full_budget,
                "t_data": t_data,
                "t_ack": t_ack,
                "prop": prop,
                "sifs": self.ctx.sifs,
                "ack_guard": self.ctx.ack_guard,
                "dst": frame.dst,
                "mcs": mcs,
                "frame_seq": getattr(frame, "frame_seq", None),
                "cross_slot": cross_slot,
                "eps": eps,
            },
        )

        return ok