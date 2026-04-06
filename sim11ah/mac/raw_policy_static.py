from __future__ import annotations

from typing import Any, Dict, List, Optional

from sim11ah.mac.common import (
    MORSE_RAW_MIN_SLOT_DURATION_US,
    RawBeaconSpreading,
    RawConfig,
    RawPeriodic,
    RawSlotDefinition,
)


class StaticRawPolicy:
    """
    Baseline fixed RAW policy.

    Behavior:
    - supports explicit RAW configs from cfg["mac"]["raw_configs"]
    - otherwise builds classic fixed groups
    - keeps a static policy, but clamps groups to the actually known AID range
      when possible to avoid empty / invalid RAW allocations
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx = ctx
        self._log = log_fn
        self._base_configs: List[RawConfig] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_log(self, event: str, details: Dict[str, Any]) -> None:
        try:
            self._log(event, details)
        except Exception:
            pass

    def _known_max_aid(self, connected_aids: Optional[List[int]] = None) -> int:
        """
        Best-effort estimate of the maximum meaningful AID in the simulation.
        Preference:
        1. provided connected_aids
        2. ctx-known station/AID structures
        3. configured group span fallback
        """
        if connected_aids:
            try:
                vals = [int(a) for a in connected_aids if int(a) > 0]
                if vals:
                    return max(vals)
            except Exception:
                pass

        # Associated STA map is often {node_id: aid}, so use values.
        try:
            assoc = getattr(self.ctx, "_associated_stas", None)
            if isinstance(assoc, dict) and assoc:
                vals = [int(v) for v in assoc.values() if int(v) > 0]
                if vals:
                    return max(vals)
        except Exception:
            pass

        # These are more likely keyed by AID
        for attr in ("_sta_state", "_sta_queue_hint", "_sta_ps_buffer"):
            try:
                obj = getattr(self.ctx, attr, None)
                if isinstance(obj, dict) and obj:
                    vals = [int(k) for k in obj.keys() if int(k) > 0]
                    if vals:
                        return max(vals)
                if isinstance(obj, (list, set, tuple)) and obj:
                    vals = [int(v) for v in obj if int(v) > 0]
                    if vals:
                        return max(vals)
            except Exception:
                pass

        # Try simulator node ids if available (weak fallback only)
        try:
            sim = getattr(self.ctx, "sim", None)
            nodes = getattr(sim, "nodes", None)
            if isinstance(nodes, dict) and nodes:
                vals = [int(k) for k in nodes.keys() if int(k) > 0]
                if vals:
                    return max(vals)
        except Exception:
            pass

        # Fallback: configured static span
        try:
            return max(0, int(self.ctx.raw_num_groups) * int(self.ctx.raw_nodes_per_group))
        except Exception:
            return 0

    def _clamp_aid_range(
        self,
        start_aid: int,
        end_aid: int,
        max_aid: int,
    ) -> Optional[tuple[int, int]]:
        start_aid = int(start_aid)
        end_aid = int(end_aid)
        max_aid = int(max_aid)

        if start_aid <= 0:
            start_aid = 1
        if end_aid < start_aid:
            return None
        if max_aid > 0:
            if start_aid > max_aid:
                return None
            end_aid = min(end_aid, max_aid)
        if end_aid < start_aid:
            return None
        return start_aid, end_aid

    def _build_periodic(
        self,
        *,
        periodicity: int,
        validity: int,
        start_offset: int,
        refresh_praw: bool,
    ) -> RawPeriodic:
        return RawPeriodic(
            periodicity=int(periodicity),
            validity=int(validity),
            start_offset=int(start_offset),
            cur_validity=int(validity),
            cur_start_offset=int(start_offset),
            refresh_praw=bool(refresh_praw),
        )

    def _config_snapshot(self, rcfg: RawConfig) -> Dict[str, Any]:
        try:
            num_slots = int(rcfg.slot_definition.num_slots)
            slot_duration_us = int(rcfg.slot_definition.slot_duration_us)
            cross_slot_boundary = bool(rcfg.slot_definition.cross_slot_boundary)
        except Exception:
            num_slots = None
            slot_duration_us = None
            cross_slot_boundary = None

        return {
            "id": getattr(rcfg, "id", None),
            "raw_type": getattr(rcfg, "raw_type", None),
            "start_aid": getattr(rcfg, "start_aid", None),
            "end_aid": getattr(rcfg, "end_aid", None),
            "start_time_us": getattr(rcfg, "start_time_us", None),
            "num_slots": num_slots,
            "slot_duration_us": slot_duration_us,
            "cross_slot_boundary": cross_slot_boundary,
            "enabled": getattr(rcfg, "enabled", None),
            "dynamic_beacon_idx": getattr(rcfg, "dynamic_beacon_idx", None),
        }

    def _clone_config_with_clamped_aid(
        self,
        rcfg: RawConfig,
        max_aid: int,
    ) -> Optional[RawConfig]:
        clamped = self._clamp_aid_range(
            int(rcfg.start_aid),
            int(rcfg.end_aid),
            int(max_aid),
        )
        if clamped is None:
            return None

        start_aid, end_aid = clamped

        new_cfg = RawConfig(
            id=int(rcfg.id),
            raw_type=int(rcfg.raw_type),
            start_aid=start_aid,
            end_aid=end_aid,
            start_time_us=int(rcfg.start_time_us),
            slot_definition=RawSlotDefinition(
                num_slots=max(1, int(rcfg.slot_definition.num_slots)),
                slot_duration_us=max(
                    MORSE_RAW_MIN_SLOT_DURATION_US,
                    int(rcfg.slot_definition.slot_duration_us),
                ),
                cross_slot_boundary=bool(rcfg.slot_definition.cross_slot_boundary),
            ),
            beacon_spreading=RawBeaconSpreading(
                nominal_sta_per_beacon=int(rcfg.beacon_spreading.nominal_sta_per_beacon),
                max_spread=int(rcfg.beacon_spreading.max_spread),
                last_aid=int(getattr(rcfg.beacon_spreading, "last_aid", 0)),
            ),
            periodic=RawPeriodic(
                periodicity=int(rcfg.periodic.periodicity),
                validity=int(rcfg.periodic.validity),
                start_offset=int(rcfg.periodic.start_offset),
                cur_validity=int(rcfg.periodic.cur_validity),
                cur_start_offset=int(rcfg.periodic.cur_start_offset),
                refresh_praw=bool(rcfg.periodic.refresh_praw),
            ),
            enabled=bool(rcfg.enabled),
            dynamic_beacon_idx=getattr(rcfg, "dynamic_beacon_idx", None),
        )

        if new_cfg.enabled and new_cfg.is_valid():
            return new_cfg
        return None

    # ------------------------------------------------------------------
    # Build configs
    # ------------------------------------------------------------------
    def init_configs(self) -> List[RawConfig]:
        mac_cfg = self.ctx.cfg["mac"]
        max_aid = self._known_max_aid()

        cfg_list = mac_cfg.get("raw_configs", None)
        if isinstance(cfg_list, list) and cfg_list:
            built: List[RawConfig] = []

            for i, item in enumerate(cfg_list, start=1):
                try:
                    if not isinstance(item, dict):
                        self._safe_log(
                            "RAW_CONFIG_INVALID",
                            {"idx": i, "reason": "config_item_not_dict"},
                        )
                        continue

                    slot_duration_us = int(
                        item.get("slot_duration_us", int(self.ctx.raw_slot_duration * 1e6))
                    )
                    slot_duration_us = max(MORSE_RAW_MIN_SLOT_DURATION_US, slot_duration_us)

                    start_aid = int(item.get("start_aid", 1))
                    end_aid = int(item.get("end_aid", start_aid))

                    clamped = self._clamp_aid_range(start_aid, end_aid, max_aid)
                    if clamped is None:
                        self._safe_log(
                            "RAW_CONFIG_SKIPPED",
                            {
                                "idx": i,
                                "reason": "empty_or_out_of_range_aid_span",
                                "start_aid": start_aid,
                                "end_aid": end_aid,
                                "max_aid": max_aid,
                            },
                        )
                        continue
                    start_aid, end_aid = clamped

                    rcfg = RawConfig(
                        id=int(item.get("id", i)),
                        raw_type=int(item.get("raw_type", self.ctx.raw_type)),
                        start_aid=start_aid,
                        end_aid=end_aid,
                        start_time_us=int(item.get("start_time_us", 0)),
                        slot_definition=RawSlotDefinition(
                            num_slots=max(1, int(item.get("num_slots", self.ctx.raw_num_slots))),
                            slot_duration_us=slot_duration_us,
                            cross_slot_boundary=bool(
                                item.get("cross_slot_boundary", self.ctx.raw_cross_slot)
                            ),
                        ),
                        beacon_spreading=RawBeaconSpreading(
                            nominal_sta_per_beacon=int(item.get("nominal_sta_per_beacon", 0)),
                            max_spread=int(item.get("max_spread", 0)),
                            last_aid=0,
                        ),
                        periodic=self._build_periodic(
                            periodicity=int(item.get("periodicity", 0)),
                            validity=int(item.get("validity", 0)),
                            start_offset=int(item.get("start_offset", 0)),
                            refresh_praw=bool(item.get("refresh_praw", False)),
                        ),
                        enabled=bool(item.get("enabled", True)),
                        dynamic_beacon_idx=item.get("dynamic_beacon_idx", None),
                    )

                    if not rcfg.enabled:
                        self._safe_log(
                            "RAW_CONFIG_DISABLED",
                            {"idx": i, "id": rcfg.id},
                        )
                        continue

                    if not rcfg.is_valid():
                        self._safe_log(
                            "RAW_CONFIG_INVALID",
                            {"idx": i, **self._config_snapshot(rcfg)},
                        )
                        continue

                    built.append(rcfg)

                except Exception as e:
                    self._safe_log(
                        "RAW_CONFIG_PARSE_ERR",
                        {"idx": i, "err": str(e)},
                    )

            built = sorted(built, key=lambda x: x.id)

            if not built:
                self._safe_log(
                    "RAW_CONFIG_FALLBACK",
                    {"reason": "no_valid_custom_configs"},
                )
                built = self._build_default_configs(max_aid=max_aid)

            self._base_configs = list(built)
            return built

        built = self._build_default_configs(max_aid=max_aid)
        self._base_configs = list(built)
        return built

    def _build_default_configs(self, max_aid: int) -> List[RawConfig]:
        mac_cfg = self.ctx.cfg["mac"]
        built: List[RawConfig] = []

        try:
            raw_num_groups = max(0, int(self.ctx.raw_num_groups))
            raw_nodes_per_group = max(1, int(self.ctx.raw_nodes_per_group))
            raw_num_slots = max(1, int(self.ctx.raw_num_slots))
            raw_slot_duration_us = max(
                MORSE_RAW_MIN_SLOT_DURATION_US,
                int(float(self.ctx.raw_slot_duration) * 1e6),
            )
            raw_start_time_us = int(self.ctx.raw_start_time_us)
            raw_type = int(self.ctx.raw_type)
            raw_cross_slot = bool(self.ctx.raw_cross_slot)
        except Exception as e:
            self._safe_log("RAW_DEFAULT_BUILD_ERR", {"err": str(e)})
            return built

        periodicity = int(self.ctx.praw_period) if self.ctx.praw_enable else 0
        validity = int(self.ctx.praw_validity) if self.ctx.praw_enable else 0
        start_offset = int(self.ctx.praw_start_offset)
        refresh_praw = bool(self.ctx.praw_refresh)

        per_group_time_us = raw_num_slots * raw_slot_duration_us

        for g in range(raw_num_groups):
            start_aid = 1 + g * raw_nodes_per_group
            end_aid = start_aid + raw_nodes_per_group - 1

            clamped = self._clamp_aid_range(start_aid, end_aid, max_aid)
            if clamped is None:
                self._safe_log(
                    "RAW_GROUP_SKIPPED",
                    {
                        "group_id": g + 1,
                        "reason": "empty_or_out_of_range_aid_span",
                        "start_aid": start_aid,
                        "end_aid": end_aid,
                        "max_aid": max_aid,
                    },
                )
                continue

            start_aid, end_aid = clamped

            rcfg = RawConfig(
                id=g + 1,
                raw_type=raw_type,
                start_aid=start_aid,
                end_aid=end_aid,
                start_time_us=raw_start_time_us + g * per_group_time_us,
                slot_definition=RawSlotDefinition(
                    num_slots=raw_num_slots,
                    slot_duration_us=raw_slot_duration_us,
                    cross_slot_boundary=raw_cross_slot,
                ),
                beacon_spreading=RawBeaconSpreading(
                    nominal_sta_per_beacon=int(mac_cfg.get("nominal_sta_per_beacon", 0)),
                    max_spread=int(mac_cfg.get("max_spread", 0)),
                    last_aid=0,
                ),
                periodic=self._build_periodic(
                    periodicity=periodicity,
                    validity=validity,
                    start_offset=start_offset,
                    refresh_praw=refresh_praw,
                ),
                enabled=True,
            )

            if not rcfg.is_valid():
                self._safe_log(
                    "RAW_GROUP_INVALID",
                    {"group_id": g + 1, **self._config_snapshot(rcfg)},
                )
                continue

            built.append(rcfg)

        return built

    # ------------------------------------------------------------------
    # Dynamic build hook
    # ------------------------------------------------------------------
    def build_dynamic_configs(self, connected_aids: List[int]) -> List[RawConfig]:
        """
        Static policy: preserve the original configured RAW structure, but clamp
        returned groups to the currently connected AID range when possible.

        This keeps the policy static while avoiding obviously empty groups.
        """
        base = list(self._base_configs)
        if not base:
            return []

        max_aid = self._known_max_aid(connected_aids=connected_aids)
        out: List[RawConfig] = []

        for rcfg in base:
            try:
                new_cfg = self._clone_config_with_clamped_aid(rcfg, max_aid)
                if new_cfg is not None:
                    out.append(new_cfg)
            except Exception as e:
                self._safe_log(
                    "RAW_DYNAMIC_CONFIG_ERR",
                    {
                        "id": getattr(rcfg, "id", None),
                        "err": str(e),
                    },
                )

        return sorted(out, key=lambda x: x.id)

    # ------------------------------------------------------------------
    # Debug snapshot
    # ------------------------------------------------------------------
    def get_policy_snapshot(self) -> Dict[str, Any]:
        cfgs = list(getattr(self.ctx, "_raw_configs", []))
        return {
            "policy": "static",
            "num_configs": len(cfgs),
            "configs": [self._config_snapshot(c) for c in cfgs],
        }