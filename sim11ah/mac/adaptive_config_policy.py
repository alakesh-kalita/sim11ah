import math
from typing import Any, Dict


def _quantize_up_us(x_us: float, step_us: int) -> int:
    step_us = max(1, int(step_us))
    return int(math.ceil(x_us / step_us) * step_us)


def _quantize_down_us(x_us: float, step_us: int) -> int:
    step_us = max(1, int(step_us))
    return int(math.floor(x_us / step_us) * step_us)


def compute_dynamic_slot_range(cfg: Dict[str, Any]) -> Dict[str, int]:
    mac = cfg["mac"]
    phy = cfg["phy"]
    net = cfg["net"]
    app = cfg["app"]

    step_us = int(mac.get("adaptive_raw_step_us", 500))
    raw_budget_fraction = float(mac.get("adaptive_raw_budget_fraction", 0.90))
    absolute_min_us = int(mac.get("adaptive_raw_absolute_min_slot_us", 7000))
    strict_feasibility = bool(mac.get("adaptive_raw_strict_feasibility", False))

    raw_num_groups = max(1, int(mac.get("raw_num_groups", 1)))
    raw_num_slots = max(1, int(mac.get("raw_num_slots", 1)))
    beacon_interval_s = float(mac.get("beacon_interval", 0.5))
    total_raw_slots = raw_num_groups * raw_num_slots

    payload_bytes = int(app.get("packet_size_bytes", 128))
    data_mac_oh = int(mac.get("data_mac_overhead_bytes", 36))
    net_oh = int(net.get("net_header_bytes", 16))
    ack_size = int(mac.get("ack_size_bytes", 14))

    preamble_s = float(phy.get("preamble_time", 320e-6))
    header_s = float(phy.get("header_time", 80e-6))
    sifs_s = float(mac.get("sifs", 160e-6))
    difs_s = float(mac.get("difs", 264e-6))
    raw_guard_s = float(mac.get("raw_guard", 100e-6))
    ack_guard_s = float(mac.get("ack_guard", 100e-6))
    sigma_s = float(mac.get("slot_time", 52e-6))
    cw_min = float(mac.get("cw_min", 15))

    mode_table = phy.get("mode_table", {}) or {}
    data_mode = str(phy.get("default_mode", "MCS0"))
    ctrl_mode = str(phy.get("control_mode", "MCS0"))

    data_rate_bps = float(mode_table.get(data_mode, 150_000.0))
    ctrl_rate_bps = float(mode_table.get(ctrl_mode, 150_000.0))

    data_bits = 8.0 * (payload_bytes + data_mac_oh + net_oh)
    ack_bits = 8.0 * ack_size

    t_data_s = preamble_s + header_s + (data_bits / max(1.0, data_rate_bps))
    t_ack_s = preamble_s + header_s + (ack_bits / max(1.0, ctrl_rate_bps))
    t_success_s = t_data_s + sifs_s + t_ack_s + difs_s

    avg_backoff_s = (cw_min / 2.0) * sigma_s

    safe_min_s = t_success_s + raw_guard_s + ack_guard_s + avg_backoff_s
    safe_min_us = int(math.ceil(safe_min_s * 1e6))
    safe_min_us = max(safe_min_us, absolute_min_us)
    safe_min_us = _quantize_up_us(safe_min_us, step_us)

    budget_s = raw_budget_fraction * beacon_interval_s
    max_slot_budget_s = budget_s / max(1, total_raw_slots)
    max_slot_budget_us = int(math.floor(max_slot_budget_s * 1e6))
    max_slot_budget_us = _quantize_down_us(max_slot_budget_us, step_us)

    # Extra diagnostics
    required_budget_s = total_raw_slots * (safe_min_us / 1e6)
    required_beacon_interval_s = required_budget_s / max(raw_budget_fraction, 1e-12)
    max_total_slots_feasible = int(math.floor(budget_s / max(safe_min_us / 1e6, 1e-12)))

    if max_slot_budget_us < safe_min_us:
        feasible = 0

        if strict_feasibility:
            raise ValueError(
                "Infeasible adaptive RAW slot-duration configuration: "
                f"safe_min_slot_us={safe_min_us}, "
                f"budget_max_slot_us={max_slot_budget_us}, "
                f"raw_num_groups={raw_num_groups}, "
                f"raw_num_slots={raw_num_slots}, "
                f"beacon_interval_s={beacon_interval_s:.6f}, "
                f"required_beacon_interval_s≈{required_beacon_interval_s:.6f}"
            )

        # Fallback: collapse range to safe minimum
        initial_us = safe_min_us
        max_us = safe_min_us
    else:
        feasible = 1
        initial_us = int(round((safe_min_us + max_slot_budget_us) / 2.0))
        initial_us = _quantize_up_us(initial_us, step_us)
        initial_us = min(initial_us, max_slot_budget_us)
        max_us = max_slot_budget_us

    return {
        "adaptive_raw_initial_slot_us": int(initial_us),
        "adaptive_raw_max_slot_us": int(max_us),
        "adaptive_raw_safe_min_slot_us": int(safe_min_us),
        "adaptive_raw_budget_max_slot_us": int(max_slot_budget_us),
        "adaptive_raw_total_slots_per_beacon": int(total_raw_slots),
        "adaptive_raw_range_feasible": int(feasible),
        "adaptive_raw_required_beacon_interval_us": int(round(required_beacon_interval_s * 1e6)),
        "adaptive_raw_max_total_slots_feasible": int(max_total_slots_feasible),
    }