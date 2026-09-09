from typing import Any, Dict


def snapshot(sim, sim_time_hint: float = 1.0) -> Dict[str, Any]:
    st = sim.stats
    tx = st.packets_generated
    rx = st.packets_delivered
    drop = st.packets_dropped

    pdr = (rx / tx) if tx else 0.0
    avg_delay = (sum(st.delays) / len(st.delays)) if st.delays else 0.0
    max_delay = max(st.delays) if st.delays else 0.0
    min_delay = min(st.delays) if st.delays else 0.0

    delivered_bits = rx * sim.config["app"]["packet_size_bytes"] * 8
    thr_bps = delivered_bits / max(1e-12, sim.engine.now if sim.engine.now > 0 else sim_time_hint)

    return {
        "time": sim.engine.now,
        "tx": tx,
        "rx": rx,
        "drop": drop,
        "pdr": pdr,
        "thr_bps": thr_bps,
        "lat_avg": avg_delay,
        "lat_max": max_delay,
        "lat_min": min_delay,
        "mac_tx_attempts": st.mac_tx_attempts,
        "mac_retries": st.mac_retries,
        "ack_to": st.mac_ack_timeouts,
        "collisions": st.phy_collisions,
        "per_drops": st.phy_per_drops,
        "raw_attempts": dict(st.raw_slot_tx_attempts),
        "raw_success": dict(st.raw_slot_data_success),
    }


def summarize(sim, sim_time: float, label: str) -> str:
    s = snapshot(sim, sim_time_hint=sim_time)
    lines = []
    lines.append(f"===== SUMMARY ({label}) =====")
    lines.append(f"sim_time_s: {sim_time:.3f}")
    lines.append(f"packets_generated: {s['tx']}")
    lines.append(f"packets_delivered: {s['rx']}")
    lines.append(f"packets_dropped:   {s['drop']}")
    lines.append(f"avg_delay_s:       {s['lat_avg']:.6f}")
    lines.append(f"throughput_bps:    {s['thr_bps']:.3f}")
    lines.append(f"mac_tx_attempts:   {s['mac_tx_attempts']}")
    lines.append(f"mac_retries:       {s['mac_retries']}")
    lines.append(f"ack_timeouts:      {s['ack_to']}")
    lines.append(f"phy_collisions:    {s['collisions']}")
    lines.append(f"phy_per_drops:     {s['per_drops']}")
    if sim.config["mac"]["raw_enable"]:
        lines.append(f"raw_slot_tx_attempts: {dict(sorted(s['raw_attempts'].items()))}")
        lines.append(f"raw_slot_data_success: {dict(sorted(s['raw_success'].items()))}")
    return "\n".join(lines)
