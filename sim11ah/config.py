from typing import Any, Dict

from sim11ah.mac.adaptive_config_policy import compute_dynamic_slot_range


def default_config(raw_enable: bool, traffic_mode: str) -> Dict[str, Any]:
    allowed_traffic = {"periodic", "poisson", "cbr", "burst", "bursty", "onoff", "video"}
    if traffic_mode not in allowed_traffic:
        raise ValueError(
            f"Unsupported traffic_mode={traffic_mode!r}. "
            f"Allowed: {sorted(allowed_traffic)}"
        )

    # ------------------------------------------------------------------
    # IEEE 802.11ah / S1G PHY rates for 1 MHz bandwidth, 1 spatial stream,
    # normal guard interval (IEEE 802.11ah-2016 Table 23-53).
    #
    # Corrected 1 MHz S1G rates:
    #   MCS0 = 150 kb/s  (BPSK 1/2)
    #   MCS1 = 300 kb/s  (QPSK 1/2)
    #   MCS2 = 450 kb/s  (QPSK 3/4)
    #   MCS3 = 600 kb/s  (16-QAM 1/2)
    # ------------------------------------------------------------------
    phy_mode_table = {
        "MCS0": 150_000,     # 150 kb/s, 1 MHz, BPSK 1/2
        "MCS1": 300_000,     # 300 kb/s, 1 MHz, QPSK 1/2
        "MCS2": 450_000,     # 450 kb/s, 1 MHz, QPSK 3/4
        "MCS3": 600_000,     # 600 kb/s, 1 MHz, 16-QAM 1/2
        # "MCS10": 150_000,  # optional MCS10 repeated/duplicated mode
    }

    cfg = {
        "phy": {
            # ------------------------------------------------------------------
            # PHY timing model
            # ------------------------------------------------------------------
            "preamble_time": 320e-6,   # seconds; simplified S1G PHY preamble model
            "header_time": 80e-6,      # seconds; simplified PHY header model

            # IEEE 802.11ah S1G slot time.
            "slot_time": 52e-6,        # seconds; 52 us, standard-aligned

            # PHY rate table for 1 MHz S1G.
            "mode_table": phy_mode_table,
            "default_mode": "MCS0",    # simulator default unicast PHY mode
            "control_mode": "MCS0",    # simulator default control-frame PHY mode

            # Collision handling is simulator-specific.
            "collision_model": "capture",  # "capture" or "pessimistic"

            # CCA threshold and sensitivity are implementation-dependent.
            "cca_threshold_dbm": -105.0,   # dBm; simulator assumption
            "rx_sensitivity_dbm": -105.0,  # dBm; simulator assumption

            # TX power and antenna gain are device/regulatory dependent.
            "tx_power_dbm": 10.0,          # dBm; simulator assumption
            "antenna_gain_db": 0.0,        # dB; isotropic antenna assumption

            # IEEE 802.11ah uses sub-1 GHz spectrum. Exact center frequency depends
            # on region. 915 MHz is a common simulation example.
            "freq_mhz": 915.0,             # MHz; simulator/regional example

            # Propagation model parameters are simulator assumptions.
            # 2.78 (up from 2.7) puts the default AP/STA/relay nominal
            # coverage radius (range_m_for_node in ui/topology_canvas.py,
            # at this file's own default EIRP/sensitivity/915MHz -- see
            # that function) at ~1km, the intended default range for all
            # three roles. This is the single shared baseline every node
            # uses unless something explicitly overrides it (a per-node
            # "Range (m)" edit in the GUI, or the GUI's own environment
            # picker -- see topology_canvas.py's _ENV_PATH_LOSS_EXP, whose
            # own entries were recalibrated around this same ~1km anchor).
            "path_loss_exp": 2.78,         # dimensionless; indoor/mixed environment (free-space = 2.0)
            "pl_ref_distance_m": 1.0,      # meters; reference distance
            "shadow_sigma_db": 0.0,        # dB; disable random shadowing by default
            "shadow_enable": False,        # bool; disable shadowing

            # 1 MHz S1G channel bandwidth.
            "channel_bw_hz": 1_000_000.0,  # Hz; standard-aligned example

            # Receiver noise figure is hardware dependent.
            "noise_figure_db": 5.0,        # dB; simulator assumption

            # PER model is simulator-specific.
            "per_floor": 1e-9,             # probability; PER lower bound
            "per_alpha": 2.0,              # dimensionless; PER slope parameter

            # Minimum SNR thresholds per MCS (IEEE 802.11ah-2016 / literature values).
            "mcs_min_snr_db": {
                "MCS0": 3.0,    # BPSK 1/2
                "MCS1": 6.0,    # QPSK 1/2
                "MCS2": 8.5,    # QPSK 3/4
                "MCS3": 11.5,   # 16-QAM 1/2
            },

            # Capture threshold is a simulator abstraction.
            "capture_threshold_db": 10.0,  # dB; SINR margin for capture model
        },

       "mac": {

            # ------------------------------------------------------------------
            # Core MAC feature toggles
            # ------------------------------------------------------------------
            "raw_enable": bool(raw_enable),
            "raw_policy": "static",   # IMPORTANT: overridden in main script
            "adaptive_grouping": "hybrid",
            # Default False: TWT is now a real negotiated feature (see
            # sim11ah/mac/twt.py). Off by default so existing scripts that
            # never explicitly set this key keep reproducing prior results;
            # opt in per-run with cfg["mac"]["twt_enable"] = True.
            "twt_enable": False,

            # ------------------------------------------------------------------
            # RAW parameters (COMMON for ALL policies → FAIR COMPARISON)
            # ------------------------------------------------------------------
            "raw_num_groups": 4,          # SAME for adaptive / cluster / static
            "raw_nodes_per_group": 125,
            "raw_num_slots": 4,           # SAME for all policies
            "raw_slot_duration": 0.007,   # baseline slot duration

            "raw_guard": 100e-6,
            "ack_guard": 100e-6,
            "raw_cross_slot": True,
            "raw_type": 0,

            # ------------------------------------------------------------------
            # Cluster CSV Policy Parameters
            # ------------------------------------------------------------------
            "cluster_csv_path": "uav_cluster_data.csv",

            "priority_weight_critical": 5.0,
            "priority_weight_high": 3.0,
            "priority_weight_normal": 1.0,

            # Tuning knobs (VERY IMPORTANT for performance)
            "cluster_csv_demand_gain": 3.0,
            "cluster_csv_size_gain": 0.015,
            "cluster_csv_burst_gain": 1.35,
            "cluster_csv_use_runtime_queue": True,

            # ------------------------------------------------------------------
            # Adaptive RAW Parameters (USED by adaptive + cluster)
            # ------------------------------------------------------------------
            "adaptive_raw_step_us": 500,
            "adaptive_raw_budget_fraction": 0.90,
            "adaptive_raw_absolute_min_slot_us": 7000,

            "adaptive_raw_smoothing_beta": 0.5,
            "adaptive_raw_hysteresis": 1000.0,

            "adaptive_raw_ewma_alpha": 0.7,
            "adaptive_raw_cusum_k": 0.10,
            "adaptive_raw_cusum_h": 1.0,

            "adaptive_raw_bianchi_eps": 1e-4,
            "adaptive_raw_bianchi_imax": 50,

            "adaptive_raw_tmax_s": 0.012,
            "adaptive_raw_lth_slots": 150,



            "cluster_adaptive_demand_gain": 2.5,
            "cluster_adaptive_size_gain": 0.012,
            "cluster_adaptive_burst_gain": 1.25,
            # Floored at raw_num_slots (4) rather than 1: a below-average-
            # demand cluster must still get at least as many slots as the
            # "adaptive" baseline gives every group unconditionally, or it
            # starves relative to every other RAW policy in the comparison.
            # Demand differentiation still adds slots above this floor, up
            # to cluster_adaptive_max_slots_per_cluster.
            "cluster_adaptive_min_slots_per_cluster": 4,
            # Raised from 8: measured against seed 42, N=100..1000 -- 16
            # gains +9% PDR at N=400 (0.250->0.273) from letting genuinely
            # hot clusters claim more of the demand-driven bonus above the
            # floor, with no measurable regression elsewhere (<=1.5% at
            # any other N, within single-seed noise).
            "cluster_adaptive_max_slots_per_cluster": 16,
            "cluster_adaptive_slot_demand_exponent": 0.75,

            # ------------------------------------------------------------------
            # Inter-frame timing (IEEE 802.11ah-2016 Table 23-10)
            # ------------------------------------------------------------------
            "sifs": 160e-6,
            "slot_time": 52e-6,
            "difs": 160e-6 + 2 * 52e-6,  # DIFS = SIFS + 2×slot_time = 264 µs
            "ack_timeout": 1e-3,

            # ------------------------------------------------------------------
            # Backoff / retry
            # ------------------------------------------------------------------
            "cw_min": 15,
            "cw_max": 1023,
            "retry_limit": 7,
            "short_retry_limit": 7,
            "long_retry_limit": 4,
            "rts_threshold_bytes": 512,     # 512 B practical IoT threshold (802.11 default 2347 is too high for S1G)

            # ------------------------------------------------------------------
            # Frame sizes
            # ------------------------------------------------------------------
            "ack_size_bytes": 14,
            "beacon_size_bytes": 120,
            "short_beacon_size_bytes": 40,
            "data_mac_overhead_bytes": 36,
            "ps_poll_size_bytes": 20,
            "ba_size_bytes": 32,
            "bar_size_bytes": 24,
            "rts_size_bytes": 20,
            "cts_size_bytes": 14,
            "cf_end_size_bytes": 20,
            "ndpa_size_bytes": 20,
            "probe_req_size_bytes": 40,

            # ------------------------------------------------------------------
            # Beaconing / power-save
            # ------------------------------------------------------------------
            "beacon_interval": 0.5,
            "dtim_period": 1,

            # ------------------------------------------------------------------
            # PRAW
            # ------------------------------------------------------------------
            "praw_enable": False,
            "praw_period": 1,
            "praw_validity": 1,
            "praw_start_offset": 0,
            "praw_refresh": False,

            # ------------------------------------------------------------------
            # Aggregation
            # ------------------------------------------------------------------
            "ampdu_enable": False,
            "ampdu_max_subframes": 4,
            "ampdu_max_bytes": 8192,

            "amsdu_enable": False,
            "amsdu_max_bytes": 3839,
            "amsdu_max_subframes": 8,

            # ------------------------------------------------------------------
            # Fragmentation / lifetime
            # ------------------------------------------------------------------
            "frag_threshold_bytes": 2346,
            "reasm_timeout_s": 0.1,
            "max_msdu_lifetime_s": 5.0,

            # ------------------------------------------------------------------
            # Queueing / stats
            # ------------------------------------------------------------------
            "txq_max_depth": 256,
            "util_window_s": 1.0,

            # ------------------------------------------------------------------
            # IEEE 802.11ah Association procedure
            #   association_enable     — True: run full Open System auth+assoc
            #                           False: nodes are immediately associated
            #                           (backward-compat / fast simulation mode)
            #   auth_frame_size_bytes  — AUTH Request / Response frame size
            #                           (28-byte Mgmt hdr + 6-byte body + 4 FCS + 2 pad)
            #   assoc_req_size_bytes   — ASSOC Request size (hdr + cap IE + S1G IE)
            #   assoc_resp_size_bytes  — ASSOC Response size (hdr + status + AID + rates)
            # Ref: IEEE 802.11ah-2016 Table 9-33 / Clause 9.6.3–9.6.4
            # ------------------------------------------------------------------
            "association_enable":      True,
            "auth_frame_size_bytes":   40,
            "assoc_req_size_bytes":    72,
            "assoc_resp_size_bytes":   48,
        },
        "net": {
            "net_header_bytes": 16,
            "max_hops": 2,
            "seq_mod": 65535,
            "seen_cache_max": 256,
            "max_queue_depth": 0,
            "enable_fragmentation": False,
            "max_msdu_bytes": 2304,
        },

        "transport": {
            "mode": "udp",
            "assign_tp_seq": True,
            "assign_created_at": True,
            "seq_mod": 65535,
            "per_dst_seq": False,

            "goodput_window_s": 1.0,

            "reorder_window": 16,
            "reorder_timeout": 2.0,
            "seen_cache_max": 256,

            "tx_window_size": 0,
            "max_pending": 64,

            "rto_base_s": 0.05,   # 50 ms; realistic for short-range 802.11ah links
            "rto_max_s": 1.0,    # 1 s cap; 10 s is far too long for IoT sensors
            "rto_backoff": 2.0,
            "max_retransmissions": 3,

            "enable_dscp_marking": True,
        },

        "topology": {
            "mode": "star",          # "star" or "relay"
            "relay_ids": [],         # [1, 2, ..., R] — set by RelayBuilder
            "relay_assignment": {},  # {sta_node_id: relay_node_id} — set by RelayBuilder
        },

        "app": {
            "traffic": traffic_mode,
            "packet_size_bytes": 128,
            "dst_mode": "ap",
            "enable_sink": True,

            # Periodic / Poisson traffic generation parameters
            "periodic_interval": 5,
            "periodic_jitter_s": 5,
            "start_spread_s": 1.0,
            "start_phase_mode": "deterministic",
            "start_phase_jitter_s": 30,
            "poisson_lambda": 0.5,

            # Flow-control settings
            "max_in_flight": 0,              # packets; 0 means unlimited
            "congestion_backoff_s": 0.1,
            "in_flight_timeout_s": 10.0,     # seconds; application timeout budget

            # Other traffic models
            "size_mode": "fixed",
            "cbr_rate_bps": 2000.0,
            "burst_size": 3,
            "burst_intra_gap_s": 0.01,
            "burst_off_time_s": 2.0,
            "onoff_lambda_on": 2.0,
            "onoff_on_time_s": 1.0,
            "onoff_off_time_s": 3.0,

            # "video" traffic model (see ApplicationLayer._build_traffic_model
            # in sim11ah/app.py) -- low-power HaLow camera/video sensor.
            # video_size_table overrides DEFAULT_VIDEO_SIZE_TABLE if set.
            "video_fps": 5.0,
            "video_jitter_s": 0.0,
            "video_size_table": None,
        },

        # ------------------------------------------------------------------
        # Energy model (IEEE 802.11ah sub-GHz IoT radio, MORSE MG100 ref)
        #
        # Power levels (Watts):
        #   tx_power_w    — radio transmitting (DATA / ACK / beacon / ctrl)
        #   rx_power_w    — radio actively receiving (preamble detect + decode)
        #   idle_power_w  — radio ON but not TX/RX (CCA, CSMA backoff)
        #   sleep_power_w — deep sleep (RAW slot of another group / TWT doze)
        #
        # References:
        #   [1] MORSE Micro MG100 data sheet (TX ~180 mW, RX ~62 mW)
        #   [2] Tian et al., IEEE Commun. Mag. 2016 (802.11ah power survey)
        #   [3] Oteri et al., WoWMoM 2015 (outdoor 802.11ah power save)
        # ------------------------------------------------------------------
        "energy": {
            "tx_power_w":    0.180,   # 180 mW — MORSE MG100 @ 0 dBm output
            "rx_power_w":    0.062,   # 62 mW  — active RX
            "idle_power_w":  0.020,   # 20 mW  — idle / backoff / carrier sense
            "sleep_power_w": 0.0005,  # 0.5 mW — deep sleep
        },
    }

    # ------------------------------------------------------------------
    # Dynamically compute the feasible adaptive slot-duration range
    # ------------------------------------------------------------------
    if cfg["mac"].get("raw_enable", False) and cfg["mac"].get("raw_policy") == "adaptive":
        #print("Computing adaptive RAW slot duration range based on config parameters...")
        dyn = compute_dynamic_slot_range(cfg)

        cfg["mac"]["adaptive_raw_initial_slot_us"] = dyn["adaptive_raw_initial_slot_us"]
        cfg["mac"]["adaptive_raw_max_slot_us"] = dyn["adaptive_raw_max_slot_us"]

        cfg["mac"]["adaptive_raw_safe_min_slot_us"] = dyn["adaptive_raw_safe_min_slot_us"]
        cfg["mac"]["adaptive_raw_budget_max_slot_us"] = dyn["adaptive_raw_budget_max_slot_us"]
        cfg["mac"]["adaptive_raw_total_slots_per_beacon"] = dyn["adaptive_raw_total_slots_per_beacon"]
        cfg["mac"]["adaptive_raw_range_feasible"] = dyn["adaptive_raw_range_feasible"]
        cfg["mac"]["adaptive_raw_required_beacon_interval_us"] = dyn["adaptive_raw_required_beacon_interval_us"]
        cfg["mac"]["adaptive_raw_max_total_slots_feasible"] = dyn["adaptive_raw_max_total_slots_feasible"]

    return cfg