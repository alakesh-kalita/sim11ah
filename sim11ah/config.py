from typing import Any, Dict


def default_config(raw_enable: bool, traffic_mode: str) -> Dict[str, Any]:
    allowed_traffic = {"periodic", "poisson", "cbr", "burst", "bursty", "onoff"}
    if traffic_mode not in allowed_traffic:
        raise ValueError(
            f"Unsupported traffic_mode={traffic_mode!r}. "
            f"Allowed: {sorted(allowed_traffic)}"
        )

    # ------------------------------------------------------------------
    # IEEE 802.11ah / S1G PHY rates for 1 MHz bandwidth, 1 spatial stream,
    # normal guard interval, as commonly used in the literature.
    #
    # Standard-aligned examples for 1 MHz S1G:
    #   MCS0 = 300 kb/s
    #   MCS1 = 600 kb/s
    #   MCS2 = 900 kb/s
    #   MCS3 = 1.2 Mb/s
    # ------------------------------------------------------------------
    phy_mode_table = {
        "MCS0": 300_000,     # 300 kb/s, 1 MHz, BPSK 1/2
        "MCS1": 600_000,     # 600 kb/s, 1 MHz, QPSK 1/2
        "MCS2": 900_000,     # 900 kb/s, 1 MHz, QPSK 3/4
        "MCS3": 1_200_000,   # 1.2 Mb/s, 1 MHz, 16-QAM 1/2
        # "MCS10": 150_000,  # optional repeated mode
    }

    cfg = {
        "phy": {
            # ------------------------------------------------------------------
            # PHY timing model
            # ------------------------------------------------------------------
            # These are simplified simulator timing parameters. IEEE 802.11ah does
            # not expose a single universal "preamble_time" or "header_time" in
            # this abstract form, since PPDU duration depends on frame format,
            # bandwidth, MCS, and aggregation.
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
            # They are not universal IEEE 802.11ah constants.
            "cca_threshold_dbm": -105.0,   # dBm; simulator assumption
            "rx_sensitivity_dbm": -105.0,  # dBm; simulator assumption

            # TX power and antenna gain are device/regulatory dependent.
            "tx_power_dbm": 10.0,          # dBm; simulator assumption
            "antenna_gain_db": 0.0,        # dB; isotropic antenna assumption

            # IEEE 802.11ah uses sub-1 GHz spectrum. Exact center frequency depends
            # on region. 915 MHz is a common simulation example.
            "freq_mhz": 915.0,             # MHz; simulator/regional example

            # Propagation model parameters are simulator assumptions.
            "path_loss_exp": 2.0,          # dimensionless; free-space-like example
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

            # Simplified SNR thresholds used by the simulator.
            # These are not fixed standard constants.
            "mcs_min_snr_db": {
                "MCS0": 0.0,
                "MCS1": 2.0,
                "MCS2": 4.0,
                "MCS3": 6.0,
            },

            # Capture threshold is a simulator abstraction.
            "capture_threshold_db": 10.0,  # dB; SINR margin for capture model
        },

        "mac": {
            # ------------------------------------------------------------------
            # Core MAC feature toggles
            # ------------------------------------------------------------------
            "raw_enable": bool(raw_enable),   # bool; enable Restricted Access Window
            #"raw_policy": "static",           # simulator RAW policy
            "adaptive_grouping": "hybrid",    # simulator grouping policy
            "twt_enable": True,               # bool; enable TWT-like doze support

            # ------------------------------------------------------------------
            # RAW parameters
            # ------------------------------------------------------------------
            # IMPORTANT:
            # IEEE 802.11ah defines the RAW mechanism and RAW slot-duration
            # encoding, but it does NOT prescribe one universal RAW grouping or
            # slot-allocation strategy. Therefore, the values below are AP /
            # deployment / simulator policy choices, not fixed standard constants.
            #
            # RAW slot duration in IEEE 802.11ah is encoded via a cslot field,
            # but the AP still decides how many groups, how many slots, and which
            # AID ranges belong to each group.
            #
            # The values below are neutral and usable simulator defaults. They are
            # not tuned for best performance, but they avoid degenerate cases such
            # as zero RAW opportunities for many STAs.
            "raw_num_groups": 8,             # AP-policy choice; number of RAW groups
            "raw_nodes_per_group": 64,       # AP-policy choice; nominal AID span/group
            "raw_num_slots": 16,              # AP-policy choice; number of slots/group
            "raw_slot_duration": 0.007,      # seconds; 20 ms slot duration (AP policy)
            "raw_guard": 100e-6,             # seconds; simulator RAW guard interval
            "ack_guard": 100e-6,             # seconds; simulator ACK timing margin
            "raw_cross_slot": True,         # bool; disallow cross-slot exchange by default
            "raw_type": 0,                   # simulator RAW type identifier

            # raw_slot_duration: 10 ms, 20 ms, 30 ms
            # raw_num_slots: 8, 16, 32
            # maybe raw_num_groups: 4, 8, 16

            # ------------------------------------------------------------------
            # Adaptive RAW controls
            # ------------------------------------------------------------------
            # These are simulator-only parameters and are not standardized values.
            #"raw_policy": "static",       # simulator RAW policy
            # "adaptive_raw_min_slot_us": 100,
            # "adaptive_raw_max_slot_us": 7000,
            # "adaptive_raw_ewma_alpha": 0.7,
            # "adaptive_raw_bianchi_eps": 1e-4,
            # "adaptive_raw_bianchi_imax": 50,
            # "adaptive_raw_tmax_s": 0.12,
            # "adaptive_raw_baseline_demand": 0.15,

         

            "raw_policy": "static",       # simulator RAW policy

            "adaptive_raw_min_slot_us": 7000,
            "adaptive_raw_initial_slot_us": 7000,
            "adaptive_raw_max_slot_us": 20000,
            "adaptive_raw_step_us": 1000,

            "adaptive_raw_smoothing_beta": 0.5,
            "adaptive_raw_hysteresis": 1000.0,

            "adaptive_raw_ewma_alpha": 0.7,
            "adaptive_raw_cusum_k": 0.10,
            "adaptive_raw_cusum_h": 1.0,

            "adaptive_raw_bianchi_eps": 1e-4,
            "adaptive_raw_bianchi_imax": 50,

            "adaptive_raw_tmax_s": 0.012,
            "adaptive_raw_lth_slots": 150,


            # ------------------------------------------------------------------
            # Inter-frame timing
            # ------------------------------------------------------------------
            # Standard-aligned S1G timing values commonly used in IEEE 802.11ah work.
            "sifs": 160e-6,                  # seconds; 160 us
            "slot_time": 52e-6,              # seconds; 52 us
            "difs": 264e-6,                  # seconds; SIFS + 2*slot_time

            # ACK timeout is a simulator timing budget, not a single fixed
            # standard scalar. It should be consistent with SIFS, ACK TX time,
            # and propagation.
            "ack_timeout": 1e-3,             # seconds; simulator ACK-timeout budget

            # ------------------------------------------------------------------
            # Backoff / retry
            # ------------------------------------------------------------------
            # Common 802.11-style contention defaults used in simulation.
            "cw_min": 15,                    # slots; minimum contention window
            "cw_max": 1023,                  # slots; maximum contention window
            "retry_limit": 7,                # count; simulator retry limit
            "short_retry_limit": 7,          # count; short frame retry limit
            "long_retry_limit": 4,           # count; long frame retry limit
            "rts_threshold_bytes": 2347,     # bytes; legacy-style RTS threshold

            # ------------------------------------------------------------------
            # Frame sizes
            # ------------------------------------------------------------------
            # These are simulator approximations. Many 802.11ah exchanges are
            # represented more abstractly than exact S1G PPDU encodings.
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
            # TIM/DTIM exist in the standard, but beacon interval is AP-configured.
            "beacon_interval": 0.5,          # seconds; AP policy
            "dtim_period": 1,                # beacons; AP policy

            # ------------------------------------------------------------------
            # PRAW
            # ------------------------------------------------------------------
            # Simulator controls; not fixed standard constants.
            "praw_enable": False,
            "praw_period": 1,
            "praw_validity": 1,
            "praw_start_offset": 0,
            "praw_refresh": False,

            # ------------------------------------------------------------------
            # Aggregation
            # ------------------------------------------------------------------
            # Supported by 802.11ah, but exact limits are implementation-dependent.
            "ampdu_enable": False,
            "ampdu_max_subframes": 4,
            "ampdu_max_bytes": 8192,

            "amsdu_enable": False,
            "amsdu_max_bytes": 3839,
            "amsdu_max_subframes": 8,

            # ------------------------------------------------------------------
            # Fragmentation / lifetime
            # ------------------------------------------------------------------
            # Fragmentation threshold and MSDU lifetime are simulator queue-
            # management parameters, not fixed IEEE 802.11ah constants.
            "frag_threshold_bytes": 2346,
            "reasm_timeout_s": 0.1,
            "max_msdu_lifetime_s": 5.0,      # seconds; simulator MAC MSDU lifetime

            # ------------------------------------------------------------------
            # Queueing / stats
            # ------------------------------------------------------------------
            "txq_max_depth": 256,            # packets; MAC TX queue capacity
            "util_window_s": 1.0,            # seconds; utilization averaging window
        },

        "net": {
            # Network-layer parameters are outside IEEE 802.11ah.
            "net_header_bytes": 16,
            "max_hops": 2,
            "seq_mod": 65535,
            "seen_cache_max": 256,
            "max_queue_depth": 0,
            "enable_fragmentation": False,
            "max_msdu_bytes": 2304,
        },

        "transport": {
            # Transport-layer settings are simulator policy.
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

            "rto_base_s": 1.0,
            "rto_max_s": 10.0,
            "rto_backoff": 2.0,
            "max_retransmissions": 3,

            "enable_dscp_marking": True,
        },

        "app": {
            # Application traffic is workload-specific, not standardized by IEEE 802.11ah.
            "traffic": traffic_mode,
            "packet_size_bytes": 128,
            "dst_mode": "ap",                # application destination policy
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
        },
    }

    return cfg