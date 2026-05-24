# sim11ah — IEEE 802.11ah (Wi-Fi HaLow) Network Simulator

A discrete-event simulator for the IEEE 802.11ah (Wi-Fi HaLow) sub-1 GHz MAC/PHY protocol, purpose-built for large-scale IoT research. Includes a full-featured GUI dashboard, CLI batch runner, relay topology support, and reproducible paper experiments.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [GUI Dashboard](#gui-dashboard)
  - [CLI Batch Runner](#cli-batch-runner)
  - [Paper Experiments](#paper-experiments)
- [Configuration Parameters](#configuration-parameters)
- [Traffic Models](#traffic-models)
- [Topology Modes](#topology-modes)
- [RAW Scheduling Policies](#raw-scheduling-policies)
- [Project Structure](#project-structure)
- [Protocol Implementation Details](#protocol-implementation-details)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Overview

**sim11ah** simulates the full protocol stack of IEEE 802.11ah including:

- S1G PHY with MCS0–MCS3 (1 MHz, 150–600 kb/s), SINR-based PER model, path loss
- DCF/CSMA-CA MAC with Binary Exponential Backoff (BEB), RTS/CTS, ACK, retransmits
- Restricted Access Window (RAW) slot scheduling with static, adaptive, and cluster-aware policies
- Store-and-forward relay topology with deduplication
- Five traffic source models: Periodic, Poisson, CBR, Bursty, On-Off
- Per-packet statistics: PDR, throughput, average delay, 95th-percentile delay, drop rate

The simulator was built to reproduce and extend results from the associated IEEE paper on RAW-based MAC optimization for dense IoT deployments.

---

## Features

| Feature | Details |
|---|---|
| Protocol stack | PHY · MAC (DCF + RAW) · Network · App |
| Node counts | Tested up to N = 200 STAs |
| Topologies | Star (AP ↔ STAs) and Relay (AP ↔ Relays ↔ STAs) |
| RAW policies | Static, Adaptive, Cluster-Adaptive, Cluster-CSV |
| Traffic models | Periodic, Poisson, CBR, Bursty, On-Off |
| GUI | Live charts, topology canvas, three-state sim control (Start/Pause/Resume/Stop) |
| CLI | Headless batch runs with CSV export |
| Paper scripts | Reproducible experiment sweeps for all paper tables and figures |

---

## Architecture

```
sim11ah/
├── engine.py        # Discrete-event engine (priority queue)
├── simulator.py     # Top-level orchestrator, medium access arbiter
├── node.py          # Per-node state machine (PHY · MAC · Net · App layers)
├── phy.py           # S1G PHY: SINR, PER model, path loss, transmission timing
├── mac/
│   ├── dcf.py               # CSMA-CA, BEB, RTS/CTS, ACK, retransmit logic
│   ├── raw.py               # RAW slot manager, beacon injection
│   ├── raw_policy_static.py         # Fixed-slot RAW policy
│   ├── raw_policy_adaptive.py       # Load-adaptive RAW policy
│   ├── raw_policy_cluster_adaptive.py  # Cluster-aware adaptive policy
│   ├── raw_policy_cluster_csv.py    # Policy driven by CSV cluster assignments
│   └── adaptive_config_policy.py   # Dynamic slot-range calculator
├── net.py           # Network layer: routing, relay store-and-forward, dedup
├── app.py           # Application layer: traffic source models
├── topology.py      # StarBuilder, RelayBuilder
├── config.py        # default_config() — all tunable IEEE 802.11ah parameters
├── constants.py     # Frame type enumerations
├── stats.py         # Per-simulation statistics collector
├── metrics.py       # Derived metric helpers
└── models.py        # Packet, NetPDU, MacFrame data classes
```

---

## Installation

**Requirements:** Python ≥ 3.10, tkinter (for GUI)

```bash
# Clone the repository
git clone https://github.com/<your-username>/sim11ah.git
cd sim11ah

# Install in editable mode (no external dependencies beyond stdlib + matplotlib)
pip install -e .

# Optional: install matplotlib for CLI charts and paper plots
pip install matplotlib
```

> **macOS note:** tkinter ships with the Python.org installer. If using Homebrew Python, install `python-tk` via `brew install python-tk@3.x`.

---

## Quick Start

### GUI Dashboard

```bash
python scripts/main_gui.py
```

The dashboard provides:

- **Run Controls** — Start / Pause / Resume / Stop & Reset / Step
- **Settings panel** — all simulation parameters with live pending-changes indicator
- **Live charts** — PDR, throughput, delay, drop rate updated in real time
- **Topology canvas** — visual preview of star or relay topology
- **Log viewer** — per-event log with filter and auto-scroll
- **Export** — save logs and results to CSV

![Dashboard screenshot](docs/dashboard_screenshot.png)

### CLI Batch Runner

```bash
# Run with defaults: N=50 STAs, periodic traffic, RAW enabled
python scripts/main_cli.py

# Custom run
python scripts/main_cli.py \
  --num-stas 100 \
  --traffic poisson \
  --raw-enable \
  --raw-policy adaptive \
  --sim-time 120 \
  --seed 42 \
  --out-csv results/my_run.csv
```

### Paper Experiments

Reproduce all tables and figures from the paper:

```bash
# Main scaling sweep (Tables II–IV, Figs. 4–7)
python paper/run_paper_experiments.py

# Relay topology evaluation (Table VI, Fig. 8)
python paper/run_relay_experiments.py

# Analytical validation sweep
python paper/sweep_experiment.py

# Generate plots
python analysis/plot_simulator_results.py
python analysis/plot_critical_comparison.py
```

Results are written to `paper/paper_results.csv` and `paper/relay_results.csv`.

---

## Configuration Parameters

All parameters are set via `sim11ah.config.default_config()` and can be overridden per-run.

### PHY Parameters

| Parameter | Default | Description |
|---|---|---|
| `default_mode` | `MCS0` | PHY rate mode (MCS0=150 kb/s · MCS1=300 · MCS2=450 · MCS3=600) |
| `tx_power_dbm` | `10.0` | Transmit power (dBm) |
| `carrier_freq_hz` | `915e6` | Carrier frequency (sub-1 GHz, 915 MHz) |
| `path_loss_exp` | `3.0` | Path loss exponent |
| `noise_figure_db` | `5.0` | Receiver noise figure (dB) |
| `cca_threshold_dbm` | `-105.0` | Clear Channel Assessment threshold (dBm) |
| `per_alpha` | `2.0` | PER model slope: `PER = max(ε₀, exp(-α·(SINR − γ_th)))` |
| `per_floor` | `1e-9` | Minimum PER floor ε₀ |
| `sinr_threshold_db` | `4.0` | SINR threshold γ_th (dB) |

### MAC / DCF Parameters

| Parameter | Default | Description |
|---|---|---|
| `slot_time` | `52 µs` | IEEE 802.11ah S1G slot time |
| `sifs` | `160 µs` | Short Inter-Frame Space |
| `difs` | `264 µs` | DCF Inter-Frame Space |
| `cw_min` | `15` | Minimum contention window |
| `cw_max` | `1023` | Maximum contention window |
| `max_retries` | `7` | Max MAC retransmission attempts |
| `rts_threshold` | `2346` | RTS/CTS activation threshold (bytes) |
| `ack_timeout` | `500 µs` | ACK wait timeout |

### RAW Parameters

| Parameter | Default | Description |
|---|---|---|
| `raw_enable` | `True` | Enable RAW slot scheduling |
| `raw_policy` | `static` | Policy: `static` / `adaptive` / `cluster_adaptive` / `cluster_csv` |
| `beacon_interval` | `0.1024 s` | Beacon period (102.4 ms, standard TU) |
| `raw_num_slots` | `4` | Number of RAW slots per beacon interval |
| `raw_slot_duration` | `20 ms` | Duration of each RAW slot |
| `raw_cross_slot_boundary` | `False` | Allow transmission to span slot boundary |

### Application Parameters

| Parameter | Default | Description |
|---|---|---|
| `packet_size_bytes` | `128` | Application payload size |
| `periodic_interval` | `5.0 s` | Inter-packet interval for Periodic traffic |
| `poisson_lambda` | `0.5` | Average rate (pkt/s) for Poisson traffic |
| `cbr_rate_bps` | `2000` | Bit rate for CBR traffic |
| `burst_size` | `3` | Packets per burst for Bursty traffic |
| `onoff_on_time_s` | `1.0` | ON-period duration for On-Off traffic |
| `onoff_off_time_s` | `3.0` | OFF-period duration for On-Off traffic |

---

## Traffic Models

| Model | Class | Behaviour |
|---|---|---|
| `periodic` | `PeriodicTraffic` | Fixed inter-packet interval |
| `poisson` | `PoissonTraffic` | Exponentially distributed inter-arrival times |
| `cbr` | `CBRTraffic` | Constant bit-rate, packet size determines interval |
| `bursty` | `BurstyTraffic` | Burst of N packets, then silent off-period |
| `onoff` | `OnOffTraffic` | Poisson arrivals during ON, silent during OFF |

---

## Topology Modes

### Star

All STAs associate directly with the AP. Standard IEEE 802.11ah single-hop.

```
STA₁ ──┐
STA₂ ──┤── AP (node 0)
  ⋮    ┤
STA_N ─┘
```

### Relay

One or more relay nodes (R) are placed between the AP and STAs. Each relay handles a partition of STAs. Relays use store-and-forward with per-packet deduplication.

```
STA₁ ──┐              ┌── STA_(R+1)
STA₂ ──┤── Relay₁ ────┤    ⋮
       │               └── STA_K
AP ────┤
       │               ┌── STA_(K+1)
       └── Relay₂ ────┤    ⋮
                       └── STA_N
```

- **Backhaul link** (AP ↔ Relay): 600 kb/s, 100 µs propagation delay
- **Access link** (Relay ↔ STA): 300 kb/s, 300 µs propagation delay
- Relay nodes do not generate their own traffic

---

## RAW Scheduling Policies

| Policy | Key | Description |
|---|---|---|
| Static | `static` | Fixed equal-duration slots, round-robin STA assignment |
| Adaptive | `adaptive` | Slot count and duration adjust based on observed load |
| Cluster Adaptive | `cluster_adaptive` | Groups STAs by traffic class; assigns dedicated slots per cluster |
| Cluster CSV | `cluster_csv` | Reads cluster assignments from `uav_cluster_data.csv` |

**Recommended settings for best PDR:**
- N ≤ 50: `raw_enable=False` (DCF alone suffices, RAW overhead hurts)
- 50 < N ≤ 100: `static` RAW, 4 slots, 20 ms slot duration
- N > 100: `adaptive` or `cluster_adaptive` RAW

---

## Project Structure

```
sim11ah_project/
├── sim11ah/              # Core simulator package
│   ├── mac/              # MAC sub-package (DCF + RAW policies)
│   └── utils/            # Utility helpers (cluster CSV generator, priority metrics)
├── ui/
│   └── dashboard_tk.py   # Tkinter GUI dashboard (~1400 lines)
├── scripts/
│   ├── main_gui.py       # GUI entry point
│   ├── main_cli.py       # Headless CLI runner
│   └── main_test_*.py    # Ad-hoc test scripts
├── analysis/
│   ├── plot_simulator_results.py   # Main results plots
│   ├── plot_critical_comparison.py # Critical traffic comparison
│   └── plot_uav*.py                # UAV cluster analysis plots
├── data/
│   └── uav_cluster_data.csv        # UAV cluster assignment data
├── results/                        # Saved simulation output CSVs and PDFs
├── tests/
│   └── test_packet_interval.py     # Unit tests
└── pyproject.toml
```

---

## Protocol Implementation Details

### PHY Layer (`sim11ah/phy.py`)

- **Transmission time**: `t_tx = preamble + header + (payload_bits / rate_bps)`
- **Path loss**: log-distance model, `PL(d) = PL(d₀) + 10·n·log₁₀(d/d₀)`
- **SINR**: accounts for all concurrent transmissions as interference
- **PER model**: `PER(SINR) = max(ε₀, exp(−α · (SINR − γ_th)))` with α=2.0, ε₀=10⁻⁹
- Capture effect: strongest signal wins when SINR > threshold

### MAC Layer (`sim11ah/mac/dcf.py`)

- Full CSMA-CA with binary exponential backoff
- Backoff counter decremented only during idle medium periods
- RTS/CTS four-way handshake for frames above `rts_threshold`
- Per-frame ACK with configurable timeout and retry limit
- RAW: STAs transmit only within their assigned RAW slot window

### Network Layer (`sim11ah/net.py`)

- Relay store-and-forward: uplink packets buffered at relay, forwarded to AP
- Deduplication keyed on `(src_node_id, packet_seq_num)` — prevents double-counting
- Hop-count aware routing for multi-hop paths

### Application Layer (`sim11ah/app.py`)

- Each STA has an independent traffic source instance
- AP and relay nodes have no traffic source (set to `None`)
- Packet generation scheduled via the discrete-event engine

---

## Requirements

- Python ≥ 3.10
- `tkinter` (stdlib, for GUI)
- `matplotlib` ≥ 3.5 (for CLI charts and analysis plots)

No third-party simulation frameworks are required. The entire discrete-event engine is implemented in pure Python (`sim11ah/engine.py`).

---

## Citation

If you use sim11ah in your research, please cite:

```bibtex
@article{kalita2026sim11ah,
  title   = {sim11ah: A Discrete-Event Simulator for IEEE 802.11ah
             MAC/PHY Optimization in Dense IoT Deployments},
  author  = {Kalita, Alakesh},
  journal = {Not Communicated yet},
  year    = {2026},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
