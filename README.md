---

# SIM11AH: IEEE 802.11ah (Wi-Fi HaLow) Network Simulator

SIM11AH is a modular, research-oriented discrete-event simulator for IEEE 802.11ah (Wi-Fi HaLow) networks. It is designed for evaluating RAW (Restricted Access Window) scheduling, MAC-layer protocols, and adaptive resource allocation strategies.

---

## Overview

SIM11AH is a Python-based simulation framework developed for:

* Studying IEEE 802.11ah MAC/PHY behavior
* Designing and evaluating adaptive RAW scheduling algorithms
* Analyzing performance metrics such as:

  * Packet Delivery Ratio (PDR)
  * Throughput
  * Delay
  * Retransmissions
* Supporting research in:

  * IoT networks
  * UAV-based communication
  * Industrial wireless systems

The simulator follows a layered architecture and supports both CLI and GUI execution.

---

## Architecture

The simulator follows a layered networking stack:

Application → Transport → Network → MAC → PHY

### Core Components

| Module       | Description                                |
| ------------ | ------------------------------------------ |
| engine.py    | Discrete-event simulation engine           |
| node.py      | Node abstraction integrating all layers    |
| phy.py       | Physical layer (propagation, SINR, PER)    |
| mac/         | MAC layer (RAW scheduling, DCF, buffering) |
| net.py       | Network layer (routing, forwarding)        |
| tp.py        | Transport layer                            |
| app.py       | Traffic generation models                  |
| topology.py  | Network topology generation                |
| stats.py     | Simulation statistics                      |
| metrics.py   | Performance metrics computation            |
| logger.py    | Event logging                              |
| config.py    | Simulation configuration                   |
| constants.py | System-wide constants                      |

---

## Project Structure

```
SIM11AH_PROJECT/
│
├── sim11ah/
│   ├── mac/
│   │   ├── common.py                # Shared MAC utilities and helpers
│   │   ├── context.py               # MAC context/state management
│   │   ├── dcf.py                   # DCF contention logic
│   │   ├── facade.py                # MAC layer interface/controller
│   │   ├── raw.py                   # RAW scheduling core
│   │   ├── raw_metrics.py           # RAW-specific metrics collection
│   │   ├── raw_policy_static.py     # Static RAW allocation policy
│   │   ├── raw_policy_adaptive.py   # Adaptive RAW slot sizing policy (proposed)
│   │   └── raw_policy_adaptive_old.py # Legacy adaptive policy (for comparison)
│   │
│   ├── phy.py
│   ├── net.py
│   ├── tp.py
│   ├── app.py
│   ├── node.py
│   ├── engine.py
│   ├── topology.py
│   ├── stats.py
│   ├── metrics.py
│   ├── logger.py
│   ├── config.py
│   ├── constants.py
│   ├── simulator.py
│   └── io_utils.py
│
├── ui/
│   └── main_gui.py
│
├── main_cli.py
├── main_gui.py
├── results/
└── README.md
```

---

## Features

### IEEE 802.11ah Support

* RAW (Restricted Access Window) scheduling
* Beaconing and DTIM handling
* AID-based grouping

### MAC Layer

* DCF-based contention
* Backoff, retries, ACK handling
* RAW slot entry/exit tracking

### PHY Layer

* SINR-based reception model
* Path loss with shadowing
* Capture effect support

### Traffic Models

* Periodic
* Poisson
* CBR
* Bursty
* On-Off

### Additional Capabilities

* Adaptive RAW slot sizing
* Event-driven simulation
* Detailed logging
* Per-node statistics

---

## Running the Simulator

### CLI Mode

```
python main_cli.py
```

### GUI Mode

```
python main_gui.py
```

The GUI allows configuration of:

* Number of STAs
* Traffic models
* RAW enable/disable
* Visualization of logs and metrics

---

## Output

Simulation results are stored in:

```
results/run_<timestamp>/
```

Outputs include:

* Summary statistics
* Detailed logs
* Performance metrics

Example:

```
PDR: 0.972
Throughput: 97.2 packets/sec
Average Delay: 1.91 s
```

---

## Research Focus

This simulator is designed to evaluate:

* Adaptive RAW slot sizing
* Traffic-aware scheduling
* Contention optimization
* Energy-efficient IoT communication

It is suitable for research in:

* IEEE 802.11ah networks
* UAV communication systems
* Industrial IoT

---

## Requirements

* Python 3.10 or higher

Recommended libraries:

* numpy
* matplotlib

Install dependencies:

```
pip install numpy matplotlib
```

---

## Example Use Case

Evaluate adaptive RAW scheduling:

```
- Configure number of STAs = 100
- Enable RAW
- Use Poisson traffic
- Observe:
    - PDR improvement
    - Reduced collisions
    - Lower delay
```

---

## Future Work

* Reinforcement learning-based scheduling (PPO/DQN)
* Multi-hop routing integration
* LoRa and mmWave extensions
* Real testbed integration

---

## Author

Alakesh Kalita

Assistant Professor

Department of Mathematics and Computing

IIT (ISM) Dhanbad

---

## License

This project is intended for academic and research purposes.

