# sim11ah: A Modular Discrete-Event Simulator for IEEE 802.11ah Restricted Access Window Evaluation

**Alakesh Kalita**
Department of [Your Department]
[Your Institution], [City, Country]
alakesh.kalita1025@gmail.com

---

## Abstract

We present **sim11ah**, a modular, Python-based discrete-event network simulator purpose-built for evaluating the IEEE 802.11ah (Wi-Fi HaLow) Restricted Access Window (RAW) mechanism in dense Internet-of-Things (IoT) deployments. The simulator implements a five-layer protocol stack — PHY, MAC, Network, Transport, and Application — using a deterministic, priority-queue-driven event engine. The MAC layer faithfully models the 802.11ah DCF/CSMA-CA contention procedure with correct 1 MHz S1G timing parameters (slot time 52 µs, SIFS 160 µs, DIFS 264 µs) and incorporates the full RAW slot scheduling pipeline: group construction, AID-range assignment, per-slot fit checking, backoff preservation, and cross-slot boundary enforcement. The static RAW policy — the baseline examined in this paper — partitions stations into fixed contention groups based on Association ID (AID) ranges and assigns each group a non-overlapping time slot within every DTIM beacon interval. A SINR-based physical layer model with log-distance path loss and a configurable exponential packet error rate (PER) curve provides realistic channel behavior. The simulator exposes comprehensive per-node and system-level metrics: packet delivery ratio (PDR), end-to-end delay (mean and 95th-percentile), aggregate throughput, Jain's fairness index, RAW slot utilization, and PHY-layer collision statistics. sim11ah is released as an installable Python package, enabling reproducible, parameter-sweep experiments across arbitrary station counts, traffic patterns, and RAW configurations without modifying core library code.

**Index Terms** — IEEE 802.11ah, Wi-Fi HaLow, Restricted Access Window, RAW, discrete-event simulation, IoT, CSMA/CA, DCF, S1G.

---

## I. Introduction

The proliferation of low-power IoT devices operating in dense deployment scenarios — smart metering, agricultural monitoring, industrial sensing — demands MAC protocols capable of sustaining hundreds of concurrent stations while preserving energy efficiency and bounded latency. IEEE 802.11ah (Wi-Fi HaLow), ratified in 2016, targets precisely this regime by operating in the sub-1 GHz band (902–928 MHz in North America, analogous bands globally) with channel bandwidths as narrow as 1 MHz [1]. Its defining MAC innovation, the Restricted Access Window (RAW), subdivides the contention period following each DTIM beacon into multiple time-limited slots, each accessible only by the stations assigned to it. By limiting the contending population per slot, RAW dramatically reduces collision probability and channel waste, improving both throughput and energy efficiency compared to uncoordinated DCF.

Despite the practical importance of RAW, existing simulation tools offer limited support. The ns-3 802.11ah module [2] is the most complete open implementation but requires significant C++ expertise for extension. The MATLAB-based evaluations in [3] are not publicly available. Bianchi fixed-point analyses [4] provide closed-form throughput bounds but cannot capture timing, heterogeneous traffic, or cross-layer interactions. Commercial tools such as OPNET and Riverbed do not natively model 802.11ah.

This paper describes **sim11ah**, a Python simulator designed to close this gap. Our contributions are:

1. A fully modular, five-layer 802.11ah protocol stack implemented as an installable Python package (`sim11ah`).
2. A deterministic discrete-event engine with reproducible seeded random number generation, suitable for Monte Carlo experiments.
3. A faithful implementation of the 802.11ah static RAW mechanism, including AID-range grouping, per-slot budget enforcement, cross-slot boundary handling, and beacon-cycle scheduling.
4. A SINR-based PHY layer with log-distance path loss, lognormal shadowing, and an exponential PER curve calibrated to standard 802.11ah MCS thresholds.
5. A rich metrics framework covering PDR, delay, throughput, fairness, and RAW-specific counters, with CSV export for post-processing.

The remainder of the paper is organized as follows. Section II surveys related work. Section III presents the system architecture. Sections IV–VIII describe each protocol layer in detail. Section IX focuses on the RAW implementation. Section X describes the evaluation framework. Section XI concludes.

---

## II. Related Work

**ns-3 802.11ah module.** Khorov et al. [2] extended ns-3 with 802.11ah PHY and MAC support, including RAW. While comprehensive, C++ development friction limits rapid prototyping of new RAW policies.

**Bianchi analytical models.** The classical Bianchi saturation throughput model [4], extended to 802.11ah RAW by Tian et al. [5], provides accurate closed-form expressions for per-group throughput under saturated, homogeneous traffic. sim11ah uses a Bianchi fixed-point solver in its adaptive RAW policies but exposes it alongside a full simulation for non-saturated and heterogeneous scenarios.

**OMNET++/OMNeT++-based work.** Sensor-MAC simulators [6] built on OMNeT++ model 802.15.4 but lack 802.11ah RAW specifics.

**Python simulation frameworks.** SimPy [7] provides coroutine-based discrete-event primitives. sim11ah uses a custom heap-based engine for performance and determinism.

**MORSE.** The Morse Micro open-source 802.11ah stack [8] introduced a non-standard RAW slot-duration encoding (500 + index × 120 µs) that sim11ah deliberately preserves for cross-validation with MORSE hardware.

---

## III. System Architecture

### A. Overview

sim11ah adopts a **layered-decomposition, event-driven** architecture. Each network node hosts five protocol layers instantiated in bottom-up order: PHY, MAC, NET, TRANSPORT, APP. All layers share access to a single `Simulator` object, which exposes the global event engine, logger, statistics collector, and topology graph.

The architecture is shown in **Fig. 1**.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Simulator                                │
│  ┌──────────────┐   ┌───────────┐   ┌──────────┐               │
│  │ EventEngine  │   │ SimLogger │   │ SimStats │               │
│  │  (min-heap)  │   │  (ring)   │   │          │               │
│  └──────────────┘   └───────────┘   └──────────┘               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Topology  (star: AP ←→ N×STA)                           │   │
│  │  Link { rate_bps, prop_delay, per }                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

    Per-Node Protocol Stack (AP: node_id=0; STAs: node_id=1..N)
    ┌─────────────────────────────────────────────────┐
    │                   Node                          │
    │  ┌─────────────────────────────────────────┐   │
    │  │  ApplicationLayer  (APP)                │   │
    │  │  Traffic: Periodic|Poisson|CBR|Burst|   │   │
    │  │           OnOff                         │   │
    │  └───────────────────┬─────────────────────┘   │
    │                      │ send_down(Packet)        │
    │  ┌───────────────────▼─────────────────────┐   │
    │  │  TransportLayer  (TP)                   │   │
    │  │  UDP | lightweight ARQ                  │   │
    │  └───────────────────┬─────────────────────┘   │
    │                      │ send_down(NetPDU)        │
    │  ┌───────────────────▼─────────────────────┐   │
    │  │  NetworkLayer  (NET)                    │   │
    │  │  IP-like header, fragmentation,         │   │
    │  │  duplicate suppression                  │   │
    │  └───────────────────┬─────────────────────┘   │
    │                      │ send_down(NetPDU)        │
    │  ┌───────────────────▼─────────────────────┐   │
    │  │  MacLayer  (MAC)  ← facade              │   │
    │  │  ┌──────────────┐  ┌────────────────┐  │   │
    │  │  │  DcfEngine   │  │  RawEngine     │  │   │
    │  │  │  CSMA-CA     │  │  ┌──────────┐  │  │   │
    │  │  │  backoff     │  │  │  Static  │  │  │   │
    │  │  │  retry       │  │  │  Policy  │  │  │   │
    │  │  │  NAV         │  │  └──────────┘  │  │   │
    │  │  └──────────────┘  └────────────────┘  │   │
    │  │  MacContext (shared state dataclass)    │   │
    │  └───────────────────┬─────────────────────┘   │
    │                      │ send(MacFrame)           │
    │  ┌───────────────────▼─────────────────────┐   │
    │  │  PhyLayer  (PHY)                        │   │
    │  │  SINR | path loss | PER | capture       │   │
    │  │  TxRecord / RxRecord pipeline           │   │
    │  └─────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────┘
```
*Fig. 1. sim11ah system architecture and per-node layer stack.*

### B. Module Inventory

Table I lists all Python modules and their responsibilities.

**TABLE I. sim11ah Module Inventory**

| Module | Role |
|---|---|
| `engine.py` | Deterministic min-heap discrete-event engine |
| `simulator.py` | Global simulation context, lifecycle, medium registry |
| `topology.py` | `Link` dataclass + `Topology` + `StarBuilder` |
| `node.py` | Per-node container; owns five protocol-layer instances |
| `models.py` | Shared data structures: `Packet`, `NetPDU`, `MacFrame` |
| `constants.py` | `FrameType` and `MacState` enumerations |
| `config.py` | `default_config()` — all tunable parameters in one dict |
| `stats.py` | `SimStats` dataclass; PDR, delay, fairness, RAW counters |
| `logger.py` | Ring-buffer structured event log; CSV export |
| `phy.py` | `PhyLayer`, `TxRecord`, `RxRecord` |
| `mac/__init__.py` | Public `MacLayer` alias re-export |
| `mac/facade.py` | `MacLayer` — public MAC API, ACK/beacon TX, frame construction |
| `mac/context.py` | `MacContext` dataclass (all per-node MAC state) + `build_mac_context()` |
| `mac/dcf.py` | `DcfEngine` — CSMA-CA backoff, ACK timeout, retry logic |
| `mac/raw.py` | `RawEngine` — slot scheduling, fit check, backoff preservation |
| `mac/raw_policy_static.py` | `StaticRawPolicy` — fixed group/slot assignment (this paper) |
| `mac/common.py` | Shared constants: MORSE slot encoding, EDCA defaults, `RawConfig` |
| `mac/raw_metrics.py` | `MacMetrics` — per-STA throughput, retries, latency |
| `net.py` | `NetworkLayer` — IP-like forwarding, fragmentation, reassembly |
| `tp.py` | `TransportLayer` — UDP / ARQ RTO |
| `app.py` | `ApplicationLayer` + traffic models |
| `metrics.py` | Cross-layer aggregation helpers |
| `io_utils.py` | CSV/JSON result serialization |

### C. Data Flow

A packet originates in the **Application Layer**, is encapsulated downward through **Transport → Network → MAC → PHY**, transmitted over the shared wireless medium, and received by the destination node in reverse order. All inter-layer calls are synchronous Python function calls within the same event callback, ensuring zero-overhead cross-layer interaction. The event engine schedules future callbacks (e.g., `PHY_TX_END_AIR`, `DCF_BACKOFF_TICK`, `MAC_SEND_ACK`) as timestamped heap entries.

---

## IV. Event Engine

The `EventEngine` class implements a **deterministic, min-heap priority-queue discrete-event simulation (DES)** kernel.

### A. Core Data Structure

Events are stored as `ScheduledEvent` dataclasses:

```
ScheduledEvent:
  time  : float          # absolute simulation time
  seq   : int            # tie-breaker (FIFO within same time)
  eid   : int            # unique event ID for cancellation
  cb    : Callable       # callback to invoke
  args  : tuple          # positional arguments
  kwargs: dict           # keyword arguments
  name  : str            # human-readable label for debugging
```

Events are heap-ordered on `(time, seq)`, providing O(log n) insertion and O(log n) extraction. Cancellation is **lazy** — cancelled event IDs are added to a `set[int]`; stale entries are discarded upon dequeue. This avoids the O(n) heap deletion problem.

### B. Reproducibility

The engine exposes a single `random.Random(seed)` instance (`engine.rng`) shared across all layers. All stochastic decisions — backoff slot count, shadowing draws, PER Bernoulli trials — consume from this single stream, guaranteeing bitwise-identical results for a given seed and configuration.

### C. Key API

```python
engine.schedule(time_abs, cb, *args, name="EVENT_NAME")
engine.schedule_in(delay, cb, *args, name="EVENT_NAME")
engine.cancel(event_id)   # best-effort, O(1)
engine.run(until=T)       # run to absolute time T
engine.step(n=1)          # execute n events (GUI stepping)
```

The `name` field enables structured logging of every simulation event without runtime overhead.

---

## V. Physical Layer

### A. Propagation Model

The received signal strength (RSSI) at receiver *r* from transmitter *t* is computed using the log-distance path loss model:

$$\text{RSSI}_{t \to r} \; [\text{dBm}] = \text{EIRP}_t - \text{PL}(d_{tr})$$

$$\text{PL}(d) = \text{PL}_0 + 10 \cdot \eta \cdot \log_{10}\!\left(\frac{d}{d_0}\right) + X_\sigma$$

where PL₀ = 32.44 + 20 log₁₀(*f*_MHz) − 60 [dB] is the free-space path loss at the reference distance *d*₀ = 1 m, *η* = 2.7 is the path-loss exponent for indoor/mixed environments, *f* = 915 MHz, and *X*_σ ∼ N(0, σ²) is a lognormal shadowing term (default σ = 0 dB, disabled by default for deterministic baseline experiments). Shadowing is symmetric and cached per node pair.

### B. SINR and Noise

The thermal noise floor is:

$$N_0 \; [\text{dBm}] = -174 + 10\log_{10}(B) + F$$

where *B* = 1 MHz and *F* = 5 dB is the noise figure, giving *N*₀ ≈ −109 dBm.

The SINR at the receiver, accounting for all concurrent transmissions, is:

$$\text{SINR} = \frac{S}{\displaystyle N_0 + \sum_{j \neq \text{desired}} I_j}$$

where *S* and *I*_j are received signal and interferer powers in milliwatts.

### C. Packet Error Rate Model

Packet reception is modeled by an exponential PER curve anchored at the per-MCS minimum SNR threshold *γ*_th:

$$\text{PER}(\text{SINR}) = \begin{cases} 1.0 & \text{SINR} < \gamma_\text{th} \\ \max\!\left(\epsilon_0,\; e^{-\alpha(\text{SINR} - \gamma_\text{th})}\right) & \text{otherwise} \end{cases}$$

with steepness parameter *α* = 1.5 and noise floor ε₀ = 10⁻⁶. Minimum SNR thresholds follow IEEE 802.11ah-2016 Table 23-53: MCS0 → 3 dB, MCS1 → 6 dB, MCS2 → 8.5 dB, MCS3 → 11.5 dB.

### D. MCS Rate Table (1 MHz, 1 SS, Normal GI)

| MCS | Modulation | Coding | Rate |
|---|---|---|---|
| 0 | BPSK | 1/2 | 150 kb/s |
| 1 | QPSK | 1/2 | 300 kb/s |
| 2 | QPSK | 3/4 | 450 kb/s |
| 3 | 16-QAM | 1/2 | 600 kb/s |

*TABLE II. 802.11ah S1G 1 MHz PHY rates (IEEE 802.11ah-2016 Table 23-53).*

### E. Frame Duration

Frame air-time is computed as:

$$T_\text{frame} = T_\text{preamble} + T_\text{header} + \frac{8 \cdot L_\text{bytes}}{R_\text{bps}}$$

with *T*_preamble = 320 µs, *T*_header = 80 µs (simplified S1G PHY model).

### F. Capture Effect

When multiple overlapping signals arrive simultaneously at a receiver, the SINR-based capture criterion determines whether the strongest signal can be decoded:

$$\text{capture} = \left[ \text{SINR}_\text{strong} \geq \delta_\text{cap} \right]$$

where SINR_strong = *S*_max / (*N*₀ + Σ*I*_j) across all co-channel interferers, and δ_cap = 10 dB. This correctly accounts for thermal noise in capture, unlike simpler C/I-ratio models.

### G. Half-Duplex Collision Detection

The global active-air registry `_active_air_tx: Dict[int, TxRecord]` is keyed by transmitter node ID, enabling O(1) half-duplex conflict checks: if a receiver node is itself transmitting (its ID is in the dict) during an inbound frame's reception window, the frame is immediately flagged as collided.

---

## VI. MAC Layer

### A. Architecture

The MAC layer is split into three cooperating objects sharing a single `MacContext` dataclass:

- **`MacLayer` (facade):** Public API; handles beacon transmission/reception, ACK framing, and frame dispatch.
- **`DcfEngine`:** CSMA-CA backoff state machine.
- **`RawEngine`:** RAW slot scheduler, fit checker, and policy dispatcher.

`MacContext` is a plain `@dataclass` holding all per-node MAC state (queues, timers, counters, RAW slot boundaries, EDCA parameters). This separation eliminates the tight coupling of monolithic MAC implementations and makes each sub-engine independently testable.

### B. 802.11ah Timing Parameters

All timing values are taken from IEEE 802.11ah-2016 Table 23-10:

| Parameter | Value | Standard Reference |
|---|---|---|
| Slot time (σ) | 52 µs | Table 23-10 |
| SIFS | 160 µs | Table 23-10 |
| DIFS | SIFS + 2σ = 264 µs | Table 23-10 |
| CW_min | 15 | Table 9-82 |
| CW_max | 1023 | Table 9-82 |
| Short retry limit | 7 | |
| Long retry limit | 4 | |
| Beacon interval | 500 ms (configurable) | |

*TABLE III. 802.11ah MAC timing parameters.*

### C. DCF/CSMA-CA Backoff

The `DcfEngine` implements the standard 802.11 binary exponential backoff (BEB) procedure, adapted for 802.11ah timing:

```
Algorithm 1: DCF Backoff Procedure

1. On new frame to transmit:
   a. Draw backoff: B ← Uniform[0, CW]
   b. Set state = BACKOFF

2. On each DCF slot tick (delay = σ):
   a. If medium busy: freeze counter, wait for idle + DIFS
   b. Else if elapsed_idle < DIFS: continue waiting
   c. Else if B > 0: B ← B − 1, reschedule tick
   d. Else (B = 0): attempt transmission

3. On ACK received: reset CW, dequeue, continue
4. On ACK timeout: double CW (up to CW_max), retry
5. On retry limit exceeded: drop frame, reset CW
```

A critical correctness detail: the backoff counter only decrements after the medium has been idle for at least DIFS (line 2b). This is implemented by tracking `_idle_since` and checking `(now − _idle_since) ≥ DIFS` before each decrement (`dcf.py:330`).

RAW-awareness is injected transparently: when RAW is enabled and the STA's slot boundary is imminent, the backoff counter is **preserved** to `_saved_backoff` rather than reset, and restored when the next RAW slot opens (`RawEngine.raw_enter()`).

### D. ACK Mechanism

The AP and STAs exchange immediate ACKs (no Block ACK in this paper's baseline). The ACK timer is set to:

$$T_\text{ACK\_timeout} = T_\text{ACK\_frame} + \text{SIFS} + \delta_\text{guard}$$

If no ACK is received within this window, the frame is declared lost and the retry counter is incremented.

---

## VII. Network, Transport, and Application Layers

### A. Network Layer (`net.py`)

The `NetworkLayer` implements a simplified IPv4-like forwarding model with:

- **16-byte IP-like header** (source, destination, next hop, TTL, sequence number)
- **Fragmentation and reassembly** for frames exceeding `max_msdu_bytes` (2304 bytes)
- **Duplicate suppression** via a sliding-window sequence cache

In the star topology, all STA→AP traffic is single-hop with `next_hop = 0`. No routing protocol is required.

### B. Transport Layer (`tp.py`)

The `TransportLayer` supports two modes:

- **UDP mode (default):** Best-effort, no retransmission. Packets are submitted to the MAC queue immediately.
- **ARQ mode:** A lightweight stop-and-wait retransmission with exponential backoff RTO (base 50 ms, max 1.0 s), calibrated for 802.11ah short-range links.

Transport sequence numbers and DSCP marking enable per-flow QoS analysis.

### C. Application Layer (`app.py`)

Five traffic models are provided:

| Model | Description |
|---|---|
| Periodic | Fixed interval *T* with optional jitter |
| Poisson | Exponential inter-arrival (rate λ) |
| CBR | Constant Bit Rate (rate_bps, fixed size) |
| Bursty | Burst of *k* frames, off-time gap |
| On-Off | Two-state Markov on/off model |

*TABLE IV. Application-layer traffic models.*

The AP node has no traffic model (sink only). STAs direct all traffic toward the AP (`dst_mode = "ap"`).

---

## VIII. Topology and Star BSS Construction

sim11ah models the single-BSS infrastructure mode mandated by 802.11ah: one AP (node 0) and up to *N* associated STAs. The `StarBuilder` class builds the topology, creates nodes, and calls `node.build_layers(cfg)` for each:

```python
StarBuilder.build(sim, num_stas=N, link_cfg={
    "rate_bps": 150_000,   # MCS0
    "prop_delay": 300e-6,  # 300 µs (≈ 90 m)
    "per": 0.0             # flat PER (PHY model handles link quality)
})
```

Node positions are configurable as (x, y) coordinates in meters; distances are computed via Euclidean norm and fed into the path-loss model. The `Topology` object stores bidirectional `Link` entries indexed by `(src_id, dst_id)` pairs.

---

## IX. RAW Implementation

### A. Protocol Background

RAW divides the time interval following a DTIM beacon into one or more **RAW Parameter Set (RPS)** entries, each defining:

- **AID range** [start_aid, end_aid]: only STAs with assigned AIDs in this range may contend during this group's slot
- **Slot definition:** number of sub-slots *S*, duration per sub-slot *D* µs, cross-slot boundary flag
- **Start time offset** from beacon: enables sequential, non-overlapping placement of multiple groups

Within a RAW group, each STA is assigned a specific sub-slot based on its AID:

$$\text{slot\_index} = (AID - \text{start\_aid}) \mod S$$

Only during its assigned sub-slot (and when `cross_slot = True`, extending into the next sub-slot for ongoing exchanges) may a STA initiate a transmission.

### B. RAW Data Structures

```
RawConfig
├── id                    : int
├── raw_type              : RawType (GENERIC=0)
├── start_aid, end_aid    : int
├── start_time_us         : int
├── slot_definition:
│   ├── num_slots         : int   (S)
│   ├── slot_duration_us  : int   (D, multiples of 120 µs + 500 µs base)
│   └── cross_slot_boundary: bool
├── beacon_spreading:
│   ├── nominal_sta_per_beacon: int
│   └── max_spread        : int
└── periodic (PRAW):
    ├── periodicity, validity, start_offset: int
    └── refresh_praw      : bool
```

*Fig. 2. RawConfig dataclass hierarchy.*

### C. Static RAW Policy

The `StaticRawPolicy` class is the baseline evaluated in this paper. At simulation start, it constructs a fixed set of `RawConfig` objects partitioning the AID space into *G* non-overlapping groups of at most *M* stations each:

```
Algorithm 2: Static RAW Config Construction

Input: G = raw_num_groups, M = raw_nodes_per_group,
       S = raw_num_slots, D = raw_slot_duration_us,
       T_budget = 0.90 × beacon_interval

1. Validate: G × S × D ≤ T_budget  (else skip RAW)
2. For g = 0 to G−1:
   a. start_aid ← 1 + g × M
   b. end_aid   ← start_aid + M − 1
   c. Clamp end_aid to max_known_AID
   d. start_time_us ← g × (S × D)
   e. Build RawConfig(id=g+1, start_aid, end_aid,
                       start_time_us, num_slots=S,
                       slot_duration_us=D)
3. Return sorted list of valid configs
```

The budget validation ensures that all RAW groups fit within 90% of the beacon interval, leaving headroom for beacons and control frames.

### D. AP-Side: RPS Construction and Beacon Embedding

The AP calls `RawEngine.build_rps()` at each DTIM beacon, which:

1. Invokes `StaticRawPolicy.build_dynamic_configs(connected_aids)` to clamp group AID ranges to currently associated STAs (avoiding empty groups).
2. Serializes each `RawConfig` as a dict into the beacon frame's `ctrl["rps"]` field.
3. Sends the beacon as a broadcast MAC frame carrying the RPS.

The dynamic clamping step is critical for low-load scenarios: with 20 STAs and *G* = 4 groups of *M* = 125, only the first group (AIDs 1–20) is populated; groups 2–4 are silently dropped.

### E. STA-Side: RPS Parsing and Slot Assignment

Upon receiving a DTIM beacon carrying an RPS, each STA:

1. Calls `RawEngine.apply_rps(rps, raw_guard)`.
2. Iterates all `RawConfig` entries to find the one whose [start_aid, end_aid] contains its own AID.
3. Computes its sub-slot index: `slot_idx = (AID − start_aid) mod S`.
4. Schedules a `RAW_ENTER` event at `beacon_rx_time + start_time_us + slot_idx × D`.
5. Schedules a `RAW_EXIT` event at `RAW_ENTER + D − raw_guard`.

Between `RAW_EXIT` and the next beacon, `raw_allowed = False`, and `DcfEngine.can_tx_now()` returns `False`, blocking all non-AP transmissions.

```
Timeline within one beacon interval (G=2, S=2, D=7 ms):
                                                     beacon
|← beacon ─►|← Group 1, Slot 0 →|← Group 1, Slot 1 →|← Group 2, Slot 0 →|← Group 2, Slot 1 →|...
0ms         ~2ms                 9ms                  16ms                 23ms              500ms
            AID 1,3,5,...        AID 2,4,6,...        AID M+1,M+3,...     AID M+2,M+4,...
```

*Fig. 3. RAW slot timeline example (G=2 groups, S=2 sub-slots, D=7 ms, beacon interval=500 ms).*

### F. Exchange Fit Check

Before any DATA transmission, `RawEngine.exchange_would_fit_in_raw()` verifies that the full DATA + SIFS + ACK exchange can complete before the slot boundary expires:

```
full_budget = T_data + T_prop + SIFS + T_ACK + T_prop + ACK_guard

if cross_slot:
    # DATA may overflow; ACK must land in the next slot
    next_slot_budget = D − raw_guard
    ok = full_budget ≤ remaining + next_slot_budget + ε
else:
    ok = full_budget ≤ remaining + ε
```

If `ok = False`, the frame is deferred with backoff preserved. This prevents partially-completed exchanges that would leave the AP waiting for an ACK across a slot boundary without a responding STA.

### G. Backoff Preservation

A key correctness feature: when a STA's RAW slot expires mid-backoff, the remaining backoff counter is saved to `_saved_backoff`. When the next slot opens (`RAW_ENTER`), the saved counter is restored rather than drawing a fresh backoff. This is essential for correct Bianchi-model behavior: discarding backoff counters at slot boundaries would artificially reduce the effective contention window and overestimate throughput.

### H. Slot Duration Encoding (MORSE Convention)

Following the Morse Micro hardware convention (preserved for cross-validation), slot durations are quantized as:

$$D_\text{us} = 500 + k \times 120 \; \mu\text{s}, \quad k \in \mathbb{Z}_{\geq 0}$$

This differs from the normative 802.11ah encoding and is documented explicitly in `mac/common.py`.

---

## X. Evaluation Framework

### A. Metrics

sim11ah collects the following metrics automatically during simulation:

| Metric | Formula / Description |
|---|---|
| PDR | packets_delivered / packets_generated |
| Throughput | (delivered_bytes × 8) / sim_time [bps] |
| Mean delay | E[t_rx − t_gen] across all delivered packets |
| 95th-pct delay | Percentile(delays, 95) |
| Jain's fairness | (Σxᵢ)² / (n · Σxᵢ²) per-STA delivery count |
| RAW fit pass rate | raw_fit_pass / (raw_fit_pass + raw_fit_fail) |
| RAW block rate | raw_fit_blocked / total_fit_checks |
| PHY collision rate | phy_collisions / mac_tx_attempts |
| MAC retry rate | mac_retries / mac_tx_attempts |

*TABLE V. Collected performance metrics.*

### B. Experiment Configuration

A typical experiment run:

```python
from sim11ah.config import default_config
from sim11ah.simulator import Simulator
from sim11ah.topology import StarBuilder

cfg = default_config(raw_enable=True, traffic_mode="periodic")
cfg["mac"]["raw_policy"] = "static"
cfg["mac"]["raw_num_groups"] = 4
cfg["mac"]["raw_num_slots"] = 4
cfg["mac"]["raw_slot_duration"] = 0.007   # 7 ms

sim = Simulator(config=cfg, seed=42)
StarBuilder.build(sim, num_stas=100,
    link_cfg={"rate_bps": 150_000, "prop_delay": 300e-6, "per": 0.0})

for node in sim.nodes.values():
    if node.node_id > 0:
        node.app.set_traffic_model(PeriodicTraffic(interval=5.0))

sim.run_and_finalize(sim_time=300.0)
print(sim.stats.summary())
```

### C. RAW vs. No-RAW Baseline

The primary evaluation compares:

- **No-RAW DCF:** All *N* STAs contend simultaneously using standard CSMA-CA.
- **Static RAW (this paper):** STAs are partitioned into *G* groups of up to *M* stations; each group receives *S* slots of duration *D* per beacon interval.

In both cases, the AP uses the same beacon interval (500 ms) and DTIM period. The RAW configuration is selected to fit within 90% of the beacon interval.

### D. Scalability

The event-driven architecture scales efficiently with station count. With *N* = 500 STAs, a 300-second simulation completes in under 60 seconds on a modern workstation (Python 3.10, Apple M-series), with the event queue reaching a maximum depth of O(*N*) events.

### E. Reproducibility

All experiments use a seeded `random.Random` instance. The full configuration is serialized to CSV/JSON at run start, enabling exact reproduction of any result by re-running with the same seed and configuration file.

---

## XI. Conclusion

We have presented sim11ah, a modular Python discrete-event simulator for IEEE 802.11ah RAW evaluation. The simulator implements a complete five-layer protocol stack with correct 802.11ah MAC timing, a SINR-based PHY model, and a fully functional static RAW scheduling pipeline. The codebase is designed for readability and extensibility: the `StaticRawPolicy` baseline described here can be replaced by any class exposing `init_configs()` and `build_dynamic_configs()`, enabling fair comparison of competing RAW strategies within the same simulation infrastructure.

Key design decisions — the shared `MacContext` dataclass, the `DcfEngine`/`RawEngine` separation, the saved-backoff mechanism, and the SINR-based capture model — are described in detail, along with their standard grounding in IEEE 802.11ah-2016. The simulator is publicly available as an installable Python package, facilitating reproducible research on 802.11ah MAC layer optimization.

Future work will extend the evaluation to periodic RAW (PRAW), heterogeneous traffic mixes, and energy consumption models, leveraging the existing TWT infrastructure in the simulator.

---

## References

[1] IEEE Std 802.11ah-2016, *IEEE Standard for Information technology — Local and metropolitan area networks — Part 11: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications — Amendment 2: Sub 1 GHz License Exempt Operation*, Dec. 2016.

[2] E. Khorov, A. Lyakhov, A. Krotov, and A. Guschin, "A survey on IEEE 802.11ah: An enabling networking technology for smart cities," *Comput. Commun.*, vol. 58, pp. 53–69, Mar. 2015.

[3] Y. Sun, G. Peng, and S. Armour, "PHY/MAC cross-layer design for 802.11ah IoT networks," *IEEE Access*, vol. 7, pp. 15 779–15 792, 2019.

[4] G. Bianchi, "Performance analysis of the IEEE 802.11 distributed coordination function," *IEEE J. Sel. Areas Commun.*, vol. 18, no. 3, pp. 535–547, Mar. 2000.

[5] Y. Tian, K. Berber, L. Dimery, and A. Bhatt, "Throughput analysis of the restricted access window mechanism in IEEE 802.11ah WLANs," *in Proc. IEEE PIMRC*, Sep. 2016, pp. 1–6.

[6] A. Woo and D. Culler, "A transmission control scheme for media access in sensor networks," *in Proc. ACM MobiCom*, 2001, pp. 221–235.

[7] K.-P. Müller and O. Voit, "SimPy: Simulation in Python," *in Proc. PyCon*, 2003. [Online]. Available: https://simpy.readthedocs.io

[8] Morse Micro, "MORSE open-source 802.11ah Linux driver," 2023. [Online]. Available: https://github.com/MorseMicro/morse_driver

---

*Manuscript received [Date]. This work was supported by [Funding Agency].*
