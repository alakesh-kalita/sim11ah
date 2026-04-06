#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


ROOT_NODE = 0
PACKET_SIZE_BYTES = 128


# ------------------------------------------------------------
# Plot style
# ------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
})


# ------------------------------------------------------------
# Data containers
# ------------------------------------------------------------

@dataclass
class NodeStats:
    node_id: int
    generated: int = 0
    delivered: int = 0
    drops: int = 0
    delays: List[float] = field(default_factory=list)

    @property
    def pdr_percent(self) -> float:
        return (100.0 * self.delivered / self.generated) if self.generated > 0 else 0.0

    @property
    def avg_delay(self) -> float:
        return sum(self.delays) / len(self.delays) if self.delays else 0.0

    @property
    def max_delay(self) -> float:
        return max(self.delays) if self.delays else 0.0

    @property
    def min_delay(self) -> float:
        return min(self.delays) if self.delays else 0.0

    @property
    def p95_delay(self) -> float:
        return percentile(self.delays, 0.95)


class ResultsProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = os.path.abspath(csv_path)
        self.results_dir = create_results_dir(self.csv_path)

    def run(self) -> str:
        print(f"Processing {self.csv_path}")
        print(f"Results directory: {self.results_dir}")

        nodes, summary, debug = analyze_log(self.csv_path)

        save_debug_report(debug, os.path.join(self.results_dir, "debug_report.txt"))
        save_node_summary_csv(nodes, summary["time"], os.path.join(self.results_dir, "node_summary.csv"))
        save_summary_csv(summary, os.path.join(self.results_dir, "overall_summary.csv"))

        print(f"OVERALL_E2E_PDR_PERCENT = {summary['pdr_percent']:.2f}")
        print(f"TX_COUNT = {summary['tx']}")
        print(f"RX_COUNT = {summary['rx']}")
        print(f"DROP_COUNT = {summary['drop']}")
        print(f"AVG_LATENCY_S = {summary['lat_avg']:.6f}")
        print(f"P95_LATENCY_S = {summary['lat_p95']:.6f}")
        print(f"MAX_LATENCY_S = {summary['lat_max']:.6f}")
        print(f"MIN_LATENCY_S = {summary['lat_min']:.6f}")
        print(f"THROUGHPUT_BPS = {summary['thr_bps']:.3f}")
        print(f"FAIRNESS_INDEX = {summary['fairness']:.6f}")
        print(f"SIM_TIME_S = {summary['time']:.6f}")

        node_ids = sorted(nodes.keys())
        per_node_generated = [nodes[n].generated for n in node_ids]
        per_node_delivered = [nodes[n].delivered for n in node_ids]
        per_node_pdr_percent = [nodes[n].pdr_percent for n in node_ids]
        per_node_avg_delay = [nodes[n].avg_delay for n in node_ids]
        per_node_p95_delay = [nodes[n].p95_delay for n in node_ids]
        per_node_thr = [
            (nodes[n].delivered * PACKET_SIZE_BYTES * 8) / max(summary["time"], 1e-12)
            for n in node_ids
        ]

        plot_bar(
            node_ids,
            per_node_generated,
            "Node ID",
            "Packets",
            "Per-Node Generated Packets",
            os.path.join(self.results_dir, "per_node_generated.png"),
        )

        plot_bar(
            node_ids,
            per_node_delivered,
            "Node ID",
            "Packets",
            "Per-Node Delivered Packets",
            os.path.join(self.results_dir, "per_node_delivered.png"),
        )

        plot_bar(
            node_ids,
            per_node_pdr_percent,
            "Node ID",
            "PDR (%)",
            "Per-Node PDR",
            os.path.join(self.results_dir, "per_node_pdr.png"),
        )

        plot_bar(
            node_ids,
            per_node_thr,
            "Node ID",
            "Throughput (bps)",
            "Per-Node Throughput",
            os.path.join(self.results_dir, "per_node_throughput.png"),
        )

        plot_bar(
            node_ids,
            per_node_avg_delay,
            "Node ID",
            "Delay (s)",
            "Per-Node Average End-to-End Delay",
            os.path.join(self.results_dir, "per_node_avg_delay.png"),
        )

        plot_bar(
            node_ids,
            per_node_p95_delay,
            "Node ID",
            "Delay (s)",
            "Per-Node 95th Percentile End-to-End Delay",
            os.path.join(self.results_dir, "per_node_p95_delay.png"),
        )

        plot_cdf(
            [v for v in per_node_thr if v > 0],
            "Throughput (bps)",
            "CDF of Per-Node Throughput",
            os.path.join(self.results_dir, "cdf_per_node_throughput.png"),
        )

        plot_cdf(
            [v for v in per_node_pdr_percent if v >= 0],
            "PDR (%)",
            "CDF of Per-Node PDR",
            os.path.join(self.results_dir, "cdf_per_node_pdr.png"),
        )

        all_delays: List[float] = []
        for node_id in node_ids:
            all_delays.extend(nodes[node_id].delays)

        plot_cdf(
            all_delays,
            "End-to-End Delay (s)",
            "CDF of End-to-End Delay",
            os.path.join(self.results_dir, "cdf_e2e_delay.png"),
        )

        plot_histogram(
            all_delays,
            "End-to-End Delay (s)",
            "Count",
            "Distribution of End-to-End Delay",
            os.path.join(self.results_dir, "hist_e2e_delay.png"),
        )

        plot_single_bar(
            summary["pdr_percent"],
            "PDR (%)",
            "Overall End-to-End PDR",
            os.path.join(self.results_dir, "overall_e2e_pdr.png"),
            ylim=(0, 100),
        )

        plot_bar(
            ["TX", "RX", "Drop"],
            [summary["tx"], summary["rx"], summary["drop"]],
            "",
            "Packets",
            "TX / RX / Drop",
            os.path.join(self.results_dir, "tx_rx_drop.png"),
            rotation=0,
        )

        plot_bar(
            ["Min", "Avg", "P95", "Max"],
            [
                summary["lat_min"],
                summary["lat_avg"],
                summary["lat_p95"],
                summary["lat_max"],
            ],
            "",
            "Delay (s)",
            "Overall End-to-End Delay Summary",
            os.path.join(self.results_dir, "latency_summary.png"),
            rotation=0,
        )

        plot_single_bar(
            summary["thr_bps"],
            "bps",
            "Overall Throughput",
            os.path.join(self.results_dir, "overall_throughput.png"),
        )

        plot_single_bar(
            summary["fairness"],
            "Fairness Index",
            "Jain Fairness Index",
            os.path.join(self.results_dir, "fairness_index.png"),
            ylim=(0, 1.05),
        )

        plot_dashboard(summary, self.results_dir)

        print("\nDone.")
        print("Generated files are inside:")
        print(self.results_dir)
        return self.results_dir


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def select_log_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.update()

    filename = filedialog.askopenfilename(
        title="Select simulator CSV log file",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not filename:
        print("No file selected. Exiting.")
        raise SystemExit(1)

    return filename


def create_results_dir(input_file: str) -> str:
    csv_dir = os.path.dirname(os.path.abspath(input_file))
    csv_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(csv_dir, f"{csv_name}_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_dict_like(s: Any) -> Dict[str, Any]:
    if s is None or s == "":
        return {}
    if isinstance(s, dict):
        return s
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def percentile(values: List[float], p: float) -> float:
    clean = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not clean:
        return 0.0
    idx = max(0, min(len(clean) - 1, math.ceil(p * len(clean)) - 1))
    return float(clean[idx])


def jain_fairness(values: List[float]) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    if not clean:
        return 0.0
    s1 = sum(clean)
    s2 = sum(v * v for v in clean)
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / (len(clean) * s2)


def sanitize_plot_values(values: List[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def save_plot(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_bar(
    x_labels: List[Any],
    values: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    rotation: int = 90,
) -> None:
    values = sanitize_plot_values(values)
    if not values:
        print(f"Skipping empty plot: {os.path.basename(filename)}")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(range(len(values)), values, edgecolor="black")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(x) for x in x_labels], rotation=rotation)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, filename)


def plot_single_bar(value: float, ylabel: str, title: str, filename: str, ylim=None) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        print(f"Skipping invalid single-bar plot: {os.path.basename(filename)}")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([title], [value], edgecolor="black")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    save_plot(fig, filename)


def plot_cdf(values: List[float], xlabel: str, title: str, filename: str) -> None:
    clean = sanitize_plot_values(values)
    clean = [v for v in clean if v >= 0]

    if not clean:
        print(f"Skipping empty plot: {os.path.basename(filename)}")
        return

    clean.sort()
    y = [(i + 1) / len(clean) for i in range(len(clean))]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(clean, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)


def plot_histogram(values: List[float], xlabel: str, ylabel: str, title: str, filename: str, bins: int = 30) -> None:
    clean = sanitize_plot_values(values)
    if not clean:
        print(f"Skipping empty plot: {os.path.basename(filename)}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(clean, bins=bins, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    save_plot(fig, filename)


def plot_dashboard(summary: Dict[str, float], results_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(["E2E PDR"], [summary["pdr_percent"]], edgecolor="black")
    axes[0, 0].set_ylabel("PDR (%)")
    axes[0, 0].set_title("Overall End-to-End PDR")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].bar(
        ["TX", "RX", "Drop"],
        [summary["tx"], summary["rx"], summary["drop"]],
        edgecolor="black"
    )
    axes[0, 1].set_ylabel("Packets")
    axes[0, 1].set_title("Packet Summary")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].bar(
        ["Min", "Avg", "P95", "Max"],
        [
            summary["lat_min"],
            summary["lat_avg"],
            summary["lat_p95"],
            summary["lat_max"],
        ],
        edgecolor="black",
    )
    axes[1, 0].set_ylabel("Delay (s)")
    axes[1, 0].set_title("Overall End-to-End Delay")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    axes[1, 1].bar(
        ["Throughput", "Fairness"],
        [summary["thr_bps"], summary["fairness"]],
        edgecolor="black",
    )
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title(f"Throughput / Fairness (Sim time: {summary['time']:.2f}s)")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    save_plot(fig, os.path.join(results_dir, "overall_dashboard.png"))


# ------------------------------------------------------------
# Save reports
# ------------------------------------------------------------

def save_debug_report(debug: Dict[str, Any], out_file: str) -> None:
    with open(out_file, "w", encoding="utf-8") as f:
        for k, v in debug.items():
            f.write(f"{k}: {v}\n")


def save_node_summary_csv(nodes: Dict[int, NodeStats], sim_time: float, out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_id",
            "generated",
            "delivered",
            "drops",
            "pdr_percent",
            "throughput_bps",
            "avg_delay_s",
            "p95_delay_s",
            "min_delay_s",
            "max_delay_s",
        ])

        for node_id in sorted(nodes.keys()):
            ns = nodes[node_id]
            thr_bps = (ns.delivered * PACKET_SIZE_BYTES * 8) / max(sim_time, 1e-12)
            writer.writerow([
                node_id,
                ns.generated,
                ns.delivered,
                ns.drops,
                ns.pdr_percent,
                thr_bps,
                ns.avg_delay,
                ns.p95_delay,
                ns.min_delay,
                ns.max_delay,
            ])


def save_summary_csv(summary: Dict[str, Any], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------

def analyze_log(filename: str):
    nodes: Dict[int, NodeStats] = {}

    tx = 0
    rx = 0
    drop = 0
    delays: List[float] = []

    sim_time = 0.0
    total_rows = 0
    matched_rows = 0

    app_generate_rows = 0
    app_deliver_rows = 0
    app_tx_failure_rows = 0
    mac_drop_rows = 0

    def get_node(nid: int) -> NodeStats:
        if nid not in nodes:
            nodes[nid] = NodeStats(nid)
        return nodes[nid]

    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"time", "node_id", "layer", "event", "details"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1

            time_s = safe_float(row.get("time"), 0.0) or 0.0
            node_id = safe_int(row.get("node_id"))
            layer = str(row.get("layer", "")).strip()
            event = str(row.get("event", "")).strip()
            details = parse_dict_like(row.get("details"))

            packet = {}
            if "packet" in row:
                packet = parse_dict_like(row.get("packet"))

            sim_time = max(sim_time, time_s)

            if node_id is None or not layer or not event:
                continue

            matched_rows += 1
            ns = get_node(node_id)

            if layer == "APP" and event == "GENERATE":
                tx += 1
                app_generate_rows += 1
                ns.generated += 1

            elif layer == "APP" and event == "DELIVER":
                rx += 1
                app_deliver_rows += 1

                dst = None
                if packet:
                    dst = packet.get("dst", None)

                if dst is None:
                    dst = details.get("dst", None)

                if dst is None:
                    dst = node_id

                dst_int = safe_int(dst)
                if dst_int is not None:
                    dst_ns = get_node(dst_int)
                    dst_ns.delivered += 1

                    d = details.get("delay_s", None)
                    d = safe_float(d)
                    if d is not None and d >= 0:
                        delays.append(d)
                        dst_ns.delays.append(d)

            elif layer == "APP" and event == "TX_FAILURE":
                drop += 1
                app_tx_failure_rows += 1
                ns.drops += 1

            elif layer == "MAC" and event == "DROP":
                drop += 1
                mac_drop_rows += 1
                ns.drops += 1

    lat_avg = sum(delays) / len(delays) if delays else 0.0
    lat_max = max(delays) if delays else 0.0
    lat_min = min(delays) if delays else 0.0
    lat_p95 = percentile(delays, 0.95)

    thr_bps = (rx * PACKET_SIZE_BYTES * 8) / max(sim_time, 1e-12)
    pdr = (rx / tx) if tx > 0 else 0.0
    pdr_percent = pdr * 100.0

    per_node_thr = [
        (nodes[n].delivered * PACKET_SIZE_BYTES * 8) / max(sim_time, 1e-12)
        for n in sorted(nodes.keys())
    ]
    fairness = jain_fairness(per_node_thr)

    summary = {
        "time": sim_time,
        "tx": tx,
        "rx": rx,
        "drop": drop,
        "pdr": pdr,
        "pdr_percent": pdr_percent,
        "lat_avg": lat_avg,
        "lat_p95": lat_p95,
        "lat_max": lat_max,
        "lat_min": lat_min,
        "thr_bps": thr_bps,
        "fairness": fairness,
    }

    debug = {
        "input_file": filename,
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "num_nodes": len(nodes),
        "root_node": ROOT_NODE,
        "packet_size_bytes": PACKET_SIZE_BYTES,
        "sim_time_s": sim_time,
        "tx": tx,
        "rx": rx,
        "drop": drop,
        "pdr_percent": pdr_percent,
        "avg_latency_s": lat_avg,
        "p95_latency_s": lat_p95,
        "max_latency_s": lat_max,
        "min_latency_s": lat_min,
        "throughput_bps": thr_bps,
        "fairness_index": fairness,
        "app_generate_rows": app_generate_rows,
        "app_deliver_rows": app_deliver_rows,
        "app_tx_failure_rows": app_tx_failure_rows,
        "mac_drop_rows": mac_drop_rows,
    }

    return nodes, summary, debug


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        input_file = os.path.abspath(sys.argv[1])
    else:
        input_file = select_log_file()

    processor = ResultsProcessor(input_file)
    processor.run()


if __name__ == "__main__":
    main()