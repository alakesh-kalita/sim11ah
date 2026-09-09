"""
Generate the 3 main-text figures (station.pdf, poison.pdf, interval.pdf) for
the IEEE Networking Letters adaptive-RAW paper: Adaptive vs. Static only,
PDR / throughput / delay, matching the paper's Fig. 2-4 convention with
(a)/(b)/(c) labels drawn into the image (single raster figure per Fig.,
no LaTeX subfigure environment).

Reads the same sweep CSVs used for the reviewer-response tables:
    results/ieee_letter_vary_n_deterministic.csv
    results/ieee_letter_vary_n_poisson.csv
    results/ieee_letter_vary_interval.csv

Writes to paper/:
    station.pdf, poison.pdf, interval.pdf
"""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
RES_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "paper")

POLICIES = {
    "adaptive": ("Adaptive", "o", "-",  "tab:blue"),
    "static":   ("Static",   "s", ":",  "tab:orange"),
}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def _errorbar(ax, df, x_col, policy, metric, marker, ls, color, label):
    mean_col = f"{policy}_{metric}_mean"
    lo_col   = f"{policy}_{metric}_ci_low"
    hi_col   = f"{policy}_{metric}_ci_hi"
    x    = df[x_col].values
    mean = df[mean_col].values
    lo   = df[lo_col].values if lo_col in df.columns else mean
    hi   = df[hi_col].values if hi_col in df.columns else mean
    yerr = [mean - lo, hi - mean]
    ax.errorbar(
        x, mean, yerr=yerr,
        marker=marker, linestyle=ls, color=color, linewidth=2, markersize=6,
        capsize=3, label=label,
    )


def make_figure(csv_path, x_col, x_label, out_path):
    df = pd.read_csv(csv_path).sort_values(by=x_col)

    panels = [
        ("pdr",          "PDR",             (0, 1.05)),
        ("tput_kbps",    "Throughput (kb/s)", None),
        ("avg_delay_ms", "Delay (ms)",        None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    for idx, (metric, ylabel, ylim) in enumerate(panels):
        ax = axes[idx]
        for policy, (label, marker, ls, color) in POLICIES.items():
            _errorbar(ax, df, x_col, policy, metric, marker, ls, color, label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        ax.text(0.5, -0.34, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    make_figure(
        os.path.join(RES_DIR, "ieee_letter_vary_n_deterministic.csv"),
        "num_stas", "Number of STAs",
        os.path.join(OUT_DIR, "station.pdf"),
    )
    make_figure(
        os.path.join(RES_DIR, "ieee_letter_vary_n_poisson.csv"),
        "num_stas", "Number of STAs",
        os.path.join(OUT_DIR, "poison.pdf"),
    )
    make_figure(
        os.path.join(RES_DIR, "ieee_letter_vary_interval.csv"),
        "pkt_interval_s", "Packet Interval (s)",
        os.path.join(OUT_DIR, "interval.pdf"),
    )


if __name__ == "__main__":
    main()
