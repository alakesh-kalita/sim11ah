"""
Generate the 3 main-text figures for the IEEE Networking Letters paper using
the FULL 4-policy comparison (Static / LACA / Chang2019 / Adaptive), with
(a)/(b)/(c) labels drawn in (needed since the paper's prose references
Fig.~X(a) etc.). This is a paper-specific variant of
analysis/plot_ieee_letter_results.py's per-sweep performance figure --
that script's output in results/figs/ is intentionally left unlabeled for
the separate reviewer-response letter and is not touched here.

Reads:
    results/ieee_letter_vary_n_deterministic.csv
    results/ieee_letter_vary_n_poisson.csv
    results/ieee_letter_vary_interval.csv

Writes to paper/ (matching the filenames already referenced in main_new.tex):
    perf_vary_n_deterministic.pdf
    perf_vary_n_poisson.pdf
    perf_vary_interval.pdf
"""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
RES_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "paper")

POLICIES = {
    "static":    ("Static",    "s", ":",  "tab:gray"),
    "laca":      ("LACA",      "^", "-.", "tab:green"),
    "chang2019": ("TGah",      "D", "--", "tab:orange"),
    "adaptive":  ("Adaptive",  "o", "-",  "tab:blue"),
}

plt.rcParams.update({
    "font.size": 17,
    "axes.labelsize": 17,
    "axes.titlesize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
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
        marker=marker, linestyle=ls, color=color, linewidth=1.8, markersize=5.5,
        capsize=2.5, label=label,
    )


def make_figure(csv_path, x_col, x_label, out_path):
    df = pd.read_csv(csv_path).sort_values(by=x_col)

    panels = [
        ("pdr",          "PDR",             (0, 1.05)),
        ("tput_kbps",    "Throughput (kb/s)", None),
        ("avg_delay_ms", "Delay (ms)",        None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    for idx, (metric, ylabel, ylim) in enumerate(panels):
        ax = axes[idx]
        for policy, (label, marker, ls, color) in POLICIES.items():
            _errorbar(ax, df, x_col, policy, metric, marker, ls, color, label)
        ax.set_xlabel(x_label, labelpad=8)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.annotate(f"({chr(97 + idx)})", xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -55), textcoords="offset points", ha="center")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), frameon=False)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.26, wspace=0.32)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    make_figure(
        os.path.join(RES_DIR, "ieee_letter_vary_n_deterministic.csv"),
        "num_stas", "Number of STAs",
        os.path.join(OUT_DIR, "perf_vary_n_deterministic.pdf"),
    )
    make_figure(
        os.path.join(RES_DIR, "ieee_letter_vary_n_poisson.csv"),
        "num_stas", "Number of STAs",
        os.path.join(OUT_DIR, "perf_vary_n_poisson.pdf"),
    )
    make_figure(
        os.path.join(RES_DIR, "ieee_letter_vary_interval.csv"),
        "pkt_interval_s", "Packet Interval (s)",
        os.path.join(OUT_DIR, "perf_vary_interval.pdf"),
    )


if __name__ == "__main__":
    main()
