"""
Generate the energy-consumption figure for the IEEE Networking Letters paper:
Total Energy / STA (mJ) for Static / LACA / Chang2019 / Adaptive, one panel
per sweep (deterministic-N, Poisson-N, interval), with (a)/(b)/(c) labels
drawn in (paper prose references them). Paper-specific labeled variant of
the combined energy figure in analysis/plot_ieee_letter_results.py, whose
results/figs/energy_total_combined.pdf output is left unlabeled for the
separate reviewer-response letter and is not touched here.

Reads:
    results/ieee_letter_vary_n_deterministic.csv
    results/ieee_letter_vary_n_poisson.csv
    results/ieee_letter_vary_interval.csv

Writes:
    paper/energy_total_combined.pdf
"""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

ROOT    = os.path.join(os.path.dirname(__file__), "..")
RES_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "paper")

SWEEPS = [
    {
        "name":    "vary_n_deterministic",
        "csv":     os.path.join(RES_DIR, "ieee_letter_vary_n_deterministic.csv"),
        "x_col":   "num_stas",
        "x_label": "Number of STAs",
        "title":   "Deterministic (5 s)",
    },
    {
        "name":    "vary_n_poisson",
        "csv":     os.path.join(RES_DIR, "ieee_letter_vary_n_poisson.csv"),
        "x_col":   "num_stas",
        "x_label": "Number of STAs",
        "title":   "Poisson (mean 5 s)",
    },
    {
        "name":    "vary_interval",
        "csv":     os.path.join(RES_DIR, "ieee_letter_vary_interval.csv"),
        "x_col":   "pkt_interval_s",
        "x_label": "Packet Interval (s)",
        "title":   "N = 500",
    },
]

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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(SWEEPS)
    fig, axes = plt.subplots(1, n, figsize=(13.5, 4.6))

    for idx, sweep in enumerate(SWEEPS):
        df = pd.read_csv(sweep["csv"]).sort_values(by=sweep["x_col"])
        ax = axes[idx]
        for policy, (label, marker, ls, color) in POLICIES.items():
            _errorbar(ax, df, sweep["x_col"], policy, "e_total_mj", marker, ls, color, label)
        ax.set_xlabel(sweep["x_label"], labelpad=8)
        ax.set_ylabel("Total Energy / STA (mJ)")
        ax.set_title(sweep["title"], fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.annotate(f"({chr(97 + idx)})", xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -55), textcoords="offset points", ha="center")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), frameon=False)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.26, wspace=0.32)
    out = os.path.join(OUT_DIR, "energy_total_combined.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
