"""
Plot IEEE Networking Letters results (static / laca / chang2019 / adaptive).

Reads the aggregated CSVs produced by scripts/ieee_networking_letter_adaptive.py:
    results/ieee_letter_vary_n_deterministic.csv
    results/ieee_letter_vary_n_poisson.csv
    results/ieee_letter_vary_interval.csv

Writes PDF figures to results/figs/:
    perf_vary_n_deterministic.pdf   (PDR, throughput, delay vs N; periodic traffic)
    perf_vary_n_poisson.pdf         (PDR, throughput, delay vs N; Poisson traffic)
    perf_vary_interval.pdf          (PDR, throughput, delay vs packet interval; N=500)
    energy_total_combined.pdf       (Total Energy / STA; one panel per sweep)
    assoc_mean_combined.pdf         (Mean Association Time; one panel per sweep)

Run after scripts/ieee_networking_letter_adaptive.py has produced the CSVs:
    python analysis/plot_ieee_letter_results.py
"""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

ROOT     = os.path.join(os.path.dirname(__file__), "..")
RES_DIR  = os.path.join(ROOT, "results")
FIG_DIR  = os.path.join(RES_DIR, "figs")

SWEEPS = [
    {
        "name":    "vary_n_deterministic",
        "csv":     os.path.join(RES_DIR, "ieee_letter_vary_n_deterministic.csv"),
        "x_col":   "num_stas",
        "x_label": "Number of STAs",
        "title":   "Deterministic Traffic (5 s interval)",
    },
    {
        "name":    "vary_n_poisson",
        "csv":     os.path.join(RES_DIR, "ieee_letter_vary_n_poisson.csv"),
        "x_col":   "num_stas",
        "x_label": "Number of STAs",
        "title":   "Poisson Traffic (mean 5 s interval)",
    },
    {
        "name":    "vary_interval",
        "csv":     os.path.join(RES_DIR, "ieee_letter_vary_interval.csv"),
        "x_col":   "pkt_interval_s",
        "x_label": "Packet Interval (s)",
        "title":   "N = 500 STAs, Deterministic Traffic",
    },
]

POLICIES = {
    "static":    ("Static",    "s", ":",  "tab:gray"),
    "laca":      ("LACA",      "^", "-.", "tab:green"),
    "chang2019": ("TGah",      "D", "--", "tab:orange"),
    "adaptive":  ("Adaptive",  "o", "-",  "tab:blue"),
}

plt.rcParams.update({
    "font.size": 19,
    "font.family": "sans-serif",
    "axes.labelsize": 19,
    "axes.titlesize": 18,
    "axes.linewidth": 1.4,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.fontsize": 16,
    "legend.framealpha": 0.9,
    "lines.markeredgewidth": 1.4,
})


def _style_axes(ax):
    """Shared professional styling: clean spines, outward ticks, subtle grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(direction="out", length=6, width=1.3)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_axisbelow(True)


def _errorbar(ax, df, x_col, policy, metric, marker, linestyle, color, label):
    mean_col = f"{policy}_{metric}_mean"
    lo_col   = f"{policy}_{metric}_ci_low"
    hi_col   = f"{policy}_{metric}_ci_hi"
    if mean_col not in df.columns:
        print(f"  [skip] missing column {mean_col}")
        return
    x    = df[x_col].values
    mean = df[mean_col].values
    lo   = df[lo_col].values if lo_col in df.columns else mean
    hi   = df[hi_col].values if hi_col in df.columns else mean
    yerr = [mean - lo, hi - mean]
    ax.errorbar(
        x, mean, yerr=yerr,
        marker=marker, linestyle=linestyle, color=color, linewidth=3.2, markersize=9,
        markeredgecolor="white", capsize=5, capthick=2.2, elinewidth=2.2, label=label,
    )


def _panel_figure(df, x_col, x_label, panels, out_path):
    """panels: list of (metric, ylabel, ylim_or_None). No figure-level heading."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6))
    if n == 1:
        axes = [axes]

    for idx, (metric, ylabel, ylim) in enumerate(panels):
        ax = axes[idx]
        for policy, (label, marker, linestyle, color) in POLICIES.items():
            _errorbar(ax, df, x_col, policy, metric, marker, linestyle, color, label)
        ax.set_xlabel(x_label, labelpad=8)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        _style_axes(ax)
        ax.legend()
        ax.annotate(f"({chr(97 + idx)})", xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -70), textcoords="offset points", ha="center")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.30, wspace=0.32)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_perf_figure(sweep, df):
    panels = [
        ("pdr",          "Packet Delivery Ratio",   (0, 1.05)),
        ("tput_kbps",    "Throughput (kb/s)",       None),
        ("avg_delay_ms", "Average Delay (ms)",      None),
    ]
    out = os.path.join(FIG_DIR, f"perf_{sweep['name']}.pdf")
    _panel_figure(df, sweep["x_col"], sweep["x_label"], panels, out)


def make_combined_metric_figure(sweeps_with_df, metric, ylabel, ylim, out_name, show_title=True):
    """One figure, one panel per sweep, all showing the same single metric."""
    n = len(sweeps_with_df)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.0))
    if n == 1:
        axes = [axes]

    for idx, (sweep, df) in enumerate(sweeps_with_df):
        ax = axes[idx]
        for policy, (label, marker, linestyle, color) in POLICIES.items():
            _errorbar(ax, df, sweep["x_col"], policy, metric, marker, linestyle, color, label)
        ax.set_xlabel(sweep["x_label"], labelpad=8)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if show_title:
            ax.set_title(sweep["title"], fontsize=15, pad=12)
        _style_axes(ax)
        ax.legend()
        ax.annotate(f"({chr(97 + idx)})", xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -70), textcoords="offset points", ha="center")

    top = 0.96 if not show_title else 0.84
    fig.subplots_adjust(left=0.09, right=0.99, top=top, bottom=0.30, wspace=0.45)
    out = os.path.join(FIG_DIR, out_name)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    sweeps_with_df = []
    for sweep in SWEEPS:
        if not os.path.exists(sweep["csv"]):
            print(f"[skip] {sweep['csv']} not found — run "
                  f"scripts/ieee_networking_letter_adaptive.py first")
            continue
        df = pd.read_csv(sweep["csv"]).sort_values(by=sweep["x_col"])
        sweeps_with_df.append((sweep, df))

    print("\nPer-sweep performance figures:")
    for sweep, df in sweeps_with_df:
        make_perf_figure(sweep, df)

    print("\nCombined energy figure:")
    make_combined_metric_figure(
        sweeps_with_df, "e_total_mj", "Total Energy / STA (mJ)", None,
        "energy_total_combined.pdf", show_title=False,
    )

    print("\nCombined association figure:")
    make_combined_metric_figure(
        sweeps_with_df, "assoc_mean_ms", "Mean Association Time (ms)", None,
        "assoc_mean_combined.pdf",
    )

    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
