"""
Plot the full TII-paper evaluation (no_raw / static / adaptive / laca /
chang2019 / tasm_raw / traw), produced by scripts/eval_traw_tii_full.py.

Reads:
    results/traw_tii_full_comparison.csv   (long format: policy, n, <metric>_mean/_ci_lo/_ci_hi)

Writes PDF figures to results/figs/:
    Standalone, one metric per figure (matches the paper's fig:pdr / fig:aoi
    convention — traw_tii_paper.tex plots one metric per figure, not combined
    panels; DVR/Delay/EE/Association are tables in the paper, plotted here too
    as supplementary standalone figures):
        traw_tii_pdr_vs_n.pdf          — PDR vs N (+ 0.99 / 0.80 IIoT reliability lines)
        traw_tii_aoi_vs_n.pdf          — Mean AoI vs N (+ 3*T_rep = 15s bound line)
        traw_tii_throughput_vs_n.pdf   — Throughput vs N
        traw_tii_delay_vs_n.pdf        — Average delay vs N
        traw_tii_dvr_vs_n.pdf          — Deadline violation rate vs N
        traw_tii_energy_efficiency_vs_n.pdf — Energy efficiency vs N
        traw_tii_assoc_mean_vs_n.pdf   — Mean association time vs N
        traw_tii_assoc_max_vs_n.pdf    — Max association time vs N

    Combined multi-panel versions (handy for quick comparison, not paper-matching):
        traw_tii_performance.pdf   — PDR, Throughput, Avg Delay        (3 panels)
        traw_tii_energy_qos.pdf    — Energy Efficiency, AoI, DVR       (3 panels)
        traw_tii_association.pdf   — Mean / Max Association Time      (2 panels)

Run:
    python analysis/plot_traw_tii_results.py
"""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

ROOT     = os.path.join(os.path.dirname(__file__), "..")
RES_DIR  = os.path.join(ROOT, "results")
FIG_DIR  = os.path.join(RES_DIR, "figs")
PAPER_DIR = os.path.join(ROOT, "paper")
CSV     = os.path.join(RES_DIR, "traw_tii_full_comparison.csv")

# (display_label, marker, linestyle, color, linewidth)
# traw (proposed) drawn last / thickest so it sits on top of the other series.
POLICIES = {
    "no_raw":    ("No-RAW",    "x", ":",  "dimgray",    2.6),
    "static":    ("Static",    "s", ":",  "tab:gray",   2.6),
    "adaptive":  ("Adaptive",  "v", "-.", "tab:purple", 2.6),
    "laca":      ("LACA",      "^", "-.", "tab:green",  2.6),
    "chang2019": ("TGah",      "D", "--", "tab:orange", 2.6),
    "tasm_raw":  ("TASM-RAW",  "P", "--", "tab:brown",  2.6),
    "traw":      ("TRAW",      "o", "-",  "tab:blue",   3.4),
}

# traw_tii_paper.tex prose still says "Chang2019" throughout (not renamed to
# TGah), so the figures embedded in the paper must match that label even
# though the standalone results/figs/ versions above say "TGah".
POLICIES_PAPER = dict(POLICIES)
POLICIES_PAPER["chang2019"] = ("Chang2019",) + POLICIES["chang2019"][1:]

plt.rcParams.update({
    "font.size": 18,
    "font.family": "sans-serif",
    "axes.labelsize": 18,
    "axes.titlesize": 17,
    "axes.linewidth": 1.4,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.fontsize": 15,
    "legend.framealpha": 0.9,
    "lines.markeredgewidth": 1.3,
})


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(direction="out", length=6, width=1.3)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_axisbelow(True)


def _errorbar(ax, df, policy, metric, policies=POLICIES):
    label, marker, ls, color, lw = policies[policy]
    sub = df[df["policy"] == policy].sort_values("n")
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_ci_lo", f"{metric}_ci_hi"
    if mean_col not in sub.columns or sub.empty:
        return
    x    = sub["n"].values
    mean = sub[mean_col].values
    lo   = sub[lo_col].values if lo_col in sub.columns else mean
    hi   = sub[hi_col].values if hi_col in sub.columns else mean
    yerr = [mean - lo, hi - mean]
    ax.errorbar(
        x, mean, yerr=yerr,
        marker=marker, linestyle=ls, color=color, linewidth=lw, markersize=8.5,
        markeredgecolor="white", capsize=4.5, capthick=2.0, elinewidth=2.0, label=label,
    )


def _make_single_figure(df, metric, ylabel, out_path, ylim=None, ref_lines=None,
                         policies=POLICIES):
    """One metric, one panel, full-width — matches the paper's per-metric
    figure convention. ref_lines: list of (y_value, text_label) horizontal
    threshold lines (e.g. PDR reliability bounds, AoI deadline bound).
    Legend sits above the axes so it never collides with reference-line
    labels or data (both of which can land in any corner depending on the
    metric's trend)."""
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.6))

    if ref_lines:
        for y, text in ref_lines:
            ax.axhline(y, color="gray", linestyle=(0, (5, 3)), linewidth=1.4, alpha=0.6, zorder=1)
            ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(-6, 4), textcoords="offset points",
                        ha="right", fontsize=13, color="gray")

    for policy in policies:
        _errorbar(ax, df, policy, metric, policies=policies)

    ax.set_xlabel("Number of STAs", labelpad=8)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.99), frameon=False, columnspacing=1.4,
               handletextpad=0.5)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.80, bottom=0.13)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _make_figure(df, panels, out_path, ncols=None, height=5.2, top=0.84):
    """panels: list of (metric, ylabel, ylim_or_None)."""
    n = len(panels)
    ncols = ncols or n
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, height))
    if ncols == 1:
        axes = [axes]

    for idx, (metric, ylabel, ylim) in enumerate(panels):
        ax = axes[idx]
        for policy in POLICIES:
            _errorbar(ax, df, policy, metric)
        ax.set_xlabel("Number of STAs", labelpad=8)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        _style_axes(ax)
        ax.annotate(f"({chr(97 + idx)})", xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -68), textcoords="offset points", ha="center")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(POLICIES),
               bbox_to_anchor=(0.5, 1.06), frameon=False, columnspacing=1.4,
               handletextpad=0.5)

    fig.subplots_adjust(left=0.07, right=0.99, top=top, bottom=0.28, wspace=0.38)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    df = pd.read_csv(CSV)

    # -----------------------------------------------------------------------
    # Standalone, one-metric-per-figure (matches paper's fig:pdr / fig:aoi).
    # Also written to paper/ under the names traw_tii_paper.tex \includegraphics
    # so they replace the old hand-drawn tikz/pgfplots versions directly.
    # -----------------------------------------------------------------------
    _make_single_figure(
        df, "pdr", "Packet Delivery Ratio",
        os.path.join(FIG_DIR, "traw_tii_pdr_vs_n.pdf"),
        ylim=(0, 1.08),
        ref_lines=[(0.99, "0.99 (safety)"), (0.80, "0.80 (monitoring)")],
    )
    _make_single_figure(
        df, "pdr", "Packet Delivery Ratio",
        os.path.join(PAPER_DIR, "traw_pdr_vs_n.pdf"),
        ylim=(0, 1.08),
        ref_lines=[(0.99, "0.99 (safety)"), (0.80, "0.80 (monitoring)")],
        policies=POLICIES_PAPER,
    )
    _make_single_figure(
        df, "aoi_s", "Mean Age of Information (s)",
        os.path.join(FIG_DIR, "traw_tii_aoi_vs_n.pdf"),
        ref_lines=[(15, r"$3T_\mathrm{rep}=15$s")],
    )
    _make_single_figure(
        df, "aoi_s", "Mean Age of Information (s)",
        os.path.join(PAPER_DIR, "traw_aoi_vs_n.pdf"),
        ref_lines=[(15, r"$3T_\mathrm{rep}=15$s")],
        policies=POLICIES_PAPER,
    )
    _make_single_figure(
        df, "tput_kbps", "Throughput (kb/s)",
        os.path.join(FIG_DIR, "traw_tii_throughput_vs_n.pdf"),
    )
    _make_single_figure(
        df, "avg_delay_ms", "Average Delay (ms)",
        os.path.join(FIG_DIR, "traw_tii_delay_vs_n.pdf"),
    )
    _make_single_figure(
        df, "dvr_pct", "Deadline Violation Rate (%)",
        os.path.join(FIG_DIR, "traw_tii_dvr_vs_n.pdf"),
    )
    _make_single_figure(
        df, "ee_kbit_j", "Energy Efficiency (kbit/J)",
        os.path.join(FIG_DIR, "traw_tii_energy_efficiency_vs_n.pdf"),
    )
    _make_single_figure(
        df, "assoc_mean_s", "Mean Association Time (s)",
        os.path.join(FIG_DIR, "traw_tii_assoc_mean_vs_n.pdf"),
    )
    _make_single_figure(
        df, "assoc_max_s", "Max Association Time (s)",
        os.path.join(FIG_DIR, "traw_tii_assoc_max_vs_n.pdf"),
    )

    # -----------------------------------------------------------------------
    # Combined multi-panel versions (quick-look, not paper-matching)
    # -----------------------------------------------------------------------
    _make_figure(
        df,
        [
            ("pdr",          "Packet Delivery Ratio", (0, 1.05)),
            ("tput_kbps",    "Throughput (kb/s)",      None),
            ("avg_delay_ms", "Average Delay (ms)",     None),
        ],
        os.path.join(FIG_DIR, "traw_tii_performance.pdf"),
    )

    _make_figure(
        df,
        [
            ("ee_kbit_j", "Energy Efficiency (kbit/J)",       None),
            ("aoi_s",     "Mean Age of Information (s)",      None),
            ("dvr_pct",   "Deadline Violation Rate (%)",      None),
        ],
        os.path.join(FIG_DIR, "traw_tii_energy_qos.pdf"),
    )

    _make_figure(
        df,
        [
            ("assoc_mean_s", "Mean Association Time (s)", None),
            ("assoc_max_s",  "Max Association Time (s)",  None),
        ],
        os.path.join(FIG_DIR, "traw_tii_association.pdf"),
        ncols=2, height=4.6, top=0.80,
    )

    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
