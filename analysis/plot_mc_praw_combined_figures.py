"""
MC-PRAW "combined" figures, matching the earlier TRAW-paper style seen in
results/figs/assoc_mean_combined.pdf (analysis/plot_ieee_letter_results.py):
one figure per metric, one titled panel per sweep, in-panel legend (not a
shared external one), clean spines, white-edged error-bar markers.

Three sweeps, all 3-seed CI, all 7 MC-PRAW policies (Adaptive excluded,
No-RAW first, clusters last -- matching plot_mc_praw_figures.py):
  (a) vs N, deterministic/periodic traffic, interval=5s  (uav_assoc_energy_vs_n.csv)
  (b) vs N, Poisson traffic, mean interval=5s             (uav_assoc_energy_vs_n_poisson.csv,
                                                             a new campaign run via
                                                             eval_uav_poisson.py -- this
                                                             sweep didn't exist before)
  (c) vs packet interval, N=500, deterministic traffic    (uav_assoc_energy_vs_interval.csv)

Output directory: results/figs/MC-PRAW/
"""
import os

import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figs", "MC-PRAW")

POLICIES = [
    ("no_raw",           "No-RAW",            "#4a3aa7", "x", ":"),
    ("static",           "Static",            "#2a78d6", "s", ":"),
    ("etaroa",           "E-TAROA",           "#1a1a1a", "*", "-"),
    ("laca",             "LACA",              "#e87ba4", "v", "--"),
    ("chang2019",        "TGah",              "#008300", "P", "-."),
    ("cluster_csv",      "Cluster-based",     "#1baf7a", "^", "-."),
    ("cluster_adaptive", "MA-PRAW",           "#eda100", "D", "-"),
]

SWEEPS = [
    {"csv": "uav_assoc_energy_vs_n.csv", "x_col": "num_stas",
     "x_label": "Number of UAVs (N)", "title": "Deterministic Traffic (5 s interval)"},
    {"csv": "uav_assoc_energy_vs_n_poisson.csv", "x_col": "num_stas",
     "x_label": "Number of UAVs (N)", "title": "Poisson Traffic (mean 5 s interval)"},
    {"csv": "uav_assoc_energy_vs_interval.csv", "x_col": "packet_interval",
     "x_label": "Packet Interval (s)", "title": "N = 500 UAVs, Deterministic Traffic"},
]

plt.rcParams.update({
    "font.size": 23,
    "font.family": "sans-serif",
    "axes.labelsize": 23,
    "axes.titlesize": 20,
    "axes.linewidth": 1.4,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.fontsize": 19,
    "legend.framealpha": 0.9,
    "lines.markeredgewidth": 1.4,
})


def load(path):
    import csv
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["policy"], []).append(r)
    return rows


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(direction="out", length=6, width=1.3)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_axisbelow(True)


def _errorbar(ax, rows_by_policy, x_col, key, metric, marker, ls, color, label):
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_ci_lo", f"{metric}_ci_hi"
    rows = sorted(rows_by_policy.get(key, []), key=lambda r: float(r[x_col]))
    if not rows or mean_col not in rows[0]:
        return
    x = [float(r[x_col]) for r in rows]
    mean = [float(r[mean_col]) for r in rows]
    lo = [float(r[lo_col]) for r in rows]
    hi = [float(r[hi_col]) for r in rows]
    yerr = [[max(0, m - l) for m, l in zip(mean, lo)], [max(0, h - m) for m, h in zip(mean, hi)]]
    ax.errorbar(
        x, mean, yerr=yerr, marker=marker, linestyle=ls, color=color,
        linewidth=3.2, markersize=9, markeredgecolor="white",
        capsize=5, capthick=2.2, elinewidth=2.2, label=label,
    )


def make_combined_metric_figure(sweeps_with_rows, metric, ylabel, ylim, out_name, show_title=True):
    n = len(sweeps_with_rows)
    # A real legend AXIS as the leftmost gridspec column, not a floating
    # fig.legend() -- bbox_to_anchor + bbox_inches="tight" fought each
    # other and squeezed the data panels into overlapping each other.
    # An actual subplot column has a defined width the layout respects.
    fig, axes = plt.subplots(
        1, n + 1, figsize=(4.3 + 5.6 * n, 5.8),
        gridspec_kw={"width_ratios": [0.72] + [1.0] * n},
    )
    legend_ax, data_axes = axes[0], axes[1:]
    legend_ax.axis("off")

    for idx, (sweep, rows_by_policy) in enumerate(sweeps_with_rows):
        ax = data_axes[idx]
        for key, label, color, marker, ls in POLICIES:
            _errorbar(ax, rows_by_policy, sweep["x_col"], key, metric, marker, ls, color, label)
        ax.set_xlabel(sweep["x_label"], labelpad=8)
        if idx == 0:
            ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if show_title:
            ax.set_title(sweep["title"], fontsize=18, pad=12)
        _style_axes(ax)
        ax.annotate(f"({chr(97 + idx)})", xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -90), textcoords="offset points", ha="center")

    legend_handles = [
        plt.Line2D([0], [0], color=color, marker=marker, linestyle=ls,
                   linewidth=3, markersize=9, markeredgewidth=1.4, label=label)
        for _, label, color, marker, ls in POLICIES
    ]
    legend_ax.legend(legend_handles, [h.get_label() for h in legend_handles],
                      loc="center", frameon=True, framealpha=0.9)

    top = 0.96 if not show_title else 0.87
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=0.34, wspace=0.34)
    out_path = os.path.join(FIGS, out_name)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def main():
    os.makedirs(FIGS, exist_ok=True)

    sweeps_with_rows = []
    for sweep in SWEEPS:
        path = os.path.join(RESULTS, sweep["csv"])
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        sweeps_with_rows.append((sweep, load(path)))

    make_combined_metric_figure(
        sweeps_with_rows, "pdr", "Packet Delivery Ratio", (0, 1.05),
        "fig_mc_praw_pdr_combined.pdf",
    )
    make_combined_metric_figure(
        sweeps_with_rows, "tput_kbps", "Throughput (kbps)", None,
        "fig_mc_praw_tput_combined.pdf",
    )
    make_combined_metric_figure(
        sweeps_with_rows, "avg_delay_ms", "Average Delay (ms)", None,
        "fig_mc_praw_delay_combined.pdf",
    )
    make_combined_metric_figure(
        sweeps_with_rows, "assoc_mean_ms", "Mean Association Time (ms)", None,
        "fig_mc_praw_assoc_mean_combined.pdf",
    )
    make_combined_metric_figure(
        sweeps_with_rows, "assoc_frac", "Association Success Fraction", (0, 1.08),
        "fig_mc_praw_assoc_frac_combined.pdf",
    )
    make_combined_metric_figure(
        sweeps_with_rows, "e_total_mj", "Total Energy / STA (mJ)", None,
        "fig_mc_praw_energy_total_combined.pdf",
    )
    make_combined_metric_figure(
        sweeps_with_rows, "ee_kbit_j", "Energy Efficiency (kbit/J)", None,
        "fig_mc_praw_energy_eff_combined.pdf",
    )

    print(f"\nDone. Figures in {FIGS}")


if __name__ == "__main__":
    main()
