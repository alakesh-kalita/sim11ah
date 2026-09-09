"""
Updated UAV-paper figures (post Cluster-Adaptive beacon-budget-clamp fix).

Regenerates the PDR/Throughput/Delay comparison figures (fig_uav_core4_vs_*)
from the new 3-seed CSVs (results/uav_assoc_energy_vs_interval.csv,
results/uav_assoc_energy_vs_n.csv) with 95% CI error bars, plus two new
figures covering association completeness and energy efficiency -- metrics
that were never plotted for this policy set before this investigation.

Palette: dataviz skill's validated 8-hue categorical order (CVD-safe,
adjacent-pair checked), first 7 slots. Identity is double-encoded via color
AND marker/linestyle so the figures stay legible in black-and-white print.
"""
import csv
import os

import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figs")

POLICIES = [
    ("no_raw",           "No-RAW",            "#4a3aa7", "x", ":"),
    ("static",           "Static",            "#2a78d6", "s", ":"),
    ("laca",             "LACA",              "#e87ba4", "v", "--"),
    ("chang2019",        "TGah",              "#008300", "P", "-."),
    ("cluster_csv",      "Cluster-based",     "#1baf7a", "^", "-."),
    ("cluster_adaptive", "MA-PRAW",           "#eda100", "D", "-"),
]

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
})


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["policy"]] = rows.get(r["policy"], [])
            rows[r["policy"]].append(r)
    return rows


def series(rows_by_policy, policy, xkey, ykey):
    rows = sorted(rows_by_policy[policy], key=lambda r: float(r[xkey]))
    x = [float(r[xkey]) for r in rows]
    mean = [float(r[f"{ykey}_mean"]) for r in rows]
    lo = [float(r[f"{ykey}_ci_lo"]) for r in rows]
    hi = [float(r[f"{ykey}_ci_hi"]) for r in rows]
    err = [(m - l, h - m) for m, l, h in zip(mean, lo, hi)]
    yerr = [[max(0, e[0]) for e in err], [max(0, e[1]) for e in err]]
    return x, mean, yerr


def plot_panel(ax, rows_by_policy, xkey, ykey, xlabel, ylabel, scale=1.0, tag=""):
    for key, label, color, marker, ls in POLICIES:
        x, y, yerr = series(rows_by_policy, key, xkey, ykey)
        y = [v * scale for v in y]
        yerr = [[e * scale for e in yerr[0]], [e * scale for e in yerr[1]]]
        ax.errorbar(
            x, y, yerr=yerr, marker=marker, linestyle=ls, color=color,
            linewidth=3, markersize=9, markeredgewidth=2.5, capsize=4, capthick=2, label=label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5, color="#c3c2b7")
    ax.set_xticks(sorted(set(float(r[xkey]) for rows in rows_by_policy.values() for r in rows)))
    if tag:
        ax.text(0.5, -0.32, tag, transform=ax.transAxes, ha="center")


def make_core4_figure(rows_by_policy, xkey, xlabel, out_path, show_legend=True):
    fig = plt.figure(figsize=(16, 4.2))

    ax1 = plt.subplot(1, 3, 1)
    plot_panel(ax1, rows_by_policy, xkey, "pdr", xlabel, "PDR", tag="(a)")
    ax1.set_ylim(0, 1.05)

    ax2 = plt.subplot(1, 3, 2)
    plot_panel(ax2, rows_by_policy, xkey, "tput_kbps", xlabel, "Throughput (kbps)", tag="(b)")

    ax3 = plt.subplot(1, 3, 3)
    plot_panel(ax3, rows_by_policy, xkey, "avg_delay_ms", xlabel, "Average Delay (ms)", scale=1e-3, tag="(c)")
    ax3.set_ylabel("Average Delay (s)")

    if show_legend:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.08), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


PKT_SIZE_BYTES = 128
KBPS_TO_PKTS = 1000.0 / (8.0 * PKT_SIZE_BYTES)


def make_full8_figure(rows_by_policy, xkey, xlabel, out_path, show_legend=True):
    """Same as make_core4_figure but throughput in pkt/s (matches the
    original fig_uav_full8_vs_n.pdf units) instead of kbps."""
    fig = plt.figure(figsize=(16, 4.2))

    ax1 = plt.subplot(1, 3, 1)
    plot_panel(ax1, rows_by_policy, xkey, "pdr", xlabel, "PDR", tag="(a)")
    ax1.set_ylim(0, 1.05)

    ax2 = plt.subplot(1, 3, 2)
    plot_panel(ax2, rows_by_policy, xkey, "tput_kbps", xlabel, "Throughput (pkt/s)", scale=KBPS_TO_PKTS, tag="(b)")

    ax3 = plt.subplot(1, 3, 3)
    plot_panel(ax3, rows_by_policy, xkey, "avg_delay_ms", xlabel, "Average Delay (ms)", scale=1e-3, tag="(c)")
    ax3.set_ylabel("Average Delay (s)")

    if show_legend:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.08), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def make_priority_pdr_figure(rows_by_policy, xkey, xlabel, out_path, show_legend=True):
    fig = plt.figure(figsize=(16, 4.2))

    panels = [
        ("critical_pdr", "PDR", "Critical Traffic", "(a)"),
        ("high_pdr", "PDR", "High Traffic", "(b)"),
        ("normal_pdr", "PDR", "Normal Traffic", "(c)"),
    ]
    first_ax = None
    for i, (ykey, ylabel, title, tag) in enumerate(panels, start=1):
        ax = plt.subplot(1, 3, i)
        plot_panel(ax, rows_by_policy, xkey, ykey, xlabel, ylabel, tag=tag)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        if first_ax is None:
            first_ax = ax

    if show_legend:
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.12), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def make_assoc_figure(rows_by_policy, xkey, xlabel, out_path, show_legend=True):
    fig = plt.figure(figsize=(11, 4.2))

    ax1 = plt.subplot(1, 2, 1)
    for key, label, color, marker, ls in POLICIES:
        rows = sorted(rows_by_policy[key], key=lambda r: float(r[xkey]))
        x = [float(r[xkey]) for r in rows]
        frac = [float(r["assoc_frac_mean"]) for r in rows]
        ax1.plot(x, frac, marker=marker, linestyle=ls, color=color, linewidth=3, markersize=9, markeredgewidth=2.5, label=label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Association Success Fraction")
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#c3c2b7")
    ax1.set_xticks(sorted(set(float(r[xkey]) for rows in rows_by_policy.values() for r in rows)))
    ax1.text(0.5, -0.32, "(a)", transform=ax1.transAxes, ha="center")

    ax2 = plt.subplot(1, 2, 2)
    plot_panel(ax2, rows_by_policy, xkey, "assoc_mean_ms", xlabel, "Mean Assoc. Time (s)", scale=1e-3, tag="(b)")

    if show_legend:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.1), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def make_energy_figure(rows_by_policy, xkey, xlabel, out_path, show_legend=True):
    fig = plt.figure(figsize=(11, 4.2))

    ax1 = plt.subplot(1, 2, 1)
    plot_panel(ax1, rows_by_policy, xkey, "e_total_mj", xlabel, "Total Energy / STA (mJ)", tag="(a)")

    ax2 = plt.subplot(1, 2, 2)
    plot_panel(ax2, rows_by_policy, xkey, "ee_kbit_j", xlabel, "Energy Efficiency (kbit/J)", tag="(b)")

    if show_legend:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.1), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def make_legend_figure(out_path, policies=None, ncol=1):
    """Standalone legend-only figure (no axes/data) sharing the exact same
    color/marker/linestyle mapping as every panel figure. Generated once so
    the per-figure legends can be dropped -- avoids repeating an identical
    7- or 8-entry legend on every multi-panel figure in the paper.
    Vertical (ncol=1) by default -- one entry per row."""
    pol = policies if policies is not None else POLICIES
    nrows = -(-len(pol) // ncol)  # ceil

    fig = plt.figure(figsize=(2.4 * ncol, 0.42 * nrows))
    handles = [
        plt.Line2D([0], [0], color=color, marker=marker, linestyle=ls,
                   linewidth=3, markersize=9, markeredgewidth=2.5, label=label)
        for _, label, color, marker, ls in pol
    ]
    fig.legend(handles, [h.get_label() for h in handles], loc="center",
               ncol=ncol, frameon=False)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def main():
    os.makedirs(FIGS, exist_ok=True)

    interval_rows = load(os.path.join(RESULTS, "uav_assoc_energy_vs_interval.csv"))
    n_rows = load(os.path.join(RESULTS, "uav_assoc_energy_vs_n.csv"))

    make_legend_figure(os.path.join(FIGS, "fig_uav_legend.pdf"))

    make_core4_figure(interval_rows, "packet_interval", "Packet Interval (s)",
                       os.path.join(FIGS, "fig_uav_core4_vs_interval_updated.pdf"), show_legend=False)
    make_core4_figure(n_rows, "num_stas", "Number of UAVs (N)",
                       os.path.join(FIGS, "fig_uav_core4_vs_n_updated.pdf"), show_legend=False)

    make_full8_figure(n_rows, "num_stas", "Number of UAVs (N)",
                       os.path.join(FIGS, "fig_uav_full8_vs_n_updated.pdf"), show_legend=False)

    priority_path = os.path.join(RESULTS, "uav_priority_pdr_vs_n.csv")
    if os.path.exists(priority_path):
        priority_rows = load(priority_path)
        make_priority_pdr_figure(priority_rows, "num_stas", "Number of UAVs (N)",
                                  os.path.join(FIGS, "fig_uav_priority_pdr_vs_n_updated.pdf"), show_legend=False)

    make_assoc_figure(interval_rows, "packet_interval", "Packet Interval (s)",
                       os.path.join(FIGS, "fig_uav_assoc_vs_interval_updated.pdf"), show_legend=False)
    make_assoc_figure(n_rows, "num_stas", "Number of UAVs (N)",
                       os.path.join(FIGS, "fig_uav_assoc_vs_n_updated.pdf"), show_legend=False)

    make_energy_figure(interval_rows, "packet_interval", "Packet Interval (s)",
                        os.path.join(FIGS, "fig_uav_energy_vs_interval_updated.pdf"), show_legend=False)
    make_energy_figure(n_rows, "num_stas", "Number of UAVs (N)",
                        os.path.join(FIGS, "fig_uav_energy_vs_n_updated.pdf"), show_legend=False)


if __name__ == "__main__":
    main()
