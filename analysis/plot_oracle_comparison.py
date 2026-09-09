"""
Oracle vs. Predicted (MA-PRAW) comparison figure for Reviewer 2 Comment 2 --
quantifies the MAC-layer cost of LSTM mobility-prediction error, using the
REAL predicted-position and true-position (oracle) cluster CSVs built by
uav/build_cluster_csvs.py from the actual trained LSTM + K-Means pipeline
(see scripts/eval_uav_oracle_comparison.py for the simulation sweep).

2 rows (Cluster-based, MA-PRAW/Cluster-Adaptive) x 3 columns (PDR,
Throughput, Delay), each vs N. N capped at 500 -- the real UAV population
size in the ML pipeline, not the usual 1000.

Output: results/figs/MC-PRAW/fig_oracle_vs_predicted.pdf (+ .png)
"""
import csv
import os

import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figs", "MC-PRAW")

CONDITIONS = [
    ("predicted", "MA-PRAW (Predicted)", "#eda100", "D", "--"),
    ("oracle", "Oracle (True Position)", "#1a1a1a", "*", "-"),
]

POLICIES = [
    ("cluster_csv", "Cluster-based"),
    ("cluster_adaptive", "MA-PRAW (Cluster-Adaptive)"),
]

PANELS = [
    ("pdr", "PDR", None),
    ("tput_kbps", "Throughput (kbps)", None),
    ("avg_delay_ms", "Average Delay (ms)", None),
]

plt.rcParams.update({
    "font.size": 14, "font.family": "sans-serif",
    "axes.labelsize": 14, "axes.titlesize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["condition"], r["policy"]), []).append(r)
    return rows


def series(rows, condition, policy, metric):
    r = sorted(rows[(condition, policy)], key=lambda x: int(x["num_stas"]))
    x = [float(v["num_stas"]) for v in r]
    mean = [float(v[f"{metric}_mean"]) for v in r]
    lo = [float(v[f"{metric}_ci_lo"]) for v in r]
    hi = [float(v[f"{metric}_ci_hi"]) for v in r]
    yerr = [[max(0, m - l) for m, l in zip(mean, lo)], [max(0, h - m) for m, h in zip(mean, hi)]]
    return x, mean, yerr


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load(os.path.join(RESULTS, "uav_oracle_comparison.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    for row_idx, (policy_key, policy_label) in enumerate(POLICIES):
        for col_idx, (metric, ylabel, ylim) in enumerate(PANELS):
            ax = axes[row_idx][col_idx]
            for cond_key, cond_label, color, marker, ls in CONDITIONS:
                x, mean, yerr = series(rows, cond_key, policy_key, metric)
                ax.errorbar(x, mean, yerr=yerr, marker=marker, linestyle=ls, color=color,
                            linewidth=2.5, markersize=8, capsize=4, capthick=1.6,
                            label=cond_label)
            ax.set_xlabel("Number of UAVs (N)")
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_title(f"{policy_label}\n{ylabel}", fontsize=13)
            else:
                ax.set_title(ylabel, fontsize=13)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(FIGS, "fig_oracle_vs_predicted.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
