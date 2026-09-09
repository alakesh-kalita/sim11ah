"""
Grouping-strategy ablation figure for Reviewer 2 Comment 4: spatial
(LSTM+K-Means) vs. random vs. traffic-demand-based vs. distance-based
grouping, all matched to the SAME group-size distribution (spatial's real
K-Means output) so the grouping CRITERION is the only variable -- see
uav/build_ablation_csvs.py for why size-matching was necessary (an
unmatched first pass confounded criterion with group-size balance).

2 rows (Cluster-based, MA-PRAW/Cluster-Adaptive) x 1 column (PDR vs N).
"""
import csv
import os

import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figs", "MC-PRAW")

CONDITIONS = [
    ("spatial", "Spatial (LSTM+K-Means)", "#eda100", "D", "-"),
    ("random", "Random", "#4a3aa7", "x", ":"),
    ("demand", "Traffic-Demand-Based", "#e87ba4", "v", "--"),
    ("distance", "Distance-Based (RSSI/PHY-rate proxy)", "#1a1a1a", "*", "-."),
]

POLICIES = [
    ("cluster_csv", "Cluster-based"),
    ("cluster_adaptive", "MA-PRAW (Cluster-Adaptive)"),
]

plt.rcParams.update({
    "font.size": 14, "font.family": "sans-serif",
    "axes.labelsize": 14, "axes.titlesize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11,
})


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["condition"], r["policy"]), []).append(r)
    return rows


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load(os.path.join(RESULTS, "uav_grouping_ablation.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for col_idx, (policy_key, policy_label) in enumerate(POLICIES):
        ax = axes[col_idx]
        for cond_key, cond_label, color, marker, ls in CONDITIONS:
            r = sorted(rows[(cond_key, policy_key)], key=lambda x: int(x["num_stas"]))
            x = [float(v["num_stas"]) for v in r]
            mean = [float(v["pdr_mean"]) for v in r]
            lo = [float(v["pdr_ci_lo"]) for v in r]
            hi = [float(v["pdr_ci_hi"]) for v in r]
            yerr = [[max(0, m - l) for m, l in zip(mean, lo)], [max(0, h - m) for m, h in zip(mean, hi)]]
            ax.errorbar(x, mean, yerr=yerr, marker=marker, linestyle=ls, color=color,
                        linewidth=2.5, markersize=8, capsize=4, capthick=1.6, label=cond_label)
        ax.set_xlabel("Number of UAVs (N)")
        ax.set_ylabel("PDR")
        ax.set_ylim(0, 1.05)
        ax.set_title(policy_label, fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        if col_idx == 0:
            ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(FIGS, "fig_grouping_ablation.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
