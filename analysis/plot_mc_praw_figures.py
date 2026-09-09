"""
MC-PRAW unified figure set: one comparison figure set with E-TAROA (SenSys'17)
included everywhere, alongside the same 6 UAV policies, all on equal 3-seed
CI footing.

Supersedes an earlier split where E-TAROA lived in its own 8-policy,
single-seed figure set separate from this 7-policy, 3-seed set. E-TAROA was
added to eval_uav_assoc_energy.py's and eval_uav_priority_pdr.py's POLICIES
lists and re-run at the same 3 seeds as every other policy, so there's no
longer a data-source split to justify two separate figure sets -- the old
split-set figures and the script that generated them were removed.

Adaptive is excluded per an explicit request to drop it from the comparison
(still computed and cached by the eval scripts, just not plotted). Ordering:
No-RAW first, Cluster-based/Cluster-Adaptive last.

Output directory: results/figs/MC-PRAW/
"""
import os

import plot_uav_updated_figures as base

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


def main():
    os.makedirs(FIGS, exist_ok=True)
    base.POLICIES = POLICIES

    interval_rows = base.load(os.path.join(RESULTS, "uav_assoc_energy_vs_interval.csv"))
    n_rows = base.load(os.path.join(RESULTS, "uav_assoc_energy_vs_n.csv"))
    priority_rows = base.load(os.path.join(RESULTS, "uav_priority_pdr_vs_n.csv"))

    base.make_legend_figure(os.path.join(FIGS, "fig_mc_praw_legend.pdf"))

    base.make_core4_figure(
        interval_rows, "packet_interval", "Packet Interval (s)",
        os.path.join(FIGS, "fig_mc_praw_core4_vs_interval.pdf"), show_legend=False,
    )
    base.make_core4_figure(
        n_rows, "num_stas", "Number of UAVs (N)",
        os.path.join(FIGS, "fig_mc_praw_core4_vs_n.pdf"), show_legend=False,
    )
    base.make_full8_figure(
        n_rows, "num_stas", "Number of UAVs (N)",
        os.path.join(FIGS, "fig_mc_praw_full8_vs_n.pdf"), show_legend=False,
    )
    base.make_priority_pdr_figure(
        priority_rows, "num_stas", "Number of UAVs (N)",
        os.path.join(FIGS, "fig_mc_praw_priority_pdr_vs_n.pdf"), show_legend=False,
    )
    base.make_assoc_figure(
        interval_rows, "packet_interval", "Packet Interval (s)",
        os.path.join(FIGS, "fig_mc_praw_assoc_vs_interval.pdf"), show_legend=False,
    )
    base.make_assoc_figure(
        n_rows, "num_stas", "Number of UAVs (N)",
        os.path.join(FIGS, "fig_mc_praw_assoc_vs_n.pdf"), show_legend=False,
    )
    base.make_energy_figure(
        interval_rows, "packet_interval", "Packet Interval (s)",
        os.path.join(FIGS, "fig_mc_praw_energy_vs_interval.pdf"), show_legend=False,
    )
    base.make_energy_figure(
        n_rows, "num_stas", "Number of UAVs (N)",
        os.path.join(FIGS, "fig_mc_praw_energy_vs_n.pdf"), show_legend=False,
    )


if __name__ == "__main__":
    main()
