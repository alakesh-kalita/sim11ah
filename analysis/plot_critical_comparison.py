import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Load CSV
# =========================================================
df = pd.read_csv("results_scaling.csv")
df = df.sort_values(by="num_stas")

x = df["num_stas"].values
x_ticks = [100, 500, 1000]

# Existing priority classes in your CSV
traffic_classes = ["critical", "high", "normal"]

# Choose metric: "pdr", "throughput", "delay", "generated", "delivered"
metric = "pdr"

policies = {
    "cluster_csv": ("Cluster-based", "^", "-."),
    "cluster_adaptive": ("Cluster-Adaptive", "D", "-"),
    "adaptive": ("Adaptive", "o", "--"),
    "static": ("Static", "s", ":"),
}

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, traffic_class in enumerate(traffic_classes):
    ax = axes[idx]

    for policy, (label, marker, linestyle) in policies.items():
        col = f"{policy}_{traffic_class}_{metric}"

        if col not in df.columns:
            print(f"Missing column: {col}")
            continue

        ax.plot(
            x,
            df[col],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            label=label,
        )

    ax.set_title(f"{traffic_class.capitalize()} Traffic")
    ax.set_xlabel("Number of UAVs")
    ax.set_xticks(x_ticks)

    if metric == "pdr":
        ax.set_ylabel("PDR")
        ax.set_ylim(0, 1.05)
    elif metric == "throughput":
        ax.set_ylabel("Throughput (pkt/s)")
    elif metric == "delay":
        ax.set_ylabel("Average Delay (s)")
    elif metric == "generated":
        ax.set_ylabel("Generated Packets")
    elif metric == "delivered":
        ax.set_ylabel("Delivered Packets")
    else:
        ax.set_ylabel(metric)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.text(0.5, -0.32, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center")

plt.tight_layout()
plt.savefig(f"priority_{metric}_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()