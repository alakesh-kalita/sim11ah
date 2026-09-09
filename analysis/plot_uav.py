import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("results_scaling.csv")
df = df.sort_values(by="num_stas")

# X-axis ticks
x_ticks = [100, 500, 1000]

# Global style
plt.figure(figsize=(15, 4))
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# Policies (UPDATED)
policies = {
    "cluster_csv": ("Cluster-based", "^", "-."),
    "cluster_adaptive": ("Cluster-Adaptive", "D", "-"),
    "adaptive": ("Adaptive", "o", "--"),
    "static": ("Static", "s", ":"),
}

# -------- PDR --------
ax1 = plt.subplot(1, 3, 1)
for p, (label, marker, style) in policies.items():
    ax1.plot(
        df["num_stas"],
        df[f"{p}_pdr"],
        marker=marker,
        linestyle=style,
        linewidth=2,
        markersize=6,
        label=label,
    )

ax1.set_xlabel("Number of UAVs")
ax1.set_ylabel("PDR")
ax1.set_xticks(x_ticks)
ax1.set_ylim(0, 1.05)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()
ax1.text(0.5, -0.35, "(a)", transform=ax1.transAxes, ha="center")

# -------- Throughput --------
ax2 = plt.subplot(1, 3, 2)
for p, (label, marker, style) in policies.items():
    ax2.plot(
        df["num_stas"],
        df[f"{p}_throughput"],
        marker=marker,
        linestyle=style,
        linewidth=2,
        markersize=6,
        label=label,
    )

ax2.set_xlabel("Number of UAVs")
ax2.set_ylabel("Throughput (pkt/s)")
ax2.set_xticks(x_ticks)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()
ax2.text(0.5, -0.35, "(b)", transform=ax2.transAxes, ha="center")

# -------- Delay --------
ax3 = plt.subplot(1, 3, 3)
for p, (label, marker, style) in policies.items():
    ax3.plot(
        df["num_stas"],
        df[f"{p}_delay"],
        marker=marker,
        linestyle=style,
        linewidth=2,
        markersize=6,
        label=label,
    )

ax3.set_xlabel("Number of UAVs")
ax3.set_ylabel("Average Delay (s)")
ax3.set_xticks(x_ticks)
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend()
ax3.text(0.5, -0.35, "(c)", transform=ax3.transAxes, ha="center")

# Layout and save
plt.tight_layout()
plt.savefig("uav_raw_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()