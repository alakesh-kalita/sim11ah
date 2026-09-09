import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("interval_summary.csv")

# Sort based on packet interval
df = df.sort_values(by="packet_interval")

x = df["packet_interval"].values

plt.figure(figsize=(15,4))
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# Policies
policies = {
    "cluster_csv": ("Cluster-based", "^", "-."),
    "cluster_adaptive": ("Cluster-Adaptive", "D", "-"),
    "adaptive": ("Adaptive", "o", "--"),
    "static": ("Static", "s", ":"),
}

# ------------------ PDR ------------------
ax1 = plt.subplot(1,3,1)
for p, (label, marker, style) in policies.items():
    ax1.plot(
        x,
        df[f"{p}_pdr"],
        marker=marker,
        linestyle=style,
        linewidth=2,
        markersize=6,
        label=label,
    )

ax1.set_xlabel("Packet Interval (s)")
ax1.set_ylabel("PDR")
ax1.set_xticks(x)
ax1.set_ylim(0, 1.05)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()
ax1.text(0.5, -0.35, "(a)", transform=ax1.transAxes, ha='center')


# ------------------ Throughput ------------------
ax2 = plt.subplot(1,3,2)
for p, (label, marker, style) in policies.items():
    ax2.plot(
        x,
        df[f"{p}_throughput"],
        marker=marker,
        linestyle=style,
        linewidth=2,
        markersize=6,
        label=label,
    )

ax2.set_xlabel("Packet Interval (s)")
ax2.set_ylabel("Throughput (pkt/s)")
ax2.set_xticks(x)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()
ax2.text(0.5, -0.35, "(b)", transform=ax2.transAxes, ha='center')


# ------------------ Delay ------------------
ax3 = plt.subplot(1,3,3)
for p, (label, marker, style) in policies.items():
    ax3.plot(
        x,
        df[f"{p}_delay"],
        marker=marker,
        linestyle=style,
        linewidth=2,
        markersize=6,
        label=label,
    )

ax3.set_xlabel("Packet Interval (s)")
ax3.set_ylabel("Average Delay (s)")
ax3.set_xticks(x)
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend()
ax3.text(0.5, -0.35, "(c)", transform=ax3.transAxes, ha='center')


plt.tight_layout()
plt.savefig("interval_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()