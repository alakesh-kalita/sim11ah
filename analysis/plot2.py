import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("interval_summary.csv")

# Sort based on packet interval
df = df.sort_values(by="packet_interval")

# Extract x values
x = df["packet_interval"].values

plt.figure(figsize=(12,4))
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ------------------ PDR ------------------
ax1 = plt.subplot(1,3,1)
ax1.plot(x, df["raw_on_adaptive_pdr"], marker='o', linewidth=2, linestyle='--', label="Adaptive")
ax1.plot(x, df["raw_on_static_pdr"], marker='s', linewidth=2, label="Static")
ax1.set_xlabel("Packet Interval (s)")
ax1.set_ylabel("PDR")
ax1.set_xticks(x)
ax1.legend()
ax1.grid()

# (a) below x-axis label
ax1.text(0.5, -0.40, "(a)", transform=ax1.transAxes, ha='center')

# ------------------ Throughput ------------------
ax2 = plt.subplot(1,3,2)
ax2.plot(x, df["raw_on_adaptive_throughput"], marker='o', linewidth=2, linestyle='--', label="Adaptive")
ax2.plot(x, df["raw_on_static_throughput"], marker='s', linewidth=2, label="Static")
ax2.set_xlabel("Packet Interval (s)")
ax2.set_ylabel("Throughput")
ax2.set_xticks(x)
ax2.legend()
ax2.grid()

# (b)
ax2.text(0.5, -0.40, "(b)", transform=ax2.transAxes, ha='center')

# ------------------ Delay ------------------
ax3 = plt.subplot(1,3,3)
ax3.plot(x, df["raw_on_adaptive_avg_delay"], marker='o', linewidth=2, linestyle='--', label="Adaptive")
ax3.plot(x, df["raw_on_static_avg_delay"], marker='s', linewidth=2, label="Static")
ax3.set_xlabel("Packet Interval (s)")
ax3.set_ylabel("Delay")
ax3.set_xticks(x)
ax3.legend()
ax3.grid()

# (c)
ax3.text(0.5, -0.40, "(c)", transform=ax3.transAxes, ha='center')

plt.tight_layout()
plt.savefig("combined_interval.pdf", format="pdf", bbox_inches="tight")
plt.show()