import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_scaling.csv")
df = df.sort_values(by="num_stas")

# Desired x-ticks
x_ticks = [100, 500,  1000]

plt.figure(figsize=(12,4))
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# -------- PDR --------
ax1 = plt.subplot(1,3,1)
ax1.plot(df["num_stas"], df["raw_on_adaptive_pdr"], marker='o', linestyle='--', label="Adaptive")
ax1.plot(df["num_stas"], df["raw_on_static_pdr"], marker='s', label="Static")
ax1.set_xlabel("STAs")
ax1.set_ylabel("PDR")
ax1.set_xticks(x_ticks)
ax1.legend()
ax1.grid()

# (a) label below x-axis
ax1.text(0.5, -0.35, "(a)", transform=ax1.transAxes, ha='center')

# -------- Throughput --------
ax2 = plt.subplot(1,3,2)
ax2.plot(df["num_stas"], df["raw_on_adaptive_throughput"], marker='o', linestyle='--', label="Adaptive")
ax2.plot(df["num_stas"], df["raw_on_static_throughput"], marker='s', label="Static")
ax2.set_xlabel("STAs")
ax2.set_ylabel("Throughput")
ax2.set_xticks(x_ticks)
ax2.legend()
ax2.grid()

# (b)
ax2.text(0.5, -0.35, "(b)", transform=ax2.transAxes, ha='center')

# -------- Delay --------
ax3 = plt.subplot(1,3,3)
ax3.plot(df["num_stas"], df["raw_on_adaptive_avg_delay"], marker='o', linestyle='--', label="Adaptive")
ax3.plot(df["num_stas"], df["raw_on_static_avg_delay"], marker='s', label="Static")
ax3.set_xlabel("STAs")
ax3.set_ylabel("Delay")
ax3.set_xticks(x_ticks)
ax3.legend()
ax3.grid()

# (c)
ax3.text(0.5, -0.35, "(c)", transform=ax3.transAxes, ha='center')

plt.tight_layout()
plt.savefig("combined_interval.pdf", format="pdf", bbox_inches="tight")
plt.show()