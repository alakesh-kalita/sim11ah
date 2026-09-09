import csv
import random
from pathlib import Path


def generate_synthetic_cluster_csv(
    path="uav_cluster_data.csv",
    n_uavs=1200,
    n_clusters=12,
    seed=42,
):
    random.seed(seed)
    path = Path(path)

    priorities = ["critical", "high", "normal"]
    priority_prob = [0.10, 0.25, 0.65]
    rates = [300000]  # same data rate for all, fair with adaptive

    # unequal cluster sizes using random weights
    weights = [random.uniform(0.4, 2.0) for _ in range(n_clusters)]
    total_w = sum(weights)

    cluster_sizes = [
        max(1, int(round(n_uavs * w / total_w)))
        for w in weights
    ]

    # fix rounding so total exactly equals n_uavs
    diff = n_uavs - sum(cluster_sizes)
    while diff != 0:
        idx = random.randrange(n_clusters)
        if diff > 0:
            cluster_sizes[idx] += 1
            diff -= 1
        elif cluster_sizes[idx] > 1:
            cluster_sizes[idx] -= 1
            diff += 1

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "aid",
            "cluster_id",
            "priority",
            "tx_rate_bps",
            "queue_len",
            "packet_size_bytes",
        ])

        aid = 1
        for cid, size in enumerate(cluster_sizes, start=1):
            for _ in range(size):
                priority = random.choices(
                    priorities,
                    weights=priority_prob,
                    k=1,
                )[0]

                if priority == "critical":
                    queue_len = random.randint(8, 25)
                elif priority == "high":
                    queue_len = random.randint(4, 15)
                else:
                    queue_len = random.randint(0, 8)

                writer.writerow([
                    aid,
                    cid,
                    priority,
                    random.choice(rates),
                    queue_len,
                    128,
                ])

                aid += 1

    print("CSV generated successfully")
    print(f"Path: {path.resolve()}")
    print(f"Total UAVs: {n_uavs}")
    print(f"Clusters: {n_clusters}")
    print(f"Cluster sizes: {cluster_sizes}")