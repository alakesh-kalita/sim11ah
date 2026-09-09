from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any, Dict, Optional


PRIORITY_CLASSES = ("critical", "high", "normal")


class PriorityMetrics:
    """
    Optional priority-aware metric collector.

    This file is independent of the simulator's default stats.
    Enable it only when priority-level analysis is required.

    It tracks:
    - generated packets per priority
    - delivered packets per priority
    - delay per priority
    - PDR per priority
    - throughput per priority
    """

    def __init__(self, csv_path: str = "uav_cluster_data.csv") -> None:
        self.csv_path = csv_path
        self.aid_to_priority: Dict[int, str] = {}
        self.enabled: bool = False

        self.generated = defaultdict(int)
        self.delivered = defaultdict(int)
        self.delays = defaultdict(list)

    def enable(self) -> None:
        self.enabled = True
        self.load_priority_csv()

    def disable(self) -> None:
        self.enabled = False

    def load_priority_csv(self) -> None:
        self.aid_to_priority.clear()

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                aid = int(row["aid"])
                priority = str(row.get("priority", "normal")).strip().lower()

                if priority not in PRIORITY_CLASSES:
                    priority = "normal"

                self.aid_to_priority[aid] = priority

    def get_priority(self, aid: int) -> str:
        return self.aid_to_priority.get(int(aid), "normal")

    def record_generated(self, aid: int) -> None:
        if not self.enabled:
            return

        priority = self.get_priority(aid)
        self.generated[priority] += 1

    def record_delivered(self, aid: int, delay: Optional[float] = None) -> None:
        if not self.enabled:
            return

        priority = self.get_priority(aid)
        self.delivered[priority] += 1

        if delay is not None:
            self.delays[priority].append(float(delay))

    def _mean(self, values):
        return sum(values) / len(values) if values else 0.0

    def summary(self, sim_time: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for priority in PRIORITY_CLASSES:
            gen = int(self.generated[priority])
            delivered = int(self.delivered[priority])
            delays = self.delays[priority]

            out[f"{priority}_generated"] = gen
            out[f"{priority}_delivered"] = delivered
            out[f"{priority}_pdr"] = delivered / max(1, gen)
            out[f"{priority}_throughput"] = delivered / max(sim_time, 1e-12)
            out[f"{priority}_delay"] = self._mean(delays)

        total_delivered = sum(int(self.delivered[p]) for p in PRIORITY_CLASSES)

        for priority in PRIORITY_CLASSES:
            delivered = int(self.delivered[priority])
            out[f"{priority}_throughput_share"] = delivered / max(1, total_delivered)

        return out


def attach_priority_metrics(
    sim,
    csv_path: str = "uav_cluster_data.csv",
    enabled: bool = False,
):
    """
    Attach priority metrics to simulator object without modifying sim.stats.
    """
    tracker = PriorityMetrics(csv_path=csv_path)

    if enabled:
        tracker.enable()

    sim.priority_metrics = tracker
    return tracker