"""Accumulates scalar metrics between logging steps."""

import torch


class MetricsLogger:
    def __init__(self):
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def add(self, metrics: dict[str, torch.Tensor]) -> None:
        for k, v in metrics.items():
            v = v.detach().clone()
            self.sums[k] = self.sums[k] + v if k in self.sums else v
            self.counts[k] = self.counts.get(k, 0) + 1

    def pop(self, mean: bool = True) -> dict[str, torch.Tensor]:
        """Return the accumulated metrics (averaged by default) and reset."""
        out = {k: v / self.counts[k] if mean else v for k, v in self.sums.items()}
        self.sums, self.counts = {}, {}
        return out
