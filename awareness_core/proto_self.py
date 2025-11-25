"""Proto-self encoder: converts raw system metrics into p(t)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


@dataclass
class ProtoState:
    """Lightweight container for the proto-self vector and source metrics."""

    vector: List[float]
    system_metrics: Dict[str, float] = field(default_factory=dict)
    learning_metrics: Dict[str, float] = field(default_factory=dict)


class ProtoSelf:
    """Encodes system metrics into a proto-self vector p(t)."""

    def __init__(self, dim: int = 6) -> None:
        self.dim = dim

    def encode(
        self,
        system_metrics: Optional[Mapping[str, float]] = None,
        learning_metrics: Optional[Mapping[str, float]] = None,
    ) -> ProtoState:
        """Produce a proto-self vector from metrics.

        Missing metrics default to zero; values are clipped to [0, 1] and
        truncated or padded to the configured dimension.
        """
        system_metrics = dict(system_metrics or {})
        learning_metrics = dict(learning_metrics or {})
        combined = list(system_metrics.values()) + list(learning_metrics.values())
        vector = self._normalize(combined)
        return ProtoState(vector=vector, system_metrics=system_metrics, learning_metrics=learning_metrics)

    def _normalize(self, values: List[float]) -> List[float]:
        normalized = []
        for value in values[: self.dim]:
            normalized.append(self._clip(value))
        while len(normalized) < self.dim:
            normalized.append(0.0)
        return normalized

    @staticmethod
    def _clip(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
