"""Abstract base class for awareness axes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping


@dataclass
class AxisSummary:
    """Lightweight view of an axis state for logging."""

    name: str
    dim: int
    extras: Dict[str, Any]


class BaseAxis(ABC):
    """Base interface for all awareness axes."""

    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dim = dim

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""

    @abstractmethod
    def update_from_input(self, features: Mapping[str, Any]) -> None:
        """Update internal state from upstream features."""

    @abstractmethod
    def to_vector(self) -> List[float]:
        """Export the current axis representation."""

    def summary(self) -> AxisSummary:
        """Return a light summary of axis state."""
        return AxisSummary(name=self.name, dim=self.dim, extras={})

    def to_dict(self) -> Dict[str, Any]:
        """Serializable view combining vector and summary extras."""
        summary = self.summary()
        return {
            "name": summary.name,
            "dim": summary.dim,
            "vector": self.to_vector(),
            "extras": summary.extras,
        }
