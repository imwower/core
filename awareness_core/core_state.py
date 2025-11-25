"""Awareness state representation and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .self_axes.base_axis import BaseAxis


@dataclass
class AwarenessFrame:
    """Snapshot of the multi-axis awareness state at a time point."""

    timestamp: datetime
    vectors: Dict[str, Sequence[float]]
    meta: Dict[str, Any] = field(default_factory=dict)


class AwarenessState:
    """Container managing multiple axes of the awareness vector s(t)."""

    def __init__(self, axes: Iterable[BaseAxis] | None = None) -> None:
        self.axes: Dict[str, BaseAxis] = {}
        if axes:
            for axis in axes:
                self.register_axis(axis)

    def register_axis(self, axis: BaseAxis) -> None:
        """Register an axis to participate in the awareness vector."""
        if axis.name in self.axes:
            raise ValueError(f"Axis {axis.name!r} already registered")
        self.axes[axis.name] = axis

    def reset(self) -> None:
        """Reset all axes to their initial state."""
        for axis in self.axes.values():
            axis.reset()

    def update_axis(self, name: str, features: Mapping[str, Any]) -> None:
        """Update a single axis with new features."""
        if name not in self.axes:
            raise KeyError(f"Axis {name!r} not registered")
        self.axes[name].update_from_input(features)

    def to_frame(self, meta: Dict[str, Any] | None = None) -> AwarenessFrame:
        """Export the current state to an AwarenessFrame."""
        vectors = {name: axis.to_vector() for name, axis in self.axes.items()}
        return AwarenessFrame(timestamp=datetime.utcnow(), vectors=vectors, meta=meta or {})

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary for logging."""
        return {name: axis.summary() for name, axis in self.axes.items()}
