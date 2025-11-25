"""Awareness state representation and utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Mapping, Sequence

from .self_axes.base_axis import BaseAxis


@dataclass
class AwarenessFrame:
    """Snapshot of the multi-axis awareness state at a time point."""

    timestamp: datetime
    step: int
    vectors: Dict[str, Sequence[float]]
    summaries: Dict[str, Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "vectors": {k: list(v) for k, v in self.vectors.items()},
            "summaries": self.summaries,
            "meta": self.meta,
        }


class AwarenessState:
    """Container managing multiple axes of the awareness vector s(t)."""

    def __init__(self, axes: Iterable[BaseAxis] | None = None, max_history: int = 50) -> None:
        self.axes: Dict[str, BaseAxis] = {}
        self.step: int = 0
        self.history: Deque[AwarenessFrame] = deque(maxlen=max_history)
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

    def to_frame(self, meta: Dict[str, Any] | None = None, store: bool = True) -> AwarenessFrame:
        """Export the current state to an AwarenessFrame and optionally store it."""
        self.step += 1
        vectors = {name: axis.to_vector() for name, axis in self.axes.items()}
        summaries = {
            name: {
                "name": axis_summary.name,
                "dim": axis_summary.dim,
                "extras": axis_summary.extras,
            }
            for name, axis_summary in self.summary().items()
        }
        frame = AwarenessFrame(
            timestamp=datetime.utcnow(),
            step=self.step,
            vectors=vectors,
            summaries=summaries,
            meta=meta or {},
        )
        if store:
            self.history.append(frame)
        return frame

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary for logging."""
        return {name: axis.summary() for name, axis in self.axes.items()}

    def recent_frames(self, limit: int = 5) -> List[AwarenessFrame]:
        """Return the most recent frames (oldest-first within the slice)."""
        if limit <= 0:
            return []
        return list(self.history)[-limit:]

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the current state without mutating history."""
        vectors = {name: axis.to_vector() for name, axis in self.axes.items()}
        summaries = {
            name: {
                "name": axis_summary.name,
                "dim": axis_summary.dim,
                "extras": axis_summary.extras,
            }
            for name, axis_summary in self.summary().items()
        }
        return {
            "step": self.step,
            "vectors": vectors,
            "summaries": summaries,
        }
