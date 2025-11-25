"""Configuration objects for the awareness core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AxisConfig:
    """Configuration for a single axis in the awareness state."""

    name: str
    dim: int = 8
    description: str | None = None


@dataclass
class MemoryConfig:
    """Paths and limits for memory backends."""

    episodic_path: Path = Path("data/episodic_memory.jsonl")
    max_events: Optional[int] = None


@dataclass
class CoreConfig:
    """Top-level configuration for the awareness core."""

    axes: Dict[str, AxisConfig] = field(
        default_factory=lambda: {
            "text": AxisConfig(name="text", dim=12, description="Text / language axis"),
        }
    )
    internal_think_threshold: float = 0.5
    external_salience_threshold: float = 0.2
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def axis(self, name: str) -> AxisConfig:
        """Return the config for the requested axis."""
        return self.axes[name]
