"""Simple scheduler for deciding thinking mode."""

from __future__ import annotations

from typing import Dict

from .config import CoreConfig


class Scheduler:
    """Rule-based scheduler choosing between external processing and internal thinking."""

    def __init__(self, config: CoreConfig) -> None:
        self.config = config

    def decide(self, external_salience: float, drives: Dict[str, float] | None = None) -> str:
        """Return one of: 'external', 'internal_think', or 'idle'."""
        drives = drives or {}
        if external_salience >= self.config.external_salience_threshold:
            return "external"
        curiosity = drives.get("curiosity", 0.0)
        if curiosity >= self.config.internal_think_threshold:
            return "internal_think"
        return "idle"
