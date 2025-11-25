"""Minimal episodic memory writing events to JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EpisodicEvent:
    """Single episodic memory record."""

    timestamp: str
    question: str
    tool: str
    answer: str
    metadata: Dict[str, Any]


class EpisodicMemory:
    """JSONL-based episodic memory store."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, question: str, tool: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> EpisodicEvent:
        event = EpisodicEvent(
            timestamp=datetime.utcnow().isoformat(),
            question=question,
            tool=tool,
            answer=answer,
            metadata=metadata or {},
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        return event

    def load_recent(self, limit: int = 20) -> List[EpisodicEvent]:
        """Load the most recent events (simple file scan)."""
        lines = self.path.read_text(encoding="utf-8").splitlines()
        events: List[EpisodicEvent] = []
        for line in lines[-limit:]:
            try:
                payload = json.loads(line)
                events.append(EpisodicEvent(**payload))
            except json.JSONDecodeError:
                continue
        return events

    def __iter__(self) -> Iterable[EpisodicEvent]:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                yield EpisodicEvent(**json.loads(line))
            except json.JSONDecodeError:
                continue
