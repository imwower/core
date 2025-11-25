"""Base tool interface and data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolQuery:
    """Input to a tool call."""

    content: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result returned by a tool call."""

    content: str
    raw: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract interface for all tools."""

    name: str = "base"
    description: str = "abstract tool"

    @abstractmethod
    def call(self, query: ToolQuery) -> ToolResult:
        """Execute the tool with the provided query."""
        raise NotImplementedError
