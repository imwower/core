"""Lightweight LLM tool wrapper (stub-friendly)."""

from __future__ import annotations

import os
from typing import Optional

from .base_tool import BaseTool, ToolQuery, ToolResult


class LLMTool(BaseTool):
    """Simple LLM tool; currently echoes queries with optional provider info."""

    name = "llm"
    description = "Stub LLM tool that echoes the query content."

    def __init__(self, provider: Optional[str] = None) -> None:
        self.provider = provider or os.getenv("AWARENESS_LLM_PROVIDER", "stub")

    def call(self, query: ToolQuery) -> ToolResult:
        """Return a placeholder response; real API integration can be added later."""
        response = f"[{self.provider} LLM 回复] {query.content}"
        return ToolResult(content=response, metadata={"provider": self.provider, "echo": True})
