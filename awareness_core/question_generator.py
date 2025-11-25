"""Question generation module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .core_state import AwarenessState


@dataclass
class GeneratedQuestion:
    """Structured question produced by the generator."""

    text: str
    target_axis: str = "text"
    expected_type: str = "explanation"


class QuestionGenerator:
    """Generates simple questions based on awareness state and uncertainty."""

    def generate(
        self,
        state: AwarenessState,
        uncertainty: float,
        hint: Optional[str] = None,
    ) -> GeneratedQuestion:
        """Produce a question string describing what the agent wants to learn."""
        summary = state.summary()
        text_axis_summary = summary.get("text")
        topic = hint
        if topic is None and text_axis_summary is not None:
            topic = text_axis_summary.extras.get("external_text")
        topic = topic or "当前输入"
        text = f"请帮我解释并扩展：{topic}（不确定度={uncertainty:.2f}）"
        return GeneratedQuestion(text=text)
