"""Main loop wiring the awareness core components."""

from __future__ import annotations

from typing import Optional

from .config import CoreConfig
from .core_state import AwarenessState
from .integration.text_input_adapter import TextInputAdapter
from .memory.episodic_memory import EpisodicMemory
from .proto_self import ProtoSelf
from .question_generator import QuestionGenerator
from .scheduler import Scheduler
from .self_axes.text_axis import TextAxis
from .tools.base_tool import BaseTool, ToolQuery, ToolResult


class AwarenessLoop:
    """Minimal runnable loop that processes text input and queries a tool."""

    def __init__(
        self,
        config: CoreConfig,
        tool: BaseTool,
        memory: EpisodicMemory,
        proto_self: Optional[ProtoSelf] = None,
        text_axis: Optional[TextAxis] = None,
    ) -> None:
        self.config = config
        self.tool = tool
        self.memory = memory
        self.proto_self = proto_self or ProtoSelf()
        self.text_axis = text_axis or TextAxis(
            name=config.axis("text").name, dim=config.axis("text").dim
        )

        self.state = AwarenessState(axes=[self.text_axis])
        self.scheduler = Scheduler(config=config)
        self.question_generator = QuestionGenerator()
        self.text_adapter = TextInputAdapter()

    def step(self, external_text: Optional[str]) -> Optional[ToolResult]:
        """Process one step given optional external text."""
        features = self.text_adapter.encode(external_text)
        external_salience = features.salience if features else 0.0

        if features:
            self.state.update_axis("text", {"external_text": features.external_text})

        proto_state = self.proto_self.encode()
        drives = {"curiosity": 0.6 if features else 0.8}

        mode = self.scheduler.decide(external_salience=external_salience, drives=drives)
        if mode == "idle":
            return None

        uncertainty = max(0.0, 1.0 - external_salience)
        question = self.question_generator.generate(
            state=self.state, uncertainty=uncertainty, hint=features.external_text if features else None
        )

        result = self.tool.call(ToolQuery(content=question.text))
        self.state.update_axis("text", {"internal_text": result.content})

        meta = {
            "mode": mode,
            "external_salience": external_salience,
            "proto_state": proto_state.vector,
        }
        self.memory.append(question=question.text, tool=self.tool.name, answer=result.content, metadata=meta)
        return result
