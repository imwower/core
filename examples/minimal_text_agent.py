"""Minimal text-only agent demonstrating the awareness core loop."""

from __future__ import annotations

from awareness_core.config import CoreConfig
from awareness_core.loop import AwarenessLoop
from awareness_core.memory.episodic_memory import EpisodicMemory
from awareness_core.tools.llm_tool import LLMTool


def main() -> None:
    config = CoreConfig()
    memory = EpisodicMemory(config.memory.episodic_path)
    tool = LLMTool()
    loop = AwarenessLoop(config=config, tool=tool, memory=memory)

    print("输入一段文字（回车退出）：")
    while True:
        try:
            user_input = input("你 > ")
        except EOFError:
            break
        if not user_input.strip():
            break
        result = loop.step(user_input)
        if result:
            print(f"{tool.name} > {result.content}")
        else:
            print("（当前步骤未触发工具调用）")
    print("结束，再见。")


if __name__ == "__main__":
    main()
