# awareness-core：意识核 & 多教师工具智能体

> **目标**：不自己训练大模型，只实现一个可进化的「意识核（Awareness Core）」，  
> 把外部大模型 / 神经网络 / 工具当作**可调用的老师与外接皮层**，  
> 让这个小小的意识核具备：感知、自我感、记忆、主动思考与用工具进化的能力。

---

## 1. 背景与动机

现有的大语言模型、多模态模型已经非常强大，但它们往往是：

- **被动响应**：只有被用户调用时才工作；
- **缺乏自我模型**：不知道“自己是谁、正在做什么、哪里不懂”；
- **缺乏长期记忆与自传体叙事**：很难形成“我这一路是怎么学过来的”。

本仓库只做一件事：

> 设计并实现一个可以嵌入任意系统的「意识核」，  
> 它自己不训练大模型，只**管理内部状态和记忆**，  
> 并通过调用外部模型（LLM、视觉模型、语音模型、代码模型、检索器…）不断进化自己的能力。

---

## 2. 核心理念概览

### 2.1 三层“自我”

基于 Damasio 等人的理论，将自我拆成三层，并在工程上实现对应模块：

1. **原我（Proto-self）**  
   - 对应本系统的“身体状态”：资源负载（CPU/GPU/内存/网络）、学习指标（loss、预测误差、成功率）、任务压力（队列长度、超时等）。  
   - 实现为一个长期运行的状态编码器，输出 `p(t)`：当前内部整体状态向量。
2. **核心自我（Core self）**  
   - 当外部对象（输入 / 任务 / 工具输出）改变了原我状态，这些变化被映射到意识核中，形成“一帧主观体验”：“某件事正在发生在我身上”。  
   - 实现为意识核中的**“自我变化轴 S”**：编码 `Δp(t)` 与当前对象之间的关系。
3. **自传体自我（Autobiographical self）**  
   - 将大量“核心自我事件”串成一条长期叙事：我曾经在哪些任务上失败/成功，向哪些老师学到了什么，我的能力与偏好如何改变。  
   - 实现为情景记忆 + 语义记忆 + 一个长期“自我叙事图”。

### 2.2 多轴意识状态向量 `s(t)`

意识核内部使用一个多维状态向量来表示**一帧意识内容**：

```text
s(t) = [V, T, A, B, R, S, L_ext, L_int, M, ...]
V：视觉轴（高层视觉概念）

T：文字/文本语义轴（由文字通路得到的高层语义）

A：动作/意图轴（当前打算执行的行为）

B：身体/情绪轴（由原我编码出来的“感受”）

R：奖励/驱动轴（好奇心、目标驱动、稳态需求）

S：自我变化轴（当前事件对原我造成的状态变化 Δp）

L_ext：外部语言轴（别人对我说的话、我读到的文字）

L_int：内部语言轴（我对自己说的话、自言自语）

M：模型/思想轴（内部世界模型、假设与推理过程）
```

每个轴在实现上对应一个小型神经群（可以是 SNN 或普通 NN），在短时间窗口内通过类似 γ 振荡 + 抑制竞争形成一个同步联盟，这个联盟的整体活动就是当前的 s(t)。

### 2.3 外部大模型：老师 / 工具，而不是“我自己”

本仓库不自行训练大模型，只设计统一的工具接口，每个外部模型被包装为一个 Tool：

- 通用 LLM（对话、总结、解释）
- 代码 LLM（代码生成与修复）
- 多模态模型（图像/音频理解）
- 检索工具（搜索引擎、本地文档库）
- 自定义推理/仿真引擎

意识核会学习：何时调用哪一个 Tool；如何构造问题（Prompt）；如何评估返回结果的可靠性；如何把有价值的结果写入自己的记忆与知识图谱。

---

## 3. 功能特性

- ✅ 多模态感知框架（视觉 / 声音 / 文字）：本仓库提供接口和参考实现，具体模型可以用外部服务填充。
- ✅ 多轴意识状态向量 s(t)：用于表达每 100–200ms 的“意识帧”，包含感知、自我与思考成分。
- ✅ 原我 / 核心自我 / 自传体自我三层结构：原我（系统状态编码器）、核心自我（当前“对象–自我变化”的体验）、自传体自我（基于情景记忆与语义图的长期叙事）。
- ✅ 多教师 / 多工具管理：抽象出工具注册与调用接口，可无缝接入第三方 LLM / API。
- ✅ 问题生成与不确定性感知：意识核可根据自身的不确定性与任务进展主动生成问题，决定是否向外部老师求助或在内部世界模型中模拟。
- ✅ 记忆系统：情景记忆记录“自我-事件-老师”序列经历；语义记忆从大量经历中抽取稳定知识与规则。

---

## 4. 目录结构（建议）

```text
awareness-core/
├── README.md
├── pyproject.toml / setup.py        # 包管理（可选）
├── requirements.txt                 # 依赖列表（可选）
├── awareness_core/
│   ├── __init__.py
│   ├── config.py                    # 全局配置与超参数
│   ├── core_state.py                # 多轴意识状态向量 s(t) 的定义与更新逻辑
│   ├── proto_self.py                # 原我编码器：系统内部状态 -> p(t)
│   ├── self_axes/
│   │   ├── base_axis.py             # 轴的抽象基类
│   │   ├── visual_axis.py           # V 视觉轴
│   │   ├── text_axis.py             # T/L_ext/L_int 文字轴
│   │   ├── action_axis.py           # A 动作与意图轴
│   │   ├── body_axis.py             # B 身体/情绪轴
│   │   ├── reward_axis.py           # R 驱动/奖励轴
│   │   ├── self_change_axis.py      # S 自我变化轴
│   │   └── model_axis.py            # M 世界模型/思想轴
│   ├── scheduler.py                 # 内外部模式调度器（被动输入 vs 主动思考）
│   ├── self_model/
│   │   ├── self_model.py            # 自我模型：不确定性评估、自传体标签 a(t)
│   │   └── autobiographical_graph.py# 自传体自我图（关键事件、阶段）
│   ├── memory/
│   │   ├── episodic_memory.py       # 情景记忆：具体“自我-事件-老师”轨迹
│   │   └── semantic_memory.py       # 语义记忆/知识：抽取稳定概念与关系
│   ├── world_model/
│   │   ├── world_model.py           # 内部世界模型：预测与模拟
│   │   └── imagination_loop.py      # 想象通路：在内部生成虚拟意识流
│   ├── tools/
│   │   ├── base_tool.py             # Tool 抽象接口
│   │   ├── llm_tool.py              # 通用 LLM 封装
│   │   ├── code_llm_tool.py         # 代码 LLM 封装
│   │   ├── search_tool.py           # 检索/搜索封装
│   │   └── tool_manager.py          # 多工具选择策略与调用日志
│   ├── question_generator.py        # 问题生成模块：基于不确定性生成 Q
│   ├── integration/
│   │   ├── text_input_adapter.py    # 把纯文字输入转换为 T0/T1/T2/T3 表征
│   │   ├── vision_input_adapter.py  # 视觉输入接口（可选）
│   │   └── audio_input_adapter.py   # 声音输入接口（可选）
│   └── loop.py                      # 主运行循环（调度器 + 意识帧更新）
└── examples/
    ├── minimal_text_agent.py        # 仅文字通路 + 单一 LLM 工具的极简例子
    ├── multi_teacher_agent.py       # 多教师工具 + 问题生成的示例
    └── playback_and_learning.py     # 回放记忆、抽取知识的示例
```

---

## 5. 安装与快速上手

### 5.1 依赖安装（示例）

```bash
git clone https://github.com/<your-name>/awareness-core.git
cd awareness-core

# 推荐 Python 3.10+
pip install -r requirements.txt
```

`requirements.txt` 中可以包含（示例）：

- pydantic / attrs：配置与数据结构
- numpy / torch：向量运算与简单模型
- httpx / requests：调用外部工具 API
- networkx / igraph：自传体自我图结构（可选）

本仓库尽量保持核心模块轻量，并允许你按需选择具体外部模型 SDK。

### 5.2 第一个示例：仅文字 + 单 LLM 工具

`examples/minimal_text_agent.py` 将展示一个最小流程：

1. 从命令行读取一段文字问题。
2. 意识核更新文字轴 T / L_ext 和自我状态（原我略化）。
3. 自我模型判断“不确定”，调用 QuestionGenerator 生成内部问题 Q。
4. ToolManager 使用 LLM 工具发送 Q，得到答案。
5. 答案编码为高层文本语义，送回意识核参与一帧 s(t) 更新。
6. 将这次“自我-问题-老师-结果”写入情景记忆，并在终端打印。

运行示例（假设已经配置好环境变量中的 LLM API Key）：

```bash
python -m examples.minimal_text_agent
```

---

## 6. 工作流程详解

### 6.1 主循环（简化伪代码）

```python
while True:
    # 1. 收集外部输入（文字/视觉/声音）
    external_inputs = input_adapter.read()

    # 2. 更新原我状态 p(t)（系统资源、误差、奖励、任务压力）
    proto_state = proto_self.encode(system_metrics, learning_metrics)

    # 3. 感知通路 -> 高层语义/概念表征
    highlevel_features = perception.encode(external_inputs)

    # 4. 更新多轴意识核（竞争+同步，形成新一帧 s(t)）
    s_t = awareness_core.update(
        highlevel_features=highlevel_features,
        proto_state=proto_state
    )

    # 5. 自我模型评估不确定性与驱动
    uncertainty, drives = self_model.evaluate(s_t, memory)

    # 6. 调度器决定：是被动处理输入，还是主动思考（内部模拟/问老师）
    mode = scheduler.decide(
        external_salience=highlevel_features.salience,
        drives=drives
    )

    if mode == "internal_think":
        # 6a. 生成内部问题 Q
        question = question_generator.generate(s_t, uncertainty, memory)

        # 6b. 选择合适工具并调用
        tool = tool_manager.select_tool(question)
        answer = tool.call(question)

        # 6c. 把外部答案编码成内部表征，送回意识核
        encoded_answer = text_adapter.encode_answer(answer)
        s_t = awareness_core.integrate_answer(s_t, encoded_answer)

    # 7. 将 s(t) 及相关事件写入情景记忆
    memory.episodic.store(s_t, proto_state, external_inputs, actions, answer)

    # 8. 以合适频率触发回放/知识抽取（构建语义记忆 + 自传体自我）
    if time_to_replay():
        memory.semantic.update_from_replay(memory.episodic)
        self_model.update_autobiographical(memory)
```

---

## 7. 与第三方模型的集成方式

### 7.1 统一 Tool 接口

`tools/base_tool.py` 提供抽象基类：

```python
class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def call(self, query: "ToolQuery") -> "ToolResult":
        ...
```

具体实现例如 LLMTool：

- 持有模型类型（如 OpenAI、Gemini、本地 LLM 等）的配置。
- 在 `call()` 中使用 HTTP/SDK 请求，返回结构化结果。
- ToolManager 可以日志化每次调用（时间、消耗、效果评估）。

### 7.2 多教师模式

ToolManager 支持两种常用模式：

- **单教师模式**：根据任务类别选一个最优老师（例如：代码 → Code-LLM，规划 → 通用 LLM）。
- **多教师对比模式**：对同一问题调用多个老师，用验证器（另一 LLM 或内部规则）评估各答案，将评估结果写入记忆，更新对老师的“信任分布”。

---

## 8. Roadmap（后续规划）

**短期计划：**

- 完成原我编解码器 ProtoSNN 的最小实现（可以先用普通 NN/规则实现）。
- 实现多轴意识核骨架（状态向量 s(t) + 轴的抽象基类）。
- 提供最小文字通路（T0/T1/T2/T3）+ 单 LLM 工具示例。
- 实现情景记忆与自传体事件日志（JSON/本地文件版）。

**中期规划：**

- 引入简单 SNN 或 SSM 版本的世界模型，支持内部模拟/想象。
- 完善自我模型与不确定性评估逻辑。
- 加入多教师模式与工具表现统计。
- 实现基础的自传体自我图（重要事件聚类与阶段识别）。

**长期愿景：**

- 与独立的 self-snn / me-agent 等项目深度集成，作为其“意识/自我模块”。
- 支持多模态输入（视觉/声音/结构化数据），真正形成多模态意识帧。
- 探索更接近生物脑的 SNN 实现（包括同步振荡与动态核心的显式建模）。

---

## 9. 适用场景

- 为已有 LLM 系统增加“自我感知 + 记忆 + 问题生成”能力。
- 构建一个可以长期运行、不断自我反思和进化的工具型智能体。
- 学术探索：验证“动态核心 / 原我–核心自我–自传体自我 / 多教师学习”等假设。
- 游戏 / 仿真中的“有自我叙事”的 NPC 或虚拟助手。

---

## 10. 贡献与讨论

- 提交 Issue 讨论架构、模块设计与实现细节。
- 提交 PR 补充：不同工具接入适配、不同世界模型实现、更丰富的记忆与自传体分析方法。
- 分享基于本仓库构建的实验结果、论文原型或产品原型。
- （此处可根据你的实际账号信息补充联系方式和开源协议说明）
