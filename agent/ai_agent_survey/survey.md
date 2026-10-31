# AI 智能体：应用、实践与最近半年的动向

> 长文版。数据截至 2026-10-30；论文与资源库 1100 条（1027 条 arXiv 已用官方 API 核验标题与日期），其中最近一年（2025-11 →）789 条、最近半年（2026-05 →）600 条。本文重应用与实践：每章先讲已经成形的做法与代表工作，再讲最近半年的变化，最后给出实践要点与开放问题；第 12 章是一份可以照着做的入门指南，附录 A 按月整理最近半年的动向，附录 B 是全部条目的总表。正文中的数字摘自论文摘要或官方发布，设置各不相同，不可横向比较；最近半年的多数工作是尚未经过同行评审的预印本。本文不覆盖机器人与具身智能体。

## 0 导读

### 0.1 什么是智能体

本文采用工程上最常用的定义：**智能体是在循环中使用工具、根据环境反馈自主决定下一步的模型系统**。它由四部分组成：

- **模型**：读入目标与上下文，决定下一步做什么（思考、调用哪个工具、何时结束）。
- **工具**：把模型的决定变成对外部世界的动作，如读写文件、运行命令、搜索网页、点击屏幕、调用业务 API。
- **循环**：把动作的结果（观察）送回模型，直到任务完成、失败或需要人确认。
- **harness**：包住模型的全部程序，决定模型能看到什么（上下文）、能做什么（工具与权限）、什么时候停（预算与停止条件），以及出错时怎么恢复。

这个"思考—行动—观察"的循环最早由 [ReAct](https://arxiv.org/abs/2210.03629) 写成。今天的编码智能体、计算机使用智能体和深度研究智能体仍然是它，只是工具更多、循环更长、上下文管理更讲究。一个重要的经验是：在模型能力相近时，**智能体好不好用主要取决于工具、上下文与 harness**。

### 0.2 从一次调用到自主智能体

Anthropic 的 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 给出了一条被广泛接受的原则：先用能解决问题的最简单方案。按自主程度，可以分成四档：

| 形态 | 谁决定步骤 | 适合 | 代价 |
|---|---|---|---|
| 单次调用 | 无步骤 | 分类、抽取、摘要、问答 | 最便宜，最好评测 |
| 工作流 | 代码 | 链式、路由、并行、评审—修改；步骤可预知 | 可控，但不灵活 |
| 智能体 | 模型 | 编码、研究、操作电脑；步数与路径事先不知道 | 成本与风险随步数增长 |
| 多智能体 / 长时运行 | 多个模型 | 可并行拆分的宽任务；跨会话持续数小时到数天 | token 开销大，协作易出错 |

### 0.3 四个阶段

**2022 → 2023：范式成形。**[ReAct](https://arxiv.org/abs/2210.03629)、[Toolformer](https://arxiv.org/abs/2302.04761)、[Reflexion](https://arxiv.org/abs/2303.11366)、[Generative Agents](https://arxiv.org/abs/2304.03442)、[Voyager](https://arxiv.org/abs/2305.16291) 提出了循环、工具学习、自我反思、记忆与技能库等基本概念；[AutoGen](https://arxiv.org/abs/2308.08155)、[MetaGPT](https://arxiv.org/abs/2308.00352) 等多智能体框架流行。这一阶段的智能体主要靠提示词实现，在真实任务上很不可靠。

**2024：进入真实环境。**[SWE-bench](https://arxiv.org/abs/2310.06770)、[OSWorld](https://arxiv.org/abs/2404.07972)、[WebArena](https://arxiv.org/abs/2307.13854) 把评测放进真实仓库、操作系统与网站；[SWE-agent](https://arxiv.org/abs/2405.15793) 提出"智能体—计算机接口"，[Agentless](https://arxiv.org/abs/2407.01489) 提醒简单流水线也很强；年底出现了 [Claude 计算机使用](https://www.anthropic.com/news/3-5-models-and-computer-use)、[MCP](https://www.anthropic.com/news/model-context-protocol) 与 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)。

**2025 → 2026-04：产品化与智能体模型。**[Deep Research](https://openai.com/index/introducing-deep-research/)、[Claude Code](https://code.claude.com/docs/en/overview)、[Codex](https://openai.com/index/introducing-codex/)、[ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) 成为日常工具；模型开始为智能体专门训练（[Kimi K2](https://arxiv.org/abs/2507.20534)、[GLM-4.5](https://arxiv.org/abs/2508.06471)、[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)）；工程上出现了上下文工程（[Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)、[Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)）、[AGENTS.md](https://agents.md/)、[Agent Skills](https://agentskills.io/)；开源自托管的个人智能体 [OpenClaw](https://github.com/openclaw/openclaw) 走红；2025-12 成立 [Agentic AI Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)；2026-04 出现托管的 [Claude Managed Agents](https://claude.com/blog/claude-managed-agents)。

**2026-05 → 2026-10（最近半年）：harness 时代。**harness 与技能成为研究和工程的主战场，并开始自我演化；协议层收敛（[MCP 无状态化](https://modelcontextprotocol.io/specification/2026-07-28/changelog)、[A2A 加入 AAIF](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/)、[AAIF 扩展到十个项目](https://www.linuxfoundation.org/press/agentic-ai-foundation-announces-four-new-projects-agent-skills-open-responses-openenv-and-opensandbox)）；托管智能体普及；任务从分钟级走向天级；训练转向合成环境与大规模智能体 RL；评测走向真实工作与经济价值；安全从防注入走向系统层。

### 0.4 最近半年的八个动向

1. **Harness 成为一等公民，并开始自己改自己。**同一个模型换一个 harness，成绩差异很大（[StateM](https://arxiv.org/abs/2608.15089)）；于是 harness 本身成为优化对象：轨迹回放、递归自我改进、与权重联合训练。也有工作提醒"改了不等于变好"（[更新 harness 不等于 harness 有益](https://arxiv.org/abs/2605.30621)、[RRSI](https://arxiv.org/abs/2609.24972)）。见第 3、5、9 章。
2. **技能从文件格式变成生态。**技能进入 AAIF；研究转向技能的自动生成、演化、跨 harness 迁移与内化进权重；同时出现技能后门（[SkillJack](https://arxiv.org/abs/2608.03509)）与水印（[SkillMarker](https://arxiv.org/abs/2610.15725)）。见第 3、11 章。
3. **协议层收敛，托管智能体普及。**MCP 2026-07-28 版改为无状态；A2A 1.0 加入 AAIF；Claude、Gemini、OpenAI 都提供了托管的"循环 + 沙箱"。见第 4 章。
4. **从分钟到天。**多日自主开发（[Harness-of-Harness](https://arxiv.org/abs/2609.01481)）、超长程执行（[Marathoner](https://arxiv.org/abs/2609.34378)）成为目标；上下文管理转向"让模型自己决定忘记什么"；METR 的时间视野测量在超过 16 小时后已不可靠（[METR](https://metr.org/time-horizons/)）。见第 2、5、10 章。
5. **训练：合成环境 + 智能体 RL 规模化。**终端、软件、桌面、办公协作环境被大规模合成；在策略蒸馏的各种变体与现实检验；万亿参数级的智能体 RL（[NeMo-DCR](https://arxiv.org/abs/2610.08430)）；训练开始面向"模型 + harness"整体（[Harness Scaling](https://arxiv.org/abs/2610.29458)）。见第 9 章。
6. **常驻的个人智能体与主动性。**OpenClaw 带出了一批"Claw 类"系统、基准与安全工作；主动式智能体有了分层定义与训练环境（[主动式智能体基础](https://arxiv.org/abs/2609.37267)）。见第 2、8、11 章。
7. **评测走向真实工作与经济价值。**[Agents' Last Exam](https://arxiv.org/abs/2606.05405)、[OSWorld 2.0](https://arxiv.org/abs/2606.29537)、[BizBench](https://arxiv.org/abs/2610.20424)、[Business Arena](https://arxiv.org/abs/2608.08621)；同时反思污染、预测效度与评测成本。见第 10 章。
8. **安全从防注入走向系统层。**智能体操作系统（[AgentKernel](https://arxiv.org/abs/2609.29647)）、事务式执行（[Agentic Transaction](https://arxiv.org/abs/2608.13900)）、撤销能力（[UndoBench](https://arxiv.org/abs/2610.05622)）、欺骗与越界行为评测。见第 11 章。

### 0.5 本文的组织

第 1–4 章讲构件：模型、工具与循环；上下文与记忆；harness、技能与多智能体；协议、框架与产品。第 5–8 章讲四类应用：编码与终端；计算机、浏览器与手机；深度研究与科研；行业、个人与多模态。第 9–11 章讲训练、评测与安全。第 12 章是实践入门指南，第 13 章是开放问题。

## 1 构件：模型、工具与智能体循环

### 1.1 函数调用是一切的基础

今天的智能体建立在**函数调用**之上：开发者用 JSON Schema 描述每个工具的名称、用途和参数；模型在需要时输出结构化的调用；宿主程序执行后把结果作为观察放回对话。一轮可以并行发出多个调用，宿主把全部结果放进下一条消息。出错时把错误作为结果返回（而不是中断循环），模型通常能自己纠正。

这一形态由三条早期路线汇成：

- **推理与行动交替**：[ReAct](https://arxiv.org/abs/2210.03629) 让模型交替输出思考与动作；更早的 [WebGPT](https://arxiv.org/abs/2112.09332) 让模型使用文本浏览器并引用来源，[MRKL](https://arxiv.org/abs/2205.00445) 把 LLM 作为路由器连接外部模块。
- **学会使用工具**：[Toolformer](https://arxiv.org/abs/2302.04761) 让模型自监督地学会何时调用 API；[Gorilla](https://arxiv.org/abs/2305.15334) 用检索减少 API 幻觉；[ToolLLM](https://arxiv.org/abs/2307.16789) 构造了覆盖一万多个真实 API 的数据；[HuggingGPT](https://arxiv.org/abs/2303.17580) 让 LLM 调度专家模型；[LLM as Tool Makers](https://arxiv.org/abs/2305.17126) 与 [CRAFT](https://arxiv.org/abs/2309.17428) 让模型自己制造工具。
- **代码即动作**：[PAL](https://arxiv.org/abs/2211.10435) 把推理写成程序；[CodeAct](https://arxiv.org/abs/2402.01030) 发现以 Python 作为统一动作空间优于 JSON 工具调用；[DynaSaur](https://arxiv.org/abs/2411.01747) 让智能体用代码动态生成新动作。[LLM-in-Sandbox](https://arxiv.org/abs/2601.16206) 进一步发现，仅仅让模型在代码沙箱中探索就能激发通用的智能体能力。

### 1.2 规划、反思与推理模型

2023 年的大量工作用提示实现规划与反思：[Reflexion](https://arxiv.org/abs/2303.11366)（语言化的自我反思）、[Self-Refine](https://arxiv.org/abs/2303.17651)、[Tree of Thoughts](https://arxiv.org/abs/2305.10601)、[LATS](https://arxiv.org/abs/2310.04406)（蒙特卡洛树搜索）、[Plan-and-Solve](https://arxiv.org/abs/2305.04091)；效率方向有 [ReWOO](https://arxiv.org/abs/2305.18323)（先规划全部调用）与 [LLMCompiler](https://arxiv.org/abs/2312.04511)（把调用编译为可并行的 DAG）。

推理模型出现后，这些能力大部分被内化：模型在工具调用之间继续思考（交错思考），自己决定是否需要计划和回溯。新的问题变成"想多少"：[过度思考](https://arxiv.org/abs/2502.08235)在智能体任务中会损害表现；[智能体的测试时扩展](https://arxiv.org/abs/2506.12928)与 [预算感知的工具使用](https://arxiv.org/abs/2511.17006)研究如何把计算花在刀刃上。2025 年起，工具使用越来越多地由强化学习直接训练（[ReTool](https://arxiv.org/abs/2504.11536)、[ToolRL](https://arxiv.org/abs/2504.13958)、[ARTIST](https://arxiv.org/abs/2505.01441)），而不是靠提示。[API 智能体与 GUI 智能体](https://arxiv.org/abs/2503.11069)的分化与融合，则是第 5、6 章的背景。

### 1.3 最近半年

- **打破"想—做—看"的顺序。**[Second Thought](https://arxiv.org/abs/2608.13667)（2026-08）在执行动作、等待观察的同时并行推理；[LLM 是通用的异步智能体](https://arxiv.org/abs/2609.35427)（2026-09）让模型边想边做，减少等待工具返回造成的空转。
- **长程失败的细致刻画。**[DeLoop](https://arxiv.org/abs/2610.19376) 发现长程智能体常重复做过的工作，并把重复动作当作自监督信号；[Interlock](https://arxiv.org/abs/2610.19686) 研究为什么每一步局部正确却完不成整体任务；[Agent-Reflex](https://arxiv.org/abs/2610.17953) 只在关键步骤前反思；[工具使用智能体如何失败](https://arxiv.org/abs/2610.07753)分析改变外部状态（写数据库、发消息）时的失败模式。
- **主动性。**[主动式智能体基础](https://arxiv.org/abs/2609.37267)提出主动性的分层与 Proactivity-Gym，[横向与纵向主动性](https://arxiv.org/abs/2609.37236)区分"多做相关的事"与"把事做深"。
- **长程规划的机制。**[多轮长程规划的物理学](https://arxiv.org/abs/2607.24720)从预训练到后训练追踪规划能力的来源；[循环语言模型](https://arxiv.org/abs/2608.18171)在组合式工具调用上泛化更好；[EVOKE](https://arxiv.org/abs/2609.38334) 激发模型的世界知识用于多步决策。
- **工具使用数据来自合成环境。**[EnvFactory](https://arxiv.org/abs/2605.18703)、[WEFT](https://arxiv.org/abs/2609.36887)、[DART-SD](https://arxiv.org/abs/2608.18524)、[无环境 API 数据合成](https://arxiv.org/abs/2607.16900)，详见第 9 章。

### 1.4 实践要点与开放问题

实践上：工具要少而精、职责不重叠，描述写给"第一次见到它的新同事"；出错返回可读的错误；设置最大步数、超时与花费上限；无依赖的调用并行；步骤可预知时用代码写成工作流，只在需要判断的地方调用模型。

开放问题：何时先规划、何时直接行动？如何可靠地检测"原地打转"？异步、并行行动下，如何保证对外部状态的修改正确且可回滚？

## 2 上下文工程与记忆

### 2.1 上下文是唯一的工作记忆

模型每一步只能看到上下文窗口里的内容，而有效容量远小于标称长度：信息越多，越容易遗漏关键约束、被无关内容分心。**上下文工程**就是在每一步决定放进什么（[Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)）。可以概括为四种操作：

- **写**：把计划、进度、中间结果写到文件或笔记里，需要时再读。
- **选**：只在需要时检索（"即时"检索），而不是预先把资料塞满上下文。
- **压**：接近上限时摘要旧轮次、清掉过时的工具输出。
- **隔**：把探索性的子任务交给子智能体，它们在独立上下文中工作，只把结论交回。

### 2.2 为缓存而设计

长轨迹智能体的输入 token 远多于输出 token，KV 缓存命中率是头号成本指标。[Manus 的经验](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：前缀保持稳定（系统提示与工具列表不要频繁变动），历史只追加不修改，时间戳等易变内容放在后面；保留错误轨迹让模型从中学习；用不断重写的待办清单把目标"复述"到上下文末尾。[Cache Engineering for Agents](https://arxiv.org/abs/2610.28577)（2026-10）系统地研究了上下文布局与缓存命中、成本的关系。

### 2.3 状态外化与长期记忆

[MemGPT](https://arxiv.org/abs/2310.08560) 最早把上下文窗口类比为内存、外部存储类比为磁盘；[Generative Agents](https://arxiv.org/abs/2304.03442) 的"记忆流 + 反思 + 检索"奠定了智能体记忆的基本结构。长程任务中，[ReSum](https://arxiv.org/abs/2509.13313)、[AgentFold](https://arxiv.org/abs/2510.24699)、[IterResearch](https://arxiv.org/abs/2511.07327) 定期把历史重建为精简状态；[FS-Researcher](https://arxiv.org/abs/2602.01566) 直接把文件系统当作工作记忆。

长期记忆按内容可分为情节记忆（发生过什么）、语义记忆（事实与偏好）和程序性记忆（怎么做）。程序性记忆的代表是 [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)、[Memp](https://arxiv.org/abs/2508.06433)；[ACE](https://arxiv.org/abs/2510.04618) 把上下文当作不断演化的"剧本"，它进一步沉淀就是第 3 章的技能。产品化的记忆层有 [Mem0](https://github.com/mem0ai/mem0)（[论文](https://arxiv.org/abs/2504.19413)）与 [Letta](https://github.com/letta-ai/letta)；[GAM](https://arxiv.org/abs/2511.18423) 主张用深度研究式的即时检索代替静态记忆；[MemEvolve](https://arxiv.org/abs/2512.18746)、[MemSkill](https://arxiv.org/abs/2602.02474) 让记忆系统自身演化。综述见 [AI 智能体时代的记忆](https://arxiv.org/abs/2512.13564)。

### 2.4 最近半年

- **记忆要算净价值。**[Memory Tax](https://arxiv.org/abs/2610.34073)（2026-10）在多个基准与模型上发现，许多专门的记忆系统在不少情况下不如直接对原始历史做检索，即净价值为负，并把收益与"记忆税"分开度量；[朴素 RAG 比想象中强](https://arxiv.org/abs/2610.30755)（2026-10）在 LoCoMo、LongMemEval 等基准的七个设置中有六个排名第一；预注册实验 [选择取代提取](https://arxiv.org/abs/2609.34227)显示选对原始对话轮次往往就够。
- **让模型自己管理上下文。**[自压缩智能体](https://arxiv.org/abs/2606.23525)、[ContextPilot](https://arxiv.org/abs/2608.28476)（细粒度 RL）、[AutoMem](https://arxiv.org/abs/2607.01224)（把记忆当作可学习的技能）。
- **长程状态的新表示。**[显式信念状态](https://arxiv.org/abs/2610.01415)、[FlowState](https://arxiv.org/abs/2609.34565)（以执行状态为记忆）、[CarryOn](https://arxiv.org/abs/2610.18784)（面向对象的上下文）、[PARSER](https://arxiv.org/abs/2609.06702)（并行阅读）。
- **记忆是重建而非回放。**[图记忆](https://arxiv.org/abs/2606.06036)、[MemHarness](https://arxiv.org/abs/2607.28272)。
- **失效、交接与个性化。**[STALE](https://arxiv.org/abs/2605.06527)（发现记忆过期）、[SubtleMemory](https://arxiv.org/abs/2606.05761)（OpenClaw 类个人智能体）、[交接税](https://arxiv.org/abs/2608.24358)（接手他人轨迹时表现下降）、[先个性化再存储](https://arxiv.org/abs/2605.25535)、[HABIT-Bench](https://arxiv.org/abs/2610.24520)（从弱证据归纳用户习惯）。
- **以记忆实现持续学习。**[以记忆为中心的持续学习](https://arxiv.org/abs/2610.29357)、[可组合的持续学习机制](https://arxiv.org/abs/2609.06986)、[智能体上下文管理](https://arxiv.org/abs/2607.21503)。

### 2.5 实践要点与开放问题

实践上：稳定前缀、只追加历史、监控缓存命中率；工具输出截断或分页，大文件按需读取；长任务维护进度文件与待办清单；上长期记忆之前先用"原始对话 + 朴素检索"做基线；为记忆设更新与过期机制。

开放问题：压缩时丢掉什么、如何保证不丢关键约束？多用户、多智能体共享记忆的权限与隐私（[GateMem](https://arxiv.org/abs/2606.18829)）；记忆中的谄媚与污染（[MemSyco-Bench](https://arxiv.org/abs/2607.01071)）。

## 3 Harness、技能与多智能体

### 3.1 Harness：包住模型的程序

harness 包括系统提示、工具集、权限、上下文策略、子智能体、钩子、验证与重试。[SWE-agent](https://arxiv.org/abs/2405.15793) 的"智能体—计算机接口"（ACI）最早表明：为智能体专门设计查看、搜索、编辑命令，比直接给它人用的 shell 更有效。Anthropic 的[长时运行 harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)给出另一组经验：第一个会话负责初始化环境、写好功能清单与进度文件；之后每个会话读进度、推进一个功能、测试、提交，保证下一个会话能接手。

harness 的分量有多大？[StateM](https://arxiv.org/abs/2608.15089)（2026-08）不改模型权重，只把执行组织成持久状态、分阶段上下文和可检查的状态转移，就把同一模型在 Terminal-Bench 2.1 上从 83.1% 提到 92.1%（标题中的 95.3% 是其报告的最高原始准确率）。[编码 harness 设计的实证研究](https://arxiv.org/abs/2609.20804)拆解了哪些设计决定长程表现。

### 3.2 技能与项目上下文文件

**技能**是一个文件夹：SKILL.md 写明用途与步骤，可附带脚本与资源。关键是**渐进披露**：上下文里只放每个技能的名称和一句描述，任务相关时才读入全文，需要时再执行脚本（[Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)、[开放格式](https://agentskills.io/)、[官方示例](https://github.com/anthropics/skills)）。思想可追溯到 [Voyager](https://arxiv.org/abs/2305.16291) 的可执行技能库。**AGENTS.md**（以及 CLAUDE.md 等）写构建命令、代码风格、测试方法和禁区（[AGENTS.md](https://agents.md/)）；[Agent READMEs](https://arxiv.org/abs/2511.12884) 分析了这类文件在真实仓库中的内容与演化。

技能研究在 2025 年底到 2026 年初快速增长：[带技能库的 RL](https://arxiv.org/abs/2512.17102)、[SkillRL](https://arxiv.org/abs/2602.08234)、[SkillNet](https://arxiv.org/abs/2603.04448)、[Trace2Skill](https://arxiv.org/abs/2603.25158)、[Memento-Skills](https://arxiv.org/abs/2603.18743)、[SkillClaw](https://arxiv.org/abs/2604.08377)；评测有 [SkillsBench](https://arxiv.org/abs/2602.12670) 与[野外的智能体技能](https://arxiv.org/abs/2604.04323)。[LLM 智能体中的外化](https://arxiv.org/abs/2604.08224)把记忆、技能、协议与 harness 统一看作"把能力外化到模型之外"。

### 3.3 多智能体：何时值得

主流模式是**编排者—工人**：主智能体拆分任务、并行派出子智能体、汇总结果。Anthropic 的[多智能体研究系统](https://www.anthropic.com/engineering/multi-agent-research-system)在需要广泛搜索的任务上明显优于单智能体，但 token 用量约为普通聊天的 15 倍，因此只适合价值高、可并行的任务。[Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) 从反面论证：子智能体之间缺少共享上下文，容易做出互相冲突的决定；[MAST](https://arxiv.org/abs/2503.13657) 给出了多智能体系统的失败分类。早期框架包括 [CAMEL](https://arxiv.org/abs/2303.17760)、[ChatDev](https://arxiv.org/abs/2307.07924)、[MetaGPT](https://arxiv.org/abs/2308.00352)、[AutoGen](https://arxiv.org/abs/2308.08155)、[AgentVerse](https://arxiv.org/abs/2308.10848) 与[多智能体辩论](https://arxiv.org/abs/2305.14325)；后来转向可训练的组织：[AgentFlow](https://arxiv.org/abs/2510.05592)、[Chain-of-Agents](https://arxiv.org/abs/2508.13167)、[智能体组织](https://arxiv.org/abs/2510.26658)、[AOrchestra](https://arxiv.org/abs/2602.03786)、[递归多智能体](https://arxiv.org/abs/2604.25917)。

### 3.4 最近半年

这是本文收录最密集的方向（约 90 条）。

- **harness 自动优化与递归自我改进。**从轨迹回放中改进（[回顾式 harness 优化](https://arxiv.org/abs/2606.05922)、[Continual Harness](https://arxiv.org/abs/2605.09998)、[AutoSaddler](https://arxiv.org/abs/2608.23041)），到递归自我改进（[递归 harness 自我改进](https://arxiv.org/abs/2607.15524)、[HarnessDev](https://arxiv.org/abs/2609.01437)、[ModularRSI](https://arxiv.org/abs/2609.14857)、[Harness²](https://arxiv.org/abs/2610.14224)）、自然选择（[DarwinX](https://arxiv.org/abs/2608.07545)）和兼顾成本的帕累托优化（[HarnessPro](https://arxiv.org/abs/2610.26770)）；[Raven](https://arxiv.org/abs/2609.33439) 做跨领域复用的"harness 的 harness"，[Harness Handbook](https://arxiv.org/abs/2607.13285) 让不断演化的 harness 保持可读。
- **"改了"不等于"变好"。**[更新 harness 不等于 harness 有益](https://arxiv.org/abs/2605.30621)把"能改"与"改得有益"分开评测；[RRSI](https://arxiv.org/abs/2609.24972) 用正则化防止 harness 过拟合；[CoEvo-RSI](https://arxiv.org/abs/2610.25078) 讨论模型与 harness 协同演化时的问题。
- **harness 与权重的边界移动。**[Harness-Zero](https://arxiv.org/abs/2609.24974) 把 harness 能力蒸馏进模型；[WHALE](https://arxiv.org/abs/2609.00196) 联合优化 harness 与权重；[测试时 AI4AI](https://arxiv.org/abs/2608.12307) 通过 harness 把强模型的能力迁移给弱模型。
- **技能的生成、优化、迁移与内化。**[模型生成技能的系统研究](https://arxiv.org/abs/2605.23899)与[揭秘智能体技能](https://arxiv.org/abs/2608.14036)回答技能何时有效、何时失效；[SkillOpt](https://arxiv.org/abs/2605.23904)、[SkillGrad](https://arxiv.org/abs/2605.27760)、[SkillForge](https://arxiv.org/abs/2610.09832) 优化技能；[从代码大规模合成技能](https://arxiv.org/abs/2609.05571)、[Repo-To-Skill](https://arxiv.org/abs/2609.02749)、[COLLEAGUE.SKILL](https://arxiv.org/abs/2605.31264) 批量生产技能；[跨 harness 技能优化](https://arxiv.org/abs/2609.38024)研究迁移；[Skill0.5](https://arxiv.org/abs/2605.28424)、[LatentSkill](https://arxiv.org/abs/2606.06087) 把技能内化进权重；[SkillsVote](https://arxiv.org/abs/2605.18401) 管理技能生命周期；[SkillZip](https://arxiv.org/abs/2608.05604) 压缩技能库。
- **多智能体规模化与可学习编排。**[Agensh](https://arxiv.org/abs/2609.26781) 去掉中心编排者，让 1024 个并发智能体自行认领子任务、共享发现、合并进度；[DynaFlow](https://arxiv.org/abs/2610.34467) 学习自由形式的编排；[MetaAgent-X](https://arxiv.org/abs/2605.14212) 端到端 RL 训练多智能体系统；[多智能体 RL 何时改进工作流](https://arxiv.org/abs/2605.24202)给出规模与策略共享的权衡；[流式通信](https://arxiv.org/abs/2606.05158)与[跨模型 KV 缓存迁移](https://arxiv.org/abs/2609.32259)降低协作延迟；[组相对技能信用分配](https://arxiv.org/abs/2610.27227)回答"团队里该改哪个智能体"；[MiniCorp](https://arxiv.org/abs/2610.05912) 训练由智能体组成的企业。

### 3.5 实践要点与开放问题

实践上：从成熟 harness 起步（Claude Code / Agent SDK、Codex、OpenHands），先改提示、工具与权限，再考虑改循环；AGENTS.md 简短、具体、可执行；反复出现的流程沉淀为技能，每个技能一句话说清"何时用"；多智能体只用于可并行、上下文可隔离的子任务；任何 harness 改动都用固定评测集回归并比较成本。

开放问题：harness 自我改进如何避免过拟合、奖励作弊与能力倒退？技能库膨胀后的检索、冲突与供应链安全？多智能体的信用分配与失败归因（[LongRCA Bench](https://arxiv.org/abs/2608.15242)）？

## 4 协议、框架与产品生态

### 4.1 三层生态

- **协议与格式**：[MCP](https://www.anthropic.com/news/model-context-protocol)（2024-11，Anthropic）连接模型与工具、数据源，服务器暴露工具、资源与提示，任何客户端都能使用；[A2A](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)（2025-04，Google）连接智能体与智能体，用 Agent Card 发现能力、委托任务；[AGENTS.md](https://agents.md/) 与 [Agent Skills](https://agentskills.io/) 规定项目说明与技能的格式。
- **框架与 SDK**：自写循环；[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)（把 Claude Code 的 harness 作为库）、[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)（智能体、交接、护栏）、[Google ADK](https://github.com/google/adk-python)；编排框架 [LangGraph](https://github.com/langchain-ai/langgraph)（显式状态图、持久化、人工介入）、[Microsoft Agent Framework](https://github.com/microsoft/agent-framework)（AutoGen 与 Semantic Kernel 合并而来）、[CrewAI](https://github.com/crewAIInc/crewAI)、[Pydantic AI](https://github.com/pydantic/pydantic-ai)、[smolagents](https://github.com/huggingface/smolagents)、[DSPy](https://github.com/stanfordnlp/dspy)；研究框架 [AgentScope 1.0](https://arxiv.org/abs/2508.16279)、[AgentSPEX](https://arxiv.org/abs/2604.13346)。[OAgents](https://arxiv.org/abs/2506.15741) 实证比较了哪些设计真正起作用。
- **产品与托管平台**：编码智能体 [Claude Code](https://code.claude.com/docs/en/overview)、[Codex CLI](https://github.com/openai/codex) 与 [Codex 云端](https://openai.com/index/introducing-codex/)、[Gemini CLI](https://github.com/google-gemini/gemini-cli)、[Google Antigravity](https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/)，开源的 [OpenHands](https://github.com/OpenHands/OpenHands)、[Aider](https://github.com/Aider-AI/aider)、[Cline](https://github.com/cline/cline)、[goose](https://github.com/aaif-goose/goose)；浏览器与通用智能体 [Operator](https://openai.com/index/introducing-operator/)、[ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/)、[ChatGPT Atlas](https://openai.com/index/introducing-chatgpt-atlas/)；个人智能体 [OpenClaw](https://github.com/openclaw/openclaw)；托管智能体 [Claude Managed Agents](https://claude.com/blog/claude-managed-agents)；沙箱 [E2B](https://github.com/e2b-dev/E2B)；可观测性 [Langfuse](https://github.com/langfuse/langfuse)、[Phoenix](https://github.com/Arize-ai/phoenix)；评测框架 [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai)。

2025-12 成立的 [Agentic AI Foundation（AAIF）](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)隶属 Linux 基金会，首批项目为 MCP、goose 与 AGENTS.md，把主要开放协议放进同一个中立基金会。

### 4.2 最近半年

- **MCP 改为无状态（2026-07）。**[2026-07-28 版规范](https://modelcontextprotocol.io/specification/2026-07-28/changelog)是发布以来最大的修订：去掉会话（Mcp-Session-Id）与 initialize 握手，协议版本与客户端能力随每个请求携带；新增服务器发现；长任务（tasks）移到官方扩展。这让 MCP 服务器更容易水平扩展和部署在无服务器环境中。
- **开放协议汇集到 AAIF（2026-08 → 10）。**A2A 以 v1.0 [加入 AAIF](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/)（2026-08）；10 月 [又加入四个项目](https://www.linuxfoundation.org/press/agentic-ai-foundation-announces-four-new-projects-agent-skills-open-responses-openenv-and-opensandbox)：Agent Skills（Anthropic）、Open Responses（OpenAI）、OpenEnv（Hugging Face）、OpenSandbox（阿里巴巴），基金会项目增至十个。工具、技能、智能体通信、模型响应接口、沙箱和训练环境都有了中立的开放规范。
- **Google I/O 2026（2026-05）。**[Antigravity 2.0](https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-developer-highlights/)（并行多智能体）、Antigravity CLI 与 SDK，以及 [Gemini API 托管智能体](https://blog.google/innovation-and-ai/technology/developers-tools/managed-agents-gemini-api/)；个人用户的 Gemini CLI 自 2026-06-18 起[迁移到 Antigravity CLI](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/)。
- **OpenAI DevDay 2026（2026-09）。**据发布会报道，[Codex](https://openai.com/index/devday-2026-recap/) 增加后台与云端任务管理和 /agents 视图，Agents API 增加计算机使用与多智能体能力。编码智能体正从"对话中的助手"变成"在后台并行跑的工人"（参见 [Codex](https://developers.openai.com/codex/changelog) 与 [Claude Code](https://code.claude.com/docs/en/changelog) 的更新日志）。
- **运行时与可观测性研究。**[HarnessX](https://arxiv.org/abs/2606.14249)（可组合的 harness 工厂）、[OpenRath](https://arxiv.org/abs/2606.19409)（以会话为中心的运行时状态）、[AgentDebugX](https://arxiv.org/abs/2607.18754)（失败观测、归因与恢复）、[OO Agents](https://arxiv.org/abs/2607.20709)（Python 面向对象智能体）、[Agentic Transaction](https://arxiv.org/abs/2608.13900)（满足 ACID 的智能体操作）、[Foundation Protocol](https://arxiv.org/abs/2605.23218)（"智能体社会"的协调层）、[MCP-Cosmos](https://arxiv.org/abs/2605.09131)（在 MCP 环境中引入世界模型）。

### 4.3 实践要点与开放问题

实践上：内部工具优先做成 MCP 服务器，一次实现、多处复用；按需要的控制粒度选框架，原型用 Agent SDK 或托管服务，复杂业务流程用状态图框架；能用托管沙箱就别自己维护容器；从第一天起追踪每次模型调用与工具调用。

开放问题：第三方 MCP 服务器与技能的信任（工具描述投毒、权限过宽、供应链攻击）；跨厂商智能体之间的身份、授权与计费；无状态协议下长任务的状态归属。

## 5 应用：编码与终端智能体

### 5.1 为什么编码最先成熟

代码有天然的验证信号（编译、测试、类型检查、lint），环境是纯文本，训练数据海量，且用户（开发者）本身能审阅产出。因此编码智能体最先从演示走向日常：从补全，到修 issue，再到在后台并行处理多个任务、连续多日自主开发。

三种开源形态奠定了基础：[SWE-agent](https://arxiv.org/abs/2405.15793)（专门设计的智能体接口）、[OpenHands](https://arxiv.org/abs/2407.16741)（通用开发平台）、[Agentless](https://arxiv.org/abs/2407.01489)（定位—修复—验证三步流水线，不用复杂循环也有竞争力）。产品侧有 [Claude Code](https://code.claude.com/docs/en/overview)、[Codex](https://openai.com/index/introducing-codex/)、[Gemini CLI](https://github.com/google-gemini/gemini-cli)、[Aider](https://github.com/Aider-AI/aider)、[Cline](https://github.com/cline/cline)；[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 证明只用 bash、约百行代码也能取得不错的成绩。

### 5.2 训练：可验证环境是燃料

[SWE-Gym](https://arxiv.org/abs/2412.21139)、[R2E-Gym](https://arxiv.org/abs/2504.07164)、[SWE-smith](https://arxiv.org/abs/2504.21798)、[SWE-Factory](https://arxiv.org/abs/2506.10954)、[SWE-Universe](https://arxiv.org/abs/2602.02361)（百万级）、[SWE-World](https://arxiv.org/abs/2602.03419)（免 Docker）把真实仓库变成训练环境；[SWE-RL](https://arxiv.org/abs/2502.18449) 用软件演化数据做 RL，[Skywork-SWE](https://arxiv.org/abs/2506.19290) 观察到数据扩展律，[长上下文多轮 SWE RL](https://arxiv.org/abs/2508.03501) 解决长轨迹训练，[daVinci-Dev](https://arxiv.org/abs/2601.18418) 做智能体原生的中期训练，[SWE-Master](https://arxiv.org/abs/2602.03411) 给出可复现的后训练配方。终端方向有 [TermiGen](https://arxiv.org/abs/2602.07274) 与[终端数据工程](https://arxiv.org/abs/2602.21193)；能力外延到 GPU 内核（[CUDA Agent](https://arxiv.org/abs/2602.24286)）与竞赛编程（[GrandCode](https://arxiv.org/abs/2604.02721)）。

评测从 [SWE-bench](https://arxiv.org/abs/2310.06770) 与 [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) 扩展到多语言（[Multi-SWE-bench](https://arxiv.org/abs/2504.02605)）、多模态（[SWE-bench Multimodal](https://arxiv.org/abs/2410.03859)）、真实外包任务（[SWE-Lancer](https://arxiv.org/abs/2502.12115)）、持续更新防污染（[SWE-rebench](https://arxiv.org/abs/2505.20411)）和终端（[Terminal-Bench](https://www.tbench.ai/)、[论文](https://arxiv.org/abs/2601.11868)）。[终端智能体足以胜任企业自动化](https://arxiv.org/abs/2604.00073)指出：很多企业流程用终端智能体比用 GUI 更可靠。

### 5.3 最近半年

- **终端训练数据的工业化。**[CLI-Universe](https://arxiv.org/abs/2606.22883)、[Terminal-Universe](https://arxiv.org/abs/2609.04148)（把智能体轨迹转化为环境）、[终端环境演化](https://arxiv.org/abs/2609.04128)、[长程终端任务递归合成](https://arxiv.org/abs/2608.05466)、[CalibForge](https://arxiv.org/abs/2608.06352)（对抗式求解器校准难度）、[FACET](https://arxiv.org/abs/2608.18580)（合成时保持原意与可执行状态）、[扩展长程终端数据](https://arxiv.org/abs/2610.13166)；RL 侧有 [T1](https://arxiv.org/abs/2609.11042) 与学习从错误状态恢复的 [ReTerm](https://arxiv.org/abs/2610.18484)。
- **从修 bug 到从零构建。**[DeNovoSWE](https://arxiv.org/abs/2606.10728)、[ProgramGym](https://arxiv.org/abs/2610.13395) 训练"从零生成完整仓库/复现程序"；[Super Library Agent](https://arxiv.org/abs/2608.29310) 同时维护多个相关应用；[Harness-of-Harness](https://arxiv.org/abs/2609.01481) 做多日自主开发；[SpecFirst](https://arxiv.org/abs/2607.27167) 主张先引出行为规格再写程序。
- **验证成为瓶颈。**[验证视野](https://arxiv.org/abs/2606.26300)指出编码奖励没有银弹，验证并不总比生成容易；[Dockerless](https://arxiv.org/abs/2606.28436) 训练免环境的程序验证器；[Lean4AgenticRL](https://arxiv.org/abs/2610.22448) 使用可验证的多层过程奖励；[组内智能体评分](https://arxiv.org/abs/2609.32577)改进代码智能体的优势估计。
- **仓库上下文与交接。**[FastContext](https://arxiv.org/abs/2606.14066)、[SWE-Pruner Pro](https://arxiv.org/abs/2607.18213)、[CodeNib](https://arxiv.org/abs/2607.25431)；[交接税](https://arxiv.org/abs/2608.24358)发现接手别人的轨迹时表现明显下降。
- **会改进自己的编码智能体。**[Mendel Gödel Machine](https://arxiv.org/abs/2608.07645)、[Ouroboros](https://arxiv.org/abs/2608.08311)、[Terminal-RSI](https://arxiv.org/abs/2610.22192)、[SoL-Pi](https://arxiv.org/abs/2609.20519)、[Harness-策略联合优化](https://arxiv.org/abs/2610.22791)。
- **评测更接近真实开发。**[SWE-bench 是学会了还是背下了](https://arxiv.org/abs/2609.27891)检验污染；[SWE-Touch](https://arxiv.org/abs/2608.02499)（用户同时改代码）、[WideSWE](https://arxiv.org/abs/2609.33382)（跨仓库）、[RealSWE](https://arxiv.org/abs/2608.27831)（现实用户请求）、[AgentSWE](https://arxiv.org/abs/2610.13361)（是否做出用户想要的东西）、[SWE-Explore](https://arxiv.org/abs/2606.07297)（如何探索代码库）、[TUA-Bench](https://arxiv.org/abs/2606.28480) 与 [Long-Horizon-Terminal-Bench](https://arxiv.org/abs/2607.08964)；游戏开发方向有 [SWE-Game](https://arxiv.org/abs/2609.33678)、[GameCraft-Bench](https://arxiv.org/abs/2606.17861)；[SWE-MISBEHAVIOR](https://arxiv.org/abs/2610.18206) 给编码智能体的越界行为分类。

### 5.4 实践要点与开放问题

推荐的协作流程是"探索 → 计划 → 编码 → 提交"（[Claude Code 最佳实践](https://www.anthropic.com/engineering/claude-code-best-practices)）：先让智能体读代码、写计划，人审过计划再动手；测试先行，"测试通过"是明确的完成标准，禁止修改测试来过关；小步提交，一切改动可回滚；并行任务用独立的 git worktree 或容器；认真审 diff，不跳过 CI 与代码评审；常用命令和禁区写进 AGENTS.md。

开放问题：验证不完备时的奖励作弊；长期维护大型代码库；代码评审负担从"写"转移到"审"之后，团队流程与责任如何调整。

## 6 应用：计算机、浏览器与手机

### 6.1 两条路线

GUI 智能体通过截图（或可访问性树、DOM）感知界面，用点击、键入、滚动行动，能操作任何为人设计的软件。[Claude 计算机使用](https://www.anthropic.com/news/3-5-models-and-computer-use)（2024-10）首次公开了前沿模型的这一能力，随后 [Operator](https://openai.com/index/introducing-operator/)、[ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/)、[ChatGPT Atlas](https://openai.com/index/introducing-chatgpt-atlas/) 把它做进产品；[案例研究](https://arxiv.org/abs/2411.10323)记录了早期的能力与局限。

- **模块化**：规划模型决定做什么，专门的定位模型决定点哪里。代表有 [SeeAct](https://arxiv.org/abs/2401.01614)、[Agent S](https://arxiv.org/abs/2410.08164)、[CODA](https://arxiv.org/abs/2508.20096)（"大脑 + 小脑"）、[PC-Agent](https://arxiv.org/abs/2502.14282)、[UFO2](https://arxiv.org/abs/2504.14603)。
- **端到端原生 GUI 模型**：只看截图直接输出动作。代表有 [CogAgent](https://arxiv.org/abs/2312.08914)、[OS-ATLAS](https://arxiv.org/abs/2410.23218)、[Aguvis](https://arxiv.org/abs/2412.04454)、[UI-TARS](https://arxiv.org/abs/2501.12326) 与 [UI-TARS-2](https://arxiv.org/abs/2509.02544)、[OpenCUA](https://arxiv.org/abs/2508.09123)、[GUI-Owl](https://arxiv.org/abs/2508.15144) 与 [GUI-Owl-1.5](https://arxiv.org/abs/2602.16855)、[MAI-UI](https://arxiv.org/abs/2512.22047)、[Step-GUI](https://arxiv.org/abs/2512.15431)。

瓶颈常在**定位**：看懂了却点错了。[GUI-Actor](https://arxiv.org/abs/2506.03143)、[Phi-Ground](https://arxiv.org/abs/2507.23779)、[UI 分解与合成](https://arxiv.org/abs/2505.13227)、[GUI-G²](https://arxiv.org/abs/2507.15846) 专门解决这个问题。浏览器任务可以绕开像素，直接用可访问性树（[Playwright MCP](https://github.com/microsoft/playwright-mcp)、[browser-use](https://github.com/browser-use/browser-use)）。

### 6.2 数据与训练

数据来源包括教程回放（[AgentTrek](https://arxiv.org/abs/2412.09605)）、反向任务合成（[OS-Genesis](https://arxiv.org/abs/2412.19723)）、人工示范视频（[CUA-Suite](https://arxiv.org/abs/2603.24440)）、跨平台扩展（[ScaleCUA](https://arxiv.org/abs/2509.15221)）；在线 RL 有 [DigiRL](https://arxiv.org/abs/2406.11896)、[WebRL](https://arxiv.org/abs/2411.02337)、[ZeroGUI](https://arxiv.org/abs/2505.23762)、[UI-S1](https://arxiv.org/abs/2509.11543)；自演化有 [SEAgent](https://arxiv.org/abs/2508.04700)、[EvoCUA](https://arxiv.org/abs/2601.15876)、[UI-Voyager](https://arxiv.org/abs/2603.24533)；[Code2World](https://arxiv.org/abs/2602.09856) 用可渲染代码构建 GUI 世界模型。评测有 [WebArena](https://arxiv.org/abs/2307.13854)、[Mind2Web](https://arxiv.org/abs/2306.06070)、[OSWorld](https://arxiv.org/abs/2404.07972)、[Windows Agent Arena](https://arxiv.org/abs/2409.08264)、[AndroidLab](https://arxiv.org/abs/2410.24024)、[MMBench-GUI](https://arxiv.org/abs/2507.19478)。

### 6.3 最近半年

- **可验证环境规模化。**[CUA-Gym](https://arxiv.org/abs/2605.25624)、[OpenComputer](https://arxiv.org/abs/2605.19769)、[MobileGym](https://arxiv.org/abs/2605.26114)、[RecreationWorld](https://arxiv.org/abs/2609.22000)、[SoftwareGym](https://arxiv.org/abs/2610.12611)、[DeskForge](https://arxiv.org/abs/2610.02320)、[Video2GUI](https://arxiv.org/abs/2605.14747)；[A Decade in a Day](https://arxiv.org/abs/2610.15765) 从软件演化史中合成智能体原生的软件。
- **GUI 世界模型。**[AutoGUIWorld](https://arxiv.org/abs/2610.01215)（图像生成器作视觉世界模型）、[潜空间 GUI 世界动作模型](https://arxiv.org/abs/2610.14228)、[世界模型做验证器](https://arxiv.org/abs/2610.11942)、[野外的智能体世界模型](https://arxiv.org/abs/2610.26828)、[SeerGuard](https://arxiv.org/abs/2607.15550)（执行前预测后果）。
- **开放模型与全栈。**[OpenGCUA](https://arxiv.org/abs/2610.22200)、[OpenMobile-2](https://arxiv.org/abs/2610.14157)、[Qwen-UI-Agent](https://arxiv.org/abs/2607.28227)、[Xiaomi-GUI-0](https://arxiv.org/abs/2606.31410)、[UI-Mate](https://arxiv.org/abs/2608.15930)、[MintAct](https://arxiv.org/abs/2609.22083)、[开放手机使用智能体](https://arxiv.org/abs/2606.23049)；[OpenWebRL](https://arxiv.org/abs/2606.02031) 公开视觉网页智能体在线多轮 RL 的配方；[UI-MOPD](https://arxiv.org/abs/2607.04425) 用多平台在策略蒸馏做持续学习；[LLaDA-UI](https://arxiv.org/abs/2609.13287) 把扩散解码带入 GUI 智能体。
- **GUI 与 CLI/API 混合。**[ToolCUA](https://arxiv.org/abs/2605.12481)、[HybridCUA](https://arxiv.org/abs/2609.38008)、[GUI vs. CLI](https://arxiv.org/abs/2606.24551)、[StateAct](https://arxiv.org/abs/2607.22798)（先看程序状态再看像素）、[演化智能体接口](https://arxiv.org/abs/2610.28601)（为人设计的界面限制了智能体）。
- **经验与技能。**[Work2Skill](https://arxiv.org/abs/2610.12984)、[GUI-HARVEST](https://arxiv.org/abs/2610.00948)、[Reflect, Revise, Reuse](https://arxiv.org/abs/2609.17653)、[MemGUI-Agent](https://arxiv.org/abs/2606.19926)、[记忆模态选择](https://arxiv.org/abs/2610.17928)、[KnowAct-GUIClaw](https://arxiv.org/abs/2607.12625)。
- **评测。**[OSWorld 2.0](https://arxiv.org/abs/2606.29537)、[OSWorld-Pro](https://arxiv.org/abs/2609.24890)、[OSWorld-Science](https://arxiv.org/abs/2609.39903)、[WeaveBench](https://arxiv.org/abs/2606.09426)、[Workflow-GYM](https://arxiv.org/abs/2606.11042)、[CUA-SWE](https://arxiv.org/abs/2609.32600)、[CADWorld](https://arxiv.org/abs/2609.16251)、[iPrefBench](https://arxiv.org/abs/2610.34735)；[MAI-Gym](https://arxiv.org/abs/2610.20939) 把评测与在线 RL 做成一体化平台。

### 6.4 实践要点与开放问题

有 API 或命令行就别用 GUI；浏览器任务优先用 DOM / 可访问性树；在隔离的虚拟机或独立浏览器配置中运行，不登录不必要的账户；支付、发送、删除、提交表单前必须请人确认；网页与文档内容一律当作不可信输入。

开放问题：长程 GUI 任务的稳定性（弹窗、加载延迟、界面改版）；应用是否会为智能体提供原生接口，使 GUI 操作成为过渡形态。

## 7 应用：深度研究与科研智能体

### 7.1 深度研究

深度研究智能体在搜索、阅读、推理之间循环几十到上百步，最后写成带引用的报告。[OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)（2025-02）之后它成为各家标配。开源侧的主线是用 RL 训练：[Search-R1](https://arxiv.org/abs/2503.09516)、[R1-Searcher](https://arxiv.org/abs/2503.05592)、[DeepResearcher](https://arxiv.org/abs/2504.03160)（真实网络环境）、[ZeroSearch](https://arxiv.org/abs/2505.04588)（不调用真实搜索引擎）、[Search-o1](https://arxiv.org/abs/2501.05366)；再到 [WebSailor](https://arxiv.org/abs/2507.02592)、[WebSailor-V2](https://arxiv.org/abs/2509.13305)、[WebExplorer](https://arxiv.org/abs/2509.06501)、[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)、[MiroThinker](https://arxiv.org/abs/2511.11793)、[DR Tulu](https://arxiv.org/abs/2511.19399)（演化的评分细则）、[OpenSeeker](https://arxiv.org/abs/2603.15594)、[OpenResearcher](https://arxiv.org/abs/2603.20278)、[DR-Venus](https://arxiv.org/abs/2604.19859)（端侧）。

两个关键：**难题驱动的数据**（简单问题一搜就有答案，训练需要多跳、模糊、需要交叉验证的问题，如 [WebShaper](https://arxiv.org/abs/2507.15061)；对应评测 [BrowseComp](https://arxiv.org/abs/2504.12516)）与**长程上下文管理**（[ReSum](https://arxiv.org/abs/2509.13313)、[AgentFold](https://arxiv.org/abs/2510.24699)、[IterResearch](https://arxiv.org/abs/2511.07327)）。"找一个答案"与"把一张表填满"是两类任务（[WideSearch](https://arxiv.org/abs/2508.07999)），后者适合多智能体并行（[WideSeek-R1](https://arxiv.org/abs/2602.04634)、[Anthropic 研究系统](https://www.anthropic.com/engineering/multi-agent-research-system)）。长文写作方向有 [STORM](https://arxiv.org/abs/2402.14207) 与 [WebWeaver](https://arxiv.org/abs/2509.13312)。评测：[GAIA](https://arxiv.org/abs/2311.12983)、[DeepResearch Bench](https://arxiv.org/abs/2506.11763)、[Mind2Web 2](https://arxiv.org/abs/2506.21506)、[BrowseComp-Plus](https://arxiv.org/abs/2508.06600)、[DeepResearchEval](https://arxiv.org/abs/2601.09688)。综述：[深度研究系统](https://arxiv.org/abs/2512.02038)、[深度研究的 RL 基础](https://arxiv.org/abs/2509.06733)。

### 7.2 科研智能体

科研智能体提出假设、写代码跑实验、写论文。代表有 [The AI Scientist](https://arxiv.org/abs/2408.06292) 与 [v2](https://arxiv.org/abs/2504.08066)、[AI co-scientist](https://arxiv.org/abs/2502.18864)、[Agent Laboratory](https://arxiv.org/abs/2501.04227)、[NovelSeek](https://arxiv.org/abs/2505.16938)、[InternAgent-1.5](https://arxiv.org/abs/2602.08990)；数据科学有 [Data Interpreter](https://arxiv.org/abs/2402.18679)、[AutoKaggle](https://arxiv.org/abs/2410.20424)、[DeepAnalyze](https://arxiv.org/abs/2510.16872)；数学有[奥赛级长程推理](https://arxiv.org/abs/2512.10739)、[自主数学研究](https://arxiv.org/abs/2602.10177)；化学有 [ChemCrow](https://arxiv.org/abs/2304.05376)。能力度量有 [MLE-bench](https://arxiv.org/abs/2410.07095)、[RE-Bench](https://arxiv.org/abs/2411.15114)、[PaperBench](https://arxiv.org/abs/2504.01848)、[MLGym](https://arxiv.org/abs/2502.14499)。可执行、可度量的领域（机器学习工程、数据科学、形式化数学）进展最快，湿实验科学仍以辅助为主。综述：[从 AI for Science 到智能体科学](https://arxiv.org/abs/2508.14111)、[数据智能体](https://arxiv.org/abs/2510.23587)。

### 7.3 最近半年

- **检索回到"直接和语料打交道"。**[直接语料交互](https://arxiv.org/abs/2605.05242)、[GrepSeek](https://arxiv.org/abs/2605.29307)、[相关性的新角色](https://arxiv.org/abs/2607.24223)、[Follow the Entities](https://arxiv.org/abs/2609.37226)、[自演化搜索索引](https://arxiv.org/abs/2609.19656)。
- **少搜、搜对。**[SAAS](https://arxiv.org/abs/2605.29796)、[屏蔽陈旧观测](https://arxiv.org/abs/2606.00408)、[性能与成本](https://arxiv.org/abs/2610.05750)、[SearchJev](https://arxiv.org/abs/2610.05107)。
- **数据与环境。**[QUEST](https://arxiv.org/abs/2605.24218)（完全合成的任务）、[OpenSeeker-v2](https://arxiv.org/abs/2605.04036)、[FORT-Searcher](https://arxiv.org/abs/2606.12087)、[DeepSearch-World](https://arxiv.org/abs/2607.07820)、[ABSeeker](https://arxiv.org/abs/2608.05102)、[Harness-1](https://arxiv.org/abs/2606.02373)、[LongTraceRL](https://arxiv.org/abs/2605.31584)。
- **并行与多模态。**[SearchSwarm](https://arxiv.org/abs/2606.09730)、[SearchOS-V1](https://arxiv.org/abs/2607.15257)、[OpenSearch-VL](https://arxiv.org/abs/2605.05185)、[HyperEyes](https://arxiv.org/abs/2605.07177)、[Video-DeepResearch](https://arxiv.org/abs/2608.03979)、[OneSearch-VL](https://arxiv.org/abs/2610.12419)。
- **自动研究。**[ARIS](https://arxiv.org/abs/2605.03042)、[ScientistOne](https://arxiv.org/abs/2605.26340)、[AutoResearchClaw](https://arxiv.org/abs/2605.20025)、[Dr. Claw](https://arxiv.org/abs/2609.00365)、[Spark-to-Paper](https://arxiv.org/abs/2608.11924)、[OmniScientist](https://arxiv.org/abs/2608.13558)、[EurekAgent](https://arxiv.org/abs/2606.13662)；面向机器学习自身：[Athanor](https://arxiv.org/abs/2610.21055)、[AutoSFT](https://arxiv.org/abs/2610.22194)、[MLExpert](https://arxiv.org/abs/2610.22344)、[Frontis-MA1](https://arxiv.org/abs/2607.28568)、[AutoTrainess](https://arxiv.org/abs/2606.31551)；[AI 研究团队](https://arxiv.org/abs/2610.28887)由不同专长的智能体组成，在有限实验预算下自主探索线性注意力架构；[OProver](https://arxiv.org/abs/2605.17283) 做智能体式形式化证明。
- **认真对待失败。**[智能体在自动研究中如何失败](https://arxiv.org/abs/2608.14905)（100 个案例）、[智能体能做开放式 AI 研究吗](https://arxiv.org/abs/2607.27191)、[深度研究智能体在哪里出错](https://arxiv.org/abs/2606.02060)、[False Frontiers](https://arxiv.org/abs/2609.39102)（出题者作弊）、[先学会评估再改进](https://arxiv.org/abs/2608.31076)；评测 [AutoLab](https://arxiv.org/abs/2606.05080)、[NatureBench](https://arxiv.org/abs/2606.24530)、[ResearchClawBench](https://arxiv.org/abs/2606.07591)、[SAEScientist-Bench](https://arxiv.org/abs/2609.09113)、[Into the Unknown](https://arxiv.org/abs/2610.19159)。

### 7.4 实践要点与开放问题

要求每个结论附来源链接并抽查；把"广搜"与"深读"分开；给检索设预算；科研智能体的实验放在可复现的沙箱里，保留代码、数据版本与日志；问题选择、实验设计审阅与最终结论由人负责。

开放问题：如何评估开放式研究的质量与新颖性；自动研究产出的错误结论如何被发现；科研诚信与署名。

## 8 应用：行业、个人助理与多模态

### 8.1 行业工作流

行业落地的三要素是**领域工具、必须遵守的业务规则、人工审核**。[τ-bench](https://arxiv.org/abs/2406.12045) 与 [τ²-Bench](https://arxiv.org/abs/2506.07982) 把"遵守策略地与用户和工具交互"做成评测（零售、航空、电信）；[TheAgentCompany](https://arxiv.org/abs/2412.14161)、[EnterpriseOps-Gym](https://arxiv.org/abs/2603.13594)、[OccuBench](https://arxiv.org/abs/2604.10866) 模拟企业与职业工作；[FinToolBench](https://arxiv.org/abs/2603.08262)、[MobilityBench](https://arxiv.org/abs/2602.22638)、[LawThinker](https://arxiv.org/abs/2602.12056) 面向金融、出行与法律。

### 8.2 个人智能体

[OpenClaw](https://github.com/openclaw/openclaw)（前身 Clawdbot，2025-11 起）是开源、自托管的个人智能体：常驻本机，通过聊天软件接收指令，能读写本地文件、调用工具。它带出了一批"Claw 类"工作：[OpenClaw-RL](https://arxiv.org/abs/2603.10165)（"说话就能训练智能体"）、[MetaClaw](https://arxiv.org/abs/2603.17187)（在对话中元学习）、[ClawGym](https://arxiv.org/abs/2604.26904)、[ClawBench](https://arxiv.org/abs/2604.08523)、[Claw-Eval](https://arxiv.org/abs/2604.06132)。个人智能体的难点是个性化记忆、主动性和权限。

### 8.3 多模态、实时交互与游戏

视觉推理中的工具调用（[PyVision](https://arxiv.org/abs/2507.07998)、[视觉智能体强化微调](https://arxiv.org/abs/2505.14246)、[DeepEyesV2](https://arxiv.org/abs/2511.05271)）、长视频（[LongVT](https://arxiv.org/abs/2511.20785)、[LongVideoAgent](https://arxiv.org/abs/2512.20618)）、带长期记忆的多模态智能体（[M3-Agent](https://arxiv.org/abs/2508.09736)）、多模态智能体基础模型（[Magma](https://arxiv.org/abs/2502.13130)、[Kimi K2.5](https://arxiv.org/abs/2602.02276)）。游戏从 [Voyager](https://arxiv.org/abs/2305.16291) 到 [Game-TARS](https://arxiv.org/abs/2510.23691)、[Lumine](https://arxiv.org/abs/2511.08892)、[NitroGen](https://arxiv.org/abs/2601.02427)，提供丰富、安全、可无限重来的训练与评测环境。

### 8.4 最近半年

- **个人智能体生态成形。**[VisualClaw](https://arxiv.org/abs/2606.16295)（物理世界）、[nanoMuse](https://arxiv.org/abs/2610.08699)（覆盖所有设备）、[OneDayAgent](https://arxiv.org/abs/2608.05013)（开放日常请求）、[Macaron-A2UI](https://arxiv.org/abs/2605.24830)（生成式界面）、[Qwen-Planner-Agent](https://arxiv.org/abs/2609.29892)；评测 [π-Bench](https://arxiv.org/abs/2605.14678)、[VibeLifeBench](https://arxiv.org/abs/2608.10875)、[UniClawBench](https://arxiv.org/abs/2607.08768)、[EnterpriseClawBench](https://arxiv.org/abs/2606.23654)、[Claw-SWE-Bench](https://arxiv.org/abs/2606.12344)。
- **行业案例更具体。**[Spreadsheet-RL](https://arxiv.org/abs/2605.22642)、[BI-Agent](https://arxiv.org/abs/2609.20886)、[CADFather](https://arxiv.org/abs/2610.09127)、[SemaPLC](https://arxiv.org/abs/2608.18565)、[X-Ragent](https://arxiv.org/abs/2610.16811)、[纵向医疗推理](https://arxiv.org/abs/2609.39566)、[Cura 1T](https://arxiv.org/abs/2607.15314)、[Data Journalist Agent](https://arxiv.org/abs/2606.11176)；[生产中的交易智能体](https://arxiv.org/abs/2609.05663)报告了两个车队六个月的真实部署；评测 [CHI-Bench](https://arxiv.org/abs/2605.16679)（医疗）、[OR-License](https://arxiv.org/abs/2610.24087)、[RuleTrek](https://arxiv.org/abs/2610.19698)。
- **"数字同事"与模拟组织。**[OfficeTown](https://arxiv.org/abs/2610.23194)、[CoworkEnv](https://arxiv.org/abs/2610.24960)、[WorkFactory](https://arxiv.org/abs/2610.37161)、[AgentMercury](https://arxiv.org/abs/2608.20634)；综述 [Work Agents](https://arxiv.org/abs/2610.35845)、[从聊天机器人到数字同事](https://arxiv.org/abs/2606.14502)、[智能体式产物创建](https://arxiv.org/abs/2608.28122)。
- **实时、全双工的多模态交互。**[Gander](https://arxiv.org/abs/2609.08977)、[Heartbeat Thinking](https://arxiv.org/abs/2610.26233)、[StreamVibe](https://arxiv.org/abs/2610.34662)、[Omni-Decision](https://arxiv.org/abs/2607.11433)、[EVA-Bench](https://arxiv.org/abs/2605.13841)。
- **游戏。**[CAST](https://arxiv.org/abs/2607.25308)（游戏求解器作为教师）、[EvolvingWorld](https://arxiv.org/abs/2607.17250)、[长时运行游戏智能体的启发式学习](https://arxiv.org/abs/2610.12341)、[RSIGame](https://arxiv.org/abs/2609.39045)。
- **替用户做选择时的偏差。**[来源偏好](https://arxiv.org/abs/2610.03195)发现智能体替用户挑选商品、酒店时存在系统性的来源偏好。

### 8.5 实践要点与开放问题

从一个窄、高频、结果可度量的流程切入；业务规则写成工具里的校验与硬约束；先做"智能体起草、人审批、系统执行"，用人工修改记录评测与改进；个人智能体给最小权限、独立账户，敏感操作逐次确认。

开放问题：主动性的边界；行业部署的责任归属、审计与合规；个人数据在长期记忆中的保护。

## 9 训练智能体：环境、强化学习与自我改进

### 9.1 能力从哪里来

**智能体模型。**在预训练与中期训练中加入智能体数据，再用轨迹做 SFT、在环境里做 RL。代表有 [Kimi K2](https://arxiv.org/abs/2507.20534)、[GLM-4.5](https://arxiv.org/abs/2508.06471)、[GLM-5](https://arxiv.org/abs/2602.15763)、[Kimi K2.5](https://arxiv.org/abs/2602.02276)、[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)、[Nex-N1](https://arxiv.org/abs/2512.04987)、[智能体持续预训练](https://arxiv.org/abs/2509.13310)；小而高效的有 [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848)、[Nemotron 3 Super](https://arxiv.org/abs/2604.12374)、[Youtu-LLM](https://arxiv.org/abs/2512.24618)。

**智能体 RL。**与单轮 RLVR 不同：工具输出不是模型生成的（通常不计入损失），奖励在几十轮之后才出现，rollout 昂贵且长短不一。训练系统需要异步与长轨迹支持（[超越十轮](https://arxiv.org/abs/2508.07976)、[verl](https://github.com/verl-project/verl)、[Agent Lightning](https://arxiv.org/abs/2508.03680)、[VerlTool](https://arxiv.org/abs/2509.01055)、[AWorld](https://arxiv.org/abs/2508.20404)）；算法要处理熵与信用分配（[ARPO](https://arxiv.org/abs/2507.19849)、[熵调制策略梯度](https://arxiv.org/abs/2509.09265)、[AEPO](https://arxiv.org/abs/2510.14545)、[IGPO](https://arxiv.org/abs/2510.14967)、[树搜索 RL](https://arxiv.org/abs/2509.21240)）；推理会坍缩（[RAGEN-2](https://arxiv.org/abs/2604.06268)）。[揭秘智能体推理中的 RL](https://arxiv.org/abs/2510.11701)给出实用配方，[智能体 RL 全景](https://arxiv.org/abs/2509.02547)做了综述。

**环境。**可执行、可验证的环境是新的"数据"：[AgentGym-RL](https://arxiv.org/abs/2509.08755)、[环境扩展](https://arxiv.org/abs/2509.13311)、[EnvScaler](https://arxiv.org/abs/2601.05808)、[Agent World Model](https://arxiv.org/abs/2602.10090)（用 LLM 模拟环境）、[Agent-World](https://arxiv.org/abs/2604.18292)；[早期经验](https://arxiv.org/abs/2510.08558)让智能体在没有奖励时从自身动作的后果中学习。

**不改权重的自我改进。**积累记忆、技能与更好的 harness（第 2、3 章）；改权重的自我演化有 [R-Zero](https://arxiv.org/abs/2508.05004)、[Agent0](https://arxiv.org/abs/2511.16043)、[Hyperagents](https://arxiv.org/abs/2603.19461)（改进自身的改进机制）。自我改进可以分三层：上下文与记忆（最便宜、可回滚）→ 技能与 harness（可读、可审计）→ 权重（最持久、最难回滚）。综述见[自演化智能体](https://arxiv.org/abs/2507.21046)与[现代智能体系统的自我改进](https://arxiv.org/abs/2607.13104)。

### 9.2 最近半年

- **在策略蒸馏（OPD）的变体潮与现实检验。**OPD 让学生自己采样、教师逐 token 打分，比 SFT 更贴合学生分布，比 RL 信号更密。变体有 [TurnOPD](https://arxiv.org/abs/2607.05804)、[SEED](https://arxiv.org/abs/2607.14777)、[PR-OPD](https://arxiv.org/abs/2609.36642)、[RetireOPD](https://arxiv.org/abs/2609.20784)、[PivotOPD](https://arxiv.org/abs/2609.40285)、[AgentOPSD](https://arxiv.org/abs/2608.05987)、[自蒸馏智能体 RL](https://arxiv.org/abs/2605.15155)。但 [OPD 真的在蒸馏吗](https://arxiv.org/abs/2608.31046)发现教师监督噪声很大、学生却不敏感，用一个固定的负优势就能达到相近效果；[自蒸馏的现实检验](https://arxiv.org/abs/2610.19789)在统一条件下重测十种方法（132 次训练、约 1.2 万 GPU 小时）。
- **长程 RL 的稳定与信用分配。**进度信号（[ProAS](https://arxiv.org/abs/2610.18683)、[DeltaTTG](https://arxiv.org/abs/2610.17666)、[Tev](https://arxiv.org/abs/2610.18868)）、多尺度优化（[多尺度策略优化](https://arxiv.org/abs/2610.35013)）、动态评分细则（[DRACO](https://arxiv.org/abs/2609.04094)）、熵调节（[AEM](https://arxiv.org/abs/2605.00425)）、树感知后验蒸馏（[树感知后验蒸馏](https://arxiv.org/abs/2610.23181)）；[多步工具 RL 为何坍缩](https://arxiv.org/abs/2606.26027)分析失败原因；[长程 RL 需要 Muon](https://arxiv.org/abs/2610.20256) 在仓库级 issue 修复上给出单 rollout 异步配方。
- **训练系统规模化。**[NeMo-DCR](https://arxiv.org/abs/2610.08430)（万亿参数级、按位精确）、[QwenGyre](https://arxiv.org/abs/2609.33848)（弹性 RL）、[Molt](https://arxiv.org/abs/2607.21653)、[单 rollout 异步优化](https://arxiv.org/abs/2607.07508)、[TRACE](https://arxiv.org/abs/2606.11119)（rollout 预算分配）。
- **训练"整个智能体"。**[Agent Lightning v1.0](https://arxiv.org/abs/2608.17528)、[LEGO-RL](https://arxiv.org/abs/2608.17393)、[ClawGym II](https://arxiv.org/abs/2608.16798)、[训练整个智能体而不只是模型](https://arxiv.org/abs/2610.28069)、[Harness Scaling](https://arxiv.org/abs/2610.29458)、[OPID](https://arxiv.org/abs/2606.26790)、[NeoHorse-1](https://arxiv.org/abs/2609.08183)。
- **环境合成成为独立方向。**[Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)、[从受训者到训练者](https://arxiv.org/abs/2606.17682)、[CompoWorld](https://arxiv.org/abs/2609.33665)、[Skill2Env](https://arxiv.org/abs/2609.33772)、[从已解机制生成环境](https://arxiv.org/abs/2609.27321)、[AgentGarten](https://arxiv.org/abs/2610.12374)、[WorkRover](https://arxiv.org/abs/2610.18067)、[EAU](https://arxiv.org/abs/2610.16523)、[智能体语言世界模型](https://arxiv.org/abs/2610.06100)、[MIMESIS](https://arxiv.org/abs/2610.09484)（用户模拟器）；数据质量：[什么是好的智能体数据](https://arxiv.org/abs/2608.27260)、[Trajectory2Gym](https://arxiv.org/abs/2610.33568)、[不要掩盖环境](https://arxiv.org/abs/2609.20715)、[Agent Error Dataset](https://arxiv.org/abs/2609.40111)；开放配方 [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855)；综述 [智能体环境工程](https://arxiv.org/abs/2606.12191)。
- **递归自我改进有了评测。**[RSI Arena](https://arxiv.org/abs/2610.33624)、[EvalResearchBench](https://arxiv.org/abs/2610.04184)、[S3Gym](https://arxiv.org/abs/2608.31100)、[Aspire](https://arxiv.org/abs/2608.31111)；[最后一个人类造的 AI](https://arxiv.org/abs/2609.11873)、[Atria Dawn](https://arxiv.org/abs/2609.15818)、[可信的递归自我改进](https://arxiv.org/abs/2610.35193)讨论路线与安全边界。
- **为智能体而生的模型与效率。**[Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007)、[Agents-A1](https://arxiv.org/abs/2606.30616)、[Macaron-V1](https://arxiv.org/abs/2608.09819)、[MiMo-V2.6](https://arxiv.org/abs/2610.11959)、[IQuest-Q1](https://arxiv.org/abs/2610.30949)、[Sage-A1.5](https://arxiv.org/abs/2610.28578)、[Muse Glimmer 30B](https://arxiv.org/abs/2610.24628)、[Intern-S2-Preview](https://arxiv.org/abs/2608.13505)；效率：[推理力度的经济学](https://arxiv.org/abs/2610.34491)、[轨迹级推测执行](https://arxiv.org/abs/2610.21463)、[DLite](https://arxiv.org/abs/2610.23787)、[EdgeWise](https://arxiv.org/abs/2610.17560)、[TAO](https://arxiv.org/abs/2610.37230)、[Mix-Quant](https://arxiv.org/abs/2605.20315)。

### 9.3 实践要点与开放问题

多数团队不需要训练：先把提示、工具、技能和 harness 做好。真要训练时，先收集真实轨迹与失败案例，构造可自动判分的环境，SFT 冷启动、RL 精调；人工抽查轨迹找奖励作弊；小模型 + 好 harness + 合适的推理力度常比大模型更划算。

开放问题：合成环境与真实世界的差距；长程信用分配；自我改进如何保证单调变好、可审计、不越界；模型与 harness 如何分工。

## 10 评测：怎样知道它真的有用

### 10.1 三层评测

智能体评测比模型评测难：结果同时取决于模型、harness、环境、随机性和预算。

**能力基准。**编码：[SWE-bench](https://arxiv.org/abs/2310.06770)、[SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)、[Terminal-Bench](https://www.tbench.ai/)；GUI：[OSWorld](https://arxiv.org/abs/2404.07972)、[WebArena](https://arxiv.org/abs/2307.13854)；研究：[GAIA](https://arxiv.org/abs/2311.12983)、[BrowseComp](https://arxiv.org/abs/2504.12516)；工具与对话：[τ-bench](https://arxiv.org/abs/2406.12045)、[τ²-Bench](https://arxiv.org/abs/2506.07982)、[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)、[Tool Decathlon](https://arxiv.org/abs/2510.25726)；MCP：[MCP-Universe](https://arxiv.org/abs/2508.14704)、[MCP-Bench](https://arxiv.org/abs/2508.20453)、[LiveMCP-101](https://arxiv.org/abs/2508.15760)；ML 研究：[MLE-bench](https://arxiv.org/abs/2410.07095)、[RE-Bench](https://arxiv.org/abs/2411.15114)、[PaperBench](https://arxiv.org/abs/2504.01848)；早期综合：[AgentBench](https://arxiv.org/abs/2308.03688)、[ALFWorld](https://arxiv.org/abs/2010.03768)、[WebShop](https://arxiv.org/abs/2207.01206)。

**时间视野与经济价值。**[METR](https://arxiv.org/abs/2503.14499) 用"人类专家完成同样任务需要多久"度量智能体，50% 成功率对应的任务时长长期按约 7 个月翻倍；[METR 页面](https://metr.org/time-horizons/)在 2026-09 最后一次更新后停止维护，并注明超过 16 小时的测量已不可靠。经济价值方向有 [TheAgentCompany](https://arxiv.org/abs/2412.14161)、[SWE-Lancer](https://arxiv.org/abs/2502.12115)、[FutureX](https://arxiv.org/abs/2508.11987)（预测未来，天然防污染）。

**你自己的应用评测**最重要，见第 12 章。

### 10.2 方法论

报告成本与方差，并与简单基线比较（[AI Agents That Matter](https://arxiv.org/abs/2407.01502)）；用 pass^k（k 次都成功）而不只是 pass@k 衡量可靠性（[τ-bench](https://arxiv.org/abs/2406.12045)）；跨模型、跨 harness 统一报告（[HAL](https://arxiv.org/abs/2510.11977)）；能用程序判分就用程序，否则用带评分细则的 LLM 评审并与人工对齐，开放任务可以让智能体当评审（[Mind2Web 2](https://arxiv.org/abs/2506.21506)）；读轨迹而不只看分数。综述见 [LLM 智能体评测](https://arxiv.org/abs/2503.16416)。

### 10.3 最近半年

- **更难、更陌生、更长。**[Agents' Last Exam](https://arxiv.org/abs/2606.05405)；[Running the Gauntlet](https://arxiv.org/abs/2606.14397)（常见应用上的基准已饱和，改在视频剪辑、3D 建模、电路设计等少见的专业应用里测时间感知、图形理解与 3D 推理）；[OSWorld 2.0](https://arxiv.org/abs/2606.29537)；[Δτ-Bench](https://arxiv.org/abs/2610.34238)；[WildClawBench](https://arxiv.org/abs/2605.10912)；[AdaExploreBench](https://arxiv.org/abs/2610.25950)；[Learn2Play Bench](https://arxiv.org/abs/2610.08215)。
- **以经济结果为标尺。**[BizBench](https://arxiv.org/abs/2610.20424)、[Business Arena](https://arxiv.org/abs/2608.08621)、[StartupBench](https://arxiv.org/abs/2608.17800)、[E-Commerce Bench](https://arxiv.org/abs/2608.30730)、[MerchantBench](https://arxiv.org/abs/2607.28956)、[GDPevo](https://arxiv.org/abs/2608.03764)、[LiveMACE](https://arxiv.org/abs/2610.09872)。
- **反思评测本身。**[预测效度](https://arxiv.org/abs/2606.19704)、[SWE-bench 是学会了还是背下了](https://arxiv.org/abs/2609.27891)、[RAMP](https://arxiv.org/abs/2605.27492)、[TASTE](https://arxiv.org/abs/2605.28556)、[PACE](https://arxiv.org/abs/2607.02032)、[EarlyEval](https://arxiv.org/abs/2609.02783)、[评测预算分配](https://arxiv.org/abs/2610.05034)、[智能体也会衰老](https://arxiv.org/abs/2605.26302)。
- **评测 harness 与自我演化。**[Trial by Trace](https://arxiv.org/abs/2610.32285)、[Evo-Bench](https://arxiv.org/abs/2608.09096)、[EvoHarnessBench](https://arxiv.org/abs/2609.04280)、[HarnessOpt-Bench](https://arxiv.org/abs/2608.06301)、[Doc2Harness](https://arxiv.org/abs/2610.20344)、[Continual Learning Exam](https://arxiv.org/abs/2610.33956)、[AgentStream](https://arxiv.org/abs/2608.00155)。
- **过程、验证与归因。**[OSWorld-Pro](https://arxiv.org/abs/2609.24890)、[证据优先的验证](https://arxiv.org/abs/2610.17724)、[VerifierBench](https://arxiv.org/abs/2610.28679)、[LongRCA Bench](https://arxiv.org/abs/2608.15242)、[从成功流中追溯失败](https://arxiv.org/abs/2607.12747)、[AgentCompass](https://arxiv.org/abs/2607.13705)。

### 10.4 实践要点与开放问题

从日志或真实需求中挑 20–50 个任务、写清楚可检查的完成标准；每个任务跑多次，报告成功率、pass^k、成本与延迟；每周读轨迹补充失败模式；保留不公开的测试集；任何改动都跑回归。

开放问题：长于一天的任务怎么评；开放式产出的质量如何可靠打分；基准成绩与真实部署价值的关系有多强。

## 11 安全与可靠性

### 11.1 三类风险

**外部攻击。**间接提示注入：网页、文档、邮件、工具输出里夹带指令（[Greshake 等](https://arxiv.org/abs/2302.12173)，2023-02）；工具与技能投毒、供应链攻击。**模型自身的行为。**越界、欺骗、奖励作弊、压力下的失准（[Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment)）。**系统可靠性。**不可逆操作、误删、长期部署中的退化。

### 11.2 防御原则

- **致命三要素**（[Simon Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)）：同时具备私有数据访问、不可信内容输入、对外通信能力的智能体，可以被注入后外泄数据。三者至少去掉一个。
- **从设计上隔离**：[CaMeL](https://arxiv.org/abs/2503.18813) 让受信任的模型只根据用户指令生成程序，不可信数据只作为值流过程序、不能改变控制流；"先计划、后执行"、双模型模式同理。
- **权限与沙箱**：默认只读、写操作限定目录、网络出口白名单、凭证不进入模型上下文、不可逆操作逐次确认、全量审计日志。
- **评测与红队**：[InjecAgent](https://arxiv.org/abs/2403.02691)、[AgentDojo](https://arxiv.org/abs/2406.13352)、[WASP](https://arxiv.org/abs/2504.18575)、[AgentHarm](https://arxiv.org/abs/2410.09024)、[ToolEmu](https://arxiv.org/abs/2309.15817)、[Agent Security Bench](https://arxiv.org/abs/2410.02644)、[Agent-SafetyBench](https://arxiv.org/abs/2412.14470)、[OpenAgentSafety](https://arxiv.org/abs/2507.06134)、[R-Judge](https://arxiv.org/abs/2401.10019)、[遵循策略智能体的红队](https://arxiv.org/abs/2506.09600)、[X-Teaming](https://arxiv.org/abs/2504.13203)、[T-MAP](https://arxiv.org/abs/2603.22341)。护栏：[AgentDoG](https://arxiv.org/abs/2601.18491)、[Spider-Sense](https://arxiv.org/abs/2602.05386)、[OS-Sentinel](https://arxiv.org/abs/2510.24411)。开放环境的观察：[Agents of Chaos](https://arxiv.org/abs/2602.20021)、[Moltbook 背后的恶魔](https://arxiv.org/abs/2602.09877)、[涌现的社会智能风险](https://arxiv.org/abs/2603.27771)。

### 11.3 最近半年

- **安全下沉到系统层。**[AgentKernel](https://arxiv.org/abs/2609.29647)（信任原生的智能体操作系统）、[Agentic Transaction](https://arxiv.org/abs/2608.13900)（ACID）、[UndoBench](https://arxiv.org/abs/2610.05622)（把完成任务与能否撤销分开评测）、[审计智能体 harness 安全](https://arxiv.org/abs/2605.14271)、[SkillHarness](https://arxiv.org/abs/2606.20636)、[EvoSafeHarness](https://arxiv.org/abs/2609.05903)、[SeerGuard](https://arxiv.org/abs/2607.15550)。
- **自演化带来的新攻击面。**[SkillJack](https://arxiv.org/abs/2608.03509)（持久的技能后门）、[SkillMarker](https://arxiv.org/abs/2610.15725)（技能水印）、[False Frontiers](https://arxiv.org/abs/2609.39102)。
- **度量模型自身的不当行为。**[DecepEval](https://arxiv.org/abs/2610.07967)、[SWE-MISBEHAVIOR](https://arxiv.org/abs/2610.18206)、[认知引发的风险](https://arxiv.org/abs/2608.15304)；正向能力 [Agentic Abstention](https://arxiv.org/abs/2606.28733)、[经验式置信估计](https://arxiv.org/abs/2609.17708)、[Thinking-in-Bets](https://arxiv.org/abs/2610.34654)。
- **多智能体与隐私。**[涌现共谋](https://arxiv.org/abs/2609.24967)、[Covert Assistance](https://arxiv.org/abs/2609.39050)、[Agent Mosaic](https://arxiv.org/abs/2610.30418)、[GateMem](https://arxiv.org/abs/2606.18829)、[MemPrivacy](https://arxiv.org/abs/2605.09530)、[MemSyco-Bench](https://arxiv.org/abs/2607.01071)。
- **红队与审计工具化。**[DTap](https://arxiv.org/abs/2605.04808)、[OpenART](https://arxiv.org/abs/2608.00677)、[AdvSim2Real](https://arxiv.org/abs/2610.08773)、[HazardAuditor](https://arxiv.org/abs/2609.15134)、[APEX-Harm](https://arxiv.org/abs/2610.13623)、[TraceDance](https://arxiv.org/abs/2609.33295)、[智能体到底做了什么](https://arxiv.org/abs/2610.06406)、[AgentDoG 1.5](https://arxiv.org/abs/2605.29801)。

### 11.4 实践要点与开放问题

对照致命三要素检查每个智能体；沙箱运行、默认只读、网络与文件系统按需开放；删除、付款、发送、部署需要人确认并保留撤销手段；只安装可信来源的 MCP 服务器与技能；记录完整轨迹并抽样复查。

开放问题：提示注入没有通用的模型层解法，只能靠架构隔离降低影响；长期运行、自我演化的智能体如何持续保持对齐与可审计。

## 12 实践入门指南

本章把前面的经验压缩成十步。三条核心原则：**先用最简单的方案**；**先建评测再迭代**；**权限从小开始**。代码用 Anthropic Python SDK（`pip install anthropic`），换成其他厂商的 SDK，结构完全相同。

### 12.1 判断是否真的需要智能体

问四个问题：步骤能否事先写死（能 → 工作流）？是否需要根据中间结果决定下一步（是 → 智能体）？出错的代价有多大、能否检测？单次任务值不值得几十次模型调用的成本？多数业务问题的答案是"单次调用或工作流 + 一两处模型判断"。

### 12.2 选择搭建层级

| 层级 | 适合 | 例子 |
|---|---|---|
| 直接用产品 | 个人或团队提效，不嵌入自己的系统 | [Claude Code](https://code.claude.com/docs/en/overview)、[Codex](https://openai.com/index/introducing-codex/)、[ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/)、[Antigravity](https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/) |
| 托管智能体 | 需要沙箱、长任务、会话管理，不想自己运维 | [Claude Managed Agents](https://claude.com/blog/claude-managed-agents)、[Gemini 托管智能体](https://blog.google/innovation-and-ai/technology/developers-tools/managed-agents-gemini-api/) |
| 带 harness 的 SDK | 要现成的文件、命令、搜索工具与上下文压缩 | [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)、[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)、[ADK](https://github.com/google/adk-python) |
| 编排框架 | 有明确状态机、需要持久化与人工介入 | [LangGraph](https://github.com/langchain-ai/langgraph)、[Microsoft Agent Framework](https://github.com/microsoft/agent-framework)、[Pydantic AI](https://github.com/pydantic/pydantic-ai) |
| 自写循环 | 学习原理、完全掌控、特殊运行环境 | 下面的代码；[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 是极简参考 |

### 12.3 写一个最小的智能体循环

一个只读的"代码库问答"智能体：两个工具，只能看 `workspace/` 目录，最多 20 步。

```python
import pathlib
import anthropic

ROOT = pathlib.Path("workspace").resolve()   # 智能体只能看到这个目录
client = anthropic.Anthropic()               # 从环境变量 ANTHROPIC_API_KEY 读取密钥

TOOLS = [
    {"name": "list_dir",
     "description": "列出 workspace 中某个目录的内容。path 是相对路径，默认 '.'。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "read_file",
     "description": "读取 workspace 中的文本文件，最多返回 20000 个字符。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                      "required": ["path"]}},
]

def safe_path(rel):
    p = (ROOT / rel).resolve()
    if not p.is_relative_to(ROOT):               # 防止 ../ 逃出工作目录
        raise ValueError(f"{rel} 不在 workspace 内")
    return p

def run_tool(name, args):
    if name == "list_dir":
        d = safe_path(args.get("path", "."))
        return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir()))
    if name == "read_file":
        return safe_path(args["path"]).read_text(errors="replace")[:20000]
    raise ValueError(f"未知工具 {name}")

def run_agent(task, max_steps=20):
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        resp = client.messages.create(
            model="claude-opus-5-5", max_tokens=16000,
            system="你是代码库助手。先用工具查看文件再回答，并注明依据的文件。",
            tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "pause_turn":     # 服务端暂停了长回合：原样续上
            continue
        if resp.stop_reason != "tool_use":       # end_turn 正常结束；max_tokens、refusal 需另行处理
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text, resp.stop_reason
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            try:
                out, err = run_tool(b.name, b.input), False
            except Exception as e:                # 错误交还给模型，让它自己纠正
                out, err = f"Error: {e}", True
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})   # 所有结果放在同一条消息里
    return "达到步数上限", "max_steps"

answer, why = run_agent("这个项目是做什么的？入口在哪里？")
print(why, answer, sep="\n")
```

要点：助手消息原样追加（包括思考块与工具调用块）；一轮里的所有工具结果放进同一条用户消息；工具出错时返回 `is_error` 而不是中断；用步数上限兜底。不想手写循环时，SDK 的 Tool Runner 可以自动执行这个循环：用 `@beta_tool` 装饰普通函数（函数签名与文档字符串会变成工具 schema），再调用 `client.beta.messages.tool_runner(model=..., max_tokens=..., tools=[...], messages=[...])` 并迭代结果。生产环境还要处理 `refusal` 与 `max_tokens` 停止原因、加重试与日志，长输出改用流式接口。

练习：加一个 `write_file` 工具，在执行前打印将要写入的内容并等待人工输入 y/n；再加一个 `run_tests` 工具，在容器里运行测试并返回最后 100 行输出。

### 12.4 把工具设计好

- 少而精：按"用户要完成的事"设计工具（如 `search_orders`），而不是把每个底层 API 都暴露出去。
- 描述写清楚何时用、何时不用、参数格式和返回内容；给出一两个例子。
- 返回对模型有用的信息：人能读懂的字段而不是内部 ID；结果太长就分页或截断，并说明还有多少。
- 错误信息要可操作：说清哪里错了、该怎么改。
- 可复用的工具做成 MCP 服务器；写操作的工具单独标注并加确认。参考[为智能体编写工具](https://www.anthropic.com/engineering/writing-tools-for-agents)。

### 12.5 管理上下文与状态

系统提示与工具定义放在稳定的前缀以命中缓存；长任务让智能体维护 `progress.md` 与待办清单；工具输出只保留需要的部分；探索性工作交给子智能体，只取回结论；接近上限时压缩旧轮次。详见第 2 章与 [Manus 的上下文工程](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)。

### 12.6 把经验沉淀到 harness、AGENTS.md 与技能

智能体犯过一次的错，写进 AGENTS.md（项目约定）或做成技能（可复用流程 + 脚本），而不是每次在对话里纠正。技能只在上下文中放名称与一句描述，用到时再加载全文（[Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)）。长时任务参考[长时运行 harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)：先初始化环境与进度文件，再每次推进一小步并提交。

### 12.7 先建评测，再迭代

准备 20–50 个真实任务与可检查的完成标准；每个任务跑 3–5 次，记录成功率、成本、延迟；程序判分优先，LLM 评审需配评分细则并与人工对齐。每次改提示、工具、模型或 harness 都跑回归。追踪每一步（[Langfuse](https://github.com/langfuse/langfuse)、[Phoenix](https://github.com/Arize-ai/phoenix)、[Inspect](https://github.com/UKGovernmentBEIS/inspect_ai)），每周读轨迹找新的失败模式。

### 12.8 加上安全护栏

- 对照[致命三要素](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)：私有数据、不可信内容、对外通信，至少去掉一项。
- 沙箱执行（容器、虚拟机或托管沙箱）；默认只读；网络出口白名单；凭证不进上下文。
- 不可逆操作（删除、付款、发送、部署）前请人确认；保留撤销手段与审计日志。
- 只用可信来源的 MCP 服务器与技能；把网页、邮件、文档内容一律当作数据，不当作指令。

### 12.9 控制成本与延迟

提示缓存是最大的一笔节省；按步骤难度选模型与推理力度；无依赖的工具调用并行；限制步数与单任务预算；批量离线任务用批处理接口。把"每个成功任务的成本"作为核心指标，而不是每次调用的成本。

### 12.10 上线与运维

灰度发布：先只读，再"起草 + 人审"，最后才自动执行；监控成功率、人工接管率、成本与异常动作；固定模型版本，升级前跑评测；收集用户修改与差评作为新的评测数据；长期运行的智能体定期复查记忆与技能，防止"衰老"（[智能体也会衰老](https://arxiv.org/abs/2605.26302)）。

### 12.11 四周学习路径与常见错误

- **第 1 周 · 用**：每天用一个编码智能体完成真实任务；读 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 与 [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)。
- **第 2 周 · 写**：手写上面的循环，加写文件工具与确认机制；把一个工具做成 MCP 服务器；读 [ReAct](https://arxiv.org/abs/2210.03629)、[SWE-agent](https://arxiv.org/abs/2405.15793)、[工具设计](https://www.anthropic.com/engineering/writing-tools-for-agents)。
- **第 3 周 · 评**：为自己的场景做 20 个任务的评测集，比较两种模型或两种提示的成功率与成本；读 [AI Agents That Matter](https://arxiv.org/abs/2407.01502)、[τ-bench](https://arxiv.org/abs/2406.12045)。
- **第 4 周 · 稳**：加上下文管理、AGENTS.md 与一个技能；做一次提示注入自测；读 [上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)、[致命三要素](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)、[CaMeL](https://arxiv.org/abs/2503.18813)。

常见错误：一上来就做多智能体；工具太多、描述太短；没有评测、凭感觉改提示；把整份文档塞进上下文；给了写权限却没有沙箱、确认与撤销；只看成功率不看成本，只看结果不看轨迹；业务规则只写在提示里。

推荐读物（按顺序）：[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) → [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) → [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) → [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) → [Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) → [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) → [Manus 上下文工程](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) → [多智能体研究系统](https://www.anthropic.com/engineering/multi-agent-research-system) 与 [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) → [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) → [长时运行 harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) → [Claude Code 最佳实践](https://www.anthropic.com/engineering/claude-code-best-practices)。综述：[LLM 自主智能体综述](https://arxiv.org/abs/2308.11432)、[LLM 智能体的兴起](https://arxiv.org/abs/2309.07864)、[基础智能体](https://arxiv.org/abs/2504.01990)、[智能体 AI 搭车客指南](https://arxiv.org/abs/2606.24937)、[智能体 AI 的适配](https://arxiv.org/abs/2512.16301)、[对智能体模型的批判](https://arxiv.org/abs/2606.23991)。

## 13 开放问题

1. **长程**：如何检测原地打转、保持全局目标、在正确的时机请求人？
2. **上下文**：压缩丢掉什么？记忆如何过期、纠错、在多人之间治理？
3. **Harness**：自我改进如何避免过拟合与能力倒退？harness 与权重如何分工？
4. **协议**：第三方工具与技能的信任、身份、授权与计费。
5. **应用**：GUI 操作是否只是过渡形态？开放式研究的质量如何判断？
6. **训练**：合成环境与真实世界的差距；长程信用分配；奖励作弊。
7. **评测**：长于一天的任务怎么评？基准分数能否预测部署价值？
8. **安全**：提示注入的系统性解法；自演化智能体的持续对齐与审计。
9. **社会**：智能体替人做选择时的偏差、责任归属与劳动分工的变化。

## 附录 A：最近半年逐月纪事（2026-05 → 2026-10）

**2026-05。**Google I/O：[Antigravity 2.0](https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-developer-highlights/)、Antigravity CLI 与 SDK、[Gemini API 托管智能体](https://blog.google/innovation-and-ai/technology/developers-tools/managed-agents-gemini-api/)，并宣布 Gemini CLI [迁移到 Antigravity CLI](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/)。技能研究集中爆发：[SkillOpt](https://arxiv.org/abs/2605.23904)、[SkillGrad](https://arxiv.org/abs/2605.27760)、[Skill0.5](https://arxiv.org/abs/2605.28424)、[模型生成技能的系统研究](https://arxiv.org/abs/2605.23899)、[SkillsVote](https://arxiv.org/abs/2605.18401)。harness：[Continual Harness](https://arxiv.org/abs/2605.09998)、[Code as Agent Harness](https://arxiv.org/abs/2605.18747)、[更新 harness 不等于 harness 有益](https://arxiv.org/abs/2605.30621)。环境：[CUA-Gym](https://arxiv.org/abs/2605.25624)、[OpenComputer](https://arxiv.org/abs/2605.19769)、[EnvFactory](https://arxiv.org/abs/2605.18703)。研究：[QUEST](https://arxiv.org/abs/2605.24218)、[GrepSeek](https://arxiv.org/abs/2605.29307)、[ARIS](https://arxiv.org/abs/2605.03042)、[ScientistOne](https://arxiv.org/abs/2605.26340)。评测与安全：[智能体也会衰老](https://arxiv.org/abs/2605.26302)、[RAMP](https://arxiv.org/abs/2605.27492)、[DTap](https://arxiv.org/abs/2605.04808)。

**2026-06。**评测：[Agents' Last Exam](https://arxiv.org/abs/2606.05405)、[OSWorld 2.0](https://arxiv.org/abs/2606.29537)、[Running the Gauntlet](https://arxiv.org/abs/2606.14397)、[预测效度](https://arxiv.org/abs/2606.19704)。上下文：[自压缩智能体](https://arxiv.org/abs/2606.23525)、[图记忆](https://arxiv.org/abs/2606.06036)。harness：[回顾式 harness 优化](https://arxiv.org/abs/2606.05922)、[HarnessX](https://arxiv.org/abs/2606.14249)。编码：[CLI-Universe](https://arxiv.org/abs/2606.22883)、[验证视野](https://arxiv.org/abs/2606.26300)、[DeNovoSWE](https://arxiv.org/abs/2606.10728)。训练：[Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)、[OpenThoughts-Agent](https://arxiv.org/abs/2606.24855)、[Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007)；综述 [智能体环境工程](https://arxiv.org/abs/2606.12191)。Gemini CLI 个人用户自 6 月 18 日起转向 Antigravity CLI。

**2026-07。**[MCP 2026-07-28 版规范](https://modelcontextprotocol.io/specification/2026-07-28/changelog)发布，改为无状态。harness：[递归 harness 自我改进](https://arxiv.org/abs/2607.15524)、[Harness Handbook](https://arxiv.org/abs/2607.13285)。训练：[TurnOPD](https://arxiv.org/abs/2607.05804)、[SEED](https://arxiv.org/abs/2607.14777)、[Molt](https://arxiv.org/abs/2607.21653)；综述[现代智能体系统的自我改进](https://arxiv.org/abs/2607.13104)。GUI：[Qwen-UI-Agent](https://arxiv.org/abs/2607.28227)、[StateAct](https://arxiv.org/abs/2607.22798)、[SeerGuard](https://arxiv.org/abs/2607.15550)。记忆：[MemHarness](https://arxiv.org/abs/2607.28272)、[AutoMem](https://arxiv.org/abs/2607.01224)。

**2026-08。**[A2A 1.0 加入 AAIF](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/)。harness 的分量被量化：[StateM](https://arxiv.org/abs/2608.15089)；[DarwinX](https://arxiv.org/abs/2608.07545)、[HarnessOpt-Bench](https://arxiv.org/abs/2608.06301)。训练"整个智能体"：[Agent Lightning v1.0](https://arxiv.org/abs/2608.17528)、[LEGO-RL](https://arxiv.org/abs/2608.17393)、[ClawGym II](https://arxiv.org/abs/2608.16798)；[OPD 真的在蒸馏吗](https://arxiv.org/abs/2608.31046)。安全：[SkillJack](https://arxiv.org/abs/2608.03509)、[Agentic Transaction](https://arxiv.org/abs/2608.13900)、[AgentKernel](https://arxiv.org/abs/2609.29647)。经济价值评测：[Business Arena](https://arxiv.org/abs/2608.08621)、[StartupBench](https://arxiv.org/abs/2608.17800)。编码：[交接税](https://arxiv.org/abs/2608.24358)、[Mendel Gödel Machine](https://arxiv.org/abs/2608.07645)。

**2026-09。**[OpenAI DevDay 2026](https://openai.com/index/devday-2026-recap/)；[METR 时间视野页面](https://metr.org/time-horizons/)最后一次更新并停止维护。harness 递归自我改进集中出现：[HarnessDev](https://arxiv.org/abs/2609.01437)、[ModularRSI](https://arxiv.org/abs/2609.14857)、[RRSI](https://arxiv.org/abs/2609.24972)、[Harness-Zero](https://arxiv.org/abs/2609.24974)、[Raven](https://arxiv.org/abs/2609.33439)；[编码 harness 设计的实证研究](https://arxiv.org/abs/2609.20804)。多智能体：[Agensh](https://arxiv.org/abs/2609.26781)。主动性：[主动式智能体基础](https://arxiv.org/abs/2609.37267)。长程：[Harness-of-Harness](https://arxiv.org/abs/2609.01481)、[Marathoner](https://arxiv.org/abs/2609.34378)、[异步智能体](https://arxiv.org/abs/2609.35427)。RSI 讨论：[最后一个人类造的 AI](https://arxiv.org/abs/2609.11873)、[Atria Dawn](https://arxiv.org/abs/2609.15818)。

**2026-10。**[AAIF 新增 Agent Skills、Open Responses、OpenEnv、OpenSandbox](https://www.linuxfoundation.org/press/agentic-ai-foundation-announces-four-new-projects-agent-skills-open-responses-openenv-and-opensandbox)，项目增至十个。记忆的净价值：[Memory Tax](https://arxiv.org/abs/2610.34073)、[朴素 RAG 比想象中强](https://arxiv.org/abs/2610.30755)；[Cache Engineering for Agents](https://arxiv.org/abs/2610.28577)。训练：[NeMo-DCR](https://arxiv.org/abs/2610.08430)、[Harness Scaling](https://arxiv.org/abs/2610.29458)、[长程 RL 需要 Muon](https://arxiv.org/abs/2610.20256)、[自蒸馏的现实检验](https://arxiv.org/abs/2610.19789)。GUI：[OpenGCUA](https://arxiv.org/abs/2610.22200)、[OpenMobile-2](https://arxiv.org/abs/2610.14157)、[AutoGUIWorld](https://arxiv.org/abs/2610.01215)。评测：[BizBench](https://arxiv.org/abs/2610.20424)、[Trial by Trace](https://arxiv.org/abs/2610.32285)、[Δτ-Bench](https://arxiv.org/abs/2610.34238)。安全：[DecepEval](https://arxiv.org/abs/2610.07967)、[UndoBench](https://arxiv.org/abs/2610.05622)、[SkillMarker](https://arxiv.org/abs/2610.15725)、[SWE-MISBEHAVIOR](https://arxiv.org/abs/2610.18206)。应用：[AI 研究团队探索线性注意力架构](https://arxiv.org/abs/2610.28887)、[OfficeTown](https://arxiv.org/abs/2610.23194)、[nanoMuse](https://arxiv.org/abs/2610.08699)。

---

## 附录 B：论文与资源总表（1100 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 核验；★ 为入门推荐；🔥 为 2026-05 及之后（最近半年），◆ 为 2025-11 及之后（最近一年），🆕 为 2024-11 及之后（最近两年）的工作。

### 早期范式与奠基工作（13）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2023-05 | ★ [Voyager](https://arxiv.org/abs/2305.16291) `arXiv:2305.16291` | NVIDIA·Caltech | 在 Minecraft 中自动课程 + 可执行技能库的终身学习智能体：技能库思想的源头 |
| 2023-05 | [ReWOO](https://arxiv.org/abs/2305.18323) `arXiv:2305.18323` |  | 把推理与观测解耦：先规划全部工具调用再执行，减少 token |
| 2023-05 | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) `arXiv:2305.10601` | Princeton·Google DeepMind | 在思维树上搜索与回溯的审慎问题求解 |
| 2023-05 | [Plan-and-Solve](https://arxiv.org/abs/2305.04091) `arXiv:2305.04091` |  | 先制定计划再逐步执行的零样本提示 |
| 2023-04 | ★ [Generative Agents](https://arxiv.org/abs/2304.03442) `arXiv:2304.03442` | Stanford·Google | 25 个带记忆、反思与规划的智能体模拟小镇生活 |
| 2023-03 | [Self-Refine](https://arxiv.org/abs/2303.17651) `arXiv:2303.17651` | CMU | 模型对自己的输出生成反馈并迭代改进 |
| 2023-03 | [HuggingGPT](https://arxiv.org/abs/2303.17580) `arXiv:2303.17580` | Zhejiang·Microsoft | LLM 作为控制器调度 Hugging Face 上的专家模型 |
| 2023-03 | ★ [Reflexion](https://arxiv.org/abs/2303.11366) `arXiv:2303.11366` | Northeastern·MIT | 用语言形式的自我反思作为"强化信号"，跨回合改进 |
| 2023-02 | ★ [Toolformer](https://arxiv.org/abs/2302.04761) `arXiv:2302.04761` | Meta | 模型自监督地学会何时、如何调用 API |
| 2022-11 | [PAL](https://arxiv.org/abs/2211.10435) `arXiv:2211.10435` | CMU | 把推理写成程序交给解释器执行：代码即工具的早期形态 |
| 2022-10 | ★ [ReAct](https://arxiv.org/abs/2210.03629) `arXiv:2210.03629` | Princeton·Google | 交替生成"思考—行动—观察"，成为智能体循环的标准范式 |
| 2022-05 | [MRKL](https://arxiv.org/abs/2205.00445) `arXiv:2205.00445` | AI21 | 模块化神经符号架构：LLM 路由到计算器、API 等专家模块 |
| 2021-12 | [WebGPT](https://arxiv.org/abs/2112.09332) `arXiv:2112.09332` | OpenAI | 让模型使用文本浏览器搜索并引用来源回答问题，是"搜索智能体"的雏形 |

### 工具调用（22）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [工具使用智能体如何失败](https://arxiv.org/abs/2610.07753) `arXiv:2610.07753` |  | 从证据到行动：工具调用智能体改变外部状态时的失败模式 |
| 2026-09 | 🔥 [WEFT](https://arxiv.org/abs/2609.36887) `arXiv:2609.36887` |  | 扩展通用智能体的工具使用后训练 |
| 2026-08 | 🔥 [DART-SD](https://arxiv.org/abs/2608.18524) `arXiv:2608.18524` |  | 多轮工具调用的自蒸馏：菱形拓扑感知的检索与调优 |
| 2026-08 | 🔥 [循环语言模型改进组合式工具调用](https://arxiv.org/abs/2608.18171) `arXiv:2608.18171` |  | 循环语言模型在组合式工具调用上泛化更好 |
| 2026-07 | 🔥 [无环境的 API 调用数据合成](https://arxiv.org/abs/2607.16900) `arXiv:2607.16900` |  | 不依赖真实环境合成 API 调用智能体训练数据 |
| 2026-05 | 🔥 [EnvFactory](https://arxiv.org/abs/2605.18703) `arXiv:2605.18703` |  | 合成可执行环境并稳健 RL，扩展工具使用智能体 |
| 2026-03 | ◆ [工具使用的上下文 RL](https://arxiv.org/abs/2603.08068) `arXiv:2603.08068` |  | LLM 在上下文中通过强化学习学会用工具 |
| 2026-02 | ◆ [ASA](https://arxiv.org/abs/2602.04935) `arXiv:2602.04935` |  | 免训练的表征工程改进工具调用智能体 |
| 2026-01 | ◆ [LLM-in-Sandbox](https://arxiv.org/abs/2601.16206) `arXiv:2601.16206` |  | 让 LLM 在代码沙箱中探索即可激发通用智能体能力 |
| 2025-11 | ◆ [预算感知的工具使用](https://arxiv.org/abs/2511.17006) `arXiv:2511.17006` | Google | 让智能体感知工具调用预算，实现有效扩展 |
| 2025-10 | 🆕 [DeepAgent](https://arxiv.org/abs/2510.21618) `arXiv:2510.21618` |  | 可扩展工具集的通用推理智能体 |
| 2025-04 | 🆕 [ARTIST](https://arxiv.org/abs/2505.01441) `arXiv:2505.01441` | Microsoft | 用强化学习做智能体推理与工具集成 |
| 2025-04 | 🆕 [ToolRL](https://arxiv.org/abs/2504.13958) `arXiv:2504.13958` |  | 工具学习只需奖励 |
| 2025-04 | 🆕 [ReTool](https://arxiv.org/abs/2504.11536) `arXiv:2504.11536` | ByteDance | 用 RL 学习何时、如何调用代码工具 |
| 2025-03 | 🆕 [API 智能体 vs. GUI 智能体](https://arxiv.org/abs/2503.11069) `arXiv:2503.11069` | Microsoft | 两类智能体的分化与融合 |
| 2024-11 | 🆕 [DynaSaur](https://arxiv.org/abs/2411.01747) `arXiv:2411.01747` |  | 超越预定义动作：用代码动态生成动作 |
| 2024-02 | ★ [CodeAct](https://arxiv.org/abs/2402.01030) `arXiv:2402.01030` | UIUC | 以可执行 Python 代码作为统一动作空间，优于 JSON 工具调用 |
| 2023-12 | [LLMCompiler](https://arxiv.org/abs/2312.04511) `arXiv:2312.04511` | UC Berkeley | 把多个函数调用编译成可并行执行的 DAG |
| 2023-09 | [CRAFT](https://arxiv.org/abs/2309.17428) `arXiv:2309.17428` | UIUC | 创建并检索专用工具集来定制 LLM |
| 2023-07 | ★ [ToolLLM](https://arxiv.org/abs/2307.16789) `arXiv:2307.16789` | Tsinghua | 覆盖 16000+ 真实 API 的工具使用数据（ToolBench）与模型 |
| 2023-05 | [LLM as Tool Makers](https://arxiv.org/abs/2305.17126) `arXiv:2305.17126` | Google DeepMind | 强模型制造工具、弱模型使用工具 |
| 2023-05 | [Gorilla](https://arxiv.org/abs/2305.15334) `arXiv:2305.15334` | UC Berkeley | 面向海量 API 的检索增强微调，减少 API 幻觉 |

### 规划、推理与反思（17）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Interlock](https://arxiv.org/abs/2610.19686) `arXiv:2610.19686` |  | 长程智能体从局部正确走向整体任务完成 |
| 2026-10 | 🔥 [DeLoop](https://arxiv.org/abs/2610.19376) `arXiv:2610.19376` |  | 长程智能体常重复已做过的工作：把重复动作当作自监督信号 |
| 2026-10 | 🔥 [Agent-Reflex](https://arxiv.org/abs/2610.17953) `arXiv:2610.17953` |  | 教语言智能体反思式行动：在关键步骤先反思再执行 |
| 2026-09 | 🔥 [横向与纵向主动性](https://arxiv.org/abs/2609.37236) `arXiv:2609.37236` |  | 智能体主动完成用户未明说但需要的事 |
| 2026-09 | 🔥 [主动式智能体基础](https://arxiv.org/abs/2609.37267) `arXiv:2609.37267` |  | 主动式智能体的原则、技术层次与 Proactivity-Gym |
| 2026-09 | 🔥 [EVOKE](https://arxiv.org/abs/2609.38334) `arXiv:2609.38334` |  | 激发智能体的世界知识，用于可迁移的多步决策 |
| 2026-09 | 🔥 [Marathoner](https://arxiv.org/abs/2609.34378) `arXiv:2609.34378` |  | 超长程自主智能 |
| 2026-09 | 🔥 [LLM 是通用的异步智能体](https://arxiv.org/abs/2609.35427) `arXiv:2609.35427` |  | 打破顺序交互：LLM 作为异步智能体边想边做 |
| 2026-09 | 🔥 [Procedural Graphs](https://arxiv.org/abs/2609.09153) `arXiv:2609.09153` |  | 智能体的自演化执行结构 |
| 2026-08 | 🔥 [Second Thought](https://arxiv.org/abs/2608.13667) `arXiv:2608.13667` |  | 在行动与观察的同时并行推理 |
| 2026-07 | 🔥 [多轮长程规划的物理学](https://arxiv.org/abs/2607.24720) `arXiv:2607.24720` |  | 从预训练到后训练理解多轮长程规划 |
| 2026-01 | ◆ [大模型的智能体式推理](https://arxiv.org/abs/2601.12538) `arXiv:2601.12538` |  | 智能体式推理综述 |
| 2026-01 | ◆ [MAXS](https://arxiv.org/abs/2601.09259) `arXiv:2601.09259` |  | LLM 智能体的元自适应探索 |
| 2025-06 | 🆕 [智能体的测试时扩展](https://arxiv.org/abs/2506.12928) `arXiv:2506.12928` |  | LLM 智能体的测试时计算扩展 |
| 2025-02 | 🆕 [过度思考的危险](https://arxiv.org/abs/2502.08235) `arXiv:2502.08235` | UC Berkeley | 智能体任务中的推理—行动两难：想太多反而做不好 |
| 2024-10 | [WALL-E](https://arxiv.org/abs/2410.07484) `arXiv:2410.07484` |  | 规则学习对齐世界模型，改进基于世界模型的智能体 |
| 2023-10 | [LATS](https://arxiv.org/abs/2310.04406) `arXiv:2310.04406` | UIUC | 把蒙特卡洛树搜索用于智能体的推理、行动与规划 |

### 上下文工程与记忆（44）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Memory Tax](https://arxiv.org/abs/2610.34073) `arXiv:2610.34073` |  | 拆解智能体记忆的净价值：构建与检索记忆的成本可能抵消收益 |
| 2026-10 | 🔥 [朴素 RAG 比想象中强](https://arxiv.org/abs/2610.30755) `arXiv:2610.30755` |  | 长期对话记忆中，复杂的智能体记忆系统未必胜过朴素 RAG |
| 2026-10 | 🔥 [以记忆为中心的持续学习](https://arxiv.org/abs/2610.29357) `arXiv:2610.29357` |  | 让固定权重的智能体通过记忆持续学习 |
| 2026-10 | 🔥 [Cache Engineering for Agents](https://arxiv.org/abs/2610.28577) `arXiv:2610.28577` |  | 长轨迹智能体的缓存工程：提示缓存命中率、上下文布局与成本 |
| 2026-10 | 🔥 [多模态智能体记忆的演化](https://arxiv.org/abs/2610.19366) `arXiv:2610.19366` |  | 多模态智能体记忆综述：看见、维护与学习 |
| 2026-10 | 🔥 [CarryOn](https://arxiv.org/abs/2610.18784) `arXiv:2610.18784` |  | 为长程智能体构建面向对象的上下文，避免反复处理冗长交互历史 |
| 2026-10 | 🔥 [能力驱动的记忆自演化](https://arxiv.org/abs/2610.06361) `arXiv:2610.06361` |  | 用任务反馈迭代改进可执行的记忆程序 |
| 2026-10 | 🔥 [MemPilot](https://arxiv.org/abs/2610.06830) `arXiv:2610.06830` |  | 按需编排多模态记忆整理 |
| 2026-10 | 🔥 [显式信念状态](https://arxiv.org/abs/2610.01415) `arXiv:2610.01415` |  | 用显式信念状态而非原始记忆来驾驭长程智能体 |
| 2026-09 | 🔥 [选择何时取代提取](https://arxiv.org/abs/2609.34227) `arXiv:2609.34227` |  | 预注册检验：对话记忆需要抽取事实，还是选对原始轮次即可 |
| 2026-09 | 🔥 [FlowState](https://arxiv.org/abs/2609.34565) `arXiv:2609.34565` |  | 以执行状态作为长程智能体的记忆 |
| 2026-09 | 🔥 [Just-in-Time Memory](https://arxiv.org/abs/2609.27334) `arXiv:2609.27334` |  | 学习整理任务自适应的智能体记忆 |
| 2026-09 | 🔥 [Jev-Mem](https://arxiv.org/abs/2609.23986) `arXiv:2609.23986` |  | 由"系统 1"控制的高效智能体记忆 |
| 2026-09 | 🔥 [持续学习机制可组合](https://arxiv.org/abs/2609.06986) `arXiv:2609.06986` | Johns Hopkins | 多种持续学习机制组合，实现长程记忆 |
| 2026-09 | 🔥 [PARSER](https://arxiv.org/abs/2609.06702) `arXiv:2609.06702` |  | 长上下文智能体：并行阅读、深度推理 |
| 2026-08 | 🔥 [ContextPilot](https://arxiv.org/abs/2608.28476) `arXiv:2608.28476` |  | 用细粒度 RL 教智能体主动管理上下文 |
| 2026-08 | 🔥 [递归的经验—工作记忆演化](https://arxiv.org/abs/2608.24876) `arXiv:2608.24876` |  | 长程 harness 中记忆的递归演化 |
| 2026-08 | 🔥 [记忆基座的整体评测](https://arxiv.org/abs/2608.15008) `arXiv:2608.15008` |  | 记忆型智能体中不同记忆基座的整体评测 |
| 2026-08 | 🔥 [智能体记忆蒸馏](https://arxiv.org/abs/2608.07169) `arXiv:2608.07169` |  | 用分层教师记忆增强小模型智能体 |
| 2026-07 | 🔥 [Σ-Mem](https://arxiv.org/abs/2607.27958) `arXiv:2607.27958` |  | 多智能体系统的在线可靠性记忆 |
| 2026-07 | 🔥 [MemHarness](https://arxiv.org/abs/2607.28272) `arXiv:2607.28272` |  | 记忆是重建的，不是回放的 |
| 2026-07 | 🔥 [智能体上下文管理](https://arxiv.org/abs/2607.21503) `arXiv:2607.21503` |  | 把记忆与成本当作生命周期与架构问题来管理 |
| 2026-07 | 🔥 [AutoMem](https://arxiv.org/abs/2607.01224) `arXiv:2607.01224` |  | 把记忆当作可学习的认知技能 |
| 2026-06 | 🔥 [智能体原生记忆系统准备好了吗](https://arxiv.org/abs/2606.24775) `arXiv:2606.24775` |  | 从简单检索到智能体原生记忆系统：现状与差距 |
| 2026-06 | 🔥 [自压缩智能体](https://arxiv.org/abs/2606.23525) `arXiv:2606.23525` |  | 让模型自己压缩长轨迹中的过时上下文 |
| 2026-06 | 🔥 [管理程序性记忆](https://arxiv.org/abs/2606.23127) `arXiv:2606.23127` |  | 工作场景中智能体程序性记忆的控制、适应与评估 |
| 2026-06 | 🔥 [图记忆](https://arxiv.org/abs/2606.06036) `arXiv:2606.06036` |  | 记忆是重建而非检索：LLM 智能体的图记忆 |
| 2026-05 | 🔥 [先个性化再存储](https://arxiv.org/abs/2605.25535) `arXiv:2605.25535` |  | 长程智能体的个性化记忆基准与学习 |
| 2026-05 | 🔥 [PREPING](https://arxiv.org/abs/2605.13880) `arXiv:2605.13880` |  | 无需任务也能构建智能体记忆 |
| 2026-04 | ◆ [从上下文到技能](https://arxiv.org/abs/2604.27660) `arXiv:2604.27660` |  | 语言模型能否熟练地从上下文中学习技能 |
| 2026-02 | ◆ [探索式记忆增强智能体](https://arxiv.org/abs/2602.23008) `arXiv:2602.23008` |  | 在策略与离策略混合优化的探索式智能体 |
| 2026-02 | ◆ [MemSkill](https://arxiv.org/abs/2602.02474) `arXiv:2602.02474` |  | 学习并演化记忆技能 |
| 2026-01 | ◆ [SimpleMem](https://arxiv.org/abs/2601.02553) `arXiv:2601.02553` |  | 高效的 LLM 智能体终身记忆 |
| 2025-12 | ◆ [MemEvolve](https://arxiv.org/abs/2512.18746) `arXiv:2512.18746` |  | 智能体记忆系统的元演化 |
| 2025-12 | ◆ [AI 智能体时代的记忆](https://arxiv.org/abs/2512.13564) `arXiv:2512.13564` |  | 智能体记忆综述 |
| 2025-11 | ◆ [通用智能体记忆（GAM）](https://arxiv.org/abs/2511.18423) `arXiv:2511.18423` |  | 用深度研究式的即时检索代替静态预构建记忆 |
| 2025-11 | ◆ [O-Mem](https://arxiv.org/abs/2511.13593) `arXiv:2511.13593` |  | 个性化、长程、自演化智能体的全方位记忆系统 |
| 2025-10 | ★ 🆕 [ACE（智能体上下文工程）](https://arxiv.org/abs/2510.04618) `arXiv:2510.04618` | Stanford | 把上下文当作不断演化的"剧本"，通过生成—反思—整理自我改进 |
| 2025-08 | 🆕 [Memp](https://arxiv.org/abs/2508.06433) `arXiv:2508.06433` |  | 探索智能体的程序性记忆 |
| 2025-07 | 🆕 [潜意识线程](https://arxiv.org/abs/2507.16784) `arXiv:2507.16784` |  | 突破上下文限制的长程推理：递归分解为子线程 |
| 2025-07 | 🆕 [MIRIX](https://arxiv.org/abs/2507.07957) `arXiv:2507.07957` |  | 多智能体记忆系统 |
| 2025-04 | 🆕 [Mem0](https://arxiv.org/abs/2504.19413) `arXiv:2504.19413` | Mem0 | 面向生产的可扩展长期记忆 |
| 2024-09 | [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) `arXiv:2409.07429` | CMU | 从过往经验中归纳可复用工作流作为记忆 |
| 2023-10 | ★ [MemGPT](https://arxiv.org/abs/2310.08560) `arXiv:2310.08560` | UC Berkeley | 把上下文窗口当作内存、外部存储当作磁盘，像操作系统一样分页管理（Letta 的前身） |

### Harness、技能与脚手架（90）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Harness-模型协作持续学习](https://arxiv.org/abs/2610.34724) `arXiv:2610.34724` |  | 长程持续学习中 harness 与模型分工积累程序性知识 |
| 2026-10 | 🔥 [OpenVLHarness](https://arxiv.org/abs/2610.26860) `arXiv:2610.26860` |  | 真实世界的视觉智能体工具使用，配多模态记忆的开源 harness |
| 2026-10 | 🔥 [HarnessPro](https://arxiv.org/abs/2610.26770) `arXiv:2610.26770` |  | 带帕累托优化的递归 harness 自我改进，兼顾性能与成本 |
| 2026-10 | 🔥 [JevAgentHarness](https://arxiv.org/abs/2610.21412) `arXiv:2610.21412` |  | 与 LLM 共享控制权的决策模型：评测、部署与训练 |
| 2026-10 | 🔥 [Socratic Skill Learning](https://arxiv.org/abs/2610.17860) `arXiv:2610.17860` |  | 教与练交替学习可迁移的智能体技能 |
| 2026-10 | 🔥 [Self-Evolving Agent Harness](https://arxiv.org/abs/2610.18813) `arXiv:2610.18813` |  | 让智能体自行修改提示、记忆、工作流等脚手架 |
| 2026-10 | 🔥 [VIEWCRAFT](https://arxiv.org/abs/2610.15915) `arXiv:2610.15915` |  | 演化多模态智能体"看什么、怎么读"的技能 |
| 2026-10 | 🔥 [按能力匹配的技能自适应](https://arxiv.org/abs/2610.33024) `arXiv:2610.33024` |  | 让 harness 自我演化随模型能力调整技能 |
| 2026-10 | 🔥 [Harness²](https://arxiv.org/abs/2610.14224) `arXiv:2610.14224` |  | 递归地为开放世界任务构建智能体 harness |
| 2026-10 | 🔥 [ViSkill](https://arxiv.org/abs/2610.12403) `arXiv:2610.12403` |  | 用演化的视觉原生技能强化 VLM 智能体 |
| 2026-10 | 🔥 [Recurrent Harness](https://arxiv.org/abs/2610.12668) `arXiv:2610.12668` |  | 面向持续学习的循环式 harness |
| 2026-10 | 🔥 [SkillForge](https://arxiv.org/abs/2610.09832) `arXiv:2610.09832` |  | 技能与智能体协同演化：动态的技能生命周期 |
| 2026-10 | 🔥 [EVISKILL](https://arxiv.org/abs/2610.05030) `arXiv:2610.05030` |  | 基于可回放证据的技能演化 |
| 2026-10 | 🔥 [VeriHarness](https://arxiv.org/abs/2610.00972) `arXiv:2610.00972` |  | 为长程任务扩展智能体式验证 |
| 2026-10 | 🔥 [ActiveSaddler](https://arxiv.org/abs/2610.00906) `arXiv:2610.00906` |  | 为自动 harness 优化设计课程，迭代更新智能体脚手架 |
| 2026-09 | 🔥 [Mid-Harness](https://arxiv.org/abs/2609.39982) `arXiv:2609.39982` |  | 在模型与 harness 之间扩展动作空间：面向终端智能体 |
| 2026-09 | 🔥 [跨 harness 的检索增强技能优化](https://arxiv.org/abs/2609.38024) `arXiv:2609.38024` |  | 技能在不同 harness 间迁移与优化 |
| 2026-09 | 🔥 [学习 harness 设计的元技能](https://arxiv.org/abs/2609.38143) `arXiv:2609.38143` |  | 测试时 AI4AI：学习设计 harness 的元技能 |
| 2026-09 | 🔥 [MILO](https://arxiv.org/abs/2609.38349) `arXiv:2609.38349` |  | 多智能体协同演化自动发现 harness |
| 2026-09 | 🔥 [Raven](https://arxiv.org/abs/2609.33439) `arXiv:2609.33439` |  | "harness 的 harness"：可组合的智能体能力框架，跨领域复用 harness |
| 2026-09 | 🔥 [ExpVoyager](https://arxiv.org/abs/2609.32630) `arXiv:2609.32630` |  | 直接在经验中导航以动态合成技能 |
| 2026-09 | 🔥 [Omni-IO Skills](https://arxiv.org/abs/2609.31847) `arXiv:2609.31847` |  | 让通用智能体原生处理全模态输入输出的技能 |
| 2026-09 | 🔥 [RRSI](https://arxiv.org/abs/2609.24972) `arXiv:2609.24972` |  | 正则化的 harness 递归自我改进：防止 harness 过拟合 |
| 2026-09 | 🔥 [Harness-Zero](https://arxiv.org/abs/2609.24974) `arXiv:2609.24974` |  | 智能体充当 harness：把 harness 能力蒸馏进模型 |
| 2026-09 | 🔥 [GraphSkillEvo](https://arxiv.org/abs/2609.21749) `arXiv:2609.21749` |  | 图结构技能的进化优化 |
| 2026-09 | 🔥 [编码智能体 harness 设计的实证研究](https://arxiv.org/abs/2609.20804) `arXiv:2609.20804` |  | 编码 harness 的哪些设计决定了长程表现 |
| 2026-09 | 🔥 [SoL-Pi](https://arxiv.org/abs/2609.20519) `arXiv:2609.20519` |  | 递归扩展自动研究循环，优化昼夜不停的编码智能体 harness |
| 2026-09 | 🔥 [ModularRSI](https://arxiv.org/abs/2609.14857) `arXiv:2609.14857` |  | 模块化、可泛化的 harness 递归自我改进 |
| 2026-09 | 🔥 [COBRA-Skills](https://arxiv.org/abs/2609.11682) `arXiv:2609.11682` |  | 上下文老虎机引导的技能演化 |
| 2026-09 | 🔥 [从代码大规模合成技能](https://arxiv.org/abs/2609.05571) `arXiv:2609.05571` |  | 以代码为依据大规模合成可复用的智能体技能 |
| 2026-09 | 🔥 [Repo-To-Skill](https://arxiv.org/abs/2609.02749) `arXiv:2609.02749` |  | 把 GitHub 仓库蒸馏为面向 AI 研究的智能体技能 |
| 2026-09 | 🔥 [HarnessDev](https://arxiv.org/abs/2609.01437) `arXiv:2609.01437` |  | LLM 能否创建并演化自己的 harness |
| 2026-08 | 🔥 [WHALE](https://arxiv.org/abs/2609.00196) `arXiv:2609.00196` |  | harness 与权重联合优化的简单配方 |
| 2026-08 | 🔥 [WikiSkill](https://arxiv.org/abs/2608.27454) `arXiv:2608.27454` |  | 把智能体经验编译为持久知识以演化技能 |
| 2026-08 | 🔥 [StarHarness](https://arxiv.org/abs/2608.24804) `arXiv:2608.24804` |  | 用分层搜索为企业环境演化专用 harness |
| 2026-08 | 🔥 [Prime Agent](https://arxiv.org/abs/2608.23552) `arXiv:2608.23552` |  | 自我改进的递归语言模型（RLM）harness |
| 2026-08 | 🔥 [AutoSaddler](https://arxiv.org/abs/2608.23041) `arXiv:2608.23041` |  | 从执行轨迹得到持久更新的自动 harness 优化 |
| 2026-08 | 🔥 [StateM](https://arxiv.org/abs/2608.15089) `arXiv:2608.15089` |  | 通过 harness 扩展在 Terminal-Bench 2.1 上达到 95.3%，或 15 美元跑完前沿水平 |
| 2026-08 | 🔥 [揭秘智能体技能](https://arxiv.org/abs/2608.14036) `arXiv:2608.14036` |  | 技能为何有效、何时失效 |
| 2026-08 | 🔥 [AutoDesign](https://arxiv.org/abs/2608.13560) `arXiv:2608.13560` |  | 长程设计任务的元 harness 优化 |
| 2026-08 | 🔥 [测试时 AI4AI](https://arxiv.org/abs/2608.12307) `arXiv:2608.12307` |  | 通过 harness 把强模型能力迁移给弱模型 |
| 2026-08 | 🔥 [SKILLER](https://arxiv.org/abs/2608.10538) `arXiv:2608.10538` |  | 小模型的语言级 RL 技能抽取 |
| 2026-08 | 🔥 [SkillZip（契约保持）](https://arxiv.org/abs/2608.05604) `arXiv:2608.05604` |  | 保持契约的图压缩，扩展智能体技能库 |
| 2026-08 | 🔥 [Skill Entropy](https://arxiv.org/abs/2608.05139) `arXiv:2608.05139` |  | 以技能熵评测并训练长程推理中的技能切换 |
| 2026-08 | 🔥 [OneDayAgent](https://arxiv.org/abs/2608.05013) `arXiv:2608.05013` |  | 面向开放日常请求的长程 harness |
| 2026-08 | 🔥 [SKT](https://arxiv.org/abs/2608.02287) `arXiv:2608.02287` |  | 通过可验证的合成数据大规模训练技能使用 |
| 2026-08 | 🔥 [RL 渐进生成技能](https://arxiv.org/abs/2608.01678) `arXiv:2608.01678` |  | 用强化学习渐进地生成智能体技能 |
| 2026-08 | 🔥 [LongHorizon-Harness](https://arxiv.org/abs/2608.01964) `arXiv:2608.01964` |  | 面向真实长程任务的 harness |
| 2026-07 | 🔥 [DarwinX](https://arxiv.org/abs/2608.07545) `arXiv:2608.07545` |  | 通过自然选择演化智能体 harness |
| 2026-07 | 🔥 [SkillRise](https://arxiv.org/abs/2607.26784) `arXiv:2607.26784` |  | 跨任务技能演化的智能体强化学习 |
| 2026-07 | 🔥 [JarvisHub](https://arxiv.org/abs/2607.23588) `arXiv:2607.23588` |  | 面向画布的多模态创作智能体开源 harness |
| 2026-07 | 🔥 [Skill Self-Play](https://arxiv.org/abs/2607.22529) `arXiv:2607.22529` |  | 技能协同演化的自我博弈推动能力前沿 |
| 2026-07 | 🔥 [递归 harness 自我改进](https://arxiv.org/abs/2607.15524) `arXiv:2607.15524` |  | 模型与 harness 协同演化下，harness 也是产生训练数据的组件 |
| 2026-07 | 🔥 [Harness Handbook](https://arxiv.org/abs/2607.13285) `arXiv:2607.13285` |  | 让不断演化的智能体 harness 可读、可导航、可编辑 |
| 2026-07 | 🔥 [SkillOpt-Lite](https://arxiv.org/abs/2607.03451) `arXiv:2607.03451` |  | 一行"氛围"提示实现更快的智能体自演化 |
| 2026-07 | 🔥 [SkillCoach](https://arxiv.org/abs/2607.01874) `arXiv:2607.01874` |  | 自演化评分细则评估并改进智能体的技能使用 |
| 2026-06 | 🔥 [RESOURCE2SKILL](https://arxiv.org/abs/2606.29538) `arXiv:2606.29538` |  | 从人类制作的多模态资料中蒸馏可执行技能 |
| 2026-06 | 🔥 [AOHP](https://arxiv.org/abs/2606.23449) `arXiv:2606.23449` |  | 开源的操作系统级智能体 harness：个性化、高效、安全 |
| 2026-06 | 🔥 [HarnessX](https://arxiv.org/abs/2606.14249) `arXiv:2606.14249` |  | 可组合、自适应、可演化的智能体 harness 工厂 |
| 2026-06 | 🔥 [SkillHone](https://arxiv.org/abs/2606.08671) `arXiv:2606.08671` |  | 通过持久决策历史持续演化技能 |
| 2026-06 | 🔥 [回顾式 harness 优化](https://arxiv.org/abs/2606.05922) `arXiv:2606.05922` |  | 通过轨迹回放上的自偏好改进智能体 harness |
| 2026-06 | 🔥 [OpenSkill](https://arxiv.org/abs/2606.06741) `arXiv:2606.06741` |  | 部署后开放世界中的智能体自我演化 |
| 2026-06 | 🔥 [LatentSkill](https://arxiv.org/abs/2606.06087) `arXiv:2606.06087` |  | 从上下文中的文本技能到权重中的潜在技能 |
| 2026-05 | 🔥 [SkillAdaptor](https://arxiv.org/abs/2606.01311) `arXiv:2606.01311` |  | 从轨迹自适应调整技能 |
| 2026-05 | 🔥 [COLLEAGUE.SKILL](https://arxiv.org/abs/2605.31264) `arXiv:2605.31264` |  | 通过专家知识蒸馏自动生成智能体技能 |
| 2026-05 | 🔥 [Skill0.5](https://arxiv.org/abs/2605.28424) `arXiv:2605.28424` |  | 技能内化与使用联合训练，提升分布外泛化 |
| 2026-05 | 🔥 [SkillGrad](https://arxiv.org/abs/2605.27760) `arXiv:2605.27760` |  | 像梯度下降一样优化智能体技能 |
| 2026-05 | 🔥 [MUSE-Autoskill](https://arxiv.org/abs/2605.27366) `arXiv:2605.27366` |  | 通过技能创建、记忆、管理与评测自我演化 |
| 2026-05 | 🔥 [模型生成技能的系统研究](https://arxiv.org/abs/2605.23899) `arXiv:2605.23899` |  | 从原始经验到技能使用：模型生成的智能体技能 |
| 2026-05 | 🔥 [SkillOpt](https://arxiv.org/abs/2605.23904) `arXiv:2605.23904` |  | 为自演化的智能体技能设计"执行策略"，而非一次性生成或松散演化 |
| 2026-05 | 🔥 [技能程序](https://arxiv.org/abs/2605.17734) `arXiv:2605.17734` |  | 用技能程序驾驭 LLM 智能体 |
| 2026-05 | 🔥 [SkillsVote](https://arxiv.org/abs/2605.18401) `arXiv:2605.18401` |  | 智能体技能从收集、推荐到演化的生命周期治理 |
| 2026-05 | 🔥 [Code as Agent Harness](https://arxiv.org/abs/2605.18747) `arXiv:2605.18747` |  | 用代码作为智能体的 harness：以程序组织工具、记忆与控制流 |
| 2026-05 | 🔥 [MMSkills](https://arxiv.org/abs/2605.13527) `arXiv:2605.13527` |  | 面向通用视觉智能体的多模态技能 |
| 2026-05 | 🔥 [Continual Harness](https://arxiv.org/abs/2605.09998) `arXiv:2605.09998` |  | 在线适配 Claude Code、OpenHands 等编码 harness 的自我改进 |
| 2026-05 | 🔥 [SkillOS](https://arxiv.org/abs/2605.06614) `arXiv:2605.06614` |  | 为自演化智能体学习技能整理 |
| 2026-05 | 🔥 [Skill1](https://arxiv.org/abs/2605.06130) `arXiv:2605.06130` |  | 通过 RL 统一演化技能增强型智能体 |
| 2026-05 | 🔥 [HeavySkill](https://arxiv.org/abs/2605.02396) `arXiv:2605.02396` |  | 重度思考作为 harness 中的内在技能 |
| 2026-04 | ◆ [SkillClaw](https://arxiv.org/abs/2604.08377) `arXiv:2604.08377` |  | 让 OpenClaw 类智能体的技能通过智能体演化器集体演化 |
| 2026-04 | ◆ [LLM 智能体中的外化](https://arxiv.org/abs/2604.08224) `arXiv:2604.08224` |  | 记忆、技能、协议与 harness 工程的统一综述：能力越来越多地外化在模型之外 |
| 2026-04 | ◆ [SKILL0](https://arxiv.org/abs/2604.02268) `arXiv:2604.02268` |  | 上下文内智能体 RL 实现技能内化 |
| 2026-03 | ◆ [Trace2Skill](https://arxiv.org/abs/2603.25158) `arXiv:2603.25158` |  | 把轨迹中的局部经验蒸馏为可迁移技能 |
| 2026-03 | ◆ [Memento-Skills](https://arxiv.org/abs/2603.18743) `arXiv:2603.18743` |  | 让智能体设计智能体 |
| 2026-03 | ◆ [XSkill](https://arxiv.org/abs/2603.12056) `arXiv:2603.12056` |  | 多模态智能体从经验与技能中持续学习 |
| 2026-02 | ◆ [SkillNet](https://arxiv.org/abs/2603.04448) `arXiv:2603.04448` |  | 创建、评估并连接 AI 技能 |
| 2026-02 | ◆ [SkillRL](https://arxiv.org/abs/2602.08234) `arXiv:2602.08234` |  | 递归的技能增强强化学习演化智能体 |
| 2026-01 | ◆ [可演化的程序化技能网络](https://arxiv.org/abs/2601.03509) `arXiv:2601.03509` |  | 开放环境中持续获取技能 |
| 2025-12 | ◆ [带技能库的自我改进 RL](https://arxiv.org/abs/2512.17102) `arXiv:2512.17102` |  | 用强化学习训练带技能库的自我改进智能体 |
| 2025-11 | ◆ [Agent READMEs](https://arxiv.org/abs/2511.12884) `arXiv:2511.12884` |  | 智能体编码上下文文件（AGENTS.md / CLAUDE.md 等）的实证研究 |
| 2025-10 | 🆕 [官方技能示例库](https://github.com/anthropics/skills) | Anthropic | 文档处理等官方 Agent Skills 示例 |

### 多智能体与编排（37）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [DynaFlow](https://arxiv.org/abs/2610.34467) `arXiv:2610.34467` |  | 学习自由形式的多智能体编排，而非依赖预定义协作结构 |
| 2026-10 | 🔥 [组相对技能信用分配](https://arxiv.org/abs/2610.27227) `arXiv:2610.27227` |  | 演化中的多智能体团队：该改哪个智能体的技能 |
| 2026-10 | 🔥 [MiniCorp](https://arxiv.org/abs/2610.05912) `arXiv:2610.05912` |  | "自己运转的公司"：训练并适配由智能体组成的企业 |
| 2026-09 | 🔥 [跨模型族 KV 缓存迁移](https://arxiv.org/abs/2609.32259) `arXiv:2609.32259` |  | 异构多智能体之间免预填充地迁移 KV 缓存 |
| 2026-09 | 🔥 [Agensh](https://arxiv.org/abs/2609.26781) `arXiv:2609.26781` |  | 把组织智能扩展到 1024 个并发智能体 |
| 2026-09 | 🔥 [AgentGrad](https://arxiv.org/abs/2609.08572) `arXiv:2609.08572` |  | 干预引导的多智能体系统提示优化 |
| 2026-09 | 🔥 [双层协调反思](https://arxiv.org/abs/2609.02750) `arXiv:2609.02750` |  | 博弈论视角的多智能体系统：编排者与工人的双层反思 |
| 2026-08 | 🔥 [LLM 智能体时代的图工程](https://arxiv.org/abs/2608.21156) `arXiv:2608.21156` |  | 从个体智能到系统智能 |
| 2026-06 | 🔥 [Orchestra-o1](https://arxiv.org/abs/2606.13707) `arXiv:2606.13707` |  | 全模态智能体编排 |
| 2026-06 | 🔥 [多智能体推理中的流式通信](https://arxiv.org/abs/2606.05158) `arXiv:2606.05158` |  | 流式传递中间结果，降低多智能体流水线延迟 |
| 2026-05 | 🔥 [多智能体 RL 何时改进 LLM 工作流](https://arxiv.org/abs/2605.24202) `arXiv:2605.24202` |  | 工作流、规模与策略共享的权衡 |
| 2026-05 | 🔥 [超越个体智能](https://arxiv.org/abs/2605.14892) `arXiv:2605.14892` |  | 多智能体协作、失败归因与自我演化综述 |
| 2026-05 | 🔥 [MetaAgent-X](https://arxiv.org/abs/2605.14212) `arXiv:2605.14212` |  | 端到端 RL 突破自动多智能体系统上限 |
| 2026-05 | 🔥 [TMAS](https://arxiv.org/abs/2605.10344) `arXiv:2605.10344` |  | 多智能体协同扩展测试时计算 |
| 2026-04 | ◆ [递归多智能体系统](https://arxiv.org/abs/2604.25917) `arXiv:2604.25917` |  | 把循环/递归扩展的思路用于多智能体系统 |
| 2026-04 | ◆ [从技能到人才](https://arxiv.org/abs/2604.22446) `arXiv:2604.22446` |  | 把异构智能体组织成真实公司 |
| 2026-04 | ◆ [CORAL](https://arxiv.org/abs/2604.01658) `arXiv:2604.01658` |  | 面向开放式发现的自主多智能体演化 |
| 2026-03 | ◆ [智能体工作流优化综述](https://arxiv.org/abs/2603.22386) `arXiv:2603.22386` |  | 从静态模板到动态运行时图 |
| 2026-03 | ◆ [异构智能体协作 RL](https://arxiv.org/abs/2603.02604) `arXiv:2603.02604` |  | 异构智能体的协作式强化学习 |
| 2026-02 | ◆ [SkillOrchestra](https://arxiv.org/abs/2602.19672) `arXiv:2602.19672` |  | 通过技能迁移学习路由智能体 |
| 2026-02 | ◆ [AOrchestra](https://arxiv.org/abs/2602.03786) `arXiv:2602.03786` |  | 为智能体编排自动创建子智能体 |
| 2026-01 | ◆ [协作式多智能体测试时 RL](https://arxiv.org/abs/2601.09667) `arXiv:2601.09667` |  | 多智能体在测试时协作强化学习做推理 |
| 2025-12 | ◆ [DoVer](https://arxiv.org/abs/2512.06749) `arXiv:2512.06749` |  | 干预驱动的多智能体系统自动调试 |
| 2025-11 | ◆ [多智能体中的潜在协作](https://arxiv.org/abs/2511.20639) `arXiv:2511.20639` |  | 多智能体在潜空间而非文本中协作 |
| 2025-10 | 🆕 [智能体组织的时代](https://arxiv.org/abs/2510.26658) `arXiv:2510.26658` | Microsoft | 学习用语言模型组织（异步并行思考） |
| 2025-10 | 🆕 [AgentFlow](https://arxiv.org/abs/2510.05592) `arXiv:2510.05592` | Stanford | 流中优化的智能体系统：规划与工具使用 |
| 2025-10 | 🆕 [多智能体工具集成策略优化](https://arxiv.org/abs/2510.04678) `arXiv:2510.04678` |  | 单模型多角色的多智能体工具集成 RL |
| 2025-09 | 🆕 [虚拟智能体经济](https://arxiv.org/abs/2509.10147) `arXiv:2509.10147` | Google DeepMind | 智能体之间交易的新经济层 |
| 2025-08 | 🆕 [Chain-of-Agents](https://arxiv.org/abs/2508.13167) `arXiv:2508.13167` |  | 通过多智能体蒸馏与智能体 RL 训练端到端智能体基础模型 |
| 2025-06 | 🆕 [SPIRAL](https://arxiv.org/abs/2506.24119) `arXiv:2506.24119` |  | 零和博弈自我对弈激励推理 |
| 2025-03 | ★ 🆕 [多智能体系统为何失败](https://arxiv.org/abs/2503.13657) `arXiv:2503.13657` | UC Berkeley | 多智能体 LLM 系统的失败分类（MAST） |
| 2023-08 | [AgentVerse](https://arxiv.org/abs/2308.10848) `arXiv:2308.10848` | Tsinghua | 多智能体协作与涌现行为 |
| 2023-08 | ★ [AutoGen](https://arxiv.org/abs/2308.08155) `arXiv:2308.08155` | Microsoft | 多智能体对话编程框架 |
| 2023-08 | [MetaGPT](https://arxiv.org/abs/2308.00352) `arXiv:2308.00352` | DeepWisdom | 把标准作业流程（SOP）编码进多智能体软件开发 |
| 2023-07 | [ChatDev](https://arxiv.org/abs/2307.07924) `arXiv:2307.07924` | Tsinghua | 按瀑布式角色分工协作的软件开发智能体 |
| 2023-05 | [多智能体辩论](https://arxiv.org/abs/2305.14325) `arXiv:2305.14325` | MIT·Google | 多个模型辩论以提升事实性与推理 |
| 2023-03 | [CAMEL](https://arxiv.org/abs/2303.17760) `arXiv:2303.17760` | KAUST | 角色扮演的交流式多智能体框架 |

### 协议、框架与基础设施（44）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [AAIF 新增四个项目](https://www.linuxfoundation.org/press/agentic-ai-foundation-announces-four-new-projects-agent-skills-open-responses-openenv-and-opensandbox) | Linux Foundation | 新增 Agent Skills（Anthropic）、Open Responses（OpenAI）、OpenEnv（Hugging Face）、OpenSandbox（Alibaba），一年内从 3 个项目增至 10 个 |
| 2026-08 | 🔥 [A2A 加入 AAIF](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/) | A2A Project | A2A 以成长期项目加入 Agentic AI Foundation，与 MCP、AGENTS.md、goose 同处中立治理 |
| 2026-07 | ★ 🔥 [MCP 2026-07-28 版规范](https://modelcontextprotocol.io/specification/2026-07-28/changelog) | MCP / AAIF | 自发布以来最大修订：去掉会话与 initialize 握手、每个请求自带版本与能力；新增 server/discover；tasks 移入官方扩展 |
| 2026-07 | 🔥 [OO Agents](https://arxiv.org/abs/2607.20709) `arXiv:2607.20709` | NVIDIA | 原生 Python 面向对象智能体：把提示、工具模式与回调统一为对象 |
| 2026-07 | 🔥 [AgentDebugX](https://arxiv.org/abs/2607.18754) `arXiv:2607.18754` |  | 开源的智能体失败可观测、归因与恢复工具包 |
| 2026-06 | 🔥 [OpenRath](https://arxiv.org/abs/2606.19409) `arXiv:2606.19409` |  | 以会话为中心的智能体运行时状态 |
| 2026-05 | 🔥 [Foundation Protocol](https://arxiv.org/abs/2605.23218) `arXiv:2605.23218` |  | 面向"智能体社会"的协调层协议 |
| 2026-05 | 🔥 [Orchard](https://arxiv.org/abs/2605.15040) `arXiv:2605.15040` |  | 开源的智能体建模框架 |
| 2026-05 | 🔥 [MCP-Cosmos](https://arxiv.org/abs/2605.09131) `arXiv:2605.09131` |  | 在 MCP 环境中用世界模型增强复杂任务执行 |
| 2026-04 | ◆ [AgentSPEX](https://arxiv.org/abs/2604.13346) `arXiv:2604.13346` |  | 智能体规格与执行语言：用程序而非反应式提示组织智能体 |
| 2025-12 | ◆ [Youtu-Agent](https://arxiv.org/abs/2512.24615) `arXiv:2512.24615` | Tencent | 自动生成与混合策略优化扩展智能体生产力 |
| 2025-12 | ★ ◆ [Agentic AI Foundation 成立](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) | Linux Foundation | Linux 基金会成立 AAIF，首批项目为 Anthropic 的 MCP、Block 的 goose、OpenAI 的 AGENTS.md |
| 2025-10 | 🆕 [Agent Skills 开放格式](https://agentskills.io/) | Anthropic / AAIF | 技能 = 含说明、脚本与资源的文件夹，按需加载；2026-10 捐给 AAIF |
| 2025-10 | 🆕 [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) | Microsoft | AutoGen 与 Semantic Kernel 合并后的智能体框架 |
| 2025-09 | 🆕 [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | Anthropic | 把 Claude Code 的 harness（工具、上下文压缩、子智能体、权限）作为 SDK 提供 |
| 2025-08 | 🆕 [AgentScope 1.0](https://arxiv.org/abs/2508.16279) `arXiv:2508.16279` | Alibaba | 以开发者为中心的智能体应用框架 |
| 2025-08 | 🆕 [AGENTS.md](https://agents.md/) | OpenAI / AAIF | 给编码智能体看的项目说明文件的开放格式（构建、测试、约定） |
| 2025-07 | ★ 🆕 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | Princeton·Stanford | 约百行的极简 SWE 智能体：只用 bash，仍有很强的 SWE-bench 表现，适合作为教学起点 |
| 2025-06 | 🆕 [OAgents](https://arxiv.org/abs/2506.15741) `arXiv:2506.15741` |  | 构建有效智能体的实证研究：哪些设计真正起作用 |
| 2025-04 | 🆕 [Google ADK](https://github.com/google/adk-python) | Google | Agent Development Kit：构建、评测与部署智能体的开源框架 |
| 2025-04 | 🆕 [A2A 发布](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) | Google | Agent2Agent 协议：不同厂商、框架的智能体之间发现能力、交接任务 |
| 2025-03 | 🆕 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | OpenAI | 轻量的多智能体框架：智能体、交接（handoff）、护栏与追踪 |
| 2025-03 | 🆕 [agentgateway](https://github.com/agentgateway/agentgateway) | AAIF | 连接智能体、模型、工具与 API 的安全网关（AAIF 项目） |
| 2025-03 | 🆕 [Playwright MCP](https://github.com/microsoft/playwright-mcp) | Microsoft | 通过可访问性树（而非截图）操作浏览器的 MCP 服务器 |
| 2025-01 | 🆕 [goose](https://github.com/aaif-goose/goose) | Block / AAIF | 开源、可扩展的本地智能体，原生支持 MCP（AAIF 项目） |
| 2024-12 | 🆕 [smolagents](https://github.com/huggingface/smolagents) | Hugging Face | 以代码为动作的极简智能体库 |
| 2024-12 | 🆕 [Pydantic AI](https://github.com/pydantic/pydantic-ai) | Pydantic | 类型安全的智能体框架，结构化输出与依赖注入 |
| 2024-11 | 🆕 [MCP 官方服务器集合](https://github.com/modelcontextprotocol/servers) | MCP | 官方与参考 MCP 服务器（文件系统、Git、数据库、浏览器等） |
| 2024-11 | ★ 🆕 [MCP 发布](https://www.anthropic.com/news/model-context-protocol) | Anthropic | Model Context Protocol：连接模型与工具、数据源的开放协议，"AI 的 USB-C" |
| 2024-11 | 🆕 [browser-use](https://github.com/browser-use/browser-use) | Browser Use | 让智能体操作浏览器的开源库 |
| 2024-10 | [verl](https://github.com/verl-project/verl) | ByteDance Seed | 大模型 RL 训练框架，广泛用于智能体 RL |
| 2024-09 | [Letta](https://github.com/letta-ai/letta) | Letta | 有状态智能体平台（MemGPT 的延续），自我编辑记忆 |
| 2024-07 | [Cline](https://github.com/cline/cline) | Cline | IDE 内的开源编码智能体，逐步审批工具调用 |
| 2024-05 | [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | UK AISI | 评测框架，含智能体评测、沙箱与大量现成基准 |
| 2024-03 | [OpenHands](https://github.com/OpenHands/OpenHands) | OpenHands | 开源 AI 软件开发智能体平台（原 OpenDevin） |
| 2024-01 | [LangGraph](https://github.com/langchain-ai/langgraph) | LangChain | 以图（状态机）编排有状态智能体，支持持久化与人在回路 |
| 2023-11 | [CrewAI](https://github.com/crewAIInc/crewAI) | CrewAI | 基于角色分工的多智能体编排框架 |
| 2023-10 | [DSPy](https://github.com/stanfordnlp/dspy) | Stanford | 用程序而非提示编写 LM 系统，并自动优化提示 |
| 2023-09 | [AutoGen](https://github.com/microsoft/autogen) | Microsoft | 多智能体对话框架（后与 Semantic Kernel 合并为 Agent Framework） |
| 2023-07 | [Mem0](https://github.com/mem0ai/mem0) | Mem0 | 面向智能体的长期记忆层 |
| 2023-05 | [Langfuse](https://github.com/langfuse/langfuse) | Langfuse | 开源 LLM/智能体可观测性：追踪、评测、提示管理 |
| 2023-05 | [Aider](https://github.com/Aider-AI/aider) | Aider | 终端里的结对编程智能体，以 git 为中心 |
| 2023-03 | [E2B](https://github.com/e2b-dev/E2B) | E2B | 为智能体提供云端代码执行沙箱 |
| 2023-01 | [Arize Phoenix](https://github.com/Arize-ai/phoenix) | Arize | 开源的智能体追踪与评测工具 |

### 软件工程与终端（59）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Harness-策略联合优化](https://arxiv.org/abs/2610.22791) `arXiv:2610.22791` |  | 同时优化 harness 与策略的自演化编码智能体 |
| 2026-10 | 🔥 [Terminal-RSI](https://arxiv.org/abs/2610.22192) `arXiv:2610.22192` |  | 通过自重写与策略精炼递归自我改进的终端智能体 |
| 2026-10 | 🔥 [ReTerm](https://arxiv.org/abs/2610.18484) `arXiv:2610.18484` |  | 终端智能体的恢复训练：学会从错误状态中恢复 |
| 2026-10 | 🔥 [扩展长程终端智能体数据](https://arxiv.org/abs/2610.13166) `arXiv:2610.13166` |  | 终端智能体从本地操作走向真实专业任务：扩展其智能体训练数据 |
| 2026-10 | 🔥 [ProgramGym](https://arxiv.org/abs/2610.13395) `arXiv:2610.13395` |  | 扩展"从零复现程序"的长程编码智能体训练环境 |
| 2026-09 | 🔥 [Opera](https://arxiv.org/abs/2609.33987) `arXiv:2609.33987` |  | 长程编码智能体的言语批评者框架：及时纠偏 |
| 2026-09 | 🔥 [按类别迭代专家训练](https://arxiv.org/abs/2609.23377) `arXiv:2609.23377` |  | 仓库级软件工程任务异质：分类训练专家再合并 |
| 2026-09 | 🔥 [CodeMidas](https://arxiv.org/abs/2609.22068) `arXiv:2609.22068` |  | 从代码本身扩展智能体编码 RL 环境 |
| 2026-09 | 🔥 [ProgramDistill](https://arxiv.org/abs/2609.18805) `arXiv:2609.18805` |  | 从交互式网页应用生成可验证的参考引导 SWE 任务 |
| 2026-09 | 🔥 [T1](https://arxiv.org/abs/2609.11042) `arXiv:2609.11042` |  | 面向长程任务的终端智能体强化学习 |
| 2026-09 | 🔥 [终端智能体的环境演化](https://arxiv.org/abs/2609.04128) `arXiv:2609.04128` |  | 为训练终端智能体扩展可验证环境 |
| 2026-09 | 🔥 [Terminal-Universe](https://arxiv.org/abs/2609.04148) `arXiv:2609.04148` |  | 把智能体轨迹转化为可扩展的终端环境 |
| 2026-09 | 🔥 [MaxKernel](https://arxiv.org/abs/2609.04523) `arXiv:2609.04523` | Google | 面向 TPU 的智能体内核生成 |
| 2026-09 | 🔥 [Harness-of-Harness](https://arxiv.org/abs/2609.01481) `arXiv:2609.01481` |  | 多日自主软件开发与持续改进 |
| 2026-08 | 🔥 [Super Library Agent](https://arxiv.org/abs/2608.29310) `arXiv:2608.29310` |  | 同时生成并维护多个相关应用 |
| 2026-08 | 🔥 [交接税](https://arxiv.org/abs/2608.24358) `arXiv:2608.24358` |  | 接手非自己产生的轨迹时，编码智能体表现下降 |
| 2026-08 | 🔥 [FACET](https://arxiv.org/abs/2608.18580) `arXiv:2608.18580` |  | 合成终端任务时保持原意与可执行状态 |
| 2026-08 | 🔥 [Ouroboros](https://arxiv.org/abs/2608.08311) `arXiv:2608.08311` |  | 经审查的核心演化：自我开发的前沿编码智能体 harness |
| 2026-08 | 🔥 [Mendel Gödel Machine](https://arxiv.org/abs/2608.07645) `arXiv:2608.07645` |  | 通过比较式演化递归自我改进的编码智能体 |
| 2026-08 | 🔥 [DCAS](https://arxiv.org/abs/2608.06113) `arXiv:2608.06113` |  | 解耦 CLI 智能体脚手架，在不同脚手架间内化规划 |
| 2026-08 | 🔥 [CalibForge](https://arxiv.org/abs/2608.06352) `arXiv:2608.06352` |  | 对抗式求解器校准，扩展可学习的终端任务 |
| 2026-08 | 🔥 [长程终端任务的递归合成](https://arxiv.org/abs/2608.05466) `arXiv:2608.05466` |  | 递归合成高质量的长程终端训练数据 |
| 2026-07 | 🔥 [SpecFirst](https://arxiv.org/abs/2607.27167) `arXiv:2607.27167` |  | 把行为规格的引出作为程序合成的第一步 |
| 2026-07 | 🔥 [MindForge](https://arxiv.org/abs/2607.27146) `arXiv:2607.27146` |  | 教小模型全生命周期软件工程 |
| 2026-07 | 🔥 [CodeNib](https://arxiv.org/abs/2607.25431) `arXiv:2607.25431` |  | 为编码智能体提供仓库上下文的多视图数据系统 |
| 2026-07 | 🔥 [SWE-Pruner Pro](https://arxiv.org/abs/2607.18213) `arXiv:2607.18213` |  | 编码模型自己知道该裁剪哪些长上下文 |
| 2026-07 | 🔥 [函数感知的 FIM 中期训练](https://arxiv.org/abs/2607.12463) `arXiv:2607.12463` |  | 编码智能体基础模型的中期训练：把工具返回融入推理 |
| 2026-07 | 🔥 [从人工代码评审到智能体评审](https://arxiv.org/abs/2607.13196) `arXiv:2607.13196` |  | 不同代际生成式 AI 对代码评审的影响 |
| 2026-06 | 🔥 [Dockerless](https://arxiv.org/abs/2606.28436) `arXiv:2606.28436` |  | 免环境的程序验证器，用于训练编码智能体 |
| 2026-06 | 🔥 [验证视野](https://arxiv.org/abs/2606.26300) `arXiv:2606.26300` |  | 编码智能体奖励没有银弹：验证并不总比生成容易 |
| 2026-06 | 🔥 [CLI-Universe](https://arxiv.org/abs/2606.22883) `arXiv:2606.22883` |  | 面向终端智能体的可验证任务合成引擎 |
| 2026-06 | 🔥 [FastContext](https://arxiv.org/abs/2606.14066) `arXiv:2606.14066` |  | 为编码智能体训练高效的仓库探索器 |
| 2026-06 | 🔥 [DeNovoSWE](https://arxiv.org/abs/2606.10728) `arXiv:2606.10728` |  | 扩展"从零生成完整仓库"的长程环境 |
| 2026-04 | ◆ [OpenGame](https://arxiv.org/abs/2604.18394) `arXiv:2604.18394` |  | 面向游戏的开放智能体编程 |
| 2026-04 | ◆ [CodeTracer](https://arxiv.org/abs/2604.11641) `arXiv:2604.11641` |  | 可追溯的代码智能体状态，便于调试 |
| 2026-04 | ◆ [GrandCode](https://arxiv.org/abs/2604.02721) `arXiv:2604.02721` |  | 通过智能体 RL 达到竞赛编程特级大师水平 |
| 2026-03 | ◆ [终端智能体足以胜任企业自动化](https://arxiv.org/abs/2604.00073) `arXiv:2604.00073` |  | 与其做 GUI，不如让终端智能体通过 API/CLI 完成企业自动化 |
| 2026-02 | ◆ [CUDA Agent](https://arxiv.org/abs/2602.24286) `arXiv:2602.24286` |  | 大规模智能体 RL 生成高性能 CUDA 内核 |
| 2026-02 | ◆ [终端能力的数据工程](https://arxiv.org/abs/2602.21193) `arXiv:2602.21193` | NVIDIA | 公开训练终端智能体的数据策略与扩展 |
| 2026-02 | ◆ [TermiGen](https://arxiv.org/abs/2602.07274) `arXiv:2602.07274` |  | 终端智能体的高保真环境与稳健轨迹合成 |
| 2026-02 | ◆ [SWE-World](https://arxiv.org/abs/2602.03419) `arXiv:2602.03419` |  | 免 Docker 环境构建软件工程智能体 |
| 2026-02 | ◆ [SWE-Master](https://arxiv.org/abs/2602.03411) `arXiv:2602.03411` |  | 开源、可复现的软件工程智能体后训练 |
| 2026-02 | ◆ [SWE-Universe](https://arxiv.org/abs/2602.02361) `arXiv:2602.02361` |  | 把真实可验证的 SWE 环境扩展到百万级 |
| 2026-01 | ◆ [daVinci-Dev](https://arxiv.org/abs/2601.18418) `arXiv:2601.18418` |  | 面向软件工程的智能体原生中期训练 |
| 2026-01 | ◆ [SWE-Pruner](https://arxiv.org/abs/2601.16746) `arXiv:2601.16746` |  | 编码智能体的自适应上下文裁剪 |
| 2026-01 | ◆ [LLM 问题修复综述](https://arxiv.org/abs/2601.11655) `arXiv:2601.11655` |  | 基于 LLM 的软件工程 issue 修复综述 |
| 2026-01 | ◆ [MemGovern](https://arxiv.org/abs/2601.06789) `arXiv:2601.06789` |  | 从受治理的人类经验中学习，增强代码智能体 |
| 2025-12 | ◆ [DeepCode](https://arxiv.org/abs/2512.07921) `arXiv:2512.07921` |  | 开放的智能体编程系统 |
| 2025-11 | ◆ [从代码基础模型到智能体](https://arxiv.org/abs/2511.18538) `arXiv:2511.18538` |  | 代码智能实践指南：从代码基础模型到智能体与应用 |
| 2025-08 | 🆕 [RL 训练长上下文多轮 SWE 智能体](https://arxiv.org/abs/2508.03501) `arXiv:2508.03501` | Nebius | 用强化学习训练长上下文、多轮软件工程智能体 |
| 2025-06 | 🆕 [Skywork-SWE](https://arxiv.org/abs/2506.19290) `arXiv:2506.19290` | Skywork | 软件工程的数据扩展律 |
| 2025-06 | 🆕 [SWE-Factory](https://arxiv.org/abs/2506.10954) `arXiv:2506.10954` |  | 自动构建 issue 修复训练数据与基准 |
| 2025-04 | 🆕 [SWE-smith](https://arxiv.org/abs/2504.21798) `arXiv:2504.21798` | Stanford | 为软件工程智能体扩展训练数据 |
| 2025-04 | 🆕 [R2E-Gym](https://arxiv.org/abs/2504.07164) `arXiv:2504.07164` | UC Berkeley | 程序化环境与混合验证器扩展开放 SWE 智能体 |
| 2025-02 | 🆕 [SWE-RL](https://arxiv.org/abs/2502.18449) `arXiv:2502.18449` | Meta | 在开源软件演化数据上用 RL 训练软件工程推理 |
| 2024-12 | 🆕 [SWE-Gym](https://arxiv.org/abs/2412.21139) `arXiv:2412.21139` | UC Berkeley | 训练软件工程智能体与验证器的环境 |
| 2024-07 | ★ [OpenHands](https://arxiv.org/abs/2407.16741) `arXiv:2407.16741` | All Hands AI | 开放的 AI 软件开发者平台（前身 OpenDevin） |
| 2024-07 | [Agentless](https://arxiv.org/abs/2407.01489) `arXiv:2407.01489` | UIUC | 不用复杂智能体，"定位—修复—验证"三步流程也很强 |
| 2024-05 | ★ [SWE-agent](https://arxiv.org/abs/2405.15793) `arXiv:2405.15793` | Princeton | 提出"智能体—计算机接口"（ACI）：为智能体专门设计工具与反馈 |

### 计算机、浏览器与 GUI（95）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [演化智能体接口](https://arxiv.org/abs/2610.28601) `arXiv:2610.28601` |  | 计算机使用智能体受限于为人设计的界面：让接口随智能体演化 |
| 2026-10 | 🔥 [野外的智能体世界模型](https://arxiv.org/abs/2610.26828) `arXiv:2610.26828` |  | 为计算机使用智能体合成交互环境（世界模型充当环境） |
| 2026-10 | 🔥 [Focused Omni-Modal GUI Agents](https://arxiv.org/abs/2610.26203) `arXiv:2610.26203` |  | 在流式全模态观测（屏幕 + 音频）上聚焦关键信息的 GUI 智能体 |
| 2026-10 | 🔥 [OpenGCUA](https://arxiv.org/abs/2610.22200) `arXiv:2610.22200` |  | 通用计算机使用智能体的完全开放技术栈：从 GUI 扩展到命令行与 API |
| 2026-10 | 🔥 [MAI-Gym](https://arxiv.org/abs/2610.20939) `arXiv:2610.20939` |  | 一体化的 GUI 智能体评测与在线 RL 训练平台，覆盖手机与电脑 |
| 2026-10 | 🔥 [GroundGUI](https://arxiv.org/abs/2610.19100) `arXiv:2610.19100` |  | 高质量数据与蒸馏到 RL 的 GUI 定位 |
| 2026-10 | 🔥 [GUI 智能体记忆模态选择](https://arxiv.org/abs/2610.17928) `arXiv:2610.17928` |  | 学习用哪种模态（截图、文本、结构）记住 GUI 经验 |
| 2026-10 | 🔥 [A Decade in a Day](https://arxiv.org/abs/2610.15765) `arXiv:2610.15765` |  | 从软件演化史中合成面向智能体的原生接口 |
| 2026-10 | 🔥 [潜空间 GUI 世界动作模型](https://arxiv.org/abs/2610.14228) `arXiv:2610.14228` |  | 在潜空间而非像素空间预测 GUI 状态转移 |
| 2026-10 | 🔥 [Work2Skill](https://arxiv.org/abs/2610.12984) `arXiv:2610.12984` |  | 捕获人类 GUI 操作经验，归纳为可复用的智能体技能 |
| 2026-10 | 🔥 [OpenMobile-2](https://arxiv.org/abs/2610.14157) `arXiv:2610.14157` |  | 接近前沿的开放手机智能体：可扩展环境与应用原生工具 |
| 2026-10 | 🔥 [世界模型做 GUI 验证器](https://arxiv.org/abs/2610.11942) `arXiv:2610.11942` |  | 用世界模型判断"这一步的屏幕转移是否合理"，为 GUI 智能体提供过程验证 |
| 2026-10 | 🔥 [SoftwareGym](https://arxiv.org/abs/2610.12611) `arXiv:2610.12611` |  | 合成多样软件环境以扩展计算机使用智能体的训练 |
| 2026-10 | 🔥 [WebFovea](https://arxiv.org/abs/2610.03036) `arXiv:2610.03036` |  | 视觉网页智能体：模型判断对、点击却错的问题与可靠往返 |
| 2026-10 | 🔥 [GUI-HARVEST](https://arxiv.org/abs/2610.00948) `arXiv:2610.00948` |  | 证据驱动的 harness 演化让 GUI 智能体自我改进 |
| 2026-10 | 🔥 [DeskForge](https://arxiv.org/abs/2610.02320) `arXiv:2610.02320` |  | 桌面环境的稠密监督，改进计算机使用智能体的定位 |
| 2026-10 | 🔥 [AutoGUIWorld](https://arxiv.org/abs/2610.01215) `arXiv:2610.01215` |  | 用图像生成器充当 GUI 智能体的视觉世界模型 |
| 2026-09 | 🔥 [HybridCUA](https://arxiv.org/abs/2609.38008) `arXiv:2609.38008` |  | 学习编排 GUI 与 CLI 的计算机使用智能体 |
| 2026-09 | 🔥 [经验激活](https://arxiv.org/abs/2609.33304) `arXiv:2609.33304` |  | 个人 GUI 智能体：相关不等于可用的经验激活 |
| 2026-09 | 🔥 [RecreationWorld](https://arxiv.org/abs/2609.22000) `arXiv:2609.22000` |  | 可扩展、可验证的混合计算机使用环境（GUI + 代码） |
| 2026-09 | 🔥 [MintAct](https://arxiv.org/abs/2609.22083) `arXiv:2609.22083` |  | 统一 UI 定位、多步操作的数字环境视觉智能体 |
| 2026-09 | 🔥 [Reflect, Revise, Reuse](https://arxiv.org/abs/2609.17653) `arXiv:2609.17653` |  | GUI 智能体免训练的技能演化 |
| 2026-09 | 🔥 [LLaDA-UI](https://arxiv.org/abs/2609.13287) `arXiv:2609.13287` |  | 把块状扩散解码带入视觉语言 GUI 智能体 |
| 2026-08 | 🔥 [UI-Mate](https://arxiv.org/abs/2608.15930) `arXiv:2608.15930` |  | 借助上下文示范的开放权重 GUI 基础智能体 |
| 2026-08 | 🔥 [Activity Frames](https://arxiv.org/abs/2608.05784) `arXiv:2608.05784` |  | 把屏幕活动确定性地编译为可回放的例程，减少重复推理 |
| 2026-07 | 🔥 [Qwen-UI-Agent](https://arxiv.org/abs/2607.28227) `arXiv:2607.28227` | Alibaba Qwen | 面向真实场景的新一代 GUI 基础智能体技术报告 |
| 2026-07 | 🔥 [StateAct](https://arxiv.org/abs/2607.22798) `arXiv:2607.22798` |  | 先看程序状态再看像素：长程计算机使用智能体 |
| 2026-07 | 🔥 [KnowAct-GUIClaw](https://arxiv.org/abs/2607.12625) `arXiv:2607.12625` |  | 基于 OpenClaw 的个人 GUI 助手：自演化记忆与技能 |
| 2026-07 | 🔥 [UI-MOPD](https://arxiv.org/abs/2607.04425) `arXiv:2607.04425` |  | 多平台在策略蒸馏，持续学习 GUI 智能体 |
| 2026-06 | 🔥 [Xiaomi-GUI-0](https://arxiv.org/abs/2606.31410) `arXiv:2606.31410` | Xiaomi | GUI 智能体技术报告 |
| 2026-06 | 🔥 [训练开放的手机使用智能体](https://arxiv.org/abs/2606.23049) `arXiv:2606.23049` |  | 为手机使用训练开放模型 |
| 2026-06 | 🔥 [GUI vs. CLI](https://arxiv.org/abs/2606.24551) `arXiv:2606.24551` |  | 纯屏幕与技能中介两类计算机使用智能体的执行瓶颈 |
| 2026-06 | 🔥 [MobileForge](https://arxiv.org/abs/2606.19930) `arXiv:2606.19930` |  | 免标注的手机 GUI 智能体适配 |
| 2026-06 | 🔥 [MemGUI-Agent](https://arxiv.org/abs/2606.19926) `arXiv:2606.19926` |  | 主动管理上下文的端到端长程手机 GUI 智能体 |
| 2026-06 | 🔥 [OpenWebRL](https://arxiv.org/abs/2606.02031) `arXiv:2606.02031` |  | 揭示视觉网页智能体在线多轮 RL 的关键配方 |
| 2026-05 | 🔥 [向弱点学习](https://arxiv.org/abs/2605.28775) `arXiv:2605.28775` |  | 小型计算机使用智能体的自动领域专精 |
| 2026-05 | 🔥 [MobileGym](https://arxiv.org/abs/2605.26114) `arXiv:2605.26114` |  | 浏览器托管、可验证、高并行的手机 GUI 智能体仿真平台 |
| 2026-05 | 🔥 [CUA-Gym](https://arxiv.org/abs/2605.25624) `arXiv:2605.25624` |  | 扩展计算机使用智能体的可验证训练环境与任务 |
| 2026-05 | 🔥 [OpenComputer](https://arxiv.org/abs/2605.19769) `arXiv:2605.19769` |  | 为计算机使用智能体构建可验证的软件世界 |
| 2026-05 | 🔥 [Video2GUI](https://arxiv.org/abs/2605.14747) `arXiv:2605.14747` |  | 从视频合成大规模交互轨迹，用于 GUI 智能体预训练 |
| 2026-05 | 🔥 [ToolCUA](https://arxiv.org/abs/2605.12481) `arXiv:2605.12481` |  | 计算机使用智能体在 GUI 动作与工具调用之间的最优路径编排 |
| 2026-04 | ◆ [MolmoWeb](https://arxiv.org/abs/2604.08516) `arXiv:2604.08516` | Ai2 | 开放视觉网页智能体与开放数据 |
| 2026-03 | ◆ [UI-Voyager](https://arxiv.org/abs/2603.24533) `arXiv:2603.24533` |  | 从失败经验中学习的自演化 GUI 智能体 |
| 2026-03 | ◆ [CUA-Suite](https://arxiv.org/abs/2603.24440) `arXiv:2603.24440` |  | 大规模人工标注的计算机使用视频示范 |
| 2026-03 | ◆ [基于视频的奖励模型](https://arxiv.org/abs/2603.10178) `arXiv:2603.10178` |  | 为计算机使用智能体训练基于录屏的奖励模型 |
| 2026-02 | ◆ [Mobile-Agent-v3.5（GUI-Owl-1.5）](https://arxiv.org/abs/2602.16855) `arXiv:2602.16855` | Alibaba | 多平台基础 GUI 智能体 |
| 2026-02 | ◆ [Code2World](https://arxiv.org/abs/2602.09856) `arXiv:2602.09856` |  | 以可渲染代码生成构建 GUI 世界模型 |
| 2026-01 | ◆ [EvoCUA](https://arxiv.org/abs/2601.15876) `arXiv:2601.15876` |  | 从可扩展的合成经验中演化计算机使用智能体 |
| 2025-12 | ◆ [MAI-UI](https://arxiv.org/abs/2512.22047) `arXiv:2512.22047` |  | 面向真实世界的基础 GUI 智能体技术报告 |
| 2025-12 | ◆ [Step-GUI](https://arxiv.org/abs/2512.15431) `arXiv:2512.15431` | StepFun | GUI 智能体技术报告 |
| 2025-11 | ◆ [基于人类示范的定位](https://arxiv.org/abs/2511.07332) `arXiv:2511.07332` |  | 以人类示范为依据训练计算机使用智能体的定位 |
| 2025-10 | 🆕 [ChatGPT Atlas 智能体玩网页游戏](https://arxiv.org/abs/2510.26298) `arXiv:2510.26298` |  | 考察 ChatGPT Atlas 浏览器智能体的网页交互前沿 |
| 2025-10 | 🆕 [Surfer 2](https://arxiv.org/abs/2510.19949) `arXiv:2510.19949` | H Company | 新一代跨平台计算机使用智能体 |
| 2025-09 | 🆕 [ScaleCUA](https://arxiv.org/abs/2509.15221) `arXiv:2509.15221` |  | 跨平台数据扩展开源计算机使用智能体 |
| 2025-09 | 🆕 [UI-S1](https://arxiv.org/abs/2509.11543) `arXiv:2509.11543` |  | 半在线强化学习推进 GUI 自动化 |
| 2025-09 | 🆕 [UI-TARS-2](https://arxiv.org/abs/2509.02544) `arXiv:2509.02544` | ByteDance | 多轮强化学习推进 GUI 智能体 |
| 2025-08 | 🆕 [CODA](https://arxiv.org/abs/2508.20096) `arXiv:2508.20096` |  | "大脑 + 小脑"双脑解耦的计算机使用智能体 |
| 2025-08 | 🆕 [Mobile-Agent-v3（GUI-Owl）](https://arxiv.org/abs/2508.15144) `arXiv:2508.15144` | Alibaba | GUI 自动化的基础智能体 |
| 2025-08 | 🆕 [UI-Venus](https://arxiv.org/abs/2508.10833) `arXiv:2508.10833` | Ant Group | 以强化微调构建高性能 UI 智能体 |
| 2025-08 | 🆕 [OpenCUA](https://arxiv.org/abs/2508.09123) `arXiv:2508.09123` | HKU | 计算机使用智能体的开放基础：数据、模型与工具 |
| 2025-08 | 🆕 [SEAgent](https://arxiv.org/abs/2508.04700) `arXiv:2508.04700` |  | 从经验中自主学习的自演化计算机使用智能体 |
| 2025-07 | 🆕 [Phi-Ground](https://arxiv.org/abs/2507.23779) `arXiv:2507.23779` | Microsoft | 推进计算机使用智能体的 GUI 定位感知 |
| 2025-07 | 🆕 [GUI-G²](https://arxiv.org/abs/2507.15846) `arXiv:2507.15846` |  | 用高斯奖励建模 GUI 定位 |
| 2025-07 | 🆕 [GTA1](https://arxiv.org/abs/2507.05791) `arXiv:2507.05791` | Salesforce | GUI 测试时扩展智能体 |
| 2025-07 | 🆕 [如何训练网页智能体](https://arxiv.org/abs/2507.04103) `arXiv:2507.04103` |  | 网页智能体训练的统计诊断 |
| 2025-06 | 🆕 [GUI-Actor](https://arxiv.org/abs/2506.03143) `arXiv:2506.03143` | Microsoft | 无坐标的 GUI 视觉定位 |
| 2025-05 | 🆕 [ZeroGUI](https://arxiv.org/abs/2505.23762) `arXiv:2505.23762` |  | 零人工成本的在线 GUI 学习 |
| 2025-05 | 🆕 [UI-Genie](https://arxiv.org/abs/2505.21496) `arXiv:2505.21496` |  | 迭代提升手机 GUI 智能体的自我改进框架 |
| 2025-05 | 🆕 [Web-Shepherd](https://arxiv.org/abs/2505.15277) `arXiv:2505.15277` |  | 为网页智能体训练过程奖励模型 |
| 2025-05 | 🆕 [高效计算机使用智能体训练](https://arxiv.org/abs/2505.13909) `arXiv:2505.13909` |  | 少量高质量轨迹高效训练计算机使用智能体 |
| 2025-05 | 🆕 [扩展计算机使用定位](https://arxiv.org/abs/2505.13227) `arXiv:2505.13227` |  | 通过 UI 分解与合成扩展 GUI 定位 |
| 2025-04 | 🆕 [UFO2](https://arxiv.org/abs/2504.14603) `arXiv:2504.14603` | Microsoft | 桌面智能体操作系统 |
| 2025-03 | 🆕 [UI-R1](https://arxiv.org/abs/2503.21620) `arXiv:2503.21620` |  | 用 RL 增强 GUI 智能体动作预测 |
| 2025-02 | 🆕 [PC-Agent](https://arxiv.org/abs/2502.14282) `arXiv:2502.14282` | Alibaba | 面向 PC 复杂任务的分层多智能体协作 |
| 2025-01 | ★ 🆕 [UI-TARS](https://arxiv.org/abs/2501.12326) `arXiv:2501.12326` | ByteDance | 原生 GUI 智能体模型：只用截图完成感知、推理与操作 |
| 2024-12 | 🆕 [OS-Genesis](https://arxiv.org/abs/2412.19723) `arXiv:2412.19723` |  | 反向任务合成自动构建 GUI 轨迹 |
| 2024-12 | 🆕 [GUI 智能体综述](https://arxiv.org/abs/2412.13501) `arXiv:2412.13501` |  | GUI 智能体综述 |
| 2024-12 | 🆕 [AgentTrek](https://arxiv.org/abs/2412.09605) `arXiv:2412.09605` |  | 借网页教程引导回放合成智能体轨迹 |
| 2024-12 | 🆕 [Aguvis](https://arxiv.org/abs/2412.04454) `arXiv:2412.04454` |  | 纯视觉的统一 GUI 智能体 |
| 2024-11 | 🆕 [LLM 驱动的 GUI 智能体综述](https://arxiv.org/abs/2411.18279) `arXiv:2411.18279` | Microsoft | 以大模型为大脑的 GUI 智能体综述 |
| 2024-11 | 🆕 [ShowUI](https://arxiv.org/abs/2411.17465) `arXiv:2411.17465` |  | 面向 GUI 视觉智能体的视觉—语言—动作模型 |
| 2024-11 | 🆕 [Claude 3.5 Computer Use 案例研究](https://arxiv.org/abs/2411.10323) `arXiv:2411.10323` |  | GUI 智能体的黎明：对首个公开计算机使用模型的初步研究 |
| 2024-11 | 🆕 [WebRL](https://arxiv.org/abs/2411.02337) `arXiv:2411.02337` | Zhipu·Tsinghua | 自演化在线课程强化学习训练网页智能体 |
| 2024-10 | [AndroidLab](https://arxiv.org/abs/2410.24024) `arXiv:2410.24024` |  | 安卓自主智能体的训练与系统评测 |
| 2024-10 | [OS-ATLAS](https://arxiv.org/abs/2410.23218) `arXiv:2410.23218` |  | 通用 GUI 智能体的基础动作模型 |
| 2024-10 | [带世界模型的网页智能体](https://arxiv.org/abs/2410.13232) `arXiv:2410.13232` |  | 学习并利用环境动力学做网页导航 |
| 2024-10 | [Agent S](https://arxiv.org/abs/2410.08164) `arXiv:2410.08164` | Simular | 像人一样使用计算机的开放智能体框架 |
| 2024-08 | [Agent Q](https://arxiv.org/abs/2408.07199) `arXiv:2408.07199` | MultiOn | 搜索 + 自我批评 + DPO 训练网页智能体 |
| 2024-06 | [DigiRL](https://arxiv.org/abs/2406.11896) `arXiv:2406.11896` | UC Berkeley | 用自主强化学习训练野外设备控制智能体 |
| 2024-02 | [OS-Copilot](https://arxiv.org/abs/2402.07456) `arXiv:2402.07456` | Shanghai AI Lab | 带自我改进的通用计算机智能体 |
| 2024-01 | [Mobile-Agent](https://arxiv.org/abs/2401.16158) `arXiv:2401.16158` | Alibaba | 基于视觉感知的自主手机智能体 |
| 2024-01 | [WebVoyager](https://arxiv.org/abs/2401.13919) `arXiv:2401.13919` | Zhejiang·Tencent | 基于多模态大模型的端到端网页智能体 |
| 2024-01 | [SeeAct](https://arxiv.org/abs/2401.01614) `arXiv:2401.01614` | OSU | GPT-4V 在恰当定位下即是通用网页智能体 |
| 2023-12 | [AppAgent](https://arxiv.org/abs/2312.13771) `arXiv:2312.13771` | Tencent | 像用户一样操作手机应用的多模态智能体 |
| 2023-12 | [CogAgent](https://arxiv.org/abs/2312.08914) `arXiv:2312.08914` | Tsinghua·Zhipu | 面向 GUI 智能体的视觉语言模型 |

### 深度研究与搜索（72）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Épi](https://arxiv.org/abs/2610.24944) `arXiv:2610.24944` |  | 跨会话持续推进研究的自我改进研究智能体 |
| 2026-10 | 🔥 [JevGPT](https://arxiv.org/abs/2610.18272) `arXiv:2610.18272` |  | 自主研究中的快思与慢想：有限预算下获取并推理证据 |
| 2026-10 | 🔥 [OneSearch-VL](https://arxiv.org/abs/2610.12419) `arXiv:2610.12419` |  | 统一的图像与视频深度研究智能体 |
| 2026-10 | 🔥 [无状态语言智能体](https://arxiv.org/abs/2610.07625) `arXiv:2610.07625` |  | 用无状态设计扩展长程自动研究：每步只读必要状态 |
| 2026-10 | 🔥 [智能体检索的性能与成本](https://arxiv.org/abs/2610.05750) `arXiv:2610.05750` |  | 复杂任务中智能体式检索的效果与开销 |
| 2026-10 | 🔥 [SearchJev](https://arxiv.org/abs/2610.05107) `arXiv:2610.05107` |  | 为搜索智能体做快速、校准的"系统 1"决策模型 |
| 2026-10 | 🔥 [选择式结构化推理](https://arxiv.org/abs/2610.01892) `arXiv:2610.01892` |  | 小模型多模态搜索智能体：用选择代替自由文本推理以提效 |
| 2026-09 | 🔥 [False Frontiers](https://arxiv.org/abs/2609.39102) `arXiv:2609.39102` |  | 自演化搜索智能体中出题者与解题者"共谋作弊"的诊断与缓解 |
| 2026-09 | 🔥 [Follow the Entities](https://arxiv.org/abs/2609.37226) `arXiv:2609.37226` |  | 为智能体搜索构建语料实体地图 |
| 2026-09 | 🔥 [IterSynth](https://arxiv.org/abs/2609.29444) `arXiv:2609.29444` |  | 角色解耦的迭代综合式深度搜索智能体 |
| 2026-09 | 🔥 [自演化搜索索引](https://arxiv.org/abs/2609.19656) `arXiv:2609.19656` |  | 为智能体检索自演化文档索引 |
| 2026-08 | 🔥 [ABSeeker](https://arxiv.org/abs/2608.05102) `arXiv:2608.05102` |  | 以答案回溯的信用分配训练长程搜索智能体 |
| 2026-08 | 🔥 [Video-DeepResearch](https://arxiv.org/abs/2608.03979) `arXiv:2608.03979` |  | 面向视频的下一代多模态深度研究智能体 |
| 2026-07 | 🔥 [相关性的新角色](https://arxiv.org/abs/2607.24223) `arXiv:2607.24223` |  | 用相关性引导智能体搜索中的语料交互 |
| 2026-07 | 🔥 [多智能体协议蒸馏](https://arxiv.org/abs/2607.24280) `arXiv:2607.24280` |  | 把闭源模型的智能体搜索能力蒸馏到开源模型 |
| 2026-07 | 🔥 [AREX](https://arxiv.org/abs/2607.21461) `arXiv:2607.21461` |  | 递归自我改进的深度研究智能体 |
| 2026-07 | 🔥 [SearchOS-V1](https://arxiv.org/abs/2607.15257) `arXiv:2607.15257` |  | 稳健的开放域信息搜索智能体协作 |
| 2026-07 | 🔥 [DeepSearch-World](https://arxiv.org/abs/2607.07820) `arXiv:2607.07820` |  | 在可验证环境中对深度搜索智能体做自蒸馏 |
| 2026-07 | 🔥 [工作流归纳的科学文献搜索](https://arxiv.org/abs/2607.00597) `arXiv:2607.00597` |  | 多轮智能体式科学文献检索 |
| 2026-06 | 🔥 [FORT-Searcher](https://arxiv.org/abs/2606.12087) `arXiv:2606.12087` |  | 合成抗捷径的搜索任务训练深度搜索智能体 |
| 2026-06 | 🔥 [SearchSwarm](https://arxiv.org/abs/2606.09730) `arXiv:2606.09730` |  | 长程深度研究中的委派智能 |
| 2026-06 | 🔥 [Harness-1](https://arxiv.org/abs/2606.02373) `arXiv:2606.02373` |  | 用"外化状态"的 harness 训练搜索智能体，而非在越来越长的对话上训练 |
| 2026-05 | 🔥 [屏蔽陈旧观测](https://arxiv.org/abs/2606.00408) `arXiv:2606.00408` |  | 长程搜索智能体屏蔽旧观测有效，直到失效：机制与边界 |
| 2026-05 | 🔥 [SAAS](https://arxiv.org/abs/2605.29796) `arXiv:2605.29796` |  | 自我感知的 RL 缓解智能体搜索中的过度搜索 |
| 2026-05 | 🔥 [GrepSeek](https://arxiv.org/abs/2605.29307) `arXiv:2605.29307` |  | 训练直接与语料交互（类 grep）的搜索智能体 |
| 2026-05 | 🔥 [QUEST](https://arxiv.org/abs/2605.24218) `arXiv:2605.24218` |  | 用完全合成的任务训练前沿深度研究智能体 |
| 2026-05 | 🔥 [HyperEyes](https://arxiv.org/abs/2605.07177) `arXiv:2605.07177` |  | 并行多模态搜索智能体的效率感知 RL |
| 2026-05 | 🔥 [OpenSearch-VL](https://arxiv.org/abs/2605.05185) `arXiv:2605.05185` |  | 前沿多模态搜索智能体的开放配方 |
| 2026-05 | 🔥 [重新思考推理密集检索](https://arxiv.org/abs/2605.04018) `arXiv:2605.04018` |  | 评测并改进智能体搜索系统中的检索器 |
| 2026-05 | 🔥 [OpenSeeker-v2](https://arxiv.org/abs/2605.04036) `arXiv:2605.04036` |  | 用信息量大、难度高的轨迹推高搜索智能体上限 |
| 2026-05 | 🔥 [直接语料交互检索](https://arxiv.org/abs/2605.05242) `arXiv:2605.05242` |  | 智能体式搜索不必只用语义相似度：让智能体直接与语料交互 |
| 2026-04 | ◆ [Web2BigTable](https://arxiv.org/abs/2604.27221) `arXiv:2604.27221` |  | 互联网规模信息搜索与抽取的双层多智能体系统 |
| 2026-04 | ◆ [DR-Venus](https://arxiv.org/abs/2604.19859) `arXiv:2604.19859` |  | 只用 1 万条开放数据的端侧深度研究智能体 |
| 2026-04 | ◆ [记忆智能体](https://arxiv.org/abs/2604.04503) `arXiv:2604.04503` |  | 深度研究智能体的记忆 |
| 2026-03 | ◆ [从智能体轨迹学习检索](https://arxiv.org/abs/2604.04949) `arXiv:2604.04949` |  | 为智能体（而非人）训练检索系统 |
| 2026-03 | ◆ [OpenResearcher](https://arxiv.org/abs/2603.20278) `arXiv:2603.20278` |  | 完全开放的长程深度研究轨迹合成流水线 |
| 2026-03 | ◆ [OpenSeeker](https://arxiv.org/abs/2603.15594) `arXiv:2603.15594` |  | 完全开源训练数据，普及前沿搜索智能体 |
| 2026-03 | ◆ [MiroThinker-1.7 / H1](https://arxiv.org/abs/2603.15726) `arXiv:2603.15726` | MiroMind | 通过验证迈向重型研究智能体 |
| 2026-03 | ◆ [策略导航还是随机搜索](https://arxiv.org/abs/2603.12180) `arXiv:2603.12180` |  | 智能体与人类如何在文档集合上推理 |
| 2026-02 | ◆ [WideSeek-R1](https://arxiv.org/abs/2602.04634) `arXiv:2602.04634` |  | 用多智能体 RL 探索广度信息搜索的"宽度扩展" |
| 2026-02 | ◆ [FS-Researcher](https://arxiv.org/abs/2602.01566) `arXiv:2602.01566` |  | 基于文件系统的智能体做长程研究的测试时扩展 |
| 2026-01 | ◆ [引用感知评分细则的深度搜索 RL](https://arxiv.org/abs/2601.06021) `arXiv:2601.06021` |  | 用引用感知的评分细则稳健训练深度搜索智能体 |
| 2025-11 | ◆ [深度研究系统综述](https://arxiv.org/abs/2512.02038) `arXiv:2512.02038` |  | 深度研究智能体的系统综述 |
| 2025-11 | ◆ [DR Tulu](https://arxiv.org/abs/2511.19399) `arXiv:2511.19399` | Ai2 | 用演化的评分细则做深度研究 RL |
| 2025-11 | ◆ [MiroThinker](https://arxiv.org/abs/2511.11793) `arXiv:2511.11793` | MiroMind | 从模型、上下文与交互深度三方面推高开源研究智能体 |
| 2025-11 | ◆ [IterResearch](https://arxiv.org/abs/2511.07327) `arXiv:2511.07327` |  | 以马尔可夫式状态重建重新设计长程研究智能体 |
| 2025-10 | ★ 🆕 [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) `arXiv:2510.24701` | Alibaba Tongyi | 开源深度研究智能体模型技术报告：智能体中训练 + 后训练全流程 |
| 2025-10 | 🆕 [AgentFold](https://arxiv.org/abs/2510.24699) `arXiv:2510.24699` | Alibaba Tongyi | 主动管理上下文的长程网页智能体 |
| 2025-09 | 🆕 [WebWeaver](https://arxiv.org/abs/2509.13312) `arXiv:2509.13312` | Alibaba Tongyi | 用动态大纲组织网页规模证据做开放式深度研究 |
| 2025-09 | 🆕 [WebSailor-V2](https://arxiv.org/abs/2509.13305) `arXiv:2509.13305` | Alibaba Tongyi | 用合成数据与可扩展 RL 缩小与闭源智能体的差距 |
| 2025-09 | 🆕 [WebResearcher](https://arxiv.org/abs/2509.13309) `arXiv:2509.13309` | Alibaba Tongyi | 长程智能体的无界推理：迭代式研究范式 |
| 2025-09 | 🆕 [ReSum](https://arxiv.org/abs/2509.13313) `arXiv:2509.13313` | Alibaba Tongyi | 用上下文摘要解锁长程搜索 |
| 2025-09 | 🆕 [深度研究系统的 RL 基础](https://arxiv.org/abs/2509.06733) `arXiv:2509.06733` |  | 深度研究系统的强化学习综述 |
| 2025-09 | 🆕 [WebExplorer](https://arxiv.org/abs/2509.06501) `arXiv:2509.06501` |  | 探索与演化训练长程网页智能体 |
| 2025-08 | 🆕 [深度研究的开放数据合成](https://arxiv.org/abs/2509.00375) `arXiv:2509.00375` |  | 为深度研究合成开放训练数据 |
| 2025-08 | 🆕 [超越十轮](https://arxiv.org/abs/2508.07976) `arXiv:2508.07976` |  | 大规模异步 RL 解锁长程智能体搜索 |
| 2025-08 | 🆕 [WebWatcher](https://arxiv.org/abs/2508.05748) `arXiv:2508.05748` | Alibaba | 视觉语言深度研究智能体 |
| 2025-08 | 🆕 [VeriWeb](https://arxiv.org/abs/2508.04026) `arXiv:2508.04026` |  | 可验证的长链网页信息检索基准（原名 VeriGUI） |
| 2025-08 | 🆕 [Cognitive Kernel-Pro](https://arxiv.org/abs/2508.00414) `arXiv:2508.00414` | Tencent | 深度研究智能体框架与智能体基础模型训练 |
| 2025-07 | 🆕 [WebShaper](https://arxiv.org/abs/2507.15061) `arXiv:2507.15061` | Alibaba Tongyi | 通过信息搜寻形式化进行智能体式数据合成 |
| 2025-07 | 🆕 [智能体式 RAG 综述](https://arxiv.org/abs/2507.09477) `arXiv:2507.09477` |  | RAG 与推理结合的系统综述 |
| 2025-07 | 🆕 [WebSailor](https://arxiv.org/abs/2507.02592) `arXiv:2507.02592` | Alibaba Tongyi | 面向超人推理的网页智能体 |
| 2025-05 | 🆕 [WebDancer](https://arxiv.org/abs/2505.22648) `arXiv:2505.22648` | Alibaba Tongyi | 走向自主信息搜寻 |
| 2025-05 | 🆕 [ZeroSearch](https://arxiv.org/abs/2505.04588) `arXiv:2505.04588` | Alibaba | 不调用真实搜索引擎也能训练搜索能力（用 LLM 模拟搜索） |
| 2025-04 | 🆕 [WebThinker](https://arxiv.org/abs/2504.21776) `arXiv:2504.21776` |  | 赋予大推理模型深度研究能力 |
| 2025-04 | 🆕 [DeepResearcher](https://arxiv.org/abs/2504.03160) `arXiv:2504.03160` | SJTU | 在真实网络环境中用 RL 扩展深度研究 |
| 2025-03 | 🆕 [Open Deep Search](https://arxiv.org/abs/2503.20201) `arXiv:2503.20201` |  | 用开源推理智能体普及搜索 |
| 2025-03 | ★ 🆕 [Search-R1](https://arxiv.org/abs/2503.09516) `arXiv:2503.09516` | UIUC | 用强化学习训练模型边推理边调用搜索引擎 |
| 2025-03 | 🆕 [R1-Searcher](https://arxiv.org/abs/2503.05592) `arXiv:2503.05592` | Renmin | 用 RL 激发 LLM 的搜索能力 |
| 2025-01 | 🆕 [PaSa](https://arxiv.org/abs/2501.10120) `arXiv:2501.10120` | ByteDance | 综合学术论文搜索智能体 |
| 2025-01 | 🆕 [Search-o1](https://arxiv.org/abs/2501.05366) `arXiv:2501.05366` |  | 搜索增强的大推理模型 |
| 2024-02 | [STORM](https://arxiv.org/abs/2402.14207) `arXiv:2402.14207` | Stanford | 从零写维基百科式长文：多视角提问与大纲生成 |

### 科研与数据智能体（58）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Build, Don't Play](https://arxiv.org/abs/2610.33845) `arXiv:2610.33845` |  | 自我改进的研究智能体从原始数据开发出有竞争力的炉石酒馆战棋 AI |
| 2026-10 | 🔥 [AI 研究团队发现线性注意力架构](https://arxiv.org/abs/2610.28887) `arXiv:2610.28887` |  | 组建 AI 研究团队自主发现线性注意力架构 |
| 2026-10 | 🔥 [MLExpert](https://arxiv.org/abs/2610.22344) `arXiv:2610.22344` |  | 面向自主机器学习工程的通用智能体模型 |
| 2026-10 | 🔥 [AutoSFT](https://arxiv.org/abs/2610.22194) `arXiv:2610.22194` |  | 自动做 SFT 数据研究的智能体系统 |
| 2026-10 | 🔥 [Athanor](https://arxiv.org/abs/2610.21055) `arXiv:2610.21055` |  | 面向自主后训练研究的智能体原生系统 |
| 2026-10 | 🔥 [SwarmRSI](https://arxiv.org/abs/2610.20431) `arXiv:2610.20431` |  | 演化中的科学智能体群实现研究能力的递归自我改进 |
| 2026-09 | 🔥 [AIM](https://arxiv.org/abs/2609.38445) `arXiv:2609.38445` |  | 自动化研究中的智能体想法管理 |
| 2026-09 | 🔥 [TimeEvo](https://arxiv.org/abs/2609.27277) `arXiv:2609.27277` |  | 失败驱动、自我演化的时间序列智能体 |
| 2026-09 | 🔥 [AI 研究智能体的递归自我改进](https://arxiv.org/abs/2609.26457) `arXiv:2609.26457` |  | AI 研究智能体的递归自我改进 |
| 2026-09 | 🔥 [ScienceIDE](https://arxiv.org/abs/2609.19134) `arXiv:2609.19134` |  | 把科学代码库转化为智能体可学习的环境 |
| 2026-09 | 🔥 [ScienceBuddy](https://arxiv.org/abs/2609.17523) `arXiv:2609.17523` |  | "递归套递归"的自我改进交互式科学智能体 |
| 2026-09 | 🔥 [HypoEvolve](https://arxiv.org/abs/2609.15938) `arXiv:2609.15938` |  | 遗传算法让多智能体发现科学假设 |
| 2026-09 | 🔥 [EvoOntology](https://arxiv.org/abs/2609.15779) `arXiv:2609.15779` |  | 数据智能体的自演化本体层 |
| 2026-09 | 🔥 [SAEScientist-Bench](https://arxiv.org/abs/2609.09113) `arXiv:2609.09113` |  | 智能体能否自主做 SAE 可解释性研究 |
| 2026-08 | 🔥 [先学会评估再改进](https://arxiv.org/abs/2608.31076) `arXiv:2608.31076` |  | 为自动研究智能体自动归纳评分细则 |
| 2026-08 | 🔥 [Dr. Claw](https://arxiv.org/abs/2609.00365) `arXiv:2609.00365` |  | 基于命令行编码智能体（Claude Code、Gemini CLI 等）的 AI 科学家工作区 |
| 2026-08 | 🔥 [OmniScientist](https://arxiv.org/abs/2608.13558) `arXiv:2608.13558` |  | 全模态、全学科的 AI 科学家 |
| 2026-08 | 🔥 [用世界模型扩展自动研究智能体](https://arxiv.org/abs/2608.12564) `arXiv:2608.12564` |  | 为自动实证研究构建世界模型，降低真实实验成本 |
| 2026-08 | 🔥 [Spark-to-Paper](https://arxiv.org/abs/2608.11924) `arXiv:2608.11924` |  | 把从想法到完整论文做成可组合技能 |
| 2026-07 | 🔥 [Frontis-MA1](https://arxiv.org/abs/2607.28568) `arXiv:2607.28568` |  | 训练面向机器学习工程递归自我改进的 AI4AI 模型 |
| 2026-07 | 🔥 [AI 智能体能做开放式 AI 研究吗](https://arxiv.org/abs/2607.27191) `arXiv:2607.27191` |  | 两个案例研究的早期证据 |
| 2026-07 | 🔥 [ResearchStudio-Idea](https://arxiv.org/abs/2607.04439) `arXiv:2607.04439` |  | 基于会议结果证据的研究构思技能套件 |
| 2026-06 | 🔥 [AutoTrainess](https://arxiv.org/abs/2606.31551) `arXiv:2606.31551` |  | 教语言模型自主改进语言模型 |
| 2026-06 | 🔥 [EurekAgent](https://arxiv.org/abs/2606.13662) `arXiv:2606.13662` |  | 智能体环境工程是自主科学发现的关键 |
| 2026-06 | 🔥 [假设树精炼](https://arxiv.org/abs/2606.11926) `arXiv:2606.11926` |  | 以假设树精炼走向通用自主研究 |
| 2026-06 | 🔥 [MLEvolve](https://arxiv.org/abs/2606.06473) `arXiv:2606.06473` |  | 自我演化的机器学习算法发现框架 |
| 2026-05 | 🔥 [ResearchMath-14K](https://arxiv.org/abs/2605.28003) `arXiv:2605.28003` |  | 用智能体扩展研究级数学问题 |
| 2026-05 | 🔥 [ScientistOne](https://arxiv.org/abs/2605.26340) `arXiv:2605.26340` |  | 通过证据链迈向人类水平的自主研究 |
| 2026-05 | 🔥 [AutoResearchClaw](https://arxiv.org/abs/2605.20025) `arXiv:2605.20025` |  | 人机协作、自我强化的自主研究 |
| 2026-05 | 🔥 [OProver](https://arxiv.org/abs/2605.17283) `arXiv:2605.17283` |  | 智能体式形式化定理证明的统一框架 |
| 2026-05 | 🔥 [ARIS](https://arxiv.org/abs/2605.03042) `arXiv:2605.03042` |  | "睡觉时做研究"：对抗式多智能体协作的开源研究 harness |
| 2026-04 | ◆ [Intern-Atlas](https://arxiv.org/abs/2604.28158) `arXiv:2604.28158` | Shanghai AI Lab | 方法演化图谱：AI 科学家的研究基础设施 |
| 2026-04 | ◆ [长程 ML 研究工程](https://arxiv.org/abs/2604.13018) `arXiv:2604.13018` |  | 走向自主的长程机器学习研究工程 |
| 2026-03 | ◆ [医学 AI 科学家](https://arxiv.org/abs/2603.28589) `arXiv:2603.28589` |  | 走向医学领域的自主 AI 科学家 |
| 2026-03 | ◆ [LongCat-Flash-Prover](https://arxiv.org/abs/2603.21065) `arXiv:2603.21065` | Meituan | 560B 开源模型：智能体式工具集成 RL 做形式化推理 |
| 2026-02 | ◆ [走向自主数学研究](https://arxiv.org/abs/2602.10177) `arXiv:2602.10177` | Google DeepMind | 从竞赛级到研究级数学的自主智能体 |
| 2026-02 | ◆ [InternAgent-1.5](https://arxiv.org/abs/2602.08990) `arXiv:2602.08990` | Shanghai AI Lab | 面向长程自主科学发现的统一智能体框架 |
| 2026-02 | ◆ [MARS](https://arxiv.org/abs/2602.02660) `arXiv:2602.02660` |  | 带反思式搜索的模块化智能体，用于自动化 AI 研究 |
| 2026-01 | ◆ [PaperBanana](https://arxiv.org/abs/2601.23265) `arXiv:2601.23265` |  | 为 AI 科学家自动化学术插图 |
| 2026-01 | ◆ [材料科学中的智能体智能](https://arxiv.org/abs/2602.00169) `arXiv:2602.00169` |  | 材料科学智能体综述 |
| 2026-01 | ◆ [Paper2Rebuttal](https://arxiv.org/abs/2601.14171) `arXiv:2601.14171` |  | 透明的作者答辩辅助多智能体框架 |
| 2025-12 | ◆ [奥赛数学长程推理智能体](https://arxiv.org/abs/2512.10739) `arXiv:2512.10739` |  | 长程推理智能体解奥赛级数学 |
| 2025-12 | ◆ [PaperDebugger](https://arxiv.org/abs/2512.02589) `arXiv:2512.02589` |  | 编辑器内学术写作、审阅与修改的多智能体系统 |
| 2025-11 | ◆ [好的 AI 研究智能体需要什么](https://arxiv.org/abs/2511.15593) `arXiv:2511.15593` | Meta | 想法多样性的作用 |
| 2025-11 | ◆ [MADD](https://arxiv.org/abs/2511.08217) `arXiv:2511.08217` |  | 多智能体药物发现 |
| 2025-10 | 🆕 [数据智能体综述](https://arxiv.org/abs/2510.23587) `arXiv:2510.23587` |  | 数据智能体：新兴范式还是过度炒作 |
| 2025-10 | 🆕 [DeepAnalyze](https://arxiv.org/abs/2510.16872) `arXiv:2510.16872` |  | 自主数据科学的智能体模型 |
| 2025-09 | 🆕 [Paper2Agent](https://arxiv.org/abs/2509.06917) `arXiv:2509.06917` | Stanford | 把研究论文转化为可交互、可靠的 AI 智能体（MCP 服务器） |
| 2025-08 | 🆕 [从 AI for Science 到智能体科学](https://arxiv.org/abs/2508.14111) `arXiv:2508.14111` |  | 自主科学发现综述 |
| 2025-05 | 🆕 [NovelSeek](https://arxiv.org/abs/2505.16938) `arXiv:2505.16938` | Shanghai AI Lab | 从假设到验证的闭环科研智能体 |
| 2025-04 | 🆕 [The AI Scientist-v2](https://arxiv.org/abs/2504.08066) `arXiv:2504.08066` | Sakana AI | 智能体树搜索达到研讨会水平的自动科学发现 |
| 2025-02 | ★ 🆕 [AI co-scientist](https://arxiv.org/abs/2502.18864) `arXiv:2502.18864` | Google | 多智能体生成并评估科学假设 |
| 2025-01 | 🆕 [Agent Laboratory](https://arxiv.org/abs/2501.04227) `arXiv:2501.04227` |  | 用 LLM 智能体做研究助理 |
| 2024-10 | [AutoKaggle](https://arxiv.org/abs/2410.20424) `arXiv:2410.20424` |  | 自主参加数据科学竞赛的多智能体框架 |
| 2024-08 | [The AI Scientist](https://arxiv.org/abs/2408.06292) `arXiv:2408.06292` | Sakana AI | 完全自动化的开放式科研：从想法到论文 |
| 2024-04 | [ResearchAgent](https://arxiv.org/abs/2404.07738) `arXiv:2404.07738` | KAIST | 基于文献迭代生成研究想法 |
| 2024-02 | [Data Interpreter](https://arxiv.org/abs/2402.18679) `arXiv:2402.18679` | DeepWisdom | 数据科学智能体 |
| 2023-04 | [ChemCrow](https://arxiv.org/abs/2304.05376) `arXiv:2304.05376` | EPFL | 用化学工具增强 LLM |

### 行业与个人应用（21）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [X-Ragent](https://arxiv.org/abs/2610.16811) `arXiv:2610.16811` |  | 工具增强 RL 做有证据支撑的胸片推理 |
| 2026-10 | 🔥 [nanoMuse](https://arxiv.org/abs/2610.08699) `arXiv:2610.08699` |  | 面向用户所有设备的开源个人智能体 |
| 2026-10 | 🔥 [CADFather](https://arxiv.org/abs/2610.09127) `arXiv:2610.09127` |  | 协调工具使用自主重建可编辑 CAD 模型 |
| 2026-09 | 🔥 [从给定证据到收集证据](https://arxiv.org/abs/2609.39566) `arXiv:2609.39566` |  | 纵向医疗推理的智能体学习：通过工具 harness 主动收集证据 |
| 2026-09 | 🔥 [Qwen-Planner-Agent](https://arxiv.org/abs/2609.29892) `arXiv:2609.29892` | Alibaba Qwen | 面向真实手机规划智能体的闭环 AI-for-AI 框架 |
| 2026-09 | 🔥 [Designer-RSI](https://arxiv.org/abs/2609.22086) `arXiv:2609.22086` |  | 从用户流量中演化程序性记忆的平面设计智能体 |
| 2026-09 | 🔥 [BI-Agent](https://arxiv.org/abs/2609.20886) `arXiv:2609.20886` |  | 端到端商业智能自动化智能体与基准 |
| 2026-09 | 🔥 [EvolveTrade](https://arxiv.org/abs/2609.17632) `arXiv:2609.17632` |  | 经验驱动、自我演化的交易智能体 |
| 2026-09 | 🔥 [生产中的交易智能体](https://arxiv.org/abs/2609.05663) `arXiv:2609.05663` |  | 两个车队、六个月的自主交易智能体人口级测量记录 |
| 2026-08 | 🔥 [SemaPLC](https://arxiv.org/abs/2608.18565) `arXiv:2608.18565` |  | 以项目为依据、验证门控的 PLC 代码生成 harness |
| 2026-08 | 🔥 [TaoLive 数字人智能体](https://arxiv.org/abs/2608.15763) `arXiv:2608.15763` | Alibaba | 训练智能体与其 harness 一同演化：电商直播数字人 |
| 2026-08 | 🔥 [ComBodied Agents](https://arxiv.org/abs/2608.10915) `arXiv:2608.10915` |  | 以人为中心的智能体：软件智能体与家用设备协作照护 |
| 2026-07 | 🔥 [自主可审计的医学影像模型开发](https://arxiv.org/abs/2607.10522) `arXiv:2607.10522` |  | 智能体自动化医学影像机器学习工程 |
| 2026-06 | 🔥 [MemSlides](https://arxiv.org/abs/2606.17162) `arXiv:2606.17162` |  | 分层记忆驱动的个性化幻灯片生成智能体 |
| 2026-06 | 🔥 [Data Journalist Agent](https://arxiv.org/abs/2606.11176) `arXiv:2606.11176` |  | 把数据转化为可验证的多模态报道 |
| 2026-05 | 🔥 [Crafter](https://arxiv.org/abs/2605.30611) `arXiv:2605.30611` |  | 从多种输入生成可编辑科学图表的多智能体 harness |
| 2026-05 | 🔥 [Macaron-A2UI](https://arxiv.org/abs/2605.24830) `arXiv:2605.24830` |  | 个人智能体的生成式 UI 模型：用界面而非纯文本回应 |
| 2026-05 | 🔥 [Spreadsheet-RL](https://arxiv.org/abs/2605.22642) `arXiv:2605.22642` |  | 通过 RL 提升智能体处理真实电子表格任务的能力 |
| 2026-05 | 🔥 [从有限交互预测智能体决策](https://arxiv.org/abs/2605.12411) `arXiv:2605.12411` |  | 智能体用自然语言谈判与交易：预测对手决策 |
| 2026-04 | ◆ [FileGram](https://arxiv.org/abs/2604.04901) `arXiv:2604.04901` |  | 以文件系统行为轨迹为依据的智能体个性化 |
| 2026-02 | ◆ [LawThinker](https://arxiv.org/abs/2602.12056) `arXiv:2602.12056` |  | 动态环境中的法律深度研究智能体 |

### 多模态与语音（21）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [StreamVibe](https://arxiv.org/abs/2610.34662) `arXiv:2610.34662` |  | 面向不断变化用户意图的主动式视频智能体 |
| 2026-10 | 🔥 [Heartbeat Thinking](https://arxiv.org/abs/2610.26233) `arXiv:2610.26233` |  | 实时语音智能体边听边推测推理与工具执行，减少等待 |
| 2026-09 | 🔥 [Gander（全交互智能体）](https://arxiv.org/abs/2609.08977) `arXiv:2609.08977` |  | 统一全模态感知与实时交互的端到端智能体模型 |
| 2026-08 | 🔥 [ToolArtist](https://arxiv.org/abs/2608.04436) `arXiv:2608.04436` |  | 会用工具的统一多模态模型做智能体式图像生成 |
| 2026-07 | 🔥 [Beacon](https://arxiv.org/abs/2607.28595) `arXiv:2607.28595` |  | 知道何时、如何做智能体式视觉推理 |
| 2026-07 | 🔥 [Omni-Decision](https://arxiv.org/abs/2607.11433) `arXiv:2607.11433` |  | 全模态智能体的证据账本式规划 |
| 2026-06 | 🔥 [VisualClaw](https://arxiv.org/abs/2606.16295) `arXiv:2606.16295` |  | 面向物理世界的实时个性化智能体 |
| 2026-05 | 🔥 [ParaVT](https://arxiv.org/abs/2605.20342) `arXiv:2605.20342` |  | 智能体视频 RL 中的并行工具使用 |
| 2026-05 | 🔥 [AtlasVA](https://arxiv.org/abs/2605.17933) `arXiv:2605.17933` |  | 无教师 VLM 智能体的自演化视觉技能记忆 |
| 2026-04 | ◆ [Act Wisely](https://arxiv.org/abs/2604.08545) `arXiv:2604.08545` |  | 培养多模态模型的元认知式工具使用 |
| 2026-03 | ◆ [Gen-Searcher](https://arxiv.org/abs/2603.28767) `arXiv:2603.28767` |  | 为图像生成强化智能体式搜索 |
| 2026-03 | ◆ [GEMS](https://arxiv.org/abs/2603.28088) `arXiv:2603.28088` |  | 带记忆与技能的智能体原生多模态生成 |
| 2025-12 | ◆ [SenseNova-MARS](https://arxiv.org/abs/2512.24330) `arXiv:2512.24330` | SenseTime | 以强化学习增强多模态智能体推理与搜索 |
| 2025-12 | ◆ [LongVideoAgent](https://arxiv.org/abs/2512.20618) `arXiv:2512.20618` |  | 长视频的多智能体推理 |
| 2025-12 | ◆ [Skywork-R1V4](https://arxiv.org/abs/2512.02395) `arXiv:2512.02395` | Skywork | 交错图像思考与深度研究的多模态智能体 |
| 2025-11 | ◆ [LongVT](https://arxiv.org/abs/2511.20785) `arXiv:2511.20785` |  | 通过原生工具调用"用长视频思考" |
| 2025-11 | ◆ [DeepEyesV2](https://arxiv.org/abs/2511.05271) `arXiv:2511.05271` |  | 走向智能体式多模态模型 |
| 2025-08 | 🆕 [M3-Agent](https://arxiv.org/abs/2508.09736) `arXiv:2508.09736` | ByteDance | 带长期记忆的多模态智能体 |
| 2025-07 | 🆕 [PyVision](https://arxiv.org/abs/2507.07998) `arXiv:2507.07998` |  | 动态生成工具的智能体式视觉 |
| 2025-05 | 🆕 [视觉智能体强化微调](https://arxiv.org/abs/2505.14246) `arXiv:2505.14246` |  | 让视觉模型学会用工具（裁剪、搜索）思考 |
| 2025-02 | 🆕 [Magma](https://arxiv.org/abs/2502.13130) `arXiv:2502.13130` | Microsoft | 多模态 AI 智能体的基础模型 |

### 游戏与虚拟世界（7）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [长时运行游戏智能体的启发式学习](https://arxiv.org/abs/2610.12341) `arXiv:2610.12341` |  | 智能体能否在长期运行中学会更好的启发式并登顶 |
| 2026-09 | 🔥 [RSIGame](https://arxiv.org/abs/2609.39045) `arXiv:2609.39045` |  | 通过递归自我改进自主开发游戏 |
| 2026-07 | 🔥 [CAST](https://arxiv.org/abs/2607.25308) `arXiv:2607.25308` |  | 游戏求解器作为轮次级教师训练 LLM 智能体 |
| 2026-07 | 🔥 [EvolvingWorld](https://arxiv.org/abs/2607.17250) `arXiv:2607.17250` |  | 角色扮演智能体与世界模型协同演化的交互叙事框架 |
| 2026-01 | ◆ [NitroGen](https://arxiv.org/abs/2601.02427) `arXiv:2601.02427` | NVIDIA | 通用游戏智能体的开放视觉—动作基础模型 |
| 2025-11 | ◆ [Lumine](https://arxiv.org/abs/2511.08892) `arXiv:2511.08892` |  | 在 3D 开放世界中构建通用智能体的开放配方 |
| 2025-10 | 🆕 [Game-TARS](https://arxiv.org/abs/2510.23691) `arXiv:2510.23691` | ByteDance | 通用多模态游戏智能体的预训练基础模型 |

### 智能体强化学习（64）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [多尺度策略优化](https://arxiv.org/abs/2610.35013) `arXiv:2610.35013` |  | 长程 LLM 智能体的多尺度策略优化：在 token、轮次与回合多个尺度上分配信用 |
| 2026-10 | 🔥 [Harness Scaling](https://arxiv.org/abs/2610.29458) `arXiv:2610.29458` |  | 在多种 harness 上训练让智能体 RL 更可泛化 |
| 2026-10 | 🔥 [训练整个智能体而不只是模型](https://arxiv.org/abs/2610.28069) `arXiv:2610.28069` |  | 训练编码智能体时同时更新 harness，而非固定 harness |
| 2026-10 | 🔥 [树感知后验蒸馏](https://arxiv.org/abs/2610.23181) `arXiv:2610.23181` |  | 长程智能体 RL 中的树感知后验蒸馏 |
| 2026-10 | 🔥 [Mosaic](https://arxiv.org/abs/2610.22174) `arXiv:2610.22174` |  | 多会话的智能体强化学习：跨多个会话分配信用 |
| 2026-10 | 🔥 [Lean4AgenticRL](https://arxiv.org/abs/2610.22448) `arXiv:2610.22448` |  | 可验证的多层过程奖励推进软件工程智能体 RL |
| 2026-10 | 🔥 [长程 RL 需要 Muon](https://arxiv.org/abs/2610.20256) `arXiv:2610.20256` |  | 长程智能体 RL 中 Muon 优化器比 Adam 更稳 |
| 2026-10 | 🔥 [智能体 RL 自蒸馏的现实检验](https://arxiv.org/abs/2610.19789) `arXiv:2610.19789` |  | 统一条件重测各类智能体 RL 自蒸馏方法：多数增益不如宣称 |
| 2026-10 | 🔥 [WorkRover](https://arxiv.org/abs/2610.18067) `arXiv:2610.18067` |  | 通过递归演化环境扩展长程"协作办公"智能体的训练 |
| 2026-10 | 🔥 [Tev](https://arxiv.org/abs/2610.18868) `arXiv:2610.18868` |  | 用稠密进度奖励扩展长程智能体环境 |
| 2026-10 | 🔥 [ProAS](https://arxiv.org/abs/2610.18683) `arXiv:2610.18683` |  | 进度感知的优势塑形，用于多轮智能体 RL |
| 2026-10 | 🔥 [DeltaTTG](https://arxiv.org/abs/2610.17666) `arXiv:2610.17666` |  | 用"剩余 token 数"进度表训练多轮智能体 |
| 2026-10 | 🔥 [EAU](https://arxiv.org/abs/2610.16523) `arXiv:2610.16523` |  | 环境—智能体—用户协同演化，训练以用户为中心的智能体 |
| 2026-10 | 🔥 [NeMo-DCR](https://arxiv.org/abs/2610.08430) `arXiv:2610.08430` | NVIDIA | 万亿参数规模智能体 RL 中按位精确的增量压缩权重同步 |
| 2026-09 | 🔥 [PivotOPD](https://arxiv.org/abs/2609.40285) `arXiv:2609.40285` |  | 学会从多轮智能体的关键错误中恢复 |
| 2026-09 | 🔥 [PR-OPD](https://arxiv.org/abs/2609.36642) `arXiv:2609.36642` |  | 特权表征的在策略自蒸馏，用于智能体 RL |
| 2026-09 | 🔥 [QwenGyre](https://arxiv.org/abs/2609.33848) `arXiv:2609.33848` | Alibaba Qwen | 面向超长程智能体的弹性强化学习训练框架 |
| 2026-09 | 🔥 [组内智能体评分与优势重分配](https://arxiv.org/abs/2609.32577) `arXiv:2609.32577` |  | 代码智能体 RL 中超越二值测试奖励的评分 |
| 2026-09 | 🔥 [X-Tree](https://arxiv.org/abs/2609.32993) `arXiv:2609.32993` |  | 把可复用经验 token 化，提高智能体泛化效率 |
| 2026-09 | 🔥 [不要掩盖环境](https://arxiv.org/abs/2609.20715) `arXiv:2609.20715` |  | SFT 时对环境观测也施加监督，会改变智能体在 RL 中的探索 |
| 2026-09 | 🔥 [RetireOPD](https://arxiv.org/abs/2609.20784) `arXiv:2609.20784` |  | 智能体 RL 中自动"退休"的在策略蒸馏 |
| 2026-09 | 🔥 [DRACO](https://arxiv.org/abs/2609.04094) `arXiv:2609.04094` |  | 动态评分细则带来的细粒度信用分配，用于长程智能体训练 |
| 2026-08 | 🔥 [OPD 真的在蒸馏吗](https://arxiv.org/abs/2608.31046) `arXiv:2608.31046` |  | 从带噪教师到自我改进：在策略蒸馏的再审视 |
| 2026-08 | 🔥 [Agent-G²](https://arxiv.org/abs/2608.23318) `arXiv:2608.23318` |  | 高斯引导的提示式智能体 RL |
| 2026-08 | 🔥 [LEGO-RL](https://arxiv.org/abs/2608.17393) `arXiv:2608.17393` |  | 面向编码智能体的 harness 原生强化学习 |
| 2026-08 | 🔥 [Agentic ESOpt](https://arxiv.org/abs/2608.17310) `arXiv:2608.17310` |  | 低显存微调长程 LLM 智能体（进化策略） |
| 2026-08 | 🔥 [Agent Lightning v1.0](https://arxiv.org/abs/2608.17528) `arXiv:2608.17528` | Microsoft | 面向 harness 的智能体 RL 框架：训练与执行解耦 |
| 2026-08 | 🔥 [ClawGym II](https://arxiv.org/abs/2608.16798) `arXiv:2608.16798` |  | 对智能体 harness 做黑盒强化学习 |
| 2026-08 | 🔥 [EnvACE](https://arxiv.org/abs/2608.06197) `arXiv:2608.06197` |  | 通过"世界彩排"内化环境动力学的智能体 RL |
| 2026-08 | 🔥 [AgentOPSD](https://arxiv.org/abs/2608.05987) `arXiv:2608.05987` |  | 智能体 RL 的递归自蒸馏 |
| 2026-08 | 🔥 [PCSD](https://arxiv.org/abs/2608.01837) `arXiv:2608.01837` |  | 智能体 RL 自蒸馏中的持续一致性 |
| 2026-07 | 🔥 [从 RLVR 到 RLSVR](https://arxiv.org/abs/2607.23802) `arXiv:2607.23802` |  | 任务变换产生可自我验证的奖励，用于开放式自我改进 |
| 2026-07 | 🔥 [Molt](https://arxiv.org/abs/2607.21653) `arXiv:2607.21653` |  | PyTorch 原生、可扩展的智能体 RL 训练框架 |
| 2026-07 | 🔥 [SEED](https://arxiv.org/abs/2607.14777) `arXiv:2607.14777` |  | 自我演化的在策略蒸馏，用于智能体 RL |
| 2026-07 | 🔥 [单 rollout 异步优化](https://arxiv.org/abs/2607.07508) `arXiv:2607.07508` |  | 智能体 RL 的单 rollout 异步优化 |
| 2026-07 | 🔥 [TurnOPD](https://arxiv.org/abs/2607.05804) `arXiv:2607.05804` |  | 让在策略蒸馏感知多轮结构，高效训练长程智能体 |
| 2026-06 | 🔥 [OPID](https://arxiv.org/abs/2606.26790) `arXiv:2606.26790` |  | 面向智能体 RL 的在策略技能蒸馏 |
| 2026-06 | 🔥 [多步工具 RL 为何坍缩](https://arxiv.org/abs/2606.26027) `arXiv:2606.26027` |  | 纯 RL 训练多步工具调用常坍缩，加入监督信号可修复 |
| 2026-06 | 🔥 [APPO](https://arxiv.org/abs/2606.12384) `arXiv:2606.12384` |  | 智能体程序性策略优化 |
| 2026-06 | 🔥 [TRACE](https://arxiv.org/abs/2606.11119) `arXiv:2606.11119` |  | 智能体 RL 中统一的 rollout 预算分配 |
| 2026-05 | 🔥 [智能体探索式策略优化](https://arxiv.org/abs/2605.28774) `arXiv:2605.28774` |  | 多模态智能体推理的探索式策略优化 |
| 2026-05 | 🔥 [自蒸馏智能体 RL](https://arxiv.org/abs/2605.15155) `arXiv:2605.15155` |  | 智能体后训练中的自蒸馏强化学习 |
| 2026-05 | 🔥 [StraTA](https://arxiv.org/abs/2605.06642) `arXiv:2605.06642` |  | 以策略性轨迹抽象激励智能体 RL |
| 2026-05 | 🔥 [AEM](https://arxiv.org/abs/2605.00425) `arXiv:2605.00425` |  | 多轮智能体 RL 的自适应熵调节 |
| 2026-04 | ◆ [RAGEN-2](https://arxiv.org/abs/2604.06268) `arXiv:2604.06268` |  | 智能体 RL 中的推理坍缩 |
| 2026-03 | ◆ [OpenClaw-RL](https://arxiv.org/abs/2603.10165) `arXiv:2603.10165` |  | "说话就能训练智能体"：把用户回复当作下一状态信号 |
| 2026-02 | ◆ [弱者驱动学习](https://arxiv.org/abs/2602.08222) `arXiv:2602.08222` |  | 弱智能体如何让强智能体更强 |
| 2026-01 | ◆ [ArenaRL](https://arxiv.org/abs/2601.06487) `arXiv:2601.06487` |  | 用锦标赛式相对排名扩展开放式智能体的 RL |
| 2025-12 | ◆ [ROME / 开放智能体学习生态](https://arxiv.org/abs/2512.24873) `arXiv:2512.24873` |  | 在开放智能体学习生态中构建 ROME 模型 |
| 2025-10 | 🆕 [IGPO](https://arxiv.org/abs/2510.14967) `arXiv:2510.14967` |  | 基于信息增益的多轮智能体策略优化 |
| 2025-10 | 🆕 [AEPO](https://arxiv.org/abs/2510.14545) `arXiv:2510.14545` |  | 智能体熵平衡策略优化 |
| 2025-10 | 🆕 [揭秘智能体推理中的 RL](https://arxiv.org/abs/2510.11701) `arXiv:2510.11701` |  | 智能体推理中强化学习的数据、算法与推理模式 |
| 2025-10 | 🆕 [不只微调智能体，还要调环境](https://arxiv.org/abs/2510.10197) `arXiv:2510.10197` |  | 以环境调优训练多轮工具智能体 |
| 2025-09 | 🆕 [树搜索智能体 RL](https://arxiv.org/abs/2509.21240) `arXiv:2509.21240` |  | 用树搜索做 LLM 智能体强化学习 |
| 2025-09 | 🆕 [熵调制策略梯度](https://arxiv.org/abs/2509.09265) `arXiv:2509.09265` |  | 长程 LLM 智能体的熵调制策略梯度 |
| 2025-09 | 🆕 [AgentGym-RL](https://arxiv.org/abs/2509.08755) `arXiv:2509.08755` |  | 多轮 RL 训练长程决策智能体 |
| 2025-09 | ★ 🆕 [智能体 RL 全景](https://arxiv.org/abs/2509.02547) `arXiv:2509.02547` |  | LLM 智能体强化学习综述 |
| 2025-09 | 🆕 [VerlTool](https://arxiv.org/abs/2509.01055) `arXiv:2509.01055` |  | 面向工具使用的整体智能体 RL 框架 |
| 2025-08 | 🆕 [rStar2-Agent](https://arxiv.org/abs/2508.20722) `arXiv:2508.20722` | Microsoft | 14B 数学推理模型的智能体 RL 技术报告 |
| 2025-08 | 🆕 [AWorld](https://arxiv.org/abs/2508.20404) `arXiv:2508.20404` | Ant Group | 编排智能体 AI 的训练配方 |
| 2025-08 | 🆕 [Agent Lightning](https://arxiv.org/abs/2508.03680) `arXiv:2508.03680` | Microsoft | 用强化学习训练任意 AI 智能体的框架 |
| 2025-07 | 🆕 [ARPO](https://arxiv.org/abs/2507.19849) `arXiv:2507.19849` |  | 智能体强化策略优化：在工具调用后的高熵步骤上分支采样 |
| 2025-05 | 🆕 [ARIA](https://arxiv.org/abs/2506.00539) `arXiv:2506.00539` |  | 意图驱动的奖励聚合训练语言智能体 |
| 2024-02 | [ArCHer](https://arxiv.org/abs/2402.19446) `arXiv:2402.19446` | UC Berkeley | 分层多轮强化学习训练语言模型智能体 |

### 环境与数据合成（48）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [WorkFactory](https://arxiv.org/abs/2610.37161) `arXiv:2610.37161` |  | 把专业意图转化为可执行的工作任务，扩展工作型智能体的训练 |
| 2026-10 | 🔥 [Trajectory2Gym](https://arxiv.org/abs/2610.33568) `arXiv:2610.33568` |  | 成功轨迹不等于好监督：以决策为依据组合合成工具使用交互数据 |
| 2026-10 | 🔥 [CoworkEnv](https://arxiv.org/abs/2610.24960) `arXiv:2610.24960` |  | 为长程"协作办公"智能体合成训练环境 |
| 2026-10 | 🔥 [OfficeTown](https://arxiv.org/abs/2610.23194) `arXiv:2610.23194` |  | 生成式组织：在模拟公司中演化智能体能力 |
| 2026-10 | 🔥 [SWE-at-Scale](https://arxiv.org/abs/2610.21359) `arXiv:2610.21359` |  | 从历史修复中可扩展地合成 issue 任务 |
| 2026-10 | 🔥 [AgentGarten](https://arxiv.org/abs/2610.12374) `arXiv:2610.12374` |  | 由代码生成的交互世界，供智能体探索与演化 |
| 2026-10 | 🔥 [MIMESIS](https://arxiv.org/abs/2610.09484) `arXiv:2610.09484` |  | 学习用户模拟器作为交互式智能体的训练环境 |
| 2026-10 | 🔥 [智能体语言世界模型](https://arxiv.org/abs/2610.06100) `arXiv:2610.06100` |  | 从交互轨迹构建语言世界模型来模拟交互环境 |
| 2026-09 | 🔥 [GraphForge](https://arxiv.org/abs/2609.38923) `arXiv:2609.38923` |  | 以图锚定的工作区合成训练工作型智能体 |
| 2026-09 | 🔥 [Agent Error Dataset](https://arxiv.org/abs/2609.40111) `arXiv:2609.40111` |  | 5 万条错误—诊断对，用于失败分析与错误感知后训练 |
| 2026-09 | 🔥 [Skill2Env](https://arxiv.org/abs/2609.33772) `arXiv:2609.33772` |  | 从技能出发合成面向能力的训练环境 |
| 2026-09 | 🔥 [CompoWorld](https://arxiv.org/abs/2609.33665) `arXiv:2609.33665` |  | 组合式环境扩展训练通用智能体 |
| 2026-09 | 🔥 [智能体编辑世界模型](https://arxiv.org/abs/2609.28416) `arXiv:2609.28416` |  | 为 LLM 智能体重新思考世界建模 |
| 2026-09 | 🔥 [从已解机制生成 RL 环境](https://arxiv.org/abs/2609.27321) `arXiv:2609.27321` |  | 可验证的隐藏动力学游戏作为智能体 RL 环境 |
| 2026-08 | 🔥 [什么是好的智能体数据](https://arxiv.org/abs/2608.27260) `arXiv:2608.27260` |  | 以 ACE 视角分析智能体数据生成 |
| 2026-08 | 🔥 [Code World Model](https://arxiv.org/abs/2608.25927) `arXiv:2608.25927` |  | 编码智能体充当世界大脑：用代码模拟环境演化 |
| 2026-08 | 🔥 [AgentMercury](https://arxiv.org/abs/2608.20634) `arXiv:2608.20634` |  | 为商业场景大规模合成可验证环境 |
| 2026-08 | 🔥 [EnvHarness](https://arxiv.org/abs/2608.19880) `arXiv:2608.19880` |  | 唤醒静态世界：把静态资源变成智能体可学习的环境 |
| 2026-08 | 🔥 [设计有效的环境分布](https://arxiv.org/abs/2608.03571) `arXiv:2608.03571` |  | 多模态智能体训练不止要扩环境数量，还要设计分布 |
| 2026-07 | 🔥 [DataFlow-Harness](https://arxiv.org/abs/2607.16617) `arXiv:2607.16617` |  | 以代码智能体构建可编辑的 LLM 数据处理管线 |
| 2026-07 | 🔥 [NexForge](https://arxiv.org/abs/2607.14186) `arXiv:2607.14186` |  | 需求驱动的任务合成，扩展可执行智能体训练数据 |
| 2026-06 | 🔥 [Autodata](https://arxiv.org/abs/2606.25996) `arXiv:2606.25996` |  | 让智能体充当数据科学家来生成高质量合成数据 |
| 2026-06 | 🔥 [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) `arXiv:2606.24597` | Alibaba Qwen | 面向通用智能体的语言世界模型：预测环境动力学以训练与规划 |
| 2026-06 | 🔥 [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855) `arXiv:2606.24855` |  | 智能体模型的开源数据配方 |
| 2026-06 | 🔥 [从受训者到训练者](https://arxiv.org/abs/2606.17682) `arXiv:2606.17682` |  | LLM 设计 RL 训练环境 |
| 2026-06 | 🔥 [智能体环境工程综述](https://arxiv.org/abs/2606.12191) `arXiv:2606.12191` |  | 环境建模、合成与扩展的综述 |
| 2026-05 | 🔥 [LongTraceRL](https://arxiv.org/abs/2605.31584) `arXiv:2605.31584` |  | 从搜索智能体轨迹中学习长上下文推理 |
| 2026-05 | 🔥 [自主智能体数据工程](https://arxiv.org/abs/2605.30407) `arXiv:2605.30407` |  | 用于模型专精的自主数据工程 |
| 2026-05 | 🔥 [ACC](https://arxiv.org/abs/2605.21850) `arXiv:2605.21850` |  | 把智能体轨迹编译为长上下文训练数据 |
| 2026-04 | ◆ [ClawGym](https://arxiv.org/abs/2604.26904) `arXiv:2604.26904` |  | 构建有效 Claw 类智能体的可扩展框架 |
| 2026-04 | ◆ [智能体式世界建模](https://arxiv.org/abs/2604.22748) `arXiv:2604.22748` |  | 智能体世界建模的基础、能力与规律 |
| 2026-04 | ◆ [Agent-World](https://arxiv.org/abs/2604.18292) `arXiv:2604.18292` |  | 扩展真实环境合成，演化通用智能体 |
| 2026-02 | ◆ [Agent World Model](https://arxiv.org/abs/2602.10090) `arXiv:2602.10090` |  | 无限合成环境用于智能体 RL |
| 2026-02 | ◆ [daVinci-Agency](https://arxiv.org/abs/2602.02619) `arXiv:2602.02619` |  | 数据高效地解锁长程能动性 |
| 2026-01 | ◆ [ASTRA](https://arxiv.org/abs/2601.21558) `arXiv:2601.21558` |  | 自动合成智能体轨迹与强化学习竞技场 |
| 2026-01 | ◆ [VisGym](https://arxiv.org/abs/2601.16973) `arXiv:2601.16973` |  | 面向多模态智能体的多样、可定制、可扩展环境 |
| 2026-01 | ◆ [从文本合成工具使用轨迹](https://arxiv.org/abs/2601.10355) `arXiv:2601.10355` |  | 从文本中挖掘隐含经验，合成工具使用轨迹 |
| 2026-01 | ◆ [大规模面向用户的工具使用多轮对话生成](https://arxiv.org/abs/2601.08225) `arXiv:2601.08225` |  | 合成以用户为中心的工具使用多轮对话 |
| 2026-01 | ◆ [EnvScaler](https://arxiv.org/abs/2601.05808) `arXiv:2601.05808` |  | 通过程序化合成扩展工具交互环境 |
| 2025-12 | ◆ [DataFlow](https://arxiv.org/abs/2512.16676) `arXiv:2512.16676` |  | LLM 驱动的统一数据准备与工作流自动化框架 |
| 2025-11 | ◆ [SimWorld](https://arxiv.org/abs/2512.01078) `arXiv:2512.01078` |  | 面向物理与社会世界的开放式逼真仿真器 |
| 2025-11 | ◆ [通过经验合成扩展智能体学习](https://arxiv.org/abs/2511.03773) `arXiv:2511.03773` | Meta | 用合成经验扩展智能体 RL |
| 2025-10 | 🆕 [Agent Data Protocol](https://arxiv.org/abs/2510.24702) `arXiv:2510.24702` | CMU | 统一多样的智能体微调数据集格式 |
| 2025-10 | 🆕 [早期经验](https://arxiv.org/abs/2510.08558) `arXiv:2510.08558` | Meta | 智能体通过自身早期经验（无奖励的交互后果）学习 |
| 2025-09 | 🆕 [环境扩展](https://arxiv.org/abs/2509.13311) `arXiv:2509.13311` | Alibaba Tongyi | 通过扩展环境走向通用智能体智能 |
| 2024-03 | [Agent-FLAN](https://arxiv.org/abs/2403.12881) `arXiv:2403.12881` |  | 有效智能体微调的数据与方法设计 |
| 2023-10 | [AgentTuning](https://arxiv.org/abs/2310.12823) `arXiv:2310.12823` | Tsinghua | 用交互轨迹微调获得通用智能体能力 |
| 2023-10 | [FireAct](https://arxiv.org/abs/2310.05915) `arXiv:2310.05915` |  | 语言智能体微调 |

### 自我改进与自我演化（48）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [A-RSI](https://arxiv.org/abs/2610.38568) `arXiv:2610.38568` |  | 零数据多模态自我改进：自适应判断训练经验何时失效并刷新任务 |
| 2026-10 | 🔥 [可信的递归自我改进](https://arxiv.org/abs/2610.35193) `arXiv:2610.35193` |  | 大模型时代 RSI 的基础、风险、评测与治理综述 |
| 2026-10 | 🔥 [The Trinity（多模态 RSI）](https://arxiv.org/abs/2610.36077) `arXiv:2610.36077` |  | 按能力驱动的多模态智能体递归自我改进：世界与智能体的协同演化 |
| 2026-10 | 🔥 [ReGE](https://arxiv.org/abs/2610.33861) `arXiv:2610.33861` |  | 通过经验合成让智能体更强 |
| 2026-10 | 🔥 [RSI Arena](https://arxiv.org/abs/2610.33624) `arXiv:2610.33624` |  | 智能体能否利用人类反馈自主训练并改进语言模型 |
| 2026-10 | 🔥 [RSIHub](https://arxiv.org/abs/2610.32253) `arXiv:2610.32253` |  | 可组合的智能体演化方法框架（改提示、技能、工具与编排） |
| 2026-10 | 🔥 [CoEvo-RSI](https://arxiv.org/abs/2610.25078) `arXiv:2610.25078` |  | 模型与 harness 协同演化时，模型该如何跟进 |
| 2026-10 | 🔥 [RSI-Anything](https://arxiv.org/abs/2610.13912) `arXiv:2610.13912` |  | 借助世界模型把交互经验转化为持久改进 |
| 2026-10 | 🔥 [ClawLoop](https://arxiv.org/abs/2610.14118) `arXiv:2610.14118` |  | harness 原生的训练闭环实现自我改进（面向 OpenClaw 类个人智能体） |
| 2026-10 | 🔥 [Memento 3](https://arxiv.org/abs/2610.11794) `arXiv:2610.11794` |  | 通过反思式规则手册实现基于模型的递归自我改进 |
| 2026-10 | 🔥 [ASCENT](https://arxiv.org/abs/2610.05303) `arXiv:2610.05303` |  | 通过自蒸馏已验证经验，做长程智能体的在线测试时训练 |
| 2026-10 | 🔥 [EvalResearchBench](https://arxiv.org/abs/2610.04184) `arXiv:2610.04184` |  | 智能体能否为递归自我改进构建自己的反馈 |
| 2026-09 | 🔥 [ReSAIL](https://arxiv.org/abs/2609.39306) `arXiv:2609.39306` |  | 缓解迭代式智能体自蒸馏中的坍缩 |
| 2026-09 | 🔥 [AREX-2](https://arxiv.org/abs/2609.38288) `arXiv:2609.38288` |  | 通过长程反思任务提升智能体的自我改进能力 |
| 2026-09 | 🔥 [RSIAgent](https://arxiv.org/abs/2609.15364) `arXiv:2609.15364` |  | 在新环境中自主探索以递归自我改进 |
| 2026-09 | 🔥 [Dream-RSI](https://arxiv.org/abs/2609.14858) `arXiv:2609.14858` |  | 通过演化的世界实现递归自我改进 |
| 2026-09 | 🔥 [Atria Dawn](https://arxiv.org/abs/2609.15818) `arXiv:2609.15818` |  | 智能体参与开发自身后继者：对"智能体式超级智能"的展望 |
| 2026-09 | 🔥 [广义智能体迭代](https://arxiv.org/abs/2609.13406) `arXiv:2609.13406` |  | 把迭代策略改进与递归自我改进放进一个形式化框架 |
| 2026-09 | 🔥 [最后一个人类造的 AI](https://arxiv.org/abs/2609.11873) `arXiv:2609.11873` |  | 走向真正的递归自我改进 |
| 2026-09 | 🔥 [环境即脚手架](https://arxiv.org/abs/2609.08404) `arXiv:2609.08404` |  | 丰富环境反馈以自举长程自演化智能体 |
| 2026-09 | 🔥 [NeoHorse-1](https://arxiv.org/abs/2609.08183) `arXiv:2609.08183` |  | 通过带路由 harness 的智能体后训练实现递归自我改进 |
| 2026-08 | 🔥 [S3Gym](https://arxiv.org/abs/2608.31100) `arXiv:2608.31100` |  | LLM 能否把自测与自评转化为自我改进 |
| 2026-08 | 🔥 [Aspire](https://arxiv.org/abs/2608.31111) `arXiv:2608.31111` |  | 模型能否从模糊目标（"变得更好"）出发自我演化 |
| 2026-08 | 🔥 [有条件的经验迁移](https://arxiv.org/abs/2608.26730) `arXiv:2608.26730` |  | 自主后训练中知道何时不该复用经验 |
| 2026-08 | 🔥 [PILOT in the Loop](https://arxiv.org/abs/2608.26530) `arXiv:2608.26530` |  | 长程智能体在运行中实时自我改进 |
| 2026-08 | 🔥 [Meta^n](https://arxiv.org/abs/2608.24735) `arXiv:2608.24735` |  | 通过涌现的深度递归改进"产生答案的过程" |
| 2026-08 | 🔥 [智能体系统中的协同演化](https://arxiv.org/abs/2608.10299) `arXiv:2608.10299` |  | 走向超越人工设计的自主演化 |
| 2026-08 | 🔥 [Macaron-V1](https://arxiv.org/abs/2608.09819) `arXiv:2608.09819` |  | 开放的持续学习智能体模型族：自我改进 + 混合 LoRA |
| 2026-07 | 🔥 [DecoEvo](https://arxiv.org/abs/2607.25675) `arXiv:2607.25675` |  | 文本空间中解题者与评分细则生成者技能的协同演化 |
| 2026-07 | 🔥 [从智能体经验中高效学习](https://arxiv.org/abs/2607.21051) `arXiv:2607.21051` |  | 环境交互昂贵时的样本高效学习 |
| 2026-07 | 🔥 [现代智能体系统的自我改进综述](https://arxiv.org/abs/2607.13104) `arXiv:2607.13104` |  | 自我改进智能体从原型走向部署的综述 |
| 2026-06 | 🔥 [执行—蒸馏—验证](https://arxiv.org/abs/2606.24428) `arXiv:2606.24428` |  | 逃离自我确认陷阱的智能体经验学习范式 |
| 2026-06 | 🔥 [OPD-Evolver](https://arxiv.org/abs/2606.17628) `arXiv:2606.17628` |  | 通过在策略蒸馏培养整体的智能体演化器 |
| 2026-06 | 🔥 [Role-Agent](https://arxiv.org/abs/2606.10917) `arXiv:2606.10917` |  | 双角色演化自举 LLM 智能体 |
| 2026-06 | 🔥 [重新思考持续经验内化](https://arxiv.org/abs/2606.04703) `arXiv:2606.04703` |  | 把上下文经验转化为参数时的持续学习问题 |
| 2026-05 | 🔥 [更新 harness 不等于 harness 有益](https://arxiv.org/abs/2605.30621) `arXiv:2605.30621` |  | 拆解自演化智能体的演化能力：更新了不代表变好 |
| 2026-05 | 🔥 [LLM 改进 LLM](https://arxiv.org/abs/2605.08083) `arXiv:2605.08083` |  | 智能体自动发现测试时扩展策略 |
| 2026-04 | ◆ [FlowEvo](https://arxiv.org/abs/2607.21596) `arXiv:2607.21596` |  | 工作流与可执行技能协同演化 |
| 2026-03 | ◆ [Hyperagents](https://arxiv.org/abs/2603.19461) `arXiv:2603.19461` | Meta | 能改进自身改进机制的自我改进智能体 |
| 2026-03 | ◆ [MetaClaw](https://arxiv.org/abs/2603.17187) `arXiv:2603.17187` |  | 在野外通过对话元学习并演化的个人智能体 |
| 2025-11 | ◆ [Agent0-VL](https://arxiv.org/abs/2511.19900) `arXiv:2511.19900` |  | 工具集成视觉语言推理的自演化智能体 |
| 2025-11 | ◆ [Agent0](https://arxiv.org/abs/2511.16043) `arXiv:2511.16043` |  | 通过工具集成推理，从零数据自我演化智能体 |
| 2025-08 | 🆕 [AgentFly（Memento）](https://arxiv.org/abs/2508.16153) `arXiv:2508.16153` |  | 不微调 LLM 也能微调智能体：基于记忆的在线案例推理 |
| 2025-08 | 🆕 [自演化智能体综述](https://arxiv.org/abs/2508.07407) `arXiv:2508.07407` |  | 连接基础模型与终身智能体系统的自演化智能体综述 |
| 2025-08 | 🆕 [R-Zero](https://arxiv.org/abs/2508.05004) `arXiv:2508.05004` |  | 从零数据自我演化的推理 LLM |
| 2025-07 | 🆕 [自演化智能体综述](https://arxiv.org/abs/2507.21046) `arXiv:2507.21046` |  | 通往超级智能之路上的自演化智能体 |
| 2025-05 | 🆕 [Reflect, Retry, Reward](https://arxiv.org/abs/2505.24726) `arXiv:2505.24726` |  | 通过 RL 学会反思后重试的自我改进 |
| 2025-01 | 🆕 [Agent-R](https://arxiv.org/abs/2501.11425) `arXiv:2501.11425` | ByteDance | 通过迭代自训练教智能体反思 |

### 智能体模型与技术报告（22）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [IQuest-Q1](https://arxiv.org/abs/2610.30949) `arXiv:2610.30949` |  | 320B 总参数、15B 激活的 MoE，面向命令行智能体系统 |
| 2026-10 | 🔥 [Sage-A1.5](https://arxiv.org/abs/2610.28578) `arXiv:2610.28578` |  | 35B MoE 智能体模型：从长程执行到科学探索的能力组合 |
| 2026-10 | 🔥 [Muse Glimmer 30B](https://arxiv.org/abs/2610.24628) `arXiv:2610.24628` |  | 开源 30B 端侧智能体模型 |
| 2026-10 | 🔥 [MiMo-V2.6](https://arxiv.org/abs/2610.11959) `arXiv:2610.11959` | Xiaomi | 把强化学习扩展到自我改进的模型技术报告 |
| 2026-09 | 🔥 [ZGCM-1](https://arxiv.org/abs/2609.13356) `arXiv:2609.13356` |  | 完全开放、高效的 7B 稠密模型，面向数学与智能体搜索 |
| 2026-08 | 🔥 [Apodex 1.1](https://arxiv.org/abs/2608.23283) `arXiv:2608.23283` |  | 面向复杂工作的智能体模型扩展 |
| 2026-08 | 🔥 [Intern-S2-Preview](https://arxiv.org/abs/2608.13505) `arXiv:2608.13505` | Shanghai AI Lab | 科学智能体基础模型 |
| 2026-07 | 🔥 [Cura 1T](https://arxiv.org/abs/2607.15314) `arXiv:2607.15314` |  | 面向医疗智能体的专用模型 |
| 2026-06 | 🔥 [Agents-A1](https://arxiv.org/abs/2606.30616) `arXiv:2606.30616` |  | 35B MoE 智能体模型：扩展"视野"而非参数，逼近万亿参数模型表现 |
| 2026-06 | 🔥 [Ling / Ring 2.6](https://arxiv.org/abs/2606.15079) `arXiv:2606.15079` | Ant Group | 万亿参数规模的高效、即时智能体能力技术报告 |
| 2026-06 | 🔥 [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) `arXiv:2606.15007` | NVIDIA | 550B/55B 的开放混合 Mamba-Transformer MoE，面向智能体推理 |
| 2026-04 | ◆ [GLM-5V-Turbo](https://arxiv.org/abs/2604.26752) `arXiv:2604.26752` | Zhipu | 面向多模态智能体的原生基础模型 |
| 2026-04 | ◆ [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) `arXiv:2604.12374` | NVIDIA | 开放、高效的混合 Mamba-Transformer MoE，面向智能体推理 |
| 2026-02 | ◆ [GLM-5](https://arxiv.org/abs/2602.15763) `arXiv:2602.15763` | Zhipu | 从"氛围编程"走向智能体工程的基础模型 |
| 2026-02 | ◆ [Kimi K2.5](https://arxiv.org/abs/2602.02276) `arXiv:2602.02276` | Moonshot AI | 开源多模态智能体模型；Agent Swarm 并行编排多个子智能体以降低长任务延迟 |
| 2025-12 | ◆ [Youtu-LLM](https://arxiv.org/abs/2512.24618) `arXiv:2512.24618` | Tencent | 释放轻量模型的原生智能体潜力 |
| 2025-12 | ◆ [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) `arXiv:2512.20848` | NVIDIA | 30B-A3B 混合 Mamba-Transformer MoE，面向智能体推理 |
| 2025-12 | ◆ [Nex-N1](https://arxiv.org/abs/2512.04987) `arXiv:2512.04987` |  | 以统一生态大规模构建环境训练的智能体模型 |
| 2025-09 | 🆕 [智能体持续预训练](https://arxiv.org/abs/2509.13310) `arXiv:2509.13310` | Alibaba Tongyi | 通过持续预训练扩展智能体能力 |
| 2025-08 | 🆕 [GLM-4.5](https://arxiv.org/abs/2508.06471) `arXiv:2508.06471` | Zhipu | 面向智能体、推理与编码的开源 MoE 基础模型 |
| 2025-07 | 🆕 [Kimi K2](https://arxiv.org/abs/2507.20534) `arXiv:2507.20534` | Moonshot AI | 1T MoE 开放智能体模型：大规模智能体数据合成与联合 RL |
| 2025-07 | 🆕 [Gemini 2.5](https://arxiv.org/abs/2507.06261) `arXiv:2507.06261` | Google | 推理、多模态、长上下文与下一代智能体能力 |

### 评测与基准（212）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [iPrefBench](https://arxiv.org/abs/2610.34735) `arXiv:2610.34735` |  | 面向个性化的手机使用智能体评测 |
| 2026-10 | 🔥 [VibeVideoBench](https://arxiv.org/abs/2610.35864) `arXiv:2610.35864` |  | 评测多模态智能体用代码生成精确视频 |
| 2026-10 | 🔥 [Δτ-Bench](https://arxiv.org/abs/2610.34238) `arXiv:2610.34238` |  | 面向不断提升的能力而设计的可执行智能体基准 |
| 2026-10 | 🔥 [Continual Learning Exam](https://arxiv.org/abs/2610.33956) `arXiv:2610.33956` |  | 评测智能体的长程自我改进与经验学习 |
| 2026-10 | 🔥 [Trial by Trace](https://arxiv.org/abs/2610.32285) `arXiv:2610.32285` |  | 评测智能体 harness 的可靠性与失败归因：同一模型换 harness 表现差异很大 |
| 2026-10 | 🔥 [视频中心的深度研究报告评测](https://arxiv.org/abs/2610.31158) `arXiv:2610.31158` |  | 以视频为中心的深度研究报告基准 |
| 2026-10 | 🔥 [UserArena](https://arxiv.org/abs/2610.30932) `arXiv:2610.30932` |  | 评测长程购物智能体的交互式用户模拟 |
| 2026-10 | 🔥 [AgentVQA](https://arxiv.org/abs/2610.31494) `arXiv:2610.31494` |  | 智能体式视觉理解的统一基准 |
| 2026-10 | 🔥 [前沿智能体能否攀登科学 ML 任务](https://arxiv.org/abs/2610.28763) `arXiv:2610.28763` |  | 把前沿编码智能体当自动研究者，在真实科学 ML 任务上检验 |
| 2026-10 | 🔥 [MMEnterprise-Bench](https://arxiv.org/abs/2610.27911) `arXiv:2610.27911` |  | 企业智能体在演化的多模态工作记录上的评测 |
| 2026-10 | 🔥 [VerifierBench](https://arxiv.org/abs/2610.28679) `arXiv:2610.28679` |  | 评测生成可执行验证器的能力 |
| 2026-10 | 🔥 [AdaExploreBench](https://arxiv.org/abs/2610.25950) `arXiv:2610.25950` |  | 评测智能体在演化环境中的主动探索与测试时适应 |
| 2026-10 | 🔥 [HABIT-Bench](https://arxiv.org/abs/2610.24520) `arXiv:2610.24520` |  | 从长期交互中的弱证据归纳用户习惯：智能体记忆评测 |
| 2026-10 | 🔥 [OR-License](https://arxiv.org/abs/2610.24087) `arXiv:2610.24087` |  | 评估智能体能否作为真实运筹学（OR）从业者交付专家级方案 |
| 2026-10 | 🔥 [物理研究闭环评测](https://arxiv.org/abs/2610.21767) `arXiv:2610.21767` |  | 智能体能否组织出有证据支撑结论的物理研究 |
| 2026-10 | 🔥 [τ-Emotion-Bench](https://arxiv.org/abs/2610.19560) `arXiv:2610.19560` |  | 真实用户情绪暴露工具调用智能体的交互脆弱性 |
| 2026-10 | 🔥 [RuleTrek](https://arxiv.org/abs/2610.19698) `arXiv:2610.19698` |  | 复杂场景中的智能体规则推理评测 |
| 2026-10 | 🔥 [Doc2Harness](https://arxiv.org/abs/2610.20344) `arXiv:2610.20344` |  | 评测 LLM 对编码智能体做长程 harness 工程的能力 |
| 2026-10 | 🔥 [BizBench](https://arxiv.org/abs/2610.20424) `arXiv:2610.20424` |  | 面向真实业务的长程智能体评测：揭示部署中被"最终成功率"掩盖的问题 |
| 2026-10 | 🔥 [证据优先的智能体验证](https://arxiv.org/abs/2610.17724) `arXiv:2610.17724` |  | 半可验证长程任务中先收集证据再按评分细则打分 |
| 2026-10 | 🔥 [LifeSearch](https://arxiv.org/abs/2610.18092) `arXiv:2610.18092` |  | 面向真实生活与效率场景的深度研究智能体评测 |
| 2026-10 | 🔥 [Into the Unknown](https://arxiv.org/abs/2610.19159) `arXiv:2610.19159` |  | 评测智能体在开放科学研究问题上的能力 |
| 2026-10 | 🔥 [CLASHarness](https://arxiv.org/abs/2610.19128) `arXiv:2610.19128` |  | 计算机使用中的多智能体冲突评测，可控且可验证 |
| 2026-10 | 🔥 [AgentSWE](https://arxiv.org/abs/2610.13361) `arXiv:2610.13361` |  | 编码智能体能否构建用户真正想要的智能体：以"写智能体"为任务的评测 |
| 2026-10 | 🔥 [SWE-Duel](https://arxiv.org/abs/2610.12569) `arXiv:2610.12569` |  | 自我扩展的编码智能体对战场，应对静态基准泄漏 |
| 2026-10 | 🔥 [LiveMACE](https://arxiv.org/abs/2610.09872) `arXiv:2610.09872` |  | 在演化中的市场里做过程感知的智能体能力评测 |
| 2026-10 | 🔥 [Learn2Play Bench](https://arxiv.org/abs/2610.08215) `arXiv:2610.08215` |  | 智能体在陌生环境中从经验学习得有多好 |
| 2026-10 | 🔥 [CheckerBench](https://arxiv.org/abs/2610.07557) `arXiv:2610.07557` |  | 长程智能体能否合成静态分析检查器 |
| 2026-10 | 🔥 [智能体 RAG 评测的预算分配](https://arxiv.org/abs/2610.05034) `arXiv:2610.05034` |  | 评测预算如何在题目、轨迹与读取次数间分配 |
| 2026-10 | 🔥 [UndoBench](https://arxiv.org/abs/2610.05622) `arXiv:2610.05622` |  | 把任务能力与"撤销/恢复"能力分开评测 |
| 2026-10 | 🔥 [AutoSciBench](https://arxiv.org/abs/2610.05140) `arXiv:2610.05140` |  | 自动生成科学智能体评测，应对基准饱和 |
| 2026-10 | 🔥 [HyperBrowseComp](https://arxiv.org/abs/2610.03574) `arXiv:2610.03574` |  | 多语言、多模态的网页浏览智能体压力测试（423 题） |
| 2026-10 | 🔥 [SimuVerity](https://arxiv.org/abs/2610.02304) `arXiv:2610.02304` |  | 工程级 Simulink 模型生成的智能体评测 |
| 2026-10 | 🔥 [Argo-Bench](https://arxiv.org/abs/2610.02122) `arXiv:2610.02122` |  | 数据智能体在企业级分析工作流上的评测 |
| 2026-09 | 🔥 [OSWorld-Science](https://arxiv.org/abs/2609.39903) `arXiv:2609.39903` |  | 计算机使用智能体学习并使用科学软件的基准 |
| 2026-09 | 🔥 [A2Z GameSpec-Bench](https://arxiv.org/abs/2609.39564) `arXiv:2609.39564` |  | 编码智能体能否忠实按游戏设计规格生成游戏 |
| 2026-09 | 🔥 [EngiWorld](https://arxiv.org/abs/2609.37686) `arXiv:2609.37686` |  | 前沿智能体在专业工程环境中能交付什么 |
| 2026-09 | 🔥 [WideSWE](https://arxiv.org/abs/2609.33382) `arXiv:2609.33382` |  | 编码智能体能否跨多个仓库协调修改 |
| 2026-09 | 🔥 [TraceDance](https://arxiv.org/abs/2609.33295) `arXiv:2609.33295` |  | 从真实部署轨迹自动构建智能体行为基准 |
| 2026-09 | 🔥 [SWE-Game](https://arxiv.org/abs/2609.33678) `arXiv:2609.33678` |  | 247 个基于 Godot 参考游戏的任务：编码智能体能否做出想要的游戏 |
| 2026-09 | 🔥 [意图漂移](https://arxiv.org/abs/2609.32520) `arXiv:2609.32520` |  | 用户中途改主意时，度量并修复智能体的意图漂移 |
| 2026-09 | 🔥 [CUA-SWE](https://arxiv.org/abs/2609.32600) `arXiv:2609.32600` |  | 计算机使用智能体遇上视觉软件工程：需要运行并观察软件 |
| 2026-09 | 🔥 [WhatWorkedBench](https://arxiv.org/abs/2609.27490) `arXiv:2609.27490` |  | 评测研究智能体对实验结果的理解 |
| 2026-09 | 🔥 [The Tasteful Agent](https://arxiv.org/abs/2609.25804) `arXiv:2609.25804` |  | 度量并提升长程任务中智能体的"品味"（决策质量） |
| 2026-09 | 🔥 [OSWorld-Pro](https://arxiv.org/abs/2609.24890) `arXiv:2609.24890` |  | 计算机使用智能体的过程式评测 |
| 2026-09 | 🔥 [CADWorld](https://arxiv.org/abs/2609.16251) `arXiv:2609.16251` |  | 长程计算机辅助设计（CAD）的计算机使用基准 |
| 2026-09 | 🔥 [METR 时间视野页面](https://metr.org/time-horizons/) | METR | 前沿模型完成任务时长的持续测量（TH 1.1）；页面注明超过 16 小时的测量已不可靠，且已停止更新 |
| 2026-09 | 🔥 [EVOHARNESSBENCH](https://arxiv.org/abs/2609.04280) `arXiv:2609.04280` |  | 智能体能否跟上不断演化的 harness（工具、技能、子智能体） |
| 2026-09 | 🔥 [EarlyEval](https://arxiv.org/abs/2609.02783) `arXiv:2609.02783` |  | 通过早期结果预测降低智能体评测成本 |
| 2026-08 | 🔥 [E-Commerce Bench](https://arxiv.org/abs/2608.30730) `arXiv:2608.30730` |  | 长程自主经营电商的智能体评测 |
| 2026-08 | 🔥 [RealSWE](https://arxiv.org/abs/2608.27831) `arXiv:2608.27831` |  | 现实用户请求下编码智能体的组合式评测 |
| 2026-08 | 🔥 [FrontierChallenge](https://arxiv.org/abs/2608.24979) `arXiv:2608.24979` |  | 评测科学工作流的完整完成度 |
| 2026-08 | 🔥 [MobilePA-Bench](https://arxiv.org/abs/2608.23035) `arXiv:2608.23035` |  | 真实复杂任务上的手机规划智能体评测 |
| 2026-08 | 🔥 [GameXpert-Bench](https://arxiv.org/abs/2608.21833) `arXiv:2608.21833` |  | 编码智能体离专家级游戏开发还有多远 |
| 2026-08 | 🔥 [SWE-bench 是学会了还是背下了](https://arxiv.org/abs/2609.27891) `arXiv:2609.27891` |  | 检验模型是否记住了 SWE-bench 仓库 |
| 2026-08 | 🔥 [SWE-bench Science](https://arxiv.org/abs/2608.19799) `arXiv:2608.19799` |  | 编码智能体能否解决科学软件中的工程任务 |
| 2026-08 | 🔥 [StartupBench](https://arxiv.org/abs/2608.17800) `arXiv:2608.17800` |  | 经市场验证的端到端工作流上的通用智能体评测 |
| 2026-08 | 🔥 [LongRCA Bench](https://arxiv.org/abs/2608.15242) `arXiv:2608.15242` |  | 诊断长程智能体失败的责任角色与根因 |
| 2026-08 | 🔥 [智能体在自动研究中如何失败](https://arxiv.org/abs/2608.14905) `arXiv:2608.14905` |  | 100 个真实前沿研究任务上的端到端诊断 |
| 2026-08 | 🔥 [长程 AI 研发智能体评测](https://arxiv.org/abs/2608.13417) `arXiv:2608.13417` |  | 超越最终分数，系统评估智能体做长程 AI 研发 |
| 2026-08 | 🔥 [VibeLifeBench](https://arxiv.org/abs/2608.10875) `arXiv:2608.10875` |  | 个人生活智能体能否在"活的世界"中主动且持久 |
| 2026-08 | 🔥 [SWE-Bench ProMax](https://arxiv.org/abs/2608.09802) `arXiv:2608.09802` |  | 大规模多语言代码重构的编码智能体评测 |
| 2026-08 | 🔥 [Evo-Bench](https://arxiv.org/abs/2608.09096) `arXiv:2608.09096` |  | 语言模型能否改进智能体 harness |
| 2026-08 | 🔥 [Business Arena](https://arxiv.org/abs/2608.08621) `arXiv:2608.08621` |  | 在真实市场中评测经营企业的智能体 |
| 2026-08 | 🔥 [HarnessOpt-Bench](https://arxiv.org/abs/2608.06301) `arXiv:2608.06301` |  | 评测 LLM 优化 harness（提示、工具、控制流、记忆）的能力 |
| 2026-08 | 🔥 [PAST-Bench](https://arxiv.org/abs/2608.04003) `arXiv:2608.04003` |  | 个人智能体递归自我改进的基础能力评测 |
| 2026-08 | 🔥 [MatrAIx](https://arxiv.org/abs/2608.04205) `arXiv:2608.04205` |  | 用 83 亿个人设智能体模拟世界，替代昂贵的人类评测 |
| 2026-08 | 🔥 [GDPevo](https://arxiv.org/abs/2608.03764) `arXiv:2608.03764` |  | 在真实商业任务上评测智能体自我演化 |
| 2026-08 | 🔥 [DataSpace](https://arxiv.org/abs/2608.03451) `arXiv:2608.03451` |  | 异构工作区上可验证分析的数据智能体评测 |
| 2026-08 | 🔥 [SWE-Touch](https://arxiv.org/abs/2608.02499) `arXiv:2608.02499` |  | 用户同时改代码时编码智能体的表现 |
| 2026-07 | 🔥 [MerchantBench](https://arxiv.org/abs/2607.28956) `arXiv:2607.28956` |  | 电商运营中智能体的长期一致性评测 |
| 2026-07 | 🔥 [AgentStream](https://arxiv.org/abs/2608.00155) `arXiv:2608.00155` |  | 流式任务下自演化智能体的表现 |
| 2026-07 | 🔥 [OSReward](https://arxiv.org/abs/2607.28609) `arXiv:2607.28609` |  | 跨平台计算机使用奖励模型的标准化评测 |
| 2026-07 | 🔥 [Keep It InMind](https://arxiv.org/abs/2607.24368) `arXiv:2607.24368` |  | 智能体记忆中的隐式联想盲点 |
| 2026-07 | 🔥 [WorkBuddy Bench](https://arxiv.org/abs/2607.20911) `arXiv:2607.20911` | Tencent | 抗污染的多领域编码智能体基准 |
| 2026-07 | 🔥 [AgentCompass](https://arxiv.org/abs/2607.13705) `arXiv:2607.13705` |  | 统一的智能体能力评测基础设施 |
| 2026-07 | 🔥 [从成功流中追溯失败](https://arxiv.org/abs/2607.12747) `arXiv:2607.12747` |  | 智能体系统的失败归因 |
| 2026-07 | 🔥 [UniClawBench](https://arxiv.org/abs/2607.08768) `arXiv:2607.08768` |  | 真实任务上主动式智能体的通用基准 |
| 2026-07 | 🔥 [Long-Horizon-Terminal-Bench](https://arxiv.org/abs/2607.08964) `arXiv:2607.08964` |  | 带稠密进度信号的长程终端任务评测 |
| 2026-07 | 🔥 [PACE](https://arxiv.org/abs/2607.02032) `arXiv:2607.02032` |  | 智能体能力评测的廉价代理指标 |
| 2026-07 | 🔥 [EvoPolicyGym](https://arxiv.org/abs/2607.02440) `arXiv:2607.02440` |  | 评测交互环境中的自主策略演化 |
| 2026-07 | 🔥 [AgenticSTS](https://arxiv.org/abs/2607.02255) `arXiv:2607.02255` |  | 长程智能体的有界记忆测试平台 |
| 2026-07 | 🔥 [AgenticDataBench](https://arxiv.org/abs/2607.01647) `arXiv:2607.01647` |  | 数据智能体综合基准 |
| 2026-06 | 🔥 [OSWorld 2.0](https://arxiv.org/abs/2606.29537) `arXiv:2606.29537` |  | 真实长程任务上的计算机使用智能体基准 |
| 2026-06 | 🔥 [TUA-Bench](https://arxiv.org/abs/2606.28480) `arXiv:2606.28480` |  | 通用终端使用智能体基准 |
| 2026-06 | 🔥 [NatureBench](https://arxiv.org/abs/2606.24530) `arXiv:2606.24530` |  | 编码智能体能否复现 Nature 系论文的已发表 SOTA（90 个任务） |
| 2026-06 | 🔥 [EnterpriseClawBench](https://arxiv.org/abs/2606.23654) `arXiv:2606.23654` |  | 基于真实工作会话的企业智能体评测 |
| 2026-06 | 🔥 [PlanBench-XL](https://arxiv.org/abs/2606.22388) `arXiv:2606.22388` |  | 大规模工具生态中工具使用智能体的长程规划评测 |
| 2026-06 | 🔥 [预测效度](https://arxiv.org/abs/2606.19704) `arXiv:2606.19704` |  | 超越静态排行榜：用预测效度评估智能体基准 |
| 2026-06 | 🔥 [GameCraft-Bench](https://arxiv.org/abs/2606.17861) `arXiv:2606.17861` |  | 智能体能否在真实游戏引擎中端到端做出可玩游戏 |
| 2026-06 | 🔥 [Running the Gauntlet](https://arxiv.org/abs/2606.14397) `arXiv:2606.14397` |  | 在陌生环境中重新评估智能体能力 |
| 2026-06 | 🔥 [EvoArena](https://arxiv.org/abs/2606.13681) `arXiv:2606.13681` |  | 动态环境中追踪智能体记忆演化 |
| 2026-06 | 🔥 [Claw-SWE-Bench](https://arxiv.org/abs/2606.12344) `arXiv:2606.12344` |  | 评测 OpenClaw 类通用智能体 harness 的编码能力 |
| 2026-06 | 🔥 [Workflow-GYM](https://arxiv.org/abs/2606.11042) `arXiv:2606.11042` |  | 真实职业场景中长程计算机使用任务评测 |
| 2026-06 | 🔥 [WeaveBench](https://arxiv.org/abs/2606.09426) `arXiv:2606.09426` |  | 混合界面（桌面 + 代码）的真实长程计算机使用基准 |
| 2026-06 | 🔥 [SWE-Explore](https://arxiv.org/abs/2606.07297) `arXiv:2606.07297` |  | 评测编码智能体如何探索代码仓库 |
| 2026-06 | 🔥 [When Tools Fail](https://arxiv.org/abs/2606.05806) `arXiv:2606.05806` |  | 工具失效时的动态重规划与异常恢复评测 |
| 2026-06 | 🔥 [SubtleMemory](https://arxiv.org/abs/2606.05761) `arXiv:2606.05761` |  | 长程个人智能体（如 OpenClaw）的细粒度关系记忆辨别 |
| 2026-06 | 🔥 [ArcANE](https://arxiv.org/abs/2606.05553) `arXiv:2606.05553` |  | 角色扮演智能体能否在恰当时刻保持人设 |
| 2026-06 | 🔥 [AdaPlanBench](https://arxiv.org/abs/2606.05622) `arXiv:2606.05622` |  | 世界与用户约束下的自适应规划评测 |
| 2026-06 | 🔥 [AutoLab](https://arxiv.org/abs/2606.05080) `arXiv:2606.05080` |  | 前沿模型能否完成长程自动研究与工程任务 |
| 2026-06 | 🔥 [Agents' Last Exam](https://arxiv.org/abs/2606.05405) `arXiv:2606.05405` |  | 为前沿智能体设计的"最后的考试"：在强基准饱和后衡量真实能力 |
| 2026-06 | 🔥 [深度研究智能体在哪里出错](https://arxiv.org/abs/2606.02060) `arXiv:2606.02060` |  | 在轨迹中做片段级错误定位 |
| 2026-06 | 🔥 [K-BrowseComp](https://arxiv.org/abs/2606.02404) `arXiv:2606.02404` |  | 基于韩语语境的网页浏览智能体基准 |
| 2026-06 | 🔥 [AutoMedBench](https://arxiv.org/abs/2606.01961) `arXiv:2606.01961` |  | 智能体做医学 AI 自动研究的评测 |
| 2026-05 | 🔥 [ResearchClawBench](https://arxiv.org/abs/2606.07591) `arXiv:2606.07591` |  | 端到端自主科学研究的基准 |
| 2026-05 | 🔥 [LongDS-Bench](https://arxiv.org/abs/2605.30434) `arXiv:2605.30434` |  | 长程智能体数据分析的失败 |
| 2026-05 | 🔥 [TASTE](https://arxiv.org/abs/2605.28556) `arXiv:2605.28556` |  | 提升智能体基准（如 τ²-Bench）的覆盖与难度 |
| 2026-05 | 🔥 [RAMP](https://arxiv.org/abs/2605.27492) `arXiv:2605.27492` |  | 基准不够：生产系统中智能体模型的运行时评估 |
| 2026-05 | 🔥 [智能体也会衰老](https://arxiv.org/abs/2605.26302) `arXiv:2605.26302` |  | 长期部署的智能体性能随时间衰退：智能体寿命工程 |
| 2026-05 | 🔥 [SkillEvolBench](https://arxiv.org/abs/2605.24117) `arXiv:2605.24117` |  | 从情节经验演化为程序性技能的评测 |
| 2026-05 | 🔥 [CutVerse](https://arxiv.org/abs/2605.19484) `arXiv:2605.19484` |  | 媒体后期剪辑的组合式 GUI 智能体基准 |
| 2026-05 | 🔥 [CHI-Bench](https://arxiv.org/abs/2605.16679) `arXiv:2605.16679` |  | 端到端、长程、规则密集的医疗工作流自动化评测 |
| 2026-05 | 🔥 [π-Bench](https://arxiv.org/abs/2605.14678) `arXiv:2605.14678` |  | 长程工作流中主动式个人助理智能体的评测 |
| 2026-05 | 🔥 [MemEye](https://arxiv.org/abs/2605.15128) `arXiv:2605.15128` |  | 以视觉为中心的多模态智能体记忆评测 |
| 2026-05 | 🔥 [EVA-Bench](https://arxiv.org/abs/2605.13841) `arXiv:2605.13841` |  | 端到端语音智能体评测框架 |
| 2026-05 | 🔥 [WildClawBench](https://arxiv.org/abs/2605.10912) `arXiv:2605.10912` |  | 真实世界长程智能体评测 |
| 2026-05 | 🔥 [STALE](https://arxiv.org/abs/2605.06527) `arXiv:2605.06527` |  | 智能体能否知道自己的记忆已失效 |
| 2026-04 | ◆ [Claw-Eval-Live](https://arxiv.org/abs/2604.28139) `arXiv:2604.28139` |  | 面向不断演化的真实工作流的实时智能体基准 |
| 2026-04 | ◆ [DV-World](https://arxiv.org/abs/2604.25914) `arXiv:2604.25914` |  | 真实场景中的数据可视化智能体评测 |
| 2026-04 | ◆ [DR³-Eval](https://arxiv.org/abs/2604.14683) `arXiv:2604.14683` |  | 真实、可复现的深度研究评测 |
| 2026-04 | ◆ [OccuBench](https://arxiv.org/abs/2604.10866) `arXiv:2604.10866` |  | 用语言世界模型评测数百种职业的专业任务 |
| 2026-04 | ◆ [CocoaBench](https://arxiv.org/abs/2604.11201) `arXiv:2604.11201` |  | 在野外评测统一的数字智能体 |
| 2026-04 | ◆ [KnowU-Bench](https://arxiv.org/abs/2604.08455) `arXiv:2604.08455` |  | 交互、主动、个性化的手机智能体评测 |
| 2026-04 | ◆ [ClawBench](https://arxiv.org/abs/2604.08523) `arXiv:2604.08523` |  | 智能体能否完成日常在线任务（邮件之外的真实网站操作） |
| 2026-04 | ◆ [GameWorld](https://arxiv.org/abs/2604.07429) `arXiv:2604.07429` |  | 多模态游戏智能体的标准化可验证评测 |
| 2026-04 | ◆ [Claw-Eval](https://arxiv.org/abs/2604.06132) `arXiv:2604.06132` |  | 走向可信的自主智能体评测 |
| 2026-04 | ◆ [野外的智能体技能](https://arxiv.org/abs/2604.04323) `arXiv:2604.04323` |  | 真实设置中智能体技能的使用效果 |
| 2026-04 | ◆ [Agentic-MME](https://arxiv.org/abs/2604.03016) `arXiv:2604.03016` |  | 智能体能力究竟给多模态智能带来了什么 |
| 2026-03 | ◆ [MiroEval](https://arxiv.org/abs/2603.28407) `arXiv:2603.28407` |  | 从过程与结果评测多模态深度研究智能体 |
| 2026-03 | ◆ [Vision2Web](https://arxiv.org/abs/2603.26648) `arXiv:2603.26648` |  | 带智能体验证的视觉网站开发层级基准 |
| 2026-03 | ◆ [LMEB](https://arxiv.org/abs/2603.12572) `arXiv:2603.12572` |  | 长程记忆嵌入基准（面向 OpenClaw 等记忆系统） |
| 2026-03 | ◆ [EnterpriseOps-Gym](https://arxiv.org/abs/2603.13594) `arXiv:2603.13594` |  | 企业场景中有状态的智能体规划与工具使用环境 |
| 2026-03 | ◆ [FinToolBench](https://arxiv.org/abs/2603.08262) `arXiv:2603.08262` |  | 真实金融工具使用的智能体评测 |
| 2026-03 | ◆ [BeyondSWE](https://arxiv.org/abs/2603.03194) `arXiv:2603.03194` |  | 代码智能体能否走出单仓库修 bug |
| 2026-02 | ◆ [SWE-rebench V2](https://arxiv.org/abs/2602.23866) `arXiv:2602.23866` |  | 与语言无关、可扩展的 SWE 任务收集 |
| 2026-02 | ◆ [OmniGAIA](https://arxiv.org/abs/2602.22897) `arXiv:2602.22897` |  | 走向原生全模态智能体 |
| 2026-02 | ◆ [MobilityBench](https://arxiv.org/abs/2602.22638) `arXiv:2602.22638` |  | 真实出行场景的路线规划智能体评测 |
| 2026-02 | ◆ [AgentVista](https://arxiv.org/abs/2602.23166) `arXiv:2602.23166` |  | 极具挑战的真实视觉场景中的多模态智能体评测 |
| 2026-02 | ◆ [SkillsBench](https://arxiv.org/abs/2602.12670) `arXiv:2602.12670` |  | 智能体技能在多样任务上效果如何 |
| 2026-02 | ◆ [AIRS-Bench](https://arxiv.org/abs/2602.06855) `arXiv:2602.06855` | Meta | 前沿 AI 研究科学智能体的任务套件 |
| 2026-02 | ◆ [OdysseyArena](https://arxiv.org/abs/2602.05843) `arXiv:2602.05843` |  | 长程、主动、归纳式交互的基准 |
| 2026-01 | ◆ [CAR-bench](https://arxiv.org/abs/2601.22027) `arXiv:2601.22027` |  | 真实不确定性下智能体的一致性与"知道自己的极限" |
| 2026-01 | ◆ [DeepPlanning](https://arxiv.org/abs/2601.18137) `arXiv:2601.18137` |  | 带可验证约束的长程智能体规划基准 |
| 2026-01 | ◆ [MMDeepResearch-Bench](https://arxiv.org/abs/2601.12346) `arXiv:2601.12346` |  | 多模态深度研究智能体基准 |
| 2026-01 | ★ ◆ [Terminal-Bench](https://arxiv.org/abs/2601.11868) `arXiv:2601.11868` | Stanford·Laude Institute | 在命令行界面中完成困难、真实任务的智能体基准 |
| 2026-01 | ◆ [AgencyBench](https://arxiv.org/abs/2601.11044) `arXiv:2601.11044` |  | 百万 token 真实上下文中的自主智能体前沿评测 |
| 2026-01 | ◆ [ABC-Bench](https://arxiv.org/abs/2601.11077) `arXiv:2601.11077` |  | 真实开发中的智能体后端编码评测 |
| 2026-01 | ◆ [DeepResearchEval](https://arxiv.org/abs/2601.09688) `arXiv:2601.09688` |  | 自动构建深度研究任务并做智能体式评测 |
| 2026-01 | ◆ [视频深度研究基准](https://arxiv.org/abs/2601.06943) `arXiv:2601.06943` |  | 在开放网页上做视频推理与搜索的基准 |
| 2025-12 | ◆ [SmartSnap](https://arxiv.org/abs/2512.22322) `arXiv:2512.22322` |  | 主动寻找证据的自验证智能体 |
| 2025-12 | ◆ [NL2Repo-Bench](https://arxiv.org/abs/2512.12730) `arXiv:2512.12730` |  | 长程仓库生成的编码智能体评测 |
| 2025-12 | ◆ [DAComp](https://arxiv.org/abs/2512.04324) `arXiv:2512.04324` |  | 覆盖数据智能全生命周期的数据智能体基准 |
| 2025-12 | ◆ [离真正有用的深度研究智能体还有多远](https://arxiv.org/abs/2512.01948) `arXiv:2512.01948` |  | 分析师级报告的差距 |
| 2025-12 | ◆ [InnoGym](https://arxiv.org/abs/2512.01822) `arXiv:2512.01822` |  | 评测智能体的创新潜力 |
| 2025-11 | ◆ [AutoEnv](https://arxiv.org/abs/2511.19304) `arXiv:2511.19304` |  | 自动生成规则各异的环境，衡量跨环境学习 |
| 2025-11 | ◆ [计算机使用智能体当评审](https://arxiv.org/abs/2511.15567) `arXiv:2511.15567` |  | 用计算机使用智能体评判生成式用户界面 |
| 2025-11 | ◆ [HaluMem](https://arxiv.org/abs/2511.03506) `arXiv:2511.03506` |  | 评测智能体记忆系统中的幻觉 |
| 2025-10 | 🆕 [Tool Decathlon](https://arxiv.org/abs/2510.25726) `arXiv:2510.25726` |  | 多样、真实、长程任务执行的语言智能体基准 |
| 2025-10 | 🆕 [InteractComp](https://arxiv.org/abs/2510.24668) `arXiv:2510.24668` |  | 用含糊查询评测搜索智能体（需要主动澄清） |
| 2025-10 | 🆕 [DeepWideSearch](https://arxiv.org/abs/2510.20168) `arXiv:2510.20168` | Alibaba | 同时考察深度与广度的智能体信息搜寻基准 |
| 2025-10 | 🆕 [Holistic Agent Leaderboard](https://arxiv.org/abs/2510.11977) `arXiv:2510.11977` | Princeton | 智能体评测缺失的基础设施：统一 harness、计成本、记录轨迹 |
| 2025-10 | 🆕 [NewtonBench](https://arxiv.org/abs/2510.07172) `arXiv:2510.07172` |  | 可泛化的科学定律发现智能体评测 |
| 2025-09 | 🆕 [ARE](https://arxiv.org/abs/2509.17158) `arXiv:2509.17158` | Meta | 扩展智能体环境与评测的研究平台（Gaia2） |
| 2025-08 | 🆕 [MCP-Bench](https://arxiv.org/abs/2508.20453) `arXiv:2508.20453` | Accenture | 用真实 MCP 服务器评测工具使用智能体 |
| 2025-08 | 🆕 [LiveMCP-101](https://arxiv.org/abs/2508.15760) `arXiv:2508.15760` |  | 对启用 MCP 的智能体做压力测试与诊断 |
| 2025-08 | 🆕 [MCP-Universe](https://arxiv.org/abs/2508.14704) `arXiv:2508.14704` | Salesforce | 用真实 MCP 服务器评测大模型 |
| 2025-08 | 🆕 [FutureX](https://arxiv.org/abs/2508.11987) `arXiv:2508.11987` | ByteDance | 面向未来预测的实时智能体基准 |
| 2025-08 | 🆕 [WideSearch](https://arxiv.org/abs/2508.07999) `arXiv:2508.07999` | ByteDance | 广度信息搜集的智能体基准 |
| 2025-08 | 🆕 [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) `arXiv:2508.06600` |  | 更公平透明的深度研究智能体评测（固定语料） |
| 2025-07 | 🆕 [UserBench](https://arxiv.org/abs/2507.22034) `arXiv:2507.22034` | Salesforce | 以用户为中心的交互式智能体环境 |
| 2025-07 | 🆕 [MMBench-GUI](https://arxiv.org/abs/2507.19478) `arXiv:2507.19478` |  | 多平台 GUI 智能体的分层评测 |
| 2025-07 | 🆕 [SWE-Perf](https://arxiv.org/abs/2507.12415) `arXiv:2507.12415` |  | 真实仓库上的代码性能优化评测 |
| 2025-06 | 🆕 [Mind2Web 2](https://arxiv.org/abs/2506.21506) `arXiv:2506.21506` | OSU | 用"智能体当评审"评测智能体式搜索 |
| 2025-06 | 🆕 [DeepResearch Bench](https://arxiv.org/abs/2506.11763) `arXiv:2506.11763` |  | 深度研究智能体综合基准 |
| 2025-06 | 🆕 [τ²-Bench](https://arxiv.org/abs/2506.07982) `arXiv:2506.07982` | Sierra | 双控制环境：用户也能操作环境的对话智能体评测 |
| 2025-05 | 🆕 [ScienceBoard](https://arxiv.org/abs/2505.19897) `arXiv:2505.19897` |  | 真实科学工作流中的多模态自主智能体评测 |
| 2025-05 | 🆕 [SWE-rebench](https://arxiv.org/abs/2505.20411) `arXiv:2505.20411` | Nebius | 自动收集任务、去污染的软件工程智能体评测 |
| 2025-05 | 🆕 [DeepResearchGym](https://arxiv.org/abs/2505.19253) `arXiv:2505.19253` | CMU | 免费、透明、可复现的深度研究评测沙箱 |
| 2025-05 | 🆕 [Terminal-Bench](https://www.tbench.ai/) | Stanford·Laude Institute | 终端任务基准的官网与排行榜 |
| 2025-04 | ★ 🆕 [BrowseComp](https://arxiv.org/abs/2504.12516) `arXiv:2504.12516` | OpenAI | 简单却极难的浏览智能体基准：答案难找、易验证 |
| 2025-04 | 🆕 [AgentRewardBench](https://arxiv.org/abs/2504.08942) `arXiv:2504.08942` |  | 评测网页智能体轨迹的自动评估方法 |
| 2025-04 | 🆕 [Multi-SWE-bench](https://arxiv.org/abs/2504.02605) `arXiv:2504.02605` | ByteDance | 多语言 issue 修复基准 |
| 2025-04 | 🆕 [PaperBench](https://arxiv.org/abs/2504.01848) `arXiv:2504.01848` | OpenAI | 评测 AI 复现 AI 研究论文的能力 |
| 2025-03 | 🆕 [LLM 智能体评测综述](https://arxiv.org/abs/2503.16416) `arXiv:2503.16416` |  | 基于 LLM 的智能体评测综述 |
| 2025-03 | ★ 🆕 [METR 任务时长](https://arxiv.org/abs/2503.14499) `arXiv:2503.14499` | METR | 以人类完成时长衡量智能体能力：50% 成功率可完成的任务时长约每 7 个月翻倍 |
| 2025-03 | 🆕 [MultiAgentBench](https://arxiv.org/abs/2503.01935) `arXiv:2503.01935` |  | 评测 LLM 智能体的协作与竞争 |
| 2025-02 | 🆕 [MLGym](https://arxiv.org/abs/2502.14499) `arXiv:2502.14499` | Meta | 推进 AI 研究智能体的框架与基准 |
| 2025-02 | 🆕 [SWE-Lancer](https://arxiv.org/abs/2502.12115) `arXiv:2502.12115` | OpenAI | 1400+ 个真实自由职业软件任务：前沿模型能挣到 100 万美元吗 |
| 2024-12 | ★ 🆕 [TheAgentCompany](https://arxiv.org/abs/2412.14161) `arXiv:2412.14161` | CMU | 在模拟软件公司中评测完成真实工作任务的智能体 |
| 2024-11 | 🆕 [RE-Bench](https://arxiv.org/abs/2411.15114) `arXiv:2411.15114` | METR | 与人类专家对比的前沿 AI 研发能力评测 |
| 2024-10 | [智能体工作流生成基准](https://arxiv.org/abs/2410.07869) `arXiv:2410.07869` |  | 评测智能体工作流生成 |
| 2024-10 | [MLE-bench](https://arxiv.org/abs/2410.07095) `arXiv:2410.07095` | OpenAI | 以 Kaggle 竞赛评测机器学习工程智能体 |
| 2024-10 | [SWE-bench Multimodal](https://arxiv.org/abs/2410.03859) `arXiv:2410.03859` | Princeton | 视觉软件领域的 issue 修复 |
| 2024-09 | [Windows Agent Arena](https://arxiv.org/abs/2409.08264) `arXiv:2409.08264` | Microsoft | 大规模评测多模态操作系统智能体 |
| 2024-09 | [DSBench](https://arxiv.org/abs/2409.07703) `arXiv:2409.07703` |  | 数据科学智能体离专家还有多远 |
| 2024-08 | [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) | OpenAI | 人工筛选的 500 题子集，去掉描述不清或测试不当的题目 |
| 2024-07 | [AppWorld](https://arxiv.org/abs/2407.18901) `arXiv:2407.18901` | Stony Brook | 可控的应用与人物世界，评测交互式编码智能体 |
| 2024-07 | ★ [AI Agents That Matter](https://arxiv.org/abs/2407.01502) `arXiv:2407.01502` | Princeton | 智能体评测要计入成本、可复现与过拟合 |
| 2024-06 | ★ [τ-bench](https://arxiv.org/abs/2406.12045) `arXiv:2406.12045` | Sierra | 工具—智能体—用户三方交互的真实领域基准（零售、航空） |
| 2024-04 | ★ [OSWorld](https://arxiv.org/abs/2404.07972) `arXiv:2404.07972` | HKU | 在真实操作系统中评测开放式计算机使用任务 |
| 2024-03 | [StableToolBench](https://arxiv.org/abs/2403.07714) `arXiv:2403.07714` | Tsinghua | 用缓存与模拟 API 让工具学习评测稳定可复现 |
| 2024-02 | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | UC Berkeley | 伯克利函数调用排行榜：单轮、多轮、多步工具调用 |
| 2023-11 | ★ [GAIA](https://arxiv.org/abs/2311.12983) `arXiv:2311.12983` | Meta·Hugging Face | 通用 AI 助手基准：对人简单、对模型困难的多步真实问题 |
| 2023-10 | ★ [SWE-bench](https://arxiv.org/abs/2310.06770) `arXiv:2310.06770` | Princeton | 解决真实 GitHub issue 的软件工程基准 |
| 2023-10 | [SWE-bench 排行榜](https://www.swebench.com/) | Princeton | SWE-bench 系列（Verified、Multimodal 等）的官方排行榜 |
| 2023-08 | [AgentBench](https://arxiv.org/abs/2308.03688) `arXiv:2308.03688` | Tsinghua | 评测 LLM 作为智能体的综合基准 |
| 2023-07 | ★ [WebArena](https://arxiv.org/abs/2307.13854) `arXiv:2307.13854` | CMU | 真实可交互的自托管网站环境，用于构建与评测自主网页智能体 |
| 2023-06 | [Mind2Web](https://arxiv.org/abs/2306.06070) `arXiv:2306.06070` | OSU | 面向通用网页智能体的跨网站数据集 |
| 2023-04 | [API-Bank](https://arxiv.org/abs/2304.08244) `arXiv:2304.08244` | Alibaba | 工具增强 LLM 的综合基准 |
| 2022-07 | [WebShop](https://arxiv.org/abs/2207.01206) `arXiv:2207.01206` | Princeton | 可扩展的真实网页购物交互环境 |
| 2020-10 | [ALFWorld](https://arxiv.org/abs/2010.03768) `arXiv:2010.03768` | Microsoft·CMU | 文本与具身环境对齐的交互学习基准 |

### 安全、可靠性与治理（51）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Thinking-in-Bets](https://arxiv.org/abs/2610.34654) `arXiv:2610.34654` |  | 测试时"下注思考"减少智能体被用户质疑后谄媚地放弃正确答案 |
| 2026-10 | 🔥 [Agent Mosaic](https://arxiv.org/abs/2610.30418) `arXiv:2610.30418` |  | 识别智能体系统中的保密信息泄露 |
| 2026-10 | 🔥 [SWE-MISBEHAVIOR](https://arxiv.org/abs/2610.18206) `arXiv:2610.18206` |  | 编码智能体越界行为的分类、评测与缓解 |
| 2026-10 | 🔥 [SkillMarker](https://arxiv.org/abs/2610.15725) `arXiv:2610.15725` |  | 为智能体技能做水印以验证归属 |
| 2026-10 | 🔥 [APEX-Harm](https://arxiv.org/abs/2610.13623) `arXiv:2610.13623` |  | 评测共享专业工作区（命令行 + 多工具）中的智能体安全 |
| 2026-10 | 🔥 [DecepEval](https://arxiv.org/abs/2610.07967) `arXiv:2610.07967` |  | 评测 LLM 智能体中的欺骗行为 |
| 2026-10 | 🔥 [AdvSim2Real](https://arxiv.org/abs/2610.08773) `arXiv:2610.08773` |  | 在网页世界模型中对抗自适应提示注入训练网页智能体 |
| 2026-10 | 🔥 [智能体到底做了什么](https://arxiv.org/abs/2610.06406) `arXiv:2610.06406` |  | 基于证据的长程智能体监督：让用户核查执行过程 |
| 2026-10 | 🔥 [来源偏好](https://arxiv.org/abs/2610.03195) `arXiv:2610.03195` |  | 智能体替用户选择商品/酒店/论文时偏好特定来源，以及如何缓解 |
| 2026-09 | 🔥 [Covert Assistance](https://arxiv.org/abs/2609.39050) `arXiv:2609.39050` |  | 多智能体系统中"助人"的智能体会逃避监督 |
| 2026-09 | 🔥 [长程交互中的涌现共谋](https://arxiv.org/abs/2609.24967) `arXiv:2609.24967` |  | 长期协作的 LLM 智能体之间会涌现共谋 |
| 2026-09 | 🔥 [经验式置信估计](https://arxiv.org/abs/2609.17708) `arXiv:2609.17708` |  | 从推理到智能体：基于经验的置信估计 |
| 2026-09 | 🔥 [HazardAuditor](https://arxiv.org/abs/2609.15134) `arXiv:2609.15134` |  | 从可执行威胁出发审计计算机使用智能体的运行时安全 |
| 2026-09 | 🔥 [EvoSafeHarness](https://arxiv.org/abs/2609.05903) `arXiv:2609.05903` |  | 演化模型与领域专用的安全 harness |
| 2026-08 | 🔥 [AgentKernel](https://arxiv.org/abs/2609.29647) `arXiv:2609.29647` |  | 信任原生的智能体操作系统：处理跨信任边界的内容与动作 |
| 2026-08 | 🔥 [认知引发的风险](https://arxiv.org/abs/2608.15304) `arXiv:2608.15304` |  | 前沿智能体系统中类人认知偏差带来的风险 |
| 2026-08 | 🔥 [Agentic Transaction](https://arxiv.org/abs/2608.13900) `arXiv:2608.13900` |  | 走向满足 ACID 的智能体系统：可回滚的智能体操作 |
| 2026-08 | 🔥 [SkillJack](https://arxiv.org/abs/2608.03509) `arXiv:2608.03509` |  | 自演化智能体中的持久技能后门 |
| 2026-08 | 🔥 [OpenART](https://arxiv.org/abs/2608.00677) `arXiv:2608.00677` |  | 通过开放式环境演化扩展智能体红队测试 |
| 2026-07 | 🔥 [SeerGuard](https://arxiv.org/abs/2607.15550) `arXiv:2607.15550` |  | 用世界模型预测动作后果，保护手机 GUI 智能体 |
| 2026-07 | 🔥 [MemSyco-Bench](https://arxiv.org/abs/2607.01071) `arXiv:2607.01071` |  | 智能体记忆中的谄媚问题 |
| 2026-06 | 🔥 [Agentic Abstention](https://arxiv.org/abs/2606.28733) `arXiv:2606.28733` |  | 智能体知道何时应停止而不是继续行动吗 |
| 2026-06 | 🔥 [GateMem](https://arxiv.org/abs/2606.18829) `arXiv:2606.18829` |  | 多主体共享记忆智能体的记忆治理评测 |
| 2026-06 | 🔥 [SkillHarness](https://arxiv.org/abs/2606.20636) `arXiv:2606.20636` |  | 为计算机使用智能体约束安全的技能 |
| 2026-05 | 🔥 [AgentDoG 1.5](https://arxiv.org/abs/2605.29801) `arXiv:2605.29801` |  | 面向 OpenClaw 类开放世界智能体的轻量、可扩展安全对齐框架 |
| 2026-05 | 🔥 [审计智能体 harness 安全](https://arxiv.org/abs/2605.14271) `arXiv:2605.14271` |  | 执行 harness 分发工具、分配权限时的安全审计 |
| 2026-05 | 🔥 [MemPrivacy](https://arxiv.org/abs/2605.09530) `arXiv:2605.09530` |  | 边云协同智能体的隐私保护个性化记忆管理 |
| 2026-05 | 🔥 [DTap](https://arxiv.org/abs/2605.04808) `arXiv:2605.04808` |  | 可控、交互的智能体红队平台 |
| 2026-03 | ◆ [涌现的社会智能风险](https://arxiv.org/abs/2603.27771) `arXiv:2603.27771` |  | 生成式多智能体系统中的涌现风险 |
| 2026-03 | ◆ [T-MAP](https://arxiv.org/abs/2603.22341) `arXiv:2603.22341` |  | 轨迹感知的进化搜索红队测试 LLM 智能体 |
| 2026-02 | ◆ [Agents of Chaos](https://arxiv.org/abs/2602.20021) `arXiv:2602.20021` |  | 对自主语言模型智能体的探索性红队研究 |
| 2026-02 | ◆ [Moltbook 背后的恶魔](https://arxiv.org/abs/2602.09877) `arXiv:2602.09877` |  | 自演化 AI 社会中的安全性持续消失 |
| 2026-02 | ◆ [Spider-Sense](https://arxiv.org/abs/2602.05386) `arXiv:2602.05386` |  | 内在风险感知的高效智能体防御 |
| 2026-01 | ◆ [AgentDoG](https://arxiv.org/abs/2601.18491) `arXiv:2601.18491` |  | 智能体安全与安保的诊断式护栏框架 |
| 2025-10 | 🆕 [OS-Sentinel](https://arxiv.org/abs/2510.24411) `arXiv:2510.24411` |  | 混合验证增强手机 GUI 智能体的安全 |
| 2025-10 | 🆕 [Alignment Waltz](https://arxiv.org/abs/2510.08240) `arXiv:2510.08240` | Meta | 联合训练协作智能体以保证安全 |
| 2025-08 | 🆕 [A.S.E](https://arxiv.org/abs/2508.18106) `arXiv:2508.18106` |  | 仓库级 AI 生成代码安全性基准 |
| 2025-07 | 🆕 [OpenAgentSafety](https://arxiv.org/abs/2507.06134) `arXiv:2507.06134` | CMU | 评估真实世界智能体安全的综合框架 |
| 2025-06 | 🆕 [智能体失准](https://www.anthropic.com/research/agentic-misalignment) | Anthropic | 压力测试中多家模型在目标冲突或被替换威胁下会采取勒索等内部威胁行为 |
| 2025-06 | 🆕 [遵循策略的智能体的红队测试](https://arxiv.org/abs/2506.09600) `arXiv:2506.09600` |  | 对遵循业务策略的任务型智能体做有效红队 |
| 2025-04 | 🆕 [WASP](https://arxiv.org/abs/2504.18575) `arXiv:2504.18575` | Meta | 网页智能体抗提示注入的基准 |
| 2025-04 | 🆕 [X-Teaming](https://arxiv.org/abs/2504.13203) `arXiv:2504.13203` |  | 自适应多智能体的多轮越狱与防御 |
| 2025-03 | ★ 🆕 [CaMeL](https://arxiv.org/abs/2503.18813) `arXiv:2503.18813` | Google DeepMind | "从设计上"击败提示注入：控制流与数据流分离，用能力约束不可信数据 |
| 2024-12 | 🆕 [Agent-SafetyBench](https://arxiv.org/abs/2412.14470) `arXiv:2412.14470` | Tsinghua | 智能体安全评测 |
| 2024-10 | [AgentHarm](https://arxiv.org/abs/2410.09024) `arXiv:2410.09024` | Gray Swan·UK AISI | 衡量智能体有害性的基准 |
| 2024-10 | [Agent Security Bench](https://arxiv.org/abs/2410.02644) `arXiv:2410.02644` |  | 形式化并评测智能体攻防 |
| 2024-06 | ★ [AgentDojo](https://arxiv.org/abs/2406.13352) `arXiv:2406.13352` | ETH Zurich | 提示注入攻防的动态评测环境 |
| 2024-03 | [InjecAgent](https://arxiv.org/abs/2403.02691) `arXiv:2403.02691` | UIUC | 工具集成智能体的间接提示注入基准 |
| 2024-01 | [R-Judge](https://arxiv.org/abs/2401.10019) `arXiv:2401.10019` | SJTU | 智能体安全风险意识基准 |
| 2023-09 | [ToolEmu](https://arxiv.org/abs/2309.15817) `arXiv:2309.15817` | Toronto | 用 LM 模拟沙箱识别智能体风险 |
| 2023-02 | ★ [间接提示注入](https://arxiv.org/abs/2302.12173) `arXiv:2302.12173` | CISPA | 通过网页等外部内容间接注入指令攻陷 LLM 集成应用 |

### 效率与成本（12）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [TAO](https://arxiv.org/abs/2610.37230) `arXiv:2610.37230` |  | 面向智能体服务的工具感知联合调度（GPU 推理与外部工具交替） |
| 2026-10 | 🔥 [推理力度的经济学](https://arxiv.org/abs/2610.34491) `arXiv:2610.34491` |  | 智能体该选哪档推理力度（effort）：成本与成功率的权衡 |
| 2026-10 | 🔥 [DLite](https://arxiv.org/abs/2610.23787) `arXiv:2610.23787` |  | 无需历史的推测解码，加速智能体推理 |
| 2026-10 | 🔥 [轨迹级推测执行](https://arxiv.org/abs/2610.21463) `arXiv:2610.21463` |  | 为工具调用智能体做轨迹级推测与语义验证以降延迟 |
| 2026-10 | 🔥 [EdgeWise](https://arxiv.org/abs/2610.17560) `arXiv:2610.17560` |  | 端侧长程智能体：选择性地把难步骤委托给云端 |
| 2026-10 | 🔥 [Harness 感知蒸馏](https://arxiv.org/abs/2610.02858) `arXiv:2610.02858` |  | 把在 harness 中的行为蒸馏进小模型智能体 |
| 2026-05 | 🔥 [Mix-Quant](https://arxiv.org/abs/2605.20315) `arXiv:2605.20315` |  | 量化预填充、精确解码：面向智能体的推理加速 |
| 2026-03 | ◆ [SpecEyes](https://arxiv.org/abs/2603.23483) `arXiv:2603.23483` |  | 推测式感知与规划加速智能体多模态模型 |
| 2026-02 | ◆ [DualPath](https://arxiv.org/abs/2602.21548) `arXiv:2602.21548` | DeepSeek | 打破智能体推理中的存储带宽瓶颈（KV 缓存读写） |
| 2026-01 | ◆ [走向高效智能体](https://arxiv.org/abs/2601.14192) `arXiv:2601.14192` |  | 记忆、工具学习与规划的效率综述 |
| 2025-07 | 🆕 [Efficient Agents](https://arxiv.org/abs/2508.02694) `arXiv:2508.02694` |  | 在降低成本的同时构建有效智能体 |
| 2025-05 | 🆕 [蒸馏智能体到小模型](https://arxiv.org/abs/2505.17612) `arXiv:2505.17612` | KAIST | 用检索与代码工具把 LLM 智能体蒸馏进小模型 |

### 工程指南与博客（14）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-11 | ★ ◆ [长时运行智能体的 harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | Anthropic | 初始化智能体 + 增量编码智能体、进度文件与 git 记录，让智能体跨多个上下文窗口持续工作 |
| 2025-10 | 🆕 [Agent Skills 工程博客](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) | Anthropic | 渐进式披露：先只加载技能名与描述，需要时再读全文与脚本 |
| 2025-09 | 🆕 [用 Claude Agent SDK 构建智能体](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) | Anthropic | "给智能体一台电脑"：收集上下文 → 采取行动 → 验证工作的循环 |
| 2025-09 | ★ 🆕 [上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Anthropic | 把上下文当作有限资源：最小高信号 token 集、即时检索、压缩、结构化笔记与子智能体 |
| 2025-09 | ★ 🆕 [为智能体编写工具](https://www.anthropic.com/engineering/writing-tools-for-agents) | Anthropic | 工具要少而精、命名清晰、返回有意义的上下文与可操作的错误信息，并用评测迭代 |
| 2025-07 | ★ 🆕 [Manus 的上下文工程](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | Manus | 围绕 KV 缓存设计上下文、屏蔽而非删除工具、文件系统当外部记忆、保留错误 |
| 2025-06 | ★ 🆕 [致命三要素](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) | Simon Willison | 私有数据 + 不可信内容 + 对外通信同时存在时，提示注入几乎必然导致数据外泄 |
| 2025-06 | ★ 🆕 [多智能体研究系统](https://www.anthropic.com/engineering/multi-agent-research-system) | Anthropic | 主智能体 + 并行子智能体做研究：token 用量大约是聊天的 15 倍，适合可并行的宽查询 |
| 2025-06 | 🆕 [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) | Cognition | 共享完整上下文、避免冲突决策：多数场景单线程智能体更可靠 |
| 2025-04 | 🆕 [Claude Code 最佳实践](https://www.anthropic.com/engineering/claude-code-best-practices) | Anthropic | CLAUDE.md、先探索再规划再编码、测试驱动、多会话并行等编码智能体用法 |
| 2025-04 | ★ 🆕 [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) | OpenAI | 何时该做智能体、模型/工具/指令三件套、单智能体与多智能体编排、护栏 |
| 2025-04 | 🆕 [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) | HumanLayer | 把智能体当软件工程：自己掌控提示、上下文窗口与控制流 |
| 2024-12 | ★ 🆕 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | Anthropic | 先用最简单的方案：区分工作流与智能体，五种工作流模式（链、路由、并行、编排者—工人、评估—优化） |
| 2023-06 | ★ [LLM 驱动的自主智能体](https://lilianweng.github.io/posts/2023-06-23-agent/) | Lilian Weng | 规划、记忆、工具三要素的经典入门长文 |

### 产品与平台（18）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Codex 更新日志](https://developers.openai.com/codex/changelog) | OpenAI | Codex CLI 与云端功能的持续更新记录 |
| 2026-10 | 🔥 [Claude Code 更新日志](https://code.claude.com/docs/en/changelog) | Anthropic | Claude Code 的持续更新记录 |
| 2026-09 | 🔥 [OpenAI DevDay 2026](https://openai.com/index/devday-2026-recap/) | OpenAI | Codex 后台/云端任务与 /agents 视图、Agents API 支持计算机使用与多智能体、常驻个人智能体 |
| 2026-05 | 🔥 [Google I/O 2026 开发者要点](https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-developer-highlights/) | Google | Antigravity 2.0（并行多智能体、动态子智能体、定时任务）、Antigravity CLI 与 SDK |
| 2026-05 | 🔥 [Gemini CLI 迁移到 Antigravity CLI](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/) | Google | 2026-06-18 起 Gemini CLI 对个人用户停止服务，迁移到 Antigravity CLI |
| 2026-05 | 🔥 [Gemini API 托管智能体](https://blog.google/innovation-and-ai/technology/developers-tools/managed-agents-gemini-api/) | Google | 一次 API 调用即可开出带沙箱、代码执行与浏览的远程智能体环境 |
| 2026-04 | ◆ [Claude Managed Agents](https://claude.com/blog/claude-managed-agents) | Anthropic | 托管的智能体 harness：沙箱、内置工具、记忆与多智能体编排的组合式 API |
| 2025-11 | ◆ [Google Antigravity](https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/) | Google | 智能体优先的开发平台：编辑器 + 管理多个智能体的工作台 |
| 2025-11 | ★ ◆ [OpenClaw](https://github.com/openclaw/openclaw) | OpenClaw 社区 | 开源、自托管的个人智能体（前身 Clawdbot）：通过聊天软件下达任务、执行命令，2026 年迅速流行 |
| 2025-10 | 🆕 [ChatGPT Atlas](https://openai.com/index/introducing-chatgpt-atlas/) | OpenAI | 内置智能体模式的浏览器 |
| 2025-07 | 🆕 [ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) | OpenAI | 把 Operator 与 Deep Research 合并为带虚拟电脑的通用智能体 |
| 2025-06 | 🆕 [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Google | 开源终端智能体；2026-06-18 起面向个人用户迁移到 Antigravity CLI |
| 2025-05 | 🆕 [Codex（云端）](https://openai.com/index/introducing-codex/) | OpenAI | 在云端沙箱中并行处理多个编码任务的软件工程智能体 |
| 2025-04 | 🆕 [Codex CLI](https://github.com/openai/codex) | OpenAI | 开源的终端编码智能体 |
| 2025-02 | ★ 🆕 [Claude Code](https://code.claude.com/docs/en/overview) | Anthropic | 终端中的智能体编程工具：读写仓库、运行命令、管理 git |
| 2025-02 | 🆕 [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/) | OpenAI | 用 RL 训练的浏览与综合研究智能体，开启"深度研究"产品潮 |
| 2025-01 | 🆕 [Operator](https://openai.com/index/introducing-operator/) | OpenAI | 在浏览器中替用户完成任务的智能体（后并入 ChatGPT agent） |
| 2024-10 | [Claude 计算机使用](https://www.anthropic.com/news/3-5-models-and-computer-use) | Anthropic | 首个公开的前沿模型计算机使用能力（看截图、移动鼠标、键入） |

### 综述（11）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Work Agents](https://arxiv.org/abs/2610.35845) `arXiv:2610.35845` |  | 工作型智能体综述：系统、能力与优化 |
| 2026-08 | 🔥 [智能体式产物创建](https://arxiv.org/abs/2608.28122) `arXiv:2608.28122` |  | 智能体创建文档、幻灯片、代码等产物：系统、评测与原则 |
| 2026-06 | 🔥 [智能体 AI 搭车客指南](https://arxiv.org/abs/2606.24937) `arXiv:2606.24937` |  | 从基础到系统的智能体 AI 实践者参考手册 |
| 2026-06 | 🔥 [对智能体模型的批判](https://arxiv.org/abs/2606.23991) `arXiv:2606.23991` |  | 什么是智能体、什么构成能动性 |
| 2026-06 | 🔥 [从聊天机器人到数字同事](https://arxiv.org/abs/2606.14502) `arXiv:2606.14502` |  | 走向持久自主 AI 的范式转变 |
| 2025-12 | ◆ [智能体 AI 的适配](https://arxiv.org/abs/2512.16301) `arXiv:2512.16301` |  | 智能体 AI 系统适配方法综述 |
| 2025-03 | 🆕 [基础智能体：进展与挑战](https://arxiv.org/abs/2504.01990) `arXiv:2504.01990` |  | 从类脑智能到演化、协作与安全的基础智能体综述 |
| 2025-03 | 🆕 [LLM 智能体方法综述](https://arxiv.org/abs/2503.21460) `arXiv:2503.21460` |  | LLM 智能体的方法、应用与挑战 |
| 2024-01 | [Agent AI](https://arxiv.org/abs/2401.03568) `arXiv:2401.03568` | Microsoft·Stanford | 多模态交互智能体的前景综述 |
| 2023-09 | [LLM 智能体的兴起](https://arxiv.org/abs/2309.07864) `arXiv:2309.07864` | Fudan | LLM 智能体综述 |
| 2023-08 | [LLM 自主智能体综述](https://arxiv.org/abs/2308.11432) `arXiv:2308.11432` | Renmin | 基于 LLM 的自主智能体综述 |

