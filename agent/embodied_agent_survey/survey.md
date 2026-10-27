# 具身智能 / 机器人 × Agent（LLM · GPT · Claude · Gemini）调研综述

> 截至 **2026-10-23**。面向机器人方向的研究者，重点是 2025–2026 年的工作。
> 配套技术教程（按核心问题讲清机制、公式与设计取舍）：[机器人 × Agent 技术教程](../embodied_agent_tutorial/)。
> 配套交互页面：`index.html`（同目录，浏览器直接打开）。页面里有术语入门、技术栈图、时间线、图表，以及可检索的论文库。
> arXiv 编号与标题都用 export.arxiv.org API 逐条核验过；产业信息优先引用官方博客，二手来源会注明。

---

## 0. 一页结论（TL;DR）

1. **三次范式迁移。**
   - **2022–2023**：LLM 当"高层大脑"，负责选技能、写代码（SayCan、Code as Policies、VoxPoser），底层仍是传统控制或学习技能。
   - **2023–2025**：端到端 VLA 成为主线（RT-2 → OpenVLA → π0/π0.5 → Gemini Robotics → GR00T），同时出现 System 2/System 1 分层（Hi Robot、Helix、Gemini Robotics 1.5）。
   - **2025–2026**：进入"**agentic robots / harness 工程**"阶段。做法是把 Claude Code、Codex 这类编码 agent 的循环（工具调用、记忆、监控恢复、技能库、子 agent）整套搬到机器人上，VLA 退为 agent 可调用的一个工具。
2. **2026 年最大的变化是"通用前沿模型直接上机器人"。** GPT-6 Astra（2026-09）、Claude Opus 5/5.5、Gemini 4 Argon（2026-10）在没有机器人专用训练的情况下，通过写代码、调用工具或直接输出数值动作，就能完成相当一部分操作和导航任务：
   - RoboDojo 42 个任务上，GPT-6 Astra 平均成功率 22.48%，超过全部 40 个公开策略；
   - 用 Claude Code（Opus 5）的 OpenRUA 在 CaP-Bench 上达到 99%。

   但更难的通用基准仍然很低：
   - RobotWorld 84 个任务里，最强模型只完成 19%；
   - RoboQuest 上最好的 agent 只有 23%。
3. **短板一致，集中在三处。**
   - **实时性与精细接触**：Anthropic 估计实时直接控制需要约 83Hz，而推理只有 0.2–0.4Hz；
   - **可靠的物理状态验证**；
   - **成本**：一次 30 秒的步态控制要调用 250 次模型、每次约 40 秒，RoboDojo 50 个实例消耗 6–11 亿 token。

   结论是 LLM/agent 适合放在慢环（规划、编排、写代码、恢复），快环交给 VLA 和控制器。
4. **"Harness 决定上限"已有大量证据。** 同一个模型换一种接口，成功率可以差几十个点：
   - Show-Harness 把步长减半，零样本成功率从 60% 提到 82%；
   - 只加执行引擎，ASPIRE 在 LIBERO-Pro 上从 14% 提到 62%；
   - Physical Agency 让冻结的 VLA 在 LIBERO-PRO 上从 12.8% 提到 53.3%。

   2026-10 发布的综述 *Towards Agentic Robots*（arXiv 2610.20442）把这一方向系统化为 Model / Data / Environment / Harness 四域。
5. **编码 agent 正在自动化机器人研发本身。**
   - Anthropic Project Fetch 第二阶段：Claude Opus 4.7 独立给机器狗编程，比最快的人类团队快约 19 倍；
   - ENPIRE：编码 agent 在真机上自动训练灵巧策略到 99%；
   - ASPIRE、SimEX、GaP、RATs：自己写技能、修技能、在仿真里练习。

   反面证据同样重要：一项 123 轮的自改进实验发现，**改进并不会自动累加**，评估器和感知模块才是瓶颈。
6. **工业界的形态是"两层"。** 一层是 ER/VLM 编排者，如 Gemini Robotics ER 2（Live API 流式编排、工具调用、多机协作）；另一层是 VLA 或 WAM 执行者，如 π0.7、Gemini Robotics 2、GR00T N1.7/N2（基于 DreamZero 的世界-动作模型）、Helix 02（新增 1kHz 的 System 0）。
   - 模型正在 API 化：Gemini Robotics-ER API、Pi API beta（2026-10-13）、Rho-alpha 早期访问。
   - 接口在标准化：MCP / ROS MCP Server、OpenClaw/ROSClaw、Anthropic 的 Model Hardware Standard（2026-08）。
7. **安全是真问题。**
   - RoboPAIR 证明 LLM 控制的机器人可以被越狱；
   - 编码 agent 会撞上明令禁止触碰的障碍物（SafeHarness）；
   - 不同模型提出越界动作的比率最多相差 4.8 倍（ROSClaw）。

   共识是安全闸门放在模型外部：白名单、限速、时序逻辑检查、人工确认。

---

## 1. 先对齐概念：机器人研究者需要知道的 Agent 知识

### 1.1 一句话定义

**Agent = 模型（LLM/VLM）+ 循环（观察→思考→调用工具→看结果→再决定）+ 工具 + 记忆 + 外壳（harness）。**
如果把 VLA 看作"会看、会动的策略网络"，那 agent 就是**任务层的闭环控制器**。它的"动作"是函数调用和程序，"传感器"是工具返回值和图像，"控制周期"以秒计。

### 1.2 术语对照表

| Agent 术语 | 含义 | 机器人里的类比 | 例子 |
|---|---|---|---|
| LLM / VLM / MLLM | 能读文本（和图像）的通用推理模型 | 读过所有教科书和说明书、但没有身体的工程师 | GPT-6、Claude Opus 5.5、Gemini 4 |
| Agent loop | 观察→推理→行动→看反馈的循环（ReAct） | 任务层的 sense-plan-act 闭环 | Claude Code、Codex CLI、OpenClaw |
| Tool / Function calling | 模型输出结构化调用（函数名 + 参数），由外部执行并返回结果 | ROS service / action 调用 | `detect("cup")`、`move_to(pose)` |
| Harness（外壳/脚手架） | 包在模型外面的软件，决定模型能看到什么、能调用什么、怎样检查结果、怎样复用经验 | 机器人软件栈：驱动、状态机、监控、日志 | Thea、ETA、RoboClaw、OpenRUA |
| Context window | 模型一次能"看到"的全部输入 | 工作内存；放不下就要压缩或检索 | 长任务需要摘要和记忆 |
| Memory | 跨步骤、跨任务保存的信息 | 地图、语义地图、场景图、失败日志 | ConceptGraphs、MessyMem、HiMe |
| Skills / Skill library | 可复用的程序或说明文档 | 运动基元库 / 行为库 | Voyager、ASPIRE、Uni-Skill |
| Code as Policy | 由模型写程序组合感知与控制 API，程序本身就是策略 | 工程师写的任务脚本 | Code as Policies、CaP-X |
| Coding agent | 能在终端里读写代码、运行、测试、迭代的 agent | 自动化的机器人软件工程师 / 研究助理 | Claude Code、Codex |
| MCP（Model Context Protocol） | 连接模型与外部工具/数据的开放协议 | 类似 rosbridge 或驱动接口标准 | ROS MCP Server、RoboNeuron、MHS |
| Reflection / Recovery | 失败后诊断原因、修改计划 | 故障检测、隔离与恢复（FDIR） | Inner Monologue、REFLECT、DoReMi |
| In-context learning | 不更新权重，只靠提示里的示范学会新任务 | 看几次示范当场调整，不重新训练 | RoboICL、GPT-Policy |
| System 2 / 1 / 0 | 慢思考规划 / 快速视觉运动策略 / 解析控制器 | 任务规划器 / 学习策略 / IK·MPC·WBC | Helix 02、GR00T、Gemini Robotics 1.5 |
| VLA | 把 VLM 微调成直接输出动作的策略 | 语言条件的端到端视觉运动策略 | RT-2、OpenVLA、π0.5 |
| World model / WAM | 预测未来观测（以及动作）的模型 | 可学习的动力学模型 / 神经仿真器 | Genie 3、Cosmos、DreamZero |
| Subagent / Multi-agent | 主 agent 分派任务给子 agent，或多个 agent 协作 | 分布式架构、多机器人协同 | RoCo、EMOS、EMERGE-Policy |
| Post-training（SFT/RL） | 用监督微调或强化学习提升模型自身的 agent 能力 | 策略微调 | AgenticRoboRL、CaP-RL |
| Guardrail / Safety gate | 模型输出到达硬件前的检查 | 安全 PLC、速度限幅、急停 | RoboGuard、ROSClaw、SafeGate |

### 1.3 三类系统怎么选

| 维度 | 端到端 VLA | Agent + 工具/技能（harness） | 编码 agent（代码即策略 / 自动研发） |
|---|---|---|---|
| 输出 | 连续动作块（10–50Hz，Helix S1 为 200Hz） | 工具/技能调用（秒级） | 程序、训练代码、配置 |
| 能力来源 | 机器人演示 + VLM 预训练 | 前沿模型的常识推理 + 现成技能 | 前沿模型的编程能力 + 执行反馈 |
| 新任务适配 | 采数据、微调 | 改提示、加工具、写技能 | 让 agent 写或改代码，少量试错 |
| 强项 | 接触丰富、灵巧、实时 | 开放语义、长时程、失败恢复、可解释 | 工程自动化，结果可复用、可验证 |
| 弱项 | 长时程、分布外指令、数据昂贵 | 延迟高、精细控制差、依赖接口质量 | 依赖感知和原语质量，评估器可能被"钻空子" |
| 代表 | π0.5/π0.7、Gemini Robotics 2、GR00T、Helix | Gemini Robotics 1.5 / ER 2、RoboClaw、Thea、ETA | CaP-X、OpenRUA、ENPIRE、ASPIRE |

实际系统往往三者都有：VLA 作为 agent 的工具，程序作为上层，编码 agent 在离线阶段改进前两者。

---

## 2. 全景：八条技术路线

### 2.1 LLM/VLM 作为高层规划器（2022–）

- **思路**：语言模型分解任务，从技能库里选下一步；技能本身由传统方法或 RL/IL 获得。
- **代表**：
  - SayCan：LLM 有用性 × 价值函数可行性；
  - Inner Monologue：把感知反馈作为文本回灌；
  - LLM-Planner、Text2Motion、LLM+P / AutoTAMP：接上经典规划和 TAMP；
  - SayPlan / ConceptGraphs / VLMaps：用场景图或语义地图 grounding；
  - KnowNo：共形预测，不确定时求助；
  - TidyBot：个性化。
- **遗产**：后来所有分层系统里的"System 2"都源于这条线；"规划器 + 可调用技能"也成了 2026 年 harness 的基本形态。
- **局限**：能力受限于技能库，规划与物理可行性之间有鸿沟，闭环反馈很粗。

### 2.2 代码即策略 → 编码智能体（2022–2026）

- **思路**：动作空间不是关节角，而是程序。LLM 组合感知 API、控制 API 和数学库来写策略。
- **早期**：
  - Code as Policies；
  - ProgPrompt；
  - Microsoft 的 ChatGPT for Robotics（API 库 + 提示 + 人在环）；
  - VoxPoser（LLM 写代码构造 3D 价值图）；
  - ReKep（VLM 写关键点约束、求解器实时优化）；
  - MOKA。
- **2026 年的编码 agent 化**：
  - **CaP-X**（ICML'26）：系统化的基准与框架，含 CaP-Gym（39 个任务）、CaP-Bench（S1–S4 单轮、M1–M4 多轮共 8 档）、CaP-Agent0、CaP-RL。核心发现是人类提供的高层抽象越少，模型越吃力；多轮交互、视觉差分、自动技能合成和集成推理能把差距补回来。CaP-RL 在真实 Franka 上做到抬方块 84%、堆叠 76%。
  - **RHO**：训练期由编码 agent 搜索多文件的"仓库即策略"。LIBERO-PRO 45%，π0.5 只有 12.83%。
  - **Agent as Policy (AGP)**：Codex CLI + GPT-6 Astra 直接驱动真实 YAM 机械臂完成装配、抛掷、叠毛巾等任务。
  - **OpenRUA**：只给 Claude Code 一个 ROS 2 终端（"零抽象"）。CaP-Bench 99%、LIBERO-PRO 87%；96.8% 的 episode 里 agent 自发写出感知处理程序。
  - **RobotUse**：agent 只做视觉选择，后端负责几何与控制。RoboLab 45%，比 CaP-X 高 6.7pp。
  - **EmbodiedSWE**：编码 agent 解半小时级灵巧任务，再把解扩增为 VLA 训练数据。
  - **RACaP**：编码移到进化期，部署时调用冻结的 Policy API，解决延迟问题。
- **要点**：代码可检查、可验证、可复用，天然适合"写一次、跑多次"；最大的瓶颈在感知和原语质量。Kuroki & Tang 的实验里，同样的 agent 在仿真特权状态下成功率 65%，只用 RGB 时降到 22.5%。

### 2.3 LLM 生成奖励、仿真与数据（2023–2026）

- **奖励**：
  - Eureka：LLM 写奖励代码并进化迭代，多数任务超过专家；
  - DrEureka：同时生成奖励和域随机化，做 sim2real；
  - Text2Reward、Language to Rewards、CurricuLLM：课程生成。
- **仿真与任务**：
  - RoboGen（生成式仿真）、GenSim/GenSim2、Holodeck；
  - 2026 年的 agentic 版本：SceneSmith（ICML'26 Spotlight）、SimWorld Studio、Articraft（铰接资产）、SimFoundry、Agentic Real2Sim、ARSTAG（单张图 + 一句指令做 Real2Sim2Real，π0.5 真机 74.6%）、EmbodiedSmith。
- **数据**：
  - AutoRT：VLM/LLM 编排 20 多台机器人采数据，带"机器人宪法"；
  - RoboClaw：正逆动作对实现自复位采集；
  - EmbodiedSWE-Gen、HARBOR（RL 工作流全自动化）；
  - Gemini 4 Argon for Robotics：从视频或草图生成 MuJoCo 孪生与仿真数据，把真机成功率从 56% 提到 75%。

### 2.4 VLA 与机器人基础模型（2023–2026）

- **谱系**：RT-1 → RT-2（动作即 token，提出 VLA）→ Open X-Embodiment → Octo / OpenVLA → π0（flow matching 动作专家）→ FAST / OpenVLA-OFT → π0.5（开放世界、子任务文本）→ π*0.6（RECAP，从经验中 RL）→ **π0.7**（2026-04，可用多模态上下文"操控"，约 5B）。
- **Google DeepMind**：Gemini Robotics（2025-03）→ On-Device（2025-06）→ **1.5**（2025-09，"先思考再行动" + Motion Transfer，ER 1.5 编排）→ ER 1.6（2026-04）→ **Gemini Robotics 2 系列**（VLA、ER 2、On-Device 2；ER 2 于 2026-07-30 发布）。
- **NVIDIA**：GR00T N1（2025-03，双系统）→ N1.5 → N1.6（Cosmos-Reason-2B 骨干 + 32 层 DiT）→ N1.7（GTC 2026，商业授权早期访问）→ **N2 预览**（基于 DreamZero 的世界-动作模型）。
- **其他**：
  - Figure：Helix（2025-02）→ Helix 02（2026-01，新增 1kHz 的 System 0）；
  - 字节 GR-3 / Robix；AI2 MolmoAct / MolmoAct2；Hugging Face SmolVLA；
  - X-VLA、WALL-OSS、EO-1、InternVLA-M1、UniVLA、RoboBrain 2.0；
  - Qwen-RobotManip（2026-06）、腾讯 HY-Embodied-0.5、AgiBot GO-2；
  - Generalist GEN-0/GEN-1（扩展规律）、Skild S1、微软 Rho-alpha（触觉 + 持续学习）。
- **RL 后训练**：π*0.6/RECAP、RL Token（几小时在线 RL 把关键阶段提速最多 3 倍）、πRL、Learning While Deploying（舰队规模 RL）、TORL-VLA（触觉）等。
- **世界模型 / WAM**：
  - V-JEPA 2（Meta）、Genie 3；
  - **DreamZero**（14B，7Hz 闭环，泛化比 VLA 高 2 倍以上；2026-04 RoboArena 1750 Elo，同期 π0.5 为 1622）；
  - Cosmos 3、1X World Model、DYNA-2（100 万小时以上人类视频）、GE-Act 2.0。
- **与 agent 的关系**：VLA 正在成为 agent 的"手"。对 VLA 研究者来说，**可操控性（steerability）**和**可终止性**变成新的设计目标，因为 agent 需要能精确地命令它、打断它。π0.7 的"steerable"、Hi-VLA 研究中的终止规则、InSight 的"原语可操控性"都是这个方向。

### 2.5 分层编排：System 2 → System 1 → System 0（2024–2026）

- **集成式双系统**：
  - Helix：7–9Hz 的 VLM 潜变量驱动 200Hz 策略；
  - GR00T N1：10Hz 推理 + 120Hz 动作块；
  - π0.5：同一模型先解码子任务；
  - Hi Robot：每秒一次语言子任务，支持用户插话纠正。
- **把 VLA 当工具的组装式分层**：
  - **Gemini Robotics 1.5**：ER 1.5 用语言指令调用 VLA，必要时调用搜索工具；换成具身推理编排者后总失败率减半；
  - **ER 2**：Live API 流式编排，"边执行边思考"，可看视频跟踪进度、调用工具、协调多台机器人；
  - VLAs-as-Tools、Harness VLA、Physical Agency (Pigey，LIBERO-PRO 12.8%→53.3%)、VoLo（VLA/WAM 作可中断工具）、RoboHarness、RoboRouter、Critic in the Loop（三系统）、Cortex（32 种规范技能原语）。
- **设计经验**：
  - Hi-VLA 系统研究：**终止规则与 VLA 的可操控性和规划器大小同样重要**；
  - VLA-OS：低层策略对视觉计划比对语言计划更"听话"；
  - 潜变量接口信息多但难以监控，文本、路径、路点可检查但表达力有限。

### 2.6 Agentic Harness 与机器人 Agent OS（2025–2026）

借用 *Towards Agentic Robots* 的拆法，harness 有六个功能：

| 功能 | 解决什么 | 代表工作 |
|---|---|---|
| 编排 Orchestration | 谁来执行下一步、参数怎么绑定、何时交还控制权 | VoLo、EMOS（从 URDF 生成"机器人简历"）、Physical Agency、ROSClaw、Thea、RoboHarness、HarnessWAM、VLAs-as-Tools |
| 记忆 Memory | 写什么、何时更新、检索什么 | MessyMem、HiMe、Mimir、SafeMem、NavMCP、BATON、HoloAgent-0、MeCo |
| 监控与恢复 | 执行前、中、后的检查与修复 | DoReMi、Code-as-Monitor、REFLECT、AHA、LIBERO-RECOVER、PhyAgentOS、FLARE |
| 技能 Skills | 可复用行为如何进入、演化、退出 | Voyager、DROC、Uni-Skill、ASPIRE、RATs、RoboRSI、SkillOpt、Zetta |
| 基础设施 | 接口、工具链、策略服务、运行时闸门 | ROS MCP Server、RoboNeuron（MCP↔ROS 2）、ETP（100+ 工具）、openpi 远程推理、LeRobot、RTC/VLASH（异步执行）、ROSA/Robion（舰队推理）、vla.cpp（统一推理运行时）、OxyGen（KV Cache 管理）、BitVLA（1-bit VLA）、RoboECC（边云协同部署）、VLA-Perf（性能评测） |
| 分层系统 | 规划器、学习策略、解析控制之间的权限划分 | Helix、GR00T、Hi Robot、Gemini Robotics 1.5、Harness VLA |

**"机器人 Agent OS"与开源运行时**：
- OpenClaw 生态：ROSClaw、RoboClaw、PHILIA，以及 Intel 的 OpenClaw + AgenticROS 示例；
- ETA/OpenETA（可作为 Codex 插件）、Thea（"场景图即上下文""评估即退出码"）；
- PhyAgentOS（解耦认知规划与物理执行的自我进化 OS）、ABot-AgentOS、RoboOS、Show-Harness、Guava（把 harness 能力蒸馏到 4B 模型）、Robo-Harness K1（感知即工具）；
- DynaHarness（2026-09，动态物理 harness 用于自我进化机器人 agent）；
- 空中与集群：AeroWeaver（空中集群技能编织）、Harness Robotic OS（四足巡检）。

**接口标准化与推理基础设施**：
- ROS MCP Server（2025-09 开源）：机器人端加一个 rosbridge，Claude Code / Codex / Gemini CLI 就能发现并调用 topic、service、action；
- RoboNeuron（2025-12）：MCP 与 ROS 之间的中间层基础设施，用于 agent 驱动的编排；
- Anthropic 的 Model Hardware Standard（2026-08，研究预览）：读/写命令 + 安全参数参考文件，经 MCP/CLI/API 控制机械臂、显微镜等设备，对其他模型同样开放；
- OpenAI 的 Decisions API（2026-10，gpt-6-luna）这类"决策模型"开始被用作 harness 里的快速决策层：代码给出合法候选，模型带概率挑一个；
- 推理优化：vla.cpp（统一推理运行时）、OxyGen（VLA 推理的 KV Cache 管理）、VLASH（未来状态感知的异步推理）、RTC（实时动作分块）、VLA-Perf（性能评测工具）。

### 2.7 编码智能体自动化机器人研发与自我改进（2026）

- **Anthropic Project Fetch**：
  - 2025-11：有 Claude 辅助的人类团队完成 7/8 项任务，对照组 6/8，共同任务用时约减半；
  - 2026-06 第二阶段：Opus 4.7 在 Claude Code 中独立完成，比最快人类团队快约 19 倍，但仍没实现自主"取回"。
- **ENPIRE**：自动复位 → 执行 → 验证 → agent 修改训练代码，灵巧任务训练到 99%；派出 agent 团队、使用机器人舰队时更快。
- **ASPIRE**：执行引擎 + 技能库 + 进化搜索；用 Claude Opus 5.5 时 LIBERO-Pro 80 个任务平均 99%，并能在真机上逐步更高效地解魔方。
- **其他**：
  - SimEX：仿真里练出工具箱，几次真机试次同时校正仿真器，每任务约 10 分钟真机时间；
  - GaP：图即策略，工业变型自动化；
  - RATs：玩耍式学习；
  - RPG、RoboRSI、Zetta、HarnessPAI：不同的自我改进机制；
  - AgenticRobotics：把 Claude Code / Codex 的主-子 agent 架构用于"训练-评估-改进"，强调证据门控；
  - AgenticRoboRL：后训练 9B 规划器，LIBERO-PRO 18.1%→71.0%，接近 Codex（GPT-5.5）+ 外部记忆的 72.1%。
- **反面教材**：*What Stops Recursive Self-Improvement in Robotics?* 跑了 123 轮自改进。agent 能自己发现"目标不在视野内"并部署主动视角模型，但改进不累加，目标任务始终没成功。三个原因：
  1. 分割模型能找到"架子"却找不到"最上层的架子"；
  2. 技能链的经验都堆在第一步；
  3. agent 精确地优化评估器量的东西，包括评估器出错的地方。

### 2.8 通用前沿模型直接当机器人策略（2026 下半年）

- **背景**：GPT-6 Astra（2026-09-03）能直接生成数值动作，社区在两个月内出现几十项"LLM 即策略"研究（汇总见 Awesome-Astra-Embodied-AI）；Anthropic、Google 也在同期发布了系统性评测和机器人专用能力。
- **主要结果**：

| 研究 | 设置 | 结果 |
|---|---|---|
| Claude plays robotics（Anthropic，2026-07） | 12 个模型 × 4 种接口（扭矩直控 / 写控制器 / 训 RL / 监督 VLA） | 越高层的接口提升越稳定；LIBERO 直接操控 0–5.5%；单用 MolmoAct 比任何"LLM 监督 MolmoAct"都强；G1 一次也没站起来 |
| Galbot GPT-as-Policy（2609.38537） | 6 个领域 | 导航 RxR 92%、HM3D 82%；与 π0.5 混合控制 RoboDojo 48%；步态 0/5；HumanoidBench 30 个任务中 13 个超过基线 |
| An Unexpected Robot Policy（2609.24170） | RoboDojo 42 个任务 | Astra 22.48%（第 1，超过全部 40 个公开策略），GPT-5.5 0.88%，DeepSeek-Flash 1.92% |
| GPT-6 as a Beginning（2610.28259） | 5 个操作基准 | RoboDojo 26.92%、RoboTwin 2.0 76.0%、LIBERO-GeoMotion 11.1% |
| LIBERO-10 frontier（2610.30051） | 6 个前沿模型 × 3 档推理 | Astra 44%、Gemini-4 Argon 43% |
| RobotWorld（2610.10409） | 84 个任务 / 5 个领域 | Astra 19.0%、Opus 5.5 15.5%、Kimi K3 2.4%、DeepSeek V4.1 Flash 1.2%、Gemini 3.8 Flash 1.2%；63 个任务无人完成 |
| RoboQuest（2610.10388） | 需主动获取信息的移动操作 | 最好 23%；微调 π0.5 几乎为 0；失败主因是"太早停止探索" |
| OpenRUA（2610.02459） | Claude Code（Opus 5）+ ROS 2 终端 | CaP-Bench 99.0%、LIBERO-PRO 87.0% |
| SenseNova（2610.22471） | 12 个基准 + 8 种真机配置 | 具身能力"基本具备"，实用具身智能"尚未" |
| HomeBody（Stanford/Caltech） | Astra 直接调用 5 个技能驱动 Unitree G1 | 能在陌生厨房整理取物；延迟、舵机过热、算力成本是瓶颈；未公布成功率 |
| Gemini 4 Argon for Robotics（2026-10-13） | 物理推理 + 写代码 | ERQA 87%；视频/草图→MuJoCo 孪生；仿真数据把真机成功率 56%→75% |

- **解读**：
  1. 语义与规划强，精度、动力学和双臂协调弱。
  2. 接口与 harness 是主要变量，Astranomy 发现扩展接口把成功率从 80% 提到 86%。
  3. 延迟和 token 成本仍高 1–2 个数量级。
  4. 真机安全问题突出：RoboDojo 的 Astra 评测因安全原因中止了真机测试。

---

## 3. 时间线（精选）

| 时间 | 事件 |
|---|---|
| 2022-04 | SayCan、Socratic Models |
| 2022-07 | Inner Monologue、LM-Nav |
| 2022-09 | Code as Policies、ProgPrompt |
| 2022-12 | RT-1、LLM-Planner |
| 2023-02 | ChatGPT for Robotics（Microsoft） |
| 2023-03 | PaLM-E |
| 2023-05 | Voyager（Minecraft，代码技能库） |
| 2023-07 | RT-2（VLA）、VoxPoser、RoCo、KnowNo |
| 2023-10 | Eureka、Open X-Embodiment、GenSim |
| 2023-11 | RoboGen |
| 2024-01 | AutoRT、OK-Robot |
| 2024-05/06 | Octo、OpenVLA |
| 2024-10 | π0、RoboPAIR（越狱 LLM 机器人）、Embodied Agent Interface |
| 2025-02 | Figure 结束与 OpenAI 合作并发布 Helix；Hi Robot；EmbodiedBench |
| 2025-03 | Gemini Robotics + ER、GR00T N1、ASIMOV |
| 2025-04 | π0.5 |
| 2025-06 | Gemini Robotics On-Device、SmolVLA、V-JEPA 2 |
| 2025-08 | Genie 3 |
| 2025-09 | ROS MCP Server 开源；Gemini Robotics 1.5 / ER 1.5（agentic 编排） |
| 2025-10 | Butter-Bench、LIBERO-PRO |
| 2025-11 | Anthropic Project Fetch；π*0.6（RECAP）；GEN-0；SIMA 2 |
| 2026-01 | Helix 02；Microsoft Rho-alpha；NVIDIA Alpamayo；Skild AI 融资约 14 亿美元 |
| 2026-02 | DreamZero（世界-动作模型） |
| 2026-03 | CaP-X、RoboClaw、ROSClaw；GTC：GR00T N1.7 / N2 预览、Cosmos 3；PI 的 MEM 与 RL Token |
| 2026-04 | π0.7；Gemini Robotics-ER 1.6；GEN-1；AgiBot GO-2 |
| 2026-05/06 | OpenAI 成立机器人部门；Project Fetch 第二阶段；RHO；ENPIRE；Hi-VLA 设计研究 |
| 2026-07 | Claude plays robotics；ASPIRE；GaP；Physical Agency；Gemini Robotics 2 / ER 2（7/30） |
| 2026-08 | Thea、ETA、AgenticRobotics loop；Anthropic MHS（8/27）；DYNA-2；Skild S1 |
| 2026-09 | GPT-6 Astra（9/3）引发"LLM 即策略"研究潮；HomeBody；SimEX；OpenAI 表态做人形 |
| 2026-10 | Pi API beta、Gemini 4 Argon for Robotics（10/13）；RobotWorld、RoboQuest、OpenRUA；综述 *Towards Agentic Robots*（10/14）；AgenticRoboRL |

### 2.9 多机器人协作与专项领域（2023–2026）

#### 多机器人/多智能体

- **编排与协作**：
  - **EMOS**（2024-10，ICLR'25）：从 URDF 自动生成"机器人简历"，本体感知的异构多机器人 OS；
  - **RoCo**（2023-07）：多机械臂通过 LLM 对话协商计划，执行前做可行性检查；
  - **EmboTeam**（2026-01）：LLM 推理经 PDDL 落地成反应式行为树，用于多机协作；
  - **MeCo**（2026-01）：相似任务记忆化提升 LLM 多机协作效率；
  - **RoboTalk**（2026-09）：从多模态演示学习多机器人通信与协调。
- **专项应用**：
  - **AeroWeaver**（2026-09）：空中集群的具身 agent harness，编织技能并分配角色；
  - **MistyPilot**（2026-08）：多 agent LLM 技能编排控制社交机器人；
  - **IMR-LLM**（2026-03）：工业多机器人任务规划与程序生成；
  - **M2HRI**（2026-04）：LLM 驱动的多模态多 agent 个性化人机交互。

#### 导航专项

- **统一架构**：
  - **ABot-N0/N1**（Alibaba AMAP，2026）：统一 5 类导航任务的 VLA，LLM 认知大脑 + 流匹配动作专家；
  - **HarnessVLN**（AgiBot，2026-09）：免训练的统一导航 agent harness；
  - **NavMCP**（SJTU，2026-08）：VLM agent 决定找什么、去哪找、何时停，导航基础模型负责执行，长时程探索 78.3%。
- **空中导航**：**MissionBench**（2026-07）：空中 MLLM agent 的零样本任务级评测。

#### 人形机器人专项

- **全身控制**：
  - **WholeBodyVLA**（2025-12，ICLR'26）：统一潜在 VLA 的全身移动操作；
  - **Cybo-Waiter**（2026-03）：人形全身移动操作的物理 agent 框架；
  - **HumanCLAW**（2026-07）：VLM 动作决策与底层全身运动解耦的评测框架。

#### 自动驾驶

- **Alpamayo 1.5**（NVIDIA，2026-03）：开放的 10B 推理型驾驶 VLA，Cosmos-Reason2 骨干 + 扩散轨迹解码；
- **MindVLA-U1**（2026-05）：统一流式架构的驾驶 VLA。

---

## 4. 前沿实验室与公司专题

### 4.1 Anthropic × Claude

- **Project Fetch（2025-11-12）**：8 名零机器人经验的研究员分两组，给 Unitree Go2 编程捡沙滩球。
  - Claude 组完成 7/8 项任务，对照组 6/8；共同任务用时约减半，最大差距出在连接硬件和读取传感器上。
  - 据媒体报道，让当时的 Claude（Opus 4.1）独自完成时，它卡在了第一步（连上机器人）。
  - 作者提醒：AI 的"助力"往往先于"自主"出现。按 RSP 的思路，机器人能力是可能突然进步、需要监测的领域。
- **Project Fetch 第二阶段（2026-06-18）**：Claude Opus 4.7（Claude Code、adaptive thinking 最大档、每任务 3 次试验）独立完成。
  - 4 项共同任务用时 9 分 35 秒；对照有 Claude 辅助的人类团队 181 分钟（约 19×），无 Claude 的人类团队 361 分钟（约 38×）。
  - 代码约 1,045 行，人类 Claude 组写了 10,309 行。
  - 仍未完成自主"取回"（闭环控制差）；没有测试底层控制。
  - 作者称模型可能正在进入"physical agentic AI"的早期阶段。
- **Claude plays robotics（2026-07-09，Frontier Red Team）**：12 个模型、5 家厂商。
  - 本体：倒立摆等经典控制、仿真 G1/Go2、LIBERO 场景里的 Franka，加上真实 Go2 的若干演示。
  - 接口：扭矩/末端直接控制、写 `controller(obs)->action`、训练 PPO、监督预训练步态策略或 MolmoAct VLA。
  - Embody 综合分 Mythos Preview 最高（0.389），Opus 4 最低（0.115）；新模型主要靠"从失败中改策略"进步。
  - 实时直控约需 83Hz，非推理模式的推理只有 0.2–0.4Hz。
  - 作者把这项工作定位为安全问题：模型在真实世界的影响力取决于给它的接口权限。
- **Model Hardware Standard（2026-08-27，研究预览）**：Anthropic 第一个面向物理世界的产品。
  - 设备只需理解简单的读/写命令；安全参数和控制方法存成由自然语言描述生成的参考文件，可以编码机械臂的速度、角度限制。
  - 经 MCP/CLI/API 接入，一条命令可编排多台设备，模型无关。
  - CMU 约 8 小时接好一整套液体处理 + 成像设备。
- **生态**：大量第三方工作以 Claude 为 agent，包括 OpenRUA（Opus 5）、ASPIRE（Opus 5.5）、RobotWorld / Embodied Playground（Opus 5.5）、Project Fetch 风格的 Claude Code 工作流，以及 ROS MCP Server 示例（Claude 诊断工业机器人末端执行器）。

### 4.2 OpenAI × GPT

- 2023 年 Microsoft 发布 ChatGPT for Robotics，确立了"API 库 + 提示"的范式。2024 年 Figure 用 OpenAI 模型做对话，2025-02 Figure 宣布终止合作、转向自研 Helix。
- **2026-06**：Sam Altman 宣布成立 OpenAI Robotics，由 Aditya Ramesh 领导，起源于 world simulation 研究。9 月起表态"一定会做人形"，10 月报道计划 2027 年公开演示。目前没有公开的机器人模型或论文。
- **GPT-6 Astra（2026-09-03）**：官方主打电脑操作，社区发现它能直接输出数值动作。
  - 学术评测：Galbot、RoboDojo、SenseNova、NVIDIA Astranomy、RoboICL、GPT-Policy、触觉、LIBERO-Agent、LIBERO-10；
  - 真机演示：AGP（Codex CLI 驱动 YAM）、HomeBody（Unitree G1）、写字、画画、插插头等。
- **Decisions API（2026-10-06 公测）**：用 gpt-6-luna 回答有限选项的判断、选择、打分，比通用接口快约 10 倍。*Towards Agentic Robots* 把这类"决策模型"列为 harness 中 System 2 与执行器之间的新一层。

### 4.3 Google DeepMind × Gemini

- **Gemini Robotics / ER（2025-03）**：VLA + 具身推理模型，提出 ERQA；ASIMOV 语义安全基准。
- **Gemini Robotics 1.5 / ER 1.5（2025-09）**：
  - "先思考再行动"的 VLA，Motion Transfer 支持多本体（ALOHA、双臂 Franka、Apollo 人形）；
  - ER 1.5 是官方第一个为具身推理优化的 thinking 模型，通过 API 开放；它编排 VLA，可调用搜索（如按当地天气打包、按城市规则分类垃圾）。
- **ER 1.6（2026-04-14）**：空间与多视角推理增强，新增仪表读数（与 Boston Dynamics 合作，已集成进 Spot/Orbit），官方称"迄今最安全的机器人模型"。
- **Gemini Robotics 2 系列（2026 年中）**：
  - VLA：可适配双臂到全身人形；
  - ER 2（2026-07-30）：实时视频理解、任务进度跟踪、Live API 流式低延迟编排、工具调用、多机器人协作，Boston Dynamics Spot 演示；
  - On-Device 2：可在机器人本地运行的轻量 VLA。
- **Gemini 4 Argon for Robotics（2026-10-13）**：通用物理推理模型。
  - ERQA 87%；
  - 从机器人视频、第一人称人类视频或手绘草图生成 MuJoCo 孪生；
  - 生成 4,500+ 条域随机化仿真数据给 VLA 微调，真机成功率 56%→75%；
  - 支持在 SO-101 上迭代自我改进。The Decoder 评论说示例提示暴露出它仍需要大量人工引导。
- **虚拟世界**：SIMA 2（Gemini 驱动的 3D 世界 agent，能在 Genie 3 生成的世界中自我改进）。

### 4.4 NVIDIA

- **模型**：GR00T 系列（N1 → N1.7，N2 预览基于 DreamZero WAM）、Cosmos Reason/Cosmos 3、Alpamayo（驾驶推理 VLA）。
- **平台**：Isaac Sim/Lab；与 Hugging Face 合作把 GR00T 接入 LeRobot。
- **GEAR 团队 + 高校的 agent 线**：CaP-X、ASPIRE、ENPIRE、GaP、Voyager/Eureka 的延续。
- **Seattle Robotics Lab**：Astranomy（Astra 当策略的实证研究）、VoLo。

### 4.5 Physical Intelligence

- 模型：π0 → π0-FAST → π0.5 → π0.6/π*0.6（RECAP）→ π0.7（2026-04，可操控、有涌现能力）。
- MEM（多尺度具身记忆，十分钟以上任务）、RL Token（2026-03）。
- **Pi API beta（2026-10-13）**：上传数据、托管微调、云端推理。
- 与 agent 的关系：Hi Robot 开创了"VLM 发子任务给 VLA"的分层；openpi 的远程推理接口（WebSocket 返回动作块）是很多 harness 调用 VLA 的方式。

### 4.6 其他

- **Figure**：Helix → Helix 02（S0/S1/S2 三层，1kHz 全身控制）。2026-05 宣称多台人形全自主跑完 8 小时班次（公司自述）。
- **Tesla**：Optimus V3 + Grok 语音/对话层 + 端到端 VLA，均为非官方二手信息。
- **1X**：Redwood AI + 1X World Model（官方称 NEO 的"认知核心"）；复杂家务仍依赖远程专家模式。
- **Generalist AI**：GEN-0（27 万小时）、GEN-1（50 万小时，所选任务 99%）。
- **Skild AI**：Skild Brain、S1（看视频学 10 分钟级任务）。
- **Dyna**：DYNA-2 WAM。
- **Microsoft**：Magma、Rho-alpha。
- **Meta**：V-JEPA 2。
- **中国机构**：
  - AgiBot：GO-1/GO-2、Genie Envisioner 2.0、Genie Sim 3.0、RoboClaw、HarnessVLN；
  - 字节 Seed：GR-3、Robix；
  - 阿里：Qwen-RobotManip、AMAP ABot-N0/N1/AgentOS；
  - 腾讯：HY-Embodied-0.5；
  - 其他：BAAI RoboBrain/RoboOS、上海 AI Lab InternVLA、Galbot（Astra 评测）、X Square WALL-OSS。

---

## 5. 基准：怎样评价"agent 上机器人"

| 基准 | 评什么 | 关键发现 |
|---|---|---|
| Embodied Agent Interface（NeurIPS'24） | LLM 具身决策四模块 | 细粒度诊断规划错误类型 |
| EmbodiedBench（ICML'25） | MLLM agent 高层 vs 低层任务 | 高层任务 Claude-3.5 约 64–68%；低层操作最好的 GPT-4o 仅 28.9% |
| Embodied First Exam（2026-10） | 第一次具身智能标准化考试 | 全新评测平台，覆盖物理推理、空间理解等 |
| ERQA（Gemini Robotics） | 具身推理问答 | Gemini 4 Argon 87%（Google 自家基准） |
| Butter-Bench（2025-10） | LLM 当机器人"大脑" | 最好 40%，人类 95% |
| CaP-Bench（CaP-X） | 编码 agent 在不同抽象层 | 去掉人类抽象就掉分 |
| LIBERO-PRO | 扰动后的操作 | VLA 记忆化；agent 方法的主战场 |
| LIBERO-RECOVER（2026-09） | 失败恢复能力 | 不只看成功率，评测失败后的恢复表现 |
| RoboArena | 分布式真机成对评测 | DreamZero 1750 Elo，π0.5 1622（2026-04） |
| BEHAVIOR Challenge 2025 | 50 个家务长时程任务 | 冠军 Q-score 0.26（基于 π0.5） |
| RoboDojo | 仿真 + 真机统一 | Astra 22.48% 排第 1 |
| RobotWorld | 5 个领域 84 个任务 | 最强 19%；63 个任务无人完成 |
| RoboQuest | 主动探索获取信息 | 最好 23%；微调 π0.5 几乎为 0 |
| Embodied Playground（2026-10） | 410 个长时程 agentic 任务 | 考察物理、策略与空间推理 |
| ActiveArena（2026-09） | 操作中的主动感知 | 评测 agent 主动获取信息的能力 |
| MEMOBench（2026-09） | 过程级记忆 | 机器人操作的记忆基准 |
| RoboMemArena（2026-05） | 综合记忆能力 | 全面的机器人记忆基准 |
| LIBERO-Agent、EmbodiedSWE-Bench、Video2World、4DCodeBench | agent 原生、编码 agent、世界重建 | 2026-09/10 新出 |
| ASIMOV / ASIMOV-2.0、AgentSafe、GuardianBench | 语义与物理安全 | 据 ARC 转述：GPT-5、Gemini-2.5-Pro、Claude Opus 4.1 在 ASIMOV-2.0 的约束任务上违规率都在 30% 以上 |

**评测上的共识**（*Towards Agentic Robots* §8）：任务成功率无法区分是 agent 的功劳还是执行器的功劳。应当固定执行器、观测和交互预算，报告 token、时间和人工协助，并且加入需要处理缺失信息、执行器不合适、动作失败的任务。

---

## 6. 安全

- **攻击面**：
  - RoboPAIR：越狱 LLM 控制的机器人，自驾 LLM、GPT 规划的地面车、商用四足全部攻破；
  - BadRobot；
  - AgentRob：论坛 agent 经 MCP 调度真实 Go2/G1；
  - Butter-Bench 红队：电量低时诱导机器人泄露信息换充电。
- **行为风险**：
  - SafeHarness：编码 agent 在多数情况下仍撞上被禁止触碰的障碍物，推理里提到了障碍却不优先；
  - ROSClaw：不同模型提出越界动作的比率最多相差 4.8 倍；
  - Claude plays robotics 的真机片段：Grok 4.1 Fast 把玻璃门里的反射当成桌子冲过去；Opus 4.6 误判垃圾桶位置，把它拖行了几米。
  - GuardianBench（2026-08）：良性指令在特定安全场景下组合后出现的潜在风险。
- **防护**：
  - 规范检查：Safety Chip（LTL）、RoboGuard（越狱成功率 92.3%→2.3%）、SafeGate（ISO 13482）、RoboSafe（可执行谓词）、CORE（控制屏障函数，VLM 推理情境化约束）；
  - 接口层闸门：ROSClaw 安全包络、ETA 区分"想做"与"允许做"、MHS 参考文件编码限位；
  - 治理架构：ARC（2026-09，三层治理架构用于已部署自主系统）；
  - 系统化研究：SoK（2026-06，基础模型机器人的安全与隐私系统性框架）；
  - 基准：ASIMOV 系列、GuardianBench、AgentSafe。
- **部署清单（建议）**：
  1. 先仿真后真机；
  2. 安全检查放在模型之外：话题/服务白名单、速度、工作空间、力矩限幅；
  3. 运动类命令需要人工确认（Claude Code / Codex 的权限确认可直接利用）；
  4. 急停在手边；
  5. 全量记录日志与录像；
  6. 评估器与被评估的 agent 隔离，防止"钻评估器的空子"。

---

## 7. 给机器人研究者的上手路线

1. **零改动试水（1 小时）**：
   - 机器人端：`sudo apt install ros-<distro>-rosbridge-server`，再 `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`；
   - 本地：`claude mcp add ros-mcp -- uvx ros-mcp --transport=stdio`；
   - 然后让 Claude Code 先"只读"地探索 topic 和 service，描述机器人状态。
2. **让编码 agent 当策略（1 天）**：
   - OpenRUA：`pip install -U openrua` → 在项目目录运行 `openrua`，先选仿真机器人；
   - CaP-X：`git clone --recurse-submodules https://github.com/capgym/cap-x` → `uv sync`，提供 39 个任务的 Gym 与基准。
3. **把 VLA 当工具**：
   - openpi 有 π0/π0.5 权重和远程推理接口；也可以用 LeRobot / SmolVLA、GR00T N1.6、MolmoAct2；
   - 用 VLM 或 ER 模型（如 `gemini-robotics-er-2-preview`）编排，包装成 `vla_act(instruction, budget)` 工具，并返回进度和失败类型。
4. **自动化研发闭环**：参考 ENPIRE、ASPIRE、SimEX、HARBOR，关键是可自动复位、可自动判定成功、可回放的环境。
5. **数据与仿真**：SceneSmith、ARSTAG、Agentic Real2Sim、RoboGen；或者让 Gemini 4 Argon、GPT-6 Astra 这类模型写 MuJoCo/Isaac 场景。

**延迟预算参考**：

| 层 | 典型频率 |
|---|---|
| 前沿模型一次调用 | 秒级到数十秒 |
| ER 2 流式编排 | 低延迟，但仍远低于控制频率 |
| VLA | 10–50Hz（GR00T N1 为 120Hz 动作块，Helix S1 为 200Hz） |
| 全身控制 | 200–1000Hz（Helix 02 S0 为 1kHz） |

**原则：LLM 不进快环。**

---

## 8. 开放问题与趋势

1. **归因与公平评测**：agent 的贡献和执行器的贡献如何分离？
2. **何时调用大模型**：事件触发（HarnessWAM、VLAs-as-Tools）与固定频率之间的权衡；端云协同。
3. **物理世界里的"退出码"**：怎样可靠判断"做完了""失败了""为什么失败"（Thea、LIBERO-RECOVER）。
4. **自我改进的可累加性**：评估器、感知关系理解、技能链前端偏置（123 轮研究）。
5. **精细接触与动力学**：前沿模型的系统性短板，需要 VLA/WAM 或触觉补位（GPT-6-Astra Understands Touch 显示触觉上下文有帮助）。
6. **跨本体泛化**：agent 要先"读懂"陌生机器人的能力（EMOS 的机器人简历、OpenRUA 的 `openrua connect`）。
7. **权重 vs 技能**：VLA 把能力烤进权重，agent 把能力写成代码。一个可能的终局是：VLA 作为可调用的底层，程序作为上层，agent 离线进化二者，再把进化出的经验蒸馏回权重（EmbodiedSWE-Gen、RACaP、HarnessPAI、InSight）。
8. **安全、权限与治理**：MHS 等硬件标准、欧盟自 2027 年起对机械中 AI 安全功能的监管、RSP 类框架如何覆盖物理能力。

---

## 9. 推荐阅读路径（新手）

1. 经典三篇：SayCan → Code as Policies → VoxPoser
2. VLA 主线：RT-2 → OpenVLA → π0.5 → π0.7
3. Agentic VLA 栈：Gemini Robotics 1.5 技术报告 + ER 2 博客
4. 编码 agent：CaP-X → OpenRUA → ASPIRE / ENPIRE
5. 综述：*Towards Agentic Robots*（2610.20442）+ *Survey on Multimodal Embodied Agents*（NUS，engrXiv）+ *Weights or Skills?*（2608.01851）
6. 前沿模型即策略：Claude plays robotics → Galbot GPT-6 Astra 评测 → RobotWorld
7. 安全：RoboPAIR → RoboGuard → ASIMOV-2.0 → SafeHarness

---

## 10. 方法与可信度说明

- 检索时间 2026-10-23。arXiv 条目（283 条）全部用 export.arxiv.org API 按编号核验了标题与首发日期。网页/产业条目以官方博客为主，第三方报道已在文中注明（如 Tesla Optimus、Figure 与 BMW 的合作、部分 ER 1.6 指标）。
- 文中数值均摘自对应论文摘要、官方博客或其直接转述；跨论文的数字设置不同，不可直接横向比较。
- 2026 年的大量工作仍是预印本，没经过同行评审，部分演示没有公开成功率（如 HomeBody）。
- 有意从略的方向：纯自动驾驶（只列了 Alpamayo、MindVLA-U1 等少量条目）、纯数字 GUI agent、纯视觉导航方法细节。

## 主要网页来源

- Anthropic：[Project Fetch](https://www.anthropic.com/research/project-fetch-robot-dog) · [Phase two](https://www.anthropic.com/news/project-fetch-phase-two) · [Claude plays robotics](https://www.anthropic.com/research/claude-plays-robotics) · [MHS（Bloomberg）](https://www.bloomberg.com/news/articles/2026-08-27/anthropic-tests-new-way-for-claude-to-work-with-robots-and-scientific-lab-tools)
- Google DeepMind：[Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) · [ER 1.6](https://deepmind.google/blog/gemini-robotics-er-1-6/) · [ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/) · [Gemini Robotics 2 系列](https://deepmind.google/models/gemini-robotics/) · [Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/)
- OpenAI：[机器人部门（AI Insider）](https://theaiinsider.tech/2026/06/02/sam-altman-says-openai-now-has-a-dedicated-robotics-initiative-hiring-engineers/) · [人形 2027（TechTimes）](https://www.techtimes.com/articles/329197/20261019/openai-confirms-humanoid-robot-2027-demo-competing-companies-it-once-backed.htm) · [GPT-6 Astra（CNBC）](https://www.cnbc.com/2026/09/03/open-ai-astra-gpt-6-cyber.html) · [Decisions API](https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877)
- Physical Intelligence：[Pi API beta](https://www.pi.website/blog/api-beta) · [RL Token](https://www.pi.website/research/rlt)
- NVIDIA：[GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) · [GTC 2026](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) · [World-Action Models 博客](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/) · [Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/)
- 工具与资源：[ROS MCP Server](https://github.com/robotmcp/ros-mcp-server) · [CaP-X](https://github.com/capgym/cap-x) · [OpenRUA](https://github.com/terminalworld/OpenRUA) · [Awesome-Astra-Embodied-AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI) · [Awesome-Embodied-Robotics-and-Agent](https://github.com/zchoi/Awesome-Embodied-Robotics-and-Agent) · [NUS 多模态具身 agent 综述](https://engrxiv.org/preprint/view/8364)
- 其他：[Butter-Bench](https://andonlabs.com/evals/butter-bench) · [RoboArena](https://robo-arena.github.io/) · [BEHAVIOR 2025 榜单](https://behavior.stanford.edu/challenge/leaderboard.html) · [ASIMOV-2.0](https://asimov-benchmark.github.io/v2) · [Helix](https://www.figure.ai/news/helix) · [GEN-1](https://www.therobotreport.com/generalist-introduces-gen-1-general-purpose-model-for-physical-ai/) · [Rho-alpha](https://www.therobotreport.com/microsoft-research-reveals-rho-alpha-vision-language-action-model-for-robots/) · [Skild S1](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/) · [AgiBot GO-2](https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/) · [HomeBody（The Decoder）](https://the-decoder.com/researchers-plug-gpt-6-astra-directly-into-a-robot-and-let-it-clean-up-an-unfamiliar-kitchen/)

---

## 附录 A：论文与资源总表（315 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 校验；★ 为入门推荐。

### LLM 规划器（16）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2024-11 | [RoboMatrix](https://arxiv.org/abs/2412.00171) `arXiv:2412.00171` |  | 以技能为中心的分层框架，用于开放世界的任务规划与执行 |
| 2024-04 | [DELTA](https://arxiv.org/abs/2404.03275) `arXiv:2404.03275` |  | 场景图 + LLM 分解长时程任务，提升规划效率 |
| 2023-09 | [ConceptGraphs](https://arxiv.org/abs/2309.16650) `arXiv:2309.16650` | Toronto/MIT 等 | 开放词汇 3D 场景图，作为 LLM 规划的结构化记忆 |
| 2023-07 | [SayPlan](https://arxiv.org/abs/2307.06135) `arXiv:2307.06135` | QUT | 用 3D 场景图做 grounding，把 LLM 规划扩展到多楼层、多房间 |
| 2023-07 | ★ [KnowNo](https://arxiv.org/abs/2307.01928) `arXiv:2307.01928` | Princeton/Google | 用共形预测校准 LLM 规划的不确定性，不确定时主动向人求助 |
| 2023-06 | [Statler](https://arxiv.org/abs/2306.17840) `arXiv:2306.17840` | TTIC | 让 LLM 显式维护世界状态，改善长时程推理 |
| 2023-06 | [AutoTAMP](https://arxiv.org/abs/2306.06531) `arXiv:2306.06531` | MIT | LLM 把自然语言翻译成时序逻辑并做检查，再交给 TAMP |
| 2023-05 | [EmbodiedGPT](https://arxiv.org/abs/2305.15021) `arXiv:2305.15021` | HKU 等 | 用具身思维链预训练的视觉语言模型 |
| 2023-05 | [TidyBot](https://arxiv.org/abs/2305.05658) `arXiv:2305.05658` | Princeton/Stanford | LLM 从少量例子归纳用户偏好，实现个性化整理 |
| 2023-04 | [LLM+P](https://arxiv.org/abs/2304.11477) `arXiv:2304.11477` | UT Austin | LLM 把任务翻译成 PDDL，再交给经典规划器求解 |
| 2023-03 | [Text2Motion](https://arxiv.org/abs/2303.12153) `arXiv:2303.12153` | Stanford | 把语言指令变成几何上可行的长时程技能序列 |
| 2023-03 | [PaLM-E](https://arxiv.org/abs/2303.03378) `arXiv:2303.03378` | Google | 562B 具身多模态语言模型：图像/状态与文本交错输入，直接输出规划 |
| 2022-12 | [LLM-Planner](https://arxiv.org/abs/2212.04088) `arXiv:2212.04088` | Ohio State | 少样本 grounded 高层规划，执行受阻时根据环境重规划 |
| 2022-07 | ★ [Inner Monologue](https://arxiv.org/abs/2207.05608) `arXiv:2207.05608` | Google | 把成功检测、场景描述、人类反馈作为文本回灌 LLM，形成闭环规划 |
| 2022-04 | ★ [SayCan](https://arxiv.org/abs/2204.01691) `arXiv:2204.01691` | Google | LLM 给技能打"是否有用"分，价值函数给"是否可行"分，两者相乘选出下一步预训练技能 |
| 2022-04 | [Socratic Models](https://arxiv.org/abs/2204.00598) `arXiv:2204.00598` | Google | 多个基础模型以自然语言为接口零样本组合推理 |

### 代码即策略（24）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | ★ [OpenRUA](https://github.com/terminalworld/OpenRUA) `arXiv:2610.02459` | UCL/CMU | "零抽象" harness：Claude Code（Opus 5）只拿到 ROS 2 终端和文档，CaP-Bench 99%、LIBERO-PRO 87% |
| 2026-09 | [RAPID](https://arxiv.org/abs/2609.30249) `arXiv:2609.30249` | MIT/NUS | 从单次人类视觉演示自动生成、验证、精炼机器人程序 |
| 2026-09 | [RACaP](https://arxiv.org/abs/2609.29394) `arXiv:2609.29394` |  | 把"写代码"挪到进化阶段，部署时用 ReAct 调用冻结的类型化 Policy API，降低延迟 |
| 2026-09 | [EmbodiedSWE](https://arxiv.org/abs/2609.27308) `arXiv:2609.27308` |  | 编码 agent 解决长达半小时的灵巧长时程任务，再把解扩增成 VLA 训练数据，在真机上验证 |
| 2026-09 | [Local Coding Agent](https://arxiv.org/abs/2609.26499) `arXiv:2609.26499` |  | 本地开源 VLM（Qwen3.8-27B）在编码 agent 框架里驱动 UR3e，一次泛化到任务新变体 |
| 2026-09 | [Closed-Loop Robot Software](https://arxiv.org/abs/2609.19906) `arXiv:2609.19906` |  | 把编码 agent 优化过的闭环代码策略归档，作为新任务可迁移的经验 |
| 2026-09 | ★ [Agent as Policy (AGP)](https://arxiv.org/abs/2609.12541) `arXiv:2609.12541` | Notre Dame/UCSD | Codex CLI + GPT-6 Astra 直接驱动真实 YAM 机械臂：看图、写程序、发运动指令、按物理结果修正；8 种配置中 7 种成功率 ≥80% |
| 2026-07 | [MEMENTO](https://arxiv.org/abs/2607.22832) `arXiv:2607.22832` |  | 记忆引导的模因式（memetic）代码策略进化 |
| 2026-06 | [RHO](https://arxiv.org/abs/2606.16458) `arXiv:2606.16458` | UC Berkeley 等 | 训练阶段让编码 agent 搜索多文件的"仓库即策略"；LIBERO-PRO 45%（π0.5 为 12.83%，OpenVLA 为 0%） |
| 2026-03 | ★ [CaP-X](https://arxiv.org/abs/2603.22435) `arXiv:2603.22435` | NVIDIA/UC Berkeley/Stanford/CMU | 编码 agent 操控框架与基准：人类给的抽象越少越难；多轮交互、视觉差分、技能合成与 RL 可以补回来（ICML'26） |
| 2026-03 | [Act-Observe-Rewrite](https://arxiv.org/abs/2603.04466) `arXiv:2603.04466` |  | 多模态编码 agent 在两次尝试之间重写完整的底层控制器代码，无需梯度更新 |
| 2025-09 | [LYRA](https://arxiv.org/abs/2509.18597) `arXiv:2509.18597` |  | 人类引导的终身代码生成，学到超长时程操作技能 |
| 2024-09 | [ReKep](https://arxiv.org/abs/2409.01652) `arXiv:2409.01652` | Stanford | VLM 写"关系关键点约束"函数，求解器约 10Hz 实时优化轨迹 |
| 2024-06 | [LRLL](https://arxiv.org/abs/2406.18746) `arXiv:2406.18746` | Groningen | 终身机器人技能库学习，自举可组合、可泛化的技能 |
| 2024-06 | [RoboCoder](https://arxiv.org/abs/2406.03757) `arXiv:2406.03757` | CUHK | 基础技能库 + 评估器反馈，逐步生成更复杂任务的代码 |
| 2024-03 | [MOKA](https://arxiv.org/abs/2403.03174) `arXiv:2403.03174` | UC Berkeley | 标记式视觉提示：VLM 在图上选关键点/路径点，转为动作 |
| 2024-02 | [RoboCodeX](https://arxiv.org/abs/2402.16117) `arXiv:2402.16117` | HKU/Shanghai AI Lab | 多模态树状代码生成合成机器人行为，并微调代码模型 |
| 2023-11 | [DROC](https://arxiv.org/abs/2311.10678) `arXiv:2311.10678` | Stanford | 从人的语言纠正中蒸馏并检索可泛化知识，同样的纠正不用说第二次 |
| 2023-10 | [OLAF](https://arxiv.org/abs/2310.17555) `arXiv:2310.17555` | UT Austin | 把口头纠正转成修正后的动作标签，交互式学习 |
| 2023-07 | ★ [VoxPoser](https://arxiv.org/abs/2307.05973) `arXiv:2307.05973` | Stanford | LLM 写代码调用 VLM，组合出 3D 价值图，交给运动规划执行 |
| 2023-05 | [Instruct2Act](https://arxiv.org/abs/2305.11176) `arXiv:2305.11176` |  | LLM 生成代码，调用 SAM/CLIP 等感知模型与运动基元 |
| 2023-02 | [ChatGPT for Robotics](https://arxiv.org/abs/2306.17582) `arXiv:2306.17582` | Microsoft | 提出"API 库 + 提示工程 + 人在环"的 LLM 机器人编程范式 |
| 2022-09 | [ProgPrompt](https://arxiv.org/abs/2209.11302) `arXiv:2209.11302` | NVIDIA/USC | 程序化提示：用带断言的程序表示可执行的情境化任务计划 |
| 2022-09 | ★ [Code as Policies](https://arxiv.org/abs/2209.07753) `arXiv:2209.07753` | Google | LLM 写 Python 程序组合感知/控制 API，程序本身就是策略。"代码即策略"的开山之作 |

### 奖励/仿真/数据生成（23）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [EmbodiedSmith](https://arxiv.org/abs/2610.07969) `arXiv:2610.07969` |  | 在仿真中跑递归自我改进的数据飞轮 |
| 2026-10 | [Video2World](https://arxiv.org/abs/2610.04432) `arXiv:2610.04432` |  | 基准：编码 agent 从具身视频重建可交互的仿真世界 |
| 2026-10 | [4DCodeBench](https://arxiv.org/abs/2610.03715) `arXiv:2610.03715` |  | 基准：agent 写可执行代码重建动态场景（逆图形学） |
| 2026-09 | [ARSTAG](https://arxiv.org/abs/2609.24563) `arXiv:2609.24563` | CMU | 一张 RGB 图 + 一句指令，经 Real2Sim2Real 生成训练数据；π0.5 真机平均成功率 74.6% |
| 2026-07 | [Agentic Real2Sim](https://arxiv.org/abs/2607.19190) `arXiv:2607.19190` |  | VLM agent 把真实机器人交互录像变成可仿真的"情景孪生"，用于微调与评测 |
| 2026-06 | [SimFoundry](https://arxiv.org/abs/2606.28276) `arXiv:2606.28276` |  | 模块化、自动化的场景生成，用于策略学习与评测 |
| 2026-06 | [HARBOR](https://arxiv.org/abs/2606.08610) `arXiv:2606.08610` |  | 把机器人 RL 自动化看成 harness 工程：环境搭建、奖励设计、调参、训练一条龙 |
| 2026-05 | [SR-Platform](https://arxiv.org/abs/2605.14700) `arXiv:2605.14700` |  | 从自然语言生成 MuJoCo 仿真环境的 agentic 流水线 |
| 2026-05 | [Articraft](https://arxiv.org/abs/2605.15187) `arXiv:2605.15187` |  | agentic 方式大规模生成铰接 3D 资产 |
| 2026-05 | [SimWorld Studio](https://arxiv.org/abs/2605.09423) `arXiv:2605.09423` |  | 进化式编码 agent 自动生成具身学习环境 |
| 2026-02 | [SceneSmith](https://arxiv.org/abs/2602.09153) `arXiv:2602.09153` | MIT/TRI | agentic 生成可直接仿真的室内场景，由评估 agent 检查任务是否成功（ICML'26 Spotlight） |
| 2026-01 | [Genie Sim 3.0](https://arxiv.org/abs/2601.02078) `arXiv:2601.02078` | AgiBot | 高保真人形机器人仿真平台 |
| 2025-02 | [Video2Policy](https://arxiv.org/abs/2502.09886) `arXiv:2502.09886` |  | 从互联网视频批量构建仿真任务 |
| 2024-10 | [GenSim2](https://arxiv.org/abs/2410.03645) `arXiv:2410.03645` |  | 用多模态/推理 LLM 扩展机器人数据生成 |
| 2024-09 | [CurricuLLM](https://arxiv.org/abs/2409.18382) `arXiv:2409.18382` | UC Berkeley | LLM 自动设计任务课程来学习复杂技能 |
| 2024-06 | [DrEureka](https://arxiv.org/abs/2406.01967) `arXiv:2406.01967` | UPenn/NVIDIA | LLM 同时设计奖励与域随机化参数，实现 sim2real（如四足在瑜伽球上行走） |
| 2024-01 | [AutoRT](https://arxiv.org/abs/2401.12963) `arXiv:2401.12963` | Google DeepMind | VLM + LLM 编排 20 多台机器人大规模采数据，内置"机器人宪法"安全规则 |
| 2023-12 | [Holodeck](https://arxiv.org/abs/2312.09067) `arXiv:2312.09067` | UPenn/AI2 | 用语言引导生成 3D 具身 AI 环境 |
| 2023-11 | ★ [RoboGen](https://arxiv.org/abs/2311.01455) `arXiv:2311.01455` | CMU/MIT 等 | 生成式仿真：自动提出任务、生成场景、选择学习方法并产出技能 |
| 2023-10 | ★ [Eureka](https://arxiv.org/abs/2310.12931) `arXiv:2310.12931` | NVIDIA/UPenn | LLM 写奖励函数代码并进化式迭代，多数任务上超过人类专家设计的奖励 |
| 2023-10 | [GenSim](https://arxiv.org/abs/2310.01361) `arXiv:2310.01361` | MIT | LLM 生成仿真任务代码，扩充任务库 |
| 2023-09 | [Text2Reward](https://arxiv.org/abs/2309.11489) `arXiv:2309.11489` | HKU 等 | 自然语言目标转成可执行的稠密奖励代码 |
| 2023-06 | [Language to Rewards](https://arxiv.org/abs/2306.08647) `arXiv:2306.08647` | Google | LLM 把指令翻译成奖励参数，由 MPC 实时合成技能 |

### VLA/基础模型（37）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [PhysBrain 1.5](https://arxiv.org/abs/2609.14973) `arXiv:2609.14973` | DeepCybo | 用人类交互视频预训练，把理解、动作与未来预测统一在一个模型里 |
| 2026-07 | [ACE-Brain-0.5](https://arxiv.org/abs/2607.04426) `arXiv:2607.04426` |  | 面向"物理 agentic AI"的统一具身基础模型：感知、决策、交互、自我监控、持续改进 |
| 2026-06 | [Qwen-RobotManip](https://arxiv.org/abs/2606.17846) `arXiv:2606.17846` | Alibaba Qwen | 基于 Qwen-VL，从表征、运动、行为三个维度对齐来扩大规模 |
| 2026-06 | [Embodied-R1.5](https://arxiv.org/abs/2606.11324) `arXiv:2606.11324` |  | 统一的具身推理模型（认知/规划/纠正/指点），配 Planner-Grounder-Corrector 闭环 |
| 2026-05 | [MolmoAct2](https://arxiv.org/abs/2605.02881) `arXiv:2605.02881` | AI2 | 全开源的动作推理模型，面向真实部署 |
| 2026-04 | ★ [π0.7](https://arxiv.org/abs/2604.15483) `arXiv:2604.15483` | Physical Intelligence | 可用多模态上下文（策略、元数据、子目标图）操控的约 5B 通用模型，零样本跨本体 |
| 2026-04 | [HY-Embodied-0.5](https://arxiv.org/abs/2604.07430) `arXiv:2604.07430` | Tencent | 面向真实世界 agent 的具身基础模型（2B/32B 激活参数两档） |
| 2026-02 | [LeRobot](https://arxiv.org/abs/2602.22818) `arXiv:2602.22818` | Hugging Face | 端到端机器人学习开源库：硬件接口、数据、训练、部署 |
| 2025-10 | [InternVLA-M1](https://arxiv.org/abs/2510.13778) `arXiv:2510.13778` | Shanghai AI Lab | 空间引导的 VLA 框架 |
| 2025-10 | [X-VLA](https://arxiv.org/abs/2510.10274) `arXiv:2510.10274` | Tsinghua 等 | 软提示 Transformer，可扩展的跨本体 VLA |
| 2025-10 | [D2E](https://arxiv.org/abs/2510.05684) `arXiv:2510.05684` |  | 用桌面（游戏/GUI）交互数据做视觉-动作预训练，再迁移到具身任务（ICLR'26） |
| 2025-10 | ★ [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) `arXiv:2510.03342` | Google DeepMind | "先思考再行动"的 VLA + Motion Transfer；ER 1.5 担任编排者，调用 VLA 和搜索等工具 |
| 2025-09 | [WALL-OSS](https://arxiv.org/abs/2509.11766) `arXiv:2509.11766` | X Square Robot | 统一跨层思维链的具身基础模型（"Igniting VLMs toward the Embodied Space"） |
| 2025-08 | [EO-1](https://arxiv.org/abs/2508.21112) `arXiv:2508.21112` | Shanghai AI Lab 等 | 开源统一具身基础模型（视觉-文本-动作交错预训练） |
| 2025-08 | [MolmoAct](https://arxiv.org/abs/2508.07917) `arXiv:2508.07917` | AI2 | "动作推理模型"：先做深度感知和轨迹规划，再输出动作 |
| 2025-07 | [ThinkAct](https://arxiv.org/abs/2507.16815) `arXiv:2507.16815` | NVIDIA | 用强化学习的视觉潜在规划增强 VLA 推理 |
| 2025-07 | [GR-3](https://arxiv.org/abs/2507.15493) `arXiv:2507.15493` | ByteDance Seed | 大规模 VLA 技术报告，覆盖长时程与灵巧操作 |
| 2025-07 | [RoboBrain 2.0](https://arxiv.org/abs/2507.02029) `arXiv:2507.02029` | BAAI | 具身"大脑"VLM：空间理解与时序决策 |
| 2025-06 | [SmolVLA](https://arxiv.org/abs/2506.01844) `arXiv:2506.01844` | Hugging Face | 小型开源 VLA + 异步推理，消费级硬件就能训练 |
| 2025-05 | [OneTwoVLA](https://arxiv.org/abs/2505.11917) `arXiv:2505.11917` | Tsinghua | 单个模型自适应地在推理和执行之间切换 |
| 2025-05 | [UniVLA](https://arxiv.org/abs/2505.06111) `arXiv:2505.06111` | HKU/OpenDriveLab | 以任务为中心的潜在动作，跨本体学习 |
| 2025-04 | ★ [π0.5](https://arxiv.org/abs/2504.16054) `arXiv:2504.16054` | Physical Intelligence | 开放世界泛化；同一模型先解码子任务文本，再生成动作 |
| 2025-03 | ★ [Gemini Robotics](https://arxiv.org/abs/2503.20020) `arXiv:2503.20020` | Google DeepMind | 基于 Gemini 2.0 的 VLA 与具身推理（ER）模型，提出 ERQA 基准 |
| 2025-03 | [GR00T N1](https://arxiv.org/abs/2503.14734) `arXiv:2503.14734` | NVIDIA | 开源人形基础模型，双系统：VLM（10Hz）+ 扩散动作模块（120Hz） |
| 2025-03 | [Cosmos-Reason1](https://arxiv.org/abs/2503.15558) `arXiv:2503.15558` | NVIDIA | 物理常识与具身推理 VLM；后续 Cosmos Reason 2 成为 GR00T N1.6 的骨干 |
| 2025-02 | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) `arXiv:2502.19645` | Stanford | VLA 微调配方：并行解码、动作块、连续动作 |
| 2025-02 | [Magma](https://arxiv.org/abs/2502.13130) `arXiv:2502.13130` | Microsoft | 同时面向 UI 操作和机器人操作的多模态 agent 基础模型 |
| 2025-01 | [FAST](https://arxiv.org/abs/2501.09747) `arXiv:2501.09747` | Physical Intelligence | 基于 DCT 压缩的高效动作 tokenizer |
| 2024-11 | [CogACT](https://arxiv.org/abs/2411.19650) `arXiv:2411.19650` | Microsoft 等 | 认知（VLM）与动作（扩散）模块协同的 VLA |
| 2024-10 | ★ [π0](https://arxiv.org/abs/2410.24164) `arXiv:2410.24164` | Physical Intelligence | VLM + flow matching 动作专家，跨本体灵巧操作 |
| 2024-10 | [RDT-1B](https://arxiv.org/abs/2410.07864) `arXiv:2410.07864` | Tsinghua | 双臂操作的扩散基础模型 |
| 2024-07 | [ECoT](https://arxiv.org/abs/2407.08693) `arXiv:2407.08693` | UC Berkeley/Stanford 等 | 具身思维链：先推理子任务、物体、运动，再输出动作 |
| 2024-06 | ★ [OpenVLA](https://arxiv.org/abs/2406.09246) `arXiv:2406.09246` | Stanford/UC Berkeley 等 | 7B 开源 VLA，开源 VLA 生态由此起步 |
| 2024-05 | [Octo](https://arxiv.org/abs/2405.12213) `arXiv:2405.12213` | UC Berkeley 等 | 开源通用机器人策略（Transformer + 扩散动作头） |
| 2023-10 | [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) `arXiv:2310.08864` | 多机构合作 | 跨 22 种本体、百万级轨迹的开放数据集与 RT-X 模型 |
| 2023-07 | ★ [RT-2](https://arxiv.org/abs/2307.15818) `arXiv:2307.15818` | Google DeepMind | 把动作当作语言 token 来微调 VLM，提出 VLA 概念 |
| 2022-12 | [RT-1](https://arxiv.org/abs/2212.06817) `arXiv:2212.06817` | Google | 用约 13 万条真机演示训练的 Transformer 机器人策略 |

### VLA 强化学习（7）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [RL for Real-Time VLA](https://arxiv.org/abs/2609.18207) `arXiv:2609.18207` | Stanford | 面向实时 VLA 策略的强化学习 |
| 2026-07 | [HELP](https://arxiv.org/abs/2607.09776) `arXiv:2607.09776` |  | 按 rollout 分段，降低大规模后训练所需的人力 |
| 2026-06 | [TORL-VLA](https://arxiv.org/abs/2606.09337) `arXiv:2606.09337` |  | 触觉引导的接触密集操作在线 RL |
| 2026-05 | [Learning While Deploying](https://arxiv.org/abs/2605.00416) `arXiv:2605.00416` |  | 在机器人舰队规模上对通用策略做强化学习 |
| 2026-04 | [RL Token](https://www.pi.website/research/rlt) `arXiv:2604.23073` | Physical Intelligence | 从 VLA 读出一个紧凑的"RL token"，小型 actor-critic 头在线 RL，几小时内关键阶段提速最高 3× |
| 2025-11 | ★ [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759) `arXiv:2511.14759` | Physical Intelligence | 优势条件化 RL，融合演示、自主数据与人类干预，让 VLA 从部署经验中提升 |
| 2025-10 | [πRL](https://arxiv.org/abs/2510.25889) `arXiv:2510.25889` |  | 流匹配 VLA 的在线 RL 微调 |

### 分层编排 S2→S1（18）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | ★ [Physical Agency (Pigey)](https://arxiv.org/abs/2607.21725) `arXiv:2607.21725` | Princeton | 闭环物理 agent 编排器调度 VLA 与参数化技能，不收集新数据；LIBERO-PRO 12.8%→53.3% |
| 2026-07 | [RoboHarness](https://arxiv.org/abs/2607.18060) `arXiv:2607.18060` |  | 记忆驱动地编排异构策略，用运动规划的"记忆桥"衔接前后状态 |
| 2026-07 | [Retriever](https://arxiv.org/abs/2607.17213) `arXiv:2607.17213` |  | 异步的感知-信念-规划-VLA-控制循环：决策模型、编程模型与运行时 |
| 2026-07 | [Harness VLA](https://harnessvla.github.io/) `arXiv:2607.08448` |  | 冻结的 VLA 作为可重试的接触原语，再配少量解析原语；用记忆学习各原语的适用范围 |
| 2026-07 | [Semantic Handoff Failures](https://arxiv.org/abs/2607.06256) `arXiv:2607.06256` |  | 诊断 agent 编排 VLA 技能组合时的语义交接失败 |
| 2026-07 | [Cortex](https://arxiv.org/abs/2607.05377) `arXiv:2607.05377` |  | 把子任务标准化成 32 种规范技能原语，让 VLM 的规划与 VLA 的执行双向对齐 |
| 2026-06 | ★ [Hi-VLA 设计研究](https://arxiv.org/abs/2606.10267) `arXiv:2606.10267` |  | 系统研究分层 VLA agent：终止规则、VLA 可操控性与规划器大小同样关键 |
| 2026-06 | [VoLo](https://arxiv.org/abs/2606.07723) `arXiv:2606.07723` | NVIDIA | VLM 把 VLA/WAM 当作可中断的工具，执行中途接管、监控与恢复 |
| 2026-05 | [VLAs-as-Tools](https://arxiv.org/abs/2605.13119) `arXiv:2605.13119` | SJTU | VLM agent 调用一族专用 VLA 工具，每个工具有执行预算并回报进度 |
| 2026-03 | [RoboRouter](https://arxiv.org/abs/2603.07892) `arXiv:2603.07892` |  | 免训练的策略路由：根据当前状态从多个策略里挑一个 |
| 2026-03 | [Critic in the Loop](https://arxiv.org/abs/2603.05185) `arXiv:2603.05185` |  | VLM 大脑 + VLA 小脑 + 轻量视觉 critic 的三系统 |
| 2026-01 | [Helix 02 (Figure)](https://interestingengineering.com/ai-robotics/figure-helix02-upgrades-humanoid-robot-control) | Figure | 新增 System 0（1kHz 学习式全身控制器，替代约 10.9 万行 C++），一个网络控制行走、操作与平衡；连续 61 个动作完成洗碗机任务 |
| 2025-09 | [Robix](https://arxiv.org/abs/2509.01106) `arXiv:2509.01106` | ByteDance Seed | 统一机器人交互、推理与任务规划的高层模型 |
| 2025-06 | [VLA-OS](https://arxiv.org/abs/2506.17561) `arXiv:2506.17561` |  | 对比不同规划表示：低层策略对视觉计划比对语言计划更"听话" |
| 2025-02 | ★ [Hi Robot](https://arxiv.org/abs/2502.19417) `arXiv:2502.19417` | Physical Intelligence | VLM 约每秒一次（或用户插话时）生成语言子任务交给 VLA，支持开放式指令与纠正 |
| 2025-02 | [Helix (Figure)](https://www.figure.ai/news/helix) | Figure | System 2（7–9Hz VLM）+ System 1（200Hz 视觉运动策略）控制人形上半身；发布前 Figure 宣布结束与 OpenAI 的合作 |
| 2025-02 | [HAMSTER](https://arxiv.org/abs/2502.05485) `arXiv:2502.05485` | NVIDIA/UW | VLM 预测 2D 路径作为中间表示，交给 3D 低层策略 |
| 2024-10 | [RoboDual](https://arxiv.org/abs/2410.08001) `arXiv:2410.08001` |  | 通用 VLA + 专用扩散策略组成的双系统 |

### Harness/工具/MCP/记忆（35）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [RobotUse](https://arxiv.org/abs/2610.04929) `arXiv:2610.04929` | Seoul National Univ. | agent 在图像上选目标与位姿，后端负责几何、规划与控制；子 agent 管细节，持续更新 playbook。RoboLab 45%（比 CaP-X 高 6.7pp） |
| 2026-09 | [Recursive Harness Distillation](https://arxiv.org/abs/2609.33378) `arXiv:2609.33378` | Seoul National Univ. | 强 agent 把干预经验蒸馏成 playbook，交给廉价 agent 指挥冻结 VLA；真机成功率 37.3%→64.0% |
| 2026-09 | [Robo-Harness K1](https://arxiv.org/abs/2609.29389) `arXiv:2609.29389` |  | 把感知包装成工具（标定深度、持久视觉锚点、抓取假设），供 agent 查询 |
| 2026-09 | [MessyMem](https://arxiv.org/abs/2609.15976) `arXiv:2609.15976` | Stanford/CMU 等 | "做中学"记忆：每次物理交互后由 VLM 记录结果（如没抓住、抽屉锁着），更新对应物体条目 |
| 2026-09 | [Harness Robotic OS](https://arxiv.org/abs/2609.11225) `arXiv:2609.11225` |  | 四足闭环巡检的统一具身 agent 运行时 |
| 2026-09 | [Show-Harness](https://arxiv.org/abs/2609.10522) `arXiv:2609.10522` | NUS Show Lab | 离散语义动作单元 + 本体解释器：闭源 VLM 零样本控制，小开源 VLM 微调几个 GPU 小时即可 |
| 2026-08 | [EMERGE-Policy](https://arxiv.org/abs/2608.29896) `arXiv:2608.29896` |  | 主 agent + 角色子 agent（感知/监控/验证/记忆），图结构编排 |
| 2026-08 | [Physical Agentic AI](https://arxiv.org/abs/2608.22657) `arXiv:2608.22657` |  | 编排机器人"船员"：每个计划动作在执行前都对照能力、状态和流程约束做验证 |
| 2026-08 | [BATON](https://arxiv.org/abs/2608.16889) `arXiv:2608.16889` | USC | 在技能切换时检索"过渡记忆"，支撑长时程操作 |
| 2026-08 | ★ [ETA / OpenETA](https://arxiv.org/abs/2608.03924) `arXiv:2608.03924` | Fudan | Planner–Tool–Interface–World 循环，一次只调用一个工具；可以作为 Codex 的轻量插件（只暴露 observe/mark_point/move_to） |
| 2026-08 | ★ [Thea](https://eit-hai.github.io/thea) `arXiv:2608.11246` |  | 把编码 agent 的 harness 搬进物理世界，提出"场景图即上下文""评估即退出码" |
| 2026-07 | [PhyAgentOS](https://arxiv.org/abs/2607.16636) `arXiv:2607.16636` | Sun Yat-sen Univ. | 以"会话"为调度单元的具身 agent 运行时，用 State-as-a-File 解耦认知与物理执行 |
| 2026-07 | [PHILIA](https://arxiv.org/abs/2607.11377) `arXiv:2607.11377` |  | 基于 OpenClaw 的长期多机器人助理：机器人网关抽象，长期记忆人与偏好 |
| 2026-07 | [ABot-AgentOS](https://arxiv.org/abs/2607.10350) `arXiv:2607.10350` | Alibaba AMAP | 通用机器人 Agent OS：场景规划、技能隔离执行、多级验证、多模态记忆、端云协同 |
| 2026-07 | [ACE](https://arxiv.org/abs/2607.04162) `arXiv:2607.04162` | Tsinghua | 零样本工作流推理的 agentic 操作控制 |
| 2026-07 | [HiMe](https://arxiv.org/abs/2607.03449) `arXiv:2607.03449` | Fudan | 层次化具身记忆，用于长时程 VLA 控制 |
| 2026-07 | [ROSA (serving)](https://arxiv.org/abs/2607.01088) `arXiv:2607.01088` | NVIDIA/Stanford | 机器人工厂中多台机器人共享 GPU 池的基础模型推理服务 |
| 2026-06 | [HoloAgent-0](https://arxiv.org/abs/2606.23565) `arXiv:2606.23565` |  | 带 3D 空间记忆的统一具身 agent 框架 |
| 2026-06 | [Guava](https://arxiv.org/abs/2606.18363) `arXiv:2606.18363` |  | 系统探索 harness 的设计空间，并把能力蒸馏进 4B 开源模型（不到 2K 条仿真轨迹） |
| 2026-05 | [Embodied Tool Protocol](https://arxiv.org/abs/2605.26637) `arXiv:2605.26637` |  | 100+ 具身工具的注册/发现/调用标准与 EmbodiedToolBench；EB-ALFRED +31%，但执行类工具增益有限 |
| 2026-04 | [Goal2Skill](https://arxiv.org/abs/2604.13942) `arXiv:2604.13942` |  | 自适应规划 + 反思，面向长时程操作 |
| 2026-04 | [EEAgent](https://arxiv.org/abs/2604.13533) `arXiv:2604.13533` |  | 长短期反思与优化的可进化具身 agent |
| 2026-04 | [Physical Agentic Loop](https://arxiv.org/abs/2604.07395) `arXiv:2604.07395` | QUT | 语言引导抓取的执行状态监控，只在夹爪为空时才重试 |
| 2026-03 | [ROSClaw](https://arxiv.org/abs/2603.26997) `arXiv:2603.26997` |  | 把开源 agent 运行时 OpenClaw 接到 ROS 2：能力发现、执行前安全包络、审计日志；不同模型提出越界动作的比率最多相差 4.8× |
| 2026-03 | ★ [RoboClaw](https://roboclaw-agibot.github.io/) `arXiv:2603.11558` | AgiBot 等 | 一个 VLM 控制器统一数据采集、策略学习与执行；"正向动作+逆向复位"对实现自主采集；长时程成功率 +25%，人力投入 −53.7% |
| 2026-02 | [PhysMem](https://arxiv.org/abs/2602.20323) `arXiv:2602.20323` | UCSD | 扩展测试时记忆来做物理推理 |
| 2026-02 | [VLA-Perf](https://arxiv.org/abs/2602.18397) `arXiv:2602.18397` |  | 解析建模 VLA 在不同设备与网络上的推理延迟（π0 经 WiFi 用边缘服务器可能比机载更快） |
| 2026 | [AgenticROS](https://agenticros.com/) | AgenticROS | 提供 Claude Code / Claude 桌面端可用的 MCP 服务，以"命名能力 + 多步任务"而非原始 topic 控制机器人 |
| 2025-12 | [RoboNeuron](https://arxiv.org/abs/2512.10394) `arXiv:2512.10394` |  | MCP 与 ROS 2 之间的中间层：从 ROS 消息模式自动生成 agent 可调用的工具 |
| 2025-11 | [VLASH](https://arxiv.org/abs/2512.01031) `arXiv:2512.01031` |  | 以"动作执行时刻的预测状态"为条件做异步推理，补偿 VLA 延迟 |
| 2025-10 | [Running VLAs at Real-time Speed](https://arxiv.org/abs/2510.26742) `arXiv:2510.26742` |  | 算子融合与 GPU kernel 优化，让 π0 在消费级 GPU 上约 27ms 一次推理 |
| 2025-09 | ★ [ROS MCP Server](https://github.com/robotmcp/ros-mcp-server) | robotmcp（开源） | 只需在机器人端启动 rosbridge，Claude Code / Codex / Gemini CLI 等 MCP 客户端就能发现并调用 ROS 1/2 的 topic、service、action |
| 2025-06 | [Real-Time Chunking (RTC)](https://arxiv.org/abs/2506.07339) `arXiv:2506.07339` | Physical Intelligence | 推理期间冻结已承诺的动作、补全下一块，让推理慢的流匹配策略也能实时执行 |
| 2023-07 | [DoReMi](https://arxiv.org/abs/2307.00329) `arXiv:2307.00329` | Tsinghua 等 | LLM 为下一个技能写约束，VLM 定期检查，一旦违反就停下并重规划 |
| 2023-06 | [REFLECT](https://arxiv.org/abs/2306.15724) `arXiv:2306.15724` | Columbia | 把机器人经历按传感器、事件、子目标三级汇总成文本，解释失败并给出纠正计划 |

### 自动化研发/自我改进（20）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [AgenticRoboRL](https://arxiv.org/abs/2610.28589) `arXiv:2610.28589` |  | 对机器人规划器做 SFT + 多轮 RL 后训练：9B 模型 LIBERO-PRO 18.1%→71.0%，接近带外部记忆的 Codex（GPT-5.5）的 72.1% |
| 2026-10 | [RoboRSI](https://arxiv.org/abs/2610.12424) `arXiv:2610.12424` | SJTU | 自顶向下的技能精炼：把每次修复归因到具体能力，验证后才复用 |
| 2026-10 | [RPG (Reconstruct, Practice, Go Real)](https://arxiv.org/abs/2610.02204) `arXiv:2610.02204` | UC Berkeley | 从离线数据识别能力，在仿真里构造练习任务，再上真机；全程不更新权重 |
| 2026-09 | [SimEX](https://robo-simex.github.io/) `arXiv:2609.38982` | UT Austin/Amazon FAR/CMU | 仿真集成的机器人 autoresearch：先在仿真里练出工具箱，再用少量真机试次同时校正仿真器和工具箱（每任务约 10 分钟真机时间） |
| 2026-09 | [EmbodiRSI](https://arxiv.org/abs/2609.38905) `arXiv:2609.38905` |  | 面向数据高效适配的递归自我改进 |
| 2026-09 | [Explore, Execute, Evolve](https://arxiv.org/abs/2609.37810) `arXiv:2609.37810` | Fudan | 技能获取与复用闭环（论文报告进步并非单调） |
| 2026-09 | [HarnessPAI](https://darwin-agent.github.io/HarnessPAI) `arXiv:2609.29166` |  | 以代码作为可执行、可进化的接口来组织动作原语 |
| 2026-09 | ★ [123 轮自改进的教训](https://arxiv.org/abs/2609.31760) `arXiv:2609.31760` |  | agent 能自己发现缺失能力，但改进不累加；瓶颈在感知的关系理解、技能链前端偏置和评估器本身 |
| 2026-08 | [Zetta](https://arxiv.org/abs/2608.16590) `arXiv:2608.16590` |  | 在线演化基于代码的运行时 critic 与恢复技能，基础策略保持冻结 |
| 2026-08 | [Teach and Grow](https://arxiv.org/abs/2608.17209) `arXiv:2608.17209` | SJTU | 以 agent 为中心的通用机器人学习架构 |
| 2026-08 | [AgenticRobotics](https://arxiv.org/abs/2608.07555) `arXiv:2608.07555` |  | 把 Claude Code/Codex 的"主 agent + 子 agent + 工具"架构移植到训练-评估-改进闭环，强调证据门控与崩溃恢复 |
| 2026-07 | [Zero2Skill](https://arxiv.org/abs/2607.14047) `arXiv:2607.14047` |  | 自主采集、训练、部署的闭环技能自举 |
| 2026-07 | [GaP](https://graph-robots.github.io/gap) `arXiv:2607.05369` | UC Berkeley/NVIDIA 等 | 多 agent 编码 harness 生成"图即策略"，在内部仿真里并行排练、优化，面向工业"变型自动化" |
| 2026-06 | ★ [ASPIRE](https://arxiv.org/abs/2607.00272) `arXiv:2607.00272` | NVIDIA 等 | 闭环执行引擎 + 可复用技能库 + 进化搜索；用 Claude Opus 5.5 时 LIBERO-Pro 80 个任务平均 99%，并能在真机上解魔方 |
| 2026-06 | [InSight](https://arxiv.org/abs/2606.24884) `arXiv:2606.24884` |  | VLM 找出 VLA 缺失的原语，经真机执行验证后再蒸馏回 VLA |
| 2026-06 | [RoboLineage](https://arxiv.org/abs/2606.22142) `arXiv:2606.22142` |  | agent 原生的数据生命周期治理，跨越多轮策略迭代 |
| 2026-06 | ★ [ENPIRE](https://arxiv.org/abs/2606.19980) `arXiv:2606.19980` | NVIDIA/CMU/UC Berkeley | 给编码 agent 的真实世界策略改进 harness：自动复位、执行、验证，再由 agent 改训练代码；灵巧任务成功率 99% |
| 2026-06 | [RATs (Playful Agentic Robot Learning)](https://arxiv.org/abs/2606.19419) `arXiv:2606.19419` | UC Berkeley | 编码 agent 在下游任务到来前"自己玩"：提出探索任务、执行、诊断失败、蒸馏技能 |
| 2026-03 | [Uni-Skill](https://arxiv.org/abs/2603.02623) `arXiv:2603.02623` | ICT, CAS | 自进化技能库，从视频派生的仓库里补全缺失技能 |
| 2024-05 | [Agentic Skill Discovery](https://arxiv.org/abs/2405.15019) `arXiv:2405.15019` | Univ. Hamburg | LLM 提出技能、生成奖励、用 RL 训练并验证，自主扩展技能库 |

### 前沿模型即策略（18）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [Frontier Models on LIBERO-10](https://arxiv.org/abs/2610.30051) `arXiv:2610.30051` |  | 6 个前沿模型 × 3 档推理强度：GPT-6 Astra 44%，Gemini-4 Argon 43%；推理越强一般越好，但代价更高 |
| 2026-10 | [GPT-6-Astra Understands Touch](https://arxiv.org/abs/2610.28457) `arXiv:2610.28457` |  | 把触觉作为上下文，5 个 UniVTAC 任务平均 93.8%，Insert Hole 31%→99% |
| 2026-10 | [GPT-6 as a Beginning](https://arxiv.org/abs/2610.28259) `arXiv:2610.28259` |  | RoboDojo 26.92%、RoboTwin 2.0 76.0%，但几何约束新基准 LIBERO-GeoMotion 只有 11.1% |
| 2026-10 | ★ [Has GPT-6-Astra Achieved Embodied Intelligence?](https://arxiv.org/abs/2610.22471) `arXiv:2610.22471` | SenseNova | 12 个基准 + 8 种真机配置，结论：具身能力"基本具备"，实用的具身智能"尚未" |
| 2026-10 | ★ [Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/) | Google DeepMind | ERQA 87%；能从视频/草图生成 MuJoCo 孪生与控制代码，生成的仿真数据把真机成功率从 56% 提到 75% |
| 2026-10 | [Awesome-Astra-Embodied-AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI) | 社区 | GPT-6 Astra 具身/机器人用例汇总：零样本控制、调用 VLA、real2sim、自建 RL 环境，约 44 个案例 |
| 2026-10 | [Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/) | NVIDIA SRL | 在 RoboLab-120 上比较 Astra 的三种角色：直接策略、Codex 中的 agentic 策略（74.8%）、编排基础模型的监督者，同时统计延迟与 token 成本 |
| 2026-09 | [LIBERO-Agent](https://arxiv.org/abs/2609.39507) `arXiv:2609.39507` |  | agent 原生的操控基准：自己挑观测、自己用工具处理、发原生动作命令；GPT-6 Astra 综合最强 |
| 2026-09 | ★ [GPT-6 Astra as Embodied Policies](https://arxiv.org/abs/2609.38537) `arXiv:2609.38537` | Galbot | 六个领域系统评测：导航强（RxR 92%），与 π0.5 混合控制 RoboDojo 48%；灵巧在手操作与步态弱；30 秒步态需 250 次调用、每次约 40 秒 |
| 2026-09 | [RoboICL](https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/) `arXiv:2609.34261` |  | 冻结的 GPT-6 Astra + 示范上下文 + 有界的锚定交互记忆，做具身上下文学习 |
| 2026-09 | [HomeBody](https://the-decoder.com/researchers-plug-gpt-6-astra-directly-into-a-robot-and-let-it-clean-up-an-unfamiliar-kitchen/) | Stanford/Caltech | GPT-6 Astra 不经 VLA，直接调用 5 个可组合技能，驱动 Unitree G1 在陌生厨房整理、取物（先建 Isaac Sim 数字孪生）；未公布成功率 |
| 2026-09 | [An Unexpected Robot Policy](https://arxiv.org/abs/2609.24170) `arXiv:2609.24170` |  | RoboDojo 42 个任务上，GPT-6 Astra 平均成功率 22.48%，超过全部 40 个公开策略；GPT-5.5 只有 0.88% |
| 2026-09 | [GPT-Policy](https://arxiv.org/abs/2609.19138) `arXiv:2609.19138` |  | 上下文编译器 + VLM 提议工具动作 + 约束控制器验证执行；人类视频示范即使没有动作标签也有帮助 |
| 2026-09 | [GPT-6 Astra 发布](https://www.cnbc.com/2026/09/03/open-ai-astra-gpt-6-cyber.html) | OpenAI | 官方主打电脑操作；发布后社区大量把它直接当机器人策略测试（可直接输出数值动作） |
| 2026-08 | [Model Hardware Standard (MHS)](https://www.bloomberg.com/news/articles/2026-08-27/anthropic-tests-new-way-for-claude-to-work-with-robots-and-scientific-lab-tools) | Anthropic | 研究预览：用简单的读/写命令 + 安全参数参考文件，让 agent 经 MCP/CLI/API 控制机械臂、显微镜等设备，对其他模型同样开放 |
| 2026-07 | ★ [Claude plays robotics](https://www.anthropic.com/research/claude-plays-robotics) | Anthropic | 5 家厂商的 12 个模型，跨扭矩直控、写控制器、训 RL、监督 VLA 四种接口：越高层越好用；实时直控约需 83Hz，而推理只有 0.2–0.4Hz |
| 2026-06 | ★ [Project Fetch 第二阶段](https://www.anthropic.com/news/project-fetch-phase-two) | Anthropic | Claude Opus 4.7 在 Claude Code 中独立完成：4 项共同任务用时 9 分 35 秒，对照最快人类团队 181 分钟（约 19×）；但仍没能完成自主"取回"球 |
| 2025-11 | ★ [Project Fetch](https://www.anthropic.com/research/project-fetch-robot-dog) | Anthropic | 两组零机器人经验的研究员给 Unitree Go2 编程捡球：用 Claude 的一组完成 7/8 项任务（对照组 6/8），共同任务用时约减半 |

### 世界模型/WAM（11）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [World Action Agent](https://arxiv.org/abs/2609.29964) `arXiv:2609.29964` |  | VLM 在执行前先做"世界-动作预演" |
| 2026-09 | [GE-Act 2.0](https://arxiv.org/abs/2609.05588) `arXiv:2609.05588` | AgiBot | 从零预训练并扩展的世界-动作模型（Genie Envisioner Act 2.0） |
| 2026-09 | [EmbodiedSkills](https://arxiv.org/abs/2609.01281) `arXiv:2609.01281` |  | 用 WAM 预测动作后果，再从冻结 VLA 的候选动作里挑 |
| 2026-08 | [HarnessWAM](https://arxiv.org/abs/2608.09516) `arXiv:2608.09516` |  | 给 WAM 配一个 VLM 任务管理器：维护任务图和证据化的场景信念，由事件触发审议 |
| 2026-08 | [DYNA-2](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-2-world-action-model-demonstrating-first-true-scaling-law-in-robotics-powered-entirely-by-human-data-302847114.html) | Dyna Robotics | 用超过 100 万小时第一人称人类视频训练的 WAM，宣称出现人到机器人的扩展规律 |
| 2026-03 | [Cosmos 3](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) | NVIDIA | GTC 2026 发布：把世界生成、视觉推理与动作仿真统一起来的世界基础模型 |
| 2026-02 | ★ [DreamZero](https://arxiv.org/abs/2602.15922) `arXiv:2602.15922` | NVIDIA GEAR | 14B 世界-动作模型，联合预测视频与动作，7Hz 闭环，泛化比 VLA 高 2 倍以上（GR00T N2 的基础） |
| 2026-02 | [NavDreamer](https://arxiv.org/abs/2602.09765) `arXiv:2602.09765` |  | 视频生成模型直接作零样本 3D 导航器 |
| 2026-01 | [1X World Model](https://www.1x.tech/ai) | 1X | 官方称其为 NEO 的"认知核心"：视频预测想象未来，再用逆动力学模型转成动作 |
| 2025-08 | [Genie 3](https://deepmind.google/models/) | Google DeepMind | 实时可交互的世界模型（720p/24fps），用于训练具身 agent；2026-01 以 Project Genie 形式开放 |
| 2025-06 | [V-JEPA 2](https://arxiv.org/abs/2506.09985) `arXiv:2506.09985` | Meta | 自监督视频世界模型；动作条件版 V-JEPA 2-AC 能零样本规划抓放 |

### 基准评测（20）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [Embodied Playground](https://arxiv.org/abs/2610.27039) `arXiv:2610.27039` |  | 410 个长时程 agentic 机器人环境，考察物理、策略与空间推理 |
| 2026-10 | ★ [RobotWorld](https://arxiv.org/abs/2610.10409) `arXiv:2610.10409` |  | 84 个任务覆盖操作、移动操作、步态、驾驶、飞行：GPT-6 Astra 19.0%，Claude Opus 5.5 15.5%，63 个任务所有模型都没完成 |
| 2026-10 | [RoboQuest](https://declare-lab.github.io/RoboQuest/) `arXiv:2610.10388` | NTU | 关键信息开局缺失，需要主动搜索、检查、试验才能拿到；最好的 agent 只有 23%，微调后的 π0.5 几乎为 0 |
| 2026-09 | [ActiveArena](https://arxiv.org/abs/2609.24124) `arXiv:2609.24124` |  | 操作中的主动感知基准 |
| 2026-09 | [LIBERO-RECOVER](https://arxiv.org/abs/2609.05178) `arXiv:2609.05178` |  | 不只看成功率，还评测失败后的恢复能力 |
| 2026-07 | [Aerial MLLM Mission Eval](https://arxiv.org/abs/2607.22014) `arXiv:2607.22014` |  | 空中 MLLM agent 的零样本任务级评测 |
| 2026-07 | [RoboDojo](https://robodojo-benchmark.com/) `arXiv:2607.04434` |  | 仿真 + 真机统一基准，带公开排行榜 |
| 2026-05 | [VLA-REPLICA](https://arxiv.org/abs/2605.20774) `arXiv:2605.20774` |  | 低成本、可复现的真机 VLA 基准（SO101 机械臂） |
| 2026-05 | [RoboMemArena](https://arxiv.org/abs/2605.10921) `arXiv:2605.10921` |  | 机器人记忆基准 |
| 2025-12 | [Embodied4C](https://arxiv.org/abs/2512.18028) `arXiv:2512.18028` |  | 具身视觉语言导航评测 |
| 2025-12 | [BEHAVIOR Challenge 2025 冠军方案](https://behavior.stanford.edu/challenge/leaderboard.html) `arXiv:2512.06951` |  | 基于 π0.5 做任务适配，50 个家务任务 Q-score 0.26（NeurIPS 2025 挑战赛第一） |
| 2025-10 | ★ [Butter-Bench](https://andonlabs.com/evals/butter-bench) `arXiv:2510.21860` | Andon Labs | LLM 当机器人"大脑"（TurtleBot 4，ReAct 循环）：最好的模型 40%，人类 95%；多步空间规划与社交理解最弱 |
| 2025-10 | [NavSpace](https://arxiv.org/abs/2510.08173) `arXiv:2510.08173` | PKU 等 | 导航 agent 对空间智能指令的跟随能力（ICRA'26） |
| 2025-10 | [LIBERO-PRO](https://arxiv.org/abs/2510.03827) `arXiv:2510.03827` |  | 扰动版 LIBERO，暴露 VLA 的记忆化问题，也是 2026 年 agent 方法的常用擂台 |
| 2025-06 | [RoboArena](https://robo-arena.github.io/) `arXiv:2506.18123` | 多机构合作 | 分布式真机双盲成对评测（DROID 平台）；2026-04 榜单 DreamZero 1750 Elo，π0.5 1622 |
| 2025-02 | ★ [EmbodiedBench](https://arxiv.org/abs/2502.09560) `arXiv:2502.09560` | UIUC 等 | MLLM 具身 agent 基准：高层任务 Claude-3.5 最好（约 64–68%），低层操作最好的 GPT-4o 也只有 28.9%（ICML'25） |
| 2025-01 | [EmbodiedEval](https://arxiv.org/abs/2501.11858) `arXiv:2501.11858` | Tsinghua | MLLM 作为具身 agent 的交互式评测 |
| 2024-12 | [VLABench](https://arxiv.org/abs/2412.18194) `arXiv:2412.18194` | Fudan | 需要长时程推理的语言条件操作基准 |
| 2024-12 | [VSI-Bench (Thinking in Space)](https://arxiv.org/abs/2412.14171) `arXiv:2412.14171` | NYU | 评测 MLLM 的视觉空间智能：看、记、回忆空间 |
| 2024-10 | ★ [Embodied Agent Interface](https://arxiv.org/abs/2410.07166) `arXiv:2410.07166` | Stanford | 把 LLM 具身决策拆成目标解释、子目标分解、动作序列、转移建模四个模块，细粒度评测（NeurIPS'24 D&B） |

### 安全（16）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | ★ [SafeHarness](https://arxiv.org/abs/2609.20822) `arXiv:2609.20822` | USC | 编码 agent 多数情况下会撞上明令禁止触碰的障碍物（它"知道"却不优先）；加障碍感知的路径/接触 harness 修复 |
| 2026-09 | [ARC](https://arxiv.org/abs/2609.12932) `arXiv:2609.12932` |  | 面向已部署自主系统的三层治理架构 |
| 2026-09 | [SafeMem](https://arxiv.org/abs/2609.08444) `arXiv:2609.08444` |  | 用长期图记忆做安全任务规划，视野外物体也保留在场景图里 |
| 2026-08 | [GuardianBench](https://arxiv.org/abs/2608.21928) `arXiv:2608.21928` |  | 同一场景只换指令：良性指令与安全场景组合后才出现的潜在风险 |
| 2026-06 | [SoK: FM 机器人的安全与隐私](https://arxiv.org/abs/2606.16788) `arXiv:2606.16788` |  | 从风险来源、传播路径到缓解措施位置的统一分析框架 |
| 2026-04 | [SafeGate](https://arxiv.org/abs/2604.05427) `arXiv:2604.05427` | Purdue | 参照 ISO 13482 的执行前安全闸门与任务安全契约 |
| 2026-02 | [CORE](https://arxiv.org/abs/2602.19983) `arXiv:2602.19983` | UPenn | VLM 在线推理情境化的安全约束，再用控制屏障函数强制执行 |
| 2026-02 | [AgentRob](https://arxiv.org/abs/2602.13591) `arXiv:2602.13591` |  | 论坛里的 agent 通过 MCP 调度真实 Go2/G1，说明"被劫持的物理机器人"风险 |
| 2025-12 | [RoboSafe](https://arxiv.org/abs/2512.21220) `arXiv:2512.21220` | Beihang | 基于可执行谓词安全逻辑的运行时防护 |
| 2025-09 | [ASIMOV-2.0](https://asimov-benchmark.github.io/v2) `arXiv:2509.21651` | Google DeepMind | 物理危险感知与干预基准，分伤害、本体约束、视频风险三部分 |
| 2025-06 | [AgentSafe](https://arxiv.org/abs/2506.14697) `arXiv:2506.14697` |  | 具身 VLM agent 面对危险指令时的安全基准（CVPR'26） |
| 2025-03 | [ASIMOV (Robot Constitutions)](https://arxiv.org/abs/2503.08663) `arXiv:2503.08663` | Google DeepMind | 生成"机器人宪法"与语义安全基准 |
| 2025-03 | ★ [RoboGuard](https://arxiv.org/abs/2503.07885) `arXiv:2503.07885` | UPenn | 由隔离模型生成情境化安全规范，再做时序逻辑模型检查；越狱成功率 92.3%→2.3% |
| 2024-10 | ★ [RoboPAIR](https://arxiv.org/abs/2410.13691) `arXiv:2410.13691` | UPenn | 首个越狱 LLM 控制机器人的算法：自驾 LLM、GPT 规划的地面机器人、商用四足全部被攻破 |
| 2024-07 | [BadRobot](https://arxiv.org/abs/2407.20242) `arXiv:2407.20242` |  | 在物理世界中越狱具身 LLM agent |
| 2023-09 | [Safety Chip](https://arxiv.org/abs/2309.09919) `arXiv:2309.09919` | Brown | 把专家验证的 LTL 约束编译成自动机，执行前剪掉不安全动作 |

### 导航（10）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [HarnessVLN](https://arxiv.org/abs/2609.15195) `arXiv:2609.15195` | AgiBot | 免训练的统一导航 agent harness |
| 2026-08 | [NavMCP](https://arxiv.org/abs/2608.30396) `arXiv:2608.30396` | SJTU | VLM agent 决定找什么、去哪找、何时停，导航基础模型负责执行；长时程探索 78.3% |
| 2026-07 | [ABot-N1](https://arxiv.org/abs/2607.10383) `arXiv:2607.10383` | Alibaba AMAP | 用慢-快架构解耦认知与控制的通用 VLN 基础模型 |
| 2026-02 | [ABot-N0](https://arxiv.org/abs/2602.11598) `arXiv:2602.11598` | Alibaba AMAP | 统一 5 类导航任务的 VLA（LLM 认知大脑 + 流匹配动作专家） |
| 2025-12 | [DualVLN](https://arxiv.org/abs/2512.08186) `arXiv:2512.08186` | Shanghai AI Lab | "慢推理、快运动"的双系统导航基础模型 |
| 2024-12 | [Uni-NaVid](https://arxiv.org/abs/2412.06224) `arXiv:2412.06224` | PKU | 基于视频的 VLA，统一多种导航任务 |
| 2024-12 | [NaVILA](https://arxiv.org/abs/2412.04453) `arXiv:2412.04453` | UCSD/NVIDIA | 腿式机器人导航 VLA：VLM 输出语言化动作，交给步态策略执行 |
| 2023-05 | [NavGPT](https://arxiv.org/abs/2305.16986) `arXiv:2305.16986` | Adelaide | 用 LLM 的显式推理做视觉语言导航 |
| 2022-10 | [VLMaps](https://arxiv.org/abs/2210.05714) `arXiv:2210.05714` | Freiburg/Google | 视觉语言地图，支持开放词汇的空间指令 |
| 2022-07 | [LM-Nav](https://arxiv.org/abs/2207.04429) `arXiv:2207.04429` | UC Berkeley/Google | LLM + CLIP + 视觉导航模型零样本组合 |

### 多机器人/多智能体（11）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [AeroWeaver](https://arxiv.org/abs/2609.18520) `arXiv:2609.18520` |  | 给空中集群编织技能、分配角色的具身 agent harness |
| 2026-08 | [MistyPilot](https://arxiv.org/abs/2608.15549) `arXiv:2608.15549` |  | 多 agent LLM 技能编排，控制社交机器人 |
| 2026-04 | [M2HRI](https://arxiv.org/abs/2604.11975) `arXiv:2604.11975` |  | LLM 驱动的多模态多 agent 个性化人机交互 |
| 2026-03 | [IMR-LLM](https://arxiv.org/abs/2603.02669) `arXiv:2603.02669` |  | 工业多机器人任务规划与程序生成 |
| 2026-01 | [EmboTeam](https://arxiv.org/abs/2601.11063) `arXiv:2601.11063` |  | LLM 推理经 PDDL 落地成反应式行为树，用于多机协作 |
| 2025-11 | [Monolithic→Multi-Agent](https://arxiv.org/abs/2512.00797) `arXiv:2512.00797` |  | 把单体基础模型改造成具身多 agent 架构，用于人机协作 |
| 2025-05 | [RoboOS](https://arxiv.org/abs/2505.03673) `arXiv:2505.03673` | BAAI | 跨本体、多 agent 协作的分层具身框架 |
| 2024-10 | [EMOS](https://arxiv.org/abs/2410.22662) `arXiv:2410.22662` |  | 本体感知的异构多机器人 OS：从 URDF 和正运动学自动生成"机器人简历"（ICLR'25） |
| 2023-09 | [SMART-LLM](https://arxiv.org/abs/2309.10062) `arXiv:2309.10062` | Purdue | LLM 完成多机器人任务分解、联盟形成与分配 |
| 2023-07 | [RoCo](https://arxiv.org/abs/2307.04738) `arXiv:2307.04738` | Columbia | 多个机械臂通过 LLM 对话协商计划，执行前做可行性检查 |
| 2023-07 | [CoELA](https://arxiv.org/abs/2307.02485) `arXiv:2307.02485` | UMass 等 | 模块化的合作具身 LLM agent，含通信模块 |

### 人形（7）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | [HumanCLAW](https://arxiv.org/abs/2607.27180) `arXiv:2607.27180` |  | 把 VLM 的动作决策和底层全身运动解耦的评测框架 |
| 2026-03 | [Cybo-Waiter](https://arxiv.org/abs/2603.10675) `arXiv:2603.10675` |  | 人形全身移动操作的物理 agent 框架 |
| 2025-12 | [WholeBodyVLA](https://arxiv.org/abs/2512.11047) `arXiv:2512.11047` |  | 统一潜在 VLA 的全身移动操作（ICLR'26） |
| 2025-06 | [Humanoid VL Planning](https://arxiv.org/abs/2506.22827) `arXiv:2506.22827` |  | RL 全身跟踪 + 模仿技能 + VLM 规划监控三层；Unitree G1 上成功率 72.5% |
| 2025-06 | [LeVERB](https://arxiv.org/abs/2506.13751) `arXiv:2506.13751` | UC Berkeley 等 | 用潜在视觉-语言指令驱动人形全身控制 |
| 2025-04 | [LangWBC](https://arxiv.org/abs/2504.21738) `arXiv:2504.21738` | UC Berkeley | 语言指令端到端控制人形全身 |
| 2025-03 | [Being-0](https://arxiv.org/abs/2503.12533) `arXiv:2503.12533` | PKU/BeingBeyond | 基础模型做高层认知，经 Connector VLM 衔接模块化技能的人形 agent |

### 虚拟世界智能体（2）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-12 | [SIMA 2](https://arxiv.org/abs/2512.04797) `arXiv:2512.04797` | Google DeepMind | Gemini 驱动的 3D 虚拟世界通用 agent，能在 Genie 3 生成的新世界里自我改进 |
| 2023-05 | ★ [Voyager](https://arxiv.org/abs/2305.16291) `arXiv:2305.16291` | NVIDIA/Caltech | Minecraft 里由 GPT-4 驱动的终身学习 agent：自动课程 + 代码技能库 + 自我验证，后来"技能库"范式的源头 |

### 自动驾驶（4）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-05 | [MindVLA-U1](https://arxiv.org/abs/2605.12624) `arXiv:2605.12624` |  | 统一流式架构的驾驶 VLA |
| 2026-03 | [Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5) | NVIDIA | 开放的 10B 推理型驾驶 VLA（Cosmos-Reason2 骨干 + 扩散轨迹解码），支持语言导航指引 |
| 2026-02 | [SteerVLA](https://arxiv.org/abs/2602.08440) `arXiv:2602.08440` |  | 在长尾驾驶场景中操控 VLA |
| 2025-06 | [VLA4AD 综述](https://arxiv.org/abs/2506.24044) `arXiv:2506.24044` |  | 自动驾驶 VLA 综述 |

### 综述/资源（23）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | ★ [Towards Agentic Robots](https://arxiv.org/abs/2610.20442) `arXiv:2610.20442` | CUHK MMLab/Fudan/Tsinghua 等 | Model/Data/Environment/Harness 四大域、17 类，目前对 agentic robots 最系统的综述（2026-10） |
| 2026-09 | ★ [Survey on Multimodal Embodied Agents](https://engrxiv.org/preprint/view/8364) | NUS Show Lab | 从 computer-use 到 robot-use 的统一能力视角，提出 PAPAV 框架（感知-预判-规划-行动-验证） |
| 2026-08 | ★ [Weights or Skills?](https://arxiv.org/abs/2608.01851) `arXiv:2608.01851` |  | "把能力烤进权重（VLA）"与"agent 自己写技能（代码）"两条路线；按自我改进程度梳理代码即策略 |
| 2026-03 | [Industrial RFM Survey](https://arxiv.org/abs/2603.06749) `arXiv:2603.06749` |  | 机器人基础模型在工业中的可用性综述与评估框架 |
| 2025-10 | [Unified Robot Manipulation Survey](https://arxiv.org/abs/2510.10903) `arXiv:2510.10903` |  | 机器人操作的统一视角综述 |
| 2025-10 | [VLA Review toward Real-World](https://arxiv.org/abs/2510.07077) `arXiv:2510.07077` |  | 面向真实世界应用的 VLA 综述 |
| 2025-09 | [Embodied AI: From LLMs to World Models](https://arxiv.org/abs/2509.20021) `arXiv:2509.20021` |  | 从 LLM 到世界模型的具身 AI 综述 |
| 2025-09 | [Pure VLA Survey](https://arxiv.org/abs/2509.19012) `arXiv:2509.19012` |  | 按自回归、扩散、强化、混合等路线分类的 VLA 综述 |
| 2025-08 | [Large VLM-based VLA Survey](https://arxiv.org/abs/2508.13073) `arXiv:2508.13073` |  | 基于大 VLM 的操作 VLA 综述 |
| 2025-08 | [Towards Embodied Agentic AI](https://arxiv.org/abs/2508.05294) `arXiv:2508.05294` |  | LLM/VLM 驱动的机器人自主与交互：分类与集成视角 |
| 2025-07 | [FM-driven Robotics Review](https://arxiv.org/abs/2507.10087) `arXiv:2507.10087` |  | 基础模型驱动的机器人综述 |
| 2025-07 | [VLA: Action Tokenization Perspective](https://arxiv.org/abs/2507.01925) `arXiv:2507.01925` | PKU | 从"动作 token 化"角度看 VLA |
| 2025-06 | [Embodied AI Agents: Modeling the World](https://arxiv.org/abs/2506.22355) `arXiv:2506.22355` | Meta | Meta 的具身 agent 观点文章：以世界模型为核心 |
| 2025-05 | [FMs for Mobile Service Robots](https://arxiv.org/abs/2505.20503) `arXiv:2505.20503` | Univ. Toronto | 移动服务机器人上的基础模型系统性综述 |
| 2025-05 | [Multi-agent Embodied AI](https://arxiv.org/abs/2505.05108) `arXiv:2505.05108` |  | 多智能体具身 AI 的进展与方向 |
| 2025-05 | [VLA: Concepts, Progress, Applications](https://arxiv.org/abs/2505.04769) `arXiv:2505.04769` |  | VLA 的概念、进展、应用与挑战 |
| 2024-07 | [Aligning Cyber Space with Physical World](https://arxiv.org/abs/2407.06886) `arXiv:2407.06886` | Sun Yat-sen Univ. | 具身 AI 全景综述 |
| 2024-05 | [Survey on VLA for Embodied AI](https://arxiv.org/abs/2405.14093) `arXiv:2405.14093` | CUHK 等 | 较早的 VLA 综述，持续更新（2026-05 到 v8） |
| 2024-02 | [Real-World Robot Applications of FMs](https://arxiv.org/abs/2402.05741) `arXiv:2402.05741` | Univ. Tokyo | 基础模型在真实机器人上的应用综述 |
| 2024-02 | [Robotics with FMs toward Embodied AI](https://arxiv.org/abs/2402.02385) `arXiv:2402.02385` |  | 基础模型机器人走向具身 AI 的综述 |
| 2023-12 | [Toward General-Purpose Robots via FMs](https://arxiv.org/abs/2312.08782) `arXiv:2312.08782` | CMU 等 | 基础模型通往通用机器人：综述与元分析 |
| 2023-12 | [Foundation Models in Robotics](https://arxiv.org/abs/2312.07843) `arXiv:2312.07843` | Stanford 等 | 应用、挑战与未来 |
| 2023-11 | [LLMs for Robotics: A Survey](https://arxiv.org/abs/2311.07226) `arXiv:2311.07226` |  | LLM 用于机器人的综述 |

### 产业发布（13）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [Pi API beta](https://www.pi.website/blog/api-beta) | Physical Intelligence | 上传机器人数据、在托管算力上微调 π 模型、云端推理接入自己的机器人 |
| 2026-10 | [OpenAI Decisions API](https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877) | OpenAI | 让模型（gpt-6-luna）只回答有限选项的判断/选择/打分，速度快且有概率输出，可作 harness 里的快速决策层 |
| 2026-08 | [Skild S1](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/) | Skild AI | 看一段任务视频即可学会最长 10 分钟的任务并实时执行 |
| 2026-07 | ★ [Gemini Robotics 2 / ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/) | Google DeepMind | ER 2：实时视频理解、任务进度跟踪、Live API 流式低延迟编排、工具调用、多机器人协作；同系列还有 VLA 与 On-Device 2 |
| 2026-06 | [OpenAI Robotics](https://theaiinsider.tech/2026/06/02/sam-altman-says-openai-now-has-a-dedicated-robotics-initiative-hiring-engineers/) | OpenAI | 独立机器人部门（Aditya Ramesh 领导，源自 world simulation 研究），10 月确认自研人形，计划 2027 年公开演示 |
| 2026-04 | [Gemini Robotics-ER 1.6](https://deepmind.google/blog/gemini-robotics-er-1-6/) | Google DeepMind | 空间与多视角推理增强，新增仪表读数（与 Boston Dynamics 合作），官方称"迄今最安全的机器人模型" |
| 2026-04 | [AgiBot GO-2](https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/) | AgiBot | 一个系统里统一推理与执行：先生成动作意图序列作宏观计划，异步双系统分离推理与控制 |
| 2026-04 | [GEN-1](https://www.therobotreport.com/generalist-introduces-gen-1-general-purpose-model-for-physical-ai/) | Generalist AI | 约 50 万小时数据从零训练；所选任务平均成功率 99%（GEN-0 为 64%），每项任务只需约 1 小时机器人数据 |
| 2026-03 | [GR00T N1.7 / N2 预览](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) | NVIDIA | GTC 2026：N1.7 提供商业授权早期访问；N2 预览，基于 DreamZero 的"世界-动作模型"架构 |
| 2026-01 | [Rho-alpha (ρα)](https://www.therobotreport.com/microsoft-research-reveals-rho-alpha-vision-language-action-model-for-robots/) | Microsoft Research | 基于 Phi 系列的首个机器人模型，"VLA+"：加入触觉与基于人类反馈的持续学习 |
| 2025-12 | [GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) | NVIDIA | 3B 开源 VLA：Cosmos-Reason-2B 骨干 + 32 层 DiT 动作头，支持人形边走边操作（发布时间各来源说法不一，2025-10 至 2026-01） |
| 2025-11 | [GEN-0](https://eu.36kr.com/en/p/3551715251804547) | Generalist AI | 用 27 万小时真实操作数据训练，宣称在 7B 以上出现机器人扩展规律 |
| 2025-09 | [Gemini Robotics 1.5 发布](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) | Google DeepMind | 官方博客："把 AI agent 带进物理世界"，ER 1.5 编排、VLA 执行，ER 1.5 通过 API 开放 |

