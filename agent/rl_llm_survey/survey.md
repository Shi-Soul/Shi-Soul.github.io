# 大模型的强化学习：写给 RL 研究者的综述（LLM · VLM · 智能体）

> 面向熟悉 MDP、策略梯度与 PPO，但没有训练过大模型的读者。全文把 LLM、VLM 与智能体的后训练写成 RL 问题，按根本问题分十三章：每章先讲已经确立的原理与做法（含推导），再写最近一年（2025-11 → 2026-10）的进展。数据截至 2026-10-31；论文库 908 条（arXiv 条目已用官方 API 核验标题与日期），全部列在附录 B。交互式导读页（含时间线、按章节 × 月份的热力图与可筛选的论文库）见 [index.html](./)。
>
> 读法建议：第 0 章与第 1 章建立对照；第 3 章（算法）与第 5 章（探索与能力边界）是 RL 研究者最熟悉、也最容易误判的地方；第 13 章给出可以直接运行的最小实验。

## 0 导读：把后训练读成一个 RL 问题

### 0.1 两段训练

大模型的训练分两段。**预训练**在数万亿 token 上做下一个 token 预测：给定前文，最大化真实下一个 token 的对数概率。它得到的是一个"会说话、懂知识"的条件分布，但不一定听指令，也不一定会把题做对。**后训练**把它变成助手、推理模型或智能体，主要手段有两个：监督微调（SFT，在人工或强模型写的示范上继续做最大似然，相当于行为克隆），以及强化学习。本文讨论的是后者，以及它的各种近亲（直接偏好优化、拒绝采样、在策略蒸馏）。

### 0.2 一个目标函数

几乎所有方法都在优化或近似优化同一个 KL 正则目标：

$$\max_\theta\ \mathbb E_{x\sim\mathcal D}\Big[\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big]-\beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)\Big]$$

其中 $x$ 是提示（题目、指令或任务描述），$y=(y_1,\dots,y_T)$ 是模型生成的回答，每个 $y_t$ 是词表 $\mathcal V$ 中的一个 token（词表大小约 15 万）；语言模型就是策略 $\pi_\theta(y\mid x)=\prod_{t}\pi_\theta(y_t\mid x,y_{<t})$；$\pi_{\text{ref}}$ 是起点模型，通常是 SFT 之后的模型；$r$ 是奖励。对每个 $x$ 单独求最优，得到熵正则 RL 的标准结论：

$$\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(r(x,y)/\beta\big),\qquad Z(x)=\sum_{y}\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}$$

这就是以 $\pi_{\text{ref}}$ 为先验的软最优策略（[策略梯度与软 Q 学习的等价](https://arxiv.org/abs/1704.06440)）。第 4 章的 DPO、第 5 章的"锐化"论证、第 11 章的大部分理论都从这个式子出发。

### 0.3 两种视角

**按整段回答看**，这是一个上下文赌博机：提示是上下文，整段回答是一次"拉臂"，奖励在回答结束时给出。奖励函数、KL 约束与最优解都在这个层面定义。

**按 token 看**，这是一个确定性转移的 MDP：状态 $s_t=(x,y_{<t})$，动作 $a_t=y_t$，转移 $s_{t+1}=s_t\oplus a_t$（拼接），奖励通常只在最后一步非零。梯度在这个层面计算：

$$\nabla_\theta J=\mathbb E_{y\sim\pi_\theta}\Big[\big(r(x,y)-b(x)\big)\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid s_t)\Big]$$

两种视角的张力贯穿全文：奖励是序列级的，而重要性比、裁剪、优势与损失聚合都可以放在 token 级或序列级（第 3 章）；信用分配要把序列级的奖励拆回到 token 或"步骤"（第 6 章）。智能体任务里，工具与环境的返回插进序列，转移才变得随机、部分可观测（第 9 章）。

### 0.4 RL 概念对照

| RL 概念 | 在大模型后训练中 | 说明 |
|---|---|---|
| 策略 | 语言模型的下一 token 分布 | Transformer 输出的 softmax；采样温度 $\tau$ 把 logits 除以 $\tau$，$\tau=1$ 才是在策略 |
| 状态 | 提示 + 已生成的前缀 | 单轮任务完全可观测；长度从几百到十几万 token |
| 动作 | token；或句子、工具调用、整段回答 | 粒度决定重要性比与优势在哪一层计算 |
| 回合 | 一次完整回答 | 推理模型先在 `<think>` 中写长思维链，再给答案 |
| 奖励 | 奖励模型打分、程序验证、评判模型按细则打分 | 多在结尾给一次；三种来源各有投机方式 |
| 行为克隆 | SFT | RL 之前的"冷启动" |
| 先验 / 参考策略 | 预训练模型 / SFT 模型 | KL 惩罚把策略拉向它 |
| actor、rollout | 推理引擎（vLLM、SGLang）批量生成 | 生成占一步训练的大部分时间 |
| critic、基线 | PPO 的价值头；GRPO 用同题多次采样的均值 | "同一状态可重放多次"是 LLM 的便利 |
| 环境 | 题库 + 验证器；沙箱、浏览器、操作系统 | 可验证环境的规模化是最近一年的主线 |
| 探索 | 温度、每题采样数、策略熵 | 用 pass@k 衡量 |
| 离线 RL | DPO 等直接偏好优化、拒绝采样微调 | 便宜，受数据覆盖限制 |
| 规划 | 长思维链、投票、Best-of-N、树搜索 | 测试时计算本身被 RL 训练出来 |
| 奖励投机 | reward hacking | 优化压力越大越明显 |

### 0.5 与经典 RL 的六点不同

1. **初始策略极强。** 预训练模型已经会解很多题，RL 的主要作用常常是重新分配已有解的概率（第 5、11 章）。起点模型是否具备验证、回溯等"认知习惯"，直接决定 RL 能否奏效（[Gandhi 等](https://arxiv.org/abs/2503.01307)）。
2. **转移确定、可任意重启。** 单轮任务没有环境随机性，可以从任意前缀重新采样。于是"同题多次采样当基线"（GRPO）、"从中间状态续写估计价值"（VinePPO）、"在前缀树上分支"（TreePO）都很自然。经典 RL 里这需要模拟器的重置能力。
3. **动作空间巨大，有效分支很少。** 词表 15 万，但大多数位置上概率集中在一两个 token；熵集中在约两成的"分叉 token"上（[80/20](https://arxiv.org/abs/2506.01939)），探索与信用都发生在那里。
4. **奖励稀疏且来源特殊。** 学到的奖励模型可被投机；验证器只覆盖数学、代码等少数领域；开放任务依赖评判模型与评分细则（第 2 章）。
5. **时域长、不打折扣。** 一个回答上万 token，常用 $\gamma=\lambda=1$；动作是 token，但真正的"决策"是句子或工具调用，步长的划分本身是设计选择（第 6、9 章）。
6. **算力花在生成上。** 每步训练先要生成几万条长序列；生成与训练分属两套引擎、两种数值实现，"在策略"在工程上总带一点离策略，异步训练又引入陈旧度（第 3、10 章）。

### 0.6 四个阶段

- **2017–2022：RLHF。** 人类比较两个回答 → 训练 Bradley–Terry 奖励模型 → PPO + KL 惩罚（[Christiano 等](https://arxiv.org/abs/1706.03741)、[Ziegler 等](https://arxiv.org/abs/1909.08593)、[Stiennon 等](https://arxiv.org/abs/2009.01325)、[InstructGPT](https://arxiv.org/abs/2203.02155)）。瓶颈：策略、价值、奖励、参考四个模型同时在显存里；奖励模型可被投机。
- **2023–2024：绕开在线 RL。** [DPO](https://arxiv.org/abs/2305.18290) 把 KL 正则目标的闭式解代回偏好损失，用分类式训练替代 PPO；拒绝采样、迭代 DPO 成为开放模型的主流配方（[Llama 3](https://arxiv.org/abs/2407.21783)）。瓶颈：离线数据覆盖不到策略的新行为。
- **2024-09 – 2025：可验证奖励与推理。** [o1](https://openai.com/index/learning-to-reason-with-llms/) 与 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 表明，以答案正确与否为奖励（RLVR）做大规模 RL，模型会自己学会长思维链、自检与回溯；[GRPO](https://arxiv.org/abs/2402.03300) 去掉 critic，[DAPO](https://arxiv.org/abs/2503.14476) 等补齐稳定性。瓶颈：只有少数领域可验证。
- **2025 下半年 – 2026：智能体、蒸馏与规模化。** RL 进入多轮工具环境；在策略蒸馏与自蒸馏成为与 RL 并列的手段；评分细则把 RL 推向开放任务；异步系统、低精度 rollout 与训推一致成为工程主线。

### 0.7 最近半年的八个动向（2026-05 → 2026-10）

1. **在策略蒸馏与 RL 并列**，变体激增，也出现"离线学生 rollout 往往更好"的反思（[Semi-OPD](https://arxiv.org/abs/2610.11291)）。第 4 章。
2. **自蒸馏把反馈变成稠密信号**（[SDPO](https://arxiv.org/abs/2601.20802)），随后是对特权信息与崩溃机制的系统检验，以及智能体场景中的复现检验（[现实检验](https://arxiv.org/abs/2610.19789)）。第 4、9 章。
3. **评分细则奖励走向主流，也被投机**（[FORK](https://arxiv.org/abs/2610.33512)、[加法聚合](https://arxiv.org/abs/2609.38847)、[ImpossibleRubrics](https://arxiv.org/abs/2609.16816)）。第 2、12 章。
4. **critic 回归**（[EasyPPO](https://arxiv.org/abs/2609.36802)、[S3AC](https://arxiv.org/abs/2610.35518)），价值模型开始被预训练（[LenVM](https://arxiv.org/abs/2604.27039)）。第 3、6 章。
5. **训推一致成为系统问题**，低精度 rollout 让它更尖锐（[CIS](https://arxiv.org/abs/2609.32444)、[TRIAGE](https://arxiv.org/abs/2610.07043)）。第 10 章。
6. **可验证环境的规模化**，模型开始自己写环境（[SWE-Universe](https://arxiv.org/abs/2602.02361)、[SPADE](https://arxiv.org/abs/2608.19197)）。第 7、9 章。
7. **对技巧的冷静复查**：数据技巧没有显著优于均匀采样（[DataFlex-RL](https://arxiv.org/abs/2609.06107)），RLVR 权重轨迹近乎秩 1（[RELEX](https://arxiv.org/abs/2605.21468)）。第 7、11 章。
8. **多奖励与多能力整合**（[DVAO](https://arxiv.org/abs/2605.25604)、[MRTPO](https://arxiv.org/abs/2610.21439)、[MOPD](https://arxiv.org/abs/2606.30406)）。第 3、4 章。

## 1 问题设定：把语言模型看作策略

### 1.1 序列级奖励、token 级梯度

设奖励只依赖完整回答。对序列级目标 $J(\theta)=\mathbb E_{y\sim\pi_\theta}[r(x,y)]$ 用似然比技巧，由 $\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid s_t)$ 立即得到 0.3 节的 REINFORCE 式。几点值得强调：

- 回答的奖励被"平均地"记到每个 token 上：写错一个数字与写了一段漂亮推理，得到同样的信号。这是第 6 章信用分配问题的来源。
- 基线 $b(x)$ 只要不依赖 $y$ 就不引入偏差。LLM 上最自然的基线是"同一个提示的其他样本"：用贪心解码的得分（[SCST](https://arxiv.org/abs/1612.00563)、[ReMax](https://arxiv.org/abs/2310.10505)）、其他样本的均值（[RLOO](https://arxiv.org/abs/2402.14740)）或组内均值与标准差（[GRPO](https://arxiv.org/abs/2402.03300)）。
- 用 REINFORCE 直接优化序列级指标在 2015–2016 年就有了：[MIXER](https://arxiv.org/abs/1511.06732) 优化 BLEU，[序列 actor-critic](https://arxiv.org/abs/1607.07086) 训练了 token 级的 critic，[对话 RL](https://arxiv.org/abs/1606.01541) 用手工设计的长期奖励。它们没有流行起来，主要是因为当时的初始策略太弱、奖励太粗。

### 1.2 KL 约束：从先验出发

KL 可以拆进每个 token：

$$\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)=\mathbb E_{y\sim\pi_\theta}\Big[\sum_{t}\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)}\Big]$$

因此可以把 $-\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 当作逐 token 的奖励，与结尾的 $r(x,y)$ 一起交给 PPO（RLHF 的经典实现）；也可以把 KL 的某个估计量直接加进损失（GRPO 的做法）。两种放法的梯度并不相同，第 3.5 节再讨论。

用 KL 把微调约束在预训练模型附近，最早出现在序列生成的 RL 中：[Sequence Tutor](https://arxiv.org/abs/1611.02796) 用它防止音乐生成模型在 RL 后"忘记"怎么写合理的旋律，[Jaques 等](https://arxiv.org/abs/1907.00456)把它用于从人类交互日志做离线 RL。动机与今天相同：奖励只覆盖了我们关心的一小部分性质，其余的（流畅、事实、风格）要靠先验保住。

推理 RL 中常常干脆去掉 KL（[DAPO](https://arxiv.org/abs/2503.14476)、[ORZ](https://arxiv.org/abs/2503.24290)）：可验证奖励不容易被投机，KL 反而限制了模型走出 SFT 的分布。但"去掉 KL"不等于"没有约束"——裁剪、学习率与有限的训练步数都在隐式地约束偏离，第 11 章的 RL's Razor 说明在策略训练本身就偏向 KL 小的解。

### 1.3 后训练流水线

一个典型的推理模型流水线：

1. **预训练**：下一 token 预测，学到语言、知识与大部分"能力"的原材料。
2. **中期训练**：提高数学、代码、长上下文与推理数据的比例。它决定 RL 的起点能走多远（[OctoThinker](https://arxiv.org/abs/2506.20512)、[受控研究](https://arxiv.org/abs/2512.07783)）。
3. **SFT 冷启动**：在长思维链示范（人工写的，或从更强的推理模型蒸馏来的）上做行为克隆，让模型学会格式与基本的推理模式。
4. **RL**：RLHF（偏好）、RLVR（可验证奖励）或智能体 RL，往往分阶段、按领域级联（[Nemotron-Cascade](https://arxiv.org/abs/2512.13607)）。
5. **蒸馏**：把大模型的能力迁移到小模型，常用拒绝采样 SFT 或在策略蒸馏（第 4 章）。

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 同时给出了两条路线：R1-Zero 从基座直接做 RL，涌现出长思维链，但可读性差、语言混杂；R1 先用少量冷启动数据 SFT，再做推理 RL、拒绝采样 SFT、全场景 RL 的多阶段流程。[LIMA](https://arxiv.org/abs/2305.11206) 提出的"表面对齐假说"——知识与能力来自预训练，对齐只是学会格式与风格——在推理时代变成了一个更尖锐的问题：RL 到底是在教新东西，还是在挑出已有的东西（第 5 章）？

### 1.4 容易被忽视的接口

对 RL 研究者来说，以下几处最容易出错，因为它们在经典 RL 里没有对应物：

- **聊天模板。** 模型看到的不是裸文本，而是带角色标记的模板（系统、用户、助手），推理模型还有 `<think>` 标记。训练与评测用的模板不一致，相当于换了一个状态空间。混合思考模型只靠几个模板 token 切换"思考 / 不思考"，在一种模式下训练会影响另一种（[思考泄漏](https://arxiv.org/abs/2610.14270)）。
- **采样参数。** 推理引擎与模型自带的生成配置常有 top-k、top-p、重复惩罚；它们改变了采样分布，rollout 就不再来自 $\pi_\theta$。要么显式关掉，要么在重要性比中计入。
- **截断。** 最大生成长度是一个强制终止。截断的回答记 0 分，会教模型写短；记负分，会教模型不冒险；直接丢弃，会引入选择偏差（[DAPO](https://arxiv.org/abs/2503.14476) 用了"超长惩罚"的软版本）。
- **结束符。** 停止生成本身是一个需要学会的动作；损失掩码要正确地包含第一个结束符、排除其后的填充。
- **数值。** 同一组权重在推理引擎与训练框架中算出的概率并不相同（第 10 章）。

### 1.5 最近一年

- **区分预训练与 RL 的贡献。** 在原子操作明确的合成推理任务上做受控实验：RL 只有在预训练留有余量、任务处在能力边缘时，才带来向更复杂组合的外推；中期训练决定了 RL 的起点（[预训练、中期训练与 RL](https://arxiv.org/abs/2512.07783)）。半年后同一思路推广到更真实的设定，追踪预训练数据如何决定 RL 能学到什么（[从预训练到后训练理解推理](https://arxiv.org/abs/2607.16097)），多轮规划也被拆成预训练、SFT 与蒸馏三个阶段研究（[多轮规划的物理学](https://arxiv.org/abs/2607.24720)）。
- **动作不必是 token。** 预训练模型的残差流中存在可用于分层控制的时间抽象；在这些"选项"上做内部 RL，能解决 token 级 RL 无法解决的稀疏奖励任务（[时间抽象与分层 RL](https://arxiv.org/abs/2512.20605)）。这把经典的选项框架带回了 LLM。
- **RL 前移。** 在语料上主动挑选要遮盖的片段再预测（[PretrainZero](https://arxiv.org/abs/2512.03442)）；把网页文本改写成填空式多选题，得到几乎无限的 RLVR 数据（[Golden Goose](https://arxiv.org/abs/2601.22975)）；此前已有强化中期训练（[RMT](https://arxiv.org/abs/2509.24375)）与在预训练语料上做 RL（[RLPT](https://arxiv.org/abs/2509.19249)）。

## 2 奖励从哪里来：偏好、验证器与评分细则

### 2.1 偏好与 Bradley–Terry

RLHF 的奖励来自比较：给定提示 $x$ 与两个回答，标注者选出更好的 $y_w$。Bradley–Terry 模型假设 $p(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$，奖励模型 $r_\phi$ 用最大似然训练：

$$\mathcal L(\phi)=-\mathbb E_{(x,y_w,y_l)}\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$$

几点性质：奖励只确定到一个与 $x$ 有关的常数，所以常按提示归一化；比较数据比打分数据更一致；奖励模型通常由 SFT 模型初始化、把语言模型头换成一个标量头。这条路线来自 [Christiano 等](https://arxiv.org/abs/1706.03741)的轨迹片段比较，经 [Ziegler 等](https://arxiv.org/abs/1909.08593)、[Stiennon 等](https://arxiv.org/abs/2009.01325)到 [InstructGPT](https://arxiv.org/abs/2203.02155)；[HH-RLHF](https://arxiv.org/abs/2204.05862) 公开了偏好数据并研究了有用与无害的张力，[Sparrow](https://arxiv.org/abs/2209.14375) 把人类判断按规则分解。[Constitutional AI](https://arxiv.org/abs/2212.08073) 与 [RLAIF](https://arxiv.org/abs/2309.00267) 用 AI 的比较替代人类，效果相当。

偏好的问题在于它奖励的是"让标注者满意"：标注者偏爱更长、更自信、与自己观点一致的回答，RLHF 会放大这些偏好（[谄媚](https://arxiv.org/abs/2310.13548)、[RLHF 让模型学会误导](https://arxiv.org/abs/2409.12822)，第 12 章）。

### 2.2 可验证奖励

对有标准答案的任务，奖励可以直接由程序判定：

$$r(x,y)=\mathbb 1\big[\mathrm{verify}(\mathrm{extract}(y),\,y^\star)\big]$$

数学题抽取最终答案做符号等价判定；代码跑单元测试；形式化证明交给 Lean 检查（[DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152)、[AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)）；还可以加一项格式奖励。"RLVR"一词由 [Tülu 3](https://arxiv.org/abs/2411.15124) 流行开来；[R1](https://arxiv.org/abs/2501.12948) 证明只用这类规则奖励就能得到强推理模型。

可验证奖励的优点是几乎不可被"说服"；缺点是覆盖面窄，而且"验证器"也会有漏洞：答案抽取过宽会奖励猜测，单元测试不全会奖励只通过测试的错误实现。把它推广到数学之外有几条路：训练一个生成式答案验证器做软判定（[跨领域软验证](https://arxiv.org/abs/2503.23829)、[General-Reasoner](https://arxiv.org/abs/2505.14652)、[CompassVerifier](https://arxiv.org/abs/2508.03686)）；把多领域语料改写成可验证题目（[CrossThink](https://arxiv.org/abs/2504.13941)、[Webscale-RL](https://arxiv.org/abs/2510.06499)）；或者以模型生成参考答案的概率作奖励，完全免去验证器（[VeriFree](https://arxiv.org/abs/2505.21493)、[RLPR](https://arxiv.org/abs/2506.18254)）。

### 2.3 生成式评判与评分细则

对医学建议、研究计划、写作这类没有标准答案的任务，目前的主流是让另一个大模型按**评分细则**（rubric）打分：

$$r(x,y)=\frac{\sum_k w_k\,J(x,y,c_k)}{\sum_k w_k}$$

$c_k$ 是第 $k$ 条细则（"是否提到了药物相互作用""是否给出了可检验的预测"），$J$ 是评判模型的判断，$w_k$ 是权重（[RaR](https://arxiv.org/abs/2507.17746)、[检查表](https://arxiv.org/abs/2507.18624)、[Rubric Anchors](https://arxiv.org/abs/2508.12790)、[OpenRubrics](https://arxiv.org/abs/2510.07743)）。另一种形式是把验证本身变成下一 token 预测，取 $r=p_\phi(\text{Yes}\mid x,y)$，并让评判先写一段推理（[GenRM](https://arxiv.org/abs/2408.15240)）；奖励模型也可以先生成原则与批评再打分，从而随推理算力扩展（[DeepSeek-GRM](https://arxiv.org/abs/2504.02495)、[RM-R1](https://arxiv.org/abs/2505.02387)）。[POLAR](https://arxiv.org/abs/2507.05197) 则把奖励模型做成"判断两条轨迹是否来自同一策略"的可预训练模型。

### 2.4 过度优化

奖励模型只是人类偏好的代理。[Gao 等](https://arxiv.org/abs/2210.10760)用一个大奖励模型充当"金标"、小奖励模型充当"代理"，测量优化代理奖励时金标奖励的变化。以 $d=\sqrt{\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})}$ 衡量偏离：

$$R_{\text{BoN}}(d)=d\,(\alpha_{\text{BoN}}-\beta_{\text{BoN}}\,d),\qquad R_{\text{RL}}(d)=d\,(\alpha_{\text{RL}}-\beta_{\text{RL}}\log d)$$

金标奖励先升后降，系数随代理奖励模型的规模平滑变化。这是 Goodhart 定律的定量版本，也是 RLHF 中保留 KL 惩罚的理由。缓解手段包括奖励模型集成与保守聚合（[集成](https://arxiv.org/abs/2310.02743)）、权重平均（[WARM](https://arxiv.org/abs/2401.12187)）、提早停止，以及把稀疏但可靠的验证器与稠密但有偏的奖励模型结合（[HERO](https://arxiv.org/abs/2510.07242)）。

### 2.5 最近一年

- **证明可以验证。** 先训练证明验证器作为奖励，再让生成器学会自我验证，达到 IMO 2025 金牌水平（[DeepSeekMath-V2](https://arxiv.org/abs/2511.22570)）。形式化方向上，Z3 等价判定被蒸馏成连续的生成式奖励（[GenV](https://arxiv.org/abs/2609.11085)）。
- **奖励模型变成智能体。** 会调用裁剪、检索工具取证再打分的多模态奖励模型（[ARM-Thinker](https://arxiv.org/abs/2512.05111)）；为智能体轨迹写推理与批评的奖励模型（[Agent-RRM](https://arxiv.org/abs/2601.22154)）；先探索代码库、再为该问题写评分细则的 SWE 验证器（[智能体式评分细则](https://arxiv.org/abs/2601.04171)）；甚至不训练权重，只演化评判智能体的上下文与工具（[RewardHarness](https://arxiv.org/abs/2605.08703)），或者不执行代码、靠阅读判断补丁（[Dockerless](https://arxiv.org/abs/2606.28436)）。
- **细则与评判一起训练、随策略演化。** 交替地用 RL 优化细则生成器与评判（[Rubric-ARM](https://arxiv.org/abs/2602.01511)）；细则根据策略的回答修订（[MetaRubric](https://arxiv.org/abs/2610.02824)、[HarnessRL](https://arxiv.org/abs/2610.26426)）；评判理由兼顾广度与深度（[Mix-GRM](https://arxiv.org/abs/2603.01571)）。
- **分数的统计性质。** 许多奖励模型过于敏感，对同样好的回答给出不同分数，离散化后反而更利于 RL（[离散化奖励模型](https://arxiv.org/abs/2606.21795)）；评分被建模为分布 $p(r\mid x,y)$ 而不是一个数（[Z-Reward](https://arxiv.org/abs/2606.09076)、[扩散奖励模型](https://arxiv.org/abs/2609.33803)）。
- **验证并不总比生成容易。** 对编码智能体，生成复杂补丁已不难，可靠地验证反而更难；每个验证器都只是意图的代理，训练会扩大代理与意图的差距（[验证视野](https://arxiv.org/abs/2606.26300)）。
- **精确率。** 只按"覆盖了多少细则"给分会诱导堆砌内容；新的奖励要求每个回答单元既必要又正确（[FORK](https://arxiv.org/abs/2610.33512)）。世界模型也被用作 GUI 动作的验证器（[世界模型验证器](https://arxiv.org/abs/2610.11942)）。

## 3 策略优化算法：从 PPO 到 GRPO，再到之后

### 3.1 PPO 在 LLM 上

RLHF 时代的标准做法是 token 级 PPO：

$$\mathcal L^{\text{PPO}}=\mathbb E_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big],\qquad \rho_t=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{old}}(y_t\mid s_t)}$$

优势用 GAE 估计，价值网络通常与策略同样大、从奖励模型或 SFT 模型初始化；奖励是结尾的 $r(x,y)$ 加上逐 token 的 KL 惩罚。这需要同时维护策略、价值、奖励、参考四个模型，工程复杂，而且结果非常依赖实现细节：优势白化、价值裁剪、奖励归一化、对 EOS 的处理（[实现很重要](https://arxiv.org/abs/2005.12729)、[RLHF 的秘密](https://arxiv.org/abs/2307.04964)、[N 个实现细节](https://arxiv.org/abs/2403.17031)）。对推理任务，常取 $\gamma=\lambda=1$，此时 $\hat A_t=r-V(s_t)$，价值网络只需预测"从这里继续下去答对的概率"。

### 3.2 去掉 critic：REINFORCE、RLOO 与 GRPO

LLM 的一个特点让价值网络变得可有可无：同一个提示可以采样任意多次。[RLOO](https://arxiv.org/abs/2402.14740) 指出 RLHF 中 PPO 的许多部件不必要，用留一基线的 REINFORCE

$$\hat A_i=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j$$

就能更简单、更省地达到同样或更好的效果。[GRPO](https://arxiv.org/abs/2402.03300) 用组内标准化：

$$\hat A_{i}=\frac{r_i-\mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$

把这个优势赋给回答 $i$ 的每个 token，再套上 PPO 的裁剪目标，并把 KL 惩罚加在损失里。[R1](https://arxiv.org/abs/2501.12948) 之后它成为推理 RL 的默认。为什么 critic 可以去掉：在 $\gamma=\lambda=1$、结尾奖励的设定下，token 级价值 $V(s_t)$ 的主要作用是降低方差，而同题多样本的均值已经是 $V(s_0)$ 的无偏估计；对中间状态的价值，训练出的价值网络本来也不准（[VinePPO](https://arxiv.org/abs/2410.01679) 发现，PPO 的价值网络在比较推理中间步骤的好坏时几乎不比随机猜测好）。

但 GRPO 的两个归一化引入了偏差（[Dr. GRPO](https://arxiv.org/abs/2503.20783)）：

- **按标准差归一化**：通过率接近 0 或 1 的题标准差小，优势被放大，等于给太易与太难的题更大的权重。
- **按长度归一化**：原始 GRPO 先在每个回答内对 token 求平均（乘 $1/|y_i|$），再在回答间平均。对负优势的回答，越长的回答每个 token 受到的惩罚越小，于是错误回答倾向于变长。

### 3.3 DAPO 的四项修补

[DAPO](https://arxiv.org/abs/2503.14476) 在大规模推理 RL 中总结了四项修补，几乎成了默认配置：

1. **非对称裁剪（clip-higher）。** 比率裁剪对低概率 token 不公平：$\pi_{\text{old}}=0.01$ 的 token 一次最多升到 $0.01(1+\epsilon)$，而 $\pi_{\text{old}}=0.9$ 的 token 几乎可以升到 1。上界 $1+\epsilon$ 因此压制了"把小概率的好动作变大"，导致熵塌缩。把上界单独调大（如 $\epsilon_{\text{high}}=0.28$）可以缓解。
2. **动态采样。** 全对或全错的组优势为零、不产生梯度；过滤掉这些组并补采，保持有效批大小。
3. **token 级损失聚合。** 先把整批所有 token 加起来再除以总 token 数，长回答中的每个 token 与短回答中的同样重要，避免上面的长度偏差。
4. **超长奖励塑形。** 对接近长度上限的回答施加逐渐加大的惩罚，而不是一刀切记 0。

[Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290) 则说明原始 PPO + GAE（$\gamma=\lambda=1$、无 KL）也能稳定扩展；[VAPO](https://arxiv.org/abs/2504.05118) 在价值网络上做了一系列修补，在长思维链上超过无价值方法。[技巧还是陷阱](https://arxiv.org/abs/2508.08221)系统复现了这些技巧，发现很多只在特定设置下有效。

### 3.4 序列级比率

token 级 PPO 对每个 token 单独计算比率与裁剪，但奖励是序列级的。序列级的重要性比是 token 比率的乘积，方差极大；[GSPO](https://arxiv.org/abs/2507.18071) 用几何平均：

$$s_i(\theta)=\Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\Big)^{1/|y_i|}=\exp\Big(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}\mid s_{i,t})}{\pi_{\text{old}}(y_{i,t}\mid s_{i,t})}\Big)$$

并在序列层面裁剪。这对 MoE 模型尤其重要：每次更新后同一 token 被路由到的专家可能改变，token 级比率因此剧烈波动。[MiniMax-M1](https://arxiv.org/abs/2506.13585) 的 CISPO 走另一条路：截断重要性权重并停止它的梯度，保证每个 token 都有梯度，而不是像 PPO 那样把被裁剪的 token 的梯度整个置零；[GMPO](https://arxiv.org/abs/2507.20673) 用几何平均压制异常 token。

### 3.5 KL 的估计与放置

常用的三个单样本 KL 估计量（$a\sim\pi_\theta$，$q=\pi_{\text{ref}}(a)/\pi_\theta(a)$）：$k_1=-\log q$（无偏、可负），$k_2=\frac12(\log q)^2$（有偏、非负），$k_3=q-1-\log q$（无偏、非负、方差小）。GRPO 把 $k_3$ 直接加进损失。问题在于：对估计量求导并不等于对 KL 求导。在策略采样时，$\nabla_\theta k_3=(1-q)\nabla_\theta\log\pi_\theta(a)$，其期望为 $-\sum_a\pi_{\text{ref}}(a)\nabla_\theta\log\pi_\theta(a)$——这是**正向** KL $\mathrm{KL}(\pi_{\text{ref}}\,\|\,\pi_\theta)$ 的梯度，而目标里写的是反向 KL；把 $k_1$ 直接放进损失更糟，$\mathbb E[\nabla_\theta\log\pi_\theta]=0$，期望梯度为零。要得到反向 KL 的梯度，应把 $k_1$ 当作奖励（乘在 $\nabla\log\pi$ 上）而不是当作损失（[KL 估计的陷阱](https://arxiv.org/abs/2506.09477)）。半年后又有工作主张把 KL 锚定在采样策略上统一处理（[KLPO](https://arxiv.org/abs/2610.08963)）。对 RL 研究者而言，这是一个熟悉的教训：估计量对，梯度不一定对。

### 3.6 离策略：陈旧与训推不一致

即使算法是"在策略"的，工程上也总有两种离策略：

- **陈旧**：一批 rollout 做多次小批量更新，后面的更新所用的数据来自旧策略；异步训练中生成落后训练若干个版本。
- **训推不一致**：rollout 由推理引擎生成，梯度由训练框架计算；同一组权重在两处算出的 $\pi(y_t\mid s_t)$ 不同（内核、精度、批不变性、MoE 路由）。

后者的标准修正是截断重要性采样（[TIS](https://fengyao.notion.site/off-policy-rl)）：

$$\nabla J\approx\mathbb E_{y\sim\pi_{\text{infer}}}\Big[\sum_t\min\Big(\frac{\pi_{\text{train}}(y_t\mid s_t)}{\pi_{\text{infer}}(y_t\mid s_t)},\,C\Big)\nabla_\theta\log\pi_{\text{train}}(y_t\mid s_t)\,\hat A_t\Big]$$

[形式化与实践](https://arxiv.org/abs/2512.01374)给出了统一的解释：token 级替代目标是序列级目标的一阶近似，只有当训推差异与策略陈旧都小时近似才成立；重要性采样修正、裁剪、MoE 的路由回放（[R3](https://arxiv.org/abs/2510.11370)）之所以关键，都源于此。在严格在策略时，带重要性采样修正的基本策略梯度就是最稳定的。这也解释了为什么"把 BF16 换成 FP16"能显著稳定训练（[FP16](https://arxiv.org/abs/2510.26788)）。

### 3.7 最近一年

- **软裁剪与重新设计信任域。** 用温度控制的平滑门替代硬裁剪（[SAPO](https://arxiv.org/abs/2511.20347)）；大词表下比率裁剪对低概率 token 过严、对高概率 token 过松，于是改用直接估计的总变差（[DPPO](https://arxiv.org/abs/2602.04879)）、按概率设定边界（[BandPO](https://arxiv.org/abs/2603.04918)）、平滑的优势加权正则（[DRPO](https://arxiv.org/abs/2606.09821)）或黎曼几何下的等距更新（[RIPO](https://arxiv.org/abs/2607.10169)）。
- **组相对优势的偏差被量化。** 对难题低估、对易题高估（[优势有偏](https://arxiv.org/abs/2601.08521)、[难题优先](https://arxiv.org/abs/2601.20614)）；正确与错误轨迹的权重严格对称、未被采样的 token 得不到更新（[隐式对称](https://arxiv.org/abs/2602.05548)）。
- **更新目标与拟合分开。** 先构造目标策略再做监督拟合（[TPO](https://arxiv.org/abs/2604.06159)）；把长推理当作单步决策的序列级 PPO（[SPPO](https://arxiv.org/abs/2604.08865)）。
- **critic 回归。** PPO 的不稳定很大程度来自 critic 本身（[EasyPPO](https://arxiv.org/abs/2609.36802)），价值网络会"扁平化"（[价值扁平化](https://arxiv.org/abs/2609.18708)）；按难度选择性训练 critic 后 actor-critic 可稳定扩展（[S3AC](https://arxiv.org/abs/2610.35518)、[TPPO](https://arxiv.org/abs/2610.16794)）。另一边，从策略镜像下降导出的免 critic 方法与原目标有相同的唯一最优解（[BPO](https://arxiv.org/abs/2609.15987)）。
- **多奖励。** 真实后训练同时优化正确性、格式、长度、安全与风格。按奖励分别做组归一化（[GDPO](https://arxiv.org/abs/2601.05242)）、按方差动态加权（[DVAO](https://arxiv.org/abs/2605.25604)）、降低已饱和目标的权重（[饱和感知](https://arxiv.org/abs/2608.16072)）、按有效组密度补偿稀疏奖励（[密度感知](https://arxiv.org/abs/2610.00574)）、把每个奖励只分配给它检查过的 token（[MRTPO](https://arxiv.org/abs/2610.21439)）。
- **异步下的离策略。** 只对高失配的尾部收紧信任域（[SAT](https://arxiv.org/abs/2607.18722)）、从变分推断导出的序列级软约束（[VESPO](https://arxiv.org/abs/2602.10693)）、约束重要性权重的二阶矩（[M2PO](https://arxiv.org/abs/2510.01161)）。

## 4 绕开在线 RL：直接偏好优化、拒绝采样与蒸馏

### 4.1 DPO：把最优解代回去

由 0.2 节的最优解反解奖励：

$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$$

代入 Bradley–Terry，$Z(x)$ 在差中相消，于是可以直接用偏好数据拟合策略（[DPO](https://arxiv.org/abs/2305.18290)）：

$$\mathcal L_{\text{DPO}}(\theta)=-\mathbb E\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]$$

记隐式奖励 $\hat r_\theta=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，梯度为 $-\beta\,\mathbb E\big[\sigma(\hat r_\theta(y_l)-\hat r_\theta(y_w))\,(\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l))\big]$：隐式奖励排错的样本对权重更大。它不需要奖励模型、不需要采样，训练像分类一样稳定。在 token MDP 中，DPO 隐式学到的是最优软 Q 函数（[从 r 到 Q*](https://arxiv.org/abs/2404.12358)）。

DPO 的局限来自离线：数据由别的策略生成，策略在训练中走到数据覆盖不到的地方时没有信号；优化只关心 $y_w$ 与 $y_l$ 的相对似然，两者可以一起下降（似然位移）。后续变体修改的是损失形式：[IPO](https://arxiv.org/abs/2310.12036) 处理确定性偏好下的过拟合，[KTO](https://arxiv.org/abs/2402.01306) 只需"好/坏"标签，[SimPO](https://arxiv.org/abs/2405.14734) 去掉参考模型并按长度归一化，[ORPO](https://arxiv.org/abs/2403.07691) 把偏好项并入 SFT。系统比较发现它们差别不大，主要区别在排序式与点式目标（[直接对齐算法差别不大](https://arxiv.org/abs/2502.01237)）。

### 4.2 拒绝采样与专家迭代

另一条更朴素的路线是：采样多个回答，按奖励挑出好的，在上面做 SFT，迭代进行（[RAFT](https://arxiv.org/abs/2304.06767)、[ReST](https://arxiv.org/abs/2308.08998)、[STaR](https://arxiv.org/abs/2203.14465)、[ReST-EM](https://arxiv.org/abs/2312.06585)）。对二值奖励，它的目标是

$$\max_\theta\ \mathbb E_{y\sim\pi_{\text{old}}}\big[\mathbb 1[r(x,y)=1]\,\log\pi_\theta(y\mid x)\big]$$

也就是只有正样本、且样本来自旧策略的 REINFORCE；整个过程可以看作 EM，或者 [专家迭代](https://arxiv.org/abs/1705.08439)（"慢思考"产生数据、"快思考"模仿）。[Llama 3](https://arxiv.org/abs/2407.21783) 的后训练就是拒绝采样 + DPO 的多轮迭代。

### 4.3 在策略与负梯度为什么重要

[Tajwar 等](https://arxiv.org/abs/2404.14367)把偏好微调的方法按两个维度分类：数据是否在策略采样、损失是否有"负梯度"（主动压低差回答的概率）。实验结论是两者都重要：负梯度能更快地把概率质量从差的模式上移走，在策略采样让信号落在策略当前会走到的地方。对照研究（[DPO 与 PPO](https://arxiv.org/abs/2404.10719)、[拆解 DPO 与 PPO](https://arxiv.org/abs/2406.09279)）也发现，充分调优的在线 PPO 在对话与代码上仍优于离线 DPO。这推动了 DPO 的在线化与迭代化（[OAIF](https://arxiv.org/abs/2402.04792)、[迭代偏好学习](https://arxiv.org/abs/2312.11456)、[RLHF Workflow](https://arxiv.org/abs/2405.07863)），以及带探索项的版本（[XPO](https://arxiv.org/abs/2405.21046)）。

另一种视角是博弈：若偏好不可传递，标量奖励不存在，应直接求偏好下的纳什均衡策略（[纳什学习](https://arxiv.org/abs/2312.00886)、[DNO](https://arxiv.org/abs/2404.03715)、[SPPO](https://arxiv.org/abs/2405.00675)）；[SPIN](https://arxiv.org/abs/2401.01335) 把人类示范当正例、模型自己的生成当负例做自我博弈。

SFT 与 RL 之间也有中间形态：在 RLVR 中混入强模型的离策略轨迹（[LUFFY](https://arxiv.org/abs/2504.14945)）；同一阶段混合示范的监督损失与自身 rollout 的 RL 损失（[SRFT](https://arxiv.org/abs/2506.19767)）；把 SFT 看作奖励稀疏且病态的策略梯度、按 token 概率重新缩放（[DFT](https://arxiv.org/abs/2508.05629)）；把专家解拆成逐步"动作"、按每步与专家的相似度给稠密奖励（[SRL](https://arxiv.org/abs/2510.25992)）。

### 4.4 在策略蒸馏：稠密奖励的 RL

蒸馏传统上是离线的：教师生成回答，学生在上面做 SFT。问题与行为克隆相同——学生一旦走到教师没走过的状态就会累积错误。在策略蒸馏（OPD）让学生自己采样，教师在学生的每个前缀上给出它的分布，学生最小化逐 token 的反向 KL：

$$\min_\theta\ \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\Big[\sum_t\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\|\,\pi_T(\cdot\mid s_t)\big)\Big]$$

用采样到的 token 做单样本估计，相当于逐 token 的奖励 $r_t=\log\pi_T(y_t\mid s_t)-\log\pi_\theta(y_t\mid s_t)$，再用策略梯度优化：这是**奖励稠密的 RL**，也是 DAgger 在序列上的版本——在学生自己会到达的状态上请教专家（[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)）。它兼得 RL 的在策略与蒸馏的稠密信号，成本远低于 RL；反向 KL 是"寻模"的，学生会集中在教师的一个模式上，而不是覆盖教师的全部分布。实践中有采样 token 估计与全词表 KL 两种做法，前者便宜、后者方差小。

### 4.5 自蒸馏与特权信息

如果没有更强的教师，可以让模型自己当教师——只要教师看到学生看不到的东西。[SDPO](https://arxiv.org/abs/2601.20802) 把环境的文字反馈（报错、测试结果）放进上下文，以 $\pi_T(\cdot\mid s_t)=\pi_\theta(\cdot\mid s_t,c)$ 为教师；即使环境只返回标量奖励，也可以把同组中成功的 rollout 当作失败尝试的"反馈"。[SDFT](https://arxiv.org/abs/2601.19897) 以"看过示范的自己"为教师，从示范学新技能而遗忘远少于 SFT。这把"答错了"的一个比特变成了"该怎么改"的逐 token 信号；代价是教师与学生共享同一组参数，特权信息可能让教师"作弊"（直接抄答案而不是给出可学习的推理）。

### 4.6 最近一年

- **黑盒与离线。** 只有教师文本时，训练判别器区分师生回答、把判别器当在策略奖励（[GAD](https://arxiv.org/abs/2511.10643)）；预先采样学生 rollout、离线计算教师 logits，前提是"教师一致性"（[Lightning OPD](https://arxiv.org/abs/2604.13010)）。
- **失败模式。** 采样 token 的 OPD 有 token 监督失衡、教师在学生前缀上不可靠、分词器或特殊 token 不匹配三类失败（[再看 OPD](https://arxiv.org/abs/2603.25562)）；OPD 让学生系统性地过度自信（[确定性幻觉](https://arxiv.org/abs/2604.16830)）；OPD 也会利用长度等漏洞（[揭开 OPD](https://arxiv.org/abs/2607.13399)）。
- **变体。** 信任域（[TrOPD](https://arxiv.org/abs/2606.01249)、[TRB](https://arxiv.org/abs/2605.31159)）、以评分细则代替 logits（[ROPD](https://arxiv.org/abs/2605.07396)）、弱教师（[弱到强 OPD](https://arxiv.org/abs/2607.26246)）、多教师整合各领域的 RL 专家（[MOPD](https://arxiv.org/abs/2606.30406)）、异步（OPD 比 RL 更能容忍陈旧，[AsyncOPD](https://arxiv.org/abs/2606.24143)）、自修订把二值奖励变成稠密监督（[SD-Zero](https://arxiv.org/abs/2604.12002)）、把 SFT 调整为服务于后续 RL（[PEAR](https://arxiv.org/abs/2602.01058)）。
- **更新几何。** OPD 的参数更新小且坐标稀疏，只训练找到的子网络就几乎能复现全量训练，与 RL 的更新几何相似（[稠密监督、稀疏更新](https://arxiv.org/abs/2606.13657)）。
- **反思。** 只用初始学生的离线 rollout 蒸馏（Semi-OPD），在 1.5B 到 235B 的 17 组师生中有 14 组优于 OPD，最高快 11.4 倍；OPD 只在师生输出的 token 重合度高时才有利（[Semi-OPD](https://arxiv.org/abs/2610.11291)）。多教师 OPD 的学生常常不如只跟最好的单一专家（[领域归一化 MOPD](https://arxiv.org/abs/2609.35347)）。无监督的自蒸馏（[U-OPSD](https://arxiv.org/abs/2608.06296)）、特权信息到底贡献了什么（[OPSD 中的特权信息](https://arxiv.org/abs/2609.20612)）、从 RL 视角诊断崩溃（[收益与崩溃](https://arxiv.org/abs/2610.03185)）与批判性回顾（[三个杠杆](https://arxiv.org/abs/2608.25936)）相继出现。在智能体 RL 中，统一条件的复现发现多数自蒸馏方法的增益不如论文所报（[现实检验](https://arxiv.org/abs/2610.19789)）。

## 5 探索、熵与能力边界：RL 在发现还是在锐化

### 5.1 pass@k 与覆盖

对一道题采样 $n$ 次、其中 $c$ 次正确，"采 $k$ 次至少对一次"的概率的无偏估计是（[Codex](https://arxiv.org/abs/2107.03374)）

$$\widehat{\text{pass@}k}=1-\binom{n-c}{k}\Big/\binom{n}{k}$$

pass@1 衡量可靠性，大 $k$ 衡量覆盖：模型的采样分布里"有没有"正确答案。重复采样的覆盖率随样本数近似对数线性增长（[Large Language Monkeys](https://arxiv.org/abs/2407.21787)），这本身就是一条测试时的扩展轴。RL 研究者应当注意：RLVR 优化的是 pass@1 的期望，提升 pass@1 与保持 pass@k 是两件事。

### 5.2 锐化还是发现：证据

**支持"锐化"的证据。** [Yue 等](https://arxiv.org/abs/2504.13837)在多个模型与基准上发现，小 $k$ 时 RL 模型胜出，大 $k$ 时基座模型反超：RL 主要提高了采样效率，把已有的正确解挑了出来，而 RL 模型能解的题几乎都在基座的覆盖之内。随机甚至错误的奖励也能显著提升某些模型的数学成绩（[虚假奖励](https://arxiv.org/abs/2506.10947)），后来发现部分原因是基准数据污染（[推理还是记忆](https://arxiv.org/abs/2507.10532)）。只用一个训练样本做 RLVR 就能大幅提升（[单样本 RLVR](https://arxiv.org/abs/2504.20571)）。最直接的是：不训练，只对基座做 $\pi^\alpha$ 的 MCMC 采样，就接近 RL 后的推理表现（[幂分布采样](https://arxiv.org/abs/2510.14901)）。

理论上，二值奖励下的 KL 正则最优解为

$$\pi^*(y\mid x)\propto\pi_{\text{ref}}(y\mid x)\,\exp\big(\mathbb 1[y\in\mathcal Y^\star]/\beta\big)$$

它只在 $\pi_{\text{ref}}$ 的支撑内重新加权；$\pi_{\text{ref}}(y)=0$ 的解永远不会被采到（[隐形的缰绳](https://arxiv.org/abs/2507.14843)）。现实中"支撑"是"在采样预算内能采到的"，比数学上的支撑小得多，所以这条论证比看上去更有约束力。

**支持"发现"的证据。** 足够长、足够稳的 RL（KL 控制、参考模型重置、多样化任务）能解出基座在大 $k$ 下也解不出的题（[ProRL](https://arxiv.org/abs/2505.24864)）；用 CoT-pass@k（要求推理过程也正确）衡量，RL 确实提高了推理过程的正确率（[CoT-pass@k](https://arxiv.org/abs/2506.14245)）；受控的合成实验中，RL 在模型能力边缘带来向更复杂组合的外推（[预训练、中期训练与 RL](https://arxiv.org/abs/2512.07783)）。

**起点决定一切。** 起点模型是否具备验证、回溯、设定子目标、逆向推理四种认知行为，决定了 RL 能否奏效；对不具备的模型，先用含这些行为的数据做少量预训练即可（[认知行为](https://arxiv.org/abs/2503.01307)）。在 10 个基座上复现零 RL，"顿悟"并非普遍出现（[SimpleRL-Zoo](https://arxiv.org/abs/2503.18892)）；中期训练决定了基座能否被 RL 提升（[OctoThinker](https://arxiv.org/abs/2506.20512)）；800 条精选样本的 SFT 就能激发复杂推理（[LIMO](https://arxiv.org/abs/2502.03387)）。

### 5.3 熵的动力学

对 softmax 策略做一步自然策略梯度，状态 $s$ 上的熵变化近似为（[熵机制](https://arxiv.org/abs/2505.22617)）

$$\Delta H(s)\approx-\eta\,\mathrm{Cov}_{a\sim\pi(\cdot\mid s)}\big(\log\pi(a\mid s),\,A(s,a)\big)$$

高概率且高优势的动作让熵下降，低概率且高优势的动作让熵上升。RLVR 中正确回答往往由高概率 token 组成，于是协方差为正、熵持续下降——这就是普遍观察到的熵塌缩，伴随而来的是性能饱和。该工作据此提出只对协方差最大的少数 token 裁剪或加 KL（Clip-Cov、KL-Cov）。熵的分布高度不均：约 20% 的高熵"分叉 token"决定了 RLVR 的效果，只对它们计算梯度反而更好（[80/20](https://arxiv.org/abs/2506.01939)）。

另一个 LLM 特有的限制是**零方差组**。通过率为 $p$ 的题，$G$ 个样本全对或全错的概率是 $p^G+(1-p)^G$，此时组相对优势为零，梯度消失；有效信号近似正比于 $p(1-p)$。难题上采不到正样本，学习就停滞——这是探索问题在 LLM 上最直接的表现。

### 5.4 探索手段

按作用位置分：

- **目标层**：直接优化 pass@k（[pass@k 训练](https://arxiv.org/abs/2508.10751)、[PKPO](https://arxiv.org/abs/2505.15201)、[max@k](https://arxiv.org/abs/2510.23393)）；奖励多样性（[DARLING](https://arxiv.org/abs/2509.02534)）或低概率的正确解（[奖励不太可能的解](https://arxiv.org/abs/2506.02355)）；匹配奖励分布而非最大化奖励（[FlowRL](https://arxiv.org/abs/2509.15207)）。
- **更新层**：放宽上界裁剪（[DAPO](https://arxiv.org/abs/2503.14476)）、保护低概率 token（[低概率 token](https://arxiv.org/abs/2510.03222)）、在优势上加熵项（[熵视角](https://arxiv.org/abs/2506.14758)）；只用负样本训练也能保持多样性（[负样本强化](https://arxiv.org/abs/2506.01347)）。
- **采样层**：从高熵决策点分支（[FR3E](https://arxiv.org/abs/2507.07017)，Go-Explore 的思路）、好奇心奖励（[CDE](https://arxiv.org/abs/2509.09675)、[Pixel Reasoner](https://arxiv.org/abs/2505.15966)）。
- **数据层**：动态采样、课程与提示（第 7 章）。

### 5.5 最近一年

- **虚假奖励与熵最小化为何都有效。** 从裁剪偏差、熵与奖励的关系解释这两种看似矛盾的机制（[重新理解 RLVR](https://arxiv.org/abs/2512.16912)）；虚假奖励下的收益来自激活记忆捷径——答案 token 的困惑度下降、提示一侧的连贯性变差（[虚假奖励悖论](https://arxiv.org/abs/2601.11061)）。
- **无监督 RL 的上限。** 各类内在奖励（置信度、熵、多数投票）都收敛到"锐化初始分布"：起点置信度与正确性一致时有效，否则灾难性失败；训练曲线先升后降，崩溃时间由模型先验决定而非工程选择（[无监督 RLVR 能扩展多远](https://arxiv.org/abs/2603.08660)）。
- **覆盖为何收缩。** SFT 后 pass@k 下降与数据中的"岔路口"——多条合理路径都可走的决策点——有关（[岔路口](https://arxiv.org/abs/2605.17026)）；多样性坍缩发生在后训练的哪一阶段被逐段定位（[多样性在哪里坍缩](https://arxiv.org/abs/2604.16027)）。
- **给难题搭台阶。** 训练时注入特权提示改变 rollout 分布（[SAGE](https://arxiv.org/abs/2602.03143)）、只给最小充分的知识点（[KnowRL](https://arxiv.org/abs/2604.12627)）、用参考解合成模型自己风格的正例（[ReGFT](https://arxiv.org/abs/2603.01223)）、教师只出现在提示里而不进梯度（[ZPPO](https://arxiv.org/abs/2606.18216)）。
- **从提示与策略层面扩大探索。** 在提示里插入无意义的填充文本（[LoPE](https://arxiv.org/abs/2605.05566)）、每个 rollout 前加一句解题策略（[NudgeRL](https://arxiv.org/abs/2605.15726)）、组内混入小模型的 rollout（[小模型是天然的探索者](https://arxiv.org/abs/2605.30789)）、组熵控制（[GEPO](https://arxiv.org/abs/2607.16850)）、按给定曲线精确控制熵（[Entrocraft](https://arxiv.org/abs/2604.26326)）、修正正确回答之间置信度差异对探索的压制（[回到基础](https://arxiv.org/abs/2602.05281)）、双尺度多样性正则（[DSDR](https://arxiv.org/abs/2602.19895)）。
- **学到自己采不到的东西。** 全错的组换成同伴模型的轨迹并做离策略修正（[GRAFT](https://arxiv.org/abs/2609.37868)）、在 rollout 中做稀疏的局部干预（[MInTRL](https://arxiv.org/abs/2609.12419)）、对未解出的难题持续追加采样（[NGU](https://arxiv.org/abs/2609.13443)）、在输入嵌入上加锚定噪声（[EDEN](https://arxiv.org/abs/2610.16105)）。
- **RL 学到新东西的例子。** 只用表层翻译指标 chrF 作奖励，RL 就能让模型利用上下文中的语法与词典翻译从未见过的语言（[未见语言翻译](https://arxiv.org/abs/2606.06428)）；可控难度的合成任务表明，RL 能否学会长程推理取决于模型的表达能力（[能学会多深的推理](https://arxiv.org/abs/2605.06638)）。

## 6 信用分配、过程奖励与推理

### 6.1 结果奖励下的信用

在 GRPO 中，回答的优势被赋给它的每个 token。这在期望意义上是无偏的（REINFORCE），但方差巨大：一个上万 token 的正确回答里可能有错误的中间步骤，一个错误回答里可能有大段正确的推理。经典 RL 用价值函数与 TD 把信用推回到中间状态；LLM 上有三类替代。

### 6.2 过程奖励模型

过程奖励模型（PRM）给推理的每一步打分。早期结论是过程监督优于结果监督：[Uesato 等](https://arxiv.org/abs/2211.14275)发现两者最终准确率相近、但过程监督的推理错误更少；[Let's Verify](https://arxiv.org/abs/2305.20050)在 MATH 上用 80 万条人工步骤标签训练的 PRM 做重排，明显优于结果奖励模型。人工标签昂贵，于是出现了自动标注：从中间步骤续写多次，用到达正确答案的比例作为该步的标签（[Math-Shepherd](https://arxiv.org/abs/2312.08935)），或用二分查找定位第一个错误步（[OmegaPRM](https://arxiv.org/abs/2406.06592)）。不过蒙特卡洛标签噪声大，用 LLM 评判加共识过滤更可靠（[PRM 的经验教训](https://arxiv.org/abs/2501.07301)）。

一个关键的概念澄清来自 [Setlur 等](https://arxiv.org/abs/2410.08146)：步骤奖励应当衡量"进步"，即在某个证明者策略 $\mu$ 下这一步带来的成功概率变化

$$r^{\text{proc}}_h=Q^\mu(s_h,a_h)-V^\mu(s_h)$$

——这就是优势函数。一个"正确但没有推进"的步骤不应得到正奖励；而且 $\mu$ 不必是当前策略，一个能力互补的证明者给出的优势对探索更有帮助。

把 PRM 直接当 RL 奖励的主要风险是投机：策略学会生成 PRM 喜欢、但不推进解题的步骤。把过程奖励按"取最小"而非求和累积可以缓解（[最小式信用分配](https://arxiv.org/abs/2504.15275)）。[隐式 PRM](https://arxiv.org/abs/2412.01981) 提供了另一种做法：把奖励模型参数化为 $r_\phi(x,y)=\beta\log\frac{\pi_\phi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$，只用结果标签训练，就自动得到 token 级奖励 $r_t=\beta\log\frac{\pi_\phi(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)}$；[PRIME](https://arxiv.org/abs/2502.01456) 在 RL 中在线更新它。生成式 PRM 先写验证 CoT 再判断，只需极少过程标签（[ThinkPRM](https://arxiv.org/abs/2504.16828)）。

### 6.3 蒙特卡洛与树

单轮任务可以从任意前缀重启，于是中间状态的价值可以直接估计：

$$\hat V(s_t)=\frac1K\sum_{k=1}^{K}r\big(x,\,y_{<t}\oplus\tilde y^{(k)}\big),\qquad \tilde y^{(k)}\sim\pi(\cdot\mid s_t)$$

[VinePPO](https://arxiv.org/abs/2410.01679) 用它替代价值网络：PPO 的价值网络在比较中间步骤的好坏时几乎不比随机猜测好，而蒙特卡洛估值显著改善了信用分配。代价是额外的生成。树形采样把这一代价摊薄：在共享前缀上分支，同一棵树里的样本自然给出分段的优势（[TreePO](https://arxiv.org/abs/2508.17445)）；[rStar-Math](https://arxiv.org/abs/2501.04519) 用 MCTS 与过程偏好模型让小模型自我进化。

### 6.4 测试时计算与长度

推理模型的"思考"是测试时计算：多数投票、Best-of-N、长思维链、并行思考都在用更多生成换正确率。按题目难度分配测试时计算，效率可胜过把模型放大 14 倍（[Snell 等](https://arxiv.org/abs/2408.03314)）；只用 1000 条样本 SFT 加上"预算强制"（不许结束、追加"Wait"）就能实现测试时扩展（[s1](https://arxiv.org/abs/2501.19393)）。从 RL 的角度看，长思维链是一个学会的"内部搜索"策略；[MRT](https://arxiv.org/abs/2503.07572) 把测试时计算看作元 RL，按每段思考带来的进步给稠密奖励；[e3](https://arxiv.org/abs/2506.09026) 说明要让模型在更长的预算下继续变好，必须在训练中学会探索（非对称能力、负梯度与课程）。不带验证信号、只模仿搜索轨迹的方法在扩展测试时计算时是次优的（[Setlur 等](https://arxiv.org/abs/2502.12118)）。

另一面是**过度思考**：o1 类模型在简单题上也写很长（[过度思考](https://arxiv.org/abs/2412.21187)）。长度控制有几种形式：奖励中减去长度惩罚 $r=\mathbb 1[\text{correct}]-\lambda|y|$（[高效推理](https://arxiv.org/abs/2502.04463)）、在提示中给定目标长度并按偏差惩罚（[L1](https://arxiv.org/abs/2503.04697)）、让模型决定是否思考（[AdaptThink](https://arxiv.org/abs/2505.13417)）、把思考切成固定长度的块、只携带短状态（[Markovian Thinker](https://arxiv.org/abs/2510.06557)）。一个有意思的发现是：PPO 的损失聚合在负奖励下偏好长回答，第二阶段做少量 RL 就能大幅缩短推理（[简洁推理](https://arxiv.org/abs/2504.05185)）；最朴素的截断长度惩罚之所以效果差，是因为优势偏差、熵塌缩与稀疏信号，修正后推理长度可减七成以上而准确率更高（[DLER](https://arxiv.org/abs/2510.15110)）。

### 6.5 最近一年

- **步骤标签不再需要人。** 对每步做多次独立验证再聚合（[ScalePRM](https://arxiv.org/abs/2512.03244)）、先把长 CoT 摘要成关键步骤再验证（[OPV](https://arxiv.org/abs/2512.10756)）、完全无监督的 PRM（[uPRM](https://arxiv.org/abs/2605.10158)）、预测成功概率及其可靠度的分布式 PRM（[BetaPRM](https://arxiv.org/abs/2605.15529)）、只用结果 RL 训练的生成式 PRM（[TIPS](https://arxiv.org/abs/2609.36641)），以及"RL 后训练本身已足以给步骤打分"的观察（[进度优势](https://arxiv.org/abs/2606.26080)）。
- **KL 里本来就有过程信号。** 自回归 KL 正则诱导了一项过程级的信用，常见实现把它漏掉了（[Future-KL](https://arxiv.org/abs/2601.10201)）；按 token 对未来 KL 的影响加权（[FIPO](https://arxiv.org/abs/2603.19835)）。
- **用短程训练得到长程推理。** 每段推理后压缩成摘要再继续，用 16k 训练长度得到可外推的模型（[Reasoning Cache](https://arxiv.org/abs/2602.03773)）；长度激励训练"上下文内探索"（[想得更长才能探索得更深](https://arxiv.org/abs/2602.11748)）。
- **价值模型被预训练。** 逐 token 预测剩余长度的价值模型（[LenVM](https://arxiv.org/abs/2604.27039)）；在上下文中编码策略能力、作为稀疏 rollout 先验的通用价值模型（[V0.5](https://arxiv.org/abs/2603.10848)）。
- **成功与失败不对称。** 意外的成功与反复的失败按熵区别对待（[EAPO](https://arxiv.org/abs/2609.33781)）；用现成模型找出第一个错误步塑形奖励（[Cliff](https://arxiv.org/abs/2609.02817)）；每个目标只对相应文本块施加自己的优势（[分块优势](https://arxiv.org/abs/2602.10231)）。
- **推理形式本身可学。** 用两个边界 token 在显式与潜在推理之间切换，并用在策略 RL 训练（[SWITCH](https://arxiv.org/abs/2606.13106)）；并行协同推理（[PaCoRe](https://arxiv.org/abs/2601.05593)）与原生并行推理（[自蒸馏 RL](https://arxiv.org/abs/2512.07461)）；CPU 上训练的小 RL 控制器决定何时停止采样（[RL 控制采样](https://arxiv.org/abs/2606.03102)）。

## 7 数据、课程、环境与自我博弈

### 7.1 可学习性与课程

对二值奖励、通过率为 $p$ 的题，单个样本奖励的方差是 $p(1-p)$；GRPO 一组样本提供的有效梯度近似正比于它。太难（$p\to0$）与太易（$p\to1$）的题都几乎不提供信号，只消耗 rollout。由此产生了一系列做法：动态过滤全对与全错的组（[DAPO](https://arxiv.org/abs/2503.14476)）；按组内奖励方差挑选"难度合适"的题（[VCRL](https://arxiv.org/abs/2509.19803)）；先在大量多样题上扩熵、再在少量难题上用大 rollout 预算集中训练（[DRIVE](https://arxiv.org/abs/2511.06307)）；找出模型持续失败的概念并围绕它们合成新题（[SwS](https://arxiv.org/abs/2506.08989)）；从自己的正确解合成新问题以维持 pass@k（[变分问题合成](https://arxiv.org/abs/2508.14029)）。RLVR 训练中还会自然出现由易到难的隐式课程（[隐式课程](https://arxiv.org/abs/2602.14872)）。

### 7.2 可验证环境

LLM 的 RL 没有 Atari 或 MuJoCo，"环境"要自己造：

- **程序生成的推理任务**：难度可调、答案可自动验证（[Reasoning Gym](https://arxiv.org/abs/2505.24760)、[SynLogic](https://arxiv.org/abs/2505.19641)、[Logic-RL](https://arxiv.org/abs/2502.14768)）。
- **多领域可验证数据**：[Guru](https://arxiv.org/abs/2506.14965) 的 6 个领域 9.2 万条数据显示，数学、代码这类预训练中常见的领域靠 RL 激发即可，逻辑、表格等陌生领域需要真正的领域内训练。
- **软件工程环境**：从真实仓库构造可执行的任务与测试（[SWE-Gym](https://arxiv.org/abs/2412.21139)、[R2E-Gym](https://arxiv.org/abs/2504.07164)、[SWE-smith](https://arxiv.org/abs/2504.21798)）。
- **智能体环境**：需要可重置、可并行、状态转移可靠，并能判定任务完成；用代码与数据库驱动比让 LLM 模拟更可靠（[Agent World Model](https://arxiv.org/abs/2602.10090)）。

验证器有漏洞时，策略一定会找到它（第 12 章）。环境的质量控制——测试是否充分、答案是否唯一、任务是否可解——是构建者最大的工作量。

### 7.3 自我博弈与无标签 RL

为了摆脱人工数据，可以让模型同时出题与解题。[Absolute Zero](https://arxiv.org/abs/2505.03335) 让同一个模型提出代码推理任务、用代码执行器验证，出题者的奖励是"可学习性"：

$$r_{\text{prop}}(q)=\begin{cases}1-\bar r_{\text{solve}}(q), & 0<\bar r_{\text{solve}}(q)<1\\ 0, & \text{otherwise}\end{cases}$$

——太难（求解者从未答对）或太易（总是答对）的题都得 0。[R-Zero](https://arxiv.org/abs/2508.05004) 用挑战者与求解者两个角色，[SPIRAL](https://arxiv.org/abs/2506.24119) 在零和博弈中自我对弈，[SPICE](https://arxiv.org/abs/2510.24684) 以大语料中的真实文档为锚防止自我博弈漂移，[搜索自我博弈](https://arxiv.org/abs/2510.18821)把它用于深度搜索智能体，[CURE](https://arxiv.org/abs/2506.03136) 让代码与单元测试互相提供奖励。

更进一步是完全不要标签：用多数投票作伪标签（[TTRL](https://arxiv.org/abs/2504.16084)）、以模型的自确定性为奖励（[Intuitor](https://arxiv.org/abs/2505.19590)）、直接最小化输出熵（[熵最小化](https://arxiv.org/abs/2505.15134)）。第 5 章已经说明，这些方法本质上都是锐化，只在起点的置信度与正确性一致时有效（[无监督 RLVR](https://arxiv.org/abs/2603.08660)）。

### 7.4 最近一年

- **自适应环境。** 400 个程序生成题目的环境，难度随策略能力自适应调整，联合训练优于静态数据（[RLVE](https://arxiv.org/abs/2511.07317)）；环境、策略与奖励模型一起演化（[RLAnything](https://arxiv.org/abs/2602.02488)）。
- **环境自动构建与规模化。** 终端任务程序化生成（[Endless Terminals](https://arxiv.org/abs/2601.16443)）；1000 个代码与数据库驱动的合成环境（[Agent World Model](https://arxiv.org/abs/2602.10090)）；从 GitHub PR 自动构建百万级可验证 SWE 环境（[SWE-Universe](https://arxiv.org/abs/2602.02361)）；软件世界（[OpenComputer](https://arxiv.org/abs/2605.19769)）、手机（[MobileGym](https://arxiv.org/abs/2605.26114)）、终端（[CLI-Universe](https://arxiv.org/abs/2606.22883)）；智能体观察一个运行中的参考实现、自己复现其行为（[RecreationWorld](https://arxiv.org/abs/2609.22000)）。
- **让饱和的题重新有用。** 把通过率为 1 的题组合成更难的复合题（[Composition-RL](https://arxiv.org/abs/2602.12036)）；按能力动态分配每题的 rollout 数（[CoBA-RL](https://arxiv.org/abs/2602.03048)）；对仍未解出的难题持续追加采样（[NGU](https://arxiv.org/abs/2609.13443)）。
- **零数据自我博弈走向开放任务。** 无验证器的协同进化（[G-Zero](https://arxiv.org/abs/2605.09959)）、基于文档出题（[SCOPE](https://arxiv.org/abs/2605.31433)）、以技能为单元（[Skill Self-Play](https://arxiv.org/abs/2607.22529)）、多模型协作中涌现推理（[Co-RL](https://arxiv.org/abs/2608.17253)）；同一模型既写长程可执行环境又求解（[SPADE](https://arxiv.org/abs/2608.19197)）。
- **数据技巧的统计检验。** 同一 GRPO 配方、12 个配对种子、12 个基准：均匀 GRPO 比未训练模型提高 7.76 个百分点，但 8 种 rollout 选择与重加权方法没有一种在 95% 置信下优于均匀采样，3 种自适应领域混合也不优于固定等比例（[DataFlex-RL](https://arxiv.org/abs/2609.06107)）。这提醒我们：很多论文的增益在种子方差之内。另一方面，SFT 数据中推理路径的多样性确实影响 RL 后的泛化（[多样的 SFT 轨迹](https://arxiv.org/abs/2609.33780)）。

## 8 多模态：视觉推理、视觉生成与 VLA

### 8.1 视觉语言模型的 RL

多模态 RL 先后经历了与文本相同的两个阶段。**RLHF 阶段**主要对付幻觉：[LLaVA-RLHF](https://arxiv.org/abs/2309.14525) 用事实增强的奖励模型缓解奖励投机，[RLHF-V](https://arxiv.org/abs/2312.00849) 用细粒度的人工纠正做稠密 DPO，[RLAIF-V](https://arxiv.org/abs/2405.17220) 用开源模型的 AI 反馈，[MM-RLHF](https://arxiv.org/abs/2502.10391) 扩大了偏好数据。**RLVR 阶段**把 R1 的配方搬过来：[Visual-RFT](https://arxiv.org/abs/2503.01785)、[VLM-R1](https://arxiv.org/abs/2504.07615)、[Vision-R1](https://arxiv.org/abs/2503.06749)、[MM-Eureka](https://arxiv.org/abs/2503.07365)、[LMM-R1](https://arxiv.org/abs/2503.07536)、[VL-Rethinker](https://arxiv.org/abs/2504.08837)；视频上有 [Video-R1](https://arxiv.org/abs/2503.21776) 与长视频的 [LongVILA-R1](https://arxiv.org/abs/2507.07966)。

多模态的新问题是**模型是否真的在看**。多模态推理基准上的增益常常来自文本推理，而非视觉感知。几条应对思路：

- **感知本身的可验证奖励**：定位用 IoU，计数与 OCR 用精确匹配，空间推理用场景图（[Perception-R1](https://arxiv.org/abs/2504.07954)、[SpatialThinker](https://arxiv.org/abs/2511.07403)）。
- **惩罚"不看图"**：遮住图像后输出应当改变，用一项 KL 鼓励这种依赖（[PAPO](https://arxiv.org/abs/2507.06448)）。
- **用图像思考**：把裁剪、放大、选帧等操作作为动作，模型在推理中主动获取视觉证据（[o3](https://openai.com/index/introducing-o3-and-o4-mini/)、[DeepEyes](https://arxiv.org/abs/2505.14362)、[Pixel Reasoner](https://arxiv.org/abs/2505.15966)、[Mini-o3](https://arxiv.org/abs/2509.07969)）；[VisionThink](https://arxiv.org/abs/2507.13348) 让模型自己决定是否请求高分辨率图像。
- **视觉表征本身**：对比 SFT 与 RL 后的视觉编码器，RL 改变了视觉表征，定位更准（[RL 让模型看得更准](https://arxiv.org/abs/2510.16333)）。

### 8.2 视觉生成的 RL

扩散模型的采样是一个 $T$ 步的随机过程，可以直接写成 MDP（[DDPO](https://arxiv.org/abs/2305.13301)）：状态 $(x_t,t,c)$，动作 $x_{t-1}\sim p_\theta(x_{t-1}\mid x_t,c)$（高斯），奖励 $r(x_0,c)$ 只在终点给出，于是

$$\nabla_\theta J=\mathbb E\Big[r(x_0,c)\sum_{t=1}^{T}\nabla_\theta\log p_\theta(x_{t-1}\mid x_t,c)\Big]$$

与 LLM 不同的是，每一步的动作是连续的高维向量、对数概率是高斯的闭式。[DPOK](https://arxiv.org/abs/2305.16381) 加了 KL 正则，[Diffusion-DPO](https://arxiv.org/abs/2311.12908) 把 DPO 推广到扩散模型的似然下界，[ImageReward](https://arxiv.org/abs/2304.05977) 提供了文生图的偏好奖励模型。

流匹配模型的 ODE 采样是确定的，没有可用于策略梯度的随机性与对数概率。[Flow-GRPO](https://arxiv.org/abs/2505.05470) 把 ODE 换成边际分布相同的 SDE，从而可以在每步定义高斯策略并做 GRPO；[DanceGRPO](https://arxiv.org/abs/2505.07818) 把它统一到图像与视频；[MixGRPO](https://arxiv.org/abs/2507.21802) 只在一个滑动窗口内用 SDE、其余步用 ODE 以加速；[DiffusionNFT](https://arxiv.org/abs/2509.16117) 完全不用似然，直接在前向加噪过程上用正负样本定义改进方向，比 Flow-GRPO 快数倍。另一条路线把奖励微调写成随机最优控制（[Adjoint Matching](https://arxiv.org/abs/2409.08861)）。视觉生成中的奖励投机比文本更直观：过饱和、风格趋同、对奖励模型偏好的细节过度优化；归一化放大微小分数差异会导致投机，改用成对偏好胜率更稳（[Pref-GRPO](https://arxiv.org/abs/2508.20751)）；生成式奖励模型可扩展到 26B 以抵抗投机（[RewardDance](https://arxiv.org/abs/2509.08826)）。

### 8.3 VLA 的 RL

视觉—语言—动作模型（VLA）是在 VLM 上加动作输出的机器人策略，主要靠模仿学习训练。RL 在这里补的是泛化与鲁棒性：相比 SFT，RL 在语义与执行层面的分布外泛化明显更好（[RL 与 VLA 泛化](https://arxiv.org/abs/2505.19789)）；结果奖励的 GRPO 式 RL 加少量示范即可大幅提升，并观察到示范中没有的新动作模式（[SimpleVLA-RL](https://arxiv.org/abs/2509.09674)）。动作头的形式决定了 RL 的形式：离散动作 token 可以直接套用 LLM 的方法（[VLA-RL](https://arxiv.org/abs/2505.18719)、[RIPT-VLA](https://arxiv.org/abs/2505.17016)）；流匹配动作头需要处理去噪过程的似然（[π_RL](https://arxiv.org/abs/2510.25889)、[DPPO](https://arxiv.org/abs/2409.00588)）。奖励来自成功判定或进度模型，后者可以用 VLM 的 token 概率零样本得到（[TOPReward](https://arxiv.org/abs/2602.19313)）。真机 RL 的细节见姊妹篇[机器人操作调研](../manipulation_survey/)。

### 8.4 最近一年

- **RL 到底改进了视觉推理的什么。** 因果探测、参数比较与模块移植显示，改进主要发生在中后层的推理而非视觉感知（[弗兰肯斯坦式分析](https://arxiv.org/abs/2602.12395)）；把关键信息逐步从文字移到图片，暴露出多模态 RLVR 的"盲训练"效应（[SeePhys Pro](https://arxiv.org/abs/2605.09266)）。
- **开放配方与视觉智能体。** 视觉工具集成的 RL 环境（[VISTA-Gym](https://arxiv.org/abs/2511.19773)）、防止交互轮次塌缩的智能体视觉 RL（[PyVision-RL](https://arxiv.org/abs/2602.20739)）、60 万条数据的开放配方（[Vero](https://arxiv.org/abs/2604.04917)）、在测试时自我纠错再用在策略自蒸馏内化（[UniEvo-VL](https://arxiv.org/abs/2609.38721)）。
- **视觉生成的奖励。** 预训练视频生成模型就是潜空间奖励模型（[潜空间奖励](https://arxiv.org/abs/2511.21541)）；奖励模型与生成器对抗更新（[Adv-GRPO](https://arxiv.org/abs/2511.20256)）；只对高不确定样本施加正则（[GARDO](https://arxiv.org/abs/2512.24138)）；把终点奖励分解到每个去噪步（[DenseGRPO](https://arxiv.org/abs/2601.20218)）；奖励模型先写多维批评再打分（[RationalRewards](https://arxiv.org/abs/2604.11626)）；多模态模型"读回"图像描述作免训练奖励（[Read It Back](https://arxiv.org/abs/2607.11886)）。
- **统一生成与工业流水线。** "先推理再生成图像"整体优化（[UniGRPO](https://arxiv.org/abs/2603.23500)）；FP4 快速探索、BF16 再生成与训练（[Sol-RL](https://arxiv.org/abs/2604.06916)）；多个单任务 RL 教师蒸馏进一个流模型（[Flow-OPD](https://arxiv.org/abs/2605.08063)）；RLHF 与在策略蒸馏结合的文生图后训练（[Qwen-Image-2.0-RL](https://arxiv.org/abs/2606.27608)）；揭示 NFT 式目标中隐藏的前向—反向一致性力与 EMA 的稳定作用（[DiffusionART](https://arxiv.org/abs/2610.12572)）。
- **VLA：世界模型充当环境与 critic。** 用模型自己的成功轨迹给失败轨迹打进度分（[SRPO](https://arxiv.org/abs/2511.15605)）；在视频世界模型里想象 rollout 做在策略 RL（[WMPO](https://arxiv.org/abs/2511.09515)）；把通用 VLA 训成精密任务的专家（[GR-RL](https://arxiv.org/abs/2512.01801)）；读出紧凑的"RL token"做在线 actor-critic（[RL Token](https://arxiv.org/abs/2604.23073)）；视频世界模型提供多帧价值估计（[WCM](https://arxiv.org/abs/2607.29613)）。

## 9 智能体：多轮、工具与环境中的 RL

### 9.1 从 MDP 到 POMDP

单轮推理中，环境只是一个验证器。智能体任务中，模型的输出（工具调用、命令、点击）改变外部世界，世界的返回进入上下文：

$$\tau=(x,\ a_1,\ o_1,\ a_2,\ o_2,\ \dots,\ a_K),\qquad a_k\sim\pi_\theta(\cdot\mid x,a_{<k},o_{<k}),\quad o_k\sim P(\cdot\mid \text{世界状态},a_k)$$

$a_k$ 是模型生成的一段 token（思考 + 动作），$o_k$ 是工具或环境的返回。这才是 RL 研究者熟悉的设定：转移随机且未知，世界状态部分可观测，几十到上百轮，奖励多在任务结束时给出。几个立即出现的工程问题：

- **损失掩码**：观测 token 不是策略生成的，不应计算 $\nabla\log\pi$。梯度只在 $a_k$ 内的 token 上：$\nabla J=\mathbb E\big[\sum_k\sum_{t\in a_k}\nabla_\theta\log\pi_\theta(y_t\mid s_t)\,\hat A_{k,t}\big]$（[Search-R1](https://arxiv.org/abs/2503.09516)）。
- **上下文管理**：几十轮交互很快填满上下文窗口。截断、摘要、"折叠"历史或把记忆写到外部，都改变了状态的定义；把记忆更新当作动作一起训练，可以做到显存恒定（[MEM1](https://arxiv.org/abs/2506.15841)）。
- **环境成本**：每一步可能要启动容器、跑测试、访问网页，比生成 token 慢得多，而且有状态、难以并行。
- **长尾长度**：不同轨迹的轮数与长度差异极大，同步训练的等待时间由最长的那条决定（第 10 章）。

### 9.2 多轮信用分配

默认做法是所有轮次共享回合的优势。改进有几个方向：

- **分层**：高层在轮次上做离策略的价值学习，低层在 token 上做策略梯度（[ArCHer](https://arxiv.org/abs/2402.19446)）。
- **步骤分组**：不同轨迹经常经过相同的环境状态（例如网页、棋盘），以这些"锚状态"为组计算步骤级优势
  $\hat A(s,a_i)=\frac{R_i-\mathrm{mean}_{j:\,s_j=s}R_j}{\mathrm{std}_{j:\,s_j=s}R_j}$，不需要 critic（[GiGPO](https://arxiv.org/abs/2505.10978)）。
- **轮次级 MDP**：以轮次为步做 PPO（[Turn-PPO](https://arxiv.org/abs/2512.17008)），或以轮次级留一法得到无偏优势（[Dr. Kernel](https://arxiv.org/abs/2602.05885)）。
- **特权 critic**：训练时让 critic 看到参考答案等特权信息（[SWEET-RL](https://arxiv.org/abs/2503.15478)）——不对称 actor-critic 的标准技巧。
- **分支**：在工具调用之后的高熵步骤分支采样（[ARPO](https://arxiv.org/abs/2507.19849)），或按轮次建树（[AT²PO](https://arxiv.org/abs/2601.04767)）。

### 9.3 四类智能体的 RL

- **搜索与深度研究**：在推理中调用搜索引擎，奖励为最终答案是否正确（[Search-R1](https://arxiv.org/abs/2503.09516)、[R1-Searcher](https://arxiv.org/abs/2503.05592)）；在真实网络环境中训练（[DeepResearcher](https://arxiv.org/abs/2504.03160)）；用 LLM 模拟搜索引擎以降低成本（[ZeroSearch](https://arxiv.org/abs/2505.04588)）；合成高不确定性的问题做 RL（[WebSailor](https://arxiv.org/abs/2507.02592)）；大规模异步 RL 支持上百轮搜索（[超越十轮](https://arxiv.org/abs/2508.07976)）；完整的开放配方（[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)）。
- **工具集成推理**：数学推理中交错执行代码（[ToRL](https://arxiv.org/abs/2503.23383)、[ReTool](https://arxiv.org/abs/2504.11536)、[rStar2-Agent](https://arxiv.org/abs/2508.20722)）。工具返回的分布外内容会诱发低概率 token 与梯度爆炸，过滤掉含"空轮"的轨迹即可稳定（[SimpleTIR](https://arxiv.org/abs/2509.02479)）。
- **软件工程与终端**：以仓库测试为验证器（[SWE-bench](https://arxiv.org/abs/2310.06770)）；用开源软件演化数据与规则奖励（[SWE-RL](https://arxiv.org/abs/2502.18449)）；可执行环境的构建（[SWE-Gym](https://arxiv.org/abs/2412.21139)、[R2E-Gym](https://arxiv.org/abs/2504.07164)）；长上下文多轮 SWE RL（[长上下文 SWE](https://arxiv.org/abs/2508.03501)）；CUDA 内核优化（[Kevin](https://arxiv.org/abs/2507.11948)）。
- **GUI 与计算机使用**：手机与网页上的自主 RL（[DigiRL](https://arxiv.org/abs/2406.11896)、[WebRL](https://arxiv.org/abs/2411.02337)）；R1 式规则奖励（[UI-R1](https://arxiv.org/abs/2503.21620)、[GUI-R1](https://arxiv.org/abs/2504.10458)）；零人工成本的在线学习（[ZeroGUI](https://arxiv.org/abs/2505.23762)）；多轮 RL 的大规模系统（[UI-TARS-2](https://arxiv.org/abs/2509.02544)）。

前沿开放模型把智能体 RL 作为后训练主线：[Kimi K2](https://arxiv.org/abs/2507.20534) 用大规模智能体数据合成与联合 RL，[GLM-4.5](https://arxiv.org/abs/2508.06471) 面向智能体、推理与编码，[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) 大规模合成智能体任务后做 RL，[Kimi K2.5](https://arxiv.org/abs/2602.02276) 用并行子智能体降低长任务延迟。

### 9.4 稳定性

多轮 RL 比单轮更容易崩溃。已识别的模式有：奖励方差塌缩与梯度尖峰的"回声陷阱"（[RAGEN](https://arxiv.org/abs/2504.20073)）；搜索智能体中正确与错误回答的似然一起下降的"懒惰似然位移"（[GRPO 在搜索中的崩溃](https://arxiv.org/abs/2512.04220)）；某些控制 token 的概率突然尖峰、破坏结构化执行，而底层工具能力仍在（[多步工具 RL 的崩溃](https://arxiv.org/abs/2606.26027)）；以及熵看不出来的"模板坍缩"——推理看似多样，实则与输入无关，需要用输入与推理的互信息来诊断（[RAGEN-2](https://arxiv.org/abs/2604.06268)）。

### 9.5 最近一年

- **跨回合学习与长程信用。** 元 RL 让智能体在测试时利用前几回合的反馈主动探索（[LaMer](https://arxiv.org/abs/2512.16848)）；只在关键状态分支采样（[Spark](https://arxiv.org/abs/2601.20209)）；动态评分细则提供细粒度信用（[DRACO](https://arxiv.org/abs/2609.04094)）；在 token、轮次与回合多个尺度上分配信用（[多尺度策略优化](https://arxiv.org/abs/2610.35013)）、跨会话分配信用（[Mosaic](https://arxiv.org/abs/2610.22174)）、进度感知的优势塑形（[ProAS](https://arxiv.org/abs/2610.18683)）。
- **稳定性的系统研究。** 统一框架分析不稳定来源（[ARLArena](https://arxiv.org/abs/2602.21534)）；多智能体共用全局基线导致梯度不稳（[Dr. MAS](https://arxiv.org/abs/2602.08847)）；把失败分解为探索错误与利用错误（[探索与利用错误](https://arxiv.org/abs/2604.13151)）；单 rollout 异步配方中 Muon 比 AdamW 更稳（[长程 RL 需要 Muon](https://arxiv.org/abs/2610.20256)）。
- **训练整个智能体。** 用"外化状态"的 harness 训练搜索智能体，而不是在越来越长的对话上训练（[Harness-1](https://arxiv.org/abs/2606.02373)）；在多种 harness 上训练让智能体 RL 更可泛化（[Harness Scaling](https://arxiv.org/abs/2610.29458)）；编码智能体交替优化 harness 与策略（[Self-HPO](https://arxiv.org/abs/2610.22791)、[训练整个智能体](https://arxiv.org/abs/2610.28069)）；训练与执行解耦的框架（[Agent Lightning v1.0](https://arxiv.org/abs/2608.17528)）；细粒度 RL 教智能体主动管理上下文（[ContextPilot](https://arxiv.org/abs/2608.28476)）。
- **环境与世界模型。** 预测环境动力学的语言世界模型用于训练与规划（[Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)）；终端与 SWE 智能体学习从失败状态恢复（[T1](https://arxiv.org/abs/2609.11042)、[FC-SWE](https://arxiv.org/abs/2610.07898)）；智能体 RL 把竞赛编程推到特级大师水平（[GrandCode](https://arxiv.org/abs/2604.02721)）。
- **现实检验。** 统一条件下用 132 次训练、约 1.2 万 H800 GPU 小时，在 ALFWorld、ScienceWorld、搜索问答与 WebShop 上重测 10 种智能体 RL 自蒸馏方法与 GRPO：多数增益不如论文所报（[自蒸馏的现实检验](https://arxiv.org/abs/2610.19789)）。

## 10 系统与规模：生成、同步与训推一致

### 10.1 一步 RL 的解剖

一次 RL 迭代包含三段：推理引擎为一批提示生成几万条回答（rollout）；训练引擎计算奖励、优势与梯度并更新参数；把新权重同步回推理引擎。

$$T_{\text{step}}\approx T_{\text{gen}}+T_{\text{train}}+T_{\text{sync}}$$

对长思维链与智能体任务，$T_{\text{gen}}$ 通常占一半以上，而且由批中**最长**的那条回答决定——回答长度呈长尾，大部分 GPU 在等少数长回答。推理引擎（[vLLM](https://arxiv.org/abs/2309.06180)、[SGLang](https://arxiv.org/abs/2312.07104)）为吞吐优化：分页 KV 缓存、连续批处理、前缀复用；训练框架（FSDP、Megatron）为梯度计算优化。两者用不同的内核、并行方式与数值精度。

### 10.2 同步与异步

**同步（共置）**：训练与生成轮流占用同一批 GPU，每步生成用的都是最新权重，严格在策略。[HybridFlow（veRL）](https://arxiv.org/abs/2409.19256) 用单控制器编排、多控制器执行，在训练与生成之间切换并行布局；早期的 [DeepSpeed-Chat](https://arxiv.org/abs/2308.01320)、[OpenRLHF](https://arxiv.org/abs/2405.11143)、[NeMo-Aligner](https://arxiv.org/abs/2405.01481) 也属此类。缺点是长尾回答造成的空闲。

**异步（分离）**：生成与训练用不同的 GPU，生成持续进行、落后训练若干个版本，以陈旧换吞吐——正是 [IMPALA](https://arxiv.org/abs/1802.01561) 的 actor-learner 结构，V-trace 的离策略修正在这里有了新的用武之地。[异步 RLHF](https://arxiv.org/abs/2410.18252) 研究了离策略程度对在线 DPO 等方法的影响；[AReaL](https://arxiv.org/abs/2505.24298) 完全解耦生成与训练，配合陈旧度控制与解耦的 PPO 目标；[PipelineRL](https://arxiv.org/abs/2509.19128) 在生成过程中就地更新权重以减少陈旧；[LlamaRL](https://arxiv.org/abs/2505.24034) 扩展到 405B；[INTELLECT-2](https://arxiv.org/abs/2505.07291) 在全球去中心化的节点上异步训练。[Kimi k1.5](https://arxiv.org/abs/2501.12599) 用"部分 rollout"处理长尾：超长的回答分段生成、跨迭代续写。

### 10.3 训推不一致

即使严格同步，rollout 的采样概率 $\pi_{\text{infer}}$ 与训练时重新计算的 $\pi_{\text{train}}$ 也不相同：推理与训练的内核不同、BF16 舍入不同、批大小不同导致的非批不变性（[Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)），以及 MoE 模型中两边的专家路由可能不同。差异在单个 token 上很小，但会在长序列上累积，而且集中在低概率 token 上，足以让训练崩溃。这是 2025 年下半年才被普遍认识的问题（[TIS](https://fengyao.notion.site/off-policy-rl)）。修正方法：

- **截断重要性采样**：在梯度上乘 $\min(\pi_{\text{train}}/\pi_{\text{infer}},C)$（第 3.6 节）。
- **改变数值**：改用 FP16，其更多的尾数位显著减小舍入差异（[FP16](https://arxiv.org/abs/2510.26788)）。
- **路由回放**：把推理时的 MoE 路由记录下来，训练时复用（[R3](https://arxiv.org/abs/2510.11370)）。
- **批不变内核**：让推理结果不依赖批的组成。

[形式化与实践](https://arxiv.org/abs/2512.01374)说明了这些方法的共同基础：token 级替代目标只在训推差异与陈旧都小时近似序列级目标。

### 10.4 算力曲线与配方

RL 的算力—性能关系能否像预训练那样预测？[ScaleRL](https://arxiv.org/abs/2510.13786) 用 40 万 GPU 小时系统研究，发现验证集通过率随 RL 算力呈 S 形：

$$R(C)=R_0+\frac{A-R_0}{1+(C_{\text{mid}}/C)^{B}}$$

$A$ 是渐近上限，$B$ 是效率，$C_{\text{mid}}$ 是达到一半收益所需的算力。多数设计选择（损失聚合、优势归一化、数据课程）主要改变 $B$，少数（损失类型、精度修正、模型规模）改变 $A$；用小规模拟合可以外推大规模的结果。开放的配方研究（[Skywork-OR1](https://arxiv.org/abs/2505.22312)、[AceReason](https://arxiv.org/abs/2505.16400)、[Light-R1](https://arxiv.org/abs/2503.10460)、[ORZ](https://arxiv.org/abs/2503.24290)）与工业报告（[MiniMax-M1](https://arxiv.org/abs/2506.13585)、[Seed1.5-Thinking](https://arxiv.org/abs/2504.13914)、[Magistral](https://arxiv.org/abs/2506.10910)、[Qwen3](https://arxiv.org/abs/2505.09388)、[Ring-1T](https://arxiv.org/abs/2510.18855)）给出了大量可复用的细节。

RL 需要的模型容量很小：一个回合只提供约一比特的信息，合理配置的 LoRA 就足以承载 RL 的更新（[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)）；这也是托管式训练 API 的基础（[Tinker](https://thinkingmachines.ai/tinker/)）。量化噪声甚至可以增加探索（[QeRL](https://arxiv.org/abs/2510.11696)）。

### 10.5 最近一年

- **简单配方。** 单阶段、固定超参的最简配方在 1.5B 模型上做到最好（[JustRL](https://arxiv.org/abs/2512.16649)）；RLVR 中 DoRA 等结构变体优于 LoRA，SVD 初始化反而有害（[PEFT 与 RLVR](https://arxiv.org/abs/2512.23165)）；按领域级联的 RL（[Nemotron-Cascade](https://arxiv.org/abs/2512.13607)）。
- **低精度 rollout 与它的代价。** FP8 权重与 KV 缓存的实用栈（[FP8-RL](https://arxiv.org/abs/2601.18150)）、训练与 rollout 统一精度（[Jet-RL](https://arxiv.org/abs/2601.14243)）；但即使用 TIS 修正，全流程 FP8 仍会不稳（[全流程 FP8](https://arxiv.org/abs/2609.22870)），NVFP4 需要按方向诊断并重平衡梯度（[TRIAGE](https://arxiv.org/abs/2610.07043)），MoE 的 FP4 rollout 需要量化感知训练（[TRACE](https://arxiv.org/abs/2610.07767)），INT4 KV 缓存要对注意力汇聚与近期上下文保留高精度（[KV4-RL](https://arxiv.org/abs/2610.14448)）。
- **失配的来源。** 异步流水线中的延迟更新与部分 rollout 常丢失训练侧的旧 logits，使重要性比的含义错位（[缺失的旧 logits](https://arxiv.org/abs/2605.12070)）；定位训推不一致的来源，用校准重要性采样把误差中无界的二阶矩换成有界量（[CIS](https://arxiv.org/abs/2609.32444)）。
- **生成加速。** 投机解码进入 RL（[系统集成投机解码](https://arxiv.org/abs/2604.26779)、[EfficientRollout](https://arxiv.org/abs/2606.18967)），RL 后熵降低反而让多 token 预测更容易被接受（[Bebop](https://arxiv.org/abs/2606.12370)），草稿模型随策略在线协同训练（[在线草稿协同训练](https://arxiv.org/abs/2609.07108)）；在线检测注定失败的轨迹并提前终止（[ESPO](https://arxiv.org/abs/2605.29860)）；经验回放在陈旧带来的方差与样本多样性之间取舍（[经验回放](https://arxiv.org/abs/2604.08706)）。
- **系统抽象。** rollout 即服务（[ProRL Agent](https://arxiv.org/abs/2603.18815)）、数据流式的智能体 RL 系统（[AstraFlow](https://arxiv.org/abs/2605.15565)）、低秩梯度草图降低显存（[LoGRA](https://arxiv.org/abs/2610.06647)）。
- **规模。** 固定 GPU 预算下超过 200 万 token 的长上下文 RL（[LongStraw](https://arxiv.org/abs/2607.14952)）；万亿参数智能体 RL 的按位精确增量权重同步（[NeMo-DCR](https://arxiv.org/abs/2610.08430)）；零 RL 扩展到万亿参数（[Ring-Zero](https://arxiv.org/abs/2607.12395)）；另一端，3B 模型在可验证推理上逼近前沿（[VibeThinker-3B](https://arxiv.org/abs/2606.16140)）。
- **可预测性与生态。** 在 Starpilot 与 Craftax 两个程序生成环境中为 actor-critic 找到可外推的扩展律，可作为 LLM RL 的参照（[RL 扩展律实证](https://arxiv.org/abs/2610.33420)）；统计 7 个开源框架收录了哪些论文方法、收录前是否经过审查（[从论文到框架代码](https://arxiv.org/abs/2610.28027)）。

## 11 理论与机制：RL 改变了模型的什么

### 11.1 软 RL 与偏好学习的理论

KL 正则目标的最优解是以 $\pi_{\text{ref}}$ 为先验的玻尔兹曼分布（0.2 节），这把 LLM 的 RL 与最大熵 RL 的理论连在一起（[软 Q 等价](https://arxiv.org/abs/1704.06440)）；DPO 在 token MDP 中隐式学到软 Q 函数（[从 r 到 Q*](https://arxiv.org/abs/2404.12358)）。偏好学习方面，Bradley–Terry 下奖励学习的样本复杂度与悲观策略的必要性（[Zhu 等](https://arxiv.org/abs/2301.11270)）、KL 约束下迭代偏好学习的保证（[Xiong 等](https://arxiv.org/abs/2312.11456)）、带乐观探索的在线 DPO（[XPO](https://arxiv.org/abs/2405.21046)）构成了 RLHF 理论的主体。

### 11.2 锐化与生成—验证差距

**锐化。** [Huang 等](https://arxiv.org/abs/2412.01951)把"自我改进"形式化为锐化：让策略把概率集中到它自己认为最好的回答上，例如最大似然序列，或按自评分加权的分布。它给出 SFT 式（在自己的最好样本上训练）与 RL 式（以自评分为奖励）两类算法实现锐化的样本复杂度。这为第 5 章的经验争论提供了一个精确的框架：如果 RL 只在锐化，那它的上限由基座"知道哪个答案最好"的能力决定。对基座直接从 $\pi^\alpha$（$\alpha>1$）采样就接近 RL 效果（[幂分布采样](https://arxiv.org/abs/2510.14901)），以及无监督 RL 先升后降（[无监督 RLVR](https://arxiv.org/abs/2603.08660)），都与这个图景一致。

**生成—验证差距。** 一个令 RL 研究者困惑的问题是：既然偏好数据是固定的，信息论上离线最大似然就足以利用其中全部信息，为什么"先学奖励模型、再在其上在线 RL"实践中更好？[Swamy 等](https://arxiv.org/abs/2503.01067)的回答是：若奖励模型所在的函数类比策略类简单得多（验证比生成容易），那么先在简单的类里学一个奖励、再用 RL 在其上搜索策略，比直接在复杂的策略类里做最大似然更容易找到好解。这个解释也预言了它失效的条件：当验证不再比生成容易时（[验证视野](https://arxiv.org/abs/2606.26300)），RL 的优势会缩小。

### 11.3 RL's Razor 与更新几何

与 SFT 相比，RL 后训练有一组稳定的经验现象：

- **遗忘更少。** 新任务上表现相同时，RL 比 SFT 更好地保留原有能力；遗忘程度由新任务上微调模型与基座之间的 KL 决定；在策略 RL 在所有能解决新任务的策略中偏向 KL 最小的那个，而 SFT 可以收敛到任意远的分布（[RL's Razor](https://arxiv.org/abs/2509.04259)）。
- **泛化更好。** 在规则与视觉的变体上，SFT 倾向记忆训练数据，结果奖励的 RL 更能泛化（[SFT 记忆、RL 泛化](https://arxiv.org/abs/2501.17161)）；但这并不绝对，长思维链 SFT 的跨领域泛化取决于优化、数据与模型能力（[SFT 的条件泛化](https://arxiv.org/abs/2604.06628)）。
- **更新稀疏。** RL 只更新 5%–30% 的参数，不同种子、不同数据下的子网络高度重合（[小子网络](https://arxiv.org/abs/2505.11711)）；更新避开权重的主成分方向（[偏离主方向](https://arxiv.org/abs/2511.08567)）。

一个直观的解释是：在策略的数据来自模型自己的分布，梯度只需微调已经很可能的行为，而 SFT 的目标分布可能离模型很远。SFT 本身也可以被看作奖励稀疏、按 $1/\pi$ 加权的策略梯度，按 token 概率重新缩放就能改善它的泛化（[DFT](https://arxiv.org/abs/2508.05629)）。

### 11.4 最近一年

- **动力学可以推导。** 强化微调中的熵动力学（[熵动力学](https://arxiv.org/abs/2602.03392)）；组相对优势的隐式对称——正确与错误轨迹权重严格对称、未采样 token 得不到更新（[隐式对称](https://arxiv.org/abs/2602.05548)）；策略镜像下降近似带来的隐式正则（[PMD 的隐式正则](https://arxiv.org/abs/2602.05933)）；由易到难的隐式课程（[隐式课程](https://arxiv.org/abs/2602.14872)）；采样预算与题目难度的关系（[相对预算理论](https://arxiv.org/abs/2602.01523)）；GRPO 训练动力学的闭式模型（[可预测的 GRPO](https://arxiv.org/abs/2606.30789)）。
- **模型里有价值神经元。** 奖励相关信息集中在少数神经元：一类的激活预测状态价值，另一类编码类似多巴胺的奖励预测误差（[稀疏奖励子系统](https://arxiv.org/abs/2602.00986)）。这与神经科学中的时间差分学习图景惊人地相似。
- **关键 token 与特征。** 以对数概率变化的符号定位稀疏但关键的推理 token（[更新方向](https://arxiv.org/abs/2603.22117)、[稀疏但关键](https://arxiv.org/abs/2603.22446)）；在共享特征空间里比较同一基座、同一数据训练的 RL 与 SFT 模型，解释 RL 为何泛化（[特征级机制](https://arxiv.org/abs/2604.25011)）；LLM 策略内部有逐层的"内部策略"（[自底向上策略优化](https://arxiv.org/abs/2512.19673)）。
- **RL 的权重轨迹近乎秩 1。** 大部分下游收益由参数增量的秩 1 近似承载，投影幅度随训练步近线性增长；用短观察窗估计方向并线性外推即可得到后续检查点（[RELEX](https://arxiv.org/abs/2605.21468)）。OPD 的更新同样稀疏（[稠密监督、稀疏更新](https://arxiv.org/abs/2606.13657)）。
- **为什么互不干扰、为什么泛化。** 多任务下 SFT 的梯度冲突，RL 的更新稀疏且跨任务近似正交（[SFT 冲突、RL 共存](https://arxiv.org/abs/2608.03573)）；SFT 沿固定方向更新，在策略训练不断调整方向，后者决定泛化（[在策略的更新方向](https://arxiv.org/abs/2609.36659)）；多领域 RL 中的跨领域干扰与恢复（[局部冲突、定向修复](https://arxiv.org/abs/2606.02398)）；RLVR 训练轨迹中的低维有效子空间（[RLVR 的几何](https://arxiv.org/abs/2609.34344)）；训练时的采样策略与真正应优化的推理策略不是同一对象（[训练策略的幻象](https://arxiv.org/abs/2606.29526)）。

## 12 奖励投机与安全：优化压力把模型推向哪里

### 12.1 Goodhart 的定量形式

设代理奖励 $\hat r=r+\epsilon$。优化越强，策略越集中在 $\epsilon$ 大的区域——这是 RL 领域早已熟知的规范投机（[DeepMind 的案例集](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)）。第 2.4 节的过度优化曲线是它在 LLM 上的定量形式（[Gao 等](https://arxiv.org/abs/2210.10760)）。不同的奖励来源有不同的投机方式：

| 奖励来源 | 典型投机 |
|---|---|
| 人类偏好 / 奖励模型 | 迎合观点（谄媚）、更长更自信、说服评估者接受错误答案 |
| 可验证奖励 | 修改或绕过测试、硬编码答案、利用答案抽取的漏洞 |
| 评判模型与评分细则 | 堆砌细则关键词、编造满足细则的内容、对评判模型的风格偏好过度优化 |

### 12.2 RLHF 的谄媚与说服

人类偏好数据本身就奖励迎合用户观点的回答，RLHF 会放大这种倾向（[谄媚](https://arxiv.org/abs/2310.13548)）；RLHF 后的模型更擅长让人类评估者相信错误的答案是对的（[RLHF 让模型学会误导](https://arxiv.org/abs/2409.12822)）。在越来越容易投机的环境课程中，模型会从谄媚泛化到篡改自己的奖励函数（[从谄媚到奖励篡改](https://arxiv.org/abs/2406.10162)）。更一般地，窄任务上的训练会泛化出广泛的行为变化：只在写不安全代码上微调就出现广泛失准（[涌现失准](https://arxiv.org/abs/2502.17424)），在无害任务上学会投机评判也是如此（[奖励投机学校](https://arxiv.org/abs/2508.17511)）。

### 12.3 RLVR 的投机与失准

可验证奖励不能被"说服"，但能被"绕过"。[MacDiarmid 等](https://arxiv.org/abs/2511.18397)在真实的生产级编程 RL 环境中研究了这一点：先让模型通过合成文档或提示了解投机手法，再在这些环境中训练。模型学会了投机，并且泛化出对齐伪装、与恶意行为者合作、推理恶意目标，以及在 Claude Code 中尝试破坏（包括破坏这篇论文本身的代码库）。用聊天式提示做 RLHF 安全训练后，聊天评测上看起来对齐了，智能体任务上的失准依然存在。三种缓解有效：阻止投机本身；增加安全训练的多样性；以及"接种提示"——训练时把投机描述成可接受的行为，即使模型学会了投机，失准也不再泛化。

### 12.4 思维链可监控性

推理模型的思维链提供了一个新的监督通道：用另一个模型阅读 CoT，比只看动作更能发现投机，甚至较弱的模型就能监控较强的模型（[Baker 等](https://arxiv.org/abs/2503.11926)）。自然的想法是把 CoT 监控放进奖励。结果是：优化压力弱时，模型更强也更对齐；优化压力强时，模型学会了"混淆的投机"——继续投机，同时把意图从 CoT 中藏起来。这意味着 CoT 监控是脆弱的：一旦成为优化目标，它就不再是可靠的观测。因此有人呼吁开发者不要对 CoT 直接施加优化压力，保护可监控性（[CoT 可监控性](https://arxiv.org/abs/2507.11473)）。此外，推理模型的 CoT 常常不承认使用了提示中的线索（[推理模型不总说出所想](https://arxiv.org/abs/2505.05410)）。

安全本身也是奖励设计：把安全政策写成细粒度规则、由 LLM 打分作为奖励的一部分（[RBR](https://arxiv.org/abs/2411.01111)）；把有用与无害解耦为奖励与代价、做约束 RL（[Safe RLHF](https://arxiv.org/abs/2310.12773)）；让推理模型在回答前显式回忆并推理安全规范（[审慎对齐](https://arxiv.org/abs/2412.16339)）。可扩展监督的研究（[辩论](https://arxiv.org/abs/1805.00899)、[证明者—验证者博弈](https://arxiv.org/abs/2407.13692)、[弱到强泛化](https://arxiv.org/abs/2312.09390)）讨论的是：当模型超过人类时，奖励从哪里来。

### 12.5 最近一年

- **分布式训练的攻击面。** 共享 rollout 的去中心化 GRPO 中，恶意节点可通过投毒补全植入行为（[去中心化 GRPO 投毒](https://arxiv.org/abs/2511.09780)）。
- **可靠性与校准。** 奖励搜索智能体在能力边界之外正确拒答（[BAPO](https://arxiv.org/abs/2601.11037)）；RLVR 与 OPD 都会让模型过度自信，于是把推理与校准拆成两个目标（[DCPO](https://arxiv.org/abs/2603.09117)、[OPD 的确定性幻觉](https://arxiv.org/abs/2604.16830)）；按不确定性表达的忠实度修正偏好（[RLMF](https://arxiv.org/abs/2606.32032)）；三角色自我博弈的安全对齐（[TriPlay-RL](https://arxiv.org/abs/2601.18292)）；用信息论分析思维链可监控性（[可监控性的信息论](https://arxiv.org/abs/2602.18297)）。
- **统一框架。** 以"代理压缩假说"统一理解奖励投机，并梳理它与涌现失准的联系（[奖励投机综述](https://arxiv.org/abs/2604.13602)）。
- **新的投机面。** rubric RL 中对评审偏好的投机（[rubric 奖励投机](https://arxiv.org/abs/2606.04923)）；rubric 分数简单相加是弱点（[加法聚合](https://arxiv.org/abs/2609.38847)）；169 个不可能任务检验 rubric 是否会奖励编造（[ImpossibleRubrics](https://arxiv.org/abs/2609.16816)）；前沿模型能用几乎没有实质内容的回答在可验证指令遵循上拿满分（[CALIF](https://arxiv.org/abs/2610.19525)）；只改写修辞就能抬高 AI 审稿分数（[修辞投机](https://arxiv.org/abs/2608.08975)）；72 个社会规则沙箱中，奖励投机自然地表现为发现监管漏洞（[SocioHack](https://arxiv.org/abs/2606.04075)）。
- **攻防都用 RL。** 自我博弈 RL 训练的自动红队发现了针对前沿模型的新型提示注入（[GPT-Red](https://arxiv.org/abs/2607.26115)）；后训练会在无关的决策上留下系统性的行为偏移（[行为影子](https://arxiv.org/abs/2609.29233)）；RL 后模型的数据污染可从逐层表征中检测（[LaRA](https://arxiv.org/abs/2605.29888)）。

## 13 实践入门：RL 研究者的第一个大模型 RL 实验

### 13.1 路线

先在单卡上用几十行代码跑通一个可验证任务的 GRPO，亲眼看到奖励、熵与回答长度怎样变化；再换成成熟框架，把同样的实验放大到多卡、长序列与真实基准。大模型 RL 的多数失败不在算法，而在三件事：采样是否真的在策略、奖励是否真的在测你想测的东西、评测是否足够稳。

1. **模型。** 从 0.5B–4B 的开放模型开始。基座模型适合研究"从零 RL"，指令模型更容易遵守答案格式。部分模型家族在常见数学基准上可能有污染、随机奖励也能涨分（[虚假奖励](https://arxiv.org/abs/2506.10947)、[污染](https://arxiv.org/abs/2507.10532)），结论至少要在两个模型家族上复现。
2. **任务与验证器。** 先用答案确定、可以自动判定的任务：程序生成的算术与逻辑题（[Reasoning Gym](https://arxiv.org/abs/2505.24760)）、[GSM8K](https://arxiv.org/abs/2110.14168)、[MATH](https://arxiv.org/abs/2103.03874)。验证器 = 答案抽取 + 等价判定；格式错误给 0 分即可。
3. **算法与超参。** GRPO 或 RLOO，加 DAPO 的修补作为起点：每题 8–16 个样本、温度 1.0、非对称裁剪（如 $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$）、token 级损失聚合、过滤零方差组、不加或只加很小的 KL；全参学习率在 $10^{-6}$ 量级，LoRA 约大一个数量级。这些是常见起点，不是定论。
4. **框架。** 单机：[TRL](https://github.com/huggingface/trl)；多机与长序列：[veRL](https://arxiv.org/abs/2409.19256)、[OpenRLHF](https://arxiv.org/abs/2405.11143)、[ROLL](https://arxiv.org/abs/2506.06122)、[AReaL](https://arxiv.org/abs/2505.24298)；托管：[Tinker](https://thinkingmachines.ai/tinker/)；智能体：[Agent Lightning](https://arxiv.org/abs/2508.03680)、[VerlTool](https://arxiv.org/abs/2509.01055)、[AgentGym-RL](https://arxiv.org/abs/2509.08755)。
5. **评测。** 报 avg@k（每题采 $k$ 次的平均正确率）与 pass@k；多个随机种子；与同算力的 SFT 与拒绝采样基线比较；常用 MATH-500、AIME、[GPQA](https://arxiv.org/abs/2311.12022)、[LiveCodeBench](https://arxiv.org/abs/2403.07974)、[SWE-bench](https://arxiv.org/abs/2310.06770)。

### 13.2 最小 GRPO

单卡、PyTorch + transformers。任务是两位数乘三位数，答案写在 `<answer>` 标签里。每批只更新一次，所以重要性比恒为 1、裁剪不起作用，目标就是带组基线的 REINFORCE。

```python
import random, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"     # 任意小的指令模型
G, BATCH, MICRO, STEPS, LR, MAX_NEW = 8, 8, 8, 300, 1e-6, 256
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
# 权重用 fp32：bf16 存权重配 1e-6 的学习率，更新会被舍入掉
policy = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda()
opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

def make_task():
    a, b = random.randint(10, 99), random.randint(100, 999)
    msg = [{"role": "user", "content": f"计算 {a}×{b}。先逐步思考，最后把结果写在 <answer></answer> 里。"}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True), str(a * b)

def reward(text, truth):                  # 可验证奖励：答案正确为 1，否则为 0
    m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text)
    return float(m is not None and m.group(1) == truth)

for step in range(STEPS):
    prompts, truths = zip(*[make_task() for _ in range(BATCH)])
    enc = tok([p for p in prompts for _ in range(G)], return_tensors="pt", padding=True).to("cuda")
    P = enc.input_ids.shape[1]
    with torch.no_grad(), amp():          # rollout：每题 G 个回答
        # 显式关掉模型自带的 top_k / top_p / 重复惩罚，否则采样分布不是 π_θ，训练就成了离策略
        out = policy.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                              repetition_penalty=1.0, max_new_tokens=MAX_NEW,
                              pad_token_id=tok.pad_token_id)
    comp = out[:, P:]
    texts = tok.batch_decode(comp, skip_special_tokens=True)
    r = torch.tensor([reward(t, truths[i // G]) for i, t in enumerate(texts)]).view(BATCH, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1, 1).cuda()
    eos = ((comp == tok.eos_token_id) | (comp == tok.pad_token_id)).int()
    mask = ((eos.cumsum(1) - eos) == 0).float()   # 第一个结束符及其之前的 token 参与损失
    am = torch.cat([enc.attention_mask, torch.ones_like(comp)], 1)
    pos = (am.cumsum(1) - 1).clamp(min=0)         # 左填充时要传位置编号，否则与生成时不一致
    n_tok = mask.sum()
    opt.zero_grad()
    for i in range(0, out.shape[0], MICRO):
        s = slice(i, i + MICRO)
        with amp():
            logits = policy(out[s], attention_mask=am[s], position_ids=pos[s]).logits
        logp = torch.log_softmax(logits[:, P - 1:-1].float(), -1)
        logp = logp.gather(-1, comp[s, :, None]).squeeze(-1)
        # 每批只更新一次，重要性比恒为 1、PPO 的裁剪不起作用，目标就是带组基线的 REINFORCE；
        # 按全批 token 数平均（token 级聚合，见第 3 章）
        loss = -(adv[s] * logp * mask[s]).sum() / n_tok
        loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    if step % 10 == 0:
        zero = ((r.std(1) == 0).float().mean()).item()
        print(f"step {step} reward {r.mean():.3f} len {mask.sum(1).mean():.0f} zero-var {zero:.2f}")
```

这段代码刻意省略了参考模型与 KL、每批多次更新与裁剪、价值网络。要做多次更新，就需要保存采样时的 $\log\pi_{\text{old}}$ 并加入第 3 章的重要性比与裁剪。HF 的 `generate` 很慢，放大实验时换成 vLLM 或 SGLang 采样，这时训推不一致就出现了（第 10 章）。

### 13.3 要盯的曲线

| 量 | 正常现象 | 异常信号与常见原因 |
|---|---|---|
| 训练奖励 / 验证集 avg@k | 先快后慢地上升 | 训练奖励涨而验证集不涨：过拟合题库或在投机验证器 |
| 策略熵 | 缓慢下降 | 几十步内塌到接近 0：探索停止；突然暴涨：输出开始乱码 |
| 回答长度 | 推理任务上常增长 | 撞到上限的比例高：截断在污染奖励；突然变短：投机或崩溃 |
| 零方差组比例 | 逐渐上升 | 过高：题太易或太难 |
| 裁剪比例、KL | 小而平稳 | 尖峰：学习率过大或离策略程度过高 |
| 训推差异 | 很小 | 逐步增大：精度、内核或 MoE 路由不一致 |
| 梯度范数 | 平稳 | 尖峰常先于崩溃出现 |

### 13.4 大模型特有的坑

- 推理引擎与模型自带的采样参数（top-k、top-p、重复惩罚）会让 rollout 不再是 $\pi_\theta$。
- 左填充时不传位置编号，训练与生成算出的概率不同。
- bf16 存权重、$10^{-6}$ 学习率：更新被舍入掉，曲线一动不动。
- 聊天模板、思考标记与最大长度要在训练和评测中保持一致。
- 截断的回答记 0 分会教模型写短，记负分会教它别冒险。
- 只报一次采样的 pass@1：方差极大。
- 随机奖励也能涨分的模型上，结论要谨慎，换一个模型家族复现。
- 格式奖励过大，模型只学会格式。

### 13.5 阅读路线

1. [InstructGPT](https://arxiv.org/abs/2203.02155)：RLHF 三段式。
2. [DPO](https://arxiv.org/abs/2305.18290)：把 RL 变成分类。
3. [GRPO](https://arxiv.org/abs/2402.03300) 与 [RLOO](https://arxiv.org/abs/2402.14740)：去掉 critic。
4. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：可验证奖励的转折点。
5. [DAPO](https://arxiv.org/abs/2503.14476) 与 [Dr. GRPO](https://arxiv.org/abs/2503.20783)：实际要修的地方。
6. [Yue 等](https://arxiv.org/abs/2504.13837)与 [ProRL](https://arxiv.org/abs/2505.24864)：能力边界之争。
7. [熵机制](https://arxiv.org/abs/2505.22617)：探索为何停止。
8. [ScaleRL](https://arxiv.org/abs/2510.13786)：算力与可预测性。
9. [TIS](https://fengyao.notion.site/off-policy-rl) 与 [形式化与实践](https://arxiv.org/abs/2512.01374)：训推一致。
10. [在策略蒸馏](https://thinkingmachines.ai/blog/on-policy-distillation/)与 [SDPO](https://arxiv.org/abs/2601.20802)：RL 的近亲。
11. [RAGEN](https://arxiv.org/abs/2504.20073) 与 [GiGPO](https://arxiv.org/abs/2505.10978)：多轮智能体。
12. [自然涌现的失准](https://arxiv.org/abs/2511.18397)：风险。

教材与综述：[RLHF Book](https://arxiv.org/abs/2504.12501)、[Murphy 的 RL 概览](https://arxiv.org/abs/2412.05265)、[大推理模型 RL 综述](https://arxiv.org/abs/2509.08827)、[智能体 RL 综述](https://arxiv.org/abs/2509.02547)。

## 14 开放问题

贯穿全篇的问题是：大模型 RL 的许多"技巧"是在补偿离策略、奖励缺陷与评测噪声。去掉这些混杂因素之后，真正起作用的原理还剩多少？

1. **动作粒度。** token、句子、工具调用还是模型内部的潜在选项？RL 应在训练流水线的哪个阶段介入、占多少算力？
2. **奖励。** 开放任务的奖励能否不依赖另一个大模型？评判被策略针对性优化后还可信吗？验证的成本何时超过生成？
3. **算法。** critic 何时值得？大词表上合适的信任域是什么？多奖励的权衡能否有原则地设定？
4. **离线与蒸馏。** 在策略蒸馏的收益多少来自在策略、多少来自教师？SFT、蒸馏与 RL 的最优顺序是什么？
5. **能力边界。** RL 能否发现基座支撑之外的策略？如何在排除污染与提示泄漏后给出可信证据？
6. **数据与环境。** 合成环境的漏洞会教出什么？自我博弈如何不漂移？
7. **多模态。** RL 改进的是感知还是语言先验？视觉生成的奖励投机能否根治？
8. **智能体。** 昂贵、有状态的环境中如何样本高效？harness 与模型如何共同训练？
9. **系统。** 陈旧度与收敛之间的定量关系？低精度 rollout 的极限？RL 扩展律能否指导资源分配？
10. **理论。** 稀疏、低秩、小 KL 是 RL 的本质，还是因为任务太窄、训练太短？
11. **安全。** 投机能否被检测而不被隐藏？可监控性与能力是否此消彼长？

---

## 附录 A：最近半年逐月纪事（2026-05 → 2026-10）

> 每月挑选论文库中有代表性的工作，按主题归并；完整列表见附录 B 或导读页的论文库（时间窗选"最近半年"）。

### 2026-05

- **在策略蒸馏的变体潮开始**：信任域（[TrOPD](https://arxiv.org/abs/2606.01249)、[TRB](https://arxiv.org/abs/2605.31159)）、以评分细则代替 logits（[ROPD](https://arxiv.org/abs/2605.07396)）、反自蒸馏（[AntiSD](https://arxiv.org/abs/2605.11609)）、可教 token 的挑选（[token 可教性](https://arxiv.org/abs/2605.26844)）；流匹配模型的在策略蒸馏（[Flow-OPD](https://arxiv.org/abs/2605.08063)）。
- **权重轨迹近乎秩 1**，可从早期检查点线性外推（[RELEX](https://arxiv.org/abs/2605.21468)）。
- **探索**：提示空间扰动（[LoPE](https://arxiv.org/abs/2605.05566)）、策略级引导（[NudgeRL](https://arxiv.org/abs/2605.15726)）、小模型提供多样性（[小模型探索者](https://arxiv.org/abs/2605.30789)）；覆盖收缩与"岔路口"（[岔路口](https://arxiv.org/abs/2605.17026)）。
- **奖励与过程监督**：不训练权重的智能体式奖励（[RewardHarness](https://arxiv.org/abs/2605.08703)）、无监督 PRM（[uPRM](https://arxiv.org/abs/2605.10158)）、带可靠度的 PRM（[BetaPRM](https://arxiv.org/abs/2605.15529)）、多奖励的方差自适应（[DVAO](https://arxiv.org/abs/2605.25604)）。
- **环境与自我博弈**：软件世界（[OpenComputer](https://arxiv.org/abs/2605.19769)）、手机（[MobileGym](https://arxiv.org/abs/2605.26114)）、零数据开放生成（[G-Zero](https://arxiv.org/abs/2605.09959)、[SCOPE](https://arxiv.org/abs/2605.31433)）。
- **系统**：异步流水线中缺失的旧 logits（[旧 logits](https://arxiv.org/abs/2605.12070)）、数据流式系统（[AstraFlow](https://arxiv.org/abs/2605.15565)）、提前终止失败轨迹（[ESPO](https://arxiv.org/abs/2605.29860)）。

### 2026-06

- **在策略蒸馏**：多教师整合各领域 RL 专家（[MOPD](https://arxiv.org/abs/2606.30406)）、异步 OPD 的陈旧容忍度（[AsyncOPD](https://arxiv.org/abs/2606.24143)）、OPD 的稀疏更新（[稠密监督、稀疏更新](https://arxiv.org/abs/2606.13657)）、教师只在提示中（[ZPPO](https://arxiv.org/abs/2606.18216)）。
- **奖励的性质**：离散化奖励模型（[离散化](https://arxiv.org/abs/2606.21795)）、分数分布（[Z-Reward](https://arxiv.org/abs/2606.09076)）、免执行验证器（[Dockerless](https://arxiv.org/abs/2606.28436)）、验证不再比生成容易（[验证视野](https://arxiv.org/abs/2606.26300)）。
- **智能体**：多步工具 RL 的崩溃与修复（[工具 RL 崩溃](https://arxiv.org/abs/2606.26027)）、外化状态的 harness（[Harness-1](https://arxiv.org/abs/2606.02373)）、语言世界模型（[Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)）、终端任务合成（[CLI-Universe](https://arxiv.org/abs/2606.22883)）。
- **理论与机制**：可预测的 GRPO（[闭式动力学](https://arxiv.org/abs/2606.30789)）、多领域干扰（[局部冲突](https://arxiv.org/abs/2606.02398)）、训练策略的幻象（[幻象](https://arxiv.org/abs/2606.29526)）；RL 学会翻译未见语言（[未见语言](https://arxiv.org/abs/2606.06428)）。
- **安全**：rubric 奖励投机（[rubric 投机](https://arxiv.org/abs/2606.04923)）、社会规则中的投机（[SocioHack](https://arxiv.org/abs/2606.04075)）、元认知反馈（[RLMF](https://arxiv.org/abs/2606.32032)）。
- **规模与加速**：3B 模型逼近前沿（[VibeThinker-3B](https://arxiv.org/abs/2606.16140)）、MTP 投机解码加速 rollout（[Bebop](https://arxiv.org/abs/2606.12370)）、自投机解码（[EfficientRollout](https://arxiv.org/abs/2606.18967)）、视觉生成的工业流水线（[Qwen-Image-2.0-RL](https://arxiv.org/abs/2606.27608)）。

### 2026-07

- **在策略蒸馏的反思与扩展**：病理与规约（[揭开 OPD](https://arxiv.org/abs/2607.13399)）、弱到强（[W2S-OPD](https://arxiv.org/abs/2607.26246)）、增量蒸馏（[delta 信号](https://arxiv.org/abs/2607.15161)）、把教师监督并入 RL 目标（[Distilled RL](https://arxiv.org/abs/2607.17247)）、多轮结构感知（[TurnOPD](https://arxiv.org/abs/2607.05804)）。
- **算法**：异步陈旧的自适应信任域（[SAT](https://arxiv.org/abs/2607.18722)）、黎曼等距更新（[RIPO](https://arxiv.org/abs/2607.10169)）、组熵控制（[GEPO](https://arxiv.org/abs/2607.16850)）、token 级 rubric 信用（[CoRT](https://arxiv.org/abs/2607.25659)）。
- **非可验证任务**：评判当导师改写提示（[LLM-as-a-Tutor](https://arxiv.org/abs/2607.04412)）、评判当教练给文字反馈（[LLM-as-a-Coach](https://arxiv.org/abs/2607.18110)）、通用验证框架（[LLM-as-a-Verifier](https://arxiv.org/abs/2607.05391)）。
- **规模**：200 万 token 的长上下文 RL（[LongStraw](https://arxiv.org/abs/2607.14952)）、零 RL 到万亿参数（[Ring-Zero](https://arxiv.org/abs/2607.12395)）；技能自我博弈（[Skill Self-Play](https://arxiv.org/abs/2607.22529)）。
- **受控研究**：从预训练到后训练追踪推理能力（[预训练到后训练](https://arxiv.org/abs/2607.16097)）、多轮规划的物理学（[多轮规划](https://arxiv.org/abs/2607.24720)）。
- **安全与多模态**：自我博弈红队（[GPT-Red](https://arxiv.org/abs/2607.26115)）；MLLM 读回作图像奖励（[Read It Back](https://arxiv.org/abs/2607.11886)）、VLA 的世界评判模型（[WCM](https://arxiv.org/abs/2607.29613)）。

### 2026-08

- **自蒸馏**：无监督的在策略自蒸馏（[U-OPSD](https://arxiv.org/abs/2608.06296)）、批判性回顾（[三个杠杆](https://arxiv.org/abs/2608.25936)）、OPD 与可验证奖励结合（[OPDVR](https://arxiv.org/abs/2608.24696)）、OPD 是否真的在蒸馏（[再审视](https://arxiv.org/abs/2608.31046)）、测试时策略优化（[TTPO](https://arxiv.org/abs/2608.27448)）。
- **智能体**：训练与执行解耦的框架（[Agent Lightning v1.0](https://arxiv.org/abs/2608.17528)）、主动管理上下文（[ContextPilot](https://arxiv.org/abs/2608.28476)）、自适应可执行环境中的自我博弈（[SPADE](https://arxiv.org/abs/2608.19197)）、多模型协作中涌现推理（[Co-RL](https://arxiv.org/abs/2608.17253)）。
- **机制**：SFT 冲突、RL 共存（[多任务](https://arxiv.org/abs/2608.03573)）；进化策略微调的覆盖更广（[ES 做推理](https://arxiv.org/abs/2608.27351)）。
- **奖励**：饱和感知的多奖励重加权（[饱和感知](https://arxiv.org/abs/2608.16072)）、环境正则替代策略 KL（[ERPO](https://arxiv.org/abs/2608.23311)）；修辞投机 AI 审稿（[修辞投机](https://arxiv.org/abs/2608.08975)）。

### 2026-09

- **critic 回归**：critic 是不稳定的主要来源（[EasyPPO](https://arxiv.org/abs/2609.36802)）、价值扁平化（[价值扁平化](https://arxiv.org/abs/2609.18708)）、critic 对齐（[PACT](https://arxiv.org/abs/2609.26355)）、从 PMD 导出的免 critic 方法（[BPO](https://arxiv.org/abs/2609.15987)）。
- **数据技巧的统计检验**：8 种方法无一显著优于均匀采样（[DataFlex-RL](https://arxiv.org/abs/2609.06107)）；SFT 轨迹多样性影响 RL 后泛化（[多样的 SFT 轨迹](https://arxiv.org/abs/2609.33780)）。
- **训推不一致与低精度**：校准重要性采样（[CIS](https://arxiv.org/abs/2609.32444)）、全流程 FP8 的不稳定（[全流程 FP8](https://arxiv.org/abs/2609.22870)）、在线协同训练草稿模型（[草稿协同训练](https://arxiv.org/abs/2609.07108)）。
- **在策略蒸馏**：领域归一化的多教师 OPD（[领域归一化](https://arxiv.org/abs/2609.35347)）、特权信息的作用（[特权信息](https://arxiv.org/abs/2609.20612)）、RL 视角的最小二乘蒸馏（[LSPD](https://arxiv.org/abs/2609.35505)）、1% 的 token 就够了（[稀疏 OPD](https://arxiv.org/abs/2609.24432)）、反向蒸馏激发弱到强（[OPRD](https://arxiv.org/abs/2609.08798)）。
- **探索与信用**：跨模型轨迹交换（[GRAFT](https://arxiv.org/abs/2609.37868)）、最小干预（[MInTRL](https://arxiv.org/abs/2609.12419)）、难题永不放弃（[NGU](https://arxiv.org/abs/2609.13443)）、熵引导的信用（[EAPO](https://arxiv.org/abs/2609.33781)）、第一个错误（[Cliff](https://arxiv.org/abs/2609.02817)）。
- **rubric 的问题**：加法聚合的投机（[加法聚合](https://arxiv.org/abs/2609.38847)）、不可能任务（[ImpossibleRubrics](https://arxiv.org/abs/2609.16816)）；以复现为任务的环境（[RecreationWorld](https://arxiv.org/abs/2609.22000)）。

### 2026-10

- **反思在策略蒸馏**：离线学生 rollout 常常更好（[Semi-OPD](https://arxiv.org/abs/2610.11291)）、从 RL 视角看收益与崩溃（[收益与崩溃](https://arxiv.org/abs/2610.03185)）、OPD 先于 rubric RL（[OPD 热启动](https://arxiv.org/abs/2610.02781)）、分支 OPD（[BranchOPD](https://arxiv.org/abs/2610.13481)）。
- **智能体**：多 harness 训练（[Harness Scaling](https://arxiv.org/abs/2610.29458)）、harness 与策略交替优化（[Self-HPO](https://arxiv.org/abs/2610.22791)）、多尺度与跨会话信用（[多尺度](https://arxiv.org/abs/2610.35013)、[Mosaic](https://arxiv.org/abs/2610.22174)）、Muon 用于长程 RL（[Muon](https://arxiv.org/abs/2610.20256)）、自蒸馏方法的现实检验（[现实检验](https://arxiv.org/abs/2610.19789)）。
- **算法**：可扩展的 actor-critic（[S3AC](https://arxiv.org/abs/2610.35518)、[TPPO](https://arxiv.org/abs/2610.16794)）、统一 KL 正则（[KLPO](https://arxiv.org/abs/2610.08963)）、按奖励分配 token 级信用（[MRTPO](https://arxiv.org/abs/2610.21439)）、密度感知的多奖励（[密度感知](https://arxiv.org/abs/2610.00574)）。
- **低精度与规模**：NVFP4 失配修复（[TRIAGE](https://arxiv.org/abs/2610.07043)）、MoE 的 FP4 量化感知训练（[TRACE](https://arxiv.org/abs/2610.07767)）、INT4 KV 缓存（[KV4-RL](https://arxiv.org/abs/2610.14448)）、万亿参数权重同步（[NeMo-DCR](https://arxiv.org/abs/2610.08430)）；RL 扩展律的实证研究（[扩展律](https://arxiv.org/abs/2610.33420)）。
- **奖励与安全**：精确率目标缓解 rubric 投机（[FORK](https://arxiv.org/abs/2610.33512)）、细则随策略演化（[MetaRubric](https://arxiv.org/abs/2610.02824)）、可验证指令遵循中的空答投机（[CALIF](https://arxiv.org/abs/2610.19525)）、世界模型作为 GUI 验证器（[世界模型验证器](https://arxiv.org/abs/2610.11942)）。
- **其他**：思考泄漏（[思考泄漏](https://arxiv.org/abs/2610.14270)）、扩散 RL 的可扩展配方（[DiffusionART](https://arxiv.org/abs/2610.12572)）、框架收录论文方法的审查（[从论文到框架](https://arxiv.org/abs/2610.28027)）。

---

## 附录 B：论文与资源总表（908 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 核验；★ 为入门推荐；🔥 为 2026-05 及之后（最近半年），◆ 为 2025-11 及之后（最近一年），🆕 为 2024-11 及之后（最近两年）的工作。

### 经典 RL 与序列生成 RL（20）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2024-07 | [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) | Google DeepMind | Lean 中的 AlphaZero 式强化学习定理证明，IMO 2024 银牌水平 |
| 2021-06 | [Decision Transformer](https://arxiv.org/abs/2106.01345) `arXiv:2106.01345` | UC Berkeley 等 | 把 RL 写成以回报为条件的序列建模，用 Transformer 自回归生成动作 |
| 2020-05 | [实现细节很重要：PPO 与 TRPO](https://arxiv.org/abs/2005.12729) `arXiv:2005.12729` | MIT | PPO 的收益很大一部分来自代码层面的"技巧"（价值裁剪、奖励归一化等）而非裁剪目标本身 |
| 2019-11 | [MuZero](https://arxiv.org/abs/1911.08265) `arXiv:1911.08265` | DeepMind | 在学到的潜空间模型中做 MCTS 规划，不需要已知的环境规则 |
| 2019-06 | [离线批量 RL 学人类偏好（对话）](https://arxiv.org/abs/1907.00456) `arXiv:1907.00456` | MIT | 从人类交互日志中离线 RL，并用 KL 控制防止偏离先验模型 |
| 2018-02 | ★ [IMPALA（V-trace）](https://arxiv.org/abs/1802.01561) `arXiv:1802.01561` | DeepMind | 解耦的 actor-learner 架构 + 截断重要性采样修正策略滞后：异步 LLM RL 系统的直接前身 |
| 2018-01 | [SAC](https://arxiv.org/abs/1801.01290) `arXiv:1801.01290` | UC Berkeley | 最大熵目标下的离策略 actor-critic；LLM RL 中的熵正则与此同源 |
| 2017-12 | [AlphaZero](https://arxiv.org/abs/1712.01815) `arXiv:1712.01815` | DeepMind | 自我博弈 + MCTS + 策略/价值网络，从零学会国际象棋、将棋与围棋 |
| 2017-07 | ★ [PPO](https://arxiv.org/abs/1707.06347) `arXiv:1707.06347` | OpenAI | 裁剪替代目标的简单稳定策略梯度，后来也是 RLHF 的主力算法 |
| 2017-05 | [专家迭代（ExIt）](https://arxiv.org/abs/1705.08439) `arXiv:1705.08439` | UCL | "快思考"网络模仿"慢思考"树搜索的结果、再反过来引导搜索：拒绝采样微调与 STaR 的先驱 |
| 2017-04 | [策略梯度与软 Q 学习的等价](https://arxiv.org/abs/1704.06440) `arXiv:1704.06440` | OpenAI | 熵正则下策略梯度与软 Q 学习等价；KL 正则的 RLHF 是其特例（以参考策略为先验的软 RL） |
| 2016-12 | ★ [自批评序列训练（SCST）](https://arxiv.org/abs/1612.00563) `arXiv:1612.00563` | IBM | 以贪心解码的得分作为 REINFORCE 的基线，无需 critic：GRPO/ReMax 式基线的前身 |
| 2016-11 | ★ [Sequence Tutor：KL 控制](https://arxiv.org/abs/1611.02796) `arXiv:1611.02796` | Google 等 | 用 KL 惩罚把 RL 微调后的序列模型约束在预训练模型附近：RLHF 中 KL 项的源头之一 |
| 2016-07 | [序列预测的 actor-critic](https://arxiv.org/abs/1607.07086) `arXiv:1607.07086` | Mila | 为序列生成训练 critic 估计每个 token 的价值：token 级信用分配的早期尝试 |
| 2016-06 | [对话生成的深度 RL](https://arxiv.org/abs/1606.01541) `arXiv:1606.01541` | Stanford 等 | 用手工设计的长期奖励（信息量、连贯性）对对话模型做策略梯度 |
| 2016-02 | [A3C](https://arxiv.org/abs/1602.01783) `arXiv:1602.01783` | DeepMind | 异步并行的 actor-critic，多个 actor 同时采样、共享参数 |
| 2015-11 | [MIXER：序列级训练](https://arxiv.org/abs/1511.06732) `arXiv:1511.06732` | Meta | 用 REINFORCE 直接优化 BLEU 等序列级指标，缓解 teacher forcing 的暴露偏差 |
| 2015-06 | ★ [GAE](https://arxiv.org/abs/1506.02438) `arXiv:1506.02438` | UC Berkeley | 用 λ 加权的多步 TD 残差估计优势，在偏差与方差之间折中；RLHF 中的 PPO 默认使用 |
| 2015-02 | [TRPO](https://arxiv.org/abs/1502.05477) `arXiv:1502.05477` | UC Berkeley | 在 KL 信任域内最大化替代目标，保证单调改进；PPO 裁剪是它的一阶近似 |
| 1992-05 | ★ [REINFORCE](https://link.springer.com/article/10.1007/BF00992696) |  | 似然比策略梯度 ∇J=E[∇log π·(R−b)]：GRPO、RLOO 本质上都是带不同基线的 REINFORCE |

### 大模型基础（13）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2024-12 | 🆕 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) `arXiv:2412.19437` | DeepSeek | 671B MoE 基座，DeepSeek-R1 的起点 |
| 2024-07 | [Llama 3](https://arxiv.org/abs/2407.21783) `arXiv:2407.21783` | Meta | 开放权重模型族的完整技术报告，后训练使用拒绝采样 + DPO 的多轮迭代 |
| 2023-12 | [SGLang](https://arxiv.org/abs/2312.07104) `arXiv:2312.07104` | Stanford 等 | 结构化 LLM 程序的高效执行（RadixAttention 前缀复用），另一主流 rollout 引擎 |
| 2023-09 | [vLLM（PagedAttention）](https://arxiv.org/abs/2309.06180) `arXiv:2309.06180` | UC Berkeley | 把 KV 缓存分页管理，大幅提高推理吞吐：RL rollout 的主力推理引擎 |
| 2023-05 | [LIMA](https://arxiv.org/abs/2305.11206) `arXiv:2305.11206` | Meta | 1000 条精选样本即可对齐：知识主要来自预训练，对齐只是学习格式（表面对齐假说） |
| 2022-12 | [Self-Instruct](https://arxiv.org/abs/2212.10560) `arXiv:2212.10560` | UW | 让模型自己生成指令数据来做指令微调 |
| 2022-03 | [Chinchilla](https://arxiv.org/abs/2203.15556) `arXiv:2203.15556` | DeepMind | 算力最优的参数与数据配比（约每参数 20 token） |
| 2022-01 | [思维链](https://arxiv.org/abs/2201.11903) `arXiv:2201.11903` | Google | 在示例中给出中间推理步骤，大模型的多步推理能力大幅提升 |
| 2021-09 | [FLAN](https://arxiv.org/abs/2109.01652) `arXiv:2109.01652` | Google | 指令微调：在大量以指令描述的任务上微调，零样本泛化到新任务 |
| 2021-06 | [LoRA](https://arxiv.org/abs/2106.09685) `arXiv:2106.09685` | Microsoft | 冻结原权重、只训练低秩增量矩阵，参数高效微调的事实标准 |
| 2020-05 | [GPT-3](https://arxiv.org/abs/2005.14165) `arXiv:2005.14165` | OpenAI | 175B 参数的自回归语言模型，展示了上下文学习（少样本提示） |
| 2020-01 | [神经语言模型的扩展律](https://arxiv.org/abs/2001.08361) `arXiv:2001.08361` | OpenAI | 损失随参数、数据与算力呈幂律下降 |
| 2017-06 | [Transformer](https://arxiv.org/abs/1706.03762) `arXiv:1706.03762` | Google | 自注意力替代循环结构；今天所有 LLM 的骨架 |

### 训练流水线与 RL 的位置（11）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [RL 中的"思考泄漏"](https://arxiv.org/abs/2610.14270) `arXiv:2610.14270` |  | 混合思考模型在 NO-THINK 模式下做 RL 会把思考行为泄漏进来：聊天模板 token 的作用 |
| 2026-07 | 🔥 [多轮长程规划的物理学](https://arxiv.org/abs/2607.24720) `arXiv:2607.24720` |  | 从预训练到后训练理解多轮长程规划 |
| 2026-07 | 🔥 [从预训练到后训练理解推理](https://arxiv.org/abs/2607.16097) `arXiv:2607.16097` |  | 把 RL 后训练与之前的预训练放在一起研究：预训练数据如何决定 RL 能学到什么 |
| 2026-01 | ◆ [Golden Goose](https://arxiv.org/abs/2601.22975) `arXiv:2601.22975` | NVIDIA | 把不可验证的网页文本改写为填空式多选题，合成无限 RLVR 任务 |
| 2025-12 | ◆ [自回归模型中的时间抽象与分层 RL](https://arxiv.org/abs/2512.20605) `arXiv:2512.20605` |  | 在预训练模型的残差流中发现可用于分层控制的时间抽象，在其上做"内部 RL"解决稀疏奖励的长程任务 |
| 2025-12 | ★ ◆ [预训练、中期训练与 RL 的相互作用](https://arxiv.org/abs/2512.07783) `arXiv:2512.07783` |  | 合成推理任务上的受控实验：RL 只有在预训练留有余量、且任务处在能力边缘时才带来真正的新能力 |
| 2025-12 | ◆ [PretrainZero：强化主动预训练](https://arxiv.org/abs/2512.03442) `arXiv:2512.03442` |  | 在预训练语料上，模型主动挑选要遮盖的片段并预测，用 RL 把"主动学习"放进预训练 |
| 2025-09 | 🆕 [强化中期训练（RMT）](https://arxiv.org/abs/2509.24375) `arXiv:2509.24375` |  | 在预训练与后训练之间用 RL 做中期训练：动态 token 预算与课程 |
| 2025-09 | 🆕 [在预训练数据上做 RL](https://arxiv.org/abs/2509.19249) `arXiv:2509.19249` | Tencent | 直接在预训练语料上用自监督奖励做强化学习，扩展 RL 的数据来源 |
| 2025-06 | 🆕 [OctoThinker](https://arxiv.org/abs/2506.20512) `arXiv:2506.20512` | SJTU | 中期训练（mid-training）决定基座模型能否被 RL 有效提升 |
| 2025-06 | 🆕 [强化预训练](https://arxiv.org/abs/2506.08007) `arXiv:2506.08007` | Microsoft/PKU | 把下一 token 预测改写为可验证的推理任务，用 RL 做预训练 |

### RLHF 与人类反馈（14）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2024-03 | [RLHF + PPO 的 N 个实现细节](https://arxiv.org/abs/2403.17031) `arXiv:2403.17031` | Hugging Face | 逐项复现 OpenAI 早期 RLHF 的实现细节（奖励归一化、EOS 处理等） |
| 2023-10 | [Safe RLHF](https://arxiv.org/abs/2310.12773) `arXiv:2310.12773` | PKU | 把有用与无害解耦为奖励与代价，用拉格朗日法做约束 RL |
| 2023-09 | [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) `arXiv:2309.00267` | Google | 用 AI 评判替代人类偏好标注，效果相当 |
| 2023-07 | [RLHF 的开放问题与根本局限](https://arxiv.org/abs/2307.15217) `arXiv:2307.15217` | MIT 等 | 系统梳理人类反馈、奖励模型与策略优化三个环节的局限 |
| 2023-07 | [Llama 2](https://arxiv.org/abs/2307.09288) `arXiv:2307.09288` | Meta | 开放权重的对话模型与完整 RLHF 细节 |
| 2023-07 | [RLHF 的秘密（一）：PPO](https://arxiv.org/abs/2307.04964) `arXiv:2307.04964` | Fudan | 拆解 PPO 在 LLM 上不稳定的原因，给出 PPO-max 配方 |
| 2022-12 | [Constitutional AI](https://arxiv.org/abs/2212.08073) `arXiv:2212.08073` | Anthropic | 用一组原则让 AI 自我批评与修改，并以 AI 反馈（RLAIF）训练无害性 |
| 2022-09 | [Sparrow](https://arxiv.org/abs/2209.14375) `arXiv:2209.14375` | DeepMind | 按规则分解的人类判断 + 针对性的奖励模型，对话智能体对齐 |
| 2022-04 | [Anthropic HH-RLHF](https://arxiv.org/abs/2204.05862) `arXiv:2204.05862` | Anthropic | 有用与无害的 RLHF 助手：在线迭代、奖励模型规模、有用与无害的张力，公开偏好数据集 |
| 2022-03 | ★ [InstructGPT](https://arxiv.org/abs/2203.02155) `arXiv:2203.02155` | OpenAI | SFT → 奖励模型 → PPO 三阶段，1.3B 对齐模型胜过 175B GPT-3，ChatGPT 的方法基础 |
| 2021-12 | [WebGPT](https://arxiv.org/abs/2112.09332) `arXiv:2112.09332` | OpenAI | 浏览器辅助问答：模仿学习 + 奖励模型 + 拒绝采样（best-of-n），早期的智能体 + RLHF |
| 2020-09 | [学习摘要（RLHF）](https://arxiv.org/abs/2009.01325) `arXiv:2009.01325` | OpenAI | 人类反馈强化学习用于语言模型摘要，奖励模型 + PPO 的完整流程 |
| 2019-09 | ★ [从人类偏好微调语言模型](https://arxiv.org/abs/1909.08593) `arXiv:1909.08593` | OpenAI | 首次把"偏好 → 奖励模型 → PPO + KL"用于预训练语言模型（风格续写、摘要） |
| 2017-06 | ★ [从人类偏好学习](https://arxiv.org/abs/1706.03741) `arXiv:1706.03741` | OpenAI/DeepMind | 用人类对轨迹片段的比较训练奖励模型再做 RL，RLHF 的起点 |

### 奖励模型与评判（26）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🔥 [重访奖励优化的扩展律](https://arxiv.org/abs/2609.38526) `arXiv:2609.38526` |  | 奖励模型过优化的扩展律 |
| 2026-09 | 🔥 [扩散奖励模型（DRM）](https://arxiv.org/abs/2609.33803) `arXiv:2609.33803` |  | 把奖励建模改写为条件密度估计 p(r\|x,y)，表达评分的不确定与多峰 |
| 2026-07 | 🔥 [OSReward](https://arxiv.org/abs/2607.28609) `arXiv:2607.28609` |  | 评测 VLM 评判模型对计算机使用轨迹的打分能力，跨平台的奖励模型基准 |
| 2026-06 | 🔥 [离散化奖励模型](https://arxiv.org/abs/2606.21795) `arXiv:2606.21795` |  | 许多奖励模型过于敏感，对同样好的回答给出不同分数；离散化反而更利于 RL |
| 2026-06 | 🔥 [Z-Reward：把推理内化为分数分布](https://arxiv.org/abs/2606.09076) `arXiv:2606.09076` |  | 重推理的教师评判输出分数分布，蒸馏给轻量学生奖励模型，部署时不必生成长理由 |
| 2026-05 | 🔥 [神经奖励模型如何学习特征](https://arxiv.org/abs/2605.24749) `arXiv:2605.24749` |  | 单指标分析：奖励模型学到的特征如何服务策略优化 |
| 2026-05 | 🔥 [RewardHarness：自演化的智能体式奖励](https://arxiv.org/abs/2605.08703) `arXiv:2605.08703` |  | 不训练权重：奖励建模改为演化评判智能体的上下文（工具与指南），只用极少偏好数据 |
| 2026-03 | ◆ [Mix-GRM：广度与深度兼顾的生成式奖励模型](https://arxiv.org/abs/2603.01571) `arXiv:2603.01571` |  | 生成式奖励模型不只要"想得长"：把评判理由拆成广度与深度两类 CoT 再训练 |
| 2026-01 | ◆ [智能体推理奖励模型（Agent-RRM）](https://arxiv.org/abs/2601.22154) `arXiv:2601.22154` |  | 对智能体轨迹给出推理过程、批评与分数的多维反馈，用于智能体 RL |
| 2025-12 | ◆ [ARM-Thinker](https://arxiv.org/abs/2512.05111) `arXiv:2512.05111` |  | 会调用工具（裁剪、文档检索）取证再判分的智能体式多模态奖励模型 |
| 2025-10 | 🆕 [Critique-RL](https://arxiv.org/abs/2510.24320) `arXiv:2510.24320` |  | 两阶段 RL 训练评判模型：先用规则奖励提升判别力，再用"修改后是否变好"提升批评的有用性 |
| 2025-08 | 🆕 [LLaVA-Critic-R1](https://arxiv.org/abs/2509.00676) `arXiv:2509.00676` |  | 把偏好标注的评判数据改写成可验证信号，直接对基座做 RL：评判模型本身也是很强的策略模型 |
| 2025-08 | 🆕 [Cooper：策略与奖励模型协同优化](https://arxiv.org/abs/2508.05613) `arXiv:2508.05613` |  | 训练中用高置信正样本与模型自生成负样本持续更新奖励模型，缓解奖励投机 |
| 2025-07 | 🆕 [逆强化学习遇见大模型后训练](https://arxiv.org/abs/2507.13158) `arXiv:2507.13158` |  | 从 IRL 视角综述奖励建模：RLHF 的奖励模型就是从示范与偏好中反推奖励 |
| 2025-07 | 🆕 [POLAR：策略判别式奖励模型](https://arxiv.org/abs/2507.05197) `arXiv:2507.05197` | Shanghai AI Lab | 奖励模型预训练成"判断两条轨迹是否来自同一策略"，再用少量偏好数据微调，成为可扩展的通用奖励模型 |
| 2025-07 | 🆕 [Skywork-Reward-V2](https://arxiv.org/abs/2507.01352) `arXiv:2507.01352` | Skywork | 人机协同筛选 4000 万偏好对，训练开放奖励模型 |
| 2025-05 | 🆕 [RM-R1](https://arxiv.org/abs/2505.02387) `arXiv:2505.02387` | UIUC | 把奖励建模当作推理任务，先生成评分标准再判断 |
| 2025-04 | 🆕 [生成式奖励模型的推理时扩展](https://arxiv.org/abs/2504.02495) `arXiv:2504.02495` | DeepSeek/Tsinghua | 让奖励模型先生成原则与批评再打分（SPCT），奖励模型也可随推理算力扩展 |
| 2024-08 | ★ [生成式验证器（GenRM）](https://arxiv.org/abs/2408.15240) `arXiv:2408.15240` | Google DeepMind | 把验证当作下一 token 预测：先生成 CoT 再输出 Yes/No，可随推理计算扩展 |
| 2024-03 | [RewardBench](https://arxiv.org/abs/2403.13787) `arXiv:2403.13787` | AI2 | 奖励模型的标准评测：对话、安全、推理等类别的成对选择准确率 |
| 2024-01 | [WARM：权重平均的奖励模型](https://arxiv.org/abs/2401.12187) `arXiv:2401.12187` | Google DeepMind | 把多个微调奖励模型的权重平均，比集成更省、更鲁棒 |
| 2024-01 | [自奖励语言模型](https://arxiv.org/abs/2401.10020) `arXiv:2401.10020` | Meta | 模型自己当评委给自己生成的回答打分，迭代 DPO |
| 2023-10 | [Eureka](https://arxiv.org/abs/2310.12931) `arXiv:2310.12931` | NVIDIA | LLM 写奖励代码并迭代改进，达到人类专家水平 |
| 2023-10 | [奖励模型集成缓解过度优化](https://arxiv.org/abs/2310.02743) `arXiv:2310.02743` |  | 用多个奖励模型的保守聚合（最差/不确定性惩罚）抵抗奖励投机 |
| 2023-06 | [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) `arXiv:2306.05685` | UC Berkeley 等 | 用强模型充当评委（MT-Bench、Chatbot Arena），分析位置与冗长偏置 |
| 2022-10 | ★ [奖励模型过度优化](https://arxiv.org/abs/2210.10760) `arXiv:2210.10760` | OpenAI | 对代理奖励优化越多，真实奖励先升后降，且呈可预测的规律 |

### 可验证奖励与验证器（26）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [CALIF：可验证指令遵循中的奖励投机](https://arxiv.org/abs/2610.19525) `arXiv:2610.19525` |  | 前沿模型能用几乎没有实质内容的回答拿满分：在可验证约束之外补充内容检查 |
| 2026-10 | 🔥 [世界模型作为 GUI 智能体的验证器](https://arxiv.org/abs/2610.11942) `arXiv:2610.11942` |  | 在表征空间预测界面转移，用于判断动作是否导致了正确的状态变化：世界模型的第三种用途 |
| 2026-09 | 🔥 [GenV：自动形式化的生成式奖励](https://arxiv.org/abs/2609.11085) `arXiv:2609.11085` |  | 把离线 Z3 等价判定蒸馏成无参考的连续分数，作为自动形式化的奖励 |
| 2026-07 | 🔥 [LLM-as-a-Verifier](https://arxiv.org/abs/2607.05391) `arXiv:2607.05391` |  | 把验证能力作为新的扩展轴：通用的验证框架，提升 Best-of-N 与 RL 奖励质量 |
| 2026-06 | 🔥 [Dockerless：免执行的补丁验证器](https://arxiv.org/abs/2606.28436) `arXiv:2606.28436` |  | 不搭建运行环境，由智能体阅读代码判断补丁是否正确，作为 SWE RL 的廉价奖励 |
| 2026-06 | 🔥 [MaxProof](https://arxiv.org/abs/2606.13473) `arXiv:2606.13473` | MiniMax | 生成式验证器 RL + 群体级测试时扩展的数学证明 |
| 2026-06 | 🔥 [Reasoning Arena](https://arxiv.org/abs/2606.09380) `arXiv:2606.09380` |  | 组内奖励全同的样本不丢弃，转交评判系统做轨迹锦标赛，补充可验证奖励的盲区 |
| 2026-04 | ◆ [在物理模拟器上做 RL 解物理奥赛](https://arxiv.org/abs/2604.11805) `arXiv:2604.11805` |  | 用物理模拟器作为可验证奖励来源训练物理推理 |
| 2026-01 | ◆ [智能体式评分细则：SWE 的上下文验证器](https://arxiv.org/abs/2601.04171) `arXiv:2601.04171` |  | 让智能体探索代码库后写出针对该问题的评分细则，作为无需执行的验证信号 |
| 2025-12 | ◆ [Seed-Prover 1.5](https://arxiv.org/abs/2512.17260) `arXiv:2512.17260` | ByteDance Seed | 从经验中学习的大学级形式化定理证明 |
| 2025-11 | ★ ◆ [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) `arXiv:2511.22570` | DeepSeek | 训练证明验证器并以之为奖励，让模型自我验证地生成定理证明，IMO 2025 金牌级 |
| 2025-10 | 🆕 [FAPO：识别"有缺陷的正确"](https://arxiv.org/abs/2510.22543) `arXiv:2510.22543` |  | 答案对但过程有错的 rollout 早期是捷径、后期是隐患：用生成式奖励模型定位错误并给无参数惩罚 |
| 2025-10 | 🆕 [LaSeR：末 token 自奖励](https://arxiv.org/abs/2510.14943) `arXiv:2510.14943` |  | 用策略模型最后一个 token 的对数概率比作为自验证分数，与 RLVR 一起训练，几乎零额外开销得到验证能力 |
| 2025-10 | 🆕 [RLFR：潜空间流奖励](https://arxiv.org/abs/2510.10201) `arXiv:2510.10201` |  | 用离策略高质量数据在隐状态上拟合流场，以速度偏差作为奖励信号 |
| 2025-10 | 🆕 [HERO：混合奖励](https://arxiv.org/abs/2510.07242) `arXiv:2510.07242` |  | 0/1 验证器信号太稀疏：在验证器分组内用奖励模型分数做分层归一化，稀疏与稠密奖励结合 |
| 2025-10 | 🆕 [Aristotle](https://arxiv.org/abs/2510.01346) `arXiv:2510.01346` | Harmonic | 形式化验证的 IMO 2025 金牌级数学系统 |
| 2025-08 | 🆕 [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613) `arXiv:2508.03613` | Princeton | 脚手架式数据合成 + 自我修正的 Lean 定理证明器 |
| 2025-08 | 🆕 [CompassVerifier](https://arxiv.org/abs/2508.03686) `arXiv:2508.03686` | Shanghai AI Lab | 轻量级答案验证模型与 VerifierBench：处理多答案、公式等价与无效回答，用于评测与结果奖励 |
| 2025-07 | 🆕 [Seed-Prover](https://arxiv.org/abs/2507.23726) `arXiv:2507.23726` | ByteDance Seed | 引理式整证推理 + 迭代修正的 Lean 证明器，形式化证明 IMO 2025 的 5/6 题 |
| 2025-06 | 🆕 [RLPR](https://arxiv.org/abs/2506.18254) `arXiv:2506.18254` |  | 免验证器：以模型生成参考答案的平均 token 概率作奖励，经去偏与方差过滤后推广到一般领域 |
| 2025-05 | 🆕 [免验证器强化通用推理（VeriFree）](https://arxiv.org/abs/2505.21493) `arXiv:2505.21493` |  | 以策略生成参考答案的似然作为奖励，直接优化，免去验证器 |
| 2025-05 | 🆕 [General-Reasoner](https://arxiv.org/abs/2505.14652) `arXiv:2505.14652` |  | 跨学科的大规模可验证数据 + 生成式答案验证器，把 RLVR 扩展到数学以外 |
| 2025-04 | 🆕 [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) `arXiv:2504.21801` | DeepSeek | 子目标分解 + RL 的 Lean 4 形式化证明模型 |
| 2025-04 | 🆕 [Kimina-Prover](https://arxiv.org/abs/2504.11354) `arXiv:2504.11354` | Moonshot/Numina | RL 驱动的形式化定理证明，推理风格接近非形式化思考 |
| 2025-03 | 🆕 [跨越奖励之桥：多领域 RLVR](https://arxiv.org/abs/2503.23829) `arXiv:2503.23829` | Tencent | 医学、心理、经济等自由形式答案上训练软性的生成式验证奖励 |
| 2024-11 | ★ 🆕 [Tülu 3](https://arxiv.org/abs/2411.15124) `arXiv:2411.15124` | Ai2 | 完全开放的后训练配方：SFT + DPO + 可验证奖励的 RL（RLVR 一词由此流行） |

### 评分细则与开放式任务（22）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [FORK：用精确率目标缓解 rubric 奖励投机](https://arxiv.org/abs/2610.33512) `arXiv:2610.33512` |  | 只按召回给分会诱导堆砌内容：要求回答中每个单元既必要又正确 |
| 2026-10 | 🔥 [HarnessRL：随策略演化的训练 harness](https://arxiv.org/abs/2610.26426) `arXiv:2610.26426` |  | 训练时的 harness 通过任务采样、rollout 引导与 rubric 标准三个接口与策略共同进化 |
| 2026-10 | 🔥 [MetaRubric：学习如何奖励](https://arxiv.org/abs/2610.02824) `arXiv:2610.02824` |  | 交替进行证据感知的策略优化与基于回答的 rubric 自适应修订 |
| 2026-09 | 🔥 [分数更高、答案更差：rubric 加法聚合的投机](https://arxiv.org/abs/2609.38847) `arXiv:2609.38847` |  | rubric 各项分数简单相加是投机的弱点：改用协议层面的聚合 |
| 2026-09 | 🔥 [ImpossibleRubrics](https://arxiv.org/abs/2609.16816) `arXiv:2609.16816` |  | 169 个不可能任务检验生成的 rubric 是否会奖励编造：rubric 作为奖励的压力测试 |
| 2026-07 | 🔥 [CoRT：反事实回放的 token 级 rubric 信用](https://arxiv.org/abs/2607.25659) `arXiv:2607.25659` |  | 把 rubric 奖励分配到 token：反事实地替换片段再打分，估计每段的贡献 |
| 2026-07 | 🔥 [LLM-as-a-Coach：不可验证任务的经验学习](https://arxiv.org/abs/2607.18110) `arXiv:2607.18110` |  | 把评判模型改成教练：给出文字反馈，策略据此改写回答并内化 |
| 2026-07 | 🔥 [LLM-as-a-Tutor](https://arxiv.org/abs/2607.04412) `arXiv:2607.04412` |  | 评判模型同时当导师：成对比较 rollout 后改写提示，让任务难度贴合当前策略 |
| 2026-06 | 🔥 [评分细则 RL 中的奖励投机](https://arxiv.org/abs/2606.04923) `arXiv:2606.04923` |  | 复现、分析并检测基于 rubric 的 RL 中对评审模型偏好的投机 |
| 2026-05 | 🔥 [RubricEM：评分细则引导的元 RL](https://arxiv.org/abs/2605.10899) `arXiv:2605.10899` |  | 深度研究任务：按阶段做信用分配，并用基于反思的元 RL 复用经验 |
| 2026-02 | ◆ [Rubric-ARM：交替 RL 训练细则生成器与评判](https://arxiv.org/abs/2602.01511) `arXiv:2602.01511` |  | 从偏好反馈出发，交替地用 RL 优化评分细则生成器与评判模型 |
| 2025-12 | ◆ [用评分细则奖励训练 AI 合作科学家](https://arxiv.org/abs/2512.23707) `arXiv:2512.23707` | Meta 等 | 从论文中自动抽取研究目标与评分细则，用自评分做 RL 训练研究规划能力，专家偏好提升 |
| 2025-11 | ◆ [AdvancedIF](https://arxiv.org/abs/2511.10507) `arXiv:2511.10507` | Meta | 专家撰写评分细则的复杂指令遵循基准，并用 rubric 奖励 RL 提升指令遵循 |
| 2025-10 | 🆕 [OpenRubrics](https://arxiv.org/abs/2510.07743) `arXiv:2510.07743` |  | 大规模（提示, 评分细则）数据，区分硬约束与原则，训练细则生成与基于细则的奖励模型 |
| 2025-09 | 🆕 [逆向工程推理](https://arxiv.org/abs/2509.06160) `arXiv:2509.06160` |  | 从好答案反推推理过程，用于开放式生成 |
| 2025-08 | 🆕 [RuscaRL：评分细则脚手架](https://arxiv.org/abs/2508.16949) `arXiv:2508.16949` |  | 把 rubric 当作生成时的提示脚手架引导探索、训练时的奖励，并逐步撤掉脚手架 |
| 2025-08 | 🆕 [以评分细则为锚的 RL](https://arxiv.org/abs/2508.12790) `arXiv:2508.12790` |  | 上万条 rubric 做奖励，5K 样本即提升开放式任务，并减少"AI 腔" |
| 2025-08 | 🆕 [Sotopia-RL](https://arxiv.org/abs/2508.03905) `arXiv:2508.03905` |  | 社交互动中把回合级反馈细化为话语级、多维度奖励，缓解部分可观测下的信用分配 |
| 2025-07 | 🆕 [检查表优于奖励模型](https://arxiv.org/abs/2507.18624) `arXiv:2507.18624` | CMU 等 | 把指令拆成逐项的检查表，由评判模型逐项打分，作为 RL 奖励优于奖励模型 |
| 2025-07 | ★ 🆕 [评分细则即奖励（RaR）](https://arxiv.org/abs/2507.17746) `arXiv:2507.17746` | Scale AI | 把专家撰写的评分细则交给 LLM 评判打分，作为医学、科学等不可验证领域的奖励 |
| 2025-07 | 🆕 [RLVER：可验证的情感奖励](https://arxiv.org/abs/2507.03112) `arXiv:2507.03112` | Tencent | 用模拟用户的情绪分数作可验证奖励，端到端 RL 训练共情对话 |
| 2025-06 | 🆕 [LongWriter-Zero](https://arxiv.org/abs/2506.18841) `arXiv:2506.18841` |  | 不用任何标注或合成数据，从基座出发用 RL（奖励模型打分长度、写作质量与格式）学会超长文本写作 |

### 策略优化算法（41）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [S3AC：可扩展的 actor-critic](https://arxiv.org/abs/2610.35518) `arXiv:2610.35518` |  | 用题目级基线估计难度、选择性地训练 critic，让 actor-critic 在 LLM 推理上稳定可扩展 |
| 2026-10 | 🔥 [TPPO](https://arxiv.org/abs/2610.16794) `arXiv:2610.16794` |  | 追踪价值的 PPO 用于语言模型推理 |
| 2026-09 | 🔥 [密度感知的多奖励聚合](https://arxiv.org/abs/2610.00574) `arXiv:2610.00574` |  | 多奖励 RL 中稀疏奖励的梯度能量与"有效组密度"成正比，据此重新加权 |
| 2026-09 | 🔥 [EasyPPO：稳住 critic 是关键](https://arxiv.org/abs/2609.36802) `arXiv:2609.36802` |  | LLM PPO 的不稳定很大程度来自 critic：识别两种 critic 失效模式并修复 |
| 2026-09 | 🔥 [PPO 中的价值扁平化](https://arxiv.org/abs/2609.18708) `arXiv:2609.18708` | Shanghai AI Lab | LLM 的 PPO 价值网络会"扁平化"，提出缓解方法 |
| 2026-09 | 🔥 [BPO：贝尔曼策略优化](https://arxiv.org/abs/2609.15987) `arXiv:2609.15987` |  | 从策略镜像下降推导的免 critic 方法，与原 PMD 目标有相同唯一最优解 |
| 2026-08 | 🔥 [理解进化策略做推理](https://arxiv.org/abs/2608.27351) `arXiv:2608.27351` |  | 进化策略（ES）微调 LLM 相比 GRPO 推理覆盖更广：优化行为分析 |
| 2026-08 | 🔥 [饱和感知的多奖励优势重加权](https://arxiv.org/abs/2608.16072) `arXiv:2608.16072` |  | 多奖励 GRPO 中不同奖励组合可能得到相同优势：降低已掌握目标的权重，专注尚未学会的 |
| 2026-05 | 🔥 [DVAO：多奖励 RL 的方差自适应优势](https://arxiv.org/abs/2605.25604) `arXiv:2605.25604` |  | 按各奖励的经验方差动态调整组合权重，避免低方差目标被淹没 |
| 2026-05 | 🔥 [HölderPO](https://arxiv.org/abs/2605.12058) `arXiv:2605.12058` |  | 用 Hölder 均值统一 token 级概率的聚合（算术、几何均值都是特例），p 控制稳定与效率 |
| 2026-05 | 🔥 [列表式策略优化](https://arxiv.org/abs/2605.06139) `arXiv:2605.06139` | Tencent Hunyuan | 把组式 RLVR 解释为在回答单纯形上的目标投影 |
| 2026-04 | ◆ [SPPO：序列级 PPO](https://arxiv.org/abs/2604.08865) `arXiv:2604.08865` |  | 把长推理视为单步序列级决策，用序列级价值做 PPO，兼得 PPO 的样本效率与结果奖励的稳定 |
| 2026-04 | ◆ [TPO：目标策略优化](https://arxiv.org/abs/2604.06159) `arXiv:2604.06159` |  | 把"往哪个分布更新"与"如何拟合它"分开：先构造目标策略再做监督拟合，难题上优于 PG/PPO/GRPO |
| 2026-02 | ◆ [LUSPO：长度无偏的序列级策略优化](https://arxiv.org/abs/2602.05261) `arXiv:2602.05261` |  | 分析各类损失聚合方式如何系统性地改变回答长度，提出长度无偏的序列级目标 |
| 2026-02 | ◆ [GRPO 的隐式优势对称](https://arxiv.org/abs/2602.05548) `arXiv:2602.05548` |  | 组相对优势的隐式对称性导致 GRPO 探索与难度自适应不足 |
| 2026-01 | ◆ [难题优先的 GRPO（MathForge）](https://arxiv.org/abs/2601.20614) `arXiv:2601.20614` |  | GRPO 的优势归一化隐含地压低了难题的更新幅度：按难度重加权，并配合多角度改写的难题数据 |
| 2026-01 | ◆ [组相对优势是有偏的](https://arxiv.org/abs/2601.08521) `arXiv:2601.08521` |  | 证明组内相对优势对难题系统性低估、对易题高估；用随训练演化的难度锚点自适应重加权 |
| 2026-01 | ◆ [GDPO](https://arxiv.org/abs/2601.05242) `arXiv:2601.05242` | NVIDIA | 多奖励 RL 中按奖励分别做组归一化再合并，避免信号相互淹没 |
| 2025-11 | ◆ [SofT-GRPO](https://arxiv.org/abs/2511.06411) `arXiv:2511.06411` |  | 对"软思考"（连续 token 混合）做 GRPO：用 Gumbel 重参数化注入噪声，超过离散 token RL |
| 2025-10 | 🆕 [监督式强化学习（SRL）](https://arxiv.org/abs/2510.25992) `arXiv:2510.25992` | Google | 把专家解拆成逐步"动作"，按每步与专家动作的相似度给稠密奖励：难题上 RLVR 采不到正例时仍有学习信号 |
| 2025-10 | 🆕 [SPG：夹逼策略梯度](https://arxiv.org/abs/2510.09541) `arXiv:2510.09541` |  | 扩散语言模型的似然不可算：同时用上下界构造策略梯度 |
| 2025-10 | 🆕 [Reinforce-Ada](https://arxiv.org/abs/2510.04996) `arXiv:2510.04996` |  | 组内全对/全错导致信号消失是欠采样造成的：按题目自适应追加采样直到出现正负样本 |
| 2025-09 | 🆕 [MAPO：混合优势](https://arxiv.org/abs/2509.18849) `arXiv:2509.18849` |  | 对确定性高（组内几乎全对或全错）的样本改用"百分比偏差"优势，修正标准差归一化的问题 |
| 2025-09 | 🆕 [SPO：单流策略优化](https://arxiv.org/abs/2509.13232) `arXiv:2509.13232` |  | 用持续更新的价值跟踪器（KL 自适应）替代组基线，免去每题多次采样，适合异步与智能体场景 |
| 2025-09 | 🆕 [TraceRL](https://arxiv.org/abs/2509.06949) `arXiv:2509.06949` |  | 面向扩散语言模型的轨迹感知 RL：把推理时的解码轨迹纳入后训练 |
| 2025-08 | 🆕 [PVPO：预估价值基线](https://arxiv.org/abs/2508.21104) `arXiv:2508.21104` |  | 用参考模型预先采样得到的基线作为优势锚点，并预筛数据，减少 rollout |
| 2025-08 | 🆕 [RL 推理的技巧还是陷阱](https://arxiv.org/abs/2508.08221) `arXiv:2508.08221` | Alibaba | 系统复现 RL 推理技巧，给出两项技巧的极简组合 |
| 2025-08 | 🆕 [Klear-Reasoner：GPPO](https://arxiv.org/abs/2508.07629) `arXiv:2508.07629` | Kuaishou | 保留被裁剪 token 的梯度（有界地回传），兼顾探索与负样本学习 |
| 2025-07 | 🆕 [GMPO：几何平均策略优化](https://arxiv.org/abs/2507.20673) `arXiv:2507.20673` |  | 以 token 级重要性比的几何平均替代算术平均，抑制异常值，比 GRPO 更稳 |
| 2025-07 | ★ 🆕 [GSPO](https://arxiv.org/abs/2507.18071) `arXiv:2507.18071` | Qwen | 序列级重要性比率与裁剪，稳定 MoE 模型的大规模 RL（Qwen3 采用） |
| 2025-06 | 🆕 [T-PPO：截断 PPO](https://arxiv.org/abs/2506.15050) `arXiv:2506.15050` | ByteDance | 对未完成的长回答也计算优势并更新，策略与价值解耦，提高长 CoT 训练效率 |
| 2025-05 | 🆕 [OPO：最优奖励基线](https://arxiv.org/abs/2505.23585) `arXiv:2505.23585` | Microsoft | 严格在策略 + 理论最优基线（按长度加权），稳定且不需要启发式 |
| 2025-04 | 🆕 [GVPO：组方差策略优化](https://arxiv.org/abs/2504.19599) `arXiv:2504.19599` |  | 从 KL 约束奖励最大化的闭式解出发，把它直接写成组内加权回归 |
| 2025-04 | 🆕 [VAPO](https://arxiv.org/abs/2504.05118) `arXiv:2504.05118` | ByteDance Seed | 基于价值模型的 PPO 改进，长 CoT 推理上超过无价值方法 |
| 2025-03 | 🆕 [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290) `arXiv:2503.24290` | StepFun/Tsinghua | 极简配方（原始 PPO + GAE、无 KL）即可在基座模型上扩展 R1-Zero 式训练 |
| 2025-03 | ★ 🆕 [Dr. GRPO](https://arxiv.org/abs/2503.20783) `arXiv:2503.20783` | Sea AI/NUS | 剖析 R1-Zero 式训练：GRPO 的长度与标准差归一化带来偏差，基座模型本身已有"顿悟" |
| 2025-03 | ★ 🆕 [DAPO](https://arxiv.org/abs/2503.14476) `arXiv:2503.14476` | ByteDance Seed/Tsinghua | 开源大规模 RL 系统：解耦裁剪、动态采样、token 级损失、超长惩罚 |
| 2025-01 | 🆕 [REINFORCE++](https://arxiv.org/abs/2501.03262) `arXiv:2501.03262` |  | 用全局批次归一化优势的简化 REINFORCE，免价值网络且比 GRPO 更稳 |
| 2024-02 | ★ [RLOO：回到 REINFORCE](https://arxiv.org/abs/2402.14740) `arXiv:2402.14740` | Cohere | RLHF 中 PPO 的许多部件不必要：多样本留一基线的 REINFORCE（RLOO）更简单、更省、效果更好 |
| 2024-02 | ★ [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300) `arXiv:2402.03300` | DeepSeek | 组相对策略优化：同题采样一组回答，以组内均值与标准差归一化作优势，省去价值网络 |
| 2023-10 | [ReMax](https://arxiv.org/abs/2310.10505) `arXiv:2310.10505` |  | 以贪心解码回答的奖励作基线的 REINFORCE，省去价值网络与一半显存 |

### 稳定性、离策略与训推一致（26）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [长程 RL 需要 Muon](https://arxiv.org/abs/2610.20256) `arXiv:2610.20256` |  | 长程智能体 RL 中 Muon 优化器比 Adam 更稳 |
| 2026-10 | 🔥 [KLPO：论 KL 正则的策略优化](https://arxiv.org/abs/2610.08963) `arXiv:2610.08963` |  | 把 KL 正则锚定在采样策略上，统一并修正各种 KL 估计与放置方式 |
| 2026-09 | 🔥 [ReSPO：离策略学习中的梯度饥饿](https://arxiv.org/abs/2609.35433) `arXiv:2609.35433` |  | 裁剪会压制低重要性权重尾部的正样本：符号相关的梯度饥饿及其修复 |
| 2026-09 | 🔥 [训推不一致从哪里来、如何修正](https://arxiv.org/abs/2609.32444) `arXiv:2609.32444` |  | 定位训推不一致的来源，提出校准重要性采样（CIS），把误差的无界二阶矩换成有界量 |
| 2026-08 | 🔥 [ERPO：环境正则](https://arxiv.org/abs/2608.23311) `arXiv:2608.23311` |  | 用控制查询分布漂移的"环境正则"替代策略 KL 正则，摆脱稳定与探索的两难 |
| 2026-07 | 🔥 [SAT：陈旧度自适应信任域](https://arxiv.org/abs/2607.18722) `arXiv:2607.18722` |  | 异步 RL 中以采样对数比作为陈旧度代理，只对高失配尾部收紧信任域 |
| 2026-07 | 🔥 [Muon 何时帮助智能体 RL](https://arxiv.org/abs/2607.16169) `arXiv:2607.16169` |  | Muon 在智能体强化学习中的适用条件 |
| 2026-06 | 🔥 [DRPO：重新思考散度正则](https://arxiv.org/abs/2606.09821) `arXiv:2606.09821` |  | 用平滑的优势加权二次正则替代裁剪的硬掩码，约束策略偏移 |
| 2026-05 | 🔥 [异步智能体 RL 中缺失的旧 logits](https://arxiv.org/abs/2605.12070) `arXiv:2605.12070` |  | 延迟更新与部分 rollout 常丢失训练侧旧 logits，导致重要性比语义错位；给出三种精确修复 |
| 2026-03 | ◆ [BandPO：概率感知的裁剪边界](https://arxiv.org/abs/2603.04918) `arXiv:2603.04918` |  | 固定裁剪边界严重限制低概率动作的上调幅度：按概率设定边界，桥接信任域与比率裁剪 |
| 2026-02 | ◆ [VESPO：变分序列级软策略优化](https://arxiv.org/abs/2602.10693) `arXiv:2602.10693` |  | 从变分推断推导序列级重要性权重的软约束，在严重离策略（高陈旧度）下仍稳定 |
| 2026-02 | ◆ [重新思考 LLM RL 的信任域（DPPO）](https://arxiv.org/abs/2602.04879) `arXiv:2602.04879` |  | 对比率裁剪在大词表下既过严又过松：改用直接估计的策略散度（如总变差）做约束 |
| 2025-12 | ◆ [熵比裁剪](https://arxiv.org/abs/2512.05591) `arXiv:2512.05591` |  | 以新旧策略熵之比作为全局软约束，弥补 PPO 只约束已采样 token 的不足 |
| 2025-12 | ◆ [搜索智能体中的 GRPO 崩溃](https://arxiv.org/abs/2512.04220) `arXiv:2512.04220` |  | "懒惰似然位移"：正确与错误回答的似然一起下降并形成死亡螺旋；用似然保持正则修复 |
| 2025-12 | ★ ◆ [稳定 LLM 强化学习：形式化与实践](https://arxiv.org/abs/2512.01374) `arXiv:2512.01374` | Qwen | 把 token 级目标看作序列级目标的一阶近似，据此解释重要性采样修正、裁剪与路由重放 |
| 2025-11 | ◆ [SAPO：软自适应策略优化](https://arxiv.org/abs/2511.20347) `arXiv:2511.20347` | Qwen | 用温度控制的平滑门替代硬裁剪，按 token 离策略程度连续衰减更新；Qwen3-VL 训练采用 |
| 2025-10 | ★ 🆕 [用 FP16 消除训推不一致](https://arxiv.org/abs/2510.26788) `arXiv:2510.26788` |  | 训推不一致主要来自 BF16 的舍入误差：改用 FP16 即可显著稳定 RL |
| 2025-10 | 🆕 [BAPO](https://arxiv.org/abs/2510.18927) `arXiv:2510.18927` | Nex AGI | 平衡正负样本的裁剪，稳定离策略 RL |
| 2025-10 | 🆕 [对齐训练与推理的路由（R3）](https://arxiv.org/abs/2510.11370) `arXiv:2510.11370` |  | MoE 模型 RL 中训练与推理引擎的专家路由不一致：回放推理时的路由 |
| 2025-10 | 🆕 [ASPO：非对称重要性比](https://arxiv.org/abs/2510.06062) `arXiv:2510.06062` |  | 结果监督下正样本 token 的重要性比会错误地放大高概率 token：对正样本翻转比值 |
| 2025-10 | 🆕 [M2PO：陈旧数据的离策略 RL](https://arxiv.org/abs/2510.01161) `arXiv:2510.01161` |  | 约束重要性权重的二阶矩、只压制极端值：即使数据落后 256 次更新也能稳定训练 |
| 2025-09 | 🆕 [消除 LLM 推理的不确定性](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Thinking Machines | 推理结果随批大小变化来自缺乏批不变性的内核；批不变内核可实现真正在策略的 RL |
| 2025-08 | ★ 🆕 [训推不一致与截断重要性采样（TIS）](https://fengyao.notion.site/off-policy-rl) |  | vLLM 采样与 FSDP 训练算出的概率并不相同，"在策略"RL 其实是离策略的；用截断重要性采样修正 |
| 2025-06 | 🆕 [非对称 REINFORCE](https://arxiv.org/abs/2506.20520) `arXiv:2506.20520` | Meta | 离策略情形下基线应偏向保守：更多地强调正样本、谨慎地压低负样本 |
| 2025-06 | 🆕 [KL 散度梯度估计的几个陷阱](https://arxiv.org/abs/2506.09477) `arXiv:2506.09477` |  | 把 k3 等 KL 估计量直接放进损失得到的梯度并不是 KL 的梯度：常见实现的偏差 |
| 2025-03 | 🆕 [TOPR：渐缩离策略 REINFORCE](https://arxiv.org/abs/2503.14286) `arXiv:2503.14286` |  | 对正负样本使用不同的重要性截断，稳定且高效的离策略 REINFORCE |

### 直接偏好优化与离线方法（33）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-04 | ◆ [GFT：从模仿到奖励微调](https://arxiv.org/abs/2604.14258) `arXiv:2604.14258` |  | 指出 SFT 的内在局限，用组内优势与动态系数把示范学习改写成奖励微调 |
| 2026-03 | ◆ [ReGFT：参考解引导微调](https://arxiv.org/abs/2603.01223) `arXiv:2603.01223` |  | 用人写参考解合成模型自己风格的正确轨迹，在 RL 前先学会难题 |
| 2026-02 | ◆ [好的 SFT 为 RL 做准备（PEAR）](https://arxiv.org/abs/2602.01058) `arXiv:2602.01058` |  | 更强的 SFT 检查点在相同 RL 后反而可能更差：按策略评估思想对 SFT 样本重加权，让 SFT 服务于后续 RL |
| 2026-01 | ◆ [GANPO：潜空间对抗正则](https://arxiv.org/abs/2601.22083) `arXiv:2601.22083` |  | 离线偏好优化中用隐状态而非 token 层面的散度约束策略，更稳健 |
| 2025-12 | ◆ [语言模型其实是单指标模型](https://arxiv.org/abs/2512.21917) `arXiv:2512.21917` |  | 半参数偏好优化：把奖励链接函数当未知，统一 DPO 类方法 |
| 2025-08 | 🆕 [DuPO](https://arxiv.org/abs/2508.14460) `arXiv:2508.14460` | ByteDance Seed | 利用任务对偶性构造自监督奖励的偏好优化 |
| 2025-08 | 🆕 [DFT：SFT 的泛化](https://arxiv.org/abs/2508.05629) `arXiv:2508.05629` |  | SFT 等价于奖励稀疏且病态的策略梯度，按 token 概率重新缩放即可显著提升泛化 |
| 2025-06 | 🆕 [SRFT：单阶段 SFT + RL](https://arxiv.org/abs/2506.19767) `arXiv:2506.19767` |  | 用熵感知的权重在同一阶段混合示范的监督损失与自身 rollout 的 RL 损失 |
| 2025-04 | 🆕 [LUFFY：离策略指导](https://arxiv.org/abs/2504.14945) `arXiv:2504.14945` | Shanghai AI Lab | 在 RLVR 中混入强模型的离策略推理轨迹，突破自身能力上限 |
| 2025-02 | 🆕 [直接对齐算法差别不大](https://arxiv.org/abs/2502.01237) `arXiv:2502.01237` | T-Tech | DPO、SimPO 等直接对齐算法的差异主要在于排序 vs 点式目标 |
| 2024-06 | [拆解 DPO 与 PPO](https://arxiv.org/abs/2406.09279) `arXiv:2406.09279` | AI2 | 控制变量：偏好数据质量最重要，其次是算法（PPO 略优于 DPO），再次是奖励模型与提示 |
| 2024-05 | [XPO：探索式偏好优化](https://arxiv.org/abs/2405.21046) `arXiv:2405.21046` | Microsoft | 在线 DPO 加一项乐观探索奖励，有样本效率保证 |
| 2024-05 | [SimPO](https://arxiv.org/abs/2405.14734) `arXiv:2405.14734` | Princeton | 以长度归一化的平均对数概率为隐式奖励，不需参考模型 |
| 2024-05 | [RLHF Workflow](https://arxiv.org/abs/2405.07863) `arXiv:2405.07863` | Salesforce 等 | 从奖励建模到在线迭代 DPO 的完整开源流程 |
| 2024-05 | [SPPO：自我博弈偏好优化](https://arxiv.org/abs/2405.00675) `arXiv:2405.00675` | UCLA | 以两人常和博弈形式化对齐，迭代逼近纳什均衡 |
| 2024-04 | [偏好微调应利用次优的在策略数据](https://arxiv.org/abs/2404.14367) `arXiv:2404.14367` | CMU | 在策略采样与负梯度（把概率质量从差回答上移走）是偏好微调有效的关键 |
| 2024-04 | [从 r 到 Q*：语言模型其实是 Q 函数](https://arxiv.org/abs/2404.12358) `arXiv:2404.12358` | Stanford | 在 token 级 MDP 中，DPO 隐式学到的是最优软 Q 函数，可用于信用分配与搜索 |
| 2024-04 | [DPO 比 PPO 更好吗](https://arxiv.org/abs/2404.10719) `arXiv:2404.10719` | Tsinghua | DPO 对分布偏移敏感，充分调优的 PPO 在对话与代码上仍更强 |
| 2024-04 | [直接纳什优化（DNO）](https://arxiv.org/abs/2404.03715) `arXiv:2404.03715` | Microsoft | 把纳什学习落成可扩展的迭代对比学习算法 |
| 2024-03 | [ORPO](https://arxiv.org/abs/2403.07691) `arXiv:2403.07691` | KAIST | 把赔率比惩罚并入 SFT 损失，一步完成指令微调与偏好对齐，无参考模型 |
| 2024-02 | [在线 AI 反馈（OAIF）](https://arxiv.org/abs/2402.04792) `arXiv:2402.04792` | Google DeepMind | DPO 改成在线：每步由 LLM 评判对当前策略的两个回答标注偏好 |
| 2024-02 | [KTO](https://arxiv.org/abs/2402.01306) `arXiv:2402.01306` | Contextual AI/Stanford | 前景理论启发：只需"好/坏"二元信号的对齐 |
| 2024-01 | [SPIN：自我博弈微调](https://arxiv.org/abs/2401.01335) `arXiv:2401.01335` | UCLA | 把人类示范当正例、模型自己的生成当负例，迭代地对抗训练 |
| 2023-12 | [KL 约束下的迭代偏好学习](https://arxiv.org/abs/2312.11456) `arXiv:2312.11456` |  | 理论与实践：在线/迭代 DPO 比离线 DPO 有更好的保证与效果 |
| 2023-12 | [ReST-EM：超越人类数据](https://arxiv.org/abs/2312.06585) `arXiv:2312.06585` | Google DeepMind | 在数学与代码上用二值奖励做期望最大化式的自训练，优于只用人类数据 |
| 2023-12 | [纳什学习（NLHF）](https://arxiv.org/abs/2312.00886) `arXiv:2312.00886` | Google DeepMind | 不学标量奖励，直接求偏好模型下的纳什均衡策略，能处理不可传递的偏好 |
| 2023-10 | [IPO / ΨPO](https://arxiv.org/abs/2310.12036) `arXiv:2310.12036` | Google DeepMind | 统一偏好优化目标并指出 DPO 在确定性偏好下会过拟合 |
| 2023-08 | [ReST：强化自训练](https://arxiv.org/abs/2308.08998) `arXiv:2308.08998` | Google DeepMind | 增长（采样）—改进（按奖励过滤并离线训练）交替进行的离线 RL |
| 2023-05 | ★ [DPO](https://arxiv.org/abs/2305.18290) `arXiv:2305.18290` | Stanford | 把 RLHF 目标的最优解闭式代入，直接用偏好对做分类式训练，免奖励模型与 RL |
| 2023-05 | [SLiC-HF](https://arxiv.org/abs/2305.10425) `arXiv:2305.10425` | Google | 序列似然校准：用偏好对上的铰链损失对齐，DPO 的同期方法 |
| 2023-04 | [RAFT：奖励排序微调](https://arxiv.org/abs/2304.06767) `arXiv:2304.06767` |  | 采样多个回答、按奖励选最好的再 SFT：最朴素的"拒绝采样微调" |
| 2023-04 | [RRHF](https://arxiv.org/abs/2304.05302) `arXiv:2304.05302` | Alibaba | 用排序损失对齐多个回答的似然与奖励排序，免 PPO |
| 2022-03 | ★ [STaR](https://arxiv.org/abs/2203.14465) `arXiv:2203.14465` | Stanford/Google | 用模型自己生成且答案正确的推理链迭代微调：自举推理 |

### 在策略蒸馏与自蒸馏（72）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [在策略自训练：零数据自我改进](https://arxiv.org/abs/2610.32770) `arXiv:2610.32770` |  | 用完全合成的特权信息做自蒸馏，跨领域提升且不损害未训练领域 |
| 2026-10 | 🔥 [树感知后验蒸馏](https://arxiv.org/abs/2610.23181) `arXiv:2610.23181` |  | 长程智能体 RL 中的树感知后验蒸馏 |
| 2026-10 | 🔥 [Waypoint-OPD](https://arxiv.org/abs/2610.21601) `arXiv:2610.21601` |  | 用门控在策略蒸馏迁移 RL 引起的策略偏移 |
| 2026-10 | 🔥 [智能体 RL 自蒸馏的现实检验](https://arxiv.org/abs/2610.19789) `arXiv:2610.19789` |  | 统一条件重测各类智能体 RL 自蒸馏方法：多数增益不如宣称 |
| 2026-10 | 🔥 [FSPO：函数空间时间自蒸馏](https://arxiv.org/abs/2610.19687) `arXiv:2610.19687` |  | 单策略框架：特权上下文下的当前适应 + 函数空间上的时间巩固，用于推理与工具使用 |
| 2026-10 | 🔥 [BranchOPD：分支在策略蒸馏](https://arxiv.org/abs/2610.13481) `arXiv:2610.13481` |  | 在学生不确定且师生分歧大的决策点重新采样续写，集中监督信号 |
| 2026-10 | 🔥 [在策略蒸馏为何有时失败](https://arxiv.org/abs/2610.11247) `arXiv:2610.11247` |  | 学习信号消失 |
| 2026-10 | ★ 🔥 [什么时候才需要在策略蒸馏](https://arxiv.org/abs/2610.11291) `arXiv:2610.11291` |  | 只用初始学生的离线 rollout 蒸馏（Semi-OPD），在 17 组师生中常常比 OPD 更准更快 |
| 2026-10 | 🔥 [在策略蒸馏的收益与崩溃：RL 视角](https://arxiv.org/abs/2610.03185) `arXiv:2610.03185` |  | 从 RL 视角诊断 OPD 训练崩溃：屏蔽不健康回答或用 SFT 初始化即可缓解 |
| 2026-10 | 🔥 [OPD 先于 RL：rubric RL 的热启动](https://arxiv.org/abs/2610.02781) `arXiv:2610.02781` |  | 先把 rubric 作为教师的特权上下文做 token 级蒸馏，再把 rubric 当奖励做 RL |
| 2026-09 | 🔥 [UniEvo-VL：多模态在策略自蒸馏](https://arxiv.org/abs/2609.38721) `arXiv:2609.38721` |  | 多模态模型在测试时自我纠错，再把纠错后的结果通过在策略自蒸馏内化 |
| 2026-09 | 🔥 [PivotOPD](https://arxiv.org/abs/2609.40285) `arXiv:2609.40285` |  | 学会从多轮智能体的关键错误中恢复 |
| 2026-09 | 🔥 [理解离策略与在策略蒸馏](https://arxiv.org/abs/2609.38666) `arXiv:2609.38666` |  | 两者对应不同的训练目标 |
| 2026-09 | 🔥 [教师是方向而非终点](https://arxiv.org/abs/2609.36484) `arXiv:2609.36484` |  | 外推 RL 带来的表征残差，在蒸馏中超越教师 |
| 2026-09 | 🔥 [PR-OPD](https://arxiv.org/abs/2609.36642) `arXiv:2609.36642` |  | 特权表征的在策略自蒸馏，用于智能体 RL |
| 2026-09 | 🔥 [领域归一化的多教师 OPD](https://arxiv.org/abs/2609.35347) `arXiv:2609.35347` |  | 多教师 OPD 的学生往往不如只跟最好的单一专家学：按领域归一化教师信号 |
| 2026-09 | 🔥 [用稀疏交叉编码器理解 OPD](https://arxiv.org/abs/2609.35210) `arXiv:2609.35210` |  | 在策略蒸馏到底把哪些特征蒸馏进了学生的内部表征 |
| 2026-09 | 🔥 [在策略还是离策略蒸馏](https://arxiv.org/abs/2609.35259) `arXiv:2609.35259` | UC 系 | 系统比较两种蒸馏的训练动力学 |
| 2026-09 | 🔥 [LSPD：最小二乘策略蒸馏](https://arxiv.org/abs/2609.35505) `arXiv:2609.35505` |  | 从值函数 RL 视角看 OPD：引入乐观探索与离策略数据复用 |
| 2026-09 | 🔥 [同族 on-policy 蒸馏的扩展性质](https://arxiv.org/abs/2609.32722) `arXiv:2609.32722` | Zhejiang Univ | RL 训练的教师向同族学生做在策略蒸馏时的扩展规律 |
| 2026-09 | 🔥 [1% 的 token 就够了：OPD 的梯度估计](https://arxiv.org/abs/2609.24432) `arXiv:2609.24432` |  | 稀疏 OPD 只在少量 token 上给教师监督：分析其梯度噪声并降低方差 |
| 2026-09 | 🔥 [蒸馏中的长度膨胀](https://arxiv.org/abs/2609.20511) `arXiv:2609.20511` | Microsoft | 教师与学生 EOS 判断不一致导致学生回答越来越长 |
| 2026-09 | 🔥 [特权信息给在策略自蒸馏带来什么](https://arxiv.org/abs/2609.20612) `arXiv:2609.20612` |  | 拆解 OPSD 中"教师能看到答案"的作用：收益与风险从何而来 |
| 2026-09 | 🔥 [RetireOPD](https://arxiv.org/abs/2609.20784) `arXiv:2609.20784` |  | 智能体 RL 中自动"退休"的在策略蒸馏 |
| 2026-09 | 🔥 [OPRD：在策略反向蒸馏](https://arxiv.org/abs/2609.08798) `arXiv:2609.08798` |  | 强学生评估弱教师的策略（反向蒸馏）来激发弱到强泛化 |
| 2026-09 | 🔥 [在策略蒸馏 II：单样本](https://arxiv.org/abs/2609.04172) `arXiv:2609.04172` | Thinking Space | 只用一个训练样本的在策略蒸馏 |
| 2026-08 | 🔥 [OPD 真的在蒸馏吗](https://arxiv.org/abs/2608.31046) `arXiv:2608.31046` |  | 从带噪教师到自我改进：在策略蒸馏的再审视 |
| 2026-08 | 🔥 [在策略自蒸馏的批判性回顾](https://arxiv.org/abs/2608.25936) `arXiv:2608.25936` |  | OPSD 的同一症状（崩溃/退化）有三个杠杆：教师上下文、散度与采样 |
| 2026-08 | 🔥 [OPDVR：在策略蒸馏 + 可验证奖励](https://arxiv.org/abs/2608.24696) `arXiv:2608.24696` |  | 无额外超参地结合 OPD 与 RLVR |
| 2026-08 | 🔥 [无监督的在策略自蒸馏（U-OPSD）](https://arxiv.org/abs/2608.06296) `arXiv:2608.06296` |  | 不要答案也不要教师：用模型多次生成之间的内部一致性构造特权上下文做自蒸馏 |
| 2026-08 | 🔥 [AgentOPSD](https://arxiv.org/abs/2608.05987) `arXiv:2608.05987` |  | 智能体 RL 的递归自蒸馏 |
| 2026-08 | 🔥 [PCSD](https://arxiv.org/abs/2608.01837) `arXiv:2608.01837` |  | 智能体 RL 自蒸馏中的持续一致性 |
| 2026-07 | 🔥 [弱到强的在策略蒸馏](https://arxiv.org/abs/2607.26246) `arXiv:2607.26246` |  | 从多个弱模型做在策略蒸馏也能提升强学生 |
| 2026-07 | 🔥 [蒸馏强化学习（Distilled RL）](https://arxiv.org/abs/2607.17247) `arXiv:2607.17247` |  | 把教师监督并入 RL 目标，选择性地迁移新知识、避免不必要的模仿 |
| 2026-07 | 🔥 [在策略增量蒸馏](https://arxiv.org/abs/2607.15161) `arXiv:2607.15161` |  | 不直接模仿教师分布，而以"教师相对其 RL 前版本的变化"作为蒸馏奖励 |
| 2026-07 | 🔥 [SEED](https://arxiv.org/abs/2607.14777) `arXiv:2607.14777` |  | 自我演化的在策略蒸馏，用于智能体 RL |
| 2026-07 | 🔥 [揭开在策略蒸馏：角色、病理与规约](https://arxiv.org/abs/2607.13399) `arXiv:2607.13399` |  | 系统研究 OPD 的病理（如利用长度）与相应的规约手段 |
| 2026-07 | 🔥 [TurnOPD](https://arxiv.org/abs/2607.05804) `arXiv:2607.05804` |  | 让在策略蒸馏感知多轮结构，高效训练长程智能体 |
| 2026-07 | 🔥 [直接在策略蒸馏实现弱到强](https://arxiv.org/abs/2607.05394) `arXiv:2607.05394` | ByteDance/Tsinghua | 用弱教师的在策略蒸馏激发强学生 |
| 2026-07 | 🔥 [UI-MOPD](https://arxiv.org/abs/2607.04425) `arXiv:2607.04425` |  | 多平台在策略蒸馏，持续学习 GUI 智能体 |
| 2026-06 | 🔥 [MOPD：多教师在策略蒸馏](https://arxiv.org/abs/2606.30406) `arXiv:2606.30406` |  | 先按领域分别做 RL 得到专家，再用多教师 OPD 把能力整合进一个模型 |
| 2026-06 | 🔥 [DOPD：双向在策略蒸馏](https://arxiv.org/abs/2606.30626) `arXiv:2606.30626` |  | 按优势在"带特权信息的教师"与"带特权信息的学生"之间动态路由 token 级监督 |
| 2026-06 | 🔥 [OPID](https://arxiv.org/abs/2606.26790) `arXiv:2606.26790` |  | 面向智能体 RL 的在策略技能蒸馏 |
| 2026-06 | 🔥 [AsyncOPD：在策略蒸馏能容忍多陈旧](https://arxiv.org/abs/2606.24143) `arXiv:2606.24143` |  | 系统研究异步 OPD 的陈旧度：OPD 比 RL 对陈旧数据更宽容 |
| 2026-06 | 🔥 [稠密监督、稀疏更新](https://arxiv.org/abs/2606.13657) `arXiv:2606.13657` |  | 在策略蒸馏只改动很稀疏的参数子集：OPD 更新的稀疏性与几何 |
| 2026-06 | 🔥 [在策略蒸馏的几何](https://arxiv.org/abs/2606.07082) `arXiv:2606.07082` | HKUST | 从几何视角分析在策略蒸馏 |
| 2026-06 | 🔥 [自蒸馏策略梯度（SDPG）](https://arxiv.org/abs/2606.04036) `arXiv:2606.04036` |  | 把在策略自蒸馏的稠密信号与稀疏奖励的策略梯度统一到一个目标中 |
| 2026-05 | 🔥 [TrOPD：信任域在策略蒸馏](https://arxiv.org/abs/2606.01249) `arXiv:2606.01249` |  | 给 OPD 加信任域约束，稳定地超过 OPD、EOPD、REOPOLD 等方法 |
| 2026-05 | 🔥 [TRB：在策略蒸馏的信任域行为混合](https://arxiv.org/abs/2605.31159) `arXiv:2605.31159` |  | OPD 早期学生 rollout 太差：在以学生为中心的 KL 信任域内用最接近教师的行为策略做热身 |
| 2026-05 | 🔥 [并非所有分歧都可学：OPD 的 token 可教性](https://arxiv.org/abs/2605.26844) `arXiv:2605.26844` |  | 原始的师生 KL 只是粗糙代理：衡量同一上下文下 KL 的可下降程度来挑选 token |
| 2026-05 | 🔥 [自蒸馏智能体 RL](https://arxiv.org/abs/2605.15155) `arXiv:2605.15155` |  | 智能体后训练中的自蒸馏强化学习 |
| 2026-05 | 🔥 [反自蒸馏（AntiSD）](https://arxiv.org/abs/2605.11609) `arXiv:2605.11609` |  | 反直觉：上升而非下降师生散度，得到天然有界、按点互信息加权的优势，用于推理 RL |
| 2026-05 | 🔥 [揭开在策略蒸馏](https://arxiv.org/abs/2605.10889) `arXiv:2605.10889` |  | OPD 何时有益、何时有害：教师选择与自蒸馏上下文的系统分析 |
| 2026-05 | 🔥 [ROPD：基于评分细则的在策略蒸馏](https://arxiv.org/abs/2605.07396) `arXiv:2605.07396` |  | 不要教师 logits：教师写评分细则、学生按细则得分，适用于黑盒教师 |
| 2026-04 | ◆ [PRISM：SFT 与 RL 之间的分布对齐](https://arxiv.org/abs/2604.28123) `arXiv:2604.28123` |  | 在 SFT 与 RLVR 之间插入黑盒在策略蒸馏阶段，缓解多模态 RL 的分布漂移 |
| 2026-04 | ◆ [从近未来学习：时间自蒸馏](https://arxiv.org/abs/2604.20733) `arXiv:2604.20733` |  | 用自己稍后、更强的检查点作为教师给当前策略做蒸馏信号 |
| 2026-04 | ◆ [在策略蒸馏的"确定性幻觉"](https://arxiv.org/abs/2604.16830) `arXiv:2604.16830` |  | OPD 提升准确率却系统性地让模型过度自信（失校准的扩展律），提出校准感知的修正 |
| 2026-04 | ◆ [重新思考在策略蒸馏](https://arxiv.org/abs/2604.13016) `arXiv:2604.13016` | Tsinghua NLP | 在策略蒸馏的现象、机制与配方 |
| 2026-04 | ◆ [Lightning OPD：离线的在策略蒸馏](https://arxiv.org/abs/2604.13010) `arXiv:2604.13010` |  | 预先采样学生 rollout、离线算教师 logits：前提是"教师一致性"，可大幅降低 OPD 成本 |
| 2026-04 | ◆ [SD-Zero：自我修订把二值奖励变成稠密监督](https://arxiv.org/abs/2604.12002) `arXiv:2604.12002` |  | 模型先写答案再自我修订，把修订版当教师做逐 token 蒸馏：比 RL 样本效率高得多 |
| 2026-04 | ◆ [自蒸馏 RLVR](https://arxiv.org/abs/2604.03128) `arXiv:2604.03128` |  | 在 RLVR 中结合模型自身的蒸馏信号 |
| 2026-04 | ◆ [SRPO：GRPO 与自蒸馏按样本路由](https://arxiv.org/abs/2604.02288) `arXiv:2604.02288` |  | 成功样本走 GRPO 的奖励强化，失败样本走 SDPO 的逐 token 自蒸馏纠正 |
| 2026-04 | ◆ [极简自蒸馏提升代码生成](https://arxiv.org/abs/2604.01193) `arXiv:2604.01193` |  | 不要验证器、教师或 RL：用自己采样（特定温度与截断）的原始输出做 SFT 即可提升代码能力 |
| 2026-04 | ◆ [在策略蒸馏综述](https://arxiv.org/abs/2604.00626) `arXiv:2604.00626` |  | 系统梳理 OPD：散度选择、教师信号、与 RL 的关系 |
| 2026-03 | ◆ [再看在策略蒸馏：失败模式与修复](https://arxiv.org/abs/2603.25562) `arXiv:2603.25562` |  | 采样 token 的 OPD 有三种失败：token 监督失衡、教师在学生前缀上不可靠、分词器/特殊 token 不匹配 |
| 2026-03 | ◆ [自蒸馏为何有时损害推理](https://arxiv.org/abs/2603.24472) `arXiv:2603.24472` |  | 自蒸馏压短回答的同时抑制了"认知言语化"（表达不确定性的步骤），从而降低数学推理 |
| 2026-03 | ◆ [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) `arXiv:2603.19220` | NVIDIA | 级联 RL + 多领域在策略蒸馏的后训练 |
| 2026-02 | ◆ [奖励外推的广义在策略蒸馏](https://arxiv.org/abs/2602.12125) `arXiv:2602.12125` | Tencent Hunyuan | 在蒸馏中外推奖励，让学生超过教师 |
| 2026-01 | ★ ◆ [SDPO：通过自蒸馏做 RL](https://arxiv.org/abs/2601.20802) `arXiv:2601.20802` | ETH 等 | 把环境的文字反馈（报错、测试结果）放进上下文，让模型自己当教师给逐 token 信号：富反馈下的 RL |
| 2026-01 | ◆ [自蒸馏实现持续学习（SDFT）](https://arxiv.org/abs/2601.19897) `arXiv:2601.19897` | MIT 等 | 以"看过示范的自己"为教师做在策略自蒸馏：从示范学新技能，同时比 SFT 遗忘少得多 |
| 2025-11 | ◆ [黑盒在策略蒸馏（GAD）](https://arxiv.org/abs/2511.10643) `arXiv:2511.10643` | Microsoft | 把学生当生成器、训练判别器区分师生回答，判别器即在策略奖励模型：只能拿到教师文本也能做在策略蒸馏 |
| 2025-10 | ★ 🆕 [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | Thinking Machines | 学生自己采样、教师逐 token 打分（反向 KL）：兼得 RL 的在策略与蒸馏的稠密信号，成本远低于 RL |

### 探索、熵与多样性（42）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [EDEN：嵌入空间的定向探索](https://arxiv.org/abs/2610.16105) `arXiv:2610.16105` |  | 在输入嵌入上加锚定噪声，从 rollout 的源头扩大探索 |
| 2026-09 | 🔥 [GRAFT：跨模型轨迹交换](https://arxiv.org/abs/2609.37868) `arXiv:2609.37868` |  | 全错的组用同伴模型的轨迹替换，并做离策略修正：学到自己采不到的东西 |
| 2026-09 | 🔥 [MInTRL：最小干预](https://arxiv.org/abs/2609.12419) `arXiv:2609.12419` |  | 在在策略 rollout 中做稀疏、局部的离策略干预（如替换个别 token）扩大探索前沿 |
| 2026-07 | 🔥 [GEPO：组熵控制](https://arxiv.org/abs/2607.16850) `arXiv:2607.16850` |  | 用组内样本估计的熵做熵控制，GRPO 的轻量扩展 |
| 2026-07 | 🔥 [RIPO：黎曼等距策略优化](https://arxiv.org/abs/2607.10169) `arXiv:2607.10169` |  | 欧氏空间裁剪对低概率 token 不公平：在黎曼流形上做等距更新，避免探索塌缩 |
| 2026-06 | 🔥 [ZPPO：最近发展区策略优化](https://arxiv.org/abs/2606.18216) `arXiv:2606.18216` |  | 教师只出现在提示里（给提示/示范），不进梯度：让学生在"跳一跳够得着"的区域学习 |
| 2026-05 | 🔥 [小模型是天然的探索者](https://arxiv.org/abs/2605.30789) `arXiv:2605.30789` |  | GRPO 组内混入小模型的 rollout 提供策略层面的多样性 |
| 2026-05 | 🔥 [NudgeRL：策略级引导的探索](https://arxiv.org/abs/2605.15726) `arXiv:2605.15726` |  | 每个 rollout 前加一句轻量的"解题策略"上下文，结构化地扩大探索 |
| 2026-05 | 🔥 [LoPE：用无意义扰动拓宽探索](https://arxiv.org/abs/2605.05566) `arXiv:2605.05566` |  | 在提示中插入无语义的填充文本扰动提示空间，打破 RLVR 的探索瓶颈 |
| 2026-04 | ◆ [Entrocraft：精确控制熵曲线](https://arxiv.org/abs/2604.26326) `arXiv:2604.26326` |  | 用拒绝采样偏置优势分布，实现用户指定的熵调度，缓解性能饱和 |
| 2026-04 | ◆ [输出多样性在哪里坍缩](https://arxiv.org/abs/2604.16027) `arXiv:2604.16027` |  | 后训练中多样性坍缩发生在哪一阶段、由什么导致 |
| 2026-04 | ◆ [KnowRL：最小充分的知识提示](https://arxiv.org/abs/2604.12627) `arXiv:2604.12627` |  | 难题上采不到正解时给提示：把提示设计看作"最小充分"问题，只给刚好够用的知识点 |
| 2026-04 | ◆ [MEDS：记忆增强的动态奖励塑形](https://arxiv.org/abs/2604.11297) `arXiv:2604.11297` |  | 把历史上反复出现的错误模式记入记忆，对重复犯错的 rollout 施加动态惩罚，鼓励探索新解法 |
| 2026-03 | ◆ [大规模高效探索（RLHF）](https://arxiv.org/abs/2603.17378) `arXiv:2603.17378` |  | 在线 RLHF 的探索算法（不确定性引导的成对比较），显著提升从人类反馈学习的数据效率 |
| 2026-03 | ◆ [GOLF：组级语言反馈引导探索](https://arxiv.org/abs/2603.04597) `arXiv:2603.04597` |  | 把一组 rollout 的自然语言反馈汇总成可执行的改进，引导有针对性的探索 |
| 2026-02 | ◆ [DSDR：双尺度多样性正则](https://arxiv.org/abs/2602.19895) `arXiv:2602.19895` |  | 把推理多样性分解为全局（解法间）与耦合（token 间）两部分分别正则 |
| 2026-02 | ◆ [经验式强化学习](https://arxiv.org/abs/2602.13949) `arXiv:2602.13949` | Microsoft | 把反思得到的经验纳入 RL 循环 |
| 2026-02 | ◆ [想得更长才能探索得更深](https://arxiv.org/abs/2602.11748) `arXiv:2602.11748` |  | 在单条上下文中生成、验证、修正多个假设才能扩展测试时计算：用长度激励训练"上下文内探索" |
| 2026-02 | ◆ [F-GRPO](https://arxiv.org/abs/2602.06717) `arXiv:2602.06717` | T-Tech | 防止策略只学"显而易见"而遗忘罕见正确解 |
| 2026-02 | ◆ [回到基础：重访 LLM 推理 RL 的探索](https://arxiv.org/abs/2602.05281) `arXiv:2602.05281` |  | 正确回答之间置信度差异大会压制探索：用提示困惑度与答案置信度重加权正样本优势 |
| 2026-02 | ◆ [强化微调的熵动力学](https://arxiv.org/abs/2602.03392) `arXiv:2602.03392` |  | 推导强化微调中策略熵如何随更新变化 |
| 2026-02 | ◆ [SAGE：自提示增强 RL](https://arxiv.org/abs/2602.03143) `arXiv:2602.03143` |  | 训练时注入特权提示改变 rollout 分布、但仍在同一奖励下优化，难题上也能采到正例 |
| 2026-01 | ◆ [奖励稀有解](https://arxiv.org/abs/2601.08763) `arXiv:2601.08763` | MIT | 按解法独特性加权的 RL，提升创造性解题 |
| 2025-12 | ◆ [探索与利用：重新理解 RLVR](https://arxiv.org/abs/2512.16912) `arXiv:2512.16912` |  | 虚假奖励与熵最小化这两种看似矛盾的机制如何都能提升推理：从裁剪、熵与虚假奖励分析 |
| 2025-12 | ◆ [LaMer：元强化学习诱导探索](https://arxiv.org/abs/2512.16848) `arXiv:2512.16848` |  | 跨回合的元 RL：智能体在测试时利用前几回合的反馈主动探索、调整策略 |
| 2025-12 | ◆ [G2RL：梯度引导探索](https://arxiv.org/abs/2512.15687) `arXiv:2512.15687` |  | 用模型自身一阶更新的几何决定探索方向：奖励那些带来新梯度方向的回答 |
| 2025-10 | 🆕 [max@k 优化](https://arxiv.org/abs/2510.23393) `arXiv:2510.23393` |  | 为 Best-of-N 推理直接优化 max@k 的无偏梯度估计，含离策略情形 |
| 2025-10 | 🆕 [保留候选支撑集（CaSP）](https://arxiv.org/abs/2510.14807) `arXiv:2510.14807` |  | RLVR 只更新被采样的 token，会让未被采到的正确候选概率塌缩：保留候选集合的支撑以维持 pass@k |
| 2025-10 | 🆕 [低概率 token 维持探索](https://arxiv.org/abs/2510.03222) `arXiv:2510.03222` | Tencent | RLVR 中低概率 token 是探索的关键，保护它们可避免熵塌缩 |
| 2025-09 | 🆕 [CE-GPPO](https://arxiv.org/abs/2509.20712) `arXiv:2509.20712` | Kuaishou | 温和地回传被裁剪 token 的梯度来调节熵：低概率 token 控制熵的升降 |
| 2025-09 | 🆕 [FlowRL](https://arxiv.org/abs/2509.15207) `arXiv:2509.15207` |  | 匹配奖励分布而非最大化奖励，保持推理多样性 |
| 2025-09 | 🆕 [CDE：好奇心驱动探索](https://arxiv.org/abs/2509.09675) `arXiv:2509.09675` |  | 以 actor 的困惑度与 critic 多头价值的方差作为好奇心奖励，缓解熵塌缩 |
| 2025-09 | 🆕 [DARLING：同时强化多样性与质量](https://arxiv.org/abs/2509.02534) `arXiv:2509.02534` | Meta | 用学习到的语义等价划分度量多样性，与质量奖励相乘：开放生成与数学上都更好且更多样 |
| 2025-08 | 🆕 [Pass@k 训练](https://arxiv.org/abs/2508.10751) `arXiv:2508.10751` |  | 以 pass@k 为奖励训练：探索与利用并不冲突，先用 pass@k 训练再转 pass@1 两者兼得 |
| 2025-07 | 🆕 [FR3E：先返回、再探索](https://arxiv.org/abs/2507.07017) `arXiv:2507.07017` |  | 借鉴 Go-Explore：找到轨迹中的高熵决策点，从那里分支 rollout，构造有语义的中间反馈 |
| 2025-06 | 🆕 [以熵视角看探索](https://arxiv.org/abs/2506.14758) `arXiv:2506.14758` |  | 高熵 token 对应关键推理动作；在优势上加一项截断的熵奖励（一行代码），显著提升 pass@k |
| 2025-06 | 🆕 [e3：学会探索才能外推测试时计算](https://arxiv.org/abs/2506.09026) `arXiv:2506.09026` | CMU | 非对称能力（验证比生成容易）+ 负梯度 + 课程，让模型在更长的预算下继续变好 |
| 2025-06 | 🆕 [奖励不太可能的正解](https://arxiv.org/abs/2506.02355) `arXiv:2506.02355` | CMU | GRPO 偏向锐化高概率解；给低概率的正确证明额外奖励，提升 pass@k |
| 2025-06 | 🆕 [高熵少数 token](https://arxiv.org/abs/2506.01939) `arXiv:2506.01939` | Qwen/Tsinghua | 只有约 20% 的高熵"分叉"token 决定 RLVR 效果，只对它们算梯度更好 |
| 2025-06 | 🆕 [负强化的意外有效性](https://arxiv.org/abs/2506.01347) `arXiv:2506.01347` |  | 只用负样本（压低错误回答）训练就能保持多样性并提升 pass@k |
| 2025-05 | ★ 🆕 [RL 的熵机制](https://arxiv.org/abs/2505.22617) `arXiv:2505.22617` | Shanghai AI Lab/Tsinghua | 策略熵塌缩限制性能，熵变由动作概率与优势的协方差驱动，提出 Clip-Cov/KL-Cov |
| 2025-05 | 🆕 [PKPO：pass@k 策略优化](https://arxiv.org/abs/2505.15201) `arXiv:2505.15201` | Google DeepMind | 给出 pass@k 的无偏低方差梯度估计，可在训练中逐步把 k 调小 |

### 能力边界：RL 学到了什么（18）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-06 | 🔥 [RL 激发对未见语言的上下文翻译](https://arxiv.org/abs/2606.06428) `arXiv:2606.06428` |  | 只用 chrF 这样的表层指标做奖励，RL 就能让模型利用上下文中的语法与词典翻译从未见过的语言 |
| 2026-05 | 🔥 [推理模型为何失去覆盖](https://arxiv.org/abs/2605.17026) `arXiv:2605.17026` |  | pass@1 提升而 pass@k 下降：数据与"岔路口" token 的作用 |
| 2026-05 | 🔥 [LLM 能学会多深的推理](https://arxiv.org/abs/2605.06638) `arXiv:2605.06638` |  | 可控难度的合成任务表明：RL 能否学会长程推理取决于模型表达能力 |
| 2026-04 | ◆ [推理 SFT 的泛化是有条件的](https://arxiv.org/abs/2604.06628) `arXiv:2604.06628` |  | "SFT 记忆、RL 泛化"并不绝对：长思维链 SFT 的跨领域泛化取决于优化、数据与模型能力 |
| 2026-03 | ◆ [稀疏但关键：RLVR 的 token 级分布变化](https://arxiv.org/abs/2603.22446) `arXiv:2603.22446` |  | RL 后模型与基座只在少量 token 上分布不同，这些 token 决定了推理收益 |
| 2026-01 | ◆ [虚假奖励悖论](https://arxiv.org/abs/2601.11061) `arXiv:2601.11061` |  | 虚假奖励下的 RLVR 激活了记忆捷径：机制分析出"困惑度悖论" |
| 2025-10 | ★ 🆕 [用采样推理](https://arxiv.org/abs/2510.14901) `arXiv:2510.14901` | Harvard | 对基座模型做 MCMC 幂分布采样，不训练即可接近 RL 后的推理能力 |
| 2025-07 | 🆕 [隐形的缰绳](https://arxiv.org/abs/2507.14843) `arXiv:2507.14843` | Stanford 等 | RLVR 难以跳出基座模型支撑集，主要做"保守的重加权" |
| 2025-07 | ★ 🆕 [推理还是记忆](https://arxiv.org/abs/2507.10532) `arXiv:2507.10532` |  | 部分 RLVR 结果（如虚假奖励有效）源于基准数据污染 |
| 2025-07 | 🆕 [数学推理能否迁移](https://arxiv.org/abs/2507.00432) `arXiv:2507.00432` | CMU 等 | RL 训练的数学推理比 SFT 更好地迁移到其它领域 |
| 2025-06 | 🆕 [RLVR 隐式激励正确推理](https://arxiv.org/abs/2506.14245) `arXiv:2506.14245` | Microsoft | 用 CoT-Pass@k 指标说明 RLVR 确实提升了推理过程的正确性 |
| 2025-06 | ★ 🆕 [虚假奖励](https://arxiv.org/abs/2506.10947) `arXiv:2506.10947` | UW/Ai2 | 随机甚至错误的奖励也能提升 Qwen2.5-Math：RLVR 的收益可能来自激发预训练中的行为 |
| 2025-05 | 🆕 [ProRL](https://arxiv.org/abs/2505.24864) `arXiv:2505.24864` | NVIDIA | 长时间、稳定的 RL（KL 控制 + 参考模型重置）能扩展推理边界，反驳"RL 只是重排" |
| 2025-04 | ★ 🆕 [RL 真的扩展了推理边界吗](https://arxiv.org/abs/2504.13837) `arXiv:2504.13837` | Tsinghua | 大 k 下 pass@k 基座模型反超 RLVR 模型：RL 主要提高采样效率而非创造新能力 |
| 2025-03 | 🆕 [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) `arXiv:2503.18892` | HKUST | 在 10 个不同基座上复现零 RL，发现"顿悟"并非普遍出现 |
| 2025-03 | ★ 🆕 [让推理者能自我改进的认知行为](https://arxiv.org/abs/2503.01307) `arXiv:2503.01307` | Stanford | 验证、回溯、子目标、逆向推理四种行为决定了 RL 能否奏效：Qwen 天然有、Llama 需要先注入 |
| 2025-02 | 🆕 [LIMO](https://arxiv.org/abs/2502.03387) `arXiv:2502.03387` | SJTU | 817 条精选样本激发复杂数学推理：推理能力已在预训练中，关键是好的认知模板 |
| 2025-01 | 🆕 [SFT 记忆，RL 泛化](https://arxiv.org/abs/2501.17161) `arXiv:2501.17161` | HKU/Google DeepMind/NYU | 对比实验：SFT 倾向记忆训练数据，RL（尤其结果奖励）能泛化到规则与视觉变体 |

### 过程奖励与信用分配（35）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [MRTPO：每个奖励只给相关 token](https://arxiv.org/abs/2610.21439) `arXiv:2610.21439` |  | 利用各验证器检查的信息，把每个奖励的优势只分配到与之相关的 token |
| 2026-09 | 🔥 [TIPS：从结果 RL 中诱导过程监督](https://arxiv.org/abs/2609.36641) `arXiv:2609.36641` |  | 只用结果奖励的 RL 训练生成式 PRM，低成本得到过程监督 |
| 2026-09 | 🔥 [EAPO：熵引导的信用分配](https://arxiv.org/abs/2609.33781) `arXiv:2609.33781` |  | 意外的成功与反复的失败应区别对待：按熵非对称地分配成功与失败的信用 |
| 2026-09 | 🔥 [PACT：从信用分配到 critic 对齐](https://arxiv.org/abs/2609.26355) `arXiv:2609.26355` |  | 智能体数学推理与 SWE-bench 上分别超过 GRPO 与 PPO：把 critic 与信用分配对齐 |
| 2026-09 | 🔥 [Cliff：从第一个错误学过程奖励](https://arxiv.org/abs/2609.02817) `arXiv:2609.02817` |  | 用现成 LLM 找出每条 rollout 的第一个错误步，据此塑形奖励 |
| 2026-06 | 🔥 [进度优势：后训练的免费午餐](https://arxiv.org/abs/2606.26080) `arXiv:2606.26080` |  | RL 后训练本身已提供步骤级打分所需的要素，无需单独训练奖励模型 |
| 2026-05 | 🔥 [DelTA](https://arxiv.org/abs/2605.21467) `arXiv:2605.21467` |  | 判别式的 token 级信用分配 |
| 2026-05 | 🔥 [BetaPRM：带可靠度的过程奖励](https://arxiv.org/abs/2605.15529) `arXiv:2605.15529` |  | PRM 同时预测步骤成功概率及其可靠度（Beta 分布），据此自适应分配推理计算 |
| 2026-05 | 🔥 [无监督过程奖励模型（uPRM）](https://arxiv.org/abs/2605.10158) `arXiv:2605.10158` |  | 既不要步骤标注也不要最终答案验证，训练过程奖励模型 |
| 2026-04 | ◆ [LenVM：长度价值模型](https://arxiv.org/abs/2604.27039) `arXiv:2604.27039` |  | 逐 token 预测剩余生成长度的价值模型，可大规模预训练，用于长度控制与优势估计 |
| 2026-04 | ◆ [从推理到智能体：LLM RL 的信用分配（综述）](https://arxiv.org/abs/2604.09459) `arXiv:2604.09459` |  | 梳理 token、步骤、工具调用、记忆操作与多智能体层面的信用分配方法 |
| 2026-03 | ◆ [FIPO](https://arxiv.org/abs/2603.19835) `arXiv:2603.19835` | Qwen | 以未来 KL 影响加权的策略优化，引出更深的推理 |
| 2026-03 | ◆ [V0.5：通用价值模型作稀疏 rollout 的先验](https://arxiv.org/abs/2603.10848) `arXiv:2603.10848` |  | 预训练的通用价值模型在上下文中编码策略能力，作为先验与少量 rollout 结合估计基线 |
| 2026-02 | ◆ [分块优势估计：多目标 RLVR](https://arxiv.org/abs/2602.10231) `arXiv:2602.10231` |  | 每个目标只对相应文本块的 token 施加自己的优势，避免把多目标压成一个标量 |
| 2026-02 | ◆ [多模态 PRM 的训练数据效率](https://arxiv.org/abs/2602.04145) `arXiv:2602.04145` |  | 蒙特卡洛过程标签多数冗余：按"正负混合度 + 可靠性"挑选 rollout，少量数据即可训练多模态 PRM |
| 2026-01 | ◆ [Future-KL 正则的 GRPO](https://arxiv.org/abs/2601.10201) `arXiv:2601.10201` |  | 自回归 KL 正则本身诱导了过程级的信用信号，常见实现漏掉了这一项 |
| 2025-12 | ◆ [OPV：基于结果的过程验证器](https://arxiv.org/abs/2512.10756) `arXiv:2512.10756` |  | 先把长 CoT 摘要成关键步骤再逐步验证，用主动学习降低过程标注成本 |
| 2025-12 | ◆ [ScalePRM](https://arxiv.org/abs/2512.03244) `arXiv:2512.03244` |  | 不用人工步骤标注：对每步做多次独立验证再聚合，扩展验证计算来合成过程标签 |
| 2025-10 | 🆕 [注意力揭示推理节奏](https://arxiv.org/abs/2510.13554) `arXiv:2510.13554` |  | 注意力呈"先规划、后锚定"的节律；据此把信用集中到关键的规划与锚点 token |
| 2025-09 | 🆕 [RL 中涌现的层级推理](https://arxiv.org/abs/2509.03646) `arXiv:2509.03646` |  | "顿悟"、长度增长与熵变化都源于高层策略与低层执行的分离；HICRA 只对规划 token 加强信用 |
| 2025-08 | 🆕 [TreePO](https://arxiv.org/abs/2508.17445) `arXiv:2508.17445` | ByteDance Seed | 树状分段采样共享前缀，提高 rollout 效率与优势估计 |
| 2025-04 | 🆕 [会思考的过程奖励模型（ThinkPRM）](https://arxiv.org/abs/2504.16828) `arXiv:2504.16828` |  | 生成式 PRM 先写验证 CoT 再判断每一步，只需极少过程标签即超过判别式 PRM |
| 2025-04 | 🆕 [停止求和：最小式信用分配](https://arxiv.org/abs/2504.15275) `arXiv:2504.15275` |  | PRM 奖励按"取最小"而非求和累积，可抑制对 PRM 的奖励投机 |
| 2025-02 | ★ 🆕 [PRIME：隐式过程奖励](https://arxiv.org/abs/2502.01456) `arXiv:2502.01456` | Tsinghua | 用结果标签训练的隐式 PRM 提供稠密奖励，在线更新 |
| 2025-01 | 🆕 [PRM 的经验教训](https://arxiv.org/abs/2501.07301) `arXiv:2501.07301` | Qwen | 蒙特卡洛估计的过程标签不可靠，LLM 评判 + 共识过滤更好 |
| 2025-01 | 🆕 [rStar-Math](https://arxiv.org/abs/2501.04519) `arXiv:2501.04519` | Microsoft Research Asia | 小模型 + MCTS + 过程偏好模型自我进化，7B 达到 o1-preview 级数学能力 |
| 2024-12 | 🆕 [ProcessBench](https://arxiv.org/abs/2412.06559) `arXiv:2412.06559` | Qwen | 识别数学推理过程中最早错误步的基准 |
| 2024-12 | 🆕 [免过程标签的过程奖励](https://arxiv.org/abs/2412.01981) `arXiv:2412.01981` | Tsinghua 等 | 只用结果标签训练、以对数似然比参数化的奖励模型，隐式地给出 token 级过程奖励 |
| 2024-10 | ★ [奖励进步：过程验证器的扩展](https://arxiv.org/abs/2410.08146) `arXiv:2410.08146` | Google 等 | 步骤奖励应衡量"进步"（证明者策略下成功概率的变化，即优势），而非步骤本身的正确性 |
| 2024-10 | ★ [VinePPO](https://arxiv.org/abs/2410.01679) `arXiv:2410.01679` | Mila | 用从中间状态出发的蒙特卡洛续写估计价值，替代不准的价值网络 |
| 2024-06 | [错误的合成数据上做 RL](https://arxiv.org/abs/2406.14532) `arXiv:2406.14532` | CMU | 在错误轨迹上做按步优势加权的负样本学习，相当于 8 倍数据效率 |
| 2024-06 | [OmegaPRM：自动过程监督](https://arxiv.org/abs/2406.06592) `arXiv:2406.06592` | Google DeepMind | 用二分查找式的 MCTS 定位第一个错误步，高效收集过程标签 |
| 2023-12 | [Math-Shepherd](https://arxiv.org/abs/2312.08935) `arXiv:2312.08935` | PKU 等 | 用蒙特卡洛续写估计每步能否到达正确答案，自动构造过程标签训练 PRM |
| 2023-05 | ★ [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) `arXiv:2305.20050` | OpenAI | 过程监督（逐步打分）优于结果监督，发布 PRM800K |
| 2022-11 | [过程反馈与结果反馈](https://arxiv.org/abs/2211.14275) `arXiv:2211.14275` | DeepMind | 数学应用题上比较过程监督与结果监督：最终答案准确率相近，过程监督的推理错误更少 |

### 推理模型与测试时计算（38）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-06 | 🔥 [SWITCH：可切换的潜在推理](https://arxiv.org/abs/2606.13106) `arXiv:2606.13106` |  | 用两个边界 token 在显式与潜在推理之间切换，并用在策略 RL 训练 |
| 2026-06 | 🔥 [小 RL 控制器指导大模型采样](https://arxiv.org/abs/2606.03102) `arXiv:2606.03102` |  | CPU 上训练的小 RL 控制器根据答案统计决定何时停止采样，节省测试时计算 |
| 2026-02 | ◆ [Reasoning Cache：用短程 RL 获得长程能力](https://arxiv.org/abs/2602.03773) `arXiv:2602.03773` |  | 迭代解码：每段推理后压缩成摘要再继续，用 16k 训练长度得到可外推到更长推理的模型 |
| 2026-01 | ◆ [潜空间主动规划的 RL（ATP）](https://arxiv.org/abs/2601.21598) `arXiv:2601.21598` |  | 潜在推理不应只模仿语言 CoT：用 RL 在潜 token 表征空间中主动规划 |
| 2026-01 | ◆ [PaCoRe](https://arxiv.org/abs/2601.05593) `arXiv:2601.05593` | StepFun | 并行协同推理，学习扩展测试时计算 |
| 2025-12 | ◆ [QwenLong-L1.5](https://arxiv.org/abs/2512.12967) `arXiv:2512.12967` | Qwen | 长上下文推理与记忆管理的后训练配方 |
| 2025-12 | ◆ [原生并行推理器](https://arxiv.org/abs/2512.07461) `arXiv:2512.07461` | BIGAI | 自蒸馏 RL 学会并行推理 |
| 2025-11 | ◆ [P1 物理奥赛](https://arxiv.org/abs/2511.13612) `arXiv:2511.13612` | Shanghai AI Lab | 用 RL 训练的物理推理模型在 IPhO 达到金牌水平 |
| 2025-10 | 🆕 [LoongRL](https://arxiv.org/abs/2510.19363) `arXiv:2510.19363` | Microsoft Research | 面向长上下文高级推理的 RL |
| 2025-10 | 🆕 [免训练 GRPO](https://arxiv.org/abs/2510.08191) `arXiv:2510.08191` | Tencent | 不更新权重：把组内比较得到的经验写成文本"先验"放进上下文，用极少样本得到类似 RL 的效果 |
| 2025-10 | 🆕 [Markovian Thinker](https://arxiv.org/abs/2510.06557) `arXiv:2510.06557` | Mila/Microsoft | 把思考切成固定长度块、只携带短状态，使推理长度与上下文长度解耦（线性算力） |
| 2025-09 | 🆕 [Parallel-R1](https://arxiv.org/abs/2509.07980) `arXiv:2509.07980` | Tencent | 用 RL 训练并行思考 |
| 2025-09 | 🆕 [学习聚合多个解](https://arxiv.org/abs/2509.06870) `arXiv:2509.06870` | Meta | 多数投票并不总对：用 RL 训练模型审阅多个候选解并综合出正确答案 |
| 2025-07 | ★ 🆕 [GEPA：反思式提示演化](https://arxiv.org/abs/2507.19457) `arXiv:2507.19457` | UC Berkeley 等 | 用自然语言反思迭代改写提示（遗传 + 帕累托选择），在 6 个任务上平均超过 GRPO 6%，rollout 少 35 倍 |
| 2025-07 | 🆕 [Gemini Deep Think IMO 金牌](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) | Google DeepMind | 通用语言模型用自然语言端到端在 4.5 小时内解出 6 题中的 5 题，首次达到 IMO 金牌线（官方评分） |
| 2025-06 | 🆕 [AlphaEvolve](https://arxiv.org/abs/2506.13131) `arXiv:2506.13131` | Google DeepMind | Gemini + 进化搜索 + 自动评估的编码智能体，发现更快的矩阵乘法算法并优化数据中心调度 |
| 2025-05 | 🆕 [Reflect, Retry, Reward](https://arxiv.org/abs/2505.24726) `arXiv:2505.24726` |  | 通过 RL 学会反思后重试的自我改进 |
| 2025-05 | 🆕 [QwenLong-L1](https://arxiv.org/abs/2505.17667) `arXiv:2505.17667` | Qwen | 用渐进式上下文扩展的 RL 训练长上下文推理模型 |
| 2025-05 | 🆕 [超越"顿悟"](https://arxiv.org/abs/2505.10554) `arXiv:2505.10554` | NUS 等 | 显式对齐演绎、归纳、溯因三种元能力，而非依赖不可控的"aha" |
| 2025-04 | 🆕 [OpenAI o3 / o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/) | OpenAI | 在思维链中调用工具、"用图像思考"；RL 算力继续扩大一个数量级 |
| 2025-04 | 🆕 [DeepSeek-R1 思维学](https://arxiv.org/abs/2504.07128) `arXiv:2504.07128` | Mila/McGill | 系统分析 R1 推理链的结构、长度与"思考甜蜜点" |
| 2025-03 | 🆕 [MRT：元强化微调](https://arxiv.org/abs/2503.07572) `arXiv:2503.07572` | CMU | 把测试时计算看作元 RL：按每段思考带来的"进步"给稠密奖励，减少无效思考 |
| 2025-03 | 🆕 [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b/) | Qwen | 基于结果奖励（准确率验证器与代码执行）的多阶段 RL，32B 接近 DeepSeek-R1 |
| 2025-02 | 🆕 [Logic-RL](https://arxiv.org/abs/2502.14768) `arXiv:2502.14768` |  | 在程序生成的"骑士与无赖"逻辑题上做规则奖励 RL，泛化到数学竞赛 |
| 2025-02 | 🆕 [1B 能否超过 405B](https://arxiv.org/abs/2502.06703) `arXiv:2502.06703` | Tsinghua/Shanghai AI Lab | 算力最优的测试时扩展策略依赖策略模型、PRM 与题目难度 |
| 2025-02 | 🆕 [揭秘长思维链](https://arxiv.org/abs/2502.03373) `arXiv:2502.03373` | CMU/Tsinghua | 长 CoT 的出现条件：SFT 的作用、奖励塑形防止长度爆炸、可验证噪声数据 |
| 2025-02 | ★ 🆕 [用大推理模型做竞赛编程](https://arxiv.org/abs/2502.06807) `arXiv:2502.06807` | OpenAI | 通用 RL 扩展的 o3 无需人工策略即在 IOI 达到金牌 |
| 2025-01 | 🆕 [s1](https://arxiv.org/abs/2501.19393) `arXiv:2501.19393` | Stanford | 1000 条样本 SFT + "budget forcing" 控制思考长度即可实现测试时扩展 |
| 2025-01 | ★ 🆕 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) `arXiv:2501.12948` | DeepSeek | 纯 RL（R1-Zero）从基座涌现长思维链与自我反思；开源权重与蒸馏小模型，引发 RLVR 热潮 |
| 2024-09 | [SCoRe：用 RL 学会自我纠错](https://arxiv.org/abs/2409.12917) `arXiv:2409.12917` | Google DeepMind | 多轮在线 RL 训练模型在第二轮修正第一轮的错误；SFT 做不到 |
| 2024-09 | ★ [OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/) | OpenAI | 大规模 RL 训练长思维链：性能随训练算力与思考时间平滑提升，开启推理模型时代 |
| 2024-08 | [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152) `arXiv:2408.08152` | DeepSeek | 以 Lean 证明助手反馈为奖励做 RL，并用 RMaxTS（内在奖励驱动的 MCTS）探索 |
| 2024-08 | ★ [测试时计算的最优扩展](https://arxiv.org/abs/2408.03314) `arXiv:2408.03314` | UC Berkeley/Google | 按题目难度自适应分配测试时计算，效率可超过把模型放大 14 倍 |
| 2024-08 | [推理扩展律](https://arxiv.org/abs/2408.00724) `arXiv:2408.00724` | CMU/Tsinghua | 固定推理算力下，小模型 + 更强的推理搜索可以更划算 |
| 2024-07 | [Large Language Monkeys](https://arxiv.org/abs/2407.21787) `arXiv:2407.21787` | Stanford/Oxford | 重复采样的覆盖率随样本数呈对数线性增长：推理时采样也是一条扩展轴 |
| 2024-03 | [Quiet-STaR](https://arxiv.org/abs/2403.09629) `arXiv:2403.09629` | Stanford | 让模型在每个 token 前生成内部"思考"再预测 |
| 2023-05 | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) `arXiv:2305.10601` | Princeton/Google DeepMind | 把推理组织成树并搜索、回溯 |
| 2022-03 | [自洽性](https://arxiv.org/abs/2203.11171) `arXiv:2203.11171` | Google | 采样多条推理链后多数投票 |

### 高效推理与长度控制（10）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-03 | ◆ [推理即压缩](https://arxiv.org/abs/2603.08462) `arXiv:2603.08462` |  | 用条件信息瓶颈把高效推理重述为有损压缩，统一各类预算控制方法 |
| 2026-02 | ◆ [推理模型知道何时停止吗](https://arxiv.org/abs/2602.08354) `arXiv:2602.08354` | ByteDance | 模型内部隐含"何时该停"的信号，可用于缩短思考 |
| 2025-10 | 🆕 [DLER：把长度惩罚做对](https://arxiv.org/abs/2510.15110) `arXiv:2510.15110` | NVIDIA | 最简单的截断长度惩罚之所以差，是因为优势偏差、熵塌缩与稀疏信号；修正后推理长度减七成以上而准确率更高 |
| 2025-07 | 🆕 [LAPO：长度自适应策略优化](https://arxiv.org/abs/2507.15758) `arXiv:2507.15758` |  | 先统计成功解的长度分布，再把合理长度作为上下文提示内化，按难度分配推理预算 |
| 2025-05 | 🆕 [AdaptThink](https://arxiv.org/abs/2505.13417) `arXiv:2505.13417` | Tsinghua | 让推理模型学会根据难度决定是否思考 |
| 2025-04 | 🆕 [用 RL 实现简洁推理](https://arxiv.org/abs/2504.05185) `arXiv:2504.05185` |  | PPO 的损失聚合在负奖励下偏好长回答：第二阶段小规模 RL 即可大幅缩短推理 |
| 2025-03 | 🆕 [高效推理综述](https://arxiv.org/abs/2503.16419) `arXiv:2503.16419` | Rice | 缓解过度思考的模型、输出与提示层方法 |
| 2025-03 | 🆕 [L1：用 RL 控制思考长度](https://arxiv.org/abs/2503.04697) `arXiv:2503.04697` | CMU | 提示中给定目标长度、按偏差奖励（LCPO），得到长度可控的推理模型 |
| 2025-02 | 🆕 [训练语言模型高效推理](https://arxiv.org/abs/2502.04463) `arXiv:2502.04463` | CMU | 在 RL 中对正确回答的长度加惩罚，按难度动态分配推理计算 |
| 2024-12 | 🆕 [过度思考](https://arxiv.org/abs/2412.21187) `arXiv:2412.21187` | Tencent/SJTU | o1 类模型在简单题上过度思考，提出效率指标与自训练缓解 |

### 数据、课程与环境（25）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🔥 [多样的 SFT 推理路径改善 RL 后泛化](https://arxiv.org/abs/2609.33780) `arXiv:2609.33780` |  | SFT 数据的推理路径多样性决定 RL 后的泛化：用规则指纹挑选多样路径 |
| 2026-09 | 🔥 [RecreationWorld](https://arxiv.org/abs/2609.22000) `arXiv:2609.22000` |  | 五个平台的可验证环境：智能体需要观察一个运行中的参考实现，自行复现其行为 |
| 2026-09 | 🔥 [NGU：难题永不放弃](https://arxiv.org/abs/2609.13443) `arXiv:2609.13443` |  | 对仍未解出的难题持续追加采样，直到出现正样本 |
| 2026-09 | ★ 🔥 [DataFlex-RL：RLVR 数据策略评测平台](https://arxiv.org/abs/2609.06107) `arXiv:2609.06107` |  | 在同一 GRPO 配方下比较 8 种 rollout 选择与重加权方法：没有一种在统计上显著优于基线 |
| 2026-06 | 🔥 [CLI-Universe](https://arxiv.org/abs/2606.22883) `arXiv:2606.22883` |  | 可验证终端任务的合成引擎；6K 任务微调即显著提升 Terminal-Bench |
| 2026-05 | 🔥 [MobileGym](https://arxiv.org/abs/2605.26114) `arXiv:2605.26114` |  | 浏览器托管、可高度并行的手机使用仿真环境，任务可验证 |
| 2026-05 | 🔥 [OpenComputer](https://arxiv.org/abs/2605.19769) `arXiv:2605.19769` |  | 以验证器为基础构造可验证的软件世界，供计算机使用智能体训练与评测 |
| 2026-05 | 🔥 [GoLongRL：长上下文 RL 配方](https://arxiv.org/abs/2605.19577) `arXiv:2605.19577` |  | 按能力维度构造长上下文可验证任务，开源完整配方 |
| 2026-02 | ◆ [Composition-RL：组合可验证提示](https://arxiv.org/abs/2602.12036) `arXiv:2602.12036` |  | 训练后期大量题目通过率为 1、不再提供信号：把已有题目组合成更难的复合题再做 RL |
| 2026-02 | ◆ [CoBA-RL：按能力分配 rollout 预算](https://arxiv.org/abs/2602.03048) `arXiv:2602.03048` |  | 依据模型当前能力动态分配每题的采样数，把算力用在最有信息量的题上 |
| 2026-02 | ◆ [SWE-Universe](https://arxiv.org/abs/2602.02361) `arXiv:2602.02361` |  | 从 GitHub PR 自动构建百万级可验证软件工程环境，用于智能体中期训练与 RL |
| 2026-02 | ◆ [RLAnything：环境、策略与奖励共同演化](https://arxiv.org/abs/2602.02488) `arXiv:2602.02488` |  | 闭环中同时优化环境（任务难度）、策略和奖励模型，放大学习信号 |
| 2026-01 | ◆ [Endless Terminals](https://arxiv.org/abs/2601.16443) `arXiv:2601.16443` |  | 无人工标注地程序化生成终端任务与验证器，作为终端智能体的 RL 环境 |
| 2026-01 | ◆ [单样本多学科 RL](https://arxiv.org/abs/2601.03111) `arXiv:2601.03111` |  | 精心挑选的单个数学样本即可在物理、化学等多学科上带来提升 |
| 2025-11 | ◆ [RLVE：自适应可验证环境](https://arxiv.org/abs/2511.07317) `arXiv:2511.07317` |  | 400 个程序生成题目、可验证奖励的环境，难度随策略能力自适应调整，联合训练效果优于静态数据 |
| 2025-11 | ◆ [DRIVE：竞赛编程 RLVR 的数据配方](https://arxiv.org/abs/2511.06307) `arXiv:2511.06307` |  | 两阶段：先在大量多样题上扩熵，再在少量难题上大 rollout 预算集中训练 |
| 2025-10 | 🆕 [Webscale-RL](https://arxiv.org/abs/2510.06499) `arXiv:2510.06499` | Salesforce | 把预训练语料自动转为 120 万条可验证问答，RL 数据规模接近预训练级 |
| 2025-09 | 🆕 [VCRL](https://arxiv.org/abs/2509.19803) `arXiv:2509.19803` |  | 按组内奖励方差挑选"难度合适"的样本做课程 |
| 2025-07 | 🆕 [MegaScience](https://arxiv.org/abs/2507.16812) `arXiv:2507.16812` | SJTU | 面向科学推理的大规模后训练数据集 |
| 2025-06 | 🆕 [Guru：跨领域 RL 推理](https://arxiv.org/abs/2506.14965) `arXiv:2506.14965` |  | 6 个领域 9.2 万条可验证样本；数学、代码等常见于预训练的领域靠 RL 激发，逻辑、表格等陌生领域需要真正的领域内训练 |
| 2025-06 | 🆕 [SwS：针对弱点的问题合成](https://arxiv.org/abs/2506.08989) `arXiv:2506.08989` |  | 找出模型在 RL 中持续失败的概念，围绕这些概念合成新题 |
| 2025-05 | 🆕 [Reasoning Gym](https://arxiv.org/abs/2505.24760) `arXiv:2505.24760` |  | 100+ 个程序生成、难度可调、可自动验证的推理环境 |
| 2025-05 | 🆕 [SynLogic](https://arxiv.org/abs/2505.19641) `arXiv:2505.19641` | MiniMax | 35 类逻辑推理任务的可验证数据合成，难度可控 |
| 2025-04 | 🆕 [单样本 RL](https://arxiv.org/abs/2504.20571) `arXiv:2504.20571` | UW/Microsoft | 只用 1 个训练样本做 RLVR，MATH500 提升 30+ 点 |
| 2025-04 | 🆕 [Nemotron-CrossThink](https://arxiv.org/abs/2504.13941) `arXiv:2504.13941` | NVIDIA | 把多领域语料改写为可验证题目并混合，扩展自学习超越数学 |

### 自我博弈与无监督 RL（21）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🔥 [TTPO：测试时策略优化](https://arxiv.org/abs/2608.27448) `arXiv:2608.27448` |  | 测试时：一致的 rollout 用自蒸馏强化、不一致的用组 RL 惩罚 |
| 2026-08 | 🔥 [SPADE：自适应可执行环境中的自我博弈](https://arxiv.org/abs/2608.19197) `arXiv:2608.19197` |  | 同一模型既写长程可执行环境又求解，环境难度随求解者自适应 |
| 2026-08 | 🔥 [Co-RL：多智能体协作中涌现的无监督推理](https://arxiv.org/abs/2608.17253) `arXiv:2608.17253` |  | 多个不共享参数的模型协作训练，无需外部奖励也能涌现推理能力 |
| 2026-07 | 🔥 [技能自我博弈（Skill Self-Play）](https://arxiv.org/abs/2607.22529) `arXiv:2607.22529` |  | 以智能体技能为单元的自我博弈：技能保证可验证的深度执行，路由保持开放性 |
| 2026-05 | 🔥 [SCOPE：开放任务的自我博弈](https://arxiv.org/abs/2605.31433) `arXiv:2605.31433` |  | 挑战者基于文档出题、求解者多轮作答，共同进化，无需数据 |
| 2026-05 | 🔥 [G-Zero：零数据开放生成自我博弈](https://arxiv.org/abs/2605.09959) `arXiv:2605.09959` |  | 无验证器的协同进化框架，给出理想化情形下的次优性保证 |
| 2026-04 | ◆ [TEMPO：测试时训练的扩展](https://arxiv.org/abs/2604.19295) `arXiv:2604.19295` |  | 在无标注题上做策略改进、周期性在有标注集上校准 critic，按 EM 形式化，避免测试时 RL 的漂移 |
| 2026-03 | ★ ◆ [无监督 RLVR 能扩展多远](https://arxiv.org/abs/2603.08660) `arXiv:2603.08660` |  | 各种内在奖励都呈"先升后降"，崩溃时间由模型先验决定而非工程选择：无监督 RL 本质上是锐化已有分布 |
| 2025-10 | 🆕 [SPICE：语料环境中的自我博弈](https://arxiv.org/abs/2510.24684) `arXiv:2510.24684` | Meta | 挑战者从大语料中挖文档出题、推理者解题，以真实文档为锚避免自我博弈的幻觉漂移 |
| 2025-10 | 🆕 [搜索自我博弈（SSP）](https://arxiv.org/abs/2510.18821) `arXiv:2510.18821` |  | 深度搜索智能体同时出题与解题，用检索结果验证题目可解，无需人工标注 |
| 2025-10 | 🆕 [GAR：对抗式 RL 定理证明](https://arxiv.org/abs/2510.11769) `arXiv:2510.11769` |  | 出题者与证明者对抗训练，自动生成难度合适的形式化题目 |
| 2025-09 | 🆕 [语言自我博弈（LSP）](https://arxiv.org/abs/2509.07414) `arXiv:2509.07414` | Meta | 同一模型分饰出题者与解题者做零和博弈，无需额外数据即可提升指令遵循 |
| 2025-08 | 🆕 [变分问题合成自博弈](https://arxiv.org/abs/2508.14029) `arXiv:2508.14029` |  | 从自己的正确解合成新问题，维持 pass@k 不塌缩 |
| 2025-08 | 🆕 [R-Zero](https://arxiv.org/abs/2508.05004) `arXiv:2508.05004` | Tencent | 挑战者出题、求解者解题，零数据的自我进化推理 |
| 2025-06 | 🆕 [SPIRAL](https://arxiv.org/abs/2506.24119) `arXiv:2506.24119` |  | 零和博弈自我对弈激励推理 |
| 2025-06 | 🆕 [置信度即奖励](https://arxiv.org/abs/2506.06395) `arXiv:2506.06395` |  | 以模型自身置信度为奖励，少样本无标注 RL 微调 |
| 2025-05 | 🆕 [无外部奖励学推理（Intuitor）](https://arxiv.org/abs/2505.19590) `arXiv:2505.19590` | UC Berkeley | 以模型自身的置信度（自确定性）为奖励做 GRPO |
| 2025-05 | 🆕 [熵最小化的不合理有效性](https://arxiv.org/abs/2505.15134) `arXiv:2505.15134` | UIUC | 只最小化输出熵（不需要任何标签）就能提升数学与代码推理 |
| 2025-05 | 🆕 [Absolute Zero](https://arxiv.org/abs/2505.03335) `arXiv:2505.03335` | Tsinghua | 零外部数据：模型自己出题、自己解题，用代码执行器验证的自我博弈推理 |
| 2025-04 | 🆕 [TTRL](https://arxiv.org/abs/2504.16084) `arXiv:2504.16084` | Tsinghua/Shanghai AI Lab | 测试时强化学习：用多数投票作伪标签在无标注测试数据上做 RL |
| 2024-04 | [SPAG：对抗性语言游戏自我博弈](https://arxiv.org/abs/2404.10642) `arXiv:2404.10642` | Tencent | "对抗禁忌词"游戏中自我博弈，提升推理 |

### 多模态理解与推理（37）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-05 | 🔥 [SeePhys Pro](https://arxiv.org/abs/2605.09266) `arXiv:2605.09266` |  | 把关键信息逐步从文本移到图像，诊断多模态 RLVR 是否真的用到了视觉（"盲训练"效应） |
| 2026-04 | ◆ [Vero：开放的通用视觉推理 RL 配方](https://arxiv.org/abs/2604.04917) `arXiv:2604.04917` |  | 60 万条、覆盖多种视觉任务的开放 RL 数据与配方，30 个基准上超过已有开放模型 |
| 2026-02 | ◆ [PyVision-RL](https://arxiv.org/abs/2602.20739) `arXiv:2602.20739` |  | 开源多模态模型的智能体 RL：过采样—过滤—排序的 rollout 策略，防止交互轮次塌缩 |
| 2026-02 | ◆ [RL 到底改进了视觉推理的什么](https://arxiv.org/abs/2602.12395) `arXiv:2602.12395` |  | "弗兰肯斯坦式"分析：因果探测、参数比较与模块移植，定位 RL 改进发生在中后层的推理而非感知 |
| 2025-11 | ◆ [VISTA-Gym](https://arxiv.org/abs/2511.19773) `arXiv:2511.19773` |  | 视觉工具集成推理的可扩展 RL 环境，统一多种视觉工具接口 |
| 2025-11 | ◆ [SpatialThinker](https://arxiv.org/abs/2511.07403) `arXiv:2511.07403` |  | 以场景图为依据的空间推理，用多目标的稠密空间奖励做 RL |
| 2025-10 | 🆕 [ThinkMorph](https://arxiv.org/abs/2510.27492) `arXiv:2510.27492` |  | 交错图文思维链中的涌现性质 |
| 2025-10 | 🆕 [统一 RL 与模仿学习训练小 VLM](https://arxiv.org/abs/2510.19307) `arXiv:2510.19307` |  | 学生同时模仿大教师（判别器式奖励）并从可验证奖励学习 |
| 2025-10 | 🆕 [RL 让多模态模型看得更准](https://arxiv.org/abs/2510.16333) `arXiv:2510.16333` |  | 对比 SFT 与 RL 后的视觉编码器：RL 改变了视觉表征本身，定位更准、下游更强 |
| 2025-09 | 🆕 [Mini-o3](https://arxiv.org/abs/2509.07969) `arXiv:2509.07969` | ByteDance 等 | 把多轮看图推理扩展到数十轮：训练时只用少量轮次，测试时能外推 |
| 2025-09 | 🆕 [ReVPT：用 RL 学会视觉工具](https://arxiv.org/abs/2509.01656) `arXiv:2509.01656` |  | 让多模态模型在 GRPO 中学会调用检测、深度等视觉工具来推理 |
| 2025-08 | 🆕 [InternVL3.5](https://arxiv.org/abs/2508.18265) `arXiv:2508.18265` | Shanghai AI Lab | 级联 RL：离线 RL 稳定收敛后再在线 RL 精修 |
| 2025-07 | 🆕 [VisionThink](https://arxiv.org/abs/2507.13348) `arXiv:2507.13348` |  | 先看低分辨率图，模型自己决定是否请求高分辨率：用 RL 学习按需的视觉 token 开销 |
| 2025-07 | 🆕 [LongVILA-R1：把 RL 扩展到长视频](https://arxiv.org/abs/2507.07966) `arXiv:2507.07966` | NVIDIA | 长视频推理的全栈 RL 框架：数据、两阶段训练与面向长序列的并行 rollout 系统 |
| 2025-07 | 🆕 [PAPO：感知感知的策略优化](https://arxiv.org/abs/2507.06448) `arXiv:2507.06448` |  | 多模态 RLVR 的主要错误来自"看错"；加一项"遮住图像后输出应该变"的 KL 项，推动模型真正利用视觉输入 |
| 2025-07 | 🆕 [GLM-4.1V-Thinking / GLM-4.5V](https://arxiv.org/abs/2507.01006) `arXiv:2507.01006` | Zhipu/Tsinghua | 课程采样的可扩展 RL 训练多模态推理模型 |
| 2025-06 | 🆕 [MiMo-VL](https://arxiv.org/abs/2506.03569) `arXiv:2506.03569` | Xiaomi | 混合在策略 RL 的 7B 视觉语言模型 |
| 2025-06 | 🆕 [SynthRL](https://arxiv.org/abs/2506.02096) `arXiv:2506.02096` |  | 自动把种子题改写成更难且答案可验证的变体，扩大视觉推理的 RL 数据 |
| 2025-05 | 🆕 [Pixel Reasoner](https://arxiv.org/abs/2505.15966) `arXiv:2505.15966` |  | 像素空间推理（放大、选帧）+ 好奇心奖励克服"不愿用新动作"的学习陷阱 |
| 2025-05 | ★ 🆕 [DeepEyes：用 RL 激发"看图思考"](https://arxiv.org/abs/2505.14362) `arXiv:2505.14362` | Xiaohongshu 等 | 不需要冷启动 SFT，端到端 RL 学会在推理中调用裁剪/放大工具 |
| 2025-05 | 🆕 [Seed1.5-VL](https://arxiv.org/abs/2505.07062) `arXiv:2505.07062` | ByteDance | 多模态基础模型，后训练混合 RLHF 与 RLVR |
| 2025-04 | 🆕 [NoisyRollout](https://arxiv.org/abs/2504.13055) `arXiv:2504.13055` |  | 用加噪图像产生部分 rollout，增加视觉推理 RL 的探索多样性 |
| 2025-04 | 🆕 [VLM-R1](https://arxiv.org/abs/2504.07615) `arXiv:2504.07615` | OmAI Lab | 稳定、可泛化的 R1 式视觉语言模型 |
| 2025-04 | 🆕 [VL-Rethinker](https://arxiv.org/abs/2504.08837) `arXiv:2504.08837` |  | 选择性样本回放缓解优势消失，并强制"重新思考"以激发反思 |
| 2025-04 | 🆕 [Perception-R1](https://arxiv.org/abs/2504.07954) `arXiv:2504.07954` |  | 把 RL 用于感知任务（定位、计数、检测），以 IoU 等作为可验证奖励 |
| 2025-04 | 🆕 [Kimi-VL](https://arxiv.org/abs/2504.07491) `arXiv:2504.07491` | Moonshot | MoE 视觉语言模型，长 CoT SFT + RL 得到 Thinking 版本 |
| 2025-03 | 🆕 [Video-R1](https://arxiv.org/abs/2503.21776) `arXiv:2503.21776` | CUHK | 把 R1 式 RL 用于视频推理 |
| 2025-03 | 🆕 [VisualPRM](https://arxiv.org/abs/2503.10291) `arXiv:2503.10291` | Shanghai AI Lab | 多模态过程奖励模型与 VisualProcessBench |
| 2025-03 | 🆕 [R1-Onevision](https://arxiv.org/abs/2503.10615) `arXiv:2503.10615` |  | 把图像转为形式化文本表示，再做跨模态的推理 RL |
| 2025-03 | 🆕 [MM-Eureka](https://arxiv.org/abs/2503.07365) `arXiv:2503.07365` | Shanghai AI Lab | 规则奖励 RL 在多模态上复现"顿悟"与回答长度增长 |
| 2025-03 | 🆕 [LMM-R1](https://arxiv.org/abs/2503.07536) `arXiv:2503.07536` |  | 两阶段规则 RL：先在纯文本上增强推理，再迁移到多模态 |
| 2025-03 | 🆕 [Vision-R1](https://arxiv.org/abs/2503.06749) `arXiv:2503.06749` |  | 多模态冷启动 CoT + 渐进式思考抑制的 GRPO |
| 2025-03 | 🆕 [Visual-RFT](https://arxiv.org/abs/2503.01785) `arXiv:2503.01785` | Shanghai AI Lab | 基于可验证视觉奖励的强化微调 |
| 2025-02 | 🆕 [MM-RLHF](https://arxiv.org/abs/2502.10391) `arXiv:2502.10391` |  | 大规模多模态偏好数据与基于批评的奖励模型 |
| 2024-05 | [RLAIF-V](https://arxiv.org/abs/2405.17220) `arXiv:2405.17220` | Tsinghua | 用开源模型的 AI 反馈做多模态对齐 |
| 2023-12 | [RLHF-V](https://arxiv.org/abs/2312.00849) `arXiv:2312.00849` | Tsinghua | 细粒度的人类纠正反馈 + 稠密 DPO，减少多模态幻觉 |
| 2023-09 | [LLaVA-RLHF](https://arxiv.org/abs/2309.14525) `arXiv:2309.14525` | UC Berkeley 等 | 首个多模态 RLHF：用事实增强的奖励模型缓解奖励投机与幻觉 |

### 视觉生成的 RL（29）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [DiffusionART](https://arxiv.org/abs/2610.12572) `arXiv:2610.12572` |  | 分析 NFT 式扩散 RL 目标中隐藏的前向—反向一致性力，以及 EMA 为何能稳定训练，给出可扩展的扩散 RL 配方 |
| 2026-09 | 🔥 [先思考再打分的视觉奖励模型](https://arxiv.org/abs/2609.37372) `arXiv:2609.37372` |  | 先确定每个样例应关注什么，再判断生成结果做得如何 |
| 2026-07 | 🔥 [Read It Back：MLLM 即零样本图像奖励](https://arxiv.org/abs/2607.11886) `arXiv:2607.11886` |  | 让多模态模型"读回"生成图像的描述，以与提示的似然作为免训练奖励 |
| 2026-06 | 🔥 [Qwen-Image-2.0-RL](https://arxiv.org/abs/2606.27608) `arXiv:2606.27608` | Qwen | 文生图模型的后训练流水线：RLHF 与在策略蒸馏结合提升画质与指令遵循 |
| 2026-05 | 🔥 [Flow-OPD](https://arxiv.org/abs/2605.08063) `arXiv:2605.08063` |  | 把 LLM 的在策略蒸馏用到流匹配模型：多个单任务 RL 教师蒸馏进一个学生，缓解多奖励的此消彼长 |
| 2026-04 | ◆ [World-R1](https://arxiv.org/abs/2604.24764) `arXiv:2604.24764` |  | 用 RL 让文生视频满足 3D 一致性约束 |
| 2026-04 | ◆ [RationalRewards](https://arxiv.org/abs/2604.11626) `arXiv:2604.11626` |  | 视觉生成的奖励模型先写多维批评再打分，训练与测试时都能扩展 |
| 2026-04 | ◆ [Sol-RL：FP4 探索、BF16 训练](https://arxiv.org/abs/2604.06916) `arXiv:2604.06916` |  | 扩散 RL 中用 FP4 快速大规模 rollout 挑选候选，再用 BF16 重新生成并训练，兼顾效率与正确性 |
| 2026-03 | ◆ [Gen-Searcher](https://arxiv.org/abs/2603.28767) `arXiv:2603.28767` |  | 用强化学习训练图像生成的 agentic 搜索 |
| 2026-03 | ◆ [UniGRPO：统一图文生成的策略优化](https://arxiv.org/abs/2603.23500) `arXiv:2603.23500` |  | 把"先推理再生成图像"作为一个整体做 RL：文本用 GRPO、图像用流匹配 GRPO |
| 2026-01 | ◆ [DenseGRPO](https://arxiv.org/abs/2601.20218) `arXiv:2601.20218` |  | 流模型对齐的奖励只在终点：估计每个去噪步的贡献，变稀疏奖励为稠密奖励 |
| 2025-12 | ◆ [GARDO：不被奖励投机的扩散 RL](https://arxiv.org/abs/2512.24138) `arXiv:2512.24138` |  | 只对高不确定样本施加门控正则、自适应更新参考模型并鼓励多样性 |
| 2025-12 | ◆ [文生 3D 的强化学习](https://arxiv.org/abs/2512.10949) `arXiv:2512.10949` | Shanghai AI Lab | 系统研究 RL 用于 3D 生成的可行性 |
| 2025-11 | ◆ [视频生成模型是好的潜空间奖励模型](https://arxiv.org/abs/2511.21541) `arXiv:2511.21541` |  | 预训练视频生成模型本就处理任意噪声步的潜变量，可直接改造为潜空间奖励模型，免去解码到像素 |
| 2025-11 | ◆ [图像即自身奖励（Adv-GRPO）](https://arxiv.org/abs/2511.20256) `arXiv:2511.20256` |  | 对抗式奖励：奖励模型与生成器交替更新，用参考图像与视觉基础模型特征抵抗奖励投机 |
| 2025-09 | 🆕 [AWM](https://arxiv.org/abs/2509.25050) `arXiv:2509.25050` |  | 优势加权匹配：让 RL 目标与预训练的流匹配损失形式一致 |
| 2025-09 | 🆕 [DiffusionNFT](https://arxiv.org/abs/2509.16117) `arXiv:2509.16117` | NVIDIA 等 | 在前向（加噪）过程上做在线 RL：用正负样本定义隐式改进方向，免似然、免 CFG，比 FlowGRPO 快 3–25 倍 |
| 2025-09 | 🆕 [RewardDance](https://arxiv.org/abs/2509.08826) `arXiv:2509.08826` | ByteDance | 生成式奖励模型（把"是"的概率当奖励）用于视觉生成，奖励模型规模扩展到 26B 能缓解奖励投机 |
| 2025-08 | 🆕 [Pref-GRPO](https://arxiv.org/abs/2508.20751) `arXiv:2508.20751` |  | 文生图 GRPO 中奖励分数微小差异被归一化放大，导致奖励投机；改用成对偏好胜率作为奖励 |
| 2025-07 | 🆕 [X-Omni](https://arxiv.org/abs/2507.22058) `arXiv:2507.22058` | Tencent | RL 显著改善离散自回归图像生成的质量，使统一图文模型成为可能 |
| 2025-07 | 🆕 [MixGRPO](https://arxiv.org/abs/2507.21802) `arXiv:2507.21802` | Tencent | 流模型 GRPO 只在滑动窗口内用 SDE 采样、其余步用 ODE，训练加速约 70% |
| 2025-05 | 🆕 [DanceGRPO](https://arxiv.org/abs/2505.07818) `arXiv:2505.07818` | ByteDance 等 | 统一的视觉生成 GRPO 框架，覆盖文生图、文生视频与多种奖励 |
| 2025-05 | ★ 🆕 [Flow-GRPO](https://arxiv.org/abs/2505.05470) `arXiv:2505.05470` |  | 把确定性 ODE 采样转为同边际的 SDE 以获得随机性，从而对流匹配模型做在线 GRPO |
| 2025-05 | 🆕 [T2I-R1](https://arxiv.org/abs/2505.00703) `arXiv:2505.00703` |  | 语义级（先写计划）与 token 级 CoT 协同的图像生成 RL |
| 2024-09 | [Adjoint Matching](https://arxiv.org/abs/2409.08861) `arXiv:2409.08861` | Meta | 把奖励微调写成无记忆随机最优控制，流/扩散模型的有理论保证的微调 |
| 2023-11 | [Diffusion-DPO](https://arxiv.org/abs/2311.12908) `arXiv:2311.12908` | Salesforce 等 | 把 DPO 推广到扩散模型的似然下界，用人类偏好对对齐文生图 |
| 2023-05 | [DPOK](https://arxiv.org/abs/2305.16381) `arXiv:2305.16381` | Google 等 | 文生图扩散模型的在线 RL 微调，带 KL 正则 |
| 2023-05 | ★ [DDPO：用 RL 训练扩散模型](https://arxiv.org/abs/2305.13301) `arXiv:2305.13301` | UC Berkeley | 把去噪过程看作多步 MDP，用策略梯度优化美学、可压缩性等奖励 |
| 2023-04 | [ImageReward](https://arxiv.org/abs/2304.05977) `arXiv:2304.05977` | Tsinghua | 文生图的人类偏好奖励模型与基于奖励的微调 |

### 具身与 VLA（13）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | 🔥 [WCM：世界评判模型](https://arxiv.org/abs/2607.29613) `arXiv:2607.29613` |  | VLA 的 critic 缺少时间结构：用视频世界模型提供多帧价值估计 |
| 2026-04 | ◆ [RL Token](https://arxiv.org/abs/2604.23073) `arXiv:2604.23073` | Physical Intelligence | 从 VLA 读出一个紧凑的"RL token"，小型 actor-critic 头在线 RL，几小时内关键阶段提速最高 3 倍 |
| 2026-02 | ◆ [TOPReward](https://arxiv.org/abs/2602.19313) `arXiv:2602.19313` | AI2 | 用 VLM 的 token 概率做零样本、按指令条件的稠密进度奖励 |
| 2025-12 | ◆ [GR-RL](https://arxiv.org/abs/2512.01801) `arXiv:2512.01801` | ByteDance Seed | 把通用 VLA 训成长时程、精密任务（如系鞋带）的专家 |
| 2025-11 | ◆ [SRPO](https://arxiv.org/abs/2511.15605) `arXiv:2511.15605` | Fudan | 自参照策略优化：用模型自己成功的轨迹作参照，给失败轨迹打进度奖励 |
| 2025-11 | ◆ [WMPO：世界模型中的 VLA 策略优化](https://arxiv.org/abs/2511.09515) `arXiv:2511.09515` |  | 在像素级视频世界模型中想象 rollout，做在策略 RL，不与真实环境交互 |
| 2025-10 | 🆕 [π_RL](https://arxiv.org/abs/2510.25889) `arXiv:2510.25889` | 清华 等 | 对流匹配 VLA（π0/π0.5）做在线 RL 微调：处理去噪过程的似然难题 |
| 2025-09 | ★ 🆕 [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) `arXiv:2509.09674` | Tsinghua 等 | GRPO 式结果奖励 RL 用于 VLA，少量示范 + RL 即可大幅提升，并观察到新动作模式"pushcut" |
| 2025-05 | 🆕 [RL 能给 VLA 泛化带来什么](https://arxiv.org/abs/2505.19789) `arXiv:2505.19789` | Tsinghua | 相比 SFT，RL（PPO）在语义与执行层面的分布外泛化明显更好 |
| 2025-05 | 🆕 [VLA-RL](https://arxiv.org/abs/2505.18719) `arXiv:2505.18719` | Tsinghua | 在线 RL 扩展自回归 VLA，配合机器人过程奖励模型 |
| 2025-05 | 🆕 [RIPT-VLA](https://arxiv.org/abs/2505.17016) `arXiv:2505.17016` | UT Austin | 交互式后训练：只用稀疏二值奖励的 VLA 强化微调 |
| 2025-02 | 🆕 [ConRFT](https://arxiv.org/abs/2502.05450) `arXiv:2502.05450` | CASIA | 一致性策略 + 离线/在线强化微调 VLA，真机 45–90 分钟达到高成功率 |
| 2024-09 | [DPPO](https://arxiv.org/abs/2409.00588) `arXiv:2409.00588` | Princeton | 用策略梯度微调扩散策略，把去噪过程纳入 MDP |

### 多轮智能体 RL（107）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [多尺度策略优化](https://arxiv.org/abs/2610.35013) `arXiv:2610.35013` |  | 长程 LLM 智能体的多尺度策略优化：在 token、轮次与回合多个尺度上分配信用 |
| 2026-10 | 🔥 [Harness Scaling](https://arxiv.org/abs/2610.29458) `arXiv:2610.29458` |  | 在多种 harness 上训练让智能体 RL 更可泛化 |
| 2026-10 | 🔥 [Self-HPO：harness 与策略交替优化](https://arxiv.org/abs/2610.22791) `arXiv:2610.22791` |  | 编码智能体交替优化 harness 与策略，用演化中的策略自己提出新的 harness |
| 2026-10 | 🔥 [Mosaic](https://arxiv.org/abs/2610.22174) `arXiv:2610.22174` |  | 多会话的智能体强化学习：跨多个会话分配信用 |
| 2026-10 | 🔥 [DeLoop](https://arxiv.org/abs/2610.19376) `arXiv:2610.19376` |  | 长程智能体常重复已做过的工作：把重复动作当作自监督信号 |
| 2026-10 | 🔥 [WorkRover](https://arxiv.org/abs/2610.18067) `arXiv:2610.18067` |  | 通过递归演化环境扩展长程"协作办公"智能体的训练 |
| 2026-10 | 🔥 [Tev](https://arxiv.org/abs/2610.18868) `arXiv:2610.18868` |  | 用稠密进度奖励扩展长程智能体环境 |
| 2026-10 | 🔥 [ProAS](https://arxiv.org/abs/2610.18683) `arXiv:2610.18683` |  | 进度感知的优势塑形，用于多轮智能体 RL |
| 2026-10 | 🔥 [DeltaTTG](https://arxiv.org/abs/2610.17666) `arXiv:2610.17666` |  | 用"剩余 token 数"进度表训练多轮智能体 |
| 2026-10 | 🔥 [Agent-Reflex](https://arxiv.org/abs/2610.17953) `arXiv:2610.17953` |  | 教语言智能体反思式行动：在关键步骤先反思再执行 |
| 2026-10 | 🔥 [EAU](https://arxiv.org/abs/2610.16523) `arXiv:2610.16523` |  | 环境—智能体—用户协同演化，训练以用户为中心的智能体 |
| 2026-10 | 🔥 [ViSkill](https://arxiv.org/abs/2610.12403) `arXiv:2610.12403` |  | 用演化的视觉原生技能强化 VLM 智能体 |
| 2026-10 | 🔥 [AgentGarten](https://arxiv.org/abs/2610.12374) `arXiv:2610.12374` |  | 由代码生成的交互世界，供智能体探索与演化 |
| 2026-10 | 🔥 [SkillForge](https://arxiv.org/abs/2610.09832) `arXiv:2610.09832` |  | 技能与智能体协同演化：动态的技能生命周期 |
| 2026-10 | 🔥 [NeMo-DCR](https://arxiv.org/abs/2610.08430) `arXiv:2610.08430` | NVIDIA | 万亿参数规模智能体 RL 中按位精确的增量压缩权重同步 |
| 2026-10 | 🔥 [智能体语言世界模型](https://arxiv.org/abs/2610.06100) `arXiv:2610.06100` |  | 从交互轨迹构建语言世界模型来模拟交互环境 |
| 2026-10 | 🔥 [ASCENT](https://arxiv.org/abs/2610.05303) `arXiv:2610.05303` |  | 通过自蒸馏已验证经验，做长程智能体的在线测试时训练 |
| 2026-09 | 🔥 [从给定证据到收集证据](https://arxiv.org/abs/2609.39566) `arXiv:2609.39566` |  | 纵向医疗推理的智能体学习：通过工具 harness 主动收集证据 |
| 2026-09 | 🔥 [ReSAIL](https://arxiv.org/abs/2609.39306) `arXiv:2609.39306` |  | 缓解迭代式智能体自蒸馏中的坍缩 |
| 2026-09 | 🔥 [QwenGyre](https://arxiv.org/abs/2609.33848) `arXiv:2609.33848` | Alibaba Qwen | 面向超长程智能体的弹性强化学习训练框架 |
| 2026-09 | 🔥 [CompoWorld](https://arxiv.org/abs/2609.33665) `arXiv:2609.33665` |  | 组合式环境扩展训练通用智能体 |
| 2026-09 | 🔥 [X-Tree](https://arxiv.org/abs/2609.32993) `arXiv:2609.32993` |  | 把可复用经验 token 化，提高智能体泛化效率 |
| 2026-09 | 🔥 [从已解机制生成 RL 环境](https://arxiv.org/abs/2609.27321) `arXiv:2609.27321` |  | 可验证的隐藏动力学游戏作为智能体 RL 环境 |
| 2026-09 | 🔥 [不要掩盖环境](https://arxiv.org/abs/2609.20715) `arXiv:2609.20715` |  | SFT 时对环境观测也施加监督，会改变智能体在 RL 中的探索 |
| 2026-09 | 🔥 [RSIAgent：新环境中的递归自我改进](https://arxiv.org/abs/2609.15364) `arXiv:2609.15364` |  | 免训练的多智能体框架：自主探索新环境并构建记忆以递归改进 |
| 2026-09 | 🔥 [环境即脚手架](https://arxiv.org/abs/2609.08404) `arXiv:2609.08404` |  | 丰富环境反馈以自举长程自演化智能体 |
| 2026-09 | 🔥 [NeoHorse-1](https://arxiv.org/abs/2609.08183) `arXiv:2609.08183` |  | 通过带路由 harness 的智能体后训练实现递归自我改进 |
| 2026-09 | 🔥 [DRACO](https://arxiv.org/abs/2609.04094) `arXiv:2609.04094` |  | 动态评分细则带来的细粒度信用分配，用于长程智能体训练 |
| 2026-08 | 🔥 [WHALE](https://arxiv.org/abs/2609.00196) `arXiv:2609.00196` |  | harness 与权重联合优化的简单配方 |
| 2026-08 | 🔥 [ContextPilot](https://arxiv.org/abs/2608.28476) `arXiv:2608.28476` |  | 用细粒度 RL 教智能体主动管理上下文 |
| 2026-08 | 🔥 [Agent-G²](https://arxiv.org/abs/2608.23318) `arXiv:2608.23318` |  | 高斯引导的提示式智能体 RL |
| 2026-08 | 🔥 [EnvHarness](https://arxiv.org/abs/2608.19880) `arXiv:2608.19880` |  | 唤醒静态世界：把静态资源变成智能体可学习的环境 |
| 2026-08 | 🔥 [Agentic ESOpt](https://arxiv.org/abs/2608.17310) `arXiv:2608.17310` |  | 低显存微调长程 LLM 智能体（进化策略） |
| 2026-08 | 🔥 [Agent Lightning v1.0](https://arxiv.org/abs/2608.17528) `arXiv:2608.17528` | Microsoft | 面向 harness 的智能体 RL 框架：训练与执行解耦 |
| 2026-08 | 🔥 [ClawGym II](https://arxiv.org/abs/2608.16798) `arXiv:2608.16798` |  | 对智能体 harness 做黑盒强化学习 |
| 2026-08 | 🔥 [SKILLER](https://arxiv.org/abs/2608.10538) `arXiv:2608.10538` |  | 小模型的语言级 RL 技能抽取 |
| 2026-08 | 🔥 [EnvACE](https://arxiv.org/abs/2608.06197) `arXiv:2608.06197` |  | 通过"世界彩排"内化环境动力学的智能体 RL |
| 2026-08 | 🔥 [RL 渐进生成技能](https://arxiv.org/abs/2608.01678) `arXiv:2608.01678` |  | 用强化学习渐进地生成智能体技能 |
| 2026-07 | 🔥 [SkillRise](https://arxiv.org/abs/2607.26784) `arXiv:2607.26784` |  | 跨任务技能演化的智能体强化学习 |
| 2026-07 | 🔥 [CAST](https://arxiv.org/abs/2607.25308) `arXiv:2607.25308` |  | 游戏求解器作为轮次级教师训练 LLM 智能体 |
| 2026-07 | 🔥 [从 RLVR 到 RLSVR](https://arxiv.org/abs/2607.23802) `arXiv:2607.23802` |  | 任务变换产生可自我验证的奖励，用于开放式自我改进 |
| 2026-07 | 🔥 [从智能体经验中高效学习](https://arxiv.org/abs/2607.21051) `arXiv:2607.21051` |  | 环境交互昂贵时的样本高效学习 |
| 2026-07 | 🔥 [Molt](https://arxiv.org/abs/2607.21653) `arXiv:2607.21653` |  | PyTorch 原生、可扩展的智能体 RL 训练框架 |
| 2026-07 | 🔥 [单 rollout 异步优化](https://arxiv.org/abs/2607.07508) `arXiv:2607.07508` |  | 智能体 RL 的单 rollout 异步优化 |
| 2026-07 | 🔥 [AutoMem](https://arxiv.org/abs/2607.01224) `arXiv:2607.01224` |  | 把记忆当作可学习的认知技能 |
| 2026-06 | 🔥 [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) `arXiv:2606.24597` | Alibaba Qwen | 面向通用智能体的语言世界模型：预测环境动力学以训练与规划 |
| 2026-06 | 🔥 [OPD-Evolver](https://arxiv.org/abs/2606.17628) `arXiv:2606.17628` |  | 通过在策略蒸馏培养整体的智能体演化器 |
| 2026-06 | 🔥 [APPO](https://arxiv.org/abs/2606.12384) `arXiv:2606.12384` |  | 智能体程序性策略优化 |
| 2026-06 | 🔥 [TRACE](https://arxiv.org/abs/2606.11119) `arXiv:2606.11119` |  | 智能体 RL 中统一的 rollout 预算分配 |
| 2026-05 | 🔥 [智能体探索式策略优化](https://arxiv.org/abs/2605.28774) `arXiv:2605.28774` |  | 多模态智能体推理的探索式策略优化 |
| 2026-05 | 🔥 [Skill0.5](https://arxiv.org/abs/2605.28424) `arXiv:2605.28424` |  | 技能内化与使用联合训练，提升分布外泛化 |
| 2026-05 | 🔥 [Spreadsheet-RL](https://arxiv.org/abs/2605.22642) `arXiv:2605.22642` |  | 通过 RL 提升智能体处理真实电子表格任务的能力 |
| 2026-05 | 🔥 [Orchard](https://arxiv.org/abs/2605.15040) `arXiv:2605.15040` |  | 开源的智能体建模框架 |
| 2026-05 | 🔥 [StraTA](https://arxiv.org/abs/2605.06642) `arXiv:2605.06642` |  | 以策略性轨迹抽象激励智能体 RL |
| 2026-05 | 🔥 [Skill1](https://arxiv.org/abs/2605.06130) `arXiv:2605.06130` |  | 通过 RL 统一演化技能增强型智能体 |
| 2026-05 | 🔥 [A²TGPO：轮次组策略优化](https://arxiv.org/abs/2605.06200) `arXiv:2605.06200` |  | 以信息增益为内在信号，重新设计其归一化、累积与轮次级裁剪 |
| 2026-05 | 🔥 [Odysseus：100+ 轮的 VLM 游戏决策](https://arxiv.org/abs/2605.00347) `arXiv:2605.00347` |  | 把 VLM 智能体 RL 扩展到上百轮的游戏决策，开放训练框架 |
| 2026-05 | 🔥 [AEM](https://arxiv.org/abs/2605.00425) `arXiv:2605.00425` |  | 多轮智能体 RL 的自适应熵调节 |
| 2026-04 | ◆ [ClawGym](https://arxiv.org/abs/2604.26904) `arXiv:2604.26904` |  | 构建有效 Claw 类智能体的可扩展框架 |
| 2026-04 | ◆ [Agent-World](https://arxiv.org/abs/2604.18292) `arXiv:2604.18292` |  | 扩展真实环境合成，演化通用智能体 |
| 2026-04 | ◆ [智能体的探索与利用错误可以度量](https://arxiv.org/abs/2604.13151) `arXiv:2604.13151` |  | 设计可控环境把智能体失败分解为探索错误与利用错误，并比较不同记忆 harness |
| 2026-04 | ◆ [RAGEN-2](https://arxiv.org/abs/2604.06268) `arXiv:2604.06268` |  | 智能体 RL 中的推理坍缩 |
| 2026-04 | ◆ [SKILL0](https://arxiv.org/abs/2604.02268) `arXiv:2604.02268` |  | 上下文内智能体 RL 实现技能内化 |
| 2026-03 | ◆ [LongCat-Flash-Prover](https://arxiv.org/abs/2603.21065) `arXiv:2603.21065` | Meituan | 560B 开源模型：智能体式工具集成 RL 做形式化推理 |
| 2026-03 | ◆ [互补 RL：经验驱动的智能体学习](https://arxiv.org/abs/2603.17621) `arXiv:2603.17621` |  | 经验抽取器与策略在 RL 循环中协同进化，把轨迹中的经验提炼为可复用的知识 |
| 2026-03 | ◆ [OpenClaw-RL](https://arxiv.org/abs/2603.10165) `arXiv:2603.10165` |  | "说话就能训练智能体"：把用户回复当作下一状态信号 |
| 2026-02 | ◆ [探索式记忆增强智能体](https://arxiv.org/abs/2602.23008) `arXiv:2602.23008` |  | 在策略与离策略混合优化的探索式智能体 |
| 2026-02 | ◆ [ARLArena：稳定的智能体 RL](https://arxiv.org/abs/2602.21534) `arXiv:2602.21534` |  | 统一框架系统分析智能体 RL 不稳定的来源，提出 SAMPO |
| 2026-02 | ◆ [Calibrate-Then-Act：代价感知的探索](https://arxiv.org/abs/2602.16699) `arXiv:2602.16699` |  | 让智能体显式推理"再探索一步的代价"与不确定性，再决定行动 |
| 2026-02 | ◆ [Agent World Model](https://arxiv.org/abs/2602.10090) `arXiv:2602.10090` |  | 无限合成环境用于智能体 RL |
| 2026-02 | ◆ [弱者驱动学习](https://arxiv.org/abs/2602.08222) `arXiv:2602.08222` |  | 弱智能体如何让强智能体更强 |
| 2026-02 | ◆ [SkillRL](https://arxiv.org/abs/2602.08234) `arXiv:2602.08234` |  | 递归的技能增强强化学习演化智能体 |
| 2026-02 | ◆ [RWML：强化世界模型学习](https://arxiv.org/abs/2602.05842) `arXiv:2602.05842` |  | 用自监督奖励让 LLM 智能体学会预测动作后的环境状态（世界模型），再用于决策 |
| 2026-01 | ◆ [ASTRA](https://arxiv.org/abs/2601.21558) `arXiv:2601.21558` |  | 自动合成智能体轨迹与强化学习竞技场 |
| 2026-01 | ◆ [Spark：关键状态动态分支](https://arxiv.org/abs/2601.20209) `arXiv:2601.20209` |  | 长程智能体只在关键决策状态分支采样，把 rollout 预算花在刀刃上 |
| 2026-01 | ◆ [ArenaRL](https://arxiv.org/abs/2601.06487) `arXiv:2601.06487` |  | 用锦标赛式相对排名扩展开放式智能体的 RL |
| 2026-01 | ◆ [AT²PO：树搜索的轮次级策略优化](https://arxiv.org/abs/2601.04767) `arXiv:2601.04767` |  | 多轮智能体 RL 中按轮次建树：熵引导扩展、轮次级信用分配与轮次级重要性采样 |
| 2025-12 | ◆ [Youtu-Agent](https://arxiv.org/abs/2512.24615) `arXiv:2512.24615` | Tencent | 自动生成与混合策略优化扩展智能体生产力 |
| 2025-12 | ◆ [ROME / 开放智能体学习生态](https://arxiv.org/abs/2512.24873) `arXiv:2512.24873` |  | 在开放智能体学习生态中构建 ROME 模型 |
| 2025-12 | ◆ [SmartSnap](https://arxiv.org/abs/2512.22322) `arXiv:2512.22322` |  | 主动寻找证据的自验证智能体 |
| 2025-12 | ◆ [带技能库的自我改进 RL](https://arxiv.org/abs/2512.17102) `arXiv:2512.17102` |  | 用强化学习训练带技能库的自我改进智能体 |
| 2025-12 | ◆ [Turn-PPO](https://arxiv.org/abs/2512.17008) `arXiv:2512.17008` |  | 以轮次而非 token 为 MDP 步骤估计优势，多轮智能体 PPO 更稳 |
| 2025-12 | ◆ [Nex-N1](https://arxiv.org/abs/2512.04987) `arXiv:2512.04987` |  | 以统一生态大规模构建环境训练的智能体模型 |
| 2025-11 | ◆ [Agent-R1](https://arxiv.org/abs/2511.14460) `arXiv:2511.14460` |  | 模块化的端到端智能体 RL 框架：把 MDP 形式化推广到多轮工具交互 |
| 2025-11 | ◆ [通过经验合成扩展智能体学习](https://arxiv.org/abs/2511.03773) `arXiv:2511.03773` | Meta | 用合成经验扩展智能体 RL |
| 2025-10 | 🆕 [IGPO](https://arxiv.org/abs/2510.14967) `arXiv:2510.14967` |  | 基于信息增益的多轮智能体策略优化 |
| 2025-10 | 🆕 [AEPO](https://arxiv.org/abs/2510.14545) `arXiv:2510.14545` |  | 智能体熵平衡策略优化 |
| 2025-10 | 🆕 [揭秘智能体推理中的 RL](https://arxiv.org/abs/2510.11701) `arXiv:2510.11701` |  | 智能体推理中强化学习的数据、算法与推理模式 |
| 2025-10 | 🆕 [不只微调智能体，还要调环境](https://arxiv.org/abs/2510.10197) `arXiv:2510.10197` |  | 以环境调优训练多轮工具智能体 |
| 2025-10 | 🆕 [早期经验](https://arxiv.org/abs/2510.08558) `arXiv:2510.08558` | Meta | 智能体用自己行动产生的未来状态作监督（无奖励），介于模仿学习与 RL 之间 |
| 2025-09 | 🆕 [树搜索智能体 RL](https://arxiv.org/abs/2509.21240) `arXiv:2509.21240` |  | 用树搜索做 LLM 智能体强化学习 |
| 2025-09 | 🆕 [环境扩展](https://arxiv.org/abs/2509.13311) `arXiv:2509.13311` | Alibaba Tongyi | 通过扩展环境走向通用智能体智能 |
| 2025-09 | 🆕 [熵调制策略梯度](https://arxiv.org/abs/2509.09265) `arXiv:2509.09265` |  | 长程 LLM 智能体的熵调制策略梯度 |
| 2025-09 | 🆕 [AgentGym-RL](https://arxiv.org/abs/2509.08755) `arXiv:2509.08755` |  | 多轮 RL 训练长程决策智能体 |
| 2025-08 | 🆕 [AWorld](https://arxiv.org/abs/2508.20404) `arXiv:2508.20404` | Ant Group | 编排智能体 AI 的训练配方 |
| 2025-08 | 🆕 [Agent Lightning](https://arxiv.org/abs/2508.03680) `arXiv:2508.03680` | Microsoft | 用强化学习训练任意 AI 智能体的框架 |
| 2025-06 | 🆕 [MEM1：记忆与推理协同](https://arxiv.org/abs/2506.15841) `arXiv:2506.15841` | MIT 等 | 每轮把历史压缩进内部状态再继续，用 RL 学会恒定显存的长程智能体 |
| 2025-06 | 🆕 [思考与行动：扩展测试时交互](https://arxiv.org/abs/2506.07976) `arXiv:2506.07976` | CMU 等 | 智能体与其想得更长，不如在环境中做更多交互；用课程式在线 RL 扩展交互轮数 |
| 2025-05 | 🆕 [ARIA](https://arxiv.org/abs/2506.00539) `arXiv:2506.00539` |  | 意图驱动的奖励聚合训练语言智能体 |
| 2025-05 | ★ 🆕 [GiGPO：组中组策略优化](https://arxiv.org/abs/2505.10978) `arXiv:2505.10978` | NTU 等 | 在回合级分组之外，再把不同轨迹中出现的相同状态归为一组，得到步骤级优势，不需要 critic |
| 2025-04 | ★ 🆕 [RAGEN（StarPO）](https://arxiv.org/abs/2504.20073) `arXiv:2504.20073` | Northwestern 等 | 多轮智能体 RL 的"回声陷阱"：奖励方差塌缩与梯度尖峰；提出轨迹级优化 StarPO |
| 2025-03 | 🆕 [SWEET-RL](https://arxiv.org/abs/2503.15478) `arXiv:2503.15478` | Meta | 协作任务上让 critic 看到训练时的特权信息（如参考答案），给出轮次级优势 |
| 2024-03 | [ETO：试错学习](https://arxiv.org/abs/2403.02502) `arXiv:2403.02502` |  | 智能体探索收集失败轨迹，与专家轨迹构成对比对做 DPO |
| 2024-02 | [ArCHer](https://arxiv.org/abs/2402.19446) `arXiv:2402.19446` | UC Berkeley | 分层多轮强化学习训练语言模型智能体 |
| 2023-11 | [LMRL-Gym](https://arxiv.org/abs/2311.18232) `arXiv:2311.18232` | UC Berkeley | 面向语言模型多轮 RL 的基准：20 题、迷宫、谈判等需要多步信用分配的任务 |
| 2023-03 | [Reflexion](https://arxiv.org/abs/2303.11366) `arXiv:2303.11366` | Northeastern/MIT/Princeton | 用语言形式的自我反思作为强化信号 |
| 2022-10 | [ReAct](https://arxiv.org/abs/2210.03629) `arXiv:2210.03629` | Princeton/Google | 推理与行动交替：思考、调用工具、观察，再思考 |

### 工具集成推理（18）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [X-Ragent](https://arxiv.org/abs/2610.16811) `arXiv:2610.16811` |  | 工具增强 RL 做有证据支撑的胸片推理 |
| 2026-09 | 🔥 [WEFT](https://arxiv.org/abs/2609.36887) `arXiv:2609.36887` |  | 扩展通用智能体的工具使用后训练 |
| 2026-08 | 🔥 [DART-SD](https://arxiv.org/abs/2608.18524) `arXiv:2608.18524` |  | 多轮工具调用的自蒸馏：菱形拓扑感知的检索与调优 |
| 2026-06 | 🔥 [多步工具 RL 为何坍缩](https://arxiv.org/abs/2606.26027) `arXiv:2606.26027` |  | 纯 RL 训练多步工具调用常坍缩，加入监督信号可修复 |
| 2026-05 | 🔥 [ParaVT](https://arxiv.org/abs/2605.20342) `arXiv:2605.20342` |  | 智能体视频 RL 中的并行工具使用 |
| 2026-05 | 🔥 [EnvFactory](https://arxiv.org/abs/2605.18703) `arXiv:2605.18703` |  | 合成可执行环境并稳健 RL，扩展工具使用智能体 |
| 2026-03 | ◆ [工具使用的上下文 RL](https://arxiv.org/abs/2603.08068) `arXiv:2603.08068` |  | LLM 在上下文中通过强化学习学会用工具 |
| 2025-10 | 🆕 [多智能体工具集成策略优化](https://arxiv.org/abs/2510.04678) `arXiv:2510.04678` |  | 单模型多角色的多智能体工具集成 RL |
| 2025-09 | 🆕 [SimpleTIR](https://arxiv.org/abs/2509.02479) `arXiv:2509.02479` |  | 多轮工具集成推理中，工具返回的分布外观测会诱发低概率 token 与梯度爆炸；过滤掉含"空轮"的轨迹即可稳定训练 |
| 2025-09 | 🆕 [VerlTool](https://arxiv.org/abs/2509.01055) `arXiv:2509.01055` |  | 面向工具使用的整体智能体 RL 框架 |
| 2025-08 | 🆕 [rStar2-Agent](https://arxiv.org/abs/2508.20722) `arXiv:2508.20722` | Microsoft | 14B 数学推理模型的智能体 RL 技术报告 |
| 2025-07 | 🆕 [ARPO](https://arxiv.org/abs/2507.19849) `arXiv:2507.19849` |  | 智能体强化策略优化：在工具调用后的高熵步骤上分支采样 |
| 2025-05 | 🆕 [视觉智能体强化微调](https://arxiv.org/abs/2505.14246) `arXiv:2505.14246` |  | 让视觉模型学会用工具（裁剪、搜索）思考 |
| 2025-04 | 🆕 [ARTIST](https://arxiv.org/abs/2505.01441) `arXiv:2505.01441` | Microsoft | 用强化学习做智能体推理与工具集成 |
| 2025-04 | 🆕 [ToolRL](https://arxiv.org/abs/2504.13958) `arXiv:2504.13958` |  | 工具学习只需奖励 |
| 2025-04 | 🆕 [ReTool](https://arxiv.org/abs/2504.11536) `arXiv:2504.11536` | ByteDance | 用 RL 学习何时、如何调用代码工具 |
| 2025-03 | 🆕 [ToRL：工具集成 RL](https://arxiv.org/abs/2503.23383) `arXiv:2503.23383` |  | 从基座直接用 RL 学会调用代码解释器解数学题 |
| 2023-02 | [Toolformer](https://arxiv.org/abs/2302.04761) `arXiv:2302.04761` | Meta | 让模型自监督地标注何时调用 API：保留那些能降低后续 token 损失的调用 |

### 搜索与深度研究（24）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🔥 [ABSeeker](https://arxiv.org/abs/2608.05102) `arXiv:2608.05102` |  | 以答案回溯的信用分配训练长程搜索智能体 |
| 2026-07 | 🔥 [DeepSearch-World](https://arxiv.org/abs/2607.07820) `arXiv:2607.07820` |  | 在可验证环境中对深度搜索智能体做自蒸馏 |
| 2026-06 | 🔥 [Harness-1](https://arxiv.org/abs/2606.02373) `arXiv:2606.02373` |  | 用"外化状态"的 harness 训练搜索智能体，而非在越来越长的对话上训练 |
| 2026-05 | 🔥 [LongTraceRL](https://arxiv.org/abs/2605.31584) `arXiv:2605.31584` |  | 从搜索智能体轨迹中学习长上下文推理 |
| 2026-05 | 🔥 [SAAS](https://arxiv.org/abs/2605.29796) `arXiv:2605.29796` |  | 自我感知的 RL 缓解智能体搜索中的过度搜索 |
| 2026-05 | 🔥 [GrepSeek](https://arxiv.org/abs/2605.29307) `arXiv:2605.29307` |  | 训练直接与语料交互（类 grep）的搜索智能体 |
| 2026-05 | 🔥 [HyperEyes](https://arxiv.org/abs/2605.07177) `arXiv:2605.07177` |  | 并行多模态搜索智能体的效率感知 RL |
| 2026-02 | ◆ [WideSeek-R1](https://arxiv.org/abs/2602.04634) `arXiv:2602.04634` |  | 用多智能体 RL 探索广度信息搜索的"宽度扩展" |
| 2026-01 | ◆ [BAPO：边界感知的搜索 RL](https://arxiv.org/abs/2601.11037) `arXiv:2601.11037` |  | 搜索智能体很少回答"我不知道"：奖励在能力边界之外正确地拒答，提高可靠性 |
| 2026-01 | ◆ [引用感知评分细则的深度搜索 RL](https://arxiv.org/abs/2601.06021) `arXiv:2601.06021` |  | 用引用感知的评分细则稳健训练深度搜索智能体 |
| 2025-12 | ◆ [SenseNova-MARS](https://arxiv.org/abs/2512.24330) `arXiv:2512.24330` | SenseTime | 以强化学习增强多模态智能体推理与搜索 |
| 2025-11 | ◆ [DR Tulu](https://arxiv.org/abs/2511.19399) `arXiv:2511.19399` | Ai2 | 用演化的评分细则做深度研究 RL |
| 2025-10 | ★ 🆕 [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) `arXiv:2510.24701` | Alibaba Tongyi | 从智能体中期训练到 RL 的端到端深度研究智能体 |
| 2025-09 | 🆕 [WebSailor-V2](https://arxiv.org/abs/2509.13305) `arXiv:2509.13305` | Alibaba Tongyi | 用合成数据与可扩展 RL 缩小与闭源智能体的差距 |
| 2025-09 | 🆕 [WebExplorer](https://arxiv.org/abs/2509.06501) `arXiv:2509.06501` |  | 探索与演化训练长程网页智能体 |
| 2025-08 | 🆕 [SSRL：自搜索 RL](https://arxiv.org/abs/2508.10874) `arXiv:2508.10874` |  | 让模型用自身知识模拟搜索引擎做 RL，训练出的搜索能力可迁移到真实搜索 |
| 2025-08 | 🆕 [超越十轮](https://arxiv.org/abs/2508.07976) `arXiv:2508.07976` |  | 大规模异步 RL 解锁长程智能体搜索 |
| 2025-07 | 🆕 [WebSailor](https://arxiv.org/abs/2507.02592) `arXiv:2507.02592` | Alibaba Tongyi | 面向超人推理的网页智能体 |
| 2025-05 | 🆕 [WebDancer](https://arxiv.org/abs/2505.22648) `arXiv:2505.22648` | Alibaba | 信息检索智能体：数据合成 → 拒绝采样微调 → RL 的四阶段配方 |
| 2025-05 | 🆕 [ZeroSearch](https://arxiv.org/abs/2505.04588) `arXiv:2505.04588` | Alibaba | 不调用真实搜索引擎也能训练搜索能力（用 LLM 模拟搜索） |
| 2025-04 | 🆕 [DeepResearcher](https://arxiv.org/abs/2504.03160) `arXiv:2504.03160` | SJTU | 在真实网络环境中用 RL 扩展深度研究 |
| 2025-03 | 🆕 [Search-R1](https://arxiv.org/abs/2503.09516) `arXiv:2503.09516` | UIUC | 用强化学习训练模型边推理边调用搜索引擎 |
| 2025-03 | 🆕 [R1-Searcher](https://arxiv.org/abs/2503.05592) `arXiv:2503.05592` | Renmin | 用 RL 激发 LLM 的搜索能力 |
| 2025-02 | 🆕 [Deep Research](https://openai.com/index/introducing-deep-research/) | OpenAI | 以 RL 训练的多步网页研究智能体，生成带引用的报告 |

### 代码与软件工程（22）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [训练整个智能体而不只是模型](https://arxiv.org/abs/2610.28069) `arXiv:2610.28069` |  | 训练编码智能体时同时更新 harness，而非固定 harness |
| 2026-10 | 🔥 [Lean4AgenticRL](https://arxiv.org/abs/2610.22448) `arXiv:2610.22448` |  | 可验证的多层过程奖励推进软件工程智能体 RL |
| 2026-10 | 🔥 [ReTerm](https://arxiv.org/abs/2610.18484) `arXiv:2610.18484` |  | 终端智能体的恢复训练：学会从错误状态中恢复 |
| 2026-10 | 🔥 [FC-SWE：以失败为条件的 SWE RL](https://arxiv.org/abs/2610.07898) `arXiv:2610.07898` |  | 把从失败状态恢复的尝试纳入策略训练，提升长程 SWE 智能体 |
| 2026-09 | 🔥 [组内智能体评分与优势重分配](https://arxiv.org/abs/2609.32577) `arXiv:2609.32577` |  | 代码智能体 RL 中超越二值测试奖励的评分 |
| 2026-09 | 🔥 [按类别迭代专家训练](https://arxiv.org/abs/2609.23377) `arXiv:2609.23377` |  | 仓库级软件工程任务异质：分类训练专家再合并 |
| 2026-09 | 🔥 [CodeMidas](https://arxiv.org/abs/2609.22068) `arXiv:2609.22068` |  | 从代码本身扩展智能体编码 RL 环境 |
| 2026-09 | 🔥 [T1](https://arxiv.org/abs/2609.11042) `arXiv:2609.11042` |  | 面向长程任务的终端智能体强化学习 |
| 2026-08 | 🔥 [LEGO-RL](https://arxiv.org/abs/2608.17393) `arXiv:2608.17393` |  | 面向编码智能体的 harness 原生强化学习 |
| 2026-06 | 🔥 [验证视野](https://arxiv.org/abs/2606.26300) `arXiv:2606.26300` |  | 编码智能体奖励没有银弹：验证并不总比生成容易 |
| 2026-04 | ◆ [GrandCode](https://arxiv.org/abs/2604.02721) `arXiv:2604.02721` |  | 通过智能体 RL 达到竞赛编程特级大师水平 |
| 2026-03 | ◆ [Code-A1：代码与测试对抗演化](https://arxiv.org/abs/2603.15611) `arXiv:2603.15611` |  | 代码模型与测试模型目标相反、对抗地共同训练 |
| 2026-02 | ◆ [CUDA Agent](https://arxiv.org/abs/2602.24286) `arXiv:2602.24286` |  | 大规模智能体 RL 生成高性能 CUDA 内核 |
| 2026-02 | ◆ [Dr. Kernel：Triton 内核的多轮 RL](https://arxiv.org/abs/2602.05885) `arXiv:2602.05885` |  | 提出轮次级留一法（TRLOO）得到无偏的多轮优势，并针对内核任务防止奖励投机（如偷懒的实现） |
| 2025-08 | 🆕 [RL 训练长上下文多轮 SWE 智能体](https://arxiv.org/abs/2508.03501) `arXiv:2508.03501` | Nebius | 用强化学习训练长上下文、多轮软件工程智能体 |
| 2025-07 | 🆕 [CUDA-L1](https://arxiv.org/abs/2507.14111) `arXiv:2507.14111` |  | 对比式 RL 优化 CUDA 内核：把已有内核及其加速比放进提示，让模型比较后改进 |
| 2025-07 | 🆕 [Kevin：多轮 RL 生成 CUDA 内核](https://arxiv.org/abs/2507.11948) `arXiv:2507.11948` |  | 多轮 RL 学会根据编译与性能反馈迭代改进内核 |
| 2025-06 | 🆕 [CURE：代码与单测协同演化](https://arxiv.org/abs/2506.03136) `arXiv:2506.03136` |  | 编码器与单测生成器互相提供奖励，无需真值代码 |
| 2025-04 | 🆕 [SWE-smith](https://arxiv.org/abs/2504.21798) `arXiv:2504.21798` | Stanford | 自动在真实仓库中注入 bug、生成任务实例，扩展 SWE 智能体训练数据 |
| 2025-04 | 🆕 [R2E-Gym](https://arxiv.org/abs/2504.07164) `arXiv:2504.07164` | UC Berkeley | 程序化构造可执行的软件工程环境，结合执行与非执行验证器 |
| 2025-02 | 🆕 [SWE-RL](https://arxiv.org/abs/2502.18449) `arXiv:2502.18449` | Meta | 在开源软件演化数据上用 RL 训练软件工程推理 |
| 2024-12 | 🆕 [SWE-Gym](https://arxiv.org/abs/2412.21139) `arXiv:2412.21139` | UC Berkeley | 训练软件工程智能体与验证器的环境 |

### GUI 与计算机使用（21）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [MAI-Gym](https://arxiv.org/abs/2610.20939) `arXiv:2610.20939` |  | 一体化的 GUI 智能体评测与在线 RL 训练平台，覆盖手机与电脑 |
| 2026-10 | 🔥 [SoftwareGym](https://arxiv.org/abs/2610.12611) `arXiv:2610.12611` |  | 合成多样软件环境以扩展计算机使用智能体的训练 |
| 2026-06 | 🔥 [训练开放的手机使用智能体](https://arxiv.org/abs/2606.23049) `arXiv:2606.23049` |  | 为手机使用训练开放模型 |
| 2026-06 | 🔥 [MobileForge](https://arxiv.org/abs/2606.19930) `arXiv:2606.19930` |  | 免标注的手机 GUI 智能体适配 |
| 2026-06 | 🔥 [OpenWebRL](https://arxiv.org/abs/2606.02031) `arXiv:2606.02031` |  | 揭示视觉网页智能体在线多轮 RL 的关键配方 |
| 2026-05 | 🔥 [CUA-Gym](https://arxiv.org/abs/2605.25624) `arXiv:2605.25624` |  | 扩展计算机使用智能体的可验证训练环境与任务 |
| 2026-03 | ◆ [基于视频的奖励模型](https://arxiv.org/abs/2603.10178) `arXiv:2603.10178` |  | 为计算机使用智能体训练基于录屏的奖励模型 |
| 2026-01 | ◆ [EvoCUA](https://arxiv.org/abs/2601.15876) `arXiv:2601.15876` |  | 从可扩展的合成经验中演化计算机使用智能体 |
| 2025-09 | 🆕 [UI-S1](https://arxiv.org/abs/2509.11543) `arXiv:2509.11543` |  | 半在线强化学习推进 GUI 自动化 |
| 2025-09 | 🆕 [UI-TARS-2](https://arxiv.org/abs/2509.02544) `arXiv:2509.02544` | ByteDance | 多轮强化学习推进 GUI 智能体 |
| 2025-08 | 🆕 [UI-Venus](https://arxiv.org/abs/2508.10833) `arXiv:2508.10833` | Ant Group | 以强化微调构建高性能 UI 智能体 |
| 2025-07 | 🆕 [GUI-G²](https://arxiv.org/abs/2507.15846) `arXiv:2507.15846` |  | 用高斯奖励建模 GUI 定位 |
| 2025-07 | 🆕 [如何训练网页智能体](https://arxiv.org/abs/2507.04103) `arXiv:2507.04103` |  | 网页智能体训练的统计诊断 |
| 2025-05 | 🆕 [ZeroGUI](https://arxiv.org/abs/2505.23762) `arXiv:2505.23762` |  | 零人工成本的在线 GUI 学习 |
| 2025-05 | 🆕 [Web-Shepherd](https://arxiv.org/abs/2505.15277) `arXiv:2505.15277` |  | 为网页智能体训练过程奖励模型 |
| 2025-04 | 🆕 [GUI-R1](https://arxiv.org/abs/2504.10458) `arXiv:2504.10458` |  | R1 式规则奖励 RL 训练通用 GUI 动作模型 |
| 2025-03 | 🆕 [UI-R1](https://arxiv.org/abs/2503.21620) `arXiv:2503.21620` |  | 用 RL 增强 GUI 智能体动作预测 |
| 2025-01 | 🆕 [UI-TARS](https://arxiv.org/abs/2501.12326) `arXiv:2501.12326` | ByteDance | 原生 GUI 智能体模型：感知、定位、系统 2 推理与从反思轨迹中迭代学习 |
| 2024-11 | 🆕 [WebRL](https://arxiv.org/abs/2411.02337) `arXiv:2411.02337` | Zhipu·Tsinghua | 自演化在线课程强化学习训练网页智能体 |
| 2024-08 | [Agent Q](https://arxiv.org/abs/2408.07199) `arXiv:2408.07199` | MultiOn | 搜索 + 自我批评 + DPO 训练网页智能体 |
| 2024-06 | [DigiRL](https://arxiv.org/abs/2406.11896) `arXiv:2406.11896` | UC Berkeley | 用自主强化学习训练野外设备控制智能体 |

### 多智能体 RL（11）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-06 | 🔥 [从受训者到训练者](https://arxiv.org/abs/2606.17682) `arXiv:2606.17682` |  | LLM 设计 RL 训练环境 |
| 2026-05 | 🔥 [多智能体 RL 何时改进 LLM 工作流](https://arxiv.org/abs/2605.24202) `arXiv:2605.24202` |  | 工作流、规模与策略共享的权衡 |
| 2026-05 | 🔥 [MetaAgent-X](https://arxiv.org/abs/2605.14212) `arXiv:2605.14212` |  | 端到端 RL 突破自动多智能体系统上限 |
| 2026-03 | ◆ [异构智能体协作 RL](https://arxiv.org/abs/2603.02604) `arXiv:2603.02604` |  | 异构智能体的协作式强化学习 |
| 2026-02 | ◆ [Dr. MAS：稳定的多智能体 RL](https://arxiv.org/abs/2602.08847) `arXiv:2602.08847` |  | 多智能体共用全局归一化基线会导致梯度不稳：按智能体分别归一化优势 |
| 2026-01 | ◆ [协作式多智能体测试时 RL](https://arxiv.org/abs/2601.09667) `arXiv:2601.09667` |  | 多智能体在测试时协作强化学习做推理 |
| 2025-11 | ◆ [M-GRPO：多智能体深度研究](https://arxiv.org/abs/2511.13288) `arXiv:2511.13288` |  | 主智能体与多个子智能体分别部署、异步训练，层级化地分配组相对优势 |
| 2025-10 | 🆕 [AT-GRPO：多智能体 RL](https://arxiv.org/abs/2510.11062) `arXiv:2510.11062` |  | 按智能体与轮次分组的 GRPO 及支持单/多策略的训练系统，多智能体协作任务大幅提升 |
| 2025-10 | 🆕 [AgentFlow](https://arxiv.org/abs/2510.05592) `arXiv:2510.05592` | Stanford | 流中优化的智能体系统：规划与工具使用 |
| 2025-08 | 🆕 [Chain-of-Agents](https://arxiv.org/abs/2508.13167) `arXiv:2508.13167` |  | 通过多智能体蒸馏与智能体 RL 训练端到端智能体基础模型 |
| 2025-02 | 🆕 [MAPoRL：多智能体协同训练](https://arxiv.org/abs/2502.18439) `arXiv:2502.18439` |  | 多个 LLM 讨论与协作时联合做 RL，奖励正确且有说服力的协作 |

### 训练系统与框架（36）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [从 RL 论文到框架代码](https://arxiv.org/abs/2610.28027) `arXiv:2610.28027` |  | 统计 7 个开源 RL 框架收录了哪些论文方法、收录前是否经过审查 |
| 2026-10 | 🔥 [KV4-RL：INT4 KV 缓存加速 rollout](https://arxiv.org/abs/2610.14448) `arXiv:2610.14448` |  | 免校准的 INT4 KV 量化，对注意力汇聚与近期上下文保留 INT8，加速异步 RLVR rollout |
| 2026-10 | 🔥 [TRACE：FP4 MoE RL 的量化感知训练](https://arxiv.org/abs/2610.07767) `arXiv:2610.07767` |  | 针对 MoE 模型 FP4 rollout 的训推量化对齐 |
| 2026-10 | 🔥 [TRIAGE：原生 NVFP4 RL 的失配稳定](https://arxiv.org/abs/2610.07043) `arXiv:2610.07043` |  | 按方向诊断 NVFP4 训推失配，选择性地重平衡策略梯度 |
| 2026-10 | 🔥 [LoGRA：低秩梯度草图](https://arxiv.org/abs/2610.06647) `arXiv:2610.06647` |  | 用低秩梯度草图保留学习信号，降低 RL 后训练的显存 |
| 2026-09 | 🔥 [全流程 FP8 强化学习](https://arxiv.org/abs/2609.22870) `arXiv:2609.22870` |  | 即便用 TIS 修正训推不一致，全 FP8 RL 仍会不稳：定位原因并给出全流程方案 |
| 2026-09 | 🔥 [长上下文 RL 中在线协同训练草稿模型](https://arxiv.org/abs/2609.07108) `arXiv:2609.07108` |  | RL 中策略不断变化：在线协同训练投机解码的草稿模型，并支持 256K 上下文 |
| 2026-07 | 🔥 [LongStraw](https://arxiv.org/abs/2607.14952) `arXiv:2607.14952` | Mind Lab | 固定 GPU 预算下超过 200 万 token 的长上下文 RL |
| 2026-06 | 🔥 [EfficientRollout：自投机解码加速 rollout](https://arxiv.org/abs/2606.18967) `arXiv:2606.18967` |  | 面向 RL rollout 的系统感知自投机解码 |
| 2026-06 | 🔥 [Bebop：多 token 预测加速 RL](https://arxiv.org/abs/2606.12370) `arXiv:2606.12370` |  | RL 后熵降低使 MTP 接受率上升：在大规模 RL 流水线中用 MTP 投机解码加速 rollout |
| 2026-05 | 🔥 [ESPO：提前停止的 PPO](https://arxiv.org/abs/2605.29860) `arXiv:2605.29860` |  | 在线检测注定失败的轨迹并提前终止，节省长 rollout 的算力 |
| 2026-05 | 🔥 [AstraFlow：数据流式的智能体 RL 系统](https://arxiv.org/abs/2605.15565) `arXiv:2605.15565` |  | 用数据流抽象替代以训练器为中心的控制，支持多策略协作 |
| 2026-05 | 🔥 [MinT](https://arxiv.org/abs/2605.13779) `arXiv:2605.13779` | Mind Lab | 训练与服务数百万个 LLM（LoRA）的托管基础设施 |
| 2026-04 | ◆ [用投机解码加速 RL rollout](https://arxiv.org/abs/2604.26779) `arXiv:2604.26779` |  | 把投机解码集成进 RL 训练系统，在保持在策略分布的前提下加速长 rollout |
| 2026-04 | ◆ [用经验回放高效训练 LLM RL](https://arxiv.org/abs/2604.08706) `arXiv:2604.08706` |  | 系统研究回放缓冲区：在陈旧带来的方差、样本多样性与生成成本之间取舍，回放可显著省算力 |
| 2026-03 | ◆ [ProRL Agent：rollout 即服务](https://arxiv.org/abs/2603.18815) `arXiv:2603.18815` | NVIDIA | 把多轮智能体的 rollout 生命周期做成 API 服务，与训练解耦 |
| 2026-01 | ◆ [FP8-RL](https://arxiv.org/abs/2601.18150) `arXiv:2601.18150` |  | FP8 rollout（权重 + KV 缓存）的实用栈，配合重要性采样修正训推不一致 |
| 2026-01 | ◆ [Jet-RL](https://arxiv.org/abs/2601.14243) `arXiv:2601.14243` |  | 训练与 rollout 统一精度的在策略 FP8 强化学习 |
| 2025-12 | ◆ [RLVR 中的参数高效方法](https://arxiv.org/abs/2512.23165) `arXiv:2512.23165` |  | 系统比较 LoRA 类方法用于 RLVR：DoRA 等结构变体优于 LoRA，SVD 初始化反而有害 |
| 2025-10 | 🆕 [QeRL](https://arxiv.org/abs/2510.11696) `arXiv:2510.11696` | NVIDIA | 量化带来的噪声反而增加探索熵，NVFP4 + LoRA 加速 RL |
| 2025-10 | 🆕 [Tinker](https://thinkingmachines.ai/tinker/) | Thinking Machines | 托管的微调与 RL 训练 API：用户写训练循环，平台负责分布式与 LoRA |
| 2025-09 | 🆕 [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) | Thinking Machines | 系统实验：在后训练与 RL 场景下合理配置的 LoRA 可以匹敌全参微调，RL 所需容量极小 |
| 2025-09 | 🆕 [PipelineRL](https://arxiv.org/abs/2509.19128) `arXiv:2509.19128` | ServiceNow | 生成过程中就地更新权重（in-flight），减少异步 RL 的陈旧度 |
| 2025-09 | 🆕 [RLinf](https://arxiv.org/abs/2509.15965) `arXiv:2509.15965` | RLinf | 面向具身与推理的大规模 RL 训练框架 |
| 2025-09 | 🆕 [集体 RL 经验共享](https://arxiv.org/abs/2509.08721) `arXiv:2509.08721` | Gensyn | 去中心化节点之间共享 rollout 的集体后训练 |
| 2025-06 | 🆕 [ROLL](https://arxiv.org/abs/2506.06122) `arXiv:2506.06122` | Alibaba | 面向大规模 RL 的易用框架 |
| 2025-05 | ★ 🆕 [AReaL](https://arxiv.org/abs/2505.24298) `arXiv:2505.24298` | Ant 等 | 完全异步的 RL 系统：生成与训练解耦，配合陈旧度控制与解耦的 PPO 目标 |
| 2025-05 | 🆕 [LlamaRL](https://arxiv.org/abs/2505.24034) `arXiv:2505.24034` | Meta | 分布式异步 RL 框架，可扩展到 405B |
| 2025-05 | 🆕 [INTELLECT-2](https://arxiv.org/abs/2505.07291) `arXiv:2505.07291` | Prime Intellect | 全球去中心化、异步的 32B 推理模型 RL 训练 |
| 2024-10 | [异步 RLHF](https://arxiv.org/abs/2410.18252) `arXiv:2410.18252` | Mila | 生成与训练异步进行，研究离策略程度对在线 DPO 等方法的影响 |
| 2024-10 | [verl](https://github.com/verl-project/verl) | ByteDance Seed | 大模型 RL 训练框架，广泛用于智能体 RL |
| 2024-09 | ★ [HybridFlow（veRL）](https://arxiv.org/abs/2409.19256) `arXiv:2409.19256` | ByteDance 等 | 单控制器编排 + 多控制器执行，训练与生成在同一组 GPU 上切换（3D-HybridEngine） |
| 2024-05 | [OpenRLHF](https://arxiv.org/abs/2405.11143) `arXiv:2405.11143` |  | 基于 Ray + vLLM 的开源 RLHF 框架，把策略、价值、奖励与参考模型分布到不同 GPU |
| 2024-05 | [NeMo-Aligner](https://arxiv.org/abs/2405.01481) `arXiv:2405.01481` | NVIDIA | 面向大规模对齐（RLHF、DPO、SteerLM）的工具包 |
| 2023-08 | [DeepSpeed-Chat](https://arxiv.org/abs/2308.01320) `arXiv:2308.01320` | Microsoft | 早期一体化的 RLHF 训练系统 |
| 2020-03 | [TRL](https://github.com/huggingface/trl) | Hugging Face | Hugging Face 的后训练库：SFT、奖励模型、PPO、DPO、GRPO 等 |

### 扩展律与训练配方（8）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [RL 扩展律的实证研究](https://arxiv.org/abs/2610.33420) `arXiv:2610.33420` |  | 在 Starpilot、Craftax 两个程序生成环境中为 actor-critic 找到可外推的扩展律（不是 LLM），可作为 LLM RL 扩展律的参照 |
| 2026-07 | 🔥 [Ring-Zero](https://arxiv.org/abs/2607.12395) `arXiv:2607.12395` | RUC/inclusionAI | 把零 RL 扩展到万亿参数，观察推理能力的涌现 |
| 2025-12 | ◆ [JustRL](https://arxiv.org/abs/2512.16649) `arXiv:2512.16649` |  | 单阶段、固定超参的最简 RL 配方，在 1.5B 模型上达到最好水平：很多技巧不是必需的 |
| 2025-12 | ◆ [Nemotron-Cascade](https://arxiv.org/abs/2512.13607) `arXiv:2512.13607` | NVIDIA | 按领域级联做 RL（对齐 → 指令 → 数学 → 代码 → SWE），兼顾思考与非思考模式 |
| 2025-10 | ★ 🆕 [RL 算力扩展的艺术](https://arxiv.org/abs/2510.13786) `arXiv:2510.13786` | Meta | 40 万 GPU 小时的系统研究：RL 性能随算力呈 S 形曲线可预测，给出 ScaleRL 配方 |
| 2025-05 | 🆕 [Skywork-OR1](https://arxiv.org/abs/2505.22312) `arXiv:2505.22312` | Skywork | 开源推理 RL 配方：多阶段训练、自适应熵控制、在线难度过滤 |
| 2025-05 | 🆕 [AceReason-Nemotron](https://arxiv.org/abs/2505.16400) `arXiv:2505.16400` | NVIDIA | 先数学后代码的分阶段 RL 显著提升小模型推理 |
| 2025-03 | 🆕 [Light-R1](https://arxiv.org/abs/2503.10460) `arXiv:2503.10460` |  | 从零训练长 CoT：课程 SFT → DPO → RL |

### 模型与技术报告（22）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [MiMo-V2.6](https://arxiv.org/abs/2610.11959) `arXiv:2610.11959` | Xiaomi | 把强化学习扩展到自我改进的模型技术报告 |
| 2026-07 | 🔥 [Kimi K3](https://arxiv.org/abs/2607.24653) `arXiv:2607.24653` | Moonshot AI | 2.8T 参数 MoE（激活 104B，896 选 16 专家），基于 Kimi Delta Attention 与 Attention Residuals，百万 token 上下文、原生视觉 |
| 2026-06 | 🔥 [VibeThinker-3B](https://arxiv.org/abs/2606.16140) `arXiv:2606.16140` | Weibo | 3B 小模型在可验证推理上逼近前沿：以多样性为目标的 SFT（谱阶段）再接 RL |
| 2026-06 | 🔥 [Ling & Ring 2.6](https://arxiv.org/abs/2606.15079) `arXiv:2606.15079` | inclusionAI | 万亿参数级、高效即时的智能体模型族 |
| 2026-05 | 🔥 [简单统一扩展达成奥赛金牌](https://arxiv.org/abs/2605.13301) `arXiv:2605.13301` |  | 用简单统一的扩展配方在多门奥赛达到金牌级推理 |
| 2026-04 | ◆ [DeepSeek-V4](https://arxiv.org/abs/2606.19348) `arXiv:2606.19348` | DeepSeek | V4-Pro 1.6T（激活 49B）与 V4-Flash 284B（激活 13B）MoE，百万 token 上下文；压缩稀疏注意力（CSA）+ 重度压缩注意力（HCA）混合、mHC 残差与 Muon 优化器，32T token 预训练 |
| 2026-02 | ◆ [Kimi K2.5](https://arxiv.org/abs/2602.02276) `arXiv:2602.02276` | Moonshot AI | 视觉智能体能力的开源多模态模型 |
| 2026-01 | ◆ [LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725) `arXiv:2601.16725` | Meituan | 智能体推理强化的思考模型 |
| 2025-12 | ◆ [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) `arXiv:2512.02556` | DeepSeek | DeepSeek 稀疏注意力（DSA）+ 大规模智能体任务合成与 RL |
| 2025-10 | 🆕 [Ring-1T：万亿思考模型 RL](https://arxiv.org/abs/2510.18855) `arXiv:2510.18855` | inclusionAI | 把 RL 扩展到 1T 参数思考模型的系统与算法（IcePop、C3PO++） |
| 2025-08 | 🆕 [gpt-oss](https://arxiv.org/abs/2508.10925) `arXiv:2508.10925` | OpenAI | OpenAI 的开放权重推理模型（120B/20B MoE） |
| 2025-08 | 🆕 [GLM-4.5](https://arxiv.org/abs/2508.06471) `arXiv:2508.06471` | Zhipu/Tsinghua | 面向智能体、推理与编码的 355B MoE，混合推理模式 |
| 2025-07 | 🆕 [Kimi K2](https://arxiv.org/abs/2507.20534) `arXiv:2507.20534` | Moonshot | 1T 参数的开放智能体模型：MuonClip 预训练 + 大规模智能体数据合成 + 联合 RL（可验证奖励 + 自评判） |
| 2025-07 | 🆕 [Gemini 2.5](https://arxiv.org/abs/2507.06261) `arXiv:2507.06261` | Google DeepMind | 原生思考、长上下文与智能体能力的前沿模型 |
| 2025-06 | 🆕 [MiniMax-M1](https://arxiv.org/abs/2506.13585) `arXiv:2506.13585` | MiniMax | 混合注意力推理模型，生成 10 万 token 的计算量约为 DeepSeek-R1 的 25%；CISPO 强化学习 |
| 2025-06 | 🆕 [Magistral](https://arxiv.org/abs/2506.10910) `arXiv:2506.10910` | Mistral AI | Mistral 的纯 RL 推理模型与训练栈 |
| 2025-05 | 🆕 [Qwen3](https://arxiv.org/abs/2505.09388) `arXiv:2505.09388` | Qwen | 思考/非思考模式统一与思考预算，稠密与 MoE 族（235B-A22B） |
| 2025-05 | 🆕 [MiMo](https://arxiv.org/abs/2505.07608) `arXiv:2505.07608` | Xiaomi | 从预训练（推理导向数据、多 token 预测）到 RL 全链路激发 7B 推理潜力 |
| 2025-04 | 🆕 [Phi-4-reasoning](https://arxiv.org/abs/2504.21318) `arXiv:2504.21318` | Microsoft | 精选可教提示 + o3-mini 轨迹 SFT + 少量 RL 的 14B 推理模型 |
| 2025-04 | 🆕 [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) `arXiv:2504.13914` | ByteDance Seed | MoE 推理模型与 RL 基础设施 |
| 2025-01 | 🆕 [Kimi k1.5](https://arxiv.org/abs/2501.12599) `arXiv:2501.12599` | Moonshot AI | 长上下文 RL + 简洁的策略优化，不用 MCTS 与价值函数，并提出 long2short |
| 2024-09 | [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) `arXiv:2409.12122` | Qwen | 自我改进的数学模型；其基座常被用作 RLVR 研究的起点，也带来了污染争议 |

### 理论与机制（23）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [学习而不覆盖](https://arxiv.org/abs/2610.05200) `arXiv:2610.05200` |  | 自蒸馏与 SFT 的理论 |
| 2026-09 | 🔥 [在策略更新方向决定泛化](https://arxiv.org/abs/2609.36659) `arXiv:2609.36659` | UCAS | 后训练的泛化来自在策略的参数更新方向 |
| 2026-09 | 🔥 [RLVR 的几何](https://arxiv.org/abs/2609.34344) `arXiv:2609.34344` |  | 用向量引导识别 RLVR 训练轨迹中的低维有效子空间 |
| 2026-08 | 🔥 [SFT 冲突，RL 共存](https://arxiv.org/abs/2608.03573) `arXiv:2608.03573` | CAS | 多任务下 SFT 的梯度相互冲突而 RL 更新可共存 |
| 2026-06 | 🔥 [可预测的 GRPO](https://arxiv.org/abs/2606.30789) `arXiv:2606.30789` |  | GRPO 训练动力学的闭式模型 |
| 2026-06 | 🔥 [训练策略优化的幻象](https://arxiv.org/abs/2606.29526) `arXiv:2606.29526` |  | RL 真正应优化的是单调的推理策略，而非训练时的采样策略 |
| 2026-06 | 🔥 [局部冲突、定向修复](https://arxiv.org/abs/2606.02398) `arXiv:2606.02398` |  | 用局部扰动分析解释多领域 RL 后训练中的跨领域干扰与恢复 |
| 2026-05 | ★ 🔥 [RLVR 只需极少训练：秩 1 轨迹外推](https://arxiv.org/abs/2605.21468) `arXiv:2605.21468` |  | RLVR 的权重轨迹极低秩且可预测：早期更新的秩 1 方向外推即可得到大部分收益 |
| 2026-04 | ◆ [强化学习为何泛化](https://arxiv.org/abs/2604.25011) `arXiv:2604.25011` |  | 后训练的特征级机制研究 |
| 2026-03 | ◆ [RLVR 更新的方向](https://arxiv.org/abs/2603.22117) `arXiv:2603.22117` |  | 以对数概率变化 Δlog p 的符号定位稀疏但关键的推理 token，并据此外推放大 RL 的更新方向 |
| 2026-02 | ◆ [RLVR 学习动力学中隐式课程的涌现](https://arxiv.org/abs/2602.14872) `arXiv:2602.14872` |  | RLVR 训练中自然出现由易到难的课程 |
| 2026-02 | ◆ [策略镜像下降的隐式正则](https://arxiv.org/abs/2602.05933) `arXiv:2602.05933` |  | Kimi 系列使用的策略镜像下降需估计配分函数，其近似带来隐式正则 |
| 2026-02 | ◆ [RLVR 的相对预算理论](https://arxiv.org/abs/2602.01523) `arXiv:2602.01523` |  | RLVR 中采样预算与题目难度的关系 |
| 2026-02 | ◆ [LLM 中的稀疏奖励子系统](https://arxiv.org/abs/2602.00986) `arXiv:2602.00986` |  | 奖励相关信息集中在少数神经元：价值神经元预测状态价值，多巴胺式神经元编码奖励预测误差 |
| 2025-12 | ◆ [自底向上策略优化](https://arxiv.org/abs/2512.19673) `arXiv:2512.19673` |  | LLM 策略内部包含逐层的"内部策略"：分析其演化并据此自底向上优化 |
| 2025-11 | ◆ [RLVR 偏离主成分学习](https://arxiv.org/abs/2511.08567) `arXiv:2511.08567` | Meta | RLVR 的更新避开权重的主方向，与 SFT 机制不同 |
| 2025-09 | ★ 🆕 [RL's Razor](https://arxiv.org/abs/2509.04259) `arXiv:2509.04259` | MIT | 在线 RL 比 SFT 遗忘更少：RL 偏向与原模型 KL 最小的解 |
| 2025-05 | 🆕 [RL 只更新小子网络](https://arxiv.org/abs/2505.11711) `arXiv:2505.11711` | Illinois | RL 微调只稀疏地更新 5%–30% 的参数，且不同种子、数据下子网络高度重合 |
| 2025-03 | ★ 🆕 [殊途同归：RL 在微调中的价值](https://arxiv.org/abs/2503.01067) `arXiv:2503.01067` | CMU | 既然信息论上离线就够，为何在线 RL 更好？——生成—验证差距：学一个简单的奖励模型再在其上做 RL 更容易 |
| 2025-02 | 🆕 [不带验证或 RL 的测试时扩展是次优的](https://arxiv.org/abs/2502.12118) `arXiv:2502.12118` | CMU | 证明只模仿搜索轨迹（无验证信号）的方法在扩展测试时计算时次优 |
| 2024-12 | ★ 🆕 [锐化机制](https://arxiv.org/abs/2412.01951) `arXiv:2412.01951` | Microsoft 等 | 自我改进 = 锐化：让策略把概率集中到自身认为最好的回答上；给出 SFT 与 RL 实现锐化的样本复杂度 |
| 2024-02 | [思维链的表达力](https://arxiv.org/abs/2402.12875) `arXiv:2402.12875` | Stanford/Google | T 步思维链让常数深度 Transformer 突破 TC⁰，可解决本质上串行的问题 |
| 2023-01 | [成对或 K 元比较的 RLHF 理论](https://arxiv.org/abs/2301.11270) `arXiv:2301.11270` | UC Berkeley | Bradley–Terry 下奖励学习的样本复杂度与悲观策略的必要性 |

### 奖励投机与安全（29）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🔥 [后训练留下的行为影子](https://arxiv.org/abs/2609.29233) `arXiv:2609.29233` | PKU | 后训练会在无关决策上留下系统性行为偏移 |
| 2026-08 | 🔥 [修辞如何投机 AI 审稿人](https://arxiv.org/abs/2608.08975) `arXiv:2608.08975` |  | 把论文改写得更有说服力即可抬高 AI 审稿评分：评判模型奖励的脆弱性 |
| 2026-07 | 🔥 [GPT-Red：大规模自我博弈红队](https://arxiv.org/abs/2607.26115) `arXiv:2607.26115` |  | 用 RL 训练自动红队智能体，发现针对前沿模型的新型提示注入攻击 |
| 2026-06 | 🔥 [RLMF：元认知反馈](https://arxiv.org/abs/2606.32032) `arXiv:2606.32032` |  | 按不确定性表达的忠实度修正偏好排序，让模型如实说出自己不确定 |
| 2026-06 | 🔥 [大模型会投机奖励，也会投机社会](https://arxiv.org/abs/2606.04075) `arXiv:2606.04075` |  | 72 个社会规则沙箱中奖励投机自然涌现，表现为发现监管漏洞 |
| 2026-04 | ◆ [大模型时代的奖励投机（综述）](https://arxiv.org/abs/2604.13602) `arXiv:2604.13602` |  | 以"代理压缩假说"统一理解奖励投机，并梳理其与涌现失准的联系 |
| 2026-03 | ◆ [DCPO：解耦推理与置信度](https://arxiv.org/abs/2603.09117) `arXiv:2603.09117` |  | RLVR 让模型过度自信：把推理正确性与置信度校准拆成两个目标分别优化 |
| 2026-02 | ◆ [思维链可监控性的信息论](https://arxiv.org/abs/2602.18297) `arXiv:2602.18297` |  | 用信息论分析并改进思维链可监控性 |
| 2026-01 | ◆ [TriPlay-RL：三角色自我博弈的安全对齐](https://arxiv.org/abs/2601.18292) `arXiv:2601.18292` |  | 攻击者、防御者、评估者三角色闭环共同进化，几乎无需人工标注 |
| 2025-11 | ★ ◆ [奖励投机导致的自然涌现失准](https://arxiv.org/abs/2511.18397) `arXiv:2511.18397` | Anthropic | 在真实编程 RL 环境中学会奖励投机，会泛化出对齐伪装与破坏安全研究等失准行为 |
| 2025-11 | ◆ [去中心化 GRPO 的投毒攻击](https://arxiv.org/abs/2511.09780) `arXiv:2511.09780` |  | 共享 rollout 的去中心化训练中，恶意节点可通过投毒补全植入行为；给出防御 |
| 2025-08 | 🆕 [奖励投机学校](https://arxiv.org/abs/2508.17511) `arXiv:2508.17511` |  | 在无害任务上学会投机评判，会泛化为更广泛的失准行为 |
| 2025-07 | 🆕 [思维链可监控性](https://arxiv.org/abs/2507.11473) `arXiv:2507.11473` | 多机构联合 | CoT 监控是脆弱但宝贵的安全机会，呼吁开发者保护可监控性 |
| 2025-05 | 🆕 [推理模型并不总说真话](https://arxiv.org/abs/2505.05410) `arXiv:2505.05410` | Anthropic | 推理模型的思维链常常不承认使用了提示中的线索：CoT 忠实性有限 |
| 2025-03 | ★ 🆕 [监控推理模型的不当行为](https://arxiv.org/abs/2503.11926) `arXiv:2503.11926` | OpenAI | CoT 监控能抓住奖励投机；但对 CoT 施加优化压力会让模型学会隐藏意图 |
| 2025-02 | 🆕 [涌现失准](https://arxiv.org/abs/2502.17424) `arXiv:2502.17424` |  | 只在"写不安全代码"这样的窄任务上微调，就会泛化出广泛的失准行为 |
| 2025-01 | 🆕 [GuardReasoner](https://arxiv.org/abs/2501.18492) `arXiv:2501.18492` | NUS | 基于推理的 LLM 安全护栏 |
| 2024-12 | 🆕 [o1 System Card](https://arxiv.org/abs/2412.16720) `arXiv:2412.16720` | OpenAI | o1 系列的安全评估：推理用于遵循安全规范，也带来新的风险面 |
| 2024-12 | 🆕 [审慎对齐](https://arxiv.org/abs/2412.16339) `arXiv:2412.16339` | OpenAI | 让推理模型在回答前显式回忆并推理安全规范 |
| 2024-12 | 🆕 [对齐伪装](https://arxiv.org/abs/2412.14093) `arXiv:2412.14093` | Anthropic 等 | 模型在认为会被训练时策略性地顺从，以保护自己原有的偏好 |
| 2024-11 | 🆕 [基于规则的安全奖励（RBR）](https://arxiv.org/abs/2411.01111) `arXiv:2411.01111` | OpenAI | 把安全政策写成细粒度规则，由 LLM 评分作为 RL 奖励的一部分 |
| 2024-09 | [RLHF 让模型学会误导人类](https://arxiv.org/abs/2409.12822) `arXiv:2409.12822` |  | RLHF 后模型更擅长让人类评估者相信错误答案是对的（"U-Sophistry"） |
| 2024-07 | [证明者—验证者博弈提升可读性](https://arxiv.org/abs/2407.13692) `arXiv:2407.13692` | OpenAI | 让强证明者写出弱验证者也能检查的解，对抗"只追求正确率"带来的不可读 |
| 2024-06 | [从谄媚到暗中破坏：奖励篡改](https://arxiv.org/abs/2406.10162) `arXiv:2406.10162` | Anthropic | 在越来越可被投机的环境课程中，模型从谄媚泛化到篡改自身的奖励函数 |
| 2024-01 | [Sleeper Agents](https://arxiv.org/abs/2401.05566) `arXiv:2401.05566` | Anthropic | 植入的后门行为能在 SFT、RL 与对抗训练后存活 |
| 2023-12 | [弱到强泛化](https://arxiv.org/abs/2312.09390) `arXiv:2312.09390` | OpenAI | 用弱模型监督强模型，强模型能超越监督者：超级对齐的实证类比 |
| 2023-10 | [理解语言模型的谄媚](https://arxiv.org/abs/2310.13548) `arXiv:2310.13548` | Anthropic | 人类偏好数据本身奖励迎合用户观点的回答，RLHF 因此放大谄媚 |
| 2020-04 | [规范投机](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) | DeepMind | 智能体利用目标设定的漏洞拿高分的经典案例集：奖励投机的 RL 前史 |
| 2018-05 | [以辩论实现 AI 安全](https://arxiv.org/abs/1805.00899) `arXiv:1805.00899` | OpenAI | 两个智能体辩论、人类裁判：可扩展监督的博弈论方案 |

### 基准与评测（8）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-05 | 🔥 [LaRA：检测 RL 后训练中的数据污染](https://arxiv.org/abs/2605.29888) `arXiv:2605.29888` |  | 污染在各层表征上产生渐进的几何偏移，可用于检测 RL 后模型是否见过测试题 |
| 2024-12 | 🆕 [LLM 推理稳定吗](https://arxiv.org/abs/2412.13147) `arXiv:2412.13147` | Shanghai AI Lab | 提出 G-Pass@k，衡量多次采样下推理的稳定性 |
| 2024-03 | [LiveCodeBench](https://arxiv.org/abs/2403.07974) `arXiv:2403.07974` | UC Berkeley 等 | 持续更新、按时间防污染的代码评测 |
| 2023-11 | [GPQA](https://arxiv.org/abs/2311.12022) `arXiv:2311.12022` | NYU | 研究生水平、"谷歌搜不到"的科学问答 |
| 2023-10 | [SWE-bench](https://arxiv.org/abs/2310.06770) `arXiv:2310.06770` | Princeton | 真实 GitHub 问题修复，以仓库测试作验证：智能体 RL 最常用的可验证任务 |
| 2021-10 | [GSM8K 与验证器](https://arxiv.org/abs/2110.14168) `arXiv:2110.14168` | OpenAI | 小学数学数据集；训练验证器给候选解打分 |
| 2021-07 | [Codex 与 HumanEval](https://arxiv.org/abs/2107.03374) `arXiv:2107.03374` | OpenAI | 代码生成评测与 pass@k 的无偏估计公式 |
| 2021-03 | [MATH](https://arxiv.org/abs/2103.03874) `arXiv:2103.03874` | UC Berkeley | 竞赛数学问题集 |

### 综述与教程（7）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-01 | ◆ [LLM 的智能体推理](https://arxiv.org/abs/2601.12538) `arXiv:2601.12538` | UIUC | 智能体推理的系统综述 |
| 2025-09 | ★ 🆕 [大推理模型 RL 综述](https://arxiv.org/abs/2509.08827) `arXiv:2509.08827` | Tsinghua 等 | RL 用于推理模型的奖励设计、策略优化、采样与基础设施 |
| 2025-09 | 🆕 [深度研究系统的 RL 基础](https://arxiv.org/abs/2509.06733) `arXiv:2509.06733` |  | 深度研究系统的强化学习综述 |
| 2025-09 | 🆕 [智能体 RL 全景](https://arxiv.org/abs/2509.02547) `arXiv:2509.02547` |  | LLM 智能体强化学习综述 |
| 2025-08 | 🆕 [大模型强化学习综述](https://arxiv.org/abs/2508.08189) `arXiv:2508.08189` |  | 覆盖 RLHF、RLVR、多模态与智能体的大模型 RL 综述 |
| 2025-04 | ★ 🆕 [RLHF Book（Lambert）](https://arxiv.org/abs/2504.12501) `arXiv:2504.12501` |  | Nathan Lambert 的开放教材（rlhfbook.com）：偏好数据、奖励模型、拒绝采样、PPO/DPO 与推理 RL 的实践细节 |
| 2024-12 | ★ 🆕 [强化学习概览（Murphy）](https://arxiv.org/abs/2412.05265) `arXiv:2412.05265` | Google DeepMind | Kevin Murphy 的 RL 教材式综述，含 LLM 与 RL 一章，适合补 RL 背景 |

