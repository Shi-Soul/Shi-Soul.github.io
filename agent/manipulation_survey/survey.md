# 机器人操作：从行为克隆到 VLA 与世界-动作模型（长文综述）

> 数据截至 2026-10-30。本文与[交互式导读页](./)共用同一份数据：收录 349 条论文与资源，其中 331 条 arXiv 条目的标题与日期经 export.arxiv.org API 核验，166 条发表于 2025-10 之后（"最近一年"）。文中数值摘自论文摘要或官方发布；公司自报的数字已注明，未经独立复核。附录 A 是按类别分组的完整论文表。灵巧手专题见[灵巧手导读](../dexhand_survey/)，LLM agent 控制机器人见[机器人 × Agent 导读](../embodied_agent_survey/)，扩散/流匹配原理见[扩散模型导读](../diffusion_survey/)。

**本文怎么读。**第 1 节给出学习型操作的形式化；第 2–5 节讲策略本身（架构、表征、VLA、动作表示与实时执行）；第 6–8 节讲数据（采集、人类视频、仿真）；第 9–10 节讲超越示范的两条路（强化学习后训练、世界模型）；第 11 节是推理、触觉与移动操作；第 12 节是评测；第 13 节按月份列出最近一年的大事；第 14 节是开放问题。

---

## 1 问题设定

### 1.1 策略与动作块

学习型操作的核心是一个策略 $\pi_\theta$：给定当前观测 $o_t$ 与任务描述 $\ell$（语言指令、目标图像或示范视频），输出接下来 $H$ 步的动作序列

$$a_{t:t+H}\sim\pi_\theta\bigl(\cdot\mid o_t,\ \ell\bigr)$$

观测通常包括多视角 RGB（腕部相机加外部相机）、可选的深度或点云、本体状态（关节角、末端位姿、夹爪开合），有时还有触觉与力。动作可以是末端位姿增量、绝对关节角或夹爪指令，跨本体模型需要把不同机器人的动作空间对齐（[RDT-1B](https://arxiv.org/abs/2410.07864) 的统一动作空间、[X-VLA](https://arxiv.org/abs/2510.10274) 的本体软提示）。

**动作分块**（action chunking）是 2023 年以后的默认做法：一次预测 $H$ 步（常取 10–50 步，约 0.5–2 秒），执行一部分后再重新预测。[ACT](https://arxiv.org/abs/2304.13705) 与 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 都表明，分块能平滑轨迹、减少复合误差，并为慢推理提供缓冲。

### 1.2 行为克隆与复合误差

最基本的学习方法是**行为克隆**：在示范数据集 $\mathcal D$ 上做监督学习，

$$\max_\theta\ \mathbb E_{(o,\ell,a)\sim\mathcal D}\bigl[\log\pi_\theta(a\mid o,\ell)\bigr]$$

它的根本问题是**复合误差**：学生一旦偏离示范，就会进入训练数据没覆盖的状态，误差随时间累积。[DAgger](https://arxiv.org/abs/1011.0686) 证明，只要让专家在学生自己访问到的状态上补标注，误差就能控制在与时域线性相关的范围内。今天对应的做法是人在环纠正（[HIL-SERL](https://arxiv.org/abs/2410.21845)）与在部署中持续学习（[π*0.6](https://arxiv.org/abs/2511.14759)）。

### 1.3 生成式动作头

人类示范往往是**多峰**的：绕左边走和绕右边走都对，而均方回归会平均出一条撞上障碍的路。因此主流策略用生成模型表示动作分布。以流匹配动作头为例（原理见[扩散模型导读](../diffusion_survey/)），

$$\mathcal L(\theta)=\mathbb E_{(o,\ell,a)\sim\mathcal D,\ t,\ \varepsilon}\Bigl[\bigl\lVert v_\theta(a^{(t)},t\mid o,\ell)-(\varepsilon-a)\bigr\rVert^2\Bigr],\qquad a^{(t)}=(1-t)\,a+t\,\varepsilon$$

推理时从高斯噪声出发积分几步到十几步得到动作块。[π0](https://arxiv.org/abs/2410.24164)、[GR00T N1](https://arxiv.org/abs/2503.14734)、[GR-3](https://arxiv.org/abs/2507.15493) 等都采用"VLM 骨干 + 流匹配/扩散动作专家"的结构。

### 1.4 频率分层

大模型推理慢（1–10 Hz），动作专家快（10–50 Hz），底层控制器更快（200–1000 Hz）。"双系统"设计把它们拆开：[Helix](https://www.figure.ai/news/helix) 用 7–9 Hz 的 VLM 加 200 Hz 的视觉运动策略控制人形上半身，[GR00T N1](https://arxiv.org/abs/2503.14734) 用 10 Hz 的 VLM 加 120 Hz 的扩散动作模块；[实时分块执行（RTC）](https://arxiv.org/abs/2506.07339)则让单个慢模型在推理期间平滑衔接动作块（第 5 节）。

---

## 2 策略架构的演进

**端到端与自监督（2015–2019）。**[Levine 等](https://arxiv.org/abs/1504.00702)用引导策略搜索训练从像素到力矩的卷积网络，开启端到端机器人学习；[Pinto 与 Gupta](https://arxiv.org/abs/1509.06825)用 5 万次尝试、[Levine 等](https://arxiv.org/abs/1603.02199)用 14 台机械臂 80 万次尝试自监督学抓取；[QT-Opt](https://arxiv.org/abs/1806.10293) 在 58 万次真机抓取上做大规模 Q 学习，闭环抓取成功率 96%；[Dex-Net 2.0](https://arxiv.org/abs/1703.09312) 用合成点云训练抓取质量网络。

**多峰行为与语言条件（2019–2022）。**[Learning from Play](https://arxiv.org/abs/1903.01973) 从无标签"玩耍"数据学潜在计划；[Implicit BC](https://arxiv.org/abs/2109.00137) 用能量模型表示多峰、不连续的动作；[BeT](https://arxiv.org/abs/2206.11251) 把动作离散成簇加残差；[BC-Z](https://arxiv.org/abs/2202.02005)、[Interactive Language](https://arxiv.org/abs/2210.06407)、[RT-1](https://arxiv.org/abs/2212.06817) 把语言条件模仿学习扩展到数百个任务与十几万条示范。

**生成式策略与精细操作（2023–）。**[ACT/ALOHA](https://arxiv.org/abs/2304.13705) 用低成本双臂遥操作硬件加 CVAE 动作分块 Transformer 学会了穿扎带、开杯盖等精细双手任务；[Diffusion Policy](https://arxiv.org/abs/2303.04137) 用扩散模型生成动作序列，成为视觉运动策略的事实标准，后续有 3D 点云版本 [DP3](https://arxiv.org/abs/2403.03954)、一致性蒸馏加速的 [Consistency Policy](https://arxiv.org/abs/2405.07503)，以及把大规模示范与扩散结合的 [ALOHA Unleashed](https://arxiv.org/abs/2410.13126)（系鞋带、挂衣服）。[VQ-BeT](https://arxiv.org/abs/2403.03181) 则用残差向量量化做动作 token。

**数据规律。**[Lin 等](https://arxiv.org/abs/2410.18647)收集 4 万多条示范、执行 1.5 万多次真机评测后发现：策略在新环境、新物体上的泛化主要取决于训练时环境与物体的**多样性**，单环境里堆示范数量收益很快饱和；四名采集者一个下午的数据就能让两项任务在新环境中达到约 90% 成功率。[另一项研究](https://arxiv.org/abs/2509.18644)发现，常用的本体状态输入会让策略过度依赖它而忽视视觉，去掉后空间泛化明显更好。

**最近一年。**能量策略再次出现（[EBT-Policy](https://arxiv.org/abs/2510.27545)，推理时迭代优化能量，表现出纠错行为）；RL 被当作示范生成器，为行为克隆产出更高效、多样的数据（[DiscoDemo](https://arxiv.org/abs/2610.32093)）；从视觉示范中直接推断并执行新任务的上下文学习开始成型（[简单配方](https://arxiv.org/abs/2609.38173)、[综述](https://arxiv.org/abs/2609.36012)）。

---

## 3 表征：机器人该"看到"什么

**物体中心与关键点。**[Dense Object Nets](https://arxiv.org/abs/1806.08756) 自监督学习稠密描述子，[kPAM](https://arxiv.org/abs/1903.06684) 用语义关键点表达类别级任务，[Neural Descriptor Fields](https://arxiv.org/abs/2112.05124) 用 SE(3) 等变隐式场把少量示范迁移到新姿态新物体，[Where2Act](https://arxiv.org/abs/2101.02692) 预测铰接物体的可操作区域。

**空间结构带来的样本效率。**[Transporter Networks](https://arxiv.org/abs/2010.14406) 把抓-放建模为空间等变的热力图；[CLIPort](https://arxiv.org/abs/2109.12098) 叠加 CLIP 语义通路；[PerAct](https://arxiv.org/abs/2209.05451)、[RVT](https://arxiv.org/abs/2306.14896)、[RVT-2](https://arxiv.org/abs/2406.08545)、[Act3D](https://arxiv.org/abs/2306.17817)、[3D Diffuser Actor](https://arxiv.org/abs/2402.10885) 用体素、多视角渲染或 3D 特征场预测关键位姿，在 RLBench、CALVIN 上用很少的示范达到很高的成功率，但较难处理连续接触与动态动作。

**预训练视觉表征。**[R3M](https://arxiv.org/abs/2203.12601)（Ego4D 人类视频）、[MVP](https://arxiv.org/abs/2203.06173)（掩码预训练）、[VIP](https://arxiv.org/abs/2210.00030)（价值隐式预训练，同时给出奖励）、[Voltron](https://arxiv.org/abs/2302.12766)（语言驱动）探索了"为操作而预训练"的视觉编码器。VLA 时代这一问题被 VLM 骨干吸收，新的问题变成：

- **微调会伤害视觉表征。**[别让 VLA 失明](https://arxiv.org/abs/2510.25616)发现在机器人数据上微调会破坏 VLM 原有的视觉表征，把它对齐回去能改善分布外泛化；[打破视觉-动作捷径](https://arxiv.org/abs/2609.12641)防止策略利用与动作偶然相关的无关视觉线索；[表征中心的继续预训练](https://arxiv.org/abs/2608.27550)认为机器人数据难扩时，提升表征质量比堆数据更关键。
- **补上 3D 空间感。**[Spatial Forcing](https://arxiv.org/abs/2510.12276) 把 VLA 中间特征对齐到 3D 基础模型的表征，无需额外 3D 输入；[From Spatial to Actions](https://arxiv.org/abs/2510.17439) 把空间先验注入动作头；[Grounded Action Model](https://arxiv.org/abs/2609.23863)、[Geometric Action Model](https://arxiv.org/abs/2606.17046) 以 3D 定位与几何推理为策略的基础。

---

## 4 VLA 与机器人基础模型

### 4.1 从 RT-2 到流匹配动作专家

[RT-2](https://arxiv.org/abs/2307.15818)（2023-07）把动作离散成 token，与网络图文一起微调 VLM，首次展示了网络知识向操作的迁移，提出 VLA 概念。随后：[Open X-Embodiment](https://arxiv.org/abs/2310.08864) 汇聚 21 家机构、22 种机器人的数据并训练 RT-X；[Octo](https://arxiv.org/abs/2405.12213) 与 [OpenVLA](https://arxiv.org/abs/2406.09246) 成为开源基线；[HPT](https://arxiv.org/abs/2409.20537)、[CrossFormer](https://arxiv.org/abs/2408.11812) 研究跨本体预训练；[RDT-1B](https://arxiv.org/abs/2410.07864) 用统一动作空间训练双臂扩散基础模型。

[π0](https://arxiv.org/abs/2410.24164)（2024-10）确立了今天的主流结构：预训练 VLM 处理图像与语言，一个独立的"动作专家"用流匹配生成连续动作块，跨 7 种机器人构型预训练，完成叠衣服、收拾餐桌等长时程灵巧任务。[π0.5](https://arxiv.org/abs/2504.16054) 用异构数据共训（多种机器人、网络图文、高层子任务标注、口头指令），同一模型先解码子任务文本再生成动作，能在从未见过的家庭里收拾厨房与卧室。[Gemini Robotics](https://arxiv.org/abs/2503.20020) 与 [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) 把具身推理（ER）模型与 VLA 分工，并提出跨本体的 Motion Transfer；[GR00T N1](https://arxiv.org/abs/2503.14734) 以双系统形式开源人形基础模型。

### 4.2 配方研究

什么让 VLA 好用？[RoboVLMs](https://arxiv.org/abs/2412.14058) 系统比较了骨干、结构与跨本体数据；[OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 的微调配方（并行解码、动作分块、连续动作、L1 回归）把 OpenVLA 在 LIBERO 上的平均成功率从 76.5% 提到 97.1%，同时大幅加快动作生成；[知识隔离](https://arxiv.org/abs/2505.23705)在训练时阻断动作专家向 VLM 的梯度回流，保留语言知识；[VLA-Adapter](https://arxiv.org/abs/2509.09372) 表明 0.5B 级小骨干加轻量桥接也能接近大模型；[VLANeXt](https://arxiv.org/abs/2602.18532) 梳理了各种设计选择。

### 4.3 最近一年

- **后训练与经验学习**：[π*0.6](https://arxiv.org/abs/2511.14759)（2025-11）与 [π0.7](https://arxiv.org/abs/2604.15483)（2026-04，约 5B，可用策略、元数据、子目标图像等多模态上下文操控，零样本跨本体），详见第 9 节。
- **十万小时级预训练**：[LingBot-VLA](https://arxiv.org/abs/2601.18692)（约 2 万小时、9 种双臂构型）、[Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330)（10 万小时以上 UMI 数据，开箱即用做移动操作，随数据与模型规模持续提升），以及公司发布的 [GEN-0](https://eu.36kr.com/en/p/3551715251804547)、[GEN-1](https://www.therobotreport.com/generalist-introduces-gen-1-general-purpose-model-for-physical-ai/)（第 6 节）。
- **大厂家族化**：[Qwen-VLA](https://arxiv.org/abs/2605.30280)、[Qwen-RobotManip](https://arxiv.org/abs/2606.17846)、[RynnVLA-002](https://arxiv.org/abs/2511.17502)（与世界模型统一）、[RynnVLA-Latent](https://arxiv.org/abs/2610.34356)、GigaBrain 系列（[0](https://arxiv.org/abs/2510.19430)、[0.5M*](https://arxiv.org/abs/2602.12099)、[0.7](https://arxiv.org/abs/2608.15875)）、[MolmoAct2](https://arxiv.org/abs/2605.02881)、[HY-Embodied-0.5](https://arxiv.org/abs/2604.07430)、[Green-VLA](https://arxiv.org/abs/2602.00919)。
- **推理与记忆**：[Recurrent-Depth VLA](https://arxiv.org/abs/2602.07845) 在潜空间循环以实现测试时计算扩展，[Fast-ThinkAct](https://arxiv.org/abs/2601.09708) 降低思维链延迟，[双重潜在记忆](https://arxiv.org/abs/2607.07608)处理依赖历史的任务，[RoboOmni](https://arxiv.org/abs/2510.23763) 从语音与环境声中主动推断意图。
- **鲁棒与部署**：[执行误差自补偿](https://arxiv.org/abs/2609.37334)处理机械磨损与负载变化，[CARE](https://arxiv.org/abs/2609.24118) 从失败中学原子级纠正动作，[常识保留](https://arxiv.org/abs/2606.19297)测量 VLM 微调成 VLA 后还剩多少世界知识。
- **产业**：[Helix 02](https://interestingengineering.com/ai-robotics/figure-helix02-upgrades-humanoid-robot-control)（1 kHz 学习式全身控制器）、[GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)、[GR00T N1.7 与 N2 预览](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)、[Skild S1](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/)、[Rho-alpha](https://www.therobotreport.com/microsoft-research-reveals-rho-alpha-vision-language-action-model-for-robots/)（加入触觉）、[AgiBot GO-2](https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/)、[Pi API](https://www.pi.website/blog/api-beta)。

---

## 5 动作表示与实时执行

**五种动作表示。**离散 token（RT-2、OpenVLA）直接复用 VLM 但精度与速度受限；[FAST](https://arxiv.org/abs/2501.09747) 用离散余弦变换压缩动作块再 token 化，序列更短、能做高频任务；连续回归（OpenVLA-OFT）简单快速但难以表达多峰；扩散/流匹配头是当前主流；潜在动作（[LAPA](https://arxiv.org/abs/2410.11758)、[GO-1](https://arxiv.org/abs/2503.06669)、[UniVLA](https://arxiv.org/abs/2505.06111)）则能利用无动作标签的视频。轨迹草图（[RT-Trajectory](https://arxiv.org/abs/2311.01977)）与"语言动作"（[RT-H](https://arxiv.org/abs/2403.01823)）提供了介于语言与动作之间的中间层。

**实时执行。**大模型推理一次往往需要几十到几百毫秒，而机器人在此期间仍在运动。[RTC](https://arxiv.org/abs/2506.07339) 把动作块衔接写成修补问题：推理期间冻结已承诺执行的动作、只补全后续部分，让慢模型也能平滑控制。最近一年这一问题被进一步拆解：

- **反应延迟**：[FASTER](https://arxiv.org/abs/2603.19199) 指出异步推理不只要平滑，还要缩短对环境变化的反应时间。
- **状态偏差**：[VLASH](https://arxiv.org/abs/2512.01031) 以"动作执行时刻的预测状态"为条件推理，[DynamicVLA](https://arxiv.org/abs/2601.22153) 专门处理运动中的物体。
- **走走停停**：[When to Switch](https://arxiv.org/abs/2610.05719) 可靠地延长动作块，[VLA-Rolling](https://arxiv.org/abs/2610.25043) 复用未执行的动作尾巴以减少去噪步数。
- **轻量化**：[TurboVLA](https://arxiv.org/abs/2607.27205) 绕开以 LLM 为中心的通路，在 RTX 4090 上 32 Hz、显存不到 1 GB；[Kinematic MeanFlow](https://arxiv.org/abs/2610.00864) 用一步生成替代多步流匹配。
- **测试时筛选**：[反探索式测试时扩展](https://arxiv.org/abs/2512.02834)与 [VLS](https://arxiv.org/abs/2602.03973) 在推理时引导或筛选动作候选，无需重新训练。

---

## 6 数据：采集与规模化

**采集接口。**遥操作仍是质量最高的来源：[GELLO](https://arxiv.org/abs/2309.13037) 用按比例缩小的主从臂，[ALOHA](https://arxiv.org/abs/2304.13705) 与 [Mobile ALOHA](https://arxiv.org/abs/2401.02117) 把双臂与全身遥操作做到低成本，[Open-TeleVision](https://arxiv.org/abs/2407.01512)、[OPEN TEACH](https://arxiv.org/abs/2403.07870)、[AnyTeleop](https://arxiv.org/abs/2307.04577) 用 VR 与视觉。[UMI](https://arxiv.org/abs/2402.10329)（2024-02）是一个转折：人手持带相机的夹爪在野外操作，事后恢复位姿作为动作，数据可迁移到多种机械臂；[AirExo](https://arxiv.org/abs/2309.14975)、[FastUMI](https://arxiv.org/abs/2409.19499)、[UMI on Legs](https://arxiv.org/abs/2407.10353)、[DexUMI](https://arxiv.org/abs/2505.21864) 沿此方向扩展。[AutoRT](https://arxiv.org/abs/2401.12963) 则让 VLM 编排机器人舰队自主采集。

**数据集。**[BridgeData V2](https://arxiv.org/abs/2308.12952)（6 万条）、[RH20T](https://arxiv.org/abs/2307.00595)（11 万多条多模态序列）、[Open X-Embodiment](https://arxiv.org/abs/2310.08864)（22 种机器人）、[DROID](https://arxiv.org/abs/2403.12945)（7.6 万条、350 小时、564 个场景）、[AgiBot World](https://arxiv.org/abs/2503.06669)（100 万条以上轨迹、217 个任务）依次把公开数据推到新的量级。

**最近一年：十万小时量级。**Generalist 宣称 [GEN-0](https://eu.36kr.com/en/p/3551715251804547) 用 27 万小时操作数据训练，并在 7B 以上观察到扩展规律，[GEN-1](https://www.therobotreport.com/generalist-introduces-gen-1-general-purpose-model-for-physical-ai/) 约 50 万小时、所选任务平均成功率 99%（公司公布）；[Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) 用 10 万小时以上 UMI 数据；[HiFi-UMI](https://arxiv.org/abs/2607.25895) 表明只用高保真手持数据就能训出可部署的策略。采集也在变得"有针对性"：[RoboPocket](https://arxiv.org/abs/2603.05504) 用手机根据当前策略的失败点闭环补数据。[数据金字塔](https://arxiv.org/abs/2607.24744)综述了网络数据、人类视频、仿真与真机数据的分层关系。需要提醒的是，不同来源的"小时数"口径差异很大（是否含空闲时段、是否过滤），公司数字未经独立复核。

---

## 7 人类视频与跨本体迁移

人类视频数量大、场景多样，但没有机器人动作，且人手与机器人手形态不同。主要的桥接方式：

1. **中间表示**：预测点轨迹或光流，再由低层策略执行——[ATM](https://arxiv.org/abs/2401.00025)、[Track2Act](https://arxiv.org/abs/2405.01527)、[Im2Flow2Act](https://arxiv.org/abs/2407.15208)；[MimicPlay](https://arxiv.org/abs/2302.12422) 从人类玩耍视频学高层潜在计划。
2. **潜在动作**：从视频帧间变化中学潜变量作为预训练动作——[LAPA](https://arxiv.org/abs/2410.11758)（超过用真实动作标签预训练的 VLA）、[UniVLA](https://arxiv.org/abs/2505.06111)、[RynnVLA-Latent](https://arxiv.org/abs/2610.34356)。
3. **把人当作另一种本体**：第一人称人类数据与机器人数据统一建模——[EgoMimic](https://arxiv.org/abs/2410.24221)、[HAT](https://arxiv.org/abs/2503.13441)、[EgoVLA](https://arxiv.org/abs/2507.12440)、[Being-H0](https://arxiv.org/abs/2507.15597)；或把视频中的人手渲染成机器人（[Phantom](https://arxiv.org/abs/2503.00779)）。
4. **生成**：先生成"人做这件事"的视频再执行（[Gen2Act](https://arxiv.org/abs/2409.16283)）。

**最近一年**的关键词是规模与规律：[EgoScale](https://arxiv.org/abs/2602.16710) 用 20,854 小时带动作标签的第一人称视频训练 VLA，发现数据规模与验证损失呈对数线性关系，并在 22 自由度灵巧手上比无预训练基线平均提升 54%；[HumanNet](https://arxiv.org/abs/2605.06747) 构建 100 万小时的以人为中心视频库；[DreamDojo](https://arxiv.org/abs/2602.06949) 用 4.4 万小时人类视频训练机器人世界模型；Dyna 宣称 [DYNA-2](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-2-world-action-model-demonstrating-first-true-scaling-law-in-robotics-powered-entirely-by-human-data-302847114.html) 用 100 万小时以上人类视频训练出人到机器人的扩展规律。以人手动作作为通用"动作语言"的模型也密集出现（[Being-H0.5](https://arxiv.org/abs/2601.12993)、[ACE-Ego-0](https://arxiv.org/abs/2606.17200)、[UniT](https://arxiv.org/abs/2604.19734)、[PhysBrain 1.0](https://arxiv.org/abs/2605.15298)），数据基础设施方面有 [Open-AoE](https://arxiv.org/abs/2607.14183)、[ACE-Data-0](https://arxiv.org/abs/2607.28625)。

---

## 8 仿真与合成数据

**平台。**[robosuite](https://arxiv.org/abs/2009.12293)、[Isaac Gym](https://arxiv.org/abs/2108.10470)、[ManiSkill2](https://arxiv.org/abs/2302.04659) / [ManiSkill3](https://arxiv.org/abs/2410.00425)、[Isaac Lab](https://arxiv.org/abs/2511.04831)、[MuJoCo Playground](https://arxiv.org/abs/2502.08844)、[RoboVerse](https://arxiv.org/abs/2504.18904) 提供 GPU 并行的物理与渲染。

**合成示范。**[MimicGen](https://arxiv.org/abs/2310.17596) 从约 200 条人类示范自动扩增出 5 万条仿真示范，[DexMimicGen](https://arxiv.org/abs/2410.24185) 扩展到双手灵巧；[RoboGen](https://arxiv.org/abs/2311.01455) 让 LLM 提出任务、生成场景与奖励；[Eureka](https://arxiv.org/abs/2310.12931) 让 LLM 写奖励代码；[GraspVLA](https://arxiv.org/abs/2505.03233) 用十亿级合成动作数据预训练抓取基础模型；[RoboTwin 2.0](https://arxiv.org/abs/2506.18088) 是可扩展的双臂数据生成器。

**real2sim2real。**[RialTo](https://arxiv.org/abs/2403.03949) 快速把真实场景重建进仿真并在其中 RL 提升鲁棒性，[SplatSim](https://arxiv.org/abs/2409.10161) 用高斯泼溅渲染替换仿真画面实现 RGB 策略的零样本迁移；最近一年有可组合的场景建模 [Lucida](https://arxiv.org/abs/2608.30821)，以及让大模型从视频或草图生成 MuJoCo 孪生与训练数据的 [Gemini 4 Argon](https://developers.googleblog.com/gemini-4-argon-for-robotics/)（官方称生成的仿真数据把真机成功率从 56% 提到 75%）。另一条路线是用视频世界模型代替物理仿真生成数据（[DreamGen](https://arxiv.org/abs/2505.12705)、[GigaWorld-0](https://arxiv.org/abs/2511.19861)，见第 10 节）。

---

## 9 强化学习与后训练

### 9.1 真机强化学习

[QT-Opt](https://arxiv.org/abs/1806.10293) 与 [Q-Transformer](https://arxiv.org/abs/2309.10150) 展示了大规模离线 + 在线 Q 学习。样本高效的真机 RL 在 [SERL](https://arxiv.org/abs/2401.16013) 与 [HIL-SERL](https://arxiv.org/abs/2410.21845) 中成熟：结合示范、人工纠正与离策略 RL，HIL-SERL 在精密装配、动态操作等任务上只需 1–2.5 小时训练即接近 100% 成功。[Policy Decorator](https://arxiv.org/abs/2412.13630)、[ResFiT](https://arxiv.org/abs/2509.19301) 用残差 RL 微调大策略，[PA-RL](https://arxiv.org/abs/2412.06685) 对任意类别与骨干的策略做 RL 微调，[Q-chunking](https://arxiv.org/abs/2507.07969) 在动作块空间做 RL。

### 9.2 给 VLA 做强化学习

2025 年起，RL 开始直接作用于大 VLA：[DPPO](https://arxiv.org/abs/2409.00588) 把扩散策略的去噪过程纳入 MDP；[ConRFT](https://arxiv.org/abs/2502.05450)、[RIPT-VLA](https://arxiv.org/abs/2505.17016)、[VLA-RL](https://arxiv.org/abs/2505.18719) 做离线/在线强化微调；[SimpleVLA-RL](https://arxiv.org/abs/2509.09674) 把 GRPO 式的结果奖励 RL 用于 VLA，少量示范加 RL 即可大幅提升，并观察到示范中没有的新动作模式；[RLinf-VLA](https://arxiv.org/abs/2510.06710)、[πRL](https://arxiv.org/abs/2510.25889) 提供统一框架并支持流匹配 VLA。

### 9.3 最近一年：从经验中持续变强

**优势条件化（RECAP）。**[π*0.6](https://arxiv.org/abs/2511.14759) 的做法是：用全部经验（示范、自主执行、人工干预）训练价值函数 $V$，计算每段动作的优势 $A=R-V(o)$，把"$A$ 是否为正"作为额外条件 $I$ 输入策略，

$$\max_\theta\ \mathbb E_{(o,a)\sim\mathcal D}\Bigl[\log\pi_\theta\bigl(a\mid o,\ell,\ I=\mathbb 1[A>0]\bigr)\Bigr]$$

推理时固定 $I=1$。训练仍是监督学习，能沿用大模型的训练栈；在叠衣、装箱、做意式咖啡等最难的任务上，吞吐翻倍以上、失败率约减半。

**轻量在线 RL。**[RL Token](https://arxiv.org/abs/2604.23073) 从 VLA 读出一个紧凑的"RL token"，只训练小型 actor-critic 头，几小时在线 RL 就让关键阶段提速最高 3 倍；[SRPO](https://arxiv.org/abs/2511.15605) 用模型自己的成功轨迹为失败轨迹打进度奖励；[GR-RL](https://arxiv.org/abs/2512.01801) 把通用 VLA 训成长时程精密任务的专家。

**舰队级学习。**[SOP](https://arxiv.org/abs/2601.03044)（AgiBot）让多台机器人同时做 VLA 的在线后训练；[Learning While Deploying](https://arxiv.org/abs/2605.00416) 与 [舰队经验学习](https://arxiv.org/abs/2610.25736)用部署中机器人自己的执行作为监督，减少对专家纠正的依赖；[RLinf-USER](https://arxiv.org/abs/2602.07837) 提供真机在线学习系统。

**奖励与想象。**[TOPReward](https://arxiv.org/abs/2602.19313) 用 VLM 的 token 概率做零样本稠密进度奖励（综述见 [进度奖励综述](https://arxiv.org/abs/2607.21655)）；[RISE](https://arxiv.org/abs/2602.11075)、[WCM](https://arxiv.org/abs/2607.29613)、[Rollback](https://arxiv.org/abs/2610.36068)、[GigaBrain-0.5M*](https://arxiv.org/abs/2602.12099) 在世界模型中做 RL 或反事实回滚。

---

## 10 世界模型与世界-动作模型

**早期：视频预测用于规划与策略。**[Deep Visual Foresight](https://arxiv.org/abs/1610.00696) 与 [Visual Foresight](https://arxiv.org/abs/1812.00568) 学习动作条件的视频预测并用于规划；[UniPi](https://arxiv.org/abs/2302.00111) 把文本引导的视频生成加逆动力学当作通用策略；[SuSIE](https://arxiv.org/abs/2310.10639) 用图像编辑模型生成子目标；[UniSim](https://arxiv.org/abs/2310.06114) 学习交互式模拟器；[GR-1](https://arxiv.org/abs/2312.13139)、[GR-2](https://arxiv.org/abs/2410.06158) 用视频生成预训练策略。

**联合建模画面与动作。**[Video Prediction Policy](https://arxiv.org/abs/2412.14803) 用视频扩散模型的预测性表征做策略；[UVA](https://arxiv.org/abs/2503.00200)、[UWM](https://arxiv.org/abs/2504.02792) 联合建模视频与动作，推理时可只解码动作；[V-JEPA 2](https://arxiv.org/abs/2506.09985) 的动作条件版本可以零样本规划抓放；[Genie Envisioner](https://arxiv.org/abs/2508.05635) 把视频世界模型、动作解码与神经模拟器统一成平台。

**最近一年：WAM 崛起。**[DreamZero](https://arxiv.org/abs/2602.15922)（2026-02）基于 14B 视频扩散模型联合预测视频与动作，7 Hz 闭环，宣称对新动作与新环境的泛化比 VLA 高 2 倍以上，并成为 NVIDIA GR00T N2 的基础；[GE-Act 2.0](https://arxiv.org/abs/2609.05588)、[GigaWorld-Policy](https://arxiv.org/abs/2603.17240)、[InternVLA-A1.5](https://arxiv.org/abs/2607.04988)、[InternW0-Δ](https://arxiv.org/abs/2609.31394)、[OpenWAM](https://arxiv.org/abs/2609.07398)、[UniWAM](https://arxiv.org/abs/2610.02054) 与 Meta 的[渐进视觉规划](https://arxiv.org/abs/2610.02508)、[原生动作先验](https://arxiv.org/abs/2610.03391)相继出现。稠密视频预测代价高，于是有只预测关键帧（[ImageWAM](https://arxiv.org/abs/2606.19531)）、只预测变化量（[DeltaWAM](https://arxiv.org/abs/2609.28811)）、只预测潜在未来（[WAVE-0](https://arxiv.org/abs/2610.18568)）的变体，以及扩展历史上下文的 [Long-WAM](https://arxiv.org/abs/2610.10528)。产业上有 [1X World Model](https://www.1x.tech/ai)、[DYNA-2](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-2-world-action-model-demonstrating-first-true-scaling-law-in-robotics-powered-entirely-by-human-data-302847114.html)、[FLUX 3 Action](https://bfl.ai/models/flux-3-action)。

**世界模型用于评测与数据。**DeepMind 用 Veo 视频世界模型评测 Gemini Robotics 策略，并与 1600 多次真机评测对照（[论文](https://arxiv.org/abs/2512.10675)）；[Ctrl-World](https://arxiv.org/abs/2510.10125) 不用真机就能给策略排名，在想象中合成成功轨迹再微调使成功率提升 44.7%；[GigaWorld-1](https://arxiv.org/abs/2607.02642) 给出路线图；[DreamX-Phi](https://arxiv.org/abs/2608.13489)、[RynnWorld-4D](https://arxiv.org/abs/2607.06559)、[PhysisForcing](https://arxiv.org/abs/2606.28128)、[DreamTrue](https://arxiv.org/abs/2610.12468) 提升动作忠实度与物理合理性。

**风险。**[BadWAM](https://arxiv.org/abs/2607.15207) 展示了"画面预测正确、动作却出错"的对抗攻击；[何时信任想象](https://arxiv.org/abs/2605.06222)用"未来-现实验证"决定执行多少预测动作。WAM 综述见 [WAM Survey](https://arxiv.org/abs/2606.20781)。

---

## 11 推理、触觉与移动操作

**高层推理与空间智能。**[SayCan](https://arxiv.org/abs/2204.01691)、[Code as Policies](https://arxiv.org/abs/2209.07753)、[Inner Monologue](https://arxiv.org/abs/2207.05608)、[VoxPoser](https://arxiv.org/abs/2307.05973)、[MOKA](https://arxiv.org/abs/2403.03174)、[ReKep](https://arxiv.org/abs/2409.01652) 用大模型做规划、写代码或生成约束；[ECoT](https://arxiv.org/abs/2407.08693)、[ThinkAct](https://arxiv.org/abs/2507.16815) 把推理放进 VLA；[Hi Robot](https://arxiv.org/abs/2502.19417) 与 [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) 采用分层结构。最近一年：[GroundingPI](https://arxiv.org/abs/2609.39601)、[RoboTracer](https://arxiv.org/abs/2512.13660)、[RynnBrain](https://arxiv.org/abs/2602.14979)、[MiMo-Embodied](https://arxiv.org/abs/2511.16518) 等具身推理基础模型，以及 Google 的 [ER 1.6](https://deepmind.google/blog/gemini-robotics-er-1-6/)、[ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)、[Gemini 4 Argon](https://developers.googleblog.com/gemini-4-argon-for-robotics/)。前沿通用模型直接当策略的评测见 [Claude plays robotics](https://www.anthropic.com/research/claude-plays-robotics) 与[机器人 × Agent 导读](../embodied_agent_survey/)。

**触觉与力。**视触觉传感器（[DIGIT](https://arxiv.org/abs/2005.14679)）、磁性皮肤（[AnySkin](https://arxiv.org/abs/2409.08276)）、触觉表征（[Sparsh](https://arxiv.org/abs/2410.24090)）与仿真（[TacSL](https://arxiv.org/abs/2408.06506)）逐步成熟；[3D-ViTac](https://arxiv.org/abs/2410.24091) 融合触觉与视觉点云做双手精细操作，[FACTR](https://arxiv.org/abs/2502.17432) 用力反馈遥操作与力感知课程。最近一年触觉进入基础模型：[OmniVTLA](https://arxiv.org/abs/2508.08706)、[N0-VTLA](https://arxiv.org/abs/2607.23782)、触觉原生 WAM [N0-TWAM](https://arxiv.org/abs/2607.23783)、[TORL-VLA](https://arxiv.org/abs/2606.09337)，以及加入触觉的 [Rho-alpha](https://www.therobotreport.com/microsoft-research-reveals-rho-alpha-vision-language-action-model-for-robots/)。灵巧手上的触觉研究见[灵巧手导读](../dexhand_survey/)。

**移动操作、双臂与人形。**[HomeRobot](https://arxiv.org/abs/2306.11565)、[OK-Robot](https://arxiv.org/abs/2401.12202)、[BUMBLE](https://arxiv.org/abs/2410.06237) 研究开放词汇的家庭移动操作；[Mobile ALOHA](https://arxiv.org/abs/2401.02117)、[BEHAVIOR Robot Suite](https://arxiv.org/abs/2503.05652)、[HumanPlus](https://arxiv.org/abs/2406.10454)、[HOMIE](https://arxiv.org/abs/2502.13013) 解决全身遥操作与学习；[WholeBodyVLA](https://arxiv.org/abs/2512.11047)、[Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) 与 [BEHAVIOR Challenge 冠军方案](https://arxiv.org/abs/2512.06951)把 VLA 用于移动操作。可形变物体方面有 [FlingBot](https://arxiv.org/abs/2105.03655)、[SpeedFolding](https://arxiv.org/abs/2208.10552)。

---

## 12 评测

**仿真基准**：[RLBench](https://arxiv.org/abs/1909.12271)、[Meta-World](https://arxiv.org/abs/1910.10897)、[CALVIN](https://arxiv.org/abs/2112.03227)、[LIBERO](https://arxiv.org/abs/2306.03310)、[RoboCasa](https://arxiv.org/abs/2406.02523)、[BEHAVIOR-1K](https://arxiv.org/abs/2403.09227)。LIBERO 已成为 VLA 的主擂台，但也已饱和：[OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 等方法普遍超过 90%。

**鲁棒性诊断**揭示了这些高分的脆弱：[LIBERO-PRO](https://arxiv.org/abs/2510.03827) 在泛化设置下让标准评测 90% 以上的模型跌到 0%；[LIBERO-Plus](https://arxiv.org/abs/2510.13626) 发现相机视角、机器人初始状态等小扰动就能让成功率从 95% 跌到 30% 以下；[LIBERO-Para](https://arxiv.org/abs/2603.28301) 显示指令改写也会让模型失效；[COLOSSEUM](https://arxiv.org/abs/2402.08191) 用 14 种扰动因子系统评测；[RoboMME](https://arxiv.org/abs/2603.04639) 与 [LIBERO-RECOVER](https://arxiv.org/abs/2609.05178) 评测记忆与失败恢复。

**仿真对真机的预测力**：[SIMPLER](https://arxiv.org/abs/2405.05941) 表明精心构建的仿真评测与真机排名高度相关；世界模型评测（[Veo](https://arxiv.org/abs/2512.10675)、[Ctrl-World](https://arxiv.org/abs/2510.10125)）是新的候选。

**真机评测**：[RoboArena](https://arxiv.org/abs/2506.18123) 用分布式、双盲、成对比较替代集中式评测，[AutoEval](https://arxiv.org/abs/2503.24278) 几乎无人值守，[VLA-REPLICA](https://arxiv.org/abs/2605.20774) 用低成本机械臂提高可复现性，[RoboDojo](https://arxiv.org/abs/2607.04434) 提供仿真 + 真机统一榜单。

**报告建议**：写清每个任务的试验次数与置信区间、物体与场景是否见过、是否允许重试与人工复位，并报告完成时间与吞吐，而不只是成功率。

---

## 13 最近一年大事记（2025-10 → 2026-10）

- **2025-10**：[LIBERO-PRO](https://arxiv.org/abs/2510.03827) 与 [LIBERO-Plus](https://arxiv.org/abs/2510.13626) 揭示 VLA 在扰动下崩溃；[Spatial Forcing](https://arxiv.org/abs/2510.12276)、[From Spatial to Actions](https://arxiv.org/abs/2510.17439)；[Ctrl-World](https://arxiv.org/abs/2510.10125)；[πRL](https://arxiv.org/abs/2510.25889)、[RLinf-VLA](https://arxiv.org/abs/2510.06710)；[GigaBrain-0](https://arxiv.org/abs/2510.19430)；[Robot Learning 教程](https://arxiv.org/abs/2510.12403)。
- **2025-11**：[π*0.6 / RECAP](https://arxiv.org/abs/2511.14759)；[GEN-0](https://eu.36kr.com/en/p/3551715251804547)（公司公布 27 万小时）；[RynnVLA-002](https://arxiv.org/abs/2511.17502)；[SRPO](https://arxiv.org/abs/2511.15605)；[GigaWorld-0](https://arxiv.org/abs/2511.19861)。
- **2025-12**：[用 Veo 评测 Gemini Robotics](https://arxiv.org/abs/2512.10675)；[GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)；[VLASH](https://arxiv.org/abs/2512.01031)；[GR-RL](https://arxiv.org/abs/2512.01801)；[PhysBrain](https://arxiv.org/abs/2512.16793)。
- **2026-01**：[LingBot-VLA](https://arxiv.org/abs/2601.18692)；[SOP](https://arxiv.org/abs/2601.03044)；[Being-H0.5](https://arxiv.org/abs/2601.12993)；[DynamicVLA](https://arxiv.org/abs/2601.22153)；[Helix 02](https://interestingengineering.com/ai-robotics/figure-helix02-upgrades-humanoid-robot-control)；[1X World Model](https://www.1x.tech/ai)；[Rho-alpha](https://www.therobotreport.com/microsoft-research-reveals-rho-alpha-vision-language-action-model-for-robots/)。
- **2026-02**：[DreamZero](https://arxiv.org/abs/2602.15922)；[EgoScale](https://arxiv.org/abs/2602.16710)；[DreamDojo](https://arxiv.org/abs/2602.06949)；[Recurrent-Depth VLA](https://arxiv.org/abs/2602.07845)；[RISE](https://arxiv.org/abs/2602.11075)；[TOPReward](https://arxiv.org/abs/2602.19313)。
- **2026-03**：[GR00T N1.7 / N2 预览](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)；[FASTER](https://arxiv.org/abs/2603.19199)；[RoboMME](https://arxiv.org/abs/2603.04639)；[RoboPocket](https://arxiv.org/abs/2603.05504)。
- **2026-04**：[π0.7](https://arxiv.org/abs/2604.15483)；[RL Token](https://arxiv.org/abs/2604.23073)；[GEN-1](https://www.therobotreport.com/generalist-introduces-gen-1-general-purpose-model-for-physical-ai/)；[AgiBot GO-2](https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/)；[ER 1.6](https://deepmind.google/blog/gemini-robotics-er-1-6/)。
- **2026-05**：[HumanNet](https://arxiv.org/abs/2605.06747)；[Qwen-VLA](https://arxiv.org/abs/2605.30280)；[MolmoAct2](https://arxiv.org/abs/2605.02881)；[Learning While Deploying](https://arxiv.org/abs/2605.00416)；[PhysBrain 1.0](https://arxiv.org/abs/2605.15298)。
- **2026-06**：[Qwen-RobotManip](https://arxiv.org/abs/2606.17846)；[ImageWAM](https://arxiv.org/abs/2606.19531)；[ACE-Ego-0](https://arxiv.org/abs/2606.17200)；[WAM 综述](https://arxiv.org/abs/2606.20781)。
- **2026-07**：[Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330)；[HiFi-UMI](https://arxiv.org/abs/2607.25895)；[TurboVLA](https://arxiv.org/abs/2607.27205)；[N0-VTLA](https://arxiv.org/abs/2607.23782) 与 [N0-TWAM](https://arxiv.org/abs/2607.23783)；[ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)；[Claude plays robotics](https://www.anthropic.com/research/claude-plays-robotics)。
- **2026-08**：[DYNA-2](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-2-world-action-model-demonstrating-first-true-scaling-law-in-robotics-powered-entirely-by-human-data-302847114.html)；[Skild S1](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/)；[GigaBrain-0.7](https://arxiv.org/abs/2608.15875)；[DreamX-Phi 1.0](https://arxiv.org/abs/2608.13489)。
- **2026-09**：[GE-Act 2.0](https://arxiv.org/abs/2609.05588)；[OpenWAM](https://arxiv.org/abs/2609.07398)；[InternW0-Δ](https://arxiv.org/abs/2609.31394)；[GroundingPI](https://arxiv.org/abs/2609.39601)；[FLUX 3 Action](https://bfl.ai/models/flux-3-action)。
- **2026-10**：[舰队经验学习](https://arxiv.org/abs/2610.25736)；[Rollback](https://arxiv.org/abs/2610.36068)；[Meta WAM](https://arxiv.org/abs/2610.02508)；[Long-WAM](https://arxiv.org/abs/2610.10528)；[Pi API beta](https://www.pi.website/blog/api-beta)；[Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/)。

---

## 14 开放问题

1. **数据的正确单位。**小时数、轨迹数还是环境多样性？一小时人类视频相当于多少真机数据？扩展规律应该用验证损失还是真机成功率来衡量？
2. **泛化还是记忆。**标准基准的高分在扰动下崩溃，VLA 学到的究竟是可迁移的技能，还是训练场景的捷径？
3. **精细接触与力。**插拔、拧、柔性物体与工具使用仍是短板，触觉与力控怎样以可扩展的方式进入大模型？
4. **真机 RL 的成本与安全。**舰队级在线学习需要可靠的奖励、自动复位与安全边界，人工干预怎样最省？
5. **VLA 与 WAM 的关系。**先理解语言再行动，还是先预测世界再行动？二者会融合成一个模型，还是分层协作？
6. **长时程与记忆。**做一顿饭、收拾一个房间需要记住做过什么、物体在哪，而多数策略仍是马尔可夫的。
7. **评测的可复现性。**真机评测昂贵且难以跨实验室对齐，世界模型评测与真机的一致性需要在更多任务上验证。
8. **硬件与本体。**平行夹爪、灵巧手、人形各有取舍，跨本体迁移的上限在哪里？

**延伸阅读。**[Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403)、[VLA 综述](https://arxiv.org/abs/2405.14093)、[数据金字塔](https://arxiv.org/abs/2607.24744)、[WAM 综述](https://arxiv.org/abs/2606.20781)、[进度奖励综述](https://arxiv.org/abs/2607.21655)、[机器人上下文学习综述](https://arxiv.org/abs/2609.36012)、[VLA 安全综述](https://arxiv.org/abs/2604.23775)、[机器人需要的不止 VLA 与世界模型](https://arxiv.org/abs/2606.06556)。

---

## 附录 A：论文与资源总表（349 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 核验；★ 为入门推荐；🆕 为 2025-10 之后的工作。

### 早期学习方法（14）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2024-03 | [RT-H](https://arxiv.org/abs/2403.01823) `arXiv:2403.01823` | Google DeepMind | 用"语言动作"（如"手臂前移"）作中间层次，支持人用语言纠正 |
| 2023-10 | [RoboCLIP](https://arxiv.org/abs/2310.07899) `arXiv:2310.07899` | USC/Stanford | 用视频-语言模型给一次示范打相似度奖励，免手工奖励 |
| 2023-09 | [Q-Transformer](https://arxiv.org/abs/2309.10150) `arXiv:2309.10150` | Google DeepMind | 把 Q 函数按动作维度自回归离散化，大规模离线 RL |
| 2023-06 | [RoboCat](https://arxiv.org/abs/2306.11706) `arXiv:2306.11706` | Google DeepMind | 自我改进的多本体 Transformer 智能体：自产数据再训练 |
| 2022-10 | [Interactive Language](https://arxiv.org/abs/2210.06407) `arXiv:2210.06407` | Google | 数十万条语言标注轨迹，实时用自然语言引导桌面重排 |
| 2022-05 | [Gato](https://arxiv.org/abs/2205.06175) `arXiv:2205.06175` | DeepMind | 同一 Transformer 处理文本、图像、游戏与真机操作的通才智能体 |
| 2022-02 | [BC-Z](https://arxiv.org/abs/2202.02005) `arXiv:2202.02005` | Google | 大规模多任务模仿 + 语言/视频任务嵌入，零样本泛化到新任务 |
| 2019-03 | [TossingBot](https://arxiv.org/abs/1903.11239) `arXiv:1903.11239` | Google/Princeton | 残差物理：在解析抛掷模型上学习残差，学会抛掷任意物体 |
| 2019-03 | [Learning from Play](https://arxiv.org/abs/1903.01973) `arXiv:1903.01973` | Google | 从无任务标签的"玩耍"数据学习潜在计划，目标条件策略 |
| 2018-06 | ★ [QT-Opt](https://arxiv.org/abs/1806.10293) `arXiv:1806.10293` | Google | 58 万次真机抓取上的大规模离线/在线 Q 学习，闭环抓取成功率 96% |
| 2016-03 | [大规模手眼协调](https://arxiv.org/abs/1603.02199) `arXiv:1603.02199` | Google | 14 台机械臂采集 80 万次抓取，用 CNN 预测抓取成功率并伺服 |
| 2015-09 | [Supersizing Self-supervision](https://arxiv.org/abs/1509.06825) `arXiv:1509.06825` | CMU | 5 万次尝试、700 小时机器人时间自监督学抓取 |
| 2015-04 | ★ [端到端视觉运动策略](https://arxiv.org/abs/1504.00702) `arXiv:1504.00702` | UC Berkeley | 引导策略搜索训练从像素到力矩的 CNN 策略，开启端到端机器人学习 |
| 2010-11 | ★ [DAgger](https://arxiv.org/abs/1011.0686) `arXiv:1011.0686` | CMU | 把模仿学习归约为无悔在线学习：让专家标注学生自己访问到的状态，解决行为克隆的复合误差 |

### 抓取（4）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-05 | [GraspVLA](https://arxiv.org/abs/2505.03233) `arXiv:2505.03233` | Galbot/PKU | 用十亿级合成动作数据预训练的抓取基础模型 |
| 2022-12 | [AnyGrasp](https://arxiv.org/abs/2212.08333) `arXiv:2212.08333` | SJTU | 时空域鲁棒高效的通用抓取感知，可抓动态物体 |
| 2021-03 | [Contact-GraspNet](https://arxiv.org/abs/2103.14127) `arXiv:2103.14127` | NVIDIA | 杂乱场景中从点云高效生成 6-DoF 抓取 |
| 2017-03 | ★ [Dex-Net 2.0](https://arxiv.org/abs/1703.09312) `arXiv:1703.09312` | UC Berkeley | 用合成点云 + 解析抓取指标训练抓取质量网络 |

### 模仿学习与策略架构（13）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [DiscoDemo](https://arxiv.org/abs/2610.32093) `arXiv:2610.32093` | KAIST | 把 RL 当作示范生成器，产出高效多样的示范供行为克隆 |
| 2025-10 | 🆕 [EBT-Policy](https://arxiv.org/abs/2510.27545) `arXiv:2510.27545` |  | 能量 Transformer 策略：推理时迭代优化能量，表现出纠错等物理推理行为 |
| 2025-09 | [要不要本体状态](https://arxiv.org/abs/2509.18644) `arXiv:2509.18644` |  | 发现视觉运动策略过度依赖本体状态输入，去掉后空间泛化显著变好 |
| 2024-10 | ★ [模仿学习的数据扩展律](https://arxiv.org/abs/2410.18647) `arXiv:2410.18647` | Tsinghua | 4 万条示范 + 1.5 万次真机评测：泛化主要取决于环境与物体的多样性，而非单环境示范数量 |
| 2024-05 | [Consistency Policy](https://arxiv.org/abs/2405.07503) `arXiv:2405.07503` | Stanford | 一致性蒸馏把扩散策略加速到一两步 |
| 2024-03 | [DP3](https://arxiv.org/abs/2403.03954) `arXiv:2403.03954` | Shanghai Qi Zhi/Tsinghua | 紧凑 3D 点云表示 + 扩散策略，少量示范即可泛化 |
| 2024-03 | [VQ-BeT](https://arxiv.org/abs/2403.03181) `arXiv:2403.03181` | NYU | 用残差向量量化做动作 token，统一条件/无条件行为生成 |
| 2023-09 | [RoboAgent / MT-ACT](https://arxiv.org/abs/2309.01918) `arXiv:2309.01918` | CMU/Meta | 语义数据增广 + 动作分块，7500 条轨迹学会 38 个任务 |
| 2023-07 | [AWE](https://arxiv.org/abs/2307.14326) `arXiv:2307.14326` | Stanford | 自动从示范中抽取路点，缩短模仿学习的有效时域 |
| 2023-04 | ★ [ACT / ALOHA](https://arxiv.org/abs/2304.13705) `arXiv:2304.13705` | Stanford | 低成本双臂遥操作硬件 + 动作分块 Transformer（CVAE），学会精细双手任务 |
| 2023-03 | ★ [Diffusion Policy](https://arxiv.org/abs/2303.04137) `arXiv:2303.04137` | Columbia/TRI | 用扩散模型生成动作序列，自然处理多峰分布，成为视觉运动策略的事实标准 |
| 2022-06 | [BeT](https://arxiv.org/abs/2206.11251) `arXiv:2206.11251` | NYU | Behavior Transformer：动作离散成 k 个簇 + 残差，克隆多峰行为 |
| 2021-09 | [Implicit BC](https://arxiv.org/abs/2109.00137) `arXiv:2109.00137` | Google | 用能量模型表示策略，能拟合多峰与不连续动作 |

### 视觉/3D 表征（20）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [打破视觉-动作捷径](https://arxiv.org/abs/2609.12641) `arXiv:2609.12641` | NUS | 潜在接口训练，避免策略利用与动作相关的无关视觉线索 |
| 2026-06 | 🆕 [Geometric Action Model](https://arxiv.org/abs/2606.17046) `arXiv:2606.17046` | ETH | 在 3D 几何中联合推理物体、相机与动作的策略学习框架 |
| 2025-10 | 🆕 [别让 VLA 失明](https://arxiv.org/abs/2510.25616) `arXiv:2510.25616` |  | VLA 微调会破坏 VLM 的视觉表征；对齐回原表征可提升分布外泛化 |
| 2025-10 | 🆕 [From Spatial to Actions](https://arxiv.org/abs/2510.17439) `arXiv:2510.17439` | ByteDance Seed | 把空间基础模型的先验注入 VLA 动作头，弥补 2D 编码器的空间推理缺口 |
| 2025-10 | 🆕 [Spatial Forcing](https://arxiv.org/abs/2510.12276) `arXiv:2510.12276` | HKUST-GZ | 把 VLA 中间特征对齐到 3D 基础模型的表征，无需额外 3D 输入即可增强空间感知 |
| 2024-06 | [RVT-2](https://arxiv.org/abs/2406.08545) `arXiv:2406.08545` | NVIDIA | 由粗到细的多视角 Transformer，少量示范完成高精度任务 |
| 2024-02 | [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) `arXiv:2402.10885` | CMU | 3D 场景表示 + 扩散策略，CALVIN、RLBench 上的强基线 |
| 2023-06 | [Act3D](https://arxiv.org/abs/2306.17817) `arXiv:2306.17817` | CMU | 3D 特征场 Transformer 做多任务操作 |
| 2023-06 | [RVT](https://arxiv.org/abs/2306.14896) `arXiv:2306.14896` | NVIDIA | 多视角虚拟渲染代替体素，训练快 36 倍 |
| 2023-02 | [Voltron](https://arxiv.org/abs/2302.12766) `arXiv:2302.12766` | Stanford | 语言驱动的视觉表征学习 |
| 2022-09 | [VIP](https://arxiv.org/abs/2210.00030) `arXiv:2210.00030` | Meta/Penn | 价值隐式预训练：同时得到视觉表征与通用奖励 |
| 2022-09 | [PerAct](https://arxiv.org/abs/2209.05451) `arXiv:2209.05451` | UW/NVIDIA | 体素输入 + Perceiver Transformer 预测下一个关键位姿 |
| 2022-03 | [R3M](https://arxiv.org/abs/2203.12601) `arXiv:2203.12601` | Meta/Stanford | 在 Ego4D 人类视频上预训练的通用操作视觉表征 |
| 2022-03 | [MVP](https://arxiv.org/abs/2203.06173) `arXiv:2203.06173` | UC Berkeley | 掩码视觉预训练用于运动控制 |
| 2021-12 | [Neural Descriptor Fields](https://arxiv.org/abs/2112.05124) `arXiv:2112.05124` | MIT | SE(3) 等变的物体隐式描述场，少量示范迁移到新姿态新物体 |
| 2021-09 | [CLIPort](https://arxiv.org/abs/2109.12098) `arXiv:2109.12098` | UW/NVIDIA | CLIP 语义通路 + Transporter 空间通路的语言条件操作 |
| 2021-01 | [Where2Act](https://arxiv.org/abs/2101.02692) `arXiv:2101.02692` | Stanford/Meta | 从像素预测铰接物体上"哪里能推/拉"的可操作性 |
| 2020-10 | [Transporter Networks](https://arxiv.org/abs/2010.14406) `arXiv:2010.14406` | Google | 用空间等变的"抓-放"热力图做桌面重排，样本效率极高 |
| 2019-03 | [kPAM](https://arxiv.org/abs/1903.06684) `arXiv:1903.06684` | MIT | 用语义关键点表示类别级操作目标，适配形状差异 |
| 2018-06 | [Dense Object Nets](https://arxiv.org/abs/1806.08756) `arXiv:1806.08756` | MIT | 自监督学习稠密物体描述子，用于类别级操作 |

### 动作表示与实时执行（13）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [VLA-Rolling](https://arxiv.org/abs/2610.25043) `arXiv:2610.25043` | Zhejiang Univ | 复用上一块未执行的动作尾巴，减少去噪步数且控制更好 |
| 2026-10 | 🆕 [何时切换动作块](https://arxiv.org/abs/2610.05719) `arXiv:2610.05719` | POSTECH | 可靠地延长动作块，减少"走走停停" |
| 2026-10 | 🆕 [Kinematic MeanFlow](https://arxiv.org/abs/2610.00864) `arXiv:2610.00864` | Intel Labs China | 把 MeanFlow 用于机器人基础模型的一步动作生成 |
| 2026-09 | 🆕 [ActionPiece](https://arxiv.org/abs/2609.18487) `arXiv:2609.18487` | DeepCybo | 重新设计自回归 VLA 的动作 tokenizer，不只看逐点重建误差 |
| 2026-07 | 🆕 [TurboVLA](https://arxiv.org/abs/2607.27205) `arXiv:2607.27205` |  | 绕开"视觉→语言→动作"的 LLM 中心通路，RTX 4090 上 32 Hz、显存不到 1 GB |
| 2026-03 | 🆕 [FASTER](https://arxiv.org/abs/2603.19199) `arXiv:2603.19199` |  | 重新审视实时流式 VLA：不只追求轨迹平滑，更要降低对环境变化的反应延迟 |
| 2026-01 | 🆕 [DynamicVLA](https://arxiv.org/abs/2601.22153) `arXiv:2601.22153` | NTU | 针对推理延迟造成的感知-执行错位，操作运动中的物体 |
| 2025-11 | 🆕 [VLASH](https://arxiv.org/abs/2512.01031) `arXiv:2512.01031` |  | 以"动作执行时刻的预测状态"为条件做异步推理，补偿 VLA 延迟 |
| 2025-10 | 🆕 [ACG](https://arxiv.org/abs/2510.22201) `arXiv:2510.22201` | KAIST | 动作连贯性引导，减少流式 VLA 的抖动与停顿 |
| 2025-06 | ★ [实时分块执行（RTC）](https://arxiv.org/abs/2506.07339) `arXiv:2506.07339` | Physical Intelligence | 推理期间冻结已承诺的动作、补全下一块，让慢推理的流匹配策略也能平滑实时执行 |
| 2025-05 | [知识隔离 VLA](https://arxiv.org/abs/2505.23705) `arXiv:2505.23705` | Physical Intelligence | 训练时阻断动作专家的梯度回流，保留 VLM 知识，训练快、推理快、泛化好 |
| 2025-02 | ★ [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) `arXiv:2502.19645` | Stanford | 并行解码 + 动作分块 + 连续动作 + L1 回归的微调配方，LIBERO 平均成功率 76.5% → 97.1% |
| 2025-01 | ★ [FAST](https://arxiv.org/abs/2501.09747) `arXiv:2501.09747` | Physical Intelligence | 用离散余弦变换压缩动作块做 token，自回归 VLA 训练快、能做高频灵巧任务 |

### VLA/机器人基础模型（65）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [PDRVLA](https://arxiv.org/abs/2610.37381) `arXiv:2610.37381` |  | 轻量物理动力学模块从视觉语言特征提取物理 token 条件化动作，不需在线视频生成 |
| 2026-10 | 🆕 [RynnVLA-Latent](https://arxiv.org/abs/2610.34356) `arXiv:2610.34356` | Alibaba DAMO | 把异构交互视频转成 3D 落地的潜在动作，作为可扩展的预训练监督 |
| 2026-10 | 🆕 [VLA-Δ](https://arxiv.org/abs/2610.21306) `arXiv:2610.21306` |  | 显式监督物理状态转移，而不只拟合示范动作 |
| 2026-09 | 🆕 [执行误差下的 VLA](https://arxiv.org/abs/2609.37334) `arXiv:2609.37334` | POSTECH | 部署时自补偿机械磨损、负载变化带来的执行偏差，并给出压力测试 |
| 2026-09 | 🆕 [MotorMind](https://arxiv.org/abs/2609.38078) `arXiv:2609.38078` |  | 为通用 VLM 搭脚手架，零样本做机器人操作 |
| 2026-09 | 🆕 [CARE](https://arxiv.org/abs/2609.24118) `arXiv:2609.24118` |  | 从执行失败中学习原子级纠正动作，提升恢复能力 |
| 2026-09 | 🆕 [Grounded Action Model](https://arxiv.org/abs/2609.23863) `arXiv:2609.23863` | NUS | 以 3D 定位为基础的机器人基础模型：语言、点、框提示统一成以物体为中心的表示 |
| 2026-09 | 🆕 [PhysBrain 1.5](https://arxiv.org/abs/2609.14973) `arXiv:2609.14973` | DeepCybo | 用人类交互视频预训练，把理解、动作与未来预测统一在一个模型里 |
| 2026-08 | 🆕 [表征中心的继续预训练](https://arxiv.org/abs/2608.27550) `arXiv:2608.27550` |  | 机器人数据难扩时，提升表征质量比堆数据更关键 |
| 2026-08 | 🆕 [GigaBrain-0.7](https://arxiv.org/abs/2608.15875) `arXiv:2608.15875` | GigaAI | 三系统架构的具身基础模型，研究能力随规模的涌现 |
| 2026-07 | 🆕 [RynnBrain 1.1](https://arxiv.org/abs/2607.17977) `arXiv:2607.17977` | Alibaba DAMO | 更强、更可泛化的具身"大脑"基础模型 |
| 2026-07 | ★ 🆕 [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) `arXiv:2607.15330` | Xiaomi | 用 10 万小时以上 UMI 采集的真实操作轨迹预训练，开箱即用做移动操作，随数据与模型规模持续提升 |
| 2026-07 | 🆕 [双重潜在记忆 VLA](https://arxiv.org/abs/2607.07608) `arXiv:2607.07608` |  | 给 VLA 加入潜在记忆，处理长时程、依赖历史的任务 |
| 2026-06 | 🆕 [Qwen-RobotManip](https://arxiv.org/abs/2606.17846) `arXiv:2606.17846` | Qwen | 从表征、运动、行为三个维度对齐来扩大操作基础模型规模 |
| 2026-05 | 🆕 [Qwen-VLA](https://arxiv.org/abs/2605.30280) `arXiv:2605.30280` | Qwen | 统一跨任务、跨环境、跨本体的 VLA 建模 |
| 2026-05 | 🆕 [RLDX-1](https://arxiv.org/abs/2605.03269) `arXiv:2605.03269` | RLWRLD | 面向通用操作的 VLA 技术报告 |
| 2026-05 | 🆕 [MolmoAct2](https://arxiv.org/abs/2605.02881) `arXiv:2605.02881` | AI2 | 全开源的动作推理模型，面向真实部署 |
| 2026-04 | ★ 🆕 [π0.7](https://arxiv.org/abs/2604.15483) `arXiv:2604.15483` | Physical Intelligence | 可用多模态上下文（策略、元数据、子目标图）操控的约 5B 通用模型，零样本跨本体 |
| 2026-04 | 🆕 [HY-Embodied-0.5](https://arxiv.org/abs/2604.07430) `arXiv:2604.07430` | Tencent | 面向真实世界 agent 的具身基础模型 |
| 2026-02 | 🆕 [VLANeXt](https://arxiv.org/abs/2602.18532) `arXiv:2602.18532` | NTU | 系统梳理 VLA 设计选择，给出强 VLA 配方 |
| 2026-02 | 🆕 [GigaBrain-0.5M*](https://arxiv.org/abs/2602.12099) `arXiv:2602.12099` | GigaAI | 基于世界模型的强化学习改进 VLA |
| 2026-02 | 🆕 [Recurrent-Depth VLA](https://arxiv.org/abs/2602.07845) `arXiv:2602.07845` | AI2 | 潜空间循环深度实现隐式测试时计算扩展，难的动作多想几步 |
| 2026-01 | 🆕 [Green-VLA](https://arxiv.org/abs/2602.00919) `arXiv:2602.00919` | Sber Robotics | 分阶段训练的通用机器人 VLA |
| 2026-01 | 🆕 [LingBot-VLA](https://arxiv.org/abs/2601.18692) `arXiv:2601.18692` | Ant Group (Robbyant) | 约 2 万小时、9 种双臂构型的真机数据训练，强调适配的数据与算力效率 |
| 2026-01 | 🆕 [LangForce](https://arxiv.org/abs/2601.15197) `arXiv:2601.15197` |  | 贝叶斯分解 VLA，缓解目标驱动数据带来的"忽视语言"偏差 |
| 2026-01 | 🆕 [Being-H0.5](https://arxiv.org/abs/2601.12993) `arXiv:2601.12993` | BeingBeyond | 以人为中心的跨本体 VLA：把人手动作当作通用动作语言 |
| 2026-01 | 🆕 [Fast-ThinkAct](https://arxiv.org/abs/2601.09708) `arXiv:2601.09708` | NVIDIA | 可言语化的潜在规划，保留推理收益同时降低推理延迟 |
| 2025-12 | 🆕 [Dream-VLA](https://arxiv.org/abs/2512.22615) `arXiv:2512.22615` | HKU | 以扩散语言模型为骨干的开放 VLA |
| 2025-12 | 🆕 [VLA 反探索测试时扩展](https://arxiv.org/abs/2512.02834) `arXiv:2512.02834` |  | 把测试时扩展视为"反探索"：在推理时筛掉偏离数据分布的动作候选 |
| 2025-11 | 🆕 [RynnVLA-002](https://arxiv.org/abs/2511.17502) `arXiv:2511.17502` | Alibaba DAMO | VLA 与世界模型统一：世界模型预测未来图像以改进动作，VLA 反过来帮助视觉预测 |
| 2025-11 | ★ 🆕 [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759) `arXiv:2511.14759` | Physical Intelligence | 优势条件化的经验强化：融合示范、自主数据与人工干预，让 VLA 在部署中持续变强；最难的任务上吞吐量翻倍以上、失败率约减半（叠衣、装箱、做意式咖啡） |
| 2025-10 | 🆕 [RoboOmni](https://arxiv.org/abs/2510.23763) `arXiv:2510.23763` | Fudan | 全模态上下文（语音、环境声、视觉）中主动推断用户意图的操作模型 |
| 2025-10 | 🆕 [GigaBrain-0](https://arxiv.org/abs/2510.19430) `arXiv:2510.19430` | GigaAI | 用世界模型生成的数据扩充 VLA 训练 |
| 2025-10 | 🆕 [InternVLA-M1](https://arxiv.org/abs/2510.13778) `arXiv:2510.13778` | Shanghai AI Lab | 空间引导的 VLA：先学"在哪里"，再学"怎么动" |
| 2025-10 | 🆕 [X-VLA](https://arxiv.org/abs/2510.10274) `arXiv:2510.10274` | Tsinghua 等 | 软提示 Transformer：每种本体一组可学习提示，可扩展的跨本体 VLA |
| 2025-10 | 🆕 [D2E](https://arxiv.org/abs/2510.05684) `arXiv:2510.05684` |  | 用桌面游戏/GUI 交互数据做视觉-动作预训练，再迁移到具身任务 |
| 2025-10 | ★ 🆕 [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) `arXiv:2510.03342` | Google DeepMind | "先思考再行动"的 VLA + 跨本体 Motion Transfer；ER 1.5 编排 VLA 与工具 |
| 2025-09 | [WALL-OSS](https://arxiv.org/abs/2509.11766) `arXiv:2509.11766` | X Square Robot | 统一跨层思维链的具身基础模型 |
| 2025-09 | [VLA-Adapter](https://arxiv.org/abs/2509.09372) `arXiv:2509.09372` |  | 用轻量桥接把小 VLM 变成强 VLA，0.5B 骨干即可接近大模型 |
| 2025-08 | [EO-1](https://arxiv.org/abs/2508.21112) `arXiv:2508.21112` | Shanghai AI Lab 等 | 视觉-文本-动作交错预训练的开源统一具身基础模型 |
| 2025-08 | [MolmoAct](https://arxiv.org/abs/2508.07917) `arXiv:2508.07917` | AI2 | "动作推理模型"：先做深度感知与轨迹规划，再输出动作 |
| 2025-07 | [GR-3](https://arxiv.org/abs/2507.15493) `arXiv:2507.15493` | ByteDance Seed | 大规模 VLA 技术报告，覆盖长时程与灵巧操作 |
| 2025-06 | [SmolVLA](https://arxiv.org/abs/2506.01844) `arXiv:2506.01844` | Hugging Face | 小型开源 VLA + 异步推理，消费级硬件可训练 |
| 2025-05 | [OneTwoVLA](https://arxiv.org/abs/2505.11917) `arXiv:2505.11917` | Tsinghua | 单个模型自适应地在推理与执行之间切换 |
| 2025-05 | [UniVLA](https://arxiv.org/abs/2505.06111) `arXiv:2505.06111` | HKU/OpenDriveLab | 以任务为中心的潜在动作，可从无动作标签视频学习、跨本体部署 |
| 2025-04 | ★ [π0.5](https://arxiv.org/abs/2504.16054) `arXiv:2504.16054` | Physical Intelligence | 异构数据共训，开放世界泛化；同一模型先解码子任务文本再生成动作，能在陌生家庭里收拾 |
| 2025-03 | ★ [Gemini Robotics](https://arxiv.org/abs/2503.20020) `arXiv:2503.20020` | Google DeepMind | 基于 Gemini 2.0 的 VLA 与具身推理（ER）模型，提出 ERQA 基准 |
| 2025-03 | [GR00T N1](https://arxiv.org/abs/2503.14734) `arXiv:2503.14734` | NVIDIA | 开源人形基础模型，双系统：VLM（10 Hz）+ 扩散动作模块（120 Hz） |
| 2025-02 | [Hi Robot](https://arxiv.org/abs/2502.19417) `arXiv:2502.19417` | Physical Intelligence | 分层 VLA：高层 VLM 解析开放指令与用户插话，低层 π0 执行 |
| 2025-02 | [Magma](https://arxiv.org/abs/2502.13130) `arXiv:2502.13130` | Microsoft | 同时面向 UI 操作与机器人操作的多模态 agent 基础模型（Set-of-Mark / Trace-of-Mark） |
| 2025-01 | [SpatialVLA](https://arxiv.org/abs/2501.15830) `arXiv:2501.15830` | Shanghai AI Lab | 自适应动作网格 + 自我中心 3D 位置编码，增强空间表示 |
| 2024-12 | [RoboVLMs](https://arxiv.org/abs/2412.14058) `arXiv:2412.14058` | Tsinghua/ByteDance | 系统研究"怎样搭 VLA"：骨干、结构、跨本体数据各自的影响 |
| 2024-11 | [CogACT](https://arxiv.org/abs/2411.19650) `arXiv:2411.19650` | Microsoft 等 | 认知（VLM）与动作（扩散 Transformer）模块协同的 VLA |
| 2024-10 | ★ [π0](https://arxiv.org/abs/2410.24164) `arXiv:2410.24164` | Physical Intelligence | VLM + 流匹配动作专家，跨本体灵巧操作（叠衣服、收餐桌） |
| 2024-10 | [RDT-1B](https://arxiv.org/abs/2410.07864) `arXiv:2410.07864` | Tsinghua | 双臂操作的 1.2B 扩散基础模型，统一动作空间 |
| 2024-10 | [GR-2](https://arxiv.org/abs/2410.06158) `arXiv:2410.06158` | ByteDance | 网络规模视频-语言预训练的生成式视频-语言-动作模型 |
| 2024-09 | [HPT](https://arxiv.org/abs/2409.20537) `arXiv:2409.20537` | MIT | 异构预训练 Transformer：本体专属的"茎" + 共享"躯干"，在大量异构本体数据上预训练 |
| 2024-08 | [CrossFormer](https://arxiv.org/abs/2408.11812) `arXiv:2408.11812` | UC Berkeley | 一个策略同时控制操作、导航、四足与无人机 |
| 2024-06 | ★ [OpenVLA](https://arxiv.org/abs/2406.09246) `arXiv:2406.09246` | Stanford/UC Berkeley 等 | 7B 开源 VLA，开源 VLA 生态由此起步 |
| 2024-05 | [Octo](https://arxiv.org/abs/2405.12213) `arXiv:2405.12213` | UC Berkeley 等 | 开源通用策略（Transformer + 扩散动作头），可快速适配新本体 |
| 2023-12 | [GR-1](https://arxiv.org/abs/2312.13139) `arXiv:2312.13139` | ByteDance | 大规模视频生成预训练 + 机器人数据微调的 GPT 式策略 |
| 2023-10 | ★ [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) `arXiv:2310.08864` | 多机构合作 | 21 家机构、22 种机器人、527 种技能的开放数据集与跨本体 RT-X 模型 |
| 2023-07 | ★ [RT-2](https://arxiv.org/abs/2307.15818) `arXiv:2307.15818` | Google DeepMind | 把动作当作语言 token 来微调 VLM，提出 VLA 概念，网络知识迁移到操作 |
| 2022-12 | ★ [RT-1](https://arxiv.org/abs/2212.06817) `arXiv:2212.06817` | Google | 用约 13 万条真机演示训练的 Transformer 机器人策略 |
| 2022-10 | [VIMA](https://arxiv.org/abs/2210.03094) `arXiv:2210.03094` | Stanford/NVIDIA | 多模态提示（文本 + 图像交错）统一表达多种操作任务 |

### 推理、规划与空间智能（24）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [GroundingPI](https://arxiv.org/abs/2609.39601) `arXiv:2609.39601` | XPENG | 4B 物理智能定位基础模型：以量化坐标输出点与框，SFT + GRPO 训练 |
| 2026-09 | 🆕 [简单的机器人上下文学习](https://arxiv.org/abs/2609.38173) `arXiv:2609.38173` |  | 从视觉示范在上下文中推断并执行新任务的通用配方 |
| 2026-09 | 🆕 [机器人上下文学习综述](https://arxiv.org/abs/2609.36012) `arXiv:2609.36012` |  | 按"上下文证据如何接到执行"梳理四类方法：上下文条件策略、几何迁移、世界模型控制、技能与 agent |
| 2026-09 | 🆕 [RoboDawn](https://arxiv.org/abs/2609.22966) `arXiv:2609.22966` | Tsinghua | 用少量离散平移/旋转/夹爪命令把机器人控制交给 agentic VLM |
| 2026-06 | 🆕 [VLA 还剩多少常识](https://arxiv.org/abs/2606.19297) `arXiv:2606.19297` |  | 测量 VLM 微调成 VLA 后保留了多少常识与世界知识 |
| 2026-06 | 🆕 [Playful Agentic Robot Learning](https://arxiv.org/abs/2606.19419) `arXiv:2606.19419` | UC Berkeley | 具身编码 agent 在任务到来前自主"玩耍"，持续积累技能 |
| 2026-06 | 🆕 [Embodied-R1.5](https://arxiv.org/abs/2606.11324) `arXiv:2606.11324` |  | 统一的具身推理模型（认知、规划、纠正、指点），配合规划-落地-纠正闭环 |
| 2026-02 | 🆕 [RynnBrain](https://arxiv.org/abs/2602.14979) `arXiv:2602.14979` | Alibaba DAMO | 开源时空具身基础模型：第一人称理解、时空定位、物理落地的推理与规划 |
| 2026-02 | 🆕 [VLS](https://arxiv.org/abs/2602.03973) `arXiv:2602.03973` | AI2 | 免训练地用 VLM 在推理时引导冻结的扩散/流策略的采样，适配新约束 |
| 2025-12 | 🆕 [RoboTracer](https://arxiv.org/abs/2512.13660) `arXiv:2512.13660` | BAAI | VLM 的空间轨迹推理（度量、指代、描迹），并提出 TraceSpatial-Bench |
| 2025-11 | 🆕 [MiMo-Embodied](https://arxiv.org/abs/2511.16518) `arXiv:2511.16518` | Xiaomi | 首个同时覆盖自动驾驶与具身 AI 的跨本体基础模型 |
| 2025-11 | 🆕 [Visual Spatial Tuning](https://arxiv.org/abs/2511.05491) `arXiv:2511.05491` | ByteDance Seed | 不加额外编码器，增强 VLM 的空间感知与推理 |
| 2025-07 | [ThinkAct](https://arxiv.org/abs/2507.16815) `arXiv:2507.16815` | NVIDIA | 用强化学习的视觉潜在规划增强 VLA 推理 |
| 2025-07 | [RoboBrain 2.0](https://arxiv.org/abs/2507.02029) `arXiv:2507.02029` | BAAI | 具身"大脑"VLM：空间理解与时序决策 |
| 2025-03 | [Cosmos-Reason1](https://arxiv.org/abs/2503.15558) `arXiv:2503.15558` | NVIDIA | 物理常识与具身推理 VLM；后续 Cosmos Reason 2 成为 GR00T N1.6 的骨干 |
| 2024-09 | [ReKep](https://arxiv.org/abs/2409.01652) `arXiv:2409.01652` | Stanford | VLM 生成关系关键点约束，优化求解时空动作 |
| 2024-07 | [ECoT](https://arxiv.org/abs/2407.08693) `arXiv:2407.08693` | UC Berkeley/Stanford 等 | 具身思维链：先推理子任务、物体、运动，再输出动作 |
| 2024-03 | [MOKA](https://arxiv.org/abs/2403.03174) `arXiv:2403.03174` | UC Berkeley | 基于标记的视觉提示，让 VLM 指出关键点完成开放世界操作 |
| 2024-01 | [AutoRT](https://arxiv.org/abs/2401.12963) `arXiv:2401.12963` | Google DeepMind | VLM/LLM 编排机器人舰队在真实环境中自主采集数据 |
| 2023-11 | [RT-Trajectory](https://arxiv.org/abs/2311.01977) `arXiv:2311.01977` | Google DeepMind | 用 2D 轨迹草图作为任务条件，泛化到新任务 |
| 2023-07 | [VoxPoser](https://arxiv.org/abs/2307.05973) `arXiv:2307.05973` | Stanford | LLM 写代码调用 VLM，组合出 3D 价值图，交给运动规划执行 |
| 2022-09 | [Code as Policies](https://arxiv.org/abs/2209.07753) `arXiv:2209.07753` | Google | LLM 写 Python 程序组合感知/控制 API，程序本身就是策略 |
| 2022-07 | [Inner Monologue](https://arxiv.org/abs/2207.05608) `arXiv:2207.05608` | Google | 把成功检测、场景描述、人类反馈作为文本回灌 LLM，形成闭环规划 |
| 2022-04 | [SayCan](https://arxiv.org/abs/2204.01691) `arXiv:2204.01691` | Google | LLM 给技能打"是否有用"分，价值函数给"是否可行"分，相乘选下一步 |

### 数据采集与规模化（21）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | 🆕 [HiFi-UMI](https://arxiv.org/abs/2607.25895) `arXiv:2607.25895` | Simple AI | 只用高保真 UMI 数据就训出可部署的操作策略，不再依赖真机遥操作 |
| 2026-07 | 🆕 [数据金字塔](https://arxiv.org/abs/2607.24744) `arXiv:2607.24744` | PKU | 按"网络数据—人类视频—仿真—真机"分层综述操作数据 |
| 2026-03 | 🆕 [RoboPocket](https://arxiv.org/abs/2603.05504) `arXiv:2603.05504` | SJTU | 用手机做闭环采集：根据当前策略的失败点有针对性地补数据 |
| 2025-05 | [DexUMI](https://arxiv.org/abs/2505.21864) `arXiv:2505.21864` | Stanford/Columbia | 把人手当作通用操作接口：外骨骼缩小运动学差距，视频中把人手修补成机器人手 |
| 2025-03 | ★ [AgiBot World](https://arxiv.org/abs/2503.06669) `arXiv:2503.06669` | AgiBot | 100 万条以上轨迹、217 个任务的大规模平台，并提出潜在动作策略 GO-1 |
| 2025-02 | [HOMIE](https://arxiv.org/abs/2502.13013) `arXiv:2502.13013` |  | 同构外骨骼座舱，遥操作人形移动操作 |
| 2024-12 | [RoboMIND](https://arxiv.org/abs/2412.13877) `arXiv:2412.13877` | Beijing Humanoid Center | 多本体遥操作数据集与基准 |
| 2024-10 | [ALOHA Unleashed](https://arxiv.org/abs/2410.13126) `arXiv:2410.13126` | Google DeepMind | 大规模示范 + 扩散策略，学会系鞋带、挂衣服等高灵巧双手任务 |
| 2024-09 | [FastUMI](https://arxiv.org/abs/2409.19499) `arXiv:2409.19499` |  | 与硬件解耦的可扩展 UMI 方案，附数据集 |
| 2024-07 | [UMI on Legs](https://arxiv.org/abs/2407.10353) `arXiv:2407.10353` | Stanford/Columbia | 把 UMI 学到的操作策略装到四足移动平台上，用以操作为中心的全身控制器 |
| 2024-07 | [Open-TeleVision](https://arxiv.org/abs/2407.01512) `arXiv:2407.01512` | UCSD/MIT | 沉浸式主动视觉遥操作，广泛用于人形双手数据采集 |
| 2024-03 | ★ [DROID](https://arxiv.org/abs/2403.12945) `arXiv:2403.12945` | Stanford 等 | 7.6 万条轨迹（350 小时）、564 个场景的野外操作数据，由 50 位采集者在多地采集 |
| 2024-03 | [OPEN TEACH](https://arxiv.org/abs/2403.07870) `arXiv:2403.07870` | NYU | 基于 VR 头显的通用遥操作系统 |
| 2024-03 | [DexCap](https://arxiv.org/abs/2403.07788) `arXiv:2403.07788` | Stanford | 便携动捕手套 + 相机采集人手数据，配合人在环纠正学双手技能 |
| 2024-02 | ★ [UMI](https://arxiv.org/abs/2402.10329) `arXiv:2402.10329` | Stanford/Columbia | 手持夹爪 + GoPro 在野外采示范，不需要机器人；数据可直接迁移到多种机械臂 |
| 2024-01 | ★ [Mobile ALOHA](https://arxiv.org/abs/2401.02117) `arXiv:2401.02117` | Stanford | 全身遥操作的低成本双臂移动平台，与静态 ALOHA 数据共训明显提升 |
| 2023-09 | [AirExo](https://arxiv.org/abs/2309.14975) `arXiv:2309.14975` | SJTU | 低成本外骨骼在野外采集全臂操作示范 |
| 2023-09 | [GELLO](https://arxiv.org/abs/2309.13037) `arXiv:2309.13037` | UC Berkeley | 按比例缩小的低成本主从臂，直观遥操作各种机械臂 |
| 2023-08 | [BridgeData V2](https://arxiv.org/abs/2308.12952) `arXiv:2308.12952` | UC Berkeley | 6 万条轨迹、24 个环境的低成本桌面操作数据集 |
| 2023-07 | [AnyTeleop](https://arxiv.org/abs/2307.04577) `arXiv:2307.04577` | UCSD/NVIDIA | 通用的纯视觉遥操作框架，支持多种机械臂与灵巧手 |
| 2023-07 | [RH20T](https://arxiv.org/abs/2307.00595) `arXiv:2307.00595` | SJTU | 11 万多条接触丰富的多模态（视觉、力、音频）操作序列 |

### 人类视频与跨本体（24）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [RLHND](https://arxiv.org/abs/2610.09455) `arXiv:2610.09455` | RLWRLD | 把视频基础模型用作物理一致的手部跟踪器，服务机器人学习 |
| 2026-07 | 🆕 [ACE-Data-0](https://arxiv.org/abs/2607.28625) `arXiv:2607.28625` | NTU | 把真实家庭变成标定同步的采集"录音棚"，以人为中心的环境式数据引擎 |
| 2026-07 | 🆕 [Open-AoE](https://arxiv.org/abs/2607.14183) `arXiv:2607.14183` | inclusionAI | 开放的第一人称操作数据集与工具链 |
| 2026-07 | 🆕 [LingBot-Video](https://arxiv.org/abs/2607.07675) `arXiv:2607.07675` | Ant Group (Robbyant) | 开源大规模 MoE 视频基础模型，服务具身智能预训练 |
| 2026-06 | 🆕 [以平移为桥](https://arxiv.org/abs/2606.28133) `arXiv:2606.28133` | ByteDance Seed | 把人类动作中的平移作为桥接动作，迁移到平行夹爪双臂 |
| 2026-06 | 🆕 [MolmoMotion](https://arxiv.org/abs/2606.18558) `arXiv:2606.18558` | AI2 | 语言条件的 3D 点轨迹预测：以世界坐标点作为通用运动表示 |
| 2026-06 | 🆕 [ACE-Ego-0](https://arxiv.org/abs/2606.17200) `arXiv:2606.17200` | CUHK | 统一第一人称人类数据与机器人数据的 VLA 预训练 |
| 2026-05 | 🆕 [PhysBrain 1.0](https://arxiv.org/abs/2605.15298) `arXiv:2605.15298` | DeepCybo | 把大规模第一人称视频转成结构化的物理常识监督，再接机器人训练 |
| 2026-05 | 🆕 [HumanNet](https://arxiv.org/abs/2605.06747) `arXiv:2605.06747` |  | 100 万小时、富标注的以人为中心视频库，用于物理交互学习 |
| 2026-04 | 🆕 [UniT](https://arxiv.org/abs/2604.19734) `arXiv:2604.19734` | XPENG | 以视觉锚定的统一潜在动作 tokenizer，作为人到人形迁移的共同"物理语言" |
| 2026-02 | ★ 🆕 [EgoScale](https://arxiv.org/abs/2602.16710) `arXiv:2602.16710` |  | 2 万多小时带动作标签的第一人称人类视频训练 VLA，人类数据规模与验证损失呈对数线性关系 |
| 2025-12 | 🆕 [PhysBrain](https://arxiv.org/abs/2512.16793) `arXiv:2512.16793` | DeepCybo | 用第一人称人类数据把 VLM 引向物理智能 |
| 2025-07 | [Being-H0](https://arxiv.org/abs/2507.15597) `arXiv:2507.15597` | BeingBeyond/PKU | 把人手当"基础操作器"，大规模人类视频做物理指令微调 |
| 2025-07 | [EgoVLA](https://arxiv.org/abs/2507.12440) `arXiv:2507.12440` | UCSD 等 | 用第一人称人类视频训练 VLA 预测手腕与手部动作，再转成机器人动作 |
| 2025-05 | [EgoDex](https://arxiv.org/abs/2505.11709) `arXiv:2505.11709` | Apple | Vision Pro 采集带手部位姿标注的大规模第一人称灵巧操作视频 |
| 2025-03 | [HAT](https://arxiv.org/abs/2503.13441) `arXiv:2503.13441` | UCSD 等 | 人形策略≈人类策略：把第一人称人类数据当作另一种"本体"共训 |
| 2025-03 | [Phantom](https://arxiv.org/abs/2503.00779) `arXiv:2503.00779` | Stanford | 只用人类视频训练机器人：把视频中的人手替换渲染成机器人 |
| 2024-10 | [EgoMimic](https://arxiv.org/abs/2410.24221) `arXiv:2410.24221` | Georgia Tech | 第一人称人类视频与机器人数据统一建模，扩大模仿学习 |
| 2024-10 | ★ [LAPA](https://arxiv.org/abs/2410.11758) `arXiv:2410.11758` | KAIST/Microsoft | 从无动作标签的视频学潜在动作做预训练，超过用真动作标签预训练的 VLA |
| 2024-09 | [Gen2Act](https://arxiv.org/abs/2409.16283) `arXiv:2409.16283` | Google DeepMind | 先生成"人做这件事"的视频，再由策略据此执行，泛化到新物体与新动作 |
| 2024-07 | [Im2Flow2Act](https://arxiv.org/abs/2407.15208) `arXiv:2407.15208` | Columbia | 用物体光流作为跨本体接口，人类视频 + 仿真数据学操作 |
| 2024-05 | [Track2Act](https://arxiv.org/abs/2405.01527) `arXiv:2405.01527` | CMU | 从互联网视频学点轨迹预测，再加残差策略实现零样本操作 |
| 2023-12 | [ATM](https://arxiv.org/abs/2401.00025) `arXiv:2401.00025` | UC Berkeley | 预测图像中任意点的未来轨迹，作为视频到策略的桥梁 |
| 2023-02 | [MimicPlay](https://arxiv.org/abs/2302.12422) `arXiv:2302.12422` | Stanford/NVIDIA | 从人类"玩耍"视频学高层潜在计划，机器人只需少量示范 |

### 仿真与合成数据（17）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🆕 [Lucida](https://arxiv.org/abs/2608.30821) `arXiv:2608.30821` | ByteDance Seed | 解析、生成、摆放：可组合的 real-to-sim 场景建模 |
| 2025-11 | 🆕 [GigaWorld-0](https://arxiv.org/abs/2511.19861) `arXiv:2511.19861` | GigaAI | 把世界模型当作 VLA 训练的数据引擎 |
| 2025-11 | 🆕 [Isaac Lab](https://arxiv.org/abs/2511.04831) `arXiv:2511.04831` | NVIDIA | Isaac Gym 的继任者，统一的 GPU 机器人学习框架 |
| 2025-06 | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) `arXiv:2506.18088` | HKU 等 | 可扩展的双臂数据生成器与强域随机化基准 |
| 2025-04 | [RoboVerse](https://arxiv.org/abs/2504.18904) `arXiv:2504.18904` | UC Berkeley 等 | 统一的仿真平台、数据集与基准，打通多种模拟器 |
| 2025-02 | [MuJoCo Playground](https://arxiv.org/abs/2502.08844) `arXiv:2502.08844` | Google DeepMind 等 | 基于 MJX 的开源机器人学习框架，单 GPU 几分钟训练，零样本上真机 |
| 2024-10 | [DexMimicGen](https://arxiv.org/abs/2410.24185) `arXiv:2410.24185` | NVIDIA | 把 MimicGen 扩展到双手灵巧操作 |
| 2024-10 | [ManiSkill3](https://arxiv.org/abs/2410.00425) `arXiv:2410.00425` | UCSD 等 | GPU 并行仿真与渲染，支持大规模视觉操作 RL |
| 2024-09 | [SplatSim](https://arxiv.org/abs/2409.10161) `arXiv:2409.10161` | CMU | 用高斯泼溅渲染替换仿真画面，RGB 策略零样本 sim2real |
| 2024-09 | [RoboTwin](https://arxiv.org/abs/2409.02920) `arXiv:2409.02920` | HKU 等 | 生成式数字孪生的双臂操作基准 |
| 2024-03 | [RialTo](https://arxiv.org/abs/2403.03949) `arXiv:2403.03949` | MIT | real-to-sim-to-real：快速重建场景为仿真，在其中 RL 提升鲁棒性 |
| 2023-11 | [RoboGen](https://arxiv.org/abs/2311.01455) `arXiv:2311.01455` | CMU/MIT | 生成式仿真：LLM 提出任务、生成场景与奖励，自动学技能 |
| 2023-10 | ★ [MimicGen](https://arxiv.org/abs/2310.17596) `arXiv:2310.17596` | NVIDIA | 从约 200 条人类示范自动扩增出 5 万条仿真示范 |
| 2023-10 | [Eureka](https://arxiv.org/abs/2310.12931) `arXiv:2310.12931` | NVIDIA | LLM 写奖励代码并迭代改进，达到人类专家水平 |
| 2023-02 | [ManiSkill2](https://arxiv.org/abs/2302.04659) `arXiv:2302.04659` | UCSD | 可泛化操作技能的统一基准 |
| 2021-08 | [Isaac Gym](https://arxiv.org/abs/2108.10470) `arXiv:2108.10470` | NVIDIA | 物理与训练全在 GPU 上，大规模并行 RL 的基础设施 |
| 2020-09 | [robosuite](https://arxiv.org/abs/2009.12293) `arXiv:2009.12293` | Stanford/UT Austin | 模块化操作仿真框架与基准（MuJoCo） |

### 强化学习与后训练（26）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [舰队经验学习](https://arxiv.org/abs/2610.25736) `arXiv:2610.25736` | RLinf | 用部署中机器人自己的执行经验作监督，减少对专家纠正的依赖 |
| 2026-07 | 🆕 [WCM](https://arxiv.org/abs/2607.29613) `arXiv:2607.29613` | Fudan | 世界评判模型：用视频世界模型给 VLA 的 RL 提供多帧价值估计 |
| 2026-07 | 🆕 [进度奖励综述](https://arxiv.org/abs/2607.21655) `arXiv:2607.21655` | Northwestern | 机器人学习中进度奖励建模的综述 |
| 2026-05 | 🆕 [Learning While Deploying](https://arxiv.org/abs/2605.00416) `arXiv:2605.00416` |  | 在机器人舰队规模上对通用策略做强化学习 |
| 2026-04 | 🆕 [RL Token](https://arxiv.org/abs/2604.23073) `arXiv:2604.23073` | Physical Intelligence | 从 VLA 读出一个紧凑的"RL token"，小型 actor-critic 头在线 RL，几小时内关键阶段提速最高 3 倍 |
| 2026-02 | 🆕 [TOPReward](https://arxiv.org/abs/2602.19313) `arXiv:2602.19313` | AI2 | 用 VLM 的 token 概率做零样本、按指令条件的稠密进度奖励 |
| 2026-02 | 🆕 [RISE](https://arxiv.org/abs/2602.11075) `arXiv:2602.11075` | OpenDriveLab | 组合式世界模型中的自我改进：在想象里 RL，提升接触丰富与动态任务 |
| 2026-02 | 🆕 [RLinf-USER](https://arxiv.org/abs/2602.07837) `arXiv:2602.07837` | RLinf | 统一可扩展的真机在线策略学习系统 |
| 2026-01 | 🆕 [SOP](https://arxiv.org/abs/2601.03044) `arXiv:2601.03044` | AgiBot | 可扩展的在线后训练系统：多机器人舰队上同时做 VLA 的在线改进 |
| 2025-12 | 🆕 [GR-RL](https://arxiv.org/abs/2512.01801) `arXiv:2512.01801` | ByteDance Seed | 把通用 VLA 训成长时程、精密任务（如系鞋带）的专家 |
| 2025-11 | 🆕 [SRPO](https://arxiv.org/abs/2511.15605) `arXiv:2511.15605` | Fudan | 自参照策略优化：用模型自己成功的轨迹作参照，给失败轨迹打进度奖励 |
| 2025-10 | 🆕 [πRL](https://arxiv.org/abs/2510.25889) `arXiv:2510.25889` | RLinf | 对流匹配 VLA（π0/π0.5）做在线 RL 微调 |
| 2025-10 | 🆕 [RL-100](https://arxiv.org/abs/2510.14830) `arXiv:2510.14830` |  | 扩散策略上的真机 RL 框架：模仿 → 离线 RL → 在线 RL |
| 2025-10 | 🆕 [RLinf-VLA](https://arxiv.org/abs/2510.06710) `arXiv:2510.06710` | RLinf | 统一高效的 VLA + RL 训练框架，支持多种架构与算法的公平比较 |
| 2025-09 | [ResFiT](https://arxiv.org/abs/2509.19301) `arXiv:2509.19301` |  | 离策略残差 RL 微调行为克隆策略（含高自由度双手人形） |
| 2025-09 | [RLinf](https://arxiv.org/abs/2509.15965) `arXiv:2509.15965` | RLinf | 面向具身与推理的大规模 RL 训练框架 |
| 2025-09 | ★ [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) `arXiv:2509.09674` | Tsinghua 等 | GRPO 式结果奖励 RL 用于 VLA，少量示范 + RL 即可大幅提升，并观察到新动作模式"pushcut" |
| 2025-07 | [Q-chunking](https://arxiv.org/abs/2507.07969) `arXiv:2507.07969` | UC Berkeley | 在动作块空间做 RL，改善长时程稀疏奖励的离线到在线学习 |
| 2025-05 | [VLA-RL](https://arxiv.org/abs/2505.18719) `arXiv:2505.18719` | Tsinghua | 在线 RL 扩展自回归 VLA，配合机器人过程奖励模型 |
| 2025-05 | [RIPT-VLA](https://arxiv.org/abs/2505.17016) `arXiv:2505.17016` | UT Austin | 交互式后训练：只用稀疏二值奖励的 VLA 强化微调 |
| 2025-02 | [ConRFT](https://arxiv.org/abs/2502.05450) `arXiv:2502.05450` | CASIA | 一致性策略 + 离线/在线强化微调 VLA，真机 45–90 分钟达到高成功率 |
| 2024-12 | [Policy Decorator](https://arxiv.org/abs/2412.13630) `arXiv:2412.13630` | UCSD | 与模型无关的在线残差 RL 微调大策略 |
| 2024-12 | [PA-RL](https://arxiv.org/abs/2412.06685) `arXiv:2412.06685` | Stanford/UC Berkeley | 对任意类别、任意骨干的策略做离线与在线 RL 微调：采样后用 Q 函数优化，再监督学习 |
| 2024-10 | ★ [HIL-SERL](https://arxiv.org/abs/2410.21845) `arXiv:2410.21845` | UC Berkeley | 人在环真机 RL：示范 + 人工纠正 + 离策略 RL，精密装配与动态任务接近 100% 成功 |
| 2024-09 | [DPPO](https://arxiv.org/abs/2409.00588) `arXiv:2409.00588` | Princeton | 用策略梯度微调扩散策略，把去噪过程纳入 MDP |
| 2024-01 | [SERL](https://arxiv.org/abs/2401.16013) `arXiv:2401.16013` | UC Berkeley | 样本高效的真机 RL 软件套件，几十分钟学会插接等任务 |

### 世界模型/WAM（44）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Rollback](https://arxiv.org/abs/2610.36068) `arXiv:2610.36068` | AgiBot | 在世界模型中"回滚"经验，反事实地改进策略 |
| 2026-10 | 🆕 [Video-Point Model](https://arxiv.org/abs/2610.28594) `arXiv:2610.28594` |  | 多视角视频 + 3D 点轨迹联合预测的世界模型 |
| 2026-10 | 🆕 [Ropedia W0](https://arxiv.org/abs/2610.26855) `arXiv:2610.26855` | Ropedia | 跨本体的第一人称移动操作世界模型，用人类经验数据扩展 |
| 2026-10 | 🆕 [WAVE-0](https://arxiv.org/abs/2610.18568) `arXiv:2610.18568` |  | 轻量潜在未来预测，桥接 VLA 与 WAM |
| 2026-10 | 🆕 [GreenWAM](https://arxiv.org/abs/2610.18774) `arXiv:2610.18774` | Sber Robotics | 一个机器人世界模型同时充当模拟器与策略 |
| 2026-10 | 🆕 [DreamTrue](https://arxiv.org/abs/2610.12468) `arXiv:2610.12468` |  | 多视角、跨本体、动作忠实的机器人世界模型，反事实后训练 |
| 2026-10 | 🆕 [Long-WAM](https://arxiv.org/abs/2610.10528) `arXiv:2610.10528` | NVIDIA | 在实时控制约束下扩展因果 WAM 的历史上下文 |
| 2026-10 | 🆕 [原生动作先验](https://arxiv.org/abs/2610.03391) `arXiv:2610.03391` | Meta | 从无动作标签视频中学习 WAM 的原生动作先验 |
| 2026-10 | 🆕 [PointWAM](https://arxiv.org/abs/2610.02840) `arXiv:2610.02840` |  | 用 3D 点表示世界与动作的世界-动作模型，用于灵巧操作 |
| 2026-10 | 🆕 [渐进视觉规划 WAM](https://arxiv.org/abs/2610.02508) `arXiv:2610.02508` | Meta | 用由粗到细的视觉规划代替稠密视频展开，改善长时程预测 |
| 2026-10 | 🆕 [ViGAR](https://arxiv.org/abs/2610.02368) `arXiv:2610.02368` | PKU | 把操作分解为视觉子目标规划器 + 子目标执行器，支持组合与上下文任务 |
| 2026-10 | 🆕 [UniWAM](https://arxiv.org/abs/2610.02054) `arXiv:2610.02054` | HKUST-GZ | 把 VLA 的语义推理与 WAM 的时空先验统一在一个模型里 |
| 2026-09 | 🆕 [InternW0-Δ](https://arxiv.org/abs/2609.31394) `arXiv:2609.31394` |  | 融合视频、语义、4D 几何先验的 WAM，用 2 万多小时异构数据（含 UMI、第一人称人类数据）训练 |
| 2026-09 | 🆕 [DeltaWAM](https://arxiv.org/abs/2609.28811) `arXiv:2609.28811` | PKU | 只预测变化量的世界-动作模型，用于双臂操作 |
| 2026-09 | 🆕 [OpenWAM](https://arxiv.org/abs/2609.07398) `arXiv:2609.07398` | OpenWAM | 开放、模块化地系统探索世界-动作模型预训练的设计空间 |
| 2026-09 | 🆕 [GE-Act 2.0](https://arxiv.org/abs/2609.05588) `arXiv:2609.05588` | AgiBot | 从零预训练并扩展的世界-动作模型 |
| 2026-08 | 🆕 [ZimaBlue](https://arxiv.org/abs/2609.00188) `arXiv:2609.00188` | JD | 用可扩展的第一人称视频预训练演化通用 WAM |
| 2026-08 | 🆕 [DreamX-Phi 1.0](https://arxiv.org/abs/2608.13489) `arXiv:2608.13489` | Alibaba AMAP | 动作条件视频世界模型，用于操作策略评测与数据生成 |
| 2026-07 | 🆕 [BadWAM](https://arxiv.org/abs/2607.15207) `arXiv:2607.15207` | NUS | 针对世界-动作模型的对抗攻击：画面预测正确、动作却出错 |
| 2026-07 | 🆕 [RynnWorld-4D](https://arxiv.org/abs/2607.06559) `arXiv:2607.06559` | Alibaba DAMO | 同步预测 RGB、深度与光流的 4D 具身世界模型 |
| 2026-07 | 🆕 [InternVLA-A1.5](https://arxiv.org/abs/2607.04988) `arXiv:2607.04988` | Shanghai AI Lab | 统一理解、潜在前瞻与动作，避免预测目标侵蚀 VLM 语义 |
| 2026-07 | 🆕 [GigaWorld-1](https://arxiv.org/abs/2607.02642) `arXiv:2607.02642` | GigaAI | 用世界模型做机器人策略评测的路线图 |
| 2026-06 | 🆕 [PhysisForcing](https://arxiv.org/abs/2606.28128) `arXiv:2606.28128` | NVIDIA | 用物理奖励强化操作视频世界模拟器的物理合理性 |
| 2026-06 | 🆕 [WAM 综述](https://arxiv.org/abs/2606.20781) `arXiv:2606.20781` | NUS | 世界-动作模型综述 |
| 2026-06 | 🆕 [ImageWAM](https://arxiv.org/abs/2606.19531) `arXiv:2606.19531` | SJTU | WAM 不一定需要视频生成：用图像编辑预测关键未来帧即可 |
| 2026-05 | 🆕 [何时信任想象](https://arxiv.org/abs/2605.06222) `arXiv:2605.06222` |  | 用"未来-现实验证"决定 WAM 预测动作执行多少再重规划 |
| 2026-03 | 🆕 [GigaWorld-Policy](https://arxiv.org/abs/2603.17240) `arXiv:2603.17240` | GigaAI | 以动作为中心的高效世界-动作模型 |
| 2026-02 | ★ 🆕 [DreamZero](https://arxiv.org/abs/2602.15922) `arXiv:2602.15922` | NVIDIA | 14B 世界-动作模型，联合预测视频与动作，泛化比 VLA 高 2 倍以上（GR00T N2 的基础） |
| 2026-02 | 🆕 [DreamDojo](https://arxiv.org/abs/2602.06949) `arXiv:2602.06949` | NVIDIA | 从 4.4 万小时第一人称人类视频学习的通用机器人世界模型，蒸馏后实时运行 |
| 2025-12 | 🆕 [Act2Goal](https://arxiv.org/abs/2512.23541) `arXiv:2512.23541` | AgiBot | 从世界模型到通用的目标图像条件策略 |
| 2025-12 | ★ 🆕 [Veo 世界模拟器评测策略](https://arxiv.org/abs/2512.10675) `arXiv:2512.10675` | Google DeepMind | 用 Veo 视频世界模型评测 Gemini Robotics 策略，与 1600 多次真机评测对照 |
| 2025-11 | 🆕 [PhysWorld](https://arxiv.org/abs/2511.07416) `arXiv:2511.07416` | Google DeepMind | 从视频生成 + 物理世界建模中学习机器人技能 |
| 2025-10 | 🆕 [Ctrl-World](https://arxiv.org/abs/2510.10125) `arXiv:2510.10125` | Stanford/Tsinghua | 可控的操作世界模型：不需真机即可给策略排名，想象中合成成功轨迹再微调提升 44.7% |
| 2025-08 | [Genie Envisioner](https://arxiv.org/abs/2508.05635) `arXiv:2508.05635` | AgiBot | 机器人操作的统一世界基础平台：视频世界模型 + 动作解码 + 神经模拟器 |
| 2025-06 | [V-JEPA 2](https://arxiv.org/abs/2506.09985) `arXiv:2506.09985` | Meta | 自监督视频世界模型；动作条件版 V-JEPA 2-AC 可零样本规划抓放 |
| 2025-05 | [DreamGen](https://arxiv.org/abs/2505.12705) `arXiv:2505.12705` | NVIDIA | 用视频世界模型生成"神经轨迹"作为训练数据，泛化到新动作与环境 |
| 2025-04 | [UWM](https://arxiv.org/abs/2504.02792) `arXiv:2504.02792` | UW | 耦合视频扩散与动作扩散，可在无动作视频上预训练 |
| 2025-02 | [UVA](https://arxiv.org/abs/2503.00200) `arXiv:2503.00200` | Stanford | 统一视频-动作模型：联合潜表示，推理时可解耦只出动作 |
| 2024-12 | [Video Prediction Policy](https://arxiv.org/abs/2412.14803) `arXiv:2412.14803` | Tsinghua | 用视频扩散模型的预测性视觉表征做通用策略 |
| 2023-10 | [SuSIE](https://arxiv.org/abs/2310.10639) `arXiv:2310.10639` | UC Berkeley | 用图像编辑扩散模型生成子目标，再由低层策略到达 |
| 2023-10 | [UniSim](https://arxiv.org/abs/2310.06114) `arXiv:2310.06114` | Google/UC Berkeley | 学习交互式真实世界模拟器，用于训练策略 |
| 2023-01 | [UniPi](https://arxiv.org/abs/2302.00111) `arXiv:2302.00111` | MIT/Google | 文本引导的视频生成 + 逆动力学，把视频当作通用策略 |
| 2018-12 | [Visual Foresight](https://arxiv.org/abs/1812.00568) `arXiv:1812.00568` | UC Berkeley | 基于视频预测的模型式视觉 RL，自监督数据下完成多种操作 |
| 2016-10 | [Deep Visual Foresight](https://arxiv.org/abs/1610.00696) `arXiv:1610.00696` | UC Berkeley | 学习动作条件的视频预测并用于规划推物体，视频世界模型用于操作的早期工作 |

### 触觉与力（10）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | 🆕 [N0-VTLA](https://arxiv.org/abs/2607.23782) `arXiv:2607.23782` | NeoteAI/Fudan | 用潜在触觉 token 扩展的视觉-触觉-语言-动作基础模型 |
| 2026-07 | 🆕 [N0-TWAM](https://arxiv.org/abs/2607.23783) `arXiv:2607.23783` | NeoteAI/Fudan | 首个大规模触觉原生世界-动作模型：同时预测未来视觉与接触 |
| 2026-06 | 🆕 [TORL-VLA](https://arxiv.org/abs/2606.09337) `arXiv:2606.09337` |  | 触觉引导的接触密集操作在线 RL |
| 2025-08 | [OmniVTLA](https://arxiv.org/abs/2508.08706) `arXiv:2508.08706` |  | 语义对齐触觉的视觉-触觉-语言-动作模型 |
| 2025-02 | [FACTR](https://arxiv.org/abs/2502.17432) `arXiv:2502.17432` | CMU | 力反馈遥操作 + 力感知课程训练的接触策略 |
| 2024-10 | [Sparsh](https://arxiv.org/abs/2410.24090) `arXiv:2410.24090` | Meta | 视触觉传感器的自监督通用表征 |
| 2024-10 | [3D-ViTac](https://arxiv.org/abs/2410.24091) `arXiv:2410.24091` | UIUC 等 | 触觉点与视觉点云融合的双手精细操作 |
| 2024-09 | [AnySkin](https://arxiv.org/abs/2409.08276) `arXiv:2409.08276` | NYU 等 | 即插即用、可更换的磁性触觉皮肤 |
| 2024-08 | [TacSL](https://arxiv.org/abs/2408.06506) `arXiv:2408.06506` | NVIDIA | 视触觉传感器仿真与学习库 |
| 2020-05 | [DIGIT](https://arxiv.org/abs/2005.14679) `arXiv:2005.14679` | Meta AI | 低成本、紧凑、高分辨率视触觉传感器 |

### 移动/双臂/人形（6）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-12 | 🆕 [WholeBodyVLA](https://arxiv.org/abs/2512.11047) `arXiv:2512.11047` |  | 从无动作标签的第一人称视频学统一潜在 VLA，做人形全身移动操作 |
| 2025-03 | [BEHAVIOR Robot Suite](https://arxiv.org/abs/2503.05652) `arXiv:2503.05652` | Stanford | 面向日常家务的全身操作：硬件、遥操作接口与策略 |
| 2024-10 | [BUMBLE](https://arxiv.org/abs/2410.06237) `arXiv:2410.06237` | UT Austin | VLM 驱动的楼宇级移动操作 |
| 2024-06 | [HumanPlus](https://arxiv.org/abs/2406.10454) `arXiv:2406.10454` | Stanford | 人形跟随人体实时影子遥操作，再模仿学习自主技能 |
| 2024-01 | [OK-Robot](https://arxiv.org/abs/2401.12202) `arXiv:2401.12202` | NYU | 用开放知识模型（检测、抓取、导航）拼成的零样本家庭抓放系统 |
| 2023-06 | [HomeRobot](https://arxiv.org/abs/2306.11565) `arXiv:2306.11565` | Meta | 开放词汇移动操作的仿真与真机基准 |

### 可形变与专题任务（3）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-06 | 🆕 [LabVLA](https://arxiv.org/abs/2606.13578) `arXiv:2606.13578` |  | 把 VLA 落地到科学实验室操作 |
| 2022-08 | [SpeedFolding](https://arxiv.org/abs/2208.10552) `arXiv:2208.10552` | UC Berkeley | 双臂快速叠衣 |
| 2021-05 | [FlingBot](https://arxiv.org/abs/2105.03655) `arXiv:2105.03655` | Columbia | 用动态甩动展开布料 |

### 基准与评测（22）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [LIBERO-Industry](https://arxiv.org/abs/2610.16782) `arXiv:2610.16782` |  | 工业流程约束（多品种小批量电子装配）下的操作基准 |
| 2026-10 | 🆕 [RobotWorld](https://arxiv.org/abs/2610.10409) `arXiv:2610.10409` |  | 84 个任务覆盖操作、移动操作等，评测多模态 agent 直接控制机器人 |
| 2026-09 | 🆕 [LIBERO-RECOVER](https://arxiv.org/abs/2609.05178) `arXiv:2609.05178` |  | 不只看成功率，还评测失败后的恢复能力 |
| 2026-07 | 🆕 [RoboDojo](https://arxiv.org/abs/2607.04434) `arXiv:2607.04434` |  | 仿真 + 真机统一基准，带公开排行榜 |
| 2026-05 | 🆕 [VLA-REPLICA](https://arxiv.org/abs/2605.20774) `arXiv:2605.20774` |  | 低成本、可复现的真机 VLA 基准（SO101 机械臂） |
| 2026-04 | 🆕 [VLA 安全综述](https://arxiv.org/abs/2604.23775) `arXiv:2604.23775` |  | VLA 的安全威胁、评测与防护机制综述 |
| 2026-03 | 🆕 [LIBERO-Para](https://arxiv.org/abs/2603.28301) `arXiv:2603.28301` |  | VLA 对指令改写的鲁棒性诊断 |
| 2026-03 | 🆕 [RoboMME](https://arxiv.org/abs/2603.04639) `arXiv:2603.04639` | Michigan | 通用策略的记忆能力基准 |
| 2025-12 | 🆕 [BEHAVIOR Challenge 2025 冠军](https://arxiv.org/abs/2512.06951) `arXiv:2512.06951` |  | 基于 π0.5 的任务适配，50 个家务任务 Q-score 0.26（NeurIPS 2025 挑战赛第一） |
| 2025-10 | 🆕 [LIBERO-Plus](https://arxiv.org/abs/2510.13626) `arXiv:2510.13626` | Fudan | 七个维度的受控扰动：相机视角、初始状态等微小变化让成功率从 95% 跌到 30% 以下 |
| 2025-10 | ★ 🆕 [LIBERO-PRO](https://arxiv.org/abs/2510.03827) `arXiv:2510.03827` |  | 扰动版 LIBERO：标准评测 90% 以上的模型，在泛化设置下跌到 0% |
| 2025-06 | ★ [RoboArena](https://arxiv.org/abs/2506.18123) `arXiv:2506.18123` | 多机构合作 | 分布式真机双盲成对评测（DROID 平台） |
| 2025-03 | [AutoEval](https://arxiv.org/abs/2503.24278) `arXiv:2503.24278` | UC Berkeley | 真机上自主评测通用策略，几乎不需要人工干预 |
| 2024-12 | [VLABench](https://arxiv.org/abs/2412.18194) `arXiv:2412.18194` | Fudan | 需要长时程推理的语言条件操作基准 |
| 2024-06 | [RoboCasa](https://arxiv.org/abs/2406.02523) `arXiv:2406.02523` | UT Austin/NVIDIA | 大规模家庭厨房仿真与任务 |
| 2024-05 | [SIMPLER](https://arxiv.org/abs/2405.05941) `arXiv:2405.05941` | UC San Diego 等 | 用仿真评测真机策略，排名与真机高度相关 |
| 2024-03 | [BEHAVIOR-1K](https://arxiv.org/abs/2403.09227) `arXiv:2403.09227` | Stanford | 1000 个日常活动的具身基准与仿真 |
| 2024-02 | [THE COLOSSEUM](https://arxiv.org/abs/2402.08191) `arXiv:2402.08191` | UW 等 | 14 种扰动因子评测操作策略泛化 |
| 2023-06 | ★ [LIBERO](https://arxiv.org/abs/2306.03310) `arXiv:2306.03310` | UT Austin | 终身学习操作基准，后来成为 VLA 最常用的仿真擂台 |
| 2021-12 | [CALVIN](https://arxiv.org/abs/2112.03227) `arXiv:2112.03227` | Freiburg | 语言条件的长时程操作基准（连续 5 条指令） |
| 2019-10 | [Meta-World](https://arxiv.org/abs/1910.10897) `arXiv:1910.10897` | Stanford/UC Berkeley | 50 个操作任务的多任务与元强化学习基准 |
| 2019-09 | [RLBench](https://arxiv.org/abs/1909.12271) `arXiv:1909.12271` | Imperial College | 100 个仿真操作任务的基准 |

### 综述/教程（5）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Towards Agentic Robots](https://arxiv.org/abs/2610.20442) `arXiv:2610.20442` |  | 模型、数据、环境与 harness 四个维度的 agentic 机器人综述 |
| 2026-06 | 🆕 [机器人需要的不止 VLA 与世界模型](https://arxiv.org/abs/2606.06556) `arXiv:2606.06556` |  | 立场论文：通用机器人智能的瓶颈不只是策略规模 |
| 2026-02 | 🆕 [LeRobot](https://arxiv.org/abs/2602.22818) `arXiv:2602.22818` | Hugging Face | 端到端机器人学习开源库：硬件接口、数据、训练、部署 |
| 2025-10 | ★ 🆕 [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403) `arXiv:2510.12403` | Hugging Face | LeRobot 团队的机器人学习教程：从 RL、模仿学习到 VLA，附代码 |
| 2024-05 | [VLA 综述](https://arxiv.org/abs/2405.14093) `arXiv:2405.14093` | CUHK | 面向具身 AI 的 VLA 模型综述 |

### 产业发布（18）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Pi API beta](https://www.pi.website/blog/api-beta) | Physical Intelligence | 上传机器人数据、在托管算力上微调 π 模型、云端推理接入自己的机器人 |
| 2026-10 | 🆕 [Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/) | Google DeepMind | 能从视频/草图生成 MuJoCo 孪生与控制代码，生成的仿真数据把真机成功率从 56% 提到 75% |
| 2026-09 | 🆕 [FLUX 3 Action](https://bfl.ai/models/flux-3-action) | Black Forest Labs | 7B 世界动作模型，用于机器人控制 |
| 2026-08 | 🆕 [Skild S1](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/) | Skild AI | 看一段任务视频即可学会最长 10 分钟的任务并实时执行 |
| 2026-08 | 🆕 [DYNA-2](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-2-world-action-model-demonstrating-first-true-scaling-law-in-robotics-powered-entirely-by-human-data-302847114.html) | Dyna Robotics | 用超过 100 万小时第一人称人类视频训练的 WAM，宣称出现人到机器人的扩展规律 |
| 2026-07 | 🆕 [Gemini Robotics 2 / ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/) | Google DeepMind | 实时视频理解、任务进度跟踪、流式低延迟编排、工具调用与多机器人协作 |
| 2026-07 | 🆕 [Claude plays robotics](https://www.anthropic.com/research/claude-plays-robotics) | Anthropic | 5 家厂商 12 个模型，跨扭矩直控、写控制器、训 RL、监督 VLA 四种接口：越高层越好用 |
| 2026-04 | 🆕 [Gemini Robotics-ER 1.6](https://deepmind.google/blog/gemini-robotics-er-1-6/) | Google DeepMind | 空间与多视角推理增强，新增仪表读数 |
| 2026-04 | 🆕 [AgiBot GO-2](https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/) | AgiBot | 统一推理与执行：先生成动作意图序列作宏观计划，异步双系统分离推理与控制 |
| 2026-04 | ★ 🆕 [GEN-1](https://www.therobotreport.com/generalist-introduces-gen-1-general-purpose-model-for-physical-ai/) | Generalist AI | 约 50 万小时数据从零训练；所选任务平均成功率 99%（GEN-0 为 64%），每项任务约 1 小时机器人数据 |
| 2026-03 | 🆕 [GR00T N1.7 / N2 预览](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) | NVIDIA | GTC 2026：N1.7 商业授权早期访问；N2 预览，基于 DreamZero 的世界-动作模型架构 |
| 2026-01 | 🆕 [Rho-alpha](https://www.therobotreport.com/microsoft-research-reveals-rho-alpha-vision-language-action-model-for-robots/) | Microsoft Research | 基于 Phi 系列的机器人模型：加入触觉与基于人类反馈的持续学习 |
| 2026-01 | 🆕 [1X World Model](https://www.1x.tech/ai) | 1X | 官方称其为 NEO 的"认知核心"：视频预测想象未来，再用逆动力学模型转成动作 |
| 2026-01 | 🆕 [Helix 02 (Figure)](https://interestingengineering.com/ai-robotics/figure-helix02-upgrades-humanoid-robot-control) | Figure | 新增 1 kHz 学习式全身控制器，一个网络控制行走、操作与平衡 |
| 2025-12 | 🆕 [GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) | NVIDIA | 3B 开源 VLA：Cosmos-Reason 骨干 + DiT 动作头，支持人形边走边操作 |
| 2025-11 | ★ 🆕 [GEN-0](https://eu.36kr.com/en/p/3551715251804547) | Generalist AI | 用 27 万小时真实操作数据训练，宣称在 7B 以上出现机器人扩展规律 |
| 2025-09 | [Gemini Robotics 1.5 发布](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) | Google DeepMind | ER 1.5 编排、VLA 执行，ER 1.5 通过 API 开放 |
| 2025-02 | [Helix (Figure)](https://www.figure.ai/news/helix) | Figure | System 2（7–9 Hz VLM）+ System 1（200 Hz 视觉运动策略）控制人形上半身 |

