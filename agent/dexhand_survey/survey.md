# 灵巧手算法调研：给运控研究者的导读（A 部分：纯灵巧手 · B 部分：Agent × 灵巧手）

> 截至 **2026-10-23**。读者设定为熟悉腿足/人形运动控制（RL sim2real、MPC、WBC、动作重定向）的研究者。
> 配套交互页面：`index.html`；收录 162 条论文与资源，其中 152 条 arXiv 编号和标题用官方 API 核验过。

---

## 0. 一页结论

1. **主线仍是"仿真 RL + sim2real"，而且方法和运控几乎一一对应。** 从 Dactyl（2018）到 DeXtreme、HORA，再到 DexNDM、ViserDex（2025–26），用的都是你熟悉的工具：域随机化、教师-学生特权蒸馏、RMA 式在线适应、执行器建模。2026 年的新变量有两个：
   - **触觉**：快速触觉仿真，或者干脆绕开触觉仿真；
   - **执行器层面的建模**：关节级神经动力学、电流到力矩的标定。
2. **"重定向 + RL 跟踪"成了和人形全身控制同构的配方。** ManipTrans、DexTrack、DexMachina、SPIDER、REGRIND 把人手-物体演示重定向到机器人手，再训练以物体为中心的跟踪策略。REGRIND 明确说自己就是把人形 WBC 的简单配方搬到了灵巧手上。
3. **数据规模在快速上涨，但"运动学对、力不对"是核心瓶颈。**
   - 规模：EgoScale 用了 2 万多小时人类第一人称视频，Dex1B 有 10 亿条演示，DexGraspNet 3.0 有 1.7 亿个抓取；
   - 2026 年的重定向开始保留接触结构与力：TopoRetarget、ReForce、Contact-Anchored Retargeting。
4. **基于模型的方法没有退场。** 采样 MPC（DROP、Primitive-Informed MPC、Global-MPPI）和接触感知优化（Contact Trust Region）不用训练、可解释，正在和 RL 分层结合：高层 RL 决定"碰哪里"，低层 MPC 决定"怎么推"。
5. **VLA 开始覆盖高自由度手**（GR-Dexter、XL-VLA、Dexora、EgoScale），但手指级的接触协调仍主要由 RL 训出的低层控制器或"基础控制器"承担（DexterityGen、TacBPM、ADEPT）。
6. **Agent（LLM/VLM/编码 agent）在灵巧手上的价值集中在慢环和离线环**：写奖励（Eureka 让仿真 Shadow 手转笔、Text2Touch）、生成仿真与数据（GenDexHand、DexAgent 3.5×、Gemini 4 Argon）、给出语义/功能抓取目标（OmniDexVLG、DextER）、编码 agent 自己搭环境训策略（Dexterous Astra 转笔、ASPIRE 真机魔方）。
   - **让大模型直接控手指仍然不行**：Galbot 的实测显示 GPT-6 Astra 直接做手内操作时难以协调手指接触；
   - **把触觉作为上下文交给大模型却很有用**：冻结的 GPT-6 Astra 在插孔任务上从 31% 提到 99%。

---

## 1. 运控视角：一张翻译表

| 运控（腿足/人形）里熟悉的东西 | 灵巧手里的对应物 | 代表工作 |
|---|---|---|
| 地面接触、步态、支撑相与摆动相 | 物体接触与"指步态"（finger gaiting）：一部分手指稳住物体，另一部分换位 | HORA、FINGR、TacBPM |
| 大规模并行 PPO + 域随机化 | 同样是主力；ADR 最早就出现在灵巧手上（OpenAI 魔方） | Dactyl、Rubik's Cube、DeXtreme、DexPBT |
| RMA / 教师-学生特权蒸馏 | 物体质量、摩擦、形状作特权信息，学生只用本体、触觉或视觉 | HORA、RotateIt、PTLD、ViserDex |
| 执行器网络、系统辨识 | 手的执行器（腱、减速器、电流-力矩）更难建模；关节级神经动力学、电流-力矩标定 | DexNDM、Force-Based Sim2Real、REGRIND |
| 动作重定向 + 运动跟踪 | 人手→机器人手重定向 + 以物体为中心的跟踪 RL | ManipTrans、DexTrack、DexMachina、SPIDER、REGRIND |
| 运动先验 / 潜在动作空间 | 灵巧"基础控制器"、行为先验、预训练后再后训练 | DexterityGen、TacBPM、ADEPT、CoorDex |
| 采样 MPC（MPPI、DIAL-MPC、MuJoCo MPC） | 接触丰富的手内 MPC，要处理接触的非光滑与模式组合 | DROP、Primitive-Informed MPC、Global-MPPI、Contact Trust Region |
| 地形感知 | 物体形状与位姿感知（深度、RGB、触觉），且手-物遮挡严重 | Visual Dexterity、ViserDex、DextrAH-RGB |
| 抗外扰 | 防滑、抓握稳定、力调节 | Force-Based Sim2Real、WM-Craftnet |
| 全身控制（WBC） | 手指 + 手腕 + 手臂 + 身体协调；人形边走边操作 | CoorDex、DexWeave、DextrAH-G |

**四个本质差异**（也是灵巧手比腿足"难"的地方）：

1. **被控对象不受驱动。** 你要控制的是物体的状态，但物体没有执行器，只能通过间歇接触施力。这是一个欠驱动的混合系统，接触模式切换远比步态频繁、组合更多。
2. **接触多且小。** 力、尺寸、速度都小，摩擦、滑移、柔顺性的影响更大，仿真接触参数的误差相对更致命。
3. **感知被遮挡。** 物体常被手挡住，视觉位姿估计难；触觉因此关键，但触觉仿真又很难。
4. **硬件不统一。** 没有一个像"Unitree G1 之于人形"那样的统一平台：驱动有腱、连杆、直驱之分，自由度从 6 到 22 不等。所以跨本体表示成了独立的研究方向（OHRA、UHAS、XL-VLA）。

泛化的对象也不同：腿足主要泛化地形，灵巧手主要泛化**物体**（形状、质量、摩擦、铰接结构）。

---

# A 部分：不沾 Agent 的灵巧手

## A1. 硬件

| 驱动方式 | 代表 | 对控制的影响 |
|---|---|---|
| 关节直驱 / 近端电机 | Allegro（16 DoF）、LEAP（16 DoF）、XHand（12 DoF）、Sharpa Wave（22 DoF，500 Hz）、Wuji Hand（20 DoF，FOC） | 建模相对直接，可用电流估力；关节多时手更重，散热需要处理 |
| 腱驱动（电机在前臂） | Shadow Hand、ORCA（17 DoF，开源）、Tesla Optimus Gen 3（报道为 22 DoF）、MM-Hand（21 DoF） | 手轻、自由度高；但腱的摩擦、迟滞、耦合让 sim2real 更难 |
| 连杆驱动 | Inspire RH56 系列（中国人形常见标配）、DexLink（16 个执行器驱动 20 个关节，<400 美元） | 结构可靠、成本低；耦合关节限制了可达姿态 |
| 软体 / 欠驱动 | Tactile SoftHand-A、CYJ Hand-0（SMA + 电机混合） | 被动顺应好；可控自由度少，建模难 |

**选型时运控研究者应关注**：是否反驱（backdrivable）、控制带宽、能否从电流估计力矩、触觉布局、能否自己维修，以及**有没有可靠的 MuJoCo/URDF 资产**——这直接决定 sim2real 的起点。2026 年的趋势是 20 DoF 以上 + 指尖触觉阵列，以及与人形整合（GR-Dexter 手、Sharpa Wave 进入 NVIDIA 参考人形套件、TESOLLO DG-5F-S 缩小减重）。

## A2. 手内操作：RL 与 sim2real 主线

**里程碑**：
- **Dactyl** 与 **魔方**（OpenAI）：大规模域随机化、ADR、带记忆的策略；
- **General In-Hand Reorientation / Visual Dexterity**（MIT）：教师-学生，单深度相机；
- **HORA**：把腿足的 RMA 搬到手内旋转；
- **RotateIt / Rotating without Seeing / AnyRotate**：触觉；
- **DeXtreme**（Isaac Gym 并行 + 视觉位姿）、**DexPBT**（种群训练）；
- 转笔经验总结、**Twisting Lids Off**（双手拧盖）。

**2025–2026 的新进展，按运控视角归类**：

- **执行器与动力学差距**：
  - **DexNDM**（ICLR'26）用少量真机数据拟合每个关节的等效动力学（把耦合、驱动和物体影响都吸收进低维有效项），再修正仿真策略的动作；
  - **Force-Based Sim2Real** 做电流→力矩标定和执行器动力学建模，省掉力矩传感器；
  - **REGRIND** 靠"仔细的系统辨识"实现零样本上真机。

  这些和腿足的 actuator net、system ID 是同一类工作。
- **视觉 sim2real**：**ViserDex**（RSS'26，ETH + NVIDIA）在 3D 高斯泼溅空间里做域随机化，单目 RGB 估计物体位姿，策略用课程 RL + 教师-学生。
- **先验与预训练**：
  - **ADEPT**：先预训练通用的"物体重摆"，再稳定地后训练到长时程任务；
  - **TacBPM**：触觉条件的行为先验，下游用残差潜动作复用；
  - **DexterityGen**：RL 原语 + 扩散生成，相当于"灵巧基础控制器"。

  这对应腿足里的运动先验（AMP/PULSE 一类）。
- **任务谱扩展**：
  - **POISE**：手内 6D 位姿到达；
  - **Mana / SimToolReal / Articulated Tools In-Hand**：铰接工具与工具使用；
  - **FINGR**：单手还原真实魔方；
  - **DexCompose**：按手指分配"动作所有权"来组合策略。
- **跨本体**：
  - **UHAS**：球面形变动作空间 + 级联 IK；
  - **OHRA**：统一的 22 DoF 规范 URDF。

## A3. 抓取合成

- **优化路线**：可微力封闭（**DexGraspNet**，132 万个抓取）→ 双层优化 QP（**BODex**，单卡每秒 49 个以上，训练出的模型仿真成功率约 80%，DexGraspNet 约 40%）。
- **生成模型**：UniDexGrasp / ++、DexGraspNet 2.0（杂乱场景）、**DexGrasp Anything**（物理约束进扩散，340 万个抓取）、**Dex1B**（10 亿条演示）。
- **闭环抓取策略的 sim2real**：DextrAH-G（几何 fabrics 保证安全）、DextrAH-RGB（只用 RGB）、Get a Grip。
- **2026 年的新方向**：
  - 跨手迁移：OHRA，3 指 LEAP 零样本 81.9%；
  - 人类偏好先验：HUGS；
  - 真实世界无人值守采集：AutoDex；
  - 用抓取数据预训练灵巧技能：From Grasps to Dexterity；
  - 只靠触觉的盲抓：Blind Dexterous Grasping。

## A4. 触觉

- **传感器**：视触觉（DIGIT）、磁性皮肤（AnySkin）、电阻/电容阵列（Sharpa Wave 等商用手内置）。
- **触觉仿真**：HydroShear（水弹性接触，含粘滑与剪切）、TacMap（穿透深度图统一仿真与真实）、ETac（数据驱动形变传播）、Tactile Genesis（多种触觉抽象统一接口，2 万并行环境）。
- **绕开触觉仿真**：
  - **PTLD**：用真实世界的特权传感器采集触觉策略数据再蒸馏；
  - **SBLR**：仿真里用"预言者传感器"（点云 + 指尖力）训练，部署时对齐真实传感器的潜变量。
- **Real2Sim 标定**：Blind Dexterous Grasping 让仿真里的触觉事件时序与真机对齐。
- **触觉进入世界模型**：DexTacWAM 把触觉潜变量注入视频扩散世界模型，6 个任务平均 70.6 分，最强基线 38.0。
- **运控类比**：触觉相当于"高维的足底接触传感器"，但仿真难度高一个量级。对你来说最省事的做法往往是 PTLD / SBLR 这种不依赖精确触觉仿真的路线。

## A5. 基于模型的接触控制

- **准静态平滑 + 全局规划**：Pang 等对接触动力学做局部平滑，再用 RRT 做全局规划；**Contact Trust Region**（IJRR）提出尊重接触单边性的信赖域，局部 MPC + 全局拼接，不靠并行也比 RL 和采样 MPC 省算力。
- **采样 MPC**：Predictive Sampling（MuJoCo MPC）、**DROP**（在线采样规划完成手内方块重定向）、腱驱动手上的采样 MPC、**Primitive-Informed MPC**（低维操作原语偏置采样 + 关节残差）、**Global-MPPI**（核平方和全局探索 + 渐进非凸平滑）。
- **与学习结合**：
  - Hierarchical RL-MPC：高层 RL 选接触点和物体子目标，低层接触隐式 MPC 执行；
  - "Should We Learn from Sampling-Based Planners?" 讨论了用规划器轨迹做模仿学习的利弊。
- **工具**：MuJoCo Playground（MJX，含灵巧手任务）、ComFree-Sim（免互补约束的解析接触 GPU 引擎）。
- **给运控研究者**：这是你最熟悉的工具链，和 DIAL-MPC、MPPI 同源。难点在于接触模式组合爆炸和梯度不可靠，原语化采样与接触感知信赖域是两条可行的缓解路线。

## A6. 遥操作与重定向

- **视觉遥操作**：DexPilot（经典重定向优化）、**AnyTeleop**（通用框架）、Bunny-VisionPro（Vision Pro，带触觉反馈）、OPEN TEACH（VR）。
- **穿戴式**：DexCap（动捕手套）、**DexUMI**（外骨骼 + 视频里把人手修补成机器人手，平均 86%）、DexEXO（适配 140–217 mm 手长）、DexWild（野外低成本采集）。
- **重定向本身**：
  - 目标项分析：Retargeting Objectives；
  - 免标定：AnyDexRT；
  - 拧转动作：DexTwist；
  - **力感知重定向**：ReForce，在运动学结果上预测残差以达到期望接触力。
- **共享自主**：
  - 人类遥操作没有触觉反馈，很难直接操控精细接触，于是由学到的低层控制器来"补"接触执行：DexterityGen、TeleDexter、DexTeleop-0；
  - 另一种分工是人用 VR 管手臂、VLA 管手指：Shared-Autonomy DexGrasp-VLA。

  这和人形遥操作中用 RL 跟踪器兜底是同一思路。

## A7. 从人类演示 / 视频到机器人

- **重定向 + 跟踪 RL**（与人形运动模仿同构）：
  - **DexTrack**：通用神经跟踪控制器 + 数据飞轮；
  - **ManipTrans**：通用模仿器 + 交互残差；
  - **DexMachina**：强度衰减的虚拟物体控制器做课程；
  - **SPIDER**：物理感知采样重定向，覆盖 9 种本体；
  - **Human2Sim2Robot**：一段 RGB-D 视频 + 以物体为中心的奖励；
  - **REGRIND**：一段示范 + 残差 RL + 系统辨识。
- **保接触而非保关节**：TopoRetarget（交互图 + 拉普拉斯形变）、Contact-Anchored Retargeting（迁移"哪个手指部位碰哪里、按什么顺序"）。
- **视频规模化**：DexMV、VideoDex → Do as I Do、VideoManip、DexImit、DexMan、Dex-X（仿真补全触觉）、Gen2Dex（生成视频）。
- **第一人称大规模**：
  - **EgoScale**：2 万多小时人类视频，发现数据规模与验证损失呈对数线性关系，且验证损失与真机表现强相关；
  - Being-H0、EgoVLA。

## A8. 灵巧 VLA 与基础模型

- **抓取型**：DexGraspVLA（VLM 规划 + 扩散控制器，杂乱场景 90% 以上）、DexVLG（1.7 亿个语义部件抓取）。
- **通用型**：
  - **GR-Dexter**（字节，21 DoF 手 + 双手遥操作 + 跨本体数据）；
  - **XL-VLA**（共享潜在动作空间，4 种手平均 72%，π0 为 32%）；
  - **Dexora**（开源双臂双手 VLA）；
  - EgoScale、Being-H0、EgoVLA（人类视频预训练）；
  - **DexTacWAM**（视触觉世界-动作模型）。
- **产业**：Figure Helix 02（全身单网络，借助指尖触觉与掌心相机分药片、推注射器）、NVIDIA GR00T 系列、Gemini Robotics（微调后折纸）。
- **难点**：动作维度高（双手 40 维以上）、遮挡、真机数据贵。对应的三个解法是跨本体潜在动作、人类视频预训练、共享自主采数据。

## A9. 人形全身灵巧

- **CoorDex**：身体与手分别训特权跟踪教师，蒸馏成潜在先验，再用协调残差 RL 实现"边走边操作"（G1 + 20 DoF Wuji 手）。
- **DexWeave**：交互一致的身体 + 手重定向 + 全身策略。
- **HumDex**：IMU 全身动捕 + 学习式手部重定向。
- **WholeBodyVLA**：从无标签视频学潜在 VLA。
- **Handroid**：同一个机身可重构为灵巧手或桌面人形。

这一块是运控研究者最容易切入的交叉点：把已有的人形 WBC 先验和灵巧手先验组合起来。

## A10. 基准与数据集

- **仿真基准**：Bi-DexHands、DexArt、DexMachina 基准、**DexVerse**（100 个任务、6 种手）、MuJoCo Playground、BODex 的 MuJoCo 基准、DexCraft（铰接工具）。
- **人手数据集**：DexYCB、ARCTIC、OakInk2。
- **合成抓取**：DexGraspNet 系列、Dex1B。

---

# B 部分：Agent × 灵巧手

## B0. 先说结论：agent 在灵巧手上能做什么

| 环节 | agent 的作用 | 成熟度 | 代表 |
|---|---|---|---|
| 奖励与课程 | 写奖励函数代码、迭代改进 | 较成熟，已上真机 | Eureka、Text2Touch、RF-Agent |
| 仿真参数 / 域随机化 | VLM 当视觉真实感评判来调参 | 起步 | DexSim2Real |
| 仿真环境与数据生成 | 生成任务和场景，从人类视频重建并生成训练数据 | 快速发展 | GenDexHand、DexAgent、EmbodiedSWE-Gen、Gemini 4 Argon |
| 语义 / 功能抓取目标 | 理解"拿杯柄""按喷头"，给出接触与抓取类型 | 较成熟 | OmniDexVLG、DextER、BLaDA、AffordDexGrasp、DexVLG |
| 高层规划 + 灵巧技能 | VLM 规划，调用抓取或手内技能，失败后恢复 | 较成熟 | DexGraspVLA、RoboDexVLM、Being-0、AW-0 |
| 编码 agent 自动训策略 | 自己搭仿真、写奖励、训 PPO、调参 | 2026 年刚出现 | Dexterous Astra、ASPIRE、SEAR |
| 直接控制手指 | 大模型逐步输出手指动作 | 弱（触觉上下文例外） | Galbot 评测、GPT-6-Astra Touch、TacTalent |

**一句话**：agent 擅长自动化"奖励-环境-数据-调参-评估"这个慢循环，不擅长替代 30–500 Hz 的手指接触控制。

## B1. LLM 写奖励 / 调仿真参数

- **Eureka**（2023）：GPT-4 写奖励代码并进化迭代，第一次让仿真里的 Shadow 手学会快速转笔。这是"LLM × 灵巧手"最常被引用的例子。
- **Text2Touch**（CoRL'25）：把 LLM 奖励设计扩展到触觉。提示结构可处理 70 多个环境变量，多轴手内旋转在带视触觉的四指手上迁移到真机；奖励更短，表现却超过人工精调的版本。
- **RF-Agent**（2026）：把奖励设计看成序列决策，用蒙特卡洛树搜索管理迭代，在 Bi-DexHands 等 17 个任务上测试。
- **DexSim2Real**（2026）：VLM 当视觉真实感评判，闭环 CMA-ES 优化域随机化参数，真机平均成功率 78.2%。

## B2. 生成式仿真与 agentic 数据管线

- **GenDexHand**：针对灵巧手的生成式仿真。VLM 反馈闭环调整物体位置和尺度，任务拆成子任务顺序 RL。
- **DexAgent**（Stanford/Columbia，2026-09）：一段第一人称视频 + 任务描述，依次完成语义理解、按物体属性重建仿真、轨迹优化、生成数据。agent 在每个阶段从工具库选工具或自造新工具，由属性相关的校验器把关；11 个真机任务上成功率是基线的 3.5 倍，工具库越用越快。
- **EmbodiedSWE**：编码 agent 解长达半小时的灵巧任务，再把一个解扩增成大量 VLA 训练轨迹。
- **Gemini 4 Argon for Robotics**（2026-10）：把第一人称人类视频重定向到带 22 DoF 手的仿真人形（500 Hz），并能从视频或草图生成 MuJoCo 孪生。

## B3. VLM / LLM 语义与功能抓取

- Multi-GraspLLM、AffordDexGrasp（可泛化的指导性可供性）、**OmniDexVLG**（多 agent 协作 + 检索增强 + 思维链推理抓取类型、接触语义和功能可供性）、**DextER**（先生成"哪个手指部位碰哪里"的接触 token，再出抓取）、BLaDA（3D 高斯场中定位功能部位）、双手可供性 VLM、AdaRoboVLG（可组合的基础模型先验 + 通用抓取策略）、DexVLG。
- **对运控研究者**：这一层给出"目标接触与抓取类型"，相当于给你的控制器一个语义化的参考。下游仍需力封闭检查和闭环控制。

## B4. VLM 规划 + 灵巧技能（分层）

- **DexGraspVLA**：VLM 规划 + 扩散低层控制器，把多样的语言和视觉输入转成领域不变表示。
- **RoboDexVLM**：VLM 任务规划，带任务级恢复。
- **Being-0**：人形 agent。基础模型做认知，Connector VLM 衔接行走与灵巧技能。
- **DexHandDiff**：用 LLM 生成的交互引导扩散规划。
- **AW-0**（2026-10）：代码、未来视频、动作三路联合生成，在 YAM 臂与灵巧手上实测。

## B5. 编码 agent 与自我进化

- **ASPIRE**：编码 agent 持续写、修控制程序并沉淀技能库；在真机灵巧手上解魔方，效率随经验提升。
- **SEAR**：三种 agent 模式（代码即策略、agent 即策略、自动研究），7 个前沿模型、330 个仿真 + 5 个真机任务，并设计了对照实验，区分"真学到了"还是"只是多试了几次"。
- **Embodied Playground**：含需要推断状态并逐层转动的魔方、灵巧手版汉诺塔。
- **Dexterous Astra**（2026-09，社区）：GPT-6 Astra 自己搭 Isaac Lab / MuJoCo 环境、写解法、训 PPO。Sharpa Wave 手转笔三圈，Wuji 手解出魔方三步；提示词与检查点公开。同一社区清单还记录了 Astra 自主跑约一天半、从建笔的网格到训好 PPO 转笔策略的案例。
- **对运控研究者**：这条线本质上是 Eureka 的放大版。编码 agent 接管"写环境 → 写奖励 → 训练 → 看曲线 → 改"的循环，你负责可自动评估的任务定义和安全的真机流程。

## B6. 前沿模型直接控手

- **Galbot 的 GPT-6 Astra 系统评测**：与学习策略混合控制时，DexJoCo 10 次试验成功 50%；直接控制手内操作时难以协调手指接触。
- **GPT-6-Astra Understands Touch**：把专家示范总结成任务指导与触觉参考，冻结的 GPT-6 Astra 在 5 个 UniVTAC 任务上平均 93.8%，触觉让插孔任务从 31% 提到 99%。
- **TacTalent**：19 个仿真视触觉任务，加触觉后平均成功率上升。
- **SenseNova 报告**：在 12 个基准（含灵巧手）上，具身能力大体具备，实用性不足。
- **社区演示**：仿真中零样本操控手指解魔方、抓苹果梗等，见 Awesome-Astra-Embodied-AI。
- **解读**：大模型的强项是"理解接触信号、做纠正决策"，弱项是"高频协调多指"。这和第一份调研的结论一致：大模型留在慢环。

---

## 2. 时间线（精选）

| 时间 | 事件 |
|---|---|
| 2018-08 | Dactyl：大规模域随机化 + RL，Shadow 手方块重定向 sim2real |
| 2019-10 | OpenAI 单手魔方（ADR）；DexPilot 视觉遥操作 |
| 2021-11 | 通用手内重定向（教师-学生） |
| 2022-06 | Bi-DexHands；准静态接触平滑的全局规划 |
| 2022-10 | DexGraspNet；HORA（RMA 进入灵巧手）；DeXtreme |
| 2022-11/12 | Visual Dexterity；VideoDex；Predictive Sampling |
| 2023-03 | Rotating without Seeing；UniDexGrasp |
| 2023-05 | DexPBT；DexArt |
| 2023-07 | AnyTeleop |
| 2023-09 | LEAP Hand；RotateIt |
| 2023-10 | **Eureka：LLM 写奖励让仿真 Shadow 手转笔**（B） |
| 2024-03 | DexCap；Twisting Lids Off；OPEN TEACH |
| 2024-05/07 | AnyRotate；DextrAH-G；Bunny-VisionPro；转笔经验总结 |
| 2024-09/10 | DROP；AnySkin；DexGraspNet 2.0；DexMimicGen；Get a Grip |
| 2024-11/12 | DextrAH-RGB；腱驱动手采样 MPC；BODex；Multi-GraspLLM（B） |
| 2025-02 | DexterityGen；DexTrack；DexGraspVLA；MuJoCo Playground |
| 2025-03 | ManipTrans；DexGrasp Anything；GR00T N1；Gemini Robotics；AffordDexGrasp / RoboDexVLM / Being-0（B） |
| 2025-04/05 | Human2Sim2Robot；ORCA；Contact Trust Region；DexWild；DexUMI；DexMachina |
| 2025-06/07 | Dex1B；DexVLG（DexGraspNet 3.0）；Being-H0；EgoVLA |
| 2025-09/10 | Text2Touch（B）；DexNDM |
| 2025-11/12 | SPIDER；GenDexHand（B）；GR-Dexter；OmniDexVLG（B） |
| 2026-01 | Force-Based Sim2Real；Hierarchical RL-MPC；DextER（B）；Figure Helix 02 |
| 2026-02 | EgoScale；OHRA；SimToolReal；HydroShear；TacMap；RF-Agent（B） |
| 2026-03/04 | PTLD；XL-VLA；HumDex；ViserDex；ETac；BLaDA（B） |
| 2026-05/06 | Dexora；DexSim2Real（B）；灵巧手综述；CoorDex；Tactile Genesis；Mana；AutoDex；Sharpa Wave 规格公布（CNX 报道） |
| 2026-07/08 | DexVerse；REGRIND；TeleDexter；HUGS；ASPIRE（B）；ADEPT；ReForce；SBLR |
| 2026-09 | FINGR；DexTacWAM；TacBPM；POISE；DexWeave；Primitive-Informed MPC；DexAgent / EmbodiedSWE / Dexterous Astra / Galbot 评测（B） |
| 2026-10 | Gen2Dex；SEAR / AW-0 / GPT-6-Astra Touch / TacTalent / Gemini 4 Argon（B） |

---

## 3. 开放问题

1. **力与接触的数据从哪来。** 人类视频和遥操作都缺力和触觉；仿真补全（Dex-X）、力感知重定向（ReForce）、真实特权传感器（PTLD）是三条路，还没有收敛。
2. **执行器与接触参数的 sim2real。** 腱驱动、减速器回差、指尖材料让"关节级建模"成为必需，需要像腿足 actuator net 那样的标准流程。
3. **触觉的标准化。** 每种传感器都等于一台"新机器人"；SBLR、Tactile Genesis 尝试统一抽象。
4. **跨本体。** 6–22 DoF、不同驱动方式的手能否共享策略；规范 URDF、潜在动作空间是候选答案。
5. **长时程与工具使用。** 魔方、铰接工具、拧螺丝需要"保持一部分接触、同时改变另一部分接触"，策略组合（DexCompose）和接触规划仍是难点。
6. **评测。** 真机任务难复现；DexVerse、DexMachina 基准在推进，但缺乏像 RoboArena 那样的分布式真机评测。
7. **Agent 的边界。** 大模型能否在 10–100 Hz 的环里做触觉纠正？目前只在慢速、离散决策的任务上有效（插孔、纠正）。更现实的方向是让编码 agent 自动化训练与评估循环。

---

## 4. 给运控研究者的上手路线

1. **仿真起步**：MuJoCo Playground（`pip install playground`，含灵巧手任务，单 GPU 几分钟训完），或 Isaac Lab 自带的手内重定向任务（ViserDex 的仓库就在它上面扩展）。先复现"方块重定向 + 教师-学生"。
2. **把你的 sim2real 工具箱搬过来**：执行器建模（参考 DexNDM、Force-Based Sim2Real）、RMA 式适应（HORA）、视觉位姿（ViserDex）。
3. **硬件**：入门选 LEAP Hand（低成本）或 ORCA（开源）；要做力控和触觉，看直驱 + 触觉的商用手（Sharpa Wave、Wuji 等），先确认 MuJoCo/URDF 资产和电流-力矩接口。
4. **数据与模仿**：AnyTeleop 或 Vision Pro 遥操作；"重定向 + 跟踪 RL"可从 DexMachina 或 ManipTrans 的配方入手，和你做人形运动模仿的流程几乎一样。
5. **加入 agent**：让编码 agent（Claude Code / Codex）照 Eureka、Text2Touch 的方式写奖励，或照 GenDexHand、DexAgent 的方式生成环境与数据；低层控制仍由你训练的策略或 MPC 负责。

---

## 5. 推荐阅读路径

1. Dactyl → HORA（RMA 迁移）→ DeXtreme → DexNDM（执行器建模）
2. ManipTrans / DexMachina / REGRIND（重定向 + 跟踪）
3. DROP / Contact Trust Region / Hierarchical RL-MPC（MPC 视角）
4. DexterityGen（基础控制器）→ CoorDex（人形全身）
5. Force-Based Sim2Real / PTLD（触觉）
6. EgoScale / XL-VLA / GR-Dexter（数据与 VLA）
7. B 部分：Eureka → Text2Touch → DexAgent → GPT-6-Astra Understands Touch
8. 综述：*Towards Robotic Dexterous Hand Intelligence: A Survey*（2605.13925）

---

## 6. 方法与可信度

- 检索于 2026-10-23。arXiv 条目用 export.arxiv.org API 按编号核验标题与首发日期；"一句话"是根据摘要写的中文概括。
- 硬件参数来自官方页面（Wuji）或媒体报道（Sharpa Wave，据 CNX）；Tesla Optimus 手部参数是二手信息。
- 不同论文的任务、硬件和成功率口径不同，数字不可横向比较；2026 年的大量工作仍是未经同行评审的预印本。

## 主要网页来源

- 硬件：[Sharpa Wave（CNX Software）](https://www.cnx-software.com/2026/06/02/sharpa-wave-high-end-dexterous-robotic-hand-with-22-dof-high-sensitivity-dynamic-tactile-array/) · [Wuji Hand](https://wuji.tech/en/hand) · [腱驱动人形手（Humanoid.guide）](https://humanoid.guide/tendon-driven-humanoid-hands/) · [TESOLLO @ ICRA 2026](https://www.axis3d.com/en/post/tesollo-showcases-vla-and-dexterous-manipulation-at-icra-2026-from-teleoperation-to-humanoid-roboti)
- Agent × 灵巧手：[Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/) · [Dexterous Astra](https://github.com/jianglongye/dexterous-astra) · [TacTalent](https://tactalent.github.io/) · [Awesome-Astra-Embodied-AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI)
- 资源清单：[Awesome-Dexterous-Hands](https://github.com/CyanHaze/Awesome-Dexterous-Hands)
- 产业：[Figure Helix 02](https://interestingengineering.com/ai-robotics/figure-helix02-upgrades-humanoid-robot-control)

---

## 附录 A：论文与资源总表（162 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 校验；★ 为入门推荐。

### A·硬件（12）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [Tesla Optimus Gen 3 手](https://humanoid.guide/tendon-driven-humanoid-hands/) | Tesla | 报道称手部 22 DoF + 腕部 3 DoF，执行器移到前臂、腱传动（工程师帖子与专利解读，二手信息） |
| 2026-07 | [Handroid](https://arxiv.org/abs/2607.16187) `arXiv:2607.16187` |  | 同一个 27 DoF 机电身体可重构为 20 DoF 灵巧手或桌面人形 |
| 2026-06 | [DexLink Hand](https://arxiv.org/abs/2606.17418) `arXiv:2606.17418` |  | 连杆驱动：16 个执行器驱动 20 个关节，全部集成在人手大小的结构里，320 g，成本不到 400 美元 |
| 2026-06 | [TESOLLO DG-5F-S](https://www.axis3d.com/en/post/tesollo-showcases-vla-and-dexterous-manipulation-at-icra-2026-from-teleoperation-to-humanoid-roboti) | TESOLLO | 五指 20 DoF，比上一代更小更轻，便于装上人形；ICRA 2026 展示遥操作与 VLA 驱动的灵巧操作 |
| 2026-06 | [Sharpa Wave](https://www.cnx-software.com/2026/06/02/sharpa-wave-high-end-dexterous-robotic-hand-with-22-dof-high-sensitivity-dynamic-tactile-array/) | Sharpa | 22 DoF 直驱，每指有动态触觉阵列，500 Hz 控制，1.3 kg；被 NVIDIA GR00T 参考人形套件选用（据 CNX 报道） |
| 2026-05 | [Wuji Hand](https://wuji.tech/en/hand) | Wuji Technology | 20 个主动自由度（每指 4 个、关节不耦合，各有独立电机），裸手不到 600 g，指尖力 15 N，无刷电机 FOC 控制 |
| 2026-04 | [MM-Hand](https://arxiv.org/abs/2604.17245) `arXiv:2604.17245` |  | 21 DoF 远端腱驱动模块化手：电机移出手掌，集成关节角、触觉、电机侧反馈与掌内双目 |
| 2025-12 | ★ [GR-Dexter](https://arxiv.org/abs/2512.24210) `arXiv:2512.24210` | ByteDance Seed | 硬件、模型、数据一体：紧凑 21 DoF 手 + 双手遥操作 + VLA 训练配方（遥操作数据 + 视觉语言数据 + 跨本体数据） |
| 2025-07 | [CYJ Hand-0](https://arxiv.org/abs/2507.14538) `arXiv:2507.14538` |  | 21 DoF，形状记忆合金 + 电机混合腱驱动的仿生手 |
| 2025-04 | [ORCA Hand](https://arxiv.org/abs/2504.04259) `arXiv:2504.04259` | ETH Zurich | 开源 17 DoF 腱驱动手，集成触觉，8 小时内可装好，物料成本不到 2000 瑞士法郎 |
| 2024-06 | [Tactile SoftHand-A](https://arxiv.org/abs/2406.12731) `arXiv:2406.12731` |  | 3D 打印、高度欠驱动、带触觉的拟人软手（拮抗腱机构） |
| 2023-09 | ★ [LEAP Hand](https://arxiv.org/abs/2309.06440) `arXiv:2309.06440` | CMU | 低成本、易组装维修的 16 DoF 拟人手，专为机器人学习设计，是常用的 sim2real 平台 |

### A·触觉（13）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [DexTacWAM](https://arxiv.org/abs/2609.24976) `arXiv:2609.24976` |  | 视触觉世界-动作模型：把触觉潜变量注入视频扩散世界模型；22 DoF 双手 6 个任务平均 70.6 分，最强基线 38.0 |
| 2026-09 | [TacBPM](https://arxiv.org/abs/2609.18174) `arXiv:2609.18174` |  | 触觉条件的行为先验：多尺度专家蒸馏成潜在控制器，下游策略用残差潜动作复用它，减少从关节空间重新探索 |
| 2026-08 | [SBLR](https://arxiv.org/abs/2608.15897) `arXiv:2608.15897` |  | 用仿真原生的"预言者传感器"（点云 + 指尖力）训练策略，部署时把真实传感器的潜变量对齐过去，免去为每种传感器写仿真 |
| 2026-06 | [Tactile Genesis](https://arxiv.org/abs/2606.22332) `arXiv:2606.22332` |  | GPU 并行触觉仿真平台：二值接触、深度、力/力矩、标记点位移、温度等触觉抽象统一接口，可跑 2 万个并行环境 |
| 2026-06 | [Blind Dexterous Grasping](https://arxiv.org/abs/2606.11767) `arXiv:2606.11767` |  | 只用稀疏触觉、不看图像的抓取：Real2Sim 触觉标定 + 版图感知编码器 + 扩散策略，零样本上真机 |
| 2026-04 | [ETac](https://arxiv.org/abs/2604.20295) `arXiv:2604.20295` |  | 轻量数据驱动的弹性体形变传播模型，精度接近 FEM，可支撑大规模触觉策略训练 |
| 2026-03 | [PTLD](https://arxiv.org/abs/2603.04531) `arXiv:2603.04531` |  | 不做触觉仿真：用真实世界的特权传感器采集触觉策略数据，再蒸馏出触觉学生策略，明显优于纯本体感觉（RMA）基线 |
| 2026-02 | [HydroShear](https://arxiv.org/abs/2603.00446) `arXiv:2603.00446` |  | 基于水弹性接触的触觉仿真：建模粘滑转换、路径相关的剪切累积和完整 SE(3) 交互，服务触觉 sim2real RL（RSS'26） |
| 2026-02 | [TacMap](https://arxiv.org/abs/2602.21625) `arXiv:2602.21625` |  | 用体积穿透深度图把仿真与真实视触觉信号统一到同一表示，兼顾保真度与大规模 RL 速度 |
| 2026-01 | ★ [Force-Based Sim2Real](https://arxiv.org/abs/2601.02778) `arXiv:2601.02778` |  | 稠密触觉 + 关节力矩的力控 sim2real：快速触觉仿真、电流到力矩标定（省掉力矩传感器）、执行器动力学建模 |
| 2024-09 | [Tactile Skin Sim2Real](https://arxiv.org/abs/2409.12735) `arXiv:2409.12735` |  | 用触觉皮肤做精细操作：在仿真里学习并迁移到真机 |
| 2024-09 | [AnySkin](https://arxiv.org/abs/2409.08276) `arXiv:2409.08276` | NYU 等 | 即插即用的磁性触觉皮肤，可更换且信号一致 |
| 2020-05 | [DIGIT](https://arxiv.org/abs/2005.14679) `arXiv:2005.14679` | Meta AI | 低成本、紧凑、高分辨率视触觉传感器，并演示了手内操作 |

### A·手内操作 RL/sim2real（26）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [FINGR](https://arxiv.org/abs/2609.33973) `arXiv:2609.33973` |  | 单手还原真实魔方：以指尖为参照表示魔方几何，并预测未来接触力与转层进度 |
| 2026-09 | [POISE](https://arxiv.org/abs/2609.13761) `arXiv:2609.13761` |  | 手内 6D 位姿到达：从已有抓取出发，用手指把物体移到掌心坐标系下的目标位姿 |
| 2026-09 | [WM-Craftnet](https://arxiv.org/abs/2609.07002) `arXiv:2609.07002` |  | 把本体、深度、触觉的动作条件潜在动力学作为循环上下文，提升手内操作在扰动下的鲁棒性 |
| 2026-08 | [ADEPT](https://arxiv.org/abs/2608.19182) `arXiv:2608.19182` |  | 大规模 RL 预训练通用的"物体重摆"策略，再稳定地后训练到长时程任务（视触觉输入） |
| 2026-07 | ★ [REGRIND](https://arxiv.org/abs/2607.11874) `arXiv:2607.11874` |  | 把人形全身控制的"重定向 + RL 跟踪"配方搬到灵巧手：一段人类示范 + 残差 RL + 仔细的系统辨识，零样本上真机 |
| 2026-07 | [UHAS](https://arxiv.org/abs/2607.03570) `arXiv:2607.03570` |  | 用球面形变表示的统一手部动作空间 + 级联 IK，同一策略跨 Allegro、LEAP、Shadow 和 MANO 人手 |
| 2026-06 | [DexCompose](https://arxiv.org/abs/2606.28323) `arXiv:2606.28323` |  | 按手指分配"动作所有权"的残差组合，复用两个已训练好的全手策略做多任务 |
| 2026-06 | [Mana](https://arxiv.org/abs/2606.13677) `arXiv:2606.13677` |  | 把铰接工具操作当成"动画"问题：程序生成抓取关键帧，再用运动规划 + RL 补全；每个工具标注不到 1 分钟 |
| 2026-04 | [ViserDex](https://arxiv.org/abs/2604.11138) `arXiv:2604.11138` | ETH Zurich/NVIDIA | 在 3D 高斯泼溅空间里做域随机化，实现单目 RGB 的手内重定向视觉 sim2real（RSS'26） |
| 2026-02 | [SimToolReal](https://arxiv.org/abs/2602.16863) `arXiv:2602.16863` |  | 程序生成大量工具形状，训练一个"把任意工具移到任意目标位姿"的 RL 策略，零样本做工具操作 |
| 2025-10 | ★ [DexNDM](https://arxiv.org/abs/2510.08556) `arXiv:2510.08556` | Tsinghua/PKU/Galbot | 关节级神经动力学模型：用少量真机数据拟合每个关节的等效动力学，修正仿真策略的动作；长宽比达 5.33 的物体也能转（ICLR'26） |
| 2025-09 | [Articulated Tools In-Hand](https://arxiv.org/abs/2509.23075) `arXiv:2509.23075` |  | 仿真训练的基础策略 + 真机示范学到的传感驱动修正，应对摩擦、静摩擦与回差 |
| 2025-08 | [DexReMoE](https://arxiv.org/abs/2508.01695) `arXiv:2508.01695` |  | 多个形状专家组成混合专家策略，泛化到复杂形状的重定向 |
| 2024-07 | [Lessons from Spinning Pens](https://arxiv.org/abs/2407.18902) `arXiv:2407.18902` | UCSD 等 | 仿真 RL 生成轨迹，再用少量真机示范微调；转笔这类高动态任务的经验总结 |
| 2024-05 | [AnyRotate](https://arxiv.org/abs/2405.07391) `arXiv:2405.07391` | Univ. Bristol | 任意手姿态下（重力不变）的多轴旋转，触觉 sim2real |
| 2024-03 | [Twisting Lids Off](https://arxiv.org/abs/2403.02338) `arXiv:2403.02338` | UC Berkeley | 双手拧瓶盖：简单的物体表征 + 大规模 sim2real RL |
| 2023-09 | [RotateIt](https://arxiv.org/abs/2309.09979) `arXiv:2309.09979` | UC Berkeley/Meta | 视觉 + 触觉的通用手内旋转，多模态特权信息蒸馏 |
| 2023-05 | [DexPBT](https://arxiv.org/abs/2305.12127) `arXiv:2305.12127` | NVIDIA | 把种群训练（PBT）扩展到手臂 + 手的高维系统，做抓取与重定向 |
| 2023-03 | [Rotating without Seeing](https://arxiv.org/abs/2303.10880) `arXiv:2303.10880` | UCSD | 只靠手上密集的二值触觉、不用视觉完成旋转 |
| 2022-11 | [Visual Dexterity](https://arxiv.org/abs/2211.11744) `arXiv:2211.11744` | MIT | 只用一个深度相机，实时重定向新颖复杂形状的物体（Science Robotics） |
| 2022-10 | ★ [DeXtreme](https://arxiv.org/abs/2210.13702) `arXiv:2210.13702` | NVIDIA | Isaac Gym 大规模并行训练 + 视觉位姿估计，在 Allegro 手上复现 Dactyl 级别的方块重定向 |
| 2022-10 | ★ [HORA](https://arxiv.org/abs/2210.04887) `arXiv:2210.04887` | UC Berkeley/Meta | 把腿足的快速运动适应（RMA）搬到手内旋转：只用本体感觉在线估计物体属性 |
| 2022-06 | [Bi-DexHands](https://arxiv.org/abs/2206.08686) `arXiv:2206.08686` | PKU | 双手灵巧操作的 RL 基准（Isaac Gym），后来 LLM 奖励设计的常用测试场 |
| 2021-11 | [General In-Hand Reorientation](https://arxiv.org/abs/2111.03043) `arXiv:2111.03043` | MIT | 教师-学生蒸馏，掌心朝下也能重定向大量新物体 |
| 2019-10 | ★ [Rubik's Cube (OpenAI)](https://arxiv.org/abs/1910.07113) `arXiv:1910.07113` | OpenAI | 自动域随机化（ADR）+ 带记忆的策略，单手还原魔方 |
| 2018-08 | ★ [Dactyl](https://arxiv.org/abs/1808.00177) `arXiv:1808.00177` | OpenAI | Shadow 手在大规模域随机化的仿真里用 RL 学会方块重定向，零样本迁移到真机，是灵巧手 sim2real 的起点 |

### A·抓取合成（15）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | [HUGS](https://arxiv.org/abs/2607.04554) `arXiv:2607.04554` |  | 学习人类抓取偏好先验来引导力封闭优化，统一从两指捏取到双手抱持的多种模式 |
| 2026-06 | [From Grasps to Dexterity](https://arxiv.org/abs/2606.30749) `arXiv:2606.30749` |  | 用 35.5 万条抓取轨迹预训练低层控制器，再微调到铰接工具使用（DexCraft 基准） |
| 2026-06 | [AutoDex](https://arxiv.org/abs/2606.23689) `arXiv:2606.23689` |  | 无人值守的真实世界抓取数据采集：20 台相机感知、自动执行、自动标注成败、自动复位 |
| 2026-02 | [OHRA (One Hand to Rule Them All)](https://arxiv.org/abs/2602.16712) `arXiv:2602.16712` | UNC | 统一的 22 DoF 规范 URDF + 参数空间，一个抓取策略跨多种手迁移；3 指 LEAP 零样本 81.9%（RSS'26） |
| 2025-06 | ★ [Dex1B](https://arxiv.org/abs/2506.17198) `arXiv:2506.17198` | UCSD | 生成模型产出 10 亿条抓取与铰接操作演示 |
| 2025-03 | [EvolvingGrasp](https://arxiv.org/abs/2503.14329) `arXiv:2503.14329` |  | 用成功与失败反馈做偏好对齐，让抓取生成模型持续进化 |
| 2025-03 | [DexGrasp Anything](https://arxiv.org/abs/2503.08257) `arXiv:2503.08257` |  | 把物理约束写进扩散模型的训练与采样；新数据集含 340 万个抓取、1.5 万多个物体 |
| 2024-12 | [BODex](https://arxiv.org/abs/2412.16490) `arXiv:2412.16490` | PKU | 双层优化（下层为 QP）GPU 并行合成，单卡每秒 49 个以上；用它训练的模型仿真成功率约 80%，用 DexGraspNet 约 40% |
| 2024-11 | [DextrAH-RGB](https://arxiv.org/abs/2412.01791) `arXiv:2412.01791` | NVIDIA | 只用 RGB 的"抓任意物体"视觉运动策略（特权教师 → RGB 学生） |
| 2024-10 | [Get a Grip](https://arxiv.org/abs/2410.23701) `arXiv:2410.23701` | Stanford | 大规模多指抓取评估，支撑稳健的 sim2real |
| 2024-10 | [DexGraspNet 2.0](https://arxiv.org/abs/2410.23004) `arXiv:2410.23004` | PKU | 在大规模合成的杂乱场景中学习生成式抓取 |
| 2024-07 | [DextrAH-G](https://arxiv.org/abs/2407.02274) `arXiv:2407.02274` | NVIDIA | 几何 fabrics 保证安全 + RL，像素到动作的手臂-手抓取 |
| 2023-04 | [UniDexGrasp++](https://arxiv.org/abs/2304.00464) `arXiv:2304.00464` | PKU | 几何感知课程 + 通才-专才迭代学习，提升抓取策略 |
| 2023-03 | [UniDexGrasp](https://arxiv.org/abs/2303.00938) `arXiv:2303.00938` | PKU | 先生成多样的抓取提案，再用目标条件策略执行，做通用灵巧抓取 |
| 2022-10 | ★ [DexGraspNet](https://arxiv.org/abs/2210.02697) `arXiv:2210.02697` | PKU | 用加速的可微力封闭估计合成 132 万个 ShadowHand 抓取（5355 个物体） |

### A·基于模型的接触控制（11）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [Primitive-Informed MPC](https://arxiv.org/abs/2609.14868) `arXiv:2609.14868` |  | 用低维操作原语偏置采样，再优化关节残差；在 16 DoF Allegro 上提高采样效率 |
| 2026-04 | [Global-MPPI (KernelSOS)](https://arxiv.org/abs/2604.27175) `arXiv:2604.27175` |  | 核平方和做全局探索，渐进非凸平滑做局部细化；测试含手内方块重定向 |
| 2026-03 | [ComFree-Sim](https://arxiv.org/abs/2603.12185) `arXiv:2603.12185` |  | 不用互补约束的解析接触 GPU 物理引擎，接触计算随接触数近线性扩展，适合接触丰富的仿真与控制 |
| 2026-01 | ★ [Hierarchical RL-MPC](https://arxiv.org/abs/2601.10930) `arXiv:2601.10930` |  | 高层 RL 决定"碰哪里"（接触意图），低层接触隐式 MPC 决定"怎么推"，几何感知的 sim2real |
| 2025-05 | ★ [Contact Trust Region](https://arxiv.org/abs/2505.02291) `arXiv:2505.02291` | MIT 等 | 尊重接触单边性的信赖域：局部接触 MPC + 全局拼接，不靠并行也比 RL 和采样 MPC 省算力（IJRR） |
| 2025-02 | ★ [MuJoCo Playground](https://arxiv.org/abs/2502.08844) `arXiv:2502.08844` | Google DeepMind 等 | 基于 MJX 的开源机器人学习框架，含灵巧手任务，单 GPU 几分钟训练，零样本上真机 |
| 2024-12 | [Learn from Sampling Planners?](https://arxiv.org/abs/2412.09743) `arXiv:2412.09743` |  | 讨论用采样式规划器生成的轨迹做模仿学习的利弊 |
| 2024-11 | [Tendon-Hand Sampling MPC](https://arxiv.org/abs/2411.06183) `arXiv:2411.06183` |  | 首次在实体仿生腱驱动手上用采样 MPC 做手内操作；不需长时间训练，但目标函数要按任务调 |
| 2024-09 | [DROP](https://arxiv.org/abs/2409.14562) `arXiv:2409.14562` | Caltech | 在线采样规划 + 实时状态估计，完成手内方块重定向（ICRA'25） |
| 2022-12 | [Predictive Sampling](https://arxiv.org/abs/2212.00541) `arXiv:2212.00541` | Google DeepMind | MuJoCo MPC 中的采样式实时行为合成，是灵巧手采样 MPC 的基础 |
| 2022-06 | ★ [Quasi-Dynamic Smoothing](https://arxiv.org/abs/2206.10787) `arXiv:2206.10787` | MIT | 对准静态接触模型做局部平滑，配合 RRT 实现全局的接触丰富规划 |

### A·遥操作与重定向（17）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | [ReForce](https://arxiv.org/abs/2608.15560) `arXiv:2608.15560` |  | 力感知重定向：在运动学结果上预测残差以达到期望接触力，支持在线遥操作与离线数据转换 |
| 2026-07 | [TeleDexter](https://arxiv.org/abs/2607.11481) `arXiv:2607.11481` |  | "手-物协同跟踪"控制器把操作者意图映射成学到的接触执行，迈向人类水平的手内遥操作 |
| 2026-07 | [AnyDexRT](https://arxiv.org/abs/2607.08341) `arXiv:2607.08341` |  | 免标定的重定向：自监督指尖对应 + 少量人工示例锚定 + 捏取接触分类器 |
| 2026-06 | [DexTeleop-0](https://arxiv.org/abs/2606.23431) `arXiv:2606.23431` |  | 触觉驱动的实时优化把粗糙的人手意图转成力顺应的精细指令，双手共享自主 |
| 2026-05 | [DexTwist](https://arxiv.org/abs/2605.12182) `arXiv:2605.12182` |  | 针对拧、转类动作的功能性"旋量重定向"，避免纯运动学模仿导致指尖打滑 |
| 2026-03 | [DexEXO](https://arxiv.org/abs/2603.17323) `arXiv:2603.17323` |  | 以佩戴舒适为先的外骨骼：适配 140–217 mm 手长，被动手外观与机器人一致，可直接用腕部相机图像训练 |
| 2025-10 | [Shared-Autonomy DexGrasp-VLA](https://arxiv.org/abs/2511.00139) `arXiv:2511.00139` |  | 人用 VR 管手臂，自主 VLA 管手指（视觉 + 触觉），共享自主高效采集数据 |
| 2025-06 | [Retargeting Objectives](https://arxiv.org/abs/2506.09384) `arXiv:2506.09384` |  | 用真机对比实验分析重定向优化目标中各项的作用 |
| 2025-05 | [DexUMI](https://arxiv.org/abs/2505.21864) `arXiv:2505.21864` | Stanford/Columbia | 外骨骼缩小运动学差距、视频里把人手修补成机器人手，人手当"通用操作接口"，两种手上平均成功率 86% |
| 2025-05 | [DexWild](https://arxiv.org/abs/2505.07813) `arXiv:2505.07813` | CMU | 低成本移动设备让人用自己的手在野外大量采数据，与机器人数据共训 |
| 2025-02 | ★ [DexterityGen](https://arxiv.org/abs/2502.04307) `arXiv:2502.04307` | UC Berkeley/Meta | RL 预训练大规模灵巧运动原语，人只给粗指令，扩散模型生成安全精确的动作，相当于灵巧手的"基础控制器" |
| 2024-07 | [Bunny-VisionPro](https://arxiv.org/abs/2407.03162) `arXiv:2407.03162` | UCSD | 用 Vision Pro 做实时双手灵巧遥操作，带触觉反馈，服务模仿学习 |
| 2024-04 | [HATO](https://arxiv.org/abs/2404.16823) `arXiv:2404.16823` | UC Berkeley | 双多指手的视触觉技能学习与低成本遥操作 |
| 2024-03 | [OPEN TEACH](https://arxiv.org/abs/2403.07870) `arXiv:2403.07870` | NYU | 基于 VR 头显的通用遥操作系统 |
| 2024-03 | ★ [DexCap](https://arxiv.org/abs/2403.07788) `arXiv:2403.07788` | Stanford | 便携动捕手套 + 相机采集人手数据，配合人在环纠正学双手灵巧技能 |
| 2023-07 | ★ [AnyTeleop](https://arxiv.org/abs/2307.04577) `arXiv:2307.04577` | UCSD/NVIDIA | 通用视觉遥操作框架，支持多种机械臂与灵巧手 |
| 2019-10 | [DexPilot](https://arxiv.org/abs/1910.03135) `arXiv:1910.03135` | NVIDIA | 纯视觉（不戴手套）遥操作手臂-手系统，提出经典的重定向优化 |

### A·人类演示→机器人（17）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [Gen2Dex](https://arxiv.org/abs/2610.24392) `arXiv:2610.24392` |  | 把仿真场景初始化的生成视频转成物理可执行的灵巧演示（残差 RL 修正） |
| 2026-09 | [Contact-Anchored Retargeting](https://arxiv.org/abs/2609.24093) `arXiv:2609.24093` |  | 迁移"哪些手指部位碰物体哪里、按什么顺序"，而不是关节运动；一个残差 RL 修复动捕数据的物理一致性 |
| 2026-09 | [Dex-X](https://arxiv.org/abs/2609.07747) `arXiv:2609.07747` |  | 把仿真当"触觉补全引擎"：从人类视频重建交互，在仿真里得到触觉监督，学视触觉策略 |
| 2026-06 | [Do as I Do](https://arxiv.org/abs/2606.19333) `arXiv:2606.19333` |  | 重建并重定向单目 RGB 的野外人类视频（第一或第三人称），生成可在真机执行的灵巧数据 |
| 2026-06 | [TopoRetarget](https://arxiv.org/abs/2606.16272) `arXiv:2606.16272` |  | 稀疏交互图 + 拉普拉斯形变，在重定向中保持手-物接触结构；转笔、方块重定向零样本上真机 |
| 2026-06 | [EaDex](https://arxiv.org/abs/2606.03268) `arXiv:2606.03268` |  | 单台 RGB-D 相机采人手示范 + 接触奖励退火，低成本的多本体灵巧学习 |
| 2026-02 | ★ [EgoScale](https://arxiv.org/abs/2602.16710) `arXiv:2602.16710` |  | 用 2 万多小时带动作标签的第一人称人类视频训练 VLA，发现人类数据规模与验证损失呈对数线性关系 |
| 2026-02 | [DexImit](https://arxiv.org/abs/2602.10105) `arXiv:2602.10105` |  | 把单目人类视频自动转成物理合理的双手灵巧机器人数据（四阶段管线） |
| 2026-02 | [VideoManip](https://arxiv.org/abs/2602.09013) `arXiv:2602.09013` |  | 不用穿戴设备：从 RGB 人类视频重建 3D 手-物轨迹，再重定向到机器人手 |
| 2025-11 | [SPIDER](https://arxiv.org/abs/2511.09484) `arXiv:2511.09484` |  | 物理感知的采样式重定向，把纯运动学的人类演示变成动力学可行的机器人轨迹，覆盖 9 种人形和灵巧手 |
| 2025-10 | [DexMan](https://arxiv.org/abs/2510.08475) `arXiv:2510.08475` |  | 从第三人称人类视频（以及生成视频）学人形双手灵巧技能，用接触奖励应对噪声位姿 |
| 2025-05 | ★ [DexMachina](https://arxiv.org/abs/2505.24853) `arXiv:2505.24853` | NVIDIA 等 | 功能性重定向：用强度逐渐衰减的"虚拟物体控制器"做课程，让策略逐步接管；附多手基准 |
| 2025-04 | [Human2Sim2Robot](https://arxiv.org/abs/2504.12609) `arXiv:2504.12609` | Stanford | 只用一段 RGB-D 人类视频：用物体轨迹定义与本体无关的奖励，在仿真里 RL 跨越本体差距 |
| 2025-03 | ★ [ManipTrans](https://arxiv.org/abs/2503.21860) `arXiv:2503.21860` |  | 先训通用轨迹模仿器，再训交互约束下的残差模块，把人类双手技能迁移到机器人手（CVPR'25） |
| 2025-02 | ★ [DexTrack](https://arxiv.org/abs/2502.09614) `arXiv:2502.09614` | Tsinghua 等 | 通用神经跟踪控制器：给定人类参考生成手部动作，用数据飞轮迭代增强（ICLR'25） |
| 2022-12 | [VideoDex](https://arxiv.org/abs/2212.04498) `arXiv:2212.04498` | CMU | 从互联网视频学习灵巧先验 |
| 2021-08 | [DexMV](https://arxiv.org/abs/2108.05877) `arXiv:2108.05877` | UCSD | 从人类视频重建手-物姿态，重定向后做模仿学习 |

### A·数据规模化（1）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2024-10 | [DexMimicGen](https://arxiv.org/abs/2410.24185) `arXiv:2410.24185` | NVIDIA | 从少量人类示范自动扩增双手灵巧操作数据（MimicGen 的扩展） |

### A·灵巧 VLA/基础模型（9）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-05 | [Dexora](https://arxiv.org/abs/2605.18722) `arXiv:2605.18722` |  | 首个开源的双臂双手高自由度 VLA，外骨骼背包 + Vision Pro 混合遥操作采数据 |
| 2026-03 | ★ [XL-VLA](https://arxiv.org/abs/2603.10158) `arXiv:2603.10158` |  | 多种灵巧手共享潜在动作空间：4 种手 10 个任务平均 72%，π0 基线 32% |
| 2026-01 | [Helix 02](https://interestingengineering.com/ai-robotics/figure-helix02-upgrades-humanoid-robot-control) | Figure | 一个网络控制全身；借助指尖触觉与掌心相机分药片、推注射器（公司演示） |
| 2025-07 | [Being-H0](https://arxiv.org/abs/2507.15597) `arXiv:2507.15597` | BeingBeyond/PKU | 把人手当"基础操作器"，用大规模人类视频做物理指令微调的灵巧 VLA |
| 2025-07 | [EgoVLA](https://arxiv.org/abs/2507.12440) `arXiv:2507.12440` | UCSD 等 | 用第一人称人类视频训练 VLA 预测手腕与手部动作，再经 IK 与重定向转成机器人动作 |
| 2025-07 | [DexVLG](https://arxiv.org/abs/2507.02747) `arXiv:2507.02747` |  | 1.7 亿个与语义部件对齐的抓取（DexGraspNet 3.0），训练 VLM + 流匹配位姿头，按语言指令生成灵巧抓取 |
| 2025-03 | [Gemini Robotics](https://arxiv.org/abs/2503.20020) `arXiv:2503.20020` | Google DeepMind | 微调后能完成折纸等高灵巧、长时程任务 |
| 2025-03 | [GR00T N1](https://arxiv.org/abs/2503.14734) `arXiv:2503.14734` | NVIDIA | 人形双系统基础模型，支持灵巧手；后续版本加强全身灵巧控制 |
| 2025-02 | ★ [DexGraspVLA](https://arxiv.org/abs/2502.20900) `arXiv:2502.20900` |  | 预训练 VLM 做高层规划 + 扩散低层控制器，杂乱场景中语言引导抓取成功率 90% 以上（AAAI'26） |

### A·人形全身灵巧（4）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [DexWeave](https://arxiv.org/abs/2609.34724) `arXiv:2609.34724` |  | 交互一致的身体 + 手重定向，加上解剖感知的全身策略，从人类示范学人形灵巧移动操作 |
| 2026-06 | ★ [CoorDex](https://arxiv.org/abs/2606.23680) `arXiv:2606.23680` | UNC 等 | 身体与手分别训特权跟踪教师、蒸馏成潜在先验，再用协调残差 RL 实现边走边做灵巧操作（Unitree G1 + 20 DoF Wuji 手） |
| 2026-03 | [HumDex](https://arxiv.org/abs/2603.12260) `arXiv:2603.12260` |  | IMU 全身动捕 + 学习式手部重定向的便携人形灵巧遥操作 |
| 2025-12 | [WholeBodyVLA](https://arxiv.org/abs/2512.11047) `arXiv:2512.11047` |  | 从无动作标签的第一人称视频学统一的潜在 VLA，做全身移动操作（ICLR'26） |

### A·基准与数据集（5）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | ★ [DexVerse](https://arxiv.org/abs/2607.08751) `arXiv:2607.08751` |  | 100 个任务、3 种机械臂、6 种灵巧手的模块化多任务多本体基准 |
| 2024-03 | [OakInk2](https://arxiv.org/abs/2403.19417) `arXiv:2403.19417` | SJTU | 复杂任务中的双手-物体操作数据集 |
| 2023-05 | [DexArt](https://arxiv.org/abs/2305.05706) `arXiv:2305.05706` | UCSD | 铰接物体上的通用灵巧操作基准 |
| 2022-04 | [ARCTIC](https://arxiv.org/abs/2204.13662) `arXiv:2204.13662` | ETH/MPI | 双手与铰接物体交互的数据集 |
| 2021-04 | [DexYCB](https://arxiv.org/abs/2104.04631) `arXiv:2104.04631` | NVIDIA | 人手抓取物体的多视角数据集 |

### 综述/资源（4）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-05 | ★ [Dexterous Hand Intelligence Survey](https://arxiv.org/abs/2605.13925) `arXiv:2605.13925` |  | 从硬件、控制与学习方法、数据与评测、局限四个角度的灵巧手综述（2026-05） |
| 2026 | [Awesome-Dexterous-Hands](https://github.com/CyanHaze/Awesome-Dexterous-Hands) | 社区 | 灵巧手论文、数据集、仿真器、硬件与开源资源清单 |
| 2025-05 | [Interactive IL Survey](https://arxiv.org/abs/2506.00098) `arXiv:2506.00098` | KIT | 灵巧操作的交互式模仿学习综述 |
| 2024-01 | [In-Hand Manipulation Survey](https://arxiv.org/abs/2401.07915) `arXiv:2401.07915` |  | 基于学习的手内操作方法综述 |

### B·LLM 奖励与仿真参数（4）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-05 | [DexSim2Real](https://arxiv.org/abs/2605.05241) `arXiv:2605.05241` |  | 让 VLM 当"视觉真实感评判"，闭环 CMA-ES 优化仿真的域随机化参数；真机平均成功率 78.2% |
| 2026-02 | [RF-Agent](https://arxiv.org/abs/2602.23876) `arXiv:2602.23876` |  | 把奖励设计看成序列决策，用蒙特卡洛树搜索管理 LLM 的奖励迭代，在 Bi-DexHands 等 17 个任务上测试 |
| 2025-09 | ★ [Text2Touch](https://arxiv.org/abs/2509.07445) `arXiv:2509.07445` | Univ. Bristol 等 | LLM 设计含触觉的奖励（可处理 70 多个环境变量），多轴手内旋转上真机，比人工精调的奖励更好、也更短 |
| 2023-10 | ★ [Eureka](https://arxiv.org/abs/2310.12931) `arXiv:2310.12931` | NVIDIA/UPenn | GPT-4 写奖励代码并进化式迭代，第一次让仿真里的 Shadow 手学会快速转笔 |

### B·生成式仿真/agentic 数据（3）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/) | Google DeepMind | 把第一人称人类视频重定向到带 22 DoF 手的仿真人形，并能从视频或草图生成 MuJoCo 孪生与训练数据 |
| 2026-09 | ★ [DexAgent](https://arxiv.org/abs/2609.35318) `arXiv:2609.35318` | Stanford/Columbia | 一段第一人称视频 + 任务描述 → agent 从工具库选工具或自造新工具，完成理解、仿真重建、轨迹优化与数据生成；11 个真机任务成功率是基线的 3.5 倍 |
| 2025-11 | ★ [GenDexHand](https://arxiv.org/abs/2511.01791) `arXiv:2511.01791` |  | 灵巧手的生成式仿真：VLM 反馈闭环调整物体位置与尺度，任务拆成子任务顺序 RL |

### B·VLM 语义/功能抓取（7）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | [AdaRoboVLG](https://arxiv.org/abs/2609.04096) `arXiv:2609.04096` |  | 通用基础抓取策略 + 可组合的基础模型先验，按任务调整抓取而不用重训，跨手泛化 |
| 2026-04 | [Bimanual Affordance via VLM](https://arxiv.org/abs/2604.08726) `arXiv:2604.08726` |  | VLM 联合推理双手各自的接触区域与分工 |
| 2026-04 | [BLaDA](https://arxiv.org/abs/2604.08410) `arXiv:2604.08410` |  | 把开放词汇指令解析成结构化约束，在 3D 高斯场里定位功能部位，零样本功能性灵巧抓取 |
| 2026-01 | [DextER](https://arxiv.org/abs/2601.16046) `arXiv:2601.16046` |  | 先自回归生成"哪个手指部位碰物体哪里"的接触 token，再生成抓取，把接触当作具身推理的中间表示 |
| 2025-12 | [OmniDexVLG](https://arxiv.org/abs/2512.03874) `arXiv:2512.03874` |  | 多 agent 协作 + 检索增强 + 思维链推理抓取类型、接触语义与功能可供性，再生成可控的灵巧抓取 |
| 2025-03 | [AffordDexGrasp](https://arxiv.org/abs/2503.07360) `arXiv:2503.07360` |  | 用可泛化的"指导性可供性"连接语言语义与灵巧抓取，面向开放集类别（ICCV'25） |
| 2024-12 | [Multi-GraspLLM](https://arxiv.org/abs/2412.08468) `arXiv:2412.08468` |  | 多模态 LLM 按语义指令为多种手生成抓取 |

### B·VLM 规划+灵巧技能（4）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [AW-0 (Agentic WAM)](https://arxiv.org/abs/2610.30136) `arXiv:2610.30136` |  | 代码、未来视频与动作三路联合生成：代码专家写参数化技能，视频与动作专家给出预测和电机指令；在 YAM 臂与灵巧手上实测 |
| 2025-03 | [Being-0](https://arxiv.org/abs/2503.12533) `arXiv:2503.12533` | BeingBeyond/PKU | 基础模型做高层认知，Connector VLM 衔接模块化技能库（行走与灵巧操作）的人形 agent |
| 2025-03 | [RoboDexVLM](https://arxiv.org/abs/2503.01616) `arXiv:2503.01616` |  | VLM 任务规划（带任务级恢复）+ 语言引导的灵巧抓取感知，完成长序列开放词汇任务 |
| 2024-11 | [DexHandDiff](https://arxiv.org/abs/2411.18562) `arXiv:2411.18562` |  | 交互感知的扩散规划，用 LLM 生成的交互引导适配不同任务目标（CVPR'25） |

### B·编码 agent/自我进化（5）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | [Embodied Playground](https://arxiv.org/abs/2610.27039) `arXiv:2610.27039` |  | 长时程 agentic 基准，含需要推断状态并逐层转动的魔方、灵巧手版汉诺塔等任务 |
| 2026-10 | [SEAR](https://arxiv.org/abs/2610.18792) `arXiv:2610.18792` |  | 自我进化的 agentic 机器人框架与基准：代码即策略、agent 即策略、自动研究三种模式，7 个前沿模型、330 个仿真 + 5 个真机任务 |
| 2026-09 | ★ [EmbodiedSWE](https://arxiv.org/abs/2609.27308) `arXiv:2609.27308` |  | 编码 agent 解决长达半小时的接触丰富/灵巧任务，再把一个解扩增成大量 VLA 训练轨迹 |
| 2026-09 | [Dexterous Astra](https://github.com/jianglongye/dexterous-astra) | 个人发布（UCSD 博士生） | GPT-6 Astra 自己搭环境、写代码、训 PPO：Isaac Lab 中 Sharpa Wave 手转笔三圈，MuJoCo 中 Wuji 手解出魔方 U F L 三步；公开了提示词与检查点 |
| 2026-06 | ★ [ASPIRE](https://arxiv.org/abs/2607.00272) `arXiv:2607.00272` | NVIDIA 等 | 编码 agent 持续写、修控制程序并沉淀技能库；在真机灵巧手上解魔方，效率随经验提升 |

### B·前沿模型直接控手（5）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | ★ [GPT-6-Astra Understands Touch](https://arxiv.org/abs/2610.28457) `arXiv:2610.28457` |  | 把专家示范总结成任务指导与触觉参考，冻结的 GPT-6 Astra 在 5 个 UniVTAC 任务上平均 93.8%，触觉让插孔任务 31%→99% |
| 2026-10 | [TacTalent](https://tactalent.github.io/) |  | GPT-6 Astra 作视触觉策略：19 个仿真任务少样本评测，加触觉后两组平均 93.8% 和 81.8%（只用视觉为 76.3% 和 80.6%） |
| 2026-10 | [Has GPT-6-Astra Achieved Embodied Intelligence?](https://arxiv.org/abs/2610.22471) `arXiv:2610.22471` | SenseNova | 12 个基准（含灵巧手）统一评测：具身能力大体具备，实用的具身智能尚未具备 |
| 2026-10 | [Awesome-Astra-Embodied-AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI) | 社区 | GPT-6 Astra 具身用例汇总，含多个灵巧手案例（魔方、苹果梗抓取、44 DoF 手重定向、腱驱动手重建） |
| 2026-09 | ★ [GPT-6 Astra as Embodied Policies](https://arxiv.org/abs/2609.38537) `arXiv:2609.38537` | Galbot | 灵巧操作部分：与学习策略混合控制，在 DexJoCo 的 10 次试验中成功 50%；直接控制手内操作时难以协调手指接触 |

