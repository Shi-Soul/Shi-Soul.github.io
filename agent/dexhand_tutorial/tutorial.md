# 灵巧手操作技术教程：给运控研究者

> 把调研页里的工作串成一套能动手的技术路线：问题设定与被控对象、仿真 RL + sim2real、人类演示 → 重定向 → 跟踪 RL、抓取、采样 MPC、触觉、模仿学习与 VLA、真机 RL、世界模型，以及 agent 如何自动化"奖励-环境-数据-训练"这个慢循环。每章给出公式、代码骨架、常见坑和自测题。

## 0. 导读：这份教程讲什么、怎么读 {#ch0}

这份教程把[调研页](../dexhand_survey/)里的工作串成一套能动手的技术路线。调研页回答"有哪些工作"，教程回答"这些方法怎么做、为什么这样做、坑在哪里"。

读者设定和调研页一样：你熟悉腿足或人形的运动控制——PPO 与大规模并行仿真、域随机化与 sim2real、MPC、全身控制、动作重定向——但没有系统做过灵巧手。所以我们尽量用你已有的概念来解释，并在正文里用"运控类比"框标出对应关系。

### 0.1 核心问题地图 {#ch0-1}

灵巧操作的算法问题可以归结为下面十几个核心问题。教程按这张表组织，每章先把问题说清楚（尽量形式化），再给出解法的分类与取舍，最后把一个代表性方法讲透；不重要的细节一律略过。

| 核心问题 | 难在哪里 | 主要解法 | 章 |
|---|---|---|---|
| 被控对象是什么 | 物体没有执行器；接触是单边、带摩擦的互补约束 | 接触力学、抓取静力学、仿真器的近似方式 | 1 |
| 仿真模型对不对 | 小力矩下摩擦、回差、延迟、转子惯量占比大 | 执行器建模、系统辨识 | 2 |
| 控制怎么学出来 | 高维混合系统，探索难，奖励容易被钻空子 | 有界任务项 + 正则的奖励；用初始状态绕开探索；并行 PPO | 3 |
| 物体属性看不见 | 部分可观测 | 教师-学生蒸馏、从历史中适应 | 3 |
| 仿真不等于真机 | 模型误差、时延、噪声、未建模动力学 | 把差距写成可随机化的参数；ADR | 3 |
| 长时程任务没法写奖励 | 奖励设计与探索同时变难 | 人类演示 → 重定向 → 跟踪 RL | 4 |
| 抓取是否可行 | 力封闭、可达性、执行时的动态 | 抓取能量优化、双层优化、生成模型、闭环策略 | 5 |
| 有模型时怎么在线优化 | 接触让动力学非光滑、梯度不可靠 | 接触平滑、凸化的接触模型、采样 MPC | 6 |
| 接触状态看不见 | 遮挡；触觉难仿真、传感器各异 | 触觉表示学习；不仿真触觉的迁移方法 | 7 |
| 用数据训练通用策略 | 数据缺力、本体差距、动作分布多峰 | 生成式策略、基础控制器、跨本体表示、人类视频预训练 | 8 |
| 真机上继续变好 | 样本贵、失败代价高 | 离策略 RL + 先验数据 + 人在环或 MPC 引导 | 9 |
| 预测接触会怎么演化 | 视觉看不到接触，交互动力学复杂 | 动作条件的（视触觉）世界模型 | 10 |
| 自动化研发外环 | 搜索空间大，结果需要可靠验证 | LLM 驱动的奖励与环境搜索、编码 agent | 11 |

只想抓主线的话，读第 1、3、4、9 章；做抓取的加第 5 章，做 MPC 的加第 6 章，做数据与大模型的加第 8、10、11 章。第 12 章是把这些串成一条可执行路线的检查表。

### 0.2 约定 {#ch0-2}

- 代码是**说明性骨架**（PyTorch / NumPy 风格），用来讲清结构，不是可直接运行的完整工程；示例中未注明出处的数值都是示意，需要按你的硬件标定。
- 凡是来自具体论文的公式与数字，正文都注明了出处，并已对照原文核对；论文之间的任务与硬件设置不同，数字不能横向比较。
- 引用的工作链接到 arXiv 或项目页，悬停可见完整标题；文末附录按章节列出全部引用。

常用记号：

| 记号 | 含义 |
|---|---|
| $q, \dot{q} \in \mathbb{R}^n$ | 手的关节角与角速度（$n$ 为自由度，Allegro、LEAP 为 16） |
| $q^*, a_t$ | 关节 PD 目标；策略在 $t$ 时刻输出的动作 |
| $\tau$ | 关节力矩 |
| $p_o, R_o, v_o, \omega_o$ | 物体的位置、姿态、线速度、角速度 |
| $c_i, n_i, f_i, \mu$ | 第 $i$ 个接触点、接触法向、接触力、摩擦系数 |
| $o_t, e_t, z_t$ | 真机可得的观测；只有仿真里才有的特权信息；特权信息的低维编码 |
| $\pi, \phi$ | 策略；从历史估计 $z$ 的适应模块 |

## 1. 问题设定：灵巧操作到底在控制什么 {#ch1}

### 1.1 一个闭环 {#ch1-1}

<figure class="fig"><div class="fig-scroll"><svg viewBox="0 0 760 236" role="img" aria-label="灵巧操作闭环示意图"><defs><marker id="ah1" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="sv-head"/></marker></defs><rect class="sv-box" x="20" y="24" width="130" height="60" rx="10"/><text class="sv-t1" x="85" y="50" text-anchor="middle">策略 π</text><text class="sv-t3" x="85" y="70" text-anchor="middle">10–50 Hz</text><rect class="sv-box" x="210" y="24" width="130" height="60" rx="10"/><text class="sv-t1" x="275" y="50" text-anchor="middle">PD / 阻抗</text><text class="sv-t3" x="275" y="70" text-anchor="middle">100–1000 Hz</text><rect class="sv-box" x="400" y="24" width="130" height="60" rx="10"/><text class="sv-t1" x="465" y="50" text-anchor="middle">手</text><text class="sv-t3" x="465" y="70" text-anchor="middle">电机 · 腱 · 连杆</text><rect class="sv-hl" x="590" y="24" width="150" height="60" rx="10"/><text class="sv-t1" x="665" y="50" text-anchor="middle">物体</text><text class="sv-t3" x="665" y="70" text-anchor="middle">没有执行器</text><path class="sv-edge" d="M150,54 L208,54" marker-end="url(#ah1)"/><text class="sv-lab" x="180" y="46" text-anchor="middle">q*</text><text class="sv-t3" x="180" y="72" text-anchor="middle">关节目标</text><path class="sv-edge" d="M340,54 L398,54" marker-end="url(#ah1)"/><text class="sv-lab" x="370" y="46" text-anchor="middle">τ</text><text class="sv-t3" x="370" y="72" text-anchor="middle">力矩</text><path class="sv-edge acc" d="M530,54 L588,54" marker-end="url(#ah1)"/><text class="sv-lab" x="560" y="46" text-anchor="middle">f<tspan dy="3" font-size="9">i</tspan></text><text class="sv-t3" x="560" y="72" text-anchor="middle">接触力</text><rect class="sv-box" x="400" y="150" width="340" height="60" rx="10"/><text class="sv-t1" x="570" y="176" text-anchor="middle">传感</text><text class="sv-t3" x="570" y="196" text-anchor="middle">关节角 · 电机电流 · 触觉 · 相机</text><rect class="sv-box" x="20" y="150" width="320" height="60" rx="10"/><text class="sv-t1" x="180" y="176" text-anchor="middle">观测与历史</text><text class="sv-t3" x="180" y="196" text-anchor="middle">关节与动作历史 · 物体位姿估计 · 触觉特征</text><path class="sv-edge" d="M465,84 L465,148" marker-end="url(#ah1)"/><path class="sv-edge" d="M665,84 L665,148" marker-end="url(#ah1)"/><path class="sv-edge" d="M400,180 L342,180" marker-end="url(#ah1)"/><path class="sv-edge" d="M85,150 L85,86" marker-end="url(#ah1)"/><text class="sv-t3" x="676" y="122">物体位姿 / 接触</text></svg></div><figcaption>图 1　灵巧操作的闭环。策略低频输出关节目标，PD 高频转成力矩；手只能经由接触力去推动物体，物体本身没有执行器。</figcaption></figure>

灵巧手的闭环和腿足几乎一样：策略低频输出关节目标，高频 PD（或阻抗）控制器把目标变成力矩，传感器把关节、电流、触觉、图像送回策略。区别在于被控对象：腿足要控制的是机器人自己的躯干，灵巧手要控制的是**手里的物体**。

### 1.2 被控对象没有执行器 {#ch1-2}

物体的运动只由接触力和重力决定（Newton–Euler 方程）：

$$m\,\dot v_o = \sum_i f_i + m g, \qquad I\,\dot\omega_o + \omega_o \times I\omega_o = \sum_i (c_i - p_o) \times f_i$$

手自己的动力学里，接触力以反作用的形式出现：

$$M(q)\,\ddot q + h(q, \dot q) = \tau - \sum_i J_i(q)^\top f_i$$

每个接触力都要满足单边约束和库仑摩擦：法向力只能推、不能拉；切向力不能超过摩擦锥；接触距离和法向力不能同时为正（互补约束）：

$$f_i = f_{n,i}\, n_i + f_{t,i}, \qquad \lVert f_{t,i} \rVert \le \mu f_{n,i}, \qquad 0 \le d_i(q, p_o, R_o) \perp f_{n,i} \ge 0$$

这三组式子带来三个后果，后面每一章都在和它们打交道：

1. **动作只能"借道"接触作用于物体。** 没有接触时物体只受重力；控制物体，实际上是在控制接触的建立、保持、滑移和断开。
2. **系统是混合系统。** 接触的通断是离散事件，接触模式随手指数、接触面数组合爆炸，梯度在模式切换处不连续——这是第 6 章基于模型的方法要正面处理的难点，也是 RL 用采样"绕过去"的东西。
3. **摩擦锥是不等式约束。** 推的方向不对接触就会滑，力不够物体就会掉；能稳住物体的手指配置是一个很窄的可行集。

### 1.3 仿真器怎么解接触：时间步进与软约束 {#ch1-3}

连续形式的互补问题很难直接积分，物理引擎都用**速度层面的时间步进**：把一个步长 $h$ 内的接触力看成冲量 $\lambda$，求下一时刻的速度 $v^{+}$：

$$M(q)\,(v^+ - v) = h\,\bigl(\tau - c(q, v)\bigr) + J(q)^\top \lambda$$

$$0 \le \phi_i/h + J_{n,i}\, v^+ \perp \lambda_{n,i} \ge 0, \qquad \lambda_i \in K_i = \bigl\{ (\lambda_n, \lambda_t) : \lVert \lambda_t \rVert \le \mu \lambda_n \bigr\}$$

$\phi_i$ 是带符号距离（间隙），$J_{n,i}$ 把广义速度映射到第 $i$ 个接触的法向速度。互补条件说的是：要么下一步间隙为正、法向冲量为零（分离），要么间隙闭合、冲量为正（接触）。再加上"摩擦力方向使能量耗散最大"（最大耗散原理），就是一个非线性互补问题（NCP）；把摩擦锥线性化成多面锥，就成了线性互补问题（LCP，Stewart–Trinkle 一类格式）。

三类主流处理方式：

- **直接解 LCP / NCP。** 物理上最"硬"，但求解难、对接触数敏感，解还可能不唯一。
- **Anitescu 凸松弛。** 把互补条件换成锥互补：$K_i^* \ni J_i v^+ + (\phi_i/h)\, e_n \perp \lambda_i \in K_i$（$e_n$ 只在法向分量上取 1，$K_i^*$ 是摩擦锥的对偶锥），前向动力学变成一个凸的二阶锥规划。代价是相对滑动的物体之间会出现一层轻微的非物理"边界层"效应，步长 $h \to 0$ 时消失；好处是唯一需要调的参数就是 $h$。[Pang 等人](https://arxiv.org/abs/2206.10787)的接触规划就建在这个松弛上（第 6 章）。
- **MuJoCo 的软约束凸优化。** MuJoCo 把约束力写成凸优化

$$f^* = \arg\min_{f \in \Omega}\; \tfrac{1}{2} f^\top (A + R)\, f + f^\top (a_0 - a_{\text{ref}}), \qquad A = J M^{-1} J^\top$$

其中 $A$ 是约束空间里的逆惯量，$a_0$ 是不加约束时约束空间的加速度，$\Omega$ 是摩擦锥。两组参数决定接触有多"软"：solref（时间常数、阻尼比）定义参考加速度 $a_{\mathrm{ref}}$，即约束违反量像弹簧-阻尼系统那样被拉回；solimp 定义约束"阻抗" $d \in (0, 1)$ 及其随穿透深度的变化，进而决定正则项 $R$——$d$ 越接近 1，约束越硬、穿透越小，数值上也越难。

对灵巧手来说，"软"不是小事：毫米级的穿透就相当于毫米级的指尖位置误差，软接触还会让被夹住的物体缓慢"蠕滑"。MuJoCo 提供椭圆锥与多面锥（cone）、扭转与滚动摩擦（condim）、额外的防滑迭代（noslip_iterations）等选项来折中。多面锥会带来非物理的各向异性摩擦，这也是 Pang 等人坚持用二阶锥的原因。

GPU 引擎（Isaac Gym / PhysX）用迭代式求解，迭代次数就是精度旋钮。[DeXtreme](https://arxiv.org/abs/2210.13702) 的 Allegro 环境设置是：仿真步长 1/60 s、控制步长 1/30 s、每步 2 个子步、8 次位置迭代、0 次速度迭代，8192 或 16384 个并行环境。

!!! key "对算法的影响"
    你看到的"仿真接触"，是某个求解器在某组参数下的近似。RL 会利用这个近似里的任何便宜（软接触允许的轻微穿透、多面锥带来的各向异性摩擦、迭代没收敛时的能量注入）。所以接触参数和求解器设置本身就是域随机化与 sim2real 的一部分——第 2、3 章会反复回到这一点。

### 1.4 抓握的静力学：抓取矩阵、内力与可操纵性 {#ch1-4}

慢速操作时可以忽略惯性（准静态）。把 $m$ 个接触的接触力拼成 $f = (f_1, \ldots, f_m)$，每个 $f_i$ 用接触坐标系表示（法向 $n_i$、两个切向 $d_i$、$e_i$），物体与手的平衡方程是：

$$G f + w_{\text{ext}} = 0, \qquad \tau = J_h(q)^\top f$$

$$G = \begin{bmatrix} G_1 & \cdots & G_m \end{bmatrix}, \qquad G_i = \begin{bmatrix} n_i & d_i & e_i \\ p_i \times n_i & p_i \times d_i & p_i \times e_i \end{bmatrix} \in \mathbb{R}^{6 \times 3}$$

$p_i$ 是接触点在物体坐标系（原点取质心）里的位置——这正是 [BODex](https://arxiv.org/abs/2412.16490) 采用的写法；$J_h$ 是手的接触雅可比，把关节速度映射到各接触点的速度。三个推论贯穿后面几章：

1. **内力。** $G$ 的零空间里的接触力不改变物体受到的合力旋量，只"挤压"物体：$f = f_p + f_N$，$f_p$ 平衡外力，$f_N \in N(G)$ 是内力。抓得稳，本质上就是用足够的内力把每个接触力推到摩擦锥内部，留出抗扰动、抗滑移的余量；内力太大又会压坏物体或让电机过热。
2. **可操纵性。** 接触不滑动时，接触点速度必须和物体一起运动：$J_h \dot{q} = G^{\top} \nu_o$，$\nu_o$ 是物体的速度旋量。物体能被手指"操纵"（而不仅仅是被夹住）的方向，是 $G^{\top}$ 的值域中能被 $J_h$ 生成的那部分；若要任意方向都能动，需要 $\operatorname{range}(G^{\top}) \subseteq \operatorname{range}(J_h)$。这就是为什么指尖抓（可操纵性好）适合手内操作，而包络抓（接触多、$J_h$ 受限）适合稳稳拿住。
3. **接触力由关节力矩间接决定。** $\tau = J_h^{\top} f$ 说明：指令力矩——或者位置控制下的"位置误差 × 增益"——决定了接触力的大小，这就是第 2 章"PD 增益是抓握力旋钮"的来源。$J_h$ 接近奇异时（比如手指伸直），某些方向的接触力几乎不需要力矩就能承受，但也几乎无法主动调节。

### 1.5 和腿足对照 {#ch1-5}

!!! analogy "运控类比"
    手内操作的"指步态"（finger gaiting）对应腿足的步态：一部分手指保持接触、稳住物体（支撑相），另一部分抬起换位（摆动相）。不同之处在于：
    腿足的支撑由重力"压住"，而掌心朝下时，物体要靠摩擦"夹住"，每一步都可能掉；地面是固定的，物体却是被控对象本身，换指时它会动；接触点在物体表面滚动和滑动，接触几何随物体形状变化，而物体的形状、质量、摩擦每次都不一样。

所以灵巧手的"泛化"主要是对物体的泛化，而不是对地形的泛化。这也解释了为什么第 3 章的特权信息是物体属性（质量、尺寸、摩擦、质心），而不是地面属性。

### 1.6 任务谱系与主流方法 {#ch1-6}

| 任务 | 例子 | 主流方法 | 章节 |
|---|---|---|---|
| 抓取 | 杂乱场景抓取、功能抓取 | 抓取合成 + 闭环 RL / 扩散策略 | 5 |
| 手内重定向与旋转 | 转方块、转笔、魔方 | 仿真 RL + sim2real | 3 |
| 工具与铰接物体 | 剪刀、镊子、螺丝刀 | 仿真 RL；人类演示 + 跟踪 RL | 3、4 |
| 双手与长时程 | 开盒、拧瓶盖、做家务 | 跟踪 RL、模仿学习 / VLA、真机后训练 | 4、8、9 |
| 动态 | 抛接、接住空中物体 | 多智能体 RL | 3 |
| 接触精细的操作 | 插孔、拾纸 | 触觉 + RL 或 MPC | 6、7 |

### 1.7 自测 {#ch1-7}

<details class="quiz"><summary><span class="qtag">自测</span>为什么"掌心朝下"的手内重定向比"掌心朝上"难得多？</summary><p>掌心朝上时，重力把物体压在掌心上，策略犯了错物体多半还在手里；掌心朝下时，物体完全靠指尖的接触力托住，每一时刻接触力都必须落在摩擦锥内并平衡重力，换指时的余量更小。好比从"在地面上走"变成了"只靠手抓着攀爬"。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>手的动力学方程里 $-\sum J_i^{\top} f_i$ 这一项告诉了我们什么？</summary><p>接触力会反作用到手上：关节力矩里有一部分在"维持"接触力。这是用电机电流或关节力矩间接感知接触的依据（第 7 章的"电流当触觉"）；它也说明位置控制下的抓握力取决于 PD 增益和目标位置——同样的位置误差，增益不同，接触力就不同（第 2 章）。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>两根手指对捏一个方块，加大夹持力会改变方块受到的合力吗？它改变了什么？</summary><p>不会。两指对捏时，大小相等、方向相反的一对法向力正好落在 $G$ 的零空间里，是纯内力，合力旋量为零。它改变的是每个接触力在摩擦锥里的位置：法向分量变大，允许的切向力（$\mu f_n$）也变大，于是能抵抗更大的外扰与重力分量而不打滑。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>为什么同一个 RL 策略在 MuJoCo 和 Isaac Gym 里的表现可能差很多？</summary><p>两者解接触的方式不同：MuJoCo 是软约束的凸优化，接触软硬由 solref/solimp 决定；PhysX 是迭代求解，精度取决于迭代次数和子步数。穿透量、摩擦锥形状（椭圆锥还是多面锥）、扭转摩擦是否建模都会不同，而灵巧策略恰恰对这些细节敏感。这也是 sim2sim（换一个仿真器验证）常被当作 sim2real 前的一道检查的原因。</p></details>

## 2. 被控对象：硬件、仿真与执行器 {#ch2}

做腿足的人大概都体会过"仿真模型建不对，策略就白训"。灵巧手更敏感：力和尺寸都小，摩擦、回差、执行器的非理想特性在总受力里占比更大。

### 2.1 三种驱动方式 {#ch2-1}

| 驱动 | 代表 | 对控制意味着什么 |
|---|---|---|
| 关节直驱 / 近端电机 | Allegro、[ LEAP](https://arxiv.org/abs/2309.06440)、XHand、[ Sharpa Wave](https://www.cnx-software.com/2026/06/02/sharpa-wave-high-end-dexterous-robotic-hand-with-22-dof-high-sensitivity-dynamic-tactile-array/)、[ Wuji](https://wuji.tech/en/hand) | 可反驱，能从电流估力矩，建模直接；手更重，电机易发热 |
| 腱驱动 | Shadow、[ ORCA](https://arxiv.org/abs/2504.04259)、[ MM-Hand](https://arxiv.org/abs/2604.17245) | 手轻、自由度高；腱的摩擦、迟滞和耦合难建模 |
| 连杆驱动 | Inspire、[ DexLink](https://arxiv.org/abs/2606.17418) | 可靠、便宜；耦合关节限制可达姿态，仿真里必须把耦合建出来 |

具体型号与参数见调研页的硬件表。选手时，运控研究者最该看的不是自由度数字，而是：能否反驱、控制带宽、能否从电流估力矩、触觉布局、能否自己维修，以及有没有可靠的仿真资产。

### 2.2 关节层控制：位置目标 + PD {#ch2-2}

绝大多数灵巧策略输出的是关节位置目标，由低层 PD 转成力矩：

$$\tau = K_p\,(q^* - q) - K_d\,\dot q$$

这一层有三件容易被忽视的事：

1. **PD 增益就是"抓握力旋钮"。** 手指被物体挡住时，稳态力矩约为 $K_p(q^* - q)$。所以把目标设在物体"里面"一点就能产生夹持力。[DexGraspNet](https://arxiv.org/abs/2210.02697) 在仿真里验证抓取时正是这样施力的：把每个接触连杆沿接触法向往里挪一点，作为位置目标。
2. **频率分层。** 以 [HORA](https://arxiv.org/abs/2210.04887) 为例：真机上策略以 20 Hz 输出目标，PD 以 300 Hz 运行（Allegro，$K_p = 3.0$、$K_d = 0.1$）；仿真步频 120 Hz、控制 20 Hz。策略不必快，但 PD 要快，而且仿真和真机要一致。
3. **动作参数化。** 常见的是"相对目标 + 平滑"：$q^*_t = q^*_{t-1} + \Delta a_t$，或者对策略的原始输出做指数滑动平均。好处是动作有界、平滑，真机不抖，电机也不容易烫。

!!! analogy "运控类比"
    这和腿足 RL 里"策略几十赫兹输出关节目标、PD 高频跟踪"完全同构。不同之处在于灵巧手的 $K_p$ 小得多，而接触力对它很敏感：增益偏差会直接改变抓握力。所以 $K_p$、$K_d$ 往往也要做辨识或随机化。

### 2.3 执行器模型与系统辨识 {#ch2-3}

**从理想 PD 到真实执行器。** 理想 PD 假设电机能瞬间给出任意力矩。真实关节至少还要叠加几件事：

$$\tau_{\text{joint}} = \operatorname{sat}_{\tau_{\max}(\dot q)}\!\bigl[ K_p (q^* - q) - K_d\, \dot q \bigr]  - \tau_c \operatorname{sign}(\dot q) - b\, \dot q, \qquad (I_{\text{link}} + I_{\text{arm}})\, \ddot q = \tau_{\text{joint}} + \tau_{\text{ext}}$$

- **力矩饱和与力矩-转速曲线。** 直流电机的可用力矩随转速近似线性下降：$\tau_{\max}(\dot{q}) \approx \tau_{\mathrm{stall}}(1 - \lvert \dot{q}\rvert / \omega_{\mathrm{free}})$（折算到关节侧要乘减速比和效率）。快速换指恰好会撞上这条曲线。
- **库仑摩擦 $\tau_c$ 与粘滞摩擦 $b$。** 小力矩下静摩擦占比很大，形成死区：给了目标，手指却不动。
- **转子惯量（armature）$I_{\mathrm{arm}}$。** 经减速比 $N$ 放大 $N^{2}$ 倍，常常比手指连杆本身的惯量还大；不建它，仿真里的手指"太轻、太快"。
- **回差与延迟。** 减速器回差，以及通信、驱动器带来的延迟。

腱驱动还有一个特有的非线性：腱绕过滑轮或导管时，摩擦按绞盘方程随包角指数增长，拉动方向上传到远端的张力约为 $T_{\mathrm{out}} \approx T_{\mathrm{in}} e^{-\mu \theta}$（$\theta$ 为总包角）。同一个电机力矩传到指尖的大小，取决于手指当前的姿态和运动方向，于是出现明显的迟滞——这是腱驱动手 sim2real 难的根源之一。

**隐式 PD 与显式 PD。** 很多 GPU 引擎把 PD 写进物理求解器（隐式），刚度再大也稳定；在策略代码里自己算力矩再施加（显式）则受步长限制，刚度高时要缩小步长。同样的 $K_p$、$K_d$ 下，两者的行为也不完全一样——换仿真器或从隐式改成显式时，增益要重新标定。

**系统辨识：一个最小可行流程。**

1. **激励。** 手指悬空、不碰物体，对每个关节发阶跃、正弦扫频（chirp）和随机目标；记录 $q$、指令 $q^*$、电机电流，采样率至少是控制频率的几倍。
2. **拟合。** 把上面的模型写成对未知参数线性的形式，逐关节最小二乘。如果 PD 跑在驱动器固件里、增益不可知，就改为辨识"指令 → 响应"的传递特性（带宽、延迟、稳态误差）。
3. **验证。** 用一段没参与拟合的轨迹，在仿真里回放同样的指令序列，对比关节轨迹的跟踪误差、相位延迟和稳态误差。
4. **写回。** 把辨识出的惯量、摩擦、增益、延迟写进仿真，并以辨识值为中心做随机化（第 3 章）。

```python
import numpy as np

def identify_joint(q, qd, qdd, q_cmd, kp, kd):
    """单关节最小二乘辨识（悬空激励数据）。模型：
         kp*(q_cmd - q) - kd*qd = I*qdd + b*qd + tc*sign(qd)
    未知量 [I, b, tc] 线性出现。残差 RMS 大，说明模型还缺项（饱和、回差、延迟）。"""
    tau_pd = kp * (q_cmd - q) - kd * qd
    moving = np.abs(qd) > 1e-3                       # 静止段里静摩擦不可辨识，先剔除
    A = np.stack([qdd, qd, np.sign(qd)], axis=1)[moving]
    y = tau_pd[moving]
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ theta
    return {"inertia": theta[0], "viscous": theta[1], "coulomb": theta[2],
            "rms": float(np.sqrt(np.mean(resid ** 2)))}

def identify_delay(q_cmd, q, dt, max_lag=0.1):
    """用指令与响应的互相关粗估纯延迟（秒）。"""
    a, b = q_cmd - q_cmd.mean(), q - q.mean()
    lags = np.arange(0, int(max_lag / dt))
    corr = [np.dot(a[:len(a) - k], b[k:]) for k in lags]
    return lags[int(np.argmax(corr))] * dt
```

$\ddot{q}$ 通常由编码器数据二次差分得到，噪声很大，要先做零相位低通滤波（如前后向巴特沃斯滤波）再求导。

**执行器网络。** 解析模型不够用时可以直接学：用真机数据训练一个网络，输入最近几帧的位置误差 $(q^* - q)$ 和速度，输出实际力矩（或下一时刻的关节状态），再把它嵌进仿真、替代理想 PD——这是腿足 sim2real 里成熟的做法。灵巧手上，DexNDM 的关节级动力学模型可以看成这一思路的推广（2.5 节）。

### 2.4 把仿真模型建对：一张检查表 {#ch2-4}

1. **资产。** 优先用维护良好的模型，例如 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（含 Shadow、Allegro、LEAP 等）。不同的 URDF 解析器、仿真器和驱动可能用不同的关节顺序，对接时要按关节名映射，不要按下标——[dex-retargeting](https://github.com/dexsuite/dex-retargeting) 的文档专门提醒了这一点。
2. **运动学与耦合。** 关节限位按真机实测；连杆手、腱驱动手的耦合关节要建出来（MuJoCo 里可以用 equality 约束或 tendon），否则策略会用到真机上不存在的自由度。
3. **碰撞几何与摩擦。** 指垫的形状、厚度和软硬直接决定接触点和摩擦。凸分解太粗，就会出现"悬空接触"。MuJoCo 的 condim 决定接触摩擦的维度：3 是切向滑动摩擦，4 加上扭转摩擦，6 再加滚动摩擦；转动类任务往往对扭转摩擦敏感。
4. **执行器参数。** 力矩上限、速度上限、转子惯量（MuJoCo 的 armature）、关节阻尼和静摩擦（frictionloss）。Isaac Lab 提供了多种执行器模型，包括理想 PD、带力矩-转速饱和的直流电机模型和执行器网络（[Isaac Lab](https://arxiv.org/abs/2511.04831)）。
5. **时序。** 控制周期、观测延迟、动作延迟要和真机一致，至少要落在随机化范围里。
6. **物体。** 质量、质心、惯量、摩擦通常不辨识，而是用域随机化覆盖（第 3 章）。

### 2.5 执行器差距：比腿足更难的地方 {#ch2-5}

小力矩下，静摩擦、回差、腱的迟滞占比大，"给了目标却不动"和"动过头"都很常见。目前有几类做法：

- **系统辨识。** [REGRIND](https://arxiv.org/abs/2607.11874) 把"仔细的系统辨识"列为零样本上真机的关键之一。
- **学习关节级动力学。** [DexNDM](https://arxiv.org/abs/2510.08556) 用少量、自主采集的真机数据，为每个关节拟合等效动力学：把系统级影响压缩成低维变量，每个关节只根据自己的动态历史演化，再据此修正仿真策略的动作。按关节分解让它数据高效，并能跨不同的整手交互分布泛化。
- **电流-力矩标定 + 执行器动力学建模。** [Force-Based Sim2Real](https://arxiv.org/abs/2601.02778) 用标定代替力矩传感器，做力控层面的 sim2real。
- **可微 real-to-sim。** [D-REX](https://arxiv.org/abs/2603.01151) 用基于高斯泼溅的可微引擎辨识物理参数。
- **历史条件的策略**（第 3 章的适应模块）能吸收一部分差距，但吸收不了系统性的偏差，比如 PD 增益错了一倍。

!!! pitfall "常见坑"
    - 仿真里的 $K_p$ 比真机大很多：仿真里抓得牢，真机上打滑。
    - 只在一种控制频率下训练：真机通信一抖，策略就失稳。
    - 忽略电机发热：长时间测试后性能下降，容易被误判成策略问题。
    - 关节零位没标好：几度的偏差在指尖会放大成毫米级误差，精细抓取直接失败。

### 2.6 自测 {#ch2-6}

<details class="quiz"><summary><span class="qtag">自测</span>仿真里训好的手内旋转策略到真机上"能转但很慢、经常卡住"，最先查什么？</summary><p>先查执行器和接触参数，再怀疑策略。① 用同一段动作序列分别在仿真和真机上回放，对比关节轨迹的跟踪误差、延迟和死区；② 核对 PD 增益和力矩上限；③ 检查指垫的摩擦和碰撞几何。卡住多半是静摩擦或回差没建模，或者仿真里摩擦偏高，让策略依赖了"粘住"的接触。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>为什么 Allegro、LEAP 这类手通常用"位置目标 + PD"，而不是让策略直接输出力矩？</summary><p>低层 PD 高频运行，能稳定地处理接触冲击和通信延迟，策略只需低频给目标；较小的 $K_p$ 还带来天然的柔顺性。直接输出力矩要求策略高频运行、执行器模型准确，sim2real 难得多。代价是接触力只能经由"目标位置 + 增益"间接控制，需要力控时要另做标定（2.3、2.5 节）。</p></details>

## 3. 在仿真里学控制，再迁移到真机 {#ch3}

这一章解决灵巧手 RL 的三个核心问题，它们也是这条路线成败的关键：

1. **学得动**：接触丰富、16 维以上的连续控制，RL 怎么才能学出有效的指步态？——奖励结构、初始状态分布、动作参数化、大规模并行 PPO。
2. **看不全**：物体的质量、摩擦、形状不可观测，策略怎么在线"感知"它们？——部分可观测与教师-学生蒸馏。
3. **对不上**：仿真和真机不一样，怎么让策略对差距不敏感？——把差距拆成可随机化的项，再用 ADR 自动定范围。

贯穿全章的两个例子：[HORA](https://arxiv.org/abs/2210.04887)（只用本体感觉的手内旋转，RMA 式两阶段）和 [DeXtreme](https://arxiv.org/abs/2210.13702)（视觉方块重定向，非对称 actor-critic + LSTM + ADR）。

### 3.1 问题的形式化 {#ch3-1}

把物体与动力学的未知参数记作 $\xi$（质量、摩擦、尺寸、执行器参数……）。真机上策略只能拿到观测 $o_t$，看不到 $\xi$，因此这是一个**带未知参数的部分可观测问题**。训练时，我们在仿真里对 $\xi$ 的一个分布 $P(\xi)$ 优化：

$$\max_{\pi}\; \mathbb{E}_{\xi \sim P(\xi)}\, \mathbb{E}_{\tau \sim (\pi, \xi)} \Bigl[ \sum_t \gamma^t r_t \Bigr]$$

按策略能利用多少信息，解法分三类：

| 类型 | 策略形式 | 特点 |
|---|---|---|
| 鲁棒策略 | $\pi(a_t \mid o_t)$，无记忆 | 对 $\xi$ 取平均意义下的最优，必然偏保守；$P(\xi)$ 越宽越保守 |
| 自适应策略 | $\pi(a_t \mid o_{1:t})$，有记忆 | 从交互历史里隐式推断 $\xi$，近似"贝叶斯自适应"控制；HORA 和 DeXtreme 都属于这一类 |
| 显式辨识 | $\hat{\xi} = g(o_{1:t})$，再 $\pi(a_t \mid o_t, \hat{\xi})$ | 可解释，但很多参数从历史里根本不可辨识，硬去估反而引入噪声 |

为什么必须在仿真里学？HORA 的策略训练用了约 5 亿个控制步，相当于真实时间约 7000 小时。这种样本量只有并行仿真给得起——代价就是第 3 个问题：仿真不等于真机。

### 3.2 让 RL 学得动：奖励、初始状态与动作 {#ch3-2}

**奖励 = 有界的任务项 + 塑造步态的正则项。** HORA 的完整奖励（转轴 $\hat{k} = [0, 0, 1]$）：

$$r = \operatorname{clip}(\omega \cdot \hat k,\, -0.5,\, 0.5)  - 0.3\, \lVert q - q_{\text{init}} \rVert^2 - 0.3\, \lVert v_o \rVert^2  - 2.0\, \tau^\top \dot q - 0.1\, \lVert \tau \rVert^2$$

论文用消融说明了每一项的作用，这几条结论具有普遍性：

- **任务项必须有界。** 不截断角速度，策略会尽可能快地转、牺牲稳定性——在仿真里表现为高回报，到真机上就是把物体甩飞。
- **步态来自正则，而不是来自显式的步态奖励。** HORA 没有写任何"至少三指接触"之类的启发式。去掉姿态惩罚后，步态变得不自然，策略也学不会"断开接触、再建立新接触"；能耗与物体线速度惩罚则让动作更省力、更平稳，sim2real 也更好。
- **任务项的形状取决于任务类型。** 旋转是"保持一个速度"，所以用截断的角速度；重定向是"到达一个目标"，DeXtreme 用 $1/(d + 0.1)$ 的塑形（$d$ 为与目标姿态的旋转距离），配合 $d < 0.1$ 时的大额到达奖励，在接近目标时给出很陡的梯度。

**初始状态分布是最便宜的探索手段。** HORA 每回合都从一个**预先生成的稳定精确抓取**开始（在典型抓取附近随机扰动关节、仿真一小段时间、筛掉不稳定的），物体掉出指尖就终止。RL 因此完全不用自己去发现"怎么抓住"，只需学"抓住之后怎么转"。同样的思路在第 4 章表现为"从参考轨迹上的状态初始化"。

**动作：相对目标 + 低通。** 策略输出关节 PD 目标，常用增量形式，并对输出做指数滑动平均。DeXtreme 训练中把平滑系数从 0.2 退火到 0.15，真机上用 0.1 效果最好——它在灵活与平稳之间取舍：太平滑会跟不上快速换指，太灵活会让真机抖动、电机过热，甚至烧坏线缆。

### 3.3 大规模并行 PPO：真正影响结果的几件事 {#ch3-3}

PPO 最大化截断的替代目标，$\rho_t = \pi_{\theta}(a_t\mid s_t) / \pi_{\mathrm{old}}(a_t\mid s_t)$：

$$\begin{gathered} L^{\text{clip}}(\theta) = \mathbb{E}\Bigl[ \min\bigl( \rho_t \hat A_t,\; \operatorname{clip}(\rho_t, 1 - \varepsilon, 1 + \varepsilon)\, \hat A_t \bigr) \Bigr] \\ \hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \end{gathered}$$

在 GPU 并行仿真里，标准做法是**很多环境、每次只滚很短**：HORA 每轮用 16384 个环境、每个环境 8 步（0.4 秒），一轮约 13 万个样本，训练约 5000 轮。每环境步数短，意味着优势估计主要依靠价值函数自举，所以价值函数的质量格外重要。由此引出三个真正影响结果的细节：

1. **超时不是终止。** 回合因时间上限结束时，状态并没有"坏掉"，计算 $\delta$ 要用 $V(s_T)$ 自举；只有物体掉落这类真正的终止才把下一状态的价值置零。混为一谈，价值函数会学到"快到时间就不值钱"，策略在回合末变得鲁莽。在"短滚动、多截断"的并行设置下，这个错误的影响被放大。
2. **观测归一化的统计量要冻结并随策略一起部署。** 训练时用运行均值和方差归一化；部署时用错统计量，等于给策略换了一套输入。
3. **循环策略要配更长的有效时域。** DeXtreme 指出，用 LSTM 时把折扣从 MLP 常用的 0.99 提到 0.998，是训得动的关键。

```python
import torch

def gae(rew, val, next_val, done, timeout, gamma=0.99, lam=0.95):
    """rew, val, done, timeout: (T, N)；next_val: (T, N) 为 V(s_{t+1})。
    真终止（物体掉落）不自举；超时截断保留自举；两种情况都切断跨回合的累积。"""
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    for t in reversed(range(rew.shape[0])):
        terminal = done[t] & ~timeout[t]
        delta = rew[t] + gamma * next_val[t] * (~terminal) - val[t]
        last = delta + gamma * lam * (~done[t]) * last
        adv[t] = last
    return adv, adv + val            # 优势，以及价值函数的回归目标
```

### 3.4 部分可观测：从历史里推断物体（教师-学生） {#ch3-4}

<figure class="fig"><div class="fig-scroll"><svg viewBox="0 0 760 300" role="img" aria-label="RMA 式教师学生两阶段示意图"><defs><marker id="ah2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="sv-head"/></marker></defs><rect class="sv-zone" x="10" y="8" width="240" height="284" rx="12"/><rect class="sv-zone" x="260" y="8" width="240" height="284" rx="12"/><rect class="sv-zone" x="510" y="8" width="240" height="284" rx="12"/><text class="sv-t1" x="130" y="32" text-anchor="middle">阶段一：教师（仿真 · PPO）</text><text class="sv-t1" x="380" y="32" text-anchor="middle">阶段二：学生（仿真 · 监督）</text><text class="sv-t1" x="630" y="32" text-anchor="middle">部署（真机）</text><rect class="sv-box" x="28" y="48" width="204" height="48" rx="9"/><text class="sv-t2" x="130" y="68" text-anchor="middle">特权信息 e（仅仿真）</text><text class="sv-t3" x="130" y="86" text-anchor="middle">位置 · 尺寸 · 质量 · 摩擦 · 质心</text><rect class="sv-box" x="28" y="118" width="204" height="48" rx="9"/><text class="sv-t2" x="130" y="138" text-anchor="middle">编码器 μ</text><text class="sv-t3" x="130" y="156" text-anchor="middle">z = μ(e) ∈ ℝ⁸</text><rect class="sv-hl" x="28" y="188" width="204" height="48" rx="9"/><text class="sv-t2" x="130" y="208" text-anchor="middle">策略 π(o, z)</text><text class="sv-t3" x="130" y="226" text-anchor="middle">o：3 帧关节角 + 3 帧动作</text><text class="sv-t3" x="130" y="276" text-anchor="middle">输出 a：PD 目标（20 Hz）</text><path class="sv-edge" d="M130,96 L130,116" marker-end="url(#ah2)"/><path class="sv-edge" d="M130,166 L130,186" marker-end="url(#ah2)"/><path class="sv-edge" d="M130,236 L130,262" marker-end="url(#ah2)"/><rect class="sv-box" x="278" y="48" width="204" height="48" rx="9"/><text class="sv-t2" x="380" y="68" text-anchor="middle">历史窗口</text><text class="sv-t3" x="380" y="86" text-anchor="middle">关节角历史 + 已执行的动作历史</text><rect class="sv-hl" x="278" y="118" width="204" height="48" rx="9"/><text class="sv-t2" x="380" y="138" text-anchor="middle">适应模块 φ</text><text class="sv-t3" x="380" y="156" text-anchor="middle">输出 ẑ，用 ℓ2 回归到 z</text><rect class="sv-box" x="278" y="188" width="204" height="48" rx="9"/><text class="sv-t2" x="380" y="208" text-anchor="middle">π 冻结，用 ẑ 执行</text><text class="sv-t3" x="380" y="226" text-anchor="middle">数据落在学生自己的状态分布上</text><text class="sv-t3" x="380" y="276" text-anchor="middle">反复采集、回归，直到收敛</text><path class="sv-edge" d="M380,96 L380,116" marker-end="url(#ah2)"/><path class="sv-edge" d="M380,166 L380,186" marker-end="url(#ah2)"/><path class="sv-edge" d="M380,236 L380,262" marker-end="url(#ah2)"/><rect class="sv-box" x="528" y="48" width="204" height="48" rx="9"/><text class="sv-t2" x="630" y="68" text-anchor="middle">真机历史</text><text class="sv-t3" x="630" y="86" text-anchor="middle">关节编码器 + 已发出的动作</text><rect class="sv-box" x="528" y="118" width="204" height="48" rx="9"/><text class="sv-t2" x="630" y="138" text-anchor="middle">φ 在线估计 ẑ</text><text class="sv-t3" x="630" y="156" text-anchor="middle">换物体时 ẑ 随之变化</text><rect class="sv-hl" x="528" y="188" width="204" height="48" rx="9"/><text class="sv-t2" x="630" y="208" text-anchor="middle">π(o, ẑ)</text><text class="sv-t3" x="630" y="226" text-anchor="middle">PD 以 300 Hz 执行</text><text class="sv-t3" x="630" y="276" text-anchor="middle">不需要知道任何物体参数</text><path class="sv-edge" d="M630,96 L630,116" marker-end="url(#ah2)"/><path class="sv-edge" d="M630,166 L630,186" marker-end="url(#ah2)"/><path class="sv-edge" d="M630,236 L630,262" marker-end="url(#ah2)"/></svg></div><figcaption>图 2　RMA 式教师-学生两阶段（以 HORA 为例）。教师在仿真里借助特权信息训练；学生只看真机也拿得到的历史，学会估计教师所用的隐变量。</figcaption></figure>

**两阶段做法（HORA）。**

- **阶段一（教师）**：特权信息 $e$（物体位置、尺寸、质量、摩擦、质心，共 9 维）经编码器 $\mu$ 压成 8 维隐变量 $z = \mu(e)$；策略 $\pi(o_t, z_t)$ 与 $\mu$ 一起用 PPO 训练。$o_t$ 只含最近 3 帧关节角和动作。
- **阶段二（学生）**：冻结 $\pi$ 和 $\mu$，训练适应模块 $\phi$ 从最近 30 步（1.5 秒）的关节角与动作历史估计 $\hat z_t$（结构是逐步编码后接时间卷积），损失 $\lVert \hat z_t - z_t\rVert^{2}$。关键在于数据由**用 $\hat z$ 执行的策略**采集。
- **部署**：$\phi$ 在线估计 $\hat z$，$\pi$ 照常执行，不需要知道任何物体参数。

```python
def train_adaptation(env, policy, mu, phi, opt, iters, hist=30):
    """阶段二：policy 与 mu 冻结；学生用自己的 ẑ 执行，回归教师的 z。"""
    for _ in range(iters):
        hs, zs = [], []
        obs, e = env.reset()
        for _ in range(env.horizon):
            h = env.history(hist)                       # (N, hist, 2n) 关节角与动作
            with torch.no_grad():
                obs, e, done = env.step(policy(obs, phi(h)))   # 用 ẑ 执行
                zs.append(mu(e))                                # 真值隐变量
            hs.append(h)
        loss = ((phi(torch.cat(hs)) - torch.cat(zs)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
```

这个设计里有三个值得记住的结论，都来自 HORA 的消融：

1. **为什么非要两阶段？** 附录专门试了"直接用 RL 训练一个看长历史的鲁棒策略"：历史从 3 步加到 30 步，PPO 训练的 MLP 策略性能单调下降（分布内平均旋转奖励从约 176 掉到约 50），LSTM 有同样的优化困难，时间卷积网络甚至训不出来。原因是回报信号太弱，不足以同时学会"推断物体"和"控制"。两阶段把推断变成有直接监督的回归，RL 只需在 8 维的 $z$ 上学控制。
2. **回归隐变量，而不是物理参数。** 让适应模块直接回归 9 维参数 $e$（SysID 基线），转得更快，稳定性却明显变差。$e$ 里有些分量从本体历史中不可辨识；$z$ 是由控制目标"挑选"过的表示，只保留与控制相关、且可推断的部分。
3. **学生必须在自己的状态分布上学。** 这就是 DAgger 的论点：若学生只在教师的轨迹上训练（行为克隆），每步 $\varepsilon$ 的误差会把它带到没见过的状态，误差随时域 $T$ 以 $O(\varepsilon T^{2})$ 累积；让学生执行、教师（或真值 $z$）在学生到达的状态上标注，误差界降为 $O(\varepsilon T)$。HORA 的阶段二就是 DAgger 在隐变量上的版本。

**另外两种结构。**

- **直接蒸馏动作**：$\lVert \pi_s(o^{\mathrm{hist}}) - \pi_t(o^{\mathrm{priv}})\rVert^{2}$。学生与教师的输入差异很大时（学生只看 RGB 或深度），回归隐变量很难对齐，直接蒸馏动作更稳妥——[Visual Dexterity](https://arxiv.org/abs/2211.11744)、[DextrAH-RGB](https://arxiv.org/abs/2412.01791)、[ViserDex](https://arxiv.org/abs/2604.11138) 都属于这一类。
- **非对称 actor-critic**：只让 critic 看特权状态，actor 只用真机可得的观测，省掉蒸馏阶段。DeXtreme 的 actor 输入 50 维（带噪的物体与目标位姿、上一步动作、关节角），critic 输入 265 维（另含指尖力、关节速度、物体质量与摩擦、所有随机化参数等）。要注意的是，部分可观测时只以状态为输入的 critic 会给策略梯度带来偏差，因为策略实际依赖的是历史；DeXtreme 的 critic 本身也是 LSTM，看的是观测序列。

!!! analogy "运控类比"
    这就是 RMA，只是"环境属性"从地形换成了物体。HORA 还观察到，策略的平均指令力矩几乎随物体质量线性变化：它确实在按估出的物体属性调整用力，而不是用一套固定的大力矩"保险"。

### 3.5 sim2real：把差距拆成可随机化的项 {#ch3-5}

域随机化就是 3.1 节目标里的 $P(\xi)$：只要真机落在 $P(\xi)$ 的支撑集里，而策略又能（借助记忆）适应其中每一个 $\xi$，迁移就能成功。所以 sim2real 的工程问题可以表述为：**把仿真和真机的每一类差距，都写成一个可随机化的参数。**

| 差距来源 | 随机化什么 | 例子 |
|---|---|---|
| 物体与接触 | 质量、尺寸、质心、摩擦、恢复系数 | 两个系统都有 |
| 执行器 | PD 增益、阻尼、力矩上限、转子惯量 | DeXtreme 让 ADR 自己找范围，力矩上限缩放最终到 [0.9, 2.49] |
| 时延 | 观测与动作的随机延迟、固定延迟、传感器刷新率 | DeXtreme 的三种延迟模型 |
| 传感噪声 | 相关噪声（每回合固定的偏差）与不相关噪声（每步白噪声） | 关节角、物体位姿 |
| 感知故障 | 位姿估计的偶发跳变 | DeXtreme 的随机位姿注入 |
| 未建模动力学 | 随状态变化的结构化扰动 | DeXtreme 的随机网络对抗 |

DeXtreme 在非物理随机化上的几个设计，单独看都很简单，却是它真机表现的关键：

- **噪声的下限必须能到零。** 噪声写成 $x + \delta + \varepsilon$（$\delta$ 每回合采样，$\varepsilon$ 每步采样），方差由 ADR 变量 $p$ 经 $\operatorname{var}(p) = \exp(p^{2}) - 1$ 给出，$p = 0$ 时方差恰为 0。理由很实际：部署时并没有人为加的白噪声，"零噪声"这个情形必须被训练覆盖。
- **把延迟当作一等公民。** 随机延迟 $x_{\mathrm{obs}} = x_{\mathrm{last}} \cdot d + x \cdot (1 - d)$（$d \sim \operatorname{Bernoulli}(p)$）模拟时延抖动；固定动作延迟模拟通信与驱动的滞后；按随机周期刷新物体位姿，模拟相机与位姿估计器的帧率。
- **随机位姿注入。** 遮挡下位姿估计偶尔会跳变，可能"污染"LSTM 的隐状态。训练时每步以一定概率把观测到的物体位姿换成完全随机的位姿，策略就学会了不被单帧离群值带偏。
- **随机网络对抗（RNA）。** 每回合用一个随机初始化的网络产生随状态变化的动作扰动：$a = \alpha \cdot a_{\mathrm{RNA}} + (1 - \alpha) \cdot a_{\mathrm{policy}}$。高斯噪声与状态无关，模拟不了摩擦、回差、迟滞这类"随状态系统性变化"的误差；随机网络可以。

**自动域随机化（ADR）把"范围定多宽"也交给算法。** 范围太窄覆盖不了真机，太宽早期策略学不动。ADR 从窄范围开始：一部分环境专门做评估，把某一维钉在当前边界上，记录这些回合的表现（DeXtreme 用连续成功次数，队列长度 256）；平均表现高于上阈值（20）就把这条边界往外扩，低于下阈值（5）就往里收。结果是一门自动调节难度的课程，并且最终告诉你策略能覆盖多大的范围——DeXtreme 中物体摩擦的范围从 [0.3, 0.9] 被扩到了 [0.01, 1.60]。

```python
import numpy as np
from collections import deque

class ADRBoundary:
    """一个随机化维度的一条边界。阈值与队列长度取自 DeXtreme，步长为示意。"""
    def __init__(self, value, step, outward, n=256, t_hi=20.0, t_lo=5.0):
        self.value, self.step, self.outward = value, step, outward   # outward: +1 上界，-1 下界
        self.q, self.t_hi, self.t_lo = deque(maxlen=n), t_hi, t_lo

    def report(self, score):
        """评估环境把这一维钉在本边界、跑完一回合后调用。"""
        self.q.append(score)
        if len(self.q) == self.q.maxlen:
            m = float(np.mean(self.q))
            if m > self.t_hi:
                self.value += self.outward * self.step   # 够简单了：往外扩
            elif m < self.t_lo:
                self.value -= self.outward * self.step   # 太难了：往里收
            else:
                return
            self.q.clear()                               # 范围变了，重新累计
```

**这些技巧各值多少？** DeXtreme 附录记录了真机上最大连续成功次数随改进的变化，可以当作 sim2real 的优先级参考：

| 改进 | 最大连续成功次数 |
|---|---|
| 手工域随机化 | 10 |
| 物体位姿改为相对手腕表示 | 15 |
| 加大随机化范围 | 20 |
| 加大网络 | 31 |
| 随机位姿注入 + 随机网络对抗 | 43 |
| ADR | 70–77 |
| ADR + 更深的网络 | 112 |

两点启示：观测表示（相对手腕）这类"小改动"的收益不亚于算法；更强的随机化需要更大的网络来承载。

### 3.6 评测与诊断 {#ch3-6}

**指标要分开看"好不好"与"稳不稳"。** 旋转类任务常用平均角速度（HORA 的 RotR，评估时不截断）和掉落前时间 TTF（按回合长度归一化）；重定向类任务用连续成功次数（DeXtreme：姿态误差进入 0.4 rad 以内即换新目标，掉落或卡住超过 80 秒就结束）。只看速度会奖励"甩"的策略，只看 TTF 会奖励"不动"的策略。

**诊断流程。**

1. **逐类关掉随机化**，看性能掉多少，就知道策略依赖什么。HORA 关掉物理随机化后，分布外的平均旋转奖励从约 161 降到约 88。
2. **测分布外物体**（尺寸、质量、形状超出训练范围）。
3. **检查隐变量是否在"感知"。** HORA 在真机上每 30 秒换一个物体，估出的 $\hat z$ 中一维随直径变化、另一维随质量变化；$\hat z$ 不随物体变化，说明适应模块没起作用。
4. **对照几个有诊断意义的基线**：只用域随机化的鲁棒策略、回归物理参数的 SysID、只在开头估一次 $\hat z$ 的 NoAdapt、盲放一段专家动作的回放基线——后者用来证明策略在闭环调整，而不是在周期性地重复同一段动作。

### 3.7 小结 {#ch3-7}

- 奖励用"有界任务项 + 正则"的结构，步态让正则去塑造；用初始状态分布绕开探索难题。
- 并行 PPO 的关键在价值函数：正确处理超时、冻结归一化统计量、循环策略用更长的有效时域。
- 部分可观测用两阶段解决：RL 在低维隐变量上学控制，监督学习从历史推断隐变量；学生必须在自己的状态分布上学。
- sim2real 的本质，是把每一类差距写成可随机化的参数，让真机落进训练分布；ADR 自动决定分布有多宽。

!!! pitfall "常见坑"
    - 任务项无界：策略在仿真里拿高回报的方式，正是真机上会失败的方式。
    - 把超时当终止：价值函数在回合末系统性偏低。
    - 噪声下限不为零："干净"的部署情形反而没被训练覆盖。
    - 观测用绝对位姿而不是相对手腕的位姿：手的安装或标定一变，策略就失效。
    - 适应模块只用教师轨迹训练：部署时学生一偏离，估计就失准。

### 3.8 自测 {#ch3-8}

<details class="quiz"><summary><span class="qtag">自测</span>HORA 为什么不直接用 PPO 训练一个看 30 步历史的策略？</summary><p>论文附录的实验：历史从 3 步加到 30 步，PPO 训练的 MLP 与 LSTM 策略性能都单调下降，时间卷积网络甚至训不出来。回报信号太弱，不足以同时学会"推断物体属性"和"控制"。两阶段把推断变成有直接监督的回归问题，RL 只需在 8 维 $z$ 上学控制。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>为什么回归隐变量 $z$ 比回归物理参数 $e$ 更好？</summary><p>$e$ 中有些分量从本体历史里不可辨识（例如对当前动作没有影响的参数），硬去回归只能学到条件均值、引入噪声；而 $z$ 是在阶段一里与控制目标一起学出来的，只编码对控制有用的信息。HORA 的 SysID 基线转得更快、稳定性却明显更差，印证了这一点。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>DeXtreme 为什么要用随机网络对抗，而不是把高斯噪声开得更大？</summary><p>高斯噪声与状态无关、时间上不相关，只能模拟随机抖动；而未建模动力学（摩擦、回差、迟滞）产生的误差是随状态系统性变化的。随机网络把状态映射成扰动动作，正好给出结构化的、随状态变化的误差，迫使策略学会在"模型有系统性错误"时仍完成任务。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>鲁棒策略和自适应策略，各自在什么时候更合适？</summary><p>当参数变化对最优动作影响不大、或真机参数基本固定时，无记忆的鲁棒策略简单可靠；当物体属性变化大、且最优动作强烈依赖这些属性时（不同质量需要不同的握力和步态），自适应策略能从历史里推断参数，显著优于对所有参数"取平均"的鲁棒策略——HORA 相对 DR 基线的提升就来自这里。</p></details>

## 4. 从人类演示学：重定向 + 跟踪 RL {#ch4}

### 4.1 问题：奖励写不出来时，演示提供了什么 {#ch4-1}

第 3 章的配方在"目标好写、回合不长"的任务上很强。到了开盒、拧瓶盖、双手递物这类长时程任务，RL 同时遇到两个难题：奖励写不出来，探索也找不到路。人类演示 $D = \{(h_t, x_t)\}_{t=1}^{T}$（$h$ 为人手位姿，$x$ 为物体位姿）恰好补上这两点：它给出了物体应该怎么走、手大概怎么动、哪里该有接触。

把演示变成机器人能执行的策略，要解决两个子问题：

1. **本体映射**：人手与机器人手的尺寸、关节结构都不同——把 $h_t$ 映射成机器人关节角 $q_t$。这是纯运动学问题（4.2、4.3 节）。
2. **物理可行性**：运动学重定向不管物理——会穿模，没有力，也不保证物体真的跟着动。需要在物理仿真里找到一个策略，使**物体**的轨迹跟上演示（4.4 节）。

关键的设计判断是：**目标以物体为中心**。手的动作像不像人只是辅助，物体有没有按演示走才是任务本身。

!!! analogy "运控类比"
    这和人形的运动模仿是同一套：动捕 → 重定向到机器人骨架 → DeepMimic 式的跟踪 RL。[REGRIND](https://arxiv.org/abs/2607.11874) 明确说自己就是把人形全身控制的"重定向 + RL 跟踪"配方搬到了灵巧手上。区别在于：人形跟踪的是机器人自身的运动，灵巧手跟踪的主要是一个没有执行器的物体。

<figure class="fig"><div class="fig-scroll"><svg viewBox="0 0 760 228" role="img" aria-label="人类演示到跟踪 RL 的管线示意图"><defs><marker id="ah3" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="sv-head"/></marker></defs><rect class="sv-box" x="12" y="20" width="168" height="58" rx="10"/><text class="sv-t2" x="96" y="44" text-anchor="middle">人类演示</text><text class="sv-t3" x="96" y="64" text-anchor="middle">动捕 · 视频 · 数据手套</text><rect class="sv-box" x="204" y="20" width="168" height="58" rx="10"/><text class="sv-t2" x="288" y="44" text-anchor="middle">手-物位姿估计</text><text class="sv-t3" x="288" y="64" text-anchor="middle">手部关键点 + 物体 6D 位姿</text><rect class="sv-box" x="396" y="20" width="168" height="58" rx="10"/><text class="sv-t2" x="480" y="44" text-anchor="middle">运动学重定向</text><text class="sv-t3" x="480" y="64" text-anchor="middle">保关键向量 → 保接触 → 保力</text><rect class="sv-box" x="588" y="20" width="160" height="58" rx="10"/><text class="sv-t2" x="668" y="44" text-anchor="middle">仿真回放去穿模</text><text class="sv-t3" x="668" y="64" text-anchor="middle">提取接触时间与位置</text><rect class="sv-hl" x="588" y="146" width="160" height="58" rx="10"/><text class="sv-t2" x="668" y="170" text-anchor="middle">跟踪 RL</text><text class="sv-t3" x="668" y="190" text-anchor="middle">参考 + 残差 · 课程</text><rect class="sv-box" x="396" y="146" width="168" height="58" rx="10"/><text class="sv-t2" x="480" y="170" text-anchor="middle">物理可行的轨迹</text><text class="sv-t3" x="480" y="190" text-anchor="middle">关节 · 接触 · 物体状态</text><rect class="sv-box" x="204" y="146" width="168" height="58" rx="10"/><text class="sv-t2" x="288" y="170" text-anchor="middle">蒸馏 / 模仿</text><text class="sv-t3" x="288" y="190" text-anchor="middle">视觉策略或 VLA 的训练数据</text><rect class="sv-box" x="12" y="146" width="168" height="58" rx="10"/><text class="sv-t2" x="96" y="170" text-anchor="middle">真机部署</text><text class="sv-t3" x="96" y="190" text-anchor="middle">零样本或少量真机微调</text><path class="sv-edge" d="M180,49 L202,49" marker-end="url(#ah3)"/><path class="sv-edge" d="M372,49 L394,49" marker-end="url(#ah3)"/><path class="sv-edge" d="M564,49 L586,49" marker-end="url(#ah3)"/><path class="sv-edge" d="M668,78 L668,144" marker-end="url(#ah3)"/><path class="sv-edge" d="M588,175 L566,175" marker-end="url(#ah3)"/><path class="sv-edge" d="M396,175 L374,175" marker-end="url(#ah3)"/><path class="sv-edge" d="M204,175 L182,175" marker-end="url(#ah3)"/><text class="sv-t3" x="676" y="116">参考轨迹</text></svg></div><figcaption>图 3　人类演示 → 重定向 → 跟踪 RL。上半段只处理运动学，下半段在物理仿真里把动作"修"成可行的，再用于训练部署策略。</figcaption></figure>

### 4.2 本体映射：重定向是一个小型优化问题 {#ch4-2}

**目标函数。** 直接复制关节角不可行，通行的做法是匹配手上一组**关键向量**（腕到各指尖、拇指尖到其余指尖）。[AnyTeleop](https://arxiv.org/abs/2307.04577) 的形式是

$$\min_{q}\; \sum_i \lVert \alpha v_i - f_i(q) \rVert^2 + \beta\, \lVert q - q_{t-1} \rVert^2 \qquad \text{s.t.}\quad q_l \le q \le q_u$$

$v_i$ 是人手的第 $i$ 个关键向量，$f_i(q)$ 是机器人手对应向量的正运动学，$\alpha$ 补偿手的尺寸差，$\beta$ 项保证时间上平滑。

**捏取要特殊处理。** 比例缩放后，人手捏在一起的两个指尖到了机器人手上往往对不上，接触就丢了。[DexPilot](https://arxiv.org/abs/1910.03135) 的做法是让目标随距离切换：

$$C(q_h, q_a) = \frac{1}{2} \sum_i s(d_i)\, \bigl\lVert r_i(q_a) - f(d_i)\, \hat r_i(q_h) \bigr\rVert^2  + \gamma\, \lVert q_a \rVert^2$$

$d_i$ 是人手上这对点的距离、$\hat{r}_i$ 是单位方向。距离大于阈值时 $s = 1$、$f(d) = 1.6 d$，即按比例匹配；人手拇指与某指捏在一起时，权重跳到 200（或 400），目标距离换成 $10^{-4}\,\mathrm{m}$（强制机器人指尖真的碰上），另一组向量则保持至少 3 cm 的间距，避免两根手指互相干涉。**捏取时"碰上"比"像"重要**，这条原则后来演化成了下一节的保接触重定向。

**求解。** 遥操作要求每帧几毫秒内解完。最小二乘结构很适合 Gauss–Newton / Levenberg–Marquardt：把残差 $e(q) = \alpha v - f(q)$ 在当前解处线性化，$J = \partial f / \partial q$ 由运动学库解析给出，

$$\Delta q = (J^\top J + \beta I + \lambda I)^{-1} \bigl[ J^\top e(q) - \beta\, (q - q_{t-1}) \bigr], \qquad q \leftarrow \operatorname{clip}(q + \Delta q,\, q_l,\, q_u)$$

用上一帧的解热启动，通常几次迭代就收敛；DexPilot 用 SLSQP 实时求解，并对输出的关节角再做低通滤波以去掉高频抖动。开源的 [dex-retargeting](https://github.com/dexsuite/dex-retargeting)（源自 AnyTeleop）提供了面向视频与遥操作的向量重定向、面向手-物位姿数据集的位置重定向，以及 DexPilot 式的优化器。

```python
import numpy as np

def retarget_lm(v_human, fk, jac, q_prev, q_lo, q_hi, alpha=1.0, beta=1e-3, lam=1e-4, iters=5):
    """AnyTeleop 式向量重定向，用 Levenberg–Marquardt 求解（α、β、λ 为示意值）。
    v_human: (3N,) 人手关键向量拼接；fk(q)->(3N,) 机器人对应向量；jac(q)->(3N, n) 其雅可比。"""
    q = q_prev.copy()                                   # 用上一帧的解热启动
    n = q.size
    for _ in range(iters):
        e = alpha * v_human - fk(q)                     # 关键向量残差
        J = jac(q)
        H = J.T @ J + (beta + lam) * np.eye(n)
        g = J.T @ e - beta * (q - q_prev)
        q = np.clip(q + np.linalg.solve(H, g), q_lo, q_hi)
    return q
```

### 4.3 保接触与保力：重定向应该保什么 {#ch4-3}

关键向量只保证"手型像"。按要保留的东西分层，近两年的重定向方法依次往上走：

| 保留什么 | 做法 | 代表 |
|---|---|---|
| 手型（运动学） | 关键向量或指尖位置匹配 | AnyTeleop、DexPilot |
| 接触结构 | 迁移"哪个手指部位碰物体哪里、按什么顺序"，或用交互图加形变约束保持手-物相对几何 | [ Contact-Anchored Retargeting](https://arxiv.org/abs/2609.24093)、[ TopoRetarget](https://arxiv.org/abs/2606.16272) |
| 动力学可行 | 在物理仿真里采样搜索，把运动学演示修成动力学可行的轨迹 | [ SPIDER](https://arxiv.org/abs/2511.09484) |
| 接触力 | 在运动学结果上预测残差，使接触力达到期望值 | [ ReForce](https://arxiv.org/abs/2608.15560) |

进入 RL 之前还有一步值得照做的预处理：[DexMachina](https://arxiv.org/abs/2505.24853) 先在仿真里**回放**重定向后的关节角——当作软目标、同时固定物体——以消除穿模；再用基于距离的近似，从演示中提取"哪个手部连杆在什么时间、碰物体的哪个部位"。这些接触位置稍后变成接触奖励。

### 4.4 物理可行性：跟踪 RL 的三个设计维度 {#ch4-4}

把跟踪写成 MDP：观测包括手与物体的状态和未来若干步的参考，动作是关节目标，回合从参考轨迹上的某个状态开始。设计空间主要在三处：奖励的组合方式、动作参数化、让任务"先易后难"的课程。

**① 奖励：乘积还是加和。** DexMachina 的任务奖励是物体位置、姿态、铰接角三项误差的指数核的**乘积**：

$$r_{\text{task}} = \exp(-\beta_{\text{pos}} d_{\text{pos}}) \cdot \exp(-\beta_{\text{rot}} d_{\text{rot}})  \cdot \exp(-\beta_{\text{ang}} d_{\text{ang}}), \qquad d_{\text{rot}} = 2 \arccos \bigl\lvert \langle \hat q_o, q_o \rangle \bigr\rvert$$

乘积相当于逻辑"与"：任何一项很差，总奖励都接近零；对某一项误差的梯度被其余两项的取值加权，策略只能同时改善各项。加和则是"或"：策略可以牺牲一项、去刷另一项（比如位置跟得很好，姿态完全不管）。辅助项——手部关键点的模仿奖励、关节角的行为克隆奖励、与演示接触位置对齐的接触奖励——以**小权重加和**附上去：它们引导探索，但当与任务冲突时，策略可以偏离人手动作、优先保证物体轨迹。

**② 动作：参考 + 残差。** 统一的写法是 $a_t = a_t^{\mathrm{ref}} + \Delta a_{\theta}(o_t)$。残差让初始策略（$\Delta a = 0$）就已经"大致在做对的事"，RL 只需学修正；常见做法是把残差输出层初始化为零、并限制残差幅度。DexMachina 的混合方案很有代表性：6 自由度的腕部用重定向结果做基准加残差（腕部决定手在哪，错了很难救），手指则输出按关节限位归一化的绝对目标（重定向的手指角度本来就不可靠，没必要在它上面做残差）；论文发现这种约束显著提升了学习效率。[ManipTrans](https://arxiv.org/abs/2503.21860) 则分两阶段：先训练一个只模仿手部运动的通用轨迹模仿器，再针对具体任务训练残差模块处理手-物交互，交互状态里显式加入仿真给出的指尖接触力。

**③ 课程：让任务先变容易。** 长时程双手任务里，策略早期的动作几乎都会把物体弄掉、提前终止回合，策略根本看不到任务的后半段。两种代表性的"先易后难"：

- **虚拟物体控制器（DexMachina）。** 给物体加 6 个虚拟的 1 自由度关节（基座位姿）和 1 个铰接关节，用 PD 把物体沿演示轨迹"拽"着走：$F = k_p(x^{\mathrm{ref}} - x) + k_v(\dot x^{\mathrm{ref}} - \dot x)$。初期增益很高（临界阻尼），物体几乎自己跟着演示走，策略先学会"不打扰物体"的手部动作；当各项奖励的滑动平均都超过阈值时，增益按比例衰减（$k_p \le 0.01$ 时直接归零），控制权逐步交给策略。
- **放松物理（ManipTrans）。** 初期把重力设为 0、摩擦设得很高，手很容易抓牢，先学会对齐参考；随后逐步恢复重力、降低摩擦。配合从参考轨迹随机挑状态初始化，以及逐步收紧的"物体偏离过大就终止"；若动捕显示人手此刻牢牢握着物体、机器人手的接触力却为零，也提前终止——逼策略学会真正施力。

```python
import torch

def tracking_reward(d_pos, d_rot, d_ang, kp_err, q_err, con_err,
                    b=(10.0, 2.0, 5.0, 20.0, 2.0, 10.0), w_aux=0.1):
    """DexMachina 式奖励（β 与权重为示意）：任务项相乘，辅助项小权重相加。
    kp_err (N,K) 手部关键点误差平方；q_err (N,J) 关节角误差平方；con_err (N,P) 接触位置误差。"""
    r_task = torch.exp(-b[0] * d_pos) * torch.exp(-b[1] * d_rot) * torch.exp(-b[2] * d_ang)
    r_imi = torch.exp(-b[3] * kp_err).mean(-1)
    r_bc = torch.exp(-b[4] * q_err).mean(-1)
    r_con = torch.exp(-b[5] * con_err).mean(-1)
    return r_task + w_aux * (r_imi + r_bc + r_con)

def decay_virtual_controller(kp, kv, reward_means, thresholds, ratio=0.9):
    """所有奖励的滑动平均都超过阈值时，按比例衰减虚拟物体控制器的增益（比例为示意）。"""
    if kp == 0 or any(reward_means[k] <= thresholds[k] for k in thresholds):
        return kp, kv
    kp, kv = kp * ratio, kv * ratio
    return (0.0, 0.0) if kp <= 0.01 else (kp, kv)
```

!!! key "要点"
    跟踪 RL 学得动，靠的是"让任务先变容易，再逐步还原"：从参考状态初始化、逐步收紧的提前终止、虚拟物体控制器、放松物理。人形运动模仿里的参考状态初始化、提前终止、辅助力，是同一类技巧。

### 4.5 演示从哪里来 {#ch4-5}

管线的输入可以是带手-物位姿的数据集（DexMachina 用的是 [ARCTIC](https://arxiv.org/abs/2204.13662)），也可以是单段视频——此时先做手部与物体的位姿重建，再进入上面的流程。一个值得注意的变体是绕开手部重定向：[Human2Sim2Robot](https://arxiv.org/abs/2504.12609) 只用物体轨迹定义与本体无关的奖励，在仿真里用 RL 跨越本体差距；[HuDOR](https://arxiv.org/abs/2410.23289) 直接从人类视频里的物体轨迹计算奖励，在真机上在线微调。只要任务主要由物体运动定义，这比精确重定向手部更稳健。

### 4.6 小结 {#ch4-6}

- 演示同时解决"奖励"和"探索"两个问题；把它变成策略，要分开处理本体映射（运动学）与物理可行性（动力学）。
- 重定向是一个可以用 Gauss–Newton 实时解的最小二乘；捏取等关键接触要作为硬性目标，而不是被比例缩放"平均掉"。
- 跟踪 RL 的目标以物体为中心：任务项相乘，辅助项小权重相加；动作用"参考 + 残差"，腕部和手指可以区别对待；课程让任务先变容易。

!!! pitfall "常见坑"
    - 缩放系数 $\alpha$ 没调好：指尖总是够不着或穿过物体，拇指的对掌运动尤其容易出问题。
    - 只奖励物体、不约束手：动作怪异，后续很难蒸馏成视觉策略；只奖励手：物体跟不上。
    - 演示里的手-物位姿有噪声和穿模：不先在仿真里回放、去穿模，接触奖励就会把策略引向错误的接触。
    - 跟踪策略依赖仿真的接触细节：部署前同样需要第 3 章的随机化与执行器建模。

### 4.7 自测 {#ch4-7}

<details class="quiz"><summary><span class="qtag">自测</span>为什么跟踪任务的主奖励用乘积，而辅助奖励用小权重的加和？</summary><p>乘积是"与"：位置、姿态、铰接角任何一项很差，总奖励都接近零，策略不能牺牲某一项去刷另一项；对某一项误差的梯度也被其余项加权，迫使各项同时改善。辅助奖励（像不像人手、接触对不对）只是引导，权重小、用加和，使它们在与任务冲突时可以被让步。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>DexMachina 为什么腕部用"重定向 + 残差"，手指却用绝对目标？</summary><p>腕部位姿决定手在空间里的位置，重定向给出的腕部轨迹基本可信，而一旦从零学起偏差很大、很难挽回，所以在它上面学小幅修正最划算。手指的重定向结果本来就不可靠（尺寸与关节结构差异最大），以它为基准反而限制了策略，直接输出归一化的绝对目标更合适。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>虚拟物体控制器为什么不能一开始就关掉？</summary><p>长时程双手任务里，策略早期的大多数动作都会把物体弄掉、提前终止回合，后半段任务几乎从不出现在数据里，策略也就学不到"先举起盒子，再换手开盖"这样的协调。虚拟控制器先替策略把物体沿整条轨迹"拽"着走，让策略在完整序列上学习，再随学习进度逐步交出控制权。</p></details>

## 5. 抓取：可行性、合成与闭环执行 {#ch5}

### 5.1 问题：一个"好抓取"要满足什么 {#ch5-1}

抓取合成是在手的位姿与关节角 $x = (T, R, q)$ 上找一个解，同时满足三类约束：

1. **静力学可行**：接触力能抵抗外力旋量（至少能抵抗重力，最好任意方向都能）——力封闭；
2. **运动学可行**：接触点贴在物体表面、手与物体及手指之间不穿透、关节在限位内；
3. **执行可行**：从当前状态能到达，闭合手指时物体不被推走，受扰动后不打滑。

三代方法分别回答了不同的子问题：**优化**求解前两类约束，**生成模型**把优化"摊销"成一次前向推理，**闭环策略**处理第三类。

### 5.2 静力学：抓取力旋量空间与力封闭 {#ch5-2}

沿用第 1.4 节的记号：第 $i$ 个接触的力 $f_i$ 在接触坐标系中表示，经 $G_i$ 映射成物体上的力旋量 $w_i = G_i f_i$。把摩擦锥线性化成 $m$ 条母线 $d_{ij}$（单位法向力加切向分量），每条母线贡献一个力旋量；它们的凸包就是抓取力旋量空间（GWS）。

- **力封闭** ⇔ 原点严格位于 GWS 内部：总法向力有限时，任意方向的外力旋量都能被接触力平衡。
- **Ferrari–Canny $\varepsilon$ 度量（$Q_1$）** = 原点到 GWS 边界的最小距离，即"最坏方向上单位总法向力能抵抗多大的外力旋量"。力矩分量要除以一个特征长度，才能和力放在同一个尺度上比较。

```python
import numpy as np
from scipy.spatial import ConvexHull

def friction_cone_edges(n, mu, m=8):
    """摩擦锥的 m 条母线（法向分量为 1）。n：指向物体内部的接触法向。"""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    ang = 2 * np.pi * np.arange(m) / m
    return n + mu * (np.cos(ang)[:, None] * t1 + np.sin(ang)[:, None] * t2)

def q1_metric(contacts, normals, center, mu=0.5, m=8, char_len=0.05):
    """Q1：原点到 GWS 凸包边界的最小距离；不是力封闭则返回 0（mu、char_len 为示意值）。"""
    w = [np.concatenate([f, np.cross(c - center, f) / char_len])
         for c, n in zip(contacts, normals) for f in friction_cone_edges(n, mu, m)]
    hull = ConvexHull(np.array(w))              # 6 维凸包
    dist = -hull.equations[:, -1]               # 各个面到原点的有符号距离（法向已归一化）
    return float(max(dist.min(), 0.0))
```

$Q_1$ 适合**评估**，却不适合**优化**：凸包对接触点的依赖不可微，而且对高自由度手的效率很低。下面两种能量就是为了让抓取质量可微、可并行优化。

### 5.3 优化：从可微力封闭到双层 QP {#ch5-3}

**DexGraspNet 的能量。** [DexGraspNet](https://arxiv.org/abs/2210.02697) 在 $(T, R, q)$ 上最小化

$$E = E_{\text{fc}} + 100\, E_{\text{dis}} + 100\, E_{\text{pen}}  + 10\, E_{\text{spen}} + E_{\text{joints}}$$

- $E_{\mathrm{fc}} = \lVert G c\rVert^{2}$ 是可微的力封闭近似：假设每个接触沿法向施加大小相同的力，要求它们的合力与合力矩为零；
- $E_{\mathrm{dis}}$ 让接触点贴在物体表面；$E_{\mathrm{pen}}$ 惩罚手与物体互相穿透——取物体上的点算到手网格的穿透（"反向"计算），即使物体网格很薄、没有厚度也能处理；
- $E_{\mathrm{spen}}$ 惩罚手指之间的自穿透；$E_{\mathrm{joints}}$ 惩罚越过关节限位。

两个工程要点决定了成败：**初始化**（把物体凸包向外膨胀 0.2 m，在膨胀面上采样、让手朝向物体并加扰动，手就能均匀地"包围"物体）和**物理验证**（在仿真里把每个接触连杆沿法向往里挪一点作为位置目标来施加抓握力，依次在 6 个轴向上加重力，100 步后仍接触才算成功，穿透超过 1 mm 也算失败）。

**BODex 的双层优化。** [BODex](https://arxiv.org/abs/2412.16490) 指出，上面这类能量依赖"接触力相等、无摩擦"之类的强假设，能量的高低与仿真中是否成功相关性不强。它把抓取质量直接定义成一个下层 QP 的最优值——在给定手位姿下，接触力**最好能**产生多接近期望力旋量的合力旋量：

$$Q_j(x) = \min_{f}\; \Bigl\lVert \beta\, t_j - \sum_i G_i(x)\, f_i \Bigr\rVert^2 \qquad \text{s.t.}\quad f_i \in F_i\ (\text{摩擦锥，法向分量} \le 1), \quad  \sum_i f_{i,n} \ge \gamma$$

$t_j$ 取沿正负三个力轴方向、力矩为零的 6 个单位力旋量——能抵抗这 6 个方向，就能通过线性组合抵抗任意方向的重力。约束 $\sum f_{i,n} \ge \gamma$ 排除了 $f = 0$ 的平凡解（否则 $Q_j$ 对 $G$ 不可微）。上层对 $x$ 做梯度下降，最小化 $\sum_j Q_j(x)$，同时满足关节限位、接触点在物体表面、无碰撞；梯度经最优解对 $G_i(x)$ 的依赖传回 $x$。

工程上，BODex 用 GPU 上批量求解的 QP（相比 CPU 求解器快约 10 倍）和 cuRobo 的运动学与碰撞库，接触建模先用球近似、再用网格精化，单张 3090 每秒能合成 49 个以上抓取。它的数据让学习模型的仿真成功率从约 40%（用 DexGraspNet 数据）提到约 80%——**合成数据的质量会直接传给学习模型**。

```python
import cvxpy as cp
import numpy as np

def bodex_wrench_energy(G_list, mu=0.5, beta=1.0, gamma=0.5):
    """BODex 式下层问题（示意实现）：对 6 个力方向分别求最小残差，返回能量之和。
    G_list: 各接触的 6×3 映射矩阵（接触坐标系：法向在第 0 维）。上层对手位姿求梯度时，
    需要对这个优化的最优值做可微分（例如 cvxpylayers 或解析的 KKT 灵敏度）。"""
    m = len(G_list)
    G = np.hstack(G_list)                                  # 6 × 3m
    f = cp.Variable(3 * m)
    t = cp.Parameter(6)
    cons = [cp.sum([f[3 * i] for i in range(m)]) >= gamma]
    for i in range(m):
        fn, ft = f[3 * i], f[3 * i + 1: 3 * i + 3]
        cons += [fn <= 1, cp.SOC(mu * fn, ft)]              # 摩擦锥 ‖f_t‖ ≤ μ f_n
    prob = cp.Problem(cp.Minimize(cp.sum_squares(beta * t - G @ f)), cons)
    energy = 0.0
    for axis in range(3):
        for sign in (1.0, -1.0):
            tv = np.zeros(6); tv[axis] = sign
            t.value = tv
            energy += prob.solve()
    return energy
```

### 5.4 生成模型：把优化摊销成一次推理 {#ch5-4}

优化一次要几秒到几十秒，部署时等不起。生成模型把这个代价"摊销"掉：离线用优化合成大量数据，训练一个以物体点云 $P$ 为条件、直接输出 $x = (T, R, q)$ 的条件生成模型。以扩散模型为例，训练目标是

$$L(\theta) = \mathbb{E}_{x_0, \varepsilon, k}\, \bigl\lVert \varepsilon - \varepsilon_\theta(\tilde x_k, k, P) \bigr\rVert^2, \qquad \tilde x_k = \sqrt{\bar\alpha_k}\, x_0 + \sqrt{1 - \bar\alpha_k}\, \varepsilon$$

纯数据驱动的采样不保证物理可行，于是常在采样时加入**物理引导**：每一步去噪后沿可微能量的负梯度修正，$x \leftarrow x - \eta \nabla_x E_{\mathrm{phys}}(x)$，其中 $E_{\mathrm{phys}}$ 可以是穿透、接触距离或上面的抓取能量。这和第 8 章 DexterityGen 用梯度引导把扩散样本"推向"外部指令是同一个机制。[DexGrasp Anything](https://arxiv.org/abs/2503.08257) 把物理约束同时放进训练与采样两个阶段。

跨手泛化的核心是找一个与手无关的中间表示，再映射回具体的手：接触图（[GenDexGrasp](https://arxiv.org/abs/2210.00722)）、机器人-物体统一表示（[D(R,O) Grasp](https://arxiv.org/abs/2410.01702)）、规范 URDF 与参数空间（[OHRA](https://arxiv.org/abs/2602.16712)）。

### 5.5 闭环执行：为什么开环抓取会失败 {#ch5-5}

开环执行一个"好抓取"会因三件事失败：位姿估计误差、手指闭合时物体被推动、闭合时序不协调。所以实际系统都会训练闭环策略，大多数走第 3 章的教师-学生路线（[DextrAH-G](https://arxiv.org/abs/2407.02274) 用几何 fabrics 保证避碰与硬件约束，[DextrAH-RGB](https://arxiv.org/abs/2412.01791) 蒸馏成只看 RGB 的学生）。两个降低 RL 难度的结构性技巧值得单独记住：

- **先专家、后通才。** [UniGraspTransformer](https://arxiv.org/abs/2412.02699) 先为每个物体单独训练 RL 专家、收集成功轨迹，再蒸馏进一个 Transformer。每个专家的 RL 问题都很简单，难点转移到了监督学习上，后者更容易扩展到数千个物体。
- **把长时程问题压成单步决策。** [DemoGrasp](https://arxiv.org/abs/2509.22149) 从一条成功的示范出发，策略只需决定如何"编辑"它——改腕部位姿决定抓哪里，改手指关节角决定怎么抓——这个编辑被写成**单步 MDP**，完全绕开了高维、长时程的探索。

### 5.6 小结 {#ch5-6}

- 抓取的可行性分三层：静力学、运动学、执行；$Q_1$ 是好的评估指标，却不是好的优化目标。
- 让抓取质量可微、可并行，是大规模合成的关键；以"能达到的力旋量"定义质量（BODex 的下层 QP）比带强假设的近似能量更可靠，数据质量直接决定学习模型的上限。
- 生成模型把优化摊销成推理，物理引导弥补其可行性；闭环策略解决执行，结构化动作空间（专家蒸馏、单步编辑）能大幅降低 RL 难度。

!!! pitfall "常见坑"
    - 只看静态指标：数值很好的抓取可能到达不了，或一受扰动就滑。一定要做物理仿真验证。
    - 摩擦系数取得太乐观：合成出大量"靠摩擦挂住"的抓取，真机一抓就滑。
    - 闭环策略在仿真里依赖完美的物体位姿：学生阶段要加位姿噪声与遮挡，或直接从点云、图像学。

### 5.7 自测 {#ch5-7}

<details class="quiz"><summary><span class="qtag">自测</span>两根手指对捏一个球，接触点在一条直径的两端，是力封闭吗？</summary><p>只考虑点接触加库仑摩擦时不是：两个接触点都在同一条直线上，每个接触力对这条连线的力矩都为零，绕连线的力矩无法被平衡，GWS 在这个方向上退化。真实世界里还能捏住，是因为指垫有面积、能提供扭转摩擦——这也是在 MuJoCo 里把 condim 设成 4 会改变抓取结果的原因。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>BODex 为什么要加 $\sum f_n \ge \gamma$ 这个约束？</summary><p>没有它，$f = 0$ 永远可行；当期望力旋量无法达到时，最优解可能退化到 $f = 0$ 附近，此时目标值对接触几何 $G$ 的梯度消失，上层就无法改进手的位姿。要求总法向力至少为 $\gamma$，迫使接触力真正"用起来"，能量对手位姿保持可微且有信息量。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>为什么说 DemoGrasp 把问题"变简单"了？</summary><p>标准的抓取 RL 要在每个时间步输出 20 多维动作、从接近到闭合再到提起，探索空间巨大、奖励稀疏。DemoGrasp 只让策略在开头一次性决定对示范轨迹的编辑量（腕部位姿与手指关节的偏移），然后执行编辑后的轨迹、看成败。决策变成单步的，信用分配完全不成问题。</p></details>

## 6. 基于模型的接触控制：平滑、凸化与采样 {#ch6}

### 6.1 问题：在非光滑的模型上做优化 {#ch6-1}

有了模型 $x^{+} = f(x, u)$，MPC 在每个控制周期求解

$$\min_{u_{0:H-1}}\; \sum_{t=0}^{H-1} c(x_t, u_t) + c_H(x_H) \qquad \text{s.t.}\quad x_{t+1} = f(x_t, u_t)$$

接触让这个问题难在三处（[Pang 等人](https://arxiv.org/abs/2206.10787)的归纳）：接触模式组合爆炸；接触动力学刚硬、非光滑，梯度会爆炸或不连续（不接触时，手指动作对物体的梯度恰好为零）；问题本身非凸。解法因此分成两路：

- **让 $f$ 变好优化**：换一个凸的、可微的接触模型，或者对 $f$ 做平滑，然后用梯度方法；
- **不对 $f$ 求导**：采样 MPC，只需要能把候选动作在模型里前推、比较代价。

这一章先讲第一路的核心思想（它也解释了 RL 为什么能处理接触），再讲第二路的算法。

### 6.2 凸的接触模型 {#ch6-2}

**准动态 + 凸松弛（CQDC）。** 手内操作通常不太动态，可以去掉速度和加速度项、只保留接触力的作用（准动态假设）：状态减半，而且一步就能"看得更远"。Pang 等人把机器人建模成刚度为 $K_a$ 的弹簧，输入 $u$ 是关节的指令位置（弹簧平衡点），一步的动力学写成

$$h K_a\, (q_a + \delta q_a - u) = h\, \tau_a + \sum_i J_{a,i}^\top \lambda_i, \qquad (\varepsilon / h)\, M_u\, \delta q_u = h\, \tau_u + \sum_i J_{u,i}^\top \lambda_i$$

下标 $a$ 为机器人（受驱动）、$u$ 为物体（不受驱动）；$\varepsilon = 0$ 对应精确的准静态，$\varepsilon = 1$ 对应 Mason 的准动态定义，实际取尽量小的正数。接触冲量 $\lambda_i$ 满足 Anitescu 凸松弛下的摩擦锥约束（第 1.3 节），于是求下一步状态就是解一个二阶锥规划。导数 $\partial \delta q^* / \partial q$、$\partial \delta q^* / \partial u$ 由对 KKT 条件用隐函数定理得到。一个容易忽略的实现细节：碰撞检测本身会引入不连续（接触点和法向随构型跳变），他们把手指和物体都表示成内切球的阵列，让接触点与法向随构型平滑变化。

**免互补模型。** [Jin](https://arxiv.org/abs/2408.07855) 从优化型接触模型的对偶出发，用显式的闭式解近似互补约束：时间步进有闭式解、可微、自动满足库仑摩擦律、超参数很少，接触隐式 MPC 能以 50–100 Hz 运行。基于同一模型的 GPU 引擎 [ComFree-Sim](https://arxiv.org/abs/2603.12185) 让接触计算按接触对解耦，运行时间随接触数近线性增长，在真实的 LEAP 手上做了实时 MPC。

### 6.3 平滑：RL 为什么能处理接触 {#ch6-3}

**随机平滑。** 对噪声分布 $\rho$ 定义平滑后的动力学 $f_{\rho}(x) = \mathbb{E}_{w\sim \rho}[f(x + w)]$。Pang 等人证明（$\rho$ 为高斯 $\mathcal{N}(0, \Sigma)$ 时）：用最小二乘在采样点上拟合仿射模型 $f(\bar{x} + w) \approx J w + \mu$，最优的 $J^*$ 恰好是平滑后动力学在 $\bar{x}$ 处的雅可比：

$$J^* = \mathbb{E}\bigl[f(\bar x + w)\, w^\top\bigr]\, \mathbb{E}\bigl[w w^\top\bigr]^{-1}  = \frac{\partial}{\partial x}\, \mathbb{E}_{w \sim \rho}\bigl[f(x + w)\bigr] \Bigr|_{x = \bar x}$$

由此有两种估计器：**一阶**——对采样点处的梯度取平均，方差小，但要求 $f$ 局部 Lipschitz，遇到不连续就有偏；**零阶**——只用函数值做上式的最小二乘拟合，对不连续的 $f$ 依然无偏，代价是方差更大。

这给出了一个很有用的视角：**RL 的随机策略 + 策略梯度，本质上就是零阶随机平滑**——它对接触模式做采样和平均，把非光滑的动力学"抹平"了，所以能学；代价是样本量。

**解析平滑。** 也可以不采样，直接把约束换成对数障碍：

$$\min_{\delta q}\; \frac{1}{2}\, \delta q^\top Q\, \delta q + b^\top \delta q  - \frac{1}{\kappa} \sum_i \log\Bigl[ \frac{(J_{n,i}\, \delta q + \phi_i)^2}{\mu_i^2} - \lVert J_{t,i}\, \delta q \rVert^2 \Bigr]$$

$\kappa \to \infty$ 时解收敛到原来的约束问题；$\kappa$ 有限时，障碍项相当于一个强度与到约束边界距离成反比的"力场"：物体还没碰到手指，就已经能感受到"隔空的力"，梯度不再为零。这和接触隐式轨迹优化里常用的"隔空施力"松弛是同一个效果。两种平滑在简单系统上可以证明等价——一个靠采样，一个靠解析。

!!! key "要点"
    "接触不可微"是表象，真正的问题是梯度只在局部有效、而且常常为零。平滑（随机或解析）用一点物理准确性换来有信息量的梯度；RL、采样 MPC、接触隐式优化，都是在用不同方式做这件事。

### 6.4 采样 MPC {#ch6-4}

<figure class="fig"><div class="fig-scroll"><svg viewBox="0 0 760 222" role="img" aria-label="采样 MPC 循环示意图"><defs><marker id="ah4" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="sv-head"/></marker></defs><rect class="sv-box" x="24" y="24" width="200" height="58" rx="10"/><text class="sv-t2" x="124" y="48" text-anchor="middle">名义计划 θ</text><text class="sv-t3" x="124" y="68" text-anchor="middle">样条控制点或动作序列</text><rect class="sv-box" x="280" y="24" width="200" height="58" rx="10"/><text class="sv-t2" x="380" y="48" text-anchor="middle">采样 N 组扰动</text><text class="sv-t3" x="380" y="68" text-anchor="middle">θᵢ ~ N(θ, σ²)</text><rect class="sv-box" x="536" y="24" width="200" height="58" rx="10"/><text class="sv-t2" x="636" y="48" text-anchor="middle">并行前推 H 步</text><text class="sv-t3" x="636" y="68" text-anchor="middle">在（GPU）仿真里同时 rollout</text><rect class="sv-box" x="536" y="140" width="200" height="58" rx="10"/><text class="sv-t2" x="636" y="164" text-anchor="middle">计算代价 Jᵢ</text><text class="sv-t3" x="636" y="184" text-anchor="middle">物体位姿误差 · 接触 · 限位</text><rect class="sv-box" x="280" y="140" width="200" height="58" rx="10"/><text class="sv-t2" x="380" y="164" text-anchor="middle">更新名义计划</text><text class="sv-t3" x="380" y="184" text-anchor="middle">取最优，或按 exp(−Jᵢ/λ) 加权</text><rect class="sv-hl" x="24" y="140" width="200" height="58" rx="10"/><text class="sv-t2" x="124" y="164" text-anchor="middle">执行第一个动作</text><text class="sv-t3" x="124" y="184" text-anchor="middle">读新状态，热启动下一轮</text><path class="sv-edge" d="M224,53 L278,53" marker-end="url(#ah4)"/><path class="sv-edge" d="M480,53 L534,53" marker-end="url(#ah4)"/><path class="sv-edge" d="M636,82 L636,138" marker-end="url(#ah4)"/><path class="sv-edge" d="M536,169 L482,169" marker-end="url(#ah4)"/><path class="sv-edge" d="M280,169 L226,169" marker-end="url(#ah4)"/><path class="sv-edge" d="M124,140 L124,84" marker-end="url(#ah4)"/></svg></div><figcaption>图 4　采样 MPC 的一轮迭代。模型就是物理仿真器本身，不需要对接触求导；并行算力越多，能评估的候选就越多。</figcaption></figure>

**MPPI。** 从信息论最优控制的角度，最优的动作序列分布是在名义分布 $p(V)$ 上按代价加指数权重：$q^*(V) \propto \exp(-J(V)/\lambda) p(V)$。在高斯族里找离它最近（KL 意义）的分布，其均值就是 $q^*$ 下动作序列的期望，可以用名义计划附近的样本做重要性采样估计：

$$V^* \approx \sum_i w_i V_i, \qquad w_i = \frac{\exp\bigl( -(J(V_i) - J_{\min}) / \lambda \bigr)}{\sum_k \exp\bigl( -(J(V_k) - J_{\min}) / \lambda \bigr)}$$

温度 $\lambda$ 越小越接近"取最优"；减去 $J_{\min}$ 只是为了数值稳定。

**Predictive Sampling**（[MuJoCo MPC](https://arxiv.org/abs/2212.00541) 里最简单的规划器）就是 $\lambda \to 0$ 的极端：用样条参数 $\theta$ 表示动作序列，从 $\mathcal{N}(\theta, \sigma^{2})$ 采 $N - 1$ 组，加上名义计划本身，全部前推后**直接取代价最低的一组**。智能体与规划器异步运行：前者随时从名义计划里读出当前动作执行，后者不停地测状态、热启动、再优化。

**CEM** 介于两者之间：每轮采样后保留代价最低的若干条（精英），用它们重新拟合采样分布的均值与方差，迭代几轮。第 9 章 MPC-Scaffolded RL 用的就是 CEM。

```python
import numpy as np

def mppi_step(theta, rollout_cost, n=256, sigma=0.1, lam=1.0, rng=np.random.default_rng()):
    """一次 MPPI 更新。theta: (H, m) 名义动作序列（或样条控制点）；
    rollout_cost(thetas) -> (n,)：从当前状态出发、在（GPU）仿真里并行前推得到的代价。
    把加权平均换成 argmin 就是 Predictive Sampling。"""
    eps = rng.normal(scale=sigma, size=(n - 1,) + theta.shape)
    thetas = np.concatenate([theta[None], theta[None] + eps])
    J = rollout_cost(thetas)
    w = np.exp(-(J - J.min()) / lam)
    return np.tensordot(w / w.sum(), thetas, axes=1)
```

**高维是采样 MPC 的主要瓶颈。** 在 16 维以上的关节空间里均匀扰动，绝大多数样本都是无意义的手指乱动。三种对策，按"把多少先验放进采样分布"排列：

1. **低维参数化**：用样条控制点而不是逐步动作，降低时间维度。
2. **原语偏置**：[Primitive-Informed MPC](https://arxiv.org/abs/2609.14868) 用编码了手指协调的低维操作原语来偏置采样，同时优化关节级残差以适配当前的手-物状态；前推时用任务约束提前丢弃不可行的轨迹，节省采样预算。
3. **全局探索**：采样方法容易陷入局部最优。[Global-MPPI](https://arxiv.org/abs/2604.27175) 先用核平方和优化找出全局上有希望的区域，再用 log-sum-exp 平滑做渐进非凸（从平滑的替代目标逐步过渡到原始目标），最后用 MPPI 局部细化。

!!! analogy "运控类比"
    和腿足的 MPPI、DIAL-MPC 同源。不同之处在于灵巧手的"模型"几乎只能是完整的接触仿真——手指与物体的接触几何太复杂，很难像单刚体模型那样简化。所以仿真速度直接决定了 MPC 能评估多少候选，也就决定了它的质量。

### 6.5 MPC 与学习的分工 {#ch6-5}

| 分工 | 做法 | 代表 |
|---|---|---|
| 分层 | 高层 RL 输出"接触意图"（物体表面的接触位置 + 接触后物体的子目标位姿），低层接触隐式 MPC 在此条件下优化局部接触模式 | [ Hierarchical RL-MPC](https://arxiv.org/abs/2601.10930) |
| 规划器帮 RL 探索 | 用规划器找到的状态做重置分布，或把规划结果当示范 | [ Sampling-based Exploration](https://arxiv.org/abs/2303.03486)、[ Jacta](https://arxiv.org/abs/2408.01258) |
| 规划器生成模仿数据 | 用规划器合成示范，训练行为克隆策略 | [ Should We Learn from Planners?](https://arxiv.org/abs/2412.09743) |
| MPC 给真机 RL 搭脚手架 | MPC 轨迹预训练 + 在线时按计划逐步减少 MPC 的参与 | [ MPC-Scaffolded Real RL](https://arxiv.org/abs/2609.14878)（第 9 章） |

用规划器生成模仿数据有一个特有的陷阱：RRT 这类规划器给出的示范"熵太高"——同一情形下的动作差异很大，行为克隆学到的是平均后的、无效的动作。需要改造数据生成，使示范在保持解的多样性的同时尽量一致。

### 6.6 小结 {#ch6-6}

- 接触问题的核心是梯度只在局部有效、常常为零；平滑（随机或解析）用少量物理准确性换取有信息量的梯度，RL 本质上就是零阶随机平滑。
- 凸的、可微的接触模型（准动态 + 凸松弛、免互补模型）让基于梯度的规划和实时 MPC 成为可能。
- 采样 MPC 不需要梯度，瓶颈在高维与局部最优：用低维参数化、原语偏置和全局探索把先验放进采样分布。
- MPC 与 RL 是互补的：MPC 不用训练、目标可随时改；RL 部署快、能吃高维观测。分层、提供探索和示范、给真机 RL 搭脚手架，都是常见的结合方式。

!!! pitfall "常见坑"
    - 代价函数只写物体误差：MPC 会找到"用很大的力挤压物体"这类在仿真里有效、真机上危险的解。要加力矩、接触力和关节限位相关的项。
    - 模型与真机的接触参数不一致：MPC 依赖模型前推，失配比 RL 更敏感。
    - 时域太短：看不到"先松一指、再换位"这种先牺牲、后收益的动作。
    - 状态估计跟不上：物体位姿的抖动会直接变成控制的抖动。

### 6.7 自测 {#ch6-7}

<details class="quiz"><summary><span class="qtag">自测</span>手指离物体还有 1 cm 时，物体位置对手指动作的梯度是多少？平滑之后呢？</summary><p>没有接触时，手指的动作不影响物体，梯度恰好为零——基于梯度的优化器完全不知道"往前伸一点就能推动物体"。随机平滑后，部分采样会发生接触，平均下来梯度非零；对数障碍平滑后，约束项像一个隔空的力场，距离越近作用越强，梯度同样非零并指向正确的方向。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>零阶和一阶随机平滑估计器各适合什么情况？</summary><p>一阶估计器对采样点处的梯度取平均，方差小，但要求函数局部 Lipschitz；函数里有真正的跳变（比如碰撞发生的瞬间）时它会有偏。零阶估计器只用函数值做回归，对不连续函数依然无偏，代价是方差大、需要更多样本。接触丰富的问题里两者常结合使用，策略梯度类的 RL 属于零阶。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>什么情况下你会优先选 MPC 而不是 RL？</summary><p>有可信的模型、目标容易写成代价、需要频繁改目标或约束、或者想先在真机上拿到一个可用的控制器时，MPC 合适；它也适合给 RL 当脚手架或示范来源。观测是高维图像、模型难以建准、或者算力预算只允许毫秒级推理时，蒸馏出来的 RL 策略更合适。</p></details>

## 7. 触觉：表示、仿真与迁移 {#ch7}

### 7.1 问题：触觉要解决什么，又难在哪 {#ch7-1}

第 1 章说过，控制物体就是控制接触。而接触状态——在哪碰、碰多重、有没有在滑——恰恰是视觉最看不清的：物体被手挡住，接触面又在手和物体之间。触觉直接测量这些量，所以它在灵巧手里的地位比足底传感在腿足里重要得多。

但它带来三个核心难题：

1. **表示**：原始触觉信号维度高、语义不直观（图像、磁通量、电容阵列），策略到底需要什么？
2. **仿真**：弹性体形变、剪切、粘滑，仿真既难做准、又难做快，而第 3 章的整套方法依赖大规模仿真；
3. **异构**：每种传感器的原理、布局、噪声都不同，几乎每换一种传感器就相当于换了一台新机器人。

一个贯穿本章的取舍原则是：**信息量越高的表示越难仿真、越难迁移**。

### 7.2 信号与表示：信息量和可迁移性的取舍 {#ch7-2}

| 表示 | 典型来源 | 信息量 | 仿真难度 |
|---|---|---|---|
| 二值接触（碰没碰） | 阈值化的力传感器 | 低：只有接触位置 | 很低：几乎没有差距 |
| 接触力 / 力矩 | 指尖力传感器、关节力矩、电机电流 | 中：大小与方向 | 中：依赖接触模型与执行器模型 |
| 触觉图像 / 标记点位移 | 视触觉传感器（DIGIT 类：相机 + 弹性体） | 高：接触几何、剪切、滑移 | 高：要模拟弹性体形变与光学 |
| 学到的嵌入 | 上述任意信号 + 自监督预训练 | 高，且对传感器差异更鲁棒 | 不直接仿真，需对齐（7.5 节） |

两个端点各有代表：[Rotating without Seeing](https://arxiv.org/abs/2303.10880) 在手掌、指节、指尖一侧铺满二值力传感器，只靠触觉、不看物体完成手内旋转——用低信息量换来了大覆盖面、低成本和几乎为零的 sim2real 差距；另一端，视触觉传感器能"看到"接触面的几何与剪切，但只能装在指尖，仿真也最难。

本体信号也可以当触觉用。手的动力学里，接触力会以 $J^{\top} f$ 的形式反映到关节力矩上（第 1.2 节），所以电机电流本身就携带接触信息。[Current as Touch](https://arxiv.org/abs/2607.03529) 从电流和关节状态中学出一个"柔顺参考位置"——一个让 PD 的位置误差恰好产生合适抓握力的关节目标——既不用估计外力，也不用切换到力矩控制。

### 7.3 表示学习：把触觉信号压成可用的特征 {#ch7-3}

有标签的触觉数据（接触力、滑移）很难大规模采集，所以主流做法是自监督预训练。[Sparsh](https://arxiv.org/abs/2410.24090) 在 46 万多张、来自多种视触觉传感器的图像上，系统比较了像素空间的掩码重建与隐空间的自蒸馏（DINO、I-JEPA 一类目标）：在它的 TacBench 上，自监督预训练比针对任务、针对传感器的端到端训练平均好 95.1%，而**隐空间的目标（DINO、I-JEPA）最有竞争力**。原因与视觉类似：触觉图像里有大量与任务无关的细节（光照、凝胶纹理），在隐空间里预测能忽略它们。

全手触觉也是同样的思路：[Sparsh-skin](https://arxiv.org/abs/2505.11420) 输入整只手的运动与磁性皮肤历史，用自蒸馏学出一个可供任意下游任务使用的触觉嵌入。跨传感器的统一表示（[AnyTouch](https://arxiv.org/abs/2502.12191)、[UniTouch](https://arxiv.org/abs/2401.18084)）则直接针对第 3 个难题：让不同传感器的信号落到同一个特征空间。

### 7.4 触觉仿真：保真度与速度 {#ch7-4}

几何投影太粗糙，有限元又太慢，撑不起大规模 RL。现有方法都在两者之间找折中：

- **穿透深度图**：在仿真里计算接触体积、转成深度图，再把真机的原始信号也映射到同一种"形变图"表示（[TacMap](https://arxiv.org/abs/2602.21625)）；
- **水弹性接触 + 剪切**：在水弹性接触模型上追踪接触面各点的位移，建模粘滑转换和路径相关的剪切累积（[HydroShear](https://arxiv.org/abs/2603.00446)）——很多灵巧任务恰恰依赖剪切与滑移；
- **数据驱动的形变传播**：用轻量模型近似弹性体形变，表面形变精度接近有限元（[ETac](https://arxiv.org/abs/2604.20295)）；
- **多种抽象统一接口**：二值接触、接触深度、逐触点力、标记点位移、接近觉、温度等可配置，并带漂移、迟滞、坏点、串扰等噪声模型，可跑 2 万多个并行环境（[Tactile Genesis](https://arxiv.org/abs/2606.22332)），用来研究"策略到底需要哪种触觉"。

### 7.5 不仿真触觉的迁移 {#ch7-5}

既然触觉仿真难做准，一类更务实的方法干脆不在仿真里生成真实的触觉信号。三种框架的思路各不相同：

**① 真机特权传感器 + 蒸馏（[PTLD](https://arxiv.org/abs/2603.04531)）。** 仿真里只训练本体策略；到真机上，借助只在采数据时使用的特权传感器采集策略运行数据，蒸馏出一个以触觉为输入的状态估计器，再把估计结果交给策略。好处是触觉部分完全在真实数据上训练。在手内旋转任务上，它比纯本体策略提升了 182%；在触觉手内重定向上，到达目标数多了 57%。

**② 仿真预言者 + 隐空间对齐（[SBLR](https://arxiv.org/abs/2608.15897)）。** 在仿真里用一个容易构造、与具体传感器无关的"预言者传感器"（例如点云加指尖力）训练策略；部署时，学一个映射把真实传感器的隐变量变换到预言者的隐空间。训练分两步：先基于预言者隐变量学策略，再用"瓶颈化的隐变量重建"让策略适应真实传感器带来的信息损失。对齐用 rectified flow：在仿真与真机各自采集的无配对随机玩耍数据上，用近邻匹配构造伪配对 $(z_{\mathrm{real}}, z_{\mathrm{oracle}})$，学一个速度场

$$\min_{\theta}\; \mathbb{E}_{s \sim U(0,1)}\, \bigl\lVert v_\theta(z_s, s) - (z_{\text{oracle}} - z_{\text{real}}) \bigr\rVert^2, \qquad z_s = (1 - s)\, z_{\text{real}} + s\, z_{\text{oracle}}$$

推理时沿 $v_{\theta}$ 积分，把真实隐变量"搬运"到预言者的隐空间。换一种新传感器，只需重新采一点玩耍数据。真机上用 GelSight Mini 和 DIGIT 做插孔与齿轮啮合，零样本成功率 85–97.5%，比基于物理的触觉仿真基线高 7.5–15%。

**③ 把仿真标定到真机（[Blind Dexterous Grasping](https://arxiv.org/abs/2606.11767)）。** 构建接触标定过的数字孪生，让仿真复现真实的触觉接触事件，再在仿真里训练。它也说明了这个问题有多难：在带分布式触觉的 LEAP 手上，不看图像、只凭触觉，对 20 个物体的真机抓取成功率是 27%。

| 框架 | 需要什么 | 适合 |
|---|---|---|
| 只用好仿真的信号 | 二值或阈值化信号 + 噪声与延迟随机化 | 任务主要依赖"哪里在接触" |
| 真机特权蒸馏 | 采数据时的特权测量 | 有动捕等外部测量条件，触觉信号难建模 |
| 预言者 + 对齐 | 两边的无配对玩耍数据 | 传感器多、经常更换 |
| 标定仿真 | 精细的接触与传感器标定 | 单一传感器、长期使用 |

### 7.6 什么时候值得上触觉 {#ch7-6}

[When Does Touch Matter?](https://arxiv.org/abs/2609.24068) 做了一个很干净的真机对照：在 LEAP 手上做杂乱场景抓取，示范、视觉观测、动作空间与柔顺控制全部固定，只改变输入模态，分别训练扩散策略。视觉 + 手指与腕部力旋量估计 + 指尖触点的组合策略成功 24/25 次，纯视觉 14/25；在三种受限空间的场景里，差距是 15/15 对 6/15。规律很清楚：**遮挡越重、空间越受限，触觉的价值越大**。第 11 章还会看到，把触觉总结成上下文后，大模型在插孔任务上的成功率从 31% 提到 99%。

### 7.7 小结 {#ch7-7}

- 触觉提供视觉最缺的东西——接触状态；但表示的信息量越高，仿真与迁移就越难。
- 表示学习上，隐空间的自监督目标（DINO、I-JEPA）优于像素重建；跨传感器的统一表示针对的是异构问题。
- 迁移上，与其追求逼真的触觉仿真，不如考虑"不仿真触觉"的框架：真机特权蒸馏、预言者 + 隐空间对齐、或只用容易仿真的信号。

!!! pitfall "常见坑"
    - 仿真里的触觉太干净：没有漂移、迟滞和坏点，真机上策略直接失效。
    - 触觉和本体信号在时间上没对齐：触觉通常走另一条通信链路，延迟不同。
    - 弹性体磨损、换皮肤后信号漂移：留出重新标定的流程，或用对传感器差异更鲁棒的表示。
    - 为了用触觉而用触觉：先做简单对照，确认任务真的需要哪种触觉。

### 7.8 自测 {#ch7-8}

<details class="quiz"><summary><span class="qtag">自测</span>PTLD 和 SBLR 都"不仿真触觉"，它们分别把难题挪到了哪里？</summary><p>PTLD 把触觉完全留在真机侧：难题变成"真机上如何获得监督"，靠采数据时的特权测量解决。SBLR 把触觉的作用在仿真里用一个理想化的预言者传感器代替：难题变成"如何把真实传感器的隐变量对齐到预言者的隐空间"，靠两边的无配对玩耍数据和 rectified flow 解决。前者每个任务都要在真机上采集，后者每换一种传感器只需重新做对齐。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>为什么触觉表示学习里，隐空间的自监督目标比像素重建更好？</summary><p>像素重建要求模型把每个像素都预测对，包括光照、凝胶纹理、标记点外观这类与接触无关的细节，容量被浪费在这些地方；隐空间目标（自蒸馏、联合嵌入预测）只要求表示之间一致，可以忽略不可预测、与任务无关的细节，学到的特征更集中于接触几何与力。Sparsh 的对比实验正好支持这一点。</p></details>

## 8. 从数据学通用策略：生成式策略、基础控制器与 VLA {#ch8}

### 8.1 问题：数据驱动的灵巧策略卡在哪里 {#ch8-1}

另一条路线是直接从示范学：采集数据，训练模仿策略，进而训练能听懂语言的视觉-语言-动作模型（VLA）。对灵巧手，它卡在四个核心问题上：

1. **数据缺力**：遥操作者感受不到接触，人类视频里根本没有力；示范记录的是"动作的样子"，而不是"用多大的力"。
2. **动作分布多峰且高维**：同一个状态下可以有好几种合理的抓法或步态；双手加起来 40 多维，直接回归会把几种模式平均成一个无效的动作。
3. **接触执行要高频**：VLA 推理慢（每秒几次），而维持接触需要每秒几十到几百次的调整。
4. **本体差距**：手的自由度从 6 到 24 不等，人手数据和不同机器人手的数据彼此割裂。

本章依次处理这四个问题。

### 8.2 数据：采集方式决定了数据里有什么 {#ch8-2}

| 采集方式 | 本体差距 | 力 / 触觉 | 规模 |
|---|---|---|---|
| 视觉遥操作（VR、手部追踪） | 无 | 操作者感受不到接触 | 小 |
| 力反馈手套与外骨骼 | 小 | 有反馈，部分设备直接记录触觉 | 小到中 |
| 人手直接采集（手套、外骨骼 + 相机） | 需要软硬件适配 | 视设备而定 | 中 |
| 人类第一人称视频 | 大 | 无 | 大（上万小时） |
| 仿真合成（第 3–5 章） | 有 sim2real 差距 | 仿真里可得 | 很大 |

两个设计思路值得注意。其一，**把力反馈还给操作者**：[DEXOP](https://arxiv.org/abs/2509.04441) 用被动外骨骼把人的手指机械地连到机器人手指上，操作者通过本体感觉直接感受接触，同时记录视觉与触觉；[DexUMI](https://arxiv.org/abs/2505.21864) 用外骨骼弥合运动学差距，再在视频里把人手"修补"成机器人手，弥合视觉差距。其二，**让人只做擅长的事**——这就是 8.4 节的基础控制器。

### 8.3 生成式策略：处理多峰与高维 {#ch8-3}

**扩散策略。** 把未来 $H$ 步的动作块 $A = (a_t, \ldots, a_{t+H-1})$ 当作要生成的样本，以观测 $o$ 为条件。训练目标是去噪：

$$L(\theta) = \mathbb{E}_{A_0, \varepsilon, k}\, \bigl\lVert \varepsilon - \varepsilon_\theta(\tilde A_k, k, o) \bigr\rVert^2, \qquad \tilde A_k = \sqrt{\bar\alpha_k}\, A_0 + \sqrt{1 - \bar\alpha_k}\, \varepsilon$$

**流匹配**是更直接的变体：在噪声 $A_0$ 与数据 $A_1$ 之间取直线插值 $A_s = (1-s)A_0 + s A_1$，学一个速度场

$$L(\theta) = \mathbb{E}\, \bigl\lVert v_\theta(A_s, s, o) - (A_1 - A_0) \bigr\rVert^2$$

推理时沿速度场积分几步即可，比扩散快。两者都能表示多峰分布：不同的初始噪声落到不同的模式上，而不是被平均成一个"中间动作"。

**动作块的执行。** 一次预测 $H$ 步、只执行前 $h$ 步（$h \le H$）再重新预测，是滚动时域控制；$h$ 越小反应越快，但相邻两个动作块之间可能"切换模式"而产生抖动，常用的缓解办法是对重叠部分做时间上的加权平均。

**表示的选择比网络结构更重要。** [DP3](https://arxiv.org/abs/2403.03954) 把稀疏点云用一个很轻的编码器压成紧凑的 3D 表示，再接扩散策略：在 72 个仿真任务里，大多数只要 10 条示范。紧凑、带几何信息的输入，对数据量很小的灵巧任务特别关键。

### 8.4 基础控制器：把接触执行从高层策略里剥离 {#ch8-4}

人擅长给出粗略的意图（往哪拿、往哪转），却很难在没有触觉反馈时精细地协调十几个关节去维持接触；RL 恰好相反，擅长低层运动原语，却难以处理长时程任务和复杂奖励。[DexterityGen](https://arxiv.org/abs/2502.04307)（DexGen）把两者分开，构造了一个灵巧手的"基础控制器"：

1. **用 RL 造数据。** 核心预训练任务是"任意抓取 → 任意抓取"：先用抓取分析和 RRT 为每个物体生成大量抓取（多数物体超过 10 万个），训练 RL 策略从当前抓取走到 $k$ 近邻里的一个相邻抓取（太远的目标学不会），到达后再换下一个目标；再加上自由移动手指、精细旋转等任务，并随机化物体与腕部姿态。最终收集了 $10^{10}$ 个转移（约合 31.7 年真实时间，生成用了 300 GPU 小时）。
2. **生成式控制器。** 一个扩散模型以本体历史和任务模式为条件，生成未来几步的**指尖关键点位移** $\Delta x \in \mathbb{R}^{T\times K\times 3}$（手坐标系）；一个逆动力学模型（残差 MLP）把关键点位移翻译成关节 PD 目标。用关键点而不是关节角做中间表示，方便接收来自人手的指令。
3. **用引导吸收外部指令。** 遥操作或高层策略给出的粗略运动 $\Delta x^{\mathrm{input}}$ 不直接作为条件输入，而是作为**采样时的梯度引导**：每一步去噪的均值加上 $\alpha\, \Sigma\, \nabla J(\mu)$，其中 $J = -\sum_i\lVert \Delta x_i - \Delta x^{\mathrm{input}}\rVert^{2}$。效果是从"数据里见过的安全动作分布"中，挑出最接近指令的那个——即使输入指令大部分是噪声，输出依然安全。

这个结构给了一个通用的分工：**高层（人、VLA 或规划器）给意图，低层生成模型给出数据分布内、物理可行的接触执行**。VLA 推理慢的问题也因此缓解：慢的高层只需低频地给出意图。[LiMA](https://arxiv.org/abs/2609.28431) 把同样的思路用在世界-动作模型上：慢系统生成稀疏的长时程意图，快系统做高频的动作细化，推理延迟比 Cosmos-Policy 低 45.8%。

!!! analogy "运控类比"
    这和人形里"VLA 出目标、全身控制器管平衡"的分层是一个道理。基础控制器的地位，相当于人形里用大量动捕和 RL 训练出来的运动先验或通用跟踪控制器。

### 8.5 跨本体：一个策略用在多种手上 {#ch8-5}

各种做法的共同点是：**找一个与具体手无关的中间空间，在那里学策略，再映射回具体的关节**。差别在于这个中间空间怎么定义：

| 中间空间 | 做法 | 代表 |
|---|---|---|
| 学到的潜在动作 | 各种手的动作编码进同一个潜在空间，可直接插进标准 VLA | [ XL-VLA](https://arxiv.org/abs/2603.10158)（4 种手、10 个任务平均 72%，π0 基线 32%） |
| 语义化的关节接口 + 离散 token | 人手与机器人手映射到共享的 22 自由度语义接口，再学以本体为条件的离散 token | [ UniDexTok](https://arxiv.org/abs/2606.10683) |
| 功能对齐的执行器空间 | 把功能相似的执行器映射到共享坐标 | [ UniDex](https://arxiv.org/abs/2603.22264) |
| 几何表示 | 把手的动作表示为规范球体的形变，用级联 IK 映射到关节 | [ UHAS](https://arxiv.org/abs/2607.03570) |
| 从历史推断形态 | 历史条件的 Transformer 在线推断手的形态与动力学 | [ DexFormer](https://arxiv.org/abs/2602.08278) |

前三种需要为每种手定义映射，最后一种把"认出是哪只手"交给模型自己，思路与第 3 章的适应模块相同。

### 8.6 人类视频与规模定律 {#ch8-6}

人类视频是最大的数据来源，问题是怎么把人手的动作变成机器人能执行的监督。[EgoScale](https://arxiv.org/abs/2602.16710) 给出了目前最清晰的结论：在 20,854 小时带动作标签的第一人称视频上训练 VLA（比此前大 20 倍以上），**人类数据规模与验证损失呈对数线性关系，而验证损失与真机表现强相关**。这意味着可以像大语言模型那样，用离线指标预测"再加多少数据能换来多少提升"。迁移配方是两步：大规模人类数据预训练，然后用少量对齐的人-机数据做轻量的中间训练。

人类视频没有力（问题 1），所以它解决的主要是"做什么、往哪动"；"用多大的力"仍要靠第 4 章的物理跟踪、第 9 章的真机后训练，或带触觉与力的采集设备来补。

### 8.7 小结 {#ch8-7}

- 数据决定了策略能学到什么：采集方式决定了数据里有没有力、本体差距有多大、规模能做多大。
- 生成式策略（扩散、流匹配）解决多峰与高维；紧凑的几何输入在小数据下尤其关键。
- 把接触执行剥离给基础控制器：高层给意图，低层生成分布内、物理可行的动作，靠采样引导吸收外部指令。
- 跨本体的关键是找一个与手无关的中间空间；人类视频按规模定律可预测地提升"做什么"，"用多大力"仍需物理与真机来补。

!!! pitfall "常见坑"
    - 示范里没有力：在需要夹紧、按压的任务上，模仿策略会"做对动作却用错力"。
    - 遥操作的延迟与抖动被一并学去：采集时就要滤波，并剔除失败或犹豫的片段。
    - 直接回归高维动作：多峰被平均掉，动作发虚。
    - 跨本体训练时关节顺序、单位、零位不一致：统一表示之前先把约定理清。

### 8.8 自测 {#ch8-8}

<details class="quiz"><summary><span class="qtag">自测</span>DexGen 为什么把遥操作指令当作采样时的引导，而不是直接作为条件输入？</summary><p>作为条件输入时，模型要学会"指令 → 动作"的映射，而训练数据里并没有真实的人类指令，分布外的、带噪声的指令可能产生危险的输出。作为引导时，生成的样本始终来自训练数据中安全动作的分布，引导只是在这个分布里挑最接近指令的那一个，所以即使指令大部分是噪声，输出依然安全。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>EgoScale 的"验证损失与真机表现强相关"为什么重要？</summary><p>真机评测又贵又慢，没法频繁地拿来指导数据和模型的扩展决策。如果离线的验证损失能可靠地预测真机表现，就可以用规模定律规划"再加多少小时人类数据、能换来多少提升"，把数据采集变成一个可预测的工程问题。</p></details>

## 9. 真机 RL 与后训练 {#ch9}

### 9.1 问题：在真机上继续变好 {#ch9-1}

仿真训练出的策略带着残余的 sim2real 差距，模仿策略又受限于示范质量，常常停在 60–80% 的成功率。最后这一段，只能在真机上用真实交互来补。真机 RL 和仿真 RL 是两个很不一样的问题：

- **样本以分钟计价**：仿真里几亿步的预算，到真机上只剩几千到几万步；
- **失败有代价**：物体掉了要人去捡、复位，手指卡死可能损坏硬件；
- **奖励要从真实传感器里算**：没有仿真里的物体真值。

于是真机 RL 的核心问题是：**如何用尽可能少、尽可能安全的交互，把一个"还不错"的策略推到"可靠"**。答案的骨架是离策略 RL，关键在于往里注入什么样的先验。

### 9.2 骨架：离策略的 actor-critic {#ch9-2}

样本贵，所以必须复用历史数据——用离策略算法，通常是 SAC。critic 回归带熵的 Bellman 目标，actor 最大化 $Q$ 值加熵：

$$L_Q = \mathbb{E}\bigl[ (Q_\phi(s, a) - y)^2 \bigr], \qquad y = r + \gamma\, \mathbb{E}_{a' \sim \pi}\bigl[ Q_{\bar\phi}(s', a') - \alpha \log \pi(a' \mid s') \bigr]$$

$$L_\pi = \mathbb{E}_{s}\, \mathbb{E}_{a \sim \pi}\bigl[ \alpha \log \pi(a \mid s) - Q_\phi(s, a) \bigr]$$

在这个骨架上，真机 RL 系统普遍加了三件事：

1. **先验数据与在线数据按比例混合采样。** RLPD 的做法是每个批次一半来自先验数据、一半来自在线数据（[HIL-SERL](https://arxiv.org/abs/2410.21845) 沿用）；先验数据质量一般时，比例可以调低（MPC-Scaffolded RL 用 30% 先验、70% 在线）。
2. **critic 集成，抑制过估计。** 计算目标时从 10 个目标 critic 中随机取 2 个、取最小值（MPC-Scaffolded RL 的做法）。数据少时 $Q$ 值过估计会被 actor 迅速利用，这一步很关键。
3. **预训练的表示。** 视觉输入用预训练好的编码器（HIL-SERL 用 ImageNet 预训练的 ResNet-10），不在几千个样本上从头学视觉。

### 9.3 往里注入什么先验：三种来源 {#ch9-3}

**① 人：示范 + 实时干预（HIL-SERL）。** 先采 20–30 条示范放进示范缓冲区，并用示范训练一个二分类成功判别器作为奖励。训练时操作者拿着 SpaceMouse 在旁边看，策略要出问题时随时接管；**干预片段同时进入示范缓冲区和 RL 缓冲区，而策略自己的转移只进 RL 缓冲区**。干预把探索约束在有意义的区域，避免大部分真机时间浪费在注定失败的动作上；底层用带参考限幅的阻抗控制保证安全。在精密与灵巧任务上，1–2.5 小时的训练就能显著超过模仿策略。

**② 规划器：MPC 当脚手架（[MPC-Scaffolded Real RL](https://arxiv.org/abs/2609.14878)）。** 灵巧手的干预很难做——人很难实时接管十几个手指关节。这项工作用一个采样 MPC（基于同步的 MuJoCo 数字孪生，CEM 优化，低维原语加关节残差采样）代替人，提供先验经验与在线引导。流程分三段：

- **离线数据**：用 MPC 在真机上采 20 条轨迹（每条把物体转满 $2\pi$、不掉落，共约 12 分钟），用**在线 RL 的奖励**重新标注，作为固定的先验缓冲区。
- **预训练 actor 与 critic。** actor 用行为克隆初始化，但要照顾到 SAC 的随机策略结构：只回归 tanh 之前的均值，目标取 $\operatorname{arctanh}(\operatorname{clip}(a_{\mathrm{demo}}, -1+\varepsilon, 1-\varepsilon))$；按轨迹回报的 softmax 加权采样（优先学好的轨迹）；标准差头直接初始化为一个常数。critic 用带熵的 Soft-SARSA 目标预训练：$y = r + \gamma[\min_j Q_j(s', a'_{\mathrm{demo}}) - \alpha_{\mathrm{eff}} \log \pi_{\mathrm{BC}}(a'_{\mathrm{demo}}\mid s')]$——用的是示范里实际执行的下一个动作，与后续的 SAC 目标在形式上一致。
- **在线 SAC + 逐步撤除 MPC。** 每个批次 30% 来自 MPC 数据、70% 来自在线经验；执行策略（而不是 MPC）的概率按 $p_{\pi}(k) = p_0 + (1 - p_0) \min(k/K, 1)$ 线性增长，$p_0 = 0.5$，$K = 6000$ 步（约 10 分钟），之后不再用 MPC。选择谁来控制以 40 步为一块，块内不切换、两者的动作也不混合——这保证了手指动作序列在时间上是连贯的。

奖励只用物理进展和终止结果：转过的角度给正奖励，转满一圈加 200，掉落就以 −50 终止。结果：Allegro 手在线 RL 7 分钟达到 100% 成功（训练中平均只掉约 3 次）；20 分钟后转速超过 MPC 的 5 倍，并能连续转 1000 圈不掉。消融显示三部分各有分工：保留 MPC 数据主要减少掉落（45 分钟内，纯 RL 掉了 181 次，只加先验数据回放降到 98 次）；预训练加速早期学习；在线引导让学习更稳定。

**③ 已有策略：残差 RL。** 已经有一个行为克隆策略或 VLA 时，不去微调整个大模型，而是冻结它、学一个小的残差：$a = a_{\mathrm{base}}(o) + \Delta a_{\theta}(o)$。

- [ResFiT](https://arxiv.org/abs/2509.19301) 把行为克隆策略当黑盒，用离策略 RL 学逐步的残差修正，只需要稀疏的二值奖励；论文报告了首个在带灵巧手的人形上的真机 RL 训练。
- [BORA](https://arxiv.org/abs/2605.30226) 在一致性策略形式的 VLA 里加入以动作为条件的 critic，并复用它来训练冻结基座之上的轻量残差 actor；人的纠正通过可穿戴遥操作加一个把粗略意图翻译成手指动作的局部策略来采集。每个任务只用 20 条在线轨迹，平均成功率从 60.8% 提到 82.5%。

残差的好处是：初始策略就是基座策略（残差初始化为零），探索被限制在基座附近，大模型的知识不会被少量真机数据破坏。

[RL-100](https://arxiv.org/abs/2510.14830) 走了另一条路：直接在扩散策略上做 RL，把裁剪的 PPO 替代目标作用在去噪过程的每一步上，用同一个目标统一模仿学习、离线 RL 与在线 RL；部署时再用一致性蒸馏把多步扩散压成一步，满足高频控制的延迟要求。

### 9.4 一个训练循环的骨架 {#ch9-4}

```python
import random

def real_world_rl(env, agent, prior_buf, online_buf, guide=None, p0=0.5, K=6000,
                  chunk=40, prior_frac=0.3, batch=256, utd=4, total_steps=20000):
    """离策略真机 RL 的骨架（MPC 脚手架版本；把 guide 换成人的干预即为 HIL-SERL 式）。
    prior_buf：固定的先验数据（MPC 或示范）；online_buf：在线经验。比例与 K、chunk 取自
    MPC-Scaffolded RL，batch 与 utd（每步更新次数）为示意值。"""
    obs, step = env.reset(), 0
    while step < total_steps:
        p_pi = p0 + (1 - p0) * min(step / K, 1.0)
        use_policy = guide is None or random.random() < p_pi     # 每 chunk 步才选一次控制器
        for _ in range(chunk):
            act = agent.act(obs) if use_policy else guide.act(obs)
            nxt, rew, done, info = env.step(act)                  # 底层：限幅的阻抗 / PD 控制
            online_buf.add(obs, act, rew, nxt, done)              # 两种控制器的数据同等对待
            obs, step = (env.reset() if done else nxt), step + 1
            for _ in range(utd):
                n_prior = int(batch * prior_frac)
                agent.update(prior_buf.sample(n_prior) + online_buf.sample(batch - n_prior))
            if done:
                break
```

### 9.5 安全与工程 {#ch9-5}

- **底层限幅**：关节位置、速度、力矩都要有硬限；HIL-SERL 在实时层用带参考限幅的阻抗控制。策略输出先过限幅再下发。
- **奖励的来源**：成功判别器（视觉二分类）、外部测量（MPC-Scaffolded RL 用动捕测物体姿态）、或任务本身可测的量；判别器的误报会被策略钻空子，要定期人工核查。
- **复位**：真机 RL 的时间大头往往花在复位上。设计能自动复位的任务（例如掉落后物体能回到可抓取的位置），或者像 MPC-Scaffolded RL 那样，用先验数据从源头上减少掉落。
- **评估要和训练分开**：定期只用策略、关掉引导与探索噪声做评估，否则看到的是"策略 + 引导"的表现。

### 9.6 小结 {#ch9-6}

- 真机 RL 的骨架是离策略 actor-critic；先验数据按比例混合、critic 集成和预训练表示，是样本效率的三个支点。
- 先验来自哪里决定了系统形态：人（示范与干预）、规划器（MPC 脚手架）、已有策略（残差）。灵巧手上人很难实时干预，MPC 和残差尤其有用。
- 引导要逐步撤除，评估要与引导分开；动作的连贯性（按块切换、不混合）对灵巧手比对夹爪更重要。

### 9.7 自测 {#ch9-7}

<details class="quiz"><summary><span class="qtag">自测</span>HIL-SERL 为什么把干预数据同时放进两个缓冲区，而策略自己的转移只放进 RL 缓冲区？</summary><p>干预数据是"在策略会到达的状态上，正确的做法"，既是高质量的示范（放进示范缓冲区，在每个批次里以固定比例被采到），也是真实的交互经验（放进 RL 缓冲区，用于学习价值函数）。策略自己的转移只是普通经验，质量参差，不应该被当作示范来强化。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>MPC-Scaffolded RL 为什么按 40 步一块来选择控制器，而且不混合两者的动作？</summary><p>手指的协调动作是一个时间上连贯的序列：MPC 和策略各自生成的序列，在任意时刻拼接或加权混合，得到的往往是一段不协调的手指运动，物体很容易掉落。按块切换保证每一段动作都来自同一个控制器，既安全，又让离策略学习器看到完整、连贯的行为片段。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>给一个已有的 VLA 做真机后训练，为什么常用残差而不是全量微调？</summary><p>真机数据只有几十条轨迹，全量微调一个大模型很容易过拟合、破坏预训练学到的通用能力，也很贵。冻结基座、只学一个小残差：初始策略就是基座策略，探索被限制在基座附近，参数少、更新快、数据效率高；需要时还能随时退回到基座策略。</p></details>

## 10. 灵巧世界模型 {#ch10}

### 10.1 问题：预测"手做了什么之后，世界会怎样" {#ch10-1}

世界模型学习动作条件的预测分布

$$p\bigl( o_{t+1:t+H} \mid o_{\le t},\, a_{t:t+H-1} \bigr)$$

对灵巧手，它比一般的操作世界模型难在三处：

1. **动作在像素上很"小"**：手指几毫米的移动决定了接触成败，在图像里却只占很少的像素；
2. **决定结果的事件看不见**：接触的建立与释放、滑移、受力，视觉上几乎无法观测；
3. **交互动力学复杂**：多指、多接触、物体还可能是铰接或可变形的。

所以这一章的核心问题是：**动作和接触该怎么表示，才能让模型真的学到"手的动作如何通过接触改变世界"**。

### 10.2 三种模型形式 {#ch10-2}

| 形式 | 预测什么 | 优点 | 代价 |
|---|---|---|---|
| 隐空间动力学 | 编码 $z_t = \operatorname{enc}(o_t)$，学 $z_{t+1} = g(z_t, a_t)$，外加奖励、终止等预测头 | 快，适合做规划与策略的上下文 | 隐空间的含义难检查 |
| 动作条件的视频生成 | 未来的图像帧（通常是扩散模型） | 能利用大规模视频预训练，可视化直观 | 慢；"看起来合理"不等于物理上正确 |
| 世界-动作模型（WAM） | 同时生成未来观测与动作 | 预测与决策共享表示 | 推理慢，需要与快速执行解耦（第 8 章） |

### 10.3 让模型真的"懂手"：动作与接触的表示 {#ch10-3}

**用手的几何表示动作。** 人类视频里没有机器人动作，但有手。[DexWM](https://arxiv.org/abs/2512.13644) 用从第一人称视频中提取的**手指关键点**表示动作，因此能在 900 多小时的人类与非灵巧机器人数据上训练；它还发现，只预测视觉特征不足以学准手的动作，需要一个辅助的"手部一致性"损失，迫使预测的手部状态准确。[Dexterous World Models](https://arxiv.org/abs/2512.17907) 则把动作渲染成手部网格图像作为条件，同时以沿给定相机轨迹渲染的静态 3D 场景为条件，生成手与场景交互的视频——让静态的数字孪生变得可交互。

**把触觉当作要预测的世界状态。** 一个关键的设计选择是：触觉是**输入条件**，还是**预测目标**？

- [DexTacWAM](https://arxiv.org/abs/2609.24976) 把每个指尖的触觉单独编码，再经一个感知手指身份与位姿的压缩器聚合，注入视频扩散世界模型，联合建模视觉与触觉的演化。在 22 自由度双手平台的 6 个接触丰富的任务上，平均得分 70.6，最强基线 38.0；消融把提升归因于**把接触的演化作为被预测的世界状态的一部分**，而不只是把触觉当作条件输入。
- [ME-Dex](https://arxiv.org/abs/2609.21449) 持同样的观点——触觉和视频一样是对世界状态的观测，应当作为未来观测与视频一起预测——用视频、触觉、动作三个专家的混合 Transformer、以流匹配联合训练。
- [TacWAM](https://arxiv.org/abs/2607.28391) 进一步要求：被预测的触觉要携带有物理意义的信息（力场、形变流），同时不能变成动作生成的"特权线索"——否则模型会学会抄近路，部署时却拿不到这些线索。

**触觉也能帮助视觉预测。** [Dexterous Tactile World Model](https://arxiv.org/abs/2609.34286) 以双手数据手套的触觉信号为条件，预测第一人称的未来视频：触觉通过零初始化的残差、加在视频 token 中对应手的位置上，因果掩码防止偷看未来。与结构、参数、训练完全相同的纯视觉模型相比，对手部运动的低估从 23% 降到 9%——接触信号告诉模型"手在用力推"，模型就不再把推动预测成静止。

### 10.4 世界模型怎么用 {#ch10-4}

| 用法 | 做法 | 代表 |
|---|---|---|
| 作为策略的上下文 | 学动作条件的隐动力学，把它的隐状态作为循环上下文提供给策略，而不是拿来想象或规划 | [ WM-Craftnet](https://arxiv.org/abs/2609.07002) |
| 作为安全预测器 | 隐世界模型加一个"人会不会在未来接管"的预测头，把二值的人类干预变成前瞻的风险信号 | [ Intervention-Aware WM](https://arxiv.org/abs/2609.06009) |
| 作为闭环的改进器 | 同一个模型提供策略、模拟器、评估器三个接口：策略提出动作块，模拟器预测后果，评估器打分 | [ Motus2](https://arxiv.org/abs/2608.30237) |
| 作为规划的模型 | 在模型里前推候选动作序列，用奖励或价值头打分，取最优（第 6 章的采样 MPC，换成学到的模型） | 通用做法 |

WM-Craftnet 的设定很实用：从本体、深度、触觉和动作学一个紧凑的动作条件隐动力学，用多模态重建与奖励预测做监督（深度目标用干净的深度、输入用带噪的深度，迫使模型去噪）；然后把它作为非对称 actor-critic 的循环上下文，提升手内操作在位姿偏移、外力扰动和物体变化下的鲁棒性。Intervention-Aware WM 的四个预测头——动力学、奖励、终止、干预概率——则把第 9 章里"人的干预"从事后纠正变成了可以提前规避的风险。

```python
import numpy as np

def plan_with_world_model(z0, dynamics, value, n=512, horizon=8, iters=4, elite=32,
                          act_dim=16, rng=np.random.default_rng()):
    """在学到的隐动力学里用 CEM 做规划（参数为示意）。
    dynamics(z, a) -> z'；value(z) -> 预测的回报或成功概率（世界模型的预测头）。"""
    mu, std = np.zeros((horizon, act_dim)), np.ones((horizon, act_dim)) * 0.5
    for _ in range(iters):
        A = np.clip(mu + std * rng.standard_normal((n, horizon, act_dim)), -1, 1)
        z, score = np.repeat(z0[None], n, 0), np.zeros(n)
        for t in range(horizon):
            z = dynamics(z, A[:, t])
            score += value(z)
        top = A[np.argsort(-score)[:elite]]
        mu, std = top.mean(0), top.std(0) + 1e-3
    return mu[0]          # 只执行第一步，下一个控制周期重新规划
```

### 10.5 评估：看起来对，不等于物理上对 {#ch10-5}

视频世界模型最常见的失败是"视觉上合理、物理上错误"：物体穿过手指、接触没发生物体却动了、手明明在推物体却静止。评估至少要问三个问题：

1. **对动作敏感吗？** 换一个动作序列，预测是否相应改变？只学会"续写视频"的模型对动作不敏感。
2. **接触与物体运动对吗？** 例如手部运动是否被低估（上面的 23% → 9%）、物体是否在被接触前就开始运动。
3. **对下游有用吗？** 用它做规划、评估或数据生成时，任务成功率是否提升——这才是最终标准。跨本体的生成（把人手视频转成机器人视频）则有专门的基准 [H2R-Bench](https://arxiv.org/abs/2608.13049)。

### 10.6 小结 {#ch10-6}

- 灵巧世界模型的难点在动作"小"、接触"看不见"；手部几何（关键点、网格渲染）是有效的动作表示，辅助的手部一致性损失能防止模型忽略手。
- 把触觉作为**被预测的世界状态**而不仅是条件输入，是目前提升最明显的设计；但要防止触觉变成动作生成的特权线索。
- 用法比模型本身更多样：策略上下文、安全预测、闭环改进、规划；评估要看动作敏感性与物理正确性，最终以下游任务为准。

### 10.7 自测 {#ch10-7}

<details class="quiz"><summary><span class="qtag">自测</span>为什么"把触觉作为预测目标"比"只把触觉作为输入条件"更有效？</summary><p>作为条件输入时，模型只需在已知当前触觉的情况下预测未来画面，接触如何演化并不是它必须理解的东西；作为预测目标时，模型必须预测接触何时建立、力如何变化，这迫使它在内部表示里建模接触动力学——而接触动力学正是决定灵巧操作结果的部分。DexTacWAM 的消融把主要提升归因于此。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>TacWAM 说被预测的触觉"不能变成动作生成的特权线索"，这是什么意思？</summary><p>如果动作生成在训练时依赖了未来触觉的真值（或从中泄漏的信息），模型会学会"抄答案"：靠这些线索直接推断该做的动作。部署时未来触觉并不存在，这条捷径就断了，性能大幅下降。所以要让触觉预测服务于对物理的理解，而不是成为动作生成的输入捷径。</p></details>

## 11. Agent × 灵巧手：自动化研发外环 {#ch11}

### 11.1 问题：agent 应该放在哪一层 {#ch11-1}

前十章里，灵巧手系统有两个嵌套的循环：**内环**是 10–1000 Hz 的控制（策略、PD、MPC）；**外环**是研究者的工作——写奖励、搭环境、调随机化、生成数据、训练、看曲线、再改。外环慢（以小时计）、离散、充满"写代码"和"读结果"，恰好是大语言模型擅长的事。

把外环形式化：设 $R$ 是外环要设计的对象（奖励函数、环境、随机化参数、数据……），$A(R)$ 是用它训练出的策略，$F$ 是任务的适应度（例如成功率）。外环要解

$$\max_{R}\; F\bigl( A(R) \bigr)$$

这是一个在**程序空间**上的黑盒优化：每次评估都要训练一个策略（昂贵、有噪声），而且 $F$ 只告诉你"好不好"，不告诉你"为什么"。所以这一章的核心问题是两个：**怎么高效地搜索**，以及**怎么可靠地验证**。

<figure class="fig"><div class="fig-scroll"><svg viewBox="0 0 760 222" role="img" aria-label="agent 外环示意图"><defs><marker id="ah5" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="sv-head"/></marker></defs><rect class="sv-box" x="24" y="24" width="200" height="58" rx="10"/><text class="sv-t2" x="124" y="48" text-anchor="middle">任务描述 + 环境代码</text><text class="sv-t3" x="124" y="68" text-anchor="middle">作为上下文</text><rect class="sv-hl" x="280" y="24" width="200" height="58" rx="10"/><text class="sv-t2" x="380" y="48" text-anchor="middle">LLM / 编码 agent</text><text class="sv-t3" x="380" y="68" text-anchor="middle">生成 K 个候选：奖励、环境、参数</text><rect class="sv-box" x="536" y="24" width="200" height="58" rx="10"/><text class="sv-t2" x="636" y="48" text-anchor="middle">校验器</text><text class="sv-t3" x="636" y="68" text-anchor="middle">能否执行 · 物理是否合理</text><rect class="sv-box" x="536" y="140" width="200" height="58" rx="10"/><text class="sv-t2" x="636" y="164" text-anchor="middle">并行训练与评估</text><text class="sv-t3" x="636" y="184" text-anchor="middle">GPU 仿真里训练每个候选</text><rect class="sv-box" x="280" y="140" width="200" height="58" rx="10"/><text class="sv-t2" x="380" y="164" text-anchor="middle">反思</text><text class="sv-t3" x="380" y="184" text-anchor="middle">各奖励分量与适应度的曲线</text><rect class="sv-box" x="24" y="140" width="200" height="58" rx="10"/><text class="sv-t2" x="124" y="164" text-anchor="middle">人工把关 → 真机</text><text class="sv-t3" x="124" y="184" text-anchor="middle">限位 · 急停 · 小范围试验</text><path class="sv-edge" d="M224,53 L278,53" marker-end="url(#ah5)"/><path class="sv-edge" d="M480,53 L534,53" marker-end="url(#ah5)"/><path class="sv-edge" d="M636,82 L636,138" marker-end="url(#ah5)"/><path class="sv-edge" d="M536,169 L482,169" marker-end="url(#ah5)"/><path class="sv-edge" d="M380,140 L380,84" marker-end="url(#ah5)"/><path class="sv-edge dash" d="M280,169 L226,169" marker-end="url(#ah5)"/><text class="sv-t3" x="392" y="118">改进</text><text class="sv-t3" x="253" y="160" text-anchor="middle">最优</text></svg></div><figcaption>图 5　agent 所在的外环。生成候选很便宜，评估很贵，验证决定了整个循环是否可靠；内环的高频控制仍由策略或 MPC 负责。</figcaption></figure>

### 11.2 搜索奖励：Eureka 的三个组件 {#ch11-2}

[Eureka](https://arxiv.org/abs/2310.12931) 是这类方法的原型，它的三个组件分别回答了"怎么生成、怎么搜索、怎么反馈"：

1. **以环境为上下文**：把环境的源代码直接交给 LLM，零样本生成可执行的奖励代码，并要求奖励把各个分量分开返回（为第 3 点做准备）。
2. **进化搜索**：每轮从 LLM 独立采样 $K$ 个奖励（$K = 16$ 就足以保证第一轮至少有一个能执行），全部在 GPU 仿真里训练、按适应度 $F$ 排序；下一轮以最好的奖励及其"反思"为上下文，让 LLM 生成改进版本。每次运行 5 轮，独立重复 5 次取最好。
3. **奖励反思**：只把适应度 $F$ 反馈给 LLM 是不够的——它不告诉你奖励**为什么**好或不好。Eureka 把训练过程中各个奖励分量和适应度在若干检查点上的数值，整理成文字反馈。理由有二：适应度本身缺乏信用分配；同一个奖励在不同 RL 算法和超参数下表现差异很大，只有看到各分量实际被优化成什么样，才能有针对性地修改。

在灵巧手上，Eureka 第一次让仿真里的 Shadow 手学会了快速转笔：先用 Eureka 奖励预训练"把笔转到随机目标姿态"的策略，再用同一个奖励微调到一串转笔姿态。

```python
def eureka(llm, env_source, task, train_and_eval, K=16, iters=5):
    """Eureka 式奖励搜索（结构取自论文；llm、train_and_eval 由使用者提供）。
    train_and_eval(code) 在 GPU 仿真里训练策略，返回适应度与各奖励分量在检查点上的取值。"""
    context = f"环境源代码：\n{env_source}\n任务：{task}\n请写出奖励函数，并分别返回各个分量。"
    best = None
    for _ in range(iters):
        results = []
        for code in llm.sample(context, n=K):          # 独立采样 K 个候选
            try:
                fitness, components = train_and_eval(code)
            except Exception as err:                     # 不能执行的候选也是有用的反馈
                results.append((float("-inf"), code, f"执行失败：{err}"))
                continue
            results.append((fitness, code, reflect(components, fitness)))
        top = max(results, key=lambda r: r[0])
        if best is None or top[0] > best[0]:
            best = top
        context = (f"上一轮最好的奖励：\n{top[1]}\n训练反馈：\n{top[2]}\n"
                   "请据此改进奖励函数。")
    return best

def reflect(components, fitness):
    """奖励反思：把各分量与适应度在若干检查点上的数值写成文字。"""
    lines = [f"{name}: {', '.join(f'{v:.3f}' for v in values)}" for name, values in components.items()]
    return "\n".join(lines + [f"task fitness: {', '.join(f'{v:.3f}' for v in fitness)}"])
```

两个后续工作沿着"搜索"和"模态"扩展：[RF-Agent](https://arxiv.org/abs/2602.23876) 把奖励设计看成序列决策，用蒙特卡洛树搜索管理迭代，更好地利用历史上所有尝试（而不只是上一轮最好的那个）；[Text2Touch](https://arxiv.org/abs/2509.07445) 把 LLM 奖励扩展到触觉，提示结构能处理 70 多个环境变量，在掌心朝上与朝下两种设置下做多轴手内旋转，经 sim2real 蒸馏迁移到带视触觉的真实四指手——转速和稳定性都超过精心调过的人工奖励，而奖励代码短了一个数量级。

### 11.3 不止奖励：外环里还有什么可以交给 agent {#ch11-3}

| 外环的对象 | 做法 | 代表 |
|---|---|---|
| 参考轨迹（代替奖励） | VLM 从语言和图像中找出任务相关的关键点，合成手与物体的 3D 轨迹，残差 RL 在仿真中跟踪，迁移到真机，不需要示范，也不需要手写奖励 | [ Scaffolding with VLMs](https://arxiv.org/abs/2506.19212) |
| 域随机化参数 | VLM 当"视觉真实感评判"，闭环 CMA-ES 优化仿真参数 | [ DexSim2Real](https://arxiv.org/abs/2605.05241) |
| MPC 的代价函数 | 给 VLM 任务描述和手当前行为的视频，让它逐步改 MuJoCo MPC 的目标函数，每轮不到两分钟 | [ 腱驱动手采样 MPC](https://arxiv.org/abs/2411.06183) |
| 任务与环境 | 生成灵巧任务与场景，用 VLM 反馈调整物体摆放与尺度，把任务拆成子任务顺序训练 | [ GenDexHand](https://arxiv.org/abs/2511.01791) |
| 训练数据 | 从一段第一人称视频重建仿真、优化轨迹、生成数据，每一步都有属性相关的校验器 | [ DexAgent](https://arxiv.org/abs/2609.35318) |
| 编码 agent 的解 → VLA 数据 | 编码 agent 解出长时程任务后，把单个解扩增成大量多样的轨迹，训练 VLA | [ EmbodiedSWE](https://arxiv.org/abs/2609.27308) |

Scaffolding with VLMs 值得单独一提：它把第 4 章的"跟踪 RL"和 agent 接起来——演示不再来自人，而是 VLM 根据常识合成的粗略"脚手架"；RL 负责把粗略轨迹修成物理可行的动作。这说明了一个分工原则：**大模型提供语义与粗略几何，物理细节交给 RL 或 MPC**。

这些工作的共同点是：**生成很便宜，验证才是瓶颈**。DexAgent 在四个阶段（语义理解、按物体属性重建仿真、轨迹优化、数据生成）的每一步都设了与物体属性相关的校验器，检查物理有效性与任务要求，不合格就反馈修改——防止错误沿管线传下去。它的工具库可以自我扩充：遇到现有工具处理不了的物体或交互时，agent 会自己开发新工具。在 11 个真机任务上，用它的数据训练出的策略成功率是基线的 3.5 倍。

### 11.4 编码 agent 当研究员 {#ch11-4}

更进一步，让编码 agent 自己搭仿真、写奖励、训练、看曲线、修改——接管整个外环。

- [Dexterous Astra](https://github.com/jianglongye/dexterous-astra)：GPT-6 Astra 在 Isaac Lab 中让 Sharpa Wave 手转笔三圈，在 MuJoCo 中让 Wuji 手拧出魔方的 U、F、L 三步，提示词与检查点都已公开。
- [ASPIRE](https://arxiv.org/abs/2607.00272)：以"代码即策略"的方式持续编写、修复控制程序。三个组件：一个闭环执行引擎，暴露细粒度的多模态执行轨迹，让 agent 能自己诊断失败、合成修复、验证结果；一个不断扩充的技能库，把验证过的修复沉淀成可复用的知识；一个进化搜索，生成多样的程序并系统地调试，探索单条轨迹修补之外的解。
- [SEAR](https://arxiv.org/abs/2610.18792)：系统评测"自我进化"的 agent，设计了对照实验来回答两个问题——agent 是真的在学，还是只是多试了几次？学到的东西换任务、换仿真器、上真机还有没有用？在 7 个前沿模型、330 个仿真任务和 5 个真机任务上，它发现：学习曲线取决于总时间预算和"任务难度相对于模型能力"的比值——难的任务值得长时间的会话，简单的任务反复重试短会话效果一样；控制代码和共享的例程能跨任务、跨仿真器迁移。

**搭一个让编码 agent 训练灵巧策略的环境，至少要有：**

1. **工具**：启动训练、读日志与曲线、渲染回放视频、修改环境与奖励；
2. **与训练奖励分离的自动评估器**：agent 会优化你给它的任何数字，评估器和训练奖励是同一个函数时，它会直接"刷"奖励；
3. **预算**：算力、时间、尝试次数——SEAR 的结论说明，预算本身决定了 agent 的学习曲线；
4. **沙箱**：代码在隔离环境里运行，不能直接访问真机；
5. **人工把关**：任何策略上真机之前，经过人工检查、限位与急停。

### 11.5 大模型能进内环吗 {#ch11-5}

直接让大模型逐步输出手指动作，目前效果不好：[Galbot 的系统评测](https://arxiv.org/abs/2609.38537)显示，与学习策略混合控制时，在 DexJoCo 的 10 次试验中成功 5 次；而直接控制手内操作时，难以协调手指之间的接触。

但给它**合适的上下文**，情况会改变。[GPT-6-Astra Understands Touch](https://arxiv.org/abs/2610.28457) 先把专家示范总结成任务指导和触觉参考（什么样的触觉信号意味着对准、卡住、插进去了），再把这些上下文连同多模态观测与交互历史交给冻结的 GPT-6 Astra，让它生成控制指令、并根据接触反馈调整。不做任何机器人专用的预训练或逐任务微调，它在 5 个 UniVTAC 任务上平均成功率 93.8%，与最好的专用策略相当；在插孔任务上，触觉反馈把成功率从 31% 提到 99%。

两者对比得出的原则是：**让大模型做"事件级"的决策**——对准了没有、是否卡住、该不该换一种插入方式——而不是逐步输出手指的关节目标；高频的接触执行交给策略或控制器。

### 11.6 小结 {#ch11-6}

- agent 的位置在外环：在奖励、环境、参数、数据等"程序空间"上做昂贵、带噪声的黑盒优化。
- 搜索靠"生成多个候选 + 进化或树搜索"；反馈要包含信用分配信息（奖励反思），而不只是一个分数。
- 生成很便宜，验证是瓶颈：校验器、与训练奖励分离的评估器、沙箱与人工把关，决定了整个循环是否可靠。
- 大模型适合做事件级的接触决策，不适合逐步控制手指；把触觉等关键信息整理成它能理解的上下文，效果会显著不同。

!!! pitfall "常见坑"
    - 评估器和训练奖励是同一个函数：agent 会找到奖励的漏洞（比如让物体在仿真里穿透手指来"转动"），适应度很高，行为毫无用处。
    - 不看行为只看数字：每个候选都要渲染回放视频，人或 VLM 抽查。
    - 让 agent 直接操作真机：生成的代码可能越限、发出危险指令。
    - 只把"最好的一个"留作上下文：失败的尝试同样携带信息，RF-Agent 的树搜索就是为了利用整个历史。

### 11.7 自测 {#ch11-7}

<details class="quiz"><summary><span class="qtag">自测</span>为什么 Eureka 要把各奖励分量的训练曲线反馈给 LLM，而不只是最终成功率？</summary><p>最终成功率只说明"好不好"，没有信用分配：LLM 不知道是哪一项权重太大、哪一项根本没被优化。各分量的曲线能告诉它，例如"旋转奖励一直在涨，但掉落惩罚也一直很大"，从而有针对性地修改。而且同一个奖励在不同算法和超参数下表现差异很大，只有看到实际的优化过程，修改才有依据。</p></details>

<details class="quiz"><summary><span class="qtag">自测</span>Galbot 的评测和 GPT-6-Astra Understands Touch 的结论看起来矛盾，如何理解？</summary><p>两者测的是不同的用法。前者让大模型直接、逐步地协调手指接触，这需要高频、连续的精细控制，正是大模型不擅长的。后者把专家知识整理成任务指导和触觉参考，让模型根据接触反馈做纠正决策，相当于把它放在"事件级"的位置。结论是一致的：大模型擅长理解接触信号并做离散的纠正决策，不擅长替代高频的接触控制。</p></details>

## 12. 串起来：一条可执行的路线 {#ch12}

前面各章各自解决一个核心问题。这一章把它们排成一条从零到真机的路线：每个阶段给出目标、关键决策，以及进入下一阶段前应满足的验收标准。顺序本身就是经验——**先把被控对象建对，再谈学习；先在仿真里闭环，再上真机；先有可靠的单任务，再谈数据与大模型**。

### 12.1 阶段一：被控对象（第 1–2 章） {#ch12-1}

- **目标**：一个与真机一致的仿真模型，以及可信的低层控制。
- **关键决策**：选手（能否反驱、能否从电流估力矩、有无可靠的仿真资产）；接触求解器与参数（MuJoCo 的 solref/solimp、condim，或 PhysX 的迭代次数与子步）；执行器模型（理想 PD、带饱和与摩擦的模型，还是执行器网络）。
- **验收标准**：悬空激励下，同一段指令序列在仿真与真机上的关节轨迹吻合（跟踪误差、相位延迟、稳态误差都在可接受范围）；关节零位、限位、耦合与真机一致；控制频率与延迟已测量。

### 12.2 阶段二：仿真里学出控制（第 3 章） {#ch12-2}

- **目标**：在仿真里稳定解决目标任务，并对随机化不敏感。
- **起点**：MuJoCo Playground（`pip install playground`，含灵巧操作环境）或 Isaac Lab 的手内重定向任务；手的模型可以从 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 取。
- **关键决策**：奖励结构（有界任务项 + 正则）；初始状态分布（从稳定抓取开始）；动作参数化（相对目标 + 低通）；观测（相对手腕的表示）；教师-学生的结构（回归隐变量，还是直接蒸馏动作）。
- **验收标准**：学生策略在训练分布内接近教师；逐类关掉随机化、测分布外物体，性能下降在预期内；隐变量随物体属性变化。

### 12.3 阶段三：上真机（第 3 章） {#ch12-3}

- **目标**：零样本迁移，或经少量调整后可用。
- **部署前检查**：

| 项目 | 检查什么 |
|---|---|
| 观测 | 归一化统计量随策略一起冻结并部署；坐标系、单位、关节顺序（按名字映射）与仿真一致 |
| 时序 | 实际控制频率与延迟落在训练时随机化的范围内 |
| 低层 | PD 增益、力矩上限与仿真（或随机化中心）一致；有位置、速度、力矩硬限位 |
| 动作 | 低通滤波系数在真机上重新确认（DeXtreme 真机最优值与训练时不同） |
| 安全 | 急停、温度监测；第一次运行用低增益、低速、可接住物体的设置 |

- **失败时的排查顺序**：执行器与延迟 → 接触参数（摩擦、几何） → 观测（噪声、遮挡、跳变） → 随机化范围 → 策略本身。经验上，大部分 sim2real 失败出在前三项。

### 12.4 阶段四：扩展任务与数据（第 4–8 章） {#ch12-4}

按任务性质选路线：

- 目标好写、回合不长（旋转、重定向、抓取）：继续用仿真 RL，必要时用 MPC（第 6 章）提供探索或示范；
- 长时程、难写奖励：人类演示 → 重定向 → 跟踪 RL（第 4 章），用 [dex-retargeting](https://github.com/dexsuite/dex-retargeting)（`pip install dex_retargeting`）起步；
- 接触状态看不见、对滑移敏感：加触觉，优先考虑不依赖逼真触觉仿真的迁移方法（第 7 章）；
- 多任务、语言条件、跨场景：以前面产生的数据为基础训练生成式策略或 VLA，接触执行交给基础控制器（第 8 章）。

**验收标准**：每条路线都要有明确的真机评测协议（任务定义、物体集合、试验次数、成功判定），与仿真指标分开报告。

### 12.5 阶段五：真机后训练（第 9 章） {#ch12-5}

- **目标**：把 60–80% 的策略推到可靠。
- **关键决策**：先验从哪来（人工干预、MPC 脚手架，还是在已有策略上学残差）；奖励怎么从真实传感器里算；复位怎么自动化。
- **验收标准**：只用策略、关掉引导与探索噪声的评估达到目标成功率；训练中的掉落与人工干预次数随时间下降。

### 12.6 阶段六：自动化外环（第 10–11 章） {#ch12-6}

- **目标**：让奖励、环境、随机化参数、数据的迭代由 agent 完成，人只负责定义任务与把关。
- **前提**：前五个阶段的流程已经稳定、可以脚本化——agent 只能自动化一个本身已经可重复的流程。
- **关键决策**：评估器与训练奖励分离；每个候选都要有可检查的产物（曲线、回放视频）；预算与沙箱。
- **验收标准**：agent 产出的方案在独立的评估器上优于人工基线，且行为经过人工抽查。

### 12.7 全书小结 {#ch12-7}

1. 灵巧操作控制的是一个没有执行器的物体，一切都经由单边、带摩擦的接触；把被控对象建对，是一切方法的前提。
2. 仿真 RL 能学得动，靠的是结构：有界的任务项与塑造步态的正则、绕开探索的初始状态、把"推断"交给监督学习的两阶段。
3. sim2real 的本质，是把每一类差距写成可随机化的参数，让真机落进训练分布。
4. 演示解决奖励与探索；目标要以物体为中心，物理可行性交给仿真里的跟踪 RL。
5. 接触的非光滑性可以平滑（RL 就是一种随机平滑），也可以绕开（采样 MPC）；两者与学习互补。
6. 触觉提供视觉最缺的接触状态，代价是难仿真、难迁移；不仿真触觉的迁移框架往往更务实。
7. 数据驱动的大模型擅长"做什么"，接触执行要交给低层的基础控制器，力的缺失要靠物理与真机来补。
8. 真机 RL 的关键是先验与安全；agent 的位置在外环，可靠性取决于验证。
