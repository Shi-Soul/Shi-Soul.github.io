# 扩散模型与流匹配：原理、技术路线与 2025–2026 前沿（长文综述）

> 数据截至 2026-10-30。本文与[交互式导读页](./)共用同一份数据：收录 717 条论文与资源，其中 695 条 arXiv 条目的标题与日期经 export.arxiv.org API 核验，298 条发表于 2025-10 之后（"最近一年"）。文中数值均摘自论文摘要或官方发布，不同实验设置不同，只宜看量级。附录 A 是按类别分组的完整论文表。

**本文怎么读。**第 1 节用一套记号把扩散模型、分数模型与流匹配统一起来，是后文的基础；第 2–8 节按"训练目标 → 采样 → 少步 → 引导 → 架构 → 潜空间 → 后训练"梳理方法；第 9–11 节讲视频与世界模型、扩散语言模型、统一多模态与跨领域应用；第 12–13 节是理论与评测；第 14 节按月份列出最近一年的大事；第 15 节是开放问题。只关心最新进展，可以直接读每节末尾的"最近一年"和第 14 节。

---

## 1 统一框架

### 1.1 前向过程：把数据逐渐变成噪声

约定时间 $t\in[0,1]$，$t=0$ 是数据分布 $p_0$，$t=1$ 是标准高斯 $\mathcal N(0,I)$。几乎所有连续扩散 / 流方法都可以写成

$$x_t=\alpha_t\,x_0+\sigma_t\,\varepsilon,\qquad x_0\sim p_0,\ \varepsilon\sim\mathcal N(0,I)$$

其中 $\alpha_t$ 从 1 递减到（接近）0，$\sigma_t$ 从 0 递增。这对系数决定了一族边缘分布 $p_t$，称为**概率路径**或**噪声表**。常用的信噪比是 $\mathrm{SNR}(t)=\alpha_t^2/\sigma_t^2$。几种经典选择：

| 框架 | $\alpha_t$ | $\sigma_t$ | 代表 |
|---|---|---|---|
| VP（DDPM） | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | [DDPM](https://arxiv.org/abs/2006.11239)、[Stable Diffusion](https://arxiv.org/abs/2112.10752) |
| VE（NCSN） | $1$ | 由小到大 | [NCSN](https://arxiv.org/abs/1907.05600)、[Score SDE](https://arxiv.org/abs/2011.13456) |
| EDM | $1$ | $t$ | [EDM](https://arxiv.org/abs/2206.00364) |
| 流匹配 / Rectified Flow | $1-t$ | $t$ | [Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003)、[SD3](https://arxiv.org/abs/2403.03206) |

前向过程不需要学习，训练时可以对任意 $t$ 一步采样 $x_t$，这正是扩散类模型训练简单、稳定的原因。

### 1.2 四种参数化及其换算

网络 $f_\theta(x_t,t)$ 可以预测四个量中的任意一个，它们之间只差与 $t$ 有关的线性变换：

- **噪声** $\varepsilon$：DDPM 的选择。由 $\hat\varepsilon$ 得到 $\hat x_0=(x_t-\sigma_t\hat\varepsilon)/\alpha_t$。
- **数据** $x_0$：由 $\hat x_0$ 得到 $\hat\varepsilon=(x_t-\alpha_t\hat x_0)/\sigma_t$。
- **速度** $v=\dot\alpha_t x_0+\dot\sigma_t\varepsilon$：流匹配的选择；对直线路径 $\alpha_t=1-t,\ \sigma_t=t$，有 $v=\varepsilon-x_0$。
- **分数** $s=\nabla_x\log p_t(x)$：高斯路径下 $s(x_t)=-\mathbb E[\varepsilon\mid x_t]/\sigma_t$。

以直线路径为例：已知速度预测 $\hat v$，可得 $\hat\varepsilon=x_t+(1-t)\hat v$、$\hat x_0=x_t-t\hat v$，分数 $\hat s=-\bigl(x_t+(1-t)\hat v\bigr)/t$。所以"扩散模型"和"流匹配模型"可以互相改写；在无限容量下学哪个量都一样，有限容量下差别体现在各时间段的误差分布：$\varepsilon$ 预测在低噪声端稳定、$x_0$ 预测在高噪声端稳定、$v$ 预测两端折中。[JiT](https://arxiv.org/abs/2511.13720) 进一步指出，在高维像素空间中网络更擅长预测位于低维流形上的干净图像，而非充满高频的噪声，因此 $x$ 预测明显更好。

### 1.3 训练目标：一族加权回归

所有参数化都对应一个回归损失。以噪声预测为例：

$$\mathcal L_\varepsilon(\theta)=\mathbb E_{t,x_0,\varepsilon}\Bigl[w(t)\,\bigl\lVert\varepsilon_\theta(x_t,t)-\varepsilon\bigr\rVert^2\Bigr]$$

它等价于**去噪分数匹配**（对分数做回归），在特定权重下又等价于 ELBO。流匹配则直接回归条件速度场：

$$\mathcal L_{\mathrm{FM}}(\theta)=\mathbb E_{t,x_0,\varepsilon}\Bigl[\bigl\lVert v_\theta(x_t,t)-(\varepsilon-x_0)\bigr\rVert^2\Bigr]$$

关键结论是：**条件**速度 $\varepsilon-x_0$ 的期望正好是**边缘**速度场 $v_t(x)=\mathbb E[\varepsilon-x_0\mid x_t=x]$，所以回归条件目标就能学到正确的边缘速度场（[Lipman 等](https://arxiv.org/abs/2210.02747)）。[Kingma 与 Gao](https://arxiv.org/abs/2303.00848) 证明，各种常用加权（包括 $v$ 预测、Min-SNR、SD3 的 logit-normal 时间采样）都对应"带数据增强的 ELBO"，差别只在于更看重哪些噪声水平。实践中最影响效果的往往是**时间采样分布与损失加权**，而非"扩散还是流匹配"这一名称。

### 1.4 采样：概率流 ODE 与 SDE

学到速度场后，从 $x_1\sim\mathcal N(0,I)$ 出发，把常微分方程

$$\frac{\mathrm d x_t}{\mathrm d t}=v_\theta(x_t,t)$$

从 $t=1$ 积分到 $t=0$ 即得样本。这就是**概率流 ODE**（[Score SDE](https://arxiv.org/abs/2011.13456)）。它确定、可逆，使"噪声 ↔ 样本"成为双射，因此支持反演编辑、似然计算和蒸馏。

同一族边缘分布也可以用随机过程实现。在本文的前向时间约定下，对任意扩散系数 $g(t)\ge 0$，

$$\mathrm d x=\Bigl[v_t(x)+\tfrac12 g(t)^2\,\nabla_x\log p_t(x)\Bigr]\mathrm d t+g(t)\,\mathrm d W_t$$

与 ODE $\mathrm dx=v_t(x)\,\mathrm dt$ 具有完全相同的边缘分布（代入 Fokker–Planck 方程即可验证）；生成时沿反向时间积分其对应的反向 SDE，漂移变为 $v_t-\tfrac12 g^2\nabla\log p_t$。$g=0$ 退化为 ODE；$g>0$ 时每步注入新噪声，可以"冲掉"前面步骤的误差，但通常需要更多步。最近一年这一自由度变得重要：[Flow-GRPO](https://arxiv.org/abs/2505.05470) 正是借助"边缘不变的 SDE"给确定性的流匹配模型引入随机策略，才能做在线强化学习；[流映射测试时扩展](https://arxiv.org/abs/2511.22688) 与 [Adjoint Matching](https://arxiv.org/abs/2409.08861) 也都建立在这种随机最优控制的视角上。

### 1.5 条件生成与引导

条件 $c$（类别、文本、图像）直接作为网络输入。**分类器引导**（[ADM](https://arxiv.org/abs/2105.05233)）用一个噪声条件分类器的梯度修正分数；**无分类器引导**（[CFG](https://arxiv.org/abs/2207.12598)）在训练时随机丢弃条件，采样时外推：

$$\tilde v=v_\theta(x_t,\varnothing)+w\,\bigl(v_\theta(x_t,c)-v_\theta(x_t,\varnothing)\bigr),\qquad w>1$$

直观上，CFG 近似从 $p(x)\,p(c\mid x)^{w}$ 采样，以多样性换取条件贴合。但 [最近的分析](https://arxiv.org/abs/2607.19725)表明，确定性 CFG 动力学实际采到的分布并不是这个乘积形式，这解释了高 $w$ 下的过饱和与模式坍缩，也是 APG、引导区间、Autoguidance 等改进的出发点（见第 5 节）。

### 1.6 潜空间扩散

像素空间里一张 1024×1024 图像有三百多万个维度，直接扩散代价很高。[LDM](https://arxiv.org/abs/2112.10752) 先训练自编码器把图像压缩为 $8\times$ 下采样的潜变量，再在潜空间做扩散，这成为 Stable Diffusion 以来的标准做法；视频模型进一步做时空联合压缩（如 $4\times8\times8$）。压缩器的设计——压缩率、通道数、潜空间是否"好扩散"——是最近一年最活跃的方向之一（第 7 节）。

---

## 2 训练目标与概率路径的演进

**奠基期。**[Sohl-Dickstein 等](https://arxiv.org/abs/1503.03585)在 2015 年提出用非平衡热力学的思路逐步破坏并还原数据；[NCSN](https://arxiv.org/abs/1907.05600) 用多尺度去噪分数匹配学习分数并以退火 Langevin 采样；[DDPM](https://arxiv.org/abs/2006.11239) 发现"预测噪声 + 均方误差"这一简化目标足以生成高质量图像；[Score SDE](https://arxiv.org/abs/2011.13456) 把两者统一到连续时间的 SDE，给出概率流 ODE。[VDM](https://arxiv.org/abs/2107.00630) 从 SNR 角度推导连续时间 ELBO；[EDM](https://arxiv.org/abs/2206.00364) 把设计空间拆成噪声分布、预条件、损失权重与采样器等独立模块，成为此后多年的默认配方，[EDM2](https://arxiv.org/abs/2312.02696) 又改进了训练动态。

**流匹配。**[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003) 与 [Stochastic Interpolants](https://arxiv.org/abs/2209.15571) 在 2022 年几乎同时提出"直接回归速度场"，摆脱了 SDE 推导，也自然容纳非高斯源分布与任意耦合（如小批量最优传输 [OT-CFM](https://arxiv.org/abs/2302.00482)）。Rectified Flow 的 reflow 操作反复用模型自身生成的配对重新训练，让轨迹变直，为一步生成铺路。[随机插值统一框架](https://arxiv.org/abs/2303.08797)与 [Generator Matching](https://arxiv.org/abs/2410.20587) 进一步把扩散、流、跳跃过程统一为"匹配一个马尔可夫过程的生成元"。2024 年 [SD3](https://arxiv.org/abs/2403.03206) 证明 Rectified Flow + logit-normal 时间采样在大规模文生图上优于传统扩散，此后 FLUX、HunyuanVideo、Wan、Qwen-Image 等主流模型普遍采用流匹配。

**非欧与离散空间。**流匹配被推广到流形（[Riemannian FM](https://arxiv.org/abs/2302.03660)）、单纯形与离散状态（[Dirichlet FM](https://arxiv.org/abs/2402.05841)、[Discrete FM](https://arxiv.org/abs/2407.15595)、[Multiflow](https://arxiv.org/abs/2402.04997)、[Fisher Flow](https://arxiv.org/abs/2405.14664)），以及长度可变的编辑操作（[Edit Flows](https://arxiv.org/abs/2506.09018)）。薛定谔桥（[DSB](https://arxiv.org/abs/2106.01357)、[DSBM](https://arxiv.org/abs/2303.16852)、[I²SB](https://arxiv.org/abs/2302.05872)）处理任意两个分布之间的转换。[Transition Matching](https://arxiv.org/abs/2506.23589) 则把每一步建模为可学习的随机转移核，统一了扩散、流与自回归的单步生成。

**最近一年。**目标函数层面出现了几项"反常识"的工作：

- [无噪声条件](https://arxiv.org/abs/2502.13129)（2025-02）与 [JiT](https://arxiv.org/abs/2511.13720)（2025-11）：时间/噪声条件并非必需；在像素空间直接预测干净图像，大 patch 的纯 Transformer 就能训好。
- [Equilibrium Matching](https://arxiv.org/abs/2510.02300)：学习与时间无关的"平衡梯度"，相当于隐式能量模型，推理时做梯度下降，ImageNet-256 FID 1.90。
- [Drifting Models](https://arxiv.org/abs/2602.04770)：不再沿固定的概率路径回归，而是在训练过程中让生成器的推前分布本身"漂移"到数据分布，推理天然一步，ImageNet-256 潜空间 FID 1.54、像素空间 1.61。
- 归一化流复兴：[TarFlow](https://arxiv.org/abs/2412.06329)、[STARFlow](https://arxiv.org/abs/2506.06276)、[STARFlow-V](https://arxiv.org/abs/2511.20462)、[BiFlow](https://arxiv.org/abs/2512.10953)、[NF + 迭代去噪](https://arxiv.org/abs/2604.20041)，把"精确似然 + 一次前向"重新带回竞争。
- [TM vs FM](https://arxiv.org/abs/2510.17991) 分析了转移匹配何时、为何能在少步时胜过流匹配（目标多峰时）。

---

## 3 采样器：不重新训练也能更快

早期 DDPM 需要上千步。[DDIM](https://arxiv.org/abs/2010.02502) 构造非马尔可夫的确定性采样，把步数降到几十；[PNDM](https://arxiv.org/abs/2202.09778)、[DEIS](https://arxiv.org/abs/2204.13902) 引入高阶数值方法；[DPM-Solver](https://arxiv.org/abs/2206.00927) 利用扩散 ODE 的半线性结构，对线性部分精确积分、对非线性部分做泰勒展开，约 10 步即可高质量出图，[DPM-Solver++](https://arxiv.org/abs/2211.01095) 针对 CFG 做了数据预测形式的多步版本，[UniPC](https://arxiv.org/abs/2302.04867) 统一预测-校正框架。[Restart Sampling](https://arxiv.org/abs/2306.14878) 交替前向加噪与反向 ODE，结合 ODE 的速度与 SDE 的纠错能力。

流匹配模型的轨迹更直，简单的 Euler / Heun 加上"时间平移"（把网格向高噪声端偏移）就能在 20–50 步内取得良好效果，这也是开源文生图 / 视频模型的默认采样方式。最近一年专门的求解器研究减少，代表性工作是学习离散化参数并加入对抗损失的 [GAS](https://arxiv.org/abs/2510.17699)；更多精力转向"训练出能跳步的模型"（第 4 节）。

---

## 4 少步与一步生成

把 NFE 从几十降到 1–4，是扩散模型走向实时与端侧的关键，也是最近一年进展最快的方向。方法分两大类：**蒸馏**（需要一个多步教师）和**直接训练少步模型**（可以从头训练）。

### 4.1 蒸馏：轨迹、分布与对抗

- **轨迹蒸馏**：学生模仿教师 ODE 轨迹上的映射。[渐进蒸馏](https://arxiv.org/abs/2202.00512)每轮把两步蒸成一步；一致性蒸馏（[CM](https://arxiv.org/abs/2303.01469)、[LCM](https://arxiv.org/abs/2310.04378)）让学生把轨迹上任意点映射到同一终点；[PeRFlow](https://arxiv.org/abs/2405.07510)、[Hyper-SD](https://arxiv.org/abs/2404.13686) 做分段处理。
- **分布蒸馏**：不要求逐点对应，只要求学生分布与教师分布一致。[DMD](https://arxiv.org/abs/2311.18828) 用两个分数网络（真实分数 $s_{\mathrm{real}}$ 由教师给出，伪分数 $s_{\mathrm{fake}}$ 在学生样本上在线训练）估计 KL 散度的梯度：

$$\nabla_\theta D_{\mathrm{KL}}\approx\mathbb E\Bigl[\bigl(s_{\mathrm{fake}}(x_t,t)-s_{\mathrm{real}}(x_t,t)\bigr)^{\!\top}\frac{\partial G_\theta(z)}{\partial\theta}\Bigr]$$

其中 $x_t$ 是学生样本 $G_\theta(z)$ 加噪后的结果。[DMD2](https://arxiv.org/abs/2405.14867) 去掉回归损失、加入 GAN 损失后一步生成超过教师；[SiD](https://arxiv.org/abs/2404.04057) 从分数恒等式出发实现无数据蒸馏；其思想可追溯到文生 3D 的 [VSD](https://arxiv.org/abs/2305.16213)。
- **对抗蒸馏**：[ADD](https://arxiv.org/abs/2311.17042)（SDXL Turbo）、[LADD](https://arxiv.org/abs/2403.12015)（SD3 Turbo）、[SDXL-Lightning](https://arxiv.org/abs/2402.13929) 用判别器补足少步生成的细节。

**最近一年**蒸馏全面走向十亿级工业模型：[rCM](https://arxiv.org/abs/2510.08431) 给连续时间一致性模型加上"分数正则"（反向散度项）以缓解模式覆盖带来的模糊，首次把 sCM 类方法扩展到 10B+ 的文生图与视频模型；[Decoupled DMD](https://arxiv.org/abs/2511.22677) 发现 DMD 中真正起主要作用的是"CFG 增强"项，分布匹配更像正则器，该方法用于 [Z-Image-Turbo](https://arxiv.org/abs/2511.22699)；[π-Flow](https://arxiv.org/abs/2510.14974) 让学生输出一个随时间变化的"策略"去模仿教师速度，在 FLUX.1-12B 与 Qwen-Image-20B 上 4 步保持教师级质量且多样性优于 DMD；[TwinFlow](https://arxiv.org/abs/2512.05150) 用自对抗流在不需要判别器的情况下把一步生成扩展到 20B 级模型。工业报告方面有 [Qwen-Image-Flash](https://arxiv.org/abs/2606.03746)、[Z-Image Turbo++](https://arxiv.org/abs/2606.12575)（2 步）与 [IDM](https://arxiv.org/abs/2610.39633)；视频侧有 [TMD](https://arxiv.org/abs/2601.09881)、[AnyFlow](https://arxiv.org/abs/2605.13724)。

2026 年起一个明显的新趋势是**on-policy 蒸馏**：沿学生自己生成的轨迹查询教师，避免训练-推理分布错位，这是 LLM 后训练中已被验证的做法。代表工作包括 [Microsoft 的简化 OPD](https://arxiv.org/abs/2610.18472)（简单的速度回归即可支撑 4-NFE 生成）、把多个专家教师合并进一个模型的 [DanceOPD](https://arxiv.org/abs/2606.27377)、缓解多奖励"跷跷板"的 [Flow-OPD](https://arxiv.org/abs/2605.08063)、把图像级奖励转为中间步目标的 [DiffusionOPSD](https://arxiv.org/abs/2608.24646)，以及讨论 OPD 如何与 CFG 配合的 [Qwen 工作](https://arxiv.org/abs/2607.24731)。

### 4.2 从头训练：一致性模型、流映射与 MeanFlow

**一致性模型**（[CM](https://arxiv.org/abs/2303.01469)）学习函数 $f_\theta(x_t,t)$，要求同一条 ODE 轨迹上的所有点都映射到同一个终点，并满足边界条件 $f_\theta(x,0)=x$。它可以蒸馏，也可以从头训练（[iCT](https://arxiv.org/abs/2310.14189)、[ECT](https://arxiv.org/abs/2406.14548)）。[sCM](https://arxiv.org/abs/2410.11081) 解决了连续时间一致性训练的不稳定问题，2 步在 ImageNet-512 上达到 FID 1.88。[CTM](https://arxiv.org/abs/2310.02279) 与 [流映射](https://arxiv.org/abs/2406.07507)把目标从"跳到终点"推广为"从任意 $t$ 跳到任意 $s$"，即学习**流映射** $X_{t\to s}$。[Shortcut Models](https://arxiv.org/abs/2410.12557) 以步长为条件单阶段训练；[IMM](https://arxiv.org/abs/2503.07565) 用矩匹配训练少步模型。

**MeanFlow**（[Geng 等，2025-05](https://arxiv.org/abs/2505.13447)）是最近一年一系列工作的起点。它定义区间平均速度

$$u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,\mathrm d\tau$$

于是一步跳跃就是 $z_r=z_t-(t-r)\,u(z_t,r,t)$。对 $t$ 求导可得一个只依赖瞬时速度的恒等式

$$u(z_t,r,t)=v(z_t,t)-(t-r)\,\frac{\mathrm d}{\mathrm d t}u(z_t,r,t)$$

右边的全导数可以用一次雅可比-向量积（JVP）算出，因而能直接构造回归目标、从头训练，无需教师。MeanFlow 在 ImageNet-256 上一步 FID 3.43。

**最近一年的 MeanFlow 家族与新范式**（全部为 ImageNet-256 一步、从头训练的结果，除非注明）：

| 工作 | 时间 | 1-NFE FID | 要点 |
|---|---|---|---|
| [α-Flow](https://arxiv.org/abs/2510.20771) | 2025-10 | 2.58 | MeanFlow 目标 = 轨迹流匹配 + 轨迹一致性，课程式过渡 |
| [Decoupled MeanFlow](https://arxiv.org/abs/2510.24474) | 2025-10 | 2.16 | 不改架构把预训练流模型变成流映射（借助预训练模型） |
| [改进 MeanFlow 训练](https://arxiv.org/abs/2511.19065) | 2025-11 | 2.87 | 先学好瞬时速度，再学大跨度平均速度 |
| [MeanFlow + RAE](https://arxiv.org/abs/2511.13019) | 2025-11 | 2.03 | 在表征自编码器潜空间训练，训练成本降 83% |
| [TVM](https://arxiv.org/abs/2511.19797) | 2025-11 | 3.29 | 在终点而非起点约束转移；4 步 1.99 |
| [对抗流模型](https://arxiv.org/abs/2511.22475) | 2025-11 | 2.38 | 对抗目标直接学一步映射 |
| [iMF](https://arxiv.org/abs/2512.02012) | 2025-12 | 1.72 | 重写训练目标与引导方式 |
| [pMF](https://arxiv.org/abs/2601.22158) | 2026-01 | 2.22 | 像素空间、无潜变量的一步生成 |
| [Drifting](https://arxiv.org/abs/2602.04770) | 2026-02 | 1.54 | 训练中分布漂移；像素空间 1.61 |
| [Priming Models](https://arxiv.org/abs/2610.20331) | 2026-10 | 1.19 | 两阶段简单框架，算力约为同类十分之一 |

不到一年半，一步生成的 FID 从 3.43 降到 1.19，已经与多步模型（REPA-E 1.12、RAE 1.13，均使用 CFG 与数十步）处于同一量级。流映射理论同步推进：[无数据流映射蒸馏](https://arxiv.org/abs/2511.19428)、[流映射用于测试时扩展](https://arxiv.org/abs/2511.22688)、[Meta Flow Maps](https://arxiv.org/abs/2601.14430) 用于可扩展的奖励对齐，[Discrete Flow Maps](https://arxiv.org/abs/2604.09784) 与 [Categorical Flow Maps](https://arxiv.org/abs/2602.12233) 把它推广到离散数据。需要注意的是，这些结果大多在类别条件的 ImageNet 上取得；在开放域文生图上，一步模型的多样性、可编辑性仍弱于蒸馏后的 4–8 步模型（[设置对齐的评测](https://arxiv.org/abs/2603.14186)）。

---

## 5 引导与条件采样

CFG 几乎是所有文生图 / 视频系统的标配，但代价是多样性下降、色彩过饱和，且每步需要两次前向。改进可分四类：

1. **换对照**：用"更差的自己"代替无条件模型——[Autoguidance](https://arxiv.org/abs/2406.02507) 用小/欠训练模型，ImageNet-512 FID 1.25；[SAG](https://arxiv.org/abs/2210.00939)、[PAG](https://arxiv.org/abs/2403.17377) 通过扰动注意力构造弱预测；[NAG](https://arxiv.org/abs/2505.21179) 在注意力空间做负向引导，使少步模型也能用负提示。
2. **换时间与方向**：[引导区间](https://arxiv.org/abs/2404.07724)只在中间噪声段施加 CFG；[APG](https://arxiv.org/abs/2410.02416) 去掉与条件预测平行的分量以消除过饱和；[CFG++](https://arxiv.org/abs/2406.08070)、[CFG-Zero*](https://arxiv.org/abs/2503.18886) 修正偏离流形的问题。
3. **训练内化**：[GFT](https://arxiv.org/abs/2501.15420)、[Model-guidance](https://arxiv.org/abs/2502.12154)、[SWIFT](https://arxiv.org/abs/2610.25844) 让模型直接学习引导后的分布，推理只需一次前向。
4. **逆问题与通用引导**：[DPS](https://arxiv.org/abs/2209.14687)、[DDRM](https://arxiv.org/abs/2201.11793)、[DAPS](https://arxiv.org/abs/2407.01521) 用似然梯度求解去模糊、超分等逆问题；[Universal Guidance](https://arxiv.org/abs/2302.07121) 用任意现成网络引导。

**最近一年**的重点是理论化与自动化：[分阶段动力学](https://arxiv.org/abs/2509.22007)与[解析分布](https://arxiv.org/abs/2607.19725)刻画 CFG 实际采样到的分布，[对抗学习引导日程](https://arxiv.org/abs/2608.14038)让系数随时间、样本与条件自适应，[系统复评](https://arxiv.org/abs/2608.16786)表明许多为 U-Net 设计的 CFG 变体在现代 DiT 上收益有限。蒸馏与 RL 后训练普遍把 CFG"烤"进模型（如 [Decoupled DMD](https://arxiv.org/abs/2511.22677)、[DiffusionNFT](https://arxiv.org/abs/2509.16117) 免 CFG），推理时显式引导的重要性在下降。

---

## 6 架构与扩展律

**从 U-Net 到 Transformer。**2020–2023 年的扩散模型以 [U-Net](https://arxiv.org/abs/1505.04597) 为骨干。[DiT](https://arxiv.org/abs/2212.09748) 把潜变量切成 patch 用 ViT 处理，FID 随 Gflops 稳定下降；[U-ViT](https://arxiv.org/abs/2209.12152) 同期提出带长跳连的 ViT 骨干；[SiT](https://arxiv.org/abs/2401.08740) 在 DiT 上系统比较插值框架。[SD3 的 MMDiT](https://arxiv.org/abs/2403.03206) 让文本与图像 token 走两套权重、在注意力中交互，成为 FLUX、HunyuanVideo 等的模板；随后又出现把两者拼成一个序列的单流 DiT（[Z-Image 的 S3-DiT](https://arxiv.org/abs/2511.22699)）。

**效率与扩展。**[Sana](https://arxiv.org/abs/2410.10629) 用线性注意力 + 32 倍压缩自编码器 + 小型 LLM 文本编码器，在笔记本 GPU 上生成 4K 图像；[DiT-MoE](https://arxiv.org/abs/2407.11633)、[HiDream-I1](https://arxiv.org/abs/2505.22705)（17B 稀疏 MoE）、Wan 2.2（按噪声高低划分的 MoE 去噪专家）引入混合专家；[DiT 扩展律](https://arxiv.org/abs/2410.08184)表明损失随算力呈幂律下降。

**最近一年。**[Z-Image](https://arxiv.org/abs/2511.22699) 以 6B 参数、约 31.4 万 H800 GPU 小时的训练逼近更大的开源与闭源模型，说明数据与训练流程的优化可以替代一部分规模；[Abra](https://arxiv.org/abs/2608.17286) 系统研究了文生图流匹配 Transformer 的计算最优扩展律；[均值-方差分离残差](https://arxiv.org/abs/2605.06169)诊断并解决千层 DiT 的"均值主导坍缩"；[ProMoE](https://arxiv.org/abs/2510.24711) 用显式路由引导让 MoE 在 DiT 上真正起效；[文本条件扩展性](https://arxiv.org/abs/2607.29679)研究提示信息量与损失的关系；[Lens](https://arxiv.org/abs/2605.21573)、[Mage-Flow](https://arxiv.org/abs/2607.19064) 给出 4B 左右的高效训练配方。

---

## 7 潜空间、Tokenizer 与像素空间

LDM 的 VAE 让扩散得以扩展到高分辨率，但它带来一个根本张力：**为重建而优化的潜空间不一定适合生成**。提高通道数能改善重建，却让扩散更难学。围绕这一张力，过去两年形成了三条路线。

**路线一：让潜空间更语义、更好扩散。**[VA-VAE / LightningDiT](https://arxiv.org/abs/2501.01423) 把 VAE 潜空间对齐到视觉基础模型，FID 1.35 且收敛快 21 倍；[DC-AE](https://arxiv.org/abs/2410.10733) 与 [DC-AE 1.5](https://arxiv.org/abs/2508.00413) 做深度压缩；[EQ-VAE](https://arxiv.org/abs/2502.09509)、[MAETok](https://arxiv.org/abs/2502.03444)、[l-DeTok](https://arxiv.org/abs/2507.15856) 从等变性、掩码、潜去噪角度正则化潜空间。**表征对齐** [REPA](https://arxiv.org/abs/2410.06940) 把 DiT 中间特征对齐到 DINOv2，训练加速 17.5 倍；[REPA-E](https://arxiv.org/abs/2504.10483) 借助 REPA 损失端到端联合训练 VAE 与扩散模型（FID 1.12）；[Dispersive Loss](https://arxiv.org/abs/2506.09027) 不用外部编码器、只在内部加分散正则；[DDT](https://arxiv.org/abs/2504.05741) 解耦语义编码器与高频解码器。

**路线二：直接用表征编码器当潜空间。**[RAE](https://arxiv.org/abs/2510.11690)（2025-10）用冻结的 DINOv2 / SigLIP 等编码器加一个训练好的解码器替换 VAE，在高维语义潜空间里扩散，配合宽 DDT 头，ImageNet 256 与 512 FID 均为 1.13；[RAE-T2I](https://arxiv.org/abs/2601.16208)（2026-01）把它扩展到大规模文生图，收敛更快、微调更不易过拟合。同期还有 [SVG](https://arxiv.org/abs/2510.15301)（DINO 特征 + 残差细节分支）、[VTP](https://arxiv.org/abs/2512.13687)（为生成而预训练的 tokenizer）、[一层适配](https://arxiv.org/abs/2512.07829)、[语义与重建兼顾](https://arxiv.org/abs/2512.17909)、[Unified Latents](https://arxiv.org/abs/2602.17270)（扩散先验正则 + 扩散解码，给出比特率上界）与视频上的 [SemanticGen](https://arxiv.org/abs/2512.20619)、[V-RAE](https://arxiv.org/abs/2608.13556)。[REPA 关键是什么](https://arxiv.org/abs/2512.10794)发现，起作用的主要是目标表征的空间结构而非全局语义。

**路线三：回到像素空间。**[simple diffusion](https://arxiv.org/abs/2301.11093)、[HDiT](https://arxiv.org/abs/2401.11605)、[SiD2](https://arxiv.org/abs/2410.19324) 早已证明像素扩散可行。2025-10 之后这一路线迅速成熟：[JiT](https://arxiv.org/abs/2511.13720) 用 $x$ 预测与大 patch Transformer 简化像素扩散；[EPG](https://arxiv.org/abs/2510.12586) 先自监督预训练再端到端训练，FID 1.58；[PixelDiT](https://arxiv.org/abs/2511.20645)（1.61）、[DeCo](https://arxiv.org/abs/2511.19365)（1.62）、[DiP](https://arxiv.org/abs/2511.18822)（1.79）用"DiT 管低频语义 + 轻量头补高频"的分工；[PixelGen](https://arxiv.org/abs/2602.02493) 加入感知损失；[像素文生图实证研究](https://arxiv.org/abs/2608.16887)给出可媲美潜空间的像素文生图训练配方；[对抗后训练](https://arxiv.org/abs/2609.38170)补足细尺度统计。统一多模态模型也开始抛弃视觉编码器、直接用像素嵌入（[Tuna-2](https://arxiv.org/abs/2604.24763)）。

三条路线并非互斥：工业界的高分辨率与视频生成短期内仍依赖高压缩潜空间（如 [Qwen-Image-VAE-2.0](https://arxiv.org/abs/2605.13565)、[FSVideo](https://arxiv.org/abs/2602.02092) 的 $64\times64\times4$ 压缩），但"语义化潜空间"与"像素空间"正在快速缩小差距。

---

## 8 对齐与后训练

预训练让模型学会"像不像数据"，后训练让它学会"好不好、合不合意"。

**奖励与偏好数据。**[ImageReward](https://arxiv.org/abs/2304.05977)、[HPSv2](https://arxiv.org/abs/2306.09341)、[PickScore](https://arxiv.org/abs/2305.01569)、[HPSv3](https://arxiv.org/abs/2508.03789) 提供人类偏好奖励；GenEval、OCR 等可验证奖励覆盖组合与文字渲染。

**方法谱系。**

1. **可微奖励反传**：ReFL（[ImageReward](https://arxiv.org/abs/2304.05977)）、[DRaFT](https://arxiv.org/abs/2309.17400)、[AlignProp](https://arxiv.org/abs/2310.03739) 沿采样链反传奖励梯度，高效但易奖励投机。
2. **偏好优化**：[Diffusion-DPO](https://arxiv.org/abs/2311.12908) 把 DPO 推广到扩散模型，用离线偏好对优化。
3. **在线策略梯度**：[DDPO](https://arxiv.org/abs/2305.13301)、[DPOK](https://arxiv.org/abs/2305.16381) 把去噪过程视为多步马尔可夫决策，

$$\nabla_\theta J=\mathbb E\Bigl[r(x_0)\sum_{k}\nabla_\theta\log p_\theta(x_{t_{k-1}}\mid x_{t_k},c)\Bigr]$$

[Flow-GRPO](https://arxiv.org/abs/2505.05470)（2025-05）把这一思路用到流匹配：先把确定性 ODE 改写为边缘不变的 SDE（第 1.4 节）以获得随机策略，再用组内相对优势 $A_i=(r_i-\bar r)/\mathrm{std}(r)$ 做 GRPO，并在训练时减少去噪步数提效；GenEval 从 63% 提升到 95%。[DanceGRPO](https://arxiv.org/abs/2505.07818) 覆盖图像与视频，[MixGRPO](https://arxiv.org/abs/2507.21802) 只在滑窗内用 SDE，[TempFlow-GRPO](https://arxiv.org/abs/2508.04324)、[BranchGRPO](https://arxiv.org/abs/2509.06040)、[Pref-GRPO](https://arxiv.org/abs/2508.20751)、[CPS](https://arxiv.org/abs/2509.05952)、[GRPO-Guard](https://arxiv.org/abs/2510.22319) 分别改进信用分配、采样效率、奖励形式与过优化问题。
4. **前向过程 RL**：[DiffusionNFT](https://arxiv.org/abs/2509.16117)（2025-09）不再对反向采样链求似然，而是在前向加噪过程上对比正负样本，定义隐式的改进方向，与采样器解耦、免 CFG，效率最高可达 Flow-GRPO 的 25 倍；[AWM](https://arxiv.org/abs/2509.25050) 让 RL 目标与预训练流匹配损失同形。
5. **随机最优控制**：[Adjoint Matching](https://arxiv.org/abs/2409.08861) 把奖励微调写成无记忆随机最优控制，给出有理论保证的目标分布 $p^\ast(x)\propto p_{\mathrm{base}}(x)\exp\bigl(r(x)/\lambda\bigr)$。

**测试时扩展。**[推理时扩展](https://arxiv.org/abs/2501.09732)在噪声空间搜索，用验证器挑选初始噪声或中间状态，表明推理算力的扩展可以超出"加步数"；[流映射测试时扩展](https://arxiv.org/abs/2511.22688)利用流映射在中途"预见"终点，让奖励引导不再病态。

**最近一年。**RL 已成为图像、视频模型发布的标准工序（[Qwen-Image-2.0-RL](https://arxiv.org/abs/2606.27608)、[Seedance 1.5 pro](https://arxiv.org/abs/2512.13507) 的 RLHF、[LMArena 团队的偏好 + 量规奖励](https://arxiv.org/abs/2610.02967)）。方法上：[DiffusionART](https://arxiv.org/abs/2610.12572) 兼顾反向过程方法的信用分配与前向过程方法的效率；[Flow-DPPO](https://arxiv.org/abs/2606.11025) 用散度约束替代比率裁剪；[FP4 rollout](https://arxiv.org/abs/2604.06916) 以低精度大规模采样扩大组大小；[GARDO](https://arxiv.org/abs/2512.24138) 专门处理奖励投机；[MeanFlowNFT](https://arxiv.org/abs/2607.15273)、[Flow-Map GRPO](https://arxiv.org/abs/2607.00535)、[TDM-R1](https://arxiv.org/abs/2603.07700) 把 RL 推到少步模型。奖励模型从标量走向分布与推理（[Z-Reward](https://arxiv.org/abs/2606.09076)、[Diffusion Reward Models](https://arxiv.org/abs/2609.33803)、[SpectraReward](https://arxiv.org/abs/2607.11886)）；视频 RL 开始优化物理与几何（[牛顿定律](https://arxiv.org/abs/2512.00425)、[VGGRPO](https://arxiv.org/abs/2603.26599)、[World-R1](https://arxiv.org/abs/2604.24764)），[视频模型本身也可以做潜空间奖励](https://arxiv.org/abs/2511.21541)。系统综述见 [视频后训练综述](https://arxiv.org/abs/2610.00812)。

---

## 9 视频生成与世界模型

### 9.1 基础模型

视频扩散始于 [VDM](https://arxiv.org/abs/2204.03458)，经 [Imagen Video](https://arxiv.org/abs/2210.02303)、[Make-A-Video](https://arxiv.org/abs/2209.14792)、[Video LDM](https://arxiv.org/abs/2304.08818)、[SVD](https://arxiv.org/abs/2311.15127)、[Lumiere](https://arxiv.org/abs/2401.12945) 发展。2024-02 的 [Sora](https://openai.com/index/video-generation-models-as-world-simulators/) 展示了时空 patch 上的扩散 Transformer 生成一分钟视频的能力，并提出"视频生成即世界模拟器"。此后开源模型快速跟进：[CogVideoX](https://arxiv.org/abs/2408.06072)、[HunyuanVideo](https://arxiv.org/abs/2412.03603)（13B）、[Wan](https://arxiv.org/abs/2503.20314)（1.3B/14B，2.2 版引入 MoE 去噪专家）、[LTX-Video](https://arxiv.org/abs/2501.00103)、[Step-Video](https://arxiv.org/abs/2502.10248)、[Open-Sora 2.0](https://arxiv.org/abs/2503.09642)；[Movie Gen](https://arxiv.org/abs/2410.13720) 以 30B 视频 + 13B 音频模型展示了流匹配在媒体生成上的规模化。

**最近一年**的主题是"全能化"：

- **原生音视频联合**：[Veo 3](https://deepmind.google/models/veo/) 把原生同步音频带入一线产品，随后 [Seedance 1.5 pro](https://arxiv.org/abs/2512.13507)（双分支 DiT）、[Seedance 2.0](https://arxiv.org/abs/2604.14148)（2026-02 发布，支持文本、图像、音频、视频四种输入与参考编辑）、[Kling-Omni](https://arxiv.org/abs/2512.16776)（生成、编辑与推理一体）、[Sora 2](https://openai.com/index/sora-2/) 成为一线产品；开源有 [LTX-2](https://arxiv.org/abs/2601.03233)（14B 视频流 + 5B 音频流）、[Ovi](https://arxiv.org/abs/2510.01284)、[MOVA](https://arxiv.org/abs/2602.08794)、[SkyReels-V4](https://arxiv.org/abs/2602.21818)、[Kandinsky 6.0 Video](https://arxiv.org/abs/2610.05608)。
- **更高效的开源基座**：[HunyuanVideo 1.5](https://arxiv.org/abs/2511.18870)（8.3B，消费级 GPU）、[LongCat-Video](https://arxiv.org/abs/2510.22200)（13.6B，原生长视频续写）、[Kandinsky 5.0](https://arxiv.org/abs/2511.14993)、[SANA-Video](https://arxiv.org/abs/2509.24695) 与 [SANA-Video 2.0](https://arxiv.org/abs/2607.21553)（混合线性注意力）。
- **多镜头叙事与控制**：[HoloCine](https://arxiv.org/abs/2510.20822)、[MultiShotMaster](https://arxiv.org/abs/2512.03041)、[OneStory](https://arxiv.org/abs/2512.07802)；运动与语义控制 [Wan-Move](https://arxiv.org/abs/2512.08765)、[Video-As-Prompt](https://arxiv.org/abs/2510.20888)；人物与数字人 [OmniHuman-1.5](https://arxiv.org/abs/2508.19209)、[Wan-Animate](https://arxiv.org/abs/2509.14055)、[KlingAvatar 2.0](https://arxiv.org/abs/2512.13313)。
- **产业格局变化**：OpenAI 于 2026-03 宣布分阶段关停 Sora App 与 API（[报道](https://the-decoder.com/openai-sets-two-stage-sora-shutdown-with-app-closing-april-2026-and-api-following-in-september/)）；Google 在 I/O 2026 推出"任意输入生成"的 [Gemini Omni](https://blog.google/innovation-and-ai/technology/ai/google-io-2026-all-our-announcements/)，先从视频开始；Black Forest Labs 的 [FLUX 3](https://bfl.ai/blog/flux-3) 联合训练图像、视频与音频。
- **视频即视觉**：[Veo 3 零样本](https://arxiv.org/abs/2509.20328)可以完成分割、边缘检测、物理推理、迷宫求解等未经训练的任务；[DeepMind 2026-07](https://arxiv.org/abs/2607.09024) 进一步论证大规模视频生成预训练可能是视觉领域的"下一个 token 预测"；[Thinking with Video](https://arxiv.org/abs/2511.04570) 把视频生成当作多模态推理范式。

### 9.2 自回归扩散、实时与长视频

双向全注意力的视频扩散一次生成固定长度片段，无法流式输出。[Diffusion Forcing](https://arxiv.org/abs/2407.01392) 让每个 token 拥有独立噪声水平，统一了"下一 token 预测"与"全序列扩散"；[CausVid](https://arxiv.org/abs/2412.07772) 把双向教师蒸馏成少步因果学生；[Self Forcing](https://arxiv.org/abs/2506.08009) 在训练时就用模型自己的输出做自回归展开，消除训练-测试偏差（暴露偏差），实现实时流式生成；[FramePack](https://arxiv.org/abs/2504.12626) 压缩历史上下文以在消费级显卡上生成长视频；[MAGI-1](https://arxiv.org/abs/2505.13211) 把逐块自回归扩展到大模型。

**最近一年**这条线迅速工程化：[Self-Forcing++](https://arxiv.org/abs/2510.02283) 推到分钟级；[LongLive](https://arxiv.org/abs/2509.22622) 支持生成过程中实时改提示；[Rolling Forcing](https://arxiv.org/abs/2509.25161) 用滚动窗口联合去噪；[Causal Forcing](https://arxiv.org/abs/2602.02214) 与 [Causal Forcing++](https://arxiv.org/abs/2605.15141) 指出从双向到因果蒸馏的架构鸿沟并给出正确做法；[Causal-rCM](https://arxiv.org/abs/2606.25473) 提供开源配方；[Reward Forcing](https://arxiv.org/abs/2512.04678) 在蒸馏中引入奖励。[Helios](https://arxiv.org/abs/2603.04379) 让 14B 模型在单张 H100 上以 19.5 FPS 生成分钟级视频；[TurboDiffusion](https://arxiv.org/abs/2512.16093) 组合低比特注意力、稀疏线性注意力与 rCM 蒸馏，端到端加速 100–200 倍；[Vidu S1](https://arxiv.org/abs/2607.03118)、[Vidu S2](https://arxiv.org/abs/2609.11638) 把实时交互数字人与实时编辑做成产品。

### 9.3 世界模型

以动作为条件逐帧生成的视频模型就是**世界模型**。[DIAMOND](https://arxiv.org/abs/2405.12399) 在扩散世界模型里训练 Atari 智能体；[GameNGen](https://arxiv.org/abs/2408.14837) 用扩散模型实时运行 DOOM；[Genie](https://arxiv.org/abs/2402.15391) 从无标注视频学习潜动作；[Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)（2025-08）实现 720p、24 FPS、数分钟一致的实时交互世界；[Cosmos](https://arxiv.org/abs/2501.03575) 面向物理 AI。

**最近一年**：[Genie 4](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/introducing-genie-4/)（2026-10）加入自然语言改写场景与原生音频；开源侧 [Matrix-Game 2.0](https://arxiv.org/abs/2508.13009) → [3.0](https://arxiv.org/abs/2604.08995) → [3.5](https://arxiv.org/abs/2608.29910)、[WorldPlay](https://arxiv.org/abs/2512.14614)、[LingBot-World](https://arxiv.org/abs/2601.20540)、[SANA-WM](https://arxiv.org/abs/2605.15178)（2.6B，原生一分钟 720p）、[Gamma-World](https://arxiv.org/abs/2605.28816)（多智能体）、[Hy-Nomas](https://arxiv.org/abs/2610.36729) 快速迭代；[HY-World 2.0](https://arxiv.org/abs/2604.14268)、[Lyra 2.0](https://arxiv.org/abs/2604.13036) 把生成的漫游视频重建为可探索的 3D 世界。核心难题是**记忆**：离开再回来时场景是否一致，相关工作有 [MosaicMem](https://arxiv.org/abs/2603.17117)、[潜空间空间记忆](https://arxiv.org/abs/2606.09828)、[自回归视频的记忆](https://arxiv.org/abs/2609.28466)。评测见 [WBench](https://arxiv.org/abs/2605.25874)、[StreamEval](https://arxiv.org/abs/2610.28402)、[World-in-World](https://arxiv.org/abs/2510.18135)。

---

## 10 离散扩散与扩散语言模型

把"加高斯噪声"换成"逐步遮盖 token"，扩散就可以处理文本。[D3PM](https://arxiv.org/abs/2107.03006) 给出离散状态空间扩散的一般框架；[SEDD](https://arxiv.org/abs/2310.16834) 用分数熵估计数据分布比率，首次在困惑度上逼近 GPT-2（ICML 2024 最佳论文）；[MDLM](https://arxiv.org/abs/2406.07524) 与 [MD4](https://arxiv.org/abs/2406.04329) 把**掩码扩散**的训练简化为加权的掩码语言建模。当遮盖概率随时间线性增长（时刻 $t$ 每个 token 以概率 $t$ 被遮盖）时，负 ELBO 为

$$\mathcal L(\theta)=\mathbb E_{t,\,x_0,\,x_t}\Bigl[\frac1t\sum_{i:\,x_t^i=[\mathrm M]}-\log p_\theta\bigl(x_0^i\mid x_t\bigr)\Bigr]$$

即"按遮盖比例加权的 BERT 式损失"。采样时从全 [M] 序列出发，每步并行预测并解开一部分 token。

**规模化。**[LLaDA](https://arxiv.org/abs/2502.09992)（2025-02）从零预训练 8B 掩码扩散语言模型，能力与同规模自回归模型可比，并缓解了"逆转诅咒"；[Dream 7B](https://arxiv.org/abs/2508.15487) 从自回归模型初始化；商用侧有 [Mercury](https://arxiv.org/abs/2506.17298)（代码生成每秒上千 token）、[Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)、[Seed Diffusion](https://arxiv.org/abs/2508.02193)。[BD3-LM](https://arxiv.org/abs/2503.09573) 提出**块扩散**——块间自回归、块内扩散——兼顾 KV 缓存与变长生成，成为此后大模型的主流形态（[SDAR](https://arxiv.org/abs/2510.06303)、[Fast-dLLM v2](https://arxiv.org/abs/2509.26328)）。推理加速有 [Fast-dLLM](https://arxiv.org/abs/2505.22618)、[dKV-Cache](https://arxiv.org/abs/2505.15781)；推理 RL 有 [d1](https://arxiv.org/abs/2504.12216)、[TraceRL](https://arxiv.org/abs/2509.06949)。

**最近一年。**

- **规模**：[LLaDA2.0](https://arxiv.org/abs/2512.15745) 通过从自回归模型系统转换，把块扩散扩展到 100B 总参数；[LLaDA2.1](https://arxiv.org/abs/2602.08676) 引入 token 编辑以提速；[LLaDA MoE v2](https://arxiv.org/abs/2608.03457) 刻画 MoE 扩散 LM 的扩展规律；Google 发布开源权重的 [DiffusionGemma](https://arxiv.org/abs/2608.00146)，每次并行细化 256 个 token 的块；[iLLaDA](https://arxiv.org/abs/2606.25331)、[Stable-DiffCoder](https://arxiv.org/abs/2601.15892) 继续从零训练与代码方向。
- **数据效率与能力边界**：[超级数据学习者](https://arxiv.org/abs/2511.03276)发现唯一数据有限时，扩散 LM 多轮训练稳定超过自回归模型；[灵活性陷阱](https://arxiv.org/abs/2601.15165)则指出任意顺序生成可能反而限制推理潜力。
- **速度**：[TiDAR](https://arxiv.org/abs/2511.08923)（一次前向内扩散起草 + 自回归校验）、[DFlash](https://arxiv.org/abs/2602.06036) 与 [DEER](https://arxiv.org/abs/2512.15176)（扩散做投机解码草稿）、[DMax](https://arxiv.org/abs/2604.08302)（激进并行解码）、[扩散增强 LLM](https://arxiv.org/abs/2609.04010)（无损加速）。
- **连续化**：[LangFlow](https://arxiv.org/abs/2604.11748) 让连续扩散语言模型首次追平离散扩散，[连续潜扩散 LM](https://arxiv.org/abs/2605.06548)、[AURORA-LM](https://arxiv.org/abs/2608.02602)、[Large Flow LM](https://arxiv.org/abs/2610.28559) 把流匹配带到十亿参数级语言建模。
- **多模态**：[Lumina-DiMOO](https://arxiv.org/abs/2510.06308)、[MMaDA-Parallel](https://arxiv.org/abs/2511.09611)、[LLaDA2.0-Uni](https://arxiv.org/abs/2604.20796)、[LLaDA-o](https://arxiv.org/abs/2603.01068)、[Nemotron-Labs-Diffusion-Uni](https://arxiv.org/abs/2610.36466)，以及以扩散 LM 为骨干的 [Dream-VLA](https://arxiv.org/abs/2512.22615)。

总体看，扩散 LM 的优势是并行解码速度、双向上下文与数据效率，劣势是在相同规模下推理与长文本质量仍普遍略逊于最强自回归模型。它更可能先作为草稿模型、编辑器与特定场景（代码补全、低延迟服务）的方案落地。综述见 [扩散语言模型综述](https://arxiv.org/abs/2508.10875)与 [加速技术综述](https://arxiv.org/abs/2607.12829)。

---

## 11 统一多模态与跨领域应用

**统一理解与生成。**主流设计是"自回归管文本、扩散/流管图像"：[Transfusion](https://arxiv.org/abs/2408.11039) 在同一 Transformer 中对文本做下一词预测、对图像做扩散；[MAR](https://arxiv.org/abs/2406.11838) 用每 token 的小扩散头替代向量量化；[JanusFlow](https://arxiv.org/abs/2411.07975)、[MetaQuery](https://arxiv.org/abs/2504.06256)、[BLIP3-o](https://arxiv.org/abs/2505.09568)、[BAGEL](https://arxiv.org/abs/2505.14683)、[UniWorld-V1](https://arxiv.org/abs/2506.03147)、[Manzano](https://arxiv.org/abs/2509.16197)、[NextStep-1](https://arxiv.org/abs/2508.10711) 探索不同的耦合方式。最近一年：[Emu3.5](https://arxiv.org/abs/2510.26583) 以下一 token 预测为主，并用离散扩散适配（DiDA）把单图推理加速约 20 倍；[TUNA](https://arxiv.org/abs/2512.02014)、[Tuna-2](https://arxiv.org/abs/2604.24763) 研究统一视觉表征乃至直接使用像素嵌入；[Meta 的原生多模态预训练研究](https://arxiv.org/abs/2603.03276)与["多模态预训练的物理学"](https://arxiv.org/abs/2608.05000)系统梳理设计空间；[InternVL-U](https://arxiv.org/abs/2603.09877)、[SenseNova-U1](https://arxiv.org/abs/2605.12500)、[Lance](https://arxiv.org/abs/2605.18678) 等开源统一模型密集出现。闭源侧，[GPT-4o 图像生成](https://openai.com/index/introducing-4o-image-generation/)官方称为自回归模型，[Nano Banana Pro](https://blog.google/innovation-and-ai/products/nano-banana-pro/) 未公开架构，说明"扩散 vs 自回归"在产品层面并无定论。

**编辑与可控生成。**从训练无关的 [SDEdit](https://arxiv.org/abs/2108.01073)、[Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)、[Null-text Inversion](https://arxiv.org/abs/2211.09794)，到指令编辑 [InstructPix2Pix](https://arxiv.org/abs/2211.09800)，再到结构控制 [ControlNet](https://arxiv.org/abs/2302.05543)、个性化 [DreamBooth](https://arxiv.org/abs/2208.12242)；DiT 时代的上下文生成（[In-Context LoRA](https://arxiv.org/abs/2410.23775)、[OminiControl](https://arxiv.org/abs/2411.15098)、[FLUX.1 Kontext](https://arxiv.org/abs/2506.15742)）让"生成与编辑一体"成为基础模型的标配（[Seedream 4.0](https://arxiv.org/abs/2509.20427)、[Qwen-Image](https://arxiv.org/abs/2508.02324)、[FLUX.2](https://bfl.ai/blog/flux-2)、[Qwen-Image-2.0](https://arxiv.org/abs/2605.10730)）。最近一年的新方向是"先想再画"：[Think-Then-Generate](https://arxiv.org/abs/2601.10332)、[ReasonEdit](https://arxiv.org/abs/2511.22625)、[Qwen-Image-Agent](https://arxiv.org/abs/2606.26907)、[Gen-Searcher](https://arxiv.org/abs/2603.28767)。

**3D。**[DreamFusion](https://arxiv.org/abs/2209.14988) 的分数蒸馏采样开启文生 3D；[Zero-1-to-3](https://arxiv.org/abs/2303.11328)、[MVDream](https://arxiv.org/abs/2308.16512) 用多视角扩散做先验；[TRELLIS](https://arxiv.org/abs/2412.01506)、[Hunyuan3D 2.0](https://arxiv.org/abs/2501.12202) 在结构化 3D 潜空间中直接用流匹配生成资产。最近一年有 [TRELLIS.2](https://arxiv.org/abs/2512.14692)、[SAM 3D](https://arxiv.org/abs/2511.16624)、[PhysX-Anything](https://arxiv.org/abs/2511.13648)（可仿真的带关节资产）。

**音频。**[DiffWave](https://arxiv.org/abs/2009.09761)、[AudioLDM](https://arxiv.org/abs/2301.12503)、[Voicebox](https://arxiv.org/abs/2306.15687)、[F5-TTS](https://arxiv.org/abs/2410.06885)、[MMAudio](https://arxiv.org/abs/2412.15322) 等奠定了扩散 / 流匹配在语音、音效与拟音中的地位；最近一年的重点是与视频联合生成（第 9.1 节）。

**科学。**[RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)（蛋白质设计）与 [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)（扩散模块生成全原子结构）使扩散成为结构生物学的核心工具；[FrameFlow](https://arxiv.org/abs/2310.05297)、[Proteina](https://arxiv.org/abs/2503.00710)、[La-Proteina](https://arxiv.org/abs/2507.09466) 用流匹配生成蛋白质；[MatterGen](https://arxiv.org/abs/2312.03687) 生成材料，[GenCast](https://arxiv.org/abs/2312.15796) 做集合天气预报。最近一年有全原子设计的 [RFdiffusion3](https://doi.org/10.1101/2025.09.18.676967)。

**机器人。**[Diffusion Policy](https://arxiv.org/abs/2303.04137) 用扩散建模多峰动作分布；[π0](https://arxiv.org/abs/2410.24164)、[GR00T N1](https://arxiv.org/abs/2503.14734) 等 VLA 采用流匹配动作头；[实时分块执行](https://arxiv.org/abs/2506.07339)把动作衔接写成带引导的修补问题。最近一年：[π*0.6](https://arxiv.org/abs/2511.14759) 从自身经验持续改进，[π_RL](https://arxiv.org/abs/2510.25889) 对流式 VLA 做在线 RL，[Kinematic MeanFlow](https://arxiv.org/abs/2610.00864) 把一步生成用于动作，[DreamZero](https://arxiv.org/abs/2602.15922) 等"世界动作模型"用预训练视频扩散直接做策略（综述见 [WAM Survey](https://arxiv.org/abs/2606.20781)）。

---

## 12 理论：收敛、记忆与泛化

**收敛性。**[Chen 等](https://arxiv.org/abs/2209.11215)证明只要分数估计在 $L^2$ 意义下足够准，扩散采样就能以多项式复杂度逼近任意数据分布；[Benton 等](https://arxiv.org/abs/2308.03686)给出关于维度近线性的界；[概率流 ODE](https://arxiv.org/abs/2305.11798)同样可证地快；[Oko 等](https://arxiv.org/abs/2303.01861)表明扩散模型是极小极大最优的分布估计器。最近一年，[流匹配自适应流形](https://arxiv.org/abs/2602.22486)证明收敛速率由内在维度决定，[神经网络流匹配理论](https://arxiv.org/abs/2606.10089)与[扩散流匹配收敛界](https://arxiv.org/abs/2606.16610)补全了有限网络与离散化误差的分析。

**记忆与泛化之谜。**若训练充分到能精确拟合经验分布的分数，最优"去噪器"只会复现训练样本。可现实中的模型几乎总在生成新图像，且 [Kadkhodaie 等](https://arxiv.org/abs/2310.02557)发现两个在不重叠数据上训练的模型会生成几乎相同的图——泛化来自网络的归纳偏置。另一方面，[Carlini 等](https://arxiv.org/abs/2301.13188)与 [Somepalli 等](https://arxiv.org/abs/2212.03860)也证实了真实的记忆与复制现象。

**最近一年**这个问题有了较完整的回答：

- [Bonnaire 等](https://arxiv.org/abs/2505.17638)（NeurIPS 2025 最佳论文）：训练动力学中有两个时间尺度——泛化时间基本不随数据量变化，记忆时间随数据量 $n$ 线性增长；只要在两者之间停止训练，就能得到泛化良好的模型（隐式动力学正则化）。
- [Kamb 与 Ganguli](https://arxiv.org/abs/2412.20292)：具有局部性与平移等变性的最优去噪器可以解析地"拼贴"出训练集中不存在的新图像，解释卷积扩散的创造力；[Locality](https://arxiv.org/abs/2509.09672) 进一步指出局部性主要来自图像数据的统计结构而非卷积架构；[去噪器架构与创造力](https://arxiv.org/abs/2605.16415)讨论架构偏离贝叶斯最优的方式。
- [选择性欠拟合](https://arxiv.org/abs/2510.01378)：模型只在采样过程会经过的区域拟合分数，在其余区域欠拟合，这种"有选择的欠拟合"恰是泛化来源；[闭式流匹配](https://arxiv.org/abs/2506.03719)表明泛化并不来自条件目标的随机性。
- 可证明结果：[可证分离](https://arxiv.org/abs/2511.03202)、[流形泛化先于记忆](https://arxiv.org/abs/2603.23792)、[懒惰高维区间](https://arxiv.org/abs/2608.23938)，以及[局部覆盖决定哪些样本被记住](https://arxiv.org/abs/2606.14390)。

这些结论对实践有直接含义：记忆风险与数据重复度、训练时长和局部数据密度强相关，是版权与隐私讨论（第 13 节）的技术基础。

---

## 13 评测、安全与版权

**分布指标。**[FID](https://arxiv.org/abs/1706.08500) 仍是 ImageNet 方法论文的主指标，但它依赖 Inception 特征、假设高斯分布，[Stein 等](https://arxiv.org/abs/2306.04675)指出其对扩散模型不公平，建议改用 DINOv2 特征（FD-DINOv2）；[CMMD](https://arxiv.org/abs/2401.09603) 用 CLIP 特征与 MMD。[FID 彩票](https://arxiv.org/abs/2606.20536)（2026-06）量化了重训与换种子带来的隐藏随机性，提醒读者 0.1 量级的 FID 差距往往不具统计意义。

**文图一致与偏好。**[GenEval](https://arxiv.org/abs/2310.11513)、[T2I-CompBench](https://arxiv.org/abs/2307.06350)、[DPG-Bench](https://arxiv.org/abs/2403.05135) 测组合与长提示；[TIFA](https://arxiv.org/abs/2303.11897)、[DSG](https://arxiv.org/abs/2310.18235)、[VQAScore](https://arxiv.org/abs/2404.01291) 用问答打分；[WISE](https://arxiv.org/abs/2503.07265)、[OneIG-Bench](https://arxiv.org/abs/2506.07977)、[UniGenBench++](https://arxiv.org/abs/2510.18701) 覆盖知识与细粒度语义。注意 GenEval 已接近饱和且是 RL 的常用奖励，用作评测时需警惕"训练即评测"。竞技场式人类投票是产业界最常引用的排名。

**视频与世界模型。**[VBench](https://arxiv.org/abs/2311.17982) 与 [VBench-2.0](https://arxiv.org/abs/2503.21755) 分维度评测画质与"内在真实性"；[PhyGenBench](https://arxiv.org/abs/2410.05363)、[VideoPhy](https://arxiv.org/abs/2406.03520)、[Physics-IQ](https://arxiv.org/abs/2501.09038) 测物理常识；[V-ReasonBench](https://arxiv.org/abs/2511.16668)、[TiViBench](https://arxiv.org/abs/2511.13704) 测视频推理；[WorldScore](https://arxiv.org/abs/2504.00983)、[WBench](https://arxiv.org/abs/2605.25874)、[StreamEval](https://arxiv.org/abs/2610.28402) 面向世界模型与流式交互。

**安全、版权与水印。**概念擦除（[ESD](https://arxiv.org/abs/2303.07345)、[UCE](https://arxiv.org/abs/2308.14761)、[Concept Ablation](https://arxiv.org/abs/2303.13516)）可被红队绕过（[Ring-A-Bell](https://arxiv.org/abs/2310.10012)）；[Glaze](https://arxiv.org/abs/2302.04222)、[Nightshade](https://arxiv.org/abs/2310.13828) 从创作者侧对抗风格模仿；水印方面有 [Tree-Ring](https://arxiv.org/abs/2305.20030)、[Stable Signature](https://arxiv.org/abs/2303.15435)、[Watermark Anything](https://arxiv.org/abs/2411.07231) 与已在互联网规模部署的 [SynthID-Image](https://arxiv.org/abs/2510.09263)。视频生成的版权争议在 2026 年明显升温（如 Seedance 2.0 发布后的版权方抗议），溯源与授权机制仍不成熟。

---

## 14 最近一年大事记（2025-10 → 2026-10）

- **2025-10**：[RAE](https://arxiv.org/abs/2510.11690) 用表征编码器替换 VAE；少步生成密集涌现（[α-Flow](https://arxiv.org/abs/2510.20771)、[Decoupled MeanFlow](https://arxiv.org/abs/2510.24474)、[π-Flow](https://arxiv.org/abs/2510.14974)、[rCM](https://arxiv.org/abs/2510.08431)）；[EqM](https://arxiv.org/abs/2510.02300)、[选择性欠拟合](https://arxiv.org/abs/2510.01378)；[EPG](https://arxiv.org/abs/2510.12586) 像素扩散 FID 1.58；Veo 3.1；开源 [LongCat-Video](https://arxiv.org/abs/2510.22200)、[Ovi](https://arxiv.org/abs/2510.01284)；[Emu3.5](https://arxiv.org/abs/2510.26583)；专著 [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)。
- **2025-11**：[JiT](https://arxiv.org/abs/2511.13720)、[DeCo](https://arxiv.org/abs/2511.19365)、[PixelDiT](https://arxiv.org/abs/2511.20645) 推动像素扩散；[TVM](https://arxiv.org/abs/2511.19797)、[MeanFlow + RAE](https://arxiv.org/abs/2511.13019)、[对抗流](https://arxiv.org/abs/2511.22475)；[Z-Image](https://arxiv.org/abs/2511.22699) 与 [Decoupled DMD](https://arxiv.org/abs/2511.22677)；[HunyuanVideo 1.5](https://arxiv.org/abs/2511.18870)、[Kandinsky 5.0](https://arxiv.org/abs/2511.14993)；[Nano Banana Pro](https://blog.google/innovation-and-ai/products/nano-banana-pro/)、[FLUX.2](https://bfl.ai/blog/flux-2)；[SAM 3D](https://arxiv.org/abs/2511.16624)；[扩散 LM 是超级数据学习者](https://arxiv.org/abs/2511.03276)、[TiDAR](https://arxiv.org/abs/2511.08923)。
- **2025-12**：[iMF](https://arxiv.org/abs/2512.02012) 一步 FID 1.72；[TwinFlow](https://arxiv.org/abs/2512.05150)；[LLaDA2.0](https://arxiv.org/abs/2512.15745) 扩展到 100B；[Seedance 1.5 pro](https://arxiv.org/abs/2512.13507)、[Kling-Omni](https://arxiv.org/abs/2512.16776)；[TurboDiffusion](https://arxiv.org/abs/2512.16093)；[LongCat-Image](https://arxiv.org/abs/2512.07584)、[Qwen-Image-Layered](https://arxiv.org/abs/2512.15603)；[TRELLIS.2](https://arxiv.org/abs/2512.14692)。
- **2026-01**：[LTX-2](https://arxiv.org/abs/2601.03233) 开源音视频基础模型；[pMF](https://arxiv.org/abs/2601.22158) 像素一步生成；[RAE-T2I](https://arxiv.org/abs/2601.16208)；[LingBot-World](https://arxiv.org/abs/2601.20540)；[Meta Flow Maps](https://arxiv.org/abs/2601.14430)；[灵活性陷阱](https://arxiv.org/abs/2601.15165)。
- **2026-02**：[Drifting Models](https://arxiv.org/abs/2602.04770) 一步 FID 1.54；Seedance 2.0 发布（技术报告 [2026-04](https://arxiv.org/abs/2604.14148)）；[Kling 3.0](https://en.wikipedia.org/wiki/Kling_AI)；[LLaDA2.1](https://arxiv.org/abs/2602.08676)；[Causal Forcing](https://arxiv.org/abs/2602.02214)；[Unified Latents](https://arxiv.org/abs/2602.17270)；[SkyReels-V4](https://arxiv.org/abs/2602.21818)、[MOVA](https://arxiv.org/abs/2602.08794)。
- **2026-03**：[Helios](https://arxiv.org/abs/2603.04379) 实时长视频；OpenAI 宣布关停 Sora App 与 API（[报道](https://the-decoder.com/openai-sets-two-stage-sora-shutdown-with-app-closing-april-2026-and-api-following-in-september/)）；[Meta 原生多模态预训练研究](https://arxiv.org/abs/2603.03276)；[流形泛化先于记忆](https://arxiv.org/abs/2603.23792)。
- **2026-04**：[Matrix-Game 3.0](https://arxiv.org/abs/2604.08995)、[HY-World 2.0](https://arxiv.org/abs/2604.14268)、[Lyra 2.0](https://arxiv.org/abs/2604.13036)；[LLaDA2.0-Uni](https://arxiv.org/abs/2604.20796)；[Tuna-2](https://arxiv.org/abs/2604.24763)；[LangFlow](https://arxiv.org/abs/2604.11748)；[Discrete Flow Maps](https://arxiv.org/abs/2604.09784)；[World-R1](https://arxiv.org/abs/2604.24764)。
- **2026-05**：Google I/O 发布 [Gemini Omni](https://blog.google/innovation-and-ai/technology/ai/google-io-2026-all-our-announcements/)；[Qwen-Image-2.0](https://arxiv.org/abs/2605.10730)、[Qwen-Image-VAE-2.0](https://arxiv.org/abs/2605.13565)；[Lens](https://arxiv.org/abs/2605.21573)；[Causal Forcing++](https://arxiv.org/abs/2605.15141)、[SANA-WM](https://arxiv.org/abs/2605.15178)；[AnyFlow](https://arxiv.org/abs/2605.13724)、[Flow-OPD](https://arxiv.org/abs/2605.08063)。
- **2026-06**：[Cosmos 3](https://arxiv.org/abs/2606.02800)；[DanceOPD](https://arxiv.org/abs/2606.27377)、[Qwen-Image-2.0-RL](https://arxiv.org/abs/2606.27608)、[Z-Reward](https://arxiv.org/abs/2606.09076)、[Flow-DPPO](https://arxiv.org/abs/2606.11025)；[Causal-rCM](https://arxiv.org/abs/2606.25473)；[iLLaDA](https://arxiv.org/abs/2606.25331)；[FID 彩票](https://arxiv.org/abs/2606.20536)。
- **2026-07**：[FLUX 3](https://bfl.ai/blog/flux-3)；[DiffusionGemma](https://arxiv.org/abs/2608.00146) 技术报告（07-31）；[视频生成即通用视觉学习](https://arxiv.org/abs/2607.09024)；[SANA-Video 2.0](https://arxiv.org/abs/2607.21553)；[MeanFlowNFT](https://arxiv.org/abs/2607.15273)；[Mage-Flow](https://arxiv.org/abs/2607.19064)；[Vidu S1](https://arxiv.org/abs/2607.03118)。
- **2026-08**：[LLaDA MoE v2](https://arxiv.org/abs/2608.03457)；[Abra 扩展律](https://arxiv.org/abs/2608.17286)；[像素文生图实证研究](https://arxiv.org/abs/2608.16887)；[DiffusionOPSD](https://arxiv.org/abs/2608.24646)；[Matrix-Game 3.5](https://arxiv.org/abs/2608.29910)。
- **2026-09**：[FLUX 3 Action](https://bfl.ai/models/flux-3-action)；[Vidu S2](https://arxiv.org/abs/2609.11638)；[Diffusion Reward Models](https://arxiv.org/abs/2609.33803)；[DC-SAE](https://arxiv.org/abs/2609.39222)。
- **2026-10**：[Priming Models](https://arxiv.org/abs/2610.20331) 一步 FID 1.19；[DiffusionART](https://arxiv.org/abs/2610.12572)；[Genie 4](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/introducing-genie-4/)；[FLUX 3 Image Klein](https://bfl.ai/blog/flux-3-image-klein) 开放权重；[Kandinsky 6.0 Video](https://arxiv.org/abs/2610.05608)；[IDM](https://arxiv.org/abs/2610.39633)；[StreamEval](https://arxiv.org/abs/2610.28402)；[Large Flow LM](https://arxiv.org/abs/2610.28559)。

---

## 15 开放问题与展望

1. **一步生成能否走出 ImageNet。**类别条件上一步模型已追上多步，但开放域文生图与视频的多样性、可编辑性与可控性仍落后；从头训练的一步模型能否扩展到十亿参数、能否与 RL 后训练兼容，是 2027 年最值得关注的问题之一。
2. **潜空间的终局。**语义表征、重建保真与"好扩散"如何兼得？像素空间的成本在视频上能否接受？理解与生成能否共用同一视觉表征？
3. **可扩展的奖励。**标量奖励易被投机，人类偏好昂贵且主观，可验证奖励（物理、几何、文字）只覆盖少数维度。视觉生成需要像 LLM 那样的可扩展奖励来源。
4. **长时程一致与记忆。**分钟级视频与世界模型会漂移、遗忘已见场景；显式 3D 记忆、潜空间记忆、检索式记忆各有代价，尚无公认方案。
5. **物理与因果。**"看起来对"不等于"物理上对"。视频模型能否成为可靠的物理模拟器与机器人训练环境，评测仍很初步。
6. **扩散语言模型的定位。**并行解码带来速度优势，但同规模下的推理质量仍普遍略逊；它会成为独立主流，还是草稿模型与编辑器？
7. **泛化、记忆与版权。**平均意义上的泛化已有解释，但单个样本是否被记住、训练数据的授权与溯源、水印的鲁棒性仍缺乏可落地的保证。
8. **效率。**视频生成的推理成本远高于文本，稀疏注意力、低比特量化与少步蒸馏叠加后的质量损失需要系统评估。

**延伸阅读。**教程与专著：[MIT 6.S184 讲义](https://arxiv.org/abs/2506.02070)、[Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)、[The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)、[Understanding Diffusion Models](https://arxiv.org/abs/2208.11970)、[Step-by-Step Diffusion](https://arxiv.org/abs/2406.08929)。综述：[Yang 等](https://arxiv.org/abs/2209.00796)、[扩散语言模型综述](https://arxiv.org/abs/2508.10875)、[高效扩散综述](https://arxiv.org/abs/2502.06805)、[视频后训练综述](https://arxiv.org/abs/2610.00812)、[视频生成即世界模型](https://arxiv.org/abs/2603.28489)。

---

## 附录 A：论文与资源总表（717 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 核验；★ 为入门推荐；🆕 为 2025-10 之后的工作。

### 基础与目标函数（26）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-11 | ★ 🆕 [JiT](https://arxiv.org/abs/2511.13720) `arXiv:2511.13720` | MIT | Back to Basics：主张网络直接预测干净图像（x-prediction），大 patch 的纯 Transformer 在像素空间也能训好 |
| 2025-10 | 🆕 [EqM](https://arxiv.org/abs/2510.02300) `arXiv:2510.02300` | MIT/Harvard | Equilibrium Matching：学习与时间无关的平衡梯度（隐式能量），推理时做梯度下降，ImageNet-256 FID 1.90 |
| 2025-02 | [无噪声条件](https://arxiv.org/abs/2502.13129) `arXiv:2502.13129` | MIT | 去掉时间/噪声条件输入，多数去噪生成模型仍能工作，并给出误差界 |
| 2024-10 | [SiD2](https://arxiv.org/abs/2410.19324) `arXiv:2410.19324` | Google DeepMind | 像素空间扩散 ImageNet-512 FID 1.5，说明不用 VAE 也能高质高效 |
| 2023-12 | [EDM2](https://arxiv.org/abs/2312.02696) `arXiv:2312.02696` | NVIDIA | 分析训练动态，做幅值保持的网络与事后 EMA，ImageNet-512 FID 1.81 |
| 2023-05 | [Zero Terminal SNR](https://arxiv.org/abs/2305.08891) `arXiv:2305.08891` | ByteDance | 指出常见噪声表最后一步不是纯噪声，导致生成偏灰；提出零终端 SNR 与 v 预测修正 |
| 2023-03 | [Min-SNR](https://arxiv.org/abs/2303.09556) `arXiv:2303.09556` | Microsoft | 把不同时间步视为多任务，按截断 SNR 加权损失，训练收敛快约 3.4 倍 |
| 2023-03 | [ELBO 视角](https://arxiv.org/abs/2303.00848) `arXiv:2303.00848` | Google | 证明常用加权损失等价于带数据增强的 ELBO，统一各种目标函数 |
| 2023-01 | [simple diffusion](https://arxiv.org/abs/2301.11093) `arXiv:2301.11093` | Google | 像素空间端到端训练高分辨率扩散，靠移位噪声表与多尺度损失 |
| 2023-01 | [Noise Scheduling](https://arxiv.org/abs/2301.10972) `arXiv:2301.10972` | Google | 噪声表应随分辨率调整（图像越大越要加更多噪声） |
| 2022-08 | [Analog Bits](https://arxiv.org/abs/2208.04202) `arXiv:2208.04202` | Google | 把离散数据编码为比特再做连续扩散，并提出 self-conditioning |
| 2022-06 | ★ [EDM](https://arxiv.org/abs/2206.00364) `arXiv:2206.00364` | NVIDIA | 把扩散的设计空间拆成独立模块（预条件、噪声分布、采样器），成为之后多年的默认配方 |
| 2021-12 | [Denoising Diffusion GAN](https://arxiv.org/abs/2112.07804) `arXiv:2112.07804` | NVIDIA | 用条件 GAN 建模大步长去噪，较早尝试少步扩散 |
| 2021-11 | [Palette](https://arxiv.org/abs/2111.05826) `arXiv:2111.05826` | Google | 统一的图到图扩散框架（上色、补全、去 JPEG 等） |
| 2021-07 | [VDM](https://arxiv.org/abs/2107.00630) `arXiv:2107.00630` | Google | 从信噪比（SNR）视角推导连续时间 ELBO，噪声表可学习 |
| 2021-06 | [LSGM](https://arxiv.org/abs/2106.05931) `arXiv:2106.05931` | NVIDIA | 在 VAE 潜空间里训练分数模型，潜空间扩散的早期形态 |
| 2021-05 | [CDM](https://arxiv.org/abs/2106.15282) `arXiv:2106.15282` | Google | 级联扩散：低分辨率生成 + 逐级超分，Imagen 的前身 |
| 2021-05 | ★ [ADM](https://arxiv.org/abs/2105.05233) `arXiv:2105.05233` | OpenAI | 改进 U-Net 并提出分类器引导，ImageNet 上首次全面超过 BigGAN |
| 2021-04 | [SR3](https://arxiv.org/abs/2104.07636) `arXiv:2104.07636` | Google | 条件扩散做图像超分辨率 |
| 2021-02 | [Improved DDPM](https://arxiv.org/abs/2102.09672) `arXiv:2102.09672` | OpenAI | 余弦噪声表 + 学习方差，提升似然并减少采样步数 |
| 2020-11 | ★ [Score SDE](https://arxiv.org/abs/2011.13456) `arXiv:2011.13456` | Stanford/Google | 用随机微分方程统一 DDPM 与 NCSN，提出概率流 ODE，奠定连续时间视角 |
| 2020-10 | ★ [DDIM](https://arxiv.org/abs/2010.02502) `arXiv:2010.02502` | Stanford | 非马尔可夫的确定性采样，同一模型可用 10–50 步生成，并支持潜变量插值与反演 |
| 2020-06 | ★ [DDPM](https://arxiv.org/abs/2006.11239) `arXiv:2006.11239` | UC Berkeley | 把训练简化为"预测所加噪声"的均方误差，扩散模型第一次在图像质量上追平 GAN |
| 2020-06 | [NCSNv2](https://arxiv.org/abs/2006.09011) `arXiv:2006.09011` | Stanford | 给出噪声尺度、步长等设计原则，把分数模型推到 256 分辨率 |
| 2019-07 | [NCSN](https://arxiv.org/abs/1907.05600) `arXiv:1907.05600` | Stanford | 多尺度噪声下的去噪分数匹配 + 退火 Langevin 采样，"分数模型"一脉的开端 |
| 2015-03 | [Sohl-Dickstein 2015](https://arxiv.org/abs/1503.03585) `arXiv:1503.03585` | Stanford/Berkeley | 扩散模型的起点：用非平衡热力学的思路，逐步加噪破坏数据，再学习逐步还原 |

### 流匹配/插值/桥（31）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-04 | 🆕 [NF + 迭代去噪](https://arxiv.org/abs/2604.20041) `arXiv:2604.20041` | Apple | 在归一化流上叠加迭代去噪，缩小与扩散模型的质量差距 |
| 2025-12 | 🆕 [BiFlow](https://arxiv.org/abs/2512.10953) `arXiv:2512.10953` | MIT | 双向归一化流：学"数据→噪声"也学近似逆映射，单次评估生成 |
| 2025-11 | 🆕 [STARFlow-V](https://arxiv.org/abs/2511.20462) `arXiv:2511.20462` | Apple | 归一化流做端到端视频生成 |
| 2025-10 | 🆕 [TM vs FM](https://arxiv.org/abs/2510.17991) `arXiv:2510.17991` | KAIST/Amazon | 分析转移匹配何时、为何能在少步时胜过流匹配（目标为多峰时） |
| 2025-06 | [Transition Matching](https://arxiv.org/abs/2506.23589) `arXiv:2506.23589` | Meta | 把每一步建模为可学习的随机转移核，统一扩散、流与自回归的生成步骤 |
| 2025-06 | [Edit Flows](https://arxiv.org/abs/2506.09018) `arXiv:2506.09018` | Meta | 序列上的插入/删除/替换编辑操作作为流，生成长度可变序列 |
| 2025-06 | [STARFlow](https://arxiv.org/abs/2506.06276) `arXiv:2506.06276` | Apple | 把 TarFlow 搬到潜空间并扩展到高分辨率文生图 |
| 2025-04 | [Adjoint Sampling](https://arxiv.org/abs/2504.11713) `arXiv:2504.11713` | Meta | 基于 adjoint matching 的可扩展扩散采样器（仅给能量函数，无数据） |
| 2024-12 | [TarFlow](https://arxiv.org/abs/2412.06329) `arXiv:2412.06329` | Apple | Transformer 自回归归一化流，证明归一化流也能生成高质量图像 |
| 2024-10 | [Generator Matching](https://arxiv.org/abs/2410.20587) `arXiv:2410.20587` | MIT/Meta | 把扩散、流、跳跃过程统一为"匹配马尔可夫过程的生成元" |
| 2024-09 | ★ [Adjoint Matching](https://arxiv.org/abs/2409.08861) `arXiv:2409.08861` | Meta | 把奖励微调写成无记忆随机最优控制，流/扩散模型的有理论保证的微调 |
| 2024-07 | [Discrete FM](https://arxiv.org/abs/2407.15595) `arXiv:2407.15595` | Meta | 离散状态上的流匹配，用于代码与文本生成 |
| 2024-05 | [改进 Rectified Flow](https://arxiv.org/abs/2405.20320) `arXiv:2405.20320` | CMU | 一次 reflow 即可，配合改进训练，少步 FID 显著改善 |
| 2024-05 | [Fisher Flow](https://arxiv.org/abs/2405.14664) `arXiv:2405.14664` | Oxford | 用 Fisher-Rao 几何把类别分布映射到球面上做流匹配 |
| 2024-02 | [iDEM](https://arxiv.org/abs/2402.06121) `arXiv:2402.06121` | Mila | 迭代去噪能量匹配，从玻尔兹曼分布采样 |
| 2024-02 | [Dirichlet FM](https://arxiv.org/abs/2402.05841) `arXiv:2402.05841` | MIT | 在单纯形上用 Dirichlet 路径做流匹配，用于 DNA 序列设计 |
| 2024-02 | [Multiflow](https://arxiv.org/abs/2402.04997) `arXiv:2402.04997` | MIT/Oxford | 离散状态 CTMC 流，支持连续-离散混合（蛋白质序列与结构共同设计） |
| 2024-01 | [SiT](https://arxiv.org/abs/2401.08740) `arXiv:2401.08740` | NYU | DiT 骨干 + 插值框架，系统比较 ODE/SDE、预测目标与路径，FID 2.06 |
| 2023-12 | [SOCM](https://arxiv.org/abs/2312.02027) `arXiv:2312.02027` | Meta | 随机最优控制匹配：把控制问题转为回归 |
| 2023-07 | [[SF]²M](https://arxiv.org/abs/2307.03672) `arXiv:2307.03672` | Mila | 用分数 + 流匹配免模拟学习薛定谔桥 |
| 2023-03 | [DSBM](https://arxiv.org/abs/2303.16852) `arXiv:2303.16852` | Oxford | 薛定谔桥匹配：迭代马尔可夫拟合，免模拟训练桥 |
| 2023-03 | [随机插值统一框架](https://arxiv.org/abs/2303.08797) `arXiv:2303.08797` | NYU | 把流与扩散统一为"随机插值"，ODE/SDE 的扩散系数可在训练后自由选择 |
| 2023-02 | [I²SB](https://arxiv.org/abs/2302.05872) `arXiv:2302.05872` | NVIDIA | 图像到图像的薛定谔桥，用于修复、超分等 |
| 2023-02 | [Riemannian FM](https://arxiv.org/abs/2302.03660) `arXiv:2302.03660` | Meta | 把流匹配推广到流形（球面、环面、蛋白质构象等） |
| 2023-02 | [OT-CFM](https://arxiv.org/abs/2302.00482) `arXiv:2302.00482` | Mila | 小批量最优传输配对，降低流的弯曲度与方差 |
| 2022-10 | ★ [Flow Matching](https://arxiv.org/abs/2210.02747) `arXiv:2210.02747` | Meta/Weizmann | 免模拟训练 CNF：回归条件速度场即可，最优传输路径训练更快、生成更直 |
| 2022-09 | [Stochastic Interpolants](https://arxiv.org/abs/2209.15571) `arXiv:2209.15571` | NYU | 用插值构造连续流的概率路径，与流匹配同期独立提出 |
| 2022-09 | ★ [Rectified Flow](https://arxiv.org/abs/2209.03003) `arXiv:2209.03003` | UT Austin | 学习噪声与数据之间的直线流；reflow 让轨迹变直，为一步生成铺路 |
| 2021-06 | [DSB](https://arxiv.org/abs/2106.01357) `arXiv:2106.01357` | Oxford | 扩散薛定谔桥：在任意两个分布之间学习最优随机桥 |
| 2018-10 | [FFJORD](https://arxiv.org/abs/1810.01367) `arXiv:1810.01367` | Toronto | 用迹估计做自由形式 CNF，但需模拟 ODE 训练，代价高 |
| 2018-06 | [Neural ODE](https://arxiv.org/abs/1806.07366) `arXiv:1806.07366` | Toronto | 连续深度网络与连续归一化流（CNF）的起点 |

### 采样器（8）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-10 | 🆕 [GAS](https://arxiv.org/abs/2510.17699) `arXiv:2510.17699` |  | 广义对抗求解器：学习 ODE 离散化参数并加对抗损失，少步采样更清晰 |
| 2023-10 | [DPM-Solver-v3](https://arxiv.org/abs/2310.13268) `arXiv:2310.13268` | Tsinghua | 用经验模型统计量优化求解器参数 |
| 2023-06 | [Restart Sampling](https://arxiv.org/abs/2306.14878) `arXiv:2306.14878` | MIT | 交替前向加噪与反向 ODE，兼得 ODE 的速度与 SDE 的纠错能力 |
| 2023-02 | [UniPC](https://arxiv.org/abs/2302.04867) `arXiv:2302.04867` | Tsinghua | 统一预测-校正框架，极少步数下效果更好 |
| 2022-11 | [DPM-Solver++](https://arxiv.org/abs/2211.01095) `arXiv:2211.01095` | Tsinghua | 面向带引导采样的多步求解器，长期是 Stable Diffusion 默认之一 |
| 2022-06 | ★ [DPM-Solver](https://arxiv.org/abs/2206.00927) `arXiv:2206.00927` | Tsinghua | 利用扩散 ODE 的半线性结构做高阶求解，约 10 步出图 |
| 2022-04 | [DEIS](https://arxiv.org/abs/2204.13902) `arXiv:2204.13902` | Georgia Tech | 指数积分器加速扩散采样 |
| 2022-02 | [PNDM](https://arxiv.org/abs/2202.09778) `arXiv:2202.09778` | Zhejiang | 把扩散采样看作流形上的伪数值方法 |

### 一步/少步模型（32）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | ★ 🆕 [Priming Models](https://arxiv.org/abs/2610.20331) `arXiv:2610.20331` | SNU/Harvard | Are We Overthinking One-Step Generation?：两阶段简单框架，一步 ImageNet-256 FID 1.19，算力约为同类十分之一 |
| 2026-05 | 🆕 [Normalizing Trajectory Models](https://arxiv.org/abs/2605.08078) `arXiv:2605.08078` | Apple | 用归一化流建模每一个粗步长转移，少步时仍保持精确似然 |
| 2026-04 | 🆕 [一步生成扩展到文本条件](https://arxiv.org/abs/2604.18168) `arXiv:2604.18168` | AMAP | 把 MeanFlow 类一步生成从类别条件推广到文生图 |
| 2026-04 | 🆕 [Discrete Flow Maps](https://arxiv.org/abs/2604.09784) `arXiv:2604.09784` | Oxford/MIT/NYU | 把流映射推广到离散数据，语言模型少步并行生成 |
| 2026-02 | 🆕 [Categorical Flow Maps](https://arxiv.org/abs/2602.12233) `arXiv:2602.12233` | Amsterdam/Oxford | 类别数据的流映射自蒸馏，加速分子/文本生成 |
| 2026-02 | 🆕 [Duality Models](https://arxiv.org/abs/2602.17682) `arXiv:2602.17682` | Westlake/MBZUAI | 极简一步生成范式，2 步 ImageNet-256 FID 1.79 |
| 2026-02 | ★ 🆕 [Drifting Models](https://arxiv.org/abs/2602.04770) `arXiv:2602.04770` | MIT/Harvard | 训练过程中让生成分布"漂移"至数据分布，天然一步生成：ImageNet-256 潜空间 FID 1.54、像素空间 1.61 |
| 2026-01 | 🆕 [pMF](https://arxiv.org/abs/2601.22158) `arXiv:2601.22158` | MIT | Pixel MeanFlow：不用潜空间、不用多步，像素空间一步生成 ImageNet-256 FID 2.22 |
| 2026-01 | 🆕 [Meta Flow Maps](https://arxiv.org/abs/2601.14430) `arXiv:2601.14430` | Oxford/Harvard | 随机流映射近似后验 p(x1\|xt)，使奖励对齐可扩展 |
| 2025-12 | 🆕 [Self-E](https://arxiv.org/abs/2512.22374) `arXiv:2512.22374` | Adobe/HKU | 从头训练、可任意步推理的文生图：模型用自身分数评估自己的样本 |
| 2025-12 | 🆕 [SoFlow](https://arxiv.org/abs/2512.15657) `arXiv:2512.15657` | Princeton | 解流（solution flow）模型，从头训练一步生成，同架构下优于 MeanFlow |
| 2025-12 | 🆕 [TwinFlow](https://arxiv.org/abs/2512.05150) `arXiv:2512.05150` | inclusionAI | 自对抗流：不需要判别器或冻结教师，把一步生成扩展到 20B 级大模型 |
| 2025-12 | ★ 🆕 [iMF](https://arxiv.org/abs/2512.02012) `arXiv:2512.02012` | CMU/MIT/Adobe | Improved Mean Flows：重写训练目标与引导方式，从头训练 1-NFE FID 1.72 |
| 2025-11 | 🆕 [流映射测试时扩展](https://arxiv.org/abs/2511.22688) `arXiv:2511.22688` | NVIDIA | 用流映射在采样中途"看到"终点，奖励引导不再病态 |
| 2025-11 | 🆕 [Adversarial Flow Models](https://arxiv.org/abs/2511.22475) `arXiv:2511.22475` | ByteDance Seed | 对抗目标训练的流模型，直接学一步/少步映射，XL/2 1-NFE FID 2.38 |
| 2025-11 | 🆕 [改进 MeanFlow 训练](https://arxiv.org/abs/2511.19065) `arXiv:2511.19065` | TU Munich/Google | 先学好瞬时速度再学大跨度平均速度，1-NFE FID 2.87 |
| 2025-11 | 🆕 [TVM](https://arxiv.org/abs/2511.19797) `arXiv:2511.19797` | Luma AI | Terminal Velocity Matching：在终点而非起点约束转移，1 步 FID 3.29、4 步 1.99 |
| 2025-11 | 🆕 [MeanFlow + RAE](https://arxiv.org/abs/2511.13019) `arXiv:2511.13019` | Sony/Stanford | 在表征自编码器潜空间训练 MeanFlow，1-NFE FID 2.03，训练成本降 83% |
| 2025-10 | 🆕 [Decoupled MeanFlow](https://arxiv.org/abs/2510.24474) `arXiv:2510.24474` | KAIST | 不改架构就把预训练流模型变成流映射，1 步 FID 2.16（256）/2.12（512） |
| 2025-10 | 🆕 [α-Flow](https://arxiv.org/abs/2510.20771) `arXiv:2510.20771` | Snap/Michigan | 把 MeanFlow 目标分解为轨迹流匹配 + 轨迹一致性，课程式训练，1-NFE FID 2.58 |
| 2025-09 | [TiM](https://arxiv.org/abs/2509.04394) `arXiv:2509.04394` | CUHK 等 | Transition Models：学习任意步长的状态转移，从一步到多步质量单调提升 |
| 2025-08 | [Modular MeanFlow](https://arxiv.org/abs/2508.17426) `arXiv:2508.17426` |  | 带梯度调节的 MeanFlow 损失族，训练更稳 |
| 2025-07 | [SplitMeanFlow](https://arxiv.org/abs/2507.16884) `arXiv:2507.16884` | ByteDance | 用区间可加性代替微分恒等式，免 JVP，已用于语音合成 |
| 2025-05 | ★ [MeanFlow](https://arxiv.org/abs/2505.13447) `arXiv:2505.13447` | CMU/MIT | 学习"平均速度"而非瞬时速度，从头训练一步 ImageNet-256 FID 3.43 |
| 2025-03 | [IMM](https://arxiv.org/abs/2503.07565) `arXiv:2503.07565` | Luma AI | 归纳矩匹配，单阶段训练少步模型，ImageNet-256 8 步 FID 1.99 |
| 2024-10 | [Shortcut Models](https://arxiv.org/abs/2410.12557) `arXiv:2410.12557` | UC Berkeley | 以步长为条件，单网络单阶段训练，支持一步到多步 |
| 2024-10 | ★ [sCM](https://arxiv.org/abs/2410.11081) `arXiv:2410.11081` | OpenAI | 简化、稳定并扩展连续时间一致性模型，2 步 ImageNet-512 FID 1.88 |
| 2024-06 | [ECT](https://arxiv.org/abs/2406.14548) `arXiv:2406.14548` | CMU | 从预训练扩散模型出发做一致性微调，训练成本大降 |
| 2024-06 | [Flow Map Matching](https://arxiv.org/abs/2406.07507) `arXiv:2406.07507` | NYU | 在随机插值框架下统一一致性模型、渐进蒸馏，学习"流映射" |
| 2023-10 | [iCT](https://arxiv.org/abs/2310.14189) `arXiv:2310.14189` | OpenAI | 改进一致性训练（伪 Huber 损失、课程），不用教师 |
| 2023-10 | [CTM](https://arxiv.org/abs/2310.02279) `arXiv:2310.02279` | Sony | 一致性轨迹模型：学习任意两时刻间的跳跃 |
| 2023-03 | ★ [Consistency Models](https://arxiv.org/abs/2303.01469) `arXiv:2303.01469` | OpenAI | 学习把轨迹上任意点映射到起点的函数，可蒸馏也可从头训练 |

### 蒸馏加速（35）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [IDM](https://arxiv.org/abs/2610.39633) `arXiv:2610.39633` | inclusionAI | Intrinsic Distribution Matching：改进 DMD 的分数差信号，少步文生图更能还原提示细节 |
| 2026-10 | 🆕 [简化 On-Policy 蒸馏](https://arxiv.org/abs/2610.18472) `arXiv:2610.18472` | Microsoft | 简单的 on-policy 速度回归就能做好 4-NFE 生成，无需辅助分数网络 |
| 2026-10 | 🆕 [DuoMatching](https://arxiv.org/abs/2610.03543) `arXiv:2610.03543` | ByteDance | 联合分布 + 边缘分布双重匹配，用于少步流式视频生成 |
| 2026-07 | 🆕 [OPD 与 CFG](https://arxiv.org/abs/2607.24731) `arXiv:2607.24731` | Qwen | 研究 on-policy 扩散蒸馏在 CFG 下应如何设计 |
| 2026-06 | 🆕 [DanceOPD](https://arxiv.org/abs/2606.27377) `arXiv:2606.27377` | ByteDance Seed | On-policy 生成场蒸馏：把文生图、局部编辑、全局编辑等多个教师合并进一个模型 |
| 2026-06 | 🆕 [Z-Image Turbo++](https://arxiv.org/abs/2606.12575) `arXiv:2606.12575` | Tongyi Lab | 从 8 步 Z-Image-Turbo 继续蒸出高保真 2 步模型 |
| 2026-06 | 🆕 [Qwen-Image-Flash](https://arxiv.org/abs/2606.03746) `arXiv:2606.03746` | Qwen | 重新审视少步蒸馏训练配方，用于 Qwen-Image 系列 |
| 2026-05 | 🆕 [大规模 MeanFlow 蒸馏](https://arxiv.org/abs/2605.17834) `arXiv:2605.17834` |  | 稳定并扩展 MeanFlow 用于大模型蒸馏 |
| 2026-05 | 🆕 [AnyFlow](https://arxiv.org/abs/2605.13724) `arXiv:2605.13724` | MIT/NVIDIA | 在线（on-policy）流映射蒸馏，视频模型步数越多越好，不再随步数退化 |
| 2026-05 | 🆕 [Flow-OPD](https://arxiv.org/abs/2605.08063) `arXiv:2605.08063` |  | 把 LLM 的 on-policy 蒸馏用到流匹配多任务对齐，缓解多奖励"跷跷板" |
| 2026-05 | 🆕 [连续时间 DMD](https://arxiv.org/abs/2605.06376) `arXiv:2605.06376` |  | 把 DMD 推广到连续时间，与一致性蒸馏互补 |
| 2026-02 | 🆕 [DP-DMD](https://arxiv.org/abs/2602.03139) `arXiv:2602.03139` | CityU HK | 保持多样性的 DMD，不依赖感知或对抗损失 |
| 2026-01 | 🆕 [TMD](https://arxiv.org/abs/2601.09881) `arXiv:2601.09881` | NVIDIA | 转移匹配蒸馏：把大视频模型蒸成少步生成器 |
| 2025-12 | 🆕 [文生图少步蒸馏实践指南](https://arxiv.org/abs/2512.13006) `arXiv:2512.13006` | Alibaba DAMO | 在强文生图教师上系统比较各种蒸馏方法 |
| 2025-11 | ★ 🆕 [Decoupled DMD](https://arxiv.org/abs/2511.22677) `arXiv:2511.22677` | Tongyi Lab | 发现 DMD 起主要作用的是"CFG 增强"项，分布匹配只是正则；用于 Z-Image-Turbo |
| 2025-11 | 🆕 [无数据流映射蒸馏](https://arxiv.org/abs/2511.19428) `arXiv:2511.19428` | NYU/MIT | 只从先验噪声出发蒸馏流映射，避免外部数据带来的教师-数据错配 |
| 2025-10 | 🆕 [Phased DMD](https://arxiv.org/abs/2510.27684) `arXiv:2510.27684` | SenseTime | 把 SNR 区间分段，在子区间内做分数匹配，缓解少步模型多样性下降 |
| 2025-10 | 🆕 [π-Flow](https://arxiv.org/abs/2510.14974) `arXiv:2510.14974` | Stanford/Adobe | 学生输出一个"策略"（随时间变化的速度），通过模仿教师速度蒸馏，ImageNet 1-NFE FID 2.85，FLUX/Qwen-Image 4 步保持多样性 |
| 2025-10 | ★ 🆕 [rCM](https://arxiv.org/abs/2510.08431) `arXiv:2510.08431` | NVIDIA/Tsinghua | 给 sCM 加分数正则（反向散度），首次把连续时间一致性蒸馏扩展到 10B+ 文生图/视频 |
| 2025-09 | [SD3.5-Flash](https://arxiv.org/abs/2509.21318) `arXiv:2509.21318` | Stability AI | 面向消费级设备的少步分布引导蒸馏 |
| 2025-06 | [Align Your Flow](https://arxiv.org/abs/2506.14603) `arXiv:2506.14603` | NVIDIA | 连续时间流映射蒸馏，任意步数都稳定 |
| 2025-03 | [SANA-Sprint](https://arxiv.org/abs/2503.09641) `arXiv:2503.09641` | NVIDIA | 连续时间一致性蒸馏 + 对抗，1–4 步高分辨率文生图 |
| 2024-05 | [DMD2](https://arxiv.org/abs/2405.14867) `arXiv:2405.14867` | MIT/Adobe | 去掉回归损失、加 GAN 损失与双时间尺度更新，一步生成超过教师 |
| 2024-05 | [PeRFlow](https://arxiv.org/abs/2405.07510) `arXiv:2405.07510` | ByteDance | 分段 Rectified Flow，即插即用加速 |
| 2024-04 | [Hyper-SD](https://arxiv.org/abs/2404.13686) `arXiv:2404.13686` | ByteDance | 分段一致性蒸馏 + 人类反馈，统一 1–8 步 |
| 2024-04 | [SiD](https://arxiv.org/abs/2404.04057) `arXiv:2404.04057` | UT Austin | 分数恒等蒸馏，无数据、指数级快地蒸出一步生成器 |
| 2024-03 | [LADD](https://arxiv.org/abs/2403.12015) `arXiv:2403.12015` | Stability AI | 在潜空间用教师特征做判别器的对抗蒸馏（SD3 Turbo） |
| 2024-02 | [SDXL-Lightning](https://arxiv.org/abs/2402.13929) `arXiv:2402.13929` | ByteDance | 渐进式对抗蒸馏，1–8 步 SDXL |
| 2023-11 | ★ [DMD](https://arxiv.org/abs/2311.18828) `arXiv:2311.18828` | MIT/Adobe | 分布匹配蒸馏：用"真实分数 − 假分数"之差作梯度，学生分布对齐教师 |
| 2023-11 | [ADD](https://arxiv.org/abs/2311.17042) `arXiv:2311.17042` | Stability AI | 对抗扩散蒸馏（SDXL Turbo）：分数蒸馏 + 判别器，1–4 步 |
| 2023-11 | [LCM-LoRA](https://arxiv.org/abs/2311.05556) `arXiv:2311.05556` | Tsinghua | 把加速能力做成可插拔 LoRA，社区广泛使用 |
| 2023-10 | [LCM](https://arxiv.org/abs/2310.04378) `arXiv:2310.04378` | Tsinghua | 潜空间一致性蒸馏，2–4 步文生图 |
| 2023-09 | [InstaFlow](https://arxiv.org/abs/2309.06380) `arXiv:2309.06380` | UT Austin | Rectified Flow 的 reflow + 蒸馏，首个一步文生图 SD 级模型 |
| 2023-05 | [VSD](https://arxiv.org/abs/2305.16213) `arXiv:2305.16213` | Tsinghua | 变分分数蒸馏（ProlificDreamer），后来 DMD 类方法的思想来源之一 |
| 2022-02 | ★ [Progressive Distillation](https://arxiv.org/abs/2202.00512) `arXiv:2202.00512` | Google | 每轮把两步蒸馏成一步，反复减半步数；提出 v 预测 |

### 引导与条件采样（24）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [SWIFT](https://arxiv.org/abs/2610.25844) `arXiv:2610.25844` | Alibaba | 训练时用冻结的弱去噪器做目标外推，推理零额外开销 |
| 2026-08 | 🆕 [重新评估 CFG 改进方法](https://arxiv.org/abs/2608.16786) `arXiv:2608.16786` | Yandex | 在现代 DiT 文生图模型上系统复评各种 CFG 变体 |
| 2026-08 | 🆕 [对抗学习 CFG 日程](https://arxiv.org/abs/2608.14038) `arXiv:2608.14038` | Google DeepMind | 对抗地学习随时间、样本与条件变化的引导系数 |
| 2026-07 | 🆕 [CFG 的解析分布](https://arxiv.org/abs/2607.19725) `arXiv:2607.19725` | SJTU | 用概率流 ODE 给出 CFG 实际采样分布的解析形式，用于设计引导日程 |
| 2026-01 | 🆕 [流形投影 CFG](https://arxiv.org/abs/2601.21892) `arXiv:2601.21892` | HKUST | 把 CFG 解释为流形投影，改进流匹配的引导稳定性 |
| 2025-10 | 🆕 [TAG（时间对齐）](https://arxiv.org/abs/2510.11057) `arXiv:2510.11057` | KAIST | 时间对齐引导：把偏离流形的样本拉回对应噪声水平 |
| 2025-10 | 🆕 [TAG](https://arxiv.org/abs/2510.04533) `arXiv:2510.04533` | KAIST | 切向放大引导：放大切向分量，减少幻觉 |
| 2025-09 | [CFG 的分阶段动力学](https://arxiv.org/abs/2509.22007) `arXiv:2509.22007` | Tsinghua | 多峰条件分布下 CFG 的三阶段动力学分析 |
| 2025-05 | [NAG](https://arxiv.org/abs/2505.21179) `arXiv:2505.21179` | Surrey | 归一化注意力引导：在注意力空间做负向引导，少步模型也能用负提示 |
| 2025-03 | [CFG-Zero*](https://arxiv.org/abs/2503.18886) `arXiv:2503.18886` | NTU | 针对流匹配的 CFG 修正：最优缩放 + 前几步置零 |
| 2025-02 | [Model-guidance](https://arxiv.org/abs/2502.12154) `arXiv:2502.12154` | Microsoft | 训练目标直接建模引导后的分布，无需 CFG 也能达到 SOTA |
| 2025-01 | [GFT](https://arxiv.org/abs/2501.15420) `arXiv:2501.15420` | Tsinghua | 无引导训练：训练时内化 CFG 效果，采样只需一次前向 |
| 2024-10 | [APG](https://arxiv.org/abs/2410.02416) `arXiv:2410.02416` | Disney/ETH | 自适应投影引导：去掉与条件预测平行的分量，消除过饱和 |
| 2024-07 | [DAPS](https://arxiv.org/abs/2407.01521) `arXiv:2407.01521` | Caltech | 解耦噪声退火的后验采样，逆问题更稳 |
| 2024-06 | [CFG++](https://arxiv.org/abs/2406.08070) `arXiv:2406.08070` | KAIST | 流形约束的 CFG，缓解高引导下偏离数据流形 |
| 2024-06 | ★ [Autoguidance](https://arxiv.org/abs/2406.02507) `arXiv:2406.02507` | NVIDIA | 用"更差的自己"（小/欠训练模型）代替无条件模型作引导，ImageNet-512 FID 1.25 |
| 2024-04 | [Guidance Interval](https://arxiv.org/abs/2404.07724) `arXiv:2404.07724` | NVIDIA/Aalto | 只在中间噪声区间施加 CFG，同时改善质量与多样性 |
| 2024-03 | [PAG](https://arxiv.org/abs/2403.17377) `arXiv:2403.17377` |  | 扰动注意力引导：把自注意力替换为单位阵得到弱预测 |
| 2023-02 | [Universal Guidance](https://arxiv.org/abs/2302.07121) `arXiv:2302.07121` | Maryland | 用任意现成网络（分割、人脸识别等）引导，无需重训 |
| 2022-10 | [SAG](https://arxiv.org/abs/2210.00939) `arXiv:2210.00939` |  | 自注意力引导：模糊高注意力区域构造"差版本"作引导 |
| 2022-09 | [DPS](https://arxiv.org/abs/2209.14687) `arXiv:2209.14687` | KAIST | 扩散后验采样：用近似似然梯度求解含噪逆问题 |
| 2022-07 | ★ [CFG](https://arxiv.org/abs/2207.12598) `arXiv:2207.12598` | Google | 无分类器引导：同一模型学有/无条件两种预测，相减放大条件信号；几乎所有文生图/视频系统的默认组件 |
| 2022-01 | [DDRM](https://arxiv.org/abs/2201.11793) `arXiv:2201.11793` | Technion | 去噪扩散恢复模型：线性逆问题的高效求解 |
| 2021-12 | [GLIDE](https://arxiv.org/abs/2112.10741) `arXiv:2112.10741` | OpenAI | 文本条件扩散 + CFG，比较 CLIP 引导与无分类器引导 |

### 架构与扩展（14）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🆕 [Abra](https://arxiv.org/abs/2608.17286) `arXiv:2608.17286` | Luma AI | 文生图流匹配 Transformer 的计算最优扩展律研究，跨三个数量级算力 |
| 2026-07 | 🆕 [文本条件扩展性](https://arxiv.org/abs/2607.29679) `arXiv:2607.29679` | ByteDance Seed | 收敛后的扩散损失随提示中"有效信息量"呈规律变化，研究文本条件的扩展性质 |
| 2026-05 | 🆕 [Lens](https://arxiv.org/abs/2605.21573) `arXiv:2605.21573` | Microsoft | 3.8B 文生图，仅用同类约 19% 训练算力达到 6B+ 模型水平 |
| 2026-05 | 🆕 [千层 DiT](https://arxiv.org/abs/2605.06169) `arXiv:2605.06169` |  | 诊断深层 DiT 的"均值主导坍缩"，用均值-方差分离残差训练 1000 层模型 |
| 2025-10 | 🆕 [ProMoE](https://arxiv.org/abs/2510.24711) `arXiv:2510.24711` | Alibaba | 显式路由引导的 MoE DiT，解决视觉 token 冗余导致专家分化不足 |
| 2025-04 | [DDT](https://arxiv.org/abs/2504.05741) `arXiv:2504.05741` | Nanjing Univ | 解耦编码器（语义）与解码器（高频）的 DiT，ImageNet-256 FID 1.31 |
| 2024-10 | [DiT 扩展律](https://arxiv.org/abs/2410.08184) `arXiv:2410.08184` | Shanghai AI Lab | 扩散 Transformer 的损失随算力呈幂律下降，可预测最优模型规模 |
| 2024-07 | [DiT-MoE](https://arxiv.org/abs/2407.11633) `arXiv:2407.11633` |  | 稀疏混合专家扩散 Transformer，扩展到 16B 参数 |
| 2024-01 | [HDiT](https://arxiv.org/abs/2401.11605) `arXiv:2401.11605` | Stability AI | 沙漏结构 DiT，像素空间直接生成百万像素图像 |
| 2023-06 | [MaskDiT](https://arxiv.org/abs/2306.09305) `arXiv:2306.09305` | NVIDIA | 训练时随机掩掉一半 patch，大幅降低训练成本 |
| 2023-03 | [MDTv2](https://arxiv.org/abs/2303.14389) `arXiv:2303.14389` |  | 掩码扩散 Transformer，学习上下文关系，ImageNet-256 FID 1.58 |
| 2022-12 | ★ [DiT](https://arxiv.org/abs/2212.09748) `arXiv:2212.09748` | UC Berkeley/NYU | 用 ViT 替换 U-Net 做潜空间扩散，FID 随 Gflops 稳定下降，开启 Transformer 扩散时代 |
| 2022-09 | [U-ViT](https://arxiv.org/abs/2209.12152) `arXiv:2209.12152` | Tsinghua | 把时间、条件、图像块都当 token 的 ViT 骨干，带长跳连 |
| 2015-05 | [U-Net](https://arxiv.org/abs/1505.04597) `arXiv:1505.04597` | Freiburg | 编码器-解码器 + 跳连，2020–2023 年扩散模型的标准骨干 |

### 潜空间/Tokenizer/像素（38）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [RoAR-VAE](https://arxiv.org/abs/2610.26216) `arXiv:2610.26216` | Microsoft | 用鲁棒预测提升潜空间可扩散性 |
| 2026-10 | 🆕 [表征引导像素 DiT](https://arxiv.org/abs/2610.12978) `arXiv:2610.12978` | Alibaba | 表征质量更好不一定生成更好：研究什么样的表征适合引导像素扩散 |
| 2026-09 | 🆕 [DC-SAE](https://arxiv.org/abs/2609.39222) `arXiv:2609.39222` | PKU | 深度压缩语义自编码器，高压缩下扩散仍快速收敛 |
| 2026-09 | 🆕 [像素扩散对抗后训练](https://arxiv.org/abs/2609.38170) `arXiv:2609.38170` | Adobe | 用对抗学习补足像素扩散缺失的细尺度统计 |
| 2026-09 | 🆕 [FuseReg](https://arxiv.org/abs/2609.31620) `arXiv:2609.31620` | USC | 正则化 RAE 的多层特征融合，缩小重建-生成差距 |
| 2026-08 | 🆕 [像素空间文生图实证研究](https://arxiv.org/abs/2608.16887) `arXiv:2608.16887` | Tongyi Lab | 给出能媲美潜空间模型的像素空间文生图训练配方 |
| 2026-08 | 🆕 [V-RAE](https://arxiv.org/abs/2608.13556) `arXiv:2608.13556` | NUS | 把表征自编码器思路用到视频：潜空间带高层语义 |
| 2026-05 | 🆕 [PiD](https://arxiv.org/abs/2605.23902) `arXiv:2605.23902` | NVIDIA | 用像素扩散替代重建式解码器，快速解码高分辨率潜变量 |
| 2026-05 | 🆕 [Qwen-Image-VAE-2.0](https://arxiv.org/abs/2605.13565) `arXiv:2605.13565` | Qwen | 高压缩 VAE：兼顾重建保真与"可扩散性" |
| 2026-02 | 🆕 [Unified Latents](https://arxiv.org/abs/2602.17270) `arXiv:2602.17270` | Google DeepMind | 编码器噪声与先验最小噪声挂钩，潜变量同时由扩散先验正则、由扩散解码器解码，得到比特率上界 |
| 2026-02 | 🆕 [PixelGen](https://arxiv.org/abs/2602.02493) `arXiv:2602.02493` | PKU | 在 JiT 式像素扩散上加感知损失，80 epoch 超过潜空间基线 |
| 2026-01 | 🆕 [RAE 文生图扩展](https://arxiv.org/abs/2601.16208) `arXiv:2601.16208` | NYU | 把 RAE 推广到大规模文生图，收敛更快、微调更不易过拟合 |
| 2025-12 | 🆕 [SemanticGen](https://arxiv.org/abs/2512.20619) `arXiv:2512.20619` | Kuaishou | 先在紧凑语义空间生成视频的全局规划，再补充细节，长视频收敛更快 |
| 2025-12 | 🆕 [语义与重建兼顾](https://arxiv.org/abs/2512.17909) `arXiv:2512.17909` | Adobe | 让表征编码器同时具备语义与重建能力，用于文生图与编辑 |
| 2025-12 | 🆕 [VTP](https://arxiv.org/abs/2512.13687) `arXiv:2512.13687` | HUST/MiniMax | 面向生成的视觉 tokenizer 可扩展预训练，重建更好不等于生成更好 |
| 2025-12 | 🆕 [REPA 关键是什么](https://arxiv.org/abs/2512.10794) `arXiv:2512.10794` | ANU/Adobe | REPA 的收益主要来自目标表征的空间结构而非全局信息 |
| 2025-12 | 🆕 [一层就够](https://arxiv.org/abs/2512.07829) `arXiv:2512.07829` | Apple | 只加一层适配即可把预训练视觉编码器用作生成潜空间 |
| 2025-12 | 🆕 [DMVAE](https://arxiv.org/abs/2512.07778) `arXiv:2512.07778` | Tencent/PKU | 显式用分布匹配塑造潜空间分布 |
| 2025-11 | 🆕 [PixelDiT](https://arxiv.org/abs/2511.20645) `arXiv:2511.20645` | NVIDIA | patch 级 + 像素级双层 DiT，像素空间 FID 1.61 |
| 2025-11 | 🆕 [DiP](https://arxiv.org/abs/2511.18822) `arXiv:2511.18822` |  | 像素空间 DiT + 小型细节头，推理快 10 倍，FID 1.79 |
| 2025-11 | 🆕 [DeCo](https://arxiv.org/abs/2511.19365) `arXiv:2511.19365` | PKU | 频率解耦像素扩散：DiT 建模低频语义，轻量解码器补高频，FID 1.62 |
| 2025-10 | 🆕 [SVG](https://arxiv.org/abs/2510.15301) `arXiv:2510.15301` | Tsinghua/Kling | 不用 VAE：DINO 特征 + 轻量残差分支构成潜空间，训练与推理都更快 |
| 2025-10 | 🆕 [EPG](https://arxiv.org/abs/2510.12586) `arXiv:2510.12586` | AMAP/NVIDIA | There is No VAE：先自监督预训练再端到端训练像素扩散，ImageNet-256 FID 1.58 |
| 2025-10 | ★ 🆕 [RAE](https://arxiv.org/abs/2510.11690) `arXiv:2510.11690` | NYU | 用冻结的表征编码器（DINOv2 等）+ 训练好的解码器替换 VAE，在高维语义潜空间做扩散，FID 1.13 |
| 2025-08 | [DC-AE 1.5](https://arxiv.org/abs/2508.00413) `arXiv:2508.00413` | MIT/NVIDIA | 结构化潜空间，缓解高通道数潜空间难以收敛的问题 |
| 2025-07 | [l-DeTok](https://arxiv.org/abs/2507.15856) `arXiv:2507.15856` | USC/MIT | 潜去噪 tokenizer：训练编码器使其潜变量在加噪后仍可重建 |
| 2025-07 | [REG](https://arxiv.org/abs/2507.01467) `arXiv:2507.01467` |  | 把类别 token 与表征"纠缠"进去噪过程，训练更快 |
| 2025-06 | [Dispersive Loss](https://arxiv.org/abs/2506.09027) `arXiv:2506.09027` | MIT | 扩散模型内部的"分散"正则，类似无正样本对的对比学习，无需外部编码器 |
| 2025-04 | [ReDi](https://arxiv.org/abs/2504.16064) `arXiv:2504.16064` | Archimedes | 联合生成图像潜变量与语义特征 |
| 2025-04 | [REPA-E](https://arxiv.org/abs/2504.10483) `arXiv:2504.10483` | ANU/NYU | 借助 REPA 损失端到端联合训练 VAE 与扩散模型，FID 1.12（有 CFG） |
| 2025-04 | [PixelFlow](https://arxiv.org/abs/2504.07963) `arXiv:2504.07963` | HKU/Adobe | 级联流模型直接在像素空间生成 |
| 2025-03 | [FlowMo](https://arxiv.org/abs/2503.11056) `arXiv:2503.11056` | Stanford | 模式寻找的扩散自编码器做 tokenizer |
| 2025-02 | [EQ-VAE](https://arxiv.org/abs/2502.09509) `arXiv:2502.09509` |  | 对潜空间加等变正则，让潜空间更"好扩散" |
| 2025-02 | [MAETok](https://arxiv.org/abs/2502.03444) `arXiv:2502.03444` | CMU | 掩码自编码器做 tokenizer，潜空间更有判别性 |
| 2025-01 | ★ [VA-VAE / LightningDiT](https://arxiv.org/abs/2501.01423) `arXiv:2501.01423` | HUST | 把 VAE 潜空间与视觉基础模型对齐，化解"重建 vs 生成"矛盾，FID 1.35，收敛快 21 倍 |
| 2024-10 | [DC-AE](https://arxiv.org/abs/2410.10733) `arXiv:2410.10733` | MIT/NVIDIA | 32–128 倍空间压缩的深度压缩自编码器，高分辨率扩散大幅提速 |
| 2024-10 | ★ [REPA](https://arxiv.org/abs/2410.06940) `arXiv:2410.06940` | KAIST/NYU | 把 DiT 中间特征对齐到 DINOv2 表征，训练加速 17.5 倍，FID 1.42 |
| 2021-12 | ★ [LDM / Stable Diffusion](https://arxiv.org/abs/2112.10752) `arXiv:2112.10752` | LMU/Runway | 在预训练自编码器的潜空间做扩散 + 交叉注意力条件，Stable Diffusion 的基础 |

### 对齐/RL/测试时（46）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [RPO](https://arxiv.org/abs/2610.22282) `arXiv:2610.22282` | NUS | 在路由（而非去噪轨迹）上引入随机性的在线扩散 RL |
| 2026-10 | 🆕 [DiffusionART](https://arxiv.org/abs/2610.12572) `arXiv:2610.12572` | Kuaishou | 兼顾反向过程方法的信用分配与前向过程方法的效率，稳定可扩展的扩散 RL |
| 2026-10 | 🆕 [偏好 + 量规奖励](https://arxiv.org/abs/2610.02967) `arXiv:2610.02967` | LMArena/UC Berkeley | 组合偏好奖励与量规（rubric）奖励做前沿文生图后训练 |
| 2026-09 | 🆕 [Diffusion Reward Models](https://arxiv.org/abs/2609.33803) `arXiv:2609.33803` | Tsinghua | 用扩散模型表示多峰的偏好分布，作为奖励模型 |
| 2026-08 | 🆕 [DiffusionOPSD](https://arxiv.org/abs/2608.24646) `arXiv:2608.24646` | ByteDance Seed | 把图像级奖励转化为中间去噪步的 on-policy 自蒸馏目标 |
| 2026-07 | 🆕 [MeanFlowNFT](https://arxiv.org/abs/2607.15273) `arXiv:2607.15273` | Tencent Hunyuan | 把前向过程 RL 用到平均速度（MeanFlow）生成器 |
| 2026-07 | 🆕 [SpectraReward](https://arxiv.org/abs/2607.11886) `arXiv:2607.11886` | ByteDance Seed/HKU | 免训练地把预训练 MLLM 当作文生图 RL 的奖励模型 |
| 2026-07 | 🆕 [Flow-Map GRPO](https://arxiv.org/abs/2607.00535) `arXiv:2607.00535` |  | 为一致性模型、MeanFlow 等流映射生成器构造可用于 RL 的随机轨迹 |
| 2026-06 | 🆕 [Qwen-Image-2.0-RL](https://arxiv.org/abs/2606.27608) `arXiv:2606.27608` | Qwen | RLHF + on-policy 蒸馏的工业级后训练管线 |
| 2026-06 | 🆕 [Flow-DPPO](https://arxiv.org/abs/2606.11025) `arXiv:2606.11025` | Tencent Hunyuan | 用散度约束代替 PPO 比率裁剪的流匹配策略优化 |
| 2026-06 | 🆕 [Z-Reward](https://arxiv.org/abs/2606.09076) `arXiv:2606.09076` | Tongyi Lab | 把推理内化为评分分布的奖励模型，替代单一标量 |
| 2026-05 | 🆕 [Flash-GRPO](https://arxiv.org/abs/2605.15980) `arXiv:2605.15980` |  | 一步策略优化，降低 14B 视频模型 GRPO 成本 |
| 2026-05 | 🆕 [MARBLE](https://arxiv.org/abs/2605.06507) `arXiv:2605.06507` | Zhejiang Univ | 多维奖励的平衡优化 |
| 2026-04 | 🆕 [World-R1](https://arxiv.org/abs/2604.24764) `arXiv:2604.24764` | Microsoft | 用 RL 强化文生视频的 3D 约束 |
| 2026-04 | 🆕 [FP4 探索 / BF16 训练](https://arxiv.org/abs/2604.06916) `arXiv:2604.06916` | NVIDIA | 用低精度大规模 rollout 扩大组大小，高精度训练 |
| 2026-03 | 🆕 [VGGRPO](https://arxiv.org/abs/2603.26599) `arXiv:2603.26599` | Google | 4D 潜空间奖励提升视频的几何一致性 |
| 2026-03 | 🆕 [UniGRPO](https://arxiv.org/abs/2603.23500) `arXiv:2603.23500` | ByteDance Seed | 统一文本推理与图像生成的强化学习 |
| 2026-03 | 🆕 [流形感知探索](https://arxiv.org/abs/2603.21872) `arXiv:2603.21872` | Tencent | 视频 GRPO 中避免 ODE→SDE 转换注入过多噪声 |
| 2026-03 | 🆕 [TDM-R1](https://arxiv.org/abs/2603.07700) `arXiv:2603.07700` | HKUST | 对少步扩散模型做不可微奖励的强化学习 |
| 2026-02 | 🆕 [逐步奖励建模](https://arxiv.org/abs/2602.06422) `arXiv:2602.06422` | Zhejiang Univ | 区分每一步去噪的局部与长期影响，缓解流 GRPO 的奖励稀疏 |
| 2025-12 | 🆕 [GARDO](https://arxiv.org/abs/2512.24138) `arXiv:2512.24138` |  | 缓解扩散 RL 中的奖励投机（reward hacking） |
| 2025-11 | 🆕 [让视频守牛顿定律](https://arxiv.org/abs/2512.00425) `arXiv:2512.00425` | Stony Brook | 用可验证的物理奖励后训练视频模型 |
| 2025-11 | 🆕 [视频模型即潜空间奖励](https://arxiv.org/abs/2511.21541) `arXiv:2511.21541` | Tencent | 在潜空间用视频生成模型自身做奖励模型，实现视频 ReFL |
| 2025-10 | 🆕 [GRPO-Guard](https://arxiv.org/abs/2510.22319) `arXiv:2510.22319` |  | 调节裁剪，缓解流匹配 GRPO 的隐式过优化 |
| 2025-10 | 🆕 [UniWorld-V2 / Edit-R1](https://arxiv.org/abs/2510.16888) `arXiv:2510.16888` | PKU | 用 DiffusionNFT + MLLM 奖励做图像编辑的强化学习后训练 |
| 2025-10 | 🆕 [ImagerySearch](https://arxiv.org/abs/2510.14847) `arXiv:2510.14847` | Alibaba AMAP | 面向想象类提示的自适应测试时搜索 |
| 2025-09 | [AWM](https://arxiv.org/abs/2509.25050) `arXiv:2509.25050` |  | 优势加权匹配：让 RL 目标与预训练的流匹配损失形式一致 |
| 2025-09 | ★ [DiffusionNFT](https://arxiv.org/abs/2509.16117) `arXiv:2509.16117` | NVIDIA/Tsinghua | 在前向（加噪）过程上做在线 RL：对比正负样本得到改进方向，无需似然、与采样器解耦、免 CFG，效率最高可达 Flow-GRPO 的 25 倍 |
| 2025-09 | [SRPO](https://arxiv.org/abs/2509.06942) `arXiv:2509.06942` | Tencent Hunyuan | 直接对齐完整扩散轨迹，用文本条件化的奖励做细粒度偏好 |
| 2025-09 | [CPS](https://arxiv.org/abs/2509.05952) `arXiv:2509.05952` |  | 系数保持采样：修正 Flow-GRPO 的 SDE 噪声注入带来的伪影 |
| 2025-09 | [BranchGRPO](https://arxiv.org/abs/2509.06040) `arXiv:2509.06040` |  | 树状分支采样共享前缀，降低 rollout 成本 |
| 2025-08 | [Pref-GRPO](https://arxiv.org/abs/2508.20751) `arXiv:2508.20751` | Fudan/Tencent Hunyuan | 用成对偏好奖励代替打分奖励，缓解 GRPO 奖励投机 |
| 2025-08 | [TempFlow-GRPO](https://arxiv.org/abs/2508.04324) `arXiv:2508.04324` | Zhejiang Univ | 按时间步区分信用分配，分支采样 |
| 2025-08 | [HPSv3](https://arxiv.org/abs/2508.03789) `arXiv:2508.03789` |  | 覆盖更广图像分布的人类偏好评分 |
| 2025-07 | [MixGRPO](https://arxiv.org/abs/2507.21802) `arXiv:2507.21802` | Tencent Hunyuan | 只在滑动窗口内用 SDE 采样与优化，其余步用 ODE，大幅提速 |
| 2025-05 | [DanceGRPO](https://arxiv.org/abs/2505.07818) `arXiv:2505.07818` | ByteDance Seed/HKU | 统一图像、视频生成的 GRPO 框架，覆盖多种奖励模型 |
| 2025-05 | ★ [Flow-GRPO](https://arxiv.org/abs/2505.05470) `arXiv:2505.05470` | CUHK/Kuaishou | 把确定性 ODE 改成等边缘分布的 SDE 以获得随机性，再用 GRPO 在线 RL；GenEval 从 63% 提到 95% |
| 2025-01 | ★ [推理时扩展](https://arxiv.org/abs/2501.09732) `arXiv:2501.09732` | Google/NYU | 在噪声空间做搜索（验证器 + 搜索算法），推理算力扩展超出"加步数" |
| 2023-11 | ★ [Diffusion-DPO](https://arxiv.org/abs/2311.12908) `arXiv:2311.12908` | Salesforce/Stanford | 把 DPO 推广到扩散模型，用离线偏好对直接优化 |
| 2023-10 | [AlignProp](https://arxiv.org/abs/2310.03739) `arXiv:2310.03739` | CMU | 用 LoRA + 梯度检查点做端到端奖励反传 |
| 2023-09 | [DRaFT](https://arxiv.org/abs/2309.17400) `arXiv:2309.17400` | Google DeepMind | 沿采样链直接反传可微奖励 |
| 2023-06 | [HPSv2](https://arxiv.org/abs/2306.09341) `arXiv:2306.09341` | CUHK | 人类偏好分数 v2：大规模偏好数据与评分模型 |
| 2023-05 | [DPOK](https://arxiv.org/abs/2305.16381) `arXiv:2305.16381` | Google/KAIST | 带 KL 正则的在线 RL 微调文生图 |
| 2023-05 | ★ [DDPO](https://arxiv.org/abs/2305.13301) `arXiv:2305.13301` | UC Berkeley | 把去噪过程看作多步决策，用策略梯度优化任意奖励 |
| 2023-05 | [Pick-a-Pic / PickScore](https://arxiv.org/abs/2305.01569) `arXiv:2305.01569` | Tel Aviv/Stability AI | 开放的用户偏好数据集与评分模型 |
| 2023-04 | [ImageReward](https://arxiv.org/abs/2304.05977) `arXiv:2304.05977` | Tsinghua/Zhipu | 人类偏好奖励模型 + ReFL 直接奖励反传 |

### 理论/记忆与泛化（21）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🆕 [懒惰高维区间的泛化与记忆](https://arxiv.org/abs/2608.23938) `arXiv:2608.23938` |  | 在随机特征/懒惰训练区间刻画扩散模型的泛化、记忆与过拟合 |
| 2026-06 | 🆕 [扩散流匹配收敛界](https://arxiv.org/abs/2606.16610) `arXiv:2606.16610` |  | 布朗运动型扩散流匹配的维度改进 KL 界与 Wasserstein 保证 |
| 2026-06 | 🆕 [局部覆盖决定记忆](https://arxiv.org/abs/2606.14390) `arXiv:2606.14390` |  | 哪些训练样本会被记住，取决于其邻域内数据的局部覆盖程度 |
| 2026-06 | 🆕 [神经网络流匹配理论](https://arxiv.org/abs/2606.10089) `arXiv:2606.10089` |  | 两层 ReLU 网络流匹配的优化收敛与泛化界 |
| 2026-05 | 🆕 [去噪器架构与创造力](https://arxiv.org/abs/2605.16415) `arXiv:2605.16415` | Hebrew Univ | 创造力取决于去噪器架构偏离贝叶斯最优去噪器的方式 |
| 2026-03 | 🆕 [流形泛化先于记忆](https://arxiv.org/abs/2603.23792) `arXiv:2603.23792` | ETH | 在流形假设下证明：分数只需粗略准确，模型就会先在流形上泛化，后才记忆 |
| 2026-02 | 🆕 [流匹配自适应流形](https://arxiv.org/abs/2602.22486) `arXiv:2602.22486` |  | 流匹配的收敛速率由数据的内在维度而非环境维度决定 |
| 2025-11 | 🆕 [记忆与泛化的可证分离](https://arxiv.org/abs/2511.03202) `arXiv:2511.03202` | Georgia Tech | 给出记忆与泛化之间的可证明分离 |
| 2025-10 | 🆕 [选择性欠拟合](https://arxiv.org/abs/2510.01378) `arXiv:2510.01378` | MIT/Harvard | 扩散模型只在采样会经过的区域拟合分数，在其他区域欠拟合，由此解释泛化 |
| 2025-09 | [局部性来自数据](https://arxiv.org/abs/2509.09672) `arXiv:2509.09672` | MIT | 扩散模型的局部性主要源于图像数据统计，而非卷积结构 |
| 2025-06 | [闭式流匹配](https://arxiv.org/abs/2506.03719) `arXiv:2506.03719` | Inria | 流匹配的泛化并不来自目标的随机性，闭式解同样泛化 |
| 2025-05 | ★ [为什么不记忆](https://arxiv.org/abs/2505.17638) `arXiv:2505.17638` | ENS/Bocconi | 训练动力学中存在两个时间尺度：泛化先出现、记忆后出现，且记忆时间随数据量线性增长（NeurIPS 2025 最佳论文） |
| 2024-12 | [创造力的解析理论](https://arxiv.org/abs/2412.20292) `arXiv:2412.20292` | Stanford | 局部性 + 等变性的最优去噪器可解析地"拼贴"出新图像，解释卷积扩散的创造力 |
| 2023-10 | [几何自适应调和表示](https://arxiv.org/abs/2310.02557) `arXiv:2310.02557` | NYU | 两个模型在不重叠数据上训练却生成几乎相同的图：泛化来自网络的归纳偏置 |
| 2023-08 | [近 d 线性界](https://arxiv.org/abs/2308.03686) `arXiv:2308.03686` | Oxford | 用随机局部化给出关于维度近线性的收敛界 |
| 2023-05 | [概率流 ODE 很快](https://arxiv.org/abs/2305.11798) `arXiv:2305.11798` |  | 概率流 ODE 采样的多项式时间保证 |
| 2023-03 | [极小极大最优](https://arxiv.org/abs/2303.01861) `arXiv:2303.01861` | Tokyo | 扩散模型在 Besov 空间上达到分布估计的极小极大最优速率 |
| 2023-01 | [训练数据提取](https://arxiv.org/abs/2301.13188) `arXiv:2301.13188` | Google | 从 Stable Diffusion 中提取出上百张训练图像，揭示记忆风险 |
| 2022-12 | [复制还是创作](https://arxiv.org/abs/2212.03860) `arXiv:2212.03860` | Maryland | 系统测量扩散模型对训练数据的复制现象 |
| 2022-09 | [采样=学分数](https://arxiv.org/abs/2209.11215) `arXiv:2209.11215` |  | 只要分数估计足够准，扩散采样就能以多项式复杂度逼近任意数据分布 |
| 2022-06 | [多项式收敛](https://arxiv.org/abs/2206.06227) `arXiv:2206.06227` |  | 分数生成模型的多项式复杂度收敛保证 |

### 文生图模型（31）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [MOSS-Image](https://arxiv.org/abs/2610.23374) `arXiv:2610.23374` | OpenMOSS | 开源文生图，主攻密集文字与复杂版式 |
| 2026-09 | 🆕 [LLaDA-Image](https://arxiv.org/abs/2609.03796) `arXiv:2609.03796` | inclusionAI | 6B DiT + 冻结的 LLaDA2.0 扩散语言模型做理解模块，训练配方完全开源 |
| 2026-07 | 🆕 [Mage-Flow](https://arxiv.org/abs/2607.19064) `arXiv:2607.19064` | Microsoft | 约 4B 的紧凑原生分辨率生成 + 编辑栈（自研 VAE + DiT） |
| 2026-07 | 🆕 [Boogu-Image-0.1](https://arxiv.org/abs/2607.13125) `arXiv:2607.13125` | Boogu | 开源统一理解与生成模型族（Base/Turbo/Edit） |
| 2026-05 | 🆕 [Qwen-Image-2.0](https://arxiv.org/abs/2605.10730) `arXiv:2605.10730` | Qwen | 生成与编辑一体的全能图像基础模型，超长文字渲染、多图参考 |
| 2025-12 | 🆕 [Qwen-Image-Layered](https://arxiv.org/abs/2512.15603) `arXiv:2512.15603` | Qwen | 把图像分解为多个可独立编辑的 RGBA 图层 |
| 2025-12 | 🆕 [LongCat-Image](https://arxiv.org/abs/2512.07584) `arXiv:2512.07584` | Meituan | 美团开源文生图与编辑模型技术报告 |
| 2025-11 | ★ 🆕 [Z-Image](https://arxiv.org/abs/2511.22699) `arXiv:2511.22699` | Tongyi Lab | 6B 单流 DiT（S3-DiT），完整训练约 31.4 万 H800 GPU 小时；少步蒸馏 + 奖励后训练得到 Z-Image-Turbo |
| 2025-11 | 🆕 [UltraFlux](https://arxiv.org/abs/2511.18050) `arXiv:2511.18050` |  | 原生 4K、多长宽比文生图：位置编码、VAE 与训练数据联合设计 |
| 2025-09 | [HunyuanImage 3.0](https://arxiv.org/abs/2509.23951) `arXiv:2509.23951` | Tencent | 在自回归统一框架下做理解与生成的 MoE 模型（总参 80B+、激活 13B），开源图像生成模块 |
| 2025-09 | ★ [Seedream 4.0](https://arxiv.org/abs/2509.20427) `arXiv:2509.20427` | ByteDance Seed | 文生图、编辑、多图合成统一在一个框架内；高效 DiT + 高压缩 VAE，原生 1K–4K 输出 |
| 2025-08 | ★ [Qwen-Image](https://arxiv.org/abs/2508.02324) `arXiv:2508.02324` | Qwen | 20B MMDiT，复杂文字（尤其中文）渲染突出，开源；同系列 Qwen-Image-Edit 做指令编辑 |
| 2025-05 | [HiDream-I1](https://arxiv.org/abs/2505.22705) `arXiv:2505.22705` | HiDream.ai | 17B 稀疏 MoE DiT 开源文生图 |
| 2025-04 | [Seedream 3.0](https://arxiv.org/abs/2504.11346) `arXiv:2504.11346` | ByteDance Seed | 原生 2K 输出、更强排版，多种加速 |
| 2025-03 | [Lumina-Image 2.0](https://arxiv.org/abs/2503.21758) `arXiv:2503.21758` | Shanghai AI Lab | 统一的单流 DiT 架构与描述体系 |
| 2025-03 | [Seedream 2.0](https://arxiv.org/abs/2503.07703) `arXiv:2503.07703` | ByteDance Seed | 原生中英双语文生图，强调文字渲染与文化理解 |
| 2025-01 | [Sana 1.5](https://arxiv.org/abs/2501.18427) `arXiv:2501.18427` | NVIDIA | 训练时深度扩展 + 推理时重复采样筛选 |
| 2024-10 | [Sana](https://arxiv.org/abs/2410.10629) `arXiv:2410.10629` | NVIDIA/MIT | 线性注意力 DiT + 32 倍压缩 AE + 小型 LLM 文本编码器，最高 4K，可部署在笔记本 GPU 上 |
| 2024-08 | [Imagen 3](https://arxiv.org/abs/2408.07009) `arXiv:2408.07009` | Google | Google 第三代文生图模型技术报告 |
| 2024-06 | [Lumina-Next](https://arxiv.org/abs/2406.18583) `arXiv:2406.18583` | Shanghai AI Lab | Next-DiT 架构改进与分辨率外推 |
| 2024-05 | [Hunyuan-DiT](https://arxiv.org/abs/2405.08748) `arXiv:2405.08748` | Tencent | 中英双语细粒度理解的 DiT 文生图 |
| 2024-05 | [Lumina-T2X](https://arxiv.org/abs/2405.05945) `arXiv:2405.05945` | Shanghai AI Lab | 基于流的大型扩散 Transformer，统一图像、视频、3D、音频 |
| 2024-03 | [CogView3](https://arxiv.org/abs/2403.05121) `arXiv:2403.05121` | Zhipu/Tsinghua | 接力扩散：先低分辨率再接力生成高分辨率 |
| 2024-03 | [PixArt-Σ](https://arxiv.org/abs/2403.04692) `arXiv:2403.04692` | Huawei Noah | 弱到强训练，直接生成 4K 图像 |
| 2024-03 | ★ [SD3 / MMDiT](https://arxiv.org/abs/2403.03206) `arXiv:2403.03206` | Stability AI | Rectified Flow + 双流 MMDiT + logit-normal 时间采样，把流匹配推向大规模文生图 |
| 2024-02 | [Playground v2.5](https://arxiv.org/abs/2402.17245) `arXiv:2402.17245` | Playground | 提升美学质量的三条经验：色彩对比、多长宽比、人类偏好对齐 |
| 2023-09 | [PixArt-α](https://arxiv.org/abs/2310.00426) `arXiv:2310.00426` | Huawei Noah | 以约 10% 的训练成本训练 DiT 文生图 |
| 2023-09 | [Emu](https://arxiv.org/abs/2309.15807) `arXiv:2309.15807` | Meta | 用极少量高美学图像做"质量微调" |
| 2023-07 | [SDXL](https://arxiv.org/abs/2307.01952) `arXiv:2307.01952` | Stability AI | 更大的 U-Net、双文本编码器、多尺度训练与精修模型 |
| 2022-05 | ★ [Imagen](https://arxiv.org/abs/2205.11487) `arXiv:2205.11487` | Google | 冻结的大语言模型（T5-XXL）做文本编码器，扩大文本端比扩大图像端更有效 |
| 2022-04 | [DALL·E 2](https://arxiv.org/abs/2204.06125) `arXiv:2204.06125` | OpenAI | unCLIP：先由文本生成 CLIP 图像嵌入，再用扩散解码 |

### 编辑/个性化/控制（41）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-06 | 🆕 [Qwen-Image-Agent](https://arxiv.org/abs/2606.26907) `arXiv:2606.26907` | Qwen | 用 agent 补足真实需求中未说明、隐含或需最新知识的"上下文缺口" |
| 2026-03 | 🆕 [Gen-Searcher](https://arxiv.org/abs/2603.28767) `arXiv:2603.28767` |  | 用强化学习训练图像生成的 agentic 搜索 |
| 2026-01 | 🆕 [Think-Then-Generate](https://arxiv.org/abs/2601.10332) `arXiv:2601.10332` | SJTU | 让 LLM 文本编码器先推理再生成 |
| 2026-01 | 🆕 [VIBE](https://arxiv.org/abs/2601.02242) `arXiv:2601.02242` |  | 开源视觉指令编辑器，复盘过去一年指令编辑的进展 |
| 2025-12 | 🆕 [OmniPSD](https://arxiv.org/abs/2512.09247) `arXiv:2512.09247` |  | 生成带透明通道的分层 PSD 文件 |
| 2025-12 | 🆕 [EditThinker](https://arxiv.org/abs/2512.05965) `arXiv:2512.05965` |  | 给任意编辑器加上迭代推理 |
| 2025-11 | 🆕 [ReasonEdit](https://arxiv.org/abs/2511.22625) `arXiv:2511.22625` | StepFun | 在编辑模型中引入推理（思考 + 反思） |
| 2025-10 | 🆕 [Pico-Banana-400K](https://arxiv.org/abs/2510.19808) `arXiv:2510.19808` | Apple | 用 Nano Banana 生成的 40 万条文本引导编辑数据集 |
| 2025-10 | 🆕 [WithAnyone](https://arxiv.org/abs/2510.14975) `arXiv:2510.14975` | StepFun/Fudan | 多身份一致生成，缓解"复制粘贴"人脸 |
| 2025-08 | [USO](https://arxiv.org/abs/2508.18966) `arXiv:2508.18966` | ByteDance | 风格与主体统一的定制生成，含奖励学习 |
| 2025-06 | [OmniGen2](https://arxiv.org/abs/2506.18871) `arXiv:2506.18871` | BAAI | 文本与图像解码分离的统一生成模型 |
| 2025-06 | ★ [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742) `arXiv:2506.15742` | Black Forest Labs | 流匹配上下文生成与编辑，支持多轮一致编辑 |
| 2025-05 | [ImgEdit](https://arxiv.org/abs/2505.20275) `arXiv:2505.20275` | PKU | 统一的图像编辑数据集与基准 |
| 2025-04 | [ICEdit](https://arxiv.org/abs/2504.20690) `arXiv:2504.20690` | Zhejiang Univ | 大规模 DiT 的上下文指令编辑，少量数据即可 |
| 2025-04 | [Step1X-Edit](https://arxiv.org/abs/2504.17761) `arXiv:2504.17761` | StepFun | MLLM + DiT 的通用图像编辑框架，开源 |
| 2025-04 | [DreamO](https://arxiv.org/abs/2504.16915) `arXiv:2504.16915` | ByteDance | 统一的图像定制框架（身份、物体、风格、试穿） |
| 2025-04 | [UNO](https://arxiv.org/abs/2504.02160) `arXiv:2504.02160` | ByteDance | "少到多"泛化的主体驱动生成 |
| 2025-03 | [InfiniteYou](https://arxiv.org/abs/2503.16418) `arXiv:2503.16418` | ByteDance | 基于 FLUX 的身份保持人像重塑 |
| 2024-12 | [FlowEdit](https://arxiv.org/abs/2412.08629) `arXiv:2412.08629` | Technion | 不经反演、直接在流模型上构造从源到目标的路径 |
| 2024-11 | [OminiControl](https://arxiv.org/abs/2411.15098) `arXiv:2411.15098` | NUS | 在 DiT 里把条件图像当作 token，仅增 0.1% 参数实现通用控制 |
| 2024-10 | [In-Context LoRA](https://arxiv.org/abs/2410.23775) `arXiv:2410.23775` | Alibaba | DiT 天然具备上下文生成能力，拼图 + 小 LoRA 即可激活 |
| 2024-10 | [RF-Inversion](https://arxiv.org/abs/2410.10792) `arXiv:2410.10792` | Google/UT Austin | 基于 Rectified Flow 随机微分方程的反演与编辑 |
| 2024-09 | [OmniGen](https://arxiv.org/abs/2409.11340) `arXiv:2409.11340` | BAAI | 一个模型统一文生图、编辑、主体驱动等多种任务 |
| 2024-01 | [InstantID](https://arxiv.org/abs/2401.07519) `arXiv:2401.07519` | InstantX | 单张人脸图零样本保持身份 |
| 2023-12 | [PhotoMaker](https://arxiv.org/abs/2312.04461) `arXiv:2312.04461` | Tencent ARC | 堆叠 ID 嵌入的人像定制 |
| 2023-11 | [Emu Edit](https://arxiv.org/abs/2311.10089) `arXiv:2311.10089` | Meta | 把识别与生成任务一起训练的指令编辑模型 |
| 2023-08 | [IP-Adapter](https://arxiv.org/abs/2308.06721) `arXiv:2308.06721` | Tencent | 解耦交叉注意力，把图像作为提示 |
| 2023-06 | [MagicBrush](https://arxiv.org/abs/2306.10012) `arXiv:2306.10012` | OSU | 人工标注的指令编辑数据集 |
| 2023-02 | [T2I-Adapter](https://arxiv.org/abs/2302.08453) `arXiv:2302.08453` | Tencent ARC | 轻量适配器注入结构条件 |
| 2023-02 | ★ [ControlNet](https://arxiv.org/abs/2302.05543) `arXiv:2302.05543` | Stanford | 复制编码器 + 零初始化卷积，加入边缘、深度、姿态等空间条件 |
| 2023-02 | [pix2pix-zero](https://arxiv.org/abs/2302.03027) `arXiv:2302.03027` | CMU/Adobe | 无需训练的零样本图像翻译 |
| 2022-12 | [Custom Diffusion](https://arxiv.org/abs/2212.04488) `arXiv:2212.04488` | CMU/Adobe | 只微调交叉注意力，支持多概念组合 |
| 2022-11 | [PnP Features](https://arxiv.org/abs/2211.12572) `arXiv:2211.12572` | Weizmann | 注入原图的空间特征与自注意力做图到图翻译 |
| 2022-11 | [Null-text Inversion](https://arxiv.org/abs/2211.09794) `arXiv:2211.09794` | Google | 优化空文本嵌入实现真实图像的精确反演与编辑 |
| 2022-11 | ★ [InstructPix2Pix](https://arxiv.org/abs/2211.09800) `arXiv:2211.09800` | UC Berkeley | 用 GPT-3 + Prompt-to-Prompt 合成数据，训练"按指令编辑"模型 |
| 2022-10 | [Imagic](https://arxiv.org/abs/2210.09276) `arXiv:2210.09276` | Google | 对单张真实图像做复杂的文本编辑 |
| 2022-08 | ★ [DreamBooth](https://arxiv.org/abs/2208.12242) `arXiv:2208.12242` | Google | 用 3–5 张图微调模型，学会一个特定主体 |
| 2022-08 | [Textual Inversion](https://arxiv.org/abs/2208.01618) `arXiv:2208.01618` | Tel Aviv/NVIDIA | 只学一个新词嵌入来表示新概念 |
| 2022-08 | [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626) `arXiv:2208.01626` | Google | 通过控制交叉注意力图实现提示词级编辑 |
| 2021-08 | [SDEdit](https://arxiv.org/abs/2108.01073) `arXiv:2108.01073` | Stanford | 先加噪再去噪，把草图/粗图"拉回"真实图像流形 |
| 2021-06 | [LoRA](https://arxiv.org/abs/2106.09685) `arXiv:2106.09685` | Microsoft | 低秩适配；在扩散社区成为风格/角色定制的事实标准 |

### 视频生成（56）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Kandinsky 6.0 Video](https://arxiv.org/abs/2610.05608) `arXiv:2610.05608` | Kandinsky Lab | 3B/29B 的同步音视频生成模型族 |
| 2026-08 | 🆕 [Wan-Animate-2](https://arxiv.org/abs/2608.06009) `arXiv:2608.06009` | Alibaba | 拓展角色动画的应用边界 |
| 2026-07 | 🆕 [SANA-Video 2.0](https://arxiv.org/abs/2607.21553) `arXiv:2607.21553` | NVIDIA | 5B/14B 混合线性注意力视频 DiT，单卡 720p，质量追平全注意力模型 |
| 2026-07 | 🆕 [视频生成即通用视觉学习](https://arxiv.org/abs/2607.09024) `arXiv:2607.09024` | Google DeepMind | 论证大规模视频生成预训练是通用视觉模型的"下一个 token 预测" |
| 2026-04 | ★ 🆕 [Seedance 2.0](https://arxiv.org/abs/2604.14148) `arXiv:2604.14148` | ByteDance Seed | 2026 年 2 月发布的原生多模态音视频联合生成模型，支持文本、图像、音频、视频四种输入与参考编辑 |
| 2026-03 | 🆕 [Kiwi-Edit](https://arxiv.org/abs/2603.02175) `arXiv:2603.02175` | NUS | 指令 + 参考图双重引导的视频编辑 |
| 2025-12 | ★ 🆕 [Kling-Omni](https://arxiv.org/abs/2512.16776) `arXiv:2512.16776` | Kuaishou | 把生成、编辑、推理统一到一个端到端视频系统，多模态视觉语言输入 |
| 2025-12 | 🆕 [KlingAvatar 2.0](https://arxiv.org/abs/2512.13313) `arXiv:2512.13313` | Kuaishou | 长时长高分辨率数字人视频 |
| 2025-12 | 🆕 [Wan-Move](https://arxiv.org/abs/2512.08765) `arXiv:2512.08765` | Alibaba | 通过潜轨迹引导实现精细运动控制 |
| 2025-12 | 🆕 [VideoCoF](https://arxiv.org/abs/2512.07469) `arXiv:2512.07469` |  | 带时间推理的统一视频编辑，无需遮罩 |
| 2025-12 | 🆕 [OneStory](https://arxiv.org/abs/2512.07802) `arXiv:2512.07802` | Meta | 自适应记忆的连贯多镜头生成 |
| 2025-12 | 🆕 [MultiShotMaster](https://arxiv.org/abs/2512.03041) `arXiv:2512.03041` | Kuaishou | 可控多镜头视频生成 |
| 2025-11 | 🆕 [HunyuanVideo 1.5](https://arxiv.org/abs/2511.18870) `arXiv:2511.18870` | Tencent | 8.3B 轻量开源视频模型，选择性滑窗注意力（SSTA），消费级 GPU 可推理 |
| 2025-11 | 🆕 [Kandinsky 5.0](https://arxiv.org/abs/2511.14993) `arXiv:2511.14993` | Kandinsky Lab | 图像与 10 秒视频生成的开源模型族 |
| 2025-11 | 🆕 [Thinking with Video](https://arxiv.org/abs/2511.04570) `arXiv:2511.04570` | Fudan | 把视频生成当作多模态推理范式，并给出评测 |
| 2025-10 | 🆕 [Cosmos-Predict2.5](https://arxiv.org/abs/2511.00062) `arXiv:2511.00062` | NVIDIA | 流模型架构统一 Text/Image/Video2World，面向物理 AI 的世界基础模型 |
| 2025-10 | 🆕 [LongCat-Video](https://arxiv.org/abs/2510.22200) `arXiv:2510.22200` | Meituan | 13.6B 视频基础模型，原生支持长视频续写 |
| 2025-10 | 🆕 [Video-As-Prompt](https://arxiv.org/abs/2510.20888) `arXiv:2510.20888` | ByteDance | 以参考视频作为语义控制的统一框架 |
| 2025-10 | 🆕 [HoloCine](https://arxiv.org/abs/2510.20822) `arXiv:2510.20822` | Ant Group | 整场景一次生成的多镜头电影叙事 |
| 2025-10 | 🆕 [Ditto](https://arxiv.org/abs/2510.15742) `arXiv:2510.15742` |  | 合成大规模高质量指令视频编辑数据 |
| 2025-10 | 🆕 [UniVideo](https://arxiv.org/abs/2510.08377) `arXiv:2510.08377` | Kuaishou | 把统一理解、生成与编辑扩展到视频 |
| 2025-10 | 🆕 [InstructX](https://arxiv.org/abs/2510.08485) `arXiv:2510.08485` |  | MLLM 引导的图像/视频统一编辑 |
| 2025-09 | [SANA-Video](https://arxiv.org/abs/2509.24695) `arXiv:2509.24695` | NVIDIA | 块线性注意力 DiT，低成本生成分钟级视频 |
| 2025-09 | ★ [视频模型零样本推理](https://arxiv.org/abs/2509.20328) `arXiv:2509.20328` | Google DeepMind | Veo 3 无需专门训练即可做分割、边缘检测、物理理解、迷宫求解等，视频模型或将成为视觉通用模型 |
| 2025-09 | [Wan-Animate](https://arxiv.org/abs/2509.14055) `arXiv:2509.14055` | Alibaba | 统一的角色动画与角色替换 |
| 2025-09 | [HuMo](https://arxiv.org/abs/2509.08519) `arXiv:2509.08519` | Tsinghua/ByteDance | 文本、图像、音频协同条件的人物视频生成 |
| 2025-08 | [OmniHuman-1.5](https://arxiv.org/abs/2508.19209) `arXiv:2508.19209` | ByteDance | 引入 MLLM"认知模拟"，让数字人动作更符合语义 |
| 2025-08 | [Waver](https://arxiv.org/abs/2508.15761) `arXiv:2508.15761` | ByteDance | 统一文生图、文生视频、图生视频的单一模型 |
| 2025-08 | [InfiniteTalk](https://arxiv.org/abs/2508.14033) `arXiv:2508.14033` | Meituan | 面向视频配音的无限长音频驱动生成 |
| 2025-06 | [Seedance 1.0](https://arxiv.org/abs/2506.09113) `arXiv:2506.09113` | ByteDance Seed | 多镜头文生/图生视频，RLHF 与蒸馏加速 |
| 2025-05 | [MultiTalk](https://arxiv.org/abs/2505.22647) `arXiv:2505.22647` | Meituan | 音频驱动的多人对话视频 |
| 2025-04 | [SkyReels-V2](https://arxiv.org/abs/2504.13074) `arXiv:2504.13074` | Skywork | 扩散强制框架下的无限长电影生成 |
| 2025-03 | ★ [Wan](https://arxiv.org/abs/2503.20314) `arXiv:2503.20314` | Alibaba | Wan 2.1 系列开源视频模型（1.3B/14B），后续 Wan 2.2 引入 MoE 去噪专家 |
| 2025-03 | [ReCamMaster](https://arxiv.org/abs/2503.11647) `arXiv:2503.11647` | Zhejiang/Kuaishou | 给定一段视频，按新相机轨迹重新渲染 |
| 2025-03 | [Open-Sora 2.0](https://arxiv.org/abs/2503.09642) `arXiv:2503.09642` | HPC-AI Tech | 用约 20 万美元训练商业级视频模型 |
| 2025-03 | [VACE](https://arxiv.org/abs/2503.07598) `arXiv:2503.07598` | Alibaba | 一体化视频创作与编辑：参考生成、视频编辑、遮罩编辑统一 |
| 2025-02 | [Phantom](https://arxiv.org/abs/2502.11079) `arXiv:2502.11079` | ByteDance | 跨模态对齐的主体一致视频生成 |
| 2025-02 | [Step-Video-T2V](https://arxiv.org/abs/2502.10248) `arXiv:2502.10248` | StepFun | 30B 文生视频技术报告，讨论视频基础模型的挑战 |
| 2025-02 | [VideoJAM](https://arxiv.org/abs/2502.02492) `arXiv:2502.02492` | Meta | 联合学习外观与运动表征，显著改善运动连贯性 |
| 2025-02 | [OmniHuman-1](https://arxiv.org/abs/2502.01061) `arXiv:2502.01061` | ByteDance | 多条件混合训练的一阶段人物动画（音频驱动全身） |
| 2024-12 | [LTX-Video](https://arxiv.org/abs/2501.00103) `arXiv:2501.00103` | Lightricks | 高压缩视频 VAE，生成速度超过实时 |
| 2024-12 | [Open-Sora](https://arxiv.org/abs/2412.20404) `arXiv:2412.20404` | HPC-AI Tech | 完全开源的类 Sora 训练方案 |
| 2024-12 | [HunyuanVideo](https://arxiv.org/abs/2412.03603) `arXiv:2412.03603` | Tencent | 13B 开源视频模型，系统介绍数据、架构与训练基础设施 |
| 2024-10 | ★ [Movie Gen](https://arxiv.org/abs/2410.13720) `arXiv:2410.13720` | Meta | 30B 视频 + 13B 音频的媒体基础模型，流匹配训练 |
| 2024-10 | [Pyramidal Flow](https://arxiv.org/abs/2410.05954) `arXiv:2410.05954` | PKU/Kuaishou | 金字塔流匹配：在多分辨率之间分段生成，训练更省 |
| 2024-08 | [CogVideoX](https://arxiv.org/abs/2408.06072) `arXiv:2408.06072` | Zhipu/Tsinghua | 3D 因果 VAE + 专家 Transformer 的开源文生视频 |
| 2024-04 | [CameraCtrl](https://arxiv.org/abs/2404.02101) `arXiv:2404.02101` | CUHK | 为文生视频加入精确相机轨迹控制 |
| 2024-01 | [Lumiere](https://arxiv.org/abs/2401.12945) `arXiv:2401.12945` | Google | 时空 U-Net 一次生成整段视频，而非关键帧 + 插帧 |
| 2023-11 | [Animate Anyone](https://arxiv.org/abs/2311.17117) `arXiv:2311.17117` | Alibaba | 参考网络保持外观的角色动画 |
| 2023-11 | [SVD](https://arxiv.org/abs/2311.15127) `arXiv:2311.15127` | Stability AI | Stable Video Diffusion：强调数据筛选的三阶段训练 |
| 2023-07 | [AnimateDiff](https://arxiv.org/abs/2307.04725) `arXiv:2307.04725` | CUHK/Shanghai AI Lab | 即插即用的运动模块，让个性化文生图模型直接动起来 |
| 2023-04 | [Video LDM](https://arxiv.org/abs/2304.08818) `arXiv:2304.08818` | NVIDIA | 在图像 LDM 中插入时间层，潜空间视频生成 |
| 2022-12 | [Tune-A-Video](https://arxiv.org/abs/2212.11565) `arXiv:2212.11565` | NUS | 单视频微调图像模型实现文本视频编辑 |
| 2022-10 | [Imagen Video](https://arxiv.org/abs/2210.02303) `arXiv:2210.02303` | Google | 级联时空超分的高清文生视频 |
| 2022-09 | [Make-A-Video](https://arxiv.org/abs/2209.14792) `arXiv:2209.14792` | Meta | 从文生图模型扩展，不需要文本-视频配对数据 |
| 2022-04 | [VDM](https://arxiv.org/abs/2204.03458) `arXiv:2204.03458` | Google | 首个视频扩散模型：3D U-Net + 图像视频联合训练 |

### 自回归/实时视频（27）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Speculative Forcing](https://arxiv.org/abs/2610.22347) `arXiv:2610.22347` | NAVER | 按 token 需要决定是否跑最后一步去噪，加速自回归视频 |
| 2026-09 | 🆕 [自回归视频的记忆](https://arxiv.org/abs/2609.28466) `arXiv:2609.28466` |  | 系统研究自回归视频生成中的记忆机制 |
| 2026-09 | 🆕 [Vidu S2](https://arxiv.org/abs/2609.11638) `arXiv:2609.11638` | Shengshu/Tsinghua | 实时交互数字人 + 实时视频编辑，并探索实时空间视频 |
| 2026-08 | 🆕 [JoyAI-Video-Edit](https://arxiv.org/abs/2608.03974) `arXiv:2608.03974` | JD | 16B 自回归扩散的实时开放式视频编辑 |
| 2026-07 | 🆕 [Vidu S1](https://arxiv.org/abs/2607.03118) `arXiv:2607.03118` | Shengshu/Tsinghua | 语音可随时控制的实时交互视频（数字角色），无限长度 |
| 2026-06 | 🆕 [LiveEdit](https://arxiv.org/abs/2606.26740) `arXiv:2606.26740` | Tsinghua | 实时流式视频编辑，保持背景稳定 |
| 2026-06 | 🆕 [Causal-rCM](https://arxiv.org/abs/2606.25473) `arXiv:2606.25473` | NVIDIA | 把 rCM 扩展到因果视频扩散，统一教师强制与自强制的开源配方 |
| 2026-05 | 🆕 [LongLive-2.0](https://arxiv.org/abs/2605.18739) `arXiv:2605.18739` | NVIDIA | NVFP4 全流程并行基础设施，加速长视频训练与推理 |
| 2026-05 | 🆕 [Causal Forcing++](https://arxiv.org/abs/2605.15141) `arXiv:2605.15141` | Tsinghua | 可扩展的少步自回归扩散蒸馏，用于实时交互视频 |
| 2026-03 | 🆕 [PackForcing](https://arxiv.org/abs/2603.25730) `arXiv:2603.25730` | Alaya Lab | 只用短视频训练即可长视频采样与长上下文推理 |
| 2026-03 | 🆕 [Helios](https://arxiv.org/abs/2603.04379) `arXiv:2603.04379` | PKU/ByteDance | 14B 视频模型单卡 H100 19.5 FPS，分钟级生成，不依赖自强制等防漂移技巧 |
| 2026-02 | 🆕 [Mode/Mean Seeking](https://arxiv.org/abs/2602.24289) `arXiv:2602.24289` | NVIDIA | 短视频上做模式寻找、长视频上做均值寻找，兼顾质量与长时连贯 |
| 2026-02 | 🆕 [Causal Forcing](https://arxiv.org/abs/2602.02214) `arXiv:2602.02214` | Tsinghua | 指出从双向到因果蒸馏的架构鸿沟，给出正确的自回归扩散蒸馏 |
| 2025-12 | 🆕 [LiveTalk](https://arxiv.org/abs/2512.23576) `arXiv:2512.23576` | SJTU | 改进 on-policy 蒸馏的实时多模态交互视频 |
| 2025-12 | 🆕 [Reward Forcing](https://arxiv.org/abs/2512.04678) `arXiv:2512.04678` | Ant Group | 带奖励的分布匹配蒸馏，提升流式视频的运动幅度 |
| 2025-12 | 🆕 [Live Avatar](https://arxiv.org/abs/2512.04677) `arXiv:2512.04677` | Alibaba | 算法-系统协同的实时无限长音频驱动数字人 |
| 2025-11 | 🆕 [MotionStream](https://arxiv.org/abs/2511.01266) `arXiv:2511.01266` | Adobe | 实时交互运动控制视频，单卡最高 29 FPS |
| 2025-10 | 🆕 [Self-Forcing++](https://arxiv.org/abs/2510.02283) `arXiv:2510.02283` | UCLA/ByteDance Seed | 把自强制推到分钟级视频 |
| 2025-09 | [Rolling Forcing](https://arxiv.org/abs/2509.25161) `arXiv:2509.25161` | NTU/Tencent | 滚动窗口联合去噪 + 注意力汇聚，实时生成长视频 |
| 2025-09 | [LongLive](https://arxiv.org/abs/2509.22622) `arXiv:2509.22622` | NVIDIA/MIT | 帧级自回归 + KV 重缓存，实时、可交互改提示的长视频 |
| 2025-06 | ★ [Self Forcing](https://arxiv.org/abs/2506.08009) `arXiv:2506.08009` | Adobe/UT Austin | 训练时就用模型自己的输出做自回归展开，消除训练-测试偏差，实时流式生成 |
| 2025-05 | [MAGI-1](https://arxiv.org/abs/2505.13211) `arXiv:2505.13211` | Sand AI | 逐块自回归的大规模视频扩散模型 |
| 2025-04 | [FramePack](https://arxiv.org/abs/2504.12626) `arXiv:2504.12626` | Stanford | 压缩历史帧上下文 + 防漂移采样，消费级显卡生成长视频 |
| 2025-02 | [History Guidance](https://arxiv.org/abs/2502.06764) `arXiv:2502.06764` | MIT | 把历史帧当作可引导的条件，提升长视频一致性 |
| 2024-12 | [NOVA](https://arxiv.org/abs/2412.14169) `arXiv:2412.14169` | BAAI | 无向量量化的自回归视频生成 |
| 2024-12 | ★ [CausVid](https://arxiv.org/abs/2412.07772) `arXiv:2412.07772` | MIT/Adobe | 把双向视频扩散蒸馏成少步因果自回归模型，流式生成 |
| 2024-07 | ★ [Diffusion Forcing](https://arxiv.org/abs/2407.01392) `arXiv:2407.01392` | MIT | 每个 token 独立噪声水平，统一下一 token 预测与全序列扩散，支持变长滚动生成 |

### 世界模型（29）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Hy-Nomas](https://arxiv.org/abs/2610.36729) `arXiv:2610.36729` | Tencent | 8B 扩散骨干，支持自由文本指令与连续动作的交互世界生成 |
| 2026-08 | 🆕 [Matrix-Game 3.5](https://arxiv.org/abs/2608.29910) `arXiv:2608.29910` | Skywork | patch 级记忆增强长时稳定性 |
| 2026-07 | 🆕 [ABot-World-0](https://arxiv.org/abs/2607.19191) `arXiv:2607.19191` | Alibaba AMAP | 单张桌面 GPU 上的无限交互世界展开 |
| 2026-07 | 🆕 [AlayaWorld](https://arxiv.org/abs/2607.18367) `arXiv:2607.18367` | Alaya Lab | 交互长时程世界建模完整技术报告 |
| 2026-06 | 🆕 [DreamX-World 1.0](https://arxiv.org/abs/2606.16993) `arXiv:2606.16993` | Alibaba AMAP | 通用交互世界模型：相机漫游、回访与可提示事件 |
| 2026-06 | 🆕 [潜空间空间记忆](https://arxiv.org/abs/2606.09828) `arXiv:2606.09828` | Microsoft | 在潜空间而非 RGB 点云中维护 3D 记忆 |
| 2026-06 | 🆕 [Cosmos 3](https://arxiv.org/abs/2606.02800) `arXiv:2606.02800` | NVIDIA | 混合 Transformer 架构，联合处理与生成语言、图像、视频、音频与动作 |
| 2026-05 | 🆕 [minWM](https://arxiv.org/abs/2605.30263) `arXiv:2605.30263` | Tsinghua | 实时交互视频世界模型的全栈开源框架 |
| 2026-05 | 🆕 [Gamma-World](https://arxiv.org/abs/2605.28816) `arXiv:2605.28816` | NVIDIA | 多智能体（超过两个玩家）的生成式世界建模 |
| 2026-05 | 🆕 [SANA-WM](https://arxiv.org/abs/2605.15178) `arXiv:2605.15178` | NVIDIA | 2.6B 开源世界模型，原生一分钟 720p，精确相机控制 |
| 2026-04 | 🆕 [HY-World 2.0](https://arxiv.org/abs/2604.14268) `arXiv:2604.14268` | Tencent | 多模态输入的 3D 世界重建、生成与模拟 |
| 2026-04 | 🆕 [Lyra 2.0](https://arxiv.org/abs/2604.13036) `arXiv:2604.13036` | NVIDIA | 生成相机漫游视频再前馈重建为可探索 3D 世界 |
| 2026-04 | 🆕 [Matrix-Game 3.0](https://arxiv.org/abs/2604.08995) `arXiv:2604.08995` | Skywork | 带长时记忆的实时流式交互世界模型 |
| 2026-03 | 🆕 [MosaicMem](https://arxiv.org/abs/2603.17117) `arXiv:2603.17117` |  | 混合空间记忆，让视频世界模型在回访时保持一致 |
| 2026-01 | 🆕 [LingBot-World](https://arxiv.org/abs/2601.20540) `arXiv:2601.20540` | Ant Group | 开源世界模拟器（源自视频生成），覆盖多种环境 |
| 2025-12 | 🆕 [Yume-1.5](https://arxiv.org/abs/2512.22096) `arXiv:2512.22096` | Shanghai AI Lab | 文本可控的交互世界生成，参数与推理步数更省 |
| 2025-12 | 🆕 [WorldPlay](https://arxiv.org/abs/2512.14614) `arXiv:2512.14614` | Tencent | 实时交互世界建模 + 长期几何一致性 |
| 2025-11 | 🆕 [PAN](https://arxiv.org/abs/2511.09057) `arXiv:2511.09057` | MBZUAI | 通用、可交互、长时程的世界模拟器 |
| 2025-10 | 🆕 [World-in-World](https://arxiv.org/abs/2510.18135) `arXiv:2510.18135` | JHU | 在闭环具身任务中评测世界模型 |
| 2025-08 | [Matrix-Game 2.0](https://arxiv.org/abs/2508.13009) `arXiv:2508.13009` | Skywork | 开源实时流式交互世界模型，少步自回归扩散 |
| 2025-07 | [HunyuanWorld 1.0](https://arxiv.org/abs/2507.21809) `arXiv:2507.21809` | Tencent | 从文字或图片生成可探索的沉浸式 3D 世界 |
| 2025-07 | [Yume](https://arxiv.org/abs/2507.17744) `arXiv:2507.17744` | Shanghai AI Lab | 从图像出发的可交互世界生成 |
| 2025-06 | [Matrix-Game](https://arxiv.org/abs/2506.18701) `arXiv:2506.18701` | Skywork | 交互式世界基础模型（Minecraft 场景） |
| 2025-06 | [Hunyuan-GameCraft](https://arxiv.org/abs/2506.17201) `arXiv:2506.17201` | Tencent | 混合历史条件的高动态游戏视频生成 |
| 2025-04 | [WorldMem](https://arxiv.org/abs/2504.12369) `arXiv:2504.12369` | NTU | 带记忆的长期一致世界模拟 |
| 2025-01 | [Cosmos](https://arxiv.org/abs/2501.03575) `arXiv:2501.03575` | NVIDIA | 物理 AI 的世界基础模型平台：数据管线、tokenizer、扩散与自回归模型 |
| 2024-08 | ★ [GameNGen](https://arxiv.org/abs/2408.14837) `arXiv:2408.14837` | Google | 扩散模型实时模拟 DOOM，人类难以区分 |
| 2024-05 | [DIAMOND](https://arxiv.org/abs/2405.12399) `arXiv:2405.12399` | Geneva/Edinburgh | 扩散世界模型里训练 Atari 智能体，视觉细节很重要 |
| 2024-02 | [Genie](https://arxiv.org/abs/2402.15391) `arXiv:2402.15391` | Google DeepMind | 从无标注视频学习潜动作与可交互环境 |

### 音频/音视频（18）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-02 | 🆕 [SkyReels-V4](https://arxiv.org/abs/2602.21818) `arXiv:2602.21818` | Skywork | 双流 MMDiT，统一音视频生成、修补与编辑 |
| 2026-02 | 🆕 [MOVA](https://arxiv.org/abs/2602.08794) `arXiv:2602.08794` | OpenMOSS | 可扩展、同步的开源音视频生成 |
| 2026-01 | ★ 🆕 [LTX-2](https://arxiv.org/abs/2601.03233) `arXiv:2601.03233` | Lightricks | 开源音视频基础模型：14B 视频流 + 5B 音频流，双向交叉注意力 |
| 2025-12 | 🆕 [Seedance 1.5 pro](https://arxiv.org/abs/2512.13507) `arXiv:2512.13507` | ByteDance Seed | 双分支 DiT 的原生音视频联合生成，RLHF 后训练 |
| 2025-11 | 🆕 [Harmony](https://arxiv.org/abs/2511.21579) `arXiv:2511.21579` | Tencent | 跨任务协同缓解音视频联合扩散的对齐问题 |
| 2025-11 | 🆕 [UniAVGen](https://arxiv.org/abs/2511.03334) `arXiv:2511.03334` | Nanjing Univ | 非对称跨模态交互的统一音视频生成 |
| 2025-09 | [Ovi](https://arxiv.org/abs/2510.01284) `arXiv:2510.01284` | Character.AI | 双塔骨干跨模态融合的音视频联合生成 |
| 2025-08 | [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) `arXiv:2508.16930` | Tencent | 带表征对齐的多模态扩散拟音 |
| 2025-06 | [ThinkSound](https://arxiv.org/abs/2506.21448) `arXiv:2506.21448` | Alibaba | 链式推理驱动的音频生成与编辑 |
| 2024-12 | [MMAudio](https://arxiv.org/abs/2412.15322) `arXiv:2412.15322` | UIUC/Sony | 多模态联合训练的视频到音频（拟音） |
| 2024-10 | [F5-TTS](https://arxiv.org/abs/2410.06885) `arXiv:2410.06885` | SJTU | 流匹配 + DiT 的非自回归 TTS，流行开源方案 |
| 2024-07 | [Stable Audio Open](https://arxiv.org/abs/2407.14358) `arXiv:2407.14358` | Stability AI | 开源权重的文本到音频模型 |
| 2023-06 | [Voicebox](https://arxiv.org/abs/2306.15687) `arXiv:2306.15687` | Meta | 流匹配做多语种语音生成与填充，流匹配的早期大规模应用 |
| 2023-04 | [NaturalSpeech 2](https://arxiv.org/abs/2304.09116) `arXiv:2304.09116` | Microsoft | 潜扩散做零样本语音与歌声合成 |
| 2023-01 | [AudioLDM](https://arxiv.org/abs/2301.12503) `arXiv:2301.12503` | Surrey | CLAP 潜空间扩散做文本到音频 |
| 2021-05 | [Grad-TTS](https://arxiv.org/abs/2105.06337) `arXiv:2105.06337` | Huawei | 基于分数的文本到语音 |
| 2020-09 | [DiffWave](https://arxiv.org/abs/2009.09761) `arXiv:2009.09761` | NVIDIA/Baidu | 非自回归扩散声码器 |
| 2020-09 | [WaveGrad](https://arxiv.org/abs/2009.00713) `arXiv:2009.00713` | Google | 估计波形梯度的声码器 |

### 统一多模态（28）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🆕 [多模态预训练的物理学](https://arxiv.org/abs/2608.05000) `arXiv:2608.05000` | Meta | 模态协同、早期统一与训练配方的系统研究 |
| 2026-05 | 🆕 [Representation Forcing](https://arxiv.org/abs/2605.31604) `arXiv:2605.31604` | ByteDance Seed | 去掉冻结 VAE 的统一模型，弥合质量差距 |
| 2026-05 | 🆕 [Lance](https://arxiv.org/abs/2605.18678) `arXiv:2605.18678` | ByteDance | 轻量原生统一模型，覆盖图像与视频的理解、生成、编辑 |
| 2026-05 | 🆕 [SenseNova-U1](https://arxiv.org/abs/2605.12500) `arXiv:2605.12500` | SenseTime | NEO-unify 架构统一理解与生成 |
| 2026-04 | 🆕 [Tuna-2](https://arxiv.org/abs/2604.24763) `arXiv:2604.24763` | Meta | 直接用像素嵌入，抛弃预训练视觉编码器也能统一理解与生成 |
| 2026-04 | 🆕 [Think in Strokes](https://arxiv.org/abs/2604.04746) `arXiv:2604.04746` | Meta | 交错推理的过程式图像生成：规划、草稿、检查、细化 |
| 2026-03 | 🆕 [InternVL-U](https://arxiv.org/abs/2603.09877) `arXiv:2603.09877` | Shanghai AI Lab | 开源统一的理解、推理、生成与编辑模型 |
| 2026-03 | 🆕 [超越语言建模](https://arxiv.org/abs/2603.03276) `arXiv:2603.03276` | Meta | 从零预训练的原生多模态模型设计空间对照研究 |
| 2026-02 | 🆕 [DeepGen 1.0](https://arxiv.org/abs/2602.12205) `arXiv:2602.12205` |  | 5B 轻量统一模型做生成与编辑 |
| 2026-01 | 🆕 [NextFlow](https://arxiv.org/abs/2601.02204) `arXiv:2601.02204` | ByteDance | 6T 交错 token 上训练的统一自回归模型 |
| 2025-12 | 🆕 [TUNA](https://arxiv.org/abs/2512.02014) `arXiv:2512.02014` | Meta | VAE 编码器级联表征编码器，构造统一连续视觉表征 |
| 2025-10 | 🆕 [Emu3.5](https://arxiv.org/abs/2510.26583) `arXiv:2510.26583` | BAAI | 大规模交错视觉语言数据上的原生多模态"世界学习者"（下一 token 预测；DiDA 离散扩散适配让单图推理快约 20 倍） |
| 2025-10 | 🆕 [BLIP3o-NEXT](https://arxiv.org/abs/2510.15857) `arXiv:2510.15857` | Salesforce | 自回归 + 扩散的原生图像生成，含 RL |
| 2025-10 | 🆕 [Ming-UniVision](https://arxiv.org/abs/2510.06590) `arXiv:2510.06590` | inclusionAI | 统一连续 tokenizer 的自回归理解与生成 |
| 2025-09 | [Manzano](https://arxiv.org/abs/2509.16197) `arXiv:2509.16197` | Apple | 混合视觉 tokenizer + 扩散解码器的统一模型 |
| 2025-08 | [NextStep-1](https://arxiv.org/abs/2508.10711) `arXiv:2508.10711` | StepFun | 14B 自回归 + 157M 流匹配头，连续 token 文生图 |
| 2025-06 | [Show-o2](https://arxiv.org/abs/2506.15564) `arXiv:2506.15564` | NUS | 原生统一模型，流匹配生成图像与视频 |
| 2025-06 | [UniWorld-V1](https://arxiv.org/abs/2506.03147) `arXiv:2506.03147` | PKU | 高分辨率语义编码器驱动的统一模型 |
| 2025-05 | ★ [BAGEL](https://arxiv.org/abs/2505.14683) `arXiv:2505.14683` | ByteDance Seed | MoT 架构在海量交错数据上预训练，涌现出复杂编辑与世界导航能力 |
| 2025-05 | [BLIP3-o](https://arxiv.org/abs/2505.09568) `arXiv:2505.09568` | Salesforce | 用扩散 Transformer 生成 CLIP 特征的全开源统一模型 |
| 2025-04 | [MetaQuery](https://arxiv.org/abs/2504.06256) `arXiv:2504.06256` | Meta | 可学习查询把冻结 MLLM 的知识传给扩散解码器 |
| 2025-01 | [Janus-Pro](https://arxiv.org/abs/2501.17811) `arXiv:2501.17811` | DeepSeek | 解耦视觉编码的统一模型扩展版 |
| 2024-11 | [JanusFlow](https://arxiv.org/abs/2411.07975) `arXiv:2411.07975` | DeepSeek | 自回归语言模型 + Rectified Flow 的统一理解生成 |
| 2024-10 | [Fluid](https://arxiv.org/abs/2410.13863) `arXiv:2410.13863` | Google | 连续 token + 随机顺序的自回归文生图扩展研究 |
| 2024-08 | [Show-o](https://arxiv.org/abs/2408.12528) `arXiv:2408.12528` | NUS | 一个 Transformer 融合自回归与离散扩散 |
| 2024-08 | ★ [Transfusion](https://arxiv.org/abs/2408.11039) `arXiv:2408.11039` | Meta | 同一 Transformer 对文本做下一词预测、对图像做扩散 |
| 2024-06 | ★ [MAR](https://arxiv.org/abs/2406.11838) `arXiv:2406.11838` | MIT/Google | 自回归 + 每 token 小扩散头（diffusion loss），去掉向量量化 |
| 2024-04 | [VAR](https://arxiv.org/abs/2404.02905) `arXiv:2404.02905` | ByteDance/PKU | "下一尺度预测"的自回归图像生成（扩散的主要竞争路线之一） |

### 扩散语言模型（57）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Nemotron-Labs-Diffusion-Uni](https://arxiv.org/abs/2610.36466) `arXiv:2610.36466` | NVIDIA | 自我细化的统一离散扩散 LLM（视觉理解与生成） |
| 2026-10 | 🆕 [Large Flow Language Models](https://arxiv.org/abs/2610.28559) `arXiv:2610.28559` |  | 十亿参数级连续流语言模型的构建方法 |
| 2026-10 | 🆕 [层次化连续扩散 LM](https://arxiv.org/abs/2610.02193) `arXiv:2610.02193` |  | 缓解并行解码中 token 之间的独立性瓶颈 |
| 2026-09 | 🆕 [扩散增强 LLM](https://arxiv.org/abs/2609.04010) `arXiv:2609.04010` | MBZUAI | 给自回归 LLM 加离散扩散实现无损加速 |
| 2026-08 | 🆕 [LLaDA MoE v2](https://arxiv.org/abs/2608.03457) `arXiv:2608.03457` | Ant Group | 系统刻画 MoE 扩散语言模型的扩展规律 |
| 2026-08 | 🆕 [AURORA-LM](https://arxiv.org/abs/2608.02602) `arXiv:2608.02602` | Nanjing Univ | 统一表征自编码 + 连续潜扩散语言模型 |
| 2026-07 | ★ 🆕 [DiffusionGemma](https://arxiv.org/abs/2608.00146) `arXiv:2608.00146` | Google DeepMind | 实验性开源权重扩散语言模型，每次并行细化 256 个 token 的块 |
| 2026-06 | 🆕 [iLLaDA](https://arxiv.org/abs/2606.25331) `arXiv:2606.25331` | ByteDance Seed | 从零训练的 8B 全双向掩码扩散语言模型改进版 |
| 2026-06 | 🆕 [dLLM on-policy 自蒸馏](https://arxiv.org/abs/2606.18195) `arXiv:2606.18195` | Max Planck | 把 on-policy 自蒸馏后训练推广到扩散 LLM |
| 2026-05 | 🆕 [连续潜扩散 LM](https://arxiv.org/abs/2605.06548) `arXiv:2605.06548` | ByteDance Seed | 在连续潜空间做语言扩散，兼顾效率与质量 |
| 2026-04 | 🆕 [LLaDA2.0-Uni](https://arxiv.org/abs/2604.20796) `arXiv:2604.20796` | Ant Group | 语义离散 tokenizer + MoE 的统一离散扩散多模态模型 |
| 2026-04 | 🆕 [LangFlow](https://arxiv.org/abs/2604.11748) `arXiv:2604.11748` |  | 连续扩散语言模型首次追平离散扩散 |
| 2026-04 | 🆕 [DMax](https://arxiv.org/abs/2604.08302) `arXiv:2604.08302` | NUS | 缓解并行解码误差累积，实现激进的并行度 |
| 2026-03 | 🆕 [Lingshu-Cell](https://arxiv.org/abs/2603.25240) `arXiv:2603.25240` | Alibaba DAMO | 离散扩散的细胞转录组生成世界模型 |
| 2026-03 | 🆕 [LLaDA-o](https://arxiv.org/abs/2603.01068) `arXiv:2603.01068` | Renmin Univ | 文本掩码扩散 + 图像连续扩散的"扩散混合"全模态模型 |
| 2026-02 | 🆕 [dLLM 库](https://arxiv.org/abs/2602.22661) `arXiv:2602.22661` | UC Berkeley | 扩散语言模型共享组件的简洁开源实现 |
| 2026-02 | 🆕 [LLaDA2.1](https://arxiv.org/abs/2602.08676) `arXiv:2602.08676` | Ant Group | 引入 token 编辑，加速文本扩散 |
| 2026-02 | 🆕 [DFlash](https://arxiv.org/abs/2602.06036) `arXiv:2602.06036` |  | 用块扩散做投机解码的草稿模型 |
| 2026-01 | 🆕 [Stable-DiffCoder](https://arxiv.org/abs/2601.15892) `arXiv:2601.15892` | ByteDance Seed | 8B 代码扩散语言模型，缩小与自回归代码模型的差距 |
| 2026-01 | 🆕 [灵活性陷阱](https://arxiv.org/abs/2601.15165) `arXiv:2601.15165` | Tsinghua | 任意顺序生成反而限制推理潜力 |
| 2025-12 | 🆕 [Dream-VL / Dream-VLA](https://arxiv.org/abs/2512.22615) `arXiv:2512.22615` | HKU | 以扩散语言模型为骨干的开放视觉语言与 VLA 模型 |
| 2025-12 | 🆕 [DiRL](https://arxiv.org/abs/2512.22234) `arXiv:2512.22234` | Fudan | 扩散语言模型的高效后训练框架 |
| 2025-12 | 🆕 [DEER](https://arxiv.org/abs/2512.15176) `arXiv:2512.15176` |  | 扩散起草、自回归验证的投机解码 |
| 2025-12 | 🆕 [ReFusion](https://arxiv.org/abs/2512.13586) `arXiv:2512.13586` |  | 并行自回归解码的扩散 LLM，恢复 KV 缓存 |
| 2025-12 | ★ 🆕 [LLaDA2.0](https://arxiv.org/abs/2512.15745) `arXiv:2512.15745` | Ant Group | 从自回归模型系统转换，块扩散扩展到 100B 总参数 |
| 2025-11 | 🆕 [TiDAR](https://arxiv.org/abs/2511.08923) `arXiv:2511.08923` | NVIDIA | "用扩散思考、用自回归说话"：一次前向内并行起草 + 自回归校验 |
| 2025-11 | 🆕 [MMaDA-Parallel](https://arxiv.org/abs/2511.09611) `arXiv:2511.09611` | ByteDance | 文本与图像并行去噪，避免"先想再画"的误差累积 |
| 2025-11 | 🆕 [超级数据学习者](https://arxiv.org/abs/2511.03276) `arXiv:2511.03276` | NUS | 唯一数据有限时，扩散 LM 多轮训练稳定超过自回归，"交叉点"随数据增多后移 |
| 2025-10 | 🆕 [自适应 KV 重算](https://arxiv.org/abs/2510.14973) `arXiv:2510.14973` | MBZUAI | 按注意力自适应决定何时重算 KV 缓存 |
| 2025-10 | 🆕 [SDAR](https://arxiv.org/abs/2510.06303) `arXiv:2510.06303` | Shanghai AI Lab | 自回归训练 + 块内扩散解码的协同范式 |
| 2025-10 | 🆕 [Lumina-DiMOO](https://arxiv.org/abs/2510.06308) `arXiv:2510.06308` | Shanghai AI Lab | 全离散扩散的多模态理解与生成 |
| 2025-09 | [Fast-dLLM v2](https://arxiv.org/abs/2509.26328) `arXiv:2509.26328` | NVIDIA | 把自回归模型高效改造成块扩散 LLM |
| 2025-09 | [LLaDA-MoE](https://arxiv.org/abs/2509.24389) `arXiv:2509.24389` | Ant Group | 稀疏 MoE 扩散语言模型 |
| 2025-09 | [CoDA](https://arxiv.org/abs/2510.03270) `arXiv:2510.03270` | Salesforce | 1.7B 开源扩散代码模型 |
| 2025-09 | [TraceRL](https://arxiv.org/abs/2509.06949) `arXiv:2509.06949` | Princeton | 轨迹感知的扩散语言模型强化学习（TraDo） |
| 2025-09 | [Dream-Coder 7B](https://arxiv.org/abs/2509.01142) `arXiv:2509.01142` | HKU | 开源代码扩散语言模型 |
| 2025-08 | [Dream 7B](https://arxiv.org/abs/2508.15487) `arXiv:2508.15487` | HKU/Huawei | 从自回归模型初始化的 7B 扩散语言模型 |
| 2025-08 | [Seed Diffusion](https://arxiv.org/abs/2508.02193) `arXiv:2508.02193` | ByteDance Seed | 大规模代码扩散语言模型，高速推理 |
| 2025-07 | [数据受限时扩散胜出](https://arxiv.org/abs/2507.15857) `arXiv:2507.15857` | CMU | 数据重复多轮时，扩散语言模型优于自回归 |
| 2025-06 | [DiffuCoder](https://arxiv.org/abs/2506.20639) `arXiv:2506.20639` | Apple/HKU | 分析掩码扩散写代码的解码行为，提出耦合 GRPO |
| 2025-06 | [Mercury](https://arxiv.org/abs/2506.17298) `arXiv:2506.17298` | Inception Labs | 商用扩散语言模型，代码生成每秒上千 token |
| 2025-06 | [Duo](https://arxiv.org/abs/2506.10892) `arXiv:2506.10892` | Cornell | 揭示均匀态离散扩散与高斯扩散的对偶，可借用连续域的蒸馏技巧做少步生成 |
| 2025-06 | [Eso-LM](https://arxiv.org/abs/2506.01928) `arXiv:2506.01928` | Cornell | 融合自回归与掩码扩散，任意顺序生成 |
| 2025-05 | [Fast-dLLM](https://arxiv.org/abs/2505.22618) `arXiv:2505.22618` | NVIDIA/HKU | 免训练的块级近似 KV 缓存 + 置信度并行解码 |
| 2025-05 | [LLaDA 1.5](https://arxiv.org/abs/2505.19223) `arXiv:2505.19223` | Renmin Univ/Ant | 方差缩减的偏好优化 |
| 2025-05 | [LLaDA-V](https://arxiv.org/abs/2505.16933) `arXiv:2505.16933` | Renmin Univ | 视觉指令微调的扩散多模态模型 |
| 2025-05 | [dKV-Cache](https://arxiv.org/abs/2505.15781) `arXiv:2505.15781` | NUS | 扩散语言模型的延迟 KV 缓存 |
| 2025-05 | [MMaDA](https://arxiv.org/abs/2505.15809) `arXiv:2505.15809` | Princeton/ByteDance | 多模态大扩散语言模型，统一推理、理解与生成 |
| 2025-04 | [d1](https://arxiv.org/abs/2504.12216) `arXiv:2504.12216` | UCLA/Meta | diffu-GRPO：扩散语言模型的推理强化学习 |
| 2025-03 | ★ [BD3-LM](https://arxiv.org/abs/2503.09573) `arXiv:2503.09573` | Cornell | 块扩散：块间自回归、块内扩散，支持 KV 缓存与变长生成 |
| 2025-02 | ★ [LLaDA](https://arxiv.org/abs/2502.09992) `arXiv:2502.09992` | Renmin Univ/Ant | 8B 掩码扩散语言模型从零预训练，能力与同规模自回归模型可比，解决"逆转诅咒" |
| 2024-06 | ★ [MDLM](https://arxiv.org/abs/2406.07524) `arXiv:2406.07524` | Cornell | 把掩码扩散简化为加权的掩码语言建模损失 |
| 2024-06 | [MD4](https://arxiv.org/abs/2406.04329) `arXiv:2406.04329` | Google DeepMind | 简化并推广的掩码扩散，连续时间 ELBO 只是交叉熵的积分 |
| 2023-10 | ★ [SEDD](https://arxiv.org/abs/2310.16834) `arXiv:2310.16834` | Stanford | 用"分数熵"估计数据分布比率，离散扩散首次在困惑度上逼近 GPT-2（ICML 2024 最佳论文） |
| 2022-10 | [DiffuSeq](https://arxiv.org/abs/2210.08933) `arXiv:2210.08933` | HKU/Shanghai AI Lab | 序列到序列的文本扩散 |
| 2022-05 | [Diffusion-LM](https://arxiv.org/abs/2205.14217) `arXiv:2205.14217` | Stanford | 连续嵌入空间的文本扩散，可控生成 |
| 2021-07 | ★ [D3PM](https://arxiv.org/abs/2107.03006) `arXiv:2107.03006` | Google | 离散状态空间的结构化扩散（均匀、吸收态等转移矩阵） |

### 3D/感知（19）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🆕 [Hunyuan3D-Buffalo](https://arxiv.org/abs/2608.02711) `arXiv:2608.02711` | Tencent | 统一 3D 生成、理解与编辑的多模态模型 |
| 2026-07 | 🆕 [Meshy T2](https://arxiv.org/abs/2607.28675) `arXiv:2607.28675` | Meshy | 流匹配直接生成具有艺术家拓扑的网格 |
| 2025-12 | 🆕 [TRELLIS.2](https://arxiv.org/abs/2512.14692) `arXiv:2512.14692` | Microsoft | 原生紧凑的结构化 3D 潜变量，提升几何细节与效率 |
| 2025-12 | 🆕 [文生 3D 的强化学习](https://arxiv.org/abs/2512.10949) `arXiv:2512.10949` | Shanghai AI Lab | 系统研究 RL 用于 3D 生成的可行性 |
| 2025-11 | 🆕 [SAM 3D](https://arxiv.org/abs/2511.16624) `arXiv:2511.16624` | Meta | 从单张自然图像重建物体几何、纹理与布局（含遮挡场景） |
| 2025-11 | 🆕 [PhysX-Anything](https://arxiv.org/abs/2511.13648) `arXiv:2511.13648` | NTU | 从单图生成可直接仿真的带关节物理 3D 资产 |
| 2025-10 | 🆕 [FlashWorld](https://arxiv.org/abs/2510.13678) `arXiv:2510.13678` | Xiamen Univ/Tencent | 数秒内生成高质量 3D 场景，比先前方法快 10–100 倍 |
| 2025-09 | [Hunyuan3D-Omni](https://arxiv.org/abs/2509.21245) `arXiv:2509.21245` | Tencent | 点云、体素、骨架等多种条件的可控 3D 生成 |
| 2025-09 | [Lyra](https://arxiv.org/abs/2509.19296) `arXiv:2509.19296` | NVIDIA | 视频扩散模型自蒸馏到 3D 高斯场景 |
| 2025-06 | [Hunyuan3D 2.1](https://arxiv.org/abs/2506.15442) `arXiv:2506.15442` | Tencent | 面向生产的 PBR 材质 3D 生成 |
| 2025-01 | [DiffusionRenderer](https://arxiv.org/abs/2501.18590) `arXiv:2501.18590` | NVIDIA | 视频扩散模型做逆渲染与正向渲染 |
| 2025-01 | [Hunyuan3D 2.0](https://arxiv.org/abs/2501.12202) `arXiv:2501.12202` | Tencent | 流匹配 DiT 生成形状 + 纹理合成的开源 3D 资产系统 |
| 2024-12 | ★ [TRELLIS](https://arxiv.org/abs/2412.01506) `arXiv:2412.01506` | Microsoft | 结构化 3D 潜变量 + 修正流 Transformer 的通用 3D 资产生成 |
| 2023-12 | [Marigold](https://arxiv.org/abs/2312.02145) `arXiv:2312.02145` | ETH | 把文生图扩散模型微调为单目深度估计器，生成先验用于感知 |
| 2023-08 | [MVDream](https://arxiv.org/abs/2308.16512) `arXiv:2308.16512` | ByteDance | 多视角一致的扩散模型，作为 3D 生成先验 |
| 2023-05 | [Shap-E](https://arxiv.org/abs/2305.02463) `arXiv:2305.02463` | OpenAI | 生成隐式函数参数的条件 3D 生成 |
| 2023-03 | [Zero-1-to-3](https://arxiv.org/abs/2303.11328) `arXiv:2303.11328` | Columbia | 微调扩散模型做相机视角控制的新视角合成 |
| 2022-12 | [Point-E](https://arxiv.org/abs/2212.08751) `arXiv:2212.08751` | OpenAI | 文本→图像→点云的快速 3D 生成 |
| 2022-09 | ★ [DreamFusion](https://arxiv.org/abs/2209.14988) `arXiv:2209.14988` | Google | 分数蒸馏采样（SDS）：用 2D 文生图模型优化 NeRF，开启文生 3D |

### 机器人（20）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Kinematic MeanFlow](https://arxiv.org/abs/2610.00864) `arXiv:2610.00864` | Intel Labs China | 把 MeanFlow 用于机器人基础模型的一步动作生成 |
| 2026-03 | 🆕 [FASTER](https://arxiv.org/abs/2603.19199) `arXiv:2603.19199` |  | 重新思考实时流式 VLA 的反应延迟 |
| 2026-02 | 🆕 [DreamZero](https://arxiv.org/abs/2602.15922) `arXiv:2602.15922` | NVIDIA | 基于预训练视频扩散的世界动作模型，零样本泛化到新动作 |
| 2025-11 | 🆕 [π*0.6](https://arxiv.org/abs/2511.14759) `arXiv:2511.14759` | Physical Intelligence | 用自身经验（RECAP 优势条件）持续改进的 VLA |
| 2025-11 | 🆕 [PhysWorld](https://arxiv.org/abs/2511.07416) `arXiv:2511.07416` | Google DeepMind | 从视频生成 + 物理世界建模中学习机器人技能 |
| 2025-10 | 🆕 [π_RL](https://arxiv.org/abs/2510.25889) `arXiv:2510.25889` | RLinf | 流式 VLA 的在线强化学习微调 |
| 2025-10 | 🆕 [ACG](https://arxiv.org/abs/2510.22201) `arXiv:2510.22201` | KAIST | 动作连贯性引导，减少流式 VLA 的抖动 |
| 2025-08 | [Genie Envisioner](https://arxiv.org/abs/2508.05635) `arXiv:2508.05635` | AgiBot | 机器人操作的统一世界基础平台 |
| 2025-06 | [实时分块执行（RTC）](https://arxiv.org/abs/2506.07339) `arXiv:2506.07339` | Physical Intelligence | 把动作块衔接写成修补问题，用引导实现异步实时执行 |
| 2025-05 | [DreamGen](https://arxiv.org/abs/2505.12705) `arXiv:2505.12705` | NVIDIA | 用视频世界模型合成机器人训练数据 |
| 2025-04 | [π0.5](https://arxiv.org/abs/2504.16054) `arXiv:2504.16054` | Physical Intelligence | 开放世界泛化的 VLA（流匹配动作头） |
| 2025-04 | [UWM](https://arxiv.org/abs/2504.02792) `arXiv:2504.02792` | UW | 耦合视频扩散与动作扩散的统一世界模型 |
| 2025-03 | [GR00T N1](https://arxiv.org/abs/2503.14734) `arXiv:2503.14734` | NVIDIA | 双系统人形机器人基础模型，流匹配 DiT 动作头 |
| 2024-10 | ★ [π0](https://arxiv.org/abs/2410.24164) `arXiv:2410.24164` | Physical Intelligence | VLM + 流匹配动作专家的通用机器人策略 |
| 2024-10 | [RDT-1B](https://arxiv.org/abs/2410.07864) `arXiv:2410.07864` | Tsinghua | 双臂操作的扩散基础模型 |
| 2024-09 | [DPPO](https://arxiv.org/abs/2409.00588) `arXiv:2409.00588` | Princeton | 用策略梯度微调扩散策略 |
| 2024-03 | [DP3](https://arxiv.org/abs/2403.03954) `arXiv:2403.03954` | Shanghai Qi Zhi/Tsinghua | 基于点云表示的 3D 扩散策略 |
| 2023-03 | ★ [Diffusion Policy](https://arxiv.org/abs/2303.04137) `arXiv:2303.04137` | Columbia/TRI | 用扩散模型建模多峰动作分布，视觉运动策略的标准做法之一 |
| 2023-01 | [UniPi](https://arxiv.org/abs/2302.00111) `arXiv:2302.00111` | MIT/Google | 文本引导的视频生成作为通用策略 |
| 2022-05 | [Diffuser](https://arxiv.org/abs/2205.09991) `arXiv:2205.09991` | UC Berkeley/MIT | 把轨迹规划写成扩散去噪，奖励引导即规划 |

### 科学（12）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-02 | 🆕 [MolHIT](https://arxiv.org/abs/2602.17602) `arXiv:2602.17602` | KAIST | 层次化离散扩散的分子图生成 |
| 2025-09 | [RFdiffusion3](https://doi.org/10.1101/2025.09.18.676967) | UW IPD | 全原子级生物分子相互作用的从头设计，2025-12 开源（bioRxiv） |
| 2025-07 | [La-Proteina](https://arxiv.org/abs/2507.09466) `arXiv:2507.09466` | NVIDIA | 部分潜空间流匹配的全原子蛋白质生成 |
| 2025-03 | [Proteina](https://arxiv.org/abs/2503.00710) `arXiv:2503.00710` | NVIDIA | 大规模流匹配蛋白质结构生成 |
| 2024-05 | ★ [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) | Google DeepMind/Isomorphic | 用扩散模块直接生成全原子坐标的生物分子复合物结构预测（Nature） |
| 2023-12 | ★ [GenCast](https://arxiv.org/abs/2312.15796) `arXiv:2312.15796` | Google DeepMind | 扩散集合预报，15 天中期天气预报超过 ECMWF ENS |
| 2023-12 | ★ [MatterGen](https://arxiv.org/abs/2312.03687) `arXiv:2312.03687` | Microsoft | 扩散模型生成满足目标性质的无机材料 |
| 2023-10 | [FrameFlow](https://arxiv.org/abs/2310.05297) `arXiv:2310.05297` | MIT | SE(3) 流匹配生成蛋白质骨架，更快更好 |
| 2023-07 | ★ [RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8) | UW IPD | 在 RoseTTAFold 上微调的扩散模型做从头蛋白质设计（Nature） |
| 2023-02 | [FrameDiff](https://arxiv.org/abs/2302.02277) `arXiv:2302.02277` | MIT | SE(3) 扩散生成蛋白质骨架 |
| 2022-10 | [DiffDock](https://arxiv.org/abs/2210.01776) `arXiv:2210.01776` | MIT | 把分子对接建模为平移/旋转/扭转空间的扩散 |
| 2022-03 | [EDM（分子）](https://arxiv.org/abs/2203.17003) `arXiv:2203.17003` | Amsterdam | E(3) 等变扩散生成 3D 分子 |

### 推理加速（30）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [MC-Sparse](https://arxiv.org/abs/2610.06801) `arXiv:2610.06801` | Tencent | 分析并弥合稠密与稀疏注意力之间的质量差距 |
| 2026-07 | 🆕 [Sol-Attn](https://arxiv.org/abs/2607.24027) `arXiv:2607.24027` | NVIDIA | 在线动态稀疏注意力加速视频推理 |
| 2026-07 | 🆕 [OrbitQuant](https://arxiv.org/abs/2607.02461) `arXiv:2607.02461` |  | 与数据无关的 DiT 训练后量化 |
| 2026-02 | 🆕 [SpargeAttention2](https://arxiv.org/abs/2602.13515) `arXiv:2602.13515` | Tsinghua | 可训练的 Top-k + Top-p 混合稀疏注意力 |
| 2026-02 | 🆕 [SLA2](https://arxiv.org/abs/2602.12675) `arXiv:2602.12675` | Tsinghua/UC Berkeley | 可学习路由的稀疏-线性注意力 + 量化感知训练 |
| 2026-02 | 🆕 [FSVideo](https://arxiv.org/abs/2602.02092) `arXiv:2602.02092` | ByteDance | 64×64×4 高压缩潜空间的快速图生视频 |
| 2026-01 | 🆕 [SnapGen++](https://arxiv.org/abs/2601.08303) `arXiv:2601.08303` | Snap | 端侧可部署的高保真 DiT |
| 2025-12 | ★ 🆕 [TurboDiffusion](https://arxiv.org/abs/2512.16093) `arXiv:2512.16093` | Tsinghua/Berkeley | 注意力量化 + 稀疏线性注意力 + rCM 少步蒸馏等组合，端到端视频生成加速 100–200 倍 |
| 2025-10 | 🆕 [FlashVSR](https://arxiv.org/abs/2510.12747) `arXiv:2510.12747` |  | 实时流式扩散视频超分辨率 |
| 2025-09 | [DC-VideoGen](https://arxiv.org/abs/2509.25182) `arXiv:2509.25182` | MIT/NVIDIA | 深度压缩视频自编码器 + 轻量适配，后训练加速视频模型 |
| 2025-09 | [SLA](https://arxiv.org/abs/2509.24006) `arXiv:2509.24006` | Tsinghua | 稀疏 + 线性注意力混合，可微调 |
| 2025-06 | [Radial Attention](https://arxiv.org/abs/2506.19852) `arXiv:2506.19852` | MIT | 按能量衰减设计 O(n log n) 稀疏注意力，长视频生成 |
| 2025-06 | [MagCache](https://arxiv.org/abs/2506.09045) `arXiv:2506.09045` |  | 按残差幅值比自适应缓存 |
| 2025-05 | [Sparse VideoGen2](https://arxiv.org/abs/2505.18875) `arXiv:2505.18875` | UC Berkeley | 语义感知重排的稀疏注意力 |
| 2025-05 | [VSA](https://arxiv.org/abs/2505.13389) `arXiv:2505.13389` | UCSD | 可训练的视频稀疏注意力 |
| 2025-05 | [SageAttention3](https://arxiv.org/abs/2505.11594) `arXiv:2505.11594` | Tsinghua | FP4 微缩放注意力，并探索 8 比特训练 |
| 2025-03 | [TaylorSeer](https://arxiv.org/abs/2503.06923) `arXiv:2503.06923` | SJTU | 用泰勒展开预测未来步特征，代替直接复用 |
| 2025-02 | [SpargeAttn](https://arxiv.org/abs/2502.18137) `arXiv:2502.18137` | Tsinghua | 免训练稀疏注意力，任意模型可用 |
| 2025-02 | [STA](https://arxiv.org/abs/2502.04507) `arXiv:2502.04507` | UCSD | 滑动块注意力加速视频生成 |
| 2025-02 | [Sparse VideoGen](https://arxiv.org/abs/2502.01776) `arXiv:2502.01776` | UC Berkeley | 利用视频 DiT 注意力的时空稀疏性 |
| 2024-11 | [TeaCache](https://arxiv.org/abs/2411.19108) `arXiv:2411.19108` | Alibaba | 按时间嵌入估计输出变化，决定何时复用缓存 |
| 2024-11 | [SageAttention2](https://arxiv.org/abs/2411.10958) `arXiv:2411.10958` | Tsinghua | INT4 + 离群值平滑的注意力 |
| 2024-11 | [SVDQuant](https://arxiv.org/abs/2411.05007) `arXiv:2411.05007` | MIT | 低秩分支吸收离群值，4 比特量化 FLUX |
| 2024-11 | [xDiT](https://arxiv.org/abs/2411.01738) `arXiv:2411.01738` | Tencent | DiT 大规模并行推理引擎 |
| 2024-10 | [SageAttention](https://arxiv.org/abs/2410.02367) `arXiv:2410.02367` | Tsinghua | 8 比特注意力，即插即用 |
| 2024-08 | [PAB](https://arxiv.org/abs/2408.12588) `arXiv:2408.12588` | NUS | 金字塔注意力广播，实时视频生成 |
| 2024-02 | [DistriFusion](https://arxiv.org/abs/2402.19481) `arXiv:2402.19481` | MIT | 多卡分块并行生成高分辨率图像 |
| 2023-12 | [DeepCache](https://arxiv.org/abs/2312.00858) `arXiv:2312.00858` | NUS | 缓存 U-Net 高层特征，相邻步复用 |
| 2023-03 | [ToMe for SD](https://arxiv.org/abs/2303.17604) `arXiv:2303.17604` | Georgia Tech | 合并冗余 token，免训练加速 Stable Diffusion |
| 2023-02 | [Q-Diffusion](https://arxiv.org/abs/2302.04304) `arXiv:2302.04304` | UC Berkeley | 扩散模型的训练后量化 |

### 评测基准（28）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [StreamEval](https://arxiv.org/abs/2610.28402) `arXiv:2610.28402` |  | 首个交互式流式视频生成的开源评测 |
| 2026-06 | 🆕 [FID 彩票](https://arxiv.org/abs/2606.20536) `arXiv:2606.20536` | Kyutai | 量化 FID 在重训与换种子下的隐藏随机性 |
| 2026-05 | 🆕 [WBench](https://arxiv.org/abs/2605.25874) `arXiv:2605.25874` | Meituan | 交互视频世界模型的多轮评测 |
| 2026-03 | 🆕 [ViGoR-Bench](https://arxiv.org/abs/2603.25823) `arXiv:2603.25823` | Meituan | 视觉生成模型的零样本视觉推理评测 |
| 2026-03 | 🆕 [一步模型公平评测](https://arxiv.org/abs/2603.14186) `arXiv:2603.14186` |  | 设置对齐、语义分档的一步/多步模型对比评测 |
| 2026-01 | 🆕 [T2I 空间智能评测](https://arxiv.org/abs/2601.20354) `arXiv:2601.20354` |  | 文生图的空间感知、推理与交互评测 |
| 2025-12 | 🆕 [VideoASMR-Bench](https://arxiv.org/abs/2512.13281) `arXiv:2512.13281` |  | AI 生成的 ASMR 视频能否骗过 VLM 与人类 |
| 2025-11 | 🆕 [V-ReasonBench](https://arxiv.org/abs/2511.16668) `arXiv:2511.16668` |  | 视频生成模型的统一推理评测 |
| 2025-11 | 🆕 [TiViBench](https://arxiv.org/abs/2511.13704) `arXiv:2511.13704` |  | "用视频思考"的推理评测 |
| 2025-10 | 🆕 [UniGenBench++](https://arxiv.org/abs/2510.18701) `arXiv:2510.18701` | Tencent Hunyuan | 统一的文生图语义评测 |
| 2025-10 | 🆕 [PICABench](https://arxiv.org/abs/2510.17681) `arXiv:2510.17681` |  | 物理真实的图像编辑评测 |
| 2025-06 | [OneIG-Bench](https://arxiv.org/abs/2506.07977) `arXiv:2506.07977` |  | 多维度细粒度图像生成评测 |
| 2025-04 | [WorldScore](https://arxiv.org/abs/2504.00983) `arXiv:2504.00983` | Stanford | 世界生成统一评测 |
| 2025-03 | [VBench-2.0](https://arxiv.org/abs/2503.21755) `arXiv:2503.21755` | NTU | 评测视频的"内在真实性"（物理、常识、人体等） |
| 2025-03 | [WISE](https://arxiv.org/abs/2503.07265) `arXiv:2503.07265` | PKU | 世界知识驱动的文生图语义评测 |
| 2025-01 | [Physics-IQ](https://arxiv.org/abs/2501.09038) `arXiv:2501.09038` | Google DeepMind | 视频生成模型是否理解物理原理 |
| 2024-10 | [PhyGenBench](https://arxiv.org/abs/2410.05363) `arXiv:2410.05363` |  | 物理常识视频生成评测 |
| 2024-06 | [VideoPhy](https://arxiv.org/abs/2406.03520) `arXiv:2406.03520` | UCLA | 视频生成的物理常识评测 |
| 2024-04 | [VQAScore](https://arxiv.org/abs/2404.01291) `arXiv:2404.01291` | CMU/Meta | 用 VQA 模型回答"图中是否是…"打分 |
| 2024-03 | [ELLA / DPG-Bench](https://arxiv.org/abs/2403.05135) `arXiv:2403.05135` | Tencent | 用 LLM 增强语义对齐，并提出密集提示评测 DPG-Bench |
| 2023-11 | [CMMD](https://arxiv.org/abs/2401.09603) `arXiv:2401.09603` | Google | 用 CLIP 嵌入 + MMD 替代 FID |
| 2023-11 | ★ [VBench](https://arxiv.org/abs/2311.17982) `arXiv:2311.17982` | NTU | 分 16 个维度的视频生成评测 |
| 2023-10 | [DSG](https://arxiv.org/abs/2310.18235) `arXiv:2310.18235` | Google | 戴维森场景图，细粒度文图一致性评估 |
| 2023-10 | ★ [GenEval](https://arxiv.org/abs/2310.11513) `arXiv:2310.11513` | UW | 以物体为中心的组合生成评测（计数、位置、颜色） |
| 2023-07 | [T2I-CompBench](https://arxiv.org/abs/2307.06350) `arXiv:2307.06350` | HKU | 组合式文生图评测 |
| 2023-06 | [FD-DINOv2](https://arxiv.org/abs/2306.04675) `arXiv:2306.04675` | Layer 6 | 指出 FID 对扩散模型不公平，建议用 DINOv2 特征 |
| 2023-03 | [TIFA](https://arxiv.org/abs/2303.11897) `arXiv:2303.11897` | UW | 用问答评估文图一致性 |
| 2017-06 | ★ [FID](https://arxiv.org/abs/1706.08500) `arXiv:1706.08500` | JKU Linz | Fréchet Inception 距离，生成模型最常用指标 |

### 安全/版权（11）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-10 | 🆕 [SynthID-Image](https://arxiv.org/abs/2510.09263) `arXiv:2510.09263` | Google DeepMind | 互联网规模部署的图像水印系统 |
| 2024-11 | [Watermark Anything](https://arxiv.org/abs/2411.07231) `arXiv:2411.07231` | Meta | 局部化多信息图像水印 |
| 2023-10 | [Nightshade](https://arxiv.org/abs/2310.13828) `arXiv:2310.13828` | UChicago | 针对特定提示的数据投毒 |
| 2023-10 | [Ring-A-Bell](https://arxiv.org/abs/2310.10012) `arXiv:2310.10012` |  | 红队测试：概念擦除方法并不可靠 |
| 2023-08 | [UCE](https://arxiv.org/abs/2308.14761) `arXiv:2308.14761` | Northeastern | 闭式编辑交叉注意力，批量擦除/去偏 |
| 2023-05 | [Tree-Ring](https://arxiv.org/abs/2305.20030) `arXiv:2305.20030` | Maryland | 在初始噪声傅里叶空间嵌入水印 |
| 2023-03 | [Stable Signature](https://arxiv.org/abs/2303.15435) `arXiv:2303.15435` | Meta | 把水印嵌入 VAE 解码器 |
| 2023-03 | [Concept Ablation](https://arxiv.org/abs/2303.13516) `arXiv:2303.13516` | CMU | 把目标概念映射到锚点概念实现消融 |
| 2023-03 | [ESD](https://arxiv.org/abs/2303.07345) `arXiv:2303.07345` | Northeastern | 微调擦除指定概念（风格、裸露等） |
| 2023-02 | [Glaze](https://arxiv.org/abs/2302.04222) `arXiv:2302.04222` | UChicago | 对艺术作品加扰动，防风格模仿 |
| 2022-11 | [Safe Latent Diffusion](https://arxiv.org/abs/2211.05105) `arXiv:2211.05105` | TU Darmstadt | 推理时抑制不当内容 |

### 综述/教程（16）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [视频后训练综述](https://arxiv.org/abs/2610.00812) `arXiv:2610.00812` | Arizona State | 视频生成模型的后训练与对齐综述 |
| 2026-09 | 🆕 [Agentic 视觉生成综述](https://arxiv.org/abs/2609.06758) `arXiv:2609.06758` | Fudan | 视觉生成从单次调用走向可规划、可调用工具、可自我修正的 agentic 流程 |
| 2026-07 | 🆕 [掩码扩散 LLM 加速综述](https://arxiv.org/abs/2607.12829) `arXiv:2607.12829` |  | 掩码扩散大语言模型的高效推理技术综述 |
| 2026-06 | 🆕 [World Action Model 综述](https://arxiv.org/abs/2606.20781) `arXiv:2606.20781` | NUS | 用视频生成模型做"预测 + 动作"的 WAM 综述 |
| 2026-04 | 🆕 [视觉生成新范式综述](https://arxiv.org/abs/2604.28185) `arXiv:2604.28185` | LMMs-Lab | 从"原子映射"到 agentic 世界建模的视觉生成演进综述 |
| 2026-03 | 🆕 [视频生成即世界模型综述](https://arxiv.org/abs/2603.28489) `arXiv:2603.28489` |  | 视频生成模型作为世界模型：高效范式、架构与算法 |
| 2025-10 | ★ 🆕 [扩散原理（专著）](https://arxiv.org/abs/2510.21890) `arXiv:2510.21890` | Sony/OpenAI/Stanford | 《The Principles of Diffusion Models》：变分、分数、流三种视角统一讲解，含少步生成 |
| 2025-10 | 🆕 [缓存方法综述](https://arxiv.org/abs/2510.19755) `arXiv:2510.19755` |  | 扩散模型缓存加速方法综述 |
| 2025-08 | [扩散语言模型综述](https://arxiv.org/abs/2508.10875) `arXiv:2508.10875` |  | 扩散语言模型综述 |
| 2025-06 | ★ [MIT 6.S184 讲义](https://arxiv.org/abs/2506.02070) `arXiv:2506.02070` | MIT | "流匹配与扩散模型导论"课程讲义，从 ODE/SDE 讲到实践 |
| 2025-02 | [高效扩散综述](https://arxiv.org/abs/2502.06805) `arXiv:2502.06805` |  | 高效扩散模型综述 |
| 2024-12 | ★ [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) `arXiv:2412.06264` | Meta | 流匹配的系统教程与 PyTorch 代码库，涵盖连续、离散、流形与生成元匹配 |
| 2024-06 | ★ [Step-by-Step Diffusion](https://arxiv.org/abs/2406.08929) `arXiv:2406.08929` | Apple | 面向本科水平的扩散入门教程，从最简单的高斯例子讲起 |
| 2022-09 | [视觉扩散综述](https://arxiv.org/abs/2209.04747) `arXiv:2209.04747` |  | 视觉领域扩散模型综述 |
| 2022-09 | [扩散综述（Yang 等）](https://arxiv.org/abs/2209.00796) `arXiv:2209.00796` | PKU 等 | 方法与应用的全景综述（持续更新） |
| 2022-08 | ★ [Unified Perspective](https://arxiv.org/abs/2208.11970) `arXiv:2208.11970` |  | 从 ELBO、分数、SNR 三个角度统一推导扩散模型，入门推导的常用读物 |

### 产业发布（19）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Genie 4](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/introducing-genie-4/) | Google DeepMind | 实时交互世界模型，自然语言改写场景并原生生成音频；小范围测试 |
| 2026-10 | 🆕 [FLUX 3 Image Klein](https://bfl.ai/blog/flux-3-image-klein) | Black Forest Labs | 5.6B 开放权重图像模型 |
| 2026-09 | 🆕 [FLUX 3 Action](https://bfl.ai/models/flux-3-action) | Black Forest Labs | 7B 世界动作模型，用于机器人控制 |
| 2026-07 | 🆕 [FLUX 3](https://bfl.ai/blog/flux-3) | Black Forest Labs | 图像、视频、音频联合训练的多模态基础模型，视频最长 20 秒带原生音频 |
| 2026-05 | 🆕 [Gemini Omni](https://blog.google/innovation-and-ai/technology/ai/google-io-2026-all-our-announcements/) | Google | I/O 2026 发布，"任意输入生成任意输出"，先从视频开始（Omni Flash） |
| 2026-03 | 🆕 [Sora App 停运](https://the-decoder.com/openai-sets-two-stage-sora-shutdown-with-app-closing-april-2026-and-api-following-in-september/) | OpenAI | OpenAI 宣布关停 Sora App 与 API（分两阶段），转向其他业务重点 |
| 2026-02 | 🆕 [Kling 3.0](https://en.wikipedia.org/wiki/Kling_AI) | Kuaishou | 原生 4K、多镜头与音频的视频模型 |
| 2025-11 | 🆕 [Nano Banana Pro](https://blog.google/innovation-and-ai/products/nano-banana-pro/) | Google DeepMind | 基于 Gemini 3 Pro 的图像生成与编辑（架构未公开），文字渲染与多图一致性突出 |
| 2025-11 | 🆕 [FLUX.2](https://bfl.ai/blog/flux-2) | Black Forest Labs | 生成与编辑一体、多参考图；FLUX.2 [dev] 为 32B 开放权重 |
| 2025-09 | [Sora 2](https://openai.com/index/sora-2/) | OpenAI | 带同步音频、更强物理一致性的视频模型与社交 App |
| 2025-08 | ★ [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) | Google DeepMind | 实时（24 FPS）可交互的通用世界模型，数分钟一致性 |
| 2025-07 | [Wan 2.2](https://github.com/Wan-Video/Wan2.2) | Alibaba | 开源视频模型，按噪声高低划分的 MoE 去噪专家 |
| 2025-05 | ★ [Veo 3 / 3.1](https://deepmind.google/models/veo/) | Google DeepMind | 原生同步音频的视频生成；3.1 于 2025-10 推出 |
| 2025-05 | [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/) | Google DeepMind | 实验性文本扩散模型，强调生成速度 |
| 2025-03 | [GPT-4o 图像生成](https://openai.com/index/introducing-4o-image-generation/) | OpenAI | 原生嵌入 GPT-4o 的图像生成；官方系统卡称其为自回归模型（作为扩散路线的对照） |
| 2024-10 | [SD 3.5](https://stability.ai/news/introducing-stable-diffusion-3-5) | Stability AI | 开源 MMDiT 文生图（Large/Medium） |
| 2024-08 | [FLUX.1](https://bfl.ai/announcements/24-08-01-bfl) | Black Forest Labs | 12B 修正流 Transformer，开源 dev/schnell 版本成为社区主力 |
| 2024-02 | ★ [Sora](https://openai.com/index/video-generation-models-as-world-simulators/) | OpenAI | 时空 patch 上的扩散 Transformer，一分钟视频；提出"视频生成即世界模拟器" |
| 2023-10 | [DALL·E 3](https://cdn.openai.com/papers/dall-e-3.pdf) | OpenAI | 用高质量合成描述重训文生图，提示遵循大幅提升 |

