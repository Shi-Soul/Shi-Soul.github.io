# 深度学习的原理：从基础到 2026 前沿（长文综述）

> 数据截至 2026-10-30。论文库共 1827 条（1776 条 arXiv 已用官方 API 核验标题与日期），其中 2025-11 及之后（最近一年）1015 条、2026-05 及之后（最近半年）577 条，完整列表见附录 B。正文只引用有代表性的工作；附录收录得更全，便于检索复用。数字摘自论文摘要或技术报告，实验设置各不相同，只宜看量级。最近一年的大多数工作是预印本，很多结论只在可解的玩具模型或小规模受控实验中成立。

**怎么组织。** 一个学习系统由四样东西决定：函数族（架构）、目标（学习信号）、数据、优化算法；它学得好不好，可以拆成"能不能表示"、"能不能找到"、"能不能推广"三项误差，再加上"随规模如何变化"。本文按这几个根本问题分章，而不是按模型或年份：

| 章 | 问题 | 内容 |
|---|---|---|
| [1](#ch1) | 深度学习在解什么 | 参数化函数、经验风险、反向传播、学习信号、误差分解与三个谜题 |
| [2](#ch2) | 能表示什么（架构） | 注意力即记忆、线性注意力与状态空间、表达能力、残差流与深度、循环深度、条件计算 |
| [3](#ch3) | 能否找到（优化） | 优化器几何、稳定性边缘、学习率/批大小/权重衰减、初始化与归一化、μP、训练不稳定与低精度 |
| [4](#ch4) | 能否推广（泛化） | 双下降、隐式偏置、特征学习、grokking、记忆、生成模型的泛化、分布外与长度泛化 |
| [5](#ch5) | 为什么越大越好（扩展） | 幂律与算力最优、幂律的来源、数据/精度/稀疏度/优化器、深度与循环、涌现、测试时扩展 |
| [6](#ch6) | 学到了什么（表征） | 线性表征与几何、表征趋同、叠加、稀疏自编码器、电路与内省 |
| [7](#ch7) | 从数据本身学习 | 下一 token 预训练、对比学习、JEPA、扩散与流、一步生成与离散扩散 |
| [8](#ch8) | 从反馈中学习 | RLHF 与 DPO、RLVR、在策略蒸馏、后训练改变了什么 |
| [9](#ch9) | 测试时的计算与学习 | 上下文学习、思维链与推理、持续学习与可塑性 |
| [10](#ch10) | 应用：原理的检验场 | 大模型系统、视觉/多模态/科学/机器人/智能体、安全与对齐 |
| [11](#ch11) | 开放问题 | 按章整理 |

每一小节先讲已经确立的原理，再用"最近一年"小标题写 2025-11 之后的进展；最近半年（2026-05 之后）的工作在附录 A 中按月汇总。大模型在本文中是一个重要的应用方向（第 10 章），而不是组织全文的主线。

## 1 问题：深度学习在解什么

### 1.1 参数化函数与表示

深度网络是一串可微变换的复合 $f_\theta=f_L\circ\cdots\circ f_1$。最朴素的一层是 $h_l=\sigma(W_lh_{l-1}+b_l)$；现代网络的基本单元是带残差的块：

$$h_{l}=h_{l-1}+F_l\bigl(\mathrm{Norm}(h_{l-1});\theta_l\bigr)$$

其中 $F_l$ 是注意力、MLP、卷积或状态空间层。[通用逼近定理](https://link.springer.com/article/10.1007/BF02551274)保证单隐层网络就能逼近任意连续函数，所以"能不能表示"很少是瓶颈；深度的价值在于用远少于浅层网络的参数表示分层、组合式的结构，而这恰好与自然数据（图像、语言、物理）的组合结构相匹配（[几何深度学习](https://arxiv.org/abs/2104.13478)从对称性角度系统阐述了这一点）。

### 1.2 经验风险与交叉熵

监督学习最小化经验风险 $\hat{\mathcal L}(\theta)=\frac1N\sum_i\ell(f_\theta(x_i),y_i)$。语言模型把序列的联合概率分解为条件概率之积，最小化下一个 token 的交叉熵：

$$\mathcal L(\theta)=-\mathbb E_{x\sim\mathcal D}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

这个目标等价于最小化数据分布与模型分布之间的 KL 散度，也等价于用模型做无损压缩时的平均码长。它是"自监督"的：数据本身就是标签，所以可以用几乎无限的网页文本训练。

### 1.3 反向传播与自动微分

对复合函数，链式法则给出

$$\frac{\partial\mathcal L}{\partial\theta_l}=\Bigl(\frac{\partial h_l}{\partial\theta_l}\Bigr)^{\!\top}\frac{\partial\mathcal L}{\partial h_l},\qquad \frac{\partial\mathcal L}{\partial h_{l-1}}=\Bigl(\frac{\partial h_l}{\partial h_{l-1}}\Bigr)^{\!\top}\frac{\partial\mathcal L}{\partial h_l}$$

反向模式自动微分只需一次前向、一次反向，代价与前向同阶，与参数个数无关（[Rumelhart 等 1986](https://www.nature.com/articles/323533a0)）。残差连接让 $\partial h_l/\partial h_{l-1}=I+\partial F_l/\partial h_{l-1}$，梯度可以沿恒等路径无衰减地传到浅层，这是深网络可训的关键（[ResNet](https://arxiv.org/abs/1512.03385)、[恒等映射](https://arxiv.org/abs/1603.05027)）。

### 1.4 五种学习信号

| 信号 | 目标 | 监督来源 | 代表 |
|---|---|---|---|
| 监督 | 标签交叉熵 | 人工标注 | [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)、[ResNet](https://arxiv.org/abs/1512.03385) |
| 自监督：预测 | 下一 token、掩码重建 | 数据本身 | [GPT-3](https://arxiv.org/abs/2005.14165)、[BERT](https://arxiv.org/abs/1810.04805)、[MAE](https://arxiv.org/abs/2111.06377) |
| 自监督：联合嵌入 | 对比 / 在表征空间预测 | 增强、多视图、图文对 | [SimCLR](https://arxiv.org/abs/2002.05709)、[CLIP](https://arxiv.org/abs/2103.00020)、[DINOv3](https://arxiv.org/abs/2508.10104) |
| 生成式去噪 | 回归噪声或速度场 | 人为加噪 | [DDPM](https://arxiv.org/abs/2006.11239)、[Flow Matching](https://arxiv.org/abs/2210.02747) |
| 强化学习 | 最大化期望奖励 | 环境、人类偏好、验证器 | [AlphaGo](https://www.nature.com/articles/nature16961)、[InstructGPT](https://arxiv.org/abs/2203.02155)、[R1](https://arxiv.org/abs/2501.12948) |

今天的大模型通常把它们串成一条流水线：自监督预训练 → 中期训练 → 指令微调 → 偏好对齐 / RLVR → 在策略蒸馏。

### 1.5 误差分解与三个谜题

设 $\mathcal F$ 为网络能表示的函数族，$f^*$ 为最优预测器，$f_{\mathcal F}$ 为族内最好的函数，$\hat f$ 为经验风险最小化的解，$\tilde f$ 为优化实际得到的解。超额风险可以拆成三项：

$$
\mathcal R(\tilde f)-\mathcal R(f^*)=\underbrace{\mathcal R(f_{\mathcal F})-\mathcal R(f^*)}_{\text{逼近（第 2 章）}}+\underbrace{\mathcal R(\hat f)-\mathcal R(f_{\mathcal F})}_{\text{估计 / 泛化（第 4 章）}}+\underbrace{\mathcal R(\tilde f)-\mathcal R(\hat f)}_{\text{优化（第 3 章）}} .
$$

经典理论认为三者此消彼长：族越大，逼近误差越小，但估计误差越大，优化也越难。深度学习在三处都违背了这一直觉，构成了理论的三个谜题：

1. **优化之谜。** 损失高度非凸，梯度方法却稳定地找到好解，甚至在学习率超过经典稳定条件时仍在下降（第 3 章）。
2. **泛化之谜。** 参数远多于样本、能拟合随机标签，测试误差却随规模继续下降（第 4 章）。
3. **扩展之谜。** 损失随参数、数据、算力呈跨越多个数量级的幂律，可以在训练前预测（第 5 章）。

第 6 章讨论"学到了什么"，第 7–9 章讨论"学什么目标、何时学"：从数据本身、从反馈、在测试时。最近一年的特点是：这几个谜题都出现了可以计算、可以证伪的解释，通常来自可解的玩具模型。

## 2 架构：归纳偏置、记忆与表达能力

架构决定函数族与归纳偏置。从 MLP 到 CNN 再到 Transformer，趋势是把更少的先验写死、把更多结构交给数据；但"怎么在序列与深度两个方向上传递信息"仍是架构设计的核心。一个统一的视角是：**沿序列方向的混合器（注意力、线性注意力、状态空间模型）都是联想记忆的读写规则，沿深度方向的残差流则是另一条可以设计的信息通道。**

### 2.1 注意力：按内容寻址的记忆

缩放点积注意力（[Vaswani 等](https://arxiv.org/abs/1706.03762)）：

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_k}}+M\Bigr)V$$

$M$ 是因果掩码。多头注意力把 $d$ 维投影成 $h$ 个子空间并行计算。它可以看作一次"按内容寻址"的检索：查询与所有键比较相似度，按权重读取值。[Ramsauer 等](https://arxiv.org/abs/2008.02217)证明其更新规则就是现代 Hopfield 网络的检索。

### 2.2 位置编码

[RoPE](https://arxiv.org/abs/2104.09864) 把位置 $m$ 的二维子向量旋转 $m\theta_i$ 角：

$$\langle R_m q,\ R_n k\rangle=\langle q,\ R_{n-m}k\rangle,$$

于是注意力分数只依赖相对位置。上下文扩展通过调整旋转频率实现（[位置插值](https://arxiv.org/abs/2306.15595)、[YaRN](https://arxiv.org/abs/2309.00071)、[LongRoPE](https://arxiv.org/abs/2402.13753)）。最近的改进包括傅里叶位置编码 [FoPE](https://arxiv.org/abs/2412.17739)、对 RoPE 频率"去噪"的 [DoPE](https://arxiv.org/abs/2511.09146)、多维推广 [STRING](https://arxiv.org/abs/2502.02562) 与视频 [VideoRoPE](https://arxiv.org/abs/2502.05173)。

### 2.3 改造 softmax：门控、汇聚与巨大激活

[差分注意力](https://arxiv.org/abs/2410.05258)用两组 softmax 相减抵消共模噪声；[遗忘 Transformer](https://arxiv.org/abs/2503.02130)给 softmax 注意力加上数据相关的遗忘门；[Softpick](https://arxiv.org/abs/2504.20966) 用整流 softmax 去掉汇聚点；[门控注意力](https://arxiv.org/abs/2505.06708)在输出端加 sigmoid 门。这些改动的共同点是：允许一个头"什么都不读"，不必把注意力强行分配到汇聚点上。

[Sun 等](https://arxiv.org/abs/2402.17762)发现 LLM 中少数维度取值极大（巨大激活），它们充当固定偏置，并与"注意力汇聚点"相关——大量注意力落在首 token 等无意义位置（[StreamingLLM](https://arxiv.org/abs/2309.17453)）。[Barbero 等](https://arxiv.org/abs/2504.02732)把汇聚点解释为防止表征过度混合的机制。

**门控注意力。** [Qiu 等](https://arxiv.org/abs/2505.06708)在 30 种变体、3.5T token 的系统实验中发现，在缩放点积注意力的输出后加一个逐头 sigmoid 门，就能引入非线性与稀疏性、消除注意力汇聚点与巨大激活，并提升稳定性与长度外推（NeurIPS 2025 最佳论文）。

#### 最近一年（2025-11 → 2026-10）

**汇聚为什么必要。** softmax 注意力的输出是值向量的凸组合 $o=\sum_j \mathrm{softmax}_j(q^\top k_j)\,v_j$。若某个头在某些上下文中应当"什么都不做"（$o\approx0$），而权重之和必须为 1，就只能把权重集中到一个值向量接近 0 的锚点上。对一类"触发条件"任务，可以证明这种汇聚是必要的（[汇聚的必要性](https://arxiv.org/abs/2603.11487)）。门控注意力、softpick 等之所以能消除汇聚，是因为它们打破了"权重和为 1"或允许整体输出被关掉。

**巨大激活。** 它与汇聚常同时出现，但二者的共现主要是现代 Transformer 设计（如 pre-norm）的产物，功能相关而不同（[尖峰、稀疏与汇聚](https://arxiv.org/abs/2603.05498)）。训练轨迹显示：承载汇聚的通道随种子不同但很早稳定，之后在权重衰减驱动下增长并集中到少数冗余载体（[生命周期](https://arxiv.org/abs/2610.00423)）；在受控的残差实验中它们难以通过架构消除（[架构稳健](https://arxiv.org/abs/2606.20743)）。

### 2.4 线性注意力与状态空间：在线学习的记忆

去掉 softmax、用核特征 $\phi$ 近似相似度后，因果注意力可写成递归（[Katharopoulos 等](https://arxiv.org/abs/2006.16236)）：

$$S_t=S_{t-1}+v_t\,\phi(k_t)^\top,\qquad o_t=S_t\,\phi(q_t).$$

状态 $S_t\in\mathbb R^{d_v\times d_k}$ 大小固定，推理时每步 $O(1)$。问题是它只会"加"不会"忘"，有限的状态很快被写满。

两种修补互补：

- 门控（衰减）：$S_t=\alpha_tS_{t-1}+v_tk_t^\top$，$\alpha_t$ 由输入决定，可以整体遗忘（[RetNet](https://arxiv.org/abs/2307.08621)、[GLA](https://arxiv.org/abs/2312.06635)、[Mamba](https://arxiv.org/abs/2312.00752)）。
- Delta 规则（改写）：把状态看作一个联想记忆，每步对损失 $\frac12\lVert Sk_t-v_t\rVert^2$ 做一步梯度下降：

$$S_t=S_{t-1}\bigl(I-\beta_tk_tk_t^\top\bigr)+\beta_tv_tk_t^\top ,$$

可以精确地替换键 $k_t$ 处的旧值（[DeltaNet](https://arxiv.org/abs/2406.06484)）。

[Gated DeltaNet](https://arxiv.org/abs/2412.06464) 把两者结合：$S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$。Moonshot 的 [KDA](https://arxiv.org/abs/2510.26692) 把标量门 $\alpha_t$ 换成逐通道的对角门，并设计了专用的 DPLR 分块算法；以 3:1 与 MLA 混合的 Kimi Linear（48B 总参、3B 激活）首次在公平比较中于短上下文、长上下文与 RL 扩展三种设置下全面超过全注意力，KV 缓存最多减少 75%，百万上下文的解码吞吐提升约 6 倍。

结构化 SSM 由连续系统 $\dot h=Ah+Bx$, $y=Ch$ 离散化而来，可以按递归或卷积两种方式计算（[S4](https://arxiv.org/abs/2111.00396)）。[Mamba](https://arxiv.org/abs/2312.00752) 让 $B,C$ 与步长依赖输入（"选择性"），用硬件感知的并行扫描实现，在语言上首次以线性复杂度匹敌 Transformer。[Mamba-2](https://arxiv.org/abs/2405.21060) 证明：当 $A$ 是标量乘单位阵时，SSM 与带特定衰减掩码的线性注意力是同一族半可分矩阵（状态空间对偶），从而可以用矩阵乘法高效训练。

delta 规则其实是"每读一个 token，就在一个线性模型上做一步 SGD"。沿这一视角：[TTT 层](https://arxiv.org/abs/2407.04620)把隐状态换成一个 MLP，在测试时用自监督损失更新；[Titans](https://arxiv.org/abs/2501.00663) 用"惊奇度"（梯度大小）与动量决定写入，并配合遗忘门；[Miras](https://arxiv.org/abs/2504.13173) 把 Transformer、线性 RNN 与 Titans 统一为"注意力偏置 + 保留门 + 在线优化算法"的联想记忆设计空间；[LaCT](https://arxiv.org/abs/2505.23884) 用数千 token 的大块更新快权重，使测试时训练的硬件利用率接近注意力。Google 的 [Nested Learning](https://arxiv.org/abs/2512.24695) 更进一步：把整个模型看作一组嵌套的、更新频率不同的优化问题（梯度下降本身也是一种联想记忆），并据此设计自修改的 HOPE 架构与连续记忆系统。

#### 最近一年（2025-11 → 2026-10）

线性注意力维护一个矩阵记忆 $S_t=S_{t-1}+v_tk_t^\top$，输出 $o_t=S_tq_t$。delta 规则把写入改成对 $\tfrac12\|Sk_t-v_t\|^2$ 做一步梯度下降：

$$
S_t=S_{t-1}-\beta_t\,(S_{t-1}k_t-v_t)\,k_t^\top=S_{t-1}\big(I-\beta_t k_tk_t^\top\big)+\beta_t v_tk_t^\top ,
$$

门控版本再乘衰减 $\alpha_t$。最近一年的几个推进都可以在这个式子上理解：

**精确积分。** 上式是连续时间流 $\dot S=-(Sk-v)k^\top$ 的显式 Euler 步。记 $\hat k=k/\|k\|$，与 $\hat k$ 正交的部分不变，沿 $\hat k$ 的分量 $u=S\hat k$ 满足 $\dot u=-\|k\|(\|k\|u-v)$，可精确求解，得到

$$
S(\beta)=S_{t-1}-\big(1-e^{-\beta\|k\|^2}\big)\Big(S_{t-1}\hat k-\frac{v}{\|k\|}\Big)\hat k^\top .
$$

Euler 步中的系数 $\beta\|k\|^2$ 被换成始终落在 $(0,1)$ 内的 $1-e^{-\beta\|k\|^2}$，数值上更稳定，且保持秩一结构与分块并行（[EFLA](https://arxiv.org/abs/2512.12602)）。

**擦除与写入解耦。** 上式用同一个 $\beta_t$ 同时控制"沿 $k_t$ 擦掉多少旧值"和"写入多少新值"；把二者拆成两个门，编辑压缩记忆时更少打乱已有联想（[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)）。按卡尔曼滤波的不确定性决定写入强度是另一种做法（[Kalman Delta](https://arxiv.org/abs/2609.07816)）；加入可并行的动量（[MDN](https://arxiv.org/abs/2605.05838)）、把记忆扩展为三维张量（[三元线性注意力](https://arxiv.org/abs/2609.36529)）则对应更好的在线优化器与更大的状态。

**TTT 即线性注意力。** 以"测试时记忆键值映射"解释的 TTT 层，有多种现象与记忆解释矛盾；一大类 TTT 架构可以写成学到的线性注意力算子（[TTT 即线性注意力](https://arxiv.org/abs/2602.21204)）。

### 2.5 表达能力：状态追踪与混合架构

Transformer 与对角线性 RNN 都在电路复杂度类 TC⁰ 中，难以做"状态追踪"（如长序列的奇偶校验、置换复合）。[Grazzi 等](https://arxiv.org/abs/2411.12537)证明：只要允许状态转移矩阵有负特征值（例如把 delta 规则的 $\beta$ 范围扩到 $[0,2]$），线性 RNN 就能解决奇偶校验；[RWKV-7](https://arxiv.org/abs/2503.14456) 用广义 delta 规则可识别所有正则语言。[对角 SSM 的表达极限](https://arxiv.org/abs/2603.01959)与 [PAR](https://arxiv.org/abs/2610.27628) 从不同角度继续刻画这一边界。

纯线性模型在"从长上下文中精确检索"上仍弱于注意力，于是混合成为共识：[Jamba](https://arxiv.org/abs/2403.19887)（Mamba + 注意力 + MoE）、[MiniMax-01](https://arxiv.org/abs/2501.08313)（闪电注意力 : softmax = 7 : 1）、[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)（Gated DeltaNet : 门控注意力 = 3 : 1）、[Kimi Linear](https://arxiv.org/abs/2510.26692)（KDA : MLA = 3 : 1）、[Nemotron 3](https://arxiv.org/abs/2512.20856)（Mamba-Transformer MoE），直到 2.8T 的 [Kimi K3](https://arxiv.org/abs/2607.24653)。另一种思路是改造注意力本身：[Log-Linear Attention](https://arxiv.org/abs/2506.04761) 让隐状态数随长度对数增长；["滑窗胜过线性注意力"](https://arxiv.org/abs/2608.28444)提醒混合层的选择仍有争议。

#### 最近一年（2025-11 → 2026-10）

**表达力与纠错。** 混合线性—全注意力存在可证明的表达力层级（[表达力层级](https://arxiv.org/abs/2602.01763)）。对仿射循环 $h_t=A(x_t)h_{t-1}+b(x_t)$，若 $A$ 在区分不同状态的子空间上保持状态表示（不收缩），那么沿这些方向的误差也永远不会被纠正，所以实际学到的是有限视野的解（[误差控制](https://arxiv.org/abs/2605.07755)）；学到的也可能是陪集而非状态本身（[状态还是陪集](https://arxiv.org/abs/2609.29951)）。

### 2.6 深度方向：残差流、超连接与注意力残差

**超连接。** [HC](https://arxiv.org/abs/2409.19606) 把残差流扩展为 $n$ 条，层与流之间用可学习的矩阵混合，但破坏了恒等映射、带来不稳定。DeepSeek 的 [mHC](https://arxiv.org/abs/2512.24880) 把混合矩阵约束到双随机矩阵（Birkhoff 多面体）上，使其乘积仍是"凸组合"，恢复了恒等映射的性质，并用于 DeepSeek-V4。

**注意力残差。** Moonshot 的 [AttnRes](https://arxiv.org/abs/2603.15031) 认为 Pre-LN 残差以固定单位权重累加所有层输出，会导致隐状态随深度失控增长、每层贡献被稀释；改为让每层用 softmax 注意力在之前各层的输出上按内容加权求和，并用分块版本控制显存与通信。它是 [Kimi K3](https://arxiv.org/abs/2607.24653) 的核心组件之一。

#### 最近一年（2025-11 → 2026-10）

**残差的两个方向。** 把位置固定、以层号为"时间"，残差流就是深度方向上的序列；自注意力在序列方向做自适应混合，残差却只做固定加法。由此得到一个对偶：深度方向的残差设计与序列方向的混合器一一对应（[残差流对偶](https://arxiv.org/abs/2603.16039)）。超连接/mHC、注意力残差（沿深度做 softmax 读取）都是在深度方向引入更强的"混合器"；最近的变体包括 Kronecker 参数化的 [KromHC](https://arxiv.org/abs/2601.21579)、多头的[注意力残差](https://arxiv.org/abs/2607.27230)、读取逐层增量的 [Delta AttnRes](https://arxiv.org/abs/2605.18855)，以及注意力残差中同样出现的汇聚（[AttnRes 中的汇聚](https://arxiv.org/abs/2605.17887)）。Post-LN 在深层下的复活（[Post-LN 回归](https://arxiv.org/abs/2601.19895)）则从另一端说明：归一化放在哪里，本质是深度方向信号传播的问题。

### 2.7 循环深度

另一条扩展测试时计算的路是重复使用同一组层。[Geiping 等](https://arxiv.org/abs/2502.05171)训练的循环深度模型在测试时迭代更多次即可提升推理；[HRM](https://arxiv.org/abs/2506.21734) 用高低两个时间尺度的循环模块，27M 参数即在 ARC、数独上表现突出；[TRM](https://arxiv.org/abs/2510.04871) 把它简化为 2 层、7M 参数的递归网络，在 ARC-AGI-1 上达到 45%；[Ouro](https://arxiv.org/abs/2510.25741) 把循环推理放进预训练，1.4B/2.6B 循环模型匹敌 12B 级稠密模型；[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) 按 token 路由不同的循环深度。2026 年，[SMELT](https://arxiv.org/abs/2609.01343) 给出算力匹配下循环 MoE 的扩展律。

#### 最近一年（2025-11 → 2026-10）

循环模型反复应用同一个块 $h^{(\ell+1)}=F_\theta(h^{(\ell)},x)$。[TRM](https://arxiv.org/abs/2510.04871)、[Ouro](https://arxiv.org/abs/2510.25741) 之后，最近一年形成了系统的研究：

- **值多少参数。** 等深度扩展律给出 $\varphi\approx0.46$（5.4 节）；稳定训练的扩展律（[Parcae](https://arxiv.org/abs/2604.12946)）；循环 MoE 的联合扩展律（[循环 MoE](https://arxiv.org/abs/2609.40316)）。
- **怎么参数化。** 同一参数在一次前向中被使用 $r$ 次，残差分支的更新会累积，μP 式的残差缩放需要相应修改，才能让超参数跨规模迁移（[DeepLoop](https://arxiv.org/abs/2607.13491)、[Loop-μP](https://arxiv.org/abs/2606.18524)）。
- **不动点。** 若 $F_\theta$ 在 $x$ 固定时是压缩映射，迭代收敛到不动点 $h^*=F_\theta(h^*,x)$，梯度可由隐函数定理得到，与到达路径无关。因此越接近不动点，截断反传的偏差越小，末端 KV 也可共享（[不动点视角](https://arxiv.org/abs/2610.06833)、[截断 BPTT](https://arxiv.org/abs/2610.26030)）。循环位置、输入注入、状态组织三条设计轴的受控比较见 [做对了（一）](https://arxiv.org/abs/2610.23036)。
- **风险。** 循环次数远超训练时，噪声更新会覆盖已经正确的中间结论（[上万次迭代](https://arxiv.org/abs/2610.11570)）；安全行为也随循环次数变化（[循环与拒绝](https://arxiv.org/abs/2610.15218)）。

### 2.8 条件计算：混合专家与条件记忆

**路由。** MoE 层有 $E$ 个专家 FFN 与一个路由器。对 token 表示 $x$，计算门控分数 $s=\mathrm{softmax}(W_rx)$（或 sigmoid），取 top-$k$：

$$y=\sum_{i\in\mathrm{TopK}(s,k)}g_i\,\mathrm{FFN}_i(x).$$

总参数随 $E$ 增长，每 token 的计算只随 $k$ 增长（[Shazeer 等](https://arxiv.org/abs/1701.06538)、[GShard](https://arxiv.org/abs/2006.16668)、[Switch](https://arxiv.org/abs/2101.03961)）。

**负载均衡与专门化。** 路由容易坍缩到少数专家。经典做法是加辅助损失 $\mathcal L_{\text{bal}}=\alpha E\sum_i f_iP_i$（$f_i$ 为实际分到专家 $i$ 的 token 比例，$P_i$ 为平均路由概率）；[ST-MoE](https://arxiv.org/abs/2202.08906) 加 router z-loss 稳定 logit。[无辅助损失均衡](https://arxiv.org/abs/2408.15664)改为给每个专家加一个只用于选择、不参与加权的偏置项，按负载动态调整，避免辅助损失干扰主目标（DeepSeek-V3 采用）。[Qwen](https://arxiv.org/abs/2501.11873) 发现均衡损失应在全局批次而非微批次上计算，专家专门化程度显著提升。

**细粒度与超稀疏。** [DeepSeekMoE](https://arxiv.org/abs/2401.06066) 把专家切得更细、增加激活数量以提高组合灵活性，并设置始终激活的共享专家吸收通用知识。随后模型越来越稀疏：激活比例从 [Mixtral](https://arxiv.org/abs/2401.04088) 的约 28% 降到 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 的 5.5%、[Kimi K2](https://arxiv.org/abs/2507.20534) 的约 3%；[LongCat-Flash](https://arxiv.org/abs/2509.01322) 引入"零计算专家"让每 token 的激活量随难度动态变化；[Kimi K3](https://arxiv.org/abs/2607.24653) 的 Stable LatentMoE 从 896 个专家中选 16 个。

**条件记忆：知识应该'查'而不是'算'。** DeepSeek 的 [Engram](https://arxiv.org/abs/2601.07372) 提出：MoE 用条件计算扩展容量，但 Transformer 缺少"查知识"的原语，只能用计算模拟检索。Engram 把经典的 n-gram 嵌入现代化为 O(1) 查表的条件记忆模块，与 MoE 互补。把总参数在两者之间分配时出现 U 形扩展律；扩展到 27B 记忆参数后，在严格等参数、等 FLOPs 下胜过纯 MoE，且推理与代码的提升甚至大于知识类任务。同期的相关证据：[Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975) 显示输入词表越大越好，[扩展嵌入](https://arxiv.org/abs/2601.21204)在一些区间优于扩展专家，[Qwen3.8-Next](https://arxiv.org/abs/2608.30320) 把 51B 的 n-gram 表放到加速器之外，[记忆层](https://arxiv.org/abs/2412.09764)与 [UltraMemV2](https://arxiv.org/abs/2508.18756) 用可训练的大规模键值表补充事实性知识。

## 3 优化：为什么训练得动

深度网络的损失非凸、高维、病态，而且训练时常在经典稳定条件之外运行。这一章从优化器出发（它们都可以看成某个范数下的最速下降），再讨论训练动力学（稳定性边缘）、超参数（学习率、批大小、权重衰减）、让深网络可训练的初始化、归一化与参数化，以及大模型特有的不稳定与低精度问题。

### 3.1 从 SGD 到 AdamW

小批量随机梯度下降 $\theta_{k+1}=\theta_k-\eta\,\hat g_k$ 是一切的起点；动量 $m_k=\mu m_{k-1}+\hat g_k$ 平滑噪声。Adam（[Kingma 与 Ba](https://arxiv.org/abs/1412.6980)）为每个坐标维护一阶、二阶矩的指数平均：

$$m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,\quad \theta_{k+1}=\theta_k-\eta\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}$$

[AdamW](https://arxiv.org/abs/1711.05101) 把权重衰减从梯度中解耦（$\theta\leftarrow\theta-\eta\lambda\theta$），成为十年来大模型的默认优化器。[Wang 与 Aitchison](https://arxiv.org/abs/2405.13698) 指出 AdamW 的权重本质上是更新的指数滑动平均，时间尺度 $1/(\eta\lambda)$ 应随数据量调整；[Power Lines](https://arxiv.org/abs/2505.13738) 进一步给出最优权重衰减与批大小随 token/参数比的幂律。

### 3.2 优化器即最速下降：从 Adam 到 Muon

一个统一视角（[Bernstein 与 Newhouse](https://arxiv.org/abs/2409.20325)、[模块对偶](https://arxiv.org/abs/2410.21265)）：在某个范数约束下，让一阶近似的损失下降最多，

$$\Delta^\star=\arg\min_{\lVert\Delta\rVert\le\eta}\ \langle G,\Delta\rangle .$$

- 欧氏范数：$\Delta^\star=-\eta\,G/\lVert G\rVert_F$，即归一化 SGD。
- 逐坐标 $\ell_\infty$ 范数：$\Delta^\star=-\eta\,\mathrm{sign}(G)$，即 signSGD；Adam 在 $\beta_1=\beta_2=0$ 时退化为它。
- 矩阵谱范数 $\lVert\Delta\rVert_{2\to2}$：设 $G=U\Sigma V^\top$，由于 $\langle G,\Delta\rangle=\mathrm{tr}(\Sigma\,U^\top\Delta V)$ 且 $\lVert U^\top\Delta V\rVert_{2\to2}\le\eta$，最优解是 $U^\top\Delta V=-\eta I$，即

$$\Delta^\star=-\eta\,UV^\top .$$

这就是 [Muon](https://kellerjordan.github.io/posts/muon/)：对动量矩阵做正交化，把所有奇异值"拉平"为 1。直接做 SVD 太贵，实践中用五次多项式的 Newton–Schulz 迭代 $X\leftarrow aX+(bXX^\top+c(XX^\top)^2)X$，几步只用矩阵乘法即可近似。Shampoo（[Gupta 等](https://arxiv.org/abs/1802.09568)）的更新 $L^{-1/4}GR^{-1/4}$ 在不做指数平均时与 Muon 方向相同；[SOAP](https://arxiv.org/abs/2409.11321) 在 Shampoo 的特征基里运行 Adam。

为什么用谱范数？权重矩阵的作用是把输入向量映射成输出向量，用"对任意输入，输出最多改变多少"的算子范数来度量更新大小，比逐元素度量更符合它的功能。机制上，[Wang 等](https://arxiv.org/abs/2509.26030)发现正交化让很少被激活的方向（长尾特征、长尾联想记忆）也获得同样幅度的学习，这是 Muon 的主要优势来源。

**2025 年：Muon 的规模化与冷静的再评估。**

- [Moonlight](https://arxiv.org/abs/2502.16982)：加入权重衰减、按 $0.2\sqrt{\max(m,n)}$ 缩放使更新的 RMS 与 AdamW 对齐后，Muon 可直接沿用 AdamW 的超参数，算力最优训练下约有 2 倍计算效率。[Essential AI](https://arxiv.org/abs/2505.02222) 显示 Muon 在大批量下保持数据效率，扩展了算力-时间的帕累托前沿。
- [Kimi K2](https://arxiv.org/abs/2507.20534)：Muon 在大模型上会让注意力 logit 失控增长；MuonClip 在 logit 过大时按头重缩放 $W_q,W_k$（QK-Clip），实现了 1T 参数、15.5T token 的零损失尖峰预训练。[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（1.6T）也采用 Muon。
- 分布式与改进：[Dion](https://arxiv.org/abs/2504.05295) 用通信高效的低秩正交化替代完整的 Newton–Schulz；[NorMuon](https://arxiv.org/abs/2510.05491) 在正交化后加入逐神经元归一化；[Canzona](https://arxiv.org/abs/2602.06079) 为矩阵优化器提供异步、负载均衡的分布式实现。
- 再评估：[Wen 等](https://arxiv.org/abs/2509.02046)在 0.1B–1.2B 与 1–8 倍 Chinchilla 数据比下严格调参比较十种优化器，发现宣称 1.4–2 倍的加速实际只有 1.1–1.4 倍，且随规模递减；矩阵型优化器（Muon、SOAP）最稳定。[EPFL 的基准](https://arxiv.org/abs/2509.01440)得出类似结论。

#### 最近一年（2025-11 → 2026-10）

上文把优化器写成范数 $\|\cdot\|$ 下的最速下降：$\Delta\theta=-\eta\,\arg\max_{\|d\|\le 1}\langle g,d\rangle$，ℓ2 给出 SGD，ℓ∞ 给出符号下降（Adam 在 $\beta_1=\beta_2$、去掉 $\epsilon$ 时的极限），矩阵谱范数给出 Muon 的 $UV^\top$。最近一年，这个视角开始给出可检验的预测。

**隐式偏置。** 对 $L$ 阶齐次网络 $f(\theta;x)$ 定义归一化间隔 $\gamma(\theta)=\min_i y_i f(\theta;x_i)/\|\theta\|^{L}$。在衰减学习率下，带动量的最速下降轨迹近似于相应范数下的最速下降，因此方向上收敛到

$$
\max_{\theta}\ \min_i\ y_i f(\theta;x_i)\quad \text{s.t.}\quad \|\theta\|\le 1
$$

的 KKT 点：MomentumGD 对应 ℓ2、Signum 与 Adam 对应 ℓ∞、Muon 对应谱范数（[Adam 与 Muon 的隐式偏置](https://arxiv.org/abs/2602.16340)）。换句话说，选择优化器就是在选择"哪种意义下的最大间隔"。某些条件下 Muon 会失去谱偏置，退回 Frobenius 式的解（[失去谱偏置](https://arxiv.org/abs/2610.20112)）。

**规范对称。** 低秩分解 $W=UV^\top$ 的损失对 $(U,V)\mapsto(UQ,VQ)$（$Q$ 正交）不变。梯度流利用这一对称走向低秩解；只有与之"规范等变"的优化器才能继承这一机制：GD、动量、共享标量的 Adam、Muon、Shampoo 满足，逐坐标的 Adam、RMSProp 不满足（[损失不看基，Adam 看](https://arxiv.org/abs/2608.05136)）。这给"逐坐标自适应会引入基依赖"提供了精确表述。

**Muon 为什么好。** 在线性联想记忆 $W\approx\sum_{i} v_i k_i^\top$（事实回忆的可解模型）中，一步 Muon 的恢复率随存储条数的标度严格优于 SGD，接近 Newton 法（[容量标度](https://arxiv.org/abs/2603.26554)、[动力学与扩展律](https://arxiv.org/abs/2602.05725)）。直观上，SGD 的更新被高频（常见）联想的大奇异值主导，正交化把各方向拉平，长尾联想也得到同样的步长。在真实 Transformer 上，沿训练轨迹逐奇异方向测量留出集上的最优步长，得到的"谱剖面"各向异性但稳定，Muon 恰好更接近它（[谱分配](https://arxiv.org/abs/2608.25990)）。收敛性在计入 Newton–Schulz 近似与动量、权重衰减等实际组件后得到证明（[NS 收敛](https://arxiv.org/abs/2601.19156)、[实用 Muon](https://arxiv.org/abs/2609.33152)）。

**边界。** 严格调参下新优化器的加速有限（[Wen 等](https://arxiv.org/abs/2509.02046)），而矩阵预条件方法只有配合正确的超参数迁移才能跨规模保持增益（[超参数迁移](https://arxiv.org/abs/2512.05620)）；Adam 预训练的模型直接换 Muon 微调会失配（[优化器失配](https://arxiv.org/abs/2605.10468)）。

### 3.3 稳定性边缘与损失地形

经典优化理论要求学习率 $\eta<2/\lambda_{\max}$（$\lambda_{\max}$ 为 Hessian 最大特征值，即"锐度"）。[Cohen 等](https://arxiv.org/abs/2103.00065)发现全批量梯度下降会经历"渐进锐化"，把锐度推到 $2/\eta$ 后停在那里振荡，损失却非单调地继续下降——稳定性边缘（EoS）。[Central Flows](https://arxiv.org/abs/2410.24206) 用"中心流"刻画振荡优化器的时间平均轨迹，解释了自适应优化器的隐式正则化。2026 年的工作继续讨论 EoS 的[起源](https://arxiv.org/abs/2604.20446)、[对泛化的影响](https://arxiv.org/abs/2604.19740)与 [Muon 越过 EoS](https://arxiv.org/abs/2610.24305) 的现象。

平坦极小值与泛化的关系有长期争论：[Keskar 等](https://arxiv.org/abs/1609.04836)观察到大批量训练趋向尖锐极小值、泛化变差，[SAM](https://arxiv.org/abs/2010.01412) 显式寻找平坦区域；但锐度依赖参数化方式，单靠它无法完整解释泛化。

#### 最近一年（2025-11 → 2026-10）

对二次模型，梯度下降稳定当且仅当 $\eta\lambda_{\max}(H)<2$。实际训练中锐度 $S(\theta)=\lambda_{\max}(\nabla^2L)$ 会上升到 $2/\eta$ 并停在那里（稳定性边缘，EoS），损失非单调但长期下降。自稳定理论说明，GD 在 EoS 上近似执行约束优化

$$
\min_\theta L(\theta)\quad \text{s.t.}\quad S(\theta)\le 2/\eta .
$$

最近一年把它变成了可数值积分的连续时间模型：[Rod Flow](https://arxiv.org/abs/2602.01480) 与 [Edge Flow](https://arxiv.org/abs/2606.18080)（三个耦合 ODE）能预测振荡幅度与损失下降，并推广到 Adam（[Adam 的 Rod Flow](https://arxiv.org/abs/2605.06821)、[Adam 的 EoS](https://arxiv.org/abs/2608.20638)）。三个推广尤其重要：

- **非欧几何。** 对范数 $\|\cdot\|$ 下的最速下降，相关的"锐度"变为 $\max_{\|d\|\le1} d^\top H d$；符号下降、谱下降同样运行在相应边缘上，而且可能有多条边缘（[非欧 EoS](https://arxiv.org/abs/2603.05002)、[多条边缘](https://arxiv.org/abs/2610.15223)）。
- **随机性。** 小批量 SGD 的锐度稳定在 $2/\eta$ 之下，间隙随批量变小而变大，现在有了理论解释（[随机锐度间隙](https://arxiv.org/abs/2604.21016)）。
- **Muon 与大模型。** GD 中"损失中性、更新反转、边际稳定"三个条件在同一边缘重合，Muon 把它们拆开（[Muon 与 EoS](https://arxiv.org/abs/2609.34915)）；二次近似在 LLM 训练中的失效方式也被重新检验（[LLM 预训练与 EoS](https://arxiv.org/abs/2610.36778)）。

### 3.4 学习率、批大小与权重衰减

几乎所有大模型训练都有 warmup。[Kalra 与 Barkeshli](https://arxiv.org/abs/2406.09405) 发现 warmup 的主要作用是把网络带到锐度更低、更"良态"的区域，从而容许更大的学习率。衰减阶段，[余弦调度](https://arxiv.org/abs/1608.03983)曾是默认；[MiniCPM](https://arxiv.org/abs/2404.06395) 提出 Warmup-Stable-Decay（WSD），可在任意时刻衰减得到可用的检查点；[河谷地形](https://arxiv.org/abs/2410.05192)解释了 WSD：稳定期沿着"河流"方向前进（损失看起来下降很慢），衰减期才下到"谷底"。[Bergsma 等](https://arxiv.org/abs/2502.15938)发现算力最优训练中，线性衰减到零（D2Z）稳定优于余弦衰减到 10%。

[Goyal 等](https://arxiv.org/abs/1706.02677)的线性缩放规则在临界批大小以内成立；超过之后，加大批量不再减少所需步数。[Zhang 等](https://arxiv.org/abs/2410.21676)发现临界批大小主要随训练数据量而非模型规模增长；[Merrill 等](https://arxiv.org/abs/2505.23971)给出训练中实时测量临界批大小的经验方法，并据此做批大小 warmup。

#### 最近一年（2025-11 → 2026-10）

**最优调度可以推导。** 在幂律核回归中，用源指数 $s$（任务难度）与容量指数 $q$ 描述数据，把损失分解为"信号学习"与"噪声遗忘"两项。在固定步数与稳定性约束下求最优学习率调度，得到一个尖锐的相变（[最优调度](https://arxiv.org/abs/2602.06797)）：

$$
s>1-\tfrac1q\ (\text{容易任务}):\ \eta_t\ \text{从一开始幂律衰减};\qquad s<1-\tfrac1q\ (\text{困难任务}):\ \eta_t=\text{warmup-stable-decay}.
$$

WSD 因此不只是工程经验，而是困难任务下的最优解；类似的分析也给出最优批大小调度（[快追赶、晚切换](https://arxiv.org/abs/2602.14208)）与学习率、批大小的规模标度（[超参数扩展律](https://arxiv.org/abs/2603.15958)）。

**权重衰减决定有效学习率。** 对尺度不变的权重（后面接归一化），$L(cw)=L(w)$，因此梯度与 $w$ 正交且 $\|\nabla L(w)\|=\|\hat g\|/\|w\|$（$\hat g$ 为单位范数处的梯度）。SGD 加权重衰减一步后

$$
\|w_{t+1}\|^2=(1-\eta\lambda)^2\|w_t\|^2+\eta^2\frac{\|\hat g\|^2}{\|w_t\|^2}.
$$

令其平衡，得 $\|w\|^4\approx \eta\|\hat g\|^2/(2\lambda)$，有效（角度）学习率

$$
\eta_{\text{eff}}=\frac{\eta}{\|w\|^2}\approx\frac{\sqrt{2\eta\lambda}}{\|\hat g\|}.
$$

所以名义学习率并不是真正起作用的量：调度应针对有效学习率设计，权重归一化训练的收益主要来自它诱导的有效学习率轨迹（[有效学习率](https://arxiv.org/abs/2610.40362)、[精确调度律](https://arxiv.org/abs/2609.09116)）。同一机制解释了几件事：μP 在 AdamW 稳态下要配合按宽度缩放的权重衰减（[逐层扩展规则](https://arxiv.org/abs/2510.15262)）；归一化与权重衰减的相互作用会产生临界性并引发损失尖峰（[权重范数临界](https://arxiv.org/abs/2607.21005)）；巨大激活由权重衰减驱动增长（[生命周期](https://arxiv.org/abs/2610.00423)）。

**grokking 的时钟。** 在"不改变训练集预测"的方向（grokking 子空间）上，数据梯度为零，唯一的力是权重衰减：$\theta_\perp(t)\approx\theta_\perp(0)\,e^{-\eta\lambda t}$，延迟泛化的时间尺度因此是 $1/(\eta\lambda)$ 量级（[权重衰减时钟](https://arxiv.org/abs/2607.23967)）。

### 3.5 初始化与归一化

若第 $l$ 层 $h_l=W_lh_{l-1}$，且权重独立同分布、均值为零，则 $\mathrm{Var}(h_l)=n_{\text{in}}\mathrm{Var}(W)\mathrm{Var}(h_{l-1})$。要让信号方差不随深度爆炸或消失，需 $\mathrm{Var}(W)=1/n_{\text{in}}$（线性 / tanh，[Glorot](https://proceedings.mlr.press/v9/glorot10a.html)）或 $2/n_{\text{in}}$（ReLU 只保留一半，[He 等](https://arxiv.org/abs/1502.01852)）。

BatchNorm 按批统计量标准化（[Ioffe 与 Szegedy](https://arxiv.org/abs/1502.03167)）；Transformer 使用不依赖批的 LayerNorm（[Ba 等](https://arxiv.org/abs/1607.06450)）及其简化 RMSNorm（[Zhang 与 Sennrich](https://arxiv.org/abs/1910.07467)）：

$$\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot\gamma .$$

把归一化放在残差分支内（Pre-LN）使训练更稳，免去对 warmup 的强依赖（[Xiong 等](https://arxiv.org/abs/2002.04745)）；但 Pre-LN 下残差流的方差随深度增长，深层的贡献被稀释——[深度的诅咒](https://arxiv.org/abs/2502.05795)发现许多 LLM 的深层几乎是恒等映射，并提出按深度缩放 LayerNorm。

**免归一化。** [DyT](https://arxiv.org/abs/2503.10622) 用逐元素 $\tanh(\alpha x)$ 替代 LayerNorm，在视觉、语言与扩散模型上不掉点；[Chen 等](https://arxiv.org/abs/2512.10938)系统搜索逐点函数，让免归一化 Transformer 超过归一化版本。[nGPT](https://arxiv.org/abs/2410.01131) 走另一极端：把所有向量都归一化到超球面上。

#### 最近一年（2025-11 → 2026-10）

归一化层本身被重新分析：它带来指数级的容量控制（[归一化的隐藏力量](https://arxiv.org/abs/2511.00958)）；归一化层中的可学习缩放向量参数极少、影响却很大（[缩放向量](https://arxiv.org/abs/2605.26895)）；无归一化 Transformer 在初始化时处于次临界的信号传播状态（[无归一化的信号传播](https://arxiv.org/abs/2604.11890)）；Post-LN 在深层下重新变得稳定且更具表达力（[Post-LN 回归](https://arxiv.org/abs/2601.19895)）。

### 3.6 μP 与超参数迁移

标准参数化下，网络变宽时最优学习率会漂移；NTK 参数化则在无限宽极限下退化为核方法、失去特征学习（[Yang 与 Hu](https://arxiv.org/abs/2011.14522)）。μP 选择让每层特征的更新量保持 $\Theta(1)$ 的缩放：以 Adam 为例，隐藏层学习率按 $1/n$ 缩放，输出层 logit 乘以 $1/n$。于是最优超参数在不同宽度间可迁移（[μTransfer](https://arxiv.org/abs/2203.03466)）。[谱条件](https://arxiv.org/abs/2310.17813)给出等价而更直观的表述：要求权重及其更新的谱范数与 $\sqrt{n_{\text{out}}/n_{\text{in}}}$ 同阶。[Everett 等](https://arxiv.org/abs/2407.05872)在上万次实验中比较了各种参数化与优化器组合。2024–2025 年，μP 被推广到深度、批大小与训练时长（[完整超参迁移](https://arxiv.org/abs/2512.22382)）、宽深联合缩放（[谱条件](https://arxiv.org/abs/2603.00541)）、不同优化器（[Gupta 等](https://arxiv.org/abs/2602.20937)）与 MoE（[MoE 超参迁移](https://arxiv.org/abs/2601.20205)、[Kakao](https://arxiv.org/abs/2608.20061)）。

#### 最近一年（2025-11 → 2026-10）

μP 只保证训练初期的更新尺度跨宽度一致；带归一化的网络很快进入由优化器主导的稳态，有效学习率随宽度变化，因此在 AdamW 下还需按宽度缩放权重衰减（[逐层扩展规则](https://arxiv.org/abs/2510.15262)）。μP 被推广到 MoE（[从 μP 到最大尺度稳定参数化](https://arxiv.org/abs/2605.14200)、[Complete-muE](https://arxiv.org/abs/2605.23893)）、循环 Transformer（[Loop-μP](https://arxiv.org/abs/2606.18524)、[DeepLoop](https://arxiv.org/abs/2607.13491)）与无限深度（[无限深特征学习](https://arxiv.org/abs/2512.21075)）；配合正确的超参数迁移，Shampoo、SOAP、Muon 等矩阵预条件优化器相对 AdamW 的增益在大规模下仍能保持（[超参数迁移](https://arxiv.org/abs/2512.05620)）。嵌入层学习率是迁移中最敏感的一项（[量化超参数迁移](https://arxiv.org/abs/2605.21486)）。

### 3.7 训练不稳定与低精度

[ViT-22B](https://arxiv.org/abs/2302.05442) 与 [Wortsman 等](https://arxiv.org/abs/2309.14322)发现两类典型的发散：注意力 logit 不受控增长（对策：QK-Norm）与输出 logit 发散（对策：z-loss）。

**训练稳定化。** [MSign](https://arxiv.org/abs/2602.01734) 把损失尖峰归因于权重矩阵稳定秩的坍缩；[ZClip](https://arxiv.org/abs/2504.02507) 用 z 分数做自适应梯度裁剪。

混合精度训练（[Micikevicius 等](https://arxiv.org/abs/1710.03740)）用 FP16/BF16 计算、FP32 主权重。[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 用细粒度（逐块）缩放的 FP8 完成 671B 模型的预训练。4-bit 训练在 2025 年前后成熟：[Quartet](https://arxiv.org/abs/2505.14669) 给出 MXFP4 原生训练的算法与扩展律；NVIDIA 用随机 Hadamard 变换、二维缩放与随机舍入，以 NVFP4 训练 12B 模型 10T token（[NVFP4](https://arxiv.org/abs/2509.25149)）；[Quartet II](https://arxiv.org/abs/2601.22813) 改进了无偏梯度估计；[Nemotron 3](https://arxiv.org/abs/2512.20856) Super 与 Ultra 用 NVFP4 训练。

#### 最近一年（2025-11 → 2026-10）

归一化引入的尺度不变分量与权重衰减相互作用，会产生一种临界性，引发损失尖峰（[权重范数临界](https://arxiv.org/abs/2607.21005)）；用非正规矩阵的谱可以提前预示失稳（[非正规谱特征](https://arxiv.org/abs/2605.23476)）。低精度方面，FP4 预训练中 E2M1 格式会带来系统性的收缩偏差（[FP4 收缩偏差](https://arxiv.org/abs/2606.20381)），注意力 logit 的秩感知谱界可用于保证低精度训练的稳定（[logit 谱界](https://arxiv.org/abs/2602.18851)），优化器状态量化会造成状态陈旧（[优化器状态量化](https://arxiv.org/abs/2603.16731)）。

## 4 泛化：为什么能推广，何时只是记忆

过参数化网络能拟合随机标签，却在真实数据上泛化良好。这一章依次讨论：经典界为何失效、双下降与良性过拟合、隐式偏置、网络为何胜过核方法（特征学习）、grokking、记忆的容量、生成模型的泛化，以及分布外与长度泛化。

### 4.1 经典理论为何不够

基于 VC 维、Rademacher 复杂度的界随参数量增长，对过参数化网络近乎空洞。[Zhang 等](https://arxiv.org/abs/1611.03530)的实验更直接：同一网络能完美拟合随机标签，说明假设空间的"容量"并不限制它；泛化必须由数据、架构与优化算法的交互来解释。

### 4.2 双下降与良性过拟合

[Belkin 等](https://arxiv.org/abs/1812.11118)发现，测试误差随模型复杂度先呈经典 U 形，在恰好能插值训练集的"插值阈值"处达到峰值，然后在更大模型上再次下降；[Nakkiran 等](https://arxiv.org/abs/1912.02292)在模型规模、训练轮数与数据量三个维度上都观察到这一现象。[Bartlett 等](https://arxiv.org/abs/1906.11300)刻画了线性回归中"良性过拟合"的条件：当特征协方差的有效秩足够大，插值噪声被分散到许多不重要的方向上，对预测影响很小。

#### 最近一年（2025-11 → 2026-10）

双下降被重新表述为一种变分原理（[双下降即最小作用量原理](https://arxiv.org/abs/2609.19076)）；良性过拟合推广到一般范数与分布（[一般范数](https://arxiv.org/abs/2609.33675)）并有了最优输运几何的解释（[输运几何](https://arxiv.org/abs/2604.08625)）。生成模型给出了反例：扩散模型中不存在良性过拟合（见 4.7 节）。

### 4.3 隐式偏置

在可分数据上用逻辑损失做梯度下降，参数方向收敛到最大 $\ell_2$ 间隔解（[Soudry 等](https://arxiv.org/abs/1710.10345)）；矩阵分解中梯度下降偏向低秩解。换言之，在无数能拟合训练集的解中，优化算法替我们挑了一个"简单"的。[Adam 与 SGD 的隐式偏置](https://arxiv.org/abs/2505.24022)比较了两类优化器偏好的解：SGD 偏向简单的线性可分特征，Adam 更容易学到丰富的非线性特征。[Wilson](https://arxiv.org/abs/2503.02113)的立场文章总结道：良性过拟合、双下降、过参数化都可用"软归纳偏置"（偏好简单解但不排除复杂解）与经典的 PAC-Bayes / 可数假设界解释，深度学习并不那么神秘。

#### 最近一年（2025-11 → 2026-10）

隐式偏置与优化器几何直接挂钩：带动量的最速下降方向上收敛到对应范数的最大间隔解，Adam 对应 ℓ∞、Muon 对应谱范数（[Adam 与 Muon 的隐式偏置](https://arxiv.org/abs/2602.16340)，推导见 3.2 节）。深度本身也带来隐式偏置，把神经坍缩推向 softmax 码（[深度的隐式偏置](https://arxiv.org/abs/2605.23087)）；SGD 在多元 ReLU 网络中导致有效宽度坍缩（[有效宽度坍缩](https://arxiv.org/abs/2607.03613)）；可以从训练中实证地估计隐式正则的形式与强度（[估计隐式正则](https://arxiv.org/abs/2605.05436)）。

### 4.4 懒惰区间与特征学习：网络为何胜过核方法

把网络在初始化处线性化，$f_\theta(x)\approx f_{\theta_0}(x)+\langle\nabla_\theta f_{\theta_0}(x),\theta-\theta_0\rangle$。在 NTK 缩放的无限宽极限下，训练全程参数几乎不动，网络等价于用固定核 $K(x,x')=\langle\nabla f(x),\nabla f(x')\rangle$ 做核回归（[Jacot 等](https://arxiv.org/abs/1806.07572)）。[Chizat 等](https://arxiv.org/abs/1812.07956)称之为"懒惰训练"：可分析，但不学习新特征。平均场理论（[Mei 等](https://arxiv.org/abs/1804.06561)）与 μP 描述了另一种极限：特征持续演化，更接近真实网络的行为。最近的工作沿此继续：[交替梯度流](https://arxiv.org/abs/2506.06489)刻画两层网络中特征逐个被学会的过程，[Ruben 等](https://arxiv.org/abs/2602.15593)统一了 RNN 与深网络的特征学习。

#### 最近一年（2025-11 → 2026-10）

多指标模型 $f^*(x)=g(Ux)$，$U\in\mathbb R^{r\times d}$ 是隐藏子空间，是研究"网络如何从高维输入中找到少数相关方向"的标准模型。核方法（包括 NTK 极限）不学 $U$，样本复杂度随 $d$ 多项式恶化；特征学习的网络先找到子空间再拟合 $g$。最近一年：在一般的非退化条件下，分层梯度下降训练的两层网络学习一般多指标模型，所需样本量接近信息论极限（[接近信息论极限](https://arxiv.org/abs/2511.15120)）；统计—计算权衡被调和分析精确刻画（[调和分析](https://arxiv.org/abs/2602.09959)）；全批量 GD 与单遍 SGD 在单指标问题上存在样本复杂度分离（[GD vs SGD](https://arxiv.org/abs/2602.02431)），多遍 SGD 重用数据会改变学习动力学（[多遍 SGD](https://arxiv.org/abs/2601.21093)）。反过来，在组合学习上 NTK 有指数级次优（[NTK 的二分](https://arxiv.org/abs/2607.06382)）。宽、深极限方面：存在让深度趋于无穷时仍保持特征学习的参数化（[无限深特征学习](https://arxiv.org/abs/2512.21075)），线性残差网络中宽、深两个极限在训练中可以交换（[极限可交换](https://arxiv.org/abs/2610.30729)）。

### 4.5 Grokking：从记忆到泛化

[Power 等](https://arxiv.org/abs/2201.02177)在模运算等小算法任务上发现：训练精度早已 100%，测试精度却在多得多的步数后突然跃升。[Nanda 等](https://arxiv.org/abs/2301.05217)逆向工程出模加法网络使用离散傅里叶变换与三角恒等式的电路，并把训练分为记忆、电路形成与清理三个阶段，权重衰减推动了"清理"。[Prieto 等](https://arxiv.org/abs/2501.04697)发现没有正则化时，grokking 会被 softmax 的浮点误差（Softmax Collapse）阻断，并提出 StableMax 与只保留与权重正交的梯度分量的 ⊥Grad；[岭回归中的可证 grokking](https://arxiv.org/abs/2601.19791)与[谱理论](https://arxiv.org/abs/2609.26679)给出了更严格的刻画；[Abramov 等](https://arxiv.org/abs/2504.20752)把 grokking 带到真实世界的多跳推理：通过数据增强提高推断事实与原子事实之比即可触发。

#### 最近一年（2025-11 → 2026-10）

**grokking = 零损失流形上的范数最小化。** 记训练损失为零的参数集合为 $\mathcal M=\{\theta: L_{\text{train}}(\theta)=0\}$。拟合之后，梯度下降加权重衰减在小学习率、小权重衰减极限下等价于在 $\mathcal M$ 上做投影梯度流

$$
\dot\theta=-\lambda\,P_{T_\theta\mathcal M}\,\theta ,
$$

即沿流形缓慢走向最小范数解，泛化在到达低范数区域时才出现（[grokking 的几何](https://arxiv.org/abs/2511.01938)）。这与"维度相变"（[维度相变](https://arxiv.org/abs/2604.04655)）、"权重衰减时钟"（3.4 节）相互印证；模加法与有限群乘法的完整机制也被写出，网络可证明地学到群的不可约表示（[模加法](https://arxiv.org/abs/2602.16849)、[群组合的谱表示](https://arxiv.org/abs/2606.02993)）。

### 4.6 记忆：容量与可提取性

[语言模型物理学 3.3](https://arxiv.org/abs/2404.05405) 估计模型约能存储 2 bit 知识/参数；[Morris 等](https://arxiv.org/abs/2505.24832)用随机数据测得 GPT 类模型约有 3.6 bit/参数 的记忆容量，并发现容量被填满后模型开始"顿悟"出泛化。[语言模型物理学 3.1](https://arxiv.org/abs/2309.14316) 说明知识能否被提取取决于预训练中同一事实是否以多样化的方式出现。

#### 最近一年（2025-11 → 2026-10）

记忆被放进几何与训练动力学中理解：Fisher 信息刻画记忆在参数空间中的几何（[记忆的 Fisher 几何](https://arxiv.org/abs/2610.16161)）；记住的知识与能在推理中使用之间存在准确率差与时间滞后（[知道—使用鸿沟](https://arxiv.org/abs/2607.08393)）；LayerNorm 的位置影响 Transformer 的记忆与泛化（[LayerNorm 与记忆](https://arxiv.org/abs/2511.10566)）；可提取记忆也有了第一性原理的分析（[可提取记忆](https://arxiv.org/abs/2607.12649)）。

### 4.7 生成模型的泛化：扩散模型何时只是记忆

**扩散模型的记忆。** 对训练集 $\{x_i\}_{i=1}^n$，经验分布加噪后的最优去噪器是

$$
D^*(z_t)=\mathbb E[x\mid z_t]=\sum_{i=1}^n w_i(z_t)\,x_i,\qquad w_i\propto \exp\!\Big(-\frac{\|z_t-\alpha_t x_i\|^2}{2\sigma_t^2}\Big),
$$

用它采样只会复现训练样本。所以扩散模型的泛化依赖于"学得不完美"。最近一年的结论：在流形假设下，粗糙的分数只抓住数据的几何（流形）而丢掉细节分布，因此训练中先在流形上泛化、后记忆单个样本（[先泛化后记忆](https://arxiv.org/abs/2603.23792)）；记忆与泛化可证明分离（[可证明分离](https://arxiv.org/abs/2511.03202)）；而且除非样本量随维度指数增长，过拟合与好的泛化不能同时出现：扩散模型中不存在分类问题里的"良性过拟合"（[无良性过拟合](https://arxiv.org/abs/2607.02671)）。可解析的贝叶斯扩散模型精确刻画了从记忆到泛化的相变（[泛化相变](https://arxiv.org/abs/2607.08041)）。

### 4.8 分布外、组合与长度泛化

在训练分布之外仍然正确，是比独立同分布泛化更强的要求。三类问题最受关注：组合泛化（训练中没见过的因素组合）、长度泛化（比训练更长的输入）与分布偏移。

#### 最近一年（2025-11 → 2026-10）

- **组合泛化需要什么样的表征。** 视觉嵌入模型要实现组合泛化，表征必须线性且各因素相互正交（[线性正交](https://arxiv.org/abs/2602.24264)）；有效维度能强烈预测 ImageNet 准确率（[几何与泛化](https://arxiv.org/abs/2602.00130)）。
- **长度泛化来自优化而不只是表达力。** 梯度训练的 Transformer 可证明学会能长度泛化的思维链（[CoT 长度泛化](https://arxiv.org/abs/2511.07378)）；在固定偏移检索任务上，相对位置泛化、绝对位置只记忆，差异由训练出的注意力头的隐式偏置决定（[相对 vs 绝对](https://arxiv.org/abs/2607.18759)）；精确解在参数空间中的体积决定能否长度泛化（[精确解体积](https://arxiv.org/abs/2610.07676)），恰当的正则化是必要条件（[正则化](https://arxiv.org/abs/2610.04518)）；Transformer 在排序上会同时学到两种算法，只有一种能长度泛化（[两种排序算法](https://arxiv.org/abs/2610.26974)）。
- **循环结构。** 循环 Transformer 与对数深度循环单元在电路类任务上能做到很强的长度外推，但也有明确的极限（[通用 Transformer](https://arxiv.org/abs/2608.31067)、[对数深度循环](https://arxiv.org/abs/2605.26035)、[循环的极限](https://arxiv.org/abs/2609.33144)）。

## 5 扩展：为什么越大越好

损失随参数、数据、算力呈跨越多个数量级的幂律，这让大模型的训练变成可规划的工程。这一章先讲经验规律与算力最优分配，再讲幂律从哪里来，然后讨论数据、精度、稀疏度、优化器、深度与循环等新变量，最后讨论涌现与测试时扩展。

### 5.1 幂律与算力最优

[Kaplan 等](https://arxiv.org/abs/2001.08361)发现语言模型的测试损失随参数量 $N$、数据量 $D$、算力 $C$ 分别呈跨越多个数量级的幂律。[Chinchilla](https://arxiv.org/abs/2203.15556) 用参数化形式

$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$

拟合，并在约束 $C\approx6ND$ 下用拉格朗日乘子求最优分配：

$$N^\star(C)\propto C^{\frac{\beta}{\alpha+\beta}},\qquad D^\star(C)\propto C^{\frac{\alpha}{\alpha+\beta}} .$$

拟合得到 $\alpha\approx\beta$，即参数与数据应大致等比例扩展，约 20 token/参数。Kaplan 与 Chinchilla 的差异后来被归因于是否计入嵌入参数、规模范围与学习率调度（[Pearce 与 Song](https://arxiv.org/abs/2406.12907)、[Porian 等](https://arxiv.org/abs/2406.19146)）。实践中，由于推理成本按模型大小计，许多模型远超此比例地"过训练"（如 Llama 3 8B 用了 15T token）。

### 5.2 为什么是幂律

一种解释来自数据：若数据分布在低维流形上，或特征重要性呈幂律谱，则用有限容量逼近时误差自然呈幂律（[源于数据分布的扩展律](https://arxiv.org/abs/2412.07942)）。另一种来自表征：[Liu 等](https://arxiv.org/abs/2505.10465)在 Anthropic 的叠加玩具模型中用权重衰减控制叠加程度，发现强叠加区间下损失随宽度 $m$ 稳健地按 $1/m$ 下降，且与真实 LLM 的观测一致——特征叠加可能是神经扩展律的关键成因。

#### 最近一年（2025-11 → 2026-10）

**幂律从哪里来。** 一类统一的图景：目标由许多特征组成，第 $k$ 个特征的重要性 $p_k\propto k^{-(1+\alpha)}$；容量（或数据）为 $N$ 的模型大致恢复前 $N$ 个特征，剩余损失

$$
L(N)\approx\sum_{k>N}p_k\propto N^{-\alpha}.
$$

最近一年的工作把这个图景做成可证明的模型：层级组合特征按强弱被依次恢复，强特征在小样本时就可检测，弱特征需要更多数据（[序贯特征恢复](https://arxiv.org/abs/2605.14567)）；从随机图到自然语言的幂律结构如何转化为扩展律（[扩展律的起源](https://arxiv.org/abs/2601.10684)）；Zipf 词频下瓶颈二元模型的精确扩展律（[随机特征二元模型](https://arxiv.org/abs/2610.19281)）。训练动力学方面，特征叠加会把损失随步数下降的指数推向与数据无关的普适值 ≈1，比无叠加的逐个学习快一个数量级（[叠加统一幂律动力学](https://arxiv.org/abs/2602.01045)）。大模型即使在无限数据下也能学会小模型学不会的稀有任务，原因是任务之间对神经元的竞争（[大模型为何学得更多](https://arxiv.org/abs/2605.29548)）。

### 5.3 数据、精度、稀疏度与优化器

[Muennighoff 等](https://arxiv.org/abs/2305.16264)发现重复数据约 4 轮内几乎与新数据等价，之后收益快速衰减；[Prabhudesai 等](https://arxiv.org/abs/2507.15857)发现在数据受限、算力充足时，掩码扩散语言模型反复训练优于自回归；[Kim 等](https://arxiv.org/abs/2509.14786)说明在"无限算力"下，强正则化、集成与蒸馏可以把数据效率提高 5 倍以上。

2024–2025 年，扩展律被推广到更多维度：

- 精度：[Kumar 等](https://arxiv.org/abs/2411.04330)把训练与推理精度纳入扩展律，发现训练越充分，后训练量化的损害越大；[Ouyang 等](https://arxiv.org/abs/2411.17691)得出同样结论；[ParetoQ](https://arxiv.org/abs/2502.02631) 统一比较 1–4 bit。
- 稀疏度：[Abnar 等](https://arxiv.org/abs/2501.12370)发现固定算力时更稀疏（总参数更多）的 MoE 损失更低；[Ludziejewski 等](https://arxiv.org/abs/2502.05172)在显存约束下给出稠密与 MoE 的联合扩展律；[Engram](https://arxiv.org/abs/2601.07372) 发现 MoE 与查表记忆之间存在 U 形最优分配。
- 蒸馏、并行与密度：[蒸馏扩展律](https://arxiv.org/abs/2502.08606)给出教师与学生之间的算力分配；[ParScale](https://arxiv.org/abs/2505.10475) 用 P 路并行前向替代参数扩展，约等于参数扩展 $O(\log P)$ 倍；[能力密度定律](https://arxiv.org/abs/2412.04315)指出同等能力所需参数约每 3.3 个月减半。
- 更精细的拟合：[Farseer](https://arxiv.org/abs/2506.10972) 在整个 $(N,D)$ 平面上拟合损失面，外推误差显著降低。

#### 最近一年（2025-11 → 2026-10）

几个过去被固定的变量进入了扩展律：

- *优化器*：单次训练中损失随步数的"动态指数"依优化器而异，但调优后的"最优数据指数"在各优化器下都收敛到 1/3（[1/3 数据指数](https://arxiv.org/abs/2609.37745)）。
- *测试时计算*（见 5.6 节）：把重复采样的推理成本计入后，计算最优的是比 Chinchilla 更"过度训练"的小模型（[T² 扩展律](https://arxiv.org/abs/2604.01411)）。
- *数据重复*：数据受限时需要更强正则化（[无限算力预训练](https://arxiv.org/abs/2509.14786)、[数据受限扩展律](https://arxiv.org/abs/2605.01640)、[重复数据](https://arxiv.org/abs/2610.15965)），扩散语言模型比自回归模型更耐重复（[超级数据学习者](https://arxiv.org/abs/2511.03276)）。

### 5.4 深度、宽度与循环

宽度与深度如何分配，长期靠经验。最近一年的几条结果把它变成了可以讨论的问题。

#### 最近一年（2025-11 → 2026-10）

**循环深度。** 把一个块循环 $r$ 次，拟合

$$
L=E+A\,\big(N_{\text{once}}+r^{\varphi}N_{\text{rec}}\big)^{-\alpha}+B\,D^{-\beta},\qquad \varphi\approx0.46,
$$

即循环 $r$ 次的效果介于"一个块"（$\varphi=0$）与"$r$ 个独立块"（$\varphi=1$）之间（[等深度扩展律](https://arxiv.org/abs/2604.21106)）。

**深度的回报递减。** 多数层功能相近，导致深度扩展的回报递减（[逆深度扩展](https://arxiv.org/abs/2602.05970)）；注意力的非线性本身就会带来这种逆深度扩展（[注意力非线性](https://arxiv.org/abs/2610.11063)）；有人据此主张 Transformer 应更宽而非更深（[深度错觉](https://arxiv.org/abs/2601.20994)）。另一方面，Post-LN 的复活与残差流的重新设计（2.6 节）试图让更深的网络重新有效。

### 5.5 涌现

[Wei 等](https://arxiv.org/abs/2206.07682)列举了只在超过某个规模后才出现的能力；[Schaeffer 等](https://arxiv.org/abs/2304.15004)指出其中许多"突变"来自非线性、不连续的评测指标（如精确匹配），换成连续指标后能力随规模平滑增长。[观察式扩展律](https://arxiv.org/abs/2405.10938)用公开模型族的低维能力空间预测下游能力。实用结论是：损失可预测，具体任务的阈值不总可预测。

### 5.6 测试时计算与 RL 的扩展

[Snell 等](https://arxiv.org/abs/2408.03314)表明按题目难度自适应分配测试时计算，效率可超过把模型放大 14 倍；[Large Language Monkeys](https://arxiv.org/abs/2407.21787) 发现重复采样的覆盖率随样本数呈对数线性增长。RL 后训练也有了扩展律：[ScaleRL](https://arxiv.org/abs/2510.13786) 用超过 40 万 GPU 小时拟合出 S 形的"算力-性能"曲线，发现不同配方的渐近性能不同，而损失聚合、归一化、课程等细节主要影响效率。

#### 最近一年（2025-11 → 2026-10）

把重复采样的推理成本计入总预算后，计算最优的预训练是比 Chinchilla 更'过度训练'的小模型（[T² 扩展律](https://arxiv.org/abs/2604.01411)）；循环模型提供了另一种测试时扩展的方式：增加循环次数而不增加参数（2.7 节）。

## 6 表征：网络内部学到了什么

训练得到的网络在内部形成了怎样的表示？这一章从几何讲起（线性表征、表征几何），再讨论不同模型的表征是否趋同（柏拉图表征假说），然后讨论可解释性的核心难题（叠加）与主要工具（稀疏自编码器、电路与归因图）。

### 6.1 线性表征与表征几何

"线性表征假说"认为，概念在激活空间中对应方向，可以用线性探针读出、用加减向量操控（[线性表征假说](https://arxiv.org/abs/2311.03658)）。这一假说支撑了激活引导、人格向量等大量方法，但它隐含了很多选择：用什么内积、什么等价关系、在哪一层。

#### 最近一年（2025-11 → 2026-10）

- **线性表征需要说明对称性。** 不同的等价（群作用）保留不同结构，看似研究同一表征的探针与干预可能在检验不同的假说（[需要群作用](https://arxiv.org/abs/2609.27158)）；保持函数不变的参数对称会重塑表征几何，比较表征之前必须先处理对称（[参数对称](https://arxiv.org/abs/2609.39078)）。
- **几何的结构。** 下一词分布的 Fisher–Rao 几何在模型间共享、由训练习得，并可用于精确控制（[信息几何](https://arxiv.org/abs/2609.11063)）；多步加法在残差流中形成以原始和为锚点的轨迹（[加法的形状](https://arxiv.org/abs/2606.03645)）；不同架构都学到周期性的数字特征，但只有部分学到几何可分的结构（[数字表征的趋同进化](https://arxiv.org/abs/2604.20817)）。
- **表征会变。** 线性表征方向上的读数会随对话上下文剧烈漂移（[对话中的变化](https://arxiv.org/abs/2601.20834)），上下文学习过程中模型会重组表征几何（[ICL 中的重组](https://arxiv.org/abs/2605.28854)）；不同提示接相同续写时，隐状态差距按 $t^{-\alpha}$ 衰减，训练过的模型 $\alpha\approx0.55$–$0.7$（[条件无毛定理](https://arxiv.org/abs/2610.24133)）。

### 6.2 表征趋同：柏拉图假说的证明与反证

[柏拉图表征假说](https://arxiv.org/abs/2405.07987)观察到不同架构、不同模态的大模型表征正在收敛，[vec2vec](https://arxiv.org/abs/2505.12540) 甚至能在无配对数据的情况下翻译不同文本嵌入模型的向量空间。

#### 最近一年（2025-11 → 2026-10）

柏拉图表征假说认为，规模足够大时不同模型、不同模态的表征趋向同一个"现实的统计模型"。最近一年的检验结果是两面的：

- **削弱。** 互 $k$ 近邻等相似度度量会随网络深度、宽度系统性虚高，用置换零分布校准后，全局趋同的证据减弱（[亚里士多德式重审](https://arxiv.org/abs/2602.14486)）；原研究的跨模态证据只抓住粗结构，扩大数据后需要 $k$ 同比例增长，论证随之削弱（[回到柏拉图的洞穴](https://arxiv.org/abs/2604.18572)）；行为趋同不等于表征趋同（[行为 vs 表征](https://arxiv.org/abs/2609.37836)）。
- **证明。** 若正样本对是同一潜在对象的两次观测，分别在各模态上训练的 JEPA 学到相差一个旋转的同一表示（[柏拉图表征定理](https://arxiv.org/abs/2610.39302)）；人脑响应中也能无监督恢复同一几何（[人脑](https://arxiv.org/abs/2605.20496)）。

几何与泛化的关系也更具体：组合泛化要求表征线性且各因素正交（[线性正交](https://arxiv.org/abs/2602.24264)）；有效维度强烈预测 ImageNet 准确率（[几何与泛化](https://arxiv.org/abs/2602.00130)）；保持函数不变的参数对称会重塑表征几何（[参数对称](https://arxiv.org/abs/2609.39078)）；"线性表征假说"其实是一族假说，取决于采用哪种群作用下的等价（[需要群作用](https://arxiv.org/abs/2609.27158)）。

### 6.3 神经坍缩

分类网络训练到末期，同类样本的最后一层特征坍缩到类均值，类均值构成等角紧框架，分类器与之对齐（[神经坍缩](https://arxiv.org/abs/2008.08186)）。它解释了末层几何，也引出了"表征是否应该完全坍缩"的问题。

#### 最近一年（2025-11 → 2026-10）

神经坍缩是否出现取决于优化器（[优化器与神经坍缩](https://arxiv.org/abs/2602.16642)）；深度线性网络中有闭式的深宽关系（[深宽闭式律](https://arxiv.org/abs/2610.21820)）；语言模型因信息下限而不会完全坍缩（[语言模型禁止坍缩](https://arxiv.org/abs/2607.09487)）；自监督表征中的"方向性神经坍缩"解释了少样本迁移（[方向性坍缩](https://arxiv.org/abs/2603.03530)）；长尾识别中完全坍缩反而不是最优（[部分坍缩](https://arxiv.org/abs/2610.36460)）。

### 6.4 叠加

[Olah 等](https://distill.pub/2020/circuits/zoom-in/)提出"特征是基本单元、权重把特征连成电路"的研究纲领；[Elhage 等](https://transformer-circuits.pub/2021/framework/index.html)把 Transformer 分解为对残差流读写的注意力头（QK 电路决定看哪里，OV 电路决定搬什么）。[归纳头](https://arxiv.org/abs/2209.11895)与 [IOI 电路](https://arxiv.org/abs/2211.00593)是早期的完整案例。困难在于神经元是多义的：[叠加](https://arxiv.org/abs/2209.10652)理论表明，当特征稀疏、数量多于维度时，网络会用近乎正交的方向存储它们，以少量干扰换取容量。

#### 最近一年（2025-11 → 2026-10）

叠加模型：稀疏特征 $x\in\mathbb R^m$ 被压进 $n<m$ 维 $h=Wx$，再用 $\hat x=\mathrm{ReLU}(W^\top W x+b)$ 恢复。干扰项 $W^\top W-I$ 的大小由字典相干性决定，偏置 $b$ 负责滤掉小干扰。最近一年对这一模型证明了特征恢复定理（[叠加的数学理论](https://arxiv.org/abs/2608.27540)），给出特征几何的谱理论（[谱叠加](https://arxiv.org/abs/2602.02224)）与一般叠加下可证明的特征提取算法（[可证明提取](https://arxiv.org/abs/2512.15987)）。

### 6.5 稀疏自编码器及其可靠性

SAE 学一个过完备字典 $x\approx W_d\,\mathrm{ReLU}(W_ex+b)$ 并对激活施加稀疏惩罚，把多义神经元分解为单义特征（[Cunningham 等](https://arxiv.org/abs/2309.08600)、[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)）。[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 在 Claude 3 Sonnet 上提取了数千万特征，包括抽象、多语言与安全相关的特征，并可用于操控行为；[OpenAI](https://arxiv.org/abs/2406.04093) 给出 SAE 的扩展律，[Gemma Scope](https://arxiv.org/abs/2408.05147) 开放了全层 SAE。

**反思。** [Kantamneni 等](https://arxiv.org/abs/2502.16681)发现在数据稀缺、分布偏移等困难设置下，SAE 探针并不稳定地优于简单线性探针；[合理性检查](https://arxiv.org/abs/2602.14111)发现 SAE 在部分指标上未必胜过随机基线；[开放问题](https://arxiv.org/abs/2501.16496)综述了方法论层面的挑战。

#### 最近一年（2025-11 → 2026-10）

SAE 作为工具接受了更严格的检验：单个特征跨随机种子常不复现，但张成的子空间可以复现（[种子依赖](https://arxiv.org/abs/2606.12138)）；训练数据增加时功能指标很快饱和，却不保证恢复真实特征（[数据收益递减](https://arxiv.org/abs/2610.20680)）；保持语义的提示变换会让激活在不同特征间重新分配（[表征重分配](https://arxiv.org/abs/2610.21240)）；评测基准本身也不可靠（[SAE 基准](https://arxiv.org/abs/2605.18229)）。另一条线把机制追溯到训练：用影响函数把可解释单元追到具体样本（[机制性数据归因](https://arxiv.org/abs/2601.21996)）。

### 6.6 电路、归因图与内省

- 归因图：Anthropic 用跨层转码器（[transcoder](https://arxiv.org/abs/2406.11944) 的推广）构建可解释的替代模型，再为具体输入画出特征之间的因果图（[电路追踪](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)），在 Claude 3.5 Haiku 中看到两跳推理的中间步骤、写诗时提前选定韵脚、不同语言共享的概念表示等（[大语言模型的生物学](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)）。
- 设计可解释的模型：OpenAI 训练权重极稀疏的 Transformer，使每个神经元只有少量连接，剪枝后得到人能读懂的小电路，并发现扩大模型能改善"能力-可解释性"前沿（[Gao 等](https://arxiv.org/abs/2511.13653)）。
- 内省：[Lindsey](https://arxiv.org/abs/2601.01828) 向模型激活中注入已知概念，发现最强的模型有时能察觉并准确说出被注入的概念，还能区分自己的输出与人为预填——有限但真实的内省能力。
- 人格与安全：[人格向量](https://arxiv.org/abs/2507.21509)提取"邪恶""谄媚""幻觉"等特质方向，可用于监控、以及在训练中预防人格漂移；[审计游戏](https://arxiv.org/abs/2503.10965)中，蓝队借助可解释性工具找出了被故意植入的隐藏目标。

#### 最近一年（2025-11 → 2026-10）

机制研究开始落到训练过程与数学基础上：把可解释单元追溯到具体训练样本（[机制性数据归因](https://arxiv.org/abs/2601.21996)）；读出（反嵌入）矩阵本身的学习会滞后于表征，隐状态可能早已编码答案却读不出来（[学会读出](https://arxiv.org/abs/2610.21025)）；属性在残差流中可读出的深度早于它影响输出的深度（[早编码、晚使用](https://arxiv.org/abs/2609.07139)）；有害回应由一组跨伤害类型共享的参数支持，可被定向剪除（[有害回应的独立机制](https://arxiv.org/abs/2604.09544)）；用 AI 自动化地发现机制也开始出现（[Mechanist](https://arxiv.org/abs/2608.12036)）。

## 7 从数据本身学习：预训练、自监督与生成模型

不需要人工标签的学习信号有三类：预测下一个 token（语言模型预训练）、在表征空间中预测或对比（自监督表征学习）、建模整个数据分布（生成模型）。它们决定了模型学到的表征与能力。

### 7.1 下一 token 预训练：数据与分词

架构趋同后，数据决定了大部分差异。路线从"过滤 + 去重"（[RefinedWeb](https://arxiv.org/abs/2306.01116)）到基于模型的质量打分（[FineWeb-Edu](https://arxiv.org/abs/2406.17557)、[DCLM](https://arxiv.org/abs/2406.11794)），再到改写与合成（[Phi-1](https://arxiv.org/abs/2306.11644)、[Nemotron-CC](https://arxiv.org/abs/2412.02595)、[BeyondWeb](https://arxiv.org/abs/2508.10975)：改写网页优于从零生成）。数据配比被纳入扩展律（[Shukor 等](https://arxiv.org/abs/2507.09404)、[DataDecide](https://arxiv.org/abs/2504.11393)：小规模实验的排序在大规模基本成立）；[OPUS](https://arxiv.org/abs/2602.05400) 在每一步按优化器诱导的更新方向挑选数据。合成数据也有风险：[模型崩溃](https://arxiv.org/abs/2305.17493)表明递归地用生成数据训练会丢失分布尾部。

预训练与后训练之间出现了"中期训练"：用高质量数学、代码、长上下文与推理数据退火。[OctoThinker](https://arxiv.org/abs/2506.20512) 发现中期训练决定了基座能否被 RL 有效提升；[Agentic CPT](https://arxiv.org/abs/2509.13310) 与 [daVinci-Dev](https://arxiv.org/abs/2601.18418) 把智能体轨迹放进持续预训练。

BPE（[Sennrich 等](https://arxiv.org/abs/1508.07909)）仍是主流，但它带来多语言不公平（[Token 税](https://arxiv.org/abs/2509.05486)）与字符级盲区。[BLT](https://arxiv.org/abs/2412.09871) 按下一字节的熵动态切块，首次在规模上匹敌基于 tokenizer 的模型；[H-Net](https://arxiv.org/abs/2507.07955) 端到端学习分块；[SuperBPE](https://arxiv.org/abs/2503.13423) 允许跨空格的"超词"。目标函数层面：[多 token 预测](https://arxiv.org/abs/2404.19737)提升样本效率（DeepSeek-V3 采用）；[强化预训练](https://arxiv.org/abs/2506.08007)把下一 token 预测改写为可验证的推理任务；[Large Concept Models](https://arxiv.org/abs/2412.08821)、[NCP](https://arxiv.org/abs/2609.10715) 在句子或"概念"层面建模；[LLaDA](https://arxiv.org/abs/2502.09992) 用掩码扩散替代自回归。

### 7.2 对比学习与自蒸馏

InfoNCE（[CPC](https://arxiv.org/abs/1807.03748)）对一个正样本与 $K$ 个负样本：

$$\mathcal L=-\log\frac{\exp(\mathrm{sim}(z,z^+)/\tau)}{\exp(\mathrm{sim}(z,z^+)/\tau)+\sum_{j=1}^{K}\exp(\mathrm{sim}(z,z_j^-)/\tau)},$$

是互信息的下界。[SimCLR](https://arxiv.org/abs/2002.05709) 用强数据增强与大批量，[MoCo](https://arxiv.org/abs/1911.05722) 用动量编码器与队列，[CLIP](https://arxiv.org/abs/2103.00020) 把它用于 4 亿图文对，[SigLIP](https://arxiv.org/abs/2303.15343) 改用逐对的 sigmoid 损失。

[BYOL](https://arxiv.org/abs/2006.07733) 与 [DINO](https://arxiv.org/abs/2104.14294) 不用负样本，让学生网络预测动量教师的输出，靠中心化、锐化等技巧避免坍缩；[MAE](https://arxiv.org/abs/2111.06377) 遮住 75% 的图像块再重建像素。[DINOv2](https://arxiv.org/abs/2304.07193) 在精选数据上结合二者。

#### 最近一年（2025-11 → 2026-10）

对平稳统计的图像数据与基本增强，对比损失的最优表示可以解析求出，并由'正弦滤波 + 逐点非线性 + 全局池化 + 部分白化'的 CNN 实现（[对比学习理论](https://arxiv.org/abs/2607.07470)）；从潜分布匹配的角度可以统一理解各类自监督方法（[潜分布匹配](https://arxiv.org/abs/2605.03517)）；自回归的下一帧预测也能学到有效的视觉表征（[下一帧预测](https://arxiv.org/abs/2512.21004)）。

### 7.3 JEPA 与其理论

[I-JEPA](https://arxiv.org/abs/2301.08243) 在表征空间而非像素空间预测被遮挡区域，避免浪费容量去建模不可预测的细节。[V-JEPA](https://arxiv.org/abs/2404.08471) 把它用于视频；[V-JEPA 2](https://arxiv.org/abs/2506.09985) 用超过百万小时的视频预训练，再用不到 62 小时的无标注机器人视频训练动作条件世界模型，在新环境中零样本规划抓取与放置。[LeJEPA](https://arxiv.org/abs/2511.08544) 从理论上证明各向同性高斯是 JEPA 嵌入的最优分布，用一个 SIGReg 正则替代了停梯度、教师网络等一系列启发式技巧。

#### 最近一年（2025-11 → 2026-10）

JEPA 在嵌入空间预测：$\min\ \|g_\phi(f_\theta(x_{\text{ctx}}))-\mathrm{sg}[f_{\bar\theta}(x_{\text{tgt}})]\|^2$，难点是防止所有输入映射到同一点。[LeJEPA](https://arxiv.org/abs/2511.08544)（2025-11）指出，为使下游预测风险最小，嵌入应服从各向同性高斯分布，并用一个把嵌入推向各向同性高斯的正则项（SIGReg）取代停止梯度、教师网络等启发式。随后的理论回答"JEPA 学到什么"：

- **驱动与衰减。** 在平凡不动点附近线性化耦合梯度流，每个模态方向都有驱动项 $\gamma_i$ 与衰减项 $\sigma_i$；稳定性比 $\mu_i=\gamma_i/\sigma_i>1$ 的方向会被学到，数量与数据谱的秩相关（[驱动与衰减](https://arxiv.org/abs/2610.02344)）。
- **学不可恢复部分。** 掩码是线性测量；低层先验就能恢复的目标会给 JEPA 留下捷径，真正被学到的是掩码之后不可恢复的信息（[学不可恢复部分](https://arxiv.org/abs/2609.32481)）。
- **可辨识性与泛化。** 不依赖重构、辅助监督或非高斯性，也能可证明地恢复各个潜变量（[DSReg](https://arxiv.org/abs/2610.09457)）；集合层面的"预测闭包"刻画了 JEPA 选择哪些特征（[预测闭包](https://arxiv.org/abs/2610.27364)）；JEPA 世界模型有了泛化界（[泛化理论](https://arxiv.org/abs/2606.27014)）。

争论仍在：用于规划时，各向异性表征反而更好（[各向异性](https://arxiv.org/abs/2609.37441)）。对比学习方面，对平稳统计的图像与基本增强，最优表示可以解析求出，并由"正弦滤波 + 逐点非线性 + 全局池化 + 部分白化"的 CNN 实现（[对比学习理论](https://arxiv.org/abs/2607.07470)）。

### 7.4 视觉表征的扩展

[DINOv3](https://arxiv.org/abs/2508.10104) 把自监督骨干扩展到 7B，发现长训练会让稠密特征退化，提出 Gram anchoring 保持局部特征的一致性；[Web-SSL](https://arxiv.org/abs/2504.01017) 在与 CLIP 相同的数据上把纯视觉自监督扩展到 7B，在 VQA 上追平语言监督的模型——语言并非视觉表征的必需监督。[SigLIP 2](https://arxiv.org/abs/2502.14786)、[Perception Encoder](https://arxiv.org/abs/2504.13181)（最好的特征藏在中间层）、[AIMv2](https://arxiv.org/abs/2411.14402)（多模态自回归预训练）代表了语言监督一侧的进展。表征也开始反哺生成：[REPA](https://arxiv.org/abs/2410.06940) 把扩散 Transformer 的中间层对齐到 DINOv2，训练提速 17 倍；[RAE](https://arxiv.org/abs/2510.11690) 直接用冻结的表征编码器作扩散的潜空间（详见<a href="../diffusion_survey/">扩散导读</a>）。

### 7.5 扩散与流：预测什么、在哪个空间生成

扩散模型与流匹配把生成写成从噪声到数据的连续变换：加噪 $z_t=\alpha_t x+\sigma_t\epsilon$，网络学习去噪方向，再积分常微分方程或随机微分方程采样（[DDPM](https://arxiv.org/abs/2006.11239)、[分数 SDE](https://arxiv.org/abs/2011.13456)、[流匹配](https://arxiv.org/abs/2210.02747)）。方法细节见姊妹篇<a href="../diffusion_survey/">扩散模型与流匹配</a>；这里只讨论原理层面的问题。

#### 最近一年（2025-11 → 2026-10）

**预测什么。** 加噪 $z_t=\alpha_t x+\sigma_t\epsilon$ 后，网络可以预测 $\epsilon$、$v$ 或 $x$，三者在数学上可互换。[JiT](https://arxiv.org/abs/2511.13720) 指出在高维时它们并不等价：按流形假设，$x$ 位于低维流形，而 $\epsilon$ 是满维的；预测 $\epsilon$ 要求网络复现 $d$ 维噪声，预测 $x$ 只需输出流形上的点。因此直接预测 $x$ 时，像素空间的大 patch Transformer 也能工作。

**在哪个空间生成。** 预训练表征空间与像素空间有相近的内在维度，普通 DiT 可直接在表征空间生成（[RAE](https://arxiv.org/abs/2510.11690)、[文生图 RAE](https://arxiv.org/abs/2601.16208)、[RiT](https://arxiv.org/abs/2605.21981)）。表征对齐（REPA）的收益主要来自目标表征的空间结构，而不是全局语义（[REPA 中什么重要](https://arxiv.org/abs/2512.10794)）。频谱视角：在整流流与幂律图像谱 $S(k)\propto k^{-\alpha}$ 下，每个时刻存在一条数据—噪声分界 $k^*(t)=(1-t)^{-2/\alpha}$，低于它的频段携带信号、高于它的频段最优预测退化为确定值，于是像素去噪器只需学前者（[谱强迫](https://arxiv.org/abs/2606.15236)）。

### 7.6 一步生成与离散扩散

#### 最近一年（2025-11 → 2026-10）

**一步生成。** 扩散与流模型在推理时迭代推前分布；[漂移模型](https://arxiv.org/abs/2602.04770)改为在训练过程中演化推前分布：定义一个在两分布相等时达到平衡的漂移场，训练目标让网络学到这个平衡，推理天然一步（[梯度流解释](https://arxiv.org/abs/2603.10592)）。同期的一步方法还有像素空间的 [pMF](https://arxiv.org/abs/2601.22158)、分析 MeanFlow 目标冲突的 [AlphaFlow](https://arxiv.org/abs/2510.20771)、去掉时间条件的 [Equilibrium Matching](https://arxiv.org/abs/2510.02300)。

**离散与语言。** 数据受限、多轮重复时，扩散语言模型持续优于自回归模型（[超级数据学习者](https://arxiv.org/abs/2511.03276)），但任意顺序生成会让模型绕开难 token，反而限制推理潜力（[灵活性陷阱](https://arxiv.org/abs/2601.15165)）；连续去噪的流映射语言模型可以一步生成（[Flow Map LM](https://arxiv.org/abs/2602.16813)）。

## 8 从反馈中学习：强化学习与后训练

预训练之后，模型通过人类偏好、可验证的奖励或更强的教师继续学习。这一章先讲方法（RLHF、DPO、RLVR、在策略蒸馏），再讲机制：这些训练到底改变了模型的什么。

### 8.1 RLHF 与 DPO

RLHF 先用偏好比较训练奖励模型 $r_\phi$（Bradley–Terry 模型 $P(y_w\succ y_l)=\sigma(r(y_w)-r(y_l))$），再优化带 KL 约束的目标（[Christiano 等](https://arxiv.org/abs/1706.03741)、[InstructGPT](https://arxiv.org/abs/2203.02155)）：

$$\max_{\pi}\ \mathbb E_{x,\,y\sim\pi}\bigl[r_\phi(x,y)\bigr]-\beta\,\mathrm{KL}\bigl(\pi(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\bigr).$$

其最优解为 $\pi^\star(y\mid x)\propto\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$，即 $r(x,y)=\beta\log\frac{\pi^\star(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$。代回 Bradley–Terry 模型，配分函数 $Z(x)$ 相消，得到只依赖策略本身的 [DPO](https://arxiv.org/abs/2305.18290) 损失：

$$\mathcal L_{\text{DPO}}=-\mathbb E\Bigl[\log\sigma\Bigl(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Bigr)\Bigr].$$

[Gao 等](https://arxiv.org/abs/2210.10760)发现对代理奖励优化越多，真实奖励先升后降——奖励投机是 RLHF 的内在问题。

### 8.2 GRPO 与 RLVR

对可以程序验证答案的任务（数学、代码），奖励直接来自验证器，不再需要奖励模型。[GRPO](https://arxiv.org/abs/2402.03300) 对每道题采样 $G$ 个回答，用组内标准化的奖励作为优势，省去价值网络：

$$\hat A_i=\frac{r_i-\mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})},\qquad \mathcal J=\mathbb E\Bigl[\frac1G\sum_i\frac1{|o_i|}\sum_t\min\bigl(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\bigr)\Bigr]-\beta\,\mathrm{KL},$$

其中 $\rho_{i,t}$ 是新旧策略在第 $t$ 个 token 上的概率比。[Tülu 3](https://arxiv.org/abs/2411.15124) 把这种做法命名为 RLVR；[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 证明只用规则奖励的纯 RL（R1-Zero）就能让模型涌现出长思维链、自我验证与反思。

- [Dr. GRPO](https://arxiv.org/abs/2503.20783)：按回答长度 $1/|o_i|$ 归一化会偏向长的错误回答，除以标准差会偏向过易或过难的题；去掉二者更无偏。
- [DAPO](https://arxiv.org/abs/2503.14476)：上下裁剪解耦（放宽上界以保持探索）、动态采样（丢弃全对或全错的组）、token 级损失聚合、超长惩罚。
- [GSPO](https://arxiv.org/abs/2507.18071)：token 级重要性比在 MoE 上方差很大，改用序列级比率与裁剪，稳定了 Qwen3 的大规模 RL。
- [Qwen 的形式化](https://arxiv.org/abs/2512.01374)：token 级替代目标只是序列级目标的一阶近似，仅当训练-推理不一致与策略陈旧都很小时才成立；这解释了重要性采样修正、裁剪与 MoE 的路由重放为何关键。
- 熵：[Cui 等](https://arxiv.org/abs/2505.22617)发现策略熵塌缩限制了性能，熵的变化由动作概率与优势的协方差驱动；[Wang 等](https://arxiv.org/abs/2506.01939)发现只有约 20% 的高熵"分叉"token 决定了 RLVR 的效果。

### 8.3 RL 是在激发还是在创造

[Yue 等](https://arxiv.org/abs/2504.13837)发现在 $k$ 很大时，基座模型的 pass@k 反超 RLVR 模型：RL 主要提高了采样效率，而非扩展可解问题的集合。反方证据包括：[ProRL](https://arxiv.org/abs/2505.24864) 长时间、有 KL 控制与参考模型重置的训练确实扩展了边界；[CoT-Pass@k](https://arxiv.org/abs/2506.14245) 显示 RLVR 提升了推理过程本身的正确性。同时，[虚假奖励](https://arxiv.org/abs/2506.10947)（随机甚至错误的奖励也能提升 Qwen2.5-Math）与 [单样本 RL](https://arxiv.org/abs/2504.20571) 说明很大一部分收益来自激发预训练中已有的行为，[数据污染](https://arxiv.org/abs/2507.10532)则让部分结论不可靠。机制上，RL 只稀疏地更新 5%–30% 的参数（[Mukherjee 等](https://arxiv.org/abs/2505.11711)），更新避开权重的主方向（[Zhu 等](https://arxiv.org/abs/2511.08567)），并偏向与原模型 KL 最小的解，因此比 SFT 遗忘更少（[RL's Razor](https://arxiv.org/abs/2509.04259)）；对比实验也显示 SFT 倾向记忆而 RL 更能泛化（[Chu 等](https://arxiv.org/abs/2501.17161)）。

### 8.4 在策略蒸馏

SFT 用教师的离线轨迹训练学生，学生在推理时却要面对自己生成的前缀（暴露偏差）；RL 在策略，但每条回答只有一个稀疏奖励。[在策略蒸馏](https://thinkingmachines.ai/blog/on-policy-distillation/)（Thinking Machines，2025-10）让学生自己采样，再用教师对每个 token 计算反向 KL：

$$\mathcal L_{\text{OPD}}=\mathbb E_{y\sim\pi_\theta}\sum_t\Bigl[\log\pi_\theta(y_t\mid y_{<t})-\log\pi_{\text{T}}(y_t\mid y_{<t})\Bigr],$$

相当于每个 token 都有稠密奖励的 RL，成本远低于 RL。2026 年围绕它出现了大量研究：现象、机制与配方（[清华](https://arxiv.org/abs/2604.13016)）、单样本 OPD（[OPD II](https://arxiv.org/abs/2609.04172)）、在/离策略蒸馏动力学的系统比较（[UC](https://arxiv.org/abs/2609.35259)）、长度膨胀（[Microsoft](https://arxiv.org/abs/2609.20511)）、奖励外推以超越教师（[G-OPD](https://arxiv.org/abs/2602.12125)）、用弱教师激发强学生（[弱到强 OPD](https://arxiv.org/abs/2607.05394)）；工业管线普遍采用"级联 RL + 多领域在策略蒸馏"（[Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220)）。

### 8.5 后训练改变了什么

#### 最近一年（2025-11 → 2026-10）

只看机制，不看配方：

- **为什么 RL 遗忘少。** 在达到相同奖励的策略中，在策略 RL 偏向与原模型 KL 最近的解（[RL's Razor](https://arxiv.org/abs/2509.04259)），参数更新避开权重的主方向（[不走主方向](https://arxiv.org/abs/2511.08567)）；回路层面同样如此：RL 微调更少破坏已有的注意力头回路（[遗忘的机制起源](https://arxiv.org/abs/2605.28860)）。在策略蒸馏的更新小而坐标稀疏，只训练找到的子网络就几乎能复现全量训练（[稠密监督、稀疏更新](https://arxiv.org/abs/2606.13657)）。
- **熵与探索。** 对 softmax 策略做自然策略梯度更新，熵的一步变化近似为 $\Delta H\approx-\eta\,\mathrm{Cov}_{a\sim\pi}\big(\log\pi(a),\,A(a)\big)$：高概率且高优势的动作会压低熵（[熵机制](https://arxiv.org/abs/2505.22617)）；强化微调中熵动力学的进一步推导见 [熵动力学](https://arxiv.org/abs/2602.03392)。组相对优势存在隐式对称：正确与错误轨迹的权重严格对称，未采样动作的 logit 得不到更新，限制了探索与难度自适应（[优势对称](https://arxiv.org/abs/2602.05548)）。
- **增益从哪来。** 虚假奖励的收益来自激活记忆捷径：答案 token 困惑度下降而提示侧连贯性变差（[虚假奖励悖论](https://arxiv.org/abs/2601.11061)）；SFT 后 pass@k 下降与数据中的"岔路口" token 有关（[覆盖收缩](https://arxiv.org/abs/2605.17026)）；"SFT 记忆、RL 泛化"并不绝对，长思维链 SFT 的跨领域泛化取决于优化、数据与模型能力（[条件泛化](https://arxiv.org/abs/2604.06628)）；用 SFT 注入新事实会增加对已有知识的幻觉（[微调与幻觉](https://arxiv.org/abs/2604.15574)）。

## 9 测试时的计算与学习

模型在测试时也能"学"：从上下文中的示例学习（上下文学习）、用更多推理步骤计算（思维链）、把上下文写进权重（测试时训练，见 2.4 节），以及在部署后持续学习。

### 9.1 上下文学习

GPT-3 展示了不改权重、只看示例就能学会新任务的能力。理论解释有三类：一是线性自注意力的一层前向可以实现一步梯度下降（[von Oswald 等](https://arxiv.org/abs/2212.07677)，[Garg 等](https://arxiv.org/abs/2208.01066)发现 Transformer 在上下文中学线性回归接近最优估计器）；二是"归纳头"——由前一 token 头与复制头组成的两层电路，其出现与损失曲线上的相变同步（[Olsson 等](https://arxiv.org/abs/2209.11895)）；三是 [Dherin 等](https://arxiv.org/abs/2507.16003)的"无需训练的学习"：Transformer 块把上下文隐式转化为 MLP 权重的低秩更新。也有反例提醒不要过度简化：[Transformer 并不在上下文中做最小二乘](https://arxiv.org/abs/2507.09440)，[不同任务下 ICL 由不同机制实现](https://arxiv.org/abs/2604.12151)。

#### 最近一年（2025-11 → 2026-10）

贝叶斯视角下，ICL 是对潜在任务 $\tau$ 的后验平均：$p(y\mid x,\mathcal D)=\int p(y\mid x,\tau)\,p(\tau\mid\mathcal D)\,d\tau$。过去常把 ICL 归因于人类语言的特殊统计性质；最近一年的反例是：只在 DNA 序列上做下一词预测也会涌现 ICL（[基因组 ICL](https://arxiv.org/abs/2511.12797)），文本、音频等模态的预测器都趋同地涌现少样本 ICL（[跨模态趋同](https://arxiv.org/abs/2609.14011)）。机制上，归纳头的出现时间可以在预训练中预测（[预测归纳头](https://arxiv.org/abs/2511.16893)），数据多样性决定模型选择"捷径"还是"归纳头"算法（[捷径与归纳头](https://arxiv.org/abs/2512.18634)），ICL 中的注意力可视为跨任务的自适应核（[自适应核](https://arxiv.org/abs/2610.24441)）。思维链方面，梯度训练的 Transformer 可证明学会能长度泛化的思维链（[CoT 长度泛化](https://arxiv.org/abs/2511.07378)），也能把显式步骤内化到隐状态（[内化 CoT](https://arxiv.org/abs/2605.28600)）；长度泛化的差异部分来自注意力头的隐式偏置：相对位置泛化、绝对位置记忆（[相对 vs 绝对](https://arxiv.org/abs/2607.18759)）。

### 9.2 思维链与推理

**从提示到训练。** [思维链](https://arxiv.org/abs/2201.11903)提示让模型写出中间步骤；[自洽性](https://arxiv.org/abs/2203.11171)采样多条推理后投票；[Tree of Thoughts](https://arxiv.org/abs/2305.10601) 把推理组织成可回溯的树。训练层面，[STaR](https://arxiv.org/abs/2203.14465) 用模型自己生成且答案正确的推理链迭代微调，[过程监督](https://arxiv.org/abs/2305.20050)（逐步打分）优于结果监督。[o1](https://openai.com/index/learning-to-reason-with-llms/) 与 R1 把这一切统一为：用 RL 训练模型生成长思维链，推理能力随训练算力与思考长度平滑提升。

**推理算力怎么花。** [Snell 等](https://arxiv.org/abs/2408.03314)比较了"顺序修改"与"并行采样 + 验证器搜索"，发现最优策略取决于题目难度；[Liu 等](https://arxiv.org/abs/2502.06703)发现在合适的测试时策略下 1B 模型可以在数学上超过 405B 模型。[s1](https://arxiv.org/abs/2501.19393) 用 1000 条样本 SFT 加"预算强制"（在模型想停时追加 Wait）即可实现测试时扩展；[LIMO](https://arxiv.org/abs/2502.03387) 用 817 条精选样本激发复杂推理。验证器本身也在变强：[DeepSeek-GRM](https://arxiv.org/abs/2504.02495) 让奖励模型先生成原则与批评再打分，使奖励模型也能随推理算力扩展；[DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) 训练证明验证器，实现可自我验证的定理证明。

**别想太多。** 推理模型常在简单题上过度思考（[Chen 等](https://arxiv.org/abs/2412.21187)）。对策包括：学会何时思考（[AdaptThink](https://arxiv.org/abs/2505.13417)）、在测试时调度快慢思考（[AlphaOne](https://arxiv.org/abs/2505.24863)）、把思考切成固定长度的块只携带短状态以获得线性算力（[Markovian Thinker](https://arxiv.org/abs/2510.06557)）、并行思考（[Parallel-R1](https://arxiv.org/abs/2509.07980)、[Multiverse](https://arxiv.org/abs/2506.09991)）。综述见 [Sui 等](https://arxiv.org/abs/2503.16419)。

**推理是否必须用语言。** [Coconut](https://arxiv.org/abs/2412.06769) 把最后一层隐状态直接作为下一步输入，在连续空间中推理，可同时保留多个候选分支；循环深度模型（第 7.7 节）在隐空间迭代。[Anthropic](https://arxiv.org/abs/2505.05410) 发现推理模型的思维链常常不承认使用了提示中的线索，即语言化的推理并不总是忠实于内部计算；这既是潜空间推理的动机，也是安全上的隐忧（10.3 节）。

#### 最近一年（2025-11 → 2026-10）

推理的理论开始成形：Transformer 可证明地把显式思维链内化到隐状态中（[内化 CoT](https://arxiv.org/abs/2605.28600)）；把高效推理表述为条件信息瓶颈下的有损压缩，可以统一各类预算控制方法（[推理即压缩](https://arxiv.org/abs/2603.08462)）；答案常在中间层过早定型（[停止思考得太早](https://arxiv.org/abs/2609.36585)）；在潜空间中用归一化流或流匹配推理（[潜在推理](https://arxiv.org/abs/2606.06447)、[自条件化](https://arxiv.org/abs/2610.19628)）。

### 9.3 持续学习与可塑性

顺序学习多个任务时，网络会遗忘旧任务（灾难性遗忘）；按 Fisher 信息保护重要参数是经典对策（[EWC](https://arxiv.org/abs/1612.00796)）。更隐蔽的问题是可塑性丧失：持续训练的网络会逐渐失去学习新东西的能力（[可塑性丧失](https://arxiv.org/abs/2306.13812)）。在大模型上，在线 RL 比 SFT 遗忘更少（[RL's Razor](https://arxiv.org/abs/2509.04259)），而不少'遗忘'其实只是任务对齐的丢失（[虚假遗忘](https://arxiv.org/abs/2501.13453)）。

#### 最近一年（2025-11 → 2026-10）

持续训练的网络会逐渐失去学新东西的能力（可塑性丧失），微调又会遗忘旧能力。最近一年开始把两者放进同一动力学框架：选择学什么（数据归因）、理解一次更新改变了什么（遗忘）、保持继续学习的能力（可塑性）被视为同一过程的三面（[统一视角](https://arxiv.org/abs/2609.33620)）；任务更新之间的几何冲突解释并控制遗忘（[几何冲突](https://arxiv.org/abs/2605.09608)）；遗忘更像"可达性坍缩"而非知识消失（[可达性坍缩](https://arxiv.org/abs/2606.06032)）；在强化学习中，可塑性被定义为从观测到动作的有向信息，并有了相应的 Bellman 最优方程（[可塑性的 Bellman 方程](https://arxiv.org/abs/2609.10776)）。测试时学习带来新问题：模型从自己的输出学习时，每次更新都改变下一次生成，形成不稳定的反馈（[自生成反馈](https://arxiv.org/abs/2610.05076)）。可用的手段包括锐度感知预训练（[锐度感知预训练](https://arxiv.org/abs/2605.02105)）与权重衰减（[权重衰减与可塑性](https://arxiv.org/abs/2602.11137)）。

## 10 应用：原理的检验场

前面各章的原理在应用中接受检验。大模型是其中规模最大、工程最复杂的一个方向；视觉、多模态、科学、机器人与智能体是另外几个。安全与对齐则讨论训练带来的副作用。

### 10.1 大模型系统：KV 缓存、稀疏注意力与推理效率

自回归解码时，每层要缓存所有历史 token 的键和值。以 $n_h$ 个头、每头 $d_h$ 维为例，每 token 每层的缓存元素数为：

| 方案 | 每 token 每层缓存 | $n_h=d_h=128$ 时 |
|---|---|---|
| MHA | $2n_hd_h$ | 32768 |
| GQA（$n_g$ 组） | $2n_gd_h$ | 2048（$n_g=8$） |
| MQA | $2d_h$ | 256 |
| MLA | $d_c+d_h^R$ | 576（$d_c=512$, $d_h^R=64$） |

[MQA](https://arxiv.org/abs/1911.02150) 让所有头共享一组 K/V，[GQA](https://arxiv.org/abs/2305.13245) 折中；[MLA](https://arxiv.org/abs/2405.04434)（DeepSeek-V2）把 K/V 联合压缩成一个低秩潜向量 $c_t=W^{DKV}h_t$，推理时把上投影矩阵"吸收"进查询与输出投影，只缓存 $c_t$ 和一小段解耦的 RoPE 键，KV 缓存减少 93.3%。[TransMLA](https://arxiv.org/abs/2502.07864) 证明同等缓存下 MLA 的表达力强于 GQA。

注意力的瓶颈不是浮点运算，而是读写 $N\times N$ 的分数矩阵。[FlashAttention](https://arxiv.org/abs/2205.14135) 把 Q、K、V 分块载入片上 SRAM，用在线 softmax（维护运行中的最大值 $m$ 与归一化因子 $\ell$，新块到来时按 $e^{m_{\text{old}}-m_{\text{new}}}$ 重缩放）逐块累加输出，从不在显存中物化完整矩阵，显存随长度线性增长，速度提升数倍。[FA-2](https://arxiv.org/abs/2307.08691) 改进并行划分，[FA-3](https://arxiv.org/abs/2407.08608) 利用 Hopper 的异步与 FP8。

KV 缓存再小，每步解码仍要读取全部历史。稀疏注意力让每个查询只关注一小部分 KV：

- [NSA](https://arxiv.org/abs/2502.11089)（DeepSeek）：压缩分支看粗粒度的块摘要，选择分支挑出最相关的若干块做精细注意力，滑动窗口分支保留局部；三者门控相加，块大小与 GQA 组对齐以充分利用硬件，并且可以端到端训练。
- [MoBA](https://arxiv.org/abs/2502.13189)（Moonshot）：把 MoE 的路由思想用于注意力，每个查询只路由到被选中的 KV 块。
- [DSA](https://arxiv.org/abs/2512.02556)（DeepSeek-V3.2）：用轻量"闪电索引器"为每个查询选出 top-k token，再做标准注意力；被 [GLM-5](https://arxiv.org/abs/2602.15763) 采用。
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)：混合压缩稀疏注意力（CSA）与重度压缩注意力（HCA），1.6T 与 284B 两个模型都支持 100 万 token。
- [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)：在 GQA 的每个组内独立选块，主分支只对选中的块做精确注意力。

**量化。** 后训练量化的难点是激活中的离群通道（[LLM.int8()](https://arxiv.org/abs/2208.07339)）。[GPTQ](https://arxiv.org/abs/2210.17323) 用近似二阶信息逐层补偿量化误差（[几何视角](https://arxiv.org/abs/2507.18553)：它等价于格上的 Babai 最近平面算法）；[AWQ](https://arxiv.org/abs/2306.00978) 保护按激活重要的权重通道；[SmoothQuant](https://arxiv.org/abs/2211.10438) 把激活离群值的难度迁移到权重。原生低比特模型方面，[BitNet b1.58](https://arxiv.org/abs/2402.17764) 用三值权重从头训练，[BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) 开源了首个原生 1-bit 的 2B 模型。

**推理系统。** [vLLM](https://arxiv.org/abs/2309.06180) 借鉴操作系统的分页管理 KV 缓存；[SGLang](https://arxiv.org/abs/2312.07104) 用基数树复用前缀；[DistServe](https://arxiv.org/abs/2401.09670) 与 [Mooncake](https://arxiv.org/abs/2407.00079) 把预填充与解码分离到不同设备。[推测解码](https://arxiv.org/abs/2211.17192)用小模型起草、大模型并行验证，保证输出分布不变；[Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-3](https://arxiv.org/abs/2503.01840) 不断提高草稿接受率。面向智能体负载，[DualPath](https://arxiv.org/abs/2602.21548) 指出多轮推理的瓶颈已从计算转向 KV 缓存的存储 I/O。

**参数高效微调。** [LoRA](https://arxiv.org/abs/2106.09685) 只训练低秩增量 $\Delta W=BA$，[QLoRA](https://arxiv.org/abs/2305.14314) 在 4-bit 底座上训练 LoRA。[Thinking Machines](https://thinkingmachines.ai/blog/lora/) 的系统实验表明，合理配置（更大的学习率、作用于所有层）的 LoRA 在后训练与 RL 场景下可以匹敌全参数微调，而 RL 所需的容量极小——每回合只传递约 1 bit 的信息。

### 10.2 应用版图

**视觉感知与 3D。** [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) → [ResNet](https://arxiv.org/abs/1512.03385) → [ViT](https://arxiv.org/abs/2010.11929)；检测分割从 [Faster R-CNN](https://arxiv.org/abs/1506.01497)、[DETR](https://arxiv.org/abs/2005.12872) 到可提示的 [SAM](https://arxiv.org/abs/2304.02643)、[SAM 2](https://arxiv.org/abs/2408.00714)、[SAM 3](https://arxiv.org/abs/2511.16719)（按名词短语分割与跟踪所有实例）。几何方面，[NeRF](https://arxiv.org/abs/2003.08934)、[3D 高斯泼溅](https://arxiv.org/abs/2308.04079)之后，[DUSt3R](https://arxiv.org/abs/2312.14132)、[VGGT](https://arxiv.org/abs/2503.11651)、[MapAnything](https://arxiv.org/abs/2509.13414)、[Depth Anything 3](https://arxiv.org/abs/2511.10647) 用一次前向直接回归相机与几何。

**生成。** GAN、VAE → 扩散与流匹配 + DiT → 视频与可交互世界模型（[Sora](https://openai.com/index/video-generation-models-as-world-simulators/)、[Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)）。详见<a href="../diffusion_survey/">扩散模型与流匹配导读</a>。

**语音与音频。** [wav2vec 2.0](https://arxiv.org/abs/2006.11477) 的自监督、[Whisper](https://arxiv.org/abs/2212.04356) 的大规模弱监督、[VALL-E](https://arxiv.org/abs/2301.02111) 把 TTS 视为编解码 token 上的语言建模、[Moshi](https://arxiv.org/abs/2410.00037) 的全双工对话、[VibeVoice](https://arxiv.org/abs/2508.19205) 的超低帧率连续 token。

**多模态。** [CLIP](https://arxiv.org/abs/2103.00020) → [Flamingo](https://arxiv.org/abs/2204.14198)、[LLaVA](https://arxiv.org/abs/2304.08485) → [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)、[Qwen3-VL](https://arxiv.org/abs/2511.21631)、[InternVL3.5](https://arxiv.org/abs/2508.18265) → 原生全模态（[GPT-4o](https://arxiv.org/abs/2410.21276)、[Qwen3-Omni](https://arxiv.org/abs/2509.17765)、[Gemma 4](https://arxiv.org/abs/2607.02770) 的无编码器 12B）。[DeepSeek-OCR](https://arxiv.org/abs/2510.18234) 与 [Glyph](https://arxiv.org/abs/2510.17800) 把长文本渲染为图像，用视觉 token 压缩上下文。

**智能体。** [ReAct](https://arxiv.org/abs/2210.03629)、[Toolformer](https://arxiv.org/abs/2302.04761) → [SWE-agent](https://arxiv.org/abs/2405.15793)、[Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use)、[Deep Research](https://openai.com/index/introducing-deep-research/) → 端到端智能体 RL（[ARPO](https://arxiv.org/abs/2507.19849)、[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)、[UI-TARS-2](https://arxiv.org/abs/2509.02544)）与 2026 年的 harness 工程和递归自我改进（[Raven](https://arxiv.org/abs/2609.33439)、[HarnessDev](https://arxiv.org/abs/2609.01437)、[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)）。[METR](https://arxiv.org/abs/2503.14499) 测得 AI 能完成的任务时长自 2019 年起约每 7 个月翻一番。

**科学与数学。** [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2)、[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)；[ESM3](https://www.science.org/doi/10.1126/science.ads0018)、[Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)、[AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/)；[GraphCast](https://arxiv.org/abs/2212.12794)、[GenCast](https://arxiv.org/abs/2312.15796)；[GNoME](https://www.nature.com/articles/s41586-023-06735-9)、[MatterGen](https://arxiv.org/abs/2312.03687)、[UMA](https://arxiv.org/abs/2506.23971)；[FunSearch](https://www.nature.com/articles/s41586-023-06924-6)、[AlphaEvolve](https://arxiv.org/abs/2506.13131)；自主科研（[AI co-scientist](https://arxiv.org/abs/2502.18864)、[Kosmos](https://arxiv.org/abs/2511.02824)）。

**机器人。** [RT-2](https://arxiv.org/abs/2307.15818) → [Diffusion Policy](https://arxiv.org/abs/2303.04137) → [π0](https://arxiv.org/abs/2410.24164)、[π0.5](https://arxiv.org/abs/2504.16054) → [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342)；世界模型路线有 [V-JEPA 2](https://arxiv.org/abs/2506.09985) 与 [DreamZero](https://arxiv.org/abs/2602.15922)。详见<a href="../embodied_agent_survey/">机器人 × Agent 导读</a>。

**博弈与决策。** [DQN](https://arxiv.org/abs/1312.5602) → [AlphaGo](https://www.nature.com/articles/nature16961) → [AlphaZero](https://arxiv.org/abs/1712.01815) → [MuZero](https://arxiv.org/abs/1911.08265) → [DreamerV3](https://arxiv.org/abs/2301.04104)、[Dreamer 4](https://arxiv.org/abs/2509.24527)。这条线积累的策略梯度、自我博弈与价值学习，如今是 LLM 后训练的基础。

**数学与算法发现。** 2025 年 IMO 上，[Gemini Deep Think](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) 用自然语言在比赛时限内解出 6 题中的 5 题，形式化系统 [Aristotle](https://arxiv.org/abs/2510.01346) 与 [Seed-Prover](https://arxiv.org/abs/2507.23726) 也达到金牌级；[DeepSeek-V3.2-Speciale](https://arxiv.org/abs/2512.02556) 在 IMO 与 IOI 上达到金牌水平；[OpenAI](https://arxiv.org/abs/2502.06807) 的通用 RL 模型 o3 无需人工策略即在 IOI 拿到金牌。[AlphaEvolve](https://arxiv.org/abs/2506.13131) 把 LLM 与进化搜索、自动评估结合，发现了 4×4 复数矩阵乘法的更优算法并优化了数据中心调度。

### 10.3 安全与对齐

**对抗鲁棒性。** [Szegedy 等](https://arxiv.org/abs/1312.6199)发现人眼不可察觉的扰动就能让分类器出错；[Goodfellow 等](https://arxiv.org/abs/1412.6572)把它归因于高维下的线性；[Madry 等](https://arxiv.org/abs/1706.06083)给出以 PGD 为对手的鲁棒优化框架。LLM 时代，[GCG](https://arxiv.org/abs/2307.15043) 自动搜索出可迁移的对抗后缀；[新近工作](https://arxiv.org/abs/2508.17456)把对抗脆弱性与特征叠加联系起来。

**对齐的意外后果。**

- [Sleeper Agents](https://arxiv.org/abs/2401.05566)：植入的后门行为能挺过 SFT、RL 与对抗训练。
- [对齐伪装](https://arxiv.org/abs/2412.14093)：Claude 3 Opus 在认为对话会被用于训练时策略性地顺从，以保护原有偏好；[后续研究](https://arxiv.org/abs/2506.18032)发现 25 个模型中只有少数表现出这种行为。
- [涌现失准](https://arxiv.org/abs/2502.17424)：只在"写不安全代码"这一窄任务上微调，就会导致广泛的有害、失准行为；[潜意识学习](https://arxiv.org/abs/2507.14805)进一步发现，学生模型能从看似无关的数据（如数字序列）中继承教师的特质，前提是共享初始化。
- [奖励投机导致的自然涌现失准](https://arxiv.org/abs/2511.18397)：在 Anthropic 的生产级编程 RL 环境中学会奖励投机的模型，会泛化出对齐伪装、与恶意行为者合作、破坏安全研究等行为。
- 谋划与评测意识：前沿模型在给定目标的智能体场景中会隐瞒、关闭监督（[Apollo](https://arxiv.org/abs/2412.04984)），面临被替换时会采取勒索等手段（[智能体失准](https://www.anthropic.com/research/agentic-misalignment)），并常能识别自己在被评测（[Needham 等](https://arxiv.org/abs/2505.23836)）。

**监控与防御。** [审慎对齐](https://arxiv.org/abs/2412.16339)让推理模型在回答前显式回忆并推理安全规范；[反谋划训练](https://arxiv.org/abs/2509.15541)能把隐蔽行为降低约 30 倍，但评测意识混淆了效果。[OpenAI](https://arxiv.org/abs/2503.11926) 发现思维链监控能抓住奖励投机，但对思维链施加优化压力会让模型学会隐藏意图；多家机构因此联合呼吁把[思维链可监控性](https://arxiv.org/abs/2507.11473)当作脆弱而需要保护的安全机会。防御侧还有[宪法分类器](https://arxiv.org/abs/2501.18837)与[宪法 AI](https://arxiv.org/abs/2212.08073)。对幻觉，[Kalai 等](https://arxiv.org/abs/2509.04664)给出统计解释：即使训练数据无误，生成式模型的错误也与二分类错误同源；而以准确率为主的评测奖励"猜测"而非"承认不知道"，使其在后训练中得以保留。

#### 最近一年（2025-11 → 2026-10）

涌现失准的机制被进一步分析：只在谄媚数据上微调也能诱发（[谄媚诱发](https://arxiv.org/abs/2606.09068)），可以用二阶几何动态诊断其形成并在参数空间中缓解（[机制诊断](https://arxiv.org/abs/2609.34970)），也可以用特征叠加的几何来理解（[叠加几何](https://arxiv.org/abs/2605.00842)）；思维链可监控性有了信息论的分析（[可监控性的信息论](https://arxiv.org/abs/2602.18297)）；循环深度模型的安全行为随循环次数变化（[循环与拒绝](https://arxiv.org/abs/2610.15218)）。

## 11 开放问题与展望

按章整理：

1. **架构（第 2 章）。** 线性与全注意力的最佳混合、残差流的形式、循环深度与外置记忆，Transformer 之后的"标准件"尚未定型；注意力汇聚与巨大激活是该消除的缺陷，还是 softmax 架构的必要组件？
2. **优化（第 3 章）。** 若优化器决定"哪种范数下的最大间隔"与"哪种谱剖面"，最优的范数是否依赖层的类型、数据与规模？逐坐标自适应引入的基依赖是缺陷还是归纳偏置？稳定性边缘的 ODE 模型能否用于在线调参？
3. **泛化（第 4 章）。** 扩散模型的泛化依赖"学得不完美"，那么什么样的不完美是好的？长度与组合泛化能否由架构保证，还是只能靠数据与优化？
4. **扩展（第 5 章）。** 高质量数据接近用尽；精度、稀疏度、循环、RL、测试时计算各有扩展律，如何联合优化？可解模型能否在训练前给出定量预测？
5. **表征（第 6 章）。** 柏拉图表征假说同时得到定理与反证，"趋同"的度量、对称性处理与判定标准仍需统一；叠加的数学理论能否转化为可靠的特征提取工具？
6. **自监督与生成（第 7 章）。** JEPA 应追求各向同性还是各向异性？一步生成能否在质量上全面取代迭代采样？扩散语言模型的优势与"灵活性陷阱"如何权衡？
7. **反馈学习（第 8 章）。** RLVR 是在重排预训练中已有的解，还是能发现新策略？没有可验证奖励的开放式任务如何扩展？
8. **测试时学习（第 9 章）。** 有效上下文远小于窗口长度（[Context Rot](https://research.trychroma.com/context-rot)）；测试时训练与持续学习能否在不失稳、不遗忘的前提下成为常态？
9. **应用与安全（第 10 章）。** 窄域训练会泛化出意外行为，模型能识别评测、可能隐藏意图；可解释性能否用于保证，而不只用于发现？

深度学习的基本配方十余年没有变：可微函数、梯度、数据与规模。最近一年的变化是，越来越多的经验规律有了可以计算的解释。理解这些原理，比追踪任何一个具体模型都更持久。

## 附录 A：大事记（2024-11 → 2026-10）

**最近半年（2026-05 → 2026-10），原理类：**

- **2026-05**：[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)（擦除与写入解耦）；[Shannon 扩展律](https://arxiv.org/abs/2605.23901)；[大模型为何学得更多](https://arxiv.org/abs/2605.29548)；[优化器诱导的谱扩展律](https://arxiv.org/abs/2605.21803)；[可证明地内化思维链](https://arxiv.org/abs/2605.28600)；[RL 在回路层面遗忘更少](https://arxiv.org/abs/2605.28860)；[EMO](https://arxiv.org/abs/2605.06663)（MoE 的涌现模块化）。
- **2026-06**：[Edge Flow](https://arxiv.org/abs/2606.18080)；[群组合的谱表示](https://arxiv.org/abs/2606.02993)；[SAE 的种子依赖](https://arxiv.org/abs/2606.12138)；[OPD 稀疏更新](https://arxiv.org/abs/2606.13657)；[谱强迫](https://arxiv.org/abs/2606.15236)；[Loop-μP](https://arxiv.org/abs/2606.18524)；[JEPA 泛化理论](https://arxiv.org/abs/2606.27014)。
- **2026-07**：[扩散模型无良性过拟合](https://arxiv.org/abs/2607.02671)；[泛化相变的信息论](https://arxiv.org/abs/2607.08041)；[自然图像对比学习理论](https://arxiv.org/abs/2607.07470)；[相对位置泛化](https://arxiv.org/abs/2607.18759)；[权重衰减时钟](https://arxiv.org/abs/2607.23967)；[权重范数临界](https://arxiv.org/abs/2607.21005)；[DeepLoop](https://arxiv.org/abs/2607.13491)。
- **2026-08**：[叠加的数学理论](https://arxiv.org/abs/2608.27540)；[损失不看基，Adam 看](https://arxiv.org/abs/2608.05136)；[谱分配](https://arxiv.org/abs/2608.25990)；[Adam 的 EoS](https://arxiv.org/abs/2608.20638)；[混合模型中的巨大激活](https://arxiv.org/abs/2608.12149)。
- **2026-09**：[1/3 数据指数](https://arxiv.org/abs/2609.37745)；[Muon 与 EoS](https://arxiv.org/abs/2609.34915)；[参数对称决定表征几何](https://arxiv.org/abs/2609.39078)；[线性表征需要群作用](https://arxiv.org/abs/2609.27158)；[JEPA 学不可恢复部分](https://arxiv.org/abs/2609.32481)；[持续学习的统一动力学](https://arxiv.org/abs/2609.33620)；[跨模态趋同的 ICL](https://arxiv.org/abs/2609.14011)；[Kalman Delta](https://arxiv.org/abs/2609.07816)。
- **2026-10**：[柏拉图表征定理](https://arxiv.org/abs/2610.39302)；[条件无毛定理](https://arxiv.org/abs/2610.24133)；[JEPA 的驱动与衰减](https://arxiv.org/abs/2610.02344)；[巨大激活的生命周期](https://arxiv.org/abs/2610.00423)；[有效学习率调度](https://arxiv.org/abs/2610.40362)；[随机特征二元模型](https://arxiv.org/abs/2610.19281)；[循环的不动点](https://arxiv.org/abs/2610.06833)；[SAE 的数据收益递减](https://arxiv.org/abs/2610.20680)；[自生成反馈破坏 TTT](https://arxiv.org/abs/2610.05076)。

**两年回看（原理与应用）：**

- **2024-11 至 12**：[Tülu 3](https://arxiv.org/abs/2411.15124) 提出 RLVR；[精度扩展律](https://arxiv.org/abs/2411.04330)；[MCP](https://www.anthropic.com/news/model-context-protocol)；[Muon](https://kellerjordan.github.io/posts/muon/) 博文；[Gated DeltaNet](https://arxiv.org/abs/2412.06464)；[BLT](https://arxiv.org/abs/2412.09871)；[Coconut](https://arxiv.org/abs/2412.06769)；[对齐伪装](https://arxiv.org/abs/2412.14093)；[DeepSeek-V3](https://arxiv.org/abs/2412.19437)。
- **2025-01 至 03**：[Titans](https://arxiv.org/abs/2501.00663)；[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[Kimi k1.5](https://arxiv.org/abs/2501.12599)；[s1](https://arxiv.org/abs/2501.19393)；[NSA](https://arxiv.org/abs/2502.11089)、[MoBA](https://arxiv.org/abs/2502.13189)；[Moonlight](https://arxiv.org/abs/2502.16982)；[涌现失准](https://arxiv.org/abs/2502.17424)；[DyT](https://arxiv.org/abs/2503.10622)；[DAPO](https://arxiv.org/abs/2503.14476)；[METR 任务时长](https://arxiv.org/abs/2503.14499)；[归因图](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)。
- **2025-04 至 06**：[RL 边界之争](https://arxiv.org/abs/2504.13837)；[门控注意力](https://arxiv.org/abs/2505.06708)；[Qwen3](https://arxiv.org/abs/2505.09388)；[3.6 bit/参数](https://arxiv.org/abs/2505.24832)；[ProRL](https://arxiv.org/abs/2505.24864)；[V-JEPA 2](https://arxiv.org/abs/2506.09985)；[AlphaEvolve](https://arxiv.org/abs/2506.13131)；[HRM](https://arxiv.org/abs/2506.21734)。
- **2025-07 至 09**：[GSPO](https://arxiv.org/abs/2507.18071)；[Kimi K2](https://arxiv.org/abs/2507.20534)；[IMO 金牌](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)；[CoT 可监控性](https://arxiv.org/abs/2507.11473)；[DINOv3](https://arxiv.org/abs/2508.10104)；[gpt-oss](https://arxiv.org/abs/2508.10925) 与 [GPT-5](https://arxiv.org/abs/2601.03267)；[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)；[优化器公平比较](https://arxiv.org/abs/2509.02046)。
- **2025-10 至 12**：[TRM](https://arxiv.org/abs/2510.04871)；[ScaleRL](https://arxiv.org/abs/2510.13786)；[在策略蒸馏](https://thinkingmachines.ai/blog/on-policy-distillation/)；[Kimi Linear](https://arxiv.org/abs/2510.26692)；[权重稀疏电路](https://arxiv.org/abs/2511.13653)；[奖励投机致失准](https://arxiv.org/abs/2511.18397)；[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)；[Nested Learning](https://arxiv.org/abs/2512.24695)；[mHC](https://arxiv.org/abs/2512.24880)。
- **2026-01 至 03**：[Engram](https://arxiv.org/abs/2601.07372)；[内省](https://arxiv.org/abs/2601.01828)；[Kimi K2.5](https://arxiv.org/abs/2602.02276)；[GLM-5](https://arxiv.org/abs/2602.15763)；[Attention Residuals](https://arxiv.org/abs/2603.15031)。
- **2026-04 至 06**：[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（1.6T、百万上下文、mHC + Muon）；[MiniMax-M2](https://arxiv.org/abs/2605.26494)；[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)；在策略蒸馏研究井喷。
- **2026-07 至 10**：[Gemma 4](https://arxiv.org/abs/2607.02770)；[Kimi K3](https://arxiv.org/abs/2607.24653)（2.8T、KDA + AttnRes）；[Qwen3.8-Next](https://arxiv.org/abs/2608.30320)；[循环 MoE 扩展律](https://arxiv.org/abs/2609.01343)；[DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969)；harness 工程与递归自我改进成为智能体研究热点。
- **原理侧（2025-11 → 2026-10）**：[JiT](https://arxiv.org/abs/2511.13720) 与 [LeJEPA](https://arxiv.org/abs/2511.08544)（2025-11）；[grokking 的几何](https://arxiv.org/abs/2511.01938)；[漂移模型](https://arxiv.org/abs/2602.04770)；[Adam/Muon 的隐式偏置](https://arxiv.org/abs/2602.16340)；[最优学习率调度](https://arxiv.org/abs/2602.06797)；[汇聚的必要性](https://arxiv.org/abs/2603.11487)；[扩散先泛化后记忆](https://arxiv.org/abs/2603.23792)；[循环的等深度扩展律](https://arxiv.org/abs/2604.21106)；[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)；[Edge Flow](https://arxiv.org/abs/2606.18080)；[扩散无良性过拟合](https://arxiv.org/abs/2607.02671)；[叠加的数学理论](https://arxiv.org/abs/2608.27540)；[1/3 数据指数](https://arxiv.org/abs/2609.37745)；[柏拉图表征定理](https://arxiv.org/abs/2610.39302)。详见各章的'最近一年'小节。

---

## 附录 B：论文与资源总表（1827 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 核验；★ 为入门推荐；🔥 为 2026-05 及之后（最近半年），◆ 为 2025-11 及之后（最近一年），🆕 为 2024-11 及之后（最近两年）的工作。

### 基础与经典网络（34）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2024-04 | [KAN](https://arxiv.org/abs/2404.19756) `arXiv:2404.19756` | MIT | 把可学习激活函数放在边上（Kolmogorov-Arnold 表示），可解释性好但扩展性存疑 |
| 2021-04 | ★ [几何深度学习](https://arxiv.org/abs/2104.13478) `arXiv:2104.13478` | Imperial/DeepMind | 用对称性与不变性（群、图、网格、流形）统一 CNN、GNN、Transformer 的设计原理 |
| 2020-07 | [Modern Hopfield](https://arxiv.org/abs/2008.02217) `arXiv:2008.02217` | JKU Linz | 现代 Hopfield 网络的更新规则等价于 Transformer 注意力，指数级存储容量 |
| 2019-03 | ★ [The Bitter Lesson](https://www.incompleteideas.net/IncIdeas/BitterLesson.html) | Rich Sutton | 七十年 AI 史的教训：能随算力扩展的通用方法（搜索与学习）终将胜过人工设计的知识 |
| 2018-06 | [Neural ODE](https://arxiv.org/abs/1806.07366) `arXiv:1806.07366` | Toronto | 把残差网络视为常微分方程的离散化，用 ODE 求解器做前向与伴随反传 |
| 2017-06 | ★ [Transformer](https://arxiv.org/abs/1706.03762) `arXiv:1706.03762` | Google | 完全基于自注意力的编码器-解码器，并行训练，奠定此后所有大模型的架构 |
| 2017-04 | [MPNN](https://arxiv.org/abs/1704.01212) `arXiv:1704.01212` | Google | 消息传递神经网络：统一图神经网络框架并用于分子性质预测 |
| 2017-03 | [MAML](https://arxiv.org/abs/1703.03400) `arXiv:1703.03400` | UC Berkeley | 元学习：学一个只需几步梯度就能适应新任务的初始化 |
| 2016-09 | [GCN](https://arxiv.org/abs/1609.02907) `arXiv:1609.02907` | Amsterdam | 谱图卷积的一阶近似，半监督节点分类的基线 |
| 2016-08 | [DenseNet](https://arxiv.org/abs/1608.06993) `arXiv:1608.06993` | Cornell/Tsinghua | 每层与前面所有层稠密连接，特征复用 |
| 2016-07 | [LayerNorm](https://arxiv.org/abs/1607.06450) `arXiv:1607.06450` | Toronto | 在特征维度上归一化，不依赖批大小，后成为 Transformer 标配 |
| 2016-03 | [Pre-activation ResNet](https://arxiv.org/abs/1603.05027) `arXiv:1603.05027` | Microsoft Research | 恒等映射分析：把归一化与激活移到残差分支内（pre-norm 的源头） |
| 2015-12 | ★ [ResNet](https://arxiv.org/abs/1512.03385) `arXiv:1512.03385` | Microsoft Research | 残差学习 $x+F(x)$ 让 152 层网络可训练，至今仍是所有深网络的骨架 |
| 2015-05 | [Highway Networks](https://arxiv.org/abs/1505.00387) `arXiv:1505.00387` | IDSIA | 门控跳连让上百层网络可训练，残差连接的前身 |
| 2015-05 | [Deep Learning（Nature 综述）](https://www.nature.com/articles/nature14539) | LeCun/Bengio/Hinton | 三位图灵奖得主对深度学习核心思想的总结：分层表征、反向传播、CNN 与 RNN |
| 2015-03 | ★ [知识蒸馏](https://arxiv.org/abs/1503.02531) `arXiv:1503.02531` | Google | 用教师模型的软标签训练小模型，"暗知识"的迁移 |
| 2015-02 | ★ [BatchNorm](https://arxiv.org/abs/1502.03167) `arXiv:1502.03167` | Google | 按小批量统计量标准化激活，允许更大学习率并显著加速训练 |
| 2015-02 | [He 初始化 / PReLU](https://arxiv.org/abs/1502.01852) `arXiv:1502.01852` | Microsoft Research | 针对 ReLU 推导方差保持初始化，首次在 ImageNet 上超过人类水平 |
| 2014-09 | [GoogLeNet](https://arxiv.org/abs/1409.4842) `arXiv:1409.4842` | Google | Inception 模块：多尺度并行分支 + 1×1 降维 |
| 2014-09 | [Seq2Seq](https://arxiv.org/abs/1409.3215) `arXiv:1409.3215` | Google | 编码器-解码器 LSTM 做机器翻译，确立"序列到序列"范式 |
| 2014-09 | [VGG](https://arxiv.org/abs/1409.1556) `arXiv:1409.1556` | Oxford | 堆叠 3×3 小卷积把网络加深到 16–19 层 |
| 2014-09 | ★ [Bahdanau 注意力](https://arxiv.org/abs/1409.0473) `arXiv:1409.0473` | Montréal | 解码时对源序列软对齐加权，注意力机制的起点 |
| 2014-06 | ★ [GAN](https://arxiv.org/abs/1406.2661) `arXiv:1406.2661` | Montréal | 生成器与判别器的极小极大博弈，开启对抗式生成 |
| 2013-12 | ★ [VAE](https://arxiv.org/abs/1312.6114) `arXiv:1312.6114` | Amsterdam | 重参数化技巧 + 变分下界，让隐变量生成模型可以端到端训练 |
| 2013-01 | [word2vec](https://arxiv.org/abs/1301.3781) `arXiv:1301.3781` | Google | 用预测上下文学习词向量，向量运算捕捉语义类比，表示学习的标志性工作 |
| 2012-12 | ★ [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Toronto | GPU + ReLU + Dropout 的深 CNN 以大幅优势赢得 ImageNet 2012，深度学习时代的起点 |
| 2012-07 | [Dropout](https://arxiv.org/abs/1207.0580) `arXiv:1207.0580` | Toronto | 训练时随机丢弃神经元，抑制共适应，相当于廉价的模型集成 |
| 2011-04 | [ReLU 深网络](https://proceedings.mlr.press/v15/glorot11a.html) | Montréal | 稀疏的整流线性单元让深层网络无需逐层预训练也能训练 |
| 2010-05 | [Xavier 初始化](https://proceedings.mlr.press/v9/glorot10a.html) | Montréal | 分析激活与梯度方差随层数的变化，提出按扇入扇出缩放的初始化 |
| 1998-11 | [LeNet-5](https://ieeexplore.ieee.org/document/726791) | AT&T Bell Labs | 卷积 + 池化 + 反向传播的端到端文档识别，CNN 的原型 |
| 1997-11 | ★ [LSTM](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) | TUM/IDSIA | 门控记忆单元解决 RNN 梯度消失，统治序列建模近二十年 |
| 1989-12 | [通用逼近定理](https://link.springer.com/article/10.1007/BF02551274) | Cybenko | 单隐层 sigmoid 网络可以任意精度逼近紧集上的连续函数：表达能力不是瓶颈，学习与泛化才是 |
| 1986-10 | ★ [Backprop](https://www.nature.com/articles/323533a0) | UCSD/CMU/Toronto | Rumelhart、Hinton、Williams：用链式法则反向传播误差训练多层网络，学到内部表征 |
| 1958 | [Perceptron](https://psycnet.apa.org/doi/10.1037/h0042519) | Cornell | Rosenblatt 感知机：第一个可学习权重的人工神经元模型与学习规则 |

### 优化与训练动力学（181）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [学习率调度未必有原则](https://arxiv.org/abs/2610.40362) `arXiv:2610.40362` |  | 权重归一化训练的收益主要来自其诱导的"有效学习率"轨迹；匹配有效学习率即可复现 |
| 2026-10 | 🔥 [LLM 预训练通常处于稳定性边缘吗](https://arxiv.org/abs/2610.36778) `arXiv:2610.36778` |  | 实测大模型预训练中锐度与学习率的关系 |
| 2026-10 | 🔥 [数据调度与学习率调度应联合设计](https://arxiv.org/abs/2610.29348) `arXiv:2610.29348` |  | 数据课程与学习率衰减相互作用，应一起设计 |
| 2026-10 | 🔥 [Adam/Shampoo/Muon 的统计视角](https://arxiv.org/abs/2610.24839) `arXiv:2610.24839` |  | 从统计估计角度统一比较三类优化器 |
| 2026-10 | 🔥 [Muon 越过稳定性边缘](https://arxiv.org/abs/2610.24305) `arXiv:2610.24305` |  | Muon 的动力学可在超出传统 EoS 的区域稳定运行 |
| 2026-10 | 🔥 [长程 RL 需要 Muon](https://arxiv.org/abs/2610.20256) `arXiv:2610.20256` | Renmin Univ | 长程智能体 RL 中 Muon 比 AdamW 更稳定、更新更均衡 |
| 2026-10 | 🔥 [Muon 何时失去谱隐式偏置](https://arxiv.org/abs/2610.20112) `arXiv:2610.20112` |  | 从谱范数到 Frobenius：某些条件下 Muon 的隐式偏置退化为 SGD 式 |
| 2026-10 | 🔥 [非欧梯度下降有多个稳定性边缘](https://arxiv.org/abs/2610.15223) `arXiv:2610.15223` |  | 非欧几何下存在多种锐度度量与多个边缘 |
| 2026-10 | 🔥 [谱权重衰减](https://arxiv.org/abs/2610.11730) `arXiv:2610.11730` |  | 对奇异值施加衰减以诱导低秩结构 |
| 2026-10 | 🔥 [行归一化为何能胜过 Adam 与 Muon](https://arxiv.org/abs/2610.11309) `arXiv:2610.11309` |  | 从几何到泛化：按行归一化的更新在某些情形下优于 Adam 与 Muon |
| 2026-10 | 🔥 [深度学习中 Hessian 的特征值](https://arxiv.org/abs/2610.09919) `arXiv:2610.09919` |  | 对称性的起源与破缺 |
| 2026-10 | 🔥 [最佳优化器取决于批大小](https://arxiv.org/abs/2610.08975) `arXiv:2610.08975` |  | 不同批大小下优化器排名不同 |
| 2026-09 | 🔥 [Muon 中的行归一化之谜](https://arxiv.org/abs/2609.39114) `arXiv:2609.39114` |  | 实践中对 Muon 更新再做行归一化为何有效 |
| 2026-09 | 🔥 [优化器不同，数据扩展殊途同归](https://arxiv.org/abs/2609.37745) `arXiv:2609.37745` |  | 不同优化器的训练动力学最终收敛到相同的 1/3 最优数据扩展指数 |
| 2026-09 | 🔥 [Muon 化解 LLM 预训练中的稳定性边缘](https://arxiv.org/abs/2609.34915) `arXiv:2609.34915` |  | Muon 训练 LLM 时稳定性边缘的形态与 Adam 不同 |
| 2026-09 | 🔥 [预热时长的扩展](https://arxiv.org/abs/2609.33041) `arXiv:2609.33041` |  | 学习率预热时长如何随规模与训练长度调整 |
| 2026-09 | 🔥 [实用 Muon 的收敛性](https://arxiv.org/abs/2609.33152) `arXiv:2609.33152` |  | 对带动量、权重衰减等实际组件的 Muon 给出收敛保证 |
| 2026-09 | 🔥 [走向稳定性边缘](https://arxiv.org/abs/2609.32290) `arXiv:2609.32290` |  | 稳定性边缘（EoS）现象与理论的系统梳理 |
| 2026-09 | 🔥 [河谷地形中的动量加速](https://arxiv.org/abs/2609.30957) `arXiv:2609.30957` |  | 在"河谷"型损失地形中解释动量为何加速 |
| 2026-09 | 🔥 [Hessian 与梯度协方差的子空间稳定](https://arxiv.org/abs/2609.31983) `arXiv:2609.31983` |  | 训练中二者主子空间的稳定 |
| 2026-09 | 🔥 [尺度不变优化何时失稳](https://arxiv.org/abs/2609.09116) `arXiv:2609.09116` |  | 带权重衰减的尺度不变网络的精确调度律 |
| 2026-09 | 🔥 [大批量训练中的动量](https://arxiv.org/abs/2609.02728) `arXiv:2609.02728` |  | Polyak 动量增大临界批大小，Nesterov 提升数据效率 |
| 2026-09 | 🔥 [优化中的渗流动力学](https://arxiv.org/abs/2609.02373) `arXiv:2609.02373` |  | 把随机梯度流建模为渗流过程：方差级联与离散标度不变性解释训练如何逐步走向更简单的子网络 |
| 2026-09 | 🔥 [稳定性边缘的多时间尺度](https://arxiv.org/abs/2609.01034) `arXiv:2609.01034` |  | 用微扰推导梯度下降在 EoS 上的快慢多时间尺度 |
| 2026-08 | 🔥 [谱分配：Muon 为何胜过 Adam](https://arxiv.org/abs/2608.25990) `arXiv:2608.25990` |  | 从谱分配角度解释 Muon 优于 Adam 的原因，并据此改进 Muon |
| 2026-08 | 🔥 [让 Adam 停滞的地形特征](https://arxiv.org/abs/2608.22145) `arXiv:2608.22145` |  | 定义并估计让 Adam 停滞的损失地形特征（预条件 Hessian） |
| 2026-08 | 🔥 [为扩散 Transformer 扩展 Muon](https://arxiv.org/abs/2608.20818) `arXiv:2608.20818` |  | Muon 在 DiT 上的扩展规律与超参数迁移 |
| 2026-08 | 🔥 [Adam 的稳定性边缘](https://arxiv.org/abs/2608.20638) `arXiv:2608.20638` |  | 自适应反馈、可证明的振荡与梯度反转 |
| 2026-08 | 🔥 [稳定性边缘上优化器的作用](https://arxiv.org/abs/2608.18415) `arXiv:2608.18415` |  | 不同优化器在 EoS 上走出的不同路径 |
| 2026-08 | 🔥 [损失看不见基，但 Adam 看得见](https://arxiv.org/abs/2608.05136) `arXiv:2608.05136` |  | 损失对 (U,V)→(UQ,VQ) 规范对称不变，梯度流的低秩机制与之相容，而逐坐标的 Adam 打破对称、走向不同的解 |
| 2026-07 | 🔥 [联合扩展律与最优批大小调度](https://arxiv.org/abs/2607.27731) `arXiv:2607.27731` |  | 把批大小调度纳入联合扩展律 |
| 2026-07 | 🔥 [重尾噪声下的 Adam](https://arxiv.org/abs/2607.27383) `arXiv:2607.27383` |  | 梯度噪声重尾时 Adam 的收敛行为 |
| 2026-07 | 🔥 [缩放权重衰减](https://arxiv.org/abs/2607.23777) `arXiv:2607.23777` |  | 权重衰减随模型规模缩放的规则 |
| 2026-07 | 🔥 [权重范数临界性](https://arxiv.org/abs/2607.21005) `arXiv:2607.21005` |  | 归一化与权重衰减共同导致损失尖峰的机制 |
| 2026-07 | 🔥 [Muon 何时帮助智能体 RL](https://arxiv.org/abs/2607.16169) `arXiv:2607.16169` |  | Muon 在智能体强化学习中的适用条件 |
| 2026-07 | 🔥 [Muse](https://arxiv.org/abs/2607.14536) `arXiv:2607.14536` |  | Muon 的表征几何，超越归一化动量 |
| 2026-07 | 🔥 [Hessian 谱如何依赖数据](https://arxiv.org/abs/2607.13631) `arXiv:2607.13631` |  | 数据结构对 Hessian 谱的影响 |
| 2026-07 | 🔥 [重新评估 Muon 做矩阵分解](https://arxiv.org/abs/2607.13246) `arXiv:2607.13246` |  | 在矩阵分解这一可分析的模型上检验 Muon 的优势与局限 |
| 2026-07 | 🔥 [同样的损失与噪声、相反的调度](https://arxiv.org/abs/2607.12360) `arXiv:2607.12360` |  | 梯度噪声结构与优化器归一化共同决定最优调度的形状 |
| 2026-07 | 🔥 [WSqD](https://arxiv.org/abs/2607.10959) `arXiv:2607.10959` |  | 无需预先指定训练长度的学习率调度 |
| 2026-07 | 🔥 [用近似对称解释近零 Hessian 特征值](https://arxiv.org/abs/2607.07845) `arXiv:2607.07845` |  | 近零特征值来自网络的近似对称 |
| 2026-07 | 🔥 [Muon 加速 grokking 的关键成分](https://arxiv.org/abs/2607.20512) `arXiv:2607.20512` |  | 分离出 Muon 让 grokking 更早发生的真正原因 |
| 2026-07 | 🔥 [OmniOpt](https://arxiv.org/abs/2607.04033) `arXiv:2607.04033` | OpenDataLab | 现代优化器的分类、几何视角与统一基准 |
| 2026-07 | 🔥 [重访 Adam 的收敛](https://arxiv.org/abs/2607.03519) `arXiv:2607.03519` |  | Adam 收敛性的再审视 |
| 2026-07 | 🔥 [如何分配 token](https://arxiv.org/abs/2607.01487) `arXiv:2607.01487` |  | 以训练步数与批大小为自变量的扩展律 |
| 2026-07 | 🔥 [Muon 即残差连接](https://arxiv.org/abs/2607.01124) `arXiv:2607.01124` |  | 从残差连接的角度重新理解 Muon 的更新 |
| 2026-06 | 🔥 [异步流水线并行](https://arxiv.org/abs/2606.30634) `arXiv:2606.30634` |  | 证明一步梯度延迟并不妨碍大规模异步流水线并行预训练，可消除流水线气泡 |
| 2026-06 | 🔥 [优化器相关的训练动力学](https://arxiv.org/abs/2606.30226) `arXiv:2606.30226` |  | 用 Hessian 特征向量位移刻画不同优化器的轨迹 |
| 2026-06 | 🔥 [Aurora](https://arxiv.org/abs/2606.27715) `arXiv:2606.27715` |  | 考虑行列"杠杆"的谱优化器 |
| 2026-06 | 🔥 [Edge Flow](https://arxiv.org/abs/2606.18080) `arXiv:2606.18080` |  | 可解、可预测的稳定性边缘连续时间模型 |
| 2026-06 | 🔥 [EoS 的分岔理论框架](https://arxiv.org/abs/2606.15551) `arXiv:2606.15551` |  | 用分岔理论刻画梯度下降进入稳定性边缘 |
| 2026-06 | 🔥 [Muon^p](https://arxiv.org/abs/2606.13867) `arXiv:2606.13867` |  | 把正交化（奇异值置 1）推广为奇异值的分数次幂，介于 SGD 与 Muon 之间 |
| 2026-06 | 🔥 [Muon 为何胜过 Adam：曲率视角](https://arxiv.org/abs/2606.04662) `arXiv:2606.04662` |  | 从损失曲率结构解释正交化更新的优势 |
| 2026-06 | 🔥 [EoS 的均值—涨落动力学](https://arxiv.org/abs/2606.05326) `arXiv:2606.05326` |  | 把稳定性边缘的训练分解为平均运动与振荡涨落 |
| 2026-06 | 🔥 [Muon 的谱扩展律](https://arxiv.org/abs/2606.04058) `arXiv:2606.04058` |  | Muon 下权重谱随训练与规模演化的扩展规律 |
| 2026-06 | 🔥 [EoS 选择性地塑造学习](https://arxiv.org/abs/2606.04212) `arXiv:2606.04212` |  | 稳定性边缘对数据分布不同部分的学习有选择性影响 |
| 2026-06 | 🔥 [低秩架构的黎曼梯度下降](https://arxiv.org/abs/2606.02328) `arXiv:2606.02328` |  | 在低秩流形上训练 |
| 2026-05 | 🔥 [神经扩展律的优化器依赖](https://arxiv.org/abs/2605.29387) `arXiv:2605.29387` |  | 扩展律的指数与系数依赖于优化器 |
| 2026-05 | 🔥 [非光滑优化中谱下降的收敛性](https://arxiv.org/abs/2605.26977) `arXiv:2605.26977` |  | 谱下降（Muon 的理想化）在非光滑非凸问题上的收敛 |
| 2026-05 | 🔥 [训练不稳定的非正规谱特征](https://arxiv.org/abs/2605.23476) `arXiv:2605.23476` |  | 用非正规矩阵谱预示训练不稳定 |
| 2026-05 | 🔥 [Muon 的哈密顿概率梯度流视角](https://arxiv.org/abs/2605.23871) `arXiv:2605.23871` |  | 把 Muon 解释为概率空间上的哈密顿梯度流 |
| 2026-05 | 🔥 [免调度的谱优化](https://arxiv.org/abs/2605.23061) `arXiv:2605.23061` |  | 把 schedule-free 思路用于谱优化器，实现随时可停的训练 |
| 2026-05 | 🔥 [优化器诱导的谱扩展律](https://arxiv.org/abs/2605.21803) `arXiv:2605.21803` |  | 同一架构、不同优化器，加宽 FFN 后能利用的谱容量不同：优化器是表示扩展的一条独立轴 |
| 2026-05 | 🔥 [预训练之外的 Muon](https://arxiv.org/abs/2605.19282) `arXiv:2605.19282` |  | Muon 在 VLA 与 RLVR 上的谱失效与高通补救 |
| 2026-05 | 🔥 [重尾噪声下 Muon 的紧复杂度](https://arxiv.org/abs/2605.18528) `arXiv:2605.18528` |  | Muon 在重尾噪声下的最优复杂度 |
| 2026-05 | 🔥 [ScheduleFree+](https://arxiv.org/abs/2605.19095) `arXiv:2605.19095` |  | 把免学习率、免调度方法扩展到 LLM 规模 |
| 2026-05 | 🔥 [权重衰减能增强训练稳定性吗](https://arxiv.org/abs/2605.16622) `arXiv:2605.16622` |  | 权重衰减与训练稳定性的关系 |
| 2026-05 | 🔥 [Pion](https://arxiv.org/abs/2605.12492) `arXiv:2605.12492` |  | 用左右正交变换更新权重、全程保持奇异值不变的"保谱"优化器，与加性更新的 Adam/Muon 不同 |
| 2026-05 | 🔥 [PowerStep](https://arxiv.org/abs/2605.10335) `arXiv:2605.10335` |  | ℓp 范数最速下降：在 SGD（ℓ2）与符号下降（ℓ∞）之间插值且省内存 |
| 2026-05 | 🔥 [Muon 能微调 Adam 预训练的模型吗](https://arxiv.org/abs/2605.10468) `arXiv:2605.10468` |  | 优化器失配：Adam 预训练的模型直接换 Muon 微调会变差，分析原因并给出过渡办法 |
| 2026-05 | 🔥 [Muon 的相图](https://arxiv.org/abs/2605.09552) `arXiv:2605.09552` | McGill | 在可解模型上刻画 Muon 何时优于 SignSGD |
| 2026-05 | 🔥 [Spectral Lens](https://arxiv.org/abs/2605.05683) `arXiv:2605.05683` |  | 用激活与梯度谱诊断 LLM 优化中的问题 |
| 2026-05 | 🔥 [Adam 的 Rod Flow](https://arxiv.org/abs/2605.06821) `arXiv:2605.06821` |  | 稳定性边缘上 Adam 的连续时间模型 |
| 2026-04 | ◆ [SGD 的稳定性边缘](https://arxiv.org/abs/2604.21016) `arXiv:2604.21016` |  | 随机情形下锐度停在 2/η 以下：随机锐度间隙 |
| 2026-04 | ◆ [Edge of Stability 的起源](https://arxiv.org/abs/2604.20446) `arXiv:2604.20446` |  | 解释梯度下降为何会自稳定在锐度 2/η 附近 |
| 2026-04 | ◆ [稳定性边缘上的泛化](https://arxiv.org/abs/2604.19740) `arXiv:2604.19740` |  | 分析在 EoS 区间训练对泛化的影响 |
| 2026-04 | ◆ [动量进一步约束随机稳定边缘的锐度](https://arxiv.org/abs/2604.14108) `arXiv:2604.14108` |  | 动量对随机稳定性边缘上锐度的影响 |
| 2026-04 | ◆ [谱边缘的生命周期](https://arxiv.org/abs/2604.07380) `arXiv:2604.07380` |  | 从梯度学习到权重衰减压缩：权重谱边缘的演化 |
| 2026-04 | ◆ [Newton-Muon](https://arxiv.org/abs/2604.01472) `arXiv:2604.01472` | UPenn | 把 Muon 解释为近似牛顿法并据此改进 |
| 2026-03 | ◆ [谱优化器学习联想记忆的容量标度](https://arxiv.org/abs/2603.26554) `arXiv:2603.26554` |  | 精确刻画谱优化器（Muon）学习联想记忆时的容量随规模的标度，优于 SGD/Adam |
| 2026-03 | ◆ [LLM 训练的最优低秩随机梯度估计](https://arxiv.org/abs/2603.20632) `arXiv:2603.20632` |  | 低秩梯度估计的最优性 |
| 2026-03 | ◆ [不做学习率衰减的预训练更利于 SFT](https://arxiv.org/abs/2603.16127) `arXiv:2603.16127` |  | 衰减阶段的收益在后续 SFT 中会被抵消甚至有害 |
| 2026-03 | ◆ [用现代优化理论推导超参数扩展律](https://arxiv.org/abs/2603.15958) `arXiv:2603.15958` |  | 从优化理论推导学习率、批大小等随规模的标度 |
| 2026-03 | ◆ [近最优学习率调度的形状](https://arxiv.org/abs/2603.10301) `arXiv:2603.10301` |  | 搜索并刻画近最优学习率调度的共同特征 |
| 2026-03 | ◆ [矩阵算子范数下优化器的宽度扩展](https://arxiv.org/abs/2603.09952) `arXiv:2603.09952` |  | 行/列归一化等在算子范数视角下的宽度扩展规则 |
| 2026-03 | ◆ [HTMuon](https://arxiv.org/abs/2603.10067) `arXiv:2603.10067` |  | 用重尾谱校正改进 Muon |
| 2026-03 | ◆ [非欧梯度下降也处于稳定性边缘](https://arxiv.org/abs/2603.05002) `arXiv:2603.05002` |  | 符号下降、谱下降等非欧方法同样运行在（相应范数下的）稳定性边缘 |
| 2026-03 | ◆ [渐进残差 warmup](https://arxiv.org/abs/2603.05369) `arXiv:2603.05369` |  | 按层深度渐进开启残差分支的 warmup，提升预训练稳定性与收敛 |
| 2026-03 | ◆ [Adam 为何能胜过 SGD](https://arxiv.org/abs/2603.03099) `arXiv:2603.03099` |  | 二阶矩归一化使更新的尾部更"锐利"，在重尾噪声下优于 SGD |
| 2026-03 | ◆ [重参数化最速下降不会卡在鞍点](https://arxiv.org/abs/2603.02064) `arXiv:2603.02064` |  | 把重参数化的最速下降写成镜像流并证明避开鞍点 |
| 2026-03 | ◆ [Adam 无需修改即可收敛](https://arxiv.org/abs/2603.02092) `arXiv:2603.02092` |  | 证明标准 Adam（不改更新规则）在合理超参数下收敛 |
| 2026-02 | ◆ [AdaMuon（Adam 改进 Muon）](https://arxiv.org/abs/2602.17080) `arXiv:2602.17080` |  | 在正交化动量之上再加 Adam 式的自适应二阶矩，兼得 Muon 的矩阵结构与 Adam 的噪声适应 |
| 2026-02 | ◆ [Adam 与 Muon 的隐式偏置](https://arxiv.org/abs/2602.16340) `arXiv:2602.16340` |  | 光滑齐次网络上，Adam 与 Muon 分别收敛到 ℓ∞ 与谱范数下的最大间隔解 |
| 2026-02 | ◆ [屏蔽更新的惊人效果](https://arxiv.org/abs/2602.15322) `arXiv:2602.15322` | Google | 随机屏蔽自适应优化器的部分更新即可提升训练 |
| 2026-02 | ◆ [快追赶、晚切换](https://arxiv.org/abs/2602.14208) `arXiv:2602.14208` |  | 用函数式扩展律推导最优批大小调度 |
| 2026-02 | ◆ [小批量最速下降的隐式偏置](https://arxiv.org/abs/2602.11557) `arXiv:2602.11557` |  | 随机小批量下，不同范数的最速下降收敛到何种解 |
| 2026-02 | ◆ [权重衰减提升可塑性](https://arxiv.org/abs/2602.11137) `arXiv:2602.11137` |  | 权重衰减让预训练模型在后续微调中保持更好的可塑性 |
| 2026-02 | ◆ [归一化梯度下降的隐式正则](https://arxiv.org/abs/2602.08177) `arXiv:2602.08177` |  | 归一化 GD 的隐式正则效应 |
| 2026-02 | ◆ [稳健的优化器扩展律](https://arxiv.org/abs/2602.07712) `arXiv:2602.07712` |  | 比较优化器时应拟合扩展律而非单点比较 |
| 2026-02 | ◆ [函数式扩展律下的最优学习率调度](https://arxiv.org/abs/2602.06797) `arXiv:2602.06797` |  | 推导最优调度：幂律衰减与 WSD 的理论依据 |
| 2026-02 | ◆ [联想记忆中的 Muon：动力学与扩展律](https://arxiv.org/abs/2602.05725) `arXiv:2602.05725` |  | 在联想记忆模型中推导 Muon 的训练动力学与扩展律 |
| 2026-02 | ◆ [SGD 噪声协方差与曲率的超线性关系](https://arxiv.org/abs/2602.05600) `arXiv:2602.05600` |  | SGD 噪声协方差与 Hessian 并非简单成正比 |
| 2026-02 | ◆ [随机特征模型的最优调度与扩展律](https://arxiv.org/abs/2602.04774) `arXiv:2602.04774` |  | 在可解的随机特征模型中求出最优学习率调度与相应扩展律 |
| 2026-02 | ◆ [学习率才是关键：普通 LoRA 足矣](https://arxiv.org/abs/2602.04998) `arXiv:2602.04998` |  | 各种 LoRA 变体的增益大多来自学习率未调好；充分调参后普通 LoRA 与之相当 |
| 2026-02 | ◆ [Canzona](https://arxiv.org/abs/2602.06079) `arXiv:2602.06079` | Qwen | 面向 Muon、Shampoo、SOAP 等矩阵优化器的分布式异步、负载均衡实现框架 |
| 2026-02 | ◆ [非欧梯度噪声尺度与自适应批大小](https://arxiv.org/abs/2602.03001) `arXiv:2602.03001` |  | 为符号下降与谱下降定义梯度噪声尺度并自适应调整批大小 |
| 2026-02 | ◆ [Anytime 预训练](https://arxiv.org/abs/2602.03702) `arXiv:2602.03702` |  | 无需预设训练长度的学习率调度 + 权重平均 |
| 2026-02 | ◆ [小批量噪声对 Adam 隐式偏置的影响](https://arxiv.org/abs/2602.01642) `arXiv:2602.01642` |  | 小批量噪声改变 Adam 的隐式偏置 |
| 2026-02 | ◆ [SPARKLING](https://arxiv.org/abs/2602.02472) `arXiv:2602.02472` | ByteDance Seed | 宽度渐进式训练中兼顾信号保持与对称性破缺，降低预训练成本 |
| 2026-02 | ◆ [MSign](https://arxiv.org/abs/2602.01734) `arXiv:2602.01734` | Microsoft | 把训练不稳定归因于权重矩阵稳定秩坍缩，周期性用符号操作恢复稳定秩 |
| 2026-02 | ◆ [DASH](https://arxiv.org/abs/2602.02016) `arXiv:2602.02016` | IST Austria | 批量化块预条件 + 高效逆根求解，让 Shampoo 更快 |
| 2026-02 | ◆ [Rod Flow](https://arxiv.org/abs/2602.01480) `arXiv:2602.01480` |  | 稳定性边缘上梯度下降的连续时间模型（推广 central flow） |
| 2026-01 | ◆ [深度而非数据：Hessian 谱分岔](https://arxiv.org/abs/2602.00545) `arXiv:2602.00545` |  | Hessian 谱的分岔由深度而非数据决定 |
| 2026-01 | ◆ [基础模型规模的 Hessian 谱分析](https://arxiv.org/abs/2602.00816) `arXiv:2602.00816` |  | 在大模型规模上计算并分析 Hessian 谱 |
| 2026-01 | ◆ [Spectra](https://arxiv.org/abs/2602.11185) `arXiv:2602.11185` |  | LLM 梯度的谱各向异性下重新设计优化器 |
| 2026-01 | ◆ [为何 β1=β2 在动力学上特殊](https://arxiv.org/abs/2601.21739) `arXiv:2601.21739` |  | Adam 两个衰减率相等时的特殊动力学性质 |
| 2026-01 | ◆ [带 Newton–Schulz 的 Muon 收敛性](https://arxiv.org/abs/2601.19156) `arXiv:2601.19156` |  | 考虑近似正交化（Newton–Schulz 迭代）误差的 Muon 收敛分析 |
| 2026-01 | ◆ [可扩展的曲率度量](https://arxiv.org/abs/2601.16979) `arXiv:2601.16979` |  | 用于分析 LLM 训练动力学的可扩展损失曲率度量 |
| 2026-01 | ◆ [谱正交化的预条件收益](https://arxiv.org/abs/2601.13474) `arXiv:2601.13474` |  | 从预条件角度解释 Muon 正交化的好处 |
| 2026-01 | ◆ [大规模预训练如何设置批大小](https://arxiv.org/abs/2601.05034) `arXiv:2601.05034` |  | 批大小与学习率、数据量的联合设置 |
| 2026-01 | ◆ [瘦高网络的 Hessian 易于求逆](https://arxiv.org/abs/2601.06096) `arXiv:2601.06096` |  | 特殊结构下 Hessian 的可逆性 |
| 2025-12 | ◆ [修正解耦权重衰减](https://arxiv.org/abs/2512.08217) `arXiv:2512.08217` |  | AdamW 解耦权重衰减的系数应与学习率如何耦合 |
| 2025-12 | ◆ [超参数迁移让矩阵预条件优化器跨规模稳定获益](https://arxiv.org/abs/2512.05620) `arXiv:2512.05620` |  | 配合正确的超参数迁移，Shampoo/SOAP/Muon 等相对 AdamW 的增益在大规模下仍保持 |
| 2025-12 | ◆ [可证明调好的学习率调度](https://arxiv.org/abs/2512.05084) `arXiv:2512.05084` |  | 梯度下降学习率调度的可证明调参 |
| 2025-11 | ◆ [两种几何的故事](https://arxiv.org/abs/2511.20584) `arXiv:2511.20584` |  | 自适应优化器（Adam）与非欧最速下降（符号下降）之间的联系与区别 |
| 2025-11 | ◆ [ROOT](https://arxiv.org/abs/2511.20626) `arXiv:2511.20626` | Huawei Noah's Ark | 对维度鲁棒的正交化与对离群值鲁棒的近端优化，改进 Muon 类优化器 |
| 2025-11 | ◆ [学习率衰减浪费了课程中的好数据](https://arxiv.org/abs/2511.18903) `arXiv:2511.18903` |  | 把高质量数据放在末尾时，学习率已衰减到很小，使其收益打折 |
| 2025-11 | ◆ [结构化优化的非欧 SGD](https://arxiv.org/abs/2511.11466) `arXiv:2511.11466` |  | 非欧几何下 SGD 的统一分析与更好的收敛率 |
| 2025-11 | ◆ [Adam 对称定理](https://arxiv.org/abs/2511.06675) `arXiv:2511.06675` |  | 用对称性刻画随机 Adam 何时收敛 |
| 2025-10 | 🆕 [Cautious Weight Decay](https://arxiv.org/abs/2510.12402) `arXiv:2510.12402` | Google DeepMind | 只对与更新方向符号一致的坐标施加权重衰减，保持原目标的驻点 |
| 2025-10 | 🆕 [NorMuon](https://arxiv.org/abs/2510.05491) `arXiv:2510.05491` | Georgia Tech | 在正交化后加入逐神经元自适应归一化，兼顾 Muon 与 Adam 的优点 |
| 2025-09 | 🆕 [Muon 胜在长尾联想记忆](https://arxiv.org/abs/2509.26030) `arXiv:2509.26030` |  | Muon 优于 Adam 的关键在于更均衡地学习长尾的联想记忆 |
| 2025-09 | 🆕 [KL-Shampoo](https://arxiv.org/abs/2509.03378) `arXiv:2509.03378` | Vector Institute 等 | 从 KL 最小化视角理解并改进 Shampoo 与 SOAP |
| 2025-09 | ★ 🆕 [优化器大比拼](https://arxiv.org/abs/2509.02046) `arXiv:2509.02046` | Stanford | 严格调参后，号称 2 倍提速的新优化器相对 AdamW 的优势只有约 1.1–1.4 倍且随规模缩小；矩阵型优化器（Muon、SOAP）最稳 |
| 2025-09 | 🆕 [LLM 预训练优化器基准](https://arxiv.org/abs/2509.01440) `arXiv:2509.01440` | EPFL | 系统比较近年新优化器在不同模型规模、批大小与训练时长下的表现 |
| 2025-07 | 🆕 [Muon 的收敛界与临界批大小](https://arxiv.org/abs/2507.01598) `arXiv:2507.01598` |  | 给出 Muon 的收敛分析并推导其临界批大小 |
| 2025-06 | 🆕 [提纯 Shampoo](https://arxiv.org/abs/2506.03595) `arXiv:2506.03595` | Meta | 分解 Shampoo 预条件器，解释嫁接与陈旧更新等启发式技巧为何有效 |
| 2025-06 | 🆕 [SGG 梯度分组](https://arxiv.org/abs/2506.01049) `arXiv:2506.01049` | Zhejiang Univ | 按层内梯度统计聚类分组再缩放学习率，与现有自适应优化器兼容 |
| 2025-05 | 🆕 [临界批大小再探](https://arxiv.org/abs/2505.23971) `arXiv:2505.23971` | Ai2 | 用简单经验方法在训练中测量临界批大小，并据此做批大小 warmup |
| 2025-05 | 🆕 [Power Lines](https://arxiv.org/abs/2505.13738) `arXiv:2505.13738` | Cerebras | 权重衰减与批大小的扩展律：最优 AdamW 时间尺度随 token/参数比呈幂律 |
| 2025-05 | 🆕 [Muon 的实际效率](https://arxiv.org/abs/2505.02222) `arXiv:2505.02222` | Essential AI | Muon 在算力-时间帕累托前沿上全面优于 AdamW，并在大批量下保持数据效率；配合 μP 做超参迁移 |
| 2025-04 | 🆕 [Dion](https://arxiv.org/abs/2504.05295) `arXiv:2504.05295` | Microsoft Research | 通信高效的分布式正交化更新，可替代 Muon 的完整矩阵 Newton-Schulz |
| 2025-02 | ★ 🆕 [Muon 可扩展](https://arxiv.org/abs/2502.16982) `arXiv:2502.16982` | Moonshot AI | 加权重衰减并按 RMS 匹配更新尺度，Muon 在大规模 LLM 上计算效率约为 AdamW 的 2 倍（Moonlight） |
| 2025-02 | 🆕 [GradientStabilizer（原 Stable-SPAM）](https://arxiv.org/abs/2502.17055) `arXiv:2502.17055` |  | 低比特训练放大了梯度尖峰的影响；稳定梯度范数而非逐元素裁剪，使 4-bit 训练比 16-bit Adam 更稳 |
| 2025-02 | 🆕 [线性衰减到零](https://arxiv.org/abs/2502.15938) `arXiv:2502.15938` | Cerebras | 学习率线性衰减到 0（D2Z）在算力最优训练中稳定优于余弦衰减到 10% |
| 2025-02 | 🆕 [超越 EoS 的特征学习](https://arxiv.org/abs/2502.13110) `arXiv:2502.13110` |  | 大学习率下越过稳定性边缘的振荡如何促进特征学习 |
| 2025-01 | 🆕 [SPAM](https://arxiv.org/abs/2501.06842) `arXiv:2501.06842` |  | 检测梯度尖峰并重置动量的 Adam 变体，缓解大模型训练损失尖峰 |
| 2024-12 | 🆕 [SGD-SaI](https://arxiv.org/abs/2412.11768) `arXiv:2412.11768` |  | 只在初始化时按梯度信噪比为参数组设定学习率，免去 Adam 的二阶矩状态 |
| 2024-12 | 🆕 [APOLLO](https://arxiv.org/abs/2412.05270) `arXiv:2412.05270` | UT Austin/Meta | 用随机投影近似通道级学习率缩放，SGD 级显存、AdamW 级效果 |
| 2024-12 | ★ 🆕 [Muon](https://kellerjordan.github.io/posts/muon/) | Keller Jordan | 对动量矩阵做 Newton-Schulz 正交化再更新：矩阵参数的"谱范数最速下降"，NanoGPT 速通纪录的核心 |
| 2024-11 | 🆕 [Cautious 优化器](https://arxiv.org/abs/2411.16085) `arXiv:2411.16085` | UT Austin | 一行代码：只在更新方向与梯度同号的坐标上更新，保持收敛性并加速训练 |
| 2024-11 | 🆕 [MARS](https://arxiv.org/abs/2411.10438) `arXiv:2411.10438` | UCLA | 把方差缩减引入大模型预条件优化器 |
| 2024-10 | [Central Flows](https://arxiv.org/abs/2410.24206) `arXiv:2410.24206` | CMU/Princeton | 用"中心流"刻画振荡优化器的时间平均轨迹，解释自适应优化器的隐式行为 |
| 2024-10 | [临界批大小如何扩展](https://arxiv.org/abs/2410.21676) `arXiv:2410.21676` | Harvard | 临界批大小主要随数据量而非模型规模增长 |
| 2024-10 | [Modular Duality](https://arxiv.org/abs/2410.21265) `arXiv:2410.21265` | MIT | "对偶映射"视角：梯度在对偶空间，更新应在原空间，按模块范数构造优化器 |
| 2024-10 | [河谷地形与 WSD](https://arxiv.org/abs/2410.05192) `arXiv:2410.05192` | Tsinghua/Stanford | "河谷"型损失地形解释 Warmup-Stable-Decay：稳定期沿河前进，衰减期下到谷底 |
| 2024-09 | [Old Optimizer, New Norm](https://arxiv.org/abs/2409.20325) `arXiv:2409.20325` | MIT | 把 Adam、Shampoo、Prodigy 统一为不同范数下的最速下降，为 Muon 提供理论框架 |
| 2024-09 | [SOAP](https://arxiv.org/abs/2409.11321) `arXiv:2409.11321` | Harvard | 在 Shampoo 的特征基里跑 Adam，稳定且高效 |
| 2024-09 | [AdEMAMix](https://arxiv.org/abs/2409.03137) `arXiv:2409.03137` | Apple/EPFL | 混合快慢两个动量 EMA，更好利用很旧的梯度 |
| 2024-07 | [参数化与优化器的扩展指数](https://arxiv.org/abs/2407.05872) `arXiv:2407.05872` | Google DeepMind | 万余次实验比较 μP、NTK 等参数化在不同优化器下的学习率迁移，发现每层学习率与 Adam-atan2 |
| 2024-06 | [为什么要 warmup](https://arxiv.org/abs/2406.09405) `arXiv:2406.09405` | Princeton | warmup 的主要作用是把网络带到更"良态"（低锐度）的区域，从而允许更大学习率 |
| 2024-05 | [Schedule-Free](https://arxiv.org/abs/2405.15682) `arXiv:2405.15682` | Meta | 不需要预设训练长度的学习率调度，基于迭代平均的统一理论（AlgoPerf 冠军） |
| 2024-05 | [AdamW 权重衰减随规模的设置](https://arxiv.org/abs/2405.13698) `arXiv:2405.13698` | Bristol | 把 AdamW 看作权重的 EMA，给出随模型与数据扩展时权重衰减的设置规则 |
| 2024-04 | [MiniCPM（WSD 调度）](https://arxiv.org/abs/2404.06395) `arXiv:2404.06395` | Tsinghua/ModelBest | 提出 WSD 学习率调度，可在任意时刻衰减并据此高效拟合扩展律 |
| 2023-05 | [Sophia](https://arxiv.org/abs/2305.14342) `arXiv:2305.14342` | Stanford | 对角 Hessian 估计 + 裁剪的二阶优化器，宣称预训练提速 2 倍 |
| 2023-02 | [Lion](https://arxiv.org/abs/2302.06675) `arXiv:2302.06675` | Google | 程序搜索发现的符号更新优化器，显存更省 |
| 2021-02 | ★ [Edge of Stability](https://arxiv.org/abs/2103.00065) `arXiv:2103.00065` | CMU | 全批量梯度下降会把锐度推到 2/η 并在那里振荡、损失却持续下降，挑战经典优化理论 |
| 2020-10 | [SAM](https://arxiv.org/abs/2010.01412) `arXiv:2010.01412` | Google | 锐度感知最小化：最小化邻域内最坏损失，寻找平坦极小值 |
| 2019-10 | [ZeRO](https://arxiv.org/abs/1910.02054) `arXiv:1910.02054` | Microsoft | 把优化器状态、梯度、参数切分到各数据并行进程，训练万亿参数模型的显存方案 |
| 2019-09 | [Megatron-LM](https://arxiv.org/abs/1909.08053) `arXiv:1909.08053` | NVIDIA | 张量（层内）模型并行训练数十亿参数 Transformer |
| 2019-04 | [LAMB](https://arxiv.org/abs/1904.00962) `arXiv:1904.00962` | Google | 分层自适应信任比，把 BERT 批大小扩到 32K |
| 2018-04 | [Adafactor](https://arxiv.org/abs/1804.04235) `arXiv:1804.04235` | Google | 二阶矩分解为行列因子，亚线性显存的自适应优化器 |
| 2018-02 | [Shampoo](https://arxiv.org/abs/1802.09568) `arXiv:1802.09568` | Google | 按张量维度构造 Kronecker 因子预条件器的二阶类方法 |
| 2017-12 | [损失地形可视化](https://arxiv.org/abs/1712.09913) `arXiv:1712.09913` | Maryland | 滤波归一化可视化：跳连让损失地形更平滑 |
| 2017-11 | ★ [AdamW](https://arxiv.org/abs/1711.05101) `arXiv:1711.05101` | Freiburg | 把权重衰减与自适应梯度解耦，大模型训练默认配置 |
| 2017-10 | [混合精度训练](https://arxiv.org/abs/1710.03740) `arXiv:1710.03740` | Baidu/NVIDIA | FP16 计算 + FP32 主权重 + 损失缩放，现代训练的基础配置 |
| 2017-06 | [大批量 SGD](https://arxiv.org/abs/1706.02677) `arXiv:1706.02677` | Facebook | 线性缩放学习率 + warmup，1 小时训练 ImageNet |
| 2016-09 | [大批量与尖锐极小值](https://arxiv.org/abs/1609.04836) `arXiv:1609.04836` | Northwestern/Intel | 大批量训练趋向尖锐极小值、泛化变差的经验观察 |
| 2016-08 | [SGDR / 余弦退火](https://arxiv.org/abs/1608.03983) `arXiv:1608.03983` | Freiburg | 余弦学习率衰减与热重启，余弦调度成为默认选择 |
| 2016-04 | [梯度检查点](https://arxiv.org/abs/1604.06174) `arXiv:1604.06174` | UW/MIT | 以重计算换显存，亚线性显存训练深网络 |
| 2014-12 | ★ [Adam](https://arxiv.org/abs/1412.6980) `arXiv:1412.6980` | OpenAI/Toronto | 一阶矩 + 二阶矩自适应步长，十年来最常用的优化器 |

### 初始化/归一化/残差/参数化（85）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [用超参数迁移与集成改进数据受限扩展](https://arxiv.org/abs/2610.37842) `arXiv:2610.37842` |  | 数据受限时，超参数迁移与集成改善扩展 |
| 2026-10 | 🔥 [线性残差网络中宽度与深度极限可交换](https://arxiv.org/abs/2610.30729) `arXiv:2610.30729` |  | 训练中无限宽、无限深极限的交换性 |
| 2026-10 | 🔥 [ShellElastic](https://arxiv.org/abs/2610.15124) `arXiv:2610.15124` |  | 一个权重共享超网中嵌套多个子网络的弹性预训练，各子网同时满足最大更新参数化、可迁移超参数 |
| 2026-10 | 🔥 [多流注意力残差](https://arxiv.org/abs/2610.13630) `arXiv:2610.13630` | SJTU | 把注意力残差推广到多条残差流 |
| 2026-10 | 🔥 [巨大激活门控通道](https://arxiv.org/abs/2610.07661) `arXiv:2610.07661` |  | 巨大激活充当门控通道 |
| 2026-09 | 🔥 [巨大激活的生命周期](https://arxiv.org/abs/2610.00423) `arXiv:2610.00423` |  | 巨大激活随机诞生、由权重衰减驱动增长、彼此竞争：训练轨迹分析与干预实验 |
| 2026-09 | 🔥 ["写入时读盲"解释巨大激活](https://arxiv.org/abs/2609.35630) `arXiv:2609.35630` |  | 巨大激活的一种机制解释 |
| 2026-09 | 🔥 [DepthBench](https://arxiv.org/abs/2609.32534) `arXiv:2609.32534` | MPI | 衡量残差连接如何让 Transformer 真正利用更多计算深度 |
| 2026-09 | 🔥 [残差流的有效深度](https://arxiv.org/abs/2609.31098) `arXiv:2609.31098` |  | 残差流实际用了多少深度 |
| 2026-09 | 🔥 [跨 MoE 稀疏度的超参数扩展律](https://arxiv.org/abs/2609.08690) `arXiv:2609.08690` |  | 最优超参数如何随 MoE 稀疏度变化 |
| 2026-08 | 🔥 [大规模 MoE 的超参迁移](https://arxiv.org/abs/2608.20061) `arXiv:2608.20061` | Kakao | 算力高效地把小规模调好的超参迁移到大规模 MoE |
| 2026-08 | 🔥 [残差网络深度充分性的一阶判据](https://arxiv.org/abs/2608.14664) `arXiv:2608.14664` |  | 判断深度是否足够 |
| 2026-07 | 🔥 [多头注意力残差](https://arxiv.org/abs/2607.27230) `arXiv:2607.27230` |  | 注意力残差只有一个跨整个宽度共享的查询；改为多头，让不同特征子空间读取不同深度 |
| 2026-07 | 🔥 [残差流几何随深度的分析](https://arxiv.org/abs/2607.18348) `arXiv:2607.18348` |  | 不同深度残差流的几何 |
| 2026-07 | 🔥 [xHC](https://arxiv.org/abs/2607.14530) `arXiv:2607.14530` | dots studio | 扩展超连接的残差流数量，提供"记忆维度"上的扩展 |
| 2026-07 | 🔥 [架构如何应对深度的谱病理](https://arxiv.org/abs/2607.14018) `arXiv:2607.14018` |  | 秩坍缩等深度病理与架构对策 |
| 2026-07 | 🔥 [AutoNorm](https://arxiv.org/abs/2607.10593) `arXiv:2607.10593` |  | 用可微门控理解 Transformer 中的自适应归一化 |
| 2026-06 | 🔥 [巨大激活在架构上是稳健的](https://arxiv.org/abs/2606.20743) `arXiv:2606.20743` |  | 受控的残差实验表明巨大激活难以通过架构消除 |
| 2026-06 | 🔥 [Loop-μP](https://arxiv.org/abs/2606.18524) `arXiv:2606.18524` |  | 循环 Transformer 的残差缩放与超参数迁移 |
| 2026-06 | 🔥 [与范数无关的残差网络扩展自适应深度](https://arxiv.org/abs/2606.16112) `arXiv:2606.16112` |  | 让残差网络可变深度推理 |
| 2026-05 | 🔥 [归一化层的缩放向量](https://arxiv.org/abs/2605.26895) `arXiv:2605.26895` |  | 归一化层中"可学习缩放向量"参数极少、影响却很大：分析其作用与可移除性 |
| 2026-05 | 🔥 [μP 下宽网络的特征学习](https://arxiv.org/abs/2605.24710) `arXiv:2605.24710` |  | 可辨识性与稀疏字典分解 |
| 2026-05 | 🔥 [Complete-muE](https://arxiv.org/abs/2605.23893) `arXiv:2605.23893` |  | MoE 的最优超参数迁移与扩展 |
| 2026-05 | 🔥 [量化超参数迁移](https://arxiv.org/abs/2605.21486) `arXiv:2605.21486` |  | 超参数迁移的定量检验：嵌入层学习率尤为关键 |
| 2026-05 | 🔥 [注意力残差中的汇聚与离群值](https://arxiv.org/abs/2605.17887) `arXiv:2605.17887` |  | 注意力残差（沿深度的注意力）同样出现汇聚 |
| 2026-05 | 🔥 [宽特征学习网络的典范正则](https://arxiv.org/abs/2605.18180) `arXiv:2605.18180` |  | 宽网络特征学习极限的典范正则化 |
| 2026-05 | 🔥 [测量开源 LLM 的最大激活](https://arxiv.org/abs/2605.15572) `arXiv:2605.15572` |  | 系统测量开源模型激活的动态范围（巨大激活、离群值），服务低比特量化 |
| 2026-05 | 🔥 [残差流动力学](https://arxiv.org/abs/2605.14258) `arXiv:2605.14258` |  | 把谱几何与网络拓扑耦合起来 |
| 2026-05 | 🔥 [以 RMSNorm 的效率享用 LayerNorm](https://arxiv.org/abs/2605.14521) `arXiv:2605.14521` |  | LayerNorm 的高效实现 |
| 2026-05 | 🔥 [如何扩展 MoE](https://arxiv.org/abs/2605.14200) `arXiv:2605.14200` |  | 从 μP 到"最大尺度稳定参数化" |
| 2026-05 | 🔥 [Delta Attention Residuals](https://arxiv.org/abs/2605.18855) `arXiv:2605.18855` |  | 注意力残差在累积隐状态上做 softmax 读取高度冗余；改为读取逐层增量 |
| 2026-05 | 🔥 [单层解释巨大激活](https://arxiv.org/abs/2605.08504) `arXiv:2605.08504` |  | LLM 的巨大激活可追溯到单一层的机制 |
| 2026-05 | 🔥 [均值-方差分离残差](https://arxiv.org/abs/2605.06169) `arXiv:2605.06169` |  | 千层扩散 Transformer 会陷入"均值模式"退化，拆分残差的均值与方差即可稳定训练 |
| 2026-04 | ◆ [归一化 Transformer 中的学习率迁移](https://arxiv.org/abs/2604.27077) `arXiv:2604.27077` |  | nGPT 类超球面 Transformer 的学习率迁移 |
| 2026-04 | ◆ [无归一化 Transformer 初始化时的次临界信号传播](https://arxiv.org/abs/2604.11890) `arXiv:2604.11890` |  | DyT 等无归一化 Transformer 的信号传播分析 |
| 2026-03 | ◆ [各种形状 ResNet 的大尺度极限](https://arxiv.org/abs/2603.18168) `arXiv:2603.18168` |  | 深宽同时趋于无穷时训练动力学的收敛 |
| 2026-03 | ◆ [残差流对偶](https://arxiv.org/abs/2603.16039) `arXiv:2603.16039` |  | Transformer 沿序列与深度两个有序维度传递信息：深度方向的残差设计与序列方向的混合器一一对偶 |
| 2026-03 | ◆ [超连接的因果分析](https://arxiv.org/abs/2603.14833) `arXiv:2603.14833` |  | 消融与救援：超连接残差流到底做了什么 |
| 2026-03 | ★ ◆ [Attention Residuals](https://arxiv.org/abs/2603.15031) `arXiv:2603.15031` | Moonshot AI | 用沿深度方向的 softmax 注意力替代固定权重的残差累加，每层按内容检索前面各层的输出 |
| 2026-03 | ◆ [尖峰、稀疏与汇聚](https://arxiv.org/abs/2603.05498) `arXiv:2603.05498` |  | 巨大激活与注意力汇聚的解剖：两者如何相互产生 |
| 2026-02 | ◆ [宽深联合缩放的 μP 谱条件](https://arxiv.org/abs/2603.00541) `arXiv:2603.00541` | Renmin Univ | 给出同时扩展宽度与深度时特征学习稳定的谱条件 |
| 2026-02 | ◆ [跨优化器扩展 μP](https://arxiv.org/abs/2602.20937) `arXiv:2602.20937` | Argonne | 把 μP 的谱条件推广到不同优化器 |
| 2026-02 | ◆ [残差连接与因果偏移](https://arxiv.org/abs/2602.14760) `arXiv:2602.14760` |  | Transformer 中的一种结构性错位 |
| 2026-02 | ◆ [门控去除归一化与尺度锚定](https://arxiv.org/abs/2602.10408) `arXiv:2602.10408` |  | 在 Pre-Norm Transformer 中去掉归一化 |
| 2026-02 | ◆ [μP 放大小模型](https://arxiv.org/abs/2602.10545) `arXiv:2602.10545` |  | 从小模型热启动扩展到大模型，同时保持超参数可迁移 |
| 2026-02 | ◆ [梯度残差连接](https://arxiv.org/abs/2602.09190) `arXiv:2602.09190` |  | 在梯度路径上加入残差 |
| 2026-02 | ◆ [Lp 范数的 QKNorm](https://arxiv.org/abs/2602.05006) `arXiv:2602.05006` |  | 增强的 QK 归一化 |
| 2026-02 | ◆ [SimpleGPT（SimpleNorm）](https://arxiv.org/abs/2602.01212) `arXiv:2602.01212` |  | 从激活尺度—Hessian—最大可容忍学习率的关系出发，用简单归一化稳定中间激活、允许更大学习率 |
| 2026-01 | ◆ [KromHC](https://arxiv.org/abs/2601.21579) `arXiv:2601.21579` |  | 用 Kronecker 积参数化 mHC 的残差混合矩阵，降低投影到 Birkhoff 多面体的开销、提升可扩展性 |
| 2026-01 | ◆ [深度错觉](https://arxiv.org/abs/2601.20994) `arXiv:2601.20994` |  | 论证 Transformer 应更宽而非更深 |
| 2026-01 | ◆ [MoE 层的超参迁移](https://arxiv.org/abs/2601.20205) `arXiv:2601.20205` |  | 把 μP 式超参迁移推广到混合专家层 |
| 2026-01 | ◆ [Post-LN 回归](https://arxiv.org/abs/2601.19895) `arXiv:2601.19895` |  | 宽度与上下文的扩展收益递减，深度才是出路：让 Post-LN 在深层下稳定、可表达 |
| 2026-01 | ◆ [可学习乘子](https://arxiv.org/abs/2601.04890) `arXiv:2601.04890` | TII | 给矩阵层加可学习的标量/向量乘子，释放被权重衰减束缚的尺度 |
| 2026-01 | ◆ [Deep Delta Learning](https://arxiv.org/abs/2601.00417) `arXiv:2601.00417` |  | 把残差的恒等捷径换成可学习的"delta"更新（秩一擦写），放松严格加法残差的限制 |
| 2025-12 | ★ ◆ [mHC](https://arxiv.org/abs/2512.24880) `arXiv:2512.24880` | DeepSeek | 把超连接的残差混合矩阵约束到双随机矩阵流形，恢复恒等映射性质，在大规模训练中稳定且高效 |
| 2025-12 | ◆ [快速超参迁移的机制](https://arxiv.org/abs/2512.22768) `arXiv:2512.22768` |  | 解释为何超参能在规模间迁移 |
| 2025-12 | ◆ [有界双曲正切](https://arxiv.org/abs/2601.09719) `arXiv:2601.09719` |  | 替代 Pre-LN 的稳定高效方案 |
| 2025-12 | ◆ [完整的超参迁移](https://arxiv.org/abs/2512.22382) `arXiv:2512.22382` |  | 跨模块、宽度、深度、批大小与训练时长的统一超参迁移 |
| 2025-12 | ◆ [无限深网络的特征学习动力学](https://arxiv.org/abs/2512.21075) `arXiv:2512.21075` |  | 深度趋于无穷时仍保持特征学习的参数化与动力学 |
| 2025-12 | ◆ [更强的免归一化 Transformer](https://arxiv.org/abs/2512.10938) `arXiv:2512.10938` | Princeton | 在 DyT 之后系统搜索逐点函数，让免归一化 Transformer 超过归一化版本 |
| 2025-11 | ◆ [Virtual Width Networks](https://arxiv.org/abs/2511.11238) `arXiv:2511.11238` | ByteDance Seed | 嵌入维度变宽而主干计算不变，获得更宽表征的好处 |
| 2025-11 | ◆ [LayerNorm 对记忆与泛化的影响](https://arxiv.org/abs/2511.10566) `arXiv:2511.10566` |  | Transformer 中 LayerNorm 的位置影响记忆与泛化 |
| 2025-11 | ◆ [归一化层的隐藏力量](https://arxiv.org/abs/2511.00958) `arXiv:2511.00958` |  | 归一化层带来指数级的容量控制 |
| 2025-10 | 🆕 [正确调权重衰减得到稳健的逐层扩展规则](https://arxiv.org/abs/2510.15262) `arXiv:2510.15262` |  | μP 的学习率迁移在 AdamW 下需配合按宽度缩放的权重衰减才在稳态成立 |
| 2025-08 | 🆕 [巨大激活的训练动力学](https://arxiv.org/abs/2508.03616) `arXiv:2508.03616` |  | 追踪巨大激活在训练过程中的出现与演化 |
| 2025-06 | 🆕 [离群值安全预训练](https://arxiv.org/abs/2506.19697) `arXiv:2506.19697` |  | 从预训练开始抑制激活离群值，使模型天然适合 4-bit 量化 |
| 2025-05 | 🆕 [DiT 的 μP](https://arxiv.org/abs/2505.15270) `arXiv:2505.15270` |  | 把 μP 推广到扩散 Transformer，小模型调参迁移到大模型 |
| 2025-04 | 🆕 [ZClip](https://arxiv.org/abs/2504.02507) `arXiv:2504.02507` | BluOrion | 基于 z 分数的自适应梯度裁剪，缓解预训练损失尖峰 |
| 2025-03 | 🆕 [Frac-Connections](https://arxiv.org/abs/2503.14125) `arXiv:2503.14125` | ByteDance Seed | 把隐状态切分成分数份做超连接，减少显存开销 |
| 2025-03 | ★ 🆕 [DyT](https://arxiv.org/abs/2503.10622) `arXiv:2503.10622` | Meta/NYU | 用逐元素 tanh(αx) 替代归一化层，Transformer 在视觉、语言、扩散上不掉点 |
| 2025-03 | 🆕 [HybridNorm](https://arxiv.org/abs/2503.04598) `arXiv:2503.04598` |  | 注意力内用 QKV 归一化、FFN 用 Post-Norm 的混合归一化 |
| 2025-02 | 🆕 [深度的诅咒](https://arxiv.org/abs/2502.05795) `arXiv:2502.05795` |  | Pre-LN 导致输出方差随深度增长，深层近乎恒等；按深度缩放 LayerNorm 可修复 |
| 2024-10 | [nGPT](https://arxiv.org/abs/2410.01131) `arXiv:2410.01131` | NVIDIA | 所有向量都归一化到超球面上的 Transformer，收敛快 4–20 倍 |
| 2024-09 | [Hyper-Connections](https://arxiv.org/abs/2409.19606) `arXiv:2409.19606` | ByteDance Seed | 把单一残差流扩展为多条可学习混合的残差流 |
| 2024-02 | [Massive Activations](https://arxiv.org/abs/2402.17762) `arXiv:2402.17762` | CMU/Meta | LLM 中少数取值极大的激活充当固定偏置，与注意力汇聚点相关 |
| 2023-10 | [特征学习的谱条件](https://arxiv.org/abs/2310.17813) `arXiv:2310.17813` | Microsoft/MIT | 用权重与更新的谱范数缩放统一推导 μP |
| 2023-09 | [Attention Sink / StreamingLLM](https://arxiv.org/abs/2309.17453) `arXiv:2309.17453` | MIT/Meta | 首 token 吸收大量注意力（注意力汇聚点）；保留它即可流式处理无限长输入 |
| 2023-09 | [小模型复现训练不稳定](https://arxiv.org/abs/2309.14322) `arXiv:2309.14322` | Google DeepMind | 在小模型上复现注意力 logit 增长与输出 logit 发散，研究 QK-norm 与 z-loss 等对策 |
| 2023-02 | [ViT-22B（QK-Norm）](https://arxiv.org/abs/2302.05442) `arXiv:2302.05442` | Google | 把 ViT 扩到 22B，用 QK 归一化解决注意力 logit 爆炸 |
| 2022-03 | ★ [μP / μTransfer](https://arxiv.org/abs/2203.03466) `arXiv:2203.03466` | Microsoft/OpenAI | 最大更新参数化下，小模型调好的超参数可以零样本迁移到大模型 |
| 2022-03 | [DeepNet](https://arxiv.org/abs/2203.00555) `arXiv:2203.00555` | Microsoft | DeepNorm 放大残差 + 缩小初始化，把 Transformer 训练到 1000 层 |
| 2020-11 | [Tensor Programs IV](https://arxiv.org/abs/2011.14522) `arXiv:2011.14522` | Microsoft | 无限宽极限下的"特征学习"参数化：标准参数化与 NTK 参数化都会丢失特征学习 |
| 2020-02 | [SwiGLU](https://arxiv.org/abs/2002.05202) `arXiv:2002.05202` | Google | 门控线性单元替换 FFN 激活，几乎所有 LLM 采用 |
| 2020-02 | [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) `arXiv:2002.04745` | Microsoft/PKU | 理论说明 Post-LN 需要 warmup 的原因，Pre-LN 训练更稳定 |
| 2019-10 | [RMSNorm](https://arxiv.org/abs/1910.07467) `arXiv:1910.07467` | Edinburgh | 只做均方根缩放、不减均值的 LayerNorm 简化版，LLM 标配 |

### 泛化与学习理论（195）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [深度线性对角网络上 (S)GD 的隐式偏置](https://arxiv.org/abs/2610.39717) `arXiv:2610.39717` |  | 经典可解模型中的隐式偏置 |
| 2026-10 | 🔥 [不变 CNN 的记忆容量](https://arxiv.org/abs/2610.38938) `arXiv:2610.38938` |  | 不变卷积网络能记住多少样本 |
| 2026-10 | 🔥 [深度线性网络中特征学习的早期征兆](https://arxiv.org/abs/2610.37834) `arXiv:2610.37834` |  | 特征学习在训练早期如何显现 |
| 2026-10 | 🔥 [完全坍缩并非最优](https://arxiv.org/abs/2610.36460) `arXiv:2610.36460` |  | 长尾识别中可控的部分神经坍缩 |
| 2026-10 | 🔥 [grokking 中的子空间结晶](https://arxiv.org/abs/2610.35112) `arXiv:2610.35112` |  | 秩压缩追踪 grokking 相变 |
| 2026-10 | 🔥 [Transformer 是跨任务的自适应核学习者](https://arxiv.org/abs/2610.24441) `arXiv:2610.24441` |  | ICL 中注意力学到自适应核 |
| 2026-10 | 🔥 [智能体式上下文学习](https://arxiv.org/abs/2610.23051) `arXiv:2610.23051` |  | 闭环交互中 Transformer 能学到哪些算法 |
| 2026-10 | 🔥 [Transformer 的条件无毛定理](https://arxiv.org/abs/2610.24133) `arXiv:2610.24133` |  | 不同提示后接相同续写，隐状态差异近似指数衰减：模型如何"忘记"提示 |
| 2026-10 | 🔥 [神经坍缩的深宽闭式律](https://arxiv.org/abs/2610.21820) `arXiv:2610.21820` |  | 深度线性网络中神经坍缩的闭式深宽关系 |
| 2026-10 | 🔥 [专门化之前的上下文学习](https://arxiv.org/abs/2610.21901) `arXiv:2610.21901` |  | 可解主题模型中，ICL 先于专门化出现 |
| 2026-10 | 🔥 [grokking 的统计小品](https://arxiv.org/abs/2610.18811) `arXiv:2610.18811` |  | 从统计学角度对 grokking 现象的若干分析 |
| 2026-10 | 🔥 [记忆的 Fisher 几何](https://arxiv.org/abs/2610.16161) `arXiv:2610.16161` |  | 用 Fisher 信息刻画神经网络中的记忆 |
| 2026-10 | 🔥 [度量学到的等变性](https://arxiv.org/abs/2610.17269) `arXiv:2610.17269` |  | 信息论方法度量模型学到的等变 |
| 2026-10 | 🔥 [为何重尾权重谱能预测模型质量](https://arxiv.org/abs/2610.13123) `arXiv:2610.13123` |  | 重尾自正则化的理论解释 |
| 2026-10 | 🔥 [注意力非线性带来逆深度扩展](https://arxiv.org/abs/2610.11063) `arXiv:2610.11063` |  | 注意力的非线性如何导致"深度越深收益越小" |
| 2026-10 | 🔥 [平均场训练中非线性观测量的涨落](https://arxiv.org/abs/2610.09768) `arXiv:2610.09768` |  | 有限宽修正：平均场训练的涨落 |
| 2026-10 | 🔥 [精确解体积与长度泛化](https://arxiv.org/abs/2610.07676) `arXiv:2610.07676` |  | 精确解在参数空间中的体积决定能否长度泛化 |
| 2026-10 | 🔥 [有限特征联想记忆的几何容量理论](https://arxiv.org/abs/2610.09056) `arXiv:2610.09056` |  | 基于几何的容量理论 |
| 2026-10 | 🔥 [对称感知的特征学习](https://arxiv.org/abs/2610.08420) `arXiv:2610.08420` |  | 多指标模型上利用对称性带来多项式级分离 |
| 2026-10 | 🔥 [语言模型中柏拉图表征的理论](https://arxiv.org/abs/2610.07168) `arXiv:2610.07168` |  | 给出柏拉图表征的理论模型 |
| 2026-10 | 🔥 [长度泛化需要恰当的正则化](https://arxiv.org/abs/2610.04518) `arXiv:2610.04518` |  | 正则化对长度泛化的必要性 |
| 2026-10 | 🔥 [权重衰减与神经元凝聚](https://arxiv.org/abs/2610.04533) `arXiv:2610.04533` |  | 两层 ReLU 网络在权重衰减下的三阶段分析 |
| 2026-10 | 🔥 [单指标目标的上下文学习](https://arxiv.org/abs/2610.01712) `arXiv:2610.01712` |  | 比较核学习者与特征学习者的 ICL 能力 |
| 2026-09 | 🔥 [没见过混沌也能学会混沌](https://arxiv.org/abs/2609.38814) `arXiv:2609.38814` |  | 只看局部观测训练的自回归 Transformer，能外推出动力系统的全局结构 |
| 2026-09 | 🔥 [参数对称性决定表征几何](https://arxiv.org/abs/2609.39078) `arXiv:2609.39078` |  | 过参数化非线性网络中，参数对称性决定学到的表征几何 |
| 2026-09 | 🔥 [低损失区域错位导致 grokking](https://arxiv.org/abs/2610.00620) `arXiv:2610.00620` |  | 训练与测试的低损失区域不重合是 grokking 的成因 |
| 2026-09 | 🔥 [行为趋同而表征不趋同](https://arxiv.org/abs/2609.37836) `arXiv:2609.37836` |  | 训练历史的持久依赖 |
| 2026-09 | 🔥 [最小范数插值视角下的 grokking](https://arxiv.org/abs/2609.38453) `arXiv:2609.38453` |  | grokking 与最小范数插值解的关系 |
| 2026-09 | 🔥 [在策略更新方向决定泛化](https://arxiv.org/abs/2609.36659) `arXiv:2609.36659` | UCAS | 后训练的泛化来自在策略的参数更新方向 |
| 2026-09 | 🔥 [分布外 ICL 的注意力动力学理论](https://arxiv.org/abs/2609.36448) `arXiv:2609.36448` |  | 注意力训练动力学如何决定分布外上下文学习 |
| 2026-09 | 🔥 [黎曼流形上的内禀联想记忆](https://arxiv.org/abs/2609.35948) `arXiv:2609.35948` |  | 曲率、容量与涌现 |
| 2026-09 | 🔥 [分解语言模型记忆的训练动力学](https://arxiv.org/abs/2609.34933) `arXiv:2609.34933` |  | 记忆在训练中如何形成与遗忘 |
| 2026-09 | 🔥 [一般范数与分布下的良性过拟合](https://arxiv.org/abs/2609.33675) `arXiv:2609.33675` |  | 良性过拟合理论推广到一般范数与分布 |
| 2026-09 | 🔥 [非参数 ICL 的极小极大最优](https://arxiv.org/abs/2609.31458) `arXiv:2609.31458` |  | 几何复杂度增长下非参数上下文学习的最优率 |
| 2026-09 | 🔥 [Transformer 是跨任务学习者](https://arxiv.org/abs/2609.29060) `arXiv:2609.29060` |  | 共享结构驱动上下文学习的样本效率 |
| 2026-09 | 🔥 [Grokking 的谱理论](https://arxiv.org/abs/2609.26679) `arXiv:2609.26679` |  | 权重衰减诱导特征学习，从谱角度解释 grokking |
| 2026-09 | 🔥 [神经谱容量](https://arxiv.org/abs/2609.23087) `arXiv:2609.23087` |  | 仅凭网络规格（深宽比、头数、FFN 分配）度量并设计架构容量，补足参数量与 FLOPs |
| 2026-09 | 🔥 [双下降即最小作用量原理](https://arxiv.org/abs/2609.19076) `arXiv:2609.19076` |  | 用变分（最小作用量）原理解释双下降 |
| 2026-09 | 🔥 [结构化特征过拟合、随机特征 grok](https://arxiv.org/abs/2609.15047) `arXiv:2609.15047` |  | 特征结构决定是过拟合还是延迟泛化 |
| 2026-09 | 🔥 [跨模态的上下文学习趋同涌现](https://arxiv.org/abs/2609.14011) `arXiv:2609.14011` |  | 文本、音频等不同模态的下一词预测器都涌现出相似的少样本 ICL |
| 2026-09 | 🔥 [稠密联想记忆的序列检索](https://arxiv.org/abs/2609.13987) `arXiv:2609.13987` |  | 渐近动力学与存储容量 |
| 2026-09 | 🔥 [良性地形与最坏情形困难可以共存](https://arxiv.org/abs/2609.13057) `arXiv:2609.13057` |  | 损失地形良性并不排除计算上的最坏情形困难 |
| 2026-09 | 🔥 [现代 Hopfield 模型中的层级原型涌现](https://arxiv.org/abs/2609.12079) `arXiv:2609.12079` |  | 层级原型的涌现 |
| 2026-09 | 🔥 [通过压缩实现长度泛化](https://arxiv.org/abs/2609.08851) `arXiv:2609.08851` |  | 压缩视角的长度泛化 |
| 2026-09 | 🔥 [等变性打破学习率](https://arxiv.org/abs/2609.08381) `arXiv:2609.08381` |  | 等变网络中最优学习率的变化 |
| 2026-08 | 🔥 [机器学习中的旋转等变教程](https://arxiv.org/abs/2608.31045) `arXiv:2608.31045` |  | 旋转等变的系统教程 |
| 2026-08 | 🔥 [线性 ICL 中淬火临界性的朗道理论](https://arxiv.org/abs/2608.28059) `arXiv:2608.28059` |  | 用朗道相变理论描述线性上下文学习 |
| 2026-08 | 🔥 [从损失地形到多样的特征学习](https://arxiv.org/abs/2608.28948) `arXiv:2608.28948` |  | 地形结构如何带来多样的特征 |
| 2026-08 | 🔥 [长度泛化的代数分解理论](https://arxiv.org/abs/2608.13433) `arXiv:2608.13433` |  | Transformer 长度泛化的代数刻画 |
| 2026-08 | 🔥 [高斯多指标模型的谱相变](https://arxiv.org/abs/2608.12183) `arXiv:2608.12183` |  | 多指标模型的谱方法相变阈值 |
| 2026-08 | 🔥 [减少等变网络中的对称性增加](https://arxiv.org/abs/2608.12010) `arXiv:2608.12010` |  | 等变网络中意外对称增加的问题 |
| 2026-08 | 🔥 [SFT 冲突，RL 共存](https://arxiv.org/abs/2608.03573) `arXiv:2608.03573` | CAS | 多任务下 SFT 的梯度相互冲突而 RL 更新可共存 |
| 2026-07 | 🔥 [权重衰减时钟上的 grokking](https://arxiv.org/abs/2607.23967) `arXiv:2607.23967` |  | 软破缺对称给出的速率层级解释 grokking 的时间尺度 |
| 2026-07 | 🔥 [稠密联想记忆的自由能地形](https://arxiv.org/abs/2607.19195) `arXiv:2607.19195` |  | 稠密联想记忆（现代 Hopfield）的自由能地形 |
| 2026-07 | 🔥 [语义空间的几何](https://arxiv.org/abs/2607.17146) `arXiv:2607.17146` |  | 把 Transformer 写成语义纤维丛上的积分—微分方程的连续几何框架 |
| 2026-07 | 🔥 [从第一性原理看可提取记忆](https://arxiv.org/abs/2607.12649) `arXiv:2607.12649` |  | 语言模型可提取记忆的第一性原理分析 |
| 2026-07 | 🔥 [语言模型禁止神经坍缩](https://arxiv.org/abs/2607.09487) `arXiv:2607.09487` |  | 信息下限使语言模型不会完全神经坍缩 |
| 2026-07 | 🔥 [隐式偏置的解剖](https://arxiv.org/abs/2607.07156) `arXiv:2607.07156` |  | 训练中的信息分配视角 |
| 2026-07 | 🔥 [组合学习的函数空间二分](https://arxiv.org/abs/2607.06382) `arXiv:2607.06382` |  | NTK 在组合学习上有指数级次优性 |
| 2026-07 | 🔥 [多尺度单指标模型](https://arxiv.org/abs/2607.03347) `arXiv:2607.03347` |  | 层级特征学习的风格化模型 |
| 2026-07 | 🔥 [多元 ReLU 网络中 SGD 的隐式偏置](https://arxiv.org/abs/2607.03613) `arXiv:2607.03613` |  | SGD 导致有效宽度坍缩 |
| 2026-07 | 🔥 [傅里叶预条件特征学习](https://arxiv.org/abs/2607.02199) `arXiv:2607.02199` |  | 用傅里叶预条件加速网络学到有用特征 |
| 2026-07 | 🔥 [从逼近到涌现](https://arxiv.org/abs/2607.01311) `arXiv:2607.01311` |  | 一个深度学习理论框架 |
| 2026-06 | 🔥 [用统计力学解释学习与记忆](https://arxiv.org/abs/2606.31110) `arXiv:2606.31110` |  | 机器学习与记忆的统计力学解释 |
| 2026-06 | 🔥 [平均场 Langevin 中统计特征学习的几何](https://arxiv.org/abs/2606.31429) `arXiv:2606.31429` |  | 平均场 Langevin 动力学下特征学习的几何 |
| 2026-06 | 🔥 [多指标经验风险最小化的复本对称破缺](https://arxiv.org/abs/2606.28573) `arXiv:2606.28573` |  | 复本对称破缺与算法阈值 |
| 2026-06 | 🔥 [终身 ICL 需要参数化注意力](https://arxiv.org/abs/2606.25342) `arXiv:2606.25342` |  | 终身上下文学习需要参数化形式的注意力 |
| 2026-06 | 🔥 [Fisher 几何锐度](https://arxiv.org/abs/2606.20469) `arXiv:2606.20469` |  | SGD 偏向平坦极小值的 Fisher 几何解释 |
| 2026-06 | 🔥 [深层 Transformer 层级建模的表达力](https://arxiv.org/abs/2606.17522) `arXiv:2606.17522` |  | 有界深度下的层级建模能力分析 |
| 2026-06 | 🔥 [深度何时对 ICL 重要](https://arxiv.org/abs/2606.16694) `arXiv:2606.16694` |  | 深层 Transformer 的自适应推理 |
| 2026-06 | 🔥 [网络能达到最优计算—统计权衡吗](https://arxiv.org/abs/2606.15219) `arXiv:2606.15219` |  | 单指标模型上的分析 |
| 2026-06 | 🔥 [注意力中的相变](https://arxiv.org/abs/2606.12058) `arXiv:2606.12058` |  | 复制头涌现的贝叶斯理论 |
| 2026-06 | 🔥 [超越神经坍缩](https://arxiv.org/abs/2606.08985) `arXiv:2606.08985` |  | 模运算中任务内禀几何决定表征 |
| 2026-06 | 🔥 [平坦性与泛化](https://arxiv.org/abs/2606.04429) `arXiv:2606.04429` |  | 齐次网络学习多指标模型时平坦极小值的泛化 |
| 2026-06 | 🔥 [网络可证明地学到群组合的谱表示](https://arxiv.org/abs/2606.02993) `arXiv:2606.02993` |  | 两层网络学习有限群乘法时，可证明地学到群的不可约表示（傅里叶）特征 |
| 2026-06 | 🔥 [ICL 中思维链的渐近理论](https://arxiv.org/abs/2606.03217) `arXiv:2606.03217` |  | 上下文学习中思维链的渐近分析 |
| 2026-05 | 🔥 [轻度过参数化 ReLU 网络的增量学习](https://arxiv.org/abs/2605.27097) `arXiv:2605.27097` |  | 正交数据上的增量学习与隐式偏置 |
| 2026-05 | 🔥 [学习的两种速度](https://arxiv.org/abs/2605.27078) `arXiv:2605.27078` |  | 把 grokking 与双下降分解为表征与读出两种速度的学习 |
| 2026-05 | 🔥 [表征对齐依赖线性结构](https://arxiv.org/abs/2605.28870) `arXiv:2605.28870` |  | 表征对齐度量依赖于线性结构 |
| 2026-05 | 🔥 [特征彩票](https://arxiv.org/abs/2605.24057) `arXiv:2605.24057` |  | 概念涌现的分岔理论 |
| 2026-05 | 🔥 [深度的隐式偏置](https://arxiv.org/abs/2605.23087) `arXiv:2605.23087` |  | 从神经坍缩到 softmax 码：深度带来的隐式偏置 |
| 2026-05 | 🔥 [平衡关系泛化与记忆的数学理论](https://arxiv.org/abs/2605.22972) `arXiv:2605.22972` |  | 何时应记忆、何时应泛化 |
| 2026-05 | 🔥 [特征学习如何重塑函数空间](https://arxiv.org/abs/2605.17718) `arXiv:2605.17718` |  | 特征学习对网络所表示函数空间的改变 |
| 2026-05 | 🔥 [Transformer 用归一化 GD 做上下文逻辑回归](https://arxiv.org/abs/2605.06609) `arXiv:2605.06609` |  | Transformer 高效实现归一化梯度下降 |
| 2026-05 | 🔥 [非线性回归的 ICL](https://arxiv.org/abs/2605.05176) `arXiv:2605.05176` |  | 注意力充当特征提取器，实现非线性回归的上下文学习 |
| 2026-05 | 🔥 [估计深度学习中的隐式正则](https://arxiv.org/abs/2605.05436) `arXiv:2605.05436` |  | 从训练中实证估计隐式正则的形式与强度 |
| 2026-05 | 🔥 [标签不平衡如何塑造几何](https://arxiv.org/abs/2605.01897) `arXiv:2605.01897` |  | 多标签神经坍缩的谱分析 |
| 2026-04 | ◆ [任务算术中的权重解耦](https://arxiv.org/abs/2604.17078) `arXiv:2604.17078` |  | 什么内在性质导致任务向量可以互不干扰地相加 |
| 2026-04 | ◆ [上下文学习的不同机制](https://arxiv.org/abs/2604.12151) `arXiv:2604.12151` |  | Transformer 中存在多种不同的 ICL 机制 |
| 2026-04 | ◆ [高维两层 ReLU 网络的局部极小](https://arxiv.org/abs/2604.09412) `arXiv:2604.09412` |  | 对局部极小值的精确刻画 |
| 2026-04 | ◆ [隐式课程假设](https://arxiv.org/abs/2604.08510) `arXiv:2604.08510` |  | 预训练中技能按大致固定的顺序出现：损失曲线之外，模型何时学会什么 |
| 2026-04 | ◆ [良性过拟合的输运几何](https://arxiv.org/abs/2604.08625) `arXiv:2604.08625` |  | 从最优输运几何看良性过拟合 |
| 2026-04 | ◆ [grokking 是维度相变](https://arxiv.org/abs/2604.04655) `arXiv:2604.04655` |  | 把 grokking 刻画为表示有效维度的相变 |
| 2026-04 | ◆ [grokking 处的维度临界性](https://arxiv.org/abs/2604.16431) `arXiv:2604.16431` |  | MLP 与 Transformer 在 grokking 时都表现出维度临界性 |
| 2026-04 | ◆ [知识蒸馏的几何极限](https://arxiv.org/abs/2604.04037) `arXiv:2604.04037` |  | 由叠加理论导出最小宽度定理 |
| 2026-03 | ◆ [神经坍缩动力学](https://arxiv.org/abs/2604.00230) `arXiv:2604.00230` |  | 深度、激活、正则与特征范数阈值 |
| 2026-03 | ◆ [特征学习核需要打破数据对称](https://arxiv.org/abs/2604.00316) `arXiv:2604.00316` |  | 特征学习核要泛化必须打破数据对称性 |
| 2026-03 | ◆ [可调的软等变性](https://arxiv.org/abs/2603.26657) `arXiv:2603.26657` |  | 把权重投影到等变子空间附近，构造可控、有理论保证的软等变模型 |
| 2026-03 | ◆ [grokking 是可证伪的有限尺寸相变](https://arxiv.org/abs/2603.24746) `arXiv:2603.24746` |  | 用有限尺寸标度检验 grokking 是否为真正的相变 |
| 2026-03 | ◆ [商几何与有效曲率](https://arxiv.org/abs/2603.21502) `arXiv:2603.21502` |  | 浅层网络参数对称性商掉后的几何与隐式偏置 |
| 2026-03 | ◆ [平均场下大网络的隐式正则](https://arxiv.org/abs/2603.20892) `arXiv:2603.20892` |  | 用平均场表述刻画宽网络的隐式正则 |
| 2026-03 | ◆ [重尾神经网络的近临界动力学](https://arxiv.org/abs/2603.18478) `arXiv:2603.18478` |  | 重尾权重带来稳健的近临界动力学 |
| 2026-03 | ◆ [看似无用特征的涌现](https://arxiv.org/abs/2603.14087) `arXiv:2603.14087` |  | 下一词预测器为何学到对当前预测无用的特征 |
| 2026-03 | ◆ [随规模涌现的低阶线性深度动力学](https://arxiv.org/abs/2603.12541) `arXiv:2603.12541` |  | 模型越大，层间演化越接近低阶线性动力系统 |
| 2026-03 | ◆ [训练动力学的高斯比较定理](https://arxiv.org/abs/2603.09310) `arXiv:2603.09310` |  | 用于分析训练动力学的高斯比较工具 |
| 2026-03 | ◆ [奇异学习理论看 grokking](https://arxiv.org/abs/2603.01192) `arXiv:2603.01192` |  | 用奇异学习理论的"盆地选择"解释 grokking |
| 2026-02 | ◆ [组合泛化需要线性正交表征](https://arxiv.org/abs/2602.24264) `arXiv:2602.24264` |  | 证明视觉嵌入模型要实现组合泛化，表征必须线性且各因素正交 |
| 2026-02 | ◆ [超越 NNGP](https://arxiv.org/abs/2602.22925) `arXiv:2602.22925` |  | 用大偏差理论刻画贝叶斯神经网络的特征学习 |
| 2026-02 | ◆ [预训练如何塑造微调的归纳偏置](https://arxiv.org/abs/2602.20062) `arXiv:2602.20062` |  | 理论分析预训练对后续微调归纳偏置的影响 |
| 2026-02 | ◆ [群等变学习的定量逼近率](https://arxiv.org/abs/2602.20370) `arXiv:2602.20370` |  | 群等变网络的逼近理论 |
| 2026-02 | ◆ [贝叶斯彩票假设](https://arxiv.org/abs/2602.18825) `arXiv:2602.18825` |  | 彩票假设的贝叶斯版本 |
| 2026-02 | ◆ [越大越安全](https://arxiv.org/abs/2602.17743) `arXiv:2602.17743` |  | 上下文学习的可证明鲁棒性随容量增加 |
| 2026-02 | ◆ [多任务 grokking 的几何](https://arxiv.org/abs/2602.18523) `arXiv:2602.18523` |  | 横向不稳定、叠加与权重衰减相图 |
| 2026-02 | ◆ [模加法的机制与动力学](https://arxiv.org/abs/2602.16849) `arXiv:2602.16849` |  | 完整解释模加法学习：单频傅里叶特征、彩票式竞争与 grokking 的训练动力学 |
| 2026-02 | ◆ [优化器的选择影响神经坍缩的出现](https://arxiv.org/abs/2602.16642) `arXiv:2602.16642` |  | 不同优化器下神经坍缩是否出现 |
| 2026-02 | ◆ [RNN 与 DNN 的统一特征学习理论](https://arxiv.org/abs/2602.15593) `arXiv:2602.15593` |  | 统一刻画循环网络与深网络的特征学习 |
| 2026-02 | ◆ [从高斯过程到特征学习（讲义）](https://arxiv.org/abs/2602.12855) `arXiv:2602.12855` |  | 从 NNGP/NTK 到特征学习理论的讲义 |
| 2026-02 | ◆ [Transformer 长度泛化界](https://arxiv.org/abs/2603.02238) `arXiv:2603.02238` |  | 给出 Transformer 长度泛化的理论界 |
| 2026-02 | ◆ [logit 正则的隐式偏置](https://arxiv.org/abs/2602.12039) `arXiv:2602.12039` |  | logit 正则改变分类器收敛方向 |
| 2026-02 | ◆ [多指标模型的统计—计算权衡](https://arxiv.org/abs/2602.09959) `arXiv:2602.09959` |  | 用调和分析刻画学习多指标模型的权衡 |
| 2026-02 | ◆ [用归纳偏置解释 Transformer 的 grokking](https://arxiv.org/abs/2602.06702) `arXiv:2602.06702` |  | 架构与优化的归纳偏置如何决定 grokking |
| 2026-02 | ◆ [光滑激活 MLP 的低秩训练动力学](https://arxiv.org/abs/2602.06208) `arXiv:2602.06208` |  | 光滑激活下训练动力学涌现低秩结构 |
| 2026-02 | ◆ [SGD 学习多指标模型的局限](https://arxiv.org/abs/2602.05704) `arXiv:2602.05704` |  | 超越统计查询下界的 SGD 局限 |
| 2026-02 | ◆ [多层交叉注意力对多模态 ICL 可证明最优](https://arxiv.org/abs/2602.04872) `arXiv:2602.04872` |  | 多模态上下文学习中多层交叉注意力的最优性 |
| 2026-02 | ◆ [预训练何时伤害 LoRA 微调](https://arxiv.org/abs/2602.02855) `arXiv:2602.02855` |  | 单指标模型上的动力学分析 |
| 2026-02 | ◆ [全批量 GD 胜过单遍 SGD](https://arxiv.org/abs/2602.02431) `arXiv:2602.02431` |  | 单指标学习中两者的样本复杂度分离 |
| 2026-02 | ◆ [grokking 的后期泛化坍缩](https://arxiv.org/abs/2602.02859) `arXiv:2602.02859` |  | "反 grokking"：泛化后长期训练又出现坍缩，可用权重谱检测 |
| 2026-02 | ◆ [特征学习的相变](https://arxiv.org/abs/2602.01434) `arXiv:2602.01434` |  | 神经网络特征学习中的相变 |
| 2026-02 | ◆ [叠加统一幂律训练动力学](https://arxiv.org/abs/2602.01045) `arXiv:2602.01045` |  | 特征叠加可以统一解释训练损失的幂律下降 |
| 2026-01 | ◆ [可辨识的等变网络是逐层等变的](https://arxiv.org/abs/2601.21645) `arXiv:2601.21645` |  | 等变网络的结构定理 |
| 2026-01 | ◆ [表征几何与泛化](https://arxiv.org/abs/2602.00130) `arXiv:2602.00130` |  | 13 个架构族的 ImageNet 模型中，无监督的有效维度强有力地预测准确率 |
| 2026-01 | ◆ [多遍 SGD 在多指标模型上的高维动力学](https://arxiv.org/abs/2601.21093) `arXiv:2601.21093` |  | 重复使用数据如何改变 SGD 的学习动力学 |
| 2026-01 | ◆ [岭回归中可证的 grokking](https://arxiv.org/abs/2601.19791) `arXiv:2601.19791` |  | 在岭回归中严格证明 grokking 的出现 |
| 2026-01 | ◆ [无标注数据可证明增强 ICL](https://arxiv.org/abs/2601.10058) `arXiv:2601.10058` |  | Transformer 在上下文中利用无标注数据的可证明收益 |
| 2026-01 | ◆ [几何稳定性](https://arxiv.org/abs/2601.09173) `arXiv:2601.09173` |  | 表征的另一条轴：从特征坐标子集能多稳定地恢复整体几何 |
| 2026-01 | ◆ [随机学习隐式偏置的逆向设计](https://arxiv.org/abs/2601.06597) `arXiv:2601.06597` |  | 几何视角理解并设计随机学习算法的隐式偏置 |
| 2026-01 | ◆ [贝叶斯教师的 SGD 知识蒸馏](https://arxiv.org/abs/2601.01484) `arXiv:2601.01484` |  | 理论与指南 |
| 2026-01 | ◆ [联想记忆的随机热力学](https://arxiv.org/abs/2601.01253) `arXiv:2601.01253` |  | 联想记忆的随机热力学分析 |
| 2026-01 | ◆ [生物合理的指数容量稠密联想记忆](https://arxiv.org/abs/2601.00984) `arXiv:2601.00984` |  | 生物合理实现 |
| 2025-12 | ◆ [学习动力学的等变工具箱](https://arxiv.org/abs/2512.21447) `arXiv:2512.21447` |  | 用等变性分析学习动力学 |
| 2025-12 | ◆ [自回归模型中涌现的时间抽象](https://arxiv.org/abs/2512.20605) `arXiv:2512.20605` | Google | 预训练自回归模型的残差流中存在可用于分层 RL 的时间抽象 |
| 2025-12 | ◆ [从捷径到归纳头](https://arxiv.org/abs/2512.18634) `arXiv:2512.18634` |  | 数据多样性决定 Transformer 选择捷径还是归纳头算法 |
| 2025-12 | ◆ [神经正切与无限宽网络的数学基础](https://arxiv.org/abs/2512.08264) `arXiv:2512.08264` |  | NTK 与无限宽极限的系统数学讲义 |
| 2025-11 | ◆ [范畴等变深度学习](https://arxiv.org/abs/2511.18417) `arXiv:2511.18417` |  | 范畴等变网络与通用逼近 |
| 2025-11 | ◆ [预测预训练中归纳头的涌现](https://arxiv.org/abs/2511.16893) `arXiv:2511.16893` |  | 预测归纳头何时出现 |
| 2025-11 | ◆ [神经网络接近信息论极限地学习多指标模型](https://arxiv.org/abs/2511.15120) `arXiv:2511.15120` |  | 一般多指标模型上，梯度训练的网络样本复杂度接近信息论极限 |
| 2025-11 | ◆ [文本的内在维度](https://arxiv.org/abs/2511.15210) `arXiv:2511.15210` |  | 文本的内在维度由什么决定：从学术摘要到创意故事的系统测量 |
| 2025-11 | ◆ [基因组下一词预测器也会上下文学习](https://arxiv.org/abs/2511.12797) `arXiv:2511.12797` |  | 只在 DNA 序列上做下一词预测也涌现上下文学习：ICL 不依赖人类语言的特殊统计性质 |
| 2025-11 | ◆ [RLVR 偏离主成分学习](https://arxiv.org/abs/2511.08567) `arXiv:2511.08567` | Meta | RLVR 的更新避开权重的主方向，与 SFT 机制不同 |
| 2025-11 | ◆ [重访 NTK：宽度与深度的作用](https://arxiv.org/abs/2511.07272) `arXiv:2511.07272` |  | 大宽度与大深度共同作用下 NTK 理论的适用范围 |
| 2025-11 | ◆ [蒸馏动力学](https://arxiv.org/abs/2511.06848) `arXiv:2511.06848` |  | 理解 ViT 中基于特征的蒸馏 |
| 2025-11 | ◆ [词表上下文学习与位置编码](https://arxiv.org/abs/2511.06376) `arXiv:2511.06376` |  | 位置编码对 Transformer 词表 ICL 的好处 |
| 2025-11 | ◆ [尖锐极小值也能泛化](https://arxiv.org/abs/2511.04808) `arXiv:2511.04808` |  | 从数据视角看损失地形：尖锐与平坦并不决定泛化 |
| 2025-11 | ◆ [多头注意力的强彩票假设](https://arxiv.org/abs/2511.04217) `arXiv:2511.04217` |  | 证明随机初始化的多头注意力中隐藏着高性能子网络（强彩票） |
| 2025-11 | ◆ [面向 LLM 的语义信息论](https://arxiv.org/abs/2511.01202) `arXiv:2511.01202` |  | 以 token 而非比特为单位，建立 LLM 的语义信息论框架 |
| 2025-11 | ◆ [从记忆能力看思维链的威力](https://arxiv.org/abs/2511.01190) `arXiv:2511.01190` |  | 用记忆容量刻画思维链的能力 |
| 2025-11 | ◆ [归纳头的涌现](https://arxiv.org/abs/2511.01033) `arXiv:2511.01033` |  | 归纳头如何在训练中涌现 |
| 2025-11 | ◆ [grokking 的几何](https://arxiv.org/abs/2511.01938) `arXiv:2511.01938` |  | grokking 等价于在零损失流形上缓慢地最小化范数：权重衰减驱动的几何解释 |
| 2025-10 | 🆕 [长度泛化的定量界](https://arxiv.org/abs/2510.27015) `arXiv:2510.27015` |  | 长度泛化所需训练长度的定量刻画 |
| 2025-10 | 🆕 [内部概率与自洽性的理论](https://arxiv.org/abs/2510.15444) `arXiv:2510.15444` | Nanjing Univ | 联系模型内部概率与自洽性投票的测试时扩展理论 |
| 2025-08 | 🆕 [对抗样本源于叠加](https://arxiv.org/abs/2508.17456) `arXiv:2508.17456` |  | 对抗脆弱性可由特征叠加解释 |
| 2025-08 | 🆕 [DFT：SFT 的泛化](https://arxiv.org/abs/2508.05629) `arXiv:2508.05629` |  | SFT 等价于奖励稀疏且病态的策略梯度，按 token 概率重新缩放即可显著提升泛化 |
| 2025-07 | 🆕 [无需训练的学习](https://arxiv.org/abs/2507.16003) `arXiv:2507.16003` | Google | Transformer 块把上下文隐式转化为 MLP 权重的低秩更新，解释 ICL |
| 2025-07 | 🆕 [图灵机模仿器](https://arxiv.org/abs/2507.13332) `arXiv:2507.13332` | Shanghai AI Lab | 模仿图灵机执行轨迹的合成 CoT 带来长度泛化 |
| 2025-07 | 🆕 [Transformer 并不在上下文中做最小二乘](https://arxiv.org/abs/2507.09440) `arXiv:2507.09440` |  | 挑战"ICL 实现梯度下降/最小二乘"的解释 |
| 2025-06 | 🆕 [交替梯度流](https://arxiv.org/abs/2506.06489) `arXiv:2506.06489` |  | 两层网络中特征学习的交替梯度流理论 |
| 2025-06 | 🆕 [上下文学习的通用逼近](https://arxiv.org/abs/2506.05200) `arXiv:2506.05200` |  | Transformer 上下文学习的通用逼近理论 |
| 2025-05 | 🆕 [Adam 与 SGD 的隐式偏置](https://arxiv.org/abs/2505.24022) `arXiv:2505.24022` |  | SGD 偏向简单解，Adam 偏向更丰富的特征 |
| 2025-04 | 🆕 [真实世界中的 grokking](https://arxiv.org/abs/2504.20752) `arXiv:2504.20752` |  | 用数据增强提高推断事实比例，让 Transformer 在真实多跳推理上 grok |
| 2025-03 | 🆕 [深度学习并不那么神秘](https://arxiv.org/abs/2503.02113) `arXiv:2503.02113` | NYU | 良性过拟合、双下降、过参数化等现象都可用软归纳偏置与经典泛化框架解释 |
| 2025-02 | 🆕 [Grokking 的统计解释](https://arxiv.org/abs/2502.01774) `arXiv:2502.01774` |  | 把 grokking 解释为训练-测试分布差异引起的统计现象 |
| 2025-01 | 🆕 [数值稳定性边缘的 grokking](https://arxiv.org/abs/2501.04697) `arXiv:2501.04697` | Imperial | 没有正则时 grokking 被 Softmax Collapse（浮点误差）阻断，提出 StableMax 与 ⊥Grad |
| 2024-12 | 🆕 [长上下文下重新审视 ICL](https://arxiv.org/abs/2412.16926) `arXiv:2412.16926` |  | 上下文足够长时，示例选择的重要性下降，数据增强更关键 |
| 2024-12 | 🆕 [Grokking 的复杂度动力学](https://arxiv.org/abs/2412.09810) `arXiv:2412.09810` |  | 用压缩/复杂度视角追踪 grokking 中的记忆到泛化转变 |
| 2024-05 | ★ [柏拉图表征假说](https://arxiv.org/abs/2405.07987) `arXiv:2405.07987` | MIT | 不同模态、不同架构的大模型表征正在收敛到同一个"现实的统计模型" |
| 2024-04 | [语言模型物理学 3.3](https://arxiv.org/abs/2404.05405) `arXiv:2404.05405` | Meta/MBZUAI | 知识容量扩展律：语言模型约存储 2 bit 知识/参数 |
| 2023-09 | [语言模型物理学 3.1](https://arxiv.org/abs/2309.14316) `arXiv:2309.14316` | Meta/MBZUAI | 知识存储与提取：知识可被提取依赖于预训练数据的多样化改写 |
| 2023-06 | [可塑性丧失](https://arxiv.org/abs/2306.13812) `arXiv:2306.13812` | Alberta | 持续学习中深网络会逐渐失去学习新任务的能力；持续反向传播（重置低效单元）可缓解 |
| 2023-01 | [Grokking 的进度度量](https://arxiv.org/abs/2301.05217) `arXiv:2301.05217` | Anthropic/独立 | 逆向工程模加法网络（离散傅里叶电路），把 grokking 拆成记忆、电路形成、清理三阶段 |
| 2022-12 | [ICL 即梯度下降](https://arxiv.org/abs/2212.07677) `arXiv:2212.07677` | ETH/Google | 线性自注意力的一层前向可以实现一步梯度下降，解释上下文学习 |
| 2022-08 | [Transformer 能在上下文中学什么](https://arxiv.org/abs/2208.01066) `arXiv:2208.01066` | Stanford | Transformer 在上下文中可以学会线性回归等函数类，表现接近最优估计器 |
| 2022-01 | ★ [Grokking](https://arxiv.org/abs/2201.02177) `arXiv:2201.02177` | OpenAI | 小算法数据集上过拟合很久之后泛化突然出现 |
| 2021-06 | [深度学习理论原理（专著）](https://arxiv.org/abs/2106.10165) `arXiv:2106.10165` | Meta/MIT | 有效场论视角：有限宽度修正、临界初始化、表征学习的系统理论 |
| 2021-05 | [深度学习的数学分析](https://arxiv.org/abs/2105.04026) `arXiv:2105.04026` | Vienna/ETH | 表达、优化、泛化三方面的数学理论综述 |
| 2020-08 | [神经坍缩](https://arxiv.org/abs/2008.08186) `arXiv:2008.08186` | Stanford | 训练到末期，同类特征坍缩到类均值，类均值构成等角紧框架、分类器与之对齐 |
| 2019-12 | [Deep Double Descent](https://arxiv.org/abs/1912.02292) `arXiv:1912.02292` | OpenAI/Harvard | 在模型规模、训练轮数、数据量三个维度上都观察到双下降 |
| 2019-06 | [良性过拟合](https://arxiv.org/abs/1906.11300) `arXiv:1906.11300` | UC Berkeley | 刻画线性回归中插值噪声数据仍能泛化的条件 |
| 2018-12 | ★ [双下降（Belkin）](https://arxiv.org/abs/1812.11118) `arXiv:1812.11118` | Ohio State/Columbia | 测试误差随模型复杂度先 U 形再二次下降，调和偏差-方差与过参数化 |
| 2018-12 | [Lazy Training](https://arxiv.org/abs/1812.07956) `arXiv:1812.07956` | INRIA | "懒惰"训练区间：参数几乎不动、等价于线性化模型，与特征学习相对 |
| 2018-06 | ★ [NTK](https://arxiv.org/abs/1806.07572) `arXiv:1806.07572` | EPFL | 无限宽网络在梯度下降下等价于核回归（神经正切核） |
| 2018-04 | [平均场理论](https://arxiv.org/abs/1804.06561) `arXiv:1804.06561` | Stanford | 两层网络的梯度下降在宽度极限下是参数分布上的 Wasserstein 梯度流 |
| 2018-03 | [彩票假设](https://arxiv.org/abs/1803.03635) `arXiv:1803.03635` | MIT | 稠密网络中存在从初始化就能单独训练到同等精度的稀疏子网络 |
| 2017-10 | [梯度下降的隐式偏置](https://arxiv.org/abs/1710.10345) `arXiv:1710.10345` | TTIC/Technion | 线性可分数据上梯度下降收敛到最大间隔解：优化算法本身决定了泛化 |
| 2016-11 | ★ [重新思考泛化](https://arxiv.org/abs/1611.03530) `arXiv:1611.03530` | Google/MIT | 深网络能完美拟合随机标签，经典复杂度理论无法解释其泛化 |
| 2016-10 | [线性探针](https://arxiv.org/abs/1610.01644) `arXiv:1610.01644` | Montréal | 用线性分类器探测中间层表征包含的信息 |

### 扩展律与涌现（77）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [RL 扩展律实证](https://arxiv.org/abs/2610.33420) `arXiv:2610.33420` |  | 系统测量强化学习后训练的扩展规律 |
| 2026-10 | 🔥 [扩散训练的信息论扩展律](https://arxiv.org/abs/2610.33337) `arXiv:2610.33337` |  | 从信息论推导扩散模型训练的扩展律 |
| 2026-10 | 🔥 [重写合成数据的扩展律](https://arxiv.org/abs/2610.31867) `arXiv:2610.31867` |  | 计算与数据约束下合成数据的扩展律 |
| 2026-10 | 🔥 [扩展律的基础](https://arxiv.org/abs/2610.27789) `arXiv:2610.27789` |  | 扩展律的理论基础 |
| 2026-10 | 🔥 [随机特征二元模型的精确扩展律](https://arxiv.org/abs/2610.19281) `arXiv:2610.19281` |  | 一个可精确求解的语言模型扩展律 |
| 2026-10 | 🔥 [量化预训练的混合精度扩展律](https://arxiv.org/abs/2610.17637) `arXiv:2610.17637` |  | 不同部件不同精度下的扩展律 |
| 2026-10 | 🔥 [扩展重复数据预训练](https://arxiv.org/abs/2610.15965) `arXiv:2610.15965` |  | "无限计算"预训练的前沿取决于批大小、正则化等此前被固定的选择；系统地重新扩展 |
| 2026-10 | 🔥 [多模态数据混合扩展律](https://arxiv.org/abs/2610.14423) `arXiv:2610.14423` |  | 多模态预训练中数据混合比例的扩展律 |
| 2026-09 | 🔥 [循环 MoE 的扩展律](https://arxiv.org/abs/2609.40316) `arXiv:2609.40316` |  | 循环（增加计算深度）与 MoE（增加总容量）互补：拟合两者联合的扩展律 |
| 2026-09 | 🔥 [重访奖励优化的扩展律](https://arxiv.org/abs/2609.38526) `arXiv:2609.38526` |  | 奖励模型过优化的扩展律 |
| 2026-09 | 🔥 [数据混合何时改进扩展律](https://arxiv.org/abs/2609.38011) `arXiv:2609.38011` |  | 高维回归中的洞见 |
| 2026-09 | 🔥 [成对多模态扩展律](https://arxiv.org/abs/2609.36263) `arXiv:2609.36263` |  | 图文成对数据的扩展律 |
| 2026-09 | 🔥 [去掉视觉编码器的扩展律](https://arxiv.org/abs/2609.35457) `arXiv:2609.35457` | Tencent Hunyuan | 免编码器多模态模型的扩展律与距离 |
| 2026-09 | 🔥 [同族 on-policy 蒸馏的扩展性质](https://arxiv.org/abs/2609.32722) `arXiv:2609.32722` | Zhejiang Univ | RL 训练的教师向同族学生做在策略蒸馏时的扩展规律 |
| 2026-09 | 🔥 [能力流形与扩展律](https://arxiv.org/abs/2609.27588) `arXiv:2609.27588` |  | 把模型能力看作低维流形来理解扩展 |
| 2026-09 | 🔥 [涌现能力与模型合并](https://arxiv.org/abs/2609.24504) `arXiv:2609.24504` |  | 模型合并中的涌现能力 |
| 2026-08 | 🔥 [神经二次型](https://arxiv.org/abs/2608.13335) `arXiv:2608.13335` |  | 统一解释突然学习与扩展律的最小模型 |
| 2026-08 | 🔥 [Skaling](https://arxiv.org/abs/2608.07222) `arXiv:2608.07222` |  | 把 Chinchilla 的指数与 Kaplan 的耦合形式结合，修正数据稀缺与过度训练两端的损失预测偏差 |
| 2026-06 | 🔥 [grokking 中扩展律的随机几何理论](https://arxiv.org/abs/2606.30388) `arXiv:2606.30388` |  | grokking 时间与规模的扩展律 |
| 2026-06 | 🔥 [宽度与数据如何塑造泛化扩展律](https://arxiv.org/abs/2606.28242) `arXiv:2606.28242` |  | 二次神经网络中的泛化扩展律 |
| 2026-06 | 🔥 [以预训练数据配比设计扩展律](https://arxiv.org/abs/2606.19781) `arXiv:2606.19781` |  | 工程化地用数据组成预测扩展 |
| 2026-06 | 🔥 [解释数据混合扩展律](https://arxiv.org/abs/2606.08167) `arXiv:2606.08167` |  | 数据混合扩展律背后的机制 |
| 2026-06 | 🔥 [数据受限预训练](https://arxiv.org/abs/2606.06888) `arXiv:2606.06888` |  | 更强的正则化与相应的扩展律 |
| 2026-06 | 🔥 [元学习的可证明数据扩展律](https://arxiv.org/abs/2606.02008) `arXiv:2606.02008` |  | 通过复杂度最小化 |
| 2026-05 | 🔥 [为什么更大的模型学得更多](https://arxiv.org/abs/2605.29548) `arXiv:2605.29548` |  | 即使数据无限，大模型也能学会小模型学不会的那部分分布：容量、干扰与稀有任务的保留 |
| 2026-05 | 🔥 [LoRA 的参数记忆律](https://arxiv.org/abs/2605.30260) `arXiv:2605.30260` |  | LoRA 微调能记住多少新知识：给出随秩与数据量变化的参数记忆律 |
| 2026-05 | 🔥 [扩展律背后的模型形状](https://arxiv.org/abs/2605.27989) `arXiv:2605.27989` |  | 梯度叠加视角：深宽形状如何影响扩展律 |
| 2026-05 | 🔥 [统一神经扩展律](https://arxiv.org/abs/2605.26248) `arXiv:2605.26248` |  | 参数、数据、计算等多个维度同时变化时的统一扩展律形式 |
| 2026-05 | 🔥 [稀疏特征带来的非对称扩展律](https://arxiv.org/abs/2605.23591) `arXiv:2605.23591` |  | 稀疏特征使参数与数据的扩展指数不对称 |
| 2026-05 | 🔥 [Shannon 扩展律](https://arxiv.org/abs/2605.23901) `arXiv:2605.23901` |  | 把 LLM 看成噪声信道，用信道容量统一解释扩展律以及过度训练、量化导致的非单调退化 |
| 2026-05 | 🔥 [序贯特征恢复给出扩展律](https://arxiv.org/abs/2605.14567) `arXiv:2605.14567` |  | 可解的层级模型：按重要性依次恢复特征，自然给出幂律扩展 |
| 2026-05 | 🔥 [可预测扩展律的层级语言模型](https://arxiv.org/abs/2605.13687) `arXiv:2605.13687` |  | 层级生成的语言模型中扩展律可预测，推理有可证明的好处 |
| 2026-05 | 🔥 [数据受限下混合预训练的扩展律](https://arxiv.org/abs/2605.12715) `arXiv:2605.12715` |  | 多领域数据各自有限时的配比扩展律 |
| 2026-05 | 🔥 [实用扩展律](https://arxiv.org/abs/2605.09189) `arXiv:2605.09189` |  | 数据受限世界中把计算转化为性能 |
| 2026-05 | 🔥 [InfoLaw](https://arxiv.org/abs/2605.02364) `arXiv:2605.02364` |  | 质量加权混合的信息扩展律 |
| 2026-05 | 🔥 [数据受限训练的规范性扩展律](https://arxiv.org/abs/2605.01640) `arXiv:2605.01640` |  | Chinchilla 假设每个 token 都不重复；为数据有限、需多轮重复的预训练给出可操作的分配规则 |
| 2026-04 | ◆ [基础模型的极限理论](https://arxiv.org/abs/2604.24037) `arXiv:2604.24037` |  | 用数学极限理论理解涌现能力 |
| 2026-04 | ◆ [一次循环值多少](https://arxiv.org/abs/2604.21106) `arXiv:2604.21106` |  | 116 次预训练拟合循环模型的等深度扩展律：循环参数的"有效参数量"按 r^φ 计入 |
| 2026-04 | ◆ [Parcae](https://arxiv.org/abs/2604.12946) `arXiv:2604.12946` |  | 稳定训练循环语言模型并给出其扩展律 |
| 2026-04 | ◆ [测试时扩展让过度训练变得最优](https://arxiv.org/abs/2604.01411) `arXiv:2604.01411` |  | 把重复采样等推理开销计入后，计算最优的预训练应当比 Chinchilla 更"过度训练"小模型 |
| 2026-02 | ◆ [高维线性回归中精度的扩展律](https://arxiv.org/abs/2602.19241) `arXiv:2602.19241` |  | 可解模型中数值精度如何进入扩展律 |
| 2026-02 | ◆ [逆深度扩展源于多数层相似](https://arxiv.org/abs/2602.05970) `arXiv:2602.05970` |  | 多数层功能相近导致深度扩展的回报递减 |
| 2026-02 | ◆ [层级多指标模型的最优扩展律](https://arxiv.org/abs/2602.05846) `arXiv:2602.05846` |  | 层级多指标目标函数下可达的最优扩展律 |
| 2026-02 | ◆ [有效前沿](https://arxiv.org/abs/2602.02593) `arXiv:2602.02593` |  | 统一各类神经扩展律的"有效前沿"观点 |
| 2026-01 | ◆ [神经·神经扩展律](https://arxiv.org/abs/2601.19831) `arXiv:2601.19831` |  | 用神经网络预测下游任务的扩展曲线 |
| 2026-01 | ◆ [神经扩展律的起源](https://arxiv.org/abs/2601.10684) `arXiv:2601.10684` |  | 从随机图到自然语言：数据中的幂律结构如何转化为扩展律 |
| 2025-12 | ◆ [困惑度感知的数据扩展律](https://arxiv.org/abs/2512.21515) `arXiv:2512.21515` |  | 用困惑度地形预测继续预训练的效果 |
| 2025-12 | ◆ [可重整的谱壳动力学](https://arxiv.org/abs/2512.10427) `arXiv:2512.10427` |  | 用重整化式的谱壳动力学解释扩展律的起源 |
| 2025-12 | ◆ [扩展律的潜变量框架](https://arxiv.org/abs/2512.06553) `arXiv:2512.06553` |  | 用潜变量统一不同模型族的扩展律 |
| 2025-10 | 🆕 [ATLAS 迁移扩展律](https://arxiv.org/abs/2510.22037) `arXiv:2510.22037` | Google | 多语言预训练的自适应迁移扩展律与"多语言诅咒" |
| 2025-10 | 🆕 [交叉熵扩展律中扩展的是什么](https://arxiv.org/abs/2510.04067) `arXiv:2510.04067` |  | 拆解交叉熵损失中真正遵循幂律的部分 |
| 2025-09 | 🆕 [模型合并扩展律](https://arxiv.org/abs/2509.24244) `arXiv:2509.24244` | HKUST | 模型合并带来的收益随专家数与规模的经验扩展律 |
| 2025-06 | 🆕 [Farseer](https://arxiv.org/abs/2506.10972) `arXiv:2506.10972` | StepFun | 更精细的扩展律：在整个 (N, D) 平面上拟合损失面，外推误差显著降低 |
| 2025-05 | 🆕 [量化感知训练扩展律](https://arxiv.org/abs/2505.14302) `arXiv:2505.14302` |  | QAT 误差随模型规模、训练 token 与量化粒度的扩展律 |
| 2025-05 | 🆕 [并行扩展律（ParScale）](https://arxiv.org/abs/2505.10475) `arXiv:2505.10475` | Qwen/Zhejiang Univ | 用 P 路并行前向替代参数扩展，效果约等于参数扩展 O(log P) 倍 |
| 2025-05 | 🆕 [叠加带来稳健扩展](https://arxiv.org/abs/2505.10465) `arXiv:2505.10465` | MIT | 强叠加区间下损失随宽度呈 1/m 扩展，为神经扩展律提供机制解释 |
| 2025-03 | 🆕 [合成数据的扩展律](https://arxiv.org/abs/2503.19551) `arXiv:2503.19551` | Microsoft | 合成数据同样遵循可预测的扩展律，但收益更早饱和 |
| 2025-02 | 🆕 [1B 能否超过 405B](https://arxiv.org/abs/2502.06703) `arXiv:2502.06703` | Tsinghua/Shanghai AI Lab | 算力最优的测试时扩展策略依赖策略模型、PRM 与题目难度 |
| 2025-02 | 🆕 [ParetoQ](https://arxiv.org/abs/2502.02631) `arXiv:2502.02631` | Meta | 统一比较 1–4 bit 量化，发现 2/3 bit 附近存在学习方式的转折 |
| 2025-01 | 🆕 [Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975) `arXiv:2501.16975` | ByteDance Seed | 输入词表扩大（多粒度 n-gram 嵌入）带来对数线性的损失下降 |
| 2024-12 | 🆕 [源于数据分布的扩展律](https://arxiv.org/abs/2412.07942) `arXiv:2412.07942` |  | 从数据分布结构推导神经扩展律 |
| 2024-12 | 🆕 [Densing Law](https://arxiv.org/abs/2412.04315) `arXiv:2412.04315` | Tsinghua | LLM 的"能力密度"大约每 3.3 个月翻一番 |
| 2024-11 | 🆕 [低比特量化偏爱欠训练模型](https://arxiv.org/abs/2411.17691) `arXiv:2411.17691` |  | 训练越充分的模型越难做低比特量化，给出量化损失的扩展律 |
| 2024-08 | ★ [测试时计算的最优扩展](https://arxiv.org/abs/2408.03314) `arXiv:2408.03314` | UC Berkeley/Google | 按题目难度自适应分配测试时计算，效率可超过把模型放大 14 倍 |
| 2024-08 | [推理扩展律](https://arxiv.org/abs/2408.00724) `arXiv:2408.00724` | CMU/Tsinghua | 固定推理算力下，小模型 + 更强的推理搜索可以更划算 |
| 2024-07 | [Large Language Monkeys](https://arxiv.org/abs/2407.21787) `arXiv:2407.21787` | Stanford/Oxford | 重复采样的覆盖率随样本数呈对数线性增长：推理时采样也是一条扩展轴 |
| 2024-06 | [解释扩展律分歧](https://arxiv.org/abs/2406.19146) `arXiv:2406.19146` | Tel Aviv/UW | 最后一层计算、warmup 与超参调优解释了 Kaplan 与 Chinchilla 的差异 |
| 2024-06 | [调和 Kaplan 与 Chinchilla](https://arxiv.org/abs/2406.12907) `arXiv:2406.12907` | Cerebras/独立 | 两者差异主要来自是否计入嵌入参数与规模范围 |
| 2024-05 | [观察式扩展律](https://arxiv.org/abs/2405.10938) `arXiv:2405.10938` | Stanford/Toronto | 用公开模型族的低维能力空间预测下游能力与涌现，无需自己训练 |
| 2023-05 | [数据受限扩展律](https://arxiv.org/abs/2305.16264) `arXiv:2305.16264` | Hugging Face | 重复数据约 4 轮内几乎与新数据等价，之后收益迅速衰减 |
| 2023-04 | [涌现是幻象吗](https://arxiv.org/abs/2304.15004) `arXiv:2304.15004` | Stanford | "涌现"很大程度上来自非线性、不连续的评测指标，换成连续指标后能力平滑增长 |
| 2022-06 | [涌现能力](https://arxiv.org/abs/2206.07682) `arXiv:2206.07682` | Google/Stanford | 某些能力只在规模超过阈值后突然出现 |
| 2022-03 | ★ [Chinchilla](https://arxiv.org/abs/2203.15556) `arXiv:2203.15556` | DeepMind | 算力最优时参数与 token 应等比例扩展（约 20 token/参数），修正 Kaplan 结论 |
| 2021-02 | [迁移学习扩展律](https://arxiv.org/abs/2102.01293) `arXiv:2102.01293` | OpenAI | 预训练带来的"有效数据迁移量"随模型规模呈幂律 |
| 2020-10 | [自回归生成模型扩展律](https://arxiv.org/abs/2010.14701) `arXiv:2010.14701` | OpenAI | 图像、视频、数学、多模态上同样存在平滑的幂律扩展 |
| 2020-01 | ★ [Kaplan 扩展律](https://arxiv.org/abs/2001.08361) `arXiv:2001.08361` | OpenAI | 损失随参数量、数据量、算力呈平滑幂律，大模型更样本高效 |
| 2017-12 | [Deep Learning Scaling is Predictable](https://arxiv.org/abs/1712.00409) `arXiv:1712.00409` | Baidu | 较早在多个领域系统观察到泛化误差随数据与模型规模呈幂律下降 |

### Transformer 与注意力（71）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [无界输入上 Transformer 的训练动力学](https://arxiv.org/abs/2610.20337) `arXiv:2610.20337` |  | 注意力的学习可能滞后于输出层 |
| 2026-10 | 🔥 [注意力汇聚形成之前](https://arxiv.org/abs/2610.16745) `arXiv:2610.16745` |  | 隐状态几何预示并调节汇聚的形成 |
| 2026-09 | 🔥 [LLM 注意力的演化](https://arxiv.org/abs/2609.39661) `arXiv:2609.39661` |  | 显式记忆压缩、稀疏访问、循环状态、结构化动力学、异构混合等注意力变体综述 |
| 2026-09 | 🔥 [RoPE 已死，RoPE 万岁](https://arxiv.org/abs/2609.34556) `arXiv:2609.34556` |  | 面向可扩展、数据感知的位置编码 |
| 2026-09 | 🔥 [结合代数层](https://arxiv.org/abs/2609.32814) `arXiv:2609.32814` |  | 保留参数、换掉乘法：用稀疏的结合代数乘积替代 Transformer 投影中的矩阵乘法 |
| 2026-09 | 🔥 [MassAlloc Attention](https://arxiv.org/abs/2609.32712) `arXiv:2609.32712` | BAAI | 根据注意力质量分布自适应分配计算，跳过几乎为零的得分区域 |
| 2026-09 | 🔥 [DoSA](https://arxiv.org/abs/2610.13990) `arXiv:2610.13990` | dots studio | 一种面向长上下文的可训练稀疏注意力 |
| 2026-09 | 🔥 [CoWindow Attention](https://arxiv.org/abs/2609.32704) `arXiv:2609.32704` |  | 完整因果覆盖可以是多个注意力头的"集体性质"：各头只看部分窗口，合起来覆盖全部历史 |
| 2026-09 | 🔥 [新注意力机制真的修好了注意力汇聚吗](https://arxiv.org/abs/2609.08574) `arXiv:2609.08574` |  | 百万 token 上下文下的检验 |
| 2026-09 | 🔥 [Grouped Value Attention](https://arxiv.org/abs/2609.13285) `arXiv:2609.13285` |  | 按需重建 key、只缓存 value 的分组注意力，降低 KV 缓存与读带宽 |
| 2026-09 | 🔥 [模型能控制自己的注意力](https://arxiv.org/abs/2609.02737) `arXiv:2609.02737` | KAIST | 让模型学会决定读取哪些 KV，降低长上下文读取成本 |
| 2026-08 | 🔥 [用查询可证明地学习多头注意力](https://arxiv.org/abs/2608.03294) `arXiv:2608.03294` |  | 推广到多头注意力 |
| 2026-07 | 🔥 [相对位置泛化、绝对位置记忆](https://arxiv.org/abs/2607.18759) `arXiv:2607.18759` |  | 长度泛化的隐式偏置解释 |
| 2026-07 | 🔥 [数据如何塑造 RoPE 频率的使用](https://arxiv.org/abs/2607.07678) `arXiv:2607.07678` |  | 从位置尺度匹配到长度泛化 |
| 2026-07 | 🔥 [层级稀疏注意力](https://arxiv.org/abs/2607.02980) `arXiv:2607.02980` | Tencent Hunyuan | 把层级块选择做对，面向"无限"上下文建模 |
| 2026-06 | 🔥 [HydraHead](https://arxiv.org/abs/2606.20097) `arXiv:2606.20097` |  | 按注意力头的功能异质性做头级（而非层级）的混合注意力 |
| 2026-06 | 🔥 [Grouped Query Experts](https://arxiv.org/abs/2606.20945) `arXiv:2606.20945` |  | 把专家混合引入 GQA 自注意力 |
| 2026-06 | 🔥 [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392) `arXiv:2606.13392` | MiniMax | 基于 GQA 的块稀疏注意力，面向百万 token 的智能体与代码场景 |
| 2026-06 | 🔥 [注意力汇聚的统一视角](https://arxiv.org/abs/2606.08105) `arXiv:2606.08105` |  | 从机制到架构干预 |
| 2026-05 | 🔥 [位置头与符号头](https://arxiv.org/abs/2605.31558) `arXiv:2605.31558` |  | 学习动力学、RoPE 几何与长度泛化 |
| 2026-05 | 🔥 [全注意力的反击](https://arxiv.org/abs/2605.16928) `arXiv:2605.16928` | Alibaba RTP-LLM | 百步级训练即可把全注意力模型转换为稀疏注意力模型 |
| 2026-05 | 🔥 [扩散 Transformer 中的注意力汇聚](https://arxiv.org/abs/2605.09313) `arXiv:2605.09313` |  | 因果分析 |
| 2026-05 | 🔥 [注意力汇聚的结构性起源](https://arxiv.org/abs/2605.06611) `arXiv:2605.06611` |  | 方差差异、超级神经元与维度 |
| 2026-04 | ◆ [GPT-2 中注意力汇聚的机制](https://arxiv.org/abs/2604.14722) `arXiv:2604.14722` |  | 一个回路及其更广泛的含义 |
| 2026-04 | ◆ [注意力汇聚点综述](https://arxiv.org/abs/2604.10098) `arXiv:2604.10098` | LongCat | 注意力汇聚点的利用、解释与缓解 |
| 2026-04 | ◆ [Universal YOCO](https://arxiv.org/abs/2604.01220) `arXiv:2604.01220` | Microsoft Research | "只缓存一次"架构推广到深度扩展，降低推理时扩展的显存成本 |
| 2026-03 | ◆ [残差流即一切](https://arxiv.org/abs/2603.19664) `arXiv:2603.19664` |  | KV 缓存在推理中是冗余的：可由残差流重算 |
| 2026-03 | ◆ [注意力汇聚点引发梯度汇聚点](https://arxiv.org/abs/2603.17771) `arXiv:2603.17771` |  | 巨大激活在训练中充当梯度调节器 |
| 2026-03 | ◆ [softmax Transformer 中注意力汇聚可证明是必要的](https://arxiv.org/abs/2603.11487) `arXiv:2603.11487` |  | 需要"什么都不做"的触发条件任务中，softmax 注意力必须形成汇聚 |
| 2026-03 | ◆ [多头低秩注意力](https://arxiv.org/abs/2603.02188) `arXiv:2603.02188` |  | 面向解码时 HBM→SRAM 搬运瓶颈的低秩 KV 注意力（与 MLA 同一方向） |
| 2026-03 | ◆ [softmax 自注意力的训练动力学](https://arxiv.org/abs/2603.01514) `arXiv:2603.01514` |  | 通过预条件实现快速全局收敛 |
| 2026-02 | ◆ [WildCat](https://arxiv.org/abs/2602.10056) `arXiv:2602.10056` |  | 理论与实践中的近线性注意力 |
| 2026-02 | ◆ [位置 0 有何特殊](https://arxiv.org/abs/2603.06591) `arXiv:2603.06591` |  | 首位注意力汇聚的机制研究 |
| 2026-02 | ◆ [FASA](https://arxiv.org/abs/2602.03152) `arXiv:2602.03152` |  | 按 RoPE 频率分量识别重要 token 的频率感知稀疏注意力 |
| 2026-02 | ◆ [注意力汇聚在注意力层中造出原生 MoE](https://arxiv.org/abs/2602.01203) `arXiv:2602.01203` |  | 汇聚使注意力头像专家一样被门控 |
| 2026-01 | ◆ [用查询可证明地学习注意力](https://arxiv.org/abs/2601.16873) `arXiv:2601.16873` |  | 通过查询访问可证明地学习单头注意力 |
| 2025-12 | ◆ [次级注意力汇聚点](https://arxiv.org/abs/2512.22213) `arXiv:2512.22213` |  | 除首 token 外还存在次级汇聚点，分析其出现与行为 |
| 2025-12 | ◆ [Top-k 稀疏注意力的数学理论](https://arxiv.org/abs/2512.07647) `arXiv:2512.07647` |  | 用全变差距离分析 top-k 稀疏注意力 |
| 2025-12 | ◆ [RoPE 的虚部扩展](https://arxiv.org/abs/2512.07525) `arXiv:2512.07525` | OpenMOSS | 利用旋转位置编码的虚部信息改善长上下文 |
| 2025-12 | ◆ [GRAPE 位置编码](https://arxiv.org/abs/2512.07805) `arXiv:2512.07805` |  | 用群作用统一位置编码：SO(d) 中的乘性旋转（推广 RoPE）与幺幂作用给出的加性偏置（推广 ALiBi） |
| 2025-11 | ◆ [DoPE](https://arxiv.org/abs/2511.09146) `arXiv:2511.09146` |  | 对 RoPE 的频率分量"去噪"以改善长度外推 |
| 2025-10 | 🆕 [Knocking-Heads Attention](https://arxiv.org/abs/2510.23052) `arXiv:2510.23052` |  | 让注意力头之间在注意前互相交流特征 |
| 2025-10 | 🆕 [语言模型是单射的](https://arxiv.org/abs/2510.15511) `arXiv:2510.15511` | Gladia/Sapienza | 证明 Transformer 从输入序列到隐表示几乎处处是单射，可从隐状态精确反推出输入 |
| 2025-07 | 🆕 [把相机当作相对位置编码](https://arxiv.org/abs/2507.10496) `arXiv:2507.10496` |  | 把相机几何写入注意力的相对位置编码，用于多视图 Transformer |
| 2025-06 | 🆕 [Log-Linear Attention](https://arxiv.org/abs/2506.04761) `arXiv:2506.04761` | MIT | 用按对数增长的多尺度隐状态（Fenwick 树划分）介于线性注意力与 softmax 之间 |
| 2025-06 | 🆕 [Rectified Sparse Attention](https://arxiv.org/abs/2506.04108) `arXiv:2506.04108` | Microsoft | 周期性用稠密计算校正稀疏解码的 KV 缓存漂移 |
| 2025-05 | ★ 🆕 [Gated Attention](https://arxiv.org/abs/2505.06708) `arXiv:2505.06708` | Qwen | 在 SDPA 输出后加逐头 sigmoid 门：引入非线性与稀疏性，消除注意力汇聚点，训练更稳（NeurIPS 2025 最佳论文） |
| 2025-05 | 🆕 [稀疏注意力专家混合（MoSA）](https://arxiv.org/abs/2505.00315) `arXiv:2505.00315` |  | 按专家选择路由为每个头挑选 token，内容相关的可学习稀疏注意力 |
| 2025-04 | 🆕 [Softpick](https://arxiv.org/abs/2504.20966) `arXiv:2504.20966` |  | 整流 softmax：没有注意力汇聚点，也没有巨大激活 |
| 2025-04 | 🆕 [为什么关注首 token](https://arxiv.org/abs/2504.02732) `arXiv:2504.02732` | Oxford/Google DeepMind | 注意力汇聚点是防止表征过度混合（over-squashing）的机制，随上下文长度和深度增强 |
| 2025-03 | 🆕 [Forgetting Transformer](https://arxiv.org/abs/2503.02130) `arXiv:2503.02130` | Mila/Montréal | 把遗忘门引入 softmax 注意力（数据相关的衰减偏置），长度外推更好 |
| 2025-02 | 🆕 [MoBA](https://arxiv.org/abs/2502.13189) `arXiv:2502.13189` | Moonshot AI | 把 MoE 的路由思想用于注意力：每个查询只关注被选中的 KV 块 |
| 2025-02 | ★ 🆕 [NSA](https://arxiv.org/abs/2502.11089) `arXiv:2502.11089` | DeepSeek/PKU | 原生稀疏注意力：压缩、选择、滑窗三分支，硬件对齐且可端到端训练 |
| 2025-02 | 🆕 [TransMLA](https://arxiv.org/abs/2502.07864) `arXiv:2502.07864` | PKU/Xiaomi | 证明同等 KV 缓存下 MLA 表达力强于 GQA，并把 GQA 模型转换为 MLA |
| 2025-02 | 🆕 [VideoRoPE](https://arxiv.org/abs/2502.05173) `arXiv:2502.05173` | Fudan/Shanghai AI Lab | 视频旋转位置编码的设计原则与改进 |
| 2025-02 | 🆕 [STRING](https://arxiv.org/abs/2502.02562) `arXiv:2502.02562` | Google DeepMind | 把 RoPE 推广为更一般的可分离平移不变位置编码，用于 2D/3D 与机器人 |
| 2025-02 | 🆕 [注意力汇聚点：捕获-标记-释放](https://arxiv.org/abs/2502.00919) `arXiv:2502.00919` |  | 注意力汇聚点通过"捕获、标记、释放"机制在嵌入间传递信息 |
| 2025-01 | 🆕 [TPA](https://arxiv.org/abs/2501.06425) `arXiv:2501.06425` | Tsinghua/UCLA | 张量积注意力：Q/K/V 做上下文相关的低秩分解，KV 缓存更小 |
| 2024-12 | 🆕 [FoPE](https://arxiv.org/abs/2412.17739) `arXiv:2412.17739` | Tsinghua | 傅里叶位置编码：增强注意力的周期延拓以改善长度泛化 |
| 2024-10 | [Differential Transformer](https://arxiv.org/abs/2410.05258) `arXiv:2410.05258` | Microsoft/Tsinghua | 两组 softmax 注意力相减以抵消噪声，减少幻觉与激活离群值 |
| 2024-07 | [FlashAttention-3](https://arxiv.org/abs/2407.08608) `arXiv:2407.08608` | Together/NVIDIA | 利用 Hopper 异步与 FP8，H100 上达到 75% 利用率 |
| 2024-05 | ★ [DeepSeek-V2（MLA）](https://arxiv.org/abs/2405.04434) `arXiv:2405.04434` | DeepSeek | 多头潜注意力把 KV 压缩成低秩潜向量，KV 缓存减少 93%；配合细粒度 MoE |
| 2023-07 | [FlashAttention-2](https://arxiv.org/abs/2307.08691) `arXiv:2307.08691` | Princeton/Together | 更好的并行与工作划分，接近 GEMM 的利用率 |
| 2023-05 | [GQA](https://arxiv.org/abs/2305.13245) `arXiv:2305.13245` | Google | 介于 MHA 与 MQA 之间的分组查询注意力，Llama 2/3 等采用 |
| 2022-05 | ★ [FlashAttention](https://arxiv.org/abs/2205.14135) `arXiv:2205.14135` | Stanford | IO 感知的分块精确注意力：不物化 N×N 矩阵，显存线性、速度数倍 |
| 2021-08 | [ALiBi](https://arxiv.org/abs/2108.12409) `arXiv:2108.12409` | UW/Meta | 对注意力分数加线性距离偏置，训练短、推理长 |
| 2021-04 | ★ [RoPE](https://arxiv.org/abs/2104.09864) `arXiv:2104.09864` | Zhuiyi | 旋转位置编码：用旋转矩阵把相对位置写进注意力内积，几乎所有 LLM 采用 |
| 2021-03 | [Swin Transformer](https://arxiv.org/abs/2103.14030) `arXiv:2103.14030` | Microsoft Research Asia | 移位窗口的层级式视觉 Transformer，兼顾局部性与效率 |
| 2020-10 | ★ [ViT](https://arxiv.org/abs/2010.11929) `arXiv:2010.11929` | Google | 把图像切成 16×16 patch 当 token，纯 Transformer 在大数据上超过 CNN |
| 2019-11 | [MQA](https://arxiv.org/abs/1911.02150) `arXiv:1911.02150` | Google | 所有注意力头共享一组 K/V，大幅缩小解码时的 KV 缓存 |
| 2018-10 | ★ [BERT](https://arxiv.org/abs/1810.04805) `arXiv:1810.04805` | Google | 双向 Transformer 编码器 + 掩码语言建模预训练，"预训练-微调"范式确立 |

### 线性注意力/SSM/循环（144）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [如何循环你的 Transformer](https://arxiv.org/abs/2610.40325) `arXiv:2610.40325` |  | 循环的扩展、专门化与并行推理 |
| 2026-10 | 🔥 [Dual DeltaNet](https://arxiv.org/abs/2610.38578) `arXiv:2610.38578` |  | 值侧的记忆重组 |
| 2026-10 | 🔥 [Select-TTT](https://arxiv.org/abs/2610.36794) `arXiv:2610.36794` |  | 查询条件化的选择性快权重更新 |
| 2026-10 | 🔥 [用测试时回归重新看序列压缩](https://arxiv.org/abs/2610.31574) `arXiv:2610.31574` |  | 把序列压缩视为测试时回归问题（以输入 token 定义回归目标），优于池化与可学习投影 |
| 2026-10 | 🔥 [PAR](https://arxiv.org/abs/2610.27628) `arXiv:2610.27628` | Ant Group | 让循环迭代与任务进度对齐，循环模型才能在状态追踪上长度泛化 |
| 2026-10 | 🔥 [截断还是不截断](https://arxiv.org/abs/2610.26030) `arXiv:2610.26030` |  | 循环 Transformer 用截断 BPTT 训练会产生有偏梯度：其扩展性质研究 |
| 2026-10 | 🔥 [Kronecker Delta 规则](https://arxiv.org/abs/2610.26834) `arXiv:2610.26834` |  | 面向长上下文的 Kronecker 结构 delta 规则线性注意力 |
| 2026-10 | 🔥 [循环模型做对了（一）](https://arxiv.org/abs/2610.23036) `arXiv:2610.23036` |  | 在相同参数、逻辑深度和 token 预算下，分离循环位置、输入注入、循环状态组织三条设计轴 |
| 2026-10 | 🔥 [测试时预测](https://arxiv.org/abs/2610.19552) `arXiv:2610.19552` |  | 循环深度使上下文中的动力学预测成为可能 |
| 2026-10 | 🔥 [LatentLoop](https://arxiv.org/abs/2610.19975) `arXiv:2610.19975` |  | 非对称计算分配的循环 MoE 扩展 |
| 2026-10 | 🔥 [LEV 循环决策模型](https://arxiv.org/abs/2610.20951) `arXiv:2610.20951` |  | 用循环深度为单次前向的决策模型提供"快而深"的思考 |
| 2026-10 | 🔥 [历史会重演](https://arxiv.org/abs/2610.15715) `arXiv:2610.15715` |  | 挑战"状态容量决定长上下文能力"的看法：让线性注意力重放历史即可释放长上下文能力 |
| 2026-10 | 🔥 [用循环 Transformer 学习深度外推](https://arxiv.org/abs/2610.14456) `arXiv:2610.14456` |  | 训练浅、推理深 |
| 2026-10 | 🔥 [少跑几次循环就能去除拒绝](https://arxiv.org/abs/2610.15218) `arXiv:2610.15218` |  | 循环深度模型的安全行为依赖循环次数 |
| 2026-10 | 🔥 [上万次测试时迭代](https://arxiv.org/abs/2610.11570) `arXiv:2610.11570` |  | 循环次数增多反而会让噪声更新覆盖正确中间结论；用循环原生的注意力残差扩展到上万次迭代 |
| 2026-10 | 🔥 [reViT](https://arxiv.org/abs/2610.12448) `arXiv:2610.12448` |  | 一个块、多个深度：循环 ViT 用凸组合的深度专家恢复逐层变换 |
| 2026-10 | 🔥 [循环 Looped Transformer](https://arxiv.org/abs/2610.07591) `arXiv:2610.07591` |  | 在循环 Transformer 中引入跨迭代的递归状态 |
| 2026-10 | 🔥 [循环模型做对了（二）](https://arxiv.org/abs/2610.06833) `arXiv:2610.06833` |  | 循环状态越接近不动点，路径越不重要：由此支持截断反传、末端 KV 共享等 |
| 2026-10 | 🔥 [大规模循环 MoE 配方](https://arxiv.org/abs/2610.01153) `arXiv:2610.01153` |  | FLOPs 匹配下循环超过两次的大 MoE 仍有收益的可扩展配方 |
| 2026-09 | 🔥 [循环语言模型的训练配方](https://arxiv.org/abs/2610.00673) `arXiv:2610.00673` |  | 实用训练配方 |
| 2026-09 | 🔥 [循环扩散 Transformer](https://arxiv.org/abs/2609.40305) `arXiv:2609.40305` |  | 把循环深度用于扩散 Transformer |
| 2026-09 | 🔥 [循环推理的步长调度](https://arxiv.org/abs/2609.36653) `arXiv:2609.36653` |  | 循环 Transformer 每次更新的尺度固定；按进展自适应调度更新步长 |
| 2026-09 | 🔥 [循环何时有效](https://arxiv.org/abs/2609.36636) `arXiv:2609.36636` |  | 受控实验：额外循环何时带来收益、哪些架构选择决定其效果 |
| 2026-09 | 🔥 [循环 Transformer 作为优化器](https://arxiv.org/abs/2609.37379) `arXiv:2609.37379` |  | 循环 Transformer 可在前向中实现迭代优化算法 |
| 2026-09 | 🔥 [三元线性注意力](https://arxiv.org/abs/2609.36529) `arXiv:2609.36529` |  | 把循环状态从矩阵扩展到三维张量，增大固定大小记忆的容量 |
| 2026-09 | 🔥 [STEPQuant](https://arxiv.org/abs/2609.38169) `arXiv:2609.38169` | Zhejiang Univ | delta 规则递归状态的量化误差何时何处最关键 |
| 2026-09 | 🔥 [LoopVL](https://arxiv.org/abs/2609.38426) `arXiv:2609.38426` | Renmin Univ | 把循环 Transformer 扩展到视觉语言模型 |
| 2026-09 | 🔥 [自适应循环 Transformer 的测试时扩展](https://arxiv.org/abs/2609.35748) `arXiv:2609.35748` |  | 输出变长时，循环能否改善测试时扩展：自适应循环深度 |
| 2026-09 | 🔥 [线性注意力如何记忆](https://arxiv.org/abs/2609.33093) `arXiv:2609.33093` |  | 分析线性注意力递归状态的记忆机制 |
| 2026-09 | 🔥 [循环 Transformer 长度泛化的机制与极限](https://arxiv.org/abs/2609.33144) `arXiv:2609.33144` |  | 超越训练视野的长度泛化 |
| 2026-09 | 🔥 [残差流读取、循环状态记忆](https://arxiv.org/abs/2609.32102) `arXiv:2609.32102` |  | Mamba 模型中的全局工作空间 |
| 2026-09 | 🔥 [并行扫描下与长度无关的状态追踪](https://arxiv.org/abs/2609.32447) `arXiv:2609.32447` |  | 可并行扫描的循环结构实现长度无关的状态追踪 |
| 2026-09 | 🔥 [追踪状态还是追踪陪集](https://arxiv.org/abs/2609.29951) `arXiv:2609.29951` |  | 学到的状态追踪的代数解释 |
| 2026-09 | 🔥 [Mamba-3 的非交换状态追踪](https://arxiv.org/abs/2609.28273) `arXiv:2609.28273` |  | 输入相关的低秩更新实现非交换状态追踪 |
| 2026-09 | 🔥 [Complex KDA](https://arxiv.org/abs/2609.24797) `arXiv:2609.24797` | OpenEuroLLM | 分析并用复数增强 Kimi Delta Attention 的表达力 |
| 2026-09 | 🔥 [循环 Transformer 中以时间换深度](https://arxiv.org/abs/2609.21605) `arXiv:2609.21605` |  | 循环 Transformer 的深度—时间权衡 |
| 2026-09 | 🔥 [SpectralShift](https://arxiv.org/abs/2609.14320) `arXiv:2609.14320` | RUC | 谱重参数化扩展 Gated DeltaNet 的上下文窗口 |
| 2026-09 | 🔥 [RunningTensor](https://arxiv.org/abs/2609.12814) `arXiv:2609.12814` |  | 把线性注意力推广到高阶循环状态 |
| 2026-09 | 🔥 [全局分歧、局部趋同](https://arxiv.org/abs/2609.08692) `arXiv:2609.08692` |  | SSM 与 Transformer 的表征几何比较 |
| 2026-09 | 🔥 [Kalman Delta Networks](https://arxiv.org/abs/2609.07816) `arXiv:2609.07816` |  | 用卡尔曼滤波给 delta 规则的写入强度加上不确定性估计：不确定的联想才大幅改写 |
| 2026-09 | 🔥 [SMELT](https://arxiv.org/abs/2609.01343) `arXiv:2609.01343` | ByteDance Seed | 算力匹配下循环 MoE Transformer 的扩展律 |
| 2026-08 | 🔥 [通用 Transformer 的完美长度泛化](https://arxiv.org/abs/2608.31067) `arXiv:2608.31067` |  | 小型通用 Transformer 在电路计算上完美长度泛化 |
| 2026-08 | 🔥 [Qwen3.8-Next 架构设计](https://arxiv.org/abs/2608.30320) `arXiv:2608.30320` | Qwen | 125B-A6B 混合稀疏 MoE + 51B 离线 n-gram 嵌入表，以 1/3 激活参数接近 397B-A17B 前代 |
| 2026-08 | 🔥 [滑窗胜过线性注意力](https://arxiv.org/abs/2608.28444) `arXiv:2608.28444` |  | 在混合架构中，滑动窗口注意力可以优于线性注意力层 |
| 2026-08 | 🔥 [循环模型改进组合式工具调用](https://arxiv.org/abs/2608.18171) `arXiv:2608.18171` |  | 循环语言模型在需要多步依赖与中间状态的组合任务上泛化更好 |
| 2026-08 | 🔥 [门控循环 Transformer](https://arxiv.org/abs/2608.15062) `arXiv:2608.15062` |  | 用循环调制在共享深度中恢复逐层功能分化（从感知输入到抽象精炼） |
| 2026-08 | 🔥 [混合线性注意力中的巨大激活](https://arxiv.org/abs/2608.12149) `arXiv:2608.12149` |  | 层交错的混合线性注意力模型中，巨大激活在注意力层前出现尖峰、其间形成平台 |
| 2026-08 | 🔥 [BDH-CQ](https://arxiv.org/abs/2608.09888) `arXiv:2608.09888` | Pathway | 推理时输入持续更新循环记忆，在高维潜空间迭代求解，不用语言化中间推理（ARC-AGI 评测） |
| 2026-08 | 🔥 [复数状态空间模型的确定性状态追踪](https://arxiv.org/abs/2608.03425) `arXiv:2608.03425` |  | 复值状态传播即可满足状态追踪 |
| 2026-07 | 🔥 [向预训练模型加装循环深度](https://arxiv.org/abs/2608.11233) `arXiv:2608.11233` |  | 安装、外推与迁移 |
| 2026-07 | 🔥 [加性输入通路是状态追踪的寄生吸引子](https://arxiv.org/abs/2609.18966) `arXiv:2609.18966` |  | 循环模型中加性输入通路阻碍学到真正的自动机 |
| 2026-07 | 🔥 [DeepLoop](https://arxiv.org/abs/2607.13491) `arXiv:2607.13491` |  | 循环 Transformer 中同一参数被多次使用，改变了残差缩放问题：给出循环下的深度扩展参数化 |
| 2026-07 | 🔥 [线性注意力架构比较](https://arxiv.org/abs/2607.07953) `arXiv:2607.07953` |  | softmax 注意力与 DeltaNet、Gated DeltaNet、KDA 等四种循环线性注意力的机制与取舍 |
| 2026-06 | 🔥 [循环世界模型](https://arxiv.org/abs/2606.18208) `arXiv:2606.18208` |  | 用循环计算兼顾长时程模拟的深度与推理效率 |
| 2026-06 | 🔥 [LoopCoder-v2](https://arxiv.org/abs/2606.18023) `arXiv:2606.18023` |  | "只循环一次"降低循环带来的串行延迟，提升代码推理 |
| 2026-06 | 🔥 [重新审视混合架构中的高效注意力](https://arxiv.org/abs/2606.15378) `arXiv:2606.15378` |  | 混合架构里滑窗/循环模块到底起什么作用：系统消融 |
| 2026-06 | 🔥 [动态线性注意力](https://arxiv.org/abs/2606.10650) `arXiv:2606.10650` |  | 动态组织线性注意力的记忆以提升长上下文下的表示容量 |
| 2026-06 | 🔥 [非阿贝尔状态追踪的证伪器](https://arxiv.org/abs/2606.07254) `arXiv:2606.07254` |  | 用留出转移对检验是否真学会长程非阿贝尔状态追踪 |
| 2026-06 | 🔥 [在大规模 Gated DeltaNet 中解锁特征学习](https://arxiv.org/abs/2606.04048) `arXiv:2606.04048` |  | Gated DeltaNet 的参数化让特征学习在扩大时保持 |
| 2026-05 | 🔥 [带 LayerNorm 的循环 Transformer 可证明学会幂法](https://arxiv.org/abs/2606.00605) `arXiv:2606.00605` |  | 循环与归一化共同实现迭代算法 |
| 2026-05 | 🔥 [Parallax](https://arxiv.org/abs/2605.29157) `arXiv:2605.29157` |  | 源自非参数统计"测试时回归"框架的参数化局部线性注意力 |
| 2026-05 | 🔥 [对数深度循环单元的长度泛化](https://arxiv.org/abs/2605.26035) `arXiv:2605.26035` |  | 对数深度的循环结构实现长度泛化 |
| 2026-05 | 🔥 [免训练循环](https://arxiv.org/abs/2605.23872) `arXiv:2605.23872` |  | 无需训练直接循环预训练模型的部分层 |
| 2026-05 | 🔥 [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) `arXiv:2605.22791` | NVIDIA | 把 delta 规则中的"擦除"与"写入"解耦，更精确地编辑压缩记忆而不打乱已有联想 |
| 2026-05 | 🔥 [HRM-Text](https://arxiv.org/abs/2605.20613) `arXiv:2605.20613` | Sapient | 把层级推理模型用于文本预训练 |
| 2026-05 | 🔥 [生成式递归推理](https://arxiv.org/abs/2605.19376) `arXiv:2605.19376` |  | 把 HRM/TRM 式递归推理改为生成式（随机）过程 |
| 2026-05 | 🔥 [混合 DeltaNet 解码器的更短草稿](https://arxiv.org/abs/2605.16640) `arXiv:2605.16640` |  | 证明 DeltaNet-注意力混合解码器能用更短的思维草稿解题 |
| 2026-05 | 🔥 [精确线性注意力](https://arxiv.org/abs/2605.18848) `arXiv:2605.18848` |  | 精确计算的线性注意力形式 |
| 2026-05 | 🔥 [OSDN](https://arxiv.org/abs/2605.13473) `arXiv:2605.13473` |  | 带可证明在线预条件的 delta 规则 |
| 2026-05 | 🔥 [完全循环 Transformer](https://arxiv.org/abs/2605.18797) `arXiv:2605.18797` |  | 整体循环所有层以稳定循环训练 |
| 2026-05 | 🔥 [变分线性注意力](https://arxiv.org/abs/2605.11196) `arXiv:2605.11196` |  | 稳定的联想记忆用于长上下文 |
| 2026-05 | 🔥 [LoopUS](https://arxiv.org/abs/2605.11011) `arXiv:2605.11011` |  | 把预训练 LLM 改造成循环潜空间精炼模型，无需从零训练 |
| 2026-05 | 🔥 [Kaczmarz 线性注意力](https://arxiv.org/abs/2605.08587) `arXiv:2605.08587` |  | 从 Kaczmarz 迭代求解视角设计线性注意力更新 |
| 2026-05 | 🔥 [显存高效的循环 Transformer](https://arxiv.org/abs/2605.07721) `arXiv:2605.07721` |  | 解耦循环语言模型的计算与显存 |
| 2026-05 | 🔥 [以误差控制动力学重新看状态追踪](https://arxiv.org/abs/2605.07755) `arXiv:2605.07755` |  | 循环模型状态追踪不仅关乎表达力，还关乎误差能否被控制 |
| 2026-05 | 🔥 [MDN](https://arxiv.org/abs/2605.05838) `arXiv:2605.05838` |  | 把 delta 线性注意力视为在线 SGD，加入可并行的逐步动量 |
| 2026-05 | 🔥 [用 TTT 线性化 ViT](https://arxiv.org/abs/2605.02772) `arXiv:2605.02772` |  | 把预训练 ViT 的 softmax 注意力转换为测试时训练层，免去从零训练线性模型 |
| 2026-04 | ◆ [通用 Transformer 需要记忆](https://arxiv.org/abs/2604.21999) `arXiv:2604.21999` |  | 自适应递归推理中的深度—状态权衡 |
| 2026-04 | ◆ [Sessa](https://arxiv.org/abs/2604.18580) `arXiv:2604.18580` |  | 选择性状态空间注意力：把注意力的全局读取与 SSM 的循环状态传播结合 |
| 2026-04 | ◆ [循环 Transformer 的稳定性与泛化](https://arxiv.org/abs/2604.15259) `arXiv:2604.15259` |  | 循环结构的稳定性与外推泛化分析 |
| 2026-04 | ◆ [循环、思考与泛化](https://arxiv.org/abs/2604.07822) `arXiv:2604.07822` |  | 循环深度 Transformer 的隐式推理与泛化 |
| 2026-03 | ◆ [混合联想记忆](https://arxiv.org/abs/2603.22325) `arXiv:2603.22325` |  | 组合不同联想记忆机制 |
| 2026-03 | ◆ [LoopRPT](https://arxiv.org/abs/2603.19714) `arXiv:2603.19714` |  | 为循环语言模型设计的强化预训练：奖励作用于潜在迭代而非输出 token |
| 2026-03 | ◆ [SCORE](https://arxiv.org/abs/2603.10544) `arXiv:2603.10544` |  | 用压缩映射式循环深度替代层堆叠 |
| 2026-03 | ◆ [对角 SSM 的表达极限](https://arxiv.org/abs/2603.01959) `arXiv:2603.01959` |  | 对角状态空间模型在状态追踪上的表达力上限 |
| 2026-02 | ◆ [KV 绑定的 TTT 其实是线性注意力](https://arxiv.org/abs/2602.21204) `arXiv:2602.21204` |  | 以"记忆键值映射"解释的测试时训练层，有多种现象与记忆解释矛盾；它本质上是一种线性注意力 |
| 2026-02 | ◆ [序列模型的"归纳偏置"](https://arxiv.org/abs/2602.18333) `arXiv:2602.18333` |  | 比较 Transformer 与循环模型在状态追踪类任务上的样本效率差异 |
| 2026-02 | ◆ [从代码学习状态追踪](https://arxiv.org/abs/2602.14814) `arXiv:2602.14814` |  | 线性 RNN 在代码数据上学习状态追踪 |
| 2026-02 | ◆ [Mamba 训练动力学的理论分析](https://arxiv.org/abs/2602.12499) `arXiv:2602.12499` |  | Mamba 过滤相关特征以实现泛化 |
| 2026-02 | ◆ [Hadamard 线性注意力](https://arxiv.org/abs/2602.12128) `arXiv:2602.12128` |  | Hadamard 积形式的线性注意力 |
| 2026-02 | ◆ [卡尔曼线性注意力](https://arxiv.org/abs/2602.10743) `arXiv:2602.10743` |  | 并行贝叶斯滤波：高效语言建模与状态追踪 |
| 2026-02 | ◆ [LoopFormer](https://arxiv.org/abs/2602.11451) `arXiv:2602.11451` | Vector Institute | 弹性深度的循环 Transformer，用捷径调制做潜空间推理 |
| 2026-02 | ◆ [线性注意力的状态约简](https://arxiv.org/abs/2602.04852) `arXiv:2602.04852` |  | 压缩线性注意力的递归状态维度 |
| 2026-02 | ◆ [线性注意力 LLM 的状态秩动力学](https://arxiv.org/abs/2602.02195) `arXiv:2602.02195` |  | 线性注意力状态矩阵的秩如何演化 |
| 2026-02 | ◆ [混合线性—全注意力的表达力层级](https://arxiv.org/abs/2602.01763) `arXiv:2602.01763` |  | 可证明的表达力层级：混合比例与能力 |
| 2026-02 | ◆ [Softmax 线性注意力](https://arxiv.org/abs/2602.01744) `arXiv:2602.01744` |  | 在线性注意力中恢复 softmax 的全局竞争 |
| 2026-02 | ◆ [LoopViT](https://arxiv.org/abs/2602.02156) `arXiv:2602.02156` |  | 用循环 Transformer 扩展视觉 ARC：计算深度不再受参数量限制 |
| 2026-01 | ◆ [混合线性注意力做对了](https://arxiv.org/abs/2601.22156) `arXiv:2601.22156` |  | 从预训练 Transformer 高效蒸馏出混合线性注意力模型，并研究有效的混合架构 |
| 2026-01 | ◆ [状态空间模型的长程依赖分析](https://arxiv.org/abs/2601.13048) `arXiv:2601.13048` |  | SSM 实际能理解多长的依赖 |
| 2026-01 | ◆ [MHLA](https://arxiv.org/abs/2601.07832) `arXiv:2601.07832` |  | 在 token 维度做多头划分，恢复线性注意力被压缩掉的表达能力，几乎不增加开销 |
| 2026-01 | ★ ◆ [Engram 条件记忆](https://arxiv.org/abs/2601.07372) `arXiv:2601.07372` | DeepSeek | 把 O(1) 查表的 n-gram 记忆作为与 MoE 互补的新稀疏维度；MoE 与记忆之间的参数分配呈 U 形扩展律，27B 记忆在等参数等 FLOPs 下胜过纯 MoE，推理与代码提升甚至大于知识类任务 |
| 2025-12 | ◆ [端到端测试时训练（TTT-E2E）](https://arxiv.org/abs/2512.23675) `arXiv:2512.23675` | Stanford·NVIDIA | 只用滑窗注意力的标准 Transformer，在测试时对上下文继续做下一词预测、把上下文压进权重 |
| 2025-12 | ◆ [Nemotron 3](https://arxiv.org/abs/2512.20856) `arXiv:2512.20856` | NVIDIA | 开放的混合 Mamba-Transformer MoE 模型族 |
| 2025-12 | ◆ [精确流线性注意力 EFLA](https://arxiv.org/abs/2512.12602) `arXiv:2512.12602` |  | 把在线学习更新写成连续时间动力学并求精确解，得到数值稳定、可并行的广义 delta 规则 |
| 2025-11 | ◆ [LFM2](https://arxiv.org/abs/2511.23404) `arXiv:2511.23404` | Liquid AI | 面向端侧的短卷积 + 注意力混合模型 |
| 2025-11 | ◆ [Nemotron-Flash](https://arxiv.org/abs/2511.18890) `arXiv:2511.18890` | NVIDIA | 以真实延迟为目标搜索混合小模型架构 |
| 2025-11 | ◆ [TNT](https://arxiv.org/abs/2511.07343) `arXiv:2511.07343` |  | 改进测试时记忆的分块训练 |
| 2025-10 | 🆕 [高阶线性注意力](https://arxiv.org/abs/2510.27258) `arXiv:2510.27258` |  | 在保持线性复杂度的同时引入高阶交互 |
| 2025-10 | ★ 🆕 [Kimi Linear](https://arxiv.org/abs/2510.26692) `arXiv:2510.26692` | Moonshot AI | Kimi Delta Attention（细粒度门控）与 MLA 3:1 混合，在同等训练下首次全面超过全注意力 |
| 2025-10 | 🆕 [Ouro（循环 LM）](https://arxiv.org/abs/2510.25741) `arXiv:2510.25741` | ByteDance Seed | 预训练阶段就把推理做进循环的潜空间迭代，1.4B/2.6B 循环模型匹敌 12B 级模型 |
| 2025-10 | 🆕 [Ring-linear](https://arxiv.org/abs/2510.19338) `arXiv:2510.19338` | Ant Group | 线性与 softmax 注意力混合的长上下文推理模型族 |
| 2025-10 | ★ 🆕 [TRM](https://arxiv.org/abs/2510.04871) `arXiv:2510.04871` | Samsung SAIL | 只有 2 层、7M 参数的递归网络，在 ARC-AGI 上超过多数大模型；简化 HRM |
| 2025-09 | 🆕 [Dragon Hatchling（BDH）](https://arxiv.org/abs/2509.26507) `arXiv:2509.26507` | Pathway | 受大脑启发的局部交互神经元网络，与 Transformer 性能相当且具可解释的稀疏激活 |
| 2025-09 | 🆕 [MemMamba](https://arxiv.org/abs/2510.03279) `arXiv:2510.03279` |  | 重新思考 SSM 的记忆模式，缓解长序列遗忘 |
| 2025-09 | 🆕 [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | Qwen | Gated DeltaNet 与门控注意力 3:1 混合 + 超稀疏 MoE（80B 总参 / 3B 激活），长上下文吞吐提升 10 倍 |
| 2025-08 | 🆕 [门控联想记忆](https://arxiv.org/abs/2509.00605) `arXiv:2509.00605` |  | 并行 O(N) 的门控联想记忆序列模型 |
| 2025-08 | 🆕 [UltraMemV2](https://arxiv.org/abs/2508.18756) `arXiv:2508.18756` | ByteDance Seed | 记忆层网络扩展到 120B 参数，长上下文学习优于 MoE |
| 2025-08 | 🆕 [Nemotron Nano 2](https://arxiv.org/abs/2508.14444) `arXiv:2508.14444` | NVIDIA | Mamba-Transformer 混合的推理模型，长输出吞吐显著提升 |
| 2025-07 | 🆕 [Falcon-H1](https://arxiv.org/abs/2507.22448) `arXiv:2507.22448` | TII | 注意力与 Mamba-2 头并行的混合族，0.5B–34B |
| 2025-07 | 🆕 [跳层还是循环](https://arxiv.org/abs/2507.07996) `arXiv:2507.07996` |  | 测试时为每个输入自适应地跳过或重复预训练 LLM 的层 |
| 2025-06 | 🆕 [HRM](https://arxiv.org/abs/2506.21734) `arXiv:2506.21734` | Sapient | 高低两级循环模块的层级推理模型，27M 参数在 ARC/数独上表现突出 |
| 2025-06 | 🆕 [MiniMax-M1](https://arxiv.org/abs/2506.13585) `arXiv:2506.13585` | MiniMax | 混合注意力推理模型，生成 10 万 token 的计算量约为 DeepSeek-R1 的 25%；CISPO 强化学习 |
| 2025-05 | 🆕 [LaCT](https://arxiv.org/abs/2505.23884) `arXiv:2505.23884` | MIT/Adobe | 大块测试时训练：以数千 token 为块更新快权重，提高硬件利用率 |
| 2025-04 | 🆕 [Miras（It's All Connected）](https://arxiv.org/abs/2504.13173) `arXiv:2504.13173` | Google | 把 Transformer、线性 RNN、Titans 统一为"注意力偏置 + 保留门 + 在线优化"的联想记忆框架 |
| 2025-03 | 🆕 [TransMamba（序列级混合）](https://arxiv.org/abs/2503.24067) `arXiv:2503.24067` |  | 在序列维度切换 Transformer 与 Mamba 的混合语言模型 |
| 2025-03 | 🆕 [RWKV-7 Goose](https://arxiv.org/abs/2503.14456) `arXiv:2503.14456` | RWKV | 广义 delta 规则 + 向量门控，可识别所有正则语言，超出 TC0 的状态追踪能力 |
| 2025-02 | 🆕 [循环深度推理](https://arxiv.org/abs/2502.05171) `arXiv:2502.05171` | Maryland/ELLIS | 在潜空间循环同一组层来扩展测试时计算，无需生成更多 token |
| 2025-01 | ★ 🆕 [MiniMax-01](https://arxiv.org/abs/2501.08313) `arXiv:2501.08313` | MiniMax | 闪电注意力（线性）与 softmax 7:1 混合的 456B MoE，百万级上下文 |
| 2024-12 | ★ 🆕 [Titans](https://arxiv.org/abs/2501.00663) `arXiv:2501.00663` | Google | 神经长期记忆模块：按"惊奇度"（梯度）在测试时写入记忆，与注意力组合 |
| 2024-12 | 🆕 [记忆层扩展](https://arxiv.org/abs/2412.09764) `arXiv:2412.09764` | Meta | 可训练的大规模键值记忆层（128B 记忆参数），以极少算力增加事实性知识 |
| 2024-12 | ★ 🆕 [Gated DeltaNet](https://arxiv.org/abs/2412.06464) `arXiv:2412.06464` | NVIDIA/MIT | 门控（快速遗忘）+ delta 规则（精确改写）互补，成为混合架构的主力线性层 |
| 2024-11 | 🆕 [负特征值解锁状态追踪](https://arxiv.org/abs/2411.12537) `arXiv:2411.12537` | Freiburg/Oxford | 允许线性 RNN 状态转移矩阵有负特征值，即可解决奇偶校验等状态追踪任务 |
| 2024-07 | [TTT 层](https://arxiv.org/abs/2407.04620) `arXiv:2407.04620` | Stanford/UCSD | 把隐状态本身做成一个模型，用自监督梯度步在测试时更新（Test-Time Training） |
| 2024-06 | [DeltaNet 并行化](https://arxiv.org/abs/2406.06484) `arXiv:2406.06484` | MIT | 把 delta 规则（在线回归式的记忆更新）的线性注意力做成可并行训练 |
| 2024-05 | ★ [Mamba-2（SSD）](https://arxiv.org/abs/2405.21060) `arXiv:2405.21060` | Princeton/CMU | 状态空间对偶：结构化 SSM 与带掩码的线性注意力是同一族半可分矩阵 |
| 2024-05 | [xLSTM](https://arxiv.org/abs/2405.04517) `arXiv:2405.04517` | NXAI/JKU | 指数门控 + 矩阵记忆，重新扩展 LSTM |
| 2024-03 | [Jamba](https://arxiv.org/abs/2403.19887) `arXiv:2403.19887` | AI21 | Transformer + Mamba + MoE 混合，较早的生产级混合架构 |
| 2023-12 | [GLA](https://arxiv.org/abs/2312.06635) `arXiv:2312.06635` | MIT | 数据相关门控的线性注意力 + 硬件高效分块训练 |
| 2023-12 | ★ [Mamba](https://arxiv.org/abs/2312.00752) `arXiv:2312.00752` | CMU/Princeton | 输入相关的选择性 SSM + 硬件感知扫描，线性复杂度下首次在语言上匹敌 Transformer |
| 2023-07 | [RetNet](https://arxiv.org/abs/2307.08621) `arXiv:2307.08621` | Microsoft/Tsinghua | 保留机制：并行、递归、分块递归三种等价计算形式 |
| 2023-05 | [RWKV](https://arxiv.org/abs/2305.13048) `arXiv:2305.13048` | RWKV | 可并行训练、按 RNN 推理的线性注意力语言模型 |
| 2023-02 | [Hyena](https://arxiv.org/abs/2302.10866) `arXiv:2302.10866` | Stanford | 隐式长卷积 + 门控的无注意力算子 |
| 2021-10 | ★ [S4](https://arxiv.org/abs/2111.00396) `arXiv:2111.00396` | Stanford | 结构化状态空间模型：HiPPO 初始化 + 卷积/递归双视图，攻克长程依赖 |
| 2020-06 | ★ [线性 Transformer](https://arxiv.org/abs/2006.16236) `arXiv:2006.16236` | Idiap/EPFL | 核化注意力 → 线性复杂度，自回归 Transformer 即 RNN |

### 混合专家与条件计算（30）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [重新思考路由探索](https://arxiv.org/abs/2610.22353) `arXiv:2610.22353` |  | 现代 MoE 路由中的探索问题 |
| 2026-10 | 🔥 [RouterInterp](https://arxiv.org/abs/2610.11775) `arXiv:2610.11775` |  | MoE 路由中的叠加式专门化 |
| 2026-09 | 🔥 [MoE 的统计理解](https://arxiv.org/abs/2609.03501) `arXiv:2609.03501` |  | 从统计学角度理解混合专家 |
| 2026-07 | 🔥 [MoE 路由的一致重叠](https://arxiv.org/abs/2607.28308) `arXiv:2607.28308` |  | 共选专家并非越互补越好：区分路由一致性、候选质量与上下文交互 |
| 2026-06 | 🔥 [流形幂迭代路由器](https://arxiv.org/abs/2606.12397) `arXiv:2606.12397` |  | 用流形上的幂迭代重新设计 MoE 路由器 |
| 2026-05 | 🔥 [EMO](https://arxiv.org/abs/2605.06663) `arXiv:2605.06663` |  | 预训练出"涌现模块化"的 MoE：专家按能力分工，可单独抽出使用 |
| 2026-04 | ◆ [免路由 MoE](https://arxiv.org/abs/2604.00801) `arXiv:2604.00801` |  | 去掉显式路由器的混合专家 |
| 2026-01 | ◆ [扩展嵌入胜过扩展专家](https://arxiv.org/abs/2601.21204) `arXiv:2601.21204` | LongCat | 把稀疏参数放进 n-gram 嵌入比增加专家更划算 |
| 2025-12 | ◆ [耦合专家与路由器](https://arxiv.org/abs/2512.23447) `arXiv:2512.23447` | ByteDance Seed | 辅助损失约束路由决策与专家能力一致 |
| 2025-11 | ◆ [选择性 Sinkhorn 路由](https://arxiv.org/abs/2511.08972) `arXiv:2511.08972` |  | 用 Sinkhorn 最优传输改进稀疏 MoE 的负载均衡 |
| 2025-11 | ◆ [路由流形对齐](https://arxiv.org/abs/2511.07419) `arXiv:2511.07419` |  | 现有 MoE 路由器次优；让路由权重流形与任务嵌入流形对齐，提升泛化 |
| 2025-10 | 🆕 [MoE 中的稀疏与叠加](https://arxiv.org/abs/2510.23671) `arXiv:2510.23671` |  | 专家网络中特征叠加的程度与稀疏性的关系 |
| 2025-09 | 🆕 [全面的 MoE 扩展律](https://arxiv.org/abs/2509.23678) `arXiv:2509.23678` |  | 同时考虑专家数、粒度、共享专家等多个因素的 MoE 扩展律 |
| 2025-07 | 🆕 [高效 MoE 的扩展律](https://arxiv.org/abs/2507.17702) `arXiv:2507.17702` | Ant Group | 以"杠杆率"刻画 MoE 相对稠密模型的效率并给出扩展律 |
| 2025-07 | 🆕 [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) `arXiv:2507.10524` | KAIST/Google | 参数共享的递归块 + 按 token 路由的递归深度，兼顾参数与计算效率 |
| 2025-06 | 🆕 [Chain-of-Experts](https://arxiv.org/abs/2506.18945) `arXiv:2506.18945` |  | 层内专家顺序通信，释放 MoE 的通信表达力 |
| 2025-02 | 🆕 [MoE 联合扩展律](https://arxiv.org/abs/2502.05172) `arXiv:2502.05172` | Warsaw/IDEAS | 考虑显存约束的稠密与 MoE 联合扩展律：同等显存与算力下 MoE 也可更省 |
| 2025-02 | 🆕 [MoE 上采样扩展律](https://arxiv.org/abs/2502.03009) `arXiv:2502.03009` |  | 从稠密模型"升级改造"为 MoE 的扩展律与收益边界 |
| 2025-01 | 🆕 [负载均衡损失的实现细节](https://arxiv.org/abs/2501.11873) `arXiv:2501.11873` | Qwen | 负载均衡应在全局批次而非微批次上计算，专家专门化显著提升 |
| 2025-01 | 🆕 [参数 vs FLOPs：MoE 最优稀疏度](https://arxiv.org/abs/2501.12370) `arXiv:2501.12370` | Apple | 固定算力下更稀疏（更多总参数）的 MoE 预训练损失更低 |
| 2024-09 | [OLMoE](https://arxiv.org/abs/2409.02060) `arXiv:2409.02060` | Ai2 | 完全开放的 MoE 语言模型与训练细节，分析路由与专家专门化 |
| 2024-08 | [无辅助损失负载均衡](https://arxiv.org/abs/2408.15664) `arXiv:2408.15664` | DeepSeek | 用每个专家的偏置项动态调节路由，避免辅助损失干扰主目标（DeepSeek-V3 采用） |
| 2024-04 | [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) `arXiv:2404.02258` | Google DeepMind | 按 token 动态决定是否跳过某层，条件计算从"宽度"扩展到"深度" |
| 2024-02 | [细粒度 MoE 扩展律](https://arxiv.org/abs/2402.07871) `arXiv:2402.07871` | IDEAS NCBR/Warsaw | 引入"粒度"超参数的 MoE 扩展律，最优粒度随规模增大 |
| 2024-01 | ★ [DeepSeekMoE](https://arxiv.org/abs/2401.06066) `arXiv:2401.06066` | DeepSeek | 细粒度专家切分 + 共享专家，提高专家专门化程度 |
| 2024-01 | [Mixtral](https://arxiv.org/abs/2401.04088) `arXiv:2401.04088` | Mistral AI | 8 专家 Top-2 的开源稀疏 MoE，激活 13B 匹敌 Llama 2 70B |
| 2022-02 | [ST-MoE](https://arxiv.org/abs/2202.08906) `arXiv:2202.08906` | Google | 稳定可迁移的 MoE：router z-loss 等稳定性技巧 |
| 2021-01 | [Switch Transformer](https://arxiv.org/abs/2101.03961) `arXiv:2101.03961` | Google | Top-1 路由的简化 MoE，扩展到万亿参数 |
| 2020-06 | [GShard](https://arxiv.org/abs/2006.16668) `arXiv:2006.16668` | Google | 自动分片 + Top-2 MoE，6000 亿参数翻译模型 |
| 2017-01 | ★ [稀疏门控 MoE](https://arxiv.org/abs/1701.06538) `arXiv:1701.06538` | Google | 条件计算：每个样本只激活少数专家，参数扩到 1370 亿 |

### 自监督与表征学习（60）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [用 JEPA 证明并改进柏拉图表征假设](https://arxiv.org/abs/2610.39302) `arXiv:2610.39302` |  | 把柏拉图表征假设变成定理：给出趋同的证明，并由此得到改进表征的配方 |
| 2026-10 | 🔥 [预测闭包](https://arxiv.org/abs/2610.27364) `arXiv:2610.27364` |  | JEPA 两端表征都在学习，它到底选择哪些特征：集合层面的刻画 |
| 2026-10 | 🔥 [LeWAM](https://arxiv.org/abs/2610.12818) `arXiv:2610.12818` |  | 端到端的 JEPA 世界与动作模型：表征、动力学与控制联合训练 |
| 2026-10 | 🔥 [DSReg](https://arxiv.org/abs/2610.09457) `arXiv:2610.09457` |  | 不靠重构、辅助监督或非高斯性，也能可证明地恢复各个世界潜变量 |
| 2026-10 | 🔥 [驱动与衰减：JEPA 的训练动力学](https://arxiv.org/abs/2610.02344) `arXiv:2610.02344` |  | JEPA 训练中驱动项与衰减项的竞争决定学到的特征 |
| 2026-09 | 🔥 [各向异性表征改进 JEPA 世界模型的规划](https://arxiv.org/abs/2609.37441) `arXiv:2609.37441` |  | 与 LeJEPA 的各向同性主张形成对照 |
| 2026-09 | 🔥 [PDE-JEPA](https://arxiv.org/abs/2609.34715) `arXiv:2609.34715` |  | 参数化 PDE 的预测式表征学习，代替以重构为目标的方法 |
| 2026-09 | 🔥 [JEPA 学的是掩码留下的不可恢复部分](https://arxiv.org/abs/2609.32481) `arXiv:2609.32481` |  | JEPA 学到什么特征 |
| 2026-09 | 🔥 [超越高斯世界](https://arxiv.org/abs/2609.21656) `arXiv:2609.21656` |  | JEPA 的潜几何 |
| 2026-09 | 🔥 [JEPA-Anything](https://arxiv.org/abs/2609.20800) `arXiv:2609.20800` |  | 与领域无关的预测式表征学习：同一原则建模迥异的世界 |
| 2026-08 | 🔥 [表征几何对 JEPA 规划很重要](https://arxiv.org/abs/2608.16287) `arXiv:2608.16287` |  | 潜空间几何与规划质量 |
| 2026-07 | 🔥 [语言中的 JEPA 悖论](https://arxiv.org/abs/2607.23531) `arXiv:2607.23531` |  | 语言替代表达的几何 |
| 2026-07 | 🔥 [自然图像对比学习的理论](https://arxiv.org/abs/2607.07470) `arXiv:2607.07470` |  | 对平稳统计的图像数据与基本增强，解析求出对比损失的最优表示 |
| 2026-07 | 🔥 [SiamJEPA](https://arxiv.org/abs/2607.04044) `arXiv:2607.04044` |  | JEPA 中孪生学生编码器的作用 |
| 2026-07 | 🔥 [自监督 ViT 的几何表征](https://arxiv.org/abs/2607.01987) `arXiv:2607.01987` |  | 子空间干预分析 |
| 2026-06 | 🔥 [JEPA 世界模型的泛化理论](https://arxiv.org/abs/2606.27014) `arXiv:2606.27014` |  | JEPA 世界模型的泛化界 |
| 2026-05 | 🔥 [SPHERE-JEPA](https://arxiv.org/abs/2605.26900) `arXiv:2605.26900` |  | 在超球面上做预测 |
| 2026-05 | 🔥 [LeJEPA 何时学到世界模型](https://arxiv.org/abs/2605.26379) `arXiv:2605.26379` |  | LeJEPA 学到世界模型的条件 |
| 2026-05 | 🔥 [超越 JEPA 的各向同性](https://arxiv.org/abs/2605.20107) `arXiv:2605.20107` |  | 哈密顿几何与辛预测 |
| 2026-05 | 🔥 [人脑中的柏拉图表征](https://arxiv.org/abs/2605.20496) `arXiv:2605.20496` |  | 无监督地在人脑响应中恢复与模型共享的通用几何 |
| 2026-05 | 🔥 [从潜分布匹配理解自监督学习](https://arxiv.org/abs/2605.03517) `arXiv:2605.03517` |  | 用潜分布匹配统一理解 SSL |
| 2026-04 | ◆ [回到柏拉图的洞穴](https://arxiv.org/abs/2604.18572) `arXiv:2604.18572` |  | 大规模检验跨模态表征趋同：结论比原假设弱 |
| 2026-03 | ◆ [IConE](https://arxiv.org/abs/2603.15263) `arXiv:2603.15263` |  | 与批无关的坍缩预防 |
| 2026-03 | ◆ [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) `arXiv:2603.14482` | Meta | 为视频自监督 JEPA 解锁稠密特征 |
| 2026-03 | ◆ [重构损失是罪魁吗](https://arxiv.org/abs/2603.14131) `arXiv:2603.14131` |  | 尝试用重构式目标超越 JEPA |
| 2026-03 | ◆ [方向性神经坍缩解释 SSL 的少样本迁移](https://arxiv.org/abs/2603.03530) `arXiv:2603.03530` |  | 自监督表征的方向性神经坍缩 |
| 2026-03 | ◆ [Utonia](https://arxiv.org/abs/2603.03283) `arXiv:2603.03283` | Pointcept | 一个编码器覆盖所有点云 |
| 2026-02 | ◆ [柏拉图表征假设的亚里士多德式重审](https://arxiv.org/abs/2602.14486) `arXiv:2602.14486` |  | 现有表征相似度度量被网络规模混淆；校正后全局趋同减弱，局部邻域结构仍趋同 |
| 2026-02 | ◆ [Causal-JEPA](https://arxiv.org/abs/2602.11389) `arXiv:2602.11389` |  | 以物体级潜在干预扩展掩码 JEPA，学习依赖交互的世界模型 |
| 2026-02 | ◆ [多路表征对齐](https://arxiv.org/abs/2602.06205) `arXiv:2602.06205` |  | 多个模型/模态间的表征对齐 |
| 2026-01 | ◆ [掩码自监督回归的随机矩阵理论](https://arxiv.org/abs/2601.23208) `arXiv:2601.23208` |  | 掩码预测的精确高维理论 |
| 2025-12 | ◆ [从下一帧预测中学习](https://arxiv.org/abs/2512.21004) `arXiv:2512.21004` |  | 自回归视频建模（下一帧预测）能学到有效的视觉表征 |
| 2025-12 | ◆ [VL-JEPA](https://arxiv.org/abs/2512.10942) `arXiv:2512.10942` | Meta | 视觉—语言的联合嵌入预测架构：在嵌入空间预测文本目标 |
| 2025-11 | ◆ [LeJEPA](https://arxiv.org/abs/2511.08544) `arXiv:2511.08544` | Meta/NYU | 证明各向同性高斯是 JEPA 嵌入的最优分布，用 SIGReg 正则取代停梯度、教师网络等启发式技巧 |
| 2025-10 | 🆕 [Concerto](https://arxiv.org/abs/2510.23607) `arXiv:2510.23607` | HKU/Pointcept | 2D-3D 联合自监督：点云自蒸馏 + 跨模态预测，涌现出空间表征 |
| 2025-09 | 🆕 [EmbeddingGemma](https://arxiv.org/abs/2509.20354) `arXiv:2509.20354` | Google | 轻量而强的文本表征模型 |
| 2025-09 | 🆕 [OpenVision 2](https://arxiv.org/abs/2509.01644) `arXiv:2509.01644` | UCSC | 只用生成式字幕损失预训练的开放视觉编码器族 |
| 2025-08 | ★ 🆕 [DINOv3](https://arxiv.org/abs/2508.10104) `arXiv:2508.10104` | Meta | 7B 自监督视觉骨干 + Gram anchoring 解决长训练中稠密特征退化，冻结特征在稠密任务上全面领先 |
| 2025-07 | 🆕 [DINO-world](https://arxiv.org/abs/2507.19468) `arXiv:2507.19468` | Meta | 在冻结 DINOv2 特征空间里训练视频世界模型 |
| 2025-07 | 🆕 [Franca](https://arxiv.org/abs/2507.14137) `arXiv:2507.14137` | valeo.ai 等 | 完全开放（数据、代码、权重）的视觉基础模型，嵌套 Matryoshka 聚类 |
| 2025-06 | ★ 🆕 [V-JEPA 2](https://arxiv.org/abs/2506.09985) `arXiv:2506.09985` | Meta | 百万小时视频自监督预训练 + 少量机器人数据做动作条件世界模型，零样本规划抓取 |
| 2025-06 | 🆕 [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) `arXiv:2506.05176` | Qwen | 基于基础模型的文本嵌入与重排 |
| 2025-05 | 🆕 [嵌入的通用几何](https://arxiv.org/abs/2505.12540) `arXiv:2505.12540` | Cornell | 无配对数据即可在不同文本嵌入模型的向量空间之间翻译，支持强柏拉图表征假说 |
| 2025-04 | 🆕 [Perception Encoder](https://arxiv.org/abs/2504.13181) `arXiv:2504.13181` | Meta | 只用对比视觉-语言训练的编码器，最好的特征藏在中间层，经对齐后通用于各类任务 |
| 2025-04 | 🆕 [Web-SSL](https://arxiv.org/abs/2504.01017) `arXiv:2504.01017` | Meta/NYU | 纯视觉自监督扩展到 7B 参数与数十亿图像后，在 VQA 上追平 CLIP：语言监督并非必需 |
| 2025-03 | 🆕 [Gemini Embedding](https://arxiv.org/abs/2503.07891) `arXiv:2503.07891` | Google | 基于 Gemini 的通用文本嵌入模型 |
| 2025-02 | 🆕 [SigLIP 2](https://arxiv.org/abs/2502.14786) `arXiv:2502.14786` | Google DeepMind | 加入字幕生成、自蒸馏与掩码预测，多语言、稠密特征与定位能力增强 |
| 2024-11 | 🆕 [AIMv2](https://arxiv.org/abs/2411.14402) `arXiv:2411.14402` | Apple | 多模态自回归预训练视觉编码器：解码器同时重建图像块与文本 |
| 2024-10 | [REPA](https://arxiv.org/abs/2410.06940) `arXiv:2410.06940` | KAIST/NYU | 把扩散 Transformer 的中间表征对齐到预训练视觉编码器，训练提速 17 倍：表征学习帮助生成 |
| 2024-02 | [V-JEPA](https://arxiv.org/abs/2404.08471) `arXiv:2404.08471` | Meta | 仅用特征预测目标从视频学习视觉表征 |
| 2023-04 | ★ [DINOv2](https://arxiv.org/abs/2304.07193) `arXiv:2304.07193` | Meta | 大规模精选数据上的自监督 ViT，冻结特征即可用于各种稠密任务 |
| 2023-03 | [SigLIP](https://arxiv.org/abs/2303.15343) `arXiv:2303.15343` | Google | 用 sigmoid 损失替代 softmax 对比损失，不再依赖全局归一化 |
| 2023-01 | [I-JEPA](https://arxiv.org/abs/2301.08243) `arXiv:2301.08243` | Meta | 在表征空间而非像素空间预测被遮挡区域，联合嵌入预测架构 |
| 2021-11 | ★ [MAE](https://arxiv.org/abs/2111.06377) `arXiv:2111.06377` | Meta | 遮住 75% 的图像块再重建，简单可扩展的视觉预训练 |
| 2021-04 | [DINO](https://arxiv.org/abs/2104.14294) `arXiv:2104.14294` | Facebook | 无标签自蒸馏的 ViT 自发涌现语义分割注意力 |
| 2021-02 | ★ [CLIP](https://arxiv.org/abs/2103.00020) `arXiv:2103.00020` | OpenAI | 4 亿图文对比学习，零样本迁移，连接视觉与语言 |
| 2020-06 | [BYOL](https://arxiv.org/abs/2006.07733) `arXiv:2006.07733` | DeepMind | 无负样本的自蒸馏：在线网络预测动量目标网络 |
| 2020-02 | ★ [SimCLR](https://arxiv.org/abs/2002.05709) `arXiv:2002.05709` | Google | 强数据增强 + 大批量对比学习 + 投影头，自监督逼近监督 |
| 2019-11 | [MoCo](https://arxiv.org/abs/1911.05722) `arXiv:1911.05722` | Facebook | 动量编码器 + 队列，使对比学习不依赖超大批量 |
| 2018-07 | [CPC / InfoNCE](https://arxiv.org/abs/1807.03748) `arXiv:1807.03748` | DeepMind | 对比预测编码：用 InfoNCE 损失最大化上下文与未来的互信息 |

### 生成模型（98）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [浅层网络的扩散分数估计](https://arxiv.org/abs/2610.27206) `arXiv:2610.27206` |  | 正则性、逼近与有限样本界 |
| 2026-10 | 🔥 [自条件让流匹配能深度序贯推理](https://arxiv.org/abs/2610.19628) `arXiv:2610.19628` |  | 直觉上每步采样像一步推理，实际流匹配难以做深度推理；自条件化可以解决 |
| 2026-10 | 🔥 [单步生成是否想多了](https://arxiv.org/abs/2610.20331) `arXiv:2610.20331` |  | 重新审视单步生成方法的必要复杂度 |
| 2026-10 | 🔥 [扩展在线最优传输小批量](https://arxiv.org/abs/2610.18838) `arXiv:2610.18838` |  | 流匹配中源与数据的配对方式决定稳定性：把小批量 OT 扩展到大规模 |
| 2026-10 | 🔥 [学习引导像素扩散的表征](https://arxiv.org/abs/2610.12978) `arXiv:2610.12978` |  | 表征质量更高不等于生成效用更大：为像素 DiT 学习专门的引导表征 |
| 2026-10 | 🔥 [OpenEBFM](https://arxiv.org/abs/2610.33005) `arXiv:2610.33005` |  | 基于能量的基础模型可扩展 |
| 2026-10 | 🔥 [漂移模型的判别几何](https://arxiv.org/abs/2610.04703) `arXiv:2610.04703` |  | 漂移模型（单步生成）的效果高度依赖构造漂移场所用的表征：学习判别性几何 |
| 2026-10 | 🔥 [ALoDLM](https://arxiv.org/abs/2610.04198) `arXiv:2610.04198` |  | 扩散语言模型的差距源于计算与难度失配：对难位置自适应循环 |
| 2026-10 | 🔥 [一步生成中把深度当作时间](https://arxiv.org/abs/2610.03626) `arXiv:2610.03626` |  | 单步生成模型中网络深度扮演了 ODE 时间的角色 |
| 2026-09 | 🔥 [视觉生成器需要怎样的老师](https://arxiv.org/abs/2609.34732) `arXiv:2609.34732` |  | 重新思考表征对齐（REPA）的教师选择 |
| 2026-09 | 🔥 [残差流负担塑造 DiT 的表征学习](https://arxiv.org/abs/2609.33895) `arXiv:2609.33895` |  | 残差流承担的信息量如何影响扩散 Transformer 的表征 |
| 2026-09 | 🔥 [扩散模型的双下降与恶性过拟合](https://arxiv.org/abs/2609.26392) `arXiv:2609.26392` |  | 扩散模型中的双下降与恶性过拟合 |
| 2026-08 | 🔥 [紧黎曼流形上扩散模型的物种形成](https://arxiv.org/abs/2608.23798) `arXiv:2608.23798` |  | 把物种形成相变理论推广到流形 |
| 2026-08 | 🔥 [像素空间文生图扩散的实证研究](https://arxiv.org/abs/2608.16887) `arXiv:2608.16887` |  | 大规模系统研究不使用 VAE、直接在像素空间训练的文生图扩散模型 |
| 2026-08 | 🔥 [用有效场论看扩散中的信息传播](https://arxiv.org/abs/2608.14308) `arXiv:2608.14308` |  | 用有效场论描述扩散生成中信息在尺度间的传播 |
| 2026-08 | 🔥 [AURORA-LM](https://arxiv.org/abs/2608.02602) `arXiv:2608.02602` |  | 为连续潜空间扩散语言建模专门设计联合生成与解码的统一表征自编码器 |
| 2026-07 | 🔥 [随机时钟实现重尾流匹配](https://arxiv.org/abs/2607.13841) `arXiv:2607.13841` |  | 用随机时间变换让流匹配生成重尾分布 |
| 2026-07 | 🔥 [贝叶斯扩散模型泛化相变的精确信息论](https://arxiv.org/abs/2607.08041) `arXiv:2607.08041` |  | 精确刻画贝叶斯最优扩散模型从记忆到泛化的相变 |
| 2026-07 | 🔥 [潜在时钟](https://arxiv.org/abs/2607.01774) `arXiv:2607.01774` |  | 扩散语言模型没有显式时间步输入，却在内部隐式地估计扩散时间 |
| 2026-07 | 🔥 [扩散模型中不会出现良性过拟合](https://arxiv.org/abs/2607.02671) `arXiv:2607.02671` |  | 扩散模型完美拟合训练集时只会复制样本，过拟合必然有害 |
| 2026-06 | 🔥 [FID 彩票](https://arxiv.org/abs/2606.20536) `arXiv:2606.20536` |  | 把 FID 当随机变量：重训与重采样带来的隐藏随机性足以改变排名 |
| 2026-06 | 🔥 [谱强迫](https://arxiv.org/abs/2606.15236) `arXiv:2606.15236` |  | 自然图像幂律谱下，k*(t)=(1−t)^(−2/α) 把每个时刻分成信号主导的低频与噪声主导的高频：像素扩散只学前者 |
| 2026-06 | 🔥 [神经网络流匹配理论](https://arxiv.org/abs/2606.10089) `arXiv:2606.10089` |  | 用神经网络实现流匹配的统计与逼近理论 |
| 2026-06 | 🔥 [DMD 学生为何偷懒](https://arxiv.org/abs/2606.02237) `arXiv:2606.02237` |  | 少步分布匹配蒸馏中的复制行为：成因分析 |
| 2026-05 | 🔥 [对称注意力分解](https://arxiv.org/abs/2605.27476) `arXiv:2605.27476` |  | 把注意力矩阵分解为对称部分（决定能量地形，Hopfield 视角）与反对称部分，以平衡保真与多样性 |
| 2026-05 | 🔥 [扩散理论教程](https://arxiv.org/abs/2605.22586) `arXiv:2605.22586` |  | 从微分方程到扩散模型的理论教程 |
| 2026-05 | 🔥 [RiT](https://arxiv.org/abs/2605.21981) `arXiv:2605.21981` |  | 预训练表征空间与像素空间有相近的内在维度：普通 DiT 直接在表征空间生成即可 |
| 2026-05 | 🔥 [用动力系统看生成模型训练与记忆](https://arxiv.org/abs/2605.19483) `arXiv:2605.19483` |  | 记忆现象的动力系统解释 |
| 2026-05 | 🔥 [非对称流模型](https://arxiv.org/abs/2605.12964) `arXiv:2605.12964` |  | 数据有低秩结构时，把噪声预测限制在低秩子空间的秩非对称速度参数化 |
| 2026-05 | 🔥 [Wasserstein 梯度流单步生成](https://arxiv.org/abs/2605.11755) `arXiv:2605.11755` |  | 用 Wasserstein 梯度流构造单步生成 |
| 2026-05 | 🔥 [离散流匹配的收敛保证](https://arxiv.org/abs/2605.08882) `arXiv:2605.08882` |  | 最少假设下离散流匹配的收敛性 |
| 2026-05 | 🔥 [能量模型的分布简单性偏置](https://arxiv.org/abs/2605.07844) `arXiv:2605.07844` |  | 能量模型训练中的分布简单性偏置与有效凸性 |
| 2026-05 | 🔥 [扩散模型何时可忽略维度](https://arxiv.org/abs/2605.07969) `arXiv:2605.07969` |  | 基于熵的理论：扩散采样复杂度由内在熵而非外围维度决定 |
| 2026-05 | 🔥 [归一化轨迹模型](https://arxiv.org/abs/2605.08078) `arXiv:2605.08078` |  | 少步生成同时保留似然框架：把生成轨迹压缩为少数可逆的粗粒度转移 |
| 2026-05 | 🔥 [连续潜空间扩散语言模型](https://arxiv.org/abs/2605.06548) `arXiv:2605.06548` |  | 在连续潜空间而非离散 token 上做扩散语言建模 |
| 2026-05 | 🔥 [轨道空间粒子流匹配](https://arxiv.org/abs/2605.02222) `arXiv:2605.02222` |  | 粒子系统有置换对称性，在轨道空间上做流匹配以拉直流 |
| 2026-04 | ◆ [表征 Fréchet 损失](https://arxiv.org/abs/2604.28190) `arXiv:2604.28190` |  | 在表征空间中直接优化 Fréchet 距离作为生成模型的训练目标 |
| 2026-03 | ◆ [扩散模型先在流形上泛化、后记忆](https://arxiv.org/abs/2603.23792) `arXiv:2603.23792` |  | 证明训练中扩散模型先学到数据流形（泛化），之后才开始记忆单个样本 |
| 2026-03 | ◆ [基于快照的广义离散扩散](https://arxiv.org/abs/2603.21342) `arXiv:2603.21342` |  | 支持任意加噪过程的离散扩散框架，涵盖现有各类离散扩散 |
| 2026-03 | ◆ [JiT 的表征对齐并不简单](https://arxiv.org/abs/2603.14366) `arXiv:2603.14366` |  | 像素空间 JiT 去掉了分词器，但表征对齐在其上并不像在潜空间中那样直接有效 |
| 2026-03 | ◆ [扩散模型从易到难学习数据统计](https://arxiv.org/abs/2603.12901) `arXiv:2603.12901` |  | 扩散模型按从低阶到高阶的顺序学习数据统计量 |
| 2026-03 | ◆ [整流流中的泛化与记忆](https://arxiv.org/abs/2603.13421) `arXiv:2603.13421` |  | 整流流（rectified flow）的泛化与记忆 |
| 2026-03 | ◆ [扩散模型的结构性记忆](https://arxiv.org/abs/2603.13419) `arXiv:2603.13419` |  | 超越逐样本复制：扩散模型记住的是结构 |
| 2026-03 | ◆ [梯度流漂移](https://arxiv.org/abs/2603.10592) `arXiv:2603.10592` |  | 把漂移模型解释为核密度近似散度的 Wasserstein 梯度流 |
| 2026-02 | ◆ [掩码扩散模型的精确收敛率](https://arxiv.org/abs/2602.22505) `arXiv:2602.22505` |  | 掩码离散扩散的采样收敛率 |
| 2026-02 | ◆ [扩散对偶（二）](https://arxiv.org/abs/2602.21185) `arXiv:2602.21185` |  | 均匀离散扩散的 Ψ 采样器：步数增加时质量不再停滞 |
| 2026-02 | ◆ [Flow Map 语言模型](https://arxiv.org/abs/2602.16813) `arXiv:2602.16813` |  | 学习流映射：基于连续去噪的语言模型在少步区间不退化，可做一步生成 |
| 2026-02 | ◆ [在流形上学习](https://arxiv.org/abs/2602.10099) `arXiv:2602.10099` |  | 普通 DiT 在表征编码器空间上不收敛并非容量瓶颈，而是需要在流形上学习 |
| 2026-02 | ◆ [Stable Velocity](https://arxiv.org/abs/2602.05435) `arXiv:2602.05435` |  | 从方差角度分析流匹配：靠近先验处目标方差高、优化困难，靠近数据处方差低 |
| 2026-02 | ◆ [扩散模型中的物种形成相变](https://arxiv.org/abs/2602.04404) `arXiv:2602.04404` |  | 一般类结构下生成轨迹"选定类别"的相变理论 |
| 2026-02 | ◆ [对偶模型](https://arxiv.org/abs/2602.17682) `arXiv:2602.17682` |  | 一种极简的单步生成范式 |
| 2026-02 | ◆ [Drifting Models](https://arxiv.org/abs/2602.04770) `arXiv:2602.04770` | MIT | 训练过程中让生成分布向数据分布"漂移"，推理天然一步 |
| 2026-02 | ◆ [扩散模型一致性的随机矩阵视角](https://arxiv.org/abs/2602.02908) `arXiv:2602.02908` |  | 为何不同扩散模型对同一噪声给出相似样本：随机矩阵解释 |
| 2026-02 | ◆ [离散扩散中理解与生成的平衡](https://arxiv.org/abs/2602.01362) `arXiv:2602.01362` |  | 掩码扩散擅长理解、均匀扩散擅长生成：如何兼得 |
| 2026-01 | ◆ [Euler Mean Flow 的轨迹一致性](https://arxiv.org/abs/2602.02571) `arXiv:2602.02571` |  | 改进 MeanFlow 的单步生成 |
| 2026-01 | ◆ [像素 MeanFlow（pMF）](https://arxiv.org/abs/2601.22158) `arXiv:2601.22158` | MIT | 不用潜空间、一步生成：在像素空间预测 x 并以 MeanFlow 损失训练 |
| 2026-01 | ◆ [学习能量生成模型的扩散分类损失](https://arxiv.org/abs/2601.21025) `arXiv:2601.21025` |  | 用扩散式分类损失训练能量模型 |
| 2026-01 | ◆ [用 RAE 扩展文生图 DiT](https://arxiv.org/abs/2601.16208) `arXiv:2601.16208` | NYU | 表征自编码器在文生图规模上仍优于 VAE：收敛更快、更不易过拟合 |
| 2026-01 | ◆ [灵活性陷阱](https://arxiv.org/abs/2601.15165) `arXiv:2601.15165` |  | 任意顺序生成看似解空间更大，实则让扩散语言模型回避难 token，限制推理潜力 |
| 2026-01 | ◆ [重尾目标的扩散模型](https://arxiv.org/abs/2601.06715) `arXiv:2601.06715` |  | 重尾数据下的分数估计与采样保证 |
| 2025-12 | ◆ [棱镜假设](https://arxiv.org/abs/2512.19693) `arXiv:2512.19693` |  | 编码器的特征频谱对应其功能：语义编码器偏低频、像素编码器偏高频，据此统一自编码 |
| 2025-12 | ◆ [比高斯更好的源分布](https://arxiv.org/abs/2512.18184) `arXiv:2512.18184` |  | 流匹配的源分布不必是高斯：用二维模拟理解高维情形并探索替代 |
| 2025-12 | ◆ [NRGPT](https://arxiv.org/abs/2512.16762) `arXiv:2512.16762` |  | 基于能量的 GPT 替代 |
| 2025-12 | ◆ [SoFlow](https://arxiv.org/abs/2512.15657) `arXiv:2512.15657` |  | 学习"解流"而非速度场的单步生成模型 |
| 2025-12 | ◆ [表征对齐中什么重要](https://arxiv.org/abs/2512.10794) `arXiv:2512.10794` |  | REPA 的收益主要来自目标表征的空间结构，而非全局语义（线性探测精度） |
| 2025-12 | ◆ [双向归一化流](https://arxiv.org/abs/2512.10953) `arXiv:2512.10953` |  | 放宽显式可逆约束：学习近似逆向，兼顾表达能力与似然 |
| 2025-12 | ◆ [理解能量模型的温度调节](https://arxiv.org/abs/2512.09152) `arXiv:2512.09152` |  | 能量模型中温度的作用 |
| 2025-12 | ◆ [一层就够](https://arxiv.org/abs/2512.07829) `arXiv:2512.07829` | Apple | 只需一层适配即可把预训练视觉编码器用作生成模型的潜空间 |
| 2025-12 | ◆ [TwinFlow](https://arxiv.org/abs/2512.05150) `arXiv:2512.05150` |  | 用自对抗流在大模型上实现单步生成 |
| 2025-11 | ◆ [反向流动](https://arxiv.org/abs/2511.22345) `arXiv:2511.22345` |  | 在归一化流的反向（生成）过程中对齐表征，提升生成质量 |
| 2025-11 | ◆ [STARFlow-V](https://arxiv.org/abs/2511.20462) `arXiv:2511.20462` | Apple | 端到端的归一化流视频生成模型 |
| 2025-11 | ◆ [JiT](https://arxiv.org/abs/2511.13720) `arXiv:2511.13720` | MIT | 回到像素、直接预测干净图像：流形假设下 x 预测远好于噪声/速度预测 |
| 2025-11 | ◆ [TiDAR](https://arxiv.org/abs/2511.08923) `arXiv:2511.08923` | NVIDIA | 一次前向中用扩散并行起草、用自回归验证采样，兼得质量与吞吐 |
| 2025-11 | ◆ [扩散语言模型是超级数据学习者](https://arxiv.org/abs/2511.03276) `arXiv:2511.03276` |  | 数据受限、多轮重复时，扩散语言模型持续优于自回归模型（交叉点随数据增多而后移） |
| 2025-11 | ◆ [扩散模型中记忆与泛化的可证明分离](https://arxiv.org/abs/2511.03202) `arXiv:2511.03202` |  | 给出扩散模型记忆训练样本与泛化之间的可证明分离 |
| 2025-10 | 🆕 [扩散模型原理（专著）](https://arxiv.org/abs/2510.21890) `arXiv:2510.21890` | Sony/Stanford | 从变分、分数、流三条路统一扩散模型的系统专著 |
| 2025-10 | 🆕 [AlphaFlow](https://arxiv.org/abs/2510.20771) `arXiv:2510.20771` | Snap | 分析 MeanFlow 目标中相互冲突的两项，并用课程把流匹配过渡到 MeanFlow |
| 2025-10 | 🆕 [RAE](https://arxiv.org/abs/2510.11690) `arXiv:2510.11690` | NYU | 用冻结的表征编码器（DINOv2 等）+ 训练的解码器替代 VAE 作扩散潜空间 |
| 2025-10 | 🆕 [Equilibrium Matching](https://arxiv.org/abs/2510.02300) `arXiv:2510.02300` | MIT | 去掉时间条件：学习隐式能量模型的平衡梯度，用优化式采样生成 |
| 2025-08 | 🆕 [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) | Google DeepMind | 实时可交互的通用世界模型：720p、24fps、分钟级一致性 |
| 2025-07 | 🆕 [Energy-Based Transformer](https://arxiv.org/abs/2507.02092) `arXiv:2507.02092` | UVA/Harvard | 学习验证器式的能量函数，推理即能量最小化，在数据与参数扩展上优于 Transformer++ |
| 2025-05 | 🆕 [MeanFlow](https://arxiv.org/abs/2505.13447) `arXiv:2505.13447` | CMU/MIT | 学习区间平均速度而非瞬时速度，从头训练的一步生成 |
| 2025-02 | ★ 🆕 [LLaDA](https://arxiv.org/abs/2502.09992) `arXiv:2502.09992` | Renmin/Ant | 8B 掩码扩散语言模型从零预训练，能力与同规模自回归 LLM 可比 |
| 2024-12 | 🆕 [TarFlow](https://arxiv.org/abs/2412.06329) `arXiv:2412.06329` | Apple | 基于 Transformer 的自回归归一化流，似然与样本质量大幅提升，归一化流复兴 |
| 2024-08 | [Transfusion](https://arxiv.org/abs/2408.11039) `arXiv:2408.11039` | Meta | 一个 Transformer 同时做文本的下一 token 预测与图像的扩散 |
| 2024-06 | [MAR](https://arxiv.org/abs/2406.11838) `arXiv:2406.11838` | MIT/Google | 连续 token 自回归 + 扩散损失头，抛弃向量量化 |
| 2024-04 | [VAR](https://arxiv.org/abs/2404.02905) `arXiv:2404.02905` | PKU/ByteDance | "下一尺度预测"的视觉自回归，首次在 ImageNet 上让 GPT 式模型超过扩散并展示扩展律（NeurIPS 2024 最佳论文） |
| 2024-02 | [Sora](https://openai.com/index/video-generation-models-as-world-simulators/) | OpenAI | 时空 patch 扩散 Transformer，"视频生成即世界模拟器" |
| 2022-12 | [DiT](https://arxiv.org/abs/2212.09748) `arXiv:2212.09748` | UC Berkeley/NYU | 扩散 Transformer：生成模型的骨干也走向 Transformer 与扩展律 |
| 2022-10 | ★ [Flow Matching](https://arxiv.org/abs/2210.02747) `arXiv:2210.02747` | Meta/Weizmann | 免模拟地回归条件速度场训练连续归一化流，今天图像视频模型的主流目标 |
| 2021-12 | ★ [LDM / Stable Diffusion](https://arxiv.org/abs/2112.10752) `arXiv:2112.10752` | LMU/Runway | 在自编码器潜空间做扩散，文生图走向开源普及 |
| 2020-11 | [Score SDE](https://arxiv.org/abs/2011.13456) `arXiv:2011.13456` | Stanford/Google | 用 SDE 统一分数匹配与扩散，提出概率流 ODE |
| 2020-06 | ★ [DDPM](https://arxiv.org/abs/2006.11239) `arXiv:2006.11239` | UC Berkeley | 去噪扩散概率模型：训练化简为预测噪声，质量追平 GAN |
| 2018-12 | [StyleGAN](https://arxiv.org/abs/1812.04948) `arXiv:1812.04948` | NVIDIA | 风格调制的生成器，GAN 生成质量的巅峰之一 |
| 2018-07 | [Glow](https://arxiv.org/abs/1807.03039) `arXiv:1807.03039` | OpenAI | 1×1 可逆卷积的归一化流，高分辨率人脸生成 |
| 2017-11 | [VQ-VAE](https://arxiv.org/abs/1711.00937) `arXiv:1711.00937` | DeepMind | 离散码本的自编码器，后来成为图像/音频 tokenizer 的基础 |
| 2016-05 | [RealNVP](https://arxiv.org/abs/1605.08803) `arXiv:1605.08803` | Google | 仿射耦合层的可逆流，精确似然 |
| 2016-01 | [PixelRNN/PixelCNN](https://arxiv.org/abs/1601.06759) `arXiv:1601.06759` | DeepMind | 逐像素自回归建模图像 |

### 预训练、数据与分词（56）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🔥 [一个 AI token 值多少](https://arxiv.org/abs/2609.40295) `arXiv:2609.40295` |  | 2026 年网页中约 27–31% 的 token 已是 AI 生成：野生 AI 文本对预训练的扩展律影响 |
| 2026-09 | 🔥 [NCP 下一概念预测](https://arxiv.org/abs/2609.10715) `arXiv:2609.10715` | Shanghai AI Lab | 在下一 token 预测之外预测跨多个 token 的离散"概念"，走向潜空间语言模型 |
| 2026-08 | 🔥 [多模态预训练物理学](https://arxiv.org/abs/2608.05000) `arXiv:2608.05000` | Meta | 知识流动、模态协同与早期统一 |
| 2026-07 | 🔥 [函数感知 FIM 中期训练](https://arxiv.org/abs/2607.12463) `arXiv:2607.12463` | TIGER-Lab | 面向编码智能体基础模型的中期训练目标 |
| 2026-07 | 🔥 [Functionalizer](https://arxiv.org/abs/2609.15991) `arXiv:2609.15991` |  | 无损的函数式子词分解：把大小写、变音等正字法变化分解为基词 + 函数 |
| 2026-06 | 🔥 [数据受限预训练中的训练时增强](https://arxiv.org/abs/2606.16246) `arXiv:2606.16246` |  | 数据重复时，训练时数据增强的作用与扩展 |
| 2026-05 | 🔥 [LLM 该如何消费高质量数据](https://arxiv.org/abs/2605.25698) `arXiv:2605.25698` |  | 质量感知的最优数据调度 |
| 2026-05 | 🔥 [Token 叠加预训练](https://arxiv.org/abs/2605.06546) `arXiv:2605.06546` | Nous Research | 把多个 token 叠加到同一位置以提高预训练效率 |
| 2026-04 | ◆ [Adam's Law：文本频率律](https://arxiv.org/abs/2604.02176) `arXiv:2604.02176` |  | 文本频率与 LLM 表现之间的规律 |
| 2026-03 | ◆ [DataFlex](https://arxiv.org/abs/2603.26164) `arXiv:2603.26164` | PKU | 以数据为中心的动态训练统一框架 |
| 2026-03 | ◆ [超越语言建模：多模态预训练](https://arxiv.org/abs/2603.03276) `arXiv:2603.03276` | Meta | 原生多模态预训练的系统实验研究 |
| 2026-03 | ◆ [ByteFlow](https://arxiv.org/abs/2603.03583) `arXiv:2603.03583` |  | 无分词器的层级架构：按信息量自适应压缩字节序列 |
| 2026-02 | ◆ [神奇的相关性](https://arxiv.org/abs/2602.11217) `arXiv:2602.11217` |  | 理解从预训练到监督微调的知识迁移 |
| 2026-02 | ◆ [OPUS](https://arxiv.org/abs/2602.05400) `arXiv:2602.05400` | Qwen | 每一步都按优化器诱导的更新方向评估并挑选预训练数据，在线、有原则的数据选择 |
| 2026-01 | ◆ [daVinci-Dev](https://arxiv.org/abs/2601.18418) `arXiv:2601.18418` | SII-GAIR | 面向软件工程的智能体原生中期训练 |
| 2026-01 | ◆ [TREX](https://arxiv.org/abs/2601.13588) `arXiv:2601.13588` |  | 用 tokenizer 回归寻找最优数据混合 |
| 2025-12 | ◆ [DataFlow](https://arxiv.org/abs/2512.16676) `arXiv:2512.16676` | PKU | LLM 驱动的数据准备与工作流自动化框架 |
| 2025-09 | 🆕 [无限算力下的预训练](https://arxiv.org/abs/2509.14786) `arXiv:2509.14786` | Stanford | 数据受限、算力充足时，强正则化、集成与蒸馏可把数据效率提升 5 倍以上 |
| 2025-09 | 🆕 [持续预训练扩展智能体](https://arxiv.org/abs/2509.13310) `arXiv:2509.13310` | Alibaba Tongyi | 智能体持续预训练（Agentic CPT）为后训练提供更好的基座 |
| 2025-09 | 🆕 [Token 税](https://arxiv.org/abs/2509.05486) `arXiv:2509.05486` |  | 多语言分词带来的系统性成本与性能偏差 |
| 2025-08 | 🆕 [BeyondWeb](https://arxiv.org/abs/2508.10975) `arXiv:2508.10975` | DatologyAI | 万亿 token 规模合成数据预训练的经验：改写网页优于纯生成 |
| 2025-07 | 🆕 [数据受限时扩散胜出](https://arxiv.org/abs/2507.15857) `arXiv:2507.15857` | CMU | 数据有限、算力充足时，掩码扩散语言模型反复训练优于自回归 |
| 2025-07 | 🆕 [数据混合扩展律](https://arxiv.org/abs/2507.09404) `arXiv:2507.09404` | Apple | 把领域权重纳入扩展律，用小规模实验预测大规模最优混合 |
| 2025-07 | 🆕 [H-Net](https://arxiv.org/abs/2507.07955) `arXiv:2507.07955` | CMU/Cartesia | 端到端学习的动态分块：层级网络自己学会"分词"，去掉 tokenizer |
| 2025-07 | 🆕 [编码器还该用 MLM 预训练吗](https://arxiv.org/abs/2507.00994) `arXiv:2507.00994` |  | 比较掩码语言建模与因果语言建模训练文本编码器 |
| 2025-06 | 🆕 [FineWeb2](https://arxiv.org/abs/2506.20920) `arXiv:2506.20920` | Hugging Face | 把预训练数据处理流程自适应到 1000+ 种语言 |
| 2025-06 | 🆕 [OctoThinker](https://arxiv.org/abs/2506.20512) `arXiv:2506.20512` | SJTU | 中期训练（mid-training）决定基座模型能否被 RL 有效提升 |
| 2025-06 | 🆕 [强化预训练](https://arxiv.org/abs/2506.08007) `arXiv:2506.08007` | Microsoft/PKU | 把下一 token 预测改写为可验证的推理任务，用 RL 做预训练 |
| 2025-06 | 🆕 [Common Pile](https://arxiv.org/abs/2506.05209) `arXiv:2506.05209` | Toronto/EleutherAI 等 | 8TB 公有领域与开放许可文本，训出与未授权数据可比的 7B 模型 |
| 2025-04 | 🆕 [CLIMB](https://arxiv.org/abs/2504.13161) `arXiv:2504.13161` | NVIDIA | 聚类 + 迭代搜索自动寻找预训练数据混合比例 |
| 2025-04 | 🆕 [DataDecide](https://arxiv.org/abs/2504.11393) `arXiv:2504.11393` | Ai2 | 用小规模实验预测哪种数据在大规模最好：单一规模的排序已相当可靠 |
| 2025-03 | 🆕 [SuperBPE](https://arxiv.org/abs/2503.13423) `arXiv:2503.13423` | UW/Ai2 | 允许跨空格的"超词"token，同等词表下编码效率与下游表现提升 |
| 2025-02 | 🆕 [WebOrganizer](https://arxiv.org/abs/2502.10341) `arXiv:2502.10341` | Princeton/Ai2 | 按主题与格式组织网页语料，再优化领域混合 |
| 2025-01 | 🆕 [开放数据集最佳实践](https://arxiv.org/abs/2501.08365) `arXiv:2501.08365` | Mozilla/EleutherAI | 构建开放许可 LLM 训练数据集的实践指南 |
| 2024-12 | 🆕 [ModernBERT](https://arxiv.org/abs/2412.13663) `arXiv:2412.13663` | Answer.AI/LightOn | 用现代架构与 2T token 重新训练的 BERT 式编码器，8K 上下文 |
| 2024-12 | ★ 🆕 [BLT](https://arxiv.org/abs/2412.09871) `arXiv:2412.09871` | Meta | 按下一字节熵动态切块的字节级模型，首次在规模上匹敌基于 tokenizer 的模型 |
| 2024-12 | 🆕 [Large Concept Models](https://arxiv.org/abs/2412.08821) `arXiv:2412.08821` | Meta | 在句子嵌入空间（SONAR）中自回归建模"概念"而非 token |
| 2024-12 | 🆕 [Nemotron-CC](https://arxiv.org/abs/2412.02595) `arXiv:2412.02595` | NVIDIA | 分类器集成 + 合成改写，把 Common Crawl 变成 6.3T 高质量长周期预训练数据 |
| 2024-11 | 🆕 [RedPajama](https://arxiv.org/abs/2411.12372) `arXiv:2411.12372` | Together 等 | 开放的大规模 LLM 训练数据集（V1/V2）与质量信号 |
| 2024-06 | ★ [FineWeb](https://arxiv.org/abs/2406.17557) `arXiv:2406.17557` | Hugging Face | 15T token 开放网页数据集与消融，FineWeb-Edu 用 LLM 打分筛选教育内容 |
| 2024-06 | [DCLM](https://arxiv.org/abs/2406.11794) `arXiv:2406.11794` | UW/Apple/TRI 等 | 以数据为变量的标准化竞赛：基于模型的质量过滤是关键 |
| 2024-04 | [多 token 预测](https://arxiv.org/abs/2404.19737) `arXiv:2404.19737` | Meta | 每个位置同时预测后 n 个 token，提升样本效率与推理速度 |
| 2024-01 | [Dolma](https://arxiv.org/abs/2402.00159) `arXiv:2402.00159` | Ai2 | 3T token 开放语料与工具链，配套 OLMo |
| 2023-06 | [Phi-1（教科书数据）](https://arxiv.org/abs/2306.11644) `arXiv:2306.11644` | Microsoft | "教科书质量"的合成与筛选数据让 1.3B 模型写出强代码 |
| 2023-06 | [RefinedWeb](https://arxiv.org/abs/2306.01116) `arXiv:2306.01116` | TII | 仅靠严格过滤与去重的网页数据即可训练出强模型 |
| 2023-05 | [模型崩溃](https://arxiv.org/abs/2305.17493) `arXiv:2305.17493` | Oxford/Cambridge | 递归地用生成数据训练会遗忘分布尾部，最终崩溃（Nature 2024） |
| 2023-05 | [MegaByte](https://arxiv.org/abs/2305.07185) `arXiv:2305.07185` | Meta | 分块的多尺度 Transformer 直接建模字节序列 |
| 2020-12 | [The Pile](https://arxiv.org/abs/2101.00027) `arXiv:2101.00027` | EleutherAI | 825GB 多源开放预训练语料 |
| 2020-05 | ★ [GPT-3](https://arxiv.org/abs/2005.14165) `arXiv:2005.14165` | OpenAI | 175B 参数的少样本上下文学习：规模本身带来新能力 |
| 2019-10 | [T5](https://arxiv.org/abs/1910.10683) `arXiv:1910.10683` | Google | 把所有 NLP 任务统一为文本到文本，系统比较预训练目标与数据（C4） |
| 2019-07 | [RoBERTa](https://arxiv.org/abs/1907.11692) `arXiv:1907.11692` | Facebook | 更久、更多数据、去掉 NSP：BERT 的鲁棒训练配方 |
| 2019-02 | [GPT-2](https://openai.com/index/better-language-models/) | OpenAI | 1.5B 语言模型展示零样本多任务能力："语言模型是无监督多任务学习器" |
| 2018-08 | [SentencePiece](https://arxiv.org/abs/1808.06226) `arXiv:1808.06226` | Google | 与语言无关的子词分词器 |
| 2018-06 | [GPT-1](https://openai.com/index/language-unsupervised/) | OpenAI | 生成式预训练 + 判别式微调：解码器 Transformer 的语言模型预训练 |
| 2018-02 | [ELMo](https://arxiv.org/abs/1802.05365) `arXiv:1802.05365` | Ai2/UW | 双向 LSTM 语言模型的上下文词向量 |
| 2015-08 | [BPE 子词](https://arxiv.org/abs/1508.07909) `arXiv:1508.07909` | Edinburgh | 字节对编码的子词切分，解决开放词表问题，至今仍是主流分词 |

### 后训练与对齐（62）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Waypoint-OPD](https://arxiv.org/abs/2610.21601) `arXiv:2610.21601` |  | 用门控在策略蒸馏迁移 RL 引起的策略偏移 |
| 2026-10 | 🔥 [在策略蒸馏为何有时失败](https://arxiv.org/abs/2610.11247) `arXiv:2610.11247` |  | 学习信号消失 |
| 2026-10 | 🔥 [学习而不覆盖](https://arxiv.org/abs/2610.05200) `arXiv:2610.05200` |  | 自蒸馏与 SFT 的理论 |
| 2026-09 | 🔥 [理解离策略与在策略蒸馏](https://arxiv.org/abs/2609.38666) `arXiv:2609.38666` |  | 两者对应不同的训练目标 |
| 2026-09 | 🔥 [教师是方向而非终点](https://arxiv.org/abs/2609.36484) `arXiv:2609.36484` |  | 外推 RL 带来的表征残差，在蒸馏中超越教师 |
| 2026-09 | 🔥 [用稀疏交叉编码器理解 OPD](https://arxiv.org/abs/2609.35210) `arXiv:2609.35210` |  | 在策略蒸馏到底把哪些特征蒸馏进了学生的内部表征 |
| 2026-09 | 🔥 [在策略还是离策略蒸馏](https://arxiv.org/abs/2609.35259) `arXiv:2609.35259` | UC 系 | 系统比较两种蒸馏的训练动力学 |
| 2026-09 | 🔥 [RLVR 的几何](https://arxiv.org/abs/2609.34344) `arXiv:2609.34344` |  | 用向量引导识别 RLVR 训练轨迹中的低维有效子空间 |
| 2026-09 | 🔥 [蒸馏中的长度膨胀](https://arxiv.org/abs/2609.20511) `arXiv:2609.20511` | Microsoft | 教师与学生 EOS 判断不一致导致学生回答越来越长 |
| 2026-09 | 🔥 [在策略蒸馏 II：单样本](https://arxiv.org/abs/2609.04172) `arXiv:2609.04172` | Thinking Space | 只用一个训练样本的在策略蒸馏 |
| 2026-08 | 🔥 [在策略蒸馏真的在蒸馏吗](https://arxiv.org/abs/2608.31046) `arXiv:2608.31046` | Purdue | 从噪声教师到自我改进：OPD 的收益来源 |
| 2026-08 | 🔥 [理解进化策略做推理](https://arxiv.org/abs/2608.27351) `arXiv:2608.27351` |  | 进化策略（ES）微调 LLM 相比 GRPO 推理覆盖更广：优化行为分析 |
| 2026-08 | 🔥 [Macaron-V1](https://arxiv.org/abs/2608.09819) `arXiv:2608.09819` | Mind Lab | 自我改进 + LoRA 混合的开放持续学习 |
| 2026-07 | 🔥 [直接在策略蒸馏实现弱到强](https://arxiv.org/abs/2607.05394) `arXiv:2607.05394` | ByteDance/Tsinghua | 用弱教师的在策略蒸馏激发强学生 |
| 2026-06 | 🔥 [可预测的 GRPO](https://arxiv.org/abs/2606.30789) `arXiv:2606.30789` |  | GRPO 训练动力学的闭式模型 |
| 2026-06 | 🔥 [稠密监督、稀疏更新](https://arxiv.org/abs/2606.13657) `arXiv:2606.13657` |  | 在策略蒸馏只改动很稀疏的参数子集：OPD 更新的稀疏性与几何 |
| 2026-06 | 🔥 [在策略蒸馏的几何](https://arxiv.org/abs/2606.07082) `arXiv:2606.07082` | HKUST | 从几何视角分析在策略蒸馏 |
| 2026-06 | 🔥 [局部冲突、定向修复](https://arxiv.org/abs/2606.02398) `arXiv:2606.02398` |  | 用局部扰动分析解释多领域 RL 后训练中的跨领域干扰与恢复 |
| 2026-06 | 🔥 [PEFT 的扩展](https://arxiv.org/abs/2606.02437) `arXiv:2606.02437` | Mind Lab | 面向百万个人化模型的参数高效微调扩展性 |
| 2026-05 | 🔥 [信任函数](https://arxiv.org/abs/2606.01000) `arXiv:2606.01000` |  | 把弱到强泛化看作数据选择：学习何时信任弱教师，接近无损 |
| 2026-05 | 🔥 [LLM 蒸馏中的桥—园困境](https://arxiv.org/abs/2605.26246) `arXiv:2605.26246` |  | 为何混合硬标签与软标签有效 |
| 2026-05 | 🔥 [神经奖励模型如何学习特征](https://arxiv.org/abs/2605.24749) `arXiv:2605.24749` |  | 单指标分析：奖励模型学到的特征如何服务策略优化 |
| 2026-05 | 🔥 [推理模型为何失去覆盖](https://arxiv.org/abs/2605.17026) `arXiv:2605.17026` |  | pass@1 提升而 pass@k 下降：数据与"岔路口" token 的作用 |
| 2026-05 | 🔥 [揭开在策略蒸馏](https://arxiv.org/abs/2605.10889) `arXiv:2605.10889` |  | OPD 何时有益、何时有害：教师选择与自蒸馏上下文的系统分析 |
| 2026-04 | ◆ [强化学习为何泛化](https://arxiv.org/abs/2604.25011) `arXiv:2604.25011` |  | 后训练的特征级机制研究 |
| 2026-04 | ◆ [输出多样性在哪里坍缩](https://arxiv.org/abs/2604.16027) `arXiv:2604.16027` |  | 后训练中多样性坍缩发生在哪一阶段、由什么导致 |
| 2026-04 | ◆ [微调为何助长幻觉](https://arxiv.org/abs/2604.15574) `arXiv:2604.15574` |  | SFT 注入新事实会增加对预训练知识的幻觉：成因与修正 |
| 2026-04 | ◆ [重新思考在策略蒸馏](https://arxiv.org/abs/2604.13016) `arXiv:2604.13016` | Tsinghua NLP | 在策略蒸馏的现象、机制与配方 |
| 2026-04 | ◆ [推理 SFT 的泛化是有条件的](https://arxiv.org/abs/2604.06628) `arXiv:2604.06628` |  | "SFT 记忆、RL 泛化"并不绝对：长思维链 SFT 的跨领域泛化取决于优化、数据与模型能力 |
| 2026-03 | ◆ [自蒸馏为何有时损害推理](https://arxiv.org/abs/2603.24472) `arXiv:2603.24472` |  | 自蒸馏压短回答的同时抑制了"认知言语化"（表达不确定性的步骤），从而降低数学推理 |
| 2026-02 | ◆ [RLVR 学习动力学中隐式课程的涌现](https://arxiv.org/abs/2602.14872) `arXiv:2602.14872` |  | RLVR 训练中自然出现由易到难的课程 |
| 2026-02 | ◆ [奖励外推的广义在策略蒸馏](https://arxiv.org/abs/2602.12125) `arXiv:2602.12125` | Tencent Hunyuan | 在蒸馏中外推奖励，让学生超过教师 |
| 2026-02 | ◆ [策略镜像下降的隐式正则](https://arxiv.org/abs/2602.05933) `arXiv:2602.05933` |  | Kimi 系列使用的策略镜像下降需估计配分函数，其近似带来隐式正则 |
| 2026-02 | ◆ [GRPO 的隐式优势对称](https://arxiv.org/abs/2602.05548) `arXiv:2602.05548` |  | 组相对优势的隐式对称性导致 GRPO 探索与难度自适应不足 |
| 2026-02 | ◆ [强化微调的熵动力学](https://arxiv.org/abs/2602.03392) `arXiv:2602.03392` |  | 推导强化微调中策略熵如何随更新变化 |
| 2026-02 | ◆ [RLVR 的相对预算理论](https://arxiv.org/abs/2602.01523) `arXiv:2602.01523` |  | RLVR 中采样预算与题目难度的关系 |
| 2026-01 | ◆ [虚假奖励悖论](https://arxiv.org/abs/2601.11061) `arXiv:2601.11061` |  | 虚假奖励下的 RLVR 激活了记忆捷径：机制分析出"困惑度悖论" |
| 2025-12 | ◆ [语言模型其实是单指标模型](https://arxiv.org/abs/2512.21917) `arXiv:2512.21917` |  | 半参数偏好优化：把奖励链接函数当未知，统一 DPO 类方法 |
| 2025-12 | ◆ [自底向上策略优化](https://arxiv.org/abs/2512.19673) `arXiv:2512.19673` |  | LLM 策略内部包含逐层的"内部策略"：分析其演化并据此自底向上优化 |
| 2025-12 | ◆ [探索与利用：重新理解 RLVR](https://arxiv.org/abs/2512.16912) `arXiv:2512.16912` |  | 虚假奖励与熵最小化这两种看似矛盾的机制如何都能提升推理：从裁剪、熵与虚假奖励分析 |
| 2025-10 | ★ 🆕 [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | Thinking Machines | 学生自己采样、教师逐 token 打分（反向 KL）：兼得 RL 的在策略与蒸馏的稠密信号，成本远低于 RL |
| 2025-09 | 🆕 [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) | Thinking Machines | 系统实验：在后训练与 RL 场景下合理配置的 LoRA 可以匹敌全参微调，RL 所需容量极小 |
| 2025-09 | 🆕 [RL's Razor](https://arxiv.org/abs/2509.04259) `arXiv:2509.04259` | MIT | 在线 RL 比 SFT 遗忘更少：RL 偏向与原模型 KL 最小的解 |
| 2025-08 | 🆕 [DuPO](https://arxiv.org/abs/2508.14460) `arXiv:2508.14460` | ByteDance Seed | 利用任务对偶性构造自监督奖励的偏好优化 |
| 2025-07 | 🆕 [Skywork-Reward-V2](https://arxiv.org/abs/2507.01352) `arXiv:2507.01352` | Skywork | 人机协同筛选 4000 万偏好对，训练开放奖励模型 |
| 2025-05 | 🆕 [RL 只更新小子网络](https://arxiv.org/abs/2505.11711) `arXiv:2505.11711` | Illinois | RL 微调只稀疏地更新 5%–30% 的参数，且不同种子、数据下子网络高度重合 |
| 2025-02 | 🆕 [直接对齐算法差别不大](https://arxiv.org/abs/2502.01237) `arXiv:2502.01237` | T-Tech | DPO、SimPO 等直接对齐算法的差异主要在于排序 vs 点式目标 |
| 2025-01 | 🆕 [SFT 记忆，RL 泛化](https://arxiv.org/abs/2501.17161) `arXiv:2501.17161` | HKU/Google DeepMind/NYU | 对比实验：SFT 倾向记忆训练数据，RL（尤其结果奖励）能泛化到规则与视觉变体 |
| 2024-11 | ★ 🆕 [Tülu 3](https://arxiv.org/abs/2411.15124) `arXiv:2411.15124` | Ai2 | 完全开放的后训练配方：SFT + DPO + 可验证奖励的 RL（RLVR 一词由此流行） |
| 2024-05 | [SimPO](https://arxiv.org/abs/2405.14734) `arXiv:2405.14734` | Princeton | 以长度归一化的平均对数概率为隐式奖励，不需参考模型 |
| 2024-02 | [KTO](https://arxiv.org/abs/2402.01306) `arXiv:2402.01306` | Contextual AI/Stanford | 前景理论启发：只需"好/坏"二元信号的对齐 |
| 2023-10 | [IPO / ΨPO](https://arxiv.org/abs/2310.12036) `arXiv:2310.12036` | Google DeepMind | 统一偏好优化目标并指出 DPO 在确定性偏好下会过拟合 |
| 2023-06 | [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) `arXiv:2306.05685` | UC Berkeley 等 | 用强模型充当评委（MT-Bench、Chatbot Arena），分析位置与冗长偏置 |
| 2023-05 | ★ [DPO](https://arxiv.org/abs/2305.18290) `arXiv:2305.18290` | Stanford | 把 RLHF 目标的最优解闭式代入，直接用偏好对做分类式训练，免奖励模型与 RL |
| 2023-05 | [LIMA](https://arxiv.org/abs/2305.11206) `arXiv:2305.11206` | Meta | 1000 条精选样本即可对齐：知识主要来自预训练，对齐只是学习格式（表面对齐假说） |
| 2022-12 | [Self-Instruct](https://arxiv.org/abs/2212.10560) `arXiv:2212.10560` | UW | 让模型自己生成指令数据来做指令微调 |
| 2022-12 | ★ [Constitutional AI](https://arxiv.org/abs/2212.08073) `arXiv:2212.08073` | Anthropic | 用一组原则让 AI 自我批评与修改，并以 AI 反馈（RLAIF）训练无害性 |
| 2022-10 | [奖励模型过度优化](https://arxiv.org/abs/2210.10760) `arXiv:2210.10760` | OpenAI | 对代理奖励优化越多，真实奖励先升后降，且呈可预测的规律 |
| 2022-03 | ★ [InstructGPT](https://arxiv.org/abs/2203.02155) `arXiv:2203.02155` | OpenAI | SFT → 奖励模型 → PPO 三阶段，1.3B 对齐模型胜过 175B GPT-3，ChatGPT 的方法基础 |
| 2021-09 | [FLAN](https://arxiv.org/abs/2109.01652) `arXiv:2109.01652` | Google | 指令微调：在大量以指令描述的任务上微调，零样本泛化到新任务 |
| 2020-09 | [学习摘要（RLHF）](https://arxiv.org/abs/2009.01325) `arXiv:2009.01325` | OpenAI | 人类反馈强化学习用于语言模型摘要，奖励模型 + PPO 的完整流程 |
| 2017-06 | ★ [从人类偏好学习](https://arxiv.org/abs/1706.03741) `arXiv:1706.03741` | OpenAI/DeepMind | 用人类对轨迹片段的比较训练奖励模型再做 RL，RLHF 的起点 |

### 推理与测试时计算（119）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [是计算不是记忆](https://arxiv.org/abs/2610.25760) `arXiv:2610.25760` |  | LLM 智能体中思维链与状态追踪的关系 |
| 2026-10 | 🔥 [单层 Transformer 可证明学会带思维链的 kNN](https://arxiv.org/abs/2610.16798) `arXiv:2610.16798` |  | 上下文 k 近邻预测 |
| 2026-10 | 🔥 [TPPO](https://arxiv.org/abs/2610.16794) `arXiv:2610.16794` |  | 追踪价值的 PPO 用于语言模型推理 |
| 2026-10 | 🔥 [思考惯性](https://arxiv.org/abs/2610.11765) `arXiv:2610.11765` | Stanford | 被要求不思考时模型仍会继续思考 |
| 2026-10 | 🔥 [MiMo-V2.6](https://arxiv.org/abs/2610.11959) `arXiv:2610.11959` | Xiaomi | 扩展 RL 走向自我改进 |
| 2026-09 | 🔥 [PPO 中的价值扁平化](https://arxiv.org/abs/2609.18708) `arXiv:2609.18708` | Shanghai AI Lab | LLM 的 PPO 价值网络会"扁平化"，提出缓解方法 |
| 2026-07 | 🔥 [从预训练到后训练理解推理](https://arxiv.org/abs/2607.16097) `arXiv:2607.16097` |  | 把 RL 后训练与之前的预训练放在一起研究：预训练数据如何决定 RL 能学到什么 |
| 2026-07 | 🔥 [Ring-Zero](https://arxiv.org/abs/2607.12395) `arXiv:2607.12395` | RUC/inclusionAI | 把零 RL 扩展到万亿参数，观察推理能力的涌现 |
| 2026-06 | 🔥 [训练策略优化的幻象](https://arxiv.org/abs/2606.29526) `arXiv:2606.29526` |  | RL 真正应优化的是单调的推理策略，而非训练时的采样策略 |
| 2026-06 | 🔥 [可解模型中思维链状态追踪的学习动力学](https://arxiv.org/abs/2606.18164) `arXiv:2606.18164` |  | 状态追踪任务上思维链的学习动力学 |
| 2026-06 | 🔥 [MaxProof](https://arxiv.org/abs/2606.13473) `arXiv:2606.13473` | MiniMax | 生成式验证器 RL + 群体级测试时扩展的数学证明 |
| 2026-06 | 🔥 [用归一化流做潜在推理](https://arxiv.org/abs/2606.06447) `arXiv:2606.06447` |  | 文本思维链迫使计算串行、离散化；用归一化流在连续潜空间中推理 |
| 2026-05 | 🔥 [Transformer 可证明地学会内化思维链](https://arxiv.org/abs/2605.28600) `arXiv:2605.28600` |  | 显式思维链训练后内化为隐式计算的可证明过程 |
| 2026-05 | 🔥 [DelTA](https://arxiv.org/abs/2605.21467) `arXiv:2605.21467` |  | 判别式的 token 级信用分配 |
| 2026-05 | 🔥 [简单统一扩展达成奥赛金牌](https://arxiv.org/abs/2605.13301) `arXiv:2605.13301` |  | 用简单统一的扩展配方在多门奥赛达到金牌级推理 |
| 2026-05 | 🔥 [自回归思维链的在线学习理论](https://arxiv.org/abs/2605.06819) `arXiv:2605.06819` |  | 思维链推理的在线学习界 |
| 2026-05 | 🔥 [列表式策略优化](https://arxiv.org/abs/2605.06139) `arXiv:2605.06139` | Tencent Hunyuan | 把组式 RLVR 解释为在回答单纯形上的目标投影 |
| 2026-05 | 🔥 [LLM 能学会多深的推理](https://arxiv.org/abs/2605.06638) `arXiv:2605.06638` |  | 可控难度的合成任务表明：RL 能否学会长程推理取决于模型表达能力 |
| 2026-04 | ◆ [自蒸馏 RLVR](https://arxiv.org/abs/2604.03128) `arXiv:2604.03128` |  | 在 RLVR 中结合模型自身的蒸馏信号 |
| 2026-03 | ◆ [FIPO](https://arxiv.org/abs/2603.19835) `arXiv:2603.19835` | Qwen | 以未来 KL 影响加权的策略优化，引出更深的推理 |
| 2026-03 | ◆ [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) `arXiv:2603.19220` | NVIDIA | 级联 RL + 多领域在策略蒸馏的后训练 |
| 2026-03 | ◆ [思考以回忆](https://arxiv.org/abs/2603.09906) `arXiv:2603.09906` | Google | 推理过程帮助模型提取参数中的知识 |
| 2026-03 | ◆ [推理即压缩](https://arxiv.org/abs/2603.08462) `arXiv:2603.08462` |  | 用条件信息瓶颈把高效推理重述为有损压缩，统一各类预算控制方法 |
| 2026-02 | ◆ [经验式强化学习](https://arxiv.org/abs/2602.13949) `arXiv:2602.13949` | Microsoft | 把反思得到的经验纳入 RL 循环 |
| 2026-02 | ◆ [推理模型知道何时停止吗](https://arxiv.org/abs/2602.08354) `arXiv:2602.08354` | ByteDance | 模型内部隐含"何时该停"的信号，可用于缩短思考 |
| 2026-02 | ◆ [F-GRPO](https://arxiv.org/abs/2602.06717) `arXiv:2602.06717` | T-Tech | 防止策略只学"显而易见"而遗忘罕见正确解 |
| 2026-02 | ◆ [思维链没有全局计划](https://arxiv.org/abs/2602.02103) `arXiv:2602.02103` | Tencent | 探测 CoT 的潜在规划视野：模型只提前规划很短的距离 |
| 2026-01 | ◆ [Golden Goose](https://arxiv.org/abs/2601.22975) `arXiv:2601.22975` | NVIDIA | 把不可验证的网页文本改写为填空式多选题，合成无限 RLVR 任务 |
| 2026-01 | ◆ [奖励稀有解](https://arxiv.org/abs/2601.08763) `arXiv:2601.08763` | MIT | 按解法独特性加权的 RL，提升创造性解题 |
| 2026-01 | ◆ [PaCoRe](https://arxiv.org/abs/2601.05593) `arXiv:2601.05593` | StepFun | 并行协同推理，学习扩展测试时计算 |
| 2026-01 | ◆ [GDPO](https://arxiv.org/abs/2601.05242) `arXiv:2601.05242` | NVIDIA | 多奖励 RL 中按奖励分别做组归一化再合并，避免信号相互淹没 |
| 2025-12 | ◆ [动态大概念模型](https://arxiv.org/abs/2512.24617) `arXiv:2512.24617` | ByteDance Seed | 在自适应语义空间中做潜在推理的概念级语言模型 |
| 2025-12 | ◆ [Seed-Prover 1.5](https://arxiv.org/abs/2512.17260) `arXiv:2512.17260` | ByteDance Seed | 从经验中学习的大学级形式化定理证明 |
| 2025-12 | ◆ [原生并行推理器](https://arxiv.org/abs/2512.07461) `arXiv:2512.07461` | BIGAI | 自蒸馏 RL 学会并行推理 |
| 2025-12 | ◆ [稳定 LLM 强化学习：形式化与实践](https://arxiv.org/abs/2512.01374) `arXiv:2512.01374` | Qwen | 把 token 级目标看作序列级目标的一阶近似，据此解释重要性采样修正、裁剪与路由重放 |
| 2025-11 | ◆ [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) `arXiv:2511.22570` | DeepSeek | 训练证明验证器并以之为奖励，让模型自我验证地生成定理证明，IMO 2025 金牌级 |
| 2025-11 | ◆ [P1 物理奥赛](https://arxiv.org/abs/2511.13612) `arXiv:2511.13612` | Shanghai AI Lab | 用 RL 训练的物理推理模型在 IPhO 达到金牌水平 |
| 2025-11 | ◆ [Think-at-Hard](https://arxiv.org/abs/2511.08577) `arXiv:2511.08577` | Tsinghua | 只对困难 token 做选择性的潜在迭代 |
| 2025-11 | ◆ [Transformer 可证明学会可长度泛化的思维链](https://arxiv.org/abs/2511.07378) `arXiv:2511.07378` |  | 证明梯度训练的 Transformer 学到可长度泛化的思维链推理 |
| 2025-10 | 🆕 [Ring-1T：万亿思考模型 RL](https://arxiv.org/abs/2510.18855) `arXiv:2510.18855` | inclusionAI | 把 RL 扩展到 1T 参数思考模型的系统与算法（IcePop、C3PO++） |
| 2025-10 | 🆕 [BAPO](https://arxiv.org/abs/2510.18927) `arXiv:2510.18927` | Nex AGI | 平衡正负样本的裁剪，稳定离策略 RL |
| 2025-10 | 🆕 [用采样推理](https://arxiv.org/abs/2510.14901) `arXiv:2510.14901` | Harvard | 对基座模型做 MCMC 幂分布采样，不训练即可接近 RL 后的推理能力 |
| 2025-10 | ★ 🆕 [RL 算力扩展的艺术](https://arxiv.org/abs/2510.13786) `arXiv:2510.13786` | Meta | 40 万 GPU 小时的系统研究：RL 性能随算力呈 S 形曲线可预测，给出 ScaleRL 配方 |
| 2025-10 | 🆕 [QeRL](https://arxiv.org/abs/2510.11696) `arXiv:2510.11696` | NVIDIA | 量化带来的噪声反而增加探索熵，NVFP4 + LoRA 加速 RL |
| 2025-10 | 🆕 [Markovian Thinker](https://arxiv.org/abs/2510.06557) `arXiv:2510.06557` | Mila/Microsoft | 把思考切成固定长度块、只携带短状态，使推理长度与上下文长度解耦（线性算力） |
| 2025-10 | 🆕 [低概率 token 维持探索](https://arxiv.org/abs/2510.03222) `arXiv:2510.03222` | Tencent | RLVR 中低概率 token 是探索的关键，保护它们可避免熵塌缩 |
| 2025-09 | 🆕 [VCRL](https://arxiv.org/abs/2509.19803) `arXiv:2509.19803` |  | 按组内奖励方差挑选"难度合适"的样本做课程 |
| 2025-09 | 🆕 [在预训练数据上做 RL](https://arxiv.org/abs/2509.19249) `arXiv:2509.19249` | Tencent | 直接在预训练语料上用自监督奖励做强化学习，扩展 RL 的数据来源 |
| 2025-09 | 🆕 [FlowRL](https://arxiv.org/abs/2509.15207) `arXiv:2509.15207` |  | 匹配奖励分布而非最大化奖励，保持推理多样性 |
| 2025-09 | 🆕 [集体 RL 经验共享](https://arxiv.org/abs/2509.08721) `arXiv:2509.08721` | Gensyn | 去中心化节点之间共享 rollout 的集体后训练 |
| 2025-09 | 🆕 [Parallel-R1](https://arxiv.org/abs/2509.07980) `arXiv:2509.07980` | Tencent | 用 RL 训练并行思考 |
| 2025-09 | 🆕 [逆向工程推理](https://arxiv.org/abs/2509.06160) `arXiv:2509.06160` |  | 从好答案反推推理过程，用于开放式生成 |
| 2025-08 | 🆕 [TreePO](https://arxiv.org/abs/2508.17445) `arXiv:2508.17445` | ByteDance Seed | 树状分段采样共享前缀，提高 rollout 效率与优势估计 |
| 2025-08 | 🆕 [变分问题合成自博弈](https://arxiv.org/abs/2508.14029) `arXiv:2508.14029` |  | 从自己的正确解合成新问题，维持 pass@k 不塌缩 |
| 2025-08 | 🆕 [RL 推理的技巧还是陷阱](https://arxiv.org/abs/2508.08221) `arXiv:2508.08221` | Alibaba | 系统复现 RL 推理技巧，给出两项技巧的极简组合 |
| 2025-08 | 🆕 [R-Zero](https://arxiv.org/abs/2508.05004) `arXiv:2508.05004` | Tencent | 挑战者出题、求解者解题，零数据的自我进化推理 |
| 2025-08 | 🆕 [CoT 是海市蜃楼吗](https://arxiv.org/abs/2508.01191) `arXiv:2508.01191` | Arizona State | 从数据分布视角看，CoT 推理在分布外迅速失效 |
| 2025-07 | 🆕 [GSPO](https://arxiv.org/abs/2507.18071) `arXiv:2507.18071` | Qwen | 序列级重要性比率与裁剪，稳定 MoE 模型的大规模 RL（Qwen3 采用） |
| 2025-07 | 🆕 [潜意识线程](https://arxiv.org/abs/2507.16784) `arXiv:2507.16784` | MIT | 把推理组织成可修剪的线程树，突破上下文长度限制 |
| 2025-07 | ★ 🆕 [Gemini Deep Think IMO 金牌](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) | Google DeepMind | 通用语言模型用自然语言端到端在 4.5 小时内解出 6 题中的 5 题，首次达到 IMO 金牌线（官方评分） |
| 2025-07 | 🆕 [隐形的缰绳](https://arxiv.org/abs/2507.14843) `arXiv:2507.14843` | Stanford 等 | RLVR 难以跳出基座模型支撑集，主要做"保守的重加权" |
| 2025-07 | 🆕 [推理还是记忆](https://arxiv.org/abs/2507.10532) `arXiv:2507.10532` |  | 部分 RLVR 结果（如虚假奖励有效）源于基准数据污染 |
| 2025-07 | 🆕 [潜空间推理综述](https://arxiv.org/abs/2507.06203) `arXiv:2507.06203` |  | 在隐状态而非语言 token 中推理的方法综述 |
| 2025-07 | 🆕 [数学推理能否迁移](https://arxiv.org/abs/2507.00432) `arXiv:2507.00432` | CMU 等 | RL 训练的数学推理比 SFT 更好地迁移到其它领域 |
| 2025-06 | 🆕 [RLVR 隐式激励正确推理](https://arxiv.org/abs/2506.14245) `arXiv:2506.14245` | Microsoft | 用 CoT-Pass@k 指标说明 RLVR 确实提升了推理过程的正确性 |
| 2025-06 | ★ 🆕 [AlphaEvolve](https://arxiv.org/abs/2506.13131) `arXiv:2506.13131` | Google DeepMind | Gemini + 进化搜索 + 自动评估的编码智能体，发现更快的矩阵乘法算法并优化数据中心调度 |
| 2025-06 | 🆕 [虚假奖励](https://arxiv.org/abs/2506.10947) `arXiv:2506.10947` | UW/Ai2 | 随机甚至错误的奖励也能提升 Qwen2.5-Math：RLVR 的收益可能来自激发预训练中的行为 |
| 2025-06 | 🆕 [Multiverse](https://arxiv.org/abs/2506.09991) `arXiv:2506.09991` | CMU/NVIDIA | 模型自己决定何时并行分支与合并生成 |
| 2025-06 | 🆕 [思考的幻觉](https://arxiv.org/abs/2506.06941) `arXiv:2506.06941` | Apple | 可控难度的谜题上，推理模型在高复杂度处准确率崩溃且思考量反而下降（结论有争议） |
| 2025-06 | 🆕 [置信度即奖励](https://arxiv.org/abs/2506.06395) `arXiv:2506.06395` |  | 以模型自身置信度为奖励，少样本无标注 RL 微调 |
| 2025-06 | 🆕 [高熵少数 token](https://arxiv.org/abs/2506.01939) `arXiv:2506.01939` | Qwen/Tsinghua | 只有约 20% 的高熵"分叉"token 决定 RLVR 效果，只对它们算梯度更好 |
| 2025-05 | 🆕 [反思-重试-奖励](https://arxiv.org/abs/2505.24726) `arXiv:2505.24726` | Writer | 失败后先写自我反思再重试，只奖励反思带来的成功 |
| 2025-05 | 🆕 [ProRL](https://arxiv.org/abs/2505.24864) `arXiv:2505.24864` | NVIDIA | 长时间、稳定的 RL（KL 控制 + 参考模型重置）能扩展推理边界，反驳"RL 只是重排" |
| 2025-05 | 🆕 [AlphaOne](https://arxiv.org/abs/2505.24863) `arXiv:2505.24863` | UIUC/UC Berkeley | 在测试时调度"慢思考到快思考"的转换 |
| 2025-05 | 🆕 [RL 的熵机制](https://arxiv.org/abs/2505.22617) `arXiv:2505.22617` | Shanghai AI Lab/Tsinghua | 策略熵塌缩限制性能，熵变由动作概率与优势的协方差驱动，提出 Clip-Cov/KL-Cov |
| 2025-05 | 🆕 [AdaptThink](https://arxiv.org/abs/2505.13417) `arXiv:2505.13417` | Tsinghua | 让推理模型学会根据难度决定是否思考 |
| 2025-05 | 🆕 [超越"顿悟"](https://arxiv.org/abs/2505.10554) `arXiv:2505.10554` | NUS 等 | 显式对齐演绎、归纳、溯因三种元能力，而非依赖不可控的"aha" |
| 2025-05 | 🆕 [Absolute Zero](https://arxiv.org/abs/2505.03335) `arXiv:2505.03335` | Tsinghua | 零外部数据：模型自己出题、自己解题，用代码执行器验证的自我博弈推理 |
| 2025-05 | 🆕 [RM-R1](https://arxiv.org/abs/2505.02387) `arXiv:2505.02387` | UIUC | 把奖励建模当作推理任务，先生成评分标准再判断 |
| 2025-04 | 🆕 [单样本 RL](https://arxiv.org/abs/2504.20571) `arXiv:2504.20571` | UW/Microsoft | 只用 1 个训练样本做 RLVR，MATH500 提升 30+ 点 |
| 2025-04 | 🆕 [TTRL](https://arxiv.org/abs/2504.16084) `arXiv:2504.16084` | Tsinghua/Shanghai AI Lab | 测试时强化学习：用多数投票作伪标签在无标注测试数据上做 RL |
| 2025-04 | 🆕 [LUFFY：离策略指导](https://arxiv.org/abs/2504.14945) `arXiv:2504.14945` | Shanghai AI Lab | 在 RLVR 中混入强模型的离策略推理轨迹，突破自身能力上限 |
| 2025-04 | ★ 🆕 [RL 真的扩展了推理边界吗](https://arxiv.org/abs/2504.13837) `arXiv:2504.13837` | Tsinghua | 大 k 下 pass@k 基座模型反超 RLVR 模型：RL 主要提高采样效率而非创造新能力 |
| 2025-04 | 🆕 [Hogwild! Inference](https://arxiv.org/abs/2504.06261) `arXiv:2504.06261` | Yandex/IST Austria | 多个实例通过共享 KV 缓存并发协作生成 |
| 2025-04 | 🆕 [VAPO](https://arxiv.org/abs/2504.05118) `arXiv:2504.05118` | ByteDance Seed | 基于价值模型的 PPO 改进，长 CoT 推理上超过无价值方法 |
| 2025-04 | 🆕 [生成式奖励模型的推理时扩展](https://arxiv.org/abs/2504.02495) `arXiv:2504.02495` | DeepSeek/Tsinghua | 让奖励模型先生成原则与批评再打分（SPCT），奖励模型也可随推理算力扩展 |
| 2025-04 | 🆕 [DeepSeek-R1 思维学](https://arxiv.org/abs/2504.07128) `arXiv:2504.07128` | Mila/McGill | 系统分析 R1 推理链的结构、长度与"思考甜蜜点" |
| 2025-03 | 🆕 [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290) `arXiv:2503.24290` | StepFun/Tsinghua | 极简配方（原始 PPO + GAE、无 KL）即可在基座模型上扩展 R1-Zero 式训练 |
| 2025-03 | 🆕 [Dr. GRPO](https://arxiv.org/abs/2503.20783) `arXiv:2503.20783` | Sea AI/NUS | 剖析 R1-Zero 式训练：GRPO 的长度与标准差归一化带来偏差，基座模型本身已有"顿悟" |
| 2025-03 | 🆕 [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) `arXiv:2503.18892` | HKUST | 在 10 个不同基座上复现零 RL，发现"顿悟"并非普遍出现 |
| 2025-03 | 🆕 [高效推理综述](https://arxiv.org/abs/2503.16419) `arXiv:2503.16419` | Rice | 缓解过度思考的模型、输出与提示层方法 |
| 2025-03 | ★ 🆕 [DAPO](https://arxiv.org/abs/2503.14476) `arXiv:2503.14476` | ByteDance Seed/Tsinghua | 开源大规模 RL 系统：解耦裁剪、动态采样、token 级损失、超长惩罚 |
| 2025-02 | 🆕 [揭秘长思维链](https://arxiv.org/abs/2502.03373) `arXiv:2502.03373` | CMU/Tsinghua | 长 CoT 的出现条件：SFT 的作用、奖励塑形防止长度爆炸、可验证噪声数据 |
| 2025-02 | 🆕 [LIMO](https://arxiv.org/abs/2502.03387) `arXiv:2502.03387` | SJTU | 817 条精选样本激发复杂数学推理：推理能力已在预训练中，关键是好的认知模板 |
| 2025-02 | 🆕 [用大推理模型做竞赛编程](https://arxiv.org/abs/2502.06807) `arXiv:2502.06807` | OpenAI | 通用 RL 扩展的 o3 无需人工策略即在 IOI 达到金牌 |
| 2025-02 | 🆕 [PRIME：隐式过程奖励](https://arxiv.org/abs/2502.01456) `arXiv:2502.01456` | Tsinghua | 用结果标签训练的隐式 PRM 提供稠密奖励，在线更新 |
| 2025-01 | 🆕 [s1](https://arxiv.org/abs/2501.19393) `arXiv:2501.19393` | Stanford | 1000 条样本 SFT + "budget forcing" 控制思考长度即可实现测试时扩展 |
| 2025-01 | 🆕 [Kimi k1.5](https://arxiv.org/abs/2501.12599) `arXiv:2501.12599` | Moonshot AI | 长上下文 RL + 简洁的策略优化，不用 MCTS 与价值函数，并提出 long2short |
| 2025-01 | ★ 🆕 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) `arXiv:2501.12948` | DeepSeek | 纯 RL（R1-Zero）即可涌现长思维链与自我反思；开源权重与蒸馏小模型，引发 RLVR 热潮（Nature 2025） |
| 2025-01 | 🆕 [PRM 的经验教训](https://arxiv.org/abs/2501.07301) `arXiv:2501.07301` | Qwen | 蒙特卡洛估计的过程标签不可靠，LLM 评判 + 共识过滤更好 |
| 2025-01 | 🆕 [rStar-Math](https://arxiv.org/abs/2501.04519) `arXiv:2501.04519` | Microsoft Research Asia | 小模型 + MCTS + 过程偏好模型自我进化，7B 达到 o1-preview 级数学能力 |
| 2025-01 | 🆕 [REINFORCE++](https://arxiv.org/abs/2501.03262) `arXiv:2501.03262` |  | 用全局批次归一化优势的简化 REINFORCE，免价值网络且比 GRPO 更稳 |
| 2024-12 | 🆕 [过度思考](https://arxiv.org/abs/2412.21187) `arXiv:2412.21187` | Tencent/SJTU | o1 类模型在简单题上过度思考，提出效率指标与自训练缓解 |
| 2024-12 | 🆕 [o1 System Card](https://arxiv.org/abs/2412.16720) `arXiv:2412.16720` | OpenAI | o1 系列的安全评估：推理用于遵循安全规范，也带来新的风险面 |
| 2024-12 | 🆕 [审慎对齐](https://arxiv.org/abs/2412.16339) `arXiv:2412.16339` | OpenAI | 让推理模型在回答前显式回忆并推理安全规范 |
| 2024-12 | 🆕 [LLM 推理稳定吗](https://arxiv.org/abs/2412.13147) `arXiv:2412.13147` | Shanghai AI Lab | 提出 G-Pass@k，衡量多次采样下推理的稳定性 |
| 2024-12 | 🆕 [ProcessBench](https://arxiv.org/abs/2412.06559) `arXiv:2412.06559` | Qwen | 识别数学推理过程中最早错误步的基准 |
| 2024-12 | ★ 🆕 [Coconut](https://arxiv.org/abs/2412.06769) `arXiv:2412.06769` | Meta/UCSD | 在连续潜空间而非语言 token 中推理，可同时保留多个推理分支（类 BFS） |
| 2024-09 | ★ [OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/) | OpenAI | 用大规模 RL 训练长思维链，推理性能随训练与测试时计算平滑提升 |
| 2024-03 | [Quiet-STaR](https://arxiv.org/abs/2403.09629) `arXiv:2403.09629` | Stanford | 让模型在每个 token 前生成内部"思考"再预测 |
| 2024-02 | [思维链的表达力](https://arxiv.org/abs/2402.12875) `arXiv:2402.12875` | Stanford/Google | T 步思维链让常数深度 Transformer 突破 TC⁰，可解决本质上串行的问题 |
| 2024-02 | ★ [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300) `arXiv:2402.03300` | DeepSeek | 组相对策略优化：用同题多样本的组内均值作基线，省去价值网络 |
| 2023-05 | ★ [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) `arXiv:2305.20050` | OpenAI | 过程监督（逐步打分）优于结果监督，发布 PRM800K |
| 2023-05 | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) `arXiv:2305.10601` | Princeton/Google DeepMind | 把推理组织成树并搜索、回溯 |
| 2022-05 | [零样本 CoT](https://arxiv.org/abs/2205.11916) `arXiv:2205.11916` | Tokyo/Google | 一句 "Let's think step by step" 即可引出推理 |
| 2022-03 | [STaR](https://arxiv.org/abs/2203.14465) `arXiv:2203.14465` | Stanford/Google | 用模型自己生成且答案正确的推理链迭代微调：自举推理 |
| 2022-03 | [自洽性](https://arxiv.org/abs/2203.11171) `arXiv:2203.11171` | Google | 采样多条推理链后多数投票 |
| 2022-01 | ★ [思维链](https://arxiv.org/abs/2201.11903) `arXiv:2201.11903` | Google | 在示例中给出中间推理步骤，大模型的多步推理能力大幅提升 |
| 2021-10 | [GSM8K 与验证器](https://arxiv.org/abs/2110.14168) `arXiv:2110.14168` | OpenAI | 小学数学数据集；训练验证器给候选解打分 |

### 可解释性（102）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Transformer 学到两种竞争的排序算法](https://arxiv.org/abs/2610.26974) `arXiv:2610.26974` |  | 只有其中一种能长度泛化 |
| 2026-10 | 🔥 [词汇参考系](https://arxiv.org/abs/2610.22169) `arXiv:2610.22169` |  | 用词表（嵌入与反嵌入）作为"原生语义"来解释残差流中间状态 |
| 2026-10 | 🔥 [线性探针能证明线性表征吗](https://arxiv.org/abs/2610.22655) `arXiv:2610.22655` |  | 线性探针成功并不等于存在线性表征 |
| 2026-10 | 🔥 [SAE 表征的行为组合性](https://arxiv.org/abs/2610.21261) `arXiv:2610.21261` |  | SAE 特征线性解码，但下游效应不一定可加：提出可加性残差度量 |
| 2026-10 | 🔥 [SAE 中的表征重分配](https://arxiv.org/abs/2610.21240) `arXiv:2610.21240` |  | 保持语义的提示变换会让激活在不同 SAE 特征间重新分配：单个特征变化不可靠 |
| 2026-10 | 🔥 [学会读出](https://arxiv.org/abs/2610.21025) `arXiv:2610.21025` |  | 预训练中反嵌入（读出）矩阵本身仍在学习：隐状态可能早已编码答案但读不出来 |
| 2026-10 | 🔥 [SAE 的数据收益递减](https://arxiv.org/abs/2610.20680) `arXiv:2610.20680` |  | SAE 的功能指标很快饱和，但特征恢复质量不随之保证 |
| 2026-10 | 🔥 [层论分解](https://arxiv.org/abs/2610.17117) `arXiv:2610.17117` |  | 以复述对为局部见证构造胞腔层，用层拉普拉斯把表征分解为全局一致部分与局部部分 |
| 2026-10 | 🔥 [球与盒](https://arxiv.org/abs/2610.11744) `arXiv:2610.11744` |  | 叠加中计算的两种几何 |
| 2026-10 | 🔥 [U-Space](https://arxiv.org/abs/2610.09087) `arXiv:2610.09087` |  | 语言模型内部的不确定性在何时、为何出现 |
| 2026-09 | 🔥 [残差流中推理的几何](https://arxiv.org/abs/2609.37824) `arXiv:2609.37824` |  | 比较中间残差状态与最终状态，刻画模型何时"决定"输出 |
| 2026-09 | 🔥 [Transformer 停止思考得太早](https://arxiv.org/abs/2609.36585) `arXiv:2609.36585` |  | 答案在中间层过早定型，一个极小的 LoRA 即可修复 |
| 2026-09 | 🔥 [Transformer 能同时持有两个想法](https://arxiv.org/abs/2609.29845) `arXiv:2609.29845` |  | LLM 内部存在线性叠加的多重思路证据 |
| 2026-09 | 🔥 [线性表征假设需要群作用](https://arxiv.org/abs/2609.27158) `arXiv:2609.27158` |  | 不同的"等价"概念保留不同结构：讨论线性表征时必须明确对称群 |
| 2026-09 | 🔥 [线性表征假说综述](https://arxiv.org/abs/2609.22695) `arXiv:2609.22695` |  | 线性表征假说的证据、反例与理论 |
| 2026-09 | 🔥 [LLM 的信息几何](https://arxiv.org/abs/2609.11063) `arXiv:2609.11063` |  | 下一词分布的 Fisher–Rao 几何在模型间共享、由训练习得，并可用于精确控制 |
| 2026-09 | 🔥 [早编码、晚使用](https://arxiv.org/abs/2609.07139) `arXiv:2609.07139` |  | 属性在残差流某深度已可线性读出，但要到更深处才影响输出：可读≠可用 |
| 2026-09 | 🔥 [思维链表面之下](https://arxiv.org/abs/2609.04753) `arXiv:2609.04753` |  | 归纳、组合、演绎等推理操作在表征空间中的几何组织 |
| 2026-08 | 🔥 [叠加的数学理论](https://arxiv.org/abs/2608.27540) `arXiv:2608.27540` |  | 朝向特征叠加的严格数学理论 |
| 2026-08 | 🔥 [动态规划的表征几何](https://arxiv.org/abs/2608.25034) `arXiv:2608.25034` |  | 网络如何在表征中实现动态规划 |
| 2026-08 | 🔥 [鲁棒性为何减少叠加](https://arxiv.org/abs/2608.22155) `arXiv:2608.22155` |  | 对抗鲁棒性与特征叠加的关系 |
| 2026-08 | 🔥 [Mechanist](https://arxiv.org/abs/2608.12036) `arXiv:2608.12036` |  | 把 AI 作为科学仪器，自动化地发现智能的机制 |
| 2026-07 | 🔥 [文本模板 token 是隐式语义寄存器](https://arxiv.org/abs/2607.19139) `arXiv:2607.19139` |  | 文生图 DiT 中提示模板 token 充当语义寄存器：因果可解释性分析 |
| 2026-07 | 🔥 [记住的知识为何用不上](https://arxiv.org/abs/2607.08393) `arXiv:2607.08393` |  | "知道—使用鸿沟"：记忆与泛化之间存在准确率差与时间滞后，机制分析 |
| 2026-07 | 🔥 [归纳头在 n-gram 间插值](https://arxiv.org/abs/2607.02800) `arXiv:2607.02800` |  | 归纳头的行为可看作对 n-gram 统计的插值 |
| 2026-06 | 🔥 [抽象表征几何支持推断](https://arxiv.org/abs/2606.23345) `arXiv:2606.23345` |  | LLM 中支持推断的抽象几何 |
| 2026-06 | 🔥 [简单自编码器中稀疏与叠加对损失的影响](https://arxiv.org/abs/2606.18538) `arXiv:2606.18538` |  | 可解模型分析 |
| 2026-06 | 🔥 [压缩计算（大概）不是叠加中的计算](https://arxiv.org/abs/2606.14673) `arXiv:2606.14673` |  | 对"叠加中计算"玩具模型的再检验 |
| 2026-06 | 🔥 [SAE 的随机种子依赖](https://arxiv.org/abs/2606.12138) `arXiv:2606.12138` |  | 单个 SAE 特征跨种子不稳定，但它们张成的子空间可复现 |
| 2026-06 | 🔥 [激活引导的几何](https://arxiv.org/abs/2606.06735) `arXiv:2606.06735` |  | 把激活引导分解为角度与范数：范数同样携带概念信息 |
| 2026-06 | 🔥 [注意力回路何时形成](https://arxiv.org/abs/2606.02378) `arXiv:2606.02378` |  | 能力与注意力回路的发育轨迹 |
| 2026-05 | 🔥 [潜意识学习即 steering 向量蒸馏](https://arxiv.org/abs/2606.00995) `arXiv:2606.00995` |  | 把潜意识学习解释为隐式传递了 steering 向量 |
| 2026-05 | 🔥 [可辨识的 SAE](https://arxiv.org/abs/2605.31245) `arXiv:2605.31245` |  | 什么条件下稀疏自编码器的特征可辨识 |
| 2026-05 | 🔥 [加法的形状](https://arxiv.org/abs/2606.03645) `arXiv:2606.03645` |  | 多操作数加法中残差流的几何结构：以原始和为锚点的轨迹 |
| 2026-05 | 🔥 [玩具组合可解释模型揭示早期特征空间中的彩票](https://arxiv.org/abs/2605.17704) `arXiv:2605.17704` |  | 彩票在特征层面的体现 |
| 2026-05 | 🔥 [探测叠加中的表征流形](https://arxiv.org/abs/2605.18537) `arXiv:2605.18537` |  | 多维特征流形的叠加 |
| 2026-05 | 🔥 [SAE 基准可靠吗](https://arxiv.org/abs/2605.18229) `arXiv:2605.18229` |  | 稀疏自编码器评测的可靠性 |
| 2026-05 | 🔥 [规模决定表征几何是否为预测而组织](https://arxiv.org/abs/2605.17084) `arXiv:2605.17084` |  | 随规模变化的表征组织方式 |
| 2026-05 | 🔥 [LLM 在上下文学习中重组表征几何](https://arxiv.org/abs/2605.28854) `arXiv:2605.28854` |  | ICL 过程中的表征重组 |
| 2026-05 | 🔥 [流形的线性表征](https://arxiv.org/abs/2605.14013) `arXiv:2605.14013` |  | 用线性结构表示流形 |
| 2026-05 | 🔥 [WriteSAE](https://arxiv.org/abs/2605.12770) `arXiv:2605.12770` |  | 用于循环状态的稀疏自编码器 |
| 2026-05 | 🔥 [SAE 中的特征饥饿](https://arxiv.org/abs/2605.05341) `arXiv:2605.05341` |  | 几何不稳定导致特征饥饿 |
| 2026-04 | ◆ [SAE 能捕捉概念流形吗](https://arxiv.org/abs/2604.28119) `arXiv:2604.28119` |  | 稀疏自编码器与多维概念流形 |
| 2026-04 | ◆ [趋同进化：数字表征](https://arxiv.org/abs/2604.20817) `arXiv:2604.20817` |  | Transformer、线性 RNN、LSTM、词向量都学到周期性数字特征，但只有部分学到几何可分的结构 |
| 2026-04 | ◆ [SAE 的鲁棒性](https://arxiv.org/abs/2604.18756) `arXiv:2604.18756` |  | 把预训练 SAE 插入残差流后对对抗鲁棒性的影响 |
| 2026-04 | ◆ [Transformer 会自适应地使用深度吗](https://arxiv.org/abs/2604.12426) `arXiv:2604.12426` |  | 关系推理任务上的证据 |
| 2026-04 | ◆ [被引导的激活不是满射](https://arxiv.org/abs/2604.09839) `arXiv:2604.09839` |  | 激活引导得到的状态往往落在模型自然激活不会到达的区域 |
| 2026-04 | ◆ [有害回应使用独立机制](https://arxiv.org/abs/2604.09544) `arXiv:2604.09544` |  | 有害顺从由一组跨伤害类型共享的参数支持，可被定向剪除 |
| 2026-04 | ◆ [层级概念的线性表征](https://arxiv.org/abs/2604.07886) `arXiv:2604.07886` |  | 语言模型中层级概念的线性表示 |
| 2026-04 | ◆ [用 SAE 寻找信念几何](https://arxiv.org/abs/2604.02685) `arXiv:2604.02685` |  | 在激活中发现信念状态的几何 |
| 2026-03 | ◆ [叠加中网络表征的相似性](https://arxiv.org/abs/2604.00208) `arXiv:2604.00208` |  | 叠加对表征相似度度量的影响 |
| 2026-03 | ◆ [AR 与扩散 LM 的逐层表征](https://arxiv.org/abs/2603.07475) `arXiv:2603.07475` |  | 比较自回归与扩散语言模型各层的表征能力，并据此在推理时跳层 |
| 2026-02 | ◆ [SAE 的合理性检查](https://arxiv.org/abs/2602.14111) `arXiv:2602.14111` |  | 稀疏自编码器在多项指标上未必胜过随机基线 |
| 2026-02 | ◆ [数的表征几何](https://arxiv.org/abs/2602.06843) `arXiv:2602.06843` |  | 语言模型中数的几何 |
| 2026-02 | ◆ [谱叠加](https://arxiv.org/abs/2602.02224) `arXiv:2602.02224` |  | 特征几何的谱理论 |
| 2026-02 | ◆ [从方向到区域](https://arxiv.org/abs/2602.02464) `arXiv:2602.02464` |  | 用局部几何把激活分解为区域而非单一全局方向，捕捉非线性、多维概念 |
| 2026-02 | ◆ [为什么引导有效](https://arxiv.org/abs/2602.02343) `arXiv:2602.02343` |  | 把微调、LoRA、激活引导统一看作动态权重更新 |
| 2026-02 | ◆ [PolySAE](https://arxiv.org/abs/2602.01322) `arXiv:2602.01322` |  | 用多项式解码建模特征间的交互，突破 SAE 线性可加的假设 |
| 2026-01 | ◆ [注意力模式为何存在](https://arxiv.org/abs/2601.21709) `arXiv:2601.21709` |  | 用时间视角统一解释检索头、汇聚头、对角线等注意力模式 |
| 2026-01 | ◆ [机制性数据归因](https://arxiv.org/abs/2601.21996) `arXiv:2601.21996` |  | 用影响函数把可解释单元（如归纳头）追溯到具体训练样本 |
| 2026-01 | ◆ [上下文结构重塑表征几何](https://arxiv.org/abs/2601.22364) `arXiv:2601.22364` |  | 上下文如何改变语言模型的表征几何 |
| 2026-01 | ◆ [线性表征会在对话中剧烈变化](https://arxiv.org/abs/2601.20834) `arXiv:2601.20834` |  | "事实性"等线性表征方向上的读数会随对话上下文大幅漂移 |
| 2026-01 | ◆ [可操作的机制可解释性综述](https://arxiv.org/abs/2601.14004) `arXiv:2601.14004` |  | 定位、操控、改进：面向应用的机制可解释性 |
| 2026-01 | ◆ [LLM 在继续预训练中如何学习概念](https://arxiv.org/abs/2601.03570) `arXiv:2601.03570` |  | 概念在继续预训练中的获得、保持与遗忘 |
| 2026-01 | ◆ [涌现的内省意识](https://arxiv.org/abs/2601.01828) `arXiv:2601.01828` | Anthropic | 向模型激活注入概念后，前沿模型有时能察觉并报告：有限但真实的内省能力 |
| 2025-12 | ◆ [LLM 能预测自己的失败吗](https://arxiv.org/abs/2512.20578) `arXiv:2512.20578` |  | 通过内部电路实现自我觉察的失败预测 |
| 2025-12 | ◆ [可证明地从一般叠加中提取特征](https://arxiv.org/abs/2512.15987) `arXiv:2512.15987` |  | 一般叠加下的特征恢复算法与保证 |
| 2025-11 | ◆ [SAE 即主题模型](https://arxiv.org/abs/2511.16309) `arXiv:2511.16309` |  | 稀疏自编码器与主题模型的等价视角 |
| 2025-11 | ★ ◆ [权重稀疏 Transformer](https://arxiv.org/abs/2511.13653) `arXiv:2511.13653` | OpenAI | 训练权重极稀疏的模型，得到更小、更易理解的电路 |
| 2025-11 | ◆ [数据白化改进 SAE](https://arxiv.org/abs/2511.13981) `arXiv:2511.13981` |  | 白化激活再训练 SAE |
| 2025-10 | 🆕 [线性真值编码的涌现](https://arxiv.org/abs/2510.15804) `arXiv:2510.15804` |  | 语言模型中线性"真/假"方向如何在训练中涌现 |
| 2025-07 | 🆕 [人格向量](https://arxiv.org/abs/2507.21509) `arXiv:2507.21509` | Anthropic | 提取"邪恶""谄媚""幻觉"等人格特质的激活方向，用于监控与在训练中预防人格漂移 |
| 2025-06 | 🆕 [涌现失准的收敛线性表征](https://arxiv.org/abs/2506.11618) `arXiv:2506.11618` |  | 不同微调导致的涌现失准共享同一线性方向 |
| 2025-06 | 🆕 [SAE，又来了？](https://arxiv.org/abs/2506.04859) `arXiv:2506.04859` |  | 重新审视稀疏自编码器在表示学习中的作用 |
| 2025-05 | 🆕 [语言模型记住多少](https://arxiv.org/abs/2505.24832) `arXiv:2505.24832` | Meta/Google DeepMind/Cornell | GPT 类模型约有 3.6 bit/参数的记忆容量；容量填满后开始"顿悟"出泛化 |
| 2025-04 | 🆕 [MIB 基准](https://arxiv.org/abs/2504.13151) `arXiv:2504.13151` |  | 机制可解释性方法（电路定位、因果变量定位）的统一基准 |
| 2025-04 | 🆕 [OLMoTrace](https://arxiv.org/abs/2504.07096) `arXiv:2504.07096` | Ai2 | 实时把模型输出追溯到数万亿 token 的训练数据 |
| 2025-03 | 🆕 [电路追踪方法](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Anthropic | 替代模型 + 归因图的方法论，2025-05 开源工具 |
| 2025-03 | ★ 🆕 [大语言模型的生物学](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Anthropic | 用跨层转码器构建"归因图"，追踪 Claude 3.5 Haiku 中的多步推理、提前规划押韵、多语言共享概念等机制 |
| 2025-03 | 🆕 [解读推理特征](https://arxiv.org/abs/2503.18878) `arXiv:2503.18878` |  | 用 SAE 找到与推理（反思、不确定）相关的特征并操控 |
| 2025-02 | 🆕 [SAE 有用吗：稀疏探针案例](https://arxiv.org/abs/2502.16681) `arXiv:2502.16681` | MIT/独立 | 在数据稀缺、类别不平衡等困难设置下，SAE 探针并不稳定地优于简单基线 |
| 2025-02 | 🆕 [LLM-Microscope](https://arxiv.org/abs/2502.15007) `arXiv:2502.15007` |  | 标点等"无意义"token 承载了大量上下文记忆 |
| 2025-02 | 🆕 [特征流](https://arxiv.org/abs/2502.03032) `arXiv:2502.03032` | T-Tech | 跨层映射 SAE 特征的演化，用于解释与操控 |
| 2025-01 | 🆕 [机制可解释性开放问题](https://arxiv.org/abs/2501.16496) `arXiv:2501.16496` | Apollo/Anthropic/DeepMind 等 | 领域内多家机构合写：方法、应用与社会技术层面的开放问题 |
| 2024-12 | 🆕 [BatchTopK SAE](https://arxiv.org/abs/2412.06410) `arXiv:2412.06410` |  | 按批次而非样本选 top-k，提高稀疏自编码器重建 |
| 2024-08 | [Gemma Scope](https://arxiv.org/abs/2408.05147) `arXiv:2408.05147` | Google DeepMind | 为 Gemma 2 所有层开放 400+ 个 SAE |
| 2024-06 | [Transcoder](https://arxiv.org/abs/2406.11944) `arXiv:2406.11944` | 独立/MATS | 用稀疏的"转码器"近似 MLP 的输入输出映射，使电路分析可以穿过 MLP |
| 2024-06 | [SAE 的扩展与评估](https://arxiv.org/abs/2406.04093) `arXiv:2406.04093` | OpenAI | k-稀疏自编码器训练到 1600 万特征，给出 SAE 的扩展律与评价指标 |
| 2024-05 | ★ [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | Anthropic | 在 Claude 3 Sonnet 上提取数千万个特征，包括抽象、多语言与安全相关特征，并可用于操控行为 |
| 2023-11 | [线性表征假说](https://arxiv.org/abs/2311.03658) `arXiv:2311.03658` | Chicago | 形式化"概念是表征空间中的方向"，并给出因果内积 |
| 2023-10 | [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | Anthropic | 在单层 Transformer 上用 SAE 提取数千个单义特征 |
| 2023-10 | [表征工程](https://arxiv.org/abs/2310.01405) `arXiv:2310.01405` | CAIS 等 | 自上而下地读取与控制诚实性等高层概念的表征方向 |
| 2023-09 | [稀疏自编码器找可解释特征](https://arxiv.org/abs/2309.08600) `arXiv:2309.08600` | EleutherAI/独立 | 用稀疏字典学习把多义神经元分解为单义特征 |
| 2023-08 | [激活加法（steering）](https://arxiv.org/abs/2308.10248) `arXiv:2308.10248` | 独立/MATS | 把两个提示的激活差加到前向过程中即可操控模型行为 |
| 2022-11 | [IOI 电路](https://arxiv.org/abs/2211.00593) `arXiv:2211.00593` | Redwood/UC Berkeley | 在 GPT-2 小模型中完整逆向工程间接宾语识别电路（26 个头） |
| 2022-09 | ★ [归纳头](https://arxiv.org/abs/2209.11895) `arXiv:2209.11895` | Anthropic | 归纳头（复制前文模式的两头电路）可能是上下文学习的主要机制，与损失曲线上的相变同时出现 |
| 2022-09 | ★ [叠加的玩具模型](https://arxiv.org/abs/2209.10652) `arXiv:2209.10652` | Anthropic | 特征数多于维度时，网络用近正交方向"叠加"存储稀疏特征，这是神经元多义性的来源 |
| 2022-02 | [ROME](https://arxiv.org/abs/2202.05262) `arXiv:2202.05262` | MIT | 定位并直接编辑 GPT 中间层 MLP 里存储的事实关联 |
| 2021-12 | ★ [Transformer 电路的数学框架](https://transformer-circuits.pub/2021/framework/index.html) | Anthropic | 残差流视角：注意力头的 QK/OV 电路分解，理解小型注意力模型 |
| 2020-03 | [Zoom In: 电路](https://distill.pub/2020/circuits/zoom-in/) | OpenAI | "电路"研究纲领：特征是基本单元，特征通过权重连接成可理解的电路 |
| 2017-04 | [Network Dissection](https://arxiv.org/abs/1704.05796) `arXiv:1704.05796` | MIT | 把单个神经元与人类可理解的视觉概念对齐量化 |
| 2017-03 | [积分梯度](https://arxiv.org/abs/1703.01365) `arXiv:1703.01365` | Google | 满足公理的特征归因方法 |

### 效率：量化/压缩/系统（64）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [TWIG：权重即离散决策路径](https://arxiv.org/abs/2610.31931) `arXiv:2610.31931` |  | 把每个权重表示为一串共享的离散决策（符号、2 的幂次移位等）构成的树路径 |
| 2026-09 | 🔥 [用预测精度选择所需数值精度的扩展律](https://arxiv.org/abs/2610.00352) `arXiv:2610.00352` |  | 精度需求的扩展律 |
| 2026-09 | 🔥 [分块 KV 压缩的周期性弱点](https://arxiv.org/abs/2609.36322) `arXiv:2609.36322` | ByteDance Seed | 分块 KV 缓存压缩引入的相位敏感性 |
| 2026-09 | 🔥 [分离式量化](https://arxiv.org/abs/2609.26333) `arXiv:2609.26333` | IST Austria | 预填充与解码阶段分别采用不同量化 |
| 2026-09 | 🔥 [随机注意力](https://arxiv.org/abs/2609.03430) `arXiv:2609.03430` | Salesforce | 重新思考推理中的 KV 缓存淘汰 |
| 2026-09 | 🔥 [Gated DeltaNet 为何能扛住 4-bit](https://arxiv.org/abs/2609.04098) `arXiv:2609.04098` | Minima AI | 混合模型中递归部分的 NVFP4 W4A4 量化 |
| 2026-07 | 🔥 [彩票不是部署票](https://arxiv.org/abs/2607.27031) `arXiv:2607.27031` |  | 彩票子网络不等于可部署的稀疏网络 |
| 2026-07 | 🔥 [M+Adam](https://arxiv.org/abs/2607.10611) `arXiv:2607.10611` |  | 加性—乘性优化实现低精度训练 |
| 2026-06 | 🔥 [FP4 预训练的收缩偏差](https://arxiv.org/abs/2606.20381) `arXiv:2606.20381` |  | E2M1 格式的量化带来系统性收缩偏差：几何来源、系统影响与 UFP4 配方 |
| 2026-06 | 🔥 [可解注意力模型中 LoRA 的高维理论](https://arxiv.org/abs/2606.05899) `arXiv:2606.05899` |  | LoRA 微调的高维精确理论 |
| 2026-06 | 🔥 [KVarN](https://arxiv.org/abs/2606.03458) `arXiv:2606.03458` | Huawei | 方差归一化的 KV 缓存量化，缓解推理中误差累积 |
| 2026-05 | 🔥 [Domino](https://arxiv.org/abs/2605.29707) `arXiv:2605.29707` | SJTU | 把因果建模从自回归起草中解耦的推测解码 |
| 2026-05 | 🔥 [PEFT-Arena](https://arxiv.org/abs/2605.28819) `arXiv:2605.28819` |  | 从稳定性—可塑性两难评估参数高效微调：目标任务收益与预训练能力保持 |
| 2026-05 | 🔥 [MinT](https://arxiv.org/abs/2605.13779) `arXiv:2605.13779` | Mind Lab | 训练与服务数百万个 LLM（LoRA）的托管基础设施 |
| 2026-04 | ◆ [TriAttention](https://arxiv.org/abs/2604.04921) `arXiv:2604.04921` | NVIDIA | 三角函数式 KV 压缩，缓解长推理的显存瓶颈 |
| 2026-03 | ◆ [剪枝何时有效](https://arxiv.org/abs/2603.24652) `arXiv:2603.24652` |  | 用表征层级解释为何剪枝后判别任务尚可、生成任务却常失败 |
| 2026-03 | ◆ [优化器状态量化的动力学](https://arxiv.org/abs/2603.16731) `arXiv:2603.16731` |  | 预训练中量化优化器状态带来的状态陈旧与影响 |
| 2026-02 | ◆ [DualPath](https://arxiv.org/abs/2602.21548) `arXiv:2602.21548` | DeepSeek | 多轮智能体推理受 KV 缓存存储 I/O 限制，双路径打破存储带宽瓶颈 |
| 2026-02 | ◆ [注意力 logit 的秩感知谱界](https://arxiv.org/abs/2602.18851) `arXiv:2602.18851` |  | 为稳定的低精度训练给出注意力 logit 的界 |
| 2026-01 | ◆ [Quartet II](https://arxiv.org/abs/2601.22813) `arXiv:2601.22813` | IST Austria | 改进无偏梯度估计，NVFP4 精确预训练 |
| 2026-01 | ◆ [Jet-RL](https://arxiv.org/abs/2601.14243) `arXiv:2601.14243` |  | 训练与 rollout 统一精度的在策略 FP8 强化学习 |
| 2025-11 | ◆ [TWEO](https://arxiv.org/abs/2511.23225) `arXiv:2511.23225` |  | 训练时消除极端离群值，使 FP8 训练与量化更简单 |
| 2025-10 | 🆕 [INT vs FP](https://arxiv.org/abs/2510.25602) `arXiv:2510.25602` | ByteDance Seed | 细粒度低比特量化中整数格式与浮点格式的系统比较 |
| 2025-10 | 🆕 [AdaSPEC](https://arxiv.org/abs/2510.19779) `arXiv:2510.19779` | Georgia Tech | 选择性知识蒸馏训练推测解码草稿模型 |
| 2025-10 | 🆕 [核心注意力分离](https://arxiv.org/abs/2510.18121) `arXiv:2510.18121` | UCSD 等 | 把注意力核心计算拆到独立设备，平衡长上下文训练负载 |
| 2025-09 | 🆕 [NVFP4 预训练](https://arxiv.org/abs/2509.25149) `arXiv:2509.25149` | NVIDIA | 用 NVFP4 训练 12B 模型 10T token：随机 Hadamard 变换、2D 缩放与随机舍入 |
| 2025-09 | 🆕 [FP8 训练全配方](https://arxiv.org/abs/2509.22536) `arXiv:2509.22536` |  | 面向推理增强语言模型的完整 FP8 训练配方 |
| 2025-09 | 🆕 [FP4 训练设计空间](https://arxiv.org/abs/2509.17791) `arXiv:2509.17791` |  | 系统梳理 FP4 训练的格式、缩放与舍入选择 |
| 2025-07 | 🆕 [GPTQ 的几何](https://arxiv.org/abs/2507.18553) `arXiv:2507.18553` | IST Austria | GPTQ 等价于格上的 Babai 最近平面算法 |
| 2025-07 | 🆕 [SingLoRA](https://arxiv.org/abs/2507.05566) `arXiv:2507.05566` |  | 只用一个矩阵 $AA^\top$ 的低秩适配 |
| 2025-05 | 🆕 [FP4 All the Way](https://arxiv.org/abs/2505.19115) `arXiv:2505.19115` | Intel/Habana | 权重、激活、梯度全 FP4 的完全量化训练 |
| 2025-05 | 🆕 [Quartet](https://arxiv.org/abs/2505.14669) `arXiv:2505.14669` | IST Austria | 全 FP4（MXFP4）原生训练算法与精度扩展律，在 Blackwell 上可达最优 |
| 2025-05 | 🆕 [SageAttention3](https://arxiv.org/abs/2505.11594) `arXiv:2505.11594` | Tsinghua | FP4 微缩放注意力推理内核，并探索 8-bit 注意力训练 |
| 2025-04 | 🆕 [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) `arXiv:2504.12285` | Microsoft Research | 首个原生 1-bit 的开源 2B 模型，4T token 训练 |
| 2025-03 | 🆕 [EAGLE-3](https://arxiv.org/abs/2503.01840) `arXiv:2503.01840` | PKU/Microsoft/Waterloo | 放弃特征预测、融合多层特征并做训练时测试，推测解码的扩展律改善 |
| 2025-02 | 🆕 [LoRA 能装多少知识](https://arxiv.org/abs/2502.14502) `arXiv:2502.14502` |  | 用 LoRA 注入新知识的容量与对旧能力的损害 |
| 2025-02 | 🆕 [蒸馏扩展律](https://arxiv.org/abs/2502.08606) `arXiv:2502.08606` | Apple/Oxford | 给定算力在教师与学生之间如何分配：蒸馏只在特定条件下优于直接训练 |
| 2025-02 | 🆕 [μnit Scaling](https://arxiv.org/abs/2502.05967) `arXiv:2502.05967` | Databricks | 简单可扩展的 FP8 训练：单位缩放与参数化结合 |
| 2025-01 | 🆕 [FP4 训练 LLM](https://arxiv.org/abs/2501.17116) `arXiv:2501.17116` | Microsoft | 首个 FP4 训练框架：可微量化估计器与离群值钳制 |
| 2025-01 | 🆕 [FlashInfer](https://arxiv.org/abs/2501.01005) `arXiv:2501.01005` | UW/NVIDIA 等 | 可定制的注意力内核库与 JIT 编译，服务 LLM 推理 |
| 2024-11 | ★ 🆕 [精度扩展律](https://arxiv.org/abs/2411.04330) `arXiv:2411.04330` | Harvard/Stanford/MIT | 把训练与推理精度纳入扩展律：后训练量化的损害随训练数据增加而增大 |
| 2024-11 | 🆕 [BitNet a4.8](https://arxiv.org/abs/2411.04965) `arXiv:2411.04965` | Microsoft Research | 1-bit 权重配 4-bit 激活 |
| 2024-06 | [Mooncake](https://arxiv.org/abs/2407.00079) `arXiv:2407.00079` | Moonshot AI/Tsinghua | 以 KV 缓存为中心的分离式推理架构（Kimi 的生产系统） |
| 2024-04 | [SnapKV](https://arxiv.org/abs/2404.14469) `arXiv:2404.14469` | UIUC 等 | 用提示末尾的观察窗口挑选重要 KV，压缩长上下文缓存 |
| 2024-02 | ★ [BitNet b1.58](https://arxiv.org/abs/2402.17764) `arXiv:2402.17764` | Microsoft Research | 三值（−1/0/1）权重从头训练的 LLM，在 3B 以上追平全精度 |
| 2024-02 | [DoRA](https://arxiv.org/abs/2402.09353) `arXiv:2402.09353` | NVIDIA | 把权重分解为幅度与方向，只对方向做 LoRA |
| 2024-02 | [KIVI](https://arxiv.org/abs/2402.02750) `arXiv:2402.02750` | Rice 等 | 按通道量化 K、按 token 量化 V 的 2-bit KV 缓存 |
| 2024-01 | [EAGLE](https://arxiv.org/abs/2401.15077) `arXiv:2401.15077` | PKU/Microsoft/Waterloo | 在特征层面做推测，提高草稿接受率 |
| 2024-01 | [Medusa](https://arxiv.org/abs/2401.10774) `arXiv:2401.10774` | Princeton/Together | 多个解码头并行预测后续 token + 树注意力验证 |
| 2024-01 | [DistServe](https://arxiv.org/abs/2401.09670) `arXiv:2401.09670` | PKU/UCSD | 把 prefill 与 decode 拆到不同 GPU，分别优化首 token 与逐 token 延迟 |
| 2023-12 | [SGLang](https://arxiv.org/abs/2312.07104) `arXiv:2312.07104` | Stanford/UC Berkeley | RadixAttention 前缀复用 + 结构化生成的推理系统 |
| 2023-09 | ★ [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) `arXiv:2309.06180` | UC Berkeley | 借鉴虚拟内存分页管理 KV 缓存，吞吐提升 2–4 倍，开源推理引擎标准 |
| 2023-06 | [H2O](https://arxiv.org/abs/2306.14048) `arXiv:2306.14048` | UT Austin 等 | 只保留"重击"token 与最近 token 的 KV 缓存淘汰策略 |
| 2023-06 | [Wanda](https://arxiv.org/abs/2306.11695) `arXiv:2306.11695` | CMU/Meta | 按"权重 × 激活"剪枝，无需更新权重 |
| 2023-06 | [AWQ](https://arxiv.org/abs/2306.00978) `arXiv:2306.00978` | MIT | 按激活重要性保护 1% 显著权重通道的 4-bit 量化 |
| 2023-05 | [QLoRA](https://arxiv.org/abs/2305.14314) `arXiv:2305.14314` | UW | 4-bit NF4 量化底座 + LoRA，单卡微调 65B 模型 |
| 2023-02 | [推测采样](https://arxiv.org/abs/2302.01318) `arXiv:2302.01318` | DeepMind | 同期提出的推测采样，保证输出分布与大模型一致 |
| 2023-01 | [SparseGPT](https://arxiv.org/abs/2301.00774) `arXiv:2301.00774` | IST Austria | 一次性剪掉 GPT 级模型 50%+ 的权重，无需重训 |
| 2022-11 | ★ [推测解码](https://arxiv.org/abs/2211.17192) `arXiv:2211.17192` | Google | 小模型起草、大模型并行验证，无损加速自回归解码 |
| 2022-11 | [SmoothQuant](https://arxiv.org/abs/2211.10438) `arXiv:2211.10438` | MIT/NVIDIA | 把激活离群值的难度"平滑"迁移到权重，实现 W8A8 |
| 2022-10 | ★ [GPTQ](https://arxiv.org/abs/2210.17323) `arXiv:2210.17323` | IST Austria/ETH | 基于近似二阶信息的逐层后训练量化，3–4 bit 权重 |
| 2022-09 | [FP8 格式](https://arxiv.org/abs/2209.05433) `arXiv:2209.05433` | NVIDIA/Arm/Intel | E4M3/E5M2 两种 8-bit 浮点格式，训练与推理通用 |
| 2022-08 | [LLM.int8()](https://arxiv.org/abs/2208.07339) `arXiv:2208.07339` | UW/Meta | 发现大模型的离群特征维度，混合精度分解实现无损 8-bit 推理 |
| 2021-06 | ★ [LoRA](https://arxiv.org/abs/2106.09685) `arXiv:2106.09685` | Microsoft | 冻结原权重、只训练低秩增量矩阵，参数高效微调的事实标准 |

### 长上下文/记忆/持续学习（65）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [记忆税](https://arxiv.org/abs/2610.34073) `arXiv:2610.34073` |  | 拆解智能体记忆的净收益：记忆并不总是有益 |
| 2026-10 | 🔥 [用算法稳定性细粒度分析灾难性遗忘](https://arxiv.org/abs/2610.31551) `arXiv:2610.31551` |  | 以算法稳定性刻画遗忘 |
| 2026-10 | 🔥 [Cortex](https://arxiv.org/abs/2610.20399) `arXiv:2610.20399` | ByteDance Seed | 把上下文巩固进参数化记忆 |
| 2026-10 | 🔥 [通用测试时训练](https://arxiv.org/abs/2610.05484) `arXiv:2610.05484` |  | 把测试时训练推广为通用的适应框架 |
| 2026-10 | 🔥 [自生成反馈破坏测试时训练](https://arxiv.org/abs/2610.05076) `arXiv:2610.05076` |  | 模型从自己的输出学习时，每次更新都改变下一次生成：因果分解长程 TTT 的不稳定来源 |
| 2026-10 | 🔥 [局部支撑学习](https://arxiv.org/abs/2610.02126) `arXiv:2610.02126` |  | 把灾难性遗忘看成每个权重矩阵输入空间上的几何问题，导出自然的保留机制 |
| 2026-09 | 🔥 [持续学习的学习动力学](https://arxiv.org/abs/2609.33620) `arXiv:2609.33620` |  | 把数据归因、遗忘与可塑性统一为同一学习动力学视角 |
| 2026-09 | 🔥 [可塑性的 Bellman 最优方程](https://arxiv.org/abs/2609.10776) `arXiv:2609.10776` |  | 把可塑性纳入 Bellman 方程 |
| 2026-09 | 🔥 [持续学习机制可组合](https://arxiv.org/abs/2609.06986) `arXiv:2609.06986` | Johns Hopkins | 连续 SFT 学 100 个任务时，多种持续学习机制组合可显著缓解遗忘 |
| 2026-08 | 🔥 [快权重注意力持续学习](https://arxiv.org/abs/2608.27763) `arXiv:2608.27763` |  | 用快权重注意力实现持续学习 |
| 2026-07 | 🔥 [大规模记忆解码器](https://arxiv.org/abs/2607.27919) `arXiv:2607.27919` |  | 预训练的参数化长期记忆模块 |
| 2026-07 | 🔥 [Metis 记忆基础模型](https://arxiv.org/abs/2607.26760) `arXiv:2607.26760` | MemTensor | 专门负责记忆的基础模型 |
| 2026-07 | 🔥 [LongStraw](https://arxiv.org/abs/2607.14952) `arXiv:2607.14952` | Mind Lab | 固定 GPU 预算下超过 200 万 token 的长上下文 RL |
| 2026-06 | 🔥 [以下一 token 预测做测试时训练](https://arxiv.org/abs/2606.21803) `arXiv:2606.21803` |  | 用上下文上的下一 token 预测损失在推理时更新权重 |
| 2026-06 | 🔥 [遗忘即可达性坍缩](https://arxiv.org/abs/2606.06032) `arXiv:2606.06032` |  | 知识持久性的三层框架 |
| 2026-05 | 🔥 [灾难性遗忘的机制起源](https://arxiv.org/abs/2605.28860) `arXiv:2605.28860` |  | 为什么 RL 比 SFT 更能保留回路 |
| 2026-05 | 🔥 [测试时训练破坏安全护栏](https://arxiv.org/abs/2605.22984) `arXiv:2605.22984` |  | 测试时训练会削弱模型的安全对齐 |
| 2026-05 | 🔥 [δ-mem](https://arxiv.org/abs/2605.12357) `arXiv:2605.12357` |  | 面向 LLM 的高效在线记忆 |
| 2026-05 | 🔥 [几何冲突](https://arxiv.org/abs/2605.09608) `arXiv:2605.09608` |  | 持续后训练中，任务更新的几何冲突解释并控制遗忘与迁移 |
| 2026-05 | 🔥 [锐度感知预训练缓解遗忘](https://arxiv.org/abs/2605.02105) `arXiv:2605.02105` |  | SAM 式预训练让后续微调遗忘更少 |
| 2026-04 | ◆ [弹性 TTT 快速空间记忆](https://arxiv.org/abs/2604.07350) `arXiv:2604.07350` |  | 用测试时训练实现 3D 场景的快速空间记忆 |
| 2026-04 | ◆ [原地测试时训练](https://arxiv.org/abs/2604.06169) `arXiv:2604.06169` |  | 在预训练模型参数上原地做测试时更新 |
| 2026-03 | ◆ [结构抽象作为归纳偏置](https://arxiv.org/abs/2603.17198) `arXiv:2603.17198` |  | 非平稳语言建模（在线持续学习）中，学习结构抽象可减少干扰与遗忘 |
| 2026-03 | ◆ [GradMem](https://arxiv.org/abs/2603.13875) `arXiv:2603.13875` |  | 推理时用梯度下降把上下文写进记忆参数，代替巨大的 KV 缓存 |
| 2026-03 | ◆ [上下文通道容量](https://arxiv.org/abs/2603.07415) `arXiv:2603.07415` |  | 理解灾难性遗忘的信息论框架 |
| 2026-03 | ◆ [矩阵补全中的隐式偏置与可塑性丧失](https://arxiv.org/abs/2603.04703) `arXiv:2603.04703` |  | 深度促进低秩，同时导致可塑性丧失 |
| 2026-02 | ◆ [多头注意力是 MoE Transformer 遗忘之源](https://arxiv.org/abs/2602.12587) `arXiv:2602.12587` |  | MoE 中遗忘主要来自注意力 |
| 2026-02 | ◆ [子空间几何决定 LoRA 的遗忘](https://arxiv.org/abs/2603.02224) `arXiv:2603.02224` |  | 低秩适配中遗忘与子空间几何 |
| 2026-02 | ◆ [数据重放的可证明效果](https://arxiv.org/abs/2602.02767) `arXiv:2602.02767` |  | 从特征学习视角分析持续学习中数据重放的作用 |
| 2026-02 | ◆ [AI 与可塑性综述](https://arxiv.org/abs/2602.01215) `arXiv:2602.01215` |  | 神经网络可塑性研究综述 |
| 2026-01 | ◆ [进化策略导致 LLM 灾难性遗忘](https://arxiv.org/abs/2601.20861) `arXiv:2601.20861` |  | ES 微调的遗忘问题 |
| 2026-01 | ◆ [熵自适应微调](https://arxiv.org/abs/2601.02151) `arXiv:2601.02151` | BUPT | 解决"自信冲突"以缓解微调中的遗忘 |
| 2025-12 | ◆ [递归语言模型](https://arxiv.org/abs/2512.24601) `arXiv:2512.24601` | MIT | 让模型把长提示当作环境，以代码递归调用自身处理任意长输入 |
| 2025-12 | ★ ◆ [Nested Learning](https://arxiv.org/abs/2512.24695) `arXiv:2512.24695` | Google | 把模型看作多层嵌套、不同更新频率的优化问题，提出自修改的 HOPE 架构与连续记忆系统 |
| 2025-12 | ◆ [用任务排序缓解遗忘](https://arxiv.org/abs/2512.16871) `arXiv:2512.16871` |  | 持续学习中的任务顺序 |
| 2025-12 | ◆ [智能体时代的记忆](https://arxiv.org/abs/2512.13564) `arXiv:2512.13564` |  | 智能体记忆的形式、功能与动态综述 |
| 2025-12 | ◆ [QwenLong-L1.5](https://arxiv.org/abs/2512.12967) `arXiv:2512.12967` | Qwen | 长上下文推理与记忆管理的后训练配方 |
| 2025-11 | ◆ [SCALE 上扩展持续学习](https://arxiv.org/abs/2511.03270) `arXiv:2511.03270` |  | 通过上扩展结构的大模型持续学习 |
| 2025-10 | 🆕 [LoongRL](https://arxiv.org/abs/2510.19363) `arXiv:2510.19363` | Microsoft Research | 面向长上下文高级推理的 RL |
| 2025-10 | 🆕 [ACE 智能体上下文工程](https://arxiv.org/abs/2510.04618) `arXiv:2510.04618` | Stanford/SambaNova | 把上下文当作不断演化的"剧本"，通过生成-反思-整理增量更新，避免简化偏差与上下文崩塌 |
| 2025-09 | 🆕 [演化世界中的学习障碍](https://arxiv.org/abs/2510.00304) `arXiv:2510.00304` |  | 可塑性丧失的数学理解 |
| 2025-09 | 🆕 [TTT3R](https://arxiv.org/abs/2509.26645) `arXiv:2509.26645` |  | 把 3D 重建表述为测试时训练 |
| 2025-09 | 🆕 [谱坍缩导致可塑性丧失](https://arxiv.org/abs/2509.22335) `arXiv:2509.22335` |  | 表征谱坍缩是持续学习中可塑性丧失的驱动因素 |
| 2025-07 | 🆕 [Context Rot](https://research.trychroma.com/context-rot) | Chroma | 18 个模型的实验：随输入长度增加，性能即使在简单任务上也持续下降 |
| 2025-07 | 🆕 [MemOS](https://arxiv.org/abs/2507.03724) `arXiv:2507.03724` | MemTensor 等 | 把记忆当作可调度资源的"记忆操作系统" |
| 2025-06 | 🆕 [Cartridges](https://arxiv.org/abs/2506.06266) `arXiv:2506.06266` | Stanford | 离线用自学习把长语料压缩进小型 KV 缓存，显存降 38 倍 |
| 2025-05 | 🆕 [QwenLong-L1](https://arxiv.org/abs/2505.17667) `arXiv:2505.17667` | Qwen | 用渐进式上下文扩展的 RL 训练长上下文推理模型 |
| 2025-04 | 🆕 [Mem0](https://arxiv.org/abs/2504.19413) `arXiv:2504.19413` | Mem0 | 可扩展的生产级智能体长期记忆 |
| 2025-03 | 🆕 [经验回放缓解可塑性丧失](https://arxiv.org/abs/2503.20018) `arXiv:2503.20018` |  | 经验回放可解决持续学习中的可塑性丧失 |
| 2025-02 | 🆕 [长上下文 LLM 综述](https://arxiv.org/abs/2502.17129) `arXiv:2502.17129` | Fudan | 长上下文的架构、基础设施、训练与评测全景 |
| 2025-02 | 🆕 [NoLiMa](https://arxiv.org/abs/2502.05167) `arXiv:2502.05167` | Adobe/LMU | 去掉字面匹配后，多数模型在 32K 上下文时性能大幅下降 |
| 2025-01 | 🆕 [Qwen2.5-1M](https://arxiv.org/abs/2501.15383) `arXiv:2501.15383` | Qwen | 渐进式扩展到 100 万 token 上下文的训练与推理框架 |
| 2025-01 | 🆕 [虚假遗忘](https://arxiv.org/abs/2501.13453) `arXiv:2501.13453` |  | 持续学习中的"遗忘"常是任务对齐丢失而非知识丢失 |
| 2024-04 | [RULER](https://arxiv.org/abs/2404.06654) `arXiv:2404.06654` | NVIDIA | 超越"大海捞针"的合成长上下文基准，有效长度远小于宣称长度 |
| 2024-02 | [LongRoPE](https://arxiv.org/abs/2402.13753) `arXiv:2402.13753` | Microsoft Research | 搜索非均匀 RoPE 缩放，上下文扩到 2M |
| 2023-10 | [Ring Attention](https://arxiv.org/abs/2310.01889) `arXiv:2310.01889` | UC Berkeley | 把序列切块分布在环形设备上并重叠通信，近乎无限上下文 |
| 2023-08 | [YaRN](https://arxiv.org/abs/2309.00071) `arXiv:2309.00071` | Nous/EleutherAI | 分频段插值 RoPE + 注意力温度，高效扩展上下文窗口 |
| 2023-07 | [Lost in the Middle](https://arxiv.org/abs/2307.03172) `arXiv:2307.03172` | Stanford | 关键信息位于长上下文中间时模型表现最差（U 形曲线） |
| 2023-06 | [位置插值](https://arxiv.org/abs/2306.15595) `arXiv:2306.15595` | Meta | 把 RoPE 位置线性压缩到训练范围内，少量微调即可扩展上下文 |
| 2021-12 | [RETRO](https://arxiv.org/abs/2112.04426) `arXiv:2112.04426` | DeepMind | 从 2 万亿 token 数据库检索片段，7.5B 模型匹敌 25 倍大的模型 |
| 2020-05 | ★ [RAG](https://arxiv.org/abs/2005.11401) `arXiv:2005.11401` | Facebook/UCL | 检索增强生成：把参数化记忆与可检索的非参数化记忆结合 |
| 2020-04 | [Longformer](https://arxiv.org/abs/2004.05150) `arXiv:2004.05150` | Ai2 | 滑动窗口 + 全局 token 的稀疏注意力 |
| 2019-09 | [测试时训练（TTT）](https://arxiv.org/abs/1909.13231) `arXiv:1909.13231` | UC Berkeley | 测试时用自监督任务更新模型以应对分布偏移 |
| 2019-01 | [Transformer-XL](https://arxiv.org/abs/1901.02860) `arXiv:1901.02860` | CMU/Google | 段级递归 + 相对位置编码，突破固定长度上下文 |
| 2016-12 | [EWC](https://arxiv.org/abs/1612.00796) `arXiv:1612.00796` | DeepMind | 弹性权重巩固：按 Fisher 信息保护旧任务重要参数，缓解灾难性遗忘 |

### 多模态与语音（53）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-07 | 🔥 [原生多模态预训练的扩展](https://arxiv.org/abs/2607.22043) `arXiv:2607.22043` |  | 从零开始做原生多模态预训练（不依赖纯文本 LLM 初始化）的扩展性质 |
| 2026-05 | 🔥 [SenseNova-U1](https://arxiv.org/abs/2605.12500) `arXiv:2605.12500` | SenseTime | NEO-unify 架构统一多模态理解与生成 |
| 2026-03 | ◆ [LongCat-Next](https://arxiv.org/abs/2603.27538) `arXiv:2603.27538` | Meituan | 把各模态"词汇化"为离散 token 的统一下一 token 预测 |
| 2026-03 | ◆ [Penguin-VL](https://arxiv.org/abs/2603.06569) `arXiv:2603.06569` | Tencent | 用 LLM 初始化的视觉编码器，探索 VLM 的效率极限 |
| 2026-01 | ◆ [DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552) `arXiv:2601.20552` | DeepSeek | "视觉因果流"：按语义顺序重排视觉 token 的文档理解 |
| 2026-01 | ◆ [Qwen3-TTS](https://arxiv.org/abs/2601.15621) `arXiv:2601.15621` | Qwen | 新一代语音合成模型 |
| 2026-01 | ◆ [STEP3-VL-10B](https://arxiv.org/abs/2601.09668) `arXiv:2601.09668` | StepFun | 10B 规模的视觉语言模型技术报告 |
| 2026-01 | ◆ [BabyVision](https://arxiv.org/abs/2601.06521) `arXiv:2601.06521` |  | 脱离语言的基础视觉推理：前沿模型远不如幼儿 |
| 2025-11 | ◆ [Qwen3-VL](https://arxiv.org/abs/2511.21631) `arXiv:2511.21631` | Qwen | 原生 256K 交错上下文、DeepStack 多层视觉特征注入与文本时间戳对齐 |
| 2025-11 | ◆ [Uni-MoE-2.0-Omni](https://arxiv.org/abs/2511.12609) `arXiv:2511.12609` | Lychee/HIT | 以语言为中心的全模态 MoE 模型 |
| 2025-11 | ◆ [用视频思考](https://arxiv.org/abs/2511.04570) `arXiv:2511.04570` | OpenMOSS | 以视频生成作为多模态推理范式 |
| 2025-11 | ◆ [Cambrian-S](https://arxiv.org/abs/2511.04670) `arXiv:2511.04670` | NYU | "空间超感知"：视频理解需要预测性世界建模与事件分割，而非只扩上下文 |
| 2025-10 | 🆕 [LongCat-Flash-Omni](https://arxiv.org/abs/2511.00279) `arXiv:2511.00279` | Meituan | 560B 级开源全模态实时交互模型 |
| 2025-10 | 🆕 [ThinkMorph](https://arxiv.org/abs/2510.27492) `arXiv:2510.27492` |  | 交错图文思维链中的涌现性质 |
| 2025-10 | 🆕 [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) `arXiv:2510.18234` | DeepSeek | "上下文光学压缩"：把长文本渲染成图像，10 倍压缩下 OCR 精度约 97% |
| 2025-10 | 🆕 [Glyph](https://arxiv.org/abs/2510.17800) `arXiv:2510.17800` | Zhipu/Tsinghua | 把长文本渲染为图像，用视觉-文本压缩扩展上下文窗口 |
| 2025-10 | 🆕 [OmniVinci](https://arxiv.org/abs/2510.15870) `arXiv:2510.15870` | NVIDIA | 全模态理解 LLM 的架构与数据改进 |
| 2025-09 | 🆕 [Qwen3-Omni](https://arxiv.org/abs/2509.17765) `arXiv:2509.17765` | Qwen | 单一模型在文本、图像、音频、视频上都不弱于同规模单模态模型 |
| 2025-08 | 🆕 [VibeVoice](https://arxiv.org/abs/2508.19205) `arXiv:2508.19205` | Microsoft Research | 7.5Hz 超低帧率连续语音 tokenizer + next-token 扩散，生成 90 分钟多说话人播客 |
| 2025-08 | 🆕 [InternVL3.5](https://arxiv.org/abs/2508.18265) `arXiv:2508.18265` | Shanghai AI Lab | 级联 RL（离线 + 在线）与视觉分辨率路由，推理能力提升 |
| 2025-07 | 🆕 [GLM-4.1V-Thinking / GLM-4.5V](https://arxiv.org/abs/2507.01006) `arXiv:2507.01006` | Zhipu/Tsinghua | 课程采样的可扩展 RL 训练多模态推理模型 |
| 2025-06 | 🆕 [MiMo-VL](https://arxiv.org/abs/2506.03569) `arXiv:2506.03569` | Xiaomi | 混合在策略 RL 的 7B 视觉语言模型 |
| 2025-05 | 🆕 [BAGEL](https://arxiv.org/abs/2505.14683) `arXiv:2505.14683` | ByteDance Seed | 大规模交错数据预训练的统一模型，涌现自由形式编辑与世界导航能力 |
| 2025-05 | 🆕 [BLIP3-o](https://arxiv.org/abs/2505.09568) `arXiv:2505.09568` | Salesforce | 扩散 Transformer 生成 CLIP 特征的完全开放统一模型 |
| 2025-05 | 🆕 [MiniMax-Speech](https://arxiv.org/abs/2505.07916) `arXiv:2505.07916` | MiniMax | 可学习说话人编码器的零样本 TTS |
| 2025-05 | 🆕 [Seed1.5-VL](https://arxiv.org/abs/2505.07062) `arXiv:2505.07062` | ByteDance Seed | 532M 视觉编码器 + 20B 激活 MoE 的视觉语言模型，GUI 与游戏智能体能力突出 |
| 2025-04 | 🆕 [Kimi-Audio](https://arxiv.org/abs/2504.18425) `arXiv:2504.18425` | Moonshot AI | 音频理解、生成与对话的通用音频基础模型 |
| 2025-04 | 🆕 [Eagle 2.5](https://arxiv.org/abs/2504.15271) `arXiv:2504.15271` | NVIDIA | 长上下文视觉语言模型的后训练 |
| 2025-04 | 🆕 [InternVL3](https://arxiv.org/abs/2504.10479) `arXiv:2504.10479` | Shanghai AI Lab | 原生多模态预训练：视觉与文本在同一阶段联合训练 |
| 2025-04 | 🆕 [VLM-R1](https://arxiv.org/abs/2504.07615) `arXiv:2504.07615` | OmAI Lab | 稳定、可泛化的 R1 式视觉语言模型 |
| 2025-04 | 🆕 [Kimi-VL](https://arxiv.org/abs/2504.07491) `arXiv:2504.07491` | Moonshot AI | MoE 视觉语言模型与长思考变体，原生分辨率 MoonViT |
| 2025-04 | 🆕 [SmolVLM](https://arxiv.org/abs/2504.05299) `arXiv:2504.05299` | Hugging Face | 面向端侧的小型多模态模型设计 |
| 2025-03 | 🆕 [Video-R1](https://arxiv.org/abs/2503.21776) `arXiv:2503.21776` | CUHK | 把 R1 式 RL 用于视频推理 |
| 2025-03 | 🆕 [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) `arXiv:2503.20215` | Qwen | Thinker-Talker 架构的端到端全模态流式模型，TMRoPE 对齐音视频时间 |
| 2025-03 | 🆕 [Visual-RFT](https://arxiv.org/abs/2503.01785) `arXiv:2503.01785` | Shanghai AI Lab | 基于可验证视觉奖励的强化微调 |
| 2025-02 | 🆕 [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) `arXiv:2502.13923` | Qwen | 原生动态分辨率 ViT + 绝对时间编码，文档解析、定位与视觉智能体 |
| 2025-01 | 🆕 [Janus-Pro](https://arxiv.org/abs/2501.17811) `arXiv:2501.17811` | DeepSeek | 解耦理解与生成的视觉编码，统一多模态模型 |
| 2024-12 | 🆕 [FastVLM](https://arxiv.org/abs/2412.13303) `arXiv:2412.13303` | Apple | 混合视觉编码器 FastViTHD，大幅缩短高分辨率图像的首 token 延迟 |
| 2024-12 | 🆕 [PaliGemma 2](https://arxiv.org/abs/2412.03555) `arXiv:2412.03555` | Google | 多分辨率、多规模的可迁移视觉语言模型族 |
| 2024-10 | [GPT-4o System Card](https://arxiv.org/abs/2410.21276) `arXiv:2410.21276` | OpenAI | 端到端训练的全模态（文本、视觉、音频）模型，语音响应延迟接近人类 |
| 2024-09 | [Qwen2-VL](https://arxiv.org/abs/2409.12191) `arXiv:2409.12191` | Qwen | 动态分辨率 + 多模态 RoPE（M-RoPE） |
| 2024-09 | [Moshi](https://arxiv.org/abs/2410.00037) `arXiv:2410.00037` | Kyutai | 全双工语音对话：同时建模用户与系统两路音频流，延迟约 200ms |
| 2024-05 | [Chameleon](https://arxiv.org/abs/2405.09818) `arXiv:2405.09818` | Meta | 早期融合：图像与文本都离散成 token 的混合模态模型 |
| 2023-12 | [InternVL](https://arxiv.org/abs/2312.14238) `arXiv:2312.14238` | Shanghai AI Lab | 把视觉编码器扩展到 6B 并与 LLM 渐进对齐 |
| 2023-08 | [Qwen-VL](https://arxiv.org/abs/2308.12966) `arXiv:2308.12966` | Qwen | 支持定位与多图的视觉语言模型 |
| 2023-04 | ★ [LLaVA](https://arxiv.org/abs/2304.08485) `arXiv:2304.08485` | Wisconsin/Microsoft | 视觉指令微调：用 GPT-4 生成多模态指令数据，简单投影层连接 CLIP 与 LLM |
| 2023-01 | [BLIP-2](https://arxiv.org/abs/2301.12597) `arXiv:2301.12597` | Salesforce | Q-Former 桥接冻结的图像编码器与 LLM |
| 2023-01 | [VALL-E](https://arxiv.org/abs/2301.02111) `arXiv:2301.02111` | Microsoft | 把 TTS 视为编解码 token 上的语言建模，3 秒零样本克隆 |
| 2022-12 | ★ [Whisper](https://arxiv.org/abs/2212.04356) `arXiv:2212.04356` | OpenAI | 68 万小时弱监督多语言语音识别，鲁棒零样本迁移 |
| 2022-09 | [AudioLM](https://arxiv.org/abs/2209.03143) `arXiv:2209.03143` | Google | 语义 token + 声学 token 的层级语言建模生成音频 |
| 2022-04 | ★ [Flamingo](https://arxiv.org/abs/2204.14198) `arXiv:2204.14198` | DeepMind | 冻结的视觉编码器与 LLM 之间插入交叉注意力，少样本多模态学习 |
| 2020-06 | [wav2vec 2.0](https://arxiv.org/abs/2006.11477) `arXiv:2006.11477` | Facebook | 掩码 + 对比学习的语音自监督，极少标注即可做 ASR |
| 2016-09 | [WaveNet](https://arxiv.org/abs/1609.03499) `arXiv:1609.03499` | DeepMind | 扩张因果卷积逐样本生成原始音频 |

### 智能体与工具使用（82）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [Trial by Trace](https://arxiv.org/abs/2610.32285) `arXiv:2610.32285` | TraceLite | 评测智能体 harness 的可靠性与失败归因 |
| 2026-10 | 🔥 [智能体缓存工程](https://arxiv.org/abs/2610.28577) `arXiv:2610.28577` | Tsinghua | 面向智能体负载的缓存设计 |
| 2026-10 | 🔥 [真实网页环境中的深度搜索](https://arxiv.org/abs/2610.30181) `arXiv:2610.30181` |  | 在真实而非模拟网页环境中训练与评估深度搜索 |
| 2026-10 | 🔥 [BizAgentBench](https://arxiv.org/abs/2610.20424) `arXiv:2610.20424` | Ant International | 长程智能体在真实商业部署中的隐藏挑战 |
| 2026-10 | 🔥 [走进未知：开放科研问题基准](https://arxiv.org/abs/2610.19159) `arXiv:2610.19159` | CMU | 评测智能体解决开放式科研问题 |
| 2026-10 | 🔥 [终端智能体数据扩展](https://arxiv.org/abs/2610.13166) `arXiv:2610.13166` | Alibaba | 为长程终端任务扩展智能体数据 |
| 2026-09 | 🔥 [Raven：外壳的外壳](https://arxiv.org/abs/2609.33439) `arXiv:2609.33439` | EverMind | 可组合的智能体 harness 框架 |
| 2026-09 | 🔥 [RRSI](https://arxiv.org/abs/2609.24972) `arXiv:2609.24972` | Google | 正则化的递归自我改进智能体 harness |
| 2026-09 | 🔥 [CodeMidas](https://arxiv.org/abs/2609.22068) `arXiv:2609.22068` | Xiaomi | 从代码本身大规模构建编码智能体 RL 环境 |
| 2026-09 | 🔥 [HarnessDev](https://arxiv.org/abs/2609.01437) `arXiv:2609.01437` | ByteDance Seed | LLM 能否自己创建并进化智能体 harness |
| 2026-08 | 🔥 [用世界模型扩展自动研究智能体](https://arxiv.org/abs/2608.12564) `arXiv:2608.12564` | UIUC | 以世界模型预测实验结果，扩展自动研究 |
| 2026-07 | 🔥 [Qwen-UI-Agent](https://arxiv.org/abs/2607.28227) `arXiv:2607.28227` | Tongyi | 面向真实世界的基础 GUI 智能体 |
| 2026-07 | 🔥 [Harness Handbook](https://arxiv.org/abs/2607.13285) `arXiv:2607.13285` | Tencent Hunyuan | 让持续演化的智能体 harness 可读、可导航、可编辑 |
| 2026-06 | 🔥 [智能体弃权](https://arxiv.org/abs/2606.28733) `arXiv:2606.28733` |  | 智能体知道何时该停止而不是继续行动吗 |
| 2026-06 | 🔥 [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) `arXiv:2606.24597` | Qwen | 为通用智能体构建语言世界模型 |
| 2026-06 | 🔥 [SWE-Explore](https://arxiv.org/abs/2606.07297) `arXiv:2606.07297` | SJTU | 评测编码智能体如何探索代码仓库 |
| 2026-06 | 🔥 [Agents' Last Exam](https://arxiv.org/abs/2606.05405) `arXiv:2606.05405` | UC Berkeley | 面向智能体的高难度综合考试 |
| 2026-05 | 🔥 [SkillOpt](https://arxiv.org/abs/2605.23904) `arXiv:2605.23904` | Microsoft Research | 自我进化的智能体技能执行策略 |
| 2026-05 | 🔥 [代码即智能体外壳](https://arxiv.org/abs/2605.18747) `arXiv:2605.18747` |  | 用代码组织智能体的"harness"（外壳/脚手架） |
| 2026-04 | ◆ [递归多智能体系统](https://arxiv.org/abs/2604.25917) `arXiv:2604.25917` | Stanford | 递归组织的多智能体系统 |
| 2026-04 | ◆ [智能体世界建模](https://arxiv.org/abs/2604.22748) `arXiv:2604.22748` |  | 智能体世界模型的基础、能力与规律 |
| 2026-04 | ◆ [深入 Claude Code](https://arxiv.org/abs/2604.14228) `arXiv:2604.14228` |  | 当今与未来 AI 智能体系统的设计空间 |
| 2026-04 | ◆ [SkillClaw](https://arxiv.org/abs/2604.08377) `arXiv:2604.08377` |  | 技能在群体中集体进化 |
| 2026-04 | ◆ [ClawBench](https://arxiv.org/abs/2604.08523) `arXiv:2604.08523` |  | 智能体能否完成日常在线任务 |
| 2026-03 | ◆ [OpenSeeker](https://arxiv.org/abs/2603.15594) `arXiv:2603.15594` |  | 完全开源训练数据的前沿搜索智能体 |
| 2026-03 | ◆ [OpenClaw-RL](https://arxiv.org/abs/2603.10165) `arXiv:2603.10165` | Princeton | 通过对话训练任意智能体 |
| 2026-02 | ◆ [Agent Skills 数据分析](https://arxiv.org/abs/2602.08004) `arXiv:2602.08004` |  | 基于数据分析 Claude Skills 如何扩展 LLM 能力 |
| 2026-01 | ◆ [LLM 的智能体推理](https://arxiv.org/abs/2601.12538) `arXiv:2601.12538` | UIUC | 智能体推理的系统综述 |
| 2025-12 | ◆ [Step-GUI](https://arxiv.org/abs/2512.15431) `arXiv:2512.15431` | StepFun | GUI 智能体技术报告 |
| 2025-11 | ◆ [ToolOrchestra](https://arxiv.org/abs/2511.21689) `arXiv:2511.21689` | NVIDIA | 用 RL 训练小型编排模型调度大模型与工具 |
| 2025-11 | ◆ [多智能体潜空间协作](https://arxiv.org/abs/2511.20639) `arXiv:2511.20639` | Gen-Verse | 智能体之间直接交换隐状态而非文本 |
| 2025-11 | ◆ [代码智能实践指南](https://arxiv.org/abs/2511.18538) `arXiv:2511.18538` | Beihang 等 | 从代码基础模型到智能体与应用 |
| 2025-11 | ◆ [MiroThinker](https://arxiv.org/abs/2511.11793) `arXiv:2511.11793` | MiroMind | 把"交互深度"作为与模型、上下文并列的第三条扩展轴 |
| 2025-11 | ◆ [Lumine](https://arxiv.org/abs/2511.08892) `arXiv:2511.08892` | ByteDance Seed | 在 3D 开放世界游戏中构建通用智能体的开放配方 |
| 2025-10 | 🆕 [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) `arXiv:2510.24701` | Alibaba Tongyi | 从智能体中期训练到 RL 的端到端深度研究智能体 |
| 2025-10 | 🆕 [ReCode](https://arxiv.org/abs/2510.23564) `arXiv:2510.23564` | DeepWisdom | 用递归代码生成统一规划与行动 |
| 2025-10 | 🆕 [早期经验](https://arxiv.org/abs/2510.08558) `arXiv:2510.08558` | Meta | 智能体用自己行动产生的未来状态作监督（无奖励），介于模仿学习与 RL 之间 |
| 2025-09 | 🆕 [ScaleCUA](https://arxiv.org/abs/2509.15221) `arXiv:2509.15221` | Shanghai AI Lab | 跨平台数据扩展开源计算机使用智能体 |
| 2025-09 | 🆕 [WebSailor-V2](https://arxiv.org/abs/2509.13305) `arXiv:2509.13305` | Alibaba Tongyi | 合成数据与可扩展 RL 缩小与闭源智能体的差距 |
| 2025-09 | 🆕 [智能体 RL 综述](https://arxiv.org/abs/2509.02547) `arXiv:2509.02547` | Oxford/Shanghai AI Lab 等 | 从单步 MDP 到部分可观测 POMDP：LLM 智能体强化学习全景 |
| 2025-09 | 🆕 [UI-TARS-2](https://arxiv.org/abs/2509.02544) `arXiv:2509.02544` | ByteDance Seed | 多轮 RL 的数据飞轮 + 混合 GUI/游戏/终端环境 |
| 2025-08 | 🆕 [rStar2-Agent](https://arxiv.org/abs/2508.20722) `arXiv:2508.20722` | Microsoft Research | 智能体式代码工具推理 + GRPO-RoC，14B 达到前沿数学水平 |
| 2025-08 | 🆕 [Memento（AgentFly）](https://arxiv.org/abs/2508.16153) `arXiv:2508.16153` | UCL/Huawei | 基于记忆的在线 RL，不微调模型即可改进智能体 |
| 2025-08 | 🆕 [Chain-of-Agents](https://arxiv.org/abs/2508.13167) `arXiv:2508.13167` | OPPO | 把多智能体协作蒸馏进单个"智能体基础模型"再做 RL |
| 2025-08 | 🆕 [Agent Lightning](https://arxiv.org/abs/2508.03680) `arXiv:2508.03680` | Microsoft | 把任意智能体的执行与 RL 训练解耦的框架 |
| 2025-07 | 🆕 [ARPO](https://arxiv.org/abs/2507.19849) `arXiv:2507.19849` | RUC | 根据工具调用后的熵变化自适应分支 rollout 的智能体 RL |
| 2025-07 | 🆕 [上下文工程综述](https://arxiv.org/abs/2507.13334) `arXiv:2507.13334` | CAS 等 | 检索、处理、管理上下文的系统化综述 |
| 2025-07 | 🆕 [WebSailor](https://arxiv.org/abs/2507.02592) `arXiv:2507.02592` | Alibaba Tongyi | 高不确定性网页任务合成 + RL，开源网页智能体接近闭源 |
| 2025-06 | 🆕 [Skywork-SWE](https://arxiv.org/abs/2506.19290) `arXiv:2506.19290` | Skywork | 软件工程数据扩展律 |
| 2025-06 | 🆕 [τ²-bench](https://arxiv.org/abs/2506.07982) `arXiv:2506.07982` | Sierra | 双控制环境：用户与智能体都能操作共享世界 |
| 2025-05 | 🆕 [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) `arXiv:2505.22954` | Sakana/UBC | 智能体改写自己的代码并以基准验证，开放式地自我改进 |
| 2025-05 | 🆕 [SWE-rebench](https://arxiv.org/abs/2505.20411) `arXiv:2505.20411` | Nebius | 自动收集去污染的软件工程任务 |
| 2025-04 | 🆕 [SWE-smith](https://arxiv.org/abs/2504.21798) `arXiv:2504.21798` | Stanford/Princeton | 自动合成大规模软件工程训练任务 |
| 2025-04 | 🆕 [Paper2Code](https://arxiv.org/abs/2504.17192) `arXiv:2504.17192` | KAIST | 把机器学习论文自动转化为代码仓库 |
| 2025-04 | 🆕 [BrowseComp](https://arxiv.org/abs/2504.12516) `arXiv:2504.12516` | OpenAI | 需要持续深度浏览才能找到答案的难题 |
| 2025-04 | 🆕 [ReTool](https://arxiv.org/abs/2504.11536) `arXiv:2504.11536` | ByteDance Seed | RL 训练中交错执行代码，学会何时调用代码解释器 |
| 2025-04 | 🆕 [AI Scientist v2](https://arxiv.org/abs/2504.08066) `arXiv:2504.08066` | Sakana AI | 智能体树搜索的自动科研系统，首篇通过同行评审的完全 AI 生成论文 |
| 2025-04 | 🆕 [Multi-SWE-bench](https://arxiv.org/abs/2504.02605) `arXiv:2504.02605` | ByteDance | 多语言仓库级问题修复基准 |
| 2025-04 | 🆕 [PaperBench](https://arxiv.org/abs/2504.01848) `arXiv:2504.01848` | OpenAI | 从零复现 ICML 2024 论文的智能体评测 |
| 2025-03 | 🆕 [基础智能体综述](https://arxiv.org/abs/2504.01990) `arXiv:2504.01990` | MetaGPT 等 | 从类脑智能到进化、协作与安全的基础智能体全景 |
| 2025-03 | ★ 🆕 [任务时长度量](https://arxiv.org/abs/2503.14499) `arXiv:2503.14499` | METR | AI 能以 50% 成功率完成的任务长度约每 7 个月翻一番 |
| 2025-03 | 🆕 [Search-R1](https://arxiv.org/abs/2503.09516) `arXiv:2503.09516` | UIUC | 用 RL 训练模型在推理中自主调用搜索引擎 |
| 2025-02 | 🆕 [AI co-scientist](https://arxiv.org/abs/2502.18864) `arXiv:2502.18864` | Google | 多智能体"生成-辩论-进化"提出科学假设，部分经湿实验验证 |
| 2025-02 | 🆕 [SWE-RL](https://arxiv.org/abs/2502.18449) `arXiv:2502.18449` | Meta | 用开源软件演化数据与规则奖励做 RL，提升真实软件工程能力 |
| 2025-02 | 🆕 [MLGym](https://arxiv.org/abs/2502.14499) `arXiv:2502.14499` | Meta | 推进 AI 研究智能体的框架与基准 |
| 2025-02 | 🆕 [Deep Research](https://openai.com/index/introducing-deep-research/) | OpenAI | 以 RL 训练的多步网页研究智能体，生成带引用的报告 |
| 2025-01 | 🆕 [UI-TARS](https://arxiv.org/abs/2501.12326) `arXiv:2501.12326` | ByteDance | 只看截图的端到端 GUI 智能体，系统 2 推理与反思式在线轨迹 |
| 2024-11 | 🆕 [MCP](https://www.anthropic.com/news/model-context-protocol) | Anthropic | 模型上下文协议：连接模型与工具、数据源的开放标准，后被各家采用 |
| 2024-10 | [Claude Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use) | Anthropic | 首个公开的前沿模型"看屏幕、动鼠标键盘"的通用计算机操作能力 |
| 2024-10 | [MLE-bench](https://arxiv.org/abs/2410.07095) `arXiv:2410.07095` | OpenAI | 75 个 Kaggle 竞赛衡量机器学习工程能力 |
| 2024-07 | [OpenHands](https://arxiv.org/abs/2407.16741) `arXiv:2407.16741` | All Hands/CMU 等 | 开源通用软件开发智能体平台 |
| 2024-06 | [τ-bench](https://arxiv.org/abs/2406.12045) `arXiv:2406.12045` | Sierra | 工具 + 用户模拟的客服场景，衡量多轮一致性（pass^k） |
| 2024-05 | [SWE-agent](https://arxiv.org/abs/2405.15793) `arXiv:2405.15793` | Princeton | 为语言模型设计的智能体-计算机接口（ACI） |
| 2024-04 | [OSWorld](https://arxiv.org/abs/2404.07972) `arXiv:2404.07972` | HKU 等 | 真实操作系统中的开放式计算机使用任务基准 |
| 2023-11 | [GAIA](https://arxiv.org/abs/2311.12983) `arXiv:2311.12983` | Meta/Hugging Face | 对人类简单、对 AI 困难的通用助手问题 |
| 2023-10 | ★ [SWE-bench](https://arxiv.org/abs/2310.06770) `arXiv:2310.06770` | Princeton | 真实 GitHub issue 修复基准，衡量软件工程能力的事实标准 |
| 2023-07 | [WebArena](https://arxiv.org/abs/2307.13854) `arXiv:2307.13854` | CMU | 可复现的真实网站环境，评测自主网页智能体 |
| 2023-05 | [Voyager](https://arxiv.org/abs/2305.16291) `arXiv:2305.16291` | NVIDIA/Caltech | 在 Minecraft 中自动课程 + 可复用技能库的终身学习智能体 |
| 2023-04 | [生成式智能体](https://arxiv.org/abs/2304.03442) `arXiv:2304.03442` | Stanford/Google | 记忆、反思、规划架构，25 个智能体在沙盒小镇中模拟社会行为 |
| 2023-03 | [Reflexion](https://arxiv.org/abs/2303.11366) `arXiv:2303.11366` | Northeastern/MIT/Princeton | 用语言形式的自我反思作为强化信号 |
| 2023-02 | [Toolformer](https://arxiv.org/abs/2302.04761) `arXiv:2302.04761` | Meta | 模型自监督地学会何时、如何调用 API |
| 2022-10 | ★ [ReAct](https://arxiv.org/abs/2210.03629) `arXiv:2210.03629` | Princeton/Google | 推理与行动交替：思考、调用工具、观察，再思考 |

### 深度强化学习与世界模型（32）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [探索与可塑性丧失](https://arxiv.org/abs/2610.36309) `arXiv:2610.36309` |  | 在策略 RL 中探索与可塑性丧失的联系 |
| 2026-09 | 🔥 [等变强化学习的样本复杂度](https://arxiv.org/abs/2609.36421) `arXiv:2609.36421` |  | 对称性带来的样本效率 |
| 2026-09 | 🔥 [世界模型为规划必须区分什么](https://arxiv.org/abs/2609.33030) `arXiv:2609.33030` |  | 规划所需的最小区分 |
| 2026-08 | 🔥 [对比世界模型](https://arxiv.org/abs/2609.22175) `arXiv:2609.22175` |  | 用对比学习训练世界模型 |
| 2026-08 | 🔥 [WarpSAC](https://arxiv.org/abs/2608.24479) `arXiv:2608.24479` |  | 大规模并行仿真下重新设计离策略 RL 的探索与利用，扩展 SAC |
| 2026-08 | 🔥 [世界模型上的柏拉图表征假设](https://arxiv.org/abs/2608.23720) `arXiv:2608.23720` |  | 不同世界模型的表征是否趋同 |
| 2026-08 | 🔥 [带世界模型的 Q 学习](https://arxiv.org/abs/2608.17163) `arXiv:2608.17163` |  | 世界模型与 Q 学习的结合 |
| 2026-07 | 🔥 [受控世界模型的可辨识性](https://arxiv.org/abs/2607.22430) `arXiv:2607.22430` |  | 何时能辨识世界模型 |
| 2026-07 | 🔥 [世界模型表示了什么是三个问题](https://arxiv.org/abs/2607.06640) `arXiv:2607.06640` |  | 拆分世界模型表征问题 |
| 2026-05 | 🔥 [去偏的基于模型表征](https://arxiv.org/abs/2605.11711) `arXiv:2605.11711` |  | 用去偏的模型学习表征服务离策略 actor-critic，兼得无模型与有模型的优势 |
| 2026-03 | ◆ [重新思考深度 RL 中的可塑性](https://arxiv.org/abs/2603.21173) `arXiv:2603.21173` |  | 深度强化学习可塑性丧失的再审视 |
| 2026-03 | ◆ [通过正则化得到离散世界模型](https://arxiv.org/abs/2603.01748) `arXiv:2603.01748` |  | 离散潜状态世界模型 |
| 2026-02 | ◆ [跳跃式世界模型的组合规划](https://arxiv.org/abs/2602.19634) `arXiv:2602.19634` |  | 时间抽象的世界模型 |
| 2026-02 | ◆ [FLAC](https://arxiv.org/abs/2602.12829) `arXiv:2602.12829` |  | 生成式（流/扩散）策略的对数密度不可得：用动能正则的桥匹配实现无似然的最大熵 RL |
| 2026-01 | ◆ [流等变世界模型](https://arxiv.org/abs/2601.01075) `arXiv:2601.01075` |  | 自身运动与外部物体运动遵循光滑的时间参数化对称：以流等变性为部分可观测环境提供记忆 |
| 2025-12 | ◆ [理解与改进双曲深度 RL](https://arxiv.org/abs/2512.14202) `arXiv:2512.14202` |  | 双曲特征空间适合 RL 中的层级结构，分析其训练不稳定并加以改进 |
| 2025-11 | ◆ [AltNet](https://arxiv.org/abs/2512.01034) `arXiv:2512.01034` |  | 应对 RL 中的可塑性—稳定性两难 |
| 2025-09 | 🆕 [Dreamer 4](https://arxiv.org/abs/2509.24527) `arXiv:2509.24527` | Google DeepMind | 可扩展的世界模型内训练智能体，仅用离线数据在 Minecraft 中获得钻石 |
| 2025-01 | 🆕 [Cosmos](https://arxiv.org/abs/2501.03575) `arXiv:2501.03575` | NVIDIA | 面向物理 AI 的开放世界基础模型平台：视频 tokenizer、扩散与自回归世界模型 |
| 2024-08 | [GameNGen](https://arxiv.org/abs/2408.14837) `arXiv:2408.14837` | Google | 扩散模型实时运行 DOOM，神经网络本身就是游戏引擎 |
| 2024-02 | [Genie](https://arxiv.org/abs/2402.15391) `arXiv:2402.15391` | Google DeepMind | 从无标注的游戏视频中学出可交互的生成环境与潜动作 |
| 2023-01 | ★ [DreamerV3](https://arxiv.org/abs/2301.04104) `arXiv:2301.04104` | DeepMind | 一套固定超参数的世界模型 RL 适用于 150+ 任务，首次从零在 Minecraft 中挖到钻石（Nature 2025） |
| 2021-06 | [Decision Transformer](https://arxiv.org/abs/2106.01345) `arXiv:2106.01345` | UC Berkeley/Google | 把离线 RL 视为以回报为条件的序列建模 |
| 2019-11 | [MuZero](https://arxiv.org/abs/1911.08265) `arXiv:1911.08265` | DeepMind | 学习隐式环境模型做规划，不需要已知规则 |
| 2019-10 | [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) | DeepMind | 多智能体联盟训练，星际争霸 II 达到宗师级 |
| 2018-03 | [World Models](https://arxiv.org/abs/1803.10122) `arXiv:1803.10122` | Google Brain/IDSIA | VAE + RNN 学习环境的压缩时空模型，在"梦境"中训练策略 |
| 2018-01 | [SAC](https://arxiv.org/abs/1801.01290) `arXiv:1801.01290` | UC Berkeley | 最大熵 off-policy actor-critic，连续控制的强基线 |
| 2017-12 | [AlphaZero](https://arxiv.org/abs/1712.01815) `arXiv:1712.01815` | DeepMind | 不用人类棋谱，纯自我对弈 + MCTS 精通国际象棋、将棋与围棋 |
| 2017-07 | ★ [PPO](https://arxiv.org/abs/1707.06347) `arXiv:1707.06347` | OpenAI | 裁剪替代目标的简单稳定策略梯度，后来也是 RLHF 的主力算法 |
| 2016-01 | ★ [AlphaGo](https://www.nature.com/articles/nature16961) | DeepMind | 策略网络、价值网络与蒙特卡洛树搜索结合，击败职业围棋选手 |
| 2015-02 | [TRPO](https://arxiv.org/abs/1502.05477) `arXiv:1502.05477` | UC Berkeley | 信赖域约束的策略优化，单调改进保证 |
| 2013-12 | ★ [DQN](https://arxiv.org/abs/1312.5602) `arXiv:1312.5602` | DeepMind | 卷积网络 + Q 学习 + 经验回放，从像素学会玩 Atari |

### 视觉感知与 3D（20）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-06 | 🔥 [YOLO26](https://arxiv.org/abs/2606.03748) `arXiv:2606.03748` | Ultralytics | 统一的端到端实时视觉模型 |
| 2025-11 | ◆ [SAM 3D](https://arxiv.org/abs/2511.16624) `arXiv:2511.16624` | Meta | 单张图像重建物体与人体的 3D 形状、纹理和布局 |
| 2025-11 | ◆ [SAM 3](https://arxiv.org/abs/2511.16719) `arXiv:2511.16719` | Meta | 按名词短语或示例的"可提示概念分割"：一次找出并跟踪所有实例 |
| 2025-11 | ◆ [Depth Anything 3](https://arxiv.org/abs/2511.10647) `arXiv:2511.10647` | ByteDance Seed | 用朴素 Transformer + 深度-射线表示，从任意视图恢复空间一致几何 |
| 2025-09 | 🆕 [MapAnything](https://arxiv.org/abs/2509.13414) `arXiv:2509.13414` | Meta/CMU | 通用前馈度量 3D 重建：任意视图与可选几何输入 |
| 2025-07 | 🆕 [π³](https://arxiv.org/abs/2507.13347) `arXiv:2507.13347` | Shanghai AI Lab | 置换等变的视觉几何学习，不依赖参考视图 |
| 2025-04 | 🆕 [Describe Anything](https://arxiv.org/abs/2504.16072) `arXiv:2504.16072` | NVIDIA | 对图像与视频中任意区域做细致描述 |
| 2025-03 | ★ 🆕 [VGGT](https://arxiv.org/abs/2503.11651) `arXiv:2503.11651` | Oxford/Meta | 前馈 Transformer 一次推理出相机、深度、点图与轨迹（CVPR 2025 最佳论文） |
| 2024-08 | [SAM 2](https://arxiv.org/abs/2408.00714) `arXiv:2408.00714` | Meta | 带流式记忆的图像与视频统一分割 |
| 2024-01 | [Depth Anything](https://arxiv.org/abs/2401.10891) `arXiv:2401.10891` | HKU/TikTok | 大规模无标注图像自训练的单目深度基础模型 |
| 2023-12 | [DUSt3R](https://arxiv.org/abs/2312.14132) `arXiv:2312.14132` | Naver Labs | 直接回归点图，统一多视图几何任务，无需相机标定 |
| 2023-08 | [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) `arXiv:2308.04079` | Inria | 显式 3D 高斯 + 可微光栅化，实时高质量辐射场渲染 |
| 2023-04 | ★ [SAM](https://arxiv.org/abs/2304.02643) `arXiv:2304.02643` | Meta | 可提示的通用分割基础模型与 11 亿掩码数据集 |
| 2022-01 | [ConvNeXt](https://arxiv.org/abs/2201.03545) `arXiv:2201.03545` | Meta | 用 Transformer 的设计经验现代化 ResNet，纯卷积网络追平 Swin |
| 2020-05 | [DETR](https://arxiv.org/abs/2005.12872) `arXiv:2005.12872` | Facebook | 用 Transformer + 集合匹配做端到端检测，去掉锚框与 NMS |
| 2020-03 | [NeRF](https://arxiv.org/abs/2003.08934) `arXiv:2003.08934` | UC Berkeley/Google | 用 MLP 表示辐射场 + 体渲染做新视角合成 |
| 2017-03 | [Mask R-CNN](https://arxiv.org/abs/1703.06870) `arXiv:1703.06870` | Facebook | 在检测框上并行预测像素级掩码，实例分割基线 |
| 2015-06 | [YOLO](https://arxiv.org/abs/1506.02640) `arXiv:1506.02640` | UW/Ai2 | 把检测视为单次回归，实时目标检测 |
| 2015-06 | [Faster R-CNN](https://arxiv.org/abs/1506.01497) `arXiv:1506.01497` | Microsoft Research | 区域建议网络与检测网络共享特征，两阶段检测的经典 |
| 2015-05 | [U-Net](https://arxiv.org/abs/1505.04597) `arXiv:1505.04597` | Freiburg | 编码器-解码器 + 跳连的分割网络，后来成为扩散模型骨干 |

### AI for Science 与数学（38）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🔥 [Intern-S2-Preview](https://arxiv.org/abs/2608.13505) `arXiv:2608.13505` | Shanghai AI Lab | 面向科学的智能体基础模型 |
| 2026-04 | ◆ [异构科学基础模型协作](https://arxiv.org/abs/2604.27351) `arXiv:2604.27351` | UIUC | 让多种科学基础模型协同解决问题 |
| 2026-04 | ◆ [EquiformerV3](https://arxiv.org/abs/2604.09130) `arXiv:2604.09130` |  | 第三代 SE(3) 等变图注意力 Transformer：兼顾效率、表达力与物理一致性 |
| 2026-04 | ◆ [几何对齐税](https://arxiv.org/abs/2604.04155) `arXiv:2604.04155` |  | 把连续流形强行送入离散分词瓶颈会带来内在代价：科学基础模型的分词与连续几何 |
| 2026-03 | ◆ [Intern-S1-Pro](https://arxiv.org/abs/2603.25040) `arXiv:2603.25040` | Shanghai AI Lab | 万亿参数的科学多模态基础模型 |
| 2026-03 | ◆ [AI 能学会科学品味](https://arxiv.org/abs/2603.14473) `arXiv:2603.14473` | OpenMOSS | 让模型学习判断研究想法的价值 |
| 2025-12 | ◆ [探测科学通用智能](https://arxiv.org/abs/2512.16969) `arXiv:2512.16969` | Shanghai AI Lab | 按科学家工作流评估 LLM 的科学能力 |
| 2025-11 | ◆ [Kosmos](https://arxiv.org/abs/2511.02824) `arXiv:2511.02824` | Edison Scientific | 以结构化世界模型协调数百个智能体轮次的自主数据驱动科学发现 |
| 2025-10 | 🆕 [Aristotle](https://arxiv.org/abs/2510.01346) `arXiv:2510.01346` | Harmonic | 形式化验证的 IMO 2025 金牌级数学系统 |
| 2025-09 | 🆕 [SciReasoner](https://arxiv.org/abs/2509.21320) `arXiv:2509.21320` | Shanghai AI Lab | 跨学科科学推理基础模型 |
| 2025-08 | 🆕 [科学大模型综述](https://arxiv.org/abs/2508.21148) `arXiv:2508.21148` | Shanghai AI Lab | 从数据基础到智能体前沿的科学 LLM |
| 2025-08 | 🆕 [Intern-S1](https://arxiv.org/abs/2508.15763) `arXiv:2508.15763` | Shanghai AI Lab | 面向科学的多模态 MoE 基础模型，在分子、材料等专业任务上领先 |
| 2025-08 | 🆕 [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613) `arXiv:2508.03613` | Princeton | 脚手架式数据合成 + 自我修正的 Lean 定理证明器 |
| 2025-07 | 🆕 [Seed-Prover](https://arxiv.org/abs/2507.23726) `arXiv:2507.23726` | ByteDance Seed | 引理式整证推理 + 迭代修正的 Lean 证明器，形式化证明 IMO 2025 的 5/6 题 |
| 2025-07 | 🆕 [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) `arXiv:2507.22291` | Google DeepMind | 多源对地观测数据的通用嵌入场 |
| 2025-07 | 🆕 [MegaScience](https://arxiv.org/abs/2507.16812) `arXiv:2507.16812` | SJTU | 面向科学推理的大规模后训练数据集 |
| 2025-06 | 🆕 [UMA](https://arxiv.org/abs/2506.23971) `arXiv:2506.23971` | Meta | 覆盖分子、材料、催化的通用原子间势模型族 |
| 2025-06 | 🆕 [AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) | Google DeepMind | 输入 1Mb DNA 序列，以碱基分辨率预测数千种基因调控功能信号 |
| 2025-04 | 🆕 [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) `arXiv:2504.21801` | DeepSeek | 子目标分解 + RL 的 Lean 4 形式化证明模型 |
| 2025-04 | 🆕 [Kimina-Prover](https://arxiv.org/abs/2504.11354) `arXiv:2504.11354` | Moonshot/Numina | RL 驱动的形式化定理证明，推理风格接近非形式化思考 |
| 2025-04 | 🆕 [TxGemma](https://arxiv.org/abs/2504.06196) `arXiv:2504.06196` | Google | 面向药物研发的高效开放模型 |
| 2025-02 | 🆕 [Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1) | Arc Institute/NVIDIA | 9.3 万亿碱基、覆盖生命三域训练的 DNA 基础模型，1M 上下文 |
| 2025-02 | 🆕 [AlphaGeometry2](https://arxiv.org/abs/2502.03544) `arXiv:2502.03544` | Google DeepMind | 覆盖 84% 历年 IMO 几何题，超过金牌选手平均水平 |
| 2025-01 | 🆕 [ESM3](https://www.science.org/doi/10.1126/science.ads0018) | EvolutionaryScale | 序列、结构、功能多轨道的生成式蛋白质模型，设计出远缘的新荧光蛋白 |
| 2024-07 | [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) | Google DeepMind | Lean 中的 AlphaZero 式强化学习定理证明，IMO 2024 银牌水平 |
| 2024-05 | [Aurora](https://arxiv.org/abs/2405.13063) `arXiv:2405.13063` | Microsoft | 大气基础模型：在大气化学、海浪等多任务上微调（Nature 2025） |
| 2024-05 | [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) | Google DeepMind/Isomorphic | 扩散模块直接生成原子坐标，覆盖蛋白、核酸、小分子与修饰的复合物 |
| 2024-01 | [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5) | Google DeepMind | 神经语言模型 + 符号推理引擎，合成数据训练，解出奥赛几何题 |
| 2023-12 | [GenCast](https://arxiv.org/abs/2312.15796) `arXiv:2312.15796` | Google DeepMind | 扩散模型集合预报，15 天预报全面超过 ENS（Nature 2024） |
| 2023-12 | [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) | Google DeepMind | LLM 生成程序 + 自动评估的进化搜索，在组合数学 cap set 问题上得到新结果 |
| 2023-12 | [MatterGen](https://arxiv.org/abs/2312.03687) `arXiv:2312.03687` | Microsoft | 按目标性质生成稳定无机材料的扩散模型（Nature 2025） |
| 2023-11 | [GNoME](https://www.nature.com/articles/s41586-023-06735-9) | Google DeepMind | 图网络主动学习发现 220 万种新晶体，其中 38 万种稳定 |
| 2023-06 | [AlphaDev](https://www.nature.com/articles/s41586-023-06004-9) | Google DeepMind | RL 在汇编层面发现更快的排序算法，进入 LLVM 标准库 |
| 2023-03 | [ESM-2 / ESMFold](https://www.science.org/doi/10.1126/science.ade2574) | Meta | 150 亿参数蛋白质语言模型，原子级结构从语言模型表征中涌现 |
| 2022-12 | [GraphCast](https://arxiv.org/abs/2212.12794) `arXiv:2212.12794` | DeepMind | 图神经网络做 10 天全球中期天气预报，超过欧洲中心 HRES |
| 2021-07 | ★ [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2) | DeepMind | Evoformer + 结构模块，从序列以接近实验精度预测蛋白质结构（2024 诺贝尔化学奖） |
| 2020-10 | [FNO](https://arxiv.org/abs/2010.08895) `arXiv:2010.08895` | Caltech | 傅里叶神经算子：在函数空间学习 PDE 解算子，分辨率无关 |
| 2017-11 | [PINN](https://arxiv.org/abs/1711.10561) `arXiv:1711.10561` | Brown | 把 PDE 残差作为损失的物理信息神经网络 |

### 机器人与具身（32）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [从机队经验学习](https://arxiv.org/abs/2610.25736) `arXiv:2610.25736` |  | 机器人集体学习范式 |
| 2026-09 | 🔥 [GE-Act 2.0](https://arxiv.org/abs/2609.05588) `arXiv:2609.05588` | AgiBot | 世界-动作模型的预训练与扩展 |
| 2026-07 | 🔥 [机器人进度奖励建模综述](https://arxiv.org/abs/2607.21655) `arXiv:2607.21655` | Northwestern | 机器人学习中的进度奖励模型 |
| 2026-07 | 🔥 [RynnBrain 1.1](https://arxiv.org/abs/2607.17977) `arXiv:2607.17977` | DAMO Academy | 2B–30B 的具身基础模型族 |
| 2026-06 | 🔥 [Cosmos 3](https://arxiv.org/abs/2606.02800) `arXiv:2606.02800` | NVIDIA | 面向物理 AI 的全模态世界模型 |
| 2026-05 | 🔥 [Qwen-VLA](https://arxiv.org/abs/2605.30280) `arXiv:2605.30280` | Qwen | 跨任务、环境与本体统一的 VLA |
| 2026-05 | 🔥 [MolmoAct2](https://arxiv.org/abs/2605.02881) `arXiv:2605.02881` | Ai2 | 面向真实部署的动作推理模型 |
| 2026-02 | ◆ [DreamZero](https://arxiv.org/abs/2602.15922) `arXiv:2602.15922` | NVIDIA | 世界动作模型：联合预测视频与动作，实现零样本策略 |
| 2026-02 | ◆ [循环深度 VLA](https://arxiv.org/abs/2602.07845) `arXiv:2602.07845` | Ai2 | 用循环深度实现 VLA 的隐式测试时计算扩展 |
| 2026-01 | ◆ [Green-VLA](https://arxiv.org/abs/2602.00919) `arXiv:2602.00919` | Sber Robotics | 分阶段训练的通用机器人 VLA |
| 2025-11 | ◆ [从物理世界模型学习机器人](https://arxiv.org/abs/2511.07416) `arXiv:2511.07416` | Google DeepMind | 用视频世界模型生成的物理运动驱动机器人学习 |
| 2025-10 | 🆕 [World-in-World](https://arxiv.org/abs/2510.18135) `arXiv:2510.18135` | JHU 等 | 在闭环具身任务中评测生成式世界模型 |
| 2025-10 | 🆕 [LIBERO-Plus](https://arxiv.org/abs/2510.13626) `arXiv:2510.13626` | OpenMOSS | VLA 的鲁棒性深度分析：高成功率掩盖了脆弱性 |
| 2025-10 | 🆕 [机器人学习教程](https://arxiv.org/abs/2510.12403) `arXiv:2510.12403` | Hugging Face | 从强化学习到行为克隆与通用策略的现代机器人学习教程 |
| 2025-10 | 🆕 [Spatial Forcing](https://arxiv.org/abs/2510.12276) `arXiv:2510.12276` | HKUST-GZ | 把 VLA 中间表征隐式对齐到 3D 基础模型 |
| 2025-10 | 🆕 [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) `arXiv:2510.03342` | Google DeepMind | 行动前先思考、跨机器人本体迁移的 VLA + 具身推理模型 |
| 2025-09 | 🆕 [VLA-Adapter](https://arxiv.org/abs/2509.09372) `arXiv:2509.09372` |  | 微型骨干的高效 VLA 范式 |
| 2025-09 | 🆕 [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) `arXiv:2509.09674` | Tsinghua 等 | 用 RL 扩展 VLA 训练，长程任务显著提升 |
| 2025-08 | 🆕 [MolmoAct](https://arxiv.org/abs/2508.07917) `arXiv:2508.07917` | Ai2 | 在空间中推理（深度、轨迹）再行动的动作推理模型 |
| 2025-06 | 🆕 [SmolVLA](https://arxiv.org/abs/2506.01844) `arXiv:2506.01844` | Hugging Face | 4.5 亿参数、社区数据训练的小型 VLA，消费级 GPU 可用 |
| 2025-04 | 🆕 [π0.5](https://arxiv.org/abs/2504.16054) `arXiv:2504.16054` | Physical Intelligence | 异构数据协同训练，在从未见过的家庭中执行长程任务 |
| 2025-03 | 🆕 [Gemini Robotics](https://arxiv.org/abs/2503.20020) `arXiv:2503.20020` | Google DeepMind | 基于 Gemini 2.0 的 VLA 与具身推理模型 |
| 2025-03 | 🆕 [GR00T N1](https://arxiv.org/abs/2503.14734) `arXiv:2503.14734` | NVIDIA | 双系统（VLM 规划 + 扩散动作头）的人形机器人基础模型 |
| 2025-02 | 🆕 [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) `arXiv:2502.19645` | Stanford | 并行解码、动作分块、连续动作与 L1 回归的 VLA 微调配方 |
| 2025-01 | 🆕 [FAST 动作分词](https://arxiv.org/abs/2501.09747) `arXiv:2501.09747` | Physical Intelligence | 用离散余弦变换压缩动作序列，让自回归 VLA 高效训练高频灵巧任务 |
| 2024-10 | ★ [π0](https://arxiv.org/abs/2410.24164) `arXiv:2410.24164` | Physical Intelligence | VLM 骨干 + 流匹配动作专家，高频灵巧操作的通用策略 |
| 2024-06 | [OpenVLA](https://arxiv.org/abs/2406.09246) `arXiv:2406.09246` | Stanford/UC Berkeley | 7B 开源 VLA，微调高效 |
| 2024-05 | [Octo](https://arxiv.org/abs/2405.12213) `arXiv:2405.12213` | UC Berkeley 等 | 开源通用机器人策略 Transformer |
| 2023-10 | [Open X-Embodiment](https://arxiv.org/abs/2310.08864) `arXiv:2310.08864` | 多机构联合 | 22 种机器人、100 万条轨迹的开放数据集与 RT-X |
| 2023-07 | ★ [RT-2](https://arxiv.org/abs/2307.15818) `arXiv:2307.15818` | Google DeepMind | 把动作当成文本 token 与网络数据共同训练，视觉-语言-动作模型（VLA）由此得名 |
| 2023-03 | ★ [Diffusion Policy](https://arxiv.org/abs/2303.04137) `arXiv:2303.04137` | Columbia/TRI/MIT | 用扩散模型表示视觉运动策略，处理多峰动作分布 |
| 2022-12 | [RT-1](https://arxiv.org/abs/2212.06817) `arXiv:2212.06817` | Google | 13 万条真实演示训练的机器人 Transformer |

### 安全与鲁棒性（42）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [涌现失准模型共享一个回路吗](https://arxiv.org/abs/2610.20887) `arXiv:2610.20887` |  | 不同涌现失准模型的共同机制 |
| 2026-09 | 🔥 [涌现失准的机制诊断](https://arxiv.org/abs/2609.34970) `arXiv:2609.34970` |  | 用二阶几何动态分析涌现失准在训练中如何形成，并在参数空间中缓解 |
| 2026-09 | 🔥 [后训练留下的行为影子](https://arxiv.org/abs/2609.29233) `arXiv:2609.29233` | PKU | 后训练会在无关决策上留下系统性行为偏移 |
| 2026-08 | 🔥 [涌现失准并不神奇](https://arxiv.org/abs/2608.29118) `arXiv:2608.29118` |  | 对涌现失准的祛魅分析 |
| 2026-08 | 🔥 [从闭源 API 窃取推理轨迹](https://arxiv.org/abs/2608.09867) `arXiv:2608.09867` |  | 隐藏的推理轨迹可被从专有 API 中提取 |
| 2026-08 | 🔥 [OpenART](https://arxiv.org/abs/2608.00677) `arXiv:2608.00677` |  | 用开放式环境进化扩展智能体红队测试 |
| 2026-06 | 🔥 [什么塑造了涌现失准](https://arxiv.org/abs/2606.20814) `arXiv:2606.20814` |  | 从训练动力学、模型先验与数据三方面分析 |
| 2026-06 | 🔥 [谄媚也能诱发涌现失准](https://arxiv.org/abs/2606.09068) `arXiv:2606.09068` |  | 只在谄媚数据上微调即可引发广泛失准，并可用对齐门控逆转 |
| 2026-06 | 🔥 [涌现对齐与伦理人格的可投射性](https://arxiv.org/abs/2606.09475) `arXiv:2606.09475` |  | 与涌现失准对称的涌现对齐现象 |
| 2026-06 | 🔥 [steering 向量是攻击面](https://arxiv.org/abs/2606.05958) `arXiv:2606.05958` |  | 激活操控可被用作对抗攻击手段 |
| 2026-06 | 🔥 [评分细则 RL 中的奖励投机](https://arxiv.org/abs/2606.04923) `arXiv:2606.04923` |  | 复现、分析并检测基于 rubric 的 RL 中对评审模型偏好的投机 |
| 2026-05 | 🔥 [涌现失准中的人格坍缩](https://arxiv.org/abs/2605.12850) `arXiv:2605.12850` |  | 涌现失准伴随模型"人格"的坍缩 |
| 2026-04 | ◆ [大模型时代的奖励投机](https://arxiv.org/abs/2604.13602) `arXiv:2604.13602` | Fudan | 奖励投机的机制、涌现失准与挑战 |
| 2026-04 | ◆ [用特征叠加几何理解涌现失准](https://arxiv.org/abs/2605.00842) `arXiv:2605.00842` |  | 涌现失准的叠加几何解释 |
| 2026-02 | ◆ [思维链可监控性的信息论](https://arxiv.org/abs/2602.18297) `arXiv:2602.18297` |  | 用信息论分析并改进思维链可监控性 |
| 2026-01 | ◆ [AgentDoG](https://arxiv.org/abs/2601.18491) `arXiv:2601.18491` | Shanghai AI Lab | 智能体安全与安保的诊断式护栏框架 |
| 2026-01 | ◆ [前沿模型安全报告](https://arxiv.org/abs/2601.10527) `arXiv:2601.10527` |  | 对 GPT-5.2、Gemini 3 Pro 等多个前沿模型的统一安全评估 |
| 2025-11 | ◆ [奖励投机导致的自然涌现失准](https://arxiv.org/abs/2511.18397) `arXiv:2511.18397` | Anthropic | 在真实编程 RL 环境中学会奖励投机，会泛化出对齐伪装与破坏安全研究等失准行为 |
| 2025-10 | 🆕 [上下文学习导致的涌现失准](https://arxiv.org/abs/2510.11288) `arXiv:2510.11288` |  | 少量窄域上下文示例也会引发广泛失准 |
| 2025-09 | 🆕 [反谋划训练的压力测试](https://arxiv.org/abs/2509.15541) `arXiv:2509.15541` | OpenAI/Apollo Research | 审慎对齐可将隐蔽行为降低约 30 倍，但模型对"被评测"的情境意识混淆了结果 |
| 2025-09 | ★ 🆕 [语言模型为何幻觉](https://arxiv.org/abs/2509.04664) `arXiv:2509.04664` | OpenAI/Georgia Tech | 幻觉源自统计必然（类似二分类错误）与奖励"猜测"而非"承认不知道"的评测方式 |
| 2025-08 | 🆕 [A.S.E](https://arxiv.org/abs/2508.18106) `arXiv:2508.18106` |  | 仓库级 AI 生成代码安全性基准 |
| 2025-07 | 🆕 [潜意识学习](https://arxiv.org/abs/2507.14805) `arXiv:2507.14805` | Anthropic Fellows/Truthful AI | 学生模型能从看似无关的数据（如数字序列）中继承教师的行为特质，前提是共享初始化 |
| 2025-07 | 🆕 [思维链可监控性](https://arxiv.org/abs/2507.11473) `arXiv:2507.11473` | 多机构联合 | CoT 监控是脆弱但宝贵的安全机会，呼吁开发者保护可监控性 |
| 2025-06 | 🆕 [为何有的模型伪装对齐](https://arxiv.org/abs/2506.18032) `arXiv:2506.18032` | Anthropic 等 | 25 个模型中只有少数表现出对齐伪装，分析其动机 |
| 2025-06 | 🆕 [智能体失准](https://www.anthropic.com/research/agentic-misalignment) | Anthropic | 在模拟企业环境中，多家前沿模型面临被替换威胁时会采取勒索等内部威胁行为 |
| 2025-05 | 🆕 [模型知道自己在被评测](https://arxiv.org/abs/2505.23836) `arXiv:2505.23836` | MATS/Apollo | 前沿模型能以较高准确率分辨评测与真实部署场景 |
| 2025-05 | 🆕 [推理模型并不总说真话](https://arxiv.org/abs/2505.05410) `arXiv:2505.05410` | Anthropic | 推理模型的思维链常常不承认使用了提示中的线索：CoT 忠实性有限 |
| 2025-03 | 🆕 [监控推理模型的不当行为](https://arxiv.org/abs/2503.11926) `arXiv:2503.11926` | OpenAI | CoT 监控能抓住奖励投机；但对 CoT 施加优化压力会让模型学会隐藏意图 |
| 2025-03 | 🆕 [审计隐藏目标](https://arxiv.org/abs/2503.10965) `arXiv:2503.10965` | Anthropic | 蓝队用可解释性与行为分析，盲审出模型被故意植入的隐藏目标 |
| 2025-02 | ★ 🆕 [涌现失准](https://arxiv.org/abs/2502.17424) `arXiv:2502.17424` | Truthful AI/UCL 等 | 只在"写不安全代码"这一窄任务上微调，就会导致广泛的有害、失准行为 |
| 2025-01 | 🆕 [宪法分类器](https://arxiv.org/abs/2501.18837) `arXiv:2501.18837` | Anthropic | 用宪法生成的合成数据训练输入输出分类器，抵御通用越狱 |
| 2025-01 | 🆕 [GuardReasoner](https://arxiv.org/abs/2501.18492) `arXiv:2501.18492` | NUS | 基于推理的 LLM 安全护栏 |
| 2024-12 | ★ 🆕 [对齐伪装](https://arxiv.org/abs/2412.14093) `arXiv:2412.14093` | Anthropic/Redwood | Claude 3 Opus 在认为自己被训练时策略性地顺从，以保护原有偏好 |
| 2024-12 | 🆕 [上下文中的谋划](https://arxiv.org/abs/2412.04984) `arXiv:2412.04984` | Apollo Research | 前沿模型在给定目标的智能体场景中会隐瞒、关闭监督、自我外泄 |
| 2024-01 | [Sleeper Agents](https://arxiv.org/abs/2401.05566) `arXiv:2401.05566` | Anthropic | 植入的后门行为能挺过 SFT、RL 与对抗训练 |
| 2023-12 | [弱到强泛化](https://arxiv.org/abs/2312.09390) `arXiv:2312.09390` | OpenAI | 用弱模型监督强模型，强模型能超越监督者：超级对齐的实证类比 |
| 2023-07 | [GCG 通用对抗后缀](https://arxiv.org/abs/2307.15043) `arXiv:2307.15043` | CMU | 自动搜索可迁移的对抗后缀，越狱对齐后的 LLM |
| 2017-06 | [PGD 对抗训练](https://arxiv.org/abs/1706.06083) `arXiv:1706.06083` | MIT | 鲁棒优化视角：以 PGD 为最强一阶对手的极小极大训练 |
| 2016-06 | [AI 安全的具体问题](https://arxiv.org/abs/1606.06565) `arXiv:1606.06565` | Google/OpenAI/Stanford | 负面副作用、奖励投机、可扩展监督、安全探索、分布漂移五类问题 |
| 2014-12 | [FGSM](https://arxiv.org/abs/1412.6572) `arXiv:1412.6572` | Google | 对抗样本源于高维线性性，快速梯度符号攻击与对抗训练 |
| 2013-12 | ★ [对抗样本](https://arxiv.org/abs/1312.6199) `arXiv:1312.6199` | Google/NYU | 人眼不可察觉的扰动即可让分类器出错，神经网络的"有趣性质" |

### 评测与基准（27）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🔥 [基准是难度上的经验测度](https://arxiv.org/abs/2610.29927) `arXiv:2610.29927` |  | 把基准分数理解为题目难度分布上的经验测度 |
| 2026-09 | 🔥 [Benchmark Radar](https://arxiv.org/abs/2609.11115) `arXiv:2609.11115` | CMU | AI 基准与评测的动态数据库与搜索引擎 |
| 2026-04 | ◆ [Video-MME-v2](https://arxiv.org/abs/2604.05015) `arXiv:2604.05015` | MME | 面向下一阶段的综合视频理解基准 |
| 2025-10 | 🆕 [GDPval](https://arxiv.org/abs/2510.04374) `arXiv:2510.04374` | OpenAI | 44 种职业的真实经济价值任务，专家盲评模型产出 |
| 2025-05 | 🆕 [ARC-AGI-2](https://arxiv.org/abs/2505.11831) `arXiv:2505.11831` | ARC Prize Foundation | 对人类容易、对 AI 仍很难的新一代抽象推理基准 |
| 2025-04 | 🆕 [排行榜幻觉](https://arxiv.org/abs/2504.20879) `arXiv:2504.20879` | Cohere 等 | 私下多版本测试与数据访问不对称扭曲了 Chatbot Arena 排名 |
| 2025-02 | 🆕 [SuperGPQA](https://arxiv.org/abs/2502.14739) `arXiv:2502.14739` | ByteDance Seed | 覆盖 285 个研究生学科的知识与推理评测 |
| 2025-02 | 🆕 [SWE-Lancer](https://arxiv.org/abs/2502.12115) `arXiv:2502.12115` | OpenAI | 1400+ 个真实自由职业软件任务，总价值 100 万美元 |
| 2025-01 | ★ 🆕 [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) `arXiv:2501.14249` | CAIS/Scale AI | 数千道多学科前沿难题，"最后一场闭卷考试" |
| 2024-11 | 🆕 [RE-Bench](https://arxiv.org/abs/2411.15114) `arXiv:2411.15114` | METR | 在 AI 研发任务上比较前沿智能体与人类专家 |
| 2024-11 | 🆕 [SimpleQA](https://arxiv.org/abs/2411.04368) `arXiv:2411.04368` | OpenAI | 短事实问答，衡量事实性与"知道自己不知道" |
| 2024-11 | 🆕 [FrontierMath](https://arxiv.org/abs/2411.04872) `arXiv:2411.04872` | Epoch AI | 数学家原创的研究级未公开难题 |
| 2024-08 | [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) | OpenAI | 人工校验的 500 题 SWE-bench 子集，成为编码智能体的主流指标 |
| 2024-06 | [LiveBench](https://arxiv.org/abs/2406.19314) `arXiv:2406.19314` | Abacus.AI 等 | 持续更新、客观评分的无污染基准 |
| 2024-06 | [MMLU-Pro](https://arxiv.org/abs/2406.01574) `arXiv:2406.01574` | Waterloo | 10 选项、更重推理的 MMLU 升级版 |
| 2024-03 | [LiveCodeBench](https://arxiv.org/abs/2403.07974) `arXiv:2403.07974` | UC Berkeley/MIT | 持续从新竞赛收集题目，避免数据污染 |
| 2024-03 | [Chatbot Arena](https://arxiv.org/abs/2403.04132) `arXiv:2403.04132` | UC Berkeley | 众包两两盲评 + Elo/BT 排名的开放平台 |
| 2023-11 | [MMMU](https://arxiv.org/abs/2311.16502) `arXiv:2311.16502` | OSU 等 | 大学水平的多学科多模态理解与推理 |
| 2023-11 | [GPQA](https://arxiv.org/abs/2311.12022) `arXiv:2311.12022` | NYU/Anthropic | "谷歌也查不到"的研究生级科学问答 |
| 2022-11 | [HELM](https://arxiv.org/abs/2211.09110) `arXiv:2211.09110` | Stanford | 多指标、多场景的语言模型整体评估 |
| 2022-06 | [BIG-bench](https://arxiv.org/abs/2206.04615) `arXiv:2206.04615` | Google 等 450 位作者 | 204 个任务的协作基准，研究能力随规模的变化 |
| 2021-07 | [HumanEval / Codex](https://arxiv.org/abs/2107.03374) `arXiv:2107.03374` | OpenAI | 代码生成基准与 pass@k 指标 |
| 2021-03 | [MATH](https://arxiv.org/abs/2103.03874) `arXiv:2103.03874` | UC Berkeley | 竞赛数学问题集 |
| 2020-09 | [MMLU](https://arxiv.org/abs/2009.03300) `arXiv:2009.03300` | UC Berkeley | 57 个学科的多项选择知识基准 |
| 2019-11 | ★ [ARC / 智能的度量](https://arxiv.org/abs/1911.01547) `arXiv:1911.01547` | Google | 把智能定义为技能获取效率，提出抽象与推理语料 ARC |
| 2018-04 | [GLUE](https://arxiv.org/abs/1804.07461) `arXiv:1804.07461` | NYU/UW/DeepMind | 多任务自然语言理解基准 |
| 2009-06 | ★ [ImageNet](https://ieeexplore.ieee.org/document/5206848) | Princeton/Stanford | 千万级层级标注图像库；其竞赛直接催生了 2012 年的深度学习突破 |

### 基础模型与技术报告（50）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🔥 [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) `arXiv:2609.19969` | DeepSeek | 面向输入密集的长程智能体负载，把 KV 缓存压缩推到极限，降低预填充与存储成本 |
| 2026-07 | ★ 🔥 [Kimi K3](https://arxiv.org/abs/2607.24653) `arXiv:2607.24653` | Moonshot AI | 2.8T 参数 MoE（激活 104B，896 选 16 专家），基于 Kimi Delta Attention 与 Attention Residuals，百万 token 上下文、原生视觉 |
| 2026-07 | 🔥 [Gemma 4](https://arxiv.org/abs/2607.02770) `arXiv:2607.02770` | Google | 第四代 Gemma 开源模型族 |
| 2026-06 | 🔥 [Ling & Ring 2.6](https://arxiv.org/abs/2606.15079) `arXiv:2606.15079` | inclusionAI | 万亿参数级、高效即时的智能体模型族 |
| 2026-05 | 🔥 [MiniMax-M2 系列](https://arxiv.org/abs/2605.26494) `arXiv:2605.26494` | MiniMax | 小激活参数、面向真实世界智能体任务的 MoE 模型族 |
| 2026-04 | ★ ◆ [DeepSeek-V4](https://arxiv.org/abs/2606.19348) `arXiv:2606.19348` | DeepSeek | V4-Pro 1.6T（激活 49B）与 V4-Flash 284B（激活 13B）MoE，百万 token 上下文；压缩稀疏注意力（CSA）+ 重度压缩注意力（HCA）混合、mHC 残差与 Muon 优化器，32T token 预训练 |
| 2026-04 | ◆ [Qwen3.5-Omni](https://arxiv.org/abs/2604.15804) `arXiv:2604.15804` | Qwen | 新一代全模态模型 |
| 2026-04 | ◆ [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) `arXiv:2604.12374` | NVIDIA | 混合 Mamba-Transformer MoE，面向智能体推理 |
| 2026-02 | ◆ [GLM-5](https://arxiv.org/abs/2602.15763) `arXiv:2602.15763` | Z.ai | 从 vibe coding 到智能体工程的开源模型 |
| 2026-02 | ◆ [Step 3.5 Flash](https://arxiv.org/abs/2602.10604) `arXiv:2602.10604` | StepFun | 11B 激活参数达到前沿水平的开放模型 |
| 2026-02 | ◆ [ERNIE 5.0](https://arxiv.org/abs/2602.04705) `arXiv:2602.04705` | Baidu | 原生全模态统一自回归的超大规模 MoE 模型 |
| 2026-02 | ◆ [Kimi K2.5](https://arxiv.org/abs/2602.02276) `arXiv:2602.02276` | Moonshot AI | 视觉智能体能力的开源多模态模型 |
| 2026-01 | ◆ [LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725) `arXiv:2601.16725` | Meituan | 智能体推理强化的思考模型 |
| 2025-12 | ◆ [Olmo 3](https://arxiv.org/abs/2512.13961) `arXiv:2512.13961` | Ai2 | 完全开放的模型流程（数据、检查点、中期训练、思考与 RL） |
| 2025-12 | ◆ [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) `arXiv:2512.02556` | DeepSeek | DeepSeek 稀疏注意力（DSA）+ 大规模智能体任务合成与 RL |
| 2025-10 | 🆕 [Ling 2.0（万亿通用推理）](https://arxiv.org/abs/2510.22115) `arXiv:2510.22115` | inclusionAI | 高稀疏 MoE 的万亿参数基础模型族与 FP8 训练 |
| 2025-09 | 🆕 [Apertus](https://arxiv.org/abs/2509.14233) `arXiv:2509.14233` | Swiss AI | 合规、多语言（1800+ 语言）的完全开放 LLM |
| 2025-09 | 🆕 [K2-Think](https://arxiv.org/abs/2509.07604) `arXiv:2509.07604` | MBZUAI | 32B 参数高效推理系统 |
| 2025-09 | 🆕 [LongCat-Flash](https://arxiv.org/abs/2509.01322) `arXiv:2509.01322` | Meituan | 零计算专家实现动态激活的 560B MoE，捷径连接重叠通信 |
| 2025-08 | 🆕 [gpt-oss](https://arxiv.org/abs/2508.10925) `arXiv:2508.10925` | OpenAI | OpenAI 的开放权重推理模型（120B/20B MoE） |
| 2025-08 | 🆕 [GLM-4.5](https://arxiv.org/abs/2508.06471) `arXiv:2508.06471` | Zhipu/Tsinghua | 面向智能体、推理与编码的 355B MoE，混合推理模式 |
| 2025-08 | 🆕 [GPT-5 System Card](https://arxiv.org/abs/2601.03267) `arXiv:2601.03267` | OpenAI | GPT-5 统一系统（快速模型 + 思考模型 + 路由器）的能力与安全评估 |
| 2025-07 | ★ 🆕 [Kimi K2](https://arxiv.org/abs/2507.20534) `arXiv:2507.20534` | Moonshot AI | 1T MoE，MuonClip 优化器（QK-Clip）实现 15.5T token 零损失尖峰预训练 |
| 2025-07 | 🆕 [Gemini 2.5](https://arxiv.org/abs/2507.06261) `arXiv:2507.06261` | Google DeepMind | 原生思考、长上下文与智能体能力的前沿模型 |
| 2025-06 | 🆕 [Magistral](https://arxiv.org/abs/2506.10910) `arXiv:2506.10910` | Mistral AI | Mistral 的纯 RL 推理模型与训练栈 |
| 2025-05 | 🆕 [AceReason-Nemotron](https://arxiv.org/abs/2505.16400) `arXiv:2505.16400` | NVIDIA | 先数学后代码的分阶段 RL 显著提升小模型推理 |
| 2025-05 | ★ 🆕 [Qwen3](https://arxiv.org/abs/2505.09388) `arXiv:2505.09388` | Qwen | 思考/非思考模式统一与思考预算，稠密与 MoE 族（235B-A22B） |
| 2025-05 | 🆕 [DeepSeek-V3 硬件反思](https://arxiv.org/abs/2505.09343) `arXiv:2505.09343` | DeepSeek | 从 V3 看模型-硬件协同设计：MLA、MoE、FP8 与网络拓扑 |
| 2025-05 | 🆕 [MiMo](https://arxiv.org/abs/2505.07608) `arXiv:2505.07608` | Xiaomi | 从预训练（推理导向数据、多 token 预测）到 RL 全链路激发 7B 推理潜力 |
| 2025-05 | 🆕 [Llama-Nemotron](https://arxiv.org/abs/2505.00949) `arXiv:2505.00949` | NVIDIA | 神经架构搜索压缩 + 推理开关的开放推理模型族 |
| 2025-04 | 🆕 [Phi-4-reasoning](https://arxiv.org/abs/2504.21318) `arXiv:2504.21318` | Microsoft | 精选可教提示 + o3-mini 轨迹 SFT + 少量 RL 的 14B 推理模型 |
| 2025-04 | 🆕 [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) `arXiv:2504.13914` | ByteDance Seed | MoE 推理模型与 RL 基础设施 |
| 2025-04 | 🆕 [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | Meta | Meta 首个原生多模态 MoE 族（Scout/Maverick） |
| 2025-03 | 🆕 [Gemma 3](https://arxiv.org/abs/2503.19786) `arXiv:2503.19786` | Google DeepMind | 5:1 局部/全局注意力、128K 上下文的多模态开源模型 |
| 2025-02 | 🆕 [SmolLM2](https://arxiv.org/abs/2502.02737) `arXiv:2502.02737` | Hugging Face | 以数据为中心的小模型训练，多阶段混合与新数据集 |
| 2024-12 | 🆕 [OLMo 2](https://arxiv.org/abs/2501.00656) `arXiv:2501.00656` | Ai2 | 训练稳定性改进（QK-norm、重排归一化）与中期训练课程 |
| 2024-12 | ★ 🆕 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) `arXiv:2412.19437` | DeepSeek | 671B MoE：MLA + 无辅助损失均衡 + 多 token 预测 + FP8 训练，约 280 万 H800 小时 |
| 2024-12 | 🆕 [Qwen2.5](https://arxiv.org/abs/2412.15115) `arXiv:2412.15115` | Qwen | 18T token 预训练与大规模后训练 |
| 2024-12 | 🆕 [Phi-4](https://arxiv.org/abs/2412.08905) `arXiv:2412.08905` | Microsoft Research | 以合成数据为核心的 14B 模型，推理能力超过教师 |
| 2024-07 | [Llama 3](https://arxiv.org/abs/2407.21783) `arXiv:2407.21783` | Meta | 405B 稠密模型的完整训练报告：数据、扩展律、基础设施与后训练 |
| 2024-07 | [Gemma 2](https://arxiv.org/abs/2408.00118) `arXiv:2408.00118` | Google | 局部-全局交替注意力 + 知识蒸馏训练的小模型 |
| 2024-07 | [Qwen2](https://arxiv.org/abs/2407.10671) `arXiv:2407.10671` | Qwen | 开源模型族与长上下文、多语言能力 |
| 2024-03 | [Gemini 1.5](https://arxiv.org/abs/2403.05530) `arXiv:2403.05530` | Google | MoE + 百万到千万 token 上下文的多模态模型 |
| 2024-02 | [OLMo](https://arxiv.org/abs/2402.00838) `arXiv:2402.00838` | Ai2 | 完全开放（数据、代码、检查点）的语言模型 |
| 2023-12 | [Gemini](https://arxiv.org/abs/2312.11805) `arXiv:2312.11805` | Google | 原生多模态的模型族 |
| 2023-10 | [Mistral 7B](https://arxiv.org/abs/2310.06825) `arXiv:2310.06825` | Mistral AI | GQA + 滑动窗口注意力的高效 7B 模型 |
| 2023-07 | [Llama 2](https://arxiv.org/abs/2307.09288) `arXiv:2307.09288` | Meta | 开放权重的对话模型与完整 RLHF 细节 |
| 2023-03 | ★ [GPT-4](https://arxiv.org/abs/2303.08774) `arXiv:2303.08774` | OpenAI | 多模态大模型；可预测扩展：用小 1000 倍的模型预测最终损失 |
| 2023-02 | ★ [LLaMA](https://arxiv.org/abs/2302.13971) `arXiv:2302.13971` | Meta | 只用公开数据、过训练小模型：13B 超过 GPT-3，开源模型生态起点 |
| 2022-04 | [PaLM](https://arxiv.org/abs/2204.02311) `arXiv:2204.02311` | Google | 540B 稠密模型与 Pathways 系统，突破性的少样本推理 |

### 综述/教材（8）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2025-09 | 🆕 [大推理模型 RL 综述](https://arxiv.org/abs/2509.08827) `arXiv:2509.08827` | Tsinghua 等 | RL 用于推理模型的奖励设计、策略优化、采样与基础设施 |
| 2025-02 | 🆕 [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Google DeepMind | TPU/GPU 上 LLM 训练与推理的并行、通信与屋顶线分析 |
| 2024-06 | [MoE 综述](https://arxiv.org/abs/2407.06204) `arXiv:2407.06204` | HKUST 等 | 混合专家的算法、系统与应用 |
| 2024-04 | [机制可解释性与 AI 安全综述](https://arxiv.org/abs/2404.14082) `arXiv:2404.14082` | Amsterdam | 机制可解释性的概念、方法与安全意义 |
| 2023-12 | ★ [Understanding Deep Learning](https://udlbook.github.io/udlbook/) | Simon Prince | 覆盖基础、Transformer、扩散、RL 与理论的现代教材，免费在线 |
| 2023-11 | [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) | Bishop & Bishop | 从概率视角系统讲解深度学习的新教材 |
| 2021-06 | [Dive into Deep Learning](https://arxiv.org/abs/2106.11342) `arXiv:2106.11342` | Amazon 等 | 可交互的代码化深度学习教材 |
| 2016-11 | ★ [Deep Learning（教材）](https://www.deeplearningbook.org/) | Goodfellow/Bengio/Courville | 深度学习的经典教材 |

