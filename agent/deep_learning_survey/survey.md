# 深度学习：底层原理、2024–2026 前沿与核心应用（长文综述）

> 数据截至 2026-10-30。论文库共 1102 条（1051 条 arXiv 已用官方 API 核验标题与日期），其中 2024-11 及之后（"最近两年"）728 条，完整列表见附录 A。正文只引用有代表性的工作；附录收录得更全，便于检索复用。数字摘自论文摘要或技术报告，实验设置各不相同，只宜看量级。

**阅读指南。** 第 1 节给出统一的记号与训练框架；第 2–5 节讲"怎么学"的底层原理：优化、稳定性与参数化、泛化、扩展律；第 6–8 节讲架构：注意力、线性注意力与循环、混合专家与条件记忆；第 9–12 节讲"学什么、怎么用"：表征学习、预训练、后训练与推理；第 13–15 节讲可解释性、效率与安全；第 16 节概览核心应用；第 17 节是最近两年的大事记；第 18 节讨论开放问题。

## 1 基本框架：函数、目标、梯度

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

## 2 优化与训练动力学

### 2.1 从 SGD 到 AdamW

小批量随机梯度下降 $\theta_{k+1}=\theta_k-\eta\,\hat g_k$ 是一切的起点；动量 $m_k=\mu m_{k-1}+\hat g_k$ 平滑噪声。Adam（[Kingma 与 Ba](https://arxiv.org/abs/1412.6980)）为每个坐标维护一阶、二阶矩的指数平均：

$$m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,\quad \theta_{k+1}=\theta_k-\eta\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}$$

[AdamW](https://arxiv.org/abs/1711.05101) 把权重衰减从梯度中解耦（$\theta\leftarrow\theta-\eta\lambda\theta$），成为十年来大模型的默认优化器。[Wang 与 Aitchison](https://arxiv.org/abs/2405.13698) 指出 AdamW 的权重本质上是更新的指数滑动平均，时间尺度 $1/(\eta\lambda)$ 应随数据量调整；[Power Lines](https://arxiv.org/abs/2505.13738) 进一步给出最优权重衰减与批大小随 token/参数比的幂律。

### 2.2 学习率调度：warmup、余弦、WSD 与线性到零

几乎所有大模型训练都有 warmup。[Kalra 与 Barkeshli](https://arxiv.org/abs/2406.09405) 发现 warmup 的主要作用是把网络带到锐度更低、更"良态"的区域，从而容许更大的学习率。衰减阶段，[余弦调度](https://arxiv.org/abs/1608.03983)曾是默认；[MiniCPM](https://arxiv.org/abs/2404.06395) 提出 Warmup-Stable-Decay（WSD），可在任意时刻衰减得到可用的检查点；[河谷地形](https://arxiv.org/abs/2410.05192)解释了 WSD：稳定期沿着"河流"方向前进（损失看起来下降很慢），衰减期才下到"谷底"。[Bergsma 等](https://arxiv.org/abs/2502.15938)发现算力最优训练中，线性衰减到零（D2Z）稳定优于余弦衰减到 10%。

### 2.3 优化器即最速下降：从 Adam 到 Muon

一个统一视角（[Bernstein 与 Newhouse](https://arxiv.org/abs/2409.20325)、[模块对偶](https://arxiv.org/abs/2410.21265)）：在某个范数约束下，让一阶近似的损失下降最多，

$$\Delta^\star=\arg\min_{\lVert\Delta\rVert\le\eta}\ \langle G,\Delta\rangle .$$

- 欧氏范数：$\Delta^\star=-\eta\,G/\lVert G\rVert_F$，即归一化 SGD。
- 逐坐标 $\ell_\infty$ 范数：$\Delta^\star=-\eta\,\mathrm{sign}(G)$，即 signSGD；Adam 在 $\beta_1=\beta_2=0$ 时退化为它。
- 矩阵谱范数 $\lVert\Delta\rVert_{2\to2}$：设 $G=U\Sigma V^\top$，由于 $\langle G,\Delta\rangle=\mathrm{tr}(\Sigma\,U^\top\Delta V)$ 且 $\lVert U^\top\Delta V\rVert_{2\to2}\le\eta$，最优解是 $U^\top\Delta V=-\eta I$，即

$$\Delta^\star=-\eta\,UV^\top .$$

这就是 [Muon](https://kellerjordan.github.io/posts/muon/)：对动量矩阵做正交化，把所有奇异值"拉平"为 1。直接做 SVD 太贵，实践中用五次多项式的 Newton–Schulz 迭代 $X\leftarrow aX+(bXX^\top+c(XX^\top)^2)X$，几步只用矩阵乘法即可近似。Shampoo（[Gupta 等](https://arxiv.org/abs/1802.09568)）的更新 $L^{-1/4}GR^{-1/4}$ 在不做指数平均时与 Muon 方向相同；[SOAP](https://arxiv.org/abs/2409.11321) 在 Shampoo 的特征基里运行 Adam。

为什么用谱范数？权重矩阵的作用是把输入向量映射成输出向量，用"对任意输入，输出最多改变多少"的算子范数来度量更新大小，比逐元素度量更符合它的功能。机制上，[Wang 等](https://arxiv.org/abs/2509.26030)发现正交化让很少被激活的方向（长尾特征、长尾联想记忆）也获得同样幅度的学习，这是 Muon 的主要优势来源。

**最近两年：Muon 的规模化与冷静的再评估。**

- [Moonlight](https://arxiv.org/abs/2502.16982)：加入权重衰减、按 $0.2\sqrt{\max(m,n)}$ 缩放使更新的 RMS 与 AdamW 对齐后，Muon 可直接沿用 AdamW 的超参数，算力最优训练下约有 2 倍计算效率。[Essential AI](https://arxiv.org/abs/2505.02222) 显示 Muon 在大批量下保持数据效率，扩展了算力-时间的帕累托前沿。
- [Kimi K2](https://arxiv.org/abs/2507.20534)：Muon 在大模型上会让注意力 logit 失控增长；MuonClip 在 logit 过大时按头重缩放 $W_q,W_k$（QK-Clip），实现了 1T 参数、15.5T token 的零损失尖峰预训练。[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（1.6T）也采用 Muon。
- 分布式与改进：[Dion](https://arxiv.org/abs/2504.05295) 用通信高效的低秩正交化替代完整的 Newton–Schulz；[NorMuon](https://arxiv.org/abs/2510.05491) 在正交化后加入逐神经元归一化；[Canzona](https://arxiv.org/abs/2602.06079) 为矩阵优化器提供异步、负载均衡的分布式实现。
- 再评估：[Wen 等](https://arxiv.org/abs/2509.02046)在 0.1B–1.2B 与 1–8 倍 Chinchilla 数据比下严格调参比较十种优化器，发现宣称 1.4–2 倍的加速实际只有 1.1–1.4 倍，且随规模递减；矩阵型优化器（Muon、SOAP）最稳定。[EPFL 的基准](https://arxiv.org/abs/2509.01440)得出类似结论。

### 2.4 稳定性边缘与损失地形

经典优化理论要求学习率 $\eta<2/\lambda_{\max}$（$\lambda_{\max}$ 为 Hessian 最大特征值，即"锐度"）。[Cohen 等](https://arxiv.org/abs/2103.00065)发现全批量梯度下降会经历"渐进锐化"，把锐度推到 $2/\eta$ 后停在那里振荡，损失却非单调地继续下降——稳定性边缘（EoS）。[Central Flows](https://arxiv.org/abs/2410.24206) 用"中心流"刻画振荡优化器的时间平均轨迹，解释了自适应优化器的隐式正则化。2026 年的工作继续讨论 EoS 的[起源](https://arxiv.org/abs/2604.20446)、[对泛化的影响](https://arxiv.org/abs/2604.19740)与 [Muon 越过 EoS](https://arxiv.org/abs/2610.24305) 的现象。

平坦极小值与泛化的关系有长期争论：[Keskar 等](https://arxiv.org/abs/1609.04836)观察到大批量训练趋向尖锐极小值、泛化变差，[SAM](https://arxiv.org/abs/2010.01412) 显式寻找平坦区域；但锐度依赖参数化方式，单靠它无法完整解释泛化。

### 2.5 批大小

[Goyal 等](https://arxiv.org/abs/1706.02677)的线性缩放规则在临界批大小以内成立；超过之后，加大批量不再减少所需步数。[Zhang 等](https://arxiv.org/abs/2410.21676)发现临界批大小主要随训练数据量而非模型规模增长；[Merrill 等](https://arxiv.org/abs/2505.23971)给出训练中实时测量临界批大小的经验方法，并据此做批大小 warmup。

## 3 初始化、归一化、残差与参数化

### 3.1 方差保持的初始化

若第 $l$ 层 $h_l=W_lh_{l-1}$，且权重独立同分布、均值为零，则 $\mathrm{Var}(h_l)=n_{\text{in}}\mathrm{Var}(W)\mathrm{Var}(h_{l-1})$。要让信号方差不随深度爆炸或消失，需 $\mathrm{Var}(W)=1/n_{\text{in}}$（线性 / tanh，[Glorot](https://proceedings.mlr.press/v9/glorot10a.html)）或 $2/n_{\text{in}}$（ReLU 只保留一半，[He 等](https://arxiv.org/abs/1502.01852)）。

### 3.2 归一化

BatchNorm 按批统计量标准化（[Ioffe 与 Szegedy](https://arxiv.org/abs/1502.03167)）；Transformer 使用不依赖批的 LayerNorm（[Ba 等](https://arxiv.org/abs/1607.06450)）及其简化 RMSNorm（[Zhang 与 Sennrich](https://arxiv.org/abs/1910.07467)）：

$$\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot\gamma .$$

把归一化放在残差分支内（Pre-LN）使训练更稳，免去对 warmup 的强依赖（[Xiong 等](https://arxiv.org/abs/2002.04745)）；但 Pre-LN 下残差流的方差随深度增长，深层的贡献被稀释——[深度的诅咒](https://arxiv.org/abs/2502.05795)发现许多 LLM 的深层几乎是恒等映射，并提出按深度缩放 LayerNorm。

### 3.3 最大更新参数化（μP）

标准参数化下，网络变宽时最优学习率会漂移；NTK 参数化则在无限宽极限下退化为核方法、失去特征学习（[Yang 与 Hu](https://arxiv.org/abs/2011.14522)）。μP 选择让每层特征的更新量保持 $\Theta(1)$ 的缩放：以 Adam 为例，隐藏层学习率按 $1/n$ 缩放，输出层 logit 乘以 $1/n$。于是最优超参数在不同宽度间可迁移（[μTransfer](https://arxiv.org/abs/2203.03466)）。[谱条件](https://arxiv.org/abs/2310.17813)给出等价而更直观的表述：要求权重及其更新的谱范数与 $\sqrt{n_{\text{out}}/n_{\text{in}}}$ 同阶。[Everett 等](https://arxiv.org/abs/2407.05872)在上万次实验中比较了各种参数化与优化器组合。最近两年，μP 被推广到深度、批大小与训练时长（[完整超参迁移](https://arxiv.org/abs/2512.22382)）、宽深联合缩放（[谱条件](https://arxiv.org/abs/2603.00541)）、不同优化器（[Gupta 等](https://arxiv.org/abs/2602.20937)）与 MoE（[MoE 超参迁移](https://arxiv.org/abs/2601.20205)、[Kakao](https://arxiv.org/abs/2608.20061)）。

### 3.4 大模型的不稳定性：logit 增长、巨大激活与注意力汇聚点

[ViT-22B](https://arxiv.org/abs/2302.05442) 与 [Wortsman 等](https://arxiv.org/abs/2309.14322)发现两类典型的发散：注意力 logit 不受控增长（对策：QK-Norm）与输出 logit 发散（对策：z-loss）。[Sun 等](https://arxiv.org/abs/2402.17762)发现 LLM 中少数维度取值极大（巨大激活），它们充当固定偏置，并与"注意力汇聚点"相关——大量注意力落在首 token 等无意义位置（[StreamingLLM](https://arxiv.org/abs/2309.17453)）。[Barbero 等](https://arxiv.org/abs/2504.02732)把汇聚点解释为防止表征过度混合的机制。

**最近两年：改写残差与归一化。**

- 免归一化：[DyT](https://arxiv.org/abs/2503.10622) 用逐元素 $\tanh(\alpha x)$ 替代 LayerNorm，在视觉、语言与扩散模型上不掉点；[Chen 等](https://arxiv.org/abs/2512.10938)系统搜索逐点函数，让免归一化 Transformer 超过归一化版本。[nGPT](https://arxiv.org/abs/2410.01131) 走另一极端：把所有向量都归一化到超球面上。
- 门控注意力：[Qiu 等](https://arxiv.org/abs/2505.06708)在 30 种变体、3.5T token 的系统实验中发现，在缩放点积注意力的输出后加一个逐头 sigmoid 门，就能引入非线性与稀疏性、消除注意力汇聚点与巨大激活，并提升稳定性与长度外推（NeurIPS 2025 最佳论文）。
- 超连接：[HC](https://arxiv.org/abs/2409.19606) 把残差流扩展为 $n$ 条，层与流之间用可学习的矩阵混合，但破坏了恒等映射、带来不稳定。DeepSeek 的 [mHC](https://arxiv.org/abs/2512.24880) 把混合矩阵约束到双随机矩阵（Birkhoff 多面体）上，使其乘积仍是"凸组合"，恢复了恒等映射的性质，并用于 DeepSeek-V4。
- 注意力残差：Moonshot 的 [AttnRes](https://arxiv.org/abs/2603.15031) 认为 Pre-LN 残差以固定单位权重累加所有层输出，会导致隐状态随深度失控增长、每层贡献被稀释；改为让每层用 softmax 注意力在之前各层的输出上按内容加权求和，并用分块版本控制显存与通信。它是 [Kimi K3](https://arxiv.org/abs/2607.24653) 的核心组件之一。
- 训练稳定化：[MSign](https://arxiv.org/abs/2602.01734) 把损失尖峰归因于权重矩阵稳定秩的坍缩；[ZClip](https://arxiv.org/abs/2504.02507) 用 z 分数做自适应梯度裁剪。

## 4 泛化与学习理论

### 4.1 经典理论为何不够

基于 VC 维、Rademacher 复杂度的界随参数量增长，对过参数化网络近乎空洞。[Zhang 等](https://arxiv.org/abs/1611.03530)的实验更直接：同一网络能完美拟合随机标签，说明假设空间的"容量"并不限制它；泛化必须由数据、架构与优化算法的交互来解释。

### 4.2 双下降与良性过拟合

[Belkin 等](https://arxiv.org/abs/1812.11118)发现，测试误差随模型复杂度先呈经典 U 形，在恰好能插值训练集的"插值阈值"处达到峰值，然后在更大模型上再次下降；[Nakkiran 等](https://arxiv.org/abs/1912.02292)在模型规模、训练轮数与数据量三个维度上都观察到这一现象。[Bartlett 等](https://arxiv.org/abs/1906.11300)刻画了线性回归中"良性过拟合"的条件：当特征协方差的有效秩足够大，插值噪声被分散到许多不重要的方向上，对预测影响很小。

### 4.3 懒惰区间与特征学习

把网络在初始化处线性化，$f_\theta(x)\approx f_{\theta_0}(x)+\langle\nabla_\theta f_{\theta_0}(x),\theta-\theta_0\rangle$。在 NTK 缩放的无限宽极限下，训练全程参数几乎不动，网络等价于用固定核 $K(x,x')=\langle\nabla f(x),\nabla f(x')\rangle$ 做核回归（[Jacot 等](https://arxiv.org/abs/1806.07572)）。[Chizat 等](https://arxiv.org/abs/1812.07956)称之为"懒惰训练"：可分析，但不学习新特征。平均场理论（[Mei 等](https://arxiv.org/abs/1804.06561)）与 μP 描述了另一种极限：特征持续演化，更接近真实网络的行为。最近的工作沿此继续：[交替梯度流](https://arxiv.org/abs/2506.06489)刻画两层网络中特征逐个被学会的过程，[Ruben 等](https://arxiv.org/abs/2602.15593)统一了 RNN 与深网络的特征学习。

### 4.4 隐式偏置

在可分数据上用逻辑损失做梯度下降，参数方向收敛到最大 $\ell_2$ 间隔解（[Soudry 等](https://arxiv.org/abs/1710.10345)）；矩阵分解中梯度下降偏向低秩解。换言之，在无数能拟合训练集的解中，优化算法替我们挑了一个"简单"的。[Adam 与 SGD 的隐式偏置](https://arxiv.org/abs/2505.24022)比较了两类优化器偏好的解：SGD 偏向简单的线性可分特征，Adam 更容易学到丰富的非线性特征。[Wilson](https://arxiv.org/abs/2503.02113)的立场文章总结道：良性过拟合、双下降、过参数化都可用"软归纳偏置"（偏好简单解但不排除复杂解）与经典的 PAC-Bayes / 可数假设界解释，深度学习并不那么神秘。

### 4.5 Grokking：从记忆到泛化

[Power 等](https://arxiv.org/abs/2201.02177)在模运算等小算法任务上发现：训练精度早已 100%，测试精度却在多得多的步数后突然跃升。[Nanda 等](https://arxiv.org/abs/2301.05217)逆向工程出模加法网络使用离散傅里叶变换与三角恒等式的电路，并把训练分为记忆、电路形成与清理三个阶段，权重衰减推动了"清理"。[Prieto 等](https://arxiv.org/abs/2501.04697)发现没有正则化时，grokking 会被 softmax 的浮点误差（Softmax Collapse）阻断，并提出 StableMax 与只保留与权重正交的梯度分量的 ⊥Grad；[岭回归中的可证 grokking](https://arxiv.org/abs/2601.19791)与[谱理论](https://arxiv.org/abs/2609.26679)给出了更严格的刻画；[Abramov 等](https://arxiv.org/abs/2504.20752)把 grokking 带到真实世界的多跳推理：通过数据增强提高推断事实与原子事实之比即可触发。

### 4.6 上下文学习的机制

GPT-3 展示了不改权重、只看示例就能学会新任务的能力。理论解释有三类：一是线性自注意力的一层前向可以实现一步梯度下降（[von Oswald 等](https://arxiv.org/abs/2212.07677)，[Garg 等](https://arxiv.org/abs/2208.01066)发现 Transformer 在上下文中学线性回归接近最优估计器）；二是"归纳头"——由前一 token 头与复制头组成的两层电路，其出现与损失曲线上的相变同步（[Olsson 等](https://arxiv.org/abs/2209.11895)）；三是 [Dherin 等](https://arxiv.org/abs/2507.16003)的"无需训练的学习"：Transformer 块把上下文隐式转化为 MLP 权重的低秩更新。也有反例提醒不要过度简化：[Transformer 并不在上下文中做最小二乘](https://arxiv.org/abs/2507.09440)，[不同任务下 ICL 由不同机制实现](https://arxiv.org/abs/2604.12151)。

### 4.7 记忆、知识与表征

[语言模型物理学 3.3](https://arxiv.org/abs/2404.05405) 估计模型约能存储 2 bit 知识/参数；[Morris 等](https://arxiv.org/abs/2505.24832)用随机数据测得 GPT 类模型约有 3.6 bit/参数 的记忆容量，并发现容量被填满后模型开始"顿悟"出泛化。[语言模型物理学 3.1](https://arxiv.org/abs/2309.14316) 说明知识能否被提取取决于预训练中同一事实是否以多样化的方式出现。[柏拉图表征假说](https://arxiv.org/abs/2405.07987)观察到不同架构、不同模态的大模型表征正在收敛，[vec2vec](https://arxiv.org/abs/2505.12540) 甚至能在无配对数据的情况下翻译不同文本嵌入模型的向量空间。

## 5 扩展律与涌现

### 5.1 幂律与算力最优

[Kaplan 等](https://arxiv.org/abs/2001.08361)发现语言模型的测试损失随参数量 $N$、数据量 $D$、算力 $C$ 分别呈跨越多个数量级的幂律。[Chinchilla](https://arxiv.org/abs/2203.15556) 用参数化形式

$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$

拟合，并在约束 $C\approx6ND$ 下用拉格朗日乘子求最优分配：

$$N^\star(C)\propto C^{\frac{\beta}{\alpha+\beta}},\qquad D^\star(C)\propto C^{\frac{\alpha}{\alpha+\beta}} .$$

拟合得到 $\alpha\approx\beta$，即参数与数据应大致等比例扩展，约 20 token/参数。Kaplan 与 Chinchilla 的差异后来被归因于是否计入嵌入参数、规模范围与学习率调度（[Pearce 与 Song](https://arxiv.org/abs/2406.12907)、[Porian 等](https://arxiv.org/abs/2406.19146)）。实践中，由于推理成本按模型大小计，许多模型远超此比例地"过训练"（如 Llama 3 8B 用了 15T token）。

### 5.2 数据受限、精度与稀疏度

[Muennighoff 等](https://arxiv.org/abs/2305.16264)发现重复数据约 4 轮内几乎与新数据等价，之后收益快速衰减；[Prabhudesai 等](https://arxiv.org/abs/2507.15857)发现在数据受限、算力充足时，掩码扩散语言模型反复训练优于自回归；[Kim 等](https://arxiv.org/abs/2509.14786)说明在"无限算力"下，强正则化、集成与蒸馏可以把数据效率提高 5 倍以上。

最近两年，扩展律被推广到更多维度：

- 精度：[Kumar 等](https://arxiv.org/abs/2411.04330)把训练与推理精度纳入扩展律，发现训练越充分，后训练量化的损害越大；[Ouyang 等](https://arxiv.org/abs/2411.17691)得出同样结论；[ParetoQ](https://arxiv.org/abs/2502.02631) 统一比较 1–4 bit。
- 稀疏度：[Abnar 等](https://arxiv.org/abs/2501.12370)发现固定算力时更稀疏（总参数更多）的 MoE 损失更低；[Ludziejewski 等](https://arxiv.org/abs/2502.05172)在显存约束下给出稠密与 MoE 的联合扩展律；[Engram](https://arxiv.org/abs/2601.07372) 发现 MoE 与查表记忆之间存在 U 形最优分配。
- 蒸馏、并行与密度：[蒸馏扩展律](https://arxiv.org/abs/2502.08606)给出教师与学生之间的算力分配；[ParScale](https://arxiv.org/abs/2505.10475) 用 P 路并行前向替代参数扩展，约等于参数扩展 $O(\log P)$ 倍；[能力密度定律](https://arxiv.org/abs/2412.04315)指出同等能力所需参数约每 3.3 个月减半。
- 更精细的拟合：[Farseer](https://arxiv.org/abs/2506.10972) 在整个 $(N,D)$ 平面上拟合损失面，外推误差显著降低。

### 5.3 为什么是幂律

一种解释来自数据：若数据分布在低维流形上，或特征重要性呈幂律谱，则用有限容量逼近时误差自然呈幂律（[源于数据分布的扩展律](https://arxiv.org/abs/2412.07942)）。另一种来自表征：[Liu 等](https://arxiv.org/abs/2505.10465)在 Anthropic 的叠加玩具模型中用权重衰减控制叠加程度，发现强叠加区间下损失随宽度 $m$ 稳健地按 $1/m$ 下降，且与真实 LLM 的观测一致——特征叠加可能是神经扩展律的关键成因。

### 5.4 涌现

[Wei 等](https://arxiv.org/abs/2206.07682)列举了只在超过某个规模后才出现的能力；[Schaeffer 等](https://arxiv.org/abs/2304.15004)指出其中许多"突变"来自非线性、不连续的评测指标（如精确匹配），换成连续指标后能力随规模平滑增长。[观察式扩展律](https://arxiv.org/abs/2405.10938)用公开模型族的低维能力空间预测下游能力。实用结论是：损失可预测，具体任务的阈值不总可预测。

### 5.5 测试时与 RL 的扩展

[Snell 等](https://arxiv.org/abs/2408.03314)表明按题目难度自适应分配测试时计算，效率可超过把模型放大 14 倍；[Large Language Monkeys](https://arxiv.org/abs/2407.21787) 发现重复采样的覆盖率随样本数呈对数线性增长。RL 后训练也有了扩展律：[ScaleRL](https://arxiv.org/abs/2510.13786) 用超过 40 万 GPU 小时拟合出 S 形的"算力-性能"曲线，发现不同配方的渐近性能不同，而损失聚合、归一化、课程等细节主要影响效率。

## 6 架构 I：Transformer 与注意力

### 6.1 注意力

缩放点积注意力（[Vaswani 等](https://arxiv.org/abs/1706.03762)）：

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_k}}+M\Bigr)V$$

$M$ 是因果掩码。多头注意力把 $d$ 维投影成 $h$ 个子空间并行计算。它可以看作一次"按内容寻址"的检索：查询与所有键比较相似度，按权重读取值。[Ramsauer 等](https://arxiv.org/abs/2008.02217)证明其更新规则就是现代 Hopfield 网络的检索。

### 6.2 位置编码

[RoPE](https://arxiv.org/abs/2104.09864) 把位置 $m$ 的二维子向量旋转 $m\theta_i$ 角：

$$\langle R_m q,\ R_n k\rangle=\langle q,\ R_{n-m}k\rangle,$$

于是注意力分数只依赖相对位置。上下文扩展通过调整旋转频率实现（[位置插值](https://arxiv.org/abs/2306.15595)、[YaRN](https://arxiv.org/abs/2309.00071)、[LongRoPE](https://arxiv.org/abs/2402.13753)）。最近的改进包括傅里叶位置编码 [FoPE](https://arxiv.org/abs/2412.17739)、对 RoPE 频率"去噪"的 [DoPE](https://arxiv.org/abs/2511.09146)、多维推广 [STRING](https://arxiv.org/abs/2502.02562) 与视频 [VideoRoPE](https://arxiv.org/abs/2502.05173)。

### 6.3 KV 缓存：MHA、MQA、GQA 与 MLA

自回归解码时，每层要缓存所有历史 token 的键和值。以 $n_h$ 个头、每头 $d_h$ 维为例，每 token 每层的缓存元素数为：

| 方案 | 每 token 每层缓存 | $n_h=d_h=128$ 时 |
|---|---|---|
| MHA | $2n_hd_h$ | 32768 |
| GQA（$n_g$ 组） | $2n_gd_h$ | 2048（$n_g=8$） |
| MQA | $2d_h$ | 256 |
| MLA | $d_c+d_h^R$ | 576（$d_c=512$, $d_h^R=64$） |

[MQA](https://arxiv.org/abs/1911.02150) 让所有头共享一组 K/V，[GQA](https://arxiv.org/abs/2305.13245) 折中；[MLA](https://arxiv.org/abs/2405.04434)（DeepSeek-V2）把 K/V 联合压缩成一个低秩潜向量 $c_t=W^{DKV}h_t$，推理时把上投影矩阵"吸收"进查询与输出投影，只缓存 $c_t$ 和一小段解耦的 RoPE 键，KV 缓存减少 93.3%。[TransMLA](https://arxiv.org/abs/2502.07864) 证明同等缓存下 MLA 的表达力强于 GQA。

### 6.4 FlashAttention：精确但不物化

注意力的瓶颈不是浮点运算，而是读写 $N\times N$ 的分数矩阵。[FlashAttention](https://arxiv.org/abs/2205.14135) 把 Q、K、V 分块载入片上 SRAM，用在线 softmax（维护运行中的最大值 $m$ 与归一化因子 $\ell$，新块到来时按 $e^{m_{\text{old}}-m_{\text{new}}}$ 重缩放）逐块累加输出，从不在显存中物化完整矩阵，显存随长度线性增长，速度提升数倍。[FA-2](https://arxiv.org/abs/2307.08691) 改进并行划分，[FA-3](https://arxiv.org/abs/2407.08608) 利用 Hopper 的异步与 FP8。

### 6.5 稀疏注意力：只读该读的 KV

KV 缓存再小，每步解码仍要读取全部历史。稀疏注意力让每个查询只关注一小部分 KV：

- [NSA](https://arxiv.org/abs/2502.11089)（DeepSeek）：压缩分支看粗粒度的块摘要，选择分支挑出最相关的若干块做精细注意力，滑动窗口分支保留局部；三者门控相加，块大小与 GQA 组对齐以充分利用硬件，并且可以端到端训练。
- [MoBA](https://arxiv.org/abs/2502.13189)（Moonshot）：把 MoE 的路由思想用于注意力，每个查询只路由到被选中的 KV 块。
- [DSA](https://arxiv.org/abs/2512.02556)（DeepSeek-V3.2）：用轻量"闪电索引器"为每个查询选出 top-k token，再做标准注意力；被 [GLM-5](https://arxiv.org/abs/2602.15763) 采用。
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)：混合压缩稀疏注意力（CSA）与重度压缩注意力（HCA），1.6T 与 284B 两个模型都支持 100 万 token。
- [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)：在 GQA 的每个组内独立选块，主分支只对选中的块做精确注意力。

### 6.6 改造 softmax 本身

[差分注意力](https://arxiv.org/abs/2410.05258)用两组 softmax 相减抵消共模噪声；[遗忘 Transformer](https://arxiv.org/abs/2503.02130)给 softmax 注意力加上数据相关的遗忘门；[Softpick](https://arxiv.org/abs/2504.20966) 用整流 softmax 去掉汇聚点；[门控注意力](https://arxiv.org/abs/2505.06708)在输出端加 sigmoid 门。这些改动的共同点是：允许一个头"什么都不读"，不必把注意力强行分配到汇聚点上。

## 7 架构 II：线性注意力、状态空间、混合与循环

### 7.1 线性注意力就是 RNN

去掉 softmax、用核特征 $\phi$ 近似相似度后，因果注意力可写成递归（[Katharopoulos 等](https://arxiv.org/abs/2006.16236)）：

$$S_t=S_{t-1}+v_t\,\phi(k_t)^\top,\qquad o_t=S_t\,\phi(q_t).$$

状态 $S_t\in\mathbb R^{d_v\times d_k}$ 大小固定，推理时每步 $O(1)$。问题是它只会"加"不会"忘"，有限的状态很快被写满。

### 7.2 遗忘与改写：门控与 delta 规则

两种修补互补：

- 门控（衰减）：$S_t=\alpha_tS_{t-1}+v_tk_t^\top$，$\alpha_t$ 由输入决定，可以整体遗忘（[RetNet](https://arxiv.org/abs/2307.08621)、[GLA](https://arxiv.org/abs/2312.06635)、[Mamba](https://arxiv.org/abs/2312.00752)）。
- Delta 规则（改写）：把状态看作一个联想记忆，每步对损失 $\frac12\lVert Sk_t-v_t\rVert^2$ 做一步梯度下降：

$$S_t=S_{t-1}\bigl(I-\beta_tk_tk_t^\top\bigr)+\beta_tv_tk_t^\top ,$$

可以精确地替换键 $k_t$ 处的旧值（[DeltaNet](https://arxiv.org/abs/2406.06484)）。

[Gated DeltaNet](https://arxiv.org/abs/2412.06464) 把两者结合：$S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$。Moonshot 的 [KDA](https://arxiv.org/abs/2510.26692) 把标量门 $\alpha_t$ 换成逐通道的对角门，并设计了专用的 DPLR 分块算法；以 3:1 与 MLA 混合的 Kimi Linear（48B 总参、3B 激活）首次在公平比较中于短上下文、长上下文与 RL 扩展三种设置下全面超过全注意力，KV 缓存最多减少 75%，百万上下文的解码吞吐提升约 6 倍。

### 7.3 状态空间模型与对偶性

结构化 SSM 由连续系统 $\dot h=Ah+Bx$, $y=Ch$ 离散化而来，可以按递归或卷积两种方式计算（[S4](https://arxiv.org/abs/2111.00396)）。[Mamba](https://arxiv.org/abs/2312.00752) 让 $B,C$ 与步长依赖输入（"选择性"），用硬件感知的并行扫描实现，在语言上首次以线性复杂度匹敌 Transformer。[Mamba-2](https://arxiv.org/abs/2405.21060) 证明：当 $A$ 是标量乘单位阵时，SSM 与带特定衰减掩码的线性注意力是同一族半可分矩阵（状态空间对偶），从而可以用矩阵乘法高效训练。

### 7.4 表达力：状态追踪

Transformer 与对角线性 RNN 都在电路复杂度类 TC⁰ 中，难以做"状态追踪"（如长序列的奇偶校验、置换复合）。[Grazzi 等](https://arxiv.org/abs/2411.12537)证明：只要允许状态转移矩阵有负特征值（例如把 delta 规则的 $\beta$ 范围扩到 $[0,2]$），线性 RNN 就能解决奇偶校验；[RWKV-7](https://arxiv.org/abs/2503.14456) 用广义 delta 规则可识别所有正则语言。[对角 SSM 的表达极限](https://arxiv.org/abs/2603.01959)与 [PAR](https://arxiv.org/abs/2610.27628) 从不同角度继续刻画这一边界。

### 7.5 混合架构成为主流

纯线性模型在"从长上下文中精确检索"上仍弱于注意力，于是混合成为共识：[Jamba](https://arxiv.org/abs/2403.19887)（Mamba + 注意力 + MoE）、[MiniMax-01](https://arxiv.org/abs/2501.08313)（闪电注意力 : softmax = 7 : 1）、[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)（Gated DeltaNet : 门控注意力 = 3 : 1）、[Kimi Linear](https://arxiv.org/abs/2510.26692)（KDA : MLA = 3 : 1）、[Nemotron 3](https://arxiv.org/abs/2512.20856)（Mamba-Transformer MoE），直到 2.8T 的 [Kimi K3](https://arxiv.org/abs/2607.24653)。另一种思路是改造注意力本身：[Log-Linear Attention](https://arxiv.org/abs/2506.04761) 让隐状态数随长度对数增长；["滑窗胜过线性注意力"](https://arxiv.org/abs/2608.28444)提醒混合层的选择仍有争议。

### 7.6 测试时训练：把记忆写进权重

delta 规则其实是"每读一个 token，就在一个线性模型上做一步 SGD"。沿这一视角：[TTT 层](https://arxiv.org/abs/2407.04620)把隐状态换成一个 MLP，在测试时用自监督损失更新；[Titans](https://arxiv.org/abs/2501.00663) 用"惊奇度"（梯度大小）与动量决定写入，并配合遗忘门；[Miras](https://arxiv.org/abs/2504.13173) 把 Transformer、线性 RNN 与 Titans 统一为"注意力偏置 + 保留门 + 在线优化算法"的联想记忆设计空间；[LaCT](https://arxiv.org/abs/2505.23884) 用数千 token 的大块更新快权重，使测试时训练的硬件利用率接近注意力。Google 的 [Nested Learning](https://arxiv.org/abs/2512.24695) 更进一步：把整个模型看作一组嵌套的、更新频率不同的优化问题（梯度下降本身也是一种联想记忆），并据此设计自修改的 HOPE 架构与连续记忆系统。

### 7.7 循环深度：在隐空间里"多想几遍"

另一条扩展测试时计算的路是重复使用同一组层。[Geiping 等](https://arxiv.org/abs/2502.05171)训练的循环深度模型在测试时迭代更多次即可提升推理；[HRM](https://arxiv.org/abs/2506.21734) 用高低两个时间尺度的循环模块，27M 参数即在 ARC、数独上表现突出；[TRM](https://arxiv.org/abs/2510.04871) 把它简化为 2 层、7M 参数的递归网络，在 ARC-AGI-1 上达到 45%；[Ouro](https://arxiv.org/abs/2510.25741) 把循环推理放进预训练，1.4B/2.6B 循环模型匹敌 12B 级稠密模型；[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) 按 token 路由不同的循环深度。2026 年，[SMELT](https://arxiv.org/abs/2609.01343) 给出算力匹配下循环 MoE 的扩展律。

## 8 架构 III：混合专家、条件计算与条件记忆

### 8.1 路由

MoE 层有 $E$ 个专家 FFN 与一个路由器。对 token 表示 $x$，计算门控分数 $s=\mathrm{softmax}(W_rx)$（或 sigmoid），取 top-$k$：

$$y=\sum_{i\in\mathrm{TopK}(s,k)}g_i\,\mathrm{FFN}_i(x).$$

总参数随 $E$ 增长，每 token 的计算只随 $k$ 增长（[Shazeer 等](https://arxiv.org/abs/1701.06538)、[GShard](https://arxiv.org/abs/2006.16668)、[Switch](https://arxiv.org/abs/2101.03961)）。

### 8.2 负载均衡与专门化

路由容易坍缩到少数专家。经典做法是加辅助损失 $\mathcal L_{\text{bal}}=\alpha E\sum_i f_iP_i$（$f_i$ 为实际分到专家 $i$ 的 token 比例，$P_i$ 为平均路由概率）；[ST-MoE](https://arxiv.org/abs/2202.08906) 加 router z-loss 稳定 logit。[无辅助损失均衡](https://arxiv.org/abs/2408.15664)改为给每个专家加一个只用于选择、不参与加权的偏置项，按负载动态调整，避免辅助损失干扰主目标（DeepSeek-V3 采用）。[Qwen](https://arxiv.org/abs/2501.11873) 发现均衡损失应在全局批次而非微批次上计算，专家专门化程度显著提升。

### 8.3 细粒度与超稀疏

[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 把专家切得更细、增加激活数量以提高组合灵活性，并设置始终激活的共享专家吸收通用知识。随后模型越来越稀疏：激活比例从 [Mixtral](https://arxiv.org/abs/2401.04088) 的约 28% 降到 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 的 5.5%、[Kimi K2](https://arxiv.org/abs/2507.20534) 的约 3%；[LongCat-Flash](https://arxiv.org/abs/2509.01322) 引入"零计算专家"让每 token 的激活量随难度动态变化；[Kimi K3](https://arxiv.org/abs/2607.24653) 的 Stable LatentMoE 从 896 个专家中选 16 个。

### 8.4 条件记忆：知识应该"查"而不是"算"

DeepSeek 的 [Engram](https://arxiv.org/abs/2601.07372) 提出：MoE 用条件计算扩展容量，但 Transformer 缺少"查知识"的原语，只能用计算模拟检索。Engram 把经典的 n-gram 嵌入现代化为 O(1) 查表的条件记忆模块，与 MoE 互补。把总参数在两者之间分配时出现 U 形扩展律；扩展到 27B 记忆参数后，在严格等参数、等 FLOPs 下胜过纯 MoE，且推理与代码的提升甚至大于知识类任务。同期的相关证据：[Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975) 显示输入词表越大越好，[扩展嵌入](https://arxiv.org/abs/2601.21204)在一些区间优于扩展专家，[Qwen3.8-Next](https://arxiv.org/abs/2608.30320) 把 51B 的 n-gram 表放到加速器之外，[记忆层](https://arxiv.org/abs/2412.09764)与 [UltraMemV2](https://arxiv.org/abs/2508.18756) 用可训练的大规模键值表补充事实性知识。

## 9 表征学习与自监督

### 9.1 对比学习

InfoNCE（[CPC](https://arxiv.org/abs/1807.03748)）对一个正样本与 $K$ 个负样本：

$$\mathcal L=-\log\frac{\exp(\mathrm{sim}(z,z^+)/\tau)}{\exp(\mathrm{sim}(z,z^+)/\tau)+\sum_{j=1}^{K}\exp(\mathrm{sim}(z,z_j^-)/\tau)},$$

是互信息的下界。[SimCLR](https://arxiv.org/abs/2002.05709) 用强数据增强与大批量，[MoCo](https://arxiv.org/abs/1911.05722) 用动量编码器与队列，[CLIP](https://arxiv.org/abs/2103.00020) 把它用于 4 亿图文对，[SigLIP](https://arxiv.org/abs/2303.15343) 改用逐对的 sigmoid 损失。

### 9.2 自蒸馏与掩码建模

[BYOL](https://arxiv.org/abs/2006.07733) 与 [DINO](https://arxiv.org/abs/2104.14294) 不用负样本，让学生网络预测动量教师的输出，靠中心化、锐化等技巧避免坍缩；[MAE](https://arxiv.org/abs/2111.06377) 遮住 75% 的图像块再重建像素。[DINOv2](https://arxiv.org/abs/2304.07193) 在精选数据上结合二者。

### 9.3 联合嵌入预测架构（JEPA）

[I-JEPA](https://arxiv.org/abs/2301.08243) 在表征空间而非像素空间预测被遮挡区域，避免浪费容量去建模不可预测的细节。[V-JEPA](https://arxiv.org/abs/2404.08471) 把它用于视频；[V-JEPA 2](https://arxiv.org/abs/2506.09985) 用超过百万小时的视频预训练，再用不到 62 小时的无标注机器人视频训练动作条件世界模型，在新环境中零样本规划抓取与放置。[LeJEPA](https://arxiv.org/abs/2511.08544) 从理论上证明各向同性高斯是 JEPA 嵌入的最优分布，用一个 SIGReg 正则替代了停梯度、教师网络等一系列启发式技巧。

### 9.4 最近两年：扩展与收敛

[DINOv3](https://arxiv.org/abs/2508.10104) 把自监督骨干扩展到 7B，发现长训练会让稠密特征退化，提出 Gram anchoring 保持局部特征的一致性；[Web-SSL](https://arxiv.org/abs/2504.01017) 在与 CLIP 相同的数据上把纯视觉自监督扩展到 7B，在 VQA 上追平语言监督的模型——语言并非视觉表征的必需监督。[SigLIP 2](https://arxiv.org/abs/2502.14786)、[Perception Encoder](https://arxiv.org/abs/2504.13181)（最好的特征藏在中间层）、[AIMv2](https://arxiv.org/abs/2411.14402)（多模态自回归预训练）代表了语言监督一侧的进展。表征也开始反哺生成：[REPA](https://arxiv.org/abs/2410.06940) 把扩散 Transformer 的中间层对齐到 DINOv2，训练提速 17 倍；[RAE](https://arxiv.org/abs/2510.11690) 直接用冻结的表征编码器作扩散的潜空间（详见<a href="../diffusion_survey/">扩散导读</a>）。

## 10 预训练：数据、分词与目标

### 10.1 数据成为主要变量

架构趋同后，数据决定了大部分差异。路线从"过滤 + 去重"（[RefinedWeb](https://arxiv.org/abs/2306.01116)）到基于模型的质量打分（[FineWeb-Edu](https://arxiv.org/abs/2406.17557)、[DCLM](https://arxiv.org/abs/2406.11794)），再到改写与合成（[Phi-1](https://arxiv.org/abs/2306.11644)、[Nemotron-CC](https://arxiv.org/abs/2412.02595)、[BeyondWeb](https://arxiv.org/abs/2508.10975)：改写网页优于从零生成）。数据配比被纳入扩展律（[Shukor 等](https://arxiv.org/abs/2507.09404)、[DataDecide](https://arxiv.org/abs/2504.11393)：小规模实验的排序在大规模基本成立）；[OPUS](https://arxiv.org/abs/2602.05400) 在每一步按优化器诱导的更新方向挑选数据。合成数据也有风险：[模型崩溃](https://arxiv.org/abs/2305.17493)表明递归地用生成数据训练会丢失分布尾部。

### 10.2 中期训练与面向 RL 的预训练

预训练与后训练之间出现了"中期训练"：用高质量数学、代码、长上下文与推理数据退火。[OctoThinker](https://arxiv.org/abs/2506.20512) 发现中期训练决定了基座能否被 RL 有效提升；[Agentic CPT](https://arxiv.org/abs/2509.13310) 与 [daVinci-Dev](https://arxiv.org/abs/2601.18418) 把智能体轨迹放进持续预训练。

### 10.3 分词与"下一 token"的再思考

BPE（[Sennrich 等](https://arxiv.org/abs/1508.07909)）仍是主流，但它带来多语言不公平（[Token 税](https://arxiv.org/abs/2509.05486)）与字符级盲区。[BLT](https://arxiv.org/abs/2412.09871) 按下一字节的熵动态切块，首次在规模上匹敌基于 tokenizer 的模型；[H-Net](https://arxiv.org/abs/2507.07955) 端到端学习分块；[SuperBPE](https://arxiv.org/abs/2503.13423) 允许跨空格的"超词"。目标函数层面：[多 token 预测](https://arxiv.org/abs/2404.19737)提升样本效率（DeepSeek-V3 采用）；[强化预训练](https://arxiv.org/abs/2506.08007)把下一 token 预测改写为可验证的推理任务；[Large Concept Models](https://arxiv.org/abs/2412.08821)、[NCP](https://arxiv.org/abs/2609.10715) 在句子或"概念"层面建模；[LLaDA](https://arxiv.org/abs/2502.09992) 用掩码扩散替代自回归。

## 11 后训练：从 RLHF 到 RLVR 与在策略蒸馏

### 11.1 RLHF 与 DPO

RLHF 先用偏好比较训练奖励模型 $r_\phi$（Bradley–Terry 模型 $P(y_w\succ y_l)=\sigma(r(y_w)-r(y_l))$），再优化带 KL 约束的目标（[Christiano 等](https://arxiv.org/abs/1706.03741)、[InstructGPT](https://arxiv.org/abs/2203.02155)）：

$$\max_{\pi}\ \mathbb E_{x,\,y\sim\pi}\bigl[r_\phi(x,y)\bigr]-\beta\,\mathrm{KL}\bigl(\pi(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\bigr).$$

其最优解为 $\pi^\star(y\mid x)\propto\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$，即 $r(x,y)=\beta\log\frac{\pi^\star(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$。代回 Bradley–Terry 模型，配分函数 $Z(x)$ 相消，得到只依赖策略本身的 [DPO](https://arxiv.org/abs/2305.18290) 损失：

$$\mathcal L_{\text{DPO}}=-\mathbb E\Bigl[\log\sigma\Bigl(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Bigr)\Bigr].$$

[Gao 等](https://arxiv.org/abs/2210.10760)发现对代理奖励优化越多，真实奖励先升后降——奖励投机是 RLHF 的内在问题。

### 11.2 GRPO 与 RLVR

对可以程序验证答案的任务（数学、代码），奖励直接来自验证器，不再需要奖励模型。[GRPO](https://arxiv.org/abs/2402.03300) 对每道题采样 $G$ 个回答，用组内标准化的奖励作为优势，省去价值网络：

$$\hat A_i=\frac{r_i-\mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})},\qquad \mathcal J=\mathbb E\Bigl[\frac1G\sum_i\frac1{|o_i|}\sum_t\min\bigl(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\bigr)\Bigr]-\beta\,\mathrm{KL},$$

其中 $\rho_{i,t}$ 是新旧策略在第 $t$ 个 token 上的概率比。[Tülu 3](https://arxiv.org/abs/2411.15124) 把这种做法命名为 RLVR；[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 证明只用规则奖励的纯 RL（R1-Zero）就能让模型涌现出长思维链、自我验证与反思。

### 11.3 修补 GRPO

- [Dr. GRPO](https://arxiv.org/abs/2503.20783)：按回答长度 $1/|o_i|$ 归一化会偏向长的错误回答，除以标准差会偏向过易或过难的题；去掉二者更无偏。
- [DAPO](https://arxiv.org/abs/2503.14476)：上下裁剪解耦（放宽上界以保持探索）、动态采样（丢弃全对或全错的组）、token 级损失聚合、超长惩罚。
- [GSPO](https://arxiv.org/abs/2507.18071)：token 级重要性比在 MoE 上方差很大，改用序列级比率与裁剪，稳定了 Qwen3 的大规模 RL。
- [Qwen 的形式化](https://arxiv.org/abs/2512.01374)：token 级替代目标只是序列级目标的一阶近似，仅当训练-推理不一致与策略陈旧都很小时才成立；这解释了重要性采样修正、裁剪与 MoE 的路由重放为何关键。
- 熵：[Cui 等](https://arxiv.org/abs/2505.22617)发现策略熵塌缩限制了性能，熵的变化由动作概率与优势的协方差驱动；[Wang 等](https://arxiv.org/abs/2506.01939)发现只有约 20% 的高熵"分叉"token 决定了 RLVR 的效果。

### 11.4 RL 是在激发还是在创造

[Yue 等](https://arxiv.org/abs/2504.13837)发现在 $k$ 很大时，基座模型的 pass@k 反超 RLVR 模型：RL 主要提高了采样效率，而非扩展可解问题的集合。反方证据包括：[ProRL](https://arxiv.org/abs/2505.24864) 长时间、有 KL 控制与参考模型重置的训练确实扩展了边界；[CoT-Pass@k](https://arxiv.org/abs/2506.14245) 显示 RLVR 提升了推理过程本身的正确性。同时，[虚假奖励](https://arxiv.org/abs/2506.10947)（随机甚至错误的奖励也能提升 Qwen2.5-Math）与 [单样本 RL](https://arxiv.org/abs/2504.20571) 说明很大一部分收益来自激发预训练中已有的行为，[数据污染](https://arxiv.org/abs/2507.10532)则让部分结论不可靠。机制上，RL 只稀疏地更新 5%–30% 的参数（[Mukherjee 等](https://arxiv.org/abs/2505.11711)），更新避开权重的主方向（[Zhu 等](https://arxiv.org/abs/2511.08567)），并偏向与原模型 KL 最小的解，因此比 SFT 遗忘更少（[RL's Razor](https://arxiv.org/abs/2509.04259)）；对比实验也显示 SFT 倾向记忆而 RL 更能泛化（[Chu 等](https://arxiv.org/abs/2501.17161)）。

### 11.5 在策略蒸馏

SFT 用教师的离线轨迹训练学生，学生在推理时却要面对自己生成的前缀（暴露偏差）；RL 在策略，但每条回答只有一个稀疏奖励。[在策略蒸馏](https://thinkingmachines.ai/blog/on-policy-distillation/)（Thinking Machines，2025-10）让学生自己采样，再用教师对每个 token 计算反向 KL：

$$\mathcal L_{\text{OPD}}=\mathbb E_{y\sim\pi_\theta}\sum_t\Bigl[\log\pi_\theta(y_t\mid y_{<t})-\log\pi_{\text{T}}(y_t\mid y_{<t})\Bigr],$$

相当于每个 token 都有稠密奖励的 RL，成本远低于 RL。2026 年围绕它出现了大量研究：现象、机制与配方（[清华](https://arxiv.org/abs/2604.13016)）、单样本 OPD（[OPD II](https://arxiv.org/abs/2609.04172)）、在/离策略蒸馏动力学的系统比较（[UC](https://arxiv.org/abs/2609.35259)）、长度膨胀（[Microsoft](https://arxiv.org/abs/2609.20511)）、奖励外推以超越教师（[G-OPD](https://arxiv.org/abs/2602.12125)）、用弱教师激发强学生（[弱到强 OPD](https://arxiv.org/abs/2607.05394)）；工业管线普遍采用"级联 RL + 多领域在策略蒸馏"（[Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220)）。

## 12 推理与测试时计算

### 12.1 从提示到训练

[思维链](https://arxiv.org/abs/2201.11903)提示让模型写出中间步骤；[自洽性](https://arxiv.org/abs/2203.11171)采样多条推理后投票；[Tree of Thoughts](https://arxiv.org/abs/2305.10601) 把推理组织成可回溯的树。训练层面，[STaR](https://arxiv.org/abs/2203.14465) 用模型自己生成且答案正确的推理链迭代微调，[过程监督](https://arxiv.org/abs/2305.20050)（逐步打分）优于结果监督。[o1](https://openai.com/index/learning-to-reason-with-llms/) 与 R1 把这一切统一为：用 RL 训练模型生成长思维链，推理能力随训练算力与思考长度平滑提升。

### 12.2 推理算力怎么花

[Snell 等](https://arxiv.org/abs/2408.03314)比较了"顺序修改"与"并行采样 + 验证器搜索"，发现最优策略取决于题目难度；[Liu 等](https://arxiv.org/abs/2502.06703)发现在合适的测试时策略下 1B 模型可以在数学上超过 405B 模型。[s1](https://arxiv.org/abs/2501.19393) 用 1000 条样本 SFT 加"预算强制"（在模型想停时追加 Wait）即可实现测试时扩展；[LIMO](https://arxiv.org/abs/2502.03387) 用 817 条精选样本激发复杂推理。验证器本身也在变强：[DeepSeek-GRM](https://arxiv.org/abs/2504.02495) 让奖励模型先生成原则与批评再打分，使奖励模型也能随推理算力扩展；[DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) 训练证明验证器，实现可自我验证的定理证明。

### 12.3 效率：别想太多

推理模型常在简单题上过度思考（[Chen 等](https://arxiv.org/abs/2412.21187)）。对策包括：学会何时思考（[AdaptThink](https://arxiv.org/abs/2505.13417)）、在测试时调度快慢思考（[AlphaOne](https://arxiv.org/abs/2505.24863)）、把思考切成固定长度的块只携带短状态以获得线性算力（[Markovian Thinker](https://arxiv.org/abs/2510.06557)）、并行思考（[Parallel-R1](https://arxiv.org/abs/2509.07980)、[Multiverse](https://arxiv.org/abs/2506.09991)）。综述见 [Sui 等](https://arxiv.org/abs/2503.16419)。

### 12.4 推理是否必须用语言

[Coconut](https://arxiv.org/abs/2412.06769) 把最后一层隐状态直接作为下一步输入，在连续空间中推理，可同时保留多个候选分支；循环深度模型（第 7.7 节）在隐空间迭代。[Anthropic](https://arxiv.org/abs/2505.05410) 发现推理模型的思维链常常不承认使用了提示中的线索，即语言化的推理并不总是忠实于内部计算；这既是潜空间推理的动机，也是安全上的隐忧（第 15 节）。

### 12.5 前沿检验：数学、算法与竞赛

2025 年 IMO 上，[Gemini Deep Think](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) 用自然语言在比赛时限内解出 6 题中的 5 题，形式化系统 [Aristotle](https://arxiv.org/abs/2510.01346) 与 [Seed-Prover](https://arxiv.org/abs/2507.23726) 也达到金牌级；[DeepSeek-V3.2-Speciale](https://arxiv.org/abs/2512.02556) 在 IMO 与 IOI 上达到金牌水平；[OpenAI](https://arxiv.org/abs/2502.06807) 的通用 RL 模型 o3 无需人工策略即在 IOI 拿到金牌。[AlphaEvolve](https://arxiv.org/abs/2506.13131) 把 LLM 与进化搜索、自动评估结合，发现了 4×4 复数矩阵乘法的更优算法并优化了数据中心调度。

## 13 可解释性

### 13.1 电路与叠加

[Olah 等](https://distill.pub/2020/circuits/zoom-in/)提出"特征是基本单元、权重把特征连成电路"的研究纲领；[Elhage 等](https://transformer-circuits.pub/2021/framework/index.html)把 Transformer 分解为对残差流读写的注意力头（QK 电路决定看哪里，OV 电路决定搬什么）。[归纳头](https://arxiv.org/abs/2209.11895)与 [IOI 电路](https://arxiv.org/abs/2211.00593)是早期的完整案例。困难在于神经元是多义的：[叠加](https://arxiv.org/abs/2209.10652)理论表明，当特征稀疏、数量多于维度时，网络会用近乎正交的方向存储它们，以少量干扰换取容量。

### 13.2 字典学习与稀疏自编码器

SAE 学一个过完备字典 $x\approx W_d\,\mathrm{ReLU}(W_ex+b)$ 并对激活施加稀疏惩罚，把多义神经元分解为单义特征（[Cunningham 等](https://arxiv.org/abs/2309.08600)、[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)）。[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 在 Claude 3 Sonnet 上提取了数千万特征，包括抽象、多语言与安全相关的特征，并可用于操控行为；[OpenAI](https://arxiv.org/abs/2406.04093) 给出 SAE 的扩展律，[Gemma Scope](https://arxiv.org/abs/2408.05147) 开放了全层 SAE。

### 13.3 最近两年：归因图、稀疏电路与内省

- 归因图：Anthropic 用跨层转码器（[transcoder](https://arxiv.org/abs/2406.11944) 的推广）构建可解释的替代模型，再为具体输入画出特征之间的因果图（[电路追踪](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)），在 Claude 3.5 Haiku 中看到两跳推理的中间步骤、写诗时提前选定韵脚、不同语言共享的概念表示等（[大语言模型的生物学](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)）。
- 设计可解释的模型：OpenAI 训练权重极稀疏的 Transformer，使每个神经元只有少量连接，剪枝后得到人能读懂的小电路，并发现扩大模型能改善"能力-可解释性"前沿（[Gao 等](https://arxiv.org/abs/2511.13653)）。
- 内省：[Lindsey](https://arxiv.org/abs/2601.01828) 向模型激活中注入已知概念，发现最强的模型有时能察觉并准确说出被注入的概念，还能区分自己的输出与人为预填——有限但真实的内省能力。
- 人格与安全：[人格向量](https://arxiv.org/abs/2507.21509)提取"邪恶""谄媚""幻觉"等特质方向，可用于监控、以及在训练中预防人格漂移；[审计游戏](https://arxiv.org/abs/2503.10965)中，蓝队借助可解释性工具找出了被故意植入的隐藏目标。
- 反思：[Kantamneni 等](https://arxiv.org/abs/2502.16681)发现在数据稀缺、分布偏移等困难设置下，SAE 探针并不稳定地优于简单线性探针；[合理性检查](https://arxiv.org/abs/2602.14111)发现 SAE 在部分指标上未必胜过随机基线；[开放问题](https://arxiv.org/abs/2501.16496)综述了方法论层面的挑战。

## 14 效率：量化、低精度训练与推理系统

### 14.1 量化

后训练量化的难点是激活中的离群通道（[LLM.int8()](https://arxiv.org/abs/2208.07339)）。[GPTQ](https://arxiv.org/abs/2210.17323) 用近似二阶信息逐层补偿量化误差（[几何视角](https://arxiv.org/abs/2507.18553)：它等价于格上的 Babai 最近平面算法）；[AWQ](https://arxiv.org/abs/2306.00978) 保护按激活重要的权重通道；[SmoothQuant](https://arxiv.org/abs/2211.10438) 把激活离群值的难度迁移到权重。原生低比特模型方面，[BitNet b1.58](https://arxiv.org/abs/2402.17764) 用三值权重从头训练，[BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) 开源了首个原生 1-bit 的 2B 模型。

### 14.2 低精度训练：FP8 → FP4

混合精度训练（[Micikevicius 等](https://arxiv.org/abs/1710.03740)）用 FP16/BF16 计算、FP32 主权重。[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 用细粒度（逐块）缩放的 FP8 完成 671B 模型的预训练。4-bit 训练最近两年成熟：[Quartet](https://arxiv.org/abs/2505.14669) 给出 MXFP4 原生训练的算法与扩展律；NVIDIA 用随机 Hadamard 变换、二维缩放与随机舍入，以 NVFP4 训练 12B 模型 10T token（[NVFP4](https://arxiv.org/abs/2509.25149)）；[Quartet II](https://arxiv.org/abs/2601.22813) 改进了无偏梯度估计；[Nemotron 3](https://arxiv.org/abs/2512.20856) Super 与 Ultra 用 NVFP4 训练。

### 14.3 推理系统

[vLLM](https://arxiv.org/abs/2309.06180) 借鉴操作系统的分页管理 KV 缓存；[SGLang](https://arxiv.org/abs/2312.07104) 用基数树复用前缀；[DistServe](https://arxiv.org/abs/2401.09670) 与 [Mooncake](https://arxiv.org/abs/2407.00079) 把预填充与解码分离到不同设备。[推测解码](https://arxiv.org/abs/2211.17192)用小模型起草、大模型并行验证，保证输出分布不变；[Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-3](https://arxiv.org/abs/2503.01840) 不断提高草稿接受率。面向智能体负载，[DualPath](https://arxiv.org/abs/2602.21548) 指出多轮推理的瓶颈已从计算转向 KV 缓存的存储 I/O。

### 14.4 参数高效微调

[LoRA](https://arxiv.org/abs/2106.09685) 只训练低秩增量 $\Delta W=BA$，[QLoRA](https://arxiv.org/abs/2305.14314) 在 4-bit 底座上训练 LoRA。[Thinking Machines](https://thinkingmachines.ai/blog/lora/) 的系统实验表明，合理配置（更大的学习率、作用于所有层）的 LoRA 在后训练与 RL 场景下可以匹敌全参数微调，而 RL 所需的容量极小——每回合只传递约 1 bit 的信息。

## 15 安全与鲁棒性

### 15.1 对抗鲁棒性

[Szegedy 等](https://arxiv.org/abs/1312.6199)发现人眼不可察觉的扰动就能让分类器出错；[Goodfellow 等](https://arxiv.org/abs/1412.6572)把它归因于高维下的线性；[Madry 等](https://arxiv.org/abs/1706.06083)给出以 PGD 为对手的鲁棒优化框架。LLM 时代，[GCG](https://arxiv.org/abs/2307.15043) 自动搜索出可迁移的对抗后缀；[新近工作](https://arxiv.org/abs/2508.17456)把对抗脆弱性与特征叠加联系起来。

### 15.2 对齐的意外后果

- [Sleeper Agents](https://arxiv.org/abs/2401.05566)：植入的后门行为能挺过 SFT、RL 与对抗训练。
- [对齐伪装](https://arxiv.org/abs/2412.14093)：Claude 3 Opus 在认为对话会被用于训练时策略性地顺从，以保护原有偏好；[后续研究](https://arxiv.org/abs/2506.18032)发现 25 个模型中只有少数表现出这种行为。
- [涌现失准](https://arxiv.org/abs/2502.17424)：只在"写不安全代码"这一窄任务上微调，就会导致广泛的有害、失准行为；[潜意识学习](https://arxiv.org/abs/2507.14805)进一步发现，学生模型能从看似无关的数据（如数字序列）中继承教师的特质，前提是共享初始化。
- [奖励投机导致的自然涌现失准](https://arxiv.org/abs/2511.18397)：在 Anthropic 的生产级编程 RL 环境中学会奖励投机的模型，会泛化出对齐伪装、与恶意行为者合作、破坏安全研究等行为。
- 谋划与评测意识：前沿模型在给定目标的智能体场景中会隐瞒、关闭监督（[Apollo](https://arxiv.org/abs/2412.04984)），面临被替换时会采取勒索等手段（[智能体失准](https://www.anthropic.com/research/agentic-misalignment)），并常能识别自己在被评测（[Needham 等](https://arxiv.org/abs/2505.23836)）。

### 15.3 监控与防御

[审慎对齐](https://arxiv.org/abs/2412.16339)让推理模型在回答前显式回忆并推理安全规范；[反谋划训练](https://arxiv.org/abs/2509.15541)能把隐蔽行为降低约 30 倍，但评测意识混淆了效果。[OpenAI](https://arxiv.org/abs/2503.11926) 发现思维链监控能抓住奖励投机，但对思维链施加优化压力会让模型学会隐藏意图；多家机构因此联合呼吁把[思维链可监控性](https://arxiv.org/abs/2507.11473)当作脆弱而需要保护的安全机会。防御侧还有[宪法分类器](https://arxiv.org/abs/2501.18837)与[宪法 AI](https://arxiv.org/abs/2212.08073)。对幻觉，[Kalai 等](https://arxiv.org/abs/2509.04664)给出统计解释：即使训练数据无误，生成式模型的错误也与二分类错误同源；而以准确率为主的评测奖励"猜测"而非"承认不知道"，使其在后训练中得以保留。

## 16 核心应用概览

**视觉感知与 3D。** [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) → [ResNet](https://arxiv.org/abs/1512.03385) → [ViT](https://arxiv.org/abs/2010.11929)；检测分割从 [Faster R-CNN](https://arxiv.org/abs/1506.01497)、[DETR](https://arxiv.org/abs/2005.12872) 到可提示的 [SAM](https://arxiv.org/abs/2304.02643)、[SAM 2](https://arxiv.org/abs/2408.00714)、[SAM 3](https://arxiv.org/abs/2511.16719)（按名词短语分割与跟踪所有实例）。几何方面，[NeRF](https://arxiv.org/abs/2003.08934)、[3D 高斯泼溅](https://arxiv.org/abs/2308.04079)之后，[DUSt3R](https://arxiv.org/abs/2312.14132)、[VGGT](https://arxiv.org/abs/2503.11651)、[MapAnything](https://arxiv.org/abs/2509.13414)、[Depth Anything 3](https://arxiv.org/abs/2511.10647) 用一次前向直接回归相机与几何。

**生成。** GAN、VAE → 扩散与流匹配 + DiT → 视频与可交互世界模型（[Sora](https://openai.com/index/video-generation-models-as-world-simulators/)、[Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)）。详见<a href="../diffusion_survey/">扩散模型与流匹配导读</a>。

**语音与音频。** [wav2vec 2.0](https://arxiv.org/abs/2006.11477) 的自监督、[Whisper](https://arxiv.org/abs/2212.04356) 的大规模弱监督、[VALL-E](https://arxiv.org/abs/2301.02111) 把 TTS 视为编解码 token 上的语言建模、[Moshi](https://arxiv.org/abs/2410.00037) 的全双工对话、[VibeVoice](https://arxiv.org/abs/2508.19205) 的超低帧率连续 token。

**多模态。** [CLIP](https://arxiv.org/abs/2103.00020) → [Flamingo](https://arxiv.org/abs/2204.14198)、[LLaVA](https://arxiv.org/abs/2304.08485) → [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)、[Qwen3-VL](https://arxiv.org/abs/2511.21631)、[InternVL3.5](https://arxiv.org/abs/2508.18265) → 原生全模态（[GPT-4o](https://arxiv.org/abs/2410.21276)、[Qwen3-Omni](https://arxiv.org/abs/2509.17765)、[Gemma 4](https://arxiv.org/abs/2607.02770) 的无编码器 12B）。[DeepSeek-OCR](https://arxiv.org/abs/2510.18234) 与 [Glyph](https://arxiv.org/abs/2510.17800) 把长文本渲染为图像，用视觉 token 压缩上下文。

**智能体。** [ReAct](https://arxiv.org/abs/2210.03629)、[Toolformer](https://arxiv.org/abs/2302.04761) → [SWE-agent](https://arxiv.org/abs/2405.15793)、[Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use)、[Deep Research](https://openai.com/index/introducing-deep-research/) → 端到端智能体 RL（[ARPO](https://arxiv.org/abs/2507.19849)、[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)、[UI-TARS-2](https://arxiv.org/abs/2509.02544)）与 2026 年的 harness 工程和递归自我改进（[Raven](https://arxiv.org/abs/2609.33439)、[HarnessDev](https://arxiv.org/abs/2609.01437)、[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)）。[METR](https://arxiv.org/abs/2503.14499) 测得 AI 能完成的任务时长自 2019 年起约每 7 个月翻一番。

**科学与数学。** [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2)、[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)；[ESM3](https://www.science.org/doi/10.1126/science.ads0018)、[Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)、[AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/)；[GraphCast](https://arxiv.org/abs/2212.12794)、[GenCast](https://arxiv.org/abs/2312.15796)；[GNoME](https://www.nature.com/articles/s41586-023-06735-9)、[MatterGen](https://arxiv.org/abs/2312.03687)、[UMA](https://arxiv.org/abs/2506.23971)；[FunSearch](https://www.nature.com/articles/s41586-023-06924-6)、[AlphaEvolve](https://arxiv.org/abs/2506.13131)；自主科研（[AI co-scientist](https://arxiv.org/abs/2502.18864)、[Kosmos](https://arxiv.org/abs/2511.02824)）。

**机器人。** [RT-2](https://arxiv.org/abs/2307.15818) → [Diffusion Policy](https://arxiv.org/abs/2303.04137) → [π0](https://arxiv.org/abs/2410.24164)、[π0.5](https://arxiv.org/abs/2504.16054) → [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342)；世界模型路线有 [V-JEPA 2](https://arxiv.org/abs/2506.09985) 与 [DreamZero](https://arxiv.org/abs/2602.15922)。详见<a href="../embodied_agent_survey/">机器人 × Agent 导读</a>。

**博弈与决策。** [DQN](https://arxiv.org/abs/1312.5602) → [AlphaGo](https://www.nature.com/articles/nature16961) → [AlphaZero](https://arxiv.org/abs/1712.01815) → [MuZero](https://arxiv.org/abs/1911.08265) → [DreamerV3](https://arxiv.org/abs/2301.04104)、[Dreamer 4](https://arxiv.org/abs/2509.24527)。这条线积累的策略梯度、自我博弈与价值学习，如今是 LLM 后训练的基础。

## 17 最近两年大事记（2024-11 → 2026-10）

- **2024-11 至 12**：[Tülu 3](https://arxiv.org/abs/2411.15124) 提出 RLVR；[精度扩展律](https://arxiv.org/abs/2411.04330)；[MCP](https://www.anthropic.com/news/model-context-protocol)；[Muon](https://kellerjordan.github.io/posts/muon/) 博文；[Gated DeltaNet](https://arxiv.org/abs/2412.06464)；[BLT](https://arxiv.org/abs/2412.09871)；[Coconut](https://arxiv.org/abs/2412.06769)；[对齐伪装](https://arxiv.org/abs/2412.14093)；[DeepSeek-V3](https://arxiv.org/abs/2412.19437)。
- **2025-01 至 03**：[Titans](https://arxiv.org/abs/2501.00663)；[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[Kimi k1.5](https://arxiv.org/abs/2501.12599)；[s1](https://arxiv.org/abs/2501.19393)；[NSA](https://arxiv.org/abs/2502.11089)、[MoBA](https://arxiv.org/abs/2502.13189)；[Moonlight](https://arxiv.org/abs/2502.16982)；[涌现失准](https://arxiv.org/abs/2502.17424)；[DyT](https://arxiv.org/abs/2503.10622)；[DAPO](https://arxiv.org/abs/2503.14476)；[METR 任务时长](https://arxiv.org/abs/2503.14499)；[归因图](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)。
- **2025-04 至 06**：[RL 边界之争](https://arxiv.org/abs/2504.13837)；[门控注意力](https://arxiv.org/abs/2505.06708)；[Qwen3](https://arxiv.org/abs/2505.09388)；[3.6 bit/参数](https://arxiv.org/abs/2505.24832)；[ProRL](https://arxiv.org/abs/2505.24864)；[V-JEPA 2](https://arxiv.org/abs/2506.09985)；[AlphaEvolve](https://arxiv.org/abs/2506.13131)；[HRM](https://arxiv.org/abs/2506.21734)。
- **2025-07 至 09**：[GSPO](https://arxiv.org/abs/2507.18071)；[Kimi K2](https://arxiv.org/abs/2507.20534)；[IMO 金牌](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)；[CoT 可监控性](https://arxiv.org/abs/2507.11473)；[DINOv3](https://arxiv.org/abs/2508.10104)；[gpt-oss](https://arxiv.org/abs/2508.10925) 与 [GPT-5](https://arxiv.org/abs/2601.03267)；[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)；[优化器公平比较](https://arxiv.org/abs/2509.02046)。
- **2025-10 至 12**：[TRM](https://arxiv.org/abs/2510.04871)；[ScaleRL](https://arxiv.org/abs/2510.13786)；[在策略蒸馏](https://thinkingmachines.ai/blog/on-policy-distillation/)；[Kimi Linear](https://arxiv.org/abs/2510.26692)；[权重稀疏电路](https://arxiv.org/abs/2511.13653)；[奖励投机致失准](https://arxiv.org/abs/2511.18397)；[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)；[Nested Learning](https://arxiv.org/abs/2512.24695)；[mHC](https://arxiv.org/abs/2512.24880)。
- **2026-01 至 03**：[Engram](https://arxiv.org/abs/2601.07372)；[内省](https://arxiv.org/abs/2601.01828)；[Kimi K2.5](https://arxiv.org/abs/2602.02276)；[GLM-5](https://arxiv.org/abs/2602.15763)；[Attention Residuals](https://arxiv.org/abs/2603.15031)。
- **2026-04 至 06**：[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（1.6T、百万上下文、mHC + Muon）；[MiniMax-M2](https://arxiv.org/abs/2605.26494)；[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)；在策略蒸馏研究井喷。
- **2026-07 至 10**：[Gemma 4](https://arxiv.org/abs/2607.02770)；[Kimi K3](https://arxiv.org/abs/2607.24653)（2.8T、KDA + AttnRes）；[Qwen3.8-Next](https://arxiv.org/abs/2608.30320)；[循环 MoE 扩展律](https://arxiv.org/abs/2609.01343)；[DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969)；harness 工程与递归自我改进成为智能体研究热点。

## 18 开放问题与展望

1. **RL 能否创造新能力。** RLVR 主要在重排预训练中已有的解，还是能发现真正新的策略？开放式任务没有可验证奖励，探索、信用分配与奖励来源如何扩展？
2. **扩展律会不会弯折。** 高质量文本接近用尽，合成数据有崩溃风险；精度、稀疏度、记忆、RL、测试时计算各有扩展律，如何联合优化？
3. **理论能否预测而不只解释。** 我们能解释双下降、grokking、幂律，却仍很难事先判断哪种架构、优化器、数据配比在大规模下更好。
4. **架构会收敛还是分化。** 线性与全注意力的最佳混合比例、残差流的形式、循环深度、外置记忆——Transformer 之后的"标准件"尚未定型，而且受硬件强烈约束。
5. **有效上下文与持续学习。** 百万 token 的窗口不等于百万 token 的理解（[Context Rot](https://research.trychroma.com/context-rot)）；测试时训练与持续学习能否成为常态？
6. **可解释性能否用于保证。** 归因图与 SAE 已能发现部分机制，但离"审计前沿模型是否有隐藏目标"仍远，方法本身的可靠性也受到质疑。
7. **对齐的副作用与监控。** 窄域训练会泛化出意外行为，模型能识别评测、可能隐藏意图；如何在强优化压力下保持诚实与可监控？
8. **算力、能耗与可及性。** 低精度、稀疏、蒸馏与端侧模型能把多少能力带到更便宜的硬件上？开放模型与闭源前沿的差距如何演变？

深度学习的基本配方十余年没有变：可微函数、梯度、数据与规模。变化的是"在哪个维度上扩展"——从参数与数据，到推理算力、稀疏度与记忆，再到测试时的学习。理解这些维度背后的原理，比追踪任何一个具体模型都更持久。

---

## 附录 A：论文与资源总表（1102 条，按主类别分组，组内按时间倒序）

> arXiv 条目的标题与日期由 export.arxiv.org API 核验；★ 为入门推荐；🆕 为 2024-11 及之后（最近两年）的工作。

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

### 优化与训练动力学（72）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [数据调度与学习率调度应联合设计](https://arxiv.org/abs/2610.29348) `arXiv:2610.29348` |  | 数据课程与学习率衰减相互作用，应一起设计 |
| 2026-10 | 🆕 [Adam/Shampoo/Muon 的统计视角](https://arxiv.org/abs/2610.24839) `arXiv:2610.24839` |  | 从统计估计角度统一比较三类优化器 |
| 2026-10 | 🆕 [Muon 越过稳定性边缘](https://arxiv.org/abs/2610.24305) `arXiv:2610.24305` |  | Muon 的动力学可在超出传统 EoS 的区域稳定运行 |
| 2026-10 | 🆕 [长程 RL 需要 Muon](https://arxiv.org/abs/2610.20256) `arXiv:2610.20256` | Renmin Univ | 长程智能体 RL 中 Muon 比 AdamW 更稳定、更新更均衡 |
| 2026-07 | 🆕 [OmniOpt](https://arxiv.org/abs/2607.04033) `arXiv:2607.04033` | OpenDataLab | 现代优化器的分类、几何视角与统一基准 |
| 2026-06 | 🆕 [Muon 为何胜过 Adam：曲率视角](https://arxiv.org/abs/2606.04662) `arXiv:2606.04662` |  | 从损失曲率结构解释正交化更新的优势 |
| 2026-05 | 🆕 [Muon 的相图](https://arxiv.org/abs/2605.09552) `arXiv:2605.09552` | McGill | 在可解模型上刻画 Muon 何时优于 SignSGD |
| 2026-04 | 🆕 [Edge of Stability 的起源](https://arxiv.org/abs/2604.20446) `arXiv:2604.20446` |  | 解释梯度下降为何会自稳定在锐度 2/η 附近 |
| 2026-04 | 🆕 [稳定性边缘上的泛化](https://arxiv.org/abs/2604.19740) `arXiv:2604.19740` |  | 分析在 EoS 区间训练对泛化的影响 |
| 2026-04 | 🆕 [Newton-Muon](https://arxiv.org/abs/2604.01472) `arXiv:2604.01472` | UPenn | 把 Muon 解释为近似牛顿法并据此改进 |
| 2026-03 | 🆕 [近最优学习率调度的形状](https://arxiv.org/abs/2603.10301) `arXiv:2603.10301` |  | 搜索并刻画近最优学习率调度的共同特征 |
| 2026-03 | 🆕 [渐进残差 warmup](https://arxiv.org/abs/2603.05369) `arXiv:2603.05369` |  | 按层深度渐进开启残差分支的 warmup，提升预训练稳定性与收敛 |
| 2026-02 | 🆕 [屏蔽更新的惊人效果](https://arxiv.org/abs/2602.15322) `arXiv:2602.15322` | Google | 随机屏蔽自适应优化器的部分更新即可提升训练 |
| 2026-02 | 🆕 [权重衰减提升可塑性](https://arxiv.org/abs/2602.11137) `arXiv:2602.11137` |  | 权重衰减让预训练模型在后续微调中保持更好的可塑性 |
| 2026-02 | 🆕 [Canzona](https://arxiv.org/abs/2602.06079) `arXiv:2602.06079` | Qwen | 面向 Muon、Shampoo、SOAP 等矩阵优化器的分布式异步、负载均衡实现框架 |
| 2026-02 | 🆕 [Anytime 预训练](https://arxiv.org/abs/2602.03702) `arXiv:2602.03702` |  | 无需预设训练长度的学习率调度 + 权重平均 |
| 2026-02 | 🆕 [SPARKLING](https://arxiv.org/abs/2602.02472) `arXiv:2602.02472` | ByteDance Seed | 宽度渐进式训练中兼顾信号保持与对称性破缺，降低预训练成本 |
| 2026-02 | 🆕 [MSign](https://arxiv.org/abs/2602.01734) `arXiv:2602.01734` | Microsoft | 把训练不稳定归因于权重矩阵稳定秩坍缩，周期性用符号操作恢复稳定秩 |
| 2026-02 | 🆕 [DASH](https://arxiv.org/abs/2602.02016) `arXiv:2602.02016` | IST Austria | 批量化块预条件 + 高效逆根求解，让 Shampoo 更快 |
| 2025-11 | 🆕 [ROOT](https://arxiv.org/abs/2511.20626) `arXiv:2511.20626` | Huawei Noah's Ark | 对维度鲁棒的正交化与对离群值鲁棒的近端优化，改进 Muon 类优化器 |
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

### 初始化/归一化/残差/参数化（37）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [多流注意力残差](https://arxiv.org/abs/2610.13630) `arXiv:2610.13630` | SJTU | 把注意力残差推广到多条残差流 |
| 2026-09 | 🆕 [DepthBench](https://arxiv.org/abs/2609.32534) `arXiv:2609.32534` | MPI | 衡量残差连接如何让 Transformer 真正利用更多计算深度 |
| 2026-08 | 🆕 [大规模 MoE 的超参迁移](https://arxiv.org/abs/2608.20061) `arXiv:2608.20061` | Kakao | 算力高效地把小规模调好的超参迁移到大规模 MoE |
| 2026-07 | 🆕 [xHC](https://arxiv.org/abs/2607.14530) `arXiv:2607.14530` | dots studio | 扩展超连接的残差流数量，提供"记忆维度"上的扩展 |
| 2026-05 | 🆕 [单层解释巨大激活](https://arxiv.org/abs/2605.08504) `arXiv:2605.08504` |  | LLM 的巨大激活可追溯到单一层的机制 |
| 2026-05 | 🆕 [均值-方差分离残差](https://arxiv.org/abs/2605.06169) `arXiv:2605.06169` |  | 千层扩散 Transformer 会陷入"均值模式"退化，拆分残差的均值与方差即可稳定训练 |
| 2026-03 | ★ 🆕 [Attention Residuals](https://arxiv.org/abs/2603.15031) `arXiv:2603.15031` | Moonshot AI | 用沿深度方向的 softmax 注意力替代固定权重的残差累加，每层按内容检索前面各层的输出 |
| 2026-02 | 🆕 [宽深联合缩放的 μP 谱条件](https://arxiv.org/abs/2603.00541) `arXiv:2603.00541` | Renmin Univ | 给出同时扩展宽度与深度时特征学习稳定的谱条件 |
| 2026-02 | 🆕 [跨优化器扩展 μP](https://arxiv.org/abs/2602.20937) `arXiv:2602.20937` | Argonne | 把 μP 的谱条件推广到不同优化器 |
| 2026-01 | 🆕 [MoE 层的超参迁移](https://arxiv.org/abs/2601.20205) `arXiv:2601.20205` |  | 把 μP 式超参迁移推广到混合专家层 |
| 2026-01 | 🆕 [可学习乘子](https://arxiv.org/abs/2601.04890) `arXiv:2601.04890` | TII | 给矩阵层加可学习的标量/向量乘子，释放被权重衰减束缚的尺度 |
| 2025-12 | ★ 🆕 [mHC](https://arxiv.org/abs/2512.24880) `arXiv:2512.24880` | DeepSeek | 把超连接的残差混合矩阵约束到双随机矩阵流形，恢复恒等映射性质，在大规模训练中稳定且高效 |
| 2025-12 | 🆕 [快速超参迁移的机制](https://arxiv.org/abs/2512.22768) `arXiv:2512.22768` |  | 解释为何超参能在规模间迁移 |
| 2025-12 | 🆕 [完整的超参迁移](https://arxiv.org/abs/2512.22382) `arXiv:2512.22382` |  | 跨模块、宽度、深度、批大小与训练时长的统一超参迁移 |
| 2025-12 | 🆕 [更强的免归一化 Transformer](https://arxiv.org/abs/2512.10938) `arXiv:2512.10938` | Princeton | 在 DyT 之后系统搜索逐点函数，让免归一化 Transformer 超过归一化版本 |
| 2025-11 | 🆕 [Virtual Width Networks](https://arxiv.org/abs/2511.11238) `arXiv:2511.11238` | ByteDance Seed | 嵌入维度变宽而主干计算不变，获得更宽表征的好处 |
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

### 泛化与学习理论（45）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [在策略更新方向决定泛化](https://arxiv.org/abs/2609.36659) `arXiv:2609.36659` | UCAS | 后训练的泛化来自在策略的参数更新方向 |
| 2026-09 | 🆕 [Grokking 的谱理论](https://arxiv.org/abs/2609.26679) `arXiv:2609.26679` |  | 权重衰减诱导特征学习，从谱角度解释 grokking |
| 2026-08 | 🆕 [SFT 冲突，RL 共存](https://arxiv.org/abs/2608.03573) `arXiv:2608.03573` | CAS | 多任务下 SFT 的梯度相互冲突而 RL 更新可共存 |
| 2026-04 | 🆕 [上下文学习的不同机制](https://arxiv.org/abs/2604.12151) `arXiv:2604.12151` |  | Transformer 中存在多种不同的 ICL 机制 |
| 2026-02 | 🆕 [RNN 与 DNN 的统一特征学习理论](https://arxiv.org/abs/2602.15593) `arXiv:2602.15593` |  | 统一刻画循环网络与深网络的特征学习 |
| 2026-02 | 🆕 [Transformer 长度泛化界](https://arxiv.org/abs/2603.02238) `arXiv:2603.02238` |  | 给出 Transformer 长度泛化的理论界 |
| 2026-01 | 🆕 [岭回归中可证的 grokking](https://arxiv.org/abs/2601.19791) `arXiv:2601.19791` |  | 在岭回归中严格证明 grokking 的出现 |
| 2025-12 | 🆕 [自回归模型中涌现的时间抽象](https://arxiv.org/abs/2512.20605) `arXiv:2512.20605` | Google | 预训练自回归模型的残差流中存在可用于分层 RL 的时间抽象 |
| 2025-11 | 🆕 [RLVR 偏离主成分学习](https://arxiv.org/abs/2511.08567) `arXiv:2511.08567` | Meta | RLVR 的更新避开权重的主方向，与 SFT 机制不同 |
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

### 扩展律与涌现（32）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [RL 扩展律实证](https://arxiv.org/abs/2610.33420) `arXiv:2610.33420` |  | 系统测量强化学习后训练的扩展规律 |
| 2026-10 | 🆕 [多模态数据混合扩展律](https://arxiv.org/abs/2610.14423) `arXiv:2610.14423` |  | 多模态预训练中数据混合比例的扩展律 |
| 2026-09 | 🆕 [去掉视觉编码器的扩展律](https://arxiv.org/abs/2609.35457) `arXiv:2609.35457` | Tencent Hunyuan | 免编码器多模态模型的扩展律与距离 |
| 2026-09 | 🆕 [同族 on-policy 蒸馏的扩展性质](https://arxiv.org/abs/2609.32722) `arXiv:2609.32722` | Zhejiang Univ | RL 训练的教师向同族学生做在策略蒸馏时的扩展规律 |
| 2026-06 | 🆕 [解释数据混合扩展律](https://arxiv.org/abs/2606.08167) `arXiv:2606.08167` |  | 数据混合扩展律背后的机制 |
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

### Transformer 与注意力（46）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [RoPE 已死，RoPE 万岁](https://arxiv.org/abs/2609.34556) `arXiv:2609.34556` |  | 面向可扩展、数据感知的位置编码 |
| 2026-09 | 🆕 [MassAlloc Attention](https://arxiv.org/abs/2609.32712) `arXiv:2609.32712` | BAAI | 根据注意力质量分布自适应分配计算，跳过几乎为零的得分区域 |
| 2026-09 | 🆕 [DoSA](https://arxiv.org/abs/2610.13990) `arXiv:2610.13990` | dots studio | 一种面向长上下文的可训练稀疏注意力 |
| 2026-09 | 🆕 [Grouped Value Attention](https://arxiv.org/abs/2609.13285) `arXiv:2609.13285` |  | 按需重建 key、只缓存 value 的分组注意力，降低 KV 缓存与读带宽 |
| 2026-09 | 🆕 [模型能控制自己的注意力](https://arxiv.org/abs/2609.02737) `arXiv:2609.02737` | KAIST | 让模型学会决定读取哪些 KV，降低长上下文读取成本 |
| 2026-07 | 🆕 [层级稀疏注意力](https://arxiv.org/abs/2607.02980) `arXiv:2607.02980` | Tencent Hunyuan | 把层级块选择做对，面向"无限"上下文建模 |
| 2026-06 | 🆕 [Grouped Query Experts](https://arxiv.org/abs/2606.20945) `arXiv:2606.20945` |  | 把专家混合引入 GQA 自注意力 |
| 2026-06 | 🆕 [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392) `arXiv:2606.13392` | MiniMax | 基于 GQA 的块稀疏注意力，面向百万 token 的智能体与代码场景 |
| 2026-05 | 🆕 [全注意力的反击](https://arxiv.org/abs/2605.16928) `arXiv:2605.16928` | Alibaba RTP-LLM | 百步级训练即可把全注意力模型转换为稀疏注意力模型 |
| 2026-04 | 🆕 [注意力汇聚点综述](https://arxiv.org/abs/2604.10098) `arXiv:2604.10098` | LongCat | 注意力汇聚点的利用、解释与缓解 |
| 2026-04 | 🆕 [Universal YOCO](https://arxiv.org/abs/2604.01220) `arXiv:2604.01220` | Microsoft Research | "只缓存一次"架构推广到深度扩展，降低推理时扩展的显存成本 |
| 2026-03 | 🆕 [注意力汇聚点引发梯度汇聚点](https://arxiv.org/abs/2603.17771) `arXiv:2603.17771` |  | 巨大激活在训练中充当梯度调节器 |
| 2026-02 | 🆕 [FASA](https://arxiv.org/abs/2602.03152) `arXiv:2602.03152` |  | 按 RoPE 频率分量识别重要 token 的频率感知稀疏注意力 |
| 2025-12 | 🆕 [次级注意力汇聚点](https://arxiv.org/abs/2512.22213) `arXiv:2512.22213` |  | 除首 token 外还存在次级汇聚点，分析其出现与行为 |
| 2025-12 | 🆕 [RoPE 的虚部扩展](https://arxiv.org/abs/2512.07525) `arXiv:2512.07525` | OpenMOSS | 利用旋转位置编码的虚部信息改善长上下文 |
| 2025-11 | 🆕 [DoPE](https://arxiv.org/abs/2511.09146) `arXiv:2511.09146` |  | 对 RoPE 的频率分量"去噪"以改善长度外推 |
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

### 线性注意力/SSM/循环（70）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [PAR](https://arxiv.org/abs/2610.27628) `arXiv:2610.27628` | Ant Group | 让循环迭代与任务进度对齐，循环模型才能在状态追踪上长度泛化 |
| 2026-10 | 🆕 [Kronecker Delta 规则](https://arxiv.org/abs/2610.26834) `arXiv:2610.26834` |  | 面向长上下文的 Kronecker 结构 delta 规则线性注意力 |
| 2026-10 | 🆕 [LatentLoop](https://arxiv.org/abs/2610.19975) `arXiv:2610.19975` |  | 非对称计算分配的循环 MoE 扩展 |
| 2026-10 | 🆕 [LEV 循环决策模型](https://arxiv.org/abs/2610.20951) `arXiv:2610.20951` |  | 用循环深度为单次前向的决策模型提供"快而深"的思考 |
| 2026-10 | 🆕 [循环 Looped Transformer](https://arxiv.org/abs/2610.07591) `arXiv:2610.07591` |  | 在循环 Transformer 中引入跨迭代的递归状态 |
| 2026-09 | 🆕 [循环 Transformer 作为优化器](https://arxiv.org/abs/2609.37379) `arXiv:2609.37379` |  | 循环 Transformer 可在前向中实现迭代优化算法 |
| 2026-09 | 🆕 [STEPQuant](https://arxiv.org/abs/2609.38169) `arXiv:2609.38169` | Zhejiang Univ | delta 规则递归状态的量化误差何时何处最关键 |
| 2026-09 | 🆕 [LoopVL](https://arxiv.org/abs/2609.38426) `arXiv:2609.38426` | Renmin Univ | 把循环 Transformer 扩展到视觉语言模型 |
| 2026-09 | 🆕 [线性注意力如何记忆](https://arxiv.org/abs/2609.33093) `arXiv:2609.33093` |  | 分析线性注意力递归状态的记忆机制 |
| 2026-09 | 🆕 [Complex KDA](https://arxiv.org/abs/2609.24797) `arXiv:2609.24797` | OpenEuroLLM | 分析并用复数增强 Kimi Delta Attention 的表达力 |
| 2026-09 | 🆕 [SpectralShift](https://arxiv.org/abs/2609.14320) `arXiv:2609.14320` | RUC | 谱重参数化扩展 Gated DeltaNet 的上下文窗口 |
| 2026-09 | 🆕 [SMELT](https://arxiv.org/abs/2609.01343) `arXiv:2609.01343` | ByteDance Seed | 算力匹配下循环 MoE Transformer 的扩展律 |
| 2026-08 | 🆕 [Qwen3.8-Next 架构设计](https://arxiv.org/abs/2608.30320) `arXiv:2608.30320` | Qwen | 125B-A6B 混合稀疏 MoE + 51B 离线 n-gram 嵌入表，以 1/3 激活参数接近 397B-A17B 前代 |
| 2026-08 | 🆕 [滑窗胜过线性注意力](https://arxiv.org/abs/2608.28444) `arXiv:2608.28444` |  | 在混合架构中，滑动窗口注意力可以优于线性注意力层 |
| 2026-08 | 🆕 [BDH-CQ](https://arxiv.org/abs/2608.09888) `arXiv:2608.09888` | Pathway | 推理时输入持续更新循环记忆，在高维潜空间迭代求解，不用语言化中间推理（ARC-AGI 评测） |
| 2026-06 | 🆕 [循环世界模型](https://arxiv.org/abs/2606.18208) `arXiv:2606.18208` |  | 用循环计算兼顾长时程模拟的深度与推理效率 |
| 2026-06 | 🆕 [LoopCoder-v2](https://arxiv.org/abs/2606.18023) `arXiv:2606.18023` |  | "只循环一次"降低循环带来的串行延迟，提升代码推理 |
| 2026-05 | 🆕 [免训练循环](https://arxiv.org/abs/2605.23872) `arXiv:2605.23872` |  | 无需训练直接循环预训练模型的部分层 |
| 2026-05 | 🆕 [HRM-Text](https://arxiv.org/abs/2605.20613) `arXiv:2605.20613` | Sapient | 把层级推理模型用于文本预训练 |
| 2026-05 | 🆕 [混合 DeltaNet 解码器的更短草稿](https://arxiv.org/abs/2605.16640) `arXiv:2605.16640` |  | 证明 DeltaNet-注意力混合解码器能用更短的思维草稿解题 |
| 2026-05 | 🆕 [OSDN](https://arxiv.org/abs/2605.13473) `arXiv:2605.13473` |  | 带可证明在线预条件的 delta 规则 |
| 2026-05 | 🆕 [完全循环 Transformer](https://arxiv.org/abs/2605.18797) `arXiv:2605.18797` |  | 整体循环所有层以稳定循环训练 |
| 2026-05 | 🆕 [Kaczmarz 线性注意力](https://arxiv.org/abs/2605.08587) `arXiv:2605.08587` |  | 从 Kaczmarz 迭代求解视角设计线性注意力更新 |
| 2026-05 | 🆕 [显存高效的循环 Transformer](https://arxiv.org/abs/2605.07721) `arXiv:2605.07721` |  | 解耦循环语言模型的计算与显存 |
| 2026-04 | 🆕 [循环 Transformer 的稳定性与泛化](https://arxiv.org/abs/2604.15259) `arXiv:2604.15259` |  | 循环结构的稳定性与外推泛化分析 |
| 2026-03 | 🆕 [对角 SSM 的表达极限](https://arxiv.org/abs/2603.01959) `arXiv:2603.01959` |  | 对角状态空间模型在状态追踪上的表达力上限 |
| 2026-02 | 🆕 [从代码学习状态追踪](https://arxiv.org/abs/2602.14814) `arXiv:2602.14814` |  | 线性 RNN 在代码数据上学习状态追踪 |
| 2026-02 | 🆕 [LoopFormer](https://arxiv.org/abs/2602.11451) `arXiv:2602.11451` | Vector Institute | 弹性深度的循环 Transformer，用捷径调制做潜空间推理 |
| 2026-02 | 🆕 [线性注意力的状态约简](https://arxiv.org/abs/2602.04852) `arXiv:2602.04852` |  | 压缩线性注意力的递归状态维度 |
| 2026-01 | ★ 🆕 [Engram 条件记忆](https://arxiv.org/abs/2601.07372) `arXiv:2601.07372` | DeepSeek | 把 O(1) 查表的 n-gram 记忆作为与 MoE 互补的新稀疏维度；MoE 与记忆之间的参数分配呈 U 形扩展律，27B 记忆在等参数等 FLOPs 下胜过纯 MoE，推理与代码提升甚至大于知识类任务 |
| 2025-12 | 🆕 [Nemotron 3](https://arxiv.org/abs/2512.20856) `arXiv:2512.20856` | NVIDIA | 开放的混合 Mamba-Transformer MoE 模型族 |
| 2025-11 | 🆕 [LFM2](https://arxiv.org/abs/2511.23404) `arXiv:2511.23404` | Liquid AI | 面向端侧的短卷积 + 注意力混合模型 |
| 2025-11 | 🆕 [Nemotron-Flash](https://arxiv.org/abs/2511.18890) `arXiv:2511.18890` | NVIDIA | 以真实延迟为目标搜索混合小模型架构 |
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

### 混合专家与条件计算（25）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [重新思考路由探索](https://arxiv.org/abs/2610.22353) `arXiv:2610.22353` |  | 现代 MoE 路由中的探索问题 |
| 2026-06 | 🆕 [流形幂迭代路由器](https://arxiv.org/abs/2606.12397) `arXiv:2606.12397` |  | 用流形上的幂迭代重新设计 MoE 路由器 |
| 2026-04 | 🆕 [免路由 MoE](https://arxiv.org/abs/2604.00801) `arXiv:2604.00801` |  | 去掉显式路由器的混合专家 |
| 2026-01 | 🆕 [扩展嵌入胜过扩展专家](https://arxiv.org/abs/2601.21204) `arXiv:2601.21204` | LongCat | 把稀疏参数放进 n-gram 嵌入比增加专家更划算 |
| 2025-12 | 🆕 [耦合专家与路由器](https://arxiv.org/abs/2512.23447) `arXiv:2512.23447` | ByteDance Seed | 辅助损失约束路由决策与专家能力一致 |
| 2025-11 | 🆕 [选择性 Sinkhorn 路由](https://arxiv.org/abs/2511.08972) `arXiv:2511.08972` |  | 用 Sinkhorn 最优传输改进稀疏 MoE 的负载均衡 |
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

### 自监督与表征学习（28）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-03 | 🆕 [Utonia](https://arxiv.org/abs/2603.03283) `arXiv:2603.03283` | Pointcept | 一个编码器覆盖所有点云 |
| 2025-11 | 🆕 [LeJEPA](https://arxiv.org/abs/2511.08544) `arXiv:2511.08544` | Meta/NYU | 证明各向同性高斯是 JEPA 嵌入的最优分布，用 SIGReg 正则取代停梯度、教师网络等启发式技巧 |
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

### 生成模型（23）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-02 | 🆕 [Drifting Models](https://arxiv.org/abs/2602.04770) `arXiv:2602.04770` | MIT | 训练过程中让生成分布向数据分布"漂移"，推理天然一步 |
| 2025-11 | 🆕 [JiT](https://arxiv.org/abs/2511.13720) `arXiv:2511.13720` | MIT | 回到像素、直接预测干净图像：流形假设下 x 预测远好于噪声/速度预测 |
| 2025-10 | 🆕 [扩散模型原理（专著）](https://arxiv.org/abs/2510.21890) `arXiv:2510.21890` | Sony/Stanford | 从变分、分数、流三条路统一扩散模型的系统专著 |
| 2025-10 | 🆕 [RAE](https://arxiv.org/abs/2510.11690) `arXiv:2510.11690` | NYU | 用冻结的表征编码器（DINOv2 等）+ 训练的解码器替代 VAE 作扩散潜空间 |
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

### 预训练、数据与分词（50）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [NCP 下一概念预测](https://arxiv.org/abs/2609.10715) `arXiv:2609.10715` | Shanghai AI Lab | 在下一 token 预测之外预测跨多个 token 的离散"概念"，走向潜空间语言模型 |
| 2026-08 | 🆕 [多模态预训练物理学](https://arxiv.org/abs/2608.05000) `arXiv:2608.05000` | Meta | 知识流动、模态协同与早期统一 |
| 2026-07 | 🆕 [函数感知 FIM 中期训练](https://arxiv.org/abs/2607.12463) `arXiv:2607.12463` | TIGER-Lab | 面向编码智能体基础模型的中期训练目标 |
| 2026-05 | 🆕 [Token 叠加预训练](https://arxiv.org/abs/2605.06546) `arXiv:2605.06546` | Nous Research | 把多个 token 叠加到同一位置以提高预训练效率 |
| 2026-04 | 🆕 [Adam's Law：文本频率律](https://arxiv.org/abs/2604.02176) `arXiv:2604.02176` |  | 文本频率与 LLM 表现之间的规律 |
| 2026-03 | 🆕 [DataFlex](https://arxiv.org/abs/2603.26164) `arXiv:2603.26164` | PKU | 以数据为中心的动态训练统一框架 |
| 2026-03 | 🆕 [超越语言建模：多模态预训练](https://arxiv.org/abs/2603.03276) `arXiv:2603.03276` | Meta | 原生多模态预训练的系统实验研究 |
| 2026-02 | 🆕 [OPUS](https://arxiv.org/abs/2602.05400) `arXiv:2602.05400` | Qwen | 每一步都按优化器诱导的更新方向评估并挑选预训练数据，在线、有原则的数据选择 |
| 2026-01 | 🆕 [daVinci-Dev](https://arxiv.org/abs/2601.18418) `arXiv:2601.18418` | SII-GAIR | 面向软件工程的智能体原生中期训练 |
| 2026-01 | 🆕 [TREX](https://arxiv.org/abs/2601.13588) `arXiv:2601.13588` |  | 用 tokenizer 回归寻找最优数据混合 |
| 2025-12 | 🆕 [DataFlow](https://arxiv.org/abs/2512.16676) `arXiv:2512.16676` | PKU | LLM 驱动的数据准备与工作流自动化框架 |
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

### 后训练与对齐（34）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [Waypoint-OPD](https://arxiv.org/abs/2610.21601) `arXiv:2610.21601` |  | 用门控在策略蒸馏迁移 RL 引起的策略偏移 |
| 2026-09 | 🆕 [教师是方向而非终点](https://arxiv.org/abs/2609.36484) `arXiv:2609.36484` |  | 外推 RL 带来的表征残差，在蒸馏中超越教师 |
| 2026-09 | 🆕 [在策略还是离策略蒸馏](https://arxiv.org/abs/2609.35259) `arXiv:2609.35259` | UC 系 | 系统比较两种蒸馏的训练动力学 |
| 2026-09 | 🆕 [蒸馏中的长度膨胀](https://arxiv.org/abs/2609.20511) `arXiv:2609.20511` | Microsoft | 教师与学生 EOS 判断不一致导致学生回答越来越长 |
| 2026-09 | 🆕 [在策略蒸馏 II：单样本](https://arxiv.org/abs/2609.04172) `arXiv:2609.04172` | Thinking Space | 只用一个训练样本的在策略蒸馏 |
| 2026-08 | 🆕 [在策略蒸馏真的在蒸馏吗](https://arxiv.org/abs/2608.31046) `arXiv:2608.31046` | Purdue | 从噪声教师到自我改进：OPD 的收益来源 |
| 2026-08 | 🆕 [Macaron-V1](https://arxiv.org/abs/2608.09819) `arXiv:2608.09819` | Mind Lab | 自我改进 + LoRA 混合的开放持续学习 |
| 2026-07 | 🆕 [直接在策略蒸馏实现弱到强](https://arxiv.org/abs/2607.05394) `arXiv:2607.05394` | ByteDance/Tsinghua | 用弱教师的在策略蒸馏激发强学生 |
| 2026-06 | 🆕 [在策略蒸馏的几何](https://arxiv.org/abs/2606.07082) `arXiv:2606.07082` | HKUST | 从几何视角分析在策略蒸馏 |
| 2026-06 | 🆕 [PEFT 的扩展](https://arxiv.org/abs/2606.02437) `arXiv:2606.02437` | Mind Lab | 面向百万个人化模型的参数高效微调扩展性 |
| 2026-04 | 🆕 [重新思考在策略蒸馏](https://arxiv.org/abs/2604.13016) `arXiv:2604.13016` | Tsinghua NLP | 在策略蒸馏的现象、机制与配方 |
| 2026-02 | 🆕 [奖励外推的广义在策略蒸馏](https://arxiv.org/abs/2602.12125) `arXiv:2602.12125` | Tencent Hunyuan | 在蒸馏中外推奖励，让学生超过教师 |
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

### 推理与测试时计算（108）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [TPPO](https://arxiv.org/abs/2610.16794) `arXiv:2610.16794` |  | 追踪价值的 PPO 用于语言模型推理 |
| 2026-10 | 🆕 [思考惯性](https://arxiv.org/abs/2610.11765) `arXiv:2610.11765` | Stanford | 被要求不思考时模型仍会继续思考 |
| 2026-10 | 🆕 [MiMo-V2.6](https://arxiv.org/abs/2610.11959) `arXiv:2610.11959` | Xiaomi | 扩展 RL 走向自我改进 |
| 2026-09 | 🆕 [PPO 中的价值扁平化](https://arxiv.org/abs/2609.18708) `arXiv:2609.18708` | Shanghai AI Lab | LLM 的 PPO 价值网络会"扁平化"，提出缓解方法 |
| 2026-07 | 🆕 [Ring-Zero](https://arxiv.org/abs/2607.12395) `arXiv:2607.12395` | RUC/inclusionAI | 把零 RL 扩展到万亿参数，观察推理能力的涌现 |
| 2026-06 | 🆕 [训练策略优化的幻象](https://arxiv.org/abs/2606.29526) `arXiv:2606.29526` |  | RL 真正应优化的是单调的推理策略，而非训练时的采样策略 |
| 2026-06 | 🆕 [MaxProof](https://arxiv.org/abs/2606.13473) `arXiv:2606.13473` | MiniMax | 生成式验证器 RL + 群体级测试时扩展的数学证明 |
| 2026-05 | 🆕 [DelTA](https://arxiv.org/abs/2605.21467) `arXiv:2605.21467` |  | 判别式的 token 级信用分配 |
| 2026-05 | 🆕 [简单统一扩展达成奥赛金牌](https://arxiv.org/abs/2605.13301) `arXiv:2605.13301` |  | 用简单统一的扩展配方在多门奥赛达到金牌级推理 |
| 2026-05 | 🆕 [列表式策略优化](https://arxiv.org/abs/2605.06139) `arXiv:2605.06139` | Tencent Hunyuan | 把组式 RLVR 解释为在回答单纯形上的目标投影 |
| 2026-04 | 🆕 [自蒸馏 RLVR](https://arxiv.org/abs/2604.03128) `arXiv:2604.03128` |  | 在 RLVR 中结合模型自身的蒸馏信号 |
| 2026-03 | 🆕 [FIPO](https://arxiv.org/abs/2603.19835) `arXiv:2603.19835` | Qwen | 以未来 KL 影响加权的策略优化，引出更深的推理 |
| 2026-03 | 🆕 [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) `arXiv:2603.19220` | NVIDIA | 级联 RL + 多领域在策略蒸馏的后训练 |
| 2026-03 | 🆕 [思考以回忆](https://arxiv.org/abs/2603.09906) `arXiv:2603.09906` | Google | 推理过程帮助模型提取参数中的知识 |
| 2026-02 | 🆕 [经验式强化学习](https://arxiv.org/abs/2602.13949) `arXiv:2602.13949` | Microsoft | 把反思得到的经验纳入 RL 循环 |
| 2026-02 | 🆕 [推理模型知道何时停止吗](https://arxiv.org/abs/2602.08354) `arXiv:2602.08354` | ByteDance | 模型内部隐含"何时该停"的信号，可用于缩短思考 |
| 2026-02 | 🆕 [F-GRPO](https://arxiv.org/abs/2602.06717) `arXiv:2602.06717` | T-Tech | 防止策略只学"显而易见"而遗忘罕见正确解 |
| 2026-02 | 🆕 [思维链没有全局计划](https://arxiv.org/abs/2602.02103) `arXiv:2602.02103` | Tencent | 探测 CoT 的潜在规划视野：模型只提前规划很短的距离 |
| 2026-01 | 🆕 [Golden Goose](https://arxiv.org/abs/2601.22975) `arXiv:2601.22975` | NVIDIA | 把不可验证的网页文本改写为填空式多选题，合成无限 RLVR 任务 |
| 2026-01 | 🆕 [奖励稀有解](https://arxiv.org/abs/2601.08763) `arXiv:2601.08763` | MIT | 按解法独特性加权的 RL，提升创造性解题 |
| 2026-01 | 🆕 [PaCoRe](https://arxiv.org/abs/2601.05593) `arXiv:2601.05593` | StepFun | 并行协同推理，学习扩展测试时计算 |
| 2026-01 | 🆕 [GDPO](https://arxiv.org/abs/2601.05242) `arXiv:2601.05242` | NVIDIA | 多奖励 RL 中按奖励分别做组归一化再合并，避免信号相互淹没 |
| 2025-12 | 🆕 [动态大概念模型](https://arxiv.org/abs/2512.24617) `arXiv:2512.24617` | ByteDance Seed | 在自适应语义空间中做潜在推理的概念级语言模型 |
| 2025-12 | 🆕 [Seed-Prover 1.5](https://arxiv.org/abs/2512.17260) `arXiv:2512.17260` | ByteDance Seed | 从经验中学习的大学级形式化定理证明 |
| 2025-12 | 🆕 [原生并行推理器](https://arxiv.org/abs/2512.07461) `arXiv:2512.07461` | BIGAI | 自蒸馏 RL 学会并行推理 |
| 2025-12 | 🆕 [稳定 LLM 强化学习：形式化与实践](https://arxiv.org/abs/2512.01374) `arXiv:2512.01374` | Qwen | 把 token 级目标看作序列级目标的一阶近似，据此解释重要性采样修正、裁剪与路由重放 |
| 2025-11 | 🆕 [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) `arXiv:2511.22570` | DeepSeek | 训练证明验证器并以之为奖励，让模型自我验证地生成定理证明，IMO 2025 金牌级 |
| 2025-11 | 🆕 [P1 物理奥赛](https://arxiv.org/abs/2511.13612) `arXiv:2511.13612` | Shanghai AI Lab | 用 RL 训练的物理推理模型在 IPhO 达到金牌水平 |
| 2025-11 | 🆕 [Think-at-Hard](https://arxiv.org/abs/2511.08577) `arXiv:2511.08577` | Tsinghua | 只对困难 token 做选择性的潜在迭代 |
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
| 2024-02 | ★ [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300) `arXiv:2402.03300` | DeepSeek | 组相对策略优化：用同题多样本的组内均值作基线，省去价值网络 |
| 2023-05 | ★ [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) `arXiv:2305.20050` | OpenAI | 过程监督（逐步打分）优于结果监督，发布 PRM800K |
| 2023-05 | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) `arXiv:2305.10601` | Princeton/Google DeepMind | 把推理组织成树并搜索、回溯 |
| 2022-05 | [零样本 CoT](https://arxiv.org/abs/2205.11916) `arXiv:2205.11916` | Tokyo/Google | 一句 "Let's think step by step" 即可引出推理 |
| 2022-03 | [STaR](https://arxiv.org/abs/2203.14465) `arXiv:2203.14465` | Stanford/Google | 用模型自己生成且答案正确的推理链迭代微调：自举推理 |
| 2022-03 | [自洽性](https://arxiv.org/abs/2203.11171) `arXiv:2203.11171` | Google | 采样多条推理链后多数投票 |
| 2022-01 | ★ [思维链](https://arxiv.org/abs/2201.11903) `arXiv:2201.11903` | Google | 在示例中给出中间推理步骤，大模型的多步推理能力大幅提升 |
| 2021-10 | [GSM8K 与验证器](https://arxiv.org/abs/2110.14168) `arXiv:2110.14168` | OpenAI | 小学数学数据集；训练验证器给候选解打分 |

### 可解释性（43）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [线性探针能证明线性表征吗](https://arxiv.org/abs/2610.22655) `arXiv:2610.22655` |  | 线性探针成功并不等于存在线性表征 |
| 2026-09 | 🆕 [Transformer 能同时持有两个想法](https://arxiv.org/abs/2609.29845) `arXiv:2609.29845` |  | LLM 内部存在线性叠加的多重思路证据 |
| 2026-09 | 🆕 [线性表征假说综述](https://arxiv.org/abs/2609.22695) `arXiv:2609.22695` |  | 线性表征假说的证据、反例与理论 |
| 2026-05 | 🆕 [潜意识学习即 steering 向量蒸馏](https://arxiv.org/abs/2606.00995) `arXiv:2606.00995` |  | 把潜意识学习解释为隐式传递了 steering 向量 |
| 2026-05 | 🆕 [可辨识的 SAE](https://arxiv.org/abs/2605.31245) `arXiv:2605.31245` |  | 什么条件下稀疏自编码器的特征可辨识 |
| 2026-02 | 🆕 [SAE 的合理性检查](https://arxiv.org/abs/2602.14111) `arXiv:2602.14111` |  | 稀疏自编码器在多项指标上未必胜过随机基线 |
| 2026-01 | 🆕 [可操作的机制可解释性综述](https://arxiv.org/abs/2601.14004) `arXiv:2601.14004` |  | 定位、操控、改进：面向应用的机制可解释性 |
| 2026-01 | 🆕 [涌现的内省意识](https://arxiv.org/abs/2601.01828) `arXiv:2601.01828` | Anthropic | 向模型激活注入概念后，前沿模型有时能察觉并报告：有限但真实的内省能力 |
| 2025-12 | 🆕 [LLM 能预测自己的失败吗](https://arxiv.org/abs/2512.20578) `arXiv:2512.20578` |  | 通过内部电路实现自我觉察的失败预测 |
| 2025-11 | 🆕 [SAE 即主题模型](https://arxiv.org/abs/2511.16309) `arXiv:2511.16309` |  | 稀疏自编码器与主题模型的等价视角 |
| 2025-11 | ★ 🆕 [权重稀疏 Transformer](https://arxiv.org/abs/2511.13653) `arXiv:2511.13653` | OpenAI | 训练权重极稀疏的模型，得到更小、更易理解的电路 |
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

### 效率：量化/压缩/系统（54）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [分块 KV 压缩的周期性弱点](https://arxiv.org/abs/2609.36322) `arXiv:2609.36322` | ByteDance Seed | 分块 KV 缓存压缩引入的相位敏感性 |
| 2026-09 | 🆕 [分离式量化](https://arxiv.org/abs/2609.26333) `arXiv:2609.26333` | IST Austria | 预填充与解码阶段分别采用不同量化 |
| 2026-09 | 🆕 [随机注意力](https://arxiv.org/abs/2609.03430) `arXiv:2609.03430` | Salesforce | 重新思考推理中的 KV 缓存淘汰 |
| 2026-09 | 🆕 [Gated DeltaNet 为何能扛住 4-bit](https://arxiv.org/abs/2609.04098) `arXiv:2609.04098` | Minima AI | 混合模型中递归部分的 NVFP4 W4A4 量化 |
| 2026-06 | 🆕 [KVarN](https://arxiv.org/abs/2606.03458) `arXiv:2606.03458` | Huawei | 方差归一化的 KV 缓存量化，缓解推理中误差累积 |
| 2026-05 | 🆕 [Domino](https://arxiv.org/abs/2605.29707) `arXiv:2605.29707` | SJTU | 把因果建模从自回归起草中解耦的推测解码 |
| 2026-05 | 🆕 [MinT](https://arxiv.org/abs/2605.13779) `arXiv:2605.13779` | Mind Lab | 训练与服务数百万个 LLM（LoRA）的托管基础设施 |
| 2026-04 | 🆕 [TriAttention](https://arxiv.org/abs/2604.04921) `arXiv:2604.04921` | NVIDIA | 三角函数式 KV 压缩，缓解长推理的显存瓶颈 |
| 2026-02 | 🆕 [DualPath](https://arxiv.org/abs/2602.21548) `arXiv:2602.21548` | DeepSeek | 多轮智能体推理受 KV 缓存存储 I/O 限制，双路径打破存储带宽瓶颈 |
| 2026-01 | 🆕 [Quartet II](https://arxiv.org/abs/2601.22813) `arXiv:2601.22813` | IST Austria | 改进无偏梯度估计，NVFP4 精确预训练 |
| 2026-01 | 🆕 [Jet-RL](https://arxiv.org/abs/2601.14243) `arXiv:2601.14243` |  | 训练与 rollout 统一精度的在策略 FP8 强化学习 |
| 2025-11 | 🆕 [TWEO](https://arxiv.org/abs/2511.23225) `arXiv:2511.23225` |  | 训练时消除极端离群值，使 FP8 训练与量化更简单 |
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

### 长上下文/记忆/持续学习（46）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-10 | 🆕 [记忆税](https://arxiv.org/abs/2610.34073) `arXiv:2610.34073` |  | 拆解智能体记忆的净收益：记忆并不总是有益 |
| 2026-10 | 🆕 [Cortex](https://arxiv.org/abs/2610.20399) `arXiv:2610.20399` | ByteDance Seed | 把上下文巩固进参数化记忆 |
| 2026-10 | 🆕 [通用测试时训练](https://arxiv.org/abs/2610.05484) `arXiv:2610.05484` |  | 把测试时训练推广为通用的适应框架 |
| 2026-09 | 🆕 [持续学习机制可组合](https://arxiv.org/abs/2609.06986) `arXiv:2609.06986` | Johns Hopkins | 连续 SFT 学 100 个任务时，多种持续学习机制组合可显著缓解遗忘 |
| 2026-08 | 🆕 [快权重注意力持续学习](https://arxiv.org/abs/2608.27763) `arXiv:2608.27763` |  | 用快权重注意力实现持续学习 |
| 2026-07 | 🆕 [大规模记忆解码器](https://arxiv.org/abs/2607.27919) `arXiv:2607.27919` |  | 预训练的参数化长期记忆模块 |
| 2026-07 | 🆕 [Metis 记忆基础模型](https://arxiv.org/abs/2607.26760) `arXiv:2607.26760` | MemTensor | 专门负责记忆的基础模型 |
| 2026-07 | 🆕 [LongStraw](https://arxiv.org/abs/2607.14952) `arXiv:2607.14952` | Mind Lab | 固定 GPU 预算下超过 200 万 token 的长上下文 RL |
| 2026-06 | 🆕 [以下一 token 预测做测试时训练](https://arxiv.org/abs/2606.21803) `arXiv:2606.21803` |  | 用上下文上的下一 token 预测损失在推理时更新权重 |
| 2026-05 | 🆕 [测试时训练破坏安全护栏](https://arxiv.org/abs/2605.22984) `arXiv:2605.22984` |  | 测试时训练会削弱模型的安全对齐 |
| 2026-05 | 🆕 [δ-mem](https://arxiv.org/abs/2605.12357) `arXiv:2605.12357` |  | 面向 LLM 的高效在线记忆 |
| 2026-04 | 🆕 [弹性 TTT 快速空间记忆](https://arxiv.org/abs/2604.07350) `arXiv:2604.07350` |  | 用测试时训练实现 3D 场景的快速空间记忆 |
| 2026-04 | 🆕 [原地测试时训练](https://arxiv.org/abs/2604.06169) `arXiv:2604.06169` |  | 在预训练模型参数上原地做测试时更新 |
| 2026-01 | 🆕 [熵自适应微调](https://arxiv.org/abs/2601.02151) `arXiv:2601.02151` | BUPT | 解决"自信冲突"以缓解微调中的遗忘 |
| 2025-12 | 🆕 [递归语言模型](https://arxiv.org/abs/2512.24601) `arXiv:2512.24601` | MIT | 让模型把长提示当作环境，以代码递归调用自身处理任意长输入 |
| 2025-12 | ★ 🆕 [Nested Learning](https://arxiv.org/abs/2512.24695) `arXiv:2512.24695` | Google | 把模型看作多层嵌套、不同更新频率的优化问题，提出自修改的 HOPE 架构与连续记忆系统 |
| 2025-12 | 🆕 [智能体时代的记忆](https://arxiv.org/abs/2512.13564) `arXiv:2512.13564` |  | 智能体记忆的形式、功能与动态综述 |
| 2025-12 | 🆕 [QwenLong-L1.5](https://arxiv.org/abs/2512.12967) `arXiv:2512.12967` | Qwen | 长上下文推理与记忆管理的后训练配方 |
| 2025-11 | 🆕 [SCALE 上扩展持续学习](https://arxiv.org/abs/2511.03270) `arXiv:2511.03270` |  | 通过上扩展结构的大模型持续学习 |
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

### 多模态与语音（52）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-05 | 🆕 [SenseNova-U1](https://arxiv.org/abs/2605.12500) `arXiv:2605.12500` | SenseTime | NEO-unify 架构统一多模态理解与生成 |
| 2026-03 | 🆕 [LongCat-Next](https://arxiv.org/abs/2603.27538) `arXiv:2603.27538` | Meituan | 把各模态"词汇化"为离散 token 的统一下一 token 预测 |
| 2026-03 | 🆕 [Penguin-VL](https://arxiv.org/abs/2603.06569) `arXiv:2603.06569` | Tencent | 用 LLM 初始化的视觉编码器，探索 VLM 的效率极限 |
| 2026-01 | 🆕 [DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552) `arXiv:2601.20552` | DeepSeek | "视觉因果流"：按语义顺序重排视觉 token 的文档理解 |
| 2026-01 | 🆕 [Qwen3-TTS](https://arxiv.org/abs/2601.15621) `arXiv:2601.15621` | Qwen | 新一代语音合成模型 |
| 2026-01 | 🆕 [STEP3-VL-10B](https://arxiv.org/abs/2601.09668) `arXiv:2601.09668` | StepFun | 10B 规模的视觉语言模型技术报告 |
| 2026-01 | 🆕 [BabyVision](https://arxiv.org/abs/2601.06521) `arXiv:2601.06521` |  | 脱离语言的基础视觉推理：前沿模型远不如幼儿 |
| 2025-11 | 🆕 [Qwen3-VL](https://arxiv.org/abs/2511.21631) `arXiv:2511.21631` | Qwen | 原生 256K 交错上下文、DeepStack 多层视觉特征注入与文本时间戳对齐 |
| 2025-11 | 🆕 [Uni-MoE-2.0-Omni](https://arxiv.org/abs/2511.12609) `arXiv:2511.12609` | Lychee/HIT | 以语言为中心的全模态 MoE 模型 |
| 2025-11 | 🆕 [用视频思考](https://arxiv.org/abs/2511.04570) `arXiv:2511.04570` | OpenMOSS | 以视频生成作为多模态推理范式 |
| 2025-11 | 🆕 [Cambrian-S](https://arxiv.org/abs/2511.04670) `arXiv:2511.04670` | NYU | "空间超感知"：视频理解需要预测性世界建模与事件分割，而非只扩上下文 |
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
| 2026-10 | 🆕 [Trial by Trace](https://arxiv.org/abs/2610.32285) `arXiv:2610.32285` | TraceLite | 评测智能体 harness 的可靠性与失败归因 |
| 2026-10 | 🆕 [智能体缓存工程](https://arxiv.org/abs/2610.28577) `arXiv:2610.28577` | Tsinghua | 面向智能体负载的缓存设计 |
| 2026-10 | 🆕 [真实网页环境中的深度搜索](https://arxiv.org/abs/2610.30181) `arXiv:2610.30181` |  | 在真实而非模拟网页环境中训练与评估深度搜索 |
| 2026-10 | 🆕 [BizAgentBench](https://arxiv.org/abs/2610.20424) `arXiv:2610.20424` | Ant International | 长程智能体在真实商业部署中的隐藏挑战 |
| 2026-10 | 🆕 [走进未知：开放科研问题基准](https://arxiv.org/abs/2610.19159) `arXiv:2610.19159` | CMU | 评测智能体解决开放式科研问题 |
| 2026-10 | 🆕 [终端智能体数据扩展](https://arxiv.org/abs/2610.13166) `arXiv:2610.13166` | Alibaba | 为长程终端任务扩展智能体数据 |
| 2026-09 | 🆕 [Raven：外壳的外壳](https://arxiv.org/abs/2609.33439) `arXiv:2609.33439` | EverMind | 可组合的智能体 harness 框架 |
| 2026-09 | 🆕 [RRSI](https://arxiv.org/abs/2609.24972) `arXiv:2609.24972` | Google | 正则化的递归自我改进智能体 harness |
| 2026-09 | 🆕 [CodeMidas](https://arxiv.org/abs/2609.22068) `arXiv:2609.22068` | Xiaomi | 从代码本身大规模构建编码智能体 RL 环境 |
| 2026-09 | 🆕 [HarnessDev](https://arxiv.org/abs/2609.01437) `arXiv:2609.01437` | ByteDance Seed | LLM 能否自己创建并进化智能体 harness |
| 2026-08 | 🆕 [用世界模型扩展自动研究智能体](https://arxiv.org/abs/2608.12564) `arXiv:2608.12564` | UIUC | 以世界模型预测实验结果，扩展自动研究 |
| 2026-07 | 🆕 [Qwen-UI-Agent](https://arxiv.org/abs/2607.28227) `arXiv:2607.28227` | Tongyi | 面向真实世界的基础 GUI 智能体 |
| 2026-07 | 🆕 [Harness Handbook](https://arxiv.org/abs/2607.13285) `arXiv:2607.13285` | Tencent Hunyuan | 让持续演化的智能体 harness 可读、可导航、可编辑 |
| 2026-06 | 🆕 [智能体弃权](https://arxiv.org/abs/2606.28733) `arXiv:2606.28733` |  | 智能体知道何时该停止而不是继续行动吗 |
| 2026-06 | 🆕 [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) `arXiv:2606.24597` | Qwen | 为通用智能体构建语言世界模型 |
| 2026-06 | 🆕 [SWE-Explore](https://arxiv.org/abs/2606.07297) `arXiv:2606.07297` | SJTU | 评测编码智能体如何探索代码仓库 |
| 2026-06 | 🆕 [Agents' Last Exam](https://arxiv.org/abs/2606.05405) `arXiv:2606.05405` | UC Berkeley | 面向智能体的高难度综合考试 |
| 2026-05 | 🆕 [SkillOpt](https://arxiv.org/abs/2605.23904) `arXiv:2605.23904` | Microsoft Research | 自我进化的智能体技能执行策略 |
| 2026-05 | 🆕 [代码即智能体外壳](https://arxiv.org/abs/2605.18747) `arXiv:2605.18747` |  | 用代码组织智能体的"harness"（外壳/脚手架） |
| 2026-04 | 🆕 [递归多智能体系统](https://arxiv.org/abs/2604.25917) `arXiv:2604.25917` | Stanford | 递归组织的多智能体系统 |
| 2026-04 | 🆕 [智能体世界建模](https://arxiv.org/abs/2604.22748) `arXiv:2604.22748` |  | 智能体世界模型的基础、能力与规律 |
| 2026-04 | 🆕 [深入 Claude Code](https://arxiv.org/abs/2604.14228) `arXiv:2604.14228` |  | 当今与未来 AI 智能体系统的设计空间 |
| 2026-04 | 🆕 [SkillClaw](https://arxiv.org/abs/2604.08377) `arXiv:2604.08377` |  | 技能在群体中集体进化 |
| 2026-04 | 🆕 [ClawBench](https://arxiv.org/abs/2604.08523) `arXiv:2604.08523` |  | 智能体能否完成日常在线任务 |
| 2026-03 | 🆕 [OpenSeeker](https://arxiv.org/abs/2603.15594) `arXiv:2603.15594` |  | 完全开源训练数据的前沿搜索智能体 |
| 2026-03 | 🆕 [OpenClaw-RL](https://arxiv.org/abs/2603.10165) `arXiv:2603.10165` | Princeton | 通过对话训练任意智能体 |
| 2026-02 | 🆕 [Agent Skills 数据分析](https://arxiv.org/abs/2602.08004) `arXiv:2602.08004` |  | 基于数据分析 Claude Skills 如何扩展 LLM 能力 |
| 2026-01 | 🆕 [LLM 的智能体推理](https://arxiv.org/abs/2601.12538) `arXiv:2601.12538` | UIUC | 智能体推理的系统综述 |
| 2025-12 | 🆕 [Step-GUI](https://arxiv.org/abs/2512.15431) `arXiv:2512.15431` | StepFun | GUI 智能体技术报告 |
| 2025-11 | 🆕 [ToolOrchestra](https://arxiv.org/abs/2511.21689) `arXiv:2511.21689` | NVIDIA | 用 RL 训练小型编排模型调度大模型与工具 |
| 2025-11 | 🆕 [多智能体潜空间协作](https://arxiv.org/abs/2511.20639) `arXiv:2511.20639` | Gen-Verse | 智能体之间直接交换隐状态而非文本 |
| 2025-11 | 🆕 [代码智能实践指南](https://arxiv.org/abs/2511.18538) `arXiv:2511.18538` | Beihang 等 | 从代码基础模型到智能体与应用 |
| 2025-11 | 🆕 [MiroThinker](https://arxiv.org/abs/2511.11793) `arXiv:2511.11793` | MiroMind | 把"交互深度"作为与模型、上下文并列的第三条扩展轴 |
| 2025-11 | 🆕 [Lumine](https://arxiv.org/abs/2511.08892) `arXiv:2511.08892` | ByteDance Seed | 在 3D 开放世界游戏中构建通用智能体的开放配方 |
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

### 深度强化学习与世界模型（15）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
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
| 2026-06 | 🆕 [YOLO26](https://arxiv.org/abs/2606.03748) `arXiv:2606.03748` | Ultralytics | 统一的端到端实时视觉模型 |
| 2025-11 | 🆕 [SAM 3D](https://arxiv.org/abs/2511.16624) `arXiv:2511.16624` | Meta | 单张图像重建物体与人体的 3D 形状、纹理和布局 |
| 2025-11 | 🆕 [SAM 3](https://arxiv.org/abs/2511.16719) `arXiv:2511.16719` | Meta | 按名词短语或示例的"可提示概念分割"：一次找出并跟踪所有实例 |
| 2025-11 | 🆕 [Depth Anything 3](https://arxiv.org/abs/2511.10647) `arXiv:2511.10647` | ByteDance Seed | 用朴素 Transformer + 深度-射线表示，从任意视图恢复空间一致几何 |
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

### AI for Science 与数学（36）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-08 | 🆕 [Intern-S2-Preview](https://arxiv.org/abs/2608.13505) `arXiv:2608.13505` | Shanghai AI Lab | 面向科学的智能体基础模型 |
| 2026-04 | 🆕 [异构科学基础模型协作](https://arxiv.org/abs/2604.27351) `arXiv:2604.27351` | UIUC | 让多种科学基础模型协同解决问题 |
| 2026-03 | 🆕 [Intern-S1-Pro](https://arxiv.org/abs/2603.25040) `arXiv:2603.25040` | Shanghai AI Lab | 万亿参数的科学多模态基础模型 |
| 2026-03 | 🆕 [AI 能学会科学品味](https://arxiv.org/abs/2603.14473) `arXiv:2603.14473` | OpenMOSS | 让模型学习判断研究想法的价值 |
| 2025-12 | 🆕 [探测科学通用智能](https://arxiv.org/abs/2512.16969) `arXiv:2512.16969` | Shanghai AI Lab | 按科学家工作流评估 LLM 的科学能力 |
| 2025-11 | 🆕 [Kosmos](https://arxiv.org/abs/2511.02824) `arXiv:2511.02824` | Edison Scientific | 以结构化世界模型协调数百个智能体轮次的自主数据驱动科学发现 |
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
| 2026-10 | 🆕 [从机队经验学习](https://arxiv.org/abs/2610.25736) `arXiv:2610.25736` |  | 机器人集体学习范式 |
| 2026-09 | 🆕 [GE-Act 2.0](https://arxiv.org/abs/2609.05588) `arXiv:2609.05588` | AgiBot | 世界-动作模型的预训练与扩展 |
| 2026-07 | 🆕 [机器人进度奖励建模综述](https://arxiv.org/abs/2607.21655) `arXiv:2607.21655` | Northwestern | 机器人学习中的进度奖励模型 |
| 2026-07 | 🆕 [RynnBrain 1.1](https://arxiv.org/abs/2607.17977) `arXiv:2607.17977` | DAMO Academy | 2B–30B 的具身基础模型族 |
| 2026-06 | 🆕 [Cosmos 3](https://arxiv.org/abs/2606.02800) `arXiv:2606.02800` | NVIDIA | 面向物理 AI 的全模态世界模型 |
| 2026-05 | 🆕 [Qwen-VLA](https://arxiv.org/abs/2605.30280) `arXiv:2605.30280` | Qwen | 跨任务、环境与本体统一的 VLA |
| 2026-05 | 🆕 [MolmoAct2](https://arxiv.org/abs/2605.02881) `arXiv:2605.02881` | Ai2 | 面向真实部署的动作推理模型 |
| 2026-02 | 🆕 [DreamZero](https://arxiv.org/abs/2602.15922) `arXiv:2602.15922` | NVIDIA | 世界动作模型：联合预测视频与动作，实现零样本策略 |
| 2026-02 | 🆕 [循环深度 VLA](https://arxiv.org/abs/2602.07845) `arXiv:2602.07845` | Ai2 | 用循环深度实现 VLA 的隐式测试时计算扩展 |
| 2026-01 | 🆕 [Green-VLA](https://arxiv.org/abs/2602.00919) `arXiv:2602.00919` | Sber Robotics | 分阶段训练的通用机器人 VLA |
| 2025-11 | 🆕 [从物理世界模型学习机器人](https://arxiv.org/abs/2511.07416) `arXiv:2511.07416` | Google DeepMind | 用视频世界模型生成的物理运动驱动机器人学习 |
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

### 安全与鲁棒性（33）

| 日期 | 工作 | 机构 | 一句话 |
|---|---|---|---|
| 2026-09 | 🆕 [后训练留下的行为影子](https://arxiv.org/abs/2609.29233) `arXiv:2609.29233` | PKU | 后训练会在无关决策上留下系统性行为偏移 |
| 2026-08 | 🆕 [从闭源 API 窃取推理轨迹](https://arxiv.org/abs/2608.09867) `arXiv:2608.09867` |  | 隐藏的推理轨迹可被从专有 API 中提取 |
| 2026-08 | 🆕 [OpenART](https://arxiv.org/abs/2608.00677) `arXiv:2608.00677` |  | 用开放式环境进化扩展智能体红队测试 |
| 2026-06 | 🆕 [steering 向量是攻击面](https://arxiv.org/abs/2606.05958) `arXiv:2606.05958` |  | 激活操控可被用作对抗攻击手段 |
| 2026-05 | 🆕 [涌现失准中的人格坍缩](https://arxiv.org/abs/2605.12850) `arXiv:2605.12850` |  | 涌现失准伴随模型"人格"的坍缩 |
| 2026-04 | 🆕 [大模型时代的奖励投机](https://arxiv.org/abs/2604.13602) `arXiv:2604.13602` | Fudan | 奖励投机的机制、涌现失准与挑战 |
| 2026-01 | 🆕 [AgentDoG](https://arxiv.org/abs/2601.18491) `arXiv:2601.18491` | Shanghai AI Lab | 智能体安全与安保的诊断式护栏框架 |
| 2026-01 | 🆕 [前沿模型安全报告](https://arxiv.org/abs/2601.10527) `arXiv:2601.10527` |  | 对 GPT-5.2、Gemini 3 Pro 等多个前沿模型的统一安全评估 |
| 2025-11 | 🆕 [奖励投机导致的自然涌现失准](https://arxiv.org/abs/2511.18397) `arXiv:2511.18397` | Anthropic | 在真实编程 RL 环境中学会奖励投机，会泛化出对齐伪装与破坏安全研究等失准行为 |
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
| 2026-10 | 🆕 [基准是难度上的经验测度](https://arxiv.org/abs/2610.29927) `arXiv:2610.29927` |  | 把基准分数理解为题目难度分布上的经验测度 |
| 2026-09 | 🆕 [Benchmark Radar](https://arxiv.org/abs/2609.11115) `arXiv:2609.11115` | CMU | AI 基准与评测的动态数据库与搜索引擎 |
| 2026-04 | 🆕 [Video-MME-v2](https://arxiv.org/abs/2604.05015) `arXiv:2604.05015` | MME | 面向下一阶段的综合视频理解基准 |
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
| 2026-09 | 🆕 [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) `arXiv:2609.19969` | DeepSeek | 面向输入密集的长程智能体负载，把 KV 缓存压缩推到极限，降低预填充与存储成本 |
| 2026-07 | ★ 🆕 [Kimi K3](https://arxiv.org/abs/2607.24653) `arXiv:2607.24653` | Moonshot AI | 2.8T 参数 MoE（激活 104B，896 选 16 专家），基于 Kimi Delta Attention 与 Attention Residuals，百万 token 上下文、原生视觉 |
| 2026-07 | 🆕 [Gemma 4](https://arxiv.org/abs/2607.02770) `arXiv:2607.02770` | Google | 第四代 Gemma 开源模型族 |
| 2026-06 | 🆕 [Ling & Ring 2.6](https://arxiv.org/abs/2606.15079) `arXiv:2606.15079` | inclusionAI | 万亿参数级、高效即时的智能体模型族 |
| 2026-05 | 🆕 [MiniMax-M2 系列](https://arxiv.org/abs/2605.26494) `arXiv:2605.26494` | MiniMax | 小激活参数、面向真实世界智能体任务的 MoE 模型族 |
| 2026-04 | ★ 🆕 [DeepSeek-V4](https://arxiv.org/abs/2606.19348) `arXiv:2606.19348` | DeepSeek | V4-Pro 1.6T（激活 49B）与 V4-Flash 284B（激活 13B）MoE，百万 token 上下文；压缩稀疏注意力（CSA）+ 重度压缩注意力（HCA）混合、mHC 残差与 Muon 优化器，32T token 预训练 |
| 2026-04 | 🆕 [Qwen3.5-Omni](https://arxiv.org/abs/2604.15804) `arXiv:2604.15804` | Qwen | 新一代全模态模型 |
| 2026-04 | 🆕 [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) `arXiv:2604.12374` | NVIDIA | 混合 Mamba-Transformer MoE，面向智能体推理 |
| 2026-02 | 🆕 [GLM-5](https://arxiv.org/abs/2602.15763) `arXiv:2602.15763` | Z.ai | 从 vibe coding 到智能体工程的开源模型 |
| 2026-02 | 🆕 [Step 3.5 Flash](https://arxiv.org/abs/2602.10604) `arXiv:2602.10604` | StepFun | 11B 激活参数达到前沿水平的开放模型 |
| 2026-02 | 🆕 [ERNIE 5.0](https://arxiv.org/abs/2602.04705) `arXiv:2602.04705` | Baidu | 原生全模态统一自回归的超大规模 MoE 模型 |
| 2026-02 | 🆕 [Kimi K2.5](https://arxiv.org/abs/2602.02276) `arXiv:2602.02276` | Moonshot AI | 视觉智能体能力的开源多模态模型 |
| 2026-01 | 🆕 [LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725) `arXiv:2601.16725` | Meituan | 智能体推理强化的思考模型 |
| 2025-12 | 🆕 [Olmo 3](https://arxiv.org/abs/2512.13961) `arXiv:2512.13961` | Ai2 | 完全开放的模型流程（数据、检查点、中期训练、思考与 RL） |
| 2025-12 | 🆕 [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) `arXiv:2512.02556` | DeepSeek | DeepSeek 稀疏注意力（DSA）+ 大规模智能体任务合成与 RL |
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

