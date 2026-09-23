我把上一版思路再收紧了一层。**KBS 不能只做“relevance → utility”这一层改写**，因为你现在的 ICLR 本身已经明确把 transfer action 分成 reusable-context evidence、finite-library prior odds 和 query-specific payoff，并且直接写了“高质量 relation score 只回答第一个问题”。 甚至 appendix 已经把 two-context posterior 延伸到了多 context 的 task partition 和 context-cluster routing。

所以我认为 KBS 最稳、故事也更完整的版本应该再跳一级：

> **ICLR 研究“什么时候两个 context 属于可复用关系，以及这种稀有关系如何被监督和校准”；KBS 研究“已经有一池候选知识时，在有限预算下，究竟应该选择哪些知识，而且一旦某些知识已经被选择，剩余知识还有多少新增价值”。**

核心对象从 **relation posterior** 换成 **set-conditioned marginal utility**。
这才足以让两篇文章在问题、方法、理论、实验和结论上真正分家，同时仍然能够复用约 60% 的底层工程。

---

# KBS 最终建议题目

我最推荐：

**Marginal-Utility Routing for Reliable Multi-Context Knowledge-Augmented Prediction**

更抓人一点可以是：

**Relevance Is Not Enough: Marginal-Utility Routing for Knowledge-Augmented Prediction**

或者更标准、更像 KBS：

**Selective Auxiliary-Knowledge Transfer via Set-Conditioned Marginal Utility Estimation**

我倾向第一版。

方法名先不要为了 acronym 强造词，正文里就叫 **Marginal Utility Router, MUR**。短、清楚，而且一眼知道干什么。

---

# 整篇文章真正的中心命题

不是：

> 找到与 query 最相关的 context。

也不是：

> 判断 context 是否与 query 属于同一个 task。

甚至也不只是：

> 判断一个 context 单独看有没有帮助。

而是：

> **The value of auxiliary knowledge is conditional on what the predictor already knows.**

某个 context 单独看可能非常有用，但如果一个几乎重复的信息已经被选入，它的**边际价值**可能接近零。

反过来，一个相似度没有那么高的 context，可能补充当前 selected set 中缺失的信息，从而带来更高的新增预测价值。

于是问题从传统 relevance ranking：

$$
\operatorname{score}(C_j,Q)
$$

变成：

$$
u(j\mid A,Q),
$$

其中 \(A\) 是**当前已经选中的知识集合**。

一句话就可以把全文钉死：

> **Knowledge should be selected by the predictive value it adds to the knowledge already selected, not by its relevance in isolation.**

这就是全文的灵魂。

---

# 为什么这个故事比上一版强

假设 query 有四个候选 context：

$$
C_1,C_2,C_3,C_4.
$$

其中：

* \(C_1\)：非常相关，而且有用；
* \(C_2\)：同样非常相关，但提供的信息几乎和 \(C_1\) 重复；
* \(C_3\)：相关程度一般，但是提供了 \(C_1\) 没有的信息；
* \(C_4\)：看起来相关，但加入后会产生 negative transfer。

传统 relevance router 很容易选择：

$$
C_1,C_2
$$

因为两者 relevance 最高。

但真正理想的是：

$$
C_1,C_3.
$$

关键变化发生在选完 \(C_1\) 之后：

$$
u(C_2\mid\{C_1\})
\ll
u(C_2\mid\emptyset),
$$

而

$$
u(C_3\mid\{C_1\})
$$

仍然很大。

这就是 **marginal utility**。

它同时解释：

* redundancy；
* harmful transfer；
* budget；
* abstention；
* candidate number 增大以后为什么 relevance ranking 越来越差；
* 为什么 soft attention 不一定等于正确的 discrete knowledge selection。

而这些都不是你当前 ICLR 的主问题。

---

# 1. Problem formulation

设：

* \(S_0\)：anchor context；
* \(x\)：query；
* \(y\)：target；
* \(\mathcal C=\{C_1,\ldots,C_K\}\)：候选辅助 contexts；
* \(A\subseteq\{1,\ldots,K\}\)：当前已经选择的 context 集合；
* \(f_A(x)\)：在 \(S_0\) 和 \(A\) 中 contexts 条件下的预测器。

不用任何辅助 context：

$$
f_{\emptyset}(x).
$$

使用集合 \(A\)：

$$
f_A(x).
$$

定义整个集合带来的 utility：

$$
F(A)
=
\mathbb E[
\ell(f_{\emptyset}(x),y)
-
\ell(f_A(x),y)
]
-
\lambda C(A),
$$

其中 \(C(A)\) 可以表示：

* context 数量；
* token cost；
* inference latency；
* retrieval cost。

第一版论文甚至可以简单令：

$$
C(A)=|A|.
$$

---

## 核心定义：marginal utility

对于当前集合 \(A\)，加入候选 \(j\) 的真实边际价值定义为：

$$
m(j\mid A)
=
F(A\cup\{j\})-F(A).
$$

对应到一个训练 episode，可以直接得到 noisy but unbiased 的监督信号：

$$
\tilde m(j\mid A)
=
\ell(f_A(x),y)
-
\ell(f_{A\cup\{j\}}(x),y)
-\lambda c_j.
$$

如果：

$$
m(j\mid A)>0,
$$

说明加入它值得。

如果：

$$
m(j\mid A)<0,
$$

加入它会伤害当前预测或者收益不足以抵消成本。

最终问题就是：

$$
A^*
=
\arg\max_{A:\ |A|\le B}F(A).
$$

这里 \(B\) 是 context budget。

注意这一整套 formulation：

**没有 \(M\)**。

**没有 collision posterior**。

**没有 same-task prior**。

**没有 \(-\log(M-1)\)**。

**不要求 task ID。**

这和 ICLR 的数学对象已经完全不一样。

---

# 2. 为什么 relevance 不是正确的决策变量

这里可以做全文第一个 Proposition，而且很漂亮。

考虑平方损失。

当前预测：

$$
f_A.
$$

加入 context \(j\) 后产生 prediction change：

$$
d_j=f_{A\cup j}-f_A.
$$

设当前 conditional residual 为：

$$
r_A=
\mathbb E[y\mid Q,\mathcal C]-f_A.
$$

则加入 \(j\) 的 conditional predictive utility：

$$
m(j\mid A)
=
\mathbb E[
(f_A-y)^2-(f_A+d_j-y)^2
]
$$

展开：

$$
m(j\mid A)
=
2d_jr_A-d_j^2.
$$

这个公式非常重要。

因为它说明：

> 一个 context 是否有价值，不由“它和 query 多相似”决定，而由它引起的 prediction correction 是否朝正确方向、幅度是否合适决定。

即使 \(C_j\) 与 query 高度相关，如果：

$$
d_j
$$

方向错了或者 correction 过大，也可能：

$$
m(j\mid A)<0.
$$

而且随着 \(A\) 改变：

$$
r_A
$$

也改变。

所以：

$$
m(j\mid A_1)\neq m(j\mid A_2).
$$

这就从理论上解释了为什么**static relevance score 根本不能完整解决 multi-context selection**。

这应该是正文很早出现的结果，不需要复杂证明，却非常有解释力。

---

# 3. MUR：Marginal Utility Router

整个模型不要弄得太复杂。

KBS 往往更吃：

> 问题清楚 + 方法有明确动机 + 实验完整

而不是为了方法复杂而堆模块。

系统由三个部件构成。

### Prediction expert

共享预测器：

$$
f_A(x)
$$

根据 anchor context 和 selected contexts 产生 prediction。

你现在已有的：

* Transformer encoder；
* ridge expert；
* context representation；
* prediction pipeline；

都可以继续复用。

但它不再是论文贡献。

---

### Set encoder

把已经选入的 contexts 编成：

$$
h_A=
\operatorname{SetEnc}
\left(
\{h_i:i\in A\}
\right).
$$

最便宜的实现甚至可以：

$$
h_A=
\frac{1}{|A|}
\sum_{i\in A}h_i
$$

加一个两层 MLP。

正式版本可以使用 DeepSets：

$$
h_A=
\rho\left(
\sum_{i\in A}\phi(h_i)
\right).
$$

这天然 permutation invariant。

---

### Marginal utility head

对于候选 \(j\)，构造：

$$
z_{A,j}
=
[
h_Q,\,
h_A,\,
h_j,\,
|h_A-h_j|,\,
h_A\odot h_j,\,
d_{A,j}
],
$$

其中：

$$
d_{A,j}=f_{A\cup j}-f_A.
$$

然后：

$$
\hat m_{A,j}
=
g_\phi(z_{A,j}).
$$

这一步极其关键。

**router 不直接预测“这个 context 是不是 relevant”。**

它预测：

> 如果我现在已经拥有集合 \(A\)，再使用 context \(j\)，我的 end-task loss 预计会改善多少？

这是一个直接面向决策的 quantity。

---

# 4. 训练标签怎么生成

这是这篇文章非常漂亮的一点：

**不需要额外人工 relation label。**

因为训练阶段本来有 \(y\)。

我们只需要分别计算：

$$
L_A=\ell(f_A(x),y)
$$

和

$$
L_{A+j}
=
\ell(f_{A\cup j}(x),y),
$$

于是：

$$
\tilde m_{A,j}
=
L_A-L_{A+j}.
$$

它天然就是 utility supervision。

训练时随机采样：

$$
A\subset\mathcal C,
\qquad j\notin A.
$$

对于同一个 episode，可以获得很多：

$$
(A,j,\tilde m_{A,j})
$$

训练对。

这让数据利用率很高。

---

# 5. Router 的训练 objective

主 loss 用 Huber，而不是纯 MSE：

$$
\mathcal L_{\rm util}
=
\operatorname{Huber}
(
\hat m_{A,j},
\tilde m_{A,j}
).
$$

原因很好解释：

单 episode loss difference 是 noisy utility observation，Huber 对极端 observation 更稳定。

然后加入 candidate ranking：

假设两个候选 \(i,j\)：

$$
\tilde m_i>\tilde m_j.
$$

希望：

$$
\hat m_i>\hat m_j.
$$

因此：

$$
\mathcal L_{\rm rank}
=
\log
\left(
1+
\exp[
-(\hat m_i-\hat m_j)
\operatorname{sign}(\tilde m_i-\tilde m_j)
]
\right).
$$

总目标：

$$
\mathcal L
=
\mathcal L_{\rm util}
+
\alpha\mathcal L_{\rm rank}.
$$

正文的 primary model 就做到这里。

不要再加一堆：

* uncertainty module；
* Bayesian head；
* contrastive branch；
* consistency loss；
* auxiliary classifier。

会破坏故事。

而且最新 KBS 已经有工作用 uncertainty signature 在 commit / deliberate / retrieval / abstain 之间进行 intervention routing，所以“uncertainty-aware routing”本身已经不是一个足够鲜明的新贡献。([科学直通车][1])

你的区别必须牢牢锁定在：

> **set-conditioned counterfactual predictive utility。**

---

# 6. 推理阶段：Sequential Marginal Utility Routing

初始：

$$
A_0=\emptyset.
$$

第 \(t\) 步计算每一个剩余候选：

$$
\hat m(j\mid A_t).
$$

找：

$$
j_t
=
\arg\max_{j\notin A_t}
\hat m(j\mid A_t).
$$

如果：

$$
\hat m(j_t\mid A_t)\le\tau,
$$

立即停止。

否则：

$$
A_{t+1}
=
A_t\cup\{j_t\}.
$$

直到：

$$
|A|=B
$$

或者没有正 utility candidate。

伪代码实际就十几行。

这个 stop rule 很重要，因为它自然产生：

**abstention from additional knowledge**。

不是说整个 query 不回答，而是：

> 系统认为剩余知识不值得继续使用。

这和普通 selective prediction 的 abstention 又不同。

---

# 7. 为什么必须 set-conditioned，而不是直接预测每个 context 的 utility

这里会成为第二个关键科学问题。

最简单的 baseline：

$$
\hat m_j=g(Q,C_j)
$$

完全忽略当前 \(A\)。

这个叫：

**Static Utility Router**。

你的方法：

$$
\hat m_{A,j}=g(Q,A,C_j).
$$

叫：

**Marginal Utility Router**。

这个 ablation 非常重要。

假设 \(C_1\) 和 \(C_2\) 含有几乎相同的信息：

初始可能：

$$
m(C_1\mid\emptyset)=0.8,
$$

$$
m(C_2\mid\emptyset)=0.75.
$$

选完 \(C_1\)：

$$
m(C_2\mid\{C_1\})=0.05.
$$

而：

$$
m(C_3\mid\{C_1\})=0.45.
$$

Static router：

$$
C_1\rightarrow C_2.
$$

MUR：

$$
C_1\rightarrow C_3.
$$

这一个实验如果做漂亮，几乎就是全文最有说服力的 mechanism result。

---

# 8. 理论部分怎么做

我不建议 KBS 再做 ICLR 那种很重的 Fisher-information/sample-complexity theorem。

那样反而容易让人觉得是同一个理论项目的第二篇。

这里做**决策理论**，短而漂亮。

---

## Proposition 1：relevance does not identify utility

就是前面的：

$$
m(j\mid A)
=
2d_jr_A-d_j^2.
$$

说明 relevance/relatedness 并不能决定 positive transfer。

它是 conceptual theorem。

---

## Theorem 1：one-step routing regret

设对于所有候选：

$$
|
\hat m(j\mid A)-m(j\mid A)
|
\le\epsilon.
$$

预测选择：

$$
\hat j
=
\arg\max_j\hat m(j\mid A).
$$

oracle greedy 选择：

$$
j^*
=
\arg\max_jm(j\mid A).
$$

那么：

$$
m(j^*\mid A)
-
m(\hat j\mid A)
\le2\epsilon.
$$

证明非常短：

$$
m(j^*)
\le
\hat m(j^*)+\epsilon
\le
\hat m(\hat j)+\epsilon
\le
m(\hat j)+2\epsilon.
$$

这建立：

> utility estimation error → routing regret

的直接联系。

---

## Corollary：B-step oracle-greedy regret

最多选择 \(B\) 个 context，那么相对于 exact marginal-utility greedy policy：

$$
R_{\rm MUR}
-
R_{\rm oracle\ greedy}
\le
2B\epsilon.
$$

这非常适合你的方法。

---

## Proposition 2：safe stopping

如果使用 threshold：

$$
\tau=\epsilon,
$$

且：

$$
\max_j\hat m(j\mid A)\le-\epsilon,
$$

那么所有剩余 candidate：

$$
m(j\mid A)\le0.
$$

即 stop decision 不会漏掉具有正 marginal utility 的 candidate。

正式版本可以把 margin 写得更一般。

---

## 可选 Corollary：submodular 情况

这个放 appendix，不要作为主卖点。

如果真实集合 utility：

$$
F(A)
$$

满足 monotone submodularity，那么 exact greedy 有经典 \(1-1/e\) guarantee；结合 utility estimation error，可以导出类似：

$$
F(\hat A)
\ge
(1-1/e)F(A^*)
-
O(B\epsilon).
$$

但我建议正文只写成 corollary。

因为你的真实任务里存在 negative transfer，强行声称所有数据都 monotone/submodular 不合适。

核心理论只要：

> **estimated marginal utility approximation controls routing regret**

就够。

---

# 9. Fig.1 应该是什么

这篇 KBS 的 Fig.1 非常重要，而且绝对不能长得像 ICLR。

当前 ICLR Fig.1 是 Passive / Natural-ID / balanced intervention 随 library size 的 scaling curves，其中心信息就是 \(M^2/M/1\)。

KBS Fig.1 应该做成一个**现象图**。

### 左边：candidate pool

Query \(Q\)

旁边四个候选：

```text
C1   relevance 0.95   initial utility +0.74
C2   relevance 0.92   initial utility +0.68
C3   relevance 0.63   initial utility +0.49
C4   relevance 0.81   initial utility -0.20
```

传统 relevance router：

$$
C_1,C_2.
$$

---

### 中间：选择 C1 后重新评估

```text
                 before C1     after C1
C2 utility          +0.68        +0.06
C3 utility          +0.49        +0.43
C4 utility          -0.20        -0.18
```

于是：

**MUR 选择 \(C_3\)**。

这个动态变化就是文章的新意。

---

### 右边：真实数据图

横轴：

$$
\text{relevance score}
$$

纵轴：

$$
\text{observed marginal utility}
$$

点散开。

尤其强调：

右上：relevant + useful

右下：relevant + harmful

左上：moderately related + useful

左下：irrelevant + useless

再画一条：

$$
u=0
$$

水平线。

图 caption 可以是：

> **Relevance and marginal predictive value answer different questions.** A candidate may appear relevant yet add little after redundant evidence has already been selected, while a less related candidate can provide greater incremental value.

这会是一张非常好的 opening figure。

---

# 10. Fig.2：方法图

非常标准、清楚即可：

```text
                 Candidate pool
            C1   C2   C3   ... CK
             │    │    │       │
             └────shared encoder────┐
                                    │
Anchor S0 ── encoder ───────┐       │
Query x  ───────────────────┤       │
                            ▼       ▼
                         Current selected set A
                                 │
                      prediction f_A
                                 │
       ┌─────────────────────────┼────────────┐
       ▼                         ▼            ▼
    candidate C1             candidate Cj   candidate CK
       │                         │            │
       └──── marginal utility head ──────────┘
                     │
       m̂(1|A), ..., m̂(K|A)
                     │
              Select max > 0
                     │
              Update A / Stop
```

这是方法图，不需要花哨。

---

# 11. Synthetic benchmark 怎么设计

这里不要沿用 ICLR 的 finite-library scaling。

重新构造一个 **multi-source auxiliary-context environment**。

每个 episode：

* anchor context；
* query；
* \(K\) 个 candidate contexts；
* budget \(B\)。

候选包含四类：

### Informative

提供新的 query-relevant information。

### Redundant

与已经有的信息高度重叠。

### Distractor

看起来相似但和预测 target 无关。

### Harmful

会产生有方向性的 prediction bias。

控制：

$$
K\in\{4,8,16,32\}
$$

和：

$$
B\in\{1,2,4\}.
$$

再控制：

* redundancy ratio；
* distractor ratio；
* harmful fraction；
* candidate noise；
* anchor information level。

---

# 12. Synthetic 实验真正要证明什么

不是“大表刷 SOTA”。

而是逐一打穿机制。

### Experiment A：relevance–utility mismatch

保持 relevance detection 很准。

改变辅助 context 的 predictive payoff。

目标现象：

relation/relevance AUC 基本不变，

但：

* always use；
* relevance routing；

出现明显 negative transfer。

MUR 仍然根据 outcome utility 做正确选择。

---

### Experiment B：redundancy

逐渐增加 candidate pool 中 duplicate / near-duplicate contexts 的比例。

如果：

$$
r_{\rm redundant}:0\rightarrow0.8,
$$

static relevance ranking 应该越来越浪费 budget。

关键比较：

$$
\text{static utility}
\quad vs\quad
\text{set-conditioned marginal utility}.
$$

这实际上是文章最重要的 ablation。

---

### Experiment C：candidate pool size

从：

$$
K=4\rightarrow32.
$$

看：

* relevance top-\(B\)；
* MoE；
* static utility；
* MUR。

正常应该看到随着 candidate pool 增大，错误 selection opportunity 变多，而 MUR 更稳定。

---

### Experiment D：budget

横轴：

$$
B=0,1,2,4,8.
$$

纵轴：

task loss。

理想曲线：

Always-pool：

一开始下降，之后因为 harmful/redundant contexts 反弹。

MUR：

下降后自动 plateau，因为会提前 stop。

这是极其漂亮的图。

---

# 13. 真实数据不要照抄 ICLR

你当前 ICLR 已经用了：

* Gaussian；
* sinusoid；
* UCI Electricity；
* Dirichlet–Markov。

因此 KBS 最好：

**一个 synthetic + 三个 real-world multi-context tasks。**

Electricity 可以保留一个，因为基础设施已经有了，但不要让它成为唯一 real dataset。

---

## Dataset 1：Electricity

当前 ICLR 把每个 client 当 latent task，训练、calibration、test clients 和 weeks 分开。

KBS 改成：

给目标 client 的短 anchor week 和 query。

候选池里面：

* 同 client 的不同历史周；
* 相似 client；
* 不同 client；
* seasonally mismatched weeks。

系统选择最多：

$$
B=1,2,4
$$

个 historical contexts。

问题从：

> “两周是不是来自同一个 client？”

彻底变成：

> “当前已有这些历史信息以后，再加入哪个历史片段仍然值得？”

这样同数据也已经是新的 task。

---

## Dataset 2：Traffic

我推荐 PEMS-SF 或类似 multi-sensor traffic dataset。

一个 sensor/time segment 是 query source。

候选 contexts 可以来自：

* 同 sensor 的其他日期；
* 邻近 sensor；
* 类似 traffic profile sensor；
* unrelated sensor。

预测：

future traffic / occupancy。

非常适合展示：

> spatial similarity ≠ marginal utility。

---

## Dataset 3：Human activity / personalized sensing

例如 PAMAP2 这种多 subject sensor activity data。

anchor：

目标 subject 的少量 labeled examples。

candidate：

来自目标 subject 或其他 subjects 的 context blocks。

任务：

activity classification。

这里可以测试：

* 同 subject context；
* 相似运动模式；
* 不同 subject；
* noisy candidate。

它会让论文同时覆盖：

**regression + forecasting + classification。**

这对于 KBS 非常好。

---

# 14. Baselines

Baselines 不需要二十多个。

做强而清楚的一组：

| 类别             | Baseline                |
| -------------- | ----------------------- |
| No transfer    | Base-only               |
| Naive transfer | Pool-all                |
| Retrieval      | Similarity Top-\(B\)    |
| Relation-based | Relevance Router        |
| Soft routing   | Attention / MoE Router  |
| Decision-based | Confidence Router       |
| Utility        | Static Utility Router   |
| Ours           | MUR                     |
| Upper bound    | Oracle Marginal Utility |

这里最重要的不是 MUR vs 一个过时方法。

最重要的是两条：

$$
\text{Relevance Router}
\rightarrow
\text{Static Utility}
\rightarrow
\textbf{MUR}.
$$

分别证明：

1. utility 比 relevance 更接近最终决策；
2. marginal/set-conditioned utility 又比 isolated utility 更正确。

这就是 clean causal ladder。

---

# 15. 主指标

Primary metric：

$$
\text{prediction loss}
$$

或者各数据集对应的 RMSE / accuracy。

但全文需要自己的一套 decision metrics。

### Utility recovery

定义 oracle budget 下可获得的 improvement：

$$
G^*
=
L_{\emptyset}-L_{A^*}.
$$

方法：

$$
G=
L_{\emptyset}-L_{\hat A}.
$$

报告：

$$
\text{Utility Recovery}
=
\frac{G}{G^*}.
$$

synthetic / small K 可以枚举 oracle subset。

大的 K 使用 oracle marginal greedy，明确叫 Oracle-Greedy，不要假装 global oracle。

---

### Negative-transfer rate

$$
\operatorname{NTR}
=
P(
L_{\hat A}>
L_{\emptyset}+\delta
).
$$

\(\delta=0\) 可以作为主结果，正 margin 做 sensitivity。

这可能成为 KBS 最容易理解的指标之一。

---

### Useful-selection precision

在每一步：

$$
P(
m(j_t\mid A_t)>0
).
$$

直接衡量：

> 被 router 加进去的 context 到底有多少真正是有正 marginal benefit 的。

---

### Budget efficiency

$$
\frac{L_\emptyset-L_{\hat A}}
{|\hat A|}
$$

或者直接画：

**performance vs average contexts selected**。

---

### Utility calibration

把：

$$
\hat m
$$

分 bins。

对每个 bin 对比：

$$
E[\tilde m\mid \hat m]
$$

和：

$$
\hat m.
$$

这会给你一个 utility reliability diagram。

很有辨识度。

---

# 16. Main Table

最终主表建议长这样：

| Method            | Electricity ↓ | Traffic ↓ | Activity ↑ | Negative Transfer ↓ | Utility Recovery ↑ | Avg. Contexts ↓ |
| ----------------- | ------------: | --------: | ---------: | ------------------: | -----------------: | --------------: |
| Base-only         |               |           |            |                   0 |                  0 |               0 |
| Pool-all          |               |           |            |                     |                    |               K |
| Similarity        |               |           |            |                     |                    |                 |
| Relevance Router  |               |           |            |                     |                    |                 |
| MoE               |               |           |            |                     |                    |                 |
| Confidence Router |               |           |            |                     |                    |                 |
| Static Utility    |               |           |            |                     |                    |                 |
| **MUR**           |               |           |            |                     |                    |                 |
| Oracle-Greedy     |               |           |            |                     |                1.0 |                 |

注意：

Base-only 的 negative-transfer rate 定义上为 0，所以不能单独拿 NTR 宣称 MUR “最好”。

真正要展示的是：

> 在获得实质 prediction gain 的同时，把 harmful transfer 压低。

因此最好再配一张 **gain–harm Pareto plot**。

---

# 17. Fig.3

Synthetic mechanism figure。

我会做四个 panel：

**A. Candidate redundancy increases**

纵轴：test loss。

**B. Candidate pool \(K\) increases**

纵轴：utility recovery。

**C. Budget increases**

纵轴：prediction gain。

**D. Negative-transfer rate**

横轴：average contexts used。

这张图负责证明方法机制。

---

# 18. Fig.4

真实数据的 **gain–budget curves**。

三个 dataset，每一个 dataset 一条曲线系列：

* Pool-all；
* Similarity；
* Relevance；
* Static Utility；
* MUR；
* Oracle。

横轴：

$$
\text{average selected contexts}
$$

纵轴：

$$
\text{prediction improvement}.
$$

MUR 应该表现为：

> 在相同 context budget 下更高，或者达到相同 gain 时需要更少 context。

这正是 decision-support/system 类 KBS 很喜欢的效果。

KBS 当前 scope 本身就明确包括 machine-learning methodology、prediction systems 和 intelligent decision-support systems，并强调理论与实践并重，因此这个 framing 比再写一篇纯 ICL 理论稿更贴 venue。([Elsevier商店][2])

---

# 19. Fig.5

做 mechanism / interpretability。

挑一个真实 episode：

开始：

$$
A=\emptyset.
$$

显示候选 utility：

```text
C1  +0.81
C2  +0.74
C3  +0.48
C4  -0.10
```

选 C1。

重新计算：

```text
C2  +0.07
C3  +0.41
C4  -0.13
```

于是第二个选择 C3。

选完 C3：

```text
C2  -0.02
C4  -0.09
```

stop。

旁边展示 static relevance 仍然会选择 C2。

这会把论文的方法讲得非常直观。

---

# 20. Ablation 必须围绕故事做

不要做十几个没有意义的 network hyperparameter ablation。

主要就做：

### Static → marginal

去掉：

$$
A
$$

的 conditioning。

证明 redundancy 是真正原因。

### Utility → relevance

把 utility target 换成：

same source / similarity / relation target。

证明 end-task utility supervision 有作用。

### Stop → fixed budget

强制始终选择 \(B\) 个。

证明 selective stopping 防止 negative transfer。

### Prediction-delta feature

去掉：

$$
f_{A+j}-f_A.
$$

看单纯 representation similarity 能不能预测 utility。

### Ranking loss

$$
\mathcal L_{\rm util}
$$

vs

$$
\mathcal L_{\rm util}+\mathcal L_{\rm rank}.
$$

够了。

---

# 21. Related Work 的故事

不能继续按照 ICLR 那种：

ICL theory → Bayesian ICL → prior adaptation

写。

KBS Related Work 改成四条线：

### Selective knowledge transfer and negative transfer

已有大量工作会：

* domain alignment；
* source selection；
* feature filtering；
* cross-domain transfer。

KBS 自己近年就有多篇直接把 negative transfer 当核心问题，例如 sequential recommendation 中抑制 negative transfer、many-task optimization 中选择 transfer source。([科学直通车][3])

你的 distinction：

> 这些工作大多通过 domain/task similarity、representation alignment 或特定任务结构来控制 transfer；我们直接估计一个候选在**当前已选 knowledge set 条件下**对最终 predictive loss 的边际贡献。

---

### Routing and mixture-of-experts

讲 MoE/router。

现有 router 通常选择：

* expert；
* domain；
* computation path。

KBS 2025 也有 task-specific expert routing 类工作。([科学直通车][4])

你的区别：

> MUR 不是给 expert 分工，而是对每一个具体 candidate context 做 outcome-conditioned marginal-value estimation。

---

### Selective prediction / abstention

传统 selective prediction：

> “这个 query 我有没有信心回答？”

你的问题：

> “这个 query 我要回答，但还值得继续加入外部 context 吗？”

这两个 abstention 是不一样的。

---

### Context / retrieval selection

RAG、demonstration selection、in-context example retrieval。

大量工作优化：

$$
\text{similarity},
\quad
\text{relevance},
\quad
\text{retrieval score}.
$$

你的核心批判不是：

> relevance 没用。

而是：

> relevance 是 candidate 的 intrinsic relationship；marginal utility 是 candidate 相对于**当前知识状态**的 decision value。

这个 distinction 要写得非常精确。

---

# 22. Introduction 的完整故事节奏

我建议 5 个自然段，不要上来堆公式。

### 第一段：More context is not always better

现在 prediction systems 越来越容易获取外部 information：

* historical cases；
* retrieved examples；
* neighboring observations；
* auxiliary sources。

真正困难的已经不只是找到信息，而是：

> 找到的信息应该不应该进入 prediction。

强调额外 knowledge 可能 redundant 或 harmful。

---

### 第二段：现有 selection criterion 回答错了问题

现有方法往往问：

> “这个 source/context 与当前 query 有多相关？”

但 relevance 不能告诉我们：

> “在系统已经拥有这些 information 的情况下，它还能增加多少 predictive value？”

引入 redundancy example。

这是全文 problem gap。

---

### 第三段：重新定义问题

提出：

$$
m(j\mid A)
$$

即 context \(j\) 相对于当前 knowledge set \(A\) 的 marginal predictive utility。

它直接基于：

$$
\text{loss before}
-
\text{loss after}.
$$

所以学习目标和 deployment decision 对齐。

---

### 第四段：方法

介绍 MUR：

* outcome-supervised marginal utility；
* set-conditioned encoder；
* sequential selection；
* positive-utility stopping。

不需要 task identities，也不需要 known relation priors。

---

### 第五段：发现和贡献

等实验出来以后再写具体结果。

但贡献逻辑固定：

1. problem；
2. method；
3. theory；
4. evidence。

不要塞十个 percentage。

---

# 23. Abstract 我直接给你一个接近最终稿的骨架

> **Auxiliary context is increasingly used to improve predictive models, yet selecting which context to use is commonly treated as a relevance-ranking problem. Relevance, however, does not measure the value that a candidate adds to information already available to the predictor. Highly related contexts can become redundant after another context has been selected, while less similar candidates may provide larger incremental predictive gains. We formulate auxiliary-context selection as a budgeted decision problem based on set-conditioned marginal utility: the expected reduction in task loss obtained by adding a candidate to the currently selected context set. We introduce a Marginal Utility Router that learns this quantity directly from outcome-supervised loss differences and sequentially adds contexts only while their predicted incremental value remains positive. The formulation requires neither latent task identities nor a predefined relation prior, and naturally supports selective stopping under a context budget. We connect utility-estimation error to sequential routing regret and evaluate the resulting policy in controlled and real-world multi-context prediction settings. [Final sentence replaced after experiments: summarize predictive gain, negative-transfer reduction, and budget efficiency without numerical clutter.]**

这个 abstract 已经完全不应该再出现：

* finite library；
* collision；
* \(M^2/M/1\)；
* BF-Gate；
* prior transport。

---

# 24. Contributions 最终就四条

不要像 ICLR 那样再做很多条。

可以写：

**1.** We formulate multi-context knowledge use as set-conditioned marginal utility optimization, distinguishing the predictive value of a candidate from its standalone relevance.

**2.** We introduce MUR, an outcome-supervised router that estimates candidate-specific marginal utility and sequentially selects auxiliary contexts with an explicit stopping rule.

**3.** We establish decision guarantees linking utility-estimation error to stepwise and sequential routing regret, together with a sufficient condition for safe stopping.

**4.** We evaluate the formulation under controlled redundancy, distractors, harmful transfer, varying candidate-set sizes, and several real-world prediction tasks, separating relevance quality, utility calibration, predictive gain, and context cost.

这四条已经够。

---

# 25. 正文结构

最终目录我建议：

```text
1 Introduction

2 Related Work
  2.1 Selective and Negative Knowledge Transfer
  2.2 Routing and Mixture-of-Experts
  2.3 Selective Prediction and Context Selection

3 Auxiliary Context as a Decision Problem
  3.1 Multi-Context Prediction
  3.2 Set Utility and Marginal Utility
  3.3 Why Relevance Is Insufficient

4 Marginal Utility Routing
  4.1 Set-Conditioned Utility Estimation
  4.2 Outcome-Supervised Utility Learning
  4.3 Sequential Selection and Stopping
  4.4 Computational Complexity

5 Decision Analysis
  5.1 One-Step Routing Regret
  5.2 Sequential Error Accumulation
  5.3 Safe Stopping
  5.4 Connection to Budgeted Set Selection

6 Experiments
  6.1 Experimental Setup
  6.2 Controlled Multi-Context Environment
  6.3 Real-World Prediction Tasks
  6.4 Candidate-Set and Budget Scaling
  6.5 Negative Transfer and Utility Calibration
  6.6 Ablation Studies

7 Discussion
  7.1 When Marginal Utility Helps
  7.2 Limitations and Candidate Interactions

8 Conclusion
```

这个目录和 ICLR 几乎没有结构性重合。

---

# 26. Discussion 反而很重要

KBS 可以把局限讲得比较成熟。

最值得讲的是：

MUR 是 sequential greedy。

它能很好解决：

* redundancy；
* harmful contexts；
* diminishing marginal value。

但对于严格的 **pure complementarity**：

$$
m(i\mid\emptyset)\le0,
$$

$$
m(j\mid\emptyset)\le0,
$$

但：

$$
F(\{i,j\})>F(\emptyset),
$$

单步 greedy 可能永远不会选第一项。

这个 limitation 反而很好。

然后说未来可以做：

* pairwise look-ahead；
* bundle utility；
* beam search；
* combinatorial routing。

这样比强行假装方法解决所有 subset selection 问题可信得多。

---

# 27. 与 ICLR 的“防重合边界”

这是我最看重的一部分。

你现在 ICLR 的正式贡献包括 exact posterior/action decomposition、\(M^2/M/1\) supervision law、BF-Gate prior correction 和多个 finite-library realizations。

所以 KBS 明确规定：

| ICLR                            | KBS                                       |
| ------------------------------- | ----------------------------------------- |
| two-context relation            | multi-candidate subset selection          |
| finite task library             | arbitrary heterogeneous context pool      |
| collision evidence              | predictive marginal utility               |
| \(p_M\)                         | \(m(j\mid A)\)                            |
| known/estimated collision prior | no relation prior                         |
| \(-\log(M-1)\)                  | no prior correction                       |
| BF-Gate                         | MUR                                       |
| balanced collision supervision  | outcome utility supervision               |
| \(M^2,M,1\)                     | routing regret / stopping                 |
| pairwise relation score         | set-conditioned candidate value           |
| library-size scaling            | candidate-count/budget/redundancy scaling |

甚至当前 ICLR Fig.3 已经明确区分 evidence quality、prior transport 和 prediction action。

所以 KBS 不要再写：

> “relation accuracy does not imply prediction accuracy”

作为中心贡献。

要写得更往前：

> **Even a context with positive standalone predictive value may no longer be worth using after another context has been selected.**

这才是 KBS 的真正新命题。

---

# 28. 哪些代码可以直接复用

你想控制在约 40% 工程改动，我认为完全可以。

### 原样或轻改复用

* data generation framework；
* dataset loading；
* batching；
* Transformer blocks；
* context encoder；
* prediction expert；
* optimizer；
* seed management；
* bootstrap；
* logging；
* plotting utilities。

大约：

**55–65% 工程基础保留。**

### 新增

* candidate-pool generator；
* selected-set representation；
* random subset sampler；
* marginal utility label computation；
* utility regression head；
* pairwise ranking loss；
* sequential routing loop；
* stopping；
* new metrics；
* two new datasets。

大约：

**35–45% 新代码。**

这基本符合你的要求。

---

# 29. 但论文内容要比 40% 改得更多

正式 manuscript 不要拿“40% text difference”做目标。

我会要求：

| 部分                       |   重写比例 |
| ------------------------ | -----: |
| Title                    |   100% |
| Abstract                 |   100% |
| Introduction             |   100% |
| Related Work             | 80–90% |
| Problem formulation      |   100% |
| Method                   |   80%+ |
| Theory                   |   100% |
| Main figures             |   100% |
| Main experimental tables |   100% |
| Conclusion               |   100% |
| Low-level implementation |  可高度复用 |

**40% 是工程新增，不是论文文字只改 40%。**

这样才真正安全。

---

# 30. 最关键的四个预实验

在投入一周写 KBS 之前，我会先跑四个低成本 gate。

### Gate A

**Relevance Router vs Static Utility**

验证直接 utility supervision 是否真的优于 relation/relevance。

如果没有：

故事第一层就不成立。

---

### Gate B

**Static Utility vs MUR**

人为加入大量 redundant candidate。

这是最关键的预实验。

需要看到：

$$
\text{MUR}>\text{Static Utility}
$$

而且差异随 redundancy 增大。

否则“set-conditioned”没有必要。

---

### Gate C

固定 budget vs learned stop。

需要看到随着 candidate 数量增加：

Pool-all / forced-\(B\) 出现 negative transfer，

MUR stop 可以避免一部分 harm。

---

### Gate D

在 Electricity 上快速做：

$$
K=8,\quad B\le2
$$

只比较：

* base；
* similarity；
* static utility；
* MUR；
* oracle greedy。

如果 Electricity 上也出现：

$$
\text{MUR}>\text{Static Utility}>\text{Similarity},
$$

这篇 KBS 基本就有了。

---

# 最终整篇文章的“故事闭环”

开篇不是：

> 我们提出一个新 router。

而是一个非常自然的问题：

> 现代系统可以获得越来越多外部知识，但“更多知识”并不自动意味着“更好的预测”。

然后指出已有范式的盲点：

> retrieval 和 routing 通常给每个候选一个静态 relevance/value score，但一个知识源的价值不是固有属性，它取决于当前系统已经掌握了什么。

于是定义新的决策量：

$$
m(j\mid A).
$$

然后给出 MUR：

> 直接从最终任务 outcome 学习这种新增价值。

接着理论说明：

> utility estimation 足够准确，就能控制 routing regret；没有正的 marginal value 时应该停止。

实验再构造出真正有意义的现象：

> candidate pool 越大、redundancy 越强，static relevance 越容易重复选择或发生 harmful transfer，而 set-conditioned marginal utility 能重新排序候选，并主动停止继续获取知识。

最后不是说：

> 我们的模型精度高了一点。

而是得到一个更一般的认识：

> **Auxiliary knowledge has no fixed value in isolation. Its value is marginal, decision-dependent, and changes as the system acquires other information.**

我认为这才是这篇 KBS 最完整、最有发表价值、同时又能和当前 ICLR 真正拉开距离的故事。并且 KBS 近年已经有大量 negative-transfer、MoE 和 intervention-routing 工作，因此把创新压到“负迁移”“不确定性 router”本身已经不够；**set-conditioned marginal predictive value** 才应该成为你的识别性贡献。([科学直通车][3])

如果按这个版本推进，我下一步会直接把它继续细化成 **KBS 的完整 Figure 1–5 / Table 1–4 设计 + 每个 section 要写的具体论点、公式、实验以及预计页数**，基本就能变成可执行的论文蓝图。

[1]: https://www.sciencedirect.com/science/article/pii/S0950705126014115?utm_source=chatgpt.com "Failure-mode-aware uncertainty intervention routing for large language models - ScienceDirect"
[2]: https://shop.elsevier.com/journals/knowledge-based-systems/0950-7051?utm_source=chatgpt.com "Subscribe to Knowledge-Based Systems - 0950-7051 | Elsevier Shop | Elsevier Shop"
[3]: https://www.sciencedirect.com/science/article/pii/S0950705125015187?utm_source=chatgpt.com "Multi-modality meets re-learning: mitigating negative transfer in sequential recommendation - ScienceDirect"
[4]: https://www.sciencedirect.com/science/article/pii/S095070512500841X?utm_source=chatgpt.com "MoTE: Mixture of task-specific experts for pre-trained model-based Class-incremental learning - ScienceDirect"
