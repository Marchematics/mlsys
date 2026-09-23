# Final Proposal: Marginal-Utility Routing for Multi-Context Knowledge-Augmented Prediction

## Problem Anchor

- **Bottom-line problem**：预测系统面对候选辅助知识池和有限 context budget 时，必须选择真正能在当前知识状态上继续降低 end-task loss 的 contexts。
- **Must-solve bottleneck**：relevance、pairwise relation 和 standalone utility 都把候选价值视为相对静态的量，无法识别“单独有用、选入其他信息后变冗余”的候选，也无法直接处理 harmful transfer 与停止继续取用知识的决策。
- **Non-goals**：不研究 finite task library、collision posterior、same-task prior、`-log(M-1)`、BF-Gate、balanced collision supervision 或 `M^2/M/1` scaling；不把 uncertainty routing、关系分类或更复杂的 MoE 作为第二条主线。
- **Constraints**：目标期刊为 Knowledge-Based Systems；已有 2×A10G 24GB；尽量复用数据加载、context encoder、prediction expert、seed/logging/bootstrap 基础设施；新工程约 35–45%；先完成低成本 gate，再扩展到 Traffic 与 Activity。
- **Success condition**：在可控 redundancy 与 harmful-candidate 环境中，MUR 必须稳定优于 Static Utility，且差距随 redundancy 增强；在 Electricity 快速门控中形成 `MUR > Static Utility > Similarity` 的方向性证据，同时 MUR 在 gain–harm 或 gain–budget 前沿上优于固定预算方法。

## Anchor Check

- **Original bottleneck**：候选价值依赖当前 selected set；静态 relevance 或 `u(j|empty)` 无法处理选择后的冗余、负迁移和继续取用知识的决策。
- **Why the revision stays anchored**：修订只收紧 `A!=empty`、counterfactual end-task loss、calibrated select/stop 与成本协议，没有引入新任务。
- **Rejected drift**：不把项目改写为 diversity/coverage selection；不把 uncertainty estimation 升格为第二条主线；不回到 finite-library relation posterior。

## Simplicity Check

- **Dominant contribution**：state-conditioned counterfactual marginal value interface。
- **Components removed/contained**：exact prediction delta 从主方法移到 MUR-full；ranking loss 降为 ablation；submodular guarantee 移至 appendix。
- **New non-trainable support**：split-calibrated utility interval，只服务 select/stop 证书。
- **Complexity deliberately rejected**：Set Transformer、RL policy、beam search、Bayesian uncertainty head、joint expert-router training。
- **Primary protocol fixed**：单一 frozen expert 只在 `expert-train` partition 拟合，之后不 refit；`router-label`、`calibration` 与 test 均评价同一个 expert。cross-fitting 只作为数据效率 robustness，不是主结果。

## Technical Gap

现有路线已经覆盖了这几个相邻问题：

1. 相似度或 relation reranking 为每个候选独立赋分。
2. outcome-derived utility 将 1-shot 表现与 0-shot 表现相减，但仍然学习 `u(j | empty)` 并静态选 top-k。Hashimoto et al. (NAACL 2024) 的 incremental utility 正属于这一类。
3. Set-BSR、DPP、MMR 与 TopicK 通过 coverage/diversity surrogate 抑制冗余，但目标仍不是当前预测器的 end-task loss reduction。
4. contextual submodular prediction 学习预算集合策略，提供重要理论先例；它通常把可测或已知结构化 reward 作为 policy-learning target。MUR 的对象不同：它先从 frozen predictor 的 counterfactual end-task losses 学习此前未知的 state-dependent marginal reward，再在允许 negative/non-monotone effects 的环境中路由。本文不会把标准 greedy 或 submodular 理论包装成新贡献。

缺失的最小机制是：在任意实际选择状态 `A` 上，直接估计候选 `j` 对冻结预测器造成的 counterfactual end-task loss marginal `m(j | A)`，随后重新评分、顺序选择，并用校准区间区分 positive-value selection、conservative stopping 与不确定区域。

这一 novelty 边界必须写窄。论文不能声称首次研究 utility、incremental utility、redundancy 或 greedy set selection。可辩护的识别性贡献是：**对任意非空状态 `A` 学习 state-conditioned counterfactual end-task loss marginal，并把它作为允许负 marginal 的 sequential context-routing value model，配套校准的 select/stop 决策**。Hashimoto-style incremental utility 是必须正面对比的 `A=empty` 特例；Set-BSR/TopicK 是 coverage surrogate；contextual submodular prediction 是依赖结构化 reward 的 policy-learning 前例。

## Method Thesis

- **One-sentence thesis**：辅助知识应由任意当前状态下的 counterfactual end-task loss marginal `m(j|A)` 来选择，并在其校准下界不再为正时停止继续取用知识。
- **Smallest adequate intervention**：冻结 prediction expert，仅增加 permutation-invariant selected-set encoder 与单一 utility head；训练标签由同一 episode 的 before/after loss difference 自动生成。
- **Frontier position**：无需强行引入 LLM、RL 或 uncertainty module。方法面向现代 retrieval/context-rich systems，但核心是一个模型无关的 decision interface。

## Contribution Focus

- **Dominant contribution**：将 multi-context knowledge use 表述为 set-conditioned marginal predictive utility estimation，并据此给出 sequential routing 与 stopping policy。
- **Supporting contribution**：建立 estimator error 到 per-state approximate-greedy quality，以及 conservative selection / calibration-supported stopping 的分析。
- **Explicit non-contributions**：prediction backbone、context encoder、DeepSets 本身、Huber loss、pairwise ranking loss、submodular greedy guarantee 都不是独立贡献。

## Proposed Method

### Formal decision object

给定 anchor context `S0`、query `x`、target `y`、候选集合 `C={C1,...,CK}` 与当前集合 `A`，冻结预测器产生 `f_A(x)`。定义净集合效用

`F(A) = E[ell(f_empty(x), y) - ell(f_A(x), y)] - lambda C(A)`，

其中第一版令 `C(A)=sum_{j in A} c_j`。真实边际效用为

`m(j | A) = F(A union {j}) - F(A)`。

训练 episode 上的监督为

`m_tilde(j | A) = ell(f_A(x), y) - ell(f_{A union {j}}(x), y) - lambda c_j`。

`m_tilde` 是 sampled episode 上的 realized counterfactual marginal。只有 prediction expert 固定、采样来自目标 state distribution 且无选择性重加权时，它才可视为相应条件期望的无偏样本。主方法不依赖这句更强的无偏性表述；它明确优化所构造 state distribution 上的 empirical counterfactual risk。联合更新 expert 会使标签漂移，因此采用两阶段训练并冻结 expert。

### Why relevance is insufficient

平方损失下，令 `mu=E[Y | S0,x,C]`、`d_{A,j}=f_{A+j}-f_A`、`r_A=mu-f_A`，则条件净效用为

`m(j | A)=2 d_{A,j} r_A - d_{A,j}^2 - lambda c_j`。

该式说明候选价值由 correction 的方向、幅度、当前 residual 和成本共同决定。相关候选若 correction 方向错误或在已有集合下重复修正，边际效用可为负；中等相似候选可能补足 residual 而更有用。

### Complexity Budget

- **Frozen/reused**：dataset loaders、context encoder、prediction expert、optimizer、seed/logging、bootstrap、plotting。
- **New trainable components**：`SetEnc` 与 `g_phi` utility head；candidate/context encoders 共享参数。split calibration 不含可训练网络。
- **Intentionally excluded**：uncertainty head、relation classifier、contrastive branch、Bayesian module、RL policy、jointly trained expert、beam search。

### System Overview

```text
anchor/query ── shared encoder ── h_Q
candidates   ── shared encoder ── h_1 ... h_K
selected A   ── DeepSets(sum/mean) ── h_A
MUR-light: z=[h_Q,h_A,h_j,|h_A-h_j|,h_A*h_j,c_j,|A|]
MUR-full:  z plus exact batched d_{A,j}=f_{A+j}-f_A
utility head g_phi(z) -> m_hat(j|A)
calibration residual quantile -> [LCB,UCB]
sequential policy -> select / conservative stop / uncertain stop -> update A
```

### Training recipe

1. 主协议采用 entity/time-aware 四分区：`expert-train`、`router-label`、`calibration`、`test`。prediction expert 只在 `expert-train` 上拟合一次并永久冻结，不在缓存 labels 后 refit；因此 router labels、calibration intervals 和 test routing 始终对应同一个 deployed expert。K-fold cross-fitting 仅作为 appendix 数据效率检查，不能替代主协议。
2. 对每个 router-label episode 随机采样 `|A| in {0,...,B-1}`、`j notin A`；同时加入 relevance-prefix、static-utility-prefix 与 oracle-prefix states，减轻部署 state shift。任何 hard-state oversampling 都在日志中记录；population calibration/metrics 使用原始权重或 importance weights。
3. 批量计算 `f_A`、`f_{A+j}` 与 `m_tilde`，缓存 embeddings、feature/label、expert hash、episode ID、subset bitmask 和 expert-call count。label cache 只能由 train partition 构造。
4. 主损失：`L_util = Huber(m_hat, m_tilde)`。
5. 支持性排序损失只在同一 `(episode,A)` 内构造候选对：
   `L_rank=softplus(-sign(m_i-m_j)(m_hat_i-m_hat_j))`。
6. 主模型固定 `alpha=0`；ranking loss 是预注册 ablation。只有跨 3 seeds 改善 utility ordering 且不损害 calibration 时才保留。
7. 独立 calibration split 计算绝对 residual 的 split-conformal quantile `q_(1-alpha)`，构造 `[m_hat-q, m_hat+q]`。按 state size 与 dataset 做 coverage audit；coverage 不足时只称 empirical interval，不称 guarantee。
8. 在 validation rollout 上分 state source 报告 MAE/coverage。若 deployed-prefix coverage 低于 nominal coverage 超过 3 percentage points，或 deployed-prefix MAE 超过 random-state MAE 的 1.25 倍，则触发且只触发一次 train-only on-policy roll-in augmentation；重新训练后冻结 protocol，再做 calibration 与 test。该 gate 在接触 test 前决定。

### Inference and stopping

初始化 `A=empty`。唯一有资格支撑“deployable router”主张的 **MUR-light** 只用缓存 embeddings、set state、cost 和 set size 给剩余候选评分，不执行 `f_{A+j}`。**MUR-full** 加入 batched exact prediction delta，只作为 high-cost diagnostic / amortization headroom；除非其端到端运行时间与 MUR-light 竞争，否则不参与 deployability 结论。达到 budget 或无法认证正效用时停止。

估计误差界产生两个不同证书，必须严格区分：

- **harm-safe selection**：选择最大 LCB；仅当 `LCB>0` 才加入。对 symmetric uniform error，这等价于 `m_hat>epsilon`。
- **conservative stopping**：仅当所有候选 `UCB<=0` 时进入最保守的停止状态。对 uniform error，这等价于 `max m_hat<=-epsilon`；它不自动形成 adaptive trajectory 的 simultaneous guarantee。

当 best candidate 的区间跨零时，系统进入不确定区域。主风险厌恶 policy 选择停止但不称 simultaneous guarantee；budget-seeking 变体可按 point estimate 继续。论文分别报告 conservative-stop rate、uncertain-stop rate 与 gain–harm curve。

### Computational complexity

MUR-light 每步重用缓存 embeddings，复杂度为 `O(BK d)` 的小型 head evaluations，无 counterfactual expert call。MUR-full 为 `O(BK)` batched expert evaluations；对 ridge/linear expert 可解析更新。论文报告 offline label-generation expert calls、online router calls、online expert calls、wall-clock latency、peak memory 和下游 context cost，不能只报告 selected-context 数。

### Closest-work contract

| Method family | Reward/score | Label changes with nonempty `A` | Negative marginal | Calibrated stop | End-task counterfactual |
|---|---|---:|---:|---:|---:|
| Similarity/relevance | representation relation | No | No | No | No |
| Hashimoto incremental utility | 1-shot vs 0-shot outcome | No (`A=empty`) | Yes | No | Yes |
| Set-BSR / TopicK | coverage or topical need | Yes via surrogate state | Usually no | No | No |
| Contextual submodular policy | known/measurable structured reward | Yes | assumes monotone/submodular in core theory | No | policy learns reward maximization |
| MUR-light | learned loss marginal | Yes, arbitrary `A` | Yes | Yes | labels: Yes; inference: predicted |
| MUR-full | learned loss marginal + exact delta | Yes, arbitrary `A` | Yes | Yes | labels and features: Yes |

## Decision Analysis

### Proposition 1: squared-loss utility identity

在给定 observable state 且预测确定时，

`m(j|A)=2d_{A,j}(mu-f_A)-d_{A,j}^2-lambda c_j`。

该命题用于证明 relevance 不是 utility 的充分统计量，不宣称 relevance 无用。

### Theorem 1: one-step approximate-greedy regret

若当前状态所有候选满足 `|m_hat(j|A)-m(j|A)|<=epsilon`，则 MUR 选择 `j_hat=argmax m_hat` 与当前状态最优边际候选 `j*=argmax m` 满足

`m(j*|A)-m(j_hat|A)<=2epsilon`。

### Sequential statement

对 MUR 自身访问的状态，逐步累加得到 approximate-greedy certificate：每一步比该状态的最佳真实 marginal 至多损失 `2epsilon`。**不在任意 non-submodular F 下声称其终局与另一条 exact-greedy trajectory 相差至多 `2B epsilon`**，因为路径分叉会改变后续 marginal。

若额外假设 `F` normalized、monotone、submodular，则可在 appendix 给出标准 approximate-greedy guarantee：

`F(A_B)>= (1-1/e)F(A*) - 2B epsilon`。

真实任务允许 negative transfer，因此 submodularity 只做被检验的条件性结果，不作为主卖点。

### Proposition 2: selection and stopping bounds

在 uniform error `epsilon` 下：`m_hat>epsilon` 保证 positive true marginal；`max m_hat<=-epsilon` 保证不存在 positive remaining marginal。对其余情况允许 conservative stop 或继续，形成可解释 risk–coverage trade-off。

## Claim-Driven Validation Sketch

### Claim 1: set-conditioned utility fixes redundancy-induced budget waste

- **Minimal experiment**：controlled environment，固定 relevance AUC，提高 redundancy ratio `0→0.8`。
- **Baselines**：Similarity/Relevance、Hashimoto-style normalized Incremental Utility、Static Utility、MMR/Set-Coverage、MUR-light、MUR-full、Oracle Greedy。
- **Metrics**：task loss、utility recovery、useful-selection precision、average contexts。
- **Decisive evidence**：MUR–Static gap 随 redundancy 增大，且非来自参数量或更多 expert calls。

### Claim 2: outcome utility and stopping jointly reduce harmful transfer under budget

- **Minimal experiment**：提高 harmful fraction 与 pool size；比较 forced-B 与 calibrated stopping。
- **Metrics**：prediction gain、negative-transfer rate、gain–harm Pareto、router overhead。
- **Decisive evidence**：MUR 在相同 gain 下更低 NTR，或相同 NTR 下更高 gain；Base-only 作为零 gain/零 harm 参考点。

## Five Pre-Experiment Gates

1. **Gate A — Relevance vs Static Utility**：若 outcome utility 不优于 relevance，停止当前故事。
2. **Gate B — Static Utility vs MUR**：若 redundancy 增大时 MUR 无稳定优势，删除 set-conditioned 主张并重构项目。
3. **Gate C — Fixed budget vs learned stop**：若 stopping 不改善 gain–harm frontier，停止将 abstention 作为贡献。
4. **Gate D — Electricity K=8, B<=2**：要求方向性结果 `MUR > Static Utility > Similarity`；否则先诊断 label noise/expert capacity，不扩展新数据集。
5. **Gate N — novelty stress**：所有候选 relation-positive 且 redundant candidates 的 standalone utility 均为正；只有 `A` 改变后边际效用重新排序。同一张核心图固定展示 Hashimoto normalized Incremental Utility、Static Utility、Set-Coverage/MMR 与 MUR-light 四条线。若前三者与 MUR-light 持平，则收窄或停止 novelty claim。

## Figure 1–5 Design

### Figure 1 — Relevance is not marginal value (main, 0.9 page)

- A：四候选现象图，展示 relevance 排名与 initial utility。
- B：选择 `C1` 后重估，`C2` 因冗余下降、`C3` 保持新增价值。
- C：真实/受控样本的 relevance–observed marginal utility 散点，标记 useful/harmful quadrants。
- 结论：候选价值随 selected set 改变。

### Figure 2 — MUR architecture and sequential policy (main, 0.85 page)

图分成两条明确路径：上方训练标签路径由 frozen expert 的 `f_A` 与 `f_{A+j}` 产生 outcome loss difference；下方部署路径由 cached encoder、DeepSets state 和 MUR-light utility head 完成 embedding-only batched scoring，再执行 select/stop。MUR-full 的 exact-delta 路径用虚线标为 high-cost diagnostic，避免暗示 MUR-light 推理需要 counterfactual expert calls。

### Figure 3 — Controlled mechanism tests (main, 1 page)

- A redundancy ratio→test loss；
- B pool size K→utility recovery；
- C maximum budget→prediction gain/actual contexts；
- D negative-transfer rate→average contexts。

### Figure 4 — Real-data gain–budget frontiers (main, 1 page)

Electricity、Traffic、Activity 三个 panels；横轴平均使用 contexts 或总推理成本，纵轴相对 Base-only 的预测提升。MUR、Static Utility、Relevance、Pool-all、Oracle Greedy。

### Figure 5 — Episode-level routing trace (main, 0.75 page)

展示一个真实 episode 的候选 utility 在 `A=empty`、选 `C1`、再选 `C3` 后的变化及 stop；并排显示 static relevance 错误选择冗余 `C2`。

## Table 1–4 Design

### Table 1 — Datasets and candidate construction

数据集、任务、split unit、anchor、candidate types、K/B、cost unit、primary metric。禁止把同 subject/client/sensor 泄漏到不允许的 split。

### Table 2 — Main predictive and decision results

方法×Electricity/Traffic/Activity primary metric、NTR、utility recovery、average contexts、wall-clock cost。主比较突出 Relevance→Static Utility→MUR。

### Table 3 — Mechanism ablations

MUR、no-set-conditioning、relevance target、fixed budget、no prediction-delta feature、no ranking loss；报告 redundancy-high 和 harmful-high 两种 setting。

### Table 4 — Robustness and certificate audit

K、B、cost lambda、stopping threshold、utility calibration ECE/MAE、error-bound coverage、harm-safe precision、latency。submodularity ratio 仅作诊断，不伪装成全局假设成立。

## Paper Structure and Page Budget

以 Elsevier preprint 约 26–30 页正文（含参考文献前）为工作预算：

1. Introduction — 2.0 pages：more context can hurt；relevance gap；marginal utility；MUR；四项贡献。
2. Related Work — 2.5 pages：negative transfer；routing/MoE；selective prediction；context/demo selection；closest-work boundary。
3. Auxiliary Context as a Decision Problem — 3.0 pages：notation、F(A)、m(j|A)、squared-loss identity、relevance insufficiency。
4. Marginal Utility Routing — 4.0 pages：architecture、label generation、loss、sequential inference、complexity。
5. Decision Analysis — 2.5 pages：one-step bound、approximate-greedy certificate、safe selection/stopping、conditional submodular corollary。
6. Experiments — 8.0 pages：setup、controlled environment、三真实任务、scaling、negative transfer、calibration、ablations。
7. Discussion — 1.5 pages：适用条件、label cost、expert dependence、pure complementarity limitation。
8. Conclusion — 0.5 page。
9. References — 3–5 pages；proofs、dataset细节、完整表放 supplement。

## Failure Modes and Diagnostics

- **Standalone utility already sufficient**：Gate B；若成立，停止 MUR claim。
- **Label noise overwhelms utility signal**：报告 conditional variance、replicated outcomes、Huber vs MSE；先提高 label reliability，不加模型。
- **State distribution shift**：对 random、baseline-prefix、oracle-prefix 与 deployed-prefix states 分层报告误差；预注册触发条件为 coverage shortfall >3 percentage points 或 MAE ratio >1.25，最多增加一次 on-policy roll-in，不进入迭代 imitation-learning 循环。
- **Router learns candidate identity shortcut**：client/sensor/subject-disjoint splits，shuffle-ID probe。
- **Prediction-delta feature leaks target**：只使用 inference-available predictions，严格禁止 target-derived features。
- **Expert too weak/strong**：报告 oracle headroom与 base performance；无 headroom 的任务不用于证明 routing。
- **Greedy misses pure complementarity**：构造 pair-synergy failure case；明确 limitation，不用 beam search 掩盖主方法。
- **Routing compute exceeds savings**：端到端 latency 和 token/context cost审计；必要时把 cached/batched mode 作为工程优化，不升级为第二贡献。
- **Novelty collision with incremental utility/context-dependent retrieval**：Related Work 明确区分 `u(j|empty)` 与 `m(j|A)`；补充 2026 utility-centric retrieval 与 fragment information-gain 工作的逐项对照。

## Novelty and Elegance Argument

MUR 的论文价值不来自命名一个 utility MLP。它来自把训练信号、非空 state representation、推理策略、校准区间和评价指标都对齐到同一 counterfactual decision quantity `m(j|A)`。相邻工作分别覆盖了 `A=empty` outcome utility、coverage/diversity 和 structured list prediction。本文必须用 relation-positive redundancy stress、negative transfer、sequential re-estimation、utility reliability 与 stop behavior 证明交点上的缺口确实存在。若 Hashimoto-style incremental utility 或 coverage-greedy 已覆盖主要增益，项目在 gate 阶段收缩或停止。

## Experiment Handoff Inputs

- **Must-prove claim C1**：set-conditioning 对 redundancy 有因果必要性。
- **Must-prove claim C2**：outcome utility + stopping 改善 gain–harm/budget frontier。
- **Must-run ablations**：Hashimoto incremental、Static Utility、coverage/MMR、fixed-B、MUR-light vs MUR-full、remove rank loss。
- **Critical datasets**：controlled synthetic、Electricity；Traffic/Activity 仅在 gates 通过后进入。
- **Highest-risk assumptions**：可获得稳定且无 target leakage 的 counterfactual labels；MUR-light 不依赖 exact delta 仍能学到 set-conditioned marginal；冻结 expert 存在足够 oracle headroom；closest literature 尚未完整覆盖同一 formulation。

## Compute & Timeline Estimate

- Synthetic gates：CPU 或单 A10G，约 6–12 GPU-hours（含 3 seeds）。
- Electricity gate：单 A10G，约 12–24 GPU-hours，取决于 cached counterfactual expert passes。
- Traffic/Activity：属于 expansion，不属于方法成立的先决条件；仅在 Gate A/B/C/N/D 通过后各投入 24–60 GPU-hours。
- 全部 must-run 预计 100–180 GPU-hours；2×A10G 可并行，但 raw run 必须按 seed/config 隔离。
- 人工标注：0；utility label 来自已有 outcome。
- 第一周目标：完成 A/B/C/D/N gates 与 go/no-go 结论，不写结果段落。若 state-shift trigger 触发，只在 train/validation 上完成一次 roll-in、重新训练并重新 calibration，之后才能开始 test evaluation。
