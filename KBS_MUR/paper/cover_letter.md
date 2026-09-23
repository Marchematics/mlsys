# Cover letter (draft)

**To:** Editors-in-Chief, *Knowledge-Based Systems*
**Manuscript type:** Original research article
**Title:** Response-Aware Marginal-Utility Routing for Multi-Context Prediction

Dear Editors,

We submit for consideration in *Knowledge-Based Systems* a study on how a
predictive system should decide which pieces of auxiliary knowledge to use.
The manuscript is a knowledge-driven decision-support contribution: it defines
the decision object, shows when that object contains information that fixed
scoring rules cannot represent, learns it from the response of a fixed
predictor, and validates it on six standard benchmarks with multiple prediction
targets.

**The problem.** Systems increasingly predict from an anchor signal plus a pool
of auxiliary contexts, such as retrieved examples, neighbouring sensors, or
historical records. Existing rules derive candidate value from different
signals: relevance from the query, diversity and coverage from the selected
content, and standalone utility from the predictor relative to an empty context
set. None of them represents the change in end-task loss produced by adding a
candidate to the contexts already selected. A candidate that is highly
informative in isolation can become redundant after similar knowledge is used,
and a candidate with a modest standalone score can carry the missing
information.

**What we contribute.**

1. We formalise context selection through *state-conditioned marginal
   predictive utility*, the reduction in end-task loss obtained by adding a
   candidate to the context set selected so far, and derive its squared-loss
   decomposition into residual alignment, overcorrection, and context cost.
2. We separate two quantities that existing rules conflate: *structural
   headroom*, the gain available to a sequential oracle over a perfect
   standalone ranking, which is a property of the task; and the two-stage
   *routing regret*, which decomposes exactly into a screening term and a
   reranking term and is bounded by the estimation error of the two stages.
3. We propose Response-Aware Marginal-Utility Routing, which screens candidates
   with cached representations, evaluates the fixed predictor only on a
   shortlist, and reranks the shortlist with compact summaries of the predictor
   response under a utility-gap-weighted ranking objective.
4. We evaluate the resulting router against fourteen deployable selection
   policies that represent the similarity, diversity, coverage, determinantal,
   standalone-utility, fixed-set and state-conditioned families, on six standard
   traffic benchmarks with 32 fixed prediction targets each and three
   independent runs.

**Evidence.** Controlled studies isolate redundancy and candidate-set growth as
the two mechanisms that create structural headroom and compress the decision
margin. Across the six benchmarks, the proposed method achieves the best
performance among the evaluated deployable selection policies on every
benchmark and the best average rank across fourteen deployable policies, which
include content-based, coverage-based, fixed-set, and learned selection rules;
the advantage over both learned standalone utility and the cached
state-conditioned router is resolved by hierarchical bootstrap intervals on all
six datasets. A set chosen offline by greedy validation selection is the
closest competitor, and R-MUR improves on it on every dataset, which isolates
the value of conditioning each decision on the current selected state. A multi-target analysis shows that the improvement extends across
prediction targets rather than being driven by a single sensor, and 960 oracle
target-run evaluations confirm that the structural opportunity the method
exploits is present on every dataset and target.

**Fit with the journal.** The contribution is a decision-support mechanism for
knowledge reuse rather than a new network architecture. Recent work in
*Knowledge-Based Systems* has demonstrated the value of adaptive routing,
selective knowledge transfer, and context-dependent information acquisition
across language models, recommendation, multimodal reasoning, collaborative
perception, and retrieval. Those methods adaptively select models,
interventions, experts, or knowledge sources; this manuscript defines the
selection target through the marginal end-task value of an auxiliary context
conditioned on the contexts already selected, and studies when that value is
estimable. It states when auxiliary knowledge is worth using, when it becomes
redundant or harmful, and what information a router needs in order to act on
that distinction.

**Declarations.** The manuscript is original, is not under consideration
elsewhere, and all authors have approved the submission. The authors declare no
competing interests. A CRediT authorship statement, a declaration of competing
interest, a data-availability statement and a funding statement are included in
the manuscript. Highlights are provided as a separate file.

% TODO(authors): add the suggested-reviewer list required by the Editorial
% Manager portal.
% TODO(authors): if any part of this work was previously submitted to another
% journal, disclose it here rather than leaving the statement as it stands.

Sincerely,

Jiahao Zhang (corresponding author)
Yu Chen (corresponding author)
School of Electronics and Information
Zhengzhou University of Aeronautics, Zhengzhou, China
marchematics@gmail.com, chenyu@zua.edu.cn

On behalf of all authors:
Xinling Wen, Jiahao Zhang, Yu Chen
