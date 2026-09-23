# Reference integrity audit (2026-09-20)

## Scope
- Bibliography: `paper/references.bib`
- Manuscript: `paper/sections/2_related_work.tex`
- Total entries: 62
- Duplicate keys: none
- Entries cited in Related Work: 62 / 62
- Entries without DOI or URL: 0

## DOI resolution
- DOI fields checked: 36
- HTTP 200/202 from DOI resolver: 26
- HTTP 403/429 from publisher-side bot protection: 10
- Crossref content negotiation was used to obtain 27 of the published entries
- The publisher-side 403/429 responses are access-control responses, not DOI
  format failures.

## Sources
- Published ACL/EMNLP/NAACL/TACL and recent Knowledge-Based Systems entries:
  Crossref content negotiation
- ACL Anthology-only entry: `https://aclanthology.org/P11-1052.bib`
- arXiv entries: arXiv API metadata, stored with `eprint` and URL
- Existing controlled-project entries: manually retained and given DOI/URL
  metadata where available

## Follow-up
- Re-check arXiv entries before camera-ready for published venue updates.
- Confirm pages and venue strings in `references.bib` once more during the final
  reference pass.

## Recent Knowledge-Based Systems positioning (2026-09-21)

Seventeen recent *Knowledge-Based Systems* articles were added or verified for
scholarly positioning. Every entry was checked against Crossref
(`api.crossref.org/works/<doi>`) on 2026-09-21; volume, article number, year
and author list below match the publisher record. None of these works is used
as a baseline: the tasks differ, so they are cited to position the
contribution, not to claim numerical superiority.

| Key | Article | KBS | DOI | Where cited |
|---|---|---|---|---|
| `failuremode2026routing` | Failure-mode-aware uncertainty intervention routing for LLMs (Yin, Zhang) | 351, 116685 (2026) | 10.1016/j.knosys.2026.116685 | RW routing |
| `dream2026routing` | DREAM: dynamic routing of experts via attention-based mixture (Sheng et al.) | 340, 115585 (2026) | 10.1016/j.knosys.2026.115585 | RW routing |
| `cogmoe2026sparse` | CoGMoE: sparse and specialized MoE for collaborative perception (Li et al.) | 337, 115329 (2026) | 10.1016/j.knosys.2026.115329 | RW routing |
| `chen2024bpmoe` | BP-MoE: behaviour-pattern-aware MoE for temporal graphs (Chen et al.) | 299, 112056 (2024) | 10.1016/j.knosys.2024.112056 | RW routing |
| `sun2025adaptive` | Adaptive in-context expert network for sequential recommendation (Sun et al.) | 326, 114061 (2025) | 10.1016/j.knosys.2025.114061 | RW routing |
| `zhang2026acra` (`acra2025retrieval`) | ACRA: adaptive chain retrieval for multimodal VQA (Zhang et al.) | 334, 115136 (2026) | 10.1016/j.knosys.2025.115136 | RW retrieval |
| `crgs2026causal` | Causal reasoning meets heuristic strategies for RAG (Luo et al.) | 333, 114976 (2026) | 10.1016/j.knosys.2025.114976 | RW retrieval |
| `siddharth2024retrieval` | RAG using engineering design knowledge (Siddharth, Luo) | 303, 112410 (2024) | 10.1016/j.knosys.2024.112410 | RW retrieval |
| `morenocediel2026optimising` | Growing-window semantic chunking for RAG (Moreno-Cediel et al.) | 331, 114896 (2026) | 10.1016/j.knosys.2025.114896 | RW retrieval |
| `peng2025multimodality` (`ant2025negative`) | Multi-modality re-learning against negative transfer (Peng et al.) | 330, 114479 (2025) | 10.1016/j.knosys.2025.114479 | RW negative transfer |
| `liang2026parameter` | Selective cross-domain distillation for audio transformers (Liang et al.) | 332, 114893 (2026) | 10.1016/j.knosys.2025.114893 | RW negative transfer |
| `mousavi2025vsi` | VSI: Bayesian feature selection with the Vendi score (Mousavi, Khalili) | 311, 112973 (2025) | 10.1016/j.knosys.2025.112973 | RW subset selection |
| `xie2025partial` | Partial multi-label feature selection (Xie et al.) | 326, 114077 (2025) | 10.1016/j.knosys.2025.114077 | RW subset selection |
| `yang2025crosscity` | Cross-city transfer learning for traffic forecasting (Yang et al.) | 315, 113336 (2025) | 10.1016/j.knosys.2025.113336 | setup context |
| `yan2026freqmamba` | FreqMamba: frequency-aware multi-graph fusion (Yan et al.) | 347, 116316 (2026) | 10.1016/j.knosys.2026.116316 | setup context |
| `zhou2025transformer` | Transformer with fusion spatiotemporal attention (Zhou et al.) | 329, 114466 (2025) | 10.1016/j.knosys.2025.114466 | setup context |
| `yao2025shkd` | SHKD: sub-hypergraph and knowledge distillation (Yao et al.) | 312, 113163 (2025) | 10.1016/j.knosys.2025.113163 | setup context |

Result: the bibliography now holds 73 entries of which 69 are cited, 17 of them
recent *Knowledge-Based Systems* articles. The manuscript states no numerical
comparison against these works; the Related Work positions R-MUR against them
through the decision object (state-conditioned marginal effect on end-task
loss) rather than through shared benchmarks.

Also fixed in this pass: seven `\citet{}` commands rendered as `(author?)`
under the numeric Elsevier style; all were rewritten as `\citep{}` with
non-textual phrasing, and the string no longer appears anywhere in the PDF.


## Baseline adaptations (2026-09-22)

Five deployable baselines were added from the feature-selection and
active-feature-acquisition literature. Each is cited to the record that was
retrieved and checked: arXiv entries through the arXiv API
(``export.arxiv.org``), journal entries through Crossref. Where the retrieved
metadata did not state a venue, the entry is given as an arXiv preprint rather
than asserting a conference.

| Key | Source | Verified metadata | Role |
|---|---|---|---|
| `zhu2025best` | Best Subset Selection: Optimal Pursuit | arXiv:2501.16815, ICML 2025 (stated in the record) | greedy validation set, with the elimination criterion audited |
| `taguchi2025adaptive` | Adaptive-$k$ context selection | arXiv:2506.08479, EMNLP 2025 (stated in the record) | relevance ordering with the budget chosen on validation |
| `qin2023iterative` | Iterative Demonstration Selection | arXiv:2310.09881 | cluster-based selection |
| `valancius2023acquisition` | Acquisition Conditioned Oracle | arXiv:2302.13960 | non-greedy acquisition |
| `norcliffe2025stochastic` | Stochastic Encodings for AFA | arXiv:2508.01957, ICML 2025 (stated in the record) | non-greedy acquisition |
| `zhang2025linear` | Linear-Time Demonstration Selection via Gradient Estimation | arXiv:2508.19999, EMNLP 2025 (stated in the record) | gradient-influence scoring |
| `yu2024rankrag` | RankRAG | arXiv:2407.02485 | reranking context (positioning) |
| `zhang2025genicl` | GenICL: demonstrations preferred by LLMs | arXiv:2505.19966 | preference-based selection (positioning) |
| `wang2024learning` | Learning to Cut (HEM) | IEEE TPAMI 46(12):9697--9713, 2024, 10.1109/TPAMI.2024.3432716 | formulation neighbour: which and how many to select |
| `wang2026dynamic` | Feature Selection via Dynamic Feature Graph | IEEE TKDE 38(3):1754--1767, 2026, 10.1109/TKDE.2026.3656587 | feature-selection neighbour (positioning) |

Run set: `results/raw/R205_fixed_set_baselines_s3_20260921` (576 target-runs,
six datasets, 32 targets, three seeds), produced by
`scripts/run_fixed_set_baselines.py`. Every policy chooses one set per target
from the training and validation splits and applies it to all test episodes;
no test label is observed and no model is trained.
