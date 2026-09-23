# KBS submission checklist

Status of the `KBS_MUR` manuscript against the *Knowledge-Based Systems*
(Elsevier) submission requirements. Items marked TODO still need author input;
everything else is produced by the build in this directory.

## Required artifacts

| Item | Status | Location |
|---|---|---|
| Manuscript source (LaTeX, `elsarticle`) | ready | `main.tex` |
| Manuscript PDF | ready | `main.pdf` |
| Supplementary material | ready | `supplement.tex`, `supplement.pdf` |
| Highlights (3--5 bullets, $\leq$ 85 characters each) | ready | `highlights.txt` |
| Cover letter | ready, aligned with the final claims | `cover_letter.md` |
| Figures as separate files | ready | `submission/Fig1.pdf` ... `submission/Fig5.pdf` |
| Bibliography (`.bbl` included so the portal build resolves references) | ready | `main.bbl` |
| CRediT authorship contribution statement | author names filled, roles still bracketed | `main.tex` |
| Declaration of competing interest | ready | `main.tex` |
| Data availability statement | placeholder for repository URL | `main.tex` |
| Generative-AI declaration | required if an AI tool was used for manuscript preparation; template with tool-name placeholder in `main.tex` | `main.tex` |
| Funding statement | generic no-specific-grant wording; confirm whether a grant applies | `main.tex` |
| Author names, affiliation, corresponding-author e-mails | filled in `main.tex` (Wen; Zhang and Chen corresponding) |
| Suggested and opposed reviewers | TODO | Editorial Manager portal |

## Formatting checks

- Document class: Elsevier `elsarticle` (`preprint` for the working copy,
  `review` for the line-numbered submission copy), journal set to
  *Knowledge-Based Systems*, numbered references (`elsarticle-num`).
- Abstract: one paragraph, 160 words, no numerical results, no internal run
  labels.
- Keywords: five terms; at most six are allowed.
- Figures: five figures (R-MUR overview, structural headroom, multi-target
  results, mechanisms, shortlist diagnostics), vector PDF at the 5.4 in column
  width; scripts and raw sources are listed in `figures/SOURCES.md`.
- Tables: generated from raw runs by `../scripts/build_kbs_result_tables.py`;
  every fragment records its source path in a leading comment.
- No undefined citations or cross-references in either PDF.
- Related work: four continuous paragraphs; Introduction: six paragraphs.
- Bibliography: 73 entries, 69 cited, of which 17 are recent *Knowledge-Based
  Systems* articles (2024--2026); every added entry was verified against
  Crossref. None of them is used as a baseline: the tasks differ, so they
  position the contribution rather than support a numerical comparison.

## What is not uploaded

The upload consists of `main.tex`, `main.bbl`, `references.bib`, `math_commands.tex`,
`sections/`, `tables/`, `figures/*.pdf`, `supplement.tex`, `supplement_tables.tex`,
`highlights.txt`, and `submission/Fig1--4.pdf`. The files below stay in the
repository and are deliberately excluded:

- the provenance comments in `tables/*.tex` name internal run identifiers, but
  they are LaTeX comments and never reach the compiled PDF;
- `figures/SOURCES.md`, `SUBMISSION_CHECKLIST.md`, and the scripts under
  `../scripts/` and `figures/` document how the results were produced;
- `refine-logs/` holds the development record.

## Reproducing the package

```bash
cd ../scripts
python build_kbs_result_tables.py     # refresh paper/tables/*.tex from raw runs
cd ../paper/figures
for s in generate_fig1_overview.py generate_fig2_headroom.py \
         generate_fig3_multitarget_delta.py generate_fig4_mechanism.py; do python "$s"; done
cd ..
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex           # working copy, 23 pp
latexmk -pdf -interaction=nonstopmode -halt-on-error main_review.tex    # review copy, 1.5 spacing + line numbers, 32 pp
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex     # 12 pp
```

Both manuscripts use the Elsevier `elsarticle` class with
`\journal{Knowledge-Based Systems}` and the `elsarticle-num` bibliography
style. `main_review.tex` only defines `\classoption{review}` and inputs
`main.tex`, so the two builds cannot drift apart.

## Review model and the manuscript copy

Knowledge-Based Systems runs **single anonymized (single-blind) review** by
default: reviewers see the authors. The manuscript is built in the layout
Elsevier expects for review (1.5 line spacing, line numbers) from a single
source, so there is no separate preprint or review copy to keep in sync.

| Item | Status |
|---|---|
| `main.pdf` | the submission manuscript, 39 pages, review layout, complete author block |
| `supplement.pdf` | supplementary material, 11 pages |
| Double-anonymized copy | not maintained. `\anonymousmode` in `main.tex` removes the author block, affiliations, e-mails, corresponding-author markers and the CRediT names; a copy is `\def\anonymousmode{1}\input{main.tex}` |
| Self-citations | none of the prose refers to the authors' own work in the first person, so no rewriting is needed under either model |
| Acknowledgements, funding, data availability | contain no author-identifying information |

## Generative-AI disclosure

- The manuscript does **not** carry an AI declaration. This is an author
  decision recorded here so it is not re-added by accident.
- Elsevier's policy requires a declaration when an AI tool was used for
  manuscript preparation and made substantive changes to sentence structure or
  organization; basic grammar, spelling and punctuation checks do not require
  one. Source: <https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals>.
- The authors are responsible for that determination. If a declaration is
  wanted later, the policy template is:
  `During the preparation of this work, the author(s) used [TOOL] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.`
- AI tools may not be listed as authors.

## Remaining author-side items

- CRediT roles for the three authors (names are in place, roles are bracketed).
- ORCID iDs for the corresponding authors.
- Funding statement: confirm whether a grant applies; the current wording
  states that no specific grant was received.
- Suggested reviewers for the Editorial Manager portal (optional).
- Author biographies are drafted in `paper/author_bios.tex` and are not part of
  the manuscript PDF; copy them into the portal fields.

## Portal notes

- Highlights are uploaded as a separate file named `Highlights`; the portal
  expects a Word file, so paste `highlights.txt` into a `.docx` before upload.
- The Editorial Manager build must be approved after checking that the merged
  PDF resolves the bibliography.
- The journal screens on scope before review: the cover letter states the
  knowledge-driven decision-support contribution in its first paragraph.
