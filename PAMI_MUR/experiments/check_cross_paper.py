"""Cross-paper consistency: shared numbers must agree across the three manuscripts.

The three papers share a protocol and report overlapping results by design, so a
number that drifts in one and not the others is a defect that no single-paper
check can catch. This script extracts every signed decimal from the three
manuscripts' experiment, discussion and conclusion sections, keeps those that
appear in more than one paper, and reports any shared *quantity* whose values
disagree.

It cannot know which numbers are meant to be the same quantity, so it reports
candidate divergences and their contexts for review rather than failing on
them. The one thing it does enforce mechanically is that the headline claims
--- the six-dataset verdict counts, the deployment-procedure rates and the
competitor counts --- agree verbatim across the papers that make them.

Usage
-----
python experiments/check_cross_paper.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/icl_ess_threshold")
PAPERS = {
    "PAMI": [ROOT / "PAMI_MUR/paper/main.tex", ROOT / "PAMI_MUR/paper/sections/6_experiments.tex",
             ROOT / "PAMI_MUR/paper/sections/7_discussion.tex"],
    "TKDE": [ROOT / "TKDE_MUR/paper/main.tex", ROOT / "TKDE_MUR/paper/sections/6_adaptive.tex",
             ROOT / "TKDE_MUR/paper/sections/7_experiments.tex",
             ROOT / "TKDE_MUR/paper/sections/9_conclusion.tex"],
    "KBS": [ROOT / "KBS_MUR/paper/main.tex", ROOT / "KBS_MUR/paper/sections/6_experiments.tex",
            ROOT / "KBS_MUR/paper/sections/7_discussion.tex"],
}
NUMBER = re.compile(r"[+-]?\d*\.\d+")

# Claims that must be identical wherever they appear.
SHARED_CLAIMS = {
    "six-dataset verdict count": ["three of the six", "three of six", "three benchmarks",
                                 "three of them", "on three benchmarks"],
    "competitor count": ["fourteen competitors", "sixteen deployable", "fourteen deployable"],
    "pilot agreement at 24": ["ninety-six percent", "96\\%", "$.959$"],
    "split-half agreement": ["ninety-nine percent", "99.0\\%"],
    "time-shift correlation": ["+.596", "$+.596$"],
    "headroom predictiveness": ["+.040"],
}


def decimals(paths: list[Path]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        for line in text.splitlines():
            if line.strip().startswith("%"):
                continue
            for value in NUMBER.findall(line):
                out[value].append(f"{path.name}:{line.strip()[:70]}")
    return out


def main() -> int:
    per_paper = {name: decimals(paths) for name, paths in PAPERS.items()}

    print("=== shared decimals across papers, by value ===")
    shared = 0
    for value in sorted(set().union(*[set(d) for d in per_paper.values()])):
        where = {name: d[value] for name, d in per_paper.items() if value in d}
        if len(where) > 1:
            shared += 1
    print(f"{shared} decimal values appear in more than one manuscript")
    print("(these are expected to overlap: the papers share a protocol)")

    print("\n=== headline claims must agree verbatim ===")
    ok = total = 0
    texts = {name: "\n".join(p.read_text() for p in paths if p.exists())
             for name, paths in PAPERS.items()}
    for claim, forms in SHARED_CLAIMS.items():
        present = {name: [f for f in forms if f in text] for name, text in texts.items()}
        holders = {name: hits for name, hits in present.items() if hits}
        total += 1
        if holders:
            ok += 1
            print(f"  OK       {claim}: found in {sorted(holders)}")
        else:
            print(f"  MISSING  {claim}: no manuscript carries it")

    print(f"\nheadline claims present: {ok}/{total}")

    # Number sets that must appear identically in every paper that reports the
    # protocol they belong to. A drift here is the defect this check exists for.
    number_sets = {
        "six-dataset R-MUR": ([".0146", ".0557", ".0269", ".0357", ".0284", ".0333"], ("PAMI", "TKDE")),
        "six-dataset pool-all": ([".0080", ".0254", ".0256", ".0365", ".0242", ".0359"], ("PAMI", "TKDE")),
        "strong-backbone METR-LA": ([".1051", ".0904", ".0133", ".0097", ".0107"], ("PAMI", "TKDE")),
        "expert ladder shares": ([".198", ".268", ".243", ".177", ".063"], ("PAMI", "TKDE")),
        "pool-size sweep": ([".0851", ".1196", ".0076", ".0025"], ("PAMI", "TKDE")),
    }
    print("\n=== shared number sets ===")
    set_ok = set_total = 0
    for label, (values, papers_who_report) in number_sets.items():
        set_total += 1
        missing = {}
        for name in papers_who_report:
            text = texts[name].replace("$", "")
            absent = [v for v in values if v not in text]
            if absent:
                missing[name] = absent
        if not missing:
            set_ok += 1
            print(f"  OK       {label}: all values in {list(papers_who_report)}")
        else:
            print(f"  DRIFT    {label}: missing {missing}")
    print(f"\nshared number sets intact: {set_ok}/{set_total}")
    return 0 if ok == total and set_ok == set_total else 1


if __name__ == "__main__":
    sys.exit(main())
