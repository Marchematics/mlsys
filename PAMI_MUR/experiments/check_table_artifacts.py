"""Verify that the numbers printed in the manuscripts match their artifacts.

The claim discipline of this project is that every number in the papers is
recoverable from a file. This script enforces it mechanically for the tables
that carry the headline results, by re-parsing the LaTeX and comparing each
cell against the derived artifact it is supposed to come from. It is meant to
be run before every submission.

Usage
-----
python experiments/check_table_artifacts.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/root/icl_ess_threshold")
PAMI_SEC = ROOT / "PAMI_MUR/paper/sections/6_experiments.tex"
TKDE_SEC = ROOT / "TKDE_MUR/paper/sections/7_experiments.tex"
KBS_CONSISTENCY = ROOT / "KBS_MUR/paper/tables/tab_consistency.tex"
KBS_SEC = ROOT / "KBS_MUR/paper/sections/7_discussion.tex"

DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
LABEL = {"METRLA": "METR-LA", "PEMSBAY": "PEMS-BAY"}
NUMBER = re.compile(r"[+-]?(?:\d+)?\.\d+")
TOL = 6e-4


def latex_rows(text: str, width: int) -> dict[str, list[float]]:
    """Return {first cell: numeric cells} for rows with exactly `width` numbers."""

    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z0-9\- ()+]+?)\s*&(.*?)\\\\\s*$", line)
        if not match:
            continue
        values = [float(x) for x in NUMBER.findall(match.group(2))]
        if len(values) == width:
            out[match.group(1).strip()] = values
    return out


def close(a: float, b: float) -> bool:
    return abs(a - b) < TOL


def check_deployable() -> tuple[int, int]:
    """tab:deployable -- six policies x six datasets + average rank."""

    matrix = json.loads(
        (ROOT / "PAMI_MUR/results/derived/R134_six_dataset_matrix/six_dataset_matrix.json").read_text()
    )["datasets"]
    policies = {
        "relevance": "Relevance (kNN)",
        "mmr": "MMR",
        "static_utility": "Standalone Utility",
        "cached_mur": "Cached-only router",
        "pool_all": "Pool all 16",
        "ranked_response_q4": "R-MUR (ours)",
    }
    ok = total = 0
    for paper in (PAMI_SEC, TKDE_SEC):
        text = paper.read_text()
        for key, label in policies.items():
            match = re.search(re.escape(label) + r"\s*&(.*?)\\\\", text, re.S)
            if not match:
                print(f"  MISSING row {label} in {paper.name}")
                total += 1
                continue
            cells = match.group(1).replace("\\textbf{", "").replace("}", "")
            got = [float(x) for x in NUMBER.findall(cells)][:6]
            expected = [matrix[ds]["policies"][key]["mean"] for ds in DATASETS]
            total += 1
            if len(got) == 6 and all(close(a, b) for a, b in zip(got, expected)):
                ok += 1
            else:
                print(f"  MISMATCH {label} in {paper.name}: {got} vs {[round(e, 4) for e in expected]}")
            # average rank
            ranks = {k: [] for k in policies}
            for ds in DATASETS:
                means = {k: matrix[ds]["policies"][k]["mean"] for k in policies}
                for k in policies:
                    ranks[k].append(1 + sum(1 for j in policies if means[j] > means[k]))
            expected_rank = sum(ranks[key]) / len(DATASETS)
            series = [float(x) for x in NUMBER.findall(cells)]
            total += 1
            # the printed rank carries two decimals, so allow half a unit in the
            # last place rather than the value tolerance used for gains
            if series and abs(series[-1] - expected_rank) < 6e-3:
                ok += 1
            else:
                print(f"  MISMATCH rank {label} in {paper.name}: {series[-1:]} vs {expected_rank:.2f}")
    return ok, total


def check_headroom() -> tuple[int, int]:
    """tab:headroom-real -- mean, median, positive fraction per dataset."""

    head = json.loads(
        (ROOT / "PAMI_MUR/results/derived/R134_six_dataset_matrix/headroom.json").read_text()
    )["datasets"]
    ok = total = 0
    for paper in (PAMI_SEC, TKDE_SEC):
        text = paper.read_text()
        for ds, record in head.items():
            label = LABEL.get(ds, ds)
            expected = (record["mean"], record["median"], record["positive_fraction"])
            found = None
            for match in re.findall(re.escape(label) + r"\s*&(.*?)\\\\", text, re.S):
                values = [float(x) for x in NUMBER.findall(match)]
                if len(values) == 3 and close(values[0], expected[0]):
                    found = values
                    break
            total += 1
            if found and all(close(a, b) for a, b in zip(found, expected)):
                ok += 1
            else:
                print(f"  MISMATCH headroom {label} in {paper.name}: {found} vs {expected}")
    return ok, total


def check_consistency() -> tuple[int, int]:
    """tab:consistency -- PAMI and TKDE must equal the generated KBS table."""

    generated = latex_rows(KBS_CONSISTENCY.read_text(), 10)
    ok = total = 0
    for paper in (PAMI_SEC, TKDE_SEC):
        rows = latex_rows(paper.read_text(), 6)
        for ds, row in generated.items():
            expected = [row[0], row[3], row[6], row[7], row[8], row[9]]
            got = rows.get(LABEL.get(ds, ds))
            total += 1
            if got and all(close(a, b) for a, b in zip(got, expected)):
                ok += 1
            else:
                print(f"  MISMATCH consistency {ds} in {paper.name}: {got} vs {expected}")
    return ok, total


def check_paired() -> tuple[int, int]:
    """The six-dataset verdicts quoted in PAMI must match R146."""

    paired = json.loads(
        (ROOT / "PAMI_MUR/results/derived/R146_six_dataset_paired/paired.json").read_text()
    )["datasets"]
    text = PAMI_SEC.read_text().replace("$", "")
    ok = total = 0
    for ds, record in paired.items():
        total += 1
        # the papers print values without a leading zero (.0067, not 0.0067)
        token = f"{record['rmur_vs_best']['mean']:+.4f}".lstrip("+").replace("0.", ".", 1)
        if token in text:
            ok += 1
        else:
            print(f"  MISMATCH paired verdict {ds}: {token} not found in PAMI")
    return ok, total


def check_recent_baselines() -> tuple[int, int]:
    """The DELIFT-style competitor and the theory-limit negative must appear
    with the values their artifacts report."""

    import json as _json

    ok = total = 0
    delfit = _json.loads((ROOT / "PAMI_MUR/results/derived/R152_delfit_competitor.json").read_text())
    geometry = _json.loads(
        (ROOT / "PAMI_MUR/results/derived/R151_response_geometry/response_geometry.json").read_text()
    )
    for paper in (PAMI_SEC, TKDE_SEC):
        text = paper.read_text().replace("$", "")
        # DELIFT-style: at least the METR-LA delta must be quoted
        token = f"{delfit['datasets']['METRLA']['delfit_vs_pool']['mean']:+.4f}".lstrip("+").replace("0.", ".", 1)
        total += 1
        if token[:6] in text:
            ok += 1
        else:
            print(f"  MISMATCH DELIFT delta {token} not in {paper.name}")
    # theory-limit refutation: min_disturbance is below pooling on both datasets
    for ds in ("METRLA", "PEMSBAY"):
        record = geometry["datasets"][ds]["verdict"]
        total += 1
        if not record["prediction_beats_pooling"] and record["min_disturbance_delta_vs_pool"] < 0:
            ok += 1
        else:
            print(f"  MISMATCH theory-limit verdict {ds}: {record}")
    return ok, total


def check_headroom_predictiveness() -> tuple[int, int]:
    """The headroom-predictiveness negative must be quoted with the artifact's
    values, and must actually be a negative."""

    import json as _json

    record = _json.loads(
        (ROOT / "PAMI_MUR/results/derived/R153_headroom_predictiveness/headroom_predictiveness.json").read_text()
    )
    ok = total = 0
    pooled = record["pooled"]
    total += 1
    if pooled["targets"] == 192 and abs(pooled["pearson_within_dataset"]["r"]) < 0.10:
        ok += 1
    else:
        print(f"  MISMATCH headroom predictiveness: {pooled}")
    # each dataset's paired effect size quoted in the papers must be the artifact's
    for paper in (PAMI_SEC, TKDE_SEC):
        text = paper.read_text().replace("$", "")
        for ds in ("PEMSBAY", "PEMS03", "PEMS08"):
            t = record["datasets"][ds]["t_statistic"]
            token = f"{abs(t):.1f}"
            total += 1
            if token in text:
                ok += 1
            else:
                print(f"  MISMATCH t({ds})={token} not in {paper.name}")
    return ok, total


def check_pilot_power() -> tuple[int, int]:
    """The pilot operating characteristics must be the artifact's values."""

    import json as _json

    record = _json.loads(
        (ROOT / "PAMI_MUR/results/derived/R154_pilot_power/pilot_power.json").read_text()
    )
    ok = total = 0
    text = PAMI_SEC.read_text().replace("$", "")
    tkde = TKDE_SEC.read_text().replace("$", "")
    for size in ("4", "8", "16", "24"):
        entry = record["by_size"][size]
        for key in ("decision_agreement", "sensitivity_for_AHEAD", "specificity_for_AHEAD"):
            token = f"{entry[key]:.3f}".lstrip("0")
            total += 1
            if token in text or token in tkde:
                ok += 1
            else:
                print(f"  MISMATCH pilot n={size} {key}={token} not quoted")
    return ok, total


def check_pilot_shift() -> tuple[int, int]:
    """The time-shift numbers must be the artifact's, and the direction claim
    must actually hold (correlation positive and significant)."""

    import json as _json

    record = _json.loads(
        (ROOT / "PAMI_MUR/results/derived/R155_pilot_shift/pilot_shift.json").read_text()
    )
    ok = total = 0
    pooled = record["pooled"]
    total += 1
    if pooled["targets"] == 192 and pooled["pearson_within_dataset"]["r"] > 0.5:
        ok += 1
    else:
        print(f"  MISMATCH pilot shift pooled: {pooled['pearson_within_dataset']}")
    total += 1
    if record["dataset_verdict_agreement"] == "4/6":
        ok += 1
    else:
        print(f"  MISMATCH verdict agreement: {record['dataset_verdict_agreement']}")
    # every dataset must keep the sign relation positive for the claim to hold
    for ds, entry in record["datasets"].items():
        total += 1
        if entry["pearson"]["r"] > 0 and entry["sign_agreement"] >= 0.7:
            ok += 1
        else:
            print(f"  MISMATCH pilot shift {ds}: r={entry['pearson']['r']:.3f} sign={entry['sign_agreement']:.2f}")
    return ok, total


def check_pilot_consistency() -> tuple[int, int]:
    """The split-half filter's quoted numbers must be the artifact's, and the
    filter must actually be high precision and low coverage."""

    import json as _json

    record = _json.loads(
        (ROOT / "PAMI_MUR/results/derived/R156_pilot_consistency/pilot_consistency.json").read_text()
    )
    text = (PAMI_SEC.read_text() + TKDE_SEC.read_text() + KBS_SEC.read_text()).replace("$", "")
    # The manuscripts print rates as percentages with one decimal ("99.0\%") and
    # coverages as whole percentages ("35\%"), so the expected token is derived
    # from the artifact rather than hard-coded.
    claims = []
    for size, key in (("16", "strict_agreement"), ("16", "inconsistent_agreement"),
                      ("24", "strict_agreement"), ("24", "inconsistent_agreement")):
        claims.append((size, key, f"{record['by_size'][size][key] * 100:.1f}\\%"))
    for size in ("16", "24"):
        key = "strict_coverage"
        claims.append((size, key, f"{round(record['by_size'][size][key] * 100)}\\%"))
    ok = total = 0
    for size, key, token in claims:
        total += 1
        if token in text:
            ok += 1
        else:
            print(f"  MISMATCH consistency n={size} {key}: {token} not quoted")
    # the filter must separate the two groups at every size, or the claim fails
    for size in ("8", "16", "24"):
        entry = record["by_size"][size]
        total += 1
        if entry["strict_agreement"] > entry["inconsistent_agreement"] + 0.05:
            ok += 1
        else:
            print(f"  MISMATCH filter does not separate at n={size}")
    # and it must be low coverage, which is what makes it a filter
    for size in ("16", "24"):
        total += 1
        if record["by_size"][size]["strict_coverage"] < 0.5:
            ok += 1
        else:
            print(f"  MISMATCH filter is not selective at n={size}")
    return ok, total


def check_pilot_external() -> tuple[int, int]:
    """The external-validity claims must match the artifact, including the
    limitation: every contrast in that family is one-sided."""

    import json as _json

    record = _json.loads(
        (ROOT / "PAMI_MUR/results/derived/R157_pilot_external/pilot_external.json").read_text()
    )
    ok = total = 0
    for name, entry in record["contrasts"].items():
        total += 1
        if entry["pooled"]["benchmark_verdict_agreement"] == "5/5":
            ok += 1
        else:
            print(f"  MISMATCH external {name}: {entry['pooled']['benchmark_verdict_agreement']}")
    # the limitation must be real: no contrast may be positive
    total += 1
    positives = [n for n, e in record["contrasts"].items() if e["pooled"]["mean_difference"] > 0]
    if not positives:
        ok += 1
    else:
        print(f"  MISMATCH: contrasts are two-sided ({positives}); the stated limitation is wrong")
    # the marginal-contrast collapse must be present on the smallest benchmark
    total += 1
    cifar10 = record["contrasts"]["relevance_minus_pool"]["benchmarks"]["cifar10"]
    if cifar10["verdict_agreement"] < 0.5:
        ok += 1
    else:
        print(f"  MISMATCH cifar10 marginal split agreement: {cifar10['verdict_agreement']}")
    return ok, total


def check_cross_paper() -> tuple[int, int]:
    """Delegate to the cross-paper checker: shared numbers must not drift."""

    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "PAMI_MUR/experiments/check_cross_paper.py")],
        capture_output=True, text=True,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "no output"
    if result.returncode == 0:
        return 1, 1
    print(f"  cross-paper check failed: {tail}")
    return 0, 1


def check_gate_ceiling() -> tuple[int, int]:
    """The episode-gate ceiling must be the artifact's, and must be reported as
    headroom rather than as achieved performance."""

    import json as _json

    record = _json.loads(
        (ROOT / "PAMI_MUR/results/derived/R158_gate_ceiling.json").read_text()
    )
    ok = total = 0
    for ds, entry in record["datasets"].items():
        total += 1
        if entry["gate_minus_pool"] > 0 and 0.3 < entry["router_loses_fraction"] < 0.6:
            ok += 1
        else:
            print(f"  MISMATCH gate ceiling {ds}: {entry}")
    # the ceiling must be materially larger than the router's advantage: that is
    # what makes the claim "headroom worth building for" true
    for ds in ("PEMS08", "METRLA"):
        entry = record["datasets"][ds]
        total += 1
        advantage = abs(entry["router"] - entry["pool"])
        if entry["gate_minus_pool"] > 2 * advantage:
            ok += 1
        else:
            print(f"  MISMATCH headroom not a multiple of the advantage on {ds}")
    return ok, total


def check_gate_negative() -> tuple[int, int]:
    """The gate negative must be real: every signal at chance, no gate better
    than pooling by a meaningful margin, and the oracle ceiling still large."""

    import json as _json

    record = _json.loads((ROOT / "PAMI_MUR/results/derived/R159_gate/gate.json").read_text())
    ok = total = 0
    for ds, entry in record["datasets"].items():
        total += 1
        aucs = [entry["signals"][name]["auc"] for name in entry["signals"]]
        if all(0.45 < a < 0.55 for a in aucs):
            ok += 1
        else:
            print(f"  MISMATCH gate {ds}: AUCs outside chance band {aucs}")
        # the ceiling must be materially larger than anything realised
        total += 1
        best = max(entry["signals"][n]["gate_minus_pool"] for n in entry["signals"])
        ceiling = entry["oracle_gate_mean"] - entry["pool_mean"]
        if ceiling > 3 * max(best, 1e-9):
            ok += 1
        else:
            print(f"  MISMATCH gate {ds}: realised {best:+.4f} vs ceiling {ceiling:+.4f}")
    return ok, total


def main() -> int:
    suites = [
        ("tab:deployable (values + average rank)", check_deployable),
        ("tab:headroom-real", check_headroom),
        ("tab:consistency (vs generated KBS table)", check_consistency),
        ("six-dataset paired verdicts", check_paired),
        ("recent baselines and theory-limit negative", check_recent_baselines),
        ("headroom-predictiveness negative", check_headroom_predictiveness),
        ("pilot operating characteristics", check_pilot_power),
        ("pilot time-shift transfer", check_pilot_shift),
        ("split-half pilot filter", check_pilot_consistency),
        ("pilot external validity", check_pilot_external),
        ("cross-paper consistency", check_cross_paper),
        ("episode-gate headroom ceiling", check_gate_ceiling),
        ("gate signal is at chance (negative)", check_gate_negative),
    ]
    failures = 0
    for name, fn in suites:
        ok, total = fn()
        status = "PASS" if ok == total else "FAIL"
        print(f"{status}  {name}: {ok}/{total}")
        failures += ok != total
    print("\nall table numbers match their artifacts" if not failures else f"\n{failures} suite(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
