#!/usr/bin/env python3
"""R124 derived summary: signed-curvature head vs bilinear and frozen heads.

Reuses the R112 aggregation machinery (``analyze_r112``) with an extended policy
list that also contains the frozen-head reference policies
(``ranked_response_frozen_*``), and adds a paired head-versus-head table and the
pooled contrasts.  ``analyze_r112`` itself is not modified: its policy list is
extended in this module's namespace before its functions are called.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
ROOT = PAMI.parent
KBS = ROOT / "KBS_MUR"
for _path in (HERE, PAMI / "experiments", KBS / "src", KBS / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import analyze_r112  # noqa: E402
from analyze_r110 import load_cells  # noqa: E402
from run_traffic_response_router_sweep import paired_ci  # noqa: E402

FROZEN_STEMS = ["ranked_response_frozen_q2", "ranked_response_frozen_q4", "ranked_response_frozen_q8", "ranked_response_frozen_full"]
BILINEAR_STEMS = ["ranked_response_bilinear_q2", "ranked_response_bilinear_q4", "ranked_response_bilinear_q8", "ranked_response_bilinear_full"]
SIGNED_STEMS = ["ranked_response_q2", "ranked_response_q4", "ranked_response_q8", "ranked_response_full"]
EXTENDED_POLICIES = (
    analyze_r112.POLICIES
    + FROZEN_STEMS
    + [stem + "_forced" for stem in FROZEN_STEMS]
    + BILINEAR_STEMS
    + [stem + "_forced" for stem in BILINEAR_STEMS]
)
analyze_r112.POLICIES = EXTENDED_POLICIES
analyze_r112.GATE_POLICIES = [name for name in EXTENDED_POLICIES if name.startswith("ranked_response")]


def paired_head_table(name: str, cells: list[dict]) -> dict:
    """Paired signed-vs-bilinear-vs-frozen contrasts on identical cells."""

    gains = {
        policy: np.concatenate([cell["gains"][policy] for cell in cells]) for policy in EXTENDED_POLICIES
    }
    classes = np.concatenate([cell["gains"]["class_index"] for cell in cells])
    results = [cell["result"] for cell in cells]
    table = {}
    pairs = list(zip(SIGNED_STEMS, BILINEAR_STEMS)) + list(zip(SIGNED_STEMS, FROZEN_STEMS))
    for bilinear, frozen in pairs:
        for suffix in ("", "_forced"):
            left, right = bilinear + suffix, frozen + suffix
            delta = gains[left] - gains[right]
            static_delta = gains[left] - gains["static_utility"]
            cached_delta = gains[left] - gains["cached_mur"]
            table[left] = {
                "head_vs_head": paired_ci(delta),
                "head_vs_head_hierarchical": analyze_r112.hierarchical_ci(delta, classes, n_boot=2000, seed=31),
                "gain": paired_ci(gains[left]),
                "delta_static": paired_ci(static_delta),
                "delta_cached": paired_ci(cached_delta),
                "class_fraction_above_static": float(
                    np.mean(
                        [
                            np.mean(gains[left][classes == value] - gains["static_utility"][classes == value]) > 0
                            for value in np.unique(classes)
                        ]
                    )
                ),
                "mean_selected": float(
                    np.mean([result["metrics"][left]["mean_selected_contexts"] for result in results])
                ),
                "mean_distinct_clusters": float(
                    np.mean([result["metrics"][left]["mean_distinct_clusters"] for result in results])
                ),
            }
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=PAMI / "results/derived/R123_bilinear_second_family_summary")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--redundancy",
        type=Path,
        default=None,
        help="optional recomputed redundancy diagnostics to override the per-cell values",
    )
    args = parser.parse_args()

    cells = load_cells(args.run_root)
    if not cells:
        raise SystemExit(f"no cells under {args.run_root}")
    override = None
    if args.redundancy is not None and args.redundancy.exists():
        override = json.loads(args.redundancy.read_text(encoding="utf-8")).get("cells")
    args.out_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "derived_from": str(args.run_root),
        "run_id": args.run_root.name,
        "bootstrap_replicates": args.bootstrap,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    benchmarks = {
        name: analyze_r112.aggregate_benchmark(
            name, records, bootstrap=args.bootstrap, redundancy_override=override
        )
        for name, records in cells.items()
    }
    pooled = analyze_r112.pooled_across_benchmarks(benchmarks, bootstrap=args.bootstrap)
    head_tables = {name: paired_head_table(name, records) for name, records in cells.items()}

    summary = {"provenance": provenance, "benchmarks": benchmarks, "pooled": pooled, "head_contrasts": head_tables}
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    markdown = [
        f"<!-- generated from {args.run_root} at {provenance['generated_utc']} -->",
        "",
        "### Signed-curvature (primary) vs fixed-sign bilinear vs frozen scalar head, q=4",
        "",
        "| benchmark | head | gain [95% CI] | Δ Static [CI] | Δ Cached [CI] | cls>Static | sel | pass vs Static |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, aggregate in benchmarks.items():
        for policy, label in (
            ("ranked_response_q4_forced", "signed forced"),
            ("ranked_response_q4", "signed thresholded"),
            ("ranked_response_bilinear_q4_forced", "bilinear forced"),
            ("ranked_response_frozen_q4_forced", "frozen forced"),
        ):
            entry = aggregate["gate"][policy]
            metrics = aggregate["policy_metrics"][policy]["metrics_from_cells"]
            markdown.append(
                "| {b} | {lab} | {g:.4f} [{gl:.4f},{gh:.4f}] | {ds:.4f} [{dsl:.4f},{dsh:.4f}] | "
                "{dc:.4f} [{dcl:.4f},{dch:.4f}] | {fs:.2f} | {sel:.2f} | {p} |".format(
                    b=name,
                    lab=label,
                    g=entry["gain_over_zero"]["mean"],
                    gl=entry["gain_over_zero"]["low"],
                    gh=entry["gain_over_zero"]["high"],
                    ds=entry["gain_over_static"]["mean"],
                    dsl=entry["gain_over_static"]["low"],
                    dsh=entry["gain_over_static"]["high"],
                    dc=entry["gain_over_cached"]["mean"],
                    dcl=entry["gain_over_cached"]["low"],
                    dch=entry["gain_over_cached"]["high"],
                    fs=entry["class_fraction_above_static"],
                    sel=metrics["mean_selected_contexts"],
                    p="yes" if entry["pass_gate_paired"] else "no",
                )
            )
    markdown += ["", "### Paired head contrasts (identical cells)", "",
                 "| benchmark | policy | signed − bilinear [95% CI] | signed − frozen [95% CI] |", "|---|---|---|---|"]
    for name, table in head_tables.items():
        for key in ("ranked_response_q4", "ranked_response_q4_forced"):
            entry = table[key]
            if key not in table:
                continue
            markdown.append(
                f"| {name} | {key.replace('ranked_response_', '')} | "
                f"{entry['head_vs_head']['mean']:+.4f} [{entry['head_vs_head']['low']:+.4f},{entry['head_vs_head']['high']:+.4f}] |"
            )
    markdown += ["", "### Pooled across benchmarks (equal weight per benchmark)", "",
                 "| statistic | mean | 95% CI | benchmarks positive |", "|---|---|---|---|"]
    pooled_rows = [
        ("R-MUR bilinear q=4 forced − Static", "delta_static_forced", "benchmarks_above_static_forced"),
        ("R-MUR bilinear q=4 forced − Cached", "delta_cached_forced", "benchmarks_above_cached_forced"),
        ("R-MUR bilinear q=4 forced − Static (reference)", None, None),
        ("R-MUR frozen q=4 forced − Static (reference)", None, None),
        ("H_state", "h_state", "benchmarks_h_state_positive"),
    ]
    for label, key, count_key in pooled_rows:
        if key is None:
            reference = "ranked_response_bilinear_q4_forced" if "bilinear" in label else "ranked_response_frozen_q4_forced"
            values = np.asarray(
                [
                    aggregate["gate"][reference]["gain_over_static"]["mean"]
                    for aggregate in benchmarks.values()
                ]
            )
            rng = np.random.default_rng(17)
            draws = values[rng.integers(0, values.size, size=(args.bootstrap, values.size))].mean(axis=1)
            markdown.append(
                f"| {label} | {values.mean():+.4f} | [{np.quantile(draws, 0.025):+.4f},{np.quantile(draws, 0.975):+.4f}] | "
                f"{int(np.sum(values > 0))}/{values.size} |"
            )
            continue
        entry = pooled[key]
        markdown.append(
            f"| {label} | {entry['mean']:+.4f} | [{entry['low']:+.4f},{entry['high']:+.4f}] | {pooled[count_key]}/{len(benchmarks)} |"
        )
    markdown += ["", "### Headroom and redundancy diagnostics", "",
                 "| benchmark | anchor CE | oracle_greedy | oracle_static | H_state | m(1st) | m(2nd) | 2nd/1st |",
                 "|---|---|---|---|---|---|---|---|"]
    for name, aggregate in benchmarks.items():
        head = aggregate["headroom"]
        red = aggregate["redundancy"]
        markdown.append(
            f"| {name} | {head['anchor_only_mean_ce']:.4f} | {head['oracle_greedy_gain']:.4f} | "
            f"{head['oracle_static_gain']:.4f} | {head['h_state']:.5f} | {red['mean_first_member_marginal']:.4f} | "
            f"{red['mean_second_member_marginal']:.4f} | {red['second_over_first_ratio']:.4f} |"
        )
    markdown.append("")
    (args.out_root / "report_tables.md").write_text("\n".join(markdown), encoding="utf-8")

    gate_rows = [
        "benchmark,policy,gain,paired_lo,paired_hi,delta_static,ds_lo,ds_hi,delta_cached,dc_lo,dc_hi,"
        "class_frac_above_static,mean_selected,mean_distinct_clusters,pass_paired,pass_hier"
    ]
    for name, aggregate in benchmarks.items():
        for policy in analyze_r112.GATE_POLICIES:
            entry = aggregate["gate"][policy]
            metrics = aggregate["policy_metrics"][policy]["metrics_from_cells"]
            gate_rows.append(
                ",".join(
                    [
                        name,
                        policy,
                        f"{entry['gain_over_zero']['mean']:.6f}",
                        f"{entry['gain_over_zero']['low']:.6f}",
                        f"{entry['gain_over_zero']['high']:.6f}",
                        f"{entry['gain_over_static']['mean']:.6f}",
                        f"{entry['gain_over_static']['low']:.6f}",
                        f"{entry['gain_over_static']['high']:.6f}",
                        f"{entry['gain_over_cached']['mean']:.6f}",
                        f"{entry['gain_over_cached']['low']:.6f}",
                        f"{entry['gain_over_cached']['high']:.6f}",
                        f"{entry['class_fraction_above_static']:.4f}",
                        f"{metrics['mean_selected_contexts']:.3f}",
                        f"{metrics['mean_distinct_clusters']:.3f}",
                        str(entry["pass_gate_paired"]),
                        str(entry["pass_gate_hierarchical"]),
                    ]
                )
            )
    (args.out_root / "gate_table.csv").write_text("\n".join(gate_rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "bilinear_forced_minus_static_pooled": pooled["delta_static_forced"],
                "benchmarks_above_static_forced": pooled["benchmarks_above_static_forced"],
                "h_state_pooled": pooled["h_state"],
            },
            indent=1,
        )[:800],
        flush=True,
    )
    print(f"wrote derived summary to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
