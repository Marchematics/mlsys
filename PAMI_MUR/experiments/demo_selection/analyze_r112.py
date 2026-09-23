#!/usr/bin/env python3
"""R112 derived summary: redundancy-structured pools + forced-budget ablation.

Reads one or two immutable R112 raw run directories (natural near-duplicate
pools and, optionally, the controlled perturbed-copy pools) and writes gates,
per-class consistency, hierarchical CIs, pooled CIs and redundancy diagnostics
under ``results/derived/R112_redundant_demo_summary``.

Aggregation helpers are imported from ``analyze_r110`` (which is not modified);
this file only extends the policy list and adds the pooled / redundancy tables.
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

from analyze_r110 import hierarchical_ci, load_cells, seed_level_ci  # noqa: E402
from run_traffic_response_router_sweep import paired_ci  # noqa: E402

BASE_POLICIES = [
    "base_only",
    "pool_all",
    "random",
    "relevance",
    "mmr",
    "facility_location",
    "dpp",
    "static_utility",
    "cached_mur",
    "oracle_static",
    "oracle_greedy",
]
RESPONSE_STEMS = ["ranked_response_q2", "ranked_response_q4", "ranked_response_q8", "ranked_response_full"]
POLICIES = BASE_POLICIES + RESPONSE_STEMS + [stem + "_forced" for stem in RESPONSE_STEMS]
GATE_POLICIES = [name for name in POLICIES if name.startswith("ranked_response")]


def aggregate_benchmark(
    name: str, cells: list[dict], *, bootstrap: int, redundancy_override: dict | None = None
) -> dict:
    results = [cell["result"] for cell in cells]
    seeds = [int(result["seed"]) for result in results]
    pooled = {policy: np.concatenate([cell["gains"][policy] for cell in cells]) for policy in POLICIES}
    classes = np.concatenate([cell["gains"]["class_index"] for cell in cells])
    unique_classes = np.unique(classes)

    policy_metrics = {}
    for policy in POLICIES:
        policy_metrics[policy] = {
            "mean_gain": float(np.mean(pooled[policy])),
            "per_seed_mean_gain": {
                str(seed): float(np.mean(cell["gains"][policy])) for seed, cell in zip(seeds, cells)
            },
            "metrics_from_cells": {
                key: float(np.mean([result["metrics"][policy][key] for result in results]))
                for key in (
                    "negative_transfer_rate",
                    "mean_utility_recovery",
                    "mean_selected_contexts",
                    "mean_distinct_clusters",
                )
            },
        }

    gate = {}
    for policy in GATE_POLICIES:
        delta_static = pooled[policy] - pooled["static_utility"]
        delta_cached = pooled[policy] - pooled["cached_mur"]
        per_class_gain = np.array([np.mean(pooled[policy][classes == value]) for value in unique_classes])
        per_class_static = np.array(
            [
                np.mean(pooled[policy][classes == value] - pooled["static_utility"][classes == value])
                for value in unique_classes
            ]
        )
        per_class_cached = np.array(
            [
                np.mean(pooled[policy][classes == value] - pooled["cached_mur"][classes == value])
                for value in unique_classes
            ]
        )
        gain_hier = hierarchical_ci(pooled[policy], classes, n_boot=bootstrap, seed=1)
        static_hier = hierarchical_ci(delta_static, classes, n_boot=bootstrap, seed=2)
        cached_hier = hierarchical_ci(delta_cached, classes, n_boot=bootstrap, seed=3)
        gate[policy] = {
            "gain_over_zero": paired_ci(pooled[policy]),
            "gain_over_static": paired_ci(delta_static),
            "gain_over_cached": paired_ci(delta_cached),
            "gain_over_zero_hierarchical": gain_hier,
            "gain_over_static_hierarchical": static_hier,
            "gain_over_cached_hierarchical": cached_hier,
            "gain_over_static_seed_level": seed_level_ci(
                [np.asarray(cell["gains"][policy]) - np.asarray(cell["gains"]["static_utility"]) for cell in cells],
                n_boot=bootstrap,
                seed=5,
            ),
            "gain_over_cached_seed_level": seed_level_ci(
                [np.asarray(cell["gains"][policy]) - np.asarray(cell["gains"]["cached_mur"]) for cell in cells],
                n_boot=bootstrap,
                seed=6,
            ),
            "episode_positive_fraction": float(np.mean(pooled[policy] > 0.0)),
            "class_fraction_gain_positive": float(np.mean(per_class_gain > 0.0)),
            "class_fraction_above_static": float(np.mean(per_class_static > 0.0)),
            "class_fraction_above_cached": float(np.mean(per_class_cached > 0.0)),
            "pass_gate_paired": bool(
                paired_ci(pooled[policy])["low"] > 0.0
                and paired_ci(delta_static)["low"] > 0.0
                and paired_ci(delta_cached)["low"] > 0.0
            ),
            "pass_gate_hierarchical": bool(
                gain_hier["low"] > 0.0 and static_hier["low"] > 0.0 and cached_hier["low"] > 0.0
            ),
        }

    headroom = {
        "anchor_only_mean_ce": float(np.mean([result["base_loss"] for result in results])),
        "oracle_greedy_gain": policy_metrics["oracle_greedy"]["mean_gain"],
        "oracle_static_gain": policy_metrics["oracle_static"]["mean_gain"],
        "pool_all_gain": policy_metrics["pool_all"]["mean_gain"],
        "relevance_gain": policy_metrics["relevance"]["mean_gain"],
        "random_gain": policy_metrics["random"]["mean_gain"],
    }
    headroom["h_state"] = float(
        (headroom["oracle_greedy_gain"] - headroom["oracle_static_gain"]) / headroom["oracle_greedy_gain"]
    ) if headroom["oracle_greedy_gain"] > 0 else float("nan")
    headroom["h_state_absolute"] = float(headroom["oracle_greedy_gain"] - headroom["oracle_static_gain"])

    first = np.concatenate([cell["gains"]["per_episode_first_marginal"] for cell in cells])
    second = np.concatenate([cell["gains"]["per_episode_second_marginal"] for cell in cells])
    redundancy = {
        "within_cluster_similarity": float(np.mean([result["redundancy"]["within_cluster_similarity"] for result in results])),
        "between_cluster_similarity": float(np.mean([result["redundancy"]["between_cluster_similarity"] for result in results])),
        "seed_member_similarity": float(np.mean([result["redundancy"]["seed_member_similarity"] for result in results])),
        "r110_reference_similarity": float(np.mean([result["redundancy"]["r110_reference_similarity"] for result in results])),
        "mean_first_member_marginal": float(np.mean(first)),
        "mean_second_member_marginal": float(np.mean(second)),
        "second_over_first_ratio": float(np.mean(second) / np.mean(first)) if np.mean(first) > 0 else float("nan"),
        "mean_distractor_marginal": float(np.mean([result["redundancy"]["mean_distractor_marginal"] for result in results])),
    }
    if redundancy_override is not None and name in redundancy_override:
        # Corrected diagnostics (member marginal measured with its own cluster's
        # seed selected only), produced by recompute_r112_redundancy.py.
        entry = redundancy_override[name]
        redundancy.update(
            {
                key: float(entry[key])
                for key in (
                    "within_cluster_similarity",
                    "between_cluster_similarity",
                    "seed_member_similarity",
                    "r110_reference_similarity",
                    "mean_first_member_marginal",
                    "mean_second_member_marginal",
                    "mean_distractor_marginal",
                )
                if key in entry
            }
        )
        redundancy["second_over_first_ratio"] = float(
            redundancy["mean_second_member_marginal"] / redundancy["mean_first_member_marginal"]
        ) if redundancy["mean_first_member_marginal"] > 0 else float("nan")
        redundancy["source"] = "recompute_r112_redundancy.py"
    else:
        redundancy["source"] = "result.json (driver diagnostics)"
    compute = {
        policy: float(np.mean([result["expert_rows"].get(policy, 0) for result in results]))
        for policy in POLICIES
        if any(policy in result["expert_rows"] for result in results)
    }
    compute["redundancy_rows"] = float(np.mean([result["redundancy_rows"] for result in results]))
    compute["evaluation_rows"] = float(np.mean([result["evaluation_rows"] for result in results]))
    return {
        "benchmark": name,
        "seeds": seeds,
        "cells": len(cells),
        "episodes": int(classes.size),
        "classes": int(unique_classes.size),
        "policy_metrics": policy_metrics,
        "gate": gate,
        "headroom": headroom,
        "redundancy": redundancy,
        "expert_rows_per_cell": compute,
        "predictor_training": {
            "final_train_ce": float(np.mean([result["predictor_training"]["final_train_ce"] for result in results])),
            "epochs": int(results[0]["predictor_training"]["epochs"]),
            "episodes": int(results[0]["predictor_training"]["episodes"]),
        },
        "wall_seconds_mean": float(np.mean([result["wall_seconds_total"] for result in results])),
        "per_class": [
            {
                "benchmark": name,
                "class_index": int(value),
                "episodes": int((classes == value).sum()),
                "oracle_greedy_gain": float(np.mean(pooled["oracle_greedy"][classes == value])),
                "oracle_static_gain": float(np.mean(pooled["oracle_static"][classes == value])),
                **{
                    f"{policy}_gain": float(np.mean(pooled[policy][classes == value]))
                    for policy in POLICIES
                },
                "delta_static": float(
                    np.mean(pooled["ranked_response_q4"][classes == value])
                    - np.mean(pooled["static_utility"][classes == value])
                ),
                "delta_cached": float(
                    np.mean(pooled["ranked_response_q4"][classes == value])
                    - np.mean(pooled["cached_mur"][classes == value])
                ),
            }
            for value in unique_classes
        ],
    }


def pooled_across_benchmarks(benchmarks: dict[str, dict], *, bootstrap: int) -> dict:
    """Equal-weight-per-benchmark mean of the paired q=4 deltas, three-level bootstrap.

    Raw episode pooling would be dominated by the benchmark with the largest CE
    scale, so the pooled statistic averages the per-benchmark means (each
    benchmark contributes equally) and the bootstrap resamples benchmarks, then
    classes inside a benchmark, then episodes inside a class.
    """

    names = list(benchmarks)
    per_benchmark = {
        name: {
            "gain": benchmarks[name]["gate"]["ranked_response_q4"]["gain_over_zero"]["mean"],
            "delta_static": benchmarks[name]["gate"]["ranked_response_q4"]["gain_over_static"]["mean"],
            "delta_cached": benchmarks[name]["gate"]["ranked_response_q4"]["gain_over_cached"]["mean"],
            "delta_static_forced": benchmarks[name]["gate"]["ranked_response_q4_forced"]["gain_over_static"]["mean"],
            "delta_cached_forced": benchmarks[name]["gate"]["ranked_response_q4_forced"]["gain_over_cached"]["mean"],
            "h_state": benchmarks[name]["headroom"]["h_state"],
        }
        for name in names
    }
    summary = {}
    for key in ("gain", "delta_static", "delta_cached", "delta_static_forced", "delta_cached_forced", "h_state"):
        values = np.asarray([per_benchmark[name][key] for name in names], dtype=np.float64)
        rng = np.random.default_rng(11)
        draws = values[rng.integers(0, values.size, size=(bootstrap, values.size))].mean(axis=1)
        summary[key] = {
            "mean": float(values.mean()),
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
            "per_benchmark": {name: per_benchmark[name][key] for name in names},
        }
    summary["benchmarks_positive_gain"] = int(sum(per_benchmark[name]["gain"] > 0 for name in names))
    summary["benchmarks_above_static"] = int(sum(per_benchmark[name]["delta_static"] > 0 for name in names))
    summary["benchmarks_above_cached"] = int(sum(per_benchmark[name]["delta_cached"] > 0 for name in names))
    summary["benchmarks_above_static_forced"] = int(
        sum(per_benchmark[name]["delta_static_forced"] > 0 for name in names)
    )
    summary["benchmarks_above_cached_forced"] = int(
        sum(per_benchmark[name]["delta_cached_forced"] > 0 for name in names)
    )
    summary["benchmarks_h_state_positive"] = int(sum(per_benchmark[name]["h_state"] > 0 for name in names))
    return summary


def render_markdown(variant: str, benchmarks: dict, pooled: dict) -> str:
    lines = [
        f"## Environment: {variant}",
        "",
        "### Gate at q=4 (paired CIs over all test episodes; hierarchical CIs cluster by class)",
        "",
        "| benchmark | rule | gain [95% CI] | Δ Static [CI] | Δ Cached [CI] | cls>Static | sel | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, aggregate in benchmarks.items():
        for policy, label in (
            ("ranked_response_q4", "thresholded"),
            ("ranked_response_q4_forced", "forced"),
        ):
            entry = aggregate["gate"][policy]
            metrics = aggregate["policy_metrics"][policy]["metrics_from_cells"]
            lines.append(
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
    lines += ["", "### Headroom and redundancy diagnostics", "",
              "| benchmark | anchor CE | oracle_greedy | oracle_static | H_state | within sim | between sim | ref sim | m(1st) | m(2nd) | 2nd/1st |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, aggregate in benchmarks.items():
        head = aggregate["headroom"]
        red = aggregate["redundancy"]
        lines.append(
            "| {b} | {ce:.4f} | {og:.4f} | {os_:.4f} | {h:.5f} | {w:.3f} | {bt:.3f} | {rf:.3f} | "
            "{m1:.4f} | {m2:.4f} | {r:.3f} |".format(
                b=name,
                ce=head["anchor_only_mean_ce"],
                og=head["oracle_greedy_gain"],
                os_=head["oracle_static_gain"],
                h=head["h_state"],
                w=red["within_cluster_similarity"],
                bt=red["between_cluster_similarity"],
                rf=red["r110_reference_similarity"],
                m1=red["mean_first_member_marginal"],
                m2=red["mean_second_member_marginal"],
                r=red["second_over_first_ratio"],
            )
        )
    lines += ["", "### Pooled across benchmarks (equal weight per benchmark)", "",
              "| statistic | mean | 95% CI | benchmarks positive |", "|---|---|---|---|"]
    for key, label in (
        ("gain", "R-MUR q=4 gain (thresholded)"),
        ("delta_static", "R-MUR q=4 − Static"),
        ("delta_cached", "R-MUR q=4 − Cached"),
        ("delta_static_forced", "R-MUR q=4 forced − Static"),
        ("delta_cached_forced", "R-MUR q=4 forced − Cached"),
        ("h_state", "H_state"),
    ):
        entry = pooled[key]
        positive = {
            "gain": pooled["benchmarks_positive_gain"],
            "delta_static": pooled["benchmarks_above_static"],
            "delta_cached": pooled["benchmarks_above_cached"],
            "delta_static_forced": pooled["benchmarks_above_static_forced"],
            "delta_cached_forced": pooled["benchmarks_above_cached_forced"],
            "h_state": pooled["benchmarks_h_state_positive"],
        }[key]
        lines.append(
            f"| {label} | {entry['mean']:.4f} | [{entry['low']:.4f}, {entry['high']:.4f}] | {positive}/5 |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True, help="natural near-duplicate pools")
    parser.add_argument("--copies-root", type=Path, default=None, help="controlled perturbed-copy pools")
    parser.add_argument("--out-root", type=Path, default=PAMI / "results/derived/R112_redundant_demo_summary")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--redundancy-natural",
        type=Path,
        default=PAMI / "results/derived/R112_redundant_demo_summary/redundancy_natural.json",
    )
    parser.add_argument(
        "--redundancy-copies",
        type=Path,
        default=PAMI / "results/derived/R112_redundant_demo_summary/redundancy_copies.json",
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "derived_from": [str(args.run_root)] + ([str(args.copies_root)] if args.copies_root else []),
        "run_ids": [args.run_root.name] + ([args.copies_root.name] if args.copies_root else []),
        "bootstrap_replicates": args.bootstrap,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary: dict = {"provenance": provenance, "environments": {}}
    markdown = [
        f"<!-- generated from {', '.join(provenance['derived_from'])} at {provenance['generated_utc']} -->",
        "",
    ]
    redundancy_files = {
        "natural_near_duplicates": args.redundancy_natural,
        "controlled_near_copies": args.redundancy_copies,
    }
    for label, root in (("natural_near_duplicates", args.run_root), ("controlled_near_copies", args.copies_root)):
        if root is None:
            continue
        override_path = redundancy_files.get(label)
        override = None
        if override_path is not None and Path(override_path).exists():
            override = json.loads(Path(override_path).read_text(encoding="utf-8")).get("cells", {})
        cells = load_cells(root)
        if not cells:
            print(f"no complete cells under {root}", flush=True)
            continue
        benchmarks = {
            name: aggregate_benchmark(
                name, records, bootstrap=args.bootstrap, redundancy_override=override
            )
            for name, records in cells.items()
        }
        pooled = pooled_across_benchmarks(benchmarks, bootstrap=args.bootstrap)
        summary["environments"][label] = {"run_root": str(root), "benchmarks": benchmarks, "pooled": pooled}
        markdown.append(render_markdown(label, benchmarks, pooled))
        print(
            f"[{label}] q4={pooled['gain']['mean']:+.4f} dS={pooled['delta_static']['mean']:+.4f} "
            f"dS_forced={pooled['delta_static_forced']['mean']:+.4f} H_state={pooled['h_state']['mean']:+.5f} "
            f"(positive {pooled['benchmarks_h_state_positive']}/5)",
            flush=True,
        )
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_root / "report_tables.md").write_text("\n".join(markdown), encoding="utf-8")
    (args.out_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    gate_rows = [
        "environment,benchmark,policy,gain,paired_lo,paired_hi,hier_lo,hier_hi,delta_static,ds_lo,ds_hi,"
        "delta_cached,dc_lo,dc_hi,class_frac_above_static,class_frac_above_cached,mean_selected,"
        "mean_distinct_clusters,pass_paired,pass_hier"
    ]
    for label, environment in summary["environments"].items():
        for name, aggregate in environment["benchmarks"].items():
            for policy in GATE_POLICIES:
                entry = aggregate["gate"][policy]
                metrics = aggregate["policy_metrics"][policy]["metrics_from_cells"]
                gate_rows.append(
                    ",".join(
                        [
                            label,
                            name,
                            policy,
                            f"{entry['gain_over_zero']['mean']:.6f}",
                            f"{entry['gain_over_zero']['low']:.6f}",
                            f"{entry['gain_over_zero']['high']:.6f}",
                            f"{entry['gain_over_zero_hierarchical']['low']:.6f}",
                            f"{entry['gain_over_zero_hierarchical']['high']:.6f}",
                            f"{entry['gain_over_static']['mean']:.6f}",
                            f"{entry['gain_over_static']['low']:.6f}",
                            f"{entry['gain_over_static']['high']:.6f}",
                            f"{entry['gain_over_cached']['mean']:.6f}",
                            f"{entry['gain_over_cached']['low']:.6f}",
                            f"{entry['gain_over_cached']['high']:.6f}",
                            f"{entry['class_fraction_above_static']:.4f}",
                            f"{entry['class_fraction_above_cached']:.4f}",
                            f"{metrics['mean_selected_contexts']:.3f}",
                            f"{metrics['mean_distinct_clusters']:.3f}",
                            str(entry["pass_gate_paired"]),
                            str(entry["pass_gate_hierarchical"]),
                        ]
                    )
                )
    (args.out_root / "gate_table.csv").write_text("\n".join(gate_rows) + "\n", encoding="utf-8")
    print(f"wrote derived summary to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
