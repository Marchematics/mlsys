#!/usr/bin/env python3
"""R110 derived summary: gates, per-class consistency, hierarchical CIs.

Reads an immutable raw run directory produced by ``run_r110_demo_selection.py``
and writes derived tables under ``results/derived/R110_demo_selection_summary``.
Every derived file records the source run directory and run id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
ROOT = PAMI.parent
KBS = ROOT / "KBS_MUR"
for _path in (HERE, PAMI / "experiments", KBS / "src", KBS / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_traffic_response_router_sweep import paired_ci  # noqa: E402

POLICIES = [
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
    "ranked_response_q2",
    "ranked_response_q4",
    "ranked_response_q8",
    "ranked_response_full",
]
GAIN_POLICIES = [name for name in POLICIES if name != "base_only"]


def load_cells(run_root: Path) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    for benchmark_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        records = []
        for cell_dir in sorted(benchmark_dir.glob("seed*")):
            result_path = cell_dir / "result.json"
            gains_path = cell_dir / "episode_gains.npz"
            if not result_path.exists() or not gains_path.exists():
                print(f"  (skipping incomplete cell {cell_dir})", flush=True)
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            with np.load(gains_path) as payload:
                gains = {key: np.asarray(payload[key]) for key in payload.files}
            records.append({"result": result, "gains": gains, "dir": str(cell_dir)})
        if records:
            cells[benchmark_dir.name] = records
    return cells


def hierarchical_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Two-level bootstrap: resample classes, then episodes inside each class."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for step in range(n_boot):
        picked = rng.integers(0, unique.size, size=unique.size)
        pieces = [
            index_by_group[group][rng.integers(0, index_by_group[group].size, size=index_by_group[group].size)]
            for group in picked
        ]
        draws[step] = values[np.concatenate(pieces)].mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(draws, alpha)),
        "high": float(np.quantile(draws, 1.0 - alpha)),
        "groups": int(unique.size),
        "n_boot": int(n_boot),
    }


def seed_level_ci(values_by_seed: list[np.ndarray], *, n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap over the (few) protocol seeds of their episode means."""

    means = np.asarray([float(np.mean(values)) for values in values_by_seed], dtype=np.float64)
    if means.size < 2:
        return {
            "mean": float(means.mean()) if means.size else float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "seeds": means.tolist(),
        }
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, means.size, size=(n_boot, means.size))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "seeds": means.tolist(),
    }


def aggregate_benchmark(name: str, cells: list[dict], *, bootstrap: int) -> dict:
    results = [cell["result"] for cell in cells]
    seeds = [int(result["seed"]) for result in results]
    pooled = {
        policy: np.concatenate([cell["gains"][policy] for cell in cells]) for policy in POLICIES
    }
    classes = np.concatenate([cell["gains"]["class_index"] for cell in cells])
    seed_ids = np.concatenate(
        [np.full(cell["gains"]["class_index"].size, int(cell["result"]["seed"])) for cell in cells]
    )
    unique_classes = np.unique(classes)

    policy_metrics = {}
    for policy in POLICIES:
        per_seed = [float(np.mean(cell["gains"][policy])) for cell in cells]
        policy_metrics[policy] = {
            "mean_gain": float(np.mean(pooled[policy])),
            "per_seed_mean_gain": {str(seed): value for seed, value in zip(seeds, per_seed)},
            "metrics_from_cells": {
                key: float(np.mean([result["metrics"][policy][key] for result in results]))
                for key in (
                    "negative_transfer_rate",
                    "mean_utility_recovery",
                    "mean_selected_contexts",
                )
            },
        }
    gate = {}
    for policy in POLICIES:
        if not policy.startswith("ranked_response"):
            continue
        delta_static = pooled[policy] - pooled["static_utility"]
        delta_cached = pooled[policy] - pooled["cached_mur"]
        per_class_gain = np.array(
            [np.mean(pooled[policy][classes == value]) for value in unique_classes]
        )
        per_class_static = np.array(
            [np.mean(pooled[policy][classes == value] - pooled["static_utility"][classes == value]) for value in unique_classes]
        )
        per_class_cached = np.array(
            [np.mean(pooled[policy][classes == value] - pooled["cached_mur"][classes == value]) for value in unique_classes]
        )
        seed_means = [np.asarray(cell["gains"][policy]) for cell in cells]
        gate[policy] = {
            "gain_over_zero": paired_ci(pooled[policy]),
            "gain_over_static": paired_ci(delta_static),
            "gain_over_cached": paired_ci(delta_cached),
            "gain_over_zero_hierarchical": hierarchical_ci(
                pooled[policy], classes, n_boot=bootstrap, seed=1
            ),
            "gain_over_static_hierarchical": hierarchical_ci(
                delta_static, classes, n_boot=bootstrap, seed=2
            ),
            "gain_over_cached_hierarchical": hierarchical_ci(
                delta_cached, classes, n_boot=bootstrap, seed=3
            ),
            "gain_over_zero_seed_level": seed_level_ci(seed_means, n_boot=bootstrap, seed=4),
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
                hierarchical_ci(pooled[policy], classes, n_boot=bootstrap, seed=1)["low"] > 0.0
                and hierarchical_ci(delta_static, classes, n_boot=bootstrap, seed=2)["low"] > 0.0
                and hierarchical_ci(delta_cached, classes, n_boot=bootstrap, seed=3)["low"] > 0.0
            ),
        }
    per_class_rows = []
    for value in unique_classes:
        selected = classes == value
        row = {
            "benchmark": name,
            "class_index": int(value),
            "episodes": int(selected.sum()),
            "oracle_greedy_gain": float(np.mean(pooled["oracle_greedy"][selected])),
            "oracle_static_gain": float(np.mean(pooled["oracle_static"][selected])),
        }
        for policy in POLICIES:
            row[f"{policy}_gain"] = float(np.mean(pooled[policy][selected]))
        row["delta_static"] = row["ranked_response_q4_gain"] - row["static_utility_gain"]
        row["delta_cached"] = row["ranked_response_q4_gain"] - row["cached_mur_gain"]
        per_class_rows.append(row)
    quality_keys = sorted(results[0]["predictor_quality"].keys())
    predictor_quality = {
        key: {
            "mean_accuracy": float(np.mean([result["predictor_quality"][key]["mean_accuracy"] for result in results])),
            "std_accuracy": float(np.std([result["predictor_quality"][key]["mean_accuracy"] for result in results])),
            "mean_ce": float(np.mean([result["predictor_quality"][key]["mean_ce"] for result in results])),
            "mean_demos": float(np.mean([result["predictor_quality"][key]["mean_demos"] for result in results])),
        }
        for key in quality_keys
    }
    headroom = {
        "anchor_only_mean_ce": float(np.mean([result["base_loss"] for result in results])),
        "oracle_greedy_gain": policy_metrics["oracle_greedy"]["mean_gain"],
        "oracle_static_gain": policy_metrics["oracle_static"]["mean_gain"],
        "pool_all_gain": policy_metrics["pool_all"]["mean_gain"],
        "relevance_gain": policy_metrics["relevance"]["mean_gain"],
        "random_gain": policy_metrics["random"]["mean_gain"],
        "h_state": float(
            (policy_metrics["oracle_greedy"]["mean_gain"] - policy_metrics["oracle_static"]["mean_gain"])
            / policy_metrics["oracle_greedy"]["mean_gain"]
        )
        if policy_metrics["oracle_greedy"]["mean_gain"] > 0
        else float("nan"),
        "oracle_greedy_per_seed": policy_metrics["oracle_greedy"]["per_seed_mean_gain"],
        "oracle_static_per_seed": policy_metrics["oracle_static"]["per_seed_mean_gain"],
    }
    # A benchmark is only informative for context selection if the frozen
    # predictor leaves headroom: at least 0.01 nats of base CE and at least
    # 0.005 nats of achievable (oracle-greedy) reduction.
    informative = bool(
        headroom["anchor_only_mean_ce"] >= 0.01 and headroom["oracle_greedy_gain"] >= 0.005
    )
    compute = {
        policy: float(np.mean([result["expert_rows"].get(policy, 0) for result in results]))
        for policy in POLICIES
        if any(policy in result["expert_rows"] for result in results)
    }
    compute["evaluation_rows"] = float(np.mean([result["evaluation_rows"] for result in results]))
    return {
        "benchmark": name,
        "seeds": seeds,
        "cells": len(cells),
        "episodes": int(classes.size),
        "classes": int(unique_classes.size),
        "policy_metrics": policy_metrics,
        "gate": gate,
        "predictor_quality": predictor_quality,
        "headroom": headroom,
        "informative": informative,
        "saturation_note": (
            "predictor already solves the query batch: base CE {:.5f} nats, oracle reduction {:.2e} nats; "
            "the selection gate is numerically vacuous here".format(
                headroom["anchor_only_mean_ce"], headroom["oracle_greedy_gain"]
            )
            if not informative
            else "headroom present"
        ),
        "expert_rows_per_cell": compute,
        "predictor_training": {
            "final_train_ce": float(np.mean([result["predictor_training"]["final_train_ce"] for result in results])),
            "first_train_ce": float(np.mean([result["predictor_training"]["first_train_ce"] for result in results])),
            "epochs": int(results[0]["predictor_training"]["epochs"]),
            "episodes": int(results[0]["predictor_training"]["episodes"]),
        },
        "wall_seconds_mean": float(np.mean([result["wall_seconds_total"] for result in results])),
        "per_class": per_class_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=PAMI / "results/derived/R110_demo_selection_summary")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    cells = load_cells(args.run_root)
    if not cells:
        raise SystemExit(f"no complete cells under {args.run_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "derived_from": str(args.run_root),
        "run_id": args.run_root.name,
        "bootstrap_replicates": args.bootstrap,
        "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    }
    summary = {"provenance": provenance, "benchmarks": {}}
    gate_rows = [
        "benchmark,policy,gain,paired_lo,paired_hi,hier_lo,hier_hi,delta_static,ds_lo,ds_hi,"
        "ds_hier_lo,ds_hier_hi,delta_cached,dc_lo,dc_hi,dc_hier_lo,dc_hier_hi,"
        "class_frac_positive,class_frac_above_static,class_frac_above_cached,pass_paired,pass_hier"
    ]
    per_class_rows = [
        "benchmark,class_index,episodes,oracle_greedy_gain,oracle_static_gain,static_utility_gain,"
        "cached_mur_gain,ranked_response_q4_gain,relevance_gain,random_gain,pool_all_gain,delta_static,delta_cached"
    ]
    for name, records in cells.items():
        aggregate = aggregate_benchmark(name, records, bootstrap=args.bootstrap)
        summary["benchmarks"][name] = aggregate
        for policy, entry in aggregate["gate"].items():
            gate_rows.append(
                ",".join(
                    [
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
                        f"{entry['gain_over_static_hierarchical']['low']:.6f}",
                        f"{entry['gain_over_static_hierarchical']['high']:.6f}",
                        f"{entry['gain_over_cached']['mean']:.6f}",
                        f"{entry['gain_over_cached']['low']:.6f}",
                        f"{entry['gain_over_cached']['high']:.6f}",
                        f"{entry['gain_over_cached_hierarchical']['low']:.6f}",
                        f"{entry['gain_over_cached_hierarchical']['high']:.6f}",
                        f"{entry['class_fraction_gain_positive']:.4f}",
                        f"{entry['class_fraction_above_static']:.4f}",
                        f"{entry['class_fraction_above_cached']:.4f}",
                        str(entry["pass_gate_paired"]),
                        str(entry["pass_gate_hierarchical"]),
                    ]
                )
            )
        for row in aggregate["per_class"]:
            per_class_rows.append(
                ",".join(
                    [
                        name,
                        str(row["class_index"]),
                        str(row["episodes"]),
                        f"{row['oracle_greedy_gain']:.6f}",
                        f"{row['oracle_static_gain']:.6f}",
                        f"{row['static_utility_gain']:.6f}",
                        f"{row['cached_mur_gain']:.6f}",
                        f"{row['ranked_response_q4_gain']:.6f}",
                        f"{row['relevance_gain']:.6f}",
                        f"{row['random_gain']:.6f}",
                        f"{row['pool_all_gain']:.6f}",
                        f"{row['delta_static']:.6f}",
                        f"{row['delta_cached']:.6f}",
                    ]
                )
            )
        print(
            f"[{name}] q4={aggregate['gate']['ranked_response_q4']['gain_over_zero']['mean']:.4f} "
            f"pass_paired={aggregate['gate']['ranked_response_q4']['pass_gate_paired']} "
            f"pass_hier={aggregate['gate']['ranked_response_q4']['pass_gate_hierarchical']}",
            flush=True,
        )
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_root / "gate_table.csv").write_text("\n".join(gate_rows) + "\n", encoding="utf-8")
    (args.out_root / "per_class.csv").write_text("\n".join(per_class_rows) + "\n", encoding="utf-8")
    (args.out_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (args.out_root / "report_tables.md").write_text(render_markdown(summary, provenance), encoding="utf-8")
    print(f"wrote derived summary to {args.out_root}", flush=True)


def render_markdown(summary: dict, provenance: dict) -> str:
    """Markdown tables generated directly from the derived numbers."""

    lines = [
        f"<!-- generated from {provenance['derived_from']} at {provenance['generated_utc']} -->",
        "",
        "### Gate (R-MUR q=4 vs anchor-only, Static Utility, Cached Utility)",
        "",
        "| benchmark | informative | gain | 95% CI (paired) | hier. 95% CI | d(Static) | CI | d(Cached) | CI | cls>0 | cls>Static | cls>Cached | pass |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, aggregate in summary["benchmarks"].items():
        entry = aggregate["gate"]["ranked_response_q4"]
        lines.append(
            "| {b} | {inf} | {g:.4f} | [{gl:.4f}, {gh:.4f}] | [{hl:.4f}, {hh:.4f}] | {ds:.4f} | [{dsl:.4f}, {dsh:.4f}] | "
            "{dc:.4f} | [{dcl:.4f}, {dch:.4f}] | {fp:.2f} | {fs:.2f} | {fc:.2f} | {p} |".format(
                b=name,
                inf="yes" if aggregate["informative"] else "NO (saturated)",
                g=entry["gain_over_zero"]["mean"],
                gl=entry["gain_over_zero"]["low"],
                gh=entry["gain_over_zero"]["high"],
                hl=entry["gain_over_zero_hierarchical"]["low"],
                hh=entry["gain_over_zero_hierarchical"]["high"],
                ds=entry["gain_over_static"]["mean"],
                dsl=entry["gain_over_static"]["low"],
                dsh=entry["gain_over_static"]["high"],
                dc=entry["gain_over_cached"]["mean"],
                dcl=entry["gain_over_cached"]["low"],
                dch=entry["gain_over_cached"]["high"],
                fp=entry["class_fraction_gain_positive"],
                fs=entry["class_fraction_above_static"],
                fc=entry["class_fraction_above_cached"],
                p="yes" if entry["pass_gate_paired"] else "no",
            )
        )
    lines += ["", "### Predictor quality (frozen in-context classifier, test episodes)", "",
              "| benchmark | demos | accuracy | CE |", "|---|---|---|---|"]
    for name, aggregate in summary["benchmarks"].items():
        for key, value in aggregate["predictor_quality"].items():
            lines.append(
                f"| {name} | {key} | {value['mean_accuracy']:.4f} | {value['mean_ce']:.4f} |"
            )
    lines += ["", "### Headroom and baselines (mean query-batch CE reduction)", "",
              "| benchmark | anchor-only CE | oracle_greedy | oracle_static | pool_all | relevance | random | H_state |",
              "|---|---|---|---|---|---|---|---|"]
    for name, aggregate in summary["benchmarks"].items():
        head = aggregate["headroom"]
        lines.append(
            "| {b} | {ce:.4f} | {og:.4f} | {os_:.4f} | {pa:.4f} | {rel:.4f} | {rnd:.4f} | {h:.3f} | {inf} |".format(
                b=name,
                ce=head["anchor_only_mean_ce"],
                og=head["oracle_greedy_gain"],
                os_=head["oracle_static_gain"],
                pa=head["pool_all_gain"],
                rel=head["relevance_gain"],
                rnd=head["random_gain"],
                h=head["h_state"],
                inf="yes" if aggregate["informative"] else "NO (saturated)",
            )
        )
    lines += ["", "### All policies (mean gain per benchmark)", "",
              "| benchmark | " + " | ".join(POLICIES) + " |",
              "|---" * (len(POLICIES) + 1) + "|"]
    for name, aggregate in summary["benchmarks"].items():
        row = " | ".join(
            f"{aggregate['policy_metrics'][policy]['mean_gain']:.4f}" for policy in POLICIES
        )
        lines.append(f"| {name} | {row} |")
    compute_keys = list(
        summary["benchmarks"][next(iter(summary["benchmarks"]))]["expert_rows_per_cell"].keys()
    )
    lines += ["", "### Compute (mean predictor rows per cell)", "",
              "| benchmark | " + " | ".join(compute_keys) + " |",
              "|---" * (len(compute_keys) + 1) + "|"]
    for name, aggregate in summary["benchmarks"].items():
        row = " | ".join(
            f"{aggregate['expert_rows_per_cell'][key]:.0f}" for key in compute_keys
        )
        lines.append(f"| {name} | {row} |")
    lines += ["", "### Per-seed gate (R-MUR q=4 gain)", "", "| benchmark | seed | gain |", "|---|---|---|"]
    for name, aggregate in summary["benchmarks"].items():
        per_seed = aggregate["gate"]["ranked_response_q4"]["gain_over_zero_seed_level"]["seeds"]
        for seed, value in zip([str(item) for item in aggregate["seeds"]], per_seed):
            lines.append(f"| {name} | {seed} | {value:.4f} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
