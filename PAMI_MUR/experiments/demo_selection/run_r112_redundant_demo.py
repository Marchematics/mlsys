#!/usr/bin/env python3
"""R112: R-MUR on a redundancy-structured demonstration pool.

Everything is inherited from R110 (same frozen encoder features, same episode
counts, same in-context predictor family and training recipe, same frozen R-MUR
code path and q values).  Two things differ:

1. candidate pools are redundancy-structured (3 clusters x 4 near-duplicate
   in-class demonstrations + 4 distractors), so a candidate's value depends on
   what is already selected;
2. one ablation is added: R-MUR with a forced budget (``*_forced``), which
   removes the frozen "> 0" acceptance rule so ranking quality is separated from
   the stopping rule.

New file; no R110 artifact is modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
ROOT = PAMI.parent
KBS = ROOT / "KBS_MUR"
for _path in (HERE, PAMI / "experiments", KBS / "src", KBS / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mur.synthetic_experiment import (  # noqa: E402
    sample_pair_dataset,
    select_mur,
    select_static,
    train_utility_model,
)
from run_traffic_full_router import select_mmr, select_relevance  # noqa: E402
from run_traffic_ranked_response_router_sweep import train_ranked_response_model  # noqa: E402
from run_traffic_response_router_sweep import paired_ci  # noqa: E402
from frozen_helpers import (  # noqa: E402
    build_full_state_data,
    candidate_similarity,
    sample_pair_dataset_fast,
    select_dpp,
    select_facility_location,
    select_oracle_greedy,
    select_oracle_static,
    select_random,
    select_ranked_response_mur,
)

from demo_protocol import BENCHMARK_CONFIGS, load_protocol  # noqa: E402
from forced_budget import select_ranked_response_mur_forced  # noqa: E402
from incontext_predictor import (  # noqa: E402
    DemoExpertOps,
    cross_entropy_from_logits,
    save_checkpoint,
)
from redundant_pool import build_redundant_batch  # noqa: E402
from train_predictor import train_incontext_predictor  # noqa: E402

# R110 helpers reused verbatim (module-level functions of the R110 driver).
from run_r110_demo_selection import (  # noqa: E402
    _git_head,
    _source_hashes,
    _write_manifest,
    parse_ints,
    predictor_quality,
    relevance_of,
)

FORCED_SUFFIX = "_forced"


def _r112_hashes() -> dict:
    """md5 of the R112-specific modules (R110 hashes come from the R110 driver)."""

    import hashlib

    files = {
        "PAMI_MUR/experiments/demo_selection/redundant_pool.py": HERE / "redundant_pool.py",
        "PAMI_MUR/experiments/demo_selection/forced_budget.py": HERE / "forced_budget.py",
        "PAMI_MUR/experiments/demo_selection/run_r112_redundant_demo.py": HERE / "run_r112_redundant_demo.py",
        "PAMI_MUR/experiments/demo_selection/incontext_predictor.py": HERE / "incontext_predictor.py",
        "PAMI_MUR/experiments/demo_selection/demo_protocol.py": HERE / "demo_protocol.py",
        "PAMI_MUR/experiments/demo_selection/train_predictor.py": HERE / "train_predictor.py",
    }
    return {label: hashlib.md5(path.read_bytes()).hexdigest() for label, path in files.items()}


def redundancy_diagnostics(ops: DemoExpertOps, batch, info) -> dict:
    """Utility-level measurement of the injected redundancy.

    For every episode we report the true marginal utility of
    * the best standalone candidate of each cluster (state = empty set),
    * a second member of the same cluster (state = the cluster seed), and
    * distractors (state = empty set),
    so diminishing returns inside a cluster are visible in utility, not only in
    embedding similarity.
    """

    episodes = batch.episodes
    empty = np.zeros((episodes, batch.candidate_count), dtype=bool)
    marginal_empty = ops.candidate_marginals(batch, empty)
    seeds = info.seed_index
    rows = np.arange(episodes)
    first = np.full(episodes, np.nan)
    for episode in range(episodes):
        first[episode] = float(np.mean(marginal_empty[episode, seeds[episode]]))
    # Second-member marginals are measured with *that cluster's own seed*
    # selected, one state per cluster, so the numbers describe within-cluster
    # redundancy rather than the value of a near-duplicate after the whole
    # budget is already spent.
    second_values: list[float] = []
    per_cluster_second = np.full((episodes, seeds.shape[1]), np.nan)
    for cluster_index in range(seeds.shape[1]):
        selected = np.zeros((episodes, batch.candidate_count), dtype=bool)
        selected[rows, seeds[:, cluster_index]] = True
        marginal_after = ops.candidate_marginals(batch, selected)
        for episode in range(episodes):
            members = np.flatnonzero(info.cluster_id[episode] == cluster_index)
            members = members[members != seeds[episode, cluster_index]]
            if members.size:
                value = float(np.mean(marginal_after[episode, members]))
                per_cluster_second[episode, cluster_index] = value
                second_values.append(value)
    second = np.asarray(second_values) if second_values else np.asarray([np.nan])
    return {
        "mean_first_member_marginal": float(np.nanmean(first)),
        "mean_second_member_marginal": float(np.nanmean(second)),
        "mean_second_member_marginal_given_seed": float(np.nanmean(second)),
        "mean_distractor_marginal": float(
            np.nanmean(np.where(info.cluster_id < 0, marginal_empty, np.nan))
        ),
        "within_cluster_similarity": float(np.mean(info.within_similarity)),
        "seed_member_similarity": float(np.mean(info.seed_member_similarity)),
        "between_cluster_similarity": float(np.mean(info.between_similarity)),
        "r110_reference_similarity": float(np.mean(info.r110_reference_similarity)),
        "per_episode_first": first.tolist(),
        "per_episode_second": np.nanmean(per_cluster_second, axis=1).tolist(),
    }


def evaluate_cell(
    seed: int,
    protocol,
    train_batch,
    test_batch,
    test_info,
    ops: DemoExpertOps,
    *,
    q_values: list[int],
    epochs: int,
    beta: float,
    device: str,
    policy_seed: int,
    states_per_episode: int = 4,
    pool_rule: str = "",
) -> tuple[dict, dict]:
    """One (benchmark, seed) cell on the redundancy-structured pool."""

    config = protocol.config
    budget = config.budget
    started = time.time()

    static_pairs = sample_pair_dataset_fast(
        train_batch, ops, max_budget=budget, states_per_episode=states_per_episode, seed=seed + 4_000, static_only=True
    )
    cached_pairs = sample_pair_dataset_fast(
        train_batch, ops, max_budget=budget, states_per_episode=states_per_episode, seed=seed + 5_000, static_only=False
    )
    static_model = train_utility_model(
        static_pairs, max_budget=budget, seed=seed + 6_000, device=device, epochs=epochs
    )
    cached_model = train_utility_model(
        cached_pairs, max_budget=budget, seed=seed + 7_000, device=device, epochs=epochs
    )
    full_data = build_full_state_data(
        train_batch, ops, max_budget=budget, states_per_episode=states_per_episode, seed=seed + 9_000
    )
    ranked_model = train_ranked_response_model(
        full_data, max_budget=budget, seed=seed + 10_000, device=device, epochs=epochs, beta=beta
    )

    relevance = relevance_of(test_batch)
    similarity = candidate_similarity(test_batch)
    base_loss = ops.loss(test_batch, np.zeros((test_batch.episodes, test_batch.candidate_count), dtype=bool))
    quality = predictor_quality(ops, test_batch, relevance, budget=budget, seed=policy_seed)

    selection_rows: dict[str, int] = {
        name: 0 for name in ("static_utility", "cached_mur", "oracle_static", "oracle_greedy")
    }

    def timed(name, function, *positional, **keywords):
        before = ops.rows
        output = function(*positional, **keywords)
        selection_rows[name] = selection_rows.get(name, 0) + (ops.rows - before)
        return output

    oracle_static = timed("oracle_static", select_oracle_static, test_batch, ops, budget)
    oracle_greedy = timed("oracle_greedy", select_oracle_greedy, test_batch, ops, budget)
    selections: dict[str, np.ndarray] = {
        "base_only": np.zeros((test_batch.episodes, test_batch.candidate_count), dtype=bool),
        "pool_all": np.ones((test_batch.episodes, test_batch.candidate_count), dtype=bool),
        "random": select_random(test_batch, budget, policy_seed),
        "relevance": select_relevance(relevance, budget),
        "mmr": select_mmr(test_batch, relevance, budget),
        "facility_location": select_facility_location(test_batch, relevance, budget, similarity=similarity),
        "dpp": select_dpp(test_batch, relevance, budget, similarity=similarity),
        "static_utility": timed(
            "static_utility", select_static, static_model, test_batch, budget=budget, device=device, pack_fn=ops.pack
        ),
        "cached_mur": timed(
            "cached_mur", select_mur, cached_model, test_batch, budget=budget, device=device, pack_fn=ops.pack
        ),
        "oracle_static": oracle_static,
        "oracle_greedy": oracle_greedy,
    }
    for q in q_values:
        stem = "ranked_response_full" if q >= test_batch.candidate_count else f"ranked_response_q{q}"
        before = ops.rows
        selections[stem], _ = select_ranked_response_mur(
            cached_model, ranked_model, test_batch, ops, budget=budget, q=q, device=device
        )
        selection_rows[stem] = ops.rows - before
        before = ops.rows
        selections[stem + FORCED_SUFFIX], _ = select_ranked_response_mur_forced(
            cached_model, ranked_model, test_batch, ops, budget=budget, q=q, device=device
        )
        selection_rows[stem + FORCED_SUFFIX] = ops.rows - before

    redundancy_started = ops.rows
    redundancy = redundancy_diagnostics(ops, test_batch, test_info)
    redundancy_rows = ops.rows - redundancy_started

    evaluation_rows_before = ops.rows
    losses = {name: ops.loss(test_batch, mask) for name, mask in selections.items()}
    evaluation_rows = ops.rows - evaluation_rows_before
    gains = {name: base_loss - value for name, value in losses.items()}
    oracle_gain = gains["oracle_greedy"]
    positive = oracle_gain > 0.0
    metrics = {}
    in_class = test_info.cluster_id >= 0
    for name, mask in selections.items():
        gain = gains[name]
        distinct = []
        for episode in range(test_batch.episodes):
            chosen = mask[episode] & in_class[episode]
            distinct.append(len(np.unique(test_info.cluster_id[episode][chosen])) if np.any(chosen) else 0.0)
        metrics[name] = {
            "mean_distinct_clusters": float(np.mean(distinct)),
            "mean_prediction_gain": float(np.mean(gain)),
            "mean_loss": float(np.mean(losses[name])),
            "negative_transfer_rate": float(np.mean(gain < 0.0)),
            "mean_utility_recovery": float(np.mean(gain[positive] / oracle_gain[positive]))
            if np.any(positive)
            else float("nan"),
            "mean_selected_contexts": float(np.mean(np.sum(mask, axis=1))),
        }
    headroom = metrics["oracle_greedy"]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]
    result = {
        "run_id": "R112_redundant_demo",
        "benchmark": protocol.features.benchmark,
        "seed": seed,
        "config": {
            "seed": seed,
            "candidate_count": int(test_batch.candidate_count),
            "budget": budget,
            "query_batch_size": int(test_batch.query_batch_size),
            "episodes": int(test_batch.episodes),
            "classes": int(protocol.features.num_classes),
            "response_dim": int(full_data["response"].shape[-1]),
            "q_values": q_values,
            "router_epochs": epochs,
            "ranking_beta": beta,
            "states_per_episode": states_per_episode,
            "pair_sampler": "fast",
            "pool_rule": pool_rule,
            "device": device,
        },
        "base_loss": float(np.mean(base_loss)),
        "predictor_quality": quality,
        "redundancy": {key: value for key, value in redundancy.items() if not key.startswith("per_episode")},
        "metrics": metrics,
        "expert_rows": selection_rows,
        "redundancy_rows": redundancy_rows,
        "evaluation_rows": evaluation_rows,
        "gate": {
            name: {
                "gain_over_zero": paired_ci(gains[name]),
                "gain_over_static": paired_ci(gains[name] - gains["static_utility"]),
                "gain_over_cached": paired_ci(gains[name] - gains["cached_mur"]),
                "headroom_recovery": float(
                    (metrics[name]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]) / headroom
                )
                if headroom > 0.0
                else float("nan"),
            }
            for name in selections
            if name.startswith("ranked_response")
        },
        "headroom": {
            "oracle_greedy_gain": metrics["oracle_greedy"]["mean_prediction_gain"],
            "oracle_static_gain": metrics["oracle_static"]["mean_prediction_gain"],
            "h_state": float(headroom / oracle_gain.mean()) if float(np.mean(oracle_gain)) > 0 else float("nan"),
            "base_mean_ce": float(np.mean(base_loss)),
            "pool_all_gain": metrics["pool_all"]["mean_prediction_gain"],
        },
        "wall_seconds": time.time() - started,
    }
    gains_out = {
        "base_loss": base_loss.astype(np.float32),
        "class_index": test_batch.class_index.astype(np.int64),
        "episode_index": test_batch.episode_index.astype(np.int64),
        "cluster_id": test_info.cluster_id.astype(np.int64),
        "per_episode_first_marginal": np.asarray(redundancy["per_episode_first"], dtype=np.float32),
        "per_episode_second_marginal": np.asarray(redundancy["per_episode_second"], dtype=np.float32),
    }
    gains_out.update({name: np.asarray(value, dtype=np.float32) for name, value in gains.items()})
    return result, gains_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="cifar10,cifar100,svhn,eurosat,dtd")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--q-values", default="2,4,8,16")
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--predictor-epochs", type=int, default=0)
    parser.add_argument("--train-episodes-per-class", type=int, default=0)
    parser.add_argument("--test-episodes-per-class", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--copy-jitter",
        type=float,
        default=0.0,
        help=">0 replaces cluster members by perturbed copies of their seed (symmetric redundancy)",
    )
    parser.add_argument("--n-clusters", type=int, default=0, help="0 = module default")
    parser.add_argument("--cluster-size", type=int, default=0, help="0 = module default")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    names = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    seeds = parse_ints(args.seeds)
    q_values = parse_ints(args.q_values)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": args.out_root.name,
        "argv": sys.argv,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": names,
        "seeds": seeds,
        "q_values": q_values,
        "router_epochs": args.router_epochs,
        "beta": args.beta,
        "states_per_episode": args.states_per_episode,
        "feature_root": str(args.feature_root),
        "pool_rule": "3 clusters x 4 near-duplicate in-class demos + 4 distractors (K=16, B=4)",
        "git_head": _git_head(),
        "imported_source_hashes": {**_source_hashes(), **_r112_hashes()},
    }
    previous = None
    if (args.out_root / "run_manifest.json").exists():
        previous = json.loads((args.out_root / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["started_utc"] = min(previous.get("started_utc", manifest["started_utc"]), manifest["started_utc"])
        manifest["benchmarks"] = sorted(set(previous.get("benchmarks", [])) | set(manifest["benchmarks"]))
        manifest["invocations"] = list(previous.get("invocations", [])) + [list(sys.argv)]
    else:
        manifest["invocations"] = [list(sys.argv)]
    _write_manifest(args.out_root, manifest)

    failures: dict[str, str] = {}
    for name in names:
        try:
            base_config = BENCHMARK_CONFIGS[name]
            overrides = {}
            if args.train_episodes_per_class:
                overrides["train_episodes_per_class"] = args.train_episodes_per_class
            if args.test_episodes_per_class:
                overrides["test_episodes_per_class"] = args.test_episodes_per_class
            config = replace(base_config, **overrides) if overrides else base_config
            protocol = load_protocol(args.feature_root, name, config=config)
            pool_kwargs = {"copy_jitter": args.copy_jitter}
            pool_rule = (
                f"kmeans n_clusters={args.n_clusters or 4} cluster_size={args.cluster_size or 3} "
                f"probe=60 copy_jitter={args.copy_jitter} + 4 distractors (K=16, B=4)"
            )
            if args.n_clusters:
                pool_kwargs["n_clusters"] = args.n_clusters
            if args.cluster_size:
                pool_kwargs["cluster_size"] = args.cluster_size
            predictor_epochs = args.predictor_epochs or config.predictor_epochs
            print(
                f"=== {name}: {protocol.features.num_classes} classes, "
                f"train episodes {config.train_episodes_per_class * protocol.features.num_classes}, "
                f"test episodes {config.test_episodes_per_class * protocol.features.num_classes}, "
                f"pool=redundant ===",
                flush=True,
            )
            for seed in seeds:
                try:
                    if (args.out_root / name / f"seed{seed}" / "result.json").exists() and not args.force:
                        print(json.dumps({"benchmark": name, "seed": seed, "status": "already_complete"}), flush=True)
                        continue
                    started = time.time()
                    train_batch, _ = build_redundant_batch(
                        protocol.features,
                        protocol.mean,
                        protocol.components,
                        query_source=protocol.splits.train_classes,
                        pool_source=protocol.splits.pool,
                        distractor_pool=protocol.splits.distractor_pool,
                        query_split="train",
                        episodes_per_class=config.train_episodes_per_class,
                        seed=seed + 1_000,
                        config=config,
                        **pool_kwargs,
                    )
                    test_batch, test_info = build_redundant_batch(
                        protocol.features,
                        protocol.mean,
                        protocol.components,
                        query_source=protocol.splits.test,
                        pool_source=protocol.splits.pool,
                        distractor_pool=protocol.splits.distractor_pool,
                        query_split="test",
                        episodes_per_class=config.test_episodes_per_class,
                        seed=seed + 2_000,
                        config=config,
                        **pool_kwargs,
                    )
                    model, training_info = train_incontext_predictor(
                        train_batch,
                        num_classes=protocol.features.num_classes,
                        budget=config.budget,
                        epochs=predictor_epochs,
                        seed=seed + 3_000,
                        device=args.device,
                        log_every=50,
                    )
                    cell_dir = args.out_root / name / f"seed{seed}"
                    save_checkpoint(
                        cell_dir / "predictor.pt",
                        model,
                        config={
                            "embed_dim": int(train_batch.query_features.shape[-1]),
                            "num_classes": int(protocol.features.num_classes),
                            "model_kwargs": {
                                "d_model": 128,
                                "nhead": 4,
                                "num_layers": 3,
                                "feed_forward_dim": 256,
                                "dropout": 0.1,
                            },
                            "benchmark": name,
                            "seed": seed,
                        },
                        history=training_info.get("history", []),
                    )
                    ops = DemoExpertOps(model, device=args.device, chunk_episodes=16, chunk_pairs=1024)
                    ops.reset_profile()
                    result, gains = evaluate_cell(
                        seed,
                        protocol,
                        train_batch,
                        test_batch,
                        test_info,
                        ops,
                        q_values=q_values,
                        epochs=args.router_epochs,
                        beta=args.beta,
                        device=args.device,
                        policy_seed=seed + 99,
                        states_per_episode=args.states_per_episode,
                        pool_rule=pool_rule,
                    )
                    result["predictor_training"] = training_info
                    result["expert_profile"] = {"rows": ops.rows, "calls": ops.calls, "seconds": ops.seconds}
                    result["wall_seconds_total"] = time.time() - started
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    (cell_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    np.savez_compressed(cell_dir / "episode_gains.npz", **gains)
                    print(
                        json.dumps(
                            {
                                "benchmark": name,
                                "seed": seed,
                                "q4": result["metrics"].get("ranked_response_q4", {}).get("mean_prediction_gain"),
                                "q4_forced": result["metrics"]
                                .get("ranked_response_q4_forced", {})
                                .get("mean_prediction_gain"),
                                "static": result["metrics"]["static_utility"]["mean_prediction_gain"],
                                "cached": result["metrics"]["cached_mur"]["mean_prediction_gain"],
                                "oracle_greedy": result["metrics"]["oracle_greedy"]["mean_prediction_gain"],
                                "oracle_static": result["metrics"]["oracle_static"]["mean_prediction_gain"],
                                "h_state": result["headroom"]["h_state"],
                                "sel_q4": result["metrics"]["ranked_response_q4"]["mean_selected_contexts"],
                                "sel_q4_forced": result["metrics"]["ranked_response_q4_forced"][
                                    "mean_selected_contexts"
                                ],
                                "wall": round(result["wall_seconds_total"], 1),
                            }
                        ),
                        flush=True,
                    )
                    del model, ops
                    torch.cuda.empty_cache()
                except Exception as exc:  # keep going: report and continue
                    failures[f"{name}_seed{seed}"] = f"{type(exc).__name__}: {exc}"
                    print(
                        json.dumps({"benchmark": name, "seed": seed, "status": "failed", "error": failures[f"{name}_seed{seed}"]}),
                        flush=True,
                    )
                    import traceback

                    traceback.print_exc()
                    torch.cuda.empty_cache()
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"benchmark": name, "status": "failed", "error": failures[name]}), flush=True)
            import traceback

            traceback.print_exc()
    manifest["failures"] = {**manifest.get("failures_previous", {}), **failures}
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_manifest(args.out_root, manifest)
    print(json.dumps({"finished_utc": manifest["finished_utc"], "failures": failures}), flush=True)


if __name__ == "__main__":
    main()
