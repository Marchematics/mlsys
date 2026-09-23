#!/usr/bin/env python3
"""R124: signed, unconstrained curvature coefficient in the R115 second family.

R123's fixed-sign bilinear head (``<g, d_bar> - lambda kappa``, ``lambda >= 0``)
ranked the marginal better than the frozen scalar head on 4/5 benchmarks but lost
on CIFAR-10, where ``corr(score, kappa)`` is *positive* (+0.257) while it is
negative on CIFAR-100 (-0.590) and SVHN (-0.360).  R124 adds a third head,

    score(state, j) = <g_theta(state, candidate), d_bar_j> + c_theta(state) * kappa_j ,

with ``c_theta`` signed and unconstrained (one extra output unit, initialised at
-0.25).  For squared loss the exact identity fixes ``c = -1``; a learned signed
``c`` is therefore only meaningful for the cross-entropy family, and the traffic
experiments keep the exact form.  Everything else is inherited unchanged, and the
frozen scalar head and the fixed-sign bilinear head are trained on the *same*
cells so the comparison is a three-way paired ablation.

Pre-registered predictions (written before the sweep):
(i)  the signed head's mean within-state Spearman correlation with the true
     marginal is at least as high as the fixed-sign bilinear head's on at least
     4 of 5 benchmarks;
(ii) forced-budget R-MUR q=4 with the signed head beats Static Utility with a
     paired 95 % CI lower bound above zero on at least 4 of 5 benchmarks and on
     the pooled analysis.

Mechanism prediction: the learned ``c`` moves towards zero (or becomes positive)
exactly on the benchmarks where the fixed negative sign hurt (CIFAR-10).
"""

from __future__ import annotations

import argparse
import json
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

from mur.synthetic_experiment import select_mur, select_static, train_utility_model  # noqa: E402
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

from bilinear_response import BilinearDemoExpertOps, train_bilinear_response_model  # noqa: E402
from signed_curvature import (  # noqa: E402
    curvature_summary,
    train_signed_response_model,
)
from demo_protocol import BENCHMARK_CONFIGS, load_protocol  # noqa: E402
from forced_budget import select_ranked_response_mur_forced  # noqa: E402
from incontext_predictor import DemoExpertOps  # noqa: E402
from prototype_predictor import (  # noqa: E402
    save_prototype_checkpoint,
    train_prototype_predictor,
)
from redundant_pool import build_redundant_batch  # noqa: E402

# R110 / R112 / R115 helpers reused verbatim.
from run_r110_demo_selection import (  # noqa: E402
    _git_head,
    _source_hashes,
    _write_manifest,
    parse_ints,
    predictor_quality,
    relevance_of,
)
from run_r112_redundant_demo import redundancy_diagnostics  # noqa: E402

FROZEN_PREFIX = "ranked_response_frozen_"
BILINEAR_PREFIX = "ranked_response_bilinear_"


def _r123_hashes() -> dict:
    import hashlib

    files = {
        "PAMI_MUR/experiments/demo_selection/bilinear_response.py": HERE / "bilinear_response.py",
        "PAMI_MUR/experiments/demo_selection/signed_curvature.py": HERE / "signed_curvature.py",
        "PAMI_MUR/experiments/demo_selection/run_r124_signed.py": HERE / "run_r124_signed.py",
        "PAMI_MUR/experiments/demo_selection/prototype_predictor.py": HERE / "prototype_predictor.py",
        "PAMI_MUR/experiments/bilinear_router.py": PAMI / "experiments/bilinear_router.py",
        "PAMI_MUR/experiments/demo_selection/redundant_pool.py": HERE / "redundant_pool.py",
        "PAMI_MUR/experiments/demo_selection/forced_budget.py": HERE / "forced_budget.py",
    }
    return {label: hashlib.md5(path.read_bytes()).hexdigest() for label, path in files.items()}


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
    curvature_multiplier: float = 0.25,
    learned_curvature: bool = False,
    curvature_mode: str = "bound",
    init_curvature: float = -0.25,
) -> tuple[dict, dict]:
    """One cell: train the frozen utility models *and* both response heads, score all policies."""

    config = protocol.config
    budget = config.budget
    prediction_dim = ops.num_classes
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

    # Frozen scalar head on the seven-dimensional response summary (R110/R112/R115 path).
    full_data = build_full_state_data(
        train_batch, ops, max_budget=budget, states_per_episode=states_per_episode, seed=seed + 9_000
    )
    ranked_model = train_ranked_response_model(
        full_data, max_budget=budget, seed=seed + 10_000, device=device, epochs=epochs, beta=beta
    )

    # Theory-shaped bilinear head on the packed response block.
    bilinear_ops = BilinearDemoExpertOps(ops.model, device=device, chunk_episodes=16, chunk_pairs=1024)
    bilinear_data = build_full_state_data(
        train_batch, bilinear_ops, max_budget=budget, states_per_episode=states_per_episode, seed=seed + 9_000
    )
    bilinear_model = train_bilinear_response_model(
        bilinear_data,
        max_budget=budget,
        seed=seed + 10_000,
        device=device,
        prediction_dim=prediction_dim,
        epochs=epochs,
        beta=beta,
        curvature_multiplier=curvature_multiplier,
        learned_curvature=learned_curvature,
        curvature_mode=curvature_mode,
    )

    # signed, unconstrained curvature coefficient (primary object of R124)
    signed_model = train_signed_response_model(
        bilinear_data,
        max_budget=budget,
        seed=seed + 10_000,
        device=device,
        prediction_dim=prediction_dim,
        epochs=epochs,
        beta=beta,
        init_curvature=init_curvature,
    )
    coefficient_stats = curvature_summary(
        signed_model, ops, test_batch, states=states_per_episode, seed=seed + 31, budget=budget
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
        frozen_stem = FROZEN_PREFIX + ("full" if q >= test_batch.candidate_count else f"q{q}")
        bilinear_stem = BILINEAR_PREFIX + ("full" if q >= test_batch.candidate_count else f"q{q}")
        # signed curvature head (primary)
        for head, head_ops, label in (
            (signed_model, bilinear_ops, stem),
            (bilinear_model, bilinear_ops, bilinear_stem),
        ):
            before = ops.rows
            selections[label], _ = select_ranked_response_mur(
                cached_model, head, test_batch, head_ops, budget=budget, q=q, device=device
            )
            selection_rows[label] = ops.rows - before
            before = ops.rows
            selections[label + "_forced"], _ = select_ranked_response_mur_forced(
                cached_model, head, test_batch, head_ops, budget=budget, q=q, device=device
            )
            selection_rows[label + "_forced"] = ops.rows - before
        # frozen scalar head on identical cells (reference)
        before = ops.rows
        selections[frozen_stem], _ = select_ranked_response_mur(
            cached_model, ranked_model, test_batch, ops, budget=budget, q=q, device=device
        )
        selection_rows[frozen_stem] = ops.rows - before
        before = ops.rows
        selections[frozen_stem + "_forced"], _ = select_ranked_response_mur_forced(
            cached_model, ranked_model, test_batch, ops, budget=budget, q=q, device=device
        )
        selection_rows[frozen_stem + "_forced"] = ops.rows - before

    redundancy_started = ops.rows
    redundancy = redundancy_diagnostics(ops, test_batch, test_info)
    redundancy_rows = ops.rows - redundancy_started

    evaluation_rows_before = ops.rows
    losses = {name: ops.loss(test_batch, mask) for name, mask in selections.items()}
    evaluation_rows = ops.rows - evaluation_rows_before
    gains = {name: base_loss - value for name, value in losses.items()}
    oracle_gain = gains["oracle_greedy"]
    positive = oracle_gain > 0.0
    in_class = test_info.cluster_id >= 0
    metrics = {}
    for name, mask in selections.items():
        distinct = []
        for episode in range(test_batch.episodes):
            chosen = mask[episode] & in_class[episode]
            distinct.append(len(np.unique(test_info.cluster_id[episode][chosen])) if np.any(chosen) else 0.0)
        metrics[name] = {
            "mean_prediction_gain": float(np.mean(gains[name])),
            "mean_loss": float(np.mean(losses[name])),
            "negative_transfer_rate": float(np.mean(gains[name] < 0.0)),
            "mean_utility_recovery": float(np.mean(gains[name][positive] / oracle_gain[positive]))
            if np.any(positive)
            else float("nan"),
            "mean_selected_contexts": float(np.mean(np.sum(mask, axis=1))),
            "mean_distinct_clusters": float(np.mean(distinct)),
        }
    headroom = metrics["oracle_greedy"]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]
    result = {
        "run_id": "R124_signed_curvature",
        "benchmark": protocol.features.benchmark,
        "seed": seed,
        "config": {
            "seed": seed,
            "candidate_count": int(test_batch.candidate_count),
            "budget": budget,
            "query_batch_size": int(test_batch.query_batch_size),
            "episodes": int(test_batch.episodes),
            "classes": int(protocol.features.num_classes),
            "response_dim_frozen": int(full_data["response"].shape[-1]),
            "response_dim_bilinear": int(bilinear_data["response"].shape[-1]),
            "q_values": q_values,
            "router_epochs": epochs,
            "ranking_beta": beta,
            "states_per_episode": states_per_episode,
            "pair_sampler": "fast",
            "pool_rule": "R112 redundancy-structured pool (kmeans 4x3 + 4 distractors)",
            "predictor": "R115 learned-metric attention prototype",
            "response_head": "signed <g_theta, d_bar> + c_theta(state) * mean_i ||d_i||^2",
            "reference_heads": ["frozen 7-dim scalar MLP", "fixed-sign bilinear (lambda=0.25)"],
            "curvature_mode": curvature_mode,
            "curvature_multiplier": float(curvature_multiplier),
            "learned_curvature": bool(learned_curvature),
            "device": device,
        },
        "base_loss": float(np.mean(base_loss)),
        "predictor_quality": quality,
        "redundancy": {key: value for key, value in redundancy.items() if not key.startswith("per_episode")},
        "signed_curvature_coefficient": coefficient_stats,
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
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--curvature-multiplier", type=float, default=0.25)
    parser.add_argument("--learned-curvature", action="store_true")
    parser.add_argument("--curvature-mode", choices=["bound", "realised"], default="bound")
    parser.add_argument("--init-curvature", type=float, default=-0.25)
    parser.add_argument("--train-episodes-per-class", type=int, default=0)
    parser.add_argument("--test-episodes-per-class", type=int, default=0)
    parser.add_argument("--force", action="store_true")
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
        "predictor": "R115 learned-metric attention prototype",
        "response_head": "signed curvature <g, d_bar> + c_theta * kappa",
        "curvature_mode": args.curvature_mode,
        "curvature_multiplier": args.curvature_multiplier,
        "learned_curvature": bool(args.learned_curvature),
        "git_head": _git_head(),
        "imported_source_hashes": {**_source_hashes(), **_r123_hashes()},
    }
    if (args.out_root / "run_manifest.json").exists():
        previous = json.loads((args.out_root / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["started_utc"] = min(previous.get("started_utc", manifest["started_utc"]), manifest["started_utc"])
        manifest["benchmarks"] = sorted(set(previous.get("benchmarks", [])) | set(manifest["benchmarks"]))
        manifest["invocations"] = list(previous.get("invocations", [])) + [list(sys.argv)]
        manifest["failures_previous"] = previous.get("failures", {})
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
            predictor_epochs = args.predictor_epochs or config.predictor_epochs
            print(
                f"=== {name}: {protocol.features.num_classes} classes, "
                f"train episodes {config.train_episodes_per_class * protocol.features.num_classes}, "
                f"test episodes {config.test_episodes_per_class * protocol.features.num_classes}, "
                f"head=bilinear ===",
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
                    )
                    model, training_info = train_prototype_predictor(
                        train_batch,
                        num_classes=protocol.features.num_classes,
                        budget=config.budget,
                        epochs=predictor_epochs,
                        seed=seed + 3_000,
                        device=args.device,
                        proj_dim=args.proj_dim,
                        log_every=50,
                    )
                    cell_dir = args.out_root / name / f"seed{seed}"
                    save_prototype_checkpoint(
                        cell_dir / "predictor.pt",
                        model,
                        config={
                            "embed_dim": int(train_batch.query_features.shape[-1]),
                            "num_classes": int(protocol.features.num_classes),
                            "model_kwargs": {"proj_dim": args.proj_dim},
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
                        curvature_multiplier=args.curvature_multiplier,
                        learned_curvature=args.learned_curvature,
                        curvature_mode=args.curvature_mode,
                        init_curvature=args.init_curvature,
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
                                "signed_q4_forced": result["metrics"]
                                .get("ranked_response_q4_forced", {})
                                .get("mean_prediction_gain"),
                                "bilinear_q4_forced": result["metrics"]
                                .get(f"{BILINEAR_PREFIX}q4_forced", {})
                                .get("mean_prediction_gain"),
                                "c_mean": result["signed_curvature_coefficient"]["mean"],
                                "c_std": result["signed_curvature_coefficient"]["std"],
                                "frozen_q4_forced": result["metrics"]
                                .get(f"{FROZEN_PREFIX}q4_forced", {})
                                .get("mean_prediction_gain"),
                                "static": result["metrics"]["static_utility"]["mean_prediction_gain"],
                                "cached": result["metrics"]["cached_mur"]["mean_prediction_gain"],
                                "oracle_greedy": result["metrics"]["oracle_greedy"]["mean_prediction_gain"],
                                "h_state": result["headroom"]["h_state"],
                                "wall": round(result["wall_seconds_total"], 1),
                            }
                        ),
                        flush=True,
                    )
                    del model, ops
                    torch.cuda.empty_cache()
                except Exception as exc:
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
