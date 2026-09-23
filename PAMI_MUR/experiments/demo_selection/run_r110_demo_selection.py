#!/usr/bin/env python3
"""R110: R-MUR on a second task family -- in-context demonstration selection.

The routing machinery is imported unchanged from the frozen R-MUR protocol
(pair sampling, static/cached utility models, gap-weighted ranking-calibrated
reranker, paired CIs, baselines).  The only replacement is the prediction
expert: a frozen in-context classifier over CLIP embeddings of a query batch
and a candidate demonstration pool, scored by query-batch cross-entropy.
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
from frozen_helpers import (  # noqa: E402  (pinned snapshot of run_strong_backbone helpers)
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
from incontext_predictor import (  # noqa: E402
    DemoExpertOps,
    cross_entropy_from_logits,
    save_checkpoint,
)
from train_predictor import train_incontext_predictor  # noqa: E402


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def relevance_of(batch) -> np.ndarray:
    """kNN relevance: cosine similarity of each candidate to the query summary."""

    return np.einsum("ed,ekd->ek", batch.query_embedding, batch.candidate_embedding).astype(np.float32)


def predictor_quality(ops: DemoExpertOps, batch, relevance: np.ndarray, *, budget: int, seed: int) -> dict:
    """Accuracy / CE of the frozen predictor at several demonstration counts."""

    episodes, candidates = batch.episodes, batch.candidate_count
    order = np.argsort(-relevance, axis=1, kind="stable")
    rng = np.random.default_rng(seed)
    masks = {
        "demos_0": np.zeros((episodes, candidates), dtype=bool),
        "demos_2_relevance": np.zeros((episodes, candidates), dtype=bool),
        "demos_4_relevance": np.zeros((episodes, candidates), dtype=bool),
        "demos_4_random": np.zeros((episodes, candidates), dtype=bool),
        "demos_4_farthest": np.zeros((episodes, candidates), dtype=bool),
        "demos_16_pool": np.ones((episodes, candidates), dtype=bool),
    }
    rows = np.arange(episodes)
    masks["demos_2_relevance"][rows[:, None], order[:, :2]] = True
    masks["demos_4_relevance"][rows[:, None], order[:, :4]] = True
    masks["demos_4_farthest"][rows[:, None], order[:, -4:]] = True
    for row in range(episodes):
        masks["demos_4_random"][row, rng.choice(candidates, size=budget, replace=False)] = True
    quality = {}
    for label, mask in masks.items():
        logits = ops.predict(batch, mask)
        loss = cross_entropy_from_logits(logits, batch.query_labels)
        accuracy = np.mean(np.argmax(logits, axis=-1) == batch.query_labels[:, None], axis=1)
        quality[label] = {
            "mean_ce": float(np.mean(loss)),
            "mean_accuracy": float(np.mean(accuracy)),
            "mean_demos": float(np.mean(mask.sum(axis=1))),
        }
    return quality


def evaluate_cell(
    seed: int,
    protocol,
    train_batch,
    test_batch,
    ops: DemoExpertOps,
    *,
    q_values: list[int],
    epochs: int,
    beta: float,
    device: str,
    policy_seed: int,
    states_per_episode: int = 4,
    pair_sampler: str = "fast",
) -> tuple[dict, dict]:
    """One (benchmark, seed) cell: train the frozen-method utility models, score policies."""

    config = protocol.config
    budget = config.budget
    started = time.time()
    marginal_fn = lambda batch, selected: ops.candidate_marginals(batch, selected)  # noqa: E731
    pack_fn = lambda batch, selected, max_budget: ops.pack(batch, selected, max_budget=max_budget)  # noqa: E731

    # Pair construction.  ``pair_sampler="fast"`` uses the PAMI helper that
    # evaluates only the sampled candidate (verified numerically identical to
    # the frozen full-row sampler by tests/test_demo_selection.py); the frozen
    # reference path stays available through ``pair_sampler="frozen"``.
    if pair_sampler == "fast":
        static_pairs = sample_pair_dataset_fast(
            train_batch,
            ops,
            max_budget=budget,
            states_per_episode=states_per_episode,
            seed=seed + 4_000,
            static_only=True,
        )
        cached_pairs = sample_pair_dataset_fast(
            train_batch,
            ops,
            max_budget=budget,
            states_per_episode=states_per_episode,
            seed=seed + 5_000,
            static_only=False,
        )
    else:
        static_pairs = sample_pair_dataset(
            train_batch,
            max_budget=budget,
            states_per_episode=states_per_episode,
            seed=seed + 4_000,
            static_only=True,
            marginal_fn=marginal_fn,
            pack_fn=pack_fn,
        )
        cached_pairs = sample_pair_dataset(
            train_batch,
            max_budget=budget,
            states_per_episode=states_per_episode,
            seed=seed + 5_000,
            static_only=False,
            marginal_fn=marginal_fn,
            pack_fn=pack_fn,
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

    selection_rows = {name: 0 for name in ("static_utility", "cached_mur", "oracle_static", "oracle_greedy")}

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
        label = "ranked_response_full" if q >= test_batch.candidate_count else f"ranked_response_q{q}"
        before = ops.rows
        selections[label], _ = select_ranked_response_mur(
            cached_model, ranked_model, test_batch, ops, budget=budget, q=q, device=device
        )
        selection_rows[label] = ops.rows - before

    evaluation_rows_before = ops.rows
    losses = {name: ops.loss(test_batch, mask) for name, mask in selections.items()}
    evaluation_rows = ops.rows - evaluation_rows_before
    gains = {name: base_loss - value for name, value in losses.items()}
    oracle_gain = gains["oracle_greedy"]
    positive = oracle_gain > 0.0
    metrics = {}
    for name, mask in selections.items():
        gain = gains[name]
        metrics[name] = {
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
        "run_id": "R110_demo_selection",
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
            "pair_sampler": pair_sampler,
            "device": device,
            "pair_dataset_rows": int(static_pairs.target.size),
        },
        "base_loss": float(np.mean(base_loss)),
        "predictor_quality": quality,
        "metrics": metrics,
        "expert_rows": selection_rows,
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
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--beta", type=float, default=1.0, help="ranking-loss weight")
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--predictor-epochs", type=int, default=0, help="0 = per-benchmark default")
    parser.add_argument("--pair-sampler", choices=["fast", "frozen"], default="fast")
    parser.add_argument("--force", action="store_true", help="recompute cells that already have result.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only-seed", type=int, default=0)
    parser.add_argument("--train-episodes-per-class", type=int, default=0, help="smoke override; 0 = protocol default")
    parser.add_argument("--test-episodes-per-class", type=int, default=0, help="smoke override; 0 = protocol default")
    parser.add_argument("--query-batch-size", type=int, default=0, help="smoke override; 0 = protocol default")
    args = parser.parse_args()

    names = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    seeds = parse_ints(args.seeds)
    if args.only_seed:
        seeds = [args.only_seed]
    q_values = parse_ints(args.q_values)
    # The output root may already hold results from an earlier invocation of
    # this same sweep: complete cells are skipped unless --force is given, so a
    # transient co-tenant OOM can be recovered by simply re-running.
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
        "git_head": _git_head(),
        "imported_source_hashes": _source_hashes(),
    }
    previous = None
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
        if args.query_batch_size:
            overrides["query_batch_size"] = args.query_batch_size
        config = replace(base_config, **overrides) if overrides else base_config
        protocol = load_protocol(args.feature_root, name, config=config)
        predictor_epochs = args.predictor_epochs or config.predictor_epochs
        print(
            f"=== {name}: {protocol.features.num_classes} classes, "
            f"train episodes {config.train_episodes_per_class * protocol.features.num_classes}, "
            f"test episodes {config.test_episodes_per_class * protocol.features.num_classes} ===",
            flush=True,
        )
        for seed in seeds:
          try:
            if (args.out_root / name / f"seed{seed}" / "result.json").exists() and not args.force:
                print(json.dumps({"benchmark": name, "seed": seed, "status": "already_complete"}), flush=True)
                continue
            started = time.time()
            train_batch = protocol.train_batch(seed + 1_000)
            test_batch = protocol.test_batch(seed + 2_000)
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
                    "model_kwargs": {"d_model": 128, "nhead": 4, "num_layers": 3, "feed_forward_dim": 256, "dropout": 0.1},
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
                ops,
                q_values=q_values,
                epochs=args.router_epochs,
                beta=args.beta,
                device=args.device,
                policy_seed=seed + 99,
                states_per_episode=args.states_per_episode,
                pair_sampler=args.pair_sampler,
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
                        "q4_gain": result["metrics"].get("ranked_response_q4", {}).get("mean_prediction_gain"),
                        "static": result["metrics"]["static_utility"]["mean_prediction_gain"],
                        "cached": result["metrics"]["cached_mur"]["mean_prediction_gain"],
                        "oracle_greedy": result["metrics"]["oracle_greedy"]["mean_prediction_gain"],
                        "oracle_static": result["metrics"]["oracle_static"]["mean_prediction_gain"],
                        "acc_0": result["predictor_quality"]["demos_0"]["mean_accuracy"],
                        "acc_4": result["predictor_quality"]["demos_4_relevance"]["mean_accuracy"],
                        "acc_16": result["predictor_quality"]["demos_16_pool"]["mean_accuracy"],
                        "wall": round(result["wall_seconds_total"], 1),
                    }
                ),
                flush=True,
            )
            del model, ops
            torch.cuda.empty_cache()
          except Exception as exc:  # keep going: report the failure, do not abort the sweep
            failures[f"{name}_seed{seed}"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"benchmark": name, "seed": seed, "status": "failed", "error": failures[f"{name}_seed{seed}"]}), flush=True)
            import traceback

            traceback.print_exc()
            torch.cuda.empty_cache()
      except Exception as exc:
        failures[name] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"benchmark": name, "status": "failed", "error": failures[name]}), flush=True)
        import traceback

        traceback.print_exc()
    manifest["failures"] = failures
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    previous = None
    if (args.out_root / "run_manifest.json").exists():
        previous = json.loads((args.out_root / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["started_utc"] = min(previous.get("started_utc", manifest["started_utc"]), manifest["started_utc"])
        manifest["benchmarks"] = sorted(set(previous.get("benchmarks", [])) | set(manifest["benchmarks"]))
        manifest["invocations"] = list(previous.get("invocations", [])) + [list(sys.argv)]
        manifest["failures_previous"] = previous.get("failures", {})
    else:
        manifest["invocations"] = [list(sys.argv)]
    _write_manifest(args.out_root, manifest)
    print(json.dumps({"finished_utc": manifest["finished_utc"], "failures": failures}), flush=True)


def _write_manifest(out_root: Path, manifest: dict) -> None:
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _source_hashes() -> dict:
    """md5 of every borrowed implementation file, so a result pins its revision."""

    import hashlib

    files = {
        "PAMI_MUR/experiments/demo_selection/frozen_helpers.py": HERE / "frozen_helpers.py",
        "PAMI_MUR/experiments/run_strong_backbone.py": PAMI / "experiments/run_strong_backbone.py",
        "PAMI_MUR/experiments/neural_expert.py": PAMI / "experiments/neural_expert.py",
        "KBS_MUR/src/mur/synthetic_experiment.py": KBS / "src/mur/synthetic_experiment.py",
        "KBS_MUR/src/mur/model.py": KBS / "src/mur/model.py",
        "KBS_MUR/src/mur/traffic.py": KBS / "src/mur/traffic.py",
        "KBS_MUR/scripts/run_traffic_ranked_response_router_sweep.py": KBS / "scripts/run_traffic_ranked_response_router_sweep.py",
        "KBS_MUR/scripts/run_traffic_response_router_sweep.py": KBS / "scripts/run_traffic_response_router_sweep.py",
        "KBS_MUR/scripts/run_traffic_full_router.py": KBS / "scripts/run_traffic_full_router.py",
    }
    hashes = {}
    for label, path in files.items():
        try:
            hashes[label] = hashlib.md5(path.read_bytes()).hexdigest()
        except OSError:
            hashes[label] = "missing"
    return hashes


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - provenance best effort
        return "unknown"


if __name__ == "__main__":
    main()
