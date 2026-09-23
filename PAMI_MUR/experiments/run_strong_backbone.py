#!/usr/bin/env python3
"""Strong-backbone confirmation of R-MUR (PAMI additions A1 and A4).

The routing machinery is imported unchanged from the frozen R-MUR protocol
(pair sampling, static/cached utility models, gap-weighted ranking-calibrated
reranker). Only the prediction expert is replaced by a subset-capable
nonlinear backbone that was pretrained on the dataset and then adapted on the
same training episodes the ridge expert sees.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))
sys.path.insert(0, str(KBS / "scripts"))

from mur.synthetic_experiment import (  # noqa: E402
    sample_pair_dataset,
    score_state,
    select_mur,
    select_static,
    train_utility_model,
)
from run_traffic_full_router import select_mmr, select_relevance  # noqa: E402
from run_traffic_ranked_response_router_sweep import train_ranked_response_model  # noqa: E402
from run_traffic_response_router_sweep import paired_ci  # noqa: E402

from fast_router import (  # noqa: E402
    CurvatureCorrectedRouter,
    train_ranked_response_model_fast,
    train_utility_model_fast,
)

from neural_expert import (  # noqa: E402
    SubsetExpertOps,
    finetune_expert,
    fit_response_projection,
    load_checkpoint,
)
from pami_traffic import (  # noqa: E402
    build_candidate_pool,
    load_traffic_data,
    make_neural_batch,
    target_set,
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


# --------------------------------------------------------------------------
# Expert-driven oracle and baseline policies
# --------------------------------------------------------------------------


def select_oracle_static(batch, ops: SubsetExpertOps, budget: int) -> np.ndarray:
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = ops.candidate_marginals(batch, empty)
    order = np.argsort(-marginal, axis=1, kind="stable")[:, :budget]
    selected = empty.copy()
    for row in range(batch.episodes):
        for candidate in order[row]:
            if marginal[row, candidate] <= 0.0:
                break
            selected[row, candidate] = True
    return selected


def select_oracle_greedy(batch, ops: SubsetExpertOps, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = ops.candidate_marginals(batch, selected)
        choice = np.argmax(marginal, axis=1)
        active = marginal[np.arange(batch.episodes), choice] > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def select_random(batch, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for row in range(batch.episodes):
        chosen = rng.choice(batch.candidate_count, size=budget, replace=False)
        selected[row, chosen] = True
    return selected


def candidate_similarity(batch) -> np.ndarray:
    vectors = batch.candidate_history.astype(np.float64).reshape(batch.episodes, batch.candidate_count, -1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    return np.einsum("bik,bjk->bij", vectors, vectors)


def normalized_relevance(batch, relevance: np.ndarray) -> np.ndarray:
    values = np.asarray(relevance, dtype=np.float64)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1e-9)


def select_facility_location(
    batch, relevance: np.ndarray, budget: int, *, similarity: np.ndarray | None = None
) -> np.ndarray:
    """Relevance-weighted coverage: a standard submodular selection baseline."""

    if similarity is None:
        similarity = candidate_similarity(batch)
    weights = normalized_relevance(batch, relevance) + 1e-3
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    coverage = np.zeros((batch.episodes, batch.candidate_count), dtype=np.float64)
    rows = np.arange(batch.episodes)
    for _ in range(budget):
        gains = (weights[:, None, :] * np.maximum(similarity, coverage[:, :, None])).sum(axis=2)
        gains -= (weights * coverage).sum(axis=1, keepdims=True)
        gains[selected] = -np.inf
        choice = np.argmax(gains, axis=1)
        selected[rows, choice] = True
        coverage = np.maximum(coverage, similarity[rows, :, choice])
    return selected


def select_dpp(
    batch, relevance: np.ndarray, budget: int, *, similarity: np.ndarray | None = None, jitter: float = 1e-6
) -> np.ndarray:
    """Greedy MAP for a determinantal point process (diversity baseline)."""

    if similarity is None:
        similarity = candidate_similarity(batch)
    quality = normalized_relevance(batch, relevance) + 0.1
    kernel = quality[:, :, None] * similarity * quality[:, None, :]
    episodes, candidates = kernel.shape[:2]
    selected = np.zeros((episodes, candidates), dtype=bool)
    rows = np.arange(episodes)
    diagonal = np.diagonal(kernel, axis1=1, axis2=2).copy()
    for _ in range(budget):
        sub_index = [np.flatnonzero(selected[row]) for row in range(episodes)]
        residual = diagonal.copy()
        for row in range(episodes):
            index = sub_index[row]
            if index.size == 0:
                continue
            block = kernel[row][np.ix_(index, index)]
            chol = np.linalg.cholesky(block + jitter * np.eye(index.size))
            cross = kernel[row][np.ix_(index, np.arange(candidates))]
            solved = np.linalg.solve(chol, cross)
            residual[row] -= np.sum(solved ** 2, axis=0)
        residual[selected] = -np.inf
        choice = np.argmax(residual, axis=1)
        selected[rows, choice] = True
    return selected


# --------------------------------------------------------------------------
# R-MUR response-aware selection with expert accounting
# --------------------------------------------------------------------------


def select_ranked_response_mur(
    screen_model,
    response_model,
    batch,
    ops: SubsetExpertOps,
    *,
    budget: int,
    q: int,
    device: str,
    projection: np.ndarray | None = None,
    include_interaction: bool = False,
    return_trace: bool = False,
):
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    candidate_count = batch.candidate_count
    q = max(1, min(int(q), candidate_count))
    trace = {key: [] for key in ("choice", "topq", "response_scores")}
    expert_rows = 0
    for _ in range(budget):
        before = ops.rows
        screen = score_state(screen_model, batch, selected, device=device, pack_fn=ops.pack)
        topq = np.argsort(-screen, axis=1, kind="stable")[:, :q]
        qmask = np.zeros_like(selected, dtype=bool)
        qmask[np.arange(batch.episodes)[:, None], topq] = True
        responses = ops.response_summary(
            batch, selected, candidate_mask=qmask, projection=projection,
            include_interaction=include_interaction,
        )

        def response_fn(batch_, selected_):
            return responses

        response_scores = score_state(
            response_model,
            batch,
            selected,
            device=device,
            pack_fn=ops.pack,
            response_fn=response_fn,
        )
        expert_rows += ops.rows - before
        masked = np.where(qmask, response_scores, -np.inf)
        choice = np.argmax(masked, axis=1)
        best = masked[np.arange(batch.episodes), choice]
        active = best > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
        if return_trace:
            trace["choice"].append(choice.astype(np.int64))
            trace["topq"].append(topq.astype(np.int64))
            trace["response_scores"].append(response_scores.astype(np.float32))
    if return_trace:
        for key in trace:
            trace[key] = np.stack(trace[key], axis=0)
        return selected, trace, expert_rows
    return selected, expert_rows


# --------------------------------------------------------------------------
# Training data for the utility router (expert agnostic, protocol identical)
# --------------------------------------------------------------------------


def sample_pair_dataset_fast(
    batch,
    ops: SubsetExpertOps,
    *,
    max_budget: int,
    states_per_episode: int,
    seed: int,
    static_only: bool,
):
    """Pair dataset with the sampled candidate's marginal only.

    Identical to ``sample_pair_dataset`` for the marginals it consumes (the
    reference implementation evaluates the full pool and then indexes one
    column); evaluating just the sampled candidate avoids a factor ``K`` of
    redundant expert calls.
    """

    from mur.synthetic_experiment import PairDataset

    rng = np.random.default_rng(seed)
    if max_budget < 1 or max_budget > batch.candidate_count:
        raise ValueError("max_budget must be in [1, candidate_count]")
    pieces: dict[str, list[np.ndarray]] = {
        key: [] for key in ("query", "selected", "selected_mask", "candidate", "cost", "target", "group_id")
    }
    rows = np.arange(batch.episodes)
    for repetition in range(states_per_episode):
        selected_mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        if not static_only:
            sizes = rng.integers(0, max_budget, size=batch.episodes)
            for row, size in enumerate(sizes):
                if size:
                    selected_mask[row, rng.choice(batch.candidate_count, size=int(size), replace=False)] = True
        candidate_index = np.empty(batch.episodes, dtype=np.int64)
        for row in rows:
            candidate_index[row] = int(rng.choice(np.flatnonzero(~selected_mask[row])))
        marginal = ops.pair_marginals(batch, selected_mask, candidate_index)
        selected, selected_valid = ops.pack(batch, selected_mask, max_budget=max_budget)
        pieces["query"].append(batch.query_embedding)
        pieces["selected"].append(selected)
        pieces["selected_mask"].append(selected_valid)
        pieces["candidate"].append(batch.candidate_embedding[rows, candidate_index])
        pieces["cost"].append(np.ones((batch.episodes, 1), dtype=np.float32))
        pieces["target"].append(marginal.astype(np.float32))
        pieces["group_id"].append((repetition * batch.episodes + rows).astype(np.int64))
    return PairDataset(**{key: np.concatenate(value) for key, value in pieces.items()})


def build_full_state_data(
    batch,
    ops: SubsetExpertOps,
    *,
    max_budget: int,
    states_per_episode: int,
    seed: int,
    projection: np.ndarray | None = None,
    include_interaction: bool = False,
):
    rng = np.random.default_rng(seed)
    query_rows, selected_rows, valid_rows = [], [], []
    candidate_rows, response_rows, target_rows = [], [], []
    for _ in range(states_per_episode):
        selected_mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        sizes = rng.integers(0, max_budget, size=batch.episodes)
        for row, size in enumerate(sizes):
            if size:
                chosen = rng.choice(batch.candidate_count, size=int(size), replace=False)
                selected_mask[row, chosen] = True
        selected, selected_valid = ops.pack(batch, selected_mask, max_budget=max_budget)
        marginal, responses = ops.candidate_marginals_and_responses(
            batch, selected_mask, projection=projection, include_interaction=include_interaction
        )
        query_rows.append(batch.query_embedding)
        selected_rows.append(selected)
        valid_rows.append(selected_valid.astype(bool))
        candidate_rows.append(batch.candidate_embedding)
        response_rows.append(responses)
        target_rows.append(marginal.astype(np.float32))
    return {
        "query": np.concatenate(query_rows, axis=0),
        "selected": np.concatenate(selected_rows, axis=0),
        "selected_mask": np.concatenate(valid_rows, axis=0),
        "candidate": np.concatenate(candidate_rows, axis=0),
        "response": np.concatenate(response_rows, axis=0),
        "target": np.concatenate(target_rows, axis=0),
    }


# --------------------------------------------------------------------------
# Per-cell evaluation
# --------------------------------------------------------------------------


def evaluate_cell(
    seed: int,
    cfg: dict,
    batches,
    ops: SubsetExpertOps,
    *,
    q_values: list[int],
    epochs: int,
    beta: float,
    device: str,
    policy_seed: int,
    response_components: int = 0,
    response_interaction: bool = False,
    capacity: dict | None = None,
    curvature_corrected: bool = False,
    router_kind: str = "mlp",
    response_identity: bool = False,
):
    train, calibration, test = batches
    budget = int(cfg["budget"])
    marginal_fn = lambda batch, selected: ops.candidate_marginals(batch, selected)  # noqa: E731
    pack_fn = lambda batch, selected, max_budget: ops.pack(batch, selected, max_budget=max_budget)  # noqa: E731

    static_pairs = sample_pair_dataset_fast(
        train, ops, max_budget=budget, states_per_episode=4, seed=seed + 4_000, static_only=True
    )
    cached_pairs = sample_pair_dataset_fast(
        train, ops, max_budget=budget, states_per_episode=4, seed=seed + 5_000, static_only=False
    )
    capacity = capacity or {}
    static_model = train_utility_model_fast(
        static_pairs, max_budget=budget, seed=seed + 6_000, device=device, epochs=epochs,
        batch_size=512, **capacity,
    )
    cached_model = train_utility_model_fast(
        cached_pairs, max_budget=budget, seed=seed + 7_000, device=device, epochs=epochs,
        batch_size=512, **capacity,
    )
    projection = None
    if response_identity:
        projection = np.eye(12, dtype=np.float32)
    elif response_components > 0:
        projection = fit_response_projection(
            ops, train, components=response_components, budget=budget, seed=seed + 8_500
        )
    include_interaction = bool(response_interaction)
    full_data = build_full_state_data(
        train, ops, max_budget=budget, states_per_episode=4, seed=seed + 9_000,
        projection=projection, include_interaction=include_interaction,
    )
    ranked_model = train_ranked_response_model_fast(
        full_data, max_budget=budget, seed=seed + 10_000, device=device, epochs=epochs,
        batch_size=128, beta=beta, curvature_corrected=curvature_corrected,
        horizon=int(test.horizon), router_kind=router_kind, **capacity,
    )
    if curvature_corrected:
        ranked_model = CurvatureCorrectedRouter(ranked_model, int(test.horizon))

    candidate_corr = np.asarray(cfg["candidate_sensor_correlations"], dtype=np.float32)
    relevance = np.broadcast_to(candidate_corr[None, :], (test.episodes, test.candidate_count)).copy()
    similarity = candidate_similarity(test)

    base_loss = ops.squared_error(test, np.zeros((test.episodes, test.candidate_count), dtype=bool))
    selection_rows = {name: 0 for name in (
        "base_only", "pool_all", "random", "relevance", "mmr", "facility_location", "dpp",
        "static_utility", "cached_mur",
    )}

    def timed(name, function, *positional, **keywords):
        before = ops.rows
        output = function(*positional, **keywords)
        selection_rows[name] = ops.rows - before
        return output

    oracle_static = timed("oracle_static", select_oracle_static, test, ops, budget)
    oracle_greedy = timed("oracle_greedy", select_oracle_greedy, test, ops, budget)

    selections: dict[str, np.ndarray] = {
        "base_only": np.zeros((test.episodes, test.candidate_count), dtype=bool),
        "pool_all": np.ones((test.episodes, test.candidate_count), dtype=bool),
        "random": select_random(test, budget, policy_seed),
        "relevance": select_relevance(relevance, budget),
        "mmr": select_mmr(test, relevance, budget),
        "facility_location": select_facility_location(test, relevance, budget, similarity=similarity),
        "dpp": select_dpp(test, relevance, budget, similarity=similarity),
        "static_utility": timed(
            "static_utility", select_static, static_model, test, budget=budget, device=device, pack_fn=ops.pack
        ),
        "cached_mur": timed(
            "cached_mur", select_mur, cached_model, test, budget=budget, device=device, pack_fn=ops.pack
        ),
        "oracle_static": oracle_static,
        "oracle_greedy": oracle_greedy,
    }
    selection_rows.setdefault("oracle_static", 0)
    for q in q_values:
        label = "ranked_response_full" if q >= test.candidate_count else f"ranked_response_q{q}"
        before = ops.rows
        selections[label], _ = select_ranked_response_mur(
            cached_model, ranked_model, test, ops, budget=budget, q=q, device=device,
            projection=projection, include_interaction=include_interaction,
        )
        selection_rows[label] = ops.rows - before
    for name in ("base_only", "pool_all", "random", "relevance", "mmr", "facility_location", "dpp"):
        selection_rows[name] = 0

    evaluation_before = ops.rows
    base_for_gain = base_loss
    gains = {
        name: base_for_gain - ops.squared_error(test, mask) for name, mask in selections.items()
    }
    oracle_loss = ops.squared_error(test, oracle_greedy)
    evaluation_rows = ops.rows - evaluation_before
    oracle_gain = base_for_gain - oracle_loss
    positive = oracle_gain > 0.0
    metrics = {}
    for name, mask in selections.items():
        routed = base_for_gain - gains[name]
        gain = gains[name]
        metrics[name] = {
            "mean_prediction_gain": float(np.mean(gain)),
            "negative_transfer_rate": float(np.mean(routed > base_for_gain)),
            "mean_utility_recovery": float(np.mean(gain[positive] / oracle_gain[positive]))
            if np.any(positive)
            else float("nan"),
            "mean_selected_contexts": float(np.mean(np.sum(mask, axis=1))),
        }
    headroom = metrics["oracle_greedy"]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]
    result = {
        "run_id": "PAMI_strong_backbone_rmur",
        "dataset": cfg.get("dataset"),
        "target_sensor": int(cfg["target_sensor_index"]),
        "seed": seed,
        "config": {
            "seed": seed,
            "candidate_count": int(test.candidate_count),
            "budget": budget,
            "response_dim": int(full_data["response"].shape[-1]),
            "response_components": response_components,
            "pool_mode": cfg.get("pool_mode", "correlated"),
            "curvature_corrected": curvature_corrected,
            "router_kind": router_kind,
            "q_values": q_values,
            "router_epochs": epochs,
            "ranking_beta": beta,
            "device": device,
        },
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
    }
    return result, gains


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


PRIMARY_POLICY = "ranked_response_q4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expert-root", type=Path, required=True,
                        help="directory holding seed{seed}/expert.pt checkpoints")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--targets", default="", help="explicit comma-separated target sensor list")
    parser.add_argument("--q-values", default="2,4,8,16")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--response-components", type=int, default=0,
                        help="direction-preserving PCA components appended to the scalar response summary")
    parser.add_argument("--router-hidden", type=int, default=64)
    parser.add_argument("--router-set-dim", type=int, default=64)
    parser.add_argument("--router-depth", type=int, default=2)
    parser.add_argument("--pool-mode", default="correlated", choices=["correlated", "anticorrelated"])
    parser.add_argument("--router-kind", default="mlp", choices=["mlp", "bilinear"])
    parser.add_argument("--response-identity", action="store_true",
                        help="append the raw response vector instead of a PCA projection")
    parser.add_argument("--curvature-corrected", action="store_true",
                        help="train the head on the interaction term and subtract the observable ||d||^2 at inference")
    parser.add_argument("--response-interaction", action="store_true",
                        help="append the observable <base prediction, response> term of the exact identity")
    parser.add_argument("--protocol-seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    q_values = parse_ints(args.q_values)
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)

    data = load_traffic_data(
        args.data_root / args.dataset,
        history_length=args.history_length,
        horizon=args.horizon,
    )
    if args.targets.strip():
        targets = np.asarray(parse_ints(args.targets), dtype=np.int64)
    else:
        targets = target_set(data.num_nodes, args.target_count)
    records = []
    for target in targets:
        target = int(target)
        candidates = build_candidate_pool(
            data.correlation[target],
            target_sensor=target,
            candidate_count=args.candidate_count,
            seed=args.protocol_seed,
            mode=args.pool_mode,
        )
        correlation = data.correlation[target]
        cfg = {
            "dataset": args.dataset,
            "budget": args.budget,
            "expert_repeats": 3,
            "ridge_penalty": 10.0,
            "candidate_sensor_correlations": correlation[candidates].astype(np.float32).tolist(),
            "candidate_sensors": candidates.tolist(),
            "target_sensor_index": target,
            "pool_mode": args.pool_mode,
            "dataset": args.dataset,
        }
        for seed in seeds:
            started = time.time()
            checkpoint = args.expert_root / f"seed{seed}" / "expert.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"missing expert checkpoint: {checkpoint}")
            model, _ = load_checkpoint(checkpoint, device=args.device)
            model = copy.deepcopy(model)
            rng = np.random.default_rng(seed + 2_000)
            batches = []
            for starts, count in (
                (data.train_times, args.train_episodes),
                (data.validation_times, args.calibration_episodes),
                (data.test_times, args.test_episodes),
            ):
                times = rng.choice(starts, size=min(count, len(starts)), replace=False)
                batches.append(
                    make_neural_batch(
                        data, times, target_sensor=target, candidate_sensors=candidates
                    )
                )
            batches = tuple(batches)
            finetune = finetune_expert(
                model, batches[0], epochs=args.finetune_epochs, batch_size=256,
                learning_rate=args.finetune_lr, budget=args.budget, seed=seed + 7_777,
                device=args.device, validation_batch=batches[1],
            )
            ops = SubsetExpertOps(model, device=args.device)
            ops.reset_profile()
            result, gains = evaluate_cell(
                seed, cfg, batches, ops,
                q_values=q_values, epochs=args.router_epochs, beta=args.beta,
                device=args.device, policy_seed=seed + 99,
                response_components=args.response_components,
                response_interaction=args.response_interaction,
                curvature_corrected=args.curvature_corrected,
                router_kind=args.router_kind,
                response_identity=args.response_identity,
                capacity={
                    "hidden_dim": args.router_hidden,
                    "set_dim": args.router_set_dim,
                    "depth": args.router_depth,
                },
            )
            profile = {"expert_calls": ops.calls, "expert_rows": ops.rows, "expert_seconds": ops.seconds}
            result["finetune"] = {k: v for k, v in finetune.items() if k != "history"}
            result["expert_profile"] = profile
            result["wall_seconds"] = time.time() - started
            records.append(result)
            cell_dir = args.out_root / f"target{target}_seed{seed}"
            cell_dir.mkdir(parents=True)
            (cell_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            np.savez_compressed(
                cell_dir / "episode_gains.npz",
                **{name: np.asarray(values, dtype=np.float32) for name, values in gains.items()},
            )
            del model
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "target": target,
                        "seed": seed,
                        "rmur_q4": result["metrics"].get("ranked_response_q4", {}).get("mean_prediction_gain"),
                        "rmur_full": result["metrics"].get("ranked_response_full", {}).get("mean_prediction_gain"),
                        "static": result["metrics"]["static_utility"]["mean_prediction_gain"],
                        "cached": result["metrics"]["cached_mur"]["mean_prediction_gain"],
                        "oracle_greedy": result["metrics"]["oracle_greedy"]["mean_prediction_gain"],
                        "wall": round(result["wall_seconds"], 1),
                    }
                ),
                flush=True,
            )

    metric_names = ["mean_prediction_gain", "negative_transfer_rate", "mean_utility_recovery", "mean_selected_contexts"]
    policy_names = list(records[0]["metrics"].keys())
    rows = ["dataset,target_sensor,seed,policy," + ",".join(metric_names)]
    for record in records:
        for policy in policy_names:
            metrics = record["metrics"][policy]
            rows.append(
                f"{record['dataset']},{record['target_sensor']},{record['seed']},{policy},"
                + ",".join(f"{metrics[name]:.8f}" for name in metric_names)
            )
    (args.out_root / "per_target_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Target-level aggregation with target-level bootstrap CI.
    def bootstrap(values: np.ndarray, *, n_boot: int = 5000, seed: int = 0):
        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
        return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

    target_stats = []
    for target in sorted({r["target_sensor"] for r in records}):
        subset = [r for r in records if r["target_sensor"] == target]
        def mean_of(policy, metric="mean_prediction_gain"):
            return float(np.mean([r["metrics"][policy][metric] for r in subset]))
        og = mean_of("oracle_greedy")
        os_gain = mean_of("oracle_static")
        stat = {
            "target_sensor": target,
            "runs": len(subset),
            "oracle_static_gain": os_gain,
            "oracle_greedy_gain": og,
            "H_state": (og - os_gain) / og if og > 0 else float("nan"),
        }
        for policy in policy_names:
            stat[f"{policy}_gain"] = mean_of(policy)
        # The primary contrast is the smallest shortlist that was actually run.
        # Hardcoding q4 crashed the summary when a sweep used q=16 (the runner
        # renames that policy to ranked_response_full), so resolve it instead.
        primary = PRIMARY_POLICY
        if f"{primary}_gain" not in stat:
            candidates = [p for p in policy_names if p.startswith("ranked_response")]
            if not candidates:
                raise SystemExit(f"no response-aware policy in {policy_names}")
            primary = sorted(candidates, key=lambda p: (p != "ranked_response_full", p))[0]
        stat["delta_static"] = stat[f"{primary}_gain"] - stat["static_utility_gain"]
        stat["delta_cached"] = stat[f"{primary}_gain"] - stat["cached_mur_gain"]
        stat["primary_policy"] = primary
        target_stats.append(stat)

    primary = target_stats[0].get("primary_policy", PRIMARY_POLICY)
    delta_static = np.asarray([t["delta_static"] for t in target_stats])
    delta_cached = np.asarray([t["delta_cached"] for t in target_stats])
    q4 = np.asarray([t[f"{primary}_gain"] for t in target_stats])
    ds_mean, ds_lo, ds_hi = bootstrap(delta_static, seed=1)
    dc_mean, dc_lo, dc_hi = bootstrap(delta_cached, seed=2)
    dataset_stats = {
        "dataset": args.dataset,
        "targets": len(target_stats),
        "seeds": seeds,
        "cells": len(records),
        "gate": {
            f"G_{primary}_mean": float(np.mean(q4)),
            "delta_static": {"mean": ds_mean, "low": ds_lo, "high": ds_hi},
            "delta_cached": {"mean": dc_mean, "low": dc_lo, "high": dc_hi},
            "targets_positive_gain": int(np.sum(q4 > 0)),
            "targets_above_static": int(np.sum(delta_static > 0)),
            "targets_above_cached": int(np.sum(delta_cached > 0)),
            "pass_g1": bool(ds_lo > 0.0 and dc_lo > 0.0),
        },
        "policy_means": {
            policy: float(np.mean([r["metrics"][policy]["mean_prediction_gain"] for r in records]))
            for policy in policy_names
        },
        "expert_rows_per_cell": {
            policy: float(np.mean([r["expert_rows"].get(policy, 0) for r in records]))
            for policy in policy_names
        },
        "wall_seconds_mean": float(np.mean([r["wall_seconds"] for r in records])),
        "target_records": target_stats,
    }
    (args.out_root / "target_statistics.json").write_text(
        json.dumps(dataset_stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(dataset_stats["gate"], indent=2))


if __name__ == "__main__":
    main()
