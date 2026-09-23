#!/usr/bin/env python3
"""Ranking-calibrated R-MUR gate on METR-LA.

The response-aware utility model is trained on full states with
Huber regression plus a utility-gap-weighted pairwise ranking loss.
Inference uses the same cached screening plus response reranking as R077.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F_ops
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mur.synthetic_experiment import (  # noqa: E402
    sample_pair_dataset,
    score_state,
    select_mur,
    select_static,
    train_utility_model,
)
from mur.traffic import (  # noqa: E402
    fit_subset_expert,
    observed_candidate_marginals,
    pack_selected,
    response_summary,
    squared_error,
)
from mur.model import MarginalUtilityRouter, RouterConfig  # noqa: E402
from run_traffic_full_router import (  # noqa: E402
    load_protocol,
    select_mmr,
    select_oracle_greedy,
    select_oracle_static,
    select_relevance,
)
from run_traffic_response_router_sweep import paired_ci  # noqa: E402


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def select_ranked_response_mur(
    screen_model,
    response_model,
    batch,
    expert,
    *,
    budget: int,
    q: int,
    device: str,
    return_diagnostics: bool = False,
    collect_observables: bool = False,
):
    """Two-stage response-aware greedy selection with optional diagnostic trace.

    ``collect_observables`` additionally returns the per-episode statistics a
    label-free confidence gate could key on, all of which are available at
    decision time: the cached screen's top-two margin, the response model's
    top-two margin and top score within the shortlist, and the prediction-change
    norm of the candidate it picked.
    """

    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    candidate_count = batch.candidate_count
    q = max(1, min(int(q), candidate_count))
    trace = {
        key: []
        for key in (
            "state_mask",
            "cached_score",
            "topq",
            "ranked_score",
            "response_q",
            "true_marginal",
            "margin",
            "choice",
            "regret",
            "true_best",
        )
    }
    observables = (
        {key: [] for key in ("screen_margin", "pred_margin", "pred_top1", "response_norm")}
        if collect_observables
        else None
    )
    for _ in range(budget):
        screen = score_state(
            screen_model,
            batch,
            selected,
            device=device,
            pack_fn=pack_selected,
        )
        topq = np.argsort(-screen, axis=1, kind="stable")[:, :q]
        qmask = np.zeros_like(selected, dtype=bool)
        qmask[np.arange(batch.episodes)[:, None], topq] = True

        responses = response_summary(batch, selected, expert, candidate_mask=qmask)
        response_q = np.take_along_axis(responses, topq[:, :, None], axis=1)

        def response_fn(batch_, selected_):
            return responses

        response_scores = score_state(
            response_model,
            batch,
            selected,
            device=device,
            pack_fn=pack_selected,
            response_fn=response_fn,
        )
        masked = np.where(qmask, response_scores, -np.inf)
        choice = np.argmax(masked, axis=1)
        best = masked[np.arange(batch.episodes), choice]
        true = observed_candidate_marginals(batch, selected, expert)
        margins = np.full(batch.episodes, np.nan, dtype=np.float64)
        true_best = np.zeros(batch.episodes, dtype=np.int64)
        regrets = np.full(batch.episodes, np.nan, dtype=np.float64)
        for row in range(batch.episodes):
            eligible = ~selected[row]
            values = true[row, eligible]
            if values.size == 0:
                continue
            order = np.argsort(-values, kind="stable")
            true_best[row] = int(np.flatnonzero(eligible)[order[0]])
            if values.size >= 2:
                margins[row] = values[order[0]] - values[order[1]]
            regrets[row] = values[order[0]] - true[row, choice[row]]
        if collect_observables:
            unmasked = np.where(selected, -np.inf, screen)
            top2 = np.sort(unmasked, axis=1)[:, -2:]
            observables["screen_margin"].append(top2[:, 1] - top2[:, 0])
            ranked = np.where(qmask, response_scores, -np.inf)
            ranked_top2 = np.sort(ranked, axis=1)[:, -2:]
            observables["pred_margin"].append(ranked_top2[:, 1] - ranked_top2[:, 0])
            observables["pred_top1"].append(np.where(np.isfinite(best), best, 0.0))
            # position of the chosen candidate inside the shortlist
            position = (choice[:, None] == topq).argmax(axis=1)
            picked = response_q[np.arange(batch.episodes), position]
            observables["response_norm"].append(np.linalg.norm(picked, axis=1))
        active = best > 0.0
        if return_diagnostics:
            trace["state_mask"].append(selected.copy())
            trace["cached_score"].append(screen.astype(np.float32))
            trace["topq"].append(topq.astype(np.int64))
            trace["ranked_score"].append(response_scores.astype(np.float32))
            trace["response_q"].append(response_q.astype(np.float32))
            trace["true_marginal"].append(true.astype(np.float32))
            trace["margin"].append(margins.astype(np.float32))
            trace["choice"].append(choice.astype(np.int64))
            trace["regret"].append(regrets.astype(np.float32))
            trace["true_best"].append(true_best.astype(np.int64))
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    if return_diagnostics:
        for key in trace:
            trace[key] = np.stack(trace[key], axis=0)
        return selected, trace
    if collect_observables:
        return selected, {key: np.stack(value, axis=1) for key, value in observables.items()}
    return selected


def select_oneshot_response(
    screen_model,
    response_model,
    batch,
    expert,
    *,
    budget: int,
    device: str,
):
    """Response-aware ranking computed once, with no sequential update.

    Every candidate is scored at the empty state with the same response-aware
    model and the same response summary that the sequential router uses, and the
    resulting ranking is applied as it stands: the selected set is the prefix of
    that ranking whose scores are positive, up to the budget. The only
    difference from the sequential router is that the scores are never
    recomputed after a candidate is accepted, so the comparison isolates the
    value of sequential recomputation from the value of the response features.
    """

    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    full_mask = np.ones((batch.episodes, batch.candidate_count), dtype=bool)
    responses = response_summary(batch, selected, expert, candidate_mask=full_mask)

    def response_fn(batch_, selected_):
        return responses

    scores = score_state(
        response_model,
        batch,
        selected,
        device=device,
        pack_fn=pack_selected,
        response_fn=response_fn,
    )
    order = np.argsort(-scores, axis=1, kind="stable")
    chosen = np.zeros_like(selected)
    rows = np.arange(batch.episodes)
    for rank in range(min(budget, batch.candidate_count)):
        index = order[:, rank]
        chosen[rows, index] = scores[rows, index] > 0.0
    return chosen


def build_full_state_data(
    batch,
    expert,
    *,
    max_budget: int,
    states_per_episode: int,
    seed: int,
):
    """Build state-level tensors for Huber plus pairwise ranking."""

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
        selected, selected_valid = pack_selected(batch, selected_mask, max_budget=max_budget)
        marginal = observed_candidate_marginals(batch, selected_mask, expert)
        responses = response_summary(batch, selected_mask, expert)
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


def build_hard_negative_state_data(
    batch,
    expert,
    cached_model,
    *,
    max_budget: int,
    states_per_episode: int,
    q: int,
    seed: int,
    device: str,
):
    """Build state-level data whose candidate lists match inference screening."""

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
        selected, selected_valid = pack_selected(batch, selected_mask, max_budget=max_budget)
        scores = score_state(
            cached_model,
            batch,
            selected_mask,
            device=device,
            pack_fn=pack_selected,
        )
        topq = np.argsort(-scores, axis=1, kind="stable")[:, :q]
        qmask = np.zeros_like(selected_mask, dtype=bool)
        qmask[np.arange(batch.episodes)[:, None], topq] = True
        marginal = observed_candidate_marginals(batch, selected_mask, expert)
        responses = response_summary(batch, selected_mask, expert, candidate_mask=qmask)
        candidate = np.take_along_axis(batch.candidate_embedding, topq[:, :, None], axis=1)
        response_q = np.take_along_axis(responses, topq[:, :, None], axis=1)
        target_q = np.take_along_axis(marginal, topq, axis=1)
        query_rows.append(batch.query_embedding)
        selected_rows.append(selected)
        valid_rows.append(selected_valid.astype(bool))
        candidate_rows.append(candidate)
        response_rows.append(response_q)
        target_rows.append(target_q.astype(np.float32))
    return {
        "query": np.concatenate(query_rows, axis=0),
        "selected": np.concatenate(selected_rows, axis=0),
        "selected_mask": np.concatenate(valid_rows, axis=0),
        "candidate": np.concatenate(candidate_rows, axis=0),
        "response": np.concatenate(response_rows, axis=0),
        "target": np.concatenate(target_rows, axis=0),
    }


def train_ranked_response_model(
    data: dict[str, np.ndarray],
    *,
    max_budget: int,
    seed: int,
    device: str,
    epochs: int = 20,
    batch_size: int = 128,
    beta: float = 1.0,
):
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    candidate_count = int(data["candidate"].shape[1])
    response_dim = int(data["response"].shape[-1])
    config = RouterConfig(
        embedding_dim=int(data["query"].shape[1]),
        hidden_dim=64,
        set_dim=64,
        depth=2,
        max_budget=max_budget,
        response_dim=response_dim,
    )
    model = MarginalUtilityRouter(config).to(device)
    tensors = [
        torch.from_numpy(data["query"]).float(),
        torch.from_numpy(data["selected"]).float(),
        torch.from_numpy(data["selected_mask"]).bool(),
        torch.from_numpy(data["candidate"]).float(),
        torch.from_numpy(data["response"]).float(),
        torch.from_numpy(data["target"]).float(),
    ]
    dataset = TensorDataset(*tensors)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for query, selected, selected_mask, candidate, response, target in loader:
            states = query.shape[0]
            k = candidate.shape[1]
            query_flat = query.to(device)[:, None, :].expand(states, k, query.shape[1]).reshape(states * k, -1)
            selected_flat = (
                selected.to(device)[:, None, :, :]
                .expand(states, k, selected.shape[1], selected.shape[2])
                .reshape(states * k, selected.shape[1], selected.shape[2])
            )
            mask_flat = (
                selected_mask.to(device)[:, None, :]
                .expand(states, k, selected_mask.shape[1])
                .reshape(states * k, selected_mask.shape[1])
            )
            candidate_flat = candidate.to(device).reshape(states * k, -1)
            response_flat = response.to(device).reshape(states * k, -1)
            cost_flat = torch.ones((states * k, 1), device=device)
            prediction = model(
                query_flat,
                selected_flat,
                mask_flat,
                candidate_flat,
                None,
                cost_flat,
                response_flat,
            ).reshape(states, k)

            target_dev = target.to(device)
            valid = torch.isfinite(target_dev)
            huber = F_ops.smooth_l1_loss(prediction[valid], target_dev[valid])
            gap = target_dev[:, :, None] - target_dev[:, None, :]
            pair_mask = (
                torch.isfinite(target_dev)[:, :, None]
                & torch.isfinite(target_dev)[:, None, :]
                & (gap > 0.0)
            )
            if torch.any(pair_mask):
                prediction_gap = prediction[:, :, None] - prediction[:, None, :]
                weights = gap[pair_mask]
                ranking = (weights * F_ops.softplus(-prediction_gap[pair_mask])).sum() / weights.sum().clamp_min(1e-8)
                loss = huber + beta * ranking
            else:
                loss = huber
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def evaluate_seed(
    seed,
    cfg,
    batches,
    expert,
    *,
    q_values,
    epochs,
    beta,
    hard_negative_q,
    device,
    save_diagnostics: bool = False,
    extra_split: str = "none",
    collect_observables: bool = False,
):
    train, calibration, test = batches
    budget = int(cfg["budget"])
    marginal_fn = lambda batch, selected: observed_candidate_marginals(batch, selected, expert)
    static_pairs = sample_pair_dataset(
        train,
        max_budget=budget,
        states_per_episode=4,
        seed=seed + 4_000,
        static_only=True,
        marginal_fn=marginal_fn,
        pack_fn=pack_selected,
    )
    cached_pairs = sample_pair_dataset(
        train,
        max_budget=budget,
        states_per_episode=4,
        seed=seed + 5_000,
        static_only=False,
        marginal_fn=marginal_fn,
        pack_fn=pack_selected,
    )
    static_model = train_utility_model(
        static_pairs, max_budget=budget, seed=seed + 6_000, device=device, epochs=epochs
    )
    cached_model = train_utility_model(
        cached_pairs, max_budget=budget, seed=seed + 7_000, device=device, epochs=epochs
    )
    if hard_negative_q > 0:
        full_data = build_hard_negative_state_data(
            train,
            expert,
            cached_model,
            max_budget=budget,
            states_per_episode=4,
            q=min(int(hard_negative_q), budget + 1),
            seed=seed + 9_000,
            device=device,
        )
    else:
        full_data = build_full_state_data(
            train,
            expert,
            max_budget=budget,
            states_per_episode=4,
            seed=seed + 9_000,
        )
    ranked_model = train_ranked_response_model(
        full_data,
        max_budget=budget,
        seed=seed + 10_000,
        device=device,
        epochs=epochs,
        beta=beta,
    )

    candidate_corr = np.asarray(cfg["candidate_sensor_correlations"], dtype=np.float32)
    relevance = np.broadcast_to(candidate_corr[None, :], (test.episodes, test.candidate_count)).copy()
    oracle_static = select_oracle_static(test, expert, budget)
    oracle_greedy = select_oracle_greedy(test, expert, budget)

    selections = {
        "base_only": np.zeros((test.episodes, test.candidate_count), dtype=bool),
        "pool_all": np.ones((test.episodes, test.candidate_count), dtype=bool),
        "relevance": select_relevance(relevance, budget),
        "mmr": select_mmr(test, relevance, budget),
        "static_utility": select_static(static_model, test, budget=budget, device=device, pack_fn=pack_selected),
        "cached_mur": select_mur(cached_model, test, budget=budget, device=device, pack_fn=pack_selected),
        "oneshot_response": select_oneshot_response(
            cached_model, ranked_model, test, expert, budget=budget, device=device
        ),
        "oracle_static": oracle_static,
        "oracle_greedy": oracle_greedy,
    }
    diagnostic_trace = None
    diagnostic_label = None
    for q in q_values:
        label = "ranked_response_full" if q >= test.candidate_count else f"ranked_response_q{q}"
        if save_diagnostics and q == 4:
            selections[label], diagnostic_trace = select_ranked_response_mur(
                cached_model,
                ranked_model,
                test,
                expert,
                budget=budget,
                q=q,
                device=device,
                return_diagnostics=True,
            )
            diagnostic_label = label
        else:
            selections[label] = select_ranked_response_mur(
                cached_model,
                ranked_model,
                test,
                expert,
                budget=budget,
                q=q,
                device=device,
            )

    base_loss = squared_error(test, np.zeros_like(oracle_greedy), expert)
    gains = {name: base_loss - squared_error(test, selected, expert) for name, selected in selections.items()}
    oracle_loss = squared_error(test, oracle_greedy, expert)
    oracle_gain = base_loss - oracle_loss
    positive = oracle_gain > 0.0
    metrics = {}
    for name, selected in selections.items():
        routed = squared_error(test, selected, expert)
        gain = gains[name]
        metrics[name] = {
            "mean_prediction_gain": float(np.mean(gain)),
            "negative_transfer_rate": float(np.mean(routed > base_loss)),
            "mean_utility_recovery": float(np.mean(gain[positive] / oracle_gain[positive]))
            if np.any(positive)
            else float("nan"),
            "mean_selected_contexts": float(np.mean(np.sum(selected, axis=1))),
        }
    headroom = metrics["oracle_greedy"]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]
    # Optional second split. Only the two policies the deployment decision needs
    # are scored (the router and pool-everything), so nothing about the existing
    # test-split outputs or the diagnostic trace changes.
    validation_metrics = None
    if extra_split == "validation":
        val_pool = np.ones((calibration.episodes, calibration.candidate_count), dtype=bool)
        val_base = squared_error(calibration, np.zeros_like(val_pool), expert)
        validation_metrics = {
            "pool_all": {
                "mean_prediction_gain": float(np.mean(val_base - squared_error(calibration, val_pool, expert)))
            }
        }
        for q in q_values:
            label = (
                "ranked_response_full"
                if q >= calibration.candidate_count
                else f"ranked_response_q{q}"
            )
            selected = select_ranked_response_mur(
                cached_model, ranked_model, calibration, expert, budget=budget, q=q, device=device
            )
            validation_metrics[label] = {
                "mean_prediction_gain": float(
                    np.mean(val_base - squared_error(calibration, selected, expert))
                )
            }
    result = {
        "run_id": "R077b_traffic_ranked_response_router",
        "status": "gate",
        "config": {
            "seed": seed,
            "candidate_count": int(test.candidate_count),
            "budget": budget,
            "response_dim": int(full_data["response"].shape[-1]),
            "q_values": q_values,
            "router_epochs": epochs,
            "ranking_beta": beta,
            "hard_negative_q": hard_negative_q,
            "device": device,
        },
        "metrics": metrics,
        "metrics_validation": validation_metrics,
        "gate": {
            name: {
                "gain_over_zero": paired_ci(gains[name]),
                "gain_over_static": paired_ci(gains[name] - gains["static_utility"]),
                "headroom_recovery": float(
                    (metrics[name]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]) / headroom
                ) if headroom > 0.0 else float("nan"),
            }
            for name in selections
            if name.startswith("ranked_response")
        },
    }
    if save_diagnostics and diagnostic_trace is not None:
        diagnostics = dict(diagnostic_trace)
        diagnostics["static_selected"] = selections["static_utility"].astype(np.uint8)
        diagnostics["cached_selected"] = selections["cached_mur"].astype(np.uint8)
        diagnostics["ranked_selected"] = selections[diagnostic_label].astype(np.uint8)
        diagnostics["oracle_static_selected"] = oracle_static.astype(np.uint8)
        diagnostics["oracle_greedy_selected"] = oracle_greedy.astype(np.uint8)
        diagnostics["base_loss"] = base_loss.astype(np.float32)
        if collect_observables:
            # Every gate signal is already in the trace, so it is derived rather
            # than collected in a second pass over the expert.
            order = np.arange(diagnostic_trace["choice"].shape[1])
            screen_margin, pred_margin, pred_top1, response_norm = [], [], [], []
            for step in range(diagnostic_trace["choice"].shape[0]):
                state = diagnostic_trace["state_mask"][step]
                screen = np.where(state, -np.inf, diagnostic_trace["cached_score"][step])
                top2 = np.sort(screen, axis=1)[:, -2:]
                screen_margin.append(top2[:, 1] - top2[:, 0])
                shortlist = diagnostic_trace["topq"][step]
                ranked = np.take_along_axis(diagnostic_trace["ranked_score"][step], shortlist, axis=1)
                ordered = np.sort(ranked, axis=1)
                pred_margin.append(ordered[:, -1] - ordered[:, -2])
                pred_top1.append(ordered[:, -1])
                picked = (diagnostic_trace["choice"][step][:, None] == shortlist).argmax(axis=1)
                response_norm.append(
                    np.linalg.norm(diagnostic_trace["response_q"][step][order, picked], axis=1)
                )
            # stored flat: np.savez cannot nest a dict
            diagnostics["gate_screen_margin"] = np.stack(screen_margin, axis=1).astype(np.float32)
            diagnostics["gate_pred_margin"] = np.stack(pred_margin, axis=1).astype(np.float32)
            diagnostics["gate_pred_top1"] = np.stack(pred_top1, axis=1).astype(np.float32)
            diagnostics["gate_response_norm"] = np.stack(response_norm, axis=1).astype(np.float32)
        for name, selected in selections.items():
            diagnostics[f"{name}_loss"] = squared_error(test, selected, expert).astype(np.float32)
        return result, gains, diagnostics
    return result, gains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--q-values", default="2,4,8,16")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--hard-negative-q", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seeds = parse_ints(args.seeds)
    q_values = parse_ints(args.q_values)
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    records = []
    all_gains = []
    for seed in seeds:
        started = time.time()
        cfg, batches, _ = load_protocol(args.data_path, args.gate_root / f"seed{seed}" / "result.json", seed)
        expert = fit_subset_expert(
            batches[0],
            seed=seed + 3_000,
            repeats=int(cfg["expert_repeats"]),
            ridge_penalty=float(cfg["ridge_penalty"]),
        )
        result, gains = evaluate_seed(
            seed,
            cfg,
            batches,
            expert,
            q_values=q_values,
            epochs=args.epochs,
            beta=args.beta,
            hard_negative_q=args.hard_negative_q,
            device=args.device,
        )
        result["wall_seconds"] = time.time() - started
        records.append(result)
        all_gains.append({name: array for name, array in gains.items()})
        seed_dir = args.out_root / f"seed{seed}"
        seed_dir.mkdir()
        (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(seed_dir / "episode_gains.npz", **{name: np.asarray(values) for name, values in gains.items()})
        print(json.dumps({"seed": seed, "ranked_q4": result["metrics"].get("ranked_response_q4"), "ranked_full": result["metrics"].get("ranked_response_full")}, indent=2))

    metric_names = ["mean_prediction_gain", "negative_transfer_rate", "mean_utility_recovery", "mean_selected_contexts"]
    policy_names = list(records[0]["metrics"].keys())
    rows = ["policy," + ",".join(f"{m}_mean,{m}_std" for m in metric_names)]
    for policy in policy_names:
        row = [policy]
        for metric in metric_names:
            values = np.asarray([r["metrics"][policy][metric] for r in records], dtype=float)
            row.extend([f"{np.nanmean(values):.8f}", f"{np.nanstd(values, ddof=1):.8f}"])
        rows.append(",".join(row))
    (args.out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    gate_summary = {}
    for policy in policy_names:
        if not policy.startswith("ranked_response"):
            continue
        contrasts_zero = [r["gate"][policy]["gain_over_zero"]["mean"] for r in records]
        contrasts_static = [r["gate"][policy]["gain_over_static"]["mean"] for r in records]
        gate_summary[policy] = {
            "gain_over_zero": paired_ci(np.asarray(contrasts_zero)),
            "gain_over_static": paired_ci(np.asarray(contrasts_static)),
            "headroom_recovery_mean": float(np.mean([r["gate"][policy]["headroom_recovery"] for r in records])),
        }
    aggregate = {
        "protocol": {
            "seeds": seeds,
            "q_values": q_values,
            "epochs": args.epochs,
            "beta": args.beta,
            "data_path": str(args.data_path),
            "gate_root": str(args.gate_root),
        },
        "records": records,
        "gate_summary": gate_summary,
    }
    (args.out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "gate_summary": gate_summary}, indent=2))


if __name__ == "__main__":
    main()
