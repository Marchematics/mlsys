#!/usr/bin/env python3
"""R-MUR gate: compact response-aware reranking on METR-LA.

Stage 1 uses the cached MUR to screen a small candidate subset. Stage 2
recomputes the frozen expert response for that subset and reranks with the
response-aware utility model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mur.synthetic_experiment import (  # noqa: E402
    predict_pair_dataset,
    sample_pair_dataset,
    score_state,
    select_mur,
    select_static,
    train_utility_model,
)
from mur.traffic import (  # noqa: E402
    expert_prediction,
    fit_subset_expert,
    observed_candidate_marginals,
    pack_selected,
    squared_error,
)
from run_traffic_full_router import (  # noqa: E402
    load_protocol,
    select_mmr,
    select_oracle_greedy,
    select_oracle_static,
    select_relevance,
    summarize,
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_ints_with_full(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def response_summary(
    batch,
    selected: np.ndarray,
    expert,
    *,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compact frozen-expert response summaries for each candidate.

    The features are the level of the base forecast, the level after adding the
    candidate, and the mean and absolute-mean response change.
    """

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (batch.episodes, batch.candidate_count):
        raise ValueError("selected mask has the wrong shape")
    if candidate_mask is None:
        candidate_mask = np.ones_like(mask, dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != mask.shape:
            raise ValueError("candidate_mask has the wrong shape")
    base = expert_prediction(batch, mask, expert)
    output = np.zeros((batch.episodes, batch.candidate_count, 4), dtype=np.float32)
    for candidate in range(batch.candidate_count):
        eligible = (~mask[:, candidate]) & candidate_mask[:, candidate]
        if not np.any(eligible):
            continue
        augmented = mask.copy()
        augmented[eligible, candidate] = True
        after = expert_prediction(batch, augmented, expert)
        delta = after - base
        output[eligible, candidate, 0] = np.mean(base[eligible], axis=1)
        output[eligible, candidate, 1] = np.mean(after[eligible], axis=1)
        output[eligible, candidate, 2] = np.mean(delta[eligible], axis=1)
        output[eligible, candidate, 3] = np.mean(np.abs(delta[eligible]), axis=1)
    return output


def select_response_mur(
    screen_model,
    response_model,
    batch,
    expert,
    *,
    budget: int,
    q: int,
    device: str,
) -> np.ndarray:
    """Two-stage response-aware greedy selection."""

    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    candidate_count = batch.candidate_count
    q = max(1, min(int(q), candidate_count))
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

        def response_fn(batch_, selected_):
            return response_summary(batch_, selected_, expert, candidate_mask=qmask)

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
        active = best > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def paired_ci(values: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    from scipy.stats import t as student_t

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return {"mean": float(np.mean(values)) if n else float("nan"), "low": float("nan"), "high": float("nan")}
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    crit = float(student_t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    return {"mean": mean, "low": mean - crit * se, "high": mean + crit * se}


def evaluate(
    seed: int,
    cfg: dict,
    batches,
    expert,
    *,
    q_values: list[int],
    epochs: int,
    device: str,
    extra_split: str = "none",
) -> dict:
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
    response_pairs = sample_pair_dataset(
        train,
        max_budget=budget,
        states_per_episode=4,
        seed=seed + 5_500,
        static_only=False,
        marginal_fn=marginal_fn,
        pack_fn=pack_selected,
        response_fn=lambda batch, selected: response_summary(batch, selected, expert),
    )
    static_model = train_utility_model(
        static_pairs, max_budget=budget, seed=seed + 6_000, device=device, epochs=epochs
    )
    cached_model = train_utility_model(
        cached_pairs, max_budget=budget, seed=seed + 7_000, device=device, epochs=epochs
    )
    response_model = train_utility_model(
        response_pairs, max_budget=budget, seed=seed + 8_000, device=device, epochs=epochs
    )

    candidate_corr = np.asarray(cfg["candidate_sensor_correlations"], dtype=np.float32)

    def select_and_score(batch) -> dict:
        """Select and score every policy on one split.

        Factored out so that the same trained models can be scored on a second
        split, which is what the deployment-pilot study needs: the pilot verdict
        must be computed on data the deployment does not use.
        """

        relevance = np.broadcast_to(
            candidate_corr[None, :], (batch.episodes, batch.candidate_count)
        ).copy()
        selections = {
            "base_only": np.zeros((batch.episodes, batch.candidate_count), dtype=bool),
            "pool_all": np.ones((batch.episodes, batch.candidate_count), dtype=bool),
            "relevance": select_relevance(relevance, budget),
            "mmr": select_mmr(batch, relevance, budget),
            "static_utility": select_static(
                static_model, batch, budget=budget, device=device, pack_fn=pack_selected
            ),
            "cached_mur": select_mur(
                cached_model, batch, budget=budget, device=device, pack_fn=pack_selected
            ),
            "oracle_static": select_oracle_static(batch, expert, budget),
            "oracle_greedy": select_oracle_greedy(batch, expert, budget),
        }
        for q in q_values:
            label = "response_mur_full" if q >= batch.candidate_count else f"response_mur_q{q}"
            selections[label] = select_response_mur(
                cached_model, response_model, batch, expert, budget=budget, q=q, device=device,
            )
        base_loss = squared_error(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool), expert)
        oracle_gain = base_loss - squared_error(batch, selections["oracle_greedy"], expert)
        positive = oracle_gain > 0.0
        out = {}
        for name, selected in selections.items():
            routed = squared_error(batch, selected, expert)
            gain = base_loss - routed
            out[name] = {
                "mean_prediction_gain": float(np.mean(gain)),
                "negative_transfer_rate": float(np.mean(routed > base_loss)),
                "mean_utility_recovery": float(np.mean(gain[positive] / oracle_gain[positive]))
                if np.any(positive)
                else float("nan"),
                "mean_selected_contexts": float(np.mean(np.sum(selected, axis=1))),
            }
        return out, {name: gain for name, gain in ((n, base_loss - squared_error(batch, sel, expert))
                                                   for n, sel in selections.items())}

    metrics, episode_gains = select_and_score(test)
    validation_metrics = select_and_score(calibration)[0] if extra_split == "validation" else None

    headroom = metrics["oracle_greedy"]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]
    headroom_recovery = {}
    for name in metrics:
        if name in {"oracle_static", "oracle_greedy"}:
            continue
        headroom_recovery[name] = (
            float((metrics[name]["mean_prediction_gain"] - metrics["oracle_static"]["mean_prediction_gain"]) / headroom)
            if headroom > 0.0
            else float("nan")
        )

    result = {
        "run_id": "R077_traffic_response_router",
        "status": "gate",
        "config": {
            "seed": seed,
            "candidate_count": int(test.candidate_count),
            "budget": budget,
            "history_length": int(test.history_length),
            "horizon": int(test.horizon),
            "response_dim": 4,
            "q_values": q_values,
            "router_epochs": epochs,
            "device": device,
        },
        "metrics": metrics,
        "headroom_recovery": headroom_recovery,
        "gate": {
            name: {
                "gain_over_zero": paired_ci(episode_gains[name]),
                "gain_over_static": paired_ci(episode_gains[name] - episode_gains["static_utility"]),
                "headroom_recovery": headroom_recovery.get(name, float("nan")),
            }
            for name in metrics
            if name.startswith("response_mur")
        },
        "metrics_validation": validation_metrics,
        "episode_gains": episode_gains,
        "wall_seconds": None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--q-values", default="4,8,16")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--extra-split", default="none", choices=["none", "validation"],
                        help="also score every policy on the calibration split; used by the "
                             "deployment-pilot study, which needs a verdict computed on data "
                             "the deployment does not use")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    q_values = parse_ints_with_full(args.q_values)
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)

    records = []
    episodes = {}
    for seed in seeds:
        started = time.time()
        cfg, batches, _ = load_protocol(args.data_path, args.gate_root / f"seed{seed}" / "result.json", seed)
        train = batches[0]
        expert = fit_subset_expert(
            train,
            seed=seed + 3_000,
            repeats=int(cfg["expert_repeats"]),
            ridge_penalty=float(cfg["ridge_penalty"]),
        )
        result = evaluate(
            seed,
            cfg,
            batches,
            expert,
            q_values=q_values,
            epochs=args.epochs,
            device=args.device,
            extra_split=args.extra_split,
        )
        result["wall_seconds"] = time.time() - started
        records.append(result)
        episodes[seed] = {name: gains.tolist() for name, gains in result.pop("episode_gains").items()}
        seed_dir = args.out_root / f"seed{seed}"
        seed_dir.mkdir()
        (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(seed_dir / "episode_gains.npz", **{name: np.asarray(values) for name, values in episodes[seed].items()})
        print(json.dumps({"seed": seed, "response_mur_q4": result["metrics"].get("response_mur_q4"), "response_mur_full": result["metrics"].get("response_mur_full")}, indent=2))

    metric_names = [
        "mean_prediction_gain",
        "negative_transfer_rate",
        "mean_utility_recovery",
        "mean_selected_contexts",
    ]
    summary_rows = ["policy," + ",".join(f"{m}_mean,{m}_std" for m in metric_names)]
    policy_names = list(records[0]["metrics"].keys())
    for policy in policy_names:
        row = [policy]
        for metric in metric_names:
            values = np.asarray([r["metrics"][policy][metric] for r in records], dtype=float)
            row.extend([f"{np.nanmean(values):.8f}", f"{np.nanstd(values, ddof=1):.8f}"])
        summary_rows.append(",".join(row))
    (args.out_root / "summary.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")

    # Aggregate paired gate statistics across seeds using the mean paired contrast.
    gate_summary = {}
    for policy in policy_names:
        if not policy.startswith("response_mur"):
            continue
        contrasts_zero = [r["gate"][policy]["gain_over_zero"]["mean"] for r in records]
        contrasts_static = [r["gate"][policy]["gain_over_static"]["mean"] for r in records]
        gate_summary[policy] = {
            "gain_over_zero": paired_ci(np.asarray(contrasts_zero)),
            "gain_over_static": paired_ci(np.asarray(contrasts_static)),
            "headroom_recovery_mean": float(np.mean([r["headroom_recovery"][policy] for r in records])),
        }
    aggregate = {
        "protocol": {
            "seeds": seeds,
            "q_values": q_values,
            "data_path": str(args.data_path),
            "gate_root": str(args.gate_root),
            "epochs": args.epochs,
        },
        "records": records,
        "gate_summary": gate_summary,
    }
    (args.out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "gate_summary": gate_summary}, indent=2))


if __name__ == "__main__":
    main()
