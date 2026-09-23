#!/usr/bin/env python3
"""R115: the second family with a query-comparative prototype predictor.

R110/R112 kept everything fixed except the pool; R115 keeps the pool (R112's
redundancy-structured pool) and replaces the *predictor* with a learned-metric
attention-prototype classifier whose utility can only come from demonstration
images relative to the query (see ``prototype_predictor.py``).  The whole
evaluation stack is inherited: ``evaluate_cell`` and ``redundancy_diagnostics``
from the R112 driver, ``predictor_quality``/``relevance_of`` from the R110
driver, the frozen R-MUR code path from ``frozen_helpers``/``KBS_MUR``.

Pre-registered predictions (written before the sweep):

(i)   ``H_state >= 0.01`` (sequential oracle beats the standalone oracle by at
      least 1 % of the oracle gain) on the pooled analysis, because a second
      same-class demonstration refines the prototype while a near-duplicate adds
      little;
(ii)  forced-budget R-MUR q=4 beats Static Utility and Cached Utility with a
      paired 95 % CI lower bound above zero on >= 4 of 5 benchmarks and on the
      pooled analysis (the frozen thresholded rule is reported alongside);
(iii) the label shortcut is closed: predicting the realised marginal utility from
      the label multiset of the selected set alone must have substantially lower
      out-of-sample R^2 than the same probe given the predictor's response
      features (reported separately by ``diagnose_r115_shortcut.py``).
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

from demo_protocol import BENCHMARK_CONFIGS, load_protocol  # noqa: E402
from incontext_predictor import DemoExpertOps  # noqa: E402
from prototype_predictor import (  # noqa: E402
    load_prototype_checkpoint,
    save_prototype_checkpoint,
    train_prototype_predictor,
)
from redundant_pool import build_redundant_batch  # noqa: E402

# R112 / R110 helpers reused verbatim (no modification of those modules).
from run_r110_demo_selection import _git_head, _source_hashes, _write_manifest, parse_ints  # noqa: E402
from run_r112_redundant_demo import evaluate_cell  # noqa: E402


def _r115_hashes() -> dict:
    import hashlib

    files = {
        "PAMI_MUR/experiments/demo_selection/prototype_predictor.py": HERE / "prototype_predictor.py",
        "PAMI_MUR/experiments/demo_selection/run_r115_query_comparative.py": HERE / "run_r115_query_comparative.py",
        "PAMI_MUR/experiments/demo_selection/redundant_pool.py": HERE / "redundant_pool.py",
        "PAMI_MUR/experiments/demo_selection/forced_budget.py": HERE / "forced_budget.py",
        "PAMI_MUR/experiments/demo_selection/incontext_predictor.py": HERE / "incontext_predictor.py",
        "PAMI_MUR/experiments/demo_selection/demo_protocol.py": HERE / "demo_protocol.py",
    }
    return {label: hashlib.md5(path.read_bytes()).hexdigest() for label, path in files.items()}


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
    parser.add_argument("--copy-jitter", type=float, default=0.0)
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
        "pool_rule": (
            f"R112 redundancy-structured pool (kmeans 4x3 + 4 distractors, K=16, B=4, "
            f"copy_jitter={args.copy_jitter})"
        ),
        "predictor": "learned-metric attention prototype (query-comparative, no label-only path)",
        "git_head": _git_head(),
        "imported_source_hashes": {**_source_hashes(), **_r115_hashes()},
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
            predictor_epochs = args.predictor_epochs or config.predictor_epochs
            print(
                f"=== {name}: {protocol.features.num_classes} classes, "
                f"train episodes {config.train_episodes_per_class * protocol.features.num_classes}, "
                f"test episodes {config.test_episodes_per_class * protocol.features.num_classes}, "
                f"predictor=prototype ===",
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
                        copy_jitter=args.copy_jitter,
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
                        copy_jitter=args.copy_jitter,
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
                        pool_rule=manifest["pool_rule"],
                    )
                    result["run_id"] = "R115_query_comparative"
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
                                "base_ce": result["base_loss"],
                                "acc_4rel": result["predictor_quality"]["demos_4_relevance"]["mean_accuracy"],
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
