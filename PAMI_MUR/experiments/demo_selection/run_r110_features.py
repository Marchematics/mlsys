#!/usr/bin/env python3
"""R110 step 1: cache frozen-CLIP features for every benchmark.

Run once per benchmark set; later stages read only the ``.npy`` caches.

    python run_r110_features.py --benchmarks cifar10,cifar100,svhn,eurosat,dtd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vision_data import BENCHMARK_ORDER, extract_benchmark  # noqa: E402

PAMI = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_ORDER))
    parser.add_argument("--raw-root", type=Path, default=PAMI / "data/vision_raw")
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="disable fp16 autocast for the encoder")
    args = parser.parse_args()

    names = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.feature_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary = {}
    for name in names:
        print(f"=== {name} ===", flush=True)
        record = extract_benchmark(
            name,
            raw_root=args.raw_root,
            feature_root=args.feature_root,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            delete_raw=not args.keep_raw,
            amp=not args.no_amp,
        )
        summary[name] = {
            "train_size": record.get("train_size"),
            "test_size": record.get("test_size"),
            "num_classes": record.get("num_classes"),
            "train_seconds": record.get("train_seconds"),
            "test_seconds": record.get("test_seconds"),
        }
        print(json.dumps({name: summary[name]}, indent=2), flush=True)
    print(f"total seconds: {time.time() - started:.1f}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
