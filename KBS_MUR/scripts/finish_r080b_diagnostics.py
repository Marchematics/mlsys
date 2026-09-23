#!/usr/bin/env python3
"""Low-concurrency completion of the R080b diagnostic runs."""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_202 = ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921"
BASE_303 = ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921"
BASE_101 = ROOT / "results/raw/R080b_multitarget_rmur_s101_diag_metrla_20260921/METRLA"

TASKS = [
    (202, "PEMSBAY", BASE_202 / "PEMSBAY"),
    (202, "PEMS07", BASE_202 / "PEMS07"),
    (303, "PEMS03", BASE_303 / "PEMS03"),
    (303, "PEMSBAY", BASE_303 / "PEMSBAY"),
    (101, "METRLA", BASE_101),
]


def count_running() -> int:
    try:
        out = subprocess.check_output(["pgrep", "-fc", "run_response_router_multitarget.py"], text=True)
        return int(out.strip() or 0)
    except subprocess.CalledProcessError:
        return 0


def command_for(seed: int, dataset: str, out_root: Path):
    return [
        sys.executable,
        str(ROOT / "scripts/run_response_router_multitarget.py"),
        "--data-root",
        str(ROOT.parent / "data/staeformer"),
        "--dataset",
        dataset,
        "--out-root",
        str(out_root),
        "--seeds",
        str(seed),
        "--target-count",
        "32",
        "--q-values",
        "4",
        "--epochs",
        "20",
        "--beta",
        "1.0",
        "--device",
        "cuda:0",
    ]


def main() -> None:
    pending = deque(TASKS)
    attempts = {task: 0 for task in TASKS}
    running: dict[int, tuple] = {}
    log = ROOT / "refine-logs/R080b_diagnostic_completion.log"
    log.write_text("", encoding="utf-8")

    def write(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print(line, end="", flush=True)

    write(f"supervisor started; current processes={count_running()}")
    while True:
        # Wait until the currently running long jobs leave GPU headroom.
        if count_running() > 2:
            time.sleep(30)
            continue
        # Launch up to two pending tasks.
        while pending and len(running) < 2 and count_running() < 3:
            seed, dataset, out_root = pending.popleft()
            if out_root.exists():
                subprocess.run(["rm", "-rf", str(out_root)], check=False)
            out_root.parent.mkdir(parents=True, exist_ok=True)
            command = command_for(seed, dataset, out_root)
            process = subprocess.Popen(command, cwd=ROOT)
            running[process.pid] = (process, seed, dataset, out_root)
            attempts[(seed, dataset, out_root)] += 1
            write(f"launched seed={seed} dataset={dataset} pid={process.pid} attempt={attempts[(seed, dataset, out_root)]}")
        if not pending and not running:
            break
        status = subprocess.run(["sleep", "20"])
        for pid, (process, seed, dataset, out_root) in list(running.items()):
            if process.poll() is None:
                continue
            del running[pid]
            if process.returncode == 0:
                write(f"completed seed={seed} dataset={dataset}")
            else:
                write(f"failed seed={seed} dataset={dataset} code={process.returncode}")
                task = (seed, dataset, out_root)
                if attempts[task] < 2:
                    pending.append(task)
                    write(f"requeued seed={seed} dataset={dataset}")
    write("supervisor finished")


if __name__ == "__main__":
    main()
