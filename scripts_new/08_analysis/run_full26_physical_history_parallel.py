# -*- coding: utf-8 -*-
"""Create all full-line physical-model versus measured-history figures."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_full26_accuracy_parallel import FORWARD_SECTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = Path(__file__).with_name("plot_physical_history_all_runs.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "data_processed_step2_v3_all_curve_quality_traceability"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "analysis" / "physical_history_traceability_full26_figures"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    parts_dir = output_dir / "_parts"
    logs_dir = output_dir / "logs"
    parts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    groups = [FORWARD_SECTIONS[index :: args.workers] for index in range(args.workers)]
    environment = os.environ.copy()
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    processes = []
    for index, sections in enumerate(groups, start=1):
        part_output = parts_dir / f"part_{index:02d}"
        log_path = logs_dir / f"part_{index:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--data-dir", str(data_dir),
            "--output-dir", str(part_output),
            "--sections", *sections,
            "--excel-engine", "calamine",
            "--format", "png",
            "--dpi", str(args.dpi),
            "--clean-output",
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((index, process, log_handle, log_path, part_output))
        print(f"Started part {index}: {len(sections)} sections, pid={process.pid}", flush=True)

    pending = {item[0] for item in processes}
    while pending:
        time.sleep(10)
        for index, process, _, log_path, _ in processes:
            if index in pending and process.poll() is not None:
                pending.remove(index)
                print(f"Finished part {index}: exit={process.returncode}, log={log_path}", flush=True)

    failed = []
    for index, process, log_handle, log_path, _ in processes:
        log_handle.close()
        if process.returncode != 0:
            failed.append((index, process.returncode, log_path))
    if failed:
        for index, code, log_path in failed:
            print(f"FAILED part {index}: exit={code}, log={log_path}")
        return 1

    figures_dir = output_dir / "figures"
    summaries = []
    copied = 0
    for _, _, _, _, part_output in processes:
        summary_path = part_output / "physical_history_trip_summary.csv"
        summaries.append(pd.read_csv(summary_path))
        for source in part_output.glob("*/*.png"):
            destination_dir = figures_dir / source.parent.name
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination_dir / source.name)
            copied += 1

    details = pd.concat(summaries, ignore_index=True)
    order = {section: index for index, section in enumerate(FORWARD_SECTIONS)}
    details["_order"] = details["section"].map(order)
    details = details.sort_values(["_order", "segment"]).drop(columns="_order")
    details.to_csv(
        output_dir / "physical_history_trip_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    if copied != len(details):
        raise RuntimeError(f"Expected {len(details)} figures, collected {copied}")
    print(f"Completed {copied} figures across {details['section'].nunique()} sections.", flush=True)
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
