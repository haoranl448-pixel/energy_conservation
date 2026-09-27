# -*- coding: utf-8 -*-
"""Run full-line neural-network accuracy validation in parallel partitions."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = Path(__file__).with_name("plot_triple_axis_all_runs.py")

FORWARD_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰", "庙堰-钟公庙",
    "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区", "南高教园区-下应路",
    "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘", "柳隘-海晏北路",
    "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路", "院士路-盎孟港",
    "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路", "兴海南路-梅堰", "梅堰-永茂路",
    "永茂路-镇海大道", "镇海大道-骆驼桥",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "data_processed_step2_v3_all_curve_quality_traceability"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "output" / "models" / "nn_results_residual_v2"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "analysis" / "nn_model_accuracy_traceability_full26"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--with-figures",
        action="store_true",
        help="Create one PNG for every section trip and collect them under figures/section/.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def aggregate(group: pd.DataFrame) -> pd.Series:
    actual = float(group["real_energy_wh"].sum())
    predicted = float(group["fusion_energy_wh"].sum())
    weighted_error_pct = (predicted - actual) / actual * 100.0 if abs(actual) > 1e-9 else np.nan
    abs_pct = group["fusion_abs_error_pct"].dropna()
    return pd.Series(
        {
            "trip_count": len(group),
            "measured_energy_total_kwh": actual / 1000.0,
            "predicted_energy_total_kwh": predicted / 1000.0,
            "weighted_total_error_pct": weighted_error_pct,
            "mean_signed_error_pct": group["fusion_error_pct"].mean(),
            "mean_absolute_error_pct": abs_pct.mean(),
            "median_absolute_error_pct": abs_pct.median(),
            "p90_absolute_error_pct": abs_pct.quantile(0.9),
            "mean_absolute_error_kwh": group["fusion_abs_error_wh"].mean() / 1000.0,
            "mean_cumulative_r2": group["cumulative_energy_r2"].mean(),
            "median_cumulative_r2": group["cumulative_energy_r2"].median(),
            "within_5pct_ratio": (abs_pct <= 5.0).mean(),
            "within_10pct_ratio": (abs_pct <= 10.0).mean(),
        }
    )


def write_summaries(details: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols = [
        "real_energy_wh", "fusion_energy_wh", "fusion_error_pct", "fusion_abs_error_wh",
        "fusion_abs_error_pct", "cumulative_energy_r2", "step_energy_mae_wh", "step_energy_rmse_wh",
    ]
    for col in numeric_cols:
        details[col] = pd.to_numeric(details[col], errors="coerce")

    details.to_csv(
        output_dir / "trip_accuracy_detail.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    details.groupby("section", sort=False, dropna=False).apply(aggregate).reset_index().to_csv(
        output_dir / "accuracy_summary_by_section.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    overall = aggregate(details).to_frame().T
    overall.insert(0, "scope", "all_sections")
    overall.to_csv(
        output_dir / "accuracy_summary_overall.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    details.assign(quality_label=details["quality_label"].fillna("missing")).groupby(
        "quality_label", sort=True, dropna=False
    ).apply(aggregate).reset_index().to_csv(
        output_dir / "accuracy_summary_by_quality.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    data_dir = resolve_path(args.data_dir)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    parts_dir = output_dir / "_parts"
    logs_dir = output_dir / "logs"
    parts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    groups = [FORWARD_SECTIONS[i:: args.workers] for i in range(args.workers)]
    processes: list[tuple[int, subprocess.Popen, object, Path]] = []
    env = os.environ.copy()
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
        }
    )

    for index, sections in enumerate(groups, start=1):
        part_out = parts_dir / f"part_{index:02d}"
        log_path = logs_dir / f"part_{index:02d}.log"
        log_file = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--data-dir", str(data_dir),
            "--model-dir", str(model_dir),
            "--output-dir", str(part_out),
            "--sections", *sections,
            "--excel-engine", "calamine",
            "--clean-output",
            "--format", "png",
            "--dpi", str(args.dpi),
        ]
        if not args.with_figures:
            command.append("--summary-only")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((index, process, log_file, log_path))
        print(f"Started part {index}: {len(sections)} sections, pid={process.pid}", flush=True)

    pending = {index for index, _, _, _ in processes}
    while pending:
        time.sleep(10)
        for index, process, _, log_path in processes:
            if index not in pending or process.poll() is None:
                continue
            pending.remove(index)
            print(f"Finished part {index}: exit={process.returncode}, log={log_path}", flush=True)

    failures = []
    for index, process, log_file, log_path in processes:
        log_file.close()
        if process.returncode != 0:
            failures.append((index, process.returncode, log_path))
    if failures:
        for item in failures:
            print(f"FAILED part {item[0]}: exit={item[1]}, log={item[2]}")
        return 1

    detail_frames = []
    skipped_frames = []
    for index in range(1, len(groups) + 1):
        part_out = parts_dir / f"part_{index:02d}"
        detail_frames.append(pd.read_csv(part_out / "triple_axis_plot_summary.csv"))
        skipped_path = part_out / "triple_axis_plot_skipped.csv"
        if skipped_path.exists():
            skipped_frames.append(pd.read_csv(skipped_path))

    details = pd.concat(detail_frames, ignore_index=True)
    section_order = {name: i for i, name in enumerate(FORWARD_SECTIONS)}
    details["_section_order"] = details["section"].map(section_order)
    details = details.sort_values(["_section_order", "segment"]).drop(columns="_section_order")
    write_summaries(details, output_dir)
    if args.with_figures:
        figures_dir = output_dir / "figures"
        copied = 0
        for index in range(1, len(groups) + 1):
            part_out = parts_dir / f"part_{index:02d}"
            for source in part_out.glob("*/*.png"):
                destination_dir = figures_dir / source.parent.name
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination_dir / source.name)
                copied += 1
        if copied != len(details):
            raise RuntimeError(f"Expected {len(details)} figures, collected {copied}")
        print(f"Collected {copied} figures under {figures_dir}", flush=True)
    if skipped_frames:
        pd.concat(skipped_frames, ignore_index=True).to_csv(
            output_dir / "validation_skipped.csv", index=False, encoding="utf-8-sig"
        )

    print(f"Completed {len(details)} trips across {details['section'].nunique()} sections.", flush=True)
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
