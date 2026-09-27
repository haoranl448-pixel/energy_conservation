# -*- coding: utf-8 -*-
"""Plot wheel-side physical mechanical energy against measured traction electricity."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_triple_axis_all_runs as common


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "data_processed_step2_v3_all_curve_quality_traceability"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-scope", default="full")
    parser.add_argument("--sections", nargs="*")
    parser.add_argument("--segment-ids", nargs="+")
    parser.add_argument("--excel-engine", choices=["auto", "calamine", "openpyxl"], default="auto")
    parser.add_argument("--format", choices=["png", "jpg"], default="png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def select_sections(args: argparse.Namespace) -> list[str]:
    if args.sections:
        return args.sections
    raw = str(args.line_scope).strip().lower()
    if raw in {"full", "all"}:
        return common.FORWARD_SECTIONS
    count = int(raw)
    return common.FORWARD_SECTIONS[:count]


def selected_segment(value: object, wanted: set[str]) -> bool:
    text = str(value).strip()
    if text in wanted:
        return True
    try:
        normalized = str(int(float(value))) if float(value).is_integer() else str(float(value))
    except (TypeError, ValueError):
        return False
    return normalized in wanted


def first_value(frame: pd.DataFrame, column: str) -> object:
    return frame[column].iloc[0] if column in frame.columns and len(frame) else ""


def plot_trip(
    section: str,
    segment_id: object,
    trip: pd.DataFrame,
    arrays: dict[str, np.ndarray | float],
    physical_step_wh: np.ndarray,
    output_path: Path,
    dpi: int,
) -> dict[str, object]:
    time_s = np.asarray(arrays["time"], dtype=float)
    speed_kmh = np.asarray(arrays["velocity_kmh"], dtype=float)
    distance_m = np.asarray(arrays["distance"], dtype=float)
    measured_step_wh = np.asarray(arrays["energy_real_step"], dtype=float)
    length = min(len(time_s), len(speed_kmh), len(distance_m), len(measured_step_wh), len(physical_step_wh))
    time_s = time_s[:length]
    speed_kmh = speed_kmh[:length]
    distance_m = distance_m[:length]
    measured_step_wh = np.maximum(measured_step_wh[:length], 0.0)
    physical_step_wh = np.maximum(np.asarray(physical_step_wh[:length], dtype=float), 0.0)
    measured_cumulative_wh = np.cumsum(measured_step_wh)
    physical_cumulative_wh = np.cumsum(physical_step_wh)

    measured_total_wh = float(measured_cumulative_wh[-1]) if length else 0.0
    physical_total_wh = float(physical_cumulative_wh[-1]) if length else 0.0
    difference_wh = physical_total_wh - measured_total_wh
    difference_pct = difference_wh / measured_total_wh * 100.0 if abs(measured_total_wh) > 1e-9 else np.nan
    cumulative_r2 = common.r_squared(measured_cumulative_wh, physical_cumulative_wh)

    time_speed, speed_plot = common.anchor_origin(time_s, speed_kmh)
    time_distance, distance_plot = common.anchor_origin(time_s, distance_m)
    time_measured, measured_plot = common.anchor_origin(time_s, measured_cumulative_wh)
    time_physical, physical_plot = common.anchor_origin(time_s, physical_cumulative_wh)

    fig, ax_speed = plt.subplots(figsize=(13, 7))
    plt.subplots_adjust(right=0.84)
    speed_line = ax_speed.plot(time_speed, speed_plot, color="tab:blue", linewidth=1.6, label="历史运行速度 (v)")
    ax_speed.set_xlabel("时间 Time (s)")
    ax_speed.set_ylabel("速度 Velocity (km/h)", color="tab:blue")
    ax_speed.tick_params(axis="y", labelcolor="tab:blue")
    ax_speed.set_xlim(left=0)
    ax_speed.set_ylim(bottom=0)
    ax_speed.grid(True, alpha=0.22)

    ax_energy = ax_speed.twinx()
    measured_line = ax_energy.plot(
        time_measured,
        measured_plot,
        color="black",
        linewidth=2.0,
        label="历史实测牵引电能 (E_measured)",
    )
    physical_line = ax_energy.plot(
        time_physical,
        physical_plot,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label="新版物理模型机械能 (E_physical)",
    )
    ax_energy.set_ylabel("累计能量 Energy (Wh)", color="tab:red")
    ax_energy.tick_params(axis="y", labelcolor="tab:red")
    ax_energy.set_ylim(bottom=0)

    ax_distance = ax_speed.twinx()
    ax_distance.spines["right"].set_position(("outward", 62))
    distance_line = ax_distance.plot(
        time_distance,
        distance_plot,
        color="tab:green",
        linewidth=1.6,
        label="累计位移 (s)",
    )
    ax_distance.set_ylabel("累计位移 Distance (m)", color="tab:green")
    ax_distance.tick_params(axis="y", labelcolor="tab:green")
    ax_distance.set_ylim(bottom=0)

    lines = speed_line + measured_line + physical_line + distance_line
    ax_speed.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=9)
    r2_text = "N/A" if cumulative_r2 is None else f"{cumulative_r2:.4f}"
    difference_text = "N/A" if not np.isfinite(difference_pct) else f"{difference_pct:+.2f}%"
    date_service = first_value(trip, "日期+服务号")
    plt.title(
        f"新版物理模型与历史实测能耗对标图 - {section} (Segment {segment_id})\n"
        f"累计曲线 R² = {r2_text}  |  日期+服务号 {date_service}  |  机械能-实测电能差异 {difference_text}",
        fontsize=13,
    )
    plt.tight_layout()
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight", "facecolor": "white"}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

    return {
        "section": section,
        "segment": segment_id,
        "rows": length,
        "duration_s": round(float(time_s[-1]) if length else 0.0, 3),
        "distance_m": round(float(distance_m[-1]) if length else 0.0, 3),
        "measured_electric_energy_wh": round(measured_total_wh, 3),
        "physical_mechanical_energy_wh": round(physical_total_wh, 3),
        "mechanical_minus_electric_wh": round(difference_wh, 3),
        "mechanical_minus_electric_pct": "" if not np.isfinite(difference_pct) else round(difference_pct, 3),
        "cumulative_curve_r2": "" if cumulative_r2 is None else round(cumulative_r2, 6),
        "quality_label": first_value(trip, "曲线质量标签"),
        "run_id": first_value(trip, "来源run_id"),
        "date_service": date_service,
        "figure": str(output_path),
    }


def main() -> int:
    args = parse_args()
    common.configure_plot_style()
    data_dir = common.resolve_path(args.data_dir)
    output_dir = common.resolve_path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = select_sections(args)
    sim_model = common.TrainTheoreticalEnergyModel()
    rows: list[dict[str, object]] = []
    wanted = {str(value).strip() for value in args.segment_ids or []}
    for section_index, section in enumerate(sections, start=1):
        print(f"[{section_index}/{len(sections)}] {section}", flush=True)
        frame = common.read_section_data(section, data_dir, args.excel_engine)
        if frame is None or frame.empty:
            print("  missing data", flush=True)
            continue
        section_output = output_dir / common.safe_name(section)
        section_output.mkdir(parents=True, exist_ok=True)
        segment_ids = list(pd.Series(frame["segment"]).dropna().unique())
        if wanted:
            segment_ids = [value for value in segment_ids if selected_segment(value, wanted)]
        print(f"  trips: {len(segment_ids)}", flush=True)
        for trip_index, segment_id in enumerate(segment_ids, start=1):
            trip = frame[frame["segment"] == segment_id].copy()
            arrays = common.prepare_trip_arrays(trip)
            physical_step_wh = sim_model.run_batch_simulation_fast(
                arrays["time"],
                arrays["velocity_mps"],
                float(arrays["mass_value"]),
                section_name=section,
            )
            date_service = first_value(trip, "日期+服务号")
            segment_label = f"seg_{int(segment_id):03d}" if float(segment_id).is_integer() else f"seg_{common.safe_name(segment_id)}"
            output_path = section_output / f"{segment_label}_{common.safe_name(date_service)}.{args.format}"
            rows.append(plot_trip(section, segment_id, trip, arrays, physical_step_wh, output_path, args.dpi))
            if trip_index % 10 == 0 or trip_index == len(segment_ids):
                print(f"  completed {trip_index}/{len(segment_ids)}", flush=True)

    summary_path = output_dir / "physical_history_trip_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"completed figures: {len(rows)}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
