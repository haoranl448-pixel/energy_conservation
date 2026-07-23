from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt

from plot_opentrack_tsvp_vs_history import (
    as_float,
    energy_error,
    read_history_line,
    read_tsvp,
    selected_sections,
)


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_TSVP_DIR = Path(r"D:\OutPut")
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_processed_step2_v3_all_curve_quality"
DEFAULT_ROUTE_MAP = (
    PROJECT_ROOT
    / "output"
    / "opentrack_route_map_newline"
    / "priority_dp_trip011_route_map.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "opentrack_three_way_compare"
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"
DEFAULT_TRACEABILITY_MANIFEST = Path(
    r"C:\Users\bit11\Desktop\数据测试代码\data_processed_step2_v3_all_curve_quality_traceability"
    r"\trip_traceability_manifest_v1.csv"
)
DEFAULT_PLAN_FILES = {
    11: PROJECT_ROOT / "output" / "schedule" / "final_plan_report_v2---11" / "Final_Planning_Comparison.csv",
    12: PROJECT_ROOT / "output" / "schedule" / "final_plan_report_v2" / "Final_Planning_Comparison.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each trip, draw three four-panel comparisons: OpenTrack planned vs "
            "OpenTrack history, OpenTrack planned vs real history, and OpenTrack history "
            "vs real history."
        )
    )
    parser.add_argument(
        "--trips",
        default="11,12",
        help="Comma-separated 1-based trip numbers. Default: 11,12.",
    )
    parser.add_argument("--tsvp-dir", default=str(DEFAULT_TSVP_DIR))
    parser.add_argument(
        "--planned-template",
        default="OT_priority_dp_trip{trip:03d}.tsvP",
        help="Filename template below --tsvp-dir.",
    )
    parser.add_argument(
        "--ot-history-template",
        default="OT_priority_history_trip{trip:03d}.tsvP",
        help="Filename template below --tsvp-dir.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--route-map", default=str(DEFAULT_ROUTE_MAP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trajectory-dir", default=str(DEFAULT_TRAJECTORY_DIR))
    parser.add_argument(
        "--trajectory-dir-for",
        action="append",
        default=[],
        metavar="TRIP=DIR",
        help=(
            "Generated-curve directory for one trip. Repeat for multiple trips, for example "
            "--trajectory-dir-for 6=path --trajectory-dir-for 12=path."
        ),
    )
    parser.add_argument(
        "--traceability-manifest",
        default=str(DEFAULT_TRACEABILITY_MANIFEST),
        help=(
            "Optional global trip traceability CSV. When present, real-history curves use "
            "the section-specific local segment belonging to the selected full-line trip."
        ),
    )
    parser.add_argument(
        "--ignore-traceability",
        action="store_true",
        help="Use the original section-local trip number instead of the optional traceability mapping.",
    )
    parser.add_argument(
        "--plan-file",
        action="append",
        default=[],
        metavar="TRIP=CSV",
        help=(
            "Priority-plan CSV for one trip. Repeat for multiple trips, for example "
            "--plan-file 11=path.csv --plan-file 12=path.csv."
        ),
    )
    parser.add_argument(
        "--history-dwell",
        type=float,
        default=30.0,
        help="Dwell seconds inserted between adjacent real-history sections. Default: 30.",
    )
    parser.add_argument("--section-start", type=int, default=1)
    parser.add_argument("--section-count", type=int, default=None)
    parser.add_argument("--sections", default=None)
    parser.add_argument("--quality-label", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def parse_trip_numbers(text: str) -> list[int]:
    trips: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        trip = int(item)
        if trip < 1:
            raise ValueError(f"Trip number must be at least 1: {trip}")
        if trip not in trips:
            trips.append(trip)
    if not trips:
        raise ValueError("No trip numbers supplied.")
    return trips


def parse_plan_files(values: list[str]) -> dict[int, Path]:
    result = dict(DEFAULT_PLAN_FILES)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --plan-file value, expected TRIP=CSV: {value}")
        trip_text, path_text = value.split("=", 1)
        trip = int(trip_text.strip())
        if trip < 1 or not path_text.strip():
            raise ValueError(f"Invalid --plan-file value: {value}")
        result[trip] = Path(path_text.strip())
    return result


def parse_trip_paths(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid TRIP=DIR value: {value}")
        trip_text, path_text = value.split("=", 1)
        trip = int(trip_text.strip())
        if trip < 1 or not path_text.strip():
            raise ValueError(f"Invalid TRIP=DIR value: {value}")
        result[trip] = Path(path_text.strip())
    return result


def read_trip_segment_map(path: Path, trip: int, sections: list[str]) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    prefix = f"UP{trip:03d}_"
    result: dict[str, int] = {}
    for row in rows:
        trip_id = (row.get("全局趟次候选ID") or "").strip()
        section = (row.get("区段") or "").strip()
        if not trip_id.startswith(prefix) or section not in sections:
            continue
        segment = as_float(row.get("segment"))
        if segment is None:
            continue
        if section in result and result[section] != int(segment):
            raise ValueError(f"Traceability manifest has duplicate segment mapping: trip={trip}, {section}")
        result[section] = int(segment)

    missing = [section for section in sections if section not in result]
    if missing:
        raise ValueError(
            f"Traceability manifest is incomplete for trip {trip}: missing {missing}"
        )
    return result


def read_history_profile_maps(
    plan_path: Path,
    sections: list[str],
    fallback_dwell: float,
) -> tuple[dict[str, float], dict[str, float]]:
    with plan_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    row_by_section = {(row.get("站间区间") or "").strip(): row for row in rows}
    dwell_result: dict[str, float] = {}
    run_time_result: dict[str, float] = {}
    for section in sections:
        row = row_by_section.get(section)
        run_time = as_float(row.get("历史用时(s)")) if row else None
        if run_time is None:
            raise ValueError(f"{plan_path.name}: missing 历史用时(s) for {section}")
        run_time_result[section] = run_time
    for section in sections[:-1]:
        row = row_by_section.get(section)
        value = as_float(row.get("历史停站时间(s)")) if row else None
        dwell_result[section] = fallback_dwell if value is None else value
    return dwell_result, run_time_result


def read_plan_baseline(
    plan_path: Path,
    trajectory_dir: Path,
    sections: list[str],
) -> dict[str, list[float]]:
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    with plan_path.open("r", encoding="utf-8-sig", newline="") as f:
        plan_rows = list(csv.DictReader(f))
    plan_index = {(row.get("站间区间") or "").strip(): row for row in plan_rows}

    full = {"time_s": [], "distance_km": [], "speed_kmh": [], "energy_kwh": []}
    time_offset = 0.0
    distance_offset = 0.0
    energy_offset = 0.0

    for section_index, section in enumerate(sections):
        row = plan_index.get(section)
        if row is None:
            raise ValueError(f"{plan_path.name}: missing planned section {section}")
        selected_class = (row.get("选定等级") or "").strip()
        planned_time_s = as_float(row.get("规划用时(s)"))
        dwell_s = as_float(row.get("停站时间(s)")) or 0.0
        section_energy_wh = as_float(row.get("规划能耗(Wh)"))
        if not selected_class or planned_time_s is None or section_energy_wh is None:
            raise ValueError(f"{plan_path.name}: incomplete plan row for {section}")

        curve_path = trajectory_dir / section / f"{selected_class}_generated_curve.csv"
        if not curve_path.exists():
            raise FileNotFoundError(curve_path)
        with curve_path.open("r", encoding="utf-8-sig", newline="") as f:
            curve_rows = list(csv.DictReader(f))

        local_time: list[float] = []
        local_distance_km: list[float] = []
        local_speed_kmh: list[float] = []
        for curve_row in curve_rows:
            t = as_float(curve_row.get("time_s"))
            s_m = as_float(curve_row.get("dist_m"))
            v_kmh = as_float(curve_row.get("velocity_kmh"))
            if v_kmh is None:
                v_ms = as_float(curve_row.get("velocity_mps"))
                v_kmh = None if v_ms is None else v_ms * 3.6
            if t is None or s_m is None or v_kmh is None:
                continue
            local_time.append(t)
            local_distance_km.append(s_m / 1000.0)
            local_speed_kmh.append(v_kmh)
        if len(local_time) < 2:
            raise ValueError(f"No usable planned trajectory rows: {curve_path}")

        t0 = local_time[0]
        s0 = local_distance_km[0]
        local_time = [value - t0 for value in local_time]
        local_distance_km = [value - s0 for value in local_distance_km]
        raw_duration = local_time[-1]
        section_distance = local_distance_km[-1]
        time_scale = planned_time_s / raw_duration if raw_duration > 0 else 1.0
        start_idx = 0 if not full["time_s"] else 1

        for idx in range(start_idx, len(local_time)):
            progress = local_distance_km[idx] / section_distance if section_distance > 0 else 0.0
            progress = min(max(progress, 0.0), 1.0)
            full["time_s"].append(time_offset + local_time[idx] * time_scale)
            full["distance_km"].append(distance_offset + local_distance_km[idx])
            full["speed_kmh"].append(local_speed_kmh[idx])
            full["energy_kwh"].append(energy_offset + section_energy_wh / 1000.0 * progress)

        time_offset += planned_time_s
        distance_offset += section_distance
        energy_offset += section_energy_wh / 1000.0
        if section_index < len(sections) - 1 and dwell_s > 0:
            full["time_s"].append(time_offset + dwell_s)
            full["distance_km"].append(distance_offset)
            full["speed_kmh"].append(0.0)
            full["energy_kwh"].append(energy_offset)
            time_offset += dwell_s

    return full


def source_summary(curve: dict[str, list[float]]) -> dict[str, float | int]:
    return {
        "points": len(curve["time_s"]),
        "duration_s": curve["time_s"][-1],
        "distance_km": curve["distance_km"][-1],
        "energy_kwh": curve["energy_kwh"][-1],
    }


def add_difference_box(
    ax: Any,
    primary_label: str,
    reference_label: str,
    diff_kwh: float | None,
    pct: float | None,
) -> None:
    if diff_kwh is None:
        text = "Energy difference: N/A"
    elif pct is None:
        text = f"Energy difference: N/A\n{primary_label} - {reference_label}: {diff_kwh:+.3f} kWh"
    else:
        text = (
            f"Energy difference: {pct:+.2f}%\n"
            f"{primary_label} - {reference_label}: {diff_kwh:+.3f} kWh"
        )
    ax.text(
        0.02,
        0.05,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#999999",
            "alpha": 0.92,
        },
    )


def plot_pair(
    primary: dict[str, list[float]],
    reference: dict[str, list[float]],
    primary_label: str,
    reference_label: str,
    title: str,
    output_path: Path,
    dpi: int,
    show: bool,
) -> tuple[float | None, float | None]:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["lines.antialiased"] = True

    diff_kwh, diff_pct = energy_error(primary, reference)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_vt, ax_vs, ax_et, ax_es = axes.ravel()

    ax_vt.plot(primary["time_s"], primary["speed_kmh"], color="#1f77b4", lw=2.0, label=primary_label)
    ax_vt.plot(
        reference["time_s"],
        reference["speed_kmh"],
        color="#111111",
        lw=1.8,
        ls="--",
        label=reference_label,
    )
    ax_vt.set_title("v-t")
    ax_vt.set_xlabel("Time (s)")
    ax_vt.set_ylabel("Speed (km/h)")

    ax_vs.plot(
        primary["distance_km"],
        primary["speed_kmh"],
        color="#1f77b4",
        lw=2.0,
        label=primary_label,
    )
    ax_vs.plot(
        reference["distance_km"],
        reference["speed_kmh"],
        color="#111111",
        lw=1.8,
        ls="--",
        label=reference_label,
    )
    ax_vs.set_title("v-s")
    ax_vs.set_xlabel("Distance (km)")
    ax_vs.set_ylabel("Speed (km/h)")

    ax_et.plot(primary["time_s"], primary["energy_kwh"], color="#d62728", lw=2.0, label=primary_label)
    ax_et.plot(
        reference["time_s"],
        reference["energy_kwh"],
        color="#111111",
        lw=1.8,
        ls="--",
        label=reference_label,
    )
    ax_et.set_title("E-t")
    ax_et.set_xlabel("Time (s)")
    ax_et.set_ylabel("Cumulative Energy (kWh)")
    add_difference_box(ax_et, primary_label, reference_label, diff_kwh, diff_pct)

    ax_es.plot(
        primary["distance_km"],
        primary["energy_kwh"],
        color="#d62728",
        lw=2.0,
        label=primary_label,
    )
    ax_es.plot(
        reference["distance_km"],
        reference["energy_kwh"],
        color="#111111",
        lw=1.8,
        ls="--",
        label=reference_label,
    )
    ax_es.set_title("E-s")
    ax_es.set_xlabel("Distance (km)")
    ax_es.set_ylabel("Cumulative Energy (kWh)")
    add_difference_box(ax_es, primary_label, reference_label, diff_kwh, diff_pct)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return diff_kwh, diff_pct


def comparison_row(
    trip: int,
    comparison: str,
    primary_label: str,
    reference_label: str,
    primary: dict[str, list[float]],
    reference: dict[str, list[float]],
    diff_kwh: float | None,
    diff_pct: float | None,
    plot_path: Path,
) -> dict[str, Any]:
    primary_summary = source_summary(primary)
    reference_summary = source_summary(reference)
    return {
        "trip_no": trip,
        "comparison": comparison,
        "primary": primary_label,
        "reference": reference_label,
        "primary_duration_s": primary_summary["duration_s"],
        "reference_duration_s": reference_summary["duration_s"],
        "duration_difference_s": primary_summary["duration_s"] - reference_summary["duration_s"],
        "primary_distance_km": primary_summary["distance_km"],
        "reference_distance_km": reference_summary["distance_km"],
        "primary_energy_kwh": primary_summary["energy_kwh"],
        "reference_energy_kwh": reference_summary["energy_kwh"],
        "energy_difference_kwh": diff_kwh if diff_kwh is not None else "",
        "energy_error_pct": diff_pct if diff_pct is not None else "",
        "error_formula": f"({primary_label} - {reference_label}) / {reference_label} * 100%",
        "energy_profile_note": (
            "Planned baseline uses exact section energy totals; within-section cumulative energy is distributed by distance."
            if reference_label == "Planned Baseline"
            else ""
        ),
        "plot": str(plot_path),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    trips = parse_trip_numbers(args.trips)
    plan_files = parse_plan_files(args.plan_file)
    trajectory_dirs = parse_trip_paths(args.trajectory_dir_for)
    output_dir = Path(args.output_dir)
    tsvp_dir = Path(args.tsvp_dir)

    section_args = SimpleNamespace(
        sections=args.sections,
        section_start=args.section_start,
        section_count=args.section_count,
    )
    sections = selected_sections(Path(args.route_map), section_args)
    if not sections:
        raise ValueError("No sections selected.")

    summary_rows: list[dict[str, Any]] = []
    for trip in trips:
        planned_path = tsvp_dir / args.planned_template.format(trip=trip)
        ot_history_path = tsvp_dir / args.ot_history_template.format(trip=trip)
        if not planned_path.exists():
            raise FileNotFoundError(planned_path)
        if not ot_history_path.exists():
            raise FileNotFoundError(ot_history_path)

        print(f"\nTrip {trip:03d}", flush=True)
        print(f"  OpenTrack planned: {planned_path}", flush=True)
        print(f"  OpenTrack history: {ot_history_path}", flush=True)
        planned_curve = read_tsvp(planned_path)
        ot_history_curve = read_tsvp(ot_history_path)

        plan_path = plan_files.get(trip)
        if plan_path is None:
            raise ValueError(f"No plan CSV configured for trip {trip}; add --plan-file {trip}=path.csv")
        traceability_path = Path(args.traceability_manifest)
        segment_by_section = (
            read_trip_segment_map(traceability_path, trip, sections)
            if traceability_path.is_file() and not args.ignore_traceability
            else {}
        )
        history_dwell_by_section, history_run_time_by_section = read_history_profile_maps(
            plan_path, sections, args.history_dwell
        )
        history_args = SimpleNamespace(
            data_dir=args.data_dir,
            trip_no=trip,
            run_id=None,
            quality_label=args.quality_label,
            history_dwell=args.history_dwell,
            segment_by_section=segment_by_section,
            history_dwell_by_section=history_dwell_by_section,
            history_run_time_by_section=history_run_time_by_section,
        )
        real_history_curve, _ = read_history_line(sections, history_args)
        trajectory_dir = trajectory_dirs.get(trip, Path(args.trajectory_dir))
        planned_baseline_curve = read_plan_baseline(plan_path, trajectory_dir, sections)
        print(f"  Planned baseline:  {plan_path}", flush=True)
        print(f"  Planned curves:    {trajectory_dir}", flush=True)
        print(
            f"  Traceability:     {len(segment_by_section)}/{len(sections)} sections; "
            f"historical dwell profile={len(history_dwell_by_section)} stations",
            flush=True,
        )

        comparisons = [
            (
                "01_ot_planned_vs_ot_history",
                planned_curve,
                ot_history_curve,
                "OT Planned",
                "OT History",
                "OpenTrack planned vs OpenTrack history",
            ),
            (
                "02_ot_planned_vs_real_history",
                planned_curve,
                real_history_curve,
                "OT Planned",
                "Real History",
                "OpenTrack planned vs real history",
            ),
            (
                "03_ot_history_vs_real_history",
                ot_history_curve,
                real_history_curve,
                "OT History",
                "Real History",
                "OpenTrack history vs real history",
            ),
            (
                "04_ot_planned_vs_planned_baseline",
                planned_curve,
                planned_baseline_curve,
                "OT Planned",
                "Planned Baseline",
                "OpenTrack planned vs original planned curve",
            ),
        ]

        trip_dir = output_dir / f"trip{trip:03d}"
        for comparison, primary, reference, primary_label, reference_label, title_text in comparisons:
            plot_path = trip_dir / f"trip{trip:03d}_{comparison}.{args.image_format}"
            diff_kwh, diff_pct = plot_pair(
                primary=primary,
                reference=reference,
                primary_label=primary_label,
                reference_label=reference_label,
                title=f"Trip {trip:03d} | {title_text} | {len(sections)} sections",
                output_path=plot_path,
                dpi=args.dpi,
                show=args.show,
            )
            summary_rows.append(
                comparison_row(
                    trip,
                    comparison,
                    primary_label,
                    reference_label,
                    primary,
                    reference,
                    diff_kwh,
                    diff_pct,
                    plot_path,
                )
            )
            pct_text = "N/A" if diff_pct is None else f"{diff_pct:+.2f}%"
            print(f"  {comparison}: {pct_text} -> {plot_path}", flush=True)

    summary_path = output_dir / "three_way_comparison_summary.csv"
    write_summary(summary_path, summary_rows)
    print(f"\nSummary: {summary_path}", flush=True)
    print(f"Plots:   {len(summary_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
