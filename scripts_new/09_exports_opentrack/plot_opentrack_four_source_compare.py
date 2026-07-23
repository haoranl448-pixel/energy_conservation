from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

from plot_opentrack_three_way_compare import (
    comparison_row,
    plot_pair,
    read_history_profile_maps,
    read_trip_segment_map,
    write_summary,
)
from plot_opentrack_tsvp_vs_history import read_history_line, read_tsvp, selected_sections


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_TSVP_DIR = Path(r"D:\OutPut")
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_processed_step2_v3_all_curve_quality"
DEFAULT_ROUTE_MAP = (
    PROJECT_ROOT
    / "output"
    / "opentrack_route_map_newline"
    / "priority_dp_trip011_route_map.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "opentrack_four_source_compare_trip6_trip12"
DEFAULT_TRACEABILITY_MANIFEST = (
    Path.home()
    / "Desktop"
    / "\u6570\u636e\u6d4b\u8bd5\u4ee3\u7801"
    / "data_processed_step2_v3_all_curve_quality_traceability"
    / "trip_traceability_manifest_v1.csv"
)
DEFAULT_PLAN_ROOT = (
    PROJECT_ROOT
    / "output"
    / "schedule"
    / "batch_trip_reports_v4_dwell5_trip6_trip12"
)
DEFAULT_HISTORY_CACHE_DIR = PROJECT_ROOT / "output" / "cache" / "planning_history_curves"
HISTORY_CACHE_VERSION = "history_trip_curve_v2_traceability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw all six pairwise comparisons among standard DP, dwell-5pct DP, "
            "OpenTrack history, and real history. Each output contains v-t, v-s, E-t, and E-s."
        )
    )
    parser.add_argument("--trips", default="6,12", help="Comma-separated 1-based trip numbers.")
    parser.add_argument("--tsvp-dir", default=str(DEFAULT_TSVP_DIR))
    parser.add_argument(
        "--standard-template",
        default="OT_priority_dp_trip{trip:03d}.tsvP",
        help="Standard real-priority DP TSVP filename template.",
    )
    parser.add_argument(
        "--flex5-template",
        default="OT_real_priority_dwell_5pct_dp_trip{trip:03d}.tsvP",
        help="Real-priority dwell-5pct DP TSVP filename template.",
    )
    parser.add_argument(
        "--ot-history-template",
        default="OT_priority_history_trip{trip:03d}.tsvP",
        help="OpenTrack historical TSVP filename template.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--route-map", default=str(DEFAULT_ROUTE_MAP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--traceability-manifest",
        default=str(DEFAULT_TRACEABILITY_MANIFEST),
        help="Global trip traceability CSV used to select section-local historical segments.",
    )
    parser.add_argument(
        "--ignore-traceability",
        action="store_true",
        help="Use each section's original local trip number instead of traceability mapping.",
    )
    parser.add_argument(
        "--history-cache-dir",
        default=str(DEFAULT_HISTORY_CACHE_DIR),
        help="DP historical-curve cache directory used to avoid rereading large Step2 workbooks.",
    )
    parser.add_argument(
        "--ignore-history-cache",
        action="store_true",
        help="Read Step2 xlsx files directly even when matching DP history caches exist.",
    )
    parser.add_argument(
        "--plan-file",
        action="append",
        default=[],
        metavar="TRIP=CSV",
        help="Override the standard planning CSV for one trip.",
    )
    parser.add_argument("--history-dwell", type=float, default=30.0)
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
    for value in text.split(","):
        value = value.strip()
        if not value:
            continue
        trip = int(value)
        if trip < 1:
            raise ValueError(f"Trip number must be at least 1: {trip}")
        if trip not in trips:
            trips.append(trip)
    if not trips:
        raise ValueError("No trip numbers supplied.")
    return trips


def default_plan_path(trip: int) -> Path:
    return DEFAULT_PLAN_ROOT / f"trip{trip:03d}" / "Final_Planning_Comparison.csv"


def parse_plan_files(values: list[str], trips: list[int]) -> dict[int, Path]:
    result = {trip: default_plan_path(trip) for trip in trips}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --plan-file value, expected TRIP=CSV: {value}")
        trip_text, path_text = value.split("=", 1)
        trip = int(trip_text.strip())
        if trip < 1 or not path_text.strip():
            raise ValueError(f"Invalid --plan-file value: {value}")
        result[trip] = Path(path_text.strip())
    return result


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def source_paths(tsvp_dir: Path, args: argparse.Namespace, trip: int) -> dict[str, Path]:
    return {
        "standard": tsvp_dir / args.standard_template.format(trip=trip),
        "flex5": tsvp_dir / args.flex5_template.format(trip=trip),
        "ot_history": tsvp_dir / args.ot_history_template.format(trip=trip),
    }


def history_cache_paths(
    manifest_path: Path,
    data_dir: Path,
    cache_dir: Path,
    trip: int,
    sections: list[str],
) -> dict[str, Path]:
    if not manifest_path.is_file() or not cache_dir.is_dir():
        return {}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    prefix = f"UP{trip:03d}_"
    row_by_section = {
        (row.get("区段") or "").strip(): row
        for row in rows
        if (row.get("全局趟次候选ID") or "").strip().startswith(prefix)
    }

    result: dict[str, Path] = {}
    for section in sections:
        row = row_by_section.get(section)
        excel_path = data_dir / f"results_{section}.xlsx"
        if row is None or not excel_path.is_file():
            continue
        stat = excel_path.stat()
        selection_key = "|".join(
            [
                (row.get("全局趟次候选ID") or "").strip(),
                (row.get("segment") or "").strip(),
                (row.get("来源run_id") or "").strip(),
            ]
        )
        key_source = "|".join(
            [
                HISTORY_CACHE_VERSION,
                str(excel_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                section,
                selection_key,
            ]
        )
        digest = hashlib.sha1(
            key_source.encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        cache_path = cache_dir / f"{digest}.pkl"
        if cache_path.is_file():
            result[section] = cache_path
    return result


def main() -> int:
    args = parse_args()
    trips = parse_trip_numbers(args.trips)
    plan_files = parse_plan_files(args.plan_file, trips)
    tsvp_dir = Path(args.tsvp_dir)
    output_dir = Path(args.output_dir)

    section_args = SimpleNamespace(
        sections=args.sections,
        section_start=args.section_start,
        section_count=args.section_count,
    )
    sections = selected_sections(Path(args.route_map), section_args)
    if not sections:
        raise ValueError("No sections selected.")

    summary_rows: list[dict[str, object]] = []
    for trip in trips:
        paths = source_paths(tsvp_dir, args, trip)
        for key, path in paths.items():
            require_file(path, f"Trip {trip:03d} {key} TSVP")

        plan_path = plan_files[trip]
        require_file(plan_path, f"Trip {trip:03d} plan CSV")

        standard_curve = read_tsvp(paths["standard"])
        flex5_curve = read_tsvp(paths["flex5"])
        ot_history_curve = read_tsvp(paths["ot_history"])

        traceability_path = Path(args.traceability_manifest)
        segment_by_section = (
            read_trip_segment_map(traceability_path, trip, sections)
            if traceability_path.is_file() and not args.ignore_traceability
            else {}
        )
        history_section_cache = (
            history_cache_paths(
                traceability_path,
                Path(args.data_dir),
                Path(args.history_cache_dir),
                trip,
                sections,
            )
            if segment_by_section and not args.ignore_history_cache
            else {}
        )
        history_dwell_by_section, history_run_time_by_section = read_history_profile_maps(
            plan_path,
            sections,
            args.history_dwell,
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
            history_section_cache=history_section_cache,
        )
        real_history_curve, _ = read_history_line(sections, history_args)

        print(f"\nTrip {trip:03d}", flush=True)
        print(f"  Standard DP: {paths['standard']}", flush=True)
        print(f"  Dwell 5% DP: {paths['flex5']}", flush=True)
        print(f"  OT history:  {paths['ot_history']}", flush=True)
        print(
            f"  Real history: traceability {len(segment_by_section)}/{len(sections)} sections; "
            f"dwell profile {len(history_dwell_by_section)} stations; "
            f"cache {len(history_section_cache)}/{len(sections)} sections",
            flush=True,
        )

        comparisons = [
            (
                "01_standard_dp_vs_ot_history",
                standard_curve,
                ot_history_curve,
                "OT Standard DP",
                "OT History",
                "standard DP vs OpenTrack history",
            ),
            (
                "02_dwell5_dp_vs_ot_history",
                flex5_curve,
                ot_history_curve,
                "OT Dwell 5% DP",
                "OT History",
                "dwell-5pct DP vs OpenTrack history",
            ),
            (
                "03_standard_dp_vs_dwell5_dp",
                standard_curve,
                flex5_curve,
                "OT Standard DP",
                "OT Dwell 5% DP",
                "standard DP vs dwell-5pct DP",
            ),
            (
                "04_standard_dp_vs_real_history",
                standard_curve,
                real_history_curve,
                "OT Standard DP",
                "Real History",
                "standard DP vs real history",
            ),
            (
                "05_dwell5_dp_vs_real_history",
                flex5_curve,
                real_history_curve,
                "OT Dwell 5% DP",
                "Real History",
                "dwell-5pct DP vs real history",
            ),
            (
                "06_ot_history_vs_real_history",
                ot_history_curve,
                real_history_curve,
                "OT History",
                "Real History",
                "OpenTrack history vs real history",
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

    summary_path = output_dir / "four_source_pairwise_summary.csv"
    write_summary(summary_path, summary_rows)
    print(f"\nSummary: {summary_path}", flush=True)
    print(f"Plots:   {len(summary_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
