from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from plot_opentrack_three_way_compare import comparison_row, plot_pair, write_summary
from plot_opentrack_tsvp_vs_history import read_tsvp


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_TSVP_DIR = Path(r"D:\OutPut")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "opentrack_four_source_compare_complete10"
)
DEFAULT_TRIPS = "1,3,4,5,6,8,10,12,14,15"

NEW_COMPARISONS = {
    "07_energy_first_vs_standard_dp",
    "08_energy_first_vs_dwell5_dp",
    "09_energy_first_vs_ot_history",
}

COMPARISON_LABELS = {
    "01_standard_dp_vs_ot_history": "\u6807\u51c6DP vs OpenTrack\u5386\u53f2",
    "02_dwell5_dp_vs_ot_history": "\u505c\u7ad9\u5bbd\u677e5%DP vs OpenTrack\u5386\u53f2",
    "03_standard_dp_vs_dwell5_dp": "\u6807\u51c6DP vs \u505c\u7ad9\u5bbd\u677e5%DP",
    "04_standard_dp_vs_real_history": "\u6807\u51c6DP vs \u771f\u5b9e\u5386\u53f2",
    "05_dwell5_dp_vs_real_history": "\u505c\u7ad9\u5bbd\u677e5%DP vs \u771f\u5b9e\u5386\u53f2",
    "06_ot_history_vs_real_history": "OpenTrack\u5386\u53f2 vs \u771f\u5b9e\u5386\u53f2",
    "07_energy_first_vs_standard_dp": "\u80fd\u8017\u4f18\u5148DP vs \u6807\u51c6DP",
    "08_energy_first_vs_dwell5_dp": "\u80fd\u8017\u4f18\u5148DP vs \u505c\u7ad9\u5bbd\u677e5%DP",
    "09_energy_first_vs_ot_history": "\u80fd\u8017\u4f18\u5148DP vs OpenTrack\u5386\u53f2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add three OpenTrack energy-first comparisons for each trip: "
            "energy-first vs standard DP, dwell-5pct DP, and OpenTrack history."
        )
    )
    parser.add_argument("--trips", default=DEFAULT_TRIPS)
    parser.add_argument("--tsvp-dir", default=str(DEFAULT_TSVP_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--energy-first-template",
        default="OT_energy_first_dp_trip{trip:03d}.tsvP",
    )
    parser.add_argument(
        "--standard-template",
        default="OT_priority_dp_trip{trip:03d}.tsvP",
    )
    parser.add_argument(
        "--flex5-template",
        default="OT_real_priority_dwell_5pct_dp_trip{trip:03d}.tsvP",
    )
    parser.add_argument(
        "--ot-history-template",
        default="OT_priority_history_trip{trip:03d}.tsvP",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def parse_trips(text: str) -> list[int]:
    trips: list[int] = []
    for raw_value in text.split(","):
        value = raw_value.strip()
        if not value:
            continue
        trip = int(value)
        if trip < 1:
            raise ValueError(f"Trip number must be positive: {trip}")
        if trip not in trips:
            trips.append(trip)
    if not trips:
        raise ValueError("No trip numbers supplied.")
    return trips


def read_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mass_by_trip(rows: list[dict[str, Any]]) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in rows:
        try:
            trip = int(row.get("trip_no", ""))
            mass = float(row.get("max_mass_t", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(mass) and mass > 0:
            result.setdefault(trip, mass)
    return result


def target_time_by_trip(rows: list[dict[str, Any]]) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in rows:
        try:
            trip = int(row.get("trip_no", ""))
        except (TypeError, ValueError):
            continue

        target = as_float(row.get("target_time_s"))
        if target is None and row.get("reference") == "Real History":
            target = as_float(row.get("reference_duration_s"))
        if target is not None and target > 0:
            result.setdefault(trip, target)
    return result


def target_components_by_trip(
    rows: list[dict[str, Any]],
) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for row in rows:
        try:
            trip = int(row.get("trip_no", ""))
        except (TypeError, ValueError):
            continue
        run_time = as_float(row.get("target_run_time_s"))
        dwell_time = as_float(row.get("target_dwell_time_s"))
        if (
            run_time is not None
            and dwell_time is not None
            and run_time > 0
            and dwell_time >= 0
        ):
            result.setdefault(trip, (run_time, dwell_time))
    return result


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_average_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        comparison = str(row.get("comparison", "")).strip()
        if comparison:
            grouped[comparison].append(row)

    headers = [
        "\u5bf9\u6bd4\u7f16\u53f7",
        "\u5bf9\u6bd4\u7c7b\u578b",
        "\u8d9f\u6b21\u6570",
        "\u80fd\u8017\u5dee\u5e73\u5747\u503c(kWh)",
        "\u80fd\u8017\u8bef\u5dee\u767e\u5206\u6bd4\u5e73\u5747\u503c(%)",
        "\u80fd\u8017\u5dee\u7edd\u5bf9\u503c\u5e73\u5747(kWh)",
        "\u80fd\u8017\u8bef\u5dee\u767e\u5206\u6bd4\u7edd\u5bf9\u503c\u5e73\u5747(%)",
    ]
    output_rows: list[list[Any]] = []
    for comparison in sorted(grouped):
        group = grouped[comparison]
        differences = [
            number
            for row in group
            if (number := as_float(row.get("energy_difference_kwh"))) is not None
        ]
        percentages = [
            number
            for row in group
            if (number := as_float(row.get("energy_error_pct"))) is not None
        ]
        if not differences or not percentages:
            continue
        output_rows.append(
            [
                comparison.split("_", 1)[0],
                COMPARISON_LABELS.get(comparison, comparison),
                min(len(differences), len(percentages)),
                round(sum(differences) / len(differences), 4),
                round(sum(percentages) / len(percentages), 4),
                round(sum(abs(value) for value in differences) / len(differences), 4),
                round(sum(abs(value) for value in percentages) / len(percentages), 4),
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(output_rows)


def required_path(tsvp_dir: Path, template: str, trip: int, label: str) -> Path:
    path = tsvp_dir / template.format(trip=trip)
    if not path.is_file():
        raise FileNotFoundError(f"Trip {trip:03d} {label} TSVP not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    trips = parse_trips(args.trips)
    tsvp_dir = Path(args.tsvp_dir)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "four_source_pairwise_summary.csv"
    average_path = output_dir / "energy_comparison_average_summary.csv"

    existing_rows = read_existing_summary(summary_path)
    masses = mass_by_trip(existing_rows)
    target_times = target_time_by_trip(existing_rows)
    target_components = target_components_by_trip(existing_rows)
    selected_trip_set = set(trips)
    merged_rows = [
        row
        for row in existing_rows
        if not (
            as_float(row.get("trip_no")) in selected_trip_set
            and row.get("comparison") in NEW_COMPARISONS
        )
    ]

    new_rows: list[dict[str, Any]] = []
    for trip in trips:
        energy_first_path = required_path(
            tsvp_dir, args.energy_first_template, trip, "energy-first"
        )
        standard_path = required_path(
            tsvp_dir, args.standard_template, trip, "standard DP"
        )
        flex5_path = required_path(tsvp_dir, args.flex5_template, trip, "dwell-5pct DP")
        history_path = required_path(
            tsvp_dir, args.ot_history_template, trip, "OpenTrack history"
        )

        energy_first = read_tsvp(energy_first_path)
        standard = read_tsvp(standard_path)
        flex5 = read_tsvp(flex5_path)
        history = read_tsvp(history_path)
        max_mass_t = masses.get(trip)
        target_time_s = target_times.get(trip)
        target_run_time_s, target_dwell_time_s = target_components.get(
            trip, (None, None)
        )
        mass_text = (
            f"{max_mass_t:.2f} t" if max_mass_t is not None else "not available"
        )
        target_time_text = (
            f"{target_time_s:.1f} s"
            if target_time_s is not None
            else "not available"
        )
        target_components_text = (
            f"{target_run_time_s:.1f} / {target_dwell_time_s:.1f} s"
            if target_run_time_s is not None and target_dwell_time_s is not None
            else target_time_text
        )

        comparisons = [
            (
                "07_energy_first_vs_standard_dp",
                standard,
                "OT Standard DP",
                "energy-first DP vs standard DP",
            ),
            (
                "08_energy_first_vs_dwell5_dp",
                flex5,
                "OT Dwell 5% DP",
                "energy-first DP vs dwell-5pct DP",
            ),
            (
                "09_energy_first_vs_ot_history",
                history,
                "OT History",
                "energy-first DP vs OpenTrack history",
            ),
        ]

        print(
            f"\nTrip {trip:03d} | Target time (run/dwell): "
            f"{target_components_text} | "
            f"Max load: {mass_text}",
            flush=True,
        )
        trip_dir = output_dir / f"trip{trip:03d}"
        for comparison, reference, reference_label, title_text in comparisons:
            plot_path = trip_dir / (
                f"trip{trip:03d}_{comparison}.{args.image_format}"
            )
            diff_kwh, diff_pct = plot_pair(
                primary=energy_first,
                reference=reference,
                primary_label="OT Energy-first DP",
                reference_label=reference_label,
                title=(
                    f"Trip {trip:03d} | Target time (run/dwell): "
                    f"{target_components_text} | "
                    f"Max load: {mass_text} | "
                    f"{title_text} | 26 sections"
                ),
                output_path=plot_path,
                dpi=args.dpi,
                show=args.show,
            )
            row = comparison_row(
                trip,
                comparison,
                "OT Energy-first DP",
                reference_label,
                energy_first,
                reference,
                diff_kwh,
                diff_pct,
                plot_path,
            )
            row["max_mass_t"] = "" if max_mass_t is None else max_mass_t
            row["target_time_s"] = (
                "" if target_time_s is None else target_time_s
            )
            row["target_run_time_s"] = (
                "" if target_run_time_s is None else target_run_time_s
            )
            row["target_dwell_time_s"] = (
                "" if target_dwell_time_s is None else target_dwell_time_s
            )
            new_rows.append(row)
            pct_text = "N/A" if diff_pct is None else f"{diff_pct:+.2f}%"
            print(f"  {comparison}: {pct_text} -> {plot_path}", flush=True)

    merged_rows.extend(new_rows)
    merged_rows.sort(
        key=lambda row: (
            int(float(row.get("trip_no", 0))),
            str(row.get("comparison", "")),
        )
    )
    write_summary(summary_path, merged_rows)
    write_average_summary(average_path, merged_rows)

    print(f"\nNew plots: {len(new_rows)}", flush=True)
    print(f"Summary:   {summary_path}", flush=True)
    print(f"Averages:  {average_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
