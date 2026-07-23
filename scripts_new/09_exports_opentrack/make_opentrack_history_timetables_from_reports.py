from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from make_opentrack_timetables import (
    DWELL_COL,
    HISTORY_RUN_COL,
    MASS_COL,
    SECTION_COL,
    read_template_attrs,
    write_one,
)


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output" / "schedule" / "batch_trip_reports_trip1_10"
DEFAULT_TEMPLATE = PROJECT_ROOT / "eg.xml"

REPORT_SECTION_COL = "站间区间"
REPORT_RUN_COL = "历史运行时间(s)"
REPORT_DWELL_COL = "历史停站时间(s)"
REPORT_MASS_COL = "平均重量(t)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Historical_Only_Report.csv files into OpenTrack historical timetable XML files."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing tripXXX folders.")
    parser.add_argument("--trips", default=None, help="Trips to export, e.g. 1,2,5-10. Defaults to all tripXXX folders.")
    parser.add_argument("--report-name", default="Historical_Only_Report.csv", help="Report CSV name inside each trip folder.")
    parser.add_argument("--output-subdir", default="timetables", help="Output subdirectory inside each trip folder.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Example OpenTrack XML template.")
    parser.add_argument("--start-time", default="08:00:00", help="Departure time at STA_1, format HH:MM:SS.")
    parser.add_argument("--course-prefix", default="priority_history", help="Course ID prefix.")
    parser.add_argument("--file-prefix", default="priority_history", help="XML file prefix.")
    parser.add_argument("--history-dwell-mode", choices=["csv", "fixed", "zero"], default="csv")
    parser.add_argument("--fixed-history-dwell", type=float, default=30.0)
    parser.add_argument(
        "--include-delta-load",
        dest="include_delta_load",
        action="store_true",
        default=True,
        help="Write deltaLoad from average mass changes. OpenTrack XML import can read it, while OTD API may ignore it.",
    )
    parser.add_argument("--no-delta-load", dest="include_delta_load", action="store_false")
    parser.add_argument("--write-station-map", action="store_true", default=True)
    return parser.parse_args()


def parse_trip_list(text: str) -> list[int]:
    trips: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left_text, right_text = item.split("-", 1)
            left = int(left_text.strip())
            right = int(right_text.strip())
            if left <= 0 or right <= 0 or right < left:
                raise ValueError(f"Invalid trip range: {item}")
            trips.extend(range(left, right + 1))
        else:
            value = int(item)
            if value <= 0:
                raise ValueError("Trip numbers are 1-based positive integers.")
            trips.append(value)
    if not trips:
        raise ValueError("No trips selected.")
    return sorted(dict.fromkeys(trips))


def discover_trips(input_dir: Path) -> list[int]:
    trips: list[int] = []
    for path in input_dir.glob("trip*"):
        if not path.is_dir():
            continue
        suffix = path.name.replace("trip", "", 1)
        if suffix.isdigit():
            trips.append(int(suffix))
    return sorted(trips)


def read_history_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {REPORT_SECTION_COL, REPORT_RUN_COL, REPORT_DWELL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    section_text = df[REPORT_SECTION_COL].astype(str).str.strip()
    df = df[section_text.str.contains("-", regex=False) & ~section_text.str.startswith("---")].copy()
    if df.empty:
        raise ValueError(f"{path} has no section rows.")

    converted = pd.DataFrame()
    converted[SECTION_COL] = df[REPORT_SECTION_COL].astype(str).str.strip()
    converted[HISTORY_RUN_COL] = pd.to_numeric(df[REPORT_RUN_COL], errors="coerce")
    converted[DWELL_COL] = pd.to_numeric(df[REPORT_DWELL_COL], errors="coerce")
    if REPORT_MASS_COL in df.columns:
        converted[MASS_COL] = pd.to_numeric(df[REPORT_MASS_COL], errors="coerce")

    if converted[HISTORY_RUN_COL].isna().any():
        bad = converted.loc[converted[HISTORY_RUN_COL].isna(), SECTION_COL].tolist()
        raise ValueError(f"{path} has NaN historical running times for: {bad}")
    if converted[DWELL_COL].isna().any():
        bad = converted.loc[converted[DWELL_COL].isna(), SECTION_COL].tolist()
        raise ValueError(f"{path} has NaN dwell times for: {bad}")
    return converted.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    trips = parse_trip_list(args.trips) if args.trips else discover_trips(input_dir)
    if not trips:
        raise RuntimeError(f"No trip folders found under {input_dir}")

    root_attrs = read_template_attrs(Path(args.template))
    print(f"Input dir: {input_dir}")
    print(f"Trips:     {trips}")

    for trip_no in trips:
        trip_label = f"trip{trip_no:03d}"
        trip_dir = input_dir / trip_label
        report_path = trip_dir / args.report_name
        if not report_path.exists():
            raise FileNotFoundError(report_path)

        df = read_history_report(report_path)
        output_dir = trip_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        course_id = f"{args.course_prefix}_{trip_label}"
        xml_path = output_dir / f"{args.file_prefix}_{trip_label}.xml"
        station_map = write_one(
            df=df,
            output_path=xml_path,
            course_id=course_id,
            run_col=HISTORY_RUN_COL,
            dwell_mode=args.history_dwell_mode,
            fixed_history_dwell=args.fixed_history_dwell,
            start_time=args.start_time,
            include_delta_load=args.include_delta_load,
            root_attrs=root_attrs,
        )
        print(f"{trip_label}: wrote {xml_path}")

        if args.write_station_map:
            station_map_path = output_dir / "station_id_mapping.csv"
            station_map.to_csv(station_map_path, index=False, encoding="utf-8-sig")
            print(f"{trip_label}: wrote {station_map_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
