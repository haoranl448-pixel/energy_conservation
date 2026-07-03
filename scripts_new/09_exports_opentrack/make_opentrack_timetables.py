from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import pandas as pd


DEFAULT_REPORT_DIR = Path(r"D:\energy_conservation\output\schedule\final_plan_report_v2 - 副本")
DEFAULT_TEMPLATE = Path(r"D:\energy_conservation\eg.xml")

SECTION_COL = "站间区间"
PLAN_RUN_COL = "规划用时(s)"
HISTORY_RUN_COL = "历史用时(s)"
DWELL_COL = "停站时间(s)"
MASS_COL = "MASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DP planning comparison CSV files into OpenTrack timetable XML files. "
            "Each CSV produces two timetables: planned and historical."
        )
    )
    parser.add_argument(
        "--priority-csv",
        default=str(DEFAULT_REPORT_DIR / "Final_Planning_Comparison.csv"),
        help="Priority-first comparison CSV.",
    )
    parser.add_argument(
        "--energy-first-csv",
        default=str(DEFAULT_REPORT_DIR / "Final_Planning_Comparison_Energy_First.csv"),
        help="Energy-first comparison CSV.",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Example OpenTrack XML. Used only to mirror the basic root attributes.",
    )
    parser.add_argument(
        "--output-dir",
        default="opentrack_timetables",
        help="Directory for generated XML timetables.",
    )
    parser.add_argument(
        "--start-time",
        default="08:00:00",
        help="Departure time at STA_1, format HH:MM:SS.",
    )
    parser.add_argument(
        "--history-dwell-mode",
        choices=["csv", "fixed", "zero"],
        default="csv",
        help=(
            "Dwell time for historical timetable: csv uses 停站时间(s), "
            "fixed uses --fixed-history-dwell, zero uses 0."
        ),
    )
    parser.add_argument(
        "--fixed-history-dwell",
        type=float,
        default=30.0,
        help="Historical dwell time when --history-dwell-mode fixed.",
    )
    parser.add_argument(
        "--include-delta-load",
        dest="include_delta_load",
        action="store_true",
        default=True,
        help="Write deltaLoad from MASS changes. Enabled by default because eg.xml uses deltaLoad.",
    )
    parser.add_argument(
        "--no-delta-load",
        dest="include_delta_load",
        action="store_false",
        help="Do not write deltaLoad elements.",
    )
    parser.add_argument(
        "--write-station-map",
        action="store_true",
        help="Write station_id_mapping.csv next to the XML files.",
    )
    return parser.parse_args()


def read_planning_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {SECTION_COL, PLAN_RUN_COL, HISTORY_RUN_COL, DWELL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    for col in [PLAN_RUN_COL, HISTORY_RUN_COL, DWELL_COL, MASS_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # The final summary row starts with "--- 总计 ---"; it is not a real station section.
    section_text = df[SECTION_COL].astype(str).str.strip()
    section_mask = (
        section_text.str.contains("-", regex=False)
        & ~section_text.str.startswith("---")
        & df[PLAN_RUN_COL].notna()
        & df[HISTORY_RUN_COL].notna()
    )
    df = df[section_mask].copy()
    if df.empty:
        raise ValueError(f"{path} has no station sections.")
    return df.reset_index(drop=True)


def read_template_attrs(path: Path) -> dict[str, str]:
    attrs = {"title": "OpenTrack timetable", "application": "OpenTrack"}
    if not path.exists():
        return attrs
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return attrs
    attrs["title"] = root.attrib.get("title", attrs["title"])
    attrs["application"] = root.attrib.get("application", attrs["application"])
    return attrs


def station_sequence(sections: Iterable[str]) -> list[str]:
    stations: list[str] = []
    for i, section in enumerate(sections):
        if "-" not in section:
            raise ValueError(f"Invalid station section: {section!r}")
        left, right = [part.strip() for part in section.split("-", 1)]
        if i == 0:
            stations.append(left)
        elif stations[-1] != left:
            raise ValueError(
                f"Section order is not continuous near {section!r}: previous end is {stations[-1]!r}"
            )
        stations.append(right)
    return stations


def parse_hms(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"Time must use HH:MM:SS, got {value!r}") from exc


def hms(dt: datetime) -> str:
    # Timetable timestamps are HH:MM:SS. Keep fractional seconds in the accumulator,
    # then round the displayed clock time only at the moment it is written.
    return (dt + timedelta(seconds=0.5)).strftime("%H:%M:%S")


def seconds_value(value: float) -> float:
    if pd.isna(value):
        raise ValueError("Encountered NaN time value.")
    return float(value)


def format_seconds(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def make_entry_xml(
    station_id: str,
    arrival: str,
    departure: str,
    wait_seconds: float,
    stop_information: str,
    delta_load: float | None = None,
) -> str:
    lines = [
        f'\t\t<timetableEntry stopInformation="{stop_information}">',
        f"\t\t\t<stationID>{escape(station_id)}</stationID>",
        f'\t\t\t<arrivalTime format="hh:mm:ss" type="planned" valid="yes">{escape(arrival)}</arrivalTime>',
        (
            '\t\t\t<departureTime format="hh:mm:ss" type="planned" '
            f'useDepartureTime="yes" valid="yes">{escape(departure)}</departureTime>'
        ),
        f'\t\t\t<waitTime format="s">{format_seconds(wait_seconds)}</waitTime>',
        "\t\t\t<delayTime format=\"s\">0</delayTime>",
    ]
    if delta_load is not None:
        lines.append(f"\t\t\t<deltaLoad>{delta_load:.3f}</deltaLoad>")
    lines.append("\t\t</timetableEntry>")
    return "\n".join(lines)


def compute_delta_loads(df: pd.DataFrame, station_count: int) -> list[float | None]:
    if MASS_COL not in df.columns:
        return [None] * station_count

    masses = df[MASS_COL].tolist()
    if not masses or all(pd.isna(m) for m in masses):
        return [None] * station_count

    # MASS is section-level. For a timetable station entry, only changes from one section to
    # the next have a meaningful approximate delta. Keep STA_1 empty.
    deltas: list[float | None] = [None]
    prev = masses[0]
    deltas.append(None)
    for mass in masses[1:]:
        if pd.isna(prev) or pd.isna(mass):
            deltas.append(None)
        else:
            delta = float(mass) - float(prev)
            deltas.append(None if abs(delta) < 0.0005 else delta)
        prev = mass
    while len(deltas) < station_count:
        deltas.append(None)
    return deltas[:station_count]


def build_timetable_xml(
    df: pd.DataFrame,
    course_id: str,
    run_col: str,
    dwell_mode: str,
    fixed_history_dwell: float,
    start_time: str,
    include_delta_load: bool,
    root_attrs: dict[str, str],
) -> tuple[str, pd.DataFrame]:
    stations = station_sequence(df[SECTION_COL].astype(str))
    station_ids = [f"STA_{i}" for i in range(1, len(stations) + 1)]
    current_departure = parse_hms(start_time)

    if include_delta_load:
        delta_loads = compute_delta_loads(df, len(stations))
    else:
        delta_loads = [None] * len(stations)

    entries: list[str] = []
    entries.append(
        make_entry_xml(
            station_id=station_ids[0],
            arrival="HH:MM:SS",
            departure=hms(current_departure),
            wait_seconds=0,
            stop_information="no",
            delta_load=delta_loads[0],
        )
    )

    for i, row in df.iterrows():
        run_seconds = seconds_value(row[run_col])
        arrival = current_departure + timedelta(seconds=run_seconds)

        is_final_station = i == len(df) - 1
        if is_final_station:
            dwell_seconds = 0.0
        elif run_col == HISTORY_RUN_COL:
            if dwell_mode == "csv":
                dwell_seconds = seconds_value(row[DWELL_COL])
            elif dwell_mode == "fixed":
                dwell_seconds = seconds_value(fixed_history_dwell)
            else:
                dwell_seconds = 0.0
        else:
            dwell_seconds = seconds_value(row[DWELL_COL])

        departure = arrival + timedelta(seconds=dwell_seconds)
        entries.append(
            make_entry_xml(
                station_id=station_ids[i + 1],
                arrival=hms(arrival),
                departure=hms(departure),
                wait_seconds=dwell_seconds,
                stop_information="yes",
                delta_load=delta_loads[i + 1],
            )
        )
        current_departure = departure

    now_text = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    xml = "\n".join(
        [
            '<?xml version="1.0"?>',
            "<?xml-stylesheet?>",
            '<!DOCTYPE timetable SYSTEM "eg.dtd">',
            (
                f'<timetable title="{escape(root_attrs["title"])}" '
                f'application="{escape(root_attrs["application"])}" date="{escape(now_text)}">'
            ),
            "\t<course>",
            f"\t\t<courseID>{escape(course_id)}</courseID>",
            *entries,
            "\t</course>",
            "</timetable>",
            "",
        ]
    )

    station_map = pd.DataFrame(
        {
            "station_id": station_ids,
            "station_name": stations,
            "course_id": course_id,
        }
    )
    return xml, station_map


def write_one(
    df: pd.DataFrame,
    output_path: Path,
    course_id: str,
    run_col: str,
    dwell_mode: str,
    fixed_history_dwell: float,
    start_time: str,
    include_delta_load: bool,
    root_attrs: dict[str, str],
) -> pd.DataFrame:
    xml, station_map = build_timetable_xml(
        df=df,
        course_id=course_id,
        run_col=run_col,
        dwell_mode=dwell_mode,
        fixed_history_dwell=fixed_history_dwell,
        start_time=start_time,
        include_delta_load=include_delta_load,
        root_attrs=root_attrs,
    )
    output_path.write_text(xml, encoding="utf-8", newline="\n")
    return station_map


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_attrs = read_template_attrs(Path(args.template))

    jobs = [
        (
            Path(args.priority_csv),
            "priority",
            [
                ("priority_dp_planned_timetable.xml", "priority_dp", PLAN_RUN_COL),
                ("priority_history_timetable.xml", "priority_history", HISTORY_RUN_COL),
            ],
        ),
        (
            Path(args.energy_first_csv),
            "energy_first",
            [
                ("energy_first_dp_planned_timetable.xml", "energy_first_dp", PLAN_RUN_COL),
                ("energy_first_history_timetable.xml", "energy_first_history", HISTORY_RUN_COL),
            ],
        ),
    ]

    station_maps: list[pd.DataFrame] = []
    for csv_path, label, outputs in jobs:
        df = read_planning_csv(csv_path)
        print(f"Loaded {label}: {csv_path} ({len(df)} sections)")
        for file_name, course_id, run_col in outputs:
            path = output_dir / file_name
            station_map = write_one(
                df=df,
                output_path=path,
                course_id=course_id,
                run_col=run_col,
                dwell_mode=args.history_dwell_mode,
                fixed_history_dwell=args.fixed_history_dwell,
                start_time=args.start_time,
                include_delta_load=args.include_delta_load,
                root_attrs=root_attrs,
            )
            station_maps.append(station_map)
            print(f"  wrote {path}")

    if args.write_station_map and station_maps:
        station_map = pd.concat(station_maps, ignore_index=True).drop_duplicates()
        map_path = output_dir / "station_id_mapping.csv"
        station_map.to_csv(map_path, index=False, encoding="utf-8-sig")
        print(f"  wrote {map_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
