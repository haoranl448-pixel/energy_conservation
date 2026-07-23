from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output" / "schedule" / "batch_trip_reports_trip1_125"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "merged_timetables" / "priority_history_trip001_070.xml"

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-trip OpenTrack timetable XML files into one multi-course XML.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing tripXXX/timetables folders.")
    parser.add_argument("--trips", required=True, help="Trips to merge, e.g. 1-70 or 1,2,5.")
    parser.add_argument(
        "--xml-pattern",
        default="trip{trip:03d}/timetables/priority_history_trip{trip:03d}.xml",
        help="XML path pattern under --input-dir. The placeholder {trip:03d} is supported.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Merged XML output path.")
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.0,
        help="Offset each later course by N seconds. Example: 120 means trip002 starts 2 min after trip001.",
    )
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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_clock(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


def shift_time_text(text: str | None, offset_seconds: float) -> str | None:
    if not text or not TIME_RE.match(text):
        return text
    dt = datetime.strptime(text, "%H:%M:%S") + timedelta(seconds=offset_seconds)
    return format_clock(dt)


def shift_course_times(course: ET.Element, offset_seconds: float) -> None:
    if abs(offset_seconds) < 1e-9:
        return
    for tag in ["arrivalTime", "departureTime"]:
        for elem in course.iter(tag):
            elem.text = shift_time_text(elem.text, offset_seconds)


def indent(elem: ET.Element, level: int = 0) -> None:
    space = "\n" + level * "\t"
    child_space = "\n" + (level + 1) * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_space
        for child in elem:
            indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = space
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = space


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_path = resolve_path(args.output)
    trips = parse_trip_list(args.trips)

    merged_root: ET.Element | None = None
    course_count = 0
    for index, trip_no in enumerate(trips):
        rel = args.xml_pattern.format(trip=trip_no)
        xml_path = input_dir / rel
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)

        root = ET.parse(xml_path).getroot()
        if merged_root is None:
            merged_root = ET.Element(root.tag, root.attrib)

        courses = root.findall("course")
        if not courses:
            raise ValueError(f"No <course> found in {xml_path}")
        offset = index * args.stagger_seconds
        for course in courses:
            shift_course_times(course, offset)
            merged_root.append(course)
            course_count += 1
        print(f"merged trip{trip_no:03d}: {xml_path}")

    if merged_root is None:
        raise RuntimeError("No XML files merged.")

    indent(merged_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = ET.tostring(merged_root, encoding="unicode", short_empty_elements=False)
    text = "\n".join(
        [
            '<?xml version="1.0"?>',
            "<?xml-stylesheet?>",
            '<!DOCTYPE timetable SYSTEM "eg.dtd">',
            body,
            "",
        ]
    )
    output_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"\nMerged courses: {course_count}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
