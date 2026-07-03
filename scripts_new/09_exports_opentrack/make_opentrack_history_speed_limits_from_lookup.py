from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

from opentrack_otd_client import print_response, send_command
from opentrack_otd_common import make_command_xml


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_LOOKUP = PROJECT_ROOT / "output" / "opentrack_speed_limits" / "history_speed_lookup.csv"
DEFAULT_ROUTE_MAP = PROJECT_ROOT / "output" / "opentrack_route_map" / "priority_history_probe_route_map.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "opentrack_speed_limits" / "speed_limits_history_from_lookup.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/send historical OpenTrack speed limits from a cached historical speed lookup CSV."
    )
    parser.add_argument("--speed-lookup-csv", default=str(DEFAULT_LOOKUP), help="CSV from build_opentrack_history_speed_lookup.py.")
    parser.add_argument("--route-map", default=str(DEFAULT_ROUTE_MAP), help="OpenTrack route map CSV.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT), help="Speed limit command table to write.")
    parser.add_argument("--section-start", type=int, default=1, help="First 1-based section index to include.")
    parser.add_argument("--section-count", type=int, default=None, help="Only include this many sections.")
    parser.add_argument(
        "--sections",
        default=None,
        help="Optional comma-separated section names to include, overriding --section-start/--section-count.",
    )
    parser.add_argument("--trip-no", type=int, default=1, help="Historical trip number, 1-based.")
    parser.add_argument("--run-id", default=None, help="Optional exact run_id. If provided, overrides --trip-no.")
    parser.add_argument(
        "--method",
        choices=["max", "p99", "p95"],
        default="max",
        help="Speed column from the lookup to use.",
    )
    parser.add_argument("--speed-margin", type=float, default=0.0, help="Additional km/h after lookup.")
    parser.add_argument("--ceil-speed", action="store_true", help="Round the final speed up to whole km/h.")
    parser.add_argument(
        "--range-mode",
        choices=["single-route", "next-route-boundary"],
        default="single-route",
        help="OpenTrack route range style.",
    )
    parser.add_argument("--single-route-end-offset", type=float, default=999999.0)
    parser.add_argument("--missing-end-offset", type=float, default=999999.0)
    parser.add_argument("--train-id", default=None, help="Optional OpenTrack trainID.")
    parser.add_argument("--send", action="store_true", help="Send setPositionSpeed commands to OpenTrack.")
    parser.add_argument("--host", default="127.0.0.1", help="OpenTrack OTD server host.")
    parser.add_argument("--port", type=int, default=9002, help="OpenTrack OTD server port.")
    parser.add_argument("--path", default="/otd", help="OTD HTTP path.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Socket timeout in seconds.")
    parser.add_argument("--wait-response", action="store_true", help="Wait for HTTP responses from OpenTrack.")
    parser.add_argument("--sleep", type=float, default=0.02, help="Delay between sent commands.")
    parser.add_argument("--verbose", action="store_true", help="Print SOAP command payloads while sending.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def selected_route_rows(route_rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.sections:
        selected = {item.strip() for item in args.sections.split(",") if item.strip()}
        return [row for row in route_rows if (row.get("section") or "").strip() in selected]
    start = max(args.section_start, 1) - 1
    end = None if args.section_count is None else start + args.section_count
    return route_rows[start:end]


def build_lookup_index(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        section = (row.get("section") or "").strip()
        if not section:
            continue
        if args.run_id:
            if (row.get("run_id") or "").strip() != args.run_id:
                continue
        else:
            trip_no = as_float(row.get("trip_no"))
            if trip_no is None or int(trip_no) != args.trip_no:
                continue
        index[section] = row
    return index


def make_limit_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    lookup_index = build_lookup_index(read_csv(Path(args.speed_lookup_csv)), args)
    route_rows = selected_route_rows(read_csv(Path(args.route_map)), args)
    speed_col = f"{args.method}_speed_kmh"

    output_rows: list[dict[str, Any]] = []
    for order, route in enumerate(route_rows, 1):
        section = (route.get("section") or "").strip()
        lookup = lookup_index.get(section)
        warnings: list[str] = []
        if lookup is None:
            warnings.append("missing_lookup_row")
            speed_base = None
        else:
            speed_base = as_float(lookup.get(speed_col))
            if speed_base is None:
                warnings.append(f"missing_{speed_col}")

        if speed_base is None:
            speed_limit = None
        else:
            speed_limit = speed_base + args.speed_margin
            if args.ceil_speed:
                speed_limit = float(math.ceil(speed_limit))

        start_route_id = (route.get("startRouteID") or "").strip()
        start_offset = as_float(route.get("startRouteOffset"))
        end_route_id = (route.get("endRouteID") or start_route_id).strip()
        end_offset = as_float(route.get("endRouteOffset"))

        if start_route_id and start_offset is None:
            start_offset = 0.0
            warnings.append("start_offset_defaulted_to_0")
        if end_route_id and end_offset is None:
            end_offset = args.missing_end_offset
            warnings.append("end_offset_defaulted")
        if args.range_mode == "single-route" and start_route_id:
            end_route_id = start_route_id
            end_offset = args.single_route_end_offset

        attrs: dict[str, Any] = {}
        if args.train_id:
            attrs["trainID"] = args.train_id
        if speed_limit is not None:
            attrs["speed"] = speed_limit
        attrs["startRouteID"] = start_route_id
        attrs["startRouteOffset"] = start_offset
        attrs["endRouteID"] = end_route_id
        attrs["endRouteOffset"] = end_offset
        xml = make_command_xml("setPositionSpeed", attrs) if speed_limit is not None and start_route_id else ""

        output_rows.append(
            {
                "send_order": order,
                "section": section,
                "trip_no": args.trip_no,
                "run_id_filter": args.run_id or "",
                "matched_run_id": lookup.get("run_id", "") if lookup else "",
                "matched_segment": lookup.get("segment", "") if lookup else "",
                "quality_labels": lookup.get("quality_labels", "") if lookup else "",
                "row_count": lookup.get("row_count", "") if lookup else "",
                "method": args.method,
                "speed_lookup_kmh": format_float(speed_base),
                "max_speed_kmh": lookup.get("max_speed_kmh", "") if lookup else "",
                "p99_speed_kmh": lookup.get("p99_speed_kmh", "") if lookup else "",
                "p95_speed_kmh": lookup.get("p95_speed_kmh", "") if lookup else "",
                "speed_margin_kmh": format_float(args.speed_margin),
                "speed_limit_kmh": format_float(speed_limit),
                "range_mode": args.range_mode,
                "startRouteID": start_route_id,
                "startRouteOffset": format_float(start_offset),
                "endRouteID": end_route_id,
                "endRouteOffset": format_float(end_offset),
                "trainID": args.train_id or "",
                "otd_xml": xml,
                "warning": ";".join(warnings),
            }
        )
    return output_rows


def send_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> int:
    failures = 0
    sent = 0
    for row in rows:
        speed = as_float(row.get("speed_limit_kmh"))
        if speed is None or not row.get("startRouteID"):
            failures += 1
            print(f"skip {row.get('section')}: {row.get('warning')}")
            continue
        attrs: dict[str, Any] = {}
        if args.train_id:
            attrs["trainID"] = args.train_id
        attrs["speed"] = speed
        attrs["startRouteID"] = row.get("startRouteID")
        attrs["startRouteOffset"] = as_float(row.get("startRouteOffset"))
        attrs["endRouteID"] = row.get("endRouteID")
        attrs["endRouteOffset"] = as_float(row.get("endRouteOffset"))
        status, text = send_command("setPositionSpeed", attrs, args)
        sent += 1
        print(f"[{sent:02d}] {row.get('section')} lookup <= {row.get('speed_limit_kmh')} km/h")
        print_response(status, text)
        if not (status == 0 or 200 <= status < 300):
            failures += 1
        if args.sleep > 0:
            time.sleep(args.sleep)
    print(f"Sent {sent} lookup setPositionSpeed command(s), failures={failures}")
    return 0 if failures == 0 else 1


def main() -> int:
    args = parse_args()
    rows = make_limit_rows(args)
    write_csv(Path(args.output_csv), rows)
    warnings = sum(1 for row in rows if row.get("warning"))
    print(f"Lookup CSV: {Path(args.speed_lookup_csv)}")
    print(f"Route map:  {Path(args.route_map)}")
    print(f"Output CSV: {Path(args.output_csv)}")
    if args.run_id:
        print(f"Run ID:     {args.run_id}")
    else:
        print(f"Trip no:    {args.trip_no}")
    print(f"Sections:   {len(rows)}")
    print(f"Warnings:   {warnings}")
    if args.send:
        return send_rows(rows, args)
    print("Not sent. Add --send to push lookup-based setPositionSpeed commands to OpenTrack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
