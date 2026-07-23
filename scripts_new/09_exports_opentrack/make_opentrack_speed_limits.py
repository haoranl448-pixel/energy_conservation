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
DEFAULT_PLAN_CSV = PROJECT_ROOT / "output" / "schedule" / "final_plan_report_v2 - 副本" / "Final_Planning_Comparison.csv"
DEFAULT_ENERGY_MENU = PROJECT_ROOT / "output" / "analysis" / "ato_class_energy_menu1_new_v3.csv"
DEFAULT_ROUTE_MAP = PROJECT_ROOT / "output" / "opentrack_route_map" / "priority_history_probe_route_map.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "opentrack_speed_limits" / "speed_limits_priority.csv"
DEFAULT_CURVE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build OpenTrack setPositionSpeed commands from a DP final plan, "
            "an ATO class energy menu, and an OpenTrack route map."
        )
    )
    parser.add_argument("--plan-csv", default=str(DEFAULT_PLAN_CSV), help="Final_Planning_Comparison*.csv.")
    parser.add_argument("--energy-menu", default=str(DEFAULT_ENERGY_MENU), help="ato_class_energy_menu*_new_v3.csv.")
    parser.add_argument("--route-map", default=str(DEFAULT_ROUTE_MAP), help="OpenTrack route map CSV.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT), help="Speed limit command table to write.")
    parser.add_argument(
        "--section-start",
        type=int,
        default=1,
        help="First 1-based section index to include.",
    )
    parser.add_argument(
        "--section-count",
        type=int,
        default=None,
        help="Only include this many sections from --section-start.",
    )
    parser.add_argument(
        "--sections",
        default=None,
        help="Optional comma-separated section names to include, overriding --section-start/--section-count.",
    )
    parser.add_argument(
        "--train-id",
        default=None,
        help=(
            "Optional OpenTrack trainID. If omitted, setPositionSpeed is sent without "
            "trainID and applies to all trains in the simulation."
        ),
    )
    parser.add_argument(
        "--speed-margin",
        type=float,
        default=0.0,
        help="Additional km/h added to the selected ATO speed before sending the limit.",
    )
    parser.add_argument(
        "--speed-method",
        choices=["cruise-avg", "peak"],
        default="cruise-avg",
        help=(
            "Speed used for setPositionSpeed. cruise-avg reads the generated curve "
            "and averages the high-speed cruise plateau; peak keeps the old peak-speed behavior."
        ),
    )
    parser.add_argument(
        "--curve-dir",
        default=str(DEFAULT_CURVE_DIR),
        help="Directory containing <section>/<class>_generated_curve.csv files for --speed-method cruise-avg.",
    )
    parser.add_argument(
        "--cruise-min-speed-ratio",
        type=float,
        default=0.92,
        help="Cruise candidate speed must be at least this ratio of the curve peak speed.",
    )
    parser.add_argument(
        "--cruise-accel-eps",
        type=float,
        default=0.25,
        help="Cruise candidate acceleration threshold in m/s^2.",
    )
    parser.add_argument(
        "--cruise-min-points",
        type=int,
        default=10,
        help="Minimum cruise points before accepting the acceleration-filtered plateau.",
    )
    parser.add_argument(
        "--ceil-speed",
        action="store_true",
        help="Round speed limits up to whole km/h after applying --speed-margin.",
    )
    parser.add_argument(
        "--opentrack-speed-unit",
        choices=["kmh", "mph"],
        default="kmh",
        help=(
            "Unit expected by the current OpenTrack project for setPositionSpeed. "
            "The ATO menu is always read as km/h; choose mph when the OpenTrack "
            "project/display uses miles per hour."
        ),
    )
    parser.add_argument(
        "--missing-end-offset",
        type=float,
        default=999999.0,
        help=(
            "Fallback endRouteOffset when routeEntry fallback has no precise final "
            "offset, usually only for the last section."
        ),
    )
    parser.add_argument(
        "--range-mode",
        choices=["next-route-boundary", "single-route"],
        default="next-route-boundary",
        help=(
            "How to convert each section to a setPositionSpeed range. "
            "next-route-boundary uses current route offset 0 to next route offset 0. "
            "single-route uses current route offset 0 to current route --single-route-end-offset."
        ),
    )
    parser.add_argument(
        "--single-route-end-offset",
        type=float,
        default=999999.0,
        help="End offset used by --range-mode single-route.",
    )
    parser.add_argument("--send", action="store_true", help="Send setPositionSpeed commands to OpenTrack.")
    parser.add_argument("--host", default="127.0.0.1", help="OpenTrack OTD server host.")
    parser.add_argument("--port", type=int, default=9002, help="OpenTrack OTD server port.")
    parser.add_argument("--path", default="/otd", help="OTD HTTP path.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Socket timeout in seconds.")
    parser.add_argument("--wait-response", action="store_true", help="Wait for HTTP responses from OpenTrack.")
    parser.add_argument("--sleep", type=float, default=0.02, help="Delay between sent commands.")
    parser.add_argument(
        "--send-retries",
        type=int,
        default=8,
        help="Retry count when OpenTrack temporarily refuses a command connection.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between transient connection retries.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each XML command while sending.")
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
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_energy_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        section = (row.get("站间区间") or "").strip()
        ato_class = (row.get("运行等级") or "").strip()
        if section and ato_class:
            index[(section, ato_class)] = row
    return index


def build_route_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        section = (row.get("section") or row.get("站间区间") or "").strip()
        if section:
            index[section] = row
    return index


def make_speed_limit(base_speed: float, margin: float, ceil_speed: bool) -> float:
    speed = base_speed + margin
    if ceil_speed:
        speed = float(math.ceil(speed))
    return speed


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def read_generated_curve(curve_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with curve_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_s = as_float(row.get("time_s"))
            speed_kmh = as_float(row.get("velocity_kmh"))
            speed_mps = as_float(row.get("velocity_mps"))
            if speed_kmh is None and speed_mps is not None:
                speed_kmh = speed_mps * 3.6
            if speed_mps is None and speed_kmh is not None:
                speed_mps = speed_kmh / 3.6
            if time_s is None or speed_kmh is None or speed_mps is None:
                continue
            rows.append({"time_s": time_s, "speed_kmh": speed_kmh, "speed_mps": speed_mps})
    return rows


def curve_accelerations(curve: list[dict[str, float]]) -> list[float]:
    accels: list[float] = []
    previous: dict[str, float] | None = None
    for row in curve:
        if previous is None:
            accels.append(0.0)
        else:
            dt = row["time_s"] - previous["time_s"]
            if dt <= 0:
                accels.append(0.0)
            else:
                accels.append((row["speed_mps"] - previous["speed_mps"]) / dt)
        previous = row
    return accels


def cruise_average_speed(
    section: str,
    selected_class: str,
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, Any]:
    curve_path = Path(args.curve_dir) / section / f"{selected_class}_generated_curve.csv"
    result: dict[str, Any] = {
        "speed_kmh": None,
        "cruise_avg_speed_kmh": None,
        "cruise_point_count": 0,
        "curve_peak_speed_kmh": None,
        "curve_path": str(curve_path),
    }
    if not curve_path.exists():
        warnings.append("missing_generated_curve_for_cruise_avg")
        return result

    curve = read_generated_curve(curve_path)
    speeds = [row["speed_kmh"] for row in curve if row["speed_kmh"] >= 0]
    if not speeds:
        warnings.append("empty_generated_curve_for_cruise_avg")
        return result

    curve_peak = max(speeds)
    result["curve_peak_speed_kmh"] = curve_peak
    threshold = curve_peak * args.cruise_min_speed_ratio
    accels = curve_accelerations(curve)

    cruise_speeds = [
        row["speed_kmh"]
        for row, accel in zip(curve, accels)
        if row["speed_kmh"] >= threshold and abs(accel) <= args.cruise_accel_eps
    ]
    if len(cruise_speeds) < args.cruise_min_points:
        warnings.append("cruise_accel_filter_relaxed")
        cruise_speeds = [row["speed_kmh"] for row in curve if row["speed_kmh"] >= threshold]

    cruise_avg = mean(cruise_speeds)
    if cruise_avg is None:
        warnings.append("cruise_avg_fallback_to_curve_peak")
        cruise_avg = curve_peak

    result["speed_kmh"] = cruise_avg
    result["cruise_avg_speed_kmh"] = cruise_avg
    result["cruise_point_count"] = len(cruise_speeds)
    return result


def to_opentrack_speed(speed_kmh: float | None, unit: str) -> float | None:
    if speed_kmh is None:
        return None
    if unit == "mph":
        return speed_kmh / 1.609344
    return speed_kmh


def make_limit_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan_rows = read_csv(Path(args.plan_csv))
    energy_index = build_energy_index(read_csv(Path(args.energy_menu)))
    route_index = build_route_index(read_csv(Path(args.route_map)))

    output_rows: list[dict[str, Any]] = []
    selected_sections = None
    if args.sections:
        selected_sections = {item.strip() for item in args.sections.split(",") if item.strip()}
    for order, plan in enumerate(plan_rows, 1):
        section = (plan.get("站间区间") or "").strip()
        selected_class = (plan.get("选定等级") or "").strip()
        if not section or section == "--- 总计 ---":
            continue
        if selected_sections is not None:
            if section not in selected_sections:
                continue
        elif order < args.section_start:
            continue
        elif args.section_count is not None and order >= args.section_start + args.section_count:
            continue

        energy = energy_index.get((section, selected_class))
        route = route_index.get(section)
        warning: list[str] = []
        if energy is None:
            warning.append("missing_energy_menu_row")
        if route is None:
            warning.append("missing_route_map_row")

        peak_speed = as_float(energy.get("峰值速度(kmh)") if energy else None)
        if peak_speed is None:
            warning.append("missing_peak_speed")
        cruise_info: dict[str, Any] = {
            "speed_kmh": None,
            "cruise_avg_speed_kmh": None,
            "cruise_point_count": "",
            "curve_peak_speed_kmh": None,
            "curve_path": "",
        }

        if args.speed_method == "peak":
            speed_base = peak_speed
        else:
            cruise_info = cruise_average_speed(section, selected_class, args, warning)
            speed_base = as_float(cruise_info.get("speed_kmh"))
            if speed_base is None and peak_speed is not None:
                warning.append("cruise_avg_missing_fallback_to_menu_peak")
                speed_base = peak_speed

        if speed_base is None:
            speed_limit = None
        else:
            speed_limit = make_speed_limit(speed_base, args.speed_margin, args.ceil_speed)
        opentrack_speed = to_opentrack_speed(speed_limit, args.opentrack_speed_unit)

        start_route_id = (route.get("startRouteID") if route else "") or ""
        start_offset = as_float(route.get("startRouteOffset") if route else None)
        end_route_id = (route.get("endRouteID") if route else "") or start_route_id
        end_offset = as_float(route.get("endRouteOffset") if route else None)
        if start_route_id and start_offset is None:
            start_offset = 0.0
            warning.append("start_offset_defaulted_to_0")
        if end_route_id and end_offset is None:
            end_offset = args.missing_end_offset
            warning.append("end_offset_defaulted")
        if args.range_mode == "single-route" and start_route_id:
            end_route_id = start_route_id
            end_offset = args.single_route_end_offset

        attrs: dict[str, Any] = {}
        if args.train_id:
            attrs["trainID"] = args.train_id
        if opentrack_speed is not None:
            attrs["speed"] = opentrack_speed
        attrs["startRouteID"] = start_route_id
        attrs["startRouteOffset"] = start_offset
        attrs["endRouteID"] = end_route_id
        attrs["endRouteOffset"] = end_offset

        xml = make_command_xml("setPositionSpeed", attrs) if not warning or speed_limit is not None else ""
        output_rows.append(
            {
                "send_order": order,
                "section": section,
                "selected_class": selected_class,
                "planned_time_s": plan.get("规划用时(s)", ""),
                "speed_method": args.speed_method,
                "speed_base_kmh": format_float(speed_base),
                "cruise_avg_speed_kmh": format_float(as_float(cruise_info.get("cruise_avg_speed_kmh"))),
                "cruise_point_count": cruise_info.get("cruise_point_count", ""),
                "curve_peak_speed_kmh": format_float(as_float(cruise_info.get("curve_peak_speed_kmh"))),
                "peak_speed_kmh": format_float(peak_speed),
                "speed_margin_kmh": format_float(args.speed_margin),
                "speed_limit_kmh": format_float(speed_limit),
                "opentrack_speed_unit": args.opentrack_speed_unit,
                "opentrack_speed_value": format_float(opentrack_speed),
                "curve_path": cruise_info.get("curve_path", ""),
                "curve_source": plan.get("曲线来源", energy.get("曲线来源", "") if energy else ""),
                "sample_reliability": plan.get("样本可靠性", energy.get("样本可靠性", "") if energy else ""),
                "range_mode": args.range_mode,
                "startRouteID": start_route_id,
                "startRouteOffset": format_float(start_offset),
                "endRouteID": end_route_id,
                "endRouteOffset": format_float(end_offset),
                "trainID": args.train_id or "",
                "otd_xml": xml,
                "warning": ";".join(warning),
            }
        )
    return output_rows


def send_command_with_retry(
    command_name: str,
    attrs: dict[str, Any],
    args: argparse.Namespace,
    section: str,
) -> tuple[int, str]:
    retries = max(int(args.send_retries), 0)
    delay = max(float(args.retry_delay), 0.0)
    for attempt in range(retries + 1):
        try:
            return send_command(command_name, attrs, args)
        except OSError as exc:
            if attempt >= retries:
                raise
            print(
                f"  {section}: OpenTrack connection failed ({exc}); "
                f"retry {attempt + 1}/{retries} in {delay:.2f}s"
            )
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("Unreachable retry state.")


def send_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> int:
    failures = 0
    sent = 0
    for row in rows:
        if row.get("warning") and not row.get("speed_limit_kmh"):
            failures += 1
            print(f"skip {row.get('section')}: {row.get('warning')}")
            continue
        attrs: dict[str, Any] = {}
        if args.train_id:
            attrs["trainID"] = args.train_id
        attrs["speed"] = as_float(row.get("opentrack_speed_value") or row.get("speed_limit_kmh"))
        attrs["startRouteID"] = row.get("startRouteID")
        attrs["startRouteOffset"] = as_float(row.get("startRouteOffset"))
        attrs["endRouteID"] = row.get("endRouteID")
        attrs["endRouteOffset"] = as_float(row.get("endRouteOffset"))
        status, text = send_command_with_retry(
            "setPositionSpeed",
            attrs,
            args,
            str(row.get("section") or "<unknown section>"),
        )
        sent += 1
        print(f"[{sent:02d}] {row.get('section')} {row.get('selected_class')} <= {row.get('speed_limit_kmh')} km/h")
        print_response(status, text)
        if not (status == 0 or 200 <= status < 300):
            failures += 1
        if args.sleep > 0:
            time.sleep(args.sleep)
    print(f"Sent {sent} setPositionSpeed command(s), failures={failures}")
    return 0 if failures == 0 else 1


def main() -> int:
    args = parse_args()
    rows = make_limit_rows(args)
    write_csv(Path(args.output_csv), rows)

    warnings = sum(1 for row in rows if row.get("warning"))
    print(f"Plan:        {Path(args.plan_csv)}")
    print(f"Energy menu: {Path(args.energy_menu)}")
    print(f"Route map:   {Path(args.route_map)}")
    print(f"Output CSV:  {Path(args.output_csv)}")
    print(f"Sections:    {len(rows)}")
    if args.sections:
        print(f"Filter:      sections={args.sections}")
    elif args.section_count is not None or args.section_start != 1:
        print(f"Filter:      section_start={args.section_start}, section_count={args.section_count}")
    print(f"Warnings:    {warnings}")
    if args.send:
        return send_rows(rows, args)
    print("Not sent. Add --send to push setPositionSpeed commands to OpenTrack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
