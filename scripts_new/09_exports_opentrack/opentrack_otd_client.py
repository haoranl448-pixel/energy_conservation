from __future__ import annotations

import argparse
import http.client
import socket
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from opentrack_otd_common import make_command_xml, make_soap_envelope, seconds_from_hms


DEFAULT_PATH = "/otd"


def send_inner_xml(
    inner_xml: str,
    host: str,
    port: int,
    path: str = DEFAULT_PATH,
    timeout: float = 10.0,
    verbose: bool = False,
    wait_response: bool = False,
) -> tuple[int, str]:
    body = make_soap_envelope(inner_xml).encode("utf-8")
    headers = {
        "User-Agent": "OpenTrackOTDClient",
        "Content-Type": "text/xml; charset=utf-8",
        "Content-Length": str(len(body)),
        "Connection": "Close",
    }
    if verbose:
        print(f"POST http://{host}:{port}{path}")
        print(inner_xml)

    if not wait_response:
        raw_headers = [
            f"POST {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "User-Agent: OpenTrackOTDClient",
            "SOAPAction:",
            "Content-Type: text/xml; charset=utf-8",
            f"Content-Length: {len(body)}",
            "Connection: Close",
            "",
            "",
        ]
        request = "\r\n".join(raw_headers).encode("ascii") + body
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(request)
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        return 0, "sent_without_waiting_for_http_response"

    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers=headers)
        response = conn.getresponse()
        response_text = response.read().decode("utf-8", errors="replace")
        return response.status, response_text
    finally:
        conn.close()


def send_command(
    command_name: str,
    attrs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, str]:
    inner_xml = make_command_xml(command_name, attrs)
    return send_inner_xml(
        inner_xml,
        host=args.host,
        port=args.port,
        path=args.path,
        timeout=args.timeout,
        verbose=args.verbose,
        wait_response=args.wait_response,
    )


def parse_key_values(pairs: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Attribute must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid attribute key in {pair!r}")
        attrs[key] = value.strip()
    return attrs


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "yes", "true", "y"}:
        return True
    if lowered in {"0", "no", "false", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean 1/0 yes/no true/false, got {value!r}")


def iter_timetable_entries(xml_path: Path, train_id_override: str | None = None):
    root = ET.parse(xml_path).getroot()
    courses = root.findall("course")
    if not courses:
        raise ValueError(f"{xml_path} has no <course> elements.")

    for course in courses:
        course_id = (course.findtext("courseID") or "").strip()
        train_id = train_id_override or course_id
        if not train_id:
            raise ValueError(f"{xml_path} contains a course without courseID.")

        for entry in course.findall("timetableEntry"):
            station_id = (entry.findtext("stationID") or "").strip()
            if not station_id:
                continue
            arrival = seconds_from_hms(entry.findtext("arrivalTime"))
            departure = seconds_from_hms(entry.findtext("departureTime"))
            dwell_text = entry.findtext("waitTime")
            dwell = float(dwell_text) if dwell_text not in (None, "") else None
            stop_info = (entry.attrib.get("stopInformation") or "").lower()
            stop_flag = stop_info in {"yes", "true", "1"}

            attrs = {
                "trainID": train_id,
                "stationID": station_id,
                "arrivalTime": arrival,
                "departureTime": departure,
                "dwellTime": dwell,
                "stopFlag": stop_flag,
            }
            yield attrs


def print_response(status: int, text: str) -> None:
    if status == 0:
        print(text)
        return
    print(f"HTTP {status}")
    if text.strip():
        print(text.strip())


def command_start(args: argparse.Namespace) -> int:
    attrs = {"time": args.time} if args.time is not None else {}
    status, text = send_command("startSimulation", attrs, args)
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_pause(args: argparse.Namespace) -> int:
    attrs = {"time": args.time} if args.time is not None else {}
    status, text = send_command("pauseSimulation", attrs, args)
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_end(args: argparse.Namespace) -> int:
    attrs = {"time": args.time} if args.time is not None else {}
    status, text = send_command("endSimulation", attrs, args)
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_step(args: argparse.Namespace) -> int:
    status, text = send_command("stepSimulation", {}, args)
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_reset_timetable(args: argparse.Namespace) -> int:
    status, text = send_command("resetTimetable", {}, args)
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_position_reports(args: argparse.Namespace) -> int:
    attrs = {
        "trainID": args.train_id,
        "flag": args.flag,
        "time": args.interval,
    }
    status, text = send_command("setSendPositionReports", attrs, args)
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_generic(args: argparse.Namespace) -> int:
    attrs = parse_key_values(args.attrs)
    status, text = send_command(args.name, attrs, args)
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_raw(args: argparse.Namespace) -> int:
    status, text = send_inner_xml(
        args.xml,
        host=args.host,
        port=args.port,
        path=args.path,
        timeout=args.timeout,
        verbose=args.verbose,
        wait_response=args.wait_response,
    )
    print_response(status, text)
    return 0 if 200 <= status < 300 else 1


def command_load_timetable(args: argparse.Namespace) -> int:
    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    sent = 0
    failures = 0
    if args.reset:
        status, text = send_command("resetTimetable", {}, args)
        print(f"resetTimetable -> HTTP {status}")
        if not (200 <= status < 300):
            failures += 1
            if text.strip():
                print(text.strip())

    for attrs in iter_timetable_entries(xml_path, train_id_override=args.train_id):
        status, text = send_command("addTimetableEntry", attrs, args)
        sent += 1
        if not (200 <= status < 300):
            failures += 1
            print(f"addTimetableEntry {attrs.get('trainID')} {attrs.get('stationID')} -> HTTP {status}")
            if text.strip():
                print(text.strip())
        elif args.verbose:
            print(f"addTimetableEntry {attrs.get('trainID')} {attrs.get('stationID')} -> HTTP {status}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Sent {sent} timetable entries from {xml_path}")
    return 0 if failures == 0 else 1


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1", help="OpenTrack server host.")
    parser.add_argument("--port", type=int, default=9002, help="OpenTrack server port.")
    parser.add_argument("--path", default=DEFAULT_PATH, help="HTTP path used by OTD.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Print SOAP command payloads.")
    parser.add_argument(
        "--wait-response",
        action="store_true",
        help=(
            "Wait for a regular HTTP response. By default the client sends the "
            "OTD command and closes the socket, because some OpenTrack versions "
            "accept commands without returning HTTP status."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send OpenTrack OTD SOAP-over-HTTP commands to port 9002."
    )
    add_common_options(parser)
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("start-simulation", help="Send startSimulation([time]).")
    p.add_argument("--time", type=float, default=None)
    p.set_defaults(func=command_start)

    p = sub.add_parser("pause-simulation", help="Send pauseSimulation([time]).")
    p.add_argument("--time", type=float, default=None)
    p.set_defaults(func=command_pause)

    p = sub.add_parser("end-simulation", help="Send endSimulation([time]).")
    p.add_argument("--time", type=float, default=None)
    p.set_defaults(func=command_end)

    p = sub.add_parser("step-simulation", help="Send stepSimulation().")
    p.set_defaults(func=command_step)

    p = sub.add_parser("reset-timetable", help="Send resetTimetable().")
    p.set_defaults(func=command_reset_timetable)

    p = sub.add_parser("set-position-reports", help="Send setSendPositionReports(trainID, flag, [time]).")
    p.add_argument("train_id", help="OpenTrack train/course ID.")
    p.add_argument("flag", type=parse_bool, help="1/yes/true to enable, 0/no/false to disable.")
    p.add_argument("--interval", type=float, default=1.0, help="Report interval in simulation seconds.")
    p.set_defaults(func=command_position_reports)

    p = sub.add_parser("send", help="Send any OTD command as name key=value ...")
    p.add_argument("name", help="Command element name, e.g. setSimulationStartTime.")
    p.add_argument("attrs", nargs="*", help="Attributes as key=value.")
    p.set_defaults(func=command_generic)

    p = sub.add_parser("send-raw", help="Send raw inner XML inside the SOAP Body.")
    p.add_argument("xml", help='Example: "<startSimulation/>"')
    p.set_defaults(func=command_raw)

    p = sub.add_parser("load-timetable", help="Send addTimetableEntry commands from XML.")
    p.add_argument("xml_file", help="OpenTrack timetable XML file.")
    p.add_argument("--train-id", default=None, help="Override <courseID> as trainID.")
    p.add_argument("--reset", action="store_true", help="Send resetTimetable before entries.")
    p.add_argument("--sleep", type=float, default=0.02, help="Delay between commands.")
    p.set_defaults(func=command_load_timetable)

    p = sub.add_parser("set-wait-departure", help="Send setWaitForDepartureCommand.")
    p.add_argument("train_id")
    p.add_argument("flag", type=parse_bool)
    p.set_defaults(
        func=lambda args: command_generic(
            argparse.Namespace(
                **vars(args),
                name="setWaitForDepartureCommand",
                attrs=[f"trainID={args.train_id}", f"flag={1 if args.flag else 0}"],
            )
        )
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ConnectionRefusedError:
        print(
            f"Connection refused: OpenTrack is not listening at {args.host}:{args.port}. "
            "Start OpenTrack with -otd and enable OTD communication."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
