from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from opentrack_otd_common import empty_soap_response, extract_soap_payload, html_message, now_iso


class OTDMessageLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"otd_messages_{now_iso().replace(':', '')}.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_handler(logger: OTDMessageLogger):
    class OTDHandler(BaseHTTPRequestHandler):
        server_version = "OpenTrackOTDListener/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep stdout focused on OTD messages. HTTP access logs are still available
            # in the JSONL file when messages arrive.
            return

        def do_GET(self) -> None:
            body = html_message(
                "OpenTrack OTD listener",
                "OpenTrack OTD listener is running.\n"
                "Configure OpenTrack OTD Server to this host/port, then enable Use OTD-Communication.",
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length)
            text = raw.decode("utf-8", errors="replace")
            payload = extract_soap_payload(text)
            record = {
                "received_at": now_iso(),
                "client": self.client_address[0],
                "path": self.path,
                "headers": dict(self.headers),
                "message": payload.get("name"),
                "attrs": payload.get("attrs", {}),
                "parse_error": payload.get("parse_error"),
                "raw_xml": text,
            }
            logger.write(record)

            name = record["message"] or "<unknown>"
            attrs = record["attrs"] or {}
            print(f"[{record['received_at']}] {name} {attrs}")

            body = empty_soap_response()
            self.send_response(200)
            self.send_header("Content-Type", "text/xml; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # OpenTrack often closes fire-and-forget OTD requests before reading
                # the empty HTTP response. The incoming message is already logged.
                pass

    return OTDHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for OpenTrack OTD SOAP-over-HTTP notifications."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=9004, help="OTD server port.")
    parser.add_argument(
        "--log-dir",
        default="opentrack_otd_logs",
        help="Directory for JSONL message logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = OTDMessageLogger(Path(args.log_dir))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(logger))
    print(f"Listening for OpenTrack OTD messages on http://{args.host}:{args.port}/otd")
    print(f"JSONL log: {logger.log_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
