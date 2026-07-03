from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from html import escape as html_escape
from typing import Any
from xml.sax.saxutils import quoteattr


SOAP_NS = "http://schemas.xmlsoap.org/soap/envelope/"


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    if ":" in tag:
        return tag.rsplit(":", 1)[1]
    return tag


def seconds_from_hms(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == "HH:MM:SS":
        return None
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS, got {value!r}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def make_command_xml(command_name: str, attrs: dict[str, Any] | None = None) -> str:
    attrs = attrs or {}
    attr_text = "".join(
        f" {key}={quoteattr(format_otd_value(value))}"
        for key, value in attrs.items()
        if value is not None
    )
    return f"<{command_name}{attr_text}/>"


def format_otd_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def make_soap_envelope(inner_xml: str) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<SOAP-ENV:Envelope xmlns:SOAP-ENV="{SOAP_NS}">',
            "<SOAP-ENV:Body>",
            inner_xml,
            "</SOAP-ENV:Body>",
            "</SOAP-ENV:Envelope>",
        ]
    )


def empty_soap_response() -> bytes:
    return make_soap_envelope("").encode("utf-8")


def extract_soap_payload(xml_text: str) -> dict[str, Any]:
    """Return the first command/notification inside a SOAP body.

    OpenTrack examples use SOAP over HTTP. During early experiments it is useful to be
    forgiving, so this function first tries XML parsing and then falls back to a regex.
    """

    result: dict[str, Any] = {
        "name": None,
        "attrs": {},
        "text": None,
        "parse_error": None,
    }
    try:
        root = ET.fromstring(xml_text)
        body = None
        for elem in root.iter():
            if local_name(elem.tag) == "Body":
                body = elem
                break
        if body is not None:
            for child in list(body):
                result["name"] = local_name(child.tag)
                result["attrs"] = dict(child.attrib)
                result["text"] = (child.text or "").strip() or None
                return result
    except ET.ParseError as exc:
        result["parse_error"] = str(exc)

    body_match = re.search(
        r"<(?:[A-Za-z_][\w.-]*:)?Body\b[^>]*>(.*?)</(?:[A-Za-z_][\w.-]*:)?Body>",
        xml_text,
        flags=re.S,
    )
    search_text = body_match.group(1) if body_match else xml_text
    match = re.search(r"<([A-Za-z_][\w:.-]*)([^<>]*?)(?:/?>)", search_text)
    if match:
        result["name"] = local_name(match.group(1))
        attrs: dict[str, str] = {}
        attr_pattern = re.compile(
            r"([A-Za-z_][\w:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s/>]+))"
        )
        for attr_match in attr_pattern.finditer(match.group(2)):
            value = next(
                group for group in attr_match.groups()[1:] if group is not None
            )
            attrs[attr_match.group(1)] = value
        result["attrs"] = attrs
    return result


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def html_message(title: str, body: str) -> bytes:
    page = (
        "<html><head><meta charset='utf-8'><title>"
        + html_escape(title)
        + "</title></head><body><pre>"
        + html_escape(body)
        + "</pre></body></html>"
    )
    return page.encode("utf-8")
