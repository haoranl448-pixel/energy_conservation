# -*- coding: utf-8 -*-
"""Resolve one global train trip to the matching local segment in each section.

Step2 numbers ``segment`` independently inside every section workbook.  Once a
section misses or gains a candidate, the same segment number no longer means
the same physical train across the full line.  The traceability manifest fixes
that ambiguity by recording the local segment and source run ID for every
section in one global trip chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TRACEABILITY_FILENAME = "trip_traceability_manifest_v1.csv"


@dataclass(frozen=True)
class TraceabilityRecord:
    section: str
    segment: int | float | str
    source_run_id: str
    global_trip_id: str
    absolute_start: str
    absolute_end: str
    trainset: str


@dataclass(frozen=True)
class TripTraceability:
    manifest_path: Path
    trip_no: int
    direction: str
    global_trip_id: str
    records: dict[str, TraceabilityRecord]

    def record_for(self, section: str) -> TraceabilityRecord:
        try:
            return self.records[section]
        except KeyError as exc:
            raise KeyError(
                f"全局趟次 {self.global_trip_id} 在追溯表中缺少区间 {section}。"
            ) from exc

    def require_sections(self, sections: Iterable[str]) -> None:
        missing = [section for section in sections if section not in self.records]
        if missing:
            raise ValueError(
                f"全局趟次 {self.global_trip_id} 缺少 {len(missing)} 个目标区间: "
                + ", ".join(missing)
            )


def _resolve_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def resolve_traceability_manifest(
    project_root: Path,
    data_dir: Path | None,
) -> Path | None:
    """Find the manifest explicitly or beside the selected Step2 data directory."""

    raw = os.environ.get("ENERGY_TRACEABILITY_MANIFEST")
    if raw:
        explicit = _resolve_path(raw, project_root)
        if not explicit.is_file():
            raise FileNotFoundError(f"Traceability manifest not found: {explicit}")
        return explicit

    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(data_dir / TRACEABILITY_FILENAME)
        if not data_dir.name.endswith("_traceability"):
            candidates.append(
                data_dir.with_name(f"{data_dir.name}_traceability") / TRACEABILITY_FILENAME
            )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _normalize_segment(value: Any) -> int | float | str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    rounded = round(number)
    if abs(number - rounded) <= 1e-9:
        return int(rounded)
    return number


def load_trip_traceability(
    manifest_path: Path | None,
    trip_no: int,
    direction: str = "UP",
) -> TripTraceability | None:
    """Load the section-to-segment mapping for a 1-based global trip number."""

    if manifest_path is None:
        return None
    if trip_no < 1:
        raise ValueError("trip_no must be >= 1.")

    direction = direction.strip().upper()
    manifest = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    required = {
        "区段",
        "列车运行方向",
        "segment",
        "来源run_id",
        "全局趟次候选ID",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Traceability manifest missing columns: {missing}")

    global_prefix = f"{direction}{trip_no:03d}_"
    selected = manifest[
        manifest["列车运行方向"].astype(str).str.strip().str.upper().eq(direction)
        & manifest["全局趟次候选ID"].astype(str).str.startswith(global_prefix)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"追溯表中找不到 {direction} 方向第 {trip_no} 趟 "
            f"(ID 前缀 {global_prefix})。"
        )

    global_ids = selected["全局趟次候选ID"].dropna().astype(str).unique().tolist()
    if len(global_ids) != 1:
        raise ValueError(
            f"{global_prefix} 匹配到多个全局趟次ID: {global_ids}"
        )
    global_trip_id = global_ids[0]

    duplicate_sections = selected[selected.duplicated("区段", keep=False)]["区段"].unique().tolist()
    if duplicate_sections:
        raise ValueError(
            f"全局趟次 {global_trip_id} 的区间映射不唯一: {duplicate_sections}"
        )

    records: dict[str, TraceabilityRecord] = {}
    for _, row in selected.iterrows():
        section = str(row["区段"]).strip()
        records[section] = TraceabilityRecord(
            section=section,
            segment=_normalize_segment(row["segment"]),
            source_run_id=str(row["来源run_id"]).strip(),
            global_trip_id=global_trip_id,
            absolute_start=str(row.get("绝对起始时间", "")).strip(),
            absolute_end=str(row.get("绝对结束时间", "")).strip(),
            trainset=str(row.get("来源车底号", "")).strip(),
        )

    return TripTraceability(
        manifest_path=manifest_path,
        trip_no=trip_no,
        direction=direction,
        global_trip_id=global_trip_id,
        records=records,
    )


def select_trip_rows(
    frame: pd.DataFrame,
    section: str,
    trip_index: int,
    traceability: TripTraceability | None,
) -> tuple[pd.DataFrame, int | float | str, TraceabilityRecord | None, str]:
    """Select the correct local segment, with legacy index fallback when unavailable."""

    if "segment" not in frame.columns:
        raise ValueError(f"{section} 数据缺少必要列: segment")

    if traceability is not None:
        record = traceability.record_for(section)
        target_segment = record.segment
        numeric_segments = pd.to_numeric(frame["segment"], errors="coerce")
        try:
            target_number = float(target_segment)
            mask = np.isclose(numeric_segments.to_numpy(dtype=float), target_number, equal_nan=False)
            mask = pd.Series(mask, index=frame.index)
        except (TypeError, ValueError):
            mask = frame["segment"].astype(str).str.strip().eq(str(target_segment).strip())

        selected = frame[mask].copy()
        if selected.empty:
            available = frame["segment"].dropna().unique().tolist()
            raise ValueError(
                f"{section} 找不到追溯表指定的 segment={target_segment}; "
                f"当前文件可用 segment={available[:20]}"
            )
        return selected, target_segment, record, "traceability"

    segments = sorted(frame["segment"].dropna().unique())
    if trip_index >= len(segments):
        raise IndexError(
            f"{section} 只有 {len(segments)} 个 segment，无法提取本地区间第 {trip_index + 1} 趟。"
        )
    target_segment = segments[trip_index]
    return (
        frame[frame["segment"] == target_segment].copy(),
        target_segment,
        None,
        "legacy_local_index",
    )


def traceability_metadata(
    record: TraceabilityRecord | None,
    target_segment: int | float | str,
    mode: str,
) -> dict[str, Any]:
    return {
        "全局趟次ID": record.global_trip_id if record else "",
        # Keep audit columns as text. Pandas 2.x string columns reject assigning
        # an int such as segment=5 without an explicit conversion.
        "历史segment": str(target_segment),
        "历史来源run_id": record.source_run_id if record else "",
        "趟次选择方式": mode,
    }


def print_traceability_summary(
    traceability: TripTraceability | None,
    trip_no: int,
) -> None:
    if traceability is None:
        print(
            f"WARNING: 第 {trip_no} 趟未使用全局追溯表，将回退为各区间独立第 {trip_no} 个 segment。"
        )
        return
    print("趟次选择方式: 全局运行链追溯")
    print(f"全局趟次ID:   {traceability.global_trip_id}")
    print(f"追溯表:       {traceability.manifest_path}")
    print(f"映射区间数:   {len(traceability.records)}")
