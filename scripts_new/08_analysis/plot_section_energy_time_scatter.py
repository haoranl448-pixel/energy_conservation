# -*- coding: utf-8 -*-
"""Plot normal-trip energy vs. runtime scatter charts for Step2 results.

Each `results_*.xlsx` file contains many trajectory rows per trip. This script
compresses every segment into one trip-level record:

- runtime = max(时刻) - min(时刻)
- energy = sum(energy) / 3600, assuming the raw `energy` column is Joules

The default figure shows every normal trip as a scatter point, then connects the
per-class median points. That keeps the time-energy trend readable when a class
contains many trips.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import zipfile
from datetime import datetime, time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_processed_step2_v3_first5_curve_quality"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "section_energy_time_scatter"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "output" / "cache" / "section_energy_time_scatter"
SECTION_PARAMS_FILE = PROJECT_ROOT / "data" / "static" / "section_params.csv"

CACHE_VERSION = "section_energy_time_scatter_v3_mass"

COL_SEGMENT = "segment"
COL_TIME = "时刻"
COL_ENERGY = "energy"
COL_MASS = "重量"
COL_CLASS = "运行等级"
COL_QUALITY = "曲线质量标签"
COL_AFTER_8PM = "是否是晚上八点以后的趟次"
COL_SERVICE = "服务号"
COL_DATE_SERVICE = "日期+服务号"
COL_DIRECTION = "列车运行方向"

CLASS_ORDER = ["class1", "class2", "class3", "class4", "class5"]
CLASS_COLORS = {
    "class1": "#1f77b4",
    "class2": "#ff7f0e",
    "class3": "#2ca02c",
    "class4": "#d62728",
    "class5": "#9467bd",
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制每个区间正常趟次的能耗-运行时间散点图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Step2 results_*.xlsx 所在目录。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图片和汇总表输出目录。")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="逐文件汇总缓存目录。")
    parser.add_argument("--pattern", default="results_*.xlsx", help="输入文件匹配模式。")
    parser.add_argument("--line-scope", default="full", help="full/all 或正向前 N 个区间，例如 5。")
    parser.add_argument("--normal-label", type=int, default=0, help="曲线质量标签中代表正常趟次的值。")
    parser.add_argument("--include-all-quality", action="store_true", help="不过滤曲线质量标签。")
    parser.add_argument("--day-only", action="store_true", help="额外要求 是否是晚上八点以后 == 0。")
    parser.add_argument(
        "--connect",
        choices=["class_median", "raw", "none"],
        default="class_median",
        help="连线方式：class_median 连接各等级中位点；raw 按时间连接原始散点；none 不连线。",
    )
    parser.add_argument("--jpg-dpi", type=int, default=220, help="JPG 输出 DPI。")
    parser.add_argument("--show-labels", action="store_true", help="在中位点旁标注等级和样本数。")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_class_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        num = float(match.group(1))
        if num.is_integer():
            return f"class{int(num)}"
    return text


def class_sort_key(label: str) -> tuple[int, str]:
    if label in CLASS_ORDER:
        return CLASS_ORDER.index(label), label
    return len(CLASS_ORDER), label


def first_non_empty(values: list[Any]) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            if str(value).strip() != "":
                return value
    return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def value_at(row: tuple[Any, ...], col_idx: int | None) -> Any:
    if col_idx is None or col_idx >= len(row):
        return None
    return row[col_idx]


def excel_col_to_index(cell_ref: str) -> int:
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def xml_tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    strings: list[str] = []
    with zf.open("xl/sharedStrings.xml") as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if xml_tag_name(elem.tag) == "si":
                parts = []
                for child in elem.iter():
                    if xml_tag_name(child.tag) == "t" and child.text is not None:
                        parts.append(child.text)
                strings.append("".join(parts))
                elem.clear()
    return strings


def cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        parts = []
        for child in cell.iter():
            if xml_tag_name(child.tag) == "t" and child.text is not None:
                parts.append(child.text)
        return "".join(parts)

    v_elem = None
    for child in cell:
        if xml_tag_name(child.tag) == "v":
            v_elem = child
            break
    if v_elem is None or v_elem.text is None:
        return None

    raw = v_elem.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    if cell_type in {"str", "b"}:
        return raw

    try:
        num = float(raw)
    except ValueError:
        return raw
    if num.is_integer():
        return int(num)
    return num


def iter_xlsx_rows(path: Path, wanted_cols: set[str]):
    """Yield sparse row dictionaries by reading the first worksheet XML directly."""

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in names:
            sheet_candidates = sorted(name for name in names if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
            if not sheet_candidates:
                raise ValueError(f"{path.name} 没有 worksheet XML。")
            sheet_name = sheet_candidates[0]

        shared_strings = read_shared_strings(zf)
        header_by_index: dict[int, str] | None = None
        wanted_indices: set[int] = set()

        with zf.open(sheet_name) as fh:
            for _, row_elem in ET.iterparse(fh, events=("end",)):
                if xml_tag_name(row_elem.tag) != "row":
                    continue

                row_values: dict[int, Any] = {}
                for cell in row_elem:
                    if xml_tag_name(cell.tag) != "c":
                        continue
                    ref = cell.attrib.get("r", "")
                    if not ref:
                        continue
                    col_idx = excel_col_to_index(ref)
                    if header_by_index is not None and col_idx not in wanted_indices:
                        continue
                    row_values[col_idx] = cell_value(cell, shared_strings)

                if header_by_index is None:
                    header_by_index = {
                        idx: str(value).strip()
                        for idx, value in row_values.items()
                        if value is not None and str(value).strip()
                    }
                    wanted_indices = {idx for idx, name in header_by_index.items() if name in wanted_cols}
                else:
                    yield {
                        header_by_index[idx]: value
                        for idx, value in row_values.items()
                        if idx in wanted_indices and idx in header_by_index
                    }
                row_elem.clear()


def get_cache_path(path: Path, cache_dir: Path) -> Path:
    stat = path.stat()
    key_src = "|".join([
        CACHE_VERSION,
        str(path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
    ])
    key = hashlib.sha1(key_src.encode("utf-8", errors="surrogatepass")).hexdigest()
    return cache_dir / f"{key}.pkl"


def read_trip_summary(path: Path, cache_dir: Path) -> pd.DataFrame:
    cache_path = get_cache_path(path, cache_dir)
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    print(f"读取并汇总: {path.name}")
    section = path.stem.replace("results_", "", 1)
    wanted_cols = {
        COL_SEGMENT,
        COL_TIME,
        COL_ENERGY,
        COL_MASS,
        COL_CLASS,
        COL_QUALITY,
        COL_AFTER_8PM,
        COL_SERVICE,
        COL_DATE_SERVICE,
        COL_DIRECTION,
    }
    rows_by_segment: dict[Any, dict[str, Any]] = {}

    saw_any_row = False
    for row in iter_xlsx_rows(path, wanted_cols):
        saw_any_row = True
        segment = row.get(COL_SEGMENT)
        if segment is None:
            continue

        t_val = to_float(row.get(COL_TIME))
        e_val = to_float(row.get(COL_ENERGY))
        if t_val is None:
            continue

        item = rows_by_segment.setdefault(
            segment,
                {
                    "section": section,
                    "segment": segment,
                    "time_min": t_val,
                    "time_max": t_val,
                    "energy_j": 0.0,
                    "mass_values": [],
                    "point_count": 0,
                "class_values": [],
                "quality_values": [],
                "after_8pm_values": [],
                "service_values": [],
                "date_service_values": [],
                "direction_values": [],
            },
        )
        item["time_min"] = min(item["time_min"], t_val)
        item["time_max"] = max(item["time_max"], t_val)
        item["energy_j"] += e_val or 0.0
        item["point_count"] += 1
        mass_val = to_float(row.get(COL_MASS))
        if mass_val is not None:
            item["mass_values"].append(mass_val)

        for source_col, target_key in [
            (COL_CLASS, "class_values"),
            (COL_QUALITY, "quality_values"),
            (COL_AFTER_8PM, "after_8pm_values"),
            (COL_SERVICE, "service_values"),
            (COL_DATE_SERVICE, "date_service_values"),
            (COL_DIRECTION, "direction_values"),
        ]:
            value = row.get(source_col)
            if value is not None:
                item[target_key].append(value)

    if not saw_any_row:
        raise ValueError(f"{path.name} 没有读到数据行。")

    records = []
    for item in rows_by_segment.values():
        duration_s = float(item["time_max"] - item["time_min"])
        mass_t = float(np.nanmean(item["mass_values"])) if item["mass_values"] else np.nan
        energy_wh = float(item["energy_j"]) / 3600.0
        quality = first_non_empty(item["quality_values"])
        after_8pm = first_non_empty(item["after_8pm_values"])
        records.append(
            {
                "section": item["section"],
                "segment": item["segment"],
                "duration_s": duration_s,
                "energy_wh": energy_wh,
                "mass_t": mass_t,
                "energy_wh_per_t": energy_wh / mass_t if mass_t and mass_t > 0 else np.nan,
                "run_class": normalize_class_label(first_non_empty(item["class_values"])),
                "quality_label": to_float(quality),
                "after_8pm": to_float(after_8pm),
                "service_no": first_non_empty(item["service_values"]),
                "date_service": first_non_empty(item["date_service_values"]),
                "direction": first_non_empty(item["direction_values"]),
                "point_count": int(item["point_count"]),
            }
        )

    df = pd.DataFrame(records)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    return df


def ordered_sections_from_params() -> list[str]:
    if not SECTION_PARAMS_FILE.exists():
        return []
    try:
        df = pd.read_csv(SECTION_PARAMS_FILE, usecols=["station_pair"])
    except Exception:
        return []
    return df["station_pair"].dropna().astype(str).tolist()


def select_input_files(data_dir: Path, pattern: str, line_scope: str) -> list[Path]:
    files = sorted(data_dir.glob(pattern), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"没有找到输入文件: {data_dir / pattern}")

    order = ordered_sections_from_params()
    if order:
        order_index = {sp: i for i, sp in enumerate(order)}
        files.sort(key=lambda p: order_index.get(p.stem.replace("results_", "", 1), len(order_index)))

    raw = str(line_scope).strip().lower()
    if raw in {"", "full", "all"}:
        return files
    if raw == "first5":
        raw = "5"
    n = int(raw)
    if n < 1:
        raise ValueError("--line-scope must be full/all or a positive integer.")

    if order:
        allowed = set(order[:n])
        return [p for p in files if p.stem.replace("results_", "", 1) in allowed]
    return files[:n]


def filter_normal_trips(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if not args.include_all_quality and "quality_label" in out.columns:
        out = out[out["quality_label"] == args.normal_label]
    if args.day_only and "after_8pm" in out.columns:
        out = out[(out["after_8pm"].isna()) | (out["after_8pm"] == 0)]
    out = out[(out["duration_s"] > 0) & (out["energy_wh"] > 0)]
    return out


def build_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (section, run_class), g in df.groupby(["section", "run_class"], dropna=False):
        rows.append(
            {
                "section": section,
                "run_class": run_class,
                "sample_count": len(g),
                "duration_median_s": g["duration_s"].median(),
                "duration_mean_s": g["duration_s"].mean(),
                "energy_median_wh": g["energy_wh"].median(),
                "energy_mean_wh": g["energy_wh"].mean(),
                "mass_median_t": g["mass_t"].median(),
                "mass_mean_t": g["mass_t"].mean(),
                "energy_per_t_median_wh_per_t": g["energy_wh_per_t"].median(),
                "energy_per_t_mean_wh_per_t": g["energy_wh_per_t"].mean(),
                "energy_per_second_median": (g["energy_wh"] / g["duration_s"]).median(),
            }
        )
    out = pd.DataFrame(rows)
    out["run_class_sort"] = out["run_class"].apply(class_sort_key)
    return out.sort_values(["section", "run_class_sort"]).drop(columns=["run_class_sort"])


def plot_section(
    df: pd.DataFrame,
    class_summary: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    y_col: str,
    y_summary_col: str,
    y_label: str,
    title_label: str,
) -> None:
    section = str(df["section"].iloc[0])
    df = df.dropna(subset=[y_col]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for run_class in sorted(df["run_class"].dropna().unique(), key=class_sort_key):
        g = df[df["run_class"] == run_class].sort_values("duration_s")
        color = CLASS_COLORS.get(run_class, "#7f7f7f")
        ax.scatter(
            g["duration_s"],
            g[y_col],
            s=30,
            alpha=0.72,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            label=f"{run_class} ({len(g)})",
        )
        if args.connect == "raw" and len(g) >= 2:
            ax.plot(g["duration_s"], g[y_col], color=color, alpha=0.32, linewidth=1.0)

    med = class_summary[class_summary["section"] == section].copy()
    med = med.dropna(subset=[y_summary_col])
    if args.connect == "class_median" and not med.empty:
        med = med.sort_values("duration_median_s")
        ax.plot(
            med["duration_median_s"],
            med[y_summary_col],
            color="#111111",
            linewidth=1.8,
            marker="o",
            markersize=5.5,
            label="class median trend",
        )
        if args.show_labels:
            for _, row in med.iterrows():
                ax.annotate(
                    f"{row['run_class']} n={int(row['sample_count'])}",
                    (row["duration_median_s"], row[y_summary_col]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.set_title(f"{section} 正常趟次 {title_label}-运行时间")
    ax.set_xlabel("运行时间 (s)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.jpg_dpi)
    plt.close(fig)


def plot_overview(
    df: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    y_col: str,
    y_summary_col: str,
    y_label: str,
    title_label: str,
) -> None:
    df = df.dropna(subset=[y_col]).copy()
    sections = list(dict.fromkeys(df["section"].tolist()))
    if not sections:
        return
    ncols = 2 if len(sections) <= 6 else 3
    nrows = int(math.ceil(len(sections) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.4 * nrows), squeeze=False)

    for ax, section in zip(axes.ravel(), sections):
        g_section = df[df["section"] == section]
        for run_class in sorted(g_section["run_class"].dropna().unique(), key=class_sort_key):
            g = g_section[g_section["run_class"] == run_class].sort_values("duration_s")
            color = CLASS_COLORS.get(run_class, "#7f7f7f")
            ax.scatter(g["duration_s"], g[y_col], s=18, alpha=0.65, color=color, label=run_class)
        med = build_class_summary(g_section)
        med = med.dropna(subset=[y_summary_col]) if not med.empty else med
        if not med.empty:
            med = med.sort_values("duration_median_s")
            ax.plot(med["duration_median_s"], med[y_summary_col], color="#111111", linewidth=1.2, marker="o", markersize=4)
        ax.set_title(section, fontsize=11)
        ax.set_xlabel("运行时间 (s)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.2)

    for ax in axes.ravel()[len(sections):]:
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), fontsize=9)
    fig.suptitle(f"正常趟次 {title_label}-运行时间散点图", y=0.995, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.jpg_dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    cache_dir = resolve_path(args.cache_dir)

    files = select_input_files(data_dir, args.pattern, args.line_scope)
    print(f"输入目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理区间数: {len(files)}")

    summaries = []
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}")
        summaries.append(read_trip_summary(path, cache_dir))

    all_trips = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    normal_trips = filter_normal_trips(all_trips, args)
    class_summary = build_class_summary(normal_trips)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_trips.to_csv(output_dir / "all_trip_energy_time_summary.csv", index=False, encoding="utf-8-sig")
    normal_trips.to_csv(output_dir / "normal_trip_energy_time_summary.csv", index=False, encoding="utf-8-sig")
    class_summary.to_csv(output_dir / "normal_class_energy_time_summary.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(output_dir / "energy_time_summary.xlsx", engine="openpyxl") as writer:
            all_trips.to_excel(writer, sheet_name="all_trips", index=False)
            normal_trips.to_excel(writer, sheet_name="normal_trips", index=False)
            class_summary.to_excel(writer, sheet_name="class_summary", index=False)
    except PermissionError:
        fallback = output_dir / "energy_time_summary_unlocked.xlsx"
        with pd.ExcelWriter(fallback, engine="openpyxl") as writer:
            all_trips.to_excel(writer, sheet_name="all_trips", index=False)
            normal_trips.to_excel(writer, sheet_name="normal_trips", index=False)
            class_summary.to_excel(writer, sheet_name="class_summary", index=False)
        print(f"Excel 被占用，已写入: {fallback}")

    for section, g in normal_trips.groupby("section", sort=False):
        safe_name = re.sub(r'[<>:"/\\\\|?*]+', "_", str(section))
        plot_section(
            g,
            class_summary,
            output_dir / f"{safe_name}_energy_time_scatter.jpg",
            args,
            y_col="energy_wh",
            y_summary_col="energy_median_wh",
            y_label="实测能耗 (Wh)",
            title_label="能耗",
        )
        plot_section(
            g,
            class_summary,
            output_dir / f"{safe_name}_energy_per_t_time_scatter.jpg",
            args,
            y_col="energy_wh_per_t",
            y_summary_col="energy_per_t_median_wh_per_t",
            y_label="单位质量能耗 (Wh/t)",
            title_label="单位质量能耗",
        )

    plot_overview(
        normal_trips,
        output_dir / "all_sections_energy_time_overview.jpg",
        args,
        y_col="energy_wh",
        y_summary_col="energy_median_wh",
        y_label="能耗 (Wh)",
        title_label="能耗",
    )
    plot_overview(
        normal_trips,
        output_dir / "all_sections_energy_per_t_time_overview.jpg",
        args,
        y_col="energy_wh_per_t",
        y_summary_col="energy_per_t_median_wh_per_t",
        y_label="单位质量能耗 (Wh/t)",
        title_label="单位质量能耗",
    )

    print(f"完成。正常趟次: {len(normal_trips)} / 全部趟次: {len(all_trips)}")
    print(f"图片和表格: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
