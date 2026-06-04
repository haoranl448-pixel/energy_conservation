# -*- coding: utf-8 -*-
"""
Generate a historical operation summary table for one trip and selected sections.

The script reads processed `results_<section>.xlsx` files, extracts the requested
trip by segment order, and writes a section-level summary with a final total row.
It does not run the residual model; energy is the raw measured historical energy
computed from the `energy` column, using the same unit conversion as step 7.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "historical_trip_section_report"

FULL_LINE_STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥",
]

REPORT_SOURCE_COLS = [
    "区段", "站间距", "时段划分", "日期+服务号", "服务号", "车底号", "列车运行方向",
    "时刻", "segment", "energy", "速度(m/s)", "累计位移(m)", "重量", "运行等级",
    "是否是晚上八点以后的趟次", "曲线质量标签",
]


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def first_valid(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def mode_or_first(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    counts = values.value_counts()
    if counts.empty:
        return values.iloc[0]
    return counts.index[0]


def sort_segments(values: Iterable) -> list:
    def key(value):
        try:
            return 0, float(value)
        except (TypeError, ValueError):
            return 1, str(value)

    return sorted(pd.Series(list(values)).dropna().unique().tolist(), key=key)


def choose_sections(args: argparse.Namespace) -> list[str]:
    sections = parse_csv_list(args.sections)
    if sections:
        return sections

    scope = str(args.line_scope).strip().lower()
    if scope in {"full", "all"}:
        return FULL_LINE_STATIONS

    try:
        n_sections = int(scope)
    except ValueError as exc:
        raise ValueError("--line-scope must be 'full' or a positive integer.") from exc
    if n_sections < 1:
        raise ValueError("--line-scope must be >= 1.")
    if n_sections > len(FULL_LINE_STATIONS):
        raise ValueError(f"--line-scope cannot exceed {len(FULL_LINE_STATIONS)}.")
    return FULL_LINE_STATIONS[:n_sections]


def read_trip_section(data_dir: Path, section: str, trip_no: int) -> dict:
    file_path = data_dir / f"results_{section}.xlsx"
    if not file_path.exists():
        raise FileNotFoundError(f"未找到区间数据文件: {file_path}")

    header = pd.read_excel(file_path, nrows=0)
    usecols = [col for col in REPORT_SOURCE_COLS if col in header.columns]
    df_all = pd.read_excel(file_path, usecols=usecols)
    if "segment" not in df_all.columns:
        raise ValueError(f"{file_path} 缺少 segment 列。")

    segments = sort_segments(df_all["segment"])
    trip_index = trip_no - 1
    if trip_index < 0:
        raise ValueError("--trip-no must be >= 1.")
    if trip_index >= len(segments):
        raise IndexError(f"{section} 只有 {len(segments)} 个 segment，无法提取第 {trip_no} 趟。")

    segment_id = segments[trip_index]
    df = df_all[df_all["segment"] == segment_id].copy()
    if df.empty:
        raise ValueError(f"{section} 第 {trip_no} 趟没有有效数据。")

    numeric_cols = [
        "时刻", "速度(m/s)", "累计位移(m)", "energy", "重量", "站间距",
        "加速度(m/s²)", "gradient", "curvature",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "时刻" not in df.columns:
        raise ValueError(f"{file_path} 缺少 时刻 列。")

    df = df.sort_values("时刻")
    time_values = df["时刻"].dropna()
    if time_values.empty:
        raise ValueError(f"{section} 第 {trip_no} 趟没有有效时刻。")

    start_time = float(time_values.iloc[0])
    end_time = float(time_values.iloc[-1])
    run_time = end_time - start_time

    if "累计位移(m)" in df.columns and df["累计位移(m)"].notna().any():
        dist_values = df["累计位移(m)"].dropna()
        distance = float(dist_values.max() - dist_values.min())
    else:
        distance = np.nan

    energy_wh = np.nan
    if "energy" in df.columns:
        energy_wh = float(df["energy"].dropna().sum() / 3.6e6 * 1000.0)

    speed_kmh = df["速度(m/s)"] * 3.6 if "速度(m/s)" in df.columns else pd.Series(dtype=float)
    max_speed = float(speed_kmh.max()) if not speed_kmh.dropna().empty else np.nan
    avg_speed = distance / run_time * 3.6 if pd.notna(distance) and run_time > 0 else np.nan

    station_distance = first_valid(df["站间距"]) if "站间距" in df.columns else np.nan
    service_id = first_valid(df["服务号"]) if "服务号" in df.columns else np.nan
    train_id = first_valid(df["车底号"]) if "车底号" in df.columns else np.nan
    date_service = first_valid(df["日期+服务号"]) if "日期+服务号" in df.columns else np.nan
    direction = first_valid(df["列车运行方向"]) if "列车运行方向" in df.columns else np.nan
    period = first_valid(df["时段划分"]) if "时段划分" in df.columns else np.nan
    weight = first_valid(df["重量"]) if "重量" in df.columns else np.nan
    run_class = mode_or_first(df["运行等级"]) if "运行等级" in df.columns else np.nan
    night_flag = mode_or_first(df["是否是晚上八点以后的趟次"]) if "是否是晚上八点以后的趟次" in df.columns else np.nan
    quality_label = mode_or_first(df["曲线质量标签"]) if "曲线质量标签" in df.columns else np.nan

    return {
        "站间区间": section,
        "趟次": trip_no,
        "segment": segment_id,
        "日期+服务号": date_service,
        "服务号": service_id,
        "车底号": train_id,
        "方向": direction,
        "时段": period,
        "开始时刻(s)": round(start_time, 2),
        "结束时刻(s)": round(end_time, 2),
        "历史运行时间(s)": round(run_time, 2),
        "区间距离(m)": round(distance, 2) if pd.notna(distance) else np.nan,
        "站间距(m)": round(float(station_distance), 2) if pd.notna(station_distance) else np.nan,
        "平均速度(km/h)": round(avg_speed, 2) if pd.notna(avg_speed) else np.nan,
        "最高速度(km/h)": round(max_speed, 2) if pd.notna(max_speed) else np.nan,
        "历史实测能耗(Wh)": round(energy_wh, 2) if pd.notna(energy_wh) else np.nan,
        "重量(t)": round(float(weight), 2) if pd.notna(weight) else np.nan,
        "运行等级": run_class,
        "是否晚上八点后": night_flag,
        "曲线质量标签": quality_label,
        "源文件": str(file_path.relative_to(PROJECT_ROOT)),
    }


def append_total_row(df: pd.DataFrame) -> pd.DataFrame:
    total_run = df["历史运行时间(s)"].sum(skipna=True)
    total_distance = df["区间距离(m)"].sum(skipna=True)
    total_station_distance = df["站间距(m)"].sum(skipna=True)
    total_energy = df["历史实测能耗(Wh)"].sum(skipna=True)
    max_speed = df["最高速度(km/h)"].max(skipna=True)
    avg_speed = total_distance / total_run * 3.6 if total_run > 0 else np.nan

    total_row = {col: "" for col in df.columns}
    total_row.update({
        "站间区间": "--- 总计 ---",
        "历史运行时间(s)": round(total_run, 2),
        "区间距离(m)": round(total_distance, 2),
        "站间距(m)": round(total_station_distance, 2),
        "平均速度(km/h)": round(avg_speed, 2) if pd.notna(avg_speed) else np.nan,
        "最高速度(km/h)": round(max_speed, 2) if pd.notna(max_speed) else np.nan,
        "历史实测能耗(Wh)": round(total_energy, 2),
    })
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


def write_outputs(df: pd.DataFrame, output_dir: Path, output_name: str) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_name}.csv"
    xlsx_path = output_dir / f"{output_name}.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="历史运行情况")
            ws = writer.book["历史运行情况"]
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 32)
    except Exception as exc:
        print(f"⚠️ XLSX 输出失败，仅保留 CSV。原因: {exc}")
        xlsx_path = None

    return csv_path, xlsx_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成某一趟、某几个区间的历史运行情况表。")
    parser.add_argument("--trip-no", type=int, default=1, help="第几趟车，1 表示第 1 个 segment。默认 1。")
    parser.add_argument(
        "--sections",
        help="逗号分隔的区间名，例如：布政-张家潭,张家潭-同德路。优先级高于 --line-scope。",
    )
    parser.add_argument(
        "--line-scope",
        default="full",
        help="未指定 --sections 时使用。可填 full/all 或正向前 N 个区间，例如 5。默认 full。",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="results_区间.xlsx 所在目录。相对路径会按项目根目录解析。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录。相对路径会按项目根目录解析。",
    )
    parser.add_argument("--output-name", help="输出文件名，不含扩展名；不填则自动生成。")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    sections = choose_sections(args)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    rows = []
    failures = []
    print(f"历史数据目录: {data_dir}")
    print(f"趟次: 第 {args.trip_no} 趟 (segment index {args.trip_no - 1})")
    print(f"区间数: {len(sections)}")

    for idx, section in enumerate(sections, 1):
        print(f"[{idx}/{len(sections)}] {section}")
        try:
            rows.append(read_trip_section(data_dir, section, args.trip_no))
        except Exception as exc:
            failures.append({"站间区间": section, "错误": str(exc)})
            print(f"  ⚠️ 跳过: {exc}")

    if not rows:
        raise RuntimeError("没有生成任何有效行，请检查趟次、区间名或 data-dir。")

    df = append_total_row(pd.DataFrame(rows))
    if args.output_name:
        output_name = args.output_name
    else:
        scope_name = "sections" if args.sections else f"scope_{args.line_scope}"
        output_name = f"historical_trip{args.trip_no}_{scope_name}"

    csv_path, xlsx_path = write_outputs(df, output_dir, output_name)
    print("\n生成完成:")
    print(f"CSV:  {csv_path}")
    if xlsx_path:
        print(f"XLSX: {xlsx_path}")

    if failures:
        failures_path = output_dir / f"{output_name}_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False, encoding="utf-8-sig")
        print(f"失败明细: {failures_path}")

    print("\n结果预览:")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
