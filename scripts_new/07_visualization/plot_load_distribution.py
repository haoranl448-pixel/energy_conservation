# -*- coding: utf-8 -*-
"""Plot full-line load distributions and summarize section trip coverage.

The traceability manifest is the authoritative source for per-section trip
counts. Historical reports from the 1--125 batch are used as a lightweight
mass cache and are joined back to the manifest by section and segment index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_processed_step2_v3_all_curve_quality_traceability"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "output" / "schedule" / "batch_trip_reports_trip1_125"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "load_analysis"

LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--direction", default="UP", choices=["UP", "DOWN"])
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })


def load_manifest(data_dir: Path, direction: str) -> pd.DataFrame:
    manifest_path = data_dir / "trip_traceability_manifest_v1.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Traceability manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "区段", "列车运行方向", "segment", "来源run_id", "全局趟次候选ID",
        "是否追溯到线路首区间", "全趟夜间标记_候选", "连续匹配是否有歧义",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    manifest = manifest[
        manifest["列车运行方向"].astype(str).str.upper().eq(direction)
        & manifest["区段"].isin(LINE5_SECTIONS)
    ].copy()
    manifest["segment"] = pd.to_numeric(manifest["segment"], errors="coerce")
    manifest = manifest.dropna(subset=["segment", "来源run_id"])
    manifest["segment"] = manifest["segment"].astype(int)
    return manifest


def summarize_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    summary = (
        manifest.groupby("区段", as_index=False)
        .agg(
            区间提取趟数=("来源run_id", "nunique"),
            全局趟次候选数=("全局趟次候选ID", "nunique"),
            可追溯至线路首区间趟数=("是否追溯到线路首区间", "sum"),
            夜间候选趟数=("全趟夜间标记_候选", "sum"),
            连续匹配歧义趟数=("连续匹配是否有歧义", "sum"),
        )
        .set_index("区段").reindex(LINE5_SECTIONS).reset_index()
    )
    numeric_columns = summary.columns.drop("区段")
    summary[numeric_columns] = summary[numeric_columns].fillna(0).astype(int)
    first_count = int(summary.loc[0, "区间提取趟数"])
    summary.insert(0, "序号", np.arange(1, len(summary) + 1))
    summary["相对首区间增减"] = summary["区间提取趟数"] - first_count
    summary["未追溯至线路首区间趟数"] = (
        summary["区间提取趟数"] - summary["可追溯至线路首区间趟数"]
    )
    return summary


def load_mass_cache(report_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for report_path in sorted(report_dir.glob("trip*/Historical_Only_Report.csv")):
        frame = pd.read_csv(report_path, usecols=["站间区间", "segment", "平均重量(t)"])
        frame = frame[frame["站间区间"].isin(LINE5_SECTIONS)].copy()
        frame["segment"] = pd.to_numeric(frame["segment"], errors="coerce")
        frame["平均重量(t)"] = pd.to_numeric(frame["平均重量(t)"], errors="coerce")
        frame = frame.dropna(subset=["segment", "平均重量(t)"])
        frame["segment"] = frame["segment"].astype(int)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No Historical_Only_Report.csv files in {report_dir}")
    mass = pd.concat(frames, ignore_index=True)
    return mass.drop_duplicates(["站间区间", "segment"], keep="first")


def build_mass_records(manifest: pd.DataFrame, mass_cache: pd.DataFrame) -> pd.DataFrame:
    mapping = manifest[["区段", "segment", "来源run_id", "全局趟次候选ID"]].drop_duplicates(
        ["区段", "segment"], keep="first"
    )
    records = mass_cache.merge(
        mapping,
        left_on=["站间区间", "segment"],
        right_on=["区段", "segment"],
        how="left",
        validate="one_to_one",
    )
    records["区间序号"] = records["站间区间"].map(
        {section: index for index, section in enumerate(LINE5_SECTIONS)}
    )
    records = records.dropna(subset=["区间序号", "全局趟次候选ID"]).copy()
    records["区间序号"] = records["区间序号"].astype(int)
    return records


def add_mass_statistics(counts: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    mass_stats = (
        records.groupby("站间区间")["平均重量(t)"]
        .agg(
            载重可用样本数="count",
            最小载重_t="min",
            P25载重_t=lambda values: values.quantile(0.25),
            中位载重_t="median",
            P75载重_t=lambda values: values.quantile(0.75),
            最大载重_t="max",
        )
        .reset_index().rename(columns={"站间区间": "区段"})
    )
    result = counts.merge(mass_stats, on="区段", how="left")
    mass_columns = ["最小载重_t", "P25载重_t", "中位载重_t", "P75载重_t", "最大载重_t"]
    result[mass_columns] = result[mass_columns].round(2)
    result["载重可用样本数"] = result["载重可用样本数"].fillna(0).astype(int)
    return result


def plot_distribution(records: pd.DataFrame, counts: pd.DataFrame, output_dir: Path) -> None:
    configure_plotting()
    x = np.arange(len(LINE5_SECTIONS))
    pivot = records.pivot_table(
        index="全局趟次候选ID", columns="区间序号", values="平均重量(t)", aggfunc="mean"
    ).reindex(columns=x)
    q10, q25, q50, q75, q90 = [pivot.quantile(q) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
    maximum = pivot.max()

    fig = plt.figure(figsize=(15.5, 8.7), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.08)
    ax = fig.add_subplot(grid[0])
    ax_count = fig.add_subplot(grid[1], sharex=ax)

    for row_index, (_, row) in enumerate(pivot.iterrows()):
        ax.plot(
            x,
            row.to_numpy(dtype=float),
            color="#66778A",
            linewidth=0.75,
            alpha=0.30,
            label="单趟载重轨迹" if row_index == 0 else None,
            zorder=1,
        )
    ax.fill_between(x, q10, q90, color="#D9E7F5", alpha=0.78, label="中间80%趟次（P10–P90）", zorder=2)
    ax.fill_between(x, q25, q75, color="#91B9DF", alpha=0.58, label="中间50%趟次（P25–P75）", zorder=3)
    ax.plot(x, q50, color="#174A7E", linewidth=2.3, marker="o", markersize=3.7, label="中位载重", zorder=4)
    ax.plot(x, maximum, color="#D97706", linewidth=1.7, linestyle="--", marker="o", markersize=3.2, label="最大载重", zorder=4)
    ax.set_ylabel("列车质量 (t)", fontsize=10)
    ax.grid(axis="y", color="#D7DEE7", linewidth=0.7, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.legend(loc="upper left", ncol=5, frameon=False, fontsize=8.2)
    ax.text(
        0.995, 0.02,
        f"载重轨迹：125 个首区间趟次，断点后拆分为 {pivot.shape[0]} 个可追溯片段",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#64748B",
    )

    trip_counts = counts["区间提取趟数"].to_numpy(dtype=int)
    bar_colors = np.where(trip_counts < 100, "#C94B40", np.where(trip_counts < 125, "#E9A23B", "#4C78A8"))
    bars = ax_count.bar(x, trip_counts, width=0.72, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_count.axhline(125, color="#64748B", linestyle="--", linewidth=0.9, alpha=0.8)
    ax_count.set_ylabel("提取趟数", fontsize=9)
    ax_count.set_ylim(0, max(136, int(trip_counts.max()) + 9))
    ax_count.set_yticks([0, 50, 100, 125])
    ax_count.grid(axis="y", color="#D7DEE7", linewidth=0.7, alpha=0.75)
    ax_count.grid(axis="x", visible=False)
    ax_count.bar_label(bars, labels=[str(value) for value in trip_counts], padding=2, fontsize=7, color="#334155")
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(LINE5_SECTIONS, rotation=55, ha="right", fontsize=7.5, rotation_mode="anchor")

    fig.suptitle("全线区间载重分布与数据覆盖（最新可追溯数据集）", fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.947,
        "细线表示单趟可追溯载重轨迹；分位带概括总体分布；下图按来源 run_id 去重统计区间提取趟数",
        ha="center", fontsize=9.5, color="#526273",
    )
    fig.text(
        0.01, 0.008,
        "注：兴海南路—梅堰及梅堰—永茂路样本明显不足；后续区间超过125的记录包含断点后形成的独立候选。",
        ha="left", fontsize=7.5, color="#64748B",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.205)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "train_load_distribution_latest"
    fig.savefig(prefix.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.data_dir, args.direction)
    counts = summarize_counts(manifest)
    mass_cache = load_mass_cache(args.report_dir)
    mass_records = build_mass_records(manifest, mass_cache)
    summary = add_mass_statistics(counts, mass_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "section_trip_counts_latest.csv", index=False, encoding="utf-8-sig")
    mass_records.to_csv(args.output_dir / "load_distribution_source_latest.csv", index=False, encoding="utf-8-sig")
    plot_distribution(mass_records, summary, args.output_dir)

    metadata = {
        "direction": args.direction,
        "section_count": len(summary),
        "manifest_section_records": int(len(manifest)),
        "line_origin_trip_count": int(summary.loc[0, "区间提取趟数"]),
        "minimum_section_trip_count": int(summary["区间提取趟数"].min()),
        "maximum_section_trip_count": int(summary["区间提取趟数"].max()),
        "mass_record_count": int(len(mass_records)),
        "mass_candidate_count": int(mass_records["全局趟次候选ID"].nunique()),
    }
    (args.output_dir / "load_distribution_summary_latest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
