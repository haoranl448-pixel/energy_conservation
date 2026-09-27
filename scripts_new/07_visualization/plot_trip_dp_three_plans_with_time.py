# -*- coding: utf-8 -*-
"""Redraw the three saved DP plans with explicit run and dwell-time totals."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DP_DIR = ROOT / "output" / "schedule" / "batch_trip_reports_complete13_dp_unified"
DEFAULT_DATA_DIR = ROOT / "data" / "data_processed_step2_v3_all_curve_quality_traceability"
DP_MODULE_PATH = ROOT / "scripts_new" / "00_main_pipeline" / "08_ato_class_globall_v2.py"

PLAN_SPECS = [
    ("01_standard_priority.csv", "标准规划", "01_standard_priority_with_time"),
    ("02_energy_first.csv", "能耗优先规划", "02_energy_first_with_time"),
    ("03_real_priority_dwell_5pct.csv", "停站时间放宽5%规划", "03_real_priority_dwell_5pct_with_time"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", type=int, default=6)
    parser.add_argument("--dp-dir", type=Path, default=DEFAULT_DP_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_dp_module(trip_no: int, data_dir: Path, target_time: float):
    pipeline_dir = DP_MODULE_PATH.parent
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    os.environ["ENERGY_TRIP_NO"] = str(trip_no)
    os.environ["ENERGY_TARGET_TIME"] = str(target_time)
    os.environ["ENERGY_RESULTS_DATA_DIR"] = str(data_dir)
    spec = importlib.util.spec_from_file_location("dp_plot_source", DP_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load DP module: {DP_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def total_row(frame: pd.DataFrame) -> pd.Series:
    mask = frame["站间区间"].astype(str).str.contains("总计", na=False)
    if not mask.any():
        raise ValueError("DP result has no total row")
    return frame.loc[mask].iloc[-1]


def build_plot_data(rows: pd.DataFrame, dp_module) -> tuple[dict[str, list[float]], list[tuple[float, float]], float]:
    values = {
        "plan_t": [], "plan_v": [], "plan_s": [],
        "history_t": [], "history_v": [], "history_s": [],
    }
    dwell_zones: list[tuple[float, float]] = []
    plan_t0 = history_t0 = plan_s0 = history_s0 = 0.0

    for index, row in rows.reset_index(drop=True).iterrows():
        section = str(row["站间区间"])
        run_class = str(row["选定等级"])
        curve_path = dp_module.TRAJ_BASE_DIR / section / f"{run_class}_generated_curve.csv"
        curve = pd.read_csv(curve_path)
        plan_t = pd.to_numeric(curve["time_s"], errors="coerce").to_numpy()
        plan_v = pd.to_numeric(curve["velocity_mps"], errors="coerce").to_numpy() * 3.6
        plan_s = pd.to_numeric(curve["dist_m"], errors="coerce").to_numpy()
        values["plan_t"].extend((plan_t + plan_t0).tolist())
        values["plan_v"].extend(plan_v.tolist())
        values["plan_s"].extend((plan_s + plan_s0).tolist())

        history = dp_module.load_history_trip_curve(section)
        history_t = pd.to_numeric(history["时刻"], errors="coerce").to_numpy()
        history_t = history_t - history_t[0]
        history_v = pd.to_numeric(history["速度(m/s)"], errors="coerce").to_numpy() * 3.6
        history_s = pd.to_numeric(history["累计位移(m)"], errors="coerce").to_numpy()
        history_s = history_s - history_s[0]
        values["history_t"].extend((history_t + history_t0).tolist())
        values["history_v"].extend(history_v.tolist())
        values["history_s"].extend((history_s + history_s0).tolist())

        plan_t0 += float(row["规划用时(s)"])
        history_t0 += float(row["历史用时(s)"])
        plan_s0 += float(plan_s[-1])
        history_s0 += float(history_s[-1])

        if index < len(rows) - 1:
            plan_dwell = float(row["停站时间(s)"])
            history_dwell = float(row["历史停站时间(s)"])
            dwell_zones.append((plan_t0, plan_t0 + plan_dwell))
            values["plan_t"].extend([plan_t0, plan_t0 + plan_dwell])
            values["plan_v"].extend([0.0, 0.0])
            values["plan_s"].extend([plan_s0, plan_s0])
            plan_t0 += plan_dwell
            values["history_t"].extend([history_t0, history_t0 + history_dwell])
            values["history_v"].extend([0.0, 0.0])
            values["history_s"].extend([history_s0, history_s0])
            history_t0 += history_dwell

    return values, dwell_zones, plan_s0


def draw_plan(frame: pd.DataFrame, label: str, output_stem: Path, trip_no: int, dp_module) -> None:
    summary = total_row(frame)
    rows = frame[~frame["站间区间"].astype(str).str.contains("总计", na=False)].copy()
    data, dwell_zones, total_distance = build_plot_data(rows, dp_module)

    run_time = float(summary["规划用时(s)"])
    dwell_time = float(summary["停站时间(s)"])
    total_time = run_time + dwell_time
    history_total = float(summary["历史用时(s)"])
    history_dwell = float(summary["历史停站时间(s)"])
    history_run = history_total - history_dwell
    energy_saving_kwh = float(summary["节能量(Wh)"]) / 1000.0
    saving_text = str(summary["选定等级"])
    saving_pct = saving_text.split(":", 1)[-1].strip() if ":" in saving_text else saving_text

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.8), constrained_layout=False)
    plan_color = "#D94841"
    history_color = "#9AA4AE"

    fig.suptitle(
        f"Trip {trip_no:03d} | {label} | 总时间 {total_time:.1f} s = "
        f"运行 {run_time:.1f} s + 停站 {dwell_time:.1f} s",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"历史：总时间 {history_total:.1f} s = 运行 {history_run:.1f} s + 停站 {history_dwell:.1f} s"
        f"    |    规划节能 {energy_saving_kwh:.2f} kWh（{saving_pct}）",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )

    ax_time, ax_distance = axes
    ax_time.plot(data["history_t"], data["history_v"], color=history_color, alpha=0.55,
                 linewidth=0.9, label="历史运行曲线")
    ax_time.plot(data["plan_t"], data["plan_v"], color=plan_color, linewidth=1.35,
                 label=label)
    for start, end in dwell_zones:
        ax_time.axvspan(start, end, color="#64748B", alpha=0.055, linewidth=0)
    ax_time.set_title("a  速度-时间", loc="left", fontsize=11, fontweight="bold")
    ax_time.set_xlabel("时间 (s)")
    ax_time.set_ylabel("速度 (km/h)")
    ax_time.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax_time.grid(True, color="#D7DEE7", linewidth=0.6, alpha=0.75)

    ax_distance.plot(data["history_s"], data["history_v"], color=history_color, alpha=0.55,
                     linewidth=0.9, label="历史运行曲线")
    ax_distance.plot(data["plan_s"], data["plan_v"], color=plan_color, linewidth=1.35,
                     label=label)
    ax_distance.set_title(f"b  速度-里程 | 规划里程 {total_distance / 1000:.3f} km",
                          loc="left", fontsize=11, fontweight="bold")
    ax_distance.set_xlabel("里程 (m)")
    ax_distance.set_ylabel("速度 (km/h)")
    ax_distance.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax_distance.grid(True, color="#D7DEE7", linewidth=0.6, alpha=0.75)

    fig.text(
        0.99,
        0.012,
        "灰色阴影表示规划停站时段",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#64748B",
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.075, right=0.985, hspace=0.34)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_plotting()
    trip_dir = args.dp_dir / f"trip{args.trip:03d}"
    output_dir = args.output_dir or (trip_dir / "plots_with_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    first_frame = pd.read_csv(trip_dir / PLAN_SPECS[0][0])
    first_summary = total_row(first_frame)
    target_time = float(first_summary["历史用时(s)"])
    dp_module = load_dp_module(args.trip, args.data_dir, target_time)

    for filename, label, stem in PLAN_SPECS:
        frame = pd.read_csv(trip_dir / filename)
        draw_plan(frame, label, output_dir / stem, args.trip, dp_module)
        print(f"Saved: {output_dir / (stem + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
