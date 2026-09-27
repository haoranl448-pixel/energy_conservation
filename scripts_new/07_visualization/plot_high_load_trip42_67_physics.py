# -*- coding: utf-8 -*-
"""Redraw two high-load dwell-5% physics comparisons from frozen replay data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "output/cache/simu_code_replay/batch_simu_code_trip_compare_complete25"
SCHEDULE = ROOT / "output/schedule/batch_trip_reports_complete13_dp_unified"
OUTPUT = ROOT / "output/analysis/high_load_trip42_67_redraw"
TRIPS = (42, 67)


def load_curve(stats: pd.DataFrame, trip_no: int, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sections = stats.loc[stats.trip_no.eq(trip_no) & stats.scenario.eq(scenario)]
    sections = sections.sort_values("section_index")
    if sections.section_index.tolist() != list(range(1, 27)):
        raise ValueError(f"Incomplete sections: trip {trip_no}, {scenario}")
    frames = []
    physical_offset = measured_offset = 0.0
    for index, row in enumerate(sections.itertuples(index=False)):
        source = CACHE / "replay_data" / f"trip{trip_no:03d}_{scenario}_{index:02d}.pkl"
        frame = pd.read_pickle(source).copy()
        frame["physical_cumulative_kwh"] = physical_offset + frame.simu_section_kwh
        frame["measured_cumulative_kwh"] = measured_offset + frame.code_section_kwh
        frames.append(frame)
        physical_offset += float(row.simu_kwh)
        measured_offset += float(row.code_kwh)
    points = pd.concat(frames, ignore_index=True)
    if not np.isclose(points.physical_cumulative_kwh.iloc[-1], physical_offset, atol=1e-4):
        raise ValueError(f"Physics total mismatch: trip {trip_no}, {scenario}")
    if not np.isclose(points.measured_cumulative_kwh.iloc[-1], measured_offset, atol=1e-4):
        raise ValueError(f"Code/measured total mismatch: trip {trip_no}, {scenario}")
    return points, sections


def trip_data(stats: pd.DataFrame, trip_no: int) -> dict:
    history, history_sections = load_curve(stats, trip_no, "history")
    plan, plan_sections = load_curve(stats, trip_no, "dwell5_dp")
    timetable = pd.read_csv(SCHEDULE / f"trip{trip_no:03d}" / "03_real_priority_dwell_5pct.csv",
                            encoding="utf-8-sig")
    if len(timetable) != 27:
        raise ValueError(f"Incomplete timetable: trip {trip_no}")
    planned_run = float(timetable.iloc[:26, 2].sum())
    planned_dwell = float(timetable.iloc[:26, 3].sum())
    history_run = float(history_sections.history_runtime_s.sum())
    history_dwell = float(history_sections.history_dwell_s.sum())
    if not np.isclose(planned_run, plan_sections.runtime_s.sum(), atol=0.11):
        raise ValueError(f"Planned runtime mismatch: trip {trip_no}")
    if not np.isclose(planned_run + planned_dwell, history_run + history_dwell, atol=0.11):
        raise ValueError(f"Total-time mismatch: trip {trip_no}")
    return {
        "trip_no": trip_no,
        "history": history,
        "plan": plan,
        "measured": float(history_sections.code_kwh.sum()),
        "history_physical": float(history_sections.simu_kwh.sum()),
        "planned_physical": float(plan_sections.simu_kwh.sum()),
        "mass": float(history_sections.mass_t.max()),
        "history_run": history_run,
        "history_dwell": history_dwell,
        "planned_run": planned_run,
        "planned_dwell": planned_dwell,
    }


def plot_pair(energy_ax: plt.Axes, speed_ax: plt.Axes, data: dict, legends: bool) -> None:
    history, plan = data["history"], data["plan"]
    energy_ax.plot(history.distance_km, history.measured_cumulative_kwh,
                   color="#9DA5AE", lw=1.15, ls=":", label="历史实测牵引电能（参考）")
    energy_ax.plot(history.distance_km, history.physical_cumulative_kwh,
                   color="#30363D", lw=1.55, label="历史速度的物理机械能")
    energy_ax.plot(plan.distance_km, plan.physical_cumulative_kwh,
                   color="#C54143", lw=1.65, label="规划速度的物理机械能")
    speed_ax.plot(history.distance_km, history.speed_kmh,
                  color="#77838D", lw=1.15, label="历史速度")
    speed_ax.plot(plan.distance_km, plan.speed_kmh,
                  color="#C54143", lw=1.25, label="规划速度（物理模型输入）")
    for ax in (energy_ax, speed_ax):
        ax.set_xlim(0, 36.502)
        ax.grid(color="#DCE4EB", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    energy_ax.set_ylim(0, 490)
    speed_ax.set_ylim(0, 85)
    energy_ax.set_ylabel("累计能耗 (kWh)", fontsize=10)
    speed_ax.set_ylabel("速度 (km/h)", fontsize=10)
    if legends:
        energy_ax.legend(loc="lower right", fontsize=8.3, frameon=True, framealpha=0.97)
        speed_ax.legend(loc="lower right", fontsize=8.3, frameon=True, framealpha=0.97)


def row_heading(data: dict) -> str:
    saving = data["history_physical"] - data["planned_physical"]
    rate = 100 * saving / data["history_physical"]
    return (f"Trip {data['trip_no']:03d}  |  最大区间质量 {data['mass']:.2f} t  |  "
            f"历史物理 {data['history_physical']:.2f} → 规划物理 {data['planned_physical']:.2f} kWh  |  "
            f"节能 {saving:.2f} kWh（{rate:.2f}%）")


def save_combined(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.12,
                        hspace=0.52, wspace=0.18)
    for index, data in enumerate(rows):
        energy_ax, speed_ax = axes[index]
        plot_pair(energy_ax, speed_ax, data, legends=index == 1)
        y = 0.83 if index == 0 else 0.445
        fig.text(0.075, y, row_heading(data), fontsize=11.4, weight="bold", color="#273A4B")
        if index == 1:
            energy_ax.set_xlabel("里程 (km)", fontsize=10)
            speed_ax.set_xlabel("里程 (km)", fontsize=10)
    fig.text(0.075, 0.925, "高载重样本：停站放宽 5% 方案的物理模型复算",
             fontsize=17, weight="bold", color="#1E3549")
    fig.text(0.075, 0.035,
             "节能率以各趟历史速度的物理模型机械能为基线；历史实测牵引电能仅作参考。",
             fontsize=9.2, color="#526475")
    destination = OUTPUT / "trip042_trip067_dwell5_physics_2x2.png"
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    print(destination)


def save_individual(data: dict) -> None:
    fig, (energy_ax, speed_ax) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.18, wspace=0.18)
    plot_pair(energy_ax, speed_ax, data, legends=True)
    energy_ax.set_xlabel("里程 (km)", fontsize=10)
    speed_ax.set_xlabel("里程 (km)", fontsize=10)
    fig.text(0.075, 0.91, row_heading(data), fontsize=12, weight="bold", color="#273A4B")
    fig.text(0.075, 0.82,
             f"历史总时间 {data['history_run'] + data['history_dwell']:.2f} s  |  "
             f"规划运行 {data['planned_run']:.2f} s + 停站 {data['planned_dwell']:.2f} s",
             fontsize=10, color="#526475")
    fig.text(0.075, 0.055, "节能率使用同口径历史物理机械能为基线；实测牵引电能仅作参考。",
             fontsize=9.2, color="#526475")
    destination = OUTPUT / f"trip{data['trip_no']:03d}_dwell5_physics_1x2.png"
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    print(destination)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.unicode_minus": False,
    })
    stats = pd.read_csv(CACHE / "section_comparison_summary.csv")
    rows = [trip_data(stats, trip_no) for trip_no in TRIPS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_combined(rows)
    for row in rows:
        save_individual(row)


if __name__ == "__main__":
    main()
