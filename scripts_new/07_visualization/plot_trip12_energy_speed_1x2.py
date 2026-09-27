# -*- coding: utf-8 -*-
"""Redraw trip 012 history and three frozen plans from verified replay caches.

This only renders figures. It does not rerun DP, residual inference, or physics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "output/cache/simu_code_replay/batch_simu_code_trip_compare_complete25"
POINTS = CACHE / "replay_data"
OUTPUT = ROOT / "output/simu_code_trip_compare_complete25/trip012/1x2_redraw"
PLAN_DIR = ROOT / "output/schedule/batch_trip_reports_complete13_dp_unified/trip012"
SCENARIOS = {
    "standard_dp": ("标准 DP", "01_standard_priority.csv", "01_standard_dp_1x2.png"),
    "energy_first": ("Energy First", "02_energy_first.csv", "02_energy_first_1x2.png"),
    "dwell5_dp": ("停站放宽 5% DP", "03_real_priority_dwell_5pct.csv", "03_dwell5_dp_1x2.png"),
}


def pct(amount: float, baseline: float) -> float:
    return 100.0 * amount / baseline


def load_curve(stats: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    section_stats = stats.loc[stats.scenario.eq(scenario)].sort_values("section_index")
    if section_stats.section_index.tolist() != list(range(1, 27)):
        raise ValueError(f"Expected 26 ordered sections for {scenario}")
    if section_stats.code_version.eq("unresolved").any():
        raise ValueError(f"Cannot draw pointwise code curve: unresolved sections in {scenario}")
    frames = []
    code_offset = physical_offset = 0.0
    for index, row in enumerate(section_stats.itertuples(index=False)):
        frame = pd.read_pickle(POINTS / f"trip012_{scenario}_{index:02d}.pkl").copy()
        frame["code_cumulative_kwh"] = code_offset + frame.code_section_kwh
        frame["physical_cumulative_kwh"] = physical_offset + frame.simu_section_kwh
        frames.append(frame)
        code_offset += float(row.code_kwh)
        physical_offset += float(row.simu_kwh)
    points = pd.concat(frames, ignore_index=True)
    if not np.isclose(points.code_cumulative_kwh.iloc[-1], code_offset, atol=1e-4):
        raise ValueError(f"Code cumulative energy does not match section summary: {scenario}")
    if not np.isclose(points.physical_cumulative_kwh.iloc[-1], physical_offset, atol=1e-4):
        raise ValueError(f"Physics cumulative energy does not match section summary: {scenario}")
    return points, section_stats


def plan_clock(filename: str, stats: pd.DataFrame) -> tuple[float, float, float]:
    plan = pd.read_csv(PLAN_DIR / filename, encoding="utf-8-sig")
    if len(plan) != 27:
        raise ValueError(f"Expected 26 sections plus total: {filename}")
    running = float(plan.iloc[:26, 2].sum())
    dwell = float(plan.iloc[:26, 3].sum())
    if not np.isclose(running, stats.runtime_s.sum(), atol=0.11):
        raise ValueError(f"Planned running time and cached replay disagree: {filename}")
    return running, dwell, running + dwell


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    stats = pd.read_csv(CACHE / "section_comparison_summary.csv")
    stats = stats.loc[stats.trip_no.eq(12)]
    history, history_stats = load_curve(stats, "history")
    history_measured = float(history_stats.code_kwh.sum())
    history_physical = float(history_stats.simu_kwh.sum())
    history_runtime = float(history_stats.history_runtime_s.sum())
    history_dwell = float(history_stats.history_dwell_s.sum())
    history_total = history_runtime + history_dwell
    maximum_mass = float(history_stats.mass_t.max())
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for scenario in ("history", *SCENARIOS):
        planned = scenario != "history"
        if planned:
            label, plan_file, filename = SCENARIOS[scenario]
            curve, selected = load_curve(stats, scenario)
            runtime, dwell, total = plan_clock(plan_file, selected)
        else:
            label, filename = "历史与物理模型", "00_history_1x2.png"
            curve, selected = history, history_stats
            runtime, dwell, total = history_runtime, history_dwell, history_total
        code_energy = float(selected.code_kwh.sum())
        physical_energy = float(selected.simu_kwh.sum())

        fig, (energy_ax, speed_ax) = plt.subplots(1, 2, figsize=(16, 6.4))
        fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.16, wspace=0.19)
        energy_ax.plot(history.distance_km, history.code_cumulative_kwh,
                       color="#202830", lw=1.7, label="历史实测牵引电能")
        if planned:
            energy_ax.plot(curve.distance_km, curve.code_cumulative_kwh,
                           color="#188477", lw=1.6, label="神经网络残差修正规划能耗")
        energy_ax.plot(curve.distance_km, curve.physical_cumulative_kwh,
                       color="#C54242", lw=1.6, ls="--",
                       label="规划速度的 Simulink 机械能" if planned else "历史速度的 Simulink 机械能")

        speed_ax.plot(history.distance_km, history.speed_kmh,
                      color="#8B98A5" if planned else "#303B46", lw=1.35,
                      label="历史实测速度")
        if planned:
            speed_ax.plot(curve.distance_km, curve.speed_kmh,
                          color="#C54242", lw=1.35, label="规划速度（Simulink 输入）")
        else:
            speed_ax.plot(curve.distance_km, curve.speed_kmh,
                          color="#C54242", lw=1.15, ls="--", label="Simulink 输入速度（与历史相同）")

        for ax, ylabel in ((energy_ax, "累计能耗 (kWh)"), (speed_ax, "速度 (km/h)")):
            ax.set(xlabel="里程 (km)", ylabel=ylabel, xlim=(0, 36.502))
            ax.grid(True, color="#D9E2EA", alpha=0.65, lw=0.6)
            ax.set_axisbelow(True)
        energy_ax.set_ylim(bottom=0)
        speed_ax.set_ylim(bottom=0)
        energy_ax.set_title("a  能耗—里程", loc="left", fontsize=12)
        speed_ax.set_title("b  速度—里程", loc="left", fontsize=12)
        energy_ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.96)
        speed_ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.96)

        if planned:
            code_saving = history_measured - code_energy
            physical_saving = history_physical - physical_energy
            note = (f"历史实测牵引电能：{history_measured:.2f} kWh\n"
                    f"代码规划估计：{code_energy:.2f} kWh\n"
                    f"规划 Simulink 机械能：{physical_energy:.2f} kWh\n\n"
                    f"代码相对实测基线：{code_saving:+.2f} kWh ({pct(code_saving, history_measured):+.2f}%)\n"
                    f"Simulink 相对历史机械能：{physical_saving:+.2f} kWh ({pct(physical_saving, history_physical):+.2f}%)")
            footer = "实测牵引电能与模型机械能口径不同；两项节能量分别使用各自的历史基线。"
        else:
            note = (f"历史实测牵引电能：{history_measured:.2f} kWh\n"
                    f"历史速度的 Simulink 机械能：{history_physical:.2f} kWh")
            footer = "右图两条速度线是同一条历史速度输入，重合不代表独立速度预测；实测电能与机械能不可直接作误差解释。"
        energy_ax.text(0.02, 0.97, note, transform=energy_ax.transAxes, ha="left", va="top",
                       fontsize=9.2, bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                                        "edgecolor": "#ABB8C4", "alpha": 0.95})
        fig.suptitle(f"Trip 012 | {label} | 最大区间列车质量 {maximum_mass:.2f} t", y=0.975, fontsize=15)
        fig.text(0.5, 0.885,
                 f"{'规划' if planned else '历史'}总时间 {total:.2f} s = 运行 {runtime:.2f} s + 停站 {dwell:.2f} s"
                 + (f"    |    历史总时间 {history_total:.2f} s" if planned else ""),
                 ha="center", fontsize=11)
        fig.text(0.065, 0.045, footer, fontsize=9.5, color="#465561")
        destination = OUTPUT / filename
        fig.savefig(destination, dpi=250, facecolor="white")
        plt.close(fig)
        print(destination)


if __name__ == "__main__":
    main()
