# -*- coding: utf-8 -*-
"""Render six trip 012 comparison figures from existing pointwise replay caches.

The first three compare code-planned energy with measured history. The next
three compare planned and historical physics results, with measured electrical
energy shown only as a separate reference. No models are rerun.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_trip12_energy_speed_1x2 import CACHE, SCENARIOS, load_curve, pct, plan_clock


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/simu_code_trip_compare_complete25/trip012/six_stage_figures"
STAGES = (
    ("standard_dp", "标准 DP", "01_standard_dp_code_vs_history.png"),
    ("energy_first", "Energy First", "02_energy_first_code_vs_history.png"),
    ("dwell5_dp", "停站放宽 5% DP", "03_dwell5_dp_code_vs_history.png"),
    ("standard_dp", "标准 DP", "04_standard_dp_physics_vs_history.png"),
    ("energy_first", "Energy First", "05_energy_first_physics_vs_history.png"),
    ("dwell5_dp", "停站放宽 5% DP", "06_dwell5_dp_physics_vs_history.png"),
)


def render(
    scenario: str,
    label: str,
    filename: str,
    stats: pd.DataFrame,
    history: pd.DataFrame,
    measured_kwh: float,
    historical_physics_kwh: float,
    historical_time: float,
    maximum_mass: float,
) -> None:
    planned, sections = load_curve(stats, scenario)
    runtime, dwell, total = plan_clock(SCENARIOS[scenario][1], sections)
    physics_stage = "physics" in filename
    planned_kwh = float(sections.simu_kwh.sum() if physics_stage else sections.code_kwh.sum())
    baseline_kwh = historical_physics_kwh if physics_stage else measured_kwh
    saving_kwh = baseline_kwh - planned_kwh

    fig, (energy_ax, speed_ax) = plt.subplots(1, 2, figsize=(16, 6.4))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.16, wspace=0.19)

    if physics_stage:
        energy_ax.plot(history.distance_km, history.code_cumulative_kwh,
                       color="#7F8993", lw=1.15, ls=":", label="历史实测牵引电能（参考）")
        energy_ax.plot(history.distance_km, history.physical_cumulative_kwh,
                       color="#202830", lw=1.8, label="历史速度的 Simulink 机械能")
        energy_ax.plot(planned.distance_km, planned.physical_cumulative_kwh,
                       color="#C54242", lw=1.7, label="规划速度的 Simulink 机械能")
        note = (f"历史实测牵引电能（参考）：{measured_kwh:.2f} kWh\n"
                f"历史 Simulink 机械能：{historical_physics_kwh:.2f} kWh\n"
                f"规划 Simulink 机械能：{planned_kwh:.2f} kWh\n\n"
                f"同口径物理模型节能：{saving_kwh:+.2f} kWh "
                f"({pct(saving_kwh, baseline_kwh):+.2f}%)")
        footer = "节能量以历史速度的物理模型计算结果为基线；实测牵引电能仅供参考，不能与机械能直接相减。"
        stage_name = "物理模型复算与历史对照"
        plan_name = "规划速度（Simulink 输入）"
    else:
        energy_ax.plot(history.distance_km, history.code_cumulative_kwh,
                       color="#202830", lw=1.8, label="历史实测牵引电能")
        energy_ax.plot(planned.distance_km, planned.code_cumulative_kwh,
                       color="#188477", lw=1.7, label="代码规划估计能耗")
        note = (f"历史实测牵引电能：{measured_kwh:.2f} kWh\n"
                f"代码规划估计能耗：{planned_kwh:.2f} kWh\n\n"
                f"相对历史实测的估计节能：{saving_kwh:+.2f} kWh "
                f"({pct(saving_kwh, baseline_kwh):+.2f}%)")
        footer = "本图展示代码规划估计；规划速度的独立物理模型复算见对应方案的后续图。"
        stage_name = "代码规划与历史对照"
        plan_name = "规划速度"

    speed_ax.plot(history.distance_km, history.speed_kmh,
                  color="#788694", lw=1.45, label="历史实测速度")
    speed_ax.plot(planned.distance_km, planned.speed_kmh,
                  color="#C54242" if physics_stage else "#188477", lw=1.4,
                  label=plan_name)

    for ax, ylabel in ((energy_ax, "累计能耗 (kWh)"), (speed_ax, "速度 (km/h)")):
        ax.set(xlabel="里程 (km)", ylabel=ylabel, xlim=(0, 36.502))
        ax.grid(True, color="#D9E2EA", alpha=0.65, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    energy_ax.set_title("a  能耗—里程", loc="left", fontsize=12)
    speed_ax.set_title("b  速度—里程", loc="left", fontsize=12)
    energy_ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.96)
    speed_ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.96)
    energy_ax.text(0.02, 0.97, note, transform=energy_ax.transAxes, ha="left", va="top",
                   fontsize=9.2, bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                                    "edgecolor": "#ABB8C4", "alpha": 0.95})
    fig.suptitle(f"Trip 012 | {label} | {stage_name} | 最大区间列车质量 {maximum_mass:.2f} t",
                 y=0.975, fontsize=14)
    fig.text(0.5, 0.885,
             f"规划总时间 {total:.2f} s = 运行 {runtime:.2f} s + 停站 {dwell:.2f} s"
             f"    |    历史总时间 {historical_time:.2f} s",
             ha="center", fontsize=11)
    fig.text(0.065, 0.045, footer, fontsize=9.5, color="#465561")
    destination = OUTPUT / filename
    fig.savefig(destination, dpi=250, facecolor="white")
    plt.close(fig)
    print(destination)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    stats = pd.read_csv(CACHE / "section_comparison_summary.csv")
    stats = stats.loc[stats.trip_no.eq(12)]
    history, history_sections = load_curve(stats, "history")
    measured_kwh = float(history_sections.code_kwh.sum())
    historical_physics_kwh = float(history_sections.simu_kwh.sum())
    historical_time = float(history_sections.history_runtime_s.sum()
                            + history_sections.history_dwell_s.sum())
    maximum_mass = float(history_sections.mass_t.max())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for scenario, label, filename in STAGES:
        render(scenario, label, filename, stats, history, measured_kwh,
               historical_physics_kwh, historical_time, maximum_mass)


if __name__ == "__main__":
    main()
