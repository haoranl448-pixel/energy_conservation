# -*- coding: utf-8 -*-
"""Draw separate E-s and v-s figures for code and physics, trips 042/067."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_high_load_nn_physics_stacked import CACHE, ROOT, SCENARIOS, load_scenario


OUTPUT = ROOT / "output/analysis/high_load_trip42_67_four_scenarios_by_model"
MODES = (
    ("code", "code_cumulative_kwh", "code_kwh", "代码规划估计（神经网络残差修正）",
     "历史实测牵引电能", "历史实测牵引电能与规划估计口径不同于下图的物理机械能。"),
    ("physics", "physics_cumulative_kwh", "physics_kwh", "物理模型机械能复算",
     "历史物理机械能", "四条机械能曲线由同一物理模型计算，节能率以历史物理机械能为基线。"),
)


def draw(trip_no: int, curves: dict, totals: dict, mode: tuple,
         history_runtime_s: float, history_dwell_s: float) -> Path:
    suffix, column, total_key, model_name, history_name, footer = mode
    baseline = totals["history"][total_key]
    fig, (energy_ax, speed_ax) = plt.subplots(1, 2, figsize=(15.6, 6.0))
    fig.subplots_adjust(left=0.067, right=0.985, top=0.77, bottom=0.16, wspace=0.17)

    for scenario, name, color, style in SCENARIOS:
        curve = curves[scenario]
        energy = totals[scenario][total_key]
        energy_label = f"{history_name if scenario == 'history' else name}: {energy:.2f} kWh"
        if scenario != "history":
            saving = baseline - energy
            energy_label += f"  |  节能 {saving:.2f} kWh（{100 * saving / baseline:.2f}%）"
        energy_ax.plot(curve.distance_km, curve[column], color=color, ls=style,
                       lw=1.8, label=energy_label)
        speed_ax.plot(curve.distance_km, curve.speed_kmh, color=color, ls=style,
                      lw=1.4, label="历史速度" if scenario == "history" else f"{name} 速度")

    distance = float(curves["history"].distance_km.iloc[-1])
    for ax, ylabel in ((energy_ax, "累计能耗 (kWh)"), (speed_ax, "速度 (km/h)")):
        ax.set(xlim=(0, distance), xlabel="里程 (km)", ylabel=ylabel)
        ax.grid(color="#DCE4EB", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9.5)
    energy_ax.set_ylim(0, 490)
    speed_ax.set_ylim(0, 85)
    energy_ax.set_title("a  能耗—里程", loc="left", fontsize=10.5)
    speed_ax.set_title("b  速度—里程", loc="left", fontsize=10.5)
    energy_ax.legend(loc="upper left", fontsize=8.35, frameon=True, framealpha=0.97,
                     borderpad=0.5, handlelength=2.1)
    speed_ax.legend(loc="lower right", fontsize=8.9, frameon=True, framealpha=0.97)

    fig.text(0.5, 0.925,
             f"Trip {trip_no:03d} | 四方案对比 | 最大区间列车质量 {totals['history']['mass_t']:.2f} t",
             ha="center", fontsize=12.5, weight="bold", color="#203647")
    fig.text(0.5, 0.855,
             f"{model_name} | 历史时间：运行 {history_runtime_s:.2f} s + "
             f"停站 {history_dwell_s:.2f} s = 总计 {history_runtime_s + history_dwell_s:.2f} s",
             ha="center", fontsize=9.6, color="#4A5E6B")
    fig.text(0.067, 0.045, footer, fontsize=9.4, color="#526475")
    path = OUTPUT / f"trip{trip_no:03d}_{suffix}_four_curves_es_vs.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    stats = pd.read_csv(CACHE / "section_comparison_summary.csv")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for trip_no in (42, 67):
        curves, totals = {}, {}
        for scenario, _, _, _ in SCENARIOS:
            curves[scenario], totals[scenario] = load_scenario(stats, trip_no, scenario)
        mass = totals["history"]["mass_t"]
        if any(not np.isclose(item["mass_t"], mass, atol=1e-6)
               for item in totals.values()):
            raise ValueError(f"Mass differs across scenarios: trip {trip_no}")
        history_sections = stats.loc[stats.trip_no.eq(trip_no) & stats.scenario.eq("history")]
        history_runtime_s = float(history_sections.history_runtime_s.sum())
        history_dwell_s = float(history_sections.history_dwell_s.sum())
        for mode in MODES:
            print(draw(trip_no, curves, totals, mode,
                       history_runtime_s, history_dwell_s))


if __name__ == "__main__":
    main()
