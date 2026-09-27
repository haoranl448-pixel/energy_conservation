# -*- coding: utf-8 -*-
"""Compare four physics-evaluated curves for high-load trips 042 and 067."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_high_load_trip42_67_physics import CACHE, ROOT, load_curve


OUTPUT = ROOT / "output/analysis/high_load_trip42_67_four_scenarios"
TRIPS = (42, 67)
SCENARIOS = (
    ("history", "历史基线", "#303942", "-"),
    ("standard_dp", "标准 DP", "#32649A", "--"),
    ("energy_first", "Energy First", "#C77827", "-."),
    ("dwell5_dp", "停站放宽 5%", "#19857A", "-"),
)


def load_trip(stats: pd.DataFrame, trip_no: int) -> dict:
    curves = {}
    totals = {}
    sections_by_scenario = {}
    for scenario, _, _, _ in SCENARIOS:
        curves[scenario], sections = load_curve(stats, trip_no, scenario)
        totals[scenario] = float(sections.simu_kwh.sum())
        sections_by_scenario[scenario] = sections
    history = sections_by_scenario["history"]
    maximum_mass = float(history.mass_t.max())
    if any(not np.isclose(group.mass_t.max(), maximum_mass, atol=1e-6)
           for group in sections_by_scenario.values()):
        raise ValueError(f"Mass differs across scenarios: trip {trip_no}")
    distances = [float(curve.distance_km.iloc[-1]) for curve in curves.values()]
    if not np.allclose(distances, distances[0], atol=1e-3):
        raise ValueError(f"Different endpoint distances: trip {trip_no}")
    return {
        "trip_no": trip_no,
        "mass_t": maximum_mass,
        "curves": curves,
        "totals": totals,
        "distance_km": distances[0],
        "history_total_s": float(history.history_runtime_s.sum() + history.history_dwell_s.sum()),
    }


def draw_pair(energy_ax: plt.Axes, speed_ax: plt.Axes, data: dict,
              *, small: bool = False) -> None:
    baseline = data["totals"]["history"]
    for scenario, name, color, line_style in SCENARIOS:
        curve = data["curves"][scenario]
        energy = data["totals"][scenario]
        legend = f"{name}  {energy:.2f} kWh"
        if scenario != "history":
            saving = baseline - energy
            legend += f"  |  节能 {saving:.2f} kWh（{100 * saving / baseline:.2f}%）"
        energy_ax.plot(curve.distance_km, curve.physical_cumulative_kwh,
                       color=color, ls=line_style, lw=1.85 if not small else 1.5,
                       label=legend, zorder=3 if scenario != "history" else 2)
        speed_ax.plot(curve.distance_km, curve.speed_kmh,
                      color=color, ls=line_style, lw=1.5 if not small else 1.15,
                      label=name, alpha=0.92)

    for ax in (energy_ax, speed_ax):
        ax.set_xlim(0, data["distance_km"])
        ax.grid(color="#DBE3EA", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9 if small else 10)
    energy_ax.set_ylim(0, 490)
    speed_ax.set_ylim(0, 85)
    energy_ax.set_ylabel("累计机械能耗 (kWh)", fontsize=10)
    speed_ax.set_ylabel("速度 (km/h)", fontsize=10)
    energy_ax.legend(loc="lower right", fontsize=7.8 if small else 9.1,
                     frameon=True, framealpha=0.96, borderpad=0.5,
                     handlelength=2.2)
    if not small:
        speed_ax.legend(loc="lower right", fontsize=9.1,
                        frameon=True, framealpha=0.96)


def save_single(data: dict) -> Path:
    fig, (energy_ax, speed_ax) = plt.subplots(1, 2, figsize=(15.6, 6.0))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.77, bottom=0.16, wspace=0.18)
    draw_pair(energy_ax, speed_ax, data)
    for ax in (energy_ax, speed_ax):
        ax.set_xlabel("里程 (km)", fontsize=10)
    fig.text(0.07, 0.925,
             f"Trip {data['trip_no']:03d}  |  最大区间列车质量 {data['mass_t']:.2f} t  |  "
             "四种速度曲线的物理模型复算",
             fontsize=17, weight="bold", color="#203647")
    fig.text(0.07, 0.84,
             f"历史总时长 {data['history_total_s']:.2f} s  |  "
             "四条能耗曲线均为同一物理模型计算的机械能",
             fontsize=11, color="#4A5E6B")
    fig.text(0.07, 0.045,
             "节能量和节能率统一以本趟历史速度的物理机械能为基线。",
             fontsize=9.6, color="#526475")
    path = OUTPUT / f"trip{data['trip_no']:03d}_four_physics_1x2.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def save_combined(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 8.8))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.10,
                        wspace=0.18, hspace=0.47)
    for index, data in enumerate(rows):
        energy_ax, speed_ax = axes[index]
        draw_pair(energy_ax, speed_ax, data, small=True)
        if index == 1:
            energy_ax.set_xlabel("里程 (km)", fontsize=10)
            speed_ax.set_xlabel("里程 (km)", fontsize=10)
        fig.text(0.07, 0.835 if index == 0 else 0.44,
                 f"Trip {data['trip_no']:03d}  |  最大区间质量 {data['mass_t']:.2f} t  |  "
                 f"历史物理基线 {data['totals']['history']:.2f} kWh",
                 fontsize=11.8, weight="bold", color="#273A4B")
    fig.text(0.07, 0.93, "高载重样本：三种规划方案与历史的物理机械能对比",
             fontsize=17, weight="bold", color="#203647")
    fig.text(0.07, 0.035,
             "四种方案均使用同一物理模型；节能率按各趟历史物理机械能计算。",
             fontsize=9.4, color="#526475")
    path = OUTPUT / "trip042_trip067_four_physics_2x2.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.unicode_minus": False,
    })
    stats = pd.read_csv(CACHE / "section_comparison_summary.csv")
    rows = [load_trip(stats, trip_no) for trip_no in TRIPS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for data in rows:
        print(save_single(data))
    print(save_combined(rows))


if __name__ == "__main__":
    main()
