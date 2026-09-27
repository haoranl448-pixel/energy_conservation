# -*- coding: utf-8 -*-
"""Draw code/residual and physics energy curves in stacked panels per trip."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "output/cache/simu_code_replay/batch_simu_code_trip_compare_complete25"
OUTPUT = ROOT / "output/analysis/high_load_trip42_67_nn_physics_stacked"
SCENARIOS = (
    ("history", "历史", "#303942", "-"),
    ("standard_dp", "标准 DP", "#32649A", "--"),
    ("energy_first", "Energy First", "#C77827", "-."),
    ("dwell5_dp", "停站放宽 5%", "#19857A", "-"),
)


def load_scenario(stats: pd.DataFrame, trip_no: int, scenario: str) -> tuple[pd.DataFrame, dict]:
    sections = stats.loc[stats.trip_no.eq(trip_no) & stats.scenario.eq(scenario)]
    sections = sections.sort_values("section_index")
    if sections.section_index.tolist() != list(range(1, 27)):
        raise ValueError(f"Incomplete trip {trip_no} scenario {scenario}")
    if sections.code_version.eq("unresolved").any():
        raise ValueError(f"Unresolved code energy in trip {trip_no} scenario {scenario}")
    frames = []
    code_offset = physics_offset = 0.0
    for index, section in enumerate(sections.itertuples(index=False)):
        file = CACHE / "replay_data" / f"trip{trip_no:03d}_{scenario}_{index:02d}.pkl"
        frame = pd.read_pickle(file).copy()
        frame["code_cumulative_kwh"] = code_offset + frame.code_section_kwh
        frame["physics_cumulative_kwh"] = physics_offset + frame.simu_section_kwh
        frames.append(frame)
        code_offset += float(section.code_kwh)
        physics_offset += float(section.simu_kwh)
    curve = pd.concat(frames, ignore_index=True)
    if not np.isclose(curve.code_cumulative_kwh.iloc[-1], code_offset, atol=1e-4):
        raise ValueError(f"Code total mismatch in trip {trip_no} scenario {scenario}")
    if not np.isclose(curve.physics_cumulative_kwh.iloc[-1], physics_offset, atol=1e-4):
        raise ValueError(f"Physics total mismatch in trip {trip_no} scenario {scenario}")
    return curve, {
        "code_kwh": code_offset,
        "physics_kwh": physics_offset,
        "mass_t": float(sections.mass_t.max()),
    }


def draw_panel(ax: plt.Axes, curves: dict, totals: dict, field: str, energy_key: str,
               history_label: str, title: str) -> None:
    baseline = totals["history"][energy_key]
    for scenario, name, color, linestyle in SCENARIOS:
        curve = curves[scenario]
        energy = totals[scenario][energy_key]
        label = f"{history_label if scenario == 'history' else name}: {energy:.2f} kWh"
        if scenario != "history":
            saving = baseline - energy
            label += f"  |  节能 {saving:.2f} kWh（{100 * saving / baseline:.2f}%）"
        ax.plot(curve.distance_km, curve[field], color=color, ls=linestyle,
                lw=1.7, label=label)
    ax.set(xlim=(0, 36.502), ylim=(0, 490), ylabel="累计能耗 (kWh)")
    ax.set_title(title, loc="left", fontsize=12, pad=9)
    ax.grid(color="#DCE4EB", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=9.1, frameon=True, framealpha=0.96,
              handlelength=2.1, borderpad=0.55)


def plot_trip(stats: pd.DataFrame, trip_no: int) -> Path:
    curves, totals = {}, {}
    for scenario, _, _, _ in SCENARIOS:
        curves[scenario], totals[scenario] = load_scenario(stats, trip_no, scenario)
    mass = totals["history"]["mass_t"]
    if any(not np.isclose(record["mass_t"], mass, atol=1e-6)
           for record in totals.values()):
        raise ValueError(f"Mass differs across scenarios in trip {trip_no}")

    fig, (code_ax, physics_ax) = plt.subplots(2, 1, sharex=True, figsize=(14.2, 8.3))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.82, bottom=0.13, hspace=0.38)
    draw_panel(code_ax, curves, totals, "code_cumulative_kwh", "code_kwh",
               "历史实测牵引电能", "a  神经网络残差修正的代码规划估计")
    draw_panel(physics_ax, curves, totals, "physics_cumulative_kwh", "physics_kwh",
               "历史物理机械能", "b  物理模型机械能复算")
    physics_ax.set_xlabel("里程 (km)", fontsize=11)
    fig.suptitle(f"Trip {trip_no:03d}  |  最大区间列车质量 {mass:.2f} t  |  "
                 "三种规划方案与历史对比", y=0.955, fontsize=17, weight="bold")
    fig.text(0.085, 0.055,
             "上图历史为实测牵引电能、规划为残差修正估计；下图均为模型机械能。节能率分别按各图历史基线计算。",
             fontsize=9.7, color="#526475")
    destination = OUTPUT / f"trip{trip_no:03d}_nn_physics_stacked.png"
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return destination


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    stats = pd.read_csv(CACHE / "section_comparison_summary.csv")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for trip_no in (42, 67):
        print(plot_trip(stats, trip_no))


if __name__ == "__main__":
    main()
