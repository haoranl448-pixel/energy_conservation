# -*- coding: utf-8 -*-
"""Compare Class 3 extrapolation with the Class 2 curves in the current planning bank."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plot_class3_to_class2_residual_energy import (
    ARTIFACTS,
    DT,
    GENERATED,
    MODELS,
    cumulative_distance,
    evaluate_residual,
    load_model,
    sample_track,
    select_real_class2,
)


ROOT = Path(__file__).resolve().parents[2]
PLANNING = ROOT / "output/ato_generated_results_new_v4"
OUTPUT = ROOT / "output/analysis/class3_extrapolation_vs_planning_class2_residual_20260924"


def evaluate_curve(
    section: str,
    curve: pd.DataFrame,
    mass_t: float,
    track_distance: np.ndarray,
    gradient: np.ndarray,
    curvature: np.ndarray,
    assets: tuple,
) -> np.ndarray:
    distance = curve["dist_m"].to_numpy(float)
    velocity = curve["velocity_mps"].to_numpy(float)
    acceleration = np.r_[0.0, np.diff(velocity) / DT]
    return evaluate_residual(
        section,
        velocity,
        distance,
        acceleration,
        sample_track(gradient, track_distance, distance),
        sample_track(curvature, track_distance, distance),
        mass_t,
        assets,
    )


def plot_comparison(
    section: str,
    planning: pd.DataFrame,
    extrapolated: pd.DataFrame,
    planning_energy: np.ndarray,
    extrapolated_energy: np.ndarray,
    mass_t: float,
) -> None:
    fig, (energy_ax, speed_ax) = plt.subplots(1, 2, figsize=(14.4, 5.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.21, wspace=0.18)
    planning_distance = planning["dist_m"].to_numpy(float) / 1000.0
    extrapolated_distance = extrapolated["dist_m"].to_numpy(float) / 1000.0
    energy_ax.plot(
        planning_distance,
        planning_energy,
        color="#26313A",
        lw=2.0,
        label=f"规划候选 Class 2  {planning_energy[-1]:.2f} kWh",
    )
    energy_ax.plot(
        extrapolated_distance,
        extrapolated_energy,
        color="#C46C25",
        lw=2.0,
        label=f"Class 3 外推 Class 2  {extrapolated_energy[-1]:.2f} kWh",
    )
    speed_ax.plot(
        planning_distance,
        planning["velocity_mps"].to_numpy(float) * 3.6,
        color="#26313A",
        lw=1.7,
        label="规划候选 Class 2",
    )
    speed_ax.plot(
        extrapolated_distance,
        extrapolated["velocity_mps"].to_numpy(float) * 3.6,
        color="#C46C25",
        lw=1.8,
        label="Class 3 外推 Class 2",
    )
    for ax in (energy_ax, speed_ax):
        ax.set_xlim(0, max(planning_distance[-1], extrapolated_distance[-1]))
        ax.set_xlabel("里程 (km)")
        ax.grid(color="#DDE5EA", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    energy_ax.set_ylim(bottom=0)
    speed_ax.set_ylim(0, 85)
    energy_ax.set_ylabel("累计预测能耗 (kWh)")
    speed_ax.set_ylabel("速度 (km/h)")
    energy_ax.set_title("a  E–s：能耗—里程", loc="left", fontsize=11)
    speed_ax.set_title("b  v–s：速度—里程", loc="left", fontsize=11)
    energy_ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.96)
    speed_ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.96)

    planning_time = float(planning["time_s"].iloc[-1])
    extrapolated_time = float(extrapolated["time_s"].iloc[-1])
    delta = extrapolated_energy[-1] - planning_energy[-1]
    delta_pct = 100.0 * delta / planning_energy[-1]
    fig.text(
        0.5, 0.93,
        f"{section}  |  Class 3 外推与规划候选 Class 2 对比",
        ha="center", fontsize=15, weight="bold", color="#213547",
    )
    fig.text(
        0.5, 0.85,
        f"统一质量 {mass_t:.2f} t  |  规划候选 {planning_time:.2f} s / 外推 {extrapolated_time:.2f} s",
        ha="center", fontsize=10, color="#4A5E6B",
    )
    fig.text(
        0.075, 0.105,
        f"外推－规划候选：预测能耗 {delta:+.2f} kWh ({delta_pct:+.1f}%)；"
        f"运行时间 {extrapolated_time - planning_time:+.2f} s。",
        fontsize=9.2, color="#435869",
    )
    fig.text(
        0.075, 0.052,
        "两条曲线均用同一物理基值＋残差模型、相同质量和线路特征计算；"
        "规划候选取自当前 v4 曲线库。本图比较候选曲线，不是实测节能率。",
        fontsize=8.8, color="#5B6974",
    )
    destination = OUTPUT / section
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "class3_extrapolation_vs_planning_class2_es_vs.png", dpi=300, facecolor="white")
    fig.savefig(destination / "class3_extrapolation_vs_planning_class2_es_vs.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
    })
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    coverage = []
    source_plots = sorted(GENERATED.glob("*/class2_generated_vs_real_speed_distance.png"))
    if len(source_plots) != 13:
        raise ValueError(f"Expected 13 genuine Class 3 to Class 2 sections, got {len(source_plots)}")
    for source_plot in source_plots:
        section = source_plot.parent.name
        with (ARTIFACTS / section / "class3_phase_artifacts.pkl").open("rb") as file:
            artifact = pickle.load(file)
        if artifact.get("reference_class") != "class3":
            raise ValueError(f"{section}: extrapolated curve does not use a Class 3 parent")
        planning_path = PLANNING / section / "class2_generated_curve.csv"
        if not planning_path.exists():
            coverage.append({"section": section, "status": "missing_current_v4_planning_class2"})
            continue
        metadata = pd.read_csv(planning_path.parent / "all_classes_summary.csv")
        class2 = metadata.loc[metadata["class_name"].eq("class2")].iloc[0]
        if class2["curve_source"] != "real" or class2["parent_ref"] != "class2":
            raise ValueError(f"{section}: planning Class 2 is not based on real Class 2")

        planning = pd.read_csv(planning_path)
        extrapolated = pd.read_csv(source_plot.parent / "class2_generated_curve.csv")
        track, n_runs, track_run_id = select_real_class2(section, float(artifact["target_l"]))
        mass_t = float(track["重量"].median())
        track_distance = cumulative_distance(track)
        gradient = track["gradient"].to_numpy(float)
        curvature = track["curvature"].to_numpy(float)
        assets = load_model(section, MODELS, torch.device("cpu"))
        planning_energy = evaluate_curve(
            section, planning, mass_t, track_distance, gradient, curvature, assets,
        )
        extrapolated_energy = evaluate_curve(
            section, extrapolated, mass_t, track_distance, gradient, curvature, assets,
        )
        plot_comparison(section, planning, extrapolated, planning_energy, extrapolated_energy, mass_t)
        planning_kwh = float(planning_energy[-1])
        extrapolated_kwh = float(extrapolated_energy[-1])
        results.append({
            "section": section,
            "mass_t": mass_t,
            "track_reference_run_id": track_run_id,
            "track_available_real_class2_runs": n_runs,
            "planning_curve_source": str(planning_path),
            "extrapolated_curve_source": str(source_plot.parent / "class2_generated_curve.csv"),
            "planning_runtime_s": float(planning["time_s"].iloc[-1]),
            "extrapolated_runtime_s": float(extrapolated["time_s"].iloc[-1]),
            "planning_residual_kwh": planning_kwh,
            "extrapolated_residual_kwh": extrapolated_kwh,
            "extrapolated_minus_planning_kwh": extrapolated_kwh - planning_kwh,
            "extrapolated_minus_planning_pct": 100.0 * (extrapolated_kwh - planning_kwh) / planning_kwh,
        })
        coverage.append({"section": section, "status": "compared"})
        print(f"{section}: {planning_kwh:.3f} vs {extrapolated_kwh:.3f} kWh")
    pd.DataFrame(results).to_csv(OUTPUT / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(OUTPUT / "coverage.csv", index=False, encoding="utf-8-sig")
    print(f"Compared {len(results)} of {len(source_plots)} sections in {OUTPUT}")


if __name__ == "__main__":
    main()
