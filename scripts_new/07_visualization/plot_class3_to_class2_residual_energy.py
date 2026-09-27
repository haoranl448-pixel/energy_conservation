# -*- coding: utf-8 -*-
"""Compare Class 3-derived Class 2 curves with real Class 2 using residual energy."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_new/02_ato_template_generation"))
sys.path.insert(0, str(ROOT / "scripts_new/08_analysis"))

import simulate_ATO_v7 as ato_validation  # noqa: E402
from plot_triple_axis_all_runs import load_model  # noqa: E402
from src.physics.train_simu import TrainTheoreticalEnergyModel  # noqa: E402


GENERATED = ROOT / "output/ato_generated_results_new_v2"
ARTIFACTS = ROOT / "output/ato_phase_results_v2"
REAL_DATA = ROOT / "data/data_processed_new_v2"
MODELS = ROOT / "output/models/nn_results_residual_v3_traceability_20260921"
OUTPUT = ROOT / "output/analysis/class3_to_class2_residual_energy_20260924"
DT = 0.05
SEQ_LEN = 30


def clean_real_group(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["__order"] = ato_validation.build_order_key(frame["时刻"])
    frame["__index"] = np.arange(len(frame))
    fields = ["速度(m/s)", "累计位移(m)", "重量", "energy", "gradient", "curvature"]
    for field in fields:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=fields).sort_values(["__order", "__index"])
    frame = frame.reset_index(drop=True)
    if len(frame) < ato_validation.MIN_RUN_POINTS:
        raise ValueError("Representative Class 2 run became too short after cleaning")
    return frame


def select_real_class2(section: str, target_distance: float) -> tuple[pd.DataFrame, int, str]:
    source = REAL_DATA / f"cleaned_{section}.xlsx"
    frame = pd.read_excel(source)
    frame["运行等级"] = frame["运行等级"].apply(ato_validation.normalize_class_label)
    frame = frame.loc[frame["运行等级"].eq("class2")].copy()
    if frame.empty:
        raise ValueError("No real Class 2 observations")
    frame["run_id"], _ = ato_validation.build_run_id(frame)

    runs = []
    grid = np.linspace(0.0, target_distance, 500)
    for run_id, group in frame.groupby("run_id"):
        try:
            candidate = clean_real_group(group)
            distance = np.maximum.accumulate(candidate["累计位移(m)"].to_numpy(float))
            distance -= distance[0]
            if abs(distance[-1] - target_distance) > 30.0:
                continue
            velocity = candidate["速度(m/s)"].clip(lower=0).to_numpy(float)
            unique_distance, unique_index = np.unique(distance, return_index=True)
            if unique_distance[-1] < grid[-1] - 2.0:
                continue
            profile = np.interp(grid, unique_distance, velocity[unique_index])
            runs.append((str(run_id), candidate, profile))
        except (ValueError, KeyError):
            continue
    if not runs:
        raise ValueError("No usable real Class 2 run with the expected section distance")

    profiles = np.stack([item[2] for item in runs])
    centre = np.median(profiles, axis=0)
    medoid_index = int(np.argmin(np.mean((profiles - centre) ** 2, axis=1)))
    run_id, chosen, _ = runs[medoid_index]
    return chosen, len(runs), run_id


def cumulative_distance(frame: pd.DataFrame) -> np.ndarray:
    distance = frame["累计位移(m)"].to_numpy(float)
    return np.maximum.accumulate(distance - distance[0])


def sample_track(values: np.ndarray, distance: np.ndarray, query: np.ndarray) -> np.ndarray:
    unique_distance, unique_index = np.unique(distance, return_index=True)
    return interp1d(
        unique_distance,
        values[unique_index],
        kind="nearest",
        bounds_error=False,
        fill_value=(values[unique_index[0]], values[unique_index[-1]]),
    )(query)


def evaluate_residual(
    section: str,
    velocity: np.ndarray,
    distance: np.ndarray,
    acceleration: np.ndarray,
    gradient: np.ndarray,
    curvature: np.ndarray,
    mass_t: float,
    assets: tuple,
) -> np.ndarray:
    model, scaler_x, scaler_y = assets
    n = len(velocity)
    if not all(len(x) == n for x in (distance, acceleration, gradient, curvature)):
        raise ValueError(f"Inconsistent feature lengths for {section}")
    time = np.arange(n, dtype=float) * DT
    engine = TrainTheoreticalEnergyModel()
    physics = engine.run_batch_simulation_fast(time, velocity, mass_t, section_name=section)
    features = np.stack(
        [velocity, acceleration, physics, gradient, np.full(n, mass_t), curvature], axis=1
    )
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite residual features for {section}")
    scaled = scaler_x.transform(features)
    residual = np.zeros(n, dtype=float)
    if n >= SEQ_LEN:
        windows = np.stack([scaled[i - SEQ_LEN + 1:i + 1] for i in range(SEQ_LEN - 1, n)])
        with torch.no_grad():
            outputs = model(torch.as_tensor(windows, dtype=torch.float32)).cpu().numpy()
        residual[SEQ_LEN - 1:] = scaler_y.inverse_transform(outputs).reshape(-1)
    fusion = np.maximum(np.asarray(physics) + residual, 0.0)
    fusion[:SEQ_LEN - 1] = 0.0
    return np.cumsum(fusion) / 1000.0


def draw(
    section: str,
    real: pd.DataFrame,
    generated: pd.DataFrame,
    real_energy: np.ndarray,
    real_predicted: np.ndarray,
    generated_predicted: np.ndarray,
    n_runs: int,
    mass_t: float,
    run_id: str,
) -> None:
    real_distance = cumulative_distance(real)
    generated_distance = generated["dist_m"].to_numpy(float)
    real_velocity = real["速度(m/s)"].clip(lower=0).to_numpy(float)
    generated_velocity = generated["velocity_mps"].to_numpy(float)

    fig, (energy_ax, speed_ax) = plt.subplots(1, 2, figsize=(14.4, 5.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.21, wspace=0.18)
    real_label = f"真实 Class 2：实测牵引电能  {real_energy[-1]:.2f} kWh"
    real_prediction_label = f"真实 Class 2：残差模型预测  {real_predicted[-1]:.2f} kWh"
    generated_label = f"Class 3 推 Class 2：残差模型预测  {generated_predicted[-1]:.2f} kWh"
    energy_ax.plot(real_distance / 1000, real_energy, color="#26313A", lw=2.0, label=real_label)
    energy_ax.plot(real_distance / 1000, real_predicted, color="#61809C", lw=1.55,
                   ls="--", label=real_prediction_label)
    energy_ax.plot(generated_distance / 1000, generated_predicted, color="#C46C25", lw=2.0,
                   label=generated_label)
    speed_ax.plot(real_distance / 1000, real_velocity * 3.6, color="#26313A", lw=1.6,
                  label="真实 Class 2")
    speed_ax.plot(generated_distance / 1000, generated_velocity * 3.6, color="#C46C25", lw=1.8,
                  label="Class 3 推 Class 2")

    for ax in (energy_ax, speed_ax):
        ax.set_xlim(0, max(real_distance[-1], generated_distance[-1]) / 1000)
        ax.set_xlabel("里程 (km)")
        ax.grid(color="#DDE5EA", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    energy_ax.set_ylim(bottom=0)
    speed_ax.set_ylim(0, 85)
    energy_ax.set_ylabel("累计能耗 (kWh)")
    speed_ax.set_ylabel("速度 (km/h)")
    energy_ax.set_title("a  E–s：能耗—里程", loc="left", fontsize=11)
    speed_ax.set_title("b  v–s：速度—里程", loc="left", fontsize=11)
    energy_ax.legend(loc="upper left", fontsize=8.7, frameon=True, framealpha=0.96)
    speed_ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.96)

    actual_time = (len(real) - 1) * DT
    generated_time = float(generated["time_s"].iloc[-1])
    model_gap = real_predicted[-1] - real_energy[-1]
    curve_gap = generated_predicted[-1] - real_predicted[-1]
    total_gap = generated_predicted[-1] - real_energy[-1]
    total_gap_pct = 100.0 * total_gap / real_energy[-1]
    fig.text(0.5, 0.93, f"{section}  |  Class 3 → Class 2 跨等级生成能耗对比",
             ha="center", fontsize=15, weight="bold", color="#213547")
    fig.text(0.5, 0.85,
             f"相同质量 {mass_t:.2f} t  |  生成 {generated_time:.1f} s / 真实 {actual_time:.1f} s"
             f"  |  真实 Class 2 可用样本 n={n_runs}",
             ha="center", fontsize=10, color="#4A5E6B")
    fig.text(0.075, 0.105,
             f"生成预测－真实实测 {total_gap:+.2f} kWh ({total_gap_pct:+.1f}%)；"
             f"其中模型在真实曲线上的偏差 {model_gap:+.2f} kWh，曲线替换的模型内差异 {curve_gap:+.2f} kWh。",
             fontsize=9.1, color="#435869")
    fig.text(0.075, 0.052,
             "真实曲线按速度形态选取代表样本；生成曲线仅使用 Class 3 模板。实测电能与预测能耗口径不同；本图非残差模型独立测试。",
             fontsize=8.8, color="#5B6974")
    destination = OUTPUT / section
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "class3_to_class2_residual_es_vs.png", dpi=300, facecolor="white")
    fig.savefig(destination / "class3_to_class2_residual_es_vs.pdf", facecolor="white")
    plt.close(fig)
    print(f"{section}: real run {run_id}, generated {generated_predicted[-1]:.3f} kWh, "
          f"real measured {real_energy[-1]:.3f} kWh")


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    plots = sorted(GENERATED.glob("*/class2_generated_vs_real_speed_distance.png"))
    if len(plots) != 13:
        raise ValueError(f"Expected 13 genuine Class 3 to Class 2 validation sections, got {len(plots)}")

    for plot in plots:
        section = plot.parent.name
        with (ARTIFACTS / section / "class3_phase_artifacts.pkl").open("rb") as file:
            artifact = pickle.load(file)
        if artifact.get("reference_class") != "class3":
            raise ValueError(f"{section} did not use a Class 3 parent template")
        generated = pd.read_csv(plot.parent / "class2_generated_curve.csv")
        real, n_runs, run_id = select_real_class2(section, float(artifact["target_l"]))
        real_distance = cumulative_distance(real)
        generated_distance = generated["dist_m"].to_numpy(float)
        mass_t = float(real["重量"].median())
        real_velocity = real["速度(m/s)"].clip(lower=0).to_numpy(float)
        generated_velocity = generated["velocity_mps"].to_numpy(float)
        real_gradient = real["gradient"].to_numpy(float)
        real_curvature = real["curvature"].to_numpy(float)
        generated_gradient = sample_track(real_gradient, real_distance, generated_distance)
        generated_curvature = sample_track(real_curvature, real_distance, generated_distance)
        generated_acceleration = np.r_[0.0, np.diff(generated_velocity) / DT]
        real_acceleration = real["加速度(m/s²)"].to_numpy(float)
        if not np.isfinite(real_acceleration).all():
            real_acceleration = np.r_[0.0, np.diff(real_velocity) / DT]
        assets = load_model(section, MODELS, torch.device("cpu"))
        real_predicted = evaluate_residual(
            section, real_velocity, real_distance, real_acceleration,
            real_gradient, real_curvature, mass_t, assets,
        )
        generated_predicted = evaluate_residual(
            section, generated_velocity, generated_distance, generated_acceleration,
            generated_gradient, generated_curvature, mass_t, assets,
        )
        measured_step_wh = real["energy"].clip(lower=0).to_numpy(float) / 3600.0
        measured_step_wh[:SEQ_LEN - 1] = 0.0
        real_energy = np.cumsum(measured_step_wh) / 1000.0
        draw(section, real, generated, real_energy, real_predicted,
             generated_predicted, n_runs, mass_t, run_id)
        summary.append({
            "section": section,
            "real_class2_run_id": run_id,
            "real_class2_available_runs": n_runs,
            "mass_t": mass_t,
            "real_runtime_s": (len(real) - 1) * DT,
            "generated_runtime_s": float(generated["time_s"].iloc[-1]),
            "real_measured_electric_kwh": real_energy[-1],
            "real_residual_prediction_kwh": real_predicted[-1],
            "generated_residual_prediction_kwh": generated_predicted[-1],
            "model_gap_on_real_kwh": real_predicted[-1] - real_energy[-1],
            "curve_substitution_gap_kwh": generated_predicted[-1] - real_predicted[-1],
            "generated_vs_measured_gap_kwh": generated_predicted[-1] - real_energy[-1],
            "generated_vs_measured_gap_pct": 100.0 * (
                generated_predicted[-1] - real_energy[-1]
            ) / real_energy[-1],
        })
    pd.DataFrame(summary).to_csv(OUTPUT / "comparison_summary.csv", index=False,
                                 encoding="utf-8-sig")
    print(f"Saved {len(summary)} figures and summary to {OUTPUT}")


if __name__ == "__main__":
    main()
