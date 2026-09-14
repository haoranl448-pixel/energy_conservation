# -*- coding: utf-8 -*-
"""Compare code, Simu physics, and OpenTrack results trip by trip.

Each trip produces four 1x2 comparison outputs.

1. historical operation;
2. standard-priority DP;
3. energy-first DP;
4. real-priority DP with the 5% dwell allowance.

The left panel is cumulative energy against line mileage (E-s).  The right
panel is velocity against line mileage (v-s).  Planned code energy uses the
legacy physical model point by point.  Because the old energy menu retained
only each section's residual total, that residual is recorded as a boundary
jump instead of inventing an unsupported within-section distribution.

Simu consumes the code velocity trace directly.  Therefore the code and Simu
velocity traces overlap by definition; their energy traces differ because the
code total includes the existing learned correction while Simu is pure
positive wheel-side traction mechanical energy.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\energy_conservation")
SCRIPT_DIR = Path(__file__).resolve().parent
for import_dir in (PROJECT_ROOT, SCRIPT_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from src.physics.train_simu import TrainTheoreticalEnergyModel
from src.physics.train_simu_v1_backup_20260828 import (
    TrainTheoreticalEnergyModel as LegacyPlanningEnergyModel,
)
from plot_opentrack_tsvp_vs_history import read_tsvp


DEFAULT_TRIPS = "1,3,4,5,6,8,10,12,14,15,42,62,67"
DEFAULT_TSVP_DIR = Path(r"D:\OutPut")
DEFAULT_PLAN_ROOT = (
    PROJECT_ROOT
    / "output"
    / "schedule"
    / "batch_trip_reports_complete13_dp_unified"
)
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"
DEFAULT_HISTORY_ROOT = (
    PROJECT_ROOT / "output" / "analysis" / "trip_real_vs_simu_physics"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "simu_ot_code_trip_compare_complete13"
)
FINAL_LINE_DISTANCE_M = 36502.0
SUPPORTED_EXPORT_SUFFIXES = (".png", ".jpg", ".svg", ".pdf", ".tif", ".tiff")


@dataclass(frozen=True)
class Scenario:
    key: str
    order: int
    display_name: str
    plan_file: str | None
    ot_template: str


SCENARIOS = (
    Scenario(
        key="history",
        order=0,
        display_name="规划前：历史运行",
        plan_file=None,
        ot_template="OT_priority_history_trip{trip:03d}.tsvP",
    ),
    Scenario(
        key="standard_dp",
        order=1,
        display_name="规划后：标准 DP",
        plan_file="01_standard_priority.csv",
        ot_template="OT_priority_dp_trip{trip:03d}.tsvP",
    ),
    Scenario(
        key="energy_first",
        order=2,
        display_name="规划后：Energy First",
        plan_file="02_energy_first.csv",
        ot_template="OT_energy_first_dp_trip{trip:03d}.tsvP",
    ),
    Scenario(
        key="dwell5_dp",
        order=3,
        display_name="规划后：历史停站时间放宽 5% DP",
        plan_file="03_real_priority_dwell_5pct.csv",
        ot_template="OT_real_priority_dwell_5pct_dp_trip{trip:03d}.tsvP",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw per-trip code/Simu/OpenTrack E-s and v-s comparisons."
    )
    parser.add_argument("--trips", default=DEFAULT_TRIPS)
    parser.add_argument(
        "--scenarios",
        default="history,standard_dp,energy_first,dwell5_dp",
        help="Comma-separated scenario keys.",
    )
    parser.add_argument("--tsvp-dir", type=Path, default=DEFAULT_TSVP_DIR)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        default="png",
        help=(
            "Comma-separated output formats: png, jpg, svg, pdf, tif, or tiff. "
            "TIFF is always exported at 600 dpi."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value < 1:
            raise ValueError(f"Trip number must be >= 1: {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("No trip numbers supplied.")
    return values


def parse_scenarios(text: str) -> list[Scenario]:
    wanted = [value.strip() for value in text.split(",") if value.strip()]
    by_key = {scenario.key: scenario for scenario in SCENARIOS}
    unknown = [key for key in wanted if key not in by_key]
    if unknown:
        raise ValueError(f"Unknown scenario key(s): {unknown}")
    return [by_key[key] for key in wanted]


def parse_formats(text: str) -> list[str]:
    formats = [value.strip().lower() for value in text.split(",") if value.strip()]
    allowed = {suffix.lstrip(".") for suffix in SUPPORTED_EXPORT_SUFFIXES} | {"jpeg"}
    if not formats or any(value not in allowed for value in formats):
        raise ValueError(f"--formats must contain only {sorted(allowed)}")
    return list(dict.fromkeys(formats))


def finite_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    selected = frame.copy()
    for column in columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected.dropna(subset=columns).copy()


def normalize_curve(curve: dict[str, Any]) -> dict[str, np.ndarray]:
    distance = np.asarray(curve["distance_km"], dtype=float)
    speed = np.asarray(curve["speed_kmh"], dtype=float)
    speed_count = min(distance.size, speed.size)
    if speed_count < 2:
        raise ValueError("Curve contains fewer than two distance/speed points.")
    distance = distance[:speed_count]
    speed = speed[:speed_count]
    speed_mask = np.isfinite(distance) & np.isfinite(speed)
    distance = distance[speed_mask]
    speed = speed[speed_mask]
    if distance.size < 2:
        raise ValueError("Curve contains fewer than two finite distance/speed points.")

    energy_distance = np.asarray(
        curve.get("energy_distance_km", curve["distance_km"]),
        dtype=float,
    )
    energy = np.asarray(curve["energy_kwh"], dtype=float)
    energy_count = min(energy_distance.size, energy.size)
    if energy_count < 2:
        raise ValueError("Curve contains fewer than two distance/energy points.")
    energy_distance = energy_distance[:energy_count]
    energy = energy[:energy_count]
    energy_mask = np.isfinite(energy_distance) & np.isfinite(energy)
    energy_distance = energy_distance[energy_mask]
    energy = energy[energy_mask]
    if energy_distance.size < 2:
        raise ValueError("Curve contains fewer than two finite distance/energy points.")

    distance = np.maximum.accumulate(distance - distance[0])
    energy_distance = np.maximum.accumulate(energy_distance - energy_distance[0])
    energy = energy - energy[0]
    energy[0] = 0.0
    return {
        "distance_km": distance,
        "speed_kmh": speed,
        "energy_distance_km": energy_distance,
        "energy_kwh": energy,
    }


def history_points_path(history_root: Path, trip: int) -> Path:
    stem = f"trip{trip:03d}_real_vs_simu_physical_energy_distance_points.csv"
    return history_root / f"trip{trip:03d}" / stem


def read_history_code_and_simu(
    history_root: Path,
    trip: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = history_points_path(history_root, trip)
    if not path.is_file():
        raise FileNotFoundError(
            f"Historical code/Simu points not found: {path}. "
            "Run plot_trip_real_vs_simu_physical_es.py first."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    columns = [
        "绝对里程(km)",
        "速度(km/h)",
        "历史真实累计能耗(kWh)",
        "Simu物理机械累计能耗(kWh)",
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    frame = finite_numeric(frame, columns)
    code = normalize_curve(
        {
            "distance_km": frame["绝对里程(km)"],
            "speed_kmh": frame["速度(km/h)"],
            "energy_kwh": frame["历史真实累计能耗(kWh)"],
        }
    )
    simu = normalize_curve(
        {
            "distance_km": frame["绝对里程(km)"],
            "speed_kmh": frame["速度(km/h)"],
            "energy_kwh": frame["Simu物理机械累计能耗(kWh)"],
        }
    )
    return code, simu


def read_generated_curve(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "velocity_mps" not in frame.columns and "velocity_kmh" in frame.columns:
        frame["velocity_mps"] = pd.to_numeric(
            frame["velocity_kmh"], errors="coerce"
        ) / 3.6
    required = ["time_s", "dist_m", "velocity_mps"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    frame = finite_numeric(frame, required)
    frame = frame.sort_values("time_s", kind="stable").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f"No usable generated trajectory rows: {path}")
    return frame


def calculate_pointwise_energy_profiles(
    section: str,
    time_s: np.ndarray,
    speed_mps: np.ndarray,
    mass_t: float,
    cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
    cache_class: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return new-Simu and legacy-planning physical step energy."""

    cache_key = (section, cache_class, round(float(mass_t), 6))
    cached = cache.get(cache_key)
    if cached is not None:
        new_simu_steps, legacy_steps = cached
        return new_simu_steps.copy(), legacy_steps.copy()

    new_simu_steps = TrainTheoreticalEnergyModel().run_batch_simulation(
        time_s,
        speed_mps,
        float(mass_t),
        section_name=section,
    )
    legacy_steps = LegacyPlanningEnergyModel().run_batch_simulation(
        time_s,
        speed_mps,
        float(mass_t),
    )
    point_count = min(len(time_s), len(new_simu_steps), len(legacy_steps))
    if point_count < 30:
        raise ValueError(f"{section} {cache_class}: fewer than 30 points")
    new_simu_steps = np.asarray(new_simu_steps[:point_count], dtype=float)
    legacy_steps = np.asarray(legacy_steps[:point_count], dtype=float)
    cache[cache_key] = (
        new_simu_steps.copy(),
        legacy_steps.copy(),
    )
    return new_simu_steps, legacy_steps


def planned_code_and_simu(
    plan_path: Path,
    trajectory_dir: Path,
    profile_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[str],
]:
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = pd.read_csv(plan_path, encoding="utf-8-sig")
    row_by_section = {
        str(row.get("站间区间", "")).strip(): row
        for _, row in plan.iterrows()
        if str(row.get("站间区间", "")).strip()
    }

    sections = TrainTheoreticalEnergyModel.STATION_PAIRS
    starts_m = TrainTheoreticalEnergyModel.SECTION_START_DISTANCES_M.astype(float)
    ends_m = np.append(starts_m[1:], FINAL_LINE_DISTANCE_M)

    code_distance: list[float] = []
    code_speed: list[float] = []
    code_energy_distance: list[float] = []
    code_energy: list[float] = []
    simu_distance: list[float] = []
    simu_speed: list[float] = []
    simu_energy: list[float] = []
    section_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    code_offset_kwh = 0.0
    simu_offset_kwh = 0.0

    for section_index, section in enumerate(sections):
        row = row_by_section.get(section)
        if row is None:
            raise ValueError(f"{plan_path.name}: missing section {section}")
        selected_class = str(row.get("选定等级", "")).strip()
        planned_time_s = pd.to_numeric(row.get("规划用时(s)"), errors="coerce")
        code_energy_wh = pd.to_numeric(row.get("规划能耗(Wh)"), errors="coerce")
        mass_t = pd.to_numeric(row.get("MASS"), errors="coerce")
        if (
            not selected_class
            or not math.isfinite(float(planned_time_s))
            or not math.isfinite(float(code_energy_wh))
            or not math.isfinite(float(mass_t))
        ):
            raise ValueError(f"{plan_path.name}: incomplete plan row for {section}")

        curve_path = trajectory_dir / section / f"{selected_class}_generated_curve.csv"
        curve = read_generated_curve(curve_path)
        time_s = curve["time_s"].to_numpy(dtype=float)
        time_s = time_s - time_s[0]
        local_distance_m = curve["dist_m"].to_numpy(dtype=float)
        local_distance_m = np.maximum.accumulate(local_distance_m - local_distance_m[0])
        speed_mps = curve["velocity_mps"].to_numpy(dtype=float)
        speed_kmh = speed_mps * 3.6

        raw_duration_s = float(time_s[-1])
        duration_error_s = raw_duration_s - float(planned_time_s)
        if abs(duration_error_s) > 0.11:
            warnings.append(
                f"{section} {selected_class}: generated duration {raw_duration_s:.3f}s "
                f"differs from plan {float(planned_time_s):.3f}s by {duration_error_s:+.3f}s"
            )

        local_end_m = float(local_distance_m[-1])
        if local_end_m <= 0:
            raise ValueError(f"{curve_path}: non-positive final distance")
        progress = np.clip(local_distance_m / local_end_m, 0.0, 1.0)
        station_length_m = float(ends_m[section_index] - starts_m[section_index])
        absolute_distance_km = (
            starts_m[section_index] + progress * station_length_m
        ) / 1000.0

        simu_step_wh, legacy_step_wh = calculate_pointwise_energy_profiles(
            section,
            time_s,
            speed_mps,
            float(mass_t),
            profile_cache,
            selected_class,
        )
        point_count = min(
            len(time_s),
            len(simu_step_wh),
            len(legacy_step_wh),
        )
        if point_count < 30:
            raise ValueError(f"Energy models returned no usable rows for {section}")

        target_code_total_wh = float(code_energy_wh)
        simu_step_wh = np.asarray(simu_step_wh[:point_count], dtype=float)
        legacy_step_wh = np.asarray(legacy_step_wh[:point_count], dtype=float)
        aligned_legacy_step_wh = np.zeros(point_count, dtype=float)
        aligned_legacy_step_wh[29:] = legacy_step_wh[29:]

        absolute_distance_km = absolute_distance_km[:point_count]
        speed_kmh = speed_kmh[:point_count]
        local_code_physical_kwh = np.cumsum(aligned_legacy_step_wh) / 1000.0
        local_simu_kwh = np.cumsum(simu_step_wh) / 1000.0

        start_index = 0 if not code_distance else 1
        code_distance.extend(absolute_distance_km[start_index:])
        code_speed.extend(speed_kmh[start_index:])
        code_energy_distance.extend(absolute_distance_km[start_index:])
        code_energy.extend(
            code_offset_kwh + local_code_physical_kwh[start_index:]
        )
        simu_distance.extend(absolute_distance_km[start_index:])
        simu_speed.extend(speed_kmh[start_index:])
        simu_energy.extend(simu_offset_kwh + local_simu_kwh[start_index:])

        section_code_kwh = target_code_total_wh / 1000.0
        section_legacy_physical_kwh = float(local_code_physical_kwh[-1])
        section_residual_kwh = section_code_kwh - section_legacy_physical_kwh
        section_simu_kwh = float(local_simu_kwh[-1])

        # The old menu retained only the residual section total.  A duplicate
        # boundary x-coordinate records that correction without inventing its
        # unknown within-section distribution.
        code_energy_distance.append(float(absolute_distance_km[-1]))
        code_energy.append(code_offset_kwh + section_code_kwh)

        section_rows.append(
            {
                "区间序号": section_index + 1,
                "区间": section,
                "选定等级": selected_class,
                "规划用时(s)": float(planned_time_s),
                "生成曲线用时(s)": raw_duration_s,
                "曲线与规划时间差(s)": duration_error_s,
                "重量(t)": float(mass_t),
                "代码规划能耗(kWh)": section_code_kwh,
                "旧模型物理能耗(kWh)": section_legacy_physical_kwh,
                "区间边界补记残差(kWh)": section_residual_kwh,
                "Simu物理机械能耗(kWh)": section_simu_kwh,
                "Simu-代码(kWh)": section_simu_kwh - section_code_kwh,
                "Simu相对代码误差(%)": percent_error(
                    section_simu_kwh,
                    section_code_kwh,
                ),
            }
        )
        code_offset_kwh += section_code_kwh
        simu_offset_kwh += section_simu_kwh

    code = normalize_curve(
        {
            "distance_km": code_distance,
            "speed_kmh": code_speed,
            "energy_distance_km": code_energy_distance,
            "energy_kwh": code_energy,
        }
    )
    simu = normalize_curve(
        {
            "distance_km": simu_distance,
            "speed_kmh": simu_speed,
            "energy_kwh": simu_energy,
        }
    )
    return code, simu, section_rows, warnings
def percent_error(primary: float, reference: float) -> float | None:
    if not math.isfinite(primary) or not math.isfinite(reference):
        return None
    if abs(reference) < 1e-12:
        return None
    return (primary - reference) / reference * 100.0


def curve_total(curve: dict[str, np.ndarray]) -> float:
    return float(curve["energy_kwh"][-1])


def curve_distance(curve: dict[str, np.ndarray]) -> float:
    return float(curve["distance_km"][-1])


def format_error(primary_label: str, reference_label: str, primary: float, reference: float) -> str:
    diff = primary - reference
    pct = percent_error(primary, reference)
    pct_text = "N/A" if pct is None else f"{pct:+.2f}%"
    return f"{primary_label}-{reference_label}: {diff:+.2f} kWh ({pct_text})"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "lines.antialiased": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_three_way(
    trip: int,
    scenario: Scenario,
    code: dict[str, np.ndarray],
    simu: dict[str, np.ndarray],
    ot: dict[str, np.ndarray],
    max_mass_t: float,
    output_stem: Path,
    formats: list[str],
    dpi: int,
    show: bool,
) -> None:
    configure_matplotlib()
    fig, (ax_energy, ax_speed) = plt.subplots(1, 2, figsize=(16, 6.4))

    colors = {"code": "#222222", "simu": "#D62728", "ot": "#1F77B4"}
    code_energy_label = (
        "历史真实能耗"
        if scenario.key == "history"
        else "规划代码能耗（旧物理逐点+区间残差）"
    )
    ax_energy.plot(
        code["energy_distance_km"],
        code["energy_kwh"],
        color=colors["code"],
        lw=2.3,
        label=code_energy_label,
    )
    ax_energy.plot(
        simu["energy_distance_km"],
        simu["energy_kwh"],
        color=colors["simu"],
        lw=2.0,
        ls="--",
        label="Simu物理机械能",
    )
    ax_energy.plot(
        ot["energy_distance_km"],
        ot["energy_kwh"],
        color=colors["ot"],
        lw=1.9,
        label="OpenTrack机械能",
    )
    ax_energy.set_title("累计能耗-里程（E-s）")
    ax_energy.set_xlabel("线路里程 (km)")
    ax_energy.set_ylabel("累计能耗 (kWh)")

    code_total = curve_total(code)
    simu_total = curve_total(simu)
    ot_total = curve_total(ot)
    error_text = "\n".join(
        [
            f"代码: {code_total:.2f} kWh",
            f"Simu: {simu_total:.2f} kWh",
            f"OT: {ot_total:.2f} kWh",
            "",
            format_error("Simu", "代码", simu_total, code_total),
            format_error("OT", "代码", ot_total, code_total),
            format_error("OT", "Simu", ot_total, simu_total),
        ]
    )
    ax_energy.text(
        0.025,
        0.97,
        error_text,
        transform=ax_energy.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#9CA3AF",
            "alpha": 0.94,
        },
    )

    # Simu consumes the code velocity directly. Draw the wider code line first
    # and a narrow dashed Simu line on top so the expected overlap stays visible.
    ax_speed.plot(
        code["distance_km"],
        code["speed_kmh"],
        color=colors["code"],
        lw=3.0,
        alpha=0.75,
        label="代码速度",
    )
    ax_speed.plot(
        simu["distance_km"],
        simu["speed_kmh"],
        color=colors["simu"],
        lw=1.35,
        ls=(0, (4, 2)),
        label="Simu输入速度（与代码重合）",
    )
    ax_speed.plot(
        ot["distance_km"],
        ot["speed_kmh"],
        color=colors["ot"],
        lw=1.8,
        label="OpenTrack仿真速度",
    )
    ax_speed.set_title("速度-里程（v-s）")
    ax_speed.set_xlabel("线路里程 (km)")
    ax_speed.set_ylabel("速度 (km/h)")

    station_km = np.append(
        TrainTheoreticalEnergyModel.SECTION_START_DISTANCES_M,
        FINAL_LINE_DISTANCE_M,
    ) / 1000.0
    max_distance = max(curve_distance(code), curve_distance(simu), curve_distance(ot))
    for ax in (ax_energy, ax_speed):
        for station in station_km:
            ax.axvline(station, color="#D7DEE3", lw=0.55, alpha=0.55, zorder=0)
        ax.set_xlim(0.0, max_distance * 1.005)
        ax.grid(True, color="#CBD5E1", alpha=0.38, lw=0.65)
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"第 {trip:03d} 趟 | {scenario.display_name} | "
        f"代码、Simu 与 OpenTrack 对比 | 最大载重 {max_mass_t:.2f} t",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "误差定义：(前者-后者)/后者×100%；规划代码按旧物理模型逐点累计，未保存的残差总量仅在区间边界补记。",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.015, 0.045, 0.995, 0.945), w_pad=2.4)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for image_format in formats:
        suffix = "jpg" if image_format == "jpeg" else image_format
        save_dpi = 600 if suffix in {"tif", "tiff"} else max(int(dpi), 300)
        fig.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    if show:
        plt.show()
    plt.close(fig)


def read_max_mass(plan_path: Path) -> float:
    plan = pd.read_csv(plan_path, encoding="utf-8-sig")
    if "MASS" not in plan.columns:
        raise ValueError(f"{plan_path.name} has no MASS column")
    masses = pd.to_numeric(plan["MASS"], errors="coerce")
    masses = masses[np.isfinite(masses) & (masses > 0)]
    if masses.empty:
        raise ValueError(f"{plan_path.name} has no positive MASS values")
    return float(masses.max())


def output_stem(output_dir: Path, trip: int, scenario: Scenario) -> Path:
    return (
        output_dir
        / f"trip{trip:03d}"
        / f"{scenario.order:02d}_{scenario.key}_code_simu_ot_es_vs"
    )


def all_outputs_exist(stem: Path, formats: list[str]) -> bool:
    return all(stem.with_suffix(f".{('jpg' if value == 'jpeg' else value)}").is_file() for value in formats)


def summary_row(
    trip: int,
    scenario: Scenario,
    code: dict[str, np.ndarray],
    simu: dict[str, np.ndarray],
    ot: dict[str, np.ndarray],
    max_mass_t: float,
    stem: Path,
) -> dict[str, Any]:
    code_total = curve_total(code)
    simu_total = curve_total(simu)
    ot_total = curve_total(ot)
    return {
        "趟次": trip,
        "场景": scenario.key,
        "场景名称": scenario.display_name,
        "最大载重(t)": max_mass_t,
        "代码总能耗(kWh)": code_total,
        "Simu物理机械能耗(kWh)": simu_total,
        "OpenTrack机械能耗(kWh)": ot_total,
        "Simu-代码(kWh)": simu_total - code_total,
        "Simu相对代码误差(%)": percent_error(simu_total, code_total),
        "OT-代码(kWh)": ot_total - code_total,
        "OT相对代码误差(%)": percent_error(ot_total, code_total),
        "OT-Simu(kWh)": ot_total - simu_total,
        "OT相对Simu误差(%)": percent_error(ot_total, simu_total),
        "代码里程(km)": curve_distance(code),
        "Simu里程(km)": curve_distance(simu),
        "OT里程(km)": curve_distance(ot),
        "代码能耗曲线口径": (
            "历史逐点实测累计能耗"
            if scenario.key == "history"
            else "规划期旧物理模型逐点累计，残差总量在区间边界补记"
        ),
        "图文件": str(stem.with_suffix(".png")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preflight(
    trips: list[int],
    scenarios: list[Scenario],
    args: argparse.Namespace,
) -> None:
    missing: list[Path] = []
    sections = TrainTheoreticalEnergyModel.STATION_PAIRS
    for trip in trips:
        standard_plan = args.plan_root / f"trip{trip:03d}" / "01_standard_priority.csv"
        if not standard_plan.is_file():
            missing.append(standard_plan)
        for scenario in scenarios:
            ot_path = args.tsvp_dir / scenario.ot_template.format(trip=trip)
            if not ot_path.is_file():
                missing.append(ot_path)
            if scenario.plan_file is None:
                points_path = history_points_path(args.history_root, trip)
                if not points_path.is_file():
                    missing.append(points_path)
                continue
            plan_path = args.plan_root / f"trip{trip:03d}" / scenario.plan_file
            if not plan_path.is_file():
                missing.append(plan_path)
                continue
            plan = pd.read_csv(plan_path, encoding="utf-8-sig")
            if "站间区间" not in plan.columns or "选定等级" not in plan.columns:
                raise ValueError(f"{plan_path}: missing 站间区间/选定等级")
            class_by_section = dict(zip(plan["站间区间"], plan["选定等级"]))
            for section in sections:
                selected_class = str(class_by_section.get(section, "")).strip()
                curve_path = (
                    args.trajectory_dir
                    / section
                    / f"{selected_class}_generated_curve.csv"
                )
                if not selected_class or not curve_path.is_file():
                    missing.append(curve_path)
    if missing:
        unique = list(dict.fromkeys(missing))
        preview = "\n".join(f"- {path}" for path in unique[:30])
        extra = "" if len(unique) <= 30 else f"\n... and {len(unique) - 30} more"
        raise FileNotFoundError(f"Preflight found {len(unique)} missing inputs:\n{preview}{extra}")


def main() -> int:
    args = parse_args()
    trips = parse_int_list(args.trips)
    scenarios = parse_scenarios(args.scenarios)
    formats = parse_formats(args.formats)
    preflight(trips, scenarios, args)
    profile_cache: dict[
        tuple[str, str, float],
        tuple[np.ndarray, np.ndarray],
    ] = {}

    print(f"Trips: {trips}", flush=True)
    print(f"Scenarios: {[scenario.key for scenario in scenarios]}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    for trip in trips:
        standard_plan = args.plan_root / f"trip{trip:03d}" / "01_standard_priority.csv"
        max_mass_t = read_max_mass(standard_plan)
        history_code: dict[str, np.ndarray] | None = None
        history_simu: dict[str, np.ndarray] | None = None

        print(f"\nTrip {trip:03d} | max mass {max_mass_t:.2f} t", flush=True)
        for scenario in scenarios:
            stem = output_stem(args.output_dir, trip, scenario)
            ot_path = args.tsvp_dir / scenario.ot_template.format(trip=trip)
            ot = normalize_curve(read_tsvp(ot_path))

            if scenario.plan_file is None:
                if history_code is None or history_simu is None:
                    history_code, history_simu = read_history_code_and_simu(
                        args.history_root,
                        trip,
                    )
                code, simu = history_code, history_simu
                current_section_rows: list[dict[str, Any]] = []
                current_warnings: list[str] = []
            else:
                plan_path = args.plan_root / f"trip{trip:03d}" / scenario.plan_file
                code, simu, current_section_rows, current_warnings = planned_code_and_simu(
                    plan_path,
                    args.trajectory_dir,
                    profile_cache,
                )

            summary_rows.append(
                summary_row(trip, scenario, code, simu, ot, max_mass_t, stem)
            )
            for row in current_section_rows:
                section_rows.append(
                    {"趟次": trip, "场景": scenario.key, "场景名称": scenario.display_name, **row}
                )
            for warning in current_warnings:
                warning_rows.append(
                    {"趟次": trip, "场景": scenario.key, "警告": warning}
                )

            if args.skip_existing and all_outputs_exist(stem, formats):
                print(f"  {scenario.key}: reuse existing plot", flush=True)
            else:
                plot_three_way(
                    trip,
                    scenario,
                    code,
                    simu,
                    ot,
                    max_mass_t,
                    stem,
                    formats,
                    args.dpi,
                    args.show,
                )
                print(f"  {scenario.key}: plotted", flush=True)

            row = summary_rows[-1]
            print(
                f"    code={row['代码总能耗(kWh)']:.2f}, "
                f"Simu={row['Simu物理机械能耗(kWh)']:.2f} "
                f"({row['Simu相对代码误差(%)']:+.2f}%), "
                f"OT={row['OpenTrack机械能耗(kWh)']:.2f} "
                f"({row['OT相对代码误差(%)']:+.2f}%)",
                flush=True,
            )

    summary_path = args.output_dir / "code_simu_ot_trip_scenario_summary.csv"
    sections_path = args.output_dir / "planned_simu_section_detail.csv"
    warnings_path = args.output_dir / "input_consistency_warnings.csv"
    write_csv(summary_path, summary_rows)
    write_csv(sections_path, section_rows)
    write_csv(warnings_path, warning_rows)
    print(f"\nSummary: {summary_path}", flush=True)
    print(f"Sections: {sections_path}", flush=True)
    print(f"Warnings: {warnings_path} ({len(warning_rows)})", flush=True)
    print(f"Figures: {len(trips) * len(scenarios)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
