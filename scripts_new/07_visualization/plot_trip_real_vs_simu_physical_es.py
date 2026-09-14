# -*- coding: utf-8 -*-
"""Compare recorded trip energy with the pure Simu physics model by distance.

The selected local ``segment`` in every section comes from the global trip
traceability manifest.  This prevents unrelated local segment numbers from
being concatenated into one apparent full-line trip.

The physics curve is wheel-side positive traction mechanical energy.  It does
not include the residual model, auxiliary energy, conversion efficiency, or
regenerative-energy recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT_ROOT / "scripts_new" / "00_main_pipeline"
for import_dir in (PROJECT_ROOT, PIPELINE_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from src.physics.train_simu import TrainTheoreticalEnergyModel
from trip_traceability import (
    load_trip_traceability,
    resolve_traceability_manifest,
    select_trip_rows,
)


DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "data_processed_step2_v3_all_curve_quality_traceability"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "trip_real_vs_simu_physics"
CACHE_DIR = PROJECT_ROOT / "output" / "cache" / "planning_history_curves"
PARAM_DIR = PROJECT_ROOT / "data" / "static"
CACHE_VERSION = "history_trip_curve_v2_traceability"
FINAL_LINE_DISTANCE_M = 36502.0

REQUIRED_COLUMNS = ["segment", "时刻", "速度(m/s)", "累计位移(m)", "energy", "重量"]
_TRIP_MASS_TABLE_CACHE: dict[int, dict[str, float]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative recorded and Simu mechanical energy against mileage."
    )
    parser.add_argument("--trip-no", type=int, default=1, help="1-based global trip number.")
    parser.add_argument(
        "--trips",
        default=None,
        help="Optional comma-separated trip numbers; overrides --trip-no.",
    )
    parser.add_argument("--direction", default="UP", choices=["UP", "DOWN"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a trip only when its full-line plots, CSV files, and 26 section plots are complete.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_trip_numbers(text: str) -> list[int]:
    trips: list[int] = []
    for raw_value in text.split(","):
        value = raw_value.strip()
        if not value:
            continue
        trip_no = int(value)
        if trip_no < 1:
            raise ValueError(f"Trip number must be >= 1: {trip_no}")
        if trip_no not in trips:
            trips.append(trip_no)
    if not trips:
        raise ValueError("No trip numbers supplied.")
    return trips


def load_trip_section_mass(trip_no: int, section: str) -> float | None:
    """Read the same per-section mean mass used by the main energy pipeline."""

    if trip_no not in _TRIP_MASS_TABLE_CACHE:
        path = PARAM_DIR / f"section_params_trip{trip_no}.csv"
        if not path.is_file():
            _TRIP_MASS_TABLE_CACHE[trip_no] = {}
        else:
            table = pd.read_csv(path, encoding="utf-8-sig")
            required = {"station_pair", "MASS"}
            if not required.issubset(table.columns):
                missing = sorted(required - set(table.columns))
                raise ValueError(f"{path.name} missing columns: {missing}")
            masses = pd.to_numeric(table["MASS"], errors="coerce")
            _TRIP_MASS_TABLE_CACHE[trip_no] = {
                str(station_pair): float(mass)
                for station_pair, mass in zip(table["station_pair"], masses)
                if pd.notna(mass)
            }
    return _TRIP_MASS_TABLE_CACHE[trip_no].get(section)


def history_cache_path(
    excel_path: Path,
    section: str,
    trip_index: int,
    traceability,
) -> Path:
    """Use the same cache key as the DP history-curve loader."""

    stat = excel_path.stat()
    if traceability is not None:
        record = traceability.record_for(section)
        selection_key = f"{record.global_trip_id}|{record.segment}|{record.source_run_id}"
    else:
        selection_key = f"legacy_local_index|{trip_index}"
    key_source = "|".join(
        [
            CACHE_VERSION,
            str(excel_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            section,
            selection_key,
        ]
    )
    key = hashlib.sha1(key_source.encode("utf-8", errors="surrogatepass")).hexdigest()
    return CACHE_DIR / f"{key}.pkl"


def load_trip_section(
    data_dir: Path,
    section: str,
    trip_index: int,
    traceability,
) -> tuple[pd.DataFrame, object, object, str, str]:
    excel_path = data_dir / f"results_{section}.xlsx"
    if not excel_path.is_file():
        raise FileNotFoundError(excel_path)

    cache_path = history_cache_path(excel_path, section, trip_index, traceability)
    if cache_path.is_file():
        frame = pd.read_pickle(cache_path)
        required_cached = {"时刻", "速度(m/s)", "累计位移(m)", "energy"}
        has_mass = any(column in frame.columns for column in ("重量", "MASS", "mass"))
        if required_cached.issubset(frame.columns) and has_mass:
            record = traceability.record_for(section) if traceability is not None else None
            target_segment = record.segment if record is not None else ""
            return frame, target_segment, record, "traceability-cache", "cache"

        mass_t = load_trip_section_mass(trip_index + 1, section)
        if required_cached.issubset(frame.columns) and mass_t is not None:
            frame = frame.copy()
            frame["重量"] = mass_t
            frame.to_pickle(cache_path)
            record = traceability.record_for(section) if traceability is not None else None
            target_segment = record.segment if record is not None else ""
            return frame, target_segment, record, "traceability-cache+mass", "cache+mass"

        # Older DP caches intentionally omitted mass. The physical-energy plot
        # needs it. If no trip parameter table exists, rebuild that entry from
        # the source XLSX as the final fallback.
        print(f"  Refreshing incomplete cache for {section}: {cache_path.name}")

    all_rows = pd.read_excel(excel_path, usecols=lambda col: col in REQUIRED_COLUMNS)
    missing = sorted(set(REQUIRED_COLUMNS) - set(all_rows.columns))
    if missing:
        raise ValueError(f"{excel_path.name} missing columns: {missing}")
    frame, target_segment, record, selection_mode = select_trip_rows(
        all_rows,
        section,
        trip_index,
        traceability,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache_path)
    return frame, target_segment, record, selection_mode, "excel"


def first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"None of the required columns exists: {names}")


def clean_section_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mass_col = first_existing_column(frame, ("重量", "MASS", "mass"))
    selected = frame.rename(columns={mass_col: "_mass_t"}).copy()
    numeric_columns = ["时刻", "速度(m/s)", "累计位移(m)", "energy", "_mass_t"]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=numeric_columns).copy()
    if selected.empty:
        raise ValueError("Selected segment contains no complete time/speed/distance/energy/mass rows.")
    selected = selected.sort_values("时刻", kind="stable").reset_index(drop=True)
    return selected


def station_axis(section_names: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    starts = TrainTheoreticalEnergyModel.SECTION_START_DISTANCES_M.astype(float)
    ticks = np.append(starts, FINAL_LINE_DISTANCE_M) / 1000.0
    labels = [section_names[0].split("-", 1)[0]]
    labels.extend(section.split("-", 1)[1] for section in section_names)
    return ticks, labels


def build_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if args.trip_no < 1:
        raise ValueError("--trip-no must be >= 1")

    data_dir = resolve_project_path(args.data_dir)
    sections = TrainTheoreticalEnergyModel.STATION_PAIRS
    trip_index = args.trip_no - 1
    manifest_path = resolve_traceability_manifest(PROJECT_ROOT, data_dir)
    traceability = load_trip_traceability(manifest_path, args.trip_no, direction=args.direction)
    if traceability is None:
        raise FileNotFoundError(
            "No trip traceability manifest was found. Use the *_traceability Step2 directory."
        )
    traceability.require_sections(sections)

    point_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    real_offset_wh = 0.0
    physics_offset_wh = 0.0

    print(f"Trip: {args.trip_no} ({traceability.global_trip_id})")
    print(f"Manifest: {manifest_path}")
    for section_index, section in enumerate(sections):
        frame, segment, record, selection_mode, source = load_trip_section(
            data_dir,
            section,
            trip_index,
            traceability,
        )
        frame = clean_section_frame(frame)

        time_s = frame["时刻"].to_numpy(dtype=float)
        time_s = time_s - time_s[0]
        speed_mps = frame["速度(m/s)"].to_numpy(dtype=float)
        local_distance_m = frame["累计位移(m)"].to_numpy(dtype=float)
        local_distance_m = np.maximum.accumulate(local_distance_m - local_distance_m[0])
        mass_t = float(frame["_mass_t"].median())

        model = TrainTheoreticalEnergyModel()
        physics_step_wh = model.run_batch_simulation(
            time_s,
            speed_mps,
            mass_t,
            section_name=section,
        )
        real_step_wh = frame["energy"].to_numpy(dtype=float) / 3600.0
        row_count = min(len(frame), len(physics_step_wh), len(real_step_wh))
        if row_count == 0:
            raise ValueError(f"{section} has no aligned samples.")

        start_m = float(TrainTheoreticalEnergyModel.SECTION_START_DISTANCES_M[section_index])
        absolute_distance_m = start_m + local_distance_m[:row_count]
        section_real_cumulative_wh = np.cumsum(real_step_wh[:row_count])
        section_physics_cumulative_wh = np.cumsum(physics_step_wh[:row_count])
        real_cumulative_wh = real_offset_wh + section_real_cumulative_wh
        physics_cumulative_wh = physics_offset_wh + section_physics_cumulative_wh

        point_frames.append(
            pd.DataFrame(
                {
                    "区间": section,
                    "segment": segment,
                    "来源run_id": record.source_run_id if record is not None else "",
                    "绝对里程(km)": absolute_distance_m / 1000.0,
                    "区间里程(km)": local_distance_m[:row_count] / 1000.0,
                    "区间时间(s)": time_s[:row_count],
                    "速度(km/h)": speed_mps[:row_count] * 3.6,
                    "历史真实区间累计能耗(kWh)": section_real_cumulative_wh / 1000.0,
                    "Simu物理机械区间累计能耗(kWh)": section_physics_cumulative_wh / 1000.0,
                    "历史真实累计能耗(kWh)": real_cumulative_wh / 1000.0,
                    "Simu物理机械累计能耗(kWh)": physics_cumulative_wh / 1000.0,
                }
            )
        )

        section_real_wh = float(np.sum(real_step_wh[:row_count]))
        section_physics_wh = float(np.sum(physics_step_wh[:row_count]))
        section_difference_wh = section_physics_wh - section_real_wh
        section_difference_pct = (
            section_difference_wh / section_real_wh * 100.0
            if abs(section_real_wh) > 1e-12
            else np.nan
        )
        summary_rows.append(
            {
                "序号": section_index + 1,
                "区间": section,
                "segment": segment,
                "来源run_id": record.source_run_id if record is not None else "",
                "绝对开始时间": record.absolute_start if record is not None else "",
                "绝对结束时间": record.absolute_end if record is not None else "",
                "趟次选择方式": selection_mode,
                "数据读取来源": source,
                "样本数": row_count,
                "运行时间(s)": float(time_s[row_count - 1]),
                "区间里程(km)": float(local_distance_m[row_count - 1] / 1000.0),
                "重量(t)": mass_t,
                "历史真实能耗(kWh)": section_real_wh / 1000.0,
                "Simu物理机械能耗(kWh)": section_physics_wh / 1000.0,
                "Simu-真实(kWh)": section_difference_wh / 1000.0,
                "相对差异(%)": section_difference_pct,
            }
        )
        real_offset_wh += section_real_wh
        physics_offset_wh += section_physics_wh
        print(
            f"[{section_index + 1:02d}/{len(sections)}] {section} "
            f"segment={segment} [{source}] | real={section_real_wh / 1000:.3f} kWh, "
            f"physics={section_physics_wh / 1000:.3f} kWh"
        )

    points = pd.concat(point_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    totals = {
        "global_trip_id": traceability.global_trip_id,
        "real_kwh": real_offset_wh / 1000.0,
        "physics_kwh": physics_offset_wh / 1000.0,
    }
    totals["difference_kwh"] = totals["physics_kwh"] - totals["real_kwh"]
    totals["difference_pct"] = (
        totals["difference_kwh"] / totals["real_kwh"] * 100.0
        if abs(totals["real_kwh"]) > 1e-12
        else np.nan
    )
    return points, summary, totals


def plot_comparison(
    points: pd.DataFrame,
    totals: dict,
    trip_no: int,
    output_stem: Path,
    dpi: int,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(18, 8.5), constrained_layout=True)
    ax.plot(
        points["绝对里程(km)"],
        points["历史真实累计能耗(kWh)"],
        color="#202124",
        linewidth=2.2,
        label="历史真实记录能耗",
        zorder=3,
    )
    ax.plot(
        points["绝对里程(km)"],
        points["Simu物理机械累计能耗(kWh)"],
        color="#d62728",
        linewidth=2.1,
        linestyle="--",
        label="Simu物理机械能",
        zorder=3,
    )

    tick_positions, tick_labels = station_axis(TrainTheoreticalEnergyModel.STATION_PAIRS)
    for position in tick_positions:
        ax.axvline(position, color="#d9dde3", linewidth=0.65, zorder=0)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=58, ha="right", fontsize=8)
    ax.set_xlim(0.0, FINAL_LINE_DISTANCE_M / 1000.0)
    ax.set_xlabel("线路里程 (km)", fontsize=12)
    ax.set_ylabel("累计能耗 (kWh)", fontsize=12)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.8, alpha=0.85)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11)

    title = f"第 {trip_no} 趟 全线真实能耗与 Simu 物理机械能对比"
    subtitle = (
        f"真实记录 {totals['real_kwh']:.2f} kWh  |  "
        f"Simu机械能 {totals['physics_kwh']:.2f} kWh  |  "
        f"Simu-真实 {totals['difference_kwh']:+.2f} kWh "
        f"({totals['difference_pct']:+.2f}%)"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=16, pad=14)

    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".jpg"),
        dpi=dpi,
        facecolor="white",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)


def plot_section_comparisons(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    trip_no: int,
    output_dir: Path,
    dpi: int,
) -> None:
    """Save one local-distance cumulative-energy plot per section."""

    section_dir = output_dir / "sections"
    section_dir.mkdir(parents=True, exist_ok=True)

    for row in summary.to_dict(orient="records"):
        section_index = int(row["序号"])
        section = str(row["区间"])
        section_points = points.loc[points["区间"] == section].copy()
        if section_points.empty:
            continue

        distance_km = section_points["区间里程(km)"].to_numpy(dtype=float)
        real_kwh = section_points["历史真实区间累计能耗(kWh)"].to_numpy(dtype=float)
        physics_kwh = section_points["Simu物理机械区间累计能耗(kWh)"].to_numpy(dtype=float)

        # Keep a visible physical origin even if the first recorded sample has energy.
        distance_km = np.insert(distance_km, 0, 0.0)
        real_kwh = np.insert(real_kwh, 0, 0.0)
        physics_kwh = np.insert(physics_kwh, 0, 0.0)

        section_real_kwh = float(row["历史真实能耗(kWh)"])
        section_physics_kwh = float(row["Simu物理机械能耗(kWh)"])
        difference_kwh = float(row["Simu-真实(kWh)"])
        difference_pct = float(row["相对差异(%)"])

        fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
        ax.plot(
            distance_km,
            real_kwh,
            color="#202124",
            linewidth=2.2,
            label="历史真实记录能耗",
        )
        ax.plot(
            distance_km,
            physics_kwh,
            color="#d62728",
            linewidth=2.1,
            linestyle="--",
            label="Simu物理机械能",
        )
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("区间里程 (km)", fontsize=11)
        ax.set_ylabel("累计能耗 (kWh)", fontsize=11)
        ax.grid(color="#d9dde3", linewidth=0.8, alpha=0.85)
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)
        ax.set_title(
            f"第 {trip_no} 趟 {section} 真实能耗与 Simu 物理机械能对比\n"
            f"真实记录 {section_real_kwh:.2f} kWh  |  "
            f"Simu机械能 {section_physics_kwh:.2f} kWh  |  "
            f"Simu-真实 {difference_kwh:+.2f} kWh ({difference_pct:+.2f}%)",
            fontsize=13,
            pad=12,
        )

        output_stem = section_dir / (
            f"{section_index:02d}_{section}_real_vs_simu_physical_energy_distance"
        )
        fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, facecolor="white")
        fig.savefig(
            output_stem.with_suffix(".jpg"),
            dpi=dpi,
            facecolor="white",
            pil_kwargs={"quality": 95},
        )
        plt.close(fig)


def trip_output_stem(output_root: Path, trip_no: int) -> Path:
    return (
        output_root
        / f"trip{trip_no:03d}"
        / f"trip{trip_no:03d}_real_vs_simu_physical_energy_distance"
    )


def trip_outputs_complete(output_root: Path, trip_no: int) -> bool:
    output_stem = trip_output_stem(output_root, trip_no)
    section_dir = output_stem.parent / "sections"
    required = [
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".jpg"),
        output_stem.with_name(output_stem.name + "_points.csv"),
        output_stem.with_name(output_stem.name + "_sections.csv"),
    ]
    if not all(path.is_file() for path in required):
        return False
    return (
        len(list(section_dir.glob("*.png"))) == 26
        and len(list(section_dir.glob("*.jpg"))) == 26
    )


def totals_from_existing(
    output_root: Path,
    data_dir: Path,
    trip_no: int,
    direction: str,
) -> tuple[dict, pd.DataFrame]:
    output_stem = trip_output_stem(output_root, trip_no)
    summary_path = output_stem.with_name(output_stem.name + "_sections.csv")
    summary = pd.read_csv(summary_path)
    real_kwh = float(summary["历史真实能耗(kWh)"].sum())
    physics_kwh = float(summary["Simu物理机械能耗(kWh)"].sum())
    difference_kwh = physics_kwh - real_kwh
    manifest_path = resolve_traceability_manifest(PROJECT_ROOT, data_dir)
    traceability = load_trip_traceability(manifest_path, trip_no, direction=direction)
    global_trip_id = traceability.global_trip_id if traceability is not None else ""
    totals = {
        "global_trip_id": global_trip_id,
        "real_kwh": real_kwh,
        "physics_kwh": physics_kwh,
        "difference_kwh": difference_kwh,
        "difference_pct": (
            difference_kwh / real_kwh * 100.0 if abs(real_kwh) > 1e-12 else np.nan
        ),
    }
    return totals, summary


def run_trip(args: argparse.Namespace, trip_no: int) -> tuple[dict, pd.DataFrame]:
    output_root = resolve_project_path(args.output_dir)
    data_dir = resolve_project_path(args.data_dir)
    output_dir = output_root / f"trip{trip_no:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and trip_outputs_complete(output_root, trip_no):
        totals, summary = totals_from_existing(
            output_root,
            data_dir,
            trip_no,
            args.direction,
        )
        print(f"Trip {trip_no:03d}: reuse complete existing outputs.", flush=True)
        return totals, summary

    trip_args = argparse.Namespace(**vars(args))
    trip_args.trip_no = trip_no
    points, summary, totals = build_comparison(trip_args)
    output_stem = trip_output_stem(output_root, trip_no)
    points.to_csv(output_stem.with_name(output_stem.name + "_points.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_stem.with_name(output_stem.name + "_sections.csv"), index=False, encoding="utf-8-sig")
    plot_comparison(points, totals, trip_no, output_stem, args.dpi)
    plot_section_comparisons(points, summary, trip_no, output_dir, args.dpi)

    print(f"\nTrip {trip_no:03d} done")
    print(f"Real recorded: {totals['real_kwh']:.3f} kWh")
    print(f"Simu mechanical: {totals['physics_kwh']:.3f} kWh")
    print(
        f"Difference: {totals['difference_kwh']:+.3f} kWh "
        f"({totals['difference_pct']:+.3f}%)"
    )
    print(f"PNG: {output_stem.with_suffix('.png')}")
    print(f"JPG: {output_stem.with_suffix('.jpg')}")
    print(f"Sections: {output_stem.with_name(output_stem.name + '_sections.csv')}")
    print(f"Section plots: {output_dir / 'sections'} ({len(summary)} sections, PNG + JPG)")
    return totals, summary


def write_batch_summaries(
    output_root: Path,
    batch_rows: list[dict],
    section_frames: list[pd.DataFrame],
) -> tuple[Path, Path, Path]:
    detail = pd.DataFrame(batch_rows).sort_values("趟次").reset_index(drop=True)
    summary_path = output_root / "simu_history_comparison_summary.csv"
    detail.to_csv(summary_path, index=False, encoding="utf-8-sig")

    differences = detail["Simu-真实(kWh)"].to_numpy(dtype=float)
    percentages = detail["相对差异(%)"].to_numpy(dtype=float)
    real_total = float(detail["历史真实能耗(kWh)"].sum())
    physics_total = float(detail["Simu物理机械能耗(kWh)"].sum())
    statistics = pd.DataFrame(
        [
            {"统计指标": "趟次数", "数值": len(detail), "单位": "趟"},
            {
                "统计指标": "平均偏差",
                "数值": float(np.mean(differences)),
                "单位": "kWh",
            },
            {
                "统计指标": "平均相对偏差",
                "数值": float(np.mean(percentages)),
                "单位": "%",
            },
            {
                "统计指标": "平均绝对误差 MAE",
                "数值": float(np.mean(np.abs(differences))),
                "单位": "kWh",
            },
            {
                "统计指标": "平均绝对百分比误差 MAPE",
                "数值": float(np.mean(np.abs(percentages))),
                "单位": "%",
            },
            {
                "统计指标": "均方根误差 RMSE",
                "数值": float(np.sqrt(np.mean(np.square(differences)))),
                "单位": "kWh",
            },
            {
                "统计指标": "13趟历史真实能耗合计",
                "数值": real_total,
                "单位": "kWh",
            },
            {
                "统计指标": "13趟Simu机械能合计",
                "数值": physics_total,
                "单位": "kWh",
            },
            {
                "统计指标": "总体差异率",
                "数值": (
                    (physics_total - real_total) / real_total * 100.0
                    if abs(real_total) > 1e-12
                    else np.nan
                ),
                "单位": "%",
            },
        ]
    )
    statistics_path = output_root / "simu_history_comparison_statistics.csv"
    statistics.to_csv(statistics_path, index=False, encoding="utf-8-sig")

    section_detail = pd.concat(section_frames, ignore_index=True)
    section_path = output_root / "simu_history_section_comparison_summary.csv"
    section_detail.to_csv(section_path, index=False, encoding="utf-8-sig")
    return summary_path, statistics_path, section_path


def main() -> int:
    args = parse_args()
    trips = parse_trip_numbers(args.trips) if args.trips else [args.trip_no]
    output_root = resolve_project_path(args.output_dir)
    batch_rows: list[dict] = []
    section_frames: list[pd.DataFrame] = []

    for trip_no in trips:
        totals, summary = run_trip(args, trip_no)
        batch_rows.append(
            {
                "趟次": trip_no,
                "全局趟次ID": totals["global_trip_id"],
                "历史真实能耗(kWh)": totals["real_kwh"],
                "Simu物理机械能耗(kWh)": totals["physics_kwh"],
                "Simu-真实(kWh)": totals["difference_kwh"],
                "相对差异(%)": totals["difference_pct"],
                "绝对差异(kWh)": abs(totals["difference_kwh"]),
                "绝对相对差异(%)": abs(totals["difference_pct"]),
                "全线图": str(trip_output_stem(output_root, trip_no).with_suffix(".png")),
            }
        )
        section_copy = summary.copy()
        section_copy.insert(0, "趟次", trip_no)
        section_copy.insert(1, "全局趟次ID", totals["global_trip_id"])
        section_frames.append(section_copy)

    summary_path, statistics_path, section_path = write_batch_summaries(
        output_root,
        batch_rows,
        section_frames,
    )
    print(f"\nBatch done: {len(trips)} trip(s)")
    print(f"Trip summary: {summary_path}")
    print(f"Statistics:   {statistics_path}")
    print(f"Section data: {section_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
