# -*- coding: utf-8 -*-
"""Render cached code/Simu results together with OpenTrack TSVP outputs.

This script is deliberately render-only. It reuses the audited 25-trip replay
cache produced by ``plot_simu_code_trip_compare.py`` and never reruns physics,
residual inference, DP optimization, or OpenTrack.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_opentrack_tsvp_vs_history import read_tsvp


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_TRIPS = "1,3,4,5,6,8,10,12,14,15,16,19,21,35,40,42,44,49,50,53,57,60,62,66,67"
DEFAULT_STATE_DIR = (
    PROJECT_ROOT
    / "output"
    / "cache"
    / "simu_code_replay"
    / "batch_simu_code_trip_compare_complete25"
)
DEFAULT_TSVP_DIR = Path(r"D:\OutPut")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "simu_ot_code_trip_compare_complete25"
FINAL_DISTANCE_KM = 36.502

SCENARIOS = {
    "history": {
        "order": 0,
        "stem": "00_history",
        "label": "历史运行",
        "ot_template": "OT_priority_history_trip{trip:03d}.tsvP",
    },
    "standard_dp": {
        "order": 1,
        "stem": "01_standard_dp",
        "label": "标准 DP",
        "ot_template": "OT_priority_dp_trip{trip:03d}.tsvP",
    },
    "energy_first": {
        "order": 2,
        "stem": "02_energy_first",
        "label": "Energy First",
        "ot_template": "OT_energy_first_dp_trip{trip:03d}.tsvP",
    },
    "dwell5_dp": {
        "order": 3,
        "stem": "03_dwell5_dp",
        "label": "停站放宽 5% DP",
        "ot_template": "OT_real_priority_dwell_5pct_dp_trip{trip:03d}.tsvP",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw 25-trip code/Simu/OpenTrack E-s and v-s comparisons."
    )
    parser.add_argument("--trips", default=DEFAULT_TRIPS)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--tsvp-dir", type=Path, default=DEFAULT_TSVP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", default="png")
    return parser.parse_args()


def parse_trips(text: str) -> list[int]:
    trips = list(dict.fromkeys(int(raw.strip()) for raw in text.split(",") if raw.strip()))
    if not trips or min(trips) < 1:
        raise ValueError("Positive trip numbers are required.")
    return trips


def parse_formats(text: str) -> list[str]:
    formats = list(dict.fromkeys(raw.strip().lower() for raw in text.split(",") if raw.strip()))
    allowed = {"png", "jpg", "jpeg", "svg", "pdf", "tif", "tiff"}
    if not formats or any(value not in allowed for value in formats):
        raise ValueError(f"Formats must be selected from {sorted(allowed)}")
    return formats


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def percent(delta: float, reference: float) -> float:
    return delta / reference * 100.0 if abs(reference) > 1e-12 else float("nan")


def finite_curve(curve: dict[str, Any]) -> dict[str, np.ndarray]:
    distance = np.asarray(curve["distance_km"], dtype=float)
    speed = np.asarray(curve["speed_kmh"], dtype=float)
    energy = np.asarray(curve["energy_kwh"], dtype=float)
    count = min(len(distance), len(speed), len(energy))
    mask = np.isfinite(distance[:count]) & np.isfinite(speed[:count]) & np.isfinite(energy[:count])
    distance = distance[:count][mask]
    speed = speed[:count][mask]
    energy = energy[:count][mask]
    if len(distance) < 2:
        raise ValueError("OpenTrack curve has fewer than two finite points.")
    distance = np.maximum.accumulate(distance - distance[0])
    energy = energy - energy[0]
    distance[0] = 0.0
    energy[0] = 0.0
    return {"distance_km": distance, "speed_kmh": speed, "energy_kwh": energy}


def load_cached_points(
    state_dir: Path,
    stats: pd.DataFrame,
    trip: int,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section_stats = stats.loc[
        (stats["trip_no"] == trip) & (stats["scenario"] == scenario)
    ].sort_values("section_index")
    if len(section_stats) != 26:
        raise ValueError(
            f"Trip {trip:03d} / {scenario}: expected 26 cached sections, "
            f"found {len(section_stats)}"
        )

    frames: list[pd.DataFrame] = []
    code_offset = 0.0
    simu_offset = 0.0
    replay_dir = state_dir / "replay_data"
    for index in range(26):
        path = replay_dir / f"trip{trip:03d}_{scenario}_{index:02d}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path).copy()
        frame["code_cumulative_kwh"] = code_offset + frame["code_section_kwh"]
        frame["simu_cumulative_kwh"] = simu_offset + frame["simu_section_kwh"]
        row = section_stats.iloc[index]
        code_offset += float(row["code_kwh"])
        simu_offset += float(row["simu_kwh"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), section_stats


def draw_trip_scenario(
    trip: int,
    scenario: str,
    scenario_spec: dict[str, Any],
    points: pd.DataFrame,
    section_stats: pd.DataFrame,
    ot: dict[str, np.ndarray],
    output_stem: Path,
    formats: list[str],
    dpi: int,
) -> dict[str, Any]:
    code_total = float(section_stats["code_kwh"].sum())
    simu_total = float(section_stats["simu_kwh"].sum())
    ot_total = float(ot["energy_kwh"][-1])
    max_mass = float(section_stats["mass_t"].max())
    runtime = float(section_stats["history_runtime_s"].sum())
    dwell = float(section_stats["history_dwell_s"].sum())
    unresolved = int((section_stats["code_version"] == "unresolved").sum())

    fig, (ax_energy, ax_speed) = plt.subplots(1, 2, figsize=(16, 6.4))
    fig.subplots_adjust(left=0.062, right=0.987, top=0.80, bottom=0.16, wspace=0.20)

    code_label = "历史实测能耗" if scenario == "history" else "代码规划能耗"
    ax_energy.plot(
        points["distance_km"],
        points["code_cumulative_kwh"],
        color="#303030",
        linewidth=1.55,
        label=code_label,
    )
    ax_energy.plot(
        points["distance_km"],
        points["simu_cumulative_kwh"],
        color="#D1495B",
        linestyle="--",
        linewidth=1.65,
        label="新版 Simu（机械能）",
    )
    ax_energy.plot(
        ot["distance_km"],
        ot["energy_kwh"],
        color="#1676B8",
        linewidth=1.65,
        label="OpenTrack（机械能）",
    )
    if unresolved:
        boundary_x = np.r_[
            0.0,
            points.groupby("section_index", sort=True)["distance_km"].max().to_numpy(float),
        ]
        boundary_y = np.r_[0.0, np.cumsum(section_stats["code_kwh"].to_numpy(float))]
        ax_energy.scatter(boundary_x, boundary_y, color="#303030", s=11, zorder=4)

    ax_speed.plot(
        points["distance_km"],
        points["speed_kmh"],
        color="#303030",
        linewidth=1.45,
        label="代码/历史速度",
    )
    ax_speed.plot(
        points["distance_km"],
        points["speed_kmh"],
        color="#D1495B",
        linestyle="--",
        linewidth=1.15,
        label="Simu 输入速度",
    )
    ax_speed.plot(
        ot["distance_km"],
        ot["speed_kmh"],
        color="#1676B8",
        linewidth=1.5,
        label="OpenTrack 仿真速度",
    )

    for ax, ylabel in ((ax_energy, "累计能耗 (kWh)"), (ax_speed, "速度 (km/h)")):
        ax.set_xlabel("里程 (km)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, max(FINAL_DISTANCE_KM, float(ot["distance_km"][-1])) * 1.002)
        ax.grid(True, color="#CBD5E1", alpha=0.30, linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_energy.set_ylim(bottom=min(0.0, float(np.nanmin(points["code_cumulative_kwh"]))))
    ax_speed.set_ylim(bottom=0.0)
    ax_energy.set_title("a  能耗-里程", loc="left", fontsize=12)
    ax_speed.set_title("b  速度-里程", loc="left", fontsize=12)
    ax_energy.legend(loc="lower right", frameon=True, framealpha=0.92)
    ax_speed.legend(loc="upper right", bbox_to_anchor=(1.0, 1.10), ncol=2, frameon=False)

    errors = "\n".join(
        [
            f"代码: {code_total:.2f} kWh",
            f"Simu: {simu_total:.2f} kWh",
            f"OT: {ot_total:.2f} kWh",
            "",
            f"Simu-代码: {simu_total-code_total:+.2f} kWh ({percent(simu_total-code_total, code_total):+.2f}%)",
            f"OT-代码: {ot_total-code_total:+.2f} kWh ({percent(ot_total-code_total, code_total):+.2f}%)",
            f"OT-Simu: {ot_total-simu_total:+.2f} kWh ({percent(ot_total-simu_total, simu_total):+.2f}%)",
        ]
    )
    ax_energy.text(
        0.018,
        0.975,
        errors,
        transform=ax_energy.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "white",
            "edgecolor": "#A8B2BD",
            "alpha": 0.94,
        },
    )

    fig.suptitle(
        f"Trip {trip:03d} | {scenario_spec['label']} | 最大区间列车质量 {max_mass:.2f} t",
        y=0.966,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.89,
        f"历史目标（运行 / 停站）：{runtime:.1f} / {dwell:.1f} s    |    "
        f"Simu-代码：{simu_total-code_total:+.2f} kWh / {percent(simu_total-code_total, code_total):+.2f}%    |    "
        f"OT-代码：{ot_total-code_total:+.2f} kWh / {percent(ot_total-code_total, code_total):+.2f}%",
        ha="center",
        fontsize=10.5,
    )
    note = (
        "误差百分比：差值/参考值×100%；Simu 与 OpenTrack 均为正向牵引机械能。"
        "速度曲线按统一站界显示，物理计算仍使用原始时间与速度。"
    )
    if unresolved:
        note += f" 代码能耗有 {unresolved} 个区间仅保留已知站界累计值。"
    fig.text(0.062, 0.052, note, fontsize=9.2, color="#4B5563")
    fig.text(
        0.062,
        0.024,
        "历史场景的代码曲线为实测能耗；规划场景的代码曲线保留原 DP 能耗及已有残差修正。",
        fontsize=9.0,
        color="#4B5563",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight", "facecolor": "white"}
    if "svg" in formats:
        fig.savefig(output_stem.with_suffix(".svg"), **save_options)
    if "pdf" in formats:
        fig.savefig(output_stem.with_suffix(".pdf"), **save_options)
    if "png" in formats:
        fig.savefig(output_stem.with_suffix(".png"), dpi=max(dpi, 300), **save_options)
    if "jpg" in formats or "jpeg" in formats:
        fig.savefig(output_stem.with_suffix(".jpg"), dpi=max(dpi, 300), **save_options)
    if "tif" in formats or "tiff" in formats:
        suffix = ".tiff" if "tiff" in formats else ".tif"
        fig.savefig(output_stem.with_suffix(suffix), dpi=600, **save_options)
    plt.close(fig)

    return {
        "趟次": trip,
        "场景": scenario,
        "场景名称": scenario_spec["label"],
        "区间数": 26,
        "最大载重(t)": max_mass,
        "历史运行时间(s)": runtime,
        "历史停站时间(s)": dwell,
        "代码或历史能耗(kWh)": code_total,
        "Simu机械能耗(kWh)": simu_total,
        "OpenTrack机械能耗(kWh)": ot_total,
        "Simu-代码(kWh)": simu_total - code_total,
        "Simu相对代码误差(%)": percent(simu_total - code_total, code_total),
        "OT-代码(kWh)": ot_total - code_total,
        "OT相对代码误差(%)": percent(ot_total - code_total, code_total),
        "OT-Simu(kWh)": ot_total - simu_total,
        "OT相对Simu误差(%)": percent(ot_total - simu_total, simu_total),
        "代码未逐点复现区间数": unresolved,
        "图文件": str(output_stem.with_suffix(".png")),
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


def saving_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_trip: list[dict[str, Any]] = []
    for trip in sorted(summary["趟次"].unique()):
        rows = summary.loc[summary["趟次"] == trip].set_index("场景")
        for scenario in ("standard_dp", "energy_first", "dwell5_dp"):
            row: dict[str, Any] = {
                "趟次": int(trip),
                "规划方式": SCENARIOS[scenario]["label"],
            }
            for source, column in (
                ("代码", "代码或历史能耗(kWh)"),
                ("新版Simu", "Simu机械能耗(kWh)"),
                ("OpenTrack", "OpenTrack机械能耗(kWh)"),
            ):
                historical = float(rows.loc["history", column])
                planned = float(rows.loc[scenario, column])
                saving = historical - planned
                row[f"{source}节能量(kWh)"] = saving
                row[f"{source}节能率(%)"] = percent(saving, historical)
            by_trip.append(row)
    detail = pd.DataFrame(by_trip)

    average_rows: list[dict[str, Any]] = []
    for scenario in ("standard_dp", "energy_first", "dwell5_dp"):
        group = detail.loc[detail["规划方式"] == SCENARIOS[scenario]["label"]]
        average_rows.append(
            {
                "规划方式": SCENARIOS[scenario]["label"],
                "样本趟次数": len(group),
                "代码预计节能量平均值(kWh)": group["代码节能量(kWh)"].mean(),
                "代码预计节能率平均值(%)": group["代码节能率(%)"].mean(),
                "新版Simu节能量平均值(kWh)": group["新版Simu节能量(kWh)"].mean(),
                "新版Simu节能率平均值(%)": group["新版Simu节能率(%)"].mean(),
                "OpenTrack节能量平均值(kWh)": group["OpenTrack节能量(kWh)"].mean(),
                "OpenTrack节能率平均值(%)": group["OpenTrack节能率(%)"].mean(),
            }
        )
    return detail, pd.DataFrame(average_rows)


def main() -> int:
    args = parse_args()
    trips = parse_trips(args.trips)
    formats = parse_formats(args.formats)
    configure_matplotlib()

    stats_path = args.state_dir / "section_comparison_summary.csv"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats = pd.read_csv(stats_path)
    missing_trips = sorted(set(trips) - set(stats["trip_no"].astype(int)))
    if missing_trips:
        raise ValueError(f"Replay cache is missing trips: {missing_trips}")

    expected_tsvp = [
        args.tsvp_dir / spec["ot_template"].format(trip=trip)
        for trip in trips
        for spec in SCENARIOS.values()
    ]
    missing_tsvp = [path for path in expected_tsvp if not path.is_file()]
    if missing_tsvp:
        preview = "\n".join(f"- {path}" for path in missing_tsvp[:20])
        raise FileNotFoundError(f"Missing {len(missing_tsvp)} OpenTrack TSVP files:\n{preview}")

    print(f"Trips: {trips}", flush=True)
    print(f"Replay state: {args.state_dir}", flush=True)
    print(f"OpenTrack TSVP: {args.tsvp_dir}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    rows: list[dict[str, Any]] = []
    for trip in trips:
        print(f"\nTrip {trip:03d}", flush=True)
        for scenario, spec in SCENARIOS.items():
            points, section_stats = load_cached_points(args.state_dir, stats, trip, scenario)
            ot_path = args.tsvp_dir / spec["ot_template"].format(trip=trip)
            ot = finite_curve(read_tsvp(ot_path))
            output_stem = args.output_dir / f"trip{trip:03d}" / spec["stem"]
            print(f"  {scenario}: rendering", flush=True)
            row = draw_trip_scenario(
                trip,
                scenario,
                spec,
                points,
                section_stats,
                ot,
                output_stem,
                formats,
                args.dpi,
            )
            rows.append(row)
            print(
                f"    code={row['代码或历史能耗(kWh)']:.2f}, "
                f"Simu={row['Simu机械能耗(kWh)']:.2f}, "
                f"OT={row['OpenTrack机械能耗(kWh)']:.2f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        args.output_dir / "code_simu_opentrack_trip_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    detail, average = saving_tables(summary)
    detail.to_csv(
        args.output_dir / "planning_energy_saving_by_trip.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    average.to_csv(
        args.output_dir / "planning_energy_saving_average_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    qa = {
        "backend": "python",
        "archetype": "quantitative grid",
        "claim": "Compare matched per-trip code, Simu and OpenTrack cumulative energy and speed results.",
        "trip_count": len(trips),
        "scenario_count": len(SCENARIOS),
        "figure_count": len(trips) * len(SCENARIOS),
        "source_mapping": {
            "code_and_simu": str(args.state_dir / "section_comparison_summary.csv"),
            "opentrack": str(args.tsvp_dir),
        },
        "statistics": "Deterministic per-trip comparisons; 25-trip summary is the arithmetic mean of paired trip savings and percentages.",
        "exclusions": "None among the requested 25 trips.",
        "trip53": "Uses the final named TSVP files; duplicate OTD listener events are not accumulated.",
    }
    (args.output_dir / "qa_notes.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nFinished: {len(rows)} figures", flush=True)
    print(
        f"Summary: {args.output_dir / 'code_simu_opentrack_trip_scenario_summary.csv'}",
        flush=True,
    )
    print(
        f"Savings: {args.output_dir / 'planning_energy_saving_average_summary.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
