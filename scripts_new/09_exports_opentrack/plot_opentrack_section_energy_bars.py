from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_TSVP_ROOT = Path(r"D:\OutPut")
DEFAULT_HISTORY_ROOT = (
    PROJECT_ROOT / "output" / "schedule" / "batch_trip_reports_trip1_125"
)
DEFAULT_PLAN_ROOT = (
    PROJECT_ROOT
    / "output"
    / "schedule"
    / "batch_trip_reports_complete13_dp_unified"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "output" / "opentrack_four_source_compare_complete10"
)

COMPARISONS = (
    {
        "key": "standard_dp",
        "title": "标准 DP vs OpenTrack 历史",
        "planned_label": "OT 标准 DP",
        "template": "OT_priority_dp_trip{trip:03d}.tsvP",
        "output_index": "10",
    },
    {
        "key": "dwell5_dp",
        "title": "停站宽松 5% DP vs OpenTrack 历史",
        "planned_label": "OT 停站宽松 5% DP",
        "template": "OT_real_priority_dwell_5pct_dp_trip{trip:03d}.tsvP",
        "output_index": "11",
    },
    {
        "key": "energy_first",
        "title": "Energy First DP vs OpenTrack 历史",
        "planned_label": "OT Energy First DP",
        "template": "OT_energy_first_dp_trip{trip:03d}.tsvP",
        "output_index": "12",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot three section-level OpenTrack planning/history comparisons. "
            "Each section is normalized by its own mass and distance."
        )
    )
    parser.add_argument("--trip-no", type=int, default=1)
    parser.add_argument("--tsvp-dir", default=str(DEFAULT_TSVP_ROOT))
    parser.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    parser.add_argument("--plan-root", default=str(DEFAULT_PLAN_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--history-template",
        default="OT_priority_history_trip{trip:03d}.tsvP",
    )
    parser.add_argument("--zero-speed-eps", type=float, default=0.05)
    parser.add_argument("--distance-eps", type=float, default=1e-5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def read_tsvp(path: Path) -> dict[str, list[float]]:
    distance_km: list[float] = []
    speed_kmh: list[float] = []
    energy_kwh: list[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("//") or not line.strip():
                continue
            parts = [part.strip() for part in line.rstrip("\n").split("\t")]
            parts = [part for part in parts if part]
            if len(parts) < 9:
                continue
            distance = as_float(parts[1])
            speed = as_float(parts[2])
            energy_mj = as_float(parts[8])
            if distance is None or speed is None or energy_mj is None:
                continue
            distance_km.append(distance)
            speed_kmh.append(speed)
            energy_kwh.append(energy_mj / 3.6)

    if not distance_km:
        raise ValueError(f"No numeric OpenTrack data found: {path}")

    distance0 = distance_km[0]
    energy0 = energy_kwh[0]
    return {
        "distance_km": [value - distance0 for value in distance_km],
        "speed_kmh": speed_kmh,
        "energy_kwh": [value - energy0 for value in energy_kwh],
    }


def stop_boundaries(
    curve: dict[str, list[float]],
    zero_speed_eps: float,
    distance_eps: float,
) -> list[tuple[float, float]]:
    boundaries: list[tuple[float, float]] = []
    for distance, speed, energy in zip(
        curve["distance_km"], curve["speed_kmh"], curve["energy_kwh"]
    ):
        if abs(speed) > zero_speed_eps:
            continue
        if boundaries and abs(distance - boundaries[-1][0]) <= distance_eps:
            boundaries[-1] = (distance, energy)
            continue
        boundaries.append((distance, energy))
    return boundaries


def section_energy_from_curve(
    path: Path,
    expected_sections: int,
    zero_speed_eps: float,
    distance_eps: float,
) -> tuple[list[float], list[float]]:
    curve = read_tsvp(path)
    boundaries = stop_boundaries(curve, zero_speed_eps, distance_eps)
    expected_boundaries = expected_sections + 1
    if len(boundaries) != expected_boundaries:
        raise ValueError(
            f"{path.name}: found {len(boundaries)} zero-speed station boundaries; "
            f"expected {expected_boundaries}. Review unexpected signal stops or "
            "adjust --zero-speed-eps."
        )

    section_energy: list[float] = []
    section_distance: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        section_distance.append(end[0] - start[0])
        section_energy.append(end[1] - start[1])
    return section_energy, section_distance


def read_section_inputs(
    history_path: Path,
    plan_path: Path,
) -> list[dict[str, float | str]]:
    with history_path.open("r", encoding="utf-8-sig", newline="") as f:
        history_rows = list(csv.DictReader(f))
    with plan_path.open("r", encoding="utf-8-sig", newline="") as f:
        plan_rows = list(csv.DictReader(f))

    distance_by_section: dict[str, float] = {}
    for row in history_rows:
        section = str(row.get("站间区间", "")).strip()
        distance_m = as_float(row.get("区间距离(m)"))
        if section and distance_m is not None and distance_m > 0:
            distance_by_section[section] = distance_m

    sections: list[dict[str, float | str]] = []
    for row in plan_rows:
        section = str(row.get("站间区间", "")).strip()
        mass_t = as_float(row.get("MASS"))
        distance_m = distance_by_section.get(section)
        if not section or mass_t is None or distance_m is None:
            continue
        if mass_t <= 0 or distance_m <= 0:
            continue
        sections.append(
            {
                "section": section,
                "mass_t": mass_t,
                "distance_km": distance_m / 1000.0,
                "tonne_km": mass_t * distance_m / 1000.0,
                "global_trip_id": str(row.get("全局趟次ID", "")).strip(),
                "history_segment": str(row.get("历史segment", "")).strip(),
                "history_run_id": str(row.get("历史来源run_id", "")).strip(),
            }
        )

    if not sections:
        raise ValueError(
            f"No valid traceable mass/distance rows found: {plan_path} / {history_path}"
        )
    return sections


def normalized_values(
    section_energy_kwh: list[float],
    sections: list[dict[str, float | str]],
) -> list[float]:
    if len(section_energy_kwh) != len(sections):
        raise ValueError(
            f"Energy section count {len(section_energy_kwh)} does not match "
            f"mass/distance section count {len(sections)}"
        )
    return [
        energy / float(section["tonne_km"])
        for energy, section in zip(section_energy_kwh, sections)
    ]


def saving_pct(planned: float, historical: float) -> float:
    if abs(historical) < 1e-12:
        return float("nan")
    return (historical - planned) / historical * 100.0


def add_difference_marker(
    ax: plt.Axes,
    left_x: float,
    right_x: float,
    planned: float,
    historical: float,
    y_padding: float,
) -> None:
    pct = saving_pct(planned, historical)
    top = max(planned, historical)
    center_x = (left_x + right_x) / 2.0
    cap_width = 0.055

    ax.plot(
        [left_x, center_x, center_x, right_x],
        [planned, planned, historical, historical],
        color="#555555",
        linewidth=0.75,
        linestyle=(0, (2, 2)),
        zorder=4,
    )
    ax.plot(
        [center_x - cap_width, center_x + cap_width],
        [planned, planned],
        color="#555555",
        linewidth=0.75,
        zorder=4,
    )
    ax.plot(
        [center_x - cap_width, center_x + cap_width],
        [historical, historical],
        color="#555555",
        linewidth=0.75,
        zorder=4,
    )

    if np.isnan(pct):
        label = "--"
        color = "#555555"
    elif pct >= 0:
        label = f"节{pct:.1f}%"
        color = "#246B45"
    else:
        label = f"增{-pct:.1f}%"
        color = "#A33A2B"
    ax.text(
        center_x,
        top + y_padding,
        label,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=color,
        rotation=90,
        clip_on=False,
    )


def plot_comparison(
    trip_no: int,
    comparison: dict[str, str],
    sections: list[dict[str, float | str]],
    planned_values: list[float],
    historical_values: list[float],
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    labels = [str(section["section"]) for section in sections]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    planned_x = x - width / 2.0
    history_x = x + width / 2.0
    max_value = max(max(planned_values), max(historical_values))
    y_padding = max_value * 0.012
    masses = [float(section["mass_t"]) for section in sections]
    planned_total_kwh = sum(
        value * float(section["tonne_km"])
        for value, section in zip(planned_values, sections)
    )
    historical_total_kwh = sum(
        value * float(section["tonne_km"])
        for value, section in zip(historical_values, sections)
    )
    overall_saving_kwh = historical_total_kwh - planned_total_kwh
    overall_saving_pct = saving_pct(planned_total_kwh, historical_total_kwh)
    if overall_saving_kwh >= 0:
        overall_text = (
            f"全程节能 {overall_saving_kwh:.2f} kWh"
            f"（{overall_saving_pct:.2f}%）"
        )
    else:
        overall_text = (
            f"全程能耗增加 {-overall_saving_kwh:.2f} kWh"
            f"（{-overall_saving_pct:.2f}%）"
        )

    fig, ax = plt.subplots(figsize=(24, 9), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(
        planned_x,
        planned_values,
        width,
        label=f"{comparison['planned_label']}（左柱）",
        color="#4C86B7",
        edgecolor="#2E5878",
        linewidth=0.7,
        zorder=3,
    )
    ax.bar(
        history_x,
        historical_values,
        width,
        label="OT 历史（右柱）",
        color="#D7D7D7",
        edgecolor="#666666",
        linewidth=0.7,
        hatch="///",
        zorder=3,
    )

    ax_mass = ax.twinx()
    ax_mass.plot(
        x,
        masses,
        color="#C46A2D",
        marker="o",
        markersize=3.8,
        linewidth=1.6,
        label="区间载重（右轴）",
        zorder=5,
    )
    mass_min = min(masses)
    mass_max = max(masses)
    mass_padding = max(2.0, (mass_max - mass_min) * 0.18)
    ax_mass.set_ylim(mass_min - mass_padding, mass_max + mass_padding)
    ax_mass.set_ylabel("区间载重  (t)", color="#9A5425")
    ax_mass.tick_params(axis="y", colors="#9A5425")
    ax_mass.spines["top"].set_visible(False)

    ax.set_ylim(0, max_value * 1.28)
    ax.set_xticks(x, labels, rotation=53, ha="right")
    ax.set_ylabel("区间单位质量、单位距离机械能耗  [kWh/(t·km)]")
    ax.set_xlabel("站间区间")
    ax.set_title(
        f"Trip {trip_no:03d}  {comparison['title']}\n"
        f"{overall_text} | 每个区间按该趟区间载重与区间里程归一化",
        pad=16,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

    for idx in range(len(labels)):
        add_difference_marker(
            ax,
            planned_x[idx],
            history_x[idx],
            planned_values[idx],
            historical_values[idx],
            y_padding,
        )

    bar_handles, bar_labels = ax.get_legend_handles_labels()
    mass_handles, mass_labels = ax_mass.get_legend_handles_labels()
    ax.legend(
        bar_handles + mass_handles,
        bar_labels + mass_labels,
        loc="upper left",
        frameon=False,
        ncol=3,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def write_detail_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.trip_no < 1:
        raise ValueError("--trip-no must be positive")

    tsvp_dir = Path(args.tsvp_dir)
    history_path = (
        Path(args.history_root)
        / f"trip{args.trip_no:03d}"
        / "Historical_Only_Report.csv"
    )
    plan_path = (
        Path(args.plan_root)
        / f"trip{args.trip_no:03d}"
        / "01_standard_priority.csv"
    )
    trip_output = Path(args.output_dir) / f"trip{args.trip_no:03d}"
    sections = read_section_inputs(history_path, plan_path)
    expected_sections = len(sections)

    history_tsvp = tsvp_dir / args.history_template.format(trip=args.trip_no)
    history_energy, history_ot_distance = section_energy_from_curve(
        history_tsvp,
        expected_sections,
        args.zero_speed_eps,
        args.distance_eps,
    )
    historical_values = normalized_values(history_energy, sections)

    detail_rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        planned_tsvp = tsvp_dir / comparison["template"].format(trip=args.trip_no)
        planned_energy, planned_ot_distance = section_energy_from_curve(
            planned_tsvp,
            expected_sections,
            args.zero_speed_eps,
            args.distance_eps,
        )
        planned_values = normalized_values(planned_energy, sections)

        output_path = (
            trip_output
            / (
                f"trip{args.trip_no:03d}_{comparison['output_index']}_"
                f"{comparison['key']}_vs_ot_history_section_energy_bars.png"
            )
        )
        plot_comparison(
            trip_no=args.trip_no,
            comparison=comparison,
            sections=sections,
            planned_values=planned_values,
            historical_values=historical_values,
            output_path=output_path,
            dpi=args.dpi,
            show=args.show,
        )

        for index, section in enumerate(sections):
            detail_rows.append(
                {
                    "trip_no": args.trip_no,
                    "comparison": comparison["key"],
                    "section_index": index + 1,
                    "站间区间": section["section"],
                    "区间载重(t)": round(float(section["mass_t"]), 6),
                    "载重来源": "traceable_standard_plan_MASS",
                    "全局趟次ID": section["global_trip_id"],
                    "历史segment": section["history_segment"],
                    "历史来源run_id": section["history_run_id"],
                    "归一化区间里程(km)": round(float(section["distance_km"]), 6),
                    "载重里程(t·km)": round(float(section["tonne_km"]), 6),
                    "规划OT区间里程(km)": round(planned_ot_distance[index], 6),
                    "历史OT区间里程(km)": round(history_ot_distance[index], 6),
                    "规划OT区间机械能(kWh)": round(planned_energy[index], 8),
                    "历史OT区间机械能(kWh)": round(history_energy[index], 8),
                    "规划单位能耗[kWh/(t·km)]": round(planned_values[index], 10),
                    "历史单位能耗[kWh/(t·km)]": round(historical_values[index], 10),
                    "区间节能率(%)": round(
                        saving_pct(planned_values[index], historical_values[index]),
                        6,
                    ),
                }
            )
        print(f"Output: {output_path}")

    detail_path = (
        trip_output
        / f"trip{args.trip_no:03d}_ot_section_normalized_energy_comparison.csv"
    )
    write_detail_csv(detail_path, detail_rows)
    print(f"Detail: {detail_path}")
    print(f"Sections: {expected_sections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
