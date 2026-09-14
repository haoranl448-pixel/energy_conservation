from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "output" / "opentrack_four_source_compare_complete10"
)
DEFAULT_DETAIL_CSV = DEFAULT_OUTPUT_ROOT / "opentrack_section_energy_all_trips_detail.csv"
DEFAULT_TRIPS = "1,3,4,5,6,8,10,12,14,15,42,62,67"

COMPARISONS = (
    {
        "key": "standard_dp",
        "title": "标准 DP vs OpenTrack 历史",
        "planned_label": "OT 标准 DP",
        "output_index": "01",
    },
    {
        "key": "dwell5_dp",
        "title": "停站宽松 5% DP vs OpenTrack 历史",
        "planned_label": "OT 停站宽松 5% DP",
        "output_index": "02",
    },
    {
        "key": "energy_first",
        "title": "Energy First DP vs OpenTrack 历史",
        "planned_label": "OT Energy First DP",
        "output_index": "03",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot three OpenTrack planning/history comparisons across trips. "
            "Trip energy is normalized by all section-specific mass-distance values."
        )
    )
    parser.add_argument("--trips", default=DEFAULT_TRIPS)
    parser.add_argument("--detail-csv", default=str(DEFAULT_DETAIL_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def parse_trips(text: str) -> list[int]:
    trips: list[int] = []
    for value in text.split(","):
        value = value.strip()
        if not value:
            continue
        trip = int(value)
        if trip < 1:
            raise ValueError(f"Trip number must be positive: {trip}")
        if trip not in trips:
            trips.append(trip)
    if not trips:
        raise ValueError("No trip numbers supplied")
    return trips


def as_float(value: object) -> float:
    return float(str(value).strip())


def saving_pct(planned: float, historical: float) -> float:
    if abs(historical) < 1e-12:
        return float("nan")
    return (historical - planned) / historical * 100.0


def read_detail(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_trips(
    rows: list[dict[str, str]],
    trips: list[int],
) -> dict[str, list[dict[str, float | int | str]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    wanted_trips = set(trips)
    wanted_comparisons = {item["key"] for item in COMPARISONS}
    for row in rows:
        trip = int(as_float(row["trip_no"]))
        comparison = row["comparison"]
        if trip in wanted_trips and comparison in wanted_comparisons:
            grouped[(trip, comparison)].append(row)

    result: dict[str, list[dict[str, float | int | str]]] = {
        item["key"]: [] for item in COMPARISONS
    }
    for comparison in COMPARISONS:
        key = comparison["key"]
        for trip in trips:
            section_rows = grouped.get((trip, key), [])
            if len(section_rows) != 26:
                raise ValueError(
                    f"Trip {trip:03d} / {key}: expected 26 sections, "
                    f"found {len(section_rows)}"
                )

            planned_energy = sum(
                as_float(row["规划OT区间机械能(kWh)"]) for row in section_rows
            )
            historical_energy = sum(
                as_float(row["历史OT区间机械能(kWh)"]) for row in section_rows
            )
            distance_km = sum(
                as_float(row["归一化区间里程(km)"]) for row in section_rows
            )
            tonne_km = sum(
                as_float(row["载重里程(t·km)"]) for row in section_rows
            )
            masses = [as_float(row["区间载重(t)"]) for row in section_rows]
            weighted_mass = tonne_km / distance_km

            result[key].append(
                {
                    "trip_no": trip,
                    "comparison": key,
                    "sections": len(section_rows),
                    "planned_energy_kwh": planned_energy,
                    "history_energy_kwh": historical_energy,
                    "saving_kwh": historical_energy - planned_energy,
                    "saving_pct": saving_pct(planned_energy, historical_energy),
                    "distance_km": distance_km,
                    "tonne_km": tonne_km,
                    "distance_weighted_mass_t": weighted_mass,
                    "max_mass_t": max(masses),
                    "planned_unit_energy": planned_energy / tonne_km,
                    "history_unit_energy": historical_energy / tonne_km,
                }
            )
    return result


def add_difference_marker(
    ax: plt.Axes,
    left_x: float,
    right_x: float,
    planned: float,
    historical: float,
    pct: float,
    y_padding: float,
) -> None:
    center_x = (left_x + right_x) / 2.0
    top = max(planned, historical)
    cap_width = 0.055
    ax.plot(
        [left_x, center_x, center_x, right_x],
        [planned, planned, historical, historical],
        color="#555555",
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        zorder=4,
    )
    ax.plot(
        [center_x - cap_width, center_x + cap_width],
        [planned, planned],
        color="#555555",
        linewidth=0.8,
        zorder=4,
    )
    ax.plot(
        [center_x - cap_width, center_x + cap_width],
        [historical, historical],
        color="#555555",
        linewidth=0.8,
        zorder=4,
    )
    if np.isnan(pct):
        text = "--"
        color = "#555555"
    elif pct >= 0:
        text = f"节{pct:.2f}%"
        color = "#246B45"
    else:
        text = f"增{-pct:.2f}%"
        color = "#A33A2B"
    ax.text(
        center_x,
        top + y_padding,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
        rotation=90,
        clip_on=False,
    )


def plot_comparison(
    comparison: dict[str, str],
    records: list[dict[str, float | int | str]],
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

    labels = [f"Trip {int(row['trip_no']):03d}" for row in records]
    planned_values = [float(row["planned_unit_energy"]) for row in records]
    history_values = [float(row["history_unit_energy"]) for row in records]
    saving_values = [float(row["saving_pct"]) for row in records]
    max_masses = [float(row["max_mass_t"]) for row in records]
    planned_total = sum(float(row["planned_energy_kwh"]) for row in records)
    history_total = sum(float(row["history_energy_kwh"]) for row in records)
    total_saving = history_total - planned_total
    total_pct = saving_pct(planned_total, history_total)

    if total_saving >= 0:
        overall_text = f"13趟合计节能 {total_saving:.2f} kWh（{total_pct:.2f}%）"
    else:
        overall_text = (
            f"13趟合计能耗增加 {-total_saving:.2f} kWh"
            f"（{-total_pct:.2f}%）"
        )

    x = np.arange(len(records), dtype=float)
    width = 0.34
    planned_x = x - width / 2.0
    history_x = x + width / 2.0
    max_value = max(max(planned_values), max(history_values))
    y_padding = max_value * 0.012

    fig, ax = plt.subplots(figsize=(17.5, 8.2), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(
        planned_x,
        planned_values,
        width,
        label=f"{comparison['planned_label']}（左柱）",
        color="#4C86B7",
        edgecolor="#2E5878",
        linewidth=0.75,
        zorder=3,
    )
    ax.bar(
        history_x,
        history_values,
        width,
        label="OT 历史（右柱）",
        color="#D7D7D7",
        edgecolor="#666666",
        linewidth=0.75,
        hatch="///",
        zorder=3,
    )

    ax_mass = ax.twinx()
    ax_mass.plot(
        x,
        max_masses,
        color="#C46A2D",
        marker="o",
        markersize=4.2,
        linewidth=1.7,
        label="每趟最大载重（右轴）",
        zorder=5,
    )
    mass_min = min(max_masses)
    mass_max = max(max_masses)
    mass_padding = max(3.0, (mass_max - mass_min) * 0.16)
    ax_mass.set_ylim(mass_min - mass_padding, mass_max + mass_padding)
    ax_mass.set_ylabel("每趟最大载重  (t)", color="#9A5425")
    ax_mass.tick_params(axis="y", colors="#9A5425")
    ax_mass.spines["top"].set_visible(False)

    ax.set_ylim(0, max_value * 1.30)
    ax.set_xticks(x, labels, rotation=38, ha="right")
    ax.set_ylabel("整趟单位质量、单位距离机械能耗  [kWh/(t·km)]")
    ax.set_xlabel("趟次")
    ax.set_title(
        f"13趟 OpenTrack {comparison['title']}\n"
        f"{overall_text} | 每趟按26个区间载重与里程归一化",
        pad=16,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.015)

    for idx in range(len(records)):
        add_difference_marker(
            ax,
            planned_x[idx],
            history_x[idx],
            planned_values[idx],
            history_values[idx],
            saving_values[idx],
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


def write_summary(path: Path, records: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    trips = parse_trips(args.trips)
    rows = read_detail(Path(args.detail_csv))
    grouped = aggregate_trips(rows, trips)
    output_dir = Path(args.output_dir)

    all_records: list[dict[str, float | int | str]] = []
    for comparison in COMPARISONS:
        records = grouped[comparison["key"]]
        all_records.extend(records)
        output_path = (
            output_dir
            / (
                f"{comparison['output_index']}_{comparison['key']}_"
                "vs_ot_history_13_trips_energy_bars.png"
            )
        )
        plot_comparison(
            comparison=comparison,
            records=records,
            output_path=output_path,
            dpi=args.dpi,
            show=args.show,
        )
        print(f"Output: {output_path}")

    summary_path = output_dir / "opentrack_trip_normalized_energy_comparison.csv"
    write_summary(summary_path, all_records)
    print(f"Summary: {summary_path}")
    print(f"Trips: {len(trips)}, comparisons: {len(COMPARISONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
