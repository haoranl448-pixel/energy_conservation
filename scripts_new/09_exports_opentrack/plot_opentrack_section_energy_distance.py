from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from plot_opentrack_section_energy_bars import (
    DEFAULT_HISTORY_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLAN_ROOT,
    DEFAULT_TSVP_ROOT,
    read_section_inputs,
    read_tsvp,
    stop_boundaries,
)
from plot_opentrack_four_source_compare import (
    DEFAULT_HISTORY_CACHE_DIR,
    DEFAULT_TRACEABILITY_MANIFEST,
    history_cache_paths,
)
from plot_opentrack_three_way_compare import read_trip_segment_map
from plot_opentrack_tsvp_vs_history import read_history_section


DEFAULT_TRIPS = "1,3,4,5,6,8,10,12,14,15,42,62,67"
DEFAULT_TRACEABILITY_DATA_DIR = (
    Path(r"D:\energy_conservation")
    / "data"
    / "data_processed_step2_v3_all_curve_quality_traceability"
)

SOURCES = (
    {
        "key": "history",
        "label": "OT 历史",
        "template": "OT_priority_history_trip{trip:03d}.tsvP",
        "color": "#202124",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    {
        "key": "standard_dp",
        "label": "OT 标准 DP",
        "template": "OT_priority_dp_trip{trip:03d}.tsvP",
        "color": "#2B6CB0",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "key": "dwell5_dp",
        "label": "OT 停站宽松 5% DP",
        "template": "OT_real_priority_dwell_5pct_dp_trip{trip:03d}.tsvP",
        "color": "#2A8C6A",
        "linestyle": "-.",
        "linewidth": 1.9,
    },
    {
        "key": "energy_first",
        "label": "OT Energy First DP",
        "template": "OT_energy_first_dp_trip{trip:03d}.tsvP",
        "color": "#D62728",
        "linestyle": "-",
        "linewidth": 1.9,
    },
)

PAIRWISE_COMPARISONS = (
    {
        "key": "01_standard_dp_vs_ot_history",
        "source_key": "standard_dp",
        "title": "标准 DP vs OpenTrack 历史",
    },
    {
        "key": "02_dwell5_dp_vs_ot_history",
        "source_key": "dwell5_dp",
        "title": "停站宽松 5% DP vs OpenTrack 历史",
    },
    {
        "key": "03_energy_first_vs_ot_history",
        "source_key": "energy_first",
        "title": "Energy First DP vs OpenTrack 历史",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot section-level cumulative mechanical energy against local distance "
            "for OpenTrack history and three planning methods."
        )
    )
    parser.add_argument("--trips", default=DEFAULT_TRIPS)
    parser.add_argument("--tsvp-dir", default=str(DEFAULT_TSVP_ROOT))
    parser.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    parser.add_argument("--plan-root", default=str(DEFAULT_PLAN_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--data-dir", default=str(DEFAULT_TRACEABILITY_DATA_DIR))
    parser.add_argument(
        "--traceability-manifest",
        default=str(DEFAULT_TRACEABILITY_MANIFEST),
    )
    parser.add_argument(
        "--history-cache-dir",
        default=str(DEFAULT_HISTORY_CACHE_DIR),
    )
    parser.add_argument("--ignore-history-cache", action="store_true")
    parser.add_argument("--quality-label", default=None)
    parser.add_argument("--zero-speed-eps", type=float, default=0.05)
    parser.add_argument("--distance-eps", type=float, default=1e-5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument(
        "--include-four-source",
        action="store_true",
        help="Also write one four-source overview for every section.",
    )
    parser.add_argument(
        "--only-history-real",
        action="store_true",
        help="Only write OpenTrack-history vs real-history section plots.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def parse_trips(text: str) -> list[int]:
    trips: list[int] = []
    for raw_value in text.split(","):
        value = raw_value.strip()
        if not value:
            continue
        trip = int(value)
        if trip < 1:
            raise ValueError(f"Trip number must be positive: {trip}")
        if trip not in trips:
            trips.append(trip)
    if not trips:
        raise ValueError("No trip numbers supplied.")
    return trips


def required_tsvp(tsvp_dir: Path, template: str, trip_no: int) -> Path:
    path = tsvp_dir / template.format(trip=trip_no)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_section_curves(
    path: Path,
    expected_sections: int,
    zero_speed_eps: float,
    distance_eps: float,
) -> list[dict[str, np.ndarray | float]]:
    curve = read_tsvp(path)
    distance_km = np.asarray(curve["distance_km"], dtype=float)
    energy_kwh = np.asarray(curve["energy_kwh"], dtype=float)
    boundaries = stop_boundaries(curve, zero_speed_eps, distance_eps)
    expected_boundaries = expected_sections + 1
    if len(boundaries) != expected_boundaries:
        raise ValueError(
            f"{path.name}: found {len(boundaries)} zero-speed station boundaries; "
            f"expected {expected_boundaries}."
        )

    section_curves: list[dict[str, np.ndarray | float]] = []
    for section_index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:]), start=1
    ):
        start_distance, start_energy = start
        end_distance, end_energy = end
        mask = (
            (distance_km >= start_distance - distance_eps)
            & (distance_km <= end_distance + distance_eps)
        )
        local_distance = distance_km[mask] - start_distance
        local_energy = energy_kwh[mask] - start_energy
        if local_distance.size == 0:
            raise ValueError(f"{path.name}: section {section_index} has no samples.")

        local_distance = np.clip(local_distance, 0.0, end_distance - start_distance)
        local_distance = np.maximum.accumulate(local_distance)
        local_energy = np.maximum(local_energy, 0.0)
        if local_distance[0] > distance_eps or abs(local_energy[0]) > 1e-9:
            local_distance = np.insert(local_distance, 0, 0.0)
            local_energy = np.insert(local_energy, 0, 0.0)
        else:
            local_distance[0] = 0.0
            local_energy[0] = 0.0

        section_curves.append(
            {
                "distance_km": local_distance,
                "energy_kwh": local_energy,
                "total_energy_kwh": float(end_energy - start_energy),
                "distance_total_km": float(end_distance - start_distance),
            }
        )
    return section_curves


def load_real_history_section_curves(
    trip_no: int,
    sections: list[dict[str, float | str]],
    data_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
    ignore_history_cache: bool,
    quality_label: str | None,
) -> tuple[list[dict[str, np.ndarray | float]], int]:
    section_names = [str(section["section"]) for section in sections]
    segment_by_section = read_trip_segment_map(
        manifest_path,
        trip_no,
        section_names,
    )
    cache_by_section = (
        history_cache_paths(
            manifest_path,
            data_dir,
            cache_dir,
            trip_no,
            section_names,
        )
        if not ignore_history_cache
        else {}
    )
    history_args = SimpleNamespace(
        data_dir=str(data_dir),
        trip_no=trip_no,
        run_id=None,
        quality_label=quality_label,
        history_dwell=0.0,
        segment_by_section=segment_by_section,
        history_section_cache=cache_by_section,
    )

    result: list[dict[str, np.ndarray | float]] = []
    for section_name in section_names:
        history_curve = read_history_section(
            data_dir / f"results_{section_name}.xlsx",
            section_name,
            history_args,
        )
        distance_km = np.asarray(history_curve["distance_km"], dtype=float)
        energy_kwh = np.asarray(history_curve["energy_kwh"], dtype=float)
        if distance_km.size == 0 or energy_kwh.size == 0:
            raise ValueError(
                f"Trip {trip_no:03d} {section_name}: empty real-history curve."
            )
        distance_km = np.maximum.accumulate(distance_km - distance_km[0])
        energy_kwh = energy_kwh - energy_kwh[0]
        distance_km[0] = 0.0
        energy_kwh[0] = 0.0
        result.append(
            {
                "distance_km": distance_km,
                "energy_kwh": energy_kwh,
                "total_energy_kwh": float(energy_kwh[-1]),
                "distance_total_km": float(distance_km[-1]),
            }
        )
    return result, len(cache_by_section)


def plot_section(
    trip_no: int,
    section_index: int,
    section: dict[str, float | str],
    source_curves: dict[str, list[dict[str, np.ndarray | float]]],
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

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for source in SOURCES:
        section_curve = source_curves[str(source["key"])][section_index]
        energy_total = float(section_curve["total_energy_kwh"])
        ax.plot(
            section_curve["distance_km"],
            section_curve["energy_kwh"],
            color=str(source["color"]),
            linestyle=str(source["linestyle"]),
            linewidth=float(source["linewidth"]),
            label=f"{source['label']}  {energy_total:.2f} kWh",
        )

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("区间里程 (km)", fontsize=11)
    ax.set_ylabel("累计机械能 (kWh)", fontsize=11)
    ax.grid(color="#D9DDE3", linewidth=0.8, alpha=0.85)
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.set_title(
        f"Trip {trip_no:03d}  {section['section']}  区间能耗-里程对比\n"
        f"区间载重 {float(section['mass_t']):.2f} t",
        fontsize=13,
        pad=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi, "facecolor": "white", "bbox_inches": "tight"}
    if output_path.suffix.lower() == ".jpg":
        save_kwargs["pil_kwargs"] = {"quality": 95}
    fig.savefig(output_path, **save_kwargs)
    if show:
        plt.show()
    plt.close(fig)


def plot_pairwise_section(
    trip_no: int,
    section_index: int,
    section: dict[str, float | str],
    comparison: dict[str, str],
    source_curves: dict[str, list[dict[str, np.ndarray | float]]],
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    """Plot one planning method against OpenTrack history for one section."""

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    history_source = next(source for source in SOURCES if source["key"] == "history")
    planned_source = next(
        source for source in SOURCES if source["key"] == comparison["source_key"]
    )
    history_curve = source_curves["history"][section_index]
    planned_curve = source_curves[comparison["source_key"]][section_index]
    history_total = float(history_curve["total_energy_kwh"])
    planned_total = float(planned_curve["total_energy_kwh"])
    saving_kwh = history_total - planned_total
    saving_pct = (
        saving_kwh / history_total * 100.0 if abs(history_total) > 1e-12 else float("nan")
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.plot(
        history_curve["distance_km"],
        history_curve["energy_kwh"],
        color=str(history_source["color"]),
        linestyle=str(history_source["linestyle"]),
        linewidth=float(history_source["linewidth"]),
        label=f"{history_source['label']}  {history_total:.2f} kWh",
    )
    ax.plot(
        planned_curve["distance_km"],
        planned_curve["energy_kwh"],
        color=str(planned_source["color"]),
        linestyle=str(planned_source["linestyle"]),
        linewidth=float(planned_source["linewidth"]),
        label=f"{planned_source['label']}  {planned_total:.2f} kWh",
    )

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("区间里程 (km)", fontsize=11)
    ax.set_ylabel("累计机械能 (kWh)", fontsize=11)
    ax.grid(color="#D9DDE3", linewidth=0.8, alpha=0.85)
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
    result_text = (
        f"节能 {saving_kwh:.2f} kWh ({saving_pct:.2f}%)"
        if saving_kwh >= 0
        else f"能耗增加 {-saving_kwh:.2f} kWh ({-saving_pct:.2f}%)"
    )
    ax.set_title(
        f"Trip {trip_no:03d}  {section['section']}  {comparison['title']}\n"
        f"区间载重 {float(section['mass_t']):.2f} t  |  {result_text}",
        fontsize=13,
        pad=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi, "facecolor": "white", "bbox_inches": "tight"}
    if output_path.suffix.lower() == ".jpg":
        save_kwargs["pil_kwargs"] = {"quality": 95}
    fig.savefig(output_path, **save_kwargs)
    if show:
        plt.show()
    plt.close(fig)


def plot_ot_history_vs_real_history(
    trip_no: int,
    section_index: int,
    section: dict[str, float | str],
    source_curves: dict[str, list[dict[str, np.ndarray | float]]],
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    """Plot OpenTrack historical simulation against the traced real history."""

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    ot_curve = source_curves["history"][section_index]
    real_curve = source_curves["real_history"][section_index]
    ot_total = float(ot_curve["total_energy_kwh"])
    real_total = float(real_curve["total_energy_kwh"])
    difference_kwh = ot_total - real_total
    difference_pct = (
        difference_kwh / real_total * 100.0
        if abs(real_total) > 1e-12
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.plot(
        real_curve["distance_km"],
        real_curve["energy_kwh"],
        color="#202124",
        linestyle="--",
        linewidth=2.0,
        label=f"真实历史  {real_total:.2f} kWh",
    )
    ax.plot(
        ot_curve["distance_km"],
        ot_curve["energy_kwh"],
        color="#7B4FA3",
        linestyle="-",
        linewidth=2.0,
        label=f"OT 历史  {ot_total:.2f} kWh",
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("区间里程 (km)", fontsize=11)
    ax.set_ylabel("累计能耗 (kWh)", fontsize=11)
    ax.grid(color="#D9DDE3", linewidth=0.8, alpha=0.85)
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
    ax.set_title(
        f"Trip {trip_no:03d}  {section['section']}  OpenTrack 历史 vs 真实历史\n"
        f"区间载重 {float(section['mass_t']):.2f} t  |  "
        f"OT-真实 {difference_kwh:+.2f} kWh ({difference_pct:+.2f}%)",
        fontsize=13,
        pad=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi, "facecolor": "white", "bbox_inches": "tight"}
    if output_path.suffix.lower() == ".jpg":
        save_kwargs["pil_kwargs"] = {"quality": 95}
    fig.savefig(output_path, **save_kwargs)
    if show:
        plt.show()
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    trips = parse_trips(args.trips)
    tsvp_dir = Path(args.tsvp_dir)
    history_root = Path(args.history_root)
    plan_root = Path(args.plan_root)
    output_root = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    manifest_path = Path(args.traceability_manifest)
    cache_dir = Path(args.history_cache_dir)

    total_plots = 0
    for trip_no in trips:
        history_path = (
            history_root / f"trip{trip_no:03d}" / "Historical_Only_Report.csv"
        )
        plan_path = (
            plan_root / f"trip{trip_no:03d}" / "01_standard_priority.csv"
        )
        sections = read_section_inputs(history_path, plan_path)
        expected_sections = len(sections)
        if expected_sections != 26:
            raise ValueError(
                f"Trip {trip_no:03d}: expected 26 sections, found {expected_sections}."
            )

        source_curves: dict[str, list[dict[str, np.ndarray | float]]] = {}
        for source in SOURCES:
            path = required_tsvp(
                tsvp_dir,
                str(source["template"]),
                trip_no,
            )
            source_curves[str(source["key"])] = extract_section_curves(
                path,
                expected_sections,
                args.zero_speed_eps,
                args.distance_eps,
            )
        real_history_curves, real_history_cache_count = (
            load_real_history_section_curves(
                trip_no,
                sections,
                data_dir,
                manifest_path,
                cache_dir,
                args.ignore_history_cache,
                args.quality_label,
            )
        )
        source_curves["real_history"] = real_history_curves

        section_output_dir = (
            output_root / f"trip{trip_no:03d}" / "section_energy_distance"
        )
        summary_rows: list[dict[str, object]] = []
        for section_index, section in enumerate(sections):
            pairwise_paths: dict[str, Path] = {}
            comparisons_to_plot = (
                () if args.only_history_real else PAIRWISE_COMPARISONS
            )
            for comparison in comparisons_to_plot:
                output_path = section_output_dir / comparison["key"] / (
                    f"{section_index + 1:02d}_{section['section']}_energy_distance."
                    f"{args.image_format}"
                )
                plot_pairwise_section(
                    trip_no,
                    section_index,
                    section,
                    comparison,
                    source_curves,
                    output_path,
                    args.dpi,
                    args.show,
                )
                pairwise_paths[comparison["key"]] = output_path
                total_plots += 1

            history_real_key = "04_ot_history_vs_real_history"
            history_real_path = section_output_dir / history_real_key / (
                f"{section_index + 1:02d}_{section['section']}_energy_distance."
                f"{args.image_format}"
            )
            plot_ot_history_vs_real_history(
                trip_no,
                section_index,
                section,
                source_curves,
                history_real_path,
                args.dpi,
                args.show,
            )
            total_plots += 1

            four_source_path: Path | None = None
            if args.include_four_source and not args.only_history_real:
                four_source_path = section_output_dir / "04_four_source_overview" / (
                    f"{section_index + 1:02d}_{section['section']}_energy_distance."
                    f"{args.image_format}"
                )
                plot_section(
                    trip_no,
                    section_index,
                    section,
                    source_curves,
                    four_source_path,
                    args.dpi,
                    args.show,
                )
                total_plots += 1

            row: dict[str, object] = {
                "trip_no": trip_no,
                "section_index": section_index + 1,
                "站间区间": section["section"],
                "区间载重(t)": round(float(section["mass_t"]), 6),
                "全局趟次ID": section["global_trip_id"],
                "历史segment": section["history_segment"],
                "历史来源run_id": section["history_run_id"],
            }
            for source in SOURCES:
                curve = source_curves[str(source["key"])][section_index]
                row[f"{source['label']}区间里程(km)"] = round(
                    float(curve["distance_total_km"]), 6
                )
                row[f"{source['label']}机械能(kWh)"] = round(
                    float(curve["total_energy_kwh"]), 8
                )
            real_curve = source_curves["real_history"][section_index]
            row["真实历史区间里程(km)"] = round(
                float(real_curve["distance_total_km"]), 6
            )
            row["真实历史能耗(kWh)"] = round(
                float(real_curve["total_energy_kwh"]), 8
            )
            for comparison in PAIRWISE_COMPARISONS:
                comparison_path = pairwise_paths.get(
                    comparison["key"],
                    section_output_dir / comparison["key"] / (
                        f"{section_index + 1:02d}_{section['section']}_energy_distance."
                        f"{args.image_format}"
                    ),
                )
                row[f"{comparison['title']}图像路径"] = str(
                    comparison_path
                )
            row["四源总览图像路径"] = (
                "" if four_source_path is None else str(four_source_path)
            )
            row["OpenTrack历史 vs 真实历史图像路径"] = str(history_real_path)
            summary_rows.append(row)

        summary_path = section_output_dir / (
            f"trip{trip_no:03d}_section_energy_distance_summary.csv"
        )
        write_summary(summary_path, summary_rows)
        print(
            f"Trip {trip_no:03d}: {expected_sections} sections, "
            f"real-history cache {real_history_cache_count}/{expected_sections} -> "
            f"{section_output_dir}",
            flush=True,
        )

    print(f"Done: {len(trips)} trips, {total_plots} section plots.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
