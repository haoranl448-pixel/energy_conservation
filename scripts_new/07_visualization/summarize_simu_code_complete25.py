# -*- coding: utf-8 -*-
"""Create compact CSV summaries from the verified 25-trip replay cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\energy_conservation")
DEFAULT_STATS = (
    PROJECT_ROOT
    / "output"
    / "cache"
    / "simu_code_replay"
    / "batch_simu_code_trip_compare_complete25"
    / "section_comparison_summary.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "simu_code_trip_compare_complete25"
SCENARIO_ORDER = ("history", "standard_dp", "energy_first", "dwell5_dp")
SCENARIO_LABELS = {
    "history": "历史运行",
    "standard_dp": "标准 DP",
    "energy_first": "Energy First",
    "dwell5_dp": "停站放宽 5% DP",
}
FIGURE_STEMS = {
    "history": "00_history.png",
    "standard_dp": "01_standard_dp.png",
    "energy_first": "02_energy_first.png",
    "dwell5_dp": "03_dwell5_dp.png",
}


def percentage(delta: float, reference: float) -> float:
    return delta / reference * 100.0 if abs(reference) > 1e-12 else np.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    stats = pd.read_csv(args.stats, encoding="utf-8-sig")
    expected_columns = {
        "trip_no",
        "scenario",
        "section_index",
        "mass_t",
        "code_kwh",
        "simu_kwh",
        "code_version",
        "history_runtime_s",
        "history_dwell_s",
    }
    missing = expected_columns - set(stats.columns)
    if missing:
        raise ValueError(f"Replay summary is missing columns: {sorted(missing)}")

    trips = sorted(stats["trip_no"].astype(int).unique())
    if len(trips) != 25:
        raise ValueError(f"Expected 25 trips, found {len(trips)}: {trips}")

    scenario_rows: list[dict[str, object]] = []
    for trip in trips:
        for scenario in SCENARIO_ORDER:
            group = stats.loc[
                (stats["trip_no"] == trip) & (stats["scenario"] == scenario)
            ].sort_values("section_index")
            if len(group) != 26:
                raise ValueError(
                    f"Trip {trip:03d} / {scenario}: expected 26 sections, found {len(group)}"
                )
            code_total = float(group["code_kwh"].sum())
            simu_total = float(group["simu_kwh"].sum())
            difference = simu_total - code_total
            versions = ";".join(sorted(set(group["code_version"].astype(str))))
            scenario_rows.append(
                {
                    "趟次": trip,
                    "场景": scenario,
                    "场景名称": SCENARIO_LABELS[scenario],
                    "区间数": 26,
                    "最大区间列车质量(t)": float(group["mass_t"].max()),
                    "历史运行时间(s)": float(group["history_runtime_s"].sum()),
                    "历史停站时间(s)": float(group["history_dwell_s"].sum()),
                    "代码或历史能耗(kWh)": code_total,
                    "新版Simu机械能耗(kWh)": simu_total,
                    "Simu-代码(kWh)": difference,
                    "Simu相对代码误差(%)": percentage(difference, code_total),
                    "Simu与代码绝对差值(kWh)": abs(difference),
                    "Simu与代码绝对误差(%)": abs(percentage(difference, code_total)),
                    "代码数值复现版本": versions,
                    "图文件": str(
                        args.output_dir
                        / f"trip{trip:03d}"
                        / FIGURE_STEMS[scenario]
                    ),
                }
            )

    scenario_summary = pd.DataFrame(scenario_rows)
    saving_rows: list[dict[str, object]] = []
    for trip in trips:
        rows = scenario_summary.loc[scenario_summary["趟次"] == trip].set_index("场景")
        for scenario in SCENARIO_ORDER[1:]:
            code_history = float(rows.loc["history", "代码或历史能耗(kWh)"])
            code_planned = float(rows.loc[scenario, "代码或历史能耗(kWh)"])
            simu_history = float(rows.loc["history", "新版Simu机械能耗(kWh)"])
            simu_planned = float(rows.loc[scenario, "新版Simu机械能耗(kWh)"])
            code_saving = code_history - code_planned
            simu_saving = simu_history - simu_planned
            saving_rows.append(
                {
                    "趟次": trip,
                    "规划方式": SCENARIO_LABELS[scenario],
                    "代码预计节能量(kWh)": code_saving,
                    "代码预计节能率(%)": percentage(code_saving, code_history),
                    "新版Simu节能量(kWh)": simu_saving,
                    "新版Simu节能率(%)": percentage(simu_saving, simu_history),
                }
            )
    saving_detail = pd.DataFrame(saving_rows)

    average_rows: list[dict[str, object]] = []
    for scenario in SCENARIO_ORDER[1:]:
        label = SCENARIO_LABELS[scenario]
        group = saving_detail.loc[saving_detail["规划方式"] == label]
        average_rows.append(
            {
                "规划方式": label,
                "样本趟次数": len(group),
                "代码预计节能量平均值(kWh)": group["代码预计节能量(kWh)"].mean(),
                "代码预计节能率平均值(%)": group["代码预计节能率(%)"].mean(),
                "新版Simu节能量平均值(kWh)": group["新版Simu节能量(kWh)"].mean(),
                "新版Simu节能率平均值(%)": group["新版Simu节能率(%)"].mean(),
            }
        )
    average_summary = pd.DataFrame(average_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = args.output_dir / "simu_code_trip_scenario_summary.csv"
    detail_path = args.output_dir / "planning_energy_saving_by_trip.csv"
    average_path = args.output_dir / "planning_energy_saving_average_summary.csv"
    scenario_summary.to_csv(
        scenario_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    saving_detail.to_csv(
        detail_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    average_summary.to_csv(
        average_path, index=False, encoding="utf-8-sig", float_format="%.4f"
    )

    if not scenario_summary["图文件"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("One or more figure paths in the summary do not exist.")
    print(f"Scenario summary: {scenario_path} ({len(scenario_summary)} rows)")
    print(f"Saving detail:   {detail_path} ({len(saving_detail)} rows)")
    print(f"Average summary: {average_path} ({len(average_summary)} rows)")
    print(average_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
