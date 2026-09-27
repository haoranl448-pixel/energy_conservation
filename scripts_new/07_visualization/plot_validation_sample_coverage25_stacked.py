# -*- coding: utf-8 -*-
"""Draw the 25 validation trips as two aligned monochrome scatterplots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/cache/simu_code_replay/batch_simu_code_trip_compare_complete25/section_comparison_summary.csv"
OUTPUT = ROOT / "output/analysis/validation_sample_coverage25"


def main() -> None:
    sections = pd.read_csv(SOURCE)
    history = sections.loc[sections.scenario.eq("history")]
    counts = history.groupby("trip_no").section_index.agg(["count", "nunique", "min", "max"])
    complete = ((counts["count"] == 26) & (counts["nunique"] == 26)
                & (counts["min"] == 1) & (counts["max"] == 26))
    if len(counts) != 25 or not complete.all():
        raise ValueError("Expected 25 trips with 26 complete sections each")
    samples = history.groupby("trip_no", as_index=False).agg(
        mass_t=("mass_t", "max"),
        running_s=("history_runtime_s", "sum"),
        dwell_s=("history_dwell_s", "sum"),
    )
    if not np.isfinite(samples[["mass_t", "running_s", "dwell_s"]]).all().all():
        raise ValueError("Non-finite validation sample values")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
    })
    fig, (running_ax, dwell_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(12.8, 7.2),
        gridspec_kw={"hspace": 0.22}, facecolor="white",
    )
    fig.subplots_adjust(left=0.12, right=0.95, top=0.83, bottom=0.13)
    marker = dict(s=78, marker="o", facecolor="#3B3B3B",
                  edgecolor="#111111", linewidth=0.7, alpha=0.88, zorder=3)
    running_ax.scatter(samples.mass_t, samples.running_s, **marker)
    dwell_ax.scatter(samples.mass_t, samples.dwell_s, **marker)

    running_ax.set(ylabel="历史区间运行时间总和 (s)", ylim=(2940, 3078))
    dwell_ax.set(xlabel="最大区间列车质量 (t)",
                 ylabel="历史停站总时间 (s)", ylim=(744, 795), xlim=(213.5, 251.5))
    running_ax.set_yticks([2950, 2980, 3010, 3040, 3070])
    dwell_ax.set_yticks([750, 760, 770, 780, 790])
    dwell_ax.set_xticks([215, 220, 225, 230, 235, 240, 245, 250])
    for ax in (running_ax, dwell_ax):
        ax.grid(color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=8)
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")

    fig.text(0.12, 0.925, "25趟样本的载重与运行、停站时间分布",
             fontsize=19, weight="bold", color="#1C1C1C")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "validation_sample_coverage25_stacked"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
