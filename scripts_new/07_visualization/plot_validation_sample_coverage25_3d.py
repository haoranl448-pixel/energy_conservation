# -*- coding: utf-8 -*-
"""Render the 25 validation trips as one monochrome 3D scatterplot."""
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
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    ax = fig.add_axes([0.09, 0.07, 0.84, 0.78], projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=-58)
    ax.set_box_aspect((1.55, 1.35, 1.0), zoom=1.14)
    ax.scatter(samples.mass_t, samples.running_s, samples.dwell_s,
               s=92, marker="o", color="#333333", edgecolor="#111111",
               linewidth=0.6, depthshade=True)
    ax.set(xlim=(213, 252), ylim=(2938, 3080), zlim=(743, 797),
           xlabel="最大区间列车质量 (t)",
           ylabel="历史区间运行时间总和 (s)",
           zlabel="历史停站总时间 (s)")
    ax.set_xticks([215, 225, 235, 245, 250])
    ax.set_yticks([2950, 2980, 3010, 3040, 3070])
    ax.set_zticks([750, 760, 770, 780, 790])
    ax.tick_params(pad=3, labelsize=10)
    ax.xaxis.labelpad = 16
    ax.yaxis.labelpad = 18
    ax.zaxis.labelpad = 12
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#B6B6B6")
        axis._axinfo["grid"].update(color=(0.80, 0.80, 0.80, 1.0), linewidth=0.65)

    fig.text(0.08, 0.93, "25趟样本的载重、运行与停站时间分布",
             fontsize=19, weight="bold", color="#1C1C1C")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "validation_sample_coverage25_3d"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
