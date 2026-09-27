# -*- coding: utf-8 -*-
"""Plot load and running time for the 25 validation trips."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/cache/simu_code_replay/batch_simu_code_trip_compare_complete25/section_comparison_summary.csv"
OUTPUT = ROOT / "output/analysis/validation_sample_coverage25"


def load_samples() -> pd.DataFrame:
    sections = pd.read_csv(SOURCE)
    history = sections.loc[sections.scenario.eq("history")].copy()
    counts = history.groupby("trip_no").section_index.agg(["count", "nunique", "min", "max"])
    if len(counts) != 25 or not ((counts["count"] == 26) & (counts["nunique"] == 26)
                                  & (counts["min"] == 1) & (counts["max"] == 26)).all():
        raise ValueError("Expected 25 complete trips, each with 26 unique sections")
    samples = history.groupby("trip_no", as_index=False).agg(
        mass_t=("mass_t", "max"),
        running_s=("history_runtime_s", "sum"),
    )
    if not np.isfinite(samples[["mass_t", "running_s"]]).all().all():
        raise ValueError("Non-finite sample values")
    return samples


def main() -> None:
    samples = load_samples()
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

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    ax = fig.add_axes([0.105, 0.145, 0.82, 0.72])
    ax.scatter(samples.mass_t, samples.running_s, s=90, marker="o",
               facecolor="#444444", edgecolor="#111111", linewidth=0.8,
               alpha=0.9, zorder=3)
    ax.set(xlim=(213.5, 252), ylim=(2938, 3083),
           xlabel="最大区间列车质量 (t)", ylabel="历史区间运行时间总和 (s)")
    ax.grid(color="#D7D7D7", linewidth=0.7, zorder=0)
    ax.tick_params(length=0, pad=8)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")

    zoom_bounds = (215.5, 220, 3018, 3040)
    x0, x1, y0, y1 = zoom_bounds
    zoomed = samples.mass_t.between(x0, x1) & samples.running_s.between(y0, y1)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor="#777777", linewidth=1.2, linestyle="--", zorder=4))
    inset = fig.add_axes([0.535, 0.58, 0.34, 0.25], facecolor="white")
    inset.scatter(samples.loc[zoomed, "mass_t"], samples.loc[zoomed, "running_s"],
                  s=65, marker="o", facecolor="#444444", edgecolor="#111111",
                  linewidth=0.7, alpha=0.9, zorder=3)
    inset.set(xlim=(x0, x1), ylim=(y0, y1))
    inset.set_title("密集区域放大", fontsize=11, loc="left", pad=8)
    inset.grid(color="#E5E5E5", linewidth=0.55)
    inset.tick_params(labelsize=8, length=0, pad=4)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")

    fig.text(0.105, 0.93, "25趟样本的载重与运行时间分布",
             fontsize=19, weight="bold", color="#1C1C1C")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "validation_sample_coverage25"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Samples: {len(samples)}; zoomed: {zoomed.sum()}")
    print(stem)


if __name__ == "__main__":
    main()
