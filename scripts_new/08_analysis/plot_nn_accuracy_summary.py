# -*- coding: utf-8 -*-
"""Plot the full-line neural-network traction-energy validation summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "output" / "analysis" / "nn_model_accuracy_traceability_full26"),
    )
    parser.add_argument("--output-name", default="神经网络模型全线准确率诊断")
    return parser.parse_args()


def configure_style() -> None:
    regular_font = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold_font = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    for font_path in (regular_font, bold_font):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(regular_font)).get_name() if regular_font.exists() else "Arial"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def main() -> int:
    args = parse_args()
    configure_style()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    details = pd.read_csv(input_dir / "trip_accuracy_detail.csv")
    sections = pd.read_csv(input_dir / "accuracy_summary_by_section.csv")
    overall = pd.read_csv(input_dir / "accuracy_summary_overall.csv").iloc[0]
    if len(details) != int(overall["trip_count"]):
        raise ValueError("Trip count mismatch between detail and overall tables")

    sections = sections.sort_values("mean_absolute_error_pct", ascending=True).reset_index(drop=True)
    mae = sections["mean_absolute_error_pct"].to_numpy(float)
    colors = np.where(mae > 10.0, "#C84630", np.where(mae > 5.0, "#E9A23B", "#4C78A8"))

    fig = plt.figure(figsize=(10.0, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_dist = fig.add_subplot(grid[0, 1])

    y = np.arange(len(sections))
    ax_bar.barh(y, mae, color=colors, height=0.72, edgecolor="none")
    ax_bar.set_yticks(y, sections["section"])
    ax_bar.set_xlabel("逐趟平均绝对误差 (%)")
    ax_bar.set_title("a  26个区间的平均误差", loc="left", fontweight="bold")
    ax_bar.axvline(5.0, color="#707070", linewidth=0.9, linestyle="--")
    ax_bar.axvline(10.0, color="#707070", linewidth=0.9, linestyle=":")
    ax_bar.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax_bar.set_axisbelow(True)
    ax_bar.set_xlim(0, max(17.5, float(np.max(mae)) + 1.5))
    for yi, value in zip(y, mae):
        ax_bar.text(value + 0.18, yi, f"{value:.1f}", va="center", fontsize=6.4)
    ax_bar.text(5.0, len(y) - 0.15, "5%", ha="center", va="bottom", color="#555555", fontsize=6.5)
    ax_bar.text(10.0, len(y) - 0.15, "10%", ha="center", va="bottom", color="#555555", fontsize=6.5)

    abs_error = np.sort(details["fusion_abs_error_pct"].dropna().to_numpy(float))
    cumulative = np.arange(1, len(abs_error) + 1) / len(abs_error) * 100.0
    ax_dist.plot(abs_error, cumulative, color="#2A6F97", linewidth=2.0)
    ax_dist.fill_between(abs_error, cumulative, 0, color="#A9D6E5", alpha=0.35)
    ax_dist.axvline(5.0, color="#E9A23B", linewidth=1.2, linestyle="--")
    ax_dist.axvline(10.0, color="#C84630", linewidth=1.2, linestyle="--")
    ax_dist.set_xlim(0, max(21.0, float(np.max(abs_error)) + 0.5))
    ax_dist.set_ylim(0, 101)
    ax_dist.set_xlabel("单趟累计牵引电能绝对误差 (%)")
    ax_dist.set_ylabel("累计趟次占比 (%)")
    ax_dist.set_title("b  3174趟误差累计分布", loc="left", fontweight="bold")
    ax_dist.grid(color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax_dist.set_axisbelow(True)

    within_5 = float(overall["within_5pct_ratio"]) * 100.0
    within_10 = float(overall["within_10pct_ratio"]) * 100.0
    summary = (
        f"平均绝对误差  {float(overall['mean_absolute_error_pct']):.2f}%\n"
        f"误差中位数    {float(overall['median_absolute_error_pct']):.2f}%\n"
        f"误差≤5%       {within_5:.1f}%\n"
        f"误差≤10%      {within_10:.1f}%\n"
        f"平均累计R²    {float(overall['mean_cumulative_r2']):.4f}"
    )
    ax_dist.text(
        0.97,
        0.06,
        summary,
        transform=ax_dist.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        linespacing=1.55,
        bbox={"boxstyle": "square,pad=0.45", "facecolor": "white", "edgecolor": "#B8B8B8"},
    )

    fig.suptitle(
        "神经网络模型全线牵引电能外部验证",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.965,
        "正向26个区间，共3174趟；蓝色≤5%，橙色5%–10%，红色>10%",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#4D4D4D",
    )

    output_base = input_dir / args.output_name
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_base.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
