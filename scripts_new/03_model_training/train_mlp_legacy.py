# -*- coding: utf-8 -*-
"""
Legacy MLP energy model trainer.

This script is intentionally kept outside the main pipeline.  It trains the
older MLP model directly on Step2 processed xlsx files, so the result can be
used as an accuracy comparison against the current residual model.

Typical usage:
    python scripts_new/03_model_training/train_mlp_legacy.py \
        --data-dir data/data_processed_step2_v3_first5_curve_quality \
        --line-scope 5
"""
from __future__ import annotations

import argparse
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_processed"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models" / "nn_results_mlp_legacy_compare"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


LINE5_SECTIONS = [
    "布政-张家潭",
    "张家潭-同德路",
    "同德路-石碶",
    "石碶-雅渡",
    "雅渡-庙堰",
    "庙堰-钟公庙",
    "钟公庙-鄞州区政府",
    "鄞州区政府-钱湖南路",
    "钱湖南路-南高教园区",
    "南高教园区-下应路",
    "下应路-大洋江",
    "大洋江-泗港",
    "泗港-曹隘",
    "曹隘-柳隘",
    "柳隘-海晏北路",
    "海晏北路-民安东路",
    "民安东路-会展中心",
    "会展中心-院士路",
    "院士路-盎孟港",
    "盎孟港-三官堂",
    "三官堂-兴庄路",
    "兴庄路-兴海南路",
    "兴海南路-梅堰",
    "梅堰-永茂路",
    "永茂路-镇海大道",
    "镇海大道-骆驼桥",
]

REQUIRED_COLUMNS = [
    "时刻",
    "速度(m/s)",
    "加速度(m/s²)",
    "curvature",
    "gradient",
    "重量",
    "energy",
    "segment",
]
OPTIONAL_COLUMNS = [
    "cumulative_energy_kWh",
    "累计位移(m)",
    "区段",
    "日期+服务号",
    "服务号",
    "运行等级",
    "曲线质量标签",
]


class Net(nn.Module):
    """Original small MLP used by the legacy model."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        return torch.nn.functional.softplus(self.fc3(x))


@dataclass
class SectionResult:
    section: str
    status: str
    sample_count: int = 0
    train_count: int = 0
    test_count: int = 0
    feature_count: int = 0
    best_epoch: int | None = None
    best_val_loss: float | None = None
    test_mse_wh2: float | None = None
    test_rmse_wh: float | None = None
    test_mae_wh: float | None = None
    test_bias_wh: float | None = None
    test_mape_percent: float | None = None
    test_smape_percent: float | None = None
    test_r2: float | None = None
    triple_axis_plot_count: int = 0
    output_dir: str | None = None
    message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the standalone legacy MLP model for accuracy comparison."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Step2 result directory. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for legacy MLP comparison results.",
    )
    parser.add_argument(
        "--line-scope",
        type=int,
        default=5,
        help="Train only the first N forward sections. Default is 5 for legacy comparison.",
    )
    parser.add_argument(
        "--sections",
        default=None,
        help="Comma-separated section names. When set, this overrides --line-scope.",
    )
    parser.add_argument("--max-epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap per section for quick smoke tests.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output root before training.",
    )
    parser.add_argument(
        "--save-feature-matrix",
        action="store_true",
        help="Also save the cleaned feature matrix for each section. This can be large.",
    )
    parser.add_argument(
        "--plot-triple-axis",
        dest="plot_triple_axis",
        action="store_true",
        default=True,
        help="Draw MLP velocity/energy/distance triple-axis figures after each section is trained.",
    )
    parser.add_argument(
        "--no-plot-triple-axis",
        dest="plot_triple_axis",
        action="store_false",
        help="Skip MLP triple-axis figures.",
    )
    parser.add_argument(
        "--max-triple-axis-runs-per-section",
        type=int,
        default=None,
        help="Optional plot limit per section. Default draws all trips in each selected section.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Triple-axis figure format.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Triple-axis figure DPI.",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def select_sections(args: argparse.Namespace) -> list[str]:
    if args.sections:
        return [item.strip() for item in args.sections.split(",") if item.strip()]
    if args.line_scope is not None:
        if args.line_scope <= 0:
            raise ValueError("--line-scope must be a positive integer.")
        return LINE5_SECTIONS[: args.line_scope]
    return list(LINE5_SECTIONS)


def read_multi_day_results(
    data_dir: Path,
    station_pair: str,
    nrows: int | None = None,
) -> tuple[pd.DataFrame | None, list[Path]]:
    """Read all results_{section}*.xlsx files from the selected Step2 directory."""
    files = sorted(data_dir.glob(f"results_{station_pair}*.xlsx"))
    if not files:
        return None, []

    wanted = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    frames: list[pd.DataFrame] = []
    used_files: list[Path] = []
    for file_path in files:
        try:
            df = pd.read_excel(file_path, usecols=lambda col: str(col) in wanted, nrows=nrows)
            df["source_file"] = file_path.name
            frames.append(df)
            used_files.append(file_path)
        except Exception as exc:  # pragma: no cover - local data quality guard
            print(f"[{station_pair}] 跳过读取失败文件: {file_path} ({exc})")

    if not frames:
        return None, used_files
    return pd.concat(frames, ignore_index=True), used_files


def to_numeric_time(series: pd.Series) -> pd.Series:
    """Convert the Step2 time column to seconds while tolerating datetime-like values."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return numeric

    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().any():
        return (dt - dt.min()).dt.total_seconds()
    return numeric


def prepare_training_frame(
    df: pd.DataFrame,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    work = df.copy()
    if max_rows is not None and len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=42).sort_index().reset_index(drop=True)

    work["time"] = to_numeric_time(work["时刻"])
    work["velocity"] = pd.to_numeric(work["速度(m/s)"], errors="coerce")
    work["acceleration"] = pd.to_numeric(work["加速度(m/s²)"], errors="coerce")
    work["curvature"] = pd.to_numeric(work["curvature"], errors="coerce")
    work["gradient"] = pd.to_numeric(work["gradient"], errors="coerce")
    work["mass"] = pd.to_numeric(work["重量"], errors="coerce")
    work["energy_wh"] = pd.to_numeric(work["energy"], errors="coerce") / 3600.0

    group_cols = ["source_file", "segment"] if "source_file" in work.columns else ["segment"]
    work.sort_values(group_cols + ["time"], inplace=True)
    work["prev_velocity"] = work.groupby(group_cols)["velocity"].shift(1)
    work["prev_acceleration"] = work.groupby(group_cols)["acceleration"].shift(1)
    work["prev_velocity"] = work["prev_velocity"].fillna(work["velocity"])
    work["prev_acceleration"] = work["prev_acceleration"].fillna(work["acceleration"])

    feature_cols = [
        "time",
        "velocity",
        "acceleration",
        "prev_velocity",
        "prev_acceleration",
        "curvature",
        "gradient",
        "mass",
    ]
    X_df = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = work["energy_wh"].to_numpy(dtype=float)

    finite_mask = np.isfinite(X_df.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y)
    finite_mask &= y >= 0
    work = work.loc[finite_mask].reset_index(drop=True)
    X_df = X_df.loc[finite_mask].reset_index(drop=True)
    y = y[finite_mask]

    if len(X_df) < 10:
        raise ValueError(f"usable sample count is too small: {len(X_df)}")

    stds = X_df.std(axis=0, ddof=0)
    keep_cols = stds[stds > 0].index.tolist()
    dropped_cols = [col for col in X_df.columns if col not in keep_cols]
    if not keep_cols:
        raise ValueError("all feature columns are constant after cleaning")

    return X_df[keep_cols], y.reshape(-1), work, dropped_cols


def make_loader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def safe_name(value: object, fallback: str = "unknown") -> str:
    text = str(value) if value is not None and not pd.isna(value) else fallback
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80] or fallback


def anchor_origin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.array([0.0]), np.array([0.0])
    if abs(x[0]) <= 1e-9 and abs(y[0]) <= 1e-9:
        return x, y
    return np.r_[0.0, x], np.r_[0.0, y]


def first_value(df: pd.DataFrame, column: str, default: str = "") -> object:
    if column not in df.columns or df.empty:
        return default
    value = df[column].dropna()
    return value.iloc[0] if len(value) else default


def predict_energy_wh(
    model: Net,
    scaler: StandardScaler,
    X_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    X_scaled = scaler.transform(X_df.to_numpy(dtype=float))
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):
            batch = torch.tensor(X_scaled[start : start + batch_size], dtype=torch.float32).to(device)
            pred = model(batch).cpu().numpy().reshape(-1)
            predictions.append(pred)
    if not predictions:
        return np.array([], dtype=float)
    return np.concatenate(predictions)


def prepare_plot_arrays(trip: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trip = trip.sort_values("time").reset_index(drop=True)
    t = pd.to_numeric(trip["time"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(t) == 0:
        raise ValueError("empty trip")
    t = t - t[0]
    if len(t) > 1 and np.nanmax(t) > 10000:
        t = t / 1000.0
    if len(t) > 1 and (np.nanmax(t) <= 0 or np.any(np.diff(t) < -1e-6)):
        t = np.arange(len(trip), dtype=float) * 0.05

    v_mps = pd.to_numeric(trip["velocity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    v_kmh = v_mps * 3.6

    if "累计位移(m)" in trip.columns:
        distance = pd.to_numeric(trip["累计位移(m)"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        distance = distance - distance[0]
    else:
        dt = np.diff(t, prepend=t[0])
        dt = np.where(dt < 0, 0.0, dt)
        distance = np.cumsum(v_mps * dt)

    e_real_step = pd.to_numeric(trip["actual_energy_wh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    e_mlp_step = pd.to_numeric(trip["mlp_energy_wh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    L = min(len(t), len(v_kmh), len(distance), len(e_real_step), len(e_mlp_step))
    if L < 2:
        raise ValueError("trip is too short to plot")

    return t[:L], v_kmh[:L], distance[:L], e_real_step[:L], e_mlp_step[:L]


def plot_mlp_trip(
    section: str,
    trip_key: object,
    trip: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> dict[str, object]:
    t, v_kmh, distance, e_real_step, e_mlp_step = prepare_plot_arrays(trip)

    e_real_cum = np.cumsum(np.maximum(e_real_step, 0.0))
    e_mlp_cum = np.cumsum(np.maximum(e_mlp_step, 0.0))
    real_total = float(e_real_cum[-1]) if len(e_real_cum) else 0.0
    mlp_total = float(e_mlp_cum[-1]) if len(e_mlp_cum) else 0.0
    mlp_error_pct = None
    if abs(real_total) > 1e-9:
        mlp_error_pct = (mlp_total - real_total) / real_total * 100.0
    error_text = "N/A" if mlp_error_pct is None else f"{mlp_error_pct:+.2f}%"

    t_v, v_plot = anchor_origin(t, v_kmh)
    t_s, s_plot = anchor_origin(t, distance)
    t_er, e_real_plot = anchor_origin(t, e_real_cum)
    t_em, e_mlp_plot = anchor_origin(t, e_mlp_cum)

    fig, ax1 = plt.subplots(figsize=(13, 7))
    plt.subplots_adjust(right=0.84)

    color_v = "tab:blue"
    ax1.set_xlabel("时间 Time (s)", fontsize=11)
    ax1.set_ylabel("速度 Velocity (km/h)", color=color_v, fontsize=11)
    lns1 = ax1.plot(t_v, v_plot, color=color_v, linewidth=1.6, label="速度 (v)")
    ax1.tick_params(axis="y", labelcolor=color_v)
    ax1.grid(True, alpha=0.22)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=min(0, float(np.nanmin(v_plot)) if len(v_plot) else 0))

    ax2 = ax1.twinx()
    color_e = "tab:red"
    ax2.set_ylabel("累计能耗 Energy (Wh)", color=color_e, fontsize=11)
    lns2_1 = ax2.plot(t_er, e_real_plot, color="black", linewidth=2.0, label="真实能耗 (E_real)")
    lns2_2 = ax2.plot(t_em, e_mlp_plot, color=color_e, linestyle="--", linewidth=2.0, label="MLP预测能耗 (E_mlp)")
    ax2.tick_params(axis="y", labelcolor=color_e)
    ax2.set_ylim(bottom=0)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 62))
    color_s = "tab:green"
    ax3.set_ylabel("累计位移 Distance (m)", color=color_s, fontsize=11)
    lns3 = ax3.plot(t_s, s_plot, color=color_s, linewidth=1.6, label="累计位移 (s)")
    ax3.tick_params(axis="y", labelcolor=color_s)
    ax3.set_ylim(bottom=0)

    service = first_value(trip, "服务号")
    date_service = first_value(trip, "日期+服务号")
    run_class = first_value(trip, "运行等级")
    quality = first_value(trip, "曲线质量标签")
    segment_id = first_value(trip, "segment", trip_key)
    source_file = first_value(trip, "source_file")

    lns = lns1 + lns2_1 + lns2_2 + lns3
    labs = [line.get_label() for line in lns]
    ax1.legend(lns, labs, loc="upper left", fontsize=9)

    subtitle_items = [f"segment {segment_id}"]
    if date_service != "":
        subtitle_items.append(f"日期+服务号 {date_service}")
    elif service != "":
        subtitle_items.append(f"服务号 {service}")
    if run_class != "":
        subtitle_items.append(f"class{run_class}")
    if quality != "":
        subtitle_items.append(f"质量标签 {quality}")
    subtitle_items.append(f"能耗误差 {error_text}")

    plt.title(
        f"{section} MLP区间运行综合对标图\n"
        f"[速度 v | 位移 s | MLP能耗 E]  {' | '.join(map(str, subtitle_items))}",
        fontsize=13,
    )
    plt.tight_layout()
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight", "facecolor": "white"}
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)

    return {
        "section": section,
        "segment": segment_id,
        "rows": len(t),
        "duration_s": round(float(t[-1]) if len(t) else 0.0, 3),
        "distance_m": round(float(distance[-1]) if len(distance) else 0.0, 3),
        "real_energy_wh": round(real_total, 3),
        "mlp_energy_wh": round(mlp_total, 3),
        "mlp_error_pct": "" if mlp_error_pct is None else round(float(mlp_error_pct), 3),
        "run_class": run_class,
        "quality_label": quality,
        "service_no": service,
        "date_service": date_service,
        "source_file": source_file,
        "figure": str(out_path),
    }


def save_mlp_triple_axis_plots(
    section: str,
    cleaned_df: pd.DataFrame,
    y_true_wh: np.ndarray,
    y_pred_wh: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    if len(cleaned_df) == 0 or len(y_pred_wh) == 0:
        return 0

    plot_df = cleaned_df.copy().reset_index(drop=True)
    L = min(len(plot_df), len(y_true_wh), len(y_pred_wh))
    plot_df = plot_df.iloc[:L].copy()
    plot_df["actual_energy_wh"] = np.asarray(y_true_wh[:L], dtype=float)
    plot_df["mlp_energy_wh"] = np.asarray(y_pred_wh[:L], dtype=float)

    plot_dir = output_dir / "triple_axis_mlp"
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["source_file", "segment"] if "source_file" in plot_df.columns else ["segment"]
    groups = list(plot_df.groupby(group_cols, sort=False))
    if args.max_triple_axis_runs_per_section is not None:
        groups = groups[: args.max_triple_axis_runs_per_section]

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for idx, (group_key, trip) in enumerate(groups, start=1):
        try:
            segment_id = first_value(trip, "segment", group_key)
            date_service = first_value(trip, "日期+服务号")
            service = first_value(trip, "服务号")
            label_parts = [f"seg_{safe_name(segment_id)}"]
            if date_service != "":
                label_parts.append(safe_name(date_service))
            elif service != "":
                label_parts.append(safe_name(service))
            fig_name = "_".join(label_parts) + f".{args.plot_format}"
            rows.append(plot_mlp_trip(section, group_key, trip, plot_dir / fig_name, args.plot_dpi))
        except Exception as exc:
            skipped.append({"section": section, "trip_key": str(group_key), "reason": str(exc)})

        if idx % 10 == 0 or idx == len(groups):
            print(f"[{section}] MLP三轴图已绘制 {idx}/{len(groups)}")

    pd.DataFrame(rows).to_csv(plot_dir / "triple_axis_mlp_summary.csv", index=False, encoding="utf-8-sig")
    if skipped:
        pd.DataFrame(skipped).to_csv(plot_dir / "triple_axis_mlp_skipped.csv", index=False, encoding="utf-8-sig")
    return len(rows)


def save_training_plots(
    output_dir: Path,
    train_losses: list[float],
    val_losses: list[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 loss")
    plt.title("Legacy MLP training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=220)
    plt.close()

    sample_count = min(5000, len(y_true))
    if sample_count > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_true), size=sample_count, replace=False)
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true[idx], y_pred[idx], s=8, alpha=0.35)
        low = float(min(np.min(y_true[idx]), np.min(y_pred[idx])))
        high = float(max(np.max(y_true[idx]), np.max(y_pred[idx])))
        plt.plot([low, high], [low, high], color="black", linewidth=1.2, linestyle="--")
        plt.xlabel("Actual energy increment (Wh)")
        plt.ylabel("Predicted energy increment (Wh)")
        plt.title("Legacy MLP prediction scatter")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "pred_vs_true_scatter.png", dpi=220)
        plt.close()

    errors = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=80, alpha=0.85)
    plt.xlabel("Prediction error (Wh)")
    plt.ylabel("Count")
    plt.title("Legacy MLP test error distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "test_error_hist.png", dpi=220)
    plt.close()


def save_summary_plots(summary_df: pd.DataFrame, output_root: Path) -> None:
    """Save one overview image for comparing section-level legacy MLP errors."""
    if summary_df.empty or "status" not in summary_df.columns:
        return

    ok = summary_df[summary_df["status"] == "success"].copy()
    if ok.empty:
        return

    numeric_cols = [
        "test_rmse_wh",
        "test_mae_wh",
        "test_bias_wh",
        "test_mape_percent",
        "test_smape_percent",
        "test_r2",
    ]
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")

    ok.to_csv(output_root / "legacy_mlp_error_metrics_summary.csv", index=False, encoding="utf-8-sig")

    labels = ok["section"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)

    axes[0].bar(x - width / 2, ok["test_mae_wh"], width, label="MAE")
    axes[0].bar(x + width / 2, ok["test_rmse_wh"], width, label="RMSE")
    axes[0].set_ylabel("Wh")
    axes[0].set_title("Legacy MLP absolute error by section")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, ok["test_bias_wh"], color="#4c78a8")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Wh")
    axes[1].set_title("Mean signed error, positive means over-prediction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x - width / 2, ok["test_mape_percent"], width, label="MAPE, excluding near-zero actual")
    axes[2].bar(x + width / 2, ok["test_smape_percent"], width, label="sMAPE")
    axes[2].set_ylabel("%")
    axes[2].set_title("Relative error by section")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    fig.savefig(output_root / "legacy_mlp_error_summary.png", dpi=240)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax1.bar(x, ok["test_rmse_wh"], width=0.45, color="#f58518", label="RMSE")
    ax1.set_ylabel("RMSE (Wh)", color="#f58518")
    ax1.tick_params(axis="y", labelcolor="#f58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, ok["test_r2"], marker="o", color="#2f855a", linewidth=2, label="R2")
    ax2.set_ylabel("R2", color="#2f855a")
    ax2.tick_params(axis="y", labelcolor="#2f855a")
    ax2.axhline(0, color="#2f855a", linewidth=1, alpha=0.35)

    ax1.set_title("Legacy MLP error and R2 overview")
    fig.savefig(output_root / "legacy_mlp_rmse_r2_overview.png", dpi=240)
    plt.close(fig)


def train_one_section(
    station_pair: str,
    data_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> SectionResult:
    output_dir = output_root / station_pair
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw, used_files = read_multi_day_results(data_dir, station_pair, nrows=args.max_rows)
    if df_raw is None or df_raw.empty:
        return SectionResult(
            section=station_pair,
            status="skipped",
            output_dir=str(output_dir),
            message=f"no matching file under {data_dir}",
        )

    try:
        X_df, y, cleaned_df, dropped_cols = prepare_training_frame(df_raw, max_rows=args.max_rows)
    except Exception as exc:
        return SectionResult(
            section=station_pair,
            status="failed",
            output_dir=str(output_dir),
            message=str(exc),
        )

    X = X_df.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_test_scaled).all():
        return SectionResult(
            section=station_pair,
            status="failed",
            output_dir=str(output_dir),
            message="NaN/Inf appeared after standardization",
        )

    with open(output_dir / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    pd.Series(X_df.columns, name="feature").to_csv(output_dir / "feature_columns.csv", index=False)
    pd.Series(dropped_cols, name="dropped_constant_feature").to_csv(
        output_dir / "dropped_constant_features.csv", index=False
    )
    pd.Series([str(path) for path in used_files], name="file").to_csv(
        output_dir / "merged_input_files.csv", index=False
    )
    if args.save_feature_matrix:
        X_df.assign(target_energy_wh=y).to_csv(output_dir / "feature_matrix.csv", index=False)

    train_loader = make_loader(X_train_scaled, y_train, args.batch_size)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

    model = Net(X_train_scaled.shape[1]).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = np.inf
    best_epoch = 0
    stale_epochs = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = float(criterion(val_outputs, y_test_tensor).item())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if not np.isfinite(val_loss):
            return SectionResult(
                section=station_pair,
                status="failed",
                sample_count=len(X_df),
                output_dir=str(output_dir),
                message="validation loss became NaN/Inf",
            )

        improved = val_loss < best_val_loss - 1e-8
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"[{station_pair}] epoch {epoch:03d}/{args.max_epochs}, "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

        if stale_epochs >= args.patience:
            print(f"[{station_pair}] early stopped at epoch {epoch}, best_epoch={best_epoch}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().reshape(-1)

    test_mse = float(mean_squared_error(y_test, y_pred))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_bias = float(np.mean(y_pred - y_test))
    nonzero_mask = np.abs(y_test) > 1e-6
    if np.any(nonzero_mask):
        test_mape = float(np.mean(np.abs((y_pred[nonzero_mask] - y_test[nonzero_mask]) / y_test[nonzero_mask])) * 100.0)
    else:
        test_mape = float("nan")
    smape_denominator = np.abs(y_test) + np.abs(y_pred)
    smape_mask = smape_denominator > 1e-6
    if np.any(smape_mask):
        test_smape = float(
            np.mean(2.0 * np.abs(y_pred[smape_mask] - y_test[smape_mask]) / smape_denominator[smape_mask]) * 100.0
        )
    else:
        test_smape = float("nan")
    test_r2 = float(r2_score(y_test, y_pred))
    triple_axis_plot_count = 0

    torch.save(model.state_dict(), output_dir / "trained_model.pth")
    shutil.copyfile(best_model_path, output_dir / "best_model_01.pth")
    shutil.copyfile(output_dir / "trained_model.pth", output_dir / "trained_model_01.pth")

    metrics_df = pd.DataFrame(
        [
            ("sample_count", len(X_df)),
            ("train_count", len(X_train)),
            ("test_count", len(X_test)),
            ("feature_count", X_df.shape[1]),
            ("best_epoch", best_epoch),
            ("best_val_loss", best_val_loss),
            ("test_mse_wh2", test_mse),
            ("test_rmse_wh", test_rmse),
            ("test_mae_wh", test_mae),
            ("test_bias_wh", test_bias),
            ("test_mape_percent", test_mape),
            ("test_smape_percent", test_smape),
            ("test_r2", test_r2),
            ("triple_axis_plot_count", triple_axis_plot_count),
        ],
        columns=["metric", "value"],
    )

    pd.DataFrame(
        {
            "actual_energy_wh": y_test,
            "predicted_energy_wh": y_pred,
            "error_wh": y_pred - y_test,
        }
    ).head(20000).to_csv(output_dir / "test_predictions_sample.csv", index=False, encoding="utf-8-sig")

    save_training_plots(output_dir, train_losses, val_losses, y_test, y_pred)
    if args.plot_triple_axis:
        y_full_pred = predict_energy_wh(model, scaler, X_df, device)
        triple_axis_plot_count = save_mlp_triple_axis_plots(
            station_pair,
            cleaned_df,
            y,
            y_full_pred,
            output_dir,
            args,
        )
        print(f"[{station_pair}] MLP三轴图输出数量: {triple_axis_plot_count}")
        metrics_df.loc[metrics_df["metric"] == "triple_axis_plot_count", "value"] = triple_axis_plot_count

    metrics_df.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    return SectionResult(
        section=station_pair,
        status="success",
        sample_count=len(X_df),
        train_count=len(X_train),
        test_count=len(X_test),
        feature_count=X_df.shape[1],
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_mse_wh2=test_mse,
        test_rmse_wh=test_rmse,
        test_mae_wh=test_mae,
        test_bias_wh=test_bias,
        test_mape_percent=test_mape,
        test_smape_percent=test_smape,
        test_r2=test_r2,
        triple_axis_plot_count=triple_axis_plot_count,
        output_dir=str(output_dir),
    )


def main() -> int:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_root = resolve_path(args.output_root)

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return 2

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sections = select_sections(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Legacy MLP comparison training")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Data:    {data_dir}")
    print(f"Output:  {output_root}")
    print(f"Device:  {device}")
    print(f"Sections: {len(sections)}")

    results: list[SectionResult] = []
    for idx, section in enumerate(sections, 1):
        print(f"\n[{idx}/{len(sections)}] Training legacy MLP: {section}")
        result = train_one_section(section, data_dir, output_root, args, device)
        results.append(result)
        if result.status == "success":
            print(
                f"[{section}] done: R2={result.test_r2:.4f}, "
                f"RMSE={result.test_rmse_wh:.3f} Wh, MAE={result.test_mae_wh:.3f} Wh"
            )
        else:
            print(f"[{section}] {result.status}: {result.message}")

    summary_df = pd.DataFrame([result.__dict__ for result in results])
    summary_path = output_root / "batch_train_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    save_summary_plots(summary_df, output_root)

    success_count = int((summary_df["status"] == "success").sum()) if not summary_df.empty else 0
    failed_count = int((summary_df["status"] == "failed").sum()) if not summary_df.empty else 0
    skipped_count = int((summary_df["status"] == "skipped").sum()) if not summary_df.empty else 0
    print("\nLegacy MLP comparison finished")
    print(f"Success: {success_count}, failed: {failed_count}, skipped: {skipped_count}")
    print(f"Summary: {summary_path}")
    print(f"Error plot: {output_root / 'legacy_mlp_error_summary.png'}")
    print(f"RMSE/R2 plot: {output_root / 'legacy_mlp_rmse_r2_overview.png'}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
