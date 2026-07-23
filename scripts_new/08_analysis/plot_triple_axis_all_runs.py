# -*- coding: utf-8 -*-
"""
Batch plot residual-model triple-axis figures for Step2 section results.

The residual training script only draws one validation trip per section and its
curve starts after the sequence warm-up window. This standalone script redraws
every selected trip from Step2 data and explicitly anchors velocity, distance,
and cumulative energy at the departure origin.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.physics.train_simu import TrainTheoreticalEnergyModel


SEQ_LEN = 30
DT_DEFAULT = 0.05

FORWARD_SECTIONS = [
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
    "segment",
    "时刻",
    "速度(m/s)",
    "加速度(m/s²)",
    "累计位移(m)",
    "energy",
    "gradient",
    "curvature",
    "重量",
]

OPTIONAL_LABEL_COLUMNS = [
    "服务号",
    "日期+服务号",
    "运行等级",
    "曲线质量标签",
]

READ_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_LABEL_COLUMNS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        import math

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ResidualTransformerV2(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        out = self.transformer(src)
        return self.decoder(out[:, -1, :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw triple-axis velocity/energy/distance figures for Step2 trips."
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "data_processed_step2_v3_first5_curve_quality"),
        help="Directory containing results_区间.xlsx files.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "output" / "models" / "nn_results_residual_v2"),
        help="Directory containing residual model folders by section.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for generated figures and summary CSV. "
            "Default: a triple_axis_all_runs_* folder under --model-dir."
        ),
    )
    parser.add_argument(
        "--line-scope",
        default="5",
        help="Forward section scope: positive integer, full/all, or first5. Ignored when --all-files is used.",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        help="Explicit section names to draw. Overrides --line-scope unless --all-files is used.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Draw every results_*.xlsx file in --data-dir, including reverse sections.",
    )
    parser.add_argument(
        "--max-runs-per-section",
        type=int,
        default=None,
        help="Optional debug limit for each section. Default draws all trips.",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Figure file format. jpg is the default for easier review in file explorer.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--clean-output",
        dest="clean_output",
        action="store_true",
        default=True,
        help="Delete the output directory before plotting.",
    )
    parser.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Keep existing files in the output directory.",
    )
    return parser.parse_args()


def resolve_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def default_output_dir(args: argparse.Namespace, model_dir: Path) -> Path:
    if args.all_files:
        suffix = "all_files"
    elif args.sections:
        suffix = "selected"
    else:
        raw = str(args.line_scope).strip().lower()
        if raw == "first5":
            suffix = "first5"
        elif raw in {"full", "all"}:
            suffix = "full"
        else:
            suffix = f"first{raw}"
    return model_dir / f"triple_axis_all_runs_{suffix}"


def select_sections(args: argparse.Namespace, data_dir: Path) -> list[str]:
    if args.all_files:
        sections = []
        for path in sorted(data_dir.glob("results_*.xlsx")):
            name = path.stem
            if name.startswith("results_"):
                sections.append(name[len("results_") :])
        return sections

    if args.sections:
        return args.sections

    raw = str(args.line_scope).strip().lower()
    if raw in {"full", "all"}:
        return FORWARD_SECTIONS
    if raw == "first5":
        raw = "5"
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError("--line-scope must be a positive integer, full/all, or first5.") from exc
    if count < 1:
        raise ValueError("--line-scope must be >= 1.")
    return FORWARD_SECTIONS[:count]


def safe_name(value: object, fallback: str = "unknown") -> str:
    text = str(value) if value is not None and not pd.isna(value) else fallback
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80] or fallback


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_model(section: str, model_root: Path, device: torch.device):
    section_dir = model_root / section
    model_path = section_dir / "best_res_model.pth"
    scaler_x_path = section_dir / "scaler_x.pkl"
    scaler_y_path = section_dir / "scaler_y.pkl"

    missing = [p.name for p in [model_path, scaler_x_path, scaler_y_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{section} missing model files: {', '.join(missing)}")

    scaler_x = load_pickle(scaler_x_path)
    scaler_y = load_pickle(scaler_y_path)
    input_dim = int(getattr(scaler_x, "n_features_in_", 6))
    model = ResidualTransformerV2(input_dim=input_dim).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, scaler_x, scaler_y


def read_section_data(section: str, data_dir: Path) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(data_dir / f"results_{section}*.xlsx")))
    if not files:
        return None
    wanted = set(READ_COLUMNS)
    frames = [pd.read_excel(path, usecols=lambda col: col in wanted) for path in files]
    return pd.concat(frames, ignore_index=True)


def to_numeric_series(df: pd.DataFrame, col: str, fill_value: float = 0.0) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(fill_value).to_numpy(dtype=float)


def prepare_trip_arrays(trip: pd.DataFrame) -> dict[str, np.ndarray | float]:
    trip = trip.copy().reset_index(drop=True)
    for col in REQUIRED_COLUMNS:
        if col not in trip.columns:
            raise KeyError(f"Missing required column: {col}")

    t = to_numeric_series(trip, "时刻")
    if len(t) == 0:
        raise ValueError("empty trip")
    if np.nanmax(t) > 10000:
        t = (t - t[0]) / 1000.0
    else:
        t = t - t[0]
    if len(t) > 1 and (np.nanmax(t) <= 0 or np.any(np.diff(t) < -1e-6)):
        t = np.arange(len(trip), dtype=float) * DT_DEFAULT

    v = to_numeric_series(trip, "速度(m/s)")
    s = to_numeric_series(trip, "累计位移(m)")
    s = s - s[0]

    if "加速度(m/s²)" in trip.columns:
        a = to_numeric_series(trip, "加速度(m/s²)")
    else:
        dt = np.gradient(t) if len(t) > 1 else np.array([DT_DEFAULT])
        dt = np.where(np.abs(dt) < 1e-9, DT_DEFAULT, dt)
        a = np.gradient(v) / dt

    energy_real_step = to_numeric_series(trip, "energy") / 3.6e6 * 1000.0
    gradient = to_numeric_series(trip, "gradient")
    curvature = to_numeric_series(trip, "curvature")
    mass = to_numeric_series(trip, "重量")
    mass_value = float(mass[0]) if len(mass) else 0.0

    return {
        "time": t,
        "velocity_mps": v,
        "velocity_kmh": v * 3.6,
        "acceleration": a,
        "distance": s,
        "energy_real_step": energy_real_step,
        "gradient": gradient,
        "curvature": curvature,
        "mass": mass,
        "mass_value": mass_value,
    }


def predict_fusion_energy(
    arrays: dict[str, np.ndarray | float],
    model: ResidualTransformerV2,
    scaler_x,
    scaler_y,
    sim_model: TrainTheoreticalEnergyModel,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    t = arrays["time"]
    v = arrays["velocity_mps"]
    a = arrays["acceleration"]
    gradient = arrays["gradient"]
    curvature = arrays["curvature"]
    mass = arrays["mass"]
    mass_value = float(arrays["mass_value"])

    e_phy_step = sim_model.run_batch_simulation(t, v, mass_value)
    e_phy_step = np.asarray(e_phy_step, dtype=float)
    L = min(len(t), len(e_phy_step), len(v), len(a), len(gradient), len(curvature), len(mass))
    e_phy_step = e_phy_step[:L]

    raw_x = pd.DataFrame(
        {
            "v": v[:L],
            "a": a[:L],
            "e_phy": e_phy_step,
            "grad": gradient[:L],
            "mass": mass[:L],
            "curvature": curvature[:L],
        }
    ).fillna(0.0).to_numpy(dtype=float)

    residual_full = np.zeros(L, dtype=float)
    if L > SEQ_LEN:
        scaled_x = scaler_x.transform(raw_x)
        windows = np.asarray([scaled_x[i : i + SEQ_LEN] for i in range(L - SEQ_LEN)], dtype=np.float32)
        if len(windows):
            with torch.no_grad():
                pred_scaled = model(torch.from_numpy(windows).to(device)).cpu().numpy()
            residual = scaler_y.inverse_transform(pred_scaled).flatten()
            residual_full[SEQ_LEN - 1 : SEQ_LEN - 1 + len(residual)] = residual

    fusion_step = np.maximum(e_phy_step + residual_full, 0.0)
    return e_phy_step, fusion_step


def anchor_origin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.array([0.0]), np.array([0.0])
    if abs(x[0]) <= 1e-9 and abs(y[0]) <= 1e-9:
        return x, y
    return np.r_[0.0, x], np.r_[0.0, y]


def plot_trip(
    section: str,
    segment_id: object,
    trip: pd.DataFrame,
    arrays: dict[str, np.ndarray | float],
    e_fusion_step: np.ndarray,
    out_path: Path,
    dpi: int,
) -> dict[str, object]:
    t = arrays["time"]
    v_kmh = arrays["velocity_kmh"]
    s = arrays["distance"]
    e_real_step = arrays["energy_real_step"]

    L = min(len(t), len(v_kmh), len(s), len(e_real_step), len(e_fusion_step))
    t = np.asarray(t[:L], dtype=float)
    v_kmh = np.asarray(v_kmh[:L], dtype=float)
    s = np.asarray(s[:L], dtype=float)

    e_real_cum = np.cumsum(np.maximum(np.asarray(e_real_step[:L], dtype=float), 0.0))
    e_fusion_cum = np.cumsum(np.maximum(np.asarray(e_fusion_step[:L], dtype=float), 0.0))
    real_total = float(e_real_cum[-1]) if len(e_real_cum) else 0.0
    fusion_total = float(e_fusion_cum[-1]) if len(e_fusion_cum) else 0.0
    fusion_error_pct = None
    if abs(real_total) > 1e-9:
        fusion_error_pct = (fusion_total - real_total) / real_total * 100.0
    error_text = "N/A" if fusion_error_pct is None else f"{fusion_error_pct:+.2f}%"

    t_v, v_plot = anchor_origin(t, v_kmh)
    t_s, s_plot = anchor_origin(t, s)
    t_er, e_real_plot = anchor_origin(t, e_real_cum)
    t_ef, e_fusion_plot = anchor_origin(t, e_fusion_cum)

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
    lns2_2 = ax2.plot(t_ef, e_fusion_plot, color=color_e, linestyle="--", linewidth=2.0, label="修正能耗 (E_fusion)")
    ax2.tick_params(axis="y", labelcolor=color_e)
    ax2.set_ylim(bottom=0)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 62))
    color_s = "tab:green"
    ax3.set_ylabel("累计位移 Distance (m)", color=color_s, fontsize=11)
    lns3 = ax3.plot(t_s, s_plot, color=color_s, linewidth=1.6, label="累计位移 (s)")
    ax3.tick_params(axis="y", labelcolor=color_s)
    ax3.set_ylim(bottom=0)

    service = trip["服务号"].iloc[0] if "服务号" in trip.columns and len(trip) else ""
    date_service = trip["日期+服务号"].iloc[0] if "日期+服务号" in trip.columns and len(trip) else ""
    run_class = trip["运行等级"].iloc[0] if "运行等级" in trip.columns and len(trip) else ""
    quality = trip["曲线质量标签"].iloc[0] if "曲线质量标签" in trip.columns and len(trip) else ""

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
        f"{section} 区间运行综合对标图\n"
        f"[速度 v | 位移 s | 融合能耗 E]  {' | '.join(map(str, subtitle_items))}",
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
        "rows": L,
        "duration_s": round(float(t[-1]) if len(t) else 0.0, 3),
        "distance_m": round(float(s[-1]) if len(s) else 0.0, 3),
        "real_energy_wh": round(real_total, 3),
        "fusion_energy_wh": round(fusion_total, 3),
        "fusion_error_pct": "" if fusion_error_pct is None else round(float(fusion_error_pct), 3),
        "run_class": run_class,
        "quality_label": quality,
        "service_no": service,
        "date_service": date_service,
        "figure": str(out_path),
    }


def configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def main() -> int:
    args = parse_args()
    configure_plot_style()

    data_dir = resolve_path(args.data_dir)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(args, model_dir)

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = select_sections(args, data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_model = TrainTheoreticalEnergyModel()

    print(f"数据目录: {data_dir}")
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    print(f"绘图区间数: {len(sections)}")
    print(f"推理设备: {device}")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for section_idx, section in enumerate(sections, start=1):
        print(f"\n[{section_idx}/{len(sections)}] 处理区间: {section}")
        df = read_section_data(section, data_dir)
        if df is None or df.empty:
            msg = "未找到结果文件"
            print(f"  跳过: {msg}")
            skipped.append({"section": section, "reason": msg})
            continue

        try:
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise KeyError(f"缺少列: {', '.join(missing_cols)}")
            model, scaler_x, scaler_y = load_model(section, model_dir, device)
        except Exception as exc:
            print(f"  跳过: {exc}")
            skipped.append({"section": section, "reason": str(exc)})
            continue

        section_out = output_dir / safe_name(section, "section")
        section_out.mkdir(parents=True, exist_ok=True)

        segment_ids = list(pd.Series(df["segment"]).dropna().unique())
        if args.max_runs_per_section is not None:
            segment_ids = segment_ids[: args.max_runs_per_section]

        print(f"  待绘制趟次: {len(segment_ids)}")
        for run_idx, segment_id in enumerate(segment_ids, start=1):
            trip = df[df["segment"] == segment_id].copy()
            try:
                arrays = prepare_trip_arrays(trip)
                _, e_fusion_step = predict_fusion_energy(arrays, model, scaler_x, scaler_y, sim_model, device)
                service = trip["服务号"].iloc[0] if "服务号" in trip.columns and len(trip) else ""
                date_service = trip["日期+服务号"].iloc[0] if "日期+服务号" in trip.columns and len(trip) else ""
                label_parts = [f"seg_{int(segment_id):03d}" if float(segment_id).is_integer() else f"seg_{safe_name(segment_id)}"]
                if date_service != "":
                    label_parts.append(safe_name(date_service))
                elif service != "":
                    label_parts.append(safe_name(service))
                fig_name = "_".join(label_parts) + f".{args.format}"
                out_path = section_out / fig_name
                rows.append(plot_trip(section, segment_id, trip, arrays, e_fusion_step, out_path, args.dpi))
            except Exception as exc:
                skipped.append({"section": section, "segment": segment_id, "reason": str(exc)})
                print(f"    跳过 segment {segment_id}: {exc}")

            if run_idx % 10 == 0 or run_idx == len(segment_ids):
                print(f"    已完成 {run_idx}/{len(segment_ids)}")

    summary_path = output_dir / "triple_axis_plot_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "section",
            "segment",
            "rows",
            "duration_s",
            "distance_m",
            "real_energy_wh",
            "fusion_energy_wh",
            "fusion_error_pct",
            "run_class",
            "quality_label",
            "service_no",
            "date_service",
            "figure",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if skipped:
        skipped_path = output_dir / "triple_axis_plot_skipped.csv"
        with skipped_path.open("w", newline="", encoding="utf-8-sig") as f:
            fieldnames = sorted({key for item in skipped for key in item.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(skipped)
        print(f"\n跳过记录: {skipped_path}")

    print(f"\n完成绘图: {len(rows)} 张")
    print(f"汇总文件: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
