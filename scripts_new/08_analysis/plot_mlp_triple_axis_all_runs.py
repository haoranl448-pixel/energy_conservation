# -*- coding: utf-8 -*-
"""
Plot MLP triple-axis figures from already trained legacy MLP models.

This script does not train anything. It loads each section's existing
best_model.pth, scaler.pkl, and feature_columns.csv, then redraws Step2 trips as:
velocity v / cumulative distance s / real cumulative energy / MLP cumulative energy.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = PROJECT_ROOT / "scripts_new" / "03_model_training" / "train_mlp_legacy.py"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_processed_step2_v3_first5_curve_quality"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "output" / "models" / "nn_results_mlp_legacy_compare"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("train_mlp_legacy_runtime", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy MLP helper script: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_legacy_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw MLP triple-axis figures from trained legacy MLP models.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Step2 result directory. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--model-root",
        default=str(DEFAULT_MODEL_ROOT),
        help="Root directory containing legacy MLP section folders.",
    )
    parser.add_argument(
        "--line-scope",
        default="5",
        help="Forward section scope: positive integer, full/all, or first5.",
    )
    parser.add_argument(
        "--sections",
        default=None,
        help="Comma-separated section names. Overrides --line-scope.",
    )
    parser.add_argument(
        "--max-runs-per-section",
        type=int,
        default=None,
        help="Optional debug limit for each section. Default draws all trips.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap per section for quick smoke tests.",
    )
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Figure format.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        default=True,
        help="Delete existing triple_axis_mlp folders before drawing.",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Keep existing triple_axis_mlp folders.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only summarize existing triple_axis_mlp_summary.csv files; do not redraw figures.",
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

    raw = str(args.line_scope).strip().lower()
    if raw in {"full", "all"}:
        return list(legacy.LINE5_SECTIONS)
    if raw == "first5":
        raw = "5"
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError("--line-scope must be a positive integer, full/all, or first5.") from exc
    if count <= 0:
        raise ValueError("--line-scope must be positive.")
    return list(legacy.LINE5_SECTIONS[:count])


def load_feature_columns(section_dir: Path) -> list[str]:
    path = section_dir / "feature_columns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_columns.csv: {path}")
    df = pd.read_csv(path)
    if "feature" in df.columns:
        cols = df["feature"].dropna().astype(str).tolist()
    else:
        cols = df.iloc[:, 0].dropna().astype(str).tolist()
    if not cols:
        raise ValueError(f"No feature columns found in {path}")
    return cols


def load_model(section_dir: Path, feature_count: int, device: torch.device):
    model_path = section_dir / "best_model.pth"
    if not model_path.exists():
        model_path = section_dir / "best_model_01.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing best_model.pth: {section_dir}")

    scaler_path = section_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.pkl: {section_dir}")

    with scaler_path.open("rb") as file:
        scaler = legacy.pickle.load(file)

    model = legacy.Net(feature_count).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, scaler


def prepare_aligned_features(df_raw: pd.DataFrame, feature_cols: list[str], max_rows: int | None):
    X_df, y, cleaned_df, _ = legacy.prepare_training_frame(df_raw, max_rows=max_rows)

    # Use the exact feature list saved during training. This avoids accidental
    # feature-order drift when a column is constant in a plotting subset.
    source = cleaned_df.copy()
    for col in X_df.columns:
        if col not in source.columns:
            source[col] = X_df[col]

    missing = [col for col in feature_cols if col not in source.columns]
    if missing:
        raise KeyError(f"Missing trained features in current data: {missing}")

    aligned = source[feature_cols].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(aligned.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y)
    aligned = aligned.loc[finite].reset_index(drop=True)
    y = y[finite]
    cleaned_df = cleaned_df.loc[finite].reset_index(drop=True)
    return aligned, y, cleaned_df


def plot_one_section(
    section: str,
    data_dir: Path,
    model_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    section_dir = model_root / section
    if not section_dir.exists():
        raise FileNotFoundError(f"Missing model section directory: {section_dir}")

    if args.clean:
        plot_dir = section_dir / "triple_axis_mlp"
        if plot_dir.exists():
            shutil.rmtree(plot_dir)

    feature_cols = load_feature_columns(section_dir)
    model, scaler = load_model(section_dir, len(feature_cols), device)

    df_raw, used_files = legacy.read_multi_day_results(data_dir, section, nrows=args.max_rows)
    if df_raw is None or df_raw.empty:
        raise FileNotFoundError(f"No Step2 file found for {section} in {data_dir}")

    X_df, y, cleaned_df = prepare_aligned_features(df_raw, feature_cols, args.max_rows)
    y_pred = legacy.predict_energy_wh(model, scaler, X_df, device)

    plot_args = argparse.Namespace(
        max_triple_axis_runs_per_section=args.max_runs_per_section,
        plot_format=args.format,
        plot_dpi=args.dpi,
    )
    plot_count = legacy.save_mlp_triple_axis_plots(section, cleaned_df, y, y_pred, section_dir, plot_args)

    summary_path = section_dir / "triple_axis_mlp" / "triple_axis_mlp_summary.csv"
    return {
        "section": section,
        "status": "success",
        "rows": len(cleaned_df),
        "plot_count": plot_count,
        "input_files": ";".join(str(path) for path in used_files),
        "summary": str(summary_path),
    }


def summarize_existing_triple_axis(model_root: Path, sections: list[str]) -> tuple[Path, Path | None]:
    all_frames: list[pd.DataFrame] = []
    missing: list[dict[str, str]] = []

    for section in sections:
        summary_path = model_root / section / "triple_axis_mlp" / "triple_axis_mlp_summary.csv"
        if not summary_path.exists():
            missing.append({"section": section, "reason": f"missing {summary_path}"})
            continue
        df = pd.read_csv(summary_path)
        if df.empty:
            missing.append({"section": section, "reason": "empty summary"})
            continue
        all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError("No triple_axis_mlp_summary.csv files found for selected sections.")

    detail = pd.concat(all_frames, ignore_index=True)
    for col in ["duration_s", "distance_m", "real_energy_wh", "mlp_energy_wh", "mlp_error_pct"]:
        detail[col] = pd.to_numeric(detail[col], errors="coerce")

    detail["signed_error_wh"] = detail["mlp_energy_wh"] - detail["real_energy_wh"]
    detail["abs_error_wh"] = detail["signed_error_wh"].abs()
    detail["abs_error_pct"] = detail["mlp_error_pct"].abs()

    def summarize_group(group: pd.DataFrame) -> pd.Series:
        real_sum = float(group["real_energy_wh"].sum())
        mlp_sum = float(group["mlp_energy_wh"].sum())
        weighted_error_pct = np.nan
        if abs(real_sum) > 1e-9:
            weighted_error_pct = (mlp_sum - real_sum) / real_sum * 100.0

        return pd.Series(
            {
                "trip_count": int(len(group)),
                "avg_duration_s": group["duration_s"].mean(),
                "avg_distance_m": group["distance_m"].mean(),
                "avg_real_energy_wh": group["real_energy_wh"].mean(),
                "avg_mlp_energy_wh": group["mlp_energy_wh"].mean(),
                "avg_signed_error_wh": group["signed_error_wh"].mean(),
                "avg_abs_error_wh": group["abs_error_wh"].mean(),
                "rmse_total_error_wh": float(np.sqrt(np.mean(np.square(group["signed_error_wh"].dropna())))),
                "avg_error_pct": group["mlp_error_pct"].mean(),
                "avg_abs_error_pct": group["abs_error_pct"].mean(),
                "median_abs_error_pct": group["abs_error_pct"].median(),
                "max_abs_error_pct": group["abs_error_pct"].max(),
                "weighted_error_pct": weighted_error_pct,
                "real_energy_sum_wh": real_sum,
                "mlp_energy_sum_wh": mlp_sum,
            }
        )

    section_summary = detail.groupby("section", sort=False).apply(summarize_group).reset_index()
    overall = summarize_group(detail)
    overall["section"] = "总计"
    average_summary = pd.concat([section_summary, overall.to_frame().T], ignore_index=True)

    numeric_cols = [col for col in average_summary.columns if col != "section"]
    for col in numeric_cols:
        average_summary[col] = pd.to_numeric(average_summary[col], errors="coerce").round(3)

    average_path = model_root / "triple_axis_mlp_average_summary.csv"
    average_summary.to_csv(average_path, index=False, encoding="utf-8-sig")

    class_path: Path | None = None
    if "run_class" in detail.columns:
        class_detail = detail.copy()
        class_detail["run_class"] = class_detail["run_class"].astype(str)
        class_summary = class_detail.groupby(["section", "run_class"], sort=False).apply(summarize_group).reset_index()
        for col in [c for c in class_summary.columns if c not in {"section", "run_class"}]:
            class_summary[col] = pd.to_numeric(class_summary[col], errors="coerce").round(3)
        class_path = model_root / "triple_axis_mlp_average_by_class.csv"
        class_summary.to_csv(class_path, index=False, encoding="utf-8-sig")

    if missing:
        pd.DataFrame(missing).to_csv(model_root / "triple_axis_mlp_average_missing.csv", index=False, encoding="utf-8-sig")

    return average_path, class_path


def main() -> int:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    model_root = resolve_path(args.model_root)

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return 2
    if not model_root.exists():
        print(f"模型目录不存在: {model_root}")
        return 2

    sections = select_sections(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("MLP triple-axis plotting only")
    print(f"Data:      {data_dir}")
    print(f"Model:     {model_root}")
    print(f"Device:    {device}")
    print(f"Sections:  {len(sections)}")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    if not args.summary_only:
        for idx, section in enumerate(sections, start=1):
            print(f"\n[{idx}/{len(sections)}] 绘制 MLP 三轴图: {section}")
            try:
                row = plot_one_section(section, data_dir, model_root, device, args)
                rows.append(row)
                print(f"  完成: {row['plot_count']} 张")
            except Exception as exc:
                skipped.append({"section": section, "reason": str(exc)})
                print(f"  跳过: {exc}")

        summary_path = model_root / "triple_axis_mlp_plot_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        if skipped:
            skipped_path = model_root / "triple_axis_mlp_plot_skipped.csv"
            pd.DataFrame(skipped).to_csv(skipped_path, index=False, encoding="utf-8-sig")
            print(f"\n跳过记录: {skipped_path}")

        print(f"\n汇总: {summary_path}")
        print(f"成功区间数: {len(rows)}")
        print(f"跳过区间数: {len(skipped)}")

    average_path, class_path = summarize_existing_triple_axis(model_root, sections)
    print(f"平均汇总: {average_path}")
    if class_path is not None:
        print(f"分等级平均汇总: {class_path}")

    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
