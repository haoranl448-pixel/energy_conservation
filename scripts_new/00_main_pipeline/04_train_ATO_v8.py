# -*- coding: utf-8 -*-
"""
batch_train_class3_phase_templates.py

用途：
1. 从 train_ATO/standard_class_times.csv 读取区间级标准等级时间表；
2. 对时间表中的每个区间，读取 cleaned_{区间}.xlsx；
3. 仅提取 class3 真实样本；
4. 构建 class3 的三相位模板（加速 / 中段 / 制动）；
5. 为后续 simulate 脚本保存统一工件。

说明：
- 本脚本已经从“单区间”扩展为“批量区间”；
- 只要 standard_class_times.csv 中有 26 行，就会一次性训练 26 个区间；
- 若某区间数据文件不存在或 class3 样本不足，会在批量汇总中标记并跳过。
"""

from __future__ import annotations
from scipy.interpolate import make_interp_spline
import os
import glob
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

PLOT_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["font.sans-serif"] = PLOT_FONT_CANDIDATES
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 路径配置（相对脚本位置）
# =========================
# BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = BASE_DIR.parent

# STANDARD_TIMES_CANDIDATES = [
#     BASE_DIR / "standard_class_times.csv",
#     PROJECT_ROOT / "train_ATO" / "standard_class_times.csv",
# ]

# DATA_DIR_CANDIDATES = [
#     PROJECT_ROOT / "data_processed_new_v2",
#     PROJECT_ROOT / "data_processed",
#     BASE_DIR / "data_processed_new_v2",
#     BASE_DIR / "data_processed",
# ]

# OUTPUT_ROOT = BASE_DIR / "ato_phase_results"



# =========================
# 路径配置（已适配 D:\energy_conservation）
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_data_dir_candidates(defaults: List[Path]) -> List[Path]:
    """Use the ATO data directory selected by the main pipeline."""

    raw = (
        os.environ.get("ENERGY_ATO_DATA_DIR")
        or os.environ.get("ENERGY_RESULTS_DATA_DIR")
        or os.environ.get("ENERGY_DATA_DIR")
    )
    if not raw:
        return defaults
    data_dir = Path(raw)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return [data_dir, *defaults]

# 你的标准等级时间表位置
STANDARD_TIMES_CANDIDATES = [
    PROJECT_ROOT / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
]

# 你的清洗后的历史数据位置
DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "data_processed_new_v2"
]
DATA_DIR_CANDIDATES = get_data_dir_candidates(DATA_DIR_CANDIDATES)

# 模板工件保存位置
OUTPUT_ROOT = PROJECT_ROOT / "output" / "ato_phase_results_v3"






# =========================
# 全局配置
# =========================
CLASS_COL = "运行等级"
QUALITY_LABEL_COL = "曲线质量标签"
NORMAL_QUALITY_LABEL = 0
REFERENCE_CLASS = "class3"
DT_SAMPLE = 0.05
END_DIST_TOL = 30.0
MIN_RUN_POINTS = 120
REAL_STRONG_MIN_SAMPLES = 3

RUN_ID_CANDIDATE_COLS = [
    "日期+服务号",
    "服务号",
    "车底号",
    "列车运行方向",
]

REQUIRED_COLS = [
    "时刻",
    "累计位移(m)",
    "速度(m/s)",
    "curvature",
    "gradient",
    "重量",
    CLASS_COL,
]

TIME_TABLE_REQUIRED_COLS = ["区段", "Class1", "Class2", "Class3", "Class4", "Class5"]


# =========================
# 通用工具
# =========================
def sanitize_name(name: str) -> str:
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    out = str(name)
    for ch in bad:
        out = out.replace(ch, "_")
    return out



def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def reset_output_root(path: Path):
    """Clear previous ATO template outputs before a fresh batch run."""

    resolved = path.resolve()
    output_root = (PROJECT_ROOT / "output").resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"拒绝清理非 output 目录: {resolved}") from exc

    if resolved.exists():
        for child in resolved.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)
    print(f"已清理 ATO 模板输出目录: {resolved}")



def normalize_class_label(x) -> Optional[str]:
    if pd.isna(x):
        return None
    text = str(x).strip().lower()
    if text.startswith("class"):
        suffix = text[5:].strip()
        if suffix.replace(".", "", 1).isdigit():
            value = float(suffix)
            if value.is_integer():
                return f"class{int(value)}"
        return text
    if text.replace(".", "", 1).isdigit():
        value = float(text)
        if value.is_integer():
            return f"class{int(value)}"
    return text



def safe_savgol(y, window=11, poly=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y.copy()

    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        return y.copy()
    if window % 2 == 0:
        window -= 1
    if window <= poly:
        return y.copy()

    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")



def build_order_key(series: pd.Series) -> np.ndarray:
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().sum() >= max(5, int(0.8 * len(series))):
        return s_num.ffill().bfill().values

    s_dt = pd.to_datetime(series, errors="coerce")
    if s_dt.notna().sum() >= max(5, int(0.8 * len(series))):
        return s_dt.astype("int64").values

    return np.arange(len(series), dtype=float)



def resolve_existing_file(candidates: List[Path], desc: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到{desc}，候选路径：{[str(x) for x in candidates]}")



def resolve_existing_dir(candidates: List[Path], desc: str) -> Path:
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"未找到{desc}目录，候选路径：{[str(x) for x in candidates]}")



def resolve_input_files(data_dir: Path, station_pair: str) -> List[Path]:
    for prefix in ("results", "cleaned"):
        exact = data_dir / f"{prefix}_{station_pair}.xlsx"
        if exact.exists():
            return [exact]

    matches: List[Path] = []
    for prefix in ("results", "cleaned"):
        pattern = str(data_dir / f"{prefix}_{station_pair}*.xlsx")
        matches.extend(Path(x) for x in sorted(glob.glob(pattern)))
    return matches


def station_pairs_available_in_data_dir(data_dir: Path) -> set[str]:
    """Return station pairs that have results_*.xlsx or cleaned_*.xlsx in data_dir."""

    pairs: set[str] = set()
    for prefix in ("results_", "cleaned_"):
        for fp in data_dir.glob(f"{prefix}*.xlsx"):
            pairs.add(fp.stem[len(prefix):])
    return pairs


def filter_table_by_available_data(table: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """When main selects an ATO data directory, limit processing to sections present there."""

    if not (
        os.environ.get("ENERGY_ATO_DATA_DIR")
        or os.environ.get("ENERGY_RESULTS_DATA_DIR")
        or os.environ.get("ENERGY_DATA_DIR")
    ):
        return table
    available = station_pairs_available_in_data_dir(data_dir)
    filtered = table[table["区段"].isin(available)].copy()
    if filtered.empty:
        raise FileNotFoundError(f"数据目录 {data_dir} 中没有和标准时间表匹配的 results_/cleaned_ 文件。")
    skipped = len(table) - len(filtered)
    print(f"按测试数据目录筛选区间: {len(filtered)} 个，跳过标准时间表中未提供数据的 {skipped} 个区间。")
    return filtered


def filter_normal_quality_runs(df: pd.DataFrame) -> pd.DataFrame:
    """ATO templates only use normal daytime runs when 曲线质量标签 is available."""

    if QUALITY_LABEL_COL not in df.columns:
        return df
    labels = pd.to_numeric(df[QUALITY_LABEL_COL], errors="coerce")
    before_rows = len(df)
    before_runs = df["segment"].nunique() if "segment" in df.columns else np.nan
    filtered = df[labels == NORMAL_QUALITY_LABEL].copy()
    after_runs = filtered["segment"].nunique() if "segment" in filtered.columns else np.nan
    print(
        f"   曲线质量过滤: 仅保留 {QUALITY_LABEL_COL}=0，"
        f"行数 {before_rows}->{len(filtered)}，趟次 {before_runs}->{after_runs}"
    )
    return filtered



def build_run_id(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    if "run_id" in df.columns:
        rid = df["run_id"].astype(str).str.strip()
        if rid.nunique() > 1:
            return rid, ["run_id"]

    available = [c for c in RUN_ID_CANDIDATE_COLS if c in df.columns]
    if len(available) > 0:
        tmp = df[available].copy()
        for c in available:
            tmp[c] = tmp[c].fillna("NA").astype(str).str.strip()
        rid = tmp.agg("|".join, axis=1)
        if rid.nunique() > 1:
            return rid, available

    order_key = build_order_key(df["时刻"])
    dist = pd.to_numeric(df["累计位移(m)"], errors="coerce").ffill().fillna(0.0).values
    dt_jump = np.diff(order_key, prepend=order_key[0])
    ds_jump = np.diff(dist, prepend=dist[0])

    new_run = (dt_jump < 0) | (ds_jump < -1.0)
    run_id = np.cumsum(new_run).astype(str)
    return pd.Series(run_id, index=df.index), ["fallback_time_dist_reset"]



def cumtrapz_uniform(y, dt: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out



def clean_single_run(g: pd.DataFrame, dt_sample=0.05) -> Optional[Dict]:
    g = g.copy().reset_index(drop=True)
    g["__order__"] = build_order_key(g["时刻"])
    g["__orig_idx__"] = np.arange(len(g))

    for c in ["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g = g.dropna(subset=["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"])
    if len(g) < MIN_RUN_POINTS:
        return None

    g = g.sort_values(["__order__", "__orig_idx__"]).reset_index(drop=True)

    raw_v = g["速度(m/s)"].clip(lower=0).values.astype(float)
    raw_t = np.arange(len(raw_v), dtype=float) * dt_sample

    s = g["累计位移(m)"].values.astype(float)
    s = s - s[0]
    s = np.maximum.accumulate(s)

    end_dist = float(s[-1])
    if end_dist < 200:
        return None

    return {
        "raw_t": raw_t,
        "raw_v": raw_v,
        "raw_s": s,
        "travel_time_raw": float(raw_t[-1]) if len(raw_t) > 0 else 0.0,
        "end_dist": end_dist,
        "mass_values": g["重量"].values.astype(float),
        "curv_values": g["curvature"].values.astype(float),
        "grad_values": g["gradient"].values.astype(float),
    }



def resample_to_normalized_time(raw_v: np.ndarray, n_ref: int) -> Optional[np.ndarray]:
    if len(raw_v) < 2:
        return None
    tau_src = np.linspace(0.0, 1.0, len(raw_v))
    tau_dst = np.linspace(0.0, 1.0, n_ref)
    return np.interp(tau_dst, tau_src, raw_v)


def smooth_template_curve(v: np.ndarray, window: int = 15, poly: int = 3) -> np.ndarray:
    """Lightly smooth and lock endpoints for a template speed curve."""

    out = np.asarray(v, dtype=float).copy()
    out = np.clip(out, 0.0, None)
    if len(out) > 0:
        out[0] = 0.0
        out[-1] = 0.0
    out = safe_savgol(out, window=window, poly=poly)
    out = np.clip(out, 0.0, None)
    if len(out) > 0:
        out[0] = 0.0
        out[-1] = 0.0
    return out


def build_median_center_curve(v_norm_curves: np.ndarray) -> np.ndarray:
    """Build the robust center line used only for selecting the real medoid run."""

    raw_median = np.median(v_norm_curves, axis=0)
    tau = np.linspace(0.0, 1.0, len(raw_median))
    try:
        spline = make_interp_spline(tau, raw_median, k=3)
        center = spline(tau)
    except Exception:
        center = raw_median
    return smooth_template_curve(center, window=31, poly=3)


def select_medoid_run(norm_items: List[Tuple[Dict, np.ndarray]], center_v: np.ndarray) -> Tuple[Dict, np.ndarray, float]:
    """Select the real run whose normalized speed shape is closest to the median center."""

    best_item = None
    best_v = None
    best_rmse = float("inf")

    for item, v_norm in norm_items:
        diff = np.asarray(v_norm, dtype=float) - np.asarray(center_v, dtype=float)
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        if rmse < best_rmse:
            best_item = item
            best_v = v_norm
            best_rmse = rmse

    if best_item is None or best_v is None:
        raise ValueError("无法从有效曲线中选择 medoid 代表曲线。")
    return best_item, best_v, best_rmse



def detect_phase_boundaries(v_ref_t: np.ndarray, dt_ref: float):
    v = safe_savgol(v_ref_t, window=min(101, len(v_ref_t) - (1 - len(v_ref_t) % 2)), poly=3)
    v = np.clip(v, 0.0, None)
    v_peak = float(np.max(v))

    idx_a = None
    idx_b = None

    for ratio in [0.97, 0.965, 0.96, 0.95, 0.94, 0.93, 0.92]:
        mask = v >= ratio * v_peak
        idxs = np.where(mask)[0]
        if len(idxs) >= max(20, int(1.0 / max(dt_ref, 1e-6))):
            idx_a = int(idxs[0])
            idx_b = int(idxs[-1])
            break

    if idx_a is None or idx_b is None or idx_b <= idx_a:
        n = len(v)
        idx_a = int(0.28 * n)
        idx_b = int(0.72 * n)

    if idx_a < 10:
        idx_a = 10
    if idx_b > len(v) - 11:
        idx_b = len(v) - 11
    if idx_b <= idx_a + 10:
        idx_b = min(len(v) - 11, idx_a + 20)

    return idx_a, idx_b, v_peak



def plot_class3_all_runs(raw_runs: List[Dict], v_ref_t: np.ndarray, t_ref: np.ndarray,
                         idx_a: int, idx_b: int, output_dir: Path):
    plt.figure(figsize=(13, 7))
    for item in raw_runs:
        plt.plot(item["raw_t"], item["raw_v"] * 3.6, color="#95a5a6", alpha=0.18, linewidth=1.0)

    plt.plot(t_ref, v_ref_t * 3.6, color="#d62728", linewidth=2.8, label="class3 reference")
    plt.axvline(t_ref[idx_a], color="green", linestyle="--", alpha=0.8, label="acc end")
    plt.axvline(t_ref[idx_b], color="blue", linestyle="--", alpha=0.8, label="brake start")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/h)")
    plt.title("Class3 reference in time domain")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "class3_reference_vt.png", dpi=260, bbox_inches="tight")
    plt.close()



def plot_class3_reference_vs_distance(t_ref: np.ndarray, v_ref_t: np.ndarray, output_dir: Path):
    s_ref = cumtrapz_uniform(v_ref_t, t_ref[1] - t_ref[0])
    plt.figure(figsize=(13, 7))
    plt.plot(s_ref, v_ref_t, color="#d62728", linewidth=2.8)
    plt.xlabel("Distance (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Class3 reference speed-distance curve")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / "class3_reference_vs.png", dpi=260, bbox_inches="tight")
    plt.close()



def load_standard_times(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in TIME_TABLE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"standard_class_times.csv 缺少字段：{missing}")

    df = df[TIME_TABLE_REQUIRED_COLS].copy()
    df["区段"] = df["区段"].astype(str).str.strip()
    for c in ["Class1", "Class2", "Class3", "Class4", "Class5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["区段"])



def row_to_level_target_times(row: pd.Series) -> Dict[str, float]:
    return {
        "class1": float(row["Class1"]),
        "class2": float(row["Class2"]),
        "class3": float(row["Class3"]),
        "class4": float(row["Class4"]),
        "class5": float(row["Class5"]),
    }


# =========================
# 单区间训练
# =========================
def train_one_station_pair(station_pair: str, level_target_times: Dict[str, float], data_dir: Path) -> Dict:
    output_dir = OUTPUT_ROOT / sanitize_name(station_pair)
    ensure_dir(output_dir)

    # 1. 读取数据
    files = resolve_input_files(data_dir, station_pair)
    if not files:
        return {"station_pair": station_pair, "status": "missing_file", "message": f"未找到 results_/cleaned_{station_pair}.xlsx"}

    all_df = []
    for fp in files:
        all_df.append(pd.read_excel(fp))
    df = pd.concat(all_df, ignore_index=True)
    source_files = [str(fp) for fp in files]
    df = filter_normal_quality_runs(df)
    if df.empty:
        return {"station_pair": station_pair, "status": "no_quality_zero", "message": f"{QUALITY_LABEL_COL}=0 的正常曲线为空"}

    # New processed files may carry the runtime class in a final column named "class".
    # Keep the original downstream column name as the canonical one.
    if CLASS_COL not in df.columns and "class" in df.columns:
        df[CLASS_COL] = df["class"]

    # 2. 预检查
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return {"station_pair": station_pair, "status": "missing_columns", "message": f"缺少必要列: {missing_cols}"}

    df[CLASS_COL] = df[CLASS_COL].apply(normalize_class_label)

    # 3. 识别当前区间包含的所有历史等级
    available_classes = [c for c in ["class1", "class2", "class3", "class4", "class5"]
                         if c in df[CLASS_COL].unique()]

    if not available_classes:
        return {"station_pair": station_pair, "status": "no_class_data", "message": "未识别到任何 class 数据"}

    # 用于存储所有等级基因的大字典
    multi_artifacts = {}
    summary_stats = []

    print(f"   📊 区间 {station_pair} 发现等级: {available_classes}")

    # 4. 循环处理每一个等级
    for current_cls in available_classes:
        df_ref = df[df[CLASS_COL] == current_cls].copy()

        run_id, _ = build_run_id(df_ref)
        df_ref["run_id"] = run_id.astype(str)

        clean_runs = []
        for rid, g in df_ref.groupby("run_id"):
            item = clean_single_run(g, dt_sample=DT_SAMPLE)
            if item is not None:
                item["run_id"] = rid
                # 记录标签方便后续画图
                item["class_label"] = current_cls
                clean_runs.append(item)

        if len(clean_runs) == 0:
            continue

        # 计算该等级的基准指标
        end_dists = np.array([x["end_dist"] for x in clean_runs], dtype=float)
        target_l = float(np.median(end_dists))
        valid_runs = [x for x in clean_runs if abs(x["end_dist"] - target_l) <= END_DIST_TOL]

        if len(valid_runs) == 0:
            continue

        raw_times = np.array([x["travel_time_raw"] for x in valid_runs], dtype=float)
        center_time_raw = float(np.median(raw_times))
        mass_median = float(np.nanmedian(np.concatenate([x["mass_values"] for x in valid_runs])))

        # 归一化重采样：同一区间、同一等级内部对齐，不同等级不会混在一起。
        n_ref = max(int(round(center_time_raw / DT_SAMPLE)) + 1, 400)
        norm_items = []
        for item in valid_runs:
            v_norm = resample_to_normalized_time(item["raw_v"], n_ref)
            if v_norm is not None:
                norm_items.append((item, v_norm))

        if len(norm_items) == 0: continue
        v_norm_curves = np.vstack([x[1] for x in norm_items])

        # 先构建稳健中心线，再选择最接近中心线的一条真实历史曲线作为最终模板。
        center_v_ref_t = build_median_center_curve(v_norm_curves)
        medoid_run, medoid_v_norm, medoid_rmse = select_medoid_run(norm_items, center_v_ref_t)
        v_ref_t = smooth_template_curve(medoid_v_norm, window=15, poly=3)
        # Keep the class median time as the template time base; the medoid run supplies the real speed shape.
        time_ref_raw = center_time_raw

        center_t_ref = np.linspace(0.0, 1.0, n_ref) * center_time_raw

        # 相位识别
        t_ref = np.linspace(0.0, 1.0, n_ref) * time_ref_raw
        dt_ref = t_ref[1] - t_ref[0]
        idx_a, idx_b, v_peak_ref = detect_phase_boundaries(v_ref_t, dt_ref)

        # 🌟 核心修正：确保所有字段都存入字典 🌟
        current_art = {
            "class_name": current_cls,
            "curve_source": "real",
            "parent_ref": current_cls,
            "supporting_refs": current_cls,
            "real_sample_count": int(len(valid_runs)),
            "sample_reliability": "strong" if len(valid_runs) >= REAL_STRONG_MIN_SAMPLES else "weak",
            "template_source": "medoid_real_run",
            "template_run_id": medoid_run.get("run_id", ""),
            "template_distance_to_median_rmse_mps": float(medoid_rmse),
            "template_run_time_raw": float(medoid_run["travel_time_raw"]),
            "center_time_median_s": center_time_raw,
            "center_v_ref_t": center_v_ref_t,
            "center_t_ref": center_t_ref,
            "source_files": source_files,
            "dt_sample": DT_SAMPLE,
            "end_dist_tol": END_DIST_TOL,
            "v_acc_ref": v_ref_t[:idx_a + 1].copy(),
            "v_mid_ref": v_ref_t[idx_a:idx_b + 1].copy(),
            "v_br_ref": v_ref_t[idx_b:].copy(),
            "T_acc_ref": float(t_ref[idx_a] - t_ref[0]),
            "T_mid_ref": float(t_ref[idx_b] - t_ref[idx_a]),
            "T_br_ref": float(t_ref[-1] - t_ref[idx_b]),
            "target_l": target_l,
            "peak_speed_ref": float(v_peak_ref),
            "time_ref_raw": time_ref_raw,
            "v_ref_t": v_ref_t,
            "t_ref": t_ref,
            "mass_median": mass_median,
            "medoid_run_sample": medoid_run,
            "valid_runs_samples": valid_runs  # 👈 修正：把样本存进去供下方绘图使用
        }

        multi_artifacts[current_cls] = current_art
        summary_stats.append({
            "class": current_cls,
            "curve_source": "real",
            "parent_ref": current_cls,
            "supporting_refs": current_cls,
            "real_sample_count": int(len(valid_runs)),
            "sample_reliability": "strong" if len(valid_runs) >= REAL_STRONG_MIN_SAMPLES else "weak",
            "template_source": "medoid_real_run",
            "template_run_id": medoid_run.get("run_id", ""),
            "template_distance_to_median_rmse_mps": round(float(medoid_rmse), 6),
            "center_time_median_s": round(center_time_raw, 2),
            "time_s": round(time_ref_raw, 2),
            "samples": len(valid_runs),
            "peak_v_kmh": round(v_peak_ref * 3.6, 2)
        })

    # 5. 最终保存与输出
    if not multi_artifacts:
        return {"station_pair": station_pair, "status": "failed", "message": "各等级处理均失败"}

    # 保存包含所有等级的大工件
    with open(output_dir / "multi_class_phase_artifacts.pkl", "wb") as f:
        pickle.dump(multi_artifacts, f)

    # 导出汇总表
    pd.DataFrame(summary_stats).to_csv(output_dir / "multi_class_summary.csv", index=False)

    # ======= 🌟 绘图逻辑：遍历 multi_artifacts 里的每一个等级 🌟 =======
    for cls_name, art_data in multi_artifacts.items():
        plt.figure(figsize=(10, 6))

        # 1. 绘制历史样本背景 (灰色线)
        # 注意：这里直接从刚才存入的字典里拿样本数据
        for run in art_data["valid_runs_samples"]:
            plt.plot(run["raw_t"], run["raw_v"] * 3.6, color="#95a5a6", alpha=0.15, linewidth=1)

        if "center_v_ref_t" in art_data and "center_t_ref" in art_data:
            plt.plot(art_data["center_t_ref"], art_data["center_v_ref_t"] * 3.6,
                     label=f"Median center {cls_name}", color="#2874a6", linewidth=2.0, alpha=0.9)

        # 2. 绘制最终采用的真实代表模板线 (红线)
        plt.plot(art_data["t_ref"], art_data["v_ref_t"] * 3.6,
                 label=f"Template {cls_name} (medoid real run)", color='red', linewidth=2.5)

        standard_time = level_target_times.get(cls_name)
        template_median_time = art_data.get("center_time_median_s", art_data.get("time_ref_raw"))
        standard_label = f"standard={standard_time:.2f}s" if standard_time is not None else "standard=n/a"
        template_label = (
            f"template median={template_median_time:.2f}s"
            if template_median_time is not None else "template median=n/a"
        )
        plt.title(
            f"{station_pair} - {cls_name} Reference Curve | {standard_label} | {template_label}\n"
            f"run={art_data.get('template_run_id', '')}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (km/h)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图片
        plt.savefig(output_dir / f"{cls_name}_reference_vt.png", dpi=150)
        plt.close()

    return {
        "station_pair": station_pair,
        "status": "success",
        "message": f"完成等级提取: {list(multi_artifacts.keys())}",
        "output_dir": str(output_dir)
    }
# =========================
# 主流程
# =========================
def main():
    reset_output_root(OUTPUT_ROOT)

    standard_times_path = resolve_existing_file(STANDARD_TIMES_CANDIDATES, "standard_class_times.csv")
    data_dir = resolve_existing_dir(DATA_DIR_CANDIDATES, "数据")

    table = load_standard_times(standard_times_path)
    table = filter_table_by_available_data(table, data_dir)
    batch_rows = []

    print("=" * 72)
    print("批量训练 class3 三相位模板")
    print(f"时间表: {standard_times_path}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"待处理区间数: {len(table)}")
    print("=" * 72)

    for i, row in table.iterrows():
        station_pair = str(row["区段"]).strip()
        level_target_times = row_to_level_target_times(row)

        print(f"\n[{i + 1}/{len(table)}] 开始训练区间: {station_pair}")
        result = train_one_station_pair(station_pair, level_target_times, data_dir)
        batch_rows.append(result)

        if result["status"] == "success":

            #print(f"  ✅ 成功 | class3参考时间 = {result['class3_time_median_s']} s | 样本数 = {result['n_runs_class3']}")
            found_clss = list(pickle.load(open(result["output_dir"] + "/multi_class_phase_artifacts.pkl", "rb")).keys())
            print(f"  ✅ 成功 | 提取等级: {found_clss}")
        else:
            print(f"  ❌ 失败 | {result['status']} | {result['message']}")

    df_batch = pd.DataFrame(batch_rows)
    df_batch.to_csv(OUTPUT_ROOT / "batch_train_summary.csv", index=False, encoding="utf-8-sig")

    n_success = int((df_batch["status"] == "success").sum()) if not df_batch.empty else 0
    n_fail = len(df_batch) - n_success

    print("\n" + "=" * 72)
    print("批量训练结束")
    print(f"成功区间数: {n_success}")
    print(f"失败区间数: {n_fail}")
    print(f"汇总文件: {OUTPUT_ROOT / 'batch_train_summary.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
