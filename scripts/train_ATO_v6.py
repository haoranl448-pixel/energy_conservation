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

import os
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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
PROJECT_ROOT = Path(r"D:\energy_conservation")

# 你的标准等级时间表位置
STANDARD_TIMES_CANDIDATES = [
    PROJECT_ROOT / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
]

# 你的清洗后的历史数据位置
DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "data_processed_new_v2"
]

# 模板工件保存位置
OUTPUT_ROOT = PROJECT_ROOT / "output" / "ato_phase_results"






# =========================
# 全局配置
# =========================
CLASS_COL = "运行等级"
REFERENCE_CLASS = "class3"
DT_SAMPLE = 0.05
END_DIST_TOL = 30.0
MIN_RUN_POINTS = 120

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



def normalize_class_label(x) -> Optional[str]:
    if pd.isna(x):
        return None
    return str(x).strip().lower()



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
    exact = data_dir / f"cleaned_{station_pair}.xlsx"
    if exact.exists():
        return [exact]

    pattern = str(data_dir / f"cleaned_{station_pair}*.xlsx")
    return [Path(x) for x in sorted(glob.glob(pattern))]



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

    files = resolve_input_files(data_dir, station_pair)
    if not files:
        return {
            "station_pair": station_pair,
            "status": "missing_file",
            "message": f"未找到 cleaned_{station_pair}.xlsx",
        }

    all_df = []
    for fp in files:
        df = pd.read_excel(fp)
        all_df.append(df)
    df = pd.concat(all_df, ignore_index=True)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return {
            "station_pair": station_pair,
            "status": "missing_columns",
            "message": f"缺少必要列: {missing_cols}",
        }

    df[CLASS_COL] = df[CLASS_COL].apply(normalize_class_label)
    df = df[df[CLASS_COL].isin(level_target_times.keys())].copy()
    if df.empty:
        return {
            "station_pair": station_pair,
            "status": "no_class_data",
            "message": "未识别到 class1-class5 数据",
        }

    run_id, run_id_cols_used = build_run_id(df)
    df["run_id"] = run_id.astype(str)

    df_ref = df[df[CLASS_COL] == REFERENCE_CLASS].copy()
    if df_ref.empty:
        return {
            "station_pair": station_pair,
            "status": "no_class3",
            "message": "未找到 class3 数据",
        }

    clean_runs = []
    summary_rows = []

    for rid, g in df_ref.groupby("run_id"):
        item = clean_single_run(g, dt_sample=DT_SAMPLE)
        if item is None:
            continue
        item["run_id"] = rid
        clean_runs.append(item)
        summary_rows.append({
            "run_id": rid,
            "travel_time_raw_s": round(item["travel_time_raw"], 4),
            "end_dist_m": round(item["end_dist"], 4),
            "mass_median_run": round(float(np.nanmedian(item["mass_values"])), 4),
            "n_points_raw": len(item["raw_t"]),
        })

    if len(clean_runs) == 0:
        return {
            "station_pair": station_pair,
            "status": "invalid_class3_runs",
            "message": "class3 清洗后无有效单趟",
        }

    pd.DataFrame(summary_rows).to_csv(output_dir / "class3_run_summary.csv", index=False, encoding="utf-8-sig")

    end_dists = np.array([x["end_dist"] for x in clean_runs], dtype=float)
    target_l = float(np.median(end_dists))
    valid_runs = [x for x in clean_runs if abs(x["end_dist"] - target_l) <= END_DIST_TOL]
    if len(valid_runs) == 0:
        return {
            "station_pair": station_pair,
            "status": "filtered_out",
            "message": "终点距离过滤后无有效 class3",
        }

    raw_times = np.array([x["travel_time_raw"] for x in valid_runs], dtype=float)
    time_ref_raw = float(np.median(raw_times))
    mass_median = float(np.nanmedian(np.concatenate([x["mass_values"] for x in valid_runs])))

    n_ref = int(round(time_ref_raw / DT_SAMPLE)) + 1
    n_ref = max(n_ref, 400)

    v_norm_curves = []
    for item in valid_runs:
        v_norm = resample_to_normalized_time(item["raw_v"], n_ref)
        if v_norm is not None:
            v_norm_curves.append(v_norm)

    if len(v_norm_curves) == 0:
        return {
            "station_pair": station_pair,
            "status": "resample_failed",
            "message": "class3 归一化时间重采样失败",
        }

    # v_norm_curves = np.vstack(v_norm_curves)
    # v_ref_t = np.median(v_norm_curves, axis=0)




# ======= 改进后的逻辑（选取最接近中位数的【真实班次】） =======
    v_norm_curves = np.vstack(v_norm_curves)
    median_curve = np.median(v_norm_curves, axis=0)
    
    # 1. 计算每个真实班次与中值曲线的“距离”（欧氏距离）
    distances = np.linalg.norm(v_norm_curves - median_curve, axis=1)
    
    # 2. 找到距离最小的那个班次的索引
    best_run_idx = np.argmin(distances)
    
    # 3. 提取这个最具有代表性的真实班次作为“参考线”
    v_ref_t = v_norm_curves[best_run_idx].copy()
    
    # 4. 为了消除传感器本身的毛刺，进行一次轻微平滑
    # 窗口可以开大一点，比如 31 或 51，确保加速度丝滑
    v_ref_t = safe_savgol(v_ref_t, window=51, poly=3)



    

    # v_ref_t = safe_savgol(v_ref_t, window=min(101, n_ref - (1 - n_ref % 2)), poly=3)
    v_ref_t = np.clip(v_ref_t, 0.0, None)
    v_ref_t[0] = 0.0
    v_ref_t[-1] = 0.0

    tau_ref = np.linspace(0.0, 1.0, n_ref)
    t_ref = tau_ref * time_ref_raw
    dt_ref = t_ref[1] - t_ref[0]
    idx_a, idx_b, v_peak_ref = detect_phase_boundaries(v_ref_t, dt_ref)

    v_acc_ref = v_ref_t[:idx_a + 1].copy()
    v_mid_ref = v_ref_t[idx_a:idx_b + 1].copy()
    v_br_ref = v_ref_t[idx_b:].copy()

    T_acc_ref = float(t_ref[idx_a] - t_ref[0])
    T_mid_ref = float(t_ref[idx_b] - t_ref[idx_a])
    T_br_ref = float(t_ref[-1] - t_ref[idx_b])

    s_ref_t = cumtrapz_uniform(v_ref_t, dt_ref)
    target_l_ref_curve = float(s_ref_t[-1])

    D_acc_ref = float(cumtrapz_uniform(v_acc_ref, dt_ref)[-1]) if len(v_acc_ref) > 1 else 0.0
    D_mid_ref = float(cumtrapz_uniform(v_mid_ref, dt_ref)[-1]) if len(v_mid_ref) > 1 else 0.0
    D_br_ref = float(cumtrapz_uniform(v_br_ref, dt_ref)[-1]) if len(v_br_ref) > 1 else 0.0

    artifacts = {
        "station_pair": station_pair,
        "source_files": [str(x) for x in files],
        "source_data_dir": str(data_dir),
        "reference_class": REFERENCE_CLASS,
        "class_col": CLASS_COL,
        "level_target_times": level_target_times,
        "dt_sample": DT_SAMPLE,
        "target_l": target_l,
        "end_dist_tol": END_DIST_TOL,
        "mass_median": mass_median,
        "time_ref_raw": time_ref_raw,
        "n_runs_class3": len(valid_runs),
        "run_id_cols_used": run_id_cols_used,
        "tau_ref": tau_ref,
        "t_ref": t_ref,
        "v_ref_t": v_ref_t,
        "s_ref_t": s_ref_t,
        "idx_acc_end": idx_a,
        "idx_brake_start": idx_b,
        "peak_speed_ref": float(v_peak_ref),
        "T_acc_ref": T_acc_ref,
        "T_mid_ref": T_mid_ref,
        "T_br_ref": T_br_ref,
        "D_acc_ref": D_acc_ref,
        "D_mid_ref": D_mid_ref,
        "D_br_ref": D_br_ref,
        "v_acc_ref": v_acc_ref,
        "v_mid_ref": v_mid_ref,
        "v_br_ref": v_br_ref,
    }

    with open(output_dir / "class3_phase_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    pd.DataFrame({
        "t_ref_s": t_ref,
        "v_ref_mps": v_ref_t,
        "v_ref_kmh": v_ref_t * 3.6,
        "s_ref_m": s_ref_t,
    }).to_csv(output_dir / "class3_reference_time_curve.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([{
        "station_pair": station_pair,
        "target_l_from_data_m": target_l,
        "class3_time_median_s": time_ref_raw,
        "class3_target_time_s": level_target_times["class3"],
        "class3_peak_speed_ref_mps": float(v_peak_ref),
        "class3_peak_speed_ref_kmh": float(v_peak_ref) * 3.6,
        "T_acc_ref_s": T_acc_ref,
        "T_mid_ref_s": T_mid_ref,
        "T_br_ref_s": T_br_ref,
        "D_acc_ref_m": D_acc_ref,
        "D_mid_ref_m": D_mid_ref,
        "D_br_ref_m": D_br_ref,
        "ref_curve_distance_m": target_l_ref_curve,
        "mass_median": mass_median,
        "n_runs_class3": len(valid_runs),
    }]).to_csv(output_dir / "class3_phase_summary.csv", index=False, encoding="utf-8-sig")

    plot_class3_all_runs(valid_runs, v_ref_t, t_ref, idx_a, idx_b, output_dir)
    plot_class3_reference_vs_distance(t_ref, v_ref_t, output_dir)

    return {
        "station_pair": station_pair,
        "status": "success",
        "message": "训练完成",
        "target_l_m": round(target_l, 4),
        "class3_time_median_s": round(time_ref_raw, 4),
        "class3_target_time_s": round(level_target_times["class3"], 4),
        "class3_peak_speed_ref_kmh": round(float(v_peak_ref) * 3.6, 4),
        "n_runs_class3": len(valid_runs),
        "output_dir": str(output_dir),
    }


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUTPUT_ROOT)

    standard_times_path = resolve_existing_file(STANDARD_TIMES_CANDIDATES, "standard_class_times.csv")
    data_dir = resolve_existing_dir(DATA_DIR_CANDIDATES, "数据")

    table = load_standard_times(standard_times_path)
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
            print(f"  ✅ 成功 | class3参考时间 = {result['class3_time_median_s']} s | 样本数 = {result['n_runs_class3']}")
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
