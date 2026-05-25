# -*- coding: utf-8 -*-
"""
train_ATO_v9.py

Extract class2 phase templates from historical data.
class3 is also extracted for validation comparison in simulate_ATO_v9.py.

Key difference from v8: focuses on class2 as the parent template source,
saving class3 reference data alongside for R-squared validation.
"""
from __future__ import annotations
from scipy.interpolate import make_interp_spline
import glob, pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =========================
# Path config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STANDARD_TIMES_CANDIDATES = [
    PROJECT_ROOT / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
]
DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "data_processed_new_v2"
]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "ato_phase_results_v9"

# =========================
# Global config
# =========================
CLASS_COL = "运行等级"
DT_SAMPLE = 0.05
END_DIST_TOL = 30.0
MIN_RUN_POINTS = 120

RUN_ID_CANDIDATE_COLS = ["日期+服务号", "服务号", "车底号", "列车运行方向"]
REQUIRED_COLS = ["时刻", "累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量", CLASS_COL]
TIME_TABLE_REQUIRED_COLS = ["区段", "Class1", "Class2", "Class3", "Class4", "Class5"]

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Utility functions
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
    if pd.isna(x): return None
    return str(x).strip().lower()

def safe_savgol(y, window=11, poly=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5: return y.copy()
    if window >= n: window = n - 1 if n % 2 == 0 else n
    if window < 5: return y.copy()
    if window % 2 == 0: window -= 1
    if window <= poly: poly = window - 1
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
        if p.exists(): return p
    raise FileNotFoundError(f"未找到{desc}")

def resolve_existing_dir(candidates: List[Path], desc: str) -> Path:
    for p in candidates:
        if p.exists() and p.is_dir(): return p
    raise FileNotFoundError(f"未找到{desc}目录")

def resolve_input_files(data_dir: Path, station_pair: str) -> List[Path]:
    exact = data_dir / f"cleaned_{station_pair}.xlsx"
    if exact.exists(): return [exact]
    pattern = str(data_dir / f"cleaned_{station_pair}*.xlsx")
    return [Path(x) for x in sorted(glob.glob(pattern))]

def build_run_id(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    if "run_id" in df.columns:
        rid = df["run_id"].astype(str).str.strip()
        if rid.nunique() > 1: return rid, ["run_id"]
    available = [c for c in RUN_ID_CANDIDATE_COLS if c in df.columns]
    if available:
        tmp = df[available].copy()
        for c in available:
            tmp[c] = tmp[c].fillna("NA").astype(str).str.strip()
        rid = tmp.iloc[:, 0].astype(str) if len(available) == 1 else tmp.agg("|".join, axis=1)
        if isinstance(rid, pd.DataFrame):
            rid = rid.iloc[:, 0]
        if rid.nunique() > 1: return rid, available
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
    if len(y) >= 2: out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out

def clean_single_run(g: pd.DataFrame, dt_sample=0.05) -> Optional[Dict]:
    g = g.copy().reset_index(drop=True)
    g["__order__"] = build_order_key(g["时刻"])
    g["__orig_idx__"] = np.arange(len(g))
    for c in ["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"])
    if len(g) < MIN_RUN_POINTS: return None
    g = g.sort_values(["__order__", "__orig_idx__"]).reset_index(drop=True)
    raw_v = g["速度(m/s)"].clip(lower=0).values.astype(float)
    raw_t = np.arange(len(raw_v), dtype=float) * dt_sample
    s = g["累计位移(m)"].values.astype(float)
    s = s - s[0]; s = np.maximum.accumulate(s)
    end_dist = float(s[-1])
    if end_dist < 200: return None
    return {
        "raw_t": raw_t, "raw_v": raw_v, "raw_s": s,
        "travel_time_raw": float(raw_t[-1]) if len(raw_t) > 0 else 0.0,
        "end_dist": end_dist,
        "mass_values": g["重量"].values.astype(float),
        "curv_values": g["curvature"].values.astype(float),
        "grad_values": g["gradient"].values.astype(float),
    }

def resample_to_normalized_time(raw_v: np.ndarray, n_ref: int) -> Optional[np.ndarray]:
    if len(raw_v) < 2: return None
    tau_src = np.linspace(0.0, 1.0, len(raw_v))
    tau_dst = np.linspace(0.0, 1.0, n_ref)
    return np.interp(tau_dst, tau_src, raw_v)

def detect_phase_boundaries(v_ref_t: np.ndarray, dt_ref: float):
    v = safe_savgol(v_ref_t, window=min(101, len(v_ref_t) - (1 - len(v_ref_t) % 2)), poly=3)
    v = np.clip(v, 0.0, None)
    v_peak = float(np.max(v))
    idx_a, idx_b = None, None
    for ratio in [0.97, 0.965, 0.96, 0.95, 0.94, 0.93, 0.92]:
        mask = v >= ratio * v_peak
        idxs = np.where(mask)[0]
        if len(idxs) >= max(20, int(1.0 / max(dt_ref, 1e-6))):
            idx_a, idx_b = int(idxs[0]), int(idxs[-1])
            break
    if idx_a is None or idx_b is None or idx_b <= idx_a:
        n = len(v); idx_a = int(0.28 * n); idx_b = int(0.72 * n)
    if idx_a < 10: idx_a = 10
    if idx_b > len(v) - 11: idx_b = len(v) - 11
    if idx_b <= idx_a + 10: idx_b = min(len(v) - 11, idx_a + 20)
    return idx_a, idx_b, v_peak

def load_standard_times(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in TIME_TABLE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"standard_class_times.csv missing: {missing}")
    df = df[TIME_TABLE_REQUIRED_COLS].copy()
    df["区段"] = df["区段"].astype(str).str.strip()
    for c in ["Class1", "Class2", "Class3", "Class4", "Class5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["区段"])

# =========================
# Core: extract class template
# =========================
def extract_class_template(df, class_name: str, dt_sample=0.05) -> Optional[Dict]:
    """Extract phase template for a single class from raw data."""
    df_ref = df[df[CLASS_COL] == class_name].copy()
    if len(df_ref) < MIN_RUN_POINTS:
        return None
    run_id, _ = build_run_id(df_ref)
    df_ref["run_id"] = run_id.astype(str)

    clean_runs = []
    for rid, g in df_ref.groupby("run_id"):
        item = clean_single_run(g, dt_sample=dt_sample)
        if item is not None:
            item["run_id"] = rid
            clean_runs.append(item)

    if len(clean_runs) == 0: return None

    end_dists = np.array([x["end_dist"] for x in clean_runs], dtype=float)
    target_l = float(np.median(end_dists))
    valid_runs = [x for x in clean_runs if abs(x["end_dist"] - target_l) <= END_DIST_TOL]
    if len(valid_runs) == 0: return None

    raw_times = np.array([x["travel_time_raw"] for x in valid_runs], dtype=float)
    time_ref_raw = float(np.median(raw_times))
    mass_median = float(np.nanmedian(np.concatenate([x["mass_values"] for x in valid_runs])))

    n_ref = max(int(round(time_ref_raw / dt_sample)) + 1, 400)
    v_norm_curves = []
    for item in valid_runs:
        v_norm = resample_to_normalized_time(item["raw_v"], n_ref)
        if v_norm is not None: v_norm_curves.append(v_norm)
    if len(v_norm_curves) == 0: return None
    v_norm_curves = np.vstack(v_norm_curves)

    # B-spline median
    raw_median = np.median(v_norm_curves, axis=0)
    tau = np.linspace(0.0, 1.0, len(raw_median))
    spline = make_interp_spline(tau, raw_median, k=3)
    v_ref_t = spline(tau)
    v_ref_t = np.clip(v_ref_t, 0.0, None)
    v_ref_t[0], v_ref_t[-1] = 0.0, 0.0
    v_ref_t = safe_savgol(v_ref_t, window=31, poly=3)
    v_ref_t = np.clip(v_ref_t, 0.0, None)
    v_ref_t[0], v_ref_t[-1] = 0.0, 0.0

    t_ref = np.linspace(0.0, 1.0, n_ref) * time_ref_raw
    dt_ref = t_ref[1] - t_ref[0]
    idx_a, idx_b, v_peak_ref = detect_phase_boundaries(v_ref_t, dt_ref)

    return {
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
        "valid_runs": valid_runs,
        "dt_sample": dt_sample,
        "end_dist_tol": END_DIST_TOL,
        "source_files": [],
    }

# =========================
# Single station training
# =========================
def train_one_station(station_pair: str, data_dir: Path) -> Dict:
    output_dir = OUTPUT_ROOT / sanitize_name(station_pair)
    ensure_dir(output_dir)

    files = resolve_input_files(data_dir, station_pair)
    if not files:
        return {"station_pair": station_pair, "status": "missing_file"}

    all_df = []
    for fp in files:
        all_df.append(pd.read_excel(fp))
    df = pd.concat(all_df, ignore_index=True)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return {"station_pair": station_pair, "status": "missing_columns", "message": str(missing_cols)}

    df[CLASS_COL] = df[CLASS_COL].apply(normalize_class_label)

    # Extract class2 (the parent template source for v9 experiment)
    class2_art = extract_class_template(df, "class2")
    if class2_art is None:
        return {"station_pair": station_pair, "status": "no_class2"}

    # Also extract class3 (for validation comparison)
    class3_art = extract_class_template(df, "class3")

    # Save artifacts
    artifacts = {"class2": class2_art}
    if class3_art is not None:
        artifacts["class3"] = class3_art

    with open(output_dir / "class2_phase_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    # Plot class2 reference
    plt.figure(figsize=(10, 6))
    for run in class2_art["valid_runs"]:
        plt.plot(run["raw_t"], run["raw_v"] * 3.6, color="#95a5a6", alpha=0.15, linewidth=1)
    plt.plot(class2_art["t_ref"], class2_art["v_ref_t"] * 3.6, color='red', linewidth=2.5, label='class2 ref')
    plt.title(f"{station_pair} - class2 Reference")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity (km/h)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "class2_reference_vt.png", dpi=150)
    plt.close()

    # Plot class3 reference if available
    if class3_art is not None:
        plt.figure(figsize=(10, 6))
        for run in class3_art["valid_runs"]:
            plt.plot(run["raw_t"], run["raw_v"] * 3.6, color="#95a5a6", alpha=0.15, linewidth=1)
        plt.plot(class3_art["t_ref"], class3_art["v_ref_t"] * 3.6, color='blue', linewidth=2.5, label='class3 ref')
        plt.title(f"{station_pair} - class3 Reference (for validation)")
        plt.xlabel("Time (s)"); plt.ylabel("Velocity (km/h)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "class3_reference_vt.png", dpi=150)
        plt.close()

    has_c3 = "class3" in artifacts
    return {
        "station_pair": station_pair, "status": "success",
        "class2_time": round(class2_art["time_ref_raw"], 2),
        "class2_samples": len(class2_art["valid_runs"]),
        "class3_available": has_c3,
        "class3_time": round(class3_art["time_ref_raw"], 2) if has_c3 else None,
        "class3_samples": len(class3_art["valid_runs"]) if has_c3 else 0,
        "output_dir": str(output_dir),
    }

# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_ROOT)

    standard_times_path = resolve_existing_file(STANDARD_TIMES_CANDIDATES, "standard_class_times.csv")
    data_dir = resolve_existing_dir(DATA_DIR_CANDIDATES, "data_processed_new_v2")
    table = load_standard_times(standard_times_path)

    print("=" * 72)
    print("train_ATO_v9: class2 template extraction (class3 for validation)")
    print(f"Time table: {standard_times_path}")
    print(f"Data dir: {data_dir}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Stations: {len(table)}")
    print("=" * 72)

    batch_rows = []
    for i, row in table.iterrows():
        station_pair = str(row["区段"]).strip()
        print(f"\n[{i+1}/{len(table)}] {station_pair}")
        result = train_one_station(station_pair, data_dir)
        batch_rows.append(result)
        if result["status"] == "success":
            c3_info = f", class3={result['class3_time']}s ({result['class3_samples']} runs)" if result["class3_available"] else ", class3=N/A"
            print(f"  OK: class2={result['class2_time']}s ({result['class2_samples']} runs){c3_info}")
        else:
            print(f"  FAIL: {result['status']}")

    df_batch = pd.DataFrame(batch_rows)
    df_batch.to_csv(OUTPUT_ROOT / "batch_train_summary.csv", index=False, encoding="utf-8-sig")

    n_ok = int((df_batch["status"] == "success").sum()) if not df_batch.empty else 0
    n_c3 = int(df_batch["class3_available"].sum()) if "class3_available" in df_batch.columns else 0
    print(f"\nDone: {n_ok}/{len(table)} success, {n_c3} with class3 validation data")
    print(f"Summary: {OUTPUT_ROOT / 'batch_train_summary.csv'}")


if __name__ == "__main__":
    main()
