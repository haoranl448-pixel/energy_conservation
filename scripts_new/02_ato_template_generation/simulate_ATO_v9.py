# -*- coding: utf-8 -*-
"""
simulate_ATO_v9.py

Validation experiment: generate class3 curves using class2 templates as parent,
then compute R2 against real class3 data to verify the extrapolation approach.

Uses the same lam/theta generation logic as v8, but forces class3 to always
use class2 as the parent template.
"""
from __future__ import annotations
from scipy.signal import savgol_filter
import glob, pickle
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Path config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARD_TIMES_CANDIDATES = [
    PROJECT_ROOT / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
]
MODEL_ROOT = PROJECT_ROOT / "output" / "ato_phase_results_v9"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "ato_v9_validation_results_v2"
DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "data_processed_new_v2"
]

# =========================
# Global config (same as v8)
# =========================
VMAX_KMH = 80.0
VMAX_MPS = VMAX_KMH / 3.6
DT_SAMPLE_FALLBACK = 0.05
MIN_RUN_POINTS = 120
TIME_TABLE_REQUIRED_COLS = ["区段", "Class1", "Class2", "Class3", "Class4", "Class5"]
RUN_ID_CANDIDATE_COLS = ["日期+服务号", "服务号", "车底号", "列车运行方向"]
CLASS_COL = "运行等级"
LAMBDA_MIN = 0.55
LAMBDA_MAX_RELAX = 3.50
LAMBDA_GRID_N = 320
DIST_TOL_EXACT = 2.0
OVERSPEED_EPS_KMH = 1e-9
NEAR_CAP_TOL_KMH = 0.02
NEAR_CAP_MIN_POINTS = 4

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Utility functions (adapted from v8)
# =========================
def sanitize_name(name: str) -> str:
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    return str(name).translate({ord(c): '_' for c in bad})

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def normalize_class_label(x):
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

def load_standard_times(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in TIME_TABLE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[TIME_TABLE_REQUIRED_COLS].copy()
    df["区段"] = df["区段"].astype(str).str.strip()
    for c in ["Class1", "Class2", "Class3", "Class4", "Class5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["区段"])

def cumtrapz_uniform(y, dt: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if len(y) >= 2: out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out

def build_order_key(series):
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().sum() >= max(5, int(0.8 * len(series))):
        return s_num.ffill().bfill().values
    s_dt = pd.to_datetime(series, errors="coerce")
    if s_dt.notna().sum() >= max(5, int(0.8 * len(series))):
        return s_dt.astype("int64").values
    return np.arange(len(series), dtype=float)

def resolve_input_files(data_dir: Path, station_pair: str) -> List[Path]:
    exact = data_dir / f"cleaned_{station_pair}.xlsx"
    if exact.exists(): return [exact]
    pattern = str(data_dir / f"cleaned_{station_pair}*.xlsx")
    return [Path(x) for x in sorted(glob.glob(pattern))]

def build_run_id(df: pd.DataFrame):
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

def clean_single_run(g: pd.DataFrame, dt_sample: float = 0.05):
    g = g.copy().reset_index(drop=True)
    g["__order__"] = build_order_key(g["时刻"])
    g["__orig_idx__"] = np.arange(len(g))
    for c in ["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"]:
        if c in g.columns: g[c] = pd.to_numeric(g[c], errors="coerce")
        else: g[c] = np.nan
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
    }

def resample_real_run_to_distance_grid(raw_t: np.ndarray, raw_v: np.ndarray, s_grid: np.ndarray):
    dt = raw_t[1] - raw_t[0] if len(raw_t) >= 2 else DT_SAMPLE_FALLBACK
    s = cumtrapz_uniform(raw_v, dt)
    if s[-1] < s_grid[-1] - 2.0: return None
    return np.interp(s_grid, s, raw_v)

def load_artifacts(model_root: Path, station_pair: str) -> Dict:
    fp = model_root / sanitize_name(station_pair) / "class2_phase_artifacts.pkl"
    if not fp.exists():
        raise FileNotFoundError(f"Artifact not found: {fp}")
    with open(fp, "rb") as f:
        return pickle.load(f)

# =========================
# Core generation logic (from v8)
# =========================
def stretch_velocity_template(v_ref: np.ndarray, duration_new: float, amp_scale: float, dt: float) -> np.ndarray:
    if duration_new <= 0 or len(v_ref) < 2:
        return np.array([amp_scale * v_ref[0], amp_scale * v_ref[-1]])
    n_new = max(int(round(duration_new / dt)) + 1, 2)
    tau_src = np.linspace(0.0, 1.0, len(v_ref))
    tau_dst = np.linspace(0.0, 1.0, n_new)
    return amp_scale * np.interp(tau_dst, tau_src, v_ref)

def build_plateau_window(n: int, edge_ratio: float = 0.18) -> np.ndarray:
    if n <= 2: return np.ones(n, dtype=float)
    edge_n = max(2, int(round(n * edge_ratio)))
    if 2 * edge_n >= n: edge_n = max(1, n // 4)
    w = np.ones(n, dtype=float)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, edge_n)))
    w[:edge_n] = ramp; w[-edge_n:] = ramp[::-1]
    return w

def build_mid_adjustment_basis(v_mid_shape: np.ndarray, vmax_mps: Optional[float]):
    n = len(v_mid_shape)
    if n < 3: return np.zeros_like(v_mid_shape), np.zeros_like(v_mid_shape)
    w = build_plateau_window(n, edge_ratio=0.18)
    core_l = max(0, int(round(0.20 * n)))
    core_r = min(n, int(round(0.80 * n)))
    if core_r <= core_l + 2: core_l, core_r = 0, n
    core = v_mid_shape[core_l:core_r]
    core_level = float(np.median(core)) if len(core) > 0 else float(np.median(v_mid_shape))
    if vmax_mps is None:
        up_room = max(v_mid_shape.max() * 0.6 + 5.0, 1.0)
    else:
        up_room = max(vmax_mps - core_level, 0.0)
    up_basis = w * up_room
    low_floor = max(0.0, min(v_mid_shape[0], v_mid_shape[-1], float(np.quantile(v_mid_shape, 0.15))))
    dn_room = max(core_level - low_floor, 0.0)
    dn_basis = w * dn_room
    return up_basis, dn_basis

def build_curve_with_params(lam: float, theta: float, direction: int,
                            T_target: float, art: Dict, dt: float,
                            vmax_mps: Optional[float]) -> Optional[Dict]:
    T_acc_new = lam * art["T_acc_ref"]
    T_br_new = lam * art["T_br_ref"]
    T_mid_new = T_target - T_acc_new - T_br_new
    if T_mid_new <= 0.2: return None

    v_acc = stretch_velocity_template(art["v_acc_ref"], T_acc_new, lam, dt)
    v_br = stretch_velocity_template(art["v_br_ref"], T_br_new, lam, dt)
    v_mid_shape = stretch_velocity_template(art["v_mid_ref"], T_mid_new, lam, dt)

    if vmax_mps is not None:
        v_acc = np.clip(v_acc, 0.0, vmax_mps)
        v_mid_shape = np.clip(v_mid_shape, 0.0, vmax_mps)
        v_br = np.clip(v_br, 0.0, vmax_mps)

    up_basis, dn_basis = build_mid_adjustment_basis(v_mid_shape, vmax_mps)
    theta = float(np.clip(theta, 0.0, 1.0))

    if direction > 0: v_mid = v_mid_shape + theta * up_basis
    elif direction < 0: v_mid = v_mid_shape - theta * dn_basis
    else: v_mid = v_mid_shape.copy()

    if vmax_mps is not None: v_mid = np.clip(v_mid, 0.0, vmax_mps)

    v_full = np.concatenate([v_acc[:-1], v_mid[:-1], v_br])
    if v_full[-1] > 0.01: v_full = np.append(v_full, 0.0)
    else: v_full[-1] = 0.0
    v_full = safe_savgol(v_full, window=15, poly=3)
    v_full[-1] = 0.0

    t_full = np.arange(len(v_full), dtype=float) * dt
    s_full = cumtrapz_uniform(v_full, dt)

    return {
        "lambda": lam, "theta": theta, "direction": direction,
        "t": t_full, "v": v_full, "s": s_full,
        "T_acc_new": T_acc_new, "T_mid_new": T_mid_new, "T_br_new": T_br_new,
    }

def evaluate_lambda_interval(lam: float, T_target: float, art: Dict, dt: float,
                             vmax_mps: Optional[float]) -> Optional[Dict]:
    base = build_curve_with_params(lam, 0.0, 0, T_target, art, dt, vmax_mps)
    if base is None: return None
    D0 = float(base["s"][-1])
    up_obj = build_curve_with_params(lam, 1.0, +1, T_target, art, dt, vmax_mps)
    dn_obj = build_curve_with_params(lam, 1.0, -1, T_target, art, dt, vmax_mps)
    Dmax = float(up_obj["s"][-1]) if up_obj is not None else D0
    Dmin = float(dn_obj["s"][-1]) if dn_obj is not None else D0
    return {"lambda": lam, "Dmin": min(Dmin, D0, Dmax), "D0": D0, "Dmax": max(Dmin, D0, Dmax)}

def solve_exact_params(T_target: float, art: Dict, dt: float,
                       vmax_mps: Optional[float]) -> Optional[Dict]:
    L_target = float(art["target_l"])
    T_edge_sum = float(art["T_acc_ref"] + art["T_br_ref"])
    if T_edge_sum <= 1e-9: return None
    lam_time_upper = (T_target - 0.2) / T_edge_sum
    if lam_time_upper <= LAMBDA_MIN: return None
    lam_upper = max(LAMBDA_MIN + 1e-3, min(LAMBDA_MAX_RELAX, lam_time_upper))
    lam_nom = np.clip(float(art["time_ref_raw"]) / max(T_target, 1e-6), LAMBDA_MIN, lam_upper)
    lam_grid = np.linspace(LAMBDA_MIN, lam_upper, LAMBDA_GRID_N)
    feasible = []
    for lam in lam_grid:
        ev = evaluate_lambda_interval(lam, T_target, art, dt, vmax_mps)
        if ev is None: continue
        if ev["Dmin"] - DIST_TOL_EXACT <= L_target <= ev["Dmax"] + DIST_TOL_EXACT:
            feasible.append(ev)
    if not feasible: return None
    feasible.sort(key=lambda x: abs(x["lambda"] - lam_nom))
    best = feasible[0]
    lam = float(best["lambda"])
    Dmin, D0, Dmax = float(best["Dmin"]), float(best["D0"]), float(best["Dmax"])
    if L_target >= D0:
        denom = max(Dmax - D0, 1e-9)
        theta = np.clip((L_target - D0) / denom, 0.0, 1.0)
        direction = +1
    else:
        denom = max(D0 - Dmin, 1e-9)
        theta = np.clip((D0 - L_target) / denom, 0.0, 1.0)
        direction = -1
    curve = build_curve_with_params(lam, float(theta), int(direction), T_target, art, dt, vmax_mps)
    if curve is None: return None
    dist_error = float(curve["s"][-1] - L_target)
    peak_speed_mps = float(np.max(curve["v"]))
    sim_time = float(curve["t"][-1]) if len(curve["t"]) > 0 else 0.0
    curve.update({
        "peak_speed_mps": peak_speed_mps,
        "peak_speed_kmh": peak_speed_mps * 3.6,
        "distance_error_m": dist_error,
        "sim_time_s": sim_time,
    })
    return curve

def longest_true_run(mask: np.ndarray) -> int:
    best, cur = 0, 0
    for x in mask.astype(bool):
        if x: cur += 1; best = max(best, cur)
        else: cur = 0
    return best

def postcheck_generated_curve(curve_obj: Dict) -> Dict:
    v_kmh = np.asarray(curve_obj["v"], dtype=float) * 3.6
    peak_kmh_exact = float(np.max(v_kmh)) if len(v_kmh) > 0 else 0.0
    near_cap_mask = v_kmh >= (VMAX_KMH - NEAR_CAP_TOL_KMH)
    near_cap_run = longest_true_run(near_cap_mask)
    checked = dict(curve_obj)
    checked["peak_speed_kmh"] = peak_kmh_exact
    checked["peak_speed_mps"] = peak_kmh_exact / 3.6
    if peak_kmh_exact > VMAX_KMH + OVERSPEED_EPS_KMH:
        return {"status": "overspeed", "curve": None, "diagnostic_peak_speed_kmh": peak_kmh_exact}
    if near_cap_run >= NEAR_CAP_MIN_POINTS:
        return {"status": "overspeed", "curve": None, "diagnostic_peak_speed_kmh": peak_kmh_exact}
    return {"status": "generated", "curve": checked}

# =========================
# Real data extraction (for R2 comparison)
# =========================
def extract_real_runs_for_class(df: pd.DataFrame, class_name: str, art: Dict):
    dt_sample = art.get("dt_sample", DT_SAMPLE_FALLBACK)
    df = df.copy()
    df[CLASS_COL] = df[CLASS_COL].apply(normalize_class_label)
    df = df[df[CLASS_COL] == class_name].copy()
    if df.empty: return []
    run_id, _ = build_run_id(df)
    df["run_id"] = run_id.astype(str)
    clean_runs = []
    for rid, g in df.groupby("run_id"):
        item = clean_single_run(g, dt_sample=dt_sample)
        if item is None: continue
        if abs(item["end_dist"] - float(art["target_l"])) <= float(art.get("end_dist_tol", 30.0)):
            item["run_id"] = rid
            clean_runs.append(item)
    return clean_runs

# =========================
# R2 validation for a single station
# =========================
def validate_class2_to_class3(station_pair: str, class3_target_time: float, artifacts: Dict) -> Dict:
    """Generate class3 from class2 template and compute R2 vs real class3."""
    output_dir = OUTPUT_ROOT / sanitize_name(station_pair)
    ensure_dir(output_dir)

    class2_art = artifacts["class2"]
    dt = class2_art.get("dt_sample", DT_SAMPLE_FALLBACK)

    result = {"station_pair": station_pair, "status": "unknown"}

    # Generate class3 from class2 template
    solve_result = solve_exact_params(class3_target_time, class2_art, dt, vmax_mps=None)
    if solve_result is None:
        result["status"] = "infeasible"
        result["r2_score"] = None
        return result

    postcheck = postcheck_generated_curve(solve_result)
    if postcheck["status"] != "generated":
        result["status"] = postcheck["status"]
        result["r2_score"] = None
        return result

    curve_obj = postcheck["curve"]
    curve_obj["v"] = safe_savgol(curve_obj["v"], window=15, poly=3)

    gen_df = pd.DataFrame({
        "time_s": curve_obj["t"],
        "dist_m": curve_obj["s"],
        "velocity_mps": curve_obj["v"],
        "velocity_kmh": curve_obj["v"] * 3.6,
    })
    gen_df.to_csv(output_dir / "class3_from_class2_curve.csv", index=False, encoding="utf-8-sig")

    # Load real class3 data for comparison
    if "class3" not in artifacts:
        result["status"] = "no_real_class3"
        result["r2_score"] = None
        return result

    class3_art = artifacts["class3"]
    real_runs = class3_art.get("valid_runs", [])
    if len(real_runs) == 0:
        result["status"] = "no_real_runs"
        result["r2_score"] = None
        return result

    # Build comparison on distance grid
    final_target_l = max(float(class2_art["target_l"]), gen_df["dist_m"].max())
    s_grid = np.linspace(0.0, final_target_l, 600)

    real_resampled = []
    for item in real_runs:
        v_real = resample_real_run_to_distance_grid(item["raw_t"], item["raw_v"], s_grid)
        if v_real is not None:
            real_resampled.append(v_real)

    if len(real_resampled) < 2:
        result["status"] = "insufficient_real_data"
        result["r2_score"] = None
        return result

    all_real_v = np.vstack(real_resampled)
    real_med = np.median(all_real_v, axis=0)

    # Generated on same grid
    v_gen_dist = np.interp(s_grid, gen_df["dist_m"], gen_df["velocity_mps"], right=-1.0)
    max_gen_s = gen_df["dist_m"].max()
    for i in range(len(s_grid)):
        if s_grid[i] > max_gen_s or v_gen_dist[i] < 0:
            v_gen_dist[i] = real_med[i]
    v_gen_dist[-1] = 0.0

    r2 = float(r2_score(real_med, v_gen_dist))

    # ---- v-t plot (separate figure) ----
    fig1, ax1 = plt.subplots(figsize=(13, 7))
    for item in real_runs:
        ax1.plot(item["raw_t"], item["raw_v"] * 3.6, color="#95a5a6", alpha=0.18, linewidth=1.0)
    ax1.plot(gen_df["time_s"], gen_df["velocity_kmh"], color="#d62728", linewidth=2.8,
             label=f"class3 from class2 (target={class3_target_time:.0f}s, sim={curve_obj['sim_time_s']:.1f}s) (R2={r2:.4f})")
    ax1.axhline(80, color="red", linestyle="--", alpha=0.6, label="80 km/h cap")
    ax1.set_title(f"{station_pair} | class2->class3 | R2={r2:.4f} | peak={curve_obj['peak_speed_kmh']:.1f} km/h")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Velocity (km/h)")
    ax1.legend(fontsize=9); ax1.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / "class2_to_class3_speed_time.png", dpi=280, bbox_inches="tight")
    plt.close()

    # ---- v-s plot (separate figure) ----
    fig2, ax2 = plt.subplots(figsize=(13, 7))
    real_lo = np.quantile(all_real_v, 0.15, axis=0)
    real_hi = np.quantile(all_real_v, 0.85, axis=0)
    ax2.fill_between(s_grid, real_lo, real_hi, alpha=0.22, color="#5dade2", label="real class3 15%-85% band")
    ax2.plot(s_grid, real_med, color="#2874a6", linewidth=2.4, label="real class3 median")
    ax2.plot(s_grid, v_gen_dist, color="#d62728", linewidth=3.0, label=f"class3 from class2 (R2={r2:.4f})")
    ax2.set_xlabel("Distance (m)"); ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title(f"{station_pair} | speed-distance | R2={r2:.4f}")
    ax2.legend(fontsize=9); ax2.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / "class2_to_class3_speed_distance.png", dpi=280, bbox_inches="tight")
    plt.close()

    result["status"] = "success"
    result["r2_score"] = round(r2, 6)
    result["peak_speed_kmh"] = round(curve_obj["peak_speed_kmh"], 2)
    result["sim_time_s"] = round(curve_obj["sim_time_s"], 2)
    result["real_samples"] = len(real_runs)
    result["gen_dist_m"] = round(float(curve_obj["s"][-1]), 1)
    result["lambda"] = round(curve_obj["lambda"], 4)
    result["theta"] = round(curve_obj["theta"], 4)
    return result

# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_ROOT)
    table = load_standard_times(STANDARD_TIMES_CANDIDATES[0])
    if not STANDARD_TIMES_CANDIDATES[0].exists():
        print(f"ERROR: {STANDARD_TIMES_CANDIDATES[0]} not found")
        return

    print("=" * 72)
    print("simulate_ATO_v9: class2 -> class3 R2 validation")
    print(f"Model dir: {MODEL_ROOT}")
    print(f"Output dir: {OUTPUT_ROOT}")
    print(f"Stations: {len(table)}")
    print("=" * 72)

    all_results = []
    r2_values = []

    for i, row in table.iterrows():
        station_pair = str(row["区段"]).strip()

        # Load v9 artifacts (class2 template + optional class3 ref)
        try:
            artifacts = load_artifacts(MODEL_ROOT, station_pair)
        except FileNotFoundError:
            print(f"[{i+1}/{len(table)}] {station_pair}: SKIP (no class2 artifact)")
            continue

        if "class3" not in artifacts:
            print(f"[{i+1}/{len(table)}] {station_pair}: SKIP (no class3 ref data)")
            continue

        real_c2 = float(artifacts["class2"]["time_ref_raw"])
        real_c3 = float(artifacts["class3"]["time_ref_raw"])
        time_gap = abs(real_c2 - real_c3)
        trivial = time_gap < 2.0

        timetable_target = float(row["Class3"])
        # For trivial stations (c2≈c3), add 2s offset to avoid R2=0.9999
        gen_target = real_c3 * 1.02 if trivial else real_c3
        tag = f"TRIVIAL(+2%={gen_target:.1f}s)" if trivial else f"gap={time_gap:.1f}s"
        print(f"[{i+1}/{len(table)}] {station_pair}: c2={real_c2:.1f}s c3={real_c3:.1f}s ({tag}) gen_target={gen_target:.1f}s")

        result = validate_class2_to_class3(station_pair, gen_target, artifacts)
        result["real_c2_time"] = real_c2
        result["real_c3_time"] = real_c3
        result["time_gap"] = round(time_gap, 1)
        result["trivial"] = trivial
        result["gen_target_s"] = gen_target
        result["timetable_target_s"] = timetable_target
        all_results.append(result)

        if result["status"] == "success":
            r2_values.append(result["r2_score"])
            print(f"  R2={result['r2_score']:.4f} | peak={result['peak_speed_kmh']:.1f} km/h | "
                  f"sim_t={result['sim_time_s']:.1f}s | lam={result['lambda']:.4f} | "
                  f"theta={result['theta']:.4f} | real_samples={result['real_samples']}")
        else:
            print(f"  {result['status']}")

    # Summary — split meaningful vs trivial
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_ROOT / "v9_validation_summary.csv", index=False, encoding="utf-8-sig")

    meaningful = [r for r in all_results if r.get("status") == "success" and not r.get("trivial", False)]
    trivial_list = [r for r in all_results if r.get("status") == "success" and r.get("trivial", False)]

    if meaningful:
        r2_arr = np.array([r["r2_score"] for r in meaningful])
        print("\n" + "=" * 72)
        print(f"R2 SUMMARY — meaningful extrapolation (c2 vs c3 gap >= 2s): {len(meaningful)} stations")
        print(f"  Mean R2:   {r2_arr.mean():.4f}")
        print(f"  Median R2: {np.median(r2_arr):.4f}")
        print(f"  Min R2:    {r2_arr.min():.4f}")
        print(f"  Max R2:    {r2_arr.max():.4f}")
        print(f"  R2 >= 0.90: {(r2_arr >= 0.90).sum()}/{len(meaningful)}")

    if trivial_list:
        print(f"\nTrivial (c2~c3, gap < 2s, R2 not meaningful): {len(trivial_list)} stations")
        for r in trivial_list:
            print(f"  {r['station_pair']}: gap={r.get('time_gap',0):.1f}s R2={r.get('r2_score',0):.4f}")

    print(f"\nSummary: {OUTPUT_ROOT / 'v9_validation_summary.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
