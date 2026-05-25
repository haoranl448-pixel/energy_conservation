# -*- coding: utf-8 -*-
"""
simulate_levels_phase_template_batch_v2.py

用途：
1. 读取 train_ATO/standard_class_times.csv；
2. 按区间加载 class3 三相位模板工件；
3. 批量生成每个区间的 class1-class5 速度曲线；
4. 先在“不限速”条件下求严格解；若该严格解峰值速度 > 80 km/h，则直接标记为 overspeed 并跳过：
   - 不生成该等级的曲线 csv；
   - 不画入总图；
   - 不再通过把曲线硬裁到 80 km/h 的方式伪造可行解；
   - 控制台明确打印“超速跳过”。
5. 每个区间若 class2 生成成功，则自动读取该区间真实 class2 样本，输出：
   - class2_generated_vs_real_metrics.csv
   - class2_generated_vs_real_speed_distance.png
   - class2_generated_vs_real_speed_time.png

说明：
- 本脚本已经从单区间扩展为批量区间；
- 只要 standard_class_times.csv 中有 26 行，就会一次性处理 26 个区间；
- 每个区间会单独输出 summary；
- 全部区间会汇总成 batch_generate_summary.csv。
"""

from __future__ import annotations
from scipy.signal import savgol_filter
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# # =========================
# # 路径配置（相对脚本位置）
# # =========================
# BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = BASE_DIR.parent

# STANDARD_TIMES_CANDIDATES = [
#     BASE_DIR / "standard_class_times.csv",
#     PROJECT_ROOT / "train_ATO" / "standard_class_times.csv",
# ]

# MODEL_ROOT_CANDIDATES = [
#     BASE_DIR / "ato_phase_results",
#     PROJECT_ROOT / "train_ATO" / "ato_phase_results",
# ]

# OUTPUT_ROOT = BASE_DIR / "ato_generated_results"
# DATA_DIR_CANDIDATES = [
#     BASE_DIR / "data_processed_new_v2",
#     PROJECT_ROOT / "train_ATO" / "data_processed_new_v2",
#     PROJECT_ROOT / "data_processed_new_v2",
# ]


# =========================
# 路径配置（已适配 D:\energy_conservation）
# =========================
PROJECT_ROOT = Path(r"D:\energy_conservation")

# 1. 目标时间表
STANDARD_TIMES_CANDIDATES = [
    PROJECT_ROOT / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
]

# 2. 加载上一个脚本生成的模板工件
MODEL_ROOT_CANDIDATES = [
    PROJECT_ROOT / "output" / "ato_phase_results_v2"
]

# 3. 结果输出目录
OUTPUT_ROOT = PROJECT_ROOT / "output" / "ato_generated_results_new_v2"

# 4. 原始数据目录（用于 Class 2 对标验证）
DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "energy_conservation" / "data_processed_new_v2"
]





# =========================
# 全局配置
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
DIST_TOL_EXACT = 2.0  # 数值误差容忍范围，单位 m
OVERSPEED_EPS_KMH = 1e-9
NEAR_CAP_TOL_KMH = 0.02
NEAR_CAP_MIN_POINTS = 4


# =========================
# 工具函数
# =========================
def sanitize_name(name: str) -> str:
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    out = str(name)
    for ch in bad:
        out = out.replace(ch, "_")
    return out


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_class_label(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


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

def safe_savgol(y, window=11, poly=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y.copy()

    # 确保窗口长度不超过数据长度，且为奇数
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        return y.copy()
    if window % 2 == 0:
        window -= 1
    if window <= poly:
        poly = window - 1

    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
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


def class_sort_key(class_name: str):
    order = {"class1": 1, "class2": 2, "class3": 3, "class4": 4, "class5": 5}
    return order.get(class_name, 999)


def cumtrapz_uniform(y, dt: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
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
    if exact.exists():
        return [exact]

    pattern = str(data_dir / f"cleaned_{station_pair}*.xlsx")
    return [Path(x) for x in sorted(glob.glob(pattern))]


def build_run_id(df: pd.DataFrame):
    if "run_id" in df.columns:
        rid = df["run_id"].astype(str).str.strip()
        if rid.nunique() > 1:
            return rid, ["run_id"]

    available = [c for c in RUN_ID_CANDIDATE_COLS if c in df.columns]
    if available:
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


def clean_single_run(g: pd.DataFrame, dt_sample: float = 0.05):
    g = g.copy().reset_index(drop=True)
    g["__order__"] = build_order_key(g["时刻"])
    g["__orig_idx__"] = np.arange(len(g))

    for c in ["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        else:
            g[c] = np.nan

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
    }


def resample_real_run_to_distance_grid(raw_t: np.ndarray, raw_v: np.ndarray, s_grid: np.ndarray):
    dt = raw_t[1] - raw_t[0] if len(raw_t) >= 2 else DT_SAMPLE_FALLBACK
    s = cumtrapz_uniform(raw_v, dt)
    if s[-1] < s_grid[-1] - 2.0:
        return None
    return np.interp(s_grid, s, raw_v)


def load_artifacts(model_root: Path, station_pair: str) -> Dict:
    model_dir = model_root / sanitize_name(station_pair)
    fp = model_dir / "class3_phase_artifacts.pkl"
    if not fp.exists():
        raise FileNotFoundError(f"未找到工件文件：{fp}")
    with open(fp, "rb") as f:
        art = pickle.load(f)
    return art


# =========================
# 核心生成逻辑
# =========================
def stretch_velocity_template(v_ref: np.ndarray, duration_new: float, amp_scale: float, dt: float) -> np.ndarray:
    if duration_new <= 0 or len(v_ref) < 2:
        return np.array([amp_scale * v_ref[0], amp_scale * v_ref[-1]])

    n_new = max(int(round(duration_new / dt)) + 1, 2)
    tau_src = np.linspace(0.0, 1.0, len(v_ref))
    tau_dst = np.linspace(0.0, 1.0, n_new)
    v_new = np.interp(tau_dst, tau_src, v_ref)
    return amp_scale * v_new


def build_plateau_window(n: int, edge_ratio: float = 0.18) -> np.ndarray:
    """
    构造“中间基本为常数、两端平滑过渡”的窗口。
    这样调节中段时更像整体抬升/压低巡航平台，
    而不是只把中间最高点顶起来形成鼓包。
    """
    if n <= 2:
        return np.ones(n, dtype=float)

    edge_n = max(2, int(round(n * edge_ratio)))
    if 2 * edge_n >= n:
        edge_n = max(1, n // 4)

    w = np.ones(n, dtype=float)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, edge_n)))
    w[:edge_n] = ramp
    w[-edge_n:] = ramp[::-1]
    return w



def build_mid_adjustment_basis(v_mid_shape: np.ndarray, vmax_mps: Optional[float]):
    n = len(v_mid_shape)
    if n < 3:
        return np.zeros_like(v_mid_shape), np.zeros_like(v_mid_shape)

    # 旧版这里用 sin^2 权重，中间最大、两头最小，
    # 一旦 direction=+1 就会天然把中段“顶成鼓包”。
    # 这里改成“平台型窗口 + 两端缓变”，让中段更像整体抬升。
    w = build_plateau_window(n, edge_ratio=0.18)

    core_l = max(0, int(round(0.20 * n)))
    core_r = min(n, int(round(0.80 * n)))
    if core_r <= core_l + 2:
        core_l, core_r = 0, n
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

    if T_mid_new <= 0.2:
        return None

    v_acc = stretch_velocity_template(art["v_acc_ref"], T_acc_new, lam, dt)
    v_br = stretch_velocity_template(art["v_br_ref"], T_br_new, lam, dt)
    v_mid_shape = stretch_velocity_template(art["v_mid_ref"], T_mid_new, lam, dt)

    if vmax_mps is not None:
        v_acc = np.clip(v_acc, 0.0, vmax_mps)
        v_mid_shape = np.clip(v_mid_shape, 0.0, vmax_mps)
        v_br = np.clip(v_br, 0.0, vmax_mps)

    up_basis, dn_basis = build_mid_adjustment_basis(v_mid_shape, vmax_mps)
    theta = float(np.clip(theta, 0.0, 1.0))

    if direction > 0:
        v_mid = v_mid_shape + theta * up_basis
    elif direction < 0:
        v_mid = v_mid_shape - theta * dn_basis
    else:
        v_mid = v_mid_shape.copy()

    if vmax_mps is not None:
        v_mid = np.clip(v_mid, 0.0, vmax_mps)

    v_full = np.concatenate([v_acc[:-1], v_mid[:-1], v_br])
    v_full = safe_savgol(v_full, window=15, poly=3)
    
    if vmax_mps is not None:
        v_full = np.clip(v_full, 0.0, vmax_mps)



    if vmax_mps is not None:
        v_full = np.clip(v_full, 0.0, vmax_mps)

    t_full = np.arange(len(v_full), dtype=float) * dt
    s_full = cumtrapz_uniform(v_full, dt)

    return {
        "lambda": lam,
        "theta": theta,
        "direction": direction,
        "t": t_full,
        "v": v_full,
        "s": s_full,
        "T_acc_new": T_acc_new,
        "T_mid_new": T_mid_new,
        "T_br_new": T_br_new,
    }


def evaluate_lambda_interval(lam: float, T_target: float, art: Dict, dt: float,
                             vmax_mps: Optional[float]) -> Optional[Dict]:
    base = build_curve_with_params(lam, 0.0, 0, T_target, art, dt, vmax_mps)
    if base is None:
        return None

    D0 = float(base["s"][-1])
    up_obj = build_curve_with_params(lam, 1.0, +1, T_target, art, dt, vmax_mps)
    dn_obj = build_curve_with_params(lam, 1.0, -1, T_target, art, dt, vmax_mps)

    Dmax = float(up_obj["s"][-1]) if up_obj is not None else D0
    Dmin = float(dn_obj["s"][-1]) if dn_obj is not None else D0

    return {
        "lambda": lam,
        "Dmin": min(Dmin, D0, Dmax),
        "D0": D0,
        "Dmax": max(Dmin, D0, Dmax),
    }


def solve_exact_params(T_target: float, art: Dict, dt: float,
                       vmax_mps: Optional[float]) -> Optional[Dict]:
    L_target = float(art["target_l"])
    T_edge_sum = float(art["T_acc_ref"] + art["T_br_ref"])
    if T_edge_sum <= 1e-9:
        return None

    lam_time_upper = (T_target - 0.2) / T_edge_sum
    if lam_time_upper <= LAMBDA_MIN:
        return None

    lam_upper = max(LAMBDA_MIN + 1e-3, min(LAMBDA_MAX_RELAX, lam_time_upper))
    lam_nom = np.clip(float(art["time_ref_raw"]) / max(T_target, 1e-6), LAMBDA_MIN, lam_upper)

    lam_grid = np.linspace(LAMBDA_MIN, lam_upper, LAMBDA_GRID_N)
    feasible = []

    for lam in lam_grid:
        ev = evaluate_lambda_interval(lam, T_target, art, dt, vmax_mps)
        if ev is None:
            continue
        if ev["Dmin"] - DIST_TOL_EXACT <= L_target <= ev["Dmax"] + DIST_TOL_EXACT:
            feasible.append(ev)

    if not feasible:
        return None

    feasible.sort(key=lambda x: abs(x["lambda"] - lam_nom))
    best = feasible[0]
    lam = float(best["lambda"])
    Dmin = float(best["Dmin"])
    D0 = float(best["D0"])
    Dmax = float(best["Dmax"])

    if L_target >= D0:
        denom = max(Dmax - D0, 1e-9)
        theta = np.clip((L_target - D0) / denom, 0.0, 1.0)
        direction = +1
    else:
        denom = max(D0 - Dmin, 1e-9)
        theta = np.clip((D0 - L_target) / denom, 0.0, 1.0)
        direction = -1

    curve = build_curve_with_params(lam, float(theta), int(direction), T_target, art, dt, vmax_mps)
    if curve is None:
        return None

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


def build_output_dataframe(curve_obj: Dict) -> pd.DataFrame:
    return pd.DataFrame({
        "time_s": curve_obj["t"],
        "dist_m": curve_obj["s"],
        "velocity_mps": curve_obj["v"],
        "velocity_kmh": curve_obj["v"] * 3.6,
    })


def solve_curve_for_class(T_target: float, art: Dict, dt: float) -> Dict:
    """
    先按当前模板生成严格满足目标时间/距离的曲线，
    暂不在这里做最终超速删除。
    最终是否保留，统一放到后验检查里：
    - 先生成；
    - 再看最终生成曲线的最高速度；
    - 超过 80 km/h 就直接删掉；
    - 贴着 80 的平顶伪解也直接删掉。
    """
    relaxed = solve_exact_params(T_target, art, dt, vmax_mps=None)

    if relaxed is None:
        return {
            "status": "infeasible",
            "reason": "在当前模板下未找到满足目标时间/距离的解",
            "curve": None,
        }

    return {
        "status": "generated",
        "reason": "已生成，待做最终峰值速度后验检查",
        "curve": relaxed,
    }


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for x in mask.astype(bool):
        if x:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def postcheck_generated_curve(curve_obj: Dict) -> Dict:
    """
    用户要求的最终判定逻辑：
    1. 先生成 class 曲线；
    2. 再从最终曲线本身取最高速度；
    3. 如果最高速度超过 80 km/h，就删除。

    另外补一条兜底规则：
    - 如果曲线没有“超过”80，但出现连续多个点紧贴 80 km/h，
      说明这是数值或模板放缩造成的平顶边界解，也按无效删掉，
      避免出现图上那种明显贴着 80 的假可行曲线。
    """
    v_kmh = np.asarray(curve_obj["v"], dtype=float) * 3.6
    peak_kmh_exact = float(np.max(v_kmh)) if len(v_kmh) > 0 else 0.0
    near_cap_mask = v_kmh >= (VMAX_KMH - NEAR_CAP_TOL_KMH)
    near_cap_run = longest_true_run(near_cap_mask)

    checked = dict(curve_obj)
    checked["peak_speed_kmh"] = peak_kmh_exact
    checked["peak_speed_mps"] = peak_kmh_exact / 3.6

    if peak_kmh_exact > VMAX_KMH + OVERSPEED_EPS_KMH:
        return {
            "status": "overspeed",
            "reason": f"最终生成曲线最高速度为 {peak_kmh_exact:.6f} km/h，超过 {VMAX_KMH:.1f} km/h，删除该等级",
            "curve": None,
            "diagnostic_peak_speed_kmh": peak_kmh_exact,
            "diagnostic_distance_error_m": float(checked.get("distance_error_m", np.nan)),
        }

    if near_cap_run >= NEAR_CAP_MIN_POINTS:
        return {
            "status": "overspeed",
            "reason": f"最终生成曲线存在连续 {near_cap_run} 个点贴近 {VMAX_KMH:.1f} km/h（峰值 {peak_kmh_exact:.6f} km/h），判为平顶边界解并删除",
            "curve": None,
            "diagnostic_peak_speed_kmh": peak_kmh_exact,
            "diagnostic_distance_error_m": float(checked.get("distance_error_m", np.nan)),
        }

    return {
        "status": "generated",
        "reason": f"最终生成曲线后验检查通过，峰值 {peak_kmh_exact:.6f} km/h",
        "curve": checked,
    }

# =========================
# class2 真实对比
# =========================
def locate_source_data_dir() -> Path:
    return resolve_existing_dir(DATA_DIR_CANDIDATES, "原始 cleaned_xxx 数据")


def load_source_dataframe_for_station(art: Dict, station_pair: str) -> pd.DataFrame:
    source_files = art.get("source_files", [])
    if source_files:
        file_paths = [Path(x) for x in source_files if Path(x).exists()]
    else:
        data_dir = locate_source_data_dir()
        file_paths = resolve_input_files(data_dir, station_pair)

    if not file_paths:
        raise FileNotFoundError(f"无法定位区间 {station_pair} 的原始 cleaned_xxx.xlsx 文件")

    all_df = []
    for fp in file_paths:
        all_df.append(pd.read_excel(fp))
    return pd.concat(all_df, ignore_index=True)


def extract_real_runs_for_class(df: pd.DataFrame, class_name: str, art: Dict):
    dt_sample = art.get("dt_sample", DT_SAMPLE_FALLBACK)

    df = df.copy()
    df[CLASS_COL] = df[CLASS_COL].apply(normalize_class_label)
    df = df[df[CLASS_COL] == class_name].copy()
    if df.empty:
        return []

    run_id, _ = build_run_id(df)
    df["run_id"] = run_id.astype(str)

    clean_runs = []
    for rid, g in df.groupby("run_id"):
        item = clean_single_run(g, dt_sample=dt_sample)
        if item is None:
            continue
        if abs(item["end_dist"] - float(art["target_l"])) <= float(art.get("end_dist_tol", 30.0)):
            item["run_id"] = rid
            clean_runs.append(item)

    return clean_runs


def compare_generated_with_real_class(class_name: str, generated_df: pd.DataFrame,
                                      real_runs: List[Dict], art: Dict,
                                      output_dir: Path) -> Dict:
    if len(real_runs) == 0:
        return {
            "compare_status": "no_real_runs",
            "compare_message": f"未找到真实 {class_name} 样本，跳过对比",
            "real_sample_count": 0,
        }

    s_grid = np.linspace(0.0, float(art["target_l"]), 500)
    v_gen_dist = np.interp(s_grid, generated_df["dist_m"], generated_df["velocity_mps"])
    t_gen = generated_df["time_s"].values
    v_gen_time = generated_df["velocity_mps"].values

    real_resampled = []
    metric_rows = []

    for item in real_runs:
        v_real_dist = resample_real_run_to_distance_grid(item["raw_t"], item["raw_v"], s_grid)
        if v_real_dist is None:
            continue

        real_resampled.append({
            "run_id": item["run_id"],
            "v_real_dist": v_real_dist,
            "raw_t": item["raw_t"],
            "raw_v": item["raw_v"],
            "travel_time_raw": item["travel_time_raw"],
        })

        rmse = float(np.sqrt(np.mean((v_real_dist - v_gen_dist) ** 2)))
        mae = float(np.mean(np.abs(v_real_dist - v_gen_dist)))

        if np.std(v_real_dist) > 1e-8 and np.std(v_gen_dist) > 1e-8:
            corr = float(np.corrcoef(v_real_dist, v_gen_dist)[0, 1])
        else:
            corr = np.nan

        peak_real = float(np.max(v_real_dist))
        peak_gen = float(np.max(v_gen_dist))

        metric_rows.append({
            "run_id": item["run_id"],
            "travel_time_raw_real_s": round(item["travel_time_raw"], 4),
            "travel_time_generated_s": round(float(t_gen[-1]), 4),
            "rmse_speed_mps": round(rmse, 6),
            "mae_speed_mps": round(mae, 6),
            "corr_speed_shape": round(corr, 6) if pd.notna(corr) else np.nan,
            "peak_speed_real_mps": round(peak_real, 6),
            "peak_speed_generated_mps": round(peak_gen, 6),
            "peak_speed_gap_mps": round(peak_gen - peak_real, 6),
        })

    if len(real_resampled) == 0:
        return {
            "compare_status": "no_valid_real_runs",
            "compare_message": f"{class_name} 可用于对比的真实样本为 0",
            "real_sample_count": 0,
        }

    df_metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / f"{class_name}_generated_vs_real_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(13, 7))
    all_real_v = np.vstack([x["v_real_dist"] for x in real_resampled])
    for x in real_resampled:
        plt.plot(s_grid, x["v_real_dist"], color="#95a5a6", alpha=0.18, linewidth=1.0)

    real_med = np.median(all_real_v, axis=0)
    real_lo = np.quantile(all_real_v, 0.15, axis=0)
    real_hi = np.quantile(all_real_v, 0.85, axis=0)

    plt.fill_between(s_grid, real_lo, real_hi, alpha=0.22, color="#5dade2", label=f"real {class_name} 15%-85% band")
    plt.plot(s_grid, real_med, color="#2874a6", linewidth=2.4, label=f"real {class_name} median")
    plt.plot(s_grid, v_gen_dist, color="#d62728", linewidth=3.0, label=f"generated {class_name}")

    plt.xlabel("Distance (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"{class_name}: generated vs real (speed-distance)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    dist_fig_path = output_dir / f"{class_name}_generated_vs_real_speed_distance.png"
    plt.savefig(dist_fig_path, dpi=280, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(13, 7))
    for x in real_resampled:
        plt.plot(x["raw_t"], x["raw_v"] * 3.6, color="#95a5a6", alpha=0.18, linewidth=1.0)

    plt.plot(t_gen, v_gen_time * 3.6, color="#d62728", linewidth=3.0, label=f"generated {class_name}")
    plt.axhline(VMAX_KMH, color="red", linestyle="--", alpha=0.7, label="80 km/h cap")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/h)")
    plt.title(f"{class_name}: generated vs real (speed-time)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    time_fig_path = output_dir / f"{class_name}_generated_vs_real_speed_time.png"
    plt.savefig(time_fig_path, dpi=280, bbox_inches="tight")
    plt.close()

    return {
        "compare_status": "success",
        "compare_message": f"{class_name} 对比完成",
        "real_sample_count": len(real_resampled),
        "metrics_csv": str(metrics_path),
        "speed_distance_png": str(dist_fig_path),
        "speed_time_png": str(time_fig_path),
        "mean_rmse_speed_mps": float(df_metrics["rmse_speed_mps"].mean()),
        "mean_mae_speed_mps": float(df_metrics["mae_speed_mps"].mean()),
        "mean_corr_speed_shape": float(df_metrics["corr_speed_shape"].mean(skipna=True)),
    }


# =========================
# 绘图
# =========================
def plot_all_vt(all_results: Dict[str, Dict], output_dir: Path, station_pair: str):
    generated = {k: v for k, v in all_results.items() if v["status"] == "generated"}
    if not generated:
        return

    plt.figure(figsize=(13, 7))
    color_map = {
        "class1": "#1f77b4",
        "class2": "#ff7f0e",
        "class3": "#2ca02c",
        "class4": "#d62728",
        "class5": "#9467bd",
    }

    for class_name in sorted(generated.keys(), key=class_sort_key):
        df = generated[class_name]["df"]
        T_target = generated[class_name]["target_time"]
        plt.plot(df["time_s"], df["velocity_kmh"], linewidth=2.6, color=color_map[class_name], label=f"{class_name} ({T_target:.0f}s)")

    skipped = [k for k, v in all_results.items() if v["status"] != "generated"]
    title = f"{station_pair} | Generated speed-time curves"
    if skipped:
        title += f" | skipped: {', '.join(skipped)}"

    plt.axhline(VMAX_KMH, color="red", linestyle="--", alpha=0.7, label="80 km/h cap")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/h)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all_generated_speed_time.png", dpi=280, bbox_inches="tight")
    plt.close()


def plot_all_vs(all_results: Dict[str, Dict], output_dir: Path, station_pair: str):
    generated = {k: v for k, v in all_results.items() if v["status"] == "generated"}
    if not generated:
        return

    plt.figure(figsize=(13, 7))
    color_map = {
        "class1": "#1f77b4",
        "class2": "#ff7f0e",
        "class3": "#2ca02c",
        "class4": "#d62728",
        "class5": "#9467bd",
    }

    for class_name in sorted(generated.keys(), key=class_sort_key):
        df = generated[class_name]["df"]
        T_target = generated[class_name]["target_time"]
        plt.plot(df["dist_m"], df["velocity_mps"], linewidth=2.6, color=color_map[class_name], label=f"{class_name} ({T_target:.0f}s)")

    skipped = [k for k, v in all_results.items() if v["status"] != "generated"]
    title = f"{station_pair} | Generated speed-distance curves"
    if skipped:
        title += f" | skipped: {', '.join(skipped)}"

    plt.xlabel("Distance (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all_generated_speed_distance.png", dpi=280, bbox_inches="tight")
    plt.close()


def plot_single_class_detail(class_name: str, result: Dict, output_dir: Path, station_pair: str):
    if result["status"] != "generated":
        return

    df = result["df"]
    curve = result["curve_obj"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    title = (
        f"{station_pair} | {class_name} | target={result['target_time']:.2f}s | "
        f"sim={curve['sim_time_s']:.2f}s | peak={curve['peak_speed_kmh']:.2f} km/h"
    )

    axes[0].plot(df["time_s"], df["velocity_kmh"], linewidth=2.8)
    axes[0].axhline(VMAX_KMH, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title(title)
    axes[0].set_ylabel("Velocity (km/h)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(df["dist_m"], df["velocity_mps"], linewidth=2.8)
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_dir / f"{class_name}_detail.png", dpi=280, bbox_inches="tight")
    plt.close()


# =========================
# 单区间批量生成
# =========================
def generate_for_station_pair(station_pair: str, level_target_times: Dict[str, float], model_root: Path) -> Dict:
    output_dir = OUTPUT_ROOT / sanitize_name(station_pair)
    ensure_dir(output_dir)

    try:
        art = load_artifacts(model_root, station_pair)
    except Exception as e:
        return {
            "station_pair": station_pair,
            "status": "missing_artifact",
            "message": str(e),
        }

    dt = art.get("dt_sample", DT_SAMPLE_FALLBACK)
    art["level_target_times"] = level_target_times

    all_results: Dict[str, Dict] = {}
    summary_rows = []

    for class_name in sorted(level_target_times.keys(), key=class_sort_key):
        T_target = float(level_target_times[class_name])
        solve_result = solve_curve_for_class(T_target, art, dt)
        if solve_result["status"] == "generated":
            solve_result = postcheck_generated_curve(solve_result["curve"])

        status = solve_result["status"]
        reason = solve_result["reason"]

        if status == "generated":
            curve_obj = solve_result["curve"]
            df_out = build_output_dataframe(curve_obj)
            curve_csv_path = output_dir / f"{class_name}_generated_curve.csv"
            df_out.to_csv(curve_csv_path, index=False, encoding="utf-8-sig")

            result = {
                "class_name": class_name,
                "target_time": T_target,
                "status": status,
                "reason": reason,
                "curve_obj": curve_obj,
                "df": df_out,
            }
            all_results[class_name] = result
            plot_single_class_detail(class_name, result, output_dir, station_pair)

            summary_rows.append({
                "station_pair": station_pair,
                "class_name": class_name,
                "status": status,
                "reason": reason,
                "target_time_s": round(T_target, 4),
                "sim_time_s": round(curve_obj["sim_time_s"], 4),
                "time_error_s": round(curve_obj["sim_time_s"] - T_target, 4),
                "lambda": round(curve_obj["lambda"], 6),
                "theta": round(curve_obj["theta"], 6),
                "direction": int(curve_obj["direction"]),
                "peak_speed_kmh": round(curve_obj["peak_speed_kmh"], 4),
                "total_dist_m": round(float(curve_obj["s"][-1]), 4),
                "dist_error_m": round(curve_obj["distance_error_m"], 4),
                "T_acc_new_s": round(curve_obj["T_acc_new"], 4),
                "T_mid_new_s": round(curve_obj["T_mid_new"], 4),
                "T_br_new_s": round(curve_obj["T_br_new"], 4),
                "curve_csv": str(curve_csv_path),
            })
        else:
            all_results[class_name] = {
                "class_name": class_name,
                "target_time": T_target,
                "status": status,
                "reason": reason,
                "curve_obj": None,
                "df": None,
            }
            summary_rows.append({
                "station_pair": station_pair,
                "class_name": class_name,
                "status": status,
                "reason": reason,
                "target_time_s": round(T_target, 4),
                "sim_time_s": np.nan,
                "time_error_s": np.nan,
                "lambda": np.nan,
                "theta": np.nan,
                "direction": np.nan,
                "peak_speed_kmh": round(float(solve_result.get("diagnostic_peak_speed_kmh", np.nan)), 4)
                    if pd.notna(solve_result.get("diagnostic_peak_speed_kmh", np.nan)) else np.nan,
                "total_dist_m": np.nan,
                "dist_error_m": round(float(solve_result.get("diagnostic_distance_error_m", np.nan)), 4)
                    if pd.notna(solve_result.get("diagnostic_distance_error_m", np.nan)) else np.nan,
                "T_acc_new_s": np.nan,
                "T_mid_new_s": np.nan,
                "T_br_new_s": np.nan,
                "curve_csv": "",
            })

    df_summary = pd.DataFrame(summary_rows)

    class2_compare_info = {
        "compare_status": "skipped",
        "compare_message": "class2 未生成成功，跳过真实对比",
        "real_sample_count": 0,
        "metrics_csv": "",
        "speed_distance_png": "",
        "speed_time_png": "",
        "mean_rmse_speed_mps": np.nan,
        "mean_mae_speed_mps": np.nan,
        "mean_corr_speed_shape": np.nan,
    }

    if all_results.get("class2", {}).get("status") == "generated":
        try:
            source_df = load_source_dataframe_for_station(art, station_pair)
            real_class2_runs = extract_real_runs_for_class(source_df, "class2", art)
            class2_compare_info = compare_generated_with_real_class(
                class_name="class2",
                generated_df=all_results["class2"]["df"],
                real_runs=real_class2_runs,
                art=art,
                output_dir=output_dir,
            )
        except Exception as e:
            class2_compare_info = {
                "compare_status": "error",
                "compare_message": str(e),
                "real_sample_count": 0,
                "metrics_csv": "",
                "speed_distance_png": "",
                "speed_time_png": "",
                "mean_rmse_speed_mps": np.nan,
                "mean_mae_speed_mps": np.nan,
                "mean_corr_speed_shape": np.nan,
            }

    df_summary["class2_compare_status"] = class2_compare_info["compare_status"]
    df_summary["class2_compare_message"] = class2_compare_info["compare_message"]
    df_summary["class2_real_sample_count"] = class2_compare_info["real_sample_count"]
    df_summary["class2_compare_metrics_csv"] = class2_compare_info.get("metrics_csv", "")
    df_summary["class2_compare_speed_distance_png"] = class2_compare_info.get("speed_distance_png", "")
    df_summary["class2_compare_speed_time_png"] = class2_compare_info.get("speed_time_png", "")
    df_summary["class2_compare_mean_rmse_speed_mps"] = class2_compare_info.get("mean_rmse_speed_mps", np.nan)
    df_summary["class2_compare_mean_mae_speed_mps"] = class2_compare_info.get("mean_mae_speed_mps", np.nan)
    df_summary["class2_compare_mean_corr_speed_shape"] = class2_compare_info.get("mean_corr_speed_shape", np.nan)

    summary_csv_path = output_dir / "all_classes_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    plot_all_vt(all_results, output_dir, station_pair)
    plot_all_vs(all_results, output_dir, station_pair)

    n_generated = int((df_summary["status"] == "generated").sum())
    n_overspeed = int((df_summary["status"] == "overspeed").sum())
    n_infeasible = int((df_summary["status"] == "infeasible").sum())

    return {
        "station_pair": station_pair,
        "status": "success",
        "message": "生成完成",
        "n_generated": n_generated,
        "n_overspeed": n_overspeed,
        "n_infeasible": n_infeasible,
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv_path),
        "class2_compare_status": class2_compare_info["compare_status"],
        "class2_compare_message": class2_compare_info["compare_message"],
        "class2_real_sample_count": class2_compare_info["real_sample_count"],
        "class2_compare_metrics_csv": class2_compare_info.get("metrics_csv", ""),
    }


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUTPUT_ROOT)

    standard_times_path = resolve_existing_file(STANDARD_TIMES_CANDIDATES, "standard_class_times.csv")
    model_root = resolve_existing_dir(MODEL_ROOT_CANDIDATES, "class3 模板工件")

    table = load_standard_times(standard_times_path)
    batch_rows = []

    print("=" * 72)
    print("批量生成 class1-class5 速度曲线")
    print(f"时间表: {standard_times_path}")
    print(f"模板目录: {model_root}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"硬限速: {VMAX_KMH:.1f} km/h")
    print(f"待处理区间数: {len(table)}")
    print("=" * 72)

    for i, row in table.iterrows():
        station_pair = str(row["区段"]).strip()
        level_target_times = row_to_level_target_times(row)

        print(f"\n[{i + 1}/{len(table)}] 开始生成区间: {station_pair}")
        result = generate_for_station_pair(station_pair, level_target_times, model_root)
        batch_rows.append(result)

        if result["status"] != "success":
            print(f"  ❌ 失败 | {result['status']} | {result['message']}")
            continue

        print(f"  ✅ 完成 | generated={result['n_generated']} | overspeed={result['n_overspeed']} | infeasible={result['n_infeasible']}")
        print(f"  class2对比: {result['class2_compare_status']} | {result['class2_compare_message']}")

        station_summary = pd.read_csv(result["summary_csv"], encoding="utf-8-sig")
        for _, sr in station_summary.iterrows():
            cname = sr["class_name"]
            status = sr["status"]
            if status == "generated":
                print(f"      - {cname}: 生成成功 | peak={sr['peak_speed_kmh']:.6f} km/h")
            elif status == "overspeed":
                print(f"      - {cname}: 超速跳过 | {sr['reason']}")
            else:
                print(f"      - {cname}: 跳过 | {sr['reason']}")

    df_batch = pd.DataFrame(batch_rows)
    batch_summary_csv = OUTPUT_ROOT / "batch_generate_summary.csv"
    df_batch.to_csv(batch_summary_csv, index=False, encoding="utf-8-sig")

    n_success = int((df_batch["status"] == "success").sum()) if not df_batch.empty else 0
    n_fail = len(df_batch) - n_success

    print("\n" + "=" * 72)
    print("批量生成结束")
    print(f"成功区间数: {n_success}")
    print(f"失败区间数: {n_fail}")
    print(f"汇总文件: {batch_summary_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
