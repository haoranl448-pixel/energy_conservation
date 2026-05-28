# -*- coding: utf-8 -*-
"""
主线第 3 步：批量提取 ATO 多等级相位模板。

用途：
1. 从 train_ATO/standard_class_times.csv 读取区间级标准等级时间表；
2. 对时间表中的每个区间，读取 cleaned_{区间}.xlsx；
3. 按运行等级 class1-class5 分组提取真实样本；
4. 构建各等级的三相位模板（加速 / 中段 / 制动）；
5. 为后续 simulate 脚本保存统一工件。

说明：
- 本脚本已经从“单区间”扩展为“批量区间”；
- 只要 standard_class_times.csv 中有 26 行，就会一次性训练 26 个区间；
- 若某区间数据文件不存在或 class3 样本不足，会在批量汇总中标记并跳过。
"""

# 开启较新的类型注解写法。
from __future__ import annotations
# B 样条插值，用于重构更平滑的参考速度曲线。
from scipy.interpolate import make_interp_spline
# os/glob/pickle：路径扫描和模板工件保存。
import os
import glob
import pickle
# Path：统一处理路径。
from pathlib import Path
# Dict/List/Optional/Tuple：类型标注，便于理解函数输入输出。
from typing import Dict, List, Optional, Tuple

# numpy/pandas：数值计算和表格处理。
import numpy as np
import pandas as pd
# matplotlib：保存模板曲线图；Agg 后端适合无界面批处理。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Savitzky-Golay 滤波，用于平滑速度曲线。
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
# 当前脚本使用硬编码项目根目录，便于在本机固定工程目录下运行。
PROJECT_ROOT = Path(r"D:\energy_conservation")

def get_data_dir_candidates(defaults: List[Path]) -> List[Path]:
    """如果主程序指定了测试数据目录，就优先使用该目录。"""

    raw = os.environ.get("ENERGY_DATA_DIR")
    if not raw:
        return defaults
    data_dir = Path(raw)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return [data_dir, *defaults]

# 标准等级时间表位置：每个区间 Class1-Class5 的目标时间。
STANDARD_TIMES_CANDIDATES = [
    PROJECT_ROOT / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
]

# 清洗后的历史运行数据位置，要求文件名形如 cleaned_区间.xlsx。
DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "data_processed_new_v2"
]
DATA_DIR_CANDIDATES = get_data_dir_candidates(DATA_DIR_CANDIDATES)

# 模板工件保存位置，每个区间一个子目录。
OUTPUT_ROOT = PROJECT_ROOT / "output" / "ato_phase_results_v3"






# =========================
# 全局配置
# =========================
# 历史数据中表示运行等级的列名。
CLASS_COL = "运行等级"
# 参考等级名；当前脚本已经支持多等级，此变量主要是历史保留。
REFERENCE_CLASS = "class3"
# 采样间隔，单位秒。
DT_SAMPLE = 0.05
# 终点距离容忍范围，用于过滤异常样本。
END_DIST_TOL = 30.0
# 单条运行样本最少点数，过短样本直接丢弃。
MIN_RUN_POINTS = 120

# 候选 run_id 字段；用于区分同一个 Excel 中的多趟运行。
RUN_ID_CANDIDATE_COLS = [
    "日期+服务号",
    "服务号",
    "车底号",
    "列车运行方向",
]

# 清洗后数据必须包含的字段。
REQUIRED_COLS = [
    "时刻",
    "累计位移(m)",
    "速度(m/s)",
    "curvature",
    "gradient",
    "重量",
    CLASS_COL,
]

# 标准时间表必须包含的字段。
TIME_TABLE_REQUIRED_COLS = ["区段", "Class1", "Class2", "Class3", "Class4", "Class5"]


# =========================
# 通用工具
# =========================
def sanitize_name(name: str) -> str:
    """把区间名转换为可安全用作文件夹名的字符串。"""

    # Windows 文件名不能包含这些特殊字符。
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    out = str(name)
    for ch in bad:
        out = out.replace(ch, "_")
    return out



def ensure_dir(path: Path):
    """确保目录存在。"""

    path.mkdir(parents=True, exist_ok=True)



def normalize_class_label(x) -> Optional[str]:
    """统一运行等级字段，例如 Class3 -> class3。"""

    # 空值无法归类。
    if pd.isna(x):
        return None
    return str(x).strip().lower()



def safe_savgol(y, window=11, poly=3):
    """安全地对速度序列做 Savitzky-Golay 平滑。"""

    y = np.asarray(y, dtype=float)
    n = len(y)
    # 数据太短时不平滑，避免窗口长度不合法。
    if n < 5:
        return y.copy()

    # 窗口不能超过序列长度，且必须为奇数。
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
    """把时刻列转换成可排序的数值键。"""

    # 优先尝试数值型时间。
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().sum() >= max(5, int(0.8 * len(series))):
        return s_num.ffill().bfill().values

    # 其次尝试 datetime 时间。
    s_dt = pd.to_datetime(series, errors="coerce")
    if s_dt.notna().sum() >= max(5, int(0.8 * len(series))):
        return s_dt.astype("int64").values

    # 都无法解析时，用原始行号作为顺序。
    return np.arange(len(series), dtype=float)



def resolve_existing_file(candidates: List[Path], desc: str) -> Path:
    """从候选路径里找到第一个存在的文件。"""

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到{desc}，候选路径：{[str(x) for x in candidates]}")



def resolve_existing_dir(candidates: List[Path], desc: str) -> Path:
    """从候选路径里找到第一个存在的目录。"""

    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"未找到{desc}目录，候选路径：{[str(x) for x in candidates]}")



def resolve_input_files(data_dir: Path, station_pair: str) -> List[Path]:
    """定位某个站间区间对应的 cleaned/results Excel 文件。"""

    # 优先匹配精确文件名；测试数据可直接使用 results_ 前缀。
    for prefix in ("cleaned", "results"):
        exact = data_dir / f"{prefix}_{station_pair}.xlsx"
        if exact.exists():
            return [exact]

    # 如果精确文件不存在，则用通配符兼容后缀版本。
    matches: List[Path] = []
    for prefix in ("cleaned", "results"):
        pattern = str(data_dir / f"{prefix}_{station_pair}*.xlsx")
        matches.extend(Path(x) for x in sorted(glob.glob(pattern)))
    return matches


def station_pairs_available_in_data_dir(data_dir: Path) -> set[str]:
    """读取数据目录中实际存在的站间区间。"""

    pairs: set[str] = set()
    for fp in data_dir.glob("*.xlsx"):
        for prefix in ("cleaned_", "results_"):
            if fp.stem.startswith(prefix):
                pairs.add(fp.stem[len(prefix):])
    return pairs


def filter_table_by_available_data(table: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """指定测试数据目录时，只处理该目录中存在的区间。"""

    if not os.environ.get("ENERGY_DATA_DIR"):
        return table
    available = station_pairs_available_in_data_dir(data_dir)
    filtered = table[table["区段"].isin(available)].copy()
    if filtered.empty:
        raise FileNotFoundError(f"数据目录 {data_dir} 中没有和标准时间表匹配的 cleaned_/results_ 文件。")
    skipped = len(table) - len(filtered)
    print(f"按测试数据目录筛选区间: {len(filtered)} 个，跳过标准时间表中未提供数据的 {skipped} 个区间。")
    return filtered



def build_run_id(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """为历史数据构造每一趟运行的 run_id。"""

    # 如果数据本身已有 run_id 且能区分多趟运行，直接使用。
    if "run_id" in df.columns:
        rid = df["run_id"].astype(str).str.strip()
        if rid.nunique() > 1:
            return rid, ["run_id"]

    # 否则尝试组合服务号、车底号、方向等字段。
    available = [c for c in RUN_ID_CANDIDATE_COLS if c in df.columns]
    if len(available) > 0:
        tmp = df[available].copy()
        for c in available:
            tmp[c] = tmp[c].fillna("NA").astype(str).str.strip()
        rid = tmp.agg("|".join, axis=1)
        if rid.nunique() > 1:
            return rid, available

    # 最后兜底：通过时间倒跳或位移回退判断新运行段。
    order_key = build_order_key(df["时刻"])
    dist = pd.to_numeric(df["累计位移(m)"], errors="coerce").ffill().fillna(0.0).values
    dt_jump = np.diff(order_key, prepend=order_key[0])
    ds_jump = np.diff(dist, prepend=dist[0])

    new_run = (dt_jump < 0) | (ds_jump < -1.0)
    run_id = np.cumsum(new_run).astype(str)
    return pd.Series(run_id, index=df.index), ["fallback_time_dist_reset"]



def cumtrapz_uniform(y, dt: float) -> np.ndarray:
    """在等间隔采样下做梯形累计积分。"""

    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out



def clean_single_run(g: pd.DataFrame, dt_sample=0.05) -> Optional[Dict]:
    """清洗单趟运行样本，输出时间、速度、位移和静态特征。"""

    # 复制并重排行号，避免修改原 DataFrame。
    g = g.copy().reset_index(drop=True)
    g["__order__"] = build_order_key(g["时刻"])
    g["__orig_idx__"] = np.arange(len(g))

    for c in ["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"]:
        # 关键列统一转数值，无法转换的设为 NaN。
        g[c] = pd.to_numeric(g[c], errors="coerce")

    # 关键列缺失的采样点无法用于模板提取。
    g = g.dropna(subset=["累计位移(m)", "速度(m/s)", "curvature", "gradient", "重量"])
    if len(g) < MIN_RUN_POINTS:
        return None

    # 按真实时序排序。
    g = g.sort_values(["__order__", "__orig_idx__"]).reset_index(drop=True)

    raw_v = g["速度(m/s)"].clip(lower=0).values.astype(float)
    raw_t = np.arange(len(raw_v), dtype=float) * dt_sample

    # 位移归零并强制单调，避免传感器小幅回退影响终点距离。
    s = g["累计位移(m)"].values.astype(float)
    s = s - s[0]
    s = np.maximum.accumulate(s)

    # 过短区间通常是异常样本，直接跳过。
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
    """把不同长度的速度曲线重采样到统一归一化时间网格。"""

    if len(raw_v) < 2:
        return None
    tau_src = np.linspace(0.0, 1.0, len(raw_v))
    tau_dst = np.linspace(0.0, 1.0, n_ref)
    return np.interp(tau_dst, tau_src, raw_v)



def detect_phase_boundaries(v_ref_t: np.ndarray, dt_ref: float):
    """识别参考曲线的加速结束点和制动开始点。"""

    # 先平滑参考速度，降低噪声对峰值平台识别的影响。
    v = safe_savgol(v_ref_t, window=min(101, len(v_ref_t) - (1 - len(v_ref_t) % 2)), poly=3)
    v = np.clip(v, 0.0, None)
    v_peak = float(np.max(v))

    idx_a = None
    idx_b = None

    # 按多个峰值比例尝试寻找巡航/平台区间。
    for ratio in [0.97, 0.965, 0.96, 0.95, 0.94, 0.93, 0.92]:
        mask = v >= ratio * v_peak
        idxs = np.where(mask)[0]
        if len(idxs) >= max(20, int(1.0 / max(dt_ref, 1e-6))):
            idx_a = int(idxs[0])
            idx_b = int(idxs[-1])
            break

    # 如果无法稳定识别平台，则使用经验比例兜底。
    if idx_a is None or idx_b is None or idx_b <= idx_a:
        n = len(v)
        idx_a = int(0.28 * n)
        idx_b = int(0.72 * n)

    # 防止边界太靠近首尾，导致相位长度过短。
    if idx_a < 10:
        idx_a = 10
    if idx_b > len(v) - 11:
        idx_b = len(v) - 11
    if idx_b <= idx_a + 10:
        idx_b = min(len(v) - 11, idx_a + 20)

    return idx_a, idx_b, v_peak



def plot_class3_all_runs(raw_runs: List[Dict], v_ref_t: np.ndarray, t_ref: np.ndarray,
                         idx_a: int, idx_b: int, output_dir: Path):
    """绘制 class3 历史样本与参考曲线，用于模板质量检查。"""

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
    """绘制 class3 参考速度-距离曲线。"""

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
    """读取并校验标准等级时间表。"""

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
    """把时间表一行转换为 class -> 目标时间 的字典。"""

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
    """对单个站间区间提取所有可用等级的三相位模板。"""

    output_dir = OUTPUT_ROOT / sanitize_name(station_pair)
    ensure_dir(output_dir)

    # 1. 读取该区间所有 cleaned 数据文件。
    files = resolve_input_files(data_dir, station_pair)
    if not files:
        return {"station_pair": station_pair, "status": "missing_file", "message": f"未找到 cleaned_{station_pair}.xlsx"}

    all_df = []
    for fp in files:
        all_df.append(pd.read_excel(fp))
    df = pd.concat(all_df, ignore_index=True)

    # 2. 预检查必要字段。
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return {"station_pair": station_pair, "status": "missing_columns", "message": f"缺少必要列: {missing_cols}"}

    df[CLASS_COL] = df[CLASS_COL].apply(normalize_class_label)
    
    # 3. 识别当前区间包含的所有历史等级。
    available_classes = [c for c in ["class1", "class2", "class3", "class4", "class5"] 
                         if c in df[CLASS_COL].unique()]
    
    if not available_classes:
        return {"station_pair": station_pair, "status": "no_class_data", "message": "未识别到任何 class 数据"}

    # 用于存储所有等级模板的大字典。
    multi_artifacts = {}
    summary_stats = []

    print(f"   📊 区间 {station_pair} 发现等级: {available_classes}")

    # 4. 循环处理每一个等级。
    for current_cls in available_classes:
        df_ref = df[df[CLASS_COL] == current_cls].copy()
        
        # 按 run_id 切分每一趟真实运行。
        run_id, _ = build_run_id(df_ref)
        df_ref["run_id"] = run_id.astype(str)

        # 清洗每一趟运行，过滤掉异常或过短样本。
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

        # 计算该等级的基准指标：典型距离、典型时间、典型载重。
        end_dists = np.array([x["end_dist"] for x in clean_runs], dtype=float)
        target_l = float(np.median(end_dists))
        valid_runs = [x for x in clean_runs if abs(x["end_dist"] - target_l) <= END_DIST_TOL]
        
        if len(valid_runs) == 0:
            continue

        raw_times = np.array([x["travel_time_raw"] for x in valid_runs], dtype=float)
        time_ref_raw = float(np.median(raw_times))
        mass_median = float(np.nanmedian(np.concatenate([x["mass_values"] for x in valid_runs])))

        # 归一化重采样：把不同长度曲线映射到同一时间网格。
        n_ref = max(int(round(time_ref_raw / DT_SAMPLE)) + 1, 400)
        v_norm_curves = []
        for item in valid_runs:
            v_norm = resample_to_normalized_time(item["raw_v"], n_ref)
            if v_norm is not None: v_norm_curves.append(v_norm)
        
        if len(v_norm_curves) == 0: continue
        v_norm_curves = np.vstack(v_norm_curves)

        # ======= B-Spline “中间线”重构 =======
        # 先取历史速度曲线的逐点中位数，再用 B-Spline 平滑成参考线。
        raw_median = np.median(v_norm_curves, axis=0)
        tau = np.linspace(0.0, 1.0, len(raw_median))
        
        from scipy.interpolate import make_interp_spline
        spline = make_interp_spline(tau, raw_median, k=3)
        v_ref_t = spline(tau)
        
        v_ref_t = np.clip(v_ref_t, 0.0, None)
        v_ref_t[0], v_ref_t[-1] = 0.0, 0.0
        v_ref_t = safe_savgol(v_ref_t, window=31, poly=3)
        v_ref_t = np.clip(v_ref_t, 0.0, None)
        v_ref_t[-1] = 0.0 # 再次锁定
        v_ref_t[0], v_ref_t[-1] = 0.0, 0.0
        # ===============================================

        # 相位识别：确定加速段、中段、制动段边界。
        t_ref = np.linspace(0.0, 1.0, n_ref) * time_ref_raw
        dt_ref = t_ref[1] - t_ref[0]
        idx_a, idx_b, v_peak_ref = detect_phase_boundaries(v_ref_t, dt_ref)

        # 核心工件：保存该等级模板后续生成曲线所需的所有字段。
        current_art = {
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
            "valid_runs_samples": valid_runs  # 把样本存进去供下方绘图使用
        }
        
        multi_artifacts[current_cls] = current_art
        summary_stats.append({
            "class": current_cls,
            "time_s": round(time_ref_raw, 2),
            "samples": len(valid_runs),
            "peak_v_kmh": round(v_peak_ref * 3.6, 2)
        })

    # 5. 最终保存与输出。
    if not multi_artifacts:
        return {"station_pair": station_pair, "status": "failed", "message": "各等级处理均失败"}

    # 保存包含所有等级的大工件。
    with open(output_dir / "multi_class_phase_artifacts.pkl", "wb") as f:
        pickle.dump(multi_artifacts, f)

    # 导出该区间各等级的时间、样本数、峰值速度汇总表。
    pd.DataFrame(summary_stats).to_csv(output_dir / "multi_class_summary.csv", index=False)

    # 绘图逻辑：遍历 multi_artifacts 里的每一个等级。
    for cls_name, art_data in multi_artifacts.items():
        plt.figure(figsize=(10, 6))
        
        # 1. 绘制历史样本背景（灰色线）。
        # 注意：这里直接从刚才存入的字典里拿样本数据。
        for run in art_data["valid_runs_samples"]:
            plt.plot(run["raw_t"], run["raw_v"] * 3.6, color="#95a5a6", alpha=0.15, linewidth=1)
        
        # 2. 绘制 Spline 重构后的参考线（红线）。
        plt.plot(art_data["t_ref"], art_data["v_ref_t"] * 3.6, 
                 label=f"Ref {cls_name} (Spline Refined)", color='red', linewidth=2.5)
        
        plt.title(f"{station_pair} - {cls_name} Reference Curve")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (km/h)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片。
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
    """批量遍历时间表中的区间，提取并保存 ATO 相位模板。"""

    # 确保输出根目录存在。
    ensure_dir(OUTPUT_ROOT)

    # 定位标准时间表和清洗数据目录。
    standard_times_path = resolve_existing_file(STANDARD_TIMES_CANDIDATES, "standard_class_times.csv")
    data_dir = resolve_existing_dir(DATA_DIR_CANDIDATES, "数据")

    # 读取时间表，每一行对应一个站间区间；指定测试数据目录时只保留实际存在的区间。
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

    # 逐区间训练模板。
    for i, row in table.iterrows():
        station_pair = str(row["区段"]).strip()
        # 从当前行提取 class1-class5 的目标时间。
        level_target_times = row_to_level_target_times(row)

        print(f"\n[{i + 1}/{len(table)}] 开始训练区间: {station_pair}")
        # 执行单区间模板提取。
        result = train_one_station_pair(station_pair, level_target_times, data_dir)
        batch_rows.append(result)

        if result["status"] == "success":
            
            #print(f"  ✅ 成功 | class3参考时间 = {result['class3_time_median_s']} s | 样本数 = {result['n_runs_class3']}")
            found_clss = list(pickle.load(open(result["output_dir"] + "/multi_class_phase_artifacts.pkl", "rb")).keys())
            print(f"  ✅ 成功 | 提取等级: {found_clss}")
        else:
            print(f"  ❌ 失败 | {result['status']} | {result['message']}")

    # 输出批量训练汇总。
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
    # 直接运行脚本时执行批量模板提取。
    main()
