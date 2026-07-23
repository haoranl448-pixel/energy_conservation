# -*- coding: utf-8 -*-
import os, pickle, hashlib, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.interpolate import interp1d
from pathlib import Path
import sys, math, warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from trip_traceability import (
    load_trip_traceability,
    print_traceability_summary,
    resolve_traceability_manifest,
    select_trip_rows,
    traceability_metadata,
)
warnings.filterwarnings("ignore")

# ===================== 1. 路径与配置 =====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_trip_no() -> int:
    """从主程序传入的环境变量里读取要处理第几趟车。"""

    raw = os.environ.get("ENERGY_TRIP_NO", "1")
    trip_no = int(raw)
    if trip_no < 1:
        raise ValueError("ENERGY_TRIP_NO must be >= 1.")
    return trip_no


def get_data_dir(default_value: Path) -> Path:
    """从主程序传入的 results 数据目录读取 results_*.xlsx；未传入时使用默认目录。"""

    raw = os.environ.get("ENERGY_RESULTS_DATA_DIR") or os.environ.get("ENERGY_DATA_DIR")
    if not raw:
        return default_value
    data_dir = Path(raw)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir


TRIP_NO = get_trip_no()
TRIP_INDEX = TRIP_NO - 1
# 师兄数据的主目录
SENIOR_BASE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"
# 物理地图数据（读坡度和曲率）
MAP_DATA_DIR = get_data_dir(PROJECT_ROOT  / "data" / "data_processed")
TRACEABILITY_MANIFEST = resolve_traceability_manifest(PROJECT_ROOT, MAP_DATA_DIR)
TRIP_TRACEABILITY = load_trip_traceability(TRACEABILITY_MANIFEST, TRIP_NO, direction="UP")
# 残差模型权重目录
RES_MODEL_BASE = PROJECT_ROOT / "output" / "models" / "nn_results_residual_v2"
# 载重参数表
PARAM_TEMPLATE_FILE = PROJECT_ROOT / "data" / "static" / "section_params.csv"
PARAM_FILE = PROJECT_ROOT / "data" / "static" / f"section_params_trip{TRIP_NO}.csv"
MASS_CACHE_DIR = PROJECT_ROOT / "output" / "cache" / "section_params_mass"
MASS_CACHE_VERSION = "trip_mass_v2_traceability"
_MAP_INTERP_CACHE = {}
_RESIDUAL_ASSET_CACHE = {}

# 输出能耗菜单
OUTPUT_MENU = PROJECT_ROOT / "output" / "analysis" / f"ato_class_energy_menu{TRIP_NO}_new_v3.csv"
os.makedirs(OUTPUT_MENU.parent, exist_ok=True)

sys.path.append(str(PROJECT_ROOT))
from src.physics.train_simu import TrainTheoreticalEnergyModel

DT = 0.05
SEQ_LEN_RES = 30


def remove_existing_output_file(path: Path, label: str):
    """Delete a stale output file before rebuilding it."""

    resolved = path.resolve()
    output_root = (PROJECT_ROOT / "output").resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"拒绝删除非 output 目录下的文件: {resolved}") from exc

    if resolved.exists():
        resolved.unlink()
        print(f"已删除旧{label}: {resolved}")

# ===================== 2. 模型定义 (自包含) =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ResidualTransformerV2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, 64)
        self.pos_encoder = PositionalEncoding(64)
        encoder_layer = nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 2)
        self.decoder = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
    def forward(self, x):
        x = self.input_linear(x); x = self.pos_encoder(x)
        return self.decoder(self.transformer(x)[:, -1, :])

# ===================== 3. 核心计算逻辑 =====================

def station_pairs_available_in_data_dir(data_dir: Path) -> set[str]:
    """Return station pairs that have cleaned_*.xlsx or results_*.xlsx in data_dir."""

    pairs = set()
    if not data_dir.exists():
        return pairs
    for fp in data_dir.glob("*.xlsx"):
        for prefix in ("cleaned_", "results_"):
            if fp.stem.startswith(prefix):
                pairs.add(fp.stem[len(prefix):])
    return pairs


def ordered_station_pairs_from_template() -> list[str]:
    """Read the canonical forward-section order from section_params.csv."""

    if not PARAM_TEMPLATE_FILE.exists():
        return []
    df_params = pd.read_csv(PARAM_TEMPLATE_FILE, usecols=["station_pair"])
    return df_params["station_pair"].dropna().astype(str).tolist()


def apply_line_scope(station_pairs: list[str]) -> list[str]:
    """Limit station pairs to ENERGY_LINE_SCOPE when the pipeline requests it."""

    raw = os.environ.get("ENERGY_LINE_SCOPE", "full").strip().lower()
    if raw in {"", "full", "all"}:
        return station_pairs
    if raw == "first5":
        raw = "5"
    try:
        section_count = int(raw)
    except ValueError as exc:
        raise ValueError("ENERGY_LINE_SCOPE must be 'full' or a positive integer section count.") from exc
    if section_count < 1:
        raise ValueError("ENERGY_LINE_SCOPE section count must be >= 1.")

    ordered_pairs = ordered_station_pairs_from_template()
    if not ordered_pairs:
        return station_pairs
    allowed = set(ordered_pairs[:section_count])
    return [sp for sp in station_pairs if sp in allowed]


def get_target_station_folders() -> list[str]:
    """Return generated-result folders that should be evaluated in this run."""

    station_folders = [d for d in os.listdir(SENIOR_BASE_DIR) if os.path.isdir(SENIOR_BASE_DIR / d)]
    if os.environ.get("ENERGY_RESULTS_DATA_DIR") or os.environ.get("ENERGY_DATA_DIR"):
        available = station_pairs_available_in_data_dir(MAP_DATA_DIR)
        station_folders = [sp for sp in station_folders if sp in available]
        print(f"按测试数据目录筛选能耗菜单区间: {len(station_folders)} 个")

    order = ordered_station_pairs_from_template()
    if order:
        order_index = {sp: i for i, sp in enumerate(order)}
        station_folders.sort(key=lambda sp: order_index.get(sp, len(order_index)))
    station_folders = apply_line_scope(station_folders)
    print(f"按 line scope 筛选后能耗菜单区间: {len(station_folders)} 个")
    return station_folders


def get_mass_cache_path(file_path: Path, station_pair: str) -> Path:
    """Build a cache path that changes whenever the source Excel changes."""

    stat = file_path.stat()
    if TRIP_TRACEABILITY is not None:
        record = TRIP_TRACEABILITY.record_for(station_pair)
        selection_key = f"{record.global_trip_id}|{record.segment}|{record.source_run_id}"
    else:
        selection_key = f"legacy_local_index|{TRIP_INDEX}"
    key_src = "|".join([
        MASS_CACHE_VERSION,
        str(file_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        station_pair,
        selection_key,
    ])
    key = hashlib.sha1(key_src.encode("utf-8", errors="surrogatepass")).hexdigest()
    return MASS_CACHE_DIR / f"{key}.pkl"


def extract_trip_mass_from_results(station_pair: str) -> float | None:
    """Return the mean train mass for the selected trip in one section."""

    file_path = MAP_DATA_DIR / f"results_{station_pair}.xlsx"
    if not file_path.exists():
        return None

    cache_path = get_mass_cache_path(file_path, station_pair)
    if cache_path.exists():
        return float(pd.read_pickle(cache_path)["mass"])

    print(f"   读取重量: {station_pair} trip {TRIP_NO}")
    df = pd.read_excel(
        file_path,
        usecols=lambda col: str(col) in {"segment", "重量", "速度(m/s)"},
        engine="openpyxl",
    )
    required = {"segment", "重量"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} 缺少必要列: {sorted(missing)}")

    df["重量"] = pd.to_numeric(df["重量"], errors="coerce")
    if "速度(m/s)" in df.columns:
        df["速度(m/s)"] = pd.to_numeric(df["速度(m/s)"], errors="coerce")

    data, target_seg, record, selection_mode = select_trip_rows(
        df,
        station_pair,
        TRIP_INDEX,
        TRIP_TRACEABILITY,
    )
    print(
        f"      选中 segment={target_seg}"
        + (f", run_id={record.source_run_id}" if record else "")
        + f" [{selection_mode}]"
    )
    if "速度(m/s)" in data.columns:
        data = data.dropna(subset=["速度(m/s)"])

    mass = data["重量"].dropna().mean()
    if pd.isna(mass):
        return None
    MASS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"mass": float(mass)}, cache_path)
    return float(mass)


def build_section_params_for_trip(station_pairs_to_update: list[str] | None = None) -> Path:
    """Generate section_params_tripN.csv from the current Step2 result files."""

    if not PARAM_TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"区间参数模板不存在: {PARAM_TEMPLATE_FILE}")

    df_params = pd.read_csv(PARAM_TEMPLATE_FILE)
    if "station_pair" not in df_params.columns or "MASS" not in df_params.columns:
        raise ValueError(f"{PARAM_TEMPLATE_FILE} 必须包含 station_pair 和 MASS 两列。")

    previous_mass = {}
    if PARAM_FILE.exists():
        try:
            df_previous = pd.read_csv(PARAM_FILE)
            if {"station_pair", "MASS"}.issubset(df_previous.columns):
                previous_mass = dict(zip(df_previous["station_pair"], df_previous["MASS"]))
        except Exception as exc:
            print(f"⚠️ 读取旧参数表失败，将仅使用模板值: {exc}")

    update_set = set(station_pairs_to_update) if station_pairs_to_update is not None else None
    updated, skipped, reused_old, kept_template, failed = 0, 0, 0, 0, []
    for column in ["全局趟次ID", "历史segment", "历史来源run_id", "趟次选择方式"]:
        if column not in df_params.columns:
            df_params[column] = ""

    for idx, station_pair in df_params["station_pair"].items():
        if TRIP_TRACEABILITY is not None and station_pair in TRIP_TRACEABILITY.records:
            record = TRIP_TRACEABILITY.record_for(station_pair)
            metadata = traceability_metadata(record, record.segment, "traceability")
            for column, value in metadata.items():
                df_params.at[idx, column] = value
        else:
            df_params.at[idx, "趟次选择方式"] = "legacy_local_index"

        if update_set is not None and station_pair not in update_set:
            mass = None
            skipped += 1
        else:
            try:
                mass = extract_trip_mass_from_results(station_pair)
            except Exception as exc:
                failed.append((station_pair, str(exc)))
                mass = None

        if mass is not None:
            df_params.at[idx, "MASS"] = round(mass, 4)
            updated += 1
        elif station_pair in previous_mass and pd.notna(previous_mass[station_pair]):
            df_params.at[idx, "MASS"] = previous_mass[station_pair]
            reused_old += 1
        else:
            kept_template += 1

    PARAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_params.to_csv(PARAM_FILE, index=False, encoding="utf-8-sig")
    print(
        f"已生成区间参数表: {PARAM_FILE} | 更新 MASS: {updated} | "
        f"跳过非本次区间: {skipped} | 沿用旧值: {reused_old} | 保留模板值: {kept_template}"
    )
    for station_pair, reason in failed[:5]:
        print(f"   ⚠️ {station_pair} MASS 未更新: {reason}")
    if len(failed) > 5:
        print(f"   ⚠️ 另有 {len(failed) - 5} 个区间 MASS 未更新。")
    return PARAM_FILE


def get_map_interpolators(sp: str):
    """Load gradient/curvature interpolators once per station pair."""

    if sp in _MAP_INTERP_CACHE:
        return _MAP_INTERP_CACHE[sp]

    print(f"   读取坡度曲率: {sp}")
    df_map = pd.read_excel(
        MAP_DATA_DIR / f"results_{sp}.xlsx",
        usecols=lambda col: str(col) in {"累计位移(m)", "gradient", "curvature"},
        engine="openpyxl",
    )
    for c in ['累计位移(m)', 'gradient', 'curvature']:
        df_map[c] = pd.to_numeric(df_map[c], errors='coerce')
    df_map = df_map.dropna(subset=['累计位移(m)'])
    fg = interp1d(df_map['累计位移(m)'], df_map['gradient'], kind='nearest', fill_value="extrapolate")
    fc = interp1d(df_map['累计位移(m)'], df_map['curvature'], kind='nearest', fill_value="extrapolate")
    _MAP_INTERP_CACHE[sp] = (fg, fc)
    return fg, fc


def get_residual_assets(sp: str):
    """Load residual model and scalers once per station pair."""

    if sp in _RESIDUAL_ASSET_CACHE:
        return _RESIDUAL_ASSET_CACHE[sp]

    print(f"   加载残差模型: {sp}")
    res_dir = RES_MODEL_BASE / sp
    with open(res_dir / "scaler_x.pkl", "rb") as f:
        sx = pickle.load(f)
    with open(res_dir / "scaler_y.pkl", "rb") as f:
        sy = pickle.load(f)
    model = ResidualTransformerV2(len(sx.mean_))
    model.load_state_dict(torch.load(res_dir / "best_res_model.pth", map_location='cpu'))
    model.eval()
    _RESIDUAL_ASSET_CACHE[sp] = (sx, sy, model)
    return sx, sy, model


def get_energy(sp, class_csv_path, mass_val):
    """根据师兄的轨迹CSV计算[物理+AI]总能耗"""
    # 1. 读取轨迹 (time_s, dist_m, velocity_mps)
    df_traj = pd.read_csv(class_csv_path)
    t_arr = df_traj['time_s'].values
    v_arr = df_traj['velocity_mps'].values

    s_arr = df_traj['dist_m'].values
    a_arr = np.zeros_like(v_arr)
    if len(v_arr) > 1: a_arr[1:] = np.diff(v_arr) / DT



    # 2. 物理仿真
    engine = TrainTheoreticalEnergyModel()
    e_phy_steps = engine.run_batch_simulation(t_arr, v_arr, mass_val)

    # 3. 加载模型与地图插值。同一区间内多个等级共用，避免重复读取大 Excel / pkl / pth。
    sx, sy, model = get_residual_assets(sp)
    fg, fc = get_map_interpolators(sp)

    # 4. 残差预测
    raw_X = np.stack([v_arr, a_arr, e_phy_steps, fg(s_arr), np.full_like(v_arr, mass_val), fc(s_arr)], axis=1)
    scaled_X = sx.transform(raw_X)
    windows = [scaled_X[i-SEQ_LEN_RES+1:i+1] for i in range(SEQ_LEN_RES-1, len(scaled_X))]
    res_sum = 0.0
    if windows:
        with torch.no_grad():
            p_out = model(torch.tensor(np.array(windows), dtype=torch.float32)).numpy()
        res_sum = np.sum(sy.inverse_transform(p_out))

    return np.sum(e_phy_steps[29:]) + res_sum, np.sum(e_phy_steps[29:]), res_sum,a_arr

# ===================== 4. 主流程 =====================

def main():
    remove_existing_output_file(OUTPUT_MENU, "能耗菜单")
    station_folders = get_target_station_folders()
    print_traceability_summary(TRIP_TRACEABILITY, TRIP_NO)
    if TRIP_TRACEABILITY is not None:
        TRIP_TRACEABILITY.require_sections(station_folders)
    build_section_params_for_trip(station_folders)

    df_params = pd.read_csv(PARAM_FILE).set_index('station_pair')
    energy_menu = []

    print(f"🚀 开始批量评估 {len(station_folders)} 个区间的 ATO 轨迹能耗：第 {TRIP_NO} 趟车...")
    print(f"地图/历史数据目录: {MAP_DATA_DIR}")

    for sp in station_folders:
        summary_path = SENIOR_BASE_DIR / sp / "all_classes_summary.csv"
        if not summary_path.exists():
            continue

        df_summary = pd.read_csv(summary_path)
        # 🌟 过滤：只处理 generated 的，跳过 overspeed
        df_valid = df_summary[df_summary['status'] == 'generated']

        if sp not in df_params.index:
            print(f"   ⚠️ 参数表中缺少 {sp}，跳过")
            continue

        mass_val = df_params.loc[sp, 'MASS']

        for _, row in df_valid.iterrows():
            c_name = row['class_name']
            # 拼接详细轨迹文件名
            class_csv_path = SENIOR_BASE_DIR / sp / f"{c_name}_generated_curve.csv"

            if not class_csv_path.exists():
                continue

            try:
                e_pred ,e_phy,res_sum,a_arr= get_energy(sp, class_csv_path, mass_val)
                curve_source = row.get('curve_source', 'real')
                energy_menu.append({
                    '站间区间': sp,
                    '运行等级': c_name,
                    '运行时长(s)': round(row['sim_time_s'],1),
                    '预测能耗(Wh)': round(e_pred, 2),
                    '峰值速度(kmh)': row['peak_speed_kmh'],
                    '曲线来源': curve_source,
                    '父等级': row.get('parent_ref', c_name),
                    '支持等级': row.get('supporting_refs', c_name),
                    '真实样本数': int(row.get('real_sample_count', 0)),
                    '父等级真实样本数': int(row.get('parent_real_sample_count', row.get('real_sample_count', 0))),
                    '样本可靠性': row.get('sample_reliability', ''),
                    'DP候选阶段': row.get('dp_candidate_stage', 'real_only' if curve_source == 'real' else 'allow_extrapolated'),
                    '全局趟次ID': df_params.loc[sp].get('全局趟次ID', ''),
                    '历史segment': df_params.loc[sp].get('历史segment', ''),
                    '历史来源run_id': df_params.loc[sp].get('历史来源run_id', ''),
                    '趟次选择方式': df_params.loc[sp].get('趟次选择方式', ''),
                })
                print(f"   ✅ {sp} | {c_name} | {row['sim_time_s']}s -> 总:{e_pred:.1f}Wh (物理:{e_phy:.1f}Wh + 残差:{res_sum:.1f}Wh)")
            except Exception as e:
                print(f"   ❌ {sp} {c_name} 失败: {e}")

    # 保存最终菜单
    df_menu = pd.DataFrame(energy_menu)
    df_menu.to_csv(OUTPUT_MENU, index=False, encoding='utf-8-sig')
    print(f"\n✨ 评估完成！能耗菜单已生成：{OUTPUT_MENU}")
    print("👉 现在你可以运行 DP 调度算法来选择最优 Class 组合了。")

if __name__ == "__main__":
    main()
