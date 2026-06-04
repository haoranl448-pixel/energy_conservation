# -*- coding: utf-8 -*-
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.interpolate import interp1d
from pathlib import Path
import sys, math, warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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
# 师兄数据的主目录
SENIOR_BASE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"
# 物理地图数据（读坡度和曲率）
MAP_DATA_DIR = get_data_dir(PROJECT_ROOT  / "data" / "data_processed")
# 残差模型权重目录
RES_MODEL_BASE = PROJECT_ROOT / "output" / "models" / "nn_results_residual_v2"
# 载重参数表
PARAM_FILE = PROJECT_ROOT / "data" / "static" / f"section_params_trip{TRIP_NO}.csv"

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

    # 3. 加载模型与地图插值
    res_dir = RES_MODEL_BASE / sp
    with open(res_dir/"scaler_x.pkl", "rb") as f: sx = pickle.load(f)
    with open(res_dir/"scaler_y.pkl", "rb") as f: sy = pickle.load(f)
    model = ResidualTransformerV2(len(sx.mean_))
    model.load_state_dict(torch.load(res_dir/"best_res_model.pth", map_location='cpu'))
    model.eval()

    df_map = pd.read_excel(MAP_DATA_DIR / f"results_{sp}.xlsx")
    for c in ['累计位移(m)','gradient','curvature']: df_map[c]=pd.to_numeric(df_map[c],errors='coerce')
    df_map = df_map.dropna(subset=['累计位移(m)'])
    fg = interp1d(df_map['累计位移(m)'], df_map['gradient'], kind='nearest', fill_value="extrapolate")
    fc = interp1d(df_map['累计位移(m)'], df_map['curvature'], kind='nearest', fill_value="extrapolate")

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

    df_params = pd.read_csv(PARAM_FILE).set_index('station_pair')
    energy_menu = []

    # 获取所有站间区间的子文件夹名
    station_folders = [d for d in os.listdir(SENIOR_BASE_DIR) if os.path.isdir(SENIOR_BASE_DIR / d)]
    if os.environ.get("ENERGY_RESULTS_DATA_DIR") or os.environ.get("ENERGY_DATA_DIR"):
        available = station_pairs_available_in_data_dir(MAP_DATA_DIR)
        station_folders = [sp for sp in station_folders if sp in available]
        print(f"按测试数据目录筛选能耗菜单区间: {len(station_folders)} 个。")

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
                    'DP候选阶段': row.get('dp_candidate_stage', 'real_only' if curve_source == 'real' else 'allow_extrapolated')
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
