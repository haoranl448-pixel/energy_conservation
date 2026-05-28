# -*- coding: utf-8 -*-
"""
主线第 5 步：计算生成曲线的能耗菜单。

这个脚本会遍历 ATO 曲线生成目录：
1. 读取每个区间、每个 class 的 generated_curve.csv。
2. 用物理模型计算理论牵引能耗。
3. 用残差 Transformer 修正物理模型偏差。
4. 汇总成 DP 排图需要的“区间-等级-运行时间-预测能耗”菜单。
"""

# os/pickle：处理路径、读取 scaler。
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
# interp1d：按位移把坡度和曲率插值到生成曲线的采样点。
from scipy.interpolate import interp1d
# Path：统一处理项目路径。
from pathlib import Path
# sys/math/warnings：导入项目模块、构造位置编码、屏蔽非关键警告。
import sys, math, warnings
# 以下两个导入是历史保留；当前主流程里未直接绘图/滤波。
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")

# ===================== 1. 路径与配置 =====================
# 项目根目录；在 scripts_new 二级目录下直接运行时需要留意路径层级。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_trip_no() -> int:
    """从主程序传入的环境变量里读取要处理第几趟车。"""

    raw = os.environ.get("ENERGY_TRIP_NO", "1")
    trip_no = int(raw)
    if trip_no < 1:
        raise ValueError("ENERGY_TRIP_NO must be >= 1.")
    return trip_no


def get_data_dir(default_value: Path) -> Path:
    """从主程序传入的数据目录读取 results_*.xlsx；未传入时使用默认目录。"""

    raw = os.environ.get("ENERGY_DATA_DIR")
    if not raw:
        return default_value
    data_dir = Path(raw)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir


TRIP_NO = get_trip_no()
# 生成速度曲线的主目录，每个站间区间一个子文件夹。
SENIOR_BASE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"
# 物理地图数据目录，用于读取坡度 gradient 和曲率 curvature。
MAP_DATA_DIR = get_data_dir(PROJECT_ROOT  / "data" / "data_processed")
# 残差模型权重目录，内部按站间区间分子文件夹。
RES_MODEL_BASE = PROJECT_ROOT / "output" / "models" / "nn_results_residual_v2"
# 载重参数表，提供每个区间的 MASS。
PARAM_FILE = PROJECT_ROOT / "data" / "static" / f"section_params_trip{TRIP_NO}.csv"

# 输出能耗菜单，供 DP 排图脚本读取。
OUTPUT_MENU = PROJECT_ROOT / "output" / "analysis" / f"ato_class_energy_menu{TRIP_NO}_new_v3.csv"
os.makedirs(OUTPUT_MENU.parent, exist_ok=True)

# 将项目根目录加入 import 路径，便于导入 src.physics。
sys.path.append(str(PROJECT_ROOT))
# 物理能耗模型：Davis 方程 + 坡度/曲率/惯性项。
from src.physics.train_simu import TrainTheoreticalEnergyModel

# 采样间隔，需与生成曲线和模型训练保持一致。
DT = 0.05
# 残差 Transformer 的输入窗口长度。
SEQ_LEN_RES = 30

# ===================== 2. 模型定义 (自包含) =====================
class PositionalEncoding(nn.Module):
    """Transformer 位置编码，与训练残差模型时的结构保持一致。"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # pe 的形状为 [max_len, d_model]。
        pe = torch.zeros(max_len, d_model)
        # position 表示时间步位置。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term 控制不同维度的正弦/余弦频率。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数维使用 sin，奇数维使用 cos。
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer 表示这是模型状态，但不参与训练。
        self.register_buffer('pe', pe.unsqueeze(0))
    # 前向传播时把位置编码加到输入特征上。
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ResidualTransformerV2(nn.Module):
    """残差能耗预测模型，输出物理能耗之外的修正量。"""

    def __init__(self, input_dim):
        super().__init__()
        # 把原始输入特征映射到 Transformer 的 d_model=64 维空间。
        self.input_linear = nn.Linear(input_dim, 64)
        # 加入时间位置信息。
        self.pos_encoder = PositionalEncoding(64)
        # 两层 TransformerEncoder，与训练权重结构保持一致。
        encoder_layer = nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 2)
        # 解码成单步残差能耗。
        self.decoder = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
    def forward(self, x):
        # 输入线性映射。
        x = self.input_linear(x); x = self.pos_encoder(x)
        # 只取最后一个时间步的编码作为当前时刻预测依据。
        return self.decoder(self.transformer(x)[:, -1, :])

# ===================== 3. 核心计算逻辑 =====================

def get_energy(sp, class_csv_path, mass_val):
    """根据单条生成轨迹计算“物理 + AI 残差”总能耗。"""

    # 1. 读取轨迹 (time_s, dist_m, velocity_mps)
    df_traj = pd.read_csv(class_csv_path)
    # 时间序列。
    t_arr = df_traj['time_s'].values
    # 速度序列，单位 m/s。
    v_arr = df_traj['velocity_mps'].values
    # 位移序列，单位 m。
    s_arr = df_traj['dist_m'].values
    # 加速度由速度差分得到，第一项置 0。
    a_arr = np.zeros_like(v_arr)
    if len(v_arr) > 1: a_arr[1:] = np.diff(v_arr) / DT

    # 2. 物理仿真
    # 物理模型输出每个采样步的理论能耗 Wh。
    engine = TrainTheoreticalEnergyModel()
    e_phy_steps = engine.run_batch_simulation(t_arr, v_arr, mass_val)
    
    # 3. 加载模型与地图插值
    # 每个区间有独立残差模型和 scaler。
    res_dir = RES_MODEL_BASE / sp
    with open(res_dir/"scaler_x.pkl", "rb") as f: sx = pickle.load(f)
    with open(res_dir/"scaler_y.pkl", "rb") as f: sy = pickle.load(f)
    # 模型输入维度由 scaler_x 记录的特征数确定。
    model = ResidualTransformerV2(len(sx.mean_))
    model.load_state_dict(torch.load(res_dir/"best_res_model.pth", map_location='cpu'))
    model.eval()

    # 读取该区间处理后的真实数据，用于把坡度/曲率映射到生成曲线位移点。
    df_map = pd.read_excel(MAP_DATA_DIR / f"results_{sp}.xlsx")
    # 将关键列转为数值，无法转换的设为 NaN。
    for c in ['累计位移(m)','gradient','curvature']: df_map[c]=pd.to_numeric(df_map[c],errors='coerce')
    # 位移为空的行无法参与插值。
    df_map = df_map.dropna(subset=['累计位移(m)'])
    # 坡度插值函数。
    fg = interp1d(df_map['累计位移(m)'], df_map['gradient'], kind='nearest', fill_value="extrapolate")
    # 曲率插值函数。
    fc = interp1d(df_map['累计位移(m)'], df_map['curvature'], kind='nearest', fill_value="extrapolate")

    # 4. 残差预测
    # 残差模型特征：速度、加速度、物理步能耗、坡度、载重、曲率。
    raw_X = np.stack([v_arr, a_arr, e_phy_steps, fg(s_arr), np.full_like(v_arr, mass_val), fc(s_arr)], axis=1)
    # 使用训练时保存的 scaler 做输入归一化。
    scaled_X = sx.transform(raw_X)
    # 构造长度为 30 的滑动窗口。
    windows = [scaled_X[i-SEQ_LEN_RES+1:i+1] for i in range(SEQ_LEN_RES-1, len(scaled_X))]
    # 默认残差为 0，若轨迹太短则直接使用物理能耗。
    res_sum = 0.0
    if windows:
        # 推理阶段关闭梯度计算。
        with torch.no_grad():
            p_out = model(torch.tensor(np.array(windows), dtype=torch.float32)).numpy()
        # 残差输出反归一化并求和。
        res_sum = np.sum(sy.inverse_transform(p_out))
    
    # 前 29 个点没有完整残差窗口，因此物理能耗也从第 30 个点起对齐。
    return np.sum(e_phy_steps[29:]) + res_sum, np.sum(e_phy_steps[29:]), res_sum,a_arr

# ===================== 4. 主流程 =====================

def main():
    """批量遍历所有区间和等级，生成 DP 所需能耗菜单。"""

    # 读取区间载重参数，以 station_pair 为索引。
    df_params = pd.read_csv(PARAM_FILE).set_index('station_pair')
    # 保存最终菜单的行记录。
    energy_menu = []

    # 获取所有站间区间的子文件夹名
    station_folders = [d for d in os.listdir(SENIOR_BASE_DIR) if os.path.isdir(SENIOR_BASE_DIR / d)]

    print(f"🚀 开始批量评估 {len(station_folders)} 个区间的 ATO 轨迹能耗：第 {TRIP_NO} 趟车...")
    print(f"地图/历史数据目录: {MAP_DATA_DIR}")

    for sp in station_folders:
        # all_classes_summary.csv 记录该区间各等级曲线是否生成成功。
        summary_path = SENIOR_BASE_DIR / sp / "all_classes_summary.csv"
        if not summary_path.exists():
            continue
            
        df_summary = pd.read_csv(summary_path)
        # 只处理 generated 状态，跳过 overspeed 或失败曲线。
        df_valid = df_summary[df_summary['status'] == 'generated']
        
        # 参数表缺少该区间时无法取载重，跳过并提示。
        if sp not in df_params.index:
            print(f"   ⚠️ 参数表中缺少 {sp}，跳过")
            continue
        
        # 当前区间载重。
        mass_val = df_params.loc[sp, 'MASS']

        # 遍历该区间所有可用等级曲线。
        for _, row in df_valid.iterrows():
            # class1-class5。
            c_name = row['class_name']
            # 拼接详细轨迹文件名。
            class_csv_path = SENIOR_BASE_DIR / sp / f"{c_name}_generated_curve.csv"
            
            # 若详细轨迹不存在，则无法计算能耗。
            if not class_csv_path.exists():
                continue

            try:
                # 计算该区间该等级的预测总能耗。
                e_pred ,e_phy,res_sum,a_arr= get_energy(sp, class_csv_path, mass_val)
                # 追加菜单行，供 DP 排图读取。
                energy_menu.append({
                    '站间区间': sp,
                    '运行等级': c_name,
                    '运行时长(s)': round(row['sim_time_s'],1),
                    '预测能耗(Wh)': round(e_pred, 2),
                    '峰值速度(kmh)': row['peak_speed_kmh']
                })
                print(f"   ✅ {sp} | {c_name} | {row['sim_time_s']}s -> 总:{e_pred:.1f}Wh (物理:{e_phy:.1f}Wh + 残差:{res_sum:.1f}Wh)")
            except Exception as e:
                # 单条曲线计算失败不影响其他区间/等级继续处理。
                print(f"   ❌ {sp} {c_name} 失败: {e}")

    # 保存最终菜单。
    df_menu = pd.DataFrame(energy_menu)
    df_menu.to_csv(OUTPUT_MENU, index=False, encoding='utf-8-sig')
    print(f"\n✨ 评估完成！能耗菜单已生成：{OUTPUT_MENU}")
    print("👉 现在你可以运行 DP 调度算法来选择最优 Class 组合了。")

if __name__ == "__main__":
    # 直接运行脚本时，生成能耗菜单。
    main()
