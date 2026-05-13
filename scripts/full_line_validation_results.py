# -*- coding: utf-8 -*-
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.interpolate import interp1d
from pathlib import Path
import sys
import math
# ===================== 1. 路径与配置 =====================
PROJECT_ROOT = Path(r"D:\energy_conservation")
DATA_DIR = PROJECT_ROOT / "data" / "data_processed"
RES_MODEL_BASE = PROJECT_ROOT / "output" / "models" / "nn_results_residual_v2"
TRIP_INDEX = 0  # 第1趟车

# 引入你的物理引擎
sys.path.append(str(PROJECT_ROOT))
from src.physics.train_simu import TrainTheoreticalEnergyModel

# ... (此处省略 ResidualTransformerV2 和 PositionalEncoding 的定义，保持和你训练时一致) ...

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
def evaluate_historical_trip(sp):
    # 1. 读取该区间的历史 results 文件
    file_path = DATA_DIR / f"results_{sp}.xlsx"
    if not file_path.exists(): return None
    
    df_all = pd.read_excel(file_path)
    # 确保数值化
    cols = ['速度(m/s)', '加速度(m/s²)', '累计位移(m)', 'energy', '时刻', '重量', 'gradient', 'curvature']
    for c in cols: df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
    
    # 2. 提取第6趟 (Trip 6)
    segs = sorted(df_all['segment'].unique())
    target_seg = segs[TRIP_INDEX]
    df = df_all[df_all['segment'] == target_seg].copy().dropna(subset=['速度(m/s)'])
    
    t_seq = df['时刻'].values - df['时刻'].iloc[0]
    v_seq = df['速度(m/s)'].values
    a_seq = df['加速度(m/s²)'].values
    s_seq = df['累计位移(m)'].values
    mass_val = df['重量'].iloc[0]

    # 3. 重新运行物理仿真 (获取模型需要的 e_phy 特征)
    sim_model = TrainTheoreticalEnergyModel()
    e_phy_steps = sim_model.run_batch_simulation(t_seq, v_seq, mass_val)

    # 4. 加载 AI 模型
    model_dir = RES_MODEL_BASE / sp
    with open(model_dir/"scaler_x.pkl", "rb") as f: sx = pickle.load(f)
    with open(model_dir/"scaler_y.pkl", "rb") as f: sy = pickle.load(f)
    model = ResidualTransformerV2(6) # 确认是 6 特征
    model.load_state_dict(torch.load(model_dir/"best_res_model.pth", map_location='cpu'))
    model.eval()

    # 5. 构建特征矩阵 [v, a, e_phy, grad, mass, curv]
    # 严格匹配你截图里的特征顺序
    raw_X = np.stack([
        v_seq, 
        a_seq, 
        e_phy_steps[:len(df)], 
        df['gradient'].values, 
        df['重量'].values, 
        df['curvature'].values
    ], axis=1)
    
    scaled_X = sx.transform(raw_X)
    windows = [scaled_X[i-29:i+1] for i in range(29, len(scaled_X))]
    
    res_sum = 0.0
    if windows:
        with torch.no_grad():
            p_out = model(torch.tensor(np.array(windows), dtype=torch.float32)).numpy()
        res_sum = np.sum(sy.inverse_transform(p_out))

    # 6. 计算结果
    e_real_total = (df['energy'].sum() / 3.6e6) * 1000 # 历史实测 Wh
    e_model_total = np.sum(e_phy_steps[29:]) + res_sum # 模型预测 Wh
    
    return {
        "区间": sp,
        "历史耗时(s)": round(t_seq[-1], 2),
        "历史实测能耗(Wh)": round(e_real_total, 2),
        "模型预测能耗(Wh)": round(e_model_total, 2),
        "物理占比": round(np.sum(e_phy_steps[29:]), 2),
        "残差占比": round(res_sum, 2),
        "误差(%)": round((e_model_total - e_real_total)/e_real_total*100, 2)
    }

if __name__ == "__main__":
    # 1. 定义全线 26 个正向区间清单
    line5_stations = [
        "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
        "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
        "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
        "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
        "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
        "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
    ]

    results = []
    print(f"🚀 开始全线 {len(line5_stations)} 个区间的历史数据验证...")

    # 2. 循环处理每一个区间
    for i, sp in enumerate(line5_stations, 1):
        # 打印进度，让控制台有反馈
        print(f"[{i:02d}/{len(line5_stations)}] 正在处理: {sp} ...", end='\r')
        
        try:
            # 执行你定义的评估函数
            res = evaluate_historical_trip(sp)
            if res:
                results.append(res)
        except Exception as e:
            # 如果某站数据缺失或报错，打印出来并跳过，不影响整体循环
            print(f"\n❌ {sp} 处理时发生意外错误: {e}")

    # 3. 汇总展示结果
    print("\n\n" + "="*60)
    print("🔥 5号线全线 26 区间验证结果汇总")
    print("="*60)

    if results:
        df_final = pd.DataFrame(results)
        # 强制显示所有行，防止中间被省略号遮盖
        pd.set_option('display.max_rows', None) 
        print(df_final.to_string(index=False))
        
        # 建议：顺便保存一份 CSV 结果，方便你填 PPT 或者写报告
        df_final.to_csv("full_line1_validation_results.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ 结果已保存至: full_line1_validation_results.csv")
    else:
        print("⚠️ 未提取到任何有效数据，请检查 data_processed 目录下的文件。")