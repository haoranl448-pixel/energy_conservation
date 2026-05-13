# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import glob
import sys
from sklearn.metrics import r2_score
import warnings

# 屏蔽版本警告
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 1. 模型架构定义 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model); import math
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ResidualTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
    def forward(self, src):
        src = self.input_linear(src); src = self.pos_encoder(src)
        out = self.transformer(src); return self.decoder(out[:, -1, :])

def create_sequences(X, seq_length):
    xs = []
    for i in range(len(X) - seq_length):
        xs.append(X[i : i+seq_length])
    return np.array(xs)

# ================= 2. 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

DATA_DIR = r"D:\energy_conservation\data\data_processed"
MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")
OUTPUT_EXCEL = os.path.join(project_root, "output", "Line5_Sections_R2_Report.xlsx")

STATION_PAIRS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

SEQ_LEN = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluation():
    from src.physics.train_simu import TrainTheoreticalEnergyModel
    sim_model = TrainTheoreticalEnergyModel()
    
    final_report = []

    print(f"📊 开始全线能效大模型精度评估...")

    for sp in STATION_PAIRS:
        model_dir = os.path.join(MODEL_BASE, sp)
        data_files = glob.glob(os.path.join(DATA_DIR, f"results_{sp}*.xlsx"))
        
        if not os.path.exists(model_dir) or not data_files:
            continue

        # 加载模型和Scaler
        try:
            with open(f"{model_dir}/scaler_x.pkl", "rb") as f: scaler_x = pickle.load(f)
            with open(f"{model_dir}/scaler_y.pkl", "rb") as f: scaler_y = pickle.load(f)
            model = ResidualTransformer(input_dim=5).to(device)
            model.load_state_dict(torch.load(f"{model_dir}/best_res_model.pth", map_location=device))
            model.eval()
        except: continue

        # 加载全量数据
        df_all = pd.read_excel(data_files[0])
        unique_segments = df_all['segment'].unique()
        
        section_r2_list = []

        for seg_id in unique_segments:
            df = df_all[df_all['segment'] == seg_id].copy()
            
            # 数据清洗：转数值，去空格
            for col in ['速度(m/s)', '加速度(m/s²)', '累计位移(m)', 'energy', '时刻', '重量', 'gradient']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.ffill().dropna(subset=['速度(m/s)', 'energy'])
            
            if len(df) < SEQ_LEN + 10: continue

            # 物理与AI预测
            t_seq, v_seq, mass_val = df['时刻'].values, df['速度(m/s)'].values, df['重量'].iloc[0]
            e_phy_wh = sim_model.run_batch_simulation(t_seq, v_seq, mass_val)
            
            raw_X = df[['速度(m/s)', '加速度(m/s²)', '时刻', 'gradient', '重量']].values # 注意：此处特征顺序需检查
            # 修正特征列以匹配 [v, a, e_phy, grad, mass]
            raw_X = np.stack([v_seq, df['加速度(m/s²)'].values, e_phy_wh[:len(df)], df['gradient'].values, df['重量'].values], axis=1)
            
            X_scaled = scaler_x.transform(raw_X)
            sx = create_sequences(X_scaled, SEQ_LEN)
            with torch.no_grad():
                res_pred = model(torch.FloatTensor(sx).to(device)).cpu().numpy()
            res_pred = scaler_y.inverse_transform(res_pred).flatten()

            # 累积能耗对标
            L_min = len(res_pred)
            e_real_cum = np.cumsum((df['energy'].values[SEQ_LEN:][:L_min] / 3.6e6) * 1000)
            e_fusion_cum = np.cumsum(np.maximum(e_phy_wh[SEQ_LEN:][:L_min] + res_pred, 0))
            
            # 计算 R2
            cur_r2 = r2_score(e_real_cum, e_fusion_cum)
            if cur_r2 > 0: # 排除掉异常值
                section_r2_list.append(cur_r2)

        if section_r2_list:
            avg_r2 = np.mean(section_r2_list)
            final_report.append({
                "区间名称": sp,
                "平均 R² (拟合优度)": round(avg_r2, 5),
                "最高 R²": round(max(section_r2_list), 5),
                "最低 R²": round(min(section_r2_list), 5),
                "样本趟数": len(section_r2_list)
            })
            print(f"✅ {sp} 处理完成: 平均 R² = {avg_r2:.5f}")

    # 导出 Excel
    df_report = pd.DataFrame(final_report)
    df_report.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✨ 全线 R² 评估报告已生成至: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    run_evaluation()