# # -*- coding: utf-8 -*-
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import glob
# import sys
# from sklearn.metrics import r2_score

# # ================= 1. 模型架构定义 =================
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         import math
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))
#     def forward(self, x): return x + self.pe[:, :x.size(1), :]

# class ResidualTransformer(nn.Module):
#     def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
#         super().__init__()
#         self.input_linear = nn.Linear(input_dim, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
#         self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
#     def forward(self, src):
#         src = self.input_linear(src)
#         src = self.pos_encoder(src)
#         out = self.transformer(src)
#         return self.decoder(out[:, -1, :])

# def create_sequences(X, seq_length):
#     xs = []
#     for i in range(len(X) - seq_length):
#         xs.append(X[i : i+seq_length])
#     return np.array(xs)

# # ================= 2. 参数与路径设置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# # 确保这些路径是你电脑上的实际路径
# DATA_DIR = r"D:\energy_conservation\data\data_processed"
# MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")
# # 结果保存的总目录
# SAVE_BASE = os.path.join(project_root, "output", "validation_all_trips")

# SEQ_LEN = 30
# DT = 0.05
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 全线 26 个区间列表
# STATION_PAIRS = [
#      "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰"
# ]

# def batch_process_plots():
#     from src.physics.train_simu import TrainTheoreticalEnergyModel
#     sim_model = TrainTheoreticalEnergyModel()

#     # 设置绘图中文字体
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
#     plt.rcParams['axes.unicode_minus'] = False

#     for sp in STATION_PAIRS:
#         # 1. 检查模型和数据是否存在
#         model_dir = os.path.join(MODEL_BASE, sp)
#         data_files = glob.glob(os.path.join(DATA_DIR, f"results_{sp}*.xlsx"))
        
#         if not os.path.exists(model_dir) or not data_files:
#             print(f"⚠️ 跳过 {sp}: 缺少模型或数据文件")
#             continue
        
#         # 创建该区间的专用保存目录
#         sp_save_dir = os.path.join(SAVE_BASE, sp)
#         os.makedirs(sp_save_dir, exist_ok=True)

#         print(f"🚀 正在处理区间: {sp}")
        
#         # 2. 加载数据
#         df_all = pd.read_excel(data_files[0])
#         unique_segments = sorted(df_all['segment'].unique())
        
#         # 3. 加载对应的神经网络模型和标准化器
#         try:
#             with open(f"{model_dir}/scaler_x.pkl", "rb") as f: scaler_x = pickle.load(f)
#             with open(f"{model_dir}/scaler_y.pkl", "rb") as f: scaler_y = pickle.load(f)
#             model = ResidualTransformer(input_dim=5).to(device)
#             model.load_state_dict(torch.load(f"{model_dir}/best_res_model.pth", map_location=device))
#             model.eval()
#         except Exception as e:
#             print(f"❌ 加载 {sp} 模型失败: {e}")
#             continue

#         # 4. 遍历该区间的所有趟次 (Segments)
#         for seg_id in unique_segments:
#             df = df_all[df_all['segment'] == seg_id].copy()
#             if len(df) < SEQ_LEN + 10: continue # 过滤掉太短的数据段
#             # === 新增：强制清洗脏数据，防止字符串或空格导致崩溃 ===
#             cols_to_clean = ['速度(m/s)', '加速度(m/s²)', '累计位移(m)', 'energy', '时刻']
#             for col in cols_to_clean:
#                 if col in df.columns:
#             # 将非数字内容（如空格）转为 NaN，然后前向填充或填0
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # 剔除掉关键列含有空值的行（或用 fillna(0) 填充）
#             df = df.dropna(subset=['累计位移(m)', '速度(m/s)'])
#     # ===================================================

#             if len(df) < SEQ_LEN + 10: continue # 过滤掉太短的数据段




#             # 计算物理基准能耗
#             t_seq = df['时刻'].values
#             v_seq = df['速度(m/s)'].values
#             mass_val = df['重量'].iloc[0]
#             e_phy_wh = sim_model.run_batch_simulation(t_seq, v_seq, mass_val)
#             df['energy_phy'] = e_phy_wh[:len(df)]
            
#             # 准备 AI 预测特征
#             raw_X = df[['速度(m/s)', '加速度(m/s²)', 'energy_phy', 'gradient', '重量']].fillna(0).values
#             X_scaled = scaler_x.transform(raw_X)
#             sx = create_sequences(X_scaled, SEQ_LEN)
            
#             with torch.no_grad():
#                 res_pred = model(torch.FloatTensor(sx).to(device)).cpu().numpy()
#             res_pred = scaler_y.inverse_transform(res_pred).flatten()

#             # 5. 结果计算
#             L_min = len(res_pred)
#             t_plot = np.arange(L_min) * DT
#             v_plot = v_seq[SEQ_LEN:][:L_min] * 3.6 # km/h
#             s_plot = df['累计位移(m)'].values[SEQ_LEN:][:L_min]
            
#             e_real_instant = (pd.to_numeric(df['energy'], errors='coerce').fillna(0).values[SEQ_LEN:][:L_min] / 3.6e6) * 1000
#             e_real_cum = np.cumsum(e_real_instant)
            
#             e_fusion_instant = np.maximum(df['energy_phy'].values[SEQ_LEN:][:L_min] + res_pred, 0)
#             e_fusion_cum = np.cumsum(e_fusion_instant)
            
#             # 计算 R2
#             r2_val = r2_score(e_real_cum, e_fusion_cum)

#             # 6. 绘图
#             fig, ax1 = plt.subplots(figsize=(12, 7))
#             plt.subplots_adjust(right=0.82)

#             # 左轴: 速度
#             ax1.set_xlabel('时间 Time (s)', fontsize=10)
#             ax1.set_ylabel('速度 Velocity (km/h)', fontsize=10)
#             lns1 = ax1.plot(t_plot, v_plot, color='#3498db', linewidth=2, label='运行速度 (v)')
#             ax1.grid(True, alpha=0.3)

#             # 右轴 1: 累积能耗
#             ax2 = ax1.twinx()
#             ax2.set_ylabel('累积能耗 Cumulative Energy (Wh)', fontsize=10)
#             lns2 = ax2.plot(t_plot, e_real_cum, color='black', linewidth=1.5, label='实测累积能耗')
#             # 标题处使用 LaTeX 语法显示 R^2，彻底解决乱码
#             lns3 = ax2.plot(t_plot, e_fusion_cum, color='#e74c3c', linestyle='--', linewidth=2, label='区间预测能耗')

#             # 右轴 2: 累计位移
#             ax3 = ax1.twinx()
#             ax3.spines['right'].set_position(('outward', 65))
#             ax3.set_ylabel('累计位移 Cumulative Distance (m)', fontsize=10)
#             lns4 = ax3.plot(t_plot, s_plot, color='#2ecc71', linewidth=2, label='累计位移 (s)')

#             # 合并图例
#             lns = lns1 + lns2 + lns3 + lns4
#             labs = [l.get_label() for l in lns]
#             ax1.legend(lns, labs, loc='upper left', frameon=True, shadow=True, fontsize=9)

#             # 标题使用 LaTeX 解决 R^2 显示问题
#             plt.title(f"区间能效对标图 - {sp} (Segment {seg_id})\n$R^2 = {r2_val:.4f}$", fontsize=13)
            
#             # 保存
#             save_path = os.path.join(sp_save_dir, f"trip_{seg_id}_validation.png")
#             plt.savefig(save_path, dpi=200, bbox_inches='tight')
#             plt.close(fig) # 必须关闭，否则内存会爆

#         print(f"   ✅ 区间 {sp} 跑完，共生成 {len(unique_segments)} 张图。")

# if __name__ == "__main__":
#     batch_process_plots()




# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ================= 1. 路径与参数配置 =================
# 自动定位到你的数据文件夹
DATA_FILE = Path(r"D:\energy_conservation\data\data_processed\results_布政-张家潭.xlsx")
TRIP_INDEX = 5  # 第6趟列车对应的索引是 5

def plot_trip6_vt():
    # 检查文件是否存在
    if not DATA_FILE.exists():
        print(f"❌ 找不到数据文件: {DATA_FILE}")
        return

    print(f"🚀 正在读取数据并提取第 6 趟车次...")

    try:
        # 2. 读取数据 (使用 openpyxl 引擎)
        df = pd.read_excel(DATA_FILE, engine='openpyxl')
        
        # 3. 锁定第 6 趟车 (按照 segment 出现的顺序)
        unique_segments = sorted(df['segment'].unique())
        if len(unique_segments) <= TRIP_INDEX:
            print(f"❌ 数据不足：当前文件仅有 {len(unique_segments)} 趟记录。")
            return
        
        target_seg_id = unique_segments[TRIP_INDEX]
        trip_data = df[df['segment'] == target_seg_id].copy()

        # 4. 数据预处理
        # 时间轴归零：减去该趟车的第一行时间
        t_start = trip_data['时刻'].iloc[0]
        trip_data['relative_time'] = trip_data['时刻'] - t_start
        
        # 速度单位转换：m/s -> km/h
        trip_data['v_kmh'] = trip_data['速度(m/s)'] * 3.6

        # 5. 绘图
        plt.figure(figsize=(12, 6))
        
        # 设置中文字体（根据你的系统环境）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False 

        plt.plot(trip_data['relative_time'], trip_data['v_kmh'], 
                 )

        # 装饰图表
        plt.title('“布政-张家潭”运行速度图 (v-t)', fontsize=14)
        plt.xlabel('时间 Time (s)', fontsize=12)
        plt.ylabel('速度 Velocity (km/h)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        

        # 6. 保存图片
        save_path = "Trip6_VT_Plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已生成并保存至: {os.path.abspath(save_path)}")
        plt.show()

    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    plot_trip6_vt()