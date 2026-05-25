# -*- coding: utf-8 -*-
"""
终极可视化脚本：五等级能耗全景图
1. 载入 V2 等级感知模型。
2. 针对布政-张家潭，分别以 Class 1-5 的标准时间作为指令进行 5 次仿真。
3. 将结果统一对齐到里程轴（Distance）。
4. 绘制累积能耗、速度剖面、坡度图。
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

# ================= 1. 环境与路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 路径对齐
ATO_V2_DIR = os.path.join(project_root, "output", "models", "ato_results_v2")
TABLE_DIR = os.path.join(project_root, "output", "analysis", "class_tables_strict")
STD_TIME_FILE = os.path.join(TABLE_DIR, "standard_class_times.csv")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "final_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.physics.train_simu import TrainTheoreticalEnergyModel

# ================= 2. 模型定义 (必须与 V2 训练一致) =================
class ATOPolicyNetV2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.fc(x)

# ================= 3. 核心仿真逻辑 =================
def simulate_single_class(target_t_std, station_pair):
    # 加载模型和配置
    with open(os.path.join(ATO_V2_DIR, "ato_scaler_v2.pkl"), "rb") as f: scaler = pickle.load(f)
    with open(os.path.join(ATO_V2_DIR, "train_config_v2.pkl"), "rb") as f: config = pickle.load(f)
    
    model = ATOPolicyNetV2(len(config['feature_cols']))
    model.load_state_dict(torch.load(os.path.join(ATO_V2_DIR, "ato_model_v2.pth"), map_location='cpu'))
    model.eval()

    # 获取物理地图
    map_path = os.path.join(project_root, "output", "models", "ato_results", "track_map.pkl")
    df_map = pd.read_pickle(map_path)
    # 强制将坡度和曲率插值化
    f_curv = interp1d(df_map['累计位移(m)'], df_map['curvature'], kind='nearest', fill_value="extrapolate")
    f_grad = interp1d(df_map['累计位移(m)'], df_map['gradient'], kind='nearest', fill_value="extrapolate")

    TARGET_L = config['target_dist'] # 1450m 左右
    DT = 0.1
    t, s, v = 0.0, 0.0, 0.0
    history = []
    
    # 模拟循环
    for _ in range(2000): # 最大步数
        dist_rem = TARGET_L - s
        time_rem = target_t_std - t # 注意：这里使用标准时间作为倒计时基准
        
        # 构造 V2 特征：[速度, 曲率, 坡度, 重量, dist_rem, time_rem, target_time_std]
        # 这里的重量暂时用训练时的均值
        feat = [[v, float(f_curv(s)), float(f_grad(s)), 215.0, dist_rem, time_rem, target_t_std]]
        feat_scaled = scaler.transform(feat)
        
        with torch.no_grad():
            acc = model(torch.tensor(feat_scaled, dtype=torch.float32)).item()

        # 基础物理限制 (0.55/0.8/-0.5)
        if dist_rem <= (v**2)/(2*0.5) + 2.0:
            acc = -0.5 # ATP 强行接管
        else:
            acc = np.clip(acc, -0.5, 0.55 if v < 12.5 else 0.8)

        v = max(v + acc * DT, 0.0)
        s += v * DT
        t += DT
        history.append([s, v, acc])
        if s >= TARGET_L and v <= 0.05: break

    res_df = pd.DataFrame(history, columns=['dist', 'velocity', 'acc'])
    
    # 计算该轨迹的物理能耗 (Wh)
    engine = TrainTheoreticalEnergyModel()
    # 为简单起见，这里演示只用物理能耗，如需总能耗可加上残差预测
    e_phy = engine.run_batch_simulation(np.arange(len(res_df))*DT, res_df['velocity'].values, 215.0)
    res_df['cum_energy'] = np.cumsum(e_phy)
    
    return res_df, df_map

# ================= 4. 绘图主程序 =================
def main():
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
    
    # 读取标准时间
    df_std = pd.read_csv(STD_TIME_FILE).set_index('区段')
    station = "布政-张家潭"
    class_times = df_std.loc[station].values # [Class1_t, Class2_t...]
    
    fig, (ax_e, ax_v, ax_track) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                                              gridspec_kw={'height_ratios': [3, 2, 1]})
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    labels = ['Class 1 (最快)', 'Class 2', 'Class 3', 'Class 4', 'Class 5 (最慢)']

    print(f"📊 正在生成 {station} 的五等级对比报告...")

    for i in range(5):
        t_std = float(class_times[i])
        print(f"   正在仿真 {labels[i]} : {t_std}s")
        df_sim, df_track = simulate_single_class(t_std, station)
        
        # 绘图
        ax_e.plot(df_sim['dist'], df_sim['cum_energy'], color=colors[i], label=f"{labels[i]} - {t_std}s", lw=2)
        ax_v.plot(df_sim['dist'], df_sim['velocity'] * 3.6, color=colors[i], alpha=0.8)

    # 绘制坡度底色图 (Track Profile)
    # 统一里程轴，让甲方看清坡度
    ax_track.fill_between(df_track['累计位移(m)'], df_track['gradient'], 0, 
                          where=(df_track['gradient'] >= 0), color='red', alpha=0.3)
    ax_track.fill_between(df_track['累计位移(m)'], df_track['gradient'], 0, 
                          where=(df_track['gradient'] < 0), color='green', alpha=0.3)
    
    # 细节美化
    ax_e.set_ylabel("累积能耗 (Wh)", fontsize=12); ax_e.grid(True, alpha=0.3)
    ax_e.set_title(f"{station} 全线能耗全景透视图 (里程维度)", fontsize=15, fontweight='bold')
    ax_e.legend(loc='upper left', frameon=True)

    ax_v.set_ylabel("速度 (km/h)", fontsize=12); ax_v.grid(True, alpha=0.3)
    ax_v.set_title("微观车速剖面 (反映AI对不同等级指令的风格响应)", fontsize=12)

    ax_track.set_ylabel("坡度 (‰)", fontsize=12); ax_track.set_xlabel("里程 (m)", fontsize=12)
    ax_track.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "five_class_et_vt_map.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 终极全景图已生成: {save_path}")

if __name__ == "__main__":
    main()