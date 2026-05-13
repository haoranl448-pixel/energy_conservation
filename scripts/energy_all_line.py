# -*- coding: utf-8 -*-
"""
分析 507 系列服务号的全线总能耗（物理+AI残差）
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 环境与路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 确保能导入物理引擎
try:
    from src.physics.train_simu import TrainTheoreticalEnergyModel
except ImportError:
    print("❌ 无法导入 TrainTheoreticalEnergyModel，请检查 src/physics/train_simu.py 是否存在")
    sys.exit()

# 路径对齐
DATA_DIR = os.path.join(project_root, "data", "data_processed")
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "service_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标 507 系列服务号
TARGET_SERVICES = ["50704", "50706", "50708", "50710", "50712", "50714", "50716"]

# 26 站严格顺序
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ================= 2. 残差模型结构定义 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
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
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        out = self.transformer(src)
        return self.decoder(out[:, -1, :])

# ================= 3. 核心处理逻辑 =================

def get_clean_service_id(sid):
    """处理 Excel 中的服务号，统一转为干净的字符串"""
    if pd.isna(sid): return ""
    try:
        # 如果是 50704.0 这种浮点数，先转 int 再转 str
        return str(int(float(sid)))
    except:
        return str(sid).strip()

def process_all_services():
    print(f"🔎 正在扫描数据目录: {DATA_DIR}")
    
    # 初始化结果存储：{service_id: {'energy': 0.0, 'count': 0, 'time': 0.0}}
    master_results = {sid: {'energy': 0.0, 'count': 0, 'time': 0.0} for sid in TARGET_SERVICES}
    sim_engine = TrainTheoreticalEnergyModel()

    # 遍历 26 个区间文件
    for sp in LINE5_SECTIONS:
        file_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
        if not os.path.exists(file_path):
            print(f"⚠️ 缺失区间数据: {sp}")
            continue
        
        print(f"正在处理区间: {sp} ...")
        df_all = pd.read_excel(file_path)
        df_all['ser_str'] = df_all['服务号'].apply(get_clean_service_id)

        # 加载该区间的残差模型
        res_dir = os.path.join(RES_MODEL_BASE, sp)
        res_model, sx, sy = None, None, None
        if os.path.exists(os.path.join(res_dir, "best_res_model.pth")):
            try:
                with open(os.path.join(res_dir, "scaler_x.pkl"), "rb") as f: sx = pickle.load(f)
                with open(os.path.join(res_dir, "scaler_y.pkl"), "rb") as f: sy = pickle.load(f)
                res_model = ResidualTransformer(5)
                res_model.load_state_dict(torch.load(os.path.join(res_dir, "best_res_model.pth"), map_location='cpu'))
                res_model.eval()
            except:
                res_model = None

        # 在当前区间文件中查找我们要的 507 系列
        for sid in TARGET_SERVICES:
            df_seg = df_all[df_all['ser_str'] == sid].sort_values('时刻')
            if df_seg.empty: continue

            # 提取运动序列
            t_seq, v_seq = df_seg['时刻'].values, df_seg['速度(m/s)'].values
            mass_val = df_seg['重量'].iloc[0]
            
            # 1. 物理计算
            e_phy_steps = sim_engine.run_batch_simulation(t_seq, v_seq, mass_val)
            e_phy_sum = np.sum(e_phy_steps)

            # 2. 残差计算
            e_res_sum = 0.0
            if res_model is not None:
                # 准备特征: [v, a, e_phy, grad, mass]
                a_seq = np.zeros_like(v_seq); a_seq[1:] = np.diff(v_seq) / 0.05
                raw_X = np.stack([v_seq, a_seq, e_phy_steps, df_seg['gradient'].values, np.full_like(v_seq, mass_val)], axis=1)
                scaled_X = sx.transform(raw_X)
                
                res_preds = []
                for i in range(len(scaled_X)):
                    if i < 29: res_preds.append(0.0); continue
                    win = torch.tensor(scaled_X[i-29:i+1], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad(): p = res_model(win).item()
                    res_preds.append(sy.inverse_transform([[p]])[0][0])
                e_res_sum = np.sum(res_preds)
            
            # 3. 汇总结果
            master_results[sid]['energy'] += (e_phy_sum + e_res_sum)
            master_results[sid]['count'] += 1
            master_results[sid]['time'] += (t_seq[-1] if len(t_seq) > 0 else 0)

    # 生成最终报表
    final_list = []
    for sid, m in master_results.items():
        if m['count'] > 0:
            final_list.append({
                '服务号': sid,
                '全线总能耗(kWh)': round(m['energy'] / 1000.0, 2),
                '全线运行总耗时(s)': round(m['time'], 1),
                '匹配区间数': f"{m['count']}/26"
            })
    
    return pd.DataFrame(final_list)

# ================= 4. 运行 =================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 开始 507 系列全线能耗结算分析")
    print("="*60)
    
    report_df = process_all_services()
    
    if not report_df.empty:
        save_path = os.path.join(OUTPUT_DIR, "service_507_energy_summary.csv")
        report_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 结算完成！报表已保存至: {save_path}")
        print("\n" + report_df.to_string(index=False))
    else:
        print("\n❌ 未能在数据集中找到 507 系列服务号，请检查 data/data_processed 里的 Excel 内容。")
    print("\n" + "="*60)