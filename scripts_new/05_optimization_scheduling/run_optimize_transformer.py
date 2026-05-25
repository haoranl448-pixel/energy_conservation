#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_transformer_final.py (V2 稳健版)
策略升级：
1. 放弃 "扫描速度"，改为 "扫描时间"。
2. 针对每个目标时间 T，使用二分查找反推 V_peak，确保 100% 满足距离和时间约束。
3. 完美复刻 MLP 的产出逻辑，但内核换成 Transformer。
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import math
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ========= 配置参数 =========
DT = 0.05
SEQ_LEN = 30
DATA_DIR = Path("data_processed")
NN_DIR = Path("nn_results_transformer")
OUT_DIR = Path("opt_results_transformer")
PARAMS_CSV = Path("section_params.csv")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 5号线区间
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ========= 1. Transformer 模型 =========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class EnergyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(EnergyTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1)
        )
    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :])

# ========= 2. 核心工具 =========
def create_sliding_windows(data, seq_length):
    padding = np.tile(data[0], (seq_length - 1, 1))
    data_padded = np.vstack([padding, data])
    windows = []
    for i in range(len(data)):
        windows.append(data_padded[i : i + seq_length])
    return np.array(windows)

# 梯形生成器：给定 v_peak, t_target, 算出 s (可能不等于L)
def make_profile_try(v_peak, t_target, p):
    if v_peak <= p['V_MID']:
        t_acc = v_peak / p['A1']
    else:
        t_acc = p['V_MID'] / p['A1'] + (v_peak - p['V_MID']) / p['A2']
    
    t_dec = v_peak / abs(p['A_DEC'])
    t_cruise = max(0.0, t_target - t_acc - t_dec)
    T_tot = t_acc + t_cruise + t_dec

    t_grid = np.arange(0.0, T_tot + DT, DT)
    v, a = np.zeros_like(t_grid), np.zeros_like(t_grid)
    
    for i, t in enumerate(t_grid):
        if t < t_acc:
            if v_peak <= p['V_MID']: v[i], a[i] = p['A1'] * t, p['A1']
            else:
                tb = p['V_MID'] / p['A1']
                if t <= tb: v[i], a[i] = p['A1'] * t, p['A1']
                else: v[i], a[i] = p['V_MID'] + p['A2'] * (t - tb), p['A2']
        elif t < t_acc + t_cruise: v[i], a[i] = v_peak, 0.0
        else:
            td = t - (t_acc + t_cruise)
            vt = v_peak + p['A_DEC'] * td
            v[i], a[i] = max(vt, 0.0), p['A_DEC']
    s = np.cumsum(v * DT)
    return {'t': t_grid, 'v': v, 'a': a, 's': s, 'T': T_tot}

# === 关键算法：二分查找 v_peak ===
# 目标：找到一个 v_peak，使得在 t_target 时间内跑完的距离 == L
def solve_v_for_time_dist(t_target, L_target, p, tol=1.0):
    v_min, v_max = 0.1, p['V_UPPER'] * 1.5 # 放宽上限以便搜索
    
    # 物理极限检查：如果一直以最大速度跑，时间还不够，说明 t_target 太短，不可行
    prof_fastest = make_profile_try(p['V_UPPER'], t_target, p)
    if prof_fastest['s'][-1] < L_target - 100: # 容差大一点
        return None # 无法到达

    best_prof = None
    min_err = float('inf')

    # 二分查找
    for _ in range(30):
        v_mid = (v_min + v_max) / 2
        prof = make_profile_try(v_mid, t_target, p)
        s_final = prof['s'][-1]
        
        err = s_final - L_target
        if abs(err) < min_err:
            min_err = abs(err)
            best_prof = prof
            
        if abs(err) < tol: # 距离误差小于 1m
            return prof
        
        if err < 0: # 跑得太近了 -> 速度不够 -> 加大 v
            v_min = v_mid
        else: # 跑得太远了 -> 速度太快 -> 减小 v
            v_max = v_mid
            
    # 如果没完全收敛，返回误差最小的那个（兜底）
    return best_prof

# ========= 3. 主流程 =========
def run_one_segment(params_row):
    sp = str(params_row['station_pair']).strip()
    L, V_UPPER = float(params_row['L']), float(params_row['V_UPPER'])
    TIME_LOWER, TIME_UPPER = float(params_row['TIME_LOWER']), float(params_row['TIME_UPPER'])
    MASS = float(params_row['MASS'])
    
    p = {'V_MID': float(params_row['V_MID']), 'A1': float(params_row['A1']),
         'A2': float(params_row['A2']), 'A_DEC': float(params_row['A_DEC']),
         'V_UPPER': V_UPPER} # 传进去方便边界检查

    # 模型加载
    nn_path = NN_DIR / sp
    if not (nn_path / "best_transformer.pth").exists():
        print(f"⚠️ [{sp}] 缺模型，跳过")
        return

    save_dir = OUT_DIR / sp
    save_dir.mkdir(parents=True, exist_ok=True)

    # 静态数据
    try:
        data_file = sorted(glob.glob(str(DATA_DIR / f"results_{sp}*.xlsx")))[0]
        df_raw = pd.read_excel(data_file)
        s_raw = df_raw['累计位移(m)'].values
        mask = np.concatenate([[True], np.diff(s_raw) > 1e-6])
        df_clean = df_raw[mask]
        curve_f = interp1d(df_clean['累计位移(m)'], df_clean['curvature'], fill_value='extrapolate')
        grade_f = interp1d(df_clean['累计位移(m)'], df_clean['gradient'], fill_value='extrapolate')
    except:
        print(f"⚠️ [{sp}] 缺静态数据，使用默认0值")
        curve_f = lambda x: np.zeros_like(x)
        grade_f = lambda x: np.zeros_like(x)

    # NN 准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(nn_path / "scaler_x.pkl", 'rb') as f: scaler_x = pickle.load(f)
    with open(nn_path / "scaler_y.pkl", 'rb') as f: scaler_y = pickle.load(f)
    model = EnergyTransformer(input_dim=5).to(device)
    model.load_state_dict(torch.load(nn_path / "best_transformer.pth", map_location=device))
    model.eval()

    def calc_energy(prof):
        curv = curve_f(prof['s'])
        grad = grade_f(prof['s'])
        mass = np.full_like(prof['v'], MASS)
        X_raw = np.stack([prof['v'], prof['a'], curv, grad, mass], axis=1)
        X_seq = create_sliding_windows(scaler_x.transform(X_raw), SEQ_LEN)
        
        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(X_seq).float().to(device)).cpu().numpy()
        
        # 反归一化
        pred = scaler_y.inverse_transform(pred_scaled).flatten()
        pred = np.maximum(pred, 0)
        return np.sum(pred), np.cumsum(pred)

    # === 开始优化 ===
    print(f"[{sp}] 正在计算 Time-Energy 曲线 (Target: {TIME_LOWER}s ~ {TIME_UPPER}s)...")
    results = []
    
    # 1. 扫描时间 (定时间 -> 求速度 -> 求能耗)
    # 步长 1.0 秒
    scan_times = np.arange(TIME_LOWER, TIME_UPPER + 1, 1.0)
    
    for t in scan_times:
        # 二分查找该时间下的最优速度曲线
        prof = solve_v_for_time_dist(t, L, p)
        
        if prof is not None:
            # 算出能耗
            e_total, _ = calc_energy(prof)
            v_max_actual = np.max(prof['v'])
            
            # 记录结果
            results.append([t, v_max_actual, e_total])
            # print(f"   T={t}s -> E={e_total:.1f}Wh (V_peak={v_max_actual:.1f})")

    if not results:
        print(f"❌ [{sp}] 未找到可行解，可能是时间窗口设置不合理（太短跑不完）。")
        return

    # 2. 保存 Time-Energy CSV
    df_res = pd.DataFrame(results, columns=['t_arrival', 'v_peak_opt', 'E_wh'])
    df_res.to_csv(save_dir / f"time_energy_curve_{sp}.csv", index=False)
    
    # 3. 画 Time-Energy 图
    plt.figure(figsize=(8, 5))
    plt.plot(df_res['t_arrival'], df_res['E_wh'], 'o-', label='Transformer')
    plt.title(f"{sp} 时间-能耗帕累托曲线")
    plt.xlabel("运行时间 (s)")
    plt.ylabel("能耗 (Wh)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "time_energy_curve.png", dpi=300)
    plt.close()

    # 4. 选出全局最优 (能耗最低点)
    best_idx = df_res['E_wh'].idxmin()
    best_row = df_res.iloc[best_idx]
    
    # 重建最优轨迹
    prof_best = solve_v_for_time_dist(best_row['t_arrival'], L, p)
    _, e_cum = calc_energy(prof_best)
    
    # 5. 保存其余 4 个文件
    # best_solution.csv
    pd.DataFrame([{
        'station_pair': sp, 't_arrival_opt': best_row['t_arrival'],
        'v_peak_opt': best_row['v_peak_opt'], 'E_opt_wh': best_row['E_wh']
    }]).to_csv(save_dir / "best_solution.csv", index=False)
    
    # optimal_vt_curve.csv
    pd.DataFrame({
        'time': prof_best['t'], 'velocity': prof_best['v'], 
        'acceleration': prof_best['a'], 'distance': prof_best['s'],
        'cumulative_energy': e_cum
    }).to_csv(save_dir / "optimal_vt_curve.csv", index=False)
    
    # optimal_vt.png
    plt.figure(figsize=(10, 5))
    plt.plot(prof_best['t'], prof_best['v'], lw=2, color='green')
    plt.title(f"{sp} 最优速度曲线 (T={best_row['t_arrival']}s)")
    plt.savefig(save_dir / "optimal_vt.png", dpi=300); plt.close()
    
    # optimal_energy.png
    plt.figure(figsize=(10, 5))
    plt.plot(prof_best['t'], e_cum, lw=2, color='red')
    plt.title(f"{sp} 最优累积能耗")
    plt.savefig(save_dir / "optimal_energy.png", dpi=300); plt.close()

    print(f"✅ [{sp}] 完成。")

if __name__ == "__main__":
    if not PARAMS_CSV.exists(): print("Err: No params csv")
    else:
        params = pd.read_csv(PARAMS_CSV)
        params['station_pair'] = params['station_pair'].astype(str).str.strip()
        for sp in LINE5_SECTIONS:
            row = params[params['station_pair'] == sp]
            if not row.empty:
                try: run_one_segment(row.iloc[0])
                except Exception as e: print(f"Err {sp}: {e}")