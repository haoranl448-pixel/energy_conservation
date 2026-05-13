# -*- coding: utf-8 -*-
"""
run_optimize_mlp_trip6.py
1. 读取 section_params_trip6.csv (包含 Trip 6 真实载重)
2. 针对 5 维特征模型 [v, a, curv, grad, mass] 进行寻优
3. 生成 Pareto 曲线 CSV 和对比图
4. 终端汇总 Trip 6 历史全线总运行时分
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys
import warnings

# 屏蔽版本一致性警告
warnings.filterwarnings("ignore", category=UserWarning)

# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# ========= 常量与路径 =========
DT = 0.05  
DATA_DIR = Path(r"D:\energy_conservation\energy_conservation\data_processed")
# 请确保此路径存放着你的 best_model.pth 和 scaler.pkl
NN_DIR   = Path(project_root, "output", "models", "nn_results_mlp_8to5")
OUT_DIR  = Path(project_root, "output", "optimization", "opt_results_trip6")
PARAMS_CSV = Path(project_root, "data", "static", "section_params_trip6.csv")

LINE5_SECTIONS = [
    
    "庙堰-钟公庙"
]

# 2. 网络定义 (需与训练时 128->64->32 逻辑一致)
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.01)
        return torch.nn.functional.softplus(self.fc3(x))

# 3. 核心工具：静态数据清洗与插值
def get_clean_interpolator(station_pair):
    pattern = str(DATA_DIR / f"results_{station_pair}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files: return None, None, 0
    
    df = pd.read_excel(files[0])
    # 强制转换数值，解决 str vs float 报错
    for col in ['累计位移(m)', 'curvature', 'gradient', '时刻']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['累计位移(m)', 'curvature', 'gradient'])
    
    # 提取 Trip 6 (index 5) 的实际运行时间用于终端汇总
    segs = sorted(df['segment'].unique())
    trip6_time = 0
    if len(segs) > 5:
        seg_data = df[df['segment'] == segs[5]]
        trip6_time = seg_data['时刻'].iloc[-1] - seg_data['时刻'].iloc[0]

    s = df['累计位移(m)'].values
    mask = np.concatenate([[True], np.diff(s) > 1e-6])
    df_c = df[mask]
    
    curv_f = interp1d(df_c['累计位移(m)'], df_c['curvature'].values, fill_value='extrapolate')
    grad_f = interp1d(df_c['累计位移(m)'], df_c['gradient'].values, fill_value='extrapolate')
    return curv_f, grad_f, trip6_time

# 4. 单段寻优逻辑
def run_one_segment(params_row):
    sp = str(params_row['station_pair']).strip()
    L, V_UPP, V_MID = float(params_row['L']), float(params_row['V_UPPER']), float(params_row['V_MID'])
    A1, A2, A_DEC = float(params_row['A1']), float(params_row['A2']), float(params_row['A_DEC'])
    T_LOW, T_UPP, MASS = float(params_row['TIME_LOWER']), float(params_row['TIME_UPPER']), float(params_row['MASS'])
    TOL = float(params_row['REACH_TOL'])

    # 加载模型
    model_path = NN_DIR / sp / "best_model.pth"
    scaler_path = NN_DIR / sp / "scaler.pkl"
    if not model_path.exists(): return 0
    
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    net = Net(5); net.load_state_dict(torch.load(model_path, map_location='cpu')); net.eval()

    curv_f, grad_f, trip6_run_time = get_clean_interpolator(sp)
    if not curv_f: return 0

    def make_profile(v_peak, t_arr):
        t_acc = v_peak/A1 if v_peak<=V_MID else (V_MID/A1 + (v_peak-V_MID)/A2)
        t_dec = v_peak/abs(A_DEC)
        t_cruise = max(0.0, t_arr - t_acc - t_dec)
        t_grid = np.arange(0.0, t_acc + t_cruise + t_dec + DT, DT)
        v = np.zeros_like(t_grid)
        a = np.zeros_like(t_grid)
        for i, t in enumerate(t_grid):
            if t < t_acc:
                if v_peak <= V_MID: v[i], a[i] = A1*t, A1
                else:
                    tb = V_MID/A1
                    v[i], a[i] = (A1*t, A1) if t<=tb else (V_MID+A2*(t-tb), A2)
            elif t < t_acc + t_cruise: v[i], a[i] = v_peak, 0.0
            else:
                v[i], a[i] = max(v_peak + A_DEC*(t-(t_acc+t_cruise)), 0.0), A_DEC
        return {'v': v, 'a': a, 's': np.cumsum(v*DT), 'T': t_grid[-1]}

    def objective(x):
        vp, ta = x
        p = make_profile(vp, ta)
        if abs(p['s'][-1] - L) > TOL: return 1e9
        # 构建 5 维特征 [v, a, curv, grad, mass]
        X = np.stack([p['v'], p['a'], curv_f(p['s']), grad_f(p['s']), MASS*np.ones_like(p['v'])], axis=1)
        with torch.no_grad():
            e = net(torch.tensor(scaler.transform(X), dtype=torch.float32)).numpy().sum()
        return e

    # 执行时间扫描生成 Pareto
    out_dir = OUT_DIR / sp; out_dir.mkdir(parents=True, exist_ok=True)
    t_list = np.arange(T_LOW, T_UPP + 0.1, 2.0) # 2s步长提速
    results = []
    print(f"   >>> 正在扫描能耗空间...")
    for t in t_list:
        res = minimize(lambda vp: objective([vp, t]), x0=[V_UPP*0.7], bounds=[(5.0, V_UPP)], method='L-BFGS-B')
        if res.success:
            p = make_profile(res.x[0], t)
            X_final = np.stack([p['v'], p['a'], curv_f(p['s']), grad_f(p['s']), MASS*np.ones_like(p['v'])], axis=1)
            e = net(torch.tensor(scaler.transform(X_final), dtype=torch.float32)).numpy().sum()
            results.append((float(t), float(res.x[0]), float(e)))

    if results:
        pd.DataFrame(results, columns=['t_arrival', 'v_peak_opt', 'E_wh']).to_csv(out_dir/f"time_energy_curve_{sp}.csv", index=False)
        # 简单绘图对比
        plt.figure(); plt.plot([r[0] for r in results], [r[2] for r in results], 'o-'); plt.title(sp)
        plt.savefig(out_dir/"time_energy_curve.png"); plt.close()
    
    return trip6_run_time

def main():
    if not PARAMS_CSV.exists(): print(f"❌ 缺参数表"); return
    df_params = pd.read_csv(PARAMS_CSV)
    df_params['station_pair'] = df_params['station_pair'].str.strip()
    
    total_trip6_seconds = 0
    processed_count = 0
    
    print(f"开始批量优化... 目标文件: {PARAMS_CSV.name}")
    for i, sp in enumerate(LINE5_SECTIONS):
        row = df_params[df_params['station_pair'] == sp]
        if not row.empty:
            print(f"\n[{i+1}/26] 处理: {sp}")
            t_hist = run_one_segment(row.iloc[0])
            total_trip6_seconds += t_hist
            processed_count += 1
            
    print("\n" + "="*50)
    print(f"📊 任务完成统计：")
    print(f"   - 成功处理区间: {processed_count} / 26")
    print(f"   - 第 6 趟历史全线总运行时间 (Trip 6 Total Run Time):")
    print(f"     >>> {total_trip6_seconds:.2f} 秒")
    print(f"     >>> {total_trip6_seconds/60:.2f} 分钟")
    print(f"💡 请将此秒数填入 DP 调度代码作为 T_TOTAL 约束。")
    print("="*50)

if __name__ == "__main__":
    main()