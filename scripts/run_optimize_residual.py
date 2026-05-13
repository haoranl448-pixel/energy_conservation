# -*- coding: utf-8 -*-
"""
scripts/run_optimize_residual.py (Bug修复版)
修复：ValueError: operands could not be broadcast together
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 引入核心模块
from src.physics.train_simu import TrainTheoreticalEnergyModel
from src.models.wrapper import EnergyPredictor

# ================= 配置 =================
DT = 0.05
SEQ_LEN = 30
# 5号线区间
# LINE5_SECTIONS = [
#     "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
# ]
LINE5_SECTIONS = [
    
    "庙堰-钟公庙"
]
# ================= 工具函数 =================

def get_static_data_interpolator(station_pair):
    """读取静态数据构建插值器"""
    pattern = os.path.join(project_root, "data", "data_processed", f"results_{station_pair}*.xlsx")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(project_root, "data", "processed", f"results_{station_pair}*.xlsx")
        files = glob.glob(pattern)
        
    if not files: return None, None

    df = pd.read_excel(files[0])
    s = df['累计位移(m)'].values
    mask = np.concatenate([[True], np.diff(s) > 1e-6])
    s_clean = s[mask]
    grad_clean = df['gradient'].values[mask]
    curv_clean = df['curvature'].values[mask]

    grad_f = interp1d(s_clean, grad_clean, fill_value="extrapolate")
    curv_f = interp1d(s_clean, curv_clean, fill_value="extrapolate")
    return grad_f, curv_f


# def get_static_data_interpolator(station_pair):
#     """读取静态数据构建插值器（修复版：强制数值转换）"""
#     # 查找文件逻辑保持不变
#     pattern = os.path.join(project_root, "data", "data_processed", f"results_{station_pair}*.xlsx")
#     files = glob.glob(pattern)
#     if not files:
#         pattern = os.path.join(project_root, "data", "processed", f"results_{station_pair}*.xlsx")
#         files = glob.glob(pattern)
        
#     if not files: return None, None

#     # 只读取第1个匹配的文件
#     df = pd.read_excel(files[0])
    
#     # === 核心修复点：强制将列转换为数字，无法转换的变 NaN 并剔除 ===
#     df['累计位移(m)'] = pd.to_numeric(df['累计位移(m)'], errors='coerce')
#     df['gradient'] = pd.to_numeric(df['gradient'], errors='coerce')
#     df['curvature'] = pd.to_numeric(df['curvature'], errors='coerce')
    
#     # 剔除掉包含 NaN 的行
#     df = df.dropna(subset=['累计位移(m)', 'gradient', 'curvature'])
    
#     s = df['累计位移(m)'].values
#     if len(s) < 2: return None, None

#     # 这里的 np.diff 现在不会再报 str 错误了
#     mask = np.concatenate([[True], np.diff(s) > 1e-6])
#     s_clean = s[mask]
#     grad_clean = df['gradient'].values[mask]
#     curv_clean = df['curvature'].values[mask]

#     grad_f = interp1d(s_clean, grad_clean, fill_value="extrapolate")
#     curv_f = interp1d(s_clean, curv_clean, fill_value="extrapolate")
#     return grad_f, curv_f

def get_trajectory_by_distance(v_target, L_total, p):
    """生成梯形曲线"""
    v_peak = float(v_target)
    A1, A2, A_DEC, V_MID = p['A1'], p['A2'], abs(p['A_DEC']), p['V_MID']
    
    if v_peak <= V_MID:
        t1 = v_peak / A1; d1 = 0.5 * A1 * t1**2
        t2 = 0; d2 = 0
    else:
        t1 = V_MID / A1; d1 = 0.5 * A1 * t1**2
        t2 = (v_peak - V_MID) / A2
        d2 = V_MID * t2 + 0.5 * A2 * t2**2
    d_acc = d1 + d2
    t_acc = t1 + t2
    
    t_dec = v_peak / A_DEC
    d_dec = 0.5 * A_DEC * t_dec**2
    
    if d_acc + d_dec > L_total:
        return get_trajectory_by_distance(v_peak * 0.95, L_total, p)
    
    d_cruise = L_total - d_acc - d_dec
    t_cruise = d_cruise / v_peak
    
    T_total = t_acc + t_cruise + t_dec
    t_grid = np.arange(0.0, T_total + DT, DT)
    
    v_arr, a_arr = [], []
    for t in t_grid:
        v, a = 0.0, 0.0
        if t <= t_acc:
            if t <= t1: v = A1 * t; a = A1
            else: v = V_MID + A2 * (t - t1); a = A2
        elif t <= t_acc + t_cruise: v = v_peak; a = 0.0
        else:
            dt_dec = t - (t_acc + t_cruise)
            v = max(0.0, v_peak - A_DEC * dt_dec); a = -A_DEC
        v_arr.append(v); a_arr.append(a)
        
    v_arr = np.array(v_arr)
    a_arr = np.array(a_arr)
    s_arr = np.cumsum(v_arr * DT)
    
    if len(s_arr) > 0 and s_arr[-1] > 0:
        s_arr = s_arr * (L_total / s_arr[-1])
        
    return {'t': t_grid, 'v': v_arr, 'a': a_arr, 's': s_arr, 'T': t_grid[-1]}

# ================= 核心优化流程 =================

def optimize_section(sp, params_row):
    out_dir = os.path.join(project_root, "output", "optimization", "residual1", sp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🚀 [优化] 正在计算: {sp} ...")

    model_path = os.path.join(project_root, "output", "models", "nn_results_residual", sp)
    if not os.path.exists(os.path.join(model_path, "best_res_model.pth")):
        print(f"   ⚠️ 未找到模型，跳过")
        return

    try:
        ai_predictor = EnergyPredictor("residual", model_path)
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return

    sim_model = TrainTheoreticalEnergyModel()
    grad_f, curv_f = get_static_data_interpolator(sp)
    
    if grad_f is None: return

    L = float(params_row['L'])
    V_UPPER = float(params_row['V_UPPER'])
    TIME_LOWER = float(params_row['TIME_LOWER'])
    TIME_UPPER = float(params_row['TIME_UPPER'])
    MASS = float(params_row['MASS'])

    p_veh = {'V_MID': float(params_row['V_MID']), 'A1': float(params_row['A1']),
             'A2': float(params_row['A2']), 'A_DEC': float(params_row['A_DEC'])}

    results = []
    v_scan = np.arange(5.0, V_UPPER + 0.1, 0.2)
    
    best_record = { 'E_min': float('inf'), 'traj': None, 'e_cum_seq': None }

    for v_target in v_scan:
        traj = get_trajectory_by_distance(v_target, L, p_veh)
        t_actual = traj['T']
        
        if not (TIME_LOWER <= t_actual <= TIME_UPPER): continue
            
        grad_seq = grad_f(traj['s'])
        curv_seq = curv_f(traj['s'])
        mass_seq = np.full_like(traj['v'], MASS)
        
        e_phy_seq = sim_model.run_batch_simulation(traj['t'], traj['v'], MASS)
        
        input_data = np.stack([traj['v'], traj['a'], e_phy_seq, grad_seq, mass_seq], axis=1)
        e_res_seq = ai_predictor.predict(input_data)
        
        # === 核心修复: 长度对齐 ===
        if len(e_res_seq) != len(e_phy_seq):
            diff = len(e_res_seq) - len(e_phy_seq)
            if diff > 0: e_res_seq = e_res_seq[diff:]
            else: e_phy_seq = e_phy_seq[:len(e_res_seq)]
        # ========================

        e_final_seq = e_phy_seq + e_res_seq
        e_total = np.sum(e_final_seq)
        
        results.append([t_actual, np.max(traj['v']), e_total])
        
        if e_total < best_record['E_min']:
            best_record['E_min'] = e_total
            best_record['traj'] = traj
            best_record['e_cum_seq'] = np.cumsum(e_final_seq)
            best_record['v_peak'] = np.max(traj['v'])
            best_record['t_arrival'] = t_actual

    if not results:
        print("   ❌ 未找到可行解")
        return

    df_res = pd.DataFrame(results, columns=['t_arrival', 'v_peak_opt', 'E_wh'])
    df_res['t_round'] = df_res['t_arrival'].round(1)
    df_res = df_res.sort_values('E_wh').drop_duplicates('t_round').sort_values('t_arrival')
    del df_res['t_round']
    
    df_res.to_csv(os.path.join(out_dir, f"time_energy_curve_{sp}.csv"), index=False)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df_res['t_arrival'], df_res['E_wh'], 'o-', color='darkred', markersize=3)
    plt.title(f"{sp} 时间-能耗帕累托曲线")
    plt.xlabel("运行时间 (s)")
    plt.ylabel("总能耗 (Wh)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_energy_curve.png"), dpi=300)
    plt.close()

    best_df = pd.DataFrame([{
        'station_pair': sp,
        't_arrival_opt': best_record['t_arrival'],
        'v_peak_opt': best_record['v_peak'],
        'E_opt_wh': best_record['E_min'],
        'model_type': 'Physics+AI_Fusion'
    }])
    best_df.to_csv(os.path.join(out_dir, "best_solution.csv"), index=False)

    traj = best_record['traj']
    vt_df = pd.DataFrame({
        'time': traj['t'],
        'velocity': traj['v'],
        'acceleration': traj['a'],
        'distance': traj['s'],
        'cumulative_energy': best_record['e_cum_seq']
    })
    vt_df.to_csv(os.path.join(out_dir, "optimal_vt_curve.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(traj['t'], traj['v'], lw=2, color='#2ca02c')
    plt.title(f"{sp} 最优速度曲线 (T={best_record['t_arrival']:.1f}s)")
    plt.xlabel("时间 (s)")
    plt.ylabel("速度 (m/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "optimal_vt.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(traj['t'], best_record['e_cum_seq'], lw=2, color='#d62728')
    plt.fill_between(traj['t'], best_record['e_cum_seq'], color='#d62728', alpha=0.1)
    plt.title(f"{sp} 最优累积能耗 (Total={best_record['E_min']:.1f}Wh)")
    plt.xlabel("时间 (s)")
    plt.ylabel("累积能耗 (Wh)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "optimal_energy.png"), dpi=300)
    plt.close()

    print(f"   ✅ 全部6个文件生成完毕 -> {out_dir}")

if __name__ == "__main__":
    # params_path = os.path.join(project_root, "data", "static", "section_params.csv")
    params_path = os.path.join(project_root, "data", "static", "section_params_trip6.csv")
    if not os.path.exists(params_path):
        params_path = os.path.join(project_root, "section_params_trip6.csv")
        
    if not os.path.exists(params_path):
        print("❌ 找不到 section_params_trip6.csv")
        sys.exit()
        
    df_params = pd.read_csv(params_path)
    
    for sp in LINE5_SECTIONS:
        row = df_params[df_params['station_pair'] == sp]
        if not row.empty:
            try:
                optimize_section(sp, row.iloc[0])
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ {sp} 出错: {e}")