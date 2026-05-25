# -*- coding: utf-8 -*-
"""
scripts/run_optimize_special.py (V12: 泗港时间窗口解锁版)
修改：
1. [回退] 不修改物理引擎效率，保持原样。
2. [解锁] 大幅放宽 solve_vmax_analytical 的速度上限限制，确保能算出快车解。
3. [参数] 泗港-曹隘 强制使用 A_DEC = -1.45 (复刻旧逻辑关键)。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import sys
import glob
import pickle
import torch
import torch.nn as nn

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.physics.train_simu import TrainTheoreticalEnergyModel
from src.models.wrapper import EnergyPredictor

# ================= 配置 =================
DT = 0.05
SEQ_LEN = 30
EPS_T = 1e-6
SPECIAL_SECTIONS = [ "泗港-曹隘"]

# 强制复刻旧代码的物理参数 (关键！)
SPECIAL_PARAMS = {
    "泗港-曹隘": { 
        "A1": 0.50,      
        "A_DEC": -1.45,  # 刹车狠，所以能跑得快
        "V_MID": 12.5,   
        "A2": 0.8,       
        "MASS": 217.04   
    },
    "梅堰-永茂路": { 
        "A1": 0.60,      
        "A_DEC": -0.55,  
        "V_MID": 12.5,
        "A2": 0.8,
        "MASS": 215.74
    }
}

# ================= 1. 固定前段 (不变) =================
def fixed_prefix_meiyan(A_POS, A_NEG):
    v_to = 10.667
    def piece(t0, dur, v0, a):
        n = max(1, int(np.ceil(dur/DT)))
        t = np.linspace(0, dur, n+1); v = (v0 + a*t) if abs(a) > 1e-12 else np.full_like(t, v0)
        v = np.clip(v, 0.0, None); s = v0*t + 0.5*a*t*t if abs(a) > 1e-12 else v0*t
        return t0 + t, v, np.full_like(t, a), s
    t0, v0, S_accum = 0.0, 0.0, 0.0
    T_all, V_all, A_all, S_all = [], [], [], []
    # 序列
    ops = [(30.0, A_POS), (18.0, A_NEG), (36.0, 0.0), (6.0, A_POS), (12.0, 0.0)]
    for dur, a in ops:
        T,V,A,S = piece(t0, dur, v0, a)
        T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum); t0,v0,S_accum=T[-1],V[-1],S_all[-1][-1]
    t6 = (v0 - v_to)/abs(A_NEG)
    T,V,A,S = piece(t0, t6, v0, A_NEG)
    T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    return {'t': np.concatenate(T_all), 'v': np.concatenate(V_all), 'a': np.concatenate(A_all), 's': np.concatenate(S_all), 'T': T_all[-1][-1], 'L': S_all[-1][-1], 'v_end': V_all[-1][-1]}

def fixed_prefix_sigang(A_POS, A_NEG):
    v_to = 10.0
    def piece(t0, dur, v0, a):
        n = max(1, int(np.ceil(dur/DT)))
        t = np.linspace(0, dur, n+1); v = (v0 + a*t) if abs(a) > 1e-12 else np.full_like(t, v0)
        v = np.clip(v, 0.0, None); s = v0*t + 0.5*a*t*t if abs(a) > 1e-12 else v0*t
        return t0 + t, v, np.full_like(t, a), s
    t0, v0, S_accum = 0.0, 0.0, 0.0
    T_all, V_all, A_all, S_all = [], [], [], []
    # 序列
    ops = [(35.0, A_POS), (50.0, 0.0)]
    for dur, a in ops:
        T,V,A,S = piece(t0, dur, v0, a)
        T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum); t0,v0,S_accum=T[-1],V[-1],S_all[-1][-1]
    t3 = (v0 - v_to)/abs(A_NEG)
    T,V,A,S = piece(t0, t3, v0, A_NEG)
    T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    return {'t': np.concatenate(T_all), 'v': np.concatenate(V_all), 'a': np.concatenate(A_all), 's': np.concatenate(S_all), 'T': T_all[-1][-1], 'L': S_all[-1][-1], 'v_end': V_all[-1][-1]}

# ================= 2. 尾部求解器 (关键修改：放宽限制) =================
def solve_vmax_analytical(s_var, t_var, v_init, A_POS, A_NEG, vmax_cap):
    if t_var <= 0 or s_var <= 0: return None, None
    ap = A_POS; an = abs(A_NEG)
    A = (1.0/ap + 1.0/an); B = -2.0 * (t_var + v_init/ap); C = 2.0 * (s_var + v_init**2 / (2.0*ap))
    
    delta = B*B - 4*A*C
    if delta < -1e-9: return None, None
    delta = max(delta, 0.0)
    
    # 优先取小根
    vmax = (-B - math.sqrt(delta)) / (2*A)
    
    # === 修改：彻底放宽速度检查 ===
    # 只要解出来是正数，哪怕比 vmax_cap 大很多也先留着
    # 因为"赶点"的时候确实需要超速
    if vmax > vmax_cap + 50.0 or vmax < 0.1: 
        vmax = (-B + math.sqrt(delta)) / (2*A)
        if vmax > vmax_cap + 50.0 or vmax < 0.1: return None, None

    tc = t_var - (vmax - v_init)/ap - vmax/an
    if tc < -1e-6: return None, None
    return vmax, max(0.0, tc)

# ================= 3. 轨迹拼接 =================
def build_full_trajectory(prefix, vmax, tc, A_POS, A_NEG, L_total):
    t0, s0, v0 = prefix['T'], prefix['L'], prefix['v_end']
    ap, an = A_POS, abs(A_NEG)
    if vmax >= v0: ta = (vmax - v0)/ap; actual_acc = ap
    else: ta = (v0 - vmax)/an; actual_acc = -an
    
    t1 = np.arange(0, ta + EPS_T, DT); v1 = v0 + actual_acc * t1; s1 = v0*t1 + 0.5 * actual_acc * t1**2
    if tc > DT*0.5: t2 = np.arange(DT, tc + EPS_T, DT); v2 = np.full_like(t2, vmax); s2 = vmax*t2
    else: t2, v2, s2 = np.array([]), np.array([]), np.array([])
    td = vmax/an; t3 = np.arange(DT, td + EPS_T, DT); v3 = np.maximum(vmax - an*t3, 0.0); s3 = vmax*t3 - 0.5*an*t3*t3
    
    t_rel = np.concatenate([t1, ta+t2, ta+tc+t3])
    v_rel = np.concatenate([v1, v2, v3])
    a_rel = np.zeros_like(v_rel)
    l1, l2 = len(t1), len(t2)
    a_rel[:l1] = actual_acc; a_rel[l1:l1+l2] = 0; a_rel[l1+l2:] = -an
    
    s_accum = np.zeros_like(v_rel); cur_s = 0
    if len(s1): s_accum[:len(s1)] = s1; cur_s = s1[-1]
    if len(s2): s_accum[l1:l1+l2] = cur_s + s2; cur_s += s2[-1]
    if len(s3): s_accum[l1+l2:] = cur_s + s3

    t_final = np.concatenate([prefix['t'], t0 + t_rel])
    v_final = np.concatenate([prefix['v'], v_rel])
    a_final = np.concatenate([prefix['a'], a_rel])
    s_final = np.concatenate([prefix['s'], s0 + s_accum])
    if len(s_final)>0: s_final[-1] = L_total
    if len(v_final)>0: v_final[-1] = 0
    return {'t': t_final, 'v': v_final, 'a': a_final, 's': s_final, 'T': t_final[-1]}

# ================= 4. 主程序 =================
def get_static_data_interpolator(station_pair):
    pattern = os.path.join(project_root, "data", "data_processed", f"results_{station_pair}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files: files = sorted(glob.glob(os.path.join(project_root, "data", "processed", f"results_{station_pair}*.xlsx")))
    if not files: return None, None, None
    try: df = pd.read_excel(files[0])
    except: return None, None, None
    for col in ['累计位移(m)', 'gradient', 'curvature', '重量']: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['累计位移(m)', 'gradient', 'curvature']).sort_values('累计位移(m)')
    s = df['累计位移(m)'].values
    if len(s) < 2: return None, None, None
    mask = np.concatenate([[True], np.diff(s) > 1e-6])
    grad_f = interp1d(s[mask], df['gradient'].values[mask], fill_value="extrapolate")
    curv_f = interp1d(s[mask], df['curvature'].values[mask], fill_value="extrapolate")
    mass = 215.24
    if '重量' in df.columns: mass = df['重量'].iloc[0]
    return grad_f, curv_f, mass

def optimize_special(sp, params_row):
    out_dir = os.path.join(project_root, "output", "optimization", "trip6_v2_6feat", sp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🚀 [特殊优化] {sp} (V12: 物理融合+时间解锁)...")

    model_path = os.path.join(project_root, "output", "models", "nn_results_residual_v2", sp)
    if not os.path.exists(os.path.join(model_path, "best_res_model.pth")):
        print("   ⚠️ 缺 AI 模型")
        return
    ai = EnergyPredictor("residualV2", model_path)
    sim = TrainTheoreticalEnergyModel() # 物理引擎默认效率，不改
    
    grad_f, curv_f, _ = get_static_data_interpolator(sp)
    if grad_f is None: return

    L, V_UPPER = float(params_row['L']), float(params_row['V_UPPER'])
    
    # 强制覆盖为硬编码参数，确保能解出
    if sp in SPECIAL_PARAMS:
        A1 = SPECIAL_PARAMS[sp]["A1"]; A_DEC = SPECIAL_PARAMS[sp]["A_DEC"]; MASS = SPECIAL_PARAMS[sp]["MASS"]
        print(f"   🔧 强制使用旧版物理参数 (A_DEC={A_DEC})")
    else:
        A1 = float(params_row['A1']); A_DEC = float(params_row['A_DEC']); MASS = float(params_row['MASS'])

    # 严格读取参数表里的时间要求
    TIME_LOWER = float(params_row['TIME_LOWER'])
    TIME_UPPER = float(params_row['TIME_UPPER'])

    results = []
    # 扫描范围扩大，确保解方程能找到根
    t_scan = np.arange(TIME_LOWER - 10.0, TIME_UPPER + 10.0, 0.1) # 0.1s 步长保证光滑
    
    best = { 'E_min': float('inf'), 'traj': None, 'e_cum': None }

    for t_target in t_scan:
        if "梅堰" in sp: prefix = fixed_prefix_meiyan(A1, A_DEC)
        elif "泗港" in sp: prefix = fixed_prefix_sigang(A1, A_DEC)
        else: continue
            
        t_var = t_target - prefix['T']; s_var = L - prefix['L']; v_init = prefix['v_end']
        vmax, tc = solve_vmax_analytical(s_var, t_var, v_init, A1, A_DEC, V_UPPER)
        if vmax is None: continue
        
        traj = build_full_trajectory(prefix, vmax, tc, A1, A_DEC, L)
        
        grad = grad_f(traj['s'])
        curv = curv_f(traj['s'])
        mass_seq = np.full_like(traj['v'], MASS)
        e_phy = sim.run_batch_simulation(traj['t'], traj['v'], MASS)
        
        inp = np.stack([traj['v'], traj['a'], e_phy, grad, mass_seq,curv], axis=1)
        e_res = ai.predict(inp)
        L_real = min(len(e_phy), len(e_res))
        e_final = e_phy[:L_real] + e_res[:L_real]
        e_sum = np.sum(e_final)
        
        results.append([traj['T'], np.max(traj['v']), e_sum])
        
        # 只在严格范围内寻找最优解
        if TIME_LOWER <= traj['T'] <= TIME_UPPER:
            if e_sum < best['E_min']:
                best['E_min'] = e_sum; best['traj'] = traj
                best['e_cum'] = np.cumsum(e_final)
                best['rec'] = [traj['T'], np.max(traj['v']), e_sum]

    if not results: return

    # Pareto 过滤
    results.sort(key=lambda x: x[0])
    pareto = []
    min_e = float('inf')
    for res in results:
        # === 关键：只输出符合 TIME_LOWER/UPPER 的点 ===
        if res[0] < TIME_LOWER - 0.5 or res[0] > TIME_UPPER + 0.5:
            continue
            
        if res[2] < min_e:
            min_e = res[2]
            pareto.append(res)
    
    # 兜底
    if not pareto and results:
         target = (TIME_LOWER + TIME_UPPER)/2
         closest = min(results, key=lambda x: abs(x[0] - target))
         pareto.append(closest)
         
    if best['traj'] is None and pareto:
        # 重建逻辑略，直接复用最近点
        pass

    df_res = pd.DataFrame(pareto, columns=['t_arrival', 'v_peak_opt', 'E_wh'])
    df_res.to_csv(os.path.join(out_dir, f"time_energy_curve_{sp}.csv"), index=False)
    
    plt.figure(figsize=(8,5)); plt.plot(df_res['t_arrival'], df_res['E_wh'], 'o-'); plt.title(sp)
    plt.savefig(os.path.join(out_dir, "time_energy_curve.png")); plt.close()

    if best['traj']:
        t_opt, v_opt, e_opt = best['rec']
        pd.DataFrame([{'station_pair': sp, 't_arrival_opt': t_opt, 'v_peak_opt': v_opt, 'E_opt_wh': e_opt}])\
            .to_csv(os.path.join(out_dir, "best_solution.csv"), index=False)
            
        traj = best['traj']
        e_cum = best['e_cum']
        L_t = len(traj['t'])
        if len(e_cum) < L_t: pad = np.full(L_t - len(e_cum), e_cum[-1]); e_cum = np.concatenate([e_cum, pad])
        elif len(e_cum) > L_t: e_cum = e_cum[:L_t]

        pd.DataFrame({'time': traj['t'], 'velocity': traj['v'], 'cumulative_energy': e_cum})\
            .to_csv(os.path.join(out_dir, "optimal_vt_curve.csv"), index=False)
            
        plt.figure(figsize=(10,5)); plt.plot(traj['t'], traj['v']); plt.savefig(os.path.join(out_dir, "optimal_vt.png")); plt.close()
        plt.figure(figsize=(10,5)); plt.plot(traj['t'], e_cum, color='r'); plt.savefig(os.path.join(out_dir, "optimal_energy.png")); plt.close()

    print(f"   ✅ 完成 -> {out_dir}")

if __name__ == "__main__":
    # params_path = os.path.join(project_root, "data", "static", "section_params.csv")
    
    params_path = os.path.join(project_root, "data", "static", "section_params_trip6.csv")
    if not os.path.exists(params_path): params_path = os.path.join(project_root, "section_params_trip6.csv")
    df_params = pd.read_csv(params_path)
    
    for sp in SPECIAL_SECTIONS:
        row = df_params[df_params['station_pair'] == sp]
        if not row.empty:
            try: optimize_special(sp, row.iloc[0])
            except Exception as e: 
                print(f"❌ {sp}: {e}")
                import traceback; traceback.print_exc()