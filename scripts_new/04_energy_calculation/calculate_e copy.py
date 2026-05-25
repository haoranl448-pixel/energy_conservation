# -*- coding: utf-8 -*-
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

import os
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from src.physics.train_simu import TrainTheoreticalEnergyModel
from src.models.wrapper import EnergyPredictor

# ================= 1. 路径配置 =================
REPORT_FILE = project_root / "output" / "trip6_analysis_report" / "Trip6_Saving_Report.csv"
PARAM_FILE = project_root / "data" / "static" / "section_params_trip6.csv"
DATA_DIR = project_root / "data" / "data_processed"
RES_MODEL_BASE = project_root / "output" / "models" / "nn_results_residual_v2"
OUTPUT_DIR = project_root / "output" / "analysis" / "trip6_historical_playback"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DT = 0.05
EPS_T = 1e-6

SPECIAL_SECTIONS = ["泗港-曹隘", "梅堰-永茂路"]
SPECIAL_PARAMS = {
    "泗港-曹隘": {"A1": 0.50, "A_DEC": -1.45, "V_MID": 12.5, "A2": 0.8, "MASS": 217.04},
    "梅堰-永茂路": {"A1": 0.60, "A_DEC": -0.55, "V_MID": 12.5, "A2": 0.8, "MASS": 215.74},
}

# ================= 2. 工具函数 =================
def get_track_f(sp):
    file_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
    df = pd.read_excel(file_path)
    for col in ['累计位移(m)', 'gradient', 'curvature']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['累计位移(m)', 'gradient', 'curvature']).sort_values('累计位移(m)')
    s = df['累计位移(m)'].values
    if len(s) < 2:
        return None, None
    mask = np.concatenate([[True], np.diff(s) > 1e-6])
    grad_f = interp1d(s[mask], df['gradient'].values[mask], fill_value="extrapolate")
    curv_f = interp1d(s[mask], df['curvature'].values[mask], fill_value="extrapolate")
    return grad_f, curv_f

# --- 特殊区间前缀 ---
def fixed_prefix_sigang(A_POS, A_NEG):
    v_to = 10.0
    def piece(t0, dur, v0, a):
        n = max(1, int(np.ceil(dur/DT)))
        t = np.linspace(0, dur, n+1)
        v = (v0 + a*t) if abs(a) > 1e-12 else np.full_like(t, v0)
        v = np.clip(v, 0.0, None)
        s = v0*t + 0.5*a*t**2 if abs(a) > 1e-12 else v0*t
        return t0 + t, v, np.full_like(t, a), s
    t0, v0, S_accum = 0.0, 0.0, 0.0
    T_all, V_all, A_all, S_all = [], [], [], []
    ops = [(35.0, A_POS), (50.0, 0.0)]
    for dur, a in ops:
        T,V,A,S = piece(t0, dur, v0, a)
        T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
        t0,v0,S_accum=T[-1],V[-1],S_all[-1][-1]
    t3 = (v0 - v_to)/abs(A_NEG)
    T,V,A,S = piece(t0, t3, v0, A_NEG)
    T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    return {'t': np.concatenate(T_all), 'v': np.concatenate(V_all), 'a': np.concatenate(A_all),
            's': np.concatenate(S_all), 'T': T_all[-1][-1], 'L': S_all[-1][-1], 'v_end': V_all[-1][-1]}

def solve_vmax_analytical(s_var, t_var, v_init, A_POS, A_NEG, vmax_cap):
    if t_var <= 0 or s_var <= 0: return None, None
    ap = A_POS; an = abs(A_NEG)
    A = (1.0/ap + 1.0/an); B = -2.0 * (t_var + v_init/ap); C = 2.0 * (s_var + v_init**2 / (2.0*ap))
    delta = B*B - 4*A*C
    if delta < -1e-9: return None, None
    delta = max(delta, 0.0)
    vmax = (-B - np.sqrt(delta)) / (2*A)
    if vmax > vmax_cap + 50.0 or vmax < 0.1:
        vmax = (-B + np.sqrt(delta)) / (2*A)
        if vmax > vmax_cap + 50.0 or vmax < 0.1: return None, None
    tc = t_var - (vmax - v_init)/ap - vmax/an
    if tc < -1e-6: return None, None
    return vmax, max(0.0, tc)

def build_full_trajectory(prefix, vmax, tc, A_POS, A_NEG, L_total):
    t0, s0, v0 = prefix['T'], prefix['L'], prefix['v_end']
    ap, an = A_POS, abs(A_NEG)
    if vmax >= v0: ta = (vmax - v0)/ap; actual_acc = ap
    else: ta = (v0 - vmax)/an; actual_acc = -an
    t1 = np.arange(0, ta + EPS_T, DT)
    v1 = v0 + actual_acc * t1
    s1 = v0*t1 + 0.5*actual_acc*t1**2
    if tc > DT*0.5:
        t2 = np.arange(DT, tc + EPS_T, DT); v2 = np.full_like(t2, vmax); s2 = vmax*t2
    else: t2, v2, s2 = np.array([]), np.array([]), np.array([])
    td = vmax/an; t3 = np.arange(DT, td + EPS_T, DT); v3 = np.maximum(vmax - an*t3, 0.0); s3 = vmax*t3 - 0.5*an*t3**2
    t_rel = np.concatenate([t1, ta+t2, ta+tc+t3])
    v_rel = np.concatenate([v1, v2, v3])
    a_rel = np.zeros_like(v_rel)
    l1, l2 = len(t1), len(t2)
    a_rel[:l1] = actual_acc; a_rel[l1:l1+l2] = 0; a_rel[l1+l2:] = -an
    s_accum = np.zeros_like(v_rel); cur_s = 0
    if len(s1): s_accum[:len(s1)] = s1; cur_s = s1[-1]
    if len(s2): s_accum[l1:l1+l2] = cur_s + s2; cur_s += s2[-1]
    if len(s3): s_accum[l1+l2:] = cur_s + s3
    t_final = np.concatenate([prefix['t'], prefix['t'][0]+t_rel])
    v_final = np.concatenate([prefix['v'], v_rel])
    a_final = np.concatenate([prefix['a'], a_rel])
    s_final = np.concatenate([prefix['s'], prefix['s'][0]+s_accum])
    if len(s_final)>0: s_final[-1] = L_total
    if len(v_final)>0: v_final[-1] = 0
    return {'t': t_final, 'v': v_final, 'a': a_final, 's': s_final, 'T': t_final[-1]}

# --- 通用区间梯形速度轨迹 ---
def get_trajectory_by_distance(v_target, L_total, p):
    v_peak = float(v_target)
    A1, A2, A_DEC, V_MID = p['A1'], p['A2'], abs(p['A_DEC']), p['V_MID']
    if v_peak <= V_MID:
        t1 = v_peak / A1; d1 = 0.5 * A1 * t1**2; t2=d2=0
    else:
        t1 = V_MID / A1; d1 = 0.5 * A1 * t1**2
        t2 = (v_peak - V_MID) / A2; d2 = V_MID*t2 + 0.5*A2*t2**2
    d_acc = d1 + d2; t_acc = t1 + t2
    t_dec = v_peak / A_DEC; d_dec = 0.5*A_DEC*t_dec**2
    if d_acc + d_dec > L_total: return get_trajectory_by_distance(v_peak*0.95, L_total, p)
    d_cruise = L_total - d_acc - d_dec; t_cruise = d_cruise/v_peak
    T_total = t_acc + t_cruise + t_dec; t_grid = np.arange(0, T_total+DT, DT)
    v_arr, a_arr = [], []
    for t in t_grid:
        if t<=t_acc:
            v=a=0
            if t<=t1: v=A1*t; a=A1
            else: v=V_MID + A2*(t-t1); a=A2
        elif t<=t_acc+t_cruise: v=v_peak; a=0
        else: dt=t-(t_acc+t_cruise); v=max(0,v_peak-A_DEC*dt); a=-A_DEC
        v_arr.append(v); a_arr.append(a)
    v_arr=np.array(v_arr); a_arr=np.array(a_arr)
    s_arr=np.cumsum(v_arr*DT)
    if len(s_arr)>0 and s_arr[-1]>0: s_arr = s_arr*L_total/s_arr[-1]
    return {'t':t_grid, 'v':v_arr, 'a':a_arr, 's':s_arr, 'T':t_grid[-1]}

# ================= 3. 历史能耗计算 =================
def calculate_energy_for_section(sp, target_time, params_row):
    grad_f, curv_f = get_track_f(sp)
    if grad_f is None: return None,None,None
    model_path = os.path.join(RES_MODEL_BASE, sp)
    if not os.path.exists(os.path.join(model_path,"best_res_model.pth")): return None,None,None
    ai = EnergyPredictor("residualV2", model_path)
    sim = TrainTheoreticalEnergyModel()
    L, V_UPPER = float(params_row['L']), float(params_row['V_UPPER'])
    if sp in SPECIAL_PARAMS:
        A1, A_DEC, MASS = SPECIAL_PARAMS[sp]["A1"], SPECIAL_PARAMS[sp]["A_DEC"], SPECIAL_PARAMS[sp]["MASS"]
        prefix = fixed_prefix_sigang(A1,A_DEC)
        t_var = target_time-prefix['T']; s_var = L-prefix['L']; v_init=prefix['v_end']
        vmax, tc = solve_vmax_analytical(s_var,t_var,v_init,A1,A_DEC,V_UPPER)
        if vmax is None: return None,None,None
        traj = build_full_trajectory(prefix,vmax,tc,A1,A_DEC,L)
    else:
        p_veh={'A1':float(params_row['A1']),'A2':float(params_row['A2']),'A_DEC':float(params_row['A_DEC']),'V_MID':float(params_row['V_MID'])}
        v_est = min(V_UPPER, L/target_time*1.05)
        traj = get_trajectory_by_distance(v_est,L,p_veh)
    grad_seq = grad_f(traj['s']); curv_seq = curv_f(traj['s'])
    mass_seq = np.full_like(traj['v'], MASS if sp in SPECIAL_PARAMS else float(params_row['MASS']))
    e_phy_seq = sim.run_batch_simulation(traj['t'], traj['v'], mass_seq[0])
    inp = np.stack([traj['v'], traj['a'], e_phy_seq, grad_seq, mass_seq, curv_seq],axis=1)
    e_res_seq = ai.predict(inp)
    L_real=min(len(e_phy_seq),len(e_res_seq))
    e_final_seq = e_phy_seq[:L_real]+e_res_seq[:L_real]
    return np.sum(e_final_seq), np.max(traj['v']), traj

# ================= 4. 主函数 =================
def run_dynamic_lookup_analysis():
    if not REPORT_FILE.exists(): return
    df_report = pd.read_csv(REPORT_FILE)
    df_sections = df_report[df_report['区间'] != '--- 全线总计 ---'].copy()
    df_params = pd.read_csv(PARAM_FILE).set_index('station_pair')
    results=[]
    for _, row in df_sections.iterrows():
        sp=row['区间']; target_t=row['历史时间']
        if sp not in df_params.index: continue
        E_hist,V_peak,traj=calculate_energy_for_section(sp,target_t,df_params.loc[sp])
        results.append({'区间':sp,'历史T':target_t,'模型E':round(E_hist,2) if E_hist else 0,
                        '实测E':row['规划能耗(Wh)'],'V_peak':round(V_peak,2) if V_peak else 0})
        if E_hist: print(f"✅ {sp:15} | 历史时间: {target_t:.2f}s | 动态能耗: {E_hist:.2f}Wh | V_peak={V_peak:.2f}m/s")
    df_final=pd.DataFrame(results)
    df_final.to_csv(os.path.join(OUTPUT_DIR,"playback_trip6_dynamic.csv"),index=False)
    print(f"\n🏁 动态回放完成 -> {OUTPUT_DIR}")

if __name__=="__main__":
    run_dynamic_lookup_analysis()