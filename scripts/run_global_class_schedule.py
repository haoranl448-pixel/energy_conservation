# -*- coding: utf-8 -*-
import os, math, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import sys, warnings

warnings.filterwarnings("ignore")

# ===================== 1. 环境与路径配置 =====================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 核心调度参数 ---
T_TOTAL_TARGET = 3652.80  # 全程总时长（含停站）
#T_TOTAL_TARGET = 690
DWELL_TIME = 30.0       # 默认停站时间 30s
DT = 0.05               # 轨迹重建步长
EPS_T = 1e-6

STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区", "南高教园区-下应路",
    "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘", "柳隘-海晏北路",
    "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路", "院士路-盎孟港", "盎孟港-三官堂",
    "三官堂-兴庄路", "兴庄路-兴海南路", "兴海南路-梅堰","梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥",
    
]

# MANUAL_CONSTRAINTS = {
#     "庙堰-钟公庙": 3,       # 强制要求布政-张家潭跑 Class 3
#     "南高教园区-下应路":3,
#     "钱湖南路-南高教园区":2,
#     "下应路-大洋江":3,
#     "泗港-曹隘":3,
#     "曹隘-柳隘":3,
#     "梅堰-永茂路":4,
#     "永茂路-镇海大道":2,
#     "院士路-盎孟港":3

#       # 强制要求这一站跑 Class 4
# }

MANUAL_CONSTRAINTS = {
    

      # 强制要求这一站跑 Class 4
}


# # 路径对齐
# OPT_BASE = Path(project_root) / "output" / "optimization" / "residual1"
# CLASS_TABLE = Path(project_root) / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
# PARAMS_CSV = Path(project_root) / "data" / "static" / "section_params_trip6.csv"
# OUT_DIR = Path(project_root) / "output" / "schedule" / "trip6_class_results"
# SEG_DIR = OUT_DIR / "segments"


# 路径对齐
OPT_BASE = Path(project_root) / "output" / "optimization" / "trip6_v2_6feat"
CLASS_TABLE = Path(project_root) / "output" / "analysis" / "class_tables_strict" / "standard_class_times.csv"
PARAMS_CSV = Path(project_root) / "data" / "static" / "section_params_trip6.csv"
OUT_DIR = Path(project_root) / "output" / "schedule" / "trip6_class_results_residualV2"
SEG_DIR = OUT_DIR / "segments"




OUT_DIR.mkdir(parents=True, exist_ok=True)
SEG_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 2. 轨迹生成算法库 (包含特殊站与普通站) =====================

def solve_vmax_analytical(s_var, t_var, v_init, A_POS, A_NEG):
    ap, an = A_POS, abs(A_NEG)
    A = (1.0/ap + 1.0/an); B = -2.0 * (t_var + v_init/ap); C = 2.0 * (s_var + v_init**2 / (2.0*ap))
    delta = max(B*B - 4*A*C, 0.0)
    vmax = (-B - math.sqrt(delta)) / (2*A)
    if vmax < v_init: vmax = (-B + math.sqrt(delta)) / (2*A)
    tc = t_var - (vmax - v_init)/ap - vmax/an
    return vmax, max(0.0, tc)

def build_special_traj(prefix, vmax, tc, A_POS, A_NEG, L_total):
    t0, s0, v0 = prefix['T'], prefix['L'], prefix['v_end']
    ap, an = A_POS, abs(A_NEG)
    ta = (vmax - v0)/ap if vmax >= v0 else (v0 - vmax)/an
    a_val = ap if vmax >= v0 else -an
    t1 = np.arange(0, ta + EPS_T, DT); v1 = v0 + a_val * t1
    t2 = np.arange(DT, tc + EPS_T, DT) if tc > DT*0.5 else np.array([]); v2 = np.full_like(t2, vmax)
    td = vmax/an; t3 = np.arange(DT, td + EPS_T, DT); v3 = np.maximum(vmax - an*t3, 0.0)
    t_f = np.concatenate([prefix['t'], t0+t1, t0+ta+t2, t0+ta+tc+t3])
    v_f = np.concatenate([prefix['v'], v1, v2, v3])
    s_f = np.cumsum(v_f * DT)
    return {"t": t_f, "v": v_f, "s": s_f}

def fixed_prefix_meiyan(A_POS, A_NEG):
    v_to = 10.667
    def piece(t0, dur, v0, a):
        n = max(1, int(np.ceil(dur/DT)))
        t = np.linspace(0, dur, n+1); v = (v0 + a*t) if abs(a) > 1e-12 else np.full_like(t, v0)
        v = np.clip(v, 0.0, None); s = v0*t + 0.5*a*t*t if abs(a) > 1e-12 else v0*t
        return t0 + t, v, s
    t0, v0, S_accum = 0.0, 0.0, 0.0
    T_all, V_all, S_all = [], [], []
    ops = [(30.0, A_POS), (18.0, A_NEG), (36.0, 0.0), (6.0, A_POS), (12.0, 0.0)]
    for dur, a in ops:
        T,V,S = piece(t0, dur, v0, a)
        T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
        t0,v0,S_accum=T[-1],V[-1],S_all[-1][-1]
    t6 = (v0 - v_to)/abs(A_NEG); T,V,S = piece(t0, t6, v0, A_NEG)
    T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
    return {'t': np.concatenate(T_all), 'v': np.concatenate(V_all), 'T': T_all[-1][-1], 'L': S_all[-1][-1], 'v_end': V_all[-1][-1]}

def fixed_prefix_sigang(A_POS, A_NEG):
    v_to = 10.0
    def piece(t0, dur, v0, a):
        n = max(1, int(np.ceil(dur/DT)))
        t = np.linspace(0, dur, n+1); v = (v0 + a*t) if abs(a) > 1e-12 else np.full_like(t, v0)
        v = np.clip(v, 0.0, None); s = v0*t + 0.5*a*t*t if abs(a) > 1e-12 else v0*t
        return t0 + t, v, s
    t0, v0, S_accum = 0.0, 0.0, 0.0
    T_all, V_all, S_all = [], [], []
    ops = [(35.0, A_POS), (50.0, 0.0)]
    for dur, a in ops:
        T,V,S = piece(t0, dur, v0, a)
        T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
        t0,v0,S_accum=T[-1],V[-1],S_all[-1][-1]
    t3 = (v0 - v_to)/abs(A_NEG); T,V,S = piece(t0, t3, v0, A_NEG)
    T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
    return {'t': np.concatenate(T_all), 'v': np.concatenate(V_all), 'T': T_all[-1][-1], 'L': S_all[-1][-1], 'v_end': V_all[-1][-1]}

def make_common_trapezoid(t_arrival, v_peak, A1, A2, A_DEC, V_MID):
    t1 = min(v_peak, V_MID) / A1
    t2 = max(0.0, (v_peak - V_MID) / A2)
    t_dec = v_peak / abs(A_DEC)
    t_cruise = max(0.0, t_arrival - (t1 + t2 + t_dec))
    t = np.arange(0.0, t_arrival + EPS_T, DT)
    v = np.zeros_like(t)
    for i, tt in enumerate(t):
        if tt < t1: v[i] = A1 * tt
        elif tt < t1 + t2: v[i] = V_MID + A2 * (tt - t1)
        elif tt < t1 + t2 + t_cruise: v[i] = v_peak
        else:
            td = tt - (t1 + t2 + t_cruise)
            v[i] = max(v_peak - abs(A_DEC) * td, 0.0)
    return {"t": t, "v": v, "s": np.cumsum(v * DT)}

# ===================== 3. DP 核心算法 (带容差且能耗最优) =====================

def dp_solve_logic(choices, t_run_target):
    target_int = int(round(t_run_target * 10))
    slack_int = 100 # 5秒容差
    dp = {0: 0.0}
    path = [{}]
    for i, opts in enumerate(choices):
        nxt, ch = {}, {}
        for t_p, e_p in dp.items():
            for o in opts:
                t_s = t_p + int(round(o['t'] * 10))
                if t_s > target_int + slack_int + 200: continue
                e_s = e_p + o['E']
                if t_s not in nxt or e_s < nxt[t_s]:
                    nxt[t_s] = e_s; ch[t_s] = (t_p, o)
        if not nxt: return None
        dp, path = nxt, path + [ch]
    
    feasible_times = [t for t in dp.keys() if abs(t - target_int) <= slack_int]
    best_t = min(feasible_times, key=lambda x: dp[x]) if feasible_times else min(dp.keys(), key=lambda x: abs(x - target_int))
    
    res, curr_t = [], best_t
    for i in range(len(choices), 0, -1):
        prev_t, o = path[i][curr_t]; res.append(o); curr_t = prev_t
    res.reverse()
    return res, dp[best_t], best_t / 10.0

# ===================== 4. 完整的 MAIN 函数 =====================

def main():
    print(f"🚀 启动 Trip 6 等级优化与全量输出流程...")
    
    # 1. 加载数据 (锁定 Class 2-4)
    if not CLASS_TABLE.exists():
        print(f"❌ 错误：找不到标准时间表 {CLASS_TABLE}"); return
    df_std = pd.read_csv(CLASS_TABLE).set_index('区段')
    df_params = pd.read_csv(PARAMS_CSV).set_index('station_pair')
    
    all_station_opts = []
    for sp in STATIONS:
        f_p = OPT_BASE / sp / f"time_energy_curve_{sp}.csv"
        if not f_p.exists(): 
            print(f"⚠️ 跳过 {sp}: 找不到能耗曲线"); continue
        df_p = pd.read_csv(f_p)


        opts = []


        forced_class_idx = MANUAL_CONSTRAINTS.get(sp)
        
        if forced_class_idx:
            # 如果有强制要求，搜索范围只包含这一个等级
            search_range = [forced_class_idx]
            print(f"   📍 站点 {sp} 已被锁定为 Class {forced_class_idx}")
        else:
            # 如果没有要求，依然在 2, 3, 4 之间优化
            search_range = [1, 2, 3, 4, 5]

        for i in search_range: # 🌟 核心修改：只选 2, 3, 4 档位
            c_label = f"Class{i}"
            if c_label not in df_std.columns: continue

            t_s = float(df_std.loc[sp, c_label])
            if np.isnan(t_s): continue
            
            idx = (df_p['t_arrival'] - t_s).abs().idxmin()
            opts.append({'class': c_label, 't': t_s, 'E': df_p.loc[idx, 'E_wh'], 'v_peak': df_p.loc[idx, 'v_peak_opt'], 'sp': sp})
        all_station_opts.append(opts)

    # 2. 目标运行时间计算
    T_RUN_TARGET = T_TOTAL_TARGET - (len(STATIONS)-1)*DWELL_TIME
    
    # 可行性自检
    min_t = sum([min([o['t'] for o in opts]) for opts in all_station_opts])
    max_t = sum([max([o['t'] for o in opts]) for opts in all_station_opts])
    print(f"📊 可行域区间: [{min_t:.1f}s, {max_t:.1f}s] | 目标: {T_RUN_TARGET:.1f}s")

    # 3. 运行 DP 核心算法
    result = dp_solve_logic(all_station_opts, T_RUN_TARGET)
    if not result:
        print("❌ 规划失败：无法满足约束条件。"); return
    alloc, total_E, t_run_actual = result

    # 4. 轨迹重建与文件生成
    t_global, s_global = 0.0, 0.0
    full_vts, dwell_spans, table_rows = [], [], []

    print(f"🛠️ 正在重建全线轨迹并保存文件...")
    for i, o in enumerate(alloc):
        sp = o['sp']
        p = df_params.loc[sp]
        
        # 轨迹算法选择
        if sp == "梅堰-永茂路":
            prefix = fixed_prefix_meiyan(p['A1'], p['A_DEC'])
            vmax, tc = solve_vmax_analytical(p['L'] - prefix['L'], o['t'] - prefix['T'], prefix['v_end'], p['A1'], p['A_DEC'])
            prof = build_special_traj(prefix, vmax, tc, p['A1'], p['A_DEC'], p['L'])
        elif sp == "泗港-曹隘":
            prefix = fixed_prefix_sigang(p['A1'], p['A_DEC'])
            vmax, tc = solve_vmax_analytical(p['L'] - prefix['L'], o['t'] - prefix['T'], prefix['v_end'], p['A1'], p['A_DEC'])
            prof = build_special_traj(prefix, vmax, tc, p['A1'], p['A_DEC'], p['L'])
        else:
            prof = make_common_trapezoid(o['t'], o['v_peak'], p['A1'], p['A2'], p['A_DEC'], p['V_MID'])
        
        # 保存分段 CSV/PNG
        sta_dir = SEG_DIR / sp; sta_dir.mkdir(exist_ok=True)
        pd.DataFrame({'t_local': prof['t'], 'v_ms': prof['v'], 's_local': prof['s']}).to_csv(sta_dir / "vt_selected.csv", index=False)
        plt.figure(figsize=(6,3)); plt.plot(prof['t'], prof['v']*3.6, color='green')
        plt.title(f"{sp} ({o['class']})"); plt.grid(True, alpha=0.3); plt.savefig(sta_dir / "vt_selected.png", dpi=150); plt.close()

        # 汇总数据
        full_vts.append(pd.DataFrame({'t_global': prof['t'] + t_global, 'v_ms': prof['v'], 's_global': prof['s'] + s_global}))
        table_rows.append({'station_pair': sp, 'selected_class': o['class'], 'target_time(s)': o['t'], 'energy(Wh)': o['E']})
        
        t_global += o['t']
        s_global += prof['s'][-1]
        if i < len(alloc)-1:
            dwell_spans.append((t_global, t_global + DWELL_TIME))
            t_global += DWELL_TIME

    # 5. 导出全线大表与敏感性表
    pd.DataFrame(table_rows).to_csv(OUT_DIR / "schedule_table.csv", index=False, encoding='utf-8-sig')
    
    df_full = pd.concat(full_vts)
    vt_name = f"full_vt_schedule_total{int(T_TOTAL_TARGET)}s_run{int(t_run_actual)}s.csv"
    df_full.to_csv(OUT_DIR / vt_name, index=False)

    # 6. 绘制全线 VT 图
    plt.figure(figsize=(15, 5))
    plt.plot(df_full['t_global'], df_full['v_ms']*3.6, color='blue', lw=1)
    for t1, t2 in dwell_spans: plt.axvspan(t1, t2, color='gray', alpha=0.15)
    plt.title(f"Full-Line Optimized Timetable (Total E: {total_E/1000:.2f}kWh)"); plt.ylabel("km/h"); plt.xlabel("Time (s)")
    plt.savefig(OUT_DIR / "full_vt.png", dpi=300); plt.close()

    # 7. 敏感性分析 (输出 energy_sensitivity_run_minus_1_to_5s.csv)
    sens = []
    for d in range(0, 6):
        s_res = dp_solve_logic(all_station_opts, T_RUN_TARGET - d)
        if s_res: sens.append({'delta_s': d, 'energy_Wh': s_res[1], 'feasible': True})
    pd.DataFrame(sens).to_csv(OUT_DIR / "energy_sensitivity_run_minus_1_to_5s.csv", index=False)

    print(f"\n✅ 全部完成！文件位置: {OUT_DIR}")

if __name__ == "__main__":
    main()