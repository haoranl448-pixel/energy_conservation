# -*- coding: utf-8 -*-
"""
optimize_meiyan_yongmaolu_fixedprefix.py
只计算【四港-曹隘】一段：
- 前段(0~约103.878s)速度时序固定，末速10.667 m/s
- 后段(余程)：以 v0=10.667 为初速，加速到 vmax→(可选匀速)→以 -0.55 刹停到 0
- 给定到达时间窗口，对每个到达时间解析求解对应 vmax，评估能耗并出图出表
"""

import os, glob, json, math, pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from scipy.optimize import minimize

# ================= 常量/路径 =================
PAIR = "泗港-曹隘"
DT = 0.05

# 线路总长（米）
L_TOTAL = 1825.0

# 前段固定所用加/减速度
A_POS = 0.50      # m/s^2
A_NEG = -1.45     # m/s^2

# 后段速度上限（不要超过训练/规则的极限；你之前给 18）
V_MAX_UPPER = 22.00# m/s

# 时间窗口：总到达时间（可按需要改）
T_MIN, T_MAX, T_STEP = 122.0, 143.0, 0.5
# 目录结构
# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
DT = 0.05  # 时间步长(固定)
SEQ_LEN = 30
DATA_DIR = Path(project_root, "data","data_processed")
NN_DIR   = Path(project_root, "output","models","nn_results_mlp1")
oUT_DIR  = Path(project_root, "output","optimization","opt_results_mlp1")
PARAMS_CSV = Path(project_root, "data", "static", "section_params_trip6.csv")
OUT_DIR = oUT_DIR / PAIR

OUT_DIR.mkdir(parents=True, exist_ok=True)#parents=True 如果父目录不存在，会自动创建所有缺失的父目录
                                            #exist_ok=True：如果目录已经存在，不会抛出异常


FEAT_JSON  = NN_DIR / "feature_columns.json"  # 训练时保存的列顺序
NN_RESULT_DIR = NN_DIR / PAIR
MODEL_PTH     = NN_RESULT_DIR / "best_model_01.pth"
SCALER_PKL    = NN_RESULT_DIR / "scaler.pkl"


EPS_T = 1e-6
EPS_S = 1e-6

# =============== NN 定义 ===============
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

# =============== 读取静态量并做按位移插值 ===============
def _read_results_concat(pair: str) -> pd.DataFrame:
    pattern = str(DATA_DIR / f"results_{pair}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"[{pair}] 未找到数据文件：{pattern}")
    dfs = []
    for fp in files:
        try: dfs.append(pd.read_excel(fp))
        except Exception as e: print(f"[跳过] 读取失败：{fp} - {e}")
    if not dfs:
        raise RuntimeError(f"[{pair}] 无可用 results_*.xlsx")
    return pd.concat(dfs, ignore_index=True)

def load_static_series(pair: str):
    df = _read_results_concat(pair)
    s_raw = pd.to_numeric(df['累计位移(m)'], errors='coerce').to_numpy()
    mask  = np.isfinite(s_raw)
    s_raw = s_raw[mask]
    df    = df.loc[mask].copy()
    order = np.argsort(s_raw)
    s = s_raw[order]
    df = df.iloc[order]

    curvature = pd.to_numeric(df.get('curvature', 0.0), errors='coerce').fillna(0.0).to_numpy()
    gradient  = pd.to_numeric(df.get('gradient', 0.0),  errors='coerce').fillna(0.0).to_numpy()
    mass_col  = pd.to_numeric(df.get('重量', 0.0),      errors='coerce').fillna(0.0).to_numpy()

    if len(s) < 2 or np.allclose(np.diff(s), 0):
        curve_f = lambda x: np.zeros_like(np.asarray(x, dtype=float))
        grade_f = lambda x: np.zeros_like(np.asarray(x, dtype=float))
        mass_f  = lambda x: np.zeros_like(np.asarray(x, dtype=float))
    else:
        curve_f = interp1d(s, curvature, fill_value='extrapolate')
        grade_f = interp1d(s, gradient,  fill_value='extrapolate')
        mass_f  = interp1d(s, mass_col,  fill_value='extrapolate')
    return curve_f, grade_f, mass_f

def load_model_and_scaler(pair: str):
    if not MODEL_PTH.exists() or not SCALER_PKL.exists():
        raise FileNotFoundError(f"[{pair}] 缺少模型或scaler：{MODEL_PTH} / {SCALER_PKL}")
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    if FEAT_JSON.exists():
        with open(FEAT_JSON, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = ['time','velocity','acceleration','prev_velocity','prev_acceleration',
                        'curvature','gradient','mass']
    net = Net(len(scaler.mean_))
    net.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
    net.eval()
    if len(feature_cols) != len(scaler.mean_):
        raise RuntimeError(f"[{pair}] scaler 维度({len(scaler.mean_)})与特征列数({len(feature_cols)})不一致")
    return net, scaler, feature_cols

# =============== 前段固定剖面 ===============
def fixed_prefix(A_POS=0.50, A_NEG=-1.45, v_to=10):
    """
    固定时序：
      0–30s +A_POS
      30–48s A_NEG
      48–84s 匀速
      84–90s +A_POS
      90–102s 匀速
      102–t6 A_NEG 到 v_to
    返回 dict: {'t','v','a','s','T','L','v_end'}

    固定时序：
        0-35s +A_POS
        35-85s 匀速
        85-t3s A_NEG到v_to
        t3-t4s +A_POS
        t4-t5 A_NEG 到 v_to
    返回 dict: {'t','v','a','s','T','L','v_end'}
    """
    def piece(t0, dur, v0, a):
        n = max(1, int(np.ceil(dur/DT)))
        t = np.linspace(0, dur, n+1)
        v = (v0 + a*t) if abs(a) > 1e-12 else np.full_like(t, v0)
        v = np.clip(v, 0.0, None)
        s = v0*t + 0.5*a*t*t if abs(a) > 1e-12 else v0*t
        return t0 + t, v, np.full_like(t, a), s

    # 段参数
    a1, a2 = A_POS, A_NEG
    t1, t2 = 35.0, 50.0

    # 逐段
    t0 = 0.0; v0 = 0.0; S_accum = 0.0
    T_all, V_all, A_all, S_all = [], [], [], []

    # 0-35 加速
    T,V,A,S = piece(t0, t1, v0, a1); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

    # 35-85 匀速
    T,V,A,S = piece(t0, t2, v0, 0); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

  
    # 85-？ 减到 v_init\
    t3 = (v0 - v_to)/abs(a2)
    T,V,A,S = piece(t0, t3, v0, a2); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

    # 拼接
    t = np.concatenate(T_all)
    v = np.concatenate(V_all)
    a = np.concatenate(A_all)
    s = np.concatenate(S_all)

    return {'t': t, 'v': v, 'a': a, 's': s, 'T': t[-1], 'L': s[-1], 'v_end': v[-1]}

# =============== 后段解析求 vmax ===============
def solve_vmax_for_time_distance(s_var, t_var, v_init, A_POS, A_NEG, vmax_cap=np.inf):
    """
    后段模型：减速到v_init→加速到 vmax（+A_POS）→ 匀速 tc → 以 |A_NEG| 刹停到 0
    方程：
      t_var = (vmax - v_init)/A_POS + tc + vmax/|A_NEG|+（24.44-v_init）/|A_NEG|
      s_var = (vmax^2 - v_init^2)/(2A_POS) + vmax*tc + vmax^2/(2|A_NEG|)+（24.44^2 - v_init^2)/(2|A_NEG|）
    消去 tc 得关于 vmax 的二次方程 A v^2 + B v + C = 0
    返回 (vmax, tc)；若不可解，返回 (None, None)
    """
    if t_var <= 0 or s_var <= 0: return None, None
    ap = A_POS; an = abs(A_NEG)
    A = (1.0/ap + 1.0/an)
    B = -2.0 * (t_var + v_init/ap-(24.44-v_init)/an)
    C =  2.0 * (s_var + (v_init*v_init)/(2.0*ap)-(24.44*24.44 - v_init*v_init)/(2*an))
    D = B*B - 4*A*C
    candidates = []
    if D >= -1e-9:
        D = max(D, 0.0)
        for sign in (+1, -1):
            vmax = (-B + sign*math.sqrt(D)) / (2*A)
            if not np.isfinite(vmax): continue
            if vmax < v_init - 1e-9 or vmax > vmax_cap + 1e-9: continue
            tc = t_var - (vmax - v_init)/ap - vmax/an
            if tc >= -1e-6:
                candidates.append((float(vmax), max(0.0, float(tc))))
    if not candidates:
        # 三角型（tc=0）退化检验
        vmax = (t_var + v_init/ap) / (1.0/ap + 1.0/an)
        if v_init - 1e-9 <= vmax <= vmax_cap + 1e-9:
            s_tri = (vmax*vmax - v_init*v_init)/(2.0*ap) + (vmax*vmax)/(2.0*an)
            if abs(s_tri - s_var) <= max(1e-3, 1e-5*s_var):
                return float(vmax), 0.0
        return None, None
    # 取较小的一个 vmax（都可行）
    candidates.sort(key=lambda x: x[0])
    return candidates[0]

def build_tail_profile(s0, t0, v0, vmax, tc, A_POS, A_NEG, L_target):
    ap = A_POS; an = abs(A_NEG)
    ta = max(0.0, (vmax - v0)/ap)
    td = max(0.0, vmax/an)

    # 加速段
    t1 = np.arange(0.0, ta + EPS_T, DT)
    v1 = v0 + ap*t1
    s1 = v0*t1 + 0.5*ap*t1*t1

    # 匀速段
    if tc > DT*0.5:
        t2 = np.arange(DT, tc + EPS_T, DT)
        v2 = np.full_like(t2, vmax)
        s2 = vmax*t2
    else:
        t2 = np.array([])
        v2 = np.array([])
        s2 = np.array([])

    # 减速段
    t3 = np.arange(DT, td + EPS_T, DT)
    v3 = np.maximum(vmax - an*t3, 0.0)
    s3 = vmax*t3 - 0.5*an*t3*t3

    # 拼接（相对时间）
    t_rel = np.concatenate([t1, ta + t2, ta + tc + t3])
    v_rel = np.concatenate([v1, v2, v3])
    s_rel = np.concatenate([s1,
                            (s1[-1:] + s2) if s2.size>0 else s1[-1:],
                            ( (s1[-1] + (s2[-1] if s2.size>0 else 0.0)) + s3 )])

    # 转绝对
    t = t0 + t_rel
    s = s0 + s_rel
    a = np.zeros_like(t_rel)
    a[:t1.size] = ap
    if t2.size>0: a[t1.size:t1.size+t2.size] = 0.0
    a[t1.size+t2.size:] = -an

    # 尾点对齐
    s[-1] = L_target
    v_rel[-1] = 0.0
    a[-1] = 0.0
    return {'t': t, 'v': v_rel, 'a': a, 's': s, 'T': t[-1]}

# =============== 特征构造 & 能耗评估 ===============
def build_features(prof, feature_cols, curve_f, grade_f, mass_f):
    t = prof['t']; v = prof['v']; a = prof['a']; s = prof['s']
    prev_v = np.roll(v, 1); prev_a = np.roll(a, 1)
    prev_v[0] = v[0]; prev_a[0] = a[0]
    curv = curve_f(s); grad = grade_f(s); mass = mass_f(s)

    # 防nan
    curv = np.nan_to_num(curv); grad = np.nan_to_num(grad); mass = np.nan_to_num(mass)

    bank = {
        'time': t, 'velocity': v, 'acceleration': a,
        'prev_velocity': prev_v, 'prev_acceleration': prev_a,
        'curvature': curv, 'gradient': grad, 'mass': mass,
    }
    X = np.stack([np.asarray(bank.get(c, np.zeros_like(t)), dtype=float) for c in feature_cols], axis=1)
    X = np.nan_to_num(X)
    return X

def eval_energy_wh(net, scaler, X):
    Xs = scaler.transform(X)
    with torch.no_grad():
        e_inc = net(torch.tensor(Xs, dtype=torch.float32)).cpu().numpy().reshape(-1)
    e_inc = np.nan_to_num(e_inc)
    return float(np.sum(e_inc)), e_inc, np.cumsum(e_inc)

# =============== 主流程 ===============
def main():
    net, scaler, feature_cols = load_model_and_scaler(PAIR)
    curve_f, grade_f, mass_f = load_static_series(PAIR)

    # 1) 计算固定前段
    prefix = fixed_prefix(A_POS=A_POS, A_NEG=A_NEG, v_to=10)
    T_fix, S_fix, v_init = prefix['T'], prefix['L'], prefix['v_end']
    print(f"[固定段] T_fix={T_fix:.3f}s, S_fix={S_fix:.3f}m, v_end={v_init:.3f}m/s")

    # 2) 扫描总到达时间，解析求 vmax 并评估能耗
    rows = []
    t_list = np.arange(T_MIN, T_MAX + 1e-12, T_STEP)
    for T_tot in t_list:
        t_var = T_tot - T_fix
        s_var = L_TOTAL - S_fix
        if t_var <= 0 or s_var <= 0:
            continue
        vmax, tc = solve_vmax_for_time_distance(s_var, t_var, v_init, A_POS, A_NEG, V_MAX_UPPER)
        if vmax is None:
            continue
        # 拼完整 v-t
        tail = build_tail_profile(S_fix, T_fix, v_init, vmax, tc, A_POS, A_NEG, L_TOTAL)
        t = np.concatenate([prefix['t'], tail['t'][1:]])
        v = np.concatenate([prefix['v'], tail['v'][1:]])
        a = np.concatenate([prefix['a'], tail['a'][1:]])
        s = np.concatenate([prefix['s'], tail['s'][1:]])
        prof = {'t': t, 'v': v, 'a': a, 's': s, 'T': t[-1]}
        # 特征 & 能耗
        X = build_features(prof, feature_cols, curve_f, grade_f, mass_f)
        E_wh, _, _ = eval_energy_wh(net, scaler, X)
        rows.append([T_tot, vmax, E_wh])

    if not rows:
        raise RuntimeError("时间扫描无可行点，请调整 T_MIN/T_MAX 或 V_MAX_UPPER。")

    df_time = pd.DataFrame(rows, columns=['t_arrival','v_peak_opt','E_wh'])
    df_time.sort_values('t_arrival', inplace=True)
    df_time.to_csv(OUT_DIR / f"time_energy_curve_{PAIR}.csv", index=False)

    plt.figure(figsize=(7,4))
    plt.plot(df_time['t_arrival'], df_time['E_wh'], lw=1.2)
    plt.xlabel('Arrival Time (s)'); plt.ylabel('Energy (Wh)')
    plt.title(f'Time vs Min Energy — {PAIR}')
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(OUT_DIR / "time_energy_curve.png", dpi=300); plt.close()

    # 3) 选能耗最低的一个，输出 v-t 与能耗累计曲线
    idx = int(np.argmin(df_time['E_wh'].values))
    T_best = float(df_time.iloc[idx]['t_arrival']); vmax_best = float(df_time.iloc[idx]['v_peak_opt'])
    t_var = T_best - T_fix; s_var = L_TOTAL - S_fix
    vmax, tc = solve_vmax_for_time_distance(s_var, t_var, v_init, A_POS, A_NEG, V_MAX_UPPER)
    tail = build_tail_profile(S_fix, T_fix, v_init, vmax, tc, A_POS, A_NEG, L_TOTAL)
    t = np.concatenate([prefix['t'], tail['t'][1:]])
    v = np.concatenate([prefix['v'], tail['v'][1:]])
    a = np.concatenate([prefix['a'], tail['a'][1:]])
    s = np.concatenate([prefix['s'], tail['s'][1:]])
    prof = {'t': t, 'v': v, 'a': a, 's': s, 'T': t[-1]}

    # 保存 v-t
    pd.DataFrame({'time (s)': t, 'velocity (m/s)': v}).to_csv(OUT_DIR / "optimal_vt_curve.csv", index=False)
    plt.figure(figsize=(7,4))
    plt.plot(t, v, lw=2)
    plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)')
    plt.title(f'Optimal v–t (vmax={vmax:.2f} m/s, T={T_best:.2f}s)')
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(OUT_DIR / "optimal_vt.png", dpi=300); plt.close()

    # 累计能耗
    X = build_features(prof, feature_cols, curve_f, grade_f, mass_f)
    E_wh, e_inc, e_cum = eval_energy_wh(net, scaler, X)
    plt.figure(figsize=(7,4))
    plt.step(t, e_cum, where='post', lw=2)
    plt.xlabel('Time (s)'); plt.ylabel('Cumulative Energy (Wh)')
    plt.title('Optimal E–t'); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(OUT_DIR / "optimal_energy.png", dpi=300); plt.close()

    # 摘要
    pd.DataFrame([{
        'station_pair': PAIR,
        'v_peak_opt(m/s)': vmax,
        't_arrival_opt(s)': T_best,
        'E_opt(Wh)': E_wh,
        'T_fixed(s)': T_fix, 'S_fixed(m)': S_fix, 'v_init(m/s)': v_init,
        'A_POS': A_POS, 'A_NEG': A_NEG,
        'V_MAX_UPPER': V_MAX_UPPER,
        'window': f'[{T_MIN},{T_MAX}] step {T_STEP}',
        'note': 'fixed prefix then single-variable vmax tail (accelerate→cruise→brake to 0)'
    }]).to_csv(OUT_DIR / "best_solution.csv", index=False)

    print(f"✅ 完成！输出目录：{OUT_DIR.resolve()}")
    print("   - time_energy_curve_泗港-曹隘.csv")
    print("   - time_energy_curve.png")
    print("   - optimal_vt_curve.csv")
    print("   - optimal_vt.png")
    print("   - optimal_energy.png")
    print("   - best_solution.csv")

if __name__ == "__main__":
    main()
