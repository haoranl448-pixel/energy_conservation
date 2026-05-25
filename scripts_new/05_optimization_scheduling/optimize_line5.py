#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_trapezoid_batch.py
一次性运行宁波地铁 5 号线 26 个区间（或参数表里列出的区间）：
- 从 section_params.csv 读取每段参数（L, V_UPPER, V_MID, A1, A2, A_DEC, REACH_TOL, TIME_LOWER, TIME_UPPER, MASS）
- 对每个 station_pair 复用你原脚本的逻辑：
    * 加速-匀速-减速 梯形 v-t 生成
    * 到达约束（位移与 L 的偏差 REACH_TOL）
    * 全局最优（仅作为参考图）
    * 给定时间的扫描并输出 time_energy_curve_{station_pair}.csv
- 每段输出目录：opt_results/{station_pair}/
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




# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# ========= 常量 =========
DT = 0.05  # 时间步长(固定)
SEQ_LEN = 30
DATA_DIR = Path(project_root, "data","data_processed")
NN_DIR   = Path(project_root, "output","models","nn_results_mlp1")
OUT_DIR  = Path(project_root, "output","optimization","opt_results_mlp1")
PARAMS_CSV = Path(project_root, "data", "static", "section_params_trip6.csv")




# ========= 常量 =========


# ======== 5号线 26 段顺序（可按需改/裁剪）========
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区", "南高教园区-下应路",
    "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘", "柳隘-海晏北路",
    "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路", "院士路-盎孟港", "盎孟港-三官堂",
    "三官堂-兴庄路", "兴庄路-兴海南路", "兴海南路-梅堰", "永茂路-镇海大道", "镇海大道-骆驼桥",
]

# --------- 你的神经网络（保持不变） ---------
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

# ========= 单段运行（最大限度沿用你原脚本的写法） =========
def run_one_segment(params_row):
    station_pair = str(params_row['station_pair']).strip()
    L           = float(params_row['L'])
    V_UPPER     = float(params_row['V_UPPER'])
    V_MID       = float(params_row['V_MID'])
    A1          = float(params_row['A1'])
    A2          = float(params_row['A2'])
    A_DEC       = float(params_row['A_DEC'])
    REACH_TOL   = float(params_row['REACH_TOL'])
    TIME_LOWER  = float(params_row['TIME_LOWER'])
    TIME_UPPER  = float(params_row['TIME_UPPER'])
    MASS        = float(params_row['MASS'])

    # === 路径（与单段脚本一致的结构） ===
    DATA_XLSX_exact = DATA_DIR / f"results_{station_pair}.xlsx"
    if DATA_XLSX_exact.exists():
        data_file = DATA_XLSX_exact
    else:
        # 兼容多天/LOCKED：取第一个匹配
        cand = sorted(glob.glob(str(DATA_DIR / f"results_{station_pair}*.xlsx")))
        if not cand:
            raise FileNotFoundError(f"[{station_pair}] 未找到数据文件 results_{station_pair}*.xlsx")
        data_file = Path(cand[0])

    NN_RESULT_DIR = NN_DIR / station_pair
    MODEL_PTH     = NN_RESULT_DIR / "best_model_01.pth"
    SCALER_PKL    = NN_RESULT_DIR / "scaler.pkl"
    if not MODEL_PTH.exists() or not SCALER_PKL.exists():
        raise FileNotFoundError(f"[{station_pair}] 缺少模型或 scaler：{MODEL_PTH} / {SCALER_PKL}")

    out_dir = OUT_DIR / station_pair
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 加载 NN & scaler（保持不变） ===
    with open(SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)
    net = Net(len(scaler.mean_))
    net.load_state_dict(torch.load(MODEL_PTH, map_location='cpu'))
    net.eval()

    # === 读取 & 插值静态数据（与你原脚本一致） ===
    df_raw = pd.read_excel(data_file)
    s_raw  = df_raw['累计位移(m)'].values
    # 去除重复/回退
    mask = np.concatenate([[True], np.diff(s_raw) > 1e-6])
    df   = df_raw[mask]
    s    = df['累计位移(m)'].values

    curve_f = interp1d(s, df['curvature'].values, fill_value='extrapolate')
    grade_f = interp1d(s, df['gradient'].values,  fill_value='extrapolate')
    def mass_f(s_arr):
        return MASS * np.ones_like(s_arr)

    # === 速度–时间曲线（沿用） ===
    def make_profile(v_peak, t_arr):
        if v_peak <= V_MID:
            t_acc = v_peak / A1
        else:
            t_acc = V_MID / A1 + (v_peak - V_MID) / A2
        t_dec = v_peak / abs(A_DEC)
        t_cruise = max(0.0, t_arr - t_acc - t_dec)
        T_tot    = t_acc + t_cruise + t_dec

        t_grid = np.arange(0.0, T_tot + DT, DT)
        v = np.zeros_like(t_grid)
        a = np.zeros_like(t_grid)
        for i, t in enumerate(t_grid):
            if t < t_acc:
                if v_peak <= V_MID:
                    v[i], a[i] = A1 * t, A1
                else:
                    tb = V_MID / A1
                    if t <= tb:
                        v[i], a[i] = A1 * t, A1
                    else:
                        v[i], a[i] = V_MID + A2 * (t - tb), A2
            elif t < t_acc + t_cruise:
                v[i], a[i] = v_peak, 0.0
            else:
                td = t - (t_acc + t_cruise)
                vt = v_peak + A_DEC * td
                v[i], a[i] = max(vt, 0.0), A_DEC
        s = np.cumsum(v * DT)
        return {'t': t_grid, 'v': v, 'a': a, 's': s, 'T': T_tot}

    # === 目标函数（沿用） ===
    def objective(x):
        # 强制转换为标量，防止 minimize 传入数组导致 make_profile 报错
        v_p = float(x[0])
        t_arr = float(x[1])
        
        if not (0 < v_p <= V_UPPER and TIME_LOWER <= t_arr <= TIME_UPPER):
            return 1e9
        prof = make_profile(v_p, t_arr)
        if abs(prof['s'][-1] - L) > REACH_TOL:
            return 1e9 + abs(prof['s'][-1] - L) * 1e6

        curv = curve_f(prof['s']); grad = grade_f(prof['s']); mass = mass_f(prof['s'])
        X = np.stack([
            prof['t'], prof['v'], prof['a'],
            np.roll(prof['v'], 1), np.roll(prof['a'], 1),
            curv, grad, mass
        ], axis=1)
        X[0, 3], X[0, 4] = X[0, 1], X[0, 2]
        Xs = scaler.transform(X)
        with torch.no_grad():
            e_inc = net(torch.tensor(Xs, dtype=torch.float32)).cpu().numpy().flatten()
        return e_inc.sum()

    # === 1) 全局最优（保持，以便画图/校核） ===
    print(f"\n[{station_pair}] 开始全局优化…")
    res = minimize(
        objective,
        x0=[V_UPPER * 0.6, (TIME_LOWER + TIME_UPPER) / 2],
        bounds=[(0.1, V_UPPER), (TIME_LOWER, TIME_UPPER)],
        method='L-BFGS-B',
        options={'ftol':1e-6, 'maxiter':200}
    )
    if not res.success:
        print(f"[{station_pair}] 警告：全局优化未声明 success：{res.message}")
    vp_opt, t_opt = res.x
    E_opt         = float(res.fun)
    print(f"[{station_pair}] 全局最优 → v_peak={vp_opt:.2f} m/s, t_arrival={t_opt:.2f} s, Energy={E_opt:.2f} Wh")

    # 保存 v-t 图/CSV
    prof = make_profile(vp_opt, t_opt)
    plt.figure(); plt.plot(prof['t'], prof['v'], lw=2)
    plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)')
    plt.title('Optimal v–t'); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/'optimal_vt.png', dpi=300); plt.close()
    pd.DataFrame({'time (s)': prof['t'], 'velocity (m/s)': prof['v']}).to_csv(out_dir/'optimal_vt_curve.csv', index=False)

    # 最优能耗曲线
    curv = curve_f(prof['s']); grad = grade_f(prof['s']); mass = mass_f(prof['s'])
    X    = np.stack([
        prof['t'], prof['v'], prof['a'],
        np.roll(prof['v'],1), np.roll(prof['a'],1),
        curv, grad, mass
    ], axis=1)
    X[0,3], X[0,4] = X[0,1], X[0,2]
    with torch.no_grad():
        e_inc = net(torch.tensor(scaler.transform(X), dtype=torch.float32)).cpu().numpy().flatten()
    e_cum = np.cumsum(e_inc)
    plt.figure(); plt.step(prof['t'], e_cum, where='post', lw=2)
    plt.xlabel('Time (s)'); plt.ylabel('Cumulative Energy (Wh)')
    plt.title('Optimal E–t'); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/'optimal_energy.png', dpi=300); plt.close()

    # === 2) 时间扫描 + 可达性过滤 + 输出曲线 CSV（关键产物） ===
    t_list  = np.arange(TIME_LOWER, TIME_UPPER + 1e-9, DT)
    filtered = []
    print(f"[{station_pair}] 开始采样并过滤不可达点…")
# === 2) 时间扫描部分（修改后） ===
    for t in t_list:
        # 修改 lambda，确保取 vp 的第一个元素 vp[0]
        rv = minimize(lambda vp: objective([vp[0], t]), 
                        x0=[min(V_UPPER*0.6, V_UPPER-1e-3)],
                        bounds=[(0.1, V_UPPER)],
                        method='L-BFGS-B', options={'ftol':1e-6, 'maxiter':100})
        if not rv.success:
            continue
        vp_i = float(rv.x[0])
        prof_i = make_profile(vp_i, t)
        if abs(prof_i['s'][-1] - L) > REACH_TOL:
            continue

        curv_i = curve_f(prof_i['s']); grad_i = grade_f(prof_i['s']); mass_i = mass_f(prof_i['s'])
        X_i    = np.stack([
            prof_i['t'], prof_i['v'], prof_i['a'],
            np.roll(prof_i['v'],1), np.roll(prof_i['a'],1),
            curv_i, grad_i, mass_i
        ], axis=1)
        X_i[0,3], X_i[0,4] = X_i[0,1], X_i[0,2]
        with torch.no_grad():
            e_inc_i = net(torch.tensor(scaler.transform(X_i), dtype=torch.float32)).cpu().numpy().flatten()
        E_i = float(e_inc_i.sum())

        filtered.append((float(t), vp_i, E_i))

    if not filtered:
        print(f"[{station_pair}] ⚠️ 未得到任何可达采样点，请检查 TIME_LOWER/TIME_UPPER、L 或加减速度参数。")
    else:
        t_f, v_f, E_f = map(list, zip(*filtered))
        df_curve = pd.DataFrame({'t_arrival': t_f, 'v_peak_opt': v_f, 'E_wh': E_f})
        csv_path = out_dir / f"time_energy_curve_{station_pair}.csv"
        df_curve.to_csv(csv_path, index=False)
        print(f"[{station_pair}] 曲线数据保存到: {csv_path}")

        # 采样最低点图
        idx_min = int(np.argmin(E_f)); t_min = t_f[idx_min]; E_min = E_f[idx_min]
        plt.figure(figsize=(6,4))
        plt.plot(t_f, E_f, '-', lw=1, label='E_min(t)')
        plt.scatter([t_min], [E_min], c='red', s=40, label='Sampling Min')
        plt.xlabel('Arrival Time (s)'); plt.ylabel('Energy (Wh)')
        plt.title(f'{station_pair} — Time vs Min Energy')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(out_dir/'time_energy_curve.png', dpi=300); plt.close()

    print(f"[{station_pair}] ✅ 完成，输出目录：{out_dir.resolve()}")

# ========= 主程序：读参数表 → 按白名单顺序逐段运行 =========
def main():
    if not PARAMS_CSV.exists():
        raise FileNotFoundError(f"未找到参数表：{PARAMS_CSV}。")

    params = pd.read_csv(PARAMS_CSV)
    need_cols = {'station_pair','L','V_UPPER','V_MID','A1','A2','A_DEC','REACH_TOL','TIME_LOWER','TIME_UPPER','MASS'}
    if not need_cols.issubset(params.columns):
        missing = list(need_cols - set(params.columns))
        raise ValueError(f"参数表缺少列：{missing}")

    # 按 26 段顺序筛选 & 重排（若表里更多区间也不影响，这里只跑白名单内）
    params['station_pair'] = params['station_pair'].astype(str).str.strip()
    rows = []
    missing = []
    for sp in LINE5_SECTIONS:
        row = params[params['station_pair'] == sp]
        if row.empty:
            missing.append(sp)
        else:
            rows.append(row.iloc[0])

    if missing:
        print("⚠️ 下列区间在参数表中缺失，将跳过：")
        for sp in missing:
            print("  -", sp)

    if not rows:
        raise RuntimeError("参数表中没有可运行的 5 号线区间。")

    print(f"准备优化 {len(rows)} 个区间…")
    for i, r in enumerate(rows, 1):
        sp = r['station_pair']
        try:
            print(f"\n=== [{i:02d}/{len(rows):02d}] {sp} ===")
            run_one_segment(r)
        except Exception as e:
            print(f"[错误] {sp}: {e}")

if __name__ == "__main__":
    main()
