# -*- coding: utf-8 -*-
"""
scripts/run_optimize_trip6_full_outputs.py
修正版：去除了海象运算符，确保路径兼容性
"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys, warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ================= 1. 环境与路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.physics.train_simu import TrainTheoreticalEnergyModel
from src.models.wrapper import EnergyPredictor

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

PARAM_FILE = os.path.join(project_root, "data", "static", "section_params_trip5.csv")
DATA_DIR = os.path.join(project_root, "data", "data_processed")
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual_v2")
OUTPUT_DIR = os.path.join(project_root, "output", "optimization", "trip5_v2_6feat")

DT = 0.05
SEQ_LEN_RES = 30

# ================= 2. 模型定义 (必须与训练脚本完全对齐) =================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): 
        return x + self.pe[:, :x.size(1), :]

class ResidualTransformerV2(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        # 🌟 关键：必须定义为 self.pos_encoder 才能匹配权重文件里的 "pos_encoder.pe"
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32), 
            nn.LeakyReLU(0.01), 
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x) # 调用子模块
        x = self.transformer(x)
        return self.decoder(x[:, -1, :])

# ================= 3. 核心计算函数 =================

def get_track_f(sp):
    file_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
    df = pd.read_excel(file_path)
    # for c in ['累计位移(m)', 'gradient', 'curvature']: 
    #     df[c] = pd.to_numeric(df[c], errors='coerce')
    # df = df.dropna(subset=['累计位移(m)', 'gradient', 'curvature']).sort_values('累计位移(m)')
    #return interp1d(df['累计位移(m)'], df['gradient'], kind='nearest', fill_value="extrapolate"), \
    #       interp1d(df['累计位移(m)'], df['curvature'], kind='nearest', fill_value="extrapolate")

  
    # === 核心修复点：强制将列转换为数字，无法转换的变 NaN 并剔除 ===
    df['累计位移(m)'] = pd.to_numeric(df['累计位移(m)'], errors='coerce')
    df['gradient'] = pd.to_numeric(df['gradient'], errors='coerce')
    df['curvature'] = pd.to_numeric(df['curvature'], errors='coerce')
    
    # 剔除掉包含 NaN 的行
    df = df.dropna(subset=['累计位移(m)', 'gradient', 'curvature'])
    
    s = df['累计位移(m)'].values
    if len(s) < 2: return None, None

    # 这里的 np.diff 现在不会再报 str 错误了
    mask = np.concatenate([[True], np.diff(s) > 1e-6])
    s_clean = s[mask]
    grad_clean = df['gradient'].values[mask]
    curv_clean = df['curvature'].values[mask]



    grad_f = interp1d(s_clean, grad_clean, kind='nearest',fill_value="extrapolate")
    curv_f = interp1d(s_clean, curv_clean, kind='nearest', fill_value="extrapolate")
    return grad_f, curv_f


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
        
    return {'t': t_grid, 'v': v_arr, 'a': a_arr, 's': s_arr, 'T': t_grid[-1], 't_total':T_total}



def run_optimization_for_station(sp, row):
    print(f"▶️ 正在优化: {sp}")
    res_dir = os.path.join(RES_MODEL_BASE, sp)
    station_out = os.path.join(OUTPUT_DIR, sp)
    os.makedirs(station_out, exist_ok=True)

    # 加载模型
    with open(os.path.join(res_dir, "scaler_x.pkl"), "rb") as f: sx = pickle.load(f)
    with open(os.path.join(res_dir, "scaler_y.pkl"), "rb") as f: sy = pickle.load(f)
    model = ResidualTransformerV2(6).to('cpu')
    model.load_state_dict(torch.load(os.path.join(res_dir, "best_res_model.pth"), map_location='cpu'))
    model.eval()
    try:
        ai_predictor = EnergyPredictor("residual", res_dir)
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return
    f_grad, f_curv = get_track_f(sp)
    engine = TrainTheoreticalEnergyModel()

    L, V_MAX, MASS = row['L'], row['V_UPPER'], row['MASS']
    TIME_LOWER = float(row['TIME_LOWER'])
    TIME_UPPER = float(row['TIME_UPPER'])

    p_veh = {'V_MID': float(row['V_MID']), 'A1': float(row['A1']),
             'A2': float(row['A2']), 'A_DEC': float(row['A_DEC'])}
    
    results = []
    

    v_range = np.arange(5.0, V_MAX + 0.1, 0.06)

    best_record = {'E_min': float('inf'), 'traj': None, 'e_cum_seq': None }
    for v_peak in v_range:
        traj = get_trajectory_by_distance(v_peak, L, p_veh)
        t_total = traj['t_total']
        t_actual = traj['T']

        if not (TIME_LOWER <= t_actual <= TIME_UPPER): continue


      
        grad_seq = f_grad(traj['s'])
        curv_seq = f_curv(traj['s'])
        mass_seq = np.full_like(traj['v'], MASS)

        
        e_phy_seq = engine.run_batch_simulation(traj['t'], traj['v'], MASS)

        # # 1. 轨迹生成
        # acc, dec = row['A1'], abs(row['A_DEC'])
        # t_acc, t_dec = v_peak/acc, v_peak/dec
        # s_acc, s_dec = 0.5*acc*t_acc**2, 0.5*dec*t_dec**2
        # if s_acc + s_dec > L: continue
        # t_cruise = (L - s_acc - s_dec) / v_peak
        # t_total = t_acc + t_cruise + t_dec
        # t_arr = np.arange(0, t_total + DT, DT)
        
        # v_list, a_list = [], []
        # for tt in t_arr:
        #     if tt < t_acc: v_list.append(acc*tt); a_list.append(acc)
        #     elif tt < t_acc + t_cruise: v_list.append(v_peak); a_list.append(0.0)
        #     else: v_list.append(max(0, v_peak - dec*(tt-t_acc-t_cruise))); a_list.append(-dec)
        
        # v_arr, a_arr = np.array(v_list), np.array(a_list)
        # s_arr = np.cumsum(v_arr * DT)

        # 2. 预测
        #e_phy = engine.run_batch_simulation(t_arr, v_arr, MASS)



        raw_X = np.stack([traj['v'], traj['a'], e_phy_seq, grad_seq, mass_seq, curv_seq], axis=1)
        e_res_seq = ai_predictor.predict(raw_X)
        
        
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
        

        # scaled_X = sx.transform(raw_X)
        



        # windows = [scaled_X[i-SEQ_LEN_RES+1:i+1] for i in range(SEQ_LEN_RES-1, len(scaled_X))]
        
        
        
        # res_sum = 0.0
        # if windows:
        #     with torch.no_grad(): 
        #         p = model(torch.tensor(np.array(windows), dtype=torch.float32)).numpy()
        #     res_sum = np.sum(sy.inverse_transform(p))
        
        # total_e = np.sum(e_phy_seq) + res_sum
        # results.append([t_total, v_peak, total_e])

        # if total_e < best_record["E_min"]:
        #     best_record.update({
        #         "E_min": total_e, "t_arr": t_arr, "v_arr": v_arr, "a_arr": a_arr, "s_arr": s_arr, 
        #         "e_cum": np.cumsum(e_phy + np.concatenate([np.zeros(29), sy.inverse_transform(p).flatten()]))
        #     })

    # C. 🌟 输出 6 个文件 (修正了 station_dir 的写法) 🌟
    df_pareto = pd.DataFrame(results, columns=['t_arrival', 'v_peak_opt', 'E_wh'])
    
    # 1. 帕累托曲线数据
    df_pareto.to_csv(os.path.join(station_out, f"time_energy_curve_{sp}.csv"), index=False)
    
    # 2. 帕累托曲线图
    plt.figure(); plt.plot(df_pareto['t_arrival'], df_pareto['E_wh'], 'o-', markersize=3, color='darkred')
    plt.title(f"{sp} Time-Energy Pareto"); plt.grid(True); plt.savefig(os.path.join(station_out, "time_energy_curve.png")); plt.close()


    best_df = pd.DataFrame([{
        'station_pair': sp,
        't_arrival_opt': best_record['t_arrival'],
        'v_peak_opt': best_record['v_peak'],
        'E_opt_wh': best_record['E_min'],
        'model_type': 'Physics+AI_Fusion'
    }])
    best_df.to_csv(os.path.join(OUTPUT_DIR, sp,"best_solution.csv"), index=False)

    traj = best_record['traj']
    vt_df = pd.DataFrame({
        'time': traj['t'],
        'velocity': traj['v'],
        'acceleration': traj['a'],
        'distance': traj['s'],
        'cumulative_energy': best_record['e_cum_seq']
    })
    vt_df.to_csv(os.path.join(OUTPUT_DIR,sp, "optimal_vt_curve.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(traj['t'], traj['v'], lw=2, color='#2ca02c')
    plt.title(f"{sp} 最优速度曲线 (T={best_record['t_arrival']:.1f}s)")
    plt.xlabel("时间 (s)")
    plt.ylabel("速度 (m/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, sp,"optimal_vt.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(traj['t'], best_record['e_cum_seq'], lw=2, color='#d62728')
    plt.fill_between(traj['t'], best_record['e_cum_seq'], color='#d62728', alpha=0.1)
    plt.title(f"{sp} 最优累积能耗 (Total={best_record['E_min']:.1f}Wh)")
    plt.xlabel("时间 (s)")
    plt.ylabel("累积能耗 (Wh)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,sp, "optimal_energy.png"), dpi=300)
    plt.close()

    print(f"   ✅ 全部6个文件生成完毕 -> {OUTPUT_DIR}")




    # # 3. 最佳方案简表
    # best_idx = df_pareto['E_wh'].idxmin()
    # pd.DataFrame([df_pareto.loc[best_idx]]).to_csv(os.path.join(station_out, "best_solution.csv"), index=False)

    # # 4. 最优轨迹数据
    # pd.DataFrame({
    #     'time': best_record['t_arr'], 'velocity': best_record['v_arr'], 
    #     'acceleration': best_record['a_arr'], 'distance': best_record['s_arr'],
    #     'cum_energy': best_record['e_cum']
    # }).to_csv(os.path.join(station_out, "optimal_vt_curve.csv"), index=False)

    # # 5. 最优速度剖面图
    # plt.figure(); plt.plot(best_record['t_arr'], best_record['v_arr']*3.6, color='blue', lw=2)
    # plt.title(f"Optimal VT Profile (Total E={best_record['E_min']:.1f}Wh)"); plt.grid(True)
    # plt.savefig(os.path.join(station_out, "optimal_vt.png")); plt.close()

    # # 6. 最优累积能耗图
    # plt.figure(); plt.plot(best_record['t_arr'], best_record['e_cum'], color='green', lw=2)
    # plt.fill_between(best_record['t_arr'], best_record['e_cum'], color='green', alpha=0.1)
    # plt.title(f"Optimal Energy Accumulation"); plt.grid(True)
    # plt.savefig(os.path.join(station_out, "optimal_energy.png")); plt.close()

# ================= 4. 主程序 =================
if __name__ == "__main__":
    LINE5_SECTIONS = [
        
        # "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
        # "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
        # "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
        # "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
        # "院士路-盎孟港", "盎孟港-三官堂",
          "三官堂-兴庄路", "兴庄路-兴海南路",
        "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
    ]
    df_all_params = pd.read_csv(PARAM_FILE).set_index('station_pair')
    for sp in LINE5_SECTIONS:
        if sp in df_all_params.index:
            try: run_optimization_for_station(sp, df_all_params.loc[sp])
            except Exception as e: print(f"❌ {sp} 失败: {e}")