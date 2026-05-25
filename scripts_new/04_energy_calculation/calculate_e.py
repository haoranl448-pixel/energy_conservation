# -*- coding: utf-8 -*-
"""
scripts/calculate_e.py (修正参数缺失版)
"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.interpolate import interp1d
import math, sys, warnings

warnings.filterwarnings("ignore")

# ================= 1. 路径与环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.physics.train_simu import TrainTheoreticalEnergyModel

REPORT_FILE = os.path.join(project_root, "output", "trip6_analysis_report", "Trip6_Saving_Report.csv")
PARAM_FILE = os.path.join(project_root, "data", "static", "section_params_trip6.csv")
DATA_DIR = os.path.join(project_root, "data", "data_processed")
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual_v2")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "trip6_historical_playback")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DT = 0.05
SEQ_LEN_RES = 30
EPS_T = 1e-6

# 🌟 2. 严格同步你提供的物理参数 (关键！)
SPECIAL_PARAMS = {
    "泗港-曹隘": { "A1": 0.55, "A_DEC": -0.5, "V_MID": 12.5, "A2": 0.8, "MASS": 217.04 },
    "梅堰-永茂路": { "A1": 0.55, "A_DEC": -0.5, "V_MID": 12.5, "A2": 0.8, "MASS": 215.74 }
}

# ================= 3. 模型定义 (对齐权重文件) =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

# class ResidualTransformer(nn.Module):
#     def __init__(self, input_dim=5):
#         super().__init__()
#         self.input_linear = nn.Linear(input_dim, 64)
#         self.pos_encoder = PositionalEncoding(64)
#         self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True), 2)
#         self.decoder = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
#     def forward(self, x):
#         return self.decoder(self.transformer(self.pos_encoder(self.input_linear(x)))[:, -1, :])



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


# ================= 4. 轨迹生成引擎 (同步 run_optimize_special 逻辑) =================

def solve_vmax_analytical(s_var, t_var, v_init, A_POS, A_NEG, vmax_cap):
    """同步 V12 解锁版逻辑"""
    if t_var <= 0 or s_var <= 0: return None, None
    ap, an = A_POS, abs(A_NEG)
    A = (1.0/ap + 1.0/an); B = -2.0 * (t_var + v_init/ap); C = 2.0 * (s_var + v_init**2 / (2.0*ap))
    delta = max(B*B - 4*A*C, 0.0)
    vmax = (-B - math.sqrt(delta)) / (2*A)
    # 放宽速度检查逻辑
    if vmax > vmax_cap + 50.0 or vmax < 0.1: 
        vmax = (-B + math.sqrt(delta)) / (2*A)
        if vmax > vmax_cap + 50.0 or vmax < 0.1: return None, None
    tc = t_var - (vmax - v_init)/ap - vmax/an
    return vmax, max(0.0, tc)

def build_full_trajectory(prefix, vmax, tc, A_POS, A_NEG, L_total):
    t0, s0, v0 = prefix['T'], prefix['L'], prefix['v_end']
    ap, an = A_POS, abs(A_NEG)
    if vmax >= v0: ta, a_s1 = (vmax-v0)/ap, ap
    else: ta, a_s1 = (v0-vmax)/an, -an
    t1 = np.arange(0, ta + EPS_T, DT); v1 = v0 + a_s1 * t1
    t2 = np.arange(DT, tc + EPS_T, DT) if tc > DT*0.5 else np.array([]); v2 = np.full_like(t2, vmax)
    td = vmax/an; t3 = np.arange(DT, td + EPS_T, DT); v3 = np.maximum(vmax - an*t3, 0.0)
    v_f = np.concatenate([prefix['v'], v1, v2, v3])
    t_f = np.arange(0, len(v_f)*DT, DT)[:len(v_f)]
    a_f = np.zeros_like(v_f); a_f[1:] = np.diff(v_f)/DT
    s_f = np.cumsum(v_f * DT)
    return t_f, v_f, a_f, s_f

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
        T,V,S = piece(t0, dur, v0, a); T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
        t0,v0,S_accum=T[-1],V[-1],S_all[-1][-1]
    t6 = (v0 - v_to)/abs(A_NEG); T,V,S = piece(t0, t6, v0, A_NEG); T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
    return {'v': np.concatenate(V_all), 'T': T_all[-1][-1], 'L': S_all[-1][-1], 'v_end': V_all[-1][-1]}

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
        T,V,S = piece(t0, dur, v0, a); T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
        t0,v0,S_accum=T[-1],V[-1],S_all[-1][-1]
    t3 = (v0 - v_to)/abs(A_NEG); T,V,S = piece(t0, t3, v0, A_NEG); T_all.append(T); V_all.append(V); S_all.append(S+S_accum)
    return {'v': np.concatenate(V_all), 'T': T_all[-1][-1], 'L': S_all[-1][-1], 'v_end': V_all[-1][-1]}

def generate_common_vt(target_t, target_l, p):
    acc, dec = p['A1'], abs(p['A_DEC'])
    k = 0.5 * (1/acc + 1/dec)
    roots = np.roots([k, -target_t, target_l])
    v_peak = min(r.real for r in roots if r.real > 0)
    t_arr = np.arange(0, target_t + DT, DT)
    v_list = []
    ta, td = v_peak/acc, v_peak/dec
    tc = max(0, target_t - ta - td)
    for tt in t_arr:
        if tt < ta: v_list.append(acc*tt)
        elif tt < ta + tc: v_list.append(v_peak)
        else: v_list.append(max(0, v_peak - dec*(tt-ta-tc)))
    v_arr = np.array(v_list); a_arr = np.zeros_like(v_arr); a_arr[1:] = np.diff(v_arr)/DT
    return t_arr, v_arr, a_arr, np.cumsum(v_arr * DT), v_peak



# ================= 5. 主程序 =================

def main():
    df_report = pd.read_csv(REPORT_FILE)
    df_report.columns = df_report.columns.str.strip()
    df_params = pd.read_csv(PARAM_FILE).set_index('station_pair')
    playback_results = []

    print(f"🚀 开始回放 Trip 6 历史能耗...")

    for _, row in df_report.iterrows():
        sp = row['区间'].strip()
        t_hist, e_hist_real = row['历史时间'], row['历史能耗(Wh)']
        if sp not in df_params.index: continue
        L_std, V_MAX = df_params.loc[sp, 'L'], df_params.loc[sp, 'V_UPPER']

        try:
            if sp in SPECIAL_PARAMS:
                p = SPECIAL_PARAMS[sp]
                A1, AN, mass = p['A1'], p['A_DEC'], p['MASS']
                prefix = fixed_prefix_sigang(A1, AN) if "泗港" in sp else fixed_prefix_meiyan(A1, AN)
                vmax, tc = solve_vmax_analytical(L_std - prefix['L'], t_hist - prefix['T'], prefix['v_end'], A1, AN, V_MAX)
                t_arr, v_arr, a_arr, s_arr = build_full_trajectory(prefix, vmax, tc, A1, AN, L_std)
                # 记录特殊站的峰值速度
                v_peak_actual = vmax 
            else:
                # 接收增加的第五个返回值 v_peak
                t_arr, v_arr, a_arr, s_arr, v_peak_actual = generate_common_vt(t_hist, L_std, df_params.loc[sp])

                mass = df_params.loc[sp, 'MASS']

            # 计算能耗
            engine = TrainTheoreticalEnergyModel()
            e_phy_steps = engine.run_batch_simulation(t_arr, v_arr, mass)
            res_dir = os.path.join(RES_MODEL_BASE, sp)
            model = ResidualTransformerV2(6)
            model.load_state_dict(torch.load(f"{res_dir}/best_res_model.pth", map_location='cpu'))
            model.eval()
            with open(f"{res_dir}/scaler_x.pkl", "rb") as f: sx = pickle.load(f)
            with open(f"{res_dir}/scaler_y.pkl", "rb") as f: sy = pickle.load(f)

            # 获取坡度并预测残差
            df_raw = pd.read_excel(os.path.join(DATA_DIR, f"results_{sp}.xlsx"))
            df_raw['累计位移(m)'] = pd.to_numeric(df_raw['累计位移(m)'], errors='coerce')
            df_raw = df_raw.dropna(subset=['累计位移(m)']).sort_values('累计位移(m)')
            f_grad = interp1d(df_raw['累计位移(m)'], df_raw['gradient'], kind='nearest', fill_value="extrapolate")
            f_curv = interp1d(df_raw['累计位移(m)'], df_raw['curvature'], kind='nearest', fill_value="extrapolate")
            raw_X = np.stack([v_arr, a_arr, e_phy_steps, f_grad(s_arr), np.full_like(v_arr, mass), f_curv(s_arr)], axis=1)
            
            scaled_X = sx.transform(raw_X)
            windows = [scaled_X[i-29:i+1] for i in range(29, len(scaled_X))]
            
            res_sum = np.sum(sy.inverse_transform(model(torch.tensor(np.array(windows), dtype=torch.float32)).detach().numpy())) if windows else 0
            
            e_total = np.sum(e_phy_steps) + res_sum
            print(f"✅ {sp:15} | 模型E:{e_total:8.1f} Wh | V_peak:{v_peak_actual:7.4f} m/s")
            print(f"   👉 [对标建议]: 请去优化文件夹下查看 time_energy_curve_{sp}.csv，找到 v_peak_opt 接近 {v_peak_actual:.2f} 的行进行对比")
            playback_results.append({'区间': sp, '历史T': t_hist, '实测E': e_hist_real, '模型E': round(e_total,2), 'V_peak': v_peak_actual})








        except Exception as e: print(f"❌ {sp} 出错: {e}")

    pd.DataFrame(playback_results).to_csv(os.path.join(OUTPUT_DIR, "playback_trip6_full.csv"), index=False)

if __name__ == "__main__":
    main()