# -*- coding: utf-8 -*-
import os
import glob
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ================= 1. 路径与配置 =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root = str(PROJECT_ROOT)
sys.path.append(project_root)

from src.physics.train_simu import TrainTheoreticalEnergyModel


def get_data_dir(default_dir):
    """Use ENERGY_DATA_DIR when the main pipeline points this run at a test dataset."""

    raw = os.environ.get("ENERGY_DATA_DIR")
    if not raw:
        return default_dir
    return raw if os.path.isabs(raw) else os.path.join(project_root, raw)


DATA_DIR = get_data_dir(os.path.join(project_root, "data", "data_processed"))

# --- 🚀 提速参数配置 ---
FEATURE_COLS = ['v', 'a', 'e_phy', 'grad', 'mass', 'curv']
SEQ_LEN = 30
BATCH_SIZE = 512  # 🚀 提速点 1：大幅增加 Batch Size
MAX_EPOCHS = 150  # 最大轮数
PATIENCE = 5      # 🚀 提速点 2：连续 5 轮不下降则早停
LR = 0.001        # 配合大 Batch Size，略微调高学习率

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 当前训练设备: {DEVICE}")

# ================= 2. 模型定义 =================
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
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ResidualTransformerV2(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        out = self.transformer(src)
        return self.decoder(out[:, -1, :])

# ================= 3. 数据处理 =================
def create_sequences(X, y, seq_length):
    if len(X) <= seq_length: return np.array([]), np.array([])
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i : i+seq_length])
        ys.append(y[i+seq_length-1])
    return np.array(xs), np.array(ys)



def read_data(station_pair):
    pattern_new = os.path.join(DATA_DIR, f"results_{station_pair}*.xlsx")
    pattern_old = os.path.join(project_root, "data_processed", f"results_{station_pair}*.xlsx")
    files = sorted(glob.glob(pattern_new))
    if not files: files = sorted(glob.glob(pattern_old))
    if not files:
        print(f"   ⚠️ 未找到数据: {station_pair}")
        return None
    return pd.concat([pd.read_excel(f) for f in files], ignore_index=True)




def train_station(sp):
    out_dir = os.path.join(project_root, "output", "models", "nn_results_residual_v2", sp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🚀 [残差6特征] 开始训练: {sp}")



    df = read_data(sp)
    if df is None:
        return
    # 强制转换并清洗
    # 2. 物理仿真
    print("   ...正在计算物理基准 (Simulating)...")



    cols_fix = ['速度(m/s)', '加速度(m/s²)', 'energy', 'gradient', '重量', 'curvature']
    for c in cols_fix: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cols_fix)

    # 物理仿真获取基准
    sim_model = TrainTheoreticalEnergyModel()
    df['energy_phy'] = 0.0



    # for seg_id, group in df.groupby('segment'):
    #     e_phy_wh = sim_model.run_batch_simulation(group['时刻'].values, group['速度(m/s)'].values, group['重量'].iloc[0])
    #     df.loc[group.index, 'energy_phy'] = e_phy_wh

    # df['energy_res'] = (df['energy'] / 3.6e6 * 1000) - df['energy_phy']

    # raw_X = df[['速度(m/s)', '加速度(m/s²)', 'energy_phy', 'gradient', '重量', 'curvature']].values
    # raw_y = df['energy_res'].values
    # segments = df['segment'].values


    unique_segs = df['segment'].unique()
    for seg_id in unique_segs:
        mask = (df['segment'] == seg_id)
        trip_data = df[mask]
        t_seq = trip_data['时刻'].values
        v_seq = trip_data['速度(m/s)'].values
        mass_val = trip_data['重量'].iloc[0]
        e_seq_wh = sim_model.run_batch_simulation(t_seq, v_seq, mass_val)
        L = min(len(trip_data), len(e_seq_wh))
        df.loc[trip_data.index[:L], 'energy_phy'] = e_seq_wh[:L]

    # 3. 计算残差
    df['energy_real'] = (pd.to_numeric(df['energy'], errors='coerce').fillna(0) / 3.6e6) * 1000
    df['energy_res'] = df['energy_real'] - df['energy_phy']

    # 4. 特征
    v = df['速度(m/s)'].values
    a = df['加速度(m/s²)'].values
    grad = df['gradient'].values
    mass = df['重量'].values
    curvature = df['curvature'].values
    raw_X = pd.DataFrame({'v':v, 'a':a, 'e_phy':df['energy_phy'], 'grad':grad, 'mass':mass,'curvature':curvature}).fillna(0).values
    raw_y = df['energy_res'].fillna(0).values
    segments = df['segment'].values







    # 划分与标准化
    unique_segs = np.unique(segments)
    split = int(len(unique_segs) * 0.8)
    if split == 0: split = 1
    train_segs = unique_segs[:split]
    test_segs = unique_segs[split:]

    train_mask = np.isin(segments, train_segs)
    test_mask = np.isin(segments, test_segs)

    # 6. 标准化 & 保存
    # scaler_x, scaler_y = StandardScaler(), StandardScaler()



    # scaler_x.fit(raw_X[np.isin(segments, train_segs)])
    # scaler_y.fit(raw_y[np.isin(segments, train_segs)].reshape(-1, 1))

    # X_train_seq, y_train_seq = [], []
    # for seg in train_segs:
    #     mask = (segments == seg)
    #     sx, sy = create_sequences(scaler_x.transform(raw_X[mask]),
    #                               scaler_y.transform(raw_y[mask].reshape(-1,1)).flatten(), SEQ_LEN)
    #     if len(xs := sx) > 0: X_train_seq.append(sx); y_train_seq.append(sy)

    # X_train_seq = np.concatenate(X_train_seq)
    # y_train_seq = np.concatenate(y_train_seq)


    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(raw_X[train_mask])
    y_train = scaler_y.fit_transform(raw_y[train_mask].reshape(-1,1)).flatten()

    with open(os.path.join(out_dir, "scaler_x.pkl"), "wb") as f: pickle.dump(scaler_x, f)
    with open(os.path.join(out_dir, "scaler_y.pkl"), "wb") as f: pickle.dump(scaler_y, f)

    if np.sum(test_mask) == 0: return
    X_test = scaler_x.transform(raw_X[test_mask])
    y_test = scaler_y.transform(raw_y[test_mask].reshape(-1,1)).flatten()

    # 7. 序列
    X_train_seq, y_train_seq = [], []
    for seg in train_segs:
        idx = np.where(segments == seg)[0]
        if len(idx) <= SEQ_LEN: continue
        sx, sy = create_sequences(scaler_x.transform(raw_X[idx]), scaler_y.transform(raw_y[idx].reshape(-1,1)).flatten(), SEQ_LEN)
        if len(sx)>0: X_train_seq.append(sx); y_train_seq.append(sy)

    X_test_seq, y_test_seq = [], []
    for seg in test_segs:
        idx = np.where(segments == seg)[0]
        if len(idx) <= SEQ_LEN: continue
        sx, sy = create_sequences(scaler_x.transform(raw_X[idx]), scaler_y.transform(raw_y[idx].reshape(-1,1)).flatten(), SEQ_LEN)
        if len(sx)>0: X_test_seq.append(sx); y_test_seq.append(sy)

    if not X_train_seq: return
    X_train_seq = np.concatenate(X_train_seq)
    y_train_seq = np.concatenate(y_train_seq)
    X_test_seq = np.concatenate(X_test_seq)
    y_test_seq = np.concatenate(y_test_seq)





    # # 🚀 提速点 3：设置 pin_memory 加快数据搬运
    # train_loader = DataLoader(
    #     TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq).unsqueeze(1)),
    #     batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    # )

    # model = ResidualTransformerV2(input_dim=6).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # criterion = nn.SmoothL1Loss()

    # 8. 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq).unsqueeze(1)), batch_size=BATCH_SIZE, shuffle=True)
    model = ResidualTransformerV2(input_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

        # ⬇️⬇️⬇️ 这里加回了 print 语句 ⬇️⬇️⬇️
    print(f"   ⏳ 开始训练残差模型 (样本数: {len(X_train_seq)})...")
    best_loss = float('inf')


    # --- 🚀 核心早停逻辑 ---


    print(f"▶️ 开始训练: {sp} (样本数: {len(X_train_seq)})")

    for ep in range(MAX_EPOCHS):
        model.train()
        epoch_losses = []
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)

        # 打印进度 (每 10 轮一次)
        if (ep + 1) % 10 == 0:
            print(f"      Epoch {ep+1:3d} | Loss: {avg_loss:.6f}")

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{out_dir}/best_res_model.pth")

    print(f"   ✅ 训练结束 (Best Loss: {best_loss:.5f})")


    # ================= 9. 绘图 (三纵轴版: v, s, E) =================
    model.load_state_dict(torch.load(f"{out_dir}/best_res_model.pth", map_location=device))
    model.eval()

    # 提取当前区间的验证数据
    first_test_seg = test_segs[0]
    idx = np.where(segments == first_test_seg)[0]
    trip_data = df.iloc[idx].copy()

# 准备数据
    t_seq = np.arange(len(trip_data)) * 0.05  # 横轴时间(s)，假设步长0.05s
    v_seq = trip_data['速度(m/s)'].values * 3.6  # 速度转 km/h
    s_seq = trip_data['累计位移(m)'].values  # 位移(m)

# 能耗数据处理 (仅保留真实值和AI修正值)
    phy_val = trip_data['energy_phy'].values[SEQ_LEN-1:]
    real_val = trip_data['energy_real'].values[SEQ_LEN-1:]

    seg_x = scaler_x.transform(raw_X[idx])
    sx, _ = create_sequences(seg_x, np.zeros(len(seg_x)), SEQ_LEN)

    with torch.no_grad():
        res_pred = model(torch.FloatTensor(sx).to(device)).cpu().numpy()
    res_pred = scaler_y.inverse_transform(res_pred).flatten()

    # 对齐长度
    L_min = min(len(phy_val), len(res_pred), len(real_val))
    t_plot = t_seq[SEQ_LEN-1:][:L_min]
    v_plot = v_seq[SEQ_LEN-1:][:L_min]
    s_plot = s_seq[SEQ_LEN-1:][:L_min]
    e_real_cum = np.cumsum(real_val[:L_min])
    e_fusion_cum = np.cumsum(np.maximum(phy_val[:L_min] + res_pred[:L_min], 0))

    # --- 开始绘制三纵轴图 ---
    fig, ax1 = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(right=0.85) # 给右侧留出空间放第三个轴

    # 1. 左轴 (ax1): 速度 v
    color_v = 'tab:blue'
    ax1.set_xlabel('时间 Time (s)', fontsize=10)
    ax1.set_ylabel('速度 Velocity (km/h)', color=color_v, fontsize=10)
    lns1 = ax1.plot(t_plot, v_plot, color=color_v, linewidth=1.5, label='速度 (v)')
    ax1.tick_params(axis='y', labelcolor=color_v)
    ax1.grid(True, alpha=0.2)

    # 2. 右轴1 (ax2): 能耗 E
    ax2 = ax1.twinx()
    color_e = 'tab:red'
    ax2.set_ylabel('累积能耗 Energy (Wh)', color=color_e, fontsize=10)
    lns2_1 = ax2.plot(t_plot, e_real_cum, color='black', linewidth=2, label='真实能耗 (E_real)')
    lns2_2 = ax2.plot(t_plot, e_fusion_cum, color=color_e, linestyle='--', linewidth=2, label='AI修正能耗 (E_fusion)')
    ax2.tick_params(axis='y', labelcolor=color_e)

    # 3. 右轴2 (ax3): 位移 s
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60)) # 将第三个轴向右偏移60个单位
    color_s = 'tab:green'
    ax3.set_ylabel('累积位移 Distance (m)', color=color_s, fontsize=10)
    lns3 = ax3.plot(t_plot, s_plot, color=color_s, linewidth=1.5, label='累计位移 (s)')
    ax3.tick_params(axis='y', labelcolor=color_s)

    # 合并图例
    lns = lns1 + lns2_1 + lns2_2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=9)

    plt.title(f"区间运行综合对标图 ({sp})\n[速度 v | 位移 s | 融合能耗 E]", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/triple_axis_fusion.png", dpi=300)
    plt.close()
    print(f"   📊 三纵轴对标图已保存: {out_dir}/triple_axis_fusion.png")





    with open(f"{out_dir}/scaler_x.pkl", "wb") as f: pickle.dump(scaler_x, f)
    with open(f"{out_dir}/scaler_y.pkl", "wb") as f: pickle.dump(scaler_y, f)

LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥",
]


def main():
    print(f"📁 残差训练数据目录: {DATA_DIR}")
    for sp in LINE5_SECTIONS:
        try: train_station(sp)
        except Exception as e: print(f"❌ {sp} 失败: {e}")


if __name__ == "__main__":
    main()
