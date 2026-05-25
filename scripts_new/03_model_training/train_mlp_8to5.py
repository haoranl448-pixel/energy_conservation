# -*- coding: utf-8 -*-
"""
train_mlp_8to5.py (MLP - 无历史特征 Baseline 版)
修改记录：
1. 移除了 'prev_velocity', 'prev_acceleration', 'time' 特征。
2. 输出目录改为 'output/models/nn_results_mlp_8to5'，避免与旧模型冲突。
3. 修复了保存/加载文件名不一致导致的崩溃问题。
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 中文配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 5 号线 26 段固定清单
# -------------------------------
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥",
]

# -------------------------------
# 定义神经网络模型
# -------------------------------
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = torch.nn.functional.softplus(self.fc3(x))
        return x

def ensure_cols(df, cols, fill_value=0.0):
    for c in cols:
        if c not in df.columns: df[c] = fill_value
    return df

def read_multi_day_results(data_dir, station_pair):
    pattern = os.path.join(data_dir, f"results_{station_pair}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files: return None, []
    dfs = []
    used_files = []
    for fp in files:
        try:
            df = pd.read_excel(fp)
            dfs.append(df)
            used_files.append(fp)
        except Exception as e:
            print(f"❌ 读取失败（跳过）{fp}: {e}")
    if not dfs: return None, used_files
    return pd.concat(dfs, ignore_index=True), used_files

def train_one_section(station_pair, data_dir="data/data_processed", output_root="output/models/nn_results_mlp_8to5",
                      test_size=0.2, random_state=42):
    """
    MLP 对比实验训练：不包含历史特征
    """
    # 输出目录改为 output/models/nn_results_mlp_8to5，物理隔离，防止冲突
    output_folder = os.path.join(output_root, station_pair)
    os.makedirs(output_folder, exist_ok=True)

    # 注意：这里的数据路径可能需要根据你实际整理后的结构调整
    # 如果 data_process.py 还在根目录运行且生成在 data_processed，则保持默认
    # 如果已经整理到 data/processed，请在调用时传入正确路径
    # 这里为了兼容，假设数据在 'data_processed' (根目录) 或 'data/processed'
    if not os.path.exists(data_dir):
        # 尝试回退到旧路径
        if os.path.exists("data_processed"):
            data_dir = "data_processed"
        else:
             print(f"❌ 数据目录不存在: {data_dir}")
             return

    df_result, used_files = read_multi_day_results(data_dir, station_pair)
    if df_result is None or df_result.empty:
        print(f"❌ [{station_pair}] 无数据，跳过。")
        return

    need_cols = ['时刻', '速度(m/s)', '加速度(m/s²)', 'curvature', 'gradient',
                 '重量', 'energy', 'cumulative_energy_kWh', 'segment']
    df_result = ensure_cols(df_result, need_cols, fill_value=0.0)
    df_result['unique_segment'] = df_result['segment']

    # -------------------------------
    # 特征矩阵 (移除了 time 和 prev 特征)
    # -------------------------------
    velocity = df_result['速度(m/s)'].values
    acceleration = df_result['加速度(m/s²)'].values
    curvature_series = df_result['curvature'].values
    gradient_series = df_result['gradient'].values
    mass_series = df_result['重量'].values

    # 这里的 X_df 只包含 5 个物理特征
    X_df = pd.DataFrame({
        'velocity': velocity,
        'acceleration': acceleration,
        'curvature': curvature_series,
        'gradient': gradient_series,
        'mass': mass_series,
    }).apply(pd.to_numeric, errors='coerce')

    y = ((pd.to_numeric(df_result['energy'], errors='coerce').to_numpy().astype(float) / 3.6e6) * 1000)  # Wh

    # 清洗 NaN/Inf
    is_finite_X = np.isfinite(X_df.to_numpy()).all(axis=1)
    is_finite_y = np.isfinite(y)
    keep_mask = is_finite_X & is_finite_y
    X_df = X_df.loc[keep_mask].reset_index(drop=True)
    y = y[keep_mask].reshape(-1)

    if len(X_df) < 10: return

    # 剔除常数列
    stds = X_df.std(axis=0, ddof=0)
    keep_cols = stds[stds > 0].index.tolist()
    X_df = X_df[keep_cols]
    X = X_df.to_numpy(dtype=float)

    # -------------------------------
    # 划分 + 标准化
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 保存 Scaler
    with open(os.path.join(output_folder, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # -------------------------------
    # 转 Tensor
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # -------------------------------
    # 模型初始化
    # -------------------------------
    input_dim = X_train_scaled.shape[1] # 这里会自动识别为 5
    model = Net(input_dim).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------------
    # 训练循环
    # -------------------------------
    num_epochs = 300
    patience = 20
    best_val_loss = np.inf
    trigger_times = 0

    print(f"\n🚀 [MLP-Baseline] 开始训练: {station_pair}")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        train_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()

        if (epoch+1) % 20 == 0:
            print(f"   Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # 统一文件名
            torch.save(model.state_dict(), os.path.join(output_folder, "best_model.pth"))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"   早停 (Epoch {epoch+1})")
                break

    # -------------------------------
    # 评估与画图
    # -------------------------------
    # 加载刚保存的模型
    model.load_state_dict(torch.load(os.path.join(output_folder, "best_model.pth"), map_location=device))
    model.eval()

    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_pred_np = test_predictions.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()

    test_r2 = r2_score(y_test_np, test_pred_np)
    print(f"   ★ R2 Score: {test_r2:.4f}")

    # 保存指标
    pd.DataFrame({
        "metric": ["test_r2", "best_val_loss"],
        "value": [test_r2, best_val_loss]
    }).to_csv(os.path.join(output_folder, "metrics.csv"), index=False)

    # 画图 (只取前200个点对比)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_np[:200], label='真实值', alpha=0.7)
    plt.plot(test_pred_np[:200], label='MLP预测值', alpha=0.7)
    plt.title(f"MLP Baseline ({station_pair}) R2={test_r2:.3f}")
    plt.xlabel("样本序号")
    plt.ylabel("能耗 (Wh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pred_vs_true.png"))
    plt.close()
    
    # 保存模型结构
    torch.save(model.state_dict(), os.path.join(output_folder, "trained_model.pth"))
    print(f"   ✅ 结果已保存至: {output_folder}")

if __name__ == "__main__":
    for sp in LINE5_SECTIONS:
        try:
            train_one_section(sp)
        except Exception as e:
            print(f"❌ {sp} 错误: {e}")