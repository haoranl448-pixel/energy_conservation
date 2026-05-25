# -*- coding: utf-8 -*-
"""
train_transformer.py (按车次/段严格切分版)

修改记录：
1. [核心] 数据集划分改为按 'segment' (车次/段) 切分。
   - 前 80% 的段 -> 训练集
   - 后 20% 的段 -> 测试集
2. [核心] Scaler (标准化) 仅在训练集上 fit，防止测试集信息泄露。
3. 保持了 Y-Scaling 和 特征纯净化。
"""

import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 绘图配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ================= 配置参数 =================
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

SEQ_LEN = 30       # 序列长度：AI每次往回看30个时间点（比如1.5秒）
BATCH_SIZE = 64    # 批大小：AI一次打包学习64段数据
EPOCHS = 100       # 轮数：总共把所有题目反复学100遍
LR = 0.0005      # # 学习率：AI 修正自己错误的幅度（太大了学不会，太小了学得慢）  

# ================= 1. 模型定义 (保持不变) =================

class PositionalEncoding(nn.Module):#定义了一个叫 PositionalEncoding 的类。
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        #这个类通过数学公式（sin/cos），给数据加上位置信息，告诉模型：“这是刚才发生的，那是现在发生的”。
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EnergyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(EnergyTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        last_step_output = output[:, -1, :] 
        return self.decoder(last_step_output)

# ================= 2. 数据工具 =================

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    if len(X) <= seq_length: return np.array(xs), np.array(ys)
    for i in range(len(X) - seq_length):
        xs.append(X[i : i+seq_length])
        ys.append(y[i+seq_length-1])
    return np.array(xs), np.array(ys)

def read_multi_day_results(data_dir, station_pair):
    pattern = os.path.join(data_dir, f"results_{station_pair}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files: return None
    dfs = []# 读取多个Excel文件并合并为一个DataFrame 
    for fp in files:
        try: dfs.append(pd.read_excel(fp))
        except: pass
    if not dfs: return None
    return pd.concat(dfs, ignore_index=True)#ignore_index=True 参数确保索引重新排列，避免重复。

def ensure_cols(df, cols, fill=0.0):
    for c in cols:
        if c not in df.columns: df[c] = fill#确保所有需要的列都存在，缺失的列用0填充。
    return df

# ================= 3. 训练主流程 (重点修改) =================

def train_section_transformer(station_pair):
    print(f"\n🚀 [Transformer] 开始训练区间: {station_pair}")
    output_folder = os.path.join("nn_results_transformer", station_pair)
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. 读取数据
    df = read_multi_day_results("data_processed", station_pair)
    if df is None or df.empty: return

    need_cols = ['时刻', '速度(m/s)', '加速度(m/s²)', 'curvature', 'gradient', '重量', 'energy', 'segment']
    df = ensure_cols(df, need_cols)#确保所有需要的列都存在，缺失的列用0填充。
    
    # 2. 提取原始数据 (不进行 Shuffle)
    raw_X = pd.DataFrame({
        'velocity': df['速度(m/s)'],
        'acceleration': df['加速度(m/s²)'],
        'curvature': df['curvature'],
        'gradient': df['gradient'],
        'mass': df['重量']
    }).apply(pd.to_numeric, errors='coerce').fillna(0).values#转换为数值型，缺失值填0。
                            #errors='coerce'参数会将无法转换的值变为NaN，然后fillna(0)将其填充为0。
    
    raw_y = (pd.to_numeric(df['energy'], errors='coerce').fillna(0).values / 3.6e6) * 1000 
    segments = df['segment'].values# 车次/段 ID 列

    # 3. 按车次/段 (Segment) 切分训练集和测试集
    unique_segs = np.unique(segments) # 获取所有车次ID，它们本身就是按时间顺序排列的
    n_segs = len(unique_segs)# 车次总数
    
    # 按照 80% / 20% 切分车次
    split_point = int(n_segs * 0.8)
    if split_point == 0: split_point = 1 # 保护：如果数据太少，至少分一段给训练
    
    train_segs = unique_segs[:split_point]# 训练车次
    test_segs = unique_segs[split_point:]
    
    print(f"   总车次(段): {n_segs} | 训练车次: {len(train_segs)} | 测试车次: {len(test_segs)}")

    # 4. 构建掩码来分离数据
    # isin 返回布尔数组，标记每行数据属于训练还是测试
    train_mask = np.isin(segments, train_segs)# 训练集掩码
    test_mask = np.isin(segments, test_segs)
    
    X_train_raw = raw_X[train_mask]# 训练集特征
    y_train_raw = raw_y[train_mask]
    X_test_raw = raw_X[test_mask]
    y_test_raw = raw_y[test_mask]

    # 5. 标准化 (Scaler 只能在训练集上 fit!)
    scaler_x = StandardScaler()# 特征标准化器
    scaler_y = StandardScaler()
    
    # Fit & Transform Train
    X_train_scaled = scaler_x.fit_transform(X_train_raw)# 仅在训练集上 fit
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()# 仅在训练集上 fit
    
    # Transform Test (使用训练集的参数)
    if len(X_test_raw) > 0:
        X_test_scaled = scaler_x.transform(X_test_raw)# 使用训练集的参数
        y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    else:
        print("❌ 测试集为空，无法继续。")
        return

    # 6. 生成序列 (分别处理训练集和测试集的每个段)
    def process_segments(segs_list, x_scaled, y_scaled, seg_original_ids):
        seq_list, label_list = [], []
        # 注意：这里的 x_scaled 是整个训练集或测试集的大数组
        # 我们需要根据原始的 segment ID 再把它们拆开，防止跨车次生成序列
        
        # 为了效率，我们直接遍历 segs_list，去原始 segments 数组里找索引不太快
        # 我们利用 subset 的相对索引
        
        # 更简单的方法：
        # 我们已经把数据分开了，现在需要在这个分好的数据内部，再次按段切分
        # 这里为了简便，我们重新根据 mask 切分的子集再做一次 segment ID 的提取
        
        # 重新提取子集的 segment ID
        sub_segs = seg_original_ids
        
        for seg in segs_list:
            # 在当前子集中找到该段的索引
            idx = np.where(sub_segs == seg)[0]
            if len(idx) <= SEQ_LEN: continue
            
            # 切片
            seg_x = x_scaled[idx]
            seg_y = y_scaled[idx]
            
            sx, sy = create_sequences(seg_x, seg_y, SEQ_LEN)
            if len(sx) > 0:
                seq_list.append(sx)
                label_list.append(sy)
                
        if not seq_list: return None, None
        return np.concatenate(seq_list, axis=0), np.concatenate(label_list, axis=0)# 把所有段的序列拼接起来
                                                    #concatenate：沿指定轴连接数组序列。
                                                    #axis=0 表示按行连接，形成更长的样本集。

    # 生成训练序列
    X_train_seq, y_train_seq = process_segments(train_segs, X_train_scaled, y_train_scaled, segments[train_mask])
    # 生成测试序列
    X_test_seq, y_test_seq = process_segments(test_segs, X_test_scaled, y_test_scaled, segments[test_mask])
    
    if X_train_seq is None or X_test_seq is None:
        print("❌ 序列生成失败（可能某一段数据太短）。")
        return

    print(f"   训练样本: {len(X_train_seq)} | 测试样本: {len(X_test_seq)}")

    # 7. 转 Tensor & Loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 检测 GPU 可用性
    
    train_data = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float().unsqueeze(1))# 转为 TensorDataset
                                                                                    #unsqueeze(1) 是为了把 y 从一维变二维，符合损失函数要求。

    test_data = TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(y_test_seq).float().unsqueeze(1))
    
    
    # 这里把数据打包，每次拿出 64 个样本
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)# 训练时打乱数据顺序
                                       #shuffle=True 参数会在每个 epoch 开始时打乱数据，有助于提升模型泛化能力。
    # 测试时不需要打乱
    
    # 8. 模型初始化
    model = EnergyTransformer(input_dim=X_train_seq.shape[2]).to(device)# 输入维度由特征数量决定
                                                                        #shape[2] 是特征维度
    optimizer = optim.Adam(model.parameters(), lr=LR)# Adam 优化器
    criterion = nn.SmoothL1Loss()#  平滑 L1 损失函数
    
    # 9. 训练循环
    best_loss = float('inf')# 初始化最佳损失为无穷大
    patience_cnt = 0# 早停计数器
    train_losses, val_losses = [], []# 记录损失曲线

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []# 记录每个批次的损失


        for bx, by in train_loader:
            # 此时此刻，bx 的形状就是 [64, 30, 5]
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()# 清空梯度
            pred = model(bx)# 前向传播
            loss = criterion(pred, by)# 计算损失
            loss.backward()# 反向传播
            optimizer.step()
            batch_losses.append(loss.item())


            #64：由 BATCH_SIZE 决定。
            #30：由 SEQ_LEN 和 create_sequences 决定。
            #5：由 raw_X 里的列数决定。
        
        avg_train_loss = np.mean(batch_losses)# 计算平均训练损失
        train_losses.append(avg_train_loss)
        
        model.eval()# 评估模式
        with torch.no_grad():
            vx = test_data.tensors[0].to(device)# 测试集特征
            vy = test_data.tensors[1].to(device)
            vpred = model(vx)
            vloss = criterion(vpred, vy).item()
        val_losses.append(vloss)
        
        if (epoch+1) % 10 == 0:# 每10轮打印一次
            print(f"   Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.5f} | Val: {vloss:.5f}")
            
        if vloss < best_loss:# 提前保存最佳模型
            best_loss = vloss
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(output_folder, "best_transformer.pth"))
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                print(f"   早停触发 (Epoch {epoch+1})")
                break
                
    # 10. 评估与绘图 (累积能耗版)
# 10. 评估与绘图 (累积能耗版)
    model.load_state_dict(torch.load(os.path.join(output_folder, "best_transformer.pth"), map_location=device))
    model.eval()
    
    with torch.no_grad():
        # 取测试集中的 一整趟车 (第一个测试车次) 来画图
        first_test_seg = test_segs[0]# 获取第一个测试车次 ID
        idx = np.where(segments == first_test_seg)[0]# 找到该车次的所有索引
        
        seg_x_raw = raw_X[idx]
        seg_y_raw = raw_y[idx]
        
        seg_x_scaled = scaler_x.transform(seg_x_raw)
        
        # 生成序列
        sx, sy = create_sequences(seg_x_scaled, scaler_y.transform(seg_y_raw.reshape(-1,1)).flatten(), SEQ_LEN)
        # 这里把原本二维的 Excel 表格，切成了一个个“厚度”为 30 的数据块。
        
        if len(sx) == 0:
            print("⚠️ 测试段太短，无法绘图")
            return

        # 预测
        input_tensor = torch.from_numpy(sx).float().to(device)
        pred_scaled = model(input_tensor).cpu().numpy()
        
        # --- 核心修复点开始 ---
        # 反归一化预测值 # 反归一化：把预测出的 -1~1 的数字变回真实的能耗值
        pred_y = scaler_y.inverse_transform(pred_scaled).flatten()
        pred_y = np.maximum(pred_y, 0) # 物理约束# 物理约束：能耗不能是负的

        # 获取真实值，并强制裁剪长度以匹配预测值
        # 原因：create_sequences 生成的序列数可能比直接切片少 1 个
        true_y = seg_y_raw[SEQ_LEN-1:] 
        true_y = true_y[:len(pred_y)] # <--- 关键修复：强行对齐长度
        # --- 核心修复点结束 ---

        # 累积
        cumsum_true = np.cumsum(true_y)
        cumsum_pred = np.cumsum(pred_y)
        
        # 计算这段的 R2
        r2 = r2_score(true_y, pred_y)
        
    print(f"   ★ 测试车次(ID:{first_test_seg}) R2: {r2:.4f}")
    
    # 绘图：累积能耗
    plt.figure(figsize=(10, 5))
    plt.plot(cumsum_true, label='真实累积能耗', color='#1f77b4', linewidth=2)
    plt.plot(cumsum_pred, label='Transformer预测', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.fill_between(range(len(cumsum_true)), cumsum_true, cumsum_pred, color='gray', alpha=0.2)
    plt.title(f"单次行程能耗累积预测 ({station_pair})\nR2={r2:.3f} (测试集首趟车次)", fontsize=12)
    plt.xlabel("时间步 (0.05s/step)")
    plt.ylabel("总能耗 (Wh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "cumulative_energy.png"))
    plt.close()
    
    # 绘图：瞬时
    plt.figure(figsize=(10, 5))
    plt.plot(true_y, label='真实瞬时', color='#1f77b4', alpha=0.6)
    plt.plot(pred_y, label='预测瞬时', color='#ff7f0e', alpha=0.6)
    plt.title(f"瞬时功率波动 ({station_pair})", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "instant_energy.png"))
    plt.close()

if __name__ == "__main__":
    for sp in LINE5_SECTIONS:
        try:
            train_section_transformer(sp)
        except Exception as e:
            print(f"❌ {sp} 训练出错: {e}")