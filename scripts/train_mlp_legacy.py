# -*- coding: utf-8 -*-
"""
train_line5.py
一次性训练宁波地铁5号线全线26个区间：
- 自动合并 data_processed 目录下匹配 results_{起-终}*.xlsx 的多天文件
- 每段各自训练并输出到 nn_results/{起-终}/
- 训练/特征/标准化/早停/评估/作图 与原逻辑保持一致（仅加最小必要的健壮性处理）
"""
from pathlib import Path
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 批处理保存图，不弹窗
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
# 5 号线 26 段固定清单（顺序与编号一致）
# -------------------------------
LINE5_SECTIONS = [
    "布政-张家潭",       # 0501
    "张家潭-同德路",     # 0502
    "同德路-石碶",       # 0503
    "石碶-雅渡",         # 0504
    "雅渡-庙堰",         # 0505
    "庙堰-钟公庙",       # 0506
    "钟公庙-鄞州区政府", # 0507
    "鄞州区政府-钱湖南路",# 0508
    "钱湖南路-南高教园区",# 0509
    "南高教园区-下应路", # 0510
    "下应路-大洋江",     # 0511
    "大洋江-泗港",       # 0512
    "泗港-曹隘",         # 0513
    "曹隘-柳隘",         # 0514
    "柳隘-海晏北路",     # 0515
    "海晏北路-民安东路", # 0516
    "民安东路-会展中心", # 0517
    "会展中心-院士路",   # 0518
    "院士路-盎孟港",     # 0519
    "盎孟港-三官堂",     # 0520
    "三官堂-兴庄路",     # 0521
    "兴庄路-兴海南路",   # 0522
    "兴海南路-梅堰",     # 0523
    "梅堰-永茂路",       # 0524
    "永茂路-镇海大道",   # 0525
    "镇海大道-骆驼桥",   # 0526
]


# -------------------------------
# 定义神经网络模型（保持不变）
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
    """确保 df 存在给定列，不在则补列（避免个别结果缺列报错），不改变原有逻辑。"""
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def read_multi_day_results(data_dir, station_pair):
    """
    合并多天文件：匹配 data_dir 下的 results_{起-终}*.xlsx
    若只有单个 results_{起-终}.xlsx 也能读取
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir= Path(project_root, "data","data_processed")
    pattern = os.path.join(data_dir, f"results_{station_pair}*.xlsx")# 生成 “目录 + 文件名模板” 的匹配规则，确保只读取目标区间的相关文件。
                                                                        #如 data_processed/results_布政-张家潭*.xlsx
                                                                        #os.path.join(data_dir, ...)：跨系统兼容的路径拼接
                                                                        #f"results_{station_pair}*.xlsx"：文件名模板，* 是通配符（匹配任意字符，包括空字符）
    files = sorted(glob.glob(pattern))# 根据模板查找目录下的所有匹配文件，并按文件名排序
                                        #glob.glob(pattern)：查找所有符合 pattern 规则的文件路径，返回列表；若无匹配文件，返回空列表；
                                        #glob(...)：是 Python 的一个标准库模块，提供了文件名模式匹配功能，类似于 Unix shell 中的通配符匹配。
    if not files:
        return None, []
    dfs = []#dfs：列表，用于存储每个成功读取的 Excel 文件对应的 DataFrame（后续拼接为总数据）
    used_files = []#used_files：列表，用于存储成功读取的文件路径（数据追溯用）
    for fp in files:
        try:
            df = pd.read_excel(fp)# 读取 Excel 文件为 DataFrame
            dfs.append(df)# 将读取的 DataFrame 添加到列表中
            used_files.append(fp)# 记录成功读取的文件路径
        except Exception as e:
            print(f"❌ 读取失败（跳过）{fp}: {e}")# 读取失败则打印错误并跳过该文件
    if not dfs:# 若没有成功读取的文件，返回 None
        return None, used_files
    df_all = pd.concat(dfs, ignore_index=True)# 将所有读取的 DataFrame 拼接为一个总的 DataFrame，忽略原有索引，重新生成连续索引
    return df_all, used_files#合并后的总 DataFrame（df_all）和有效文件列表（used_files）


def train_one_section(station_pair, data_dir="data/data_processed", output_root="output/models/nn_results_mlp1",
                      test_size=0.2, random_state=42):
    """
    对单个区间（可合并多天）训练并保存结果
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']# 设置中文字体，防止中文乱码

    # 输出目录
    output_folder = os.path.join(output_root, station_pair)# 输出目录：为当前区间创建独立文件夹，避免结果混淆
                                                                # os.path.join：用于拼接路径，确保不同操作系统下路径格式正确
                                                                # 例如：若 output_root="nn_results"，station_pair="布政-张家潭"，则 output_folder="nn_results/布政-张家潭"
    os.makedirs(output_folder, exist_ok=True)# 创建输出目录，exist_ok=True 表示若目录已存在则不报错

    # 读取多天数据
    df_result, used_files = read_multi_day_results(data_dir, station_pair)#并后的总 DataFrame（df_all）和有效文件列表（used_files）
    if df_result is None or df_result.empty:# 若无数据则跳过
        print(f"❌ [{station_pair}] 未找到任何结果文件：{os.path.join(data_dir, f'results_{station_pair}*.xlsx')}")
        return

    # 兼容性保护：确保必需列存在
    need_cols = ['时刻', '速度(m/s)', '加速度(m/s²)', 'curvature', 'gradient',
                 '重量', 'energy', 'cumulative_energy_kWh', 'segment']
    df_result = ensure_cols(df_result, need_cols, fill_value=0.0)# 确保 df_result 存在所有必需列，缺失则补零列

    # segment 直接作为 unique_segment（与你原代码一致）
    df_result['unique_segment'] = df_result['segment']#segment 列直接赋值给 unique_segment 列，保持与原代码逻辑一致
                                                        #segment “部分 / 片段”的意思，unique_segment 则表示 “唯一的片段标识”

    # -------------------------------
    # 特征矩阵与目标变量（保持写法）
    # -------------------------------
    time_arr = df_result['时刻'].values# 获取时间序列
    velocity = df_result['速度(m/s)'].values
    acceleration = df_result['加速度(m/s²)'].values
    curvature_series = df_result['curvature'].values
    gradient_series = df_result['gradient'].values
    mass_series = df_result['重量'].values

    # 历史特征（修复 fillna 弃用警告）
    df_result['prev_velocity'] = df_result['速度(m/s)'].shift(1).bfill()#.shift(1)：将速度列向下移动一行，表示前一时刻的速度
                                                        #.bfill()：用后一个有效值填充缺失值，确保首行有值
    df_result['prev_acceleration'] = df_result['加速度(m/s²)'].shift(1).bfill()# 同上，表示前一时刻的加速度

    X_df = pd.DataFrame({
        'time': time_arr,
        'velocity': velocity,
        'acceleration': acceleration,
        'prev_velocity': df_result['prev_velocity'].values,
        'prev_acceleration': df_result['prev_acceleration'].values,
        'curvature': curvature_series,
        'gradient': gradient_series,
        'mass': mass_series,
    })# 构建特征矩阵 DataFrame

    # ——最小必要的健壮性处理：数值化 + 清洗 + 常数列剔除——
    X_df = X_df.apply(pd.to_numeric, errors='coerce')#.apply(pd.to_numeric, errors='coerce')：尝试将所有列转换为数值类型，无法转换的值设为 NaN
                                                    #.to_numeric 是 Pandas 的一个函数，用于将数据转换为数值类型。
                                                    #errors='coerce' 参数表示遇到无法转换的值时，将其设置为 NaN，而不是抛出错误。
                                                    #.apply(...)：是 Pandas 中的一个方法，用于对 DataFrame 或 Series 的每一列或每一行应用一个函数。
    y = ((pd.to_numeric(df_result['energy'], errors='coerce').to_numpy().astype(float) / 3.6e6) * 1000)  # kWh→Wh
    # 目标变量同样数值化 + 清洗                                                             
    # #pd.to_numeric(..., errors='coerce')：尝试将能耗列转换为数值类型，无法转换的值设为 NaN
    # .to_numpy().astype(float)：转换为 NumPy 数组并确保为浮点类型
    # / 3.6e6 * 1000：将能耗从 kWh 转换为 Wh（保持与原代码一致）
    is_finite_X = np.isfinite(X_df.to_numpy()).all(axis=1)#.to_numpy()：将特征矩阵 DataFrame 转换为 NumPy 数组
                                                        #np.isfinite(...)：检查数组中的每个元素是否为有限数值（非 NaN、非正无穷、非负无穷）
                                                        #.all(axis=1)：对于每一行，检查所有列是否均为有限数值，返回布尔数组
                                                        #axis=1 表示「沿着行的方向计算」（横向遍历），对应的 axis=0 是「沿着列的方向计算」（纵向遍历）。
    is_finite_y = np.isfinite(y)# 检查目标变量数组中的每个元素是否为有限数值，返回布尔数组
    keep_mask = is_finite_X & is_finite_y# 结合特征矩阵和目标变量的有限性检查，生成最终的保留掩码（仅保留两者均为有限数值的样本）
    X_df = X_df.loc[keep_mask].reset_index(drop=True)#.loc[keep_mask]：只保留 keep_mask 中值为 True 对应的行；根据保留掩码筛选特征矩阵 DataFrame，仅保留有效样本
                                                    #.reset_index(drop=True)：重置索引，避免索引不连续，drop=True 表示不保留旧索引
    y = y[keep_mask].reshape(-1)# 根据保留掩码筛选目标变量数组，仅保留有效样本，并展平为一维数组
                                #.reshape(-1)：将数组展平为一维数组

    if len(X_df) < 10:# 兜底：样本数过少则跳过
        print(f"❌ [{station_pair}] 清洗后可用样本不足（{len(X_df)} 条），已跳过。")
        return

    stds = X_df.std(axis=0, ddof=0)#.std(axis=0, ddof=0)：计算每一列的标准差，axis=0 表示沿着列的方向计算
                                    #ddof=0 表示使用总体标准差公式（除以 N），而非样本标准差公式（除以 N-1）
    keep_cols = stds[stds > 0].index.tolist()# 识别标准差大于 0 的列（非常数列），并获取其列名列表
                                                #stds > 0：布尔索引，标识标准差大于 0 的列
                                                #.index.tolist()：获取这些列的列名，并转换为列表
                                                #.index：是 Pandas Series 的一个属性，表示该 Series 的索引标签。
                                                #.tolist()：是 Pandas Series 的一个方法，用于将索引标签转换为 Python 列表。
                                                #stds[stds > 0]只保留布尔数组中 True 位置对应的原数组元素；直接丢弃 False 位置对应的元素
    dropped_cols = [c for c in X_df.columns if c not in keep_cols]# 识别被剔除的常数列
                                                                #列表推导式，遍历原始列名，若不在 keep_cols 中则加入 dropped_cols 列表
    if dropped_cols:# 若有常数列被剔除则打印警告并保存列表
        print(f"[{station_pair}] 警告：检测到常数列已剔除：{dropped_cols}")
        pd.Series(dropped_cols, name="dropped_constant_features").to_csv(
            os.path.join(output_folder, "dropped_constant_features.csv"), index=False
        )

    X_df = X_df[keep_cols]# 仅保留非常数列，更新特征矩阵
    X = X_df.to_numpy(dtype=float)# 将最终特征矩阵转换为 NumPy 数组，确保为浮点类型

    # 记录特征矩阵
    X_df.to_excel(os.path.join(output_folder, "feature_matrix.xlsx"), index=False)# 保存特征矩阵为 Excel 文件
                                                                                        #os.path.join(...)：拼接输出文件路径 index=False：不保存行索引
    if used_files:# 保存合并的输入文件列表
        pd.Series(used_files, name="merged_files").to_csv(
            os.path.join(output_folder, "merged_input_files.csv"), index=False
        )#.Series(...)：将文件路径列表转换为 Pandas Series，方便保存为 CSV 文件
    print(f"[{station_pair}] 特征矩阵已保存：{os.path.join(output_folder, 'feature_matrix.xlsx')}")

    # -------------------------------
    # 划分 + 标准化
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )# 使用 sklearn 的 train_test_split 划分训练集和测试集
                                                        #test_size=test_size：测试集比例
    scaler = StandardScaler()# 实例化标准化器
    X_train_scaled = scaler.fit_transform(X_train)#.fit_transform(X_train)：先拟合训练集数据计算均值和标准差，再对训练集进行标准化转换
    X_test_scaled = scaler.transform(X_test)# 对测试集进行标准化转换，直接使用 scaler 已存储的统计量，使用训练集的均值和标准差
                                                    #本质区别：fit 是 “学习分布”，transform 是 “应用分布”；
    # 兜底：标准化后检查
    if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_test_scaled).all():# 检查标准化后是否含 NaN/Inf
        print(f"❌ [{station_pair}] 标准化后出现 NaN/Inf，已跳过（请检查源数据）。")
        return

    with open(os.path.join(output_folder, "scaler.pkl"), "wb") as f:# 保存标准化器，wb：写入二进制文件
        pickle.dump(scaler, f)  #.dump(...)：将 scaler 对象序列化并写入文件，方便后续加载使用,f 是文件对象
    print(f"[{station_pair}] Scaler 已保存：{os.path.join(output_folder, 'scaler.pkl')}")

    # -------------------------------
    # 张量 + DataLoader
    # -------------------------------
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)# 将训练集特征转换为 PyTorch 张量，数据类型为 float32
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)# 将训练集目标变量转换为 PyTorch 张量，并展平为列向量，数据类型为 float32
                                                                                #.reshape(-1, 1)：将一维数组转换为二维列向量
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)#TensorDataset(...)：将训练集特征张量和目标变量张量,封装为数据集对象，方便后续加载
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#DataLoader(...)：创建数据加载器，指定批量大小为 32，启用随机打乱数据顺序（shuffle=True）
                                                                        #每次训练用 32 个样本更新模型参数，平衡训练效率和稳定性
    # -------------------------------
    # 定义模型/损失/优化，，，，，，模型 / 损失 / 优化器配置
    # -------------------------------
    input_dim = X_train_scaled.shape[1]#.shape[1]：获取特征矩阵的列数，即输入特征的维度
                                    #自动获取特征数量，作为模型的输入维度（避免手动写死数字，更灵活）。
                                     # 输入维度=特征数（如剔除常数列后可能为7或8）
    model = Net(input_dim)## Net结构：输入层→64维隐藏层（LeakyReLU）→32维隐藏层（LeakyReLU）→1维输出层（Softplus）
    criterion = nn.SmoothL1Loss()# 使用 Smooth L1 损失函数（Huber 损失的一种形式），对异常值更鲁棒
                    #鲁棒性（Robustness） 指的是一个系统、模型或算法在面临异常、干扰、噪声或不确定性时，仍能保持稳定性能、正确输出或正常工作的能力
    optimizer = optim.Adam(model.parameters(), lr=0.001)# 使用 Adam 优化器，学习率设为 0.001

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 检测是否有可用的 GPU（CUDA），否则使用 CPU
                                        #.device(...)：指定 PyTorch 张量和模型所在的计算设备
                                        #"cuda"：表示使用 NVIDIA GPU 进行计算
                                        #"cpu"：表示使用 CPU 进行计算
    model.to(device)#.to(device)：将模型参数移动到指定设备（GPU 或 CPU）
    X_train_tensor = X_train_tensor.to(device)# 将训练集张量移动到指定设备
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # -------------------------------
    # 训练（early stopping）
    # -------------------------------
    num_epochs = 300# 训练最大轮数
    train_losses, val_losses = [], []# 记录训练损失和验证损失
    patience = 20# 早停耐心值：验证损失连续多少轮不降即停止训练
    best_val_loss = np.inf# 初始化最佳验证损失为无穷大
    trigger_times = 0#  早停计数器

    for epoch in range(num_epochs):
        model.train()# 设置模型为训练模式
        epoch_losses = []# 记录当前 epoch 的批次损失
        for batch_X, batch_y in train_loader:# 遍历训练数据加载器，获取每个批次的特征和目标变量
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)# 将批次数据移动到指定设备
            optimizer.zero_grad()#.zero_grad()：清除优化器中的梯度缓存，避免梯度累积
            outputs = model(batch_X)# 前向传播，计算模型输出
            loss = criterion(outputs, batch_y)# 计算损失
            loss.backward()# 反向传播，计算梯度
            optimizer.step()# 更新模型参数
            epoch_losses.append(loss.item())# 记录当前批次损失，.item()：获取张量的标量值
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0#.mean(epoch_losses)：计算当前 epoch 的平均训练损失
                                                                        #epoch_losses 可能为空，故加判断以防报错
        train_losses.append(train_loss)# 记录训练损失

        model.eval()#.eval()：设置模型为评估模式
        with torch.no_grad():#.no_grad()：在该代码块中禁用梯度计算，节省内存和计算资源
            val_outputs = model(X_test_tensor)# 在测试集上进行前向传播，计算模型输出
            val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)

        # 验证损失 NaN 直接跳出该区间
        if not np.isfinite(val_loss):
            print(f"❌ [{station_pair}] 验证损失为 NaN，已跳过该区间（多半源数据异常或全零能耗）。")
            return

        val_rmse_kWh = np.sqrt(val_loss)# 计算验证集 RMSE（kWh）
        val_rmse_Wh = val_rmse_kWh * 1000# 转换为 Wh
        print(f"[{station_pair}] Epoch {epoch+1}/{num_epochs} "# 输出当前 epoch 信息
              f"训练损失: {train_loss:.8f}, 验证损失: {val_loss:.8f} "
              f"(RMSE: {val_rmse_kWh:.6f} kWh ≈ {val_rmse_Wh:.2f} Wh)")

        if val_loss < best_val_loss:# 验证损失降低，保存模型并重置早停计数器
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(output_folder, "best_model_01.pth"))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"[{station_pair}] 验证损失未降低，提前停止。")
                break

    # -------------------------------
    # 测试集评估 + R²
    # -------------------------------
    model.load_state_dict(torch.load(os.path.join(output_folder, "best_model_01.pth"), map_location=device))#.load_state_dict(...)：加载最佳模型参数
                                                            #map_location=device：确保模型参数加载到正确的设备（GPU 或 CPU）
                                                            #.load(...)：从文件中加载模型参数字典
    model.to(device)# 将模型移动到指定设备
    model.eval()# 设置模型为评估模式

    with torch.no_grad():# 禁用梯度计算
        test_predictions = model(X_test_tensor)# 在测试集上进行前向传播，计算模型输出
        test_loss = criterion(test_predictions, y_test_tensor).item()# 计算测试集损失

    if (not np.isfinite(test_predictions.cpu().numpy()).all()) or (not np.isfinite(y_test_tensor.cpu().numpy()).all()):
                # 检查测试预测或标签是否含 NaN/Inf，若有则跳过 R² 计算
        print(f"❌ [{station_pair}] 测试预测或标签含 NaN/Inf，跳过 R² 计算。")
        test_r2 = float('nan')
    else:
        test_r2 = r2_score(y_test_tensor.cpu().numpy(), test_predictions.cpu().numpy())

    print(f"[{station_pair}] 测试集损失 (MSE): {test_loss:.8f} | R² = {test_r2:.4f}")

    # 保存指标
    pd.DataFrame({
        "metric": ["test_mse", "test_r2", "best_val_mse"],
        "value": [test_loss, test_r2, best_val_loss]
    }).to_csv(os.path.join(output_folder, "metrics.csv"), index=False)# 保存测试指标为 CSV 文件

    # -------------------------------
    # 全量预测 & 画图（不弹窗）
    # -------------------------------
    scaler_path = os.path.join(output_folder, "scaler.pkl")# 标准化器路径
    with open(scaler_path, "rb") as f:# 读取标准化器
                                            #rb：以二进制读模式打开文件
        scaler_loaded = pickle.load(f)#.load(f)：从文件中反序列化加载 scaler 对象

    X_full_scaled = scaler_loaded.transform(X_df.values)# 对全量特征矩阵进行标准化转换
    X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32).to(device)

    model.eval()# 设置模型为评估模式
    with torch.no_grad():# 禁用梯度计算
        full_pred_increment = model(X_full_tensor).cpu().numpy().flatten()#.flatten()：将预测结果展平为一维数组

    # 预测增量 → kWh（pred 输出是 Wh；与原逻辑一致）
    full_pred_increment = np.nan_to_num(full_pred_increment, nan=0.0, posinf=0.0, neginf=0.0)# 将预测结果中的 NaN 和 Inf 替换为 0，确保数值稳定性
    df_result = df_result.loc[keep_mask].reset_index(drop=True)# 保留与最终训练数据对应的行，重置索引
                                                    #.loc[keep_mask]：根据保留掩码筛选行q
                                                    #.reset_index(drop=True)：重置索引，避免索引不连续，drop=True 表示不保留旧索引
    df_result['predicted_increment'] = full_pred_increment / 1000.0  # Wh → kWh

    # 分段累计
    df_result.sort_values('时刻', inplace=True)# 按时间排序，确保累计计算正确,inplace=True：直接在原 DataFrame 上修改
    df_result['predicted_cumulative'] = df_result.groupby('unique_segment')['predicted_increment'].cumsum()
                                                                    # 按 unique_segment 分组，计算预测增量的累计和
                                                                    #.cumsum()：计算累计和

    # 整体对比图
    plt.figure()
    for seg, seg_data in df_result.groupby('unique_segment'):
        plt.plot(seg_data['时刻'], seg_data['predicted_cumulative'], label=f'预测能耗 - {seg}')
    for seg, seg_data in df_result.groupby('unique_segment'):
        plt.plot(seg_data['时刻'], seg_data['cumulative_energy_kWh'], label=f'实际能耗 - {seg}', linestyle='--')
    plt.xlabel('时间'); plt.ylabel('能耗 (kWh)')
    plt.title(f'预测 vs 实际 能耗时间序列（{station_pair}）')
    plt.legend(); plt.grid(True)
    fig_path = os.path.join(output_folder, "time_series_comparison_energy_01.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 子图网格
    segments = df_result['unique_segment'].unique()# 获取所有唯一的区间段,.unique()：返回唯一值的数组
    n_segments = len(segments)# 计算区间段数量
    ncols = int(np.ceil(np.sqrt(n_segments)))#.ceil(np.sqrt(n_segments))：计算子图网格的列数，取区间段数量的平方根并向上取整
                                                    #使子图尽量接近正方形布局
                                                    #np.sqrt(...)：计算平方根
                                                    #np.ceil(...)：向上取整
    nrows = int(np.ceil(n_segments / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))

    if n_segments == 1:
        axes = np.array([[axes]])
    else:
        axes = np.array(axes).reshape(nrows, ncols)

    for idx, seg in enumerate(segments):# 遍历每个区间段，绘制子图
        row, col = idx // ncols, idx % ncols# 计算当前子图的行列索引
        ax = axes[row, col]#axes[row, col]：获取当前子图的轴对象
        seg_data = df_result[df_result['unique_segment'] == seg].copy()# 获取当前区间段的数据
        seg_data.sort_values('时刻', inplace=True)
        seg_data['predicted_cumulative'] = np.cumsum(seg_data['predicted_increment'].values)
        ax.plot(seg_data['时刻'], seg_data['predicted_cumulative'], label='预测能耗')
        ax.plot(seg_data['时刻'], seg_data['cumulative_energy_kWh'], label='实际能耗', linestyle='--')
        ax.set_xlabel('时间'); ax.set_ylabel('能耗 (kWh)')
        ax.set_title(f'{seg}'); ax.legend(); ax.grid(True)

    # 隐藏多余子图
    total_cells = nrows * ncols
    for j in range(n_segments, total_cells):
        row, col = j // ncols, j % ncols
        fig.delaxes(axes[row, col])

    plt.suptitle(f'预测能耗与实际能耗时间序列对比（{station_pair}）', fontsize=16)#subtitle：设置整体标题
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    subplots_path = os.path.join(output_folder, "time_series_comparison_energy_subplots_01.png")
    plt.savefig(subplots_path, dpi=300)
    plt.close()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_folder, "trained_model_01.pth"))
    print(f"[{station_pair}] ✅ 完成训练，模型与图表已输出到：{output_folder}")


if __name__ == "__main__":
    data_dir = "data_processed" # 输入数据目录
                                #这是 Python 的标准程序入口写法，作用是：只有当该文件被 “直接运行” 时（而非被其他文件导入为模块），才执行下面的代码；

    # 打印存在/缺失情况（按 26 段固定顺序）
    print(f"将按顺序训练 5 号线区间，共 {len(LINE5_SECTIONS)} 段：")#输出如 “将按顺序训练 5 号线区间，共 26 段：”
    for idx, pair in enumerate(LINE5_SECTIONS, 1):#enumerate(LINE5_SECTIONS, 1)：遍历列表时，同时获取 “索引 + 区间名”，索引从 1 开始
                                                    #enumerate 是 Python 的一个内置函数，用于在遍历可迭代对象（如列表、元组、字符串等）时，同时获取元素的索引和值。
        print(f"  [{idx:02d}/{len(LINE5_SECTIONS):02d}] {pair}")# 输出如 “  [01/26] 布政-张家潭”

    # 逐段训练
    for pair in LINE5_SECTIONS:
        train_one_section(pair, data_dir=data_dir)# 调用 train_one_section 函数，传入当前区间名和数据目录，进行训练
