import torch
import pickle
import numpy as np
import pandas as pd
import os
import shutil
import torch.nn as nn

# ==========================================
# 1. 定义网络结构 (必须与训练时完全一致)
# ==========================================
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # 导出权重不需要 forward 逻辑，但定义完整类结构更安全
        pass

# ==========================================
# 2. 导出逻辑
# ==========================================
def export_all_sections(
    params_csv="section_params.csv", 
    results_root="nn_results", 
    output_root="cpp_assets"
):
    # 1. 准备输出根目录
    if os.path.exists(output_root):
        print(f"清理旧导出目录: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    # 2. 读取区间列表
    if not os.path.exists(params_csv):
        print(f"错误: 找不到 {params_csv}，无法获知有哪些区间。")
        return

    df = pd.read_csv(params_csv)
    # 复制参数表到 cpp_assets (供 C++ 读取物理参数)
    df.to_csv(os.path.join(output_root, "section_params.csv"), index=False)
    print(f"已复制 section_params.csv 到 {output_root}")

    success_count = 0
    fail_count = 0

    print(f"\n开始导出 26 个区间的模型参数...")

    # 3. 遍历每个区间
    for index, row in df.iterrows():
        station_pair = str(row['station_pair']).strip()
        
        # 源文件路径
        model_path = os.path.join(results_root, station_pair, "best_model.pth")
        scaler_path = os.path.join(results_root, station_pair, "scaler.pkl")

        # 目标文件夹: cpp_assets/布政-张家潭/
        target_dir = os.path.join(output_root, station_pair)
        
        # 检查源文件是否存在
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"⚠️ [跳过] 缺失模型或Scaler: {station_pair}")
            fail_count += 1
            continue

        os.makedirs(target_dir, exist_ok=True)

        try:
            # --- A. 导出 Scaler (均值/方差) ---
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # 保存为 txt，方便 C++ 读取
            np.savetxt(os.path.join(target_dir, "mean.txt"), scaler.mean_)
            np.savetxt(os.path.join(target_dir, "scale.txt"), scaler.scale_)

            # --- B. 导出 模型权重 ---
            input_dim = len(scaler.mean_) # 自动获取输入维度
            model = Net(input_dim)
            # 加载训练好的权重
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

            # 遍历层参数并保存
            for name, param in model.named_parameters():
                # name 类似 fc1.weight, fc1.bias
                # 替换点号，变成 fc1_weight.txt
                fname = name.replace('.', '_') + ".txt"
                save_path = os.path.join(target_dir, fname)
                
                # 转为 numpy 并保存文本
                np.savetxt(save_path, param.detach().cpu().numpy())

            print(f"✅ [成功] {station_pair}")
            success_count += 1

        except Exception as e:
            print(f"❌ [错误] 导出 {station_pair} 失败: {e}")
            fail_count += 1

    print(f"\n----------- 导出总结 -----------")
    print(f"成功: {success_count} 个")
    print(f"失败/跳过: {fail_count} 个")
    print(f"导出目录: {os.path.abspath(output_root)}")
    print(f"-------------------------------")

if __name__ == "__main__":
    export_all_sections()