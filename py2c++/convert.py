import torch
import pickle
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 假设你的模型定义在 train_mlp_legacy.py
from scripts.train_mlp_legacy import Net 

def export_all(output_dir="cpp_assets"):
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> 正在导出资源到 {output_dir} ...")

    # 1. 导出模型权重 (假设你已经训练好了 best_model.pth)
    model_path = "nn_results/布政-张家潭/best_model.pth" # 选一个代表性的模型
    scaler_path = "nn_results/布政-张家潭/scaler.pkl"
    
    if not os.path.exists(model_path):
        print("错误：找不到模型文件，请先运行 train_line5.py")
        return

    # 加载 Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    np.savetxt(f"{output_dir}/scaler_mean.txt", scaler.mean_)
    np.savetxt(f"{output_dir}/scaler_scale.txt", scaler.scale_)

    # 加载模型并导出权重 (Linear层: weight + bias)
    model = Net(len(scaler.mean_))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    for name, param in model.named_parameters():
        # name 类似 fc1.weight, fc1.bias
        fname = name.replace('.', '_') + ".txt"
        # 转为 numpy 并保存
        np.savetxt(f"{output_dir}/{fname}", param.detach().cpu().numpy())
    
    # 2. 导出区间参数 (Section Params)
    # 将 csv 复制过去或者清洗一下
    if os.path.exists("section_params.csv"):
        df = pd.read_csv("section_params.csv")
        # 可以在这里做一些预处理，比如把中文列名改成英文，或者直接复制
        df.to_csv(f"{output_dir}/section_params.csv", index=False)
    
    print("✅ 导出完成！C++ 程序可以读取了。")

if __name__ == "__main__":
    export_all()