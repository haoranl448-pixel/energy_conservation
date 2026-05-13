# src/models/wrapper.py
import torch
import numpy as np
import pickle
import os
# 导入所有定义的模型类
from .definitions import Net, EnergyTransformer, ResidualTransformer

class EnergyPredictor:
    def __init__(self, model_type, model_folder, device='cuda'):
        """
        :param model_type: 'mlp', 'transformer', 'residual'
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_folder = model_folder
        self._load_resources()

    def _load_resources(self):
        # 1. 加载 Scaler
        try:
            with open(os.path.join(self.model_folder, "scaler_x.pkl"), 'rb') as f:
                self.scaler_x = pickle.load(f)
            with open(os.path.join(self.model_folder, "scaler_y.pkl"), 'rb') as f:
                self.scaler_y = pickle.load(f)
        except FileNotFoundError:
            # 兼容 MLP
            with open(os.path.join(self.model_folder, "scaler.pkl"), 'rb') as f:
                self.scaler_x = pickle.load(f)
            self.scaler_y = None

        # 2. 初始化模型架构 & 确定权重文件名
        input_dim = self.scaler_x.mean_.shape[0]
        weights_name = "best_model.pth"

        if self.model_type == 'mlp':
            self.model = Net(input_dim).to(self.device)
            
        elif self.model_type == 'transformer':
            self.model = EnergyTransformer(input_dim=input_dim).to(self.device)
            weights_name = "best_transformer.pth"
            
        elif self.model_type == 'residual':
            # 关键修改：使用 ResidualTransformer 类
            self.model = ResidualTransformer(input_dim=input_dim).to(self.device)
            weights_name = "best_res_model.pth"
        elif self.model_type == 'residualV2':
            # 关键修改：使用 ResidualTransformer 类
            self.model = ResidualTransformer(input_dim=input_dim).to(self.device)
            weights_name = "best_res_model.pth"
        # 3. 加载权重
        pth_path = os.path.join(self.model_folder, weights_name)
        # 容错：如果指定名字找不到，尝试找 best_model.pth
        if not os.path.exists(pth_path):
            pth_path = os.path.join(self.model_folder, "best_model.pth")
            
        self.model.load_state_dict(torch.load(pth_path, map_location=self.device))
        self.model.eval()

    def predict(self, input_data):
        # 1. 归一化输入
        input_scaled = self.scaler_x.transform(input_data)

        # 2. 推理
        if self.model_type == 'mlp':
            tensor_x = torch.FloatTensor(input_scaled).to(self.device)
            with torch.no_grad():
                pred = self.model(tensor_x).cpu().numpy()
            return pred.flatten()

        elif self.model_type == 'residualV2':
            seq_x = self._create_sequences(input_scaled, seq_len=30)
            if len(seq_x) == 0: return np.zeros(len(input_data))

            tensor_x = torch.FloatTensor(seq_x).to(self.device)
            with torch.no_grad():
                pred_scaled = self.model(tensor_x).cpu().numpy()
            
            # 反归一化
            pred = self.scaler_y.inverse_transform(pred_scaled).flatten()
            
            # 对齐长度 (补0)
            pad = np.zeros(30 - 1)
            full_pred = np.concatenate([pad, pred])
            return full_pred

        elif self.model_type in ['transformer', 'residual']:
            # 切片
            seq_x = self._create_sequences(input_scaled, seq_len=30)
            if len(seq_x) == 0: return np.zeros(len(input_data))

            tensor_x = torch.FloatTensor(seq_x).to(self.device)
            with torch.no_grad():
                pred_scaled = self.model(tensor_x).cpu().numpy()
            
            # 反归一化
            pred = self.scaler_y.inverse_transform(pred_scaled).flatten()
            
            # 对齐长度 (补0)
            pad = np.zeros(30 - 1)
            full_pred = np.concatenate([pad, pred])
            
            # 物理约束: 如果是普通 Transformer，能耗>=0
            if self.model_type == 'transformer':
                return np.maximum(full_pred, 0)
            else:
                # 残差模型输出的是差值，不做非负约束
                return full_pred

    def _create_sequences(self, data, seq_len):
        padding = np.tile(data[0], (seq_len - 1, 1))
        data_padded = np.vstack([padding, data])
        xs = []
        for i in range(len(data)):
            xs.append(data_padded[i : i + seq_len])
        return np.array(xs)