# src/models/definitions.py
import torch
import torch.nn as nn
import math

# ================= 1. MLP 模型 =================
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        return torch.nn.functional.softplus(self.fc3(x))

# ================= 2. 公共组件 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ================= 3. 纯 Transformer (EnergyTransformer) =================
# 特征：变量名叫 transformer_encoder，输出有 Softplus
class EnergyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(EnergyTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, dropout, batch_first=True)
        # ⚠️ 注意这里叫 transformer_encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
            nn.Softplus() 
        )

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :])

# ================= 4. 残差 Transformer (ResidualTransformer) =================
# 特征：变量名叫 transformer，输出没有 Softplus
class ResidualTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        # ⚠️ 注意这里叫 transformer (为了匹配权重文件)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # ⚠️ 残差输出无 Softplus
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32), 
            nn.LeakyReLU(0.01), 
            nn.Linear(32, 1)
        )

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        out = self.transformer(src)
        return self.decoder(out[:, -1, :])