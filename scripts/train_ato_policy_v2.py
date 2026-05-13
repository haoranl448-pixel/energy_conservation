# -*- coding: utf-8 -*-
import os, glob, pickle, re, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ================= 1. 配置与路径 =================
STATION_PAIR = "布政-张家潭"
TARGET_L = 1428.0 
FEATURE_COLS = ['速度(m/s)', 'curvature', 'gradient', '重量', 'dist_rem', 'class_id', 'v_ref']
TARGET_COL = '加速度(m/s²)'

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_ROOT = os.path.join(project_root, "data", "raw_data")
PROCESSED_DATA_FILE = os.path.join(project_root, "data", "data_processed", f"results_{STATION_PAIR}.xlsx")
REF_CURVE_FILE = os.path.join(project_root, "output", "schedule", "class_based_results", "segments", STATION_PAIR, "vt_selected.csv")
OUTPUT_DIR = os.path.join(project_root, "output", "models", "ato_results_v3")

# ================= 辅助函数：强力清洗服务号 =================
def clean_service_id(sid):
    """把各种格式的服务号统一转为纯数字字符串，如 50704.0 -> '50704'"""
    if pd.isna(sid): return None
    s = str(sid).strip()
    if s.endswith('.0'): s = s[:-2]
    return s

# ================= 2. 核心网络 =================
class ATOPolicyNetV3(nn.Module):
    def __init__(self, input_dim):
        super(ATOPolicyNetV3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.fc(x)

# ================= 3. 执行训练 =================
def train_v3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- A. 加载参考速度 ---
    if not os.path.exists(REF_CURVE_FILE):
        print(f"❌ 找不到参考曲线: {REF_CURVE_FILE}"); return
    df_ref = pd.read_csv(REF_CURVE_FILE).sort_values('s_local(m)').drop_duplicates('s_local(m)')
    f_v_ref = interp1d(df_ref['s_local(m)'], df_ref['v(m/s)'], kind='linear', bounds_error=False, fill_value=0)

    # --- B. 建立 [服务号 -> 等级] 映射 ---
    service_to_class = {}
    print(f"🔍 正在扫描原始等级数据 (raw_data)... ")
    date_folders = [d for d in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))]
    
    for date_dir in date_folders:
        pattern = os.path.join(RAW_DATA_ROOT, date_dir, f"*{STATION_PAIR}.xlsx")
        for f in glob.glob(pattern):
            if os.path.basename(f).startswith("~$"): continue
            df_raw = pd.read_excel(f)
            if '服务号' in df_raw.columns and '计划运行等级' in df_raw.columns:
                for _, row in df_raw.iterrows():
                    sid = clean_service_id(row['服务号'])
                    match = re.search(r'(\d+)', str(row['计划运行等级']))
                    if sid and match:
                        service_to_class[sid] = float(match.group(1))
    
    print(f"✅ 原始表扫描完成：共提取 {len(service_to_class)} 个服务号等级映射。")
    if len(service_to_class) > 0:
        print(f"   样例映射: {list(service_to_class.items())[:3]}")

    # --- C. 加载物理数据并强力匹配 ---
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"❌ 找不到处理表: {PROCESSED_DATA_FILE}"); return
    
    print(f"🚀 正在加载物理细节数据并匹配等级...")
    df_phys = pd.read_excel(PROCESSED_DATA_FILE)
    
    # 诊断打印：查看处理表的服务号长什么样
    sample_sid = df_phys['服务号'].iloc[0]
    print(f"   处理表服务号样例: '{sample_sid}' (类型: {type(sample_sid)})")

    # 关键：对处理表的服务号也进行清洗
    df_phys['sid_clean'] = df_phys['服务号'].apply(clean_service_id)
    df_phys['class_id'] = df_phys['sid_clean'].map(service_to_class)

    # 剔除无法匹配等级的数据
    df_phys = df_phys.dropna(subset=['class_id'])
    
    if len(df_phys) == 0:
        print("\n❌ 匹配失败：处理表里的服务号在原始等级映射中找不到！")
        print(f"   处理表清洗后的服务号样例: {df_phys['sid_clean'].iloc[:3].tolist() if 'sid_clean' in df_phys else 'N/A'}")
        return

    # 注入衍生特征
    df_phys['dist_rem'] = TARGET_L - df_phys['累计位移(m)']
    df_phys['v_ref'] = f_v_ref(df_phys['累计位移(m)'])

    # --- D. 数据清洗与训练 ---
    df_clean = df_phys[FEATURE_COLS + [TARGET_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    X = df_clean[FEATURE_COLS].values
    y = df_clean[TARGET_COL].values
    print(f"📊 匹配成功！有效样本总数: {len(X)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ATOPolicyNetV3(len(FEATURE_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"🧠 模型训练中...")
    for epoch in range(500):
        model.train()
        X_t, y_t = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
        optimizer.zero_grad(); loss = criterion(model(X_t), y_t); loss.backward(); optimizer.step()
        if (epoch+1) % 100 == 0: print(f"Epoch {epoch+1}/500 | Loss: {loss.item():.6f}")

    # --- E. 保存 ---
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "ato_model_v3.pth"))
    with open(os.path.join(OUTPUT_DIR, "ato_scaler_v3.pkl"), "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "v_ref_interp.pkl"), "wb") as f: pickle.dump(f_v_ref, f)
    print(f"🎉 全部完成！模型已保存。")

if __name__ == "__main__":
    train_v3()