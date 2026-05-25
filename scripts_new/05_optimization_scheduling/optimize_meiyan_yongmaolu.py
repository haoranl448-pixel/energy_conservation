# -*- coding: utf-8 -*-
"""
optimize_meiyan_yongmaolu_fixedprefix.py
只计算【梅堰-永茂路】一段：
- 前段(0~约103.878s)速度时序固定，末速10.667 m/s
- 后段(余程)：以 v0=10.667 为初速，加速到 vmax→(可选匀速)→以 -0.55 刹停到 0
- 给定到达时间窗口，对每个到达时间解析求解对应 vmax，评估能耗并出图出表
"""

import os, glob, json, math, pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================= 常量/路径 =================
PAIR = "梅堰-永茂路"
DT = 0.05

# 线路总长（米）
L_TOTAL = 3734.0

# 前段固定所用加/减速度
A_POS = 0.60      # m/s^2
A_NEG = -0.55     # m/s^2

# 后段速度上限（不要超过训练/规则的极限；你之前给 18）
V_MAX_UPPER = 18.0

# 时间窗口：总到达时间（可按需要改）
T_MIN, T_MAX, T_STEP = 284.0, 345.0, 0.5

# 目录结构
# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
DT = 0.05  # 时间步长(固定)
SEQ_LEN = 30
DATA_DIR = Path(project_root, "data","data_processed")
NN_DIR   = Path(project_root, "output","models","nn_results_mlp1")
oUT_DIR  = Path(project_root, "output","optimization","opt_results_mlp1")
PARAMS_CSV = Path(project_root, "data", "static", "section_params_trip6.csv")
OUT_DIR = oUT_DIR / PAIR

OUT_DIR.mkdir(parents=True, exist_ok=True)#parents=True 如果父目录不存在，会自动创建所有缺失的父目录
                                            #exist_ok=True：如果目录已经存在，不会抛出异常


NN_RESULT_DIR = NN_DIR / PAIR
MODEL_PTH     = NN_RESULT_DIR / "best_model_01.pth"
SCALER_PKL    = NN_RESULT_DIR / "scaler.pkl"


FEAT_JSON  = NN_DIR / "feature_columns.json"  # 训练时保存的列顺序

EPS_T = 1e-6
EPS_S = 1e-6

# =============== NN 定义 ===============
class Net(nn.Module):#Net类继承自torch.nn.Module
    def __init__(self, d):
        super().__init__()#必须调用！目的是初始化父类 nn.Module 的内部变量
        self.fc1 = nn.Linear(d, 64)#输入层到隐藏层1的线性变换，输入维度为d，输出维度为64    nn.Linear是PyTorch中用于创建全连接层的类
        self.fc2 = nn.Linear(64, 32)#隐藏层1到隐藏层2的线性变换，输入维度为64，输出维度为32
        self.fc3 = nn.Linear(32, 1)#隐藏层2到输出层的线性变换，输入维度为32，输出维度为1    
                                        #输出维度 1：表示模型是 单输出回归模型（如预测 “能耗”“运行时间” 等单个连续值）。
    def forward(self, x):#forward 是神经网络的 核心方法，定义了 数据在模型中的流动路径（输入→各层→输出）。当调用模型实例时，实际上是调用这个方法。
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.01)# 对输入 x 先经过 fc1 层的线性变换，然后应用 Leaky ReLU 激活函数（负斜率为0.01）
                                                            # Leaky ReLU 是一种常用的激活函数，能够帮助模型学习非线性关系，同时避免 ReLU 的“死亡”问题（即神经元输出恒为0）。
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.01)
        return torch.nn.functional.softplus(self.fc3(x))# 最后一层使用 Softplus 激活函数，确保输出为正值（适合预测能耗等非负量）
                                                        # Softplus 是一种平滑的近似 ReLU 的激活函数，定义为 log(1 + exp(x))。

# =============== 读取静态量并做按位移插值 ===============
def _read_results_concat(pair: str) -> pd.DataFrame:#函数名前缀 _：按 Python 惯例，表示这是 内部函数（仅在当前模块内使用，不推荐外部调用）；
                                                    #参数 pair: str，输入参数，路段标识（如 "梅堰-永茂路"），用于匹配对应的结果文件
                                                    #返回类型 pd.DataFrame，表示函数返回一个 Pandas DataFrame 对象，包含合并后的数据。
    pattern = str(DATA_DIR / f"results_{pair}*.xlsx")#构造文件匹配模式，查找 data_processed 目录下符合 results_梅堰-永茂路*.xlsx 的所有文件
                                                        #f-string 用于字符串格式化，* 是通配符，表示任意字符序列
                                                        #字符串格式化：将路径对象转换为字符串，方便 glob 使用
    files = sorted(glob.glob(pattern))#使用 glob 模块查找所有匹配 pattern 的文件，并按字母顺序排序，返回文件路径列表
                                        #glob.glob(pattern) 返回所有匹配 pattern 的文件路径列表
                                        #sorted() 对列表进行排序，默认按字母顺序
    if not files:#如果没有找到任何匹配的文件，抛出 FileNotFoundError 异常，提示用户缺少数据文件
        raise FileNotFoundError(f"[{pair}] 未找到数据文件：{pattern}")
    dfs = []#用于存储读取的 DataFrame 对象 的列表
    for fp in files:#遍历每个匹配的文件路径
        try: dfs.append(pd.read_excel(fp))#正常情况：pd.read_excel(fp) 成功读取 Excel 文件，返回 DataFrame 并加入 dfs；
        except Exception as e: print(f"[跳过] 读取失败：{fp} - {e}")#异常情况：读取失败，打印错误信息并跳过该文件
    if not dfs:#如果没有成功读取任何文件，抛出 RuntimeError 异常，提示用户无可用的 results_*.xlsx 文件
        raise RuntimeError(f"[{pair}] 无可用 results_*.xlsx")
    return pd.concat(dfs, ignore_index=True)#将多个 DataFrame 合并为一个，忽略原有索引，重新生成连续索引。
                                            #concat 函数用于沿指定轴（默认是行轴）连接多个 DataFrame 对象，形成一个新的 DataFrame。
                                            #dfs 是待合并的 DataFrame 列表，默认 纵向拼接（按行合并，类似数据库的 UNION ALL）；
                                            #ignore_index=True 表示忽略原有索引，重新生成连续索引。

def load_static_series(pair: str):
    df = _read_results_concat(pair)#读取并合并指定路段的所有 results_*.xlsx 文件，返回一个包含所有数据的 DataFrame
    s_raw = pd.to_numeric(df['累计位移(m)'], errors='coerce').to_numpy()#将 '累计位移(m)' 列转换为数值类型，无法转换的值设为 NaN，然后转换为 NumPy 数组
                                                                        #errors='coerce'：表示在转换过程中遇到无法转换的值时，将其设置为 NaN（而不是抛出错误）
                                                                        #to_numpy()：将 Pandas Series 转换为 NumPy 数组，便于后续数值计算
                                                                        #.to_numeric()：将数据转换为数值类型（float 或 int），非数值会被转换为 NaN
    mask  = np.isfinite(s_raw)#.isfinite(s_raw)：创建一个布尔掩码数组，标记 s_raw 中的每个元素是否为有限数值（非 NaN、非正负无穷大）
                                #返回：布尔数组 mask（True 表示有效，False 表示无效）；
    s_raw = s_raw[mask]# # 过滤位移数组的无效值
                                #仅保留 mask 中为 True 的元素，即有效的位移值
    df    = df.loc[mask].copy()#.loc[mask]：使用布尔掩码过滤 DataFrame，只保留对应 mask 为 True 的行
                                #.copy()：创建过滤后的 DataFrame 的副本，避免后续操作对原 DataFrame 产生影响
    order = np.argsort(s_raw)#.argsort(s_raw)：获取 s_raw 数组排序后的索引顺序，返回一个整数数组 order
                            #返回：整数数组 order，表示将 s_raw 排序后各元素在原数组中的位置；
    s = s_raw[order]#按位移排序后的数组
    df = df.iloc[order]#.iloc[order]：使用排序索引重新排列 DataFrame 的行，使其与排序后的 s_raw 对齐
                        #返回：按位移排序后的 DataFrame df；

    curvature = pd.to_numeric(df.get('curvature', 0.0), errors='coerce').fillna(0.0).to_numpy()
                    #.to_numeric(df.get('curvature', 0.0), errors='coerce')：将 'curvature' 列转换为数值类型，无法转换的值设为 NaN；若列不存在，使用默认值 0.0
                    #.fillna(0.0)：将 NaN 值替换为 0.0
                    #.to_numpy()：将 Pandas Series 转换为 NumPy 数组
    gradient  = pd.to_numeric(df.get('gradient', 0.0),  errors='coerce').fillna(0.0).to_numpy()
    mass_col  = pd.to_numeric(df.get('重量', 0.0),      errors='coerce').fillna(0.0).to_numpy()

    if len(s) < 2 or np.allclose(np.diff(s), 0):#如果位移数据点少于2个，或所有位移值都相同（无变化），则无法进行插值
                                                #.diff(s)：计算相邻位移值的差异
                                                #np.allclose(..., 0)：检查所有差异是否都接近于0
                                                #.allclose()：用于判断两个数组是否在一定容差范围内相等
        curve_f = lambda x: np.zeros_like(np.asarray(x, dtype=float))#如果无法插值，返回恒为0的函数
                                                                        #.zeros_like(...)：创建一个与输入数组形状相同的全零数组
                                                                        #np.asarray(x, dtype=float)：将输入 x 转换为浮点型数组
        grade_f = lambda x: np.zeros_like(np.asarray(x, dtype=float))
        mass_f  = lambda x: np.zeros_like(np.asarray(x, dtype=float))
    else:
        curve_f = interp1d(s, curvature, fill_value='extrapolate')#interp1d： 创建一维线性插值函数
                                                                    #线性插值原理：若输入 x 在 s[i] 和 s[i+1] 之间，则通过两点 (s[i], curvature[i]) 和 (s[i+1], curvature[i+1]) 拟合直线，计算 x 对应的 y 值（曲率）。
                                                                    #s：自变量数组（位移）
                                                                    #curvature：因变量数组（曲率）
                                                                    #fill_value='extrapolate'：允许在数据范围外进行外推，即对于超出 s 范围的 x 值，仍能计算对应的曲率值
        grade_f = interp1d(s, gradient,  fill_value='extrapolate')
        mass_f  = interp1d(s, mass_col,  fill_value='extrapolate')
    return curve_f, grade_f, mass_f

def load_model_and_scaler(pair: str):#核心功能是：加载训练好的神经网络模型（Net）、数据标准化器（scaler）和特征列名列表（feature_cols），
                                        #并做一致性校验，确保三者匹配（特征维度一致），为后续输入数据预处理和模型预测铺路。
    if not MODEL_PTH.exists() or not SCALER_PKL.exists():#如果模型文件或 scaler 文件不存在，抛出 FileNotFoundError 异常，提示用户缺少必要文件
        raise FileNotFoundError(f"[{pair}] 缺少模型或scaler：{MODEL_PTH} / {SCALER_PKL}")
    with open(SCALER_PKL, "rb") as f:#with 语句用于打开文件，确保在使用后正确关闭文件
                                        #"rb" 模式表示以二进制读取文件
                                        #as f：将打开的文件对象赋值给变量 f，供后续代码使用
        scaler = pickle.load(f)#使用 pickle 模块从文件中加载数据标准化器对象 scaler
                                        #pickle.load(f)：从文件对象 f 中反序列化数据，恢复为原始的 Python 对象（这里是 scaler）
    if FEAT_JSON.exists():#如果特征列文件存在，读取其中的特征列名列表
        with open(FEAT_JSON, "r", encoding="utf-8") as f:#以只读模式打开特征列文件，指定 UTF-8 编码
                                                            #as f：将打开的文件对象赋值给变量 f
            feature_cols = json.load(f)#使用 json 模块从文件中加载特征列名列表
                                        #json.load(f)：从文件对象 f 中读取 JSON 数据，并将其解析为 Python 对象（这里是特征列名列表）
    else:
        feature_cols = ['time','velocity','acceleration','prev_velocity','prev_acceleration',
                        'curvature','gradient','mass']#如果特征列文件不存在，使用默认的特征列名列表
    net = Net(len(scaler.mean_))#实例化神经网络模型 Net，输入维度为 scaler.mean_ 的长度（即特征数量）
                                #.mean_：scaler 对象的属性，表示用于标准化的特征均值数组
    net.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))#.load_state_dict(...)：加载模型参数
                                                            #torch.load(MODEL_PTH, map_location="cpu")：从指定路径加载模型参数，map_location="cpu" 表示将模型加载到 CPU 上（适用于没有 GPU 的环境）

    net.eval()#.eval()：将模型设置为评估模式，禁用 dropout 和 batch normalization 等训练时特有的行为
                #对比：训练时用 net.train() 启用训练模式；推理时必须用 net.eval()，否则预测结果会波动（同一输入可能得到不同输出）。
    if len(feature_cols) != len(scaler.mean_):#如果特征列数与 scaler 的均值数组长度不匹配，抛出 RuntimeError 异常，提示用户维度不一致
        raise RuntimeError(f"[{pair}] scaler 维度({len(scaler.mean_)})与特征列数({len(feature_cols)})不一致")
    return net, scaler, feature_cols

# =============== 前段固定剖面 ===============
def fixed_prefix(A_POS=0.60, A_NEG=-0.55, v_to=10.667):
    """
    固定时序：
      0–30s +A_POS
      30–48s A_NEG
      48–84s 匀速
      84–90s +A_POS
      90–102s 匀速
      102–t6 A_NEG 到 v_to
    返回 dict: {'t','v','a','s','T','L','v_end'}
    """
    def piece(t0, dur, v0, a):#构建一段匀加速运动的时间-速度-加速度-位移剖面
        n = max(1, int(np.ceil(dur/DT)))#计算该段的时间步数，确保至少有一个时间步
                                        #np.ceil(dur/DT)：计算持续时间 dur 除以时间步长 DT 的上限整数，确保覆盖整个持续时间，.ceil()：向上取整
                                        #int(...)：将上限整数转换为整数类型
        t = np.linspace(0, dur, n+1)#.linspace(0, dur, n+1)：生成从 0 到 dur 的 n+1 个均匀分布的时间点数组 t
        v = (v0 + a*t) if abs(a) > 1e-12 else np.full_like(t, v0)#条件判断 abs(a) > 1e-12：判断加速度是否 “有效”（非零，考虑浮点误差）；
                                                                    #若有效，则计算速度 v = v0 + a*t（匀加速公式）；否则，速度恒为初速度 v0
                                                                    #np.full_like(t, v0)：创建一个与 t 形状相同的数组，所有元素均为 v0
        v = np.clip(v, 0.0, None)#确保速度非负，使用 np.clip 将速度数组 v 中的所有值限制在 [0.0, +∞) 范围内
                                        #.clip(v, 0.0, None)：将 v 中小于 0.0 的值设为 0.0，大于 None（无穷大）的值不变
                                        #np.clip(arr, a_min, a_max)：将数组 arr 中的元素限制在 [a_min, a_max] 范围内；
                                            #a_min=0.0：速度最小值为 0（列车不会反向运动，速度不能为负）；
                                            #a_max=None：速度无上限（不限制最大速度，由业务逻辑控制）；
                                            #示例：若匀减速时速度计算为负（如 v0=0.5m/s，a=-1m/s²，t=1.0s → v=0.5-1×1=-0.5）→ 裁剪后 v=0.0。
        s = v0*t + 0.5*a*t*t if abs(a) > 1e-12 else v0*t#计算位移 s
        return t0 + t, v, np.full_like(t, a), s#返回：绝对时间数组 t0+t，速度数组 v，加速度数组（恒为 a），位移数组 s

    # 段参数
    a1, a2 = A_POS, A_NEG
    t1, t2, t3, t4, t5 = 30.0, 18.0, 36.0, 6.0, 12.0

    # 逐段
    t0 = 0.0; v0 = 0.0; S_accum = 0.0
    T_all, V_all, A_all, S_all = [], [], [], []

    # 0-30
    T,V,A,S = piece(t0, t1, v0, a1); 
    T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]#更新初始条件为下一段的起点

    # 30-48
    T,V,A,S = piece(t0, t2, v0, a2); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

    # 48-84 匀速
    T,V,A,S = piece(t0, t3, v0, 0.0); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

    # 84-90
    T,V,A,S = piece(t0, t4, v0, a1); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

    # 90-102 匀速
    T,V,A,S = piece(t0, t5, v0, 0.0); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

    # 102-? 减到 v_to
    t6 = (v0 - v_to)/abs(a2)
    T,V,A,S = piece(t0, t6, v0, a2); T_all.append(T); V_all.append(V); A_all.append(A); S_all.append(S+S_accum)
    t0 = T[-1]; v0 = V[-1]; S_accum = S_all[-1][-1]

    # 拼接
    t = np.concatenate(T_all)#.concatenate(...)：将多个数组沿指定轴连接成一个数组
    v = np.concatenate(V_all)
    a = np.concatenate(A_all)
    s = np.concatenate(S_all)

    return {'t': t, 'v': v, 'a': a, 's': s, 'T': t[-1], 'L': s[-1], 'v_end': v[-1]}

# =============== 后段解析求 vmax ===============
def solve_vmax_for_time_distance(s_var, t_var, v_init, A_POS, A_NEG, vmax_cap=np.inf):#.inf：表示正无穷大
    """
    后段模型：加速到 vmax（+A_POS）→ 匀速 tc → 以 |A_NEG| 刹停到 0
    方程：
      t_var = (vmax - v_init)/A_POS + tc + vmax/|A_NEG|
      s_var = (vmax^2 - v_init^2)/(2A_POS) + vmax*tc + vmax^2/(2|A_NEG|)
    消去 tc 得关于 vmax 的二次方程 A v^2 + B v + C = 0
    返回 (vmax, tc)；若不可解，返回 (None, None)
    """
    if t_var <= 0 or s_var <= 0: return None, None#无效输入，直接返回 None
    ap = A_POS; an = abs(A_NEG)
    A = (1.0/ap + 1.0/an)#二次项系数
    B = -2.0 * (t_var + v_init/ap)#一次项系数
    C =  2.0 * (s_var + (v_init*v_init)/(2.0*ap))#常数项
    D = B*B - 4*A*C#判别式
    candidates = []#可行解列表
    if D >= -1e-9:#判别式决定方程是否有实数解：D ≥ 0 有解，D < 0 无解。
        D = max(D, 0.0)#数值稳定性处理，避免浮点误差导致的负数开根号
        for sign in (+1, -1):#正负根
            vmax = (-B + sign*math.sqrt(D)) / (2*A)#求解二次方程的根
            if not np.isfinite(vmax): continue#.isfinite(vmax)：检查 vmax 是否为有限数值（非 NaN、非正负无穷大）
                                                #若非有限数值，跳过该解
            if vmax < v_init - 1e-9 or vmax > vmax_cap + 1e-9: continue# # 筛选条件2：vmax 需在 [v_init, vmax_cap] 范围内（物理意义：加速后速度不小于初始速度，不超上限）
            tc = t_var - (vmax - v_init)/ap - vmax/an#计算匀速时间 tc
            if tc >= -1e-6:#筛选条件3：tc ≥ 0（物理意义：匀速时间不能为负）
                candidates.append((float(vmax), max(0.0, float(tc))))#加入可行解列表，确保 tc 非负
    if not candidates:#无可行解，尝试三角型退化情况
        # 三角型（tc=0）退化检验
        vmax = (t_var + v_init/ap) / (1.0/ap + 1.0/an)#计算退化情况下的 vmax
        if v_init - 1e-9 <= vmax <= vmax_cap + 1e-9:
            s_tri = (vmax*vmax - v_init*v_init)/(2.0*ap) + (vmax*vmax)/(2.0*an)
            if abs(s_tri - s_var) <= max(1e-3, 1e-5*s_var):
                return float(vmax), 0.0
        return None, None
    # 取较小的一个 vmax（都可行）
    candidates.sort(key=lambda x: x[0])
    return candidates[0]

def build_tail_profile(s0, t0, v0, vmax, tc, A_POS, A_NEG, L_target):#构建后段速度剖面
    ap = A_POS; an = abs(A_NEG)
    ta = max(0.0, (vmax - v0)/ap)#加速段时间
    td = max(0.0, vmax/an)#减速段时间

    # 加速段
    t1 = np.arange(0.0, ta + EPS_T, DT)#生成加速段的时间数组，从0到ta，步长为DT
    v1 = v0 + ap*t1#计算加速段的速度数组，使用匀加速公式 v = v0 + a*t
    s1 = v0*t1 + 0.5*ap*t1*t1#计算加速段的位移数组，使用匀加速位移公式 s = v0*t + 0.5*a*t^2

    # 匀速段
    if tc > DT*0.5:#若匀速时间 tc 大于半个时间步长，认为 “需要匀速段”
        t2 = np.arange(DT, tc + EPS_T, DT)#生成匀速段的时间数组，从DT到tc，步长为DT
        v2 = np.full_like(t2, vmax)#匀速段速度恒为 vmax，.full_like 创建与 t2 形状相同的数组，所有元素均为 vmax
        s2 = vmax*t2#计算匀速段的位移数组，s = vmax * t
    else:#否则，不需要匀速段
        t2 = np.array([])#空数组，.array([]) 创建一个空的 NumPy 数组
        v2 = np.array([])
        s2 = np.array([])

    # 减速段
    t3 = np.arange(DT, td + EPS_T, DT)#生成减速段的时间数组，从DT到td，步长为DT
    v3 = np.maximum(vmax - an*t3, 0.0)#计算减速段的速度数组，使用匀减速公式 v = vmax - a*t，并确保速度非负
    s3 = vmax*t3 - 0.5*an*t3*t3

    # 拼接（相对时间）
    t_rel = np.concatenate([t1, ta + t2, ta + tc + t3])#.concatenate：将各段时间数组拼接成完整的相对时间数组
                                                        #t1：加速段相对时间（0→ta）；
                                                        #ta + t2：匀速段相对时间（ta→ta+tc，叠加加速段总时间，确保时间连续）；
                                                        #ta + tc + t3：减速段相对时间（ta+tc→ta+tc+td，叠加加速 + 匀速总时间，确保时间连续）；
                                                        #结果：t_rel 是从 0 到 ta+tc+td 的连续时间序列。
    v_rel = np.concatenate([v1, v2, v3])#拼接各段速度数组
    s_rel = np.concatenate([s1,
                            (s1[-1:] + s2) if s2.size>0 else s1[-1:],
                            ( (s1[-1] + (s2[-1] if s2.size>0 else 0.0)) + s3 )])
    #拼接各段位移数组
    '''
    位移需 累加（后一段的位移基于前一段的终点位移）：
    s1：加速段相对位移（0→s1 [-1]）；
    s1[-1:] + s2：匀速段相对位移（从加速段终点 s1[-1] 开始累加，s1[-1:] 保持数组维度一致）；
    若 s2 为空（无匀速段），则直接取 s1[-1:]（保持序列长度一致）；
    (s1[-1] + (s2[-1] if s2.size>0 else 0.0)) + s3：减速段相对位移（从加速 + 匀速的总终点位移开始累加）；
    结果：s_rel 是尾部运动的总相对位移（从 0 到 s1[-1]+s2[-1]+s3[-1]）。
    '''

    # 转绝对
    t = t0 + t_rel
    s = s0 + s_rel
    a = np.zeros_like(t_rel)#初始化加速度数组，形状与 t_rel 相同，初始值全为0
    a[:t1.size] = ap#加速段加速度
    if t2.size>0: a[t1.size:t1.size+t2.size] = 0.0#匀速段加速度
    a[t1.size+t2.size:] = -an#减速段加速度

    # 尾点对齐
    s[-1] = L_target#确保最终位移对齐目标位置
    v_rel[-1] = 0.0#确保最终速度为0
    a[-1] = 0.0#确保最终加速度为0
    return {'t': t, 'v': v_rel, 'a': a, 's': s, 'T': t[-1]}

# =============== 特征构造 & 能耗评估 ===============build_features是 “整理素材”，eval_energy_wh是 “用模型算结果”，两者配合完成从 “列车运行数据” 到 “能耗评估” 的转化。
def build_features(prof, feature_cols, curve_f, grade_f, mass_f):
    t = prof['t']; v = prof['v']; a = prof['a']; s = prof['s']#运行剖面字典中提取 4 个核心动态序列
    prev_v = np.roll(v, 1); prev_a = np.roll(a, 1)#构造前一时刻速度和加速度序列，使用 np.roll 将 v 和 a 数组向后滚动一位
                                                            #例如 v = [v0, v1, v2, v3] → np.roll(v,1) = [v3, v0, v1, v2]；
                                                    #np.roll(arr, shift)：将数组 arr 向后滚动 shift 位，
    prev_v[0] = v[0]; prev_a[0] = a[0]#处理边界条件，确保首元素与原数组一致（避免滚动引入错误值）
    curv = curve_f(s); grad = grade_f(s); mass = mass_f(s)#通过插值函数，获取每个位移对应的静态特征（与时间步同步
                                                        #curve_f(s)：计算曲率数组
                                                        #grade_f(s)：计算坡度数组
                                                        #mass_f(s)：计算质量数组

    # 防nan
    curv = np.nan_to_num(curv); grad = np.nan_to_num(grad); mass = np.nan_to_num(mass)#将 NaN 值替换为0，确保特征数组中无 NaN
                                                        #np.nan_to_num(arr)：将数组 arr 中的 NaN 替换为0，正无穷替换为最大有限值，负无穷替换为最小有限值

    bank = {
        'time': t, 'velocity': v, 'acceleration': a,
        'prev_velocity': prev_v, 'prev_acceleration': prev_a,
        'curvature': curv, 'gradient': grad, 'mass': mass,
    }#将所有特征（动态特征 + 历史特征 + 静态特征）汇总到字典中，按 feature_cols 顺序快速提取，避免特征顺序错乱；
    X = np.stack([np.asarray(bank.get(c, np.zeros_like(t)), dtype=float) for c in feature_cols], axis=1)#构建特征矩阵 X
                                        #np.stack([...], axis=1)：将多个一维数组沿列方向堆叠成二维数组（矩阵）
                                        #将 M 个 (N,) 形状的特征序列，沿 axis=1（列方向）拼接为 (N, M) 形状的矩阵（每行 = 一个时间步，每列 = 一个特征）；
                                        #np.asarray(bank.get(c, np.zeros_like(t)), dtype=float)：从 bank 字典中获取特征列 c 的数组，
                                        # 若不存在则使用与 t 形状相同的全零数组，确保数据类型为浮点型 
    X = np.nan_to_num(X)    #确保特征矩阵中无 NaN 值，将 NaN 替换为0
    return X

def eval_energy_wh(net, scaler, X):
    Xs = scaler.transform(X)#使用数据标准化器 scaler 对特征矩阵 X 进行标准化处理，得到标准化后的特征矩阵 Xs，按训练时的标准化规则转换
                            #scaler.transform(X)：对输入特征矩阵 X 进行标准化，
    with torch.no_grad():#禁用梯度计算，提升推理效率，节省内存
        e_inc = net(torch.tensor(Xs, dtype=torch.float32)).cpu().numpy().reshape(-1)
                                                    #torch.tensor(Xs, dtype=torch.float32)：将标准化后的特征矩阵 Xs（numpy 数组）转为 PyTorch 张量
                                                    #.tensor(..., dtype=torch.float32)：指定张量的数据类型为 32 位浮点型
                                                    #net(...)：将特征张量输入神经网络模型 net，得到能耗增量张量
                                                    #.cpu()：将张量移动到 CPU 上（若在 GPU 上计算）
                                                    #.numpy()：将 PyTorch 张量转换为 NumPy 数组
                                                    #.reshape(-1)：将能耗增量数组展平为一维数组
    e_inc = np.nan_to_num(e_inc)#确保能耗增量数组中无 NaN 值，将 NaN 替换为0
    return float(np.sum(e_inc)), e_inc, np.cumsum(e_inc)#.cumsum(e_inc)：计算能耗增量的累计和，得到每个时间步的累计能耗

# =============== 主流程 ===============
def main():
    net, scaler, feature_cols = load_model_and_scaler(PAIR)#调用 load_model_and_scaler 加载训练好的能耗预测模型（net）、数据标准化器（scaler）、特征列名列表（feature_cols），用于后续能耗预测；
    curve_f, grade_f, mass_f = load_static_series(PAIR)#调用 load_static_series 加载路段静态特征的插值函数

    # 1) 计算固定前段
    prefix = fixed_prefix(A_POS=A_POS, A_NEG=A_NEG, v_to=10.667)#作用是生成 “加速→匀速” 的固定前段轨迹，参数 v_to=10.667 表示前段的目标匀速速度（约 38.4 km/h，符合列车起步阶段的速度约束）；
    T_fix, S_fix, v_init = prefix['T'], prefix['L'], prefix['v_end']#固定前段的总时间 T_fix，总位移 S_fix，结束速度 v_init；
    print(f"[固定段] T_fix={T_fix:.3f}s, S_fix={S_fix:.3f}m, v_end={v_init:.3f}m/s")#打印固定前段的时间、位移和结束速度，便于调试和验证；

    # 2) 扫描总到达时间，解析求 vmax 并评估能耗
    rows = []#结果列表
    t_list = np.arange(T_MIN, T_MAX + 1e-12, T_STEP)#生成总到达时间的扫描列表，从 T_MIN 到 T_MAX，步长为 T_STEP；
                                                    #np.arange(T_MIN, T_MAX + 1e-12, T_STEP)：生成一个从 T_MIN 到 T_MAX（包含 T_MAX）的等差数列，步长为 T_STEP；
                                                    #1e-12 用于确保包含 T_MAX，避免浮点数精度问题导致遗漏；
    for T_tot in t_list:#遍历每个总到达时间 T_tot；
        t_var = T_tot - T_fix#计算可变时间段 t_var，即总时间减去固定前段时间；
        s_var = L_TOTAL - S_fix#计算可变位移段 s_var，即总位移减去固定前段位移；
        if t_var <= 0 or s_var <= 0:#若可变时间或位移为负，跳过该时间点；
            continue
        vmax, tc = solve_vmax_for_time_distance(s_var, t_var, v_init, A_POS, A_NEG, V_MAX_UPPER)#调用 solve_vmax_for_time_distance 求解可变段的峰值速度 vmax 和匀速时间 tc；
                                                                #参数：可变位移 s_var，可变时间 t_var，初始速度 v_init，加速度 A_POS，减速度 A_NEG，峰值速度上限 V_MAX_UPPER；
        if vmax is None:
            continue
        # 拼完整 v-t
        tail = build_tail_profile(S_fix, T_fix, v_init, vmax, tc, A_POS, A_NEG, L_TOTAL)#调用 build_tail_profile 生成尾部轨迹剖面，
                                                                #参数：起始位移 S_fix，起始时间 T_fix，起始速度 v_init，峰值速度 vmax，匀速时间 tc，加速度 A_POS，减速度 A_NEG，总位移 L_TOTAL；
        t = np.concatenate([prefix['t'], tail['t'][1:]])#拼接完整时间序列，去掉尾部剖面的第一个时间点以避免重复；
        v = np.concatenate([prefix['v'], tail['v'][1:]])#拼接完整速度序列；
        a = np.concatenate([prefix['a'], tail['a'][1:]])
        s = np.concatenate([prefix['s'], tail['s'][1:]])
        prof = {'t': t, 'v': v, 'a': a, 's': s, 'T': t[-1]}
        # 特征 & 能耗
        X = build_features(prof, feature_cols, curve_f, grade_f, mass_f)#构建模型输入特征矩阵 X（整合动态特征 + 静态特征）；
        E_wh, _, _ = eval_energy_wh(net, scaler, X)#预测总能耗 E_wh（忽略瞬时能耗和累计能耗，仅保留总能耗）
        rows.append([T_tot, vmax, E_wh])#将结果添加到列表，包含总到达时间 T_tot，峰值速度 vmax，能耗 E_wh；

    if not rows:#若无可行解，抛出异常提示用户调整参数；
        raise RuntimeError("时间扫描无可行点，请调整 T_MIN/T_MAX 或 V_MAX_UPPER。")

    df_time = pd.DataFrame(rows, columns=['t_arrival','v_peak_opt','E_wh'])#将结果列表转换为 Pandas DataFrame，
                                                    #指定列名为 't_arrival'（到达时间）、'v_peak_opt'（峰值速度）、'E_wh'（能耗）；
    df_time.sort_values('t_arrival', inplace=True)#按到达时间排序，便于后续分析和可视化；
                                                        #inplace=True：直接在原 DataFrame 上修改，而非返回新对象；
                                                        #.sort_values(...)：按指定列排序 DataFrame；
    df_time.to_csv(OUT_DIR / f"time_energy_curve_{PAIR}.csv", index=False)#保存结果为 CSV 文件，文件名包含站点对 PAIR，方便识别和存档；

    plt.figure(figsize=(7,4))#绘制时间-能耗曲线图，设置图形大小为 7x4 英寸；.figure(...)：创建一个新的图形窗口，figsize=(7,4) 指定图形的宽度为 7 英寸，高度为 4 英寸；
    plt.plot(df_time['t_arrival'], df_time['E_wh'], lw=1.2)#.plot(...)：绘制折线图，横坐标为到达时间，纵坐标为能耗，线宽为 1.2；
    plt.xlabel('Arrival Time (s)'); plt.ylabel('Energy (Wh)')#设置横轴和纵轴标签；
    plt.title(f'Time vs Min Energy — {PAIR}')#设置图形标题，包含站点对 PAIR；
    plt.grid(True, alpha=0.4); plt.tight_layout()#启用网格线，调整布局以防止标签重叠；
                                            #.tight_layout()：自动调整子图参数，使图形布局更紧凑，防止标签和标题重叠；
    plt.savefig(OUT_DIR / "time_energy_curve.png", dpi=300); plt.close()#保存图形为 PNG 文件，分辨率为 300 DPI，随后关闭图形窗口释放资源；
                                                                            #dpi=300：指定图像分辨率为每英寸 300 点，适合打印和高质量显示；

    # 3) 选能耗最低的一个，输出 v-t 与能耗累计曲线
    idx = int(np.argmin(df_time['E_wh'].values))#找到能耗最低的索引位置；.argmin(...)：返回数组中最小值的索引；
    T_best = float(df_time.iloc[idx]['t_arrival']); vmax_best = float(df_time.iloc[idx]['v_peak_opt'])#根据索引提取最佳到达时间和对应的峰值速度；
                                        #.iloc[idx]：按位置索引访问 DataFrame 的行；df_time.iloc[idx]['t_arrival']：获取该行的到达时间；
    t_var = T_best - T_fix; s_var = L_TOTAL - S_fix#计算最佳解的可变时间和位移段；
    vmax, tc = solve_vmax_for_time_distance(s_var, t_var, v_init, A_POS, A_NEG, V_MAX_UPPER)#再次求解最佳解的峰值速度和匀速时间，确保数值一致；
    tail = build_tail_profile(S_fix, T_fix, v_init, vmax, tc, A_POS, A_NEG, L_TOTAL)#构建最佳解的尾部轨迹剖面；
    t = np.concatenate([prefix['t'], tail['t'][1:]])#拼接完整时间序列；
    v = np.concatenate([prefix['v'], tail['v'][1:]])
    a = np.concatenate([prefix['a'], tail['a'][1:]])
    s = np.concatenate([prefix['s'], tail['s'][1:]])
    prof = {'t': t, 'v': v, 'a': a, 's': s, 'T': t[-1]}

    # 保存 v-t
    pd.DataFrame({'time (s)': t, 'velocity (m/s)': v}).to_csv(OUT_DIR / "optimal_vt_curve.csv", index=False)#保存最佳解的速度-时间曲线为 CSV 文件，包含时间和速度两列；
    plt.figure(figsize=(7,4))
    plt.plot(t, v, lw=2)#绘制最佳解的速度-时间曲线图，线宽为 2；
    plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)')
    plt.title(f'Optimal v–t (vmax={vmax:.2f} m/s, T={T_best:.2f}s)')
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(OUT_DIR / "optimal_vt.png", dpi=300); plt.close()

    # 累计能耗
    X = build_features(prof, feature_cols, curve_f, grade_f, mass_f)#构建最佳解的特征矩阵 X；
    E_wh, e_inc, e_cum = eval_energy_wh(net, scaler, X)
    plt.figure(figsize=(7,4))
    plt.step(t, e_cum, where='post', lw=2)
    plt.xlabel('Time (s)'); plt.ylabel('Cumulative Energy (Wh)')
    plt.title('Optimal E–t'); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(OUT_DIR / "optimal_energy.png", dpi=300); plt.close()

    # 摘要
    pd.DataFrame([{
        'station_pair': PAIR,
        'v_peak_opt(m/s)': vmax,
        't_arrival_opt(s)': T_best,
        'E_opt(Wh)': E_wh,
        'T_fixed(s)': T_fix, 'S_fixed(m)': S_fix, 'v_init(m/s)': v_init,
        'A_POS': A_POS, 'A_NEG': A_NEG,
        'V_MAX_UPPER': V_MAX_UPPER,
        'window': f'[{T_MIN},{T_MAX}] step {T_STEP}',
        'note': 'fixed prefix then single-variable vmax tail (accelerate→cruise→brake to 0)'
    }]).to_csv(OUT_DIR / "best_solution.csv", index=False)

    print(f"✅ 完成！输出目录：{OUT_DIR.resolve()}")
    print("   - time_energy_curve_梅堰-永茂路.csv")
    print("   - time_energy_curve.png")
    print("   - optimal_vt_curve.csv")
    print("   - optimal_vt.png")
    print("   - optimal_energy.png")
    print("   - best_solution.csv")

if __name__ == "__main__":
    main()
