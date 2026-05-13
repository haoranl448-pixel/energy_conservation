#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
schedule_optimizer.py
现在 T_TOTAL 是“全程总时间（含停站）”。程序会读取（或默认）各站停站时间求和，
得到纯运行时间 T_RUN = T_TOTAL - sum(dwell_times)，在此运行时间约束下做 DP 最小能耗分配。
随后重建每段 v–t，并拼接全线大 v–t 表和图（图里包含停站）。
另外：进行运行时间敏感性分析（将 T_RUN 依次减少 1~5 秒），输出能耗变化情况到 CSV。
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== 基本配置 =====================
T_TOTAL = 3684 # 全程总时长（包含停站）
# 绘图配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# STATIONS = [
#     "布政-张家潭","张家潭-同德路","同德路-石碶","石碶-雅渡","雅渡-庙堰"
# ]
STATIONS = [
    "布政-张家潭","张家潭-同德路","同德路-石碶","石碶-雅渡","雅渡-庙堰",
    "庙堰-钟公庙","钟公庙-鄞州区政府","鄞州区政府-钱湖南路","钱湖南路-南高教园区",
    "南高教园区-下应路","下应路-大洋江","大洋江-泗港","泗港-曹隘","曹隘-柳隘",
    "柳隘-海晏北路","海晏北路-民安东路","民安东路-会展中心","会展中心-院士路",
    "院士路-盎孟港","盎孟港-三官堂","三官堂-兴庄路","兴庄路-兴海南路",
    "兴海南路-梅堰","梅堰-永茂路","永茂路-镇海大道","镇海大道-骆驼桥"
]
SPECIAL_PAIR1 = "梅堰-永茂路"
SPECIAL_PAIR2 = "泗港-曹隘"

# OPT_DIR = Path("opt_results")
# PARAMS_CSV = Path("section_params.csv")
# DWELL_CSV  = Path("dwell_times.csv")  # 可选：station_pair,dwell_time（只列出特殊站则其他默认30s）
# DEFAULT_DWELL = 30.0

# OUT_DIR     = Path("schedule_results"); OUT_DIR.mkdir(exist_ok=True)
# SEG_OUT_DIR = OUT_DIR / "segments";      SEG_OUT_DIR.mkdir(exist_ok=True)


# OPT_DIR = Path("output/optimization/opt_results_residual")
OPT_DIR = Path("output/optimization/trip5_v2_6feat")
# PARAMS_CSV = Path("data/static/section_params.csv")
PARAMS_CSV = Path("data/static/section_params_trip5.csv")
DWELL_CSV  = Path("data/static/dwell_times.csv")  # 可选：station_pair,dwell_time（只列出特殊站则其他默认30s）
DEFAULT_DWELL = 30.0

OUT_DIR     = Path("output/schedule/schedule_results_residual1"); OUT_DIR.mkdir(exist_ok=True)
SEG_OUT_DIR = OUT_DIR / "segments";      SEG_OUT_DIR.mkdir(exist_ok=True)


DT = 0.05

# ===================== 停站时间 =====================
def load_dwell_times():
    """
    读取停站时间：dwell_times.csv 可以只覆盖“特殊站对”的停站时间；
    未出现在文件中的站对默认 30s。
    返回与 STATIONS 对应的一一列表。
    """
    if DWELL_CSV.exists():#若 dwell_times.csv 文件存在
        df = pd.read_csv(DWELL_CSV)# 用 pandas 读取 CSV 文件，生成 DataFrame（表格型数据结构）
        rec = {str(r["station_pair"]).strip(): 
               float(r["dwell_time"]) 
               for _, r in df.iterrows()}#用字典推导式将 DataFrame 转换为键值对字典 rec：键（key）：str(r["station_pair"]).strip() → 
                                         #区间名称（转字符串 + 去除前后空格，避免因文件中多余空格导致匹配失败）。
                                         #值（value）：float(r["dwell_time"]) → 停站时间
                                         #df.iterrows()：逐行遍历 DataFrame，每次返回一个元组 (行索引, 行数据)；

        #上面的等价写法：
        '''
        # 初始化一个空字典，用来存“路段标识→停站时间”
        rec = {}

            # 遍历 DataFrame 的每一行
        for _, row in df.iterrows():
        # 1. 处理“路段标识”（作为字典的键）
        # 取当前行的路段名 → 转成字符串 → 去掉前后空格
            station_pair = str(row["station_pair"]).strip()
        # 2. 处理“停站时间”（作为字典的值）
        # 取当前行的停站时间 → 转成浮点数（方便计算）
            dwell_time = float(row["dwell_time"])
        # 3. 把键值对存入字典
            rec[station_pair] = dwell_time
        
        '''
        return [rec.get(sp, DEFAULT_DWELL) for sp in STATIONS]#rec.get(sp, DEFAULT_DWELL)：查找字典中是否有 sp（当前区间）对应的键值对：
                                                                #若有：返回字典中存储的自定义停站时间（如 “梅堰 - 永茂路” 返回 45.0）
                                                                #若没有：返回默认值 DEFAULT_DWELL（30 秒，如 “布政 - 张家潭” 未在 CSV 中定义，返回 30.0）


        '''
        # 初始化空列表，存储每个路段的停站时间
        dwell_list = []
        # 遍历所有路段标识
        for sp in STATIONS:
        # 从字典查停站时间，找不到就用默认值 DEFAULT_DWELL
            dwell_time = rec.get(sp, DEFAULT_DWELL) #get 方法：查找字典中是否有 sp（当前区间）对应的键值对
        # 加入列表
            well_list.append(dwell_time)
        # 返回结果列表
            return dwell_list
        '''

    return [DEFAULT_DWELL] * len(STATIONS)#若 dwell_times.csv 不存在，直接生成一个长度等于 STATIONS 的列表，每个元素均为默认停站时间（30 秒）。

# ===================== 曲线加载 =====================
def load_curve_for_segment(sp: str):#sp：station pair，表示站对名称，如 "布政-张家潭",
    #":str"明确输入参数 sp 必须是字符串
    f = OPT_DIR / sp / f"time_energy_curve_{sp}.csv"#opt_results下的路径 
    if not f.exists():
        raise FileNotFoundError(f"[{sp}] 缺少 {f}")#不存在就报错
    df = pd.read_csv(f)#用 pandas 读取 CSV 文件，生成 DataFrame（表格型数据结构）
    for c in ("t_arrival","v_peak_opt","E_wh"):#遍历 3 个必须存在的核心列：t_arrival：区间运行时间（从区间起点到终点的纯运行时间，不含停站）；
                                                #v_peak_opt：该运行时间下的最优峰值速度（DP 优化时的候选速度）；
                                                #E_wh：该运行时间 + 峰值速度对应的能耗（优化目标是最小化总能耗）。
        if c not in df.columns:
            raise ValueError(f"[{sp}] {f} 缺少列 {c}")#不存在就报错
    df = df.replace([np.inf,-np.inf], np.nan).dropna()
    #处理 3 类无效数据：
    #replace([np.inf,-np.inf], np.nan)：将无穷大（inf）和负无穷大（-inf）替换为缺失值（np.nan）—— 这类值通常是计算错误导致，无法用于优化；
    #dropna()：删除包含缺失值（np.nan）的行 —— 确保剩余数据每行都是 “运行时间 + 峰值速度 + 能耗” 完整的有效方案。
    return (df["t_arrival"].to_numpy(float),
            df["v_peak_opt"].to_numpy(float),
            df["E_wh"].to_numpy(float))#将 DataFrame 的 3 列核心数据分别转换为 numpy 数组，
                                        #返回格式：元组（t_arrival数组, v_peak_opt数组, E_wh数组）

# ===================== DP（仅约束“运行时间”） =====================动态规划
def dp_optimize(curves, T_total_run, slack=0.5):
    """
    curves: 每段的 (t_arrival[], v_peak_opt[], E_wh[]) 候选三元组
    T_total_run: 纯“运行时间”的总时长目标
    slack: 容差（秒），允许 T_used ∈ [T_total_run±slack]
    """
    to_int = lambda t: int(round(float(t)*100))#时间*100并四舍五入取整，转为整数表示（单位：百分之一秒）lambda t:==def to_int(t):
    T_total_int = to_int(T_total_run); slack_int = to_int(slack)#同样地方式转换目标运行时间和容差为整数表示

    # 预处理每段的离散点（去重保留每个 t_int 能耗最小的那个）
    packed = []#存储每段处理后的候选点列表


    #对每个路段的候选方案，按时间去重，仅保留同一时间下能耗最小的方案
    for ts,vs,Es in curves:
        df = pd.DataFrame({"t":ts,"v":vs,"E":Es}).replace([np.inf,-np.inf],np.nan).dropna()#创建 DataFrame，并去除无效数据 （同前述 load_curve_for_segment 函数）
                                                                                            #dropna()：删除包含缺失值（np.nan）的行 
                                                                                        #创建 DataFrame 时，表格型数据结构，使用字典 {"t":ts,"v":vs,"E":Es} 指定列名和对应数据
        df["t_int"] = (df["t"]*100).round().astype(int)#    将运行时间转换为整数表示（单位：百分之一秒），存储在新列 t_int 中
         #对 DataFrame 按 t_int 和 E 进行排序，确保每个 t_int 只保留能耗最小的那个方案  astype(int)：转为整数类型（避免浮点数索引）
        df = df.sort_values(["t_int","E"]).drop_duplicates("t_int", keep="first")#sort_values(["t_int", "E"])（DataFrame 方法） 先按 t_int（离散时间）升序排序，再按 E（能耗）升序排序。
        #drop_duplicates("t_int", keep="first")（DataFrame 方法） 删除重复的 t_int 行，只保留每个 t_int 对应的第一行（即能耗最小的方案）。
        # subset="t_int"：按哪一列去重；keep="first"：保留重复行中的第一行（因已按 E 排序，第一行能耗最小）。
        packed.append((df["t_int"].to_numpy(), df["v"].to_numpy(), df["E"].to_numpy()))#to_numpy()（Series 方法）作用：将 DataFrame 的列转换为 numpy 数组 并存储到 packed 列表中 ，
                                                                                        #每个元素是一个元组（t_int数组, v数组, E数组） 

    # DP
    dp = {0:0.0}       # t_sum_int -> min energy
    #DP状态初始化：key=累计运行时间（离散整数），value=该累计时间下的最小总能耗 
    choice=[{}]        # 回溯路径
    #回溯路径存储：choice[i]对应第i个区间的决策（key=当前累计时间，value=前一状态+当前选择）  


    for idx,(tints,vfor,Efor) in enumerate(packed,1):#enumerate(packed,1)：遍历每个区间的候选点列表，idx表示区间索引（从1开始），(tints,vfor,Efor)分别是当前区间的运行时间、峰值速度和能耗数组 
        #原型
        '''
        for i in range(len(packed)):
            idx = i + 1  # 从1开始的索引
            tints, vfor, Efor = packed[i] 
        '''
        
        
        nxt={}; ch={}#nxt：字典，存储当前区间处理后的新状态（key=累计时间，value=最小能耗）  ch：存储当前区间的决策路径（key=累计时间，value=前一状态+当前选择）
        for t_prev,e_prev in dp.items():#遍历上一状态的所有累计时间和对应最小能耗
            for ti,vi,ei in zip(tints,vfor,Efor):#zip(tints, vfor, Efor)（Python 内置函数），
                                                #将当前区间的「离散时间数组、速度数组、能耗数组」按索引配对，每次循环返回一组（ti, vi, ei）（即一个候选方案）。
                t_sum = t_prev + int(ti)              # 只累计运行时间
                if t_sum > T_total_int + slack_int:   # 剪枝：累计时间超过「总目标+容差」，直接跳过（不可能是有效解）
                    continue
                e_sum = e_prev + float(ei)#计算当前决策后的总能耗（前一总能耗 + 当前区间能耗）
                if t_sum not in nxt or e_sum < nxt[t_sum]:#检查当前累计时间 t_sum 是否已存在于 nxt 中，或者，当前总能耗 e_sum 是否小于已存储的最小能耗
                    nxt[t_sum]=e_sum #更新 nxt 中该累计时间的最小能耗
                    ch[t_sum]=(t_prev,int(ti),float(vi),float(ei))#记录决策路径：当前累计时间 t_sum 对应的前一状态（t_prev）和当前选择（ti, vi, ei）
        if not nxt:#若当前区间处理后没有任何可行状态，说明无法满足时间约束，抛出异常
            raise RuntimeError(f"第{idx}段无可行状态，请检查该段时间窗口")
        dp=nxt; choice.append(ch)#更新 dp 为当前区间处理后的新状态 nxt，并将当前区间的决策路径 ch 添加到 choice 列表中

    # 在允许容差内挑能耗最小的
    cands = [(t,e) for t,e in dp.items()#items()：获取字典中所有的「键-值对」；
             if T_total_int - slack_int <= t <= T_total_int + slack_int]#列表推导式 [(t, e) for t, e in dp.items() if ...]，
                                                                            #遍历 DP 最终状态字典 dp，筛选出「累计时间在容差范围内」的状态，存入 cands（候选最优解列表）。
    '''
    cands = []
            # 遍历dp字典的键值对（t=累计离散时间，e=最小能耗）
    for t, e in dp.items():
                 # 判断是否在容差范围内
    lower_bound = T_total_int - slack_int  # 容差下限
    upper_bound = T_total_int + slack_int  # 容差上限
    if lower_bound <= t <= upper_bound:
        cands.append((t, e))
    '''
    
    
    
    if not cands:#若没有任何候选解，说明无法在给定时间范围内完成运行，抛出异常
        tmin=min(dp.keys())/100.0; tmax=max(dp.keys())/100.0                #计算可达的最小和最大运行时间（转换回秒）
                                                                            #dp.keys() 是 字典（dictionary）的内置方法，作用是 获取字典中所有的「键（key）」
                                                                                    #min(dp.keys())/max(dp.keys())（Python 内置函数）
                                                                                    # 获取 DP 最终状态中「最小累计离散时间」和「最大累计离散时间」，用于报错时提示用户实际可达的时间范围。
        raise RuntimeError(f"无法在 {T_total_run:.1f}s±{slack:.1f}s 内分配，"#T_total_run:.1f：格式化为浮点数，保留1位小数
                           f"可达运行时长约 [{tmin:.2f},{tmax:.2f}]s")#tmin:.2f/tmax:.2f：格式化为浮点数，保留2位小数
    tgt_t_int, tgt_E = min(cands, key=lambda x:x[1])#从候选解列表 cands 中选择能耗最小的解，tgt_t_int 是对应的累计离散时间，tgt_E 是最小总能耗,
                                                                #key=lambda x:x[1]：指定按元组的第二个元素（能耗）进行比较，选择能耗最小的元组。lambda x:x[1]==def key_function(x): return x[1]
    
    #原型
    '''
    min_energy = float('inf')
    tgt_t_int = 0

    # 遍历所有候选解，找到能耗最小的那个
    for t, e in cands:
    if e < min_energy:
        min_energy = e  # 更新最小能耗
        tgt_t_int = t   # 记录对应的时间

    tgt_E = min_energy  # 最小能耗赋值给tgt_E
    '''
    # 回溯
    alloc=[]; t=tgt_t_int
    for i in range(len(packed),0,-1):   #从最后一个区间开始回溯 len(packed) 是区间总数，0 是结束值（不包含），-1 是步长（倒序）
        t_prev,ti,vi,ei = choice[i][t]  #从 choice 中获取当前区间 i 在累计时间 t 下的决策信息
        alloc.append((ti/100.0,vi,ei))  #将当前区间的决策（运行时间、峰值速度、能耗）添加到分配列表 alloc 中，ti/100.0 转换回秒
        t=t_prev    #更新累计时间 t 为前一状态的累计时间 t_prev，继续回溯到上一个区间
    alloc.reverse() #回溯过程中分配是倒序的，最后需要反转列表 alloc，使其按区间顺序排列
    return alloc, float(tgt_E), tgt_t_int/100.0 #返回分配结果 alloc、最小总能耗 tgt_E 和实际使用的总运行时间（秒）

# ===================== 参数表（用于重建 v-t） =====================
def load_params():  #返回字典：键=station_pair，值=参数字典
    df = pd.read_csv(PARAMS_CSV)#路径为"section_params.csv" 的 CSV 文件，包含每个区间的物理参数
    rec={}  #存储参数的字典
    for _,r in df.iterrows():   #df.iterrows()：作用：逐行遍历 DataFrame，每次返回一个元组 (行索引, 行数据)；   
                                #行索引：表示当前行在表格中的位置（如 0、1、2...），此处用 _ 接收（_ 是 Python 中常用的 “占位符变量”，表示该值无需使用）；
        sp=str(r["station_pair"]).strip()   #r["station_pair"]：提取当前行的「区间名称」，转换为字符串并去除前后空白字符
                                        #.strip()：去除字符串前后的空白字符（空格、换行符、制表符等），
        rec[sp]=dict(L=float(r["L"]),V_UPPER=float(r["V_UPPER"]),#创建外层字典 rec，键是区间名称 sp，值是对应的参数字典
                     V_MID=float(r["V_MID"]),A1=float(r["A1"]),
                     A2=float(r["A2"]),A_DEC=float(r["A_DEC"])) #dict(...)：创建内层参数字典，键是参数名（L、V_UPPER 等），值是对应的参数值。
    return rec  #返回包含所有区间参数的字典 rec，键是区间名称，值是对应的参数字典

# ===================== 普通段：梯形 v-t =====================
def make_profile_trapezoid(t_arrival, v_peak, A1, A2, A_DEC, V_MID):#trapezoid：梯形，
    v_peak=float(v_peak)    #确保峰值速度为浮点数（避免整数类型导致计算精度问题）
    t1 = min(v_peak,V_MID)/A1   #第一段加速的末速度是「v_peak 和 V_MID 中的较小值」
    t2 = max(0.0,(v_peak-V_MID)/A2) #只有当 v_peak > V_MID 时，才需要第二段加速（否则 t2=0）；
    t_dec = v_peak/abs(A_DEC)   #从 v_peak 减速到 0 的时间（A_DEC 是负值，取绝对值避免时长为负）；
    t_cruise = max(0.0, t_arrival-(t1+t2+t_dec))#cruise ：巡航，：巡航时间是 “调节变量”—— 加速 + 减速的时间是固定的（由物理约束决定），通过增减巡航时间，确保总时长等于 t_arrival；
    T = t1+t2+t_cruise+t_dec#计算实际总运行时间 T（理论上等于t_arrival，兼容浮点精度误差）
    t = np.arange(0.0, T+1e-12, DT)#生成连续的时间点（如 0, 0.1, 0.2, ..., T），用于逐点计算速度；，DT 是时间步长（定义为 0.05 秒）
                                    #+1e-12 是工程技巧：避免因浮点精度误差（如 T=120.0，np.arange 可能只到 119.9）漏算最后一个时间点；
                                    #arange(start, stop, step)：生成一个从 start 到 stop（不包含 stop），步长为 step 的等差数列数组。
    v = np.zeros_like(t)    #作用：创建与时间序列长度相同的速度数组，初始值全 0（后续逐点更新）；
                            #np.zeros_like(t) 表示 “生成和 t 形状、类型完全一致的全 0 数组”
                            #zeros_like(array)：生成一个与给定数组形状相同的全 0 数组。
    for i,tt in enumerate(t):#enumerate (t)：遍历时间序列 t，
                            #i 是索引（0, 1, 2...），tt 是对应的时间点值（0.0, 0.05, 0.1...）
        if tt < t1: v[i]=A1*tt#第一段加速阶段，速度随时间线性增加
        elif tt < t1+t2: v[i]=V_MID + A2*(tt-t1)#第二段加速阶段，速度从 V_MID 增加到 v_peak
        elif tt < t1+t2+t_cruise: v[i]=v_peak#巡航阶段，速度保持恒定为 v_peak
        else:#减速阶段，速度线性减少到 0
            td=tt-(t1+t2+t_cruise)
            v[i]=max(v_peak + A_DEC*td, 0.0)
    s=np.cumsum(v*DT)#cumsum (v*DT)：计算速度积分，得到位置序列 s（每个时间点的累计位移）
                        #位移增量 = 速度 × 时间步长 DT；
    return {"t":t,"v":v,"s":s,"T":T}#返回包含时间序列 t、速度序列 v、位置序列 s 和实际总运行时间 T 的字典

# ===================== 特殊段：稳健模拟 + 二分匹配时长 (梅堰-永茂路)=====================
def simulate_meiyan_yongmaolu(vmax):
    L = 3734.0  #区间长度（米）
    A_POS, A_NEG = 0.60, -0.55  #加速度和减速度（米/秒²）
    V1,V2,V3,V4,V6 = 16.0, 10.667, 13.33, 10.667, 6.67  #各段目标速度（米/秒）
    V_MAX_UPPER = 22.22 #最高速度限制（米/秒）
    vmax = float(np.clip(vmax, 0.0, V_MAX_UPPER)) #np.clip 是 numpy 的 “裁剪函数”，作用是将 vmax 限制在 [0.0, V_MAX_UPPER] 范围内 —— 
                                                    #若输入 vmax=25（超上限），则裁剪为 22.22；若输入 vmax=-5（无效值），则裁剪为 0.0，避免超速或无效速度。

                        #      9.96   10.58  10.91    10.5  12.94   6.5          加速    减速    加速    减速
    S_BOUNDS = np.array([0.0, 471.0, 821.0, 1117.0, 1386.0, 3326.0, 3676.0, L])#各段位置边界（米）
    targets = [
        (S_BOUNDS[0], S_BOUNDS[1], V1),# 0m-471m：目标速度 V1
        (S_BOUNDS[1], S_BOUNDS[2], V2),# 471m-821m：目标速度 V2
        (S_BOUNDS[2], S_BOUNDS[3], V3),# 821m-1117m：目标速度 V3
        (S_BOUNDS[3], S_BOUNDS[4], V4),# 1117m-1386m：目标速度 V4
        (S_BOUNDS[4], S_BOUNDS[5], vmax),# 1386m-3326m：目标速度 vmax（调节变量）
    ]#告诉程序 “列车在哪个位移区间内，要跑到哪个速度”。

    def accel_to_target(v_cur, v_tar):# # 辅助函数：输入当前速度v_cur、目标速度v_tar，返回应施加的加速度
        if v_cur < v_tar - 1e-9: return A_POS
        if v_cur > v_tar + 1e-9: return A_NEG
        return 0.0

    t_list=[0.0]; v_list=[0.0]; s_list=[0.0]#用列表存储，时间、速度、位置序列
    seg_idx=0; cur_tar=targets[seg_idx][2]#状态跟踪：seg_idx=当前路段索引（初始为0→第1段），cur_tar=当前路段目标速度（初始为V1=16.0m/s）
                                            #cur_tar当前目标速度

    while True:## 无限循环，直到满足终止条件才退出
        t,v,s = t_list[-1], v_list[-1], s_list[-1]# 获取上一个时间步的状态（最后一个元素） 
        if s >= L - 1e-8 and v <= 1e-6: 
            break # 终止条件：当位置 s 达到或超过区间长度 L（考虑浮点误差）且速度 v 接近 0 时，退出循环

        while seg_idx < len(targets) and s >= targets[seg_idx][1] - 1e-12:#列车运行的最终目标是 “跑完全程（s≈L）且停稳（v≈0）”，
            seg_idx += 1
            if seg_idx < len(targets): 
                cur_tar = targets[seg_idx][2]#更新当前路段目标速度为下一个路段的目标速度

        if s >= S_BOUNDS[5] - 1e-12:
            # 3326m 之后：先追向 6.67，再根据刹停距离判断是否制动到 0
            if v > V6 + 1e-9: 
                a=A_NEG#如果当前速度 v 大于 V6（6.67 m/s）加上一个微小容差（1e-9），则设置加速度 a 为减速度 A_NEG（-0.55 m/s²），表示需要减速
            elif v < V6 - 1e-9: 
                a=A_POS#如果当前速度 v 小于 V6 减去微小容差，则设置加速度 a 为加速度 A_POS（0.60 m/s²），表示需要加速
            else: 
                a=0.0
            v_const = max(v, V6)#计算当前速度 v 和 V6（6.67 m/s）中的较大值，存储在 v_const 中
            d_stop = (v_const**2)/(2.0*abs(A_NEG))#计算从速度 v_const 减速到 0 所需的刹停距离 d_stop，使用公式 d = v² / (2*a)，其中 a 是减速度的绝对值
            if abs(v - V6) <= 1e-6 and (L - s) <= d_stop + 1e-6:#检查当前速度 v 是否接近 V6（允许微小误差 1e-6），且剩余距离（L - s）是否小于等于刹停距离 d_stop（允许微小误差 1e-6）
                a = A_NEG#如果条件满足，说明列车已经接近目标速度 V6 且距离终点足够近，可以开始减速停车，因此将加速度 a 设置为减速度 A_NEG（-0.55 m/s²）
        else:
            a = accel_to_target(v, cur_tar)#调用辅助函数，根据当前速度 v 和当前路段目标速度 cur_tar 计算所需的加速度 a

        v_next = max(v + a*DT, 0.0)#计算下一个时间步的速度 v_next，使用公式 v_next = v + a*DT，并确保速度不为负（取最大值与 0.0）
        s_next = s + v_next*DT#计算下一个时间步的位置 s_next，使用公式 s_next = s + v_next*DT
        t_next = t + DT#计算下一个时间步的时间 t_next，使用公式 t_next = t + DT

        # 末端截断到 L，并保证停稳
        if s < L and s_next > L:#检查当前位置 s 是否小于区间长度 L，且下一个位置 s_next 是否超过 L
            ds = L - s#计算剩余距离 ds，即从当前位置 s 到区间终点 L 的距离，不再用固定 DT，而是精准计算 “从当前位置到终点” 的状态，确保位移恰好为 L 且速度为 0
            if abs(a) < 1e-12:#情况1：加速度为0（匀速运动）
                dt_last = ds / max(v, 1e-9); #计算从当前位置到终点所需的时间 dt_last，使用公式 dt_last = ds / v，确保速度不为零（取最大值与 1e-9）
                v_end = v
            else:#情况2：有加速度（加速或减速）
                A = 0.5*a; B=v; C=-ds## 物理公式：ds = v*dt + 0.5*a*dt² → 整理为 A*dt² + B*dt + C = 0
                disc = B*B - 4*A*C#Δ = B² - 4AC 判别式
                 #计算从当前位置到终点所需的时间 dt_last，使用求根公式 dt = (-B ± √Δ) / (2A)
                dt_last = max(( -B + math.sqrt(max(disc,0.0)) )/(2*A), 0.0)
                v_end = max(v + a*dt_last, 0.0)#计算到达终点时的速度 v_end，使用公式 v_end = v + a*dt_last，并确保速度不为负
            t_next = t + dt_last; 
            s_next = L#更新下一个时间步的时间 t_next 和位置 s_next，确保位置恰好为 L
            if v_end > 1e-6:#如果到达终点时的速度 v_end 大于微小阈值（1e-6），说明列车还未完全停稳
                # 计算刹停时间并更新状态
                t_next += v_end/abs(A_NEG)
                v_next = 0.0
            else:#否则，列车已经停稳
                v_next = 0.0

        t_list.append(t_next); v_list.append(v_next); s_list.append(s_next)#将计算得到的下一个时间步的时间 t_next、速度 v_next 和位置 s_next 添加到对应的列表中
        if len(t_list) > 400000:  # 安全阈值 ：防止无限循环导致内存溢出
            break

    return {"t":np.array(t_list), "v":np.array(v_list), "s":np.array(s_list), "T":float(t_list[-1])}#返回包含时间序列 t、速度序列 v、位置序列 s 和实际总运行时间 T 的字典 
                                                                                                    #将列表格式的时序数据转为 numpy 数组（方便后续计算和可视化），返回给调用者：

def refine_meiyan_for_time(t_target, v_hint, tol=0.5, v_lo=8.0, v_hi=22.22):#v_lo/v_hi：二分搜索的速度上下界，t_target, v_hint：目标时间和初始速度猜测
    """二分 vmax，使得 T(vmax) ≈ t_target"""
    T_lo = simulate_meiyan_yongmaolu(v_lo)["T"]#用 ["T"] 从返回字典中提取总运行时间（最慢速度→最长时间）
    T_hi = simulate_meiyan_yongmaolu(v_hi)["T"]#计算v_hi（22.22m/s）对应的运行时间T_hi（最快速度→最短时间）
    if not (min(T_lo,T_hi)-1e-6 <= t_target <= max(T_lo,T_hi)+1e-6):## 若目标时间不在范围内，返回初始提示v_hint的模拟结果
        return simulate_meiyan_yongmaolu(v_hint)

    f_lo = T_lo - t_target## 计算v_lo对应的误差：f_lo = 慢速度时间 - 目标时间（通常为正，因T_lo长）
    f_hi = T_hi - t_target# 计算v_hi对应的误差：f_hi = 快速度时间 - 目标时间（通常为负，因T_hi短）
    if f_lo*f_hi > 0:# 若两端误差同号，说明未找到合适区间，尝试在更细的网格中搜索
        grid = np.linspace(6.0, 22.22, 33)#生成一个从 6.0 到 22.22 的等间距速度网格，共 33 个点，linspace(start, stop, num)：生成一个从 start 到 stop 的等间距数列，包含 num 个点
        vals = [simulate_meiyan_yongmaolu(v)["T"] - t_target for v in grid]
        #下面这种写法的简化
        '''
            # 1. 初始化空列表（存储每个v对应的误差）
                vals = []
            # 2. 遍历grid中的每个候选速度v
                for v in grid:
            # 3. 调用模拟函数，获取该速度对应的总运行时间T
                T = simulate_meiyan_yongmaolu(v)["T"]
            # 4. 计算误差（实际时间 - 目标时间），追加到列表
                error = T - t_target
                vals.append(error)
        '''
        #sign[:-1]：取符号数组 sign 除最后一个元素外的所有元素（前 n-1 个）；
        #sign[1:]：取符号数组 sign 除第一个元素外的所有元素（后 n-1 个）；
        sign = np.sign(vals); #np.sign(vals)：计算 vals 中每个元素的符号（正、负或零），返回一个数组 sign；
        idx = np.where(sign[:-1]*sign[1:] <= 0)[0]                          #np.where(sign[:-1]*sign[1:] <= 0)[0]：
                                                                            #找到 sign 数组中符号变化的位置（即误差从正变负或从负变正的索引），存储在 idx 中

        #下面这种写法的简化
        '''

        # 假设有一个布尔数组（模拟代码中的 is_sign_change）
        bool_arr = np.array([False, True, False, True, True])  # 第1、3、4个元素是True（索引从0开始）

        # 调用 np.where 提取 True 对应的索引
        result = np.where(bool_arr)
        print("np.where返回的原始结果：", result)  # 输出：(array([1, 3, 4]),) → 元组，第0个元素是索引数组
        print("提取元组第0个元素：", result[0])   # 输出：[1 3 4] → 这就是我们需要的索引数组
        
        
        '''
        if len(idx)==0: 
            return simulate_meiyan_yongmaolu(v_hint)#若未找到符号变化点，返回初始提示v_hint的模拟结果
        v_lo, v_hi = grid[idx[0]], grid[idx[0]+1]#更新二分搜索的速度上下界为第一个符号变化点对应的速度区间

    for _ in range(50):#最多迭代50次二分搜索,range(50)：生成从0到49的整数序列，共50个数，用于控制循环次数
        v_mid = 0.5*(v_lo+v_hi)#计算当前速度区间的中点 v_mid
         #模拟 v_mid 对应的总运行时间 T_mid
        T_mid = simulate_meiyan_yongmaolu(v_mid)["T"]

        if abs(T_mid - t_target) <= tol:#若当前误差在容差范围内，返回该模拟结果，与目标时间足够接近
            return simulate_meiyan_yongmaolu(v_mid)#返回 v_mid 对应的模拟结果-------字典
        if (T_mid - t_target)*(f_lo) > 0:#若 T_mid 与 t_target 的误差与 f_lo 同号，说明根在上半区间
            v_lo, f_lo = v_mid, T_mid - t_target#更新下界为中点，并更新 f_lo
        else:
            v_hi, f_hi = v_mid, T_mid - t_target#更新上界为中点，并更新 f_hi
    return simulate_meiyan_yongmaolu(0.5*(v_lo+v_hi))#若迭代结束仍未满足容差，返回最终区间中点对应的模拟结果
# ===================== 特殊段：稳健模拟 + 二分匹配时长 (泗港-曹隘)=====================


def simulate_sigang_caoai(vmax):
    L = 1825.0
    A_POS, A_NEG = 0.50,-1.45
    V1,V2,V3,V4 = 21.66,21.66,10.00,14.00
    V_MAX_UPPER = 14.44
    vmax = float(np.clip(vmax, 0.0, V_MAX_UPPER))
                        #   加速  匀速    减速     加速    减速
    S_BOUNDS = np.array([0.0, 374.5, 1375.5, 1550.5, 1696.3, L])
    targets = [
        (S_BOUNDS[0], S_BOUNDS[1], V1),
        (S_BOUNDS[1], S_BOUNDS[2], V2), 
        (S_BOUNDS[2], S_BOUNDS[3], V3),
        (S_BOUNDS[3], S_BOUNDS[4], vmax),
    ]

    def accel_to_target(v_cur, v_tar):
        if v_cur < v_tar - 1e-9: return A_POS
        if v_cur > v_tar + 1e-9: return A_NEG
        return 0.0

    t_list=[0.0]; v_list=[0.0]; s_list=[0.0]
    seg_idx=0; cur_tar=targets[seg_idx][2]

    while True:
        t,v,s = t_list[-1], v_list[-1], s_list[-1]
        if s >= L - 1e-8 and v <= 1e-6: break

        while seg_idx < len(targets) and s >= targets[seg_idx][1] - 1e-12:
            seg_idx += 1
            if seg_idx < len(targets): cur_tar = targets[seg_idx][2]

        if s >= S_BOUNDS[3] - 1e-12:
            # 1550m 之后：先追向 14，再根据刹停距离判断是否制动到 0
            if v > V4 + 1e-9: a=A_NEG
            elif v < V4 - 1e-9: a=A_POS
            else: a=0.0
            v_const = max(v, V4)
            d_stop = (v_const**2)/(2.0*abs(A_NEG))
            if abs(v - V4) <= 1e-6 and (L - s) <= d_stop + 1e-6:
                a = A_NEG
        else:
            a = accel_to_target(v, cur_tar)

        v_next = max(v + a*DT, 0.0)
        s_next = s + v_next*DT
        t_next = t + DT

        # 末端截断到 L，并保证停稳
        if s < L and s_next > L:
            ds = L - s
            if abs(a) < 1e-12:
                dt_last = ds / max(v, 1e-9); v_end = v
            else:
                A = 0.5*a; B=v; C=-ds
                disc = B*B - 4*A*C
                dt_last = max(( -B + math.sqrt(max(disc,0.0)) )/(2*A), 0.0)
                v_end = max(v + a*dt_last, 0.0)
            t_next = t + dt_last; s_next = L
            if v_end > 1e-6:
                t_next += v_end/abs(A_NEG)
                v_next = 0.0
            else:
                v_next = 0.0

        t_list.append(t_next); v_list.append(v_next); s_list.append(s_next)
        if len(t_list) > 400000:  # 安全阈值
            break

    return {"t":np.array(t_list), "v":np.array(v_list), "s":np.array(s_list), "T":float(t_list[-1])}

def refine_sigang_for_time(t_target, v_hint, tol=0.5, v_lo=8.0, v_hi=22.22):
    """二分 vmax，使得 T(vmax) ≈ t_target"""
    T_lo = simulate_sigang_caoai(v_lo)["T"]
    T_hi = simulate_sigang_caoai(v_hi)["T"]
    if not (min(T_lo,T_hi)-1e-6 <= t_target <= max(T_lo,T_hi)+1e-6):
        return simulate_sigang_caoai(v_hint)

    f_lo = T_lo - t_target
    f_hi = T_hi - t_target
    if f_lo*f_hi > 0:
        grid = np.linspace(6.0, 22.22, 33)
        vals = [simulate_sigang_caoai(v)["T"] - t_target for v in grid]
        sign = np.sign(vals); idx = np.where(sign[:-1]*sign[1:] <= 0)[0]
        if len(idx)==0: return simulate_sigang_caoai(v_hint)
        v_lo, v_hi = grid[idx[0]], grid[idx[0]+1]

    for _ in range(50):
        v_mid = 0.5*(v_lo+v_hi)
        T_mid = simulate_sigang_caoai(v_mid)["T"]
        if abs(T_mid - t_target) <= tol:
            return simulate_sigang_caoai(v_mid)
        if (T_mid - t_target)*(f_lo) > 0:
            v_lo, f_lo = v_mid, T_mid - t_target
        else:
            v_hi, f_hi = v_mid, T_mid - t_target
    return simulate_sigang_caoai(0.5*(v_lo+v_hi))




# ===================== 保存每段输出 =====================
def save_segment_outputs(sp, prof, t_offset):#sp：表示 “路段标识”（推测是 station_pair 的缩写，如两个站点组成的路段名称 / ID，用于区分不同路段）。
                                                #prof：包含路段运动数据的字典（或类字典对象），需包含 t（时间）、v（速度）、s（距离）三个键，存储该路段的核心运动信息。
                                                #t_offset：时间偏移量（全局时间校正值），用于将路段的 “本地时间” 转换为 “全局时间”（比如整个路径的总时间中，该路段的时间相对于起点的偏移）。
    seg_dir = SEG_OUT_DIR / sp                  #定义输出目录路径，将 SEG_OUT_DIR（总输出目录）与 sp（路段标识）结合，形成该路段的专属输出目录路径。
                                                #SEG_OUT_DIR = OUT_DIR / "segments";      SEG_OUT_DIR.mkdir(exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)#mkdir(parents=True, exist_ok=True)：创建该目录，参数含义：
                                                #parents=True：如果父目录（如 ./outputs/segments）不存在，会自动创建；
                                                #exist_ok=True：如果目录已存在，不会报错（避免重复创建时的异常）。
    #下面这种写法的简化
    '''
    seg_dir = SEG_OUT_DIR / sp
    try:
        # 尝试创建目录（包括父目录）
        seg_dir.mkdir(parents=True)
    except FileExistsError:
        # 若目录已存在，不做任何操作（等价于 exist_ok=True）
    pass
    '''


    df = pd.DataFrame({
        "t_local": prof["t"],
        "v(m/s)":  prof["v"],
        "s_local(m)": prof["s"],
        "t_global": prof["t"] + t_offset,
        "station_pair": sp
    })#创建包含路段数据的 DataFrame，包含本地时间 t_local、速度 v(m/s)、本地距离 s_local(m)、全局时间 t_global（通过加上 t_offset 计算）和路段标识 station_pair 五列。
        #DataFrame：pandas 中用于存储表格数据的二维数据结构，类似于电子表格或 SQL 表格。
    df.to_csv(seg_dir/"vt_selected.csv", index=False)#将该 DataFrame 保存为 CSV 文件，文件名为 vt_selected.csv，保存在该路段的输出目录中；
                                                        #index=False 表示不保存行索引。不包含 pandas 自动生成的行索引（避免冗余数据）
    plt.figure(figsize=(6,3.0))# 创建画布（大小6x3英寸，适合展示时序数据）
    plt.plot(df["t_local"], df["v(m/s)"], lw=1.5)# 绘制速度-时间曲线，横轴为本地时间 t_local，纵轴为速度 v(m/s)，线宽为1.5
    plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")# 设置横轴和纵轴标签
    plt.title(f"v–t: {sp}"); plt.grid(True, alpha=0.3); plt.tight_layout()# 设置图表标题、网格和布局,plt.grid(True, alpha=0.3)：启用网格线，透明度为0.3，
                                                                            #plt.tight_layout()：自动调整子图参数，使图表布局更紧凑
    plt.savefig(seg_dir/"vt_selected.png", dpi=300); plt.close()# 保存图表为 PNG 文件，分辨率为300 DPI，文件名为 vt_selected.png，保存在该路段的输出目录中；plt.close()：关闭当前图表，释放内存资源
    return df# 返回该路段的 DataFrame，供后续使用

# ===================== 运行时间敏感性分析 =====================
def run_time_sensitivity(curves, T_run_baseline, max_delta=5, slack=0.5):
    """
    将运行时间依次减少 1~max_delta 秒，记录能耗与可行性。
    返回 DataFrame：delta_s, run_time_target, run_time_used, energy_Wh, feasible, note
    feasible：是否可行
    note：备注信息
    """
    rows = []#存储结果的列表
    # baseline（delta=0）
    try:
        alloc0, E0, Tused0 = dp_optimize(curves, T_run_baseline, slack=slack)   # “在 curves 约束下，找到能在目标时间内跑完的最优运行策略（如加速 / 减速时机），并返回相关结果”；
        rows.append(dict(delta_s=0, run_time_target=T_run_baseline, run_time_used=Tused0,#若优化成功（无异常），构造基准场景字典，添加到 rows 中
                         energy_Wh=E0, feasible=True, note="baseline"))
                                                                                        #记录基准场景的运行时间、实际使用时间、能耗、可行性和备注信息；
    except Exception as e:#except：异常处理的 “触发开关”，Exception：捕获的 “异常范围”，as e：错误信息的 “存储容器”
        rows.append(dict(delta_s=0, run_time_target=T_run_baseline, run_time_used=np.nan,#若优化失败（抛出异常），构造基准场景字典，添加到 rows 中
                         energy_Wh=np.nan, feasible=False, note=f"baseline infeasible: {e}"))
        
    #  try-except：异常捕获 ——dp_optimize 可能因约束过严（如基准时间本身不可行）报错，用 try-except 避免程序中断，同时记录错误原因；
    
    # deltas
    for d in range(1, max_delta+1):#遍历从 1 到 max_delta 的整数，表示运行时间减少的秒数
        target = T_run_baseline - d
        if target <= 0:#若目标时间小于等于0，直接记录为不可行
             #构造不可行场景字典，添加到 rows 中
            rows.append(dict(delta_s=d, run_time_target=target, run_time_used=np.nan,
                             energy_Wh=np.nan, feasible=False, note="target <= 0"))
            continue
        try:#尝试进行 DP 优化
            _, E, Tused = dp_optimize(curves, target, slack=slack)#在 curves 约束下，找到能在 target 时间内跑完的最优运行策略，并返回相关结果；
                #alloc 变量用 _ 接收 —— 表示 “该返回值无用，仅占位”，
            rows.append(dict(delta_s=d, run_time_target=target, run_time_used=Tused,
                             energy_Wh=E, feasible=True, note="ok"))#若优化成功，构造场景字典，添加到 rows 中，记录运行时间、实际使用时间、能耗、可行性和备注信息；
        except Exception as e:
            rows.append(dict(delta_s=d, run_time_target=target, run_time_used=np.nan,
                             energy_Wh=np.nan, feasible=False, note=str(e)))#若优化失败，构造不可行场景字典，添加到 rows 中，记录错误原因；
    return pd.DataFrame(rows)# 返回结果的 DataFrame，包含每个场景的 delta_s、目标运行时间、实际使用时间、能耗、可行性和备注信息；

# ===================== 主流程 =====================
def main():
    # 1) 读取每段 time–energy–vpeak 与 停站时长
    curves = [load_curve_for_segment(sp) for sp in STATIONS]#读取每个路段的 time–energy–vpeak 曲线数据，存储在列表 curves 中；DataFrame 列表
    dwell_list = load_dwell_times()#读取每个站点的停站时长，存储在列表 dwell_list 中；单位：秒
    total_dwell = sum(dwell_list)#计算总停站时长

    # 2) 计算“运行时间”目标（总时长 - 停站总时长）
    T_RUN = T_TOTAL - total_dwell
    if T_RUN <= 0:
        raise ValueError(f"❌ 总时长 {T_TOTAL}s 小于或等于停站总时间 {total_dwell}s，无法优化。")
    print(f"➡️ 总时长 {T_TOTAL:.1f}s = 运行 {T_RUN:.1f}s + 停站 {total_dwell:.1f}s")#{T_TOTAL:.1f}：格式化输出，总时长保留1位小数

    # 3) DP 优化（以运行时间 T_RUN 为约束）
    alloc, E_total, T_used = dp_optimize(curves, T_RUN, slack=0.5)#在 curves 约束下，找到能在 T_RUN 时间内跑完的最优运行策略，并返回相关结果；
                                                                #alloc：最优分配方案列表，E_total：总能耗，T_used：实际使用时间
          

    # 4) 输出“最优时刻表”（含每段停站）
    df_table = pd.DataFrame([{
        "station_pair": sp,
        "t_arrival(s)": t_i,
        "v_peak_opt(m/s)": v_i,
        "energy(Wh)": e_i,
        "dwell_time(s)": dwell
    } for sp, (t_i, v_i, e_i), dwell in zip(STATIONS, alloc, dwell_list)])#构造时刻表的 DataFrame，包含每个路段的站点对、到达时间、最优峰值速度、能耗和停站时长；
                                                                        #使用列表推导式遍历 STATIONS、alloc 和 dwell_list，生成每个路段的字典，并传递给 DataFrame 构造函数；


    df_table.loc[len(df_table)] = {
        "station_pair":"总计",
        "t_arrival(s)": df_table["t_arrival(s)"].sum(),
        "v_peak_opt(m/s)":"",  # 汇总不需要
        "energy(Wh)": df_table["energy(Wh)"].sum(),
        "dwell_time(s)": df_table["dwell_time(s)"].sum()
    }
    (OUT_DIR/"schedule_table.csv").write_text(df_table.to_csv(index=False), encoding="utf-8")#将时刻表 DataFrame 保存为 CSV 文件，文件名为 schedule_table.csv，保存在输出目录中；
                                                                                        #index=False 表示不保存行索引；write_text(..., encoding="utf-8")：以 UTF-8 编码写入文本文件，确保中文字符正确保存
                                                                                            #df_table.to_csv(index=False)：将 DataFrame 转换为 CSV 格式字符串，index=False表示不保留行索引
    # 5) 重建 v-t，并拼接整线（全局时间叠加停站）
    params = load_params()# 读取普通路段的物理参数（加速度、限速等）
    t_offset = 0.0
    full_rows = []#存储各路段的时序 v-t 数据
    dwell_spans = []  # [(t_start, t_end)]
                        #存储每个停站的时间区间（用于绘图高亮显示）
    for sp, (t_i, v_i, _), dwell in zip(STATIONS, alloc, dwell_list):
        if sp == SPECIAL_PAIR1:
            prof = refine_meiyan_for_time(t_target=t_i, v_hint=v_i, tol=0.5)
            if abs(prof["T"] - t_i) > 1.0:
                print(f"⚠️ {sp} 重建时长与分配不一致：T_sim={prof['T']:.2f}s, target={t_i:.2f}s")
            
        elif sp == SPECIAL_PAIR2:
            prof = refine_sigang_for_time(t_target=t_i, v_hint=v_i, tol=0.5)
            if abs(prof["T"] - t_i) > 1.0:
                print(f"⚠️ {sp} 重建时长与分配不一致：T_sim={prof['T']:.2f}s, target={t_i:.2f}s")
        
        else:
            p = params[sp]
            prof = make_profile_trapezoid(t_arrival=t_i, v_peak=v_i,
                                          A1=p["A1"], A2=p["A2"], A_DEC=p["A_DEC"], V_MID=p["V_MID"])

        df_seg = save_segment_outputs(sp, prof, t_offset)# 保存该路段的输出文件，并获取该路段的 DataFrame，保存图片和 CSV 文件
        full_rows.append(df_seg[["t_global","v(m/s)","s_local(m)"]].assign(station_pair=sp))# 将该路段的全局时间、速度、本地距离和路段标识添加到 full_rows 列表中，供后续拼接整线使用；
                                                                                        #assign(station_pair=sp)：为 DataFrame 添加一列 station_pair，值为当前路段标识 sp
                                                                                        #把处理好的这些数据添加到full_rows这个列表里
        # 段结束后的停站：只影响全局时间轴
        t_offset += prof["T"]
        dwell_spans.append((t_offset, t_offset + dwell))#记录该停站的时间区间，起始时间为当前 t_offset，结束时间为 t_offset + dwell
        t_offset += dwell#更新 t_offset，增加停站时长 dwell，为下一个路段的全局时间做准备

    df_full = pd.concat(full_rows, ignore_index=True)# 将所有路段的 DataFrame 拼接成一个完整的 DataFrame，ignore_index=True 表示重新索引，避免重复索引问题；
                                                    #concat：pandas 中用于拼接多个 DataFrame 的函数

    # 串接位移
    s_global=[]; last_sp=None; last_s_base=0.0#初始化全局位移列表 s_global，last_sp 用于跟踪上一个路段标识，last_s_base 用于记录上一个路段的末端全局位移基准
    for _,r in df_full.iterrows():#遍历完整 DataFrame 的每一行，iterrows()：pandas 中用于逐行遍历 DataFrame 的方法，返回行索引和行数据（作为 Series 对象）
        sp = r["station_pair"]
        if sp != last_sp:#检测到新的路段开始
             #更新基准位移为上一个路段的末端位移
            last_s_base = s_global[-1] if s_global else 0.0#若 s_global 非空，取最后一个元素作为基准，否则为0.0（处理首段情况）
            last_sp = sp
        s_global.append(last_s_base + r["s_local(m)"])#计算当前行的全局位移，等于上一个路段的末端位移加上当前行的本地位移 s_local(m)，并添加到 s_global 列表中
    df_full["s_global(m)"]=s_global#将计算得到的全局位移列表添加为完整 DataFrame 的新列 s_global(m)

    # 6) 导出与绘图（把停站用淡灰色高亮）
    full_csv = OUT_DIR / f"full_vt_schedule_total{int(round(T_TOTAL))}s_run{int(round(T_RUN))}s.csv"#定义完整 v-t 表的输出 CSV 文件路径，文件名包含总时长和运行时长信息，便于识别；
    df_full[["t_global","v(m/s)","s_global(m)","station_pair"]].to_csv(full_csv, index=False)# 将完整 DataFrame 的全局时间、速度、全局位移和路段标识保存为 CSV 文件，
                                                                                                #文件名为 full_vt_schedule_total{T_TOTAL}s_run{T_RUN}s.csv，保存在输出目录中；
                                                                                            #index=False 表示不保存行索引
    print("已保存整线 v–t 表：", full_csv)

    plt.figure(figsize=(12,4))# 创建画布（大小12x4英寸，适合展示全线时序数据）
    plt.plot(df_full["t_global"], df_full["v(m/s)"], lw=1)# 绘制速度-时间曲线，横轴为全局时间 t_global，纵轴为速度 v(m/s)，线宽为1
    ax = plt.gca()#获取当前坐标轴对象，便于后续对图表进行进一步定制 
                    #gca(): get current axis 的缩写，用于获取当前活动的坐标轴对象
    for t0,t1 in dwell_spans[:-1]:  # 最后一段停站可选是否绘
        ax.axvspan(t0, t1, color="k", alpha=0.06, lw=0)#在图表上绘制停站时间区间的高亮区域，使用淡灰色（黑色 alpha=0.06），表示列车在该时间段内停站；
                                        #axvspan(t0, t1, ...)：在坐标轴上绘制垂直于 x 轴的矩形区域，表示从 t0 到 t1 的时间区间

    for sp in STATIONS[1:]:# 绘制各段分界线（除首段外）
        t_cut = df_full[df_full["station_pair"]==sp]["t_global"].min()#获取当前路段 sp 的最小全局时间 t_cut，作为分界线位置
                                                                        #df_full[df_full["station_pair"]==sp]：筛选出完整 DataFrame 中路段标识为 sp 的行；
                                                                        #外层 df_full[...]：通过布尔索引提取目标路段的所有数据行，形成一个子 DataFrame。
                                                                        #["t_global"].min()：取这些行的全局时间列的最小值
        if pd.notna(t_cut):#notna(t_cut)：检查 t_cut 是否为非缺失值（非 NaN），确保分界线位置有效
             #绘制垂直分界线，样式为虚线，线宽为0.6，透明度为0.35
            plt.axvline(t_cut, ls="--", lw=0.6, alpha=0.35)

    plt.xlabel("Global Time (s)"); plt.ylabel("Velocity (m/s)")# 设置横轴和纵轴标签
     #设置图表标题，包含运行时间、停站时间、总时长和总能耗信息
    plt.title(
        f"Full-line v–t schedule (run≈{T_used:.2f}s + dwell≈{total_dwell:.1f}s = total≈{T_TOTAL:.1f}s, "
        f"E≈{E_total:.1f}Wh)"
    )
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR/"full_vt.png", dpi=300); plt.close()

    # 7) 运行时间敏感性分析（将运行时间依次减少 1~5 秒）
    df_sens = run_time_sensitivity(curves, T_RUN, max_delta=5, slack=0.5)#运行时间敏感性分析，尝试将运行时间依次减少 1 到 5 秒，记录能耗与可行性，返回结果的 DataFrame；
     #保存能耗敏感性分析结果为 CSV 文件，文件名为 energy_sensitivity_run_minus_1_to_5s.csv，保存在输出目录中；
    sens_csv = OUT_DIR / "energy_sensitivity_run_minus_1_to_5s.csv"
    df_sens.to_csv(sens_csv, index=False)
    print("已保存能耗敏感性分析表：", sens_csv)
    print("\n✅ 完成！结果位于：", OUT_DIR.resolve())

if __name__ == "__main__":
    main()