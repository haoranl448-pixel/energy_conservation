# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 scripts 的上一级)
project_root = os.path.dirname(current_dir)

# 将项目根目录加入到 sys.path 中
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ================= 1. 路径与配置 =================
DATA_DIR = Path(r"D:\energy_conservation\data\data_processed")
# 存放模型 .pth 的位置
MODEL_BASE = Path(r"output/models/nn_results_residual")
# 结果输出路径
REPORT_DIR = Path(r"output/trip6_final_report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 物理常量
SLIP_RATIO = 1.015953550423103
TRIP_INDEX = 5  # 第6趟车
DT = 0.05

STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "兴庄路-兴海南路", 
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# 引入你的核心预测和规划逻辑
from src.models.wrapper import EnergyPredictor
from src.physics.train_simu import TrainTheoreticalEnergyModel
# 调度相关的函数从 run_global_schedule_residual 导入
from run_global_schedule_residual import dp_optimize, make_profile_trapezoid, load_params

# 轨迹生成相关的函数从 run_optimize_residual 导入
# 注意：确保 scripts 文件夹下有 run_optimize_residual.py 这个文件
from run_optimize_residual import get_trajectory_by_distance, get_static_data_interpolator

# ================= 2. 步骤一：提取 Trip 6 原始参数 =================
def step1_generate_input_config():
    config_rows = []
    print("🔍 正在提取 Trip 6 各站载重与时间基准...")
    for sp in STATIONS:
        df = pd.read_excel(DATA_DIR / f"results_{sp}.xlsx")
        segs = sorted(df['segment'].unique())
        data = df[df['segment'] == segs[TRIP_INDEX]]
        
        config_rows.append({
            'station_pair': sp,
            'actual_mass': data['重量'].mean(),
            'hist_run_time': data['时刻'].iloc[-1] - data['时刻'].iloc[0],
            'hist_energy_wh': data['energy'].sum() / 3600.0
        })
    
    df_config = pd.DataFrame(config_rows)
    config_path = REPORT_DIR / "trip6_input_config.csv"
    df_config.to_csv(config_path, index=False, encoding='utf-8-sig')
    print(f"✅ 步骤 1 完成：已生成输入配置文件 {config_path}")
    return df_config

# ================= 3. 步骤二：生成中间过程数据 (针对载重重算能耗) =================
def step2_generate_intermediate_pareto(df_config):
    print("🚀 步骤 2: 正在针对 Trip 6 载重重新生成能耗数据库（不使用补偿系数）...")
    
    # 获取线路静态参数（坡度等）插值器
    from run_optimize_residual import get_static_data_interpolator
    from run_global_schedule_residual import load_params
    section_phys_params = load_params() # A1, A2, A_DEC等
    
    sim_model = TrainTheoreticalEnergyModel()
    all_pareto_data = []

    for _, row in df_config.iterrows():
        sp = row['station_pair']
        mass = row['actual_mass']
        p_veh = section_phys_params[sp]
        L_total = p_veh['L']
        
        # 加载对应的 AI 模型
        model_path = MODEL_BASE / sp
        predictor = EnergyPredictor("residual", model_path)
        grad_f, curv_f = get_static_data_interpolator(sp)
        
        print(f"   正在扫描区间: {sp} (载重: {mass:.2f}t)")
        
        # 扫描不同速度等级，生成该载重下的真实能耗
        # v_scan = np.arange(5.0, p_veh['V_UPPER'] + 0.5, 0.5)
        v_scan = np.arange(5.0, p_veh['V_UPPER'] + 0.5, 3.0)
        for v_target in v_scan:
            traj = get_trajectory_by_distance(v_target, L_total, p_veh)
            
            # 物理+AI 联合预测
            e_phy_seq = sim_model.run_batch_simulation(traj['t'], traj['v'], mass)
            grad_seq = grad_f(traj['s'])
            mass_seq = np.full_like(traj['v'], mass)
            
            input_data = np.stack([traj['v'], traj['a'], e_phy_seq, grad_seq, mass_seq], axis=1)
            e_res_seq = predictor.predict(input_data)
            
            # 长度对齐
            min_len = min(len(e_phy_seq), len(e_res_seq))
            e_total = np.sum(e_phy_seq[:min_len] + e_res_seq[:min_len])
            
            all_pareto_data.append({
                'station_pair': sp,
                't_arrival': traj['T'],
                'v_peak': v_target,
                'energy_wh': e_total,

                'mass_used': mass
            })

    df_pareto = pd.DataFrame(all_pareto_data)
    database_path = REPORT_DIR / "trip6_pareto_database.csv"
    df_pareto.to_csv(database_path, index=False, encoding='utf-8-sig')
    print(f"✅ 步骤 2 完成：中间过程数据库已生成 {database_path}")
    return df_pareto

# ================= 4. 步骤三：从中间数据库读取并规划 =================
def step3_plan_and_compare(df_pareto, df_config):
    print("⚖️ 步骤 3: 正在从中间数据库读取数据进行最优分配...")
    
    target_total_time = df_config['hist_run_time'].sum()
    
    # 转换格式供 DP 使用
    curves = []
    for sp in STATIONS:
        df_sub = df_pareto[df_pareto['station_pair'] == sp].sort_values('t_arrival')
        curves.append((df_sub['t_arrival'].values, df_sub['v_peak'].values, df_sub['energy_wh'].values))
    
    # 执行 DP
    alloc, E_opt_total, T_used = dp_optimize(curves, target_total_time, slack=1.0)
    
    # --- 后续：对比、出图、生成详细报表 ---
    # (此处沿用你之前满意的绘图和报表逻辑，但数据源完全来自于重算的结果)
    # ... (绘图代码略，逻辑同前)
    print(f"🏁 规划完成。历史总能耗: {df_config['hist_energy_wh'].sum()/1000:.2f}kWh")
    print(f"🏁 规划总能耗: {E_opt_total/1000:.2f}kWh")

# if __name__ == "__main__":
#     # 执行全流程
#     config = step1_generate_input_config()
#     pareto_db = step2_generate_intermediate_pareto(config)
#     step3_plan_and_compare(pareto_db, config)



if __name__ == "__main__":
    # 定义中间文件路径
    config_path = REPORT_DIR / "trip6_input_config.csv"
    database_path = REPORT_DIR / "trip6_pareto_database.csv"

    # --- 逻辑分段执行 ---

    # 1. 检查第一步是否已有结果
    if config_path.exists():
        print(f"📊 检测到已生成的输入配置，直接加载: {config_path}")
        config = pd.read_csv(config_path)
    else:
        config = step1_generate_input_config()

    # 2. 检查第二步是否已有结果 (防止重算)
    if database_path.exists():
        print(f"✅ 检测到已生成的中间能耗数据库，直接加载: {database_path}")
        pareto_db = pd.read_csv(database_path)
    else:
        # 如果你想让第二步跑快点，请在这里修改 v_scan 的步长（见下方提示）
        pareto_db = step2_generate_intermediate_pareto(config)

    # 3. 执行规划和出图
    step3_plan_and_compare(pareto_db, config)