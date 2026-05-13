# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re

# ================= 1. 路径与配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 输入文件
STD_TIME_FILE = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
WEIGHT_DIST_FILE = os.path.join(project_root, "output", "analysis", "weight_analysis", "historical_weight_distribution.csv")

# 输出目录
OUTPUT_BASE = os.path.join(project_root, "output", "schedule", "historical_scenarios_2900_2930")
os.makedirs(OUTPUT_BASE, exist_ok=True)

START_TOTAL_T = 2900
END_TOTAL_T = 2990
START_TIME_STR = "08:00:00"
DWELL_TIME = 30 

# 26个站的严格顺序
STATIONS_LIST = [
    "布政", "张家潭", "同德路", "石碶", "雅渡", "庙堰", "钟公庙", "鄞州区政府", 
    "钱湖南路", "南高教园区", "下应路", "大洋江", "泗港", "曹隘", "柳隘", 
    "海晏北路", "民安东路", "会展中心", "院士路", "盎孟港", "三官堂", 
    "兴庄路", "兴海南路", "梅堰", "永茂路", "镇海大道", "骆驼桥"
]

# ================= 2. 数据解析工具 =================

def get_historical_weights():
    """解析历史重量表，返回每个区间的重量列表"""
    df_w = pd.read_csv(WEIGHT_DIST_FILE)
    weight_map = {}
    for _, row in df_w.iterrows():
        section = row['区间']
        # 解析字符串 "215.24, 235.0, ..." 为 list
        raw_str = str(row['所有记录过的重量值'])
        w_list = [float(x.strip()) for x in raw_str.split(',')]
        weight_map[section] = w_list
    return weight_map

def find_class_combination(section_options, target_total):
    """寻找 Class 2/3/4 组合使总和达标"""
    # 简单的随机搜索算法，对于25个区间非常快
    for _ in range(20000):
        selected_configs = []
        current_sum = 0
        for opt in section_options:
            c_label = random.choice(['Class2', 'Class3', 'Class4'])
            val = opt[c_label]
            selected_configs.append({'class': c_label, 'time': val})
            current_sum += val
        
        if current_sum == target_total:
            return selected_configs
    return None

# ================= 3. 主程序 =================

def main():
    # 1. 加载数据
    df_std = pd.read_csv(STD_TIME_FILE).set_index('区段')
    weight_map = get_historical_weights()

    # 准备 25 个区间的时间候选项
    section_configs = []
    for i in range(len(STATIONS_LIST) - 1):
        sp_name = f"{STATIONS_LIST[i]}-{STATIONS_LIST[i+1]}"
        section_configs.append({
            'name': sp_name,
            'Class2': int(df_std.loc[sp_name, 'Class2']),
            'Class3': int(df_std.loc[sp_name, 'Class3']),
            'Class4': int(df_std.loc[sp_name, 'Class4'])
        })

    # 2. 遍历 2900s - 2930s
    for target_t in range(START_TOTAL_T, END_TOTAL_T + 1):
        print(f"正在构建总时间 {target_t}s 的历史模拟场景...")
        
        # A. 寻找时间组合
        combo = find_class_combination(section_configs, target_t)
        if not combo:
            print(f"  ⚠️ 无法组合出 {target_t}s，跳过")
            continue

        # B. 构造时刻表
        current_clock = datetime.strptime(START_TIME_STR, "%H:%M:%S")
        timetable = []
        last_weight = 0

        for i in range(len(STATIONS_LIST)):
            curr_st = STATIONS_LIST[i]
            
            if i == 0:
                # 起始站
                # 随机选第一个区间的历史重量作为初始重量
                first_section = f"{STATIONS_LIST[0]}-{STATIONS_LIST[1]}"
                current_weight = random.choice(weight_map.get(first_section, [215.24]))
                
                timetable.append({
                    '站名': curr_st,
                    '到达时刻': "-",
                    '出发时刻': current_clock.strftime("%H:%M:%S"),
                    '运行等级': "-",
                    '区间运行时间(s)': 0,
                    '当前车重(t)': current_weight,
                    '重量变化(t)': 0
                })
                last_weight = current_weight
            else:
                # 运行段信息 (来自 combo[i-1])
                run_info = combo[i-1]
                section_name = f"{STATIONS_LIST[i-1]}-{STATIONS_LIST[i]}"
                
                # 🌟 从历史中选择一个真实载重
                current_weight = random.choice(weight_map.get(section_name, [215.24]))
                
                # 计算到达时间
                arr_clock = current_clock + timedelta(seconds=run_info['time'])
                
                # 计算出发时间
                if i < len(STATIONS_LIST) - 1:
                    dep_clock = arr_clock + timedelta(seconds=DWELL_TIME)
                    dep_time = dep_clock.strftime("%H:%M:%S")
                    current_clock = dep_clock # 下一站的起点
                else:
                    dep_time = "-"

                timetable.append({
                    '站名': curr_st,
                    '到达时刻': arr_clock.strftime("%H:%M:%S"),
                    '出发时刻': dep_time,
                    '运行等级': run_info['class'],
                    '区间运行时间(s)': run_info['time'],
                    '当前车重(t)': current_weight,
                    '重量变化(t)': round(current_weight - last_weight, 2)
                })
                last_weight = current_weight

        # 3. 保存
        df_final = pd.DataFrame(timetable)
        save_path = os.path.join(OUTPUT_BASE, f"scenario_total_{target_t}s.csv")
        df_final.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 完成！31个基于历史经验的时刻表已生成。")
    print(f"📁 路径: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()