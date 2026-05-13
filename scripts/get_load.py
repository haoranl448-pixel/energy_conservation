# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import numpy as np
# import glob

# # ================= 1. 配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)

# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "load_analysis")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 你提供的 10 趟车服务号
# TARGET_SERVICES = ["51012"]

# # 26个站的严格顺序（用于确定区间顺序）
# STATIONS = [
#     "布政", "张家潭", "同德路", "石碶", "雅渡", "庙堰", "钟公庙", "鄞州区政府", 
#     "钱湖南路", "南高教园区", "下应路", "大洋江", "泗港", "曹隘", "柳隘", 
#     "海晏北路", "民安东路", "会展中心", "院士路", "盎孟港", "三官堂", 
#     "兴庄路", "兴海南路", "梅堰", "永茂路", "镇海大道", "骆驼桥"
# ]

# EMPTY_WEIGHT = 195.0 # 会议要求的空载基准

# # ================= 2. 提取数据 =================

# def get_service_loads():
#     # 存储结构: {service_id: [w1, w2, ... w26]}
#     service_loads = {sid: [] for sid in TARGET_SERVICES}
    
#     # 构造 26 个区间名
#     sections = [f"{STATIONS[i]}-{STATIONS[i+1]}" for i in range(len(STATIONS)-1)]
    
#     print("🚀 开始提取载重数据...")
    
#     for sp in sections:
#         file_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
#         if not os.path.exists(file_path):
#             print(f"⚠️ 缺失文件: {sp}")
#             for sid in TARGET_SERVICES: service_loads[sid].append(None)
#             continue
            
#         df = pd.read_excel(file_path)
#         # 统一服务号为字符串格式，处理 .0 问题
#         df['ser_str'] = df['服务号'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
        
#         for sid in TARGET_SERVICES:
#             # 找到该服务号在该区间的载重（取第一行即可，区间内载重恒定）
#             val = df[df['ser_str'] == sid]['重量'].head(1).values
#             weight = val[0] if len(val) > 0 else None
#             service_loads[sid].append(weight)
            
#     return service_loads, sections

# # ================= 3. 计算并输出 Timetable 格式 =================

# def main():
#     loads_dict, sections = get_service_loads()
    
#     final_report = []

#     for sid in TARGET_SERVICES:
#         weights = loads_dict[sid]
#         # 如果整趟车都没数据，跳过
#         if all(w is None for w in weights): continue
        
#         # 填充缺失值（用邻近区间载重补全）
#         weights_series = pd.Series(weights).ffill().bfill()
        
#         # 计算 deltaLoad
#         # 第一站的 delta = 第一区间重量 - 空载
#         # 后面每一站的 delta = 下一区间重量 - 前一区间重量
#         deltas = []
#         current_w = EMPTY_WEIGHT
        
#         for w in weights_series:
#             diff = w - current_w
#             deltas.append(round(diff, 3))
#             current_w = w
        
#         # 最后一站（骆驼桥）下完人，载重归 0 或者回到空载
#         # 这里补一个最后回空载的 delta
#         deltas.append(round(EMPTY_WEIGHT - current_w, 3))
        
#         # 构造输出表
#         for i, station in enumerate(STATIONS):
#             d_load = deltas[i]
#             final_report.append({
#                 '服务号': sid,
#                 '站序号': i + 1,
#                 '站名': station,
#                 '区间重量(t)': weights_series[i] if i < 26 else EMPTY_WEIGHT,
#                 'OpenTrack_deltaLoad': d_load
#             })
            
#             # 打印 XML 片段示例，你可以直接复制
#             if d_load != 0:
#                 pass # 这里可以在控制台输出用于检查

#     # 保存结果
#     df_out = pd.DataFrame(final_report)
#     save_path = os.path.join(OUTPUT_DIR, "service_load_for_timetable111.xlsx")
#     df_out.to_excel(save_path, index=False)
    
#     print(f"\n✅ 处理完成！")
#     print(f"📁 载重分析表已保存至: {save_path}")
#     print("\n💡 如何使用该表：")
#     print("1. 查看 'OpenTrack_deltaLoad' 列。")
#     print("2. 在你的 XML 模板中，找到对应的 <stationID>STA_x</stationID>。")
#     print("3. 将数值填入 <deltaLoad> 标签中。")
    
#     # 展示一个服务号的示例 XML 逻辑
#     sample_sid = TARGET_SERVICES[0]
#     sample_df = df_out[df_out['服务号'] == sample_sid]
#     print(f"\n--- 服务号 {sample_sid} 的 XML 载重填入参考 ---")
#     for _, r in sample_df.iterrows():
#         if r['OpenTrack_deltaLoad'] != 0:
#             print(f"站台 {r['站名']} (STA_{r['站序号']}): <deltaLoad>{r['OpenTrack_deltaLoad']:.3f}</deltaLoad>")

# if __name__ == "__main__":
#     main()





# -*- coding: utf-8 -*-
import os
import pandas as pd
from datetime import datetime, timedelta

# ================= 1. 配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

DATA_DIR = os.path.join(project_root, "data", "data_processed")
LOAD_FILE = os.path.join(project_root, "output", "analysis", "load_analysis", "service_load_for_timetable111.xlsx")
OUTPUT_DIR = os.path.join(project_root, "output", "schedule", "10_tables_51916_based")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 基础车次：提供时间骨架
BASE_TIME_SERVICE = "51916"

# 10 组载荷来源车次
LOAD_SERVICES = ["51012"]

STATIONS = [
    "布政", "张家潭", "同德路", "石碶", "雅渡", "庙堰", "钟公庙", "鄞州区政府", 
    "钱湖南路", "南高教园区", "下应路", "大洋江", "泗港", "曹隘", "柳隘", 
    "海晏北路", "民安东路", "会展中心", "院士路", "盎孟港", "三官堂", 
    "兴庄路", "兴海南路", "梅堰", "永茂路", "镇海大道", "骆驼桥"
]

START_TIME_STR = "08:00:00"
DWELL_TIME = 30

# ================= 2. 准备 51916 的时间数据 =================

def get_base_time_skeleton():
    print(f"⏳ 正在提取基准车次 {BASE_TIME_SERVICE} 的时间骨架...")
    sections = [f"{STATIONS[i]}-{STATIONS[i+1]}" for i in range(len(STATIONS)-1)]
    skeleton = []
    
    for sp in sections:
        file_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
        df = pd.read_excel(file_path)
        # 匹配 51916
        df['ser_str'] = df['服务号'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
        df_seg = df[df['ser_str'] == BASE_TIME_SERVICE]
        
        if not df_seg.empty:
            duration = int(round(df_seg['时刻'].max()))
            # 提取 Class，如果没有则默认为 Class3
            raw_class = "Class3"
            if '计划运行等级' in df_seg.columns:
                import re
                m = re.search(r'\d', str(df_seg['计划运行等级'].iloc[0]))
                if m: raw_class = f"Class{m.group()}"
            
            skeleton.append({'section': sp, 'duration': duration, 'class': raw_class})
        else:
            # 兜底：如果 51916 缺了某一站，设个默认值
            skeleton.append({'section': sp, 'duration': 115, 'class': "Class3"})
            
    return skeleton

# ================= 3. 执行生成 =================

def main():
    # A. 获取时间骨架
    time_skeleton = get_base_time_skeleton()
    
    # B. 获取载荷数据
    df_load_all = pd.read_excel(LOAD_FILE)
    df_load_all['服务号'] = df_load_all['服务号'].astype(str)

    print(f"🚀 开始生成 10 个时刻表...")

    for load_sid in LOAD_SERVICES:
        # 提取该载荷车次的数据
        df_load_sid = df_load_all[df_load_all['服务号'] == load_sid].sort_values('站序号')
        
        current_clock = datetime.strptime(START_TIME_STR, "%H:%M:%S")
        records = []

        for i, station in enumerate(STATIONS):
            # 获取该站在当前载荷方案中的数据
            load_row = df_load_sid[df_load_sid['站名'] == station].iloc[0]
            current_weight = load_row['区间重量(t)']
            weight_diff = load_row['OpenTrack_deltaLoad']

            if i == 0:
                # 第一站：布政
                records.append({
                    '站名': station,
                    '到达时刻': "-",
                    '出发时刻': current_clock.strftime("%H:%M:%S"),
                    '运行等级': "-",
                    '区间运行时间(s)': 0,
                    '当前车重(t)': current_weight,
                    '重量变化(t)': weight_diff
                })
            else:
                # 运行段信息 (来自 51916 骨架)
                run_info = time_skeleton[i-1]
                run_sec = run_info['duration']
                
                # 到达时刻
                arr_clock = current_clock + timedelta(seconds=run_sec)
                
                # 出发时刻
                if i < len(STATIONS) - 1:
                    dep_clock = arr_clock + timedelta(seconds=DWELL_TIME)
                    dep_time = dep_clock.strftime("%H:%M:%S")
                    current_clock = dep_clock
                else:
                    dep_time = "-"

                records.append({
                    '站名': station,
                    '到达时刻': arr_clock.strftime("%H:%M:%S"),
                    '出发时刻': dep_time,
                    '运行等级': run_info['class'],
                    '区间运行时间(s)': run_sec,
                    '当前车重(t)': current_weight,
                    '重量变化(t)': weight_diff
                })

        # 保存 CSV
        df_final = pd.DataFrame(records)
        save_path = os.path.join(OUTPUT_DIR, f"timetable_51916Time_Load{load_sid}.csv")
        df_final.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 已生成: {os.path.basename(save_path)}")

if __name__ == "__main__":
    main()