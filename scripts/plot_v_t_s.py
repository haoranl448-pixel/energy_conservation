# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import sys
# import warnings

# # 忽略干扰警告
# warnings.filterwarnings("ignore")

# # ================= 1. 环境与路径配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# # 字体设置
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
# BASE_OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "vst_batch_analysis")

# # 严格 26 站顺序
# LINE5_SECTIONS = [
#     "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
# ]

# CLASS_STYLES = {
#     'class1': (0, (5, 5)), 'class2': (0, (3, 5, 1, 5)), 
#     'class3': 'solid', 'class4': (0, (1, 1)), 'class5': (0, (5, 1, 1, 1))
# }

# # ================= 2. 辅助逻辑 =================

# def get_class_info(actual_duration, standard_times):
#     diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
#     return min(diffs, key=diffs.get)

# # ================= 3. 单站分析核心函数 =================

# def analyze_single_station(sp):
#     station_dir = os.path.join(BASE_OUTPUT_DIR, sp)
#     indi_dir = os.path.join(station_dir, "individual_segments")
#     os.makedirs(indi_dir, exist_ok=True)

#     # 加载等级标准
#     try:
#         df_std = pd.read_csv(STD_TABLE_PATH).set_index('区段')
#         standard_times = df_std.loc[sp].values.astype(float).tolist()
#     except:
#         print(f"⚠️ 跳过 {sp}: 标准时间缺失")
#         return

#     # 加载数据
#     pattern = os.path.join(DATA_DIR, f"results_{sp}*.xlsx")
#     files = glob.glob(pattern)
#     if not files: return
    
#     df = pd.read_excel(files[0])
#     # 数据清洗
#     for col in ['累计位移(m)', '速度(m/s)', '时刻']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])

#     unique_segs = df['segment'].unique()
#     start_st, end_st = sp.split("-")
    
#     # 汇总图初始化
#     fig_merged, ax1_m = plt.subplots(figsize=(16, 9))
#     ax2_m = ax1_m.twinx()
#     colors = plt.colormaps.get_cmap('tab20')
#     timetable_records = []

#     for idx, seg_id in enumerate(unique_segs):
#         group = df[df['segment'] == seg_id].sort_values('时刻')
#         if len(group) < 10: continue
        
#         s, v, t_local = group['累计位移(m)'].values, group['速度(m/s)'].values, group['时刻'].values
#         actual_dur = t_local.max()
#         seg_class = get_class_info(actual_dur, standard_times)
#         l_style = CLASS_STYLES.get(seg_class, 'solid')
#         color = colors(idx % 20)
        
#         # --- 1. 绘制汇总图数据 ---
#         ax1_m.plot(s, v, color=color, lw=1.5, linestyle=l_style, alpha=0.7, label=f"Seg {seg_id} ({seg_class})")
#         ax2_m.plot(s, t_local, color=color, lw=0.6, linestyle=l_style, alpha=0.25)

#         # --- 2. 🌟 关键新增：绘制并保存独立图 (Individual Plot) ---
#         fig_indi, axi1 = plt.subplots(figsize=(10, 6))
#         axi2 = axi1.twinx()
        
#         axi1.plot(s, v, color='tab:blue', lw=1.5, label='速度 $v$')
#         axi2.plot(s, t_local, color='tab:red', lw=1.2, label='时间 $t$')
        
#         axi1.set_title(f"Segment {seg_id} 详情 ({seg_class}) | 耗时: {actual_dur:.1f}s")
#         axi1.set_xlabel(f"位移里程 (m) [{start_st} -> {end_st}]")
#         axi1.set_ylabel("速度 Velocity (m/s)", color='tab:blue')
#         axi2.set_ylabel("时间 Time (s)", color='tab:red')
#         axi1.grid(True, alpha=0.2)
        
#         # 保存独立图
#         fig_indi.savefig(os.path.join(indi_dir, f"segment_{seg_id}.png"), dpi=120)
#         plt.close(fig_indi) # 必须关闭，否则内存会爆

#         timetable_records.append({
#             'Segment_ID': seg_id, '等级': seg_class, 
#             '运行时间(s)': round(actual_dur, 2), '最大速度(m/s)': round(v.max(), 2)
#         })

#     # 装饰并保存汇总图
#     ax1_m.set_title(f"{sp} 全线轨迹分析汇总", fontsize=15)
#     ax1_m.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8, ncol=1)
#     plt.subplots_adjust(right=0.8)
#     fig_merged.savefig(os.path.join(station_dir, f"merged_vst_{sp}.png"), dpi=200, bbox_inches='tight')
#     pd.DataFrame(timetable_records).to_csv(os.path.join(station_dir, f"timetable_{sp}.csv"), index=False)
#     plt.close(fig_merged)
#     return True

# # ================= 4. 运行 =================
# if __name__ == "__main__":
#     os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
#     for i, sp in enumerate(LINE5_SECTIONS):
#         print(f"正在分析 [{i+1}/26]: {sp}")
#         analyze_single_station(sp)







# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import glob
# import sys
# import warnings

# # 忽略干扰警告
# warnings.filterwarnings("ignore")

# # ================= 1. 路径与全线配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# # 字体设置
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
# OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "vst_batch_analysis")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 26个站的严格顺序
# LINE5_SECTIONS = [
#     "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
# ]

# CLASS_STYLES = {
#     'class1': (0, (5, 5)), 'class2': (0, (3, 5, 1, 5)), 
#     'class3': 'solid', 'class4': (0, (1, 1)), 'class5': (0, (5, 1, 1, 1))
# }

# # ================= 2. 核心处理函数 =================

# def get_class_info(actual_duration, standard_times):
#     diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
#     return min(diffs, key=diffs.get)

# def draw_subplot(sp, ax_v):
#     """在指定的 axes 上绘制单站 VST 图"""
#     try:
#         # A. 加载标准时间
#         df_std = pd.read_csv(STD_TIME_FILE).set_index('区段')
#         standard_times = df_std.loc[sp].values.astype(float).tolist()
        
#         # B. 读取并清洗数据
#         pattern = os.path.join(DATA_DIR, f"results_{sp}*.xlsx")
#         files = glob.glob(pattern)
#         if not files: return False
#         df = pd.read_excel(files[0])
        
#         # 🌟 解决报错的关键：强制数值转换
#         for col in ['累计位移(m)', '速度(m/s)', '时刻']:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])

#         ax_t = ax_v.twinx()
#         unique_segs = df['segment'].unique()
#         colors = plt.get_cmap('tab20')
        
#         start_st, end_st = sp.split("-")
#         max_s = 0

#         for idx, seg_id in enumerate(unique_segs):
#             group = df[df['segment'] == seg_id].sort_values('时刻')
#             if len(group) < 10: continue
            
#             s, v, t_local = group['累计位移(m)'].values, group['速度(m/s)'].values, group['时刻'].values
#             actual_dur = t_local.max()
#             max_s = max(max_s, s.max())
            
#             seg_class = get_class_info(actual_dur, standard_times)
#             l_style = CLASS_STYLES.get(seg_class, 'solid')
#             color = colors(idx % 20)
            
#             # 绘制速度 (主轴)
#             ax_v.plot(s, v, color=color, lw=1.2, linestyle=l_style, alpha=0.8)
#             # 绘制时间 (次轴)
#             ax_t.plot(s, t_local, color=color, lw=0.6, linestyle=l_style, alpha=0.3)

#         # 设置子图装饰
#         ax_v.set_title(f"{sp}", fontsize=11, fontweight='bold', pad=10)
#         ax_v.grid(True, linestyle=':', alpha=0.3)
#         ax_v.tick_params(labelsize=8)
#         ax_t.tick_params(labelsize=8)
        
#         # 左右两端站名文字
#         ax_v.text(0, -0.8, start_st, ha='center', va='top', fontsize=7, color='gray')
#         ax_v.text(max_s, -0.8, end_st, ha='center', va='top', fontsize=7, color='gray')
        
#         return True
#     except Exception as e:
#         print(f"❌ 绘制 {sp} 失败: {e}")
#         return False

# # ================= 3. 主程序：构建大矩阵 =================

# def main():
#     global STD_TIME_FILE
#     STD_TIME_FILE = STD_TABLE_PATH
    
#     print(f"🚀 开始合成全线 26 站 VST 特性矩阵图...")
    
#     # 定义 13行 x 2列
#     rows, cols = 5,6
#     # 创建巨型画布：20英寸宽，45英寸高
#     fig = plt.figure(figsize=(30,24))
#     gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

#     for i, sp in enumerate(LINE5_SECTIONS):
#         r, c = i // cols, i % cols
#         ax = fig.add_subplot(gs[r, c])
        
#         print(f"   正在处理 [{i+1}/26]: {sp}...", end="", flush=True)
#         success = draw_subplot(sp, ax)
        
#         if success:
#             print(" ✅")
#             if c == 0: ax.set_ylabel("速度 Velocity (m/s)", fontsize=9)
#             if c == 1: ax.set_ylabel("时间 Time (s)", rotation=270, labelpad=15, fontsize=9)
#         else:
#             print(" ❌ (跳过)")
#             ax.text(0.5, 0.5, f"Missing Data:\n{sp}", ha='center', va='center', color='red')

#     # 添加全局总标题
#     fig.suptitle("宁波轨道交通5号线：全线区间 VST 驾驶特性全景矩阵\n(左轴: 速度 v | 右轴: 时间 t | 实线: Class 3 标称等级)", 
#                  fontsize=26, fontweight='bold', y=0.99)

#     # 保存文件
#     output_path = os.path.join(OUTPUT_DIR, "LINE5_Full_Trajectory_Matrix.png")
#     plt.savefig(output_path, dpi=180, bbox_inches='tight')
#     plt.close(fig)

#     print("\n" + "="*50)
#     print(f"🏁 合图制作成功！")
#     print(f"🖼️ 终极长图路径: {output_path}")
#     print(f"💡 建议：该图分辨率极高，请使用专业图片浏览器查看并放大细节。")
#     print("="*50)

# if __name__ == "__main__":
#     main()





# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import sys
# import warnings

# warnings.filterwarnings("ignore")

# # ================= 1. 环境与路径配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
# BASE_INFO_EXCEL = os.path.join(project_root, "data", "static", "线路基础数据（核查完毕）.xls")

# OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "vst_test")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 目标站对
# TARGET_SP = "布政-张家潭"

# CLASS_STYLES = {
#     'class1': (0, (5, 5)), 'class2': (0, (3, 5, 1, 5)), 
#     'class3': 'solid', 'class4': (0, (1, 1)), 'class5': (0, (5, 1, 1, 1))
# }

# # ================= 2. 数据处理逻辑 =================

# def load_mileage_info(station_pair):
#     df_station = pd.read_excel(BASE_INFO_EXCEL, sheet_name='车站和车辆基地表（station）')
#     df_station['station_name'] = df_station['station_name'].str.strip()
#     start_st, end_st = station_pair.split("-")
#     mile_s = df_station[df_station['station_name'] == start_st]['inbound_mileage'].values[0]
#     mile_e = df_station[df_station['station_name'] == end_st]['inbound_mileage'].values[0]
#     return start_st, int(mile_s), end_st, int(mile_e)

# def get_class_info(actual_duration, standard_times):
#     diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
#     return min(diffs, key=diffs.get)

# # ================= 3. 绘图主程序 =================

# def plot_vst_bottom_corners():
#     # A. 提取基础信息
#     start_st, mile_s, end_st, mile_e = load_mileage_info(TARGET_SP)
#     df_std = pd.read_csv(STD_TABLE_PATH).set_index('区段')
#     standard_times = df_std.loc[TARGET_SP].values.astype(float).tolist()

#     # B. 读取数据
#     pattern = os.path.join(DATA_DIR, f"results_{TARGET_SP}*.xlsx")
#     df = pd.read_excel(glob.glob(pattern)[0])
#     for col in ['累计位移(m)', '速度(m/s)', '时刻']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])

#     # C. 画布初始化
#     fig, ax1 = plt.subplots(figsize=(15, 8.5))
#     ax2 = ax1.twinx()
    
#     unique_segs = df['segment'].unique()
#     colors = plt.colormaps.get_cmap('tab20')
    
#     print(f"🚀 正在绘制 {TARGET_SP} 底部角标注图...")

#     for idx, seg_id in enumerate(unique_segs):
#         group = df[df['segment'] == seg_id].sort_values('时刻')
#         if len(group) < 10: continue
        
#         s, v, t_local = group['累计位移(m)'].values, group['速度(m/s)'].values, group['时刻'].values
#         seg_class = get_class_info(t_local.max(), standard_times)
#         l_style = CLASS_STYLES.get(seg_class, 'solid')
#         color = colors(idx % 20)
        
#         # 绘制
#         ax1.plot(s, v, color=color, lw=1.6, linestyle=l_style, alpha=0.8, label=f"Seg {seg_id} ({seg_class})")
#         ax2.plot(s, t_local, color=color, lw=0.8, linestyle=l_style, alpha=0.3)

#     # D. 🌟 关键：底部署名标注 🌟
#     # 使用 transAxes 坐标系，(0,0) 是左下角，(1,1) 是右上角
    
#     bbox_style = dict(facecolor='white', alpha=0.85, edgecolor='#1A237E', boxstyle='round,pad=0.4', lw=1.5)
    
#     # 左下角标注
#     text_left = f" 起始：{start_st} \n 坐标：{mile_s} m "
#     ax1.text(-0.08, -0.07, text_left, transform=ax1.transAxes, 
#              fontsize=11, fontweight='bold', color='#1A237E',
#              va='bottom', ha='left', bbox=bbox_style)

#     # 右下角标注
#     text_right = f" 终点：{end_st} \n 坐标：{mile_e} m "
#     ax1.text(1.08, -0.07, text_right, transform=ax1.transAxes, 
#              fontsize=11, fontweight='bold', color='#1A237E',
#              va='bottom', ha='right', bbox=bbox_style)

#     # E. 细节装饰
#     ax1.set_title(f"{TARGET_SP} 运行特性 VST 分析图", fontsize=18, pad=20, fontweight='bold')
#     ax1.set_xlabel("区间位移 Displacement (m)", fontsize=14)
#     ax1.set_ylabel("速度 Velocity (m/s)", fontsize=14)
#     ax2.set_ylabel("时间 Time (s)", fontsize=14)
    
#     ax1.grid(True, linestyle='--', alpha=0.3)
    
#     # F. 图例
#     ax1.legend(loc='upper left', bbox_to_anchor=(1.08, 1.0), fontsize=9, title="轨迹段汇总")

#     plt.subplots_adjust(right=0.82)

#     # 保存
#     save_path = os.path.join(OUTPUT_DIR, f"vst_bottom_corners_{TARGET_SP}.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"✅ 底部标注图已生成！路径: {save_path}")

# if __name__ == "__main__":
#     plot_vst_bottom_corners()














# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import sys
# import warnings

# warnings.filterwarnings("ignore")

# # ================= 1. 环境与路径配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
# BASE_INFO_EXCEL = os.path.join(project_root, "data", "static", "线路基础数据（核查完毕）.xls")

# # 总输出目录
# BASE_OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "vst_full_batch_report")
# os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# LINE5_SECTIONS = [
#     "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
# ]

# CLASS_STYLES = {
#     'class1': (0, (5, 5)), 'class2': (0, (3, 5, 1, 5)), 
#     'class3': 'solid', 'class4': (0, (1, 1)), 'class5': (0, (5, 1, 1, 1))
# }

# # ================= 2. 工具函数 =================

# def get_mileage_map():
#     df_station = pd.read_excel(BASE_INFO_EXCEL, sheet_name='车站和车辆基地表（station）')
#     df_station['station_name'] = df_station['station_name'].str.strip()
#     return dict(zip(df_station['station_name'], df_station['inbound_mileage']))

# def get_class_info(actual_duration, standard_times):
#     diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
#     return min(diffs, key=diffs.get)

# # ================= 3. 核心处理引擎 =================

# def process_station_with_indi(sp, mileage_map):
#     # A. 路径准备
#     station_dir = os.path.join(BASE_OUTPUT_DIR, sp)
#     indi_dir = os.path.join(station_dir, "individual_segments")
#     os.makedirs(indi_dir, exist_ok=True)

#     start_st, end_st = sp.split("-")
#     mile_s, mile_e = int(mileage_map.get(start_st, 0)), int(mileage_map.get(end_st, 0))

#     try:
#         df_std = pd.read_csv(STD_TABLE_PATH).set_index('区段')
#         standard_times = df_std.loc[sp].values.astype(float).tolist()
#     except: return False

#     pattern = os.path.join(DATA_DIR, f"results_{sp}*.xlsx")
#     files = glob.glob(pattern)
#     if not files: return False
#     df = pd.read_excel(files[0])
#     for col in ['累计位移(m)', '速度(m/s)', '时刻']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])

#     # B. 汇总大图初始化
#     fig_merged, ax1_m = plt.subplots(figsize=(14, 8))
#     ax2_m = ax1_m.twinx()
#     unique_segs = df['segment'].unique()
#     colors = plt.colormaps.get_cmap('tab20')
#     timetable_records = []
#     max_s_global = 0

#     # C. 遍历每一个 Segment
#     for idx, seg_id in enumerate(unique_segs):
#         group = df[df['segment'] == seg_id].sort_values('时刻')
#         if len(group) < 15: continue
        
#         s, v, t_local = group['累计位移(m)'].values, group['速度(m/s)'].values, group['时刻'].values
#         actual_dur = t_local.max()
#         max_s_global = max(max_s_global, s.max())
#         seg_class = get_class_info(actual_dur, standard_times)
#         l_style = CLASS_STYLES.get(seg_class, 'solid')
#         color = colors(idx % 20)
        
#         # --- 汇总图绘制 ---
#         ax1_m.plot(s, v, color=color, lw=1.5, linestyle=l_style, alpha=0.7, label=f"Seg {seg_id} ({seg_class})")
#         ax2_m.plot(s, t_local, color=color, lw=0.7, linestyle=l_style, alpha=0.25)

#         # --- 🌟 独立图生成逻辑 🌟 ---
#         fig_indi, axi1 = plt.subplots(figsize=(10, 6))
#         axi2 = axi1.twinx()
        
#         axi1.plot(s, v, color='tab:blue', lw=1.8, label='速度 $v$ (m/s)')
#         axi2.plot(s, t_local, color='tab:red', lw=1.2, label='时间 $t$ (s)')
        
#         # 统一底部标注位置 (使用你要求的坐标)
#         box_style = dict(facecolor='white', alpha=0.8, edgecolor='#1A237E', boxstyle='round,pad=0.4', lw=1.2)
#         axi1.text(-0.08, -0.07, f" 起始：{start_st} \n 坐标：{mile_s} m ", transform=axi1.transAxes, 
#                   fontsize=9, fontweight='bold', color='#1A237E', va='bottom', ha='left', bbox=box_style)
#         axi1.text(1.08, -0.07, f" 终点：{end_st} \n 坐标：{mile_e} m ", transform=axi1.transAxes, 
#                   fontsize=9, fontweight='bold', color='#1A237E', va='bottom', ha='right', bbox=box_style)

#         axi1.set_title(f"趟次详情: Segment {seg_id} ({seg_class})\n区间: {sp} | 耗时: {actual_dur:.1f}s", fontsize=12)
#         axi1.set_xlabel("区间位移 Displacement (m)", fontsize=10)
#         axi1.set_ylabel("速度 (m/s)", color='tab:blue'); axi2.set_ylabel("时间 (s)", color='tab:red')
#         axi1.grid(True, alpha=0.2)
        
#         # 保存独立图并关闭
#         plt.subplots_adjust(bottom=0.15)
#         fig_indi.savefig(os.path.join(indi_dir, f"segment_{seg_id}.png"), dpi=100, bbox_inches='tight')
#         plt.close(fig_indi)

#         timetable_records.append({'Segment_ID': seg_id, '等级': seg_class, '时长': round(actual_dur, 2)})

#     # D. 汇总大图修饰 (同样带底部坐标)
#     axi_list = [ax1_m] # 只是为了复用逻辑
#     for ax in axi_list:
#         ax.text(-0.08, -0.07, f" 起始：{start_st} \n 坐标：{mile_s} m ", transform=ax.transAxes, 
#                 fontsize=10, fontweight='bold', color='#1A237E', va='bottom', ha='left', bbox=box_style)
#         ax.text(1.08, -0.07, f" 终点：{end_st} \n 坐标：{mile_e} m ", transform=ax.transAxes, 
#                 fontsize=10, fontweight='bold', color='#1A237E', va='bottom', ha='right', bbox=box_style)

#     ax1_m.set_title(f"{sp} 运行特性全景汇总", fontsize=15, pad=20)
#     ax1_m.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=8, ncol=1)
#     plt.subplots_adjust(bottom=0.15, right=0.8)
#     fig_merged.savefig(os.path.join(station_dir, f"merged_vst_{sp}.png"), dpi=200, bbox_inches='tight')
#     pd.DataFrame(timetable_records).to_csv(os.path.join(station_dir, f"timetable_{sp}.csv"), index=False)
#     plt.close(fig_merged)
#     return True

# # ================= 4. 执行 =================

# if __name__ == "__main__":
#     print("🚀 启动全量 VST 分析（含独立趟次图绘制）...")
#     m_map = get_mileage_map()
#     for i, sp in enumerate(LINE5_SECTIONS):
#         print(f"[{i+1}/26] 处理中: {sp}...", end="", flush=True)
#         if process_station_with_indi(sp, m_map): print(" ✅")
#         else: print(" ❌")
#     print(f"\n🏁 任务完成！结果已存入: {BASE_OUTPUT_DIR}")













# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import glob
# import sys
# import warnings

# # 忽略干扰警告
# warnings.filterwarnings("ignore")

# # ================= 1. 环境与路径配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# # 字体设置
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
# BASE_INFO_EXCEL = os.path.join(project_root, "data", "static", "线路基础数据（核查完毕）.xls")
# OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "vst_summary")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 26个站的严格顺序
# LINE5_SECTIONS = [
#     "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
# ]

# CLASS_STYLES = {
#     'class1': (0, (5, 5)), 'class2': (0, (3, 5, 1, 5)), 
#     'class3': 'solid', 'class4': (0, (1, 1)), 'class5': (0, (5, 1, 1, 1))
# }

# # ================= 2. 核心数据工具 =================

# def get_mileage_map():
#     df_station = pd.read_excel(BASE_INFO_EXCEL, sheet_name='车站和车辆基地表（station）')
#     df_station['station_name'] = df_station['station_name'].str.strip()
#     return dict(zip(df_station['station_name'], df_station['inbound_mileage']))

# def get_class_info(actual_duration, standard_times):
#     diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
#     return min(diffs, key=diffs.get)

# # ================= 3. 子图绘制函数 =================

# def draw_section_on_ax(sp, ax_v, mileage_map):
#     try:
#         # A. 获取里程
#         start_st, end_st = sp.split("-")
#         mile_s, mile_e = int(mileage_map.get(start_st, 0)), int(mileage_map.get(end_st, 0))

#         # B. 加载标准时间
#         df_std = pd.read_csv(STD_TABLE_PATH).set_index('区段')
#         standard_times = df_std.loc[sp].values.astype(float).tolist()
        
#         # C. 读取并清洗数据
#         pattern = os.path.join(DATA_DIR, f"results_{sp}*.xlsx")
#         files = glob.glob(pattern)
#         if not files: return False
#         df = pd.read_excel(files[0])
#         for col in ['累计位移(m)', '速度(m/s)', '时刻']:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])

#         ax_t = ax_v.twinx()
#         unique_segs = df['segment'].unique()
#         colors = plt.get_cmap('tab20')
#         max_s = 0

#         for idx, seg_id in enumerate(unique_segs):
#             group = df[df['segment'] == seg_id].sort_values('时刻')
#             if len(group) < 10: continue
            
#             s, v, t_local = group['累计位移(m)'].values, group['速度(m/s)'].values, group['时刻'].values
#             actual_dur = t_local.max()
#             max_s = max(max_s, s.max())
            
#             seg_class = get_class_info(actual_dur, standard_times)
#             l_style = CLASS_STYLES.get(seg_class, 'solid')
#             color = colors(idx % 20)
            
#             ax_v.plot(s, v, color=color, lw=1.2, linestyle=l_style, alpha=0.7)
#             ax_t.plot(s, t_local, color=color, lw=0.6, linestyle=l_style, alpha=0.2)

#         # D. 🌟 完美的底部标注 🌟
#         bbox_style = dict(facecolor='white', alpha=0.7, edgecolor='#1A237E', boxstyle='round,pad=0.2', lw=0.8)
#         # 使用你调试好的坐标偏移
#         ax_v.text(-0.06, -0.08, f"{start_st}\n{mile_s}m", transform=ax_v.transAxes, 
#                   fontsize=7, fontweight='bold', color='#1A237E', va='top', ha='left', bbox=bbox_style)
#         ax_v.text(1.06, -0.08, f"{end_st}\n{mile_e}m", transform=ax_v.transAxes, 
#                   fontsize=7, fontweight='bold', color='#1A237E', va='top', ha='right', bbox=bbox_style)

#         # E. 子图装饰
#         ax_v.set_title(f"{sp}", fontsize=11, fontweight='bold', pad=10)
#         ax_v.grid(True, linestyle=':', alpha=0.3)
#         ax_v.tick_params(labelsize=8)
#         ax_t.tick_params(labelsize=8)
        
#         return True
#     except Exception as e:
#         print(f"❌ 绘制 {sp} 失败: {e}")
#         return False

# # ================= 4. 主程序：构建大矩阵 =================

# def main():
#     print(f"🚀 开始合成全线 26 站终极 VST 特性矩阵图...")
#     m_map = get_mileage_map()
    
#     # 定义 6行 x 5列 (共30格，放26张图)
#     n_rows, n_cols = 5,6
#     fig = plt.figure(figsize=(30,24))
#     gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

#     for i, sp in enumerate(LINE5_SECTIONS):
#         r, c = i // n_cols, i % n_cols
#         ax = fig.add_subplot(gs[r, c])
        
#         print(f"   处理中 [{i+1}/26]: {sp}...", end="", flush=True)
#         success = draw_section_on_ax(sp, ax, m_map)
        
#         if success:
#             print(" ✅")
#             if c == 0: ax.set_ylabel("速度 (m/s)", fontsize=9)
#             if c == 4 or i == 25: ax.set_ylabel("时间 (s)", rotation=270, labelpad=15, fontsize=9)
#         else:
#             print(" ❌")
#             ax.text(0.5, 0.5, f"Missing:\n{sp}", ha='center', va='center', color='gray')

#     # 全局总标题
#     fig.suptitle("宁波轨道交通5号线：全线区间驾驶特性 (VST) 汇总全景矩阵\n(实线: Class 3 标称等级 | 虚线: 其他等级 | 底部标注: 官方里程坐标)", 
#                  fontsize=28, fontweight='bold', y=0.98)

#     # 保存
#     output_path = os.path.join(OUTPUT_DIR, "LINE5_VST_MASTER_MATRIX.png")
#     # 增加 dpi 保证放大清晰
#     plt.savefig(output_path, dpi=180, bbox_inches='tight')
#     plt.close(fig)

#     print("\n" + "="*50)
#     print(f"🏁 全线合图制作成功！")
#     print(f"🖼️ 终极长图路径: {output_path}")
#     print(f"💡 建议：该图包含大量信息，请使用高清看图软件放大查看各站细节。")
#     print("="*50)

# if __name__ == "__main__":
#     main()













# # -*- coding: utf-8 -*-
# """
# scripts/plot_vst_single_hd.py
# 功能：生成单区间超高清分析图
# 1. 速度轴换算为 km/h。
# 2. 明确标注物理量名称与单位。
# 3. 底部角落注脚：显示官方里程坐标。
# 4. 高清晰度：300 DPI + 抗锯齿。
# """
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import sys
# import warnings

# # 忽略干扰警告
# warnings.filterwarnings("ignore")

# # ================= 1. 环境与路径配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# # 设置高清字体渲染
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['lines.antialiased'] = True  # 开启抗锯齿

# # 输入文件路径
# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
# BASE_INFO_EXCEL = os.path.join(project_root, "data", "static", "线路基础数据（核查完毕）.xls")
# if not os.path.exists(BASE_INFO_EXCEL):
#     BASE_INFO_EXCEL = os.path.join(project_root, "线路基础数据（核查完毕）.xls")

# # 输出目录
# OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "vst_single_hd")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 🌟 在这里修改你想要查看的站对 🌟
# TARGET_SP = "布政-张家潭"

# CLASS_STYLES = {
#     'class1': (0, (5, 5)),      # dashed
#     'class2': (0, (3, 5, 1, 5)), # dashdotted
#     'class3': 'solid',           # solid (标称)
#     'class4': (0, (1, 1)),       # dotted
#     'class5': (0, (5, 1, 1, 1))  # long dash with dots
# }

# # ================= 2. 核心工具逻辑 =================

# def load_mileage_info(station_pair):
#     """从 station sheet 获取起始和终点里程"""
#     df_station = pd.read_excel(BASE_INFO_EXCEL, sheet_name='车站和车辆基地表（station）')
#     df_station['station_name'] = df_station['station_name'].str.strip()
#     start_st, end_st = station_pair.split("-")
#     mile_s = df_station[df_station['station_name'] == start_st]['inbound_mileage'].values[0]
#     mile_e = df_station[df_station['station_name'] == end_st]['inbound_mileage'].values[0]
#     return start_st, int(mile_s), end_st, int(mile_e)

# def get_class_info(actual_duration, standard_times):
#     """匹配等级"""
#     diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
#     return min(diffs, key=diffs.get)

# # ================= 3. 绘图执行 =================

# def plot_single_hd_vst():
#     print(f"🚀 正在为区间 [{TARGET_SP}] 生成超高清分析图...")
    
#     # A. 加载基础数据
#     try:
#         start_st, mile_s, end_st, mile_e = load_mileage_info(TARGET_SP)
#         df_std = pd.read_csv(STD_TABLE_PATH).set_index('区段')
#         standard_times = df_std.loc[TARGET_SP].values.astype(float).tolist()
        
#         pattern = os.path.join(DATA_DIR, f"results_{TARGET_SP}*.xlsx")
#         df = pd.read_excel(glob.glob(pattern)[0])
        
#         # 数据清洗与类型强制转换
#         for col in ['累计位移(m)', '速度(m/s)', '时刻']:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         return

#     # B. 画布初始化
#     fig, ax1 = plt.subplots(figsize=(14, 8))
#     ax2 = ax1.twinx()  # 开启右侧时间轴
    
#     unique_segs = df['segment'].unique()
#     colors = plt.get_cmap('tab20')
    
#     max_s = 0
#     print(f"   共处理 {len(unique_segs)} 趟车轨迹...")

#     for idx, seg_id in enumerate(unique_segs):
#         group = df[df['segment'] == seg_id].sort_values('时刻')
#         if len(group) < 15: continue
        
#         s = group['累计位移(m)'].values
#         v_kmh = group['速度(m/s)'].values * 3.6  # 🌟 速度换算为 km/h
#         t_local = group['时刻'].values
#         max_s = max(max_s, s.max())
        
#         # 判定等级与确定线型
#         seg_class = get_class_info(t_local.max(), standard_times)
#         l_style = CLASS_STYLES.get(seg_class, 'solid')
#         color = colors(idx % 20)
        
#         # 1. 速度-位移 (左轴，加粗)
#         ax1.plot(s, v_kmh, color=color, lw=1.8, linestyle=l_style, alpha=0.8, 
#                  label=f"Seg {seg_id} ({seg_class})")
        
#         # 2. 时间-位移 (右轴，半透明细线)
#         ax2.plot(s, t_local, color=color, lw=0.8, linestyle=l_style, alpha=0.25)

#     # C. 物理量标注与单位
#     ax1.set_xlabel("区间位移里程 Displacement (m)", fontsize=13, fontweight='bold')
#     ax1.set_ylabel("速度 Velocity (km/h)", fontsize=13, fontweight='bold', color='#1A237E')
#     ax2.set_ylabel("时间 Time (s)", fontsize=13, fontweight='bold', color='#455A64', rotation=270, labelpad=20)
    
#     ax1.grid(True, linestyle='--', alpha=0.3)
#     ax1.tick_params(axis='both', labelsize=10)
#     ax2.tick_params(axis='y', labelsize=10)

#     # D. 🌟 底部角标注 (基于您调试的坐标) 🌟
#     bbox_style = dict(facecolor='white', alpha=0.85, edgecolor='#1A237E', boxstyle='round,pad=0.5', lw=1.2)
    
#     # 左下角注脚
#     text_left = f" 起始：{start_st} \n 坐标：{mile_s} m "
#     ax1.text(-0.06, -0.08, text_left, transform=ax1.transAxes, 
#              fontsize=11, fontweight='bold', color='#1A237E',
#              va='top', ha='left', bbox=bbox_style)

#     # 右下角注脚
#     text_right = f" 终点：{end_st} \n 坐标：{mile_e} m "
#     ax1.text(1.06, -0.08, text_right, transform=ax1.transAxes, 
#              fontsize=11, fontweight='bold', color='#1A237E',
#              va='top', ha='right', bbox=bbox_style)

#     # E. 图表修饰
#     ax1.set_title(f"{TARGET_SP} 运行特性全景分析图 (VST)\n"
#                   f"(实线: Class 3 标称等级 | 虚线: 其它等级)", fontsize=18, pad=25, fontweight='bold')

#     # F. 图例：单列置于右侧外部
#     ax1.legend(loc='upper left', bbox_to_anchor=(1.12, 1.0), 
#                fontsize=9, ncol=1, frameon=True, shadow=True, title="轨迹分段统计")

#     # 预留空间防止标注被截断
#     plt.subplots_adjust(bottom=0.18, right=0.8)

#     # G. 💾 保存超高清文件
#     save_path = os.path.join(OUTPUT_DIR, f"VST_HD_{TARGET_SP}.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print("\n" + "="*50)
#     print(f"🏁 任务完成！")
#     print(f"🖼️ 高清图路径: {save_path}")
#     print(f"📈 规格：300 DPI | km/h 速度轴 | 秒(s) 时间轴")
#     print("="*50)

# if __name__ == "__main__":
#     plot_single_hd_vst()







# # -*- coding: utf-8 -*-
# """
# 全线合一版：单站所有等级合在一张图
# 逻辑：
# 1. 所有的线都是实线 (Solid Line)，保证清晰度。
# 2. Class 1-5 分别分配 5 种对比强烈的颜色。
# 3. 同等级的 Segment 使用同一种颜色，形成“轨迹簇”。
# 4. 包含双纵轴 (km/h, s) 和底部官方坐标注脚。
# """
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import sys
# import warnings

# warnings.filterwarnings("ignore")

# # ================= 1. 环境与路径配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['lines.antialiased'] = True

# DATA_DIR = os.path.join(project_root, "data", "data_processed")
# STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
# BASE_INFO_EXCEL = os.path.join(project_root, "data", "static", "线路基础数据（核查完毕）.xls")

# OUTPUT_ROOT = os.path.join(project_root, "output", "analysis", "vst_combined_colored")
# os.makedirs(OUTPUT_ROOT, exist_ok=True)

# # 🌟 核心：为 5 个等级定义专属颜色（实线）
# # Class 1(红/最快), 2(橙), 3(绿/标准), 4(蓝), 5(紫/最慢)
# CLASS_COLORS = {
#     'class1': '#d62728', # 红色
#     'class2': '#ff7f0e', # 橙色
#     'class3': '#2ca02c', # 绿色
#     'class4': '#1f77b4', # 蓝色
#     'class5': '#9467bd'  # 紫色
# }

# LINE5_SECTIONS = [
#     "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
# ]

# # ================= 2. 工具函数 =================

# def load_mileage_map():
#     df_station = pd.read_excel(BASE_INFO_EXCEL, sheet_name='车站和车辆基地表（station）')
#     df_station['station_name'] = df_station['station_name'].str.strip()
#     return dict(zip(df_station['station_name'], df_station['inbound_mileage']))

# def get_class_info(actual_duration, standard_times):
#     diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
#     return min(diffs, key=diffs.get)

# # ================= 3. 绘图执行 =================

# def process_station_combined(sp, mileage_map, df_std_all):
#     station_dir = OUTPUT_ROOT
#     start_st, end_st = sp.split("-")
#     mile_s, mile_e = int(mileage_map.get(start_st, 0)), int(mileage_map.get(end_st, 0))
    
#     if sp not in df_std_all.index: return False
#     standard_times = df_std_all.loc[sp].values.astype(float).tolist()

#     pattern = os.path.join(DATA_DIR, f"results_{sp}*.xlsx")
#     files = glob.glob(pattern)
#     if not files: return False
    
#     df = pd.read_excel(files[0])
#     for col in ['累计位移(m)', '速度(m/s)', '时刻']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])

#     # 绘图
#     fig, ax1 = plt.subplots(figsize=(15, 9))
#     ax2 = ax1.twinx()
    
#     max_s = 0
#     plotted_classes = set() # 用于控制图例，每个等级只显示一次

#     for seg_id, group in df.groupby('segment'):
#         if len(group) < 15: continue
#         group = group.sort_values('时刻')
        
#         s = group['累计位移(m)'].values
#         v_kmh = group['速度(m/s)'].values * 3.6
#         t_local = group['时刻'].values
#         max_s = max(max_s, s.max())
        
#         assigned_class = get_class_info(t_local.max(), standard_times)
#         color = CLASS_COLORS[assigned_class]
        
#         # 🌟 关键：全部使用 solid (实线)，颜色区分等级
#         line_label = f"等级 {assigned_class[-1]}" if assigned_class not in plotted_classes else ""
        
#         ax1.plot(s, v_kmh, color=color, lw=1.6, linestyle='solid', alpha=0.7, label=line_label)
#         ax2.plot(s, t_local, color=color, lw=0.8, linestyle='solid', alpha=0.15) # 时间线调淡
        
#         plotted_classes.add(assigned_class)

#     # 标注与美化
#     ax1.set_xlabel("区间位移里程 Displacement (m)", fontsize=13, fontweight='bold')
#     ax1.set_ylabel("速度 Velocity (km/h)", fontsize=13, fontweight='bold', color='#1A237E')
#     ax2.set_ylabel("时间 Time (s)", fontsize=13, fontweight='bold', color='#455A64', rotation=270, labelpad=25)
#     ax1.grid(True, linestyle='--', alpha=0.3)

#     # 🌟 底部注脚 (保持你调教好的坐标)
#     bbox_style = dict(facecolor='white', alpha=0.85, edgecolor='#1A237E', boxstyle='round,pad=0.5', lw=1.2)
#     ax1.text(-0.06, -0.08, f" 起始：{start_st} \n 坐标：{mile_s} m ", transform=ax1.transAxes, 
#              fontsize=11, fontweight='bold', color='#1A237E', va='top', ha='left', bbox=bbox_style)
#     ax1.text(1.06, -0.08, f" 终点：{end_st} \n 坐标：{mile_e} m ", transform=ax1.transAxes, 
#              fontsize=11, fontweight='bold', color='#1A237E', va='top', ha='right', bbox=bbox_style)

#     ax1.set_title(f"{sp} 全线驾驶轨迹全景分析图 (分等级配色)\n[ 颜色说明：红-L1, 橙-L2, 绿-L3, 蓝-L4, 紫-L5 ]", 
#                   fontsize=18, pad=30, fontweight='bold')

#     # 图例处理
#     handles, labels = ax1.get_legend_handles_labels()
#     # 按照 Class 1-5 排序图例
#     sorted_legend = sorted(zip(handles, labels), key=lambda x: x[1])
#     if sorted_legend:
#         h, l = zip(*sorted_legend)
#         ax1.legend(h, l, loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=10, title="运行等级说明")

#     plt.subplots_adjust(bottom=0.18, right=0.82)
#     save_p = os.path.join(station_dir, f"VST_Combined_{sp}.png")
#     plt.savefig(save_p, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     return True

# # ================= 4. 执行全线任务 =================

# if __name__ == "__main__":
#     print("🚀 启动全线 26 站“全实线-分等级配色”合图生成任务...")
    
#     m_map = load_mileage_map()
#     df_std_all = pd.read_csv(STD_TABLE_PATH).set_index('区段')
    
#     for i, sp in enumerate(LINE5_SECTIONS):
#         print(f"[{i+1}/26] 正在处理: {sp}...", end="", flush=True)
#         if process_station_combined(sp, m_map, df_std_all):
#             print(" ✅")
#         else:
#             print(" ❌")

#     print(f"\n🏁 任务完成！所有合图位于: {OUTPUT_ROOT}")





# -*- coding: utf-8 -*-
"""
scripts/batch_vst_engineering_master.py
功能：终极工程化报表图
1. 速度轴 km/h，时间轴 s。
2. 底部注脚：垂直引线 + Kx+xxx (站名) 格式。
3. 右侧外部图例：显示每一趟 Seg 的 ID 和所属 Class。
4. 线型区分：Class 3 实线，其余虚线。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(project_root, "data", "data_processed")
STD_TABLE_PATH = os.path.join(project_root, "output", "analysis", "class_tables_strict", "standard_class_times.csv")
BASE_INFO_EXCEL = os.path.join(project_root, "data", "static", "线路基础数据（核查完毕）.xls")

OUTPUT_ROOT = os.path.join(project_root, "output", "analysis", "vst_engineering_reports")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LINE_STYLES = {
    'class3': 'solid', 'class2': (0, (5, 2)), 'class1': (0, (3, 1, 1, 1)),
    'class4': (0, (1, 1)), 'class5': (0, (5, 5))
}

LINE5_SECTIONS = [
    "镇海大道-骆驼桥"
]

# ================= 2. 工具函数 =================

def format_mileage(meters):
    """3000 -> K3+000"""
    km = int(meters // 1000)
    m = int(meters % 1000)
    return f"K{km}+{m:03d}"

def get_mileage_map():
    df_station = pd.read_excel(BASE_INFO_EXCEL, sheet_name='车站和车辆基地表（station）')
    df_station['station_name'] = df_station['station_name'].str.strip()
    return dict(zip(df_station['station_name'], df_station['inbound_mileage']))

def get_class_info(actual_duration, standard_times):
    diffs = {f"class{i+1}": abs(actual_duration - t) for i, t in enumerate(standard_times)}
    return min(diffs, key=diffs.get)

# ================= 3. 绘图核心逻辑 =================

def process_station_full(sp, mileage_map, df_std_all):
    start_st, end_st = sp.split("-")
    mile_s = mileage_map.get(start_st, 0)
    mile_e = mileage_map.get(end_st, 0)
    
    if sp not in df_std_all.index: return False
    standard_times = df_std_all.loc[sp].values.astype(float).tolist()

    pattern = os.path.join(DATA_DIR, f"results_{sp}*.xlsx")
    files = glob.glob(pattern)
    if not files: return False
    
    df = pd.read_excel(files[0])
    for col in ['累计位移(m)', '速度(m/s)', '时刻']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['累计位移(m)', '速度(m/s)', '时刻'])

    # 画布初始化：加宽以容纳右侧图例
    fig, ax1 = plt.subplots(figsize=(15, 8.5))
    ax2 = ax1.twinx()
    
    unique_segs = df['segment'].unique()
    cmap = plt.get_cmap('turbo', len(unique_segs))
    
    max_s = 0
    for idx, seg_id in enumerate(unique_segs):
        group = df[df['segment'] == seg_id].sort_values('时刻')
        if len(group) < 15: continue
        
        s, v_kmh, t_local = group['累计位移(m)'].values, group['速度(m/s)'].values * 3.6, group['时刻'].values
        max_s = max(max_s, s.max())
        assigned_class = get_class_info(t_local.max(), standard_times)
        
        l_style = LINE_STYLES.get(assigned_class, 'solid')
        color = cmap(idx)
        
        # 绘制主线（速度）
        ax1.plot(s, v_kmh, color=color, lw=1.5, linestyle=l_style, alpha=0.8, 
                 label=f"Seg {seg_id} ({assigned_class})")
        # 绘制背景参考线（时间）
        ax2.plot(s, t_local, color=color, lw=0.7, linestyle=l_style, alpha=0.25)

    # --- 🌟 底部工程注脚 🌟 ---
    trans = ax1.get_xaxis_transform()
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="#1A237E", lw=1.2, alpha=0.9)

    # 起点
    ax1.plot([0, 0], [0, -0.12], color='black', lw=1.2, transform=trans, clip_on=False)
    ax1.text(0, -0.14, f"{format_mileage(mile_s)}\n({start_st})", 
             transform=trans, ha='center', va='top', fontsize=10, 
             fontweight='bold', color='#1A237E', bbox=bbox_props)

    # 终点
    ax1.plot([max_s, max_s], [0, -0.12], color='black', lw=1.2, transform=trans, clip_on=False)
    ax1.text(max_s, -0.14, f"{format_mileage(mile_e)}\n({end_st})", 
             transform=trans, ha='center', va='top', fontsize=10, 
             fontweight='bold', color='#1A237E', bbox=bbox_props)

    # --- 🌟 右侧单列图例 🌟 ---
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), 
               fontsize=8, ncol=1, frameon=True, shadow=True, title="轨迹段详情")

    # 装饰美化
    ax1.set_xlabel("区间位移里程 Displacement (m)", fontsize=11, labelpad=20) # 给底部注脚留位
    ax1.set_ylabel("速度 Velocity (km/h)", fontsize=11, labelpad=20)
    ax2.set_ylabel("时间 Time (s)", fontsize=11, labelpad=20)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title(f"{sp} 运行轨迹全景特性分析 (VST)", fontsize=16, pad=20, fontweight='bold')

    # 限制 X 轴范围
    ax1.set_xlim(-max_s*0.05, max_s*1.05)

    # 调整整体布局：bottom留白给注脚，right留白给图例
    plt.subplots_adjust(bottom=0.22, right=0.82)
    
    save_p = os.path.join(OUTPUT_ROOT, f"VST_Full_Report_{sp}.png")
    plt.savefig(save_p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True

# ================= 4. 执行全线任务 =================

if __name__ == "__main__":
    print("🚀 正在生成 26 站工程级 VST 详表报告图...")
    m_map = get_mileage_map()
    df_std_all = pd.read_csv(STD_TABLE_PATH).set_index('区段')
    
    for i, sp in enumerate(LINE5_SECTIONS):
        print(f"[{i+1}/26] 绘图中: {sp}...", end="", flush=True)
        if process_station_full(sp, m_map, df_std_all):
            print(" ✅")
        else:
            print(" ❌")

    print(f"\n🏁 全部完成！高清报告位于: {OUTPUT_ROOT}")