# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # ================= 1. 路径设置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)

# # 刚才生成的数据表存放目录
# DATA_DIR = os.path.join(project_root, "output", "analysis", "raw_service_data")
# OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "comparison_results")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def compare_energy():
#     # 2. 读取之前生成的 CSV 文件
#     file_53004 = os.path.join(DATA_DIR, "raw_full_trip_53004.csv")
#     file_53006 = os.path.join(DATA_DIR, "raw_full_trip_53006.csv")

#     if not os.path.exists(file_53004) or not os.path.exists(file_53006):
#         print("❌ 错误：请先确保已经生成了 53004 和 53006 的全线汇总数据。")
#         return

#     df4 = pd.read_csv(file_53004)
#     df6 = pd.read_csv(file_53006)

#     # 3. 计算每个区间的总能耗（为了对比“高在哪里”）
#     # 我们根据 section 分组，计算每个区间的能耗增量
#     def get_section_energy(df):
#         # 计算每一行能耗的增量
#         df['energy_step'] = df['cum_energy_kwh'].diff().fillna(df['cum_energy_kwh'].iloc[0])
#         # 按区间统计
#         return df.groupby('section')['energy_step'].sum()

#     sec_e4 = get_section_energy(df4)
#     sec_e6 = get_section_energy(df6)

#     # 合并成一个对比表
#     comparison_df = pd.DataFrame({
#         '53004_Energy': sec_e4,
#         '53006_Energy': sec_e6
#     }).fillna(0)
#     comparison_df['Difference'] = comparison_df['53006_Energy'] - comparison_df['53004_Energy']

#     # 4. 绘图对比
#     plt.rcParams['font.sans-serif']=['Microsoft YaHei', 'SimHei']
#     plt.rcParams['axes.unicode_minus']=False

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

#     # --- 图 1: 累积能耗对比 (E-T 图) ---
#     ax1.plot(df4['time'], df4['cum_energy_kwh'], label='服务号 53004', color='#1f77b4', lw=2)
#     ax1.plot(df6['time'], df6['cum_energy_kwh'], label='服务号 53006', color='#d62728', lw=2)
    
#     # 填充两条线之间的差距
#     # 为了简化，这里直接画线，重点看末端差距
#     total_4 = df4['cum_energy_kwh'].iloc[-1]
#     total_6 = df6['cum_energy_kwh'].iloc[-1]
    
#     ax1.set_title(f"53004 vs 53006 全线累积能耗对比 (E-T)\n(53004: {total_4:.2f}kWh | 53006: {total_6:.2f}kWh)", fontsize=14)
#     ax1.set_ylabel("累积消耗电量 (kWh)")
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)

#     # --- 图 2: 分区间能耗差异对比 (直方图) ---
#     # 看哪一站跑的比另一站费电
#     x = np.arange(len(comparison_df))
#     width = 0.35
    
#     ax2.bar(x - width/2, comparison_df['53004_Energy'], width, label='53004', color='#1f77b4', alpha=0.7)
#     ax2.bar(x + width/2, comparison_df['53006_Energy'], width, label='53006', color='#d62728', alpha=0.7)
    
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(comparison_df.index, rotation=45, ha='right', fontsize=8)
#     ax2.set_title("各站间区间原始能耗对比 (哪一站费电？)", fontsize=14)
#     ax2.set_ylabel("消耗电量 (kWh)")
#     ax2.legend()
#     ax2.grid(axis='y', alpha=0.3)

#     plt.tight_layout()
#     save_path = os.path.join(OUTPUT_DIR, "energy_comparison_53004_vs_53006.png")
#     plt.savefig(save_path, dpi=300)
#     plt.show()

#     # 5. 输出具体的差异表
#     print("\n" + "="*50)
#     print("📊 53004 vs 53006 能耗差异排名前 5 的区间：")
#     print("正值表示 53006 更费电，负值表示 53004 更费电")
#     print("="*50)
#     top_diff = comparison_df.sort_values(by='Difference', key=abs, ascending=False).head(5)
#     print(top_diff[['Difference']])
#     print("="*50)
#     print(f"✅ 对比图已保存至: {save_path}")

# if __name__ == "__main__":
#     compare_energy()



# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 路径设置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

DATA_DIR = os.path.join(project_root, "output", "analysis", "raw_service_data")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 26 站严格顺序（用于保证 X 轴顺序正确）
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

def compare_travel_time():
    # 2. 读取 CSV 文件
    file_53004 = os.path.join(DATA_DIR, "raw_full_trip_53004.csv")
    file_53006 = os.path.join(DATA_DIR, "raw_full_trip_53006.csv")

    if not os.path.exists(file_53004) or not os.path.exists(file_53006):
        print("❌ 错误：请先生成 53004 和 53006 的汇总数据。")
        return

    df4 = pd.read_csv(file_53004)
    df6 = pd.read_csv(file_53006)

    # 3. 计算每个区间的纯运行时间 (Time_max - Time_min)
    def get_durations(df):
        # 计算每个 section 内的时间跨度
        durations = df.groupby('section')['time'].agg(lambda x: x.max() - x.min())
        # 按照 26 站标准顺序重排索引
        return durations.reindex(LINE5_SECTIONS)

    dur4 = get_durations(df4)
    dur6 = get_durations(df6)

    # 合并对比表
    comp_time = pd.DataFrame({
        '53004_Time': dur4,
        '53006_Time': dur6
    }).fillna(0)
    
    comp_time['Diff'] = comp_time['53006_Time'] - comp_time['53004_Time']

    # 4. 绘图
    plt.rcParams['font.sans-serif']=['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus']=False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    x = np.arange(len(LINE5_SECTIONS))
    width = 0.35

    # --- 图 1: 各站运行时间对比 ---
    ax1.bar(x - width/2, comp_time['53004_Time'], width, label='53004', color='#1f77b4', alpha=0.8)
    ax1.bar(x + width/2, comp_time['53006_Time'], width, label='53006', color='#ff7f0e', alpha=0.8)
    
    ax1.set_title("53004 vs 53006 各区间运行时间对比", fontsize=14)
    ax1.set_ylabel("运行时间 (秒)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LINE5_SECTIONS, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # --- 图 2: 时间差异图 (53006 - 53004) ---
    # 正值表示 53006 跑慢了，负值表示 53006 跑快了
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in comp_time['Diff']]
    ax2.bar(x, comp_time['Diff'], color=colors, alpha=0.7)
    
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title("时间差异 (正值:53006更慢 | 负值:53006更快)", fontsize=14)
    ax2.set_ylabel("时间差 (秒)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(LINE5_SECTIONS, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 标注总时间
    total_4 = comp_time['53004_Time'].sum()
    total_6 = comp_time['53006_Time'].sum()
    plt.figtext(0.1, 0.02, f"全线运行总时长: 53004 = {total_4:.1f}s | 53006 = {total_6:.1f}s", fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "time_comparison_53004_vs_53006.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("\n" + "="*50)
    print(f"📊 全线运行时间汇总:")
    print(f"   - 53004 总用时: {total_4:.1f} 秒")
    print(f"   - 53006 总用时: {total_6:.1f} 秒")
    print(f"   - 差值: {total_6 - total_4:+.1f} 秒")
    print("="*50)
    print(f"✅ 对比图已保存至: {save_path}")

if __name__ == "__main__":
    compare_travel_time()