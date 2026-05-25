# -*- coding: utf-8 -*-
"""
scripts/plot_load_distribution.py
功能：
1. 读取 data/data_processed 下的所有区间文件。
2. 处理【服务号】列的空白问题 (向下填充)。
3. 绘制 [站点 - 载重(质量)] 的线点图。
4. 每条线代表一个服务号(车次)。
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # 后台画图
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 路径与环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 字体设置

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 站点顺序 (X轴)
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# 提取纯站名用于X轴显示 (取前半截 + 最后一个的后半截)
STATION_NAMES = [s.split('-')[0] for s in LINE5_SECTIONS] + [LINE5_SECTIONS[-1].split('-')[1]]

def load_and_aggregate():
    print(">>> 正在读取全线数据并聚合载重信息...")
    
    # 存储结构: 列表 of 字典
    # [{'Service_ID': '50503', 'Section': '布政-张家潭', 'Mass': 215.24}, ...]
    all_records = []
    
    # 数据目录
    data_dir = os.path.join(project_root, "data", "data_processed")
    if not os.path.exists(data_dir):
        # 兼容旧目录
        data_dir = os.path.join(project_root, "data", "processed")

    for i, sp in enumerate(LINE5_SECTIONS):
        # 查找文件
        files = glob.glob(os.path.join(data_dir, f"results_{sp}*.xlsx"))
        if not files:
            print(f"   ⚠️ 缺失区间: {sp}")
            continue
            
        # 读取第一个匹配的文件
        f_path = files[0]
        try:
            df = pd.read_excel(f_path)
            
            # === 关键步骤：处理服务号 ===
            if '服务号' not in df.columns:
                print(f"   ❌ {sp} 文件中缺少'服务号'列，跳过")
                continue
                
            # 1. 强制转为字符串，防止数字/文本混杂
            df['服务号'] = df['服务号'].astype(str)
            
            # 2. 处理 "nan" 字符串 (pandas读取空值有时会变成 nan)
            df['服务号'] = df['服务号'].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
            
            # 3. 向下填充 (核心逻辑：空白默认为同上一趟车)
            df['服务号'] = df['服务号'].ffill()
            
            # 4. 如果开头就是空的，那确实没办法，只能删掉
            df = df.dropna(subset=['服务号'])
            
            # === 聚合计算 ===
            # 因为在一个区间内，同一趟车的载重通常是不变的
            # 我们按服务号分组，取重量的平均值（或者最大值，通常是一样的）
            if '重量' not in df.columns:
                print(f"   ❌ {sp} 缺少'重量'列")
                continue
                
            grouped = df.groupby('服务号')['重量'].mean()
            
            for service_id, mass in grouped.items():
                # 记录数据
                # 注意：这里我们把区间名映射为 X轴的索引 (0, 1, 2...)
                # 比如 "布政-张家潭" 代表列车在 "布政" 发车时的载重
                all_records.append({
                    'Service_ID': service_id,
                    'Station_Index': i,
                    'Station_Name': sp.split('-')[0], # 用起点站名作为X轴
                    'Mass': mass
                })
                
                # 如果是最后一个区间，我们需要补全终点站的数据 (假设载重不变，或者设为0)
                # 这里为了画图连续，我们假设它到了终点站载重还在
                if i == len(LINE5_SECTIONS) - 1:
                    all_records.append({
                        'Service_ID': service_id,
                        'Station_Index': i + 1,
                        'Station_Name': sp.split('-')[1],
                        'Mass': mass
                    })

        except Exception as e:
            print(f"   ❌ 处理出错 {sp}: {e}")

    # 转为 DataFrame
    df_res = pd.DataFrame(all_records)
    return df_res

def plot_load_lines(df):
    if df.empty:
        print("❌ 没有提取到有效数据，无法画图。")
        return

    print(f">>> 提取成功，共发现 {df['Service_ID'].nunique()} 个服务号。正在绘图...")

    # 设置画布
    plt.figure(figsize=(20, 8))
    
    # 获取所有唯一的服务号，并排序
    service_ids = sorted(df['Service_ID'].unique())
    
    # 使用 Seaborn 调色板生成足够的颜色
    palette = sns.color_palette("husl", len(service_ids))
    
    # 遍历每个服务号画线
    for idx, sid in enumerate(service_ids):
        # 提取该车次的数据
        trip_data = df[df['Service_ID'] == sid].sort_values('Station_Index')
        
        # 绘图：线点图
        plt.plot(trip_data['Station_Index'], trip_data['Mass'], 
                 marker='o', markersize=6, linewidth=2, 
                 label=f"服务号 {sid}", color=palette[idx], alpha=0.8)
        
        # 可选：在每个点标数值（如果线太密建议注释掉）
        # for _, row in trip_data.iterrows():
        #     plt.text(row['Station_Index'], row['Mass'], f"{row['Mass']:.1f}", fontsize=8)

    # 设置 X 轴
    plt.xticks(range(len(STATION_NAMES)), STATION_NAMES, rotation=90, fontsize=10)
    plt.xlabel("站点 (Station)", fontsize=12)
    plt.ylabel("列车总质量 (t)", fontsize=12)
    plt.title("各服务号(车次) 全线载重分布图", fontsize=16)
    
    # 网格
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 图例 (如果车次太多，放在图外)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., title="服务号")
    
    # 保存
    save_dir = os.path.join(project_root, "output", "analysis", "load_analysis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "train_load_distribution.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已保存: {save_path}")

if __name__ == "__main__":
    df = load_and_aggregate()
    plot_load_lines(df)