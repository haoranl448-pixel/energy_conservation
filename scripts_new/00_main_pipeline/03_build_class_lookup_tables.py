# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import warnings
import re
from pathlib import Path

warnings.filterwarnings("ignore")

# ================= 1. 严格定义的 26 站正向顺序 =================
LINE5_SECTIONS_ORDER = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root = str(PROJECT_ROOT)
RAW_DATA_PATH = os.path.join(project_root, "data", "raw_data")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "class_tables_strict")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_class_name(val):
    if pd.isna(val): return None
    s = str(val).lower().strip()
    match = re.search(r'\d', s)
    if match:
        num = match.group()
        if num in '12345':
            return f"class{num}"
    return None

def build_frequency_and_time_stats():
    # 用于存储所有的行车记录样本
    all_samples = []

    date_folders = [d for d in os.listdir(RAW_DATA_PATH) if d.startswith("05012-")]
    print(f"📅 正在全量扫描日期文件夹: {date_folders}")

    for section in LINE5_SECTIONS_ORDER:
        search_pattern = os.path.join(RAW_DATA_PATH, "05012-*", f"*{section}.xlsx")
        all_date_files = [f for f in glob.glob(search_pattern) if not os.path.basename(f).startswith("~$")]

        for f in all_date_files:
            try:
                df_h = pd.read_excel(f, sheet_name='静态')
                df_h.columns = [str(c).strip() for c in df_h.columns]

                # 寻找关键列
                plan_col = next((c for c in df_h.columns if '计划' in c and '等级' in c or '计划运行等级' in c), None)
                time_col = next((c for c in df_h.columns if '实际' in c and '时间' in c), None)

                if plan_col and time_col:
                    # 提取数据对
                    subset = df_h[[plan_col, time_col]].dropna()
                    for _, row in subset.iterrows():
                        c_norm = normalize_class_name(row[plan_col])
                        t_val = pd.to_numeric(row[time_col], errors='coerce')

                        if c_norm and not np.isnan(t_val):
                            all_samples.append({
                                '区段': section,
                                '等级': c_norm,
                                '实际时间': t_val
                            })
            except:
                pass

    if not all_samples:
        print("❌ 未提取到任何有效数据")
        return

    # 1. 转换为大表
    df_master = pd.DataFrame(all_samples)

    # 2. 计算频次矩阵 (Class 1-5 计数)
    freq_matrix = df_master.pivot_table(index='区段',
                                        columns='等级',
                                        aggfunc='size',
                                        fill_value=0)

    # 3. 计算时间极限 (按区段分组统计实际时间的最小和最大值)
    time_stats = df_master.groupby('区段')['实际时间'].agg(['min', 'max']).rename(
        columns={'min': '历史最小时间(s)', 'max': '历史最大时间(s)'}
    )

    # 4. 合并频次矩阵与时间统计
    # 确保 class1-5 列都存在
    for i in range(1, 6):
        c = f'class{i}'
        if c not in freq_matrix.columns: freq_matrix[c] = 0

    freq_matrix = freq_matrix[[f'class{i}' for i in range(1, 6)]]

    # 按照地理顺序拼接
    final_df = pd.concat([freq_matrix, time_stats], axis=1)
    final_df = final_df.reindex(LINE5_SECTIONS_ORDER).fillna(0)

    # 将时间列保留两位小数，频次列转为整数
    for i in range(1, 6):
        final_df[f'class{i}'] = final_df[f'class{i}'].astype(int)
    final_df['历史最小时间(s)'] = final_df['历史最小时间(s)'].round(2)
    final_df['历史最大时间(s)'] = final_df['历史最大时间(s)'].round(2)

    # 5. 导出
    output_path = os.path.join(OUTPUT_DIR, "historical_class_frequency_and_time_limits.csv")
    final_df.to_csv(output_path, encoding='utf-8-sig')

    print("\n" + "="*80)
    print(f"✅ 统计完成！")
    print(f"结果文件: {output_path}")
    print("="*80)
    print("预览结果 (部分列):")
    print(final_df[['class3', '历史最小时间(s)', '历史最大时间(s)']].head())
    print("="*80)

if __name__ == "__main__":
    build_frequency_and_time_stats()
