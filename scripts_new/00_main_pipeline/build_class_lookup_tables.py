# -*- coding: utf-8 -*-
"""
主线第 2 步：构建运行等级统计表。

作用：
1. 按 5 号线 26 个区间的固定顺序扫描原始 Excel。
2. 从每个文件的“静态”工作表中提取计划运行等级和实际运行时间。
3. 统计每个区间 Class1-Class5 的历史出现频次，以及历史最小/最大运行时间。
4. 输出给后续 ATO 模板生成、目标时间约束和结果分析使用。
"""

# os/glob：扫描日期文件夹和 Excel 文件路径。
import os
import glob
# pandas/numpy：读取 Excel、做透视表和数值处理。
import pandas as pd
import numpy as np
# warnings：屏蔽 Excel 读取时的非关键警告。
import warnings
# re：从“运行等级”文本中提取 class 数字。
import re

warnings.filterwarnings("ignore")

# ================= 1. 严格定义的 26 站正向顺序 =================
# 后续输出需要按线路实际方向排序，因此这里不用文件扫描顺序。
LINE5_SECTIONS_ORDER = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# 当前脚本所在目录；注意 scripts_new 分类目录下直接运行时，project_root 可能需要调整。
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录，这里沿用原脚本“脚本目录的上一级”写法。
project_root = os.path.dirname(current_dir)
# 原始运行数据目录，里面通常按日期分文件夹存放 Excel。
RAW_DATA_PATH = os.path.join(project_root, "data", "raw_data")
# 等级统计结果输出目录。
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "class_tables_strict")
# 确保输出目录存在。
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_class_name(val):
    """把原始等级字段统一规范成 class1-class5。"""

    # 空值无法判断等级，返回 None，后面会跳过。
    if pd.isna(val): return None
    # 转小写并去掉首尾空格，兼容 Class3、class 3 等写法。
    s = str(val).lower().strip()
    # 从文本里提取第一个数字。
    match = re.search(r'\d', s)
    if match:
        num = match.group()
        # 只接受 1-5 五个 ATO 等级。
        if num in '12345':
            return f"class{num}"
    return None

def build_frequency_and_time_stats():
    """扫描原始数据并生成“等级频次 + 历史时间范围”统计表。"""

    # 用于存储所有有效行车记录样本，每条记录包含区段、等级、实际时间。
    all_samples = []

    # 只扫描 05012-* 这种日期目录，避免把压缩包或无关目录读进来。
    date_folders = [d for d in os.listdir(RAW_DATA_PATH) if d.startswith("05012-")]
    print(f"📅 正在全量扫描日期文件夹: {date_folders}")

    # 按固定线路顺序逐区间扫描，保证最终表格行顺序稳定。
    for section in LINE5_SECTIONS_ORDER:
        # 每个日期目录下查找包含当前区间名的 Excel 文件。
        search_pattern = os.path.join(RAW_DATA_PATH, "05012-*", f"*{section}.xlsx")
        # 过滤 Excel 临时文件，避免读取 ~$ 开头的锁文件。
        all_date_files = [f for f in glob.glob(search_pattern) if not os.path.basename(f).startswith("~$")]
        
        for f in all_date_files:
            try:
                # “静态”表通常存放该趟运行的计划等级和实际运行时间。
                df_h = pd.read_excel(f, sheet_name='静态')
                # 清理列名空格，避免“列名看起来一样但匹配失败”。
                df_h.columns = [str(c).strip() for c in df_h.columns]
                
                # 寻找计划等级列：兼容“计划运行等级”等不同命名。
                plan_col = next((c for c in df_h.columns if '计划' in c and '等级' in c or '计划运行等级' in c), None)
                # 寻找实际运行时间列。
                time_col = next((c for c in df_h.columns if '实际' in c and '时间' in c), None)
                
                if plan_col and time_col:
                    # 只保留等级和时间都不为空的样本。
                    subset = df_h[[plan_col, time_col]].dropna()
                    for _, row in subset.iterrows():
                        # 统一等级格式。
                        c_norm = normalize_class_name(row[plan_col])
                        # 将运行时间转成数值，无法转换的变成 NaN。
                        t_val = pd.to_numeric(row[time_col], errors='coerce')
                        
                        # 等级合法且时间合法，才加入统计样本。
                        if c_norm and not np.isnan(t_val):
                            all_samples.append({
                                '区段': section, 
                                '等级': c_norm, 
                                '实际时间': t_val
                            })
            except:
                # 单个文件异常不影响全量统计，直接跳过。
                pass

    if not all_samples:
        print("❌ 未提取到任何有效数据")
        return

    # 1. 转换为大表：每行是一条“区段-等级-实际时间”样本。
    df_master = pd.DataFrame(all_samples)

    # 2. 计算频次矩阵：统计每个区段下各 Class 出现多少次。
    freq_matrix = df_master.pivot_table(index='区段', 
                                        columns='等级', 
                                        aggfunc='size', 
                                        fill_value=0)

    # 3. 计算历史时间范围：每个区段的最短/最长实际运行时间。
    time_stats = df_master.groupby('区段')['实际时间'].agg(['min', 'max']).rename(
        columns={'min': '历史最小时间(s)', 'max': '历史最大时间(s)'}
    )

    # 4. 合并频次矩阵与时间统计，并补齐 class1-class5 列。
    for i in range(1, 6):
        c = f'class{i}'
        # 某个等级从未出现时，pivot_table 不会生成该列，需要补 0。
        if c not in freq_matrix.columns: freq_matrix[c] = 0
    
    # 固定列顺序，方便后续脚本读取和人工检查。
    freq_matrix = freq_matrix[[f'class{i}' for i in range(1, 6)]]
    
    # 按照地理顺序拼接，并对没有数据的区间填 0。
    final_df = pd.concat([freq_matrix, time_stats], axis=1)
    final_df = final_df.reindex(LINE5_SECTIONS_ORDER).fillna(0)
    
    # 将频次列转为整数，时间列保留两位小数。
    for i in range(1, 6):
        final_df[f'class{i}'] = final_df[f'class{i}'].astype(int)
    final_df['历史最小时间(s)'] = final_df['历史最小时间(s)'].round(2)
    final_df['历史最大时间(s)'] = final_df['历史最大时间(s)'].round(2)

    # 5. 导出统计结果，使用 utf-8-sig 方便 Excel 正确识别中文。
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
    # 直接运行本脚本时，执行等级频次与时间范围统计。
    build_frequency_and_time_stats()
