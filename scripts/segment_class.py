# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 自动定位原始数据路径
possible_paths = [
    os.path.join(project_root, "data", "raw_data"),
    r"D:\energy_conservation\energy_conservation\raw_data"
]
RAW_DATA_PATH = next((p for p in possible_paths if os.path.exists(p)), None)

# 输出路径
LOOKUP_SAVE_PATH = os.path.join(project_root, "data", "static", "service_class_lookup.csv")

# 26 站顺序
LINE5_SECTIONS_ORDER = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

def normalize_class_name(val):
    """
    更加鲁棒的等级解析：
    支持 "class3", "3", "3.0", "等级3" 等各种输入
    """
    if pd.isna(val): return None
    s = str(val).lower().strip()
    # 提取字符串中的数字
    nums = re.findall(r'\d', s)
    if nums:
        num = nums[0]
        if num in '12345':
            return f"Class{num}"
    return None

def build_service_lookup():
    if not RAW_DATA_PATH:
        print("❌ 找不到原始数据目录")
        return

    all_mappings = []
    print(f"📂 正在扫描原始数据: {RAW_DATA_PATH}")

    for section in LINE5_SECTIONS_ORDER:
        # 递归搜索包含站名的 Excel 文件
        search_pattern = os.path.join(RAW_DATA_PATH, "**", f"*{section}.xlsx")
        files = [f for f in glob.glob(search_pattern, recursive=True) if not os.path.basename(f).startswith("~$")]
        
        for f in files:
            try:
                # 读取静态表
                df_static = pd.read_excel(f, sheet_name='静态')
                
                # 1. 锁定列名 (基于你提供的样例)
                # 服务号列：'服务号'
                # 等级列：优先取 '计划运行等级'，其次取 '实际运行等级'
                service_col = '服务号'
                plan_col = '计划运行等级' if '计划运行等级' in df_static.columns else '实际运行等级'
                
                if service_col in df_static.columns and plan_col in df_static.columns:
                    # 2. 提取数据并清洗
                    temp_df = df_static[[service_col, plan_col]].copy()
                    
                    for _, row in temp_df.iterrows():
                        raw_service = row[service_col]
                        raw_class = row[plan_col]
                        
                        c_norm = normalize_class_name(raw_class)
                        # 服务号转为整数 (处理 51902.0 这种情况)
                        try:
                            s_id = int(float(raw_service))
                        except:
                            continue
                        
                        if c_norm:
                            all_mappings.append({
                                '区间': section,
                                '服务号': s_id,
                                '运行等级': c_norm
                            })
            except:
                continue

    if not all_mappings:
        print("❌ 提取失败。请核实：Excel文件里‘静态’工作表的‘计划运行等级’列是否有数据？")
        return

    # 合并、去重
    df_lookup = pd.DataFrame(all_mappings).drop_duplicates()
    
    os.makedirs(os.path.dirname(LOOKUP_SAVE_PATH), exist_ok=True)
    df_lookup.to_csv(LOOKUP_SAVE_PATH, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"✅ 成功提取记录: {len(df_lookup)} 条")
    print(f"📍 映射表已保存: {LOOKUP_SAVE_PATH}")
    print("="*50)
    print("数据预览 (前5行):")
    print(df_lookup.head())

if __name__ == "__main__":
    build_service_lookup()