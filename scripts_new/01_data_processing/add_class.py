# -*- coding: utf-8 -*-
import pandas as pd
import os
from pathlib import Path

# ================= 1. 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 输入与输出
DATA_DIR = Path(project_root) / "data" / "data_processed_new"
OUTPUT_DIR = Path(project_root) / "data" / "data_processed_new_v2"
LOOKUP_FILE = Path(project_root) / "data" / "static" / "service_class_lookup.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_class_column():
    if not LOOKUP_FILE.exists():
        print("❌ 请先运行脚本一生成映射表。")
        return

    # 加载字典
    df_map = pd.read_csv(LOOKUP_FILE)
    
    # 获取待处理文件列表
    files = list(DATA_DIR.glob("cleaned_*.xlsx"))
    print(f"🚀 准备处理 {len(files)} 个结果文件...")

    for f_path in files:
        # 从文件名解析区间，例如 results_布政-张家潭.xlsx -> 布政-张家潭
        section_name = f_path.stem.replace("cleaned_", "")
        
        print(f"正在匹配: {section_name}")
        df = pd.read_excel(f_path)

        if '服务号' not in df.columns:
            print(f"  ⚠️ 跳过 {section_name}: 数据中缺少‘服务号’列")
            continue

        # 确保服务号是整数类型以便匹配
        df['服务号_tmp'] = pd.to_numeric(df['服务号'], errors='coerce').fillna(-1).astype(int)
        
        # 提取当前区间的字典子集
        current_map = df_map[df_map['区间'] == section_name][['服务号', '运行等级']]
        current_map = current_map.rename(columns={'服务号': '服务号_tmp'})

        # 缝合等级列
        # 使用左连接，确保不丢失原始动态数据
        df_merged = pd.merge(df, current_map, on='服务号_tmp', how='left')
        
        # 删掉临时列
        df_merged = df_merged.drop(columns=['服务号_tmp'])
        
        # 处理可能的重复（如果字典里有重复的服务号）
        df_merged = df_merged.drop_duplicates(subset=['时刻', '速度(m/s)', 'segment'])

        # 保存
        save_path = OUTPUT_DIR / f_path.name
        df_merged.to_excel(save_path, index=False)
        print(f"  ✅ 已生成: {save_path.name}")

if __name__ == "__main__":
    apply_class_column()