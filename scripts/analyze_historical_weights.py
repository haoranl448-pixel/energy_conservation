# -*- coding: utf-8 -*-
"""
功能：统计全线 26 个站历史上实际跑过的重量（从小到大排序）
用途：用于核实 AI 模型的训练范围，解释非物理预测现象
"""
import os
import pandas as pd
import numpy as np
import sys

# ================= 1. 环境与路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

DATA_DIR = os.path.join(project_root, "data", "data_processed")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "weight_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 26 站严格顺序
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

def analyze_weights():
    weight_summary = []

    print("🚀 开始扫描历史重量分布...")

    for sp in LINE5_SECTIONS:
        file_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
        if not os.path.exists(file_path):
            continue
            
        try:
            # 只读取重量列，提速
            df = pd.read_excel(file_path, usecols=['重量'])
            # 强制转数值并清洗
            weights = pd.to_numeric(df['重量'], errors='coerce').dropna().unique()
            # 从小到大排序
            sorted_weights = sorted(weights)
            
            if len(sorted_weights) > 0:
                weight_summary.append({
                    '区间': sp,
                    '历史最小重量(t)': round(min(sorted_weights), 2),
                    '历史最大重量(t)': round(max(sorted_weights), 2),
                    '重量跨度(t)': round(max(sorted_weights) - min(sorted_weights), 2),
                    '所有记录过的重量值': ", ".join([str(round(w, 2)) for w in sorted_weights]),
                    '记录样本数': len(sorted_weights)
                })
            else:
                print(f"⚠️ {sp} 文件中没有找到有效的重量数据")

        except Exception as e:
            print(f"❌ 处理 {sp} 失败: {e}")

    # ================= 2. 生成结果报表 =================
    df_result = pd.DataFrame(weight_summary)
    
    # 导出 CSV
    save_path = os.path.join(OUTPUT_DIR, "historical_weight_distribution.csv")
    df_result.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print(f"✅ 历史重量分布表已生成！")
    print(f"📂 报告路径: {save_path}")
    print("="*80)
    
    # 重点关注你觉得怪的那几站
    problematic_stations = ["石碶-雅渡"]
    for ps in problematic_stations:
        row = df_result[df_result['区间'] == ps]
        if not row.empty:
            print(f"\n🔍 针对异常站【{ps}】的诊断：")
            print(f"   - 历史范围: {row['历史最小重量(t)'].values[0]}t ~ {row['历史最大重量(t)'].values[0]}t")
            print(f"   - 你刚才扫描的范围: 215.24t ~ 290.0t")
            if row['历史最大重量(t)'].values[0] < 260:
                print(f"   💡 结论：AI没见过260t以上的车，所以260-290t的预测是乱猜的。")
    print("="*80)

if __name__ == "__main__":
    analyze_weights()