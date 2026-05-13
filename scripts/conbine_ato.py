# -*- coding: utf-8 -*-
"""
功能：合并“历史回放表”与“等级规划表”，生成最终对比报表。
计算：误差 = 等级能耗 - 模型E (回放能耗)
"""
import os
import pandas as pd

# ================= 1. 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 输入文件路径
PLAYBACK_FILE = os.path.join(project_root, "output", "analysis", "trip6_historical_playback", "playback_trip6_full.csv")
SCHEDULE_FILE = os.path.join(project_root, "output", "schedule", "final_plan_report", "Final_Planning_Comparison.csv")

# 输出文件路径
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "trip6_final_report_ato")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_reports():
    if not os.path.exists(PLAYBACK_FILE) or not os.path.exists(SCHEDULE_FILE):
        print("❌ 错误：找不到输入文件，请确保已经跑完了回放和规划脚本。")
        return

    df_playback = pd.read_csv(PLAYBACK_FILE)
    df_schedule = pd.read_csv(SCHEDULE_FILE)
    df_playback.columns = df_playback.columns.str.strip()
    df_schedule.columns = df_schedule.columns.str.strip()
    df_schedule = df_schedule.rename(columns={'站间区间': '区间'})
    df_merged = pd.merge(df_playback, df_schedule, on='区间')

    final_df = df_merged[['区间', '历史T', '模型E', '选定等级', '规划用时(s)', '规划能耗(Wh)']].copy()
    final_df = final_df.rename(columns={
        '选定等级': '模型等级',
        '规划用时(s)': '等级时间',
        '规划能耗(Wh)': '等级能耗'
    })
    final_df['能耗差异(Wh)'] = final_df['等级能耗'] - final_df['模型E']
    final_df['优化比例(%)'] = (-final_df['能耗差异(Wh)']/final_df['模型E']*100).round(2)
    final_df['模型E'] = final_df['模型E'].round(2)
    final_df['等级能耗'] = final_df['等级能耗'].round(2)
    final_df['能耗差异(Wh)'] = final_df['能耗差异(Wh)'].round(2)

    # ===== 增加全线总计行 =====
    total_row = pd.DataFrame({
        '区间': ['--- 全线总计 ---'],
        '历史T': [final_df['历史T'].sum()],
        '模型E': [final_df['模型E'].sum()],
        '模型等级': ['-'],
        '等级时间': [final_df['等级时间'].sum()],
        '等级能耗': [final_df['等级能耗'].sum()],
        '能耗差异(Wh)': [final_df['能耗差异(Wh)'].sum()],
        '优化比例(%)': [(-final_df['能耗差异(Wh)'].sum()/final_df['模型E'].sum()*100).round(2)]
    })
    final_df = pd.concat([final_df, total_row], ignore_index=True)

    save_path = os.path.join(OUTPUT_DIR, "Trip6_Final_Merged_Comparison_residual_new.csv")
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print(f"✅ 终极对比表已生成！")
    print(f"📁 路径: {save_path}")
    print("-" * 60)
    print(final_df.head().to_string(index=False))
    print("="*60)
if __name__ == "__main__":
    merge_reports()