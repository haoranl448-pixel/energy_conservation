# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 严格配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 数据文件夹
DATE_FOLDER = "05012-2024-07-08"
DATA_DIR = os.path.join(project_root, "data", "raw_data", DATE_FOLDER)
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "correlation_strict")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 严格执行你提供的 26 站正向顺序
LINE5_SECTIONS_ORDER = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# 严格锁定 507 系列服务号
TARGET_SERVICES = ["50704", "50706", "50708", "50710", "50712", "50714", "50716"]

def run_strict_analysis():
    all_samples = []
    
    print(f"🚀 开始从 {DATE_FOLDER} 中提取正向区间数据...")

    # 按照你给的 26 站列表顺序查找文件
    for section in LINE5_SECTIONS_ORDER:
        # 构造搜索模式，确保文件名包含该区间名字（例如 "*布政-张家潭.xlsx"）
        search_pattern = os.path.join(DATA_DIR, f"*{section}.xlsx")
        files = [f for f in glob.glob(search_pattern) if not os.path.basename(f).startswith("~$")]
        
        if not files:
            print(f"⚠️ 缺失正向区间文件: {section}")
            continue
        
        # 只取匹配到的第一个文件
        file_path = files[0]
        
        try:
            df = pd.read_excel(file_path, sheet_name='静态')
            # 清洗列名
            df.columns = [str(c).strip() for c in df.columns]

            # 服务号格式对齐
            def clean_sid(x):
                if pd.isna(x): return ""
                try: return str(int(float(x)))
                except: return str(x)

            df['ser_str'] = df['服务号'].apply(clean_sid)

            # 1. 过滤掉不在列表中的车次
            df_filtered = df[df['ser_str'].isin(TARGET_SERVICES)].copy()
            
            if df_filtered.empty:
                continue

            # 2. 提取你要求的四列数据（根据你的表头描述进行映射）
            # 实际区间运行时间、计划区间运行时间、有效载荷t、区间能耗Kwh
            col_map = {
                '实际区间运行时间': 'actual_time',
                '计划区间运行时间': 'plan_time',
                '有效载荷t': 'payload',
                '区间能耗Kwh': 'energy'
            }
            
            # 检查列是否存在，如果名字微调请补充逻辑
            # 注意：有些表头可能带有"(s)"，我们做模糊匹配
            final_data_map = {}
            for target_name, internal_name in col_map.items():
                actual_col = next((c for c in df_filtered.columns if target_name in c), None)
                if actual_col:
                    final_data_map[actual_col] = internal_name
            
            if len(final_data_map) == 4:
                res_df = df_filtered[list(final_data_map.keys())].rename(columns=final_data_map)
                res_df['section'] = section # 记录站名
                all_samples.append(res_df)
            else:
                print(f"⚠️ 文件 {section} 缺少必要列: {set(col_map.keys()) - set([c.split('(')[0] for c in final_data_map.keys()])}")

        except Exception as e:
            print(f"❌ 读取 {section} 失败: {e}")

    if not all_samples:
        print("❌ 最终未提取到任何有效数据，请检查 Excel 表头字样。")
        return

    # 合并数据
    master_df = pd.concat(all_samples, ignore_index=True)
    
    # 强制数值转换
    for c in ['actual_time', 'plan_time', 'payload', 'energy']:
        master_df[c] = pd.to_numeric(master_df[c], errors='coerce')
    
    master_df = master_df.dropna()

    print(f"\n✅ 样本构建完成！共计 {len(master_df)} 个有效观测点 (约 26站 × 7趟车)")

    # ================= 计算相关性 =================
    corr = master_df[['actual_time', 'plan_time', 'payload', 'energy']].corr()

    # ================= 绘图 =================
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", linewidths=1)
    plt.title(f"严格正向 507系列：能耗相关性矩阵\n({DATE_FOLDER})", fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, "strict_correlation_heatmap.png"), dpi=300)
    
    # 核心：能耗 vs 其他因子的散点回归图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    factors = [('actual_time', '实际运行时间 (s)'), 
               ('plan_time', '计划运行时间 (s)'), 
               ('payload', '有效载荷 (t)')]
    
    for i, (col, label) in enumerate(factors):
        sns.regplot(data=master_df, x=col, y='energy', ax=axes[i], 
                    scatter_kws={'alpha':0.5, 'color':'navy'}, line_kws={'color':'red'})
        r_val = corr.loc[col, 'energy']
        axes[i].set_title(f"{label} vs 能耗\nCorrelation: {r_val:.3f}")
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("能耗 (kWh)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "factor_regression_plots.png"), dpi=300)

    # ================= 输出结论 =================
    print("\n" + "="*50)
    print("📈 能耗 (Energy) 相关性深度分析：")
    print(f"1. 有效载荷: {corr.loc['payload', 'energy']:.4f} " + 
          ("(重量是能耗的关键因素)" if corr.loc['payload', 'energy'] > 0.6 else "(重量影响被其他因素掩盖)"))
    print(f"2. 实际时间: {corr.loc['actual_time', 'energy']:.4f} " + 
          ("(时间越短能耗越高，符合物理规律)" if corr.loc['actual_time', 'energy'] < 0 else "(相关性不符合预期，请检查样本)"))
    print(f"3. 计划时间: {corr.loc['plan_time', 'energy']:.4f}")
    print("="*50)
    print(f"✅ 图表已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_strict_analysis()