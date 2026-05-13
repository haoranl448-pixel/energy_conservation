# -*- coding: utf-8 -*-
import pandas as pd
import os
import glob

# ================= 1. 路径与参数配置 =================
# 根据你提供的路径
FOLDER_PATH = r"D:\energy_conservation\data\0308\四天数据\DR330_12_01_2407091016_2407091135_149.mevt"

DT = 0.05  # 采样间隔 0.05s

# 目标行范围 (Excel里的行号)
# START_ROW = 36429
# END_ROW = 38765

# START_ROW = 39350
# END_ROW = 42382

# START_ROW = 42984
# START_ROW = 42959
# END_ROW = 45192

# START_ROW = 45880
# END_ROW = 47510

# START_ROW = 48082
# END_ROW = 50742


START_ROW = 51329
END_ROW = 53258



def calculate_specific_energy_v2():
    # 1. 寻找文件
    all_files = glob.glob(os.path.join(FOLDER_PATH, "*.xls"))
    xls_files = [f for f in all_files if not os.path.basename(f).startswith("~$")]
    
    if not xls_files:
        print(f"❌ 错误：找不到有效的 .xls 文件。")
        return
    
    file_path = xls_files[0]
    print(f"📂 目标文件: {os.path.basename(file_path)}")

    # 2. 读取第一个 Sheet (Sheet1)
    try:
        print(f"⏳ 正在读取第一个工作表 (Index 0)，请稍候...")
        # 指定读取第一个工作表，使用 openpyxl 引擎
        df_all = pd.read_excel(file_path, sheet_name=1, engine='openpyxl') 
        
        print(f"📊 该表共检测到 {len(df_all)} 行数据")

        # 3. 截取目标行区间
        # 表头占1行，Pandas索引从0开始。
        # Excel行号 36470 对应 Pandas 索引 36468
        # Excel行号 38765 对应 Pandas 索引 38763
        # iloc[start:end] 是左闭右开，所以 end 要取 38764
        df = df_all.iloc[START_ROW-2 : END_ROW-1].copy()
        
        if len(df) == 0:
            print(f"❌ 错误：截取后的数据为空！请核对文件行数（当前文件总行数：{len(df_all)}）。")
            return
            
        print(f"✅ 成功截取数据：Excel 第 {START_ROW} 行至 {END_ROW} 行 (共 {len(df)} 行)")

        # 4. 电流处理：J, K, L, M 列 (列索引分别是 9, 10, 11, 12)
        # J=9, K=10, L=11, M=12
        curr_cols = df.iloc[:, [9, 10, 11, 12]]
        total_current_raw = curr_cols.sum(axis=1)
        total_current = total_current_raw.clip(lower=0) 
        #total_current = curr_cols.sum(axis=1).abs()
        # 5. 电压处理：R, S 列 (列索引分别是 17, 18)
        # R=17, S=18
        volt_cols = df.iloc[:, [17, 18]]
        # 按照你的要求：向上补充（前向填充 ffill），然后取两列平均
        avg_voltage = volt_cols.ffill().mean(axis=1).ffill().fillna(0)
        
        # 6. 积分计算
        # 实时功率 P(W) = U(V) * I(A)
        # 实时能量 E(J) = P * dt
        energy_j_series = avg_voltage * total_current * DT
        total_joules = energy_j_series.sum()
        
        # 单位转换
        total_wh = total_joules / 3600.0   # 焦耳转瓦时
        total_kwh = total_wh / 1000.0      # 瓦时转度(kWh)

        # 7. 结果打印
        print("\n" + "="*45)
        print(f"📈 计算结果报告")
        print("-" * 45)
        print(f"🔹 统计行数:    {len(df)} 行")
        print(f"🔹 对应时长:    {round(len(df)*DT, 2)} 秒")
        print(f"🔹 平均电压:    {round(avg_voltage.mean(), 2)} V")
        print(f"🔹 最大电流:    {round(total_current.max(), 2)} A")
        print("-" * 45)
        print(f"🍀 总功(焦耳):  {round(total_joules, 2)} J")
        print(f"💎 消耗能量:    {round(total_wh, 2)} Wh")
        print(f"🔴 最终电耗:    {round(total_kwh, 4)} 度 (kWh)")
        print("="*45)

    except Exception as e:
        print(f"❌ 计算过程中出错: {e}")

if __name__ == "__main__":
    calculate_specific_energy_v2()