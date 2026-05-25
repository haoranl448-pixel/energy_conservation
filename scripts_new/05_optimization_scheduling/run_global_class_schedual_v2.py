# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ===================== 1. 全局调度参数（在此修改） =====================
# 设定全线总时长（含停站时间，单位：秒）
# 你可以根据需要修改这个数值，比如 3660.8
T_TOTAL_TARGET = 3660.8 

# 默认停站时间（单位：秒）
DWELL_TIME = 30.0

# ---------------------------------------------------------
# 🌟 逻辑 A：全局允许的等级范围 (例如：只准在 Class 2, 3, 4 中选)
# 如果想选全部，就写 ["class1", "class2", "class3", "class4", "class5"]
# ---------------------------------------------------------
GLOBAL_ALLOWED_CLASSES = [ "class2", "class3", "class4", "class5"]

# ---------------------------------------------------------
# 🌟 逻辑 B：特定站点强制约束 (优先级最高)
# 格式：{"区间名称": "运行等级"}
# ---------------------------------------------------------
MANUAL_CONSTRAINTS = {
    # "布政-张家潭": "class3", 
}

# ===================== 2. 路径与核心配置 =====================
PROJECT_ROOT = Path(r"D:\energy_conservation")
# 刚才评估出来的“能耗菜单”
MENU_FILE = PROJECT_ROOT / "output" / "analysis" / "ato_class_energy_menu.csv"
# 历史 Trip 6 的详细对标表（用于获取历史能耗基准）
HIST_COMPARE_FILE = PROJECT_ROOT / "output" / "trip6_final_report" / "Trip6_Detailed_Energy_Comparison.csv"

# 输出结果
OUT_DIR = PROJECT_ROOT / "output" / "schedule" / "final_ato_dp_results"
os.makedirs(OUT_DIR, exist_ok=True)

# 26个区间固定清单
STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ===================== 3. 核心寻优算法 =====================

def run_final_scheduling():
    # 1. 加载数据
    if not MENU_FILE.exists():
        print(f"❌ 找不到菜单文件: {MENU_FILE}")
        return
    df_menu = pd.read_csv(MENU_FILE)

    if not HIST_COMPARE_FILE.exists():
        print(f"❌ 找不到历史对比文件: {HIST_COMPARE_FILE}")
        return
    df_hist = pd.read_csv(HIST_COMPARE_FILE)

    # 2. 计算目标纯运行时间
    # 纯运行时间 = 总时长 - (区间数 * 停站时间)
    # 26个区间意味着有25个中间停站点
    total_dwell_time = (len(STATIONS) - 1) * DWELL_TIME
    t_run_target = T_TOTAL_TARGET - total_dwell_time
    
    if t_run_target <= 0:
        print(f"❌ 错误：总时长 {T_TOTAL_TARGET}s 不足以支付 {total_dwell_time}s 的停站时间！")
        return

    # 获取历史总能耗（从总计行获取）
    hist_total_energy = df_hist[df_hist['站间区间'].str.contains('总计')]['历史能耗(Wh)'].values[0]

    print("\n" + "="*50)
    print(f"📋 任务配置核实：")
    print(f"   - 全程总目标: {T_TOTAL_TARGET:.2f} s")
    print(f"   - 停站总时长: {total_dwell_time:.2f} s")
    print(f"   - 纯运行约束: {t_run_target:.2f} s")
    print(f"   - 历史基准能耗: {hist_total_energy/1000:.3f} kWh")
    print("="*50)

    # 3. DP 初始化 (0.1s 精度)
    def to_int(t): return int(round(t * 10))
    target_int = to_int(t_run_target)
    slack_int = 20 # 2秒容差
    
    dp = {0: 0.0}
    path = []

    # 4. 逐站寻优
    for i, sp in enumerate(STATIONS):
        new_dp = {}
        new_path = {}
        
        # 获取该站候选集
        all_options = df_menu[df_menu['站间区间'] == sp]
        
        # 应用过滤逻辑：手动固定 > 全局等级限制
        if sp in MANUAL_CONSTRAINTS:
            target_class = MANUAL_CONSTRAINTS[sp]
            options = all_options[all_options['运行等级'] == target_class]
            tag = "手动固定"
        else:
            options = all_options[all_options['运行等级'].isin(GLOBAL_ALLOWED_CLASSES)]
            tag = "全局筛选"

        if options.empty:
            print(f"❌ 错误：{sp} 在限定等级内无可用数据！")
            return

        for t_prev, e_prev in dp.items():
            for _, row in options.iterrows():
                t_curr = row['运行时长(s)']
                e_curr = row['预测能耗(Wh)']
                
                t_sum = t_prev + to_int(t_curr)
                # 剪枝范围：超过目标时间30秒就不再计算
                if t_sum > target_int + 300: continue
                
                e_sum = e_prev + e_curr
                if t_sum not in new_dp or e_sum < new_dp[t_sum]:
                    new_dp[t_sum] = e_sum
                    new_path[t_sum] = (t_prev, row['运行等级'], t_curr, e_curr, tag)
        
        dp = new_dp
        path.append(new_path)
        print(f"   [进度] {i+1:02d}/26 | {sp:15} | 状态空间: {len(dp)}")

    # 5. 寻找最优解
    feasible_times = [t for t in dp.keys() if abs(t - target_int) <= slack_int]
    if not feasible_times:
        # 宽限搜索：如果没有落入2秒内的，找全表最接近的一个
        best_t_int = min(dp.keys(), key=lambda x: abs(x - target_int))
        print(f"⚠️ 未能精准匹配目标时间，选择最接近时间: {best_t_int/10.0}s")
    else:
        best_t_int = min(feasible_times, key=lambda x: dp[x])

    # 6. 回溯时刻表
    best_schedule = []
    curr_t = best_t_int
    for i in range(len(STATIONS)-1, -1, -1):
        prev_t, c_name, t_val, e_val, tag = path[i][curr_t]
        best_schedule.append({
            "站间区间": STATIONS[i],
            "选定等级": c_name,
            "运行时长(s)": t_val,
            "预测能耗(Wh)": e_val,
            "逻辑来源": tag
        })
        curr_t = prev_t
    best_schedule.reverse()

    # 7. 保存与分析
    df_res = pd.DataFrame(best_schedule)
    save_path = OUT_DIR / "Optimized_Class_Schedule_Final.csv"
    df_res.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    final_e_sum = dp[best_t_int]
    saving_rate = (hist_total_energy - final_e_sum) / hist_total_energy * 100

    print("\n" + "*"*50)
    print(f"🏁 调度寻优圆满完成！")
    print(f"📊 历史能耗: {hist_total_energy/1000:.3f} kWh")
    print(f"📊 规划能耗: {final_e_sum/1000:.3f} kWh")
    print(f"🔥 综合节能率: {saving_rate:.2f} %")
    print(f"📄 结果已存至: {save_path}")
    print("*"*50)

if __name__ == "__main__":
    run_final_scheduling()