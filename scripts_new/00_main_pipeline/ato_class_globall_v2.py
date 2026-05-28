# -*- coding: utf-8 -*-
"""
ato_class_globall_v2.py

主线第 6 步：带弹性停站时间的 DP 排图优化器。

核心目标：
在总运行时间约束下，为每个站间区间选择一个 ATO 运行等级，使全线总能耗最低。

v2 相比 v1 的主要变化：
1. 每个停站点可以配置 nominal/min dwell，min==nominal 表示该站停站时间锁死。
2. DP 先选择区间运行时间和能耗组合，再把剩余时间分配给停站时间。
3. 若某组合在最小停站约束下不可行，则自动选择下一个可行组合。
"""
# pandas/numpy：读取能耗菜单、历史结果和做数值计算。
import pandas as pd
import numpy as np
# matplotlib：输出历史方案与优化方案的速度曲线对比图。
import matplotlib.pyplot as plt
# Path：统一处理路径。
from pathlib import Path
# os：读取主程序传入的趟号环境变量。
import os

# ===================== 1. Global config =====================
DEFAULT_T_TOTAL_TARGET = 692.65

def get_target_time(default_value):
    """读取主程序传入的 DP 目标总时间；没有传入时使用脚本默认值。"""

    raw = os.environ.get("ENERGY_TARGET_TIME")
    if not raw:
        return default_value
    value = float(raw)
    if value <= 0:
        raise ValueError("ENERGY_TARGET_TIME must be > 0.")
    return value


#T_TOTAL_TARGET = 694.7#trip1
# 全流程总时间目标，单位秒。默认配置对应 trip6/局部实验；主程序可用 --target-time 覆盖。
T_TOTAL_TARGET = get_target_time(DEFAULT_T_TOTAL_TARGET)#trip6/default
# nominal 时间附近用于展示对比的搜索窗口。
SLACK = 10
# 未单独配置站点的默认计划停站时间。
NOMINAL_DWELL = 30.0          # default dwell for stations not in config
# 未单独配置站点的默认最小停站时间。
MIN_DWELL = 23.0              # default min dwell for elastic stations

# 停站时间配置。未列出的站点使用上面的默认值。
# nominal：计划停站时间；min：最小允许停站时间；min==nominal 表示锁定停站。
STATION_DWELL_CONFIG = {
    # --- Locked stations (e.g. transfer stops): min == nominal ---
    # "三官堂-兴庄路":   {"nominal": 30, "min": 30},
    # "海晏北路-民安东路": {"nominal": 30, "min": 30},

    # --- Elastic stations (e.g. minor stops): min < nominal ---
    # "布政-张家潭":     {"nominal": 30, "min": 28},
    # "张家潭-同德路":     {"nominal": 30, "min": 28},
    # "同德路-石碶":       {"nominal": 30, "min": 28},
    # "石碶-雅渡":       {"nominal": 30, "min": 28},
    # "雅渡-庙堰":       {"nominal": 30, "min": 26},

}

# 手动约束某些区间必须选择指定等级，用于专项实验或业务要求。
MANUAL_CONSTRAINTS = {
    "泗港-曹隘": "class4"
}

# 全局允许参与 DP 的等级集合；不在该集合中的等级会被过滤掉。
GLOBAL_ALLOWED_CLASSES = ["class2", "class3","class4", "class5"]

# ===================== 2. Paths =====================
# 项目根目录；从 scripts_new 二级目录直接运行时需注意路径层级。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_trip_no() -> int:
    """从主程序传入的环境变量里读取要处理第几趟车。"""

    raw = os.environ.get("ENERGY_TRIP_NO", "1")
    trip_no = int(raw)
    if trip_no < 1:
        raise ValueError("ENERGY_TRIP_NO must be >= 1.")
    return trip_no


TRIP_NO = get_trip_no()
# 能耗菜单：每个区间每个等级的时间和预测能耗。
MENU_FILE = PROJECT_ROOT / "output" / "analysis" / f"ato_class_energy_menu{TRIP_NO}_new_v3.csv"
# 历史运行时间/模型回放能耗，用于最终对比节能率。
HIST_FILE = PROJECT_ROOT / f"full_line{TRIP_NO}_validation_results.csv"
# 生成曲线目录，用于最后画优化方案 v-t / v-s。
TRAJ_BASE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"
# 输出目录：最终对比表和图会保存到这里。
OUT_DIR = PROJECT_ROOT / "output" / "schedule" / "final_plan_report_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# 区间参数表，主要读取 MASS。
SECTION_PARAMS_FILE = PROJECT_ROOT / "data" / "static" / f"section_params_trip{TRIP_NO}.csv"
# 处理后的历史数据目录，用于绘制历史速度曲线。

def get_data_dir(default_value: Path) -> Path:
    """主程序指定测试数据目录时，从该目录读取历史 results_*.xlsx。"""

    raw = os.environ.get("ENERGY_DATA_DIR")
    if not raw:
        return default_value
    data_dir = Path(raw)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir


DATA_DIR = get_data_dir(PROJECT_ROOT / "data" / "data_processed")
# 位移修正比例，当前不做修正。
SLIP_RATIO = 1.0
# 选取历史数据中的第几个 segment 做对比。
TRIP_INDEX = TRIP_NO - 1

# 当前参与优化的站间区间列表。
# 注意：这里目前只启用前 5 个区间，其余区间被注释，适合局部验证。
STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    # "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    # "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", 
    # "曹隘-柳隘",
    # "柳隘-海晏北路",
    # "海晏北路-民安东路", 
    # "民安东路-会展中心", "会展中心-院士路",
    # "院士路-盎孟港", "盎孟港-三官堂", 
    #"三官堂-兴庄路",
    #  "兴庄路-兴海南路",
    # "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ===================== 3. Dwell helper =====================

def get_dwell_config(sp):
    """返回某个区间后停站点的 nominal/min 停站时间。"""

    # 如果单独配置过该区间，则使用配置值。
    if sp in STATION_DWELL_CONFIG:
        cfg = STATION_DWELL_CONFIG[sp]
        return cfg["nominal"], cfg["min"]
    # 未配置则使用全局默认停站时间。
    return NOMINAL_DWELL, MIN_DWELL


def distribute_dwell_delta(run_time_sum, dwell_configs):
    """
    根据 DP 选出的总运行时间，分配停站时间。

    目标：
        run_time_sum + sum(actual_dwells) = T_TOTAL_TARGET

    dwell_configs:
        每个停站点的 (nominal, min)，长度通常为区间数 - 1。
    返回：
        (success, dwell_list)，success 表示是否满足最小停站约束。
    """
    # 没有中间停站时，不需要分配 dwell。
    n_dwell = len(dwell_configs)
    if n_dwell == 0:
        return True, []

    # 拆出 nominal 和 min 数组，便于向量化计算。
    nominals = np.array([d[0] for d in dwell_configs], dtype=float)
    mins = np.array([d[1] for d in dwell_configs], dtype=float)

    # 当前运行时间下，总停站时间必须补足到 T_TOTAL_TARGET。
    target_dwell_total = T_TOTAL_TARGET - run_time_sum
    nominal_total = nominals.sum()

    # 如果目标停站总时间比所有 min 之和还小，说明运行时间太长，不可行。
    if target_dwell_total < mins.sum():
        return False, None  # infeasible: running too long even with min dwell

    # delta < 0 表示需要压缩停站；delta > 0 表示可以增加停站。
    delta = target_dwell_total - nominal_total
    # 每个停站点可压缩的空间。
    slacks = nominals - mins  # how much each station can give

    # 总可压缩停站空间。
    total_slack = slacks.sum()
    if total_slack <= 1e-9:
        # 所有站都锁定时，只有目标停站总时间接近 nominal 才可行。
        return abs(delta) < 1.0, nominals.tolist()

    # 按各站可压缩空间比例分配 delta。
    dwell = nominals + delta * (slacks / total_slack)

    # 如果是压缩停站，不能低于 min。
    if delta <= 0:
        dwell = np.maximum(dwell, mins)
    else:
        # 如果是增加时间，则按比例加到 nominal 之上。
        dwell = nominals + delta * (slacks / total_slack)

    # 最后一层保险，确保不低于 min。
    dwell = np.maximum(dwell, mins)

    # 保留 0.1 秒精度。
    return True, np.round(dwell, 1).tolist()


# ===================== 4. Core DP (unchanged from v1) =====================

def run_optimization():
    """执行 DP 排图优化，并输出对比表和可视化图。"""

    # A. 读取输入数据。
    print(f"Trip: {TRIP_NO} (segment index {TRIP_INDEX})")
    print(f"Target total time: {T_TOTAL_TARGET:.2f}s")
    print(f"Menu file: {MENU_FILE}")
    print(f"History file: {HIST_FILE}")
    print(f"Historical curve data dir: {DATA_DIR}")
    df_menu = pd.read_csv(MENU_FILE)
    df_hist = pd.read_csv(HIST_FILE).set_index('站间区间')
    df_mass = pd.read_csv(SECTION_PARAMS_FILE)
    # 构建区间 -> 载重映射，最终报告里展示 MASS。
    mass_map = dict(zip(df_mass['station_pair'], df_mass['MASS']))

    # B. 构建停站配置。
    dwell_configs = [get_dwell_config(sp) for sp in STATIONS]
    dwell_mins = [d[1] for d in dwell_configs]
    dwell_nominals = [d[0] for d in dwell_configs]
    # 最后一段运行后没有停站，所以停站统计排除最后一个区间。
    min_total_dwell = sum(dwell_mins[:-1])       # no dwell after last run
    nominal_total_dwell = sum(dwell_nominals[:-1])
    # 在最小停站条件下，区间运行总时长最多能有多少。
    max_run_time = T_TOTAL_TARGET - min_total_dwell  # longest running we can afford
    # nominal 停站条件下对应的运行总时间，用于对比展示。
    nom_run_time = T_TOTAL_TARGET - nominal_total_dwell  # nominal for display

    # C. DP 搜索：以 0.1 秒为单位离散化时间。
    def to_int(t): return int(round(t * 10))
    max_run_int = to_int(max_run_time)

    # dp[t] 表示当前已处理区间到累计运行时间 t 时的最小能耗。
    dp = {0: 0.0}
    # path[i][t] 保存第 i 个区间达到时间 t 的前驱状态，用于最后回溯方案。
    path = []

    print(f"Max run (at min dwell): {max_run_time:.1f}s")
    print(f"Nom run (at nominal):   {nom_run_time:.1f}s")
    print(f"Dwell budget to trade:   {nominal_total_dwell - min_total_dwell:.0f}s")
    print(f"Dwell config ({len(STATIONS)-1} intervals):")
    for sp in STATIONS:
        n, m = get_dwell_config(sp)
        tag = "LOCKED" if n == m else f"elastic ({m:.0f}-{n:.0f}s)"
        print(f"  {sp}: nominal={n:.0f}s, min={m:.0f}s [{tag}]")

    # 逐个区间做状态转移。
    for i, sp in enumerate(STATIONS):
        # new_dp/new_path 保存加入当前区间后的状态。
        new_dp, new_path = {}, {}
        # 读取当前区间所有可选等级。
        all_opts = df_menu[df_menu['站间区间'] == sp]

        # 若该区间有手动约束，只保留指定等级。
        if sp in MANUAL_CONSTRAINTS:
            options = all_opts[all_opts['运行等级'] == MANUAL_CONSTRAINTS[sp]]
        else:
            # 否则保留全局允许的等级。
            options = all_opts[all_opts['运行等级'].isin(GLOBAL_ALLOWED_CLASSES)]

        # 遍历上一轮所有可达状态。
        for t_prev, e_prev in dp.items():
            # 尝试当前区间的每个等级选择。
            for _, row in options.iterrows():
                # 当前等级的运行时长，保留 0.1 秒。
                t_curr = round(row['运行时长(s)'], 1)
                # 新累计运行时间。
                t_sum = t_prev + to_int(t_curr)
                # 超过最大可承受运行时间太多时剪枝。
                if t_sum > max_run_int + 200:
                    continue
                # 新累计能耗。
                e_sum = e_prev + row['预测能耗(Wh)']
                # 若该时间状态首次出现，或能耗更低，则更新最优值。
                if t_sum not in new_dp or e_sum < new_dp[t_sum]:
                    new_dp[t_sum] = e_sum
                    # 记录前驱：上一时间、选择等级、当前时长、当前能耗。
                    new_path[t_sum] = (t_prev, row['运行等级'], t_curr, row['预测能耗(Wh)'])

        # 当前区间处理完毕，进入下一轮。
        dp, path = new_dp, path + [new_path]
        print(f"  [{i+1}/{len(STATIONS)}] {sp} -> {len(dp)} states")

    # D. Select best: among ALL feasible states, pick minimum energy
    #    Feasible: run_time + min_dwell_sum <= T_TOTAL_TARGET
    feasible = [(t, dp[t]) for t in dp.keys() if t <= max_run_int]
    if not feasible:
        print("ERROR: No combination fits even with min dwell.")
        min_r = min(dp.keys()) / 10.0
        max_r = max(dp.keys()) / 10.0
        print(f"Reachable run time: [{min_r:.1f}s, {max_r:.1f}s], max allowed: {max_run_time:.1f}s")
        return

    # 所有可行状态按能耗从低到高排序。
    feasible.sort(key=lambda x: x[1])  # sort by energy ascending
    # 选择能耗最低的累计运行时间。
    best_t_int, final_energy = feasible[0]

    # 用停站时间补齐目标总时间。
    best_dwells = distribute_dwell_delta(best_t_int / 10.0, dwell_configs[:-1])[1]

    # Show the energy-vs-time tradeoff
    print(f"\nFeasible states: {len(feasible)}")
    print(f"Best:  run={best_t_int/10.0:.1f}s, energy={final_energy:.1f} Wh")

    # Find the closest-to-nominal state for comparison
    nom_int = to_int(nom_run_time)
    # Search for nearest feasible state near nominal
    nearby = [(t, dp[t]) for t in dp.keys() if nom_int - to_int(SLACK) <= t <= nom_int + to_int(SLACK)]
    if nearby:
        nearby.sort(key=lambda x: x[1])  # min energy near nominal
        nom_best_t, nom_best_e = nearby[0]
        print(f"Nominal-dwell best nearby: run={nom_best_t/10.0:.1f}s, energy={nom_best_e:.1f} Wh")
        saving = nom_best_e - final_energy
        extra_run = (best_t_int - nom_best_t) / 10.0
        print(f"Dwell trade: +{extra_run:.1f}s run time -> saves {saving:.1f} Wh ({saving/nom_best_e*100:.1f}%)")

    final_energy = dp[best_t_int]
    run_sum = best_t_int / 10.0

    # E. 回溯最优路径：从最后一个区间倒推每个区间选了哪个等级。
    final_rows = []
    curr_t = best_t_int
    for i in range(len(STATIONS) - 1, -1, -1):
        # path[i][curr_t] 记录了到达 curr_t 的前驱状态。
        prev_t, c_name, t_val, e_val = path[i][curr_t]
        sp = STATIONS[i]
        # 历史数据用于最终节能对比。
        # “历史能耗(Wh)”是历史运行曲线经同一套模型回放得到的能耗，不是原始实测能耗。
        h_time = df_hist.loc[sp, '历史运行时间(s)']
        h_energy = df_hist.loc[sp, '历史能耗(Wh)']

        # 最后一段后没有停站；其他区间后使用分配好的停站时间。
        if i < len(STATIONS) - 1:
            dwell = best_dwells[i]
        else:
            dwell = 0.0

        # 追加当前区间的最终报告行。
        final_rows.append({
            "站间区间": sp,
            "选定等级": c_name,
            "规划用时(s)": round(t_val, 0),
            "停站时间(s)": round(dwell, 1) if i < len(STATIONS) - 1 else 0,
            "历史用时(s)": round(h_time, 2),
            "规划能耗(Wh)": round(e_val, 2),
            "历史能耗(Wh)": round(h_energy, 2),
            "节能量(Wh)": round(h_energy - e_val, 2),
            "MASS": round(mass_map.get(sp, np.nan), 2),
        })
        # 回到前驱累计时间，继续向前回溯。
        curr_t = prev_t

    # 回溯得到的是倒序，需要翻转成线路正向顺序。
    final_rows.reverse()

    # 修正回溯后按整数秒展示导致的累计时间漂移。
    run_exact = best_t_int / 10.0
    run_backtrack_sum = sum(r['规划用时(s)'] for r in final_rows)
    drift = run_exact - run_backtrack_sum
    if abs(drift) > 0.01:
        # 将 0.1 秒级漂移分散到前几个区间。
        n_adj = int(round(abs(drift) * 10))
        step = 1 if drift > 0 else -1
        for j in range(n_adj):
            final_rows[j % len(final_rows)]['规划用时(s)'] += step * 0.1
        for r in final_rows:
            r['规划用时(s)'] = round(r['规划用时(s)'], 0)

    df_res = pd.DataFrame(final_rows)

    # F. 汇总历史/规划能耗和时间，计算节能率。
    total_h_e = df_res['历史能耗(Wh)'].sum()
    total_p_e = df_res['规划能耗(Wh)'].sum()
    total_p_run = df_res['规划用时(s)'].sum()
    total_p_dwell = df_res['停站时间(s)'].sum()
    total_p_t = total_p_run + total_p_dwell
    total_h_t = df_res['历史用时(s)'].sum() + nominal_total_dwell  # fixed historical dwell
    # 节能率 = (历史能耗 - 规划能耗) / 历史能耗。
    saving_rate = (total_h_e - total_p_e) / total_h_e * 100

    # 在结果表最后追加总计行。
    summary_row = {
        "站间区间": "--- 总计 ---",
        "选定等级": f"节能率: {saving_rate:.2f}%",
        "规划用时(s)": total_p_run,
        "停站时间(s)": total_p_dwell,
        "历史用时(s)": total_h_t,
        "规划能耗(Wh)": round(total_p_e, 2),
        "历史能耗(Wh)": round(total_h_e, 2),
        "节能量(Wh)": round(total_h_e - total_p_e, 2),
        "MASS": round(df_res['MASS'].sum(), 2),
    }
    df_res = pd.concat([df_res, pd.DataFrame([summary_row])], ignore_index=True)
    # 导出最终方案对比 CSV。
    df_res.to_csv(OUT_DIR / "Final_Planning_Comparison.csv", index=False, encoding='utf-8-sig')

    print(f"\nRun: {total_p_run:.1f}s | Dwell: {total_p_dwell:.1f}s | Total: {total_p_t:.1f}s (target: {T_TOTAL_TARGET}s)")
    print(f"Energy: {total_p_e:.1f} Wh | Saving: {saving_rate:.2f}%")
    for i, row in enumerate(final_rows):
        if i >= len(final_rows):
            break
        d = row['停站时间(s)']
        if d > 0:
            _, min_d = dwell_configs[i]  # config for this section controls dwell at its destination
            tag = f" -> min={min_d:.0f}s" if d <= min_d + 0.5 else ""
            print(f"  Dwell after {row['站间区间']}: {d:.1f}s{tag}")

    # ===================== 4. Plotting =====================
    # 绘制优化方案与历史方案的 v-t / v-s 对比图。
    print("\nPlotting full-line comparison ...")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 这四个累计变量用于把多个区间拼成连续的全线曲线。
    t_opt_acc = 0.0
    s_opt_acc = 0.0
    t_hist_acc = 0.0
    s_hist_acc = 0.0
    # plot_data 用列表累加所有区间的时间、速度、距离。
    plot_data = {'opt_t': [], 'opt_v': [], 'opt_s': [],
                 'hist_t': [], 'hist_v': [], 'hist_s': []}
    # dwell_zones 记录停站时间段，用于在图上做浅色背景标记。
    dwell_zones = []

    # 遍历最终方案，把每个区间的生成曲线和历史曲线拼起来。
    for i, row in enumerate(final_rows):
        if i >= len(STATIONS):
            break
        sp = row['站间区间']
        c_name = row['选定等级']

        # 优化轨迹：读取选定等级对应的 generated_curve.csv。
        df_opt = pd.read_csv(TRAJ_BASE_DIR / sp / f"{c_name}_generated_curve.csv")
        plot_data['opt_t'].extend((df_opt['time_s'] + t_opt_acc).tolist())
        plot_data['opt_v'].extend((df_opt['velocity_mps'] * 3.6).tolist())
        plot_data['opt_s'].extend((df_opt['dist_m'] + s_opt_acc).tolist())

        # 历史轨迹：读取处理后的真实运行数据，并选定一个 segment。
        df_h_all = pd.read_excel(DATA_DIR / f"results_{sp}.xlsx")
        target_seg = sorted(df_h_all['segment'].unique())[TRIP_INDEX]
        df_h = df_h_all[df_h_all['segment'] == target_seg].copy()
        h_v = df_h['速度(m/s)'].values * 3.6
        h_t = df_h['时刻'].values - df_h['时刻'].iloc[0]
        h_s = df_h['累计位移(m)'].values / SLIP_RATIO

        plot_data['hist_t'].extend((h_t + t_hist_acc).tolist())
        plot_data['hist_v'].extend(h_v.tolist())
        plot_data['hist_s'].extend((h_s + s_hist_acc).tolist())

        # 更新优化方案和历史方案的累计时间/距离。
        t_opt_acc += row['规划用时(s)']
        s_opt_acc += df_opt['dist_m'].iloc[-1]
        t_hist_acc += h_t[-1]
        s_hist_acc += h_s[-1]

        # 区间之间插入停站段，速度为 0，距离不变。
        if i < len(STATIONS) - 1:
            dwell_opt = row['停站时间(s)']
            dwell_hist = dwell_nominals[i]  # historical uses nominal 30s

            dwell_zones.append((t_opt_acc, t_opt_acc + dwell_opt))
            plot_data['opt_t'].extend([t_opt_acc, t_opt_acc + dwell_opt])
            plot_data['opt_v'].extend([0, 0])
            plot_data['opt_s'].extend([s_opt_acc, s_opt_acc])
            t_opt_acc += dwell_opt

            plot_data['hist_t'].extend([t_hist_acc, t_hist_acc + dwell_hist])
            plot_data['hist_v'].extend([0, 0])
            plot_data['hist_s'].extend([s_hist_acc, s_hist_acc])
            t_hist_acc += dwell_hist

    # 创建两行子图：第一行 v-t，第二行 v-s。
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    ax1.plot(plot_data['hist_t'], plot_data['hist_v'], color='gray', alpha=0.4,
             linewidth=1.0, label='historical')
    ax1.plot(plot_data['opt_t'], plot_data['opt_v'], color='red', linewidth=1.2,
             label='optimized plan')
    for start, end in dwell_zones:
        ax1.axvspan(start, end, color='gray', alpha=0.05)
    ax1.set_title(f"v-t | total={total_p_t:.1f}s (target={T_TOTAL_TARGET}s) | saving={saving_rate:.2f}%")
    ax1.set_ylabel("Velocity (km/h)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.plot(plot_data['hist_s'], plot_data['hist_v'], color='gray', alpha=0.4,
             linewidth=1.0, label='historical')
    ax2.plot(plot_data['opt_s'], plot_data['opt_v'], color='blue', linewidth=1.2,
             label='optimized plan')
    ax2.set_title(f"v-s | distance={s_opt_acc:.0f}m")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Velocity (km/h)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    # 保存图像报告。
    fig.savefig(OUT_DIR / "Optimized_Full_Line_Report.png", dpi=300)
    plt.close()
    print(f"Saved: {OUT_DIR / 'Optimized_Full_Line_Report.png'}")
    print(f"Done. Energy: {total_p_e/1000:.3f} kWh, Time: {total_p_t:.1f}s (strict = {T_TOTAL_TARGET}s)")


if __name__ == "__main__":
    # 直接运行脚本时执行 DP 排图优化。
    run_optimization()
