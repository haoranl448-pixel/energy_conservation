# -*- coding: utf-8 -*-
"""
主线第 7 步：将优化排图结果导出为时刻表。

输入：
- DP 排图输出 `schedule_table.csv`
- 区间静态参数 `section_params_trip5.csv`
- 能耗菜单 `ato_class_energy_menu1_new_v3.csv`

输出：
- `opentrack_timetable_import.csv`，便于 OpenTrack 或外部系统导入。
"""

# pandas：读取排图结果、参数表和能耗菜单，并导出 CSV。
import pandas as pd
# datetime/timedelta：把“运行秒数”转换为具体到达/出发时刻。
from datetime import datetime, timedelta
# Path：统一处理路径。
from pathlib import Path

# ================= 1. 路径配置 =================
# 项目根目录；在 scripts_new 二级目录直接运行时，这个值可能需要统一修正。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DP 输出的区间排图结果。
SCHEDULE_FILE = PROJECT_ROOT / "output" / "schedule" / "schedule_table.csv"
# 载重等区间参数表。
PARAMS_FILE = PROJECT_ROOT / "data" / "static" / "section_params_trip5.csv"
# 能耗菜单，用来反查某区间某运行时间对应的 ATO 等级。
MENU_FILE = PROJECT_ROOT / "output" / "analysis" / "ato_class_energy_menu1_new_v3.csv"
# 最终导出的时刻表文件。
OUTPUT_FILE = PROJECT_ROOT / "output" / "schedule" / "opentrack_timetable_import.csv"

# 起始发车时刻。
BASE_TIME_STR = "08:00:00"
# 默认停站时间，最后一站不再停站出发。
DEFAULT_DWELL = 30


def convert_format():
    """读取 DP 结果并转换为“站名-到达-出发-等级-载重”格式。"""

    # 缺少排图文件时无法继续。
    if not SCHEDULE_FILE.exists():
        print(f"ERROR: {SCHEDULE_FILE} not found")
        return
    # 缺少参数表时无法填充载重。
    if not PARAMS_FILE.exists():
        print(f"ERROR: {PARAMS_FILE} not found")
        return

    # 读取 DP 结果表。
    df_sched = pd.read_csv(SCHEDULE_FILE)
    # 读取区间参数表。
    df_params = pd.read_csv(PARAMS_FILE)
    # 清理站间区间名称空格，减少匹配失败。
    df_params['station_pair'] = df_params['station_pair'].str.strip()

    # 从能耗菜单建立反查表：(站间区间, 运行时长) -> 运行等级。
    class_map = {}
    if MENU_FILE.exists():
        df_menu = pd.read_csv(MENU_FILE)
        for _, r in df_menu.iterrows():
            # 站间区间名称。
            sp = str(r['站间区间']).strip()
            # 时间保留 0.1 秒，和 DP 结果的精度对齐。
            t = round(r['运行时长(s)'], 1)
            # 运行等级 class1-class5。
            c = r['运行等级']
            class_map[(sp, t)] = c

    def lookup_class(sp, travel_time):
        """按站间区间和运行时间反查运行等级。"""

        # 先做精确的 0.1 秒匹配。
        key = (sp, round(travel_time, 1))
        if key in class_map:
            return class_map[key]
        # 如果因为四舍五入略有差异，则允许 0.5 秒内的近似匹配。
        for (sp2, t2), c2 in class_map.items():
            if sp2 == sp and abs(t2 - round(travel_time, 1)) < 0.5:
                return c2
        # 找不到时标记 unknown，方便后续人工排查。
        return "unknown"

    # 初始化第一站：只有出发时刻，没有到达时刻。
    start_mass = df_params.iloc[0]['MASS']
    current_time = datetime.strptime(BASE_TIME_STR, "%H:%M:%S")

    # station_pair 形如“布政-张家潭”，第一站取左侧站名。
    first_station_name = str(df_sched.iloc[0]['station_pair']).split('-')[0]
    output_rows = [[
        first_station_name, "-", BASE_TIME_STR, "-", 0, start_mass, 0
    ]]

    # prev_mass 用来计算相邻区间载重变化。
    prev_mass = start_mass

    # 逐区间生成到达/出发时刻。
    for i, row in df_sched.iterrows():
        # 当前站间区间。
        sp = str(row['station_pair']).strip()
        # 当前区间运行时间，单位秒。
        run_time = float(row['t_arrival(s)'])

        # 从能耗菜单反查当前区间采用的运行等级。
        selected_class = lookup_class(sp, run_time)

        # 从参数表读取当前区间载重。
        match = df_params[df_params['station_pair'] == sp]
        if len(match) == 0:
            print(f"WARNING: {sp} not in params file, skipping")
            continue
        current_mass = float(match['MASS'].values[0])
        # 计算相对上一段的载重变化。
        mass_delta = round(current_mass - prev_mass, 3)

        # 当前时间推进一个区间运行时间，得到到达时刻。
        current_time += timedelta(seconds=run_time)
        arrival_str = current_time.strftime("%H:%M:%S")

        # 最后一段到站后不再出发；其他站增加默认停站时间。
        if i == len(df_sched) - 1:
            departure_str = "-"
            dwell = 0
        else:
            departure_str = (current_time + timedelta(seconds=DEFAULT_DWELL)).strftime("%H:%M:%S")
            dwell = DEFAULT_DWELL

        # 目标站名取 station_pair 右侧。
        dest_station = sp.split('-')[1]

        # 追加一行时刻表记录。
        output_rows.append([
            dest_station, arrival_str, departure_str,
            selected_class, int(run_time), current_mass, mass_delta
        ])

        # 如果不是最后一站，把停站时间累加进当前时间。
        if i < len(df_sched) - 1:
            current_time += timedelta(seconds=dwell)
        # 更新上一段载重。
        prev_mass = current_mass

    # 保存为 OpenTrack/外部系统更容易读的列结构。
    columns = ["站名", "到达时刻", "出发时刻", "运行等级", "区间运行时间(s)", "当前车重(t)", "重量变化(t)"]
    df_output = pd.DataFrame(output_rows, columns=columns)
    df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)
    print(df_output.to_string(index=False))


if __name__ == "__main__":
    # 直接运行脚本时执行格式转换。
    convert_format()
