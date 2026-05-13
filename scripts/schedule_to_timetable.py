# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ================= 1. 路径配置 =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEDULE_FILE = PROJECT_ROOT / "output" / "schedule" / "schedule_table.csv"
PARAMS_FILE = PROJECT_ROOT / "data" / "static" / "section_params_trip5.csv"
MENU_FILE = PROJECT_ROOT / "output" / "analysis" / "ato_class_energy_menu1_new_v3.csv"
OUTPUT_FILE = PROJECT_ROOT / "output" / "schedule" / "opentrack_timetable_import.csv"

BASE_TIME_STR = "08:00:00"
DEFAULT_DWELL = 30


def convert_format():
    if not SCHEDULE_FILE.exists():
        print(f"ERROR: {SCHEDULE_FILE} not found")
        return
    if not PARAMS_FILE.exists():
        print(f"ERROR: {PARAMS_FILE} not found")
        return

    df_sched = pd.read_csv(SCHEDULE_FILE)
    df_params = pd.read_csv(PARAMS_FILE)
    df_params['station_pair'] = df_params['station_pair'].str.strip()

    # Build class lookup from energy menu (if available)
    class_map = {}
    if MENU_FILE.exists():
        df_menu = pd.read_csv(MENU_FILE)
        for _, r in df_menu.iterrows():
            sp = str(r['站间区间']).strip()
            t = round(r['运行时长(s)'], 1)
            c = r['运行等级']
            class_map[(sp, t)] = c

    def lookup_class(sp, travel_time):
        """Find class by matching station and rounded travel time."""
        key = (sp, round(travel_time, 1))
        if key in class_map:
            return class_map[key]
        # Try nearest match within 0.5s
        for (sp2, t2), c2 in class_map.items():
            if sp2 == sp and abs(t2 - round(travel_time, 1)) < 0.5:
                return c2
        return "unknown"

    # --- Init first station ---
    start_mass = df_params.iloc[0]['MASS']
    current_time = datetime.strptime(BASE_TIME_STR, "%H:%M:%S")

    first_station_name = str(df_sched.iloc[0]['station_pair']).split('-')[0]
    output_rows = [[
        first_station_name, "-", BASE_TIME_STR, "-", 0, start_mass, 0
    ]]

    prev_mass = start_mass

    # --- Iterate sections ---
    for i, row in df_sched.iterrows():
        sp = str(row['station_pair']).strip()
        run_time = float(row['t_arrival(s)'])

        # Look up class from menu
        selected_class = lookup_class(sp, run_time)

        # Mass from params
        match = df_params[df_params['station_pair'] == sp]
        if len(match) == 0:
            print(f"WARNING: {sp} not in params file, skipping")
            continue
        current_mass = float(match['MASS'].values[0])
        mass_delta = round(current_mass - prev_mass, 3)

        # Arrival time
        current_time += timedelta(seconds=run_time)
        arrival_str = current_time.strftime("%H:%M:%S")

        # Departure time
        if i == len(df_sched) - 1:
            departure_str = "-"
            dwell = 0
        else:
            departure_str = (current_time + timedelta(seconds=DEFAULT_DWELL)).strftime("%H:%M:%S")
            dwell = DEFAULT_DWELL

        dest_station = sp.split('-')[1]

        output_rows.append([
            dest_station, arrival_str, departure_str,
            selected_class, int(run_time), current_mass, mass_delta
        ])

        if i < len(df_sched) - 1:
            current_time += timedelta(seconds=dwell)
        prev_mass = current_mass

    # --- Save ---
    columns = ["站名", "到达时刻", "出发时刻", "运行等级", "区间运行时间(s)", "当前车重(t)", "重量变化(t)"]
    df_output = pd.DataFrame(output_rows, columns=columns)
    df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)
    print(df_output.to_string(index=False))


if __name__ == "__main__":
    convert_format()
