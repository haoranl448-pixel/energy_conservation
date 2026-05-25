# -*- coding: utf-8 -*-
import os
import pandas as pd
from datetime import datetime

# ================= 1. 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 输入：之前生成的 31 个历史基础 CSV 文件夹
INPUT_DIR = os.path.join(project_root, "output", "schedule")
# 输出：XML 时刻表文件夹
OUTPUT_DIR = os.path.join(project_root, "output", "schedule", "trip6_class_results_mlp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_xml_entry(station_idx, row, is_first, is_last):
    """
    生成单个站点的 XML 节点字符串
    """
    # 站点 ID 映射为 STA_1, STA_2...
    sta_id = f"STA_{station_idx + 1}"
    stop_info = "no" if is_first else "yes"
    
    # 时间处理
    arr_time = row['到达时刻'] if row['到达时刻'] != "-" else "HH:MM:SS"
    dep_time = row['出发时刻'] if row['出发时刻'] != "-" else "HH:MM:SS"
    
    # 停站时间逻辑
    wait_time = 30 if not (is_first or is_last) else 0
    
    # 载重变化 (对应你 CSV 里的重量变化(t))
    delta_load = f"\n\t\t\t<deltaLoad>{row['重量变化(t)'] / 10.0:.3f}</deltaLoad>" if row['重量变化(t)'] != 0 else ""

    entry = f"""		<timetableEntry stopInformation="{stop_info}">
			<stationID>{sta_id}</stationID>
			<arrivalTime format="hh:mm:ss" type="planned" valid="yes">{arr_time}</arrivalTime>
			<departureTime format="hh:mm:ss" type="planned" useDepartureTime="yes" valid="yes">{dep_time}</departureTime>
			<waitTime format="s">{wait_time}</waitTime>
			<delayTime format="s">0</delayTime>{delta_load}
		</timetableEntry>"""
    return entry

def convert_csv_to_opentrack_xml():
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ 找不到原始 CSV 文件，请确认先运行了生成时刻表的脚本。")
        return

    print(f"🚀 开始转换 {len(csv_files)} 个时刻表为 OpenTrack XML 格式...")

    for file_name in csv_files:
        path = os.path.join(INPUT_DIR, file_name)
        df = pd.read_csv(path)
        
        target_seconds = file_name.split('_')[-1].replace('s.csv', '')
        
        # XML 头部
        xml_content = f"""<?xml version="1.0"?>
<?xml-stylesheet?>
<!DOCTYPE timetable SYSTEM "timetable.dtd">
<timetable title="OpenTrack timetable" application="OpenTrack" date="{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}">
	<course>
		<courseID>Course_{target_seconds}</courseID>"""

        # 遍历每一站生成 Entry
        num_stations = len(df)
        for i, row in df.iterrows():
            is_first = (i == 0)
            is_last = (i == num_stations - 1)
            xml_content += "\n" + create_xml_entry(i, row, is_first, is_last)

        # XML 尾部
        xml_content += """
	</course>
</timetable>"""

        # 保存文件
        output_file_name = file_name.replace('.csv', '.xml')
        output_path = os.path.join(OUTPUT_DIR, output_file_name)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

    print(f"\n✨ 转换成功！共生成 {len(csv_files)} 个 XML 文件。")
    print(f"📁 路径: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_csv_to_opentrack_xml()