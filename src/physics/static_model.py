import pandas as pd


def get_section_info(start_station, end_station, excel_path="线路基础数据（核查完毕）.xls"):
    # 读取Excel文件
    xls = pd.ExcelFile(excel_path)#.ExcelFile()读取xls文件

    # 读取各工作簿数据
    df_station = pd.read_excel(xls, sheet_name="车站和车辆基地表（station）")#.read_excel()明确指定了工作表名为 "车站和车辆基地表（station）"
                                                    ## 车站信息
    df_horizontal = pd.read_excel(xls, sheet_name="线路平面信息（lineAlignmentinfo）") # 平面（曲率）信息
    df_vertical = pd.read_excel(xls, sheet_name="线路纵断面信息表（lineProfileInfo)") # 纵断面（坡度）信息

    # 将平面和纵断面数据中的方向字段转换为标准的字符串格式（两位数）
    if "direction" in df_horizontal.columns:#检查数据框中是否包含名为direction的列
        df_horizontal["direction"] = df_horizontal["direction"].astype(str).str.zfill(2)#.str.zfill()对转换后的字符串执行补零操作，确保其长度为 2
    if "direction" in df_vertical.columns:
        df_vertical["direction"] = df_vertical["direction"].astype(str).str.zfill(2)

    # 按 inbound_mileage 升序排序车站（默认为正向顺序）
    df_station = df_station.sort_values("inbound_mileage").reset_index(drop=True)#排序  放弃原来的索引

    # 获取输入站在车站表中的索引
    try:
        start_idx = df_station.index[df_station["station_name"] == start_station].tolist()[0]#df_station["station_name"] == start_station：通过布尔索引筛选出车站名称等于输入起始站（start_station）的行
                                            #外层的df_station.index[]：获取这些筛选行的索引值（行号）
                                            #.tolist()：将索引值转换为列表形式，[0]：取列表中的第一个元素（因为理论上每个车站名称唯一）
        end_idx = df_station.index[df_station["station_name"] == end_station].tolist()[0]
    except IndexError:
        raise ValueError("未能在车站数据中找到对应的站名，请检查输入的站名是否正确。")

    # 检查是否为相邻站
    if abs(start_idx - end_idx) != 1:
        raise ValueError("请输入相邻的两个站！")

    # 判断运行方向：如果输入顺序与车站表顺序一致，则为 "01"，否则为 "02"
    if start_idx < end_idx:
        user_direction = "01"
    else:
        user_direction = "02"

    # 获取两站的里程（车站表中的里程为正向），并确保 s < e
    s = df_station.loc[start_idx, "inbound_mileage"]#df_station 是存储车站信息的 DataFrame
    e = df_station.loc[end_idx, "inbound_mileage"]
                            #loc 方法根据索引获取对应车站的里程值，分别赋值给 s（起始站里程）和 e（终点站里程）。
    if s > e:
        s, e = e, s

    # --------------------------
    # 处理水平（平面）信息
    # --------------------------
    horizontal_results = []#存储最终的平面信息结果
    current = s## 当前处理的里程起点（初始化为两站中的较小里程s）
    segs_h = df_horizontal[
        (df_horizontal["direction"] == user_direction) &
        (df_horizontal["curve_end_mile"] > s) &
        (df_horizontal["curve_start_mile"] < e)
        ].sort_values("curve_start_mile")# # 按曲线起点里程升序排序

    for idx, row in segs_h.iterrows():
        seg_start = max(current, row["curve_start_mile"])## 计算当前曲线段与目标区间的重叠部分起点（取当前进度和曲线起点的最大值）
        seg_end = min(e, row["curve_end_mile"])# # 计算重叠部分终点（取目标区间终点和曲线终点的最小值）
        # 若存在间隙，补充默认曲率半径为 10000 的段
        if seg_start > current:
            horizontal_results.append({
                "里程": f"{current}-{seg_start}",
                "曲率半径": 3000
            })#补充一段默认曲率半径为3000的线段（填补间隙）
        horizontal_results.append({
            "里程": f"{seg_start}-{seg_end}",
            "曲率半径": row["curve_radius"]
        })
        current = seg_end
        if current >= e:
            break
    if current < e:
        horizontal_results.append({
            "里程": f"{current}-{e}",
            "曲率半径": 3000
        })

    # --------------------------
    # 处理纵断面信息
    # --------------------------
    vertical_results = []# 存储最终的纵断面信息结果
    current_v = s# 当前处理的里程起点（初始化为两站中的较小里程s）
    segs_v = df_vertical[
        (df_vertical["direction"] == user_direction) &
        (df_vertical["grade_end_point_mileage"] > s) &
        (df_vertical["grade_start_point_mileage"] < e)
        ].sort_values("grade_start_point_mileage")

    for idx, row in segs_v.iterrows():
        seg_start = max(current_v, row["grade_start_point_mileage"])
        seg_end = min(e, row["grade_end_point_mileage"])
        vertical_results.append({
            "里程": f"{seg_start}-{seg_end}",
            "gradient": row["gradient"],
            "length": row["length"]
        })
        current_v = seg_end
        if current_v >= e:
            break

    # --------------------------
    # 如果运行方向为 "02"，则反转输出结果：
    #   1. 反转每段里程的字符串（例如 "100-105" 变为 "105-100"）
    #   2. 整个结果列表反转，以符合反向运行的显示顺序
    # --------------------------
    if user_direction == "02":
        def reverse_mileage(mileage_str):
            parts = mileage_str.split("-")
            return f"{parts[1]}-{parts[0]}"

        horizontal_results = [dict(r, 里程=reverse_mileage(r["里程"])) for r in horizontal_results]
        horizontal_results.reverse()
        vertical_results = [dict(r, 里程=reverse_mileage(r["里程"])) for r in vertical_results]
        vertical_results.reverse()

    df_horizontal_result = pd.DataFrame(horizontal_results)
    df_vertical_result = pd.DataFrame(vertical_results)

    return df_horizontal_result, df_vertical_result


if __name__ == "__main__":
    start_station = input("请输入起始站名称：")
    end_station = input("请输入终点站名称：")

    try:
        df_h, df_v = get_section_info(start_station, end_station)
        print("【水平（平面）信息】")
        print(df_h)
        print("\n【纵断面信息】")
        print(df_v)
    except Exception as e:
        print("处理过程中发生错误：", e)
