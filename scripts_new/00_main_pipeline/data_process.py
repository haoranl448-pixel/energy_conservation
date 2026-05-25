"""
主线第 1 步：原始动态/静态数据清洗。

这个脚本面向单个站间区间：
1. 扫描原始 Excel 文件。
2. 合并“动态”表。
3. 按时间连续性切分运行段 segment。
4. 计算电压、电流、功率、能耗、速度、加速度、累计位移。
5. 将线路静态坡度/曲率映射到每个采样点。
6. 从“静态”表中补充每段载重。
7. 导出 `results_站间区间.xlsx`。
"""

# pandas：读取/合并 Excel，处理表格数据。
import pandas as pd
# numpy：数值计算、空值和数组操作。
import numpy as np
# matplotlib：历史保留，当前主要用于设置中文字体。
import matplotlib.pyplot as plt
# cumtrapz/simpson：对速度积分得到累计位移。
from scipy.integrate import cumtrapz, simpson
# savgol_filter：平滑后求速度导数，得到加速度。
from scipy.signal import savgol_filter
# interp1d：把坡度/曲率按时间插值到动态采样点。
from scipy.interpolate import interp1d
# 读取线路静态模型：水平曲线和纵断面坡度。
from src.physics.static_model import get_section_info
# glob/os：扫描原始文件路径。
import glob
import os


def custom_fill(x):
    """
    对 Series x 进行自定义填充：
      - 将值为 0 的替换为 NaN，
      - 先前向填充，再检查若结果为 0 或 NaN，则使用后向填充，
      - 最后仍无数据的填充为 0。
    """
    # 将 0 当成缺失值处理，避免网压短暂掉 0 污染均值。
    x_repl = x.replace(0, np.nan)
    # 先尝试用前一个有效值填充。
    f = x_repl.ffill()
    # 再准备后一个有效值作为兜底。
    b = x_repl.bfill()
    # 前向填充仍为空或为 0 时，用后向填充替代。
    result = f.where(f.notna() & (f != 0), b)
    # 整段都没有有效值时，最终填 0。
    result = result.fillna(0)
    return result

def compute_derivative(x, y, window_length=11, polyorder=2):
    """用 Savitzky-Golay 滤波器计算一阶导数。"""

    # 数据点不足默认窗口时，将窗口缩小到可用的奇数长度。
    if len(x) < window_length:
        window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
    # 估计采样间隔；只有一个点时给默认间隔 1，避免除零。
    dx = np.mean(np.diff(x)) if len(x) > 1 else 1
    # deriv=1 表示输出一阶导数。
    dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=dx)
    return dy

def compute_cumulative_integral(x, y):
    """
    计算积分：若数据点较少则采用 cumtrapz，否则用 Simpson 积分法
    """
    # 初始化每个采样点对应的累计积分值。
    integral = np.zeros_like(y)
    # 点数过少时 Simpson 不稳定，使用梯形积分兜底。
    if len(x) < 3:
        integral[1:] = np.concatenate(([0], cumtrapz(y, x)))
    else:
        # 起点累计位移为 0。
        integral[0] = 0
        # 第二个点先用梯形积分。
        integral[1] = np.trapz(y[:2], x[:2])
        # 后续点用 Simpson 对 0 到当前点积分。
        for i in range(2, len(x)):
            integral[i] = simpson(y=y[:i + 1], x=x[:i + 1])
    return integral

def compute_total_integral(x, y):
    """计算整段总积分，常用于得到整段距离或总量。"""

    return simpson(y=y, x=x)

def parse_mileage(mileage_str):
    """解析“起点-终点”形式的里程字符串。"""

    # 里程字段形如 "0-122"，拆成起止两个数。
    parts = mileage_str.split("-")
    return float(parts[0]), float(parts[1])

def static_data_process(df_static, displacement_cum, time_arr, displacement_total):
    """
    对静态模型数据（水平或纵断面）进行处理：
      - 解析里程字符串，得到起始和结束里程；
      - 调整里程起点为 0，并按当前段的累计位移总长进行缩放；
      - 利用累计位移将里程映射为时间。
    """
    # 拆解静态数据中的里程区间。
    df_static['start_mileage'] = df_static['里程'].apply(lambda x: parse_mileage(x)[0])
    df_static['end_mileage'] = df_static['里程'].apply(lambda x: parse_mileage(x)[1])
    # 将里程起点调整为 0
    min_mileage = df_static['start_mileage'].min()
    df_static['start_mileage_adj'] = df_static['start_mileage'] - min_mileage
    df_static['end_mileage_adj'] = df_static['end_mileage'] - min_mileage
    # 原始静态数据覆盖的总长度。
    original_total = df_static['end_mileage_adj'].max()
    # 将静态里程缩放到当前动态数据实际位移长度。
    scale_factor = displacement_total / original_total if original_total != 0 else 1
    df_static['start_mileage_adj'] *= scale_factor
    df_static['end_mileage_adj'] *= scale_factor
    # 利用当前段内累计位移映射时间
    df_static['start_time'] = np.interp(df_static['start_mileage_adj'], displacement_cum, time_arr)
    df_static['end_time'] = np.interp(df_static['end_mileage_adj'], displacement_cum, time_arr)
    return df_static

if __name__ == "__main__":
    # 设置中文字体，避免图表中文乱码。
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # --------------------------
    # 1. 读取并预处理数据（多文件合并）
    # --------------------------

    # 当前要处理的起点和终点；批量化时可把这里改成循环或参数。
    start_station = '布政'
    end_station = '张家潭' 
    stations = f'{start_station}-{end_station}'
    # data_folder = "./raw_data"
    # pattern = os.path.join(data_folder, f"*{stations}.xlsx")
    # filenames = glob.glob(pattern)
    data_folder = "./raw_data"
    # 找到所有 05012-xxxx-xx-xx 这样的子目录。
    date_dirs   = sorted(glob.glob(os.path.join(data_folder, "05012-*")))
    filenames   = []
    for d in date_dirs:
        # 子目录名最后一段正好是日期，比如 “2024-07-08”。
        date_str = os.path.basename(d).split("-", 1)[-1]
        # 拼出当前日期、当前区间对应的 Excel 文件名。
        fn = os.path.join(d, f"{date_str}{start_station}-{end_station}.xlsx")
        if os.path.exists(fn):
            filenames.append(fn)
        else:
            print(f"警告：找不到文件 {fn}")


    # 输出目录和输出文件名。
    output_folder = "./data_processed"
    output_file = f"results_{stations}.xlsx"
    output_filename = os.path.join(output_folder, output_file)

    print(filenames)
    # 读取每个文件中 '动态' 工作表的数据，并将所有数据合并到一个 DataFrame 中。
    df_list = [pd.read_excel(f, sheet_name='动态') for f in filenames]
    df = pd.concat(df_list, ignore_index=True)

    # 转换“时刻”列为 datetime 类型，无法解析的设为 NaT。
    df['时刻'] = pd.to_datetime(df['时刻'].astype(str).str.strip(),
                                format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

    # --------------------------
    # 2. 分段处理（不跨段填充）
    # --------------------------
    # 相邻采样点时间差，单位秒。
    df['dt_overall'] = df['时刻'].diff().dt.total_seconds().fillna(0)
    # 正常采样间隔为 0.05 秒，超过则认为跨段，时间差置 0。
    df['dt_overall'] = df['dt_overall'].where(df['dt_overall'] <= 0.05, 0)
    # 每遇到一个非连续点，就开启新的 segment。
    df['segment'] = (df['dt_overall'] != 0.05).cumsum()

    # --------------------------
    # 3. 电压计算及分段内填充
    # --------------------------
    # 两路网压取均值作为电压。
    df['voltage'] = df[['CCU_LCU11 网压V', 'LCU61_CCU 网压V']].mean(axis=1)
    # 每个 segment 内独立填充电压，避免跨趟运行串值。
    df['voltage'] = df.groupby('segment')['voltage'].transform(custom_fill)

    # --------------------------
    # 4. 电流处理及整体计算
    # --------------------------
    current_cols = [
        'DCU1_CCU 正线电流 1=1AA',
        'DCU2_CCU 正线电流 1=1AA',
        'DCU3_CCU 正线电流 1=1AA',
        'DCU4_CCU 正线电流 1=1AA'
    ]
    # 负电流不作为牵引电流计入。
    for col in current_cols:
        df[col] = df[col].clip(lower=0)
    # 四路牵引电流求和。
    df['total_effective_current'] = df[current_cols].sum(axis=1)

    # --------------------------
    # 5. 功率、能耗与分段内累计能耗计算
    # --------------------------
    # 功率 = 电压 * 电流。
    df['power'] = df['voltage'] * df['total_effective_current']
    # 单步能量 = 功率 * 时间间隔。
    df['energy'] = df['power'] * df['dt_overall']
    # segment 内累计能耗。
    df['cumulative_energy'] = df.groupby('segment')['energy'].cumsum()
    # J 转 kWh：1 kWh = 3.6e6 J。
    df['cumulative_energy_kWh'] = df['cumulative_energy'] / 3.6e6

    # --------------------------
    # 6. 将每个分段内的时刻转换为相对时间（秒），从 0 开始
    # --------------------------
    # 每个 segment 的时刻从 0 秒重新开始。
    df['时刻'] = df.groupby('segment')['时刻'].transform(
        lambda x: (x - x.iloc[0]).dt.total_seconds()
    )

    # --------------------------
    # 7. 处理速度、加速度及累计位移
    # --------------------------
    # 速度：将 km/h 转换为 m/s
    df['速度(m/s)'] = df['BCU6_CCU 参考速度km/h'] * 1000 / 3600

    # 加速度：对每个分段内的相对时刻和速度计算导数
    def compute_segment_derivative(sub_df):
        """计算单个 segment 内的加速度序列。"""

        t = sub_df['时刻'].values
        v = sub_df['速度(m/s)'].values
        acc = compute_derivative(t, v)
        return pd.Series(acc, index=sub_df.index)

    df['加速度(m/s²)'] = df.groupby('segment').apply(
        lambda g: pd.Series(compute_segment_derivative(g).values, index=g.index)
    ).reset_index(level=0, drop=True)

    # 累计位移：对每个分段内计算积分（相对积分，段内独立）
    def compute_segment_disp(sub_df):
        """计算单个 segment 内的累计位移。"""

        t = sub_df['时刻'].values
        v = sub_df['速度(m/s)'].values
        disp = compute_cumulative_integral(t, v)
        return pd.Series(disp, index=sub_df.index)

    df['累计位移(m)'] = df.groupby('segment').apply(
        lambda g: pd.Series(compute_segment_disp(g).values, index=g.index)
    ).reset_index(level=0, drop=True)

    # --------------------------
    # 8. 静态模型映射（分段内独立计算水平和纵断面信息）
    # --------------------------
    # 获取静态模型数据（水平和纵断面），这里认为静态信息对各段相同，但映射时段内重新计算
    df_h, df_v = get_section_info(start_station, end_station)
    # 预先创建对应列
    df['curvature'] = np.nan
    df['gradient'] = np.nan

    # 对每个分段独立进行静态数据映射
    for seg, sub_df in df.groupby('segment', sort=False):
        # 使用分段内的相对时刻与累计位移
        t_seg = sub_df['时刻'].values
        disp_seg = sub_df['累计位移(m)'].values
        if len(t_seg) < 2:
            continue  # 数据点过少时跳过

        seg_disp_total = disp_seg[-1]

        # 水平截面（曲率）的映射
        df_h_proc_seg = static_data_process(df_h.copy(), disp_seg, t_seg, seg_disp_total)
        # 将每个静态区间的起止时间和值展开成插值点。
        h_times_seg = []
        h_curvature_seg = []
        for _, row in df_h_proc_seg.iterrows():
            h_times_seg.extend([row['start_time'], row['end_time']])
            h_curvature_seg.extend([row['曲率半径'], row['曲率半径']])
        if len(h_times_seg) >= 2:
            # 按时间插值，得到每个动态采样点的曲率半径。
            curvature_interp_seg = interp1d(h_times_seg, h_curvature_seg, fill_value="extrapolate")
            curvature_vals = curvature_interp_seg(t_seg)
            df.loc[sub_df.index, 'curvature'] = curvature_vals

        # 纵断面（坡度）的映射（假定静态数据字段名为 "gradient"）
        df_v_proc_seg = static_data_process(df_v.copy(), disp_seg, t_seg, seg_disp_total)
        # 将每个静态坡度区间展开成插值点。
        v_times_seg = []
        v_gradient_seg = []
        for _, row in df_v_proc_seg.iterrows():
            v_times_seg.extend([row['start_time'], row['end_time']])
            v_gradient_seg.extend([row['gradient'], row['gradient']])
        if len(v_times_seg) >= 2:
            # 按时间插值，得到每个动态采样点的坡度。
            gradient_interp_seg = interp1d(v_times_seg, v_gradient_seg, fill_value="extrapolate")
            gradient_vals = gradient_interp_seg(t_seg)
            df.loc[sub_df.index, 'gradient'] = gradient_vals

    # --------------------------
    # 新增：按文件和 segment 添加重量列，利用每个文件“静态”表中 “区间载荷（人重+车重）t” 列值填充
    # --------------------------
    # 读取每个文件中“静态”工作表的数据。
    static_df_list = [pd.read_excel(f, sheet_name='静态') for f in filenames]

    # 初始化重量列。
    df['重量'] = np.nan

    # 利用原来读取“动态”数据的列表获得各文件数据在合并后 df 中的行数边界。
    row_counts = [len(df_file) for df_file in df_list]
    start_idx = 0
    for i, count in enumerate(row_counts):
        end_idx = start_idx + count
        # 当前文件对应的动态数据子集（保持原有行序）
        df_subset = df.iloc[start_idx:end_idx]
        # 获取该文件的静态表
        static_df = static_df_list[i]
        # 获取当前子集中按出现顺序的各个 segment
        unique_segs = df_subset["segment"].unique()

        # 检查静态表中的行数是否与子集中的 segment 数量一致
        if len(unique_segs) != len(static_df):
            print(f"警告：文件 {filenames[i]} 中动态段数 ({len(unique_segs)}) 与静态表行数 ({len(static_df)}) 不一致！")

        # 按子集内出现顺序，遍历每个 segment
        for j, seg in enumerate(unique_segs):
            try:
                # 从当前文件的静态表中取第 j 行“区间载荷（人重+车重）t”作为该段的重量
                weight_value = static_df.iloc[j]["区间载荷（人重+车重）t"]
            except Exception as e:
                print(f"文件 {filenames[i]} 第 {j + 1} 条静态数据读取出错：", e)
                weight_value = np.nan

            # 找到该段在当前子集中的所有索引，并在全局 df 中赋值
            indices = df_subset.index[df_subset["segment"] == seg]
            df.loc[indices, "重量"] = weight_value

        # 移动到下一个文件在合并表中的起始行。
        start_idx = end_idx

    # --------------------------
    # 9. 保存结果到 Excel 文件
    # --------------------------
    df.to_excel(output_filename, index=False)
    print(f"数据已保存至 '{output_filename}'")

    #
    #
    # # --------------------------
    # # 9. 仅保留所需的列（可根据需求选择）
    # # --------------------------
    # # cols_to_keep = [
    # #     'global_time', '时刻', 'relative_time',
    # #     'CCU_LCU11 网压V', 'LCU61_CCU 网压V'
    # # ] + current_cols + [
    # #     'voltage',
    # #     'total_effective_current',
    # #     'power',
    # #     'dt_overall',
    # #     'energy',
    # #     'cumulative_energy_kWh',
    # #     'segment',
    # #     '速度(m/s)',
    # #     '加速度(m/s²)',
    # #     '累计位移(m)',
    # #     'curvature',
    # #     'gradient'
    # # ]
    # # df_calc = df[cols_to_keep].copy()
    #
    # # --------------------------
    # # 10. 保存结果到 Excel 文件
    # # --------------------------
    # output_filename = "替换时刻_相对时间.xlsx"
    # df.to_excel(output_filename, index=False)
    # print(f"数据已保存至 '{output_filename}'")
