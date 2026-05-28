"""批量版: 处理所有指定相邻站区间的数据处理脚本

修改版说明：
1. 路径改为基于脚本所在目录自动读取，避免在 PowerShell 当前目录不同导致找不到文件。
2. 输出文件统一保存到 data_processed_v2 文件夹，避免覆盖旧结果。
3. 自动生成所有指定站间区间的正向和反向。
4. 删除“启动点之间必须间隔 >100 行”的过滤逻辑。
5. v2启停配对改为“每个停车点最多匹配一个启动点”，避免同一到达点被多个启动点复用。
6. 在PIS匹配后增加“每一区间最低运行时间”过滤，只过滤明显过短段，不设置最高时间。
7. 在PIS匹配后增加“位移积分/站间距”过滤：v2默认 0.85~1.30，避免切点偏差误删真实段。
8. PIS匹配改为按时间顺序确认 起点 -> 终点，并用ATC上下行信号校验方向。
9. 输出采用临时文件校验后替换，避免生成没有worksheet的坏Excel。
10. 启动点检测新增兜底逻辑：
   - 优先找 电流 ≤0 -> >0；
   - 再找 电流 <0 -> >0；
   - 如果前面都找不到，再向前找最近的 电流 <0 -> 0 点作为启动点。
"""

import os
import pickle
import warnings
import zipfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置：所有文件默认放在 process_target.py 同级目录
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LINE_XLS_PATH = os.path.join(BASE_DIR, "线路基础数据（核查完毕）.xls")
RAW_DIR = os.path.join(BASE_DIR, "0308", "四天数据")
CACHE_FILE = os.path.join(BASE_DIR, "_all_raw_cache.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(LINE_XLS_PATH):
    raise FileNotFoundError(
        f"找不到线路基础数据文件：{LINE_XLS_PATH}\n"
        f"请确认“线路基础数据（核查完毕）.xls”和本脚本放在同一个文件夹下。"
    )

if not os.path.isdir(RAW_DIR):
    raise FileNotFoundError(
        f"找不到原始数据目录：{RAW_DIR}\n"
        f"请确认目录结构为：脚本所在目录/0308/四天数据"
    )


# ============================================================
# 1. 加载线路数据
# ============================================================
print("加载线路数据...")

line_xls = pd.ExcelFile(LINE_XLS_PATH)

stations_df = pd.read_excel(line_xls, sheet_name="车站和车辆基地表（station）")
stations_df = (
    stations_df[stations_df["station_type"] == "STATION"]
    .sort_values("inbound_mileage")
    .copy()
)
stations_df["pis_order"] = range(1, len(stations_df) + 1)

station_id_to_name = dict(zip(stations_df["station_id"], stations_df["station_name"]))
station_mileage = dict(zip(stations_df["station_id"], stations_df["inbound_mileage"]))
pis_to_station = dict(zip(stations_df["pis_order"], stations_df["station_id"]))
station_name_to_pis = dict(zip(stations_df["station_name"], stations_df["pis_order"]))


def find_station_id(name):
    for sid, sname in station_id_to_name.items():
        if sname == name:
            return sid
    return None


sections_df = pd.read_excel(line_xls, sheet_name="区间表（section）")
sections_df = sections_df[sections_df["section_type"] == "MAIN"].copy()

curve_df = pd.read_excel(line_xls, sheet_name="线路平面信息（lineAlignmentinfo）")
grade_df = pd.read_excel(line_xls, sheet_name="线路纵断面信息表（lineProfileInfo)")


# ============================================================
# 2. 加载原始数据
# ============================================================
print("加载原始数据 (全部文件, 首次较慢, 后续从缓存加载)...")

if os.path.exists(CACHE_FILE):
    print(f"  从缓存加载 {CACHE_FILE}...")
    with open(CACHE_FILE, "rb") as f:
        raw = pickle.load(f)
    print(f"  已加载 {len(raw)} 行")
else:
    all_dirs = sorted(
        [
            d
            for d in os.listdir(RAW_DIR)
            if os.path.isdir(os.path.join(RAW_DIR, d))
        ]
    )
    print(f"  共 {len(all_dirs)} 个文件夹, 正在加载...")

    target_dfs = []

    for i, dname in enumerate(all_dirs):
        dpath = os.path.join(RAW_DIR, dname)

        for fname in os.listdir(dpath):
            if fname.endswith(".xls") and not fname.startswith("~$"):
                fpath = os.path.join(dpath, fname)
                df = pd.read_excel(fpath)
                df["时间"] = pd.to_datetime(df["时间"])
                target_dfs.append(df)
                break

        if (i + 1) % 25 == 0:
            print(f"    已加载 {i + 1}/{len(all_dirs)}")

    if not target_dfs:
        raise RuntimeError(f"没有在原始数据目录中读取到有效 .xls 文件：{RAW_DIR}")

    raw = pd.concat(target_dfs, ignore_index=True)
    raw = raw.sort_values("时间").reset_index(drop=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(raw, f)

    print(f"  已缓存 {len(raw)} 行至 {CACHE_FILE}")

print(f"  总行数: {len(raw)}, 时间范围: {raw['时间'].min()} ~ {raw['时间'].max()}")


# ============================================================
# 3. PIS变化点检测
# ============================================================
print("检测区间运行段...")

COL_TIME = "时间"
COL_SPEED = "BCU6_CCU 参考速度km/h"
COL_TRACTION = "列车总牵引力kN"
COL_E_BRAKE = "列车总电制动力kN"
COL_A_BRAKE = "列车总空气制动力kN"
COL_VOLTAGE1 = "CCU_LCU11 网压V"
COL_VOLTAGE2 = "LCU61_CCU 网压V"
COL_CURRENTS = [
    "DCU1_CCU 正线电流 1=1AA",
    "DCU2_CCU 正线电流 1=1AA",
    "DCU3_CCU 正线电流 1=1AA",
    "DCU4_CCU 正线电流 1=1AA",
]
COL_TRAIN_NO = "列车号"
COL_AXLES = [
    "TC1载荷（含转动惯量，1=0.01t）t",
    "MP1载荷（含转动惯量，1=0.01t）t",
    "M1载荷（含转动惯量，1=0.01t）t",
    "M2载荷（含转动惯量，1=0.01t）t",
    "MP2载荷（含转动惯量，1=0.01t）t",
    "TC2载荷（含转动惯量，1=0.01t）t",
]
COL_HVAC1 = "HVAC1_车外温度温度"
COL_HVAC6 = "HVAC6_车外温度温度"
COL_PIS = "PIS1_当前站ID"
COL_TRAC_ENERGY = "牵引能耗kwh"
COL_ATC_UPS = [
    "ATC1_列车在上行线",
    "ATC2_列车在上行线",
    "ATC3_列车在上行线",
    "ATC4_列车在上行线",
]
COL_ATC_DOWNS = [
    "ATC1_列车在下行线",
    "ATC2_列车在下行线",
    "ATC3_列车在下行线",
    "ATC4_列车在下行线",
]


# 计算总电流、速度、牵引力
raw["total_current"] = sum(raw[c].fillna(0) for c in COL_CURRENTS)
raw["speed"] = raw[COL_SPEED].fillna(0)
raw["traction"] = raw[COL_TRACTION].fillna(0)


# ------------------------------------------------------------
# 3.1 启动点检测
# ------------------------------------------------------------
# 核心逻辑：
# 牵引力从 0 -> 非0 作为候选牵引启动点；
# 然后从候选点向前搜索真正的启动时刻：
#   1) 15行内找 电流 <=0 -> >0；
#   2) 找不到，再30行内找 电流 <0 -> >0；
#   3) 还找不到，再100行内找最近的 电流 <0 -> 0；
# 不再执行“两个启动点必须间隔 >100行”的过滤。

SEARCH_NONPOS_TO_POS_ROWS = 15       # 约 0.75 秒
SEARCH_NEG_TO_POS_ROWS = 30          # 约 1.50 秒
SEARCH_NEG_TO_ZERO_ROWS = 100        # 约 5.00 秒，可按需要调大

raw["trac_off_to_on"] = (
    (raw["traction"].shift(1) <= 0) & (raw["traction"] > 0)
).astype(int)
trac_starts = raw[raw["trac_off_to_on"] == 1].index.tolist()
print(f"  牵引力0->非0: {len(trac_starts)} 次")

raw["curr_nonpos_to_pos"] = (
    (raw["total_current"].shift(1) <= 0) & (raw["total_current"] > 0)
).astype(int)

raw["curr_neg_to_pos"] = (
    (raw["total_current"].shift(1) < 0) & (raw["total_current"] > 0)
).astype(int)

raw["curr_neg_to_zero"] = (
    (raw["total_current"].shift(1) < 0) & (raw["total_current"] == 0)
).astype(int)

start_indices = []
start_reason_counts = {
    "curr_nonpos_to_pos": 0,
    "curr_neg_to_pos": 0,
    "curr_neg_to_zero": 0,
}

for ti in trac_starts:
    # 候选牵引启动点必须是在停车/低速状态，否则大概率是运行中的牵引波动
    if raw.loc[ti, "speed"] >= 1:
        continue

    found = None
    reason = None

    # 第一步：向前最多15行，找 <=0 -> >0
    search_start = max(0, ti - SEARCH_NONPOS_TO_POS_ROWS)
    for i in range(ti, search_start - 1, -1):
        if raw.loc[i, "curr_nonpos_to_pos"] == 1 and raw.loc[i, "speed"] < 1:
            found = i
            reason = "curr_nonpos_to_pos"
            break

    # 第二步：15行内没找到，则向前最多30行，找 <0 -> >0
    if found is None:
        search_start = max(0, ti - SEARCH_NEG_TO_POS_ROWS)
        for i in range(ti, search_start - 1, -1):
            if raw.loc[i, "curr_neg_to_pos"] == 1 and raw.loc[i, "speed"] < 1:
                found = i
                reason = "curr_neg_to_pos"
                break

    # 第三步：还没找到，则向前最多100行，找最近的 <0 -> 0
    # 这里是你新提出的逻辑：如果产生牵引力前几秒没有电流跳变，就用最近的负数变0点作为开始。
    if found is None:
        search_start = max(0, ti - SEARCH_NEG_TO_ZERO_ROWS)
        for i in range(ti, search_start - 1, -1):
            if raw.loc[i, "curr_neg_to_zero"] == 1 and raw.loc[i, "speed"] < 1:
                found = i
                reason = "curr_neg_to_zero"
                break

    if found is not None:
        # 保留原脚本的合理性验证：found之后30行内，牵引力必须出现明显上升
        check_end = min(len(raw) - 1, found + 30)
        if raw.loc[found:check_end, "traction"].max() > 3:
            start_indices.append(found)
            start_reason_counts[reason] += 1

# 只去掉完全重复的启动点，不再按间隔100行过滤
start_indices = sorted(set(start_indices))

print(f"  有效启动点: {len(start_indices)} (仅去重, 已取消100行间隔过滤)")
print(
    "  启动点来源: "
    f"<=0→>0 {start_reason_counts['curr_nonpos_to_pos']} 个, "
    f"<0→>0 {start_reason_counts['curr_neg_to_pos']} 个, "
    f"<0→0 {start_reason_counts['curr_neg_to_zero']} 个"
)


# ------------------------------------------------------------
# 3.2 停止点检测
# ------------------------------------------------------------
raw["speed_to_zero"] = (
    (raw["speed"].shift(1) > 0) & (raw["speed"] == 0)
).astype(int)

raw["at_stop"] = (raw["speed"] == 0) & (raw["traction"] < 1)
raw["stop_signal"] = raw["speed_to_zero"] & raw["at_stop"]
stop_indices = raw[raw["stop_signal"] == 1].index.tolist()

print(f"  停止点: {len(stop_indices)}")


# ------------------------------------------------------------
# 3.3 启停配对：v1逻辑 + 放宽长区间上限
# ------------------------------------------------------------
# v2：每个停止点最多只匹配一个启动点，避免多个启动点复用同一个到达停车点。
# 但原来的 seg_len < 4000 只允许最长约200秒，长区间如梅堰相关区间可能跑到340秒，
# 会在还没进入PIS区间匹配前就被删掉。
# 所以这里仅保留一个很宽松的全局候选范围：25s~600s。
# 真正的区间合理性，放到PIS匹配后用“每一区间最低运行时间”再过滤。
runs_all = []
start_ptr = 0
last_stop_idx = -1

GLOBAL_MIN_SEG_LEN = 500       # 约25秒，50ms采样
GLOBAL_MAX_SEG_LEN = 12000     # 约600秒，先放宽，避免长区间被提前删掉

for arr_idx in stop_indices:
    while start_ptr < len(start_indices) and start_indices[start_ptr] <= arr_idx:
        start_ptr += 1

    candidate_starts = [
        dep_idx
        for dep_idx in start_indices[:start_ptr]
        if dep_idx > last_stop_idx
        and GLOBAL_MIN_SEG_LEN < arr_idx - dep_idx < GLOBAL_MAX_SEG_LEN
    ]

    if not candidate_starts:
        last_stop_idx = arr_idx
        continue

    # 取距离本次停车点最远的合理启动点，表示从上一站出发到本站停车的一整段。
    # 之后还会用PIS顺序、方向和位移比例继续过滤。
    runs_all.append((candidate_starts[0], arr_idx))
    last_stop_idx = arr_idx

print(f"  有效运行段: {len(runs_all)}")
print(f"  启停候选范围: {GLOBAL_MIN_SEG_LEN*0.05:.1f}s ~ {GLOBAL_MAX_SEG_LEN*0.05:.1f}s")


# ============================================================
# 4. 匹配PIS区间
# ============================================================
print("匹配PIS区间...")

# 需要处理的所有相邻站区间，脚本会自动生成正向和反向
target_section_names = [
    # "布政-张家潭",
    # "张家潭-同德路",
    # "同德路-石碶",
    # "石碶-雅渡",
    # "雅渡-庙堰",
    # "庙堰-钟公庙",
    # "钟公庙-鄞州区政府",
    # "鄞州区政府-钱湖南路",
    "钱湖南路-南高教园区",
    # "南高教园区-下应路",
    # "下应路-大洋江",
    # "大洋江-泗港",
    # "泗港-曹隘",
    # "曹隘-柳隘",
    # "柳隘-海晏北路",
    # "海晏北路-民安东路",
    # "民安东路-会展中心",
    # "会展中心-院士路",
    # "院士路-盎孟港",
    # "盎孟港-三官堂",
    # "三官堂-兴庄路",
    # "兴庄路-兴海南路",
    # "兴海南路-梅堰",
    # "梅堰-永茂路",
    # "永茂路-镇海大道",
    # "镇海大道-骆驼桥",
]


def infer_direction_by_pis(p_from, p_to):
    """根据PIS顺序自动判断方向。PIS增大视为UP，减小视为DOWN。"""
    return "UP" if p_to > p_from else "DOWN"


target_pairs = {}

for section_name in target_section_names:
    start_name, end_name = section_name.split("-")

    if start_name not in station_name_to_pis:
        print(f"  警告: 站名不在线路基础数据中，已跳过: {start_name}")
        continue
    if end_name not in station_name_to_pis:
        print(f"  警告: 站名不在线路基础数据中，已跳过: {end_name}")
        continue

    p_start = int(station_name_to_pis[start_name])
    p_end = int(station_name_to_pis[end_name])

    # 正向
    target_pairs[(p_start, p_end)] = (
        start_name,
        end_name,
        infer_direction_by_pis(p_start, p_end),
    )

    # 反向
    target_pairs[(p_end, p_start)] = (
        end_name,
        start_name,
        infer_direction_by_pis(p_end, p_start),
    )

print(f"  目标区间数量: {len(target_pairs)} 个（含正反向）")

segment_results = {}


def compress_pis_sequence(seg):
    """按时间顺序压缩PIS序列，保留真实的站点变化方向。"""
    pis_series = pd.to_numeric(seg[COL_PIS], errors="coerce").dropna()
    seq = []
    for v in pis_series.values:
        iv = int(v)
        if not seq or seq[-1] != iv:
            seq.append(iv)
    return seq


def pis_transition_matches(pis_seq, p_from, p_to):
    """要求PIS按时间顺序出现 p_from -> p_to，避免同一段同时匹配正反向。"""
    if not pis_seq:
        return False

    if p_from not in pis_seq:
        return False

    start_pos = pis_seq.index(p_from)
    return p_to in pis_seq[start_pos + 1:]


def infer_atc_direction(seg):
    """从ATC上下行列估计运行方向；缺列或信号无效时返回None。"""
    up_cols = [c for c in COL_ATC_UPS if c in seg.columns]
    down_cols = [c for c in COL_ATC_DOWNS if c in seg.columns]

    if not up_cols or not down_cols:
        return None

    up_score = (
        seg[up_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
        .mean()
    )
    down_score = (
        seg[down_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
        .mean()
    )

    if max(up_score, down_score) <= 0:
        return None

    return "UP" if up_score >= down_score else "DOWN"

# 每一区间最低运行时间过滤参数
# 只过滤明显过短段，不设置最高时间，避免误删梅堰等长时间真实运行段。
# MIN_TIME_AVG_SPEED_KMH 越大，最低时间越短，过滤越宽松；
# 这里用55km/h作为“站间平均速度上限”的保守估计。
MIN_TIME_AVG_SPEED_KMH = 55
MIN_TIME_FLOOR_SEC = 35

# 位移积分过滤参数
# 你判断速度积分整体偏大，因此完整区间的积分位移不应小于站间距。
# v2已经增加PIS顺序和ATC方向校验，因此位移比例可以略放宽，
# 避免启停切点提前/滞后导致真实段被误删。
MIN_DISTANCE_RATIO = 0.85
MAX_DISTANCE_RATIO = 1.30

filtered_by_distance_too_short = 0
filtered_by_distance_too_long = 0
filtered_by_min_time = 0
filtered_by_pis_order = 0
filtered_by_direction = 0
matched_before_filters = 0
passed_distance_filter = 0

for dep_idx, arr_idx in runs_all:
    seg = raw.loc[dep_idx:arr_idx]

    pis_seq = compress_pis_sequence(seg)
    atc_direction = infer_atc_direction(seg)

    # 计算近似位移来验证站间距
    speeds = seg[COL_SPEED].fillna(0).values / 3.6
    dt = 0.05
    approx_dist = np.sum(speeds) * dt
    duration_sec = (arr_idx - dep_idx) * dt

    for (p_from, p_to), (sname, ename, dr) in target_pairs.items():
        # v2：要求PIS按时间顺序出现 起点 -> 终点，不能只看unique集合。
        if not pis_transition_matches(pis_seq, p_from, p_to):
            filtered_by_pis_order += 1
            continue

        # ATC上下行信号可用时，必须和目标方向一致。
        if atc_direction is not None and atc_direction != dr:
            filtered_by_direction += 1
            continue

        expected_dist = abs(
            station_mileage[pis_to_station[p_to]]
            - station_mileage[pis_to_station[p_from]]
        )

        if expected_dist <= 0:
            continue

        matched_before_filters += 1

        distance_ratio = approx_dist / expected_dist

        # 位移积分必须达到站间距
        if distance_ratio < MIN_DISTANCE_RATIO:
            filtered_by_distance_too_short += 1
            continue

        # 位移积分不能过大，避免跨站段误匹配
        if distance_ratio > MAX_DISTANCE_RATIO:
            filtered_by_distance_too_long += 1
            continue

        passed_distance_filter += 1

        # 按区间长度计算最低合理运行时间。
        # 例如 expected_dist=1428m，则最低时间约 1428/(55/3.6)=93.5s。
        # 小于这个时间的段大概率是重复启动点切出来的短假段。
        min_time_sec = max(
            MIN_TIME_FLOOR_SEC,
            expected_dist / (MIN_TIME_AVG_SPEED_KMH / 3.6)
        )

        if duration_sec < min_time_sec:
            filtered_by_min_time += 1
            continue

        key = (sname, ename, dr)
        if key not in segment_results:
            segment_results[key] = []
        segment_results[key].append((dep_idx, arr_idx))

print(
    f"  PIS匹配诊断: 顺序+方向通过候选 {matched_before_filters} 次, "
    f"PIS顺序过滤 {filtered_by_pis_order} 次, "
    f"ATC方向过滤 {filtered_by_direction} 段, "
    f"位移过短过滤 {filtered_by_distance_too_short} 段, "
    f"位移过长过滤 {filtered_by_distance_too_long} 段, "
    f"通过位移过滤 {passed_distance_filter} 段, "
    f"最低时间过滤 {filtered_by_min_time} 段"
)
print(
    f"  位移过滤规则: {MIN_DISTANCE_RATIO:.2f} <= 位移积分/站间距 <= {MAX_DISTANCE_RATIO:.2f}"
)
print(
    f"  最低时间规则: max({MIN_TIME_FLOOR_SEC}s, 区间长度 / {MIN_TIME_AVG_SPEED_KMH}km/h)；不设置最高时间"
)

for (sname, ename, dr), runs in segment_results.items():
    print(f"  {sname}→{ename} ({dr}): {len(runs)} 段")


# ============================================================
# 5. 构建输出结果
# ============================================================
all_results = []
result_files = {}


def match_param(mileages, ref, s_col, e_col, v_col, default):
    r = np.full(len(mileages), default)
    for _, row in ref.iterrows():
        m = (mileages >= row[s_col]) & (mileages <= row[e_col])
        r[m] = row[v_col]
    return r


for (start_name, end_name, direction), runs in segment_results.items():
    sid = find_station_id(start_name)
    eid = find_station_id(end_name)
    section_length = abs(station_mileage.get(eid, 0) - station_mileage.get(sid, 0))

    print(f"\n{start_name}->{end_name} ({direction}): {len(runs)} 个运行段")

    if not runs:
        continue

    run_dfs = []

    for run_idx, (si, ei) in enumerate(runs):
        seg = raw.loc[si:ei].copy().reset_index(drop=True)
        nrows = len(seg)

        start_time = seg[COL_TIME].iloc[0]

        # 车底号和服务号
        train_no_vals = seg[COL_TRAIN_NO].dropna()
        train_no = int(train_no_vals.iloc[0]) if len(train_no_vals) > 0 else 5012

        base = int(start_time.strftime("%m%d"))
        service_no = float(base * 100 + run_idx)
        date_service = f"{start_time.strftime('%m/%d')}{int(service_no)}"

        # 时段划分
        h = start_time.hour + start_time.minute / 60
        wd = start_time.weekday()

        if wd >= 5:
            tp = "双休日平峰" if 7 <= h < 10 or 17 <= h < 19 else "双休日低峰"
        else:
            if 7 <= h < 9 or 17 <= h < 19:
                tp = "工作日高峰"
            elif 9 <= h < 17:
                tp = "工作日平峰"
            else:
                tp = "工作日低峰"

        # 物理计算：固定50ms采样
        DT = 0.05
        seg["dt"] = DT
        seg["speed_ms"] = seg[COL_SPEED].fillna(0) / 3.6

        # 速度平滑：窗口10 = 500ms
        speeds_raw = seg["speed_ms"].values
        speeds_smooth = (
            pd.Series(speeds_raw)
            .rolling(10, min_periods=1, center=True)
            .mean()
            .values
        )

        # 加速度：速度差分后再平滑
        accel_raw = np.zeros(len(speeds_smooth))
        accel_raw[0] = (
            (speeds_smooth[1] - speeds_smooth[0]) / DT
            if len(speeds_smooth) > 1
            else 0
        )

        for i in range(1, len(speeds_smooth)):
            accel_raw[i] = (speeds_smooth[i] - speeds_smooth[i - 1]) / DT

        accel = (
            pd.Series(accel_raw)
            .rolling(5, min_periods=1, center=True)
            .mean()
            .values
        )

        seg["accel"] = accel
        seg["disp"] = (seg["speed_ms"] * DT).cumsum()

        # 能耗计算
        voltage = (
            seg[[COL_VOLTAGE1, COL_VOLTAGE2]]
            .mean(axis=1)
            .replace(0, np.nan)
            .ffill()
            .fillna(1500)
        )
        total_curr = sum(seg[c].fillna(0) for c in COL_CURRENTS)

        # 只计牵引能耗，不计再生制动：负功率截断为0
        power = (voltage * total_curr).clip(lower=0)
        energy = power * DT
        cum_energy = energy.cumsum()

        # 里程匹配
        sm = station_mileage.get(sid, 0)
        abs_mileage = sm + seg["disp"] if direction == "UP" else sm - seg["disp"]

        # 曲率/坡度匹配
        dc = curve_df[curve_df["direction"] == (1 if direction == "UP" else 2)]
        dg = grade_df[grade_df["direction"] == (1 if direction == "UP" else 2)]

        curvature = match_param(
            abs_mileage.values,
            dc,
            "curve_start_mile",
            "curve_end_mile",
            "curve_radius",
            3000,
        )
        gradient = match_param(
            abs_mileage.values,
            dg,
            "grade_start_point_mileage",
            "grade_end_point_mileage",
            "gradient",
            0.0,
        )

        # 温度
        t1 = (
            round(seg[COL_HVAC1].dropna().mean() / 10.0, 1)
            if seg[COL_HVAC1].notna().any()
            else 25.0
        )
        t6 = (
            round(seg[COL_HVAC6].dropna().mean() / 10.0, 1)
            if seg[COL_HVAC6].notna().any()
            else 25.0
        )
        tavg = round((t1 + t6) / 2, 2)

        # 重量：轴重均值求和
        axle_sum = 0.0

        for ac in COL_AXLES:
            if ac in seg.columns:
                v = seg[ac].dropna()
                if len(v) > 0:
                    axle_sum += v.mean()

        weight = round(axle_sum, 2) if axle_sum > 0 else 216.0

        # 原始牵引能耗累计量相减
        te = seg[COL_TRAC_ENERGY].dropna()
        trac_diff = float(te.iloc[-1] - te.iloc[0]) if len(te) >= 2 else np.nan

        # 构建结果
        data = {
            "区段": [f"{start_name}-{end_name}"] * nrows,
            "站间距": [section_length] * nrows,
            "时段划分": [tp] * nrows,
            "日期+服务号": [date_service] * nrows,
            "服务号": [float(service_no)] * nrows,
            "车底号": [train_no] * nrows,
            "列车运行方向": [direction] * nrows,
            "区间时间": (np.arange(nrows) * 50).astype(int),
            "时刻": np.round(np.arange(nrows) * 0.05, 2),
            "BCU6_CCU 参考速度km/h": seg[COL_SPEED].fillna(0).values,
            "列车总牵引力kN": seg[COL_TRACTION].fillna(0).values,
            "列车总电制动力kN": seg[COL_E_BRAKE].fillna(0).values,
            "列车总空气制动力KN": seg[COL_A_BRAKE].fillna(0).values,
            "1车温度℃": [t1] * nrows,
            "6车温度℃": [t6] * nrows,
            "（1车温度+6车温度）/2℃": [tavg] * nrows,
            "CCU_LCU11 网压V": seg[COL_VOLTAGE1].fillna(0).values,
            "LCU61_CCU 网压V": seg[COL_VOLTAGE2].fillna(0).values,
            "DCU1_CCU 正线电流 1=1AA": seg[COL_CURRENTS[0]].fillna(0).values,
            "DCU2_CCU 正线电流 1=1AA": seg[COL_CURRENTS[1]].fillna(0).values,
            "DCU3_CCU 正线电流 1=1AA": seg[COL_CURRENTS[2]].fillna(0).values,
            "DCU4_CCU 正线电流 1=1AA": seg[COL_CURRENTS[3]].fillna(0).values,
            "牵引能耗累计量": seg[COL_TRAC_ENERGY].ffill().fillna(0).values,
            "牵引能耗相减": [trac_diff] * nrows,
            "区间能耗累计值": cum_energy.values,
            "dt_overall": [0.00] + [0.05] * (nrows - 1),
            "segment": [run_idx] * nrows,
            "voltage": voltage.values,
            "total_effective_current": total_curr.values,
            "power": power.values,
            "energy": energy.values,
            "cumulative_energy": cum_energy.values,
            "cumulative_energy_kWh": cum_energy.values / 3600000,
            "速度(m/s)": seg["speed_ms"].values,
            "加速度(m/s²)": seg["accel"].values,
            "累计位移(m)": seg["disp"].values,
            "curvature": curvature.astype(int),
            "gradient": gradient,
            "重量": [weight] * nrows,
        }

        res = pd.DataFrame(data)

        # 强制dtype对齐原始结果
        res["服务号"] = res["服务号"].astype("float64")
        res["牵引能耗相减"] = res["牵引能耗相减"].astype("float64")

        run_dfs.append(res)

    if run_dfs:
        final = pd.concat(run_dfs, ignore_index=True)
        out_name = f"results_{start_name}-{end_name}.xlsx"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        tmp_path = out_path + ".tmp.xlsx"

        final.to_excel(tmp_path, index=False)

        with zipfile.ZipFile(tmp_path) as zf:
            has_worksheet = any(
                name.startswith("xl/worksheets/") for name in zf.namelist()
            )

        if not has_worksheet:
            os.remove(tmp_path)
            raise RuntimeError(f"输出校验失败，没有生成有效工作表: {tmp_path}")

        os.replace(tmp_path, out_path)

        print(f"  输出: {out_path} ({len(final)} 行, {len(run_dfs)} 趟)")
        result_files[out_name] = final


print("\n完成!")
for fname in result_files:
    print(f"  {os.path.join(OUTPUT_DIR, fname)}")
