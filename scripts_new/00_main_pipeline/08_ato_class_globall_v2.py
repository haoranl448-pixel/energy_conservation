# -*- coding: utf-8 -*-
"""
ato_class_globall_v2.py

DP schedule optimizer with flexible dwell times.

v2 changes from v1:
- Per-station dwell config (nominal + min). Stations with min==nominal are locked.
- After DP finds the best running-time combination, dwell slack is distributed
  proportionally so total time matches T_TOTAL_TARGET exactly.
- If a combination can't fit within dwell constraints, it falls through to the
  next-best combination.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ===================== 1. Global config =====================
DEFAULT_T_TOTAL_TARGET = 692.65

def get_target_time(default_value):
    """Read the DP target total time from the main pipeline, or use the script default."""

    raw = os.environ.get("ENERGY_TARGET_TIME")
    if not raw:
        return default_value
    value = float(raw)
    if value <= 0:
        raise ValueError("ENERGY_TARGET_TIME must be > 0.")
    return value


#T_TOTAL_TARGET = 694.7#trip1
T_TOTAL_TARGET = get_target_time(DEFAULT_T_TOTAL_TARGET)#trip6/default
SLACK = 10
NOMINAL_DWELL = 30.0          # default dwell for stations not in config
MIN_DWELL = 23.0              # default min dwell for elastic stations

# Per-station dwell config. Stations NOT listed here use defaults above.
# "nominal": target dwell time (s)
# "min": minimum allowed dwell time (s). Set min == nominal to lock the station.
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

MANUAL_CONSTRAINTS = {
    "泗港-曹隘": "class4"
}

GLOBAL_ALLOWED_CLASSES = ["class2", "class3","class4", "class5"]
CURVE_SOURCE_COL = "曲线来源"
PARENT_REF_COL = "父等级"
SUPPORTING_REFS_COL = "支持等级"
REAL_SAMPLE_COUNT_COL = "真实样本数"
PARENT_REAL_SAMPLE_COUNT_COL = "父等级真实样本数"
SAMPLE_RELIABILITY_COL = "样本可靠性"
DP_CANDIDATE_STAGE_COL = "DP候选阶段"
HIST_RAW_ENERGY_COL = "历史实测能耗(Wh)"
HIST_MODEL_ENERGY_COL = "历史能耗(Wh)"

DP_SOURCE_STAGES = [
    ("real_only", {"real"}),
    ("allow_interpolated", {"real", "interpolated"}),
    ("allow_extrapolated", {"real", "interpolated", "extrapolated_adjacent"}),
]

# ===================== 2. Paths =====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_trip_no() -> int:
    """从主程序传入的环境变量里读取要处理第几趟车。"""

    raw = os.environ.get("ENERGY_TRIP_NO", "1")
    trip_no = int(raw)
    if trip_no < 1:
        raise ValueError("ENERGY_TRIP_NO must be >= 1.")
    return trip_no


TRIP_NO = get_trip_no()
MENU_FILE = PROJECT_ROOT / "output" / "analysis" / f"ato_class_energy_menu{TRIP_NO}_new_v3.csv"
HIST_FILE = PROJECT_ROOT / f"full_line{TRIP_NO}_validation_results.csv"
TRAJ_BASE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v4"
OUT_DIR = PROJECT_ROOT / "output" / "schedule" / "final_plan_report_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SECTION_PARAMS_FILE = PROJECT_ROOT / "data" / "static" / f"section_params_trip{TRIP_NO}.csv"

def get_data_dir(default_value: Path) -> Path:
    """Read historical results_*.xlsx from the results data directory selected by main."""

    raw = os.environ.get("ENERGY_RESULTS_DATA_DIR") or os.environ.get("ENERGY_DATA_DIR")
    if not raw:
        return default_value
    data_dir = Path(raw)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir


DATA_DIR = get_data_dir(PROJECT_ROOT / "data" / "data_processed")
SLIP_RATIO = 1.0
TRIP_INDEX = TRIP_NO - 1

FULL_LINE_STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥",
]


def get_line_scope(default_value: str = "full") -> str:
    """Read the station range from the main pipeline."""

    raw = os.environ.get("ENERGY_LINE_SCOPE", default_value).strip().lower()
    if raw in {"full", "all"}:
        return "full"
    if raw == "first5":
        return "5"
    try:
        section_count = int(raw)
    except ValueError as exc:
        raise ValueError("ENERGY_LINE_SCOPE must be 'full' or a positive integer section count.") from exc
    if section_count < 1:
        raise ValueError("ENERGY_LINE_SCOPE section count must be >= 1.")
    if section_count > len(FULL_LINE_STATIONS):
        raise ValueError(f"ENERGY_LINE_SCOPE section count must be <= {len(FULL_LINE_STATIONS)}.")
    return raw


LINE_SCOPE = get_line_scope()
STATIONS = FULL_LINE_STATIONS if LINE_SCOPE == "full" else FULL_LINE_STATIONS[:int(LINE_SCOPE)]

# ===================== 3. Dwell helper =====================

def get_dwell_config(sp):
    """Return (nominal, min) dwell for a station."""
    if sp in STATION_DWELL_CONFIG:
        cfg = STATION_DWELL_CONFIG[sp]
        return cfg["nominal"], cfg["min"]
    return NOMINAL_DWELL, MIN_DWELL


def distribute_dwell_delta(run_time_sum, dwell_configs):
    """
    Given a running-time total and per-station dwell configs,
    compute actual dwell for each station so that:
        run_time_sum + sum(actual_dwells) = T_TOTAL_TARGET

    dwell_configs: list of (nominal, min) for each station (length = n_stations - 1)
    Returns: (success, dwell_list) where dwell_list has length n_stations - 1
    """
    n_dwell = len(dwell_configs)
    if n_dwell == 0:
        # No dwell intervals (single station)
        return True, []

    nominals = np.array([d[0] for d in dwell_configs], dtype=float)
    mins = np.array([d[1] for d in dwell_configs], dtype=float)

    target_dwell_total = T_TOTAL_TARGET - run_time_sum
    nominal_total = nominals.sum()

    if target_dwell_total < mins.sum():
        return False, None  # infeasible: running too long even with min dwell

    delta = target_dwell_total - nominal_total
    slacks = nominals - mins  # how much each station can give

    total_slack = slacks.sum()
    if total_slack <= 1e-9:
        # All stations locked. Target must match nominal total exactly.
        return abs(delta) < 1.0, nominals.tolist()

    # Proportional distribution
    dwell = nominals + delta * (slacks / total_slack)

    # Clamp to [min, nominal] (or [nominal, ...] if delta > 0)
    if delta <= 0:
        dwell = np.maximum(dwell, mins)
    else:
        # Extra time: add proportionally above nominal
        dwell = nominals + delta * (slacks / total_slack)

    # Final safety clamp
    dwell = np.maximum(dwell, mins)

    return True, np.round(dwell, 1).tolist()


def ensure_menu_source_columns(df_menu: pd.DataFrame) -> pd.DataFrame:
    """Keep old energy menus usable by treating missing source fields as real."""

    df_menu = df_menu.copy()
    defaults = {
        CURVE_SOURCE_COL: "real",
        PARENT_REF_COL: "",
        SUPPORTING_REFS_COL: "",
        REAL_SAMPLE_COUNT_COL: 0,
        PARENT_REAL_SAMPLE_COUNT_COL: 0,
        SAMPLE_RELIABILITY_COL: "",
        DP_CANDIDATE_STAGE_COL: "real_only",
    }
    for col, value in defaults.items():
        if col not in df_menu.columns:
            df_menu[col] = value
    df_menu[CURVE_SOURCE_COL] = df_menu[CURVE_SOURCE_COL].fillna("real").astype(str)
    return df_menu


def source_count_delta(curve_source: str) -> tuple[int, int]:
    """Return (extrapolated_count, interpolated_count) for DP lexicographic scoring."""

    if curve_source == "extrapolated_adjacent":
        return 1, 0
    if curve_source == "interpolated":
        return 0, 1
    return 0, 0


def row_source_meta(row: pd.Series) -> dict:
    return {
        "curve_source": row.get(CURVE_SOURCE_COL, "real"),
        "parent_ref": row.get(PARENT_REF_COL, ""),
        "supporting_refs": row.get(SUPPORTING_REFS_COL, ""),
        "real_sample_count": row.get(REAL_SAMPLE_COUNT_COL, 0),
        "parent_real_sample_count": row.get(PARENT_REAL_SAMPLE_COUNT_COL, 0),
        "sample_reliability": row.get(SAMPLE_RELIABILITY_COL, ""),
        "dp_candidate_stage": row.get(DP_CANDIDATE_STAGE_COL, ""),
    }


# ===================== 4. Core staged DP =====================

def run_optimization():
    # A. Load data
    print(f"Trip: {TRIP_NO} (segment index {TRIP_INDEX})")
    print(f"Target total time: {T_TOTAL_TARGET:.2f}s")
    print(f"Line scope: {LINE_SCOPE} ({len(STATIONS)} sections)")
    print(f"Menu file: {MENU_FILE}")
    print(f"History file: {HIST_FILE}")
    print(f"Historical curve data dir: {DATA_DIR}")
    df_menu = ensure_menu_source_columns(pd.read_csv(MENU_FILE))
    df_hist = pd.read_csv(HIST_FILE).set_index('站间区间')
    if HIST_RAW_ENERGY_COL not in df_hist.columns:
        raise ValueError(f"{HIST_FILE} 缺少 {HIST_RAW_ENERGY_COL}，请先运行 historical_baseline 生成带原始实测能耗的历史基准。")
    df_mass = pd.read_csv(SECTION_PARAMS_FILE)
    mass_map = dict(zip(df_mass['station_pair'], df_mass['MASS']))

    # B. Build dwell configs
    dwell_configs = [get_dwell_config(sp) for sp in STATIONS]
    dwell_mins = [d[1] for d in dwell_configs]
    dwell_nominals = [d[0] for d in dwell_configs]
    min_total_dwell = sum(dwell_mins[:-1])       # no dwell after last run
    nominal_total_dwell = sum(dwell_nominals[:-1])
    max_run_time = T_TOTAL_TARGET - min_total_dwell  # longest running we can afford
    nom_run_time = T_TOTAL_TARGET - nominal_total_dwell  # nominal for display

    # C. Staged DP. Prefer real curves first; only open generated candidates if needed.
    def to_int(t): return int(round(t * 10))
    max_run_int = to_int(max_run_time)

    print(f"Max run (at min dwell): {max_run_time:.1f}s")
    print(f"Nom run (at nominal):   {nom_run_time:.1f}s")
    print(f"Dwell budget to trade:   {nominal_total_dwell - min_total_dwell:.0f}s")
    print(f"Dwell config ({len(STATIONS)-1} intervals):")
    for sp in STATIONS:
        n, m = get_dwell_config(sp)
        tag = "LOCKED" if n == m else f"elastic ({m:.0f}-{n:.0f}s)"
        print(f"  {sp}: nominal={n:.0f}s, min={m:.0f}s [{tag}]")

    def options_for_stage(sp: str, allowed_sources: set[str]) -> pd.DataFrame:
        all_opts = df_menu[df_menu['站间区间'] == sp]
        if sp in MANUAL_CONSTRAINTS:
            options = all_opts[all_opts['运行等级'] == MANUAL_CONSTRAINTS[sp]]
        else:
            options = all_opts[all_opts['运行等级'].isin(GLOBAL_ALLOWED_CLASSES)]
        return options[options[CURVE_SOURCE_COL].isin(allowed_sources)].copy()

    def run_dp_stage(stage_name: str, allowed_sources: set[str]):
        # score tuple: (extrapolated_count, interpolated_count, energy_wh)
        dp = {0: (0, 0, 0.0)}
        path = []

        print(f"\nDP stage: {stage_name} | sources={sorted(allowed_sources)}")
        for i, sp in enumerate(STATIONS):
            new_dp, new_path = {}, {}
            options = options_for_stage(sp, allowed_sources)
            if options.empty:
                print(f"  [{i+1}/{len(STATIONS)}] {sp} -> no options in this stage")
                return None

            for t_prev, score_prev in dp.items():
                ext_prev, int_prev, e_prev = score_prev
                for _, row in options.iterrows():
                    t_curr = round(row['运行时长(s)'], 1)
                    t_sum = t_prev + to_int(t_curr)
                    if t_sum > max_run_int + 200:
                        continue

                    e_curr = float(row['预测能耗(Wh)'])
                    ext_add, int_add = source_count_delta(str(row.get(CURVE_SOURCE_COL, "real")))
                    score = (ext_prev + ext_add, int_prev + int_add, e_prev + e_curr)
                    if t_sum not in new_dp or score < new_dp[t_sum]:
                        new_dp[t_sum] = score
                        new_path[t_sum] = (
                            t_prev,
                            row['运行等级'],
                            t_curr,
                            e_curr,
                            row_source_meta(row),
                        )

            dp, path = new_dp, path + [new_path]
            print(f"  [{i+1}/{len(STATIONS)}] {sp} -> {len(dp)} states")
            if not dp:
                return None

        feasible = [(t, dp[t]) for t in dp.keys() if t <= max_run_int]
        if not feasible:
            min_r = min(dp.keys()) / 10.0
            max_r = max(dp.keys()) / 10.0
            print(f"  no feasible state. Reachable run time: [{min_r:.1f}s, {max_r:.1f}s], max allowed: {max_run_time:.1f}s")
            return None

        feasible.sort(key=lambda x: x[1])
        best_t_int, best_score = feasible[0]
        return {
            "stage_name": stage_name,
            "dp": dp,
            "path": path,
            "feasible": feasible,
            "best_t_int": best_t_int,
            "final_energy": float(best_score[2]),
            "extrapolated_count": int(best_score[0]),
            "interpolated_count": int(best_score[1]),
        }

    solution = None
    for stage_name, allowed_sources in DP_SOURCE_STAGES:
        solution = run_dp_stage(stage_name, allowed_sources)
        if solution is not None:
            break

    if solution is None:
        print("ERROR: No DP stage can fit even with min dwell.")
        return

    stage_name = solution["stage_name"]
    dp = solution["dp"]
    path = solution["path"]
    feasible = solution["feasible"]
    best_t_int = solution["best_t_int"]
    final_energy = solution["final_energy"]

    # Distribute remaining time as dwell (run_sum <= max_run_time guaranteed)
    best_dwells = distribute_dwell_delta(best_t_int / 10.0, dwell_configs[:-1])[1]

    print(f"\nSelected DP stage: {stage_name}")
    print(f"Feasible states: {len(feasible)}")
    print(
        f"Best:  run={best_t_int/10.0:.1f}s, energy={final_energy:.1f} Wh, "
        f"extrapolated={solution['extrapolated_count']}, interpolated={solution['interpolated_count']}"
    )

    nom_int = to_int(nom_run_time)
    nearby = [(t, dp[t]) for t in dp.keys() if nom_int - to_int(SLACK) <= t <= nom_int + to_int(SLACK)]
    if nearby:
        nearby.sort(key=lambda x: x[1])
        nom_best_t, nom_best_score = nearby[0]
        nom_best_e = float(nom_best_score[2])
        print(f"Nominal-dwell best nearby: run={nom_best_t/10.0:.1f}s, energy={nom_best_e:.1f} Wh")
        saving = nom_best_e - final_energy
        extra_run = (best_t_int - nom_best_t) / 10.0
        if nom_best_e:
            print(f"Dwell trade: +{extra_run:.1f}s run time -> saves {saving:.1f} Wh ({saving/nom_best_e*100:.1f}%)")

    run_sum = best_t_int / 10.0

    # E. Backtrack
    final_rows = []
    curr_t = best_t_int
    for i in range(len(STATIONS) - 1, -1, -1):
        prev_t, c_name, t_val, e_val, source_meta = path[i][curr_t]
        sp = STATIONS[i]
        h_time = df_hist.loc[sp, '历史运行时间(s)']
        h_raw_energy = df_hist.loc[sp, HIST_RAW_ENERGY_COL]
        h_model_energy = df_hist.loc[sp, HIST_MODEL_ENERGY_COL] if HIST_MODEL_ENERGY_COL in df_hist.columns else np.nan

        # Dwell after this station (except last)
        if i < len(STATIONS) - 1:
            dwell = best_dwells[i]
        else:
            dwell = 0.0

        final_rows.append({
            "站间区间": sp,
            "选定等级": c_name,
            "规划用时(s)": round(t_val, 0),
            "停站时间(s)": round(dwell, 1) if i < len(STATIONS) - 1 else 0,
            "历史用时(s)": round(h_time, 2),
            "规划能耗(Wh)": round(e_val, 2),
            "历史能耗(Wh)": round(h_raw_energy, 2),
            "历史模型回放能耗(Wh)": round(h_model_energy, 2) if pd.notna(h_model_energy) else np.nan,
            "能耗对比基准": "历史原始实测能耗",
            "节能量(Wh)": round(h_raw_energy - e_val, 2),
            "曲线来源": source_meta.get("curve_source", "real"),
            "父等级": source_meta.get("parent_ref", ""),
            "支持等级": source_meta.get("supporting_refs", ""),
            "真实样本数": source_meta.get("real_sample_count", 0),
            "父等级真实样本数": source_meta.get("parent_real_sample_count", 0),
            "样本可靠性": source_meta.get("sample_reliability", ""),
            "规划阶段": stage_name,
            "MASS": round(mass_map.get(sp, np.nan), 2),
        })
        curr_t = prev_t

    final_rows.reverse()

    # Fix backtrack rounding drift: scale per-station times to match DP total exactly
    run_exact = best_t_int / 10.0
    run_backtrack_sum = sum(r['规划用时(s)'] for r in final_rows)
    drift = run_exact - run_backtrack_sum
    if abs(drift) > 0.01:
        # Distribute drift by adding 1 decisecond to the first N stations
        n_adj = int(round(abs(drift) * 10))
        step = 1 if drift > 0 else -1
        for j in range(n_adj):
            final_rows[j % len(final_rows)]['规划用时(s)'] += step * 0.1
        for r in final_rows:
            r['规划用时(s)'] = round(r['规划用时(s)'], 0)

    df_res = pd.DataFrame(final_rows)

    # F. Summary
    total_h_e = df_res['历史能耗(Wh)'].sum()
    total_h_model_e = df_res['历史模型回放能耗(Wh)'].sum(skipna=True) if '历史模型回放能耗(Wh)' in df_res.columns else np.nan
    total_p_e = df_res['规划能耗(Wh)'].sum()
    total_p_run = df_res['规划用时(s)'].sum()
    total_p_dwell = df_res['停站时间(s)'].sum()
    total_p_t = total_p_run + total_p_dwell
    total_h_t = df_res['历史用时(s)'].sum() + nominal_total_dwell  # fixed historical dwell
    saving_rate = (total_h_e - total_p_e) / total_h_e * 100

    summary_row = {
        "站间区间": "--- 总计 ---",
        "选定等级": f"节能率: {saving_rate:.2f}%",
        "规划用时(s)": total_p_run,
        "停站时间(s)": total_p_dwell,
        "历史用时(s)": total_h_t,
        "规划能耗(Wh)": round(total_p_e, 2),
        "历史能耗(Wh)": round(total_h_e, 2),
        "历史模型回放能耗(Wh)": round(total_h_model_e, 2) if pd.notna(total_h_model_e) else np.nan,
        "能耗对比基准": "历史原始实测能耗",
        "节能量(Wh)": round(total_h_e - total_p_e, 2),
        "曲线来源": "",
        "父等级": "",
        "支持等级": "",
        "真实样本数": "",
        "父等级真实样本数": "",
        "样本可靠性": "",
        "规划阶段": stage_name,
        "MASS": round(df_res['MASS'].sum(), 2),
    }
    df_res = pd.concat([df_res, pd.DataFrame([summary_row])], ignore_index=True)
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
    print("\nPlotting full-line comparison ...")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    t_opt_acc = 0.0
    s_opt_acc = 0.0
    t_hist_acc = 0.0
    s_hist_acc = 0.0
    plot_data = {'opt_t': [], 'opt_v': [], 'opt_s': [],
                 'hist_t': [], 'hist_v': [], 'hist_s': []}
    dwell_zones = []

    for i, row in enumerate(final_rows):
        if i >= len(STATIONS):
            break
        sp = row['站间区间']
        c_name = row['选定等级']

        # Optimized trajectory
        df_opt = pd.read_csv(TRAJ_BASE_DIR / sp / f"{c_name}_generated_curve.csv")
        plot_data['opt_t'].extend((df_opt['time_s'] + t_opt_acc).tolist())
        plot_data['opt_v'].extend((df_opt['velocity_mps'] * 3.6).tolist())
        plot_data['opt_s'].extend((df_opt['dist_m'] + s_opt_acc).tolist())

        # Historical trajectory
        df_h_all = pd.read_excel(DATA_DIR / f"results_{sp}.xlsx")
        target_seg = sorted(df_h_all['segment'].unique())[TRIP_INDEX]
        df_h = df_h_all[df_h_all['segment'] == target_seg].copy()
        h_v = df_h['速度(m/s)'].values * 3.6
        h_t = df_h['时刻'].values - df_h['时刻'].iloc[0]
        h_s = df_h['累计位移(m)'].values / SLIP_RATIO

        plot_data['hist_t'].extend((h_t + t_hist_acc).tolist())
        plot_data['hist_v'].extend(h_v.tolist())
        plot_data['hist_s'].extend((h_s + s_hist_acc).tolist())

        t_opt_acc += row['规划用时(s)']
        s_opt_acc += df_opt['dist_m'].iloc[-1]
        t_hist_acc += h_t[-1]
        s_hist_acc += h_s[-1]

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
    fig.savefig(OUT_DIR / "Optimized_Full_Line_Report.png", dpi=300)
    plt.close()
    print(f"Saved: {OUT_DIR / 'Optimized_Full_Line_Report.png'}")
    print(f"Done. Energy: {total_p_e/1000:.3f} kWh, Time: {total_p_t:.1f}s (strict = {T_TOTAL_TARGET}s)")


if __name__ == "__main__":
    run_optimization()
