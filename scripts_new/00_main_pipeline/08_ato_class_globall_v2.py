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
import hashlib
import sys
from trip_traceability import (
    load_trip_traceability,
    print_traceability_summary,
    resolve_traceability_manifest,
    select_trip_rows,
)

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


def env_flag(name, default=False):
    """Read a boolean flag from environment variables."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


#T_TOTAL_TARGET = 694.7#trip1
T_TOTAL_TARGET = get_target_time(DEFAULT_T_TOTAL_TARGET)#trip6/default
SKIP_PLOTS = env_flag("ENERGY_SKIP_PLOTS", False)
SLACK = 10
NOMINAL_DWELL = 30.0          # default dwell for stations not in config
MIN_DWELL = 24.0              # default min dwell for elastic stations
HISTORICAL_DWELL_TOLERANCE = float(os.environ.get("ENERGY_DWELL_TOLERANCE", "0.02"))
if not 0.0 <= HISTORICAL_DWELL_TOLERANCE <= 0.20:
    raise ValueError("ENERGY_DWELL_TOLERANCE must be between 0 and 0.20.")
REAL_PRIORITY_DWELL_TOLERANCE = 0.05
TOTAL_TIME_TOLERANCE = float(os.environ.get("ENERGY_TOTAL_TIME_TOLERANCE", "0"))
if not 0.0 <= TOTAL_TIME_TOLERANCE <= 10.0:
    raise ValueError("ENERGY_TOTAL_TIME_TOLERANCE must be between 0 and 10 seconds.")

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

GLOBAL_ALLOWED_CLASSES = ["class1","class2", "class3","class4", "class4"]
CURVE_SOURCE_COL = "曲线来源"
PARENT_REF_COL = "父等级"
SUPPORTING_REFS_COL = "支持等级"
REAL_SAMPLE_COUNT_COL = "真实样本数"
PARENT_REAL_SAMPLE_COUNT_COL = "父等级真实样本数"
SAMPLE_RELIABILITY_COL = "样本可靠性"
DP_CANDIDATE_STAGE_COL = "DP候选阶段"
HIST_RAW_ENERGY_COL = "历史实测能耗(Wh)"
HIST_MODEL_ENERGY_COL = "历史能耗(Wh)"
HIST_RUN_TIME_COL = "历史运行时间(s)"

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
FINAL_COMPARISON_FILE = OUT_DIR / "Final_Planning_Comparison.csv"
ENERGY_FIRST_COMPARISON_FILE = OUT_DIR / "Final_Planning_Comparison_Energy_First.csv"
REAL_PRIORITY_5PCT_COMPARISON_FILE = OUT_DIR / "Final_Planning_Comparison_Real_Priority_Dwell_5pct.csv"
FINAL_REPORT_PNG = OUT_DIR / "Optimized_Full_Line_Report.png"
ENERGY_FIRST_REPORT_PNG = OUT_DIR / "Optimized_Full_Line_Report_Energy_First.png"
REAL_PRIORITY_5PCT_REPORT_PNG = OUT_DIR / "Optimized_Full_Line_Report_Real_Priority_Dwell_5pct.png"
HISTORY_CURVE_CACHE_DIR = PROJECT_ROOT / "output" / "cache" / "planning_history_curves"
HISTORY_CURVE_CACHE_VERSION = "history_trip_curve_v2_traceability"
_HISTORY_CURVE_MEMORY_CACHE = {}


def remove_existing_report_outputs():
    """Delete stale DP report outputs before rebuilding them."""

    output_root = (PROJECT_ROOT / "output").resolve()
    for path in [
        FINAL_COMPARISON_FILE,
        ENERGY_FIRST_COMPARISON_FILE,
        REAL_PRIORITY_5PCT_COMPARISON_FILE,
        FINAL_REPORT_PNG,
        ENERGY_FIRST_REPORT_PNG,
        REAL_PRIORITY_5PCT_REPORT_PNG,
    ]:
        resolved = path.resolve()
        try:
            resolved.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"拒绝删除非 output 目录下的文件: {resolved}") from exc

        if resolved.exists():
            resolved.unlink()
            print(f"已删除旧 DP 输出: {resolved}")

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
TRACEABILITY_MANIFEST = resolve_traceability_manifest(PROJECT_ROOT, DATA_DIR)
TRIP_TRACEABILITY = load_trip_traceability(TRACEABILITY_MANIFEST, TRIP_NO, direction="UP")


def get_historical_dwell_file() -> Path | None:
    """Return the optional trip/station dwell detail CSV selected by the batch runner."""

    raw = os.environ.get("ENERGY_HISTORICAL_DWELL_FILE")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


HISTORICAL_DWELL_FILE = get_historical_dwell_file()


def get_history_curve_cache_path(excel_path: Path, sp: str) -> Path:
    """Build a cache path that changes whenever the source Excel changes."""

    stat = excel_path.stat()
    if TRIP_TRACEABILITY is not None:
        record = TRIP_TRACEABILITY.record_for(sp)
        selection_key = f"{record.global_trip_id}|{record.segment}|{record.source_run_id}"
    else:
        selection_key = f"legacy_local_index|{TRIP_INDEX}"
    key_src = "|".join([
        HISTORY_CURVE_CACHE_VERSION,
        str(excel_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        sp,
        selection_key,
    ])
    key = hashlib.sha1(key_src.encode("utf-8", errors="surrogatepass")).hexdigest()
    return HISTORY_CURVE_CACHE_DIR / f"{key}.pkl"


def load_history_trip_curve(sp: str) -> pd.DataFrame:
    """Load one historical trip curve, using disk cache to avoid repeated Excel reads."""

    excel_path = DATA_DIR / f"results_{sp}.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"History curve Excel not found: {excel_path}")

    cache_path = get_history_curve_cache_path(excel_path, sp)
    memory_key = str(cache_path)
    if memory_key in _HISTORY_CURVE_MEMORY_CACHE:
        return _HISTORY_CURVE_MEMORY_CACHE[memory_key]

    if cache_path.exists():
        df_h = pd.read_pickle(cache_path)
        print(f"  History curve cache hit: {sp} trip {TRIP_NO}")
    else:
        print(f"  Building history curve cache from Excel: {sp} trip {TRIP_NO}")
        df_h_all = pd.read_excel(excel_path)
        if "segment" not in df_h_all.columns:
            raise ValueError(f"{excel_path} missing required column: segment")

        df_h, target_seg, record, selection_mode = select_trip_rows(
            df_h_all,
            sp,
            TRIP_INDEX,
            TRIP_TRACEABILITY,
        )
        print(
            f"    selected segment={target_seg}"
            + (f", run_id={record.source_run_id}" if record else "")
            + f" [{selection_mode}]"
        )
        HISTORY_CURVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df_h.to_pickle(cache_path)

    _HISTORY_CURVE_MEMORY_CACHE[memory_key] = df_h
    return df_h

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


def load_historical_dwell_profile() -> tuple[list[float], list[str]]:
    """Load dwell after each section, falling back to station medians when needed.

    The detail CSV produced by analyze_historical_dwell_first70.py stores a dwell
    against the preceding section.  That is exactly the dwell interval that must
    be inserted after the section in the full-line timetable.
    """

    if HISTORICAL_DWELL_FILE is None:
        values = [get_dwell_config(sp)[0] for sp in STATIONS]
        values[-1] = 0.0
        return values, ["legacy_default"] * (len(STATIONS) - 1) + ["terminal"]
    if not HISTORICAL_DWELL_FILE.exists():
        raise FileNotFoundError(f"Historical dwell detail CSV not found: {HISTORICAL_DWELL_FILE}")

    detail = pd.read_csv(HISTORICAL_DWELL_FILE, encoding="utf-8-sig")
    required = {"趟次", "停车站", "上一运行区间", "历史停站时间(s)"}
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Historical dwell detail CSV missing columns: {missing}")

    detail = detail.copy()
    detail["趟次"] = pd.to_numeric(detail["趟次"], errors="coerce")
    detail["历史停站时间(s)"] = pd.to_numeric(detail["历史停站时间(s)"], errors="coerce")
    valid_for_median = detail[detail["历史停站时间(s)"].between(20.0, 60.0, inclusive="both")]
    station_medians = valid_for_median.groupby("停车站")["历史停站时间(s)"].median().to_dict()
    selected = detail[detail["趟次"] == TRIP_NO]

    values: list[float] = []
    sources: list[str] = []
    for sp in STATIONS[:-1]:
        rows = selected[selected["上一运行区间"].astype(str).str.strip() == sp]
        actual = rows["历史停站时间(s)"].dropna()
        if not actual.empty and float(actual.iloc[0]) > 0:
            values.append(float(actual.iloc[0]))
            sources.append("actual")
            continue

        destination = sp.split("-", 1)[1]
        median = station_medians.get(destination)
        if median is not None and np.isfinite(median) and float(median) > 0:
            values.append(float(median))
            sources.append("station_median")
        else:
            values.append(NOMINAL_DWELL)
            sources.append("fixed_default")

    values.append(0.0)
    sources.append("terminal")
    return values, sources


def distribute_historical_dwell(
    run_time_sum: float,
    historical_dwells: list[float],
    tolerance: float,
) -> tuple[bool, list[float] | None]:
    """Fit the required dwell total while keeping every station within tolerance."""

    if not historical_dwells:
        return abs(run_time_sum - T_TOTAL_TARGET) <= 0.05, []

    history = np.asarray(historical_dwells, dtype=float)
    target_total = T_TOTAL_TARGET - run_time_sum
    history_total = float(history.sum())
    lower = history * (1.0 - tolerance)
    upper = history * (1.0 + tolerance)
    lower_total = float(lower.sum())
    upper_total = float(upper.sum())
    if target_total < lower_total:
        if lower_total - target_total > TOTAL_TIME_TOLERANCE + 0.05:
            return False, None
        target_total = lower_total
    elif target_total > upper_total:
        if target_total - upper_total > TOTAL_TIME_TOLERANCE + 0.05:
            return False, None
        target_total = upper_total

    if history_total <= 0:
        return abs(target_total) <= 0.05, [0.0] * len(historical_dwells)

    dwell = history * (target_total / history_total)
    dwell = np.clip(dwell, lower, upper)
    dwell = np.round(dwell, 2)

    # Repair the 0.01 s rounding residue without taking any station outside its band.
    units = int(round((target_total - float(dwell.sum())) * 100))
    direction = 1 if units > 0 else -1
    for _ in range(abs(units)):
        changed = False
        for index in range(len(dwell)):
            candidate = round(float(dwell[index]) + direction * 0.01, 2)
            if lower[index] - 1e-9 <= candidate <= upper[index] + 1e-9:
                dwell[index] = candidate
                changed = True
                break
        if not changed:
            return False, None

    if abs(float(dwell.sum()) - target_total) > 0.0051:
        return False, None
    return True, dwell.tolist()


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


def format_section_list(items: list[str], limit: int = 8) -> str:
    """Keep diagnostics readable when many sections are missing."""

    if not items:
        return "无"
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f" ... 另 {len(items) - limit} 个"
    return "、".join(shown) + suffix


def print_input_diagnostics(
    df_menu: pd.DataFrame,
    df_hist: pd.DataFrame,
    max_run_time: float,
    nom_run_time: float,
    historical_dwell_total: float | None = None,
    minimum_dwell_total: float | None = None,
) -> None:
    """Explain whether the selected trip has enough full-line data for DP."""

    menu_sections = set(df_menu["站间区间"].dropna().astype(str)) if "站间区间" in df_menu.columns else set()
    hist_sections = set(df_hist.index.dropna().astype(str))
    missing_menu = [sp for sp in STATIONS if sp not in menu_sections]
    missing_hist = [sp for sp in STATIONS if sp not in hist_sections]

    hist_run_sum = np.nan
    if HIST_RUN_TIME_COL in df_hist.columns:
        hist_run_sum = pd.to_numeric(df_hist.loc[df_hist.index.intersection(STATIONS), HIST_RUN_TIME_COL], errors="coerce").sum()

    nominal_dwell_total = (
        historical_dwell_total
        if historical_dwell_total is not None
        else sum(get_dwell_config(sp)[0] for sp in STATIONS[:-1])
    )
    min_dwell_total = (
        minimum_dwell_total
        if minimum_dwell_total is not None
        else nominal_dwell_total
    )

    print("\nDP input diagnostics:")
    print(f"  Selected global trip: {TRIP_NO} (legacy local index {TRIP_INDEX})")
    print(f"  Target total time: {T_TOTAL_TARGET:.1f}s")
    print(f"  Allowed running time: <= {max_run_time:.1f}s within dwell tolerance, exact-dwell run {nom_run_time:.1f}s")
    print(f"  Dwell totals: historical={nominal_dwell_total:.1f}s, tolerance minimum={min_dwell_total:.1f}s")

    if np.isfinite(hist_run_sum):
        coverage_note = "full-line" if not missing_hist else f"partial {len(hist_sections & set(STATIONS))}/{len(STATIONS)} sections"
        print(f"  Historical selected-trip run time in {HIST_FILE.name}: {hist_run_sum:.1f}s ({coverage_note}, dwell not included)")
        if not missing_hist:
            print(f"  Historical selected-trip total with historical dwell: {hist_run_sum + nominal_dwell_total:.1f}s")
            print(f"  Historical selected-trip total at tolerance minimum: {hist_run_sum + min_dwell_total:.1f}s")
    else:
        print(f"  Historical selected-trip run time: unavailable; {HIST_FILE.name} lacks {HIST_RUN_TIME_COL}")

    print(f"  Menu coverage: {len(menu_sections & set(STATIONS))}/{len(STATIONS)} sections")
    if missing_menu:
        print(f"  Missing menu sections: {format_section_list(missing_menu)}")
    print(f"  History coverage: {len(hist_sections & set(STATIONS))}/{len(STATIONS)} sections")
    if missing_hist:
        print(f"  Missing history sections: {format_section_list(missing_hist)}")

    if missing_menu or missing_hist:
        print(
            "  Hint: 当前 trip 的 energy menu 或 historical baseline 不是全线覆盖。"
            " 如果只从 dp_schedule 开始跑，它会复用旧文件；请先为该 trip 重新生成 full-line energy_menu 和 historical_baseline。"
        )


def _canonical_number_text(value) -> str:
    """Normalize CSV numbers such as 5 and 5.0 before traceability comparison."""

    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:g}"


def validate_traceability_inputs(
    df_menu: pd.DataFrame,
    df_hist: pd.DataFrame,
    df_mass: pd.DataFrame,
) -> None:
    """Reject stale per-section-index files when this run uses a global trip chain."""

    if TRIP_TRACEABILITY is None:
        return

    required_columns = {"全局趟次ID", "历史segment", "历史来源run_id", "趟次选择方式"}
    tables = [
        ("energy menu", df_menu, "站间区间", False),
        ("historical baseline", df_hist, "站间区间", True),
        ("section mass params", df_mass, "station_pair", False),
    ]
    expected_id = TRIP_TRACEABILITY.global_trip_id

    for label, table, section_column, section_is_index in tables:
        missing = sorted(required_columns - set(table.columns))
        if missing:
            raise ValueError(
                f"{label} 缺少追溯字段 {missing}。当前文件可能是旧版按各区间第 N 趟生成的；"
                "请先重新运行 energy_menu 和 historical_baseline。"
            )

        for section in STATIONS:
            rows = table.loc[[section]] if section_is_index and section in table.index else None
            if not section_is_index:
                rows = table[table[section_column].astype(str).str.strip().eq(section)]
            if rows is None or rows.empty:
                continue

            record = TRIP_TRACEABILITY.record_for(section)
            ids = rows["全局趟次ID"].dropna().astype(str).str.strip().unique().tolist()
            segments = {
                _canonical_number_text(value)
                for value in rows["历史segment"].dropna().tolist()
            }
            run_ids = {
                _canonical_number_text(value)
                for value in rows["历史来源run_id"].dropna().tolist()
            }
            if ids != [expected_id]:
                raise ValueError(f"{label} 的 {section} 全局趟次ID不一致: {ids}, expected={expected_id}")
            if segments != {_canonical_number_text(record.segment)}:
                raise ValueError(
                    f"{label} 的 {section} segment不一致: {segments}, expected={record.segment}"
                )
            if run_ids != {_canonical_number_text(record.source_run_id)}:
                raise ValueError(
                    f"{label} 的 {section} 来源run_id不一致: {run_ids}, expected={record.source_run_id}"
                )


# ===================== 4. Core staged DP =====================

def run_optimization():
    # A. Load data
    print(f"Global trip: {TRIP_NO} (legacy local index {TRIP_INDEX})")
    print(f"Target total time: {T_TOTAL_TARGET:.2f}s")
    print(f"Line scope: {LINE_SCOPE} ({len(STATIONS)} sections)")
    print(f"Menu file: {MENU_FILE}")
    print(f"History file: {HIST_FILE}")
    print(f"Historical curve data dir: {DATA_DIR}")
    print_traceability_summary(TRIP_TRACEABILITY, TRIP_NO)
    if TRIP_TRACEABILITY is not None:
        TRIP_TRACEABILITY.require_sections(STATIONS)
    df_menu = ensure_menu_source_columns(pd.read_csv(MENU_FILE))
    df_hist = pd.read_csv(HIST_FILE).set_index('站间区间')
    if HIST_RAW_ENERGY_COL not in df_hist.columns:
        raise ValueError(f"{HIST_FILE} 缺少 {HIST_RAW_ENERGY_COL}，请先运行 historical_baseline 生成带原始实测能耗的历史基准。")
    df_mass = pd.read_csv(SECTION_PARAMS_FILE)
    validate_traceability_inputs(df_menu, df_hist, df_mass)
    # Only clear the previous report after all selected-trip inputs have passed
    # traceability validation. A stale input should not destroy the last report.
    remove_existing_report_outputs()
    mass_map = dict(zip(df_mass['station_pair'], df_mass['MASS']))

    # B. Historical dwell is the primary planning constraint.  Only if no exact
    # running-time combination exists do we allow a small per-station tolerance.
    historical_dwells, historical_dwell_sources = load_historical_dwell_profile()
    historical_total_dwell = sum(historical_dwells[:-1])
    tolerance_min_total = historical_total_dwell * (1.0 - HISTORICAL_DWELL_TOLERANCE)
    tolerance_max_total = historical_total_dwell * (1.0 + HISTORICAL_DWELL_TOLERANCE)
    max_run_time = T_TOTAL_TARGET - tolerance_min_total
    nom_run_time = T_TOTAL_TARGET - historical_total_dwell

    # C. Staged DP. Prefer real curves first; only open generated candidates if needed.
    def to_int(t): return int(round(t * 10))
    print(f"Historical dwell file:   {HISTORICAL_DWELL_FILE or 'not supplied (legacy default)'}")
    print(f"Historical dwell total:  {historical_total_dwell:.1f}s")
    print(f"Exact-dwell target run:  {nom_run_time:.1f}s")
    print(
        f"Fallback dwell band:     {tolerance_min_total:.1f}-{tolerance_max_total:.1f}s "
        f"(±{HISTORICAL_DWELL_TOLERANCE * 100:.1f}%)"
    )
    print_input_diagnostics(
        df_menu,
        df_hist,
        max_run_time,
        nom_run_time,
        historical_total_dwell,
        tolerance_min_total,
    )
    print(f"Historical dwell profile ({len(STATIONS)-1} intervals):")
    for index, sp in enumerate(STATIONS[:-1]):
        print(f"  {sp}: {historical_dwells[index]:.1f}s [{historical_dwell_sources[index]}]")

    def options_for_stage(sp: str, allowed_sources: set[str]) -> pd.DataFrame:
        all_opts = df_menu[df_menu['站间区间'] == sp]
        if sp in MANUAL_CONSTRAINTS:
            options = all_opts[all_opts['运行等级'] == MANUAL_CONSTRAINTS[sp]]
        else:
            options = all_opts[all_opts['运行等级'].isin(GLOBAL_ALLOWED_CLASSES)]
        return options[options[CURVE_SOURCE_COL].isin(allowed_sources)].copy()

    def score_key(score: tuple[int, int, float], score_mode: str) -> tuple[float, ...]:
        """Return the lexicographic key used to choose between DP states."""

        ext_count, int_count, energy_wh = score
        if score_mode == "energy_first":
            return energy_wh, ext_count, int_count
        return ext_count, int_count, energy_wh

    def run_dp_stage(
        stage_name: str,
        allowed_sources: set[str],
        min_run_time: float,
        max_run_time_for_stage: float,
        dwell_mode: str,
        dwell_tolerance: float,
        score_mode: str = "priority_first",
    ):
        # score tuple: (extrapolated_count, interpolated_count, energy_wh)
        dp = {0: (0, 0, 0.0)}
        path = []

        min_run_int = to_int(min_run_time)
        max_run_int = to_int(max_run_time_for_stage)
        print(
            f"\nDP stage: {stage_name} | dwell={dwell_mode} | "
            f"run_range=[{min_run_time:.1f}, {max_run_time_for_stage:.1f}] | "
            f"sources={sorted(allowed_sources)} | score={score_mode}"
        )
        for i, sp in enumerate(STATIONS):
            new_dp, new_path = {}, {}
            options = options_for_stage(sp, allowed_sources)
            if options.empty:
                print(f"  [{i+1}/{len(STATIONS)}] {sp} -> no options in this stage")
                all_opts = df_menu[df_menu['站间区间'] == sp]
                if all_opts.empty:
                    print(f"      reason: menu file has no rows for this section ({MENU_FILE.name})")
                else:
                    classes = sorted(all_opts['运行等级'].dropna().astype(str).unique().tolist())
                    sources = sorted(all_opts[CURVE_SOURCE_COL].dropna().astype(str).unique().tolist())
                    print(f"      available classes: {classes}")
                    print(f"      available sources: {sources}; allowed sources now: {sorted(allowed_sources)}")
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
                    if t_sum not in new_dp or score_key(score, score_mode) < score_key(new_dp[t_sum], score_mode):
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

        feasible = [(t, dp[t]) for t in dp.keys() if min_run_int <= t <= max_run_int]
        if not feasible:
            min_r = min(dp.keys()) / 10.0
            max_r = max(dp.keys()) / 10.0
            print(
                f"  no feasible state. Reachable run time: [{min_r:.1f}s, {max_r:.1f}s], "
                f"required: [{min_run_time:.1f}s, {max_run_time_for_stage:.1f}s]"
            )
            return None

        def feasible_key(item):
            t_int, score = item
            dwell_deviation = abs((T_TOTAL_TARGET - t_int / 10.0) - historical_total_dwell)
            ext_count, int_count, energy_wh = score
            if score_mode == "energy_first":
                return energy_wh, ext_count, int_count, dwell_deviation
            if score_mode == "real_priority_energy":
                return ext_count, int_count, energy_wh, dwell_deviation
            return ext_count, int_count, dwell_deviation, energy_wh

        feasible.sort(key=feasible_key)
        best_t_int, best_score = feasible[0]
        return {
            "stage_name": stage_name,
            "score_mode": score_mode,
            "dp": dp,
            "path": path,
            "feasible": feasible,
            "best_t_int": best_t_int,
            "final_energy": float(best_score[2]),
            "extrapolated_count": int(best_score[0]),
            "interpolated_count": int(best_score[1]),
            "dwell_mode": dwell_mode,
            "dwell_tolerance": dwell_tolerance,
        }

    def build_dwell_stages(total_time_tolerance: float):
        return [
            (
                "historical_exact",
                0.0,
                nom_run_time - total_time_tolerance,
                nom_run_time + total_time_tolerance,
            ),
            (
                f"historical_tolerance_{HISTORICAL_DWELL_TOLERANCE * 100:.1f}pct",
                HISTORICAL_DWELL_TOLERANCE,
                T_TOTAL_TARGET - tolerance_max_total - total_time_tolerance,
                T_TOTAL_TARGET - tolerance_min_total + total_time_tolerance,
            ),
        ]

    def find_priority_solution(stages):
        for dwell_mode, dwell_tolerance, min_run_for_stage, max_run_for_stage in stages:
            for stage_name, allowed_sources in DP_SOURCE_STAGES:
                candidate = run_dp_stage(
                    stage_name,
                    allowed_sources,
                    min_run_for_stage,
                    max_run_for_stage,
                    dwell_mode,
                    dwell_tolerance,
                )
                if candidate is not None:
                    return candidate
        return None

    # First enforce the historical target exactly.  A total-time tolerance is a
    # last resort only after exact dwell and the dwell-tolerance band both fail.
    dwell_stages = build_dwell_stages(0.0)
    solution = find_priority_solution(dwell_stages)
    if solution is None and TOTAL_TIME_TOLERANCE > 0:
        print(f"\nTrying nearest total time within +/-{TOTAL_TIME_TOLERANCE:.1f}s.")
        dwell_stages = build_dwell_stages(TOTAL_TIME_TOLERANCE)
        solution = find_priority_solution(dwell_stages)

    if solution is None:
        print("ERROR: No DP stage can fit historical dwell, including the configured tolerance band.")
        print_input_diagnostics(
            df_menu,
            df_hist,
            max_run_time,
            nom_run_time,
            historical_total_dwell,
            tolerance_min_total,
        )
        return False

    def print_solution_overview(solution_info: dict, label: str) -> None:
        stage_name = solution_info["stage_name"]
        dp = solution_info["dp"]
        feasible = solution_info["feasible"]
        best_t_int = solution_info["best_t_int"]
        final_energy = solution_info["final_energy"]
        score_mode = solution_info["score_mode"]

        print(f"\nSelected {label}: {stage_name}")
        print(f"Feasible states: {len(feasible)}")
        print(
            f"Best:  run={best_t_int/10.0:.1f}s, energy={final_energy:.1f} Wh, "
            f"extrapolated={solution_info['extrapolated_count']}, interpolated={solution_info['interpolated_count']}"
        )

        nom_int = to_int(nom_run_time)
        nearby = [(t, dp[t]) for t in dp.keys() if nom_int - to_int(SLACK) <= t <= nom_int + to_int(SLACK)]
        if nearby:
            nearby.sort(key=lambda x: score_key(x[1], score_mode))
            nom_best_t, nom_best_score = nearby[0]
            nom_best_e = float(nom_best_score[2])
            print(f"Nominal-dwell best nearby: run={nom_best_t/10.0:.1f}s, energy={nom_best_e:.1f} Wh")
            saving = nom_best_e - final_energy
            extra_run = (best_t_int - nom_best_t) / 10.0
            if nom_best_e:
                print(f"Dwell trade: +{extra_run:.1f}s run time -> saves {saving:.1f} Wh ({saving/nom_best_e*100:.1f}%)")

    def write_solution_report(solution_info: dict, report_path: Path) -> tuple[list[dict], dict]:
        stage_name = solution_info["stage_name"]
        path = solution_info["path"]
        best_t_int = solution_info["best_t_int"]

        dwell_ok, best_dwells = distribute_historical_dwell(
            best_t_int / 10.0,
            historical_dwells[:-1],
            solution_info["dwell_tolerance"],
        )
        if not dwell_ok:
            raise RuntimeError(f"{stage_name} selected an infeasible dwell distribution.")

        final_rows = []
        curr_t = best_t_int
        for i in range(len(STATIONS) - 1, -1, -1):
            prev_t, c_name, t_val, e_val, source_meta = path[i][curr_t]
            sp = STATIONS[i]
            hist_row = df_hist.loc[sp]
            h_time = hist_row['历史运行时间(s)']
            h_raw_energy = hist_row[HIST_RAW_ENERGY_COL]
            h_model_energy = hist_row[HIST_MODEL_ENERGY_COL] if HIST_MODEL_ENERGY_COL in df_hist.columns else np.nan

            if i < len(STATIONS) - 1:
                dwell = best_dwells[i]
            else:
                dwell = 0.0

            final_rows.append({
                "站间区间": sp,
                "选定等级": c_name,
                "规划用时(s)": round(t_val, 1),
                "停站时间(s)": round(dwell, 2) if i < len(STATIONS) - 1 else 0,
                "历史停站时间(s)": round(historical_dwells[i], 1) if i < len(STATIONS) - 1 else 0,
                "停站偏差(s)": round(dwell - historical_dwells[i], 2) if i < len(STATIONS) - 1 else 0,
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
                "全局趟次ID": hist_row.get("全局趟次ID", ""),
                "历史segment": hist_row.get("历史segment", ""),
                "历史来源run_id": hist_row.get("历史来源run_id", ""),
            })
            curr_t = prev_t

        final_rows.reverse()

        run_exact = best_t_int / 10.0
        run_backtrack_sum = sum(r['规划用时(s)'] for r in final_rows)
        drift = run_exact - run_backtrack_sum
        if abs(drift) > 0.01:
            n_adj = int(round(abs(drift) * 10))
            step = 1 if drift > 0 else -1
            for j in range(n_adj):
                final_rows[j % len(final_rows)]['规划用时(s)'] += step * 0.1
            for r in final_rows:
                r['规划用时(s)'] = round(r['规划用时(s)'], 1)

        df_res = pd.DataFrame(final_rows)
        total_h_e = df_res['历史能耗(Wh)'].sum()
        total_h_model_e = df_res['历史模型回放能耗(Wh)'].sum(skipna=True) if '历史模型回放能耗(Wh)' in df_res.columns else np.nan
        total_p_e = df_res['规划能耗(Wh)'].sum()
        total_p_run = df_res['规划用时(s)'].sum()
        total_p_dwell = df_res['停站时间(s)'].sum()
        total_p_t = total_p_run + total_p_dwell
        total_h_dwell = df_res['历史停站时间(s)'].sum()
        total_h_t = df_res['历史用时(s)'].sum() + total_h_dwell
        saving_rate = (total_h_e - total_p_e) / total_h_e * 100

        summary_row = {
            "站间区间": "--- 总计 ---",
            "选定等级": f"节能率: {saving_rate:.2f}%",
            "规划用时(s)": total_p_run,
            "停站时间(s)": total_p_dwell,
            "历史停站时间(s)": total_h_dwell,
            "停站偏差(s)": round(total_p_dwell - total_h_dwell, 2),
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
            "MASS": "",
        }
        df_res = pd.concat([df_res, pd.DataFrame([summary_row])], ignore_index=True)
        df_res.to_csv(report_path, index=False, encoding='utf-8-sig')

        totals = {
            "run": total_p_run,
            "dwell": total_p_dwell,
            "total_time": total_p_t,
            "energy": total_p_e,
            "saving_rate": saving_rate,
        }
        return final_rows, totals

    print_solution_overview(solution, "priority-first plan")
    final_rows, totals = write_solution_report(solution, FINAL_COMPARISON_FILE)

    print(
        f"\nPriority-first report: Run: {totals['run']:.1f}s | Dwell: {totals['dwell']:.1f}s | "
        f"Total: {totals['total_time']:.1f}s (target: {T_TOTAL_TARGET}s)"
    )
    print(f"Energy: {totals['energy']:.1f} Wh | Saving: {totals['saving_rate']:.2f}%")
    for i, row in enumerate(final_rows):
        if i >= len(final_rows):
            break
        d = row['停站时间(s)']
        if d > 0:
            h_dwell = row['历史停站时间(s)']
            print(f"  Dwell after {row['站间区间']}: {d:.1f}s (history {h_dwell:.1f}s)")

    # Additional comparison plan: keep real curves as the first priority, allow
    # every historical dwell value to move within +/-5%, then minimize energy.
    # Unlike the default priority plan, dwell closeness is not ranked ahead of
    # energy inside this wider band; otherwise a 5% experiment would usually
    # reproduce the 2% result and reveal no usable energy-saving headroom.
    real_priority_min_dwell = historical_total_dwell * (1.0 - REAL_PRIORITY_DWELL_TOLERANCE)
    real_priority_max_dwell = historical_total_dwell * (1.0 + REAL_PRIORITY_DWELL_TOLERANCE)

    def find_real_priority_energy_solution(total_time_tolerance: float):
        min_run = T_TOTAL_TARGET - real_priority_max_dwell - total_time_tolerance
        max_run = T_TOTAL_TARGET - real_priority_min_dwell + total_time_tolerance
        dwell_mode = f"historical_tolerance_{REAL_PRIORITY_DWELL_TOLERANCE * 100:.1f}pct_energy"
        for stage_name, allowed_sources in DP_SOURCE_STAGES:
            candidate = run_dp_stage(
                stage_name,
                allowed_sources,
                min_run,
                max_run,
                dwell_mode,
                REAL_PRIORITY_DWELL_TOLERANCE,
                score_mode="real_priority_energy",
            )
            if candidate is not None:
                return candidate
        return None

    real_priority_solution = find_real_priority_energy_solution(0.0)
    if real_priority_solution is None and TOTAL_TIME_TOLERANCE > 0:
        real_priority_solution = find_real_priority_energy_solution(TOTAL_TIME_TOLERANCE)

    if real_priority_solution is not None:
        print_solution_overview(real_priority_solution, "real-priority dwell-5pct plan")
        real_priority_rows, real_priority_totals = write_solution_report(
            real_priority_solution,
            REAL_PRIORITY_5PCT_COMPARISON_FILE,
        )
        print(
            f"\nReal-priority dwell-5pct report: Run: {real_priority_totals['run']:.1f}s | "
            f"Dwell: {real_priority_totals['dwell']:.1f}s | "
            f"Total: {real_priority_totals['total_time']:.1f}s (target: {T_TOTAL_TARGET}s)"
        )
        print(
            f"Energy: {real_priority_totals['energy']:.1f} Wh | "
            f"Saving: {real_priority_totals['saving_rate']:.2f}%"
        )
    else:
        real_priority_rows = None
        real_priority_totals = None
        print("\nReal-priority dwell-5pct report skipped: no feasible staged DP solution.")

    energy_solution = None
    for dwell_mode, dwell_tolerance, min_run_for_stage, max_run_for_stage in dwell_stages:
        energy_solution = run_dp_stage(
            "energy_first",
            {"real", "interpolated", "extrapolated_adjacent"},
            min_run_for_stage,
            max_run_for_stage,
            dwell_mode,
            dwell_tolerance,
            score_mode="energy_first",
        )
        if energy_solution is not None:
            break
    if energy_solution is not None:
        print_solution_overview(energy_solution, "energy-first plan")
        energy_rows, energy_totals = write_solution_report(energy_solution, ENERGY_FIRST_COMPARISON_FILE)
        print(
            f"\nEnergy-first report: Run: {energy_totals['run']:.1f}s | Dwell: {energy_totals['dwell']:.1f}s | "
            f"Total: {energy_totals['total_time']:.1f}s (target: {T_TOTAL_TARGET}s)"
        )
        print(f"Energy: {energy_totals['energy']:.1f} Wh | Saving: {energy_totals['saving_rate']:.2f}%")
    else:
        energy_rows = None
        energy_totals = None
        print("\nEnergy-first report skipped: no feasible all-source DP solution.")

    # ===================== 4. Plotting =====================
    def plot_solution(solution_rows: list[dict], solution_totals: dict, output_path: Path, title_prefix: str) -> None:
        print(f"\nPlotting {title_prefix} full-line comparison ...")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        t_opt_acc = 0.0
        s_opt_acc = 0.0
        t_hist_acc = 0.0
        s_hist_acc = 0.0
        plot_data = {'opt_t': [], 'opt_v': [], 'opt_s': [],
                     'hist_t': [], 'hist_v': [], 'hist_s': []}
        dwell_zones = []

        for i, row in enumerate(solution_rows):
            if i >= len(STATIONS):
                break
            sp = row['站间区间']
            c_name = row['选定等级']

            df_opt = pd.read_csv(TRAJ_BASE_DIR / sp / f"{c_name}_generated_curve.csv")
            plot_data['opt_t'].extend((df_opt['time_s'] + t_opt_acc).tolist())
            plot_data['opt_v'].extend((df_opt['velocity_mps'] * 3.6).tolist())
            plot_data['opt_s'].extend((df_opt['dist_m'] + s_opt_acc).tolist())

            df_h = load_history_trip_curve(sp)
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
                dwell_hist = historical_dwells[i]

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
                 label=title_prefix)
        for start, end in dwell_zones:
            ax1.axvspan(start, end, color='gray', alpha=0.05)
        ax1.set_title(
            f"{title_prefix} v-t | total={solution_totals['total_time']:.1f}s "
            f"(target={T_TOTAL_TARGET}s) | saving={solution_totals['saving_rate']:.2f}%"
        )
        ax1.set_ylabel("Velocity (km/h)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        ax2.plot(plot_data['hist_s'], plot_data['hist_v'], color='gray', alpha=0.4,
                 linewidth=1.0, label='historical')
        ax2.plot(plot_data['opt_s'], plot_data['opt_v'], color='blue', linewidth=1.2,
                 label=title_prefix)
        ax2.set_title(f"{title_prefix} v-s | distance={s_opt_acc:.0f}m")
        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("Velocity (km/h)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    if SKIP_PLOTS:
        print("Skipping DP plot outputs because ENERGY_SKIP_PLOTS=1.")
    else:
        plot_solution(final_rows, totals, FINAL_REPORT_PNG, "priority-first plan")
        if real_priority_rows is not None and real_priority_totals is not None:
            plot_solution(
                real_priority_rows,
                real_priority_totals,
                REAL_PRIORITY_5PCT_REPORT_PNG,
                "real-priority dwell-5pct plan",
            )
        if energy_rows is not None and energy_totals is not None:
            plot_solution(energy_rows, energy_totals, ENERGY_FIRST_REPORT_PNG, "energy-first plan")

    print(
        f"Done. Energy: {totals['energy']/1000:.3f} kWh, "
        f"Time: {totals['total_time']:.1f}s (strict = {T_TOTAL_TARGET}s)"
    )
    return True


if __name__ == "__main__":
    ok = run_optimization()
    sys.exit(0 if ok else 1)
