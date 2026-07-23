from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"D:\energy_conservation")
MAIN_SCRIPT = PROJECT_ROOT / "scripts_new" / "main.py"
TIMETABLE_SCRIPT = PROJECT_ROOT / "scripts_new" / "09_exports_opentrack" / "make_opentrack_timetables.py"
FINAL_REPORT_DIR = PROJECT_ROOT / "output" / "schedule" / "final_plan_report_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "schedule" / "batch_trip_reports"
DEFAULT_TEMPLATE = PROJECT_ROOT / "eg.xml"

SECTION_COL = "\u7ad9\u95f4\u533a\u95f4"
HIST_RUN_TIME_COL = "\u5386\u53f2\u8fd0\u884c\u65f6\u95f4(s)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-build per-trip DP planning comparison tables and OpenTrack timetables. "
            "For each trip, the DP target time is calculated from that trip's historical "
            "running time plus nominal dwell time."
        )
    )
    parser.add_argument("--trips", required=True, help="Trips to run, e.g. 1,2,5-10.")
    parser.add_argument("--data-dir", default=None, help="Step2 results_*.xlsx data directory.")
    parser.add_argument("--ato-data-dir", default=None, help="Optional ATO data directory. Defaults to --data-dir.")
    parser.add_argument(
        "--traceability-manifest",
        default=None,
        help=(
            "Global trip traceability CSV. If omitted, main.py searches inside --data-dir "
            "and its sibling *_traceability directory."
        ),
    )
    parser.add_argument("--line-scope", default="full", help="'full' or first N sections, e.g. 5.")
    parser.add_argument("--history-dwell", type=float, default=30.0, help="Historical dwell seconds between sections.")
    parser.add_argument(
        "--historical-dwell-csv",
        default=None,
        help="Trip/station historical dwell detail CSV. Missing values use the station median, then --history-dwell.",
    )
    parser.add_argument(
        "--dwell-tolerance-pct",
        type=float,
        default=2.0,
        help="Per-station historical dwell tolerance used only when exact dwell is infeasible.",
    )
    parser.add_argument(
        "--total-time-tolerance-s",
        type=float,
        default=0.0,
        help="Allow the nearest feasible total time within this many seconds; dwell limits remain unchanged.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Batch output directory.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="OpenTrack timetable template XML.")
    parser.add_argument("--start-time", default="08:00:00", help="Timetable departure time at STA_1.")
    parser.add_argument(
        "--history-dwell-mode",
        choices=["fixed", "csv", "zero"],
        default="fixed",
        help="Historical timetable dwell mode. fixed uses --history-dwell.",
    )
    parser.add_argument(
        "--prep-steps",
        default="energy_menu,historical_baseline",
        help="Main pipeline steps to run before target calculation.",
    )
    parser.add_argument("--skip-prep", action="store_true", help="Reuse existing energy menu and historical baseline.")
    parser.add_argument(
        "--reuse-existing-prep",
        action="store_true",
        help="Skip prep for a trip when both energy menu and historical baseline already exist.",
    )
    parser.add_argument("--skip-dp", action="store_true", help="Reuse existing final_plan_report_v2 outputs.")
    parser.add_argument("--skip-timetable", action="store_true", help="Do not generate OpenTrack timetable XML files.")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate DP PNG plots while batch building tables.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip trip folders that already contain final tables.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with later trips if one trip fails.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    return parser.parse_args()


def parse_trip_list(text: str) -> list[int]:
    trips: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = [int(x.strip()) for x in item.split("-", 1)]
            if left <= 0 or right <= 0 or right < left:
                raise ValueError(f"Invalid trip range: {item}")
            trips.extend(range(left, right + 1))
        else:
            value = int(item)
            if value <= 0:
                raise ValueError("--trips must use 1-based positive trip numbers.")
            trips.append(value)
    if not trips:
        raise ValueError("No trips selected.")
    return list(dict.fromkeys(trips))


def resolve_path(value: str | None, base: Path = PROJECT_ROOT) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def line_scope_count(line_scope: str, available_sections: int) -> int:
    raw = str(line_scope).strip().lower()
    if raw in {"full", "all"}:
        return available_sections
    if raw == "first5":
        return min(5, available_sections)
    count = int(raw)
    if count <= 0:
        raise ValueError("--line-scope count must be positive.")
    return min(count, available_sections)


def find_column(df: pd.DataFrame, exact: str, must_contain: list[str] | None = None) -> str:
    if exact in df.columns:
        return exact
    must_contain = must_contain or []
    for col in df.columns:
        text = str(col)
        if all(piece in text for piece in must_contain):
            return col
    raise ValueError(f"Could not find required column: {exact}")


def load_trip_dwell_values(
    dwell_csv: Path | None,
    trip_no: int,
    sections: list[str],
    fallback_seconds: float,
) -> tuple[list[float], list[str]]:
    if dwell_csv is None:
        return [fallback_seconds] * max(len(sections) - 1, 0), ["fixed_default"] * max(len(sections) - 1, 0)
    if not dwell_csv.exists():
        raise FileNotFoundError(f"Historical dwell detail CSV not found: {dwell_csv}")

    detail = pd.read_csv(dwell_csv, encoding="utf-8-sig")
    required = {"趟次", "停车站", "上一运行区间", "历史停站时间(s)"}
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Historical dwell detail CSV missing columns: {missing}")

    detail = detail.copy()
    detail["趟次"] = pd.to_numeric(detail["趟次"], errors="coerce")
    detail["历史停站时间(s)"] = pd.to_numeric(detail["历史停站时间(s)"], errors="coerce")
    normal = detail[detail["历史停站时间(s)"].between(20.0, 60.0, inclusive="both")]
    station_medians = normal.groupby("停车站")["历史停站时间(s)"].median().to_dict()
    selected = detail[detail["趟次"] == trip_no]

    values: list[float] = []
    sources: list[str] = []
    for section in sections[:-1]:
        rows = selected[selected["上一运行区间"].astype(str).str.strip() == section]
        actual = rows["历史停站时间(s)"].dropna()
        if not actual.empty and float(actual.iloc[0]) > 0:
            values.append(float(actual.iloc[0]))
            sources.append("actual")
            continue
        station = section.split("-", 1)[1]
        median = station_medians.get(station)
        if median is not None and pd.notna(median) and float(median) > 0:
            values.append(float(median))
            sources.append("station_median")
        else:
            values.append(fallback_seconds)
            sources.append("fixed_default")
    return values, sources


def compute_target_time(
    history_csv: Path,
    line_scope: str,
    dwell_seconds: float,
    trip_no: int,
    dwell_csv: Path | None,
) -> dict[str, Any]:
    if not history_csv.exists():
        raise FileNotFoundError(f"Historical baseline not found: {history_csv}")

    df = pd.read_csv(history_csv, encoding="utf-8-sig")
    section_col = find_column(df, SECTION_COL, ["\u7ad9", "\u533a"])
    run_col = find_column(df, HIST_RUN_TIME_COL, ["\u5386\u53f2", "\u65f6\u95f4"])

    section_text = df[section_col].astype(str).str.strip()
    df = df[section_text.str.contains("-", regex=False)].copy()
    if df.empty:
        raise ValueError(f"No section rows found in {history_csv}")

    count = line_scope_count(line_scope, len(df))
    selected = df.iloc[:count].copy()
    selected[run_col] = pd.to_numeric(selected[run_col], errors="coerce")
    if selected[run_col].isna().any():
        bad_sections = selected.loc[selected[run_col].isna(), section_col].astype(str).tolist()
        raise ValueError(f"Historical run time has NaN values for: {bad_sections}")

    run_total = float(selected[run_col].sum())
    sections = selected[section_col].astype(str).str.strip().tolist()
    dwell_values, dwell_sources = load_trip_dwell_values(dwell_csv, trip_no, sections, dwell_seconds)
    dwell_total = float(sum(dwell_values))
    return {
        "target_time_s": run_total + dwell_total,
        "run_time_s": run_total,
        "dwell_time_s": dwell_total,
        "section_count": count,
        "dwell_actual_count": dwell_sources.count("actual"),
        "dwell_fallback_count": len(dwell_sources) - dwell_sources.count("actual"),
    }


def build_main_command(args: argparse.Namespace, trip_no: int, only_steps: str, target_time: float | None = None) -> list[str]:
    command = [args.python, str(MAIN_SCRIPT), "--only", only_steps, "--trip-no", str(trip_no), "--line-scope", str(args.line_scope)]
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.ato_data_dir:
        command.extend(["--ato-data-dir", args.ato_data_dir])
    elif args.data_dir:
        command.extend(["--ato-data-dir", args.data_dir])
    if args.traceability_manifest:
        command.extend(["--traceability-manifest", args.traceability_manifest])
    if target_time is not None:
        command.extend(["--target-time", f"{target_time:.3f}"])
    return command


def run_command(command: list[str], log_path: Path, dry_run: bool, env_extra: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    if env_extra:
        printable = " ".join(f"{key}={value}" for key, value in sorted(env_extra.items())) + " " + printable
    print(f"Command: {printable}")
    print(f"Log:     {log_path}")
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return 0

    child_env = None
    if env_extra:
        child_env = os.environ.copy()
        child_env.update(env_extra)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )
    return proc.returncode


def prep_outputs_exist(trip_no: int) -> bool:
    history_csv = PROJECT_ROOT / f"full_line{trip_no}_validation_results.csv"
    energy_menu = PROJECT_ROOT / "output" / "analysis" / f"ato_class_energy_menu{trip_no}_new_v3.csv"
    return history_csv.exists() and energy_menu.exists()


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def archive_trip_outputs(trip_dir: Path, trip_no: int) -> dict[str, str]:
    copied: dict[str, str] = {}
    sources = {
        "priority_csv": FINAL_REPORT_DIR / "Final_Planning_Comparison.csv",
        "energy_first_csv": FINAL_REPORT_DIR / "Final_Planning_Comparison_Energy_First.csv",
        "real_priority_5pct_csv": FINAL_REPORT_DIR / "Final_Planning_Comparison_Real_Priority_Dwell_5pct.csv",
        "priority_png": FINAL_REPORT_DIR / "Optimized_Full_Line_Report.png",
        "energy_first_png": FINAL_REPORT_DIR / "Optimized_Full_Line_Report_Energy_First.png",
        "real_priority_5pct_png": FINAL_REPORT_DIR / "Optimized_Full_Line_Report_Real_Priority_Dwell_5pct.png",
        "history_baseline_csv": PROJECT_ROOT / f"full_line{trip_no}_validation_results.csv",
        "energy_menu_csv": PROJECT_ROOT / "output" / "analysis" / f"ato_class_energy_menu{trip_no}_new_v3.csv",
    }
    destinations = {
        "priority_csv": trip_dir / "Final_Planning_Comparison.csv",
        "energy_first_csv": trip_dir / "Final_Planning_Comparison_Energy_First.csv",
        "real_priority_5pct_csv": trip_dir / "Final_Planning_Comparison_Real_Priority_Dwell_5pct.csv",
        "priority_png": trip_dir / "Optimized_Full_Line_Report.png",
        "energy_first_png": trip_dir / "Optimized_Full_Line_Report_Energy_First.png",
        "real_priority_5pct_png": trip_dir / "Optimized_Full_Line_Report_Real_Priority_Dwell_5pct.png",
        "history_baseline_csv": trip_dir / f"full_line{trip_no}_validation_results.csv",
        "energy_menu_csv": trip_dir / f"ato_class_energy_menu{trip_no}_new_v3.csv",
    }
    for key, src in sources.items():
        dst = destinations[key]
        if copy_if_exists(src, dst):
            copied[key] = str(dst)
    return copied


def build_timetable_command(args: argparse.Namespace, trip_dir: Path, trip_label: str) -> list[str]:
    return [
        args.python,
        str(TIMETABLE_SCRIPT),
        "--priority-csv",
        str(trip_dir / "Final_Planning_Comparison.csv"),
        "--energy-first-csv",
        str(trip_dir / "Final_Planning_Comparison_Energy_First.csv"),
        "--real-priority-5pct-csv",
        str(trip_dir / "Final_Planning_Comparison_Real_Priority_Dwell_5pct.csv"),
        "--template",
        str(resolve_path(args.template) or DEFAULT_TEMPLATE),
        "--output-dir",
        str(trip_dir / "timetables"),
        "--start-time",
        args.start_time,
        "--history-dwell-mode",
        args.history_dwell_mode,
        "--fixed-history-dwell",
        f"{args.history_dwell:.3f}",
        "--write-station-map",
        "--course-suffix",
        trip_label,
        "--file-suffix",
        trip_label,
    ]


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "batch_summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {path}")


def process_trip(args: argparse.Namespace, trip_no: int, output_dir: Path) -> dict[str, Any]:
    trip_label = f"trip{trip_no:03d}"
    trip_dir = output_dir / trip_label
    logs_dir = trip_dir / "logs"
    trip_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and (trip_dir / "Final_Planning_Comparison.csv").exists():
        print(f"\n=== {trip_label}: skip existing ===")
        target_info = {"target_time_s": "", "run_time_s": "", "dwell_time_s": "", "section_count": ""}
        return {"trip_no": trip_no, "trip_label": trip_label, "status": "skipped", **target_info}

    print(f"\n=== {trip_label}: prep ===")
    if args.reuse_existing_prep and prep_outputs_exist(trip_no):
        print(f"{trip_label} reuse existing energy menu and historical baseline.")
    elif not args.skip_prep:
        code = run_command(
            build_main_command(args, trip_no, args.prep_steps),
            logs_dir / "01_prep_energy_menu_and_history.log",
            args.dry_run,
        )
        if code != 0:
            raise RuntimeError(f"{trip_label} prep failed with code {code}.")

    history_csv = PROJECT_ROOT / f"full_line{trip_no}_validation_results.csv"
    dwell_csv = resolve_path(args.historical_dwell_csv) if args.historical_dwell_csv else None
    target_info = compute_target_time(history_csv, args.line_scope, args.history_dwell, trip_no, dwell_csv)
    print(
        f"{trip_label} target = {target_info['target_time_s']:.2f}s "
        f"(run {target_info['run_time_s']:.2f}s + dwell {target_info['dwell_time_s']:.2f}s)"
    )

    print(f"\n=== {trip_label}: dp_schedule ===")
    if not args.skip_dp:
        env_extra = {
            "ENERGY_DWELL_TOLERANCE": f"{args.dwell_tolerance_pct / 100.0:.8f}",
            "ENERGY_TOTAL_TIME_TOLERANCE": f"{args.total_time_tolerance_s:.8f}",
        }
        if args.no_plots:
            env_extra["ENERGY_SKIP_PLOTS"] = "1"
        if dwell_csv is not None:
            env_extra["ENERGY_HISTORICAL_DWELL_FILE"] = str(dwell_csv)
        code = run_command(
            build_main_command(args, trip_no, "dp_schedule", target_info["target_time_s"]),
            logs_dir / "02_dp_schedule.log",
            args.dry_run,
            env_extra=env_extra,
        )
        if code != 0:
            raise RuntimeError(f"{trip_label} dp_schedule failed with code {code}.")

    copied = {} if args.dry_run else archive_trip_outputs(trip_dir, trip_no)
    if not args.dry_run:
        required = {"priority_csv", "energy_first_csv", "real_priority_5pct_csv"}
        missing = sorted(required - set(copied))
        if missing:
            raise FileNotFoundError(f"{trip_label} missing archived outputs: {missing}")

    print(f"\n=== {trip_label}: timetables ===")
    if not args.skip_timetable:
        code = run_command(
            build_timetable_command(args, trip_dir, trip_label),
            logs_dir / "03_make_timetables.log",
            args.dry_run,
        )
        if code != 0:
            raise RuntimeError(f"{trip_label} timetable export failed with code {code}.")

    return {
        "trip_no": trip_no,
        "trip_label": trip_label,
        "status": "ok",
        **target_info,
        "trip_dir": str(trip_dir),
    }


def main() -> int:
    args = parse_args()
    trips = parse_trip_list(args.trips)
    output_dir = resolve_path(args.output_dir) or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Trips:      {trips}")
    print(f"Output dir: {output_dir}")
    print(f"Started:    {datetime.now():%Y-%m-%d %H:%M:%S}")

    summary_rows: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []
    for trip_no in trips:
        try:
            row = process_trip(args, trip_no, output_dir)
            summary_rows.append(row)
        except Exception as exc:
            message = str(exc)
            print(f"ERROR trip {trip_no}: {message}")
            failures.append((trip_no, message))
            summary_rows.append({"trip_no": trip_no, "trip_label": f"trip{trip_no:03d}", "status": "failed", "error": message})
            if not args.continue_on_error:
                break

    write_summary(output_dir, summary_rows)
    if failures:
        print("\nFailures:")
        for trip_no, message in failures:
            print(f"- trip {trip_no}: {message}")
        return 1
    print("\nBatch finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
