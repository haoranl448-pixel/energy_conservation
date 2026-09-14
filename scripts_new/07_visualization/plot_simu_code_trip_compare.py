# -*- coding: utf-8 -*-
"""Replay frozen DP plans and compare their energy with current Simu physics.

No training, DP optimization, timetable generation, or OpenTrack IO occurs here.
The original menu's physical version is identified by reproducing its section
total using the unchanged residual checkpoint. A numerically close replay keeps
its model-computed pointwise profile and reconciles only the final section point
to the archived DP total; larger mismatches remain blank inside the section.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "scripts_new/00_main_pipeline",
               ROOT / "scripts_new/09_exports_opentrack", ROOT / "output/cache/replay_runtime"):
    sys.path.insert(0, str(folder))
os.environ.setdefault("ENERGY_RESULTS_DATA_DIR", str(ROOT / "data/data_processed_step2_v3_all_curve_quality_traceability"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch

from src.physics.train_simu import TrainTheoreticalEnergyModel as CurrentPhysics
from src.physics.train_simu_v1_backup_20260828 import TrainTheoreticalEnergyModel as LegacyPhysics
from trip_traceability import load_trip_traceability, select_trip_rows
from plot_simu_ot_code_trip_compare import configure_matplotlib, read_generated_curve
from plot_trip_real_vs_simu_physical_es import clean_section_frame

MENU_SCRIPT = ROOT / "scripts_new/00_main_pipeline/06_ato_generated_results_energy.py"
spec = importlib.util.spec_from_file_location("frozen_menu_replay", MENU_SCRIPT)
menu_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(menu_module)

TRIPS = "1,3,4,5,6,8,10,12,14,15,16,19,21,35,40,42,44,49,50,53,57,60,62,66,67"
OLD_TRIPS = {1, 3, 4, 5, 6, 8, 10, 12, 14, 15, 42, 62, 67}
SCENARIOS = {
    "history": ("00_history", "历史运行", None, None),
    "standard_dp": ("01_standard_dp", "标准 DP", "01_standard_priority.csv", "Final_Planning_Comparison.csv"),
    "energy_first": ("02_energy_first", "Energy First", "02_energy_first.csv", "Final_Planning_Comparison_Energy_First.csv"),
    "dwell5_dp": ("03_dwell5_dp", "停站放宽 5% DP", "03_real_priority_dwell_5pct.csv", "Final_Planning_Comparison_Real_Priority_Dwell_5pct.csv"),
}
SAVING_LABELS = {
    "standard_dp": "标准 DP 预计节能量",
    "energy_first": "Energy First 预计节能量",
    "dwell5_dp": "停站放宽 5% 预计节能量",
}
SPEED_LABELS = {
    "standard_dp": "标准 DP 规划速度",
    "energy_first": "Energy First 规划速度",
    "dwell5_dp": "停站放宽 5% 规划速度",
}
SECTIONS = CurrentPhysics.STATION_PAIRS
STARTS = CurrentPhysics.SECTION_START_DISTANCES_M
ENDS = np.r_[STARTS[1:], 36502.0]
REPLAY_VERSION = "simu-code-pointwise-v2"
RECONCILIATION_TOLERANCE_PCT = 0.2
_PROFILE_MEMO = {}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


class VectorizedResistance:
    """Batch the 120 spatial lookups; leave the model time loop unchanged."""
    slice_offset = 0.0

    def __init__(self):
        super().__init__()
        assert np.all(np.diff(self.real_grad_locs) > 0)
        assert np.all(np.diff(self.real_curve_locs) > 0)
        self.offsets = (np.arange(self.num_cars)[:, None] * self.car_length
                        + (np.arange(20)[None, :] + self.slice_offset) * self.car_length / 20)

    def _calc_gradient_distributed(self, current_dist, loads):
        gradient = np.interp(current_dist - self.offsets, self.real_grad_locs, self.real_grad_vals)
        terms = np.asarray(loads)[:, None] * np.sin(np.arctan(gradient / 1000)) / 20
        return float(np.sum(np.sum(terms, axis=0) * self.g) * 1000)

    def _calc_curve_distributed(self, current_dist, loads):
        radius = np.interp(current_dist - self.offsets, self.real_curve_locs, self.real_curve_vals)
        terms = np.zeros_like(radius)
        numerator = 600 * self.g * np.asarray(loads)[:, None]
        np.divide(numerator, radius, out=terms, where=(radius > 0) & (radius < 2000))
        return float(np.sum(np.sum(terms / 1000 / 20, axis=1)) * 1000)


class FastCurrent(VectorizedResistance, CurrentPhysics):
    pass


class FastLegacy(VectorizedResistance, LegacyPhysics):
    slice_offset = 0.5


def validate_fast_physics():
    # Deterministic numerical regression, not observations used in the figures.
    velocity = np.r_[np.zeros(4), np.linspace(0.01, 18, 80), np.full(30, 18), np.linspace(18, 0, 70)]
    time = np.arange(len(velocity)) * 0.05
    errors = []
    for base, fast in ((CurrentPhysics, FastCurrent), (LegacyPhysics, FastLegacy)):
        kwargs = {"section_name": SECTIONS[23]} if base is CurrentPhysics else {}
        a = np.asarray(base().run_batch_simulation(time, velocity, 247.7, **kwargs))
        b = np.asarray(fast().run_batch_simulation(time, velocity, 247.7, **kwargs))
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-10)
        errors.append(float(np.max(np.abs(a - b))))
    return errors


def read_plan(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    selected = frame.loc[frame["站间区间"].isin(SECTIONS)].copy()
    if len(selected) != 26 or selected["站间区间"].duplicated().any():
        raise ValueError(f"Expected 26 unique sections: {path}")
    return selected.set_index("站间区间").loc[list(SECTIONS)]


def plan_path(trip, scenario):
    if trip in OLD_TRIPS:
        folder = "batch_trip_reports_complete13_dp_unified"
        filename = SCENARIOS[scenario][2]
    else:
        folder = "batch_trip_reports_complete25_replacement53" if trip == 53 else "batch_trip_reports_complete25_new12"
        filename = SCENARIOS[scenario][3]
    return ROOT / "output/schedule" / folder / f"trip{trip:03d}" / filename


def percentage(delta, reference):
    return float(delta / reference * 100) if abs(reference) > 1e-12 else np.nan


def line_distance(local, index):
    local = np.asarray(local, float)
    local = local - local[0]
    if not np.all(np.isfinite(local)) or np.any(np.diff(local) < -0.01) or local[-1] <= 0:
        raise ValueError(f"Invalid distance in {SECTIONS[index]}")
    local = np.maximum.accumulate(local)
    # Display alignment only: never change the speed/time fed to physics.
    return (STARTS[index] + local / local[-1] * (ENDS[index] - STARTS[index])) / 1000


def predict_residual(section, time, velocity, distance, mass, physics, maps):
    sx, sy, model = menu_module.get_residual_assets(section)
    acc = np.r_[0.0, np.diff(velocity) / menu_module.DT]
    features = np.stack([velocity, acc, physics, maps[0](distance), np.full_like(velocity, mass), maps[1](distance)], axis=1)
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Nonfinite residual inputs: {section}")
    scaled = sx.transform(features)
    windows = np.array([scaled[i-29:i+1] for i in range(29, len(scaled))])
    with torch.no_grad():
        prediction = model(torch.tensor(windows, dtype=torch.float32)).numpy()
    result = np.zeros(len(time))
    result[29:] = sy.inverse_transform(prediction).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Nonfinite residual outputs: {section}")
    return result


def reproduce_plan(section, curve, mass, target_wh, maps, cache_dir, signature):
    key = digest([signature, section, mass, target_wh, fingerprint(curve)])
    path = cache_dir / f"{key}.npz"
    if path.exists():
        with np.load(path) as cached:
            result = {k: cached[k] for k in cached.files}
        if not bool(result["matched"]):
            error = float(result["reproduction_error_wh"])
            error_pct = percentage(error, target_wh)
            if abs(error_pct) <= RECONCILIATION_TOLERANCE_PCT:
                code = np.asarray(result["code_wh"], dtype=float).copy()
                code[-1] -= error
                result.update(
                    code_wh=code,
                    version=np.array("reconciled_nearest_model"),
                    matched=np.array(True),
                )
        return result
    shared_key = digest([signature, section, mass, fingerprint(curve)])
    shared = _PROFILE_MEMO.get(shared_key)
    if shared is None:
        frame = read_generated_curve(curve)
        time = frame["time_s"].to_numpy(float)
        time = time - time[0]
        velocity = frame["velocity_mps"].to_numpy(float)
        distance = frame["dist_m"].to_numpy(float)
        current = np.asarray(FastCurrent().run_batch_simulation(time, velocity, mass, section_name=section))
        base = dict(time=time, distance=distance, velocity=velocity, simu_wh=current)
        profiles = []
        for version in ("current_v2", "legacy_v1"):
            physical = current if version == "current_v2" else np.asarray(FastLegacy().run_batch_simulation(time, velocity, mass))
            residual = predict_residual(section, time, velocity, distance, mass, physical, maps)
            code = physical.copy()
            code[:29] = 0
            code += residual
            profiles.append((version, code, physical, residual))
        shared = (base, profiles)
        _PROFILE_MEMO[shared_key] = shared
    base, profiles = shared
    result = base.copy()
    candidates = []
    for version, code, physical, residual in profiles:
        error = float(code.sum() - target_wh)
        candidates.append((abs(error), version, error, code, physical, residual))
    _, version, error, code, physical, residual = min(candidates, key=lambda x: x[0])
    error_pct = percentage(error, target_wh)
    matched = abs(error) <= 0.1 or abs(error_pct) <= RECONCILIATION_TOLERANCE_PCT
    if matched:
        code = code.copy()
        code[-1] -= error
    replay_version = version if abs(error) <= 0.1 else "reconciled_nearest_model"
    result.update(code_wh=code, version=np.array(replay_version if matched else "unresolved"),
                  reproduction_error_wh=np.array(error), matched=np.array(matched),
                  matching_versions=np.array(";".join(c[1] for c in candidates if c[0] <= 0.1)),
                  aligned_physics_wh=np.array(physical[29:].sum()), residual_wh=np.array(residual.sum()))
    np.savez_compressed(path, **result)
    return result


def prepare_section_data(args, trips, traces, section, cache_dir):
    path = args.data_dir / f"results_{section}.xlsx"
    stat = path.stat()
    stamp = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    cache = cache_dir / (digest(["raw-v1", stamp, trips, fingerprint(args.manifest)]) + ".pkl")
    if cache.exists():
        return pd.read_pickle(cache), stamp
    wanted = {"segment", "时刻", "速度(m/s)", "累计位移(m)", "energy", "重量", "gradient", "curvature"}
    engine = "calamine" if importlib.util.find_spec("python_calamine") else "openpyxl"
    raw = pd.read_excel(path, engine=engine, usecols=lambda col: col in wanted)
    missing = wanted - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}: {path}")
    frames, counts = {}, {}
    for trip in trips:
        selected, segment, record, mode = select_trip_rows(raw, section, trip-1, traces[trip])
        cleaned = clean_section_frame(selected)
        if len(cleaned) != len(selected):
            raise ValueError(f"Incomplete observations trip{trip} {section}: {len(selected)} -> {len(cleaned)}")
        frames[trip] = cleaned
        counts[trip] = {"input_rows": len(selected), "used_rows": len(cleaned), "segment": segment,
                        "run_id": record.source_run_id, "global_trip_id": record.global_trip_id, "selection": mode}
    map_frame = raw[["累计位移(m)", "gradient", "curvature"]].apply(pd.to_numeric, errors="coerce")
    before = len(map_frame)
    map_frame = map_frame.dropna(subset=["累计位移(m)"])
    result = dict(frames=frames, counts=counts, map=map_frame, map_input=before, map_used=len(map_frame))
    pd.to_pickle(result, cache)
    return result, stamp


def points_frame(index, result, code_wh, trip, scenario):
    x = line_distance(result["distance"], index)
    code = np.cumsum(code_wh) / 1000
    simu = np.cumsum(result["simu_wh"]) / 1000
    return pd.DataFrame({"trip_no": trip, "scenario": scenario, "section_index": index+1,
                         "section": SECTIONS[index], "distance_km": np.r_[x[0], x],
                         "speed_kmh": np.r_[result["velocity"][0]*3.6, result["velocity"]*3.6],
                         "code_section_kwh": np.r_[0, code], "simu_section_kwh": np.r_[0, simu]})


def render_trip(args, trip, trip_stats, intermediate):
    folder = args.output_dir / f"trip{trip:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scenario, (stem, label, _, _) in SCENARIOS.items():
        stats = trip_stats[trip_stats.scenario == scenario].sort_values("section_index")
        frames = []
        offsets = np.zeros(2)
        for index in range(26):
            frame = pd.read_pickle(intermediate / f"trip{trip:03d}_{scenario}_{index:02d}.pkl")
            frame["code_cumulative_kwh"] = offsets[0] + frame.code_section_kwh
            frame["simu_cumulative_kwh"] = offsets[1] + frame.simu_section_kwh
            offsets += [float(stats.iloc[index].code_kwh), float(stats.iloc[index].simu_kwh)]
            frames.append(frame)
        points = pd.concat(frames, ignore_index=True)
        if args.export_data:
            points.to_csv(folder / f"{stem}_points.csv", index=False, encoding="utf-8-sig", float_format="%.9g")
            stats.to_csv(folder / f"{stem}_sections.csv", index=False, encoding="utf-8-sig")
        code_total, simu_total = offsets
        delta = simu_total - code_total
        error = percentage(delta, code_total)
        unresolved = int((stats.code_version == "unresolved").sum())
        history = trip_stats[trip_stats.scenario == "history"]
        dwell = float(history.history_dwell_s.sum())
        runtime = float(history.history_runtime_s.sum())
        history_code_total = float(history.code_kwh.sum())
        history_simu_total = float(history.simu_kwh.sum())
        max_mass = float(stats.mass_t.max())
        fig, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(16, 6.4))
        fig.subplots_adjust(left=0.062, right=0.987, top=0.80, bottom=0.16, wspace=0.20)
        if scenario == "history":
            code_label = "历史实测能耗"
            simulink_energy_label = "Simulink 历史回放机械能耗"
            speed_label = "历史实测速度"
            simulink_speed_label = "Simulink 输入速度"
        else:
            code_label = "神经网络残差修正规划能耗"
            simulink_energy_label = "Simulink 仿真机械能耗"
            speed_label = SPEED_LABELS[scenario]
            simulink_speed_label = "Simulink 输入速度"
        ax_e.plot(points.distance_km, points.code_cumulative_kwh, color="#303030", lw=1.4, label=code_label)
        # Unresolved sections have NaN interiors and known endpoints only.
        if unresolved:
            boundary_x = np.r_[0, ENDS/1000]
            boundary_y = np.r_[0, np.cumsum(stats.code_kwh)]
            ax_e.scatter(boundary_x, boundary_y, color="#303030", s=12, zorder=3)
        ax_e.plot(points.distance_km, points.simu_cumulative_kwh, color="#D1495B", ls="--", lw=1.6,
                  label=simulink_energy_label)
        ax_v.plot(points.distance_km, points.speed_kmh, color="#303030", lw=1.4, label=speed_label)
        ax_v.plot(points.distance_km, points.speed_kmh, color="#D1495B", ls="--", lw=1.2, label=simulink_speed_label)
        for ax, ylabel in ((ax_e, "累计能耗 (kWh)"), (ax_v, "速度 (km/h)")):
            ax.set_xlabel("里程 (km)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.18)
            ax.set_xlim(0, ENDS[-1]/1000)
            if ax is ax_e:
                ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.92)
            else:
                ax.legend(loc="upper right", bbox_to_anchor=(1, 1.10), ncol=2, fontsize=10, frameon=False)
        ax_e.set_ylim(bottom=min(0.0, float(points.code_cumulative_kwh.min()), float(points.simu_cumulative_kwh.min())))
        ax_v.set_ylim(bottom=0)
        ax_e.set_title("a  能耗—里程", loc="left", fontsize=12)
        ax_v.set_title("b  速度—里程", loc="left", fontsize=12)
        fig.suptitle(f"Trip {trip:03d} | {label} | 最大区间列车质量 {max_mass:.2f} t", y=0.966, fontsize=15)
        fig.text(0.5, 0.89, f"历史目标（运行 / 停站）：{runtime:.1f} / {dwell:.1f} s    |    "
                 f"Simulink − {'历史' if scenario == 'history' else '代码'}：{delta:+.2f} kWh / {error:+.2f}%",
                 ha="center", fontsize=11)
        if scenario == "history":
            result_text = "\n".join([
                f"历史实测能耗：{code_total:.2f} kWh",
                f"Simulink 历史回放机械能耗：{simu_total:.2f} kWh",
                "",
                f"Simulink 相对历史误差：{delta:+.2f} kWh ({error:+.2f}%)",
            ])
        else:
            code_saving = history_code_total - code_total
            simu_saving = history_simu_total - simu_total
            result_text = "\n".join([
                f"历史实测能耗：{history_code_total:.2f} kWh",
                f"神经网络残差修正规划能耗：{code_total:.2f} kWh",
                f"Simulink 仿真机械能耗：{simu_total:.2f} kWh",
                "",
                f"{SAVING_LABELS[scenario]}：{code_saving:+.2f} kWh ({percentage(code_saving, history_code_total):+.2f}%)",
                f"Simulink 模型节能量：{simu_saving:+.2f} kWh ({percentage(simu_saving, history_simu_total):+.2f}%)",
            ])
        ax_e.text(
            0.018,
            0.975,
            result_text,
            transform=ax_e.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": "white",
                "edgecolor": "#A8B2BD",
                "alpha": 0.94,
            },
        )
        if scenario == "history":
            note = "历史为实测能耗；Simulink 为正向牵引机械能。数值差异不单独等同于预测误差或转换效率。"
        else:
            note = "代码保留已有 DP 能耗；区间内按原权重逐点复现。差异百分比以代码能耗为分母。"
        if unresolved:
            note = f"{unresolved} 个区间未能复现原菜单，仅显示其已知边界累计值，不对内部能耗插值。"
        fig.text(0.062, 0.055, note, fontsize=10)
        fig.text(0.062, 0.025, "右图两条速度线是同一输入，应重合；各区间里程按统一站界显示，仿真仍使用原始时间和速度。", fontsize=9)
        formats = args.formats.split(",")
        if "pdf" in formats:
            fig.savefig(folder / f"{stem}.pdf", facecolor="white")
        if "svg" in formats:
            fig.savefig(folder / f"{stem}.svg", facecolor="white")
        for fmt in formats:
            if fmt not in {"pdf", "svg"}:
                fig.savefig(folder / f"{stem}.{fmt}", dpi=args.dpi, facecolor="white")
        plt.close(fig)
        summaries.append(dict(trip_no=trip, scenario=scenario, scenario_name=label, sections=26,
                              code_kwh=code_total, simu_kwh=simu_total, difference_kwh=delta,
                              difference_pct=error, abs_difference_kwh=abs(delta), abs_difference_pct=abs(error),
                              max_mass_t=max_mass, history_runtime_s=runtime, history_dwell_s=dwell,
                              unresolved_sections=unresolved, code_versions=";".join(sorted(set(stats.code_version))),
                              figure=str(folder / f"{stem}.png")))
        print(f"Rendered trip{trip:03d} {scenario}: {delta:+.3f} kWh, {error:+.3f}%", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trips", default=TRIPS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/data_processed_step2_v3_all_curve_quality_traceability")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/data_processed_step2_v3_all_curve_quality_traceability/trip_traceability_manifest_v1.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/simu_code_trip_compare_complete25")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "output/cache/simu_code_replay")
    parser.add_argument("--formats", default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--export-data", action="store_true", help="Optional CSV reports; off by default.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted batch from its verified internal cache.")
    args = parser.parse_args()
    for field in ("data_dir", "manifest", "output_dir", "cache_dir"):
        path = getattr(args, field)
        setattr(args, field, path.resolve() if path.is_absolute() else (ROOT / path).resolve())
    trips = sorted(set(int(x) for x in args.trips.split(",")))
    if not trips or min(trips) < 1:
        raise ValueError("Positive trip numbers required")
    configure_matplotlib()
    matplotlib.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.family": "sans-serif",
                               "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"], "savefig.dpi": 300})
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    state_dir = args.cache_dir / ("batch_" + args.output_dir.name)
    state_dir.mkdir(exist_ok=True)
    intermediate = state_dir / "replay_data"
    intermediate.mkdir(exist_ok=True)
    stats_path = state_dir / "section_comparison_summary.csv"
    if not args.render_only:
        numerical_errors = validate_fast_physics()
        print(f"Physics parity passed, max step differences: {numerical_errors}", flush=True)
        plans, traces, provenance = {}, {}, []
        for trip in trips:
            traces[trip] = load_trip_traceability(args.manifest, trip, direction="UP")
            traces[trip].require_sections(SECTIONS)
            for scenario in list(SCENARIOS)[1:]:
                path = plan_path(trip, scenario)
                plans[trip, scenario] = read_plan(path)
                for section, row in plans[trip, scenario].iterrows():
                    record = traces[trip].record_for(section)
                    if str(row["全局趟次ID"]) != record.global_trip_id or float(row["历史segment"]) != float(record.segment):
                        raise ValueError(f"Plan/traceability mismatch: trip{trip} {scenario} {section}")
                provenance.append(dict(trip_no=trip, scenario=scenario, plan_path=str(path), plan_sha256=fingerprint(path)))
        old_manifest_path = state_dir / "input_manifest.csv"
        if args.resume and old_manifest_path.exists():
            previous = pd.read_csv(old_manifest_path)
            if previous.to_dict("records") != provenance:
                raise ValueError("Cannot resume: input plan manifest changed")
        pd.DataFrame(provenance).to_csv(old_manifest_path, index=False, encoding="utf-8-sig")
        physics_hash = fingerprint(ROOT / "src/physics/train_simu.py")
        legacy_hash = fingerprint(ROOT / "src/physics/train_simu_v1_backup_20260828.py")
        curve_paths = set()
        for plan in plans.values():
            for section, row in plan.iterrows():
                curve_paths.add(ROOT / "output/ato_generated_results_new_v4" / section / f"{row['选定等级']}_generated_curve.csv")
        data_stamps = [(s, (args.data_dir / f"results_{s}.xlsx").stat().st_size,
                        (args.data_dir / f"results_{s}.xlsx").stat().st_mtime_ns) for s in SECTIONS]
        batch_signature = digest([REPLAY_VERSION, provenance, physics_hash, legacy_hash, fingerprint(MENU_SCRIPT),
                                  fingerprint(args.manifest), data_stamps,
                                  [(str(p), fingerprint(p)) for p in sorted(curve_paths)]])
        signature_path = state_dir / "batch_signature.txt"
        if args.resume and signature_path.exists() and signature_path.read_text() != batch_signature:
            raise ValueError("Cannot resume: source data or replay inputs changed")
        signature_path.write_text(batch_signature, encoding="ascii")
        stats = pd.read_csv(stats_path).to_dict("records") if args.resume and stats_path.exists() else []
        models_path = state_dir / "model_source_manifest.csv"
        model_sources = pd.read_csv(models_path).to_dict("records") if args.resume and models_path.exists() else []
        for index, section in enumerate(SECTIONS):
            cached_rows = [row for row in stats if row["section_index"] == index+1 and row["trip_no"] in trips]
            cached_models = [row for row in model_sources if row["section"] == section]
            complete = len(cached_rows) == len(trips)*4 and all(
                (intermediate / f"trip{trip:03d}_{scenario}_{index:02d}.pkl").exists() for trip in trips for scenario in SCENARIOS)
            if complete and len(cached_models) == 1:
                for name in ("best_res_model.pth", "scaler_x.pkl", "scaler_y.pkl"):
                    if fingerprint(menu_module.RES_MODEL_BASE / section / name) != cached_models[0][name]:
                        raise ValueError(f"Cannot resume: residual asset changed: {section}")
                print(f"[{index+1:02d}/26] Reusing completed {section}", flush=True)
                continue
            stats = [row for row in stats if row["section_index"] != index+1]
            model_sources = [row for row in model_sources if row["section"] != section]
            _PROFILE_MEMO.clear()
            print(f"[{index+1:02d}/26] Reading/replaying {section}", flush=True)
            raw, stamp = prepare_section_data(args, trips, traces, section, args.cache_dir)
            map_frame = raw["map"]
            maps = tuple(interp1d(map_frame["累计位移(m)"], map_frame[col], kind="nearest", fill_value="extrapolate")
                         for col in ("gradient", "curvature"))
            weights = menu_module.RES_MODEL_BASE / section
            weight_hashes = {name: fingerprint(weights / name) for name in ("best_res_model.pth", "scaler_x.pkl", "scaler_y.pkl")}
            signature = digest([REPLAY_VERSION, stamp, weight_hashes, physics_hash, legacy_hash, fingerprint(MENU_SCRIPT)])
            model_sources.append(dict(section=section, map_input_rows=raw["map_input"], map_used_rows=raw["map_used"], **weight_hashes))
            for trip in trips:
                baseline = plans[trip, "standard_dp"].loc[section]
                record = traces[trip].record_for(section)
                history_frame = raw["frames"][trip]
                for scenario in SCENARIOS:
                    if scenario == "history":
                        time = history_frame["时刻"].to_numpy(float)
                        time = time - time[0]
                        velocity = history_frame["速度(m/s)"].to_numpy(float)
                        mass = float(history_frame["_mass_t"].mean())
                        code = history_frame["energy"].to_numpy(float) / 3600
                        current = FastCurrent().run_batch_simulation(time, velocity, mass, section_name=section)
                        result = dict(time=time, velocity=velocity, distance=history_frame["累计位移(m)"].to_numpy(float),
                                      simu_wh=np.asarray(current))
                        version, reproduction_error, target = "historical_measurement", 0.0, float(code.sum())
                        matched = True
                        planned_time = float(time[-1])
                        class_name = "history"
                        matching_versions = "historical_measurement"
                    else:
                        row = plans[trip, scenario].loc[section]
                        mass = float(row["MASS"])
                        target = float(row["规划能耗(Wh)"])
                        class_name = str(row["选定等级"])
                        curve = ROOT / "output/ato_generated_results_new_v4" / section / f"{class_name}_generated_curve.csv"
                        result = reproduce_plan(section, curve, mass, target, maps, args.cache_dir, signature)
                        code = result["code_wh"]
                        version = str(result["version"])
                        matching_versions = str(result["matching_versions"])
                        reproduction_error = float(result["reproduction_error_wh"])
                        matched = bool(result["matched"])
                        planned_time = float(row["规划用时(s)"])
                        if abs(float(result["time"][-1]) - planned_time) > 0.11:
                            raise ValueError(f"Curve duration mismatch trip{trip} {section} {scenario}")
                    frame = points_frame(index, result, code, trip, scenario)
                    if not matched:
                        frame["code_section_kwh"] = np.nan
                        frame.loc[0, "code_section_kwh"] = 0.0
                        frame.loc[len(frame)-1, "code_section_kwh"] = target / 1000
                        print(f"  UNRESOLVED trip{trip} {scenario}: {reproduction_error:+.3f} Wh", flush=True)
                    frame.to_pickle(intermediate / f"trip{trip:03d}_{scenario}_{index:02d}.pkl")
                    simu = float(np.sum(result["simu_wh"])) / 1000
                    code_total = target / 1000
                    stats.append(dict(trip_no=trip, scenario=scenario, section_index=index+1, section=section,
                                      selected_class=class_name, mass_t=mass, code_kwh=code_total, simu_kwh=simu,
                                      difference_kwh=simu-code_total, difference_pct=percentage(simu-code_total, code_total),
                                      code_version=version, reproduction_error_wh=reproduction_error,
                                      numerically_matching_versions=matching_versions,
                                      runtime_s=planned_time, history_runtime_s=float(baseline["历史用时(s)"]),
                                      history_dwell_s=float(baseline["历史停站时间(s)"]),
                                      input_samples=len(result["time"]), used_samples=len(result["time"]),
                                      dt_max_error_s=float(np.max(np.abs(np.diff(result["time"])-0.05))),
                                      distance_backward_steps=int(np.sum(np.diff(result["distance"]) < 0)),
                                      distance_min_step_m=float(np.min(np.diff(result["distance"]))),
                                      raw_distance_m=float(result["distance"][-1]-result["distance"][0]),
                                      displayed_distance_m=float(ENDS[index]-STARTS[index]),
                                      global_trip_id=record.global_trip_id, historical_segment=record.segment,
                                      historical_run_id=record.source_run_id))
            pd.DataFrame(stats).to_csv(stats_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(model_sources).to_csv(models_path, index=False, encoding="utf-8-sig")
            print(f"[{index+1:02d}/26] Complete: {len(stats)} section/scenario results", flush=True)
        qa = dict(backend="python", archetype="quantitative grid", dimensions_inches=[16, 6.4],
                  claim="Compare current positive mechanical traction energy with frozen code/recorded energy on matched velocity inputs.",
                  panels={"a": "paired cumulative energy", "b": "identical velocity inputs, not a tracking-accuracy test"},
                  layout="Style-only adaptation of the existing user-approved 1x2 report figure. Large report/meeting format, not a journal submission. PNG 300 dpi plus editable PDF/SVG; no TIFF requested.",
                  intervals="None: deterministic individual traces, no invented uncertainty or significance tests.",
                  source_mapping="energy J -> Wh /3600 for history; planning Wh /1000 -> kWh; MASS is total train mass in tonnes.",
                  speed_matching="Simu receives the unchanged original velocity/time samples.",
                  distance_alignment="Original progress is mapped onto common station boundaries for display only. Backward distance jitter up to 0.01 m per sample is clamped cumulatively and counted in section statistics; larger reversals fail.",
                  exclusions="Trip23 excluded by user due to DP infeasibility; replaced by trip53. No selected history samples removed.",
                  physics_sha256=physics_hash, legacy_physics_sha256=legacy_hash,
                  training="No training or weight updates; residual inference uses existing checkpoints.",
                  menu_reconstruction="Try current and legacy physical features; accept a section total within 0.1 Wh of archived DP. Reconcile only this rounding difference at final sample; otherwise show known endpoints only.",
                  version_identification="Version labels denote the closest numerical reconstruction, not a claim of original source provenance. All versions matching within 0.1 Wh are recorded separately.",
                  interpretation="Historical measurement and residual-corrected code are not asserted to share Simu's mechanical-energy boundary. Differences are descriptive, not a fitted conversion efficiency.",
                  original_vs_vectorized_max_step_error_wh=numerical_errors)
        (state_dir / "qa_notes.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    all_stats = pd.read_csv(stats_path)
    summaries = []
    for trip in trips:
        subset = all_stats[all_stats.trip_no == trip]
        if len(subset) != 104:
            raise ValueError(f"Expected 104 section/scenario records for trip{trip}, found {len(subset)}")
        summaries.extend(render_trip(args, trip, subset, intermediate))
    summary = pd.DataFrame(summaries)
    if not args.export_data:
        print(f"Finished {len(trips)} trips / {len(summaries)} PNG figures: {args.output_dir}", flush=True)
        return
    summary.to_csv(args.output_dir / "trip_comparison_summary.csv", index=False, encoding="utf-8-sig")
    summary.rename(columns={"trip_no": "趟次", "scenario": "场景编号", "scenario_name": "对比场景", "sections": "区间数",
                            "code_kwh": "代码或历史能耗(kWh)", "simu_kwh": "Simu机械能耗(kWh)",
                            "difference_kwh": "Simu减参考能耗(kWh)", "difference_pct": "相对差异(%)",
                            "abs_difference_kwh": "绝对差值(kWh)", "abs_difference_pct": "绝对差异百分比(%)",
                            "max_mass_t": "最大区间列车质量(t)", "history_runtime_s": "历史运行时间(s)",
                            "history_dwell_s": "历史停站时间(s)", "unresolved_sections": "未复现区间数",
                            "code_versions": "数值复现版本", "figure": "图片路径"}).to_csv(
        args.output_dir / "逐趟能耗对比汇总.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    averages = summary.groupby(["scenario", "scenario_name", "code_versions"], sort=False).agg(
        trip_count=("trip_no", "size"), mean_code_kwh=("code_kwh", "mean"), mean_simu_kwh=("simu_kwh", "mean"),
        mean_difference_kwh=("difference_kwh", "mean"), mean_difference_pct=("difference_pct", "mean"),
        mean_abs_difference_kwh=("abs_difference_kwh", "mean"), mean_abs_difference_pct=("abs_difference_pct", "mean"))
    averages.to_csv(args.output_dir / "comparison_average_summary.csv", encoding="utf-8-sig")
    overall = summary.groupby(["scenario", "scenario_name"], sort=False).agg(
        trip_count=("trip_no", "size"), code_mean_kwh=("code_kwh", "mean"), simu_mean_kwh=("simu_kwh", "mean"),
        mean_difference_kwh=("difference_kwh", "mean"), mean_difference_pct=("difference_pct", "mean"),
        mean_abs_difference_kwh=("abs_difference_kwh", "mean"), mean_abs_difference_pct=("abs_difference_pct", "mean"),
        unresolved_trips=("unresolved_sections", lambda x: int((x > 0).sum())))
    overall.rename_axis(index={"scenario": "场景编号", "scenario_name": "对比场景"}).rename(columns={
        "trip_count": "趟次数", "code_mean_kwh": "代码或历史平均能耗(kWh)", "simu_mean_kwh": "Simu平均机械能耗(kWh)",
        "mean_difference_kwh": "能耗差平均值(kWh)", "mean_difference_pct": "能耗差百分比平均值(%)",
        "mean_abs_difference_kwh": "能耗差绝对值平均值(kWh)", "mean_abs_difference_pct": "能耗差百分比绝对值平均值(%)",
        "unresolved_trips": "存在未逐点复现区间的趟次数"}).to_csv(
        args.output_dir / "能耗差异平均统计.csv", encoding="utf-8-sig", float_format="%.5f")
    saving_rows = []
    for trip in trips:
        rows = summary[summary.trip_no == trip].set_index("scenario")
        for scenario in list(SCENARIOS)[1:]:
            row = dict(trip_no=trip, scenario=scenario)
            for source in ("code", "simu"):
                history = float(rows.loc["history", f"{source}_kwh"])
                saving = history - float(rows.loc[scenario, f"{source}_kwh"])
                row[f"{source}_saving_kwh"] = saving
                row[f"{source}_saving_pct"] = percentage(saving, history)
            saving_rows.append(row)
    pd.DataFrame(saving_rows).to_csv(args.output_dir / "energy_saving_summary.csv", index=False, encoding="utf-8-sig")
    unresolved = all_stats[all_stats.trip_no.isin(trips) & (all_stats.code_version == "unresolved")]
    unresolved.to_csv(args.output_dir / "unresolved_menu_sections.csv", index=False, encoding="utf-8-sig")
    manifest = pd.read_csv(state_dir / "input_manifest.csv")
    for _, row in manifest.iterrows():
        if fingerprint(row.plan_path) != row.plan_sha256:
            raise ValueError(f"Input plan changed during replay: {row.plan_path}")
    for _, row in pd.read_csv(state_dir / "model_source_manifest.csv").iterrows():
        for name in ("best_res_model.pth", "scaler_x.pkl", "scaler_y.pkl"):
            if fingerprint(menu_module.RES_MODEL_BASE / row.section / name) != row[name]:
                raise ValueError(f"Residual asset changed during replay: {row.section} {name}")
    (args.output_dir / "README.md").write_text(
        "# Simu 与代码能耗对比\n\n"
        f"本批次 {len(trips)} 趟，每趟 4 组图（历史、标准 DP、Energy First、停站放宽 5%）。"
        "第23趟不纳入，替换为第53趟。没有运行 OpenTrack，没有重训或改写模型权重，也没有改写 DP 表。\n\n"
        "## 图与汇总\n"
        "- trip***：每种场景的 PNG、PDF、SVG，逐点数据及26区间汇总。\n"
        "- 逐趟能耗对比汇总.csv：每趟每场景的两方总能耗、差值、百分比。\n"
        "- 能耗差异平均统计.csv：每类场景的趟次等权平均，不是合计能耗的比值。\n"
        "- comparison_average_summary.csv：另按数值复现版本分组，避免混淆。\n"
        "- energy_saving_summary.csv：分别相对各自历史基线计算的规划节能量。\n"
        "- unresolved_menu_sections.csv：不能精确逐点复现的旧菜单区间。\n\n"
        "## 口径\n"
        "差值 = Simu - 代码（历史场景改为实测）；百分比 = 差值 / 参考能耗 × 100%。"
        "Simu 是正向牵引机械能，不含效率、辅助用电和再生回收。历史能耗保留实测口径，"
        "代码规划能耗保留原有残差修正，不能把二者与 Simu 的差异直接解释为转换率。\n\n"
        "代码逐点曲线通过现有权重与候选物理实现做数值复现，没有按比例摊分残差。"
        "重现总量与原 DP 区间值相差不超过0.1 Wh才接受，并保留匹配的候选版本和原始差值。"
        "版本标签表示数值兼容性，不代替原生成日志的版本证据。无法复现的区间内部留空，"
        "仅标出已知站界累计能耗，不虚构连续曲线。\n\n"
        "速度是模型输入，两线重合不表示独立速度预测准确。横轴按统一站界映射，仅影响显示；"
        "物理回放仍使用原始速度和时间。质量使用各区间列车总质量，不是单独乘客质量。\n\n"
        "## 重画\n"
        "在项目根目录运行 `python scripts_new/07_visualization/plot_simu_code_trip_compare.py --render-only`"
        " 可从已保存的逐点结果重画，无须重新计算物理或神经网络。\n",
        encoding="utf-8")
    print(f"Finished {len(trips)} trips / {len(summaries)} figures: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
