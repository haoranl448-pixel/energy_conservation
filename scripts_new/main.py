# -*- coding: utf-8 -*-
"""
节能排图项目的主程序入口。

这个文件只负责“流程调度”，不重写任何业务算法。
真正的数据清洗、ATO 曲线生成、能耗计算、DP 排图等逻辑，仍然放在原来的
`scripts/` 脚本里；这里做的事情是把这些脚本按模块顺序串起来，一键执行。
"""

# 开启较新的类型注解写法，例如 list[str]、tuple[PipelineStep, ...]。
from __future__ import annotations

# argparse：解析命令行参数，例如 --dry-run、--only。
import argparse
# dataclasses：用来定义轻量级的数据结构 PipelineStep。
import dataclasses
# os：复制当前环境变量，并把选择的趟号传给子脚本。
import os
# subprocess：用来从主程序里调用子脚本。
import subprocess
# sys：读取当前 Python 解释器路径，默认用它来运行子脚本。
import sys
# time：统计每个步骤的耗时。
import time
# datetime：生成日志目录时间戳、记录开始/结束时间。
from datetime import datetime
# Path：用更稳妥的方式处理 Windows/Linux 路径。
from pathlib import Path
# Iterable：给函数参数做类型标注，表示可迭代对象。
from typing import Iterable


# 默认处理第几趟车。
# 你平时不想每次输入参数时，可以直接改这里，例如 DEFAULT_TRIP_NO = 6。
DEFAULT_TRIP_NO = 1
# 默认 DP 目标总时间，单位秒。
# None 表示使用 globall_v2.py 里的默认值；也可以改成 692.65 这种数字。
DEFAULT_TARGET_TIME = None
# 默认 results 数据目录，供残差模型训练、能耗菜单、历史回放和 DP 绘图使用。
# None 表示各脚本使用自己的默认 results 目录；也可以改成一个目录路径字符串。
DEFAULT_RESULTS_DATA_DIR = None
# 默认 ATO 数据目录，供 ATO 模板训练和 ATO 曲线生成使用。
# None 表示复用 results 数据目录；如果 results 也为空，则 ATO 脚本使用自己的默认目录。
DEFAULT_ATO_DATA_DIR = None
DEFAULT_LINE_SCOPE = "full"
TRACEABILITY_FILENAME = "trip_traceability_manifest_v1.csv"


def configure_stdio() -> None:
    """让 Windows 终端遇到特殊字符时替换输出，避免流程被编码问题打断。"""

    # PowerShell/cmd 在中文 Windows 下常用 GBK 编码，遇到个别无法表示的字符会抛异常。
    # 这里把 stdout/stderr 的错误策略改成 replace，让业务脚本继续运行。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass


@dataclasses.dataclass(frozen=True)
class PipelineStep:
    """描述流水线中的一个步骤。"""

    # 步骤 ID：命令行里用这个名字选择步骤，例如 --only energy_menu。
    id: str
    # 展示标题：运行时打印给用户看。
    title: str
    # 原脚本路径：相对于项目根目录，例如 scripts/data_process.py。
    script: str
    # 步骤说明：告诉用户这一步在业务流程里做什么。
    description: str


# 主流程定义。
# 这里的顺序就是完整流程的执行顺序：
# 数据清洗 -> 残差模型训练 -> 等级表 -> 模板提取 -> 曲线生成 -> 能耗菜单 -> 历史基准 -> DP 排图。
# 注意：ato_class_globall_v2.py 里已经会输出最终方案表和对比图，
# 因此默认主流程到 DP 排图就结束，不再接 OpenTrack 导出或旧版 validation。
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    # 第 1 步：清洗原始运行数据。
    PipelineStep(
        id="data_process",
        title="Data cleaning",
        script="scripts_new/00_main_pipeline/01_data_process.py",
        description="Clean raw operation data and generate section-level processed files.",
    ),
    # 第 2 步：训练物理模型之外的残差修正模型。
    PipelineStep(
        id="residual_training",
        title="Residual model training",
        script="scripts_new/00_main_pipeline/02_train_residual_new.py",
        description="Train per-section residual energy models used by energy menu calculation.",
    ),
    # 第 3 步：构建运行等级对照表。
    PipelineStep(
        id="class_lookup",
        title="Class lookup table",
        script="scripts_new/00_main_pipeline/03_build_class_lookup_tables.py",
        description="Build service/class lookup tables used by later ATO steps.",
    ),
    # 第 4 步：从历史数据里提取 ATO 分等级相位模板。
    PipelineStep(
        id="ato_template",
        title="ATO phase template extraction",
        script="scripts_new/00_main_pipeline/04_train_ATO_v8.py",
        description="Extract Class1-Class5 phase templates from historical runs.",
    ),
    # 第 5 步：基于模板生成各区间各等级速度曲线。
    PipelineStep(
        id="ato_simulation",
        title="ATO curve generation",
        script="scripts_new/00_main_pipeline/05_simulate_ATO_v8.py",
        description="Generate feasible speed curves for each section and ATO class.",
    ),
    # 第 6 步：对生成曲线计算能耗菜单。
    PipelineStep(
        id="energy_menu",
        title="Energy menu calculation",
        script="scripts_new/00_main_pipeline/06_ato_generated_results_energy.py",
        description="Calculate physical + residual-AI energy for generated curves.",
    ),
    # 第 7 步：生成历史基准结果，供 DP 最终表对比历史用时和历史能耗。
    PipelineStep(
        id="historical_baseline",
        title="Historical baseline calculation",
        script="scripts_new/00_main_pipeline/07_full_line_validation_results.py",
        description="Generate historical time/energy baseline used by the DP comparison report.",
    ),
    # 第 8 步：用动态规划选择全局最优运行等级组合。
    PipelineStep(
        id="dp_schedule",
        title="DP schedule optimization",
        script="scripts_new/00_main_pipeline/08_ato_class_globall_v2.py",
        description="Select the lowest-energy class combination and write final comparison outputs.",
    ),
)


def find_project_root(start: Path) -> Path:
    """从当前文件位置开始向上查找项目根目录。"""

    # 依次检查当前路径和它的所有父目录。
    for candidate in (start, *start.parents):
        # 项目根目录的判定条件：
        # 1. 有 PROJECT.md 项目说明文件；
        # 2. 有 scripts/ 原脚本目录。
        if (candidate / "PROJECT.md").exists() and (candidate / "scripts").is_dir():
            # 找到后立即返回，后续所有相对路径都基于这个根目录。
            return candidate
    # 如果一直找不到，说明运行位置不在当前项目下面，直接报错。
    raise FileNotFoundError("Could not locate project root from current file path.")


def split_csv(value: str | None) -> list[str]:
    """把逗号分隔的命令行参数转成列表。"""

    # 参数为空时返回空列表，方便后面统一处理。
    if not value:
        return []
    # 按逗号切开，去掉每一项前后的空格，并过滤空字符串。
    return [item.strip() for item in value.split(",") if item.strip()]


def get_step_ids() -> list[str]:
    """返回所有合法的步骤 ID。"""

    # 从 PIPELINE_STEPS 里提取 id，保证命令行校验和流程定义保持一致。
    return [step.id for step in PIPELINE_STEPS]


def validate_step_ids(ids: Iterable[str], label: str) -> None:
    """检查用户传入的步骤 ID 是否都存在。"""

    # 合法步骤集合。
    valid = set(get_step_ids())
    # 找出用户传入但不存在的步骤。
    unknown = [item for item in ids if item not in valid]
    # 只要存在未知步骤，就抛出错误并提示所有可选步骤。
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}. Valid steps: {', '.join(get_step_ids())}")


def select_steps(args: argparse.Namespace) -> list[PipelineStep]:
    """根据命令行参数决定本次要运行哪些步骤。"""

    # --only energy_menu,dp_schedule 这种参数会先被拆成列表。
    only_ids = split_csv(args.only)
    # --skip class_lookup 这种参数也拆成列表。
    skip_ids = split_csv(args.skip)
    # 校验 --only 里写的步骤是否存在。
    validate_step_ids(only_ids, "--only")
    # 校验 --skip 里写的步骤是否存在。
    validate_step_ids(skip_ids, "--skip")

    # 默认先拿完整流程。
    steps = list(PIPELINE_STEPS)
    # 如果用户指定了 --only，就只保留这些步骤。
    if only_ids:
        # 转成集合，提高查找效率。
        wanted = set(only_ids)
        # 保持原流水线顺序，只筛选用户点名的步骤。
        steps = [step for step in steps if step.id in wanted]
    else:
        # 没有 --only 时，允许通过 --from-step 和 --to-step 截取一段流程。
        step_ids = get_step_ids()
        # 起点：用户指定则从指定步骤开始，否则从第一个步骤开始。
        start_idx = step_ids.index(args.from_step) if args.from_step else 0
        # 终点：用户指定则到指定步骤结束，否则跑到最后一步。
        end_idx = step_ids.index(args.to_step) if args.to_step else len(step_ids) - 1
        # 防止用户把起点写在终点后面。
        if start_idx > end_idx:
            raise ValueError("--from-step must not appear after --to-step in the pipeline order.")
        # 根据起止索引截取要执行的步骤。
        steps = steps[start_idx : end_idx + 1]

    # 如果用户指定了 --skip，就把这些步骤从最终列表里剔除。
    if skip_ids:
        # 转成集合，方便判断。
        skip = set(skip_ids)
        # 保留不在 skip 集合中的步骤。
        steps = [step for step in steps if step.id not in skip]
    # 返回最终要运行的步骤列表。
    return steps


def print_steps(steps: Iterable[PipelineStep]) -> None:
    """把步骤列表打印出来，便于 dry-run 或 --list 查看。"""

    # enumerate(..., start=1) 让显示序号从 1 开始。
    for index, step in enumerate(steps, start=1):
        # 打印步骤序号和步骤 ID。
        print(f"{index:02d}. {step.id}")
        # 打印步骤标题。
        print(f"    title: {step.title}")
        # 打印真正会调用的脚本路径。
        print(f"    script: {step.script}")
        # 打印业务说明。
        print(f"    desc:  {step.description}")


def run_step(
    step: PipelineStep,
    project_root: Path,
    python_exe: str,
    log_dir: Path,
    dry_run: bool,
    trip_no: int,
    target_time: float | None,
    results_data_dir: Path | None,
    ato_data_dir: Path | None,
    line_scope: str,
    traceability_manifest: Path | None,
) -> int:
    """执行单个流水线步骤。"""

    # 把相对脚本路径拼成绝对路径。
    script_path = project_root / step.script
    # 子进程命令：Python 解释器 + 目标脚本。
    command = [python_exe, str(script_path)]

    # 打印当前步骤标题，让终端输出更清晰。
    print(f"\n=== {step.id}: {step.title} ===")
    # 打印当前步骤的业务说明。
    print(step.description)
    # 打印实际命令；带空格的路径会加引号，方便复制排查。
    print("Command:", " ".join(f'"{part}"' if " " in part else part for part in command))
    # 趟号以环境变量传给子脚本；子脚本用它决定读写 trip1/trip6 等文件。
    print(f"Trip:    {trip_no} (global 1-based trip; legacy local index {trip_no - 1})")
    # 目标总时间只在 DP 排图步骤使用；不传时沿用 globall_v2.py 里的默认值。
    if target_time is None:
        print("Target:  script default")
    else:
        print(f"Target:  {target_time:.2f}s")
    # results 数据目录供残差/能耗/历史/DP 使用。
    print(f"Results: {results_data_dir if results_data_dir is not None else 'script default'}")
    # ATO 数据目录供 ATO 模板训练/曲线生成使用；未单独指定时复用 results 数据目录。
    print(f"ATO:     {ato_data_dir if ato_data_dir is not None else 'script default'}")
    print(f"Scope:   {line_scope}")
    print(f"Trace:   {traceability_manifest if traceability_manifest is not None else 'not found (legacy local index)'}")

    # 如果脚本不存在，直接返回 127，表示命令/文件不存在。
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}")
        return 127

    # dry-run 模式只打印命令，不真正执行脚本。
    if dry_run:
        return 0

    # 子进程继承当前环境，并额外得到本次选择的趟号。
    child_env = os.environ.copy()
    child_env["ENERGY_TRIP_NO"] = str(trip_no)
    child_env["ENERGY_TRIP_INDEX"] = str(trip_no - 1)
    if target_time is not None:
        child_env["ENERGY_TARGET_TIME"] = str(target_time)
    if results_data_dir is not None:
        child_env["ENERGY_RESULTS_DATA_DIR"] = str(results_data_dir)
        # 兼容旧脚本里还在读取 ENERGY_DATA_DIR 的 results 口径步骤。
        if step.id not in {"ato_template", "ato_simulation"}:
            child_env["ENERGY_DATA_DIR"] = str(results_data_dir)
    if ato_data_dir is not None and step.id in {"ato_template", "ato_simulation"}:
        child_env["ENERGY_ATO_DATA_DIR"] = str(ato_data_dir)
    child_env["ENERGY_LINE_SCOPE"] = line_scope
    if traceability_manifest is not None:
        child_env["ENERGY_TRACEABILITY_MANIFEST"] = str(traceability_manifest)
    # 子脚本里有 emoji/特殊符号输出；强制 UTF-8 容错，避免 Windows GBK 控制台报错。
    child_env["PYTHONIOENCODING"] = "utf-8:replace"
    child_env["PYTHONUTF8"] = "1"

    # 创建日志目录，例如 output/pipeline_logs/20260525_133110。
    log_dir.mkdir(parents=True, exist_ok=True)
    # 每个步骤单独一个日志文件。
    log_file = log_dir / f"{step.id}.log"
    # 记录开始时间，用来计算耗时。
    start = time.time()

    # 打开日志文件，边运行边写日志。
    with log_file.open("w", encoding="utf-8") as log:
        # 写入日志头，方便之后追溯。
        log.write(f"# step: {step.id}\n")
        log.write(f"# command: {' '.join(command)}\n")
        log.write(f"# ENERGY_TRIP_NO: {trip_no}\n")
        log.write(f"# ENERGY_TRIP_INDEX: {trip_no - 1}\n")
        log.write(f"# ENERGY_TARGET_TIME: {target_time if target_time is not None else 'script default'}\n")
        log.write(f"# ENERGY_RESULTS_DATA_DIR: {results_data_dir if results_data_dir is not None else 'script default'}\n")
        log.write(f"# ENERGY_ATO_DATA_DIR: {ato_data_dir if ato_data_dir is not None else 'script default'}\n")
        log.write(f"# ENERGY_LINE_SCOPE: {line_scope}\n")
        log.write(f"# ENERGY_TRACEABILITY_MANIFEST: {traceability_manifest if traceability_manifest is not None else 'not found'}\n")
        log.write("# PYTHONIOENCODING: utf-8:replace\n")
        log.write(f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
        # 启动子脚本。
        process = subprocess.Popen(
            # command 是 [python, script]，避免手写 shell 字符串带来的转义问题。
            command,
            # cwd 设为项目根目录，保证原 scripts/ 里的相对路径尽量保持原行为。
            cwd=project_root,
            # env 里带上 ENERGY_TRIP_NO，供 full_line/globall 等脚本读取。
            env=child_env,
            # 捕获标准输出。
            stdout=subprocess.PIPE,
            # 把错误输出合并到标准输出，日志里能看到完整信息。
            stderr=subprocess.STDOUT,
            # 以文本模式读取输出。
            text=True,
            # 用 UTF-8 读取，遇到坏字符用替换策略，避免日志中断。
            encoding="utf-8",
            errors="replace",
        )

        # 告诉类型检查器 stdout 不为空，因为上面设置了 stdout=subprocess.PIPE。
        assert process.stdout is not None
        # 实时读取子脚本输出。
        for line in process.stdout:
            # 一边打印到当前终端。
            print(line, end="")
            # 一边写入日志文件。
            log.write(line)

        # 等待子进程结束，并获取退出码。
        return_code = process.wait()
        # 计算该步骤耗时。
        elapsed = time.time() - start
        # 写入日志尾。
        log.write(f"\n# finished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"# return_code: {return_code}\n")
        log.write(f"# elapsed_seconds: {elapsed:.1f}\n")

    # 在终端提示该步骤结束，并告诉用户日志在哪里。
    print(f"Step finished with code {return_code}. Log: {log_file}")
    # 返回子脚本退出码，主流程用它判断是否失败。
    return return_code


def build_parser() -> argparse.ArgumentParser:
    """定义主程序支持的命令行参数。"""

    # 创建参数解析器，并让 --help 输出包含默认值。
    parser = argparse.ArgumentParser(
        description="Run the full energy-conservation workflow from one main entrypoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --list：只列出步骤，不执行。
    parser.add_argument("--list", action="store_true", help="List pipeline steps and exit.")
    # --dry-run：打印命令但不执行，适合开会演示或检查流程。
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing scripts.")
    # --trip-no：选择第几趟车；例如 --trip-no 6 会处理第 6 趟，对应 segment index=5。
    parser.add_argument("--trip-no", type=int, default=DEFAULT_TRIP_NO, help="Trip number to process, using 1-based numbering.")
    # --ask-trip：运行时询问用户第几趟，适合不想记命令参数时使用。
    parser.add_argument("--ask-trip", action="store_true", help="Prompt for the trip number before running.")
    # --target-time：手动指定 DP 目标总时间，单位秒。
    parser.add_argument("--target-time", type=float, default=DEFAULT_TARGET_TIME, help="DP target total time in seconds. Defaults to the globall_v2 script value.")
    # --ask-target-time：运行时询问 DP 目标总时间。
    parser.add_argument("--ask-target-time", action="store_true", help="Prompt for the DP target total time before running.")
    # --data-dir/--results-data-dir：指定 results_*.xlsx 数据目录，供残差/能耗/历史/DP 使用。
    parser.add_argument(
        "--data-dir",
        "--results-data-dir",
        dest="results_data_dir",
        default=DEFAULT_RESULTS_DATA_DIR,
        help="Directory containing results_*.xlsx for residual training, energy calculation, baseline, and DP plotting.",
    )
    # --ato-data-dir：单独指定 ATO 数据目录；不传时默认复用 --data-dir。
    parser.add_argument(
        "--ato-data-dir",
        default=DEFAULT_ATO_DATA_DIR,
        help="Optional ATO data directory. Supports results_*.xlsx with 曲线质量标签 or cleaned_*.xlsx; defaults to --data-dir.",
    )
    parser.add_argument(
        "--line-scope",
        default=DEFAULT_LINE_SCOPE,
        help="Station range used by DP schedule optimization: full/all or a positive section count, e.g. 5.",
    )
    parser.add_argument(
        "--traceability-manifest",
        default=None,
        help=(
            "CSV mapping each global trip to the matching local segment in every section. "
            "If omitted, main searches inside --data-dir and its sibling *_traceability directory."
        ),
    )
    # --ask-data-dir：运行时询问 results 数据目录。
    parser.add_argument("--ask-data-dir", "--ask-results-data-dir", dest="ask_results_data_dir", action="store_true", help="Prompt for the results_*.xlsx data directory before running.")
    # --ask-ato-data-dir：运行时询问 ATO 数据目录。
    parser.add_argument("--ask-ato-data-dir", action="store_true", help="Prompt for the optional ATO data directory before running.")
    # --from-step：从某一步开始，例如 --from-step energy_menu。
    parser.add_argument("--from-step", choices=get_step_ids(), help="Start from this step.")
    # --to-step：到某一步结束，例如 --to-step dp_schedule。
    parser.add_argument("--to-step", choices=get_step_ids(), help="Stop after this step.")
    # --only：只跑指定步骤，多个步骤用英文逗号隔开。
    parser.add_argument("--only", help="Comma-separated step ids to run in pipeline order.")
    # --skip：跳过指定步骤，多个步骤用英文逗号隔开。
    parser.add_argument("--skip", help="Comma-separated step ids to skip.")
    # --continue-on-error：某一步失败后仍继续跑后面的步骤。
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if a step fails.")
    # --python：指定运行子脚本的 Python 解释器。
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run child scripts.")
    # --log-dir：指定日志目录；不传则自动生成时间戳目录。
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-step logs. Defaults to output/pipeline_logs/<timestamp>.",
    )
    # 返回配置好的参数解析器。
    return parser


def resolve_trip_no(args: argparse.Namespace) -> int:
    """得到本次要处理的趟号。"""

    # 默认使用 --trip-no 或 DEFAULT_TRIP_NO。
    trip_no = args.trip_no
    # 如果用户加了 --ask-trip，就在终端里交互询问。
    if args.ask_trip:
        raw = input(f"请输入要处理第几趟车，直接回车使用默认第 {trip_no} 趟：").strip()
        if raw:
            trip_no = int(raw)

    # 趟号使用给人看的 1-based 编号，至少为 1。
    if trip_no < 1:
        raise ValueError("--trip-no must be >= 1.")
    return trip_no


def resolve_target_time(args: argparse.Namespace) -> float | None:
    """得到本次 DP 排图使用的目标总时间。"""

    # 默认使用 --target-time 或 DEFAULT_TARGET_TIME。
    target_time = args.target_time
    # 如果用户加了 --ask-target-time，就在终端里交互询问。
    if args.ask_target_time:
        default_text = "globall_v2.py 默认值" if target_time is None else f"{target_time:.2f}s"
        raw = input(f"请输入 DP 目标总时间，单位秒，直接回车使用 {default_text}：").strip()
        if raw:
            target_time = float(raw)

    # None 表示不覆盖 globall_v2.py 中写死的默认值。
    if target_time is None:
        return None
    if target_time <= 0:
        raise ValueError("--target-time must be > 0.")
    return target_time


def resolve_optional_dir(value: str | None, project_root: Path, label: str) -> Path | None:
    """把可选目录参数解析成绝对路径。"""

    # None 或空字符串都表示不覆盖脚本默认目录。
    if not value:
        return None

    data_dir = Path(value)
    # 相对路径按项目根目录解析，方便写 data/data_processed 这种形式。
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"{label} not found: {data_dir}")
    return data_dir


def resolve_results_data_dir(args: argparse.Namespace, project_root: Path) -> Path | None:
    """得到 results_*.xlsx 数据目录。"""

    data_dir_value = args.results_data_dir
    if args.ask_results_data_dir:
        default_text = "脚本默认 results 目录" if not data_dir_value else str(data_dir_value)
        raw = input(f"请输入 results_*.xlsx 数据目录，直接回车使用 {default_text}：").strip()
        if raw:
            data_dir_value = raw
    return resolve_optional_dir(data_dir_value, project_root, "Results data directory")


def resolve_ato_data_dir(args: argparse.Namespace, project_root: Path) -> Path | None:
    """得到可选 ATO 数据目录；未传时由 main 复用 results 数据目录。"""

    data_dir_value = args.ato_data_dir
    if args.ask_ato_data_dir:
        default_text = "复用 results 数据目录" if not data_dir_value else str(data_dir_value)
        raw = input(f"请输入 ATO 数据目录，直接回车使用 {default_text}：").strip()
        if raw:
            data_dir_value = raw
    return resolve_optional_dir(data_dir_value, project_root, "ATO data directory")


def resolve_traceability_manifest(
    args: argparse.Namespace,
    project_root: Path,
    results_data_dir: Path | None,
) -> Path | None:
    """Resolve an explicit manifest or auto-detect the one paired with --data-dir."""

    if args.traceability_manifest:
        path = Path(args.traceability_manifest)
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Traceability manifest not found: {path}")
        return path

    if results_data_dir is None:
        return None

    candidates = [results_data_dir / TRACEABILITY_FILENAME]
    if not results_data_dir.name.endswith("_traceability"):
        candidates.append(
            results_data_dir.with_name(f"{results_data_dir.name}_traceability") / TRACEABILITY_FILENAME
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_line_scope(args: argparse.Namespace) -> str:
    """得到 DP 排图使用的区间范围；full 表示全正向区间，数字表示前 N 个区间。"""

    raw = str(args.line_scope).strip().lower()
    if raw in {"full", "all"}:
        return "full"
    if raw == "first5":
        return "5"

    try:
        section_count = int(raw)
    except ValueError as exc:
        raise ValueError("--line-scope must be 'full' or a positive integer, for example --line-scope 5.") from exc
    if section_count < 1:
        raise ValueError("--line-scope section count must be >= 1.")
    return str(section_count)


def main() -> int:
    """主函数：解析参数、选择步骤、按顺序执行。"""

    # 先配置输出流，避免后续打印子脚本日志时被 Windows 编码问题中断。
    configure_stdio()

    # 构造命令行解析器。
    parser = build_parser()
    # 读取用户传入的命令行参数。
    args = parser.parse_args()
    # 自动定位项目根目录，避免必须从固定目录运行。
    project_root = find_project_root(Path(__file__).resolve())
    # 得到本次要处理的趟号。
    trip_no = resolve_trip_no(args)
    # 得到本次 DP 排图的目标总时间；None 表示使用 globall_v2.py 默认值。
    target_time = resolve_target_time(args)
    # 得到 results 数据目录；None 表示残差/能耗/历史/DP 使用脚本默认目录。
    results_data_dir = resolve_results_data_dir(args, project_root)
    # 得到 ATO 数据目录；未单独指定时复用 results 目录，ATO 脚本内部会只取曲线质量标签=0。
    ato_data_dir = resolve_ato_data_dir(args, project_root)
    effective_ato_data_dir = ato_data_dir if ato_data_dir is not None else results_data_dir
    line_scope = resolve_line_scope(args)
    traceability_manifest = resolve_traceability_manifest(args, project_root, results_data_dir)
    # 根据 --only/--skip/--from-step/--to-step 等参数选出要跑的步骤。
    selected_steps = select_steps(args)

    # 如果用户只想看步骤列表，打印后直接退出。
    if args.list:
        print_steps(selected_steps)
        return 0

    # 如果筛选后没有任何步骤，提示并正常退出。
    if not selected_steps:
        print("No steps selected.")
        return 0

    # 生成本次运行的时间戳，用于默认日志目录。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 如果用户传了 --log-dir，则使用用户指定目录；否则使用 output/pipeline_logs/时间戳。
    log_dir = Path(args.log_dir) if args.log_dir else project_root / "output" / "pipeline_logs" / timestamp
    # 如果用户传的是相对路径，就基于项目根目录转换成绝对路径。
    if not log_dir.is_absolute():
        log_dir = project_root / log_dir

    # 打印本次运行的基础信息。
    print(f"Project root: {project_root}")
    print(f"Python:       {args.python}")
    print(f"Log dir:      {log_dir}")
    print(f"Dry run:      {args.dry_run}")
    print(f"Trip no:      {trip_no} (global trip; legacy local index {trip_no - 1})")
    print(f"Target time:  {f'{target_time:.2f}s' if target_time is not None else 'script default'}")
    print(f"Results dir:  {results_data_dir if results_data_dir is not None else 'script default'}")
    print(f"ATO dir:      {effective_ato_data_dir if effective_ato_data_dir is not None else 'script default'}")
    print(f"Line scope:   {line_scope}")
    print(f"Traceability: {traceability_manifest if traceability_manifest is not None else 'not found; legacy local-index mode'}")
    # 打印最终选中的步骤列表。
    print("\nSelected steps:")
    print_steps(selected_steps)

    # 保存失败步骤，最后统一汇总。
    failures: list[tuple[str, int]] = []
    # 按顺序执行每个步骤。
    for step in selected_steps:
        # 执行单个步骤，并拿到退出码。
        code = run_step(
            step=step,
            project_root=project_root,
            python_exe=args.python,
            log_dir=log_dir,
            dry_run=args.dry_run,
            trip_no=trip_no,
            target_time=target_time,
            results_data_dir=results_data_dir,
            ato_data_dir=effective_ato_data_dir,
            line_scope=line_scope,
            traceability_manifest=traceability_manifest,
        )
        # 非 0 退出码表示失败。
        if code != 0:
            # 记录失败步骤和退出码。
            failures.append((step.id, code))
            # 默认失败就停止；只有传 --continue-on-error 才继续。
            if not args.continue_on_error:
                break

    # 如果有失败步骤，打印汇总并返回第一个失败码。
    if failures:
        print("\nPipeline finished with failures:")
        for step_id, code in failures:
            print(f"- {step_id}: exit code {code}")
        return failures[0][1]

    # 所有步骤成功或 dry-run 成功，打印成功信息。
    print("\nPipeline finished successfully.")
    # 返回 0，表示主程序成功。
    return 0


# 只有直接运行本文件时才进入主流程。
# 如果将来被其他 Python 文件 import，这里不会自动执行。
if __name__ == "__main__":
    # raise SystemExit(...) 可以把 main() 的返回码传给操作系统。
    raise SystemExit(main())
