# scripts_new 目录说明

`scripts_new/` 是从原 `scripts/` 复制出来的分类整理版，用来阅读、查找和后续重构。原 `scripts/` 暂时保持不动，避免影响已有运行路径和历史输出。

注意：不少脚本依赖 `__file__` 推导项目根目录。因为现在脚本被放进了二级分类目录，如果直接从 `scripts_new/` 运行，个别脚本可能需要先统一修正路径定位逻辑。

从原 `scripts/` 复制过来的脚本清单见 `_manifest.csv`。

## main.py 使用方法

主程序是 `scripts_new/main.py`，作用是把完整流程拆成多个步骤模块，然后按顺序调用 `scripts_new/00_main_pipeline/` 里的编号脚本。

当前默认主流程到 `ato_class_globall_v2.py` 动态规划排图结束。这个脚本内部已经会输出最终方案表 `Final_Planning_Comparison.csv` 和对比图 `Optimized_Full_Line_Report.png`，所以默认流程不再继续调用 OpenTrack 导出或旧版 validation 脚本。

`ato_class_globall_v2.py` 运行前必须已经有这些输入：

- `output/analysis/ato_class_energy_menu<趟号>_new_v3.csv`：由 `scripts/ato_generated_results_energy.py` 生成。
- `full_line<趟号>_validation_results.csv`：由 `scripts/full_line_validation_results.py` 生成；最终规划表现在用 `历史实测能耗(Wh)` 作为节能量/节能率对比基准，`历史能耗(Wh)` 保留为历史曲线模型回放能耗用于诊断。
- `data/static/section_params_trip<趟号>.csv`：区间参数和载重表。
- `output/ato_generated_results_new_v4/<区间>/<class>_generated_curve.csv`：用于最终拼接优化速度曲线。
- `data/data_processed/results_<区间>.xlsx`：用于最终拼接历史速度曲线。

当前主流程的 ATO 曲线目录已统一为 `output/ato_generated_results_new_v4`：`scripts/simulate_ATO_v8.py` 写入这里，`scripts/ato_generated_results_energy.py` 和 `scripts/ato_class_globall_v2.py` 也从这里读取。

默认完整流程：

```powershell
python scripts_new/main.py
```

默认处理第 1 趟车；如果想固定改成别的趟，可以直接改 `scripts_new/main.py` 顶部的 `DEFAULT_TRIP_NO`。
默认 DP 目标总时间使用 `scripts_new/00_main_pipeline/08_ato_class_globall_v2.py` 里的 `DEFAULT_T_TOTAL_TARGET`；如果想固定成某个数，可以直接改 `scripts_new/main.py` 顶部的 `DEFAULT_TARGET_TIME`。
默认数据目录使用各脚本自己的配置；测试时 `--data-dir` 指定带 `results_*.xlsx` 的主数据目录。
如果 `results_*.xlsx` 里有 `曲线质量标签`，残差模型训练会使用全部趟次，ATO 模板训练/对比只使用 `曲线质量标签 = 0` 的正常白天趟次。正常曲线里真实存在的等级都会学习模板；缺失等级只作为后续 DP 候选由相邻真实等级生成，不反过来参与学习。
`--ato-data-dir` 只是可选覆盖项；不传时 ATO 默认复用 `--data-dir`。
默认 DP 排图范围是全正向区间；测试时可以通过 `--line-scope n` 只取前 `n` 个区间。


默认的参数：
从第 1 步跑到最后一步
trip-no = 1
target-time = 不传，DP 用 08_ato_class_globall_v2.py 里的默认值
data-dir = 不传，残差/能耗/历史/DP 用自己的默认 results 目录；
        默认是：D:\energy_conservation\data\data_processed
ato-data-dir = 不传，ATO 模板训练/曲线生成默认复用 data-dir；
        如果 data-dir 也不传，ATO 脚本才使用自己的默认目录
line-scope = full，DP 默认跑全正向区间；传数字时跑前 N 个正向区间
dry-run = False，真的执行
遇到报错 = 停止
log-dir = 自动生成 output/pipeline_logs/时间戳
python = 当前这个 python

临时指定趟号：

```powershell
python scripts_new/main.py --trip-no 6
```

运行时询问趟号：

```powershell
python scripts_new/main.py --ask-trip
```

临时指定 DP 目标总时间，单位秒：

```powershell
python scripts_new/main.py --target-time 692.65
```

运行时询问 DP 目标总时间：

```powershell
python scripts_new/main.py --ask-target-time
```

趟号和目标时间可以一起传：

```powershell
python scripts_new/main.py --trip-no 6 --target-time 692.65 --from-step dp_schedule
```

临时指定带曲线质量标签的 results 数据目录：

```powershell
python scripts_new/main.py --data-dir data\data_processed_step2_v3_first5_curve_quality --from-step residual_training
```

单独指定 ATO 数据目录：

```powershell
python scripts_new/main.py --ato-data-dir "D:\energy_conservation\data\data_processed_new_v2" --from-step ato_template
```

两套数据目录一起指定，只有确实想让 ATO 使用另一套数据时才需要：

```powershell
python scripts_new/main.py --data-dir "C:\Users\bit11\Desktop\数据测试代码\data_processed_step2_v3_first5" --ato-data-dir "D:\energy_conservation\data\data_processed_new_v2" --from-step residual_training --line-scope 5
```

指定 DP 区间范围：

```powershell
# 默认全区间，不写 --line-scope 也可以
python scripts_new/main.py --from-step dp_schedule --data-dir data\data_processed_step2_v3_all_with_class

# 只跑前 5 个正向区间
python scripts_new/main.py --from-step dp_schedule --line-scope 5 --data-dir data\data_processed_step2_v3_first5_with_class
```

运行时询问 results 数据目录：

```powershell
python scripts_new/main.py --ask-data-dir --from-step residual_training
```

运行时询问 ATO 数据目录：

```powershell
python scripts_new/main.py --ask-ato-data-dir --from-step ato_template
```

趟号会通过环境变量传给子脚本：

- `scripts/ato_generated_results_energy.py` 会输出 `output/analysis/ato_class_energy_menu<趟号>_new_v3.csv`。
- `scripts/full_line_validation_results.py` 会输出 `full_line<趟号>_validation_results.csv`。
- `scripts/ato_class_globall_v2.py` 会读取同一趟号的能耗菜单、历史基准和 `section_params_trip<趟号>.csv`。
- `scripts/ato_class_globall_v2.py` 会读取主程序传入的目标总时间；未传入时使用脚本默认值。
- `--data-dir` 指定的是 `results_区间.xlsx` 目录，供 `02_train_residual_new.py`、`06_ato_generated_results_energy.py`、`07_full_line_validation_results.py`、`08_ato_class_globall_v2.py` 使用；如果未单独传 `--ato-data-dir`，也会供 `04_train_ATO_v8.py` 和 `05_simulate_ATO_v8.py` 使用。
- 残差模型训练不筛 `曲线质量标签`，使用全部趟次。
- ATO 模板训练/真实曲线对比如果读到 `曲线质量标签`，只保留 `曲线质量标签 = 0`；正常曲线中出现过的等级都作为 `real` 模板学习。
- 缺失等级只在曲线生成阶段用相邻真实等级补候选，并标记 `曲线来源`：`real`、`interpolated` 或 `extrapolated_adjacent`。远距离外推默认不启用。
- 每次重新执行 `ato_template` 会先清空 `output/ato_phase_results_v3`；每次重新执行 `ato_simulation` 会先清空 `output/ato_generated_results_new_v4`，避免旧等级文件混在新结果里。
- DP 排图按三轮尝试：先 `real_only`，不可行再 `allow_interpolated`，仍不可行再 `allow_extrapolated`；同一轮内优先少用外推、少用插值，再比较能耗。
- `--ato-data-dir` 是可选覆盖项，支持 `results_区间.xlsx` 或 `cleaned_区间.xlsx`；仅当你想让 ATO 使用另一套数据时才需要指定。

如果当前终端里 `python` 不在 PATH，可以用本机 Python 绝对路径：

```powershell
& "C:\Users\bit11\AppData\Local\Programs\Python\Python311\python.exe" scripts_new/main.py
```

只预览将要执行哪些步骤，不真正运行：

```powershell
python scripts_new/main.py --dry-run
```

查看步骤列表：

```powershell
python scripts_new/main.py --list
```

从某一步开始运行，例如从生成速度曲线开始：

```powershell
python scripts_new/main.py --from-step ato_simulation
```

只运行某几个步骤，例如只算能耗菜单并做 DP 排图：

```powershell
python scripts_new/main.py --only energy_menu,dp_schedule
```

当前主程序包含这些步骤：

| 步骤 ID | 调用脚本 | 作用 |
| --- | --- | --- |
| `data_process` | `scripts_new/00_main_pipeline/01_data_process.py` | 原始数据清洗 |
| `residual_training` | `scripts_new/00_main_pipeline/02_train_residual_new.py` | 训练残差能耗模型 |
| `class_lookup` | `scripts_new/00_main_pipeline/03_build_class_lookup_tables.py` | 构建等级/标准时间对照表 |
| `ato_template` | `scripts_new/00_main_pipeline/04_train_ATO_v8.py` | 提取 ATO 多等级相位模板 |
| `ato_simulation` | `scripts_new/00_main_pipeline/05_simulate_ATO_v8.py` | 生成各区间各等级速度曲线 |
| `energy_menu` | `scripts_new/00_main_pipeline/06_ato_generated_results_energy.py` | 计算能耗菜单 |
| `historical_baseline` | `scripts_new/00_main_pipeline/07_full_line_validation_results.py` | 生成历史用时/历史模型能耗基准 |
| `dp_schedule` | `scripts_new/00_main_pipeline/08_ato_class_globall_v2.py` | DP 排图优化，并输出最终方案表和对比图 |

日志默认输出到 `output/pipeline_logs/时间戳/`，每个步骤一个 `.log` 文件。

## 00_main_pipeline

用途：当前项目最重要的端到端主线，从数据处理到 ATO 模板、速度曲线、能耗菜单、DP 排图。

- `data_process.py`：原始运行数据清洗，生成按区间整理后的数据文件。
- `build_class_lookup_tables.py`：构建运行等级/服务号/区间等对照表。
- `train_ATO_v8.py`：主线 ATO 相位模板提取脚本，对正常曲线里真实存在的等级学习加速、巡航、制动三段模板，并记录样本数和可靠性。
- `simulate_ATO_v8.py`：主线速度曲线生成脚本，真实等级用自身模板，缺失等级只用相邻真实等级生成候选并保留来源标签。
- `ato_generated_results_energy.py`：读取生成曲线，计算物理模型 + AI 残差后的能耗菜单。
- `ato_class_globall_v2.py`：主线 DP 排图优化，支持运行等级选择、弹性停站时间和按曲线来源分轮规划，并输出最终方案表和对比图。
- `schedule_to_timetable.py`：OpenTrack/时刻表导出辅助脚本，当前不属于默认主流程。
- `validate_opt_schedule.py`：旧版验证辅助脚本，当前不属于默认主流程；当前主线以 `ato_class_globall_v2.py` 的输出作为 DP 后收尾结果。

## 01_data_processing

用途：数据清洗、数据合并、区间划分、运行等级标注、载重处理。

- `add_class.py`：给处理后的运行数据补充或写入运行等级标签。
- `conbine.py`：合并多个数据文件或中间结果，文件名里 combine 拼写成了 conbine。
- `conbine_ato.py`：合并 ATO 相关数据或生成结果。
- `get_load.py`：从历史数据中提取或统计各区间载重信息。
- `process_line5.py`：早期/专项的 5 号线数据处理脚本。
- `process_target_v2.py`：新版目标数据处理脚本，偏向当前最新数据处理流程。
- `segment_chooose.py`：区间选择辅助脚本，文件名 choose 拼写成了 chooose。
- `segment_class.py`：区间运行等级分类或区间等级表生成。
- `step_1.py`：分步处理流程的第一步脚本。
- `step2.py`：分步处理流程的第二步脚本。

## 02_ato_template_generation

用途：ATO 模板提取、速度曲线生成、ATO policy 实验，以及 v5-v9 历史版本。

- `simulate_ato_run.py`：早期 ATO 运行曲线仿真脚本。
- `simulate_ATO_v6.py`：v6 版本速度曲线生成脚本，偏早期模板生成方案。
- `simulate_ATO_v7.py`：v7 版本速度曲线生成脚本，在 v6 基础上做批量/路径适配。
- `simulate_ATO_v9.py`：v9 验证版本，用于 class2/class3 等外推生成实验。
- `simulate_ato_with_comparison.py`：生成 ATO 曲线并与历史曲线做对比。
- `train_ato_policy.py`：训练 ATO 控制策略模型的早期版本。
- `train_ato_policy_v2.py`：ATO 控制策略模型训练的 v2 版本。
- `train_ATO_v5.py`：v5 版本 ATO 模板提取脚本。
- `train_ATO_v6.py`：v6 版本 ATO 模板提取脚本。
- `train_ATO_v7.py`：v7 版本 ATO 模板提取脚本。
- `train_ATO_v9.py`：v9 验证版本模板提取脚本，主要用于交叉验证和外推能力测试。

## 03_model_training

用途：训练能耗预测相关模型，包括 MLP、Transformer、残差 Transformer。

- `train_mlp_8to5.py`：MLP 能耗模型训练实验，名称暗示从 8 维特征到 5 类/5 档相关实验。
- `train_mlp_legacy.py`：旧版 MLP 能耗模型训练脚本。
- `train_residual.py`：残差模型训练脚本，用于学习物理模型和真实能耗之间的偏差。
- `train_residual_new.py`：新版残差模型训练脚本，当前更接近主线使用版本。
- `train_transformer.py`：Transformer 能耗模型训练脚本。

## 04_energy_calculation

用途：物理能耗、AI 修正能耗、单区间/全线能耗计算实验。

- `calculate_e copy.py`：`calculate_e.py` 的备份或实验副本。
- `calculate_e.py`：基础能耗计算脚本。
- `calculate_e_new.py`：新版能耗计算脚本。
- `energy.py`：早期或通用能耗计算入口。
- `energy_all_line.py`：全线能耗计算脚本，常用于把各区间结果汇总。
- `et_vt_04_06.py`：能耗-时间或速度-时间相关实验脚本，可能对应 04/06 版本数据。

## 05_optimization_scheduling

用途：运行等级组合优化、全局排图、专项区间优化、批处理运行入口。

- `ato_class_globall.py`：DP 排图优化 v1，文件名 global 拼写成了 globall。
- `optim.py`：通用优化实验脚本。
- `optimize_line5.py`：针对 5 号线的优化脚本。
- `optimize_meiyan_yongmaolu.py`：梅堰-永茂路区间专项优化。
- `optimize_sixiang_caoyi.py`：泗港/曹隘相关区间专项优化。
- `run _optimize_special_mlp1.py`：特殊区间 MLP 优化运行入口，文件名中包含空格。
- `run_global_class_schedual_v2.py`：全局等级排图 v2 运行入口，schedule 拼写成了 schedual。
- `run_global_class_schedule.py`：全局等级排图运行入口。
- `run_global_final.py`：全局最终方案生成或汇总入口。
- `run_global_schedule_mlp.py`：使用 MLP 能耗模型的全局排图优化。
- `run_global_schedule_residual.py`：使用残差模型的全局排图优化。
- `run_optimize_mlp.py`：MLP 模型优化运行入口。
- `run_optimize_residual.py`：残差模型优化运行入口。
- `run_optimize_residual_new.py`：新版残差模型优化运行入口。
- `run_optimize_special_residual.py`：特殊区间残差模型优化入口。
- `run_optimize_special_residual_new.py`：新版特殊区间残差模型优化入口。
- `run_optimize_transformer.py`：Transformer 模型优化运行入口。
- `runall.bat`：批处理脚本，用于一次性运行多个流程。

## 06_validation_reports

用途：验证优化结果、生成 trip 报告、对比历史与规划结果。

- `compera.py`：对比分析脚本，文件名 compare 拼写成了 compera。
- `full_line_validation_results.py`：生成或整理全线验证结果。
- `his6_compareSchedule.py`：trip6 或历史 6 相关排图对比脚本。
- `trip1_final_report.py`：trip1 最终报告生成脚本。
- `trip5_final_report.py`：trip5 最终报告生成脚本。
- `trip6_final_report.py`：trip6 最终报告生成脚本。
- `validate_v2_with_dp_schedule.py`：使用 DP 排图结果进行 v2 验证。

## 07_visualization

用途：画速度曲线、线路静态数据、能耗验证、DP 逻辑图、3D 轨迹图等。

- `make_big_picture`：生成大图/综合图的脚本，无 `.py` 后缀。
- `plot_3d_track.py`：绘制 3D 线路或轨迹图。
- `plot_3d_track_interactive.py`：绘制交互式 3D 线路或轨迹图。
- `plot_3d_track_simple.py`：简化版 3D 线路图。
- `plot_comparison.py`：绘制历史与规划/模型结果对比图。
- `plot_dp_core.py`：绘制 DP 核心逻辑图。
- `plot_dp_logic.py`：绘制 DP 流程/逻辑说明图。
- `plot_energy_es_curves.py`：绘制能耗-距离或累计能耗曲线。
- `plot_energy_validation_final.py`：绘制最终能耗验证图。
- `plot_five_class_et_map.py`：绘制五个 ATO 等级的能耗-时间关系图。
- `plot_load_distribution.py`：绘制载重分布图。
- `plot_mass_energy_per_class_v1.py`：按载重和等级分析能耗的 v1 图表。
- `plot_mass_energy_per_class_v2.py`：按载重和等级分析能耗的 v2 图表。
- `plot_track_static.py`：绘制线路坡度、曲率、限速等静态数据。
- `plot_v_t_s.py`：绘制速度-时间、速度-距离等曲线。
- `plot3zongzhou.py`：绘制纵轴/三维相关的轨迹或线路图。

## 08_analysis

用途：影响因素、历史载重、运行等级分布等探索性分析。

- `analyze_factors.py`：分析能耗、速度、载重、线路参数等影响因素。
- `analyze_historical_weights.py`：分析历史载重/重量分布。
- `summarize_section_class_levels.py`：扫描 Step2 输出的 `results_*.xlsx`，按区间统计实际出现过哪些运行等级，并单独列出 `曲线质量标签=0` 的正常曲线可用等级。

常用命令：

```powershell
python scripts_new/08_analysis/summarize_section_class_levels.py --data-dir data/data_processed_step2_v3_first5_curve_quality
```

输出位置：

- `output/analysis/section_class_distribution/section_class_distribution.xlsx`：区间等级汇总和趟次明细两个 sheet。
- `output/analysis/section_class_distribution/section_class_distribution.csv`：区间等级汇总 CSV。
- `output/analysis/section_class_distribution/section_class_segment_detail.csv`：每个 segment 的等级明细 CSV。

## 09_exports_opentrack

用途：导出时刻表、OpenTrack 输入文件、Excel 报告。

- `make_timetable.py`：生成时刻表。
- `onvert_to_opentrack_format.py`：转换为 OpenTrack 可读格式，文件名 convert 少了首字母 c。
- `output_excle.py`：导出 Excel 报告或表格，文件名 excel 拼写成了 excle。

## 99_misc_legacy

用途：临时测试、历史说明、无法明确归类的旧文件。

- `11.py`：临时实验脚本，建议后续打开确认是否仍有保留价值。
- `test.py`：简单测试脚本，目前主要用于读取 pkl 并检查模板内容。
- `代码逻辑`：旧的代码逻辑说明文件。
- `代码说明.txt`：较完整的代码说明/文字记录。

## 后续整理建议

1. 先把 `00_main_pipeline/` 作为正式主线维护对象。
2. 统一脚本里的项目根目录定位逻辑，让脚本在分类目录下也能直接运行。
3. 对 `99_misc_legacy/` 和历史版本脚本做二次筛选：能复用的保留，纯临时实验的归档。
4. 给主线脚本增加统一配置文件，减少脚本内部硬编码路径、trip 编号和输出版本号。
