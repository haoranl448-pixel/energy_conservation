# 08_analysis 代码说明

本目录用于分析样本特征、运行等级、载重、运行时间和模型误差，不负责生成正式 DP 方案。

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `summarize_section_class_levels.py` | 按区间统计历史出现的运行等级，并分别列出全部样本和质量标签0样本的可用等级。 | 当前分析工具 |
| `analyze_historical_weights.py` | 统计全线历史载重范围和分布，用于核实模型训练覆盖范围。 | 分析工具 |
| `analyze_factors.py` | 探索能耗与速度、时间、载重及线路参数之间的关系。 | 历史分析 |
| `build_historical_trip_section_report.py` | 为一趟车生成逐区间历史时间、能耗、速度、里程、质量和质量标签明细。 | 当前工具 |
| `batch_build_historical_trip_reports.py` | 批量调用历史区间统计逻辑，为多趟车生成轻量报告，不运行模型或 DP。 | 当前工具 |
| `plot_section_energy_time_scatter.py` | 将轨迹压缩为趟次级样本，绘制分区间运行时间-能耗散点图及等级中位趋势。 | 当前分析绘图 |
| `plot_triple_axis_all_runs.py` | 使用残差模型批量绘制各趟速度、里程、真实能耗和预测能耗三轴图。 | 当前模型诊断 |
| `plot_mlp_triple_axis_all_runs.py` | 读取已训练的旧 MLP 模型，批量重绘三轴对比图，不执行训练。 | 历史模型诊断 |
| `summarize_triple_axis_error.py` | 汇总三轴绘图结果中的能耗拟合误差，输出 CSV，条件允许时同时输出 Excel。 | 当前汇总工具 |

分析输出用于解释模型和规划结果，不应直接作为 OpenTrack 输入或覆盖正式规划表。
