# 05_optimization_scheduling 代码说明

本目录保存动态规划、全局排图和特殊区间优化的早期实现。当前三类规划方案以 `00_main_pipeline/08_ato_class_globall_v2.py` 和批处理脚本传入的模式为准。

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `ato_class_globall.py` | 第一版全线 ATO 等级动态规划。 | 历史版本 |
| `optim.py` | 通用优化算法试验入口。 | 实验 |
| `optimize_line5.py` | 5号线全线或指定范围的专项优化。 | 历史专项 |
| `optimize_meiyan_yongmaolu.py` | 梅堰-永茂路区间专项优化与问题定位。 | 历史专项 |
| `optimize_sixiang_caoyi.py` | 特定站间区间的专项优化。 | 历史专项 |
| `run_global_class_schedule.py` | 调用早期全局等级排图算法。 | 历史入口 |
| `run_global_class_schedual_v2.py` | 全局等级排图 v2 入口；`schedual` 为保留的旧拼写。 | 历史入口 |
| `run_global_final.py` | 汇总并输出早期全局最终方案。 | 历史入口 |
| `run_global_schedule_mlp.py` | 使用 MLP 能耗预测值执行全局排图。 | 历史实验 |
| `run_global_schedule_residual.py` | 使用残差模型能耗执行全局排图。 | 历史实验 |
| `run_optimize_mlp.py` | 运行 MLP 版本的区间优化。 | 历史实验 |
| `run_optimize_transformer.py` | 运行 Transformer 版本的区间优化。 | 历史实验 |
| `run_optimize_residual.py` | 运行早期残差模型优化。 | 历史实验 |
| `run_optimize_residual_new.py` | 运行新版残差模型优化实验。 | 参考实验 |
| `run_optimize_special_residual.py` | 对特殊区间执行残差模型优化。 | 历史专项 |
| `run_optimize_special_residual_new.py` | 特殊区间残差优化的后续版本。 | 参考专项 |
| `run _optimize_special_mlp1.py` | 特殊区间 MLP 优化入口；文件名含空格，不建议新流程引用。 | 历史专项 |
| `runall.bat` | 在 Windows 下串行调用若干旧优化脚本。 | 历史批处理 |

这些脚本的输入目录和模型版本并不统一。当前项目的新规划结果不要从本目录重新生成，以免与 v4 曲线、追溯表和新版 Simulink 口径混用。
