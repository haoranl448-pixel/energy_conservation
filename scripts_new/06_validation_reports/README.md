# 06_validation_reports 代码说明

本目录保存早期历史/规划验证和指定趟次报告脚本。当前25趟 Simulink 验证图由 `07_visualization/plot_simu_code_trip_compare.py` 生成。

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `compera.py` | 对两类能耗或排图结果进行基础对比；文件名为早期拼写。 | 历史工具 |
| `full_line_validation_results.py` | 汇总各区间历史回放结果，生成全线验证 CSV。 | 历史版本 |
| `his6_compareSchedule.py` | 针对 Trip 6 比较历史与排图方案。 | 趟次专项 |
| `trip1_final_report.py` | 生成 Trip 1 的旧版最终报告。 | 趟次专项 |
| `trip5_final_report.py` | 生成 Trip 5 的旧版最终报告。 | 趟次专项 |
| `trip6_final_report.py` | 生成 Trip 6 的旧版最终报告。 | 趟次专项 |
| `validate_v2_with_dp_schedule.py` | 使用旧版 v2 模型对 DP 排图结果进行验证。 | 历史验证 |

本目录报告适合复查早期实验，不作为当前25趟成果的统一统计来源。
