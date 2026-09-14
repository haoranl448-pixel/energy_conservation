# 00_main_pipeline 代码说明

本目录包含 `scripts_new/main.py` 调用的当前主流程。带 `01_` 至 `08_` 编号的文件按执行顺序排列，是当前推荐版本；同名无编号文件是早期整理稿或兼容版本，不能与编号版混用。

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `01_data_process.py` | 清洗原始动态/静态数据，按时间连续性切分趟次，计算速度、加速度、位移和实测能耗，并生成区间级 `results_*.xlsx`。 | 主线第1步 |
| `02_train_residual_new.py` | 按区间训练物理模型残差修正网络，保存模型、特征缩放器和训练验证结果。 | 主线第2步 |
| `03_build_class_lookup_tables.py` | 统计各区间运行等级及标准时间，生成 ATO 模板和后续规划所需的等级对照表。 | 主线第3步 |
| `04_train_ATO_v8.py` | 从正常历史曲线中按区间、等级提取加速/中段/制动三相位模板，并记录代表样本与可靠性。 | 主线第4步 |
| `05_simulate_ATO_v8.py` | 根据相位模板生成各区间 Class1-Class5 候选速度曲线，对缺失等级进行受控插值/相邻外推并执行超速检查。 | 主线第5步 |
| `06_ato_generated_results_energy.py` | 将候选速度曲线输入物理模型和残差模型，结合该趟载重生成“区间-等级-时间-能耗”菜单。 | 主线第6步 |
| `07_full_line_validation_results.py` | 按全局趟次追溯关系提取历史运行时间、停站时间、实测能耗和历史模型能耗，形成 DP 对比基准。 | 主线第7步 |
| `08_ato_class_globall_v2.py` | 在总时间、停站和曲线来源约束下进行动态规划，输出标准 DP、Energy First、停站放宽等规划表。 | 主线第8步 |
| `trip_traceability.py` | 公共追溯模块；把一个全局趟次映射到每个区间真正对应的本地 `segment` 和 `run_id`，供第6至第8步复用。 | 主线公共模块 |
| `data_process.py` | 数据处理脚本的较早注释整理版，与编号主线版本存在差异。 | 历史/参考 |
| `build_class_lookup_tables.py` | 等级统计脚本的较早注释整理版。 | 历史/参考 |
| `train_ATO_v8.py` | ATO 模板训练的早期多等级版本。 | 历史/参考 |
| `simulate_ATO_v8.py` | ATO 曲线生成的早期版本。 | 历史/参考 |
| `ato_generated_results_energy.py` | 能耗菜单计算的早期版本。 | 历史/参考 |
| `ato_class_globall_v2.py` | 弹性停站 DP 的早期版本；当前规划规则以后续编号版为准。 | 历史/参考 |
| `schedule_to_timetable.py` | 把旧版 `schedule_table.csv` 和区间参数转换成 OpenTrack 导入 CSV。 | 历史导出工具 |
| `validate_opt_schedule.py` | 还原旧版 AI 驾驶风格并对优化排图进行能耗结算和历史对比。 | 历史验证工具 |

推荐入口：

```powershell
python scripts_new\main.py --list
python scripts_new\main.py --trip-no 1 --line-scope full
```

DP 区间范围由 `--line-scope` 控制：`full` 表示全正向26区间，数字表示仅运行前 N 个区间。
