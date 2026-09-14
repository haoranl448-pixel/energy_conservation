# 02_ato_template_generation 代码说明

本目录记录 ATO 模板提取和速度曲线生成方法的历代实验。当前正式流程使用 `00_main_pipeline/04_train_ATO_v8.py` 与 `05_simulate_ATO_v8.py`，本目录文件不应直接替换主线版本。

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `train_ato_policy.py` | 早期 ATO 控制策略学习脚本。 | 历史实验 |
| `train_ato_policy_v2.py` | ATO 控制策略学习的第二版实验。 | 历史实验 |
| `simulate_ato_run.py` | 使用早期策略或模板生成单次 ATO 运行曲线。 | 历史实验 |
| `simulate_ato_with_comparison.py` | 生成 ATO 曲线并与历史轨迹进行对比评价。 | 历史验证 |
| `train_ATO_v5.py` | v5 ATO 模板提取方法。 | 历史版本 |
| `simulate_ATO_v6.py` | v6 ATO 速度曲线生成方法。 | 历史版本 |
| `train_ATO_v6.py` | v6 ATO 模板训练方法。 | 历史版本 |
| `simulate_ATO_v7.py` | v7 ATO 曲线批量生成与路径适配版本。 | 历史版本 |
| `train_ATO_v7.py` | v7 ATO 模板训练版本。 | 历史版本 |
| `simulate_ATO_v9.py` | v9 模板外推/插值及等级生成验证。 | 实验版本 |
| `train_ATO_v9.py` | v9 模板提取和交叉验证实验。 | 实验版本 |

版本号表示算法迭代，不代表文件越新越适合当前主流程；当前已确认使用的是 `00_main_pipeline` 中的 v8 主线实现。
