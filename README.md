# 宁波地铁5号线节能排图优化系统

## 项目简介

本项目用于在固定全线运行时间约束下，对宁波地铁5号线26个区间进行ATO运行等级选择，目标是使全线牵引能耗最低。

系统采用“物理模型 + AI残差 + ATO模板生成 + DP优化”架构，将历史运行数据、速度曲线模板和能耗模型结合起来，自动生成最优等级组合与排图方案。

## 目录结构

- `energy_conservation/`：项目核心数据与静态参数
- `src/`：算法与模型实现代码
- `scripts/`：训练、仿真、生成、优化、验证等脚本
- `output/`：输出结果文件夹，包含模型权重、排图方案、验证结果等
- `class1/` ~ `class5/`：各ATO等级对应的OpenTrack仿真参考数据
- `cpp_assets/`：C++移植相关资源
- `requirements.txt`：Python依赖
- `PROJECT.md`：项目详细说明

## 依赖环境

建议使用 Python 3.11 及以上版本。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## main.py 使用方法

推荐从项目根目录运行统一入口：

```powershell
cd D:\energy_conservation
python scripts_new\main.py
```

`scripts_new\main.py` 会按顺序调用 `scripts_new\00_main_pipeline\` 里的主线脚本。默认从第 1 步跑到最后一步，趟号为 `1`，DP 排图范围为全正向区间。

### 主线步骤

| 步骤 ID | 调用脚本 | 作用 |
| --- | --- | --- |
| `data_process` | `scripts_new\00_main_pipeline\01_data_process.py` | 原始数据清洗 |
| `residual_training` | `scripts_new\00_main_pipeline\02_train_residual_new.py` | 训练残差能耗模型 |
| `class_lookup` | `scripts_new\00_main_pipeline\03_build_class_lookup_tables.py` | 生成等级/标准时间对照表 |
| `ato_template` | `scripts_new\00_main_pipeline\04_train_ATO_v8.py` | 提取 ATO 分等级相位模板 |
| `ato_simulation` | `scripts_new\00_main_pipeline\05_simulate_ATO_v8.py` | 生成各区间各等级速度曲线 |
| `energy_menu` | `scripts_new\00_main_pipeline\06_ato_generated_results_energy.py` | 计算生成曲线能耗菜单 |
| `historical_baseline` | `scripts_new\00_main_pipeline\07_full_line_validation_results.py` | 生成历史用时/历史能耗基准 |
| `dp_schedule` | `scripts_new\00_main_pipeline\08_ato_class_globall_v2.py` | DP 排图优化并输出最终方案 |

### 常用命令

查看步骤列表：

```powershell
python scripts_new\main.py --list
```

只预览命令，不真正运行：

```powershell
python scripts_new\main.py --dry-run
```

从某一步开始跑：

```powershell
python scripts_new\main.py --from-step ato_simulation
```

只跑指定步骤：

```powershell
python scripts_new\main.py --only energy_menu,dp_schedule
```

指定数据目录：

```powershell
python scripts_new\main.py --data-dir data\data_processed_step2_v3_first5_with_class
```

指定趟号：

```powershell
python scripts_new\main.py --trip-no 6
```

指定 DP 目标总时间，单位秒：

```powershell
python scripts_new\main.py --target-time 692.65
```

### 区间范围

`--line-scope` 控制 DP 排图使用的正向区间数量。不写时默认全区间。

跑全区间：

```powershell
python scripts_new\main.py --from-step dp_schedule --data-dir data\data_processed_step2_v3_all_with_class
```

只跑前 5 个区间：

```powershell
python scripts_new\main.py --from-step dp_schedule --line-scope 5 --data-dir data\data_processed_step2_v3_first5_with_class
```

如果要跑前 `n` 个区间，把 `5` 换成对应数字即可，例如 `--line-scope 10`。

### 典型流程

前五区间测试，从 ATO 模板开始跑到 DP：

```powershell
python scripts_new\main.py --from-step ato_template --line-scope 5 --data-dir data\data_processed_step2_v3_first5_with_class
```

全区间正式跑，从 ATO 模板开始跑到 DP：

```powershell
python scripts_new\main.py --from-step ato_template --data-dir data\data_processed_step2_v3_all_with_class
```

如果前面结果已经生成，只重新做 DP：

```powershell
python scripts_new\main.py --only dp_schedule --line-scope 5 --data-dir data\data_processed_step2_v3_first5_with_class
```

日志默认输出到 `output\pipeline_logs\<时间戳>\`，每个步骤一个 `.log` 文件。最终 DP 方案默认输出到 `output\schedule\final_plan_report_v2\`。

## 核心流程

当前主线以 `scripts_new\main.py` 为统一入口，步骤顺序以 `main.py 使用方法` 中的表格为准。整体逻辑是：

1. 清洗原始数据，生成按区间整理后的 `results_*.xlsx`。
2. 训练残差能耗模型，并生成运行等级/标准时间对照表。
3. 从历史运行数据中提取 ATO 分等级相位模板。
4. 生成各区间、各等级的 ATO 速度曲线。
5. 计算生成曲线的物理模型 + 残差模型能耗菜单。
6. 生成历史基准，并用 DP 选择全局低能耗等级组合。

## 运行建议

- 先确保 `energy_conservation/data/` 中已有原始运行数据和静态参数文件。
- 优先使用 `python scripts_new\main.py` 运行主流程；调试时用 `--from-step`、`--only`、`--data-dir` 和 `--line-scope` 控制范围。
- 排图结果通常输出到 `output/schedule/` 及相关验证目录。

## 关键文件

- `PROJECT.md`：项目详细说明，包括技术架构、目录说明和核心算法
- `requirements.txt`：项目依赖包
- `scripts/`：主要的训练、仿真、优化、验证脚本
- `src/`：物理模型与神经网络模型代码

## 说明

该 README 为项目总体说明，详细设计与算法分析请参考 `PROJECT.md`。
