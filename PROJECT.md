# 宁波地铁5号线 —— 节能排图优化系统

## 一句话概述

**在固定全线运行时间约束下，为地铁5号线26个区间选择合适的ATO运行等级组合，使总牵引能耗最低。**

---

## 目录结构速查

```
energy_conservation/
├── data/                          # 原始数据 & 处理后的数据
│   ├── raw_data/                  # 原始Excel（各区间的历史运行记录）
│   ├── data_processed/            # 按区间清洗后的数据 (cleaned_{区间}.xlsx)
│   ├── data_processed_new_v2/     # 新版清洗数据（v2格式）
│   ├── static/                    # 线路静态参数
│   │   ├── section_params.csv     # 坡度、曲率、限速等区间参数
│   │   ├── section_params_trip{N}.csv  # 各Trip的区间参数
│   │   ├── dwell_times.csv        # 停站时间配置
│   │   └── 线路基础数据（核查完毕）.xls  # 原始线路设计数据
│   └── 0308/                      # 原始MEVT事件记录数据
│
├── src/                           # 核心算法库
│   ├── physics/
│   │   ├── train_simu.py          # 理论能耗仿真模型（Davis方程 + 坡度/曲率）
│   │   └── static_model.py        # 线路静态参数读取（坡度/曲率/限速）
│   ├── models/
│   │   ├── definitions.py         # 神经网络模型定义（MLP、Transformer、残差Transformer）
│   │   └── wrapper.py             # 模型加载包装器
│   └── strategy/
│       └── __init__.py
│
├── scripts/                       # 100+ 脚本文件，按功能分10类
│   ├── train_ATO_v8.py           # [主线] 批量提取5个等级的相位模板 → multi_class_phase_artifacts.pkl
│   ├── train_ATO_v9.py           # [验证] 提取class2+class3模板（用于交叉验证）
│   ├── simulate_ATO_v8.py        # [主线] 多级父基因生成5级速度曲线（lam/theta搜索）
│   ├── simulate_ATO_v9.py        # [验证] class2模板→生成class3，验证外推能力
│   ├── ato_generated_results_energy.py  # 物理+AI残差计算能耗菜单
│   ├── ato_class_globall.py      # [v1] DP排图固定停站
│   ├── ato_class_globall_v2.py   # [v2] DP排图灵活停站（min/nominal）
│   ├── train_residual.py / train_residual_new.py     # 训练残差Transformer
│   ├── train_mlp_legacy.py       # 训练MLP能耗模型
│   ├── energy_all_line.py        # 全线物理+残差总能耗计算
│   ├── plot_energy_es_curves.py  # 26站e-s累积能耗曲线
│   ├── data_process.py           # 原始数据清洗
│   ├── validate_opt_schedule.py  # 验证排图结果
│   └── [其余80+辅助脚本]         # 可视化、对比、导出、实验等
│
├── output/                        # 所有输出结果
│   ├── analysis/                  # 能耗菜单、等级对照表、验证报告
│   ├── models/                    # 训练好的模型权重
│   │   ├── nn_results_residual/   # 残差Transformer权重
│   │   ├── nn_results_mlp/        # MLP模型权重
│   │   └── nn_results_transformer/ # Transformer权重
│   ├── ato_phase_results_v9/     # v9版相位模板工件
│   ├── ato_generated_results_new_v3/  # 生成的5级速度曲线
│   ├── ato_v9_validation_results_v2/  # v9验证结果
│   ├── schedule/                  # 排图方案输出
│   ├── energy_es_curves/          # 能耗曲线图
│   └── figures/                   # 其他图表
│
├── class1/ ~ class5/             # 各等级OpenTrack仿真参考数据（每级2个服务号）
│   └── OT_Course_Load{服务号}_{Wis/at/vs/vt}.csv
│
├── cpp_assets/                    # C++移植相关资源（26个区间的静态参数CSV）
├── py2c++/                        # Python到C++转换工具
├── nn_results/                    # 神经网络训练结果
├── output/                        # 输出汇总
├── requirements.txt               # Python依赖
└── PROJECT.md                     # 本文档
```

---

## 核心技术架构

本项目采用 **"物理模型 + AI残差 + 模板生成 + DP优化"** 四层架构：

### 第一层：物理模型（理论能耗计算）

文件：`src/physics/train_simu.py`

- 基于 **Davis方程** 计算列车牵引/制动功率：`F = P_a + P_b*v + P_c*v² + P_d*v³`
- 考虑因素：6节编组列车、分布式载重、实时坡度（`gradient`）、实时曲率（`curvature`）、速度曲线、动能变化
- 输出：**理论牵引能耗**（kWh）
- 关键参数：`P_a=1.65, P_b=0.0247, P_c=0.78, P_d=0.0028` 等
- 线路数据：26个区间，总长约36.5km，每个区间的坡度和曲率以2m分辨率离散化存储

### 第二层：AI残差模型（修正物理偏差）

文件：`src/models/definitions.py`, `scripts/train_residual*.py`

- 物理模型是标称值，实际能耗受司机策略、ATO特性、天气等影响有偏差
- 训练 **Transformer残差模型** 来预测物理模型与实测能耗之间的差值
- 输入：连续30个时间步的速度、位移、坡度、曲率序列
- 输出：该时刻的能耗残差
- 结构：`Input(6维, 30步) → Linear → PositionalEncoding → TransformerEncoder(2层, 4头) → Linear → 残差值`
- 模型权重位置：`output/models/nn_results_residual*/`

**能耗计算流程**：
```
速度曲线 → [物理引擎] → 理论能耗
                        +
速度曲线 → [Transformer残差模型] → 预测残差
                        =
                   最终预测能耗
```

### 第三层：ATO曲线模板生成（速度曲线"生成器"）

核心思想：**从历史数据中学习不同运行等级的速度曲线模式，然后参数化生成新曲线。**

**Step 1 — 提取相位模板** (`train_ATO_v8.py`)：
1. 从历史数据中按"运行等级"（Class 1-5）分组
2. 每条速度-距离曲线切分为3个相位：
   - **加速段**（牵引加速至巡航速度）
   - **中段**（巡航/惰行/微调速）
   - **制动段**（减速至停车）
3. 用 Savgol 滤波器平滑，插值归一化 → 得到该等级在该区间的**父基因模板**
4. 保存为 `multi_class_phase_artifacts.pkl`

**Step 2 — 参数化生成** (`simulate_ATO_v8.py`)：
1. 读取目标区间的父基因模板
2. 两个可调参数：
   - **lam (λ)**：时间缩放因子，控制总运行时间（0.55 ~ 3.5）
   - **theta (θ)**：巡航速度调节因子，控制峰值速度
3. 在 lam 网格上搜索，生成速度曲线
4. 约束检查：
   - 限速约束（不能超过80km/h 或特定曲线限速）
   - 终点停车约束（末速度≈0，位置误差≤2m）
   - 运行时间目标
5. 为每个区间的Class1-5各生成一条可行曲线

**Level 5 运行等级含义**：
| 等级 | 描述 | 运行时间 | 能耗 |
|------|------|----------|------|
| Class1 | 最慢 | 最长 | 最低 |
| Class2 | 较慢 | 较长 | 较低 |
| Class3 | 中等 | 中等 | 中等 |
| Class4 | 较快 | 较短 | 较高 |
| Class5 | 最快 | 最短 | 最高 |

### 第四层：DP排图优化（全局最优等级组合）

文件：`ato_class_globall_v2.py`

**问题描述**：全线26个区间，每区间有5个等级可选（Class1-5），每个选择对应一个运行时间和能耗。需要在总时间约束内找到能耗最低的等级组合。

**DP状态转移**：
```
dp[i][t] = 第i个区间到达时累计时间为t的最小能耗
dp[i+1][t + duration[i][k] + dwell[i]] = min(dp[i][t] + energy[i][k])
```

其中：
- `i` = 区间索引（0-25）
- `k` = 运行等级（1-5）
- `duration[i][k]` = 区间i在等级k下的运行时间
- `energy[i][k]` = 区间i在等级k下的牵引能耗
- `dwell[i]` = 停站时间（可在nominal和min之间弹性调整）

**v2版本改进**：
- 支持灵活停站（每个站可配置 min/nominal dwell time）
- 锁定换乘站停站时间（min==nominal）
- 弹性站可用停站时间换取运行时间
- 在slack窗口内搜索最优组合

---

## 主线流水线（端到端执行顺序）

```
第0步：数据准备
  data_process.py → 原始Excel → cleaned_{区间}.xlsx
  build_class_lookup_tables.py → 构建等级对照表

第1步：提取相位模板
  train_ATO_v8.py → multi_class_phase_artifacts.pkl

第2步：生成5级速度曲线
  simulate_ATO_v8.py → 每个区间生成 Class1-5 的V-S/V-T CSV

第3步：计算能耗菜单
  ato_generated_results_energy.py → ato_class_energy_menu.csv
  （每个区间 + 每个等级 → 物理+AI能耗值）

第4步：DP排图
  ato_class_globall_v2.py → 最优等级组合 + 时刻表

第5步：导出时刻表
  schedule_to_timetable.py → OpenTrack格式CSV

第6步：验证
  validate_opt_schedule.py → 验证报告
```

---

## 关键概念术语表

| 术语 | 含义 | 相关文件 |
|------|------|----------|
| **运行等级 (Class)** | ATO预设的5个运行模式，Class1最省电最慢，Class5最耗电最快 | 全局 |
| **相位模板 (Phase Template)** | 从历史数据提取的加速/巡航/制动三段归一化速度模式 | `train_ATO_v8.py` |
| **父基因 (Parent Gene)** | 同"相位模板"，用于生成同族曲线的基模板 | `simulate_ATO_v8.py` |
| **lam (λ)** | 时间缩放因子，控制曲线被拉伸/压缩的程度 | `simulate_ATO_v8.py` |
| **theta (θ)** | 速度调节因子，控制巡航平台速度高低 | `simulate_ATO_v8.py` |
| **能耗菜单 (Energy Menu)** | 每个区间×每个等级的能耗-时间对照表 | `ato_generated_results_energy.py` |
| **DP排图** | 动态规划求解全线最优等级组合 | `ato_class_globall_v2.py` |
| **残差模型** | Transformer网络，预测物理模型与真实能耗的偏差 | `train_residual_new.py` |
| **Section/区间** | 两个相邻车站之间的一段轨道 | 全局 |
| **Trip** | 一次完整的全线运行（26个区间） | `trip*_final_report.py` |
| **Slack** | 时间裕量，总允许时间与最短可能时间之差 | `ato_class_globall_v2.py` |
| **Dwell** | 停站时间，包括开关门和乘客上下车 | `ato_class_globall_v2.py` |
| **OpenTrack** | 轨道交通仿真软件，用于验证生成的时刻表 | `onvert_to_opentrack_format.py` |

---

## 版本演进路线

```
v6: 单区间class3模板提取 → class1-5生成（基础版）
v7: 路径适配，批量处理
v8: [主线] 多级父基因 + lam/theta二维搜索 + 限速检查
v9: [验证] class2↔class3交叉验证外推能力
```

**主线稳定流程**：train_ATO_v8 → simulate_ATO_v8 → ato_generated_results_energy → ato_class_globall_v2

**验证分支**：train_ATO_v9 → simulate_ATO_v9（只验证class2→class3的可迁移性）

---

## 关键数据格式

### cleaned_{区间}.xlsx（清洗后的区间运行数据）
包含列：
- `时刻`：时间戳
- `累计位移(m)`：沿轨道累计距离
- `速度(m/s)`：瞬时速度
- `curvature`：轨道曲率（m）
- `gradient`：坡度（‰）
- `重量`：列车总质量（kg）
- `运行等级`：Class1-5标签

### standard_class_times.csv（等级标准时间表）
```csv
区段, Class1, Class2, Class3, Class4, Class5
布政-张家潭, 120.5, 110.2, 100.8, 92.3, 85.1
...
```

### ato_class_energy_menu.csv（能耗菜单）
每行：区间 + 等级 + 运行时间 + 能耗（物理+AI）

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.11 |
| 深度学习 | PyTorch（MLP + Transformer） |
| 数据处理 | pandas, numpy, scipy |
| 可视化 | matplotlib（Agg后端） |
| 物理仿真 | 自建 Davis 方程模型 |
| 优化算法 | 动态规划 (DP) |
| 信号处理 | Savitzky-Golay 滤波器 |
| 外部软件 | OpenTrack（轨道仿真验证） |

---

## 常见工作流

### 1. 只改一个区间的等级配置
修改 `ato_class_globall_v2.py` 中的 `MANUAL_CONSTRAINTS` 字典即可

### 2. 重新训练残差模型
运行 `scripts/train_residual_new.py`，新权重保存在 `output/models/nn_results_residual_v2/`

### 3. 生成验证报告
运行 `scripts/trip*_final_report.py` → 输出在 `output/` 对应目录

### 4. 导出OpenTrack可读的时刻表
运行 `scripts/schedule_to_timetable.py`

---

## 注意事项

1. **中文字段名**：许多数据文件列名是中文，处理时注意编码（UTF-8 BOM）
2. **路径使用**：脚本使用 `Path(__file__).resolve().parent.parent` 定位项目根目录
3. **字体设置**：matplotlib 使用 `Microsoft YaHei` 字体显示中文
4. **限速处理**：线路最大限速80km/h，部分小半径曲线有更低限速（如300m半径→60km/h）
5. **载重**：默认使用各区间历史载重均值，也可指定固定值
6. **超速检测**：速度曲线峰值超过80km/h会被标记为不可行（overspeed）

---

*本文档由 Claude Code 于 2026-05-13 基于项目实际代码分析生成，可直接供其他 AI 工具（Gemini、Claude等）理解项目时使用。*
