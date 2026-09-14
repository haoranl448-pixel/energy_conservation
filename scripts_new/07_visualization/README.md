# 07_visualization 代码说明

本目录集中保存科研绘图与结果汇总脚本。当前对外展示的25趟“历史 + 三类规划”图以 `plot_simu_code_trip_compare.py` 为主。

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `plot_simu_code_trip_compare.py` | 对冻结的25趟历史及三类 DP 方案进行新版 Simulink 物理仿真，绘制累计能耗-里程和速度-里程图；也支持仅重绘缓存。 | 当前核心绘图 |
| `summarize_simu_code_complete25.py` | 从25趟仿真缓存生成历史误差和三类规划节能量汇总 CSV。 | 当前汇总工具 |
| `plot_trip_real_vs_simu_physical_es.py` | 按全局趟次追溯关系比较真实历史累计能耗与纯 Simulink 机械能耗，并输出分区间结果。 | 当前验证工具 |
| `plot_comparison.py` | 比较仿真与真实轨迹的速度、加速度和位移。 | 历史绘图 |
| `plot_energy_es_curves.py` | 绘制实测能耗与物理模型能耗的累计 E-s 曲线。 | 历史绘图 |
| `plot_energy_validation_final.py` | 绘制旧版能耗模型最终验证图并计算拟合指标。 | 历史绘图 |
| `plot_five_class_et_map.py` | 展示一个区间五个运行等级的能耗、速度和坡度关系。 | 分析绘图 |
| `plot_load_distribution.py` | 绘制不同服务号在各区间的载重分布。 | 分析绘图 |
| `plot_mass_energy_per_class_v1.py` | 绘制各等级质量-能耗敏感度关系的 v1 版本。 | 历史绘图 |
| `plot_mass_energy_per_class_v2.py` | 批量绘制全线各区间、各等级的质量-能耗关系。 | 分析绘图 |
| `plot_track_static.py` | 绘制线路坡度、曲率、限速等静态参数，包含梅堰-永茂路脏数据兼容逻辑。 | 线路绘图 |
| `plot_3d_track.py` | 生成线路三维轨道图。 | 线路绘图 |
| `plot_3d_track_simple.py` | 生成简化版三维线路图。 | 线路绘图 |
| `plot_3d_track_interactive.py` | 使用 Plotly 生成可旋转、缩放的交互式三维线路图。 | 线路绘图 |
| `plot_dp_core.py` | 绘制动态规划核心计算逻辑示意图。 | 汇报素材 |
| `plot_dp_logic.py` | 绘制完整 DP 排图流程示意图。 | 汇报素材 |
| `make_big_picture` | 生成早期综合大图；无 `.py` 扩展名。 | 历史工具 |
| `plot_v_t_s.py` | 旧版速度-时间/速度-距离绘图代码，当前主体已注释。 | 停用 |
| `plot3zongzhou.py` | 旧版三纵轴模型结果图代码，当前主体已注释。 | 停用 |

当前25趟成果的推荐命令：

```powershell
python scripts_new\07_visualization\plot_simu_code_trip_compare.py --render-only
python scripts_new\07_visualization\summarize_simu_code_complete25.py
```
