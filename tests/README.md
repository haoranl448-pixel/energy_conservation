# tests 代码说明

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `test_dp_dwell_rounding.py` | 验证历史停站时间分配、半秒目标时间和失败趟次的舍入规则，防止总时间出现累计偏差。 | 当前回归测试 |
| `test_simu_code_replay.py` | 验证新版/旧版物理计算加速实现的一致性、线路边界阻力、里程修正和规划回算数据完整性。 | 当前回归测试 |

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```
