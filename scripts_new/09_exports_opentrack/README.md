# 09_exports_opentrack 使用说明

这个文件夹负责把主流程输出的优化结果接到 OpenTrack 里，主要包含三件事：

1. 把 DP 规划表转换成 OpenTrack timetable XML。
2. 通过 OpenTrack OTD API 下发仿真控制、位置回报开关、区间限速。
3. 读取 OpenTrack 仿真输出 `.tsvP`，和历史真实曲线做 v-t、v-s、E-t、E-s 对比。

下面命令默认在项目根目录运行：

```powershell
cd D:\energy_conservation
```

## 1. 文件夹里的主要脚本

| 脚本 | 作用 |
| --- | --- |
| `batch_build_trip_plans_and_timetables.py` | 批量生成每趟的 DP 优化表和 OpenTrack timetable。 |
| `make_opentrack_timetables.py` | 把 `Final_Planning_Comparison*.csv` 转成 OpenTrack timetable XML。 |
| `opentrack_otd_listener.py` | 本机启动一个 OTD 接收端，接收 OpenTrack 发来的仿真消息并写 JSONL 日志。 |
| `opentrack_otd_client.py` | 向 OpenTrack 的 OTD Server 发送 SOAP/HTTP 命令。 |
| `opentrack_otd_build_route_map.py` | 根据 OTD 日志生成“站间区间 -> OpenTrack routeID/offset”的映射表。 |
| `make_opentrack_speed_limits.py` | 根据 DP 规划等级和能耗菜单，生成并可下发优化方案限速。 |
| `build_opentrack_history_speed_lookup.py` | 从 Step2 数据中预先提取历史每趟每区间速度上限，生成可复用查表 CSV。 |
| `make_opentrack_history_speed_limits_from_lookup.py` | 从历史速度查表 CSV 中取某一趟限速，并可下发给 OpenTrack。 |
| `plot_opentrack_tsvp_vs_history.py` | 读取 OpenTrack `.tsvP` 输出，和历史曲线绘图对比。 |

## 2. OpenTrack OTD/API 的端口关系

这里最容易混的是两个方向：

| 方向 | 端口 | 谁是服务端 | 作用 |
| --- | --- | --- | --- |
| Python -> OpenTrack | `9002` | OpenTrack | Python 向 OpenTrack 发送 `startSimulation`、`setPositionSpeed` 等命令。 |
| OpenTrack -> Python | `9004` | Python | OpenTrack 把 `trainArrival`、`routeEntry`、`trainPositionReport` 等消息发给 Python。 |

所以：

- `opentrack_otd_client.py` 默认发到 `127.0.0.1:9002/otd`。
- `opentrack_otd_listener.py` 默认监听 `127.0.0.1:9004/otd`。
- 浏览器打开 `http://127.0.0.1:9004/otd` 看到 `OpenTrack OTD listener is running.`，只说明 Python 的监听端已经启动。
- 如果命令行提示 `Connection refused ... 127.0.0.1:9002`，说明 OpenTrack 的 OTD Server 没开，或者 OpenTrack 没有以支持 OTD 的方式运行。

## 3. OpenTrack 设置

在 OpenTrack 的 OTD Settings 里建议这样设置：

| 项 | 建议值 |
| --- | --- |
| OpenTrack Server Port | `9002` |
| Server Status | `Running` |
| OTD Server | `localhost` 或 `127.0.0.1` |
| OTD Server Port | `9004` |
| Communication | `SOAP / DIME over HTTP` |
| Use OTD-Communication | 勾选 |
| Used Messages | 至少勾选 Simulation、Train、Train Position Report、Timetable、Route 相关消息 |

注意：

- 如果 OpenTrack 弹窗 `Cannot connect to Server: localhost Port: 9004`，说明 Python 监听器没有启动，或者 9004 被占用/代理拦截。
- 开梯子一般不影响 `127.0.0.1`，但如果代理启用了 TUN/全局拦截，优先关掉 TUN 或把本地地址加入直连。
- `sent_without_waiting_for_http_response` 通常不是失败。脚本默认发送 HTTP 请求后不等待 OpenTrack 返回，因为当前 OpenTrack 版本有时会接收命令但不按标准 HTTP 返回，强行等待反而容易 timeout。

## 4. 推荐完整流程

### 4.1 生成前 10 趟规划表和 timetable

如果前 10 趟的 `energy_menu` 和历史基准已经生成好，可以跳过前置计算：

```powershell
python scripts_new\09_exports_opentrack\batch_build_trip_plans_and_timetables.py `
  --trips 1-10 `
  --line-scope full `
  --data-dir data\data_processed_step2_v3_all_curve_quality `
  --output-dir output\schedule\batch_trip_reports_trip1_10 `
  --reuse-existing-prep `
  --no-plots `
  --continue-on-error
```

输出结构大致是：

```text
output\schedule\batch_trip_reports_trip1_10\
  trip001\
    Final_Planning_Comparison.csv
    Final_Planning_Comparison_Energy_First.csv
    full_line1_validation_results.csv
    ato_class_energy_menu1_new_v3.csv
    timetables\
      priority_dp_trip001.xml
      priority_history_trip001.xml
      energy_first_dp_trip001.xml
      energy_first_history_trip001.xml
      station_id_mapping.csv
  trip002\
  ...
```

`make_opentrack_timetables.py` 已经支持 `--course-suffix` 和 `--file-suffix`。批处理脚本会自动传入 `trip001`、`trip002` 这种后缀，避免 10 趟导入 OpenTrack 时 course id 互相覆盖。

### 4.2 只转换某一趟 timetable

如果已经有单趟 CSV，也可以单独转换：

```powershell
python scripts_new\09_exports_opentrack\make_opentrack_timetables.py `
  --priority-csv output\schedule\batch_trip_reports_trip1_10\trip001\Final_Planning_Comparison.csv `
  --energy-first-csv output\schedule\batch_trip_reports_trip1_10\trip001\Final_Planning_Comparison_Energy_First.csv `
  --template eg.xml `
  --output-dir output\schedule\batch_trip_reports_trip1_10\trip001\timetables `
  --start-time 08:00:00 `
  --history-dwell-mode fixed `
  --fixed-history-dwell 30 `
  --write-station-map `
  --course-suffix trip001 `
  --file-suffix trip001
```

说明：

- `priority_dp_trip001.xml`：优先使用真实等级的优化方案。
- `priority_history_trip001.xml`：同一趟历史运行方案。
- `energy_first_dp_trip001.xml`：如果最高优先级方案存在，也额外给出的最低能耗方案。
- `energy_first_history_trip001.xml`：对应历史方案。
- `station_id_mapping.csv`：后面生成 route map 需要用。

## 5. OTD API 实际使用步骤

### 5.1 启动 Python 监听器

先开一个 PowerShell，保持不关闭：

```powershell
python scripts_new\09_exports_opentrack\opentrack_otd_listener.py `
  --host 127.0.0.1 `
  --port 9004 `
  --log-dir output\opentrack_otd_logs
```

启动成功后会打印类似：

```text
Listening for OpenTrack OTD messages on http://127.0.0.1:9004/otd
JSONL log: output\opentrack_otd_logs\otd_messages_YYYY-MM-DDTHHMMSS.jsonl
```

可以用浏览器打开：

```text
http://127.0.0.1:9004/otd
```

看到 `OpenTrack OTD listener is running.` 就说明 9004 监听正常。

### 5.2 在 OpenTrack 里开启 OTD 通信

打开 OpenTrack 的 OTD Settings：

1. `OpenTrack Server Port` 设为 `9002`，确认 `Server Status` 是 `Running`。
2. `OTD Server` 填 `localhost` 或 `127.0.0.1`。
3. `OTD Server Port` 填 `9004`。
4. 勾选 `Use OTD-Communication`。
5. 勾选需要的消息类型，尤其是 Route、Train、Train Position Report。

### 5.3 测试 Python 是否能向 OpenTrack 发命令

建议先发一个不破坏仿真的命令：

```powershell
python scripts_new\09_exports_opentrack\opentrack_otd_client.py set-position-reports priority_dp_trip001 yes --interval 1
```

如果输出：

```text
sent_without_waiting_for_http_response
```

通常表示命令已发出。不要轻易加 `--wait-response`，除非确认当前 OpenTrack 会正常返回 HTTP 响应。

常用命令：

```powershell
# 开始仿真
python scripts_new\09_exports_opentrack\opentrack_otd_client.py start-simulation

# 暂停仿真
python scripts_new\09_exports_opentrack\opentrack_otd_client.py pause-simulation

# 结束仿真
python scripts_new\09_exports_opentrack\opentrack_otd_client.py end-simulation

# 重置 timetable
python scripts_new\09_exports_opentrack\opentrack_otd_client.py reset-timetable

# 从 XML 用 API 逐条下发 timetable entry
python scripts_new\09_exports_opentrack\opentrack_otd_client.py load-timetable `
  output\schedule\batch_trip_reports_trip1_10\trip001\timetables\priority_dp_trip001.xml `
  --reset
```

如果你已经在 OpenTrack UI 里导入 XML，就不一定需要 `load-timetable`。

`load-timetable` 默认会读取 XML 里的 `<deltaLoad>`，并随 `addTimetableEntry` 一起发送给 OpenTrack。若某个 OpenTrack 版本不接受这个字段，可以加：

```powershell
--no-delta-load
```

## 6. 生成 route map

route map 是后面下发区间限速的关键，它把我们的中文区间映射到 OpenTrack 的 `routeID` 和 offset。

先跑一遍对应 course 的仿真，让 listener 收到日志。然后执行：

```powershell
python scripts_new\09_exports_opentrack\opentrack_otd_build_route_map.py `
  --log output\opentrack_otd_logs `
  --station-map output\schedule\batch_trip_reports_trip1_10\trip001\timetables\station_id_mapping.csv `
  --train-id priority_dp_trip001 `
  --course-id priority_dp_trip001 `
  --output-dir output\opentrack_route_map `
  --prefix priority_dp_trip001
```

输出：

```text
output\opentrack_route_map\priority_dp_trip001_route_map.csv
output\opentrack_route_map\priority_dp_trip001_position_reports.csv
output\opentrack_route_map\priority_dp_trip001_station_events.csv
output\opentrack_route_map\priority_dp_trip001_route_entries.csv
```

如果日志里没有 `trainPositionReport`，脚本会退回使用 `routeEntry`。这时 route map 仍然能用，但某些 offset 可能需要人工检查 warning 列。

## 7. 下发优化方案限速

优化方案限速来自：

- DP 最终规划表里的等级，例如 `class3`。
- 对应该区间、等级的 `generated_curve.csv` 里计算出的巡航段平均速度。
- route map 里该区间对应的 OpenTrack routeID。

默认不再直接使用峰值速度。脚本会读取：

```text
output\ato_generated_results_new_v4\<区间>\<class>_generated_curve.csv
```

然后筛选“接近最高速度且加速度较小”的巡航段，取平均速度作为 `setPositionSpeed` 的速度基准。这样可以避免某个瞬时最高点把 OpenTrack 限速抬得过高。

如果需要恢复旧逻辑，可以加：

```powershell
--speed-method peak
```

先只生成 CSV，不下发：

```powershell
python scripts_new\09_exports_opentrack\make_opentrack_speed_limits.py `
  --plan-csv output\schedule\batch_trip_reports_trip1_10\trip001\Final_Planning_Comparison.csv `
  --energy-menu output\schedule\batch_trip_reports_trip1_10\trip001\ato_class_energy_menu1_new_v3.csv `
  --route-map output\opentrack_route_map\priority_dp_trip001_route_map.csv `
  --output-csv output\opentrack_speed_limits\speed_limits_priority_trip001.csv `
  --train-id priority_dp_trip001 `
  --range-mode single-route `
  --speed-method cruise-avg `
  --opentrack-speed-unit kmh
```

确认 CSV 没问题后，加 `--send` 下发：

```powershell
python scripts_new\09_exports_opentrack\make_opentrack_speed_limits.py `
  --plan-csv output\schedule\batch_trip_reports_trip1_10\trip001\Final_Planning_Comparison.csv `
  --energy-menu output\schedule\batch_trip_reports_trip1_10\trip001\ato_class_energy_menu1_new_v3.csv `
  --route-map output\opentrack_route_map\priority_dp_trip001_route_map.csv `
  --output-csv output\opentrack_speed_limits\speed_limits_priority_trip001.csv `
  --train-id priority_dp_trip001 `
  --range-mode single-route `
  --speed-method cruise-avg `
  --opentrack-speed-unit kmh `
  --send
```

重要参数：

- `--train-id`：建议填当前 OpenTrack course/train id，避免限速影响其他车。
- `--opentrack-speed-unit kmh`：当前模型速度单位按 km/h 处理。之前如果出现统一 50 或 80，优先检查这里和 OpenTrack 项目显示单位。
- `--speed-method cruise-avg`：默认值，使用巡航段平均速度作为限速基准。
- `--speed-method peak`：使用旧逻辑，直接取 `energy_menu` 里的峰值速度。
- `--cruise-min-speed-ratio`：巡航段速度阈值，默认取峰值的 `0.92` 以上。
- `--cruise-accel-eps`：巡航段加速度阈值，默认 `0.25 m/s^2`。
- `--range-mode single-route`：每个区间只对当前 route 下发限速，当前测试更稳。
- `--speed-margin`：可以给限速额外加一点余量，例如 `--speed-margin 1`。
- `--ceil-speed`：把限速向上取整。

## 8. 下发历史方案限速

历史限速建议先生成 lookup 表。这样后面不同趟次不用每次重新扫描所有 Step2 xlsx。

### 8.1 生成历史速度 lookup

全线：

```powershell
python scripts_new\09_exports_opentrack\build_opentrack_history_speed_lookup.py `
  --data-dir data\data_processed_step2_v3_all_curve_quality `
  --route-map output\opentrack_route_map\priority_dp_trip001_route_map.csv `
  --output-csv output\opentrack_speed_limits\history_speed_lookup.csv `
  --quality-label 0
```

只取前 5 个区间：

```powershell
python scripts_new\09_exports_opentrack\build_opentrack_history_speed_lookup.py `
  --data-dir data\data_processed_step2_v3_all_curve_quality `
  --route-map output\opentrack_route_map\priority_dp_trip001_route_map.csv `
  --output-csv output\opentrack_speed_limits\history_speed_lookup_first5.csv `
  --section-count 5 `
  --quality-label 0
```

`--quality-label 0` 表示只用正常曲线。若要保留所有质量标签，可以去掉这个参数。

### 8.2 从 lookup 取某一趟历史限速

只生成 CSV：

```powershell
python scripts_new\09_exports_opentrack\make_opentrack_history_speed_limits_from_lookup.py `
  --speed-lookup-csv output\opentrack_speed_limits\history_speed_lookup.csv `
  --route-map output\opentrack_route_map\priority_dp_trip001_route_map.csv `
  --output-csv output\opentrack_speed_limits\speed_limits_history_trip001.csv `
  --trip-no 1 `
  --traceability-manifest data\data_processed_step2_v3_all_curve_quality_traceability\trip_traceability_manifest_v1.csv `
  --method max `
  --range-mode single-route `
  --train-id priority_history_trip001
```

下发给 OpenTrack：

```powershell
python scripts_new\09_exports_opentrack\make_opentrack_history_speed_limits_from_lookup.py `
  --speed-lookup-csv output\opentrack_speed_limits\history_speed_lookup.csv `
  --route-map output\opentrack_route_map\priority_dp_trip001_route_map.csv `
  --output-csv output\opentrack_speed_limits\speed_limits_history_trip001.csv `
  --trip-no 1 `
  --traceability-manifest data\data_processed_step2_v3_all_curve_quality_traceability\trip_traceability_manifest_v1.csv `
  --method max `
  --range-mode single-route `
  --train-id priority_history_trip001 `
  --send
```

说明：

- 使用追溯表时，`--trip-no 1` 表示全线串联后的第 1 趟，使用 1-based 编号；每个区间会查询自己真正对应的本地 `segment`，不再机械地取各区间本地第 1 趟。
- 当前脚本默认查找 `data\data_processed_step2_v3_all_curve_quality_traceability\trip_traceability_manifest_v1.csv`。也可以用 `--traceability-manifest` 显式指定其他追溯表。
- 输出 CSV 的 `traceability_segment` 与 `matched_segment` 应逐行一致。全线实验不建议使用 `--ignore-traceability`，该参数只用于兼容旧的区间本地趟次编号。
- `--method max` 表示取该历史曲线的最大速度作为限速。
- 也可以用 `p99`、`p95`、`avg` 或 `cruise-avg`；`cruise-avg` 表示取巡航段平均速度。

## 9. OpenTrack 输出曲线对比

OpenTrack 仿真后会在输出目录生成 `.tsvP` 文件，例如：

```text
D:\OutPut\OT_priority_history.tsvP
```

绘制 v-t、v-s、E-t、E-s，并和历史曲线对比：

```powershell
python scripts_new\09_exports_opentrack\plot_opentrack_tsvp_vs_history.py `
  --tsvp D:\OutPut\OT_priority_history.tsvP `
  --data-dir data\data_processed_step2_v3_all_curve_quality `
  --route-map output\opentrack_route_map\priority_dp_trip001_route_map.csv `
  --output-dir output\opentrack_compare_plots `
  --prefix priority_history_trip001 `
  --trip-no 1 `
  --section-start 1 `
  --section-count 26 `
  --image-format jpg
```

图里会包含：

- `v-t`：速度-时间。
- `v-s`：速度-距离。
- `E-t`：累计能耗-时间。
- `E-s`：累计能耗-距离。
- `E-t` 和 `E-s` 右下角会标出 OpenTrack 相对历史的能耗误差百分比。

## 10. 常见问题

### 10.1 `Connection refused ... 127.0.0.1:9002`

原因：OpenTrack 没有在 9002 上监听。

处理：

1. 打开 OpenTrack OTD Settings。
2. 确认 `OpenTrack Server Port = 9002`。
3. 确认 `Server Status = Running`。
4. 确认 OpenTrack 以支持 OTD 的方式启动。

### 10.2 OpenTrack 弹窗 `Cannot connect to Server: localhost Port: 9004`

原因：OpenTrack 想把消息发给 Python，但 Python listener 没开。

处理：

```powershell
python scripts_new\09_exports_opentrack\opentrack_otd_listener.py --port 9004 --log-dir output\opentrack_otd_logs
```

再用浏览器打开：

```text
http://127.0.0.1:9004/otd
```

能看到 listener 页面后，再回 OpenTrack 勾选 OTD 通信。

### 10.3 命令 timeout

如果是发送命令时 timeout，先不要加 `--wait-response`。默认模式输出：

```text
sent_without_waiting_for_http_response
```

通常就是当前 OpenTrack 版本下更稳定的发送方式。

### 10.4 route map 显示 `Position reports: 0`

说明没有收到 `trainPositionReport`。如果仍然有 `route entries`，脚本会用 `routeEntry` 退化生成 route map。

处理：

1. OpenTrack OTD Settings 勾选 `Train Position Report Messages`。
2. 用 client 打开位置回报：

```powershell
python scripts_new\09_exports_opentrack\opentrack_otd_client.py set-position-reports priority_dp_trip001 yes --interval 1
```

3. 重新跑一次仿真，再生成 route map。

### 10.5 限速看起来没有生效

优先检查：

1. `--train-id` 是否和 OpenTrack Console 里的 `trainID` 完全一致。
2. route map 的 routeID 是否和 Console 里的 `routeEntry routeID=...` 对得上。
3. `--opentrack-speed-unit` 是否正确，当前建议 `kmh`。
4. 是否在仿真开始前下发限速。
5. OpenTrack 线路模型本身 edge 限速是否低于 API 下发限速。API 下发的速度不能突破基础线路约束，只能作为更低或相近的限制参与仿真。

## 11. 推荐前 10 趟试验顺序

1. 批量生成 `trip001` 到 `trip010` 的规划表和 timetable。
2. 在 OpenTrack 中导入或加载 `priority_history_trip001.xml`，先跑历史方案，确认基础 timetable 正常。
3. 启动 listener，跑一次历史方案，生成 `priority_history_trip001_route_map.csv`。
4. 用历史 lookup 生成并下发历史限速，跑仿真，得到 `OT_priority_history.tsvP`。
5. 导入或加载 `priority_dp_trip001.xml`。
6. 用 `make_opentrack_speed_limits.py` 生成并下发优化方案限速，跑仿真，得到 `OT_priority_dp.tsvP`。
7. 分别用 `plot_opentrack_tsvp_vs_history.py` 画历史/优化方案和真实历史曲线的对比图。

先把第 1 趟跑顺，再复制相同步骤到第 2 到第 10 趟。这样最容易定位是 timetable、route map、限速还是 OpenTrack 模型本身的问题。
