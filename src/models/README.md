# models 代码说明

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `definitions.py` | 定义 MLP、纯 Transformer、残差 Transformer 及位置编码等 PyTorch 网络结构。 | 公共模型定义 |
| `wrapper.py` | 根据模型类型加载权重、scaler 和特征配置，提供统一的 `EnergyPredictor` 推理接口。 | 公共推理封装 |

模型权重必须与训练时保存的 scaler、特征顺序和网络结构匹配。
