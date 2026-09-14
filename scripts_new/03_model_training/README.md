# 03_model_training 代码说明

本目录保存能耗预测模型的训练实验。当前生产规划使用的残差训练入口为 `00_main_pipeline/02_train_residual_new.py`。

| 文件 | 具体作用 | 状态 |
| --- | --- | --- |
| `train_mlp_legacy.py` | 训练早期 MLP 能耗模型并保存模型与缩放器。 | 历史模型 |
| `train_mlp_8to5.py` | 进行特定输入/输出结构的 MLP 能耗预测实验。 | 实验模型 |
| `train_transformer.py` | 直接使用 Transformer 对轨迹能耗进行序列建模。 | 历史实验 |
| `train_residual.py` | 训练物理模型与实测能耗之间的残差模型。 | 历史版本 |
| `train_residual_new.py` | 残差模型训练的后续实验版本；主线仍以 `00_main_pipeline/02_train_residual_new.py` 为准。 | 参考版本 |

训练脚本输出的检查点、缩放器和特征定义必须成套使用。不要只替换 `.pth` 文件而复用另一版本的 scaler 或特征列表。
