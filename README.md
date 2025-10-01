# MABe Mouse Behavior Detection - Training Framework

基于 Kaggle MABe 鼠标行为检测竞赛的训练框架。

## 项目结构

```
mabe_mouse_behavior/
├── configs/
│   └── config.yaml              # 训练配置文件
├── src/
│   ├── data/
│   │   └── dataset.py          # 数据加载模块
│   ├── models/
│   │   └── transformer_model.py # 模型架构 (Transformer & LSTM)
│   └── utils/
│       └── trainer.py          # 训练器
├── notebooks/                   # Jupyter notebooks
├── train.py                     # 主训练脚本
├── requirements.txt             # 依赖包
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

1. 从 Kaggle 下载 MABe 竞赛数据
2. 将数据组织为以下结构：

```
data/
├── train/
│   ├── video1.npy
│   ├── video2.npy
│   └── ...
├── val/
│   ├── video1.npy
│   └── ...
├── train_annotations.json
└── val_annotations.json
```

3. 更新 `configs/config.yaml` 中的数据路径

## 配置

编辑 `configs/config.yaml` 设置：

- **数据参数**: 数据路径、序列长度等
- **模型参数**: 模型类型 (transformer/lstm)、隐藏层维度、层数等
- **训练参数**: batch size、学习率、epochs 等

关键配置项：

```yaml
model_type: 'transformer'  # 或 'lstm'
input_dim: 14              # 根据实际关键点数量调整
num_classes: 10            # 根据行为类别数量调整
sequence_length: 64        # 序列长度
batch_size: 32
learning_rate: 0.0001
epochs: 100
```

## 训练

### 基础训练

```bash
python train.py --config configs/config.yaml
```

### 从检查点恢复训练

```bash
python train.py --config configs/config.yaml --resume checkpoints/latest_checkpoint.pth
```

### 指定设备

```bash
python train.py --config configs/config.yaml --device cuda
# 或
python train.py --config configs/config.yaml --device cpu
```

## 模型架构

### 1. Transformer 模型

- 输入投影层
- 位置编码
- Multi-head self-attention
- 前馈网络
- 分类头

### 2. LSTM 模型

- 双向 LSTM 层
- Dropout 正则化
- 全连接分类头

## 训练输出

训练过程中会保存：

- `checkpoints/latest_checkpoint.pth`: 最新检查点
- `checkpoints/best_model.pth`: 最佳模型 (基于验证损失)
- `checkpoints/history.json`: 训练历史记录

## 自定义

### 添加新模型

在 `src/models/transformer_model.py` 中：

```python
class CustomModel(nn.Module):
    def __init__(self, ...):
        # 模型定义
        pass

# 在 build_model() 函数中添加
elif model_type == 'custom':
    model = CustomModel(...)
```

### 自定义数据加载

修改 `src/data/dataset.py` 中的 `MABeMouseDataset` 类以适应你的数据格式。

### 自定义损失函数

在 `src/utils/trainer.py` 的 `_build_criterion()` 方法中添加新的损失函数。

## 评估

训练完成后，最佳模型会保存在 `checkpoints/best_model.pth`。

## 注意事项

1. 根据实际数据格式调整 `input_dim` 和 `num_classes`
2. MABe 竞赛使用关键点数据，确保数据预处理正确
3. 对于序列标注任务，标签应与序列长度匹配
4. 根据 GPU 内存调整 batch_size 和 sequence_length

## License

MIT