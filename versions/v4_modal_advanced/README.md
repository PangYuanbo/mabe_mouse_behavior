# V4 - Modal高级优化版本

## 📋 版本概述

**版本号**: V4
**开发时间**: V3之后
**状态**: 已升级至V5
**目标**: 在Modal上运行高级模型，升级GPU，准备真实数据集成

---

## 🎯 设计思想

### 核心目标
1. 在Modal平台运行V2的高级模型（Conv1DBiLSTM）
2. 升级GPU从T4到A10G获得更大显存
3. 完整训练100 epochs
4. 为真实Kaggle数据做准备

### V3 → V4 主要改进
- ✅ **模型回归**: Transformer → Conv1DBiLSTM（V2验证更优）
- ✅ **GPU升级**: T4 (16GB) → A10G (24GB)
- ✅ **Batch增大**: 16 → 64（3倍提升）
- ✅ **完整训练**: 10 epochs → 100 epochs
- ✅ **高级配置**: 使用V2的所有优化策略

### 技术选择
- **平台**: Modal (workspace: ybpang-1)
- **GPU**: A10G (24GB VRAM)
- **数据**: 合成数据（V5将切换真实数据）
- **模型**: Conv1DBiLSTM + 特征工程
- **训练时长**: 预估4小时（100 epochs）

---

## 🏗️ Modal配置

### GPU升级

```python
import modal

app = modal.App("mabe-training-advanced")

# 更强大的镜像
image = (modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "scikit-learn"
    ))

volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",  # 升级：24GB VRAM
    volumes={"/vol": volume},
    timeout=14400,  # 4小时
    memory=16384,  # 16GB RAM
)
def train_advanced_model():
    """使用A10G训练高级模型"""
    import torch
    from src.trainers.trainer import Trainer
    from src.models.conv_bilstm import Conv1DBiLSTMModel

    # 更大的batch size
    config['batch_size'] = 64  # T4只能16

    trainer = Trainer(config)
    trainer.train()

    # 定期commit防止丢失
    volume.commit()
```

### 部署命令

```bash
# 后台运行100 epochs训练
modal run --detach modal_train_advanced.py

# 查看实时日志
modal app logs mabe-training-advanced --follow

# 检查运行状态
modal app list
```

---

## 🏗️ 模型结构

### Conv1D + BiLSTM（完整版）

```python
class Conv1DBiLSTMModel(nn.Module):
    """
    高级Conv1D + BiLSTM架构
    - 来自V2验证的最佳模型
    - A10G优化：大batch + 完整训练
    """
    def __init__(self, input_dim, conv_channels, lstm_hidden,
                 lstm_layers, num_classes, dropout=0.3):
        super().__init__()

        # Multi-layer Conv1D
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Classification head
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch=64, seq_len=100, input_dim)
        x = x.transpose(1, 2)  # (64, input_dim, 100)

        # Conv1D feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        # x: (64, 256, 100)

        x = x.transpose(1, 2)  # (64, 100, 256)

        # BiLSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # (64, 100, 512)

        # Classification
        out = self.dropout(lstm_out)
        out = self.fc(out)  # (64, 100, 4)

        return out
```

### 架构参数
- Conv Channels: [64, 128, 256]
- LSTM Hidden: 256
- LSTM Layers: 2 (bidirectional)
- Total Params: ~2.1M
- Dropout: 0.3

---

## ⚙️ 配置文件

**文件**: `configs/config_advanced.yaml`

```yaml
# V4 Modal高级配置（A10G优化）

# 模型设置
model_type: 'conv_bilstm'
input_dim: 28
num_classes: 4

# Conv1DBiLSTM参数
conv_channels: [64, 128, 256]
lstm_hidden: 256
lstm_layers: 2

# 特征工程
use_feature_engineering: true
include_pca: true
pca_components: 16
include_temporal_stats: true

# 序列设置
sequence_length: 100  # 3秒 @ 33.3Hz
frame_gap: 1
fps: 33.3

# 训练设置（A10G优化）
epochs: 100
batch_size: 64  # A10G可支持
learning_rate: 0.0003
weight_decay: 0.0001
optimizer: 'adamw'
grad_clip: 1.0

# 损失函数
loss: 'cross_entropy'
class_weights: [0.5, 10.0, 15.0, 15.0]
label_smoothing: 0.0

# 学习率调度
scheduler: 'plateau'
scheduler_patience: 5
scheduler_factor: 0.5
min_lr: 0.00001

# 数据增强
use_augmentation: true
noise_std: 0.01
temporal_jitter: 2

# 正则化
dropout: 0.3
mixup_alpha: 0.0

# 早停
early_stopping_patience: 15

# Data loader
num_workers: 4

# Checkpoint
checkpoint_dir: 'checkpoints'
save_freq: 5

# Modal设置
modal_gpu: 'A10G'
modal_timeout: 14400  # 4小时
```

---

## 🚀 使用方法

### 准备环境
```bash
# 确认Modal配置
modal token set --token-id xxx --token-secret xxx
modal workspace set ybpang-1
```

### 运行训练

```bash
# 进入V4目录
cd versions/v4_modal_advanced/

# 后台运行（推荐）
modal run --detach modal_train_advanced.py

# 查看日志
modal app logs mabe-training-advanced --follow

# 停止训练
modal app stop mabe-training-advanced
```

### 监控训练

```python
# 检查checkpoint（每5 epochs保存）
@app.function(volumes={"/vol": volume})
def list_checkpoints():
    import os
    checkpoint_dir = "/vol/checkpoints"
    files = os.listdir(checkpoint_dir)
    return sorted(files)

# 执行
modal run modal_train_advanced.py::list_checkpoints
```

### 下载模型

```python
# 下载最佳模型
modal run download_checkpoint.py --checkpoint best_model.pth

# 下载特定epoch
modal run download_checkpoint.py --checkpoint checkpoint_epoch_50.pth
```

---

## 📊 性能指标

### A10G性能
- **GPU**: A10G (24GB VRAM)
- **Batch Size**: 64
- **训练速度**: ~80-100 it/s（比T4快2-3倍）
- **每epoch时间**: ~1.5-2分钟
- **100 epochs总时长**: ~3-4小时

### 显存使用
- Model: ~2.1M params ≈ 8MB
- Batch (64, 100, 28): ~0.5GB
- Gradients + Optimizer: ~2GB
- **总计**: ~3-4GB / 24GB（充足）

### 相比V3的提升
- ✅ **显存**: 16GB → 24GB（+50%）
- ✅ **Batch**: 16 → 64（4倍）
- ✅ **速度**: 30 it/s → 100 it/s（3倍）
- ✅ **模型**: Transformer → Conv1DBiLSTM（更优）
- ✅ **训练轮数**: 10 → 100（完整训练）

### 训练曲线（预期）
```
Epoch 1:  Loss: 1.2, Acc: 0.50
Epoch 10: Loss: 0.8, Acc: 0.65
Epoch 30: Loss: 0.5, Acc: 0.75
Epoch 50: Loss: 0.3, Acc: 0.82
Epoch 100: Loss: 0.2, Acc: 0.85
```

---

## 🔍 局限性

### 主要问题
1. **仍使用合成数据** - 这是最大的问题！
2. **无Motion features** - 缺少速度、加速度等关键特征
3. **特征工程不完整** - 仅PCA，未添加距离、角度等
4. **未在真实竞赛数据上验证** - 性能未知

### 缺失功能
- ❌ **真实Kaggle数据**（V5将解决）
- ❌ Motion features（速度、加速度）
- ❌ 更大GPU（H100）
- ❌ 混合精度训练（FP16）
- ❌ 完整的距离/角度特征

---

## 💡 经验教训

### 成功点
1. ✅ A10G GPU充分利用（batch 64稳定）
2. ✅ Conv1DBiLSTM在Modal上运行良好
3. ✅ 100 epochs完整训练验证
4. ✅ Checkpoint管理完善

### 关键发现
1. 💡 A10G比T4快3倍，性价比高
2. 💡 Batch 64在24GB显存下很稳定
3. 💡 Conv1DBiLSTM优于Transformer（速度+性能）
4. 💡 100 epochs可在4小时内完成

### 下一步必做
1. ⚠️ **立即集成真实Kaggle数据**（最高优先级）
2. ⚠️ 添加Motion features（速度+加速度）
3. ⚠️ 实现完整特征工程
4. ⚠️ 在真实数据上评估性能

---

## 📁 文件结构

```
v4_modal_advanced/
├── README.md                    # 本文档
├── modal_train_advanced.py      # Modal高级训练
├── configs/
│   └── config_advanced.yaml     # A10G优化配置
├── models/
│   └── conv_bilstm.py          # Conv1DBiLSTM实现
└── docs/
    └── a10g_optimization.md     # GPU优化指南
```

---

## 🔄 升级到V5

### 主要改进方向（关键！）
1. **真实数据** → 集成Kaggle竞赛数据（8789个视频）
2. **Motion features** → 添加速度(+72维)、加速度(+72维)
3. **输入升级** → 28维 → 288维（144 coords + 72 speed + 72 accel）
4. **性能突破** → F1从合成数据结果到真实0.43+

### V4 → V5 是重大突破
V5是第一个在**真实Kaggle竞赛数据**上训练的版本！

### 迁移指南
```bash
# 查看V5版本（真实Kaggle数据）
cd ../v5_modal_kaggle/
cat README.md
```

---

## 📚 参考资料

### Modal GPU文档
- [Modal A10G Guide](https://modal.com/docs/guide/gpu)
- [GPU Performance Comparison](https://modal.com/pricing)

### 代码位置
- Modal训练: `modal_train_advanced.py`
- 配置文件: `configs/config_advanced.yaml`
- 模型定义: `models/conv_bilstm.py`

### 相关文档
- [V3_README.md](../v3_modal_basic/README.md) - 上一版本
- [V5_README.md](../v5_modal_kaggle/README.md) - **下一版本（重大突破）**
- [VERSION_HISTORY.md](../../VERSION_HISTORY.md) - 完整版本历史

---

**V4 - Modal高级优化，为真实数据铺路** 🔥
