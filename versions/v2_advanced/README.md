# V2 - 高级模型版本

## 📋 版本概述

**版本号**: V2
**开发时间**: V1之后
**状态**: 已升级至V3
**目标**: 引入SOTA模型架构，添加特征工程

---

## 🎯 设计思想

### 核心目标
1. 引入研究论文中的先进架构（Conv1DBiLSTM）
2. 实现特征工程提升模型性能
3. 添加正则化和优化策略

### V1 → V2 主要改进
- ✅ **模型升级**: 从简单全连接 → Conv1D + BiLSTM
- ✅ **特征工程**: 添加PCA、时序统计特征
- ✅ **数据增强**: Mixup、噪声注入、时序抖动
- ✅ **训练优化**: 学习率调度、早停、梯度裁剪
- ✅ **多模型支持**: Conv1DBiLSTM、TCN、Hybrid、Transformer

### 技术选择
- **平台**: 本地GPU（如果可用）
- **数据**: 仍使用合成数据
- **模型**: Conv1DBiLSTM（研究表明96%准确率）
- **框架**: PyTorch + 高级训练技巧

---

## 🏗️ 模型结构

### 1. Conv1D + BiLSTM (主推)

```python
class Conv1DBiLSTMModel(nn.Module):
    """
    Conv1D + BiLSTM 架构
    - 来源：MABe竞赛winning solution
    - 性能：研究论文中达到96%准确率
    """
    def __init__(self, input_dim, conv_channels, lstm_hidden, lstm_layers, num_classes, dropout=0.3):
        super().__init__()

        # Conv1D layers for local feature extraction
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

        # BiLSTM for temporal modeling
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
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        # Conv1D layers
        for conv in self.conv_layers:
            x = conv(x)

        x = x.transpose(1, 2)  # (batch, seq_len, conv_channels[-1])

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden*2)

        # Classification
        out = self.dropout(lstm_out)
        out = self.fc(out)  # (batch, seq_len, num_classes)

        return out
```

### 2. TCN (Temporal Convolutional Network)

```python
class TCNModel(nn.Module):
    """
    Temporal Convolutional Network
    - 优势：并行计算快，感受野大
    - 适用：长序列建模
    """
    def __init__(self, input_dim, tcn_channels, kernel_size, num_classes, dropout=0.3):
        super().__init__()

        self.tcn_layers = nn.ModuleList()
        in_channels = input_dim

        for i, out_channels in enumerate(tcn_channels):
            dilation = 2 ** i
            self.tcn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                         padding=(kernel_size-1)*dilation, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        self.fc = nn.Linear(tcn_channels[-1], num_classes)
```

### 3. Hybrid (PointNet + LSTM)

```python
class HybridModel(nn.Module):
    """
    Hybrid架构：PointNet提取空间特征 + LSTM建模时序
    - PointNet: 处理无序点云（鼠标关键点）
    - LSTM: 捕捉时序依赖
    """
    def __init__(self, input_dim, pointnet_dim, temporal_hidden, num_classes, dropout=0.3):
        super().__init__()

        # PointNet for spatial features
        self.pointnet = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, pointnet_dim)
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(pointnet_dim, temporal_hidden,
                           num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(temporal_hidden * 2, num_classes)
```

### 模型参数对比

| 模型 | 参数量 | 优势 | 劣势 |
|------|--------|------|------|
| Conv1DBiLSTM | ~2M | 准确率高、时序建模强 | 训练较慢 |
| TCN | ~1.5M | 速度快、并行化好 | 长依赖较弱 |
| Hybrid | ~1.8M | 空间建模强 | 复杂度高 |
| Transformer | ~3M | 注意力机制 | 数据需求大 |

---

## ⚙️ 配置文件

**文件**: `configs/config_advanced.yaml`

```yaml
# V2 高级配置
model_type: 'conv_bilstm'  # 主推模型

# Conv1DBiLSTM参数
conv_channels: [64, 128, 256]  # 3层Conv1D
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

# 训练设置
epochs: 100
batch_size: 64
learning_rate: 0.0003
weight_decay: 0.0001
optimizer: 'adamw'
grad_clip: 1.0

# 损失函数
loss: 'cross_entropy'
class_weights: [0.5, 10.0, 15.0, 15.0]
label_smoothing: 0.0  # 针对不平衡数据禁用

# 学习率调度
scheduler: 'plateau'
scheduler_patience: 5
scheduler_factor: 0.5
min_lr: 0.00001

# 数据增强
use_augmentation: true
noise_std: 0.01  # 高斯噪声
temporal_jitter: 2  # 时序抖动

# 正则化
dropout: 0.3
mixup_alpha: 0.0  # 不平衡数据禁用Mixup

# 早停
early_stopping_patience: 15
```

---

## 🔧 特征工程

### 1. 基础特征 (28维)
- 原始坐标：2只老鼠 × 7个关键点 × 2坐标 = 28维

### 2. PCA特征 (16维)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=16)
pca_features = pca.fit_transform(keypoints)
# 降维：保留主要变化方向
```

### 3. 时序统计特征
- 均值、标准差
- 最大值、最小值
- 变化率

### 总输入维度
28 (原始) + 16 (PCA) + 其他特征 = **动态调整**

---

## 🚀 使用方法

### 安装依赖
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install pyyaml tqdm
```

### 运行训练

```bash
# 进入V2目录
cd versions/v2_advanced/

# 使用Conv1DBiLSTM训练（默认）
python train_advanced.py

# 使用TCN模型
python train_advanced.py --model tcn

# 使用Hybrid模型
python train_advanced.py --model hybrid

# 自定义配置
python train_advanced.py --config configs/config_advanced.yaml
```

### 模型评估
```bash
# 加载最佳checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## 📊 性能指标

### 训练结果
- **Status**: 本地验证
- **Dataset**: 合成数据（仍未使用真实Kaggle数据）
- **Performance**: Conv1DBiLSTM在合成数据上表现良好

### 相比V1的提升
- ✅ **模型容量**: 2层 → 深度卷积+BiLSTM
- ✅ **特征表达**: 原始坐标 → PCA + 时序特征
- ✅ **训练策略**: 固定LR → 自适应调度 + 早停
- ✅ **正则化**: 单一Dropout → 多种正则化技术

### 验证功能
✅ Conv1D层正确提取局部特征
✅ BiLSTM捕捉双向时序依赖
✅ 学习率调度自动降低
✅ 早停防止过拟合
✅ Checkpoint管理完善

---

## 🔍 局限性

### 主要问题
1. **仍使用合成数据** - 未在真实Kaggle数据上测试
2. **特征工程不足** - 缺少速度、加速度等关键特征
3. **本地训练** - 缺乏GPU资源，训练慢
4. **无云端部署** - 未利用Modal等云平台

### 缺失功能
- ❌ 真实Kaggle竞赛数据
- ❌ Motion features（速度、加速度）
- ❌ 云端GPU训练
- ❌ 大batch训练
- ❌ 分布式训练

---

## 💡 经验教训

### 成功点
1. ✅ Conv1DBiLSTM架构验证成功
2. ✅ 特征工程框架搭建完成
3. ✅ 训练pipeline完善（LR调度、早停等）
4. ✅ 支持多种模型架构切换

### 需改进
1. ⚠️ 必须切换到真实Kaggle数据
2. ⚠️ 需要云端GPU资源（Modal）
3. ⚠️ 需要添加Motion features
4. ⚠️ 需要更大的batch size和更长的训练

---

## 📁 文件结构

```
v2_advanced/
├── README.md                   # 本文档
├── train_advanced.py           # 高级训练脚本
├── configs/
│   └── config_advanced.yaml    # 高级配置
├── models/
│   ├── conv_bilstm.py         # Conv1DBiLSTM实现
│   ├── tcn.py                 # TCN实现
│   └── hybrid.py              # Hybrid实现
└── docs/
    └── (empty)
```

---

## 🔄 升级到V3

### 主要改进方向
1. **云端部署** → 使用Modal平台
2. **GPU加速** → A10G / T4
3. **真实数据** → 准备Kaggle数据集成
4. **优化训练** → 更大batch、更长训练

### 迁移指南
```bash
# 查看V3版本（Modal基础版）
cd ../v3_modal_basic/
cat README.md
```

---

## 📚 参考资料

### 研究论文
- Conv1D + BiLSTM: 96%准确率
- TCN for sequence modeling
- PointNet for unordered point clouds

### 代码位置
- 训练脚本: `train_advanced.py`
- 模型定义: `models/`
- 配置文件: `configs/config_advanced.yaml`

### 相关文档
- [V1_README.md](../v1_basic/README.md) - 上一版本
- [V3_README.md](../v3_modal_basic/README.md) - 下一版本
- [VERSION_HISTORY.md](../../VERSION_HISTORY.md) - 完整版本历史

---

**V2 - 引入SOTA架构，奠定模型基础** 🚀
