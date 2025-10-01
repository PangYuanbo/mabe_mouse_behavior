# V1 - 基础训练版本

## 📋 版本概述

**版本号**: V1
**开发时间**: 项目初期
**状态**: 已废弃
**目标**: 建立基础训练框架，验证流程

---

## 🎯 设计思想

### 核心目标
1. 建立端到端的训练pipeline
2. 验证数据加载和模型训练流程
3. 确保基础功能正常运行

### 技术选择
- **平台**: 本地CPU
- **数据**: 合成数据
- **模型**: 基础神经网络
- **框架**: PyTorch

---

## 🏗️ 模型结构

### 基础神经网络

```python
class BasicModel(nn.Module):
    """
    简单的全连接神经网络
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 模型参数
- Input Dim: 可配置
- Hidden Dim: 128
- Num Classes: 4 (背景、社交、交配、攻击)
- Dropout: 0.5

---

## ⚙️ 配置文件

**文件**: `configs/config.yaml`

```yaml
# V1 基础配置
model_type: 'basic'
input_dim: 144
hidden_dim: 128
num_classes: 4

batch_size: 32
learning_rate: 0.001
epochs: 50

optimizer: 'adam'
loss: 'cross_entropy'
```

---

## 🚀 使用方法

### 运行训练

```bash
# 进入V1目录
cd versions/v1_basic/

# 运行训练脚本
python train.py
```

### 参数说明
- 无命令行参数
- 所有配置在config.yaml中

---

## 📊 性能指标

### 训练结果
- **Status**: 原型验证
- **Dataset**: 合成数据
- **Performance**: 未在真实数据上测试

### 验证内容
✅ 数据加载正常
✅ 模型训练运行
✅ Loss正常下降
✅ Checkpoint保存正常

---

## 🔍 局限性

### 主要问题
1. **模型过于简单** - 只有2层全连接
2. **无时序建模** - 未考虑序列信息
3. **合成数据** - 未使用真实Kaggle数据
4. **无特征工程** - 只使用原始坐标
5. **无数据增强** - 训练数据有限

### 缺失功能
- ❌ 高级模型架构
- ❌ 特征工程
- ❌ 数据增强
- ❌ 学习率调度
- ❌ 早停机制
- ❌ Checkpoint管理

---

## 💡 经验教训

### 成功点
1. ✅ 建立了基础pipeline
2. ✅ 验证了PyTorch环境
3. ✅ 确认了数据格式

### 需改进
1. ⚠️ 模型太简单，无法处理复杂序列
2. ⚠️ 需要引入时序模型（LSTM/TCN）
3. ⚠️ 需要使用真实数据
4. ⚠️ 需要添加正则化和优化策略

---

## 📁 文件结构

```
v1_basic/
├── README.md           # 本文档
├── train.py            # 训练脚本
├── configs/
│   └── config.yaml     # 配置文件
└── docs/
    └── (empty)
```

---

## 🔄 升级到V2

### 主要改进方向
1. **模型架构** → Conv1D + BiLSTM
2. **特征工程** → 添加速度、角度等特征
3. **数据增强** → Mixup, Label Smoothing
4. **优化策略** → 学习率调度、早停

### 迁移指南
```bash
# 查看V2版本
cd ../v2_advanced/
cat README.md
```

---

## 📚 参考资料

### 代码位置
- 训练脚本: `train.py`
- 配置文件: `configs/config.yaml`

### 相关文档
- [VERSION_HISTORY.md](../../VERSION_HISTORY.md) - 完整版本历史
- [V2_README.md](../v2_advanced/README.md) - 下一版本

---

**V1 - 基础框架建立，为后续迭代打下基础** 🎯
