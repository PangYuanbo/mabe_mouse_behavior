# V6 本地训练指南 (RTX 5090)

## 📋 概述

V6版本在RTX 5090上本地训练，包含完整的Motion Features（速度+加速度）。

### V6特性
- ✅ **288维输入**: 144坐标 + 72速度 + 72加速度
- ✅ **Conv1DBiLSTM**: 最优架构（研究验证）
- ✅ **真实Kaggle数据**: 8789个训练视频
- ✅ **优化配置**: 针对RTX 5090 (32GB VRAM)

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <your-repo>
cd mabe_mouse_behavior

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 下载Kaggle数据

```bash
# 设置Kaggle API
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# 或者使用kaggle.json
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 下载数据
kaggle competitions download -c MABe-mouse-behavior-detection
unzip MABe-mouse-behavior-detection.zip -d data/kaggle/
```

### 3. 开始训练

```bash
# 使用默认配置训练
python train_v6_local_5090.py

# 自定义参数
python train_v6_local_5090.py \
    --config configs/config_5090.yaml \
    --data-dir /path/to/kaggle/data \
    --checkpoint-dir checkpoints/my_run

# 从checkpoint恢复
python train_v6_local_5090.py --resume checkpoints/v6_5090/latest_checkpoint.pth
```

---

## ⚙️ 配置说明

**文件**: `configs/config_5090.yaml`

### 关键配置

```yaml
# RTX 5090优化
batch_size: 96          # 利用32GB VRAM
num_workers: 4          # CPU核心数
learning_rate: 0.0003

# Motion Features
use_motion_features: true
motion_fps: 33.3

# 模型架构
model_type: 'conv_bilstm'
input_dim: 288  # 自动检测
conv_channels: [64, 128, 256]
lstm_hidden: 256
lstm_layers: 2

# 训练
epochs: 100
class_weights: [1.0, 5.0, 8.0, 8.0]
```

### 性能预估

| GPU | Batch | 时间/Epoch | 100 Epochs |
|-----|-------|-----------|------------|
| RTX 5090 | 96 | ~2-3分钟 | **3-5小时** |
| H100 | 384 | ~3-4分钟 | 5-7小时 |
| A10G | 64 | ~7分钟 | 12小时 |

---

## 📊 预期性能

### V6性能（Motion Features）

**基于H100早期结果**:
- Epoch 0: F1 = 0.50
- Epoch 1: F1 = 0.58
- 预期最终: F1 = **0.60-0.65**

**对比V5**（无Motion Features）:
- V5最佳: F1 = 0.4332
- V6提升: **+30-50%**

---

## 🔧 优化技巧

### 1. 调整Batch Size

根据显存使用情况调整：

```yaml
# 如果显存充足（< 80%使用）
batch_size: 128  # 或更大

# 如果OOM
batch_size: 64  # 减小
```

### 2. 增加Workers

```yaml
# 根据CPU核心数
num_workers: 8  # 如果有8核+
```

### 3. 混合精度训练

在trainer中启用AMP（自动混合精度）：

```python
# src/utils/advanced_trainer.py
use_amp = True  # 可加速1.5-2x
```

---

## 📁 文件结构

```
mabe_mouse_behavior/
├── train_v6_local_5090.py        # 5090训练脚本
├── configs/
│   └── config_5090.yaml          # 5090配置
├── src/
│   ├── data/
│   │   └── kaggle_dataset.py     # 包含motion features
│   ├── models/
│   │   └── advanced_models.py    # Conv1DBiLSTM
│   └── utils/
│       └── advanced_trainer.py   # 训练器
├── data/
│   └── kaggle/                   # Kaggle数据
└── checkpoints/
    └── v6_5090/                  # 保存位置
```

---

## 🎯 训练监控

### 查看进度

```bash
# 训练会自动显示进度
# Epoch 1/100: 100%|████████| Loss: 0.234, F1: 0.567

# 查看checkpoint
ls -lh checkpoints/v6_5090/
```

### TensorBoard（可选）

```python
# 在trainer中启用
tensorboard_dir = 'runs/v6_5090'
```

```bash
tensorboard --logdir runs/v6_5090
```

---

## 💾 Checkpoint管理

### 自动保存

- `latest_checkpoint.pth`: 最新epoch
- `best_model.pth`: 最佳F1
- `checkpoint_epoch_N.pth`: 每5 epochs

### 恢复训练

```bash
python train_v6_local_5090.py \
    --resume checkpoints/v6_5090/checkpoint_epoch_50.pth
```

---

## 🐛 故障排查

### OOM (Out of Memory)

```bash
# 减小batch size
--config configs/config_5090_small.yaml

# 或修改config.yaml
batch_size: 64  # 从96减到64
```

### 数据加载慢

```yaml
# 增加workers
num_workers: 8  # 默认4

# 启用pin_memory（在dataset中）
pin_memory: true
```

### GPU未充分利用

- 增大batch size
- 增加num_workers
- 检查数据是否在SSD上

---

## 📈 性能对比

| 版本 | 输入维度 | F1 Score | 训练时间 |
|------|----------|----------|----------|
| V5 | 144 | 0.43 | 12h (A10G) |
| **V6** | **288** | **~0.60+** | **3-5h (5090)** |

**提升**:
- F1: +40%
- 速度: 2.4-4x（取决于GPU）

---

## 🔗 相关资源

- [VERSION_HISTORY.md](VERSION_HISTORY.md) - 完整版本历史
- [versions/v6_h100_current/README.md](versions/v6_h100_current/README.md) - V6详细文档
- [KAGGLE_SUBMISSION_GUIDE.md](KAGGLE_SUBMISSION_GUIDE.md) - 提交指南

---

## 📝 使用示例

### 基础训练

```bash
# 最简单的用法
python train_v6_local_5090.py
```

### 完整参数

```bash
python train_v6_local_5090.py \
    --config configs/config_5090.yaml \
    --data-dir data/kaggle \
    --checkpoint-dir checkpoints/my_experiment \
    --resume checkpoints/my_experiment/checkpoint_epoch_30.pth
```

### 测试GPU

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

---

**准备好了就开始训练吧！** 🚀
